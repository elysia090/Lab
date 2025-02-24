import ast
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from collections import defaultdict
from typing import List, Tuple, Dict
import time

# 定数
BITS_PER_MASK = 64  # ビットマスクごとのビット数
MAX_MASK_COUNT = 10  # 最大マスク数（最大640ノードをサポート）

# CFG（制御フローグラフ）を構築するクラス
class CFGBuilder(ast.NodeVisitor):
    def __init__(self):
        self.cfg: Dict[int, List[int]] = defaultdict(list)  # ノードから後続ノードへのエッジ
        self.predecessors: Dict[int, List[int]] = defaultdict(list)  # ノードへの先行ノード
        self.current_node: int = 0  # 現在のノード番号
        self.node_count: int = 0  # 総ノード数
        self.labels: Dict[int, ast.AST] = {} # ノードのラベル

    def add_edge(self, from_node: int, to_node: int) -> None:
        """ノード間にエッジを追加する"""
        self.cfg[from_node].append(to_node)
        self.predecessors[to_node].append(from_node)

    def new_node(self) -> int:
        """新しいノードを作成"""
        self.node_count += 1
        return self.node_count - 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """関数定義ノードを処理"""
        self.current_node = 0
        self.node_count = 1
        self.labels[0] = node
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        """If文を処理し、分岐をCFGに反映"""
        current_node = self.current_node
        then_node = self.new_node()
        self.labels[then_node] = node
        self.add_edge(current_node, then_node)
        self.current_node = then_node
        for stmt in node.body:
            self.visit(stmt)
        
        if_end_node = self.current_node # if文の最後のノードを保持

        if node.orelse:
            else_node = self.new_node()
            self.labels[else_node] = node.orelse
            self.add_edge(current_node, else_node)
            self.current_node = else_node
            for stmt in node.orelse:
                self.visit(stmt)
            
            next_node = self.new_node()
            self.labels[next_node] = f"next_{node.lineno}"  # わかりやすいラベル
            self.add_edge(if_end_node, next_node)
            self.add_edge(self.current_node, next_node)
            self.current_node = next_node
        else:
            next_node = self.new_node()
            self.labels[next_node] = f"next_{node.lineno}"  # わかりやすいラベル
            self.add_edge(current_node, next_node)
            self.add_edge(if_end_node, next_node)
            self.current_node = next_node

    def visit_Return(self, node: ast.Return) -> None:
        """Return文を処理"""
        return_node = self.new_node()
        self.labels[return_node] = node
        self.add_edge(self.current_node, return_node)
        self.current_node = return_node
        

    def generic_visit(self, node: ast.AST) -> None:
        """その他のASTノードを汎用的に処理"""
        if isinstance(node, ast.FunctionDef):
           for stmt in node.body:
                self.visit(stmt)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, ast.AST):
                    self.visit(item)
        else:
            current_node = self.new_node()
            self.labels[current_node] = node
            self.add_edge(self.current_node, current_node)
            self.current_node = current_node
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.AST):
                            self.visit(item)
                elif isinstance(value, ast.AST):
                    self.visit(value)

# 先行ノードリストを準備する関数
def prepare_predecessor_data(
    cfg: Dict[int, List[int]], 
    predecessors: Dict[int, List[int]], 
    num_nodes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """CFGから先行ノードリストとオフセットを生成"""
    pred_list: List[int] = []
    pred_offsets: List[int] = [0]
    offset: int = 0
    for node in range(num_nodes):
        preds = predecessors[node]
        pred_list.extend(preds)
        offset += len(preds)
        pred_offsets.append(offset)
    return np.array(pred_list, dtype=np.int32), np.array(pred_offsets, dtype=np.int32)

# ドミネーターツリーを構築する関数
def build_dominator_tree(dom: np.ndarray, num_nodes: int, mask_count: int) -> dict[int, list[int]]:
    """ドミネーター集合からドミネーターツリーを構築"""
    from collections import defaultdict
    tree: dict[int, list[int]] = defaultdict(list)
    BITS_PER_MASK = 64  # 仮に64ビットと定義（実際の値に合わせて調整）

    for node in range(1, num_nodes):
        idom = None  # 即時ドミネーター
        for i in range(num_nodes):
            mask_index = i // BITS_PER_MASK
            bit_index = i % BITS_PER_MASK
            if mask_index < mask_count:
                # ビット演算のために型を明示的にキャスト
                dom_value = np.uint64(dom[node][mask_index])
                bit_value = np.uint64(1 << bit_index)
                if (dom_value & bit_value) != 0 and i != node:
                    if idom is None or (np.uint64(dom[idom][mask_index]) & bit_value) == 0:
                        idom = i
        if idom is not None:
            tree[idom].append(node)
    return tree

# CUDAカーネルの定義
cuda_code = f"""
#define BITS_PER_MASK {BITS_PER_MASK}
#define MAX_MASK_COUNT {MAX_MASK_COUNT}

__global__ void compute_dominator(
    unsigned long long *dom, 
    int *predecessors, 
    int *pred_offsets, 
    int num_nodes, 
    int num_preds, 
    int mask_count) 
{{
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    if (mask_count > MAX_MASK_COUNT) return;

    if (node == 0) {{
        dom[node * mask_count] = 1ULL << (node % BITS_PER_MASK);
        for (int i = 1; i < mask_count; i++) {{
            dom[node * mask_count + i] = 0;
        }}
        return;
    }}

    unsigned long long intersection[MAX_MASK_COUNT];
    for (int i = 0; i < mask_count; i++) {{
        intersection[i] = ~0ULL;  
    }}
    int start = pred_offsets[node];
    int end = (node + 1 < num_nodes) ? pred_offsets[node + 1] : num_preds;
    for (int i = start; i < end; i++) {{
        int pred = predecessors[i];
        for (int j = 0; j < mask_count; j++) {{
            intersection[j] &= dom[pred * mask_count + j];
        }}
    }}
    int mask_index = node / BITS_PER_MASK;
    int bit_index = node % BITS_PER_MASK;
    if (mask_index < mask_count) {{
        intersection[mask_index] |= (1ULL << bit_index);
    }}
    for (int i = 0; i < mask_count; i++) {{
        dom[node * mask_count + i] = intersection[i];
    }}
}}
"""

# メイン処理
def compute_dominators_on_gpu(source_code: str) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """GPUを使用してドミネーターを計算し、ドミネーターツリーを返す"""
    # 1. CFGの構築
    tree = ast.parse(source_code)
    builder = CFGBuilder()
    builder.visit(tree)
    num_nodes = builder.node_count

    # 2. 先行ノードリストの準備
    predecessors, pred_offsets = prepare_predecessor_data(builder.cfg, builder.predecessors, num_nodes)
    num_preds = len(predecessors)

    # 3. ドミネーター集合の初期化
    mask_count = (num_nodes + BITS_PER_MASK - 1) // BITS_PER_MASK
    if mask_count > MAX_MASK_COUNT:
        raise ValueError(f"mask_count ({mask_count}) exceeds MAX_MASK_COUNT ({MAX_MASK_COUNT})")
    dom = np.zeros((num_nodes, mask_count), dtype=np.uint64)
    dom[0, 0] = 1  # ルートノード

    # 4. GPUメモリにデータを転送
    predecessors_gpu = cuda.to_device(predecessors)
    pred_offsets_gpu = cuda.to_device(pred_offsets)
    dom_gpu = cuda.to_device(dom)

    # 5. CUDAカーネルのコンパイルと実行
    module = SourceModule(cuda_code)
    compute_dominator = module.get_function("compute_dominator")
    block_size = 256
    grid_size = (num_nodes + block_size - 1) // block_size

    # 6. ドミネーター計算の反復更新
    for iteration in range(num_nodes):
        old_dom = dom.copy()
        compute_dominator(
            dom_gpu, predecessors_gpu, pred_offsets_gpu,
            np.int32(num_nodes), np.int32(num_preds), np.int32(mask_count),
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )
        cuda.memcpy_dtoh(dom, dom_gpu)
        if np.array_equal(old_dom, dom):
            break

    # 7. ドミネーターツリーの構築
    dom_tree = build_dominator_tree(dom, num_nodes, mask_count)
    return dom, dom_tree

# テスト用コード（計測付き）
if __name__ == "__main__":
    source_code_nested = """
def nested_example(a, b, c):
    if a > b:
        if b > c:
            x = a + b
        else:
            x = a - b
    else:
        if a > c:
            y = b + c
        else:
            if c > 10:
               y = 100
            else:
               y = -100
    return x + y
"""

    num_runs = 100
    total_time = 0

    for _ in range(num_runs):
        start_time = time.time()
        dominators_nested, dom_tree_nested = compute_dominators_on_gpu(source_code_nested)
        end_time = time.time()
        total_time += (end_time - start_time)

    average_time = (total_time / num_runs) * 1000  # ミリ秒に変換
    print(f"{num_runs}回の実行の平均時間: {average_time:.3f} ms")

    # 結果の表示（最初の実行の結果）
    print("ドミネーター結果（ビットマスク）:")
    for i, dom in enumerate(dominators_nested):
        print(f"Node {i}: {dom.tolist()}")  # numpy arrayをリストに変換
    print("ドミネーターツリー:")
    for parent, children in dom_tree_nested.items():
        print(f"Node {parent} -> {children}")
