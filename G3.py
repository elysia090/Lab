import ast
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from collections import defaultdict
from typing import List, Dict
import time
import graphviz  # CFG 視覚化用

# 定数
BITS_PER_MASK = 64
MAX_MASK_COUNT = 10

# --------------------------------------------------
# CFG構築モジュール
# --------------------------------------------------
class CFGBuilder(ast.NodeVisitor):
    def __init__(self):
        self.cfg: Dict[int, List[int]] = defaultdict(list)
        self.predecessors: Dict[int, List[int]] = defaultdict(list)
        self.current_node: int = 0
        self.node_count: int = 0
        self.labels: Dict[int, ast.AST] = {}  # ラベルを AST ノードオブジェクト自体にする

    def add_edge(self, from_node: int, to_node: int) -> None:
        self.cfg[from_node].append(to_node)
        self.predecessors[to_node].append(from_node)

    def new_node(self) -> int:
        self.node_count += 1
        return self.node_count - 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.current_node = 0
        self.node_count = 1
        self.labels[0] = node
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        current_node = self.current_node
        then_node = self.new_node()
        self.labels[then_node] = node
        self.add_edge(current_node, then_node)
        self.current_node = then_node
        for stmt in node.body:
            self.visit(stmt)
        if_end_node = self.current_node
        if node.orelse:
            else_node = self.new_node()
            self.labels[else_node] = node.orelse
            self.add_edge(current_node, else_node)
            self.current_node = else_node
            for stmt in node.orelse:
                self.visit(stmt)
            next_node = self.new_node()
            self.labels[next_node] = f"next_{node.lineno}"
            self.add_edge(if_end_node, next_node)
            self.add_edge(self.current_node, next_node)
            self.current_node = next_node
        else:
            next_node = self.new_node()
            self.labels[next_node] = f"next_{node.lineno}"
            self.add_edge(current_node, next_node)
            self.add_edge(if_end_node, next_node)
            self.current_node = next_node

    def visit_Return(self, node: ast.Return) -> None:
        return_node = self.new_node()
        self.labels[return_node] = node
        self.add_edge(self.current_node, return_node)
        self.current_node = return_node

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, ast.FunctionDef):
            for stmt in node.body:
                self.visit(stmt)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, ast.AST):
                    self.visit(item)
        else:
            current_node = self.new_node()
            self.labels[current_node] = type(node).__name__
            self.add_edge(self.current_node, current_node)
            self.current_node = current_node
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.AST):
                            self.visit(item)
                elif isinstance(value, ast.AST):
                    self.visit(value)

def prepare_predecessor_data(cfg, predecessors, num_nodes):
    pred_list: List[int] = []
    pred_offsets: List[int] = [0]
    offset = 0
    for node in range(num_nodes):
        preds = predecessors[node]
        pred_list.extend(preds)
        offset += len(preds)
        pred_offsets.append(offset)
    return np.array(pred_list, dtype=np.int32), np.array(pred_offsets, dtype=np.int32)

def build_dominator_tree(dom, num_nodes, mask_count):
    from collections import defaultdict
    tree: Dict[int, List[int]] = defaultdict(list)
    for node in range(1, num_nodes):
        idom = None
        for i in range(num_nodes):
            mask_index = i // BITS_PER_MASK
            bit_index = i % BITS_PER_MASK
            if mask_index < mask_count:
                dom_value = np.uint64(dom[node][mask_index])
                bit_value = np.uint64(1 << bit_index)
                if (dom_value & bit_value) != 0 and i != node:
                    if idom is None or (np.uint64(dom[idom][mask_index]) & bit_value) == 0:
                        idom = i
        if idom is not None:
            tree[idom].append(node)
    return tree

def visualize_cfg(cfg: Dict[int, List[int]], labels: Dict[int, any], filename: str = 'cfg'):
    """Graphviz を使って CFG を視覚化する"""
    dot = graphviz.Digraph(comment='Control Flow Graph')
    for node_id in sorted(labels.keys()):
        label = str(type(labels[node_id]).__name__) if isinstance(labels[node_id], ast.AST) else str(labels[node_id])
        dot.node(str(node_id), label)
    for from_node, to_nodes in cfg.items():
        for to_node in to_nodes:
            dot.edge(str(from_node), str(to_node))
    dot.render(filename, format='png', view=False)
    print(f"CFG を {filename}.png に保存しました。")

# --------------------------------------------------
# GPUドミネーター計算モジュール
# --------------------------------------------------
cuda_code = f"""
#define BITS_PER_MASK {BITS_PER_MASK}
#define MAX_MASK_COUNT {MAX_MASK_COUNT}

__global__ void compute_dominator(
    unsigned long long *dom,
    int *predecessors,
    int *pred_offsets,
    int num_nodes,
    int num_preds,
    int mask_count,
    int *d_changed)
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

    bool changed = false;
    for (int i = 0; i < mask_count; i++) {{
        if (dom[node * mask_count + i] != intersection[i]) {{
            changed = true;
            break;
        }}
    }}
    if (changed) {{
        for (int i = 0; i < mask_count; i++) {{
            dom[node * mask_count + i] = intersection[i];
        }}
        atomicExch(&d_changed[0], 1);
    }}
}}
"""

def run_gpu_dominator_kernel(dom_gpu, predecessors_gpu, pred_offsets_gpu,
                               num_nodes, num_preds, mask_count,
                               d_changed, block_size, grid_size, stream, compute_dominator):
    """GPU カーネルを1回実行し、非同期ストリームで同期する"""
    compute_dominator(
        dom_gpu, predecessors_gpu, pred_offsets_gpu,
        np.int32(num_nodes), np.int32(num_preds), np.int32(mask_count),
        d_changed,
        block=(block_size, 1, 1), grid=(grid_size, 1), stream=stream
    )
    stream.synchronize()

def compute_dominators_on_gpu(source_code: str):
    """GPU を用いてドミネーター計算を行い、ドミネーターツリーを返す"""
    # 1. CFG の構築
    tree = ast.parse(source_code)
    builder = CFGBuilder()
    builder.visit(tree)
    num_nodes = builder.node_count

    # 2. 先行ノードデータの準備
    predecessors, pred_offsets = prepare_predecessor_data(builder.cfg, builder.predecessors, num_nodes)
    num_preds = len(predecessors)

    # 3. ドミネーター集合の初期化
    mask_count = (num_nodes + BITS_PER_MASK - 1) // BITS_PER_MASK
    if mask_count > MAX_MASK_COUNT:
        raise ValueError(f"mask_count ({mask_count}) exceeds MAX_MASK_COUNT ({MAX_MASK_COUNT})")
    dom = np.zeros((num_nodes, mask_count), dtype=np.uint64)
    dom[0, 0] = 1

    # 4. GPU メモリ転送とストリームの設定
    start_time = time.time()
    predecessors_gpu = cuda.to_device(predecessors)
    pred_offsets_gpu = cuda.to_device(pred_offsets)
    dom_gpu = cuda.to_device(dom)
    # 変更フラグ（ホスト側は1要素の配列）
    d_changed = cuda.mem_alloc(np.zeros(1, dtype=np.int32).nbytes)
    module = SourceModule(cuda_code)
    compute_dominator = module.get_function("compute_dominator")
    block_size = 256
    grid_size = (num_nodes + block_size - 1) // block_size
    stream = cuda.Stream()

    # 5. 収束判定ループ：各反復で d_changed のみをチェックし、
    #    反復内での全体 dom 配列のホストコピーを削減
    while True:
        changed_host = np.zeros(1, dtype=np.int32)  # 書き込み可能なバッファ
        cuda.memcpy_htod_async(d_changed, changed_host, stream)
        run_gpu_dominator_kernel(dom_gpu, predecessors_gpu, pred_offsets_gpu,
                                 num_nodes, num_preds, mask_count,
                                 d_changed, block_size, grid_size, stream, compute_dominator)
        cuda.memcpy_dtoh(changed_host, d_changed)
        if changed_host[0] == 0:
            break
    # 収束後、最終的な dom 配列をホストにコピー
    cuda.memcpy_dtoh(dom, dom_gpu)
    end_time = time.time()

    # 6. ドミネーターツリー構築
    dom_tree = build_dominator_tree(dom, num_nodes, mask_count)
    return dom, dom_tree, (end_time - start_time) * 1000  # ms

# --------------------------------------------------
# テスト & 計測部
# --------------------------------------------------
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
        _, _, elapsed_time = compute_dominators_on_gpu(source_code_nested)
        total_time += elapsed_time
    average_time = total_time / num_runs
    print(f"{num_runs}回の実行の平均時間: {average_time:.3f} ms")

    dominators_nested, dom_tree_nested, _ = compute_dominators_on_gpu(source_code_nested)
    print("ドミネーター結果（ビットマスク）:")
    for i, dom_val in enumerate(dominators_nested):
        print(f"Node {i}: {dom_val.tolist()}")
    print("ドミネーターツリー:")
    for parent, children in dom_tree_nested.items():
        print(f"Node {parent} -> {children}")

    # CFG を視覚化
    tree = ast.parse(source_code_nested)
    builder = CFGBuilder()
    builder.visit(tree)
    visualize_cfg(builder.cfg, builder.labels, filename='nested_example_cfg')
