#!/usr/bin/env python3
"""
  - AST を用いた CFG 構築
  - 前処理：各ノードの前駆リストを一度に構築してから NumPy 配列に変換
  - GPU 上での dominator および immediate dominator 計算
    → ダブルバッファリングによる共有メモリ利用で同期呼び出しを削減
  - ベンチマーク機能の追加
"""

import ast
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set

import numpy as np
import cupy as cp
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# ========================================================
# 1. CFG 構築（AST → CFG）
# ========================================================
@dataclass
class Block:
    id: int
    label: str = ""
    successors: List[int] = field(default_factory=list)

class CFG:
    def __init__(self) -> None:
        self.blocks: Dict[int, Block] = {}
    
    def add_block(self, block: Block) -> None:
        self.blocks[block.id] = block
    
    def get_graph(self) -> Dict[int, List[int]]:
        return {block.id: block.successors for block in self.blocks.values()}
    
    def get_labels(self) -> Dict[int, str]:
        return {block.id: block.label for block in self.blocks.values()}

class CFGBuilder:
    def __init__(self) -> None:
        self.cfg = CFG()
        self.next_block_id = 0
    
    def new_block(self, label: str = "") -> Block:
        block = Block(self.next_block_id, label)
        self.cfg.add_block(block)
        self.next_block_id += 1
        return block
    
    def build_sequence(self, stmts: List[ast.stmt]) -> Tuple[Block, Block]:
        if not stmts:
            empty_block = self.new_block("")
            return empty_block, empty_block
        
        first_block = None
        prev_block = None
        current_exit = None

        for stmt in stmts:
            if isinstance(stmt, ast.If):
                cond_label = self._expr_to_str(stmt.test)
                cond_block = self.new_block(cond_label)
                if prev_block:
                    prev_block.successors.append(cond_block.id)
                
                then_entry, then_exit = self.build_sequence(stmt.body)
                if stmt.orelse:
                    else_entry, else_exit = self.build_sequence(stmt.orelse)
                else:
                    else_entry = self.new_block("")
                    else_exit = self.new_block("")
                
                cond_block.successors.extend([then_entry.id, else_entry.id])
                join_block = self.new_block("")
                then_exit.successors.append(join_block.id)
                else_exit.successors.append(join_block.id)
                current_exit = self.new_block("")
                join_block.successors.append(current_exit.id)
                prev_block = current_exit
                if first_block is None:
                    first_block = cond_block

            elif isinstance(stmt, (ast.While, ast.For)):
                label = self._expr_to_str(stmt.test) if isinstance(stmt, ast.While) else self._expr_to_str(stmt.iter)
                cond_block = self.new_block(label)
                if prev_block:
                    prev_block.successors.append(cond_block.id)
                body_entry, body_exit = self.build_sequence(stmt.body)
                cond_block.successors.append(body_entry.id)
                body_exit.successors.append(cond_block.id)
                exit_block = self.new_block("")
                cond_block.successors.append(exit_block.id)
                current_exit = exit_block
                prev_block = exit_block
                if first_block is None:
                    first_block = cond_block

            else:
                simple_label = self._stmt_to_str(stmt)
                simple_block = self.new_block(simple_label)
                if prev_block:
                    prev_block.successors.append(simple_block.id)
                current_exit = simple_block
                prev_block = simple_block
                if first_block is None:
                    first_block = simple_block
        return first_block, current_exit
    
    def build(self, stmts: List[ast.stmt]) -> CFG:
        entry, exit_blk = self.build_sequence(stmts)
        final_exit = self.new_block("")
        exit_blk.successors.append(final_exit.id)
        return self.cfg
    
    def _expr_to_str(self, expr: ast.expr) -> str:
        try:
            return ast.unparse(expr).strip()
        except Exception:
            return str(expr)
    
    def _stmt_to_str(self, stmt: ast.stmt) -> str:
        try:
            return ast.unparse(stmt).strip()
        except Exception:
            return str(stmt)

def ast_to_cfg(source: str) -> Tuple[Dict[int, List[int]], Dict[int, str]]:
    tree = ast.parse(source)
    builder = CFGBuilder()
    cfg_obj = builder.build(tree.body)
    return cfg_obj.get_graph(), cfg_obj.get_labels()

# ========================================================
# 2. グラフ前処理：固定サイズ前駆データの構築（最適化版）
# ========================================================
class GraphPreprocessor:
    @staticmethod
    def build_predecessor_fixed(graph: Dict[int, List[int]], num_nodes: int) -> Tuple[cp.ndarray, cp.ndarray, int]:
        # 各ノードの前駆リストを辞書で一括構築
        pred_lists = {i: [] for i in range(num_nodes)}
        for src, succs in graph.items():
            for dst in succs:
                if 0 <= dst < num_nodes:
                    pred_lists[dst].append(src)
        max_pred = max((len(lst) for lst in pred_lists.values()), default=0)
        
        # 固定サイズの配列に変換
        pred_indices = np.full((num_nodes, max_pred), -1, dtype=np.int32)
        for i in range(num_nodes):
            for j, p in enumerate(pred_lists[i]):
                pred_indices[i, j] = p
        pred_counts = np.array([len(pred_lists[i]) for i in range(num_nodes)], dtype=np.int32)
        
        # CuPy 配列へ転送
        pred_counts_gpu = cp.asarray(pred_counts)
        pred_indices_gpu = cp.asarray(pred_indices.flatten())
        return pred_counts_gpu, pred_indices_gpu, max_pred

# ========================================================
# 3. GPU dominator 計算（ダブルバッファリングによる同期削減版）
# ========================================================
class DominatorCalculator:
    # ダブルバッファリング用カーネルコード
    COMBINED_KERNEL_CODE = r'''
    #define MAX_PRED %d
    extern "C"
    __global__ void compute_doms_and_idom(const short n, const short root, const unsigned int U,
                                            const int * __restrict__ pred_counts,
                                            const int * __restrict__ pred_indices,
                                            unsigned int * __restrict__ dom,
                                            short * __restrict__ idom,
                                            const int iterations)
    {
        // ダブルバッファ用に共有メモリ領域を2つ確保：サイズは n ごと
        extern __shared__ unsigned int shared_mem[];
        unsigned int* dom_in  = shared_mem;         // サイズ n
        unsigned int* dom_out = shared_mem + n;       // サイズ n
        
        int tid = threadIdx.x;
        if (tid < n)
            dom_in[tid] = dom[tid];
        __syncthreads();

        for (int it = 0; it < iterations; it++) {
            unsigned int newDom;
            if (tid == root) {
                newDom = (1u << tid);
            } else {
                newDom = U;
                int count = __ldg(&pred_counts[tid]);
                #pragma unroll
                for (int j = 0; j < MAX_PRED; j++) {
                    if (j < count) {
                        int p = __ldg(&pred_indices[tid * MAX_PRED + j]);
                        newDom &= dom_in[p];
                    }
                }
                newDom |= (1u << tid);
            }
            // 各スレッドが結果を書き込む
            dom_out[tid] = newDom;
            __syncthreads();
            // バッファを入れ替える（ポインタのスワップ）
            unsigned int* temp = dom_in;
            dom_in = dom_out;
            dom_out = temp;
            __syncthreads();
        }
        
        // 最終結果は dom_in にある（反復回数が偶数の場合）
        if (tid < n) {
            dom[tid] = dom_in[tid];
            if (tid == root) {
                idom[tid] = -1;
            } else {
                unsigned int mask = dom_in[tid] & ~(1u << tid);
                idom[tid] = (mask != 0) ? (31 - __clz(mask)) : -1;
            }
        }
    }
    '''
    
    def __init__(self, num_nodes: int, root: int, U: int, max_pred: int) -> None:
        self.num_nodes = num_nodes
        self.root = root
        self.U = U
        self.max_pred = max_pred
        self.block_size = num_nodes    # CFG サイズが小さい前提：全ノードを1ブロックで処理
        self.grid_size = 1
        kernel_source = self.COMBINED_KERNEL_CODE % self.max_pred
        self.module = SourceModule(kernel_source, options=["--use_fast_math"])
        self.combined_kernel = self.module.get_function("compute_doms_and_idom")
    
    def run(self, pred_counts_gpu: cp.ndarray, pred_indices_gpu: cp.ndarray,
            dom_init: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        num_nodes = self.num_nodes
        dom_host = cuda.pagelocked_empty(num_nodes, dtype=np.uint32)
        idom_host = cuda.pagelocked_empty(num_nodes, dtype=np.int16)
        dom_gpu  = cuda.mem_alloc(dom_init.nbytes)
        idom_gpu = cuda.mem_alloc(idom_host.nbytes)
        
        cuda.memcpy_htod(dom_gpu, dom_init)
        iterations = np.int32(num_nodes)
        # 共有メモリサイズは、ダブルバッファリング用に 2 * n * sizeof(uint32)
        shared_mem_size = self.block_size * np.dtype(np.uint32).itemsize * 2
        
        self.combined_kernel(np.int16(num_nodes), np.int16(self.root), np.uint32(self.U),
                             np.intp(int(pred_counts_gpu.data.ptr)),
                             np.intp(int(pred_indices_gpu.data.ptr)),
                             dom_gpu, idom_gpu, iterations,
                             block=(self.block_size, 1, 1),
                             grid=(self.grid_size, 1),
                             shared=shared_mem_size)
        cuda.Context.synchronize()
        
        cuda.memcpy_dtoh(dom_host, dom_gpu)
        cuda.memcpy_dtoh(idom_host, idom_gpu)
        cuda.Context.synchronize()
        
        return dom_host, idom_host

# ========================================================
# 4. 補助関数
# ========================================================
def bitmask_to_set(mask: int) -> Set[int]:
    result = set()
    index = 0
    while mask:
        if mask & 1:
            result.add(index)
        mask >>= 1
        index += 1
    return result

# ========================================================
# 5. 事前計算処理
# ========================================================
def precompute_data(source_code: str) -> Tuple[
        Dict[int, List[int]], Dict[int, str], int, int,
        cp.ndarray, cp.ndarray, int, int, np.ndarray]:
    # 1. CFG 構築
    graph, labels = ast_to_cfg(source_code)
    num_nodes = len(graph)
    root = 0
    # 2. 固定サイズ前駆データ生成（最適化版）
    pred_counts_gpu, pred_indices_gpu, max_pred = GraphPreprocessor.build_predecessor_fixed(graph, num_nodes)
    # 3. 初期 dominator セットの構築
    U = (1 << num_nodes) - 1
    dom_init = np.empty(num_nodes, dtype=np.uint32)
    for v in range(num_nodes):
        dom_init[v] = (1 << v) if v == root else U
    return graph, labels, num_nodes, root, pred_counts_gpu, pred_indices_gpu, max_pred, U, dom_init

# ========================================================
# 6. エンドツーエンド実行パイプライン（実行時間計測付き、print時間除外）
# ========================================================
def run_dominator_computation(sample_code: str, verbose: bool = True) -> Tuple[float, float, dict]:
    # ----- 計算部分の時間計測開始（printは含まない） -----
    start_total = time.time()
    graph, labels, num_nodes, root, pred_counts_gpu, pred_indices_gpu, max_pred, U, dom_init = precompute_data(sample_code)
    dominator_calc = DominatorCalculator(num_nodes, root, U, max_pred)
    start_gpu = time.time()
    dom_host, idom_host = dominator_calc.run(pred_counts_gpu, pred_indices_gpu, dom_init)
    gpu_time = (time.time() - start_gpu) * 1000  # ミリ秒単位
    comp_time = time.time() - start_total        # 計算処理のみの時間（秒）
    # ----- 計算終了、ここで時間測定を完了（以下の print は計測に含まれない） -----
    
    # 結果をまとめる
    result = {
        'graph': graph,
        'labels': labels,
        'dom_host': dom_host,
        'idom_host': idom_host,
        'max_pred': max_pred,
        'num_nodes': num_nodes
    }
    
    if verbose:
        print("\n==== Constructed CFG (Precomputed) ====")
        for node_id in sorted(graph.keys()):
            print(f"{node_id}: {labels[node_id]} -> {graph[node_id]}")
        print(f"Max predecessors per node: {max_pred}")
        print(f"\nCombined GPU Kernel Total Time: {gpu_time:.3f} ms")
        
        print("\n==== Dominator Sets (Dom) ====")
        for v in range(num_nodes):
            dom_set = sorted(bitmask_to_set(int(dom_host[v])))
            print(f"v = {v:2d}: {dom_set}")
        
        print("\n==== Immediate Dominators (idom) ====")
        for v in range(num_nodes):
            print(f"v = {v:2d}: idom = {idom_host[v]}")
    
    # 戻り値として、計算時間（秒）、GPU カーネル時間（ミリ秒）、および結果データを返す
    return comp_time, gpu_time, result

# ========================================================
# 7. ベンチマーク処理の追加
# ========================================================
def benchmark_pipeline(sample_code: str, iterations: int = 10) -> None:
    total_times = []
    gpu_times = []
    for i in range(iterations):
        # verbose を False にして出力を抑制（printの時間は含まれない）
        comp_time, gpu_time, _ = run_dominator_computation(sample_code, verbose=False)
        total_times.append(comp_time)
        gpu_times.append(gpu_time)
    avg_total = np.mean(total_times) * 1000  # ミリ秒換算
    avg_gpu = np.mean(gpu_times)
    print(f"\n=== Benchmark Results over {iterations} iterations ===")
    print(f"Average Total Pipeline Time (without print): {avg_total:.3f} ms")
    print(f"Average GPU Kernel Time: {avg_gpu:.3f} ms")

# ========================================================
# 8. メイン処理
# ========================================================
def main() -> None:
    sample_code = """
a = 1
if a > 0:
    print("positive")
    a = a - 1
else:
    print("non-positive")
    a = a + 1
while a < 10:
    a += 1
print(a)
"""
    print("=== Dominator Computation Pipeline (Optimized Version) ===")
    # 1回の実行結果を詳細表示（print時間は計測に含まれない）
    comp_time, gpu_time, _ = run_dominator_computation(sample_code, verbose=True)
    print(f"\nTotal Computation Time (without print): {comp_time*1000:.3f} ms")
    
    # ベンチマーク実行（例として10回実行）
    benchmark_pipeline(sample_code, iterations=10)

if __name__ == '__main__':
    main()

