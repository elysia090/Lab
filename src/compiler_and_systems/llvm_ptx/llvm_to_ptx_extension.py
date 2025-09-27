#!/usr/bin/env python3
"""
Dominator Computation Pipeline with Host-side Result Readout

このパイプラインは以下の処理を行います:
1. AST から CFG を構築し dominator 計算を実施
2. 得られた結果（dominator, immediate dominator）を用いて、
   LLVM IR 内で各スレッドが自分の tid に対応する和を計算し、出力バッファに書き込むカーネルを生成
3. LLVM IR に NVVM アノテーションを付加し、llc を用いて NVPTX 用 PTX コードに変換
4. PyCUDA でカーネルを実行し、ホスト側で結果を読み出して表示
"""

import ast
import time
import subprocess
import tempfile
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional

import numpy as np
import cupy as cp
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


from llvmlite import ir, binding

# -------------------------
# 1. CFG 構築（AST → CFG）
# -------------------------
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
    def build_sequence(self, stmts: List[ast.stmt]) -> Tuple[Block, Optional[Block]]:
        if not stmts:
            empty_block = self.new_block("<empty>")
            return empty_block, empty_block
        first_block = None
        prev_block = None
        current_exit: Optional[Block] = None
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
                    else_entry = self.new_block("<empty>")
                    else_exit = self.new_block("<empty>")
                cond_block.successors.extend([then_entry.id, else_entry.id])
                join_block = self.new_block("<empty>")
                if then_exit is not None:
                    then_exit.successors.append(join_block.id)
                if else_exit is not None:
                    else_exit.successors.append(join_block.id)
                current_exit = self.new_block("<empty>")
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
                if body_exit is not None:
                    body_exit.successors.append(cond_block.id)
                exit_block = self.new_block("<empty>")
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
        final_exit = self.new_block("<empty>")
        if exit_blk is not None:
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

# -------------------------
# 2. グラフ前処理
# -------------------------
class GraphPreprocessor:
    @staticmethod
    def build_predecessor_fixed(graph: Dict[int, List[int]], num_nodes: int) -> Tuple[cp.ndarray, cp.ndarray, int]:
        pred_lists: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        for src, succs in graph.items():
            for dst in succs:
                if 0 <= dst < num_nodes:
                    pred_lists[dst].append(src)
        max_pred = max((len(lst) for lst in pred_lists.values()), default=0)
        pred_indices = np.full((num_nodes, max_pred), -1, dtype=np.int32)
        for i in range(num_nodes):
            for j, p in enumerate(pred_lists[i]):
                pred_indices[i, j] = p
        pred_counts = np.array([len(pred_lists[i]) for i in range(num_nodes)], dtype=np.int32)
        pred_counts_gpu = cp.asarray(pred_counts)
        pred_indices_gpu = cp.asarray(pred_indices.flatten())
        return pred_counts_gpu, pred_indices_gpu, max_pred

# -------------------------
# 3. GPU Dominator 計算
# -------------------------
class DominatorCalculator:
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
        extern __shared__ unsigned int shared_mem[];
        unsigned int* dom_in  = shared_mem;
        unsigned int* dom_out = shared_mem + n;
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
            dom_out[tid] = newDom;
            __syncthreads();
            unsigned int* temp = dom_in;
            dom_in = dom_out;
            dom_out = temp;
            __syncthreads();
        }
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
        self.block_size = num_nodes  # 全ノードを 1 ブロックで処理
        self.grid_size = 1
        kernel_source = self.COMBINED_KERNEL_CODE % self.max_pred
        self.module = SourceModule(kernel_source, options=["--use_fast_math"])
        self.combined_kernel = self.module.get_function("compute_doms_and_idom")
    def run(self, pred_counts_gpu: cp.ndarray, pred_indices_gpu: cp.ndarray,
            dom_init: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        num_nodes = self.num_nodes
        dom_host = cuda.pagelocked_empty(num_nodes, dtype=np.uint32)
        idom_host = cuda.pagelocked_empty(num_nodes, dtype=np.int16)
        dom_gpu = cuda.mem_alloc(dom_init.nbytes)
        idom_gpu = cuda.mem_alloc(idom_host.nbytes)
        cuda.memcpy_htod(dom_gpu, dom_init)
        iterations = np.int32(num_nodes)
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

# -------------------------
# 4. ユーティリティ関数
# -------------------------
def bitmask_to_set(mask: int) -> Set[int]:
    """ビットマスクを集合に変換"""
    result = set()
    index = 0
    while mask:
        if mask & 1:
            result.add(index)
        mask >>= 1
        index += 1
    return result

# -------------------------
# 5. 前計算処理
# -------------------------
def precompute_data(source_code: str) -> Tuple[
        Dict[int, List[int]], Dict[int, str], int, int,
        cp.ndarray, cp.ndarray, int, int, np.ndarray]:
    """
    ソースコードから CFG を構築し、グラフの前処理および dominator 初期値を設定
    """
    graph, labels = ast_to_cfg(source_code)
    num_nodes = len(graph)
    root = 0
    pred_counts_gpu, pred_indices_gpu, max_pred = GraphPreprocessor.build_predecessor_fixed(graph, num_nodes)
    U = (1 << num_nodes) - 1
    dom_init = np.empty(num_nodes, dtype=np.uint32)
    for v in range(num_nodes):
        dom_init[v] = (1 << v) if v == root else U
    return graph, labels, num_nodes, root, pred_counts_gpu, pred_indices_gpu, max_pred, U, dom_init

# -------------------------
# 6. エンドツーエンドパイプライン
# -------------------------
def run_dominator_computation(source_code: str, verbose: bool = True) -> Tuple[float, float, dict]:
    """
    ソースコードから CFG 構築、GPU で dominator 計算を実施し、結果を表示
    """
    start_total = time.time()
    graph, labels, num_nodes, root, pred_counts_gpu, pred_indices_gpu, max_pred, U, dom_init = precompute_data(source_code)
    dominator_calc = DominatorCalculator(num_nodes, root, U, max_pred)
    start_gpu = time.time()
    dom_host, idom_host = dominator_calc.run(pred_counts_gpu, pred_indices_gpu, dom_init)
    gpu_time = (time.time() - start_gpu) * 1000  # ミリ秒単位
    comp_time = time.time() - start_total
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
    return comp_time, gpu_time, result

# -------------------------
# 7. ベンチマーク処理
# -------------------------
def benchmark_pipeline(source_code: str, iterations: int = 10) -> None:
    """複数回実行してパイプラインの平均実行時間を表示"""
    total_times = []
    gpu_times = []
    for _ in range(iterations):
        comp_time, gpu_time, _ = run_dominator_computation(source_code, verbose=False)
        total_times.append(comp_time)
        gpu_times.append(gpu_time)
    avg_total = np.mean(total_times) * 1000  # ミリ秒換算
    avg_gpu = np.mean(gpu_times)
    print(f"\n=== Benchmark Results over {iterations} iterations ===")
    print(f"Average Total Pipeline Time (without print): {avg_total:.3f} ms")
    print(f"Average GPU Kernel Time: {avg_gpu:.3f} ms")

# -------------------------
# 8. LLVM IR / PTX 生成とカーネル実行（ホスト側で読みだす）
# -------------------------
def execute_ptx_kernel(result: dict) -> None:
    """
    dominator 計算結果を LLVM IR のグローバル変数に配置し、
    各スレッドが自分の tid と dominator, immediate dominator の和を計算して
    出力バッファに書き込むカーネル関数 (main_kernel) を生成する。
    その後、生成された PTX コードを用いてカーネルを実行し、
    ホスト側で出力結果を読み出して表示する。
    """
    num_nodes = result['num_nodes']
    dom_values = [int(val) for val in result['dom_host']]
    idom_values = [int(val) for val in result['idom_host']]

    module = ir.Module(name="ptx_module")
    module.triple = "nvptx64-nvidia-cuda"

    # グローバル変数定義
    dom_array_type = ir.ArrayType(ir.IntType(32), num_nodes)
    dom_array = ir.GlobalVariable(module, dom_array_type, name="dom_host")
    dom_array.linkage = "internal"
    dom_array.global_constant = True
    dom_array.initializer = ir.Constant(dom_array_type, [ir.Constant(ir.IntType(32), v) for v in dom_values])

    idom_array_type = ir.ArrayType(ir.IntType(16), num_nodes)
    idom_array = ir.GlobalVariable(module, idom_array_type, name="idom_host")
    idom_array.linkage = "internal"
    idom_array.global_constant = True
    idom_array.initializer = ir.Constant(idom_array_type, [ir.Constant(ir.IntType(16), v) for v in idom_values])

    # カーネル関数: void main_kernel(i32* result)
    func_type = ir.FunctionType(ir.VoidType(), [ir.PointerType(ir.IntType(32))])
    kernel_func = ir.Function(module, func_type, name="main_kernel")
    result_arg = kernel_func.args[0]
    result_arg.name = "result"

    entry_block = kernel_func.append_basic_block(name="entry")
    builder = ir.IRBuilder(entry_block)

    # スレッドID取得
    tid_func = ir.Function(module, ir.FunctionType(ir.IntType(32), []), name="llvm.nvvm.read.ptx.sreg.tid.x")
    tid = builder.call(tid_func, [])

    # if (tid < num_nodes)
    num_nodes_const = ir.Constant(ir.IntType(32), num_nodes)
    cmp = builder.icmp_signed("<", tid, num_nodes_const)
    then_block = kernel_func.append_basic_block("then")
    else_block = kernel_func.append_basic_block("else")
    builder.cbranch(cmp, then_block, else_block)

    # then ブロック: 結果計算＆出力バッファへ書き込み
    builder.position_at_start(then_block)
    dom_ptr = builder.gep(dom_array, [ir.Constant(ir.IntType(32), 0), tid])
    dom_val = builder.load(dom_ptr)
    idom_ptr = builder.gep(idom_array, [ir.Constant(ir.IntType(32), 0), tid])
    idom_val = builder.load(idom_ptr)
    idom_val_i32 = builder.sext(idom_val, ir.IntType(32))
    sum_val = builder.add(dom_val, idom_val_i32)
    res_ptr = builder.gep(result_arg, [tid])
    builder.store(sum_val, res_ptr)
    builder.branch(else_block)

    # else ブロック: 終了
    builder.position_at_start(else_block)
    builder.ret_void()

    # --- NVVM カーネルアノテーションを追加 ---
    llvm_ir = str(module)
    annotation = r"""
!nvvm.annotations = !{!0}
!0 = !{void (i32*)* @main_kernel, !"kernel", i32 1}
"""
    llvm_ir += annotation

    print("\nGenerated LLVM IR for kernel (with NVVM annotation):")
    print(llvm_ir)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ll", delete=False) as f:
        f.write(llvm_ir)
        ll_file = f.name
    ptx_file = ll_file.replace(".ll", ".ptx")
    subprocess.run(["llc", "-O3", "-march=nvptx64", "-mcpu=sm_35",
                    "-unroll-threshold=150", ll_file, "-o", ptx_file], check=True)
    with open(ptx_file, "r") as f:
        ptx = f.read()
    os.remove(ll_file)
    os.remove(ptx_file)
    print("\nGenerated PTX Code for kernel (with NVVM annotation):")
    print(ptx)

    # PyCUDA によりカーネル実行
    mod_cuda = cuda.module_from_buffer(ptx.encode("utf-8"))
    kernel = mod_cuda.get_function("main_kernel")
    result_host = np.zeros(num_nodes, dtype=np.int32)
    result_gpu = cuda.mem_alloc(result_host.nbytes)
    block_size = (256, 1, 1)
    grid_size = (1, 1)
    kernel(result_gpu, block=block_size, grid=grid_size)
    cuda.Context.synchronize()
    cuda.memcpy_dtoh(result_host, result_gpu)
    print("\nKernel output (host side read):")
    for i, val in enumerate(result_host):
        print(f"Result for tid {i}: {val}")

# -------------------------
# 9. メイン処理
# -------------------------
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
    print("=== Dominator Computation Pipeline ===")
    comp_time, gpu_time, result = run_dominator_computation(sample_code, verbose=True)
    print(f"\nTotal Computation Time: {comp_time*1000:.3f} ms")
    benchmark_pipeline(sample_code, iterations=10)
    # ホスト側で結果を読み出すカーネルを実行
    execute_ptx_kernel(result)

if __name__ == '__main__':
    main()

"""　結果
=== Dominator Computation Pipeline ===

==== Constructed CFG (Precomputed) ====
0: a = 1 -> [1]
1: a > 0 -> [2, 4]
2: print('positive') -> [3]
3: a = a - 1 -> [6]
4: print('non-positive') -> [5]
5: a = a + 1 -> [6]
6: <empty> -> [7]
7: <empty> -> [8]
8: a < 10 -> [9, 10]
9: a += 1 -> [8]
10: <empty> -> [11]
11: print(a) -> [12]
12: <empty> -> []
Max predecessors per node: 2

Combined GPU Kernel Total Time: 2.195 ms

==== Dominator Sets (Dom) ====
v =  0: [0]
v =  1: [0, 1]
v =  2: [0, 1, 2]
v =  3: [0, 1, 2, 3]
v =  4: [0, 1, 4]
v =  5: [0, 1, 4, 5]
v =  6: [0, 1, 6]
v =  7: [0, 1, 6, 7]
v =  8: [0, 1, 6, 7, 8]
v =  9: [0, 1, 6, 7, 8, 9]
v = 10: [0, 1, 6, 7, 8, 10]
v = 11: [0, 1, 6, 7, 8, 10, 11]
v = 12: [0, 1, 6, 7, 8, 10, 11, 12]

==== Immediate Dominators (idom) ====
v =  0: idom = -1
v =  1: idom = 0
v =  2: idom = 1
v =  3: idom = 2
v =  4: idom = 1
v =  5: idom = 4
v =  6: idom = 1
v =  7: idom = 6
v =  8: idom = 7
v =  9: idom = 8
v = 10: idom = 8
v = 11: idom = 10
v = 12: idom = 11

Total Computation Time: 979.419 ms

=== Benchmark Results over 10 iterations ===
Average Total Pipeline Time (without print): 0.845 ms
Average GPU Kernel Time: 0.137 ms

Generated LLVM IR for kernel (with NVVM annotation):
; ModuleID = "ptx_module"
target triple = "nvptx64-nvidia-cuda"
target datalayout = ""

@"dom_host" = internal constant [13 x i32] [i32 1, i32 3, i32 7, i32 15, i32 19, i32 51, i32 67, i32 195, i32 451, i32 963, i32 1475, i32 3523, i32 7619]
@"idom_host" = internal constant [13 x i16] [i16 -1, i16 0, i16 1, i16 2, i16 1, i16 4, i16 1, i16 6, i16 7, i16 8, i16 8, i16 10, i16 11]
define void @"main_kernel"(i32* %"result")
{
entry:
  %".3" = call i32 @"llvm.nvvm.read.ptx.sreg.tid.x"()
  %".4" = icmp slt i32 %".3", 13
  br i1 %".4", label %"then", label %"else"
then:
  %".6" = getelementptr [13 x i32], [13 x i32]* @"dom_host", i32 0, i32 %".3"
  %".7" = load i32, i32* %".6"
  %".8" = getelementptr [13 x i16], [13 x i16]* @"idom_host", i32 0, i32 %".3"
  %".9" = load i16, i16* %".8"
  %".10" = sext i16 %".9" to i32
  %".11" = add i32 %".7", %".10"
  %".12" = getelementptr i32, i32* %"result", i32 %".3"
  store i32 %".11", i32* %".12"
  br label %"else"
else:
  ret void
}

declare i32 @"llvm.nvvm.read.ptx.sreg.tid.x"()

!nvvm.annotations = !{!0}
!0 = !{void (i32*)* @main_kernel, !"kernel", i32 1}


Generated PTX Code for kernel (with NVVM annotation):
//
// Generated by LLVM NVPTX Back-End
//

.version 3.2
.target sm_35
.address_size 64

	// .globl	main_kernel             // -- Begin function main_kernel
.global .align 4 .b8 dom_host[52] = {1, 0, 0, 0, 3, 0, 0, 0, 7, 0, 0, 0, 15, 0, 0, 0, 19, 0, 0, 0, 51, 0, 0, 0, 67, 0, 0, 0, 195, 0, 0, 0, 195, 1, 0, 0, 195, 3, 0, 0, 195, 5, 0, 0, 195, 13, 0, 0, 195, 29, 0, 0};
.global .align 2 .b8 idom_host[26] = {255, 255, 0, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 6, 0, 7, 0, 8, 0, 8, 0, 10, 0, 11, 0};
                                        // @main_kernel
.visible .entry main_kernel(
	.param .u64 main_kernel_param_0
)
{
	.reg .pred 	%p<2>;
	.reg .b16 	%rs<2>;
	.reg .b32 	%r<5>;
	.reg .b64 	%rd<10>;

// %bb.0:                               // %entry
	mov.u32 	%r1, %tid.x;
	setp.gt.s32 	%p1, %r1, 12;
	@%p1 bra 	LBB0_2;
// %bb.1:                               // %then
	ld.param.u64 	%rd2, [main_kernel_param_0];
	cvta.to.global.u64 	%rd1, %rd2;
	mul.wide.s32 	%rd3, %r1, 4;
	mov.u64 	%rd4, dom_host;
	add.s64 	%rd5, %rd4, %rd3;
	ld.global.nc.u32 	%r2, [%rd5];
	mul.wide.s32 	%rd6, %r1, 2;
	mov.u64 	%rd7, idom_host;
	add.s64 	%rd8, %rd7, %rd6;
	ld.global.nc.u16 	%rs1, [%rd8];
	cvt.s32.s16 	%r3, %rs1;
	add.s32 	%r4, %r2, %r3;
	add.s64 	%rd9, %rd1, %rd3;
	st.global.u32 	[%rd9], %r4;
LBB0_2:                                 // %else
	ret;
                                        // -- End function
}


Kernel output (host side read):
Result for tid 0: 0
Result for tid 1: 3
Result for tid 2: 8
Result for tid 3: 17
Result for tid 4: 20
Result for tid 5: 55
Result for tid 6: 68
Result for tid 7: 201
Result for tid 8: 458
Result for tid 9: 971
Result for tid 10: 1483
Result for tid 11: 3533
Result for tid 12: 7630

"""
