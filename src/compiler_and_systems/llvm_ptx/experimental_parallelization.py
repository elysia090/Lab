#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  - グラフ前処理はGPU上で完結してホストとデバイス間の転送を削減
  - 永続型カーネル内では不要な同期呼び出しを削減し、各スレッドでのシフト演算を事前計算
  - 必要に応じて、複数ストリームやCUDA Graphsによるオーバーラッピングも検討可能
"""

import numpy as np
import cupy as cp
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time

# ----------------------------------------------------
# 1. グラフ前処理 (GPU上で完結)
# ----------------------------------------------------
def graph_to_edge_list(graph):
    """辞書形式のグラフ {vertex: [neighbors]} をエッジリスト (src, dst) に変換"""
    src_list = [v for v, neighbors in graph.items() for _ in neighbors]
    dst_list = [w for v, neighbors in graph.items() for w in neighbors]
    return np.array(src_list, dtype=np.int32), np.array(dst_list, dtype=np.int32)

def prepare_graph_arrays_vectorized(graph):
    """
    グラフをGPU上で前処理して以下のCuPy配列を作成:
      - sorted_src_gpu: 各頂点の前駆頂点をまとめたフラット配列（宛先でソート済み）
      - offsets_gpu: 各頂点の前駆頂点リストの開始オフセット
    ホストとデバイス間の転送を削減するため、結果はGPU上に保持します。
    """
    src, dst = graph_to_edge_list(graph)
    n = len(graph)
    src_gpu = cp.asarray(src)
    dst_gpu = cp.asarray(dst)
    order = cp.argsort(dst_gpu)
    sorted_src_gpu = src_gpu[order]
    sorted_dst_gpu = dst_gpu[order]
    offsets_gpu = cp.searchsorted(sorted_dst_gpu, cp.arange(n + 1, dtype=sorted_dst_gpu.dtype))
    if offsets_gpu.dtype != cp.int32:
        offsets_gpu = offsets_gpu.astype(cp.int32)
    return sorted_src_gpu, offsets_gpu

# ----------------------------------------------------
# 2. GPU 前処理カーネル (init_preproc)
# ----------------------------------------------------
init_kernel_code = r'''
extern "C"
__global__ void init_preproc(int n, int root, unsigned int U,
                             unsigned int *dom, int *gray)
{
    int v = blockDim.x * blockIdx.x + threadIdx.x;
    if (v >= n) return;
    gray[v] = v ^ (v >> 1);
    unsigned int my_bit = (1u << v);
    dom[v] = (v == root) ? my_bit : U;
}
'''
mod_init = SourceModule(init_kernel_code, options=["--use_fast_math"])
init_preproc = mod_init.get_function("init_preproc")

# ----------------------------------------------------
# 3. 永続型カーネル (固定点反復 + 即時 dominator 計算)
#    restrict 指定子と __ldg() による読み出しの最適化
#    ※ n <= warp size (例: n≦32) を前提
# ----------------------------------------------------
persistent_idom_kernel_code = r'''
extern "C"
__global__ void persistent_idom_kernel(int n, int root, unsigned int U,
                                         const int * __restrict__ pred_offsets,
                                         const int * __restrict__ pred_list,
                                         int *idom, unsigned int *dom_out)
{
    int v = threadIdx.x;
    extern __shared__ unsigned int s_dom[];

    unsigned int my_bit = (1u << v);
    
    if (v < n)
        s_dom[v] = (v == root) ? my_bit : U;
    
    __syncwarp();
    
    for (int iter = 0; iter < 10000; iter++) {
        bool local_changed = false;
        if (v < n && v != root) {
            unsigned int newDom = U;
            int start = __ldg(&pred_offsets[v]);
            int end   = __ldg(&pred_offsets[v+1]);
            for (int i = start; i < end; i++) {
                int u = __ldg(&pred_list[i]);
                newDom &= s_dom[u];
            }
            newDom |= my_bit;
            if (newDom != s_dom[v]) {
                s_dom[v] = newDom;
                local_changed = true;
            }
        }
        __syncwarp();  
        unsigned int ballot = __ballot_sync(0xFFFFFFFF, local_changed);
        if (ballot == 0)
            break;
    }
    __syncwarp();
    
    if (v < n)
        dom_out[v] = s_dom[v];
    
    if (v < n) {
        if (v == root)
            idom[v] = -1;
        else {
            unsigned int mask = s_dom[v] & ~(1u << v);
            idom[v] = (mask) ? (31 - __clz(mask)) : -1;
        }
    }
}
'''
mod_persistent = SourceModule(persistent_idom_kernel_code, options=["--use_fast_math"])
persistent_idom_kernel = mod_persistent.get_function("persistent_idom_kernel")

# ----------------------------------------------------
# 4. ビットマスクから集合への変換 (デバッグ用)
# ----------------------------------------------------
def bitmask_to_set(mask):
    s = set()
    i = 0
    while mask:
        if mask & 1:
            s.add(i)
        mask >>= 1
        i += 1
    return s

# ----------------------------------------------------
# 5. メイン処理 (さらに最適化版)
# ----------------------------------------------------
def main():
    # グラフ定義 (例: 5 頂点)
    graph = {
        0: [1, 2],
        1: [3],
        2: [3],
        3: [4],
        4: []
    }
    root = 0
    n = len(graph)
    
    results = {}
    t0 = time.time()
    
    # グラフ前処理をGPU上で実施（ホストとの転送を削減）
    pred_list_gpu, pred_offsets_gpu = prepare_graph_arrays_vectorized(graph)
    t1 = time.time()
    results["GPU-based Graph Preprocessing"] = (t1 - t0) * 1000
    
    # dominator セット初期値 U（各ビットが 1 の集合）
    U = (1 << n) - 1
    
    # ピン留めメモリの確保（ホスト）
    dom_host   = cuda.pagelocked_empty(n, dtype=np.uint32)
    gray_host  = cuda.pagelocked_empty(n, dtype=np.int32)
    idom_host  = cuda.pagelocked_empty(n, dtype=np.int32)
    
    # GPU バッファの確保
    dom_gpu    = cuda.mem_alloc(dom_host.nbytes)
    gray_gpu   = cuda.mem_alloc(gray_host.nbytes)
    idom_gpu   = cuda.mem_alloc(idom_host.nbytes)
    
    # ※ 必要に応じ、複数ストリームやCUDA Graphs API を用いて
    #     ホスト→デバイス転送・カーネル起動のオーバーラッピングによる最適化も検討可能
    
    # GPU 前処理カーネルの実行
    block_size_init = 128
    grid_size_init  = (n + block_size_init - 1) // block_size_init
    start_event = cuda.Event()
    end_event   = cuda.Event()
    start_event.record()
    init_preproc(np.int32(n), np.int32(root), np.uint32(U),
                 dom_gpu, gray_gpu,
                 block=(block_size_init, 1, 1), grid=(grid_size_init, 1))
    end_event.record()
    end_event.synchronize()
    results["GPU Preprocessing Kernel"] = start_event.time_till(end_event)
    
    # 永続型カーネルの実行
    # pred_list_gpu, pred_offsets_gpu はすでにGPU上にあるためホスト→GPU転送は不要
    pred_list_ptr    = int(pred_list_gpu.data.ptr)
    pred_offsets_ptr = int(pred_offsets_gpu.data.ptr)
    
    block_size_persistent = n if n > 0 else 1
    grid_size_persistent  = 1
    shared_mem_size = block_size_persistent * np.uint32(0).nbytes
    persistent_start = time.time()
    persistent_idom_kernel(np.int32(n), np.int32(root), np.uint32(U),
                             np.intp(pred_offsets_ptr), np.intp(pred_list_ptr),
                             idom_gpu, dom_gpu,
                             block=(block_size_persistent, 1, 1), grid=(grid_size_persistent, 1),
                             shared=shared_mem_size)
    cuda.Context.synchronize()
    persistent_end = time.time()
    results["Persistent Kernel"] = (persistent_end - persistent_start) * 1000
    
    # 結果の取得
    cuda.memcpy_dtoh(dom_host, dom_gpu)
    cuda.memcpy_dtoh(idom_host, idom_gpu)
    cuda.memcpy_dtoh(gray_host, gray_gpu)
    
    total_compute_time = (time.time() - t0) * 1000
    
    # 結果出力
    print("\n==== Execution Time Results (ms) ====")
    for key, val in results.items():
        print(f"{key}: {val:.3f} ms")
    print("Total Compute Time (excl. printing): {:.3f} ms".format(total_compute_time))
    
    print("\n==== Dominator Sets (Dom) ====")
    for v in range(n):
        print("  v = {:2d}: {}".format(v, sorted(bitmask_to_set(dom_host[v]))))
    
    print("\n==== Immediate Dominators (idom) ====")
    for v in range(n):
        print("  v = {:2d}: idom = {}".format(v, idom_host[v]))
    
    print("\n==== GrayCode Labels ====")
    for v in range(n):
        print("  v = {:2d}: gray = {}".format(v, gray_host[v]))
    
if __name__ == '__main__':
    main()
