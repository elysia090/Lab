#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最適化された GPU-only Immediate Dominator Calculation with Persistent Kernel and Vectorized Graph Preprocessing

【最適化ポイント】
- グラフ前処理部分でリスト内包表記を用いてループオーバーヘッドを低減
- CuPy の処理結果を非同期転送＋CUDAストリームで取得（CUDAピン留めメモリへの転送効率向上）
- 固定点反復カーネルでは、n<=warp size（例：n≦32）を前提として __syncwarp() を使用し、
  ブロック全体の同期(__syncthreads())のオーバーヘッドを削減
"""

import numpy as np
import cupy as cp
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time

# ----------------------------------------------------
# 1. グラフ前処理 (ホスト側)
# ----------------------------------------------------
def graph_to_edge_list(graph):
    """辞書形式のグラフ {vertex: [neighbors]} をエッジリスト (src, dst) に変換"""
    src_list = [v for v, neighbors in graph.items() for _ in neighbors]
    dst_list = [w for v, neighbors in graph.items() for w in neighbors]
    return np.array(src_list, dtype=np.int32), np.array(dst_list, dtype=np.int32)

def prepare_graph_arrays_vectorized(graph):
    """
    グラフを前処理して以下の配列を作成:
      - pred_list_np: 各頂点の前駆頂点をまとめたフラット配列（宛先でソート済み）
      - pred_offsets_np: 各頂点の前駆頂点リストの開始オフセット
    """
    src, dst = graph_to_edge_list(graph)
    n = len(graph)
    
    # CuPy により GPU 上で計算
    src_gpu = cp.asarray(src)
    dst_gpu = cp.asarray(dst)
    order = cp.argsort(dst_gpu)
    sorted_src_gpu = src_gpu[order]
    sorted_dst_gpu = dst_gpu[order]
    # cp.arange の結果が int64 にならないように、dtype を合わせる
    offsets_gpu = cp.searchsorted(sorted_dst_gpu, cp.arange(n + 1, dtype=sorted_dst_gpu.dtype))
    # 型が int32 であることを保証（ピン留めメモリの型と一致させる）
    if offsets_gpu.dtype != cp.int32:
        offsets_gpu = offsets_gpu.astype(cp.int32)
    
    # 非同期転送のための CUDA ストリーム作成
    stream = cuda.Stream()
    # CUDA ピン留めメモリに転送（非同期転送）
    pred_list_np = cuda.pagelocked_empty(int(sorted_src_gpu.size), dtype=np.int32)
    cp.cuda.runtime.memcpyAsync(
        pred_list_np.ctypes.data, 
        sorted_src_gpu.data.ptr, 
        int(sorted_src_gpu.nbytes), 
        cp.cuda.runtime.memcpyDeviceToHost, 
        stream.handle)
    stream.synchronize()  # 転送完了待ち

    pred_offsets_np = cuda.pagelocked_empty(int(offsets_gpu.size), dtype=np.int32)
    cp.cuda.runtime.memcpyAsync(
        pred_offsets_np.ctypes.data, 
        offsets_gpu.data.ptr, 
        int(offsets_gpu.nbytes), 
        cp.cuda.runtime.memcpyDeviceToHost, 
        stream.handle)
    stream.synchronize()
    
    return pred_list_np, pred_offsets_np

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
    dom[v] = (v == root) ? (1u << v) : U;
}
'''
mod_init = SourceModule(init_kernel_code, options=["--use_fast_math"])
init_preproc = mod_init.get_function("init_preproc")

# ----------------------------------------------------
# 3. 最適化された持続型カーネル (固定点反復 + 即時 dominator 計算)
# (n <= warp size の場合に __syncwarp() を使用)
# ----------------------------------------------------
persistent_idom_kernel_code = r'''
extern "C"
__global__ void persistent_idom_kernel(int n, int root, unsigned int U,
                                         const int *pred_offsets, const int *pred_list,
                                         int *idom, unsigned int *dom_out)
{
    int v = threadIdx.x;
    extern __shared__ unsigned int s_dom[];
    
    if (v < n)
        s_dom[v] = (v == root) ? (1u << v) : U;
    __syncwarp();  
    
    for (int iter = 0; iter < 10000; iter++) {
        __syncwarp();
        bool local_changed = false;
        if (v < n && v != root) {
            unsigned int newDom = U;
            int start = pred_offsets[v];
            int end = pred_offsets[v+1];
            for (int i = start; i < end; i++) {
                int u = pred_list[i];
                newDom &= s_dom[u];
            }
            newDom |= (1u << v);
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
# 5. メイン処理
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
    
    # グラフ前処理 (CuPy による GPU 上での計算＋非同期転送)
    pred_list_np, pred_offsets_np = prepare_graph_arrays_vectorized(graph)
    t1 = time.time()
    results["Host Graph Preprocessing (GPU-based)"] = (t1 - t0) * 1000
    
    # dominator セット初期値 U（各ビットが 1 の集合）
    U = (1 << n) - 1
    
    # ピン留めメモリの確保 (ホスト)
    dom_host   = cuda.pagelocked_empty(n, dtype=np.uint32)
    gray_host  = cuda.pagelocked_empty(n, dtype=np.int32)
    idom_host  = cuda.pagelocked_empty(n, dtype=np.int32)
    
    # GPU バッファの確保
    dom_gpu    = cuda.mem_alloc(dom_host.nbytes)
    gray_gpu   = cuda.mem_alloc(gray_host.nbytes)
    idom_gpu   = cuda.mem_alloc(idom_host.nbytes)
    
    # 非同期転送用のストリーム作成
    stream = cuda.Stream()
    pred_list_gpu = cuda.mem_alloc(pred_list_np.nbytes)
    cuda.memcpy_htod_async(pred_list_gpu, pred_list_np, stream)
    pred_offsets_gpu = cuda.mem_alloc(pred_offsets_np.nbytes)
    cuda.memcpy_htod_async(pred_offsets_gpu, pred_offsets_np, stream)
    stream.synchronize()
    
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
    
    # 持続型カーネルの実行
    # ※ n <= warp size (例: n≦32) を前提としているため、ブロックサイズは n
    block_size_persistent = n if n > 0 else 1
    grid_size_persistent  = 1
    shared_mem_size = block_size_persistent * np.uint32(0).nbytes
    persistent_start = time.time()
    persistent_idom_kernel(np.int32(n), np.int32(root), np.uint32(U),
                           pred_offsets_gpu, pred_list_gpu,
                           idom_gpu, dom_gpu,
                           block=(block_size_persistent, 1, 1), grid=(grid_size_persistent, 1),
                           shared=shared_mem_size)
    cuda.Context.synchronize()
    persistent_end = time.time()
    results["Persistent Kernel (Fixed-Point + Immediate Dominator)"] = (persistent_end - persistent_start) * 1000
    
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
