#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU‐only Immediate Dominator Calculation using PyCUDA

【前提】
- グラフは 0～n-1 の頂点番号が DFS 順序（または近似）になっていると仮定（例: 5頂点）
- 各頂点の dominator set は 32ビット整数（ビットマスク）で表現（n <= 32）
- 初期 dominator set の計算、固定点反復、即時支配者の計算を可能な限り CUDA 上で実施

【処理の流れ】
1. ホスト側でグラフの前駆リストを作成し、ピン留めメモリとして GPU に転送
2. GPU 前処理カーネル (init_preproc) により、各頂点の初期 dominator set および GrayCode ラベルを並列計算
3. GPU 固定点反復カーネル (update_dom_shared_opt) により、dominators の固定点反復を実施
4. GPU 即時支配者計算カーネル (compute_idom) により、各頂点の即時支配者を計算（v==root は -1）
5. 結果をホスト側へ転送し、各フェーズの計算時間と結果を出力（print の時間は含まない）
"""

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time

# ====================================================
# 1. ホスト側前処理：グラフの前駆リスト作成と配列準備
# ====================================================
def compute_predecessors(graph):
    """グラフ（隣接リスト形式）から各頂点の前駆リストを作成"""
    pred = {v: [] for v in graph}
    for v in graph:
        for w in graph[v]:
            pred[w].append(v)
    return pred

def prepare_graph_arrays(graph):
    """
    各頂点の前駆リストをフラットな配列と、各頂点の開始位置配列に変換
    戻り値: (pred_list_np, pred_offsets_np)
    """
    n = len(graph)
    pred = compute_predecessors(graph)
    pred_list = []
    pred_offsets = [0]
    for v in range(n):
        plist = pred.get(v, [])
        pred_list.extend(plist)
        pred_offsets.append(len(pred_list))
    # ピン留めメモリで確保（小規模なら十分高速）
    pred_list_np = cuda.pagelocked_empty(len(pred_list), dtype=np.int32)
    pred_list_np[:] = np.array(pred_list, dtype=np.int32)
    pred_offsets_np = cuda.pagelocked_empty(len(pred_offsets), dtype=np.int32)
    pred_offsets_np[:] = np.array(pred_offsets, dtype=np.int32)
    return pred_list_np, pred_offsets_np

# ====================================================
# 2. GPU 前処理カーネル：初期 dominator set と GrayCode の計算
# ====================================================
init_kernel_code = r'''
extern "C"
__global__ void init_preproc(int n, int root, unsigned int U,
                             unsigned int *dom, int *gray)
{
    int v = blockDim.x * blockIdx.x + threadIdx.x;
    if (v >= n) return;
    // GrayCode 計算: v XOR (v >> 1)
    gray[v] = v ^ (v >> 1);
    // dominator set 初期化：root は自分のみ、その他は全ビット1
    if (v == root)
        dom[v] = (1 << v);
    else
        dom[v] = U;
}
'''

mod_init = SourceModule(init_kernel_code, options=["--use_fast_math"])
init_preproc = mod_init.get_function("init_preproc")

# ====================================================
# 3. 最適化した固定点反復カーネル（共有メモリ・ブロック内変更集約）
# ====================================================
update_kernel_code = r'''
extern "C"
__global__ void update_dom_shared_opt(int n, unsigned int U, unsigned int *dom,
                                      const int *pred_offsets, const int *pred_list,
                                      int root, int *changed)
{
    // 各ブロックが担当する頂点範囲の決定
    int block_start = blockIdx.x * blockDim.x;
    int block_end = block_start + blockDim.x;
    if (block_end > n) block_end = n;
    int local_idx = threadIdx.x;
    int global_idx = block_start + local_idx;
    
    extern __shared__ unsigned int s_dom[];
    if (global_idx < block_end)
        s_dom[local_idx] = dom[global_idx];
    __syncthreads();
    
    // ブロック内変更を集約する共有変数
    __shared__ int block_changed;
    if (local_idx == 0)
        block_changed = 0;
    __syncthreads();
    
    // 固定点反復
    for (int iter = 0; iter < 1000; iter++) {
        bool local_changed = false;
        if (global_idx < block_end && global_idx != root) {
            unsigned int newDom = U;
            int start = pred_offsets[global_idx];
            int end = pred_offsets[global_idx + 1];
            for (int i = start; i < end; i++) {
                int u = pred_list[i];
                if (u >= block_start && u < block_end)
                    newDom &= s_dom[u - block_start];
                else
                    newDom &= dom[u];
            }
            newDom |= (1 << global_idx);
            if (newDom != s_dom[local_idx]) {
                s_dom[local_idx] = newDom;
                local_changed = true;
            }
        }
        __syncthreads();
        if (local_changed)
            atomicOr(&block_changed, 1);
        __syncthreads();
        if (block_changed == 0)
            break;
        if (local_idx == 0)
            block_changed = 0;
        __syncthreads();
    }
    if (global_idx < block_end)
        dom[global_idx] = s_dom[local_idx];
    
    if (local_idx == 0 && block_changed)
        atomicExch(changed, 1);
}
'''

mod_update = SourceModule(update_kernel_code, options=["--use_fast_math"])
update_dom_shared_opt = mod_update.get_function("update_dom_shared_opt")

# ====================================================
# 4. GPU 即時支配者計算カーネル：各頂点の dominator set から即時支配者を計算
# ====================================================
# このカーネルでは、各頂点 v (v != root) について、
# 「dom[v] に含まれる候補の中で最大の頂点番号（v自身を除く）を即時支配者として選ぶ」
# なお、グラフが DFS 順序になっていると仮定します。
idom_kernel_code = r'''
extern "C"
__global__ void compute_idom(int n, const unsigned int *dom, int root, int *idom)
{
    int v = blockDim.x * blockIdx.x + threadIdx.x;
    if (v >= n) return;
    if (v == root) {
        idom[v] = -1;
        return;
    }
    int candidate = -1;
    unsigned int dmask = dom[v];
    // vは dominator set に含まれているが除外
    for (int d = 0; d < n; d++) {
        if (d == v) continue;
        if (dmask & (1u << d)) {
            if (d > candidate) candidate = d;
        }
    }
    idom[v] = candidate;
}
'''

mod_idom = SourceModule(idom_kernel_code, options=["--use_fast_math"])
compute_idom = mod_idom.get_function("compute_idom")

# ====================================================
# 5. ホスト側後処理：結果の受信（DFS 順序計算は不要と仮定）
# ====================================================
def bitmask_to_set(mask):
    """ビットマスクを集合に変換（デバッグ用）"""
    s = set()
    i = 0
    while mask:
        if mask & 1:
            s.add(i)
        mask >>= 1
        i += 1
    return s

# ====================================================
# 6. メイン処理と実行時間計測（計算時間のみ：ms 単位、出力は最後）
# ====================================================
def main():
    # --- グラフ定義（例: 5頂点） ---
    graph = {
        0: [1, 2],
        1: [3],
        2: [3],
        3: [4],
        4: []
    }
    root = 0
    n = len(graph)
    
    results = {}  # 各フェーズの計算時間記録
    
    # ホスト側前処理：グラフ配列作成
    t0 = time.time()
    pred_list_np, pred_offsets_np = prepare_graph_arrays(graph)
    t1 = time.time()
    results["Host Graph Preprocessing"] = (t1 - t0)*1000  # ms
    
    # GPU 用初期化（ピン留めメモリ利用）
    U = (1 << n) - 1
    dom_host = cuda.pagelocked_empty(n, dtype=np.uint32)
    gray_host = cuda.pagelocked_empty(n, dtype=np.int32)
    changed_host = cuda.pagelocked_empty(1, dtype=np.int32)
    
    # GPU バッファの確保
    dom_gpu = cuda.mem_alloc(dom_host.nbytes)
    gray_gpu = cuda.mem_alloc(gray_host.nbytes)
    changed_gpu = cuda.mem_alloc(changed_host.nbytes)
    
    # 前駆情報の GPU 転送
    pred_list_gpu = cuda.mem_alloc(pred_list_np.nbytes)
    cuda.memcpy_htod(pred_list_gpu, pred_list_np)
    pred_offsets_gpu = cuda.mem_alloc(pred_offsets_np.nbytes)
    cuda.memcpy_htod(pred_offsets_gpu, pred_offsets_np)
    
    # 非同期ストリーム作成
    stream = cuda.Stream()
    
    # GPU 前処理カーネル呼び出し（非同期）
    block_size_init = 128
    grid_size_init = (n + block_size_init - 1) // block_size_init
    start_event = cuda.Event()
    end_event = cuda.Event()
    start_event.record(stream)
    init_preproc(np.int32(n), np.int32(root), np.uint32(U),
                 dom_gpu, gray_gpu,
                 block=(block_size_init, 1, 1), grid=(grid_size_init, 1), stream=stream)
    end_event.record(stream)
    stream.synchronize()
    results["GPU Preprocessing Kernel"] = start_event.time_till(end_event)
    
    # GPU 固定点反復ループ（最適化版カーネル、非同期）
    block_size = 128
    grid_size = (n + block_size - 1) // block_size
    shared_mem_size = block_size * np.uint32(0).nbytes
    iteration = 0
    compute_start = time.time()
    while True:
        iteration += 1
        changed_host[0] = 0
        cuda.memcpy_htod_async(changed_gpu, changed_host, stream)
        update_dom_shared_opt(np.int32(n), np.uint32(U), dom_gpu,
                              pred_offsets_gpu, pred_list_gpu,
                              np.int32(root), changed_gpu,
                              block=(block_size, 1, 1), grid=(grid_size, 1),
                              shared=shared_mem_size, stream=stream)
        cuda.memcpy_dtoh_async(changed_host, changed_gpu, stream)
        stream.synchronize()
        if changed_host[0] == 0:
            break
    compute_end = time.time()
    results["GPU Fixed-Point Iteration Loop"] = (compute_end - compute_start)*1000
    results["Iteration Count"] = iteration
    
    # GPU から dominator set 結果取得（非同期）
    cuda.memcpy_dtoh_async(dom_host, dom_gpu, stream)
    stream.synchronize()
    
    # GPU 即時支配者計算カーネル呼び出し
    idom_host = cuda.pagelocked_empty(n, dtype=np.int32)
    block_size_idom = 128
    grid_size_idom = (n + block_size_idom - 1) // block_size_idom
    start_event.record(stream)
    compute_idom(np.int32(n), dom_gpu, np.int32(root), cuda.Out(idom_host),
                 block=(block_size_idom, 1, 1), grid=(grid_size_idom, 1), stream=stream)
    end_event.record(stream)
    stream.synchronize()
    results["GPU Immediate Dominator Kernel"] = start_event.time_till(end_event)
    
    total_compute_time = (time.time() - t0)*1000  # ms（出力時間は含まない）
    
    # --- 以下、出力部（計算時間には含めない） ---
    print("\n==== Execution Time Results (ms) ====")
    for key, val in results.items():
        if key == "Iteration Count":
            print(f"{key}: {val}")
        else:
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

