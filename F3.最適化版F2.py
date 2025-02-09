#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

- グラフは辞書形式（例：5頂点）
- 各頂点の dominator set は 32ビット整数（ビットマスク）で表現
- GPU 上で、初期化カーネルにより dominator set の初期設定と GrayCode 計算を非同期・並列実行
- 固定点反復は、ブロック内で共有メモリとブロック単位の変更集約を用いて早期終了判定するカーネルで実施
- ホスト側前処理はピン留めメモリ、非同期転送、ストリームを活用して最適化
- 各フェーズの計算時間をミリ秒単位で測定し、出力（print）の時間は計測に含めない
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
    各頂点の前駆リストをフラットな配列と各頂点の開始位置配列に変換
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
    // dominator set 初期化：root は自分のみ、他は全ビット1
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
# 4. ホスト側後処理：DFS 順序と即時支配者決定
# ====================================================
def dfs_order(graph, root):
    """根からの DFS 順序を計算し、各頂点に順序番号を割り当てる"""
    order = {}
    visited = set()
    counter = 0
    def dfs(v):
        nonlocal counter
        visited.add(v)
        order[v] = counter
        counter += 1
        for w in graph[v]:
            if w not in visited:
                dfs(w)
    dfs(root)
    return order

def compute_immediate_dominators(dom, order, root, n):
    """
    各頂点 v (v ≠ root) について、dom[v] から v を除く候補の中で、
    DFS 順序が最大の頂点を即時支配者として選択
    """
    idom = {}
    for v in range(n):
        if v == root:
            continue
        candidates = []
        for d in range(n):
            if d == v:
                continue
            if dom[v] & (1 << d):
                candidates.append(d)
        if candidates:
            idom[v] = max(candidates, key=lambda d: order[d])
        else:
            idom[v] = None
    return idom

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
# 5. メイン処理（計算部分のみの時間計測）と最終出力（printは計測対象外）
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
    
    results = {}  # 各フェーズの計算時間を記録
    
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
    results["GPU Preprocessing Kernel"] = start_event.time_till(end_event)  # ms
    
    # GPU 固定点反復ループ（最適化版カーネル、非同期）
    block_size = 128
    grid_size = (n + block_size - 1) // block_size
    shared_mem_size = block_size * np.uint32(0).nbytes
    iteration = 0
    compute_start = time.time()  # 計算部分の開始時刻
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
    compute_end = time.time()  # 計算部分の終了時刻
    results["GPU Fixed-Point Iteration Loop"] = (compute_end - compute_start)*1000  # ms
    results["Iteration Count"] = iteration
    
    # GPU から結果取得（非同期）
    cuda.memcpy_dtoh_async(dom_host, dom_gpu, stream)
    stream.synchronize()
    
    # ホスト側 DFS 順序計算
    t2 = time.time()
    order = dfs_order(graph, root)
    t3 = time.time()
    results["Host DFS Order Calculation"] = (t3 - t2)*1000  # ms
    
    # ホスト側 即時支配者決定
    t4 = time.time()
    idom = compute_immediate_dominators(dom_host, order, root, n)
    t5 = time.time()
    results["Host Immediate Dominator Calculation"] = (t5 - t4)*1000  # ms
    
    # GPU 前処理結果の GrayCode の取得（非同期）
    cuda.memcpy_dtoh_async(gray_host, gray_gpu, stream)
    stream.synchronize()
    
    # 計算部分の終了後の時刻を保存（出力は含めない）
    total_compute_time = (time.time() - t0)*1000  # ms
    
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
    
    print("\n==== DFS Order ====")
    print(order)
    
    print("\n==== Immediate Dominators (idom) ====")
    for v in sorted(idom.keys()):
        print("  v = {:2d}: idom = {}".format(v, idom[v]))
    
    print("\n==== GrayCode Labels ====")
    for v in range(n):
        print("  v = {:2d}: gray = {}".format(v, gray_host[v]))

if __name__ == '__main__':
    main()
