#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

このコードは、各ブロック（GPUのブロック単位）ごとに対象頂点の dominator set を共有メモリにロードし、
局所的な固定点反復を実施してからグローバルメモリに書き戻すことで、
グローバル同期の頻度を下げ、メモリアクセスの高速化を図るものです。

dominator tree の計算自体が、コンパイラ最適化やプログラム解析の分野で古典的かつ難解な問題とされ、
従来は Lengauer–Tarjan アルゴリズムなどが使われていました。

これを並列化し、さらに固定点反復とビットベクトル（半格子の性質）による表現に置き換えることで、
問題の本質に対して新しいアプローチを提示しています。

※ 各 dominator set は、頂点数 n (n ≤ 32) を前提に 32bit 整数（ビットマスク）で表現しています。

"""

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# ====================================================
# 1. ホスト側の前処理（グラフ前駆リストの作成など）
# ====================================================
def compute_predecessors(graph):
    """グラフ（隣接リスト形式）から各頂点の前駆リストを作成する"""
    pred = {v: [] for v in graph}
    for v in graph:
        for w in graph[v]:
            pred[w].append(v)
    return pred

def prepare_graph_arrays(graph):
    """
    各頂点の前駆リストをフラットな配列と、各頂点の開始位置の配列に変換する。
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
    # numpy 配列に変換（int32 型）
    pred_list_np = np.array(pred_list, dtype=np.int32)
    pred_offsets_np = np.array(pred_offsets, dtype=np.int32)
    return pred_list_np, pred_offsets_np

# ====================================================
# 2. GPU 前処理カーネル：初期 dominator set と GrayCode の計算
# ====================================================
# 各頂点 v に対して：
#   - GrayCode[v] = v XOR (v >> 1)
#   - dominator set (dom[v]) = (1<<v) (if v == root) それ以外は全ビット1 (U)
init_kernel_code = r'''
extern "C"
__global__ void init_preproc(int n, int root, unsigned int U,
                             unsigned int *dom, int *gray)
{
    int v = blockDim.x * blockIdx.x + threadIdx.x;
    if (v >= n) return;
    // Gray code 計算: v XOR (v >> 1)
    gray[v] = v ^ (v >> 1);
    // dominator set 初期化：根は自分のみ、その他は全ビット1
    if (v == root)
        dom[v] = (1 << v);
    else
        dom[v] = U;
}
'''

mod_init = SourceModule(init_kernel_code, options=["--use_fast_math"])
init_preproc = mod_init.get_function("init_preproc")

# ====================================================
# 3. 共有メモリを用いた局所固定点反復カーネル
# ====================================================
kernel_code = r'''
extern "C"
__global__ void update_dom_shared(int n, unsigned int U, unsigned int *dom,
                                  const int *pred_offsets, const int *pred_list,
                                  int root, int *changed)
{
    // 各ブロックが担当する頂点の範囲
    int block_start = blockIdx.x * blockDim.x;
    int block_end = block_start + blockDim.x;
    if (block_end > n) block_end = n;
    int local_idx = threadIdx.x;
    int global_idx = block_start + local_idx;
    
    // 動的共有メモリ領域（各頂点の dominator set）
    extern __shared__ unsigned int s_dom[];
    
    // 担当する頂点があれば、グローバルメモリから共有メモリへロード
    if (global_idx < block_end)
        s_dom[local_idx] = dom[global_idx];
    __syncthreads();
    
    bool local_changed;
    for (int iter = 0; iter < 1000; iter++) { // 最大反復回数（安全策）
        local_changed = false;
        if (global_idx < block_end && global_idx != root) {
            unsigned int newDom = U;
            int start = pred_offsets[global_idx];
            int end = pred_offsets[global_idx + 1];
            for (int i = start; i < end; i++) {
                int u = pred_list[i];
                // もし u が同一ブロック内なら、共有メモリから読み出す
                if (u >= block_start && u < block_end) {
                    newDom &= s_dom[u - block_start];
                } else {
                    newDom &= dom[u];
                }
            }
            newDom |= (1 << global_idx);
            if (newDom != s_dom[local_idx]) {
                s_dom[local_idx] = newDom;
                local_changed = true;
            }
        }
        __syncthreads();
        // ブロック内の全スレッドで変更がなければ終了
        unsigned int ballot = __ballot_sync(0xFFFFFFFF, local_changed);
        if (ballot == 0)
            break;
        if (local_idx == 0 && ballot != 0)
            atomicExch(changed, 1);
        __syncthreads();
    }
    
    // 結果をグローバルメモリへ書き戻す
    if (global_idx < block_end)
        dom[global_idx] = s_dom[local_idx];
}
'''

mod = SourceModule(kernel_code, options=["--use_fast_math"])
update_dom_shared = mod.get_function("update_dom_shared")

# ====================================================
# 4. ホスト側 DFS と即時支配者決定（後処理）
# ====================================================
def dfs_order(graph, root):
    """根からの DFS 順序を計算して各頂点に順序番号を割り当てる"""
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
    DFS 順序が最大の頂点を即時支配者として選択する。
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
# 5. メイン処理
# ====================================================
def main():
    # --- グラフ定義（例: 5頂点） ---
    # 0: [1,2]
    # 1: [3]
    # 2: [3]
    # 3: [4]
    # 4: []
    graph = {
        0: [1, 2],
        1: [3],
        2: [3],
        3: [4],
        4: []
    }
    root = 0
    n = len(graph)
    
    # --- 前処理：グラフ配列の準備 ---
    pred_list_np, pred_offsets_np = prepare_graph_arrays(graph)
    
    # --- GPU 用初期化：dom と gray の並列前処理 ---
    # ここで U = (1<<n) - 1
    U = (1 << n) - 1
    # dom_host は一旦確保（サイズ n, dtype uint32）
    dom_host = np.empty(n, dtype=np.uint32)
    # gray 用配列（各頂点の GrayCode ラベル、int32）
    gray_host = np.empty(n, dtype=np.int32)
    
    # GPU メモリへ転送用バッファ
    dom_gpu = cuda.mem_alloc(dom_host.nbytes)
    gray_gpu = cuda.mem_alloc(gray_host.nbytes)
    # 初期化カーネルでセットするので、内容は初期化不要
    
    # --- 変更フラグ用バッファ ---
    changed_host = np.zeros(1, dtype=np.int32)
    changed_gpu = cuda.mem_alloc(changed_host.nbytes)
    
    # --- グラフ前駆情報の GPU 転送 ---
    pred_list_gpu = cuda.mem_alloc(pred_list_np.nbytes)
    cuda.memcpy_htod(pred_list_gpu, pred_list_np)
    pred_offsets_gpu = cuda.mem_alloc(pred_offsets_np.nbytes)
    cuda.memcpy_htod(pred_offsets_gpu, pred_offsets_np)
    
    # --- GPU 前処理カーネル呼び出し（初期 dominator set & GrayCode 計算） ---
    block_size_init = 128
    grid_size_init = (n + block_size_init - 1) // block_size_init
    init_preproc(np.int32(n), np.int32(root), np.uint32(U), 
                 dom_gpu, gray_gpu,
                 block=(block_size_init, 1, 1), grid=(grid_size_init, 1))
    # ※ 結果は後で dominator 固定点反復で使用
    
    # --- GPU 固定点反復ループ（共有メモリ版） ---
    block_size = 128
    grid_size = (n + block_size - 1) // block_size
    shared_mem_size = block_size * np.uint32(0).nbytes  # 各ブロックで block_size 個の uint32
    iteration = 0
    while True:
        iteration += 1
        changed_host[0] = 0
        cuda.memcpy_htod(changed_gpu, changed_host)
        
        update_dom_shared(np.int32(n), np.uint32(U), dom_gpu,
                          pred_offsets_gpu, pred_list_gpu,
                          np.int32(root), changed_gpu,
                          block=(block_size, 1, 1), grid=(grid_size, 1),
                          shared=shared_mem_size)
        
        cuda.memcpy_dtoh(changed_host, changed_gpu)
        if changed_host[0] == 0:
            break
    print("GPU 固定点反復終了（反復回数 =", iteration, ")")
    
    # --- GPU から結果の dominator set を取得 ---
    cuda.memcpy_dtoh(dom_host, dom_gpu)
    
    print("支配集合 (Dom):")
    for v in range(n):
        print("  v = {:2d}: {}".format(v, sorted(bitmask_to_set(dom_host[v]))))
    
    # --- ホスト側 DFS 順序の計算 ---
    order = dfs_order(graph, root)
    print("DFS 順序:", order)
    
    # --- 即時支配者の決定 ---
    idom = compute_immediate_dominators(dom_host, order, root, n)
    print("即時支配者 (idom):")
    for v in sorted(idom.keys()):
        print("  v = {:2d}: idom = {}".format(v, idom[v]))
    
    # --- （オプション）GPU 前処理結果の GrayCode を取得して表示 ---
    gray_result = np.empty_like(gray_host)
    cuda.memcpy_dtoh(gray_result, gray_gpu)
    print("GrayCode ラベル:")
    for v in range(n):
        print("  v = {:2d}: gray = {}".format(v, gray_result[v]))
    
if __name__ == '__main__':
    main()

