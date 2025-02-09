import numpy as np
from numba import cuda

def graph_to_edge_list(graph):
    """
    グラフ（隣接リスト形式、例: {頂点: [隣接頂点, ...] }）を
    (src, dst) のエッジリストに変換する。
    """
    src_list = []
    dst_list = []
    for v, neighbors in graph.items():
        for w in neighbors:
            src_list.append(v)
            dst_list.append(w)
    src = np.array(src_list, dtype=np.int32)
    dst = np.array(dst_list, dtype=np.int32)
    return src, dst

def prepare_graph_arrays_vectorized(graph):
    """
    グラフ（隣接リスト形式）から、前駆情報を
    1. フラットな前駆配列（pred_list_np）
    2. 各頂点ごとの開始位置（pred_offsets_np）
    として求める統合実装例。
    
    ・まずグラフをエッジリストに変換し，
      destination でソートすることで、各頂点の前駆が連続に並ぶようにする。
    ・np.searchsorted による排他的スキャンで各頂点のオフセットを算出する。
    ・最後に、結果を CUDA のピン留めメモリ（pagelocked memory）に配置する。
    
    戻り値: (pred_list_np, pred_offsets_np)
    """
    # 1. エッジリストに変換
    src, dst = graph_to_edge_list(graph)
    
    # グラフの頂点数（頂点番号が 0 から n-1 と仮定）
    n = len(graph)
    
    # 2. destination (w) によってソート（各頂点の前駆を連続に並べる）
    order = np.argsort(dst)
    sorted_src = src[order]   # 各エッジの「前駆」側の頂点番号
    sorted_dst = dst[order]   # ソート済みの宛先頂点
    
    # 3. 各頂点 v の開始オフセットを np.searchsorted で算出
    #    np.arange(n+1) により 0,1,...,n についての開始位置を求める
    pred_offsets = np.searchsorted(sorted_dst, np.arange(n+1))
    
    # 4. 結果を CUDA のピン留めメモリ領域に格納
    #    （これにより、GPUへの転送が効率的に行える）
    pred_list_np = cuda.pinned_array(len(sorted_src), dtype=np.int32)
    pred_list_np[:] = sorted_src
    pred_offsets_np = cuda.pinned_array(len(pred_offsets), dtype=np.int32)
    pred_offsets_np[:] = pred_offsets

    
    return pred_list_np, pred_offsets_np
