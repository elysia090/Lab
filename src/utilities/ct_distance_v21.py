# ct_distance_v21.py — O(1) distance with UB=LB certificate and audit (v2.1)
# Dependencies: numpy
import os, sys, numpy as np, heapq, argparse, time, json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

INF = 1e30

@dataclass
class CSR:
    indptr: np.ndarray  # int64, shape [N+1]
    indices: np.ndarray # int32, shape [M]
    weights: np.ndarray # float32, shape [M]
    N: int

def load_csr(path: str) -> 'CSR':
    data = np.load(path)
    return CSR(indptr=data['indptr'], indices=data['indices'], weights=data['weights'], N=int(data['N']))

def save_csr(path: str, csr: 'CSR') -> None:
    np.savez(path, indptr=csr.indptr, indices=csr.indices, weights=csr.weights, N=np.int64(csr.N))

def reverse_csr(csr: 'CSR') -> 'CSR':
    indptr, idx, w, N = csr.indptr, csr.indices, csr.weights, csr.N
    M = idx.shape[0]
    counts = np.zeros(N, dtype=np.int64)
    for u in range(N):
        counts[idx[indptr[u]:indptr[u+1]]] += 1
    indptr_r = np.zeros(N+1, dtype=np.int64)
    np.cumsum(counts, out=indptr_r[1:])
    indices_r = np.empty(M, dtype=np.int32)
    weights_r = np.empty(M, dtype=np.float32)
    cursor = indptr_r.copy()
    for u in range(N):
        for e in range(indptr[u], indptr[u+1]):
            v = int(idx[e]); pos = cursor[v]
            indices_r[pos] = u
            weights_r[pos] = w[e]
            cursor[v] += 1
    return CSR(indptr=indptr_r, indices=indices_r, weights=weights_r, N=N)

def dijkstra_from_source(csr: 'CSR', s: int) -> np.ndarray:
    N = csr.N; indptr, idx, w = csr.indptr, csr.indices, csr.weights
    dist = np.full(N, INF, dtype=np.float32); dist[s] = 0.0
    pq: List[Tuple[float,int]] = [(0.0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: 
            continue
        for e in range(indptr[u], indptr[u+1]):
            v = int(idx[e]); nd = d + float(w[e])
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist

def dijkstra_all_to_all_from_sources(csr: 'CSR', sources: List[int]) -> np.ndarray:
    H = len(sources); N = csr.N
    D = np.full((H, N), INF, dtype=np.float32)
    for si, s in enumerate(sources):
        D[si] = dijkstra_from_source(csr, int(s))
    return D

def quantize_rows_float32_to_int16(D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    H, N = D.shape
    scales = np.zeros(H, dtype=np.float32)
    q = np.zeros_like(D, dtype=np.int16)
    for i in range(H):
        finite = D[i] < INF
        m = float(np.max(D[i][finite])) if np.any(finite) else 1.0
        s = max(m / 32767.0, 1e-8)
        scales[i] = s
        row = D[i].copy()
        row[~finite] = 32767.0 * s
        q[i] = np.clip(np.rint(row / s), -32767, 32767).astype(np.int16)
    return q, scales

def dequantize_int16_rows(q: np.ndarray, scales: np.ndarray) -> np.ndarray:
    H, N = q.shape
    out = q.astype(np.float32)
    for i in range(H):
        out[i] = out[i] * scales[i]
    # leave sentinel as-is; caller may reinterpret near-sentinel as INF
    return out

@dataclass
class HubDataV21:
    hubs: np.ndarray      # int32 [H]
    Din: np.ndarray       # float32 [H,N] hub->v
    Dout: np.ndarray      # float32 [N,H] u->hub
    Hout: np.ndarray      # int32 [N,K] indices into hubs[]
    Hin: np.ndarray       # int32 [N,K]
    K: int
    H: int
    N: int
    landmarks: np.ndarray     # int32 [m]
    DL_fwd: np.ndarray        # float32 [m,N]
    DL_rev: np.ndarray        # float32 [m,N]
    qDin: np.ndarray = None
    qDin_scale: np.ndarray = None
    qDoutT: np.ndarray = None
    qDoutT_scale: np.ndarray = None

    def save(self, path: str) -> None:
        np.savez(path,
                 hubs=self.hubs, Din=self.Din, Dout=self.Dout, 
                 Hout=self.Hout, Hin=self.Hin, K=np.int64(self.K),
                 H=np.int64(self.H), N=np.int64(self.N),
                 landmarks=self.landmarks, DL_fwd=self.DL_fwd, DL_rev=self.DL_rev,
                 qDin=self.qDin if self.qDin is not None else np.array([], dtype=np.int16),
                 qDin_scale=self.qDin_scale if self.qDin_scale is not None else np.array([], dtype=np.float32),
                 qDoutT=self.qDoutT if self.qDoutT is not None else np.array([], dtype=np.int16),
                 qDoutT_scale=self.qDoutT_scale if self.qDoutT_scale is not None else np.array([], dtype=np.float32))

    @staticmethod
    def load(path: str) -> 'HubDataV21':
        d = np.load(path)
        return HubDataV21(
            hubs=d['hubs'], Din=d['Din'], Dout=d['Dout'],
            Hout=d['Hout'], Hin=d['Hin'], K=int(d['K']), H=int(d['H']), N=int(d['N']),
            landmarks=d['landmarks'], DL_fwd=d['DL_fwd'], DL_rev=d['DL_rev'],
            qDin=(d['qDin'] if 'qDin' in d and d['qDin'].size>0 else None),
            qDin_scale=(d['qDin_scale'] if 'qDin_scale' in d and d['qDin_scale'].size>0 else None),
            qDoutT=(d['qDoutT'] if 'qDoutT' in d and d['qDoutT'].size>0 else None),
            qDoutT_scale=(d['qDoutT_scale'] if 'qDoutT_scale' in d and d['qDoutT_scale'].size>0 else None),
        )

def select_hubs_uniform(N: int, H: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, N, size=H, dtype=np.int32)

def select_landmarks_farpoints(csr: 'CSR', m: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    N = csr.N
    L = np.empty(m, dtype=np.int32)
    L[0] = int(rng.integers(0, N))
    dist = dijkstra_from_source(csr, int(L[0]))
    for i in range(1, m):
        cand = int(np.argmax(dist))
        L[i] = cand
        dist2 = dijkstra_from_source(csr, cand)
        dist = np.minimum(dist, dist2)
    return L

def build_hubs_v21(csr: 'CSR', H: int = 16, K: int = 4, m_landmarks: int = 8, seed: int = 1,
                   quantize: bool = False) -> HubDataV21:
    N = csr.N
    hubs = select_hubs_uniform(N, H, seed)
    Din = dijkstra_all_to_all_from_sources(csr, list(map(int, hubs)))  # [H,N]
    csr_rev = reverse_csr(csr)
    Drev = dijkstra_all_to_all_from_sources(csr_rev, list(map(int, hubs)))  # [H,N] on reverse
    Dout = Drev.T.copy()  # [N,H]

    Hout = np.argsort(Dout, axis=1)[:, :K].astype(np.int32)  # [N,K]
    Hin  = np.argsort(Din.T, axis=1)[:, :K].astype(np.int32) # [N,K]

    L = select_landmarks_farpoints(csr, m=m_landmarks, seed=seed+7)
    DL_fwd = dijkstra_all_to_all_from_sources(csr, list(map(int, L)))      
    DL_rev = dijkstra_all_to_all_from_sources(csr_rev, list(map(int, L)))  

    hub = HubDataV21(hubs=hubs, Din=Din, Dout=Dout, Hout=Hout, Hin=Hin,
                     K=K, H=H, N=N, landmarks=L, DL_fwd=DL_fwd, DL_rev=DL_rev)

    if quantize:
        qDin, qDin_scale = quantize_rows_float32_to_int16(Din)
        qDoutT, qDoutT_scale = quantize_rows_float32_to_int16(Dout.T.copy())
        hub.qDin, hub.qDin_scale = qDin, qDin_scale
        hub.qDoutT, hub.qDoutT_scale = qDoutT, qDoutT_scale

    return hub

@dataclass
class QueryResult:
    ub: float
    lb: float
    exact: bool
    hub_idx: int
    k_scan: int
    margin: float

def compute_lb_from_landmarks(hub: HubDataV21, u: int, v: int) -> float:
    d1 = float(np.max(np.abs(hub.DL_fwd[:, u] - hub.DL_fwd[:, v])))
    d2 = float(np.max(np.abs(hub.DL_rev[:, u] - hub.DL_rev[:, v])))
    return max(d1, d2)

def query_v21(hub: HubDataV21, u: int, v: int) -> QueryResult:
    Hout_u = hub.Hout[u]; Hin_v  = hub.Hin[v]
    inmask = np.zeros(hub.H, dtype=np.int8); inmask[Hin_v] = 1
    best = INF; second = INF; best_h = -1; scanned = 0

    for hi in Hout_u:
        if inmask[hi]:
            d = float(hub.Dout[u, hi] + hub.Din[hi, v])
            scanned += 1
            if d < best:
                second = best; best = d; best_h = int(hi)
            elif d < second:
                second = d

    if best_h < 0:
        for hi in range(hub.H):
            d = float(hub.Dout[u, hi] + hub.Din[hi, v])
            scanned += 1
            if d < best:
                second = best; best = d; best_h = int(hi)
            elif d < second:
                second = d

    ub = best
    lb = compute_lb_from_landmarks(hub, u, v)
    exact = (abs(ub - lb) <= 1e-6)
    margin = float(second - best) if second < INF/2 else float('inf')
    return QueryResult(ub=ub, lb=lb, exact=exact, hub_idx=best_h, k_scan=scanned, margin=margin)

@dataclass
class AuditEntry:
    u: int
    v: int
    hub_idx: int
    ub: float
    lb: float
    margin: float

def promote_pair(audit_log: Dict[Tuple[int,int], AuditEntry], res: QueryResult, u: int, v: int) -> None:
    audit_log[(u,v)] = AuditEntry(u=u, v=v, hub_idx=res.hub_idx, ub=res.ub, lb=res.lb, margin=res.margin)

def audit_ok_after_update(audit_log: Dict[Tuple[int,int], AuditEntry], u: int, v: int,
                          max_leg_delta: float) -> bool:
    e = audit_log.get((u,v))
    if e is None: 
        return False
    return (2.0 * max_leg_delta) <= (e.margin * 0.5)

def cmd_build(args):
    csr = load_csr(args.csr)
    hub = build_hubs_v21(csr, H=args.H, K=args.K, m_landmarks=args.m, seed=args.seed, quantize=args.quantize)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    hub.save(args.out)
    print('wrote', args.out, 'N=', hub.N, 'H=', hub.H, 'K=', hub.K, 'm=', len(hub.landmarks), 'quant=', bool(hub.qDin is not None))

def cmd_bench(args):
    hub = HubDataV21.load(args.hubs)
    rng = np.random.default_rng(args.seed)
    Q = args.Q; N = hub.N
    audit_log: Dict[Tuple[int,int], AuditEntry] = {}
    exact_hits = 0; eq_hits = 0
    t0 = time.perf_counter()
    for i in range(Q):
        u = int(rng.integers(0, N)); v = int(rng.integers(0, N))
        res = query_v21(hub, u, v)
        if res.exact: exact_hits += 1
        if abs(res.ub - res.lb) <= 1e-6: eq_hits += 1
        if i < args.promote_top:
            promote_pair(audit_log, res, u, v)
    t1 = time.perf_counter()
    avg_us = (t1 - t0) / Q * 1e6
    print(json.dumps({'Q':Q, 'avg_us_per_query':avg_us, 'exact_ratio':exact_hits/Q, 'eq_ratio':eq_hits/Q, 
                      'promoted':len(audit_log), 'N':N, 'H':hub.H, 'K':hub.K}, indent=2))

def cmd_query(args):
    hub = HubDataV21.load(args.hubs)
    res = query_v21(hub, args.u, args.v)
    print(json.dumps({'u':args.u, 'v':args.v, 'ub':res.ub, 'lb':res.lb, 'exact':res.exact, 
                      'hub_idx':int(res.hub_idx), 'k_scan':res.k_scan, 'margin':res.margin}, indent=2))


def cmd_upgrade(args):
    # old hub file with keys: hubs, Din, Dout, Hout, Hin, K, H, N
    old = np.load(args.old)
    hubs = old['hubs']; Din = old['Din']; Dout = old['Dout']
    Hout = old['Hout']; Hin = old['Hin']
    K = int(old['K']); H = int(old['H']); N = int(old['N'])
    # need CSR to compute landmarks
    csr = load_csr(args.csr)
    assert N == csr.N, "CSR N mismatch"
    L = select_landmarks_farpoints(csr, m=args.m, seed=args.seed+7)
    DL_fwd = dijkstra_all_to_all_from_sources(csr, list(map(int, L)))
    DL_rev = dijkstra_all_to_all_from_sources(reverse_csr(csr), list(map(int, L)))
    hub = HubDataV21(hubs=hubs, Din=Din, Dout=Dout, Hout=Hout, Hin=Hin,
                     K=K, H=H, N=N, landmarks=L, DL_fwd=DL_fwd, DL_rev=DL_rev)
    if args.quantize:
        qDin, qDin_scale = quantize_rows_float32_to_int16(Din)
        qDoutT, qDoutT_scale = quantize_rows_float32_to_int16(Dout.T.copy())
        hub.qDin, hub.qDin_scale = qDin, qDin_scale
        hub.qDoutT, hub.qDoutT_scale = qDoutT, qDoutT_scale
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    hub.save(args.out)
    print('upgraded ->', args.out)


def build_argparser():
    ap = argparse.ArgumentParser(description='ct_distance_v21 — O(1) distance with UB=LB certificate and audit')
    sub = ap.add_subparsers(required=True)

    up = sub.add_parser('upgrade-from-old', help='attach landmarks/audit to old hub npz')
    up.add_argument('--old', required=True)
    up.add_argument('--csr', required=True)
    up.add_argument('--out', required=True)
    up.add_argument('--m', type=int, default=8)
    up.add_argument('--seed', type=int, default=1)
    up.add_argument('--quantize', action='store_true')
    up.set_defaults(func=cmd_upgrade)

    b = sub.add_parser('build-csr-hubs', help='build hub+landmark data from CSR graph')
    b.add_argument('--csr', required=True)
    b.add_argument('--out', required=True)
    b.add_argument('--H', type=int, default=16)
    b.add_argument('--K', type=int, default=4)
    b.add_argument('--m', type=int, default=8, help='landmarks')
    b.add_argument('--seed', type=int, default=1)
    b.add_argument('--quantize', action='store_true')
    b.set_defaults(func=cmd_build)

    bn = sub.add_parser('bench-csr', help='benchmark random queries on built hubs')
    bn.add_argument('--hubs', required=True)
    bn.add_argument('--Q', type=int, default=10000)
    bn.add_argument('--seed', type=int, default=1)
    bn.add_argument('--promote-top', type=int, default=0)
    bn.set_defaults(func=cmd_bench)

    pb = sub.add_parser('build-csr-portals', help='build portal (bounded-overlap) hubs')
    pb.add_argument('--csr', required=True)
    pb.add_argument('--out', required=True)
    pb.add_argument('--C', type=int, default=64, help='number of clusters')
    pb.add_argument('--pmax', type=int, default=4, help='portals per adjacent pair')
    pb.add_argument('--K', type=int, default=None)
    pb.add_argument('--m', type=int, default=8)
    pb.add_argument('--seed', type=int, default=1)
    pb.add_argument('--quantize', action='store_true')
    pb.set_defaults(func=cmd_build_portal)

    q = sub.add_parser('query', help='single query')
    q.add_argument('--hubs', required=True)
    q.add_argument('--u', type=int, required=True)
    q.add_argument('--v', type=int, required=True)
    q.set_defaults(func=cmd_query)

    return ap

if __name__ == '__main__':
    ap = build_argparser()
    args = ap.parse_args()
    args.func(args)

# --------------------------- PORTAL COVER BUILDER (bounded-overlap) -----------
@dataclass
class Cover:
    centers: np.ndarray        # int32 [C]
    assign: np.ndarray         # int32 [N], cluster id of each node
    neighbors: Dict[int, List[int]]  # cluster adjacency list (undirected)
    borders: Dict[Tuple[int,int], np.ndarray]  # pair -> np.int32 nodes near boundary (union)

def farthest_point_centers(csr: CSR, C: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    N = csr.N
    centers = np.empty(C, dtype=np.int32)
    centers[0] = int(rng.integers(0, N))
    Dmin = dijkstra_from_source(csr, int(centers[0]))
    for i in range(1, C):
        c = int(np.argmax(Dmin))
        centers[i] = c
        Dc = dijkstra_from_source(csr, c)
        Dmin = np.minimum(Dmin, Dc)
    return centers

def build_voronoi_cover(csr: CSR, C: int, seed: int) -> Tuple[Cover, np.ndarray]:
    # returns cover and distances from centers: Dcenters [C,N]
    centers = farthest_point_centers(csr, C, seed)
    Dcenters = dijkstra_all_to_all_from_sources(csr, list(map(int, centers)))  # [C,N]
    # assign each node to closest center (ties broken by lower index)
    assign = np.argmin(Dcenters, axis=0).astype(np.int32)  # [N]
    # cluster adjacency and border nodes
    neighbors: Dict[int, set] = {i:set() for i in range(C)}
    borders: Dict[Tuple[int,int], set] = {}
    indptr, idx = csr.indptr, csr.indices
    N = csr.N
    for u in range(N):
        cu = int(assign[u])
        for e in range(indptr[u], indptr[u+1]):
            v = int(idx[e]); cv = int(assign[v])
            if cu != cv:
                i, j = (cu, cv) if cu < cv else (cv, cu)
                neighbors[i].add(j); neighbors[j].add(i)
                key = (i,j)
                if key not in borders: borders[key] = set()
                borders[key].add(u); borders[key].add(v)
    borders_np: Dict[Tuple[int,int], np.ndarray] = {k: np.array(sorted(list(s)), dtype=np.int32) for k,s in borders.items()}
    neighbors_list: Dict[int, List[int]] = {i: sorted(list(s)) for i,s in neighbors.items()}
    return Cover(centers=centers, assign=assign, neighbors=neighbors_list, borders=borders_np), Dcenters

def select_portals_for_pair(pair: Tuple[int,int], border_nodes: np.ndarray, centers: np.ndarray, 
                            Dcenters: np.ndarray, p_max: int) -> np.ndarray:
    i, j = pair
    # score node by closeness to both centers: s = d(ci,x) + d(cj,x)
    di = Dcenters[i, border_nodes]
    dj = Dcenters[j, border_nodes]
    s = di + dj
    order = np.argsort(s, kind="mergesort")
    chosen = border_nodes[order[:min(p_max, border_nodes.shape[0])]]
    return chosen.astype(np.int32)

def build_portal_hubs_v21(csr: CSR, C: int = 64, p_max: int = 4, K: int = None, seed: int = 1,
                          m_landmarks: int = 8, quantize: bool = False) -> HubDataV21:
    """
    Steps:
      1) Voronoi cover with C centers (farthest-point).
      2) For each adjacent cluster pair (i,j), pick up to p_max portals from boundary by dual-center score.
      3) Global portal set P = union P_ij -> treated as hubs[].
      4) Precompute Din, Dout as usual from portals.
      5) For each node v in cluster c, label L(v) = union over neighbors[c] of P_{min(c,j),max(c,j)}.
         If |L(v)| > K, keep K nearest portals by Dout[v,*]. If |L(v)| < K, pad by nearest portals in P.
    """
    cover, Dcenters = build_voronoi_cover(csr, C=C, seed=seed)
    # select portals per pair
    pair_to_portals_nodes: Dict[Tuple[int,int], np.ndarray] = {}
    for pair, border_nodes in cover.borders.items():
        if border_nodes.size == 0:
            continue
        pair_to_portals_nodes[pair] = select_portals_for_pair(pair, border_nodes, cover.centers, Dcenters, p_max)
    # map portal node -> hub index
    if len(pair_to_portals_nodes) > 0:
        all_portal_nodes = np.unique(np.concatenate([v for v in pair_to_portals_nodes.values()]))
    else:
        all_portal_nodes = np.zeros(0, dtype=np.int32)
    hub_index_of_node: Dict[int,int] = {int(n): i for i, n in enumerate(all_portal_nodes)}
    H = all_portal_nodes.shape[0]
    # distances from portals
    Din = dijkstra_all_to_all_from_sources(csr, list(map(int, all_portal_nodes)))       # [H,N] hub->v
    csr_rev = reverse_csr(csr)
    Drev = dijkstra_all_to_all_from_sources(csr_rev, list(map(int, all_portal_nodes)))  # [H,N] on reverse
    Dout = Drev.T.copy()                                                                # [N,H]
    N = csr.N
    # choose K if not specified: max neighbor degree * p_max (empirical bound)
    if K is None:
        degs = [len(cover.neighbors.get(c, [])) for c in range(C)]
        K = max(1, int(max(degs + [1]) * p_max))
    K = int(K)
    # build labels (fixed length)
    Hout = np.full((N, K), -1, dtype=np.int32)
    Hin  = np.full((N, K), -1, dtype=np.int32)
    # pre-build per cluster union of portals
    cluster_portal_list: Dict[int, np.ndarray] = {}
    for c in range(C):
        ports = []
        for j in cover.neighbors.get(c, []):
            pair = (c, j) if c < j else (j, c)
            pj = pair_to_portals_nodes.get(pair, np.zeros(0, dtype=np.int32))
            for node in pj:
                ports.append(hub_index_of_node[int(node)])
        cluster_portal_list[c] = np.unique(np.array(ports, dtype=np.int32)) if len(ports)>0 else np.zeros(0, dtype=np.int32)
    # assign per node
    arangeH = np.arange(H, dtype=np.int32)
    for v in range(N):
        c = int(cover.assign[v])
        cand = cluster_portal_list[c]
        if cand.size == 0 and H>0:
            cand = arangeH
        # pick up to K nearest by Dout[v,*]
        if cand.size > K:
            order = np.argsort(Dout[v, cand])
            keep = cand[order[:K]]
        else:
            keep = cand
        # pad if fewer than K with nearest global portals not already present
        if keep.size < K and H>0:
            mask = np.ones(H, dtype=bool)
            mask[keep] = False
            add_order = np.argsort(Dout[v, mask])
            to_add = np.flatnonzero(mask)[add_order[:(K - keep.size)]]
            keep = np.concatenate([keep, to_add])
        Hout[v, :K] = keep
        Hin[v, :K]  = keep
    # landmarks
    L = select_landmarks_farpoints(csr, m=m_landmarks, seed=seed+11)
    DL_fwd = dijkstra_all_to_all_from_sources(csr, list(map(int, L)))
    DL_rev = dijkstra_all_to_all_from_sources(csr_rev, list(map(int, L)))
    hub = HubDataV21(hubs=all_portal_nodes.astype(np.int32), Din=Din, Dout=Dout, Hout=Hout, Hin=Hin,
                     K=K, H=H, N=N, landmarks=L, DL_fwd=DL_fwd, DL_rev=DL_rev)
    if quantize:
        qDin, qDin_scale = quantize_rows_float32_to_int16(Din)
        qDoutT, qDoutT_scale = quantize_rows_float32_to_int16(Dout.T.copy())
        hub.qDin, hub.qDin_scale = qDin, qDin_scale
        hub.qDoutT, hub.qDoutT_scale = qDoutT, qDoutT_scale
    return hub

def cmd_build_portal(args):
    csr = load_csr(args.csr)
    hub = build_portal_hubs_v21(csr, C=args.C, p_max=args.pmax, K=args.K, seed=args.seed, 
                                m_landmarks=args.m, quantize=args.quantize)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    hub.save(args.out)
    print(json.dumps({"out":args.out, "N":hub.N, "H":hub.H, "K":hub.K, "C":args.C, "p_max":args.pmax}, indent=2))
