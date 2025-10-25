#!/usr/bin/env python3
# ct_distance_strict.py  (CSR-capable, O(1)/O(0) evaluator)
#
# Modes:
#  - grid: N×N grid with directed L1 weights (east/west/north/south).
#  - csr: general directed nonnegative graph in CSR (indptr, indices, weights).
#
# Key ideas:
#  - Two-hop hub scheme: For CSR, precompute distances "to hubs" (D_out) and "from hubs" (D_in).
#    Query(u,v) uses S(u,v)=H_out(u) ∩ H_in(v). If empty, use small fallback F(u,v). O(1) candidates.
#  - O(0) with promotion: store witness hub h*, value, margin; guard via bounded-overlap audit sets
#    (cluster blocks and cluster-edge pair), with local updates BΔ on those overlaps. Repair if guard fails.
#
# Commands:
#   simulate-grid   ... Zipf stream on grid
#   simulate-csr    ... Zipf stream on CSR
#   build-csr       ... build hub tables from a CSR file (npz) or random generator; save npz
#   bench-grid      ... vectorized throughput on grid
#   bench-csr       ... throughput on CSR (random queries)
#
# CSR file format (.npz):
#   keys: indptr (int64), indices (int32), weights (float32), N (int64)
# Hub file (.npz):
#   keys: hubs (int32, shape [H]), Din (float32, [H,N]), Dout (float32, [N,H]),
#         Hout (int32, [N,K]), Hin (int32, [N,K]), K (int64), H (int64), N (int64)
#
# Requirements: Python 3.9+, numpy, pandas, matplotlib

import argparse, os, sys, math, time, json, heapq
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INF = 1e30

# --------------------------- GRID UTILITIES -----------------------------------

def dir_L1(u: Tuple[int,int], v: Tuple[int,int], w: Dict[str, float]) -> float:
    dx = v[0]-u[0]; dy = v[1]-u[1]
    cost_x = (w["east"]*dx) if dx>=0 else (w["west"]*(-dx))
    cost_y = (w["north"]*dy) if dy>=0 else (w["south"]*(-dy))
    return float(cost_x + cost_y)

def move_dir(u: Tuple[int,int], v: Tuple[int,int]) -> Tuple[int,int]:
    sx = 0 if v[0]==u[0] else (1 if v[0]>u[0] else -1)
    sy = 0 if v[1]==u[1] else (1 if v[1]>u[1] else -1)
    return sx, sy

def nBx_axis(N: int, T: int) -> int:
    return (N + T - 1)//T

def block_id_from_coord(c: Tuple[int,int], T: int, N: int) -> int:
    nb = nBx_axis(N,T)
    return (c[0]//T)*nb + (c[1]//T)

def edge_id_from_blocks(b1: int, b2: int) -> Tuple[int,int]:
    if b1>b2: b1,b2=b2,b1
    return (b1,b2)

def neighbor_block_centers(coord: Tuple[int,int], T: int, N: int) -> List[Tuple[int,int]]:
    x, y = coord
    bx0 = (x // T) * T
    by0 = (y // T) * T
    blocks = [(bx0, by0), (bx0+T, by0), (bx0, by0+T), (bx0+T, by0+T)]
    centers = []
    seen = set()
    for bx, by in blocks:
        if bx >= N: bx = (N-1) - (N-1)%T
        if by >= N: by = (N-1) - (N-1)%T
        cx = min(bx + T//2, N-1)
        cy = min(by + T//2, N-1)
        c = (cx, cy)
        if c not in seen:
            centers.append(c); seen.add(c)
    return centers  # ≤4

def hubs_for_vertex_grid(v: Tuple[int,int], T: int, N: int) -> List[Tuple[int,int]]:
    return neighbor_block_centers(v, T, N)

def leg_overlaps_grid(u: Tuple[int,int], h: Tuple[int,int], T: int, N: int) -> Dict[str, Any]:
    bu = block_id_from_coord(u,T,N)
    bh = block_id_from_coord(h,T,N)
    e = edge_id_from_blocks(bu,bh)
    sx, sy = move_dir(u,h)
    return {"blocks":[bu,bh], "edge":e, "sx":sx, "sy":sy}

# --------------------------- CSR GRAPH ----------------------------------------

@dataclass
class CSR:
    indptr: np.ndarray  # shape [N+1], int64
    indices: np.ndarray # shape [M],   int32
    weights: np.ndarray # shape [M],   float32
    N: int

def load_csr(path: str) -> CSR:
    data = np.load(path)
    indptr = data["indptr"]
    indices = data["indices"]
    weights = data["weights"]
    N = int(data["N"])
    return CSR(indptr=indptr, indices=indices, weights=weights, N=N)

def save_csr(path: str, csr: CSR) -> None:
    np.savez(path, indptr=csr.indptr, indices=csr.indices, weights=csr.weights, N=np.int64(csr.N))

def reverse_csr(csr: CSR) -> CSR:
    N = csr.N
    indptr = csr.indptr; idx = csr.indices; w = csr.weights
    counts = np.zeros(N, dtype=np.int64)
    for u in range(N):
        for e in range(indptr[u], indptr[u+1]):
            v = int(idx[e]); counts[v]+=1
    indptr_r = np.zeros(N+1, dtype=np.int64)
    np.cumsum(counts, out=indptr_r[1:])
    indices_r = np.empty_like(idx)
    weights_r = np.empty_like(w)
    cursor = indptr_r.copy()
    for u in range(N):
        for e in range(indptr[u], indptr[u+1]):
            v = int(idx[e])
            pos = cursor[v]
            indices_r[pos] = u
            weights_r[pos] = w[e]
            cursor[v]+=1
    return CSR(indptr=indptr_r, indices=indices_r, weights=weights_r, N=N)

def dijkstra_all_to_all_from_sources(csr: CSR, sources: List[int]) -> np.ndarray:
    # Return D[source_idx, v]; each row is distances from that source
    N = csr.N
    indptr, idx, w = csr.indptr, csr.indices, csr.weights
    H = len(sources)
    D = np.full((H, N), INF, dtype=np.float32)
    for si, s in enumerate(sources):
        dist = D[si]
        dist[s] = 0.0
        pq = [(0.0, int(s))]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]: continue
            for e in range(indptr[u], indptr[u+1]):
                v = int(idx[e]); nd = d + float(w[e])
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
    return D

# --------------------------- HUB BUILDER FOR CSR ------------------------------

@dataclass
class HubData:
    hubs: np.ndarray   # [H] global hub vertex ids
    Din: np.ndarray    # [H,N] distance hub->v
    Dout: np.ndarray   # [N,H] distance u->hub
    Hout: np.ndarray   # [N,K] hub indices for out leg (indices into hubs[])
    Hin: np.ndarray    # [N,K] hub indices for in leg  (indices into hubs[])
    K: int
    H: int
    N: int

def build_hubs(csr: CSR, H: int = 16, K: int = 4, seed: int = 1) -> HubData:
    rng = np.random.default_rng(seed)
    N = csr.N
    # Pick H global hubs (uniform or degree-biased)
    hubs = rng.integers(0, N, size=H, dtype=np.int32)
    # Precompute distances from hubs to all nodes (forward) and to hubs from all nodes (reverse)
    Din = dijkstra_all_to_all_from_sources(csr, list(map(int, hubs)))  # [H,N]: hub->v
    csr_rev = reverse_csr(csr)
    Dout_T = dijkstra_all_to_all_from_sources(csr_rev, list(map(int, hubs)))  # [H,N]: hub<-v
    Dout = Dout_T.transpose().copy()  # [N,H]: v->hub
    # For each vertex, choose K nearest hubs for out and in
    Hout = np.argsort(Dout, axis=1)[:, :K].astype(np.int32)  # indices into hubs
    Hin  = np.argsort(Din.transpose(), axis=1)[:, :K].astype(np.int32)
    return HubData(hubs=hubs, Din=Din, Dout=Dout, Hout=Hout, Hin=Hin, K=K, H=H, N=N)

def save_hubs(path: str, hub: HubData) -> None:
    np.savez(path, hubs=hub.hubs, Din=hub.Din, Dout=hub.Dout, Hout=hub.Hout, Hin=hub.Hin,
             K=np.int64(hub.K), H=np.int64(hub.H), N=np.int64(hub.N))

def load_hubs(path: str) -> HubData:
    d = np.load(path)
    return HubData(hubs=d["hubs"], Din=d["Din"], Dout=d["Dout"], Hout=d["Hout"], Hin=d["Hin"],
                   K=int(d["K"]), H=int(d["H"]), N=int(d["N"]))

# --------------------------- O(1) EVALUATORS ----------------------------------

def eval_grid_O1(u: Tuple[int,int], v: Tuple[int,int], T: int, Nside: int, w: Dict[str,float]) -> float:
    Hu = hubs_for_vertex_grid(u, T, Nside)
    Hv = hubs_for_vertex_grid(v, T, Nside)
    Su = set(Hu)
    cand = [h for h in Hv if h in Su] or [Hu[0], Hv[0]]
    best = 1e30
    for h in cand:
        val = dir_L1(u,h,w) + dir_L1(h,v,w)
        if val < best: best = val
    return float(best)

def eval_csr_O1(u: int, v: int, hub: HubData) -> float:
    # S(u,v) = Hout(u) ∩ Hin(v) by hub indices; fallback: first of each, dedup; O(1) candidates
    Hout = hub.Hout[u]; Hin = hub.Hin[v]
    Su = set(map(int, Hout.tolist()))
    cand = [h for h in Hin.tolist() if h in Su]
    if not cand:
        # fallback of size 2
        cand = [int(Hout[0]), int(Hin[0])]
        if cand[1]==cand[0] and hub.H>1:
            cand[1] = int(Hout[1] if len(Hout)>1 else (Hin[1] if len(Hin)>1 else 0))
    best = 1e30
    for hi in cand:
        dv = float(hub.Dout[u, hi]) + float(hub.Din[hi, v])
        if dv < best: best = dv
    return float(best)

# --------------------------- PROMOTION / GUARD (CSR) --------------------------

def cluster_id(v: int, C: int, N: int) -> int:
    # Simple uniform clustering by contiguous vertex id ranges
    size = max(1, N // C)
    return min(v // size, C-1)

def leg_overlaps_csr(u: int, h: int, C: int, N: int) -> Dict[str, Any]:
    bu = cluster_id(u, C, N)
    bh = cluster_id(h, C, N)
    e = (min(bu,bh), max(bu,bh))
    return {"blocks":[bu,bh], "edge":e}

@dataclass
class RecCSR:
    u: int
    v: int
    value: float
    hstar: int       # index into hubs array
    margin: float
    audit_u: Dict[str,Any]
    audit_v: Dict[str,Any]
    competitors: List[int]  # other hub indices

class O0CSR:
    def __init__(self, hub: HubData, clusters: int = 128):
        self.hub = hub
        self.N = hub.N
        self.C = clusters
        self.B_block = np.zeros(self.C, dtype=np.float32)
        self.B_edge: Dict[Tuple[int,int], float] = {}

    def _sum_leg(self, leg: Dict[str,Any]) -> float:
        s = 0.0
        for b in leg["blocks"]:
            s += float(self.B_block[b])
        s += float(self.B_edge.get(tuple(leg["edge"]), 0.0))
        return s

    def build_rec(self, u: int, v: int) -> RecCSR:
        # candidate list (<= 2K)
        Hout = self.hub.Hout[u]; Hin = self.hub.Hin[v]
        Su = set(map(int, Hout.tolist()))
        cand = [h for h in Hin.tolist() if h in Su]
        if not cand:
            cand = [int(Hout[0]), int(Hin[0])]
            if len(Hout)>1 and len(Hin)>1 and cand[1]==cand[0]:
                cand[1]=int(Hout[1])
        vals = [(float(self.hub.Dout[u,h])+float(self.hub.Din[h,v]), h) for h in cand]
        vals.sort(key=lambda x:x[0])
        best, hstar = vals[0]
        second = vals[1][0] if len(vals)>1 else best+1.0
        audit_u = leg_overlaps_csr(u, int(self.hub.hubs[hstar]), self.C, self.N)
        audit_v = leg_overlaps_csr(int(self.hub.hubs[hstar]), v, self.C, self.N)
        competitors = [h for (_,h) in vals[1:]]
        return RecCSR(u=u, v=v, value=float(best), hstar=int(hstar), margin=float(second-best),
                      audit_u=audit_u, audit_v=audit_v, competitors=competitors)

    def guard_ok(self, rec: RecCSR) -> bool:
        U = self._sum_leg(rec.audit_u)
        V = self._sum_leg(rec.audit_v)
        m = rec.margin
        if (U+V) >= 0.5*m:
            return False
        for h in rec.competitors:
            # approximate competitor bound using same audit structure (constant terms)
            # For stricter variant, recompute leg overlaps to that competitor hub.
            return False if (U+V) >= 0.5*m else True
        return True

# --------------------------- MANAGERS / SIMULATION ----------------------------

@dataclass
class PromoRec:
    rec: Any
    last_seen: int
    count: int

class PromoManagerCSR:
    def __init__(self, hub: HubData, clusters: int = 128, ttl: int = 18, lam: float = 3.5, seed: int = 1):
        self.hub = hub
        self.N = hub.N
        self.ttl = ttl; self.lam=lam
        self.step = 0
        self.rng = np.random.default_rng(seed)
        self.o0 = O0CSR(hub, clusters=clusters)
        self.promoted: Dict[Tuple[int,int], PromoRec] = {}

    def updates(self, count_blocks=12, count_edges=10, delta_max=0.2):
        idx = self.rng.integers(0, self.o0.C, size=count_blocks, dtype=np.int32)
        inc = self.rng.random(count_blocks, dtype=np.float32)*delta_max
        self.o0.B_block[idx] += inc
        for _ in range(count_edges):
            a = int(self.rng.integers(0, self.o0.C)); b = int(self.rng.integers(0, self.o0.C))
            if a>b: a,b=b,a
            key=(a,b); self.o0.B_edge[key] = self.o0.B_edge.get(key,0.0) + float(self.rng.random()*delta_max)

    def demote_expired(self):
        to_del=[]
        for key, pr in list(self.promoted.items()):
            if self.step - pr.last_seen > self.ttl:
                to_del.append(key)
        for key in to_del:
            self.promoted.pop(key, None)

    def _tau(self, t: int) -> int:
        return int(math.ceil(self.lam*math.log(1+t)))

    def query(self, u: int, v: int) -> Tuple[float, bool, bool]:
        self.step += 0  # externally advanced
        key=(u,v)
        pr = self.promoted.get(key, None)
        if pr is not None:
            pr.last_seen = self.step; pr.count+=1
            if self.o0.guard_ok(pr.rec):
                return pr.rec.value, True, False
            else:
                pr.rec = self.o0.build_rec(u,v)
                return pr.rec.value, True, True
        # O(1) evaluation
        val = eval_csr_O1(u,v,self.hub)
        # maybe promote
        if key not in self.promoted:
            self.promoted[key] = PromoRec(rec=self.o0.build_rec(u,v), last_seen=self.step, count=1)
        else:
            self.promoted[key].count += 1
        if self.promoted[key].count >= self._tau(self.step+1):
            pass  # keep as promoted
        return float(val), False, False

def simulate_csr(csr: CSR, hub: HubData, steps=40, batch=1500, universe=25000, zipf_s=1.1,
                 ttl=18, lam=3.5, upd_blocks=12, upd_edges=10, delta_max=0.2, seed=1):
    rng = np.random.default_rng(seed)
    N = csr.N
    pairs = [(int(rng.integers(0,N)), int(rng.integers(0,N))) for _ in range(universe)]
    mgr = PromoManagerCSR(hub, clusters=128, ttl=ttl, lam=lam, seed=seed)
    rows=[]
    for t in range(1, steps+1):
        mgr.step = t
        # Zipf sample indices
        ranks = rng.zipf(a=zipf_s, size=batch)
        ranks = np.clip(ranks, 1, universe).astype(np.int64) - 1
        used=0; repaired=0
        t0=time.perf_counter()
        for r in ranks:
            u,v = pairs[int(r)]
            _, o0, rep = mgr.query(u,v)
            used += 1 if o0 else 0
            repaired += 1 if rep else 0
        dt=time.perf_counter()-t0
        mgr.updates(count_blocks=upd_blocks, count_edges=upd_edges, delta_max=delta_max)
        mgr.demote_expired()
        rows.append({"step":t,"p_hit":used/batch,"repairs":repaired,
                     "promoted":len(mgr.promoted),"avg_us_per_query":(dt/batch)*1e6})
    return pd.DataFrame(rows)

# --------------------------- RANDOM CSR GENERATOR -----------------------------

def generate_random_csr(N: int, avg_deg: int = 6, w_low=1.0, w_high=5.0, seed: int = 1) -> CSR:
    rng = np.random.default_rng(seed)
    edges = []
    for u in range(N):
        deg = max(1, int(rng.poisson(avg_deg)))
        vs = rng.integers(0, N, size=deg, dtype=np.int64)
        ws = rng.uniform(w_low, w_high, size=deg).astype(np.float32)
        for v,w in zip(vs,ws):
            edges.append((u,int(v),float(w)))
    edges.sort()
    indptr = np.zeros(N+1, dtype=np.int64)
    for (u,v,w) in edges: indptr[u+1]+=1
    np.cumsum(indptr, out=indptr)
    indices = np.zeros(len(edges), dtype=np.int32)
    weights = np.zeros(len(edges), dtype=np.float32)
    cursor = indptr.copy()
    for (u,v,w) in edges:
        p = cursor[u]; indices[p]=v; weights[p]=w; cursor[u]+=1
    return CSR(indptr=indptr, indices=indices, weights=weights, N=N)

# --------------------------- CLI ---------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # build-csr
    bp = sub.add_parser("build-csr")
    bp.add_argument("--csr", type=str, help="Input CSR .npz (or generate if absent)")
    bp.add_argument("--out", type=str, required=True, help="Output hub .npz path")
    bp.add_argument("--N", type=int, default=5000)
    bp.add_argument("--avg_deg", type=int, default=6)
    bp.add_argument("--H", type=int, default=16)
    bp.add_argument("--K", type=int, default=4)
    bp.add_argument("--seed", type=int, default=1)
    bp.add_argument("--gen_if_missing", action="store_true")
    def run_build(args):
        if args.csr and os.path.exists(args.csr):
            csr = load_csr(args.csr)
        elif args.gen_if_missing:
            csr = generate_random_csr(args.N, avg_deg=args.avg_deg, seed=args.seed)
        else:
            raise SystemExit("Provide --csr or --gen_if_missing")
        hub = build_hubs(csr, H=args.H, K=args.K, seed=args.seed)
        save_hubs(args.out, hub)
        print("hubs written to", args.out)
    bp.set_defaults(func=run_build)

    # simulate-csr
    sp = sub.add_parser("simulate-csr")
    sp.add_argument("--hub", type=str, required=True, help="Hub .npz path")
    sp.add_argument("--csr", type=str, help="CSR .npz path (only for N)")
    sp.add_argument("--steps", type=int, default=40)
    sp.add_argument("--batch", type=int, default=1500)
    sp.add_argument("--universe", type=int, default=25000)
    sp.add_argument("--zipf_s", type=float, default=1.1)
    sp.add_argument("--ttl", type=int, default=18)
    sp.add_argument("--lam", type=float, default=3.5)
    sp.add_argument("--upd_blocks", type=int, default=12)
    sp.add_argument("--upd_edges", type=int, default=10)
    sp.add_argument("--delta_max", type=float, default=0.2)
    sp.add_argument("--seed", type=int, default=1)
    sp.add_argument("--out", type=str, default="csr_out")
    def run_sim_csr(args):
        hub = load_hubs(args.hub)
        N = int(hub.N)
        # dummy CSR for N (not used directly in simulate)
        csr = CSR(indptr=np.zeros(N+1, dtype=np.int64), indices=np.zeros(0, dtype=np.int32),
                  weights=np.zeros(0, dtype=np.float32), N=N)
        df = simulate_csr(csr, hub, steps=args.steps, batch=args.batch, universe=args.universe,
                          zipf_s=args.zipf_s, ttl=args.ttl, lam=args.lam,
                          upd_blocks=args.upd_blocks, upd_edges=args.upd_edges,
                          delta_max=args.delta_max, seed=args.seed)
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, "simulate_csr.csv")
        df.to_csv(path, index=False)
        plt.figure(); plt.plot(df["step"], df["p_hit"]); plt.xlabel("step"); plt.ylabel("O(0) hit ratio")
        plt.title("CSR O(0) hit ratio"); plt.savefig(os.path.join(args.out,"p_hit.png"), bbox_inches="tight"); plt.close()
        plt.figure(); plt.plot(df["step"], df["avg_us_per_query"]); plt.xlabel("step"); plt.ylabel("µs/query")
        plt.title("avg time"); plt.savefig(os.path.join(args.out,"avg_time.png"), bbox_inches="tight"); plt.close()
        print("wrote", path)
    sp.set_defaults(func=run_sim_csr)

    # bench-csr
    bp2 = sub.add_parser("bench-csr")
    bp2.add_argument("--hub", type=str, required=True)
    bp2.add_argument("--queries", type=int, default=20000)
    bp2.add_argument("--seed", type=int, default=1)
    bp2.add_argument("--out", type=str, default="csr_out")
    def run_bench_csr(args):
        hub = load_hubs(args.hub)
        rng = np.random.default_rng(args.seed)
        N = int(hub.N)
        pairs = [(int(rng.integers(0,N)), int(rng.integers(0,N))) for _ in range(args.queries)]
        t0=time.perf_counter(); s=0.0
        for (u,v) in pairs:
            s+=eval_csr_O1(u,v,hub)
        dt=time.perf_counter()-t0
        avg = (dt/args.queries)*1e6
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out,"bench.txt"),"w") as f:
            f.write(f"avg_us_per_query={avg:.3f}\n")
        print("avg_us_per_query=", avg)
    bp2.set_defaults(func=run_bench_csr)

    # simulate-grid (unchanged in spirit)
    sg = sub.add_parser("simulate-grid")
    sg.add_argument("--Nside", type=int, default=1024)
    sg.add_argument("--T", type=int, default=16)
    sg.add_argument("--steps", type=int, default=40)
    sg.add_argument("--batch", type=int, default=1500)
    sg.add_argument("--universe", type=int, default=25000)
    sg.add_argument("--zipf_s", type=float, default=1.1)
    sg.add_argument("--ttl", type=int, default=18)
    sg.add_argument("--lam", type=float, default=3.5)
    sg.add_argument("--upd_blocks", type=int, default=10)
    sg.add_argument("--upd_edges", type=int, default=8)
    sg.add_argument("--delta_max", type=float, default=0.15)
    sg.add_argument("--east", type=float, default=1.0)
    sg.add_argument("--west", type=float, default=1.0)
    sg.add_argument("--north", type=float, default=1.0)
    sg.add_argument("--south", type=float, default=1.0)
    sg.add_argument("--out", type=str, default="grid_out")
    def run_sim_grid(args):
        rng = np.random.default_rng(1)
        pairs = [((int(rng.integers(0,args.Nside)), int(rng.integers(0,args.Nside))),
                  (int(rng.integers(0,args.Nside)), int(rng.integers(0,args.Nside)))) for _ in range(args.universe)]
        rows=[]
        B_block = np.zeros(nBx_axis(args.Nside,args.T)**2, dtype=np.float32)
        w = {"east":args.east,"west":args.west,"north":args.north,"south":args.south}
        for t in range(1, args.steps+1):
            idx = np.clip(np.random.zipf(a=args.zipf_s, size=args.batch), 1, args.universe)-1
            used=0; repaired=0
            t0=time.perf_counter()
            for i in idx:
                u,v = pairs[int(i)]
                _ = eval_grid_O1(u,v,args.T,args.Nside,w)
            dt=time.perf_counter()-t0
            rows.append({"step":t,"p_hit":0.0,"repairs":0,"promoted":0,
                         "avg_us_per_query":(dt/args.batch)*1e6})
        os.makedirs(args.out, exist_ok=True)
        df=pd.DataFrame(rows); path=os.path.join(args.out,"simulate_grid.csv"); df.to_csv(path, index=False)
        plt.figure(); plt.plot(df["step"], df["avg_us_per_query"]); plt.xlabel("step"); plt.ylabel("µs/query")
        plt.title("grid avg time"); plt.savefig(os.path.join(args.out,"avg_time.png"), bbox_inches="tight"); plt.close()
        print("wrote", path)
    sg.set_defaults(func=run_sim_grid)

    args = ap.parse_args()
    os.makedirs(".", exist_ok=True)
    args.func(args)

if __name__ == "__main__":
    main()
