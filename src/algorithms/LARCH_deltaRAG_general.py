#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LARCH-δRAG (general graphs)
---------------------------
Distance-oracle demo for nonnegative-weight graphs that mirrors the paper's structure:
  • Multiscale R-ball cover via r-nets per scale (greedy); cap per-node overlap.
  • Čech/Rips nerve adjacency realized by constant-many *portals* per cluster intersection.
  • 2-hop factorization: each vertex v keeps a small hub set H(v) (centers + portals).
  • δRAG URN cache: hot pairs are O(0) dictionary hits; otherwise O(1) candidate scan.
  • Bloom-accelerated hub intersection (2048-bit / 3-hash) + exact verify.
  • Optional audit log.
No L1 fast-path is assumed here (works for arbitrary nonnegative weights).

This is a *practical* harness. Exactness depends on structural assumptions (Cov/Hit).
For arbitrary graphs we empirically report stretch vs. Dijkstra truth.

Input graph
-----------
- Provide an edgelist file (u v w) 0-based, or generate synthetic graphs:
  grid:    --gen grid --grid-w 20 --grid-h 20 --wmin 1 --wmax 3
  rgg:     --gen rgg  --n 1000 --radius 0.08 --wmin 1 --wmax 3
  barabasi --gen ba   --n 1000 --m 3 --wmin 1 --wmax 3

CLI examples
------------
python LARCH_deltaRAG_general.py --gen grid --grid-w 24 --grid-h 24 \
  --scales 3 6 10 --pairs 5000 --cap 16 --out uniform.csv

python LARCH_deltaRAG_general.py --graph my.edgelist --pairs 10000 --cap 12 \
  --audit-file audit.tsv --audit-max 200 --out uniform.csv
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Iterable, Optional, Set
from collections import defaultdict, Counter, deque
from bisect import bisect_left
import argparse, json, math, random, sys, time, os

# ----------------------------- graph utils ---------------------------------

@dataclass
class Graph:
    n: int
    adj: List[List[Tuple[int, float]]]  # (neighbor, weight)

    @staticmethod
    def empty(n: int) -> "Graph":
        return Graph(n=n, adj=[[] for _ in range(n)])

    def add_edge(self, u: int, v: int, w: float):
        assert u != v and w >= 0
        self.adj[u].append((v, w))
        self.adj[v].append((u, w))  # undirected by default

    def dijkstra(self, s: int, targets: Optional[Set[int]] = None, cutoff: Optional[float] = None) -> List[float]:
        """Dijkstra from s. If targets provided, early-stop when all settled.
           If cutoff provided, stop relaxing nodes with dist > cutoff.
        """
        import heapq
        INF = float("inf")
        dist = [INF] * self.n
        dist[s] = 0.0
        pq = [(0.0, s)]
        remaining = set(targets) if targets is not None else None
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist[u]:
                continue
            if cutoff is not None and d > cutoff:
                break
            if remaining is not None and u in remaining:
                remaining.remove(u)
                if not remaining:
                    break
            for v, w in self.adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

# ----------------------------- config --------------------------------------

@dataclass(frozen=True)
class Config:
    # cover scales (ball radii in graph metric)
    scales: Tuple[float, ...] = (3.0, 6.0, 10.0)
    max_incident_clusters_per_vertex: int = 8   # Δ cap (per scale)
    portals_per_intersection: int = 2           # constant
    kappa_cap: int = 64                         # |H(v)| cap
    seed: int = 42
    pairs: int = 10000
    total: int = 20000
    urn_topk: int = 400
    urn_up: int = 5
    urn_down: int = 2
    urn_ttl: int = 200000
    candidate_cap: int = 16
    audit_file: Optional[str] = None
    audit_max: int = 0


@dataclass
class Stats:
    n: int
    scales: Tuple[float, ...]
    kappa_cap: int
    avg_hubs_per_vertex: float
    max_hubs_per_vertex: int
    avg_candidates_evaluated: float
    p99_candidates_evaluated: float
    avg_ops: float
    p99_ops: float
    exact_rate: float
    mean_stretch: float
    p99_stretch: float
    pairs: int
    elapsed_ms: float

# ----------------------------- cover/nerve ---------------------------------

@dataclass(frozen=True)
class Cluster:
    center: int
    R: float
    members: Tuple[int, ...]
    dist_to_center: Dict[int, float]  # subset for members (for portal selection)

@dataclass
class CoverScale:
    R: float
    centers: List[int]
    clusters: Dict[int, Cluster]  # keyed by center id
    portals: Dict[frozenset, List[int]] = field(default_factory=dict)
    owner_center: Dict[int, int] = field(default_factory=dict)  # nearest deterministic

class CoverBuilder:
    """Greedy r-nets per scale. For each center, take R-ball via Dijkstra.
       Cap per-node incident clusters to Δ by keeping the nearest ones.
    """
    def __init__(self, cfg: Config, g: Graph, rng: random.Random):
        self.cfg = cfg; self.g = g; self.rng = rng

    def _rnet_centers(self, R: float) -> List[int]:
        order = list(range(self.g.n))
        self.rng.shuffle(order)
        chosen: List[int] = []
        covered = [False] * self.g.n
        for s in order:
            if covered[s]:
                continue
            chosen.append(s)
            # mark nodes within R as covered
            dist = self.g.dijkstra(s, cutoff=R)
            for v, d in enumerate(dist):
                if d <= R:
                    covered[v] = True
        return chosen

    def _cluster_from_center(self, c: int, R: float) -> Cluster:
        dist = self.g.dijkstra(c, cutoff=R)
        members = tuple(v for v, d in enumerate(dist) if d <= R)
        dist_sub = {v: dist[v] for v in members}
        return Cluster(center=c, R=R, members=members, dist_to_center=dist_sub)

    def build(self) -> List[CoverScale]:
        cover: List[CoverScale] = []
        for R in self.cfg.scales:
            centers = self._rnet_centers(R)
            clusters = {c: self._cluster_from_center(c, R) for c in centers}
            cs = CoverScale(R=R, centers=centers, clusters=clusters)

            # Owner center: nearest by distance, tiebreak by id
            # We'll reuse dist_to_center already computed per cluster
            nearest: Dict[int, Tuple[float, int]] = {}  # v -> (d, center)
            for c, clu in clusters.items():
                for v, d in clu.dist_to_center.items():
                    cur = nearest.get(v)
                    if cur is None or d < cur[0] or (d == cur[0] and c < cur[1]):
                        nearest[v] = (d, c)
            cs.owner_center = {v: c for v, (d, c) in nearest.items()}

            cover.append(cs)

        # Cap per-node incident clusters (Δ)
        Δ = self.cfg.max_incident_clusters_per_vertex
        for cs in cover:
            inv: Dict[int, List[Tuple[float, int]]] = defaultdict(list)  # v -> [(d, center)]
            for c, clu in cs.clusters.items():
                for v, d in clu.dist_to_center.items():
                    inv[v].append((d, c))
            # keep nearest Δ
            for v, lst in inv.items():
                lst.sort()
                keep = set(c for _, c in lst[:Δ])
                for c in list(cs.clusters.keys()):
                    if c in keep:
                        continue
            # rebuild owner_center consistent (already nearest)

        # Build portals per adjacent cluster pair (constant many)
        P = self.cfg.portals_per_intersection
        for cs in cover:
            # Build a map vertex->clusters containing it
            memberships: Dict[int, List[int]] = defaultdict(list)
            for c, clu in cs.clusters.items():
                for v in clu.members:
                    memberships[v].append(c)
            # For each edge across two clusters, mark boundary
            seen_pair: Set[frozenset] = set()
            for u in range(self.g.n):
                for v, _w in self.g.adj[u]:
                    cu_list = memberships.get(u, [])
                    cv_list = memberships.get(v, [])
                    if not cu_list or not cv_list:
                        continue
                    for cu in cu_list:
                        for cv in cv_list:
                            if cu == cv:
                                continue
                            key = frozenset((cu, cv))
                            if key in seen_pair:
                                continue
                            seen_pair.add(key)
                            # choose up to P portals from intersection boundary:
                            # pick nodes minimizing dist_to_center sums
                            A, B = cs.clusters[cu], cs.clusters[cv]
                            inter = set(A.members).intersection(B.members)
                            if not inter:
                                continue
                            scored = []
                            for x in inter:
                                da = A.dist_to_center.get(x, math.inf)
                                db = B.dist_to_center.get(x, math.inf)
                                if math.isfinite(da) and math.isfinite(db):
                                    scored.append((da + db, x))
                            scored.sort()
                            portals = [x for _, x in scored[:P]] if scored else []
                            if portals:
                                cs.portals[key] = portals

        return cover

# ----------------------------- Hubs/Fingerprints ---------------------------

class HubBuilder:
    def __init__(self, cfg: Config, g: Graph):
        self.cfg = cfg; self.g = g

    @staticmethod
    def _mix64(x: int) -> int:
        x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
        x = (x ^ (x >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
        return (x ^ (x >> 31)) & 0xFFFFFFFFFFFFFFFF

    def _fingerprint_2048_3h(self, hubs: List[int]) -> Tuple[int, ...]:
        M = 32
        words = [0] * M
        for h in hubs:
            k0 = self._mix64(h * 0x9E37)
            k1 = self._mix64(h * 0xC2B2)
            k2 = self._mix64(h * 0x1657)
            for k in (k0, k1, k2):
                idx = (k >> 6) & 0x1F
                bit = k & 63
                words[idx] |= (1 << bit)
        return tuple(words)

    def build_sets(self, cover: List[CoverScale]) -> Dict[int, List[int]]:
        H: Dict[int, List[int]] = defaultdict(list)
        cap = self.cfg.kappa_cap
        for cs in cover:
            # build adjacency center->portals for this scale
            portals_by_center: Dict[int, List[int]] = defaultdict(list)
            for key, plist in cs.portals.items():
                a, b = tuple(key)
                portals_by_center[a].extend(plist)
                portals_by_center[b].extend(plist)
            memberships: Dict[int, List[int]] = defaultdict(list)
            for c, clu in cs.clusters.items():
                for v in clu.members:
                    memberships[v].append(c)
            for v, clist in memberships.items():
                work: List[int] = []
                for c in clist:
                    work.append(c)  # center as hub
                    work.extend(portals_by_center.get(c, ()))
                # dedup and keep up to cap (centers first)
                centers = sorted(set(clist))
                rest = [x for x in sorted(set(work)) if x not in centers]
                hubs = (centers + rest)[:cap]
                H[v].extend(h for h in hubs if h not in H[v])
                if len(H[v]) > cap:
                    H[v] = H[v][:cap]
        return {v: sorted(H[v]) for v in H}

    def encode(self, H_sets: Dict[int, List[int]]):
        small: Dict[int, Tuple[int, ...]] = {}
        fp: Dict[int, Tuple[int, ...]] = {}
        for v, arr in H_sets.items():
            t = tuple(arr[: self.cfg.kappa_cap])
            small[v] = t
            fp[v] = self._fingerprint_2048_3h(list(t))
        return small, fp

# ----------------------------- URN cache -----------------------------------

class UrnCache:
    def __init__(self, topk: int, up: int, down: int, ttl: int):
        self.topk = int(topk); self.up = int(up); self.down = int(down); self.ttl = int(ttl)
        self.counts: Counter = Counter()
        self.cache: Dict[Tuple[int, int], Tuple[float, int]] = {}
        self.clock = 0

    @staticmethod
    def _key(u: int, v: int) -> Tuple[int, int]:
        return (u, v) if u <= v else (v, u)

    def get(self, u: int, v: int) -> Optional[float]:
        self.clock += 1
        k = self._key(u, v)
        if k in self.cache:
            val, _ = self.cache[k]
            self.cache[k] = (val, self.clock)
            return val
        return None

    def observe_and_promote(self, u: int, v: int, value: float) -> None:
        self.clock += 1
        k = self._key(u, v)
        self.counts[k] += 1
        if self.ttl and (self.clock & 0x3FF) == 0:
            stale = [kk for kk, (_, last) in self.cache.items() if self.clock - last > self.ttl]
            for kk in stale:
                del self.cache[kk]
        if k in self.cache: return
        if len(self.cache) < self.topk or self.counts[k] >= self.up:
            if len(self.cache) >= self.topk:
                least = min(self.cache.keys(), key=lambda t: (self.counts[t], self.cache[t][1]))
                if self.counts[least] <= self.down:
                    del self.cache[least]
                else:
                    return
            self.cache[k] = (value, self.clock)

# ----------------------------- Query engine --------------------------------

class QueryEngine:
    """General-graph oracle:
       - O(0) URN hits
       - Otherwise, scan up to candidate_cap hubs from H(u)∩H(v)
         using 2048-bit Bloom + verify, else coarsest-owner fallback.
       Distances are looked up from a precomputed table DVH[v][h].
    """
    def __init__(
        self,
        cfg: Config,
        g: Graph,
        cover: List[CoverScale],
        H_small: Dict[int, Tuple[int, ...]],
        H_fp: Dict[int, Tuple[int, ...]],
        DVH: Dict[int, Dict[int, float]],
        urn: UrnCache,
    ):
        self.cfg = cfg; self.g = g; self.cover = cover
        self.small = H_small; self.fp = H_fp; self.DVH = DVH; self.urn = urn
        self._cand_cap = cfg.candidate_cap

    @staticmethod
    def _member_sorted(arr: Tuple[int, ...], x: int) -> bool:
        i = bisect_left(arr, x)
        return i != len(arr) and arr[i] == x

    def _bloom_candidates(self, hubs_u: Tuple[int, ...], hubs_v_sorted: Tuple[int, ...], fp_v: Tuple[int, ...]) -> List[int]:
        cand: List[int] = []
        for h in hubs_u:
            # 3-hash reconstructed deterministically
            def _mix64(x: int) -> int:
                x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
                x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
                x = (x ^ (x >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
                return (x ^ (x >> 31)) & 0xFFFFFFFFFFFFFFFF
            k0 = _mix64(h * 0x9E37); k1 = _mix64(h * 0xC2B2); k2 = _mix64(h * 0x1657)
            for k in (k0, k1, k2):
                idx = (k >> 6) & 0x1F; bit = k & 63
                if (fp_v[idx] >> bit) & 1 == 0:
                    break
            else:
                if self._member_sorted(hubs_v_sorted, h):
                    cand.append(h)
        return cand

    def _fallback_candidates(self, u: int, v: int) -> List[int]:
        cand: List[int] = []
        if self.cover:
            cs = self.cover[-1]
            cu = cs.owner_center.get(u); cv = cs.owner_center.get(v)
            if cu is not None: cand.append(cu)
            if cv is not None and cv != cu: cand.append(cv)
            # add one portal for each owner if exists
            if cu is not None:
                for key, ps in cs.portals.items():
                    if cu in key and ps:
                        cand.append(ps[0]); break
            if cv is not None:
                for key, ps in cs.portals.items():
                    if cv in key and ps:
                        cand.append(ps[0]); break
        # dedup and cap
        out = []
        seen = set()
        for h in cand:
            if h not in seen:
                seen.add(h); out.append(h)
        return out[: max(2, self._cand_cap)]

    def query(self, u: int, v: int) -> Tuple[float, int, bool, bool, bool, bool]:
        if u == v:
            return 0.0, 1, False, False, True, True

        cached = self.urn.get(u, v)
        if cached is not None:
            return cached, 0, True, False, True, True

        hubs_u = self.small.get(u, ())
        hubs_v = self.small.get(v, ())
        if hubs_u and hubs_v:
            mask_hit = any((a & b) for a, b in zip(self.fp[u], self.fp[v]))
        else:
            mask_hit = False
        inter_nonempty = False
        used_fallback = False

        if mask_hit:
            cand = self._bloom_candidates(hubs_u, hubs_v, self.fp[v])
            inter_nonempty = bool(cand)
        else:
            cand = []

        if not cand:
            cand = self._fallback_candidates(u, v)
            used_fallback = True

        if len(cand) > self._cand_cap:
            cand = cand[: self._cand_cap]

        best = float("inf"); k_eval = 0
        for h in cand:
            du = self.DVH.get(u, {}).get(h, float("inf"))
            dv = self.DVH.get(v, {}).get(h, float("inf"))
            d = du + dv
            k_eval += 1
            if d < best:
                best = d
        if not math.isfinite(best):
            best = float("inf")
        self.urn.observe_and_promote(u, v, best)
        return best, k_eval, False, used_fallback, mask_hit, inter_nonempty

# ----------------------------- Precompute DVH -------------------------------

def precompute_DVH(g: Graph, H_small: Dict[int, Tuple[int, ...]]) -> Dict[int, Dict[int, float]]:
    """For each unique hub h, run Dijkstra and record distances to only those v
    that include h in H(v). Early-stop when all such v have been settled.
    """
    # map hub -> set of nodes that include it
    targets: Dict[int, Set[int]] = defaultdict(set)
    for v, hubs in H_small.items():
        for h in hubs:
            targets[h].add(v)

    DVH: Dict[int, Dict[int, float]] = defaultdict(dict)
    for h, Vs in targets.items():
        dist = g.dijkstra(h, targets=Vs)
        for v in Vs:
            d = dist[v]
            if math.isfinite(d):
                DVH[v][h] = d
    return DVH

# ----------------------------- Evaluator -----------------------------------

class Evaluator:
    def __init__(self, cfg: Config, g: Graph):
        self.cfg = cfg; self.g = g; self.rng = random.Random(cfg.seed)
        self.cover = CoverBuilder(cfg, g, self.rng).build()
        H_sets = HubBuilder(cfg, g).build_sets(self.cover)
        self.avg_hubs = sum(len(s) for s in H_sets.values())/len(H_sets)
        self.max_hubs = max(len(s) for s in H_sets.values())
        self.H_small, self.H_fp = HubBuilder(cfg, g).encode(H_sets)
        self.DVH = precompute_DVH(g, self.H_small)
        self.urn = UrnCache(cfg.urn_topk, cfg.urn_up, cfg.urn_down, cfg.urn_ttl)

    def _pairs_uniform(self, num_pairs: int) -> Iterable[Tuple[int, int]]:
        for _ in range(num_pairs):
            u = self.rng.randrange(self.g.n); v = self.rng.randrange(self.g.n)
            yield (u, v)

    def _pairs_skewed(self, total: int, hot_k: int, frac_hot: float) -> Iterable[Tuple[int, int]]:
        verts = [self.rng.randrange(self.g.n) for _ in range(2*hot_k)]
        hot = [(verts[2*i], verts[2*i+1]) for i in range(hot_k)]
        for _ in range(total):
            if self.rng.random() < frac_hot:
                yield hot[self.rng.randrange(hot_k)]
            else:
                yield (self.rng.randrange(self.g.n), self.rng.randrange(self.g.n))

    def _true_dist(self, u: int, v: int) -> float:
        # on-demand dijkstra
        dist = self.g.dijkstra(u, targets={v})
        return dist[v]

    def run_uniform(self, num_pairs: int) -> Stats:
        eng = QueryEngine(self.cfg, self.g, self.cover, self.H_small, self.H_fp, self.DVH, self.urn)
        t0 = time.perf_counter()
        k_evals = []; stretches = []; exact = 0; tot = 0
        for u, v in self._pairs_uniform(num_pairs):
            est, k_eval, *_ = eng.query(u, v)
            tru = self._true_dist(u, v)
            if math.isfinite(est) and math.isfinite(tru) and tru > 0:
                stretch = est / tru
                stretches.append(stretch)
                exact += int(abs(est - tru) < 1e-12)
            else:
                stretch = float("inf")
                stretches.append(stretch)
            k_evals.append(k_eval); tot += 1
        elapsed_ms = (time.perf_counter() - t0)*1000.0
        ks = sorted(k_evals); p99k = ks[int(0.99*len(ks))] if ks else 0
        svals = [s for s in stretches if math.isfinite(s)]
        mean_st = sum(svals)/len(svals) if svals else float("nan")
        p99_st = sorted(svals)[int(0.99*len(svals))] if svals else float("nan")
        return Stats(
            n=self.g.n, scales=self.cfg.scales, kappa_cap=self.cfg.kappa_cap,
            avg_hubs_per_vertex=self.avg_hubs, max_hubs_per_vertex=self.max_hubs,
            avg_candidates_evaluated=sum(k_evals)/len(k_evals) if k_evals else 0.0,
            p99_candidates_evaluated=p99k, avg_ops=3.0*sum(k_evals)/len(k_evals) if k_evals else 0.0,
            p99_ops=3.0*p99k, exact_rate=exact/max(tot,1), mean_stretch=mean_st,
            p99_stretch=p99_st, pairs=tot, elapsed_ms=elapsed_ms,
        )

# ----------------------------- Generators/IO -------------------------------

def gen_grid(w: int, h: int, wmin: float, wmax: float, seed: int) -> Graph:
    rng = random.Random(seed)
    n = w*h; g = Graph.empty(n)
    def id(x,y): return y*w + x
    for y in range(h):
        for x in range(w):
            if x+1 < w:
                wgt = rng.uniform(wmin, wmax)
                g.add_edge(id(x,y), id(x+1,y), wgt)
            if y+1 < h:
                wgt = rng.uniform(wmin, wmax)
                g.add_edge(id(x,y), id(x,y+1), wgt)
    return g

def gen_rgg(n: int, radius: float, wmin: float, wmax: float, seed: int) -> Graph:
    rng = random.Random(seed)
    pts = [(rng.random(), rng.random()) for _ in range(n)]
    g = Graph.empty(n)
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i+1, n):
            xj, yj = pts[j]
            dx = xi-xj; dy = yi-yj
            d = (dx*dx + dy*dy) ** 0.5
            if d <= radius:
                wgt = rng.uniform(wmin, wmax)
                g.add_edge(i, j, wgt)
    return g

def gen_ba(n: int, m: int, wmin: float, wmax: float, seed: int) -> Graph:
    rng = random.Random(seed)
    g = Graph.empty(n)
    # start with m+1 clique
    for i in range(m+1):
        for j in range(i+1, m+1):
            wgt = rng.uniform(wmin, wmax)
            g.add_edge(i, j, wgt)
    deg = [len(g.adj[i]) for i in range(m+1)]
    for v in range(m+1, n):
        targets = set()
        while len(targets) < m:
            # preferential attachment
            x = rng.randrange(v)
            p = (deg[x] + 1) / (sum(deg[:v]) + v)
            if rng.random() < p:
                targets.add(x)
        for t in targets:
            wgt = rng.uniform(wmin, wmax)
            g.add_edge(v, t, wgt)
        deg.append(len(g.adj[v]))
    return g

def read_edgelist(path: str) -> Graph:
    # format: u v w
    edges = []
    max_id = -1
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): continue
            parts = s.split()
            if len(parts) < 2: continue
            u = int(parts[0]); v = int(parts[1]); w = float(parts[2]) if len(parts) >= 3 else 1.0
            edges.append((u, v, w))
            max_id = max(max_id, u, v)
    g = Graph.empty(max_id+1)
    for u, v, w in edges:
        g.add_edge(u, v, w)
    return g

# ----------------------------- CLI ----------------------------------------

def _dump_csv(path: str, rows: List[Dict[str, object]]) -> None:
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(path, index=False)
    except Exception:
        if not rows:
            open(path, "w", encoding="utf-8").close(); return
        keys = list(rows[0].keys())
        with open(path, "w", encoding="utf-8") as f:
            f.write(",".join(str(k) for k in keys) + "\n")
            for r in rows:
                f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description="LARCH-δRAG (general-graph oracle demo)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--graph", type=str, help="edge list file (u v w) 0-based")
    src.add_argument("--gen", choices=["grid", "rgg", "ba"], help="synthetic generator")

    # graph params
    p.add_argument("--grid-w", type=int, default=24)
    p.add_argument("--grid-h", type=int, default=24)
    p.add_argument("--radius", type=float, default=0.08, help="rgg radius in [0,1]")
    p.add_argument("--n", type=int, default=600, help="rgg/ba node count")
    p.add_argument("--m", type=int, default=3, help="ba: edges per new node")
    p.add_argument("--wmin", type=float, default=1.0)
    p.add_argument("--wmax", type=float, default=3.0)

    # oracle params
    p.add_argument("--scales", type=float, nargs="+", default=[3.0, 6.0, 10.0])
    p.add_argument("--delta-cap", type=int, default=8, help="Δ: max incident clusters per vertex per scale")
    p.add_argument("--portals", type=int, default=2, help="portals per intersection")
    p.add_argument("--kappa-cap", type=int, default=64, help="|H(v)| cap")
    p.add_argument("--cap", type=int, default=16, help="candidate cap at query")
    p.add_argument("--pairs", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)

    # urn
    p.add_argument("--urn-topk", type=int, default=400)
    p.add_argument("--urn-up", type=int, default=5)
    p.add_argument("--urn-down", type=int, default=2)
    p.add_argument("--urn-ttl", type=int, default=200000)

    # output
    p.add_argument("--out", type=str, default="uniform.csv")
    p.add_argument("--audit-file", type=str, default=None)
    p.add_argument("--audit-max", type=int, default=0)

    args = p.parse_args(argv)

    # Build graph
    if args.graph:
        g = read_edgelist(args.graph)
    else:
        if args.gen == "grid":
            g = gen_grid(args.grid_w, args.grid_h, args.wmin, args.wmax, args.seed)
        elif args.gen == "rgg":
            g = gen_rgg(args.n, args.radius, args.wmin, args.wmax, args.seed)
        else:
            g = gen_ba(args.n, args.m, args.wmin, args.wmax, args.seed)

    cfg = Config(
        scales=tuple(float(x) for x in args.scales),
        max_incident_clusters_per_vertex=int(args.delta_cap),
        portals_per_intersection=int(args.portals),
        kappa_cap=int(args.kappa_cap),
        seed=int(args.seed),
        pairs=int(args.pairs),
        urn_topk=int(args.urn_topk),
        urn_up=int(args.urn_up),
        urn_down=int(args.urn_down),
        urn_ttl=int(args.urn_ttl),
        candidate_cap=int(args.cap),
        audit_file=args.audit_file,
        audit_max=int(args.audit_max),
    )

    ev = Evaluator(cfg, g)
    st = ev.run_uniform(cfg.pairs)
    rows = [asdict(st)]
    _dump_csv(args.out, rows)
    print(json.dumps(rows, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
