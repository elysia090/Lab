#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LARCH-δRAG (refined) — grid L1 harness with receipts and bitset hubs
- O(1) query via 2-hop hub factorization with lattice cover / Čech nerve portals
- O(0) via urn dictionary (hysteresis + TTL)
- Exact on 2D L1 grid with axis portals; optional diagonal portals
- Deterministic, auditable, reproducible

Python >= 3.10
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Set, Optional, Iterable
from collections import defaultdict, Counter
import argparse, json, math, random, sys, time, csv, hashlib

Coord = Tuple[int, int]

# ------------ basic L1 metric ------------

def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# ------------ config / receipts / stats ------------

@dataclass(frozen=True)
class Config:
    n: int = 32
    scales: Tuple[int, ...] = (3, 6, 10)              # radii R_ℓ
    add_diagonal_portal: bool = False                 # add (±2R,±2R) portals
    urn_topk: int = 2000
    urn_hyst_up: int = 5                              # promote threshold
    urn_hyst_down: int = 2                            # evict threshold
    urn_ttl: int = 200000                             # forget if not used
    seed: int = 42
    pairs: int = 8000
    skewed: bool = False
    hot_k: int = 200
    frac_hot: float = 0.85
    total: int = 20000
    crop: float = 1.0                                 # 0<crop<=1.0 sample only central crop*n
    audit_frac: float = 0.0                           # 0..1 dump receipts for this fraction
    zorder_layout: bool = True                        # improve cache locality
    export_jsonl: Optional[str] = None
    export_csv: Optional[str] = None
    runs: int = 1                                     # repeat uniform runs
    n_list: Optional[Tuple[int, ...]] = None          # override n with a list to sweep
    diag_portal_first: bool = False                   # diagonal portal priority
    # sanity bounds (debugging / invariants)
    max_scales: int = 8
    max_incident_clusters_per_vertex: int = 16

@dataclass
class Receipt:
    conf_sha256: str
    cover_sha256: str
    hubs_sha256: str

@dataclass
class Stats:
    n: int
    scales: Tuple[int, ...]
    kappa_design: int
    max_hubs_per_vertex: int
    avg_hubs_per_vertex: float
    avg_candidates: float
    p50_candidates: float
    p95_candidates: int
    p99_candidates: int
    exact_rate: float
    mean_stretch: float
    p99_stretch: float
    cache_hit_rate: float
    Δ_empirical_max: int
    κ_empirical_p99: int
    pairs: int
    elapsed_ms: float

# ------------ cover / clusters / portals ------------

@dataclass(frozen=True)
class Cluster:
    R: int
    center: Coord
    vertices: Tuple[Coord, ...]

@dataclass
class CoverScale:
    R: int
    centers: List[Coord]
    clusters: Dict[Coord, Cluster]
    # portals between centers, keyed by frozenset({cA, cB}) -> portal Coord
    portals: Dict[frozenset, Coord] = field(default_factory=dict)

class CoverBuilder:
    """
    Four-phase shifted lattice of L1 balls, stride 2R (correct).
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cover: List[CoverScale] = []

    def _grid_l1_ball(self, c: Coord, R: int, n: int) -> Tuple[Coord, ...]:
        x0, y0 = c
        vs: List[Coord] = []
        for dx in range(-R, R + 1):
            rem = R - abs(dx)
            x = x0 + dx
            if x < 0 or x >= n: 
                continue
            for dy in range(-rem, rem + 1):
                y = y0 + dy
                if 0 <= y < n:
                    vs.append((x, y))
        return tuple(vs)

    def _place_centers(self, R: int, n: int) -> List[Coord]:
        step = max(2 * R, 2)   # correct stride
        centers = set()
        for ox in (0, R):
            for oy in (0, R):
                x = ox
                while x < n:
                    y = oy
                    while y < n:
                        centers.add((x, y))
                        y += step
                    x += step
        return sorted(centers)

    def build(self, n: int) -> List[CoverScale]:
        assert len(self.cfg.scales) <= self.cfg.max_scales
        self.cover = []
        for R in self.cfg.scales:
            centers = self._place_centers(R, n)
            clusters = {c: Cluster(R=R, center=c, vertices=self._grid_l1_ball(c, R, n)) for c in centers}
            self.cover.append(CoverScale(R=R, centers=centers, clusters=clusters))
        return self.cover

class PortalBuilder:
    """
    Axis-neighbor centers at ±(2R,0),(0,2R); diagonals optional ±(2R,±2R).
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg

    @staticmethod
    def _intersection_portal(cA: Cluster, cB: Cluster) -> Optional[Coord]:
        inter = set(cA.vertices).intersection(cB.vertices)
        if not inter:
            return None
        mx = (cA.center[0] + cB.center[0]) // 2
        my = (cA.center[1] + cB.center[1]) // 2
        # pick L1-closest to midpoint; break ties lexicographically (deterministic)
        return min(inter, key=lambda p: (abs(p[0]-mx) + abs(p[1]-my), p[0], p[1]))

    def add_portals(self, cover: List[CoverScale]) -> None:
        for cs in cover:
            cs.portals = {}
            R = cs.R
            deltas_axis = [(2*R,0), (-2*R,0), (0,2*R), (0,-2*R)]
            deltas_diag = [(2*R,2*R), (-2*R,-2*R), (2*R,-2*R), (-2*R,2*R)] if self.cfg.add_diagonal_portal else []
            # deterministic order; optional diagonal priority control
            deltas = (deltas_diag + deltas_axis) if self.cfg.diag_portal_first else (deltas_axis + deltas_diag)
            centers_set = set(cs.centers)
            for (cx, cy) in cs.centers:
                for dx, dy in deltas:
                    nx, ny = (cx + dx, cy + dy)
                    if (nx, ny) not in centers_set:
                        continue
                    cA, cB = cs.clusters[(cx, cy)], cs.clusters[(nx, ny)]
                    p = self._intersection_portal(cA, cB)
                    if p is not None:
                        key = frozenset(((cx, cy), (nx, ny)))
                        cs.portals.setdefault(key, p)

# ------------ hubs: build + bitset encoding ------------

class HubBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    @staticmethod
    def _invert(cover: List[CoverScale], n: int) -> List[Dict[Coord, List[Cluster]]]:
        inv_per_scale: List[Dict[Coord, List[Cluster]]] = []
        for cs in cover:
            inv: Dict[Coord, List[Cluster]] = defaultdict(list)
            for c in cs.clusters.values():
                for v in c.vertices:
                    inv[v].append(c)
            inv_per_scale.append(inv)
        return inv_per_scale

    def build_sets(self, cover: List[CoverScale], n: int) -> Dict[Coord, Set[Coord]]:
        H: Dict[Coord, Set[Coord]] = defaultdict(set)
        inv_per_scale = self._invert(cover, n)
        for cs, inv in zip(cover, inv_per_scale):
            for v, clus in inv.items():
                if len(clus) > self.cfg.max_incident_clusters_per_vertex:
                    # still include; we only warn if desired
                    pass
                for clu in clus:
                    H[v].add(clu.center)
                    # portals on edges incident to this cluster
                    for key, portal in cs.portals.items():
                        if clu.center in key:
                            H[v].add(portal)
        return H

    @staticmethod
    def _zorder_key(p: Coord) -> int:
        # 16-bit interleave (for n<=65535) — enough here
        x, y = p
        def spread(k: int) -> int:
            k = (k | (k << 8)) & 0x00FF00FF
            k = (k | (k << 4)) & 0x0F0F0F0F
            k = (k | (k << 2)) & 0x33333333
            k = (k | (k << 1)) & 0x55555555
            return k
        return (spread(x) << 1) | spread(y)

    def encode_bitsets(self, H_sets: Dict[Coord, Set[Coord]], zorder: bool = True):
        # global hub id mapping
        hubs: List[Coord] = sorted({h for s in H_sets.values() for h in s},
                                   key=(self._zorder_key if zorder else lambda p: (p[0], p[1])))
        hub_to_idx: Dict[Coord, int] = {h:i for i,h in enumerate(hubs)}
        # per-vertex bitset
        bitset: Dict[Coord, int] = {}
        for v, s in H_sets.items():
            m = 0
            for h in s:
                m |= (1 << hub_to_idx[h])
            bitset[v] = m
        return hubs, hub_to_idx, bitset

# ------------ URN dictionary (O(0)) ------------

class UrnCache:
    """
    Pólya-like reinforcement with hysteresis + TTL.
    """
    def __init__(self, topk: int, up: int, down: int, ttl: int):
        self.topk = int(topk)
        self.up = max(1, int(up))
        self.down = max(0, int(down))
        self.ttl = int(ttl)
        self.counts: Counter = Counter()
        self.cache: Dict[Tuple[Coord, Coord], Tuple[int,int]] = {}  # key -> (value, last_seen_tick)
        self.clock = 0

    @staticmethod
    def _key(u: Coord, v: Coord) -> Tuple[Coord, Coord]:
        return (u, v) if u <= v else (v, u)

    def get(self, u: Coord, v: Coord) -> Optional[int]:
        self.clock += 1
        k = self._key(u, v)
        if k in self.cache:
            val, _ = self.cache[k]
            self.cache[k] = (val, self.clock)
            return val
        return None

    def observe_and_promote(self, u: Coord, v: Coord, value: int) -> None:
        self.clock += 1
        k = self._key(u, v)
        self.counts[k] += 1
        # TTL decay
        if self.ttl and (self.clock & 0x3FF) == 0:  # periodic sweep
            dead = [kk for kk,(val,last) in self.cache.items() if self.clock - last > self.ttl]
            for kk in dead:
                del self.cache[kk]
        # already cached
        if k in self.cache:
            return
        # promote only if count passes up-threshold, or free slots
        if len(self.cache) < self.topk or self.counts[k] >= self.up:
            # if full, evict the min-count entry whose count <= down
            if len(self.cache) >= self.topk:
                least = min(self.cache.keys(), key=lambda t: (self.counts[t], self.cache[t][1]))
                if self.counts[least] <= self.down:
                    del self.cache[least]
                else:
                    return
            self.cache[k] = (value, self.clock)

# ------------ query engine with receipts ------------

@dataclass
class Witness:
    u: Coord
    v: Coord
    best: int
    truth: int
    chosen_hub: Optional[Coord]
    candidates: int
    cand_hubs: List[Coord]

class QueryEngine:
    def __init__(self, cfg: Config, cover: List[CoverScale],
                 hubs: List[Coord], hub_to_idx: Dict[Coord,int],
                 bitset: Dict[Coord,int], urn: UrnCache):
        self.cfg = cfg
        self.cover = cover
        self.hubs = hubs
        self.hub_to_idx = hub_to_idx
        self.bitset = bitset
        self.urn = urn

    def _coarsest_fallback(self, u: Coord, v: Coord) -> List[int]:
        """
        Fallback hubs: coarsest centers of the clusters containing u and v.
        """
        cs = self.cover[-1]
        cu = next((c.center for c in cs.clusters.values() if u in c.vertices), None)
        cv = next((c.center for c in cs.clusters.values() if v in c.vertices), None)
        idxs: List[int] = []
        if cu is not None and cu in self.hub_to_idx:
            idxs.append(self.hub_to_idx[cu])
        if cv is not None and cv in self.hub_to_idx:
            idxs.append(self.hub_to_idx[cv])
        return idxs

    @staticmethod
    def _iter_bits(mask: int):
        while mask:
            lsb = mask & -mask
            idx = (lsb.bit_length() - 1)
            yield idx
            mask ^= lsb

    def query(self, u: Coord, v: Coord, audit: bool = False) -> Tuple[int, int, bool, Optional[Witness]]:
        if u == v:
            w = Witness(u, v, 0, 0, None, 1, [])
            return 0, 1, False, (w if audit else None)
        cached = self.urn.get(u, v)
        if cached is not None:
            if audit:
                w = Witness(u, v, cached, manhattan(u,v), None, 0, [])
                return cached, 0, True, w
            return cached, 0, True, None

        bs_u = self.bitset[u]
        bs_v = self.bitset[v]
        inter_mask = bs_u & bs_v
        cand_idx: List[int] = list(self._iter_bits(inter_mask))
        if not cand_idx:
            cand_idx = self._coarsest_fallback(u, v)

        best = math.inf
        chosen = None
        for i in cand_idx:
            h = self.hubs[i]
            d = manhattan(u, h) + manhattan(h, v)
            if d < best:
                best = d
                chosen = h
        best = 0 if not math.isfinite(best) else int(best)
        self.urn.observe_and_promote(u, v, best)
        if audit:
            w = Witness(u, v, best, manhattan(u, v), chosen, len(cand_idx), [self.hubs[i] for i in cand_idx][:32])
            return best, len(cand_idx), False, w
        return best, len(cand_idx), False, None

# ------------ evaluator / runners ------------

class Evaluator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self._build_all()

    def _build_all(self):
        n = self.cfg.n
        cover = CoverBuilder(self.cfg).build(n)
        PortalBuilder(self.cfg).add_portals(cover)
        H_sets = HubBuilder(self.cfg).build_sets(cover, n)
        hubs, hub_to_idx, bitset = HubBuilder(self.cfg).encode_bitsets(H_sets, zorder=self.cfg.zorder_layout)
        urn = UrnCache(self.cfg.urn_topk, self.cfg.urn_hyst_up, self.cfg.urn_hyst_down, self.cfg.urn_ttl)
        # receipts
        conf_sha = hashlib.sha256(json.dumps(asdict(self.cfg), sort_keys=True).encode()).hexdigest()
        cover_digest = hashlib.sha256()
        for cs in cover:
            cover_digest.update(str((cs.R, tuple(cs.centers))).encode())
            # only counts to keep stable
            cover_digest.update(str(len(cs.portals)).encode())
        hubs_sha = hashlib.sha256(str(len(hubs)).encode() + b"|" + str(sum(len(s) for s in H_sets.values())).encode()).hexdigest()
        self.receipt = Receipt(conf_sha, cover_digest.hexdigest(), hubs_sha)

        self.cover = cover
        self.H_sets = H_sets
        self.hubs = hubs
        self.hub_to_idx = hub_to_idx
        self.bitset = bitset
        self.urn = urn

        self.max_hubs = max(len(H_sets[v]) for v in H_sets)
        self.avg_hubs = sum(len(H_sets[v]) for v in H_sets) / len(H_sets)
        # Δ empirical max per scale
        self.delta_max = self._empirical_delta_max()

    def _empirical_delta_max(self) -> int:
        # maximum number of clusters covering a vertex across all scales
        invs = []
        for cs in self.cover:
            inv: Dict[Coord, int] = defaultdict(int)
            for clu in cs.clusters.values():
                for v in clu.vertices:
                    inv[v] += 1
            invs.append(max(inv.values() or [0]))
        return max(invs or [0])

    def _pairs_uniform(self, num_pairs: int) -> Iterable[Tuple[Coord,Coord]]:
        n = self.cfg.n
        # central crop
        crop = max(0.0, min(1.0, float(self.cfg.crop)))
        x0 = y0 = int((1.0 - crop) * n / 2)
        x1 = y1 = n - x0
        for _ in range(num_pairs):
            yield (self.rng.randrange(x0, x1), self.rng.randrange(y0, y1)), \
                  (self.rng.randrange(x0, x1), self.rng.randrange(y0, y1))

    def _pairs_skewed(self, total: int, hot_k: int, frac_hot: float):
        n = self.cfg.n
        crop = max(0.0, min(1.0, float(self.cfg.crop)))
        x0 = y0 = int((1.0 - crop) * n / 2)
        x1 = y1 = n - x0
        verts = [(self.rng.randrange(x0, x1), self.rng.randrange(y0, y1)) for _ in range(2 * hot_k)]
        hot_pairs = [(verts[2*i], verts[2*i+1]) for i in range(hot_k)]
        for _ in range(total):
            if self.rng.random() < frac_hot:
                yield hot_pairs[self.rng.randrange(hot_k)]
            else:
                yield (self.rng.randrange(x0, x1), self.rng.randrange(y0, y1)), \
                      (self.rng.randrange(x0, x1), self.rng.randrange(y0, y1))

    def run_uniform(self, num_pairs: int, audit_frac: float = 0.0,
                    jsonl: Optional[str] = None) -> Stats:
        engine = QueryEngine(self.cfg, self.cover, self.hubs, self.hub_to_idx, self.bitset, self.urn)
        t0 = time.perf_counter()
        exact = 0
        total = 0
        stretches: List[float] = []
        cand_sizes: List[int] = []
        cache_hits = 0
        κ_empirical: List[int] = []
        dump_audit = jsonl is not None and audit_frac > 0.0
        f = open(jsonl, "a", encoding="utf-8") if dump_audit else None

        for u, v in self._pairs_uniform(num_pairs):
            audit = (dump_audit and (self.rng.random() < audit_frac))
            est, k, hit, w = engine.query(u, v, audit=audit)
            truth = manhattan(u, v)
            exact += int(est == truth)
            total += 1
            if est != truth:
                stretches.append(est / max(truth, 1))
            cand_sizes.append(k)
            κ_empirical.append(k)
            cache_hits += int(hit)
            if w and f:
                record = {
                    "u": w.u, "v": w.v, "best": w.best, "truth": w.truth,
                    "chosen_hub": w.chosen_hub, "candidates": w.candidates,
                    "cand_hubs_sample": w.cand_hubs, "receipt": asdict(self.receipt)
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if f:
            f.close()

        cand_sorted = sorted(cand_sizes)
        def pctl(arr, p):
            if not arr: return 0
            idx = min(len(arr)-1, max(0, int(p * len(arr))))
            return arr[idx]
        p50 = pctl(cand_sorted, 0.50)
        p95 = pctl(cand_sorted, 0.95)
        p99 = pctl(cand_sorted, 0.99)

        mean_stretch = (sum(stretches) / len(stretches)) if stretches else 1.0
        p99_stretch = (sorted(stretches)[min(len(stretches)-1, int(0.99*len(stretches)))] if stretches else 1.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        # κ p99
        κ_p99 = sorted(κ_empirical)[min(len(κ_empirical)-1, int(0.99*len(κ_empirical)))] if κ_empirical else 0

        # design κ (upper bound) = L · Δ · (1 + d_port)
        L = len(self.cfg.scales)
        Δ = self.delta_max
        d_port = 4 + (1 if self.cfg.add_diagonal_portal else 0)
        kappa_design = L * Δ * (1 + d_port)

        return Stats(
            n=self.cfg.n,
            scales=self.cfg.scales,
            kappa_design=kappa_design,
            max_hubs_per_vertex=self.max_hubs,
            avg_hubs_per_vertex=self.avg_hubs,
            avg_candidates=sum(cand_sizes) / len(cand_sizes) if cand_sizes else 0.0,
            p50_candidates=p50,
            p95_candidates=p95,
            p99_candidates=p99,
            exact_rate=exact / total if total else 1.0,
            mean_stretch=mean_stretch,
            p99_stretch=p99_stretch,
            cache_hit_rate=cache_hits / total if total else 0.0,
            Δ_empirical_max=self.delta_max,
            κ_empirical_p99=κ_p99,
            pairs=total,
            elapsed_ms=elapsed_ms,
        )

    def run_skewed(self, total: int, hot_k: int, frac_hot: float) -> Dict[str, float|int]:
        engine = QueryEngine(self.cfg, self.cover, self.hubs, self.hub_to_idx, self.bitset, self.urn)
        cand_total = 0
        hits = 0
        for u, v in self._pairs_skewed(total, hot_k, frac_hot):
            est, k, hit, _ = engine.query(u, v, audit=False)
            cand_total += k
            hits += int(hit)
        return {
            "n": self.cfg.n,
            "avg_candidates_overall": cand_total / max(total, 1),
            "cache_hit_rate_overall": hits / max(total, 1),
            "urn_cache_size": len(engine.urn.cache),
            "hot_k": hot_k,
            "frac_hot": frac_hot,
            "total": total,
        }

# ------------ CLI / IO ------------

def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LARCH-δRAG refined harness (grids, receipts, bitsets).")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--n-list", type=int, nargs="+", help="sweep over multiple n; overrides --n")
    p.add_argument("--scales", type=int, nargs="+", default=[3,6,10])
    p.add_argument("--diagonal", action="store_true")
    p.add_argument("--diag-first", action="store_true")
    p.add_argument("--pairs", type=int, default=8000)
    p.add_argument("--skewed", action="store_true")
    p.add_argument("--hot-k", type=int, default=200)
    p.add_argument("--frac-hot", type=float, default=0.85)
    p.add_argument("--total", type=int, default=20000)
    p.add_argument("--urn-topk", type=int, default=2000)
    p.add_argument("--urn-up", type=int, default=5)
    p.add_argument("--urn-down", type=int, default=2)
    p.add_argument("--urn-ttl", type=int, default=200000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--crop", type=float, default=1.0)
    p.add_argument("--audit-frac", type=float, default=0.0)
    p.add_argument("--export-jsonl", type=str)
    p.add_argument("--export-csv", type=str)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--no-zorder", action="store_true")
    return p.parse_args(argv)

def _cfg_from_args(args: argparse.Namespace, n_override: Optional[int] = None) -> Config:
    return Config(
        n=int(n_override if n_override is not None else args.n),
        scales=tuple(sorted(set(int(x) for x in args.scales))),
        add_diagonal_portal=bool(args.diagonal),
        diag_portal_first=bool(args.diag_first),
        urn_topk=int(args.urn_topk),
        urn_hyst_up=int(args.urn_up),
        urn_hyst_down=int(args.urn_down),
        urn_ttl=int(args.urn_ttl),
        seed=int(args.seed),
        pairs=int(args.pairs),
        skewed=bool(args.skewed),
        hot_k=int(args.hot_k),
        frac_hot=float(args.frac_hot),
        total=int(args.total),
        crop=float(args.crop),
        audit_frac=float(args.audit_frac),
        zorder_layout=not bool(args.no_zorder),
        export_jsonl=args.export_jsonl,
        export_csv=args.export_csv,
        runs=int(args.runs),
        n_list=tuple(args.n_list) if args.n_list else None,
    )

def _write_csv(path: str, rows: List[Stats]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(asdict(rows[0]).keys()))
        for r in rows:
            w.writerow(list(asdict(r).values()))

def main(argv: List[str]) -> int:
    args = _parse_args(argv)
    cfg0 = _cfg_from_args(args)
    results: List[Stats] = []
    json_out = {}
    n_list = cfg0.n_list if cfg0.n_list else (cfg0.n,)
    for n in n_list:
        # repeated runs per n for stability
        for run in range(cfg0.runs):
            cfg = cfg0 if n == cfg0.n else _cfg_from_args(args, n_override=n)
            ev = Evaluator(cfg)
            stats = ev.run_uniform(num_pairs=cfg.pairs, audit_frac=cfg.audit_frac, jsonl=cfg.export_jsonl)
            json_out_key = f"n={cfg.n}/run={run+1}"
            json_out[json_out_key] = {"uniform": asdict(stats), "receipt": asdict(ev.receipt)}
            if cfg.skewed:
                json_out[json_out_key]["skewed"] = ev.run_skewed(total=cfg.total, hot_k=cfg.hot_k, frac_hot=cfg.frac_hot)
            results.append(stats)

    print(json.dumps(json_out, ensure_ascii=False, indent=2))
    if cfg0.export_csv:
        _write_csv(cfg0.export_csv, results)
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
