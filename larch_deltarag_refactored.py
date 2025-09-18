# larch_deltarag_refactored.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Iterable
from collections import defaultdict, Counter
import argparse, json, math, random, sys

Coord = Tuple[int, int]

def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def clamp_nonneg_int(x: int) -> int:
    return int(max(0, x))

@dataclass(frozen=True)
class Cluster:
    scale: int
    center: Coord
    vertices: Tuple[Coord, ...]

@dataclass
class CoverScale:
    R: int
    centers: List[Coord] = field(default_factory=list)
    clusters: Dict[Coord, Cluster] = field(default_factory=dict)
    portals: Dict[frozenset, Coord] = field(default_factory=dict)

@dataclass
class Config:
    n: int = 32
    scales: Tuple[int, ...] = (3, 6, 10)
    urn_topk: int = 2000
    seed: int = 42
    add_diagonal_portal: bool = False
    max_scales: int = 8
    max_cluster_degree: int = 8
    max_incident_clusters_per_vertex: int = 8

@dataclass
class Stats:
    n: int
    scales: Tuple[int, ...]
    max_hubs_per_vertex: int
    avg_hubs_per_vertex: float
    avg_candidates: float
    p99_candidates: int
    exact_rate: float
    mean_stretch: float
    p99_stretch: float
    cache_hit_rate: float
    pairs: int

class UrnCache:
    def __init__(self, topk: int):
        self.topk = int(topk)
        self.counts: Counter = Counter()
        self.cache: Dict[Tuple[Coord, Coord], int] = {}

    @staticmethod
    def _key(u: Coord, v: Coord) -> Tuple[Coord, Coord]:
        return (u, v) if u <= v else (v, u)

    def get(self, u: Coord, v: Coord) -> Optional[int]:
        return self.cache.get(self._key(u, v))

    def observe_and_promote(self, u: Coord, v: Coord, value: int) -> None:
        k = self._key(u, v)
        self.counts[k] += 1
        if k in self.cache:
            return
        if len(self.cache) < self.topk:
            self.cache[k] = value
            return
        least = min(self.cache.keys(), key=lambda t: self.counts[t])
        if self.counts[k] > self.counts[least]:
            del self.cache[least]
            self.cache[k] = value

@dataclass
class Cluster:
    scale: int
    center: Coord
    vertices: Tuple[Coord, ...]

class CoverScale:
    def __init__(self, R: int, centers: List[Coord], clusters: Dict[Coord, Cluster]):
        self.R = R
        self.centers = centers
        self.clusters = clusters
        self.portals: Dict[frozenset, Coord] = {}

class CoverBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cover: List[CoverScale] = []

    def _grid_l1_ball(self, c: Coord, R: int) -> Tuple[Coord, ...]:
        n = self.cfg.n
        x0, y0 = c
        vs: List[Coord] = []
        for dx in range(-R, R + 1):
            rem = R - abs(dx)
            for dy in range(-rem, rem + 1):
                x, y = x0 + dx, y0 + dy
                if 0 <= x < n and 0 <= y < n:
                    vs.append((x, y))
        return tuple(vs)

    def _place_centers(self, R: int) -> List[Coord]:
        n = self.cfg.n
        step = max(2 * R, 2)
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

    def build(self) -> List[CoverScale]:
        assert len(self.cfg.scales) <= self.cfg.max_scales
        self.cover = []
        for R in self.cfg.scales:
            centers = self._place_centers(R)
            clusters = {c: Cluster(scale=R, center=c, vertices=self._grid_l1_ball(c, R)) for c in centers}
            self.cover.append(CoverScale(R=R, centers=centers, clusters=clusters))
        return self.cover

class PortalBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    @staticmethod
    def _intersection_portal(cA: Cluster, cB: Cluster) -> Optional[Coord]:
        inter = set(cA.vertices).intersection(cB.vertices)
        if not inter:
            return None
        mx = (cA.center[0] + cB.center[0]) // 2
        my = (cA.center[1] + cB.center[1]) // 2
        return min(inter, key=lambda p: (abs(p[0] - mx) + abs(p[1] - my), p[0], p[1]))

    def add_portals(self, cover: List[CoverScale]) -> None:
        for cs in cover:
            cs.portals = {}
            step = max(cs.R, 1)
            for (cx, cy) in cs.centers:
                for nx, ny in ((cx + step, cy), (cx - step, cy), (cx, cy + step), (cx, cy - step)):
                    if (nx, ny) in cs.clusters:
                        cA, cB = cs.clusters[(cx, cy)], cs.clusters[(nx, ny)]
                        p = self._intersection_portal(cA, cB)
                        if p is not None:
                            cs.portals[frozenset(((cx, cy), (nx, ny)))] = p
            if self.cfg.add_diagonal_portal:
                for (cx, cy) in cs.centers:
                    for nx, ny in ((cx + step, cy + step), (cx - step, cy - step),
                                   (cx + step, cy - step), (cx - step, cy + step)):
                        if (nx, ny) in cs.clusters:
                            cA, cB = cs.clusters[(cx, cy)], cs.clusters[(nx, ny)]
                            p = self._intersection_portal(cA, cB)
                            if p is not None:
                                cs.portals.setdefault(frozenset(((cx, cy), (nx, ny))), p)
                                break

class HubBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    @staticmethod
    def _invert(cover: List[CoverScale]) -> List[Dict[Coord, List[Cluster]]]:
        inv_per_scale: List[Dict[Coord, List[Cluster]]] = []
        for cs in cover:
            inv: Dict[Coord, List[Cluster]] = defaultdict(list)
            for c in cs.clusters.values():
                for v in c.vertices:
                    inv[v].append(c)
            inv_per_scale.append(inv)
        return inv_per_scale

    def build(self, cover: List[CoverScale]) -> Dict[Coord, Set[Coord]]:
        H: Dict[Coord, Set[Coord]] = defaultdict(set)
        inv_per_scale = self._invert(cover)
        for cs, inv in zip(cover, inv_per_scale):
            for v, clus in inv.items():
                if len(clus) > self.cfg.max_incident_clusters_per_vertex:
                    pass
                for clu in clus:
                    H[v].add(clu.center)
                    for key, portal in cs.portals.items():
                        if clu.center in key:
                            H[v].add(portal)
        return H

class QueryEngine:
    def __init__(self, cfg: Config, cover: List[CoverScale], H: Dict[Coord, Set[Coord]], urn: UrnCache):
        self.cfg = cfg
        self.cover = cover
        self.H = H
        self.urn = urn

    def _coarsest_fallback(self, u: Coord, v: Coord) -> Set[Coord]:
        cs = self.cover[-1]
        cu = next((c.center for c in cs.clusters.values() if u in c.vertices), None)
        cv = next((c.center for c in cs.clusters.values() if v in c.vertices), None)
        extra = set()
        if cu is not None:
            extra.add(cu)
        if cv is not None:
            extra.add(cv)
        return extra

    def query(self, u: Coord, v: Coord) -> Tuple[int, int, int, bool]:
        if u == v:
            return 0, 1, len(self.H[u]), False
        cached = self.urn.get(u, v)
        if cached is not None:
            return cached, 0, len(self.H[u]), True
        inter = self.H[u].intersection(self.H[v])
        if not inter:
            inter = self._coarsest_fallback(u, v)
        best = math.inf
        for h in inter:
            d = manhattan(u, h) + manhattan(h, v)
            if d < best:
                best = d
        best = 0 if not math.isfinite(best) else int(best)
        self.urn.observe_and_promote(u, v, best)
        return best, len(inter), len(self.H[u]), False

class Evaluator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        cover = CoverBuilder(cfg).build()
        PortalBuilder(cfg).add_portals(cover)
        H = HubBuilder(cfg).build(cover)
        urn = UrnCache(cfg.urn_topk)
        self.engine = QueryEngine(cfg, cover, H, urn)
        self.max_hubs = max(len(H[v]) for v in H)
        self.avg_hubs = sum(len(H[v]) for v in H) / len(H)

    def _pairs_uniform(self, num_pairs: int):
        n = self.cfg.n
        for _ in range(num_pairs):
            yield (self.rng.randrange(n), self.rng.randrange(n)), (self.rng.randrange(n), self.rng.randrange(n))

    def _pairs_skewed(self, total: int, hot_k: int, frac_hot: float):
        n = self.cfg.n
        verts = [(self.rng.randrange(n), self.rng.randrange(n)) for _ in range(2 * hot_k)]
        hot_pairs = [(verts[2*i], verts[2*i+1]) for i in range(hot_k)]
        for _ in range(total):
            if self.rng.random() < frac_hot:
                yield hot_pairs[self.rng.randrange(hot_k)]
            else:
                yield (self.rng.randrange(n), self.rng.randrange(n)), (self.rng.randrange(n), self.rng.randrange(n))

    def run_uniform(self, num_pairs: int) -> Stats:
        exact = 0
        total = 0
        stretches = []
        cand_sizes = []
        cache_hits = 0
        for u, v in self._pairs_uniform(num_pairs):
            est, k, _, hit = self.engine.query(u, v)
            truth = manhattan(u, v)
            exact += int(est == truth)
            total += 1
            if est != truth:
                stretches.append(est / max(truth, 1))
            cand_sizes.append(k)
            cache_hits += int(hit)
        cand_sorted = sorted(cand_sizes)
        p99_idx = int(0.99 * len(cand_sorted)) if cand_sorted else 0
        p99 = cand_sorted[p99_idx] if cand_sorted else 0
        mean_stretch = (sum(stretches) / len(stretches)) if stretches else 1.0
        p99_stretch = (sorted(stretches)[p99_idx] if stretches else 1.0)
        return Stats(
            n=self.cfg.n,
            scales=self.cfg.scales,
            max_hubs_per_vertex=self.max_hubs,
            avg_hubs_per_vertex=self.avg_hubs,
            avg_candidates=sum(cand_sizes) / len(cand_sizes) if cand_sizes else 0.0,
            p99_candidates=p99,
            exact_rate=exact / total if total else 1.0,
            mean_stretch=mean_stretch,
            p99_stretch=p99_stretch,
            cache_hit_rate=cache_hits / total if total else 0.0,
            pairs=total,
        )

    def run_skewed(self, total: int, hot_k: int, frac_hot: float):
        cand_total = 0
        hits = 0
        for u, v in self._pairs_skewed(total, hot_k, frac_hot):
            est, k, _, hit = self.engine.query(u, v)
            cand_total += k
            hits += int(hit)
        return {
            "n": self.cfg.n,
            "avg_candidates_overall": cand_total / max(total, 1),
            "cache_hit_rate_overall": hits / max(total, 1),
            "urn_cache_size": len(self.engine.urn.cache),
            "hot_k": hot_k,
            "frac_hot": frac_hot,
            "total": total,
        }

def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LARCH-δRAG refactored experimental harness (grids).")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--scales", type=int, nargs="+", default=[3, 6, 10])
    p.add_argument("--urn-topk", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--diagonal", action="store_true")
    p.add_argument("--pairs", type=int, default=8000)
    p.add_argument("--skewed", action="store_true")
    p.add_argument("--hot-k", type=int, default=200)
    p.add_argument("--frac-hot", type=float, default=0.85)
    p.add_argument("--total", type=int, default=20000)
    return p.parse_args(argv)

def main(argv: List[str]) -> int:
    args = _parse_args(argv)
    cfg = Config(
        n=clamp_nonneg_int(args.n),
        scales=tuple(sorted(set(int(x) for x in args.scales))),
        urn_topk=clamp_nonneg_int(args.urn_topk),
        seed=int(args.seed),
        add_diagonal_portal=bool(args.diagonal),
    )
    ev = Evaluator(cfg)
    stats = ev.run_uniform(num_pairs=int(args.pairs))
    out = {"uniform": stats.__dict__}
    if args.skewed:
        out["skewed"] = ev.run_skewed(total=int(args.total), hot_k=int(args.hot_k), frac_hot=float(args.frac_hot))
    print(json.dumps(out, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
