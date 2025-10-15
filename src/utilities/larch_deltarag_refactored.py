"""Implementation of the LARCH-δRAG harness described in ``LARCH-δRAG.md``.

The goal of this module is to provide a compact, self-consistent reference for the
multiscale lattice-cover construction, hub/portal factorisation, and URN-based
query cache outlined in the specification.  Earlier revisions of this file mixed
multiple experimental branches together and duplicated several class definitions;
this version removes those inconsistencies and keeps the implementation closely
aligned with the prose description.

Only simple integer L1 grids are considered here.  The layout follows the
pseudocode in Section 4 of the document:

* ``preprocess`` constructs the multiscale cover, portals, and hub labels.
* ``query`` evaluates the two-hop distance estimator (with URN caching).
* ``fallback`` provides a constant-size rescue set when hub intersections are
  empty on a query.
* ``lb_ub`` exposes the shared-hub lower/upper bounds from the same candidate
  set.

The code is intentionally explicit to make auditing easy; every helper has a
single responsibility and the data structures are small enough to reason about
by inspection.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Iterator, List, Optional, Sequence, Tuple
from collections import Counter, defaultdict
import argparse
import json
import math
import random
import sys
import time

Coord = Tuple[int, int]


# ---------------------------------------------------------------------------
# Basic geometry
# ---------------------------------------------------------------------------

def manhattan(a: Coord, b: Coord) -> int:
    """Return the L1 distance between two lattice coordinates."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _midpoint(a: Coord, b: Coord) -> Coord:
    """Helper used when selecting portals; returns the integer midpoint."""
    return (a[0] + b[0]) // 2, (a[1] + b[1]) // 2


# ---------------------------------------------------------------------------
# Configuration and statistics containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    """Parameters controlling the lattice cover and evaluation harness."""

    n: int = 32
    scales: Tuple[int, ...] = (3, 6, 10)

    # Portal sampling parameters (axis midpoints + optional diagonal representative)
    portals_per_edge: int = 2
    add_diagonal_portal: bool = True

    # Hub label cap (for sanity checks; queries remain constant even when unused)
    kappa_cap: int = 64

    # URN cache parameters (distribution-aware O(0) path)
    urn_topk: int = 400
    urn_promote: int = 5
    urn_demote: int = 2
    urn_ttl: int = 200_000

    # Experiment knobs
    seed: int = 42
    pairs: int = 10_000
    total: int = 20_000
    hot_k: int = 200
    frac_hot: float = 0.85


@dataclass(frozen=True)
class QueryResult:
    distance: int
    candidates: int
    ops: int
    cache_hit: bool
    fallback_used: bool
    witness: Optional[Coord]


@dataclass(frozen=True)
class Stats:
    n: int
    scales: Tuple[int, ...]
    avg_hubs_per_vertex: float
    max_hubs_per_vertex: int
    avg_candidates_evaluated: float
    p99_candidates_evaluated: float
    exact_rate: float
    avg_ops: float
    p99_ops: float
    cache_hit_rate: float
    pairs: int
    elapsed_ms: float


# ---------------------------------------------------------------------------
# Cover and portals
# ---------------------------------------------------------------------------

@dataclass
class CoverScale:
    R: int
    centers: Tuple[Coord, ...]
    clusters: Dict[Coord, Tuple[Coord, ...]]
    owner_center: Dict[Coord, Coord]
    portals: Dict[frozenset, Tuple[Coord, ...]] = field(default_factory=dict)
    portals_by_center: Dict[Coord, Tuple[Coord, ...]] = field(default_factory=dict)


class CoverBuilder:
    """Construct four-phase shifted lattice covers with portal metadata."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def build(self) -> Tuple[CoverScale, ...]:
        cover: List[CoverScale] = []
        for R in self.cfg.scales:
            centers = self._place_centers(R)
            clusters: Dict[Coord, Tuple[Coord, ...]] = {}
            owner_center: Dict[Coord, Coord] = {}
            for c in centers:
                vertices = self._grid_l1_ball(c, R)
                clusters[c] = vertices
                for v in vertices:
                    prev = owner_center.get(v)
                    if prev is None:
                        owner_center[v] = c
                    else:
                        dv = manhattan(v, c)
                        dp = manhattan(v, prev)
                        if dv < dp or (dv == dp and c < prev):
                            owner_center[v] = c
            cs = CoverScale(R=R, centers=centers, clusters=clusters, owner_center=owner_center)
            self._populate_portals(cs)
            cover.append(cs)
        return tuple(cover)

    def _grid_l1_ball(self, c: Coord, R: int) -> Tuple[Coord, ...]:
        n = self.cfg.n
        x0, y0 = c
        vertices: List[Coord] = []
        for dx in range(-R, R + 1):
            x = x0 + dx
            if x < 0 or x >= n:
                continue
            rem = R - abs(dx)
            for dy in range(-rem, rem + 1):
                y = y0 + dy
                if 0 <= y < n:
                    vertices.append((x, y))
        return tuple(vertices)

    def _place_centers(self, R: int) -> Tuple[Coord, ...]:
        n = self.cfg.n
        step = max(2 * R, 2)
        centers: List[Coord] = []
        for ox in (0, R):
            for oy in (0, R):
                x = -R + ox
                while x <= (n - 1 + R):
                    y = -R + oy
                    while y <= (n - 1 + R):
                        centers.append((x, y))
                        y += step
                    x += step
        centers = sorted(set(centers))
        return tuple(centers)

    def _populate_portals(self, cs: CoverScale) -> None:
        portals: Dict[frozenset, Tuple[Coord, ...]] = {}
        portals_by_center: Dict[Coord, List[Coord]] = defaultdict(list)
        offsets: Sequence[Tuple[int, int]] = (
            (2 * cs.R, 0), (-2 * cs.R, 0), (0, 2 * cs.R), (0, -2 * cs.R)
        )
        diag_offsets: Sequence[Tuple[int, int]] = (
            (2 * cs.R, 2 * cs.R), (2 * cs.R, -2 * cs.R),
            (-2 * cs.R, 2 * cs.R), (-2 * cs.R, -2 * cs.R)
        ) if self.cfg.add_diagonal_portal else ()
        for center in cs.centers:
            for dx, dy in (*offsets, *diag_offsets):
                neighbour = (center[0] + dx, center[1] + dy)
                if neighbour not in cs.clusters:
                    continue
                key = frozenset((center, neighbour))
                if key in portals:
                    continue
                inter = self._intersection(cs.clusters[center], cs.clusters[neighbour])
                if not inter:
                    continue
                pts = self._select_portals(center, neighbour, inter)
                if not pts:
                    continue
                portals[key] = pts
                for c in key:
                    portals_by_center[c].extend(pts)
        cs.portals = portals
        cs.portals_by_center = {
            c: tuple(dict.fromkeys(pts)) for c, pts in portals_by_center.items()
        }

    @staticmethod
    def _intersection(a: Tuple[Coord, ...], b: Tuple[Coord, ...]) -> Tuple[Coord, ...]:
        aset = set(a)
        return tuple(sorted(p for p in b if p in aset))

    def _select_portals(
        self, c_a: Coord, c_b: Coord, inter: Tuple[Coord, ...]
    ) -> Tuple[Coord, ...]:
        if not inter:
            return tuple()
        mx, my = _midpoint(c_a, c_b)
        ranked = sorted(inter, key=lambda p: (manhattan(p, (mx, my)), p))
        selected: List[Coord] = list(ranked[: self.cfg.portals_per_edge])
        if self.cfg.add_diagonal_portal:
            diag = min(
                inter,
                key=lambda p: (
                    abs((p[0] - mx) - (p[1] - my)),
                    manhattan(p, (mx, my)),
                    p,
                ),
            )
            selected.append(diag)
        seen = set()
        ordered: List[Coord] = []
        for p in selected:
            if p not in seen:
                seen.add(p)
                ordered.append(p)
        return tuple(ordered)


# ---------------------------------------------------------------------------
# Hub construction
# ---------------------------------------------------------------------------

@dataclass
class Preprocessed:
    cfg: Config
    cover: Tuple[CoverScale, ...]
    hub_sets: Dict[Coord, Tuple[Coord, ...]]
    coarsest_owner: Dict[Coord, Coord]
    coarsest_portals: Dict[Coord, Tuple[Coord, ...]]
    avg_hubs: float
    max_hubs: int


class HubBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def build(self, cover: Tuple[CoverScale, ...]) -> Preprocessed:
        if not cover:
            raise ValueError("At least one scale is required to build hubs")
        hubs: Dict[Coord, set] = defaultdict(set)
        for cs in cover:
            for center, vertices in cs.clusters.items():
                portal_list = cs.portals_by_center.get(center, ())
                for v in vertices:
                    hv = hubs[v]
                    hv.add(center)
                    hv.update(portal_list)
        hub_sets: Dict[Coord, Tuple[Coord, ...]] = {
            v: tuple(sorted(h)) for v, h in hubs.items()
        }
        max_hubs = max((len(h) for h in hub_sets.values()), default=0)
        avg_hubs = (
            sum(len(h) for h in hub_sets.values()) / len(hub_sets)
            if hub_sets
            else 0.0
        )
        if max_hubs > self.cfg.kappa_cap:
            raise ValueError(
                f"Hub cap exceeded: observed {max_hubs} hubs while kappa_cap={self.cfg.kappa_cap}"
            )
        coarsest = cover[-1]
        coarsest_portals = {
            c: coarsest.portals_by_center.get(c, tuple())
            for c in coarsest.centers
        }
        return Preprocessed(
            cfg=self.cfg,
            cover=cover,
            hub_sets=hub_sets,
            coarsest_owner=coarsest.owner_center,
            coarsest_portals=coarsest_portals,
            avg_hubs=avg_hubs,
            max_hubs=max_hubs,
        )


# ---------------------------------------------------------------------------
# URN cache (distribution-aware O(0) path)
# ---------------------------------------------------------------------------

class UrnCache:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.topk = int(cfg.urn_topk)
        self.promote = int(cfg.urn_promote)
        self.demote = int(cfg.urn_demote)
        self.ttl = int(cfg.urn_ttl)
        self.tick = 0
        self.cache: Dict[Tuple[Coord, Coord], int] = {}
        self.counts: Counter = Counter()
        self.last_seen: Dict[Tuple[Coord, Coord], int] = {}

    @staticmethod
    def _key(u: Coord, v: Coord) -> Tuple[Coord, Coord]:
        return (u, v) if u <= v else (v, u)

    def step(self) -> None:
        self.tick += 1
        if self.ttl <= 0 or not self.cache:
            return
        stale: List[Tuple[Coord, Coord]] = [
            k for k, last in self.last_seen.items() if self.tick - last > self.ttl
        ]
        for k in stale:
            self.cache.pop(k, None)
            self.counts.pop(k, None)
            self.last_seen.pop(k, None)

    def get(self, u: Coord, v: Coord) -> Optional[int]:
        key = self._key(u, v)
        val = self.cache.get(key)
        if val is None:
            return None
        # Freshen timestamp to avoid TTL eviction while being used.
        self.last_seen[key] = self.tick
        return val

    def observe(self, u: Coord, v: Coord, value: int) -> None:
        key = self._key(u, v)
        self.counts[key] += 1
        self.last_seen[key] = self.tick
        if key in self.cache:
            if self.counts[key] <= self.demote:
                # Gentle demotion for recently cold entries.
                self.cache.pop(key, None)
            else:
                self.cache[key] = value
            return
        if self.counts[key] < self.promote or self.topk <= 0:
            return
        if len(self.cache) >= self.topk:
            victim = min(
                self.cache,
                key=lambda k: (self.counts[k], self.last_seen[k])
            )
            if self.counts[victim] > self.counts[key]:
                return
            self.cache.pop(victim, None)
            self.counts.pop(victim, None)
            self.last_seen.pop(victim, None)
        self.cache[key] = value


# ---------------------------------------------------------------------------
# Query logic (Preprocess / Query / Fallback / LB_UB)
# ---------------------------------------------------------------------------

class QueryEngine:
    def __init__(self, data: Preprocessed, urn: Optional[UrnCache] = None):
        self.data = data
        self.urn = urn or UrnCache(data.cfg)

    def preprocess(self) -> Preprocessed:
        return self.data

    def query(self, u: Coord, v: Coord) -> QueryResult:
        if u == v:
            return QueryResult(distance=0, candidates=1, ops=0, cache_hit=False, fallback_used=False, witness=u)
        self.urn.step()
        cached = self.urn.get(u, v)
        if cached is not None:
            return QueryResult(distance=cached, candidates=0, ops=0, cache_hit=True, fallback_used=False, witness=None)
        hubs_u = self.data.hub_sets.get(u, tuple())
        hubs_v = self.data.hub_sets.get(v, tuple())
        inter = self._intersection(tuple(hubs_u), hubs_v)
        fallback_used = False
        if not inter:
            inter = self._fallback(u, v)
            fallback_used = True
        best = math.inf
        witness: Optional[Coord] = None
        for h in inter:
            dist = manhattan(u, h) + manhattan(h, v)
            if dist < best:
                best = dist
                witness = h
        distance = 0 if not math.isfinite(best) else int(best)
        self.urn.observe(u, v, distance)
        candidates = len(inter)
        ops = 3 * candidates
        return QueryResult(
            distance=distance,
            candidates=candidates,
            ops=ops,
            cache_hit=False,
            fallback_used=fallback_used,
            witness=witness,
        )

    def lb_ub(self, u: Coord, v: Coord) -> Tuple[int, int, int]:
        if u == v:
            return 0, 0, 1
        self.urn.step()
        hubs_u = self.data.hub_sets.get(u, tuple())
        hubs_v = self.data.hub_sets.get(v, tuple())
        inter = self._intersection(tuple(hubs_u), hubs_v)
        if not inter:
            inter = self._fallback(u, v)
        lbs = [abs(manhattan(u, h) - manhattan(v, h)) for h in inter]
        ubs = [manhattan(u, h) + manhattan(h, v) for h in inter]
        lb = max(lbs) if lbs else 0
        ub = min(ubs) if ubs else 0
        return lb, ub, len(inter)

    def _fallback(self, u: Coord, v: Coord) -> Tuple[Coord, ...]:
        owners = self.data.coarsest_owner
        portals = self.data.coarsest_portals
        seeds: List[Coord] = []
        for vertex in (u, v):
            center = owners.get(vertex)
            if center is None:
                continue
            seeds.append(center)
            seeds.extend(portals.get(center, tuple()))
        ordered: List[Coord] = []
        seen = set()
        for s in seeds:
            if s not in seen:
                seen.add(s)
                ordered.append(s)
        return tuple(ordered)

    @staticmethod
    def _intersection(a: Sequence[Coord], b: Sequence[Coord]) -> Tuple[Coord, ...]:
        if not a or not b:
            return tuple()
        aset = set(a)
        return tuple(p for p in b if p in aset)


# ---------------------------------------------------------------------------
# Evaluator / CLI glue
# ---------------------------------------------------------------------------

class Evaluator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        cover = CoverBuilder(cfg).build()
        data = HubBuilder(cfg).build(cover)
        self.engine = QueryEngine(data)
        self.data = data

    def _uniform_pairs(self, num_pairs: int) -> Iterator[Tuple[Coord, Coord]]:
        n = self.cfg.n
        for _ in range(num_pairs):
            yield (
                (self.rng.randrange(n), self.rng.randrange(n)),
                (self.rng.randrange(n), self.rng.randrange(n)),
            )

    def _skewed_pairs(self, total: int, hot_k: int, frac_hot: float) -> Iterator[Tuple[Coord, Coord]]:
        n = self.cfg.n
        hot_vertices = [
            (self.rng.randrange(n), self.rng.randrange(n)) for _ in range(2 * hot_k)
        ]
        hot_pairs = [
            (hot_vertices[2 * i], hot_vertices[2 * i + 1]) for i in range(hot_k)
        ]
        for _ in range(total):
            if self.rng.random() < frac_hot and hot_pairs:
                yield hot_pairs[self.rng.randrange(len(hot_pairs))]
            else:
                yield (
                    (self.rng.randrange(n), self.rng.randrange(n)),
                    (self.rng.randrange(n), self.rng.randrange(n)),
                )

    def run_uniform(self, num_pairs: int) -> Stats:
        start = time.perf_counter()
        exact = 0
        cache_hits = 0
        candidates: List[int] = []
        ops_list: List[int] = []
        for (u, v) in self._uniform_pairs(num_pairs):
            res = self.engine.query(u, v)
            truth = manhattan(u, v)
            exact += int(res.distance == truth)
            cache_hits += int(res.cache_hit)
            candidates.append(res.candidates)
            ops_list.append(res.ops)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        cand_sorted = sorted(candidates)
        ops_sorted = sorted(ops_list)
        def percentile(sorted_list: List[int], pct: float) -> float:
            if not sorted_list:
                return 0.0
            idx = min(len(sorted_list) - 1, int(math.floor(pct * (len(sorted_list) - 1))))
            return float(sorted_list[idx])
        return Stats(
            n=self.cfg.n,
            scales=self.cfg.scales,
            avg_hubs_per_vertex=self.data.avg_hubs,
            max_hubs_per_vertex=self.data.max_hubs,
            avg_candidates_evaluated=sum(candidates) / len(candidates) if candidates else 0.0,
            p99_candidates_evaluated=percentile(cand_sorted, 0.99),
            exact_rate=exact / num_pairs if num_pairs else 1.0,
            avg_ops=sum(ops_list) / len(ops_list) if ops_list else 0.0,
            p99_ops=percentile(ops_sorted, 0.99),
            cache_hit_rate=cache_hits / num_pairs if num_pairs else 0.0,
            pairs=num_pairs,
            elapsed_ms=elapsed_ms,
        )

    def run_skewed(self, total: int, hot_k: int, frac_hot: float) -> Dict[str, float]:
        cand_total = 0
        hits = 0
        for u, v in self._skewed_pairs(total, hot_k, frac_hot):
            res = self.engine.query(u, v)
            cand_total += res.candidates
            hits += int(res.cache_hit)
        denom = max(total, 1)
        return {
            "n": float(self.cfg.n),
            "avg_candidates_overall": cand_total / denom,
            "cache_hit_rate_overall": hits / denom,
            "urn_cache_size": float(len(self.engine.urn.cache)),
            "hot_k": float(hot_k),
            "frac_hot": float(frac_hot),
            "total": float(total),
        }


def preprocess(cfg: Config) -> Preprocessed:
    cover = CoverBuilder(cfg).build()
    return HubBuilder(cfg).build(cover)


def query(data: Preprocessed, u: Coord, v: Coord) -> QueryResult:
    return QueryEngine(data).query(u, v)


def fallback(data: Preprocessed, u: Coord, v: Coord) -> Tuple[Coord, ...]:
    return QueryEngine(data)._fallback(u, v)


def lb_ub(data: Preprocessed, u: Coord, v: Coord) -> Tuple[int, int, int]:
    return QueryEngine(data).lb_ub(u, v)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LARCH-δRAG refactored experimental harness (grids).")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--scales", type=int, nargs="+", default=[3, 6, 10])
    p.add_argument("--urn-topk", type=int, default=400)
    p.add_argument("--urn-promote", type=int, default=5)
    p.add_argument("--urn-demote", type=int, default=2)
    p.add_argument("--urn-ttl", type=int, default=200000)
    p.add_argument("--pairs", type=int, default=10_000)
    p.add_argument("--skewed", action="store_true")
    p.add_argument("--hot-k", type=int, default=200)
    p.add_argument("--frac-hot", type=float, default=0.85)
    p.add_argument("--total", type=int, default=20_000)
    p.add_argument("--no-diagonal", action="store_true")
    p.add_argument("--portals-per-edge", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = _parse_args(argv)
    cfg = Config(
        n=max(1, int(args.n)),
        scales=tuple(sorted({int(s) for s in args.scales})),
        portals_per_edge=max(1, int(args.portals_per_edge)),
        add_diagonal_portal=not args.no_diagonal,
        urn_topk=max(0, int(args.urn_topk)),
        urn_promote=max(1, int(args.urn_promote)),
        urn_demote=max(0, int(args.urn_demote)),
        urn_ttl=max(0, int(args.urn_ttl)),
        seed=int(args.seed),
    )
    evaluator = Evaluator(cfg)
    stats = evaluator.run_uniform(int(args.pairs))
    output = {"uniform": asdict(stats)}
    if args.skewed:
        output["skewed"] = evaluator.run_skewed(
            total=int(args.total),
            hot_k=max(1, int(args.hot_k)),
            frac_hot=float(args.frac_hot),
        )
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
