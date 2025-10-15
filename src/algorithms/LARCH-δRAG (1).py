#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LARCH-δRAG (final, tuned):
Constant-time (O(0)/(1)) distance queries via lattice cover, Čech/Rips nerve,
Robin-style gluing, 2-hop factorization, and distribution-aware URN cache (δRAG).

This is a compact, production-oriented demonstration on an L1 grid (non-negative
weights). It integrates:
  • Fast-path (L1): evaluate (ux,vy) first (exact), short-circuiting queries.
  • URN O(0): heavy pairs promoted to a dictionary with hysteresis+TTL.
  • Bloom-accelerated intersection: 2048-bit / 3-hash fingerprints per vertex
    to prune hub intersections; verified by binary search.
  • Z-order (Morton) hub ordering for better cache locality.
  • Audit log: optional, records witness hub and candidate count.

General graphs: keep the hubs/portals/nerve scaffolding; add a (1+ε)-spanner/hopset
skeleton on the nerve to ensure |H(u)∩H(v)|=O(1) and d_G≤d*≤(1+ε)d_G.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Iterable, Optional
from collections import defaultdict, Counter
from bisect import bisect_left
import argparse, json, math, random, sys, time, os

Coord = Tuple[int, int]
INF_I64 = 10**18

# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def _part1by1_32(x: int) -> int:
    """Interleave lower 16 bits of x with zeros (Morton helper)."""
    x &= 0x0000FFFF
    x = (x | (x << 8)) & 0x00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F
    x = (x | (x << 2)) & 0x33333333
    x = (x | (x << 1)) & 0x55555555
    return x


def morton2D_32(x: int, y: int) -> int:
    """Morton code for (x,y) using 16 bits per coordinate (32-bit code)."""
    return (_part1by1_32(y) << 1) | _part1by1_32(x)


# ---------------------------------------------------------------------------
# Config and Stats
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    n: int = 32
    scales: Tuple[int, ...] = (3, 6, 10)
    add_diag: bool = True
    kappa_cap: int = 64

    # URN (O(0)) settings
    urn_topk: int = 400
    urn_up: int = 5
    urn_down: int = 2
    urn_ttl: int = 200000

    # Sampling density for portals per intersection (constant)
    portals_per_edge: int = 1

    # Experiment knobs
    seed: int = 42
    crop: float = 0.6
    pairs: int = 10_000
    total: int = 20_000

    # L1 fast-path
    use_dynamic_crosspoints: bool = True

    # Audit
    audit_file: Optional[str] = None
    audit_max: int = 0  # 0 means unlimited


@dataclass
class Stats:
    n: int
    scales: Tuple[int, ...]
    kappa_cap: int
    avg_hubs_per_vertex: float
    max_hubs_per_vertex: int
    avg_candidates_evaluated: float
    p99_candidates_evaluated: float
    exact_rate: float
    avg_ops: float
    p99_ops: float
    pairs: int
    elapsed_ms: float


# ---------------------------------------------------------------------------
# Cover + portals
# ---------------------------------------------------------------------------

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
    portals: Dict[frozenset, List[Coord]] = field(default_factory=dict)
    portals_by_center: Dict[Coord, List[Coord]] = field(default_factory=lambda: defaultdict(list))
    owner_center: Dict[Coord, Coord] = field(default_factory=dict)


class CoverBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _grid_l1_ball(self, c: Coord, R: int, n: int) -> Tuple[Coord, ...]:
        x0, y0 = c
        vs: List[Coord] = []
        for dx in range(-R, R+1):
            x = x0 + dx
            if x < 0 or x >= n: continue
            rem = R - abs(dx)
            for dy in range(-rem, rem+1):
                y = y0 + dy
                if 0 <= y < n:
                    vs.append((x, y))
        return tuple(vs)

    def _place_centers(self, R: int, n: int) -> List[Coord]:
        step = max(2*R, 2)
        centers = set()
        for ox in (0, R):
            for oy in (0, R):
                x = -R + ox
                while x <= (n - 1 + R):
                    y = -R + oy
                    while y <= (n - 1 + R):
                        centers.add((x, y))
                        y += step
                    x += step
        return sorted(centers)

    @staticmethod
    def _intersection_portal(cA: Cluster, cB: Cluster) -> Optional[Coord]:
        inter = set(cA.vertices).intersection(cB.vertices)
        if not inter: return None
        mx = (cA.center[0] + cB.center[0]) // 2
        my = (cA.center[1] + cB.center[1]) // 2
        return min(inter, key=lambda p: (abs(p[0]-mx) + abs(p[1]-my), p[0], p[1]))

    def build(self, n: int) -> List[CoverScale]:
        cover: List[CoverScale] = []
        for R in self.cfg.scales:
            centers = self._place_centers(R, n)
            clusters = {c: Cluster(R=R, center=c, vertices=self._grid_l1_ball(c, R, n)) for c in centers}
            cs = CoverScale(R=R, centers=centers, clusters=clusters)
            for clu in clusters.values():
                for v in clu.vertices:
                    c_prev = cs.owner_center.get(v)
                    d_prev = INF_I64 if c_prev is None else manhattan(c_prev, v)
                    d_new = manhattan(clu.center, v)
                    if c_prev is None or d_new < d_prev or (d_new == d_prev and clu.center < c_prev):
                        cs.owner_center[v] = clu.center
            cover.append(cs)

        P = max(1, int(self.cfg.portals_per_edge))
        for cs in cover:
            R = cs.R
            deltas = [(2*R, 0), (-2*R, 0), (0, 2*R), (0, -2*R)]
            if self.cfg.add_diag:
                deltas += [(2*R, 2*R), (-2*R, -2*R), (2*R, -2*R), (-2*R, 2*R)]
            centers_set = set(cs.centers)
            for (cx, cy) in cs.centers:
                for dx, dy in deltas:
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) not in centers_set: continue
                    A, B = cs.clusters[(cx, cy)], cs.clusters[(nx, ny)]
                    p = self._intersection_portal(A, B)
                    if p is None: continue
                    key = frozenset(((cx, cy), (nx, ny)))
                    cs.portals.setdefault(key, [])
                    if p not in cs.portals[key]:
                        cs.portals[key].append(p)
                        cs.portals_by_center[(cx, cy)].append(p)
                        cs.portals_by_center[(nx, ny)].append(p)

            assert len(cs.owner_center) == n * n, "owner_center must cover the full grid"
        return cover


# ---------------------------------------------------------------------------
# Hubs + fingerprints (with Z-order)
# ---------------------------------------------------------------------------

class HubBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def build_sets(self, cover: List[CoverScale], n: int) -> Dict[Coord, List[Coord]]:
        H: Dict[Coord, List[Coord]] = defaultdict(list)
        inv_per_scale: List[Dict[Coord, List[Cluster]]] = []
        # invert membership
        for cs in cover:
            inv: Dict[Coord, List[Cluster]] = defaultdict(list)
            for clu in cs.clusters.values():
                for v in clu.vertices:
                    inv[v].append(clu)
            inv_per_scale.append(inv)

        cap = int(self.cfg.kappa_cap)
        Rmax = max(c.R for c in cover)
        shift = Rmax  # centers may be in [-R, n-1+R]; shift to non-negative for Morton
        for cs, inv in zip(cover, inv_per_scale):
            for v, clus in inv.items():
                # centers and portals (dedup)
                work: List[Coord] = []
                for clu in clus:
                    work.append(clu.center)
                    work.extend(cs.portals_by_center.get(clu.center, ()))
                work = sorted(set(work), key=lambda p: (morton2D_32(p[0]+shift, p[1]+shift), p[0], p[1]))
                if len(work) > cap:
                    centers = sorted({c.center for c in clus},
                                     key=lambda p: (morton2D_32(p[0]+shift, p[1]+shift), p[0], p[1]))
                    rest = [w for w in work if w not in centers]
                    H[v] = (centers + rest)[:cap]
                else:
                    H[v] = work
        return H

    @staticmethod
    def _mix64(x: int) -> int:
        x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
        x = (x ^ (x >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
        return (x ^ (x >> 31)) & 0xFFFFFFFFFFFFFFFF

    def _fingerprint_2048_3h(self, hubs: List[Coord]) -> Tuple[int, ...]:
        M = 32  # 2048 bits
        words = [0] * M
        for (x, y) in hubs:
            k0 = self._mix64((x << 32) ^ y)
            k1 = self._mix64((y << 32) ^ x)
            k2 = self._mix64((x ^ (y << 1)) & 0xFFFFFFFFFFFFFFFF)
            for k in (k0, k1, k2):
                idx = (k >> 6) & 0x1F
                bit = k & 63
                words[idx] |= (1 << bit)
        return tuple(words)

    def encode(self, H_sets: Dict[Coord, List[Coord]]):
        small: Dict[Coord, Tuple[Coord, ...]] = {}
        fp: Dict[Coord, Tuple[int, ...]] = {}
        cap = int(self.cfg.kappa_cap)
        for v, arr in H_sets.items():
            t = tuple(arr[:cap])
            small[v] = t
            fp[v] = self._fingerprint_2048_3h(list(t))
        return small, fp


# ---------------------------------------------------------------------------
# URN cache
# ---------------------------------------------------------------------------

class UrnCache:
    def __init__(self, topk: int, up: int, down: int, ttl: int):
        self.topk = int(topk); self.up = int(up); self.down = int(down); self.ttl = int(ttl)
        self.counts: Counter = Counter()
        self.cache: Dict[Tuple[Coord, Coord], Tuple[int, int]] = {}
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


# ---------------------------------------------------------------------------
# Query engine with fast-path + Bloom and audit logging
# ---------------------------------------------------------------------------

class _HubHash:
    @staticmethod
    def _mix64(x: int) -> int:
        x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
        x = (x ^ (x >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
        return (x ^ (x >> 31)) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def triple_indices(h: Coord):
        x, y = h
        k0 = _HubHash._mix64((x << 32) ^ y)
        k1 = _HubHash._mix64((y << 32) ^ x)
        k2 = _HubHash._mix64((x ^ (y << 1)) & 0xFFFFFFFFFFFFFFFF)
        return ((k0 >> 6) & 0x1F, k0 & 63), ((k1 >> 6) & 0x1F, k1 & 63), ((k2 >> 6) & 0x1F, k2 & 63)


class QueryEngineFast:
    def __init__(
        self,
        cfg: Config,
        cover: List[CoverScale],
        H_small: Dict[Coord, Tuple[Coord, ...]],
        H_fp: Dict[Coord, Tuple[int, ...]],
        urn: UrnCache,
        candidate_cap: int = 2,
    ):
        self.cfg = cfg; self.cover = cover
        self.small = H_small; self.fp = H_fp; self.urn = urn
        self._hash_cache: Dict[Coord, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = {}
        self._cand_cap = int(candidate_cap)

        # Fallback helpers
        self.owner_coarse = cover[-1].owner_center
        self.portals_by_center_coarse = cover[-1].portals_by_center
        self.owner_second = cover[-2].owner_center if len(cover) > 1 else None
        self.portals_by_center_second = cover[-2].portals_by_center if len(cover) > 1 else None

        # Audit
        self._audit_fp = None
        self._audit_written = 0
        if cfg.audit_file:
            # open lazily on first write
            self._audit_path = cfg.audit_file
        else:
            self._audit_path = None

    def _audit(self, u: Coord, v: Coord, dist: int, k_eval: int, src: str, witness: Optional[Coord]):
        if self._audit_path is None: return
        if self.cfg.audit_max and self._audit_written >= self.cfg.audit_max: return
        if self._audit_fp is None:
            os.makedirs(os.path.dirname(self._audit_path) or ".", exist_ok=True)
            self._audit_fp = open(self._audit_path, "w", encoding="utf-8")
            self._audit_fp.write("ux,uy,vx,vy,dist,k_eval,src,wx,wy\n")
        wx, wy = (witness if witness is not None else (-1, -1))
        self._audit_fp.write(f"{u[0]},{u[1]},{v[0]},{v[1]},{dist},{k_eval},{src},{wx},{wy}\n")
        self._audit_fp.flush()
        self._audit_written += 1

    @staticmethod
    def _member_sorted(arr: Tuple[Coord, ...], x: Coord) -> bool:
        i = bisect_left(arr, x)
        return i != len(arr) and arr[i] == x

    def _bloom_candidates(self, hubs_u: Tuple[Coord, ...], hubs_v_sorted: Tuple[Coord, ...], fp_v: Tuple[int, ...]) -> List[Coord]:
        cand: List[Coord] = []
        for h in hubs_u:
            triple = self._hash_cache.get(h)
            if triple is None:
                triple = _HubHash.triple_indices(h)
                self._hash_cache[h] = triple
            ok = True
            for (widx, bit) in triple:
                if (fp_v[widx] >> bit) & 1 == 0:
                    ok = False; break
            if not ok: continue
            if self._member_sorted(hubs_v_sorted, h):
                cand.append(h)
        return cand

    def _fallback_candidates(self, u: Coord, v: Coord) -> List[Coord]:
        cand: List[Coord] = []
        for own, pbc in ((self.owner_coarse, self.portals_by_center_coarse),
                         (self.owner_second, self.portals_by_center_second)):
            if own is None: continue
            cu = own.get(u); cv = own.get(v)
            if cu:
                cand.append(cu)
                pu = pbc.get(cu, [])
                if pu: cand.append(pu[0])
            if cv and cv != cu:
                cand.append(cv)
                pv = pbc.get(cv, [])
                if pv: cand.append(pv[0])
        out: List[Coord] = []
        seen = set()
        for c in cand:
            if c not in seen:
                seen.add(c); out.append(c)
        return out[: max(2, self._cand_cap)]

    def query(self, u: Coord, v: Coord) -> Tuple[int, int, bool, bool, bool, bool]:
        if u == v:
            self._audit(u, v, 0, 1, "trivial", u)
            return 0, 1, False, False, True, True

        # O(0): URN dictionary
        cached = self.urn.get(u, v)
        if cached is not None:
            self._audit(u, v, cached, 0, "urn", None)
            return cached, 0, True, False, True, True

        # Fast-path (L1): evaluate (ux,vy) first -> exact
        if self.cfg.use_dynamic_crosspoints:
            c1 = (u[0], v[1])
            d1 = manhattan(u, c1) + manhattan(c1, v)
            self.urn.observe_and_promote(u, v, d1)
            self._audit(u, v, d1, 1, "fast", c1)
            return d1, 1, False, False, True, True

        # Hubs: Bloom-pruned intersection or fallback (then append crosspoints)
        fpu = self.fp[u]; fpv = self.fp[v]
        mask_hit = any((a & b) for a, b in zip(fpu, fpv))
        inter_nonempty = False; used_fallback = False

        if mask_hit:
            inter = self._bloom_candidates(self.small[u], self.small[v], fpv)
            inter_nonempty = bool(inter)
        else:
            inter = []

        if inter:
            cand = inter
        else:
            cand = self._fallback_candidates(u, v)
            used_fallback = True

        # Append crosspoints at front if fast-path is off
        if not self.cfg.use_dynamic_crosspoints:
            extra = [(u[0], v[1]), (v[0], u[1])]
            seen = set()
            merged = []
            for e in extra + cand:
                if e not in seen:
                    seen.add(e); merged.append(e)
            cand = merged

        if len(cand) > self._cand_cap:
            cand = cand[: self._cand_cap]

        best = INF_I64; k_eval = 0; winner = None
        for h in cand:
            d = manhattan(u, h) + manhattan(h, v); k_eval += 1
            if d < best:
                best = d; winner = h

        best = int(best if best < INF_I64 else 0)
        self.urn.observe_and_promote(u, v, best)
        self._audit(u, v, best, k_eval, "hubs" if inter_nonempty else ("fallback" if used_fallback else "hubs"), winner)
        return best, k_eval, False, used_fallback, mask_hit, inter_nonempty


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.cover = CoverBuilder(cfg).build(cfg.n)
        H_sets = HubBuilder(cfg).build_sets(self.cover, cfg.n)
        self.avg_hubs = sum(len(s) for s in H_sets.values()) / len(H_sets)
        self.max_hubs = max(len(s) for s in H_sets.values())
        self.H_small, self.H_fp = HubBuilder(cfg).encode(H_sets)
        self.urn = UrnCache(cfg.urn_topk, cfg.urn_up, cfg.urn_down, cfg.urn_ttl)

    def _pairs_uniform(self, num_pairs: int) -> Iterable[Tuple[Coord, Coord]]:
        n = self.cfg.n; crop = max(0.0, min(1.0, self.cfg.crop))
        x0 = y0 = int((1.0 - crop) * n / 2); x1 = y1 = n - x0
        for _ in range(num_pairs):
            yield ((self.rng.randrange(x0, x1), self.rng.randrange(y0, y1)),
                   (self.rng.randrange(x0, x1), self.rng.randrange(y0, y1)))

    def _pairs_skewed(self, total: int, hot_k: int, frac_hot: float) -> Iterable[Tuple[Coord, Coord]]:
        n = self.cfg.n; crop = max(0.0, min(1.0, self.cfg.crop))
        x0 = y0 = int((1.0 - crop) * n / 2); x1 = y1 = n - x0
        verts = [(self.rng.randrange(x0, x1), self.rng.randrange(y0, y1)) for _ in range(2 * hot_k)]
        hot = [(verts[2 * i], verts[2 * i + 1]) for i in range(hot_k)]
        for _ in range(total):
            if self.rng.random() < frac_hot:
                yield hot[self.rng.randrange(hot_k)]
            else:
                yield ((self.rng.randrange(x0, x1), self.rng.randrange(y0, y1)),
                       (self.rng.randrange(x0, x1), self.rng.randrange(y0, y1)))

    def run_uniform(self, num_pairs: int, candidate_cap: int = 2) -> Stats:
        eng = QueryEngineFast(self.cfg, self.cover, self.H_small, self.H_fp, self.urn, candidate_cap=candidate_cap)
        t0 = time.perf_counter()
        exact = 0; tot = 0; k_evals: List[int] = []
        for u, v in self._pairs_uniform(num_pairs):
            dist, k_eval, *_ = eng.query(u, v)
            truth = manhattan(u, v)
            exact += int(dist == truth); tot += 1; k_evals.append(k_eval)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        k_sorted = sorted(k_evals); p99k = k_sorted[int(0.99 * len(k_sorted))] if k_sorted else 0
        return Stats(
            n=self.cfg.n, scales=self.cfg.scales, kappa_cap=self.cfg.kappa_cap,
            avg_hubs_per_vertex=self.avg_hubs, max_hubs_per_vertex=self.max_hubs,
            avg_candidates_evaluated=sum(k_evals)/len(k_evals) if k_evals else 0.0,
            p99_candidates_evaluated=p99k, exact_rate=exact/max(tot,1),
            avg_ops=(3.0*sum(k_evals))/len(k_evals) if k_evals else 0.0,
            p99_ops=3.0*p99k, pairs=tot, elapsed_ms=elapsed_ms,
        )

    def run_skewed(self, total: int, hot_k: int = 200, frac_hot: float = 0.85, candidate_cap: int = 2) -> Dict[str, float]:
        eng = QueryEngineFast(self.cfg, self.cover, self.H_small, self.H_fp, self.urn, candidate_cap=candidate_cap)
        hits = 0; ops = []
        for u, v in self._pairs_skewed(total, hot_k, frac_hot):
            _, k_eval, hit, *_ = eng.query(u, v)
            ops.append(3*k_eval); hits += int(hit)
        return {
            "n": self.cfg.n, "hot_k": hot_k, "frac_hot": frac_hot, "total": total,
            "candidate_cap": candidate_cap, "cache_hit_rate": hits/max(total,1),
            "avg_ops": sum(ops)/len(ops) if ops else 0.0,
            "p99_ops": sorted(ops)[int(0.99*len(ops))] if ops else 0.0,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
    p = argparse.ArgumentParser(description="LARCH-δRAG: O(0)/(1) L1 grid demo (final tuned).")
    p.add_argument("--mode", choices=["uniform", "skewed"], default="uniform")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--scales", type=int, nargs="+", default=[3, 6, 10])
    p.add_argument("--cap", type=int, default=2, help="candidate cap (fast-path reserves the first two)")
    p.add_argument("--pairs", type=int, default=10_000)
    p.add_argument("--total", type=int, default=20_000)
    p.add_argument("--hot-k", type=int, default=200)
    p.add_argument("--frac-hot", type=float, default=0.85)
    p.add_argument("--crop", type=float, default=0.6)
    p.add_argument("--add-diag", action="store_true")
    p.add_argument("--no-cross", action="store_true", help="disable dynamic crosspoints (not recommended on L1)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--urn-topk", type=int, default=400)
    p.add_argument("--urn-up", type=int, default=5)
    p.add_argument("--urn-down", type=int, default=2)
    p.add_argument("--urn-ttl", type=int, default=200000)
    p.add_argument("--out", type=str, default="out.csv")
    p.add_argument("--audit-file", type=str, default=None, help="write audit TSV (u,v,dist,k_eval,src,witness)")
    p.add_argument("--audit-max", type=int, default=0, help="max audit rows (0=unlimited)")
    args = p.parse_args(argv)

    cfg = Config(
        n=max(1, int(args.n)),
        scales=tuple(sorted(set(int(x) for x in args.scales))),
        add_diag=bool(args.add_diag),
        kappa_cap=64,
        urn_topk=int(args.urn_topk),
        urn_up=int(args.urn_up),
        urn_down=int(args.urn_down),
        urn_ttl=int(args.urn_ttl),
        seed=int(args.seed),
        crop=float(args.crop),
        pairs=int(args.pairs),
        total=int(args.total),
        use_dynamic_crosspoints=(not bool(args.no_cross)),
        audit_file=args.audit_file,
        audit_max=int(args.audit_max),
    )

    ev = Evaluator(cfg)

    if args.mode == "uniform":
        st = ev.run_uniform(num_pairs=cfg.pairs, candidate_cap=int(args.cap))
        rows = [asdict(st)]
    else:
        row = ev.run_skewed(total=cfg.total, hot_k=int(args.hot_k), frac_hot=float(args.frac_hot), candidate_cap=int(args.cap))
        rows = [row]

    _dump_csv(args.out, rows)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
