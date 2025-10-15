#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# larch_deltarag_o01_fixed.py
# O(0)/(1) 強制・決定論・監査可能な 2D L1 グリッド実証ハーネス

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional, Iterable
from collections import defaultdict, Counter
import argparse, json, math, random, sys, time, hashlib

Coord = Tuple[int, int]

# -------------------- 基本 L1 --------------------
def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

# -------------------- Config / Receipts / Stats --------------------
@dataclass(frozen=True)
class Config:
    n: int = 32
    scales: Tuple[int, ...] = (3, 6, 10)
    add_diag: bool = False
    seed: int = 42
    # URN (O(0))
    urn_topk: int = 2000
    urn_up: int = 5
    urn_down: int = 2
    urn_ttl: int = 200000
    # O(1) 強制
    kappa_cap: int = 64            # |H(v)| のハード上限（既定 64）
    crop: float = 1.0
    # 実験
    pairs: int = 8000
    skewed: bool = False
    hot_k: int = 200
    frac_hot: float = 0.85
    total: int = 20000
    # 監査
    audit_frac: float = 0.0
    export_jsonl: Optional[str] = None

@dataclass
class Receipt:
    conf_sha256: str
    cover_sha256: str
    hubs_sha256: str

@dataclass
class Stats:
    n: int
    scales: Tuple[int, ...]
    kappa_cap: int
    avg_hubs_per_vertex: float
    max_hubs_per_vertex: int
    avg_candidates: float
    p50_candidates: int
    p95_candidates: int
    p99_candidates: int
    fallback_rate: float
    empty_intersection_rate: float
    exact_rate: float
    mean_stretch: float
    p99_stretch: float
    cache_hit_rate: float
    pairs: int
    elapsed_ms: float

# -------------------- Cover / Portals --------------------
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
    portals: Dict[frozenset, Coord] = field(default_factory=dict)
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
        step = max(2*R, 2)  # 正：±2R
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
        cover: List[CoverScale] = []
        for R in self.cfg.scales:
            centers = self._place_centers(R, n)
            clusters = {c: Cluster(R=R, center=c, vertices=self._grid_l1_ball(c, R, n)) for c in centers}
            cs = CoverScale(R=R, centers=centers, clusters=clusters)
            # owner_center（各点→属クラスタ中心）
            for clu in clusters.values():
                for v in clu.vertices:
                    cs.owner_center[v] = clu.center
            cover.append(cs)
        # portals：軸隣接 ±(2R,0),(0,±2R)、対角は任意
        for cs in cover:
            R = cs.R
            deltas = [(2*R,0),(-2*R,0),(0,2*R),(0,-2*R)]
            if self.cfg.add_diag:
                deltas += [(2*R,2*R),(-2*R,-2*R),(2*R,-2*R),(-2*R,2*R)]
            centers_set = set(cs.centers)
            for (cx, cy) in cs.centers:
                for dx, dy in deltas:
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) not in centers_set: continue
                    A, B = cs.clusters[(cx, cy)], cs.clusters[(nx, ny)]
                    inter = set(A.vertices).intersection(B.vertices)
                    if not inter: continue
                    mx, my = (cx + nx)//2, (cy + ny)//2
                    p = min(inter, key=lambda t: (abs(t[0]-mx)+abs(t[1]-my), t[0], t[1]))
                    key = frozenset(((cx, cy), (nx, ny)))
                    if key not in cs.portals:
                        cs.portals[key] = p
                        cs.portals_by_center[(cx, cy)].append(p)
                        cs.portals_by_center[(nx, ny)].append(p)
        return cover

# -------------------- Hubs (bounded) --------------------
class HubBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def build_sets(self, cover: List[CoverScale], n: int) -> Dict[Coord, List[Coord]]:
        H: Dict[Coord, List[Coord]] = defaultdict(list)
        # 逆引き：各スケールで v -> 属するクラスタ列
        inv_per_scale: List[Dict[Coord, List[Cluster]]] = []
        for cs in cover:
            inv: Dict[Coord, List[Cluster]] = defaultdict(list)
            for clu in cs.clusters.values():
                for v in clu.vertices:
                    inv[v].append(clu)
            inv_per_scale.append(inv)

        cap = int(self.cfg.kappa_cap)
        for cs, inv in zip(cover, inv_per_scale):
            for v, clus in inv.items():
                work: List[Coord] = []
                for clu in clus:
                    work.append(clu.center)
                    work.extend(cs.portals_by_center.get(clu.center, ()))
                # 決定論的ソート・重複除去
                work = sorted(set(work), key=lambda p: (p[0], p[1]))
                # ハードキャップ：中心優先→残り
                if len(work) > cap:
                    centers = sorted({c.center for c in clus}, key=lambda p: (p[0], p[1]))
                    rest = [w for w in work if w not in centers]
                    H[v] = (centers + rest)[:cap]
                else:
                    H[v] = work
        return H

    # --- 決定論 SplitMix64 系 ---
    @staticmethod
    def _mix64(x: int) -> int:
        x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
        x = (x ^ (x >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
        return (x ^ (x >> 31)) & 0xFFFFFFFFFFFFFFFF

    def make_fingerprint_1024_3h(self, hubs: List[Coord]) -> Tuple[int, ...]:
        # 1024bit = 16×64bit、3ハッシュ Bloom
        M = 16
        words = [0] * M
        for (x, y) in hubs:
            k0 = self._mix64((x << 32) ^ y)
            k1 = self._mix64((y << 32) ^ x)
            k2 = self._mix64((x ^ (y << 1)) & 0xFFFFFFFFFFFFFFFF)
            for k in (k0, k1, k2):
                idx = (k >> 6) & 0xF   # 0..15
                bit = k & 63          # 0..63
                words[idx] |= (1 << bit)
        return tuple(words)

    def encode(self, H_sets: Dict[Coord, List[Coord]]):
        small: Dict[Coord, Tuple[Coord, ...]] = {}
        fp: Dict[Coord, Tuple[int, ...]] = {}
        for v, arr in H_sets.items():
            t = tuple(arr[:self.cfg.kappa_cap])
            small[v] = t
            fp[v] = self.make_fingerprint_1024_3h(list(t))
        return small, fp

# -------------------- URN (O(0)) --------------------
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
            for kk in stale: del self.cache[kk]
        if k in self.cache: return
        if len(self.cache) < self.topk or self.counts[k] >= self.up:
            if len(self.cache) >= self.topk:
                least = min(self.cache.keys(), key=lambda t: (self.counts[t], self.cache[t][1]))
                if self.counts[least] <= self.down:
                    del self.cache[least]
                else:
                    return
            self.cache[k] = (value, self.clock)

# -------------------- Query Engine --------------------
@dataclass
class Witness:
    u: Coord
    v: Coord
    best: int
    truth: int
    chosen_hub: Optional[Coord]
    candidates: int
    used_fallback: bool
    cand_sample: List[Coord]

class QueryEngine:
    def __init__(self, cfg: Config, cover: List[CoverScale],
                 H_small: Dict[Coord, Tuple[Coord, ...]],
                 H_fp: Dict[Coord, Tuple[int, ...]],
                 urn: UrnCache):
        self.cfg = cfg
        self.cover = cover
        self.small = H_small
        self.fp = H_fp
        self.urn = urn
        self.owner_coarse = cover[-1].owner_center
        self.portals_by_center_coarse = cover[-1].portals_by_center
        self.owner_second = cover[-2].owner_center if len(cover) > 1 else None
        self.portals_by_center_second = cover[-2].portals_by_center if len(cover) > 1 else None

    @staticmethod
    def _intersect_small(a: Tuple[Coord, ...], b: Tuple[Coord, ...]) -> List[Coord]:
        # a,b は辞書順ソート済み、O(κ)
        i = j = 0
        out: List[Coord] = []
        while i < len(a) and j < len(b):
            if a[i] == b[j]:
                out.append(a[i]); i += 1; j += 1
            elif a[i] < b[j]:
                i += 1
            else:
                j += 1
        return out

    def _fallback_candidates(self, u: Coord, v: Coord) -> List[Coord]:
        cand: List[Coord] = []
        # 粗い→第二粗いの順で定数個ピック
        for own, pbc in ((self.owner_coarse, self.portals_by_center_coarse),
                         (self.owner_second, self.portals_by_center_second)):
            if own is None: continue
            cu = own.get(u); cv = own.get(v)
            if cu: cand.append(cu)
            if cv and cv != cu: cand.append(cv)
            pu = pbc.get(cu, []) if cu else []
            pv = pbc.get(cv, []) if cv else []
            if pu: cand.append(pu[0])
            if pv: cand.append(pv[0])
        # 決定論のため去重・上限
        out: List[Coord] = []
        seen = set()
        for c in cand:
            if c not in seen:
                seen.add(c); out.append(c)
        return out[:8]  # 定数

    def query(self, u: Coord, v: Coord, audit: bool = False) -> Tuple[int, int, bool, bool, bool, Optional[Witness]]:
        if u == v:
            w = Witness(u, v, 0, 0, None, 1, False, [])
            return 0, 1, False, False, True, (w if audit else None)

        cached = self.urn.get(u, v)
        if cached is not None:
            if audit:
                w = Witness(u, v, cached, manhattan(u, v), None, 0, False, [])
                return cached, 0, True, False, True, w
            return cached, 0, True, False, True, None

        # 1024bit（16ワード）AND の任意一致で一次ヒット
        fpu = self.fp[u]; fpv = self.fp[v]
        mask_hit = any((a & b) for a, b in zip(fpu, fpv))
        if mask_hit:
            inter = self._intersect_small(self.small[u], self.small[v])
        else:
            inter = []

        used_fallback = False
        if inter:
            cand = inter
        else:
            cand = self._fallback_candidates(u, v)
            used_fallback = True

        best = math.inf; chosen = None
        for h in cand:
            d = manhattan(u, h) + manhattan(h, v)
            if d < best:
                best = d; chosen = h
        best = 0 if not math.isfinite(best) else int(best)
        self.urn.observe_and_promote(u, v, best)

        if audit:
            w = Witness(u, v, best, manhattan(u, v), chosen, len(cand), used_fallback, cand[:8])
            return best, len(cand), False, used_fallback, mask_hit, w
        return best, len(cand), False, used_fallback, mask_hit, None

# -------------------- Evaluator --------------------
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
        # receipts
        conf_sha = hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()
        cover_digest = hashlib.sha256()
        for cs in self.cover:
            cover_digest.update(str((cs.R, len(cs.centers), len(cs.portals))).encode())
        hubs_sha = hashlib.sha256(str(sum(len(v) for v in self.H_small.values())).encode()).hexdigest()
        self.receipt = Receipt(conf_sha, cover_digest.hexdigest(), hubs_sha)

    def _pairs_uniform(self, num_pairs: int) -> Iterable[Tuple[Coord, Coord]]:
        n = self.cfg.n; crop = max(0.0, min(1.0, self.cfg.crop))
        x0 = y0 = int((1.0 - crop) * n / 2); x1 = y1 = n - x0
        for _ in range(num_pairs):
            yield (self.rng.randrange(x0, x1), self.rng.randrange(y0, y1)), \
                  (self.rng.randrange(x0, x1), self.rng.randrange(y0, y1))

    def _pairs_skewed(self, total: int, hot_k: int, frac_hot: float):
        n = self.cfg.n; crop = max(0.0, min(1.0, self.cfg.crop))
        x0 = y0 = int((1.0 - crop) * n / 2); x1 = y1 = n - x0
        verts = [(self.rng.randrange(x0, x1), self.rng.randrange(y0, y1)) for _ in range(2*hot_k)]
        hot = [(verts[2*i], verts[2*i+1]) for i in range(hot_k)]
        for _ in range(total):
            if self.rng.random() < frac_hot:
                yield hot[self.rng.randrange(hot_k)]
            else:
                yield (self.rng.randrange(x0, x1), self.rng.randrange(y0, y1)), \
                      (self.rng.randrange(x0, x1), self.rng.randrange(y0, y1))

    def run_uniform(self, num_pairs: int, audit_frac: float = 0.0, jsonl: Optional[str] = None) -> Stats:
        eng = QueryEngine(self.cfg, self.cover, self.H_small, self.H_fp, self.urn)
        t0 = time.perf_counter()
        exact = tot = hits = 0
        stretches: List[float] = []
        cand_sizes: List[int] = []
        fallbacks = 0
        empty_hits = 0
        dump = jsonl and audit_frac > 0.0
        f = open(jsonl, "a", encoding="utf-8") if dump else None

        for u, v in self._pairs_uniform(num_pairs):
            audit = dump and (self.rng.random() < audit_frac)
            est, k, hit, used_fb, had_fp, w = eng.query(u, v, audit=audit)
            truth = manhattan(u, v)
            exact += int(est == truth)
            tot += 1
            if est != truth:
                stretches.append(est / max(truth, 1))
            cand_sizes.append(k)
            hits += int(hit)
            fallbacks += int(used_fb)
            empty_hits += int((not had_fp) and used_fb)
            if w and f:
                f.write(json.dumps({
                    "u": w.u, "v": w.v, "best": w.best, "truth": w.truth,
                    "chosen_hub": w.chosen_hub, "candidates": w.candidates,
                    "used_fallback": w.used_fallback, "cand_sample": w.cand_sample,
                    "receipt": asdict(self.receipt)
                }, ensure_ascii=False) + "\n")

        if f: f.close()

        cand_sorted = sorted(cand_sizes)
        def pct(a, p):
            if not a: return 0
            i = min(len(a)-1, max(0, int(p*len(a))))
            return a[i]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return Stats(
            n=self.cfg.n,
            scales=self.cfg.scales,
            kappa_cap=self.cfg.kappa_cap,
            avg_hubs_per_vertex=self.avg_hubs,
            max_hubs_per_vertex=self.max_hubs,
            avg_candidates=sum(cand_sizes)/len(cand_sizes) if cand_sizes else 0.0,
            p50_candidates=pct(cand_sorted, 0.50),
            p95_candidates=pct(cand_sorted, 0.95),
            p99_candidates=pct(cand_sorted, 0.99),
            fallback_rate=fallbacks / max(tot, 1),
            empty_intersection_rate=empty_hits / max(tot, 1),
            exact_rate=exact / max(tot, 1),
            mean_stretch=(sum(stretches)/len(stretches) if stretches else 1.0),
            p99_stretch=(sorted(stretches)[min(len(stretches)-1, int(0.99*len(stretches)))] if stretches else 1.0),
            cache_hit_rate=hits / max(tot, 1),
            pairs=tot,
            elapsed_ms=elapsed_ms,
        )

    def run_skewed(self, total: int, hot_k: int, frac_hot: float) -> Dict[str, float | int]:
        eng = QueryEngine(self.cfg, self.cover, self.H_small, self.H_fp, self.urn)
        total_cand = 0; hits = 0
        for u, v in self._pairs_skewed(total, hot_k, frac_hot):
            est, k, hit, _, _, _ = eng.query(u, v, audit=False)
            total_cand += k; hits += int(hit)
        return {
            "n": self.cfg.n,
            "avg_candidates_overall": total_cand/max(total, 1),
            "cache_hit_rate_overall": hits/max(total, 1),
            "urn_cache_size": len(self.urn.cache),
            "hot_k": hot_k, "frac_hot": frac_hot, "total": total,
        }

# -------------------- CLI --------------------
def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LARCH-δRAG O(0)/(1) enforced (deterministic FP, receipts).")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--scales", type=int, nargs="+", default=[3, 6, 10])
    p.add_argument("--diag", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--kappa-cap", type=int, default=64)
    p.add_argument("--pairs", type=int, default=8000)
    p.add_argument("--crop", type=float, default=1.0)
    # URN
    p.add_argument("--urn-topk", type=int, default=2000)
    p.add_argument("--urn-up", type=int, default=5)
    p.add_argument("--urn-down", type=int, default=2)
    p.add_argument("--urn-ttl", type=int, default=200000)
    # skewed
    p.add_argument("--skewed", action="store_true")
    p.add_argument("--hot-k", type=int, default=200)
    p.add_argument("--frac-hot", type=float, default=0.85)
    p.add_argument("--total", type=int, default=20000)
    # audit
    p.add_argument("--audit-frac", type=float, default=0.0)
    p.add_argument("--export-jsonl", type=str)
    return p.parse_args(argv)

def cfg_from_args(a: argparse.Namespace) -> Config:
    return Config(
        n=int(a.n), scales=tuple(sorted(set(int(x) for x in a.scales))),
        add_diag=bool(a.diag), seed=int(a.seed),
        kappa_cap=int(a.kappa_cap), pairs=int(a.pairs), crop=float(a.crop),
        urn_topk=int(a.urn_topk), urn_up=int(a.urn_up), urn_down=int(a.urn_down), urn_ttl=int(a.urn_ttl),
        skewed=bool(a.skewed), hot_k=int(a.hot_k), frac_hot=float(a.frac_hot), total=int(a.total),
        audit_frac=float(a.audit_frac), export_jsonl=a.export_jsonl
    )

def main(argv: List[str]) -> int:
    args = parse_args(argv)
    cfg = cfg_from_args(args)
    ev = Evaluator(cfg)
    stats = ev.run_uniform(num_pairs=cfg.pairs, audit_frac=cfg.audit_frac, jsonl=cfg.export_jsonl)
    out = {"uniform": asdict(stats), "receipt": asdict(ev.receipt)}
    if cfg.skewed:
        out["skewed"] = ev.run_skewed(total=cfg.total, hot_k=cfg.hot_k, frac_hot=cfg.frac_hot)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

