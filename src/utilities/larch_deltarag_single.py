
#!/usr/bin/env python3
# larch_deltarag_single.py
# General-graph LARCH-δRAG: cover + portals + hubs, packed labels,
# hopset-like neighbor overlay, and CEA greedy exact O(1) neighborhood.
# Python 3.9+. Dependencies: numpy, matplotlib
from __future__ import annotations

import argparse, json, time, math, random, heapq
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Sequence, Set
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---------------- Graph ----------------
Vertex = int

@dataclass
class Graph:
    n: int
    directed: bool = False
    adj: Dict[Vertex, List[Tuple[Vertex, float]]] = field(default_factory=lambda: defaultdict(list))
    def add_edge(self, u: Vertex, v: Vertex, w: float) -> None:
        if w < 0: raise ValueError("edge weights must be nonnegative")
        self.adj[u].append((v, float(w)))
        if not self.directed:
            self.adj[v].append((u, float(w)))
    def neighbors(self, u: Vertex) -> List[Tuple[Vertex, float]]:
        return self.adj.get(u, [])

def grid_graph(side: int) -> Graph:
    g = Graph(n=side*side, directed=False)
    def idx(x,y): return x*side + y
    for x in range(side):
        for y in range(side):
            u = idx(x,y)
            if x+1<side: g.add_edge(u, idx(x+1,y), 1.0)
            if y+1<side: g.add_edge(u, idx(x,y+1), 1.0)
    return g

# ---------------- Shortest paths ----------------
def is_unit_weight(g: Graph) -> bool:
    for u in range(g.n):
        for v,w in g.neighbors(u):
            if abs(w - 1.0) > 1e-9:
                return False
    return True

def dijkstra_until_targets(g: Graph, source: Vertex, targets: Optional[Set[Vertex]]=None, cutoff: Optional[float]=None):
    dist = {source: 0.0}
    pq = [(0.0, source)]
    remaining = set(targets) if targets is not None else None
    while pq:
        d,u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        if cutoff is not None and d > cutoff:
            break
        if remaining is not None and u in remaining:
            remaining.remove(u)
            if not remaining:
                break
        for v,w in g.neighbors(u):
            nd = d + w
            if cutoff is not None and nd > cutoff:
                continue
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq,(nd,v))
    return dist

def bfs_until_targets(g: Graph, source: int, targets: Optional[Set[int]] = None, cutoff: Optional[float] = None) -> Dict[int, float]:
    q = deque([source])
    dist = {source: 0}
    remaining = set(targets) if targets is not None else None
    R = int(cutoff) if cutoff is not None else None
    while q:
        u = q.popleft()
        du = dist[u]
        if R is not None and du > R:
            continue
        if remaining is not None and u in remaining:
            remaining.remove(u)
            if not remaining:
                break
        for v,_ in g.neighbors(u):
            if v in dist: continue
            nd = du + 1
            if R is not None and nd > R: continue
            dist[v] = nd
            q.append(v)
    return {k: float(v) for k,v in dist.items()}

def sp_tree_with_pred(g: Graph, source: int, cutoff: Optional[float]=None, target: Optional[int]=None):
    dist = {source: 0.0}
    pred = {source: -1}
    pq = [(0.0, source)]
    while pq:
        d,u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        if cutoff is not None and d > cutoff:
            break
        if target is not None and u == target:
            break
        for v,w in g.neighbors(u):
            nd = d + w
            if cutoff is not None and nd > cutoff:
                continue
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                pred[v] = u
                heapq.heappush(pq,(nd,v))
    return dist, pred

def recover_path(pred: Dict[int,int], target: int) -> List[int]:
    if target not in pred: return []
    path = []
    x = target
    while x != -1:
        path.append(x)
        x = pred.get(x, -1)
    path.reverse()
    return path

# ---------------- Cover + portals ----------------
@dataclass
class CoverScale:
    R: float
    centers: Tuple[Vertex, ...]
    clusters: Dict[Vertex, Tuple[Vertex, ...]]
    d_to_center: Dict[Vertex, Dict[Vertex, float]]
    portals: Dict[frozenset, Tuple[Vertex, ...]]
    portals_by_center: Dict[Vertex, Tuple[Vertex, ...]]
    owner_center: Dict[Vertex, Vertex]

def _ball_vertices(g: Graph, c: Vertex, R: float, use_bfs: bool):
    dist = bfs_until_targets(g, c, cutoff=R) if use_bfs else dijkstra_until_targets(g, c, cutoff=R)
    verts = [v for v,d in dist.items() if d <= R]
    return dist, verts

def build_cover(g: Graph, scales: Sequence[float], seed: int=42, portals_per_edge: int=3, use_bfs: Optional[bool]=None) -> Tuple[CoverScale,...]:
    if use_bfs is None:
        use_bfs = is_unit_weight(g)
    rng = random.Random(seed)
    out = []
    for R in scales:
        order = list(range(g.n)); rng.shuffle(order)
        covered = [False]*g.n
        centers = []
        clusters = {}
        d_to_center = {}
        owner_center = {}
        for v in order:
            if covered[v]: continue
            centers.append(v)
            dmap, verts = _ball_vertices(g, v, R, use_bfs=use_bfs)
            clusters[v] = tuple(sorted(verts))
            d_to_center[v] = dmap
            for u in verts:
                if not covered[u]:
                    covered[u] = True
                    owner_center[u] = v
        inv = defaultdict(list)
        for c, verts in clusters.items():
            for u in verts: inv[u].append(c)
        portals = defaultdict(list)
        portals_by_center = defaultdict(set)
        for u, owners in inv.items():
            if len(owners) < 2: continue
            owners.sort()
            for i in range(len(owners)):
                ca = owners[i]
                for j in range(i+1, len(owners)):
                    cb = owners[j]
                    da = d_to_center[ca].get(u, math.inf)
                    db = d_to_center[cb].get(u, math.inf)
                    if math.isfinite(da) and math.isfinite(db):
                        score = (abs(da-db), da+db, u)
                        portals[frozenset((ca,cb))].append((score,u))
        final_portals = {}
        for key, arr in portals.items():
            arr.sort(key=lambda t:t[0])
            seen=set(); chosen=[]
            for _, v in arr:
                if v in seen: continue
                seen.add(v); chosen.append(v)
                if len(chosen) >= portals_per_edge: break
            if chosen:
                final_portals[key] = tuple(chosen)
                for c in key:
                    for p in chosen: portals_by_center[c].add(p)
        out.append(CoverScale(
            R=R,
            centers=tuple(sorted(centers)),
            clusters=clusters,
            d_to_center=d_to_center,
            portals={k:v for k,v in final_portals.items()},
            portals_by_center={c: tuple(sorted(vs)) for c,vs in portals_by_center.items()},
            owner_center=owner_center
        ))
    return tuple(out)

# ---------------- Hubs + sparse v-h distances ----------------
@dataclass
class Preprocessed:
    cover: Tuple[CoverScale, ...]
    hub_sets: Dict[Vertex, Tuple[Vertex, ...]]
    hub_to_targets: Dict[Vertex, Set[Vertex]]
    d_vh: Dict[Vertex, Dict[Vertex, float]]
    coarsest_owner: Dict[Vertex, Vertex]
    coarsest_portals_by_center: Dict[Vertex, Tuple[Vertex, ...]]
    avg_hubs: float
    max_hubs: int

def build_hubs(g: Graph, cover: Tuple[CoverScale, ...], use_bfs: Optional[bool]=None) -> Preprocessed:
    if use_bfs is None:
        use_bfs = is_unit_weight(g)
    hubs_by_v = defaultdict(set)
    hub_to_targets = defaultdict(set)
    for cs in cover:
        for center, verts in cs.clusters.items():
            plist = cs.portals_by_center.get(center, ())
            for v in verts:
                H = hubs_by_v[v]
                H.add(center)
                H.update(plist)
                hub_to_targets[center].add(v)
                for p in plist:
                    hub_to_targets[p].add(v)
    hub_sets = {v: tuple(sorted(H)) for v, H in hubs_by_v.items()}
    max_hubs = max((len(H) for H in hub_sets.values()), default=0)
    avg_hubs = (sum(len(H) for H in hub_sets.values()) / len(hub_sets)) if hub_sets else 0.0
    d_vh = defaultdict(dict)
    for h, targets in hub_to_targets.items():
        dist = bfs_until_targets(g, h, targets=set(targets)) if use_bfs else dijkstra_until_targets(g, h, targets=set(targets))
        for v in targets:
            d_vh[v][h] = dist.get(v, math.inf)
    coarsest = cover[-1]
    return Preprocessed(
        cover=cover,
        hub_sets=hub_sets,
        hub_to_targets=hub_to_targets,
        d_vh=d_vh,
        coarsest_owner=coarsest.owner_center,
        coarsest_portals_by_center={c: coarsest.portals_by_center.get(c, tuple()) for c in coarsest.centers},
        avg_hubs=avg_hubs,
        max_hubs=max_hubs
    )

# ---------------- Packed labels + engine ----------------
@dataclass
class PackedData:
    hubs: List[np.ndarray]          # dtype=int32, sorted strictly increasing
    dists: List[np.ndarray]         # dtype=float32, same length/order as hubs[v]
    coarsest_owner: Dict[int, int]
    coarsest_portals_by_center: Dict[int, Tuple[int, ...]]
    total_entries: int
    bytes_payload: int

def pack_labels(data: Preprocessed, n_vertices: int) -> PackedData:
    hubs_arr: List[np.ndarray] = [np.empty((0,), dtype=np.int32) for _ in range(n_vertices)]
    dists_arr: List[np.ndarray] = [np.empty((0,), dtype=np.float32) for _ in range(n_vertices)]
    total = 0
    for v, H in data.hub_sets.items():
        hs = np.fromiter(H, dtype=np.int32, count=len(H))
        ds = np.empty((len(H),), dtype=np.float32)
        m = data.d_vh.get(v, {})
        for i, h in enumerate(H):
            ds[i] = float(m.get(h, math.inf))
        hubs_arr[v] = hs
        dists_arr[v] = ds
        total += len(H)
    return PackedData(
        hubs=hubs_arr,
        dists=dists_arr,
        coarsest_owner=data.coarsest_owner,
        coarsest_portals_by_center=data.coarsest_portals_by_center,
        total_entries=total,
        bytes_payload=total * (4+4)
    )

class PackedEngine:
    __slots__ = ("pd",)
    def __init__(self, pd: PackedData):
        self.pd = pd
    def _fallback(self, u: int, v: int) -> np.ndarray:
        owners = self.pd.coarsest_owner
        portals = self.pd.coarsest_portals_by_center
        seeds = []
        for x in (u, v):
            c = owners.get(x)
            if c is not None:
                seeds.append(c)
                seeds.extend(portals.get(c, ()))
        if not seeds:
            return np.empty((0,), dtype=np.int32)
        return np.array(sorted(set(seeds)), dtype=np.int32)
    def query(self, u: int, v: int):
        if u == v:
            return 0.0, 1, False, u
        Hu = self.pd.hubs[u]; Hv = self.pd.hubs[v]
        Du = self.pd.dists[u]; Dv = self.pd.dists[v]
        i = j = 0
        best = math.inf; wit = -1; candidates = 0
        while i < Hu.size and j < Hv.size:
            a = Hu[i]; b = Hv[j]
            if a == b:
                s = float(Du[i] + Dv[j]); candidates += 1
                if s < best: best = s; wit = int(a)
                i += 1; j += 1
            elif a < b:
                i += 1
            else:
                j += 1
        fb = False
        if not np.isfinite(best):
            fb = True
            seeds = self._fallback(u, v)
            if seeds.size > 0:
                for h in seeds:
                    iu = np.searchsorted(Hu, h)
                    iv = np.searchsorted(Hv, h)
                    du = float(Du[iu]) if iu < Hu.size and Hu[iu] == h else math.inf
                    dv = float(Dv[iv]) if iv < Hv.size and Hv[iv] == h else math.inf
                    s = du + dv; candidates += 1
                    if s < best: best = s; wit = int(h)
        return float(best), int(candidates), bool(fb), (None if wit < 0 else wit)

# ---------------- Hopset-like neighbor overlay ----------------
def center_neighbors_by_distance(cs, budget:int=2) -> Dict[int, Tuple[int,...]]:
    neigh: Dict[int, List[Tuple[float,int]]] = {c: [] for c in cs.centers}
    for key, plist in cs.portals.items():
        ca, cb = tuple(key)
        best = math.inf
        for p in plist:
            da = cs.d_to_center[ca].get(p, math.inf)
            db = cs.d_to_center[cb].get(p, math.inf)
            s = da + db
            if s < best: best = s
        if math.isfinite(best):
            neigh[ca].append((best, cb))
            neigh[cb].append((best, ca))
    picked: Dict[int, Tuple[int,...]] = {}
    for c, items in neigh.items():
        items.sort(key=lambda t:t[0])
        picked[c] = tuple(v for _, v in items[:budget])
    return picked

def build_with_neighbor_overlay(g: Graph, cover: Tuple[CoverScale,...], nbhd_budget:int=2, use_bfs: Optional[bool]=None) -> Preprocessed:
    if use_bfs is None:
        use_bfs = is_unit_weight(g)
    base = build_hubs(g, cover, use_bfs=use_bfs)
    cs = cover[-1]
    picked = center_neighbors_by_distance(cs, budget=nbhd_budget)
    hub_to_targets = {h: set(t) for h, t in base.hub_to_targets.items()}
    for c, verts in cs.clusters.items():
        for n in picked.get(c, ()):
            hub_to_targets.setdefault(n, set()).update(verts)
    d_vh = {v: dict(m) for v, m in base.d_vh.items()}
    for h, targets in hub_to_targets.items():
        need = {v for v in targets if h not in d_vh.get(v, {})}
        if not need: continue
        dist = bfs_until_targets(g, h, targets=set(need)) if use_bfs else dijkstra_until_targets(g, h, targets=set(need))
        for v in need:
            d_vh.setdefault(v, {})[h] = dist.get(v, math.inf)
    hubs_by_v = {v: set(H) for v, H in base.hub_sets.items()}
    for c, verts in cs.clusters.items():
        for n in picked.get(c, ()):
            for v in verts:
                hubs_by_v[v].add(n)
    hub_sets = {v: tuple(sorted(H)) for v, H in hubs_by_v.items()}
    return Preprocessed(
        cover=cover,
        hub_sets=hub_sets,
        hub_to_targets=hub_to_targets,
        d_vh=d_vh,
        coarsest_owner=base.coarsest_owner,
        coarsest_portals_by_center=base.coarsest_portals_by_center,
        avg_hubs=sum(len(H) for H in hub_sets.values())/len(hub_sets),
        max_hubs=max(len(H) for H in hub_sets.values())
    )

# ---------------- Exact O(1) neighborhood via CEA greedy ----------------
@dataclass
class Failure:
    u: int
    v: int
    d_true: float
    h_mid: int

def recover_midpoint_on_path(path: List[int], dist_from_u: Dict[int,float], total_len: float) -> Optional[int]:
    half = total_len / 2.0
    best = (1e18, None)
    for v in path:
        d = dist_from_u.get(v, 1e18)
        if abs(d - half) < best[0]:
            best = (abs(d - half), v)
    return best[1]

def collect_failures(g: Graph, eng: PackedEngine, tau: float, use_bfs: bool=True) -> List[Failure]:
    fails: List[Failure] = []
    for u in range(g.n):
        dist_u = bfs_until_targets(g, u, cutoff=tau) if use_bfs else dijkstra_until_targets(g, u, cutoff=tau)
        for v, dtrue in dist_u.items():
            if v <= u: 
                continue
            d2, _, _, _ = eng.query(u, v)
            if abs(d2 - dtrue) < 1e-9:
                continue
            dist_u2, pred = sp_tree_with_pred(g, u, cutoff=tau, target=v)
            path = recover_path(pred, v)
            if not path:
                continue
            h = recover_midpoint_on_path(path, dist_u2, dtrue)
            if h is None:
                continue
            fails.append(Failure(u=u, v=v, d_true=dtrue, h_mid=h))
    return fails

def greedy_fix(g: Graph, base_pd: PackedData, failures: List[Failure], tau: float, use_bfs: bool=True):
    extra: Dict[int, Dict[int, float]] = defaultdict(dict)
    hub_to_pairs: Dict[int, List[int]] = defaultdict(list)
    for idx, f in enumerate(failures):
        hub_to_pairs[f.h_mid].append(idx)
    uncovered = set(range(len(failures)))
    while uncovered:
        best_h = None; best_gain = -1
        for h, idxs in hub_to_pairs.items():
            gain = sum(1 for i in idxs if i in uncovered)
            if gain > best_gain:
                best_gain = gain; best_h = h
        if best_h is None or best_gain <= 0: break
        d_h = bfs_until_targets(g, best_h, cutoff=tau) if use_bfs else dijkstra_until_targets(g, best_h, cutoff=tau)
        for i in list(hub_to_pairs[best_h]):
            if i not in uncovered: continue
            f = failures[i]
            du = d_h.get(f.u, math.inf); dv = d_h.get(f.v, math.inf)
            if not (math.isfinite(du) and math.isfinite(dv)): continue
            extra[f.u][best_h] = float(du)
            extra[f.v][best_h] = float(dv)
            uncovered.remove(i)
    return extra, len(failures) - len(uncovered), len(uncovered)

def merge_and_pack(base_pd: PackedData, extra: Dict[int, Dict[int, float]]) -> PackedData:
    n = len(base_pd.hubs)
    hubs_arr: List[np.ndarray] = [None]*n
    dists_arr: List[np.ndarray] = [None]*n
    total = 0
    for v in range(n):
        baseH = list(base_pd.hubs[v])
        baseD = list(base_pd.dists[v])
        merged = dict(zip(baseH, baseD))
        for h, d in extra.get(v, {}).items():
            if d < merged.get(h, math.inf):
                merged[h] = d
        H_sorted = sorted(merged.keys())
        D_sorted = [float(merged[h]) for h in H_sorted]
        aH = np.array(H_sorted, dtype=np.int32)
        aD = np.array(D_sorted, dtype=np.float32)
        hubs_arr[v] = aH; dists_arr[v] = aD
        total += aH.size
    return PackedData(
        hubs=hubs_arr, dists=dists_arr,
        coarsest_owner=base_pd.coarsest_owner,
        coarsest_portals_by_center=base_pd.coarsest_portals_by_center,
        total_entries=total, bytes_payload=total*(4+4)
    )

# ---------------- CLI ops ----------------
def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2))
    print(f"[saved] {path}")

def op_viz_grid(args):
    side = args.side
    R = tuple(args.scales)
    K = args.portals
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    g = grid_graph(side)
    cover = build_cover(g, scales=R, seed=args.seed, portals_per_edge=K)
    data = build_hubs(g, cover)
    pd = pack_labels(data, g.n)
    eng = PackedEngine(pd)

    def xy(idx): return (idx // side, idx % side)

    # Plot 1: Graph + centers + coarsest portals
    plt.figure(figsize=(7,7))
    drawn=set()
    for u in range(g.n):
        for v,w in g.neighbors(u):
            if (v,u) in drawn: continue
            x1,y1 = xy(u); x2,y2 = xy(v)
            plt.plot([y1,y2],[x1,x2], linewidth=0.7)
            drawn.add((u,v))
    for cs in cover:
        xs=[xy(c)[1] for c in cs.centers]; ys=[xy(c)[0] for c in cs.centers]
        plt.scatter(xs, ys, s=28, label=f"centers R={cs.R}")
    coarsest = cover[-1]
    portal_vertices=set()
    for plist in coarsest.portals_by_center.values():
        for p in plist: portal_vertices.add(p)
    if portal_vertices:
        xs=[xy(p)[1] for p in portal_vertices]; ys=[xy(p)[0] for p in portal_vertices]
        plt.scatter(xs, ys, s=10, marker='x', label="coarsest portals")
    plt.title("graph, centers, portals")
    plt.gca().invert_yaxis(); plt.axis("equal"); plt.legend(loc="upper right"); plt.tight_layout()
    fig1=outdir/"viz_graph_centers_portals.png"; plt.savefig(fig1, dpi=180); plt.close()

    # Plot 2: One query visualization
    rng = random.Random(args.seed+123)
    u = rng.randrange(g.n); v = rng.randrange(g.n)
    dist, k, fb, wit = eng.query(u, v)
    # candidates
    def inter_candidates(u,v):
        Hu=pd.hubs[u]; Hv=pd.hubs[v]
        i=j=0; res=[]
        while i<Hu.size and j<Hv.size:
            a=Hu[i]; b=Hv[j]
            if a==b: res.append(int(a)); i+=1; j+=1
            elif a<b: i+=1
            else: j+=1
        if res: return res, False
        owners=pd.coarsest_owner; portals=pd.coarsest_portals_by_center
        seeds=[]
        for x in (u,v):
            c=owners.get(x)
            if c is not None:
                seeds.append(c); seeds.extend(portals.get(c, ()))
        res=sorted(set(seeds))
        return res, True
    cand, fb2 = inter_candidates(u,v)
    plt.figure(figsize=(7,7))
    drawn=set()
    for a in range(g.n):
        for b,w in g.neighbors(a):
            if (b,a) in drawn: continue
            x1,y1 = xy(a); x2,y2 = xy(b)
            plt.plot([y1,y2],[x1,x2], linewidth=0.4)
            drawn.add((a,b))
    ux,uy = xy(u); vx,vy = xy(v)
    plt.scatter([uy],[ux], s=90, marker='o', label=f"u={u}")
    plt.scatter([vy],[vx], s=90, marker='s', label=f"v={v}")
    cand_x=[xy(h)[0] for h in cand]; cand_y=[xy(h)[1] for h in cand]
    plt.scatter(cand_y, cand_x, s=60, marker='^', label=f"candidates ({len(cand)})")
    if wit is not None and math.isfinite(dist):
        wx,wy = xy(wit); plt.scatter([wy],[wx], s=120, marker='*', label=f"witness={wit}")
    plt.title(f"query: d={dist:.3f}, cand={len(cand)}, fallback={fb or fb2}")
    plt.gca().invert_yaxis(); plt.axis("equal"); plt.legend(loc="upper right"); plt.tight_layout()
    fig2=outdir/"viz_query.png"; plt.savefig(fig2, dpi=180); plt.close()

    # Plot 3: Candidate histogram
    pairs = args.pairs
    cand_counts = []; exact = 0
    for _ in range(pairs):
        a = rng.randrange(g.n); b = rng.randrange(g.n)
        d2, k2, fb3, w2 = eng.query(a, b)
        cand_counts.append(k2)
        truth = dijkstra_until_targets(g, a, targets={b}).get(b, math.inf)
        exact += int(abs(d2 - truth) < 1e-9)
    plt.figure(figsize=(7,4))
    bins = np.arange(0, max(cand_counts)+2) - 0.5
    plt.hist(cand_counts, bins=bins)
    plt.title(f"candidates/pair (n={pairs}) avg={np.mean(cand_counts):.2f} p99={np.percentile(cand_counts,99):.0f} exact={exact/pairs:.2%}")
    plt.xlabel("candidates"); plt.ylabel("freq"); plt.tight_layout()
    fig3=outdir/"viz_cand_hist.png"; plt.savefig(fig3, dpi=180); plt.close()

    meta = {"graph": f"grid {side}x{side}", "scales": list(R), "portals": K, "pairs": pairs,
            "figures": {"graph_centers_portals": str(fig1), "query": str(fig2), "cand_hist": str(fig3)}}
    save_json(meta, outdir/"viz_meta.json")

def op_hopset(args):
    side=args.side; R=tuple(args.scales); K=args.portals; B=args.budget
    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    g=grid_graph(side); cover=build_cover(g,R,seed=args.seed,portals_per_edge=K)
    base=build_hubs(g,cover)
    pd_base=pack_labels(base,g.n); eng_base=PackedEngine(pd_base)
    # augmented
    aug=build_with_neighbor_overlay(g, cover, nbhd_budget=B)
    pd_aug=pack_labels(aug,g.n); eng_aug=PackedEngine(pd_aug)
    # trial
    rng=random.Random(args.seed+99)
    pairs=[(rng.randrange(g.n), rng.randrange(g.n)) for _ in range(args.pairs)]
    def bench(eng):
        lat=[]
        for _ in range(200):
            a,b=pairs[rng.randrange(len(pairs))]; eng.query(a,b)
        for a,b in pairs:
            t0=time.perf_counter(); eng.query(a,b); t1=time.perf_counter()
            lat.append((t1-t0)*1e6)
        return float(np.percentile(lat,50)), float(np.percentile(lat,95)), float(np.percentile(lat,99))
    p50b,p95b,p99b=bench(eng_base)
    p50a,p95a,p99a=bench(eng_aug)
    res={"graph":f"grid-{side}x{side}","R":list(R),"K":K,"B":B,
         "base_us":{"p50":p50b,"p95":p95b,"p99":p99b},
         "aug_us":{"p50":p50a,"p95":p95a,"p99":p99a},
         "avg_hubs_base":float(sum(len(h) for h in pd_base.hubs))/len(pd_base.hubs),
         "avg_hubs_aug":float(sum(len(h) for h in pd_aug.hubs))/len(pd_aug.hubs)}
    save_json(res, outdir/"hopset_result.json")

def op_exact_o1(args):
    side=args.side; R=tuple(args.scales); K=args.portals; tau=args.tau
    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    g=grid_graph(side); cover=build_cover(g,R,seed=args.seed,portals_per_edge=K)
    base=build_hubs(g,cover); pd=pack_labels(base,g.n); eng=PackedEngine(pd)
    # failures and greedy fix
    use_bfs=True
    fails=collect_failures(g, eng, tau=tau, use_bfs=use_bfs)
    extra,fixed,remain=greedy_fix(g, pd, fails, tau=tau, use_bfs=use_bfs)
    pd_new=merge_and_pack(pd, extra); eng_new=PackedEngine(pd_new)
    # verify & bench
    total=0; exact=0
    for u in range(g.n):
        dist_u=bfs_until_targets(g,u,cutoff=tau)
        for v,dtrue in dist_u.items():
            if v<=u: continue
            total+=1
            d2,_,_,_=eng_new.query(u,v)
            if abs(d2-dtrue)<1e-9: exact+=1
    rng=random.Random(args.seed+7)
    pairs=[(rng.randrange(g.n), rng.randrange(g.n)) for _ in range(args.pairs)]
    lat=[]
    for _ in range(200): a,b=pairs[rng.randrange(len(pairs))]; eng_new.query(a,b)
    for a,b in pairs:
        t0=time.perf_counter(); eng_new.query(a,b); t1=time.perf_counter()
        lat.append((t1-t0)*1e6)
    res={"graph":f"grid-{side}x{side}","R":list(R),"K":K,"tau":tau,
         "failures":len(fails),"fixed":fixed,"remaining":remain,
         "exact_rate_le_tau": exact/max(1,total),
         "avg_hubs_per_vertex": float(sum(len(h) for h in pd_new.hubs))/len(pd_new.hubs),
         "latency_us":{"p50":float(np.percentile(lat,50)),"p95":float(np.percentile(lat,95))}}
    save_json(res, outdir/"exact_o1_result.json")

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="LARCH-δRAG single script")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_viz = sub.add_parser("viz-grid", help="visualize grid + centers/portals + candidate histogram")
    ap_viz.add_argument("--side", type=int, default=16)
    ap_viz.add_argument("--scales", type=float, nargs="+", default=[2.0,4.0,8.0])
    ap_viz.add_argument("--portals", type=int, default=3)
    ap_viz.add_argument("--pairs", type=int, default=2000)
    ap_viz.add_argument("--seed", type=int, default=42)
    ap_viz.add_argument("--outdir", type=str, default="out_viz")
    ap_viz.set_defaults(func=op_viz_grid)

    ap_hs = sub.add_parser("hopset", help="neighbor-overlay hopset benchmark")
    ap_hs.add_argument("--side", type=int, default=32)
    ap_hs.add_argument("--scales", type=float, nargs="+", default=[3.0,6.0])
    ap_hs.add_argument("--portals", type=int, default=2)
    ap_hs.add_argument("--budget", type=int, default=2)
    ap_hs.add_argument("--pairs", type=int, default=3000)
    ap_hs.add_argument("--seed", type=int, default=42)
    ap_hs.add_argument("--outdir", type=str, default="out_hopset")
    ap_hs.set_defaults(func=op_hopset)

    ap_e = sub.add_parser("exact-o1", help="exact O(1) neighborhood via CEA greedy")
    ap_e.add_argument("--side", type=int, default=32)
    ap_e.add_argument("--scales", type=float, nargs="+", default=[3.0,6.0])
    ap_e.add_argument("--portals", type=int, default=2)
    ap_e.add_argument("--tau", type=float, default=4.0)
    ap_e.add_argument("--pairs", type=int, default=2000)
    ap_e.add_argument("--seed", type=int, default=42)
    ap_e.add_argument("--outdir", type=str, default="out_exact")
    ap_e.set_defaults(func=op_exact_o1)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
