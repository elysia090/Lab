# larch_deltaRAG.py
# ASCII reference implementation for LARCH-deltaRAG on 2D L1 grid graphs.
# - four-phase multiscale L1 lattice cover
# - Cech/Rips nerve, finite portals per overlap
# - hub sets per vertex, O(1) two-hop distance queries
# - optional O(0) dictionary (lightweight)
#
# Didactic, not optimized; suitable for prototyping and unit tests.

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional

Array = np.ndarray
Coord = Tuple[int, int]

def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

@dataclass
class CoverScale:
    R: int
    centers: List[Coord]                  # grid centers
    balls: Dict[Coord, List[Coord]]       # center -> vertices (subset of V)
    neighbors: Dict[Coord, List[Coord]]   # Cech/Rips 1-skeleton adjacency
    portals: Dict[Tuple[Coord,Coord], List[Coord]]  # overlap -> finite portal list

@dataclass
class Preproc:
    scales: List[CoverScale]
    hubs: Dict[Coord, Set[Coord]]         # H(v): set of hubs for vertex v
    dist_vh: Dict[Tuple[Coord,Coord], int]# table d(v,h) for v in V, h in hubs[v]
    kappa: int

class LARCH:
    def __init__(self, W: int, H: int, R0: int = 4, L: int = 3) -> None:
        self.W = W; self.H = H
        self.V = [(x,y) for x in range(W) for y in range(H)]
        self.R0 = R0; self.L = L

    # ---------- cover and nerve ----------

    def _four_phase_centers(self, R: int) -> List[Coord]:
        step = 2*R
        phases = [(0,0), (R,0), (0,R), (R,R)]
        cs: List[Coord] = []
        for dx,dy in phases:
            xs = range(dx, self.W, step)
            ys = range(dy, self.H, step)
            for x in xs:
                for y in ys:
                    cs.append((x,y))
        return cs

    def _ball_vertices(self, c: Coord, R: int) -> List[Coord]:
        cx, cy = c
        verts: List[Coord] = []
        for x in range(max(0, cx-R), min(self.W, cx+R+1)):
            rem = R - abs(x-cx)
            y1 = max(0, cy-rem); y2 = min(self.H-1, cy+rem)
            for y in range(y1, y2+1):
                verts.append((x,y))
        return verts

    def _neighbors(self, centers: List[Coord], R: int) -> Dict[Coord, List[Coord]]:
        adj: Dict[Coord, List[Coord]] = {c:[] for c in centers}
        for i,c in enumerate(centers):
            for j in range(i+1, len(centers)):
                d = manhattan(c, centers[j])
                if d <= 2*R:
                    adj[c].append(centers[j])
                    adj[centers[j]].append(c)
        return adj

    def _portals_overlap(self, U: List[Coord], V: List[Coord], cU: Coord, cV: Coord) -> List[Coord]:
        mx = (cU[0]+cV[0])//2
        my = (cU[1]+cV[1])//2
        cand = [(mx,my), (mx, my+1 if my+1 < self.H else my), (mx, my-1 if my-1 >= 0 else my)]
        S = set(U).intersection(V)
        return [p for p in cand if p in S] or list(S)[:1]

    def build_cover(self) -> List[CoverScale]:
        scales: List[CoverScale] = []
        for li in range(self.L):
            R = self.R0 * (2**li)
            centers = self._four_phase_centers(R)
            balls = {c: self._ball_vertices(c, R) for c in centers}
            neigh = self._neighbors(centers, R)
            portals: Dict[Tuple[Coord,Coord], List[Coord]] = {}
            for c in centers:
                for c2 in neigh[c]:
                    key = (c,c2) if c < c2 else (c2,c)
                    if key in portals: 
                        continue
                    portals[key] = self._portals_overlap(balls[c], balls[c2], c, c2)
            scales.append(CoverScale(R=R, centers=centers, balls=balls, neighbors=neigh, portals=portals))
        return scales

    # ---------- hubs and distances ----------

    def build_hubs(self, scales: List[CoverScale]) -> Tuple[Dict[Coord, Set[Coord]], int]:
        hubs: Dict[Coord, Set[Coord]] = {v:set() for v in self.V}
        for sc in scales:
            for c, verts in sc.balls.items():
                for v in verts:
                    hubs[v].add(c)  # center as a hub
                for c2 in sc.neighbors[c]:
                    key = (c,c2) if c < c2 else (c2,c)
                    for p in sc.portals[key]:
                        for v in sc.balls[c]: hubs[v].add(p)
                        for v in sc.balls[c2]: hubs[v].add(p)
        kappa = max(len(hs) for hs in hubs.values())
        return hubs, kappa

    def precompute(self) -> Preproc:
        scales = self.build_cover()
        hubs, kappa = self.build_hubs(scales)
        dist_vh: Dict[Tuple[Coord,Coord], int] = {}
        for v, Hset in hubs.items():
            for h in Hset:
                dist_vh[(v,h)] = manhattan(v,h)
        return Preproc(scales=scales, hubs=hubs, dist_vh=dist_vh, kappa=kappa)

    # ---------- two-hop and dictionary ----------

    def two_hop_distance(self, pp: Preproc, u: Coord, v: Coord) -> Tuple[int, Coord]:
        Hu = pp.hubs[u]; Hv = pp.hubs[v]
        S = Hu.intersection(Hv)
        if not S:
            sc = pp.scales[-1]
            cu = min(sc.centers, key=lambda c: manhattan(c,u))
            cv = min(sc.centers, key=lambda c: manhattan(c,v))
            S = {cu, cv}
        best = None; hbest = None
        for h in S:
            cand = pp.dist_vh[(u,h)] + pp.dist_vh[(v,h)]
            if best is None or cand < best:
                best = cand; hbest = h
        return int(best), hbest

class O0Dict:
    """Tiny O(0) dictionary for frequent pairs (no concurrency in this reference)."""
    def __init__(self):
        self.store: Dict[Tuple[Coord,Coord], int] = {}

    def get(self, u: Coord, v: Coord) -> Optional[int]:
        return self.store.get((u,v))

    def put(self, u: Coord, v: Coord, dist: int) -> None:
        self.store[(u,v)] = int(dist)
