
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Iterable
from collections import defaultdict, Counter
import math, heapq, random
from bisect import bisect_left

# ---------------- Graph ----------------
@dataclass
class Graph:
    n:int
    adj:List[List[Tuple[int,float]]]
    coords:Optional[List[Tuple[int,int]]]=None  # (x,y) for grid graphs
    grid_w:Optional[int]=None
    grid_h:Optional[int]=None
    uniform_unit:bool=False

    @staticmethod
    def empty(n:int)->"Graph":
        return Graph(n=n, adj=[[] for _ in range(n)], coords=None)

    def add_edge(self,u:int,v:int,w:float):
        assert u!=v and w>=0
        self.adj[u].append((v,w)); self.adj[v].append((u,w))

    def dijkstra(self,s:int,targets:Optional[Set[int]]=None,cutoff:Optional[float]=None)->List[float]:
        INF=float("inf"); dist=[INF]*self.n; dist[s]=0.0; pq=[(0.0,s)]
        rem=None if targets is None else set(targets)
        while pq:
            d,u=heapq.heappop(pq)
            if d!=dist[u]: continue
            if cutoff is not None and d>cutoff: break
            if rem is not None and u in rem:
                rem.remove(u)
                if not rem: break
            for v,w in self.adj[u]:
                nd=d+w
                if nd<dist[v]:
                    dist[v]=nd; heapq.heappush(pq,(nd,v))
        return dist

    def manhattan(self,u:int,v:int)->int:
        (x1,y1)=self.coords[u]; (x2,y2)=self.coords[v]
        return abs(x1-x2)+abs(y1-y2)

# ---------------- Config ----------------
@dataclass(frozen=True)
class Config:
    scales:Tuple[float,...]=(3.0,6.0,10.0)
    max_incident_clusters_per_vertex:int=8
    portals_per_intersection:int=1
    kappa_cap:int=128
    seed:int=42
    pairs:int=10000
    urn_topk:int=400          # if <=0, URN is disabled (safe no-op)
    urn_up:int=5
    urn_down:int=2
    urn_ttl:int=200000
    candidate_cap:int=16
    epsilon:float=0.10
    audit_file:Optional[str]=None
    audit_max:int=0
    grid_exact:bool=False     # use 4-phase lattice cover (grids)
    grid_oracle:bool=False    # exact O(1) for uniform unit grid

# ---------------- Cover structures ----------------
@dataclass(frozen=True)
class Cluster:
    center:int; R:float; members:Tuple[int,...]; dist_to_center:Dict[int,float]

@dataclass
class CoverScale:
    R:float
    centers:List[int]
    clusters:Dict[int,Cluster]
    portals:Dict[frozenset,List[int]]=field(default_factory=dict)
    owner_center:Dict[int,int]=field(default_factory=dict)
    spanner_edges:List[Tuple[int,int,float,int]]=field(default_factory=list)

# ---------------- CoverBuilder ----------------
class CoverBuilder:
    def __init__(self,cfg:Config,g:Graph,rng:random.Random):
        self.cfg=cfg; self.g=g; self.rng=rng

    def _rnet_centers(self,R:float)->List[int]:
        order=list(range(self.g.n)); self.rng.shuffle(order)
        chosen=[]; covered=[False]*self.g.n
        for s in order:
            if covered[s]: continue
            chosen.append(s)
            dist=self.g.dijkstra(s,cutoff=R)
            for v,d in enumerate(dist):
                if d<=R: covered[v]=True
        return chosen

    def _cluster_from_center(self,c:int,R:float)->Cluster:
        dist=self.g.dijkstra(c,cutoff=R)
        members=tuple(v for v,d in enumerate(dist) if d<=R)
        return Cluster(center=c,R=R,members=members,dist_to_center={v:dist[v] for v in members})

    def _grid_phase_centers(self,R:int)->List[int]:
        assert self.g.coords is not None and self.g.grid_w is not None and self.g.grid_h is not None
        W=self.g.grid_w; H=self.g.grid_h
        def vid(x,y): return y*W+x
        centers=set()
        step=max(2*R,2)  # spacing is 2R for L1-diamond cover
        for ox in (0,R):
            for oy in (0,R):
                x=ox
                while x<W:
                    y=oy
                    while y<H:
                        centers.add((x,y))
                        y+=step
                    x+=step
        return [vid(x,y) for (x,y) in sorted(centers)]

    def build(self)->List[CoverScale]:
        cover=[]
        for R in self.cfg.scales:
            if self.cfg.grid_exact and self.g.coords is not None and float(R).is_integer():
                centers=self._grid_phase_centers(int(R))
            else:
                centers=self._rnet_centers(R)
            clusters={c:self._cluster_from_center(c,R) for c in centers}
            cs=CoverScale(R=R,centers=centers,clusters=clusters)
            nearest={}
            for c,clu in clusters.items():
                for v,d in clu.dist_to_center.items():
                    cur=nearest.get(v)
                    if cur is None or d<cur[0] or (d==cur[0] and c<cur[1]):
                        nearest[v]=(d,c)
            cs.owner_center={v:c for v,(_,c) in nearest.items()}
            cover.append(cs)

        # portals per scale
        for cs in cover:
            cs.portals={}
            if self.cfg.grid_exact and self.g.coords is not None and float(cs.R).is_integer():
                W=self.g.grid_w; H=self.g.grid_h
                def vid(x,y): return y*W+x
                step=max(2*int(cs.R),2)
                centers=set(cs.centers)
                for c in cs.centers:
                    x0,y0=self.g.coords[c]
                    for dx,dy in ((step,0),(-step,0),(0,step),(0,-step)):
                        nx,ny=x0+dx,y0+dy
                        if 0<=nx<W and 0<=ny<H:
                            nb=vid(nx,ny)
                            if nb in centers:
                                A,B=cs.clusters[c],cs.clusters[nb]
                                inter=set(A.members).intersection(B.members)
                                if not inter: continue
                                mx, my = (x0+nx)//2, (y0+ny)//2
                                p = vid(mx,my)
                                if p in inter:
                                    cs.portals[frozenset((c,nb))]=[p]
                                else:
                                    best=None; bestp=None
                                    for x in inter:
                                        da=A.dist_to_center.get(x,math.inf)
                                        db=B.dist_to_center.get(x,math.inf)
                                        s=da+db
                                        if best is None or s<best:
                                            best=s; bestp=x
                                    if bestp is not None:
                                        cs.portals[frozenset((c,nb))]=[bestp]
            else:
                P=self.cfg.portals_per_intersection
                memberships:Dict[int,List[int]]=defaultdict(list)
                for c,clu in cs.clusters.items():
                    for v in clu.members: memberships[v].append(c)
                seen=set()
                for u in range(self.g.n):
                    for v,_ in self.g.adj[u]:
                        for cu in memberships.get(u,()):
                            for cv in memberships.get(v,()):
                                if cu==cv: continue
                                key=frozenset((cu,cv))
                                if key in seen: continue
                                seen.add(key)
                                A,B=cs.clusters[cu],cs.clusters[cv]
                                inter=set(A.members).intersection(B.members)
                                if not inter: continue
                                scored=[]
                                for x in inter:
                                    da=A.dist_to_center.get(x,math.inf)
                                    db=B.dist_to_center.get(x,math.inf)
                                    if math.isfinite(da) and math.isfinite(db):
                                        scored.append((da+db,x))
                                scored.sort()
                                if scored:
                                    cs.portals[key]=[x for _,x in scored[:P]]
        return cover

# ---------------- Nerve Spanner (optional) ----------------
class NerveSpanner:
    def __init__(self,epsilon:float): self.t=1.0+max(0.0,float(epsilon))
    @staticmethod
    def _dij(n:int,adjS:Dict[int,List[Tuple[int,float]]],s:int,t:int)->float:
        INF=float("inf"); dist=[INF]*n; dist[s]=0.0; pq=[(0.0,s)]
        while pq:
            d,u=heapq.heappop(pq)
            if d!=dist[u]: continue
            if u==t: return d
            for v,w in adjS.get(u,()):
                nd=d+w
                if nd<dist[v]: dist[v]=nd; heapq.heappush(pq,(nd,v))
        return dist[t]
    def prune(self,cs:CoverScale)->None:
        centers=sorted(cs.centers); idx={c:i for i,c in enumerate(centers)}; n=len(centers)
        edges=[]
        for key,plist in cs.portals.items():
            a,b=tuple(sorted(list(key))); A,B=cs.clusters[a],cs.clusters[b]
            best_w=math.inf; best_p=None
            for p in plist:
                da=A.dist_to_center.get(p,math.inf); db=B.dist_to_center.get(p,math.inf)
                if math.isfinite(da) and math.isfinite(db):
                    w=da+db
                    if w<best_w: best_w=w; best_p=p
            if best_p is not None: edges.append((best_w,a,b,best_p))
        edges.sort(key=lambda t:(t[0],t[1],t[2],t[3]))
        adjS:Dict[int,List[Tuple[int,float]]]=defaultdict(list)
        kept={}
        for w,a,b,p in edges:
            ia,ib=idx[a],idx[b]
            cur=self._dij(n,adjS,ia,ib)
            if not math.isfinite(cur) or cur>self.t*w:
                adjS[ia].append((ib,w)); adjS[ib].append((ia,w))
                kept[frozenset((a,b))]=[p]
        cs.portals=kept
        cs.spanner_edges=[(a,b,w,p) for (w,a,b,p) in edges if frozenset((a,b)) in kept]

# ---------------- HubBuilder ----------------
class HubBuilder:
    def __init__(self,cfg:Config,g:Graph): self.cfg=cfg; self.g=g

    def build_sets(self,cover:List[CoverScale])->Dict[int,List[int]]:
        H:Dict[int,List[int]]=defaultdict(list); cap=self.cfg.kappa_cap
        for cs in cover:
            owner=cs.owner_center
            portals_by_center:Dict[int,List[int]]=defaultdict(list)
            for key,plist in cs.portals.items():
                a,b=tuple(key); portals_by_center[a].extend(plist); portals_by_center[b].extend(plist)
            for v in range(self.g.n):
                c=owner.get(v)
                hubs=[]
                if c is not None:
                    hubs.append(c)
                    hubs.extend(portals_by_center.get(c,()))
                seen=set(); out=[]
                for h in hubs:
                    if h not in seen:
                        seen.add(h); out.append(h)
                    if len(out)>=cap: break
                if out:
                    H[v]=out
        return {v:sorted(H[v]) for v in H}

    @staticmethod
    def _mix64(x:int)->int:
        x=(x+0x9E3779B97F4A7C15)&0xFFFFFFFFFFFFFFFF
        x=(x^(x>>30))*0xBF58476D1CE4E5B9&0xFFFFFFFFFFFFFFFF
        x=(x^(x>>27))*0x94D049BB133111EB&0xFFFFFFFFFFFFFFFF
        return (x^(x>>31))&0xFFFFFFFFFFFFFFFF

    def _fingerprint(self,hubs:List[int])->Tuple[int,...]:
        M=32; words=[0]*M
        for h in hubs:
            for mul in (0x9E37,0xC2B2,0x1657):
                k=self._mix64(h*mul); idx=(k>>6)&0x1F; bit=k&63; words[idx]|=(1<<bit)
        return tuple(words)

    def encode(self,H_sets:Dict[int,List[int]]):
        small={}; fp={}
        for v,arr in H_sets.items():
            t=tuple(arr[: self.cfg.kappa_cap]); small[v]=t; fp[v]=self._fingerprint(list(t))
        return small,fp

# ---------------- Urn & Query ----------------
class UrnCache:
    def __init__(self,topk:int,up:int,down:int,ttl:int):
        self.topk=int(topk); self.up=int(up); self.down=int(down); self.ttl=int(ttl)
        self.counts:Counter=Counter(); self.cache:Dict[Tuple[int,int],Tuple[float,int]]={}; self.clock=0
    @staticmethod
    def _key(u:int,v:int)->Tuple[int,int]: return (u,v) if u<=v else (v,u)
    def get(self,u:int,v:int)->Optional[float]:
        self.clock+=1; k=self._key(u,v)
        if k in self.cache:
            val,_=self.cache[k]; self.cache[k]=(val,self.clock); return val
        return None
    def observe_and_promote(self,u:int,v:int,value:float)->None:
        # Safe no-op when topk<=0 (URN disabled)
        if self.topk<=0: return
        self.clock+=1; k=self._key(u,v); self.counts[k]+=1
        if self.ttl and (self.clock & 0x3FF)==0:
            stale=[kk for kk,(_,last) in self.cache.items() if self.clock-last>self.ttl]
            for kk in stale: del self.cache[kk]
        if k in self.cache: return
        if len(self.cache)<self.topk or self.counts[k]>=self.up:
            if len(self.cache)>=self.topk:
                least=min(self.cache.keys(), key=lambda t:(self.counts[t], self.cache[t][1]))
                if self.counts[least]<=self.down: del self.cache[least]
                else: return
            self.cache[k]=(value,self.clock)

class QueryEngine:
    def __init__(self,cfg:Config,g:Graph,cover:List[CoverScale],H_small:Dict[int,Tuple[int,...]],H_fp:Dict[int,Tuple[int,...]],DVH:Dict[int,Dict[int,float]],urn:UrnCache):
        self.cfg=cfg; self.g=g; self.cover=cover; self.small=H_small; self.fp=H_fp; self.DVH=DVH; self.urn=urn; self._cand_cap=cfg.candidate_cap
    @staticmethod
    def _member_sorted(arr:Tuple[int,...],x:int)->bool:
        i=bisect_left(arr,x); return i!=len(arr) and arr[i]==x

    def _bloom_candidates(self,hubs_u:Tuple[int,...], hubs_v_sorted:Tuple[int,...], fp_v:Tuple[int,...])->List[int]:
        cand=[]
        def _mix64(x:int)->int:
            x=(x+0x9E3779B97F4A7C15)&0xFFFFFFFFFFFFFFFF
            x=(x^(x>>30))*0xBF58476D1CE4E5B9&0xFFFFFFFFFFFFFFFF
            x=(x^(x>>27))*0x94D049BB133111EB&0xFFFFFFFFFFFFFFFF
            return (x^(x>>31))&0xFFFFFFFFFFFFFFFF
        for h in hubs_u:
            k0=_mix64(h*0x9E37); k1=_mix64(h*0xC2B2); k2=_mix64(h*0x1657)
            ok=True
            for k in (k0,k1,k2):
                idx=(k>>6)&0x1F; bit=k&63
                if (fp_v[idx]>>bit)&1==0: ok=False; break
            if ok and self._member_sorted(hubs_v_sorted,h): cand.append(h)
        return cand

    def _fallback_candidates(self,u:int,v:int)->List[int]:
        cand=[]
        if self.cover:
            cs=self.cover[-1]; cu=cs.owner_center.get(u); cv=cs.owner_center.get(v)
            if cu is not None: cand.append(cu)
            if cv is not None and cv!=cu: cand.append(cv)
            for key,ps in cs.portals.items():
                if cu is not None and cu in key and ps: cand.append(ps[0]); break
            for key,ps in cs.portals.items():
                if cv is not None and cv in key and ps: cand.append(ps[0]); break
        out=[]; seen=set()
        for h in cand:
            if h not in seen: seen.add(h); out.append(h)
        return out[: max(2,self._cand_cap)]

    def query(self,u:int,v:int)->Tuple[float,int,bool,bool,bool,bool]:
        if u==v: return 0.0,1,False,False,True,True
        # Grid oracle path: exact O(1) for uniform unit grid
        if self.cfg.grid_oracle and self.g.uniform_unit and self.g.coords is not None:
            d=float(self.g.manhattan(u,v))
            return d, 1, True, False, True, True

        cached=self.urn.get(u,v)
        if cached is not None: return cached,0,True,False,True,True
        hubs_u=self.small.get(u,()); hubs_v=self.small.get(v,()); mask_hit=False
        if hubs_u and hubs_v: mask_hit=any((a & b) for a,b in zip(self.fp[u], self.fp[v]))
        inter_nonempty=False; used_fallback=False
        if mask_hit:
            cand=self._bloom_candidates(hubs_u,hubs_v,self.fp[v]); inter_nonempty=bool(cand)
        else: cand=[]
        if not cand: cand=self._fallback_candidates(u,v); used_fallback=True
        if len(cand)>self._cand_cap: cand=cand[:self._cand_cap]
        best=float("inf"); k_eval=0
        for h in cand:
            du=self.DVH.get(u,{}).get(h,float("inf")); dv=self.DVH.get(v,{}).get(h,float("inf"))
            d=du+dv; k_eval+=1
            if d<best: best=d
        if not math.isfinite(best): best=float("inf")
        self.urn.observe_and_promote(u,v,best)
        return best,k_eval,False,used_fallback,mask_hit,inter_nonempty

    # Lower bound using shared hubs: LB = max_h |d(u,h)-d(v,h)|
    def query_bounds(self,u:int,v:int)->Tuple[float,float,int]:
        ub,k,*_ = self.query(u,v)
        hubs_u = self.small.get(u, ()); hubs_v = self.small.get(v, ())
        lb = 0.0
        if hubs_u and hubs_v:
            # compute intersection efficiently (both sorted tuples)
            i=j=0
            while i<len(hubs_u) and j<len(hubs_v):
                if hubs_u[i]==hubs_v[j]:
                    h=hubs_u[i]
                    du=self.DVH.get(u,{}).get(h,float("inf"))
                    dv=self.DVH.get(v,{}).get(h,float("inf"))
                    if math.isfinite(du) and math.isfinite(dv):
                        diff=abs(du-dv); 
                        if diff>lb: lb=diff
                    i+=1; j+=1
                elif hubs_u[i]<hubs_v[j]: i+=1
                else: j+=1
        return ub, lb, k

# ---------------- Precompute DVH ----------------
def precompute_DVH(g:Graph,H_small:Dict[int,Tuple[int,...]])->Dict[int,Dict[int,float]]:
    targets:Dict[int,Set[int]]=defaultdict(set)
    for v,hubs in H_small.items():
        for h in hubs: targets[h].add(v)
    DVH:Dict[int,Dict[int,float]]=defaultdict(dict)
    if g.uniform_unit and g.coords is not None:
        for h,Vs in targets.items():
            hx,hy=g.coords[h]
            for v in Vs:
                vx,vy=g.coords[v]
                DVH[v][h]=abs(hx-vx)+abs(hy-vy)
        return DVH
    for h,Vs in targets.items():
        dist=g.dijkstra(h,targets=Vs)
        for v in Vs:
            d=dist[v]
            if math.isfinite(d): DVH[v][h]=d
    return DVH

# ---------------- Evaluator ----------------
@dataclass
class Stats:
    n:int; scales:Tuple[float,...]; kappa_cap:int; avg_hubs_per_vertex:float; max_hubs_per_vertex:int
    avg_candidates_evaluated:float; p99_candidates_evaluated:float; avg_ops:float; p99_ops:float
    exact_rate:float; mean_stretch:float; p99_stretch:float; pairs:int; elapsed_ms:float

class Evaluator:
    def __init__(self,cfg:Config,g:Graph):
        self.cfg=cfg; self.g=g; self.rng=random.Random(cfg.seed)
        self.cover=CoverBuilder(cfg,g,self.rng).build()
        if self.cfg.epsilon and self.cfg.epsilon>0.0:
            sp=NerveSpanner(self.cfg.epsilon)
            for cs in self.cover: sp.prune(cs)
        H_sets=HubBuilder(cfg,g).build_sets(self.cover)
        self.avg_hubs=sum(len(s) for s in H_sets.values())/len(H_sets) if H_sets else 0.0
        self.max_hubs=max((len(s) for s in H_sets.values()), default=0)
        self.H_small,self.H_fp=HubBuilder(cfg,g).encode(H_sets)
        self.DVH=precompute_DVH(g,self.H_small)
        self.urn=UrnCache(cfg.urn_topk,cfg.urn_up,cfg.urn_down,cfg.urn_ttl)
    def _pairs_uniform(self,num_pairs:int)->Iterable[Tuple[int,int]]:
        for _ in range(num_pairs):
            u=self.rng.randrange(self.g.n); v=self.rng.randrange(self.g.n); yield (u,v)
    def _true_dist(self,u:int,v:int)->float:
        dist=self.g.dijkstra(u,targets={v}); return dist[v]
    def run_uniform(self,num_pairs:int)->Stats:
        eng=QueryEngine(self.cfg,self.g,self.cover,self.H_small,self.H_fp,self.DVH,self.urn)
        import time as _t
        t0=_t.perf_counter()
        k_evals=[]; stretches=[]; exact=0; tot=0
        for u,v in self._pairs_uniform(num_pairs):
            res = eng.query(u,v)
            est = res[0]
            k = res[1]  # type: ignore
            tru=self._true_dist(u,v)
            if math.isfinite(est) and math.isfinite(tru) and tru>0:
                s=est/tru; stretches.append(s); exact+=int(abs(est-tru)<1e-12)
            k_evals.append(k); tot+=1
        elapsed_ms=(_t.perf_counter()-t0)*1000.0
        ks=sorted(k_evals); p99k=ks[int(0.99*len(ks))] if ks else 0
        svals=[s for s in stretches if math.isfinite(s)]
        mean_st=sum(svals)/len(svals) if svals else float("nan")
        p99_st=sorted(svals)[int(0.99*len(svals))] if svals else float("nan")
        return Stats(n=self.g.n,scales=self.cfg.scales,kappa_cap=self.cfg.kappa_cap,
                     avg_hubs_per_vertex=self.avg_hubs,max_hubs_per_vertex=self.max_hubs,
                     avg_candidates_evaluated=sum(k_evals)/len(k_evals) if k_evals else 0.0,
                     p99_candidates_evaluated=p99k, avg_ops=3.0*(sum(k_evals)/len(k_evals) if k_evals else 0.0),
                     p99_ops=3.0*p99k, exact_rate=exact/max(tot,1), mean_stretch=mean_st, p99_stretch=p99_st,
                     pairs=tot, elapsed_ms=elapsed_ms)

# -------------- Generators --------------
def gen_grid(w:int,h:int,wmin:float,wmax:float,seed:int)->Graph:
    rng=random.Random(seed); n=w*h; g=Graph.empty(n)
    g.grid_w=w; g.grid_h=h; coords=[]
    def vid(x,y): return y*w+x
    uniform = (abs(wmin-wmax)<1e-12 and abs(wmin-1.0)<1e-12)
    for y in range(h):
        for x in range(w):
            coords.append((x,y))
            if x+1<w:
                g.add_edge(vid(x,y), vid(x+1,y), rng.uniform(wmin,wmax))
            if y+1<h:
                g.add_edge(vid(x,y), vid(x,y+1), rng.uniform(wmin,wmax))
    g.coords=coords; g.uniform_unit=uniform
    return g

def gen_rgg(n:int,radius:float,wmin:float,wmax:float,seed:int)->Graph:
    rng=random.Random(seed); pts=[(rng.random(),rng.random()) for _ in range(n)]
    g=Graph.empty(n)
    for i in range(n):
        xi,yi=pts[i]
        for j in range(i+1,n):
            xj,yj=pts[j]; dx=xi-xj; dy=yi-yj; d=(dx*dx+dy*dy)**0.5
            if d<=radius: g.add_edge(i,j,rng.uniform(wmin,wmax))
    return g

def gen_ba(n:int,m:int,wmin:float,wmax:float,seed:int)->Graph:
    rng=random.Random(seed); g=Graph.empty(n)
    for i in range(m+1):
        for j in range(i+1,m+1):
            g.add_edge(i,j,rng.uniform(wmin,wmax))
    deg=[len(g.adj[i]) for i in range(m+1)]
    for v in range(m+1,n):
        targets=set()
        while len(targets)<m:
            x=rng.randrange(v)
            p=(deg[x]+1)/(sum(deg[:v])+v)
            if rng.random()<p: targets.add(x)
        for t in targets: g.add_edge(v,t,rng.uniform(wmin,wmax))
        deg.append(len(g.adj[v]))
    return g
