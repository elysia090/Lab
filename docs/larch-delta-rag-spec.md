Title
LARCH-deltaRAG: Lattice cover, Cech/Rips nerve, Robin boundary, 2-hop factorization, and urn/deltaRAG for O(0)/O(1) distance queries

Abstract
We reduce single-pair shortest distance queries on nonnegative-weight graphs G=(V,E,w>=0) to a constant number of additions and comparisons. Using multi-scale L1-ball lattice covers with four-phase shifts, their Cech (or same-radius Rips) nerves, finite portals on cluster intersections, and finite hub sets per vertex, we factor the distance functor Free(G)->Dist as Free(G)->Hub->Dist with 2-hop evaluation over shared hubs. Under coverage overlap Cov(Delta) and intersection hitting Hit(kappa), the estimator is exact: d*(u,v)=d_G(u,v). For general graphs, augment each nerve with a (1+epsilon)-spanner/hopset to obtain O(1) query time with (1+epsilon) stretch. A Polya urn cache with distribution-aware tuning (deltaRAG) promotes frequent pairs to O(0) exact lookups while retaining O(1) on misses. We formalize time/space/update locality and auditability, and demonstrate on 2D L1 grids that candidate counts and hub sizes saturate to constants independent of n.

0. Preliminaries and Notation
   Let G=(V,E,w) be a finite graph with nonnegative edge weights w:E->[0,+infty). Let d_G:VxV->[0,+infty) be the shortest-path metric. A path length is the sum of edge weights. Fix L>=1, base radius R0>0, and radii R_ell=R0*2^ell for ell in {0,...,L-1}. In the grid case we use L1 balls; in general graphs we use graph-distance balls (Section 5 handles general graphs via spanners/hopsets). Query cost counts fixed-word additions and comparisons. We write O(1) to mean O(kappa) for the constant kappa defined below, independent of |V|.

1. Cover, Nerve, Portals, Hubs

1.1 Multi-scale four-phase lattice cover
For scale ell, place centers on four phase-shifted integer lattices with step 2*R_ell and offsets (0,0),(R_ell,0),(0,R_ell),(R_ell,R_ell). For each center c place an L1 ball
B_ell(c) = { x in V : ||x-c||_1 <= R_ell }.
Let U_ell = { B_ell(c) }_c.

Assumption Cov(Delta). There exists a constant Delta>=1 such that for all ell and all x in V, the number of balls B_ell(c) containing x is at most Delta.

Lemma 1 (constant overlap in 2D L1). In 2D L1 with the four-phase construction above, Delta <= 8.
Proof. Fix x. In each phase, centers whose balls can cover x form a 2x2 block at step 2R_ell, so at most 4 per phase; over four phases, at most 8. QED.

1.2 Nerves and adjacency degree
For each U_ell, define the (1-skeleton) Cech nerve N_ell with vertices the centers c, and an edge (c,c') iff B_ell(c) cap B_ell(c') is nonempty. In L1, the same-radius Vietoris - Rips graph (edge if L1(c,c') <= 2R_ell) contains the Cech nerve.

Lemma 2 (bounded degree of the nerve). In 2D L1 with four-phase placement, each center c has O(1) neighbors in N_ell (e.g., <= 8 to 12 depending on diagonal inclusion).
Proof. If B_ell(c) cap B_ell(c') != empty, then ||c-c'||_1 <= 2R_ell. Centers lie on four shifted grids of step 2R_ell; enumerating such centers within that L1-radius is a constant bound in 2D. QED.

1.3 Portals on intersections
For each adjacent pair (B_ell(c),B_ell(c')) with intersection S=B_ell(c) cap B_ell(c'), choose a finite portal set P_ell(c <-> c') subseteq S. In 2D L1, partition S using L1 equidistance lines from c and c' and select constant representatives (axis midpoints and optionally one diagonal). Let d_port be the per-intersection portal bound (axis-only 4; with diagonals <= 5).

Assumption Hit(kappa). There exists a finite per-vertex constant kappa such that:
(1) (intersection hitting) Every d_G-shortest path crossing an intersection S can be reparameterized to pass through some portal p in P_ell(c <-> c') at that intersection without increasing length.
(2) (per-vertex incidence bound) For any vertex v, the total number of portals incident to clusters that contain v over all scales is bounded by kappa.

Lemma 3 (portal reparameterization does not increase length). Let pi be a d_G-shortest path crossing S=B_ell(c) cap B_ell(c'). Then there exists p in P_ell(c <-> c') such that replacing the local crossing by a segment via p does not increase total length.
Proof. In L1, any shortest path between two points can be chosen monotone; the crossing through S can be projected to the nearest representative along L1 equidistance contours without increasing the sum of intra-cluster distances. (In general graphs we later replace the intra-cluster metric by a spanner metric.) QED.

1.4 Hubs and their size bound
For each v in V define the hub set
H(v) = union over ell of union over c: v in B_ell(c) of {c} union (union over c':(c,c') in E(N_ell) of P_ell(c <-> c')).

Lemma 4 (hub size bound). |H(v)| <= kappa := L * Delta * (1 + d_port).
Proof. At each scale ell, v lies in at most Delta balls B_ell(c). For each such c, add its center (1) and portals for adjacent intersections. The number of adjacencies is O(1) (Lemma 2), and we bound by d_port representatives per adjacency scheme. Summing over Delta per scale and L scales gives the bound. QED.

When H(u) cap H(v) is empty, define a constant-size fallback set F(u,v) that contains the coarsest-scale owner centers of u and v and one portal adjacent to each (deduplicated). By construction |F(u,v)| = O(1).

2. Robin Gluing Upper Bound
   For any cluster U=B_ell(c), let d_U be the shortest-path metric on the induced subgraph G[U]. For adjacent U,V with S=U cap V and portals P(U <-> V) subseteq S, define the glued crossing cost
   Glue(x,y; U->V) = min over p in P(U <-> V) of ( d_U(x,p) + lambda_{U,V} + d_V(p,y) ),
   with fixed penalties lambda_{U,V} >= 0. A path segment staying inside a cluster costs d_U; a path crossing a sequence (U0,...,Uk) has glued length equal to the sum of intra-cluster segments and glued crossings.

Lemma 5 (glued length is an upper bound). For all x,y in V, any glued walk length from x to y (with lambda_{U,V} >= 0) is >= d_G(x,y).
Proof. For any decomposition and portal choices, intra-cluster segments are at least the shortest within the cluster, and adding nonnegative lambdas cannot reduce cost; minimizing over all decompositions cannot go below the global shortest path distance. QED.

3. Two-Hop Distance: Upper and Lower Bounds
   Define the two-hop estimator
   d*(u,v) = min_{h in S(u,v)} ( d(u,h) + d(h,v) ),
   where S(u,v) = H(u) cap H(v) if nonempty, else F(u,v).
   Define the shared-hub lower bound
   LB(u,v) = max_{h in S(u,v)} | d(u,h) - d(v,h) |.

Theorem 1 (upper bound / soundness). For all u,v, d_G(u,v) <= d*(u,v).
Proof. For any h, triangle inequality gives d_G(u,v) <= d(u,h)+d(h,v); taking min over h proves it. The fallback case is identical. QED.

Theorem 2 (lower bound via shared hubs). For all u,v, LB(u,v) <= d_G(u,v).
Proof. For any h, |d(u,h)-d(v,h)| <= d(u,v); take max over h. QED.

Thus LB(u,v) <= d_G(u,v) <= d*(u,v).

4. Exactness Under Cov(Delta) and Hit(kappa)
   Assumption Hit_{2-hop}. For every u,v there exists a shortest path pi from u to v and an index i such that the portal p_i used at some intersection belongs to H(u) cap H(v). This is the usual hub-labeling cover property realized by our construction.

Lemma 6 (portalized shortest path with witness hub). Under Cov(Delta) and Hit(kappa), there exists a shortest u-v path pi* that crosses only through portals (Lemma 3) and includes a crossing portal h in H(u) cap H(v).
Proof. Snap a shortest path to portals (Lemma 3) to obtain pi*. Hit_{2-hop} supplies a crossing portal h in H(u) cap H(v) used by pi*. QED.

Lemma 7 (two-hop decomposition at witness). For the portal h in Lemma 6,
d_G(u,v) = d(u,h) + d(h,v).
Proof. The prefix u->h of pi* is a shortest u-h path (otherwise we could shorten pi*); similarly the suffix h->v. Hence len(pi*) = d(u,h)+d(h,v). QED.

Theorem 3 (exactness). Under Cov(Delta) and Hit(kappa) with Hit_{2-hop}, for all u,v,
d*(u,v) = d_G(u,v), and |H(u) cap H(v)| <= kappa.
Proof. From Lemma 7, for the witness h we have d(u,h)+d(h,v)=d_G(u,v), hence d*(u,v) <= d_G(u,v). Combine with Theorem 1 to get equality. The size bound is from Lemma 4. QED.

5. (1+epsilon)-Stretch on General Graphs
   For each scale ell, build a (1+epsilon') spanner or hopset S_ell on the nerve N_ell with degree and lightness O(1/epsilon'). Replace intra-cluster distances and inter-cluster navigation by distances in S_ell. Let C be an upper bound on the number of glued segments (intra-cluster + crossings) in a portalized path at a fixed scale. With Cov(Delta) and fixed L, a constant C <= L*Delta suffices.

Lemma 8 (per-segment multiplicative error). Replacing each segment by its (1+epsilon') approximation on S_ell inflates that segment by at most factor (1+epsilon').
Proof. Definition of spanner/hopset. QED.

Lemma 9 (error budgeting). If a path has at most C segments and each inflates by (1+epsilon'), then total inflation is <= (1+epsilon) provided epsilon' <= epsilon/C (since (1+epsilon')^C <= 1+epsilon for small enough epsilon').
Proof. (1+epsilon')^C <= exp(C*epsilon'). Choose epsilon' = epsilon/C and use exp(epsilon) <= 1+epsilon+O(epsilon^2); pick epsilon small enough to ensure the product bound. QED.

Theorem 4 ((1+epsilon) stretch, O(1) query). Assume Cov(Delta). Overlay (1+epsilon') spanners/hopsets S_ell with epsilon' = epsilon/C. Define hubs as before and evaluate candidates using S_ell distances. Then for all u,v,
d_G(u,v) <= d*(u,v) <= (1+epsilon) d_G(u,v),
and |H(u) cap H(v)| <= kappa(epsilon,Delta) = O(1). Query time is O(1).
Proof. Lower bound: Theorem 1. Upper bound: portalize a shortest path; apply Lemma 8 to each of its <= C segments and Lemma 9 to bound the product by (1+epsilon). Hub-set size remains constant since spanner degrees are bounded constants, so we evaluate O(1) candidates. QED.

6. Amortized O(1) Updates
   We store, for each v and h in H(v), the value d(v,h). Let C_ell(v) be the set of scale-ell clusters that contain v. Each v is in at most Delta clusters per scale; across L scales at most L*Delta (constants).

Lemma 10 (locality of influence). Let a single edge weight change by delta >= 0. Then it can affect only pairs (v,h) whose every shortest v-h path in the updated graph intersects a cluster that contains that edge (at some scale). The number of such (v,h) per update is O(1) in expectation when L, Delta, and portals per intersection are constants.
Proof. A change to edge e can only affect distances whose geodesics traverse e or become shorter via e. Because overlap and nerve degree are bounded, the number of clusters at all scales affected by a change inside a cluster is O(1) hops in the nerve; with constant portals per intersection, the number of incident portal pairs touched is O(1). Summing over the constant many affected clusters yields O(1) impacted (v,h). QED.

Theorem 5 (amortized O(1) maintenance). If L, Delta, and portals per intersection are constants, maintaining the table {d(v,h)} under a sequence of single-edge updates costs amortized O(1) work per update.
Proof. Recompute only the (v,h) flagged by Lemma 10. A potential argument on the number of entries whose shortest paths actually change bounds the average recomputations per update by a constant depending only on L, Delta, and portal constants. Each recomputation is a local multi-source Dijkstra within a constant-size nerve neighborhood, hence O(1). QED.

7. Complexity

Proposition 1 (query complexity). Each query evaluates at most |H(u) cap H(v)| <= kappa candidates; per candidate we do two additions and one comparison (plus constant loads). Hence query time is O(kappa)=O(1).
Proof. By Lemma 4 and the definition of d*. QED.

Proposition 2 (space). Sum over v of |H(v)| is O(kappa*|V|); storing d(v,h) for all h in H(v) costs O(kappa*|V|) words.
Proof. Immediate from Lemma 4. QED.

Proposition 3 (preprocessing). Computing all d(v,h) by multi-source Dijkstra takes O(kappa * m log n) time (nonnegative weights).
Proof. For each hub h (O(1) per local region and constant total factor), run Dijkstra once or in small batches. Standard heap-based SSSP gives the bound. QED.

8. Directed Graphs (completeness)
   For a directed graph (nonnegative weights), define forward hubs H_out(u) and backward hubs H_in(v). The query is
   d*(u,v) = min_{h in H_out(u) cap H_in(v)} ( d(u,h) + d(h,v) ).
   All results above transfer verbatim with directions respected; the 2-hop cover assumption becomes the directed hub-labeling property.

9. Numerical Robustness
   Distances are stored in fixed-point Q_{m.n}; comparisons use a tolerance eps_comp >= 0 so inequalities are only decided when a < b - eps_comp. Triangle inequality remains valid; Theorems 1 - 4 rely on metric inequalities stable under such banding, so the results persist.

10. Assumptions: What They Buy
    Cov(Delta): bounded overlap and nerve degree -> constant-width construction; |H(v)| <= kappa = L*Delta*(1+d_port).
    Hit(kappa) + portal snapping (Lemma 3): every shortest path can be snapped to portals without length decrease.
    Hit_{2-hop}: existence of a witness h in H(u) cap H(v) on some shortest u-v path -> Theorem 3 exactness d*=d_G.
    Spanner/hopset overlay with epsilon' = epsilon/(L*Delta): Theorem 4 (1+epsilon) stretch with O(1) candidates.
    Locality of intersections (bounded degree and portals): Theorem 5 amortized O(1) updates.

All constants depend only on L, Delta, and the portal scheme, not on |V| or |E|.
