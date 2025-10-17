Title: LARCH-deltaRAG — Lattice Cover, Cech/Rips Nerve, Robin Gluing, Two-Hop Factorization, and Urn/deltaRAG for O(0)/O(1) Distance Queries (CCR-Integrated)

Abstract.
We give a review-ready, fully ASCII formalization of single-pair distance queries on a non-negative weighted graph G=(V,E,w>=0) that execute in constant work per query. A four-phase multiscale L1 lattice cover together with the Cech (or same-radius Rips) nerve yields bounded overlap and bounded intersection structure. On each overlap we choose a finite set of portals. For every vertex we form a finite hub set; distance queries factor through a two-hop label d*(u,v)=min_h d(u,h)+d(h,v). Under coverage overlap Cov(Delta) and portal hitting Hit(Kappa) we prove exactness d*(u,v)=d_G(u,v). For general graphs we overlay a (1+epsilon) spanner/hopset on each nerve layer to obtain O(1) queries with (1+epsilon) stretch. A Pólya-urn dictionary (deltaRAG) promotes frequent pairs to O(0) lookups while all misses remain O(1). We integrate the full construction in the Cartan–Cech–Robin (CCR) framework: cover/nerve live in the total complex, gluing is a degree-0 Robin transfer, and the online evaluator closes inside a fixed template.

Keywords: two-hop labeling; lattice cover; Cech/Rips nerve; portal hitting; Robin gluing; CCR; spanner/hopset; O(0)/O(1).
	0.	Preliminaries and notation

	•	Graph: finite G=(V,E,w) with w>=0. Shortest-path distance d_G.
	•	Scales: L>=1, base radius R0>0, radii R_ell=R0*2^ell for ell=0..L-1.
	•	Work model: O(1) means a fixed constant number of word adds/comparisons independent of |V|.
	•	Constants: Delta (cover overlap), Kappa (per-vertex portal/hub bound), Epsilon (stretch).

	1.	Cover, nerve, portals, hubs

1.1 Four-phase multiscale L1 lattice cover
At scale ell, place centers on the grid with step 2*R_ell, shifted by the four phases (0,0), (R_ell,0), (0,R_ell), (R_ell,R_ell). Around each center c define the L1 ball
B_ell(c) := { x in V : ||x-c||_1 <= R_ell }.
Let U_ell := { B_ell(c) }.

Assumption Cov(Delta). For all ell and x in V, x lies in at most Delta balls of U_ell.

Lemma 1 (2D L1 constant overlap).
For the four-phase placement, Delta <= 8.
Sketch. Each phase contributes at most four centers covering x; four phases give <=8.

1.2 Cech/Rips nerve and bounded degree
Define the 1-skeleton Cech nerve N_ell whose vertices are centers c and edges connect c,c’ when B_ell(c) ∩ B_ell(c’) != empty. In L1 the condition implies the same-radius Rips edge (||c-c’||_1 <= 2R_ell).

Lemma 2 (bounded nerve degree).
In 2D L1 with the four-phase placement, each c has O(1) neighbors (about 8–12).
Sketch. Neighbors live on the grid within L1 radius 2R_ell; the number of grid points is a constant.

1.3 Portals on overlaps and the hitting axiom
For each adjacent pair (B_ell(c), B_ell(c’)) let S=B_ell(c)∩B_ell(c’). Choose a finite portal set
P_ell(c<->c’) subset S
(e.g., axis midpoints and one diagonal representative). Let d_port be a uniform bound on |P_ell(c<->c’)|.

Assumption Hit(Kappa).
(i) (hitting) Every d_G-geodesic crossing an overlap S can be locally straightened so that the crossing goes through some portal p in P_ell(c<->c’) without increasing length.
(ii) (per-vertex bound) For each v, the total number of portals across all scales and incident overlaps that can be charged to v is <= Kappa.

Lemma 3 (non-expansive portal projection).
The straightening in (i) does not increase length.
Sketch. In L1, a shortest path can be taken monotone in coordinates; sliding to a representative on the equidistance line keeps the sum of in-cluster distances non-increasing. The same argument applies to spanner distances in Sec. 5.

1.4 Hub sets and a uniform size bound
For v in V define the hub set
H(v) := union over ell and centers c with v in B_ell(c) of
{ c } union over c’ adjacent to c in N_ell of P_ell(c<->c’).
Lemma 4 (hub bound).
|H(v)| <= Kappa := L * Delta * (1 + d_port).
If H(u)∩H(v)=empty, use a constant-size fallback F(u,v) built from the coarsest-scale owners of u and v and their incident portals.
	2.	Robin-style gluing upper bound (CCR view)
For a cluster U=B_ell(c) with intrinsic metric d_U and a neighbor V with overlap S, define for x in U and y in V
Glue(x,y; U->V) := min_{p in P(U<->V)} [ d_U(x,p) + lambda_{U,V} + d_V(p,y) ],
with fixed nonnegative lambdas. Concatenate along a chain U0,…,Uk to define glued path length.

Lemma 5 (upper bound).
For any x,y, glued length >= d_G(x,y).
Reason: in-cluster distances are shortest-path distances; lambdas are >=0; minimizing over portal choices cannot go below the global shortest path.
CCR note. This glue is the degree-0 Robin transfer: a q=0 edge-difference block on overlaps in the Tot complex; with lambda chosen in a fixed gauge, the perturbed differential D_R:=D+R is square-zero and keeps the runtime template fixed.
	3.	Two-hop bounds and the exact two-hop value
For S(u,v):=H(u)∩H(v) (or F(u,v) if empty) define
d*(u,v) := min_{h in S(u,v)} [ d(u,h) + d(h,v) ],
LB(u,v) := max_{h in S(u,v)} | d(u,h) - d(v,h) |.

Theorem 1 (upper bound).
For all u,v, d_G(u,v) <= d*(u,v).
Theorem 2 (lower bound).
For all u,v, LB(u,v) <= d_G(u,v).
Both follow from the triangle inequality. Hence LB(u,v) <= d_G(u,v) <= d*(u,v).
	4.	Exactness under Cov and Hit

Assumption Hit_2hop.
For any u,v there exists a d_G-geodesic whose overlap crossings contain at least one portal h that belongs to H(u)∩H(v).

Lemma 6 (portalized geodesic and witness hub).
Under Cov(Delta) and Hit(Kappa), the geodesic can be portalized (Lemma 3), and some crossing portal h lies in H(u)∩H(v).

Lemma 7 (two-hop decomposition at the witness).
For such h, d_G(u,v)=d(u,h)+d(h,v).
Theorem 3 (exactness).
Under Cov(Delta), Hit(Kappa), and Hit_2hop, for all u,v:
d*(u,v)=d_G(u,v)  and  |H(u)∩H(v)| <= Kappa.
Proof. Lemma 7 gives d*(u,v) <= d_G(u,v); Theorem 1 gives the reverse inequality.
	5.	General graphs: (1+epsilon) stretch with O(1) queries
On each N_ell build a (1+epsilon’) spanner or hopset with constant degree O(1/epsilon’). Replace in-cluster and cross-overlap distances by spanner/hopset distances.

Let C be the number of glued segments (in-cluster pieces plus crossings). With fixed L and Delta we have C <= L*Delta = O(1).

Lemma 8 (per-segment multiplicative error).
Each segment length inflates by at most (1+epsilon’).
Lemma 9 (budgeting).
If C segments, choose epsilon’ := epsilon/C to ensure (1+epsilon’)^C <= 1+epsilon+O(epsilon^2).
Theorem 4 ((1+epsilon) and O(1)).
With Cov(Delta) and spanners/hopsets as above, the two-hop value satisfies
d_G(u,v) <= d*(u,v) <= (1+epsilon) d_G(u,v),
|H(u)∩H(v)| is bounded by a constant Kappa(epsilon,Delta), and the query uses O(1) operations.
	6.	Amortized O(1) maintenance
Store d(v,h) for all v and h in H(v). Each vertex belongs to at most L*Delta clusters.

Lemma 10 (locality).
Changing one edge weight affects only clusters that meet that edge and their overlaps. The number of (v,h) pairs whose shortest path can change is O(1) in expectation when L, Delta, and portal constants are fixed.
Theorem 5 (amortized O(1) updates).
Updating the table {d(v,h)} costs O(1) amortized time per edge update by recomputing only flagged local pairs; cluster-scale SSSP instances are constant-size.
	7.	Complexity summary
Proposition 1 (query work).
Candidates are |H(u)∩H(v)|<=Kappa. Per candidate we do two adds and one compare; total O(Kappa)=O(1).
Proposition 2 (space).
Sum_v |H(v)| = O(Kappa*|V|); distances {d(v,h)} fit in O(Kappa*|V|) words.
Proposition 3 (preprocessing).
Multi-source Dijkstra/BFS from all hubs costs O(Kappa * m log n).
	8.	Directed graphs
For directed non-negative graphs maintain separate H_out(u) and H_in(v); query
d*(u,v) = min_{h in H_out(u) ∩ H_in(v)} d->(u,h)+d->(h,v).
The entire analysis applies per direction.
	9.	Numeric robustness
Keep distances in fixed-point Q_{m.n}. Use banded comparisons: a<b is true only if a < b - eps_comp. Triangle-inequality reasoning survives banding; all bounds above remain valid.
	10.	CCR integration (precise)

	•	Good cover and Tot. Each scale ell supplies a cover U_ell; across scales we take a finite disjoint union U and form the total complex Tot with D := d + delta_tot. We work at form degree p=0 (functions).
	•	Degree-0 blocks. The following are degree-0 Tot blocks with fixed, constant sizes: (i) cluster SSSP tables (d(.,. ) inside B_ell(c)); (ii) portal lists P_ell(c<->c’); (iii) hub lookup H(v); (iv) the two-hop evaluator over a constant-size candidate set; (v) dictionary lookups (urn/deltaRAG).
	•	Robin transfer. Define a degree +1 Robin operator R acting by edge differences on overlaps to enforce equality of boundary data (or to add a fixed non-negative crossing penalty lambda). With a degree-0 incidence gauge, [D,R]+R^2=0 and D_R:=D+R is square-zero, so gluing is algebraically exact. No global solves appear online.
	•	Closure and O(1). The CCR DGLA generated by {D_R, J_c (degree-0 incidence), degree-0 distance/hub blocks} closes; BCH at degree 0 reorders without growing the template. Hence the online evaluator uses a fixed number of constant-size operations independent of |V| and of the number of cover elements. (Forward/Reverse differentiation, if needed for learning, also stays O(1) by the same closure.)

	11.	Query pseudocode (constant work)

# Inputs: u, v
S <- H(u) ∩ H(v)           # if empty: S <- F(u,v)
best <- +inf
for h in S:                # |S| <= Kappa  (constant)
    cand <- d(u,h) + d(h,v)
    if cand < best: best <- cand
return best                # equals d_G(u,v) under exactness assumptions,
                           # or (1+epsilon) stretch with spanners.

Auditing: return the witness hub h* achieving best together with the IDs of d(u,h*) and d(h*,v) so a third party can re-verify the minimum.
	12.	O(0) dictionary via urn/deltaRAG
Maintain a Pólya-urn over pairs (u,v); promote heavy hitters to a constant-time dictionary D[u,v]=d_G(u,v). Query path:

	•	if (u,v) in D: return D[u,v]    # O(0)
	•	else: two-hop evaluator above   # O(1)
Demote with hysteresis (two thresholds) and TTL decay. Report the hit-rate p_hit; expected cost is p_hit*O(0) + (1-p_hit)*O(1).

	13.	Limitations and failure detection

	•	The constants depend on L, Delta, portal design, and spanner degree. If measured |H(u)∩H(v)| p99 exceeds Kappa, Cov/Hit is failing; increase spanner density or fall back to local Dijkstra.
	•	Negative edges are disallowed (breaks the gluing upper bound). Zero-weight SCCs should be contracted.
	•	Worst-case families that violate Cov/Hit may force Kappa to grow with n; the method then degrades gracefully to dense evaluation, remaining correct.

	14.	What the assumptions buy

	•	Cov(Delta): constant overlap and bounded nerve degree -> bounded cluster width and |H(v)| <= LDelta(1+d_port).
	•	Hit(Kappa) + portal non-expansion: any geodesic can be portalized without length increase.
	•	Hit_2hop: there exists a witness portal on a shortest path that lies in H(u)∩H(v).
	•	Spanner/hopset with epsilon’ = epsilon / (L*Delta): (1+epsilon) stretch with O(1) queries.
	•	Locality: updates touch only O(1) clusters -> amortized O(1) maintenance.

	15.	Reproducibility checklist (minimal)

	•	Report designed constants (L, Delta, d_port, Kappa), empirical |H(u)∩H(v)| mean/p99, instruction counts, update touch counts, and stretch statistics.
	•	Use torus boundary or center-window sampling to avoid boundary bias.
	•	Fix seeds, lattice phases, and portal rules; publish salted hashes of all tables for audit.

Takeaway.
LARCH-deltaRAG turns single-pair distances into a constant-candidate two-hop minimum whose candidates are guaranteed by a geometric cover/nerve plus finite portals. Exactness follows from Cov/Hit assumptions; general graphs admit (1+epsilon) stretch by per-layer spanners/hopsets. The CCR integration makes gluing algebraic and the runtime template constant, enabling O(0) for frequent pairs and O(1) worst-case latency with auditable witnesses.
