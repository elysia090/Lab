Title: Zero-Time Distance Queries: Necessary and Sufficient Conditions via Cech Nerves, Incidence Moebius Calculus, and Cartan–Cech–Robin Gluing

Abstract.
We characterize when single-pair distance queries on finite nonnegative weighted graphs admit zero-time answers (O(0)) for a maintained subset of pairs and constant-time answers (O(1)) for all pairs. The evaluator is a two-hop distance scheme built on a bounded-overlap Cech nerve with finite portal sets and finite vertex hub sets. We axiomatize the cover, nerve, portals, hubs, and the Cartan–Cech–Robin (CCR) gluing operator, and we use the incidence algebra with Moebius inversion to certify constant-time maintenance under local updates. The main results are equivalences: (i) O(1) evaluability for all pairs holds if and only if the candidate set size is uniformly bounded by a constant kappa determined by cover parameters; (ii) exact answers for all pairs hold if and only if every pair admits a portalized geodesic that meets a witness hub; (iii) constancy of the online evaluation scheme under cover refinement and degree-0 incidence gauge holds if and only if the CCR Maurer–Cartan and Cartan identities hold; (iv) O(1) maintenance under local updates holds if and only if the overlap poset has bounded height so that Moebius inversion has finite support per leg; (v) persistent O(0) for any maintained pair holds if and only if a margin inequality is satisfied at every update. We give O(1) guard checks, constant-size audit objects, and standard probabilistic guarantees (via a Polya urn) for selecting frequently requested pairs.
	1.	Language, structures, and standing assumptions

1.1 Language L.
Sorts: V (vertices), O (overlaps), S (scales), R (nonnegative reals).
Symbols:
B(s,c) subset V            cluster at scale s and center c
Overlap o subset V          each o = B(s,c) cap B(s,c’)
Adj_s(c,c’)                 cluster-center adjacency at scale s
P(o) subset Overlap(o)      finite portal set
H: V -> Fin(V)              hub map v |-> finite set H(v)
d: V x V -> R               shortest-path distance, triangle inequality, d(x,x)=0
Prom(u,v)                   maintained-pair predicate
BDelta: O -> R              per-overlap nonnegative update bound
Parameters: Delta, kappa, L, d_port, gamma in R_{\ge 0}

1.2 Structures.
An L-structure interprets the above on a finite nonnegative weighted graph G=(V,E,w>=0) with a finite good cover U at scales S={0,…,L-1}. Distances live in the Lawvere quantale ([0,inf], +, 0, >=).
	2.	Axioms

T_core (bounded cover and nerve).
(A1) For all v and s, |{ c : v in B(s,c) }| <= Delta.
(A2) For all s,c, |{ c’ : Adj_s(c,c’) }| = O(1).
(A3) For all overlaps o, |P(o)| <= d_port.

T_kappa (two-hop scaffold).
(A4) For all v, |H(v)| <= kappa := L * Delta * (1 + d_port).
Definition (two-hop evaluator). For u,v in V let
S(u,v) := H(u) cap H(v) if nonempty; else S(u,v) := F(u,v) with |F(u,v)| = O(1).
Eval(u,v) := min_{h in S(u,v)} [ d(u,h) + d(h,v) ].

T_hit (hitting and exactness).
(A5) Portal nonexpansion: portalizing any overlap crossing on a geodesic does not increase length.
(A6) For all u,v there exists a geodesic that contains some portal h with h in H(u) cap H(v).

T_CCR (CCR gluing and scheme invariance).
Let Tot := ⊕_{p,q} C^q(U; Omega^p), delta_tot := (-1)^p delta, D := d + delta_tot. There exists a degree +1 derivation R (Robin transfer) such that
(A7) (D+R)^2 = 0 in End(Tot).
(A8) [D+R, iota_X] = L_X and [D+R, L_X] = 0 for all vector fields X.
(A9) HPL smallness: gamma := ||h|| * ( ||alpha|| ||E|| + ||beta|| ||S|| ||L_X|| ) < 1 for the standard Cech–de Rham contraction (iota, pi, h) at a fixed Sobolev level.

T_loc+mu (local updates and incidence Moebius).
(A10) Each update changes weights only on a finite set Upt subset O; BDelta(o) bounds the contribution from o to any leg length variation.
(A11) The overlap poset P (nonempty intersections ordered by inclusion) has bounded height. For any leg x->h,
|Delta d(x,h)| <= sum_{o touching the leg} BDelta(o),
and the sum has O(1) terms (finite Moebius support).

T_prom (promotion and margin).
(A12) Prom(u,v) means a stored value D[u,v] and a witness h* in S(u,v).
(A13) Margin: m(u,v) := min_{h != h*} [d(u,h)+d(h,v)] - [d(u,h*)+d(h*,v)] >= eta(u,v) > 0.
	3.	O(1) evaluability

Theorem 3.1 (if and only if).
Under T_core ∪ T_kappa, Eval(u,v) is a bounded-size term (min over at most kappa candidates) and evaluable in O(1) word operations independent of |V|. Conversely, if no uniform bound on |S(u,v)| exists, there are families forcing superconstant work.
	4.	Exactness

Theorem 4.1 (if and only if).
Assume T_core ∪ T_kappa. For all u,v,
Eval(u,v) = d(u,v)  iff  T_hit holds.
Proof.
If hitting fails, every geodesic avoids H(u) cap H(v); by triangle inequality d(u,h)+d(h,v) > d(u,v) for all h. If hitting holds, pick h on a geodesic; equality holds at that h.
	5.	Scheme invariance and CCR

Definition 5.1 (online scheme).
The online scheme is the finite set of degree-0 operations: lookups d(x,h), additions, comparisons, and a bounded loop over S(u,v), plus emitting the witness and table ids.

Theorem 5.2 (if and only if).
Assume T_core. The online scheme is constant under cover refinement and degree-0 incidence gauge if and only if T_CCR holds.
Corollary 5.3 (HPL bound).
Under (A9), truncating the HPL series yields residual at most gamma^{m+1}/(1-gamma), independent of |V|.
	6.	Maintenance with Moebius locality

Definition 6.1 (leg aggregates for a maintained pair).
For Prom(u,v) with witness h*, set
U := sum_{o touching leg u->h*} BDelta(o)
V := sum_{o touching leg h*->v} BDelta(o)
For any competitor h != h*, set
C_h := sum_{o touching legs u->h and h->v} BDelta(o)

Theorem 6.2 (if and only if).
Under T_loc+mu, U, V, and each C_h are computable in O(1) per update event; conversely, if the overlap poset height is unbounded or updates touch unboundedly many overlaps, no uniform O(1) guard test exists.
	7.	Zero-time persistence

Definition 7.1 (O(0) at time t).
A maintained pair (u,v) is O(0) at time t if Query(u,v) returns D[u,v] by a dictionary read without recomputing legs on the query path.

Theorem 7.2 (if and only if margin guard).
Assume T_core ∪ T_kappa ∪ T_loc+mu ∪ T_prom. For Prom(u,v) with witness h*, the following are equivalent for every update event:
(i) D[u,v]=d(u,v) and Query(u,v) is O(0).
(ii) U + V < m(u,v)/2 and for all h != h*, C_h < m(u,v)/2.
Proof.
(ii => i) The stored best changes by at most U+V; each competitor improves by at most C_h; neither can cross half the margin. (i => ii) If either inequality fails, some update can flip the argmin or alter the value beyond the stored one.

Corollary 7.3 (decidable guard and audit object).
The guard is checkable in O(1). A constant-size audit object suffices:
(value, h*, ids_{u,h*}, ids_{h*,v}, m, U, V).
	8.	Closure properties

Theorem 8.1.
Assume T_core ∪ T_CCR.
(a) Permutation equivariance: Eval(pi u, pi v) = Eval(u,v).
(b) Phase invariance: changing lattice phases preserves Delta, kappa, and the evaluation term.
(c) Monoidal equivariance: scaling all weights by c>0 scales Eval by c and preserves minimizers.
(d) Refinement stability: bounded-overlap refinements do not alter the online scheme or the candidate bound.
	9.	Necessary conditions and failure modes

Theorem 9.1.
There exist graph families showing:
(a) If bounded overlap fails, sup_{u,v} |H(u) cap H(v)| grows with |V|, so no uniform O(1) bound exists.
(b) If hitting fails for some pairs, exactness fails by Theorem 4.1.
(c) If gamma >= 1, the HPL contraction need not converge; enforcing consistency may require online solves, breaking scheme constancy.
(d) If locality or bounded Moebius support fails, the guard cannot be checked in O(1).
	10.	Selection of maintained pairs (off-path)

Model.
Queries Q_t are i.i.d. from a distribution pi on V x V. Maintain counts C_t(u,v). Promote when C_t(u,v) >= tau_t with tau_t = o(t); demote by TTL with hysteresis. Assume guards are enforced at each update.

Theorem 10.1.
If pi(u,v) > 0 then with probability 1 there exists T such that Prom(u,v) for all t >= T. The expected query cost converges to p_hit * O(0) + (1 - p_hit) * O(1), where p_hit is the stationary mass of promoted pairs.
	11.	Minimal assumption profiles

G1 O(1) evaluability for all pairs: T_core ∪ T_kappa.
G2 Exactness for all pairs: T_core ∪ T_kappa ∪ T_hit.
G3 Scheme constancy: T_core ∪ T_CCR.
G4 O(1) maintenance and O(1) guard checks: T_loc+mu.
G5 Persistent O(0) for maintained pairs: T_prom plus the margin guard.

Notes.
The framework is metric-agnostic beyond nonnegativity; directed graphs use separate H_out and H_in. Negative edges are excluded; zero-weight strongly connected components should be contracted. All constants are independent of |V|; if empirical p99 of |H(u) cap H(v)| exceeds kappa or the MC residual is nonzero, the preconditions are violated and guarantees should be suspended until restored.
