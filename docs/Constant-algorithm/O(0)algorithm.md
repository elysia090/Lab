Title: O(0) Distance Queries as an Invariant Oracle: Necessary and Sufficient Conditions via Cech Nerves, Incidence Moebius Calculus, and CCR Gluing

Abstract.
We formalize when single-pair distance queries on finite nonnegative weighted graphs admit zero-time answers (O(0)) for a promoted subset of pairs, while all other queries remain constant-time (O(1)). The evaluator is a two-hop invariant oracle built over a bounded-overlap Cech nerve with finite portal sets and vertex hub sets. We axiomatize the cover, nerve, portals, hubs, Robin transfer, and the Cartan–Cech–Robin (CCR) structure. We prove the following equivalences. (1) O(1) evaluability for all pairs holds iff the candidate set size is uniformly bounded by a constant kappa derived from cover parameters. (2) Exactness of the oracle for all pairs holds iff every pair admits a portalized geodesic that hits a witness hub (Hit_2hop). (3) Constancy of the online template and its stability under cover refinement and degree-0 gauge holds iff the CCR Maurer–Cartan and Cartan identities are satisfied. (4) O(1) maintenance under local edge updates holds iff the overlap poset has bounded height, so that Moebius inversion on the incidence algebra has finite support per affected leg. (5) Persistent O(0) for a promoted pair holds iff a margin guard inequality is satisfied at every update. We also give decidable guard checks in O(1), audit objects of constant size, and probabilistic guarantees for heavy-hitter promotion by a Pólya urn.
	1.	Language, structures, and standing measurement

1.1 Language L.
Sorts: V (vertices), O (overlaps), S (scales), R (nonnegative reals).
Symbols:
B(s,c) subset V for cluster at scale s and center c.
Overlap o subset V with o = B(s,c) cap B(s,c’).
Adj_s(c,c’) for cluster-center adjacency at scale s.
P(o) subset Overlap(o) for finite portal set.
H: V -> Fin(V) for hub map.
d: V x V -> R for shortest-path distance with triangle inequality and d(x,x)=0.
Prom(u,v) for promotion predicate.
BDelta: O -> R for per-overlap nonnegative update bound.
Parameters: Delta, kappa, L, d_port, gamma in R_{\ge 0}.

1.2 Structures.
An L-structure interprets the above over a finite nonnegative weighted graph G=(V,E,w>=0) with a finite good cover U at scales S={0,…,L-1}. Distances live in the Lawvere quantale ([0,inf], +, 0, >=).
	2.	Axiom sets

T_core (bounded cover and nerve).
(A1) For all v and s, the number of clusters containing v is at most Delta.
(A2) For all s,c, the number of neighbors Adj_s(c,c’) is O(1).
(A3) For all overlaps o, the portal set P(o) has size at most d_port.

T_kappa (two-hop scaffold).
(A4) For all v, |H(v)| <= kappa := L * Delta * (1 + d_port).
Definition 2.1 (Oracle term). For u,v in V let
S(u,v) := H(u) cap H(v) if nonempty, else S(u,v) := F(u,v) where |F(u,v)| = O(1).
Oracle(u,v) := min_{h in S(u,v)} [ d(u,h) + d(h,v) ].

T_hit (hitting and exactness).
(A5) Portal nonexpansion: replacing a geodesic overlap crossing by a portal crossing does not increase length.
(A6) Hit_2hop: for all u,v there exists a geodesic containing a portal h with h in H(u) cap H(v).

T_CCR (CCR gluing and template invariance).
Let Tot := ⊕_{p,q} C^q(U; Omega^p), delta_tot := (-1)^p delta, D := d + delta_tot. There exists a degree +1 derivation R on Tot such that
(A7) Maurer–Cartan: (D+R)^2 = 0 in End(Tot).
(A8) Cartan: [D+R, iota_X] = L_X and [D+R, L_X] = 0 for all vector fields X.
(A9) HPL smallness: gamma := ||h|| * ( ||alpha|| ||E|| + ||beta|| ||S|| ||L_X|| ) < 1 for the standard Cech–de Rham contraction (iota, pi, h) at a fixed Sobolev level.

T_loc+mu (local updates and incidence Moebius).
(A10) Each update changes weights only in a finite set Upt subset O; BDelta(o) bounds the induced path-length contribution from overlap o.
(A11) The overlap poset P of nonempty intersections ordered by inclusion has bounded height. The incidence algebra I(P) with zeta z and Moebius mu satisfies that for any leg x->h,
|Delta d(x,h)| <= sum_{o touching the leg} BDelta(o),
and the sum has O(1) terms.

T_prom (promotion and margin).
(A12) Prom(u,v) indicates that D[u,v] and a witness h* in S(u,v) are stored.
(A13) Margin: m(u,v) := min_{h != h*} [d(u,h)+d(h,v)] - [d(u,h*)+d(h*,v)] >= eta(u,v) > 0.
	3.	O(1) evaluability

Theorem 3.1 (iff).
Under T_core ∪ T_kappa, Oracle(u,v) is definable by a bounded-size term and is evaluable in O(1) word operations independent of |V|; conversely, if no uniform bound on |S(u,v)| exists, there are graph families forcing superconstant work.
Proof.
S(u,v) has size at most kappa by (A4). Min over a constant set is computable by a constant-depth comparator tree with a fixed number of additions and comparisons. For necessity, take a sequence of instances with |H(u) cap H(v)| growing; any exact evaluator must inspect a growing number of candidates.
	4.	Exactness

Theorem 4.1 (Oracle exactness iff Hit_2hop).
Assume T_core ∪ T_kappa. Then for all u,v,
Oracle(u,v) = d(u,v) if and only if T_hit holds.
Proof.
If Hit_2hop fails, every geodesic from u to v avoids H(u) cap H(v); by triangle inequality d(u,h)+d(h,v) > d(u,v) for all h, so Oracle(u,v) > d(u,v). If Hit_2hop holds, pick h on a geodesic with h in H(u) cap H(v); then d(u,v)=d(u,h)+d(h,v) and Oracle(u,v) attains equality at h.
	5.	Template invariance and CCR

Definition 5.1 (template).
An online template is the finite set of degree-0 operations used by the evaluator: table lookups d(x,h), additions, comparisons, bounded loops over S(u,v), and audit output of the witness and table ids.

Theorem 5.2 (iff).
Assume T_core. The online template is constant across cover refinements and degree-0 incidence gauges if and only if T_CCR holds.
Proof.
If T_CCR holds, overlap consistency constraints are enforced algebraically by (A7) and commute with incidence actions by (A8), so refinement maps intertwine D and R and preserve degree-0 evaluation. Conversely, if (D+R)^2 != 0, there exists a degree-2 curvature K = (1/2)[D+R, D+R] that does not vanish on some q=0 generator after refinement; then enforcing consistency requires extra corrective operations, contradicting constancy of the template.

Corollary 5.3 (HPL bound).
Under (A9), any preprocessing that truncates the HPL series has residual bounded by gamma^{m+1}/(1-gamma), independent of |V|. Thus algebraic splice errors are uniformly controlled.
	6.	Maintenance with Moebius locality

Definition 6.1 (leg aggregates).
For a promoted pair (u,v) with witness h*, set
U := sum_{o touching the leg u->h*} BDelta(o),
V := sum_{o touching the leg h*->v} BDelta(o).
For any competitor h != h* set
C_h := sum_{o touching the legs u->h and h->v} BDelta(o).

Theorem 6.2 (iff).
Under T_loc+mu, for any update event, U, V, and C_h are computable in O(1) time and space; conversely, if the overlap poset height is unbounded or updates can touch unboundedly many overlaps, no uniform O(1) guard test exists.
Proof.
Bounded height implies that for each leg the set of contributing overlaps is O(1). Without bounded height or locality, a single update can change path lengths through arbitrarily long inclusion chains, forcing superconstant inspection.
	7.	O(0) persistence

Definition 7.1 (O(0) at time t).
A promoted pair (u,v) is O(0) at time t if Query(u,v) returns D[u,v] by a dictionary read without recomputing any leg values on the query path.

Theorem 7.2 (iff margin guard).
Assume T_core ∪ T_kappa ∪ T_loc+mu ∪ T_prom. For a promoted pair (u,v) with witness h*, the following are equivalent for every update event.
(i) Persistence: D[u,v]=d(u,v) and Query(u,v) is O(0).
(ii) Margin guard: U + V < m(u,v)/2 and for all h != h*, C_h < m(u,v)/2.
Proof.
(ii implies i) The stored best value changes by at most U+V; each competitor can improve by at most C_h. Both are strictly less than m/2, so the argmin h* and the value remain unchanged. (i implies ii) If either inequality fails, some update can flip the argmin or alter the value beyond the stored one, contradicting persistence.

Corollary 7.3 (decidable guard and audit object).
The guard is checkable in O(1). An audit object of constant size suffices for third-party verification:
(value, h*, ids_{u,h*}, ids_{h*,v}, m, U, V).
	8.	Closure properties

Theorem 8.1 (equivariance and stability).
Assume T_core ∪ T_CCR.
(a) Permutation equivariance: Oracle(pi u, pi v)=Oracle(u,v) for any permutation pi of V.
(b) Cover-phase invariance: changing lattice phases preserves Delta, kappa, and the evaluator term.
(c) Monoidal equivariance: scaling all weights by c>0 scales Oracle by c and preserves argmin sets.
(d) Refinement stability: cover refinements preserving bounded overlap do not alter the template or candidate bound.
Proof.
Degree-0 blocks commute with relabelings and phase changes; positive scaling is a strong monoidal endofunctor on ([0,inf],+,0,>=); CCR identities ensure refinement stability.
	9.	Impossibility and necessity

Theorem 9.1 (necessary conditions).
There exist graph families such that:
(a) If bounded overlap fails, sup_{u,v} |H(u) cap H(v)| grows with |V|, hence no uniform O(1) bound on evaluation exists.
(b) If Hit_2hop fails for some pairs, exactness fails by Theorem 4.1.
(c) If gamma >= 1, the HPL contraction does not converge uniformly; enforcing consistency may require online solves, breaking template constancy.
(d) If locality or bounded Moebius support fails, the guard cannot be checked in O(1).
	10.	Promotion guarantees (off-path, probabilistic)

Model.
Queries Q_t are i.i.d. from a distribution pi on V x V. Maintain counts C_t(u,v) and promote when C_t(u,v) >= tau_t with tau_t = o(t); demote by TTL with hysteresis. Assume maintenance enforces the guard at each update.

Theorem 10.1 (almost-sure promotion and asymptotic cost).
If pi(u,v) > 0 then with probability 1 there exists T such that Prom(u,v) holds for all t >= T. The expected query cost converges to p_hit * O(0) + (1 - p_hit) * O(1) where p_hit is the stationary mass of promoted pairs. For Zipf tails with exponent greater than 1, a constant-size dictionary attains p_hit close to 1.
Proof.
Strong law of large numbers gives almost-sure promotion when tau_t = o(t). The cost decomposition follows from the law of total expectation.
	11.	Minimal assumption profiles

G1 O(1) evaluability for all pairs: T_core ∪ T_kappa.
G2 Exactness for all pairs: T_core ∪ T_kappa ∪ T_hit (iff).
G3 Template invariance: T_core ∪ T_CCR (iff).
G4 O(1) maintenance and O(1) guard checks: T_loc+mu (iff).
G5 Persistent O(0) for promoted pairs: T_prom plus the margin guard (iff).
	12.	Notes on scope

The theory is metric-agnostic beyond nonnegativity and the standard quantale; directed graphs use separate H_out and H_in in the same formalism. Negative edges are excluded. Zero-weight strongly connected components should be contracted. All constants are independent of |V|; when empirical p99 of |H(u) cap H(v)| exceeds kappa or the MC residual is nonzero, the preconditions are violated and O(0)/O(1) guarantees must be suspended until repaired.

Acknowledgment of equivalence of terminology.
The invariant oracle described above is equivalent in meaning to O(0) algorithm and to computational wormhole in the sense of a measurement-stable, template-stable evaluator with the iff properties stated in Sections 4, 5, 6, and 7.
