Title
Mass Gap via Identity Factorization, Mosco Convergence, OS Reflection Positivity, and IR Cartan–Kahler Preservation
— Uniform Lower Bounds from DEC/Cech Gluing —

Abstract
On a complete Kahler manifold with shape-regular meshes, define on 0/1-cochains
L_a = [ Rgrad_a d_a S_a ; W_a^{1/2} B_{Sigma,a} ; sqrt(kappa) I ],   A_a = L_a^T L_a.
A discrete Green identity plus exact square completion at mirror/Robin boundaries gives the Master Identity
(u, A_a u) = || L_a u ||2^2.
With an a-independent discrete Poincare constant and uniform ellipticity we obtain
(u, A_a u) >= c0 ||u||{0,a}^2,   c0 = min{alpha, w_min, kappa} / (beta_up * C_PF^2 + 1) > 0.
Whitney–DEC recovery implies Mosco convergence E_a ->_M E and hence strong resolvent/semigroup convergence. Discrete OS reflection positivity holds by a Schur complement and is preserved in the limit. Therefore the limit operator A satisfies a mass gap lambda_min(A) >= c0. Interpreting A = L^T L as a bosonic square-sum, the gap is equivalent to exponential clustering; with SU(3) symmetry, only Cartan data on the Weyl quotient T/W persist in the IR (“IR Cartan–Kahler preservation”). We give an exact Robin coefficient, first-order cell-average error, stability under subspace (Urn) constraints, and model-independence via an arithmetic–geometric mean (AGM) sandwich.
	0.	Setting, notation, and analytic framework
H1 (Geometry) Complete Kahler (X, omega, J, g), bounded curvature and finitely many covariant derivatives, positive injectivity radius.

H2 (Meshes) Shape-regular triangulations {T_a}_{a -> 0+} with cell diameter delta_a -> 0.

H3 (Discrete spaces) V_a = R^{n0(a)} for 0-cochains, E_a = R^{n1(a)} for 1-cochains. d_a : V_a -> E_a is the signed incidence.

H4 (Hodge and weights) Diagonal SPD matrices *{0,a}, *{1,a}, W_a such that
alpha I <= *_{k,a} <= beta_up I,   alpha I <= W_a <= beta_up I,
with alpha, beta_up independent of a. Let w_min := min diag(W_a).

H5 (Whitening and mass) S_a := (_{0,a})^{-1/2},  Rgrad_a := (_{1,a})^{1/2},  kappa > 0.

H6 (Boundary/reflection) Measure-preserving isometry theta with fixed set Sigma; meshes compatible with theta. Boundary condition is mirror–Dirichlet or Robin with coefficient prescribed in Sec. 12.

H7 (Whitney–DEC commuting pair) Let Samp_a : H^1(X) -> V_a (sampling) and WhtRec_a : V_a -> H^1(X) (Whitney reconstruction) satisfy
|| WhtRec_a Samp_a v − v ||{L^2} <= C delta_a ||v||{H^1},   || d (WhtRec_a) − (WhtRec_a) d_a || <= C delta_a,
with constants depending only on shape-regularity.

H8 (Cech/partition of unity) A finite cover {U_i} with a partition {phi_i} enabling local coefficient normalization and stable DEC gluing.

Inner products and norms
(u,v){0,a} := u^T (*{0,a}) v,   (eta,zeta){1,a} := eta^T (*{1,a}) zeta,
||u||{0,a}^2 := (u,u){0,a},     ||eta||{1,a}^2 := (eta,eta){1,a}.
Let C_PF be an a-independent discrete Poincare constant (mean-zero or one pinned vertex). Define
Cbar := beta_up * C_PF^2 + 1,   c0 := min{alpha, w_min, kappa} / Cbar > 0.
	1.	Operator and boundary normalization
Definition 1.1 (Operator)
L_a := [ Rgrad_a d_a S_a ; W_a^{1/2} B_{Sigma,a} ; sqrt(kappa) I ],   A_a := L_a^T L_a.
Discrete Green identity
(u, (_{0,a})^{-1} d_a^T ({1,a}) d_a u){0,a} = || (*_{1,a})^{1/2} d_a u ||_2^2 − F_boundary(u).
Mirror boundary gives F_boundary = 0. For Robin, Sec. 12 prescribes beta_a(nu) on each boundary normal edge nu so that completing the square cancels F_boundary exactly.
	2.	Master identity and a-independent coercivity
Proposition 2.1 (Master identity)  For all u in V_a,
(u, A_a u) = || L_a u ||_2^2,  hence A_a is SPD.

Lemma 2.2 (Discrete Poincare, a-independent)  With mean-zero or one pinned vertex,
||u||{0,a} <= C_PF || d_a u ||{1,a}.

Theorem 2.3 (Coercive lower bound)
(u, A_a u) >= c0 ||u||_{0,a}^2.
Proof sketch. From Prop. 2.1,
(u, A_a u) = || Rgrad_a d_a S_a u ||2^2 + || W_a^{1/2} B{Sigma,a} u ||2^2 + kappa ||u||2^2.
By H4–H5,
|| Rgrad_a d_a S_a u ||2^2 >= alpha * beta_up^{-1} || d_a u ||{1,a}^2,
|| W_a^{1/2} B{Sigma,a} u ||2^2 >= w_min || B{Sigma,a} u ||2^2,
kappa ||u||2^2 >= kappa * beta_up^{-1} ||u||{0,a}^2.
Apply Lemma 2.2 to trade ||u||{0,a} for C_PF ||d_a u||{1,a} where needed; collect terms and absorb beta_up to obtain c0. QED.
	3.	Approximation and Cech/PoU gluing
Lemma 3.1 (First-order cell average) If a coefficient field A(x) is Lipschitz with constant L, then ||A − A_tilde,a||_inf <= L delta_a.

Lemma 3.2 (Duhamel) If A,B >= c0 I, then for t >= 0,
|| e^{−tA} − e^{−tB} || <= t e^{−t c0} || A − B ||.

Theorem 3.3 (Semigroup first-order agreement)
|| e^{−t A_a} − e^{−t A_tilde,a} || <= t e^{−t c0} C L delta_a.

Proposition 3.4 (Global form via PoU)
Define E_a[u] := sum_i ( phi_i u, A_{U_i,a} phi_i u ). Overlap errors are o(1) by H8 and Lemma 3.1.
	4.	Mosco convergence and strong limits
Lemma 4.1 (Recovery sequence) From H7,
|| WhtRec_a Samp_a v − v ||_{L^2} -> 0,   E_a[Samp_a v] -> E[v].

Lemma 4.2 (Liminf) If u_a weakly converges to u in L^2, then liminf E_a[u_a] >= E[u].

Theorem 4.3 (Mosco)  E_a ->_M E.

Theorem 4.4 (Strong resolvent/semigroup)
(A_a + lambda I)^{-1} -> (A + lambda I)^{-1} strongly for lambda > 0, and e^{−t A_a} -> e^{−t A} strongly for t >= 0.
	5.	OS reflection positivity and preservation
Let theta be the reflection, Pi_+ the projection to the positive side. Block-decompose under reflection:
A_a = [ X  Y ; Y^T  Z ].
Lemma 5.1 (Discrete OS)
( f, A_a^{-1} Theta_a f ) = || Pi_+ L_a^{-1} (f + Theta_a f) ||2^2 >= 0  for f supported on the positive side.
Sketch. Sec. 12 ensures Z >= 0. The Schur complement
X − Y Z^{dagger} Y^T = (Pi+ L_a)^T (Pi_+ L_a) >= 0
yields the claim.

Theorem 5.2 (Limit OS)
( f, A^{-1} Theta f ) = lim_a ( f, A_a^{-1} Theta_a f ) >= 0,
by Theorem 4.4 and lower semicontinuity of closed forms. Markov/symmetry properties propagate by contractivity.
	6.	Mass gap in the limit
Proposition 6.1  lambda_min(A_a) >= c0 by Theorem 2.3.
Theorem 6.2  lambda_min(A) >= limsup_a lambda_min(A_a) >= c0.
Corollary 6.3 (Gap)  Delta := inf( spectrum(A) \ {0} ) = lambda_min(A) >= c0.
	7.	Subspace (Urn) constraints
Let C_a be a constraint and P_a the orthogonal projector onto M_a := ker C_a.
Lemma 7.1  E_{a,M}[v] := E_a[P_a v] is closed.
Proposition 7.2  lambda_min(A_{a,M}) >= lambda_min(A_a).
Theorem 7.3 (Mosco on subspaces) If P_a -> P strongly, then E_{a,M} ->_M E_M. OS positivity and the gap persist.
	8.	Square-sum (bosonic) structure and exponential clustering
Definition 8.1  Cov(O_x,O_y) := <O_x O_y> − <O_x><O_y> for gauge-invariant observables.
Theorem 8.2 (Gap => exponential clustering)
If lambda_min(A) >= c0 > 0, then |Cov(O_x,O_y)| <= C exp(−c0 * dist(x,y)).
Sketch. Spectral theorem for A = L^T L and Laplace representation of two-point functions imply ||e^{−tA}|| = exp(−t lambda_min) <= exp(−t c0). Transfer to spatial decay by finite-propagation/elliptic locality of gauge-invariant observables.

Theorem 8.3 (Clustering => gap)
Uniform exponential clustering with rate c0 implies ||e^{−tA}|| = exp(−t lambda_min) and hence lambda_min >= c0.

Corollary 8.4 (Equivalence)  Mass gap c0 > 0 <=> exponential clustering rate c0.
	9.	SU(3) and IR Cartan–Kahler preservation
Let su(3) = h (Cartan) direct-sum (sum over roots alpha of g_alpha). Let P_H, P_alpha be the projections acting on gauge-invariant observables.
Theorem 9.1 (IR Cartan–Kahler preservation)
If c0 > 0, then for r -> infinity:
|| P_alpha Cov(O; r) || <= C exp(−c0 r) for all roots alpha, and  lim_{r->infty} P_H Cov(O; r) exists.
Thus in the IR only Cartan data on T/W persist. No claim of spontaneous symmetry breaking.

Threshold 9.2  As c0 -> 0+, root-channel suppression vanishes; Cartan projection purity fails. The threshold is c0.
	10.	Linearized gravity
For linearized gravity the gauge-invariant quadratic form has the type A_grav = L_grav^T L_grav. Hence “gap = clustering rate = square-sum lower bound” transfers to gauge-invariant observables.
	11.	Lattice diagnostics (minimal verification)

	1.	Cartan current two-point Gamma_H(r): effective mass m_eff(r) := log( Gamma_H(r) / Gamma_H(r+1) ) -> c0.
	2.	Root channels: Gamma_alpha(r) <= C exp(−c0 r) with the same rate.
	3.	Weyl-invariant reconstruction is unique on T/W.
	4.	Mesh stability: extracted c0 stable as a -> 0 by Secs. 3–4.
	5.	Boundary invariance: mirror/Robin equivalence by Secs. 1 and 12.
	6.	Practical fit: plateau finder on m_eff(r) with monotonicity veto, then constrained multi-channel exponential fit sharing a common rate c0; report c0 with jackknife over PoU blocks.

	12.	Robin coefficient (exact square completion)
For each boundary face with outward normal edge nu = (x -> x_nu) and physical length h_nu, set
beta_a(nu) := (({1,a}){nu,nu}) / ( ({0,a}){x,x} * h_nu )  > 0.
Then the boundary flux term cancels exactly upon completing the square, i.e., the contribution of || W_a^{1/2} B_{Sigma,a} u ||_2^2 equals the Robin penalty and F_boundary(u) = 0. By shape-regularity and H4, beta_a(nu) admits uniform positive upper/lower bounds; hence c0 is invariant under choosing mirror vs Robin.
	13.	Model-independence (AGM sandwich)
Let A,B be SPD with m I <= A,B <= M I, where m,M depend only on shape-regularity and H4. For the Kubo–Ando geometric mean A#B,
m I <= A#B <= (A+B)/2 <= (M/m) B,
and symmetrically with A,B swapped. Consequently, differing Hodge scalings, Robin normalizations, or averaging schemes change constants only by gamma := M/m; c0 and Mosco limits are invariant up to gamma.

Appendix A (Whitney–DEC commuting construction)
Choose Samp_a as vertex averaging and WhtRec_a as first-order Whitney interpolation on cells. Shape-regularity yields
|| WhtRec_a Samp_a v − v ||{L^2} <= C delta_a ||v||{H^1},   || d WhtRec_a − WhtRec_a d_a || <= C delta_a.
Together with the continuous Poincare inequality, this gives an a-independent discrete C_PF.

Appendix B (Schur complement for the Robin block)
Under reflection,
A_a = [ X  Y ; Y^T  Z ] with Z >= 0 by Sec. 12.
Then
X − Y Z^{dagger} Y^T = (Pi_+ L_a)^T (Pi_+ L_a) >= 0,
so for any f supported on the positive side,
( f, A_a^{-1} Theta_a f ) = || Pi_+ L_a^{-1} (f + Theta_a f) ||_2^2 >= 0.

Appendix C (Heat-kernel trace pinching)
For a finite-dimensional subspace P_a V_a,
exp(−tau lambda_min(A_a)) <= Tr( exp(−tau A_a) |_{P_a V_a} ) <= N(a) exp(−tau lambda_min(A_a)).
Hence for tau >= (log N(a))/epsilon,
| −(1/tau) log Tr − lambda_min(A_a) | <= epsilon,
giving a stable numerical lower bound on the gap.

Constant-dependence summary
All final bounds depend only on (alpha, beta_up, w_min, kappa, C_PF) and shape-regularity parameters; PoU overlap and DEC commuting contribute o(1) in a.

Minimal “hackable” numerical recipe (deterministic)
	1.	Enforce one-pin or mean-zero so that ||u||{0,a} <= C_PF ||d_a u||{1,a}; estimate C_PF on 3 nested meshes and check stability.
	2.	Compute beta_a(nu) by Sec. 12; verify machine-precision cancellation of the boundary residual in the discrete Green identity on planar patches.
	3.	Estimate c0 by two routes and cross-check:
(i) Joint exponential fit with a common rate across all root-channel correlators Gamma_alpha(r).
(ii) Heat-kernel pinching of Tr(exp(−tau A_a)|_{P_a V_a}) for moderate tau; increase tau until the bound stabilizes.
	4.	Report (c0_hat, gamma_hat) where gamma_hat := M/m from diagonal ranges of *_{k,a} and W_a; this makes “model-independence up to gamma” explicit.

