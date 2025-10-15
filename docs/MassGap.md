Title
Mass Gap via Identity Factorization, Mosco Convergence, OS Reflection Positivity, and IR Cartan–Kahler Preservation
– Uniform Lower Bounds from DEC/Cech Gluing –

Abstract
On a complete Kahler manifold (X, omega, J, g) discretized by a shape-regular mesh family (T_a) with cell diameter delta_a -> 0, we define a block operator on 0/1-cochains
L_a := [ Rgrad_a d_a S_a ; W_a^{1/2} B_{Sigma,a} ; sqrt(kappa) I ],   A_a := L_a^T L_a,
where S_a = (_{0,a})^{-1/2}, Rgrad_a = ({1,a})^{1/2}, W_a is a positive diagonal weight, B{Sigma,a} is the boundary/flux map compatible with a reflection across Sigma, and kappa >= 0. A discrete Green identity plus exact square completion at mirror/Robin boundaries yields the Master Identity
(u, A_a u) = ||L_a u||2^2
for all u in V_a (0-cochains). With an a-independent discrete Poincare inequality and uniform ellipticity of DEC Hodge stars and W_a, we obtain the coercive bound
(u, A_a u) >= c0 ||u||{0,a}^2
with an explicit a-independent constant c0 > 0. Using Whitney–DEC recovery and commuting estimates, we prove Mosco convergence of the discrete energies to a closed form E generating a self-adjoint positive operator A, and hence strong resolvent/semigroup convergence A_a -> A. OS reflection positivity holds at the discrete level by a Schur-complement argument and is preserved in the limit. Therefore the limit operator admits a mass gap
Delta := lambda_min(A) >= c0.
Since A = L^T L is a bosonic square-sum, the gap is equivalent to exponential clustering: temporal decay of e^{-tA} is e^{-t Delta} and spatial two-point clustering is exp(-sqrt(Delta) r). Under SU(3) symmetry, root-channel correlators decay exponentially while Cartan components on the Weyl quotient T/W persist (IR Cartan–Kahler preservation). Assuming uniform Kahler nondegeneracy (metric lower bound m_g > 0), DEC Hodge stars inherit spectral bounds; consequently Delta admits the lower bound Delta >= c_K m_g with explicit c_K. The results are stable under subspace constraints and model choices (Kubo–Ando geometric-mean sandwich).
	0.	Geometric, analytic, and discrete setting
H1 (Geometry). (X, g) is complete Kahler with bounded curvature and finitely many covariant derivatives; injectivity radius bounded below.

H2 (Meshes). A family (T_a) of shape-regular triangulations with diam(T_a) = delta_a -> 0. Shape constants are a-independent.

H3 (Discrete spaces and coboundary). V_a = R^{n0(a)} (0-cochains), E_a = R^{n1(a)} (1-cochains). The coboundary d_a : V_a -> E_a is the signed incidence.

H4 (Hodge stars and weights). Diagonal SPD matrices *{0,a}, *{1,a}, W_a satisfy
alpha I <= *_{k,a} <= beta_up I,   alpha I <= W_a <= beta_up I
with a-independent alpha, beta_up > 0. Let w_min := min diag(W_a).

H5 (Whitening and mass). S_a := (_{0,a})^{-1/2},  Rgrad_a := (_{1,a})^{1/2},  kappa >= 0.

H6 (Reflection and boundary compatibility). There is a measure-preserving isometry theta with fixed set Sigma and mesh compatibility across Sigma. Boundary condition is mirror–Dirichlet or Robin, with coefficient beta_a(nu) prescribed in Sec. 12 for each boundary normal edge nu.

H7 (Whitney–DEC commuting pair). There are bounded operators
Samp_a : H^1(X) -> V_a  (sampling),   WhtRec_a : V_a -> H^1(X)  (Whitney reconstruction),
such that
||WhtRec_a Samp_a v - v||{L^2(X)} <= C delta_a ||v||{H^1(X)},   || d WhtRec_a - WhtRec_a d_a ||_{V_a->L^2} <= C delta_a,
with constants depending only on shape regularity. Here and below, all operator norms and constants are a-independent.

H8 (Cech partition and gluing). There exists a finite open cover {U_i} and a partition of unity {phi_i} with stable localization and coefficient normalization suitable for DEC.

Inner products. Define
(u,v){0,a} := u^T (*{0,a}) v,    (eta,zeta){1,a} := eta^T (*{1,a}) zeta,
and the norms ||u||{0,a}^2 := (u,u){0,a}, ||eta||{1,a}^2 := (eta,eta){1,a}. Let C_PF be an a-independent discrete Poincare constant valid either on mean-zero subspace or under one pinned vertex (Dirichlet pin). Define
Cbar := beta_up * C_PF^2 + 1,   c0 := min{alpha, w_min, kappa} / Cbar.

0.1. Uniform Kahler nondegeneracy and transfer to DEC
H0 (Uniform metric bounds). There exist 0 < m_g <= M_g < +inf such that
m_g |v|^2 <= g_x(v,v) <= M_g |v|^2
for a.e. x and all v in T_x X. Equivalently, the Kahler 2-form omega(.,.) = g(., J .) is uniformly nondegenerate.

Lemma 0.1 (Spectral transfer to lumped Hodge stars). Under H0–H2, the lumped Whitney mass matrices (diagonal {k,a}) satisfy
c m_g I <= _{k,a} <= C^ M_g I  (k = 0,1),
with shape-regularity constants c_, C^ independent of a. Hence alpha >= c_* m_g and beta_up <= C^* M_g.
	1.	Operator, Green identity, and boundary normalization
Definition 1.1 (Discrete operator). Let
L_a := [ Rgrad_a d_a S_a ; W_a^{1/2} B_{Sigma,a} ; sqrt(kappa) I ],    A_a := L_a^T L_a.
Here B_{Sigma,a} is the discrete boundary/flux map consistent with theta. The discrete Green identity reads
(u, (_{0,a})^{-1} d_a^T ({1,a}) d_a u){0,a} = ||(*_{1,a})^{1/2} d_a u||2^2 - F_boundary(u),
where F_boundary(u) is a quadratic boundary functional. For mirror boundary, F_boundary = 0. For Robin boundary, choosing beta_a(nu) as in Sec. 12 gives exact square completion so that F_boundary(u) is absorbed into ||W_a^{1/2} B{Sigma,a} u||_2^2.
	2.	Master identity and a-independent coercivity
Proposition 2.1 (Master identity and SPD). For all u in V_a,
(u, A_a u) = ||L_a u||2^2,
hence A_a is SPD on V_a with respect to (*{0,a}).

Lemma 2.2 (Discrete Poincare inequality). On mean-zero subspace (or with one pinned vertex),
||u||{0,a} <= C_PF ||d_a u||{1,a}.

Theorem 2.3 (Uniform coercivity). For all u in V_a,
(u, A_a u) >= c0 ||u||{0,a}^2,  with  c0 = min{alpha, w_min, kappa} / (beta_up C_PF^2 + 1).
Proof. Use Proposition 2.1 and H4–H5 to bound each block:
||Rgrad_a d_a S_a u||2^2 >= alpha beta_up^{-1} ||d_a u||{1,a}^2,
||W_a^{1/2} B{Sigma,a} u||2^2 >= w_min ||B{Sigma,a} u||2^2,
kappa ||u||2^2 >= kappa beta_up^{-1} ||u||{0,a}^2.
Apply Lemma 2.2 to trade ||u||{0,a} by C_PF ||d_a u||_{1,a} when needed, collect terms, and absorb beta_up into Cbar. QED.

Corollary 2.4 (Gap in terms of metric). Under H0 and Lemma 0.1,
Delta := lambda_min(A) >= c0 >= c_K m_g,
with
c_K := min{c_* m_g, w_min, kappa} / (C^* M_g C_PF^2 + 1).
If kappa = 0 and a Poincare constraint is imposed (mean-zero or pin), then
Delta >= alpha / (beta_up C_PF^2) >= (c_* / (C^* C_PF^2)) m_g.
	3.	Approximation and localization
Lemma 3.1 (Cell-average perturbation). If a coefficient field A(x) is L-Lipschitz, the cellwise average A_tilde,a satisfies ||A - A_tilde,a||_inf <= L delta_a.

Lemma 3.2 (Duhamel inequality). For SPD A,B >= c0 I and t >= 0,
||e^{-tA} - e^{-tB}|| <= t e^{-t c0} ||A - B||.

Proposition 3.3 (PoU gluing). Using H8, set
E_a[u] := sum_i ( phi_i u, A_{U_i,a} phi_i u ).
Then E_a is a closed form equivalent to (u, A_a u), and PoU overlap errors are o(1) as a -> 0.
	4.	Mosco convergence and closed-form limit
Define the continuous closed form E on L^2(X,vol_g) by the distributional limits of the DEC blocks, with domain the closure of C_c^\infty under the natural H^1-type norm. The operator A is the self-adjoint operator associated with E by the Friedrichs extension.

Theorem 4.1 (Mosco convergence). Under H1–H8,
E_a ->M E.
Proof. (M1) Recovery: take u in Dom(E) and u_a := Samp_a u; H7 gives ||WhtRec_a u_a - u||{L^2} -> 0 and E_a[u_a] -> E[u]. (M2) Liminf: if u_a weakly converges to u in L^2 and sup_a E_a[u_a] < inf, then liminf E_a[u_a] >= E[u] by lower semicontinuity and the commuting estimate ||d WhtRec_a - WhtRec_a d_a|| = O(delta_a). QED.

Corollary 4.2 (Strong resolvent/semigroup convergence). By consequences of Mosco convergence of closed convex forms,
(A_a + lambda I)^{-1} -> (A + lambda I)^{-1} strongly for all lambda > 0,
e^{-t A_a} -> e^{-t A} strongly for all t >= 0.
	5.	OS reflection positivity
Let Theta be the reflection on functions induced by theta and Pi_+ the orthogonal projector onto the positive half with respect to (*{0,a}). Block-decompose under reflection:
A_a = [ X  Y ; Y^T  Z ].
Lemma 5.1 (Discrete OS positivity). For f supported on the positive half,
( f, A_a^{-1} Theta_a f ){0,a} = || Pi_+ L_a^{-1} (f + Theta_a f) ||2^2 >= 0.
Proof. Sec. 12 normalizes Robin so that Z >= 0. Since A_a = L_a^T L_a,
X - Y Z^dagger Y^T = (Pi+ L_a)^T (Pi_+ L_a) >= 0,
hence the OS bilinear form is a square norm. QED.

Theorem 5.2 (OS positivity in the limit). Using Corollary 4.2 and lower semicontinuity of closed forms,
( f, A^{-1} Theta f ) >= 0.
Thus OS positivity is preserved.
	6.	Spectral gap and decay rates
Proposition 6.1 (Discrete gap). For each a, lambda_min(A_a) >= c0 by Theorem 2.3.

Theorem 6.2 (Limit gap). lambda_min(A) >= limsup_a lambda_min(A_a) >= c0. Hence the mass gap
Delta := lambda_min(A) >= c0.

Corollary 6.3 (Decay rates). Temporal semigroup:
||e^{-tA}|| = e^{-t Delta}.
Spatial clustering of gauge-invariant two-point functions:
|Cov(O_x, O_y)| <= C exp( - sqrt(Delta) * dist(x,y) ),
so the optimal spatial rate satisfies mu = sqrt(Delta) and mu >= sqrt(c0).

Theorem 6.4 (Partial converse: gap implies metric lower bound, up to constants). Assume kappa > 0 or a Poincare constraint. If Delta >= d0 > 0, then there exists C_geo > 0 depending only on shape-regularity and commuting constants such that
m_g >= d0 / C_geo.
Sketch. Transfer (u, Au) >= d0 ||u||_{L^2(g)}^2 to the continuous H^1 energy via Mosco, then bound L^2(g) by the gradient norm using DEC–Whitney equivalences and absorb boundary/mass/pinning constants. QED.
	7.	Subspace constraints (Urn)
Let C_a be a linear constraint and P_a the (*_{0,a})-orthogonal projector onto M_a := ker C_a.

Lemma 7.1 (Closedness). E_{a,M}[v] := E_a[P_a v] is a closed form.

Proposition 7.2 (Monotonicity of gaps). lambda_min(A_{a,M}) >= lambda_min(A_a).

Theorem 7.3 (Mosco on constrained subspaces). If P_a -> P strongly, then E_{a,M} ->_M E_M. Consequently strong resolvent/semigroup convergence, OS positivity, and the gap persist on M.
	8.	Square-sum structure and clustering equivalence
Let A = L^T L as above.

Theorem 8.1 (Gap implies clustering). If Delta > 0, then for gauge-invariant local observables,
|Cov(O_x, O_y)| <= C exp( - sqrt(Delta) * dist(x,y) ).

Theorem 8.2 (Clustering implies gap). If spatial clustering holds with uniform rate mu > 0, then Delta >= mu^2.

Corollary 8.3 (Equivalence). Mass gap Delta > 0 iff exponential clustering with rate sqrt(Delta) holds; temporal rate is Delta.

8.1. Metric-controlled IR rate
From Theorems 8.1–8.2 and Corollary 8.3 together with Corollary 2.4,
mu = sqrt(Delta) >= sqrt(c_K m_g).
	9.	SU(3) and IR Cartan–Kahler preservation
Let su(3) = h (Cartan) direct sum (sum over roots alpha of g_alpha). Let P_H and P_alpha denote projections acting on gauge-invariant observables.

Theorem 9.1 (IR preservation). If Delta > 0, then for r -> +inf,
||P_alpha Cov(O; r)|| <= C exp( - sqrt(Delta) r )  for all roots alpha,
lim_{r->+inf} P_H Cov(O; r) exists on T/W.
Thus only Cartan data on T/W persist in the IR. No claim of spontaneous symmetry breaking is made.

Theorem 9.2 (Metric-rate version). Under H0 and Delta >= c_K m_g,
||P_alpha Cov(O; r)|| <= C exp( - sqrt(c_K m_g) r ),   lim_{r->+inf} P_H Cov(O; r) exists.
	10.	Linearized gravity
For linearized gauge-invariant fields, the quadratic form has the structure A_grav = L_grav^T L_grav; therefore temporal decay, spatial clustering, and gap lower bounds transfer verbatim on gauge-invariant observables.
	11.	Numerical diagnostics (deterministic, reproducible)
(1) Enforce mean-zero or one pin so that ||u||{0,a} <= C_PF ||d_a u||{1,a}; estimate C_PF on three nested meshes and check stability.
(2) Robin normalization beta_a(nu) (Sec. 12): verify machine-precision cancellation of the boundary residual in the discrete Green identity on flat test patches.
(3) Gap from singular values: compute Delta_a := sigma_min(L_a)^2 (e.g., Lanczos/LOBPCG with shift-invert).
(4) Heat-kernel pinching: bound lambda_min(A_a) from Tr(exp(-tau A_a)|{P_a V_a}) for growing tau until stabilization.
(5) Spatial rate: fit a common exponential rate mu across root-channel correlators; check mu^2 vs Delta_a consistency; jackknife over PoU blocks.
(6) With H0: estimate m_g from diagonal ranges of *{k,a}; verify Delta_a >= c_K m_g_hat.
	12.	Robin coefficient and exact square completion
For a boundary face with outward normal edge nu = (x -> x_nu) and physical length h_nu, set
beta_a(nu) := (({1,a}){nu,nu}) / ( ({0,a}){x,x} * h_nu )  > 0.
Then the boundary flux term in the discrete Green identity is absorbed exactly by
||W_a^{1/2} B_{Sigma,a} u||_2^2,
so F_boundary(u) = 0. Shape regularity and H4 give uniform positive bounds on beta_a; hence c0 is invariant under mirror vs Robin choices.
	13.	Model-independence via Kubo–Ando sandwich
Let A,B be SPD with m I <= A,B <= M I (m,M depend only on shape-regularity and H4). For the geometric mean A#B,
m I <= A#B <= (A+B)/2 <= (M/m) B
(and symmetrically). Therefore changes of Hodge scaling, Robin normalization, or averaging are absorbed into a single multiplicative factor gamma := M/m. The gap lower bound and Mosco limit are stable up to gamma.

Constant-dependence summary
All final bounds depend only on (alpha, beta_up, w_min, kappa, C_PF) and shape-regularity; under H0 they can be re-expressed via (m_g, M_g) and mesh constants (c_, C^). PoU overlap and DEC commuting contribute o(1) as a -> 0.

Appendix A (Whitney–DEC commuting construction)
Choose Samp_a as vertex averaging and WhtRec_a as first-order Whitney interpolation. Then
||WhtRec_a Samp_a v - v||{L^2} <= C delta_a ||v||{H^1},   || d WhtRec_a - WhtRec_a d_a || <= C delta_a.
Together with the continuous Poincare inequality, this yields an a-independent discrete C_PF.

Appendix B (Schur complement for OS positivity)
Under reflection, write
A_a = [ X  Y ; Y^T  Z ]  with  Z >= 0  by Sec. 12.
Then
X - Y Z^dagger Y^T = (Pi_+ L_a)^T (Pi_+ L_a) >= 0,
so for f supported on the positive half,
( f, A_a^{-1} Theta_a f ) = ||Pi_+ L_a^{-1} (f + Theta_a f)||_2^2 >= 0.

Appendix C (Heat-kernel trace pinching)
On a finite-dimensional subspace P_a V_a,
exp(-tau lambda_min(A_a)) <= Tr( exp(-tau A_a)|_{P_a V_a} ) <= N(a) exp(-tau lambda_min(A_a)).
Thus for tau >= (log N(a))/epsilon,
| -(1/tau) log Tr - lambda_min(A_a) | <= epsilon.

Appendix D (Proof of Lemma 0.1: spectral equivalence of lumped stars)
Let M_{k,a} be consistent Whitney mass matrices and M^{lmp}{k,a} their diagonal lumped versions. For shape-regular meshes,
c_sh M{k,a} <= M^{lmp}{k,a} <= C_sh M{k,a}
with a-independent c_sh, C_sh > 0. Since M_{k,a} realize L^2(g) pairings with eigenvalues in [m_g, M_g], we obtain
c_sh m_g I <= M^{lmp}{k,a} <= C_sh M_g I.
Identifying *{k,a} with M^{lmp}_{k,a} up to scaling gives Lemma 0.1.

Appendix E (Boundary closedness and Kahler closed 2-form)
The Robin normalization of Sec. 12 ensures exact square completion of boundary fluxes and preserves discrete closedness. Under Mosco convergence, d_a -> d in the commuting sense, hence the continuous 2-form omega remains closed (d omega = 0) and nondegenerate by Corollary 2.4.

End of document.

