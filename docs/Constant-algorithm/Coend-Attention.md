Title: CCR-Coend-Attention — Profunctor Attention via Coends Integrated with Cartan–Cech–Robin (CCR) for O(1) Streaming, Safe-Z/Auto-k, and Certified Bounds

Abstract.
We give a review-ready, fully ASCII, CCR-integrated formalization of coend-based attention over a commutative semiring (or a quantale). Scores are computed as the profunctor composition Q:X^op x K->S and K^dagger:K^op x Y->S via a coend S(x,y)=∫^k Q(x,k) ⊗ K^dagger(k,y). Normalization is performed without materializing the score matrix, using a two-pass streaming log-sum-exp that retains only (m,Z) per query. Sublinear selection (candidates, upper bounds, Auto-k) is backed by closed-form L1 error bounds. Updates are O(1) by a seqlock read schema under Release/Acquire. We embed all pieces in the CCR framework on a finite good cover: the operators become degree-0 Tot blocks, gluing is enforced by a Robin transfer R_{alpha,beta}, Cartan identities are preserved, and forward/reverse/mixed differentials close inside a constant-size operator template. Complexity, stability, and approximation guarantees hold after CCR gluing.

Keywords: semiring attention; profunctor; coend; streaming log-sum-exp; Safe-Z; Auto-k; CCR; Robin transfer; HPL; BCH; O(1).
	1.	Setting and objects
1.1 Semiring and values. Let S=(S, ⊕, ⊗, 0, 1) be a commutative semiring; if needed a quantale. Let (Val, ⊞, 0_Val) be a commutative monoid with an S-action ⊙: S x Val -> Val that distributes over ⊞.

1.2 Profunctors and coend. Let Q:X^op x K->S and K^dagger:K^op x Y->S. Define the kernel
S(x,y) = ∫^{k in K} Q(x,k) ⊗ K^dagger(k,y).
For finite discrete K this is a finite sum:
S(x,y) = ⊕_{k in K} Q(x,k) ⊗ K^dagger(k,y).

1.3 Normalization and output. Choose a map phi:S->R_{>=0} consistent with S (e.g., softmax: phi=exp, tropical min-plus: phi=Id). For a fixed y, define
Z(y) := sum_{x in X} phi(S(x,y)),
alpha(x|y) := phi(S(x,y)) / Z(y),
out(y) := ⊞_{x in X} [ alpha(x|y) ⊙ V(x) ].
	2.	Streaming normalization without score storage
Two-pass log-sum-exp for each y:
First pass (scan x in X):
keep m := running max of s_x, Z := normalized partition;
when a new s > m: Z <- (isfinite(m)? Z*exp(m-s)+1 : 1), m <- s;
else Z <- Z + exp(s - m).
Second pass:
alpha_x = exp(s_x - m) / Z; accumulate out.
The state (m,Z) is O(1); exponent arguments are <= 0, avoiding over/underflow.
	3.	Sublinear selection with error guarantees
Let scores s_(1) >= s_(2) >= …; set m = s_(1), a_i := exp(s_(i) - m). Head mass H_k := sum_{i<=k} a_i; tail mass T_k := sum_{i>k} a_i. Full distribution p_i := a_i/(H_k+T_k). Truncation q keeps top-k and renormalizes on H_k.
Theorem 3.1 (L1 bound for top-k).
||p - q||_1 = 2T_k/(H_k+T_k) = 2R/(1+R),  R := T_k/H_k.
Corollary 3.2 (Auto-k).
||p - q||_1 <= eps  iff  R <= eps/(2-eps).
One pass over sorted candidates suffices to find the minimal k.

Safe-Z with missing candidates. Suppose C subset X are examined, P = X\C are pruned with an upper bound UB >= max_{x in P} s(x).
Let Rhat := sum_{x in P} exp(UB - m) <= |P| exp(UB - m), and Zhat := sum_{x in C} exp(s(x)-m) + Rhat.
Theorem 3.3 (monotone over-normalization and L1 bound).
Zhat >= Z, and with p_hat(x) := exp(s(x)-m)/Zhat,
||p - p_hat||_1 <= 2Rhat/(1+Rhat).
	4.	Update in O(1) with seqlock read
Each slot stores (seq, val) atomically in one cacheline. Writer: seq++(odd) -> val<-new -> seq++(even). Reader: Acquire seq1 -> val -> Acquire seq2; accept if seq1==seq2 and even. Under Release/Acquire, successful reads linearize at the even observation. Writer is wait-free (3 atomics); reader is lock-free.
	5.	Roofline-guided tile length
Let on-chip capacity be C bytes, feature dimension d, element size s bytes. If scores are not stored, per tile K_b of size T the traffic is 2Tds, so
T = floor( C / (2 d s) ).
If scores of size s’ are stored (T^2 elements), choose the largest integer T satisfying
2 T d s + T^2 s’ <= C.
	6.	CCR framework: good cover, Tot, Robin transfer
6.1 Total complex and signs. Let U = {U_i}_{i=1..P} be a finite good cover of a smooth manifold M. For p,q>=0:
C^q(U; Omega^p) := product over |S|=q+1 of Omega^p(U_S),
Tot^N := direct sum over p+q=N.
Let delta be the Cech coboundary. On form degree p, set
delta_tot := (-1)^p * delta,  D := d + delta_tot.
Then d^2=0, delta_tot^2=0, and [d,delta_tot]=0, hence D^2=0. Cartan holds: [D, iota_X] = L_X, [D, L_X] = 0.

6.2 Degree-0 blocks for attention. We work at p=0 (functions). The coend kernel and normalization objects are Tot degree q=0 operators. We realize per-patch tiles K_b of size T as fixed-size degree-0 blocks acting on local arrays. For normalization, we treat (m,Z) as elements of a commutative monoid with the associative, commutative combine:
combine((m1,Z1),(m2,Z2)) := (m, Z1exp(m1-m) + Z2exp(m2-m)),  m:=max(m1,m2).
combine is a degree-0 block; its Jacobian and adjoint are also degree-0 and tabulated offline.

6.3 Robin transfer and flatness. Define the Robin operator R_{alpha,beta} as a graded derivation (degree +1) that enforces consistency of q=0 data across overlaps via edge differences. On q=0:
(R_{alpha,beta} v){ij} := alpha * (v_i - v_j),
and extend derivationally to all bidegrees. Here v can be a scalar v in {m, Z, Zhat, H_k, …} or a small fixed tuple. Let
D_R := D + R{alpha,beta}.
Flatness condition:
D_R^2 = 0  iff  [D, R_{alpha,beta}] + R_{alpha,beta}^2 = 0.
With R acting by pure edge differences on q=0 and delta_tot carrying the Cech alternating sum, the curvature identity becomes an explicit q:0->2 overlap equation that is satisfied when m,Z,… agree on overlaps. Choosing a degree-0 incidence gauge J_c commuting with R (vertex scaling) keeps the runtime template fixed.

6.4 Contraction and HPL. Let (iota, pi, h) be bounded contraction data between (Omega^(M), d) and (Tot^, D). Define
gamma := ||h|| * alpha * ||E||   (E is the q=0 edge-difference operator on the chosen normed scale).
Assume gamma < 1 (certified offline). The perturbed homotopy
h_R := h * sum_{m>=0} (- R_{alpha,beta} h)^m
converges and yields a contraction (iota, pi, h_R) between (Omega^, d) and (Tot^, D_R). Cohomology is preserved: H^(Tot, D_R) ≅ H^_{dR}(M).

6.5 DGLA closure and degree-0 BCH split. Let g be the graded Lie algebra generated by {D_R (deg+1), iota_X (deg-1), L_X (deg 0), J_a (deg_Cech(a)), and all degree-0 attention blocks: tile-coend, combine, Safe-Z, Auto-k, semiring maps, Val-actions}. The degree-0 subalgebra generated by L_X and J_c with |c|=0 is commutative up to the product structure and reorders by BCH without creating new operators. Therefore forward, reverse, and mixed differentials of any CCR-Coend-Attention expression remain inside the same constant-size block family.
	7.	CCR encoding of Safe-Z and Auto-k
7.1 Safe-Z as degree-0. Each patch U_i carries candidates C_i and an upper bound UB_i on pruned scores; compute local (m_i, Zc_i, Rhat_i). The global Zhat is
(m, Zhat) = combine_i (m_i, Zc_i + Rhat_i).
The L1 error bound ||p - p_hat||_1 <= 2Rhat/(1+Rhat) with Rhat := sum_i Rhat_i / sum_i Zc_i
is preserved by combine because combine is monotone in each argument and rescaling is consistent.

7.2 Auto-k as degree-0. For each patch keep a descending list of local scores and their cumulative head H_{i,k}. After Zhat is known, the global stopping rule
2 (Zhat - sum_i H_{i,k_i}) / Zhat <= eps
is implemented with a constant-size heap merging the patch heads; the number of steps depends only on a fixed bound for the number of active patch heads, hence O(1) in CCR.
	8.	Complexity, stability, and correctness (CCR-integrated)
Definition 8.1 (constant time in CCR).
The number and sizes of blocks touched online do not depend on the number of patches P, mesh sizes, or expression depth; they depend only on cover multiplicity nu, tile length T, and fixed operator norms used to certify gamma<1.

Theorem 8.2 (algebraic correctness).
With S(x,y) realized by per-patch coend tiles and normalization as above, and with D_R^2=0, the glued probabilities alpha and outputs out are equal to their global definitions when no truncation is applied; with Safe-Z/Auto-k the L1 bounds remain valid post-gluing.

Theorem 8.3 (O(1) forward/reverse).
Under nu,T fixed and gamma<1, the JVP, VJP, and HVP for CCR-Coend-Attention evaluate in O(1) time and memory: they call a fixed number of constant-size degree-0 blocks (tile-coend, combine, Safe-Z, Auto-k, semiring/Val maps) and the adjoints of the same blocks.

Theorem 8.4 (numerical stability).
Two-pass log-sum-exp keeps exp(s-m) in (0,1]; combine applies only exponential differences; bounding the number of combines by a function of nu yields a backward error O(u * C(nu)) in IEEE floating point. Safe-Z/Auto-k do not destabilize normalization because they act by monotone over-normalization and truncation with explicit L1 control.
	9.	Offline tables and runtime template
Offline (once per cover, per device family):
T1 fixed tile kernels for coend on size-T blocks (for chosen S and Val).
T2 combine and its Jacobian/adjoint in closed form.
T3 Safe-Z and Auto-k small Jacobians/adjoints.
T4 incidence actions J_c for degree-0 reweightings and a gauge fixing making [R,J_c]=0.
T5 contraction data (iota, pi, h_R) and certified gamma<1 with a stored truncation order m that meets a target tail epsilon_tail.

Online (per query y):
O1 per-patch tile scans to build (m_i, Z_i) and candidate lists; O(1) state.
O2 Safe-Z local Rhat_i; combine to (m, Zhat).
O3 heap-like O(1) merge to determine Auto-k; compute out via degree-0 blocks.
O4 seqlock read for dynamic indices (O(1), linearizable).
	10.	Limitations and refresh policy
O(1) claims require a fixed cover U, fixed tile length T and bounded nu, certified gamma<1, bounded derivative tables, and calibrated candidate upper bounds. If these drift, re-run offline certification (tables, m for h_R). Worst-case distributions may force the sublinear path to degenerate to dense evaluation; the system remains correct and stable but loses sublinearity.
	11.	Proof sketches (selected)
11.1 D_R^2=0 and edge equation. With R acting by edge differences on q=0, [delta_tot, R]=0 and [d, R]=0 on functions. Expanding (D+R)^2 on bidegree (p,2) yields the overlap equation
delta_tot(R v) + R(delta_tot v) = 0,
which is satisfied precisely when v is consistent on overlaps; hence D_R^2=0.

11.2 Monoid combine and adjoint. Let C := R x R_{>0} with operation
(m1,Z1) * (m2,Z2) := (m, Z1 e^{m1-m} + Z2 e^{m2-m}), m:=max(m1,m2).
C is an Abelian monoid; the Jacobian is a constant-size matrix with entries bounded by in-range exponentials. Thus JVP/VJP are constant-size degree-0 blocks.

11.3 Preservation of L1 bounds under combine. If Zhat_i >= Z_i for all i, then for the combined pair
(m, Zhat) = *{i} (m_i, Zhat_i),  (m, Z) = *{i} (m_i, Z_i),
we have Zhat >= Z by monotonicity of * in each argument. The normalized distributions satisfy the same L1 bound as the local Safe-Z bound by applying Theorem 3.3 to the combined ratio Rhat.
	12.	Minimal pseudocode (CCR-aligned)
12.1 Per-patch streaming for a tile block K_b (degree-0 block)
function tile_coend_and_mZ(Q_tile, K_tile):
// returns (m_tile, Z_tile) and partial sums for candidates
12.2 Global normalization via combine
(m,Z) <- ( -inf, 0 )
for i in patches:
(m,Z) <- combine( (m,Z), (m_i, Z_i) )
12.3 Safe-Z + Auto-k
Zhat <- combine over i of (m_i, Zc_i + Rhat_i)
choose minimal k by 2*(Zhat - sum_i H_{i,k_i})/Zhat <= eps
12.4 Output
out <- sum over selected x of alpha(x|y) ⊙ V(x)

Conclusion.
CCR-Coend-Attention realizes profunctor attention with streaming normalization, Safe-Z, Auto-k, and O(1) updates as degree-0 Tot blocks glued by a Robin transfer on a good cover. The resulting pipeline satisfies: (i) algebraic correctness by D_R^2=0 and HPL transfer, (ii) O(1) forward/reverse/mixed derivatives by DGLA closure and BCH splitting, (iii) explicit, preserved L1 error bounds for Safe-Z/Auto-k after CCR gluing, and (iv) stable IEEE floating-point behavior due to log-domain two-pass normalization and bounded combine depth. The design is immediately implementable with fixed tables and offers constant-latency primitives for large-scale attention under semiring and quantale choices.
