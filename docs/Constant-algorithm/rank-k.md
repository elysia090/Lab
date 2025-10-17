Title: O(1) Low-Rank Update Kernel (rank-k), CCR-Compatible, Review-Ready

Abstract.
We give a constant-time (O(1)) evaluation kernel for log-determinants, selected solves (with fixed projections), selected traces, quadratic forms, and projected gradients in matrices subject to a fixed-rank update. Let A in R^{n x n} be invertible (SPD preferred) and consider A_u := A + U V^T with U,V in R^{n x k} and k fixed. All n-dependent work is performed offline against fixed dictionaries of right-hand sides, test matrices, and projections. Online queries reduce to a constant number of k x k factorizations and small dense products that do not scale with n. The kernel integrates into constant-size degree-0 blocks in Cartan-Cech-Robin (CCR) pipelines; reverse-mode derivatives reuse the same constant-size blocks. Algebraic correctness follows from the Woodbury identity and the matrix determinant lemma; numerical stability follows from standard backward-stable triangular and LDL^T solves on a k x k core with conditioning guards.
	1.	Notation and setting
Matrices are real. Dimensions: A in R^{n x n}, U,V in R^{n x k} with fixed k >= 1, n arbitrary. Let T := A^{-1} and S := T U in R^{n x k}. Define the k x k core
K := I_k + V^T S = I_k + V^T A^{-1} U.
We assume A is factorizable offline (Cholesky if SPD, otherwise pivoted LDL^T or an appropriate sparse direct factorization). All dictionaries and projections listed below are finite and fixed before any online query.
	2.	Query class and O(1) convention
Let B := {b_j}{j=1..J} be a dictionary of vectors, C := {c_i}{i=1..I} a dictionary of vectors, and P := {P_ell in R^{n x p_ell}} a set of projection matrices with bounded total projection dimension p_tot := sum_ell p_ell. Let {B_L}_{L=1..Lmax} be a set of low-rank test matrices for traces with factorizations B_L = P_L Q_L^T, rank(B_L)=r_L, sum_L r_L = r_tot bounded.
Online queries are restricted to:
(i) logdet(A_u),
(ii) projected solves: compute P_ell^T y where y = (A_u)^{-1} b_j,
(iii) selected traces: tr((A_u)^{-1} B_L),
(iv) quadratic forms: c_i^T (A_u)^{-1} c_i,
(v) projected gradients of logdet(A_u) w.r.t. U,V against fixed test directions.
Definition (O(1) query). A query is O(1) if the online arithmetic and memory traffic depend only on k, p_tot, r_tot, I, J, and not on n.
	3.	Exact algebraic identities
Woodbury identity:
(A + U V^T)^{-1} = T - S K^{-1} V^T T.
Determinant lemma:
det(A + U V^T) = det(A) det(K), hence logdet(A_u) = logdet(A) + logdet(K).
Selected trace for any B:
tr((A_u)^{-1} B) = tr(T B) - tr(K^{-1} V^T T B S).
Solve for a dictionary vector b:
y = (A_u)^{-1} b = y0 - S z, where y0 := T b and z := K^{-1} (V^T y0).
Quadratic form for c:
q = c^T (A_u)^{-1} c = c^T y0 - m^T K^{-1} m, with y0 := T c, m := V^T y0.
Gradients for fixed A. With W := K^{-1} and Frobenius pairing:
grad_U logdet = A^{-T} V W^T,   grad_V logdet = S W.
If A is SPD then A^{-T} = A^{-1}. If U = V (symmetric update), then K is SPD and W is symmetric.
	4.	Offline precomputation and storage
Inputs fixed offline: A, U, V, dictionaries B, C, projections P, and trace factors {P_L,Q_L}.
Precompute and store:
(1) Factorization of A (and logdet(A) if SPD via Cholesky).
(2) T P_ell for all projections; denote AP_ell := T P_ell.
(3) Y := [T b_j] and Y_c := [T c_i] (multiple right-hand sides).
(4) S := T U and, if needed, R := T V (for non-symmetric cases).
(5) G := V^T S, K := I_k + G, and a stable factorization of K (Cholesky if SPD, pivoted LDL^T otherwise).
(6) For each B_L = P_L Q_L^T: AP_L := T P_L and QS_L := Q_L^T S.
(7) Baseline traces tr(T B_L) via tr(AP_L Q_L^T).
(8) Adjoint data for reverse mode: cached transposed triangular or LDL^T solves.
All arrays of size O(n) or O(n * const) appear only here.

Offline complexity and memory (dense upper bounds).
Factorization of A: O(n^3). Multiple RHS solves for M columns total:
M := k + p_tot + I + J + r_tot.
Cost O(n^2 M). Memory O(n M) to store T times these columns.
Core K construction and factorization: O(k^3).
For sparse A with nnz(A) nonzeros, replace O(n^3) and O(n^2 M) by the costs of the chosen sparse direct method with the same M.
	5.	Online evaluation rules and costs
All online rules call solve_k to apply the stored factorization of K. No n-vector is formed online.
(i) logdet(A_u) = logdet(A) + logdet(K). Cost O(k^2) to sum the k log pivots.
(ii) Projected solve P_ell^T y for y = (A_u)^{-1} b_j:
y0 := Y[:,j], m := V^T y0 in R^{k}, z := solve_k(K, m),
return P_ell^T y = P_ell^T y0 - (P_ell^T S) z,
where P_ell^T y0 = (AP_ell)^T b_j was precomputable or can be read from AP_ell,Y.
Cost O(k^2 + k p_ell).
(iii) Selected trace tr((A_u)^{-1} B_L):
H_L := (V^T AP_L) (QS_L) in R^{k x k},
corr := tr(solve_k(K, H_L)),
return tr(T B_L) - corr.
Cost O(k^3 + k^2 r_L).
(iv) Quadratic form for c_i:
y0 := Y_c[:,i], m := V^T y0, return c_i^T y0 - m^T solve_k(K, m).
Cost O(k^2).
(v) Projected gradients (U,V). Using the identities in Section 3 and the projection trick in (ii), report only fixed projections of grad_U and grad_V; no n x k materialization.
Cost O(k^2) per projected component.
Therefore each query is O(1) in n.
	6.	Correctness theorems
Theorem 6.1 (Algebraic correctness).
Given T, S, and K as above, items (i)–(iv) return the exact algebraic quantities defined by Woodbury and the determinant lemma; item (v) returns exact gradients for fixed A under the Frobenius pairing.
Sketch. Substitute (A + U V^T)^{-1} = T - S K^{-1} V^T T, cyclically permute traces, and use d logdet = tr(K^{-1} dK) with dK = (dV)^T S + V^T T dU.

Theorem 6.2 (O(1) time and memory).
Assume k, p_tot, r_tot, I, J are fixed and all offline arrays are stored. Then each online query invokes a constant number of k x k solves and small dense products; no O(n)-sized array is formed online.
Sketch. Count operations in Section 5. All n-dependent quantities are precomputed and reused.
	7.	Numerical stability
Assume IEEE floating point with unit roundoff u.
Lemma 7.1 (Core stability).
If A is SPD and U = V then K is SPD. Cholesky factorization of K with solve_k is backward stable; the computed W_hat = K_hat^{-1} satisfies K + deltaK with ||deltaK|| = O(u) ||K|| and W_hat = (K + deltaK)^{-1} exactly. For general invertible A, a pivoted LDL^T factorization of K is backward stable provided sigma_min(K) >= kappa_min > 0, enforced by an offline guard.
Lemma 7.2 (Error propagation to queries).
Let cond(K) := ||K|| * ||K^{-1}|| in a consistent norm. Then:
(i) logdet(K) via Cholesky/LDL^T has absolute error O(u) * sum_p |log d_p| + O(u) * k, where d_p are the diagonal pivots; relative error is O(u * cond(K)) when pivots are uniformly bounded away from 0 and infinity.
(ii) m^T K^{-1} m computed as m^T solve_k(K, m) has relative backward error O(u) * cond(K).
(iii) tr(K^{-1} H) computed as tr(solve_k(K, H)) has backward error O(u) * cond(K) * ||H||_* (nuclear norm).
All online rules are linear in the outputs of solve_k and small products, hence inherit these bounds.
Guard policy. Track kappa_K := sigma_min(K). If kappa_K < kappa_min (preset), either add a ridge to A or re-orthogonalize U,V in the T-inner product so that ||V^T S|| stays within a preset range. Monitor rho := ||K W_hat - I||; if rho exceeds a threshold, refactor K offline.
	8.	Update variability and legitimate O(1) extensions
When U,V must vary online, impose a low-dimensional generator:
U = Phi a,  V = Psi b,
with Phi,Psi in R^{n x d} fixed offline and a,b in R^{d x k} online with fixed d. Store T Phi and T Psi offline. Then S = T U = (T Phi) a and V^T S = b^T (Psi^T T Phi) a are assembled from constant-size blocks; K builds in O(k^2 d) and factorizes in O(k^3). This preserves the O(1) property with constants depending on k and d, not on n.
	9.	CCR integration (degree-0 blocks)
In CCR pipelines, declare each of the following as degree-0 Tot blocks: projection reads P_ell^T y0, small dense products on R^{k x k} or R^{k}, and solve_k. On fixed CCR bases, degree-0 incidence actions commute with these blocks or incur a bounded narrow commutator; by BCH splitting the runtime template does not grow. Reverse mode replaces each block by its adjoint: transposed triangular/LDL^T solves for solve_k and transposes for small products. No new operator appears online.
	10.	Streaming sequences and rank control
For a stream of low-rank updates with effective rank k_eff drifting upward, maintain a T-inner-product incremental compression (QR/SVD) to project back to rank k with tolerance tau, so that ||V^T S - V_compressed^T S_compressed|| <= tau and sigma_min(K) remains above kappa_min. Trigger a refresh when rho = ||K W_hat - I|| or kappa_K degrades past thresholds.
	11.	Worked example (k = 1)
Let u,v in R^{n}, s := T u, g := v^T s (scalar), and y0 := T b.
logdet(A_u) = logdet(A) + log(1 + g).
(A_u)^{-1} b = y0 - s * (v^T y0)/(1 + g).
c^T (A_u)^{-1} c = c^T y0 - (v^T y0)^2/(1 + g) with y0 := T c.
For B = p q^T, tr((A_u)^{-1} B) = tr(T B) - (v^T T p)(q^T s)/(1 + g).
All quantities use pre-stored y0, s, T p, and scalars; online cost is constant.
	12.	Implementation guidance
(a) Prefer SPD A and symmetric updates U = V so that K is SPD and Cholesky applies.
(b) Store all small blocks in structure-of-arrays (SoA) layout; unroll k-sized kernels.
(c) For non-SPD K use pivoted LDL^T; compute slogdet via diagonal of D and keep a sign accumulator.
(d) For traces, form H_L := (V^T AP_L) (QS_L) with the fewer GEMM calls implied by ranks r_L.
(e) Cache both forward and transposed triangular/LDL^T solves for reverse mode.
(f) Expose monitoring hooks for rho and kappa_K; refactor when thresholds are crossed.
	13.	Limitations
(a) Full solution vectors y in R^{n} are out of scope online; only fixed projections P_ell^T y or fixed functionals are O(1).
(b) If U,V change arbitrarily in n-dimensional spaces, O(1) is not guaranteed; use the generator model in Section 8 or coarse parameter grids with certified interpolation.
(c) Offline memory scales with n times the total number of stored columns; ensure that M := k + p_tot + I + J + r_tot fits the target.
	14.	Complexity and memory summary
Offline (dense upper bounds):
factor(A) : O(n^3),
solves(T * M columns) : O(n^2 M),
memory : O(n M) plus O(k^2) for K and its factorization.
Online:
logdet : O(k^2),
projected solve : O(k^2 + k p_ell),
selected trace : O(k^3 + k^2 r_L),
quadratic form : O(k^2),
projected gradients : O(k^2) per reported projection.
All constants are independent of n.
	15.	Reproducibility checklist
State k, p_tot, r_tot, I, J, factorization type for A, factorization type for K, thresholds (kappa_min, rho), and any generator dimensions d. Record numeric tolerances for compression and refresh. Fix the order of floating-point reductions in k x k kernels to keep bitwise reproducibility.
	16.	Conclusion
By relocating all n-dependent operations to a one-time offline phase and restricting online queries to fixed projections and low-rank traces, the Woodbury and determinant identities become a constant-time evaluation kernel. The same k x k core K and its factorization support forward evaluations and reverse-mode derivatives. Under routine conditioning guards on K, the kernel is both algebraically exact and numerically stable, integrates as degree-0 blocks in CCR pipelines, and remains O(1) with respect to n. The design is immediately applicable to Schur-complement based PDE and graph pipelines, Gaussian and linear models, and monitoring tasks that require logdet, projected solves, and traces under low-rank updates.
