# O(1) Low-Rank Update Kernel (rank-k), CCR-Compatible, Review-Ready

## Abstract
We present a constant-time (O(1)) evaluation kernel for solves, log-determinants, selected traces, quadratic forms, and gradients in matrices subject to fixed-rank updates. Let A be an offline baseline matrix and consider A_u := A + U V^T with U,V in R^{n x k} and k fixed. All n-dependent work is performed offline on a fixed dictionary of right-hand sides, test matrices, and projections. Online queries reduce to a constant number of k x k factorizations and small dense products. The kernel integrates with the Cartan–Čech–Robin (CCR) framework: degree-0 incidence actions remain in a BCH-split, reverse mode is the adjoint of the same constant-size blocks, and no global solves appear at runtime. Correctness is exact at the algebraic level; numerical stability is ensured by conditioning guards on a k x k core.
## 1. Setting and query class
Matrices are real; n is arbitrary. The baseline A is invertible (SPD preferred), fixed offline. The update is A_u := A + U V^T with rank k fixed. A query is any of the following, with all test objects drawn from fixed offline dictionaries:
(i) logdet(A_u);
(ii) selected solves (A_u)^{-1} b projected to a fixed low-dimensional subspace;
(iii) selected traces tr((A_u)^{-1} B) where B has fixed low rank;
(iv) quadratic forms c^T (A_u)^{-1} c for c in a fixed set;
(v) gradients of logdet(A_u) with respect to U and V, reported through fixed projections.
Constant time means independence of n for every query in this class, under fixed k and fixed dictionaries.
## 2. Exact identities
Let T := A^{-1}, S := T U in R^{n x k}, and define the k x k core
K := I_k + V^T S = I_k + V^T A^{-1} U.
### Woodbury identity
(A + U V^T)^{-1} = T − S K^{-1} V^T T.
### Matrix determinant lemma
det(A + U V^T) = det(A) det(K), hence logdet(A_u) = logdet(A) + logdet(K).
### Selected trace for any B
tr((A_u)^{-1} B) = tr(T B) − tr(K^{-1} V^T T B S).
### Solve for a dictionary vector b
y := (A_u)^{-1} b = y0 − S z, with y0 := T b and z := K^{-1} (V^T y0).
### Quadratic form for c
q := c^T (A_u)^{-1} c = c^T y0 − (V^T y0)^T K^{-1} (V^T y0) with y0 := T c.
### Gradients of logdet with A fixed
Let W := K^{-1}.
d(logdet) = tr(W dK),  dK = (dV)^T S + V^T T dU, hence with the Frobenius inner product
grad_U logdet = A^{-T} V W^T,   grad_V logdet = S W.
For SPD A one has A^{-T} = A^{-1}; if U = V (symmetric update) then K is SPD and W = W^T.
## 3. Offline precomputation and storage
Fix dictionaries B := {b_j}{j=1..J}, C := {c_i}{i=1..I}, and a set of projection matrices P := {P_\ell in R^{n x p_\ell}} with bounded total projection dimension. For each low-rank test matrix B_l, store a factorization B_l = P_l Q_l^T of rank r_l with sum r_l bounded.
### Precompute and store
(1) Baseline scalars: logdet(A) (via Cholesky if SPD). For each B_l, store baseline traces tr(T B_l). For each projection P_\ell, store A^{-1} P_\ell.
(2) Dictionary solves: Y := [T b_j] and Y_c := [T c_i].
(3) Update sticks: S := T U and, if needed, R := T V (for non-symmetric cases).
(4) Core and factorization: G := V^T S, K := I_k + G, together with a stable factorization of K (Cholesky if SPD; pivoted LDL^T otherwise).
(5) Selected-trace auxiliaries: for each B_l = P_l Q_l^T, store AP_l := T P_l and QS := Q_l^T S.
(6) Adjoint data for reverse mode: adjoints of all stored blocks under the chosen inner products in Tot overlaps.
All O(n*) objects are offline. Online accesses use only k x k solves and small dense products.
## 4. Online evaluation rules (constant-time)
Use the stored factorization of K; denote by solve_k the associated triangular or LDL^T solve.
(i) logdet(A_u) = logdet(A) + logdet(K). The second term is the sum of k log-pivots; cost O(k^2).
(ii) Projected solve. For a fixed projection P_\ell, report P_\ell^T y with y = (A_u)^{-1} b_j using
P_\ell^T y = P_\ell^T y0 − (P_\ell^T S) z,  y0 := column j of Y,  z := solve_k(K, V^T y0).
No n-vector is formed online; cost depends only on k and p_\ell.
(iii) Selected trace. For B_l = P_l Q_l^T, compute
corr := tr(K^{-1} V^T T B_l S) = tr(K^{-1} H_l) with H_l := (V^T AP_l)(QS) in R^{k x k}.
Return tr(T B_l) − tr(solve_k(K, H_l)). Cost O(k^3 + k^2 r_l).
(iv) Quadratic form. With c_i in the dictionary, let y0 := column i of Y_c and m := V^T y0. Then
q = c_i^T y0 − m^T solve_k(K, m). Cost O(k^2).
(v) Gradients of logdet. Report only projected gradients against fixed test directions U_test, V_test using the formulas in Section 2 and the projection trick as in (ii). No n x k object is materialized online.
## 5. CCR integration
Represent each stored offline block on Tot overlaps using the CCR bases (finite band-limit K_overlaps). Degree-0 incidence actions J_c commute with L_X and the rank-k kernel; by the degree-0 BCH split they reorder without growing the template. Online evaluation invokes a fixed number of Tot-local blocks: projection, small dense products, and solve_k. Reverse mode replaces each block by its adjoint; the factorization adjoint uses the standard triangular/LDL^T transposed solves. No new operators appear.
## 6. Correctness and complexity theorems
### Theorem 6
1 (Algebraic correctness).
With T, S, and K defined as above, items (i)–(iv) evaluate the exact algebraic quantities implied by the Woodbury identity and the matrix determinant lemma; item (v) returns exact gradients under the Frobenius pairing when A is fixed.
Sketch. Substitute (A + U V^T)^{-1} = T − S K^{-1} V^T T and use tr(XY)=tr(YX). For gradients, d logdet = tr(K^{-1} dK) with dK linear in dU and dV yields the stated forms.

### Theorem 6
2 (O(1) time and memory).
Assume k, the ranks r_l, and projection dimensions p_\ell are fixed, and that all dictionaries are finite and fixed offline. Then each online query uses a constant number of k x k solves and small dense products; no n-dependent allocation or arithmetic occurs at runtime.
Sketch. Count operations in Section 4; all n-dependent arrays are precomputed once and reused.

### Theorem 6
3 (Stability).
If A is SPD and U=V, then K is SPD and the Cholesky-based solve_k is backward stable. For general invertible A, a pivoted LDL^T factorization of K is backward stable provided sigma_min(K) >= kappa_min > 0, enforced by an offline guard.
Sketch. Standard stability of triangular and LDL^T solves; all online paths touch only K and precomputed arrays.
## 7. Certification and refresh
Define kappa_K := sigma_min(K) and monitor its offline lower bound. If a slowly drifting parameterization of U,V is allowed, enforce kappa_K >= kappa_min > 0 by either a ridge on A or by re-orthogonalizing columns of U and rows of V^T relative to the T-inner product. Track the residual rho := ||K W − I|| for W from solve_k; if rho exceeds a fixed threshold, refactor K offline. For SPD A with U=V, enforce a uniform spectral bound on V^T S by shrinking the update energy.
## 8. Scope, limitations, and domain notes
The O(1) guarantee holds for the query class in Section 1. Materializing a full solution vector y in R^n is not O(1); instead, report fixed projections or functionals. Extensions to streaming sequences of low-rank updates are permitted so long as an effective rank k_eff remains bounded; when k_eff grows, compress back to rank k offline. When integrating with PDE or graph pipelines, take A as the offline Schur complement on overlaps so that U,V lie in the same finite CCR bases; this preserves constant block sizes.
## 9. Worked example (k=1)
Let u,v in R^n, s := T u, g := v^T s (scalar), and y0 := T b.
logdet(A_u) = logdet(A) + log(1 + g).
(A_u)^{-1} b = y0 − s * (v^T y0)/(1 + g).
c^T (A_u)^{-1} c = c^T y0 − (v^T y0)^2/(1 + g) with y0 := T c.
For B = p q^T (rank 1), tr((A_u)^{-1} B) = tr(T B) − (v^T T p)(q^T s)/(1 + g).
All quantities use only pre-stored y0, s, T p and small scalars; online cost is constant.
## 10. Implementation guidance
Prefer SPD A and symmetric updates U=V to keep K SPD. When A varies in a small parameter box, pretabulate S(theta), Y(theta), and projection images at a coarse grid and use constant-degree interpolation with certified bounds. In CCR code, declare the k x k solve and all small dense products as degree-0 Tot blocks so that BCH splitting keeps the runtime template fixed in size. For reverse mode, cache the triangular solves’ transposed variants and reuse the same call graph.
