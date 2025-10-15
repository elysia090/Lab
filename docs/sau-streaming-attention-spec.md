Title: Streaming Attention Unit (SAU) -- Self-Normalized Random-Feature Attention for O(1)/token Streaming
Subtitle: Positive random features, ratio bounds, exponential decay
Status: Internal technical memo under internal review; not a formal publication.

Abstract
We present the Streaming Attention Unit (SAU), a constant-memory O(1)/token executor for softmax attention with kernel w(q,k)=exp((q * k)/tau), tau>0. Using positive random features we maintain sufficient statistics Z in R^{r x d_v} and z in R^r so that y_hat(q)=(phi(q)^T Z)/(phi(q)^T z+lambda). We prove unbiasedness of the exponential-kernel feature map (unclipped), non-asymptotic pointwise error bounds with main term r^{-1/2}, time-uniform streaming bounds under exponential decay via a clipped-increment martingale Freedman argument, uniform-over-queries bounds on compact domains, and a minimax lower bound Omega(r^{-1/2}). We give numerically robust design rules (Kahan/Neumaier, extended-precision accumulators, exponent clipping, lambda-calibration), operational monitors, and an evaluation protocol. Memory is O(r d_v+r) independent of sequence length.

0. Notation and setup
   d: input dim; d_v: value dim. Keys K={k_j} subset R^d, values V={v_j} subset R^{d_v}, query q in R^d. Temperature tau>0 (default tau= sqrt d). Number of features r. Vector norm || * || is Euclidean; matrix norms || * ||_F, || * ||*op. For a random variable X, sub-exponential (psi_1) norm is ||X||*{psi1}:=inf{t>0: E[exp(|X|/t)] <= 2}.

Assumption A (regularity; enforced for theory and deployment)
A1 Input bound. ||x|| <= R for all x in {q} union K (or inputs are L2-normalized; R known).
A2 Feature map and clipping. Draw w_i ~ N(0,I_d), i=1..r. Define the unclipped positive features
phi_i(x):=r^{-1/2} * exp( w_i^T x / sqrt tau - ||x||^2/(2tau) ).
Operational features use hard clipping at level c >= 30:
phi_i^c(x):=min{ phi_i(x), r^{-1/2} * exp(c) }.
We write phi:=phi^c throughout updates/queries to ensure bounded increments.
A3 Streaming decay. gamma in (0,1]. For (k_t,v_t) at time t,
Z_t=gamma Z_{t - 1}+phi(k_t)v_t^T, z_t=gamma z_{t - 1}+phi(k_t).
A4 Regularization. lambda>0. Define beta:=mu_eff+lambda, where mu_eff:=inf_q E[B_t(q)]=(inf_q m(q))/(1 - gamma), m(q):=E[w(q,k_t)].
A5 Value bound. ||v_t|| <= V_max almost surely (or v_t sub-Gaussian with parameter V_max).

Estimator and sufficient statistics
A(q):= sum _j w(q,k_j)v_j, B(q):= sum _j w(q,k_j).
A_r(q):=phi(q)^T Z, B_r(q):=phi(q)^T z.
y_hat(q):=A_r(q)/(B_r(q)+lambda).

1. Positive random features for the exponential kernel
   Lemma 1 (unbiasedness without clipping). With phi unclipped,
   E[ phi(q)^T phi(k) ] = exp( (q * k)/tau ).
   Proof. Let w ~ N(0,I_d). For any u in R^d and t in R, E[exp(t w^T u)]=exp(t^2||u||^2/2). Put t=1/ sqrt tau and u=q+k to get E[exp(w^T(q+k)/ sqrt tau)]=exp(||q+k||^2/(2tau)). Multiply by exp( - (||q||^2+||k||^2)/(2tau)) to obtain exp((q * k)/tau). Summing i=1..r cancels r^{-1}.

Lemma 2 (clipping bias). Let sigma^2:=||q+k||^2/tau. For c >= 30 there exists Delta_clip(R,tau,c) >= 0 with
 - Delta_clip <= E[ phi^c(q)^T phi^c(k) ] - exp((q * k)/tau) <= 0,
and
Delta_clip <= C * exp( - (c - sigma^2/2)^2 / (2sigma^2) ) * poly(sigma,c).
Hence for bounded inputs (A1), choosing c in [30,50] makes Delta_clip below the r^{-1/2} main term in our bounds.
Proof. Write X:=exp(U_q+U_k - (||q||^2+||k||^2)/(2tau)) with (U_q,U_k) jointly Gaussian, and a:=r^{-1/2}exp(c). Then phi^c(q)phi^c(k)=min{X,a^2}. Decompose E[min{X,a^2}] - E[X] over the tail event {X>a^2}. Use Gaussian tail bounds and that E[X * 1_{X>a^2}] <= E[X * exp( - t log X)] * E[exp(t log X) * 1_{X>a^2}] for a suitable t to obtain the stated form.

2. Ratio control and master decomposition
   Lemma 3 (vector ratio bound). Let a,a_0 in R^{d_v}, b,b_0 in R with b,b_0 >= beta>0. Then
   || a/b - a_0/b_0 || <= ||a - a_0||/beta + ||a_0|| * |b - b_0|/beta^2.
   Proof. Write a/b - a_0/b_0 = [(a - a_0)b_0 + a_0(b_0 - b)]/(bb_0) and bound denominators by beta.
   Applied with a=A_r(q), a_0=A(q), b=B_r(q)+lambda, b_0=B(q)+lambda gives
   || y_hat(q) - y(q) || <= (num dev)/beta + (||A(q)|| * den dev)/beta^2 + Bias_lambda + Bias_clip,
   where Bias_lambda:=||A(q)|| * lambda/beta^2 and Bias_clip from Lemma 2 contributes via both numerator and denominator.

3. Pointwise non-asymptotic bounds with explicit constants
   We bound deviations of A_r and B_r using Bernstein for bounded/sub-exponential variables.

Scaling constants (deterministic, given r,c,gamma,R,V_max,tau):
M_phi := r^{-1/2} * exp(c) (pointwise bound on any phi_i)
M_prod := exp(2c) (since sum _i phi_i(q)phi_i(k) <= sum _i M_phi^2 = r * (r^{-1}exp(2c)) = exp(2c))
K_A := M_prod * V_max (single-step magnitude scale for numerator's new contribution)
K_B := exp(2c) (single-step magnitude scale for denominator's new contribution)

Lemma 4 (coordinatewise boundedness and psi1 norms). For fixed data {k_j,v_j} and query q, each coordinate of A_r(q) is an average of i.i.d. bounded nonnegative variables with absolute bound <= K_A, hence || * ||*{psi1} <= K_A/ln 2. Similarly, B_r(q) has coordinate-free bound <= K_B and || * ||*{psi1} <= K_B/ln 2.
Proof. Bounded variable X with |X| <= M satisfies ||X||_{psi1} <= M/ln 2.

Proposition 1 (vector Bernstein with explicit constants). Let X_i in R^{d_v} be i.i.d., E[X_i]=mu, ||X_i|| <= K_A. Let Sigma:=E[(X_i - mu)(X_i - mu)^T]. Then for any delta in (0,1), with probability >= 1 - delta,
|| (1/r) sum _{i=1}^r X_i - mu || <= C_1 sqrt ( (tr Sigma + ||Sigma||_op * log(1/delta)) / r ) + C_2 K_A (log(1/delta))/r,
for absolute constants C_1,C_2 (e.g., C_1=4, C_2=2).
We apply this to coordinates of A_r and scalar B_r using Sigma bounds implied by K_A,K_B.

Theorem 1 (pointwise bound, explicit). Under Assumption A with phi=phi^c and i.i.d./ORF/SRHT features, for any fixed q and delta in (0,1), with probability >= 1 - delta,
|| y_hat(q) - y(q) || <= 
[ C_A sqrt ( (d_v + log(1/delta))/r ) + C_A' (log(1/delta))/r ] * (1/beta)

* ||A(q)|| * [ C_B sqrt ( log(1/delta)/r ) + C_B' (log(1/delta))/r ] * (1/beta^2)
* Bias_lambda + Bias_clip,
  where we can choose
  C_A := 4 * K_A, C_A' := 2 * K_A, C_B := 4 * K_B, C_B' := 2 * K_B.
  Proof. Apply Proposition 1 to A_r's d_v coordinates with K_A and to scalar B_r with K_B, then union bound (adds log d_v absorbed into d_v+log(1/delta)), and finally Lemma 3 plus bias terms. Constants come from the explicit Bernstein form above.

Corollary 1 (feature count for target error). Fix relative error target epsilon and confidence 1 - delta. If ||A(q)||/B(q) is O(1) and beta is bounded away from 0 by design, then r = Theta( (d_v + log(1/delta))/epsilon^2 ) suffices.

4. Time-uniform streaming bounds (exponential decay)
   We control sup_{t <= T} errors for Z_t,z_t updated with decay gamma.

Martingale setup. For a fixed q and coordinate u in {1..d_v}, define the centered increment
Delta^A_t(u) := phi(q)^T[ phi(k_t)v_t(u) - E(phi(k_t)v_t(u) | F_{t - 1}) ],
and for the denominator
Delta^B_t := phi(q)^T[ phi(k_t) - E(phi(k_t) | F_{t - 1}) ].
With clipping, |Delta^A_t(u)| <= 2K_A and |Delta^B_t| <= 2K_B almost surely. Define the exponentially-weighted martingales
M^A_t(u) := sum *{s=1}^t gamma^{t - s} Delta^A_s(u), M^B_t := sum *{s=1}^t gamma^{t - s} Delta^B_s.
Their predictable quadratic variations satisfy
V^A_t(u) := sum *{s=1}^t gamma^{2(t - s)} E[(Delta^A_s(u))^2 | F*{s - 1}] <= C_A^2 * (1 - gamma^2)^{-1},
and similarly V^B_t <= C_B^2 * (1 - gamma^2)^{-1}, where C_A,C_B depend on second moments under clipping (bounded by K_A,K_B).

Freedman inequality (bounded increments). For any epsilon>0,
P( sup_{t <= T} M_t >= epsilon ) <= exp( - epsilon^2 / (2( v + b epsilon/3 )) ),
assuming V_t <= v and |Delta_t| <= b almost surely.

Theorem 2 (time-uniform bound). Under Assumption A with clipping level c and decay gamma, for any delta in (0,1) and horizon T >= 1, with probability >= 1 - delta,
sup_{t <= T} || y_hat_t(q) - y_t(q) || <= 
[ C_hat_A sqrt ( (d_v + log( (1/delta) * log T )) / r ) + C_hat_A' (log( (1/delta) * log T ))/r ] * (1/beta)

* ||A_t(q)|| * [ C_hat_B sqrt ( log( (1/delta) * log T ) / r ) + C_hat_B' (log( (1/delta) * log T ))/r ] * (1/beta^2)
* Bias_lambda + Bias_clip,
  with constants inflated by at most a factor O( (1 - gamma)^{-1/2} ) relative to Theorem 1 due to V_t bounds.
  Proof. Apply Freedman to each coordinate martingale with b=2K_A (num) and b=2K_B (den), v=O((1 - gamma)^{-1}), union bound over coordinates and a logarithmic peeling over t (delivering the polylog T term), then compose via Lemma 3.

5. Uniform-over-queries bounds on compact domains
   We require Lipschitzness of y_hat(q) in q.

Gradient of phi. For unclipped phi_i,
 nabla phi_i(x) = phi_i(x) * ( w_i/ sqrt tau - x/tau ).
Under clipping, treat phi_i(x)=min{phi_i^un(x),M_phi}. Then for almost all x (outside the measure-zero set where clipping is tight and gradient is set-valued), we have
|| nabla phi_i(x)|| <= phi_i(x) * ( ||w_i||/ sqrt tau + ||x||/tau ) <= M_phi * ( ||w_i||/ sqrt tau + R/tau ).

High-probability bound on max_i||w_i||. For delta_w in (0,1),
P( max_i ||w_i|| <= L_w ) >= 1 - delta_w, with L_w := sqrt d + sqrt (2 log(r/delta_w)).

Lipschitz constants. Condition on the event {max_i||w_i|| <= L_w}. Then
|| nabla (phi(q)^T Z)|| <= ||Z||_op * ( sum _i || nabla phi_i(q)||^2)^{1/2}
 <= ||Z||_op * sqrt r * M_phi * ( L_w/ sqrt tau + R/tau ).
Similarly for phi(q)^T z. Using quotient rule,
|| nabla y_hat(q)|| <= ( || nabla (phi^T Z)|| / (den+lambda) ) + ( ||phi^T Z|| * || nabla (phi^T z)|| / (den+lambda)^2 ).
Bound ||phi^T Z|| <= ||phi|| * ||Z||_op <= sqrt r * M_phi * ||Z||_op, and den >= beta - lambda >= mu_eff.

Theorem 3 (uniform in q on X={||q|| <= R}). With probability >= 1 - (delta+delta_w),
sup_{q in X} || y_hat(q) - y(q) || <= RHS_pointwise( r, delta/N(X,epsilon) ) + 2 L_y epsilon,
where N(X,epsilon) is an epsilon-covering number of X, RHS_pointwise is Theorem 1's right-hand side, and a valid Lipschitz constant is
L_y := ( sqrt r * M_phi * ||Z||_op / beta ) * ( L_w/ sqrt tau + R/tau )
+ ( r * M_phi^2 * ||Z||_op * ||z|| / beta^2 ) * ( L_w/ sqrt tau + R/tau ).
Choose epsilon to balance covering vs. Lipschitz terms.

6. Minimax lower bound Omega(r^{-1/2})
   We reduce to kernel-mean estimation under clipping, then transfer through the ratio map.

Construction. Fix q with ||q|| <= R. Consider two distributions P_0,P_1 on k with ||k|| <= R such that
| E_{P_1}[ w(q,k) ] - E_{P_0}[ w(q,k) ] | = Delta_K,
with KL(P_1||P_0) small (Le Cam two-point). Let values be deterministic with ||v||=1 and aligned to a fixed coordinate. Any estimator of y(q)=A/B reduces to estimating the kernel mean in numerator and denominator. Under clipped positive RF, each feature coordinate produces bounded observations with variance proxy bounded below by a constant >0 (depending on R,tau,c). By classical two-point or van Trees arguments for bounded/sub-exponential mean estimation, the minimax risk scales >= C_0 r^{-1/2}. The ratio map (a,b) -> a/(b+lambda) is 1/beta-Lipschitz on {b >= beta - lambda}, hence
inf_y_hat sup_P E||y_hat(q) - y(q)|| >= (C_0/beta) r^{-1/2} = Omega(r^{-1/2}).
This matches Theorems 1 - 2 up to constants and logs.

7. Computational complexity and memory
   Per update (k_t,v_t): compute phi(k_t) in O(r d); update Z by one rank-1 GER in O(r d_v); update z in O(r).
   Per query: compute phi(q) in O(r d); num=phi(q)^T Z in O(r d_v); den=phi(q)^T z in O(r); ratio in O(d_v).
   With fixed r,d_v these are O(1)/token. Resident memory: r d_v + r scalars.

8. Numerically robust implementation rules (deployable checklists)
   R1 Precision. Default FP64 for accumulations. Keep z in extended precision if available (80-bit or software compensated). Use Kahan/Neumaier compensation per row of Z and for z.
   R2 Exponent handling. Let u_i(x)=w_i^T x/ sqrt tau - ||x||^2/(2tau). Compute per-batch u_i^cen:=u_i - max_j u_j, then exp(u_i^cen); rescale by exp(max_j u_j) only if needed. Hard-clip u_i at c (i.e., phi_i <= r^{-1/2}exp(c)). Record clip rate pi_c.
   R3 No inverses. Only inner products and one scalar divide per query.
   R4 Quantization. Optional per-row scale quantization for Z (8/4-bit). Keep z unquantized. Re-estimate scales on drift (pi_c up or dynamic-range widening).
   R5 Kernel fusion. Fuse phi(q) computation with GEMV: compute phi(q), then phi(q)^T Z and phi(q)^T z within one kernel to share loads.
   R6 Reciprocal. Compute inv := 1/(den+lambda) via hardware rcp/rsqrt and one Newton step; reuse inv for y_hat=num * inv.
   R7 Floors/guards. Use y_hat <- num/(max(den,beta_floor)+lambda) with beta_floor <= beta to prevent NaN/Inf.
   R8 Seeds/orthogonalization. Prefer ORF/SRHT for variance reduction. Persist seeds, transforms, scales for bitwise reproducibility.

9. Anytime-valid calibration and monitors
   C1 r (features). Start r from r ~ C * (d_v+log(1/delta))/epsilon^2; double r if observed error band is violated.
   C2 tau (temperature). Default tau= sqrt d; L2-normalize inputs. If oversensitivity or insensitivity is detected, line-search tau in [0.7,1.4] * sqrt d.
   C3 lambda (regularization). Maintain rolling median of observed den(q) over a calibration set Q_calib. Set lambda=rho * median_den with rho in [0.01,0.05]. Update lambda monotonically nondecreasing only (anytime-valid).
   C4 c (clip). Choose c in {30,40,50}. Monitor pi_c; if pi_c>1% persistently, adjust c downward only. Optional conservative debias alpha_debias in [1.0,1.05] applied multiplicatively to both num and den, but cap at 1.0.
   C5 gamma (decay). Choose by effective window L_eff=1/(1 - gamma). For abrupt shifts, temporarily reduce gamma (short window), restore after cooldown.

Online monitors (make the theory operational)
M1 shr:=B_r/(B_r+lambda) distribution; target median in [0.95,0.99].
M2 pi_c (clip rate) and Delta_den:=|den_true - den_meas|/den_true proxies; keep pi_c small enough that Delta_clip <= main r^{-1/2} term.
M3 V_t proxies for Freedman: track empirical quadratic variation and max increment estimates; ensure thresholds consistent with delta.
M4 Drift flags: if den falls below beta_floor or pi_c spikes, trigger recovery policy (Sec. 10).

10. Failure modes and recovery
    F1 Denominator thinning. If den ~ 0, increase lambda, then raise gamma to thicken mu_eff; if still unstable, temporarily switch a short local window to exact softmax and harvest calibration pairs.
    F2 Distribution shift. Re-normalize inputs; raise c; temporarily increase r.
    F3 Quantization saturation. Re-estimate per-row scales; pause Z quantization; keep z unquantized.
    F4 Underflow. If - ||x||^2/(2tau) large, shrink input scale, raise c, keep FP64/extended accumulators.

11. Evaluation protocol (for reproducibility)
    Data. Long-form text, dialogue logs, continuous time series representative of deployment.
    Baselines. Exact softmax; cumulative linear attention; Performer/FAVOR+ (PORF/SRHT variants).
    Metrics. L2/L infinity and relative RMSE vs exact softmax; p50/p99 latency vs sequence length n (expect SAU flat); memory vs r; pi_c; shr distribution.
    Sweeps. r in {16,32,64,128,256,512,1024}; gamma in {1.0,0.99,0.995,0.999}; lambda in {0.5%,1%,2%,5% of median den}; c in {30,40,50}; input scale alpha in {0.25,0.5,1.0}; tau in {0.7,1.0,1.4} * sqrt d.
    Ablations. Clipping on/off; Kahan on/off; extended precision on/off; ORF/SRHT vs i.i.d.; lambda calibration strategies.
    Expected plots. error vs r (log - log slope ~ - 1/2); latency vs n flat; shr concentrated in [0.95,0.99]; pi_c<1%.

12. Related work (positioning; concise)
    SAU shares positive-RF exponential-kernel linearization with Performer-type methods (e.g., FAVOR+). Distinctives here: (i) attention as a ratio estimator with explicit denominator control beta; (ii) time-uniform streaming guarantees under clipping (anytime-valid); (iii) codified numerics and calibration; (iv) audit-friendly reproducibility.

13. Reproducibility and audit (O(0) receipts)
    Persist per run: {seed, r, tau, c, lambda, gamma, RF type (iid/ORF/SRHT), matrix layout, build id, per-row quant scales}. Store cryptographic hashes of Z and z (e.g., Merkle roots) and acceptance thresholds. External auditors verify settings+hashes+threshold checks without raw inputs.

Appendix A. Explicit proofs and constants
A.1 Proof of Lemma 1. Already given; complete by dominated convergence to swap expectation and sum; measurability trivial as phi_i are Borel.
A.2 Proof of Lemma 2. Let U:=w^T(q+k)/ sqrt tau ~ N(0,sigma^2). Write X:=exp(U - sigma^2/2), so E[X]=1 and phi^c product differs from X by truncation at a:=exp(c). Then
0 >= E[min{X,a}] - E[X] = - E[(X - a)*+].
Chernoff bound: for t in (0,1), E[(X - a)*+] <= E[X * 1_{X >= a}] <= E[X^{1 - t}] * a^t * P(X >= a)^{1 - t}. Using X=exp(U - sigma^2/2), choose t=sigma^2/(2c) (when c>sigma^2/2) to minimize the exponent, obtaining
E[(X - a)*+] <= exp( - (c - sigma^2/2)^2/(2sigma^2) ) * poly(sigma,c).
Tensorizing over i and absorbing r via M_phi keeps the same functional form for Delta_clip.
A.3 Vector Bernstein used in Theorem 1. For bounded ||X_i|| <= K, E[X_i]=mu, the standard inequality yields
P( ||(1/r) sum (X_i - mu)|| >= t ) <= 2 * exp( - c * min{ r t^2 / v^2, r t / K } ),
with v^2 >= ||Cov(X_i)||** (trace or operator surrogate). Solving for t at confidence delta gives the explicit constants C_A,C_A'.
A.4 Freedman details for Theorem 2. For M_t= sum gamma^{t - s}Delta_s with |Delta_s| <= b, the predictable variation satisfies V_t <= (b^2/4) * (1 - gamma^2)^{-1} up to constants from second moments; plug into Freedman to obtain
P( sup_{t <= T} M_t >= epsilon ) <= exp( - epsilon^2 / (2( v + b epsilon/3 )) ).
Apply with v=O((1 - gamma)^{-1}) and union bound over d_v.
A.5 Lipschitz constant derivation. From quotient rule,
 nabla y_hat = [ ( nabla phi^T Z)(den+lambda) - (phi^T Z)( nabla phi^T z) ] / (den+lambda)^2.
Bounding || nabla phi|| by sqrt ( sum _i || nabla phi_i||^2) and using || nabla phi_i|| <= M_phi(L_w/ sqrt tau + R/tau) on {max_i||w_i|| <= L_w} gives the stated L_y.

Appendix B. Deterministic pseudocode (CPU/GPU-friendly)

Build/Update (streaming)

```
inputs: r, tau, gamma, lambda, clip c, seed; stream (k_t, v_t)
init:
  draw w_i ~ N(0, I_d) or construct ORF/SRHT
  Z <- zeros(r, d_v); z <- zeros(r); beta_floor <- small positive
for t = 1..:
  # phi(k_t) in log-space with clipping
  u_i <- (w_i^T k_t)/sqrt(tau) - ||k_t||^2/(2tau) for all i
  u_i <- min(u_i, c) # hard clip in log domain
  phi_i <- r^{-1/2} * exp(u_i)
  # compensated accumulations
  Z <- gamma * Z + GER(phi, v_t^T) with Kahan/Neumaier per row
  z <- gamma * z + phi with Kahan/Neumaier
  log{seed,r,tau,c,lambda,gamma,type}; log_hash(Z); log_hash(z)
```

Query

```
input: q
u_i <- (w_i^T q)/sqrt(tau) - ||q||^2/(2tau); u_i <- min(u_i, c)
phi_i <- r^{-1/2} * exp(u_i)
num <- phi^T Z
den <- phi^T z
ans <- num / (max(den, beta_floor) + lambda)
```

Appendix C. Practical defaults and knobs
tau= sqrt d; inputs L2-normalized. lambda=1%..5% of rolling median den on a calibration set. r=128..512 typical; increase if r^{-1/2} noise dominates. c=30..50 targeting pi_c <= 1%. gamma in {0.99,0.995,0.999} per desired L_eff. ORF/SRHT features preferred for variance reduction. Monitors M1 - M4 must be wired with alarms that trigger recovery policies F1 - F4.
