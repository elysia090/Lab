Title: SERA — Self-Normalized Random-Feature Attention for O(1)/token Streaming
Subtitle: Positive random features, ratio bounds, exponential decay
Status: Internal technical memo under internal review; not a formal publication.

Abstract
We present SERA, a constant-memory O(1)/token executor for softmax attention with kernel w(q,k)=exp((q·k)/τ), τ>0. Using positive random features we maintain sufficient statistics Z∈R^{r×d_v} and z∈R^r so that ŷ(q)=(φ(q)^T Z)/(φ(q)^T z+λ). We prove unbiasedness of the exponential-kernel feature map (unclipped), non-asymptotic pointwise error bounds with main term r^{-1/2}, time-uniform streaming bounds under exponential decay via a clipped-increment martingale Freedman argument, uniform-over-queries bounds on compact domains, and a minimax lower bound Ω(r^{-1/2}). We give numerically robust design rules (Kahan/Neumaier, extended-precision accumulators, exponent clipping, λ-calibration), operational monitors, and an evaluation protocol. Memory is O(r d_v+r) independent of sequence length.

0. Notation and setup
   d: input dim; d_v: value dim. Keys K={k_j}⊂R^d, values V={v_j}⊂R^{d_v}, query q∈R^d. Temperature τ>0 (default τ=√d). Number of features r. Vector norm ||·|| is Euclidean; matrix norms ||·||_F, ||·||*op. For a random variable X, sub-exponential (ψ_1) norm is ||X||*{ψ1}:=inf{t>0: E[exp(|X|/t)]≤2}.

Assumption A (regularity; enforced for theory and deployment)
A1 Input bound. ||x||≤R for all x∈{q}∪K (or inputs are L2-normalized; R known).
A2 Feature map and clipping. Draw w_i∼N(0,I_d), i=1..r. Define the unclipped positive features
φ_i(x):=r^{-1/2}·exp( w_i^T x / √τ − ||x||^2/(2τ) ).
Operational features use hard clipping at level c≥30:
φ_i^c(x):=min{ φ_i(x), r^{-1/2}·exp(c) }.
We write φ:=φ^c throughout updates/queries to ensure bounded increments.
A3 Streaming decay. γ∈(0,1]. For (k_t,v_t) at time t,
Z_t=γ Z_{t−1}+φ(k_t)v_t^T,  z_t=γ z_{t−1}+φ(k_t).
A4 Regularization. λ>0. Define β:=μ_eff+λ, where μ_eff:=inf_q E[B_t(q)]=(inf_q m(q))/(1−γ), m(q):=E[w(q,k_t)].
A5 Value bound. ||v_t||≤V_max almost surely (or v_t sub-Gaussian with parameter V_max).

Estimator and sufficient statistics
A(q):=∑_j w(q,k_j)v_j,  B(q):=∑_j w(q,k_j).
A_r(q):=φ(q)^T Z,  B_r(q):=φ(q)^T z.
ŷ(q):=A_r(q)/(B_r(q)+λ).

1. Positive random features for the exponential kernel
   Lemma 1 (unbiasedness without clipping). With φ unclipped,
   E[ φ(q)^T φ(k) ] = exp( (q·k)/τ ).
   Proof. Let w∼N(0,I_d). For any u∈R^d and t∈R, E[exp(t w^T u)]=exp(t^2||u||^2/2). Put t=1/√τ and u=q+k to get E[exp(w^T(q+k)/√τ)]=exp(||q+k||^2/(2τ)). Multiply by exp(−(||q||^2+||k||^2)/(2τ)) to obtain exp((q·k)/τ). Summing i=1..r cancels r^{-1}.

Lemma 2 (clipping bias). Let σ^2:=||q+k||^2/τ. For c≥30 there exists Δ_clip(R,τ,c)≥0 with
−Δ_clip ≤ E[ φ^c(q)^T φ^c(k) ] − exp((q·k)/τ) ≤ 0,
and
Δ_clip ≤ C · exp( − (c − σ^2/2)^2 / (2σ^2) ) · poly(σ,c).
Hence for bounded inputs (A1), choosing c∈[30,50] makes Δ_clip below the r^{-1/2} main term in our bounds.
Proof. Write X:=exp(U_q+U_k − (||q||^2+||k||^2)/(2τ)) with (U_q,U_k) jointly Gaussian, and a:=r^{-1/2}exp(c). Then φ^c(q)φ^c(k)=min{X,a^2}. Decompose E[min{X,a^2}]−E[X] over the tail event {X>a^2}. Use Gaussian tail bounds and that E[X·1_{X>a^2}] ≤ E[X·exp(−t log X)]·E[exp(t log X)·1_{X>a^2}] for a suitable t to obtain the stated form.

2. Ratio control and master decomposition
   Lemma 3 (vector ratio bound). Let a,a_0∈R^{d_v}, b,b_0∈R with b,b_0≥β>0. Then
   || a/b − a_0/b_0 || ≤ ||a−a_0||/β + ||a_0||·|b−b_0|/β^2.
   Proof. Write a/b − a_0/b_0 = [(a−a_0)b_0 + a_0(b_0−b)]/(bb_0) and bound denominators by β.
   Applied with a=A_r(q), a_0=A(q), b=B_r(q)+λ, b_0=B(q)+λ gives
   || ŷ(q) − y(q) || ≤ (num dev)/β + (||A(q)||·den dev)/β^2 + Bias_λ + Bias_clip,
   where Bias_λ:=||A(q)||·λ/β^2 and Bias_clip from Lemma 2 contributes via both numerator and denominator.

3. Pointwise non-asymptotic bounds with explicit constants
   We bound deviations of A_r and B_r using Bernstein for bounded/sub-exponential variables.

Scaling constants (deterministic, given r,c,γ,R,V_max,τ):
M_φ := r^{-1/2}·exp(c)  (pointwise bound on any φ_i)
M_prod := exp(2c)       (since ∑_i φ_i(q)φ_i(k) ≤ ∑_i M_φ^2 = r·(r^{-1}exp(2c)) = exp(2c))
K_A := M_prod·V_max     (single-step magnitude scale for numerator’s new contribution)
K_B := exp(2c)          (single-step magnitude scale for denominator’s new contribution)

Lemma 4 (coordinatewise boundedness and ψ1 norms). For fixed data {k_j,v_j} and query q, each coordinate of A_r(q) is an average of i.i.d. bounded nonnegative variables with absolute bound ≤ K_A, hence ||·||*{ψ1} ≤ K_A/ln 2. Similarly, B_r(q) has coordinate-free bound ≤ K_B and ||·||*{ψ1} ≤ K_B/ln 2.
Proof. Bounded variable X with |X|≤M satisfies ||X||_{ψ1} ≤ M/ln 2.

Proposition 1 (vector Bernstein with explicit constants). Let X_i∈R^{d_v} be i.i.d., E[X_i]=μ, ||X_i||≤K_A. Let Σ:=E[(X_i−μ)(X_i−μ)^T]. Then for any δ∈(0,1), with probability ≥1−δ,
|| (1/r)∑_{i=1}^r X_i − μ || ≤ C_1 √( (tr Σ + ||Σ||_op·log(1/δ)) / r ) + C_2 K_A (log(1/δ))/r,
for absolute constants C_1,C_2 (e.g., C_1=4, C_2=2).
We apply this to coordinates of A_r and scalar B_r using Σ bounds implied by K_A,K_B.

Theorem 1 (pointwise bound, explicit). Under Assumption A with φ=φ^c and i.i.d./ORF/SRHT features, for any fixed q and δ∈(0,1), with probability ≥1−δ,
|| ŷ(q) − y(q) || ≤
[ C_A √( (d_v + log(1/δ))/r ) + C_A' (log(1/δ))/r ] · (1/β)

* ||A(q)|| · [ C_B √( log(1/δ)/r ) + C_B' (log(1/δ))/r ] · (1/β^2)
* Bias_λ + Bias_clip,
  where we can choose
  C_A := 4·K_A,   C_A' := 2·K_A,   C_B := 4·K_B,   C_B' := 2·K_B.
  Proof. Apply Proposition 1 to A_r’s d_v coordinates with K_A and to scalar B_r with K_B, then union bound (adds log d_v absorbed into d_v+log(1/δ)), and finally Lemma 3 plus bias terms. Constants come from the explicit Bernstein form above.

Corollary 1 (feature count for target error). Fix relative error target ε and confidence 1−δ. If ||A(q)||/B(q) is O(1) and β is bounded away from 0 by design, then r = Θ( (d_v + log(1/δ))/ε^2 ) suffices.

4. Time-uniform streaming bounds (exponential decay)
   We control sup_{t≤T} errors for Z_t,z_t updated with decay γ.

Martingale setup. For a fixed q and coordinate u∈{1..d_v}, define the centered increment
Δ^A_t(u) := φ(q)^T[ φ(k_t)v_t(u) − E(φ(k_t)v_t(u) | F_{t−1}) ],
and for the denominator
Δ^B_t := φ(q)^T[ φ(k_t) − E(φ(k_t) | F_{t−1}) ].
With clipping, |Δ^A_t(u)| ≤ 2K_A and |Δ^B_t| ≤ 2K_B almost surely. Define the exponentially-weighted martingales
M^A_t(u) := ∑*{s=1}^t γ^{t−s} Δ^A_s(u),   M^B_t := ∑*{s=1}^t γ^{t−s} Δ^B_s.
Their predictable quadratic variations satisfy
V^A_t(u) := ∑*{s=1}^t γ^{2(t−s)} E[(Δ^A_s(u))^2 | F*{s−1}] ≤ C_A^2 · (1−γ^2)^{-1},
and similarly V^B_t ≤ C_B^2 · (1−γ^2)^{-1}, where C_A,C_B depend on second moments under clipping (bounded by K_A,K_B).

Freedman inequality (bounded increments). For any ε>0,
P( sup_{t≤T} M_t ≥ ε ) ≤ exp( − ε^2 / (2( v + b ε/3 )) ),
assuming V_t ≤ v and |Δ_t| ≤ b almost surely.

Theorem 2 (time-uniform bound). Under Assumption A with clipping level c and decay γ, for any δ∈(0,1) and horizon T≥1, with probability ≥1−δ,
sup_{t≤T} || ŷ_t(q) − y_t(q) || ≤
[ Ĉ_A √( (d_v + log( (1/δ)·log T )) / r ) + Ĉ_A' (log( (1/δ)·log T ))/r ] · (1/β)

* ||A_t(q)|| · [ Ĉ_B √( log( (1/δ)·log T ) / r ) + Ĉ_B' (log( (1/δ)·log T ))/r ] · (1/β^2)
* Bias_λ + Bias_clip,
  with constants inflated by at most a factor O( (1−γ)^{-1/2} ) relative to Theorem 1 due to V_t bounds.
  Proof. Apply Freedman to each coordinate martingale with b=2K_A (num) and b=2K_B (den), v=O((1−γ)^{-1}), union bound over coordinates and a logarithmic peeling over t (delivering the polylog T term), then compose via Lemma 3.

5. Uniform-over-queries bounds on compact domains
   We require Lipschitzness of ŷ(q) in q.

Gradient of φ. For unclipped φ_i,
∇φ_i(x) = φ_i(x)·( w_i/√τ − x/τ ).
Under clipping, treat φ_i(x)=min{φ_i^un(x),M_φ}. Then for almost all x (outside the measure-zero set where clipping is tight and gradient is set-valued), we have
||∇φ_i(x)|| ≤ φ_i(x)·( ||w_i||/√τ + ||x||/τ ) ≤ M_φ·( ||w_i||/√τ + R/τ ).

High-probability bound on max_i||w_i||. For δ_w∈(0,1),
P( max_i ||w_i|| ≤ L_w ) ≥ 1−δ_w,  with L_w := √d + √(2 log(r/δ_w)).

Lipschitz constants. Condition on the event {max_i||w_i||≤L_w}. Then
||∇(φ(q)^T Z)|| ≤ ||Z||_op · (∑_i ||∇φ_i(q)||^2)^{1/2}
≤ ||Z||_op · √r · M_φ · ( L_w/√τ + R/τ ).
Similarly for φ(q)^T z. Using quotient rule,
||∇ ŷ(q)|| ≤ ( ||∇(φ^T Z)|| / (den+λ) ) + ( ||φ^T Z|| · ||∇(φ^T z)|| / (den+λ)^2 ).
Bound ||φ^T Z|| ≤ ||φ||·||Z||_op ≤ √r·M_φ·||Z||_op, and den≥β−λ≥μ_eff.

Theorem 3 (uniform in q on X={||q||≤R}). With probability ≥1−(δ+δ_w),
sup_{q∈X} || ŷ(q) − y(q) || ≤ RHS_pointwise( r, δ/N(X,ε) ) + 2 L_y ε,
where N(X,ε) is an ε-covering number of X, RHS_pointwise is Theorem 1’s right-hand side, and a valid Lipschitz constant is
L_y := ( √r·M_φ·||Z||_op / β )·( L_w/√τ + R/τ )
+ ( r·M_φ^2·||Z||_op·||z|| / β^2 )·( L_w/√τ + R/τ ).
Choose ε to balance covering vs. Lipschitz terms.

6. Minimax lower bound Ω(r^{-1/2})
   We reduce to kernel-mean estimation under clipping, then transfer through the ratio map.

Construction. Fix q with ||q||≤R. Consider two distributions P_0,P_1 on k with ||k||≤R such that
| E_{P_1}[ w(q,k) ] − E_{P_0}[ w(q,k) ] | = Δ_K,
with KL(P_1||P_0) small (Le Cam two-point). Let values be deterministic with ||v||=1 and aligned to a fixed coordinate. Any estimator of y(q)=A/B reduces to estimating the kernel mean in numerator and denominator. Under clipped positive RF, each feature coordinate produces bounded observations with variance proxy bounded below by a constant >0 (depending on R,τ,c). By classical two-point or van Trees arguments for bounded/sub-exponential mean estimation, the minimax risk scales ≥ C_0 r^{-1/2}. The ratio map (a,b)↦a/(b+λ) is 1/β-Lipschitz on {b≥β−λ}, hence
inf_ŷ sup_P E||ŷ(q)−y(q)|| ≥ (C_0/β) r^{-1/2} = Ω(r^{-1/2}).
This matches Theorems 1–2 up to constants and logs.

7. Computational complexity and memory
   Per update (k_t,v_t): compute φ(k_t) in O(r d); update Z by one rank-1 GER in O(r d_v); update z in O(r).
   Per query: compute φ(q) in O(r d); num=φ(q)^T Z in O(r d_v); den=φ(q)^T z in O(r); ratio in O(d_v).
   With fixed r,d_v these are O(1)/token. Resident memory: r d_v + r scalars.

8. Numerically robust implementation rules (deployable checklists)
   R1 Precision. Default FP64 for accumulations. Keep z in extended precision if available (80-bit or software compensated). Use Kahan/Neumaier compensation per row of Z and for z.
   R2 Exponent handling. Let u_i(x)=w_i^T x/√τ − ||x||^2/(2τ). Compute per-batch u_i^cen:=u_i−max_j u_j, then exp(u_i^cen); rescale by exp(max_j u_j) only if needed. Hard-clip u_i at c (i.e., φ_i ≤ r^{-1/2}exp(c)). Record clip rate π_c.
   R3 No inverses. Only inner products and one scalar divide per query.
   R4 Quantization. Optional per-row scale quantization for Z (8/4-bit). Keep z unquantized. Re-estimate scales on drift (π_c↑ or dynamic-range widening).
   R5 Kernel fusion. Fuse φ(q) computation with GEMV: compute φ(q), then φ(q)^T Z and φ(q)^T z within one kernel to share loads.
   R6 Reciprocal. Compute inv := 1/(den+λ) via hardware rcp/rsqrt and one Newton step; reuse inv for ŷ=num·inv.
   R7 Floors/guards. Use ŷ←num/(max(den,β_floor)+λ) with β_floor≤β to prevent NaN/Inf.
   R8 Seeds/orthogonalization. Prefer ORF/SRHT for variance reduction. Persist seeds, transforms, scales for bitwise reproducibility.

9. Anytime-valid calibration and monitors
   C1 r (features). Start r from r≈C·(d_v+log(1/δ))/ε^2; double r if observed error band is violated.
   C2 τ (temperature). Default τ=√d; L2-normalize inputs. If oversensitivity or insensitivity is detected, line-search τ∈[0.7,1.4]·√d.
   C3 λ (regularization). Maintain rolling median of observed den(q) over a calibration set Q_calib. Set λ=ρ·median_den with ρ∈[0.01,0.05]. Update λ monotonically nondecreasing only (anytime-valid).
   C4 c (clip). Choose c∈{30,40,50}. Monitor π_c; if π_c>1% persistently, adjust c downward only. Optional conservative debias α_debias∈[1.0,1.05] applied multiplicatively to both num and den, but cap at 1.0.
   C5 γ (decay). Choose by effective window L_eff=1/(1−γ). For abrupt shifts, temporarily reduce γ (short window), restore after cooldown.

Online monitors (make the theory operational)
M1 shr:=B_r/(B_r+λ) distribution; target median in [0.95,0.99].
M2 π_c (clip rate) and Δ_den:=|den_true−den_meas|/den_true proxies; keep π_c small enough that Δ_clip≲ main r^{-1/2} term.
M3 V_t proxies for Freedman: track empirical quadratic variation and max increment estimates; ensure thresholds consistent with δ.
M4 Drift flags: if den falls below β_floor or π_c spikes, trigger recovery policy (Sec. 10).

10. Failure modes and recovery
    F1 Denominator thinning. If den≈0, increase λ, then raise γ to thicken μ_eff; if still unstable, temporarily switch a short local window to exact softmax and harvest calibration pairs.
    F2 Distribution shift. Re-normalize inputs; raise c; temporarily increase r.
    F3 Quantization saturation. Re-estimate per-row scales; pause Z quantization; keep z unquantized.
    F4 Underflow. If −||x||^2/(2τ) large, shrink input scale, raise c, keep FP64/extended accumulators.

11. Evaluation protocol (for reproducibility)
    Data. Long-form text, dialogue logs, continuous time series representative of deployment.
    Baselines. Exact softmax; cumulative linear attention; Performer/FAVOR+ (PORF/SRHT variants).
    Metrics. L2/L∞ and relative RMSE vs exact softmax; p50/p99 latency vs sequence length n (expect SERA flat); memory vs r; π_c; shr distribution.
    Sweeps. r∈{16,32,64,128,256,512,1024}; γ∈{1.0,0.99,0.995,0.999}; λ∈{0.5%,1%,2%,5% of median den}; c∈{30,40,50}; input scale α∈{0.25,0.5,1.0}; τ∈{0.7,1.0,1.4}·√d.
    Ablations. Clipping on/off; Kahan on/off; extended precision on/off; ORF/SRHT vs i.i.d.; λ calibration strategies.
    Expected plots. error vs r (log–log slope ≈ −1/2); latency vs n flat; shr concentrated in [0.95,0.99]; π_c<1%.

12. Related work (positioning; concise)
    SERA shares positive-RF exponential-kernel linearization with Performer-type methods (e.g., FAVOR+). Distinctives here: (i) attention as a ratio estimator with explicit denominator control β; (ii) time-uniform streaming guarantees under clipping (anytime-valid); (iii) codified numerics and calibration; (iv) audit-friendly reproducibility.

13. Reproducibility and audit (O(0) receipts)
    Persist per run: {seed, r, τ, c, λ, γ, RF type (iid/ORF/SRHT), matrix layout, build id, per-row quant scales}. Store cryptographic hashes of Z and z (e.g., Merkle roots) and acceptance thresholds. External auditors verify settings+hashes+threshold checks without raw inputs.

Appendix A. Explicit proofs and constants
A.1 Proof of Lemma 1. Already given; complete by dominated convergence to swap expectation and sum; measurability trivial as φ_i are Borel.
A.2 Proof of Lemma 2. Let U:=w^T(q+k)/√τ∼N(0,σ^2). Write X:=exp(U−σ^2/2), so E[X]=1 and φ^c product differs from X by truncation at a:=exp(c). Then
0 ≥ E[min{X,a}]−E[X] = −E[(X−a)*+].
Chernoff bound: for t∈(0,1), E[(X−a)*+] ≤ E[X·1_{X≥a}] ≤ E[X^{1−t}]·a^t·P(X≥a)^{1−t}. Using X=exp(U−σ^2/2), choose t=σ^2/(2c) (when c>σ^2/2) to minimize the exponent, obtaining
E[(X−a)*+] ≤ exp( − (c−σ^2/2)^2/(2σ^2) )·poly(σ,c).
Tensorizing over i and absorbing r via M_φ keeps the same functional form for Δ_clip.
A.3 Vector Bernstein used in Theorem 1. For bounded ||X_i||≤K, E[X_i]=μ, the standard inequality yields
P( ||(1/r)∑(X_i−μ)|| ≥ t ) ≤ 2·exp( − c·min{ r t^2 / v^2, r t / K } ),
with v^2≥||Cov(X_i)||** (trace or operator surrogate). Solving for t at confidence δ gives the explicit constants C_A,C_A'.
A.4 Freedman details for Theorem 2. For M_t=∑ γ^{t−s}Δ_s with |Δ_s|≤b, the predictable variation satisfies V_t≤(b^2/4)·(1−γ^2)^{-1} up to constants from second moments; plug into Freedman to obtain
P( sup_{t≤T} M_t ≥ ε ) ≤ exp( − ε^2 / (2( v + b ε/3 )) ).
Apply with v=O((1−γ)^{-1}) and union bound over d_v.
A.5 Lipschitz constant derivation. From quotient rule,
∇ ŷ = [ (∇φ^T Z)(den+λ) − (φ^T Z)(∇φ^T z) ] / (den+λ)^2.
Bounding ||∇φ|| by √(∑_i ||∇φ_i||^2) and using ||∇φ_i||≤M_φ(L_w/√τ + R/τ) on {max_i||w_i||≤L_w} gives the stated L_y.

Appendix B. Deterministic pseudocode (CPU/GPU-friendly)

Build/Update (streaming)

```
inputs: r, τ, γ, λ, clip c, seed; stream (k_t, v_t)
init:
  draw w_i ~ N(0, I_d) or construct ORF/SRHT
  Z ← zeros(r, d_v); z ← zeros(r); β_floor ← small positive
for t = 1..:
  # φ(k_t) in log-space with clipping
  u_i ← (w_i^T k_t)/sqrt(τ) − ||k_t||^2/(2τ)           for all i
  u_i ← min(u_i, c)                                     # hard clip in log domain
  φ_i ← r^{-1/2} * exp(u_i)
  # compensated accumulations
  Z ← γ·Z + GER(φ, v_t^T) with Kahan/Neumaier per row
  z ← γ·z + φ          with Kahan/Neumaier
  log{seed,r,τ,c,λ,γ,type}; log_hash(Z); log_hash(z)
```

Query

```
input: q
u_i ← (w_i^T q)/sqrt(τ) − ||q||^2/(2τ);  u_i ← min(u_i, c)
φ_i ← r^{-1/2} * exp(u_i)
num ← φ^T Z
den ← φ^T z
ans ← num / (max(den, β_floor) + λ)
```

Appendix C. Practical defaults and knobs
τ=√d; inputs L2-normalized. λ=1%..5% of rolling median den on a calibration set. r=128..512 typical; increase if r^{-1/2} noise dominates. c=30..50 targeting π_c≤1%. γ∈{0.99,0.995,0.999} per desired L_eff. ORF/SRHT features preferred for variance reduction. Monitors M1–M4 must be wired with alarms that trigger recovery policies F1–F4.
