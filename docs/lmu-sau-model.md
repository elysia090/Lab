Linear Memory Unit x Streaming Attention Unit: A Constant-Time, Auditable Streaming Learning Stack

Abstract
This document specifies a production-grade architecture that fuses two complementary constant-time components: the Linear Memory Unit (LMU), a lossless hashed linear model with injective addressing and exact equivalence to a reference sparse linear learner, and the Streaming Attention Unit (SAU), a streaming attention module based on positive random features with anytime-valid e-process risk control. Together they form a single streaming system that (i) predicts and learns in O(1) per event/token with fixed worst-case bounds, (ii) provides deterministic, scan-checkable auditability, (iii) exposes clear health signals for self-regulation under distribution drift, and (iv) remains reversible and reproducible under snapshot/restore.

1. Goals and SLOs
   Functional SLOs: Each sample produces three numbers in constant time: y_hat_lmu from LMU, y_hat_sau from SAU, and a fused output y_hat = Gate * y_hat_sau + (1 - Gate) * y_hat_lmu. Gate itself is computed in constant time from predictable, logged diagnostics.
   Latency SLOs: Fixed constant steps per event; p99 <= 2 x p50; no dependence on sequence length t when SAU uses gamma in (0,1).
   Correctness SLOs: LMU is exactly equivalent to the reference sparse linear learner up to IEEE-754 rounding. SAU approximates decayed softmax attention with uniform-in-time r^{-1/2} error for gamma in (0,1); its risk control is anytime-valid via Ville's inequality.
   Availability SLOs: No reader pauses. All structure rebuilds publish with a single atomic swap and are reclaimed by epoch/hazard mechanisms. Even in pathological conditions, hot-path operations complete within configured constants.

2. Component Recap

2.1 Linear Memory Unit (LMU, exact and reversible)
Address space: Two layers with zero collision for known keys. Base is a dense weight array addressed by an MPHF over K_base and guarded by Keys_base for exact membership. Delta is a bounded cuckoo dictionary with parameters (d choices, bucket size b, stash S, relocation cap L, overflow ring Q, single emergency slot).
Hot-path bounds: LOCATE <= d * b + S + 1 primitive probes; INSERT <= d * b + L + S + 1 + c0. With defaults d=2, b=4, S=8, L=8, Q=16, C_lookup <= 17 and C_insert <= 25 (engineering can shave constants to ~19).
Generational stability: Readers pin ver_base and ver_delta for a sample via RCU/Epoch. Rebuilds run on a snapshot and publish {h2, W2, Keys2} by one release-store; readers observe by acquire-load.
Scheduling inequality: Let C be delta capacity; tau_low<tau_high load thresholds; lambda_max the upper bound on novel-key arrival rate; T_rb the rebuild+remap bound. Choose C, tau_low, tau_high so C(tau_high - tau_low) >= lambda_max * T_rb + margin. This guarantees delta does not saturate before publication and preserves the constant-time insert bound.

Reference equations (self-contained)
   Prediction: for sparse sample x_t={(id_j,val_j)},
   \(\hat{y}_{\text{lmu},t} = b + \sum_j w[id_j] val_j.\)
   Update with learning rate eta_t and L2 lambda:
   \(w[id_j] \leftarrow (1-eta_t lambda) w[id_j] + eta_t e_t val_j\),
   \(b \leftarrow b + eta_t e_t\), with residual e_t = y_t - \hat{y}_t.

2.2 Streaming Attention Unit (SAU)
Positive random features: phi_i(x) = r^{-1/2} exp(w_i^T x/ sqrt tau - ||x||^2/(2tau)), w_i ~ N(0, I). Optional pairing (w, - w) reduces variance. Exponent clipping to g in [-c,c] is logged; target clip rate rho_clip in [0.1%,1%].
Streaming sufficient statistics: R_t = gamma R_{t - 1}+phi_t v_t^T and s_t=gamma s_{t - 1}+phi_t (optionally H_t for fixed rank r_v via U^T v_t). All updates use Kahan compensation; denominators stay in extended precision until division.
Query-only whitening: phi_w(q) = diag(sigma_t^2+epsilon)^{-1/2} phi(q); denominator den = phi_w^T s_t + lambda_*(t) >= 0; readout y_hat_sau = (phi_w^T R_t)/den (or U(phi_w^T H_t/den) for low-rank).
Anytime-valid risk: Build a predictable mixture e-process E_mix(t) from self-normalized coordinates. Ville's inequality gives P(sup_t E_mix >= 1/alpha) <= alpha. Maintain a logged margin m_t = log(1/alpha) - log E_mix(t).
RJ half-split diagnostic: Split PRF coordinates into two halves; monitor RJ_gap = ||y_hat - 0.5(y_hat^(1)+y_hat^(2))||/(||y_hat||+epsilon). Under weak dependence, RJ_gap = Theta(r^{-1/2}); a time-uniform threshold C_RJ sqrt (log((t or e)/delta)/r) governs yellow/red policy.

Self-contained formulae
   Feature update: R_t = gamma R_{t - 1} + phi(k_t) v_t^T,
   s_t = gamma s_{t - 1} + phi(k_t).
   Query: phi_w(q) = diag(sigma^2+epsilon)^{-1/2} phi(q),
   y_hat_sau(q) = (phi_w(q)^T R_t) / (phi_w(q)^T s_t + lambda_*(t)).
   Mixture e-process: E_mix(t) = sum_{lambda in Lambda} pi_lambda exp( sum_{i <= t} lambda g_i - psi_i(lambda) ),
   margin m_t = log(1/alpha) - log E_mix(t).

3. Data Plane: End-to-End Flow
   Ingest: For each event (k_t, v_t), SAU updates phi_t, R_t, s_t (and H_t if used), preconditioner (mu_t, sigma_t^2), risk E_mix, and RJ diagnostics in O(1)/token with fixed r and r_v. In parallel, LMU updates per-feature weights via LOCATEREF_OR_INSERT in constant steps; bias b updates once per sample.
   Predict: LMU computes y_hat_lmu by a single pass over sparse features; SAU computes y_hat_sau given a query q with constant-time whitening+GEMV. The fusion layer computes Gate and y_hat. All three are logged with versions ver_sau/ver_base/ver_delta.
   Learn: LMU applies the exact SGD(+L2) step to the unique weight reference for each touched id; SAU updates its decayed statistics independently of the fusion result, keeping risk accounting strictly predictable.

4. Control Plane: Health-Aware Self-Regulation
   Predictability discipline: Any control variable that affects the next step's computation (lambda_*(t), beta_floor, Gate clamping) must be computed from F_{t - 1} -- i.e., based only on information logged before the step -- to preserve e-process validity and RCU consistency.
   SAU -> LMU early warning: Rising RJ_gap and shrinking margin m_t anticipate instability (variance under-resolution, denominator weakness, or drift). When either crosses a threshold persistently, the system lowers tau_low or increases C on the next LMU rebuild cycle, and temporarily reduces LMU's lr to de-risk updates.
   Denominator safety: If den falls below a predictable floor beta_floor, SAU raises lambda_*(t) by a small predictable increment; the Gate is clamped away from pure-attention in that interval. Both actions are logged and auditable.
   Gating policy: Gate = sigma(w_g^T z_gate), with z_gate composed of [RJ_gap, m_t, den, ||phi_w||_1, rolling |e|, context keys]. Gate is learned by LMU (so the gate's parameters are themselves exact and interpretable). When m_t<0 or repeated RJ "red," Gate is predictably stepped down for a fixed window and then allowed to recover exponentially.

5. Fusion Patterns (usable simultaneously)
   Attention-as-Feature: Turn SAU's stabilized readouts and diagnostics into features for LMU. Example: z_attn = [ (phi_w^T R_t)/den, (phi_w^T H_t)/den, RJ_gap, m_t, den, ||phi_w||_1 ]. LMU then learns a linear combination with injective addressing, giving full attribution and bit-reproducible updates.
   Linear-as-Gate: Let LMU learn Gate from SAU's health and context. Prediction y_hat = Gate * y_hat_sau + (1 - Gate) * y_hat_lmu. Under low confidence (m_t small, RJ large), Gate drops smoothly toward LMU; under strong margins, Gate moves toward SAU.
   Meta-linear on [y_hat_sau, y_hat_lmu]: Use LMU to regress the two-dimensional basis [y_hat_sau, y_hat_lmu] to the target. Coefficients drift with the environment and remain exactly audited thanks to LMU's injective mapping.

6. Sample-Scoped Algorithm (pseudocode, single thread)
   pin(ver_sau, ver_base, ver_delta)
   acc = b_lmu
   for (id,val) in x: if (p=LOCATEREF_LMU(id)) acc += (*p)*val
   y_hat_lmu = acc
   phi_q = phi(q); phi_w = whiten(phi_q; sigma_t^2)
   den = dot(phi_w, s_t) + lambda_*(t) // >= beta_floor
   y_hat_sau = dot(phi_w, R_t) / den // or U(dot(phi_w,H_t)/den)
   Gate = clamp( sigma(w_g^T z_gate), g_min(m_t), g_max(m_t) )
   y_hat = Gate*y_hat_sau + (1 - Gate)*y_hat_lmu
   e = y - y_hat
   for (id,val) in x:
   r = LOCATEREF_OR_INSERT_LMU(id)
   *r = (1 - lr*l2)*(*r) + lr*e*val
   b_lmu += lr*e
   SAU.ingest(k_t, v_t) // updates R_t, s_t, (H_t), mu, sigma^2, E_mix, RJ
   if RJ_red or m_t < 0: Gate_policy.decrease(); beta_floor up ; lambda_* up ; tau_low down 
   maybe_enqueue_LMU_rebuild()
   emit_audit_record({versions, RJ_gap, m_t, C_lookup_max, C_insert_max, ring_occ, rho_clip, ...})

7. Persistence and Recoverability
   Snapshot contents are minimal but sufficient for bit-equivalent restart:
   LMU: MPHF blob, W_base, Keys_base, delta dump (keys+weights+location kind), b, lr, l2, d,b,S,L,Q,C, tau_low, tau_high, hash seeds, versions.
   SAU: R_t, s_t, (H_t), mu_t, sigma_t^2, risk state (E_mix or per-lambda accumulators), PRF seeds, gamma, tau, clipping c, lambda grid and lambda_*.
   Atomicity: Write data blobs, then a versioned manifest with checksums; the manifest's Merkle link ties into the global audit chain. On recovery, either the old or the new base is referenced; delta_dump ensures no lost keys. Risk state resumes with the same predictable schedule.

8. Monitoring, Audit, and One-Pass Verification
   Merkle-chained audit record per step:
   { t, ver_sau, ver_base, ver_delta, seeds: {PRF, hash}, r, r_v, tau, gamma, rho_clip, RJ_gap, margin m_t, beta_floor, lambda_*, den stats, Kahan residuals, LMU: {C_lookup_p100, C_insert_p100, ring/stash occupancy, emergency_used}, rebuild timings, merkle_prev, merkle_curr }
   Verifier conditions:
   Merkle chain continuity holds; Ville margin m_t >= 0 for configured alpha; RJ policy thresholds respected; LMU's observed step counts never exceed configured constants; MPHF membership Keys_base[h(id)]==id for all base entries; if linear certificates are present, Farkas-style tolerances pass. The verifier runs in O(1) memory by streaming the log once.

9. Configuration and Sizing

9.1 LMU
Defaults: d=2, b=4, S=8, L=8, Q=16, tau_low=0.6, tau_high=0.8.
Capacity: From C(tau_high - tau_low) >= lambda_max * T_rb + margin, solve C >= (lambda_max * T_rb + margin)/(tau_high - tau_low). Pick margin at least the p99.9 novelty burst and 2 x the measured rebuild spread.
Hashing: 64-bit keyed hashing; rotate salts per rebuild; keep salts under a secret manager; never log plaintext salts.

9.2 SAU
Feature count r: For error target epsilon and confidence 1 - delta, r ~ C0 (r_v + log(1/delta)) / epsilon^2, with C0 driven by conditioning constants. Increase r under high d_v or low tau.
Temperature tau: Start near sqrt d and tune within x [0.7, 1.4].
Decay gamma: Choose from desired effective window L_eff = 1/(1 - gamma). For length-free guarantees use gamma in [0.95,0.995]; for gamma=1 declare a horizon T and turn on beta_floor.
Regularization lambda_*: Maintain a small lambda grid around empirical denominator quantiles; project into ( - 1/b', 1/b') using predictable envelope estimates; average predictably to form lambda_*.
Clipping c: Tune to hit rho_clip in [0.1%,1%]; log max exponent seen. Keep exponent arithmetic as exp(g - clamp), subtracting ||x||^2/(2tau) before exp to prevent overflow.

10. Sharding and Scale-Out
    Partition by a top-level hash of id so each shard maintains an independent LMU and SAU state. Route queries (q, x) consistently to the same shard during a session. Publish generations independently per shard; snapshot and restore are shard-local. For multi-tenant deployments, isolate audit chains per tenant and provide tenant-side verifiers.

11. Failure Modes and Playbooks
    SAU instability: m_t falls or RJ_gap persists in red. Actions (predictable, logged): increase beta_floor; nudge lambda_* upward; clamp Gate down; if allowed, raise r; optionally retune tau. Expect Gate to bias toward LMU until margins recover.
    LMU pressure: ring occupancy grows or emergency slot used. Actions: lower tau_low to trigger earlier rebuild; increase C at next rollout; bump rebuild parallelism; re-estimate lambda_max; temporarily reduce LMU lr to calm updates.
    Numeric anomalies: Any NaN/Inf at ingress causes the sample to be quarantined and logged; denominators remain in extended precision; never quantize s or denominators; Kahan residuals are tracked and alarmed if they drift above thresholds.
    Design-violation tripwire: If LMU ever hits the "unreachable" post-emergency path, raise a hard alarm; freeze new inserts (reads still O(1)); trigger an immediate rebuild; widen C or thresholds before resuming inserts.

12. Security and Privacy
    Keys_base prevents false-positive membership and ensures exact addressing -- no ghost collisions. Salts and seeds are rotated and kept under KMS; snapshots redact sensitive seeds. Audit logs are append-only and tamper-evident; WORM storage is recommended. Optional per-tenant encryption ensures at-rest isolation.

13. Determinism and Reproducibility Switches
    A "stable" mode fixes event order, enables compensated summation globally, pins lambda_* schedules, and disables any nondeterministic parallelism. Seeds for PRF and hashing are versioned. In stable mode, repeated runs over the same log produce bit-identical outputs (modulo platform-dependent FP).

14. Minimal Service API (illustrative)
    SAU.Ingest(k_t, v_t, t) -> Ack{ver_sau, merkle_curr}
    SAU.Query(q, t, lowrank?:bool) -> {y_hat_sau, den, RJ_gap, margin, ver_sau}
    LMU.Update(x[], y, lr, l2, t) -> Ack{ver_base, ver_delta, C_lookup_p100, C_insert_p100}
    LMU.Predict(x[], t) -> {y_hat_lmu, ver_base, ver_delta}
    Fuse.Predict(x[], q, ctx, t) -> {y_hat, y_hat_sau, y_hat_lmu, Gate, diagnostics:{RJ_gap, margin, den}, versions}
    Auditor.Append(json_blob) -> Ack{ok}

15. Worked Scenario (drift with novelty bursts)
    Phase 1 (steady): SAU margins high; RJ low; Gate ~0.8 so attention dominates; LMU keeps up with sparse linear corrections. LMU's ring near empty; rebuilds happen infrequently at tau_low=0.6.
    Phase 2 (novelty spike): RJ red 6/10; margins thin; den dips. The controller predictably raises beta_floor and lambda_*; Gate clamps to <= 0.4; LMU tau_low is lowered to 0.5 and a rebuild is enqueued early; insert remains O(1) thanks to stash/ring/emergency. p99 stays within SLO, and the Merkle audit records all actions.
    Phase 3 (stabilization): With a published base and refreshed seeds, ring drains; RJ recovers; margins rise. The controller releases the clamp gradually and returns lambda_* toward zero. The fused system reverts to attention-heavy predictions while preserving the exact linear fallback.

16. Why this pairing works
    SAU provides length-free, constant-time contextualization with quantitative, time-uniform risk signals. LMU provides an exact, interpretable, zero-collision backbone whose hot path is also strictly constant-time. The fusion layer uses SAU's health to meter how much attention to trust moment-by-moment and uses LMU to guarantee a safe landing when conditions degrade. Because both components are reversible and auditable, the entire stack admits post-hoc replay, forensics, and certification without sacrificing latency.

17. Practical defaults and first deployment
    Start with SAU low-rank values (r_v ~ 64 - 128) and r sized for epsilon ~ 3 - 5% at 95% confidence; gamma ~ 0.98; tau ~ sqrt d; c tuned to rho_clip ~ 0.5%. Start LMU with the default cuckoo parameters and C solved from your measured lambda_max and T_rb with a 2 x safety factor. Deploy the fusion as Meta-linear on [y_hat_sau, y_hat_lmu]; once RJ/margins stabilize, introduce Attention-as-Feature and Linear-as-Gate to increase adaptability and interpretability.

18. Summary
    LMU and SAU combine into a constant-time, auditable, and self-regulating streaming learner. LMU guarantees exactness and interpretability under strict O(1) per-event bounds; SAU delivers length-free contextual attention with anytime-valid risk accounting. The control plane welds them together predictably: health signals modulate gates and rebuilds without ever violating hot-path constraints. The result is a real-time system that scales, survives drift, and can be verified and replayed end-to-end -- no magic, just tight engineering.
