Hlm × Sera: A Constant-Time, Auditable Streaming Learning Stack

Abstract
This document specifies a production-grade architecture that fuses two complementary constant-time components: Hlm, a lossless hashed linear model with injective addressing and exact equivalence to a reference sparse linear learner, and Sera, a streaming attention module based on positive random features with anytime-valid e-process risk control. Together they form a single streaming system that (i) predicts and learns in O(1) per event/token with fixed worst-case bounds, (ii) provides deterministic, scan-checkable auditability, (iii) exposes clear health signals for self-regulation under distribution drift, and (iv) remains reversible and reproducible under snapshot/restore.

1. Goals and SLOs
   Functional SLOs: Each sample produces three numbers in constant time: ŷ_hlm from Hlm, ŷ_sera from Sera, and a fused output ŷ = Gate · ŷ_sera + (1−Gate) · ŷ_hlm. Gate itself is computed in constant time from predictable, logged diagnostics.
   Latency SLOs: Fixed constant steps per event; p99 ≤ 2×p50; no dependence on sequence length t when Sera uses γ∈(0,1).
   Correctness SLOs: Hlm is exactly equivalent to the reference sparse linear learner up to IEEE-754 rounding. Sera approximates decayed softmax attention with uniform-in-time r^{−1/2} error for γ∈(0,1); its risk control is anytime-valid via Ville’s inequality.
   Availability SLOs: No reader pauses. All structure rebuilds publish with a single atomic swap and are reclaimed by epoch/hazard mechanisms. Even in pathological conditions, hot-path operations complete within configured constants.

2. Component Recap

2.1 Hlm (Hashed Linear Model, exact and reversible)
Address space: Two layers with zero collision for known keys. Base is a dense weight array addressed by an MPHF over K_base and guarded by Keys_base for exact membership. Delta is a bounded cuckoo dictionary with parameters (d choices, bucket size b, stash S, relocation cap L, overflow ring Q, single emergency slot).
Hot-path bounds: LOCATE ≤ d·b + S + 1 primitive probes; INSERT ≤ d·b + L + S + 1 + c0. With defaults d=2, b=4, S=8, L=8, Q=16, C_lookup ≤ 17 and C_insert ≤ 25 (engineering can shave constants to ~19).
Generational stability: Readers pin ver_base and ver_delta for a sample via RCU/Epoch. Rebuilds run on a snapshot and publish {h2, W2, Keys2} by one release-store; readers observe by acquire-load.
Scheduling inequality: Let C be delta capacity; τ_low<τ_high load thresholds; λ_max the upper bound on novel-key arrival rate; T_rb the rebuild+remap bound. Choose C, τ_low, τ_high so C(τ_high−τ_low) ≥ λ_max·T_rb + margin. This guarantees delta does not saturate before publication and preserves the constant-time insert bound.

2.2 Sera (Streaming e-process random-feature attention)
Positive random features: φ_i(x) = r^{−1/2} exp(w_i^T x/√τ − ||x||^2/(2τ)), w_i ~ N(0, I). Optional pairing (w,−w) reduces variance. Exponent clipping to g∈[−c,c] is logged; target clip rate ρ_clip∈[0.1%,1%].
Streaming sufficient statistics: R_t = γR_{t−1}+φ_t v_t^T and s_t=γs_{t−1}+φ_t (optionally H_t for fixed rank r_v via U^T v_t). All updates use Kahan compensation; denominators stay in extended precision until division.
Query-only whitening: φ_w(q) = diag(σ_t^2+ε)^{−1/2} φ(q); denominator den = φ_w^T s_t + λ_*(t)≥0; readout ŷ_sera = (φ_w^T R_t)/den (or U(φ_w^T H_t/den) for low-rank).
Anytime-valid risk: Build a predictable mixture e-process E_mix(t) from self-normalized coordinates. Ville’s inequality gives P(sup_t E_mix≥1/α)≤α. Maintain a logged margin m_t = log(1/α) − log E_mix(t).
RJ half-split diagnostic: Split PRF coordinates into two halves; monitor RJ_gap = ||ŷ − 0.5(ŷ^(1)+ŷ^(2))||/(||ŷ||+ε). Under weak dependence, RJ_gap = Θ(r^{−1/2}); a time-uniform threshold C_RJ√(log((t∨e)/δ)/r) governs yellow/red policy.

3. Data Plane: End-to-End Flow
   Ingest: For each event (k_t, v_t), Sera updates φ_t, R_t, s_t (and H_t if used), preconditioner (μ_t, σ_t^2), risk E_mix, and RJ diagnostics in O(1)/token with fixed r and r_v. In parallel, Hlm updates per-feature weights via LOCATEREF_OR_INSERT in constant steps; bias b updates once per sample.
   Predict: Hlm computes ŷ_hlm by a single pass over sparse features; Sera computes ŷ_sera given a query q with constant-time whitening+GEMV. The fusion layer computes Gate and ŷ. All three are logged with versions ver_sera/ver_base/ver_delta.
   Learn: Hlm applies the exact SGD(+L2) step to the unique weight reference for each touched id; Sera updates its decayed statistics independently of the fusion result, keeping risk accounting strictly predictable.

4. Control Plane: Health-Aware Self-Regulation
   Predictability discipline: Any control variable that affects the next step’s computation (λ_*(t), β_floor, Gate clamping) must be computed from F_{t−1}—i.e., based only on information logged before the step—to preserve e-process validity and RCU consistency.
   Sera→Hlm early warning: Rising RJ_gap and shrinking margin m_t anticipate instability (variance under-resolution, denominator weakness, or drift). When either crosses a threshold persistently, the system lowers τ_low or increases C on the next Hlm rebuild cycle, and temporarily reduces Hlm’s lr to de-risk updates.
   Denominator safety: If den falls below a predictable floor β_floor, Sera raises λ_*(t) by a small predictable increment; the Gate is clamped away from pure-attention in that interval. Both actions are logged and auditable.
   Gating policy: Gate = σ(w_g^T z_gate), with z_gate composed of [RJ_gap, m_t, den, ||φ_w||_1, rolling |e|, context keys]. Gate is learned by Hlm (so the gate’s parameters are themselves exact and interpretable). When m_t<0 or repeated RJ “red,” Gate is predictably stepped down for a fixed window and then allowed to recover exponentially.

5. Fusion Patterns (usable simultaneously)
   Attention-as-Feature: Turn Sera’s stabilized readouts and diagnostics into features for Hlm. Example: z_attn = [ (φ_w^T R_t)/den, (φ_w^T H_t)/den, RJ_gap, m_t, den, ||φ_w||_1 ]. Hlm then learns a linear combination with injective addressing, giving full attribution and bit-reproducible updates.
   Linear-as-Gate: Let Hlm learn Gate from Sera’s health and context. Prediction ŷ = Gate·ŷ_sera + (1−Gate)·ŷ_hlm. Under low confidence (m_t small, RJ large), Gate drops smoothly toward Hlm; under strong margins, Gate moves toward Sera.
   Meta-linear on [ŷ_sera, ŷ_hlm]: Use Hlm to regress the two-dimensional basis [ŷ_sera, ŷ_hlm] to the target. Coefficients drift with the environment and remain exactly audited thanks to Hlm’s injective mapping.

6. Sample-Scoped Algorithm (pseudocode, single thread)
   pin(ver_sera, ver_base, ver_delta)
   acc = b_hlm
   for (id,val) in x: if (p=LOCATEREF_Hlm(id)) acc += (*p)*val
   ŷ_hlm = acc
   φ_q = φ(q); φ_w = whiten(φ_q; σ_t^2)
   den = dot(φ_w, s_t) + λ_*(t)    // ≥ β_floor
   ŷ_sera = dot(φ_w, R_t) / den    // or U(dot(φ_w,H_t)/den)
   Gate = clamp( σ(w_g^T z_gate), g_min(m_t), g_max(m_t) )
   ŷ = Gate*ŷ_sera + (1−Gate)*ŷ_hlm
   e = y − ŷ
   for (id,val) in x:
   r = LOCATEREF_OR_INSERT_Hlm(id)
   *r = (1 − lr*l2)*(*r) + lr*e*val
   b_hlm += lr*e
   Sera.ingest(k_t, v_t)       // updates R_t, s_t, (H_t), μ, σ^2, E_mix, RJ
   if RJ_red or m_t < 0: Gate_policy.decrease(); β_floor↑; λ_*↑; τ_low↓
   maybe_enqueue_Hlm_rebuild()
   emit_audit_record({versions, RJ_gap, m_t, C_lookup_max, C_insert_max, ring_occ, ρ_clip, …})

7. Persistence and Recoverability
   Snapshot contents are minimal but sufficient for bit-equivalent restart:
   Hlm: MPHF blob, W_base, Keys_base, delta dump (keys+weights+location kind), b, lr, l2, d,b,S,L,Q,C, τ_low, τ_high, hash seeds, versions.
   Sera: R_t, s_t, (H_t), μ_t, σ_t^2, risk state (E_mix or per-λ accumulators), PRF seeds, γ, τ, clipping c, λ grid and λ_*.
   Atomicity: Write data blobs, then a versioned manifest with checksums; the manifest’s Merkle link ties into the global audit chain. On recovery, either the old or the new base is referenced; delta_dump ensures no lost keys. Risk state resumes with the same predictable schedule.

8. Monitoring, Audit, and One-Pass Verification
   Merkle-chained audit record per step:
   { t, ver_sera, ver_base, ver_delta, seeds: {PRF, hash}, r, r_v, τ, γ, ρ_clip, RJ_gap, margin m_t, β_floor, λ_*, den stats, Kahan residuals, Hlm: {C_lookup_p100, C_insert_p100, ring/stash occupancy, emergency_used}, rebuild timings, merkle_prev, merkle_curr }
   Verifier conditions:
   Merkle chain continuity holds; Ville margin m_t ≥ 0 for configured α; RJ policy thresholds respected; Hlm’s observed step counts never exceed configured constants; MPHF membership Keys_base[h(id)]==id for all base entries; if linear certificates are present, Farkas-style tolerances pass. The verifier runs in O(1) memory by streaming the log once.

9. Configuration and Sizing

9.1 Hlm
Defaults: d=2, b=4, S=8, L=8, Q=16, τ_low=0.6, τ_high=0.8.
Capacity: From C(τ_high−τ_low) ≥ λ_max·T_rb + margin, solve C ≥ (λ_max·T_rb + margin)/(τ_high−τ_low). Pick margin at least the p99.9 novelty burst and 2× the measured rebuild spread.
Hashing: 64-bit keyed hashing; rotate salts per rebuild; keep salts under a secret manager; never log plaintext salts.

9.2 Sera
Feature count r: For error target ε and confidence 1−δ, r ≈ C0 (r_v + log(1/δ)) / ε^2, with C0 driven by conditioning constants. Increase r under high d_v or low τ.
Temperature τ: Start near √d and tune within ×[0.7, 1.4].
Decay γ: Choose from desired effective window L_eff = 1/(1−γ). For length-free guarantees use γ∈[0.95,0.995]; for γ=1 declare a horizon T and turn on β_floor.
Regularization λ_*: Maintain a small λ grid around empirical denominator quantiles; project into (−1/b′, 1/b′) using predictable envelope estimates; average predictably to form λ_*.
Clipping c: Tune to hit ρ_clip∈[0.1%,1%]; log max exponent seen. Keep exponent arithmetic as exp(g−clamp), subtracting ||x||^2/(2τ) before exp to prevent overflow.

10. Sharding and Scale-Out
    Partition by a top-level hash of id so each shard maintains an independent Hlm and Sera state. Route queries (q, x) consistently to the same shard during a session. Publish generations independently per shard; snapshot and restore are shard-local. For multi-tenant deployments, isolate audit chains per tenant and provide tenant-side verifiers.

11. Failure Modes and Playbooks
    Sera instability: m_t falls or RJ_gap persists in red. Actions (predictable, logged): increase β_floor; nudge λ_* upward; clamp Gate down; if allowed, raise r; optionally retune τ. Expect Gate to bias toward Hlm until margins recover.
    Hlm pressure: ring occupancy grows or emergency slot used. Actions: lower τ_low to trigger earlier rebuild; increase C at next rollout; bump rebuild parallelism; re-estimate λ_max; temporarily reduce Hlm lr to calm updates.
    Numeric anomalies: Any NaN/Inf at ingress causes the sample to be quarantined and logged; denominators remain in extended precision; never quantize s or denominators; Kahan residuals are tracked and alarmed if they drift above thresholds.
    Design-violation tripwire: If Hlm ever hits the “unreachable” post-emergency path, raise a hard alarm; freeze new inserts (reads still O(1)); trigger an immediate rebuild; widen C or thresholds before resuming inserts.

12. Security and Privacy
    Keys_base prevents false-positive membership and ensures exact addressing—no ghost collisions. Salts and seeds are rotated and kept under KMS; snapshots redact sensitive seeds. Audit logs are append-only and tamper-evident; WORM storage is recommended. Optional per-tenant encryption ensures at-rest isolation.

13. Determinism and Reproducibility Switches
    A “stable” mode fixes event order, enables compensated summation globally, pins λ_* schedules, and disables any nondeterministic parallelism. Seeds for PRF and hashing are versioned. In stable mode, repeated runs over the same log produce bit-identical outputs (modulo platform-dependent FP).

14. Minimal Service API (illustrative)
    Sera.Ingest(k_t, v_t, t) → Ack{ver_sera, merkle_curr}
    Sera.Query(q, t, lowrank?:bool) → {ŷ_sera, den, RJ_gap, margin, ver_sera}
    Hlm.Update(x[], y, lr, l2, t) → Ack{ver_base, ver_delta, C_lookup_p100, C_insert_p100}
    Hlm.Predict(x[], t) → {ŷ_hlm, ver_base, ver_delta}
    Fuse.Predict(x[], q, ctx, t) → {ŷ, ŷ_sera, ŷ_hlm, Gate, diagnostics:{RJ_gap, margin, den}, versions}
    Auditor.Append(json_blob) → Ack{ok}

15. Worked Scenario (drift with novelty bursts)
    Phase 1 (steady): Sera margins high; RJ low; Gate ~0.8 so attention dominates; Hlm keeps up with sparse linear corrections. Hlm’s ring near empty; rebuilds happen infrequently at τ_low=0.6.
    Phase 2 (novelty spike): RJ red 6/10; margins thin; den dips. The controller predictably raises β_floor and λ_*; Gate clamps to ≤0.4; Hlm τ_low is lowered to 0.5 and a rebuild is enqueued early; insert remains O(1) thanks to stash/ring/emergency. p99 stays within SLO, and the Merkle audit records all actions.
    Phase 3 (stabilization): With a published base and refreshed seeds, ring drains; RJ recovers; margins rise. The controller releases the clamp gradually and returns λ_* toward zero. The fused system reverts to attention-heavy predictions while preserving the exact linear fallback.

16. Why this pairing works
    Sera provides length-free, constant-time contextualization with quantitative, time-uniform risk signals. Hlm provides an exact, interpretable, zero-collision backbone whose hot path is also strictly constant-time. The fusion layer uses Sera’s health to meter how much attention to trust moment-by-moment and uses Hlm to guarantee a safe landing when conditions degrade. Because both components are reversible and auditable, the entire stack admits post-hoc replay, forensics, and certification without sacrificing latency.

17. Practical defaults and first deployment
    Start with Sera low-rank values (r_v≈64–128) and r sized for ε≈3–5% at 95% confidence; γ≈0.98; τ≈√d; c tuned to ρ_clip≈0.5%. Start Hlm with the default cuckoo parameters and C solved from your measured λ_max and T_rb with a 2× safety factor. Deploy the fusion as Meta-linear on [ŷ_sera, ŷ_hlm]; once RJ/margins stabilize, introduce Attention-as-Feature and Linear-as-Gate to increase adaptability and interpretability.

18. Summary
    Hlm and Sera combine into a constant-time, auditable, and self-regulating streaming learner. Hlm guarantees exactness and interpretability under strict O(1) per-event bounds; Sera delivers length-free contextual attention with anytime-valid risk accounting. The control plane welds them together predictably: health signals modulate gates and rebuilds without ever violating hot-path constraints. The result is a real-time system that scales, survives drift, and can be verified and replayed end-to-end—no magic, just tight engineering.
