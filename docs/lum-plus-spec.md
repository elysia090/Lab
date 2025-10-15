Hlm x Sera x Lum x Enc x Gen x Cluster/Multimodal
A Constant-Time, Auditable, Streaming Architecture (Plain ASCII, no separators)

ABSTRACT
This document specifies a production-grade, end-to-end architecture that delivers constant-time O(1)/token (or O(1)/event) operation for learning, attention, exact finite nonlinearity, and infinite-horizon rational memory, with full auditability and deterministic replay. The stack fuses: Hlm (lossless hashed linear model with injective addressing and exact sparse learning), Sera (streaming decayed-softmax attention via positive random features with anytime-valid e-process risk control), Lum (exact finite algebraic lift plus fixed-order rational memory via SOS/ARMA, fully O(1)), Enc/Gen (streaming encoder/decoder), and Cluster/Multimodal (session-sticky sharding with bounded fan-in cross-modal reads). All hot-path loops iterate over configuration-time constants only; no runtime hashing, probing, or variable-length search occurs. Every step emits a Merkle-chained audit record.

0. NOTATION AND CONTRACTS
   Time index t; per step we process at most a fixed number of events/tokens. Inputs: sparse features x_t for Hlm ( <= F_max ids per sample), modality embeddings x_t^m for Sera/Lum (m in {txt,aud,img,vid,tab,sens}), and a decoder query q_t. Outputs: prediction/logits y_hat_t in a fixed, small dimension d_y. Predictable means F_{t - 1}-measurable; any control that affects step t is computed using logs up to t - 1. Arithmetic is IEEE-754 double; decayed sums use Kahan/Neumaier compensation; denominators are kept in extended precision until readout.

1. COMPONENTS (SINGLE-NODE VIEW)
   Hlm (Hashed linear, exact and injective). Base: MPHF h: K_base -> [0..n - 1], arrays W_base[n], Keys_base[n]. Delta: bounded cuckoo (d choices, bucket size b), stash S, overflow ring Q, single emergency slot, W_delta. Bounds with defaults: C_lookup <= d * b + S + 1 (e.g., <= 17), C_insert <= d * b + L + S + 1 + c0 (e.g., <= 25). Rebuild safety: choose capacity C and thresholds tau_low<tau_high to satisfy C(tau_high - tau_low) >= lambda_max * T_rb + margin. Publication is a snapshot rebuild followed by a single release-store swap; readers pin versions via RCU/Epoch and never pause.
   Sera (Streaming attention with PRF and e-process). PRF basis phi_i(x)=r^{-1/2} exp(w_i^T x/ sqrt tau - ||x||^2/(2tau)) with clipping and optional (w, - w) pairing. Sufficient statistics per side (enc/dec): R_t=gamma R_{t - 1}+phi_t v_t^T and s_t=gamma s_{t - 1}+phi_t. Query-only whitening: phi_w(q)=diag(sigma^2+epsilon)^{-1/2}phi(q), readout (phi_w^T R)/(phi_w^T s+lambda_*). Risk: predictable mixture e-process E_mix(t), margin m_t=log(1/alpha) - log E_mix(t); RJ half-split monitors variance. For gamma in (0,1): length-free O(r^{-1/2}) uniform-in-time approximation and anytime-valid Type-I control.
   Lum (Exact finite nonlinearity + rational memory, O(1)). Finite algebraic lift with pre-enumerated coordinate ids (no runtime insertion) and fixed-order rational memory via L second-order sections (SOS) or ARMA(p,q). A small fixed head combines u_t and selected states. All loops are over constants T_phi, K, L, p, q, so per-token work/state are constant.
   Enc (Streaming encoders, per modality). Reversible tokenization/featureization -> embeddings -> Sera ingest and optional Lum step -> sparse Hlm featureization. Per-event cost O(r_m d_m)+O(r_m d_v,m) (+O(r_m r_v,m) if low-rank)+O(#ids) * O(1).
   Gen (Streaming decoder). Builds q_t, performs Sera self-attention and up to K_mod cross-modal Sera reads, optional Lum self/cross processing, Hlm gate and linear add-on, then risk-aware sampling.

   Reference hot-path formulas.
    - Hlm prediction/update: for sparse features x_t={(id_j,val_j)}, compute
     \(\hat{y}_{\text{hlm},t} = b + \sum_j w[id_j] val_j\) and update each weight by
     \(w[id_j] \leftarrow (1-\eta_t\lambda) w[id_j] + \eta_t e_t val_j\) with residual \(e_t = y_t - \hat{y}_t\), while \(b \leftarrow b + \eta_t e_t\).
    - Sera ingest/query: using \(\phi_i(x) = r^{-1/2}\exp(w_i^\top x / \sqrt{\tau} - \lVert x \rVert^2/(2\tau))\), maintain
     \(R_t = \gamma R_{t-1} + \phi(k_t) v_t^\top\) and \(s_t = \gamma s_{t-1} + \phi(k_t)\); query with \(\phi_w(q)=\text{diag}(\sigma^2+\varepsilon)^{-1/2}\phi(q)\) gives
     \(\hat{y}_{\text{sera}}(q) = (\phi_w(q)^\top R_t)/(\phi_w(q)^\top s_t + \lambda_*(t))\).
    - Lum rational section (representative SOS): for state tuple (a_t,b_t) and input u_t,
     \(a_t = u_t - \sum_{i=1}^{p} \alpha_i a_{t-i}\),
     \(b_t = \beta_0 a_t + \sum_{j=1}^{q} \beta_j b_{t-j}\),
     with all indices bounded by fixed p,q so runtime remains O(1). Aggregated Lum output is \(y_{\text{lum},t} = h^\top[b_t, \text{selected states}]\).
    - Fusion: the simplex gate computed by Hlm obeys
     \([\alpha_t,\beta_t,\gamma_t] = \text{softmax}(W_g z_t)\) and the combined prediction satisfies
     \(\hat{y}_t = \alpha_t y_{\text{sera}} + \beta_t y_{\text{lum}} + \gamma_t z_{\text{lin}}\) with \(\alpha_t+\beta_t+\gamma_t=1\).

2. FUSION LOGIC (CONSTANT-TIME)
   Parallel simplex Mixture-of-Experts (default). Define y_sera = y_self^sera + sum_{m in M_used} w_m y_m^sera, y_lum = y_self^lum + sum_{m in M_used} v_m y_m^lum (optional), and z_lin the exact sparse linear add-on from Hlm. Hlm learns a simplex gate G=[alpha,beta,gamma] over a fixed diagnostic vector z_gate, producing y_hat = alpha * y_sera + beta * y_lum + gamma * z_lin with alpha,beta,gamma >= 0 and alpha+beta+gamma=1. Alternatives remain O(1): meta-linear over [y_self^sera, y_cross^sera, y_self^lum, y_cross^lum, z_lin]; cascades Lum -> Sera (use Lum as value head) or Sera -> Lum (feed y_sera into the small Lum head). All dimensions are fixed and small.

3. DATA PLANE (TOKEN/EVENT STEPS)
   Encoder step per modality m. Tokenize/featureize to x_t^m; call Sera_m.ingest to update (R_m, s_m, mu_m, sigma^2_m), risk stats, RJ_m with constant work; optionally call Lum_m.Step(x_t^m) to obtain y_lum^m and diagnostics (denom_min, fired_count); emit sparse Hlm features (n-grams/meta/diagnostic buckets) and update Hlm with exact O(1) inserts.
   Decoder step. Build q_t from Emb(prev_token) and a tiny state; Sera self readout y_hat_self^sera=(phi_w^T R_dec)/(phi_w^T s_dec+lambda_*^self) with RJ_self and margin_sera; Sera cross readouts for a fixed K_mod modalities; Lum self y_self^lum = Lum_dec.Step(Emb(prev_token)); optional Lum cross y_lum^m from encoder shards; Hlm computes gate G=[alpha,beta,gamma] and z_lin from fixed features; fuse to logits and sample with temperature/top-k/p chosen predictably from margins/RJ; perform online updates (Sera_dec.ingest on realized token; Hlm exact updates if supervised); emit a fixed-size audit record; enqueue any predictable control actions for t+1.

4. CONTROL PLANE (PREDICTABLE AND AUDITABLE)
   Health signals, fixed dimension. From Sera: margin_sera, RJ_self, {RJ_m}, den_self, {den_m}. From Lum: margin_lum (predictable e-process over fixed coordinates), denom_min, fired_count, Kahan residual flags. From Hlm: ring/stash occupancy, C_lookup/C_insert p100, emergency usage.
   Predictable actions (F_{t - 1}-measurable). Gating: clamp alpha under low margins or persistent RJ "red" and redistribute mass to beta/gamma. Denominators: raise lambda_* and beta_floor predictably when den falls below floor. Sampling: reduce temperature/top-p under low margins or RJ alarms. Hlm rebuild: when early-warning triggers, lower tau_low and start rebuild so C(tau_high - tau_low) >= lambda_max * T_rb + margin remains true. Learning rate: scale Hlm lr down while margins are low, recover gradually.
   Invariants. Hot-path O(1) never changes due to control; publication is a single atomic swap; readers pin generations at step entry.

5. CLUSTER AND MULTIMODAL (BOUNDED FAN-IN/OUT)
   Sharding and placement. Session-sticky routing selects a home shard S_h for the decoder; each modality m runs Enc_m + Sera_m (+ Lum_m) on its shard, co-locating GPU-heavy modalities where possible; the decoder issues <= K_mod parallel RPCs per token to fetch cross-modal Sera/Lum readouts (bounded, fixed K_mod).
   Backpressure and shedding. If a remote shard lags, reuse the last y_m for <= B steps (fixed); after B, drop modality m from M_used until healthy; the gate reweights automatically; O(1) is preserved.
   Audit tree. Each shard emits a Merkle leaf per step; a root combiner periodically publishes merkle_root = H(sorted(leaves) || prev_root); a one-pass verifier checks leaf chains and the root.

6. STATE, PERSISTENCE, DETERMINISM
   Snapshots (bit-reproducible on the same HW/build). Hlm: MPHF blob, W_base, Keys_base, delta dump (keys+weights+location), bias, lr,l2, parameters (d,b,S,L,Q,C, tau_low, tau_high), seeds, versions. Sera (enc/dec): R,s,(H), mu,sigma^2, risk accumulators or E_mix state, r,r_v,gamma,tau, clipping c, lambda grid and lambda_* schedule, PRF seed. Lum: lift accumulators M (+ compensation), DelayBuf head, SOS/ARMA states and coeffs, head weights, numeric flags, prev_hash. A versioned manifest with checksums and a Merkle link closes the snapshot; restore maps blobs and resumes with pinned generations.
   Determinism mode. Fix event order; enable compensation everywhere; pin lambda_* schedules; disable nondeterministic parallelism; version seeds/salts. Replaying the same log yields bit-identical outputs modulo platform FP rules.

7. COMPLEXITY AND SIZING (ALL CONSTANTS)
   Per event/token. Enc_m: O(r_m d_m)+O(r_m d_v,m) (+O(r_m r_v,m)) + O(#ids) * O(1). Gen: one Sera self + <= K_mod Sera cross + optional Lum self/cross + Hlm gate/linear. Hlm ops: LOCATE <= ~17 probes; INSERT <= ~25 primitive ops with defaults.
   Parameter rules of thumb. Sera features r ~ C0 * (r_v + log(1/delta))/epsilon^2 for target error epsilon and confidence 1 - delta; tau ~ sqrt d; gamma_dec ~ 0.98 - 0.997; clipping tuned to rho_clip ~ 0.5%. Lum: choose T_phi by bounding emitted coordinates at given sparsity and degree; L ~ 8 - 16 (SOS) or p=q ~ 4 - 8 (ARMA). Hlm capacity: solve C from slack >= lambda_max * T_rb + margin (use p99.9 novelty bursts for the margin); start with tau_low=0.6, tau_high=0.8.

8. MONITORING AND AUDIT (FIXED-SIZE RECORDS)
   Per-step JSON, fixed fields: {t, shard_id, role:{Enc_m|Gen}, versions:{ver_sera_m_enc, ver_sera_dec, ver_lum_m, ver_base, ver_delta}, PRF_seed_hash_m, hash_salt_hash, Sera:{r_m,r_v_m,tau_m,gamma_m,rho_clip_m,RJ_m,margin_m,den_stats_m,beta_floor_m,lambda_*_m}, Lum:{T_phi,K,L|p,q,denom_min,fired_count,kahan_resid_max,margin_lum}, Hlm:{C_lookup_p100,C_insert_p100,ring,stash,emergency}, z_gate_digest, Gate:{alpha,beta,gamma}, z_lin_digest, rebuild_time_ms, slack_minus_arrivals, merkle_prev, merkle_curr}.
   Verifier (one pass). Check Merkle continuity, Ville margins >= 0 and RJ policy adherence, Hlm step counts <= configured bounds with MPHF membership spot checks (Keys_base[h(id)]=id), and Lum fired_count <= T_phi with denominator floors applied where required.

9. FAILURE MODES AND DEGRADATION (HOT PATH UNCHANGED)
   Low margins or persistent RJ "red" (Sera or Lum). Clamp alpha, raise beta and/or gamma; raise beta_floor and nudge lambda_*; reduce temperature/top-p; optionally increase r on a planned rollout.
   Hlm pressure (ring rising or emergency used). Lower tau_low to trigger an early rebuild; increase C next rollout; temporarily reduce Hlm lr; hard-alarm on emergency and freeze new inserts until swap (reads continue).
   Remote modality lag. Cache <= B steps then drop the modality from M_used; gate reweights toward self/text/Hlm.
   Numeric anomalies (NaN/Inf). Quarantine the step, log, fall back to linear; denominators remain extended-precision; never quantize s or denominators.

10. MINIMAL SERVICE API (PLAIN TEXT)
    Enc_m.Ingest(event) -> Ack{ver_sera_m, ver_lum_m, merkle_curr}
    Enc_m.Summary() -> {digests(R_m,s_m[,Lum]), versions}
    Gen.DecodeStep(prev_tok, ctx) -> {tok, logits_digest, y_self^sera, {y_m^sera}, y_self^lum?, {y_m^lum?}, Gate{alpha,beta,gamma}, diagnostics, versions}
    Gen.LearnStep(x_dec_ids, y, lr, l2) -> Ack{ver_base, ver_delta}
    Hlm.Predict/Update(...) -> exact sparse operations with bounded probes
    Lum.Step/Query/Snapshot/Restore/Diagnostics(...) -> fixed fields, O(1)
    Auditor.Append(json) -> Ack{ok}
    AuditRoot.Collect(leaves) -> {merkle_root, prev_root}

11. BRING-UP CHECKLIST
    Hlm: O(1) bounds observed; scheduling inequality holds; rebuild path tested.
    Sera: PRF clipping logged; whitening at query only; e-process margins monotone-valid; RJ policy wired to predictable actions.
    Lum: T_phi/L/(p,q) fixed; denominator floors in place; compensation enabled; diagnostics exported.
    Enc: reversible tokenization; fixed-size Hlm features; per-modality ingest O(1).
    Gen: self + <= K_mod cross Sera; optional Lum self/cross; simplex gate via Hlm; risk-aware sampler.
    Cluster: session-sticky routing; <= K_mod RPCs/token; cache-then-shed backpressure; audit root aggregator.
    Snapshot/restore drill; determinism-mode replay equals bitwise output on the same HW/build.

12. DECODER TOKEN PSEUDOCODE (END-TO-END, CONSTANT-TIME)
    pin(ver_sera_dec, {ver_sera_enc[m]}, ver_lum_dec, {ver_lum_enc[m]}, ver_base, ver_delta)
    q = QProj(Emb(prev_tok), ctx_state)
    phi = PRF(q); phi_w = whiten(phi; sigma^2_dec)
    den_self = dot(phi_w, s_dec) + lambda_*^self
    yS_self = dot(phi_w, R_dec) / den_self
    RJ_self, margin_sera = diag_dec()
    yL_self = Lum_dec.Step(Emb(prev_tok))
    for m in M_used:
    q_m = CrossProj_m(q, ctx); phi_m = PRF_m(q_m); phi_wm = whiten_m(phi_m; sigma^2_enc_m)
    den_m = dot(phi_wm, s_enc_m) + lambda_*^m
    yS_m[m] = dot(phi_wm, R_enc_m) / den_m
    RJ_m[m], margin_m[m] = diag_enc_m()
    for m in M_used:
    yL_m[m] = Lum_enc_m.Query_or_Tap() # optional
    z_gate = [RJ_self, {RJ_m[m]}, margin_sera, min_m margin_m[m], margin_lum, den_self, {den_m}, denom_min_lum, fired_count_lum, ||phi_w||_1, ctx_bins]
    [alpha,beta,gamma] = Hlm.SimplexGate(z_gate)
    z_lin = Hlm.Linear(x_dec_ids)
    yS_cross = Sigma_m w_m yS_m[m]
    yL_cross = Sigma_m v_m yL_m[m] # optional
    logits = alpha*(yS_self + yS_cross) + beta*(yL_self + yL_cross) + gamma*z_lin
    (temp, topk, topp) = RiskToSampling(min(margin_sera, margin_lum, min_m margin_m[m]), RJ_self, {RJ_m})
    tok = Sample(logits, temp, topk, topp)
    Sera_dec.ingest(Emb(tok))
    if need_controls: plan{lambda_* up , beta_floor up , clamp alpha, early Hlm rebuild} # applies from t+1
    AuditEmit(...) ; return tok

13. WHY THIS IS A FOUNDATIONAL ARCHITECTURE
    Constant-time everywhere: every subsystem's hot path is bounded by configuration-time constants; history length never appears in runtime costs. Safety-first control: anytime-valid margins and RJ govern gates, denominators, and sampling predictably without violating martingale or RCU invariants. Exactness and interpretability: Hlm matches the reference sparse linear learner up to rounding; Lum is exact within its finite-lift + fixed-order memory family; Sera provides length-free contextualization with explicit risk signals. Auditability and replay: per-step Merkle logs, a one-pass verifier, and deterministic snapshots yield forensic-grade traceability. Multimodal at bounded fan-in: the decoder consults a fixed small number of remote memories per token, so cluster cost remains O(1). The result is a low-latency, high-trust, multimodal foundation that scales without sacrificing correctness, interpretability, or reproducibility.
