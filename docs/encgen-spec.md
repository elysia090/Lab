Linear Memory Unit x Streaming Attention Unit x Input Encoder Unit x Output Generator Unit: A Constant-Time, Auditable, Streaming Seq2Seq Architecture
ABSTRACT
This document specifies a fully streaming, constant-time encoder-generator (decoder) stack that integrates four production modules:

* Linear Memory Unit (LMU): a lossless hashed linear model with injective addressing and exact-equivalence sparse learning.
* Streaming Attention Unit (SAU): decayed-softmax attention via positive random features (PRF) with query-only whitening and anytime-valid e-process risk control.
* Input Encoder Unit (IEU): a reversible streaming encoder that emits SAU keys/values and LMU sparse features in O(1)/token.
* Output Generator Unit (OGU): a streaming decoder that performs self- and cross-attention through SAU, fuses LMU signals via a constant-time gate, and samples under time-uniform risk.
  All hot-path operations have strict, fixed worst-case step bounds independent of history length. All state transitions are versioned (RCU/Epoch), audit-logged (Merkle chain), and snapshot/restore-ready for bit-equivalent replay.

0. NOTATION AND SCOPE
   Sequences: input tokens x_1..x_T, output tokens y_1..y_L.
   Embeddings: emb(x), emb(y) in R^d.
   PRF count r, low-rank value dimension r_v, model value dim d_v.
   Decay gamma in (0,1] with L_eff = 1/(1-gamma).
   Temperature tau>0.
   All logs are natural. IEEE-754 double, Kahan compensation for decayed sums.
   Predictable == F_{t-1}-measurable (previsible).
   We target O(1)/event(token) with fixed constants for all hot-path functions.

1. OBJECTS
1.1 Linear Memory Unit (LMU)
   Base: MPHF h: K_base->[0..n - 1], arrays W_base[n], Keys_base[n], ver_base.
   Delta: bounded cuckoo table D(d,b,S,L,Q), stash S, overflow ring Q, emergency slot 1, W_delta aligned, ver_delta.
   Bias b, seeds (secret), thresholds tau_low,tau_high, capacity C.

1.2 Streaming Attention Unit (SAU)
Shared PRF weights W={w_i}_i=1..r, seed_prf (secret).
IEU-side state: R_enc (r x d_v) or (r x r_v), s_enc (r), mu_enc (r), sig2_enc (r), ver_sau_enc.
Dec-side state: R_dec, s_dec, mu_dec, sig2_dec, ver_sau_dec.
Risk/e-process mixtures per side: E_mix_enc(t), E_mix_dec(t); RJ diagnostics per side.

1.3 Input Encoder Unit (IEU)
Streaming tokenizer (reversible), embedding projector, K/V head (W_k,W_v or identity), featureizer for LMU.

1.4 Output Generator Unit (OGU)
Self-attention SAU over generated tokens; cross-attention SAU over encoder memory; linear gate/logit add-on via LMU; sampler tuned by margins.

Reference formulas (hot path)
   Sparse linear head (LMU): For encoder features x_enc,t = {(id_j, val_j)},
   \(\hat{y}_{\text{lmu},t} = b + \sum_j w[id_j] \cdot val_j.\)
   After observing loss residual e_t, the exact SGD+L2 update applied in-place is
   \(w[id_j] \leftarrow (1-\eta_t \lambda)\, w[id_j] + \eta_t\, e_t\, val_j\) and
   \(b \leftarrow b + \eta_t e_t.\)
   Streaming attention state (SAU): With positive random features
   \(\phi_i(x) = r^{-1/2}\exp(w_i^\top x / \sqrt{\tau} - \lVert x \rVert^2 / (2\tau))\), the encoder update per token is
   \(R_{\text{enc}} \leftarrow \gamma_{\text{enc}} R_{\text{enc}} + \phi(k_t) v_t^\top\) and
   \(s_{\text{enc}} \leftarrow \gamma_{\text{enc}} s_{\text{enc}} + \phi(k_t).\)
   Query readout uses whitening \(\phi_w(q) = \text{diag}(\sigma^2 + \varepsilon)^{-1/2} \phi(q)\) and evaluates
   \(\hat{y}_{\text{sau}}(q) = (\phi_w(q)^\top R)/(\phi_w(q)^\top s + \lambda_*(t)).\)
   Gate (learned by LMU) combines the heads via a logistic projection:
   \(G_t = \sigma(w_g^\top z_t)\),
   \(\hat{y}_t = G_t \hat{y}_{\text{sau}} + (1-G_t) \hat{y}_{\text{lmu}} + z_{\text{lin}}.\)
   Risk process: the anytime-valid mixture e-process maintained on predictable statistics satisfies
   \(E_{\text{mix}}(t) = \sum_{\lambda \in \Lambda} \pi_\lambda \exp\Big(\sum_{i \le t} \lambda g_i - \psi_i(\lambda)\Big)\)
   with logged margin \(m_t = \log(1/\alpha) - \log E_{\text{mix}}(t)\).

2. INPUT/OUTPUT CONTRACTS
   Input to IEU per token t:
   bytes or tok_id, timestamp t.
   IEU emits:
   (k_t, v_t) in R^d x R^{d_v} (or v_t in U-lowrank), LMU feature ids x_enc,t, and an audit record.

Decode step t (OGU):
Inputs: prev token y_{t - 1}, optional teacher token y**t, encoder versions, context features.
Outputs: sampled token y_t, logits digest, yhat_sau*{self,cross}, yhat_lmu_addon, Gate G_t, audit record.

3. FORMAL TARGETS
   Decayed attention (for any query q):
   A_t(q) = sum_{j<=t} gamma^{t - j} exp(q^T k_j / tau) v_j
   B_t(q) = sum_{j<=t} gamma^{t - j} exp(q^T k_j / tau)
   y_t(q) = A_t(q) / B_t(q)
   SAU approximates y_t(q) in O(1)/token with uniform-in-t r^{-1/2} error for gamma<1, with stabilized variant using query-only whitening and predictable lambda_* to keep denominators safe.

4. INVARIANTS (ALL COMPONENTS)
   I1 Injective addressing (LMU): any known id resolves to exactly one weight pointer (delta OR base), never both.
   I2 Generational stability: readers pin versions {ver_base,ver_delta,ver_sau_enc,ver_sau_dec} at step entry and never mix generations within a step.
   I3 Boundedness: each hot-path function completes in at most a fixed number of primitive operations (constants depend only on configuration params, not on U or t).
   I4 Auditability: every step emits a Merkle-chained record sufficient for deterministic replay and one-pass verification.
   I5 Predictability: control changes (lambda_*, beta_floor, gate clamps) are F_{t-1}-measurable to preserve e-process validity.

5. POSITIVE RANDOM FEATURES (PRF)
   For x in R^d:
   phi_i(x) = r^{-1/2} * exp( w_i^T x / sqrt(tau) - ||x||^2 / (2*tau) ), i=1..r
   Clip exponent g in [-c,c]; optionally pair (w, - w).
   Unbiased kernelization (no clipping):
   E_W[phi(q)^T phi(k)] = exp(q^T k / tau).
   Clipping bias budgeted by small clip rate rho_clip in [0.1%,1%] and logged.

6. SAU STATES AND UPDATES
   6.1 IEU-side SAU
   Let phi_t = phi(k_t).
   R_enc <- gamma_enc * R_enc + GER_Kahan(phi_t, v_t^T) // r x d_v (or r x r_v)
   s_enc <- gamma_enc * s_enc + KahanAdd(phi_t) // r
   Preconditioner at ingest:
   mu_enc <- (1 - beta_mu) * mu_enc + beta_mu * phi_t
   sig2_enc<- (1 - beta_sig)*sig2_enc + beta_sig * (phi_t - mu_enc)^2 + eps
   Risk/e-process (predictable):
   self-normalized stats on phi_t build E_mix_enc(t); margin_enc m_enc(t)=log(1/alpha) - log E_mix_enc(t)
   RJ_half (IEU): split coordinates; RJ_gap_enc reported.

6.2 Dec-side SAU (self-attention memory)
On each generated/teacher token with key k_t^dec and value v_t^dec:
R_dec <- gamma_dec * R_dec + GER_Kahan(phi(k_t^dec), v_t^dec^T)
s_dec <- gamma_dec * s_dec + KahanAdd(phi(k_t^dec))
Preconditioner mu_dec,sig2_dec updated predictably as in encoder.
Risk/e-process E_mix_dec(t); margins m_dec(t); RJ_gap_dec.

6.3 SAU Query (any side)
Given query q and side S in {enc,dec}:
phi_q = phi(q)
phi_w = diag(sig2_S + eps)^{-1/2} * phi_q // query-only whitening
den_S = phi_w^T s_S + lambda_*^S(t) // >= 0, predictable
yhat_S = (phi_w^T R_S) / den_S // or U( phi_w^T H_S / den_S ) if low-rank
Return yhat_S, den_S, RJ_gap_S, margin_S. Complexity O(r d) + O(r d_v) (or O(r r_v)).

7. ENCODER (IEU)
   7.1 Streaming Tokenization (reversible)

* Deterministic BPE/SP with byte fallback.
* Emit per token: {tok_id, byte_span, digests{raw_bytes,tok_seq}}.
* Merkle chaining: merkle_curr = H( step_body || merkle_prev ).

7.2 Embedding and K/V Heads

* x_t = Emb(tok_id) in R^d (learned/frozen).
* k_t = W_k * x_t ; v_t = W_v * x_t (or v_t_lowrank = U^T * x_t).
* SAU ingest IEU: update R_enc,s_enc,mu_enc,sig2_enc, E_mix_enc, RJ_gap_enc.

7.3 LMU Featureizer

* Emit sparse ids: n-grams, type/shape, doc/meta tags, encoder diagnostics (e.g., den_enc bucketed, RJ_gap_enc bucketed, margin_enc bucketed).
* Insert/update via LMU.LOCATEREF_OR_INSERT with strict O(1).

7.4 Complexity and Memory

* Per token: O(r d) + O(r d_v) (or O(r r_v)) + O(#ids)*O(1).
* State: O(r d_v)+O(r) + O(U_enc). No growth in t.

8. GENERATOR (OGU)
   8.1 Decoder Query Construction

* q_t = W_q * emb(y_{t - 1}) + S_tiny // small recurrent state if desired (kept O(1)/token)
* Optional conditioning on shallow context (e.g., time, speaker).

8.2 Self- and Cross-Attention via SAU
Self:
(yhat_self, den_self, RJ_self, m_self) = SAU.Query(q_t; dec)
Cross (encoder memory):
(yhat_cross, den_cross, RJ_cross, m_cross) = SAU.Query(q_t; enc)
Both use the SAME PRF basis W for tight cache and shared whitening stats per side.

8.3 LMU Fusion (gate + linear add-on)
Diagnostics vector z_t:
z_t = [RJ_self, RJ_cross, m_self, m_cross, den_self, den_cross, ||phi_w||_1, rolling |e|, context bins...] // all predictably computed
Gate (exact, interpretable):
G_t = sigma( w_g^T z_t ) // w_g learned by LMU on sparse features derived from z_t
Context composition:
c_t = G_t * yhat_self + (1 - G_t) * yhat_cross // or meta-linear head learned by LMU on [yhat_self,yhat_cross]
Logit linear add-on (exact sparse):
z_lin = LMU.Linear(x_dec_t) // optional per-token features (topic, persona, plan)

8.4 Logits and Risk-Aware Sampling
logits_t = W_out * c_t + b_out + z_lin
margin_t = m_dec(t) // from SAU decoder e-process
temperature_t = f_temp(margin_t) // monotone nonincreasing in risk
(top_k, top_p) = f_filter(margin_t, RJ_self, RJ_cross)
token_t = Sample(logits_t, temperature_t, top_k, top_p)
Predictable safety actions when margin_t<0 or RJ "red":

* Clamp G_t <= G_max(margin_t) (favor cross or linear).
* Increase beta_floor and lambda_*^S predictably for next step.
* Reduce temperature and top_p bounds.

8.5 Online Learning (optional, supervised)
e_t = loss_grad(y_t*, logits_t) or e_t = y_t* - yhat_t (for regression heads).
LMU updates exact per-id weights touched at step t.
SAU_dec ingest with k_t^dec, v_t^dec from the realized token embedding.

8.6 Complexity
Per token: 2 x SAU query + small projections + LMU gate/linear; overall O(r d) + O(r d_v) (or O(r r_v)) + O(#ids)*O(1).

9. LMU DETAILS (RECAP, IMPLEMENTATION-GRADE)
   9.1 Bounds and Defaults
   d=2, b=4, S=8, L=8, Q=16
   C_lookup <= 2*4 + 8 + 1 = 17
   C_insert <= 2*4 + 8 + 8 + 1 + c0 <= 25 + c0 (often ~19)
   9.2 Operations
   LOCATEREF(id):
   scan d*b delta slots, then S stash, then MPHF base membership via Keys_base[h(id)]; return &W if found; else NULL.
   LOCATEREF_OR_INSERT(id):
   if LOCATEREF != NULL return it
   if vacancy in candidate buckets: place, return
   else do <=L relocations (bounded), else stash if space, else ring enqueue (stable shadow), else emergency slot CAS; return pointer.
   9.3 Rebuild Scheduling (safety inequality)
   slack = C*(tau_high - tau_low)
   require slack >= lambda_max * T_rb + margin
   start rebuild at tau_low; guarantee swap before tau_high with probability 1 under the bound; maintain O(1) inserts indefinitely.
   9.4 Concurrency and Publication
   Build {h2,W2,Keys2} off-thread; publish pointer/version by single release-store; readers acquire-load at step entry; free old gen after epoch quiescence.

10. CONTROL PLANE (PREDICTABLE GOVERNORS)
    10.1 SAU -> LMU Early-Warning
    Persistent RJ "red" or thinning margins -> lower tau_low (earlier rebuild), increase C at next rollout, reduce LMU learning rate temporarily.
    10.2 Denominator Safety
    If den_S < beta_floor: raise lambda_*^S predictably and clamp Gate down.
    10.3 Gate Policy
    Gate decreases stepwise when margin_t<0; recovers exponentially when margins positive over a window. All thresholds logged and predeclared.
    10.4 Learning-Rate Governance
    LMU lr_scaled = lr_base * g(margin_t) with monotone g; ensures stable exact updates during turbulence.

11. SNAPSHOT / RESTORE
    11.1 Snapshot Payload
    LMU: MPHF_blob, W_base, Keys_base, delta_dump(keys+weights+location), b, lr,l2, d,b,S,L,Q,C,tau_low,tau_high, seeds, versions.
    SAU IEU/Dec: R,s,(H), mu, sig2, risk accumulators (per-lambda logs or E_mix), r, r_v, gamma, tau, clip c, lambda grid, lambda_* schedules, PRF seed.
    Shared: audit rolling hash, last merkle_curr.
    11.2 Atomicity
    Two-phase write: data blobs then manifest with checksums; manifest links to Merkle chain. Crash leaves either old or new base referenced; delta_dump prevents loss.
    11.3 Restore
    Map arrays, rebuild handles, re-establish versions, resume predictable schedules; no reader pause on first request.

12. AUDIT AND ONE-PASS VERIFICATION
    12.1 Per-Step Record (JSON concept, kept ASCII)
    {
    t, role:{IEU|OGU}, ver_sau_enc, ver_sau_dec, ver_base, ver_delta,
    PRF_seed_hash, hash_salt_hash, r, r_v, tau, gamma_enc, gamma_dec,
    rho_clip_enc, rho_clip_dec,
    RJ_enc, RJ_self, RJ_cross, margin_enc, margin_dec,
    beta_floor, lambda_star_self, lambda_star_cross,
    den_self, den_cross,
    Gate, z_lin_digest,
    LMU:{C_lookup_p100, C_insert_p100, ring_occ, stash_occ, emergency_used},
    rebuild_time_ms, slack_minus_arrivals,
    kahan_residuals:{R,s,(H)}, saturation_events,
    merkle_prev, merkle_curr
    }
    12.2 Verifier (one pass, O(1) memory)

* Check merkle_prev -> curr continuity.
* Validate Ville margins: log(1/alpha) - log E_mix_S(t) >= 0 for S in {enc,dec}.
* Check RJ policy thresholds and action logs.
* Confirm LMU step counts <= configured constants; check Keys_base[h(id)] == id on sampled entries (or full scan offline).
* Optional: verify linear Farkas certificates if present.

13. PSEUDOCODE (ASCII)

13.1 IEU Ingest
EncIngest(tok):
x = Emb(tok)
k = W_k * x
v = W_v * x // or v_low = U^T * x
phi_k = PRF(k) // clipped, paired optional
R_enc = gamma_enc * R_enc + GER_Kahan(phi_k, v^T)
s_enc = gamma_enc * s_enc + KahanAdd(phi_k)
mu_enc, sig2_enc = EWM2_update(phi_k)
update_e_process_enc(phi_k) // predictable envelope
RJ_enc = RJ_half_split_enc()
emit_LMU_features_from_encoder(tok, diagnostics_from{den_enc, RJ_enc, margin_enc})
AuditEmit(...)

13.2 OGU Decode Step
GenDecodeStep(prev_token, ctx):
pin_versions(ver_sau_enc, ver_sau_dec, ver_base, ver_delta)
q = QProj(Emb(prev_token), ctx_state)
// SAU self
phi_q = PRF(q); phi_w_dec = whiten(phi_q; sig2_dec)
den_self = dot(phi_w_dec, s_dec) + lambda_star_self
yhat_self = dot(phi_w_dec, R_dec) / den_self
RJ_self, margin_dec = diagnostics_dec()
// SAU cross
phi_w_enc = whiten(phi_q; sig2_enc)
den_cross = dot(phi_w_enc, s_enc) + lambda_star_cross
yhat_cross = dot(phi_w_enc, R_enc) / den_cross
RJ_cross, margin_enc = diagnostics_enc() // from pinned record
// LMU fusion
z_gate = featurize(RJ_self, RJ_cross, margin_dec, margin_enc, den_self, den_cross, norm1(phi_w_dec), ctx)
G = clamp( sigmoid( LMU.Linear(z_gate_ids) ), g_min(margin_dec), g_max(margin_dec) )
z_lin = LMU.Linear(x_dec_ids) // exact sparse contribution
c = G * yhat_self + (1 - G) * yhat_cross
logits = W_out * c + b_out + z_lin
(temp, topk, topp) = risk_to_sampling(margin_dec, RJ_self, RJ_cross)
y = Sample(logits, temp, topk, topp)
AuditEmit(...)
return y

13.3 OGU Learn Step (optional supervised)
GenLearnStep(x_dec_ids, y_true, logits):
e = grad_loss(logits, y_true)
for id,val in x_dec_ids:
p = LOCATEREF_OR_INSERT(id)
*p = (1 - lr*l2) * (*p) + lr * e * val
b_lmu += lr * e

13.4 SAU Decoder Ingest (after emitting y)
SauDecIngest(y):
x = Emb(y)
k = W_k_dec * x
v = W_v_dec * x
phi_k = PRF(k)
R_dec = gamma_dec * R_dec + GER_Kahan(phi_k, v^T)
s_dec = gamma_dec * s_dec + KahanAdd(phi_k)
mu_dec, sig2_dec = EWM2_update(phi_k)
update_e_process_dec(phi_k)

14. COMPLEXITY SUMMARY (PER TOKEN/EVENT)
    IEU ingest: O(r d) PRF + O(r d_v) GER (+O(r r_v) if low-rank) + O(r) stats + O(#ids)*O(1) LMU updates.
    OGU step: 2 x (O(r d) + O(r d_v)) + O(#ids)*O(1) + tiny projections.
    LMU ops: LOCATE <= 17 primitive probes; INSERT <= 25+c0; independent of U,F,t.

15. MEMORY LAYOUT AND CACHE
    PRF:

* W (r x d) row-major; exp arguments computed as (w_i^T x)/sqrt(tau) - ||x||^2/(2*tau).
* Avoid per-call salt fetch; keep seed in TLS.
  SAU:
* R_* (r x d_v) row-major; s_* (r) contiguous; mu/sig2 contiguous.
* Use AOSoA if SIMD helps.
  LMU:
* Buckets aligned to cache lines; (key,fingerprint,widx) packed; W_delta parallel array of float.
* Keys_base and W_base parallel arrays for stride-1 access.

16. DEFAULTS AND SIZING
    SAU:
    r ~ C0 * (r_v + log(1/delta)) / epsilon^2 for target error epsilon and confidence 1 - delta.
    r_v in [64,128], tau ~ sqrt(d), gamma_enc in [0.97,0.99], gamma_dec in [0.98,0.997].
    clip c tuned for rho_clip ~ 0.5%; beta_floor in [1e - 6, 1e - 4] when gamma -> 1.
    LMU:
    d=2, b=4, S=8, L=8, Q=16; tau_low=0.6, tau_high=0.8.
    C from lambda_max and T_rb with >= 2 x safety.

17. FAILURE MODES AND RUNBOOK
    SAU margins low or RJ red:
    action: beta_floor up , lambda_*^S up predictably, Gate clamp down , reduce temperature/top-p, optionally r up ; log all.
    LMU ring pressure or emergency used:
    action: tau_low down (early rebuild), increase C at next deployment, rebuild parallelism up , lr down temporarily, confirm slack >= arrivals.
    Denominator instability:
    action: enforce beta_floor>0, extended precision accumulation, never quantize s or denominators.
    NaN/Inf:
    action: quarantine sample, record in audit, fall back to linear path (G -> 0) until healthy.

18. SECURITY AND PRIVACY

* PRF seeds and hash salts under KMS; rotate per rebuild; never log plaintext.
* Audit logs are append-only (WORM); per-tenant encryption and chain separation.
* Keys_base ensures no false-positive membership.

19. DETERMINISM MODE

* Fix event order; enable Kahan everywhere; pin lambda_* schedule; disable nondeterministic parallelism.
* Version seeds; bit-identical replay of audit stream should reproduce outputs modulo platform FP.

20. API SKETCH (ASCII)
    IEU.Ingest({bytes|tok_id,t}) -> Ack{ver_sau_enc, merkle_curr}
    IEU.ExportSummary() -> {digest(R_enc), digest(s_enc), ver_sau_enc, stats}
    OGU.DecodeStep({prev_tok,ctx,t}) -> {tok, logits_digest, yhat_self, yhat_cross, G, diagnostics, versions}
    OGU.LearnStep({x_dec_ids, y_true, lr, l2, t}) -> Ack{ver_base, ver_delta}
    LMU.Predict({x_ids,t}) -> {yhat_lmu, ver_base, ver_delta}
    LMU.Update({x_ids,y,lr,l2,t}) -> Ack{C_lookup_p100, C_insert_p100, ring_occ, emergency_used}
    Auditor.Append({json}) -> Ack{ok}

21. END-TO-END SCENARIO (DRIFT + NOVELTY BURSTS)
    Phase A steady:
    margins high, RJ low, Gate ~0.8, LMU ring ~ 0, rebuilds rare.
    Phase B burst:
    RJ red 6/10, margins thin, den dips; controller predictably raises beta_floor, lambda_*; clamps Gate<=0.4; LMU lowers tau_low and enqueues early rebuild; inserts remain O(1); p99 stable.
    Phase C recover:
    base published; seeds rotated; ring drains; margins recover; clamps relax; Gate returns to attention-heavy.

22. CHECKLIST (IMPLEMENTATION BRING-UP)
    [ ] LMU: enforce I1 - I4; log bounds; enable rebuild inequality.
    [ ] SAU: PRF clipping logged; whitening at query only; denominators in extended precision; RJ/e-process connected to policy.
    [ ] IEU: reversible tokenizer; K/V heads; LMU featureizer; audit stream on.
    [ ] OGU: two SAU queries/token; gate via LMU; risk-aware sampler; teacher forcing path.
    [ ] Audit: Merkle chain; one-pass verifier exercised; snapshot/restore rehearsal.

23. SUMMARY
    By co-designing IEU and OGU around SAU's PRF-based, length-free attention and LMU's injective, exact linearity, the stack achieves: strict O(1)/token/event hot paths; deterministic audit and replay; quantitative health signals that steer gates, denominators, and rebuilds predictably; and interpretable contributions at every stage. It scales horizontally by sharding ids, remains reversible under crash/restore, and sustains real-time SLOs under drift without sacrificing correctness.
