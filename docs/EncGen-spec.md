Hlm × Sera × Enc × Gen : A Constant-Time, Auditable, Streaming Seq2Seq Architecture
ABSTRACT
This document specifies a fully streaming, constant-time encoder–generator (decoder) stack that integrates:

* Hlm: a lossless hashed linear model with injective addressing and exact-equivalence sparse learning.
* Sera: decayed-softmax attention via positive random features (PRF) with query-only whitening and anytime-valid e-process risk control.
* Enc: a reversible streaming encoder that emits Sera keys/values and Hlm sparse features in O(1)/token.
* Gen: a streaming decoder that performs self- and cross-attention through Sera, fuses Hlm signals via a constant-time gate, and samples under time-uniform risk.
  All hot-path operations have strict, fixed worst-case step bounds independent of history length. All state transitions are versioned (RCU/Epoch), audit-logged (Merkle chain), and snapshot/restore-ready for bit-equivalent replay.

0. NOTATION AND SCOPE
   Sequences: input tokens x_1..x_T, output tokens y_1..y_L.
   Embeddings: emb(x), emb(y) in R^d.
   PRF count r, low-rank value dimension r_v, model value dim d_v.
   Decay gamma in (0,1] with L_eff = 1/(1−gamma).
   Temperature tau>0.
   All logs are natural. IEEE-754 double, Kahan compensation for decayed sums.
   Predictable == F_{t−1}-measurable (previsible).
   We target O(1)/event(token) with fixed constants for all hot-path functions.

1. OBJECTS
   1.1 Hlm (Hashed Linear Model)
   Base: MPHF h: K_base->[0..n−1], arrays W_base[n], Keys_base[n], ver_base.
   Delta: bounded cuckoo table D(d,b,S,L,Q), stash S, overflow ring Q, emergency slot 1, W_delta aligned, ver_delta.
   Bias b, seeds (secret), thresholds tau_low,tau_high, capacity C.

1.2 Sera (Streaming E-process Random-Feature Attention)
Shared PRF weights W={w_i}_i=1..r, seed_prf (secret).
Enc-side state: R_enc (r x d_v) or (r x r_v), s_enc (r), mu_enc (r), sig2_enc (r), ver_sera_enc.
Dec-side state: R_dec, s_dec, mu_dec, sig2_dec, ver_sera_dec.
Risk/e-process mixtures per side: E_mix_enc(t), E_mix_dec(t); RJ diagnostics per side.

1.3 Enc (Encoder)
Streaming tokenizer (reversible), embedding projector, K/V head (W_k,W_v or identity), featureizer for Hlm.

1.4 Gen (Generator/Decoder)
Self-attention Sera over generated tokens; cross-attention Sera over encoder memory; linear gate/logit add-on via Hlm; sampler tuned by margins.

2. INPUT/OUTPUT CONTRACTS
   Input to Enc per token t:
   bytes or tok_id, timestamp t.
   Enc emits:
   (k_t, v_t) in R^d x R^{d_v} (or v_t in U-lowrank), Hlm feature ids x_enc,t, and an audit record.

Decode step t (Gen):
Inputs: prev token y_{t−1}, optional teacher token y**t, encoder versions, context features.
Outputs: sampled token y_t, logits digest, yhat_sera*{self,cross}, yhat_hlm_addon, Gate G_t, audit record.

3. FORMAL TARGETS
   Decayed attention (for any query q):
   A_t(q) = sum_{j<=t} gamma^{t−j} exp(q^T k_j / tau) v_j
   B_t(q) = sum_{j<=t} gamma^{t−j} exp(q^T k_j / tau)
   y_t(q) = A_t(q) / B_t(q)
   Sera approximates y_t(q) in O(1)/token with uniform-in-t r^{−1/2} error for gamma<1, with stabilized variant using query-only whitening and predictable lambda_* to keep denominators safe.

4. INVARIANTS (ALL COMPONENTS)
   I1 Injective addressing (Hlm): any known id resolves to exactly one weight pointer (delta OR base), never both.
   I2 Generational stability: readers pin versions {ver_base,ver_delta,ver_sera_enc,ver_sera_dec} at step entry and never mix generations within a step.
   I3 Boundedness: each hot-path function completes in at most a fixed number of primitive operations (constants depend only on configuration params, not on U or t).
   I4 Auditability: every step emits a Merkle-chained record sufficient for deterministic replay and one-pass verification.
   I5 Predictability: control changes (lambda_*, beta_floor, gate clamps) are F_{t−1}-measurable to preserve e-process validity.

5. POSITIVE RANDOM FEATURES (PRF)
   For x in R^d:
   phi_i(x) = r^{−1/2} * exp( w_i^T x / sqrt(tau) − ||x||^2 / (2*tau) ), i=1..r
   Clip exponent g in [−c,c]; optionally pair (w,−w).
   Unbiased kernelization (no clipping):
   E_W[phi(q)^T phi(k)] = exp(q^T k / tau).
   Clipping bias budgeted by small clip rate rho_clip in [0.1%,1%] and logged.

6. SERA STATES AND UPDATES
   6.1 Enc-side Sera
   Let phi_t = phi(k_t).
   R_enc  <- gamma_enc * R_enc + GER_Kahan(phi_t, v_t^T)      // r x d_v  (or r x r_v)
   s_enc  <- gamma_enc * s_enc + KahanAdd(phi_t)               // r
   Preconditioner at ingest:
   mu_enc  <- (1−beta_mu) * mu_enc + beta_mu * phi_t
   sig2_enc<- (1−beta_sig)*sig2_enc + beta_sig * (phi_t − mu_enc)^2 + eps
   Risk/e-process (predictable):
   self-normalized stats on phi_t build E_mix_enc(t); margin_enc m_enc(t)=log(1/alpha)−log E_mix_enc(t)
   RJ_half (Enc): split coordinates; RJ_gap_enc reported.

6.2 Dec-side Sera (self-attention memory)
On each generated/teacher token with key k_t^dec and value v_t^dec:
R_dec  <- gamma_dec * R_dec + GER_Kahan(phi(k_t^dec), v_t^dec^T)
s_dec  <- gamma_dec * s_dec + KahanAdd(phi(k_t^dec))
Preconditioner mu_dec,sig2_dec updated predictably as in encoder.
Risk/e-process E_mix_dec(t); margins m_dec(t); RJ_gap_dec.

6.3 Sera Query (any side)
Given query q and side S in {enc,dec}:
phi_q   = phi(q)
phi_w   = diag(sig2_S + eps)^{−1/2} * phi_q          // query-only whitening
den_S   = phi_w^T s_S + lambda_*^S(t)                // >= 0, predictable
yhat_S  = (phi_w^T R_S) / den_S                      // or U( phi_w^T H_S / den_S ) if low-rank
Return yhat_S, den_S, RJ_gap_S, margin_S. Complexity O(r d) + O(r d_v) (or O(r r_v)).

7. ENCODER (Enc)
   7.1 Streaming Tokenization (reversible)

* Deterministic BPE/SP with byte fallback.
* Emit per token: {tok_id, byte_span, digests{raw_bytes,tok_seq}}.
* Merkle chaining: merkle_curr = H( step_body || merkle_prev ).

7.2 Embedding and K/V Heads

* x_t = Emb(tok_id) in R^d  (learned/frozen).
* k_t = W_k * x_t ; v_t = W_v * x_t   (or v_t_lowrank = U^T * x_t).
* Sera ingest Enc: update R_enc,s_enc,mu_enc,sig2_enc, E_mix_enc, RJ_gap_enc.

7.3 Hlm Featureizer

* Emit sparse ids: n-grams, type/shape, doc/meta tags, encoder diagnostics (e.g., den_enc bucketed, RJ_gap_enc bucketed, margin_enc bucketed).
* Insert/update via Hlm.LOCATEREF_OR_INSERT with strict O(1).

7.4 Complexity and Memory

* Per token: O(r d) + O(r d_v) (or O(r r_v)) + O(#ids)*O(1).
* State: O(r d_v)+O(r) + O(U_enc). No growth in t.

8. GENERATOR (Gen)
   8.1 Decoder Query Construction

* q_t = W_q * emb(y_{t−1}) + S_tiny   // small recurrent state if desired (kept O(1)/token)
* Optional conditioning on shallow context (e.g., time, speaker).

8.2 Self- and Cross-Attention via Sera
Self:
(yhat_self, den_self, RJ_self, m_self) = Sera.Query(q_t; dec)
Cross (encoder memory):
(yhat_cross, den_cross, RJ_cross, m_cross) = Sera.Query(q_t; enc)
Both use the SAME PRF basis W for tight cache and shared whitening stats per side.

8.3 Hlm Fusion (gate + linear add-on)
Diagnostics vector z_t:
z_t = [RJ_self, RJ_cross, m_self, m_cross, den_self, den_cross, ||phi_w||_1, rolling |e|, context bins...]  // all predictably computed
Gate (exact, interpretable):
G_t = sigma( w_g^T z_t )  // w_g learned by Hlm on sparse features derived from z_t
Context composition:
c_t = G_t * yhat_self + (1−G_t) * yhat_cross   // or meta-linear head learned by Hlm on [yhat_self,yhat_cross]
Logit linear add-on (exact sparse):
z_lin = Hlm.Linear(x_dec_t)  // optional per-token features (topic, persona, plan)

8.4 Logits and Risk-Aware Sampling
logits_t = W_out * c_t + b_out + z_lin
margin_t = m_dec(t)           // from Sera decoder e-process
temperature_t = f_temp(margin_t)         // monotone nonincreasing in risk
(top_k, top_p) = f_filter(margin_t, RJ_self, RJ_cross)
token_t = Sample(logits_t, temperature_t, top_k, top_p)
Predictable safety actions when margin_t<0 or RJ “red”:

* Clamp G_t <= G_max(margin_t) (favor cross or linear).
* Increase beta_floor and lambda_*^S predictably for next step.
* Reduce temperature and top_p bounds.

8.5 Online Learning (optional, supervised)
e_t = loss_grad(y_t*, logits_t) or e_t = y_t* − yhat_t (for regression heads).
Hlm updates exact per-id weights touched at step t.
Sera_dec ingest with k_t^dec, v_t^dec from the realized token embedding.

8.6 Complexity
Per token: 2×Sera query + small projections + Hlm gate/linear; overall O(r d) + O(r d_v) (or O(r r_v)) + O(#ids)*O(1).

9. HLM DETAILS (RECAP, IMPLEMENTATION-GRADE)
   9.1 Bounds and Defaults
   d=2, b=4, S=8, L=8, Q=16
   C_lookup <= 2*4 + 8 + 1 = 17
   C_insert <= 2*4 + 8 + 8 + 1 + c0 <= 25 + c0  (often ~19)
   9.2 Operations
   LOCATEREF(id):
   scan d*b delta slots, then S stash, then MPHF base membership via Keys_base[h(id)]; return &W if found; else NULL.
   LOCATEREF_OR_INSERT(id):
   if LOCATEREF != NULL return it
   if vacancy in candidate buckets: place, return
   else do <=L relocations (bounded), else stash if space, else ring enqueue (stable shadow), else emergency slot CAS; return pointer.
   9.3 Rebuild Scheduling (safety inequality)
   slack = C*(tau_high − tau_low)
   require slack >= lambda_max * T_rb + margin
   start rebuild at tau_low; guarantee swap before tau_high with probability 1 under the bound; maintain O(1) inserts indefinitely.
   9.4 Concurrency and Publication
   Build {h2,W2,Keys2} off-thread; publish pointer/version by single release-store; readers acquire-load at step entry; free old gen after epoch quiescence.

10. CONTROL PLANE (PREDICTABLE GOVERNORS)
    10.1 Sera→Hlm Early-Warning
    Persistent RJ “red” or thinning margins -> lower tau_low (earlier rebuild), increase C at next rollout, reduce Hlm learning rate temporarily.
    10.2 Denominator Safety
    If den_S < beta_floor: raise lambda_*^S predictably and clamp Gate down.
    10.3 Gate Policy
    Gate decreases stepwise when margin_t<0; recovers exponentially when margins positive over a window. All thresholds logged and predeclared.
    10.4 Learning-Rate Governance
    Hlm lr_scaled = lr_base * g(margin_t) with monotone g; ensures stable exact updates during turbulence.

11. SNAPSHOT / RESTORE
    11.1 Snapshot Payload
    Hlm: MPHF_blob, W_base, Keys_base, delta_dump(keys+weights+location), b, lr,l2, d,b,S,L,Q,C,tau_low,tau_high, seeds, versions.
    Sera Enc/Dec: R,s,(H), mu, sig2, risk accumulators (per-lambda logs or E_mix), r, r_v, gamma, tau, clip c, lambda grid, lambda_* schedules, PRF seed.
    Shared: audit rolling hash, last merkle_curr.
    11.2 Atomicity
    Two-phase write: data blobs then manifest with checksums; manifest links to Merkle chain. Crash leaves either old or new base referenced; delta_dump prevents loss.
    11.3 Restore
    Map arrays, rebuild handles, re-establish versions, resume predictable schedules; no reader pause on first request.

12. AUDIT AND ONE-PASS VERIFICATION
    12.1 Per-Step Record (JSON concept, kept ASCII)
    {
    t, role:{Enc|Gen}, ver_sera_enc, ver_sera_dec, ver_base, ver_delta,
    PRF_seed_hash, hash_salt_hash, r, r_v, tau, gamma_enc, gamma_dec,
    rho_clip_enc, rho_clip_dec,
    RJ_enc, RJ_self, RJ_cross, margin_enc, margin_dec,
    beta_floor, lambda_star_self, lambda_star_cross,
    den_self, den_cross,
    Gate, z_lin_digest,
    Hlm:{C_lookup_p100, C_insert_p100, ring_occ, stash_occ, emergency_used},
    rebuild_time_ms, slack_minus_arrivals,
    kahan_residuals:{R,s,(H)}, saturation_events,
    merkle_prev, merkle_curr
    }
    12.2 Verifier (one pass, O(1) memory)

* Check merkle_prev→curr continuity.
* Validate Ville margins: log(1/alpha) − log E_mix_S(t) >= 0 for S in {enc,dec}.
* Check RJ policy thresholds and action logs.
* Confirm Hlm step counts <= configured constants; check Keys_base[h(id)] == id on sampled entries (or full scan offline).
* Optional: verify linear Farkas certificates if present.

13. PSEUDOCODE (ASCII)

13.1 Enc Ingest
EncIngest(tok):
x = Emb(tok)
k = W_k * x
v = W_v * x          // or v_low = U^T * x
phi_k = PRF(k)       // clipped, paired optional
R_enc = gamma_enc * R_enc + GER_Kahan(phi_k, v^T)
s_enc = gamma_enc * s_enc + KahanAdd(phi_k)
mu_enc, sig2_enc = EWM2_update(phi_k)
update_e_process_enc(phi_k)       // predictable envelope
RJ_enc = RJ_half_split_enc()
emit_Hlm_features_from_encoder(tok, diagnostics_from{den_enc, RJ_enc, margin_enc})
AuditEmit(...)

13.2 Gen Decode Step
GenDecodeStep(prev_token, ctx):
pin_versions(ver_sera_enc, ver_sera_dec, ver_base, ver_delta)
q  = QProj(Emb(prev_token), ctx_state)
// Sera self
phi_q = PRF(q); phi_w_dec = whiten(phi_q; sig2_dec)
den_self = dot(phi_w_dec, s_dec) + lambda_star_self
yhat_self = dot(phi_w_dec, R_dec) / den_self
RJ_self, margin_dec = diagnostics_dec()
// Sera cross
phi_w_enc = whiten(phi_q; sig2_enc)
den_cross = dot(phi_w_enc, s_enc) + lambda_star_cross
yhat_cross = dot(phi_w_enc, R_enc) / den_cross
RJ_cross, margin_enc = diagnostics_enc()   // from pinned record
// Hlm fusion
z_gate = featurize(RJ_self, RJ_cross, margin_dec, margin_enc, den_self, den_cross, norm1(phi_w_dec), ctx)
G = clamp( sigmoid( Hlm.Linear(z_gate_ids) ), g_min(margin_dec), g_max(margin_dec) )
z_lin = Hlm.Linear(x_dec_ids)   // exact sparse contribution
c = G * yhat_self + (1−G) * yhat_cross
logits = W_out * c + b_out + z_lin
(temp, topk, topp) = risk_to_sampling(margin_dec, RJ_self, RJ_cross)
y = Sample(logits, temp, topk, topp)
AuditEmit(...)
return y

13.3 Gen Learn Step (optional supervised)
GenLearnStep(x_dec_ids, y_true, logits):
e = grad_loss(logits, y_true)
for id,val in x_dec_ids:
p = LOCATEREF_OR_INSERT(id)
*p = (1 − lr*l2) * (*p) + lr * e * val
b_hlm += lr * e

13.4 Sera Decoder Ingest (after emitting y)
SeraDecIngest(y):
x = Emb(y)
k = W_k_dec * x
v = W_v_dec * x
phi_k = PRF(k)
R_dec = gamma_dec * R_dec + GER_Kahan(phi_k, v^T)
s_dec = gamma_dec * s_dec + KahanAdd(phi_k)
mu_dec, sig2_dec = EWM2_update(phi_k)
update_e_process_dec(phi_k)

14. COMPLEXITY SUMMARY (PER TOKEN/EVENT)
    Enc ingest:  O(r d) PRF + O(r d_v) GER (+O(r r_v) if low-rank) + O(r) stats + O(#ids)*O(1) Hlm updates.
    Gen step:    2×(O(r d) + O(r d_v)) + O(#ids)*O(1) + tiny projections.
    Hlm ops:     LOCATE <= 17 primitive probes; INSERT <= 25+c0; independent of U,F,t.

15. MEMORY LAYOUT AND CACHE
    PRF:

* W (r x d) row-major; exp arguments computed as (w_i^T x)/sqrt(tau) − ||x||^2/(2*tau).
* Avoid per-call salt fetch; keep seed in TLS.
  Sera:
* R_* (r x d_v) row-major; s_* (r) contiguous; mu/sig2 contiguous.
* Use AOSoA if SIMD helps.
  Hlm:
* Buckets aligned to cache lines; (key,fingerprint,widx) packed; W_delta parallel array of float.
* Keys_base and W_base parallel arrays for stride-1 access.

16. DEFAULTS AND SIZING
    Sera:
    r ≈ C0 * (r_v + log(1/δ)) / ε^2   for target error ε and confidence 1−δ.
    r_v ∈ [64,128], tau ≈ sqrt(d), gamma_enc ∈ [0.97,0.99], gamma_dec ∈ [0.98,0.997].
    clip c tuned for rho_clip ≈ 0.5%; beta_floor in [1e−6, 1e−4] when gamma→1.
    Hlm:
    d=2, b=4, S=8, L=8, Q=16; tau_low=0.6, tau_high=0.8.
    C from lambda_max and T_rb with >= 2× safety.

17. FAILURE MODES AND RUNBOOK
    Sera margins low or RJ red:
    action: beta_floor↑, lambda_*^S↑ predictably, Gate clamp↓, reduce temperature/top-p, optionally r↑; log all.
    Hlm ring pressure or emergency used:
    action: tau_low↓ (early rebuild), increase C at next deployment, rebuild parallelism↑, lr↓ temporarily, confirm slack >= arrivals.
    Denominator instability:
    action: enforce beta_floor>0, extended precision accumulation, never quantize s or denominators.
    NaN/Inf:
    action: quarantine sample, record in audit, fall back to linear path (G→0) until healthy.

18. SECURITY AND PRIVACY

* PRF seeds and hash salts under KMS; rotate per rebuild; never log plaintext.
* Audit logs are append-only (WORM); per-tenant encryption and chain separation.
* Keys_base ensures no false-positive membership.

19. DETERMINISM MODE

* Fix event order; enable Kahan everywhere; pin lambda_* schedule; disable nondeterministic parallelism.
* Version seeds; bit-identical replay of audit stream should reproduce outputs modulo platform FP.

20. API SKETCH (ASCII)
    Enc.Ingest({bytes|tok_id,t}) -> Ack{ver_sera_enc, merkle_curr}
    Enc.ExportSummary() -> {digest(R_enc), digest(s_enc), ver_sera_enc, stats}
    Gen.DecodeStep({prev_tok,ctx,t}) -> {tok, logits_digest, yhat_self, yhat_cross, G, diagnostics, versions}
    Gen.LearnStep({x_dec_ids, y_true, lr, l2, t}) -> Ack{ver_base, ver_delta}
    Hlm.Predict({x_ids,t}) -> {yhat_hlm, ver_base, ver_delta}
    Hlm.Update({x_ids,y,lr,l2,t}) -> Ack{C_lookup_p100, C_insert_p100, ring_occ, emergency_used}
    Auditor.Append({json}) -> Ack{ok}

21. END-TO-END SCENARIO (DRIFT + NOVELTY BURSTS)
    Phase A steady:
    margins high, RJ low, Gate ~0.8, Hlm ring≈0, rebuilds rare.
    Phase B burst:
    RJ red 6/10, margins thin, den dips; controller predictably raises beta_floor, lambda_*; clamps Gate<=0.4; Hlm lowers tau_low and enqueues early rebuild; inserts remain O(1); p99 stable.
    Phase C recover:
    base published; seeds rotated; ring drains; margins recover; clamps relax; Gate returns to attention-heavy.

22. CHECKLIST (IMPLEMENTATION BRING-UP)
    [ ] Hlm: enforce I1–I4; log bounds; enable rebuild inequality.
    [ ] Sera: PRF clipping logged; whitening at query only; denominators in extended precision; RJ/e-process connected to policy.
    [ ] Enc: reversible tokenizer; K/V heads; Hlm featureizer; audit stream on.
    [ ] Gen: two Sera queries/token; gate via Hlm; risk-aware sampler; teacher forcing path.
    [ ] Audit: Merkle chain; one-pass verifier exercised; snapshot/restore rehearsal.

23. SUMMARY
    By co-designing Enc and Gen around Sera’s PRF-based, length-free attention and Hlm’s injective, exact linearity, the stack achieves: strict O(1)/token/event hot paths; deterministic audit and replay; quantitative health signals that steer gates, denominators, and rebuilds predictably; and interpretable contributions at every stage. It scales horizontally by sharding ids, remains reversible under crash/restore, and sustains real-time SLOs under drift without sacrificing correctness.
