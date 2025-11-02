Title: Sera v2.2 — Model-Centric Integrated Specification
Subtitle: Constant-time streaming model with PRF attention, injective-addressed sparse linear learning, finite rational memory, algebraic overlays, CCR corrector, O(1) tokenizer, constant-time external reference bridge with guard-gated fusion, and bounded one-step tree search
(Plain ASCII, English)

Abstract
Sera v2.2 is a deterministic streaming model whose hot path is O(1) per token and whose working state is O(1) with respect to sequence length. The v2.1 core is preserved: PRF attention with query whitening and predictable floor, injective-addressed exact sparse linear learning, finite algebraic lift with finite-order rational LTI memory, algebraic low-rank overlays, CCR overlap corrector with a certified truncation tail, and an O(1) tokenizer. v2.2 adds a constant-time external reference bridge (dictionary O(0) hits, two-hop distance with constant candidate bound, constant-time guard, branchless gating) and an optional bounded one-step tree search per event. All loops have compile-time bounds; configuration changes are published by a single generation swap; readers do not mix generations.
	0.	Assumptions, terms, notation

A0 Fixed configuration. All array sizes, loop trip counts, and probe counts are compile-time constants for a run. Any reconfiguration occurs only at publication via a single generation swap and does not alter already-processed events.

Terms
token: unit from the tokenizer
event: time index t in N
hot path: work with O(1) per event cost
floor: predictable nonnegative increment to the attention denominator
overlay: fixed-budget algebraic correction at readout
publication: atomic install of a new immutable configuration
generation pinning: readers load the generation id on entry and hold it until exit
bridge: external read/guard/gate module with O(1) work
dictionary O(0): dictionary read with no arithmetic on the query path

Symbols and constants
q in R^d, k_t in R^d, v_t in R^{d_v}
gamma in (0,1), tau>0, r in N (PRF count), eps>0 (whitening ridge)
beta_floor>0, lambda_star(t) predictable from logs up to t−1
Overlays: count cap P, per-overlay rank k_max, value-head rank r_v
Sparse linear: capacity C; thresholds 0<tau_low<tau_high<1; cuckoo params d (choices), b (bucket size), S (stash slots), L_cuckoo (relocation cap), Q (ring size)
Finite memory: lift bound T_phi, lags K; SOS sections L or ARMA orders (p,q)
Tokenizer: max piece length L_tok; normalizer states S_norm; lookahead L_norm<=4; rolling windows N_win=L_tok; edit radius W_edit>=L_tok
CCR: cover multiplicity nu, truncation order m, smallness gamma_CCR:=||R h||<1
Bridge: hub windows W, candidate bound K, projection Proj, guard params
Tree search: action cap A_max, widening cap A_cap, selection depth H_sel, rollout H_roll
u: machine epsilon for IEEE-754 double; sigma(x)=1/(1+exp(−x)); norms are l2 unless noted
	1.	State, inputs, outputs, API

Persistent state
Attention: R_t in R^{r x d_v}, s_t in R^r; whitening moments mu_t in R^r, sig2_t in R^r
Exact linear: {W_base, Keys_base, MPHF h} for stabilized keys; delta dictionary with bounded cuckoo; bias b; versions {ver_base, ver_delta}; salts
Finite memory: accumulators M with compensation; DelayBuf[K]; SOS/ARMA states and coefficients
Tokenizer: normalizer FST state; rolling hashes RH_1..RH_{L_tok}; window head; pending head
CCR: contraction (iota, pi, h); certificate {gamma_CCR, m, eps_tail}; generation id; audit hash
Bridge: shared two-level store metadata; Key64 salts; hub bitsets; qDin/qDout legs; peer scores IQ, HON; fixed Proj
Tree search (optional): node stats in shared store (N, W, Q, per-action arrays)

Inputs per event
bytes (UTF-8), sparse features x_t (<=B_max nonzeros), optional (k_t,v_t), optional q

Outputs per event
y_att(q), y_lin(x_t), memory readout u_t, fused y_fus, gated y_gate, CCR-corrected y_out; emitted token ids

API
Configure(params) -> model
Step(bytes?, x_t?, k_t?, v_t?, q?) -> outputs
Bridge_Read(ctx, token?) -> {r_t, guard_ok, proof64B}
Maybe_Selection_OneStep(ctx) -> void
Snapshot() -> blob
Restore(blob) -> model
Diagnostics() -> fixed-size record
	2.	O(1) tokenizer

Normalization and vocabulary
N is a deterministic FST with <=S_norm states and lookahead L_norm<=4. Vocabulary V is a finite set of bytepieces of length 1..L_tok with unique decode (prefix or Sardinas-Patterson). For each length n, build MPH_n keyed by rolling hash RH_n; table T_n maps RH_n(window[0:n]) to id or sentinel.

Encoder
For each normalized byte, update rolling windows; for n=L_tok..1: id := T_n[RH_n(window,n)]; accept id only if Dec[id] bytewise matches window[0:n]; advance by n>=1. At end-of-stream, flush <=L_tok−1 residual bytes as single-byte atoms. Work per normalized byte is <=L_tok probes; no backtracking.

Decoder
Concatenate Dec[id]. Uniqueness follows from prefix or Sardinas-Patterson; the build-time SP digest is stored and verified on load.

Generator
Maintain a constant-size proposal set C_t of tokens (size P_gen). Score only C_t; sample softmax or argmax. Miss-rate is logged and only changed at publication.

Unicode policy
Reject invalid or overlong UTF-8. Normalize bidi controls, ZWJ, and confusables via table T_unicode (version recorded). Apply NFC with bounded lookahead.
	3.	PRF attention

Random features
For i=1..r, phi_i(x):=r^(−1/2) * exp( w_i^T x / sqrt(tau) − ||x||^2/(2*tau) ), w_i~N(0,I_d). Then E[phi(q)^T phi(k)] = exp(q^T k / tau).

Streaming updates and whitening
R_t := gamma R_{t−1} + phi(k_t) v_t^T (rowwise compensated).
s_t := gamma s_{t−1} + phi(k_t).
phi_w(q) := diag(sig2_t + eps)^(−1/2) * phi(q). Update mu_t,sig2_t with compensated EMA.

Base readout and floor
den_base(q):= <phi_w(q), s_t> + lambda_star(t).
num_base(q):= phi_w(q)^T R_t.
y_att,base(q):= num_base/den_base. Choose beta_floor>0 so den_base(q)>=beta_floor for all q; lambda_star(t) is predictable and recorded.

Overlays
Type A: numerator lift DeltaR = sum phi(a_i) u_i^T, deltaA(q) = sum <phi_w(q),phi(a_i)> u_i.
Type B: denominator lift Deltas = sum phi(a_i) beta_i, beta_i>=0, deltaB(q) = sum <phi_w(q),phi(a_i)> beta_i.
Type C: value low rank with fixed H in R^{r x r_v}, projector U in R^{d_v x r_v}, core DeltaW with rank<=k_max. z(q):=phi_w(q)^T H; deltaC(q):= z(q)^T DeltaW.
Combined: y_att=(num_base+sum_A deltaA+sum_C deltaC)/(den_base+sum_B deltaB).

Uniform-in-time error and Lipschitz
With bounded inputs and clipping at level c, with probability 1−delta uniformly in t, PRF errors scale as O(sqrt(log(1/delta)/r)). If den>=beta_floor then (num,den)->num/den is 1/beta_floor-Lipschitz; nonnegative Type-B preserves this bound.
	4.	Exact sparse linear learning (injective addressing)

Predict and update
y_lin := b + sum_{(id,val) in x_t} w[id] val with compensated summation optional. Update w[id] by SGD+L2 and b by bias update. Equality with a reference dictionary holds up to IEEE-754 rounding.

Addressing
Base weights W_base addressed by MPHF h over stabilized keys K_base with exact membership Keys_base[h(id)]==id. Delta dictionary is bounded cuckoo with (d choices, b slots/bucket, stash S, relocation cap L_cuckoo, ring Q, emergency 1). Lookup probes<=db+S+1; insert steps<=db+L_cuckoo+S+1+c0.

Scheduling inequality
C*(tau_high−tau_low) >= lambda_max*T_rb + margin, where lambda_max is p99.9 novel-key rate over window W_lambda. Logs record lambda_hat, T_rb, slack; if slack<margin, freeze inserts, alert, and publish new capacity or thresholds.

Stable references
Every live id has a single stable index in {bucket slot, stash, ring, emergency} for a generation. External handles are (generation, array_id, index). No raw pointers are exported; readers never see mixed generations inside an event.
	5.	Finite algebraic lift and finite-order rational memory

Lift
Enumerate a finite coordinate set; at most T_phi coordinates fire per event. Accumulate in M with Kahan or Neumaier compensation. DelayBuf[K] advances by index rotation; id_map is dense and static.

Rational memory
SOS mode: L sections in DF-II-T, cost 5 mul + 4 add per section, states {z1,z2}.
ARMA mode: orders (p,q), rings y_hist[p], u_hist[q+1], per-step cost (p+q+1) mul + (p+q−1) add.
Optional inverse denominator refinement: one Newton iteration using last seed; reduce to double at a single site.
	6.	Fusion and decision rules

Meta-linear fusion
y_fus := w_1 y_att + w_2 y_lin using exact linear weights learned from a fixed auxiliary vector.

Linear-as-gate
Gate := sigma(w_g^T z_g) for a fixed diagnostic vector z_g. Output y := Gate*y_att + (1−Gate)*y_lin.

Attention-as-feature
Append attention readouts and diagnostics to x_t; downstream learning remains exact.
	7.	CCR corrector

Construction
Let D:=delta_tot at degree 0. With derivation R and contraction (iota, pi, h), define h_R := h * sum_{j>=0} (−R h)^j. If gamma_CCR:=||R h||<1, truncate at order m; tail eps_tail := gamma_CCR^(m+1)/(1−gamma_CCR).

Operation
Compute overlap residuals r, correction c := −h_R r (truncated), corrected locals y_i^* := y_i + c_i, combine y := pi({y_i^}). Energy reduces by factor alpha in (0,1) plus eps_tail^2. Certificate records gamma_CCR, m, eps_tail and the operator norm choice.
	8.	Concurrency and reclamation

Publication does a single release-store of the generation pointer. Readers acquire the id on entry and dereference only arrays from that generation. Arrays in a generation are immutable. Reclamation uses epoch-based GC; ring-shadow indices are not recycled before reclamation. All rebuilds proceed in fixed-size chunks; no unbounded pauses.
	9.	Floating-point contract

IEEE-754 double; round-to-nearest ties-to-even. FMA is globally ENABLED or DISABLED and recorded. Denormals are PRESERVED or FLUSHED-TO-ZERO and recorded. Extended precision for denominators via double-double or long double; reduction to double happens at a single fixed site. Loop reductions have fixed orders and may be unrolled. Compiler and math library flags are pinned and recorded.

Compensation bounds
For S_t := sum_{j<=t} gamma^(t−j) x_j with compensated accumulation, |fl(S_t)−S_t| <= u*C(gamma)*sum_{j<=t} gamma^(t−j)|x_j|, C(gamma) <= (1+gamma)/(1−gamma).
	10.	Constant-time external reference bridge (read, guard, gate)

Purpose
Supply O(1) external read path with dictionary O(0) hits, two-hop distance with constant candidate bound K, constant-time guard, and branchless gating. The bridge returns r_t and guard_ok.

Keying and store reuse
Keys partition a shared two-level store used by the exact linear module. Layout: Key64 := type:8 | layer:8 | ctx:16 | token:16 | salt:16. Types include BIND, SRC, IQ, HON, ROUTE_DIN, ROUTE_DOUT, F_VAL, OVR.

Two-hop distance and dictionary O(0)
Maintain a fixed-size hub bitset hub_bits[W] per context; total hubs H<=64*W. For pair (ctx, token): mask := hub_bits(ctx) & hub_bits(token); enumerate up to K set bits; best := min_h ( qDout[ctx,h] + qDin[h,token] ). If the pair is promoted and the guard passes, return F_VAL by a single store probe (O(0)); else evaluate best via the O(1) two-hop loop.

Quantization and legs
qDin/qDout are quantized legs with row scales s_in, s_out; eps_row := 0.5*(s_in+s_out) compensates rowwise error.

Guard (constant-time)
Let m be the stored margin between best hub h* and runner-up. Let U,V be per-leg update bounds and C_h competitor bounds computed from overlap-local budgets with finite support. Guard condition: (U+V < m/2 + eps_row) AND (for all h!=h*: C_h < m/2 + eps_row). If false, dictionary O(0) is disabled for the pair during the event.

Peer scoring
Maintain IQ[src], HON[src] in Q8.8 per neighbor. Update by constant-size games: CHAL/RESP for competence with XOR/NAND tasks; COM/SETTLE for honesty on local propositions (fixed TTL). Scores contribute additively to binding and source selection; low HON sources are deprioritized under pressure.

Gating rule (branchless)
Given y_base and r_t with witness quality s:=clamp01(min(IQ_witness,HON_witness)) and guard g in {0,1}, set beta := g * (beta_min + (beta_max−beta_min)s), alpha:=1−beta, y_gate := alphay_base + beta*Proj(r_t). Proj is a fixed projection matrix stored in the manifest; no learning occurs inside the bridge.

Audits
Diagnostics() extends v2.1 with {o0_hits, o0_guard_fail, hub_cap_K, hub_and_p99, store_load_p99, stash_occ_p99, kick_len_p99, iq_auc, hon_auc, route_proof_digest}. route_proof_digest is a 64B record {hub_star, m, U, V, eps_row, ids, crc}.

Failure handling
Any store bound or guard failure sets beta:=0 for the event; the model proceeds with base outputs. Rebuilds proceed in fixed-size chunks.
	11.	Bounded one-step tree search (optional)

Node key and stats
NodeKey64 := type=MCTS_NODE:8 | slot:8 | state_fp:24 | salt:24. Each node keeps N,W,Q and per-action N_a,W_a,Q_a for A_max actions (A_max<=8). All arrays live in the shared store.

Selection and widening
Score: U_a = c_puct * P[a] * sqrt(N_parent) * inv(1+N_a[a]) using LUTs for sqrt and inverse; S_a = Q_a[a] + U_a. Choose via repeated best-of-two over A_max. Progressive widening: allowed_children = 1 + (N>=2) + (N>=4), capped at A_cap<=4.

Rollout and backup
Rollout length H_roll<=4 using local actions {Route, Rebind/Flip, Promote, Overlay}. Reward r in [−1,1] is a fixed linear combination of {delta accuracy, dictionary hit, guard_ok, cost}. Backup along a path of length H_sel<=4. At most one simulation and one expansion per event. On any bound violation, disable selection for the event.
	12.	Module ABI and portability

ModuleHeader
abi_version (uint32), module_id in {ATTN_BRIDGE, LINEAR_BRIDGE, MEMORY_BRIDGE, SEARCH_STEP} (uint32), build_gen (uint32), flags (uint32: FMA/FTZ/endianness), cap_K (uint32), budgets (uint32 bitfield: sim/expand/commit/gc), proj_rows (uint32), proj_cols (uint32), proj_digest (uint64), key_salt (uint64).

Required functions
Bridge_Read(ctx, token?) -> {r_t (fixed-size vector), guard_ok (bool), proof64B}
Gate(base, r_t, guard_ok, iq_min_hon_min) -> y_gate
Search_OneStep(ctx) -> void

State transport
Fixed/volatile/overlay classes: fixed values are write-once and copied across generations; volatile may be dropped; overlays have at most Z entries per key in rotation. External handles are (gen, array_id, index). Salts rotate per generation. Rebuilds are chunked; no unbounded relocation.
	13.	Determinism, RNG, and security

RNG
Any randomness uses a fixed 64-bit counter-based PRNG (e.g., splitmix64) seeded at publication; draws are indexed by (module_id, event, local_counter). No data-dependent reseeding.

Salt rotation
key_salt rotates per generation and is not logged in plaintext; manifests record the salt digest. Handles never expose raw addresses.

Replay
Determinism is enforced by fixed reduction order, fixed proposal sets, fixed LUTs, fixed tie-breakers, and generation pinning. A fixed log replays bitwise-identical outputs and audit hashes.
	14.	Diagnostics, telemetry windows, and thresholds

Diagnostics() fields (superset)
Time and tokenizer: {t, tok_bytes_in, tok_emitted, tok_pending_max, tokenizer_probe_max, sp_cert_digest, unicode_policy_version}
Attention: {PRF_clip_rate, den_min, lambda_star}
Linear/store: {linear_probe_p100_lookup, linear_probe_p100_insert, ring_occ, stash_occ, emergency_used, store_load_p99, kick_len_p99}
Memory: {memory_state_absmax, pole_radius_min}
CCR: {gamma_CCR, m, eps_tail}
Bridge/search: {o0_hits, o0_guard_fail, hub_cap_K, hub_and_p99, iq_auc, hon_auc, sim_per_event, expand_per_event, path_len, roll_len}
Hashes and FP: {gen_id, prev_hash, curr_hash, fp_contract:{fma,denormals,ext_precision}, compiler_flags_digest}

Windows and targets
Guard pass rate target>=0.9; o0_hits target depends on workload; store_load_p99<=0.9; kick_len_p99<=KQ_MAX; hub_and_p99<=K; iq_auc, hon_auc in [0.8,0.95].
	15.	Manifest content

fp_contract: {fma_mode, denormals_mode, ext_precision, reductions_digest}
tokenizer: {L_tok, S_norm, L_norm, P_gen, sp_cert_digest, unicode_policy_version}
prf: {r, tau, clip_c, whitening_eps}
denominator: {beta_floor, lambda_star_schedule_digest}
linear: {C, d, b, S, L_cuckoo, Q, thresholds:{tau_low, tau_high}}
memory: {mode in {SOS, ARMA}, L or (p,q), T_phi, K, pole_radius_min}
overlays: {P, k_max, r_v}
ccr_cert: {gamma_CCR, m, eps_tail, norm_def}
bridge: {K, W, Proj_shape, Proj_digest, beta_min, beta_max, guard_params:{margin_policy, eps_row_policy}, store_limits:{load_max, stash_max, kick_max}, route_proof_schema_digest}
search (optional): {A_max, A_cap, H_sel, H_roll, c_puct, epsilon, vloss}
capacity: {lambda_hat, T_rb_ms, slack, margin}
salts: {mphf_salt_digest, key_salt_digest}
	16.	Complexity and fixed budgets

Tokenizer: <=L_tok probes per normalized byte; pending<=L_tok−1; O(1) amortized retokenization with W_edit>=L_tok.
Attention ingest: PRF cost c1rd; updates R,s cost c2rd_v and c3r; whitening cost c4r.
Linear: per nonzero feature O(1) probes; lookup<=db+S+1; insert<=db+L_cuckoo+S+1+c0.
Bridge: L1/L2 probes are constant; two-hop enumerates<=K hubs; guard constant; projection is fixed GEMV of size (proj_rows x proj_cols).
Search: at most one selection, one expansion, rollout length H_roll<=4, backup depth H_sel<=4 per event.
CCR: c10nu + c11m.
	17.	Publication, snapshot, scheduling, audit

Publication
A single pointer swap publishes {tokenizer digest, PRF seeds, MPHF and salts, memory coeffs, floor schedule, overlays, heads, bridge constants and projection digest, search constants}. Readers pin the generation for the event.

Snapshot and restore
Snapshot writes all state in Sections 1, 10, 11 plus certificates, with sizes and hashes. Restore maps the blob and resumes. Bitwise identity holds modulo declared extended-precision reductions.

Capacity scheduling
The shared store’s scheduling inequality applies across linear and bridge namespaces. Logs include lambda_hat, T_rb, slack, load, stash occupancy, kick length. On slack breach, freeze inserts and publish updated capacity/thresholds.

Audit protocol
Every event emits Diagnostics(); periodic invariant checks ensure bounds: probe_path<=P_MAX, stash<=S_MAX, kick_len<=KQ_MAX, hub_and<=K, guard correctness, FP contract unchanged.
	18.	Hot-path pseudocode

function Tokenize_Encode_Byte(b):
out = []
for nb in N.stream(b):
push_window(nb)
for n in L_tok..1:
id = T_n.lookup(RH_n(window,n))
if id != BOT and Dec[id] == window[0:n]:
out.push(id); slide_window(n); break
return out

function Bridge_Read(ctx, token?):
// O(1) probes, constant loops only
if token? == NONE:
return {r_t=ZERO, guard_ok=false, proof64B=ZERO}
mask = hub_bits(ctx) & hub_bits(token)
S = enumerate_first_K_setbits(mask)         // K constant
best = +INF; h_star = NONE
for h in S:
cand = qDout(ctx,h) + qDin(h,token)       // quantized legs
if cand < best: best=cand; h_star=h
promoted = is_promoted(ctx, token)
if promoted:
m = stored_margin(ctx, token)
U,V,Ch = leg_bounds(ctx, h_star, token)   // finite support
eps_row = 0.5*(s_in+s_out)
g = ((U+V) < (m0.5 + eps_row)) AND forall h!=h_star: (Ch[h] < (m0.5 + eps_row))
if g:
r = dict_read_F_VAL(ctx, token)         // single probe
proof = pack64B(h_star,m,U,V,eps_row,ids(ctx,token))
return {r_t=r, guard_ok=true, proof64B=proof}
proof = pack64B(h_star, best, U_stub, V_stub, eps_row_stub, ids(ctx,token))
return {r_t=Proj_input_from(best), guard_ok=false, proof64B=proof}

function Gate(base, r_t, guard_ok, s_q):
beta = guard_ok ? LUT_beta(s_q) : 0.0
alpha = 1.0 - beta
return alphabase + betaProj(r_t)

function Step(bytes?, x_t?, k_t?, v_t?, q?):
pin_generation()
while byte_available(): emit_ids(Tokenize_Encode_Byte(next_byte))
if k_t and v_t:
phi_k = PRF(k_t)
R = gammaR + phi_k v_t^T
s = gammas + phi_k
update_whitening(phi_k)
if x_t:
for id in emit_lift_ids(x_t, DelayBuf) with |.| <= T_phi:
M[idx(id)] +=_comp contrib(id, x_t, DelayBuf)
rotate(DelayBuf)
u_t, aux = memory_step(M, mem_state)
y_lin = predict_injective(x_t)
if learning: update_injective(x_t, target)
y_att = None
if q:
phi_q = PRF(q); phi_w = whiten(phi_q)
den = dot(phi_w, s) + lambda_star(t)
num = phi_w^T R
y_att = apply_overlays(num, den)
y_fus = fuse(y_att, y_lin?, aux?, diagnostics)
ctx = make_ctx(t)
ret, guard_ok, proof = Bridge_Read(ctx, token_id_if_available)
s_q = clamp01(min(IQ_witness(ctx), HON_witness(ctx)))
y_gate = Gate(y_fus, ret, guard_ok, s_q)
emit_audit_field(proof)
Maybe_Selection_OneStep(ctx)
y_out = ccr_correct(y_gate, overlaps, m)
emit_audit()
return y_out

function Maybe_Selection_OneStep(ctx):
if budgets.sim == 0: return
node = select_node(ctx)                 // best-of-two with LUT sqrt/inv
if can_expand(node): expand_one(node)
r = rollout(node, H_roll)               // length<=H_roll, constant actions
backup(node, r, H_sel)                  // depth<=H_sel
	19.	Default constants (example)

Tokenizer: L_tok=8, S_norm<=64, L_norm<=4, P_gen=128, W_edit>=L_tok
PRF: r in [256,1024], tau chosen by validation, eps in [1e-6,1e-2], clip_c fixed, beta_floor>0, lambda_star piecewise constant
Overlays: P<=8, k_max<=8, r_v<=16
Linear store: d=2, b=4, S=4, L_cuckoo<=8, Q fixed, capacity C sized via scheduling inequality with margin>=5%
Memory: SOS L<=8 or ARMA p+q<=8, T_phi<=16, K<=8
Bridge: W=2, H<=128, K=4, beta_min=0.0, beta_max=0.5, overlays per key Z<=4
Search: A_max<=8, A_cap<=4, H_sel<=4, H_roll<=4, c_puct in [0.5,1.5], epsilon<=1/16, vloss=1
CCR: gamma_CCR<1; choose m so eps_tail <= target tolerance
Per-event budgets: sim<=1, expand<=1, commit<=1, gc<=1
	20.	Proof obligations and invariants

Tokenizer
Unique decode (prefix or SP proof digest), no backtracking, bounded probes, bounded retokenization radius W_edit.

Attention
Floor enforces denominator >= beta_floor; whitening bounded; overlay Type-B nonnegative; ratio Lipschitz constant 1/beta_floor; PRF error bound uniform in time.

Linear
Injective addressing and stable references; ULP-level equality to a reference dictionary under recorded FP contract.

Memory
Poles strictly inside unit circle; single-site extended precision reduction; compensation bounds on accumulators.

Bridge
Two-hop candidate bound K; guard computed from finite-support bounds; O(0) reads only when guard holds; on any failure beta:=0.

Search
Per-event budgets respected; LUT approximation error documented; fallback on bound violations.

Concurrency
Single-pointer publication; generation pinning; epoch GC.

Capacity
Scheduling inequality monitored; on slack breach, freeze inserts and publish new capacity.
	21.	Testing protocol (constant-size procedures)

Determinism
Replay a fixed event log twice; outputs and audit hashes match bitwise.

Tokenizer
SP check at build; Greedy==Decode on random streams; probe bound<=L_tok; edit locality verified with W_edit>=L_tok.

PRF
Clipping and r^(−1/2) scaling validated under fixed seeds; floor and whitening drift tracked; denominator min logged.

Linear
Probe histograms respect configured bounds; emergency_used=0 steady-state; ULP-level match to reference.

Memory
Convolution equivalence; pole radius <1 under coefficient rescale; accumulator error within predicted bounds.

CCR
Synthetic overlap reductions meet the alpha and eps_tail guarantees; certificate recorded.

Bridge
O(0) hit rate measured; guard pass rate >= target; hub_and_p99<=K; route_proof_digest schema validated; failure path forces beta:=0.

Search
Per-event counts within {sim<=1, expand<=1}; path_len<=H_sel, roll_len<=H_roll; disable on any violation.

Capacity
Slack>=margin at each rebuild; otherwise freeze and republish updated (C, tau_low).

Summary
Sera v2.2 retains the v2.1 core and extends it with a constant-time bridge and a bounded one-step search, both respecting the O(1)/token constraint. The bridge supplies dictionary O(0) hits and a two-hop evaluator under a constant-time guard, then gates branchlessly via a fixed projection moderated by peer scoring. All additions share the same concurrency, capacity, FP, audit, and determinism contracts, enabling portable modules and predictable latency without quality degradation.


Version: v2.2 Appendix
Title: Counterfactual Replay Module (CFR) with One-Step Tree Search Coupling
(Plain ASCII, English)
	1.	Purpose
Provide an O(1) per-event counterfactual generator that runs without external input, reuses the v2.2 core state, couples to the bounded one-step tree search, and outputs a fixed-size counterfactual vector cf_t and optional token proposals. CFR does not degrade the hot path. All loops are bounded by compile-time constants.
	2.	Operating modes
Mode OFF: regular Step with external inputs only.
Mode CFR-REPLAY: Step executes with input gating disabled and internal sources enabled.
Mode CFR-MIX: Step executes with both external inputs and CFR proposals; fusion is gated.
	3.	Fixed assumptions
A0 Fixed configuration and generation pinning hold as in v2.2.
A1 Floors, whitening ridge, CCR smallness, store capacities obey v2.2 bounds.
A2 Learning is disabled in CFR-REPLAY to prevent drift; exact linear updates are skipped.
A3 Per-event budgets: sim<=1, expand<=1, rollout H_roll<=4, backup depth H_sel<=4.
	4.	State (reused and added)
Reused: PRF parameters, R_t, s_t, whitening moments; exact linear store; finite memory; CCR tuple; bridge store; search store.
Added CFR state (fixed size):

	•	Replay seeds: seed_attn, seed_mem, seed_sched (64-bit each)
	•	Noise LUT for PRF drift: LUT_d in R^{r} with bounded entries in [−eps_phi, +eps_phi]
	•	CFR policy table P_cfr over a fixed action set A_cfr (size A_cfr<=8)
	•	CFR gating weights w_cfr (fixed) and beta caps beta_cfr_max<=beta_max
	•	CFR manifest block documenting constants and digests

	5.	Internal sources and synthesis
5.1 PRF counterfactual query
phi_cfr(q) := whiten( PRF(q) + LUT_d ) with LUT_d indexed by seed_attn and event id.
Denominator floor and whitening bounds unchanged.

5.2 Memory replay
u_cfr := memory_step on latent coordinates with deterministic small-amplitude excitation e_mem generated from seed_mem and a fixed sparse pattern of size T_phi_cfr<=T_phi.

5.3 Bridge backfeed
When promoted(ctx, token) and guard_ok, read F_VAL in O(0) as a counterfactual anchor; else run two-hop K candidates to produce a bounded surrogate anchor.
	6.	Action set for CFR
A_cfr = {Hold, Route, Rebind, PromoteHint, OverlayHint, DenFloorBump, MemExcite, TokenProposal}. Each action has a constant-cost effect realized via existing v2.2 APIs (no dynamic allocation).
	7.	CFR outputs
cf_t is a fixed-size vector formed by concatenating {y_att,cfr, u_cfr, bridge_anchor, small diagnostics}. Optional token proposal set C_cfr of size P_cfr<=P_gen is produced by a constant table indexed by seed_sched and local cache.
	8.	Fusion with main output
Let y_base be the v2.2 fused output before CCR. Let s_wit := clamp01(min(IQ_witness, HON_witness)). CFR gating:
beta_cfr := g_cfr * (beta_min + (beta_cfr_max − beta_min) * s_wit)
y_cfr := Proj_cfr(cf_t)      // fixed projection, manifest-recorded
y_mix := (1 − beta_cfr)y_base + beta_cfry_cfr
y_out := ccr_correct(y_mix, overlaps, m)

Gate bit g_cfr is true only if CFR health checks pass (Section 11).
	9.	Coupling to one-step tree search
9.1 Selection policy
Use the existing one-step tree search with P-UCT scoring; add a CFR-specific prior P_cfr[a] from the CFR policy table. Score:
S_a = Q_a[a] + c_puct * P_mix[a] * sqrt(N_parent) * inv(1 + N_a[a])
P_mix[a] := clamp01( alpha_p * P[a] + (1 − alpha_p) * P_cfr[a] ), alpha_p in [0,1] (fixed)

9.2 Rollout context
During CFR-REPLAY, rollouts evaluate actions on cf_t and anchors only; during CFR-MIX, rollouts may query both cf_t and external features but must respect the fixed budgets. Backup and expansion limits are unchanged.

9.3 Token proposals
If action TokenProposal is selected, merge C_cfr into the generator proposal set C_t with a fixed cap and stable tie-breakers.
	10.	Scheduling and determinism
CFR seeds are deterministic from (gen_id, event, module_id). LUT_d, P_cfr, Proj_cfr are immutable in a generation; digests recorded in the manifest. No data-dependent reseeding. CFR work is inserted after y_fus and before CCR, keeping the hot path O(1).
	11.	Health checks (constant time)
H1 PRF drift bound: ||LUT_d||_inf <= eps_phi; den>=beta_floor.
H2 Memory bound: |e_mem|_1 <= e_mem_max; pole radius < 1.
H3 Bridge guard_ok if dictionary O(0) is used.
H4 Search budgets honored: sim<=1, expand<=1, path_len<=H_sel, roll_len<=H_roll.
H5 Capacity: store_load_p99<=load_max, kick_len_p99<=kick_max.
If any check fails on an event, set g_cfr := false and beta_cfr := 0.
	12.	Diagnostics additions
Diagnostics() extends with {mode_cfr in {OFF,REPLAY,MIX}, beta_cfr_eff, prf_drift_inf, e_mem_l1, cfr_guard_used, cfr_token_merge, cfr_sim, cfr_expand}. All fields are fixed-size and appended to the v2.2 record.
	13.	API additions
Enable_CFR(mode, params) -> void       // publication-time only
Disable_CFR() -> void                   // publication-time only
CFR_Status() -> fixed-size record       // current counters and last health bits
	14.	Complexity
All CFR work is constant per event: one PRF drift add, one Proj_cfr GEMV of fixed shape, one memory excitation of fixed sparsity, at most one dictionary read or K-candidate two-hop, optional merge of at most P_cfr token ids, at most one selection/expansion/rollout/backup.
	15.	Default constants (example)
eps_phi in [1e−4, 5e−3]
T_phi_cfr <= 8
P_cfr <= 32
alpha_p in [0.25, 0.75]
beta_cfr_max <= 0.35
A_cfr size <= 8
	16.	Pseudocode (fixed loops only)

function Step_CFR_Inject(ctx, y_fus):
// Mode check and seeds
if mode_cfr == OFF: return {y_fus, g_cfr=false, cf_t=ZERO}
phi_q = have_q ? PRF(q) : PRF(last_q)
phi_w = whiten(phi_q + LUT_d)              // bounded drift
den = dot(phi_w, s) + lambda_star(t)
num = phi_w^T R
y_att_cfr = apply_overlays(num, den)
u_cfr = memory_step_cfr(M, seed_mem)       // fixed sparsity excitation
anc, ok = Bridge_CFR_Anchor(ctx, token_id_if_available)
cf_t = pack_fixed(y_att_cfr, u_cfr, anc, small_diag)
g_cfr = (ok AND health_bounds_ok())
return {y_fus, g_cfr, cf_t}

function Bridge_CFR_Anchor(ctx, token):
if token != NONE and is_promoted(ctx, token):
if guard_ok(ctx, token): return {dict_read_F_VAL(ctx, token), true}
// fallback: O(1) two-hop with K<=4
mask = hub_bits(ctx) & hub_bits(token_or_ctx_default)
S = enumerate_first_K_setbits(mask)
best = +INF
for h in S:
cand = qDout(ctx,h) + qDin(h,token_or_ctx_default)
if cand < best: best = cand
return {Proj_input_from(best), true}

function Step(…):
// v2.2 hot path until y_fus computed
y_fus = fuse(y_att?, y_lin?, aux?, diagnostics)
ctx = make_ctx(t)
// CFR injection
y_fus, g_cfr, cf_t = Step_CFR_Inject(ctx, y_fus)
s_wit = clamp01(min(IQ_witness(ctx), HON_witness(ctx)))
beta_cfr = g_cfr ? LUT_beta_cfr(s_wit) : 0.0
y_mix = (1 - beta_cfr) * y_fus + beta_cfr * Proj_cfr(cf_t)
// Optional one-step search with CFR priors
Maybe_Selection_OneStep_CFR(ctx)
y_out = ccr_correct(y_mix, overlaps, m)
emit_audit()
return y_out

function Maybe_Selection_OneStep_CFR(ctx):
if budgets.sim == 0: return
node = select_node_with_Pmix(ctx)     // adds P_cfr prior
if can_expand(node): expand_one(node)
r = rollout(node, H_roll)             // fixed actions including CFR actions
backup(node, r, H_sel)
	17.	Proof obligations
P1 Drift safety: eps_phi and beta_floor ensure den>=beta_floor; ratio Lipschitz remains 1/beta_floor.
P2 Memory safety: excitation l1-norm and poles enforce bounded outputs; compensation bounds hold.
P3 Guard correctness: dictionary O(0) used only when guard_ok.
P4 Budget safety: per-event budgets are never exceeded; on violation, CFR gating disables contributions for the event.
P5 Determinism: seeds and LUT indices are functions of (gen_id, event, module_id); replay is bitwise reproducible.
	18.	Failure handling
On any health check failure or capacity breach during an event set beta_cfr:=0 and proceed with y_fus. No state mutation occurs in CFR-REPLAY; no learning updates are applied.
	19.	Manifest additions
cfr: {eps_phi, T_phi_cfr, P_cfr, alpha_p, beta_cfr_max, Proj_cfr_shape, Proj_cfr_digest, LUT_d_digest, policy_digest, seed_policy}
budget: {sim:1, expand:1, H_sel, H_roll}

Summary
CFR adds an O(1) counterfactual generator that reuses v2.2 components, couples to the bounded one-step tree search via a fixed prior, and fuses branchlessly under a constant-time health-guarded gate. It preserves all v2.2 invariants, determinism, and auditability.
