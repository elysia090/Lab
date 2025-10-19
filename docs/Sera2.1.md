Title: Sera v2.1 — Model-Centric Integrated Specification (peer-review ready)
Subtitle: Constant-time streaming model with PRF attention, injective-addressed sparse linear learning, finite rational memory, algebraic overlays, CCR corrector, and an O(1) tokenizer
(Plain ASCII, English)

Abstract
Sera is a single streaming model whose hot path is O(1) per token and whose working state is O(1) with respect to sequence length. The model integrates: (i) positive-random-feature (PRF) attention with query-only whitening and a predictable denominator floor; (ii) an injective-addressed sparse linear learner that is exactly equivalent to a dictionary reference up to IEEE-754 rounding; (iii) a finite algebraic lift with finite-order rational LTI memory; (iv) algebraic low-rank overlays applied at readout; (v) a Cartan-Cech-Robin (CCR) overlap corrector with a certified truncation tail; and (vi) an O(1) tokenizer with bounded lookahead, exact inverse, and an O(1) generator. This document normalizes terminology and fixes assumptions, certificates, concurrency semantics, a floating-point contract, and auditable capacity scheduling. All loop bounds are compile-time constants; publications are single pointer swaps; readers never mix generations.
	0.	Assumptions, terminology, notation

Assumption A0 (Fixed configuration). All loop bounds and table sizes are fixed at configuration time and do not scale with sequence length, dataset size, or vocabulary size during a run. Any reconfiguration is published by a single generation swap and does not retroactively change already processed tokens.

Terms
token: unit produced or consumed by the tokenizer
event: time index t in N
hot path: operations whose per-event cost is bounded by configuration-time constants
floor: nonnegative, predictable increment to the attention denominator
overlay: fixed-budget algebraic correction at readout
publication: atomic installation of a new immutable configuration
generation pinning: readers snapshot the generation id at entry and use it for the entire event

Symbols and constants
q in R^d (query), k_t in R^d (key), v_t in R^{d_v} (value)
gamma in (0,1), tau>0, r in N (PRF count), eps>0 (whitening ridge)
beta_floor>0, lambda_star(t) predictable from logs up to t-1
Overlays: cap P, per-overlay rank k_max, value-head rank r_v
Sparse linear: capacity C; thresholds 0<tau_low<tau_high<1; bounded-cuckoo params d (choices), b (bucket size), S (stash), L_cuckoo (relocation cap), Q (ring size)
Finite memory: lift bound T_phi, lag count K; SOS sections L or ARMA orders (p,q)
Tokenizer: max piece length L_tok, normalizer states S_norm, lookahead L_norm<=4, rolling windows N_win=L_tok, edit radius W_edit>=L_tok
CCR: cover multiplicity nu, truncation order m, smallness gamma_CCR = ||R h|| < 1
u: machine epsilon for IEEE-754 double; all norms are l2 unless stated; sigma(x)=1/(1+exp(−x)).
	1.	Model state, inputs, outputs, API

Persistent state
Attention: R_t in R^{r x d_v}, s_t in R^r; whitening moments (mu_t, sig2_t).
Exact linear learner: {W_base, Keys_base, MPHF h} for stabilized keys; bounded-cuckoo delta dictionary; bias b; versions {ver_base, ver_delta}; keyed-hash salts.
Finite memory: lift accumulators M with compensation; DelayBuf head; SOS or ARMA states.
Tokenizer: normalization FST state; rolling hashes RH_1..RH_{L_tok}; window head; pending head.
CCR: contraction tuple (iota, pi, h); certificate {gamma_CCR, m, eps_tail}; generation id; audit hash.

Inputs per event: bytes (UTF-8), sparse features x_t with at most B_max nonzeros, optional pair (k_t, v_t), optional q.
Outputs: y_att(q), y_lin(x_t), u_t (memory readout), y_fus, optionally CCR-corrected y_out; emitted token ids.

API sketch
Configure(params) -> model
Step(bytes?, x_t?, k_t?, v_t?, q?) -> outputs
Snapshot() -> blob
Restore(blob) -> model
Diagnostics() -> fixed-size record
	2.	O(1) tokenizer (encoder, decoder, generator)

Normalization and vocabulary
A deterministic streaming normalizer N is a finite-state transducer with at most S_norm states and lookahead L_norm<=4. Vocabulary V is a finite set of bytepieces of length 1..L_tok satisfying coverage, unique decode (prefix or Sardinas-Patterson), bounded length L_tok, and compatibility with N. For each length n, build a minimal perfect hash MPH_n over V_n keyed by a rolling hash RH_n. Tables T_n map RH_n(window[0:n]) to a candidate id or a sentinel.

Tokenizer acceptance and flush
For window head W and n=L_tok..1 compute key:=RH_n(W), id:=T_n[key]. Accept id only if Dec[id] matches W[0:n] bytewise; otherwise continue with n−1. At end-of-stream, emit residual bytes (<=L_tok−1) as single-byte atoms. The pending buffer is at most L_tok−1 bytes.

Encoder
Greedy longest-match with bounded lookahead; at most L_tok table probes per normalized byte; no backtracking; advance by at least 1 byte on each acceptance.

Decoder
Exact inverse by concatenating Dec[id]; uniqueness follows from prefix or Sardinas-Patterson.

Generator and proposal policy
Maintain a constant-size proposal set C_t of token ids (size P_gen, fixed) from O(1) tables (atoms, short n-grams, local cache). Score only C_t from model readouts; sample softmax over C_t or take argmax. Proposal miss rate is logged; if the miss rate over a fixed window exceeds a threshold, tables are enlarged only at publication and the new bound is logged.

Unicode policy
Invalid UTF-8 and overlong encodings are rejected. Bidi controls, ZWJ, and confusables are normalized by a fixed table T_unicode (version recorded). Combining marks follow NFC with bounded lookahead (<=4 bytes). Policies are deterministic and recorded.

Lemma T-NB (No backtracking and bounded work)
If V satisfies unique decode and longest-match coverage, the encoder performs <=L_tok probes per normalized byte and never backtracks; acceptance consumes >=1 byte. End-of-stream correctness follows from the flush rule. A build-time Sardinas-Patterson check is run and its digest sp_cert_digest is stored in the manifest and verified on load.
	3.	PRF attention (state, whitening, floor, overlays)

Features and unbiasedness
For i=1..r and x in R^d, define phi_i(x) := r^(−1/2) * exp( w_i^T x / sqrt(tau) − ||x||^2/(2*tau) ), with w_i ~ N(0,I_d) i.i.d. Then E[phi(q)^T phi(k)] = exp(q^T k / tau).

Streaming statistics and whitening
R_t := gamma R_{t−1} + phi(k_t) v_t^T ; s_t := gamma s_{t−1} + phi(k_t).
phi_w(q) := diag(sig2_t + eps)^(−1/2) * phi(q). Row updates use Kahan or Neumaier compensation.

Base readout and floor
den_base(q):= <phi_w(q), s_t> + lambda_star(t) . num_base(q):= phi_w(q)^T R_t . y_att,base(q):= num_base / den_base. Choose beta_floor>0 so den_base(q) >= beta_floor deterministically. The floor schedule is predictable and piecewise-constant: lambda_star(t) = lambda_0 + sum_{j<t} Delta_j with Delta_j>=0 and Delta_j measurable w.r.t. the filtration F_j.

Overlays (fixed budgets)
Type A (numerator): DeltaR = sum_i phi(a_i) u_i^T, deltaA(q) = sum_i <phi_w(q), phi(a_i)> u_i.
Type B (denominator): Deltas = sum_i phi(a_i) beta_i, with beta_i>=0, deltaB(q) = sum_i <phi_w(q), phi(a_i)> beta_i.
Type C (value low rank): fixed H in R^{r x r_v}, projector U in R^{d_v x r_v}, rank<=k_max core DeltaW; z(q):=phi_w(q)^T H; deltaC(q):= z(q)^T DeltaW.
Combined attention: y_att(q) := (num_base + sum_A deltaA + sum_C deltaC) / (den_base + sum_B deltaB).

Theorem P-UT (Uniform-in-time PRF error, clipped)
Let F_t := sigma( (k_j,v_j){j<=t}, (w_i){i<=r} ). Under bounded inputs ||k_t||<=R_k and ||q||<=R_q, clipping at level c, and gamma in (0,1), for any fixed q and any delta in (0,1), with probability at least 1−delta, uniformly over t>=0,
|phi(q)^T R_t − A_t(q)| <= C_R(gamma,tau,c,R_q,R_k) * sqrt(log(1/delta)/r) and
|<phi(q), s_t> − B_t(q)| <= C_s(gamma,tau,c,R_q,R_k) * sqrt(log(1/delta)/r),
where A_t(q) := sum_{j<=t} gamma^(t−j) exp(q^T k_j / tau) v_j and B_t(q) similarly. Constants depend only on (gamma,tau,c,R_q,R_k).

Lemma R-L1 (Ratio Lipschitz with floor)
If den(q) >= beta_floor then (num,den) -> num/den is 1/beta_floor-Lipschitz; overlays with beta_i>=0 preserve this bound. Whitening drift appears as an additive term in the ratio error.

Overlay identifiability (O-ID)
Type-B beta_i>=0. The predictable floor lambda_star(t) is a safety control and is not optimized. Type-A and Type-C cores are learned on fixed budgets with l2 (or nuclear-norm for Type-C) regularization to avoid degeneracy.
	4.	Exact sparse linear learning (injective addressing)

Reference predictor and update
y_lin := b + sum_{(id,val) in x_t} w[id]val.
Update (SGD+L2): w[id] <- (1 − lrl2) w[id] + lr * e * val with e := y − y_hat ; b <- b + lr * e. Prediction may use compensated summation; equality holds up to IEEE-754 rounding.

Addressing and bounds
Base array W_base indexed by a minimal perfect hash h over stabilized keys K_base; exact membership via Keys_base[h(id)] == id.
Delta dictionary: bounded cuckoo with params (d, b, S, L_cuckoo, Q) and one emergency slot; ring shadows provide stable indices.
Lookup bound: <= db + S + 1 probes. Insert bound: <= db + L_cuckoo + S + 1 + c0 steps.

Scheduling inequality and audit
Capacity C and thresholds satisfy C*(tau_high − tau_low) >= lambda_maxT_rb + margin. Here lambda_max is the p99.9 novel-key rate over a window W_lambda under peak load. For each rebuild, log lambda_hat, T_rb, and slack := C(tau_high − tau_low) − lambda_hat*T_rb. If slack < margin, freeze inserts, raise alert, and publish increased C or lower tau_low.

Invariant H-S (Stable references)
Each live id occupies exactly one index in {bucket slot, stash slot, ring shadow, emergency}; indices are stable within a generation; only (generation, array_id, index) triples are exported; no pointer into movable arrays is exported. Publication never reuses indices within a still-reachable generation.

Theorem H-E1 (Exactness up to IEEE-754 rounding)
Under injection, stability, boundedness, and exactness invariants, predictions and per-feature updates match the dictionary reference up to rounding. Sketch: uniqueness of address, no mixed generations within an event, and bounded hot-path steps imply identical read/write site as the reference; compensation bounds prediction order error.
	5.	Finite algebraic lift and finite-order rational memory

Lift
A finite coordinate set is pre-enumerated. At most T_phi coordinates fire per event; accumulators M[m_eff] use Kahan or Neumaier compensation; DelayBuf[K] advances by index rotation only; id_map provides dense indices with no runtime allocation.

Rational memory
SOS mode: L sections, Direct Form II Transposed, per-section cost 5 multiplies + 4 additions, states {z1,z2}.
ARMA mode: orders (p,q) with rings y_hist[p], u_hist[q+1], per-step cost (p+q+1) multiplies + (p+q−1) additions.
Optional rational readouts maintain denominators in extended precision; one Newton refinement of inv_den uses the last step’s seed; reduce to double once per step.
	6.	Fusion and decision rules

Meta-linear fusion
y_fus := w_1y_att + w_2y_lin with weights learned by the exact linear learner from a fixed auxiliary vector.

Linear-as-gate
Gate := sigma(w_g^T z_g) with z_g a fixed small vector of diagnostics (RJ half-split, e-process margin, denominator stats) and finite-memory auxiliaries. Output y := Gate*y_att + (1 − Gate)*y_lin. Gate parameters are exact linear weights.

Attention-as-feature
Append stabilized attention readouts and diagnostics to the sparse feature vector; learning remains exact.
	7.	CCR corrector (constructive certificate)

Structure
Work at degree p=0, so D := delta_tot. Let R be a degree +1 derivation; define D_R := D + R with flatness D_R^2 = 0 (i.e., [D,R] + R^2 = 0). With contraction (iota, pi, h) and gamma_CCR := ||R h|| < 1, define the perturbed homotopy h_R := h * sum_{j>=0} (−R h)^j; truncate at order m with tail bound eps_tail := gamma_CCR^(m+1) / (1 − gamma_CCR).

Operation and guarantee
Local predictions on patches produce overlap residuals r. Compute c := − h_R r (truncated) and corrected locals y_i^* := y_i + c_i; combine y := pi({y_i^}). Energy reduction: E_ovl^ <= alpha * E_ovl + eps_tail^2 for some alpha in (0,1) depending only on (nu, m).
CCR certificate in audit records the operator norm definition, gamma_CCR, m, and eps_tail; assert gamma_CCR<1 at publication.
	8.	Concurrency and reclamation

Concurrency model
Publication performs a single release-store of the generation pointer. Readers acquire-load the generation on entry and dereference only arrays from that generation until exit. Arrays in a generation are immutable. Reclamation uses epoch-based GC: a generation is freed only after all readers that could observe it have advanced their epochs. Ring-shadow indices are never recycled before reclamation.
	9.	Floating-point contract and numerics

Floating-point contract
All arithmetic uses IEEE-754 double with round-to-nearest ties-to-even. FMA is either ENABLED or DISABLED uniformly and recorded. Denormals are either PRESERVED or FLUSHED-TO-ZERO and recorded. Extended precision for denominators is implemented via double-double or long double; reduction to double occurs at a single fixed site. Reduction order in all loops is fixed and may be unrolled. Compiler and linker flags and math libraries are pinned; no vendor BLAS is used for compensated sums. Settings are written into the audit manifest.

Compensation and rounding
For S_t := sum_{j<=t} gamma^(t−j) x_j accumulated with compensation,
|fl(S_t) − S_t| <= u * C(gamma) * sum_{j<=t} gamma^(t−j) |x_j| with C(gamma) <= (1+gamma)/(1−gamma). One Newton step v_t := v_{t−1}(2 − den_tv_{t−1}) achieves post-step relative error within double precision given the floor and a predictable seed.
	10.	Complexity and bounds (non-asymptotic)

Tokenizer encode per input byte: <= L_tok table probes + O(1) normalizer steps; pending buffer <= L_tok−1; local edit retokenization O(1) amortized with W_edit >= L_tok.
Attention ingest: PRF eval c1rd; R update c2rd_v; s and whitening c3r.
Finite memory: <= T_phi compensated adds; SOS 5L mul + 4L add or ARMA (p+q+1) mul + (p+q−1) add.
Exact linear learner: O(1) per touched feature; lookup <= db+S+1; insert <= db+L_cuckoo+S+1+c0.
Query: PRF eval c1rd; GEMV r x d_v cost c4rd_v; diagnostics c5r; overlays O(P*(r + k_max^2 + r_v)); CCR c10nu + c11m.
	11.	Publication, snapshot, audit

Publication
A new configuration (vocabulary and SP digest, FST, PRF seeds, MPHF and salts, memory coefficients, floor schedule, overlays, heads) is released by a single generation-pointer store; readers pin the id for the entire event.

Snapshot and restore
Snapshot writes a fixed-size blob with all state in Section 1 plus certificates and a versioned manifest with sizes and hashes; restore maps the blob and resumes. Bitwise identity holds modulo declared extended-precision reductions.

Audit record (fixed fields)
{ t, tok_bytes_in, tok_emitted, tok_pending_max, tokenizer_probe_max, sp_cert_digest, unicode_policy_version, PRF_clip_rate, den_min, lambda_star, PRF_digests, y_digests, linear_probe_p100_lookup, linear_probe_p100_insert, ring_occ, stash_occ, emergency_used, memory_state_absmax, pole_radius_min, CCR_cert:{gamma_CCR,m,eps_tail}, gen_id, prev_hash, curr_hash, fp_contract:{fma,denormals,ext_precision}, compiler_flags_digest }

Audit manifest (example schema)
{ gen_id, fp_contract, tokenizer:{L_tok,S_norm,sp_cert_digest}, prf:{r,tau,clip_c,clip_rate}, denominator:{beta_floor,lambda_star}, linear:{C,d,b,S,L_cuckoo,Q}, memory:{mode,L or p_q,pole_radius_min}, ccr_cert:{gamma_CCR,m,eps_tail}, capacity:{lambda_hat,T_rb_ms,slack} }.
	12.	Testing protocol (constant-size procedures)

Determinism: replay a fixed log twice; outputs and audit hashes match bitwise.
Tokenizer: Sardinas-Patterson passes at build; Greedy==Decode on random streams; probe bound <= L_tok; edit locality with window W_edit>=L_tok; pending_max logged.
PRF: r^(−1/2) scaling under clipping for fixed (gamma, tau, R_q, R_k, c); floor effectiveness and whitening drift tracked.
Exact linear: replay equivalence to a dictionary reference at ULP-level; probe histograms respect configured bounds; emergency_used=0 in steady state.
Finite memory: exactness against reference convolution (SOS/ARMA); stability under coefficient rescale; pole radius < 1.
CCR: synthetic two-patch residuals show energy reduction within bound; (gamma_CCR,m,eps_tail) logged at publication.
Capacity: for each rebuild, slack >= margin; otherwise automatic freeze of inserts then publication with updated (C,tau_low).
	13.	Minimal assumptions for deployment

A1 Bounded inputs: sup_t ||k_t|| <= R_k and sup ||q|| <= R_q.
A2 gamma in (0,1); beta_floor>0; lambda_star(t) predictable and logged.
A3 Overlay caps P, k_max, r_v fixed; selection emits constant-size sets with a documented comparator and tiebreak.
A4 Linear capacity C and thresholds satisfy the scheduling inequality with measured lambda_max and T_rb; salts rotate at publication and are not logged in plaintext.
A5 CCR smallness: gamma_CCR<1 is computed with a documented operator norm; m chosen so eps_tail meets tolerance.
A6 Determinism: seeds, evaluation order, tie-breakers, floors, compile flags, and publication logs are fixed and recorded.
	14.	Hot-path pseudocode (fixed loops only)

function Tokenize_Encode_Byte(b):
out = []
for nb in N.stream(b):                  // <= L_norm outputs
push_window(nb)                       // O(1)
for n in L_tok..1:                    // unrolled
id = T_n.lookup(RH_n(window,n))
if id != BOT and Dec[id] == window[0:n]:
out.push(id); slide_window(n); break
return out

function Step(bytes?, x_t?, k_t?, v_t?, q?):
pin_generation()
while byte_available(): emit_ids(Tokenize_Encode_Byte(next_byte))
if k_t and v_t:
phi_k = PRF(k_t)
R = gammaR + phi_k v_t^T             // compensated rows
s = gammas + phi_k
update_whitening(phi_k)
if x_t:
for id in emit_lift_ids(x_t, DelayBuf) with |.| <= T_phi:
M[idx(id)] +=_comp contrib(id, x_t, DelayBuf)
rotate(DelayBuf)
u_t, aux = memory_step(M, mem_state)   // SOS or ARMA
y_lin = predict_injective(x_t)         // bounded probes
if learning: update_injective(x_t, target)
if q:
phi_q = PRF(q); phi_w = whiten(phi_q)
den = dot(phi_w, s) + lambda_star(t)   // extended precision
num = phi_w^T R
y_att = apply_overlays(num, den)       // A/B/C, fixed caps
y_fus = fuse(y_att, y_lin?, aux?, diagnostics)
y_out = ccr_correct(y_fus, overlaps, m)
emit_audit(); return y_out
emit_audit(); return y_lin?
	15.	Summary

Sera v2.1 is a deterministic, audit-ready streaming model with strictly bounded hot-path loops. PRF attention is controlled uniformly in time via r and clipping, stabilized by a predictable floor and whitening; the sparse linear learner is exact with injective addressing and strict O(1) bounds; the finite rational memory realizes its class exactly with fixed operations and state; the tokenizer is O(1) per byte with unique decode, no backtracking, and bounded retokenization under local edits; overlays add constrained algebraic flexibility; the CCR corrector admits a certified tail. Concurrency, capacity scheduling, floating-point behavior, and certificates are explicit and verifiable. 
