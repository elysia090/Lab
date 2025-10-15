Lum module (plain ASCII, single module, fully O(1)/token)

Overview
Lum is a streaming operator that combines a finite algebraic lift (exact nonlinearity expressible with a fixed, finite set of coordinates) and a finite-order rational memory (exact infinite-horizon linear dynamics realized with a fixed cascade of second-order sections or a fixed low-order difference equation). All per-token work and state are bounded by configuration-time constants and do not depend on sequence length. No dictionaries or variable-length searches are used at runtime; all references are direct array indices determined at configuration time.

0. Notation and fixed assumptions

* Input at time t: x_t in R^d. Only up to B_max nonzeros per token; B_max is set at configuration.
* Output: y_t in R^{d_y}; d_y is a small constant.
* Lift: a finite set of coordinates Phi = {Phi_i}_{i in U}, |U| = m (constant). Each coordinate has a unique integer id produced by pack(tag,deg,lag,indices). Only coordinates enumerated at configuration can be touched at runtime.
* Memory: an exactly realizable rational transfer function H(z) = P(z)/Q(z) implemented either as a cascade of L second-order sections (SOS) or as a fixed-order difference equation with orders p and q. L, p, q are small constants.
* Arithmetic: IEEE-754 double. All running sums use compensated summation (Kahan or Neumaier). For ratio readouts that require a denominator, the denominator is kept in extended precision until the final division.

1. Classes handled exactly (inside the model family)

* Nonlinearity: any mapping representable by a finite algebraic lift, e.g., fixed-degree polynomials, finite-lag Volterra terms, and finite rational forms. All required lifted coordinates are enumerated in advance.
* Infinite-horizon memory: any LTI kernel with a rational transfer of fixed degree, realized exactly by an SOS cascade or a fixed ARMA difference equation. No truncation or decay is needed to keep O(1) cost.
* Hybrid: u_t = ReadLift(M_t) from the lift plane, then y_t = Memory(u_t) plus a fixed-size linear head over a small set of auxiliary features. Every step remains O(1).

2. Configuration-time constants and bounds

* T_phi: hard upper bound on the number of lifted coordinates that can fire per token (function of B_max, degree D, number of lags K); fixed at configuration.
* K: number of discrete lags used by the lift (ring buffer length); fixed.
* L: number of second-order sections in the memory cascade; fixed.
* (p, q): AR and MA orders if using the difference form; fixed.
* m_eff: number of lift accumulators actually materialized (possibly less than m due to symmetry or index filtering); fixed.
* d_y, size of auxiliary feature vector F_t, and the number of head weights W are all fixed small constants.

3. Data structures (all fixed-size, direct indexing)
   3.1 Lift

* DelayBuf[K]: ring of pointers to sparse views of past inputs for the selected lags. Rotation moves only a head index; no data copies.
* M[m_eff]: fixed array of accumulators (double), each with a compensation term for Kahan/Neumaier.
* id_map: a compile-time or load-time table that maps a canonical packed id to a dense array index in M. At runtime we never allocate or rehash; only direct array access is used.
* pack/unpack: pack(tag,deg,lag,indices) -> uint64 or uint128. Only ids present in id_map are ever generated.

3.2 Memory

* SOS[L]: for section s, coefficients {b0,b1,b2,a1,a2} and state {z1,z2} (Direct Form II Transposed).
* OR difference ring buffers: y_hist[p], u_hist[q] plus fixed coefficients {a_i}*{i=1..p}, {b_j}*{j=0..q}.

3.3 Head and auxiliaries

* W: fixed vector or small matrix of head weights.
* F_t: fixed-size auxiliary features constructed from {u_t, selected s-components, or other fixed-size lift summaries}. F_t dimension is fixed at configuration.

4. Per-token algorithm (complete O(1))

Input: x_t

Step 1. Lift coordinate emission and accumulation

* Extract the up-to-B_max nonzeros of x_t as index/value pairs (S).
* EmitLift(S, DelayBuf) deterministically enumerates at most T_phi coordinate ids that must be updated this token. Each id is guaranteed to be listed in id_map; obtain its array position pos via id_map[pos] = id.
* For each emitted coordinate u:
  M[pos].value  = KahanAdd(M[pos].value,  contrib(u; S, DelayBuf))
  M[pos].comp   = updated compensation term
* Rotate DelayBuf in O(1) by moving the head index only.

Step 2. Lift readout

* u_t = ReadLift(M) by evaluating a fixed number of linear or rational forms over M. If a rational readout is used:
  den_acc is maintained in extended precision
  num, den are computed with fixed operation counts
  inv_den uses one Newton-Raphson refinement from the previous inv_den as initial seed
  u_t = num * inv_den
* All operation counts are compile-time constants because the involved coordinates are fixed.

Step 3. Memory update
Option A: SOS cascade (L fixed)

* y_mem = u_t
* For s = 1..L (fully unrolled in production):
  tmp   = y_mem
  y_mem = b0[s]*tmp + z1[s]
  z1[s] = b1[s]*tmp - a1[s]*y_mem + z2[s]
  z2[s] = b2[s]*tmp - a2[s]*y_mem
* Total per section: 5 multiplies + 4 adds; across L sections this is fixed.

Option B: difference form (p, q fixed)

* y_mem = sum_{i=1..p} a_i * y_hist[i] + sum_{j=0..q} b_j * u_hist[j]
* Rotate the y_hist and u_hist rings by advancing indices; no data copies.

Step 4. Output head

* Compute F_t from a fixed set of values (e.g., u_t and selected z1/z2 entries). Size(F_t) is fixed.
* y_t = y_mem + W^T * F_t

Step 5. Audit record

* Append one fixed-size record:
  {t, fired_count, digest_ids, digest_M, y_digest, kahan_residual_flags, extended_precision_flags, prev_hash, curr_hash}
* The record size and the number of fields are fixed; hashing cost is constant.

Output: y_t

5. Query path (for a separate query vector q)

* Compute phi(q) by evaluating the fixed r components required by the configured readout. The indices to evaluate are prelisted; no dynamic selection.
* num_q = phi(q)^T * A (A is a fixed linear form over M)
* den_q = phi(q)^T * s + lambda (den_q maintained in extended precision until final division)
* y_hat = combine(num_q, den_q) or return the vector numerator/denominator components as requested. All operations are fixed-count O(1).

6. Eliminations and reductions that enforce O(1)

* No runtime dictionaries or perfect hashing. id_map is a dense, static array. All coordinate references are direct array indices. No insertion at runtime.
* All loops are over constants (T_phi, K, L, p, q). In production builds they are unrolled. There are no data-dependent branches inside the hot path.
* All rings use index rotation; no memmove or reallocation.
* All rational readouts use exactly one Newton-Raphson refine step with the previous inverse as the seed; this fixes the iteration count at 1.
* Denominators and a small, fixed set of critical accumulators are kept in extended precision across steps and reduced to double only at readout; conversion sites are fixed and counted.
* Memory update is implemented either as L SOS sections (with fixed per-section op counts) or as fixed-order ARMA difference equations. There is no O(n) state write. L or (p,q) are chosen as small constants at configuration.
* Learning of any coefficients is not performed on the hot path. If adaptation is required, it uses a fixed schedule and precomputed factorizations so that the per-token work remains O(1).

7. Operation and memory budgets (illustrative)

* Lift with T_phi = 64:
  about 64 compensated adds + a fixed number (<= 16) of simple ops for normalization or clamping

* SOS with L = 8:
  8 * (5 mul + 4 add) = 40 mul + 32 add

* Difference form with p = q = 4:
  9 mul + 8 add

* Head with 8 auxiliaries:
  8 mul + 7 add

* Audit:
  one hash update + fixed-length write

* All above numbers are constants independent of sequence length.

* Memory footprint (all fixed):
  M[m_eff] doubles + m_eff compensations
  DelayBuf[K] pointers or indices
  SOS states: 2*L doubles
  Difference rings: p + q doubles
  Head weights W and auxiliary scratch of fixed size

8. Numerical rules

* Every accumulator uses Kahan or Neumaier compensation with a fixed update pattern.
* For rational readouts, the denominator is stored and updated in extended precision; only one final reduction to double is performed per token.
* SOS uses Direct Form II Transposed to minimize state blow-up; coefficient scaling is allowed but is planned offline or on a slow path and takes effect at t+1 to preserve predictability.
* Operation order is fixed to ensure bitwise reproducibility across runs on the same hardware and compiler settings.

9. Snapshot and restore

* Snapshot writes a single fixed-size blob containing:
  M values and compensations, DelayBuf head index, SOS states or ring contents, coefficients for lift and memory, head weights, numeric flags, prev_hash
* Restore maps the blob and resumes. No reallocation or rebuilding. Behavior matches bit-for-bit except for hardware-dependent extended-precision reductions, which are controlled at build time.

10. Determinism and versioning

* Update order is fixed. Any coefficient or structure change is published by a single atomic generation swap. Readers pin the generation at entry; writers never block readers.
* Randomness is not used on the hot path. Any seed used at configuration (e.g., to build id_map) is stored and hashed into the audit chain.

11. API (plain text)
    Configure(params) -> lum
    params:
    lift_spec: {family, degree D, lags {tau_k} with K constant, id_map}
    memory_spec: {mode: "sos" or "diff", L or (p,q), coefficients}
    head_spec: {aux_list, W}
    numeric: {use_compensation, extended_precision_for: {denom,...}, clamp_limits}
    bounds: {T_phi, B_max}
    Step(lum, x_t) -> y_t
    Query(lum, q) -> y_hat or {num, den}
    Snapshot(lum) -> blob
    Restore(blob) -> lum
    Diagnostics(lum) -> fixed fields {fired_count_last, max_fired_seen, kahan_max_resid, denom_min, state_absmax, audit_hash}

12. Parameter selection

* Choose T_phi by bounding the emitted coordinate count for the chosen lift at the specified sparsity B_max and degree D; if needed, prune index patterns at configuration to enforce T_phi.
* Choose memory mode:
  SOS: pick L equal to the number of required complex-conjugate pairs and real poles/zeros; L rarely needs to exceed 8..16 for many periodic or resonant patterns.
  Difference: pick p and q for the targeted ARMA behavior (e.g., p=q=4..8).
* Head auxiliaries: keep 4..16 features drawn from u_t and selected state components to avoid growing op counts.
* Numeric floors for ratios: initialize floor at 1%..5% of typical denominators and adjust only on a slow path with effect from the next token.

13. Failure modes and fixed responses

* Denominator too small: apply a fixed additive floor lambda (configured), keep the activation count in diagnostics, and continue. Operation count unchanged.
* State drift near |lambda| = 1 poles: scale coefficients by a fixed factor on a slow path and activate at t+1. Hot-path cost unchanged.
* Fired coordinates exceed T_phi (configuration violation): drop surplus emissions deterministically by a fixed priority rule and log a counter. No dynamic allocation or search.
* Overflow risk in SOS states: clamp using fixed symmetric limits with counters; plan a coefficient rescale on the slow path.

14. Testing protocol (all constant-size procedures)

* Determinism test: run a fixed event log twice and check that y_t and audit hashes match bitwise.
* Exactness test for nonlinearity: create signals within the configured lift family and verify that outputs match a reference implementation to rounding error.
* Exactness test for memory: choose an explicit impulse response represented by SOS or ARMA and verify y_t equals full convolution (modulo rounding).
* Bound compliance: measure fired_count <= T_phi at all times; verify per-section op counts and ring rotations are constant.
* Numeric stability: track max absolute compensated residuals and denominator minima; assert under configured thresholds.

15. Example configurations

* Cubic nonlinearity plus seasonal memory
  lift: family=poly, D=3, T_phi fixed by pruning pairs
  memory: mode=sos, L=12 (monthly harmonics)
  head: 8 auxiliaries from {u_t, selected z1/z2}
* Volterra-2 with two fixed lags plus 2-mode resonator
  lift: family=volterra, K=2
  memory: mode=sos, L=2 (complex-conjugate pair)
* Rational lift and ARMA(4,4)
  lift: family=rational with separate numerator/denominator stats in M
  memory: mode=diff, p=4, q=4

16. Rationale for O(1)

* The lift never allocates or probes dynamic structures; only a fixed number of accumulators are updated per token by direct index.
* The memory update uses a fixed cascade or fixed-order rings; both have constant operation counts independent of sequence length.
* The head uses a fixed number of features and coefficients.
* Ratios use a fixed one-step inverse refinement and extended-precision storage with a single final reduction.
* All slow-path adjustments (coefficient rescale, floor changes) are predictable and take effect at t+1; the hot path remains unchanged.

17. Pseudocode (low-level)

struct LumState {
// lift
DelayBuf[K];          // ring heads only
double M[m_eff];      // accumulators
double C[m_eff];      // compensation terms for Kahan
// memory (choose one)
// SOS
double b0[L], b1[L], b2[L], a1[L], a2[L];
double z1[L], z2[L];
// or difference
double a[p], b[q+1];
double y_hist[p], u_hist[q+1];
// head
double W[H];          // H = fixed head feature count
// audit
hash_t prev_hash;
}

function Step(L, x_t):
S = sparse_nonzeros(x_t)                     // <= B_max
Uc = emit_lift_ids(S, L.DelayBuf)           // <= T_phi
for each id in Uc:
idx = id_map[id]                         // direct array index
val = contrib(id, S, L.DelayBuf)         // fixed arithmetic
y   = L.M[idx] + val
c   = (y - L.M[idx]) - val               // Neumaier comp
L.M[idx] = y
L.C[idx] = L.C[idx] + c
rotate(L.DelayBuf)                           // index++ mod K

u_t = read_lift(L.M, L.C)                    // fixed linear/rational forms

// memory: SOS example
y_mem = u_t
for s in 0..L-1:                             // unrolled
tmp   = y_mem
y_mem = L.b0[s]*tmp + L.z1[s]
L.z1[s] = L.b1[s]*tmp - L.a1[s]*y_mem + L.z2[s]
L.z2[s] = L.b2[s]*tmp - L.a2[s]*y_mem

// head
F = build_aux_features(u_t, L)               // fixed H-length
y = y_mem + dot(L.W, F)

// audit (fixed-size)
rec = digest(Uc, L.M, y, L.prev_hash)
L.prev_hash = hash(rec)

return y

This document specifies a single, product-independent module that implements exact finite algebraic nonlinearity and exact finite-order rational memory with constant per-token time and constant working state. All potential O(n) work has been reduced to fixed constants by pre-enumeration of lift coordinates, direct array indexing, second-order section factorization, loop unrolling, fixed-count inverse refinement for ratios, and elimination of runtime allocation and search.
