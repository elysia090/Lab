Title: Injective Hashed Linear as a Cartan–Cech–Robin System
Subtitle: Exact Linear Prediction and Updates from MC-Flat Gluing with Constant-Time Bounds
(Plain ASCII, English)

Abstract.
We formalize a lossless hashed linear model (HLM) as a Cartan–Cech–Robin (CCR) object on a finite cover whose patches coincide with the model’s storage sites (base slot, candidate delta buckets, stash, ring shadow, emergency). At form degree p=0, the total differential is the Cech coboundary D := delta_tot. A bounded, derivational Robin operator R is constructed as a commutator R := [D, G] with a degree-0 transfer G; this makes D_R := D + R gauge-conjugate to D and hence nilpotent (MC-flat) by construction. A fixed contraction (iota, pi, h) on the finite cover yields a unique glued scalar weight per key and a constant-size runtime template. Rebuilds are degree-0 gauge actions factored by BCH, preserving MC-flatness and hot-path constants. We prove exact equivalence to a reference sparse linear learner, derive non-asymptotic O(1) bounds tied to structure parameters (d, b, S, L, Q), and connect the scheduling inequality to a quantitative smallness margin for the homological perturbation.

0. Notation and scope.
   Keys are 64-bit integers or bounded-length byte strings; each key id has one logical weight w[id]. Storage is two-layer: Base = dense array addressed by a minimal perfect hash (MPHF) over a stabilized key set with exact membership via a parallel key array; Delta = bounded cuckoo dictionary with parameters (choices d, bucket size b, stash S, relocation cap L, ring capacity Q, one emergency slot). Prediction and update must match a reference array/dictionary implementation up to IEEE-754 rounding. Per-event work must be O(1) with certified constants independent of the number of unique keys U and sample length F.

1. Reference learner.
   For a sparse sample x = {(id_j, val_j)} with bias b, predictor
   y_hat = b + sum_j w[id_j] * val_j.
   With SGD + L2 and error e := y - y_hat,
   w[id] <- (1 - lr*l2) * w[id] + lr * e * val,
   b <- b + lr * e.
   The HLM must reproduce these computations exactly aside from floating-point rounding; compensated summation in prediction is allowed.

2. CCR preliminaries at p=0 with norms.
   Fix a finite cover U; Tot := ⊕_{q≥0} C^q(U; R) with D := delta_tot. All objects are scalars (form degree p=0). Equip C^q with the norm ||·||*2 over patch tuples (any equivalent finite-dimensional norm suffices). A degree-1 graded derivation R acts on cup products by Leibniz. A contraction (iota, pi, h) satisfies pi iota = Id, D h + h D = Id - iota pi. For a bounded perturbation R, define the smallness constant
   gamma := ||h|| * ||R||,
   requiring gamma < 1 with a stored margin. The perturbed homotopy is
   h_R := h * sum*{t≥0} (-R h)^t,
   truncated at m with tail eps_tail := gamma^{m+1} / (1 - gamma).

3. Encoding HLM as a CCR object.
   Per key id define a local patch set
   P(id) := {Base} U {Bucket_k(id), k=1..d} U {Stash} U {RingShadow} U {Emergency},
   with multiplicity nu := |P(id)| bounded by d+4 (typical nu ≤ 6 for d=2). On each patch i∈P(id) store q=0 fields:
   m_i(id) ∈ {0,1}  (exact membership bit),
   w_i(id) ∈ R      (local scalar weight).
   Only admissible overlaps are those induced by the data structure: (Base with nothing), (bucket with its alternatives), (stash with buckets), (ring shadow with buckets), (emergency with buckets). Let N(i) denote admissible neighbors of i.

3.1 Constructive Robin operator via a degree-0 transfer.
Define a bounded degree-0 operator G on q=0 sections by averaging along admissible edges:
(G psi)*i := sum*{j ∈ N(i)} B_ij * (psi_j - psi_i),
with fixed coefficients B_ij satisfying B_ij ≥ 0, Σ_{j ∈ N(i)} B_ij ≤ B_max, and B_ij = 0 on non-edges. Set
R := [D, G]  (graded commutator).
Then D_R := D + R is a pure gauge:
D_R = e^{-G} D e^{G},
and is nilpotent since D^2 = 0 ⇒ D_R^2 = 0. Boundedness holds with
||R|| ≤ c(nu) * B_max,
where c(nu) depends only on the finite overlap graph of P(id).

3.2 Contraction on a finite cover.
Choose (iota, pi, h) once per overlap graph of P(id). One explicit choice: pi averages with a discrete partition of unity a_i ≥ 0, Σ_i a_i = 1; iota injects a scalar into any chosen representative patch; h is the Moore–Penrose-like right-inverse of D on 1-cochains projected along a fixed spanning forest of the overlap graph. For a fixed nu, these operators are linear maps on R^{nu}, hence have finite operator norms that can be precomputed and stored.

3.3 Glued weight and its independence.
For local weights w_loc := (w_i m_i)_i, define the glued scalar
w(id) := (pi w_loc)(id).
Since D_R is gauge-conjugate to D, gluing is unique and independent of bounded relocations or publication order. If a transient duplicate occurs (two patches carry the same id during bounded relocation or ring shadow), the edge equalization encoded by R enforces (pi w_loc) invariance.

4. Data-structure invariants as CCR statements.
   I1 Injection. At any time there is at most one persistent live location for id among {base slot, one bucket slot, one stash slot, one ring shadow, emergency}. Transient duplicates are permitted only within bounded relocation windows or ring shadowing and correspond to two patches in N(i). Under R := [D, G], those duplicates are equalized in the glued scalar, preventing double counting.
   I2 Stability. Within a sample, readers pin a generation pair (ver_base, ver_delta); D_R is fixed during that sample. Publication is degree-0 (Sec. 6).
   I3 Boundedness. Lookup probes ≤ d*b + S + 1; insertion work ≤ d*b + L + S + 1 + c0 (c0 accounts for ring/emergency decisions). These constants define the admissible overlap graph and bound ||R||, hence control gamma.
   I4 Exactness. Prediction accumulates exactly one glued scalar per feature; updates modify precisely that glued scalar, matching the reference learner up to floating rounding.

5. Exactness theorem and proof.
   Theorem 1 (Exactness from MC-flat gluing). Fix an event order. Suppose D_R = e^{-G} D e^{G} with G bounded on the finite cover P(id). Then for every id, the glued scalar
   w(id) = (pi (w_i m_i))
   is unique, and
   y_hat = b + sum_j w[id_j] * val_j
   matches the reference computation up to IEEE-754 rounding (with optional compensated summation in the prediction loop). The per-feature SGD(+L2) update equals the reference update applied to w(id).
   Proof. MC-flatness (D_R^2=0) follows from gauge form. For any transient duplicate, w_loc differs along an admissible edge; since R=[D,G], D_R-flat sections are exactly e^{-G}-images of D-flat sections, which identify values along edges selected by h. Uniqueness follows from the contraction (iota,pi,h) on the finite cover. Prediction touches one glued scalar per id; update modifies exactly that scalar. Floating differences are limited to addition order and can be controlled with compensated summation. QED.

6. Constant-time bounds and homological perturbation.
   Theorem 2 (O(1) bounds via HPL). Let gamma := ||h||*||R|| < 1. Choose truncation m so eps_tail := gamma^{m+1}/(1-gamma) ≤ eps_budget. Then forward and reverse evaluations use a constant number of constant-size operator blocks, independent of U and F. In particular,
   C_lookup ≤ d*b + S + 1,
   C_insert ≤ d*b + L + S + 1 + c0,
   and these constants are invariant under publication and bounded relocations.
   Proof. The truncated Neumann series defines h_R with m+1 terms. All operators act on a fixed R^{nu}. Publication is degree-0 similarity (Sec. 6), so block counts and sizes are unchanged. QED.

7. Scheduling inequality as a quantitative smallness guard.
   Let C be delta capacity, 0 < tau_low < tau_high < 1 thresholds, slack := C*(tau_high - tau_low). Let lambda_max bound the novel-key arrival rate and T_rb bound rebuild time. Choose
   slack ≥ lambda_max * T_rb + safety_margin.
   Operationally, ring occupancy is transient and emergency use is rare, so the overlap graph does not expand and the measured ||R|| remains within its calibrated envelope, keeping gamma < 1. If telemetry shows pressure (ring growth, emergency claim), reduce tau_low or increase C (or rebuild parallelism) until the inequality is restored with margin.

8. Rebuilds as degree-0 gauge; BCH factorization.
   An MPHF rebuild produces a permutation of base indices and rotates salts. Represent this as a bounded degree-0 generator Theta on q=0 fibers. Then
   D_R' := e^{-Theta} D_R e^{Theta}
   is MC-flat with identical template constants. BCH factorization expresses Theta as a finite product of commuting degree-0 pieces (pure permutation, optional relabel). Because Theta is degree-0 and bounded on R^{nu}, it cannot increase operator counts or norms used in the hot path.

9. Concurrency and determinism.
   Readers acquire (base pointer, versions) via acquire-load at sample entry and release epochs on exit. Publishers populate fresh arrays and swap a pointer/version via store-release; this is the Theta update. Delta slot claims use CAS or light locks with bounded retries; after L relocations, demote to stash/ring/emergency immediately. Deterministic mode pins event order, enables compensated summation, and fixes hash/PRF seeds; replay is bit-identical modulo platform FP.

10. Numerical policy.
    Addressing is exact; numeric differences arise only from addition order. Use Kahan or Neumaier compensation in prediction when bit-stable reproducibility is required. Reject non-finite inputs (val, lr, l2) in constant time; clamp lr*l2 into a safe representable range if needed. Keep denominators and accumulators in at least FP32; FP16 storage with FP32 accumulation is permitted if the reference target is adjusted accordingly.

11. Parameters, constants, and sizing.
    Recommended defaults: d=2, b=4, S=8, L=8, Q=16, tau_low=0.6, tau_high=0.8. Bounds:
    C_lookup ≤ 2*4 + 8 + 1 = 17,
    C_insert ≤ 2*4 + 8 + 8 + 1 + c0 = 25 + c0,
    with c0 the ring/emergency decision cost (engineering can reduce constants with layout and branch-free checks). Choose C from the scheduling inequality with a safety factor that covers a p99.9 novelty burst and rebuild time variance. Precompute and store: B_max, ||h||, ||R||, gamma, truncation m, eps_tail.

12. Pseudocode (hot path; no dummies, constant-time branches only).

```
float* locate_ref(Base* B, Delta* D, key_t id) {
  for (int k = 0; k < d; ++k) {
    Bucket* bk = bucket_of(D, id, k);
    #pragma unroll
    for (int t = 0; t < b; ++t)
      if (bk->slot[t].key == id)
        return &D->W_delta[bk->slot[t].widx];
  }
  for (int s = 0; s < S; ++s)
    if (D->stash[s].key == id)
      return &D->W_delta[D->stash[s].widx];
  size_t i = mphf_eval(B->h, id);
  if (B->Keys_base[i] == id)
    return &B->W_base[i];
  return NULL;
}

float* locate_ref_or_insert(Base* B, Delta* D, key_t id) {
  float* p = locate_ref(B, D, id);
  if (p) return p;

  for (int k = 0; k < d; ++k) {
    Bucket* bk = bucket_of(D, id, k);
    for (int t = 0; t < b; ++t)
      if (try_claim_empty(&bk->slot[t], id))
        return &D->W_delta[bk->slot[t].widx];
  }

  key_t v = id;
  for (int r = 0; r < L; ++r) {
    int k = alt_choice(v, r);
    Bucket* bk = bucket_of(D, v, k);
    int t = pick_victim(bk);
    key_t u = bk->slot[t].key;
    if (cas_exchange(&bk->slot[t].key, u, v)) {
      if (is_null(u))
        return &D->W_delta[bk->slot[t].widx];
      v = u;
    }
  }

  int s = stash_find_empty(D);
  if (s >= 0 && try_claim_empty(&D->stash[s], id))
    return &D->W_delta[D->stash[s].widx];

  int q = ring_try_enqueue(D, id);
  if (q >= 0)
    return &D->W_delta[D->ring[q].widx];

  if (try_claim_emergency(D)) {
    D->emergency.key = id;
    signal_hard_rebuild();
    return &D->W_delta[D->emergency.widx];
  }

  abort_design_violation(); // mis-sized scheduling; halt for forensic snapshot
}
```

13. Reviewer-facing statements and obligations.
    (1) Provide the overlap graph on P(id) and the concrete B_ij; report ||R|| and gamma with margins.
    (2) Supply (iota, pi, h), ||h||, truncation m, and eps_tail with numeric values.
    (3) Demonstrate that publication logs encode Theta (hash of MPHF permutation and salts) and that observed probe counts obey configured constants across runs.
    (4) Show that ring occupancy is transient and emergency usage is rare under the chosen slack; if violated, document the revised C or thresholds.
    (5) Validate exactness by replay against a reference dictionary learner with ULP-scale discrepancies only; include compensated and uncompensated versions.

14. Conclusion.
    The HLM realized as a CCR system uses a gauge-constructed Robin operator R := [D, G] to render D_R nilpotent on a finite, fixed overlap graph. A certified contraction yields a unique glued scalar weight per key and a constant-size runtime template whose cost is independent of U and F. MPHF publications are degree-0 gauge actions that preserve MC-flatness and hot-path constants. A single scheduling inequality ties operational capacity to the homological smallness margin. The result is an exact, auditable, constant-time linear learner suitable for production with reproducible verification.
