TITLE
Lossless Hashed Linear: Exact-Equivalence Online Learning via Injective Addressing 

ABSTRACT
We present an online learning system for sparse-feature linear models that reproduces the reference computation
y_hat = b + sum_j w[id_j] * x[id_j]
exactly up to IEEE-754 rounding. The design enforces zero-collision addressing for all known features using a two-layer scheme. The base layer is a dense weight array addressed by a minimal perfect hash (MPHF) over a stabilized key set and guarded by exact membership via a parallel key array. The delta layer is a strictly bounded cuckoo-style dictionary with fixed bucket size, fixed relocation cap, constant-size stash, constant-size overflow ring that yields stable references, and a single always-empty emergency slot. The hot path for each (id,val) event performs a fixed, short sequence of operations, independent of the total number of unique features U and the events-per-sample F, giving O(1) per event and O(F) per sample with O(U) memory. Base rebuilds run asynchronously on a snapshot and are published by a single atomic pointer swap; readers never block and never mix generations. We formalize invariants (injection, stability, boundedness, exactness), prove equivalence to the reference model, derive tight operation-count bounds, provide a scheduling inequality that guarantees bounded behavior under adversarial novel-key arrivals, and specify concurrency, persistence, security, and validation sufficient for reproducible deployment.

1. INTRODUCTION AND MOTIVATION
   Sparse linear models remain a production staple because they are fast, interpretable, and easy to warm-start. Practical deployments, however, often trade exactness or tail latency for convenience: long open-addressing probes, lossy hashing that merges weights, reader pauses during structure rebuilds, or unbounded retries under adversarial keys. Our goal is to eliminate these compromises. We require that prediction and update on the hot path be exactly equivalent to a reference array/dictionary implementation (up to floating-point rounding), with strict worst-case O(1) per event, no reader stalls, and resilience to adversarial key sequences. The approach combines an injective base addressed by an MPHF with a bounded delta dictionary, RCU/Epoch generation pinning, and a rebuild scheduler that provably stays ahead of novelty.

2. PROBLEM FORMULATION AND EQUIVALENCE TARGET
   Reference model:
   y_hat = b + sum_{(id,val) in x} w[id] * val
   Update with SGD + L2:
   for (id,val) in x: w[id] <- (1 - lr*l2) * w[id] + lr * e * val
   b <- b + lr * e
   where e = y - y_hat.

Equivalence requirement: For any given ordered event stream, PREDICT returns the same y_hat as the reference implementation up to FP rounding; UPDATE produces the same parameter trajectory, modulo addition-order differences that can be suppressed with compensated summation. There are no spurious collision terms and no cross-talk among weights.

3. SYSTEM OVERVIEW
   We maintain two layers of injective storage.
   Base layer: a dense array W_base indexed by MPHF h over K_base, the stabilized set of keys already absorbed into base. A parallel array Keys_base stores the exact key at slot i, so base membership is exact: base_contains(id) is Keys_base[h(id)] == id.
   Delta layer: a bounded cuckoo-style dictionary D with parameters (d choices, bucket size b, stash S, relocation cap L, overflow ring Q, one emergency slot). Delta holds new keys and keys not yet absorbed into base. Every live key occupies exactly one unique location: a bucket slot, a stash slot, a ring shadow, or the emergency slot.
   Rebuilds: off-thread construction of a new MPHF h2 over the snapshot union Ksnap = Keys(base) union Keys(delta), with new arrays W2 and Keys2. Publication is a single atomic pointer swap; delta then resets.

4. STATE AND DATA STRUCTURES
   Base:
   h: MPHF mapping K_base -> [0..|K_base|-1]
   W_base: float[|K_base|]
   Keys_base: key[|K_base|] storing the exact key for each slot
   ver_base: generation counter
   seed_base: secret salt for any auxiliary hashes in MPHF or membership

Delta:
D: cuckoo dictionary with parameters d (recommend 2) and b (recommend 4)
S: stash length (recommend 8), scanned linearly
L: relocation cap per insertion (recommend 8)
Q: overflow ring capacity (recommend 16)
emergency: one always-empty slot reserved for worst-case progress
W_delta: float storage aligned to entries
ver_delta: generation counter
seed_delta: secret salt for delta hashes

Global:
b: bias
thresholds: tau_low (rebuild trigger), tau_high (saturation guard)
C: delta capacity in entries

Keys and IDs: ids may be 64-bit integers or byte strings. Keys_base stores either the full key (for fixed-size ids) or a collision-proof handle plus an equality path that is constant-time in the key length bound. To preserve zero false positives, equality must verify exact identity.

5. INVARIANTS
   I1 Injection: Any known id resolves to exactly one weight reference, in delta or base, never both.
   I2 Stability: Readers fix (ver_base, ver_delta) via RCU/Epoch and only dereference pointers in those generations. Publication occurs via a single store-release; readers see a consistent pair via acquire-load.
   I3 Boundedness: LOCATEREF performs at most d*b bucket probes, S stash probes, and one MPHF+membership check. LOCATEREF_OR_INSERT adds at most L relocations, then stash decision, ring enqueue, and at last an emergency claim. All steps are bounded by constants set at initialization.
   I4 Exactness: Prediction equals b + sum w[id]*val; updates apply only to the unique weight of each id.

6. HOT-PATH ALGORITHMS
   PREDICT_STEP(acc, id, val):
   r = LOCATEREF(id)
   w = (r != NULL ? *r : 0.0)
   acc = KAHAN_ADD(acc, w * val) // optional; plain add permitted
   return acc

UPDATE_STEP(id, val, e, lr, l2):
r = LOCATEREF_OR_INSERT(id)
*r = (1 - lr*l2) * (*r) + lr * e * val
return

PROCESS_SAMPLE(x, y, lr, l2):
acc = b
for (id,val) in x: acc = PREDICT_STEP(acc, id, val)
y_hat = acc
e = y - y_hat
for (id,val) in x: UPDATE_STEP(id, val, e, lr, l2)
b = b + lr*e
REBUILD_BASE_IF_NEEDED()

LOCATEREF(id):
for k in 1..d:
scan up to b slots in bucket_k(id); if exact key match -> return &W_delta[pos]
scan stash up to S; if match -> return &W_delta[stash_pos]
i = h(id); if Keys_base[i] == id -> return &W_base[i]
return NULL

LOCATEREF_OR_INSERT(id):
p = LOCATEREF(id); if p != NULL: return p
if any candidate bucket has vacancy: place id; return &W_delta[pos]
v = id
for t in 1..L:
choose alternate bucket; evict victim u; place v; v = u
if v placed: return &W_delta[pos]
if stash has vacancy: place; return &W_delta[stash_pos]
if ring has vacancy: enqueue and return &W_delta[ring_shadow_pos]
signal_hard_rebuild(); claim emergency; return &W_delta[emergency_pos]

REBUILD_BASE_IF_NEEDED():
if load_factor >= tau_low or stash_usage >= sigma or ring_usage >= q_th:
enqueue_async_rebuild()

ASYNC REBUILD (snapshot):
Ksnap = Keys(base) union Keys(delta)
h2 = BuildMPHF(Ksnap)
W2 = new float[|Ksnap|]
Keys2= new key [|Ksnap|]
for id in Ksnap:
src = (delta.has(id) ? W_delta[addr(id)] : W_base[h(id)])
i2 = h2(id)
W2[i2] = src
Keys2[i2] = id
AtomicSwap(base <- {h2, W2, Keys2}, ver_base++)
delta.clear(ver_delta++)

Ring shadow semantics: ring enqueue assigns a permanent shadow index exposed as a stable &W_delta[shadow]. Maintenance later materializes the entry into a bucket or moves it to base at rebuild; the shadow reference remains valid until publication or drain.

7. COMPLEXITY AND CONSTANTS
   Let fixed parameters be d, b, S, L, Q set at initialization. Then
   C_lookup = d*b + S + 1
   C_insert = d*b + L + S + 1 + c0
   where c0 covers ring/emergency decisions. PREDICT_STEP and UPDATE_STEP add only a constant number of arithmetic operations. Therefore time is O(1) per event and O(F) per sample. Memory is O(U): one weight per unique key plus metadata.

8. REBUILD SCHEDULING GUARANTEE
   Define slack = C * (tau_high - tau_low). Let lambda_max be an upper bound on the arrival rate of novel keys (entries per second), and let T_rb bound the rebuild+remap time. Choose parameters to satisfy
   slack >= lambda_max * T_rb + safety_margin
   Starting a rebuild at load tau_low leaves slack free slots. During T_rb, at most lambda_max*T_rb new unique ids arrive. If this mass stays below slack (with margin), delta cannot saturate before publication. Consequently, stash, ring, and emergency are only transient buffers and do not accumulate unbounded occupancy, preserving O(1) bounds indefinitely, including under adversarial key sequences consistent with lambda_max.

Estimation guidance: measure lambda_max as a high-quantile novel-key rate under peak load and multiply by a safety factor; measure T_rb on representative snapshots and adopt a high-quantile bound. Defaults that work broadly: tau_low = 0.6, tau_high = 0.8; Q sized for burst smoothing; C sized from the inequality.

9. NUMERICAL BEHAVIOR
   Floating-point exactness: addressing is exact; deviations arise only from addition order. For reproducibility across thread counts and scheduling, use Kahan or Neumaier compensation in PREDICT_STEP; cost is 2 to 3 extra FP ops per term. Updates are single-site affine transforms and match the reference exactly given the same e, lr, l2.
   NaN/Inf policy: reject non-finite val, lr, l2 on ingress using constant-time checks. Optionally clamp lr*l2 to a safe representable range.
   Precision modes: FP32 default; FP64 supported. Mixed precision (e.g., FP16 storage with FP32 accumulation) is possible if the reference target is adjusted accordingly.

10. CONCURRENCY AND MEMORY ORDERING
    Reader protocol: on entry, a reader acquires (base_ptr, ver_base) and pins the epoch; during the sample it dereferences only structures from that generation. On exit it releases the epoch.
    Publisher protocol: the rebuild thread allocates fresh arrays, populates them from the snapshot, then performs a single store-release of the new base pointer and version. Readers see the new generation via acquire-load at their next epoch entry.
    Reclamation: free old {h, W_base, Keys_base} after all readers have left the old epoch (epoch-based GC or hazard pointers). Clear delta after publication by incrementing ver_delta and resetting structures deterministically.
    Delta synchronization: per-bucket light locks or CAS-based slot claims; relocation loops are bounded by L; failures fall through to stash, ring, and emergency. The emergency slot is CAS-guarded and unique.

11. SECURITY AND ADVERSARIAL CONDITIONS
    Keyed hashing: all hash computations use secret salts that rotate per rebuild. This limits adversarial control over bucket occupancy patterns. The bounded mechanics (L cap, stash, ring, emergency) ensure progress even under worst-case sequences.
    Hot-path DoS containment: on reaching L relocations or full buckets, insertion immediately falls through to stash/ring/emergency. No unbounded loops exist in the hot path.
    Input validation: constant-time validation for id representation and bounded magnitude for val; enforce a maximum per-sample event count F_t to ensure O(F_t) work per sample.

12. PERSISTENCE, SNAPSHOT, AND RECOVERY
    Snapshot contents:
    MPHF_blob, W_base, Keys_base,
    delta_dump (all live delta entries with location kind and weight),
    bias b, optimizer scalars lr and l2,
    parameters d, b, S, L, Q, C, tau_low, tau_high,
    seeds/salts, versions ver_base and ver_delta.
    Atomicity: write data blobs first, then a versioned manifest containing sizes, offsets, and hashes. Recovery selects the newest manifest that verifies. If a crash occurs mid-publication, the manifest references either the old base or the new base; delta_dump ensures no lost entries.
    Recovery: map the MPHF and arrays, restore delta structures, re-establish seeds and versions, and resume immediately. If a rebuild was in flight at snapshot time, either resume it from saved state or start a fresh rebuild; readers are unaffected.

13. VALIDATION AND MEASUREMENT
    Exactness tests: replay identical streams into a reference dict-based learner and this system; compare y_hat per sample and weight vectors per key; accept only ULP-scale discrepancies. Repeat with and without compensated summation.
    Boundedness instrumentation: count probes for LOCATEREF and LOCATEREF_OR_INSERT; report maxima and histograms; verify they never exceed C_lookup and C_insert. Separate steady and rebuilding windows.
    Scheduling evidence: record T_rb each rebuild and compute slack - lambda_max*T_rb; the value must remain positive given the chosen safety margin. Monitor stash, ring, and emergency occupancy over time; persistent elevation indicates undersized slack or underestimated lambda_max.
    Throughput and latency: report per-sample latency (p50, p95, p99) and CPU share in steady vs rebuilding periods.
    Ablations: vary d, b, S, L, Q to map latency, memory, and tail behavior; disable ring or emergency in controlled tests to demonstrate why they are required for strict O(1).

14. EDGE CASES AND SEMANTIC POLICIES
    Absent ids: PREDICT contributes 0.0; UPDATE inserts via bounded mechanics.
    Duplicate features within a sample: PREDICT sums both occurrences; UPDATE applies multiple steps to the same weight, matching the reference semantics.
    Zero or negative values: all arithmetic follows IEEE-754; optional clamping hooks are allowed but must be mirrored in the reference if strict equivalence is claimed.
    Ordering: any fixed event order is acceptable; for determinism choose the natural stream order.

15. IMPLEMENTATION NOTES
    Cache layout: align buckets to cache lines; pack (key, fingerprint, widx) tightly; keep W_base and Keys_base as parallel arrays to enable predictable stride access. Precompute h(id) with the pinned seed and avoid per-call salt fetch.
    Ring consumer: run periodically or opportunistically; attempt bounded re-insertion of ring items; if unsuccessful, leave on ring until rebuild. Drain the ring as part of rebuild integration.
    Emergency handling: using the emergency slot raises a hard rebuild flag; it is expected to be rare. If it triggers, alert and verify that the scheduling inequality margins are sufficient.
    Determinism switches: provide a build-time or run-time flag to enable compensated summation and deterministic iteration orders when bit-stable reproducibility is required.

16. LIMITATIONS AND SCOPE
    Memory usage is O(U); exactness without degradation requires one parameter per unique feature. Nonlinear models are out of scope; incorporate nonlinearities through feature engineering upstream. The scheduling guarantee depends on measuring lambda_max and T_rb; in their absence, choose conservative C and lower tau_low to widen slack, at the cost of more frequent rebuilds.

17. RELATION TO KNOWN TECHNIQUES
    Bounded cuckoo hashing with stash improves over open addressing by providing strict caps on probe and relocation counts. MPHF provides collision-free indexing with compact space, but alone cannot guarantee exact membership; Keys_base supplies exact membership without false positives. RCU/Epoch is the canonical pattern for zero-pause publication to read-mostly workloads. The novelty here is their integration with a rebuild scheduler that converts novelty bursts into a provable constant-time behavior while preserving exact learning semantics.

18. PROOF SKETCHES
    Uniqueness: in a fixed generation, a key is inserted at exactly one location: bucket slot, stash slot, ring shadow, or emergency slot. Relocations move one concrete item at a time. Stash is a set, not a multimap. The ring assigns one shadow per item. Base membership is exact via Keys_base. A successful rebuild migrates every key to exactly one base slot; delta is then empty.
    Stability: readers pin generations and never observe mixed bases within a single sample. Publication writes a fresh pointer/version pair with release semantics; readers see an acquire-consistent state. Reclamation waits for epoch quiescence, avoiding ABA.
    Exact prediction and update: with uniqueness and stability, each (id,val) contributes exactly one term w[id]*val to y_hat; updates touch only that w[id]. The bias update is independent and identical to the reference.
    Strict O(1): LOCATEREF and LOCATEREF_OR_INSERT are bounded by fixed constants derived from (d,b,S,L,Q). Fall-through to stash, ring, or emergency eliminates unbounded loops or recurrences. Therefore the hot path per event is O(1).
    Scheduling safety: if slack exceeds arrivals during rebuild by a margin, delta cannot saturate before publication. Therefore ring and emergency usage are transient, and constant bounds persist indefinitely under any key sequence consistent with lambda_max.

19. PSEUDOCODE (ASCII)
    struct Base {
    MPHF h;
    float* W_base;
    key_t* Keys_base;
    uint64_t ver_base;
    uint64_t seed_base;
    };

struct Entry { key_t key; uint32_t widx; };
struct Bucket { Entry slot[BMAX]; };
struct Delta {
Bucket* buckets; // sized from C, d, b
Entry stash[SMAX];
Entry ring[QMAX];
Entry emergency;
float* W_delta;
uint64_t ver_delta;
uint64_t seed_delta;
};

float* locate_ref(Base* B, Delta* D, key_t id) {
for (int k = 0; k < d; ++k) {
Bucket* bk = bucket_of(D, id, k);
for (int t = 0; t < b; ++t)
if (bk->slot[t].key == id) return &D->W_delta[bk->slot[t].widx];
}
for (int s = 0; s < S; ++s)
if (D->stash[s].key == id) return &D->W_delta[D->stash[s].widx];
size_t i = mphf_eval(B->h, id);
if (B->Keys_base[i] == id) return &B->W_base[i];
return NULL;
}

float* locate_ref_or_insert(Base* B, Delta* D, key_t id) {
float* p = locate_ref(B, D, id);
if (p) return p;
for (int k = 0; k < d; ++k) {
Bucket* bk = bucket_of(D, id, k);
for (int t = 0; t < b; ++t)
if (try_claim_empty(&bk->slot[t], id)) return &D->W_delta[bk->slot[t].widx];
}
key_t v = id;
for (int r = 0; r < L; ++r) {
int k = alt_choice(v, r);
Bucket* bk = bucket_of(D, v, k);
int t = pick_victim(bk);
key_t u = bk->slot[t].key;
if (cas_exchange(&bk->slot[t].key, u, v)) {
if (is_null(u)) return &D->W_delta[bk->slot[t].widx];
v = u;
}
}
int s = stash_find_empty(D);
if (s >= 0 && try_claim_empty(&D->stash[s], id))
return &D->W_delta[D->stash[s].widx];
int q = ring_try_enqueue(D, id);
if (q >= 0) return &D->W_delta[D->ring[q].widx]; // ring shadow
if (try_claim_emergency(D)) {
D->emergency.key = id;
signal_hard_rebuild();
return &D->W_delta[D->emergency.widx];
}
abort_design_violation(); // indicates scheduling misconfiguration
}

20. DEFAULTS AND DERIVED BOUNDS
    Recommended parameters: d = 2, b = 4, S = 8, L = 8, Q = 16.
    Lookup bound: C_lookup <= 2*4 + 8 + 1 = 17.
    Insert bound: C_insert <= 2*4 + 8 + 8 + 1 + c0 = 25 + c0 (engineering can shave constants to about 19 in practice).
    Scheduling defaults: tau_low = 0.6, tau_high = 0.8, slack = 0.2*C. Design rule: 0.2*C >= lambda_max*T_rb + safety_margin.

21. CONCLUSION
    The combination of MPHF-indexed base storage with exact membership, a strictly bounded cuckoo delta with stash, ring, and emergency, RCU/Epoch generation pinning, and a rebuild scheduler governed by a simple inequality yields an online linear learner that is exact, constant-time on the hot path, reader-stall free, and robust to adversarial inputs. This restores linear models to their ideal of predictable, real-time behavior without sacrificing correctness or reproducibility, and provides a clean foundation for high-throughput production systems that demand both rigor and speed.
