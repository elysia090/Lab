Title: Canonical Join-DAG with Admission-Controlled Evidence Checkpoints and Materialized Derived Indices
Specification v0.0.1
Status: Draft (semantics frozen; no implementation-defined behavior permitted)

1. Goals

1.1 Primary goals

A. Partial-order world: the base history is a DAG of events with a causal partial order, not a total order.

B. Confluence and idempotence by construction: all authoritative state updates are pointwise joins over join-semilattices, so evaluation is order-independent and replay-safe.

C. Constant-time hot-path validation: accepting and relaying high-frequency DeltaEvents is worst-case bounded by protocol constants and does not require reading prior state (except the canonical previous checkpoint root for the epoch).

D. Fast verifiable reads: clients verify values using a canonical checkpoint root plus a bounded membership proof.

E. Non-regressing extensibility: new tags and policies can be introduced without increasing per-event validation cost; evolution occurs via derived indices and governance keys defined as bounded joins.

F. Optimization-closed semantics: bandwidth allocation, admission, and ranking influence are deterministic functions of protocol constants, canonical roots, and event bytes, not mutable consensus state.

G. Mathematical closure: all normative choices (encoding, admission, inclusion caps, selection, fold semantics, derived indices) are deterministic and leave no implementation-defined degrees of freedom.

1.2 Non-goals

A. Global instant finality for every individual DeltaEvent.

B. Confidentiality/privacy beyond authenticity and integrity.

C. Minimizing total global work across all participants; the protocol minimizes per-node protocol work, not global economic cost.

D. Allowing implementations to choose incompatible cryptographic primitives or encodings.

2. Normative language

The key words MUST, MUST NOT, SHOULD, SHOULD NOT, MAY are to be interpreted as described in RFC 2119.

3. Notation

3.1 Bitstrings
{0,1}^n denotes the set of n-bit strings.

3.2 Hash
H(x) is SHA-256. Output is 256 bits.

3.3 Concatenation
x || y denotes byte concatenation.

3.4 Integers
All integer fields are fixed-width and little-endian unless stated otherwise.

3.5 Leading zeros
clz256(x) is the number of leading zero bits of a 256-bit value x.

3.6 Lexicographic order
For byte strings of equal length, lexicographic order is unsigned byte order.

4. Protocol identity constants (v0.0.1)

All constants below are part of the v0.0.1 protocol identity and MUST be identical across compliant implementations.

4.1 Address space and neighborhood

SPACE_BITS = 256
DEGREE_D = 8
NEIGHBOR_CONST[i] = H("nbr" || U32LE(i)) for i in 0..DEGREE_D-1

NEIGHBOR_PERM is a public permutation P: {0,1}^256 -> {0,1}^256 defined as an 8-round Feistel network:

Let v = L || R where L,R are 128-bit.
For r = 0..7:
F = Trunc128(H("F" || U32LE(r) || R))
(L, R) = (R, L xor F)
Then P(v) = L || R.

Neighbor function:
N_i(v) = P(v xor NEIGHBOR_CONST[i])

4.2 Event limits

MAX_PARENTS = 8
MAX_OPS = 16
MAX_EVENT_BYTES = 2048

4.3 Derived-materialization limits

WINDOW_MAX = 64
TOP_MAX = 32
CANDIDATE_MAX = 8
INDEX_MAX = 64

4.4 Epoch parameters

EPOCH_GENESIS = 0
CP_WINDOW = 1

Interpretation: a node MUST accept checkpoint candidates for epoch e only while its local highest accepted epoch is ≤ e+CP_WINDOW. After the node accepts any checkpoint with epoch > e+CP_WINDOW, it MUST reject all future checkpoints for epoch e.

4.5 Admission categories

CATEGORY_COUNT = 4

CAT_DATA = 0
CAT_RANK = 1
CAT_GOV = 2
CAT_RESERVED = 3

Per-category caps:

TARGET[cat] and SLACK[cat] are protocol constants (U16).
CAP[cat] = TARGET[cat] + SLACK[cat].

Per-category threshold bounds:

TMIN[cat], TMAX[cat], TargetZ[cat] are protocol constants (U8).

Genesis thresholds:

T0[cat] are protocol constants (U8) defining T_0[cat].

4.6 Cost weights

COST_BASE = 1
COST_PER_OP = 1
COST_PER_256B = 1
COST_HEAVY = 2

4.7 Signal diffusion parameters

SIG_T = 3
SIG_RHO_NUM = 1
SIG_RHO_DEN = 2

Numeric format for diffusion computations: signed fixed-point Q32.32 in int64.

4.8 Score stabilization constants

SURPLUS_CAP = 8

5. Cryptographic primitives

5.1 Hash
H is SHA-256 as defined above.

5.2 Signatures
Signature scheme is Ed25519.

pk encoding: 32 bytes
sig encoding: 64 bytes

SIGCHECK(pk, msg, sig) returns true iff sig is a valid Ed25519 signature of msg under pk.

5.3 Transcript hash
TR_e is defined in Section 12 and is used as a deterministic, epoch-bound salt.

6. Canonical encoding

6.1 Primitive encodings

U8(x): 1 byte
U16LE(x): 2 bytes
U32LE(x): 4 bytes
U64LE(x): 8 bytes
I64LE(x): 8 bytes two’s complement
B32(x): 32 bytes

BYTES(b): U16LE(len(b)) || b, where 0 ≤ len(b) ≤ 65535

VEC(ENC_elem, xs): U16LE(n) || ENC_elem(x1) || ... || ENC_elem(xn), with n = len(xs)

STR(s): BYTES(ASCII(s)), ASCII MUST contain only bytes 0x20..0x7E.

6.2 Canonical structure rule

For every structure type in this specification, the canonical byte encoding is exactly the concatenation of its fields encoded in the specified order using the primitives above. No alternative representation is permitted.

6.3 Canonical sorting rule

When a field is specified as “sorted”, the sort order MUST be lexicographic on the canonical bytes of the items.

6.4 Canonical hash binding

All hashes in this protocol are computed over canonical bytes. Re-encoding from a semantic object is forbidden; the canonical bytes are the source of truth.

7. Key space, tag IDs, and standard tags

7.1 Key space

K = {0,1}^256. Keys are exactly 32 bytes.

7.2 Tag ID embedding (self-describing keys)

For any key k, define tag_id(k) = k[0] (the first byte).

A compliant implementation MUST interpret tag_id(k) as the authoritative tag identifier for that key. Tag recovery from hashes or strings is forbidden.

7.3 Standard tag table

The protocol defines a fixed TagSpec table indexed by tag_id. Each TagSpec entry defines:

Category(tag_id)
ValueModel(tag_id)
PayloadShape(tag_id)
Normalizer(tag_id, ctx, payload)
CapRule(tag_id, ctx, normalized_payload)
AuthRule(tag_id, pk, k, normalized_payload)
Lift(tag_id, ctx, normalized_payload) -> DeltaValue (value domain element)
MaxValueBytes(tag_id) (U16, bound on stored state value encoding)

All entries below are protocol constants.

The v0.0.1 standard tag IDs:

0x01 OBJ (content-addressed objects)
0x02 LOG (windowed logs)
0x03 TOP (top-k rankings)
0x04 KEEP (retention votes)
0x05 CMD (command declarations)
0x06 CMDVOTE (command votes)
0x07 SCHEMA (schema declarations)
0x08 SCHEMAVOTE (schema votes)
0x09 PTR (name resolution candidates)
0x0A POS (position candidates)
0x0B IDX (indexer discovery candidates)
0x0C CUT (optional cut declarations)

System-derived, checkpoint-materialized tags (DeltaEvents MUST NOT write these):

0xE0 SIG (materialized signal b(v))
0xE1 CMDIDX (materialized active command index)
0xE2 SCHEMAIDX (materialized active schema index)

All other tag IDs are reserved and MUST be rejected if used in DeltaEvent ops.

8. State model

8.1 Authoritative data

The world’s authoritative data is the set of accepted DeltaEvents and accepted CheckpointEvents.

State at a checkpoint is derived by evaluating the joins induced by the DeltaEvents selected by that checkpoint plus deterministic system-derived virtual ops (Section 16), starting from the previous checkpoint state.

8.2 Join-semilattice requirement

For every key k, the value domain X_k and join operator ⊔_k MUST be commutative, associative, and idempotent.

8.3 Value models (closed family)

All TagSpec ValueModel entries MUST be one of the following three closed models.

Model A: MAX_BY(order_key)

Domain: either EMPTY or a single record R.
Join: choose the record with maximal order_key under a fixed total order; EMPTY is the identity.

Model B: TOPK_SET(K, order_key)

Domain: a set S of records, |S| ≤ K.
Join: S ⊔ T = TopK_K(order_key, S ∪ T), where TopK_K returns the K maximal records by order_key under a fixed total order. Ties MUST be broken inside order_key; the total order MUST be complete.

Model C: TOPK_MAP(K, mkey, best, order_key)

Domain: a set S of records, |S| ≤ K.
Define BestByKey(S): for each equivalence class by mkey, select exactly one record via best under a fixed total order (complete).
Join: S ⊔ T = TopK_K(order_key, BestByKey(S ∪ T)).

8.4 Canonical value encoding

Each tag defines a canonical state value encoding:

For MAX_BY: either U8(0) for EMPTY, or U8(1) || ENC(record)

For TOPK_SET / TOPK_MAP: VEC(ENC(record), records_sorted_by_order_key_descending), with no duplicates under the tag’s record identity rule.

MaxValueBytes(tag_id) MUST bound the encoded state value.

9. Event identity and canonical normalization

9.1 Event ID

All events have an ID:

id = H(canon_bytes_without_id)

Nodes MUST recompute and verify id matches.

9.2 DeltaEvent structure

DeltaEvent fields:

type_tag: U8(0x01)
epoch: U32
parents: VEC(B32, parents)
ops: VEC(ENC_Op, ops)
pk: B32 (Ed25519 pk)
sig: 64 bytes
nonce_incl: B32

Op structure:

k: B32
payload: BYTES(payload_bytes)

Total canonical encoded size MUST be ≤ MAX_EVENT_BYTES.

9.3 DeltaEvent normalization (MUST)

Let d be a DeltaEvent. Define DeltaEventNormalized(d) as follows:

A. parents are sorted ascending and deduplicated; if length > MAX_PARENTS reject.

B. ops are processed in the given order to produce normalized op records:

For each op:

1. tag_id = tag_id(op.k)
2. tag_id MUST be a standard non-reserved tag ID in Section 7.3 and MUST NOT be a system-derived tag (0xE0..0xE2). Otherwise reject.
3. payload_bytes are parsed using PayloadShape(tag_id). If parsing fails reject.
4. ctx is computed as defined in Section 11.
5. payload_norm = Normalizer(tag_id, ctx, payload_parsed). If Normalizer rejects, reject.
6. CapRule(tag_id, ctx, payload_norm) MUST be true; else reject.
7. AuthRule(tag_id, pk, op.k, payload_norm) MUST be true; else reject.

C. Key-local consolidation: group normalized ops by identical k. For each key k, define the consolidated DeltaValue:

DeltaValue_k = JoinLifted(tag_id(k), ctx, {payload_norm_i for this k})

Where JoinLifted is defined as:

* For each payload_norm_i compute lifted value xi = Lift(tag_id, ctx, payload_norm_i) in X_k.
* DeltaValue_k is the join of all xi under ⊔_k (well-defined by semilattice laws).

D. The normalized ops list is the set of consolidated ops, sorted by k ascending. Each normalized op stores:
k and a canonical payload_bytes that is ENC(DeltaValue_k) in the canonical state value encoding for the tag’s ValueModel.

E. The canonical bytes of the normalized DeltaEvent are the encoding of the DeltaEvent with normalized parents and normalized ops.

9.4 Delta signature message (MUST)

sigmsg_delta = H("sigmsg" || canon_delta_bytes_without_sig)

The signer signs sigmsg_delta. Verification uses this exact bytestring.

9.5 CheckpointEvent structure

CheckpointEvent fields:

type_tag: U8(0x02)
epoch: U32
prev_root: B32
batch_commit: B32
cut_commit: B32
delta_keys_commit: B32
delta_nf_commit: B32
sig_materialized_flag: U8 (0 or 1)
root: B32
meta: BYTES(meta_bytes)
proof: BYTES(proof_bytes)
pk: B32
sig: 64 bytes

sig_materialized_flag MUST be 1 in v0.0.1 (signal is materialized into state; Section 16.4).

9.6 Checkpoint signature message (MUST)

sigmsg_cp = H("cpsig" || canon_checkpoint_bytes_without_sig)

10. Cost, category, and admission tickets

10.1 Category of a DeltaEvent

For each op key k, op_category(k) = Category(tag_id(k)) from TagSpec.

The delta category is cat(d) = max precedence of its ops under precedence:

CAT_RANK > CAT_GOV > CAT_DATA > CAT_RESERVED.

If any op has CAT_RESERVED, the delta MUST be rejected.

10.2 Cost function

Let bytes(d) be the canonical byte length of DeltaEventNormalized(d).

heavy_count is the number of ops whose tag_id is TOP (0x03).

C(d) = COST_BASE

* COST_PER_OP * ops_count
* COST_PER_256B * ceil(bytes(d)/256)
* COST_HEAVY * heavy_count

All arithmetic is unsigned. C(d) ≥ 1.

10.3 Threshold parameters

T_e[cat] are the thresholds for epoch e, computed deterministically from canonical checkpoints (Section 13).

10.4 Ticket and work bits

Let prev_root = root_star(e-1). Let TR_{e-1} be defined in Section 12.

ticket(d,e) = H( "incl" || U32LE(e) || U8(cat(d)) || prev_root || TR_{e-1} || id(d) || nonce_incl(d) )

z(d,e) = clz256(ticket(d,e))

10.5 Required work bits

reqbits(d,e) = T_e[cat(d)] + ceil_log2(C(d))

ceil_log2(n) for n≥1 is the smallest integer r such that 2^r ≥ n.

10.6 Work-validity and surplus

A DeltaEvent d for epoch e is work-valid iff z(d,e) ≥ reqbits(d,e).

surplus(d,e) = z(d,e) - reqbits(d,e), an unsigned integer.

surplus_clip(d,e) = min(surplus(d,e), SURPLUS_CAP).

11. Context values (ctx) and selection key

11.1 ctx definition

For validating or lifting an op in DeltaEvent d at epoch e, ctx is:

ctx.e = e
ctx.prev_root = root_star(e-1)
ctx.TR_prev = TR_{e-1}
ctx.id = id(d)
ctx.ticket = ticket(d,e)
ctx.z = z(d,e)
ctx.reqbits = reqbits(d,e)
ctx.surplus = surplus(d,e)
ctx.surplus_clip = surplus_clip(d,e)
ctx.pk = pk(d)
ctx.pk_hash = H("pk" || pk)
ctx.op_index = the index of the op within the normalized ops array (0-based)
ctx.k = op.k

11.2 Deterministic tie key

tie32(ctx) = Trunc32(H("tie" || ctx.ticket || ctx.id || ctx.k || U32LE(ctx.op_index) || ctx.pk_hash))

Where Trunc32 takes the first 32 bytes.

11.3 Deterministic weight

w(ctx) = ctx.surplus (as U16, saturated to 65535)

12. Canonical roots, transcripts, and read anchors

12.1 Canonical checkpoint root

root_star(e) is the root of CanonicalCheckpoint(e) as defined in Section 14.

12.2 Transcript hash

TR_e = H("tr" || U32LE(e) || root_star(e) || meta_bytes_of_CanonicalCheckpoint(e) || batch_commit || cut_commit || delta_keys_commit || delta_nf_commit)

12.3 Read anchor

A client MUST select an epoch e and use root_star(e) as the read anchor. All reads MUST be verified against root_star(e) using the membership proof in Section 15.

13. Acceptance rules

13.1 AcceptDelta(d)

A node MUST accept and MAY relay a DeltaEvent d iff all conditions hold:

A. The event bytes decode as a DeltaEvent and the canonical normalized form DeltaEventNormalized(d) exists and its canonical bytes length ≤ MAX_EVENT_BYTES.

B. parents length ≤ MAX_PARENTS and ops length in 1..MAX_OPS after normalization.

C. SIGCHECK(pk, sigmsg_delta, sig) == true, where sigmsg_delta is defined in Section 9.4.

D. The node knows root_star(e-1) (or genesis prev_root = 0x00..00 for e=0) and TR_{e-1}, can compute T_e, and work-validity holds: z(d,e) ≥ reqbits(d,e).

E. All ops satisfy TagSpec parsing, Normalizer, CapRule, AuthRule, and CAT_RESERVED is not used.

Nodes MUST reject otherwise.

Nodes SHOULD deduplicate by id and MUST NOT relay duplicates.

13.2 AcceptCheckpoint(q)

A node MUST accept and MAY relay a CheckpointEvent q iff:

A. q.epoch is within the local acceptance window implied by CP_WINDOW. Specifically, if the node has accepted any checkpoint at epoch Emax, it MUST reject q if q.epoch < Emax - CP_WINDOW.

B. SIGCHECK(pk, sigmsg_cp, sig) == true.

C. q.prev_root equals root_star(q.epoch-1) once root_star(q.epoch-1) is known. If q.epoch=0 then prev_root MUST be 0x00..00.

D. VerifyCheckpointProof(q) == true as defined in Section 15.5.

Nodes MUST reject otherwise.

14. Canonical checkpoint selection

14.1 Candidate set

For epoch e, let Q_e be the set of accepted checkpoints with epoch == e that pass VerifyCheckpointProof.

14.2 Meta format (normative)

meta_bytes is the canonical encoding of:

meta.count_data: U16
meta.count_rank: U16
meta.count_gov: U16
meta.kthZ_data: U8
meta.kthZ_rank: U8
meta.kthZ_gov: U8
meta.surplus_sum_clip: U64

Counts MUST equal the number of included deltas of each category in the checkpoint.

kthZ_cat is the z-value of the CAP[cat]-th strongest included delta in that category, where deltas are ordered by z descending and tie32 ascending; if fewer than CAP[cat] deltas are included for that category, kthZ_cat MUST be 0.

surplus_sum_clip MUST equal the sum over all included deltas of surplus_clip(d,e), accumulated in U64.

VerifyCheckpointProof MUST attest all meta fields.

14.3 Score vector (MUST)

ScoreQ(q) is the lexicographic tuple:

( meta.surplus_sum_clip,
meta.kthZ_rank,
meta.kthZ_gov,
meta.kthZ_data,
inv_tie )

Where inv_tie = bitwise-not of H("cp_tie" || TR_e || q.pk || q.root) interpreted as a 256-bit unsigned integer, so “smaller hash wins” is implemented as larger inv_tie.

14.4 Canonical selection

CanonicalCheckpoint(e) is the q in Q_e with maximal ScoreQ(q) under lexicographic order.

Once selected, q.root becomes root_star(e) and TR_e is computed as in Section 12.2.

14.5 Practical finality

A node MUST treat root_star(e) as final once it has accepted any checkpoint for epoch e+CP_WINDOW. After that point it MUST NOT accept any new checkpoint candidates for epoch e.

15. Authenticated state root and proofs

15.1 Root structure (fixed, not pluggable)

The state root is a fixed-depth Sparse Merkle Tree over 256-bit keys, SMT256.

Define:

val_hash(vbytes) = H("val" || vbytes)

Leaf hash:
leaf_hash(k, present, vbytes) =
if present == 1 then H("leaf" || k || val_hash(vbytes))
else H("leaf" || k || B32(0x00..00))

Internal node hash:
node_hash(left, right) = H("node" || left || right)

Root computation:
Given a mapping from keys to present values, all unspecified keys are treated as present==0. The SMT root is computed by hashing leaves at depth 256 and internal nodes upward using the corresponding key bit to select left/right.

15.2 Membership proof format (fixed)

A MembershipProof for key k is:

present: U8 (0 or 1)
vbytes: BYTES (value bytes, MUST be empty if present==0)
siblings: VEC(B32, hashes) with length exactly 256, where siblings[i] is the sibling hash at depth i (0 is root-adjacent, 255 is leaf-adjacent)

Verification:
To verify, compute leaf = leaf_hash(k, present, vbytes). Then iteratively combine with siblings along the path determined by bits of k (MSB-first), using node_hash, to reconstruct root'. The proof is valid iff root' == root_star(e) (or == prev_root when verifying old state).

15.3 Non-membership proof

A non-membership proof is a membership proof with present==0.

15.4 Read verification

VerifyMembership(root, k, vbytes, proof) returns true iff:
A. proof.present == 1
B. proof.vbytes == vbytes
C. SMT verification yields root
D. len(vbytes) ≤ MaxValueBytes(tag_id(k))

VerifyNonMembership(root, k, proof) returns true iff:
A. proof.present == 0
B. proof.vbytes is empty
C. SMT verification yields root

15.5 Checkpoint proof (fixed statement, fixed witness format)

A CheckpointProof is a witness that allows the verifier to recompute and validate the checkpoint deterministically.

The proof bytes decode as:

P.included_ids: VEC(B32, ids_sorted_ascending)
P.frontier_ids: VEC(B32, ids_sorted_ascending)
P.delta_keys: VEC(B32, keys_sorted_ascending_unique)
P.delta_nf: VEC(ENC_KV, kv_sorted_by_key) where ENC_KV = B32(k) || BYTES(delta_value_bytes_for_k)
P.new_values: VEC(ENC_KV, kv_sorted_by_key) (the post-join values for the same keys)
P.prev_multiproof: VEC(MembershipProof, one per key in P.delta_keys) against prev_root
P.root_multiproof: VEC(MembershipProof, one per key in P.delta_keys) against q.root

All vectors MUST be sorted as indicated, and the key sets MUST match across delta_keys, delta_nf, new_values, prev_multiproof, root_multiproof.

VerifyCheckpointProof(q) MUST perform all checks below and return true iff all succeed:

1. q.batch_commit == MerkleRoot(P.included_ids), using SHA-256 Merkle with leaves H("mleaf"||B32(id)) and parent H("mnode"||left||right). If odd count, duplicate last.

2. q.cut_commit == MerkleRoot(P.frontier_ids) with the same Merkle rules.

3. The verifier MUST obtain the canonical normalized bytes of every delta id in P.included_ids. If any is missing, verification fails.

4. For each included delta d:
   A. d.epoch == q.epoch
   B. DeltaEventNormalized(d) exists and its id matches the referenced id
   C. SIGCHECK(pk, sigmsg_delta, sig) is true
   D. work-validity holds using root_star(e-1) and TR_{e-1}
   E. All TagSpec checks for all ops succeed (parsing, Normalizer, CapRule, AuthRule) and CAT_RESERVED not used.

5. Inclusion caps:
   For each category cat, let IncludedCat be included deltas whose cat(d)=cat. The checkpoint MUST include at most CAP[cat] deltas per category. If any exceeds, fail.

6. Boundary statistics:
   For each cat, compute kthZ_cat as in Section 14.2 from IncludedCat and verify it equals meta.kthZ_cat. Compute meta.count_* and verify. Compute surplus_sum_clip and verify.

7. delta_keys_commit and delta_nf_commit:
   Compute KΔ_e as the set of keys touched by included deltas after delta normalization consolidation (Section 9.3). Sort unique and verify equal to P.delta_keys. Verify q.delta_keys_commit equals MerkleRoot(P.delta_keys) using the Merkle rules.

Compute for each key k in KΔ_e the aggregated epoch delta value Δk_e as the join of all lifted values for that key across all included deltas, where Lift uses ctx for each op (Section 11) and join uses ⊔_k. Encode each Δk_e as canonical value bytes and verify equal to P.delta_nf entries. Verify q.delta_nf_commit equals MerkleRoot(P.delta_nf) where leaves are H("mleaf"||k||val_hash(delta_value_bytes)).

8. State transition correctness on touched keys:
   For each key k in P.delta_keys:
   A. Verify MembershipProof in P.prev_multiproof[k] against q.prev_root, yielding (present_old, old_bytes). If present_old==0 then old_bytes is treated as the canonical default value for that tag, which MUST be the empty value encoding of the tag’s ValueModel.
   B. Decode old_bytes into OldValue under the tag’s ValueModel.
   C. Decode Δk bytes into DeltaValue under the tag’s ValueModel.
   D. NewValue = OldValue ⊔_k DeltaValue.
   E. Encode NewValue to new_bytes and verify new_bytes equals the corresponding entry in P.new_values.
   F. Verify MembershipProof in P.root_multiproof[k] against q.root with present==1 and vbytes==new_bytes.

9. Untouched keys:
   The proof format does not enumerate untouched keys. The verifier accepts that the root is correct if all touched-key updates are correct and the root multiproofs are valid. This is well-defined because the SMT membership proofs fix the committed root.

10. Frontier correctness:
    Using P.included_ids and the parent lists from the included deltas, compute the induced edges (x -> y) if y lists x as a parent and both x and y are in the included set. Define Frontier as included ids with no outgoing edge. Sort and verify it equals P.frontier_ids.

11. System-derived materialization:
    sig_materialized_flag MUST be 1. The checkpoint MUST include deterministic virtual ops defined in Section 16. Specifically, it MUST materialize SIG, CMDIDX, SCHEMAIDX values. These materialized keys MUST be included in P.delta_keys and P.delta_nf as part of the touched key set. The verifier MUST recompute these virtual ops deterministically and include them in Δk aggregation; otherwise verification fails.

If all checks pass, VerifyCheckpointProof returns true.

16. Deterministic derived indices and materialization (authoritative-by-derivation)

16.1 Principle

Derived indices and signals are deterministic functions of (q.epoch, q.prev_root, included deltas, protocol constants). They are not written by DeltaEvents. They are materialized into authoritative state by deterministic virtual ops that are included in every checkpoint and verified by the checkpoint proof.

16.2 CMDIDX materialization

Let ActiveSetSize = INDEX_MAX (protocol constant).

For each command name hash cmdh (32 bytes), define:
CMD key: tag CMD, key bytes are arbitrary but standard clients SHOULD use k = 0x05 || Trunc248(cmdh).
CMDVOTE key similarly under CMDVOTE.

Votes are derived as:
vote_weight(cmdh) = max w(ctx) across included CMDVOTE ops for cmdh in epoch e, and also across historical state at that key via join (MAX_BY or SCALAR MAX model). In v0.0.1, CMDVOTE ValueModel MUST be MAX_BY with record = (weight=u16, tie32, src_id) and join is max lexicographic on (weight, tie32).

Define CMDIDX value as TOPK_SET(ActiveSetSize) over records:
(cmdh:32, weight:u16, tie32:32)
ordered by (weight desc, tie32 asc).

The checkpoint MUST insert a virtual op that updates key TAG=CMDIDX for epoch e with the computed TOPK_SET record set.

16.3 SCHEMAIDX materialization

Analogous to CMDIDX, using SCHEMAVOTE weight and schema_id hash.

16.4 SIG materialization

For each included delta id x in epoch e:

v(x) = H("sigv" || x) interpreted as a 256-bit vertex
ord(x) = low64(H("sigo" || x)) as U64
mass(x) = surplus_clip(x,e) as non-negative int64

Define SigWindow(v) as the multiset of (ord(x), x) for all included x with v(x)=v.

Define b(v) as the sum of mass(x) over the WINDOW_MAX smallest entries of SigWindow(v) by (ord, x).

The checkpoint MUST materialize SIG by writing, for every vertex v that appears as v(x) for some included x, a key k_sig(v) under tag SIG (0xE0) whose value is MAX_BY with record = (b(v):i64, tie32=H("sig"||v)[:32]) and order_key based on b(v) then tie32.

Standard key derivation for k_sig(v):
k_sig(v) = 0xE0 || Trunc248(H("sigk" || U32LE(e) || v))

This key derivation is normative for SIG and MUST be used by the checkpoint materialization. DeltaEvents MUST NOT write SIG.

16.5 Signal diffusion (client-side, derived)

Given root_star(e), clients MAY compute diffusion potential λ_hat using b(v) values read from SIG keys and the neighborhood function N_i, using the fixed parameters DEGREE_D, SIG_T, SIG_RHO.

17. Networking and relay

17.1 Relay policy

Nodes MUST relay accepted CheckpointEvents subject to local rate limits.

Nodes SHOULD relay accepted DeltaEvents subject to local resource controls. A node MUST be able to validate and decide relay using only the delta bytes plus known root_star(e-1), TR_{e-1}, and threshold parameters.

17.2 Bounded candidate buffers (recommended)

Nodes SHOULD maintain per-category buffers of size at most CAP[cat] (or a small multiple) keyed by (z desc, tie32 asc), retaining the best tickets and dropping worse ones. This affects relay behavior only, not correctness.

18. Security considerations (normative consequences)

18.1 No encoding ambiguity

All hashes, signatures, costs, and IDs depend on canonical bytes defined in Section 6 and Section 9. Any implementation that uses alternative encoding is non-compliant.

18.2 No tag recovery ambiguity

Category, value model, and all validation semantics are derived from tag_id(k)=k[0] and the TagSpec table. Tag recovery from strings or hashing is forbidden.

18.3 No user-controlled scalar dominance in bounded joins

All bounded selection is by fixed total order keys that must end in tie32 derived from ctx, preventing comparison ambiguity. Payload fields MAY exist but MUST be normalized and MUST NOT be the primary determinant of retention unless bounded by CapRule and dominated by work-derived keys.

18.4 Grinding surface control

ticket binds to prev_root and TR_{e-1}. Score is a lexicographic vector with clipped surplus and boundary stats. CP_WINDOW bounds reorg time by rule.

19. Compliance requirements

A v0.0.1 compliant implementation MUST:

A. Implement canonical encoding and normalization exactly (Sections 6 and 9).
B. Implement SHA-256, Ed25519, SMT256 proofs exactly (Sections 5 and 15).
C. Implement AcceptDelta and AcceptCheckpoint exactly (Section 13).
D. Implement canonical checkpoint selection and transcript computation exactly (Sections 12 and 14).
E. Implement threshold update exactly (Section 20).
F. Reject any DeltaEvent op whose tag is system-derived (0xE0..0xE2) or reserved.
G. Materialize derived indices and signal via deterministic virtual ops in every checkpoint and verify them (Section 16).

20. Threshold update rule

Genesis: T_0[cat] = T0[cat].

For e ≥ 0, after root_star(e) is established, compute:

T_{e+1}[cat] = clamp( T_e[cat] + sign(meta.kthZ_cat - TargetZ[cat]), TMIN[cat], TMAX[cat] )

sign(x) = +1 if x>0, 0 if x=0, -1 if x<0, where subtraction is in signed integers.

21. Versioning

v0.0.1 is identified by:

A. Protocol constants (Section 4)
B. Canonical encoding and normalization rules (Sections 6 and 9)
C. Hash and signature schemes (Section 5)
D. TagSpec table including tag IDs, categories, value models, shapes, normalization, caps, auth, lift (Section 7)
E. Admission rules (Sections 10–13)
F. Root structure and proof formats (Section 15)
G. Canonical selection and scoring (Section 14)
H. Threshold update rule (Section 20)
I. Derived materialization rules (Section 16)

Any change to any of the above constitutes a new protocol version.

Appendix A. TagSpec normative payload shapes and rules (mandatory)

This appendix is normative and completes TagSpec. All encodings are in canonical bytes as defined in Section 6.

A.1 OBJ (0x01)

Category: CAT_DATA
ValueModel: MAX_BY with a single admissible record
PayloadShape:
payload = B32(obj_id) || B32(blob_hash)

Normalizer: identity
CapRule: requires obj_id == blob_hash
AuthRule: always true
Lift: record = (obj_id, blob_hash), order_key = (0, H("obj"||obj_id)[:32])
Join semantics: since CapRule forces obj_id==blob_hash, all admissible records for a given key MUST be byte-identical; if not, checkpoint verification fails.

MaxValueBytes: 1 + 64

A.2 LOG (0x02)

Category: CAT_DATA
ValueModel: TOPK_SET(WINDOW_MAX)
PayloadShape:
payload = U64LE(ord_hint) || BYTES(item) where len(item) ≤ 64

Normalizer: ord_hint unchanged; item unchanged
CapRule: len(item) ≤ 64
AuthRule: always true
Lift: record = (ord=ord_hint, src=id(d), tie=tie32(ctx), item=item)
order_key = (w(ctx) desc, tie asc, ord asc, H(item) asc)
MaxValueBytes: bounded by WINDOW_MAX * (8+32+32+2+64) + overhead

A.3 TOP (0x03)

Category: CAT_RANK
ValueModel: TOPK_SET(TOP_MAX)
PayloadShape:
payload = I64LE(score_hint) || BYTES(item) where len(item) ≤ 64

Normalizer: identity
CapRule: len(item) ≤ 64
AuthRule: always true
Lift: record = (score=score_hint, src=id(d), tie=tie32(ctx), item=item)
order_key = (w(ctx) desc, tie asc, score desc, H(item) asc)
MaxValueBytes: bounded by TOP_MAX * (8+32+32+2+64) + overhead

A.4 KEEP (0x04)

Category: CAT_GOV
ValueModel: MAX_BY
PayloadShape: payload = BYTES(obj_id_bytes) with len==32

Normalizer: identity
CapRule: len==32
AuthRule: always true
Lift: record = (weight = min(w(ctx), 65535), tie=tie32(ctx))
order_key = (weight desc, tie asc)
MaxValueBytes: small (fixed)

A.5 CMD (0x05)

Category: CAT_GOV
ValueModel: MAX_BY
PayloadShape:
payload = B32(cmd_name_hash) || B32(obj_id) || BYTES(cost_vec) || B32(schema_id)

Normalizer: identity
CapRule: cost_vec length ≤ 128
AuthRule: always true
Lift: record = (obj_id, cost_vec, schema_id, tie=tie32(ctx))
order_key = (w(ctx) desc, tie asc)
MaxValueBytes: bounded

A.6 CMDVOTE (0x06)

Category: CAT_GOV
ValueModel: MAX_BY
PayloadShape: payload = B32(cmd_name_hash)

Normalizer: identity
CapRule: always true
AuthRule: always true
Lift: record = (weight=min(w(ctx),65535), tie=tie32(ctx), src=id(d))
order_key = (weight desc, tie asc)
MaxValueBytes: fixed small

A.7 SCHEMA (0x07)

Category: CAT_GOV
ValueModel: MAX_BY
PayloadShape: payload = B32(schema_id) || B32(obj_id)

Normalizer: identity
CapRule: always true
AuthRule: always true
Lift: record = (obj_id, tie=tie32(ctx))
order_key = (w(ctx) desc, tie asc)
MaxValueBytes: fixed small

A.8 SCHEMAVOTE (0x08)

Category: CAT_GOV
ValueModel: MAX_BY
PayloadShape: payload = B32(schema_id)

Normalizer: identity
CapRule: always true
AuthRule: always true
Lift: record = (weight=min(w(ctx),65535), tie=tie32(ctx), src=id(d))
order_key = (weight desc, tie asc)
MaxValueBytes: fixed small

A.9 PTR (0x09)

Category: CAT_DATA
ValueModel: TOPK_MAP(CANDIDATE_MAX)
PayloadShape:
payload = B32(scope) || BYTES(name) || U8(mode) || U32LE(until) || B32(ref) || BYTES(aux) where len(aux) ≤ 64

Normalizer: mode MUST be 0 or 1
CapRule: len(name) ≤ 64 and len(aux) ≤ 64
AuthRule: owner = H("pk"||pk) is implicit; always true
Lift: record = (owner=ctx.pk_hash, scope, name_hash=H("n"||name), mode, until, ref, aux_hash=H("a"||aux), tie=tie32(ctx), weight=min(w(ctx),65535))
mkey = owner
best within owner: (mode desc, weight desc, until desc, tie asc)
order_key across owners: (mode desc, weight desc, until desc, tie asc)
MaxValueBytes: bounded by CANDIDATE_MAX

A.10 POS (0x0A)

Category: CAT_DATA
ValueModel: TOPK_MAP(CANDIDATE_MAX)
PayloadShape: payload = BYTES(pos_bytes) where len(pos_bytes) ≤ 64

Normalizer: identity
CapRule: len(pos_bytes) ≤ 64
AuthRule: always true
Lift: record = (owner=ctx.pk_hash, pos_hash=H("pos"||pos_bytes), tie=tie32(ctx), weight=min(w(ctx),65535))
mkey = owner
best: (weight desc, tie asc)
order_key: (weight desc, tie asc)
MaxValueBytes: bounded

A.11 IDX (0x0B)

Category: CAT_DATA
ValueModel: TOPK_MAP(CANDIDATE_MAX)
PayloadShape: payload = B32(vertex) || U32LE(until) || BYTES(endpoint) where len(endpoint) ≤ 128

Normalizer: identity
CapRule: len(endpoint) ≤ 128
AuthRule: always true
Lift: record = (owner=ctx.pk_hash, vertex, until, ep_hash=H("ep"||endpoint), tie=tie32(ctx), weight=min(w(ctx),65535))
mkey = owner
best: (weight desc, until desc, tie asc)
order_key: (weight desc, until desc, tie asc)
MaxValueBytes: bounded

A.12 CUT (0x0C)

Category: CAT_GOV
ValueModel: TOPK_SET(INDEX_MAX)
PayloadShape: payload = BYTES(cut_payload) len ≤ 128

Normalizer: identity
CapRule: len ≤ 128
AuthRule: always true
Lift: record = (src=id(d), tie=tie32(ctx), payload_hash=H("cut"||cut_payload))
order_key = (w(ctx) desc, tie asc)
MaxValueBytes: bounded

A.13 SIG (0xE0), CMDIDX (0xE1), SCHEMAIDX (0xE2)

These tags are system-derived. DeltaEvents MUST NOT write them; AcceptDelta MUST reject any op with these tag_ids.

Their Lift/Cap/Auth are only used for deterministic virtual ops inside checkpoint verification.

End of Specification v0.0.1
