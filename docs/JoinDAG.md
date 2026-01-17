Title: Canonical Join-DAG with Admission-Controlled Evidence Checkpoints and Materialized Derived Indices
Specification v0.0.1
Status: Draft (semantics frozen; no implementation-defined behavior permitted)

1. Goals

1.1 Primary goals

A. Partial-order world: the base history is a DAG of events with a causal partial order, not a total order.

B. Confluence and idempotence by construction: authoritative state updates are pointwise joins over join-semilattices, so evaluation is order-independent and replay-safe.

C. Constant-time hot-path validation: accepting and relaying high-frequency DeltaEvents is bounded by protocol constants and does not require reading prior state beyond the canonical previous checkpoint root and epoch parameters.

D. Fast verifiable reads: clients verify values using a canonical checkpoint root plus bounded membership and non-membership proofs.

E. Non-regressing extensibility: evolution occurs via bounded join-based indices and governance tags; adding new derived indices MUST NOT increase per-DeltaEvent validation cost.

F. Optimization-closed semantics: admission, bandwidth allocation, and ranking influence are deterministic functions of protocol constants, canonical roots, and event bytes.

G. Mathematical closure: all normative choices (encoding, admission, inclusion caps, canonical selection, fold semantics, derived materialization) are deterministic and leave no implementation-defined degrees of freedom.

1.2 Non-goals

A. Global instant finality for every individual DeltaEvent.

B. Confidentiality/privacy beyond authenticity and integrity.

C. Minimizing total global work across all participants; the protocol bounds per-node protocol work, not global economic cost.

2. Normative language

MUST, MUST NOT, SHOULD, SHOULD NOT, MAY are as in RFC 2119.

3. Notation

3.1 Bitstrings
{0,1}^n denotes n-bit strings.

3.2 Hash
H(x) is SHA-256. Output is 256 bits.

3.3 Concatenation
x || y denotes byte concatenation.

3.4 Integers
All integer fields are fixed-width little-endian unless stated otherwise.

3.5 Leading zeros
clz256(x) is the number of leading zero bits of a 256-bit value x.

3.6 Lex order
For equal-length byte strings, lex order is unsigned byte lexicographic.

4. Protocol identity constants (v0.0.1)

All constants in this section are part of the v0.0.1 identity and MUST be identical across compliant implementations.

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
MAX_OPS = 8
MAX_EVENT_BYTES = 2048

4.3 Epoch and inclusion limits

EPOCH_GENESIS = 0

CATEGORY_COUNT = 4

CAT_DATA = 0
CAT_RANK = 1
CAT_GOV = 2
CAT_RESERVED = 3

Targets and slack (fixed):

TARGET[CAT_DATA] = 24
SLACK[CAT_DATA] = 8
CAP[CAT_DATA] = 32

TARGET[CAT_RANK] = 12
SLACK[CAT_RANK] = 4
CAP[CAT_RANK] = 16

TARGET[CAT_GOV] = 12
SLACK[CAT_GOV] = 4
CAP[CAT_GOV] = 16

TARGET[CAT_RESERVED] = 0
SLACK[CAT_RESERVED] = 0
CAP[CAT_RESERVED] = 0

MAX_INCLUDED = CAP[CAT_DATA] + CAP[CAT_RANK] + CAP[CAT_GOV] = 64

Additional per-tag caps inside CAT_GOV (enforced during checkpoint verification only):

CAP_TX = 8
CAP_VMRES = 4

4.4 Admission thresholds

Threshold bounds:

TMIN[DATA]=0, TMAX[DATA]=32
TMIN[RANK]=0, TMAX[RANK]=32
TMIN[GOV]=0,  TMAX[GOV]=32
TMIN[RES]=0,  TMAX[RES]=0

Genesis thresholds:

T0[DATA]=10
T0[RANK]=12
T0[GOV]=12
T0[RES]=0

Target boundary z-values for feedback control (on z_eff, defined in Section 11.5):

TargetZ[DATA]=16
TargetZ[RANK]=18
TargetZ[GOV]=18
TargetZ[RES]=0

4.5 Cost weights

COST_BASE = 1
COST_PER_OP = 1
COST_PER_256B = 1
COST_HEAVY = 2

4.6 Signal diffusion parameters

SIG_T = 3
SIG_RHO_NUM = 1
SIG_RHO_DEN = 2

Numeric format: signed fixed-point Q32.32 in int64. Division rounds toward zero.

4.7 Score stabilization

SURPLUS_CAP = 8

4.8 Value and proof bounds

MAX_META_BYTES = 128
MAX_PROOF_BYTES = 524288
MAX_CHECKPOINT_BYTES = 589824

VM_PROOF_MAX = 131072
VM_MAX_OUTPUTS = 8
VM_MAX_OUTPUT_BYTES = 256

TX_MAX_ITEMS = 8
TX_MAX_LOCKS = 4
TX_MAX_UNIQS = 4
TX_MAX_CAS = 4

4.9 Genesis transcript constant

TR_MINUS_1 = H("tr_genesis" || B32(0x00..00))

5. Cryptographic primitives

5.1 Hash
H is SHA-256.

5.2 Signatures
Signature scheme is Ed25519.

pk encoding: 32 bytes
sig encoding: 64 bytes

SIGCHECK(pk, msg, sig) returns true iff sig is a valid Ed25519 signature of msg under pk.

6. Canonical encoding

6.1 Primitive encodings

U8(x): 1 byte
U16LE(x): 2 bytes
U32LE(x): 4 bytes
U64LE(x): 8 bytes
I64LE(x): 8 bytes two's complement
B32(x): 32 bytes

BYTES(b): U16LE(len(b)) || b, where 0 <= len(b) <= 65535

VEC_T(xs): U16LE(n) || ENC_T(x1) || ... || ENC_T(xn)

STR(s): BYTES(ASCII(s)), ASCII bytes MUST be in 0x20..0x7E.

6.2 Canonical structure rule
For every structure type in this specification, its canonical byte encoding is exactly the concatenation of its fields encoded in the specified order using the primitives above. No alternative representation is permitted.

6.3 Canonical sorting rule
When a field is specified as "sorted", the sort order MUST be lexicographic on the canonical bytes of items.

6.4 Canonical hash binding
All hashes and signature messages are computed over canonical bytes defined by this section and the event canonicalization rules.

7. Key derivation and tag IDs

7.1 Key space
K = {0,1}^256. Keys are exactly 32 bytes.

7.2 Tag ID embedding
For any key k, tag_id(k) = k[0] (first byte). This is authoritative.

7.3 Truncation
Trunc248(x) is the last 31 bytes of the 32-byte value x.

7.4 KeyDerive
For any tag_id t and canonical parts bytes P:
KeyDerive(t, P) = U8(t) || Trunc248(H("k" || U8(t) || P))

7.5 Reserved tags
Any tag_id not explicitly listed as standard in Section 8.2 MUST be rejected if used in a DeltaEvent op.

8. TagSpec and value models (normative)

8.1 Value model family (closed)

All tags use exactly one of the following value models; each yields a join-semilattice.

Model A: MAX_BY
State is EMPTY or a single record R. Join selects the record with maximal order_key under a complete total order; EMPTY is identity.

Model B: TOPK_SET(K)
State is a set S of records with |S| <= K. Join is TopK_K(S union T) by a complete total order (order_key). Ties MUST be resolved inside order_key.

Model C: TOPK_MAP(K)
State is a set S of records with |S| <= K. Each record has an mkey. BestByKey selects exactly one record per mkey by a complete total order (best_key). Then TopK_K is applied by a complete total order (order_key).

8.2 Standard tags (v0.0.1)

User-writable tags (DeltaEvents MAY write only these):

0x01 OBJ
0x02 LOG
0x03 TOP
0x04 KEEP
0x05 CMD
0x06 CMDVOTE
0x07 SCHEMA
0x08 SCHEMAVOTE
0x09 PTR
0x0A POS
0x0B IDX
0x0C UNIQ
0x0D LOCK
0x0E TX
0x0F TOMB
0x10 VMRES

System-derived tags (DeltaEvents MUST NOT write):

0xE0 SIG
0xE1 CMDIDX
0xE2 SCHEMAIDX
0xE3 SIGIDX
0xE4 TXIDX
0xE5 VMIDX

8.3 TagSpec interface (normative)

Each tag t has:

Parse_t(payload_bytes) -> payload_struct or FAIL
KeyCheck_t(payload_struct) -> key k_expected
Normalize_t(ctx0, payload_struct) -> payload_norm or FAIL
CapRule_t(ctx0, payload_norm) -> bool
AuthRule_t(pk, key, payload_norm) -> bool
Lift_t(ctx1, payload_norm) -> record in X_t
DefaultBytes_t -> canonical empty value bytes
MaxValueBytes_t -> U16 bound

Constraint: Normalize/CapRule/AuthRule MUST depend only on ctx0 (Section 11.1). They MUST NOT depend on id, ticket, z, z_eff, surplus, or any value derived from them.

8.4 Category mapping

Category(t):

OBJ, LOG, PTR, POS, IDX are CAT_DATA
TOP is CAT_RANK
KEEP, CMD, CMDVOTE, SCHEMA, SCHEMAVOTE, UNIQ, LOCK, TX, TOMB, VMRES are CAT_GOV
System-derived tags are CAT_RESERVED

Any DeltaEvent op with CAT_RESERVED MUST be rejected.

9. Events

9.1 DeltaEvent fields

DeltaEvent is transmitted as canonical bytes of:

type_tag: U8(0x01)
epoch: U32LE
parents: VEC(B32)
ops: VEC(Op)
pk: B32
nonce_incl: B32
sig: 64 bytes

Op is:

k: B32
payload: BYTES(payload_bytes)

Total encoded size MUST be <= MAX_EVENT_BYTES.

9.2 CheckpointEvent fields

CheckpointEvent is transmitted as canonical bytes of:

type_tag: U8(0x02)
epoch: U32LE
prev_root: B32
batch_commit: B32
cut_commit: B32
meta: BYTES(meta_bytes)
root: B32
proof: BYTES(proof_bytes)
pk: B32
sig: 64 bytes

Constraints:

len(meta_bytes) <= MAX_META_BYTES
len(proof_bytes) <= MAX_PROOF_BYTES
Total checkpoint bytes <= MAX_CHECKPOINT_BYTES

10. Canonicalization, IDs, and signature messages

10.1 Delta canonicalization

Define CanonDelta(d_raw) -> d or FAIL:

A. Decode d_raw as DeltaEvent structure; if fail, FAIL.
B. parents: sort ascending, deduplicate; if length > MAX_PARENTS, FAIL.
C. ops: decode; for each op, let t = tag_id(k). Require t is a user-writable standard tag. Otherwise FAIL.
D. For each op: payload_struct = Parse_t(payload_bytes). If fail, FAIL.
E. KeyCheck: require KeyCheck_t(payload_struct) == k. Else FAIL.
F. Define ctx0 as in Section 11.1. Compute payload_norm = Normalize_t(ctx0, payload_struct). If fail, FAIL.
G. Require CapRule_t(ctx0, payload_norm) == true. Else FAIL.
H. Require AuthRule_t(pk, k, payload_norm) == true. Else FAIL.
I. Canonical payload bytes MUST be produced by re-encoding payload_norm using the tag's canonical payload encoding. Replace op.payload_bytes with these canonical bytes.
J. ops: sort ascending by k. Reject if any duplicate k exists. Reject if op count is 0 or > MAX_OPS.
K. Re-encode the event with canonicalized parents and ops; ensure size <= MAX_EVENT_BYTES.

CanonDelta returns the canonicalized DeltaEvent d.

10.2 Core digest, Delta ID, and Delta signature message

Let bytes_without_sig(d) be the canonical encoding of d with sig field omitted (type_tag through nonce_incl).
Define delta_core(d) = H("core" || bytes_without_sig(d)).

id(d) = H("id" || delta_core(d)).

sigmsg_delta(d) = H("sigmsg" || delta_core(d)).

10.3 Checkpoint core digest, ID, and signature message

Let cp_bytes_without_sig be the canonical encoding of the checkpoint with sig field omitted (type_tag through pk).
Define cp_core(cp) = H("cpcore" || cp_bytes_without_sig).

id(cp) = H("id" || cp_core(cp)).
sigmsg_cp(cp) = H("cpsig" || cp_core(cp)).

11. Admission: category, cost, ticket, and ctx

11.1 ctx0 and ctx1 (non-circular)

For any op within a DeltaEvent at epoch e:

ctx0 contains:
e
prev_root = root_star(e-1) for e>0, else B32(0x00..00)
TR_prev = TR_{e-1} for e>0, else TR_MINUS_1
pk (from the DeltaEvent)
pk_hash = H("pk" || pk)
k (the op key)
op_rank = H("oprank" || k || payload_bytes_canonical) interpreted as B32 for tie-breaking only

ctx1 extends ctx0 with:
delta_id = id(d)
cat = cat(d)
cost = C(d)
ticket, z, reqbits, surplus, surplus_clip, z_eff (defined below)

Constraint: TagSpec Normalize/Cap/Auth MUST use only ctx0.

11.2 Delta category

cat(d) is the maximum precedence among categories of its ops' tags under precedence:

CAT_RANK > CAT_GOV > CAT_DATA > CAT_RESERVED.

If any op is CAT_RESERVED, reject.

11.3 Cost

Let bytes(d) be the length of canonical bytes of d including sig.
heavy_count is the number of ops whose tag is TOP (0x03).

C(d) = COST_BASE

* COST_PER_OP * ops_count
* COST_PER_256B * ceil(bytes(d)/256)
* COST_HEAVY * heavy_count

11.4 Threshold parameters

T_e[cat] are the thresholds for epoch e (Section 16). Genesis is Section 4.4.

11.5 Ticket, work bits, and z_eff

ticket(d,e) = H("incl" || U32LE(e) || U8(cat(d)) || ctx0.prev_root || ctx0.TR_prev || id(d) || nonce_incl)

z(d,e) = clz256(ticket(d,e))

reqbits(d,e) = T_e[cat(d)] + ceil_log2(C(d))

ceil_log2(n) is the smallest integer r such that 2^r >= n, n>=1.

workvalid(d,e) holds iff z(d,e) >= reqbits(d,e)

surplus(d,e) = z(d,e) - reqbits(d,e)
surplus_clip(d,e) = min(surplus(d,e), SURPLUS_CAP)

Define z_eff(d,e) = reqbits(d,e) + surplus_clip(d,e).
Equivalently, z_eff(d,e) = min(z(d,e), reqbits(d,e) + SURPLUS_CAP).

11.6 Canonical tie primitives (op_tie256 unified)

Define op_tie256(d,e,op) = H("optie" || ticket(d,e) || id(d) || op.k || H(op.payload_bytes_canonical)).

Define delta_order(d,e) as the complete total order key:
delta_order(d,e) = ( z_eff(d,e) desc, ticket(d,e) asc, id(d) asc )

All bounded selections and deterministic TopK procedures in this specification MUST end in a complete order; delta_order and op_tie256 provide such closure.

12. Acceptance rules

12.1 AcceptDelta

A node MUST accept and MAY relay a DeltaEvent d_raw for epoch e iff:

A. CanonDelta(d_raw) succeeds producing canonical d.
B. SIGCHECK(pk, sigmsg_delta(d), sig) == true.
C. Node knows root_star(e-1) and TR_{e-1} (or genesis for e=0), can compute T_e, and workvalid(d,e) holds.
D. cat(d) is not CAT_RESERVED.

Otherwise it MUST reject. Nodes SHOULD deduplicate by id(d) and MUST NOT relay duplicates.

12.2 AcceptCheckpoint

A node MUST accept and MAY relay a CheckpointEvent cp iff:

A. cp decodes; meta/proof lengths and total size satisfy bounds.
B. SIGCHECK(cp.pk, sigmsg_cp(cp), cp.sig) == true.
C. For e=0, cp.prev_root is all-zero; for e>0, cp.prev_root == root_star(e-1) once root_star(e-1) is known.
D. VerifyCheckpointProof(cp) == true (Section 14).

Otherwise it MUST reject.

13. Commitments

13.1 Merkle root

A Merkle tree uses SHA-256:

leaf = H("mleaf" || leaf_bytes)
parent = H("mnode" || left || right)
If odd count at a level, duplicate the last.

MerkleRoot([]) = H("mempty")

13.2 batch_commit

batch_commit is MerkleRoot of the sorted list of included delta ids (ascending B32). List length MUST be <= MAX_INCLUDED.

13.3 cut_commit

Let B be the included delta id set. For x,y in B, define an induced edge x -> y if y lists x as a parent.
Frontier(B) is the subset of B with no outgoing edge in the induced relation.
cut_commit is MerkleRoot of the sorted list of Frontier(B) ids.

14. Checkpoint proof: SMT256, multi-update, deterministic recomputation

14.1 State root structure (SMT256)

The authoritative state is a sparse Merkle tree over 256-bit keys.

val_hash(vbytes) = H("val" || vbytes)

leaf_hash(k, present, vbytes) =
if present==1 then H("leaf" || k || val_hash(vbytes))
else H("leaf" || k || B32(0x00..00))

node_hash(left, right) = H("node" || left || right)

Bits of key are MSB-first. Root is depth 0. Leaves are depth 256.

Absent keys are representable and verifiable via present==0 proofs.

14.2 Default value bytes

For any key k, its tag is t=tag_id(k). DefaultBytes_t MUST be used as the value bytes when a key is absent in the SMT and needs decoding for join.

14.3 SMT multiproof format (for a set of keys)

Given a sorted unique key list K = [k1..km], an SMT multiproof consists of:

A. keys: VEC(B32, K)
B. leaves: VEC(LeafValue, m items), aligned with keys
LeafValue = U8(present) || BYTES(vbytes)
If present==0 then vbytes MUST be empty. If present==1 then len(vbytes) <= MaxValueBytes(tag_id(k)).
C. siblings: VEC(SibNode, n items)
SibNode = ENC_NODEID(d,p) || B32(hash)

Node identifier encoding:

ENC_NODEID(d,p) = U16LE(d) || BYTES(prefix_bytes)

prefix_bytes is ceil(d/8) bytes containing the first d bits (MSB-first within each byte), and all unused low bits in the last byte MUST be zero.

Canonical order for node identifiers is increasing by (d, prefix_bytes lex).

14.4 Deterministic multiproof verification algorithm

Given root R, key list K, leaf values, and siblings:

1. For each i, compute Li = leaf_hash(ki, present_i, vbytes_i). Insert into a map M keyed by node-id at depth 256 with prefix=ki.
2. For each sibling entry (d,p)->h: insert into a map S keyed by node-id. If an entry already exists with a different hash, FAIL.
3. For depth d from 255 down to 0: for each node-id (d,p) that is a parent of any node-id already in M:
   Let left child id = (d+1,p||0), right child id = (d+1,p||1).
   Let left hash be M[left] if present, else S[left] must exist. Similarly for right.
   Compute parent hash = node_hash(left_hash, right_hash). Insert into M[(d,p)] ensuring no conflicting hash.
4. Require M[(0,empty)] == R.

This algorithm is deterministic and MUST be used.

14.5 Checkpoint proof structure (self-contained)

Checkpoint proof bytes decode as ProofV001:

included: VEC(IncludedDelta, items_sorted_ascending_by_id)
IncludedDelta = B32(delta_id) || BYTES(delta_bytes)

keys: VEC(B32, keys_sorted_unique)
prev_smt: SMTMultiproof over (keys, leaves, siblings) against cp.prev_root

Constraints:
len(included) <= MAX_INCLUDED
included ids MUST be sorted unique
Each delta_bytes length MUST be <= MAX_EVENT_BYTES
keys MUST be sorted unique
prev_smt.keys MUST equal keys exactly

14.6 System-derived materialization (virtual ops)

Each checkpoint MUST include deterministic virtual ops computed from (epoch e, prev_root, included deltas, protocol constants) as follows.

Define MAX_INDEX = 32 (protocol constant).

Phase 1 uses only included deltas (user ops). Phase 2 adds system ops and system effects.

Phase 1 state: S' = S_{e-1} join Δ_user(e).
Phase 2 state: S_e = S' join Δ_sys(e,S').

Δ_sys includes:

A. SIG and SIGIDX (Section 18.4)
B. CMDIDX (Section 18.2)
C. SCHEMAIDX (Section 18.3)
D. TX system effects and TXIDX (Section 18.6)
E. VMRES system effects and VMIDX (Section 18.7)

All system-derived keys MUST appear in the checkpoint touched key set if they are updated (i.e., if their Δ is non-empty under the tag's semilattice).

14.7 VerifyCheckpointProof(cp) (normative)

Let e = cp.epoch. Verification MUST proceed:

1. Decode ProofV001. If decode fails, return false.
2. Extract included ids and bytes from ProofV001.included. For each item:
   Run CanonDelta(delta_bytes) to produce canonical d. Require id(d) equals included delta_id. Require d.epoch == e. Require SIGCHECK(d.pk, sigmsg_delta(d), d.sig) == true. Require workvalid(d,e) == true. If any fails, return false.
   Let IncludedDeltas be the resulting list, ordered by ascending id. Let IncludedIDs be their ids.
3. Verify cp.batch_commit equals MerkleRoot(IncludedIDs).
   Verify cp.cut_commit equals MerkleRoot(Frontier(IncludedIDs)) computed using the parent lists from IncludedDeltas.
4. Enforce inclusion caps: Let IncludedCat[cat] be IncludedDeltas by cat(d). Require |IncludedCat[cat]| <= CAP[cat] for each cat.
   Also enforce per-tag caps inside CAT_GOV: count of included deltas containing a TX op MUST be <= CAP_TX; count of included deltas containing a VMRES op MUST be <= CAP_VMRES.
5. Recompute meta fields and require exact equality with cp.meta (Section 14.8).
6. Compute the touched key set KeysTouched deterministically:
   KeysTouched_user = union of all op.k in IncludedDeltas.
   Compute Phase1 deltas Δ_user (step 9) and Phase1 state S' on demand for bounded keys.
   Compute KeysTouched_sys = union of all keys updated or read by Δ_sys(e,S') as specified in Section 18.
   KeysTouched = sorted unique union of KeysTouched_user and KeysTouched_sys.
   Require ProofV001.keys == KeysTouched exactly. Otherwise return false.
7. Verify prev_smt multiproof against cp.prev_root using Section 14.4. If fails, return false.
   Let OldBytes[k] be the decoded leaf vbytes if present==1, else DefaultBytes_tag(k).
8. Build Δ_user (key -> delta value) from IncludedDeltas:
   For each IncludedDelta d and each op within d: build ctx0 and then ctx1 for that op. payload_norm is already enforced by CanonDelta.
   Compute record = Lift_t(ctx1,payload_norm) and fold into Δ_user[k] using the tag's value model join.
   Omit keys whose delta is the empty element.
9. Compute S' (Phase 1) on KeysTouched_user:
   For each key k in KeysTouched_user: decode OldBytes[k] as OldValue under the tag's value model. Decode Δ_user[k] (or empty) as DeltaValue. NewValue' = OldValue join DeltaValue. Encode to NewBytes'[k].
   For keys not in KeysTouched_user, NewBytes' is OldBytes.
10. Compute Δ_sys and Phase 2 state bytes NewBytes[k] for all KeysTouched per Section 18, using S' (NewBytes') as input.
11. SMT multi-update root recomputation:
    Recompute the new root by replacing leaf hashes for KeysTouched with leaf_hash(k, present=1, NewBytes[k]) and re-hashing upward consistently using the siblings map from prev_smt. The computed root MUST equal cp.root.
12. If all checks pass, return true.

14.8 Meta format (normative)

cp.meta bytes encode:

meta.count_data: U16LE
meta.count_rank: U16LE
meta.count_gov: U16LE
meta.kthZeff_data: U8
meta.kthZeff_rank: U8
meta.kthZeff_gov: U8
meta.surplus_sum_clip: U64LE

Definitions:

count_cat is the number of included deltas with category cat.

kthZeff_cat:
Order IncludedCat[cat] by delta_order(d,e).
If |IncludedCat[cat]| < CAP[cat], kthZeff_cat = 0.
Else kthZeff_cat = z_eff value of the element at position CAP[cat] (1-based) in that order.

surplus_sum_clip is sum over all included deltas of surplus_clip(d,e), computed in mathematical integers; if the sum exceeds 2^64-1 the checkpoint MUST be rejected.

The verifier MUST recompute these values and require exact match.

15. Canonical checkpoint selection and transcript

15.1 Candidate set
For epoch e, Q_e is the set of accepted checkpoints with cp.epoch==e and VerifyCheckpointProof(cp)==true.

15.2 Score vector (MUST be circular-free)

Define cp_tie = H("cp_tie" || U32LE(e) || cp.prev_root || TR_{e-1} || cp.batch_commit || cp.cut_commit || cp.root || H("pk"||cp.pk))
Define inv_tie = bitwise-not(cp_tie) interpreted as 256-bit unsigned.

Score(cp) = lexicographic tuple:
( meta.surplus_sum_clip, meta.kthZeff_rank, meta.kthZeff_gov, meta.kthZeff_data, inv_tie )

15.3 CanonicalCheckpoint
CanonicalCheckpoint(e) is the cp in Q_e with maximal Score(cp). Ties cannot remain due to inv_tie.
Define root_star(e) = CanonicalCheckpoint(e).root.

15.4 Transcript
TR_e = H("tr" || U32LE(e) || root_star(e) || meta_bytes || batch_commit || cut_commit), where meta_bytes and commits are from CanonicalCheckpoint(e).
For e=0, TR_{-1} is TR_MINUS_1.

16. Threshold update rule

Genesis: T_0[cat] = T0[cat].

For e >= 0, after root_star(e) is established:

T_{e+1}[cat] = clamp( T_e[cat] + sign( kthZeff_e[cat] - TargetZ[cat] ), TMIN[cat], TMAX[cat] )

kthZeff_e[cat] is taken from meta of CanonicalCheckpoint(e).
All arithmetic in the sign argument is done in mathematical integers.

sign(x) = +1 if x>0, 0 if x==0, -1 if x<0.

17. Reads

17.1 Read anchor
A client MUST select an epoch e and use root_star(e) as the read anchor.

17.2 Membership proof
A client reads key k by obtaining (present, vbytes, smt_proof) and verifying it against root_star(e) using the deterministic SMT multiproof verification specialized to a single key (equivalent to Section 14.4). Non-membership is present==0. Clients MUST NOT trust unverified values.

17.3 Logical deletion view (TOMB)
If the client also reads tomb_k = KeyDerive(0x0F, B32(k)) and verifies its value, then it MAY apply the TOMB effective rule defined in Appendix A.16 to treat k as logically deleted at epoch e. This does not change the underlying SMT semantics.

18. Derived materialization and indices (deterministic)

18.1 Principle
Derived indices and signal fields are deterministic functions of (e, prev_root, included deltas, protocol constants, and S' on bounded keys). They are not written by DeltaEvents. They are materialized into state during checkpoint evaluation (Phase 2) and therefore are verifiable by membership proofs.

18.2 CMDIDX (system-derived)

CMD candidates are cmdh values (B32).

Candidate set for epoch e:
CmdCand = (all cmdh appearing in included CMD or CMDVOTE ops in epoch e) union (cmdh listed in previous epoch's CMDIDX value, if present)

Vote weight for cmdh is the current value stored at key CMDVOTE(cmdh) in S' (Phase 1 state), interpreted as a u16 weight (Appendix A).

CMDIDX value is TOPK_SET(MAX_INDEX) of records (cmdh, weight, tie):
tie = H("cmdidx"||cmdh)
order_key = (weight desc, tie asc)

CMDIDX key:
k_cmdidx = KeyDerive(0xE1, U32LE(e))

18.3 SCHEMAIDX (system-derived)

Analogous to CMDIDX using schema_id (B32) and SCHEMAVOTE weights.

SCHEMAIDX key:
k_schemaidx = KeyDerive(0xE2, U32LE(e))

18.4 SIG and SIGIDX (system-derived)

For each included delta id x:

v(x) = H("sigv" || x) interpreted as 256-bit vertex
ord(x) = low64(H("sigo" || x)) as u64
mass(x) = surplus_clip(x,e) as non-negative integer

For each vertex v, define SigWindow(v) as the multiset of pairs (ord(x), x) for included x with v(x)=v.

Define b(v) as the sum of mass(x) over the WINDOW_MAX smallest elements of SigWindow(v) by (ord asc, x asc), where WINDOW_MAX = 64.

SIG key for a vertex v in epoch e:
k_sig(e,v) = KeyDerive(0xE0, U32LE(e) || v)

SIG value at k_sig(e,v) is MAX_BY containing record (b:i64, tie:B32) where tie = H("sig"||U32LE(e)||v).
order_key = (b desc, tie asc)

If a vertex v does not appear in SigWindow, SIG key MAY be absent; its b(v) is treated as 0 by clients (non-membership).

SIGIDX is TOPK_SET(MAX_INDEX) over records (v, b, tie) for all v that appear in this epoch, ordered by (b desc, tie asc).
Key: k_sigidx = KeyDerive(0xE3, U32LE(e))

18.5 Client diffusion (derived, non-authoritative)

Given root_star(e), clients MAY compute diffusion potential using b(v) read from SIG keys and the neighborhood function N_i, with fixed parameters DEGREE_D, SIG_T, SIG_RHO.

18.6 TX system effects and TXIDX (system-derived)

18.6.1 TX winner extraction (Phase 2 input)
Let TxKeys be the set of TX op keys appearing in IncludedDeltas (tag 0x0E).
Compute the Phase 1 TX state at those keys under MAX_BY to obtain at most one winning TX record per TX key.

18.6.2 Preconditions (deterministic, evaluated on S')
For a candidate TX record r associated with source delta d_src (the delta that contributed the winning record), define owner_src = Owner(d_src.pk) and lvl_src = min(surplus_clip(d_src,e),255).

Let r.kind be APPLY or CANCEL.

If r.kind == CANCEL: it produces no item effects; it only contributes to TXIDX.

If r.kind == APPLY: it is eligible only if all conditions hold:

A. Locks: for each lock_id in r.locks, the current LOCK value at key KeyDerive(0x0D,B32(lock_id)) in S' exists and has owner == owner_src and until_epoch >= e.
B. Uniq claims: for each uniq_key in r.uniq_keys (each is a UNIQ key), the current UNIQ value in S' exists and has owner == owner_src.
C. CAS: for each (k, expect_val_hash) in r.cas, compute cur = S'(k) value bytes (absent uses DefaultBytes_tag(k)); require val_hash(cur) == expect_val_hash.
D. Tomb: for each target key k in r.items, read tomb_k = KeyDerive(0x0F,B32(k)) in S'. If tomb indicates deleted at epoch e per Appendix A.16, then TX fails.

18.6.3 Conflict resolution and atomicity (wins-all, deterministic)
Let EligibleTX be the set of APPLY TX records passing preconditions.

For each target key k, define BestTX(k) as the eligible TX whose source delta has maximal key:
( lvl_src desc, id(d_src) asc ), and that contains an item for k. If no eligible TX writes k, BestTX(k) is none.

A TX is Accepted iff for every item key k it writes, BestTX(k) equals that TX. This is a deterministic wins-all rule that forbids partial commit.

18.6.4 Application
For each Accepted TX, apply each item (k, payload_bytes_canonical) as a virtual op to key k:
Build ctx0_item and ctx1_item for the source delta d_src, with op.k = k and payload_bytes_canonical as specified.
Compute record = Lift_tag(k)(ctx1_item, payload_norm_item) and join it into the Phase 2 delta for k under that tag's value model.

18.6.5 TXIDX
TXIDX is TOPK_SET(MAX_INDEX) of records (tx_key, status, lvl, src_id, tie), ordered by (lvl desc, src_id asc, tie asc), where status encodes APPLY_ACCEPTED, APPLY_REJECTED, or CANCEL.
tx_key is the TX key. src_id is id(d_src). tie = H("txidx"||tx_key||src_id).
Key: k_txidx = KeyDerive(0xE4, U32LE(e)).

18.6.6 KeysTouched_sys contributions for TX
KeysTouched_sys MUST include, for each winning TX record r (even if later rejected by preconditions):
A. All referenced LOCK keys.
B. All referenced UNIQ keys.
C. All referenced CAS keys.
D. All referenced item target keys.
E. All referenced TOMB keys for item target keys.
F. k_txidx for epoch e.

18.7 VMRES system effects and VMIDX (system-derived)

18.7.1 VMRES winner extraction
Let VmKeys be VMRES keys in IncludedDeltas (tag 0x10). Under MAX_BY, obtain at most one winning VMRES record per VMRES key.

18.7.2 Output commit and bounded verification
Each VMRES record contains: program_hash, input_commit, output_commit, proof_bytes, outputs.

The verifier MUST check:
A. len(proof_bytes) <= VM_PROOF_MAX and |outputs| <= VM_MAX_OUTPUTS and each output payload length <= VM_MAX_OUTPUT_BYTES.
B. output_commit equals MerkleRoot of the sorted list of H("vmout"||k||payload_bytes_canonical) for outputs.
C. VerifyVMProof(program_hash, input_commit, output_commit, proof_bytes) == true, where VerifyVMProof is a deterministic bounded-time verifier fixed by this specification. If this check fails, the VMRES produces no outputs.

This specification defines VerifyVMProof as a pluggable but identity-bound verifier whose exact algorithm and parameters MUST be included in v0.0.1 Appendix B. If Appendix B is absent, VMRES MUST be treated as always failing verification. (This avoids implementation-defined behavior.)

18.7.3 Application with tomb filter
For each verified VMRES record, for each output (k, payload_bytes_canonical):
If TOMB(k) indicates deleted at epoch e, ignore that output; otherwise apply it as a virtual op using Lift_tag(k) with ctx1 derived from the source delta of the VMRES.

18.7.4 VMIDX
VMIDX is TOPK_SET(MAX_INDEX) of records (job_key, program_hash, lvl, src_id, tie) ordered by (lvl desc, src_id asc, tie asc).
job_key is the VMRES key. tie = H("vmidx"||job_key||src_id).
Key: k_vmidx = KeyDerive(0xE5, U32LE(e)).

18.7.5 KeysTouched_sys contributions for VMRES
KeysTouched_sys MUST include, for each winning VMRES record:
A. All output target keys.
B. All referenced TOMB keys for output target keys.
C. k_vmidx for epoch e.

19. Networking and relay

19.1 Relay policy
Nodes MUST relay accepted CheckpointEvents subject to local rate limits.
Nodes SHOULD relay accepted DeltaEvents subject to local resource controls. Nodes MUST be able to validate and decide relay using only the delta bytes plus known (root_star(e-1), TR_{e-1}, thresholds).

19.2 Bounded candidate buffers (recommended)
Nodes SHOULD maintain per-category buffers of size at most CAP[cat] (or a small multiple) keyed by delta_order(d,e), retaining the best deltas and dropping worse ones. This affects relay behavior only, not correctness.

20. Security considerations (normative consequences)

20.1 No encoding ambiguity
All IDs, tickets, costs, and signature messages depend on canonical bytes, and CanonDelta re-encodes payloads canonically. Any alternative encoding is non-compliant.

20.2 No tag recovery ambiguity
Category, parsing, key checks, normalization, caps, auth, lifting, and value bounds are derived from tag_id(k)=k[0] and fixed TagSpec rules. Tag recovery from strings is forbidden.

20.3 No circular definitions
TagSpec Normalize/Cap/Auth depend only on ctx0. id/ticket/z/surplus appear only in ctx1 and are used only in Lift and derived computations after canonicalization.

20.4 Checkpoint transition correctness is enforced
The checkpoint proof binds the included set and provides an SMT multiproof sufficient to reconstruct prev_root and deterministically recompute the post-update root. Untouched keys cannot be altered without breaking recomputation equality.

20.5 DoS surfaces are bounded by constants
All per-event and per-checkpoint work is bounded by MAX_* constants and CAP_* caps. Nodes MAY apply local relay/storage rate limits, but MUST NOT change verification semantics.

21. Compliance requirements

A v0.0.1 compliant implementation MUST:

A. Implement CanonDelta, core digests, IDs, and signature messages exactly (Sections 10–12).
B. Implement SHA-256, Ed25519, MerkleRoot, and SMT256 exactly (Sections 5, 13, 14).
C. Implement VerifyCheckpointProof exactly, including Phase 1/2 semantics and root recomputation (Section 14) and derived semantics (Section 18).
D. Implement canonical checkpoint selection and transcript exactly (Section 15).
E. Implement threshold update exactly (Section 16).
F. Reject any DeltaEvent op with a non-standard or system-derived tag (Section 8.2).
G. Enforce all explicit bounds (Sections 4, 9, 14, 18).

Appendix A. TagSpec payloads, key checks, value models, and defaults (normative)

Common helper:
Owner(pk) = H("pk" || pk) (32 bytes)

A.1 OBJ (0x01)

Category: DATA
Value model: MAX_BY

Key parts: B32(obj_id)
Key: KeyDerive(0x01, B32(obj_id))

Payload: B32(obj_id) || B32(blob_hash)
Parse: exactly 64 bytes.
KeyCheck: KeyDerive(0x01, B32(obj_id)).
Normalize: identity. CapRule: true. AuthRule: true.

Lift(ctx1):
tie = op_tie256(d,e,op)
record = (blob_hash:B32, tie:B32)
order_key = (bitwise-not(blob_hash) asc, tie asc)

State value encoding: present==0 means absent. If present, vbytes = B32(blob_hash).
DefaultBytes: empty (absent treated as no blob).
MaxValueBytes: 32.

A.2 LOG (0x02)

Category: DATA
Value model: TOPK_SET with K=64

Key parts: B32(scope) || B32(topic)
Key: KeyDerive(0x02, parts)

Payload: B32(scope) || B32(topic) || U64LE(ord) || BYTES(item)
Constraint: len(item) <= 64

Parse/KeyCheck accordingly. Normalize: identity. CapRule: len(item)<=64. AuthRule: true.

Lift(ctx1):
weight = min(surplus_clip(d,e), 255)
tie = op_tie256(d,e,op)
src = id(d)
record = (ord:u64, item:bytes<=64, weight:u8, tie:B32, src:B32)
order_key = (weight desc, tie asc, ord asc, H(item) asc, src asc)

State value encoding: VEC of records sorted by order_key best-first, truncated to K.
DefaultBytes: empty vector.
MaxValueBytes: 8192.

A.3 TOP (0x03)

Category: RANK
Value model: TOPK_SET with K=32

Key parts: B32(scope) || B32(metric)
Key: KeyDerive(0x03, parts)

Payload: B32(scope) || B32(metric) || I64LE(score) || BYTES(item)
Constraint: len(item) <= 64

Lift(ctx1):
weight = min(surplus_clip(d,e), 255)
tie = op_tie256(d,e,op)
src = id(d)
record = (score:i64, item:bytes<=64, weight:u8, tie:B32, src:B32)
order_key = (weight desc, tie asc, score desc, H(item) asc, src asc)

DefaultBytes: empty vector.
MaxValueBytes: 8192.

A.4 KEEP (0x04)

Category: GOV
Value model: MAX_BY

Key parts: B32(obj_id)
Key: KeyDerive(0x04, B32(obj_id))

Payload: B32(obj_id)

Lift(ctx1):
weight = min(surplus_clip(d,e), 255)
tie = op_tie256(d,e,op)
record = (weight:u8, tie:B32)
order_key = (weight desc, tie asc)

State value encoding: vbytes = U8(weight)
DefaultBytes: U8(0)
MaxValueBytes: 1

A.5 CMD (0x05)

Category: GOV
Value model: MAX_BY

Key parts: B32(cmdh)
Key: KeyDerive(0x05, B32(cmdh))

Payload: B32(cmdh) || B32(obj_id) || B32(schema_id) || BYTES(cost_vec)
Constraint: len(cost_vec) <= 64

Lift(ctx1):
weight = min(surplus_clip(d,e), 255)
tie = op_tie256(d,e,op)
record = (obj_id:B32, schema_id:B32, cost_vec:bytes, weight:u8, tie:B32)
order_key = (weight desc, tie asc)

State value encoding: vbytes = B32(obj_id) || B32(schema_id) || BYTES(cost_vec)
DefaultBytes: empty (absent means undeclared).
MaxValueBytes: 130.

A.6 CMDVOTE (0x06)

Category: GOV
Value model: MAX_BY

Key parts: B32(cmdh)
Key: KeyDerive(0x06, B32(cmdh))

Payload: B32(cmdh)

Lift(ctx1):
weight = min(surplus_clip(d,e), 255)
tie = op_tie256(d,e,op)
record = (weight:u8, tie:B32)
order_key = (weight desc, tie asc)

State value encoding: vbytes = U8(weight)
DefaultBytes: U8(0)
MaxValueBytes: 1

A.7 SCHEMA (0x07)

Category: GOV
Value model: MAX_BY

Key parts: B32(schema_id)
Key: KeyDerive(0x07, B32(schema_id))

Payload: B32(schema_id) || B32(obj_id)

Lift(ctx1):
weight = min(surplus_clip(d,e), 255)
tie = op_tie256(d,e,op)
record = (obj_id:B32, weight:u8, tie:B32)
order_key = (weight desc, tie asc)

State value encoding: vbytes = B32(obj_id)
DefaultBytes: empty
MaxValueBytes: 32

A.8 SCHEMAVOTE (0x08)

Category: GOV
Value model: MAX_BY

Key parts: B32(schema_id)
Key: KeyDerive(0x08, B32(schema_id))

Payload: B32(schema_id)

Lift(ctx1):
weight = min(surplus_clip(d,e), 255)
tie = op_tie256(d,e,op)
record = (weight:u8, tie:B32)
order_key = (weight desc, tie asc)

State value encoding: vbytes = U8(weight)
DefaultBytes: U8(0)
MaxValueBytes: 1

A.9 PTR (0x09)

Category: DATA
Value model: TOPK_MAP with K=8, mkey=owner

Key parts: B32(scope) || BYTES(name)
Key: KeyDerive(0x09, parts)

Payload: B32(scope) || BYTES(name) || U8(mode) || U32LE(until) || B32(ref) || BYTES(aux)
Constraints: len(name) <= 64, len(aux) <= 64, mode in {0,1}

Lift(ctx1):
owner = Owner(pk)
level = min(surplus_clip(d,e), 255)
tie = op_tie256(d,e,op)
aux_hash = H("aux"||aux)
record = (owner:B32, mode:u8, level:u8, until:u32, ref:B32, aux_hash:B32, tie:B32)
best_key within owner = (mode desc, level desc, until desc, tie asc)
order_key across owners = same tuple

State value encoding: VEC of up to K records sorted best-first.
DefaultBytes: empty vector.
MaxValueBytes: 8192.

A.10 POS (0x0A)

Category: DATA
Value model: TOPK_MAP with K=8, mkey=owner

Key parts: B32(owner)
Key: KeyDerive(0x0A, B32(owner))

Payload: B32(owner) || BYTES(pos)
Constraint: len(pos) <= 64

AuthRule: require owner == Owner(pk).

Lift(ctx1):
level = min(surplus_clip(d,e), 255)
tie = op_tie256(d,e,op)
pos_hash = H("pos"||pos)
record = (owner:B32, level:u8, pos_hash:B32, tie:B32)
best_key = (level desc, tie asc)
order_key = same

DefaultBytes: empty vector.
MaxValueBytes: 4096.

A.11 IDX (0x0B)

Category: DATA
Value model: TOPK_MAP with K=8, mkey=owner

Key parts: B32(vertex)
Key: KeyDerive(0x0B, B32(vertex))

Payload: B32(vertex) || B32(owner) || U32LE(until) || BYTES(endpoint)
Constraint: len(endpoint) <= 128

AuthRule: require owner == Owner(pk).

Lift(ctx1):
level = min(surplus_clip(d,e), 255)
tie = op_tie256(d,e,op)
ep_hash = H("ep"||endpoint)
record = (owner:B32, level:u8, until:u32, ep_hash:B32, tie:B32)
best_key = (level desc, until desc, tie asc)
order_key = same

DefaultBytes: empty vector.
MaxValueBytes: 4096.

A.12 UNIQ (0x0C)

Category: GOV
Value model: MAX_BY

Key parts: B32(namespace) || B32(value_hash)
Key: KeyDerive(0x0C, parts)

Payload: B32(namespace) || B32(value_hash)

Lift(ctx1):
owner = Owner(pk)
lvl = min(surplus_clip(d,e), 255)
src = id(d)
tie = op_tie256(d,e,op)
record = (owner:B32, lvl:u8, src:B32, tie:B32)
order_key = (lvl desc, src asc, tie asc)

State value encoding: vbytes = B32(owner) || U8(lvl) || B32(src)
DefaultBytes: empty (absent means unclaimed)
MaxValueBytes: 65

A.13 LOCK (0x0D)

Category: GOV
Value model: MAX_BY

Key parts: B32(lock_id)
Key: KeyDerive(0x0D, B32(lock_id))

Payload: B32(lock_id) || U32LE(lease_until_epoch)

Lift(ctx1):
owner = Owner(pk)
lvl = min(surplus_clip(d,e), 255)
src = id(d)
tie = op_tie256(d,e,op)
record = (owner:B32, until:u32, lvl:u8, src:B32, tie:B32)
order_key = (lvl desc, until desc, src asc, tie asc)

State value encoding: vbytes = B32(owner) || U32LE(until) || U8(lvl) || B32(src)
DefaultBytes: empty (absent means unlocked)
MaxValueBytes: 69

A.14 TX (0x0E)

Category: GOV
Value model: MAX_BY

Key parts: U32LE(epoch) || B32(tx_id)
Key: KeyDerive(0x0E, parts)

Payload encoding (canonical):
U32LE(epoch) || B32(tx_id) || U8(kind) ||
VEC(B32, locks_sorted_unique) ||
VEC(B32, uniq_keys_sorted_unique) ||
VEC(CAS, cas_sorted_unique_by_key) ||
VEC(TxItem, items_sorted_unique_by_k)

CAS = B32(k) || B32(expect_val_hash)
TxItem = B32(k) || BYTES(item_payload_bytes)

Constraints (CapRule):
epoch MUST equal ctx0.e
kind in {0,1} where 0=APPLY, 1=CANCEL
|locks| <= TX_MAX_LOCKS, |uniq_keys| <= TX_MAX_UNIQS, |cas| <= TX_MAX_CAS, |items| <= TX_MAX_ITEMS
If kind==CANCEL then items MUST be empty
Each TxItem.k MUST be a user-writable standard tag and MUST NOT be CAT_RESERVED
Each item_payload_bytes MUST pass Parse/KeyCheck/Normalize/Cap/Auth for its target tag using ctx0 derived from the outer ctx0 with k=TxItem.k, and MUST be re-emitted in canonical form inside TX.

AuthRule: true (per-item AuthRule enforced during Normalize).

Lift(ctx1):
lvl = min(surplus_clip(d,e), 255)
src = id(d)
tie = op_tie256(d,e,op)
record = (kind:u8, lvl:u8, src:B32, tie:B32, locks, uniq_keys, cas, items)
order_key = (lvl desc, kind desc, src asc, tie asc) where CANCEL > APPLY for kind desc

State value encoding: canonical re-encoding of the record fields excluding order_key
DefaultBytes: empty
MaxValueBytes: 2048 (MUST be enforced)

A.15 TOMB (0x0F)

Category: GOV
Value model: MAX_BY

Key parts: B32(target_k)
Key: KeyDerive(0x0F, B32(target_k))

Payload: B32(target_k) || U8(kind) || U32LE(until_epoch)
kind: 0=DEL, 1=UNDEL

Lift(ctx1):
lvl = min(surplus_clip(d,e), 255)
src = id(d)
tie = op_tie256(d,e,op)
record = (kind:u8, until:u32, lvl:u8, src:B32, tie:B32)
order_key = (lvl desc, kind desc, until desc, src asc, tie asc) where DEL > UNDEL for kind desc

State value encoding: vbytes = U8(kind) || U32LE(until) || U8(lvl) || B32(src)
DefaultBytes: empty
MaxValueBytes: 38

Effective rule (normative):
Let tomb be the decoded TOMB value for target key k at epoch e.
If tomb.kind==DEL and (tomb.until==0xFFFFFFFF or e <= tomb.until) then k is logically deleted at epoch e; otherwise it is not deleted.

A.16 VMRES (0x10)

Category: GOV
Value model: MAX_BY

Key parts: U32LE(epoch) || B32(job_id)
Key: KeyDerive(0x10, parts)

Payload:
U32LE(epoch) || B32(job_id) || B32(program_hash) || B32(input_commit) || B32(output_commit) || BYTES(proof_bytes) || VEC(VMOut, outputs_sorted_unique_by_k)
VMOut = B32(k) || BYTES(payload_bytes_canonical)

Constraints (CapRule):
epoch MUST equal ctx0.e
len(proof_bytes) <= VM_PROOF_MAX
|outputs| <= VM_MAX_OUTPUTS
each output payload length <= VM_MAX_OUTPUT_BYTES
each output MUST pass Parse/KeyCheck/Normalize/Cap/Auth for its target tag using ctx0 derived from the outer ctx0 with k=VMOut.k, and MUST be re-emitted in canonical form
output_commit MUST equal MerkleRoot(sorted list of H("vmout"||k||payload_bytes_canonical))

Lift(ctx1):
lvl = min(surplus_clip(d,e), 255)
src = id(d)
tie = op_tie256(d,e,op)
record = (lvl:u8, src:B32, tie:B32, program_hash, input_commit, output_commit, proof_bytes, outputs)
order_key = (lvl desc, src asc, tie asc)

State value encoding: canonical re-encoding of the record fields excluding order_key
DefaultBytes: empty
MaxValueBytes: 2048 (MUST be enforced)

A.17 System-derived tags

SIG (0xE0), CMDIDX (0xE1), SCHEMAIDX (0xE2), SIGIDX (0xE3), TXIDX (0xE4), VMIDX (0xE5) are not user-writable; CanonDelta MUST reject any op with these tags. Their keys and values are produced only during checkpoint Phase 2 per Section 18 and MUST be included in KeysTouched when updated.

Appendix B. VerifyVMProof (normative, v0.0.1 identity)

This appendix MUST define a deterministic bounded-time verifier VerifyVMProof(program_hash, input_commit, output_commit, proof_bytes). If this appendix is not present in an implementation, VMRES verification MUST always fail.

End of Specification v0.0.1
