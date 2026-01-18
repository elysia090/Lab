Title: Canonical Join-DAG with Declaration-Based Reads, Local Modules, and Commitment-Abstraction Checkpoints
Specification v0.01
Status: Draft (semantics frozen; no implementation-defined behavior permitted)
	1.	Goals

1.1 Primary goals

A. Partial-order world: the base history is a DAG of events with a causal partial order (links), while authoritative state evolution is confluent and order-independent.

B. Confluence and idempotence by construction: authoritative updates are pointwise joins over join-semilattices, so evaluation is replay-safe and order-independent.

C. Constant-time hot-path validation: accepting and relaying high-frequency DeltaEvents is bounded by protocol constants and does not require reading prior state beyond known epoch anchors (prev state root and transcript).

D. Declaration-based reads: all state reads performed by protocol-defined evaluation MUST be declared in advance by the event/module footprint and MUST be bounded by constants.

E. Module locality: derived indices and system effects are computed by a fixed set of local modules whose read/write footprints are state-independent (depend only on epoch, body bytes, and constants), enabling deterministic closure and bounded proofs.

F. Commitment abstraction: the state commitment is specified by a closed, identity-bound commitment scheme instance, while the core semantics are expressed over an abstract key-value state.

G. Non-regressing extensibility: adding a new module in a future version MUST NOT increase per-DeltaEvent admission cost; it MAY increase per-checkpoint evaluation cost only by bounded constants.

H. Optimization-closed semantics: admission, ranking, and selection are deterministic functions of protocol constants, epoch anchors, and canonical bytes.

1.2 Non-goals

A. Global instant finality for every individual DeltaEvent.

B. Confidentiality/privacy beyond authenticity and integrity.

C. Minimizing total global work across all participants; the protocol bounds per-node protocol work, not global economic cost.

D. Implementation-defined pluggable verification logic. v0.01 is closed.
	2.	Normative language

MUST, MUST NOT, SHOULD, SHOULD NOT, MAY are as in RFC 2119.
	3.	Notation

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

3.7 Sets and sorting
All sets serialized on the wire MUST be sorted by canonical bytes and MUST be unique.
	4.	Protocol identity constants (v0.01)

All constants in this section are part of the v0.01 identity and MUST be identical across compliant implementations.

4.1 Address space and neighborhood (optional, non-consensus utility)

SPACE_BITS = 256
DEGREE_D = 8
NEIGHBOR_CONST[i] = H(“nbr” || U32LE(i)) for i in 0..DEGREE_D-1

NEIGHBOR_PERM is a public permutation P: {0,1}^256 -> {0,1}^256 defined as an 8-round Feistel network:

Let v = L || R where L,R are 128-bit.
For r = 0..7:
F = Trunc128(H(“F” || U32LE(r) || R))
(L, R) = (R, L xor F)
Then P(v) = L || R.

Neighbor function:
N_i(v) = P(v xor NEIGHBOR_CONST[i])

4.2 Size limits

MAX_LINKS = 8
MAX_READS = 32
MAX_OPS = 8
MAX_EVENT_BYTES = 2048

MAX_BODY_DELTAS = 64

MAX_META_BYTES = 64

MAX_BUNDLE_BYTES = 1048576
MAX_AUDIT_BYTES = 1048576

4.3 Epoch and categories

EPOCH_GENESIS = 0

CATEGORY_COUNT = 4

CAT_DATA = 0
CAT_RANK = 1
CAT_GOV = 2
CAT_RESERVED = 3

4.4 Inclusion caps (per checkpoint)

CAP[CAT_DATA] = 32
CAP[CAT_RANK] = 16
CAP[CAT_GOV] = 16
CAP[CAT_RESERVED] = 0

MAX_INCLUDED = 64

Additional per-tag caps inside CAT_GOV (enforced during checkpoint verification):

CAP_TX = 8

4.5 Admission thresholds

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

Target boundary z-values for feedback control (on z_eff, defined in Section 11.6):

TargetZ[DATA]=16
TargetZ[RANK]=18
TargetZ[GOV]=18
TargetZ[RES]=0

4.6 Cost weights

COST_BASE = 1
COST_PER_OP = 1
COST_PER_256B = 1
COST_HEAVY = 2

4.7 Surplus stabilization

SURPLUS_CAP = 8

4.8 Commitment instance

STATE_COMMITMENT = SMT256 over SHA-256 as specified in Appendix C.
All state audit proofs and membership/non-membership proofs MUST use this instance in v0.01.

4.9 Genesis transcript constant

TR_MINUS_1 = H(“tr_genesis” || B32(0x00..00))
	5.	Cryptographic primitives

5.1 Hash
H is SHA-256.

5.2 Signatures
Signature scheme is Ed25519.

pk encoding: 32 bytes
sig encoding: 64 bytes

SIGCHECK(pk, msg, sig) returns true iff sig is a valid Ed25519 signature of msg under pk.
	6.	Canonical encoding

6.1 Primitive encodings

U8(x): 1 byte
U16LE(x): 2 bytes
U32LE(x): 4 bytes
U64LE(x): 8 bytes
I64LE(x): 8 bytes two’s complement
B32(x): 32 bytes

BYTES(b): U16LE(len(b)) || b, where 0 <= len(b) <= 65535

VEC_T(xs): U16LE(n) || ENC_T(x1) || … || ENC_T(xn)

STR(s): BYTES(ASCII(s)), ASCII bytes MUST be in 0x20..0x7E.

6.2 Canonical structure rule
For every structure type in this specification, its canonical byte encoding is exactly the concatenation of its fields encoded in the specified order using the primitives above. No alternative representation is permitted.

6.3 Canonical sorting rule
When a field is specified as “sorted unique”, the order MUST be lexicographic on canonical bytes and duplicates MUST be rejected.

6.4 Canonical hash binding
All hashes and signature messages are computed over canonical bytes defined by this section and the event canonicalization rules.
	7.	Keys, tags, and value models (normative)

7.1 Key space
K = {0,1}^256. Keys are exactly 32 bytes.

7.2 Tag ID embedding
For any key k, tag_id(k) = k[0] (first byte). This is authoritative.

7.3 Truncation
Trunc248(x) is the last 31 bytes of a 32-byte value x.

7.4 KeyDerive
For any tag_id t and canonical parts bytes P:
KeyDerive(t, P) = U8(t) || Trunc248(H(“k” || U8(t) || P))

7.5 Categories
Category(tag) is fixed by Table in Section 7.8.

7.6 Value model family (closed)

All tags use exactly one of the following value models; each yields a join-semilattice.

Model A: MAX_BY
State is EMPTY or a single record R. Join selects the record with maximal order_key under a complete total order; EMPTY is identity.

Model B: TOPK_SET(K)
State is a set S of records with |S| <= K. Join is TopK_K(S union T) by a complete total order (order_key). Ties MUST be resolved inside order_key.

Model C: TOPK_MAP(K)
State is a set S of records with |S| <= K. Each record has an mkey. BestByKey selects exactly one record per mkey by a complete total order (best_key). Then TopK_K is applied by a complete total order (order_key).

7.7 TagSpec interface (normative)

Each user-writable tag t defines:

Parse_t(payload_bytes) -> payload_struct or FAIL
KeyCheck_t(payload_struct) -> key k_expected
Normalize_t(ctx0, payload_struct) -> payload_norm or FAIL
CapRule_t(ctx0, payload_norm) -> bool
AuthRule_t(pk, key, payload_norm) -> bool
Reads_t(ctx0, payload_norm) -> sorted unique VEC(B32) of keys (may be empty)
Lift_t(ctx1, payload_norm, tie256) -> record in X_t
DefaultBytes_t -> canonical empty value bytes
MaxValueBytes_t -> U16 bound

Constraints:

A. Normalize/CapRule/AuthRule/Reads MUST depend only on ctx0. They MUST NOT depend on delta_id, ticket, z, z_eff, surplus, or any value derived from them.

B. Reads_t MUST include every state key that any protocol-defined evaluation may read as a consequence of this op in v0.01. Reads_t MUST NOT include CAT_RESERVED keys.

C. Lift_t MUST be pure and MUST NOT read state; it may use ctx1 values (ticket, surplus, etc.) and tie256.

7.8 Standard tags (v0.01)

User-writable tags (DeltaEvents MAY write only these):

0x01 OBJ (DATA)
0x02 LOG (DATA)
0x03 TOP (RANK)
0x04 KEEP (GOV)
0x05 CMD (GOV)
0x06 CMDVOTE (GOV)
0x07 SCHEMA (GOV)
0x08 SCHEMAVOTE (GOV)
0x09 PTR (DATA)
0x0A POS (DATA)
0x0B IDX (DATA)
0x0C UNIQ (GOV)
0x0D LOCK (GOV)
0x0E TX (GOV)
0x0F TOMB (GOV)

System-derived tags (user-writable MUST NOT write):

0xE0 SIG (RESERVED)
0xE1 CMDIDX (RESERVED)
0xE2 SCHEMAIDX (RESERVED)
0xE3 SIGIDX (RESERVED)
0xE4 TXIDX (RESERVED)

Any op with CAT_RESERVED MUST be rejected by CanonDelta.
	8.	Events and wire messages

8.1 DeltaEvent (wire)

DeltaEvent is transmitted as canonical bytes of:

type_tag: U8(0x01)
epoch: U32LE
links: VEC(B32) sorted unique (length <= MAX_LINKS)
reads: VEC(B32) sorted unique (length <= MAX_READS)
ops: VEC(Op)
pk: B32
nonce_incl: B32
sig: 64 bytes

Op is:

k: B32
payload: BYTES(payload_bytes)

Total encoded size MUST be <= MAX_EVENT_BYTES.

8.2 CheckpointHeader (wire)

CheckpointHeader is transmitted as canonical bytes of:

type_tag: U8(0x02)
epoch: U32LE
prev_state_root: B32
body_commit: B32
state_root: B32
meta: BYTES(meta_bytes) (len <= MAX_META_BYTES)
pk: B32
sig: 64 bytes

8.3 BodyBundle (wire)

BodyBundle is transmitted as canonical bytes of:

type_tag: U8(0x03)
epoch: U32LE
body_commit: B32
deltas: VEC(DeltaBytes) where each DeltaBytes is BYTES(delta_event_bytes)

Constraints:

A. Total bytes MUST be <= MAX_BUNDLE_BYTES.
B. |deltas| MUST be <= MAX_BODY_DELTAS.
C. Each delta_event_bytes MUST decode as a DeltaEvent structure and MUST be canonical under CanonDelta (Section 9).

8.4 AuditBundle (wire)

AuditBundle is transmitted as canonical bytes of:

type_tag: U8(0x04)
epoch: U32LE
prev_state_root: B32
body_commit: B32
keys_need_old: VEC(B32) sorted unique
state_proof: BYTES(proof_bytes)

Constraints:

A. Total bytes MUST be <= MAX_AUDIT_BYTES.
B. keys_need_old MUST be sorted unique.
C. proof_bytes MUST be a valid SMT256 multiproof against prev_state_root for exactly keys_need_old as specified in Appendix C.
	9.	Canonicalization and delta digest

9.1 CanonDelta(d_raw) -> d or FAIL

A. Decode d_raw as DeltaEvent. If fail, FAIL.
B. Enforce links sorted unique and length <= MAX_LINKS.
C. Decode ops; require 1 <= ops_count <= MAX_OPS.
D. For each op, let t = tag_id(k). Require t is a user-writable standard tag. Otherwise FAIL.
E. For each op: payload_struct = Parse_t(payload_bytes). If fail, FAIL.
F. KeyCheck: require KeyCheck_t(payload_struct) == k. Else FAIL.
G. Define ctx0 as in Section 10.1. Compute payload_norm = Normalize_t(ctx0, payload_struct). If fail, FAIL.
H. Require CapRule_t(ctx0, payload_norm) == true. Else FAIL.
I. Require AuthRule_t(pk, k, payload_norm) == true. Else FAIL.
J. Replace op.payload_bytes with the canonical re-encoding of payload_norm as defined by the tag spec.
K. Sort ops ascending by k. Reject duplicates.
L. Compute the required read set R_req as the sorted unique union over ops of Reads_t(ctx0, payload_norm) for that op.
M. Require that reads field equals R_req exactly. Otherwise FAIL.
N. Re-encode the event with canonical links, reads, and ops; ensure size <= MAX_EVENT_BYTES.

CanonDelta returns the canonicalized DeltaEvent d.

9.2 Delta digest and signature message

Let bytes_without_sig(d) be the canonical encoding of d with sig field omitted (type_tag through nonce_incl).
Define delta_digest(d) = H(“delta” || bytes_without_sig(d)).

Define:

id(d) = delta_digest(d)
sigmsg_delta(d) = delta_digest(d)
	10.	ctx0/ctx1, category, cost, ticket, and ordering

10.1 ctx0 and ctx1

For any op within a DeltaEvent at epoch e:

ctx0 contains:

e
prev_state_root = state_root_star(e-1) for e>0, else B32(0x00..00)
TR_prev = TR_{e-1} for e>0, else TR_MINUS_1
pk (from the DeltaEvent)
pk_hash = H(“pk” || pk)
k (the op key)
op_rank = H(“oprank” || k || payload_bytes_canonical) interpreted as B32 for tie-breaking only

ctx1 extends ctx0 with:

delta_id = id(d)
cat = cat(d)
cost = C(d)
ticket, z, reqbits, surplus, surplus_clip, z_eff (defined below)

10.2 Delta category

cat(d) is the maximum precedence among categories of its ops’ tags under precedence:

CAT_RANK > CAT_GOV > CAT_DATA > CAT_RESERVED.

If any op is CAT_RESERVED, reject.

10.3 Cost

Let bytes(d) be the length of canonical bytes of d including sig.
heavy_count is the number of ops whose tag is TOP (0x03).

C(d) = COST_BASE
	•	COST_PER_OP * ops_count
	•	COST_PER_256B * ceil(bytes(d)/256)
	•	COST_HEAVY * heavy_count

10.4 Threshold parameters

T_e[cat] are the thresholds for epoch e (Section 15). Genesis is Section 4.5.

10.5 Ticket and work bits

ticket(d,e) = H(“incl” || U32LE(e) || U8(cat(d)) || ctx0.prev_state_root || ctx0.TR_prev || id(d) || nonce_incl)

z(d,e) = clz256(ticket(d,e))

reqbits(d,e) = T_e[cat(d)] + ceil_log2(C(d))

ceil_log2(n) is the smallest integer r such that 2^r >= n, n>=1.

workvalid(d,e) holds iff z(d,e) >= reqbits(d,e)

10.6 Surplus and z_eff

surplus(d,e) = z(d,e) - reqbits(d,e)
surplus_clip(d,e) = min(surplus(d,e), SURPLUS_CAP)

Define z_eff(d,e) = reqbits(d,e) + surplus_clip(d,e).
Equivalently, z_eff(d,e) = min(z(d,e), reqbits(d,e) + SURPLUS_CAP).

10.7 Canonical tie primitive

Define op_tie256(d,e,op) = H(“optie” || ticket(d,e) || id(d) || op.k || H(op.payload_bytes_canonical)).

10.8 Delta order key

Define delta_order(d,e) as the complete total order key:

delta_order(d,e) = ( z_eff(d,e) desc, ticket(d,e) asc, id(d) asc )

All deterministic TopK procedures and bounded selections in v0.01 MUST end in a complete order; delta_order and op_tie256 provide such closure.
	11.	Admission rules

11.1 AcceptDelta

A node MUST accept and MAY relay a DeltaEvent d_raw for epoch e iff:

A. CanonDelta(d_raw) succeeds producing canonical d.
B. SIGCHECK(pk, sigmsg_delta(d), sig) == true.
C. Node knows state_root_star(e-1) and TR_{e-1} (or genesis for e=0), can compute T_e, and workvalid(d,e) holds.
D. cat(d) is not CAT_RESERVED.

Otherwise it MUST reject. Nodes SHOULD deduplicate by id(d) and MUST NOT relay duplicates.

11.2 Receipt set (local, normative for optimization)

A node MAY maintain a local Receipt_e set of delta ids accepted by AcceptDelta for epoch e.

Receipt_e is not a wire object. It is a local cache that MAY be used to avoid re-verifying signature/work for deltas whose id is already in Receipt_e. Using Receipt_e MUST NOT change the acceptance semantics of CheckpointHeader validity as defined in Section 13.
	12.	Body commitment

12.1 Body leaf and root

For a canonical DeltaEvent d, define delta_bytes(d) as its full canonical bytes including sig.

Define leaf(d) = H(“bleaf” || delta_bytes(d)).

Given a list of included deltas B = [d1..dn], define BodyCommit(B) as:

A. Compute leaves L = [leaf(di)] for all di.
B. Sort L ascending lex and reject duplicates.
C. body_commit = MerkleRoot(L) using SHA-256:

leaf_node = H(“mleaf” || leaf_bytes)
parent = H(“mnode” || left || right)
If odd count at a level, duplicate the last.
MerkleRoot([]) = H(“mempty”)

12.2 Canonical BodyBundle ordering

A BodyBundle MUST include deltas such that, when parsed into canonical deltas, their leaf list equals exactly the leaves used in body_commit, and the bundle deltas MUST be ordered by ascending leaf (lex on B32 leaf values). This is a canonical ordering rule.
	13.	Checkpoints: validity, evaluation, and selection

13.1 Meta format (derived, normative)

meta_bytes encode:

meta.count_data: U16LE
meta.count_rank: U16LE
meta.count_gov: U16LE
meta.kthZeff_data: U8
meta.kthZeff_rank: U8
meta.kthZeff_gov: U8
meta.surplus_sum_clip: U64LE

Definitions (for epoch e and body B):

count_cat is the number of included deltas with category cat.

Let IncludedCat[cat] be included deltas filtered by cat(d). Sort IncludedCat[cat] by delta_order(d,e) best-first.

If |IncludedCat[cat]| < CAP[cat], kthZeff_cat = 0.
Else kthZeff_cat = z_eff value of the element at position CAP[cat] (1-based) in that order.

surplus_sum_clip is sum over all included deltas of surplus_clip(d,e), computed in mathematical integers; if the sum exceeds 2^64-1 the checkpoint is invalid.

13.2 CheckpointHeader digest and signature message

Let hdr_bytes_without_sig be the canonical encoding of the header with sig omitted (type_tag through pk).
Define hdr_digest = H(“hdr” || hdr_bytes_without_sig).
sigmsg_hdr = hdr_digest.

13.3 AcceptCheckpointHeader (syntax-only)

A node MUST accept and MAY relay a CheckpointHeader hdr iff:

A. hdr decodes and meta length <= MAX_META_BYTES.
B. SIGCHECK(hdr.pk, sigmsg_hdr(hdr), hdr.sig) == true.
C. For e=0, hdr.prev_state_root is all-zero; for e>0, hdr.prev_state_root == state_root_star(e-1) once known.

This is a syntax-and-link check only. Full validity requires verification with the corresponding bundles (Section 13.4).

13.4 VerifyCheckpoint (full validity)

A checkpoint for epoch e consists of:
	•	CheckpointHeader hdr
	•	BodyBundle body (with matching epoch and body_commit)
	•	Either:
	•	Stateful evaluation using a locally stored state S_{e-1}, or
	•	Stateless audit using AuditBundle audit

A node MUST consider hdr fully valid iff all conditions below hold.

13.4.1 Body verification

A. body.epoch == hdr.epoch == e.
B. body.body_commit == hdr.body_commit.
C. Parse all body.deltas and canonicalize with CanonDelta, producing canonical deltas B ordered by ascending leaf. Require bundle ordering is correct and leaves are unique.
D. Require BodyCommit(B) == hdr.body_commit.

13.4.2 Delta validity and caps

A. For each d in B: require d.epoch == e.
B. Require SIGCHECK(d.pk, sigmsg_delta(d), d.sig) == true.
C. Require workvalid(d,e) == true.
D. Enforce inclusion caps by category: |IncludedCat[cat]| <= CAP[cat].
E. Enforce per-tag caps in GOV: the number of included deltas containing a TX op MUST be <= CAP_TX.

13.4.3 KeysNeedOld (declaration-based)

Define per-delta write keys:

W_delta(d) = { op.k | op in d.ops }

Define per-delta read keys:

R_delta(d) = d.reads (already enforced equal to union of Reads_t over ops)

Let Modules be the fixed module set in Appendix B. For each module M, define its declared footprint:

(R_M, W_M) = Footprint_M(e, B, const)

Footprint_M MUST depend only on (e, B, const), not on any state bytes.

Define:

R_body = union over d in B of R_delta(d) union union over M of R_M
W_body = union over d in B of W_delta(d) union union over M of W_M

KeysNeedOld = sorted unique union of (R_body union W_body).

13.4.4 Audit precondition for stateless verification

If verifying statelessly, require:

A. audit.epoch == e, audit.prev_state_root == hdr.prev_state_root, audit.body_commit == hdr.body_commit.
B. audit.keys_need_old equals KeysNeedOld exactly.
C. audit.state_proof is a valid SMT256 multiproof opening exactly KeysNeedOld against hdr.prev_state_root (Appendix C), producing OldBytes[k] for all keys.

If verifying statefully, the node MUST have OldBytes[k] for all keys in KeysNeedOld from its local S_{e-1}.

13.4.5 Phase 1: user delta fold

For each included delta d and each op in d.ops:

A. Build ctx0 and ctx1 for that op. ctx1 values (ticket, surplus, z_eff, etc.) use e and hdr.prev_state_root and TR_{e-1}.
B. Compute tie256 = op_tie256(d,e,op).
C. Compute record = Lift_t(ctx1, payload_norm, tie256).
D. Join record into Δ_user[op.k] under that tag’s value model join. Omit keys whose delta equals the empty element.

Then compute Phase 1 intermediate state:

S’(k) = Join( Decode(OldBytes[k]), Δ_user[k] ) for keys where applicable, else Decode(OldBytes[k]).
Encode S’(k) back to bytes NewBytes1[k] as needed by modules.

Phase 1 MUST NOT read any key outside KeysNeedOld.

13.4.6 Phase 2: module evaluation

For each module M in Modules:

A. Let (R_M, W_M) be its footprint.
B. Provide the module the projection S’|{R_M} using the Phase 1 bytes for those keys (using OldBytes for keys not written in Phase 1).
C. Compute Δ_M = Eval_M(e, B, S’|{R_M}, const).
D. Require dom(Δ_M) ⊆ W_M.

Combine all module deltas by key-wise join:

Δ_sys = ⊔_M Δ_M

Compute final bytes for keys in W_body:

NewBytes[k] = Encode( Join( Decode(OldBytes[k]), Δ_user[k], Δ_sys[k] ) ) as defined by tag value models, with omitted deltas treated as empty.

13.4.7 State root recomputation

Using SMT256 multi-update recomputation (Appendix C):
	•	Starting from hdr.prev_state_root and the multiproof sibling set (stateless) or from the locally stored SMT (stateful), recompute the new root after updating exactly the keys in W_body to NewBytes[k] with present=1.
	•	The computed root MUST equal hdr.state_root.

13.4.8 Meta verification

Recompute meta_bytes from B and e per Section 13.1 and require exact equality with hdr.meta.

If all checks pass, hdr is fully valid.

13.5 Canonical checkpoint selection and transcript

13.5.1 Candidate set

For epoch e, Q_e is the set of fully valid checkpoint headers for epoch e under Section 13.4 (i.e., headers for which required bundles were obtained and verified).

13.5.2 Score and tie

Define hdr_tie = H(“hdr_tie” || U32LE(e) || hdr.prev_state_root || TR_{e-1} || hdr.body_commit || hdr.state_root || H(“pk”||hdr.pk))
Define inv_tie = bitwise-not(hdr_tie) interpreted as 256-bit unsigned.

Score(hdr) = lexicographic tuple:

( meta.surplus_sum_clip, meta.kthZeff_rank, meta.kthZeff_gov, meta.kthZeff_data, inv_tie )

13.5.3 CanonicalCheckpoint

CanonicalCheckpoint(e) is the hdr in Q_e with maximal Score(hdr). Ties cannot remain due to inv_tie.

Define state_root_star(e) = CanonicalCheckpoint(e).state_root.

13.5.4 Transcript

TR_e = H(“tr” || U32LE(e) || state_root_star(e) || CanonicalCheckpoint(e).meta || CanonicalCheckpoint(e).body_commit)

For e=0, TR_{-1} is TR_MINUS_1.
	14.	Client reads (state commitment)

A client selects an epoch e and uses state_root_star(e) as anchor.

Membership/non-membership proofs MUST be SMT256 openings as specified in Appendix C.
Absent keys are representable and verifiable via present==0 openings.
	15.	Threshold update rule

Genesis: T_0[cat] = T0[cat].

For e >= 0, after state_root_star(e) is established:

T_{e+1}[cat] = clamp( T_e[cat] + sign( kthZeff_e[cat] - TargetZ[cat] ), TMIN[cat], TMAX[cat] )

kthZeff_e[cat] is taken from meta of CanonicalCheckpoint(e).
All arithmetic in the sign argument is done in mathematical integers.

sign(x) = +1 if x>0, 0 if x==0, -1 if x<0.
	16.	Networking and availability

16.1 Relay policy

Nodes MUST relay accepted CheckpointHeader messages subject to local rate limits.

Nodes SHOULD relay accepted DeltaEvents subject to local resource controls. Nodes MUST be able to validate and decide relay using only the delta bytes plus known (state_root_star(e-1), TR_{e-1}, thresholds).

16.2 Bundle retrieval

BodyBundle and AuditBundle MAY be fetched on demand. A node MUST NOT treat a CheckpointHeader as fully valid unless it has verified the required bundles per Section 13.4.

16.3 Availability failure handling

If a node cannot obtain bundles for a header, that header is not in Q_e for that node and cannot be selected as canonical by that node. This is an availability constraint and is explicitly in scope.
	17.	Security considerations (normative consequences)

17.1 No encoding ambiguity

All digests, tickets, and signatures depend on canonical bytes. CanonDelta re-emits payloads canonically. Any alternative encoding is non-compliant.

17.2 No circular definitions

Normalize/Cap/Auth/Reads depend only on ctx0. Ticket/z/surplus appear only in ctx1 and are used only after canonicalization.

17.3 Declaration-based reads are enforced

CanonDelta requires reads field equals the union of Reads_t across ops. Modules have declared, state-independent footprints. Any evaluation that would read outside declared footprints is non-compliant.

17.4 DoS surfaces are bounded by constants

All per-event and per-checkpoint work is bounded by MAX_* constants and CAP_* caps. Nodes MAY apply local relay/storage rate limits, but MUST NOT change verification semantics.

Appendix A. TagSpec payloads, key checks, read declarations, value models, and defaults (normative)

Common helper:

Owner(pk) = H(“pk” || pk) (32 bytes)

All value encodings below are canonical and MUST be used.

A.1 OBJ (0x01)

Category: DATA
Value model: MAX_BY

Key parts: B32(obj_id)
Key: KeyDerive(0x01, B32(obj_id))

Payload: B32(obj_id) || B32(blob_hash)

Parse: exactly 64 bytes.
KeyCheck: KeyDerive(0x01, B32(obj_id)).
Normalize: identity.
CapRule: true.
AuthRule: true.
Reads: empty.

Lift(ctx1, payload_norm, tie256):
record = (blob_hash:B32, tie:B32) where tie = tie256
order_key = (bitwise-not(blob_hash) asc, tie asc)

State value bytes: present==0 means absent; if present, vbytes = B32(blob_hash).
DefaultBytes: empty (absent treated as no blob).
MaxValueBytes: 32.

A.2 LOG (0x02)

Category: DATA
Value model: TOPK_SET with K=64

Key parts: B32(scope) || B32(topic)
Key: KeyDerive(0x02, parts)

Payload: B32(scope) || B32(topic) || U64LE(ord) || BYTES(item) with len(item) <= 64

Normalize: identity.
CapRule: len(item) <= 64.
AuthRule: true.
Reads: empty.

Lift:
weight = min(surplus_clip(d,e), 255) using ctx1
record = (ord:u64, item:bytes<=64, weight:u8, tie:B32, src:B32)
src = id(d), tie = tie256
order_key = (weight desc, tie asc, ord asc, H(item) asc, src asc)

DefaultBytes: empty vector.
MaxValueBytes: 8192.

A.3 TOP (0x03)

Category: RANK
Value model: TOPK_SET with K=32

Key parts: B32(scope) || B32(metric)
Key: KeyDerive(0x03, parts)

Payload: B32(scope) || B32(metric) || I64LE(score) || BYTES(item), len(item) <= 64

Normalize: identity.
CapRule: len(item) <= 64.
AuthRule: true.
Reads: empty.

Lift:
weight = min(surplus_clip(d,e), 255)
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

Normalize: identity.
CapRule: true.
AuthRule: true.
Reads: empty.

Lift:
weight = min(surplus_clip(d,e), 255)
record = (weight:u8, tie:B32) where tie = tie256
order_key = (weight desc, tie asc)

State bytes: U8(weight).
DefaultBytes: U8(0).
MaxValueBytes: 1.

A.5 CMD (0x05)

Category: GOV
Value model: MAX_BY

Key parts: B32(cmdh)
Key: KeyDerive(0x05, B32(cmdh))

Payload: B32(cmdh) || B32(obj_id) || B32(schema_id) || BYTES(cost_vec) with len(cost_vec) <= 64

Normalize: identity.
CapRule: len(cost_vec) <= 64.
AuthRule: true.
Reads: empty.

Lift:
weight = min(surplus_clip(d,e), 255)
record = (obj_id:B32, schema_id:B32, cost_vec:bytes, weight:u8, tie:B32)
order_key = (weight desc, tie asc)

State bytes: B32(obj_id) || B32(schema_id) || BYTES(cost_vec).
DefaultBytes: empty.
MaxValueBytes: 130.

A.6 CMDVOTE (0x06)

Category: GOV
Value model: MAX_BY

Key parts: B32(cmdh)
Key: KeyDerive(0x06, B32(cmdh))

Payload: B32(cmdh)

Normalize: identity.
CapRule: true.
AuthRule: true.
Reads: empty.

Lift:
weight = min(surplus_clip(d,e), 255)
record = (weight:u8, tie:B32)
order_key = (weight desc, tie asc)

State bytes: U8(weight).
DefaultBytes: U8(0).
MaxValueBytes: 1.

A.7 SCHEMA (0x07)

Category: GOV
Value model: MAX_BY

Key parts: B32(schema_id)
Key: KeyDerive(0x07, B32(schema_id))

Payload: B32(schema_id) || B32(obj_id)

Normalize: identity.
CapRule: true.
AuthRule: true.
Reads: empty.

Lift:
weight = min(surplus_clip(d,e), 255)
record = (obj_id:B32, weight:u8, tie:B32)
order_key = (weight desc, tie asc)

State bytes: B32(obj_id).
DefaultBytes: empty.
MaxValueBytes: 32.

A.8 SCHEMAVOTE (0x08)

Category: GOV
Value model: MAX_BY

Key parts: B32(schema_id)
Key: KeyDerive(0x08, B32(schema_id))

Payload: B32(schema_id)

Normalize: identity.
CapRule: true.
AuthRule: true.
Reads: empty.

Lift:
weight = min(surplus_clip(d,e), 255)
record = (weight:u8, tie:B32)
order_key = (weight desc, tie asc)

State bytes: U8(weight).
DefaultBytes: U8(0).
MaxValueBytes: 1.

A.9 PTR (0x09)

Category: DATA
Value model: TOPK_MAP with K=8, mkey=owner

Key parts: B32(scope) || BYTES(name)
Key: KeyDerive(0x09, parts)

Payload: B32(scope) || BYTES(name) || U8(mode) || U32LE(until) || B32(ref) || BYTES(aux)
Constraints: len(name) <= 64, len(aux) <= 64, mode in {0,1}

Normalize: identity.
CapRule: constraints.
AuthRule: true.
Reads: empty.

Lift:
owner = Owner(pk)
level = min(surplus_clip(d,e), 255)
aux_hash = H(“aux”||aux)
record = (owner:B32, mode:u8, level:u8, until:u32, ref:B32, aux_hash:B32, tie:B32)
best_key within owner = (mode desc, level desc, until desc, tie asc)
order_key across owners = same tuple

DefaultBytes: empty vector.
MaxValueBytes: 8192.

A.10 POS (0x0A)

Category: DATA
Value model: TOPK_MAP with K=8, mkey=owner

Key parts: B32(owner)
Key: KeyDerive(0x0A, B32(owner))

Payload: B32(owner) || BYTES(pos) with len(pos) <= 64

Normalize: identity.
CapRule: len(pos) <= 64.
AuthRule: require owner == Owner(pk).
Reads: empty.

Lift:
level = min(surplus_clip(d,e), 255)
pos_hash = H(“pos”||pos)
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

Payload: B32(vertex) || B32(owner) || U32LE(until) || BYTES(endpoint) with len(endpoint) <= 128

Normalize: identity.
CapRule: len(endpoint) <= 128.
AuthRule: require owner == Owner(pk).
Reads: empty.

Lift:
level = min(surplus_clip(d,e), 255)
ep_hash = H(“ep”||endpoint)
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

Normalize: identity.
CapRule: true.
AuthRule: true.
Reads: empty.

Lift:
owner = Owner(pk)
lvl = min(surplus_clip(d,e), 255)
src = id(d)
record = (owner:B32, lvl:u8, src:B32, tie:B32)
order_key = (lvl desc, src asc, tie asc)

State bytes: B32(owner) || U8(lvl) || B32(src).
DefaultBytes: empty.
MaxValueBytes: 65.

A.13 LOCK (0x0D)

Category: GOV
Value model: MAX_BY

Key parts: B32(lock_id)
Key: KeyDerive(0x0D, B32(lock_id))

Payload: B32(lock_id) || U32LE(lease_until_epoch)

Normalize: identity.
CapRule: true.
AuthRule: true.
Reads: empty.

Lift:
owner = Owner(pk)
lvl = min(surplus_clip(d,e), 255)
src = id(d)
record = (owner:B32, until:u32, lvl:u8, src:B32, tie:B32)
order_key = (lvl desc, until desc, src asc, tie asc)

State bytes: B32(owner) || U32LE(until) || U8(lvl) || B32(src).
DefaultBytes: empty.
MaxValueBytes: 69.

A.14 TOMB (0x0F)

Category: GOV
Value model: MAX_BY

Key parts: B32(target_k)
Key: KeyDerive(0x0F, B32(target_k))

Payload: B32(target_k) || U8(kind) || U32LE(until_epoch)
kind: 0=DEL, 1=UNDEL

Normalize: identity.
CapRule: true.
AuthRule: true.
Reads: empty.

Lift:
lvl = min(surplus_clip(d,e), 255)
src = id(d)
record = (kind:u8, until:u32, lvl:u8, src:B32, tie:B32)
order_key = (lvl desc, kind desc, until desc, src asc, tie asc) where DEL > UNDEL for kind desc

State bytes: U8(kind) || U32LE(until) || U8(lvl) || B32(src).
DefaultBytes: empty.
MaxValueBytes: 38.

Effective deletion rule (normative):
Let tomb be the decoded TOMB value for target key k at epoch e.
If tomb.kind==DEL and (tomb.until==0xFFFFFFFF or e <= tomb.until) then k is logically deleted at epoch e; otherwise it is not deleted.

A.15 TX (0x0E)

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

A. epoch MUST equal ctx0.e
B. kind in {0,1} where 0=APPLY, 1=CANCEL
C. |locks| <= 4, |uniq_keys| <= 4, |cas| <= 4, |items| <= 8
D. If kind==CANCEL then items MUST be empty
E. Each TxItem.k MUST be a user-writable standard tag and MUST NOT be CAT_RESERVED
F. Each item_payload_bytes MUST pass Parse/KeyCheck/Normalize/Cap/Auth for its target tag using ctx0 derived from the outer ctx0 with k=TxItem.k, and MUST be re-emitted in canonical form inside TX.

AuthRule: true (per-item AuthRule enforced during Normalize).

Reads_t (declaration):

If kind==CANCEL: Reads is empty.
If kind==APPLY: Reads is the sorted unique union of:
	•	KeyDerive(0x0D, B32(lock_id)) for each lock_id in locks
	•	each uniq_key in uniq_keys (these are UNIQ keys)
	•	each k in cas (the CAS target keys)
	•	KeyDerive(0x0F, B32(item.k)) for each item target key item.k

Lift:

lvl = min(surplus_clip(d,e), 255)
src = id(d)
record = (kind:u8, lvl:u8, src:B32, tie:B32, locks, uniq_keys, cas, items)
order_key = (lvl desc, kind desc, src asc, tie asc) where CANCEL > APPLY for kind desc

State bytes: canonical re-encoding of fields excluding order_key.
DefaultBytes: empty.
MaxValueBytes: 2048 (MUST be enforced).

Appendix B. System modules (normative, v0.01 closed set)

All system effects are computed as local modules. Modules MUST satisfy:
	•	Footprint_M depends only on (epoch e, included body B canonical bytes, constants)
	•	Eval_M reads only keys in R_M and writes only keys in W_M
	•	Both footprints are bounded by constants implied by MAX_INCLUDED, MAX_OPS, MAX_READS, and CAP_*.

Modules in v0.01:

B.1 CMDIDX module

Key: k_cmdidx = KeyDerive(0xE1, U32LE(e))
Value model: TOPK_SET(MAX_INDEX) with MAX_INDEX = 32

Footprint:

W_M = { k_cmdidx }
R_M includes:
	•	For each included delta containing CMD or CMDVOTE ops, the corresponding CMDVOTE keys (0x06) for cmdh found in those ops.
	•	The previous epoch index key KeyDerive(0xE1, U32LE(e-1)) if e>0.

Eval:

Candidate cmdh set = cmdh values appearing in included CMD or CMDVOTE ops union cmdh values listed in previous CMDIDX value (if present).
Weight for cmdh is the u16 value stored at CMDVOTE(cmdh) in S’ (absent => 0).
Record = (cmdh, weight, tie) where tie = H(“cmdidx”||cmdh)
order_key = (weight desc, tie asc)
Output is TopK_SET(MAX_INDEX).

B.2 SCHEMAIDX module

Analogous to CMDIDX using SCHEMAVOTE (0x08) and SCHEMA/SCHEMAVOTE ops.
Key: k_schemaidx = KeyDerive(0xE2, U32LE(e))
Previous key KeyDerive(0xE2, U32LE(e-1)) if e>0.

B.3 SIG module (SIG and SIGIDX)

Constants: WINDOW_MAX = 64, MAX_INDEX = 32

Define for each included delta id x:

v(x) = H(“sigv” || x) interpreted as 256-bit vertex
ord(x) = low64(H(“sigo” || x)) as u64
mass(x) = surplus_clip(x,e) as non-negative integer

SIG key for vertex v:

k_sig(e,v) = KeyDerive(0xE0, U32LE(e) || v)

SIGIDX key:

k_sigidx = KeyDerive(0xE3, U32LE(e))

Footprint:

W_M includes:
	•	k_sigidx
	•	k_sig(e,v) for each vertex v that appears among {v(x) | x in included deltas}

R_M is empty.

Eval:

For each vertex v, define SigWindow(v) = multiset of pairs (ord(x), x) for included x with v(x)=v.
Let b(v) be sum of mass(x) over the WINDOW_MAX smallest elements of SigWindow(v) by (ord asc, x asc).

Write SIG at k_sig(e,v) as MAX_BY of record (b:i64, tie:B32) where tie = H(“sig”||U32LE(e)||v), order_key = (b desc, tie asc). If v has no entries, SIG key MAY be absent (no write).

Write SIGIDX as TOPK_SET(MAX_INDEX) over records (v, b, tie) for all v that appear, ordered by (b desc, tie asc), with tie = H(“sigidx”||v).

B.4 TX module (TX effects and TXIDX)

TXIDX key: k_txidx = KeyDerive(0xE4, U32LE(e))
MAX_INDEX = 32

Footprint:

W_M includes:
	•	k_txidx
	•	For each winning TX record (defined below), all item target keys written by that record (the TxItem.k keys)

R_M includes, for each winning TX record r of kind APPLY:
	•	All LOCK keys KeyDerive(0x0D, B32(lock_id)) referenced by r.locks
	•	All UNIQ keys referenced by r.uniq_keys
	•	All CAS target keys referenced by r.cas
	•	All TOMB keys KeyDerive(0x0F, B32(item.k)) for each item target key item.k

Winning TX record extraction (state-independent):

For each TX key k_tx = KeyDerive(0x0E, U32LE(e)||B32(tx_id)), the Phase 1 TX value is MAX_BY over records contributed in included deltas. Since TX keys are epoch-scoped, the default is empty and the winner depends only on included deltas.

Let WinningTX be the set of (k_tx, r, d_src) where r is the winning record and d_src is the included delta that contributed it (ties resolved by order_key already).

Eval:

For each winning record r:

If r.kind == CANCEL: it produces no item effects.

If r.kind == APPLY: evaluate eligibility on S’ using only the declared reads:

A. Locks: for each lock key, require LOCK exists with owner == Owner(d_src.pk) and until_epoch >= e.
B. Uniq claims: for each uniq key, require UNIQ exists with owner == Owner(d_src.pk).
C. CAS: for each (k, expect_hash), require val_hash(S’(k)) == expect_hash (absent uses DefaultBytes_tag(k)).
D. Tomb: for each item target key k, if TOMB(k) indicates deleted at epoch e, TX fails.

Conflict resolution (wins-all, deterministic):

Let lvl_src = min(surplus_clip(d_src,e),255).

For each target key k, define BestTX(k) as the eligible APPLY TX with maximal (lvl_src desc, id(d_src) asc) that writes k.
A TX is Accepted iff for every item key it writes, BestTX(k) equals that TX.

Application:

For each Accepted TX, apply each item (k, item_payload_bytes_canonical) as a virtual op at key k:
	•	Use ctx0_item derived from d_src ctx0 with k = item.k, and payload bytes as provided (already canonical in TX payload).
	•	Compute payload_norm_item by re-parsing and normalizing under the target tag rules (MUST succeed, else treat as no-op).
	•	Compute record = Lift_tag(k)(ctx1 derived from d_src, payload_norm_item, tie computed as H(“txitem”||id(d_src)||k||H(payload))).
	•	Join into Δ_tx[k].

TXIDX:

Write TXIDX as TOPK_SET(MAX_INDEX) over records (tx_key, status, lvl, src_id, tie) ordered by (lvl desc, src_id asc, tie asc).
status encodes APPLY_ACCEPTED, APPLY_REJECTED, or CANCEL.
tie = H(“txidx”||tx_key||src_id).

Appendix C. SMT256 commitment, openings, and multiproofs (normative)

C.1 SMT256 root

The authoritative state is a sparse Merkle tree over 256-bit keys.

val_hash(vbytes) = H(“val” || vbytes)

leaf_hash(k, present, vbytes) =
if present==1 then H(“leaf” || k || val_hash(vbytes))
else H(“leaf” || k || B32(0x00..00))

node_hash(left, right) = H(“node” || left || right)

Bits of key are MSB-first. Root is depth 0. Leaves are depth 256.

Absent keys are representable and verifiable via present==0 proofs.

C.2 Default value bytes

For any key k, its tag is t=tag_id(k). DefaultBytes_t MUST be used as the value bytes when a key is absent in the SMT and the protocol needs a decoded value for join.

C.3 SMT multiproof format

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

C.4 Deterministic multiproof verification algorithm

Given root R, key list K, leaf values, and siblings:
	1.	For each i, compute Li = leaf_hash(ki, present_i, vbytes_i). Insert into a map M keyed by node-id at depth 256 with prefix=ki.
	2.	For each sibling entry (d,p)->h: insert into a map S keyed by node-id. If an entry already exists with a different hash, FAIL.
	3.	For depth d from 255 down to 0: for each node-id (d,p) that is a parent of any node-id already in M:
Let left child id = (d+1,p||0), right child id = (d+1,p||1).
Let left hash be M[left] if present, else S[left] must exist. Similarly for right.
Compute parent hash = node_hash(left_hash, right_hash). Insert into M[(d,p)] ensuring no conflicting hash.
	4.	Require M[(0,empty)] == R.

C.5 Deterministic multi-update recomputation (stateless)

Given:
	•	prev_root R_prev
	•	a multiproof that opens KeysNeedOld and provides siblings sufficient to recompute paths
	•	a set of updated keys U (subset of KeysNeedOld) and their NewBytes[k]

Compute:
	1.	Verify the multiproof against R_prev using C.4, yielding OldBytes for KeysNeedOld.
	2.	For each k in U, define the new leaf hash Lk_new = leaf_hash(k, present=1, NewBytes[k]).
	3.	Replace the corresponding leaf entries in M at depth 256 with Lk_new for k in U, leaving other opened leaves as-is.
	4.	Recompute upward hashes deterministically using the same sibling map S as in C.4.
	5.	The resulting root is the recomputed new root.

This recomputation MUST be used for stateless verification in Section 13.4.7.

End of Specification v0.01
