Specification v0.0.1
Title: Canonical Join-DAG with Proof-Carried Checkpoints and Derived Optimization Views
Status: Draft (protocol-level semantics frozen; proof system and root structure pluggable under fixed interfaces)
	1.	Goals

1.1 Primary goals

A. Partial order world: the base history is a DAG of events with a causal partial order, not a total order.

B. Confluence and idempotence by construction: all state updates are pointwise joins over join-semilattices, so evaluation is order-independent and replay-safe.

C. Constant verification on the hot path: accepting and relaying high-frequency events is constant-time and does not require reading prior state.

D. Fast verifiable reads: clients can verify values using a canonical checkpoint root plus a short inclusion proof.

E. Non-regressing extensibility: new commands and policies can be introduced without increasing per-event verification cost.

F. Optimization-first semantics: canonical selection and higher-level behavior are defined by deterministic derived views (including a KKT-like potential field), not by mutable consensus state.

1.2 Non-goals

A. Global instant finality for every individual delta event.

B. Mandatory global indexing; indexing is optional and discovered as a derived view.

C. Privacy guarantees beyond basic cryptographic authenticity; confidentiality is out of scope.
	2.	Normative language

The key words MUST, MUST NOT, SHOULD, SHOULD NOT, MAY are to be interpreted as described in RFC 2119.
	3.	Notation

3.1 Bitstrings

{0,1}^n denotes the set of n-bit strings.

3.2 Hash

H(x) denotes a 256-bit cryptographic hash function.

3.3 Concatenation

x || y denotes byte concatenation of encodings of x and y.

3.4 Integer ranges

All integer fields are fixed-width and little-endian unless stated otherwise.
	4.	Constants (protocol parameters)

All constants below are part of the v0.0.1 protocol identity and MUST be identical across compliant implementations.

4.1 Space and neighborhood

SPACE_BITS = 256
DEGREE_D = 8
NEIGHBOR_CONST[i] = H(“nbr” || u32(i)) for i in 0..DEGREE_D-1

NEIGHBOR_PERM is a public permutation P: {0,1}^256 -> {0,1}^256 defined as an 8-round Feistel network:

Let v = L || R where L,R are 128-bit.
For r = 0..7:
F = Trunc128(H(“F” || u32(r) || R))
(L, R) = (R, L xor F)
Then P(v) = L || R.

Neighbor function:
N_i(v) = P(v xor NEIGHBOR_CONST[i])

4.2 Event limits

MAX_PARENTS = 8
MAX_OPS = 16
MAX_EVENT_BYTES = 2048

4.3 Data type limits

WINDOW_MAX = 64
TOP_MAX = 32
CANDIDATE_MAX = 8
COUNTER_CAP_MAX = 2^20

4.4 Epoch and difficulty

EPOCH_GENESIS = 0
TARGET_BITS_BASE = 18

Difficulty is expressed in leading-zero bits, using clz256(x) = number of leading zero bits in a 256-bit value x.

4.5 KKT-like view parameters

KKT_T = 3
KKT_RHO_NUM = 1
KKT_RHO_DEN = 2

Numeric format for KKT computations:
Signed fixed-point Q32.32 in int64.
	5.	Cryptographic primitives

5.1 Hash

H MUST be collision-resistant and preimage-resistant. Output size MUST be 256 bits.

5.2 Signatures

SIGCHECK(pk, msg, sig) -> bool MUST verify a digital signature under public key pk.

Signature scheme is pluggable, but pk and sig encodings MUST be unambiguous and fixed-length within the scheme.

5.3 Proof system interface (checkpoint and membership)

Two proof interfaces are required. Concrete instantiations are out of scope for v0.0.1, but interfaces are normative.

A. Checkpoint proof interface

VerifyCheckpointProof(epoch, prev_root, cut_commit, batch_commit, root, meta, proof) -> bool

MUST run in time independent of the number of delta events included in the checkpoint (constant-time verifier).

B. Membership proof interface

VerifyMembership(root, key, value, proof) -> bool

MUST run in time independent of the global state size (typically logarithmic in tree height but bounded by a fixed constant for fixed-height structures).
	6.	Key space and key derivation

6.1 Key space

K = {0,1}^256.

All keys are derived by:
KEY(tag, parts…) = H(tag || encode(parts…))

6.2 Standard tags (v0.0.1)

“obj”       immutable objects
“log”       windowed logs
“top”       top-k rankings
“keep”      retention votes
“vote”      governance votes (general)
“cmd”       command declarations
“cmdvote”   command votes
“inj”       dipole injection logs
“ptr”       name->ref candidates
“pos”       position candidates
“idx”       indexer endpoint candidates
“cut”       cut declarations (optional if cut is carried only by checkpoints)

Tags are ASCII strings. Encodings of parts MUST be canonical and length-delimited.
	7.	State model

7.1 State is derived, not primary

The world’s primary data is the set of accepted delta events and accepted checkpoints.

State S at a given checkpoint is the evaluation of join operations described by delta events selected by that checkpoint, starting from the previous checkpoint state.

7.2 Join-semilattice requirement

Every key k has an associated value type X_k with a join operator join_k.

join_k MUST be commutative, associative, and idempotent.

7.3 Standard value types (fixed set)

v0.0.1 defines exactly six standard value types. All delta operations MUST target keys whose types are one of these.

Type 1: SETONCE
Representation: either EMPTY or VALUE[32].
Join rule: if either side is VALUE, the result is the lexicographically smaller VALUE by bytes; if one is EMPTY, return the other.
Rationale: makes join total and deterministic even under conflicts, while behaving as “set once” when honest.

Type 2: LWW (last-write-wins)
Representation: (epoch:u32, value:bytes[32]).
Join rule: pick the larger tuple by (epoch, value) lexicographic.

Type 3: COUNTER_CAP
Representation: level:u32 where 0 <= level <= COUNTER_CAP_MAX.
Join rule: max(level).

Type 4: WINDOW64
Representation: up to WINDOW_MAX entries of (ord:u64, item:bytes[32]).
Join rule: union entries then keep the WINDOW_MAX smallest by (ord, item). Deterministic.

Type 5: TOP32
Representation: up to TOP_MAX entries of (score:i64, item:bytes[32]).
Join rule: union entries then keep the TOP_MAX largest by (score, item). Deterministic.

Type 6: CANDIDATE_SET_8
Representation: up to CANDIDATE_MAX entries of Candidate records:
mode:u8 (0=ALIVE, 1=DEAD)
until:u32
ref:bytes[32]
owner:bytes[32] (hash of pk or fixed pk encoding hash)
level:u32
aux:bytes[32] (optional payload, e.g., endpoint)
Join rule: union then keep the CANDIDATE_MAX largest by the Candidate ordering defined per key-tag below.

Candidate ordering is a total order determined by tag:
For “ptr”:
Primary: mode (DEAD > ALIVE)
Secondary: level (higher better)
Tertiary: until (higher better)
Quaternary: H(owner || ref || aux) (lower better as tie-breaker)
For “pos”:
Primary: level (higher better)
Secondary: H(owner || ref) (lower better)
For “idx”:
Primary: level (higher better)
Secondary: until (higher better)
Tertiary: H(owner || aux) (lower better)
	8.	Events

8.1 Event identity

All events have an id:
id = H(encode(event_without_id))

Nodes MUST recompute and verify id matches the event body.

8.2 DeltaEvent

DeltaEvent fields:
type = “delta”
epoch:u32
parents: array[0..MAX_PARENTS] of bytes[32] (event ids)
ops: array[1..MAX_OPS] of Op
pk: public key encoding (scheme-defined)
sig: signature
nonce: bytes[32]

Op fields:
k: bytes[32]
vtype: u8 (1..6 as defined above)
payload: bytes (type-specific encoding)

DeltaEvent encoding MUST be canonical. Total encoded size MUST be <= MAX_EVENT_BYTES.

8.3 CheckpointEvent

CheckpointEvent fields:
type = “checkpoint”
epoch:u32
prev_root: bytes[32]
cut_commit: bytes[32]
batch_commit: bytes[32]
root: bytes[32]
meta: bytes (canonical encoding; fixed max size 256 bytes)
proof: bytes (canonical encoding; fixed max size 8192 bytes)
pk, sig

CheckpointEvent size MUST be bounded by a fixed maximum (e.g., 10KB). Exact bound is implementation-defined but MUST be enforced.
	9.	Epoch, beacon, and difficulty

9.1 Canonical checkpoint root

Let root_star(epoch) be the root of the canonical checkpoint of that epoch (defined in Section 11).

Genesis:
root_star(0) = GENESIS_ROOT (fixed constant in the genesis configuration)

9.2 Beacon

beacon(epoch) is defined as:
beacon(epoch) = H(root_star(epoch-1) || u32(epoch))

For epoch = 0, beacon(0) = H(GENESIS_ROOT || u32(0)).

9.3 Cost function for DeltaEvent

Let bytes(e) be the encoded byte length of the DeltaEvent.

Define heavy_count as the number of ops whose key tag is in {“top”, “param”} (v0.0.1 reserves “param” for future; if absent, heavy_count counts only “top”).

Define:
C(e) = 1 + 1ops_count + 1(bytes(e)/256 rounded up) + 2*heavy_count

All arithmetic is in unsigned integers.

9.4 Difficulty requirement

Compute:
h = H(beacon(epoch) || H(pk) || nonce || H(ops_encoding))
Let z = clz256(h).

Required leading zeros:
req = TARGET_BITS_BASE + ceil_log2(C(e))

AcceptDelta requires:
z >= req

ceil_log2(n) for n>=1 is the smallest integer r such that 2^r >= n.
	10.	Acceptance rules

10.1 AcceptDelta

A node MUST accept and MAY relay a DeltaEvent e iff all conditions hold:

A. Canonical encoding, size <= MAX_EVENT_BYTES.
B. parents length <= MAX_PARENTS and ops length in 1..MAX_OPS.
C. SIGCHECK(pk, H(epoch || parents || ops || nonce), sig) == true.
D. Every op is well-typed and payload encoding matches vtype with limits.
E. Difficulty requirement holds (Section 9.4).

A node MUST reject otherwise.

Nodes SHOULD deduplicate by event id and MUST NOT relay duplicates.

10.2 AcceptCheckpoint

A node MUST accept and MAY relay a CheckpointEvent q iff:

A. q.epoch is either the next epoch after the highest epoch for which the node knows a canonical checkpoint, or within a small bounded window (implementation MAY accept within [current_epoch, current_epoch+1]).
B. SIGCHECK(pk, H(all_checkpoint_fields_except_sig), sig) == true.
C. VerifyCheckpointProof(q.epoch, q.prev_root, q.cut_commit, q.batch_commit, q.root, q.meta, q.proof) == true.
D. q.prev_root equals root_star(q.epoch-1) once a canonical checkpoint for epoch-1 is known.

Nodes MUST reject otherwise.
	11.	Canonical checkpoint selection (partial order, single canonical view)

11.1 Candidate set

For a given epoch e, let Q_e be the set of accepted checkpoints with epoch == e and valid proof.

11.2 Score function

ScoreQ(q) MUST be computable from q.meta alone in constant time.

v0.0.1 defines meta as:
meta.work_bits_sum:u64
meta.batch_count:u32
meta.batch_commit:bytes[32] (must match q.batch_commit)

ScoreQ(q) = meta.work_bits_sum, tie-breaker is smaller H(q.root || H(q.pk)).

The proof MUST attest that meta.work_bits_sum equals the sum over all included delta events of their achieved leading-zero bits z (as computed in Section 9.4) truncated to 16 bits each and summed in u64, and that batch_commit is the Merkle root of included delta ids.

11.3 Canonical selection

CanonicalCheckpoint(e) is:
pick q in Q_e with maximal ScoreQ(q); break ties by the tie-breaker.

Once selected, q.root defines root_star(e) and becomes the anchor for reads and beacons.

11.4 Finality of canonical root

A node SHOULD treat root_star(e) as stable after it has observed CanonicalCheckpoint(e) and at least one accepted checkpoint for epoch e+1 that references it as prev_root.
	12.	Batch and cut commitments

12.1 Batch commitment

batch_commit is a Merkle root over the sorted list of included delta event ids (ascending bytes).

12.2 Cut commitment

cut_commit is a Merkle root over the sorted list of frontier delta event ids (ascending bytes), intended to summarize the covered region of the delta DAG.

v0.0.1 does not require nodes to validate cut semantics beyond proof verification; cut is primarily for audit and reproducible replay.
	13.	State evaluation semantics for checkpoints

13.1 Deterministic fold order inside proofs

Because join is commutative and idempotent, the mathematical result is order-independent. For circuit determinism, the proof system MUST fold included delta events in ascending order of delta id, and within each delta, ops in ascending order of (key, vtype, payload bytes).

13.2 Applying ops

For each op (k, vtype, payload):
decode payload to DeltaValue in X_k
state_value[k] = join_k(state_value[k], DeltaValue)

13.3 Key typing

Key typing is determined by the first successful op for that key in the fold order within the checkpoint, and MUST remain consistent thereafter. If a later op attempts a different vtype for the same key, the checkpoint proof MUST fail.
	14.	Reads

14.1 Read anchor

A client MUST select an epoch e and use root_star(e) as the read anchor.

14.2 Membership proof

To read key k:
obtain (value, proof) from any source
verify VerifyMembership(root_star(e), k, value, proof) == true

14.3 Trust model

Clients MUST NOT trust unverified values. Clients MAY trust a canonical checkpoint root if they can verify its proof or obtain it from multiple independent sources.
	15.	Derived views (canonical functions)

Derived views are deterministic functions of a checkpoint root and local parameters. They are not consensus state and MUST NOT be written as authoritative values.

15.1 Name resolution (PTR)

For (scope, name), let k = KEY(“ptr”, scope, name).
Let C be the CandidateSet_8 value at k.

ResolvePTR(root, scope, name, now_epoch) returns the best candidate c in C such that:
c.mode == ALIVE
c.until >= now_epoch
If none exist, return NONE.

Best is determined by the “ptr” Candidate ordering in Section 7.3.

15.2 Position resolution (POS)

k = KEY(“pos”, H(pk)).
ResolvePOS returns the best candidate by “pos” Candidate ordering.

15.3 Indexer discovery (IDX)

k = KEY(“idx”, v) for a vertex v in {0,1}^256.
ResolveIDX returns the best candidate with until >= now_epoch by “idx” ordering.

15.4 Active command set (governance view)

Command declaration:
CMD key: KEY(“cmd”, cmd_name) of type SETONCE containing (obj_id, cost_vec).
Command vote:
CMDVOTE key: KEY(“cmdvote”, cmd_name) of type COUNTER_CAP containing level.

ActiveSet(root) is defined as:
Consider all commands whose CMD declaration exists in root.
Define vote(cmd) = CMDVOTE.level (default 0 if absent).
Define cost(cmd) = cost_vec from CMD declaration.
Compute score(cmd) = vote(cmd) - dot(price_vec, cost(cmd)).
Select top K_ACTIVE commands by score(cmd), ties broken by H(cmd_name) ascending.
K_ACTIVE and price_vec are protocol constants for v0.0.1:
K_ACTIVE = 64
price_vec = (1, 1, 1) in integer units.

15.5 KKT-like potential view

15.5.1 Injection measurement b(v)

For a vertex v:
inj_key = KEY(“inj”, v)
inj_window = WINDOW64 at inj_key
Each entry is (ord, item) where item encodes (eid, signed_q:i32).

Define b(v) as the sum of signed_q over all unique eid entries in the window, in int64.

15.5.2 Neighbor averaging operator P

For a function f: V -> Q32.32:
(P f)(v) = (1/DEGREE_D) * sum_{i=0..DEGREE_D-1} f(N_i(v))

All arithmetic in Q32.32 with rounding toward zero.

15.5.3 Potential lambda_hat

Define b_q(v) as b(v) promoted to Q32.32 (b(v) << 32).

Define:
lambda_0 = b_q
lambda_hat = lambda_0
For t = 1..KKT_T:
lambda_t = P(lambda_{t-1})
lambda_hat += (KKT_RHO_NUM / KKT_RHO_DEN)^t * lambda_t

Where the scaling factor is applied in Q32.32 with exact rational scaling for rho = 1/2:
multiply by 1, then shift right by t bits (for rho^t).

15.5.4 Gradient for routing and ranking

For v and neighbor i:
grad_i(v) = lambda_hat(v) - lambda_hat(N_i(v))

Clients MAY use grad_i(v) to choose moves, prioritize queries, or weight ranking scores.
	16.	Standard operations (high-level op library)

These are convenience operations that compile to one DeltaEvent containing one or more JOIN ops.

16.1 OBJ_PUT_ONCE(obj_id, blob_hash)
JOIN(KEY(“obj”, obj_id), SETONCE(blob_hash))

16.2 LOG_APPEND(scope, topic, ord, item)
JOIN(KEY(“log”, scope, topic), WINDOW64_add(ord, item))

16.3 TOP_PUSH(scope, metric, score, item)
JOIN(KEY(“top”, scope, metric), TOP32_add(score, item))

16.4 KEEP_LEVEL(obj_id, level)
JOIN(KEY(“keep”, obj_id), COUNTER_CAP(level))

16.5 CMD_DECLARE(cmd_name, obj_id, cost_vec)
JOIN(KEY(“cmd”, cmd_name), SETONCE(obj_id || cost_vec))

16.6 CMD_VOTE(cmd_name, level)
JOIN(KEY(“cmdvote”, cmd_name), COUNTER_CAP(level))

16.7 DIPOLE_INJECT(u, v, q, ord)
Encodes two WINDOW64 entries with same eid and opposite signed_q:
eid = H(“inj” || u || v || ord || H(pk))
JOIN(KEY(“inj”, u), WINDOW64_add(ord, encode(eid, +q)))
JOIN(KEY(“inj”, v), WINDOW64_add(ord, encode(eid, -q)))

16.8 PTR_OFFER(scope, name, ref, until, level)
owner = H(pk)
cand = Candidate(ALIVE, until, ref, owner, level, aux=0)
JOIN(KEY(“ptr”, scope, name), CANDIDATE_add(cand))

16.9 PTR_TOMBSTONE(scope, name, until, level)
owner = H(pk)
cand = Candidate(DEAD, until, ref=0, owner, level, aux=0)
JOIN(KEY(“ptr”, scope, name), CANDIDATE_add(cand))

16.10 POS_OFFER(pos, level)
owner = H(pk)
cand = Candidate(ALIVE, until=0, ref=pos, owner, level, aux=0)
JOIN(KEY(“pos”, owner), CANDIDATE_add(cand))

16.11 IDX_ADVERTISE(v, endpoint_hash, until, level)
owner = H(pk)
cand = Candidate(ALIVE, until, ref=0, owner, level, aux=endpoint_hash)
JOIN(KEY(“idx”, v), CANDIDATE_add(cand))
	17.	Networking and relay

17.1 Relay policy

Nodes MUST relay accepted DeltaEvents and CheckpointEvents to peers, subject to local rate limits.

Nodes MAY drop events under resource pressure, but MUST NOT violate acceptance rules when they do relay.

17.2 Deduplication

Nodes MUST deduplicate by id. Memory for deduplication MAY be bounded with a time-based eviction policy.
	18.	Security considerations

18.1 Sybil and spam

AcceptDelta includes a difficulty requirement tied to a canonical beacon. This provides a resource cost per event independent of identity. Implementations SHOULD enforce per-peer and per-subnet rate limits as additional defenses.

18.2 Eclipse and partition

Because canonical roots are selected by score among valid checkpoints, partitions can cause temporary divergence. Clients SHOULD query multiple independent sources for canonical checkpoints.

18.3 Checkpoint grinding

Beacon depends on previous canonical root; difficulty ties deltas to the epoch. Checkpoint selection uses proven meta.work_bits_sum, reducing arbitrary selection.

18.4 State poisoning

All authoritative reads are anchored to canonical roots plus membership proofs. Unverified data MUST NOT be trusted.
	19.	Compliance requirements

A v0.0.1 compliant implementation MUST:

A. Implement AcceptDelta and AcceptCheckpoint exactly as specified.
B. Implement the six standard value types and their join rules.
C. Implement canonical checkpoint selection.
D. Implement membership verification against canonical roots.
E. Provide deterministic derived views (ResolvePTR, ActiveSet, KKT potential).
	20.	Versioning

v0.0.1 is identified by the tuple of protocol constants (Section 4), standard tags (Section 6.2), value types (Section 7.3), acceptance rules (Section 10), and canonical selection rules (Section 11). Any change to these constitutes a new protocol version.

Appendix A. Rationale for the minimal primitive core

The only indispensable state transition is pointwise join (JOIN). Delta emission (EMITDelta) is necessary to introduce updates into the partial-order world. Checkpoint emission (EMITCheckpoint) is necessary to make state derivation and reads fast and verifiable. Membership reads (OPEN) are necessary for clients to verify values. All other operations are compilations into JOIN ops plus derived-view functions.

Appendix B. Implementation notes (non-normative)

A. Proof system choice: transparent proofs reduce long-term operational risk; compressed proofs can be added later without changing the interfaces if VerifyCheckpointProof remains constant-time.

B. Root structure choice: a sparse Merkle tree is straightforward; alternative authenticated dictionaries are acceptable if VerifyMembership meets the interface.

C. Indexers: indexing is optional; discovery can be done via IDX candidates to reduce reliance on out-of-band configuration.

