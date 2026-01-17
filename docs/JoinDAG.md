Specification v0.01
Title: Canonical Join-DAG with Admission-Controlled Evidence Checkpoints and Unified Signal Field
Status: Draft (protocol semantics frozen; proof system and authenticated root structure pluggable under fixed interfaces)

1. Goals

1.1 Primary goals

A. Partial order world: the base history is a DAG of events with a causal partial order, not a total order.

B. Confluence and idempotence by construction: all state updates are pointwise joins over join-semilattices, so evaluation is order-independent and replay-safe.

C. Constant verification on the hot path: accepting and relaying high-frequency DeltaEvents is constant-time and does not require reading prior state.

D. Fast verifiable reads: clients verify values using a canonical checkpoint root plus a short membership (and optional non-membership) proof.

E. Non-regressing extensibility: new commands and policies can be introduced without increasing per-event verification cost; evolution occurs via derived views and governance keys.

F. Optimization-closed semantics: bandwidth allocation, “fees”, and exploration signals are defined by deterministic admission thresholds and surplus work, not by mutable consensus state.

G. Mathematical closure: all normative choices (admission, inclusion caps, canonical selection, fold order, derived views) are deterministic functions of protocol constants plus canonical roots and proofs.

1.2 Non-goals

A. Global instant finality for every individual DeltaEvent.

B. Mandatory global indexing; indexing is optional and discovered as a derived view.

C. Confidentiality/privacy beyond authenticity; encryption and anonymity are out of scope.

D. Minimizing total global work (e.g., PoW attempts) across all participants; the protocol minimizes per-node protocol work, not global economic cost.

2. Normative language

The key words MUST, MUST NOT, SHOULD, SHOULD NOT, MAY are to be interpreted as described in RFC 2119.

3. Notation

3.1 Bitstrings
{0,1}^n denotes the set of n-bit strings.

3.2 Hash
H(x) denotes a 256-bit cryptographic hash function.

3.3 Concatenation
x || y denotes byte concatenation of encodings of x and y.

3.4 Integers
All integer fields are fixed-width and little-endian unless stated otherwise.

3.5 Leading zeros
clz256(x) is the number of leading zero bits of a 256-bit value x.

4. Protocol constants (v0.01 identity)

All constants below are part of the v0.01 protocol identity and MUST be identical across compliant implementations.

4.1 Address space and neighborhood

SPACE_BITS = 256
DEGREE_D = 8
NEIGHBOR_CONST[i] = H("nbr" || u32(i)) for i in 0..DEGREE_D-1

NEIGHBOR_PERM is a public permutation P: {0,1}^256 -> {0,1}^256 defined as an 8-round Feistel network:

Let v = L || R where L,R are 128-bit.
For r = 0..7:
F = Trunc128(H("F" || u32(r) || R))
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

4.4 Epoch parameters

EPOCH_GENESIS = 0

4.5 Category system (admission control)

CATEGORY_COUNT = 4

Categories are numbered:

CAT_DATA = 0
CAT_RANK = 1
CAT_GOV = 2
CAT_RESERVED = 3

Per-category targets and slack:

TARGET[cat] are protocol constants (u16).
SLACK[cat] are protocol constants (u16).
CAP[cat] = TARGET[cat] + SLACK[cat].

Per-category threshold bounds:

TMIN[cat] and TMAX[cat] are protocol constants (u8).

Genesis thresholds:

T0[cat] are protocol constants (u8) defining Threshold(cat, epoch=0).

4.6 Cost weights

Cost is defined in Section 9 and uses these constants:

COST_BASE = 1
COST_PER_OP = 1
COST_PER_256B = 1
COST_HEAVY = 2

4.7 Unified signal diffusion parameters

SIG_T = 3
SIG_RHO_NUM = 1
SIG_RHO_DEN = 2

Numeric format for diffusion computations: signed fixed-point Q32.32 in int64.

5. Cryptographic primitives

5.1 Hash
H MUST be collision-resistant and preimage-resistant. Output MUST be 256 bits.

5.2 Signatures
SIGCHECK(pk, msg, sig) -> bool MUST verify a digital signature under public key pk.
Signature scheme is pluggable, but pk and sig encodings MUST be unambiguous and fixed-length within the scheme.

5.3 Proof system interfaces (checkpoint and membership)

Two proof interfaces are required. Concrete instantiations are out of scope for v0.01, but interfaces are normative.

A. Checkpoint proof interface

VerifyCheckpointProof(epoch, prev_root, batch_commit, cut_commit, root, meta, proof) -> bool

The verifier MUST run in time independent of the global state size and independent of the number of non-included network events. The verifier’s runtime MAY depend on protocol constants and on fixed maximums specified by this document.

B. Membership proof interface

VerifyMembership(root, key, value, proof) -> bool
VerifyNonMembership(root, key, proof) -> bool (optional, if provided by the root structure)

The verifier MUST run in time independent of the global state size. For fixed-depth authenticated dictionaries, runtime MUST be bounded by a protocol constant.

6. Key space and canonical key derivation

6.1 Key space

K = {0,1}^256.

All keys are derived by:
KEY(tag, parts...) = H(tag || encode(parts...))

Tags are ASCII strings. Encodings of parts MUST be canonical and length-delimited.

6.2 Standard tags (v0.01)

"obj"        immutable objects (content-addressed references)
"log"        windowed logs
"top"        top-k rankings
"keep"       retention votes
"vote"       general governance votes (reserved)
"cmd"        command declarations
"cmdvote"    command votes
"schema"     schema declarations for help/menu generation
"schemavote" schema votes
"ptr"        name->ref candidates
"pos"        position candidates
"idx"        indexer endpoint candidates
"cut"        cut declarations (optional; cut is also carried by checkpoints)

"inj" is reserved for signal windows if explicit injection is used by applications; the unified signal field in this spec is a derived view and MUST NOT be written as authoritative state.

7. State model

7.1 Primary data

The world’s primary data is the set of accepted DeltaEvents and accepted CheckpointEvents.

State at a checkpoint is derived by evaluating join operations described by the delta events selected by that checkpoint, starting from the previous checkpoint state.

7.2 Join-semilattice requirement

Every key k has an associated value type X_k with a join operator join_k.

join_k MUST be commutative, associative, and idempotent.

7.3 Standard value types (fixed set)

v0.01 defines exactly six standard value types. All explicit delta operations MUST target keys whose types are one of these.

Type 1: SETONCE
Representation: either EMPTY or VALUE[32].
Join rule: if either side is VALUE, the result is the lexicographically smaller VALUE by bytes; if one is EMPTY, return the other.

Type 2: LWW
Representation: (epoch:u32, value:bytes).
Join rule: pick the larger tuple by (epoch, value) lexicographic.

Type 3: COUNTER_CAP
Representation: level:u32 where 0 <= level <= COUNTER_CAP_MAX.
Join rule: max(level).

Type 4: WINDOW64
Representation: up to WINDOW_MAX entries of (ord:u64, item:bytes).
Join rule: union entries then keep the WINDOW_MAX smallest by (ord, item).

Type 5: TOP32
Representation: up to TOP_MAX entries of (score:i64, item:bytes).
Join rule: union entries then keep the TOP_MAX largest by (score, item).

Type 6: CANDIDATE_SET_8
Representation: up to CANDIDATE_MAX entries of Candidate records:
mode:u8 (0=ALIVE, 1=DEAD)
until:u32
ref:bytes
owner:bytes (hash of pk or fixed pk encoding hash)
level:u32
aux:bytes (optional payload, e.g., endpoint)

Join rule: union then keep the CANDIDATE_MAX largest by a total order determined by tag:

For "ptr":
Primary: mode (DEAD > ALIVE)
Secondary: level (higher better)
Tertiary: until (higher better)
Quaternary: H(owner || ref || aux) (lower better)

For "pos":
Primary: level (higher better)
Secondary: H(owner || ref) (lower better)

For "idx":
Primary: level (higher better)
Secondary: until (higher better)
Tertiary: H(owner || aux) (lower better)

8. Events

8.1 Event identity

All events have an id:
id = H(encode(event_without_id))

Nodes MUST recompute and verify id matches the event body.

8.2 DeltaEvent

DeltaEvent fields:

type = "delta"
epoch:u32
parents: array[0..MAX_PARENTS] of bytes[32] (event ids)
ops: array[1..MAX_OPS] of Op
pk: public key encoding (scheme-defined, fixed-length for the scheme)
sig: signature
nonce_incl: bytes[32]

Op fields:

k: bytes[32]
vtype: u8 (1..6 as defined above)
payload: bytes (type-specific canonical encoding)

DeltaEvent encoding MUST be canonical. Total encoded size MUST be <= MAX_EVENT_BYTES.

8.3 CheckpointEvent

CheckpointEvent fields:

type = "checkpoint"
epoch:u32
prev_root: bytes[32]
batch_commit: bytes[32]
cut_commit: bytes[32]
root: bytes[32]
meta: bytes (canonical encoding; fixed max size 512 bytes)
proof: bytes (canonical encoding; fixed max size 16384 bytes)
pk, sig

CheckpointEvent size MUST be bounded by a fixed maximum (e.g., <= 20KB). Exact bound is implementation-defined but MUST be enforced.

9. Cost, category, and admission tickets

9.1 Category of a DeltaEvent

Each Op key k has a tag derived as the first ASCII tag used in KEY(tag, ...). Implementations MUST be able to recover the tag used for standard keys (Section 6.2).

Category mapping is:

If any op’s key tag is "top" then cat = CAT_RANK.
Else if any op’s key tag is in {"cmd","cmdvote","schema","schemavote","keep","vote"} then cat = CAT_GOV.
Else cat = CAT_DATA.

All ops in a delta MUST map to either the same category or to a category that is not “lighter” than the delta’s category under the precedence CAT_RANK > CAT_GOV > CAT_DATA. The delta’s category is the maximum precedence among its ops. This rule is deterministic and MUST be used.

9.2 Cost function

Let bytes(d) be the encoded byte length of the DeltaEvent.

Define heavy_count as the number of ops whose key tag is in {"top"}.

Define:

C(d) = COST_BASE
+ COST_PER_OP * ops_count
+ COST_PER_256B * ceil(bytes(d)/256)
+ COST_HEAVY * heavy_count

All arithmetic is in unsigned integers. C(d) >= 1.

9.3 Epoch threshold parameters

Define Threshold(epoch e, category cat) as T_e[cat], computed by the deterministic update rule in Section 12, anchored on canonical checkpoint meta.

9.4 Inclusion ticket and work bits

Define:

ticket(d,e) = H( H("incl" || u32(e)) || H("cat" || u8(cat(d))) || id(d) || nonce_incl(d) )

Let z(d,e) = clz256(ticket(d,e)).

9.5 Required work bits

Define:

reqbits(d,e) = T_e[cat(d)] + ceil_log2(C(d))

ceil_log2(n) for n>=1 is the smallest integer r such that 2^r >= n.

9.6 Validity and surplus (“fee”)

A DeltaEvent d for epoch e is work-valid iff:

z(d,e) >= reqbits(d,e)

Define surplus bits:

surplus(d,e) = z(d,e) - reqbits(d,e) (u8, 0..255)

surplus16(d,e) = min(surplus(d,e), 65535) for accumulation.

This document uses “fee” as the paid surplus work; there is no separate token fee in v0.01.

10. Acceptance rules

10.1 AcceptDelta

A node MUST accept and MAY relay a DeltaEvent d iff all conditions hold:

A. Canonical encoding; size <= MAX_EVENT_BYTES.
B. parents length <= MAX_PARENTS and ops length in 1..MAX_OPS.
C. SIGCHECK(pk, H("sigmsg" || u32(epoch) || parents || ops || nonce_incl), sig) == true.
D. Every op is well-typed; payload encoding matches vtype and respects limits in Section 7.3.
E. The node knows the canonical checkpoint root for epoch e-1 (or genesis for e=0) and can compute Threshold T_e; and work-validity holds: z(d,e) >= reqbits(d,e).

Nodes MUST reject otherwise.

Nodes SHOULD deduplicate by event id and MUST NOT relay duplicates.

10.2 AcceptCheckpoint

A node MUST accept and MAY relay a CheckpointEvent q iff:

A. q.epoch is either the next epoch after the highest epoch for which the node knows a canonical checkpoint, or within a small bounded lookahead window (implementation MAY accept within [current_epoch, current_epoch+1]).
B. SIGCHECK(pk, H("cpsig" || all_checkpoint_fields_except_sig), sig) == true.
C. VerifyCheckpointProof(q.epoch, q.prev_root, q.batch_commit, q.cut_commit, q.root, q.meta, q.proof) == true.
D. q.prev_root equals root_star(q.epoch-1) once a canonical checkpoint for epoch-1 is known.

Nodes MUST reject otherwise.

11. Canonical checkpoint selection

11.1 Candidate set

For epoch e, let Q_e be the set of accepted checkpoints with epoch == e and valid proof.

11.2 Meta format (normative)

meta is the canonical encoding of:

meta.count_data:u16
meta.count_rank:u16
meta.count_gov:u16
meta.count_reserved:u16
meta.surplus_sum:u64
meta.batch_commit:bytes (must equal q.batch_commit)
meta.cut_commit:bytes (must equal q.cut_commit)

Counts MUST equal the number of included deltas of each category in the checkpoint.

surplus_sum MUST equal the sum over all included deltas of surplus(d,e) truncated to 16 bits each and summed in u64.

The checkpoint proof MUST attest all meta fields.

11.3 Score function

ScoreQ(q) MUST be computable from q.meta alone in constant time.

ScoreQ(q) = meta.surplus_sum

Tie-breaker: smaller H(q.root || H(q.pk)) wins.

11.4 Canonical selection

CanonicalCheckpoint(e) is the q in Q_e with maximal ScoreQ(q); ties broken by the tie-breaker.

Once selected, q.root defines root_star(e) and becomes the anchor for reads and for deriving Threshold parameters for epoch e+1.

11.5 Practical finality

A node SHOULD treat root_star(e) as stable after it has observed CanonicalCheckpoint(e) and at least one accepted checkpoint for epoch e+1 that references it as prev_root.

12. Admission caps and threshold update

12.1 Inclusion caps

For epoch e, a checkpoint MUST include at most CAP[cat] deltas per category cat. These caps are protocol constants.

12.2 Threshold update rule

Let T_e[cat] be the threshold used to validate deltas in epoch e.

Genesis: T_0[cat] = T0[cat] for all categories.

For e >= 0, after root_star(e) is established, compute T_{e+1}[cat] by:

err = count_e[cat] - TARGET[cat]
step = sign(err) where sign(x) = +1 if x>0, 0 if x=0, -1 if x<0
T_{e+1}[cat] = clamp(T_e[cat] + step, TMIN[cat], TMAX[cat])

count_e[cat] is taken from meta of CanonicalCheckpoint(e).

This rule is deterministic and MUST be used by all compliant implementations.

13. Batch and cut commitments

13.1 Batch commitment

batch_commit is a Merkle root over the sorted list of included delta event ids (ascending bytes). The list length MUST be <= Σ CAP[cat].

13.2 Cut definition and commitment

Let B be the included delta id set of the checkpoint, and let the induced parent relation be edges (x -> y) where y lists x as a parent and both x,y are in B.

Define Frontier(B) as the subset of B that has no outgoing edge in the induced relation (maximal elements).

cut_commit is a Merkle root over the sorted list of Frontier(B) ids (ascending bytes).

The proof MUST attest that cut_commit matches Frontier(B) computed from the included deltas’ parent lists.

14. State evaluation semantics for checkpoints

14.1 Deterministic fold order

Because join is commutative and idempotent, the mathematical result is order-independent. For circuit determinism and reproducibility, the proof system MUST fold included deltas in ascending order of delta id, and within each delta, ops in ascending order of (k, vtype, payload bytes).

14.2 Applying ops

For each op (k, vtype, payload):

decode payload to DeltaValue in X_k
state_value[k] = join_k(state_value[k], DeltaValue)

14.3 Key typing

Key typing is determined by the first successful op for that key in the fold order within the checkpoint and MUST remain consistent thereafter. If a later op attempts a different vtype for the same key, the checkpoint proof MUST fail.

15. Reads

15.1 Read anchor

A client MUST select an epoch e and use root_star(e) as the read anchor.

15.2 Membership proof

To read key k:

obtain (value, proof) from any source
verify VerifyMembership(root_star(e), k, value, proof) == true

Clients MUST NOT trust unverified values.

15.3 Optional non-membership

If VerifyNonMembership is available for the chosen root structure, clients MAY use it to verify absence.

16. Derived views (deterministic, non-authoritative)

Derived views are deterministic functions of a checkpoint root and protocol constants. They are not consensus state and MUST NOT be written as authoritative values.

16.1 Name resolution (PTR)

For (scope, name), let k = KEY("ptr", scope, name).
Let C be the CandidateSet_8 value at k.

ResolvePTR(root, scope, name, now_epoch) returns the best candidate c in C such that:
c.mode == ALIVE and c.until >= now_epoch.
If none exist, return NONE.

Best is determined by the "ptr" ordering in Section 7.3.

16.2 Position resolution (POS)

k = KEY("pos", H(pk)).
ResolvePOS returns the best candidate by "pos" ordering.

16.3 Indexer discovery (IDX)

k = KEY("idx", v) for a vertex v in {0,1}^256.
ResolveIDX returns the best candidate with until >= now_epoch by "idx" ordering.

16.4 Active command set (governance view)

Command declaration:
CMD key: KEY("cmd", cmd_name) of type SETONCE containing (obj_id, cost_vec, schema_id).

Command vote:
CMDVOTE key: KEY("cmdvote", cmd_name) of type COUNTER_CAP containing level.

ActiveSet(root) is defined as:

Consider all cmd_name where CMD exists.
vote(cmd) = CMDVOTE.level (default 0 if absent).
Select top K_ACTIVE commands by vote(cmd), ties broken by H(cmd_name) ascending.

K_ACTIVE is a protocol constant (u16).

16.5 Help/menu generation (self-describing client surface)

Schema declaration:
SCHEMA key: KEY("schema", schema_id) of type SETONCE containing an object id pointing to a canonical schema blob (CBOR or canonical JSON).

Schema vote:
SCHEMAVOTE key: KEY("schemavote", schema_id) of type COUNTER_CAP.

HelpMenu(root) is defined as:

1. Compute ActiveSet(root).
2. For each active command, read its schema_id from CMD value.
3. Resolve schemas by selecting highest SCHEMAVOTE.level (ties by H(schema_id)), then reading SCHEMA object.
4. Produce a deterministic menu/render plan for the client.

Clients SHOULD implement HelpMenu(root) and MUST ensure all displayed actions are derived from the verified root.

16.6 Unified signal field and HDR diffusion (routing and ranking)

This section defines a deterministic “signal” derived from included deltas of a checkpoint, and a bounded diffusion-based potential used for exploration and ranking. It replaces any need for separate KKT optimization state.

16.6.1 Signal vertex mapping

For an included delta id x in epoch e:

v(x) = H("sigv" || x) interpreted as a 256-bit vertex.
ord(x) = low64( H("sigo" || x) ) interpreted as u64.

Define mass(x) = surplus(x,e) as signed int64 (non-negative).

16.6.2 Signal sample window per vertex

Define a virtual multiset per vertex v:

SigWindow(v) contains entries (ord(x), item(x)) for all included delta ids x with v(x)=v, where
item(x) = H("item" || x) truncated to 32 bytes together with a deterministic encoding that allows recovering mass(x) and x in the proof system.

Because the protocol state types are idempotent but do not include a native sum type, this signal field is defined as a derived view. Implementations MAY materialize it into explicit "inj" WINDOW64 keys as an application choice, but MUST treat the derived definition as the source of truth for this spec’s HDR computations.

16.6.3 Local signal b(v)

b(v) is the sum of mass(x) over the WINDOW_MAX smallest (ord, item) entries of SigWindow(v), interpreted as int64.

This is a deterministic bounded aggregation (reservoir of strongest ord entries) and is well-defined even if many x map to the same v.

16.6.4 Neighbor averaging operator

For a function f: V -> Q32.32, define:

(P f)(v) = (1/DEGREE_D) * sum_{i=0..DEGREE_D-1} f(N_i(v))

All arithmetic is Q32.32 with rounding toward zero.

16.6.5 Potential lambda_hat

Let b_q(v) = b(v) promoted to Q32.32 by left shift 32 bits.

Define:

lambda_0 = b_q
lambda_hat = lambda_0
For t = 1..SIG_T:
lambda_t = P(lambda_{t-1})
lambda_hat += (SIG_RHO_NUM / SIG_RHO_DEN)^t * lambda_t

For SIG_RHO = 1/2, scaling is exact by right shift t bits in Q32.32.

16.6.6 Gradient for routing and ranking

For v and neighbor i:

grad_i(v) = lambda_hat(v) - lambda_hat(N_i(v))

Clients MAY use grad_i(v) to choose moves, prioritize queries, weight ranking scores, or bias traversal. Because SIG_T and DEGREE_D are constants, computing grad_i(v) requires a bounded number of OPEN operations if b(v) is materialized, or a bounded proof-assisted query if b(v) is served by indexers.

17. Standard operations (high-level op library)

These are convenience operations that compile to one DeltaEvent containing one or more join ops.

17.1 OBJ_PUT_ONCE(obj_id, blob_hash)
JOIN(KEY("obj", obj_id), SETONCE(blob_hash))

17.2 LOG_APPEND(scope, topic, ord, item)
JOIN(KEY("log", scope, topic), WINDOW64_add(ord, item))

17.3 TOP_PUSH(scope, metric, score, item)
JOIN(KEY("top", scope, metric), TOP32_add(score, item))

17.4 KEEP_LEVEL(obj_id, level)
JOIN(KEY("keep", obj_id), COUNTER_CAP(level))

17.5 CMD_DECLARE(cmd_name, obj_id, cost_vec, schema_id)
JOIN(KEY("cmd", cmd_name), SETONCE(obj_id || cost_vec || schema_id))

17.6 CMD_VOTE(cmd_name, level)
JOIN(KEY("cmdvote", cmd_name), COUNTER_CAP(level))

17.7 SCHEMA_DECLARE(schema_id, obj_id)
JOIN(KEY("schema", schema_id), SETONCE(obj_id))

17.8 SCHEMA_VOTE(schema_id, level)
JOIN(KEY("schemavote", schema_id), COUNTER_CAP(level))

17.9 PTR_OFFER(scope, name, ref, until, level)
owner = H(pk)
cand = Candidate(ALIVE, until, ref, owner, level, aux=0)
JOIN(KEY("ptr", scope, name), CANDIDATE_add(cand))

17.10 PTR_TOMBSTONE(scope, name, until, level)
owner = H(pk)
cand = Candidate(DEAD, until, ref=0, owner, level, aux=0)
JOIN(KEY("ptr", scope, name), CANDIDATE_add(cand))

17.11 POS_OFFER(pos, level)
owner = H(pk)
cand = Candidate(ALIVE, until=0, ref=pos, owner, level, aux=0)
JOIN(KEY("pos", owner), CANDIDATE_add(cand))

17.12 IDX_ADVERTISE(v, endpoint_hash, until, level)
owner = H(pk)
cand = Candidate(ALIVE, until, ref=0, owner, level, aux=endpoint_hash)
JOIN(KEY("idx", v), CANDIDATE_add(cand))

18. Networking and relay

18.1 Relay policy

Nodes MUST relay accepted CheckpointEvents subject to local rate limits.

Nodes SHOULD relay accepted DeltaEvents subject to admission and resource controls. To preserve property (C), nodes MUST be able to validate and decide relay using only the delta body plus known Threshold parameters.

18.2 Admission-gated relay

Nodes SHOULD prioritize relaying DeltaEvents that are work-valid (Section 9.6) and SHOULD drop others.

Nodes MAY additionally apply per-peer and per-subnet rate limits.

18.3 Bounded candidate buffers (recommended)

To bound inbound and storage regardless of network n, nodes SHOULD maintain per-category buffers of size CAP[cat] (or a small multiple) keyed by ticket lexicographic order, retaining the best tickets and dropping worse ones. This does not affect correctness of acceptance rules, only relay policy.

19. Security considerations

19.1 Sybil and spam

Work-validity ties each delta to epoch parameters via ticket(d,e), and required work increases with cost. This imposes a resource cost per delta independent of identity. Relay gating and bounded buffers limit per-node exposure.

19.2 Eclipse and partition

Canonical roots are selected by score among valid checkpoints. Partitions can cause temporary divergence. Clients SHOULD query multiple independent sources for canonical checkpoints and verify proofs.

19.3 Checkpoint grinding

Checkpoint score is surplus_sum attested by proof; to increase score, a producer must include more or higher-surplus deltas, which requires corresponding work. Tie-breakers are deterministic to avoid ambiguity.

19.4 State poisoning

All authoritative reads are anchored to canonical roots plus membership proofs. Unverified data MUST NOT be trusted.

19.5 Signal manipulation

The unified signal field is derived from surplus paid by included deltas. Manipulating the field requires paying surplus work. The diffusion potential is deterministic and bounded.

20. Compliance requirements

A v0.01 compliant implementation MUST:

A. Implement AcceptDelta and AcceptCheckpoint exactly as specified.
B. Implement the six standard value types and their join rules.
C. Implement canonical checkpoint selection and threshold update.
D. Implement membership verification against canonical roots.
E. Implement deterministic derived views (ResolvePTR, ResolvePOS, ResolveIDX, ActiveSet, HelpMenu, Unified Signal Diffusion).

21. Versioning

v0.01 is identified by the tuple of protocol constants (Section 4), standard tags (Section 6.2), value types (Section 7.3), acceptance rules (Section 10), canonical selection rules (Section 11), and threshold update rule (Section 12). Any change to these constitutes a new protocol version.

Appendix A. Rationale for the minimal primitive core (non-normative)

The indispensable state transition is pointwise join (JOIN). Delta emission introduces updates into the partial-order world. Checkpoint emission makes state derivation and reads fast and verifiable. Membership reads allow clients to verify values. Admission thresholds and surplus define bandwidth allocation and “fees” without mutable consensus. All other operations compile to JOIN ops plus derived-view functions.

Appendix B. Implementation notes (non-normative)

A. Proof system choice: a fixed-shape proof or accumulator-style proof can satisfy VerifyCheckpointProof while keeping verifier time bounded by protocol constants.
B. Root structure choice: a fixed-depth sparse Merkle tree offers bounded VerifyMembership cost.
C. Indexers: indexing is optional; IDX candidates allow discovery without out-of-band configuration.
D. Materializing the signal field: servers MAY precompute and serve b(v) for selected vertices to reduce client OPEN calls, but clients MUST treat all served values as untrusted unless verified against the canonical root.
