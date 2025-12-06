BEM Tool and Resource Layer v0.0.1
Tool Bus, Resource Catalog, and External Integration
(Integrated English ASCII Draft, aligned with BEM v0.0.1 core)
	0.	Scope and Goals

0.1 Purpose

This document defines the Tool and Resource Layer v0.0.1 for the Boolean Expert Machine (BEM) v0.0.1. It specifies:
	1.	How external tools (filesystems, databases, solvers, LLMs, simulators, environments, UI gateways) are represented and invoked from BEM.
	2.	A unified Tool Bus abstraction in SHARED memory for enqueueing tool requests with bounded fast-path cost.
	3.	A Resource Catalog that maps stable identifiers in U to logical resources such as files, directories, database connections, and external environments.
	4.	Execution rules ensuring that BEM fast-path complexity remains bounded and independent of external latencies while allowing rich external integration.
	5.	Profiles for filesystem and database integration, and hooks for other tool kinds, consistent with the BEM execution model, TEACH, PROVER, and PoX.

0.2 Non-goals

This document does not define:
	1.	Any particular filesystem implementation or POSIX semantics.
	2.	Any concrete database engine (PostgreSQL, ClickHouse, etc).
	3.	Any network protocol, RPC format, or host operating system API.
	4.	Any specific serialization format beyond requiring a deterministic, self-describing encoding for tool payloads (for example CBOR or an equivalent scheme).
	5.	Any particular external tool or task set; only the abstraction and constraints.

0.3 Assumptions
	1.	BEM core, ISA, and segments are as in BEM v0.0.1:
	1.	STATE, SHARED, TRACE, WORK, TEACH, PATCH_QUEUE, ENV, PROVER, etc.
	2.	Identifiers U with layout defined in the core spec.
	3.	COP instructions for co-processor calls with bounded cost.
	2.	TEACH, PoX, and PROVER may be implemented partly using external tools, but from BEM core’s point of view their interfaces remain as defined in the core spec.

0.4 Design goals

G1 (bounded fast-path)
Tool interactions from STEP_FAST are limited to O(1) enqueue operations into a bounded Tool Bus, with no busy-waiting or unbounded computation.

G2 (uniform external interface)
All external tools (filesystem, database, solver, LLM, environment, human-feedback gateway) share a common TOOL_REQ descriptor format and status protocol in SHARED.

G3 (stable resource handles)
Resources (files, directories, database connections, logical environments) are identified by U-space identifiers and accessed via O(1) slot tables, never by raw string paths or DSN strings on fast-path.

G4 (host implementation transparency)
The host runtime may implement tools in-process, out-of-process, or on remote machines without modifying BEM core semantics.

G5 (auditability)
Tool invocations and results are logged to TRACE and participate in the hash chain and Merkle trees defined in the core specification.

G6 (compatibility with higher layers)
The Tool and Resource Layer is suitable as a substrate for PROVER, external environments, and the Human Feedback Layer v0.0.1, without changing BEM’s fast-path cost model.
	1.	Identifiers and Classes

1.1 U-space reuse

BEM core defines a 32-bit identifier domain:

U = {0, 1, …, 2^32 − 1}

with bit layout:

u = [class(6) | ecc(6) | bucket(10) | local(10)]

where:
	1.	class ∈ [0, 63] is the object kind.
	2.	ecc ∈ [0, 63] are parity/ECC bits used for corruption detection.
	3.	bucket ∈ [0, 1023] is a 10-bit shard or routing index.
	4.	local ∈ [0, 1023] is a 10-bit sub-index within (class, bucket).

The Tool and Resource Layer reuses this structure without changing field widths. For resource-related classes, bucket is interpreted as a resource shard index, and local as an index within that shard.

1.2 Additional identifier classes

The following class tags in U are reserved (exact numeric values are configuration constants):
	1.	class = fs_node
Logical filesystem node (file, directory, symlink, volume marker, virtual node).
	2.	class = db_conn
Logical database connection or logical database handle.
	3.	class = tool_kind
Logical tool kind (filesystem tool, database tool, prover tool, LLM tool, environment tool, human-feedback tool, etc).
	4.	class = env_task (optional)
Logical external environment or external simulator task descriptor.

These classes coexist with expert, task, template, patch, and other classes from the BEM core.

1.3 Slot mappings

For each resource class, there is a slot mapping stored in SHARED:
	1.	FS_NODE table: index i ∈ [0, F)
Maps fs_node ids to concrete filesystem metadata.
	2.	DB_CONN table: index j ∈ [0, D)
Maps db_conn ids to database connection metadata.
	3.	TOOL_KIND table: index t ∈ [0, T)
Maps tool_kind ids to tool type roles and policies.

The mapping between a U id and a slot index is implementation-defined. A typical pattern:
	1.	For a given fs_node id u:
	1.	class(u) = fs_node.
	2.	bucket(u) encodes a shard index.
	3.	local(u) encodes an index within that shard.
	2.	FS_NODE slots are allocated such that (class=fs_node, bucket, local) uniquely identify one entry.

The Tool and Resource Layer requires:
	1.	For each active resource id u, there is exactly one slot in the corresponding table.
	2.	The host runtime maintains any auxiliary indices it needs; BEM core only relies on table slots and id equality.
	3.	Resource Catalog Segment

2.1 Overview

The Resource Catalog is a logical subset of SHARED that exposes constant-time access to resource metadata. It includes at minimum:
	1.	FS_NODE table for filesystem nodes.
	2.	FS_ALIAS table for frequently used filesystem nodes.
	3.	DB_CONN table for database connections.
	4.	TOOL_KIND table for known tool kinds.

Implementations may add a generic RESOURCE table, but FS_NODE, FS_ALIAS, DB_CONN, and TOOL_KIND are normative.

2.2 FS_NODE table

For each filesystem node slot i in [0, F):

FS_NODE[i] = (
id,         // U, class=fs_node
parent_id,  // U, class=fs_node or null id for root
kind,       // u8
backend_id, // u16
flags,      // u16
hash_name,  // u64
hash_path   // u64
)

Semantics:
	1.	id
	1.	Must be a U identifier with class = fs_node.
	2.	Must be unique among all active FS_NODE entries.
	2.	parent_id
	1.	Either a valid fs_node id or a distinguished null id representing a root.
	2.	Defines a logical tree or DAG; exact topology is host-defined.
	3.	kind
	1.	0 = regular file
	2.	1 = directory
	3.	2 = symbolic link
	4.	Other values are reserved for future kinds (for example virtual node, volume root).
	4.	backend_id
	1.	Identifies a logical filesystem backend or volume (for example HOST_FS0, OBJECT_STORE0).
	2.	Interpretation is host-defined.
	5.	flags (bitmask)
At minimum:
	1.	bit 0: read-allowed
	2.	bit 1: write-allowed
	3.	bit 2: append-only
	4.	bit 3: ephemeral (temporary data)
	5.	bit 4: sensitive (subject to stricter logging or access controls)
Other bits are configuration-dependent.
	6.	hash_name
	1.	64-bit hash of the basename (the last path component) under a fixed hash function.
	2.	Used for quick comparisons; not a security primitive.
	7.	hash_path
	1.	64-bit hash of the logical path (for example absolute or root-relative) under a fixed hash function.
	2.	Host runtime may use it to map to an OS path or an object-key.

The FS_NODE table does not mandate a particular directory layout. It only ensures that given an id, BEM can load metadata in O(1).

2.3 FS_ALIAS table

FS_ALIAS is a small alias table for frequently used filesystem nodes:

For each alias index k in [0, A):

FS_ALIAS[k] in U

Constraints:
	1.	FS_ALIAS[k] is either:
	1.	A valid fs_node id referencing some FS_NODE entry, or
	2.	A distinguished null id.
	2.	FS_ALIAS indices are used directly in CFG as small immediates.

Typical alias assignments:

0 = DATA_ROOT
1 = LOG_ROOT
2 = SNAPSHOT_ROOT
3 = CONFIG_ROOT
4 = TRACE_EXPORT_ROOT
etc.

CFG fragments may:
	1.	Load FS_ALIAS[k] into a register in one or two instructions.
	2.	Pass that id to tool requests without manipulating strings.

2.4 DB_CONN table

For each database connection slot j in [0, D):

DB_CONN[j] = (
id,          // U, class=db_conn
backend_id,  // u16
flags,       // u16
config_tag   // u32
)

Semantics:
	1.	id
	1.	Must be a U identifier with class = db_conn.
	2.	Unique among DB_CONN entries.
	2.	backend_id
	1.	Identifies logical DB backend (for example POSTGRES_MAIN, KV_STORE0).
	2.	Interpretation is host-defined.
	3.	flags (bitmask)
At minimum:
	1.	bit 0: read-allowed
	2.	bit 1: write-allowed
	3.	bit 2: transactional
	4.	bit 3: analytics / heavy query allowed
	4.	config_tag
	1.	Opaque 32-bit tag used by host runtime to look up connection strings, credentials, pools, etc.
	2.	BEM core treats it as an integer with no structure.

2.5 TOOL_KIND table

For each tool kind slot t in [0, T):

TOOL_KIND[t] = (
id,         // U, class=tool_kind
role_flags, // u32
policy_tag  // u32
)

Semantics:
	1.	id
	1.	Must be a U identifier with class = tool_kind.
	2.	Unique among TOOL_KIND entries.
	2.	role_flags (bitmask)
Recommended bits:
	1.	bit 0: filesystem tool
	2.	bit 1: database tool
	3.	bit 2: prover tool
	4.	bit 3: LLM tool
	5.	bit 4: environment tool
	6.	bit 5: human-feedback / UI gateway tool
Other bits are reserved.
	3.	policy_tag
	1.	Opaque policy identifier for host runtime (ratelimits, timeouts, sandboxing profile, etc).
	2.	BEM core does not interpret it.

2.6 Optional generic resource table

Implementations MAY introduce a generic RESOURCE table:

RESOURCE[k] = (
id,        // U
class_tag, // u8
backend_id,// u16
flags,     // u16
meta_ptr   // u64
)

but v0.0.1 does not depend on RESOURCE. It is informative and intended for implementations that want to unify FS_NODE and DB_CONN metadata routed through a single host mapping layer.
	3.	Tool Bus Segment

3.1 TOOL_REQ entry format

The Tool Bus is a bounded array of request descriptors in SHARED:

For each slot q in [0, Q):

TOOL_REQ[q] = (
tool_kind_id, // U, class=tool_kind
op,           // u16
flags,        // u16
resource_id,  // U
in_ptr,       // u64
in_len,       // u32
out_ptr,      // u64
out_cap,      // u32
status,       // u8
err_code,     // u16
reserved      // u8
)

Fields:
	1.	tool_kind_id
	1.	U identifier with class=tool_kind, selecting the logical tool family (filesystem, database, prover, environment, LLM, HITL gateway, etc).
	2.	op
	1.	Tool-specific operation code (for example FS_READ, DB_EXEC_QUERY).
	2.	The numeric values are defined per tool kind.
	3.	flags
	1.	Operation-specific flags as a bitmask.
	2.	Defined per tool kind, but must not affect the structure of TOOL_REQ.
	4.	resource_id
	1.	Logical resource being operated on.
	2.	For filesystem tools: fs_node id.
	3.	For database tools: db_conn id.
	4.	For environment tools: env_task id or another class as configured.
	5.	For tools that do not need a resource, a distinguished null id may be used.
	5.	in_ptr, in_len
	1.	Byte address and length of the request payload in SHARED (or a designated I/O buffer region logically part of SHARED/STATE).
	2.	Payload must be encoded deterministically (for example CBOR, fixed-struct encoding).
	3.	BEM core does not interpret payload contents beyond length bounds.
	6.	out_ptr, out_cap
	1.	Byte address and maximum length of the response buffer.
	2.	The tool must not write beyond out_cap bytes starting at out_ptr.
	7.	status
	1.	0 = empty (slot is free)
	2.	1 = pending (enqueued by BEM, not yet processed by tool)
	3.	2 = done (processed, response available)
	4.	3 = error (tool failed, see err_code)
	8.	err_code
	1.	Tool-specific error code, meaningful when status=3.
	2.	For status ∈ {0,1,2}, err_code SHOULD be 0.
	9.	reserved
	1.	Reserved for future use.
	2.	BEM MUST initialize reserved to 0 when creating a request.

3.2 Request lifecycle

The normative lifecycle of a tool request is:
	1.	Allocation
	1.	A BEM routine (mid-path or slow-path; or a bounded helper callable from fast-path) searches TOOL_REQ for a slot q with status = 0.
	2.	If no empty slot is available, the routine:
	1.	MAY skip creating the request and log a backpressure event in WORK or TRACE, or
	2.	MAY overwrite a low-priority pending request according to a policy in WORK.
Overwrite policies are implementation-specific and not mandated by v0.0.1.
	2.	Initialization
	1.	The routine writes:
	1.	TOOL_REQ[q].tool_kind_id = chosen tool kind id.
	2.	TOOL_REQ[q].op           = chosen operation code.
	3.	TOOL_REQ[q].flags        = chosen flags.
	4.	TOOL_REQ[q].resource_id  = resource handle or null id.
	5.	TOOL_REQ[q].in_ptr       = payload pointer.
	6.	TOOL_REQ[q].in_len       = payload length.
	7.	TOOL_REQ[q].out_ptr      = response buffer pointer.
	8.	TOOL_REQ[q].out_cap      = response buffer capacity.
	9.	TOOL_REQ[q].err_code     = 0.
	2.	TOOL_REQ[q].reserved     = 0.
	3.	Enqueue
	1.	The routine sets TOOL_REQ[q].status = 1 (pending).
	2.	This transition marks the request as visible to external workers.
	4.	Processing by external worker
	1.	External workers scan TOOL_REQ entries for status=1.
	2.	For each q:
	1.	Read tool_kind_id, op, flags, resource_id, in_ptr, in_len, out_ptr, out_cap.
	2.	Map resource_id to a concrete host resource via Resource Catalog and host configuration.
	3.	Decode the payload at in_ptr.
	4.	Execute the host-level operation (filesystem I/O, DB query, solver call, LLM inference, environment step, HITL UI action, etc).
	5.	Encode the response into out_ptr, writing at most out_cap bytes.
	6.	Set TOOL_REQ[q].err_code if an error occurred.
	7.	Set TOOL_REQ[q].status to:
	1.	2 = done, if operation succeeded.
	2.	3 = error, if an error occurred.
	3.	Workers MUST NOT modify tool_kind_id, op, flags, resource_id, in_ptr, in_len, out_ptr, out_cap.
	5.	Completion handling
	1.	Mid-path or slow-path BEM routines scan TOOL_REQ entries for status ∈ {2,3}.
	2.	For each such q:
	1.	Decode response from [out_ptr, out_ptr + used_bytes) as defined by the tool.
	2.	Update STATE, SHARED, WORK, or TEACH as appropriate.
	3.	Log the tool invocation and outcome to TRACE (see Section 7).
	4.	Set TOOL_REQ[q].status = 0 (empty) when processing is complete.

3.3 COP binding

The BEM ISA includes a generic co-processor instruction:

COP op_id, rs_arg, rd_res

The Tool and Resource Layer binds certain op_id values to Tool Bus operations. At minimum:
	1.	COP_TOOL_ENQUEUE (normative)

Semantics:
	1.	Input:
	1.	rs_arg: pointer to a small descriptor in SHARED or STATE containing:
	1.	tool_kind_id
	2.	resource_id
	3.	op
	4.	flags
	5.	in_ptr, in_len
	6.	out_ptr, out_cap
	2.	Output:
	1.	rd_res: integer result:
	1.	≥ 0: the chosen TOOL_REQ index q.
	2.	< 0: an error code (for example −1 = no free slot, −2 = invalid tool_kind_id).

The COP implementation:
	1.	Searches TOOL_REQ for a free slot (status=0) in bounded time.
Q is a configuration constant and small enough that an O(Q) scan is acceptable.
	2.	On success, initializes TOOL_REQ[q] and sets status=1.
	3.	Returns q in rd_res, or a negative error code on failure.

Fast-path code MAY inline the equivalent logic as long as:
	1.	The per-step cost remains bounded by C_step_max.
	2.	The externally observable semantics of TOOL_REQ are preserved.
	3.	Filesystem Integration Profile v0.0.1

4.1 Filesystem tool kind

A distinguished tool_kind id TOOL_FS is reserved for filesystem operations. Its TOOL_KIND entry MUST have:
	1.	role_flags bit “filesystem tool” set.
	2.	policy_tag configured according to deployment needs.

4.2 Path resolution

BEM core MUST NOT manipulate arbitrary path strings on fast-path. Path resolution is a host responsibility.
	1.	At initialization or snapshot restore time:
	1.	The host runtime populates FS_NODE and FS_ALIAS from a configuration source (for example a static mapping from logical names to OS paths or object keys).
	2.	BEM code obtains fs_node ids by:
	1.	Loading FS_ALIAS[k] for common locations, or
	2.	Reading ids from configuration stored in SHARED, or
	3.	Following parent-child relationships in FS_NODE in bounded loops on mid-path or slow-path.
	3.	Filesystem TOOL_REQ entries use resource_id = fs_node id.
The host runtime maps (backend_id, id, hash_path) to a concrete OS-level path or object key.

4.3 Filesystem operations

For TOOL_FS, the following op codes are defined:

0 = FS_STAT
1 = FS_READ
2 = FS_WRITE
3 = FS_LIST

Payload layouts are defined as follows (normative default; implementations may extend but not change existing fields):
	1.	FS_STAT
	1.	in payload: optional flags; MAY be empty.
	2.	out payload: fixed-size struct:
	1.	size_bytes (u64)
	2.	mtime_seconds (u64)
	3.	kind (u8)
	4.	flags (u16)
	5.	reserved (padding)
	2.	FS_READ
	1.	in payload:
	1.	offset (u64)
	2.	length (u32)
	2.	Host runtime:
	1.	Reads up to length bytes from file at offset.
	2.	Writes min(length, out_cap) bytes into [out_ptr, out_ptr + used).
	3.	Encodes used_bytes (u32) at the start of the out payload (or in a small header), followed by raw bytes.
	3.	FS_WRITE
	1.	in payload:
	1.	offset (u64)
	2.	length (u32)
	3.	data is placed directly at in_ptr+header_size or in a separate buffer as defined by convention.
	2.	Host runtime:
	1.	Writes up to length bytes from the payload to the file at the given offset.
	2.	Out payload: number of bytes successfully written (u32) and an optional status code.
	4.	FS_LIST
	1.	in payload:
	1.	options bitmask (recursive, include_dirs, include_files)
	2.	max_entries (u32)
	2.	out payload:
	1.	entry_count (u32)
	2.	For each entry:
	1.	child_id (U, fs_node)
	2.	kind (u8)

FS_READ and FS_WRITE MUST respect FS_NODE.flags; if a write is attempted on a read-only node, the tool MUST set status=3 and an appropriate err_code.

4.4 Caching

BEM may treat data read via FS_READ as cached in STATE or SHARED. Caching policy (size limits, eviction) is implementation-specific and not part of this spec. Caching MUST NOT change TOOL_REQ semantics.
	5.	Database Integration Profile v0.0.1

5.1 Database tool kind

A distinguished tool_kind id TOOL_DB is reserved for database operations. Its TOOL_KIND entry MUST set a “database tool” bit in role_flags.

5.2 DB connections

DB_CONN entries are initialized by the host runtime using deployment configuration. BEM code uses DB_CONN[j].id as resource_id in TOOL_REQ entries for TOOL_DB.

5.3 Database operations

For TOOL_DB, the following op codes are defined:

0 = DB_EXEC_QUERY
1 = DB_META  (optional; schema/introspection)

DB_EXEC_QUERY:
	1.	resource_id: db_conn id.
	2.	in payload: encoded query object:

QUERY_REQ = (
query_kind,  // u8 (0=text,1=logical_op,2=prepared_id)
limit,       // u32
flags,       // u32 (read-only, transactional hints)
payload…   // query text or logical description, deterministically encoded
)
	3.	out payload: encoded result summary:

QUERY_RES = (
row_count,   // u32 (rows returned or affected)
col_count,   // u16
meta_flags,  // u16
data…      // results encoded in fixed schema or small prefix
)

Implementations are encouraged to provide:
	1.	Compact, fixed-schema encodings (for example fixed-size columns, small row slices).
	2.	Aggregated metrics (row_count, latency, estimated cost) that BEM can use directly as features or rewards.

DB_META is optional and MAY expose schema information. Its exact payload is implementation-defined.

5.4 DB as environment or shared memory

Two main usage patterns:
	1.	Slow shared memory or knowledge base
	1.	BEM queries DB sporadically to fetch parameters, coefficients, or knowledge.
	2.	Results are written into SHARED and later consumed by bandits, TEACH, or environment drivers.
	2.	External environment or simulator
	1.	TEACH templates may define tasks whose transition dynamics are driven by DB queries.
	2.	The environment driver uses TOOL_DB to fetch next-state / reward information asynchronously.
	3.	Fast-path only consumes ready results, never blocking on DB latency.
	3.	Other Tool Kinds

6.1 Prover tool

A tool_kind id TOOL_PROVER MAY be defined for external SAT/SMT/CEGIS engines backing the PROVER interface in the core spec.
	1.	PROVER’s SAT_CHECK and CEGIS calls may be implemented by:
	1.	Enqueueing TOOL_REQ entries with tool_kind_id=TOOL_PROVER and appropriate op codes.
	2.	Mid/slow-path workers translating these to concrete solver calls.
	3.	Passing results back to PROVER’s logical interface.
	2.	From the perspective of BEM core, PROVER still exposes SAT_CHECK/CEGIS as in Section 6.1 of the core spec. Whether PROVER is backed by TOOL_PROVER or inline code is an implementation choice.

6.2 Environment tool

A tool_kind id TOOL_ENV MAY be used for external environments and simulators:
	1.	resource_id may be an env_task id (class=env_task) or a task-specific handle.
	2.	op codes could include:
	1.	ENV_RESET
	2.	ENV_STEP
	3.	ENV_INFO
	3.	These calls integrate with TEACH by:
	1.	Storing env descriptors in ENV or SHARED.
	2.	Using TOOL_ENV as the transport to host-level simulators.

The spec does not fix these formats; they must follow TOOL_REQ semantics.

6.3 LLM and HITL gateway tools

A tool_kind id TOOL_LLM MAY be used to reach external language models, and a tool_kind id TOOL_HINL MAY be used as a gateway for Human Feedback Layer v0.0.1.
	1.	TOOL_LLM payloads SHOULD be deterministic and schema-based (prompt, parameters, truncation limits).
	2.	TOOL_HINL payloads SHOULD wrap HINL_TASK creation and updates, but the Human Feedback Layer spec is the normative reference for HINL semantics.
	3.	Execution Model and Complexity Constraints

7.1 Fast-path constraints

Within STEP_FAST and other fast-path primitives, BEM MAY:
	1.	Read and write STATE and local SHARED fields.
	2.	Call COP_TOOL_ENQUEUE (or inline equivalent) to enqueue TOOL_REQ entries, with bounded cost.
	3.	Read previously produced tool results that are guaranteed ready by configuration (for example pre-loaded configuration data).

Fast-path MUST NOT:
	1.	Busy-wait for TOOL_REQ.status to change from 1 to {2,3}.
	2.	Poll TOOL_REQ in loops whose iteration count depends on external tool latency.
	3.	Depend on the completion of any specific TOOL_REQ in order to bound C_step_max.

Any reading of tool results on fast-path MUST be done under a static policy that either:
	1.	Only uses data that is known to be pre-populated or cached, or
	2.	Uses stale-but-safe defaults when data is not ready.

7.2 Mid-path responsibilities

Mid-path routines (executed periodically):
	1.	Scan TOOL_REQ for entries with status ∈ {2,3}.
	2.	For each completed entry q:
	1.	Decode the response payload.
	2.	Update STATE, SHARED, WORK, TEACH, or ENV descriptors.
	3.	Append a TRACE entry describing the tool call (Section 8).
	4.	Set TOOL_REQ[q].status = 0.
	3.	Run GRPO_LITE, TEACH updates, statistics aggregation, and simple scheduling decisions while respecting their own bounded cost constraints.

Mid-path MUST treat Q and the number of completed requests checked per tick as configuration-bounded to keep per-call cost finite.

7.3 Slow-path responsibilities

Slow-path routines (Level 2 and 3 in the core spec) are responsible for:
	1.	Structural patch generation and verification, including heavy use of PROVER and TOOL_PROVER.
	2.	PoX scoring and patch scheduling.
	3.	Snapshot creation and restore.
	4.	Offline analysis and bulk operations.

Slow-path MAY issue TOOL_REQ entries for large operations (for example long-running solver calls, bulk DB queries, batch environment rollouts). Its cost does not affect C_step_max as long as all fast-path invariants are respected.

7.4 Complexity invariants

The Tool and Resource Layer MUST preserve:
	1.	Per-step fast-path cost bounded by a constant C_step_max that depends only on configuration (W, B, K_exp, etc), not on external latencies or TOOL_REQ backlog.
	2.	Q (the number of TOOL_REQ slots) finite and bounded. Searching for free or completed slots is bounded cost per mid/slow-path tick.
	3.	No external tool can cause an unbounded delay in STEP_FAST.
	4.	All modifications to FS_NODE, DB_CONN, TOOL_KIND, FS_ALIAS, and TOOL_REQ respect the core determinism and logging model (hash chain and Merkle roots).
	5.	Logging, Trace, and Replay

8.1 Tool trace entries

TRACE (as defined in the core spec) SHOULD include entries for tool invocations. For each TOOL_REQ slot q, when its processing is completed:

TRACE_tool_entry = (
time_or_step,   // step counter or logical time
task_id,        // current task_id_t or a reserved id
bucket,         // routing bucket or reserved
slot,           // expert slot or reserved
tool_kind_id,   // U, class=tool_kind
op,             // u16
resource_id,    // U
req_hash,       // u64 hash of request payload
res_hash,       // u64 hash of response payload
status,         // u8
err_code        // u16
)

Constraints:
	1.	HASH unit (from the core spec) MUST be used to compute req_hash and res_hash with a fixed hash function.
	2.	TRACE entries are integrated into the hash chain H exactly as other entries.

8.2 Use in analysis and replay

Offline tools MAY:
	1.	Reconstruct all tool invocations, their resources, and payload hashes.
	2.	Correlate tool outcomes with reward, regret, PoX scores, patch acceptance, and TEACH evolution.
	3.	Detect anomalies (unexpected error patterns, latency distributions).
	4.	Use the combination of TRACE and snapshots to replay or simulate BEM’s behavior with or without reissuing external tool calls.
	5.	Versioning and Compatibility

9.1 Version tag

Tool and Resource Layer v0.0.1 is identified by a version field stored in WORK or a configuration segment, for example:

TOOL_LAYER_VERSION = 0x00000001

Snapshots SHOULD record this version.

9.2 Forwards compatibility

Future versions MAY:
	1.	Add new tool kinds and new TOOL_KIND role_flags.
	2.	Add new op codes for existing tool kinds (FS, DB, PROVER, ENV, LLM, HINL).
	3.	Extend TOOL_REQ with additional fields by:
	1.	Using previously reserved bits/bytes, or
	2.	Appending new fields after reserved while preserving the layout and meaning of existing fields.

Compatibility rules:
	1.	A v0.0.1 runtime MUST treat unknown tool_kind_id or unknown op codes as errors:
	1.	Set TOOL_REQ[q].status = 3 (error).
	2.	Set TOOL_REQ[q].err_code to a generic “unsupported” value.
	2.	A v0.0.2+ runtime SHOULD be able to load snapshots written by v0.0.1 as long as:
	1.	FS_NODE, FS_ALIAS, DB_CONN, TOOL_KIND, and TOOL_REQ layouts are unchanged for existing fields.
	2.	Unused reserved bytes are treated as zero or ignored.

9.3 Relation to Human Feedback Layer

Human Feedback Layer v0.0.1 is defined on top of this Tool and Resource Layer:
	1.	TOOL_HINL is a tool_kind id with a role_flag indicating “human-feedback gateway”.
	2.	Human Feedback Layer defines HINL_TASK, HINL_* payloads, and how TOOL_REQ is used to interface with external UI workers.
	3.	This Tool and Resource Layer ensures that TOOL_HINL can operate with the same bounded fast-path and auditability guarantees as other tools.

End of BEM Tool and Resource Layer v0.0.1 tightened draft.
