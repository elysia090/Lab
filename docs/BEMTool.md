BEM Tool and Resource Layer v0.0.1
Tool Bus, Resource Catalog, and External Integration
	0.	Scope and Goals

0.1 Purpose

This document defines the Tool and Resource Layer v0.0.1 for the Boolean Expert Machine (BEM) v0.0.1. It specifies:
	1.	How external tools (filesystems, databases, LLMs, solvers, simulators, environments) are represented and invoked from BEM.
	2.	A unified Tool Bus abstraction in SHARED memory for enqueueing tool requests with bounded fast-path cost.
	3.	A Resource Catalog that maps stable integer identifiers in U to logical resources such as files, directories, and database connections.
	4.	Execution rules ensuring that BEM fast-path complexity remains bounded and independent of episode length while allowing rich external integration.
	5.	Profiles for filesystem and database integration consistent with the BEM execution model.

This layer is intended as an extension of the BEM core specification and assumes the existence of STATE, SHARED, TRACE, PROOF, WORK, and TEACH segments, and the BEM ISA including COP instructions.

0.2 Non-goals

This document does not define:
	1.	Any particular filesystem implementation or POSIX semantics.
	2.	Any concrete database engine (PostgreSQL, ClickHouse, etc).
	3.	Any network protocol, RPC format, or host operating system API.
	4.	Any particular encoding format beyond recommending a deterministic, self-describing encoding (for example CBOR) for tool payloads.
	5.	Any specific external tool or environment task set; only the abstraction and constraints.

0.3 Design goals

The Tool and Resource Layer is designed to satisfy:

G1 (bounded fast-path)
Tool interactions from the fast-path are limited to O(1) enqueue operations into a bounded Tool Bus, with no busy-waiting or unbounded computation.

G2 (uniform external interface)
All external tools (filesystem, database, solver, LLM, environment) share a common request descriptor format and status protocol in SHARED.

G3 (stable resource handles)
Resources (files, directories, database connections, logical environments) are identified by U-space identifiers and looked up via O(1) slot tables, never by raw string paths or DSN strings in fast-path.

G4 (host implementation transparency)
The host runtime may implement tools in-process, out-of-process, or on remote machines without modifying BEM core semantics.

G5 (auditability)
Tool invocations and results are logged to TRACE and can participate in the hash chain and Merkle trees defined in the core specification.
	1.	Identifier Classes and Resource IDs

1.1 U-space reuse

The BEM core defines a 32-bit domain:

U = {0, 1, …, 2^32 - 1}

with structure:

u = [class(6) | ecc(6) | shard(6) | local(14)]

where class, ecc, shard, and local are interpreted as small integers. The Tool and Resource Layer introduces additional classes in this domain.

1.2 New identifier classes

The following classes are reserved for the Tool and Resource Layer:
	1.	class = fs_node
Logical filesystem node (file, directory, symbolic link, or volume marker).
	2.	class = db_conn
Logical database connection or logical database handle.
	3.	class = tool_kind
Logical tool kind (filesystem tool, database tool, LLM tool, solver tool, environment tool, etc).
	4.	class = env_task (optional)
Logical environment or task descriptor for external simulators or task engines.

The exact integer values for class are chosen at configuration time and must be consistent across STATE, SHARED, and host runtime.

1.3 Slot mappings for resource classes

For each resource class, a slot mapping is defined:

slot_fs   : U -> {0..F-1} union {bottom}
slot_db   : U -> {0..D-1} union {bottom}
slot_tool : U -> {0..T-1} union {bottom}

where:
	1.	slot_fs(u) != bottom if and only if u is an allocated fs_node identifier.
	2.	slot_db(u) != bottom if and only if u is an allocated db_conn identifier.
	3.	slot_tool(u) != bottom if and only if u is an allocated tool_kind identifier.
	4.	For each mapping, u != v and both allocated implies slot_x(u) != slot_x(v).

Slot mappings and their inverses are stored in SHARED in the Resource Catalog segment. Rebalancing may change slot_x(u) but must preserve resource content and semantics of references that use identifiers u.
	2.	Resource Catalog Segment

2.1 Overview

The Resource Catalog is a logical subset of SHARED that exposes O(1) access to resource metadata. It defines at minimum:
	1.	FS_NODE table for filesystem nodes.
	2.	FS_ALIAS table for frequently used filesystem nodes.
	3.	DB_CONN table for database connections.
	4.	TOOL_KIND table for known tool kinds.

2.2 FS_NODE table

For each filesystem node slot i in [0, F):

FS_NODE[i] = (id, parent_id, kind, hash_name, hash_path, backend_id, flags)

where:
	1.	id in U with class = fs_node.
	2.	parent_id in U with class = fs_node or a distinguished null id for root.
	3.	kind in {0,1,2,…}
0 = regular file
1 = directory
2 = symbolic link
Other values are implementation-specific.
	4.	hash_name in Z_64
Hash of the basename (for example 64-bit hash of the last path component).
	5.	hash_path in Z_64
Hash of the logical path (absolute or root-relative).
	6.	backend_id in Z_16
Identifies which filesystem backend or volume to use (for example host filesystem, object store, in-memory store).
	7.	flags in Z_16
Bitmask indicating read-only, write-allowed, append-only, ephemeral, sensitive, or other policy flags.

Host runtime is responsible for mapping (backend_id, id) or (backend_id, hash_path) to concrete OS-level paths or storage locations.

2.3 FS_ALIAS table

Frequently used filesystem nodes are exposed via a small alias table:

FS_ALIAS[k] in U for k in {0..A-1}

Each FS_ALIAS[k] must satisfy:
	1.	FS_ALIAS[k] is either a valid fs_node identifier or a distinguished null id.
	2.	FS_ALIAS[k] is loaded into registers by simple LD from a fixed SHARED offset.

Typical usage:

0 = DATA_ROOT
1 = LOG_ROOT
2 = SNAPSHOT_ROOT
3 = CONFIG_ROOT
etc.

CFG programs can use small alias indices as immediates and resolve them to fs_node ids via a single memory load, enabling “instant” reference to key filesystem locations without string manipulation.

2.4 DB_CONN table

For each database connection slot j in [0, D):

DB_CONN[j] = (id, backend_id, flags, config_handle)

where:
	1.	id in U with class = db_conn.
	2.	backend_id in Z_16
Identifies which logical DB backend to use (for example POSTGRES_MAIN, ANALYTICS_DB).
	3.	flags in Z_16
Bitmask specifying read-only, write-allowed, transactional, etc.
	4.	config_handle in Z_64
Opaque handle used by host runtime to look up connection strings or credentials. BEM core treats it as an opaque integer.

DB connections may refer to SQL engines, key-value stores, or other logical stores.

2.5 TOOL_KIND table

For each tool kind slot t in [0, T):

TOOL_KIND[t] = (id, role_flags)

where:
	1.	id in U with class = tool_kind.
	2.	role_flags in Z_32
Encodes the logical type, for example:
bit 0: filesystem tool
bit 1: database tool
bit 2: prover tool
bit 3: LLM tool
bit 4: environment tool
etc.

The mapping from TOOL_KIND index to semantics is defined by host runtime and configuration.

2.6 Optional generic resource table

If needed, a generic RESOURCE table may unify FS_NODE and DB_CONN:

RESOURCE[k] = (id, class, backend_id, flags, meta_ptr)

where meta_ptr points into SHARED for class-specific metadata. For v0.0.1, FS_NODE and DB_CONN are sufficient and normative; RESOURCE is informative.
	3.	Tool Bus Segment

3.1 TOOL_REQ entry format

The Tool Bus is a bounded array of request descriptors in SHARED:

For each slot q in [0, Q):

TOOL_REQ[q] = (tool_kind_id, op, flags, resource_id, in_ptr, in_len, out_ptr, out_cap, status, err_code, reserved)

where:
	1.	tool_kind_id in U with class = tool_kind.
Selects the logical tool family (filesystem, database, prover, etc).
	2.	op in Z_16
Tool-specific operation code. Examples:
For filesystem:
0 = STAT, 1 = READ, 2 = WRITE, 3 = LIST
For database:
0 = EXEC_QUERY, 1 = PREPARE, 2 = FETCH, etc.
	3.	flags in Z_16
Operation-specific flags. For example:
For filesystem:
bit 0: allow partial read
bit 1: append mode
For database:
bit 0: transactional
bit 1: read-only
	4.	resource_id in U
The logical resource being operated on. For example:
fs_node id for filesystem operation.
db_conn id for database operation.
env_task id for environment operation.
	5.	in_ptr in Z_64
Pointer into SHARED or STATE where the request payload is stored. Payload encoding is implementation-defined but should be deterministic (e.g. CBOR). BEM ISA loads and stores treat in_ptr as a byte address in the conceptual address space for SHARED and STATE.
	6.	in_len in Z_32
Length in bytes of the request payload.
	7.	out_ptr in Z_64
Pointer into SHARED or STATE where the tool should write its response.
	8.	out_cap in Z_32
Maximum number of bytes that may be written starting at out_ptr.
	9.	status in Z_8
Status code controlled by BEM and host runtime:
0 = empty (slot is free)
1 = pending (enqueued by BEM, not yet processed by tool)
2 = done (processed, response available)
3 = error (tool failed, see err_code)
	10.	err_code in Z_16
Tool-specific error code, valid when status = 3.
	11.	reserved in Z_8
Reserved for future use; must be initialized to 0 by BEM when creating a request.

3.2 Tool request lifecycle

A tool request follows this lifecycle:
	1.	Allocation
BEM searches TOOL_REQ for a slot q with status = 0 (empty).
If no such slot exists, it may:
a) Skip creation and log a backpressure event, or
b) Overwrite a low-priority pending request according to a policy in WORK.
Overwriting is implementation-specific and not required by v0.0.1.
	2.	Initialization
BEM writes:
TOOL_REQ[q].tool_kind_id = desired tool kind id
TOOL_REQ[q].op           = chosen operation
TOOL_REQ[q].flags        = chosen flags
TOOL_REQ[q].resource_id  = resource handle
TOOL_REQ[q].in_ptr       = payload pointer
TOOL_REQ[q].in_len       = payload length
TOOL_REQ[q].out_ptr      = response buffer pointer
TOOL_REQ[q].out_cap      = response buffer capacity
TOOL_REQ[q].err_code     = 0
TOOL_REQ[q].reserved     = 0
	3.	Enqueue
BEM sets TOOL_REQ[q].status = 1 (pending).
This transition marks the request as visible to external workers.
	4.	Processing
External workers scan TOOL_REQ entries for status = 1, process them, and write:
	•	Response bytes to [out_ptr, out_ptr + used_bytes)
	•	TOOL_REQ[q].err_code if needed
	•	TOOL_REQ[q].status = 2 (done) or 3 (error)
Workers must not modify tool_kind_id, op, flags, resource_id, or in_ptr/in_len.
	5.	Completion handling
Mid-path or slow-path BEM routines scan for status in {2,3}, read responses, update STATE/SHARED as needed, log to TRACE, and then reset TOOL_REQ[q].status to 0 (empty) to reuse the slot.

3.3 COP binding

The BEM ISA includes a generic co-processor call:

COP op_id, rs_arg, rd_res

The Tool and Resource Layer binds certain op_id values to Tool Bus operations. For example:

op_id = COP_TOOL_ENQUEUE

Semantics:

Input:

rs_arg: pointer to a small descriptor in SHARED or STATE containing tool_kind_id, resource_id, op, flags, and payload location.

Output:

rd_res: integer result with the chosen TOOL_REQ index q or a negative error code if allocation failed.

Implementation:
	1.	BEM-CORE loads descriptor from [rs_arg].
	2.	BEM-CORE searches TOOL_REQ for an empty slot q in O(1) or O(Q) time with Q bounded and small.
	3.	If slot found, BEM-CORE populates TOOL_REQ[q] as above and sets status = 1.
	4.	rd_res is set to q.

Fast-path code may inline this logic without explicitly calling COP, as long as behavior is equivalent. The normative semantics is the enqueue of a TOOL_REQ entry with bounded cost.
	4.	Filesystem Integration Profile v0.0.1

4.1 Filesystem tool kind

A distinguished tool_kind id TOOL_FS is reserved for filesystem operations. TOOL_KIND entry for TOOL_FS must set role_flags with a bit indicating “filesystem”.

4.2 Path resolution

BEM does not manipulate raw path strings. Path resolution is a host responsibility and proceeds as follows:
	1.	FS_NODE and FS_ALIAS are initialized at startup or snapshot restore time.
Host runtime constructs FS_NODE entries using a configuration file (for example a JSON map from logical names to OS paths).
	2.	BEM code obtains a file_id either by:
a) Loading FS_ALIAS[k] for a common location, or
b) Following parent-child relationships in FS_NODE, or
c) Having file_id embedded in configuration.
	3.	TOOL_FS requests use resource_id = file_id.

Host runtime maps (backend_id, file_id) to concrete paths and uses OS-level APIs to perform IO.

4.3 Filesystem operations

For TOOL_FS, op codes are defined as:

0 = FS_STAT
1 = FS_READ
2 = FS_WRITE
3 = FS_LIST

Request payload formats (suggested):
	1.	FS_STAT
in payload: empty or small flags.
out payload: fixed-size struct with size, mtime, kind, flags.
	2.	FS_READ
in payload: (offset, length) in bytes.
Host runtime reads up to length bytes from file at offset and writes into [out_ptr, out_ptr + min(length, out_cap)).
	3.	FS_WRITE
in payload: (offset, length, data) where data resides in a separate region; in_ptr and in_len describe data.
Host runtime writes data to the file at offset. out payload may contain number of bytes written or status.
	4.	FS_LIST
in payload: listing options (recursive, pattern, limit).
out payload: list of child fs_node ids or hashed names.

Fast-path must not wait for FS completion. Mid-path routines may poll for TOOL_REQ entries with tool_kind_id = TOOL_FS and status in {2,3}.

4.4 Caching

BEM can treat data read via FS_READ as cached in STATE or SHARED. Policy variables:
	1.	Max cached bytes.
	2.	Eviction policy (LRU, FIFO, pinned for critical config files).
	3.	Flags in FS_NODE.flags to decide whether data may be cached.

Caching policy is implementation-specific and does not affect the Tool Bus semantics.
	5.	Database Integration Profile v0.0.1

5.1 Database tool kind

A distinguished tool_kind id TOOL_DB is reserved for database operations. TOOL_KIND entry for TOOL_DB must set role_flags with a bit indicating “database”.

5.2 DB connections

DB_CONN entries are initialized by host runtime using configuration files or environment. BEM core treats DB_CONN[j].config_handle as opaque and uses DB_CONN[j].id as resource_id in TOOL_REQ.

5.3 Database operations

Example op codes for TOOL_DB:

0 = DB_EXEC_QUERY
1 = DB_PREPARE (optional)
2 = DB_FETCH (optional)
3 = DB_META (schema, introspection)

DB_EXEC_QUERY:
	1.	resource_id: db_conn id.
	2.	in payload: encoded query object containing:
	•	optional query kind (text, prepared id, logical operation)
	•	parameters (encoded deterministically)
	•	result shaping hints (for example limits, projection).
	3.	out payload: encoded result summary or a small fixed prefix of rows, or an opaque handle to an out-of-band result set managed by the host.

The DB tool is encouraged to:
	1.	Normalize results into a compact fixed schema (for example rows of fixed-size fields or columnar fragments) to simplify BEM parsing.
	2.	Provide aggregated metrics (row count, latency, cost estimate) that BEM can use directly as reward or features.

5.4 DB as environment or shared memory

Two typical roles:
	1.	Slow shared memory or knowledge base
BEM uses TOOL_DB for occasional lookups or updates; results are written into SHARED and then used in O(small) features for routing, bandits, or TEACH.
	2.	Environment for tasks
Task templates in TEACH may specify DB operations as part of their dynamics. Rewards r_t may be defined from DB responses (for example constraint satisfaction, number of rows matching a condition, query cost minimization). Tool requests remain asynchronous and are integrated via mid-path updates.
	3.	External Worker Model

6.1 Worker responsibilities

External workers are processes or threads that:
	1.	Have read-write access to SHARED and TRACE segments, or to a synchronized representation of these segments.
	2.	Periodically scan TOOL_REQ for entries with status = 1 (pending).
	3.	For each such entry:
a) Interpret tool_kind_id and op.
b) Map resource_id to concrete host-level resource using Resource Catalog and host configuration.
c) Decode request payload at in_ptr with length in_len.
d) Execute the corresponding host-level operation (filesystem IO, DB query, solver call, LLM invocation, environment step, etc).
e) Encode the response into out_ptr with capacity out_cap.
f) Set status to 2 (done) or 3 (error) and populate err_code if needed.

Workers must ensure:
	1.	Never writing beyond out_cap bytes.
	2.	Never modifying TOOL_REQ fields other than status, err_code, and response data.
	3.	Failing safely: on error, set status to 3 and a non-zero err_code.

6.2 Consistency and idempotence

Because BEM may re-read or replay logs, workers are encouraged to implement idempotent operations when possible. For example:
	1.	Writes identified by a unique tool request key derived from TRACE index and hash of payload.
	2.	Logical upsert semantics for DB operations.

Exact policies are implementation-specific but must not violate the Tool Bus semantics.

6.3 Security boundaries

The Tool and Resource Layer is agnostic to security boundaries, but typical deployments enforce:
	1.	Resource scoping via backend_id and flags.
	2.	Access control in host runtime, mapping resource_id to concrete OS paths or credentials only when allowed by policy.
	3.	Logging of tool invocations to TRACE for audit.
	4.	Execution Model and Complexity Constraints

7.1 Fast-path allowed operations

Within STEP_FAST and other fast-path routines, BEM may:
	1.	Read and write STATE and localized SHARED fields.
	2.	Call ANN_QUERY and other bounded co-processor operations as defined by core spec, provided their runtime cost is bounded by configuration constants.
	3.	Enqueue TOOL_REQ entries as described in Section 3, including:
a) Locating a free slot.
b) Writing tool_kind_id, resource_id, op, flags, in_ptr, in_len, out_ptr, out_cap.
c) Setting status to 1 (pending).

Fast-path must not:
	1.	Busy-wait for status to change from 1 to 2 or 3.
	2.	Perform polling loops whose iteration count depends on external tool latency.
	3.	Inspect out payloads that are not guaranteed to be ready by configuration.

7.2 Mid-path responsibilities

Mid-path routines (executed periodically):
	1.	Scan TOOL_REQ for entries with status in {2,3}.
	2.	For each completed entry:
a) Decode response payload from [out_ptr, out_ptr + used_bytes).
b) Update STATE, SHARED, and WORK (for example bandit priors, TEACH statistics, scheduler context).
c) Log tool invocation and outcome to TRACE, including tool_kind_id, op, resource_id, status, err_code, and hashes of request and response.
d) Reset TOOL_REQ[q].status to 0 (empty) once processing is complete.
	3.	Run GRPO_LITE updates, teacher updates, log aggregation, and template extraction as defined in core spec.

Mid-path may use approximate time or step counts to schedule its own execution but must maintain bounded per-call costs.

7.3 Slow-path responsibilities

Slow-path routines (Level 2 and 3) are responsible for:
	1.	Structural patch generation and verification.
	2.	PROVER and CEGIS calls, which themselves can use the Tool Bus with tool_kind_id = TOOL_PROVER.
	3.	PoX scoring and patch scheduling.
	4.	Snapshot and restore procedures.

Slow-path is allowed to run arbitrarily complex algorithms, as long as:
	1.	Fast-path remains bounded and independent of slow-path workload.
	2.	Only verified patches that satisfy VC(Delta) and PoX thresholds are applied to configuration.

7.4 Complexity invariants

Overall, the Tool and Resource Layer must preserve the following:
	1.	Per-step fast-path cost is bounded by a constant that depends only on configuration parameters, not on the number or latency of external tool requests.
	2.	TOOL_REQ size Q is finite and bounded; searching for free slots is bounded cost (for example Q is small or scan is capped).
	3.	No external tool can cause unbounded fast-path delay; external latencies affect only mid/slow-path scheduling and the availability of updated data.
	4.	Logging and Trace Integration

8.1 Tool logs

TRACE entries should include minimal but sufficient information to reconstruct tool interactions. For each tool invocation, BEM logs:
	1.	Time or step index.
	2.	task_id and context_hash at invocation.
	3.	tool_kind_id, op, resource_id, TOOL_REQ index.
	4.	A hash of request payload.
	5.	A hash of response payload once processed.
	6.	status and err_code.

Hashes are computed via the HASH unit and integrated into the log hash chain as in the core spec.

8.2 Use for replay and analysis

Offline analysis tools may:
	1.	Reconstruct which external operations were performed.
	2.	Correlate external behavior with BEM performance metrics, PoX scores, and bandit statistics.
	3.	Decide which external tools or backends to modify or optimize.
	4.	Example Control Flows

9.1 Reading a configuration file
	1.	A CFG fragment in mid-path wants to read a small configuration file at logical location CONFIG_ROOT / “env.json”.
	2.	Host runtime has initialized FS_ALIAS[3] = file_id for CONFIG_ROOT and a child FS_NODE for “env.json” with id env_file_id.
	3.	BEM mid-path routine:
a) Loads env_file_id into a register.
b) Allocates a buffer in SHARED at buf_ptr with capacity buf_cap.
c) Allocates a TOOL_REQ slot q.
d) Writes:
TOOL_REQ[q].tool_kind_id = TOOL_FS
TOOL_REQ[q].op           = FS_READ
TOOL_REQ[q].flags        = 0
TOOL_REQ[q].resource_id  = env_file_id
TOOL_REQ[q].in_ptr       = pointer to a small struct with offset=0, length=buf_cap
TOOL_REQ[q].in_len       = size of that struct
TOOL_REQ[q].out_ptr      = buf_ptr
TOOL_REQ[q].out_cap      = buf_cap
TOOL_REQ[q].status       = 1
	4.	External filesystem worker reads the entry, maps env_file_id to an OS path, reads the file, writes content into [buf_ptr, buf_ptr + used], sets status=2.
	5.	On next mid-path tick, BEM detects status=2, decodes the configuration, updates SHARED, logs the action, and resets status to 0.

9.2 Executing a database query as environment step
	1.	A task template describes an environment where the next observation depends on the result of a SQL query.
	2.	BEM mid-path or environment executor constructs a query object and encodes it at query_ptr, length query_len.
	3.	It allocates a TOOL_REQ slot with:
TOOL_REQ[q].tool_kind_id = TOOL_DB
TOOL_REQ[q].op           = DB_EXEC_QUERY
TOOL_REQ[q].resource_id  = db_conn_id from DB_CONN
TOOL_REQ[q].in_ptr       = query_ptr
TOOL_REQ[q].in_len       = query_len
TOOL_REQ[q].out_ptr      = result_ptr
TOOL_REQ[q].out_cap      = result_cap
TOOL_REQ[q].status       = 1
	4.	DB worker executes the query and writes a compact result summary into [result_ptr, result_ptr + used], sets status=2.
	5.	Mid-path reads the result, updates environment state in SHARED or STATE, computes reward r_t, and calls BANDIT_UPDATE_STEP.
	6.	Fast-path remains unaffected by the DB latency; it only processes new observations and rewards when available.
	7.	Versioning and Backwards Compatibility

10.1 Version tag

This document defines Tool and Resource Layer v0.0.1. Implementations should record this version in WORK or configuration metadata and in snapshot descriptors.

10.2 Backwards compatibility

Future versions may:
	1.	Add new op codes for TOOL_FS and TOOL_DB.
	2.	Add new tool kinds in TOOL_KIND.
	3.	Extend TOOL_REQ with additional fields, preferably by repurposing reserved bits or appending fields while maintaining existing layout.

Compatibility rules:
	1.	A v0.0.1 runtime must ignore unknown tool_kinds and unknown op values, treating them as errors.
	2.	A v0.0.2 or later runtime should be able to read snapshots created by v0.0.1 if TOOL_REQ, FS_NODE, FS_ALIAS, and DB_CONN layouts are unchanged for existing fields.

End of BEM Tool and Resource Layer v0.0.1.
