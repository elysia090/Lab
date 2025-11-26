Title
Text-Constrained Matching Core (TCM-Core)
Version v0.0.1 – Specification

Status
Draft, implementation-oriented, self-contained

Language
ASCII, English only
	0.	Scope, goals, non-goals

0.1 Scope

This document defines TCM-Core v0.0.1 as a generic assignment engine for problems of the form:

“Given patterns, documents, and capacity constraints, decide online which
patterns to assign to which documents at which time slots.”

The core provides:
	1.	A shared model for patterns, documents, time slots, and capacity constraints.
	2.	A text-based candidate generator (IndexCore) based on suffix-array-style indexing.
	3.	An offline planner (PLAN0) that computes per-entity weights and per-slot capacity profiles.
	4.	An online serving engine (SERVE0) that decides assignments for individual requests within
the capacities defined by PLAN0.

The design is domain-agnostic. Patterns and documents may represent ads and users, notifications and recipients, rules and events, tasks and workers, or any other entities that fit the assignment model.

0.2 Goals

TCM-Core is designed to:
	1.	Provide a single abstraction for “who gets what, when, and how much” under constraints.
	2.	Separate offline planning (global optimization) from online serving (low-latency decisions).
	3.	Allow text-based candidate generation using a shared index (suffix array + FM-index style).
	4.	Support capacity constraints on both pattern and document sides at multiple time scales.
	5.	Achieve per-request serving complexity approximately O(k log k), where k is the number
of candidate patterns for a given request.
	6.	Be simple enough in v0.0.1 to implement in a small system, while leaving room for later
versions to add Lagrangian, bandit, and HFT-style optimizations.

0.3 Non-goals

TCM-Core v0.0.1 does not:
	1.	Define any specific machine learning model for base relevance scores.
	2.	Provide guarantees beyond simple capacity compliance and greedy maximizing of a local score.
	3.	Handle general Markov decision processes or long-horizon reinforcement learning.
	4.	Express non-linear global constraints beyond simple capacity and per-slot limits.
	5.	Specify distributed systems details (deployment, sharding, fault tolerance).
	6.	Model and terminology

1.1 Entities

The core entities are:
	1.	Patterns P = {P_i | i in 1..K}
Each pattern represents something that can be assigned to documents:
	•	ad campaign, notification template, moderation rule, etc.
	2.	Documents D = {D_j | j in 1..M}
Each document represents a potential target of assignment:
	•	user, content item, page slot, event, etc.
	3.	Time slots S = {s_1, …, s_S}
A finite set of discrete time intervals covering the planning horizon:
	•	for example, hourly buckets over one day.

Assignments are defined at the granularity of (pattern, document, time slot).

1.2 Assignment variable

For each (i, j, s), define binary variable:

x_{i,j,s} = 1 if pattern P_i is assigned to document D_j in time slot s,
0 otherwise.

In many implementations, x_{i,j,s} is not materialized explicitly; assignments are decided online when a request arrives.

1.3 Objective

The core assumes the existence of a non-negative weight function:

w: P x D x S -> R_{>= 0}

where w(i, j, s) represents the utility (or expected utility) of assigning pattern P_i to document D_j in slot s. Examples:
	•	Expected click probability
	•	Expected revenue
	•	Relevance score
	•	Any scalar combining these

v0.0.1 does not constrain how w is computed. It may be a learned model, a heuristic function, or a constant.

1.4 Constraints

The core supports the following capacities:
	1.	Pattern-side total capacity
For each pattern i:
sum_{j,s} x_{i,j,s} <= c_i_total
	2.	Pattern-side per-slot capacity
For each pattern i and slot s:
sum_{j} x_{i,j,s} <= c_i[s]
	3.	Document-side total capacity
For each document j:
sum_{i,s} x_{i,j,s} <= C_j_total
	4.	Document-side per-slot capacity
For each document j and slot s:
sum_{i} x_{i,j,s} <= C_j[s]

All capacities are non-negative integers. Capacities may be omitted; omitted capacities are treated as “very large” or effectively infinite in the implementation.

Additional filters are handled as hard exclusion rules, not as numeric capacities. For example:
	•	Pattern i is only valid in certain countries.
	•	Document j is only eligible for specific pattern types.

These are encoded as pattern and document attributes (metadata) and applied as filters in candidate generation and serving.

1.5 Offline vs online decisions

The core splits the problem:
	•	Offline: PLAN0
Uses a coarse approximation to the global assignment problem to compute per-pattern, per-document,
and per-slot control parameters.
	•	Online: SERVE0
For each request r, given a concrete document and current time slot, selects at most L patterns
using a greedy scoring function guided by the offline parameters and updates remaining capacities.

	2.	Data model and configuration

2.1 Pattern representation

Each pattern P_i has:
	•	pattern_id: unique identifier
	•	text: UTF-8 or ASCII text, used for the text index
	•	optional embedding: fixed-length numeric vector
	•	budget:
	•	c_i_total (integer)
	•	c_i[s] for all s in S (optional; aggregated to match horizon)
	•	constraints:
	•	simple attributes (country, device, type, etc.)
	•	priority (integer or float; optional)

Formally, define:

PatternConfig_i = {
id: string,
text: string,
embedding: optional vector,
c_total: int,
c_slot: map<slot, int>,
attributes: map<string, string or list>,
priority: float
}

2.2 Document representation

Each document D_j has:
	•	doc_id: unique identifier
	•	text: UTF-8 or ASCII text, used for indexing and serving context
	•	optional embedding: fixed-length numeric vector
	•	capacity:
	•	C_j_total (integer; optional)
	•	C_j[s] for all s in S (optional)
	•	metadata: country, device, segments, etc.

Formally:

DocumentConfig_j = {
id: string,
text: string,
embedding: optional vector,
C_total: int or null,
C_slot: map<slot, int> or null,
attributes: map<string, string or list>
}

2.3 Time slots

The planning horizon is a contiguous interval [T_start, T_end). The implementation must:
	•	Choose a slot granularity (e.g. 1 hour, 5 minutes, 1 day).
	•	Partition the horizon into S slots, each represented by an index s in 0..S-1.
	•	Provide a mapping from wall-clock timestamp to slot index.

The mapping is:

slot_index: timestamp -> s in {0, …, S-1}

The horizon and slot mapping are input to PLAN0.

2.4 Base relevance function

TCM-Core assumes a base relevance function:

base_rel: (P_i, D_j, request_context) -> float

The function may use:
	•	Precomputed features (pattern and document attributes, embeddings).
	•	Request-specific context (e.g. current page, query string).

v0.0.1 does not define base_rel. It is supplied by the embedding/ML subsystem or the caller. TCM-Core only requires:
	•	base_rel(i, j, ctx) >= 0
	•	higher values mean more desirable.

	3.	IndexCore v0.0.1

3.1 Purpose

IndexCore provides text-based candidate generation in both directions:
	1.	Given a pattern P_i, find candidate documents D_j that may be relevant.
	2.	Given a document D_j or request context, find candidate patterns P_i.

The implementation uses a static text index for large, relatively stable data, plus optional dynamic structures for recent updates.

3.2 Static text index for documents

For documents D_j, the implementation SHOULD support a static index:
	•	Concatenate all document texts into a single string T:
T = D_1_text # D_2_text # … # D_M_text $
where # and $ are separator symbols not appearing in any document text.
	•	Build a suffix array SA over T using a linear-time or near-linear-time algorithm
(e.g. SA-IS).
	•	Build the Burrows–Wheeler Transform (BWT) of T and an FM-index on top of it.
	•	Build a bitvector B_doc_start of length |T| such that B_doc_start[pos] = 1
if and only if pos is the start of some document D_j.
	•	Support rank1(B_doc_start, pos) in O(1) time (succinct bitvector structure).

Given a pattern query string q, the FM-index MUST support:
	•	Finding the range [l, r) of suffix array positions where q is a prefix.
	•	For each k in [l, r), mapping SA[k] -> pos, then pos -> doc_id via rank1.

The complexity target is:
	•	O(|q|) time to find [l, r).
	•	O(occ) time to enumerate occ occurrences and map to documents.

3.3 Static text index for patterns (optional)

The implementation MAY build a similar index over pattern texts to support:
	•	Document-to-pattern candidate generation based on text match.
	•	Clustering or analysis over patterns.

This is optional in v0.0.1. If present, the same suffix array and FM-index principles apply.

3.4 Embedding and ANN index (optional)

The implementation MAY support an embedding index:
	•	Each document D_j has vector embedding e_j.
	•	Each pattern P_i has vector embedding u_i.
	•	The system builds an approximate nearest neighbor (ANN) index:
	•	For documents: doc-ANN over e_j
	•	For patterns: pattern-ANN over u_i

Given a query embedding v, the ANN tool returns top-K nearest neighbors. The implementation MAY use HNSW, IVF-PQ, or any other suitable ANN approach.

3.5 Candidate generation API (logical)

IndexCore exports two logical operations:
	1.	SearchDocsForPattern(i):
Input: PatternConfig_i, optional K_docs
Output: Set of document IDs CandidateDocs(i)
Implementation:
	•	Use pattern text and/or embedding to query the index.
	•	Apply filters based on attributes (country, type, etc.).
	•	Return up to K_docs candidates.
	2.	SearchPatternsForDoc(j, ctx):
Input: DocumentConfig_j, request_context ctx, optional K_patterns
Output: Set of pattern IDs CandidatePatterns(j, ctx)
Implementation:
	•	Use document text, context text, and/or embedding.
	•	Apply filters based on attributes and current time.
	•	Return up to K_patterns candidates.

These APIs are conceptual; concrete RPC or function signatures may differ.
	4.	Offline planner PLAN0 v0.0.1

4.1 Role

PLAN0 computes:
	1.	Per-pattern weights alpha_i.
	2.	Per-document weights beta_j (optional).
	3.	Per-pattern per-slot target capacities c_i_slot[s], consistent with c_i_total.
	4.	Any additional static parameters needed for SERVE0.

PLAN0 runs infrequently (for example, once per horizon or every few hours). Its output is immutable until the next run and is identified by a plan_snapshot_id.

4.2 Input

PLAN0 takes:
	•	The list of patterns with PatternConfig_i.
	•	The list of documents with DocumentConfig_j.
	•	Time horizon and slot configuration.
	•	A candidate generator IndexCore.SearchDocsForPattern.
	•	A base relevance estimator w0(i, j), independent of slot.

v0.0.1 does not require w0 to depend on slot. Slot-level variation is expressed by c_i[s].

4.3 Candidate edge generation

PLAN0 MUST generate a candidate bipartite graph:

E_cand subset of P x D

For each i in P:
	•	Call CandidateDocs(i) = IndexCore.SearchDocsForPattern(i).
	•	For each j in CandidateDocs(i), add edge (i, j) to E_cand.

Duplicates MUST be removed.

The planner MAY cap the number of candidate docs per pattern to K_docs_plan to bound complexity.

4.4 Simplified global optimization

v0.0.1 uses a simplified global optimization:
	1.	Define edge weight w_ij = w0(i, j), e.g. base_rel averaged over history.
	2.	Solve the following relaxed problem without time slots:
maximize   sum_{(i,j) in E_cand} w_ij * x_ij
subject to sum_j x_ij <= c_i_total,   for all i
sum_i x_ij <= C_j_total,   for all j
0 <= x_ij <= 1
	3.	This is a linear program (LP) or a max-flow-like problem if capacities are integer.

For v0.0.1 the implementation SHOULD:
	•	Use a standard LP solver or a max-flow solver on a transformed network.
	•	Or apply a simple heuristic if a full solver is not available:
	•	For example, sort edges by w_ij and greedily allocate until capacities are filled.

The exact solution method is implementation-defined in v0.0.1.

4.5 Deriving pattern weights alpha_i

From the result x_ij (or the greedy heuristic):
	•	Compute total assigned volume per pattern:
assigned_i = sum_j x_ij

Define:

alpha_i = f(assigned_i, c_i_total)

A simple default:

alpha_i = 1.0 if c_i_total > 0
alpha_i = 0.0 if c_i_total = 0

A slightly more informed default:

utilization_i = assigned_i / max(1, c_i_total)
alpha_i = 1.0 + gamma * (1.0 - utilization_i)

where gamma is a tunable constant, and alpha_i is clamped to a reasonable range.

v0.0.1 does not fix the exact function; it only requires alpha_i >= 0 and monotonic in utilization_i (under-utilized patterns may be up-weighted).

4.6 Deriving per-slot pattern capacities c_i_slot[s]

Given c_i_total for each i, PLAN0 must split this total budget across slots:
	•	Input: an empirical or assumed traffic distribution p_s over slots s in S.
	•	For each pattern i:
target_i[s] = round(p_s * c_i_total)
	•	Adjust target_i[s] to ensure:
sum_s target_i[s] = c_i_total

If no traffic distribution is known, PLAN0 MAY use a uniform distribution:

p_s = 1 / |S|

The resulting target_i[s] become the per-slot capacities c_i[s] used by SERVE0.

4.7 Optional document weights beta_j

Optionally, the planner may compute per-document weights beta_j to reflect:
	•	Document importance
	•	Document scarcity
	•	Desired load balancing

For v0.0.1, beta_j MAY be simply set to 1.0 for all j.

4.8 Planner output

PLAN0 produces a PlanSnapshot:

PlanSnapshot = {
id: string,                    // plan_snapshot_id
horizon_start: timestamp,
horizon_end: timestamp,
slots: list,
alpha: map<pattern_id, float>,
beta: map<doc_id, float>,
c_slot: map<pattern_id, map<slot, int>>,
C_slot: map<doc_id, map<slot, int>>,  // may be derived from DocumentConfig
metadata: map<string, string>         // versioning, tuning parameters
}

SERVE0 must reference a PlanSnapshot by id.
	5.	Online serving SERVE0 v0.0.1

5.1 Role

SERVE0 is the online engine that:
	•	Receives a serve request r containing:
	•	doc_id or document context
	•	current timestamp
	•	optional filters
	•	limit L
	•	Selects up to L patterns to assign.
	•	Updates remaining capacities (pattern and document side).

SERVE0 aims to:
	•	Respect capacities (c_i_total, c_i[s], C_j_total, C_j[s]).
	•	Approximate maximization of local utility based on w(i, j, s).
	•	Maintain per-request latency within tight bounds.

5.2 Input per request

A serve request r contains:
	•	request_id: unique identifier
	•	doc_id: identifier of target document, or null if context-only
	•	context_text: optional text describing current context (page, query, etc.)
	•	timestamp: current time
	•	limit L: max number of patterns to return
	•	filters:
	•	allow list of pattern_id or pattern prefixes (optional)
	•	deny list of pattern_id or pattern prefixes (optional)
	•	plan_snapshot_id: optional, defaults to latest available snapshot

5.3 Preprocessing per request

SERVE0 MUST:
	1.	Map timestamp to slot index s using the slot mapping defined in the PlanSnapshot.
	2.	Resolve doc_id to DocumentConfig_j (or construct a temporary DocumentConfig if only context_text is given).
	3.	Fetch PlanSnapshot parameters (alpha_i, beta_j, c_i_slot[s], C_j_slot[s]) for the referenced plan_snapshot_id.

5.4 Candidate generation per request

SERVE0 MUST compute a candidate pattern set:

Cand(r) = CandidatePatterns(j, ctx)

where:
	•	j is doc_id (or a synthetic ID for context-only).
	•	ctx includes context_text, timestamp, and metadata.

This is done via IndexCore.SearchPatternsForDoc. The implementation MUST enforce:
	•	A soft or hard upper bound K_patterns on |Cand(r)| for latency control.
	•	Filters based on attributes (country, device, type, etc.).
	•	Filters from the request (allow/deny lists).

Candidates that violate static constraints (e.g. pattern not valid in this country) MUST be excluded.

5.5 Capacity state

SERVE0 maintains runtime state:
	•	remaining_total_i: remaining global capacity for pattern i
	•	remaining_slot_i[s]: remaining capacity for pattern i in slot s
	•	remaining_total_j: remaining global capacity for document j (optional)
	•	remaining_slot_j[s]: remaining capacity for document j in slot s (optional)

The state is initialized from:
	•	remaining_total_i := c_i_total
	•	remaining_slot_i[s] := c_i_slot[s]
	•	remaining_total_j := C_j_total (if defined)
	•	remaining_slot_j[s] := C_j[s] (if defined)

For v0.0.1, these counters MAY be maintained in a single process or per-shard. Exact consistency across distributed instances is out-of-scope.

5.6 Local scoring function

For each candidate pattern i in Cand(r), SERVE0 computes a score:

score(i, j, s, r) = base_rel(i, j, ctx)
* alpha_i
* beta_j
* budget_factor_i(s)

where:
	•	base_rel(i, j, ctx) is provided by the external scoring system.
	•	alpha_i is from the PlanSnapshot.
	•	beta_j is from the PlanSnapshot or 1.0.
	•	budget_factor_i(s) is computed from the utilization of pattern i in slot s.

A simple budget_factor_i(s) in v0.0.1:
	•	Let used_slot_i[s] = c_i_slot[s] - remaining_slot_i[s].
	•	Let utilization_slot_i[s] = used_slot_i[s] / max(1, c_i_slot[s]).
	•	Define:
budget_factor_i(s) = 1.0 if utilization_slot_i[s] <= 1.0
0.0 if utilization_slot_i[s] > 1.0

That is, assignments stop when the per-slot capacity is exhausted; before that, budget_factor_i(s) = 1.0. More sophisticated functions are allowed but not required.

Candidates i for which:

remaining_total_i <= 0 or remaining_slot_i[s] <= 0

MUST be excluded from scoring.

If document capacities are configured, SERVE0 MUST also check:

remaining_total_j > 0 and remaining_slot_j[s] > 0

Otherwise, the document is not eligible for additional assignments in this request.

5.7 Selection

Given:
	•	Candidate set Cand(r)
	•	Scores score(i, j, s, r) >= 0

SERVE0 MUST:
	1.	Filter candidates with positive score and sufficient capacity.
	2.	Select up to L candidates with the highest scores.

Selection may be implemented as:
	•	Full sort by score in descending order, then take the first L.
	•	Partial selection using nth_element or a small heap, to improve performance.

Tie-breaking is implementation-defined. A stable, deterministic tie-breaking rule (for example, by pattern_id) is RECOMMENDED.

5.8 Capacity updates

For each selected pattern i*:
	•	remaining_total_{i*} := remaining_total_{i*} - 1
	•	remaining_slot_{i*}[s] := remaining_slot_{i*}[s] - 1

If document capacities are configured:
	•	remaining_total_j := remaining_total_j - 1
	•	remaining_slot_j[s] := remaining_slot_j[s] - 1

The serve response MUST reflect the patterns actually assigned so that feedback events can be correlated.

5.9 Feedback

SERVE0 expects feedback events to update external models and, optionally, future PLAN runs. Feedback is not directly part of the assignment algorithm but is part of the system behavior.

Each feedback record should contain:
	•	request_id
	•	doc_id
	•	pattern_id
	•	event_type (impression, click, conversion, etc.)
	•	value (numeric reward)
	•	timestamp

v0.0.1 does not specify how feedback updates base_rel or PLAN0. Implementations MAY:
	•	Compute CTR or other metrics per pattern and per pattern-document pair.
	•	Feed these statistics into the next PLAN0 run.
	•	Train or update base_rel models.

	6.	Complexity and performance targets

6.1 Offline complexity

IndexCore:
	•	Building SA and FM-index over T of length N:
O(N) or O(N log N), depending on the algorithm.

PLAN0:
	•	Candidate edge generation:
O(sum_i |pattern_text_i| + sum_i occ_i)
where occ_i is the number of text matches, capped by K_docs_plan per pattern.
	•	Global optimization:
Depends on the LP or heuristic algorithm. For greedy:

- Sorting edges by w_ij:
    O(|E_cand| log |E_cand|)
- Greedy allocation:
    O(|E_cand|)



6.2 Online complexity per request

SERVE0:
	•	Candidate generation:
O(|context_text| + K_patterns)
(dominated by the index lookup plus simple filters)
	•	Scoring and selection:
	•	O(K_patterns) for score computation.
	•	O(K_patterns log K_patterns) for full sort, or O(K_patterns + L log K_patterns) for partial selection.

The implementation SHOULD choose K_patterns and L to keep per-request latency within the target bounds for the deployment platform (for example, p50 under a few milliseconds).
	7.	Conformance and extensibility

7.1 Required behaviors

A conformant v0.0.1 implementation MUST:
	1.	Support patterns and documents with capacities c_i_total and c_i[s] at minimum.
	2.	Implement IndexCore candidate generation in at least one direction (pattern→docs or doc→patterns), based on text matching or a comparable mechanism.
	3.	Provide PLAN0 that:
	•	Generates candidate edges using IndexCore.
	•	Produces at least (alpha_i, c_i[s]) for each pattern.
	4.	Provide SERVE0 that:
	•	Uses the PlanSnapshot parameters (alpha_i, c_i[s]) and capacity state.
	•	Selects up to L patterns greedily per request.
	•	Updates capacities accordingly.
	5.	Enforce hard capacity limits: patterns with exhausted global or slot capacity MUST not be selected.

7.2 Recommended behaviors

Implementations SHOULD:
	1.	Maintain per-document capacities when appropriate.
	2.	Use succinct data structures (rank/select bitvectors) where memory is tight and text size is large.
	3.	Keep base_rel as a separate model or module, so that TCM-Core can evolve independently of the scoring mechanism.
	4.	Persist PlanSnapshot and capacity state to allow recovery and audit.

7.3 Extensibility beyond v0.0.1

The following extensions are explicitly left for future versions:
	•	Lagrangian-based planner with dual variables and more advanced alpha_i definitions.
	•	Online primal-dual updates to adjust alpha_i and c_i[s] dynamically.
	•	Bandit algorithms to handle exploration/exploitation over pattern-document pairs.
	•	HFT-style low-latency tuning (lock-free, core pinning, bucketized budget factors).
	•	Multi-objective optimization (fairness, risk, etc.) expressed as additional constraints.

	8.	Security and safety considerations

8.1 Budget and abuse

Implementations MUST consider:
	•	Preventing capacities and budgets from being manipulated by untrusted callers.
	•	Isolating plan and capacity state from direct external modification.

8.2 Privacy

The specification does not mandate any particular handling of personal data. A deployment MUST:
	•	Ensure compliance with applicable privacy and data protection regulations.
	•	Avoid exposing PlanSnapshot contents, capacity state, or detailed index internals to unauthorized actors.

8.3 Robustness

Serving implementations SHOULD:
	•	Handle missing or invalid pattern or document configurations gracefully.
	•	Fallback to an empty result or a safe default when capacity state is inconsistent or missing.
	•	Ensure that failures in PLAN0 or IndexCore do not produce undefined behavior in SERVE0.

	9.	Summary

TCM-Core v0.0.1 defines:
	•	A general bipartite assignment model with capacities on patterns and documents across time slots.
	•	A text-based candidate generation subsystem (IndexCore) based on suffix array and FM-index concepts.
	•	A simple offline planner (PLAN0) that computes per-pattern weights and per-slot capacities.
	•	A greedy online serving engine (SERVE0) that respects capacities and maximizes local scores per request.

This specification is intended to be:
	•	Narrow in scope (assignment and capacities only).
	•	Modular (separating indexing, planning, and serving).
	•	Sufficiently precise to implement a working system and extend it in later versions.


Title
TCM-Core API
Version v0.0.1

Status
Draft, implementation-oriented, stable surface for TCM-Core v0.0.1

Language
ASCII, English only
	0.	Scope and goals

0.1 Scope

This document defines the HTTP API for TCM-Core v0.0.1.

The API exposes the following capabilities:
	1.	Managing documents (targets of assignment).
	2.	Managing patterns (assignable items).
	3.	Managing and inspecting plan snapshots (offline planner output).
	4.	Serving online decisions (SERVE0).
	5.	Recording feedback events.
	6.	Querying basic decision logs.

The API is intentionally narrow and mirrors the core TCM-Core concepts: documents, patterns, time slots, plan snapshots, and serve requests.

0.2 Non-goals

The v0.0.1 API does not:
	1.	Define authentication or billing. These are deployment-specific.
	2.	Provide UI, dashboards, or workflow orchestration.
	3.	Provide full-text search or analytics APIs over logs.
	4.	Define multi-tenant management endpoints. Tenanting is an internal concern.
	5.	Specify any particular machine learning models or their management lifecycle.
	6.	General conventions

1.1 Base URL and versioning

All endpoints in v0.0.1 are rooted at a common prefix:

/tcm/v0/…

The leading slash is mandatory. The “tcm” and “v0” segments are reserved.

Future versions MUST use different version segments (for example /tcm/v1) and MUST NOT change the behavior of /tcm/v0 endpoints in a backward-incompatible way.

1.2 Transport

The API is expected to run over HTTPS. HTTP semantics are as follows:
	1.	All write operations use HTTP POST.
	2.	Read-only operations use HTTP GET.
	3.	The server SHOULD respond with appropriate HTTP status codes:
	•	2xx for success
	•	4xx for client errors
	•	5xx for server errors

1.3 Content type and encoding

Requests with bodies MUST set:

Content-Type: application/json

Responses with JSON bodies MUST set:

Content-Type: application/json

All JSON is encoded as UTF-8. Field names and documented literal values MUST be restricted to ASCII characters.

1.4 Resource identifiers

The following identifiers are used throughout the API:
	1.	doc_id: string; unique per document within a tenant.
	2.	pattern_id: string; unique per pattern within a tenant.
	3.	plan_snapshot_id: string; unique per plan snapshot.
	4.	request_id: string; unique per serve request.

The server MUST treat doc_id and pattern_id as opaque strings, except for uniqueness and equality. Clients MAY use any ASCII-safe convention for their own IDs.

1.5 Time representation

All timestamps MUST be RFC 3339 strings in UTC, for example:

“2025-11-27T01:23:45Z”

Time slot keys in per-slot capacity maps MUST use the same format.

1.6 Error model

On error, the server MUST return a non-2xx HTTP status code and a JSON body of the form:

{
“error”: {
“code”: “string-machine-code”,
“message”: “Human readable message”,
“details”: { … optional arbitrary JSON … }
}
}

Examples of error codes:

“invalid_argument”
“not_found”
“conflict”
“plan_unavailable”
“internal”

The “message” field MUST be suitable for logs and debugging, not for end users. The “details” object MAY contain structured information such as field errors.
	2.	Endpoint overview

The v0.0.1 API defines the following endpoints:

Documents
POST /tcm/v0/documents:upsert
POST /tcm/v0/documents:delete

Patterns
POST /tcm/v0/patterns:upsert
POST /tcm/v0/patterns:delete

Plans
GET  /tcm/v0/plans
GET  /tcm/v0/plans/{plan_snapshot_id}
POST /tcm/v0/plans:run    (optional, see 4.3)

Serve (online decisions)
POST /tcm/v0/serve

Feedback
POST /tcm/v0/feedback

Decision logs
GET  /tcm/v0/decisions

Implementations MAY add additional endpoints under different paths. They MUST NOT repurpose or change the semantics of the endpoints defined here.
	3.	Documents API

Documents represent targets of assignment (for example users, pages, items, events). They correspond to DocumentConfig_j in the core spec.

3.1 POST /tcm/v0/documents:upsert

Purpose:

Create or update a document configuration.

Request body:

{
“doc_id”: “string, required”,
“text”: “string, required”,
“embedding”: [float, …],               // optional
“capacity”: {                            // optional
“total”: 123,                          // C_total, optional
“per_slot”: {                          // C_slot[s], optional
“2025-11-27T00:00:00Z”: 10,
“2025-11-27T01:00:00Z”: 5
}
},
“attributes”: {                          // optional
“key1”: “value or array”,
“country”: “JP”,
“segments”: [“high_value”, “beta”]
}
}

Field semantics:

doc_id
Required. Identifier of the document. Unique per tenant. Idempotent key for upsert.

text
Required. Text representation used by IndexCore for candidate generation.

embedding
Optional. Fixed-length numeric vector used by ANN index if available.

capacity.total
Optional. Non-negative integer. Total number of assignments allowed for this document across the planning horizon. If omitted, treated as “unbounded” in v0.0.1.

capacity.per_slot
Optional. Mapping from slot timestamps to non-negative integers. Each key is a time slot string. If present, these are treated as C_j[s]. Missing slots are treated as zero or derived from total, depending on implementation.

attributes
Optional. Opaque key-value map used for filtering and scoring. Values MAY be strings, numbers, booleans, or arrays of these.

Response:

On success:

HTTP 200 OK

{
“doc_id”: “same as request”
}

The operation MUST be idempotent: sending the same doc_id with the same body MUST yield the same logical state.

3.2 POST /tcm/v0/documents:delete

Purpose:

Delete a document configuration.

Request body:

{
“doc_id”: “string, required”
}

Behavior:
	1.	If the document exists, it MUST be marked as deleted and excluded from future candidate generation and serving.
	2.	If the document does not exist, the server MAY treat this as success or MAY return not_found. It SHOULD prefer idempotent success.

Response:

On success:

HTTP 200 OK

{
“doc_id”: “same as request”
}
	4.	Patterns API

Patterns represent assignable items (for example campaigns, notifications, rules). They correspond to PatternConfig_i in the core spec.

4.1 POST /tcm/v0/patterns:upsert

Purpose:

Create or update a pattern configuration.

Request body:

{
“pattern_id”: “string, required”,
“text”: “string, required”,
“embedding”: [float, …],               // optional
“budget”: {                              // required at least one of total or per_slot
“total”: 100000,                       // c_total, optional
“per_slot”: {                          // c_slot[s], optional
“2025-11-27T00:00:00Z”: 10000,
“2025-11-27T01:00:00Z”: 5000
}
},
“constraints”: {                         // optional, implementation-defined
“per_doc_max”: 10
},
“attributes”: {                          // optional
“country_allow”: [“JP”, “US”],
“type”: “notify”
},
“priority”: 10.0                         // optional
}

Field semantics:

pattern_id
Required. Identifier of the pattern. Unique per tenant. Idempotent key for upsert.

text
Required. Text description used by IndexCore for candidate generation.

embedding
Optional. Fixed-length numeric vector used by ANN index if available.

budget.total
Optional. Non-negative integer. Total assignments allowed for this pattern across the planning horizon (c_i_total). If omitted and per_slot is provided, total MAY be derived as the sum of per_slot.

budget.per_slot
Optional. Mapping from slot timestamps to non-negative integers c_i[s]. If omitted, PLAN0 MAY derive per-slot capacities from budget.total and traffic distribution.

constraints
Optional. Opaque key-value map describing additional constraints. v0.0.1 does not inspect these fields, but implementations MAY interpret well-known keys.

attributes
Optional. Opaque key-value map used for filtering and scoring.

priority
Optional. Numeric priority hint. Higher values indicate more important patterns. v0.0.1 MAY ignore this field or use it to influence alpha_i in planning.

Response:

On success:

HTTP 200 OK

{
“pattern_id”: “same as request”
}

The operation MUST be idempotent.

4.2 POST /tcm/v0/patterns:delete

Purpose:

Delete a pattern configuration.

Request body:

{
“pattern_id”: “string, required”
}

Behavior:
	1.	If the pattern exists, it MUST be marked as deleted and excluded from future candidate generation and serving.
	2.	If the pattern does not exist, the server MAY treat this as success or MAY return not_found. It SHOULD prefer idempotent success.

Response:

On success:

HTTP 200 OK

{
“pattern_id”: “same as request”
}
	5.	Plans API

Plans represent outputs of the offline planner PLAN0: per-pattern weights and per-slot capacities.

5.1 GET /tcm/v0/plans

Purpose:

List available plan snapshots.

Query parameters:

none in v0.0.1

Response:

HTTP 200 OK

{
“plans”: [
{
“plan_snapshot_id”: “string”,
“horizon_start”: “timestamp”,
“horizon_end”: “timestamp”
},
…
]
}

Semantics:
	1.	The list MUST include at least the currently active plan snapshot, if any.
	2.	The list MAY be truncated; the server MAY expose only a recent subset.

5.2 GET /tcm/v0/plans/{plan_snapshot_id}

Purpose:

Retrieve details of a specific plan snapshot.

Path parameter:

plan_snapshot_id
Required. Identifier of a plan snapshot as returned by /plans.

Response on success:

HTTP 200 OK

{
“plan_snapshot_id”: “string”,
“horizon_start”: “timestamp”,
“horizon_end”: “timestamp”,
“slots”: [“timestamp_slot_0”, “timestamp_slot_1”, …],
“alpha”: {
“pattern_id_1”: 1.0,
“pattern_id_2”: 0.8
},
“c_slot”: {
“pattern_id_1”: {
“timestamp_slot_0”: 100,
“timestamp_slot_1”: 50
},
“pattern_id_2”: {
“timestamp_slot_0”: 200
}
},
“metadata”: {
“planner_version”: “string”,
“notes”: “optional”
}
}

Field semantics:

slots
Ordered list of slot identifiers used by this plan. Each element is a timestamp string representing the start of a slot.

alpha
Map from pattern_id to non-negative float weight alpha_i. Used by SERVE0 in scoring.

c_slot
Map from pattern_id to per-slot capacity maps c_i[s]. Values are non-negative integers.

metadata
Optional map with planner-specific information.

If the plan_snapshot_id is unknown, the server MUST return 404 Not Found with an error body.

5.3 POST /tcm/v0/plans:run

Purpose:

Request that the server run PLAN0 and create a new plan snapshot. This endpoint is optional in v0.0.1 and MAY be disabled in managed deployments.

Request body:

{
“horizon_start”: “timestamp, optional”,
“horizon_end”: “timestamp, optional”,
“slot_granularity”: “string, optional”
}

If fields are omitted, the server uses implementation-defined defaults.

Response on success:

HTTP 202 Accepted or 200 OK

{
“plan_snapshot_id”: “string”
}

Semantics:
	1.	The server MAY run PLAN0 synchronously or asynchronously.
	2.	If PLAN0 is asynchronous, the plan_snapshot_id refers to a plan that will appear in /plans once ready.
	3.	The client SHOULD handle the possibility that the returned plan is not immediately usable.
	4.	Serve API

The serve endpoint exposes SERVE0: online decision making for individual requests.

6.1 POST /tcm/v0/serve

Purpose:

Given a document (or context) and a time, select up to L patterns to assign.

Request body:

{
“request_id”: “string, required”,
“doc”: {
“doc_id”: “string, optional”,
“context_text”: “string, optional”,
“attributes”: {
“key”: “value or array”
}
},
“timestamp”: “timestamp, required”,
“limit”: 3,
“filters”: {
“allow_patterns”: [“pattern_id or prefix”, “…”],
“deny_patterns”: [“pattern_id or prefix”, “…”]
},
“plan_snapshot_id”: “string, optional”,
“debug”: false
}

Field semantics:

request_id
Required. Unique identifier for this serve request. Used to correlate decisions and feedback.

doc
Required object. Either:
- References an existing document via doc_id, optionally with additional context_text and attributes.
- Or, omits doc_id and provides context_text and attributes only, for transient contexts.

If doc_id is provided, the server SHOULD look up the corresponding DocumentConfig. Attributes in the request MAY override or augment stored attributes.

timestamp
Required. Current time of the request. Used to map to a slot index s.

limit
Required. Maximum number of patterns to return. Must be a positive integer. The server MAY enforce an upper bound.

filters.allow_patterns
Optional. List of pattern_ids or implementation-defined prefixes that are allowed. If non-empty, only matching patterns are considered.

filters.deny_patterns
Optional. List of pattern_ids or prefixes that are disallowed. Matching patterns MUST NOT be returned.

plan_snapshot_id
Optional. If present, the server MUST attempt to use this plan snapshot for alpha and capacity. If absent, the server MUST use the currently active plan.

debug
Optional boolean. If true, the server MAY include additional debug information in the response. v0.0.1 does not standardize debug fields.

Response on success:

HTTP 200 OK

{
“request_id”: “same as request”,
“plan_snapshot_id”: “string in use”,
“slot”: “timestamp slot key”,
“decisions”: [
{
“pattern_id”: “pattern:1”,
“score”: 0.823
},
{
“pattern_id”: “pattern:2”,
“score”: 0.801
}
]
}

Field semantics:

plan_snapshot_id
The identifier of the plan snapshot actually used for this decision.

slot
The slot key derived from timestamp (for example the slot start time).

decisions
Ordered list of selected patterns, most preferred first. The length is less than or equal to limit. If no suitable patterns exist, the list MAY be empty.

The server MUST:
	1.	Enforce pattern and document capacities according to the active plan and runtime state.
	2.	Exclude patterns that violate static constraints or filters.
	3.	Use a deterministic tie-breaking rule where possible (for example pattern_id order) to improve reproducibility.
	4.	Feedback API

Feedback events allow the system to record outcomes (impressions, clicks, conversions, custom events) for decisions. They correspond to the feedback records in the core spec.

7.1 POST /tcm/v0/feedback

Purpose:

Record an event for a specific (request_id, doc_id, pattern_id) combination.

Request body:

{
“request_id”: “string, required”,
“doc_id”: “string, required”,
“pattern_id”: “string, required”,
“event”: “string, required”,
“value”: 1.0,
“timestamp”: “timestamp, required”,
“metadata”: {
“key”: “value”
}
}

Field semantics:

event
Required. Event type. Typical values include:
“impression”
“click”
“convert”
“dismiss”
“custom:”

v0.0.1 does not interpret event types. They are stored and may be used by future planner runs or external models.

value
Optional numeric value associated with the event (for example revenue, weight). Defaults to 1.0 if omitted.

metadata
Optional key-value object for additional fields (for example experiment labels, device info).

Response on success:

HTTP 200 OK

{
“status”: “ok”
}

The server MUST treat feedback events as append-only. It MUST NOT attempt to deduplicate events by default.
	8.	Decision logs API

The decisions endpoint provides a minimal way to retrieve historical decisions for debugging and analysis.

8.1 GET /tcm/v0/decisions

Purpose:

List past decisions matching simple filters.

Query parameters:

from
Optional. Start timestamp (inclusive) as RFC 3339 string.

to
Optional. End timestamp (exclusive) as RFC 3339 string.

doc_id
Optional. Filter by document identifier.

pattern_id
Optional. Filter by containing pattern identifier.

request_id
Optional. Filter by request identifier.

limit
Optional. Maximum number of items to return. Positive integer. The server MAY enforce an upper bound.

cursor
Optional. Opaque pagination cursor returned by a previous call.

Response on success:

HTTP 200 OK

{
“items”: [
{
“request_id”: “string”,
“timestamp”: “timestamp”,
“doc_id”: “string”,
“pattern_ids”: [“pattern:1”, “pattern:2”],
“plan_snapshot_id”: “string”,
“slot”: “timestamp slot key”
}
],
“next_cursor”: “string or null”
}

Semantics:
	1.	The server MAY return fewer than limit items even if more are available.
	2.	If next_cursor is non-null, the client MAY pass it to a subsequent request to retrieve more items.
	3.	The set of fields returned for each item is fixed in v0.0.1 as shown above.
	4.	Compatibility and evolution

9.1 Backward compatibility

The server MUST preserve the following guarantees for v0.0.1:
	1.	Existing endpoints and HTTP methods remain available.
	2.	Existing required request fields remain required and keep their meaning.
	3.	Existing response fields remain present with compatible types.

The server MAY add:
	1.	New optional request fields.
	2.	New optional response fields.
	3.	New endpoints under the /tcm/v0 prefix.

Clients SHOULD:
	1.	Ignore unknown fields in responses.
	2.	Not rely on the absence of optional fields.

9.2 Relation to core TCM-Core v0.0.1 spec

The API v0.0.1 is a direct mapping of TCM-Core v0.0.1 concepts:
	1.	DocumentConfig_j is exposed via /documents:upsert and /documents:delete.
	2.	PatternConfig_i is exposed via /patterns:upsert and /patterns:delete.
	3.	PlanSnapshot is exposed via /plans and /plans/{id}.
	4.	SERVE0 behavior is exposed via /serve.
	5.	Feedback records are exposed via /feedback.
	6.	Decision records are exposed via /decisions.

The API does not specify internal data structures, indexing strategies, or planning algorithms. These are defined in the core TCM-Core specification and are implementation details from the API perspective.

9.3 Authentication placeholder

v0.0.1 does not specify authentication. In practical deployments, it is RECOMMENDED to:
	1.	Require an API key or bearer token via HTTP headers.
	2.	Scope identifiers and operations to a tenant associated with the credentials.
	3.	Log all modification operations for audit.

These concerns are intentionally left out of this versioned API surface.

