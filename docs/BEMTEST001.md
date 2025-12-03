BEM v0.0.1 – Boolean Expert Machine, Self-Improving Hardware-Oriented Core Specification
(Refined Draft, ASCII only)
	0.	Scope, Objectives, Assumptions

0.1 Scope

This document defines BEM v0.0.1 as a hardware-oriented, finite-state abstract machine with:

0.1.1 Core characteristics
	•	Bit-sliced integer control core with a fixed set of co-processors.
	•	Constant-time fast path per logical step, independent of episode length and history length.
	•	Finite table of Boolean experts, each a local state transformer over bit-sliced state.
	•	Online learning over experts using bandit-style statistics and fixed-point log-weights.
	•	Structural self-improvement via patches (delta) that modify experts, CFG, routing, and invariants.
	•	Verification kernel that blocks unsafe or unproven structural changes.
	•	Scheduler that autonomously allocates compute between acting, synthetic work, and verification.

0.2 Primary objectives

BEM v0.0.1 is designed to be an autonomous self-improvement engine under fixed safety and resource constraints.

Let:
	•	Task distribution D_task be stationary or slowly drifting.
	•	Safety specification S_safe be fixed at configuration time.
	•	Compute budget B_compute be positive.

Objective:
	1.	As meta-time m increases, the structural configuration C_m should evolve so that:
	•	Task-level performance metrics (for example average reward, negative regret, or benchmark score) improve in expectation over sufficiently long windows.
	•	Safety violation rate stays below a configured bound delta_safe (ideally zero).
	2.	The full loop
	•	generate structural proposals,
	•	verify proposals with respect to S_safe,
	•	accept or reject them and update C,
MUST be executable without human intervention, except for:
	•	initial configuration of tasks, safety specs, and environment interfaces,
	•	external decisions to snapshot, deploy, or rollback image states.

0.3 Non-goals

This specification does not define:
	•	Concrete instruction encoding, microarchitecture, or physical layout.
	•	Concrete numeric performance or power envelopes.
	•	Any host programming language, ABI, or OS integration.
	•	Neural network primitives on the fast path.
	•	Exact choice of learning algorithms, beyond constraints on:
	•	complexity,
	•	safety,
	•	data flow.

External neural or heuristic components MAY be used by the PROPOSER, but no structural change may affect the trusted fast path until verified and accepted by the verification kernel.

0.4 Notation and complexity model

0.4.1 Indices and time scales
	•	t = 0, 1, 2, … : fast-path step index.
	•	e = 0, 1, 2, … : episode index.
	•	m = 0, 1, 2, … : meta-step index for slow path (patch-level operations).

Fast path is defined per-step t. Mid path is defined per episode or per fixed number of steps. Slow path is defined per meta-step m.

0.4.2 Parameters
	•	N: number of global state bits.
	•	K: number of shared memory bits, K >= N.
	•	P: number of fixed-point parameters.
	•	W: SIMD width (number of logical lanes).
	•	S: maximum number of expert slots.
	•	d: routing feature dimension.
	•	k_large: candidate set size for ANN, e.g. 16 to 64.
	•	k_small: final selected candidate count, e.g. 1 to 4.
	•	D_max: maximum ANN search depth.
	•	M_max: maximum ANN neighbor degree.

All parameters are fixed at design or configuration time and remain constant at runtime.

0.4.3 Complexity

Per-step complexity is measured as:
	•	Number of BEM-CORE instructions executed.
	•	Number of co-processor invocations.

For all valid configurations, there must exist global constants:
	•	C_obs, C_ANN, C_route, C_expert_batch, C_stats,

such that the fast-path step cost satisfies:

Cost_step(t) <= C_obs + C_ANN + C_route + C_expert_batch + C_stats

for all t, independent of episode length and total history length.
	1.	State, Configuration, and History

1.1 Decomposition

At meta-step m, BEM maintains:
	•	X_t : fast-path execution state at step t.
	•	C_m : structural configuration.
	•	H_m : history, logs, proofs, patch metadata.

For simplicity in specification, we often write C and H and omit m when context is clear.

1.2 Execution state X_t

X_t = (s_t, M_t, Theta_t)
	•	s_t: bitvector of length N (global logical state).
	•	M_t: bitvector of length K (shared memory).
	•	Theta_t: integer vector of length P (fixed-point parameters).

Fast-path instructions:
	•	MAY read and write s_t and M_t.
	•	MAY read Theta_t.
	•	MUST NOT modify structural tables and PROOF state.

Mid-path operations:
	•	MAY update Theta_t, bandit statistics, and counters.
	•	MUST respect complexity bounds for mid-path.

1.3 Structural configuration C

C = (EXPERT, CFG, ANN, INVAR, ROUTE_META, WORK_CONF, VERIFIER_CONF, ID_MAPS, LAYOUT)

This includes:
	•	EXPERT table: structural and learned parameters of experts.
	•	CFG: instruction array and control-flow graph.
	•	ANN: routing index and its parameters.
	•	INVAR: invariants and safety formulas.
	•	ROUTE_META: routing configuration per task and per context.
	•	WORK_CONF: PoX weights, difficulty, scheduler hyperparameters.
	•	VERIFIER_CONF: solver options, timeouts, proof formats.
	•	ID_MAPS and LAYOUT: identifier mappings, segment layout information.

Fast path MAY read parts of C but MUST NOT write to them.

1.4 History and logs H

H = (TRACE_LOG, PROOF_STATE, PATCH_QUEUE_META, SNAPSHOT_META, ACCOUNTING)

Where:
	•	TRACE_LOG: step, episode, and patch events, stored append-only.
	•	PROOF_STATE: CNF formulas, solver contexts, proofs, unsat cores, CEGIS examples.
	•	PATCH_QUEUE_META: queue Q_patch of pending patches, gain and cost estimates.
	•	SNAPSHOT_META: metadata of snapshots and rollback points.
	•	ACCOUNTING: long-run statistics for performance, safety, and PoX.

Slow-path operations MAY update any part of H. Fast path MUST restrict itself to constant-time appends or counters where configured.

1.5 Access constraints
	•	Fast path:
	•	Read X, C.
	•	Append minimal TRACE_LOG entries.
	•	No writes to EXPERT structures, CFG, PROOF_STATE, or PATCH_QUEUE_META.
	•	Mid path:
	•	Read X, C, H.
	•	Update Theta_t, bandit stats, simple counters.
	•	Slow path:
	•	Read X (optionally), C, H.
	•	Modify C and H via atomic application of verified patches.

	2.	Identifiers, Segments, Slot Mapping

2.1 Identifier space U

Let U be the integer set {0, 1, …, 2^32 - 1}. Each identifier u in U is interpreted as:
	•	class: 6 high-order bits, object class, range 0 to 63.
	•	ecc: 6 bits, error-correcting code over shard and local fields.
	•	shard: 6 bits, logical shard.
	•	local: 14 bits, shard-local index.

Binary layout:

[ class(6) | ecc(6) | shard(6) | local(14) ]

Gray code mapping:
	•	g(u) = u xor (u >> 1)

Similarity between identifiers a and b:
	•	sim(a, b) = 32 - HammingDistance(g(a), g(b))

The mapping g and similarity score MAY be used in ANN sharding and expert clustering.

2.2 Memory segments

Logical segments:
	•	STATE: bit-sliced representation of global state s_t for W lanes.
	•	SHARED: shared memory M_t, Theta_t, configuration and counters.
	•	EXPERT: expert table and per-expert stats and parameters.
	•	CFG: instruction array and CFG node metadata.
	•	TRACE_LOG: run-time logs and patch events.
	•	PROOF_STATE: CNF, solver state, proofs, CEGIS traces.
	•	WORK: PoX configuration, scheduler statistics, drift metrics.

Implementation MAY map these to pages, regions, or devices arbitrarily, but MUST ensure no observable aliasing between logical segments.

2.3 STATE representation

STATE contains N entries:
	•	S_pl[k] is a W-bit word representing bit k of s_t across W lanes.
	•	For lane l in [0, W-1], the lane-local bit s_t^(l)[k] equals bit l of S_pl[k].

Shared memory M_t and parameters Theta_t are not bit-sliced. They are stored in SHARED segment as contiguous bitvectors or fixed-width integers.

2.4 EXPERT table and slot mapping

EXPERT slots are indexed 0 to S-1. Each slot i holds:
	•	id_i: identifier in U.
	•	R_spec_i: read specification.
	•	C_rep_i: circuit representation.
	•	W_spec_i: write specification.
	•	stats_i: bandit and usage statistics.
	•	params_i: log-weight and loss accumulators.
	•	routing_meta_i: routing-related metadata, tags, and versions.

For tabular object types (for example experts), a slot function is defined:
	•	slot: U -> {0, …, S-1} union {bottom}

Requirements:
	•	For any allocated expert identifier u, slot(u) != bottom and EXPERT[slot(u)].id_i == u.
	•	For any two distinct allocated experts u != v, slot(u) != slot(v).
	•	slot(u) and its inverse must be stored in SHARED, consistent under rebalancing.
	•	Rebalancing MAY change slot(u) but MUST preserve:
	•	expert semantics,
	•	referential integrity for all references that use identifiers u, not slot indices.

	3.	Hardware Modules

3.1 BEM-CORE

BEM-CORE is the integer control core that:
	•	Executes a fixed instruction set including:
	•	integer arithmetic,
	•	bitwise operations,
	•	control transfer,
	•	loads and stores,
	•	co-processor commands.
	•	Implements the fast-path loop and scheduling logic.
	•	Dispatches co-processor operations to:
	•	BIT-ALU,
	•	ANN unit,
	•	PROVER unit,
	•	PROPOSER unit,
	•	HASH/ECC,
	•	LOG unit.

3.2 BIT-ALU

Supports operations on W-bit words:
	•	bitwise_and(a, b),
	•	bitwise_or(a, b),
	•	bitwise_xor(a, b),
	•	bitwise_not(a),
	•	shift_left(x, k),
	•	shift_right(x, k),
	•	rotate_left(x, k),
	•	rotate_right(x, k),
	•	popcount(x) returning an integer,
	•	parity(x) = popcount(x) mod 2,
	•	blend(mask, a, b) returning (mask ? a : b) lane-wise.

BIT-ALU operations must have latency bounded by a small constant independent of W.

3.3 ANN unit

Interface:
	•	ANN_QUERY(q, f, k, config) -> candidate_list

Inputs:
	•	q: 32-bit query key.
	•	f: bitvector of length d (routing features).
	•	k: integer, 1 <= k <= k_large.
	•	config: routing configuration (per-task, per-bucket parameters).

Output:
	•	candidate_list: list of at most k expert slots or identifiers.

Constraints:
	•	Internal search depth <= D_max.
	•	Per-node neighbor degree <= M_max.
	•	Worst-case instruction count per ANN_QUERY bounded by a constant given D_max, M_max.

ANN unit maintains:
	•	index structures keyed by:
	•	shard_id = high bits or Gray-coded prefix of q,
	•	bucket_id = selected bits of q.
	•	small per-bucket lists of:
	•	representative experts,
	•	recently high-performing experts.

3.4 PROVER unit

Interface:
	•	SAT_CHECK(cnf_id) -> {SAT, UNSAT, UNKNOWN}
	•	PROOF_CHECK(cnf_id, proof_id) -> {ACCEPT, REJECT}
	•	HOARE_CHECK(cfg_id, annotations_id) -> {ACCEPT, REJECT}
	•	CEGIS(spec_id, hyp_class_id) -> (candidate_id or NONE, cex_id or NONE)

Responsibilities:
	•	Maintain incremental CNF contexts with identifiers:
	•	baseCNF_id for global context,
	•	deltaCNF_id for per-patch or per-spec additions.
	•	Provide unsat cores when SAT_CHECK returns UNSAT.
	•	Support Hoare-style VC construction and checking.
	•	Provide CEGIS-style synthesis and counterexample generation.

PROVER MUST be pure with respect to X_t: no modification of execution state. All persistent changes flow through C and H via patch application or proof storage.

3.5 PROPOSER unit

Interface:
	•	PROPOSE_PATCH(context_id, budget) -> list of patches delta.
	•	PROPOSE_INVARIANT(context_id, budget) -> list of invariant candidates.
	•	PROPOSE_EXPERT(template_id, budget) -> list of expert candidates.

Responsibilities:
	•	Generate untrusted structural proposals based on:
	•	logs,
	•	templates,
	•	heuristics,
	•	or external models.

budget is an integer bound on allowed compute or search depth.

All outputs from PROPOSER are treated as candidates only. Trust and safety are enforced by PROVER and patch acceptance logic.

3.6 HASH/ECC unit

Responsibilities:
	•	ECC encode and decode for identifiers and memory blocks.
	•	Hash functions:
	•	hash(bytes) -> fixed-length hash (for example 256 bits).
	•	Merkle tree operations:
	•	recompute parent hashes from child hashes.
	•	Hash chain updates:
	•	H_{k+1} = hash(concat(H_k, entry_k)).

3.7 LOG unit

Responsibilities:
	•	Append structured entries to TRACE_LOG.
	•	Maintain the hash chain H_k for log integrity.
	•	Update counters and moving averages needed by WORK segment.

	4.	Observations, Tasks, and Routing Features

4.1 Observation function

A deterministic function O defines:
	•	obs_t = O(s_t, M_t) = (task_id_t, context_hash_t, local_bits_t, feature_bits_t)

Components:
	•	task_id_t:
	•	small integer identifying task family or benchmark class.
	•	context_hash_t:
	•	32-bit or 64-bit rolling hash:
context_hash_t = H_ctx(context_hash_{t-1}, summary(obs_{t-1}, r_{t-1}, a_{t-1}))
	•	H_ctx uses only integer arithmetic and bitwise operations.
	•	local_bits_t:
	•	small fixed subset of bits from s_t and M_t.
	•	indices determined by configuration for the task.
	•	feature_bits_t:
	•	bitvector of size d computed by F_feat(s_t, M_t, context_hash_t).

4.2 Query key derivation

From obs_t:
	•	q_t = F_id(task_id_t, context_hash_t)
	•	f_t = feature_bits_t

Requirements for F_id:
	•	Must be deterministic.
	•	Must provide enough dispersion over shards and buckets for ANN_INDEX.
	•	Must use only integer and bitwise operations.

4.3 Task metadata and environment

BEM MAY maintain additional per-task metadata in SHARED:
	•	per-task counters N_tau,
	•	per-task reward statistics,
	•	task-specific routing and learning hyperparameters.

The external environment is not normatively specified, but must be callable in a way that:
	•	given action(s) (for example expert selection), returns reward r_t and optionally new observations.

	5.	Experts and Circuits

5.1 Expert semantics

Each expert i defines a local transformer:
	•	R_i: read function,
	•	C_i: combinational Boolean circuit,
	•	W_i: write function.

Given execution state (s, M):
	1.	x = R_i(s, M)
	•	x is a bitvector of length n_i (0 <= n_i <= N_in_max).
	2.	y = C_i(x)
	•	y is a bitvector of length m_i (0 <= m_i <= N_out_max).
	3.	(s_new, M_new) = W_i(s, M, y)

R_spec_i and W_spec_i define:
	•	which bits or regions of STATE and SHARED are read or written,
	•	how addressing and indexing are computed,
	•	how masks are applied.

5.2 Circuit representation

C_rep_i must support bit-sliced evaluation for W lanes.

Allowed representations include:
	•	Algebraic normal form (ANF):
	•	each output bit is XOR of monomials,
	•	each monomial is AND of a subset of input bits.
	•	Reduced ordered binary decision diagrams (ROBDD) or equivalent DAG:
	•	fixed variable ordering,
	•	merged isomorphic subgraphs,
	•	no redundant nodes.

Implementation:
	•	Inputs of all lanes are packed as W-bit bitvectors.
	•	Intermediate nodes are W-bit values manipulated via BIT-ALU.
	•	Evaluation cost per expert is bounded by a constant C_expert_max.

5.3 Expert cost bound

For each expert i, there exist constants:
	•	N_in_max >= n_i,
	•	N_out_max >= m_i,
	•	C_expert_max >= cost of evaluating C_rep_i for one BEM_EXPERT_BATCH.

C_expert_max cannot depend on episode length, history length, or current step t.
	6.	Fast Path: Algorithm and Guarantees

6.1 Fast-path step pseudocode

Conceptual pseudocode for one fast path step t:
	1.	obs_t = O(s_t, M_t)
	2.	(q_t, f_t) = (F_id(task_id_t, context_hash_t), feature_bits_t)
	3.	C_large = ANN_QUERY(q_t, f_t, k_large, config(task_id_t))
	4.	C_small or i_t = ROUTE_SELECT(C_large, stats, z, task_id_t)
	5.	BEM_EXPERT_BATCH(C_small or i_t, STATE, SHARED)
	6.	STATS_UPDATE(i_t or C_small, r_t, task_id_t)

The external environment contribution is:
	•	compute r_t and update of external state, which may inject information into s_{t+1}, M_{t+1} for the next step.

6.2 Routing selection

Given:
	•	candidate set C_large (size <= k_large),
	•	bandit statistics stats_i,tau,
	•	log-weights z_i,
	•	task id tau,

the routing module computes:
	•	mean_i,tau = wins_i,tau / max(1, visits_i,tau)
	•	exploration_bonus_i,tau = c_explore * sqrt( log(max(1, N_tau)) / max(1, visits_i,tau) )
	•	score_i,tau = mean_i,tau + exploration_bonus_i,tau

It then selects either:
	•	C_small: top k_small experts by score_i,tau (deterministic), or
	•	a single i_t sampled from a distribution derived from (score_i,tau, z_i).

The choice must be:
	•	deterministic given random seed and state,
	•	bounded-cost in terms of arithmetic operations.

6.3 Lane assignment schemes

The following schemes are allowed:
	•	Global expert:
	•	select one i_t, apply it to all lanes.
	•	Grouped lanes:
	•	partition lanes into fixed or dynamic groups, one expert per group.
	•	Per-lane experts:
	•	assign expert per lane with lane-specific masks; more expensive and SHOULD be used rarely.

BEM_EXPERT_BATCH executes experts in a bit-sliced manner across all lanes and returns updated STATE and SHARED.

6.4 Fast-path complexity guarantee

Per fast-path step t, total cost Cost_step(t) must satisfy:
	•	Cost_step(t) <= C_obs + C_ANN + C_route + C_expert_batch + C_stats

where each constant depends only on configuration parameters and not on t.

No PROVER operation, CEGIS call, or heavy structural search may be executed on the fast path.
	7.	Rewards and Bandit Learning

7.1 Reward definition

At each step t, after acting, BEM receives reward r_t.

Constraints:
	•	R_min <= r_t <= R_max for fixed constants R_min, R_max.

7.2 Loss and accumulators

For a given expert i and task tau:
	•	per-step loss:
	•	loss_t(i, tau) = -r_t if expert i was selected for task tau at step t,
	•	loss_t(i, tau) = 0 otherwise.
	•	cumulative loss:
	•	L_i,tau = sum of loss_t(i, tau) over steps.
	•	cumulative squared loss:
	•	S_i,tau = sum of loss_t(i, tau)^2.

All of L_i,tau, S_i,tau, wins_i,tau, visits_i,tau are stored as fixed-point or integer counters with scaling chosen at configuration time.

7.3 Log-weights update

At mid-path or update checkpoints:
	•	hat_l_i,tau = L_i,tau / max(1, visits_i,tau)
	•	hat_l_clipped = clamp(hat_l_i,tau, -L_max, L_max)
	•	eta_i,tau = c_eta / sqrt(S_i,tau + epsilon_eta)

Then:
	•	z_i_new = z_i - eta_i,tau * hat_l_clipped

If abs(z_i_new - z_i) > Delta_z_max, clip to z_i +/- Delta_z_max.

Periodically, z_i values MAY be renormalized to avoid numeric overflow or collapse (for example subtracting a common offset).

7.4 Stats update

When expert i is used at step t for task tau, BEM updates:
	•	visits_i,tau += 1
	•	visits_total_i += 1
	•	wins_i,tau += r_t or suitable reward proxy
	•	wins_total_i += r_t
	•	N_tau += 1

These updates may occur on the fast path or just after it, but MUST be bounded-cost linear operations on fixed-width integers.
	8.	Structural Patches: Model and Lifecycle

8.1 Patch model

A patch delta is a finite, structured object describing a candidate change to configuration C and possibly to H.

Typical patch classes include:
	•	expert_split,
	•	expert_merge,
	•	macro_create,
	•	macro_refine,
	•	circuit_superopt,
	•	invariant_add,
	•	invariant_strengthen,
	•	ann_reindex,
	•	route_meta_update.

Each patch delta has:
	•	patch_id: identifier in U.
	•	patch_class: class field of patch_id.
	•	target_ids: list of identifiers for objects it modifies.
	•	proposal_data: new descriptors or parameters.
	•	expected_metrics: initial estimates of delta_R, delta_S, delta_C, delta_I.
	•	estimated_cost: estimate C_verif(delta) for verification.

8.2 Patch lifecycle states and transitions

States:
	•	GENERATED:
	•	patch produced by PROPOSER.
	•	QUEUED:
	•	patch inserted in Q_patch with initial estimates.
	•	VERIFIED:
	•	patch with VC(delta) proven as UNSAT by PROVER and PoX score above threshold D.
	•	ACCEPTED:
	•	patch applied atomically to C and recorded in TRACE_LOG and WORK.
	•	REJECTED:
	•	patch rejected due to failed checks or low PoX score; counterexamples stored in PROOF_STATE or TRACE_LOG.

Transitions:
	•	GENERATED -> QUEUED:
	•	by PROPOSER call.
	•	QUEUED -> VERIFIED or REJECTED:
	•	by PROVER and PoX policy.
	•	VERIFIED -> ACCEPTED:
	•	by application function APPLY_PATCH.
	•	VERIFIED or QUEUED -> REJECTED:
	•	by timeouts, failures, or explicit discard rules.

8.3 Patch generation

When a context (for example an episode or error region) is selected for structural analysis:
	1.	BEM-CORE invokes PROPOSER:
	•	delta_list = PROPOSE_PATCH(context_id, budget)
	2.	For each delta in delta_list:
	•	assign patch_id,
	•	classify patch_class,
	•	compute or estimate initial expected_metrics and estimated_cost,
	•	insert into Q_patch with initial priority.

Patch generation MUST NOT modify C or PROOF_STATE.

8.4 Patch verification

Scheduler picks a patch delta from Q_patch based on priority.

Verification steps:
	1.	Construct VC(delta) representing:
	•	safety requirements (no new BAD reachability),
	•	equivalence requirements (behavior preserved on domain D),
	•	invariant consistency (Hoare annotations satisfied).
	2.	Call SAT_CHECK(VC(delta)):
	•	If UNSAT:
	•	patch is consistent with encoded properties.
	•	If SAT:
	•	violation exists; request counterexample cex_id if possible.
	•	If UNKNOWN:
	•	patch remains QUEUED or moves to REJECTED according to policy.
	3.	Optionally use HOARE_CHECK and PROOF_CHECK for annotated CFG fragments.

PoX filter:
	•	compute score(delta) = w_R * delta_R + w_S * delta_S + w_C * delta_C + w_I * delta_I.
	•	compare score(delta) with difficulty D.

Decision:
	•	If SAT or safety violated:
	•	state -> REJECTED, log counterexamples in H.
	•	Else if UNSAT and score(delta) >= D:
	•	state -> VERIFIED.
	•	Else:
	•	state -> REJECTED or deferred, depending on policy.

8.5 Patch application

PATCH_APPLY(delta) executes:
	•	Acquire exclusive access to configuration C.
	•	Apply structural changes described in delta to:
	•	EXPERT, CFG, ANN, INVAR, ROUTE_META, WORK_CONF, or other parts of C.
	•	Update any Merkle roots or hashes covering modified segments using HASH/ECC.
	•	Append patch event to TRACE_LOG including:
	•	patch_id,
	•	patch_class,
	•	target_ids,
	•	proof references,
	•	PoX components and score(delta),
	•	parent hash.

After application, delta is in ACCEPTED state.
	9.	Verification and Invariants

9.1 CNF representation

PROOF_STATE stores CNF formulas of the form:
	•	Each clause is represented by (pos_bits, neg_bits) where:
	•	pos_bits[j] == 1 means variable j appears positively.
	•	neg_bits[j] == 1 means variable j appears negatively.

Formulas are collections of clauses referenced by cnf_id.

9.2 Invariants and CFG

CFG nodes v are associated with invariants:
	•	P_v: CNF formula representing property at entry of node v.

For edge (u -> v) with instruction sequence instr(u_to_v), a weakest precondition transformer:
	•	WP_instr(u_to_v, post) -> pre

must be defined, mapping postcondition CNF to precondition CNF.

Invariant consistency condition:
	•	For each edge (u -> v):
	•	P_u implies WP_instr(u_to_v, P_v).

Violations are encoded into VC(delta) when patch delta affects the CFG or invariants.

9.3 Verification conditions for patches

VC(delta) includes:
	•	Safe reachability:
	•	expressions capturing that BAD states are unreachable under patched configuration, or at least not newly reachable.
	•	Equivalence on domain D:
	•	for example, for all inputs in domain D, old and new behavior agree on specified outputs.
	•	Hoare consistency:
	•	updated invariants still satisfy CFG edges.

PROVER is used to decide SAT or UNSAT for VC(delta) with resource limits and incremental contexts.

9.4 CEGIS integration

CEGIS(spec_id, hyp_class_id) serves as a two-part process:
	•	If a satisfying candidate exists in hypothesis class hyp_class:
	•	CEGIS may return candidate_id and no counterexample.
	•	If current candidate is invalid:
	•	CEGIS returns cex_id describing a counterexample assignment.

This mechanism is used to:
	•	synthesize new experts,
	•	merge experts with preserved or improved behavior,
	•	derive or strengthen invariants.

	10.	PoX: Patch Scoring and Prioritization

10.1 Components

For each patch delta, BEM maintains estimates:
	•	delta_R: expected change in regret or reward on reference tasks.
	•	delta_S: expected change in safety margin (for example number of proven properties).
	•	delta_C: expected change in fast-path cost per step.
	•	delta_I: expected change in uncertainty or information gain.

10.2 Score definition

PoX score is defined as:

score(delta) = w_R * delta_R + w_S * delta_S + w_C * delta_C + w_I * delta_I

where:
	•	w_R, w_S, w_C, w_I are nonnegative weights stored in WORK_CONF.

10.3 Difficulty and thresholds

WORK_CONF stores:
	•	D: difficulty threshold.
	•	moving averages:
	•	E_score: recent mean of score(delta) for ACCEPTED patches.
	•	Var_score: recent variance of score(delta).

A patch delta is eligible for acceptance only if:
	•	VC(delta) is UNSAT (no violation),
	•	score(delta) >= D.

10.4 Priority in Q_patch

Each patch delta in Q_patch has a priority:

prio(delta) = score(delta) / max(epsilon, C_verif(delta))

where:
	•	C_verif(delta) is estimated verification cost,
	•	epsilon is a small positive constant.

Scheduler SHOULD serve patches in approximate descending order of prio, subject to fairness across patch classes and domains.
	11.	Scheduling, Time Scales, and Drift Handling

11.1 Time scales

BEM runs three coupled loops:
	1.	Fast loop (step-level):
	•	executes fast path for acting in environment.
	2.	Mid loop (episode-level):
	•	aggregates stats, updates weights, logs summaries.
	3.	Slow loop (meta-level):
	•	generates patches, verifies them, adjusts scheduler parameters.

11.2 Compute allocation vector

Let:
	•	alpha_act: fraction of compute for acting on real tasks (fast path).
	•	alpha_synth: fraction of compute for synthetic tasks and self-play.
	•	alpha_verify: fraction of compute for verification and patch evaluation.

Constraint:
	•	alpha_act + alpha_synth + alpha_verify = 1.

Scheduler maintains alpha_* based on summary statistics in WORK and ACCOUNTING.

11.3 Scheduling policy

Scheduler uses the following inputs:
	•	backlog_verify: number of QUEUED patches.
	•	mean_PoX_yield: moving average of score(delta) for ACCEPTED patches.
	•	safety_violation_rate: moving average of BAD occurrences or VC failures.
	•	performance_trend: change in task-level metrics over recent window.

Heuristic requirements:
	•	If backlog_verify is large and mean_PoX_yield is positive:
	•	alpha_verify SHOULD be increased.
	•	If performance_trend is flat or negative and safety_violation_rate is low:
	•	alpha_synth SHOULD be increased to generate more structured data for improvement.
	•	If safety_violation_rate approaches or exceeds delta_safe:
	•	application of new patches MUST be suspended.
	•	alpha_verify for patches that relax invariants SHOULD be reduced or set to zero.

Exact functional form is left to implementation but MUST be deterministic given state and configuration.

11.4 Drift detection

Let:
	•	W_drift: window size for drift analysis.
	•	performance_metric_t: scalar task-level metric (for example average reward).
	•	safety_violation_rate_t: current estimate of violation rate.

If over any window of length W_drift:
	•	performance_metric_t decreases by more than delta_perf, and
	•	mean_PoX_yield <= 0,

then BEM MUST enter degraded mode:
	•	Do not APPLY_PATCH for new patches.
	•	Continue running fast path and verification for monitoring only.
	•	Optionally mark this region in TRACE_LOG.

If safety_violation_rate_t > delta_safe:
	•	BEM MUST immediately suspend patch application.
	•	Optionally call ROLLBACK to a safe snapshot (see below).

11.5 Rollback

ROLLBACK(snapshot_id) MUST:
	•	Replace current C by the configuration stored in the snapshot.
	•	Restore H partially or minimally as configured.
	•	Reset safety_violation_rate estimates based on snapshot metadata.

Rollback events MUST be logged in TRACE_LOG with references to snapshot_id and safety metrics that triggered rollback.
	12.	Snapshot and Resume

12.1 Snapshot abstraction

SNAPSHOT(label) -> snapshot_id performs:
	•	Serialize C and selected parts of H into an image.
	•	Store metadata including:
	•	engine version,
	•	timestamp,
	•	recent performance metrics,
	•	recent safety metrics,
	•	PoX statistics,
	•	label.

Snapshot gating:
	•	SNAPSHOT SHOULD be allowed only when:
	•	safety_violation_rate <= delta_safe,
	•	mean_PoX_yield >= 0 over a configured window,
unless explicitly overridden.

12.2 Resume

RESUME(snapshot_id) MUST:
	•	Restore C to the snapshot version.
	•	Restore or reconstruct H according to implementation-defined policy.
	•	Initialize X according to environment configuration (for example fresh start or warm restart).

After RESUME, BEM must satisfy the same semantic invariants as if it had been running continuously with that C.

12.3 External deployment

External systems MAY:
	•	choose snapshot_id as deployment target,
	•	treat snapshot_id as a versioned artifact,
	•	run BEM instances rooted at different snapshots for evaluation or production.

BEM core specification does not constrain external deployment policies.
	13.	Three-Level Game Interpretation (Informative)

13.1 Level 1: task game (fast loop)
	•	Player 1: BEM fast-path controller.
	•	Player 2: environment.

Each step:
	•	BEM sees observation obs_t and chooses expert(s).
	•	Environment returns reward r_t and affects next state.
	•	Safety cost c_t is derived from BAD and fast-path cost.

This is a contextual bandit or constrained MDP where:
	•	BEM minimizes regret relative to safe policies,
	•	safety constraints are imposed by invariants and verification results.

13.2 Level 2: self-design game (slow loop)
	•	State: structural configuration C.
	•	Actions: patches delta that map C to C’ when verified.
	•	Payoff: PoX score minus verification cost.

This defines a meta-level game in which BEM chooses delta to improve long-run performance and safety while minimizing complexity and verification overhead.

13.3 Level 3: resource allocation game (scheduler)
	•	State: summaries of:
	•	performance_trend,
	•	safety_violation_rate,
	•	mean_PoX_yield,
	•	backlog_verify.
	•	Actions: compute allocation vector alpha.

Scheduler solves a resource allocation game where:
	•	goals are to maximize long-run PoX yield per compute unit,
	•	subject to safety constraint on violation rate.

13.4 Coupling
	•	Level 1 performance depends on C from level 2.
	•	Level 2 PoX signals depend on data generated by level 1 and synthetic runs.
	•	Level 3 allocation affects how much compute is available for level 1 and level 2.

The normative parts of this specification ensure:
	•	fast path is bounded and safe,
	•	patches are verified and scored before acceptance,
	•	scheduler reacts to drift and safety signals,

so that, in the limit of sufficient compute, BEM behaves as an autonomous self-improvement engine under explicit safety and complexity constraints.
