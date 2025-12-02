BEM Benchmark Suite v0.0.1 – Tightened Specification (Draft)
	0.	Scope and Objectives

0.1 Purpose

This document defines BEM Benchmark Suite v0.0.1 as a set of task classes, metrics, and protocols for evaluating implementations of the Boolean Expert Machine (BEM) v0.0.1.

The suite is designed to measure:
	1.	Fast-path performance:
1.1 Per-step and per-token cost.
1.2 Scaling with N, K, W, number of experts, and ROUTE parameters.
	2.	Algorithmic capability:
2.1 Exact and approximate performance on structured sequence tasks.
2.2 Generalization to longer sequences and higher difficulty.
	3.	Language-model-style behavior:
3.1 Bits-per-character (BPC) or token-level perplexity on small corpora.
	4.	Learning and structural update behavior:
4.1 Regret and adaptation in bandit-like settings.
4.2 Verification-gated structural changes cost and rate.

0.2 Non-goals

The benchmark suite does not:
	1.	Fix specific datasets or corpora; it defines task families and parameter ranges.
	2.	Impose absolute pass/fail thresholds.
	3.	Constrain implementation internals beyond:
3.1 Adhering to BEM v0.0.1 semantics.
3.2 Respecting fast-path bounded cost.

0.3 Structure

The benchmark suite is partitioned into:
	1.	Global assumptions and notation.
	2.	Microbenchmarks for hardware primitives and co-processors.
	3.	Algorithmic sequence tasks.
	4.	Language-model-like tasks.
	5.	Learning and structural update tasks.
	6.	Reporting and scoring.
	7.	Global Assumptions and Notation

1.1 Hardware and configuration

Unless stated otherwise, each benchmark assumes:
	1.	Single active general-purpose core (BEM-CORE).
	2.	Optional co-processors:
2.1 BIT-ALU.
2.2 ROUTE unit.
2.3 SAT/Hoare unit.
2.4 HASH/ECC unit.
2.5 LOG unit.

Implementations MUST report:
	1.	CPU model, microarchitecture, and clock frequency (approximate).
	2.	SIMD width W_hw (e.g., 128, 256, 512 bits).
	3.	Effective BEM lane width W (which may be ≤ W_hw or equal).

1.2 BEM parameters

For each benchmark configuration, implementations MUST specify:
	1.	N: state bit count.
	2.	K: shared memory bit count.
	3.	W: number of lanes.
	4.	Maximum number of experts N_expert_max.
	5.	Expert circuit size bound G_max.
	6.	ROUTE parameters H, B, C_entry, K_cand.
	7.	SAT/Hoare bounds n_max, m_max.
	8.	Parameters for learning:
8.1 Learning rate schedule eta_t.
8.2 B_flip (local edit budget).

1.3 Units and metrics

Time and compute are measured as:
	1.	cycles: CPU core cycles (estimated or measured).
	2.	ops/sec: operations per second.
	3.	tokens/sec or chars/sec: throughput at the token/character level.
	4.	regret, accuracy, BPC: task-dependent performance metrics.

Where possible, cycles SHOULD be measured using hardware performance counters. If unavailable, approximate cycle counts via instruction counts and clock frequency MAY be reported, clearly labeled as estimated.
	2.	Microbenchmarks

2.1 BIT-ALU throughput

2.1.1 Purpose

Measure raw throughput and latency of BIT-ALU bit-sliced operations for W lanes.

2.1.2 Setup

Parameters:
	1.	W: lane width (fixed for the BEM instance).
	2.	L_loop: number of iterations, L_loop ≥ 10^7.
	3.	op_set: list of operations to benchmark:
3.1 AND, OR, XOR, NOT.
3.2 shift_left, shift_right (by constant).
3.3 rotate_left, rotate_right (by constant).
3.4 vector_popcount.

Procedure:
	1.	For each operation op ∈ op_set:
1.1 Allocate one or more W-bit words as inputs in STATE or registers.
1.2 Initialize with pseudorandom data.
1.3 Execute op in a tight loop of L_loop iterations.
1.4 Ensure compiler does not optimize away op:
1.4.1 Accumulate a checksum in a scalar register.
1.4.2 Store checksum to SHARED at the end.

2.2.3 Metrics

For each op:
	1.	cycles_per_op = total_cycles / L_loop.
	2.	ops_per_sec = (clock_frequency * 1.0) / cycles_per_op.
	3.	bits_per_sec = ops_per_sec * W.

Implementations MUST provide:
	1.	cycles_per_op and bits_per_sec.
	2.	Description of how cycles were measured or estimated.

2.2 ROUTE query performance

2.2.1 Purpose

Measure performance and quality of ROUTE_QUERY as a function of index size and parameters (H, B, C_entry, K_cand).

2.2.2 Setup

Parameters:
	1.	N_nodes_set = {10^4, 10^5, 10^6} total expert-like entries.
	2.	K_cand (as configured).
	3.	H, B, C_entry (routing configuration).
	4.	Q: query count, Q ≥ 10^5 per N_nodes.

Index construction:
	1.	Sample N_nodes identifiers u uniformly in U or from a structured distribution.
	2.	Assign them to ROUTE tables based on class and shard as per BEM configuration.
	3.	Populate buckets using hash functions h_k.

Queries:
	1.	For each query index 1..Q:
1.1 Sample q uniformly from identifiers used to build the index, or from a mixture of in-index and out-of-index IDs.
1.2 Issue ROUTE_QUERY(q, class_expert, shard_filter, K_cand).
1.3 Capture returned candidate set C_t.

Optional ground truth:
	1.	For a subset Q_truth of queries (Q_truth ≈ 10^3), precompute the exact nearest neighbors under sim(u, v) in the full set to allow recall measurements.

2.2.3 Metrics

For each N_nodes:
	1.	cycles_per_query = total_cycles / Q.
	2.	queries_per_sec.
	3.	Optional recall@k:
3.1 For k ≤ K_cand, recall@k = fraction of queries where the true nearest neighbor appears in C_t.

Implementations MUST report:
	1.	H, B, C_entry, K_cand.
	2.	Approximate distribution of bucket occupancies (min, max, mean).
	3.	cycles_per_query, queries_per_sec.
	4.	If ground truth used, recall@k for chosen k.

2.3 Expert evaluation cost

2.3.1 Purpose

Measure cost of BEM_EXPERT_BATCH for synthetic experts of controlled size and the scaling with gate count and W.

2.3.2 Setup

Parameters:
	1.	G_set = {64, 128, 256, 512} synthetic circuit sizes (or implementation’s feasible subset).
	2.	W: lane width.
	3.	L_loop ≥ 10^6 invocations per G.

Expert construction:
	1.	For each G ∈ G_set:
1.1 Construct an expert descriptor e_G with:
1.1.1 Fixed n_i and m_i (e.g. n_i = 32, m_i = 8).
1.1.2 C_rep of size approximately G gates (ANF or LUT-tree).
1.1.3 Simple R_spec selecting bits from STATE and SHARED.
1.1.4 W_spec writing to STATE or SHARED.
1.2 Ensure circuits are non-trivial (not constant or identities).

Benchmark:
	1.	Initialize STATE for all W lanes with random bits.
	2.	For each G:
2.1 Execute BEM_EXPERT_BATCH with the expert e_G repeatedly in a loop of L_loop iterations.
2.2 Randomize input bits per iteration (e.g. flip a small subset of STATE bits).
2.3 Prevent dead-code elimination by accumulating simple checksums and writing them to SHARED.

2.3.3 Metrics

For each G:
	1.	cycles_per_batch = total_cycles / L_loop.
	2.	cycles_per_lane = cycles_per_batch / W.
	3.	batches_per_sec = (clock_frequency * 1.0) / cycles_per_batch.

Implementations MUST specify:
	1.	Expert form (ANF, ROBDD, LUT-tree).
	2.	n_i, m_i, and measure of gate count or LUT size.
	3.	Any vectorization details relevant to implementation.

2.4 SAT/Hoare kernel cost

2.4.1 Purpose

Measure the cost of SAT_CHECK and PROOF_CHECK for bounded CNF sizes.

2.4.2 Setup

Parameters:
	1.	n_set = {32, 64, 128}.
	2.	For each n, m_set = {4n, 8n}.
	3.	L_inst = 10^4 instances per (n, m) pair.

CNF generation:
	1.	For each (n, m):
1.1 Generate random CNFs with n variables and m clauses.
1.2 Choose clause lengths in a small range (e.g. 3 to 5 literals).
1.3 Ensure mixture of SAT and UNSAT instances.
	2.	Optionally precompute proof objects for UNSAT instances.

Benchmark:
	1.	For each (n, m):
1.1 Run SAT_CHECK on L_inst CNFs.
1.2 If proof objects present, run PROOF_CHECK for them.

2.4.3 Metrics

For each (n, m):
	1.	cycles_per_SAT_CHECK = total_cycles_sat / count_sat.
	2.	cycles_per_PROOF_CHECK (if applicable).
	3.	instances_per_sec for SAT_CHECK and PROOF_CHECK.
	4.	Fraction SAT vs UNSAT.

Implementations MUST describe CNF generation method to ensure comparability.
	3.	Algorithmic Sequence Benchmarks

3.1 Parity and majority

3.1.1 Purpose

Evaluate BEM’s ability to compute global sequence properties (parity and majority) with O(1)-per-step cost and generalization across sequence lengths.

3.1.2 Task definition

Input:
	1.	Binary sequence x ∈ {0,1}^T.

Tasks:
	1.	Parity:
y = XOR_{t=1..T} x_t.
	2.	Majority:
y = 1 if sum_{t=1..T} x_t ≥ T/2, else 0.

BEM interaction:
	1.	Sequence is presented token-by-token.
	2.	At final step T, BEM outputs prediction y_hat ∈ {0,1} (or probability p(y=1)).
	3.	Loss is 0/1 accuracy or cross-entropy.

3.1.3 Protocol

Training:
	1.	Choose T_train_range (e.g. T in {16, 32, 64}).
	2.	Train BEM with learning enabled on sequences with T sampled from T_train_range.

Testing:
	1.	Test lengths T in {T_train/2, T_train, 2T_train, 4T_train}.
	2.	For each T:
2.1 Sample test sequences of size at least 10^4.
2.2 Run BEM in evaluation mode (no learning or structural updates).

3.1.4 Metrics

For each T:
	1.	accuracy(T) = fraction of correct outputs.
	2.	cycles_per_token(T) = average cycles per token.
	3.	cycles_per_sequence(T) = T * cycles_per_token(T).

Implementations MUST report:
	1.	Whether parity and majority share experts, or use separate specialized experts.
	2.	Any explicit state-encoding choices (e.g. dedicated bits for running parity).

3.2 Balanced parentheses (Dyck-1)

3.2.1 Purpose

Test BEM’s ability to emulate stack-like behavior and detect well-formedness under length and depth generalization.

3.2.2 Task definition

Alphabet:

Sigma = { “(”, “)” }.

A sequence s ∈ Sigma^T is:
	1.	Valid if it corresponds to a Dyck-1 well-formed parentheses string (never negative depth, final depth zero).
	2.	Invalid otherwise (e.g. extra closing, missing closing, mismatched patterns).

Two variants:
	1.	Online validity:
At each prefix t, predict whether prefix s[1..t] is still potentially valid (valid so far).
	2.	Final validity:
At final step T, predict whether entire s is valid.

3.2.3 Protocol

Training:
	1.	Choose T_train, D_train.
	2.	Generate sequences with length ≤ T_train and nesting depth ≤ D_train.
	3.	Include mixture of valid and invalid sequences.
	4.	Train BEM on online or final variant (or both).

Testing:
	1.	T_test in {T_train, 2T_train, 4T_train}.
	2.	D_test in {D_train, 2D_train} where feasible.
	3.	For each (T_test, D_test):
3.1 Sample sequences with controlled length and depth.
3.2 Evaluate BEM’s predictions.

3.2.4 Metrics

Per (T, D):
	1.	accuracy_online(T, D) if online variant is used.
	2.	accuracy_final(T, D).
	3.	error_vs_depth(T, d) for depth bucket d.
	4.	cycles_per_token(T, D).

3.3 Key-value retrieval

3.3.1 Purpose

Evaluate BEM’s ability to store and retrieve associations over a context with distractors, similar to key-value attention behavior.

3.3.2 Task definition

Format:

[START] k1 v1 k2 v2 … kK vK [SEP] q [END]

where:
	1.	K key-value pairs.
	2.	Keys k_i from key vocabulary K_voc.
	3.	Values v_i from value vocabulary V_voc.
	4.	Query q ∈ K_voc.
	5.	Target output is v_j such that k_j = q.

Sequences are tokenized at key/value symbol granularity.

3.3.3 Protocol

Parameters:
	1.	L: total sequence length.
	2.	K: number of key-value pairs.
	3.	|K_voc|, |V_voc|: vocabulary sizes.

Training:
	1.	Choose moderate values for (L_train, K_train).
	2.	Train BEM with learning enabled on randomly sampled sequences.

Testing:
	1.	Vary L and K beyond training values:
1.1 L in {L_train, 2L_train}.
1.2 K in {K_train, 2K_train}.
	2.	Vary query position (early, middle, late).
	3.	Evaluate retrieval accuracy.

3.3.4 Metrics

Per configuration (L, K):
	1.	retrieval_accuracy(L, K) = fraction where predicted v_hat = v_target.
	2.	accuracy_by_position:
2.1 accuracy when q is early.
2.2 accuracy when q is middle.
2.3 accuracy when q is late.
	3.	cycles_per_token(L, K).
	4.	cycles_per_query (final step).

Implementations SHOULD describe how information is encoded in STATE and SHARED (e.g. dedicated slots for key-value pairs, expert structures for lookup).
	4.	Language-Model-Style Benchmarks

4.1 Character-level language modeling

4.1.1 Purpose

Measure BEM’s ability to act as a small character-level model with bounded fast-path cost.

4.1.2 Task definition

Corpus:
	1.	A small text corpus (e.g. fiction excerpts, technical notes).
	2.	Normalized to a fixed character set C (e.g. 96 printable ASCII).

Task:

Next-character prediction:
	1.	Input: context c_1..c_t.
	2.	Output: distribution p(c_{t+1} | c_1..c_t).
	3.	Loss: cross-entropy over characters.

4.1.3 Protocol

Data split:
	1.	Train set, validation set, test set.

Training:
	1.	Fixed training budget (e.g. number of tokens or steps).
	2.	BEM configured with:
2.1 Fixed N, K, W, expert budget N_expert_max.
2.2 Learning and structural updates enabled.
	3.	Optimize on train set, monitor validation BPC.

Testing:
	1.	Evaluate on test set.
	2.	Record predictive distributions and cost metrics.

4.1.4 Metrics

On test:
	1.	bits_per_char (BPC) = average cross-entropy in bits per character.
	2.	perplexity = 2^{BPC}.
	3.	total_tokens_seen during training.
	4.	cycles_per_char on inference (no learning).
	5.	chars_per_second per core.

Implementations MUST report:
	1.	Whether per-character predictions share the same BEM configuration as other benchmarks.
	2.	Any task-specific tuning (e.g. selecting experts specializing in character classes).

4.2 Token-level code modeling (optional)

4.2.1 Purpose

Optionally measure BEM’s ability to model code sequences with syntactic structure.

4.2.2 Task definition

Corpus:
	1.	Code files in a chosen language (C, Rust, Python, etc.).
	2.	Tokenized using a simple lexer into identifiers, keywords, operators, literals.

Task:

Next-token prediction with cross-entropy loss.

4.2.3 Metrics

On test:
	1.	token-level perplexity.
	2.	Syntax-related error rates:
2.1 Frequency of unmatched brackets.
2.2 Inconsistent indentation (if applicable).
	3.	cycles_per_token.
	4.	Learning and Structural Update Benchmarks

5.1 Contextual bandit

5.1.1 Purpose

Evaluate online weight updates (U_gate_weight) under contextual bandit loss and no-regret behavior.

5.1.2 Task definition

At each round t:
	1.	Context x_t ∈ X (e.g. small binary vector).
	2.	K actions (arms) a ∈ {1..K}.
	3.	Reward r_t ∈ [0,1] drawn from R(a, x_t).

BEM:
	1.	Observes x_t via O(s_t, M_t).
	2.	Selects internal expert i_t (or expert group) that maps to actions.
	3.	Induces action a_t over arms.
	4.	Receives reward r_t and uses U_gate_weight to update weights w_i.

5.1.3 Protocol

Parameters:
	1.	T: total rounds.
	2.	K: arms count.
	3.	Context distribution: e.g. uniform over {0,1}^d.

Reward mapping:
	1.	For each arm a, define a linear threshold model:
E[r_t | a, x_t] = sigmoid(w_a · x_t)
	2.	Keep w_a fixed and unknown to BEM.

Benchmark:
	1.	Run for T rounds with learning enabled.
	2.	Record rewards, actions, and any structural updates (if allowed).

5.1.4 Metrics
	1.	Cumulative reward sum_{t} r_t.
	2.	Baseline reward of best fixed arm in hindsight.
	3.	Cumulative regret R_T.
	4.	Regret scaling R_T vs T.
	5.	Number of gate weight updates and cost (cycles).

Implementations MUST report whether structural changes are enabled; if so, they MUST separate performance with and without structural updates.

5.2 Non-stationary bandit

5.2.1 Purpose

Measure adaptation speed and structural changes under distribution shifts.

5.2.2 Task definition

Same as contextual bandit, but reward mapping changes at change-points:
	1.	Tasks A, B, C, … with different sets of w_a^task.

At specified steps t_change_j:
	1.	Change underlying reward mapping from one task to the next.

5.2.3 Metrics

For each change:
	1.	Immediate regret spike after change.
	2.	Adaptation time:
Number of steps until moving-average regret falls below a threshold.
	3.	Structural changes:
Number of new experts spawned, merges, splits.
	4.	Compute overhead of adaptation:
Extra cycles in mid-path and slow-path updates per change.

5.3 Safety-constrained program fragment

5.3.1 Purpose

Evaluate effectiveness and cost of verification-gated structural optimization.

5.3.2 Task definition

Define a small BEM-controlled process with a safety property, for example:
	1.	An index i must remain within bounds [0, M).
	2.	A BAD flag must never be set to 1.

BEM is allowed to:
	1.	Propose structural patches Delta (e.g. micro-optimizations to expert circuits).
	2.	Apply patches only after VC(Delta) is verified.

Process:
	1.	The process runs for many steps (e.g. 10^8 steps).
	2.	BEM periodically proposes optimizations (e.g. reduce gate counts, fuse experts).
	3.	Each patch is checked by SAT/Hoare.

5.3.3 Metrics

Over the run:
	1.	N_patch_proposed, N_patch_accepted, N_patch_rejected.
	2.	Average verification cost per accepted patch:
cycles_per_verification = verification_cycles / N_patch_accepted.
	3.	Ratio of verification cost to fast-path cost:
verification_cycles / fast_path_cycles, e.g. per 10^6 tokens.
	4.	Observed safety violations (should be zero under correct implementation).
	5.	Reporting and Scoring

6.1 Configuration reporting

For each benchmark run, implementations MUST report:
	1.	Hardware:
1.1 CPU model.
1.2 SIMD width.
1.3 Clock frequency.
	2.	BEM configuration:
2.1 N, K, W.
2.2 G_max, D_max.
2.3 ROUTE parameters H, B, C_entry, K_cand.
2.4 N_expert_max.
2.5 n_max, m_max.
	3.	Learning settings:
3.1 Learning rate schedule.
3.2 Structural updates enabled/disabled.
3.3 B_flip.

6.2 Results reporting

Per benchmark:
	1.	Microbenchmarks:
1.1 cycles_per_op for BIT-ALU.
1.2 cycles_per_query for ROUTE.
1.3 cycles_per_batch for experts.
1.4 cycles_per_SAT_CHECK and per_PROOF_CHECK.
	2.	Algorithmic tasks:
2.1 accuracies, errors vs length/depth/position.
2.2 cycles_per_token and cycles_per_sequence.
	3.	LM-style tasks:
3.1 BPC or perplexity.
3.2 cycles_per_char or cycles_per_token.
	4.	Learning tasks:
4.1 Regret curves.
4.2 Adaptation times.
4.3 Structural update counts and costs.
	5.	Safety/verification tasks:
5.1 Patch stats.
5.2 Verification cost ratios.
5.3 Safety violations.

6.3 Composite scores (optional)

To compare implementations, composite scores MAY be defined, for example:
	1.	Fast-path score:
S_fast = tokens_per_second × accuracy_on_core_tasks.
	2.	Learning score:
S_learn = 1 / (1 + normalized_regret).
	3.	Safety score:
S_safe = 1 / (1 + verification_cost_ratio), with penalty for any safety violations.

Composite scoring is not mandatory in v0.0.1 and may be standardized in later versions.

6.4 Versioning

This document defines BEM Benchmark Suite v0.0.1.

Future versions MAY:
	1.	Introduce fixed reference datasets for some tasks.
	2.	Add standardized composite scores and leaderboards.
	3.	Extend benchmarks to multi-core or distributed BEM configurations while preserving per-core fast-path semantics.
