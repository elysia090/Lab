BEM Benchmark Suite v0.0.1 – Information-Theoretic and CS-Oriented Specification (Revised)
	0.	Scope and Objectives

0.1 Purpose

This document defines BEM Benchmark Suite v0.0.1 as a family of probabilistic tasks, metrics, and protocols for evaluating implementations of the Boolean Expert Machine (BEM) v0.0.1.

The suite is designed to measure, in information-theoretic and CS terms:
	1.	Fast-path performance
1.1 Per-step cost in cycles under a fixed cost model.
1.2 Scaling with BEM parameters (N, K, W, number of experts, ROUTE parameters).
1.3 Asymptotic behavior in sequence length T for fixed configuration.
	2.	Algorithmic capability
2.1 Risk (error probability) on structured sequence tasks defined as distributions over (X, Y).
2.2 Generalization in T and difficulty parameters (e.g. nesting depth, number of keys).
2.3 Ability to implement known algorithmic behaviors (parity, stack-like behavior, dictionary lookup) with O(1) fast-path cost.
	3.	Language-model-style performance
3.1 Empirical cross-entropy and bits-per-symbol on small corpora.
3.2 Comparison to entropy rate of the data-generating distribution where known.
	4.	Online learning and structural update behavior
4.1 Regret under contextual bandit tasks.
4.2 Adaptation under non-stationary environments.
4.3 Cost and rate of verification-gated structural changes.

0.2 Non-goals

The benchmark suite does not:
	1.	Fix specific public datasets; it defines task families and statistical regimes.
	2.	Impose mandatory pass/fail thresholds.
	3.	Constrain implementation details beyond:
3.1 Semantic conformance to BEM v0.0.1.
3.2 Bounded fast-path cost per step.

0.3 Evaluation viewpoint

All tasks are defined as probability distributions over sequences and labels. For each benchmark:
	1.	There is an underlying distribution P over inputs X and outputs Y (or rewards R).
	2.	A BEM instance with configuration C induces a (possibly adaptive) stochastic policy pi_C.
	3.	For supervised-like tasks, we evaluate empirical risk and cross-entropy as estimators of expected risk.
	4.	For bandit-like tasks, we evaluate regret relative to an oracle or a reference class.
	5.	For LM-like tasks, we evaluate cross-entropy relative to the empirical distribution of the test corpus.
	6.	Global Assumptions, Notation, and Cost Model

1.1 Probability and datasets

Let:
	1.	(Omega, F, P) be a probability space underlying each benchmark family.
	2.	X denote input random variables, Y denote targets, R denote rewards.
	3.	D_train, D_val, D_test be i.i.d. samples from the specified distribution or from a specified generative process.

Empirical quantities:
	1.	For losses l(theta; x, y) we define empirical risk on a sample S = {(x_i, y_i)} as:
R_hat_S(theta) = (1 / |S|) * sum_i l(theta; x_i, y_i)
	2.	For sequence prediction with per-symbol loss l_t(theta; c_{1..t}, c_{t+1}):
H_hat(theta) = (1 / n) * sum_{t=1..n} l_t(theta; history, next_symbol)
where l_t is measured in bits if log base 2 is used.

1.2 Complexity and cost model

We use a simple single-core cost model:
	1.	BEM-CORE executes instructions on a single CPU core.
	2.	Co-processors (BIT-ALU, ROUTE, SAT, HASH, LOG) are modeled as instructions or calls whose cost is charged in cycles to the same core.
	3.	Fast-path cost per logical step must be O(1) in T by design; benchmarks measure actual constant factors.

Units:
	1.	cycles: core cycles (preferably via hardware counters).
	2.	steps: logical BEM steps (e.g. tokens).
	3.	cycles_per_step = total_cycles / total_steps.
	4.	tokens_per_second = (clock_frequency_hz) / cycles_per_step.

Implementations MUST report:
	1.	CPU model, microarchitecture, and reported base clock.
	2.	SIMD width W_hw (bits) and effective BEM lane width W (lanes).
	3.	Whether cycle counts are measured via counters or estimated.

1.3 BEM configuration parameters

For each benchmark configuration, implementations MUST specify:
	1.	N: global state bits.
	2.	K: shared memory bits.
	3.	W: lanes (logical).
	4.	N_expert_max: maximum experts.
	5.	Expert circuit budget: e.g. gate count G_max or LUT count.
	6.	ROUTE parameters: index fanout and bounds (e.g. H, B, C_entry, K_cand) or equivalent.
	7.	SAT/Hoare constraints: n_max variables, m_max clauses.
	8.	Learning hyperparameters (when applicable):
8.1 Bandit learning rates and exploration parameters.
8.2 Structural update budgets (e.g. B_flip or patch budget per 10^6 steps).
	9.	Microbenchmarks (Capacity and Constant Factors)

These benchmarks are deterministic or pseudo-random, not statistical tasks. They provide a baseline capacity measure for bit-sliced operations, routing, and verification.

2.1 BIT-ALU throughput

2.1.1 Purpose

Measure throughput and latency of bit-sliced bitvector operations on W lanes. This approximates the capacity of the BEM fast-path for Boolean transformations.

2.1.2 Setup

Parameters:
	1.	W: lane width (from BEM configuration).
	2.	L_loop: iterations, L_loop >= 10^7.
	3.	op_set: operations:
	•	AND, OR, XOR, NOT
	•	shift_left, shift_right by fixed constant
	•	rotate_left, rotate_right by fixed constant
	•	popcount

Procedure for each op:
	1.	Allocate one or more W-bit words as inputs (in STATE or registers).
	2.	Initialize inputs with pseudorandom bits (fixed seed).
	3.	Run a tight loop of L_loop times:
	•	apply op to operands
	•	update a scalar checksum (e.g. XOR or ADD) to inhibit dead-code elimination
	4.	At the end, store the checksum to SHARED.

2.1.3 Metrics

For each op:
	1.	cycles_per_op = (total_cycles) / L_loop
	2.	ops_per_second = clock_frequency_hz / cycles_per_op
	3.	bits_per_second = ops_per_second * W

Implementations MUST document:
	1.	Exact code pattern used.
	2.	How cycles were measured or estimated.

2.2 ROUTE query performance

2.2.1 Purpose

Measure time and quality of ROUTE_QUERY as index size grows. This approximates capacity of the ANN/ROUTE unit used for expert routing.

2.2.2 Setup

Define a similarity function sim(u, v) as in the BEM spec (e.g. Gray-coded Hamming similarity). Let U_expert be a finite set of identifiers.

Parameters:
	1.	N_nodes_set = {10^4, 10^5, 10^6}
	2.	K_cand: candidate list size (e.g. 16 or 32)
	3.	Routing configuration parameters (H, B, C_entry, or equivalent)
	4.	Query count Q >= 10^5 per N_nodes

Index:
	1.	Sample U_expert from U (uniform or structured as specified).
	2.	Build ROUTE index as in the implementation.

Queries:
	1.	For each query i in 1..Q:
1.1 Sample q from a mixture of:
- in-index identifiers (prob p_in)
- new identifiers (prob 1 - p_in)
1.2 Issue ROUTE_QUERY(q, class_expert, shard_filter, K_cand)
1.3 Record the returned candidate set C_i

Optional ground truth:
	1.	For a subset Q_truth of queries (e.g. 10^3), compute exact nearest neighbors in U_expert under sim, using a reference implementation.

2.2.3 Metrics

For each N_nodes:
	1.	cycles_per_query = total_cycles / Q
	2.	queries_per_second = clock_frequency_hz / cycles_per_query
	3.	Recall metrics (if ground truth used):
	•	recall@k = P(nearest neighbor in C_i[1..k])
	•	optionally mean rank of true neighbor

Implementations MUST report:
	1.	N_nodes, K_cand, routing parameters
	2.	Distribution over bucket occupancy (min, max, mean)
	3.	cycles_per_query, queries_per_second
	4.	recall@k for chosen k if ground truth used

2.3 Expert evaluation cost

2.3.1 Purpose

Measure cost per expert evaluation in BEM_EXPERT_BATCH for synthetic experts with controlled gate counts. This estimates constant factors for simultaneous evaluation on W lanes.

2.3.2 Setup

Parameters:
	1.	G_set = {64, 128, 256, 512} or a feasible subset
	2.	W: lane width
	3.	L_loop >= 10^6 per G

For each G:
	1.	Construct an expert e_G with:
	•	fixed n_i and m_i (e.g. n_i = 32, m_i = 8)
	•	Boolean circuit C_rep with about G gates (in chosen representation, e.g. ANF or ROBDD)
	•	R_spec selecting bits from STATE/SHARED
	•	W_spec writing back to STATE/SHARED
	2.	Ensure the expert is non-trivial (non-constant, non-identity).
	3.	Initialize STATE lanes with pseudorandom bits (fixed seed).
	4.	Run L_loop iterations of:
	•	BEM_EXPERT_BATCH(e_G, STATE, SHARED)
	•	small random perturbations to inputs (e.g. flipping a few bits)
	•	maintain a checksum written to SHARED

2.3.3 Metrics

For each G:
	1.	cycles_per_batch = total_cycles / L_loop
	2.	cycles_per_lane = cycles_per_batch / W
	3.	batches_per_second = clock_frequency_hz / cycles_per_batch

Implementations MUST specify:
	1.	Circuit representation type
	2.	Definition of gate count G
	3.	Any vectorization or bit-slicing details that affect W

2.4 SAT/Hoare kernel cost

2.4.1 Purpose

Measure cost per SAT_CHECK and PROOF_CHECK for bounded CNF sizes, to characterize verification overhead.

2.4.2 Setup

Parameters:
	1.	n_set = {32, 64, 128} variables
	2.	For each n, m_set = {4n, 8n} clauses
	3.	L_inst = 10^4 instances per (n, m)

For each (n, m):
	1.	Generate CNFs Phi_j with:
	•	n variables, m clauses
	•	clause lengths in a small range (e.g. 3 to 5)
	•	mixture of SAT and UNSAT instances
	2.	Optionally generate proofs for UNSAT instances.

Benchmark:
	1.	Run SAT_CHECK on all Phi_j
	2.	If proofs exist, run PROOF_CHECK on corresponding (Phi_j, proof_j)

2.4.3 Metrics

For each (n, m):
	1.	cycles_per_SAT_CHECK = total_sat_cycles / L_inst
	2.	cycles_per_PROOF_CHECK (if applicable)
	3.	instances_per_second
	4.	empirical fraction SAT

Implementations MUST document the CNF generation process.
	3.	Algorithmic Sequence Benchmarks (Risk vs Length and Difficulty)

All tasks in this section are specified as distributions over sequences and labels. We measure error probabilities and scaling with length T and difficulty parameters.

3.1 Parity and majority

3.1.1 Task definition

For a given T, let X_T be sequences x in {0,1}^T drawn i.i.d. from Bernoulli(1/2) per bit.

Labels:
	1.	Parity:
Y = XOR_{t=1..T} x_t
	2.	Majority:
Y = 1 if sum_t x_t >= T/2, else 0

BEM interaction:
	1.	Sequence x_1..x_T is streamed to BEM, one bit token per step.
	2.	At step T, BEM produces prediction y_hat in {0,1} or a probability p_hat = P(y = 1 | history).
	3.	Loss:
	•	0/1 loss: l_01 = 1[y_hat != Y]
	•	cross-entropy: l_ce = - Y log p_hat - (1 - Y) log(1 - p_hat)

3.1.2 Protocol

Training regime:
	1.	T_train_set = {T1, T2, …} (e.g. {16, 32, 64})
	2.	D_train: sequences with length T sampled from T_train_set
	3.	BEM is trained with online learning enabled until a fixed budget of sequences or steps.

Test regime:
	1.	T_test_set extends T_train_set:
	•	e.g. {T1/2, T1, 2T1, 4T1} or {T_train_max, 2 T_train_max, 4 T_train_max}
	2.	For each T in T_test_set:
	•	draw at least 10^4 sequences
	•	run BEM in evaluation mode (learning disabled).

3.1.3 Metrics

For each T:
	1.	empirical risk (0/1):
R_hat_01(T) = P_hat(y_hat != Y)
	2.	empirical cross-entropy (if probabilistic outputs):
H_hat_ce(T)
	3.	cycles_per_token(T)
	4.	cycles_per_sequence(T) = T * cycles_per_token(T)

Theoretical note: For an ideal implementation with O(1)-state parity tracking, R_01(T) can be 0 for all T; this task checks whether BEM can realize such finite-state encodings for increasing T.

3.2 Balanced parentheses (Dyck-1)

3.2.1 Task definition

Alphabet Sigma = { “(”, “)” }.

Let S_T be sequences s in Sigma^T, generated by:
	1.	Sample s from a specified generator that produces:
	•	valid Dyck-1 strings (never negative depth, final depth zero)
	•	invalid strings (from perturbations like extra closing, missing closing, etc.)
at defined proportions (e.g. 0.5 valid / 0.5 invalid).

Two labeling variants:
	1.	Online validity:
At each t, Y_t = 1 if the prefix s[1..t] is prefix-valid (depth >= 0), else 0.
	2.	Final validity:
Y_final = 1 if s is fully valid Dyck-1, else 0.

3.2.2 Protocol

Parameters:
	1.	T_train: maximum length seen in training
	2.	D_train: maximum nesting depth in training
	3.	T_test: set including T_train and larger values (e.g. T_train, 2T_train, 4T_train)
	4.	D_test: {D_train, 2 D_train} where feasible

Training:
	1.	Generate sequences with length <= T_train and depth <= D_train.
	2.	Train BEM on online or final labels (or both).

Testing:
	1.	For each (T, D) in T_test x D_test:
	•	generate sequences with length approx T and depth distribution truncated at D
	•	evaluate BEM in evaluation mode

3.2.3 Metrics

For each (T, D):
	1.	online accuracy (if used):
accuracy_online(T, D) = fraction of correct Y_t predictions over all t
	2.	final accuracy:
accuracy_final(T, D) on Y_final
	3.	error_vs_depth(T, d):
error rate conditioned on sequences with max depth in bucket d
	4.	cycles_per_token(T, D)

This task probes whether BEM can implement a bounded representation of depth that generalizes beyond training T and D.

3.3 Key-value retrieval

3.3.1 Task definition

We define a synthetic associative memory task over a discrete vocabulary.

Vocabulary:
	1.	Key set K_voc of size |K_voc|
	2.	Value set V_voc of size |V_voc|

Sequence format for a sample:

[START] k1 v1 k2 v2 … kK vK [SEP] q [END]

where:
	1.	K key-value pairs (k_i, v_i) with k_i in K_voc, v_i in V_voc
	2.	Query q in K_voc
	3.	Label Y = v_j where k_j = q; if q not present, Y is a special NULL value or random.

Tokenization: one token per key or value symbol.

3.3.2 Protocol

Parameters:
	1.	L: total sequence length (determined by K and tokenization)
	2.	K: number of key-value pairs (task difficulty)
	3.	|K_voc|, |V_voc|

Training:
	1.	Fix L_train, K_train, |K_voc|, |V_voc|.
	2.	Sample sequences by drawing keys and values uniformly or from specified distributions.
	3.	Train BEM with online learning.

Testing:
	1.	Vary:
	•	K in {K_train, 2 K_train}
	•	query position (early, middle, late)
	•	optionally |K_voc|, |V_voc|
	2.	For each config, draw at least 10^4 sequences.
	3.	Evaluate BEM in evaluation mode.

3.3.3 Metrics

For each (L, K) and query position regime:
	1.	retrieval_accuracy(L, K) = P_hat(v_hat = Y)
	2.	accuracy_by_position (early/middle/late)
	3.	cycles_per_token(L, K)
	4.	cycles_per_query (cost of final answer step)

Implementations SHOULD document how key-value information is represented in STATE/SHARED and whether specialized experts are used.
	4.	Language-Model-Style Benchmarks (Cross-Entropy and Entropy Rate)

4.1 Character-level language modeling

4.1.1 Task definition

Let C be a fixed character set (e.g. 96 printable ASCII). A corpus is a finite sequence c_1..c_n in C.

Task: next-character prediction.

At each position t (1 <= t < n):
	1.	Input: prefix c_1..c_t.
	2.	Output: predictive distribution p_theta(c_{t+1} | c_1..c_t) over C.
	3.	Loss per position:
l_t = - log_2 p_theta(c_{t+1} | c_1..c_t) (bits)

4.1.2 Protocol

Data split:
	1.	D_train, D_val, D_test as contiguous or random splits.
	2.	Corpus statistics (size, empirical character distribution) MUST be reported.

Training:
	1.	Fix:
	•	BEM configuration (N, K, W, expert budget)
	•	training budget: number of tokens or gradient steps
	2.	Train on D_train with online or mini-batch updates.
	3.	Use D_val for early stopping or hyperparameter selection.

Testing:
	1.	Freeze parameters / structure (no learning).
	2.	Run on D_test and compute cross-entropy:
H_hat_test(theta) = (1 / |D_test|) * sum_t l_t
	3.	Optional: estimate entropy rate H_emp of the corpus by standard methods (e.g. n-gram models) for reference.

4.1.3 Metrics

On D_test:
	1.	bits_per_char = H_hat_test(theta)
	2.	perplexity = 2^{bits_per_char}
	3.	tokens_per_second on inference:
	•	tokens_per_second = clock_frequency_hz / cycles_per_char
	4.	cycles_per_char

Additional:
	1.	Training tokens consumed
	2.	Whether the same BEM configuration is used for all benchmarks or specialized for LM

4.2 Token-level code modeling (optional)

4.2.1 Task definition

Similar to 4.1 but over a token vocabulary derived from code (identifiers, keywords, operators, etc.). Per-token loss and perplexity are measured.

4.2.2 Metrics
	1.	token-level bits_per_token and perplexity
	2.	syntax-related error rates (optional):
	•	unmatched brackets, etc.
	3.	cycles_per_token
	4.	Learning and Structural Update Benchmarks (Regret and Adaptation)

5.1 Contextual bandit (stochastic)

5.1.1 Task definition

We consider a standard K-armed contextual bandit.

At each round t:
	1.	Context X_t in {0,1}^d drawn i.i.d. from a fixed distribution P_X (e.g. uniform).
	2.	Action set A = {1,…,K}.
	3.	For each action a, reward R_t(a) in [0,1] with expectation:
E[R_t(a) | X_t = x] = mu_a(x) = sigma(w_a . x)
where sigma is a bounded activation (e.g. logistic), w_a are fixed unknown parameters.

BEM:
	1.	Observes X_t via the observation function O(s_t, M_t).
	2.	Chooses expert(s), inducing an action a_t in A.
	3.	Receives reward R_t = R_t(a_t).
	4.	Updates bandit weights and possibly structure.

5.1.2 Protocol

Parameters:
	1.	d: context dimension (e.g. 8, 16)
	2.	K: number of actions (e.g. 4, 8)
	3.	T: horizon (e.g. 10^5 or 10^6)
	4.	Distribution P_X and weight vectors w_a are fixed and reported up to randomness seeds or descriptions.

Benchmark:
	1.	Initialize BEM.
	2.	Run for T rounds with learning enabled.
	3.	Log actions, rewards, and structural events.

Oracle:
	1.	For each x, define a* = argmax_a mu_a(x).
	2.	The optimal expected reward at x is mu_{a*}(x).

Regret:
	1.	Regret after T rounds:
R_T = sum_{t=1..T} (mu_{a*t}(X_t) - mu{a_t}(X_t))
where a*_t is the optimal arm given X_t.

Since mu_a are known in the generator, R_T can be computed.

5.1.3 Metrics
	1.	empirical regret R_T
	2.	normalized regret R_T / T
	3.	regret curve R_t versus t (log-log or log-linear)
	4.	number of bandit updates and their cost in cycles
	5.	total cycles per round (fast-path + learning overhead)

Implementations MUST report whether structural changes (expert split/merge) are enabled; if yes, they MUST report the number and type of structural patches during the run.

5.2 Non-stationary bandit

5.2.1 Task definition

Same contextual bandit as 5.1, but with piecewise-stationary segments.

Let there be J segments with change points 1 < t_1 < … < t_J:
	1.	On segment j (t_{j-1}+1 .. t_j), each arm a has parameters w_a^{(j)}.
	2.	At each change point, reward mapping changes abruptly.

5.2.2 Protocol

Parameters:
	1.	same d, K, base T
	2.	number of changes J and change interval lengths
	3.	different parameter sets w_a^{(j)}

Benchmark:
	1.	Run BEM for T rounds with learning and structural updates enabled.
	2.	For each change point, record time and subsequent behavior.

5.2.3 Metrics

For each change:
	1.	immediate regret spike:
	•	regret accumulated in a short window after change
	2.	adaptation time:
	•	minimal tau such that average regret over a window of size W_reg falls below a threshold
	3.	number and type of structural updates triggered by the change
	4.	additional compute cost (mid/slow-path cycles) attributable to adaptation

5.3 Safety-constrained program with verified optimization

5.3.1 Task definition

We consider a simple safety property S (e.g. an index i remains in [0, M), or a BAD flag is never set).

The BEM-controlled process:
	1.	Executes a loop for T_large steps, manipulating a state variable i and other registers.
	2.	Safety property S is specified in Hoare/VC form.
	3.	BEM is allowed to propose structural patches Delta (e.g. circuit simplifications or fusions).

Patch application is allowed only if:
	1.	VC(Delta) is proven UNSAT by SAT/Hoare unit.
	2.	Optional additional constraints are met (e.g. PoX score threshold).

5.3.2 Protocol

Parameters:
	1.	T_large (e.g. 10^8 steps)
	2.	Patch proposal budget (e.g. at most P_max proposals per 10^6 steps)
	3.	Safety property S fully specified

Benchmark:
	1.	Run process with learning and patch proposals enabled.
	2.	BEM periodically proposes patches for hot experts.
	3.	Each patch is checked by SAT/Hoare; if accepted, it modifies EXPERT/CFG.

5.3.3 Metrics

Over the run:
	1.	N_patch_proposed, N_patch_accepted, N_patch_rejected
	2.	verification_cycles_total and:
verification_cycles_per_accepted = verification_cycles_total / N_patch_accepted
	3.	Ratio:
verification_to_fast_path_ratio = verification_cycles_total / fast_path_cycles_total
	4.	Observed safety violations (must be zero for correct implementations)
	5.	Change in fast-path cost:
	•	average cycles_per_step before and after significant patches
	6.	Reporting and Scoring

6.1 Configuration reporting

For any published result, implementers MUST report:
	1.	Hardware
	•	CPU model and microarchitecture
	•	SIMD width W_hw
	•	base clock frequency
	2.	BEM configuration
	•	N, K, W
	•	N_expert_max, G_max
	•	ROUTE parameters
	•	SAT/Hoare limits
	3.	Learning and structural settings
	•	bandit hyperparameters
	•	GRPO-related parameters if used
	•	structural update budgets and policies

6.2 Results reporting

For each benchmark:
	1.	Microbenchmarks
	•	BIT-ALU: cycles_per_op, bits_per_second
	•	ROUTE: cycles_per_query, queries_per_second, optional recall
	•	BEM_EXPERT_BATCH: cycles_per_batch and per_lane
	•	SAT/Hoare: cycles_per_SAT_CHECK, per_PROOF_CHECK, instance distributions
	2.	Algorithmic sequence tasks
	•	empirical risks / accuracies as functions of T, depth, K, etc.
	•	cycles_per_token, cycles_per_sequence as functions of T and difficulty
	3.	LM-style tasks
	•	bits_per_char or bits_per_token
	•	perplexity
	•	cycles_per_char or cycles_per_token
	•	training budget used
	4.	Learning and structural tasks
	•	regret curves R_t
	•	adaptation times
	•	structural update counts and types
	•	overhead in cycles for learning and verification
	5.	Safety tasks
	•	patch proposal/acceptance statistics
	•	verification cost ratios
	•	safety violations, if any

6.3 Optional composite scores

Composite scores MAY be defined for convenience:
	1.	Fast-path score:
S_fast = f(tokens_per_second, accuracy_on_core_tasks)
Example: S_fast = tokens_per_second * (1 - error_rate_on_parity_and_kv)
	2.	Learning score:
S_learn = 1 / (1 + normalized_regret)
	3.	Safety score:
S_safe = g(verification_to_fast_path_ratio, safety_violations)
where any non-zero safety violation severely penalizes S_safe.

These are informative and not normative for v0.0.1.

6.4 Versioning

This document is BEM Benchmark Suite v0.0.1, information-theoretic and CS-oriented revision of the earlier tight specification.

Future versions MAY:
	1.	Fix reference datasets for LM-style tasks.
	2.	Standardize composite scores and leaderboards.
	3.	Add multi-core and distributed BEM benchmarks while preserving per-core fast-path semantics and clear probabilistic task definitions.
