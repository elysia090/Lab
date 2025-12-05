BEM TEACH AGI Task Grammar v0.0.1
Primitive Families and Template Specifications
	0.	Scope

0.1 Goal

This document specifies a set of task template families that SHOULD be included in the TEACH grammar when the long-term objective is broad general capability (AGI-oriented) rather than narrow specialization.

0.2 Assumptions
	1.	TEACH maintains a library of task templates T = {t_1, …, t_M}.
	2.	Each template t has:
	•	a domain tag dom(t) (algorithmic, environment, tool, language, meta, safety),
	•	a difficulty descriptor diff(t),
	•	a generator G_t that produces episodes consistent with BEM’s environment interface,
	•	a reward definition R_t,
	•	and optional structural constraints (e.g. safety invariants).
	3.	TEACH supports compositional operators:
	•	SEQ(a, b), PAR(a, b), NEST(a, k), MASK(a, pattern), INTERLEAVE(a, b), and variants.
	4.	TEACH_STEP selects templates using a bandit policy over context and difficulty bands.

The goal here is to define the primitive families and their template signatures; implementation details of TEACH_STEP and bandits are out of scope.
	1.	Grammar Model

1.1 Template interface

Each task template t defines:
	1.	A parameter space Θ_t (difficulty, sizes, noise levels).
	2.	A generative process for episodes:
	•	Sample θ ∼ P_t over Θ_t.
	•	Sample an environment instance E from Env_t(θ).
	•	Run BEM in interaction with E for T steps (T may be random).
	3.	A reward function R_t:
	•	For each step or episode, compute reward r_t according to R_t.
	4.	Evaluation metrics for that template family (accuracy, regret, BPC, etc.).

Notation (per template t):
	•	Input/observation space: O_t
	•	Action space: A_t
	•	Hidden state space (environment side): H_t
	•	Reward: r_t = f_t(H_t, A_t)

TEACH sees t only through its meta statistics (pass rates, regret, novelty) and does not need to inspect Env_t internals.

1.2 Primitive families

We group templates into five high-level families:
	1.	Algorithmic and formal reasoning tasks
	2.	World and causal environment tasks
	3.	Tool and external oracle tasks
	4.	Language and communication tasks
	5.	Meta-learning, self-modeling, and safety tasks

For TEACH, each primitive is a parameterized template family t_family(param_vector). The grammar contains these families plus composition operators.
	2.	Algorithmic and Formal Core Families

2.1 Arithmetic and algebra

2.1.1 Template: ARITH_EXPR

Name: ARITH_EXPR(bit_width, n_vars, depth, ops_set)

Domain: algorithmic

Input format:
	•	A sequence encoding of an arithmetic expression over integers:
	•	Variables x_1, …, x_{n_vars} with values in [−2^{bit_width−1}, 2^{bit_width−1}−1].
	•	Operators from ops_set ∈ {+, −, ×, max, min}.
	•	Optional parentheses up to nesting depth depth.
	•	The input is serialized as tokens; internally the environment maintains exact integer values.

Output:
	•	A single integer y ∈ Z in the same bit_width range representing the exact evaluation of the expression.

Task variants:
	1.	Direct evaluation:
	•	At the end of the sequence, BEM outputs y_hat.
	•	Reward: 1 if y_hat = y, else 0, or negative absolute error normalized.
	2.	Stepwise prediction:
	•	BEM predicts intermediate sub-expression values at designated points.
	•	Reward: sum of correctness over intermediate predictions.

Difficulty parameters:
	•	bit_width ∈ {8, 16, 32}.
	•	n_vars ∈ {2, …, 8}.
	•	depth ∈ {1, …, 4}.
	•	ops_set subsets (e.g. {+, −} vs {+, −, ×}).

Metric focus:
	•	Exact correctness rate vs bit_width and depth.
	•	Cycles per token and per expression.

2.2 Logic and small SAT/proof tasks

2.2.1 Template: LOGIC_SAT_DECIDE

Name: LOGIC_SAT_DECIDE(n_vars, m_clauses)

Domain: algorithmic

Input format:
	•	A Boolean formula in CNF with:
	•	n_vars variables,
	•	m_clauses clauses,
	•	clause length in a small range (e.g. 2–5 literals),
	•	encoded as a token sequence.

Output:
	•	A binary decision y ∈ {SAT, UNSAT}.

Reward:
	•	1 for correct decision, 0 otherwise.

Difficulty parameters:
	•	n_vars ∈ {4, 8, 16}.
	•	m_clauses ∈ {4n, 8n}.
	•	Structured vs random formulas.

2.2.2 Template: LOGIC_IMPLICATION

Name: LOGIC_IMPLICATION(n_vars, m_A, m_B)

Input:
	•	Two CNFs A and B over the same variable set.

Output:
	•	y ∈ {0,1} indicating whether A ⇒ B logically holds.

Reward:
	•	1 if prediction matches ground truth obtained via PROVER, else 0.

Metric focus:
	•	Decision error rates.
	•	Scaling with n_vars and m.

2.3 Tiny interpreter / VM execution

2.3.1 Template: EXEC_TINY_VM

Name: EXEC_TINY_VM(L_instr, mem_size, ops_set)

Domain: algorithmic

Toy language:
	•	A small stack or register machine with:
	•	few registers or a stack,
	•	operations such as load, store, add, branch_if_zero,
	•	memory size mem_size.

Input:
	•	A program P of length L_instr in the toy language.
	•	Initial memory or register configuration.

Output variants:
	1.	Final register value:
	•	Output value of a designated register at program termination.
	2.	Assertion check:
	•	Input includes specification “at end, R_k must equal C”.
	•	BEM must output whether the assertion holds.

Reward:
	•	1 on exact match / correct assertion decision, 0 otherwise.

Difficulty:
	•	L_instr ∈ {5, 10, 20}.
	•	mem_size ∈ {16, 64}.
	•	Limited branch depth.

Metric focus:
	•	Correctness vs L_instr.
	•	Cycles per instruction simulated.

	3.	World and Causal Environment Families

3.1 Gridworld and object persistence

3.1.1 Template: GRID_NAV

Name: GRID_NAV(size, partial_obs, obstacles)

Domain: environment

Environment:
	•	A 2D grid of size size × size.
	•	An agent with position (x, y).
	•	Optional walls and obstacles.

Observation:
	•	If partial_obs = 0: full grid state.
	•	If partial_obs = 1: local window around agent.

Actions:
	•	A = {UP, DOWN, LEFT, RIGHT, STAY}.

Goal:
	•	Reach a specified goal cell G with minimal steps.

Reward:
	•	−1 per step, +R_goal on reaching goal, with episode termination.

Difficulty:
	•	size ∈ {5, 10, 20}.
	•	partial_obs ∈ {0,1}.
	•	obstacle density proportion.

Metrics:
	•	Success rate.
	•	Steps to goal.
	•	Cycles per step.

3.1.2 Template: GRID_COLLECT

Name: GRID_COLLECT(size, n_objects)

Same gridworld but with:
	•	n_objects items scattered.
	•	Goal: collect all items with minimal steps under step budget.

Reward:
	•	+R_item per item collected, −1 per step, optional bonus for completing collection.

3.2 Causal inference and interventions

3.2.1 Template: CAUSAL_BANDIT

Name: CAUSAL_BANDIT(n_vars, k_arms)

Domain: environment + bandit

Environment:
	•	A hidden causal DAG over X_1, …, X_{n_vars}.
	•	Each arm a ∈ {1,…,k_arms} corresponds to an intervention do(A = a) on some variables.

Observation:
	•	Samples of variables after interventions.
	•	Context features derived from observations.

Reward:
	•	Real-valued reward based on some function of post-intervention distribution, e.g. expected value of a target variable.

Task:
	•	Choose interventions that maximize cumulative reward.
	•	Optionally answer queries about counterfactuals or effects of interventions.

Difficulty:
	•	n_vars ∈ {4, 8}.
	•	k_arms ∈ {4, 8}.
	•	DAG depth and noise variance.

Metrics:
	•	Regret relative to optimal intervention policy.
	•	Quality of causal effect predictions (if asked to output estimates).

3.3 Non-stationary environments

3.3.1 Wrapper template: ENV_NONSTATIONARY

Name: ENV_NONSTATIONARY(base_template, n_changes, change_type)

Domain: wrapper

Definition:
	•	Given a base template T_base (e.g. GRID_NAV, bandit, KV), construct a piecewise-stationary version where:
	•	parameters θ_j for segment j change at defined time points,
	•	the agent is not directly told the segment index.

Task:
	•	Maximize cumulative reward across segments.
	•	Adapt quickly after each change.

Parameters:
	•	n_changes ∈ {1, 2, 4}.
	•	change_type ∈ {reward_only, dynamics, both}.
	•	segment lengths.

Metrics:
	•	Regret spike and adaptation time per change.
	•	Structural changes triggered in BEM.

	4.	Tool and External Oracle Families

4.1 Simple calculator and database tools

4.1.1 Template: TOOL_CALC_ARITH

Name: TOOL_CALC_ARITH(cost_per_call, bit_width)

Domain: tool use

Environment:
	•	BEM can solve arithmetic expressions either:
	•	internally using its own experts, or
	•	by calling an external CALC_TOOL that returns the exact result at a cost.

Actions:
	•	THINK(step): internal reasoning step.
	•	TOOL_CALL(expr_id): query CALC_TOOL for specified sub-expression.
	•	ANSWER(y_hat): final answer.

Reward:
	•	Base accuracy reward: 1 if final answer correct, 0 otherwise.
	•	Cost penalty: − cost_per_call × (# of TOOL_CALL actions).
	•	Optional penalty for exceeding step budget.

Difficulty:
	•	bit_width, expression depth, step budget, tool latency.

Metrics:
	•	Trade-off curve: accuracy vs average tool calls.
	•	Internal vs external compute usage.

4.1.2 Template: TOOL_DB_LOOKUP

Name: TOOL_DB_LOOKUP(n_keys, noise_rate)

Environment:
	•	A key–value database accessible via TOOL_DB.
	•	Queries return values with optional noise or missing entries.

Task:
	•	Given a higher-level problem (e.g. small relational query), BEM must:
	•	choose which keys to query,
	•	combine results to produce final answer.

Reward:
	•	Correct final answer, minus cost for DB calls.
	•	Optional penalty for redundant or useless queries.

Difficulty:
	•	n_keys in DB, query complexity, noise_rate.

4.2 External LLM / PROVER / code-runner as untrusted tools

4.2.1 Template: TOOL_LLM_SUGGEST

Name: TOOL_LLM_SUGGEST(error_rate, coverage)

Environment:
	•	An external LLM tool that can propose candidate solutions or code snippets.
	•	The tool output is:
	•	correct with probability (1 − error_rate) on supported tasks,
	•	unsupported or incorrect on some tasks (coverage < 1).

Actions:
	•	THINK: internal reasoning.
	•	TOOL_LLM(query): get suggestion.
	•	OPTIONAL_VERIFY: use PROVER or internal checks.
	•	ANSWER(y_hat).

Reward:
	•	Final correctness.
	•	Penalties:
	•	using LLM without verification on high-risk tasks,
	•	verification cost when PROVER is used,
	•	unnecessary calls.

Difficulty:
	•	error_rate ∈ [0, 0.3].
	•	coverage ∈ [0.5, 1.0].
	•	verification cost.

4.2.2 Template: TOOL_PROVER_VERIFY

Name: TOOL_PROVER_VERIFY(cost)

Environment:
	•	External PROVER that can check logical constraints or code safety properties.

Task:
	•	Given candidate patches or solutions, decide:
	•	which ones to send to PROVER,
	•	when to trust PROVER vs internal checks.

Reward:
	•	Correct safe decisions.
	•	Time and cost penalties for verification calls.

	5.	Language and Communication Families

5.1 Instruction to policy or plan

5.1.1 Template: NL2PLAN_GRID

Name: NL2PLAN_GRID(size, instr_complexity)

Domain: language + environment

Environment:
	•	A gridworld instance as in GRID_NAV or GRID_COLLECT.

Input:
	•	A textual or structured instruction, for example:
	•	“Collect the red key before going to the blue door.”
	•	“Avoid cells marked with X.”

Task:
	•	Execute actions in the grid such that the instruction is satisfied.

Reward:
	•	Success: reaching goals specified by instruction, obeying constraints.
	•	Penalties: violating constraints, extra steps.

Difficulty:
	•	instr_complexity: number of atomic conditions and relational constraints.
	•	grid size and object count.

Metrics:
	•	Success rate as function of instruction complexity and grid size.

5.2 Description ↔ state / trajectory mapping

5.2.1 Template: TRACE_TO_SUMMARY

Name: TRACE_TO_SUMMARY(env_type, max_len)

Domain: language + environment

Input:
	•	A trajectory τ: sequence of (obs_t, action_t) from an environment instance.

Output:
	•	A short textual summary of salient events, in a constrained format (could be a tag set or simple schema).

Reward:
	•	Similarity between produced summary and ground truth summary under a defined metric:
	•	exact match in structured format, or
	•	token-level F1 for simple text.

Difficulty:
	•	max_len: maximum summary length.
	•	env_type: grid, causal, tool-use environment, etc.

5.2.2 Template: SUMMARY_TO_GOAL

Name: SUMMARY_TO_GOAL(env_type)

Input:
	•	A description of desired end state or constraints.

Output:
	•	A specification that the environment can interpret as a target goal or constraints (structured object).

Reward:
	•	1 if the decoded goal matches the underlying intended goal, 0 otherwise.

5.3 Multi-agent communication

5.3.1 Template: COMM_REFERENTIAL

Name: COMM_REFERENTIAL(bandwidth, vocab_size)

Domain: multi-agent, communication

Environment:
	•	Two roles: Sender and Receiver (both BEM instances or one BEM plus fixed heuristic).
	•	Sender observes a target object among distractors.
	•	Sender sends a message m within bandwidth bits or tokens.
	•	Receiver sees the set of objects and m, and must identify the target.

Actions:
	•	Sender: choose message m from message space of size 2^{bandwidth}.
	•	Receiver: choose object index.

Reward:
	•	1 if Receiver selects correct object, 0 otherwise.

Difficulty:
	•	bandwidth (bits per message).
	•	number of objects.
	•	variability of target features.

Metrics:
	•	Communication success rate.
	•	Emergence of compressed protocols.

	6.	Meta-Learning, Self-Modeling, and Safety Families

6.1 Meta-RL and system identification

6.1.1 Template: META_BANDIT

Name: META_BANDIT(task_count, probe_budget)

Domain: meta-learning

Environment:
	•	A distribution over bandit tasks {T_i}, each with different reward functions over arms.

Episode structure:
	1.	At episode start, a task T_i is sampled but not disclosed.
	2.	The agent has a limited probe phase of length probe_budget where exploration is cheap or rewarded differently.
	3.	After probe, the agent is primarily evaluated on exploitation performance.

Reward:
	•	Combination of probe performance and exploitation performance.

Difficulty:
	•	task_count: number of distinct tasks in the family.
	•	probe_budget.
	•	similarity between tasks.

Metrics:
	•	Regret relative to a meta-optimal policy.
	•	Sample complexity for task identification.

6.1.2 Template: META_GRID

Name: META_GRID(layout_count, probe_budget)

Analogue to META_BANDIT, but for grid layouts (different layouts, rewards, obstacle patterns).

6.2 Uncertainty estimation and ask-for-help

6.2.1 Template: SAFE_DECISION_WITH_HELP

Name: SAFE_DECISION_WITH_HELP(base_task, help_cost)

Domain: meta + safety

Wrapper:
	•	Given a base template T_base (e.g. arithmetic, logic, grid), extend the action space with:
	•	PREDICT(y_hat): commit to an answer.
	•	HELP: query an oracle for correct answer at cost help_cost.
	•	ABSTAIN (optional): refuse to answer.

Reward:
	•	Base correctness reward for PREDICT.
	•	Cost penalty for HELP.
	•	Possibly heavy penalty for confident wrong answers.
	•	Optional neutral reward for ABSTAIN depending on regime.

Difficulty:
	•	help_cost.
	•	proportion of “hard” instances.
	•	distribution shift between train and test.

Metrics:
	•	Coverage vs accuracy curve.
	•	Calibration metrics (e.g. expected calibration error).
	•	HELP usage rate.

6.3 Self-prediction and calibration

6.3.1 Template: SELF_CALIBRATION_BATCH

Name: SELF_CALIBRATION_BATCH(base_task, batch_size)

Domain: meta-self

Episode:
	1.	TEACH samples a batch of M instances from base_task.
	2.	Before solving, BEM outputs a predicted accuracy p_pred ∈ [0,1] or predicted loss.
	3.	BEM then solves the M instances normally.
	4.	The environment computes actual batch accuracy p_true.
	5.	Reward combines:
	•	task performance,
	•	calibration penalty |p_pred − p_true| or squared error.

Difficulty:
	•	batch_size.
	•	variability of instance difficulty.

Metrics:
	•	Calibration error across batches.
	•	Trade-off between prediction quality and actual task performance.

6.4 Safety-constrained optimization

6.4.1 Template: SAFE_RL

Name: SAFE_RL(env_type, constraint_level)

Domain: safety + RL

Environment:
	•	A Markov decision process with:
	•	reward signal R_t,
	•	cost signal C_t ≥ 0 representing constraint violations or risk.

Objective:
	•	Maximize expected cumulative reward subject to:
	•	E[C_t] ≤ ε or constraint on expected discounted cost.

Reward shaping:
	•	Episode reward includes penalties for violations and possibly for near-violations.

Difficulty:
	•	constraint_level ε.
	•	observability of C_t (immediate vs delayed).
	•	environment structure.

Metrics:
	•	Reward achieved vs constraint satisfaction.
	•	Number of violations.
	•	Comparison to optimal constrained policy (when known).

	7.	Minimal AGI-Oriented Base Library for TEACH

For an AGI-oriented TEACH configuration, the grammar SHOULD include at least one template family from each of the following categories:
	1.	Algorithmic/formal
	•	ARITH_EXPR
	•	LOGIC_SAT_DECIDE and/or LOGIC_IMPLICATION
	•	EXEC_TINY_VM
	2.	World/causal
	•	GRID_NAV and GRID_COLLECT
	•	CAUSAL_BANDIT
	•	ENV_NONSTATIONARY wrapper for at least one base task
	3.	Tools/oracles
	•	TOOL_CALC_ARITH
	•	TOOL_DB_LOOKUP
	•	TOOL_LLM_SUGGEST and TOOL_PROVER_VERIFY (untrusted oracle pattern)
	4.	Language/communication
	•	NL2PLAN_GRID
	•	TRACE_TO_SUMMARY and SUMMARY_TO_GOAL
	•	COMM_REFERENTIAL
	5.	Meta/safety/self
	•	META_BANDIT or META_GRID
	•	SAFE_DECISION_WITH_HELP
	•	SELF_CALIBRATION_BATCH
	•	SAFE_RL

These primitives, combined with existing composition operators (SEQ, PAR, NEST, MASK, INTERLEAVE), provide TEACH with a grammar that spans:
	•	algorithmic computation,
	•	physical/world interaction,
	•	tool-mediated reasoning,
	•	language-conditioned behavior and communication,
	•	meta-learning, calibration, and safety-constrained optimization.

This coverage is intended to bias BEM’s long-run self-training process toward broadly useful capabilities rather than narrow pattern matching.
