BEM TEACH – AGI Task Grammar v0.0.1
(Integrated English ASCII Draft, no separators)

0.	Scope and Goals

0.1 Purpose

This document defines the TEACH subsystem and AGI task grammar v0.0.1 for BEM v0.0.1. It specifies:
1.	The internal representation of tasks and task templates.
2.	A grammar over tasks that is closed under composition and grounded in BEM’s ISA and environment simulators.
3.	A family of BaseProblem classes that serve as an initial bias but are not a hard floor for task granularity.
4.	Curriculum and difficulty mechanics implemented via contextual bandits.
5.	The interface between TEACH and the BEM fast path, structural update subsystem, and PoX scheduler.
6.	Conditions under which TEACH can bootstrap RL “from nothing” and refine tasks down to ISA-level programs.

The goal is to make TEACH a minimal but sufficient mechanism for automatic task generation and curriculum, such that:
1.	Any finite BEM-programmed environment with bounded complexity can be represented as a task or a composition of tasks.
2.	BaseProblem classes are information-theoretically meaningful and computationally primitive, but not semantically arbitrary.
3.	The teacher can, in principle, generate tasks that probe all capabilities BEM can express under its ISA and structural update model.

0.2 Non-goals

This document does not:
1.	Fix specific datasets, simulators, or domains.
2.	Define external APIs for human annotation or evaluation.
3.	Constrain environment implementation details beyond the abstract interfaces described here.
4.	Define any floating-point behavior inside TEACH; all TEACH control logic must be implementable in fixed-point and integer arithmetic.

0.3 Relation to BEM Core

This grammar is layered on top of the existing BEM spec:
1.	TEACH is encoded in the TEACH memory segment.
2.	TEACH interacts with the fast path only via:
2.1	task_id_t and teacher_task_desc_t fields in STATE/SHARED.
2.2	TEACH_CALL co-processor interface.
2.3	Environment selection and configuration.
3.	Structural updates (patches) may modify TEACH configuration via the same verification and PoX constraints as other structural changes.

1.	Task Object Model

1.1 Task instance

A task instance tau is defined as a tuple:

tau = (id_tau, env_id, template_id, params_id, difficulty_band, horizon, reward_spec)

where:
1.	id_tau in U, class = task_instance.
2.	env_id in U, class = env: identifier of environment simulator.
3.	template_id in U, class = task_template or bottom if direct env.
4.	params_id in U, class = task_params: pointer to parameter block.
5.	difficulty_band in Z_small: integer bin for curriculum (e.g. 0..7).
6.	horizon in Z_32: maximum episode length in steps.
7.	reward_spec: identifier of reward mapping and scaling parameters.

The environment simulator env_id is responsible for:
1.	Producing observations that BEM sees as state bits or tokens.
2.	Updating internal environment state.
3.	Producing rewards r_t in [R_min, R_max].
4.	Signaling episode termination.

1.2 Task template

A task template t is an abstract recipe for constructing task instances. It is defined as:

t = (id_t, domain, op_tree, base_refs, param_schema, d_est, stats)

where:
1.	id_t in U, class = task_template.
2.	domain in Z_small: integer label for domain (math, code, game, safety, control, core, etc).
3.	op_tree: composition tree built using grammar operators (Section 3).
4.	base_refs: references to BaseProblem families or env_ids used as leaves.
5.	param_schema: schema describing how parameters are sampled (ranges, distributions).
6.	d_est in R_fixed: estimated difficulty.
7.	stats: TEACH bandit statistics for this template:
7.1	usage_count_t (integer).
7.2	pass_rate_t (fixed-point).
7.3	regret_proxy_t (fixed-point).
7.4	novelty_t (fixed-point).
7.5	last_update_step_t (integer).

Task instances are sampled from templates by instantiating param_schema for parameter values and then compiling the op_tree into a specific env configuration.

1.3 Task family and difficulty band

TEACH groups templates into:
1.	Family f: determined by domain and structural type of op_tree.
2.	Difficulty band k: coarse integer band so that typical success probability for a competent agent remains in a target range (e.g. 0.2–0.8).

Given a requested (domain d, difficulty band k), TEACH selects an appropriate template t and instantiates it into tau via TEACH_STEP.

2.	Primitive Substrate

The task grammar is defined over two primitive layers:
1.	ISA-level program fragments and control (Level -1).
2.	BaseProblem families (Level 0).

2.1 ISA-level fragment tasks (Level -1)

Level -1 tasks are defined purely in terms of BEM’s own CFG and ISA, without reference to external semantic domains.

A Level -1 task template t_isa is:

t_isa = (id_t, domain = core, op_tree = CFG_FRAGMENT(cfg_id), base_refs = { }, param_schema, d_est, stats)

Here:
1.	CFG_FRAGMENT(cfg_id) denotes a specification that:
1.1	Selects a CFG subgraph inside C[pc].
1.2	Defines initial conditions over s_0 and M_0.
1.3	Defines a stopping condition (max steps or specific state predicate).
1.4	Defines reward as a function of local predicates on STATE / SHARED.
2.	Example ISA-level objectives:
2.1	Minimize cycles_per_step subject to satisfying a local invariant.
2.2	Maximize throughput of BEM_EXPERT_BATCH on synthetic circuits.
2.3	Hit a specific bit pattern in STATE within H steps.

These tasks exist even “from nothing” because they only require:
1.	Pre-existing CFG and ISA, which are static definitions.
2.	A trivial pseudo-environment that simply runs BEM’s own core on synthetic inputs.

Level -1 tasks ensure that TEACH can always construct some RL problems over BEM’s own runtime behavior, even with no external BaseProblem families defined.

2.2 BaseProblem families (Level 0)

BaseProblem families are environment templates that are:
1.	Information-theoretically simple.
2.	Computation-theoretically primitive in the sense that they capture basic combinatorial and decision structures.
3.	Expressible as finite BEM environments with bounded complexity.
4.	Not tied to any specific human semantic domain beyond basic CS/math primitives.

Each BaseProblem family B_k is:

B_k = (id_B, env_class, param_schema_B, canonical_operators, difficulty_map)

A conforming TEACH v0.0.1 implementation MUST provide at least the following initial BaseProblem families:

1.	Bitwise aggregation:
1.1	parity over {0,1}^T.
1.2	majority / threshold over {0,1}^T.
1.3	popcount in a bounded range.

2.	Stack and context-free structure:
2.1	Dyck-1 (balanced parentheses).
2.2	Small stack machine with push/pop/peek and bounded depth.

3.	Key-value and memory:
3.1	key-value retrieval with K pairs and a query.
3.2	limited associative memory retrieval with distractors.

4.	Bandits:
4.1	K-armed Bernoulli bandit.
4.2	contextual bandit with binary context.
4.3	non-stationary bandit with change-points.

5.	Small games (game-theoretic primitives):
5.1	2×2 matrix games with fixed opponent policy (e.g. prisoner’s dilemma, coordination, matching pennies).
5.2	simple signaling games with two-stage communication.
5.3	single-item second-price auctions with a fixed opponent strategy.

6.	Simple SAT / safety:
6.1	bounded SAT instances.
6.2	bounded index safety (stay in-bounds; avoid BAD flag).

7.	Sequential / algorithmic primitives:
7.1	bounded integer addition and subtraction over small widths (e.g. 8-bit or 16-bit).
7.2	comparison and min/max over short lists of bounded length.
7.3	small sorting-network evaluation tasks on bounded arrays.
7.4	bounded-distance (Hamming or edit distance) computation over short bit-strings.

8.	Token-sequence modeling:
8.1	finite-vocabulary next-token prediction over bounded-length sequences.
8.2	finite-state transducer emulation over token streams.
8.3	bounded copying / reversal / simple permutation of token sequences.

For each family 1–8:
1.	The BaseProblem defines a generator of environment configurations env_id.
2.	Each env_id corresponds to a finite simulator whose internal dynamics are implementable under BEM’s ISA with bounded cost and bounded horizon.
3.	param_schema_B specifies how concrete instance parameters (e.g. T, K, payoff matrices, SAT size) are sampled within configured ranges.
4.	canonical_operators defines a minimal interface to build composition via the grammar in Section 3.
5.	difficulty_map defines how instance parameters map into nominal difficulty regions.

The system MAY introduce additional BaseProblem families via structural update and CEGIS. However, any TEACH v0.0.1 implementation that omits any of the families 1–8 above is non-conforming.

2.3 Bias vs closure

BaseProblems are a bias because:
1.	They are pre-chosen to cover classic CS and game-theoretic primitives that are known to span many downstream tasks.
2.	They determine the early curriculum and initial shaping of experts and routing.

They are not a hard floor because:
1.	TEACH is allowed to define direct Level -1 tasks that skip BaseProblem families entirely.
2.	TEACH can mutate and compose templates to generate tasks that no longer resemble the original BaseProblems.
3.	Structural update and CEGIS can synthesize new environment simulators and inject them as new BaseProblem families or derived templates.

3.	Grammar Syntax and Semantics

3.1 Operators

The TEACH task grammar defines operators over task templates:
1.	LEAF(B_k): a BaseProblem family leaf.
2.	LEAF_ISA(cfg_id): an ISA-level fragment leaf (Level -1).
3.	SEQ(a, b): sequential composition (b after a).
4.	PAR(a, b): parallel composition of sub-environments.
5.	NEST(a, depth_k): nested application of a to depth <= depth_k.
6.	MASK(a, mask_pattern): apply masking / corruption of observations or rewards.
7.	INTERLEAVE(a, b): interleaved stream of tokens/events from a and b.
8.	REPEAT(a, n_min, n_max): random repetition of a.
9.	BRANCH(cond, a, b): choose between a and b based on a simple predicate on initial parameters or environment state.
10.	LOOP(a, stop_cond): apply a in a loop until stop_cond or horizon.

Each operator must:
1.	Compile to a finite environment graph with bounded horizon.
2.	Preserve bounded per-step cost.
3.	Be representable as CFG plus environment descriptors in TEACH/SHARED.

3.2 Structural representation

The op_tree of a template t is encoded as:
1.	A DAG or tree of nodes, each node:
node = (op_type, children_ids, local_params)
2.	op_type in {LEAF, LEAF_ISA, SEQ, PAR, NEST, MASK, INTERLEAVE, REPEAT, BRANCH, LOOP}.
3.	children_ids: indices of child nodes.
4.	local_params: small fixed parameter vector (e.g. depth_k, mask_pattern, n_min, n_max, stop_cond encoding).

TEACH maintains for each template:
1.	op_tree in TEACH segment.
2.	A compiled environment skeleton env_skel(t) in SHARED/CFG/STATE that actual env instances refine at runtime.

3.3 Semantics of key operators

3.3.1 SEQ(a, b)

Given two task templates a, b, SEQ(a, b) is a template whose instantiated environment runs:
1.	Run env_a for an episode (bounded horizon H_a).
2.	At termination, transform a state summary (s_a_final, r_a, auxiliary stats) into initial conditions for env_b using a deterministic mapping.
3.	Run env_b for another episode (bounded horizon H_b).
4.	Overall reward is a weighted sum or concatenation of rewards from a and b as defined by reward_spec.

3.3.2 PAR(a, b)

PAR(a, b) is a template whose instantiated environment:
1.	Maintains internal states for env_a and env_b.
2.	On each step, advances both sub-envs (or alternates them according to a fixed or sampled schedule), and merges their observations into a combined observation:
obs_t = concat(obs_t^a, obs_t^b).
3.	Reward is a function of (r_t^a, r_t^b) and optional interaction terms (e.g. weighted sum, max, or structured payoff).

3.3.3 NEST(a, depth_k)

NEST(a, depth_k) samples a depth d in [1, depth_k] and composes a recursively (e.g. as nested brackets, nested subepisodes, or nested levels of an environment). For example:
1.	Dyck-like tasks with nested parentheses where each use of a corresponds to one layer.
2.	Function call stacks where a is “call body” and nesting corresponds to call depth, with bounded depth_k.

3.3.4 INTERLEAVE(a, b)

INTERLEAVE(a, b) interleaves two token/source streams with a predefined or sampled pattern. Typical use cases:
1.	Context plus query sequences (context from a, query from b).
2.	Two conversations / channels mixing.
3.	Noisy distractor streams, where one stream carries signal and the other carries distractors.

3.3.5 MASK(a, mask_pattern)

MASK(a, mask_pattern) modifies either:
1.	Observations: hide bits, add noise, drop tokens, or coarsen information.
2.	Rewards: partially mask intermediate rewards or delay reward visibility.

This allows building POMDP variants from simpler fully observable MDP tasks.

3.4 Closure property

The grammar is designed such that:
1.	Any finite environment that can be encoded as a BEM-controlled simulator with bounded horizon and bounded state can be represented as some template t:
t = grammar_expression(LEAF/LEAF_ISA, operators)
2.	TEACH is allowed to introduce new LEAF or LEAF_ISA nodes via CEGIS and structural patches, so the grammar extends over time while remaining within bounded-complexity constraints.

4.	Difficulty, Curriculum, and Meta-Reward

4.1 Difficulty estimate d_est

Each template t has an estimated difficulty d_est, defined as a function of:
1.	pass_rate_t (moving average of success probability).
2.	expert usage statistics and regret on that template.
3.	horizon and complexity metrics (size of op_tree, internal simulator size).

A simple form is:

d_est(t) = a0 + a1 * (1 - pass_rate_t) + a2 * log(1 + horizon_t) + a3 * complexity(t)

where:
1.	a0..a3 are configuration constants stored in TEACH.
2.	complexity(t) is a bounded function of op_tree size, BaseProblem parameters, and composition structure.

4.2 Difficulty bands

Difficulty bands are coarse bins over d_est, e.g.:

band 0: d_est < d0
band 1: d0 <= d_est < d1
…
band K-1: d_{K-2} <= d_est

TEACH ensures that within a domain:
1.	Some fraction of tasks presented to BEM fall into “learning-friendly” bands (e.g. 0.2 <= pass_rate <= 0.8).
2.	Harder bands are sampled as competence improves (as measured by pass_rate and regret trends).

4.3 Meta-reward for templates

For template t, TEACH tracks a meta-reward:

meta_reward_t = f(pass_rate_t, regret_proxy_t, novelty_t)

Design choice example:
1.	Encourage templates where:
1.1	pass_rate is neither 0 nor 1 (learning signal).
1.2	regret over episodes on that template is decreasing.
1.3	novelty_t indicates new regions of behavior or state space.
2.	Penalize templates that are:
2.1	Trivialized (pass_rate ~ 1 with zero regret and low novelty).
2.2	Degenerate (pass_rate ~ 0 with no improvement).
2.3	Overlapping strongly with many other templates (low novelty).

5.	Teacher Bandit and Grammar Evolution

5.1 TEACH bandit state

For each template t and difficulty band k (or domain d), TEACH keeps:
1.	usage_count_t,k.
2.	meta_loss_sum_t,k.
3.	meta_loss_sq_sum_t,k.
4.	z_t,k: log-weight analogous to z_i,tau for experts.
5.	N_t,k: total selections within that context.

Meta-loss is defined as an affine transform of meta_reward:

meta_loss_t = (R_meta_max - meta_reward_t) / (R_meta_max - R_meta_min)

5.2 TEACH_BANDIT.CHOOSE

Given domain d, band k, and context x_teach (curriculum state), TEACH_BANDIT.CHOOSE selects a template t* from candidate set T_d,k:
1.	Compute index_t using the same UCB-V style index as BANDIT_CORE, over meta_loss stats.
2.	Optionally introduce a prior term from z_t,k and contextual features x_teach.
3.	Select t* greedily or stochastically as with experts.

5.3 Mutation and expansion

When template t* is chosen, TEACH may with probability p_mut generate a new template t_new:
1.	Sample a small local transformation of the op_tree of t*:
1.1	Replace a leaf with SEQ(leaf, leaf2) using another BaseProblem or template leaf.
1.2	Insert MASK or INTERLEAVE with another BaseProblem or template.
1.3	Increase nesting depth within configured limits.
1.4	Swap band-specific parameters within safe ranges.
2.	Ensure op_tree size and horizon remain under configured bounds.
3.	Assign d_est(t_new) from initial heuristics plus a small random perturbation.
4.	Initialize bandit stats for t_new with priors derived from t*.

TEACH thus maintains an evolving set of templates, growing the task space while respecting computational constraints.

5.4 Injection of new LEAF / LEAF_ISA nodes

Structural update and CEGIS can propose:
1.	New BaseProblem-like environment families (NewB) with formal specs.
2.	New ISA fragments (NEW_CFG_FRAGMENT) that expose particular internal structures of BEM.

Once verified and accepted via PoX, TEACH can add:
1.	LEAF(NewB) nodes as new primitive leaves.
2.	LEAF_ISA(new_cfg_id) nodes pointing to new CFG fragments.

This ensures long-run closure: TEACH is not restricted to the initial BaseProblem set, but any conforming implementation must at all times provide the initial families 1–8.

6.	Interaction with BEM Core

6.1 TEACH_CALL interface

TEACH_CALL(env) is a co-processor call with:

Input:
1.	domain d.
2.	difficulty band k.
3.	curriculum context x_teach (opaque fixed-length vector derived from WORK and TEACH).
4.	external hints (e.g. human-specified domain or safety mode), if present.

Output:
1.	task_instance_id tau_id.
2.	task_id_t: small integer representing task family.
3.	teacher_task_desc_t: pointer for T_id in BEM.
4.	env_config pointer (env_id plus parameter block).

6.2 Fast path usage

STEP_FAST uses:
1.	task_id_t: to select bandit stats per task family.
2.	context_hash_t: to build routing queries.
3.	env_config: indirectly via environment simulator integrated in CFG or external driver.

BANDIT_UPDATE_STEP uses:
1.	r_t based on reward_spec for tau.
2.	task_id_t and last_choice to update expert bandits.

6.3 Mid-path usage

Mid-path routines:
1.	Aggregate per-task and per-template statistics from TRACE.
2.	Update pass_rate_t, regret_proxy_t, novelty_t.
3.	Run GRPO_LITE-like updates on z_i,tau and z_t,k.
4.	Update d_est and difficulty bands.

7.	Game-Theoretic Coverage

7.1 Single-agent regret

Because BaseProblem includes bandit, contextual bandit, non-stationary bandit, and because TEACH can assemble MDP-like environments via grammar operators, BEM plus TEACH can:
1.	Approximate any finite-horizon MDP as an environment under the bounded-complexity constraints.
2.	Apply no-regret bandit logic over expert choices per task.
3.	Learn policies with sublinear regret relative to the best fixed expert policy class within each task family.

7.2 Multi-agent primitives

BaseProblem’s game families (matrix games, signaling, auctions) ensure:
1.	Simple two-player games can be realized as BaseProblem leaves with env_id encoding opponent strategy.
2.	TEACH can compose them via SEQ, INTERLEAVE, MASK to create games with delayed feedback, imperfect information, or composite payoffs.

In the limit, TEACH can represent:
1.	Repeated games with fixed opponent policy classes.
2.	Contextual games where context encodes opponent type.
3.	Meta-games where BEM is optimizing over its own patch and template selection.

7.3 No-regret meta-learning

Because TEACH uses the same bandit machinery as BEM’s expert selection:
1.	Template selection itself is subject to no-regret guarantees over meta-reward, under the same stochastic assumptions.
2.	Scheduler META_STEP chooses between act / synth / verify under a contextual bandit scheme.
3.	As long as environments and templates respect bounded cost, regret bounds for these meta-bandits hold.

8.	Safety and Verification Coupling

8.1 Safety-aware tasks

TEACH must respect safety constraints:
1.	Certain templates are tagged as safety-critical.
2.	BAD_t indicators and safety violations are recorded in TRACE.
3.	Templates that systematically correlate with safety violations are down-weighted or suppressed by TEACH_BANDIT and by structural update policy.

8.2 VC and PoX coupling

Structural patches that modify TEACH:
1.	Must define VC(Delta_TEACH) capturing:
1.1	Preservation of safety tags and invariants.
1.2	Bounds on horizon, cost, and op_tree size.
1.3	Consistency of difficulty bands and domain labels.
2.	Are accepted only if:
2.1	VC(Delta_TEACH) is UNSAT (no violation).
2.2	PoX score(Delta_TEACH) >= D threshold.

TEACH therefore evolves under the same formal gatekeeping as experts and CFG.

9.	Cold Start and “RL from Nothing”

9.1 Minimal initial configuration

Even with no BaseProblem families beyond the required set defined in 2.2, TEACH can run by:
1.	Defining a minimal Level -1 template set:
1.1	t_cycle: minimize cycles per step on a simple CFG fragment subject to a trivial invariant.
1.2	t_invariant: maintain a simple invariant over STATE bits.
1.3	t_random_reward_probe: probe random reward functions over internal behavior to explore BEM’s dynamics.
2.	Using TEACH_BANDIT over these Level -1 templates.
3.	Allowing structural updates to:
3.1	Synthesize simple environment simulators derived from internal behavior.
3.2	Introduce them as new BaseProblem families or derived templates.

9.2 With BaseProblems present

With the BaseProblem families in 2.2 available at initialization, TEACH gains:
1.	A richer set of well-understood RL tasks that provide dense learning signal over combinatorial, memory, algorithmic, game, and token-sequence patterns.
2.	Immediate exposure to classic CS and game-theoretic patterns (parity, stack, KV, bandits, simple games, SAT, basic sequential algorithms, simple sequence modeling).
3.	A strong inductive bias that improves sample efficiency and stability of early RL.

However, TEACH is not restricted to these; it may:
1.	Decompose composed tasks into effective Level -1 fragments via structural logging and template extraction.
2.	Use CEGIS and synthesis to generate new descendants of BaseProblems and entirely new primitive leaves.

9.3 Emergent sub-task discovery

Because:
1.	TRACE logs episode-level and step-level information including high regret zones.
2.	Structural updates can extract macros and CFG fragments (T_extract).
3.	TEACH can build templates from CFG fragments via LEAF_ISA.

The system can effectively discover and schedule sub-tasks that are “smaller” than the initial BaseProblems, corresponding to:
1.	Frequently reused control fragments.
2.	Local transformations over bits and memory.
3.	Gate conditions and routing fragments that are individually learnable and reusable.

Thus the grammar supports:
1.	Top-down composition from BaseProblems via SEQ / PAR / NEST / INTERLEAVE / MASK / LOOP / BRANCH / REPEAT.
2.	Bottom-up discovery of smaller fragments via template extraction and ISA-level leaves.

End of BEM TEACH – AGI Task Grammar v0.0.1 revised high-quality draft.
