BEM TEACH – AGI Task Grammar v0.0.1
(Integrated English ASCII Draft, no separators)
	0.	Scope and Goals

0.1 Purpose

This document refines the TEACH component of BEM v0.0.1 by specifying an AGI-oriented task grammar that:
	1.	Treats tasks as compositional objects over:
1.1 Environment problems (external dynamics, rewards).
1.2 Program fragments (CFG and EXPERT sequences).
1.3 Hybrids that couple environment and internal behavior.
	2.	Ensures closure:
2.1 Any executable CFG fragment or EXPERT sequence can become a template.
2.2 The grammar is closed under projection, concatenation, and simple mutations.
2.3 Base problems are a bias, not the only primitives.
	3.	Supports no-regret curriculum:
3.1 Templates compete via meta-bandits.
3.2 Sub-tasks (smaller fragments) can emerge automatically under pressure from PoX and regret.

The goal is that, once TEACH is instantiated, BEM can:
	•	Start from a small set of biased base problems and trivial program fragments.
	•	Automatically discover and schedule smaller and more structured sub-tasks, down to near-ISA granularity where useful.
	•	Use the same bandit machinery to allocate training to both semantic tasks and internal program fragments.

0.2 Non-goals

This document does not:
	1.	Fix any particular environment API or simulator.
	2.	Specify concrete corpora or datasets.
	3.	Define a separate TEACH-ISA; instead, it reuses CFG and EXPERT representations.
	4.	Fix the exact meta-reward function; it specifies required ingredients and constraints.

0.3 Relationship to BEM v0.0.1

This document refines:
	•	TEACH segment contents.
	•	The notion of “task templates” and “task grammar” referenced in sections 2.1, 8.5, 10, 14, 17.5 of the integrated BEM spec.

All changes are additive and compatible with the existing step and bandit semantics.
	1.	Task Objects and Template Types

1.1 Task Template

A task template t is a symbolic object stored in TEACH with:
	•	id_t in U (class = task_template).
	•	type_t in {env, prog, hybrid}.
	•	domain_t in D_domain (math, code, reasoning, control, safety, etc).
	•	structure_t: grammar tree (see 3).
	•	bindings_t: parameter ranges and environment bindings.
	•	stats_t: online statistics for meta-bandit and curriculum.
	•	meta_bandit_t: bandit state for TEACH_BANDIT (visits, loss_sum, etc).

1.2 Task Instance

A task instance I(t, seed) is a concrete episode specification generated from template t and random seed:

I(t, seed) = (env_spec, prog_spec, reward_contract, stop_criteria)

where:
	•	env_spec describes the external environment (or NONE for pure internal tasks).
	•	prog_spec describes required or constrained internal behavior (CFG fragments, expert sets, invariants).
	•	reward_contract defines how step and episode rewards r_t, R_episode are computed.
	•	stop_criteria define maximum steps, success conditions, and failure conditions.

1.3 Template Types

Type env:
	•	structure_t describes an environment problem (e.g., bandit, gridworld, algorithmic sequence).
	•	prog_spec is unconstrained except for global safety.
	•	Reward is purely environment based.

Type prog:
	•	structure_t describes one or more CFG fragments or EXPERT sequences.
	•	env_spec is NONE (or a trivial environment).
	•	Reward is derived from internal behavior metrics (see 5).

Type hybrid:
	•	structure_t combines env and prog components.
	•	env_spec and prog_spec are both non-trivial.
	•	Reward is a function of both environment performance and internal behavior.

	2.	Representation in TEACH Segment

2.1 TEACH Segment Layout

TEACH holds:
	1.	Template table TEMPL[0..N_template_max-1], each entry:
TEMPL[i] = (id_t, type_t, domain_t, structure_ptr, bindings_ptr, stats_t, meta_bandit_t)
	2.	Grammar parameters:
	•	D_task: max grammar depth.
	•	L_frag_max: max length of prog fragments (in CFG nodes or expert steps).
	•	N_template_max: max number of live templates.
	•	p_mut: base mutation probability per template selection.
	•	p_split_frag: probability of sub-fragment extraction.
	•	p_merge_frag: probability of template merge.
	•	decay rates for stats and meta-bandits.
	3.	Curriculum statistics:
	•	per difficulty band (e.g., k = 0..K_difficulty-1):
	•	pass_rate_band[k], usage_band[k], novelty_band[k].
	•	per domain: domain usage, regret, safety measures.

2.2 Link to CFG and EXPERT

Each prog or hybrid template can reference:
	•	CFG nodes via ids in U (class = cfg_node).
	•	EXPERT slots via ids in U (class = expert).

structure_t encodes references as:
	•	node sequences: [v_0, v_1, …, v_m].
	•	expert sequences: [i_0, i_1, …, i_m].
	•	constraints: allowed transitions, lane masks, invariants.

These references are symbolic and resolved at task instantiation time.
	3.	Grammar Structure

3.1 Grammar Trees

structure_t is a rooted tree built from:
	•	Leaves:
	•	LEAF_ENV(p): environment primitive p.
	•	LEAF_PROG(f): program fragment f (CFG path or expert sequence).
	•	Internal nodes (combinators):
	•	SEQ(a, b): run a then b.
	•	PAR(a, b): run a and b in parallel lanes or independent episodes.
	•	NEST(a, depth_range): nest a to a sampled depth.
	•	MASK(a, pattern): add corruption or masking.
	•	INTERLEAVE(a, b): interleave event streams from a and b.
	•	LOOP(a, k_range): repeat a k times.
	•	BRANCH(cond, a, b): conditional subtask selection.
	•	REDUCE(a, op): summarization (e.g., aggregate rewards, counts).
	•	MAP_LANES(a, layout): apply a across lane groups.

3.2 Environment Primitives (Base Problems)

LEAF_ENV(p) refers to environment problems such as:
	•	Bit sequence tasks (parity, majority, Dyck-1, key-value retrieval).
	•	Contextual bandits and non-stationary bandits.
	•	Simple control or navigation problems.
	•	Language-model-style next-token tasks on small corpora.

These are bias choices. The grammar does not rely on them being semantically fundamental; they are simply initial templates in TEMPL.

3.3 Program Primitives

LEAF_PROG(f) refers to fragments f extracted from CFG and EXPERT:
	•	CFG p-path: v_0 -> v_1 -> … -> v_m with m <= L_frag_max.
	•	Expert sequence: [i_0, …, i_m] with m <= L_frag_max.
	•	Mixed path: (CFG nodes plus annotated expert calls).

Initially, LEAF_PROG fragments may be:
	•	trivial (single basic block, single expert), or
	•	simple paths corresponding to known routines (e.g., routing, bandit update).

Over time, TEACH can introduce new LEAF_PROG via template extraction (see 6).
	4.	Task Instance Generation

4.1 Template Selection

Given a domain d and difficulty band k requested by S_domain or the scheduler:
	1.	Collect candidate templates:
T_dk = { t | domain_t = d and difficulty_t near band k }
	2.	Use TEACH_BANDIT.CHOOSE over T_dk:
	•	meta_bandit_t holds UCB-like stats.
	•	The arm is template id_t.
	•	The meta-loss is derived from meta_reward_t (see 5).
	3.	Let t* be the selected template.

4.2 Template Mutation

With probability p_mut (configurable per band and domain):
	1.	Sample a small set of grammar edits on structure_t*:
	•	Replace a sub-node with a simpler variant (e.g., SEQ(a, b) -> a).
	•	Insert, delete, or swap children in SEQ, PAR, INTERLEAVE.
	•	Adjust depth_range, k_range within bounds.
	•	Replace LEAF_ENV(p) with a nearby environment variant.
	•	Replace LEAF_PROG(f) with a sub-fragment of f (if available).
	2.	The resulting structure defines t_new, with:
	•	id_new, stats_new initialized, meta_bandit_new initialized.
	•	domain_new and type_new inherited with minor adjustments.
	3.	With probability p_split_frag, enforce at least one LEAF_PROG to be replaced with a strictly shorter fragment, ensuring downward closure.
	4.	Insert t_new into TEMPL if capacity allows; otherwise, optionally evict an underperforming template.

4.3 Instance Materialization

Given t_selected (t* or t_new) and seed:
	1.	Traverse structure_t to create env_spec and prog_spec:
	•	For LEAF_ENV: sample environment parameters from bindings_t.
	•	For LEAF_PROG: resolve CFG / EXPERT ids and define:
	•	fragment boundaries (entry, exit).
	•	allowed transitions.
	•	constraints on expert usage.
	2.	Define reward_contract:
	•	environment reward from env_spec (if any).
	•	program reward from internal metrics:
	•	regret reduction on fragment.
	•	cost reduction on fragment.
	•	BAD rate on fragment.
	•	equivalence / safety metrics if applicable.
	3.	Define stop_criteria:
	•	max_steps (bounded by configuration).
	•	success/failure predicates on env and internal state.
	•	safety overrides.
	4.	Produce an opaque teacher_task_desc_t to be consumed by T_id and the environment manager.
	5.	Meta-Reward for Templates

5.1 Required Ingredients

For each template t, TEACH maintains:
	•	pass_rate_t: moving average of success probability on instances.
	•	regret_t: moving average of per-episode regret on tasks instanced from t.
	•	cost_t: moving average of cycles per step or per episode.
	•	info_gain_t: optional measure (e.g., coverage of new contexts, fragment diversity).
	•	safety_t: rate of BAD events associated with t.

5.2 Meta-Reward Definition

meta_reward_t is a scalar in [0, 1], defined by:

meta_reward_t = sigma( w_pass * pass_rate_t
- w_regret * norm_regret_t
- w_cost * norm_cost_t
+ w_info * norm_info_t
- w_safe * norm_bad_t )

where:
	•	sigma is a bounded squashing function (e.g., logistic), implemented in fixed-point.
	•	norm_* are normalized versions of the underlying metrics (by band and domain).
	•	w_* are configurable weights stored in TEACH or WORK.

5.3 Bandit Loss for Templates

TEACH_BANDIT uses a loss in [0,1]:

loss_template_t = 1 - meta_reward_t

and maintains:
	•	visits_t, loss_sum_t, loss_sq_sum_t, z_t

in meta_bandit_t, using the same UCB-V style index as BANDIT_CORE.
	6.	Grammar Evolution and Fragment Extraction

6.1 Template Extraction T_extract

From TRACE and CFG, TEACH constructs candidate program fragments:
	1.	Identify hot regions:
	•	high regret contributions.
	•	high BAD occurrences.
	•	high cost concentration.
	•	frequent occurrence across tasks.
	2.	For each hot path P:
	•	P = [u_0, …, u_L], where u_j are CFG nodes or expert calls.
Generate fragment candidates:
	•	whole path P if L <= L_frag_max.
	•	all contiguous subpaths P[i:j], 0 <= i < j <= L, (j-i) <= L_frag_max.
	•	optionally, prefixes P[0:j] and suffixes P[i:L].
	3.	Deduplicate fragments based on id sets and local structure hashes.
	4.	For each surviving fragment f:
	•	Build LEAF_PROG(f) and wrap in a minimal template t_f (type = prog).
	•	Initialize stats_t_f and meta_bandit_t_f.
	•	Insert into TEMPL if capacity allows, or push into a candidate pool.

6.2 Downward Closure

To guarantee that smaller program tasks can emerge:
	1.	When T_extract installs a template with fragment f of length L > 1, it SHOULD also:
	•	Install 1 or more sub-fragments of f (length < L), or
	•	Mark f as splittable for later TEACH_STEP mutation.
	2.	TEACH_STEP, under p_split_frag, MUST prefer mutations that strictly reduce fragment length or reduce the number of nodes in the grammar tree when:
	•	cost_t is high, or
	•	regret_t is high.

This enforces a bias toward discovering simpler sub-tasks around difficult or expensive regions.

6.3 Template Simplification and Pruning

Periodically:
	1.	Identify templates with:
	•	very low usage_t and low meta_reward_t, or
	•	high redundancy with other templates via structure similarity and overlapping support.
	2.	Mark for pruning:
	•	Remove from TEMPL or demote to a cold pool.
	•	Keep their historical stats in compressed form for analysis if needed.
	3.	Optionally merge similar templates:
	•	For env templates: merge parameter ranges and adjust bindings.
	•	For prog templates: construct a merged fragment with small generalization, verified via PROVER.
	4.	Relationship to ISA and BEM Core

7.1 Closure over Program Space

The TEACH grammar is required to satisfy:
	1.	Any executable CFG path or EXPERT sequence of length 1..L_frag_max is admissible as a LEAF_PROG.
	2.	Grammar combinators SEQ, PAR, INTERLEAVE, LOOP, NEST are sufficient to embed these fragments into larger templates without requiring additional ISA primitives.
	3.	No special “TEACH-ISA” is introduced; CFG and EXPERT representations are the sole program carriers.

7.2 Base Problems as Bias, Not Semantics

Base problems (LEAF_ENV):
	1.	Are chosen for practical reasons:
	•	diagnostic power and interpretability.
	•	coverage of common algorithmic and reasoning patterns.
	2.	Do not have semantic privilege:
	•	TEACH_BANDIT treats environment and program templates with the same bandit core.
	•	meta_reward_t is defined for all templates using the same functional form.
	•	Over time, TEACH can allocate more probability mass to program fragments if they yield better PoX and regret improvements.

7.3 Automatic Sub-task Emergence

Given the above constraints, the system can, in principle:
	•	Start with a small set of biased base problems plus trivial program fragments.
	•	Through T_extract and TEACH_STEP mutations, generate and refine:
	•	smaller, more focused fragments around difficult regions.
	•	hybrid templates combining external tasks with internal behavior constraints.
	•	Allocate training and verification resources based on meta-reward and PoX.

	8.	Safety and Cost Constraints

8.1 Bounded TEACH Cost

TEACH operations must obey:

Cost_TEACH_step <= C_teach_max

independent of episode length, achieved by:
	•	Bounding D_task, L_frag_max, template count, and mutation operations per TEACH_STEP call.
	•	Using approximate bandit indices and small constant-time approximations for meta_reward and normalization.

8.2 Safety Coupling

Templates must respect global safety constraints:
	•	reward_contract must penalize or nullify templates that systematically increase BAD.
	•	meta_reward_t must include a safety term norm_bad_t with non-zero weight w_safe.
	•	The contextual scheduler (Algorithm D) may reduce alpha_synth or alpha_act for domains or bands with elevated safety_t.

	9.	Informative Notes

9.1 AGI Orientation

The grammar is designed so that:
	•	The space of tasks spans both “what the agent does in the world” (env) and “how the agent internally computes” (prog).
	•	Over long training runs, the teacher can:
	•	discover reusable algorithmic fragments,
	•	tighten them via verification and superoptimization,
	•	and redeploy them as primitives in more complex tasks.

9.2 Practical Initialization

A practical v0.0.1 deployment might:
	1.	Seed TEMPL with:
	•	a small set of env base problems (parity, majority, Dyck-1, key-value, bandit).
	•	a small set of trivial prog fragments (routing, bandit update, a few known CFG blocks).
	2.	Enable T_extract and ensure:
	•	hot regions are logged with enough detail.
	•	sub-fragment extraction is implemented.
	•	meta_reward includes PoX and fragment-level metrics.
	3.	Let TEACH_BANDIT and scheduler gradually shift mass from hand-picked tasks to discovered fragments where they improve PoX.

End of BEM TEACH – AGI Task Grammar v0.0.1.
