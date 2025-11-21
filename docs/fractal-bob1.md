Title: Fractal Bob v0.0.1 – Industrial Online Control Core Specification
Status: Draft
Language: ASCII, English only
	0.	Conventions and Scope

0.1 Mathematical conventions

R        : real numbers
N        : non-negative integers {0,1,2,…}
N+       : positive integers {1,2,…}
{0,1}^*  : finite bitstrings
{0,1}^N  : infinite bitstreams

For a random variable X, E[X] denotes expectation.
For a function f and input x, f(x) is the value of f at x.

0.2 System overview

Fractal Bob v0.0.1 (the Core) is an online decision and learning system defined by:

(a) A single shared parameter store w: K -> R with O(1) access.
(b) A structured key space K with hierarchical prefix metadata.
(c) A linear prediction core used by all tasks.
(d) A Monte Carlo tree search module.
(e) A hierarchical representation module.
(f) A Bayesian reconstruction module for partially observed inputs.
(g) A task abstraction for base and meta control.

The specification constrains structure, interfaces, and complexity. It does not fix particular external encoders, simulators, or environment details.

0.3 Non-goals

The Core does not attempt:
	•	Exact logical omniscience or full decidability of any expressive theory.
	•	Closed-form optimality guarantees in arbitrary environments.
	•	A fixed choice of external models or feature encoders.

	1.	Environment and Interaction Model

1.1 Time and interaction

Time steps are t in N. At each step t:
	•	s_t in S        : environment state or observation.
	•	a_t in A        : action chosen by the Core.
	•	r_t in R        : reward returned by the environment.
	•	s_{t+1} in S    : next state.

The environment defines a (possibly unknown) kernel:

P_env(s_{t+1}, r_t | s_t, a_t)

The Core observes (s_t, r_t, optional aux_t) and must choose a_t.

1.2 Event and history

Define:
	•	e_t = (s_t, a_t, r_t, s_{t+1}, aux_t)
	•	H_t = (e_0, …, e_{t-1})

aux_t may contain external solver outputs, diagnostics, or other signals.

The Core is online: its internal state at time t is determined by (H_t, w_t, fixed seeds, configuration).

1.3 Objectives

The Core is used to:
	•	Maximize expected discounted return
J = E[ sum_{t=0..∞} gamma^t r_t ] for some gamma in (0,1].
	•	Minimize a weighted sum of auxiliary task losses
L_total(w) = sum_{T in Tasks} lambda_T E[ ell_T(w; context_T) ]

No single optimization algorithm is mandated; the structure is designed so that gradient-based incremental methods can be applied with bounded per-step cost.
	2.	Key Space and Parameter Store

2.1 Structured key space

The parameter key space K is a Cartesian product:

K = Hyp x Level x Role x PrefixMeta x Slot

where:
	•	Hyp         : hypothesis identifier (finite or countable set).
	•	Level in N  : control level (0 = base, higher = meta).
	•	Role        : semantic role of the parameter.
	•	PrefixMeta  : hierarchical prefix metadata.
	•	Slot in N   : local index within a role and prefix.

2.1.1 Hypothesis

Hyp is an identifier for a logical subsystem. Examples (not mandatory):

Hyp_ENV     : base environment control.
Hyp_MCTS    : Monte Carlo tree search statistics and policy.
Hyp_REP     : representation basis and gates.
Hyp_BR      : Bayesian reconstruction templates.
Hyp_META_i  : meta tasks (scheduler, loss mixing, etc).

Hyp is represented as an integer or enumeration.

2.1.2 Level

Level in N indicates control depth:

0 : base control (direct environment actions).
1 : meta control (scheduling, loss mixing, representation management).
n : further meta layers (optional).

2.1.3 Role

Role is an enumeration distinguishing parameter semantics. Recommended roles:

ROLE_FEAT    : generic linear-feature weights.
ROLE_Q       : cumulative value (for MCTS).
ROLE_N       : visit count (for MCTS, treated as real).
ROLE_P       : policy parameters (log-linear).
ROLE_V       : state value parameters.
ROLE_REP_B   : representation basis vector components.
ROLE_REP_G   : representation gating parameters.
ROLE_BR_HYP  : Bayesian reconstruction template parameters.
ROLE_META    : generic meta control parameters.
ROLE_MISC    : reserved.

The same Role can be used in multiple Hyp and Level combinations.

2.1.4 Prefix metadata

For any object x (state, history segment, template index, etc), define:
	•	BASE_CODE(x) in {0,1}^*: a finite bitstring.
	•	seed_code: a global secret seed (constant across run).

Define an infinite pseudo-random bitstream:

C_infty(x) in {0,1}^N

via a deterministic stream PRF:

C_infty(x) = STREAM_PRF(seed_code, BASE_CODE(x))

For any depth L in N, define the prefix:

p_L(x) = C_infty(x)[0:L]  (first L bits)

PrefixMeta is defined as:

PrefixMeta = (L, p_L)

Only a finite subset of prefixes are active at any given time; parameters are allocated only for active prefixes.

2.1.5 Slot

Slot is an integer in N with constraint:

0 <= Slot < K_max(Hyp, Level, Role, L)

for some configuration function K_max specifying the maximum number of distinct parameters per (Hyp, Level, Role, prefix depth L). K_max is specified at deployment time.

2.2 Abstract key tuple and concrete ID

The semantic key is the tuple:

k = (Hyp, Level, Role, PrefixMeta, Slot) in K

For implementation, a concrete ID in a finite integer space is derived via:

RAW_KEY_BITS = PRF(seed_key, encode(Hyp, Level, Role, PrefixMeta, Slot, extra_context))
ID = ENCODE_BITS_TO_INT(RAW_KEY_BITS)

where:
	•	seed_key is a separate global secret seed.
	•	extra_context is optional, fixed per role design.
	•	ENCODE_BITS_TO_INT is a stable mapping from bitstrings to integer identifiers.

The parameter store uses ID as its internal key. The semantic key k is used in the spec.

2.3 Global parameter store interface

The parameter store maintains a mapping w: K -> R, implemented via the internal mapping from ID to a real value. The required interface is:

READ(w, k) -> float
ADD(w, k, delta: float)

Semantics:
	•	READ(w, k)
If k has an assigned value, return that value.
Otherwise, return 0.0 (an uninitialized parameter is treated as zero).
	•	ADD(w, k, delta)
Conceptually: w(k) <- w(k) + delta.
If k was not present, it is initialized as 0.0 before the addition.

The parameter store MAY provide additional functions:

SNAPSHOT(w) -> snapshot_blob
RESTORE(snapshot_blob) -> new_state

but these are orthogonal to the learning logic.

2.4 Lossless Hashed Linear implementation constraints

An implementation of the parameter store is regarded as compliant if the following hold.

2.4.1 Semantic constraints
	•	For every k in K, there is at most one stored value v = w(k).
	•	No two distinct keys k1 != k2 share the same stored location; there must be no collisions for live keys.
	•	READ and ADD on different keys behave as if they operate on a logical sparse map from K to R.

2.4.2 Complexity constraints

For all sequences of key operations of length T (READs and ADDs):
	•	The worst-case time for each READ and ADD is bounded by a constant C_store independent of:
	•	T (the number of operations so far),
	•	U_T (the number of distinct keys used so far),
	•	and N_T (the number of distinct states or tasks seen so far).

Memory usage MUST be O(U_T) for some constant factor, where U_T is the count of keys k with w(k) != 0.

A minimal perfect hash plus bounded delta dictionary is a suitable design but not mandated.
	3.	Linear Prediction Core

3.1 Feature collections

For any task T, define a feature function:

FEATURES_T(ctx) -> list [(k_1, v_1), …, (k_m, v_m)]

where:
	•	ctx: a task-specific context object.
	•	k_i in K: parameter keys.
	•	v_i in R: feature values.
	•	m is bounded by a task-specific constant m_T independent of time.

3.2 Linear score

Given w and features [(k_i, v_i)]:

score_T(ctx; w) = sum_{i=1..m} READ(w, k_i) * v_i

3.3 Probabilistic outputs

Binary case:
	•	p(y=1 | ctx, w) = sigma(score_T(ctx; w))
where sigma(z) = 1 / (1 + exp(-z))

Multiclass case with candidate actions C(ctx):
	•	For each a in C(ctx), compute features_T(ctx, a) and score_T(ctx, a; w).
	•	p(a | ctx, w) = exp(score_T(ctx,a; w)) / sum_{b in C(ctx)} exp(score_T(ctx,b; w))

Other distributions (Gaussian, etc.) MAY be used, but are not specified.

3.4 Loss functions and gradients

Each task T defines a loss function:

ell_T(w; ctx, y)

where y is the target or label.

Examples:

Binary logistic cross-entropy:
	•	score = score_T(ctx; w)
	•	p = sigma(score)
	•	ell_T = -[ y * log(p + eps) + (1-y) * log(1-p + eps) ]

Gradients:

For the binary logistic example, define error term:
	•	e = p - y

Then for each (k_i, v_i):
	•	grad(k_i) = e * v_i

The Core update for a learning rate eta_T and L2 coefficient lambda_L2_T is:
	•	For each k_i:
	•	delta = -eta_T * grad(k_i) - eta_T * lambda_L2_T * READ(w, k_i)
	•	ADD(w, k_i, delta)

The specification requires that:
	•	For any task T, the number of ADD calls per update is bounded by a constant proportional to m_T, independent of time.

3.5 Task definition

A task T is defined by:
	•	Hyp_T     : Hypothesis identifier.
	•	Level_T   : control level.
	•	Role set  : set of Roles this task uses.
	•	FEATURES_T(ctx) : feature function.
	•	CANDIDATES_T(ctx) : optional candidate set (e.g., actions).
	•	LOSS_T(w; ctx, y) and its gradients.
	•	UPDATE_SCHEDULE_T : when and how often updates are applied.

All tasks share the same global parameter store w and use the same READ/ADD operations.
	4.	Monte Carlo Tree Search Module

4.1 Search state and actions

Define a search state set S_search and search action set A_search.

The module uses a function:

SEARCH_STATE(s_t, H_t, z_hat_t) -> s_root

to map current environment state s_t, history H_t, and optional reconstructed context z_hat_t to a search root state s_root in S_search.

There is a mapping:

MAP_SEARCH_ACTION_TO_ENV(a_search) -> a_env in A

which maps chosen search actions to environment actions.

4.2 Node identification and prefixing

A search node is characterized by (s_search, depth d).

For such (s, d), and an action a in A_search, define:
	•	PrefixMeta_MCTS(s) = (L_MCTS, p_{L_MCTS}(s)) for some fixed L_MCTS.

Now define the keys:
	•	k_N(s,a,d) = (Hyp_MCTS, Level_MCTS, ROLE_N, PrefixMeta_MCTS(s), Slot_N(a,d))
	•	k_W(s,a,d) = (Hyp_MCTS, Level_MCTS, ROLE_Q, PrefixMeta_MCTS(s), Slot_W(a,d))

where Slot_N and Slot_W are deterministic, bounded mappings from (a, d) to small integers.

4.3 Reading statistics

Define:
	•	N(s,a,d) = max(0, READ(w, k_N(s,a,d)))
	•	W(s,a,d) = READ(w, k_W(s,a,d))
	•	Q(s,a,d) = W(s,a,d) / max(1, N(s,a,d))

4.4 Policy prior for search

For the prior search policy, for each state s and action a, define:

PrefixMeta_P(s) = (L_P, p_{L_P}(s))

For indices j in {0, …, K_P-1}:
	•	k_P(s,a,j) = (Hyp_MCTS, Level_MCTS, ROLE_P, PrefixMeta_P(s), Slot_P(a,j))
	•	psi_j(s,a) : feature value, derived from PRF and possibly from representation h(s).

Then:
	•	score_P(s,a) = sum_j READ(w, k_P(s,a,j)) * psi_j(s,a)
	•	P_prior(a | s) = exp(score_P(s,a)) / sum_b exp(score_P(s,b))

4.5 Value estimate

Define a value model:

PrefixMeta_V(s) = (L_V, p_{L_V}(s))

For indices i in {0, …, K_V-1}:
	•	k_V(s,i) = (Hyp_MCTS, Level_MCTS, ROLE_V, PrefixMeta_V(s), Slot_V(i))
	•	phi_i(s) : feature value based on h(s) and PRF.

Then:
	•	V_hat(s) = sum_i READ(w, k_V(s,i)) * phi_i(s)

4.6 Selection rule

At node (s, d), with candidate actions A_search(s):
	•	total_N = sum_{b in A_search(s)} N(s,b,d)
	•	For each a:
	•	U(s,a,d) = c_puct * P_prior(a|s) * sqrt(total_N + 1.0) / (1.0 + N(s,a,d))
	•	S(s,a,d) = Q(s,a,d) + U(s,a,d)

Selection chooses:
	•	a* = argmax_{a in A_search(s)} S(s,a,d)

c_puct is a positive constant configured at deployment time.

4.7 Simulation

A single simulation from root s_root proceeds:
	1.	Initialization:
	•	s_0 = s_root
	•	For d = 0..D_max-1:
	2.	At depth d:
	•	If s_d is a terminal state or d == D_max:
	•	Break.
	•	Select action a_d using selection rule at (s_d, d).
	•	Simulate transition:
	•	Use environment, model, or external simulator to obtain:
	•	r_d in R
	•	s_{d+1} in S_search
	3.	After reaching terminal depth L_sim:
	•	Compute returns G_d for d = 0..L_sim-1:
	•	For example, discounted sum:
G_d = sum_{k=d..L_sim-1} gamma^{k-d} r_k
	4.	Backup:
	•	For d from 0 to L_sim-1:
	•	ADD(w, k_N(s_d,a_d,d), +1.0)
	•	ADD(w, k_W(s_d,a_d,d), +G_d)

A fixed number S_per_step of simulations MUST be used per environment time step to bound computation.

4.8 Action selection for environment

After S_per_step simulations at s_root:
	•	Choose a_root by either:
	•	a_root = argmax_a N(s_root,a,0)
or
	•	sample a_root from normalized N(s_root,a,0) over actions a.

Then apply:
	•	a_t = MAP_SEARCH_ACTION_TO_ENV(a_root)

	5.	Hierarchical Representation Module

5.1 Base representation

For any object x (typically s_t or s_search), define a base representation:

h_base(x) in R^{d_rep}

h_base(x) is provided by an external encoder or by internal derived features. This spec does not constrain its origin beyond determinism.

5.2 Representation nodes

For each active prefix p = (L, p_L), the representation module uses:
	•	Basis vector B_p in R^{d_rep}
	•	Gating parameters G_p in R^{g_max}

These are tied to parameters in the store:
	•	For i in {0..d_rep-1}:
	•	k_B(L,p,i) = (Hyp_REP, Level_REP, ROLE_REP_B, (L,p_L), Slot=i)
	•	B_p[i] = READ(w, k_B(L,p,i))
	•	For j in {0..g_max-1}:
	•	k_G(L,p,j) = (Hyp_REP, Level_REP, ROLE_REP_G, (L,p_L), Slot=j)
	•	G_p[j] = READ(w, k_G(L,p,j))

5.3 Gating function

Define a gating function:

GATE(G_p, h) -> g in [0,1]

Example implementation:
	•	g = sigma( sum_{j} a_j * G_p[j] + sum_{k} b_k * h[k] + c )

where a_j, b_k, c are fixed constants or small learnable parameters managed as another task using the same store.

5.4 Hierarchical composition of representation

Given x, define its prefix chain:
	•	path(x) = [p_0, p_1, …, p_{L_use}]
where p_L = C_infty(x)[0:L]

Define:
	•	h_0(x) = h_base(x)

For L from 0 to L_use:
	•	p = p_L
	•	B = B_p
	•	G = G_p
	•	g_L(x) = GATE(G, h_L(x))
	•	h_{L+1}(x) = (1 - alpha_L * g_L(x)) * h_L(x) + alpha_L * g_L(x) * B

where alpha_L in [0,1] are fixed coefficients.

Final representation:
	•	h(x) = h_{L_use+1}(x)

h(x) is used downstream in FEATURES_T, in MCTS features, and in Bayesian reconstruction features.

5.5 Training of representation parameters

Representation parameters (ROLE_REP_B, ROLE_REP_G) are not updated directly by the representation module; instead:
	•	Tasks that use h(x) in their features generate gradients that backpropagate to the parameters that define h(x) via the chain rule.
	•	Optionally, a dedicated representation management task at some Level >= 1:
	•	aggregates prefix-level statistics (e.g., loss, variance),
	•	defines its own FEATURES_REP_MANAGE,
	•	updates ROLE_REP_B and ROLE_REP_G via the linear core.

	6.	Bayesian Reconstruction Module

6.1 Partial observation model

Let z_partial denote a partially observed input, constructed from:
	•	subsets of H_t, s_t, external context, etc.
	•	z_partial is represented internally via BASE_CODE(z_partial) and potentially via h(z_partial).

6.2 Latent templates

Let H_BR be a finite or countable set of template indices h (latent hypotheses for missing structure or assumptions).

6.3 Approximate posterior

For each template h in H_BR, define BR features:

FEATURES_BR(z_partial, h) -> list [(k_i, v_i)]

with keys k_i typically using:
	•	Hyp = Hyp_BR
	•	Level = Level_BR
	•	Role = ROLE_BR_HYP
	•	PrefixMeta_BR(z_partial) = (L_BR, p_{L_BR}(z_partial))
	•	Slot depending on h and feature index.

Define the unnormalized score:

s_BR(h | z_partial; w) = sum_{(k_i,v_i)} READ(w, k_i) * v_i

Then the approximate posterior is:

q(h | z_partial; w) = exp(s_BR(h|z_partial; w)) / sum_{h’ in H_sub} exp(s_BR(h’|z_partial; w))

where H_sub is a finite subset of H_BR considered at this step (bounded in size by a constant H_max).

6.4 Reconstruction function

A reconstruction function:

RECONSTRUCT(z_partial, h) -> z_hat

returns an internal representation z_hat that fills in missing fields of z_partial according to template h. z_hat may include:
	•	a normalized problem statement,
	•	inferred default parameters,
	•	extended context for downstream tasks.

z_hat is then used as:
	•	input to SEARCH_STATE,
	•	part of ctx for base and meta tasks,
	•	input for further BR layers if necessary.

6.5 Reconstruction loss and gradients

For a given z_partial, outcome o (e.g., correct/incorrect, reward), and w, define a reconstruction loss ell_BR(w; z_partial, o).

Example: if a downstream predictive model M uses z_hat to predict o with likelihood p(o | z_hat, w), then:

ell_BR(w; z_partial, o) = - log sum_{h in H_sub} q(h | z_partial; w) * p(o | RECONSTRUCT(z_partial,h), w)

Gradients are obtained via:
	•	d ell_BR / d w(k) = sum over h in H_sub of:
	•	partial derivatives of q(h|z_partial; w) and p(o | z_hat(h); w) w.r.t. w(k)

The implementation uses the linear prediction core to compute s_BR and derivatives of q(h|z_partial). The predictive term p(o|z_hat,h) is treated as another task or module.
	7.	Task Abstraction and Meta-Control

7.1 Task components

Every task T is specified by:
	•	Hyp_T        : hypothesis identifier.
	•	Level_T      : control level.
	•	ROLES_T      : set of Roles used by this task.
	•	OBSERVE_T    : function mapping (H_t, e_t, auxiliary state) -> ctx_T.
	•	FEATURES_T   : function ctx_T -> list of (k_i, v_i).
	•	LABEL_T      : function mapping (H_t, e_t, auxiliary state) -> y_T (target).
	•	LOSS_T       : function (w, ctx_T, y_T) -> value and gradients.
	•	POLICY_T     : optional function mapping (w, ctx_T) -> decision (e.g., action index or schedule decision).
	•	UPDATE_SCHEDULE_T : meta-rule deciding whether to update T at step t.

7.2 Base control task

An example base task for environment policy:
	•	Hyp_T        = Hyp_ENV
	•	Level_T      = 0
	•	OBSERVE_T    : builds ctx from s_t, h(s_t), z_hat_t
	•	FEATURES_T   : emits (k_i, v_i) for action-specific weights (ROLE_FEAT)
	•	CANDIDATES_T : environment actions available at s_t
	•	LABEL_T      : may use returns, value targets, or policy gradient targets
	•	LOSS_T       : reinforcement learning loss (e.g., actor-critic)
	•	POLICY_T     : selects a_t given p(a|ctx_T,w) or via combination with MCTS
	•	UPDATE_SCHEDULE_T : defines how often and how many steps are used for training.

7.3 Meta tasks

Examples of meta tasks include:

Scheduler task:
	•	Hyp_T = Hyp_META_sched
	•	Level_T = 1
	•	OBSERVE_T: inspects losses, queue lengths, resource usage.
	•	POLICY_T: outputs which tasks to activate at step t (including MCTS).
	•	Parameters are stored under ROLE_META.

Loss mixing task:
	•	Hyp_T = Hyp_META_loss
	•	Level_T = 1
	•	OBSERVE_T: reads recent per-task loss statistics.
	•	POLICY_T: outputs mixing weights lambda_T for losses.
	•	Parameters stored in ROLE_META.

Prefix management task:
	•	Hyp_T = Hyp_META_prefix
	•	Level_T = 1
	•	OBSERVE_T: monitors performance and diversity within prefixes.
	•	POLICY_T: decides to split or merge prefixes.
	•	Splitting a prefix p into finer prefixes p0, p1,… may trigger allocation of new parameters in those prefixes and possible redistribution of data.
	•	Implementation MUST keep the number of new prefixes per step bounded to preserve O(1) operations.

Representation management task:
	•	Hyp_T = Hyp_META_rep
	•	OBSERVE_T: aggregates prefix-level loss metrics.
	•	Updates ROLE_REP_B and ROLE_REP_G to improve predictive performance.

7.4 Execution schedule

At each step t, the Core executes the following high-level loop:
	1.	Input:
	•	Receive s_t and r_{t-1}.
	•	Collect any external aux_t.
	2.	Reconstruction:
	•	Construct z_partial_t from (H_t, s_t, aux_t).
	•	For the BR module:
	•	sample or enumerate H_sub subset of templates.
	•	compute q(h | z_partial_t; w_t).
	•	choose h_t (e.g., MAP or soft selection).
	•	compute z_hat_t = RECONSTRUCT(z_partial_t, h_t).
	3.	Representation:
	•	Compute h(s_t) via base encoder and hierarchical composition.
	4.	Meta control:
	•	For each meta task T with Level_T >= 1:
	•	if UPDATE_SCHEDULE_T indicates evaluation at step t:
	•	compute ctx_T, POLICY_T, and possibly apply updates to w_t.
	5.	MCTS:
	•	If scheduler or configuration indicates:
	•	compute root s_root = SEARCH_STATE(s_t, H_t, z_hat_t).
	•	run S_per_step simulations as in section 4.7 using w_t.
	6.	Action selection:
	•	Combine base control policy and MCTS results as configured (e.g., weighted mixture or override).
	•	Produce a_t and send to environment.
	7.	Logging:
	•	Observe r_t and s_{t+1}.
	•	Form e_t = (s_t, a_t, r_t, s_{t+1}, aux_t).
	•	Append e_t to H_{t+1}.
	8.	Task updates:
	•	For each base or meta task T whose UPDATE_SCHEDULE_T indicates training at step t:
	•	compute ctx_T and y_T.
	•	compute features and loss.
	•	compute gradients and apply ADD(w_t, k, delta) as required.
	9.	Prepare for next step:
	•	Set w_{t+1} as the updated parameter store.
	•	Proceed to step t+1.

Concrete scheduling policy is configurable and may itself be learned.
	8.	Complexity and Scaling Guarantees

8.1 Per-step operation bounds

For each time step t, the Core MUST be configured so that:
	•	The number of tasks evaluated and updated is bounded by a constant T_max.
	•	For each evaluated task T:
	•	The number of features emitted per invocation m_T is bounded by a constant.
	•	The number of gradient and update operations is bounded by a constant multiple of m_T.
	•	The number of MCTS simulations S_per_step is bounded by a constant.
	•	Each simulation visits at most D_max depths, and at each node:
	•	selection uses a bounded number of READ operations,
	•	backup uses a bounded number of ADD operations.
	•	Reconstruction module:
	•	considers at most H_max templates per step,
	•	emits a bounded number of BR features per template.

Together with an O(1) Lossless Hashed Linear parameter store, this ensures:
	•	The total number of READ and ADD operations per time step is bounded by a constant C_total independent of:
	•	t (time step index),
	•	N_t (number of distinct states seen so far),
	•	U_t (number of keys used so far).

8.2 Memory usage and fractal scaling

Let U_t be the number of keys with w_t(k) != 0.

The Core MUST satisfy:
	•	New keys can be introduced only through:
	•	activation of previously inactive prefixes,
	•	allocation of new slots within existing (Hyp, Level, Role, PrefixMeta) combinations,
	•	tasks explicitly requesting additional slots based on performance metrics.

Prefix activation policies MUST be defined so that:
	•	Simple regions (prefixes with low loss and low variability) are not repeatedly subdivided.
	•	Prefix splitting is triggered only when measurable error or heterogeneity within the prefix exceeds configurable thresholds.
	•	The number of new prefixes and new slots created per step is bounded.

Under these constraints, U_t grows primarily with:
	•	the intrinsic complexity of the environment and tasks,
	•	the resolution required by the thresholds,

and not directly with the raw count of distinct visited states.
	9.	Implementation and Determinism

9.1 Implementation language

Any language and runtime can be used, provided that:
	•	The Lossless Hashed Linear store satisfies semantics and complexity requirements.
	•	The PRF and STREAM_PRF facilities are deterministic and collision-resistant in practice.
	•	Memory and time bounds are respected.

9.2 Deterministic behavior

To achieve deterministic replay, the implementation MUST ensure:
	•	All PRF and STREAM_PRF calls use fixed seeds and input encodings.
	•	Environment randomness is controlled by explicit seeds when replaying.
	•	Task scheduling, including meta decisions, uses only deterministic functions of (H_t, w_t, seeds).
	•	Floating point computations use fixed precision and deterministic ordering where bit-for-bit reproducibility is required.

SNAPSHOT and RESTORE MUST capture seeds, configuration, and all non-parameter task state necessary for replay.

9.3 Persistence

A snapshot SHOULD contain at minimum:
	•	All non-zero parameter key-value pairs w(k).
	•	All seeds (seed_key, seed_code, and other PRF seeds).
	•	Task configurations and hyperparameters.
	•	Any auxiliary state not encoded in w (e.g., running averages, counters).

RESTORE(snapshot) MUST reconstruct the Core in a state such that, when replayed on the same environment log, decisions are reproduced (up to allowed floating point differences).
	10.	Limitations and Extensions

10.1 Limitations

The Core specification:
	•	Does not guarantee convergence or optimality in arbitrary environments.
	•	Does not fix concrete forms of external encoders or simulators.
	•	Does not specify a unique choice of loss functions or meta tasks.

It defines a structure and complexity discipline in which such choices can be made.

10.2 Possible extensions

Future versions MAY extend:
	•	Role set to include structured memories (e.g., sequence-specific roles).
	•	Task class to include limited non-linear heads with bounded cost.
	•	More precise statistical guarantees on prefix splitting policies.
	•	Interfaces to specialized solvers (e.g., theorem provers) as oracles under the same key and task framework.

