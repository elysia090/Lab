Title: Fractal Bob v0.0.1 – Industrial Online Control Core Specification (Tightened)

Status: Draft
Language: ASCII, English only
	0.	Conventions, scope, and configuration

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

0.4 Fixed configuration and global bounds

A Core configuration specifies a finite set of constants that remain fixed for the lifetime of a run:

T_max         : maximum number of tasks evaluated per time step.
U_max_step    : maximum number of tasks updated per time step (U_max_step <= T_max).
m_T_max       : global upper bound on features per task invocation.
S_per_step    : maximum number of MCTS simulations per time step.
D_max         : maximum depth per MCTS simulation.
H_max_BR      : maximum number of Bayesian reconstruction templates per step.
L_prefix_max  : maximum prefix depth used by any module.
P_split_max   : maximum number of new prefixes that may be activated per time step.
Slot_alloc_max: maximum number of new slots that may be allocated per time step.
K_max_global  : upper bound on K_max(Hyp,Level,Role,L) over all arguments.

These constants are part of the configuration and do not change during a run. They may differ between deployments.

All per-step complexity guarantees in Section 8 are formulated in terms of these constants and do not depend on the elapsed horizon, number of distinct states, or number of keys.

0.5 Deterministic seeds and configuration identifiers

The Core uses a finite set of deterministic seeds:

seed_key      : seed for parameter ID derivation PRF.
seed_code     : seed for prefix stream PRF.
seed_task     : seed for any task-internal randomized choices (e.g., BR template subsampling).
seed_mcts     : seed for MCTS randomized exploration (if used).
seed_env      : optional seed controlling environment stochasticity in replay mode.

All seeds are recorded in configuration metadata and treated as immutable for the lifetime of a run. Reconfiguration, if any, proceeds by explicit publication of a new immutable configuration with new seeds and does not reinterpret past events.
	1.	Environment and interaction model

1.1 Time and interaction

Time steps are t in N. At each step t:
	•	s_t in S        : environment state or observation.
	•	a_t in A        : action chosen by the Core.
	•	r_t in R        : reward returned by the environment.
	•	s_{t+1} in S    : next state.

The environment defines a (possibly unknown) kernel:

P_env(s_{t+1}, r_t | s_t, a_t)

The Core observes (s_t, r_t, optional aux_t) and must choose a_t. The environment is treated as black-box except for any simulators exposed to MCTS or BR as oracles.

1.2 Event and history

Define:
	•	e_t = (s_t, a_t, r_t, s_{t+1}, aux_t)
	•	H_t = (e_0, …, e_{t-1})

aux_t may contain external solver outputs, diagnostics, or other signals that are considered read-only inputs to tasks.

1.3 Objectives

The Core is used to:
	•	Maximize expected discounted return
J = E[ sum_{t=0..∞} gamma^t r_t ] for some gamma in (0,1].
	•	Minimize a weighted sum of auxiliary task losses
L_total(w) = sum_{T in Tasks} lambda_T E[ ell_T(w; context_T) ]

where lambda_T >= 0 are configuration constants. No single optimization algorithm is mandated; the structure is designed so that gradient-based incremental methods can be applied with bounded per-step cost.

1.4 Determinism requirement

Given a fixed configuration (including seeds, constants, and any task configuration), a fixed initial state, and a fixed environment log (or environment kernel with fixed seed_env), the Core MUST behave deterministically:
	•	Task scheduling decisions are deterministic functions of (H_t, w_t, seeds).
	•	Any internal randomized choices (e.g., in BR or MCTS) are derived from fixed seeds and indices (t, task id, local counters) with no data-dependent reseeding.
	•	Floating point operations use fixed precision and fixed reduction orders where deterministic replay is required.

	2.	Key space and parameter store

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
Hyp_META_i  : meta tasks (scheduler, loss mixing, prefix management, etc).

Hyp is represented as an integer or enumeration. The set of active hypotheses is finite and fixed by configuration.

2.1.2 Level

Level in N indicates control depth:

0 : base control (direct environment actions).
1 : meta control (scheduling, loss mixing, representation management).
n : further meta layers (optional).

The configuration includes a maximum level L_max_Level. All tasks and keys satisfy Level <= L_max_Level.

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

The same Role can be used in multiple Hyp and Level combinations. The configuration must specify, for each (Hyp,Level), which Roles are permitted.

2.1.4 Prefix metadata

For any object x (state, history segment, template index, etc), define:
	•	BASE_CODE(x) in {0,1}^*: a finite bitstring encoding.
	•	seed_code: a global secret seed (constant across run).

Define an infinite pseudo-random bitstream:

C_infty(x) in {0,1}^N

via a deterministic stream PRF:

C_infty(x) = STREAM_PRF(seed_code, BASE_CODE(x))

For any depth L in {0, …, L_prefix_max}, define the prefix:

p_L(x) = C_infty(x)[0:L]  (first L bits)

PrefixMeta is defined as:

PrefixMeta = (L, p_L)

Only a finite subset of prefixes are active at any given time; parameters are allocated only for active prefixes. The configuration specifies:
	•	L_prefix_max  : maximum allowed prefix depth.
	•	Initial active prefixes at t=0.
	•	Policies for activating new prefixes (Section 8.2).

2.1.5 Slot

Slot is an integer in N with constraint:

0 <= Slot < K_max(Hyp, Level, Role, L)

for some configuration function K_max specifying the maximum number of distinct parameters per (Hyp, Level, Role, prefix depth L). K_max must satisfy:

K_max(Hyp, Level, Role, L) <= K_max_global

for all arguments. Slot allocation is deterministic and must respect per-step allocation bounds (Section 8.2).

2.2 Abstract key tuple and concrete ID

The semantic key is the tuple:

k = (Hyp, Level, Role, PrefixMeta, Slot) in K

For implementation, a concrete ID in a finite integer space is derived via:

RAW_KEY_BITS = PRF(seed_key, encode(Hyp, Level, Role, PrefixMeta, Slot, extra_context))
ID = ENCODE_BITS_TO_INT(RAW_KEY_BITS)

where:
	•	seed_key is a global secret seed.
	•	extra_context is optional, fixed per role design (e.g., to distinguish different namespaces).
	•	ENCODE_BITS_TO_INT is a stable mapping from bitstrings to integer identifiers (e.g., interpreting a fixed-length bitstring as an unsigned integer).

The parameter store uses ID as its internal key. The semantic key k is used in the specification and must be reconstructible from ID and auxiliary metadata when required for debugging or audit.

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

The parameter store MAY provide:

SNAPSHOT(w) -> snapshot_blob
RESTORE(snapshot_blob) -> new_state

SNAPSHOT and RESTORE must preserve the logical mapping from K to R and any auxiliary state required by tasks to maintain deterministic replay.

2.4 Lossless Hashed Linear implementation constraints

An implementation of the parameter store is regarded as compliant if the following hold.

2.4.1 Semantic constraints
	•	For every k in K, there is at most one stored value v = w(k).
	•	No two distinct keys k1 != k2 share the same stored location; there must be no collisions for live keys. If internal hashing structures are used, any collision resolution must preserve logical independence of keys.
	•	READ and ADD on different keys behave as if they operate on a logical sparse map from K to R.

2.4.2 Complexity constraints

Consider any sequence of key operations of length T (READs and ADDs) produced by a compliant Core execution.

There exists a constant C_store (depending only on configuration) such that:
	•	The worst-case time for each READ and each ADD is bounded by C_store, independent of:
	•	T (the number of operations so far),
	•	U_T (the number of distinct keys used so far),
	•	N_T (the number of distinct states or tasks seen so far).

Memory usage MUST be O(U_T) for some constant factor, where U_T is the count of keys k with w(k) != 0.

Incremental rehashing, compaction, or structural updates of the underlying store are permitted only if they can be decomposed into a sequence of bounded-cost steps interleaved with normal Core operations, such that no single READ or ADD exceeds the C_store bound. Global stop-the-world rehashes that violate this bound are not permitted.
	3.	Linear prediction core

3.1 Feature collections

For any task T, define a feature function:

FEATURES_T(ctx) -> list [(k_1, v_1), …, (k_m, v_m)]

where:
	•	ctx: a task-specific context object.
	•	k_i in K: parameter keys.
	•	v_i in R: feature values.
	•	m is bounded by a task-specific constant m_T, and m_T <= m_T_max for all tasks T.

FEATURES_T(ctx) must be computed in O(m_T) time using only a bounded number of READ calls to w and a bounded amount of local computation.

3.2 Linear score

Given w and features [(k_i, v_i)]:

score_T(ctx; w) = sum_{i=1..m} READ(w, k_i) * v_i

This score is used as the basic linear prediction primitive.

3.3 Probabilistic outputs

Binary case:
	•	p(y=1 | ctx, w) = sigma(score_T(ctx; w))
where sigma(z) = 1 / (1 + exp(-z))

Multiclass case with candidate actions C(ctx):
	•	For each a in C(ctx), compute FEATURES_T(ctx, a) and score_T(ctx, a; w).
	•	p(a | ctx, w) = exp(score_T(ctx,a; w)) / sum_{b in C(ctx)} exp(score_T(ctx,b; w))

The candidate set C(ctx) must be finite and its size bounded by a configuration constant C_max_actions for each relevant task. Other distributions (Gaussian, etc.) MAY be used, but are not specified.

3.4 Loss functions and gradients

Each task T defines a loss function:

ell_T(w; ctx, y)

where y is the target or label. Examples:

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
delta = -eta_T * grad(k_i) - eta_T * lambda_L2_T * READ(w, k_i)
ADD(w, k_i, delta)

The specification requires that:
	•	For any task T, the number of ADD calls per update is bounded by a constant proportional to m_T, independent of time.
	•	Any additional regularizers (e.g., L1) must be implemented via a bounded number of extra ADD calls per update.

3.5 Task definition

A task T is defined by:
	•	Hyp_T     : Hypothesis identifier.
	•	Level_T   : control level.
	•	Role set  : set of Roles this task uses.
	•	FEATURES_T(ctx) : feature function.
	•	CANDIDATES_T(ctx) : optional candidate set (e.g., actions).
	•	LOSS_T(w; ctx, y) and its gradients.
	•	UPDATE_SCHEDULE_T : rule deciding when updates are applied.

All tasks share the same global parameter store w and use the same READ/ADD operations. For each task T, the configuration must specify:
	•	An upper bound on the frequency of evaluation and updates (e.g., at most once per time step).
	•	Whether T is base-level (Level_T = 0) or meta-level (Level_T >= 1).
	•	Any dependencies on outputs of other tasks or modules (e.g., requiring h(s_t) or BR outputs).

	4.	Monte Carlo tree search module

4.1 Search state and actions

Define a search state set S_search and search action set A_search. The module uses a function:

SEARCH_STATE(s_t, H_t, z_hat_t) -> s_root in S_search

to map current environment state s_t, history H_t, and optional reconstructed context z_hat_t to a search root state s_root.

There is a mapping:

MAP_SEARCH_ACTION_TO_ENV(a_search) -> a_env in A

which maps chosen search actions to environment actions. This mapping is deterministic and specified by configuration.

4.2 Node identification and prefixing

A search node is characterized by (s_search, depth d), where d in {0, …, D_max}. For such (s, d), and an action a in A_search, define:
	•	PrefixMeta_MCTS(s) = (L_MCTS, p_{L_MCTS}(s)) for some fixed L_MCTS <= L_prefix_max.

Now define the keys:
	•	k_N(s,a,d) = (Hyp_MCTS, Level_MCTS, ROLE_N, PrefixMeta_MCTS(s), Slot_N(a,d))
	•	k_W(s,a,d) = (Hyp_MCTS, Level_MCTS, ROLE_Q, PrefixMeta_MCTS(s), Slot_W(a,d))

where Slot_N and Slot_W are deterministic, bounded mappings from (a, d) to small integers less than K_max(Hyp_MCTS, Level_MCTS, ROLE_N or ROLE_Q, L_MCTS).

4.3 Reading statistics

Define:
	•	N(s,a,d) = max(0, READ(w, k_N(s,a,d)))
	•	W(s,a,d) = READ(w, k_W(s,a,d))
	•	Q(s,a,d) = W(s,a,d) / max(1, N(s,a,d))

These provide visit counts, cumulative returns, and average returns for use in selection.

4.4 Policy prior for search

For the prior search policy, for each state s and action a, define:
	•	PrefixMeta_P(s) = (L_P, p_{L_P}(s)) for some fixed L_P <= L_prefix_max.

For indices j in {0, …, K_P-1}:
	•	k_P(s,a,j) = (Hyp_MCTS, Level_MCTS, ROLE_P, PrefixMeta_P(s), Slot_P(a,j))
	•	psi_j(s,a) : feature value, derived from PRF and possibly from representation h(s).

Then:
	•	score_P(s,a) = sum_j READ(w, k_P(s,a,j)) * psi_j(s,a)
	•	P_prior(a | s) = exp(score_P(s,a)) / sum_b exp(score_P(s,b))

The number K_P of features per (s,a) is a configuration constant.

4.5 Value estimate

Define a value model:
	•	PrefixMeta_V(s) = (L_V, p_{L_V}(s)) for some fixed L_V <= L_prefix_max.

For indices i in {0, …, K_V-1}:
	•	k_V(s,i) = (Hyp_MCTS, Level_MCTS, ROLE_V, PrefixMeta_V(s), Slot_V(i))
	•	phi_i(s) : feature value based on h(s) and PRF.

Then:
	•	V_hat(s) = sum_i READ(w, k_V(s,i)) * phi_i(s)

The number K_V of value features is a configuration constant.

4.6 Selection rule

At node (s, d), with candidate actions A_search(s):
	•	total_N = sum_{b in A_search(s)} N(s,b,d)

For each a in A_search(s):
	•	U(s,a,d) = c_puct * P_prior(a|s) * sqrt(total_N + 1.0) / (1.0 + N(s,a,d))
	•	S(s,a,d) = Q(s,a,d) + U(s,a,d)

Selection chooses:
	•	a* = argmax_{a in A_search(s)} S(s,a,d)

c_puct is a positive constant configured at deployment time. Any ties are broken deterministically (e.g., by fixed ordering over actions).

4.7 Simulation

A single simulation from root s_root proceeds:
	1.	Initialization:
	•	s_0 = s_root
	2.	For d = 0..D_max-1:
	•	If s_d is a terminal state or d == D_max:
	•	Break.
	•	Select action a_d using the selection rule at (s_d, d).
	•	Simulate transition:
	•	Using environment, model, or external simulator obtain:
	•	r_d in R
	•	s_{d+1} in S_search
	3.	After reaching terminal depth L_sim <= D_max:
	•	Compute returns G_d for d = 0..L_sim-1; for example, discounted sum:
G_d = sum_{k=d..L_sim-1} gamma^{k-d} r_k
	4.	Backup:
	•	For d from 0 to L_sim-1:
	•	ADD(w, k_N(s_d,a_d,d), +1.0)
	•	ADD(w, k_W(s_d,a_d,d), +G_d)

A fixed number S_per_step of simulations MUST be used per environment time step to bound computation. If S_per_step is zero, the MCTS module can be disabled.

4.8 Action selection for environment

After S_per_step simulations at s_root:
	•	Choose a_root by either:
	•	a_root = argmax_a N(s_root,a,0)
or
	•	sample a_root from normalized N(s_root,a,0) over actions a (using seed_mcts and deterministic indexing).

Then apply:
	•	a_t = MAP_SEARCH_ACTION_TO_ENV(a_root)

	5.	Hierarchical representation module

5.1 Base representation

For any object x (typically s_t or s_search), define a base representation:

h_base(x) in R^{d_rep}

h_base(x) is provided by an external encoder or by internal derived features. The spec does not constrain its origin beyond determinism and a fixed dimension d_rep. The configuration must specify which objects x admit a base representation.

5.2 Representation nodes

For each active prefix p = (L, p_L), the representation module uses:
	•	Basis vector B_p in R^{d_rep}
	•	Gating parameters G_p in R^{g_max}

These are tied to parameters in the store:

For i in {0..d_rep-1}:
	•	k_B(L,p,i) = (Hyp_REP, Level_REP, ROLE_REP_B, (L,p_L), Slot=i)
	•	B_p[i] = READ(w, k_B(L,p,i))

For j in {0..g_max-1}:
	•	k_G(L,p,j) = (Hyp_REP, Level_REP, ROLE_REP_G, (L,p_L), Slot=j)
	•	G_p[j] = READ(w, k_G(L,p,j))

The configuration fixes d_rep, g_max, and the Level_REP used for representation parameters.

5.3 Gating function

Define a gating function:

GATE(G_p, h) -> g in [0,1]

Example implementation:
	•	g = sigma( sum_{j} a_j * G_p[j] + sum_{k} b_k * h[k] + c )

where a_j, b_k, c are fixed constants or small learnable parameters managed as another task using the same store. Regardless of implementation, the cost of GATE must be bounded by a configuration constant C_gate, independent of the number of prefixes or time.

5.4 Hierarchical composition of representation

Given x, define its prefix chain:
	•	path(x) = [p_0, p_1, …, p_{L_use}]

where p_L = C_infty(x)[0:L] for L in {0, …, L_use}, and L_use <= L_prefix_max_use <= L_prefix_max is a configuration constant. The mapping from x to L_use must be deterministic and bounded.

Define:
	•	h_0(x) = h_base(x)

For L from 0 to L_use:
	•	p = p_L
	•	B = B_p
	•	G = G_p
	•	g_L(x) = GATE(G, h_L(x))
	•	h_{L+1}(x) = (1 - alpha_L * g_L(x)) * h_L(x) + alpha_L * g_L(x) * B

where alpha_L in [0,1] are fixed coefficients (possibly dependent on L but not on t). Final representation:
	•	h(x) = h_{L_use+1}(x)

h(x) is used downstream in FEATURES_T, in MCTS features, and in Bayesian reconstruction features as needed. The number of active prefixes on the path and the number of representation updates per step must respect the bounds in Section 8.2.

5.5 Training of representation parameters

Representation parameters (ROLE_REP_B, ROLE_REP_G) are not updated directly by the representation module; instead:
	•	Tasks that use h(x) in their features generate gradients that backpropagate to the parameters that define h(x) via the chain rule and the linear core.
	•	Optionally, a dedicated representation management task at some Level >= 1:
	•	aggregates prefix-level statistics (e.g., loss, variance),
	•	defines its own FEATURES_REP_MANAGE,
	•	updates ROLE_REP_B and ROLE_REP_G via the linear core.

The number of such management tasks and their invocation frequency must respect the global scheduling bounds.
	6.	Bayesian reconstruction module

6.1 Partial observation model

Let z_partial denote a partially observed input, constructed from:
	•	subsets of H_t, s_t, external context, etc.

z_partial is represented internally via BASE_CODE(z_partial) and potentially via h(z_partial) using the same encoding and representation pipeline as for states.

6.2 Latent templates

Let H_BR be a finite or countable set of template indices h (latent hypotheses for missing structure or assumptions). The configuration specifies:
	•	A finite template family or a generator procedure.
	•	A maximum per-step subset size H_max_BR.

6.3 Approximate posterior

For each template h in H_BR, define BR features:

FEATURES_BR(z_partial, h) -> list [(k_i, v_i)]

with keys k_i typically using:
	•	Hyp = Hyp_BR
	•	Level = Level_BR
	•	Role = ROLE_BR_HYP
	•	PrefixMeta_BR(z_partial) = (L_BR, p_{L_BR}(z_partial)) with L_BR <= L_prefix_max
	•	Slot determined by h and feature index.

Define the unnormalized score:

s_BR(h | z_partial; w) = sum_{(k_i,v_i)} READ(w, k_i) * v_i

Then the approximate posterior is:

q(h | z_partial; w) = exp(s_BR(h|z_partial; w)) / sum_{h’ in H_sub} exp(s_BR(h’|z_partial; w))

where H_sub is a finite subset of H_BR considered at this step, chosen deterministically from H_BR and of size at most H_max_BR. The selection rule for H_sub can be based on precomputed priors, heuristics, or fixed sampling patterns, but must be deterministic under fixed seeds.

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

The cost of RECONSTRUCT per template must be bounded by a constant C_reconstruct, and the total cost per step is bounded by C_reconstruct * H_max_BR plus linear core operations.

6.5 Reconstruction loss and gradients

For a given z_partial, outcome o (e.g., correct/incorrect, reward), and w, define a reconstruction loss ell_BR(w; z_partial, o).

Example: if a downstream predictive model M uses z_hat to predict o with likelihood p(o | z_hat, w), then:

ell_BR(w; z_partial, o) = - log sum_{h in H_sub} q(h | z_partial; w) * p(o | RECONSTRUCT(z_partial,h), w)

Gradients are obtained via:
	•	derivatives of q(h|z_partial; w) and p(o | z_hat(h); w) w.r.t. w(k), using the linear prediction core wherever applicable.

The implementation must ensure that the number of READ and ADD operations contributed by BR per time step is bounded by a constant depending only on H_max_BR, m_T_max, and the number of tasks that consume z_hat.
	7.	Task abstraction and meta-control

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

OBSERVE_T, LABEL_T, and POLICY_T must be computable in bounded time per call, with bounds specified as configuration constants for each task and respecting global step bounds.

7.2 Task dependency discipline

Tasks may depend on:
	•	Environment inputs (s_t, r_{t-1}, aux_t).
	•	Outputs of other tasks from the same time step, provided there is a well-founded evaluation order.
	•	Long-term state encoded in w or task-specific auxiliary state.

The configuration must define a partial order over tasks such that:
	•	If T2 depends on outputs of T1, then T1 is evaluated before T2 at each step.
	•	There is no cyclic dependency in the evaluation graph.

Meta tasks with Level_T >= 1 may read statistics from base tasks (Level 0) and from other meta tasks with lower Level, but not from tasks at higher Level, to avoid cyclic dependencies. This induces a natural order by (Level_T, task_id).

7.3 Base control task

An example base task for environment policy:
	•	Hyp_T        = Hyp_ENV
	•	Level_T      = 0
	•	OBSERVE_T    : builds ctx from s_t, h(s_t), z_hat_t.
	•	FEATURES_T   : emits (k_i, v_i) for action-specific weights (ROLE_FEAT).
	•	CANDIDATES_T : environment actions available at s_t.
	•	LABEL_T      : may use returns, value targets, or policy gradient targets.
	•	LOSS_T       : reinforcement learning loss (e.g., actor-critic).
	•	POLICY_T     : selects a_t given p(a|ctx_T,w) or via combination with MCTS.
	•	UPDATE_SCHEDULE_T : defines how often and how many steps are used for training.

The configuration must state whether MCTS overrides or mixes with the base policy and how their outputs are combined.

7.4 Meta tasks

Examples of meta tasks include:

Scheduler task:
	•	Hyp_T = Hyp_META_sched
	•	Level_T = 1
	•	OBSERVE_T: inspects losses, queue lengths, resource usage, and possibly BR diagnostics.
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
	•	OBSERVE_T: monitors performance and diversity within prefixes using prefix-indexed statistics.
	•	POLICY_T: decides to split or merge prefixes.
	•	Splitting a prefix p into finer prefixes p0, p1,… may trigger allocation of new parameters in those prefixes and possible redistribution of data.
	•	Implementation MUST keep the number of new prefixes per step bounded by P_split_max and the number of new slots per step bounded by Slot_alloc_max.

Representation management task:
	•	Hyp_T = Hyp_META_rep
	•	Level_T = 1 (or higher)
	•	OBSERVE_T: aggregates prefix-level loss metrics.
	•	UPDATES: ROLE_REP_B and ROLE_REP_G to improve predictive performance via the linear core.

7.5 Execution schedule

At each step t, the Core executes the following high-level loop:
	1.	Input:
	•	Receive s_t and r_{t-1}.
	•	Collect any external aux_t.
	2.	Reconstruction:
	•	Construct z_partial_t from (H_t, s_t, aux_t).
	•	For the BR module:
	•	select H_sub subset of templates, |H_sub| <= H_max_BR.
	•	compute q(h | z_partial_t; w_t).
	•	choose h_t (e.g., MAP or soft selection under fixed seeds).
	•	compute z_hat_t = RECONSTRUCT(z_partial_t, h_t).
	3.	Representation:
	•	Compute h(s_t) via base encoder and hierarchical composition.
	4.	Meta control:
	•	For each meta task T in non-decreasing order of Level_T:
	•	if UPDATE_SCHEDULE_T indicates evaluation at step t:
	•	compute ctx_T, POLICY_T, and possibly apply updates to w_t.
	5.	MCTS:
	•	If scheduler or configuration indicates MCTS is active:
	•	compute root s_root = SEARCH_STATE(s_t, H_t, z_hat_t).
	•	run S_per_step simulations as in Section 4.7 using w_t.
	6.	Action selection:
	•	Combine base control policy and MCTS results as configured (e.g., weighted mixture, override, or switching rule).
	•	Produce a_t and send to environment.
	7.	Logging:
	•	Observe r_t and s_{t+1}.
	•	Form e_t = (s_t, a_t, r_t, s_{t+1}, aux_t).
	•	Append e_t to H_{t+1}.
	8.	Task updates:
	•	For each base or meta task T whose UPDATE_SCHEDULE_T indicates training at step t:
	•	compute ctx_T and y_T.
	•	compute features and loss via the linear core.
	•	compute gradients and apply ADD(w_t, k, delta) as required.
	9.	Prepare for next step:
	•	Set w_{t+1} as the updated parameter store (or treat w_t as updated in-place with appropriate snapshotting).
	•	Proceed to step t+1.

Concrete scheduling policy (e.g., which tasks are evaluated every step, which tasks are evaluated every k steps) is configurable and may itself be learned, subject to the global bounds in Section 8.
	8.	Complexity and scaling guarantees

8.1 Per-step operation bounds

For each time step t, the Core MUST be configured so that:
	•	The number of tasks evaluated (OBSERVE_T / POLICY_T) is bounded by T_max.
	•	The number of tasks updated (LOSS_T / gradients / ADD) is bounded by U_max_step <= T_max.
	•	For each evaluated task T:
	•	The number of features emitted per invocation m_T is bounded by m_T_max.
	•	The number of gradient and update operations is bounded by a constant multiple of m_T_max.
	•	The number of MCTS simulations S_per_step is bounded by a constant S_per_step.
	•	Each simulation visits at most D_max depths, and at each node:
	•	selection uses a bounded number C_select of READ operations.
	•	backup uses a bounded number C_backup of ADD operations.
	•	The reconstruction module:
	•	considers at most H_max_BR templates per step.
	•	emits a bounded number of BR features per template.
	•	The representation module:
	•	uses at most L_prefix_max_use prefix levels per representation.
	•	activates at most P_split_max new prefixes per step.
	•	allocates at most Slot_alloc_max new slots per step.

Together with an O(1) Lossless Hashed Linear parameter store, this ensures:
	•	The total number of READ and ADD operations per time step is bounded by a constant C_total that depends only on configuration constants (T_max, U_max_step, m_T_max, S_per_step, D_max, H_max_BR, L_prefix_max_use, P_split_max, Slot_alloc_max, K_max_global), and does not depend on:
	•	t (time step index),
	•	N_t (number of distinct states seen so far),
	•	U_t (number of keys used so far),
	•	length of the history H_t.

8.2 Memory usage and fractal scaling

Let U_t be the number of keys with w_t(k) != 0.

The Core MUST satisfy:
	•	New keys can be introduced only through:
	•	activation of previously inactive prefixes,
	•	allocation of new slots within existing (Hyp, Level, Role, PrefixMeta) combinations,
	•	tasks explicitly requesting additional slots based on performance metrics and governed by Slot_alloc_max.

Prefix activation policies MUST be defined so that:
	•	Simple regions (prefixes with low loss and low variability) are not repeatedly subdivided.
	•	Prefix splitting is triggered only when measurable error or heterogeneity within the prefix exceeds configurable thresholds.
	•	The number of new prefixes created per step is bounded by P_split_max.

Under these constraints, U_t grows primarily with:
	•	the intrinsic complexity of the environment and tasks,
	•	the resolution required by the thresholds,

and not directly with the raw count of distinct visited states. The fractal nature of PrefixMeta allows refinement only where needed, while keeping per-step operations and per-prefix capacity bounded.
	9.	Implementation and determinism

9.1 Implementation language

Any language and runtime can be used, provided that:
	•	The Lossless Hashed Linear store satisfies semantic and complexity requirements in Section 2.
	•	The PRF and STREAM_PRF facilities are deterministic and collision-resistant in practice.
	•	Memory and time bounds per step are respected for all configurations.

9.2 Deterministic behavior

To achieve deterministic replay, the implementation MUST ensure:
	•	All PRF and STREAM_PRF calls use fixed seeds and input encodings as specified in configuration.
	•	Environment randomness is controlled by explicit seeds when replaying (seed_env).
	•	Task scheduling, including meta decisions, uses only deterministic functions of (H_t, w_t, seeds, configuration constants).
	•	Floating point computations use fixed precision (e.g., IEEE-754 double) and deterministic ordering of operations where bit-for-bit reproducibility is required.
	•	Any nondeterminism from concurrency or parallelism is eliminated or fixed via deterministic reduction orders and synchronization.

SNAPSHOT and RESTORE MUST capture seeds, configuration, and all non-parameter task state necessary for replay.

9.3 Persistence

A snapshot SHOULD contain at minimum:
	•	All non-zero parameter key-value pairs w(k).
	•	All seeds (seed_key, seed_code, seed_task, seed_mcts, seed_env if used).
	•	Task configurations and hyperparameters.
	•	Any auxiliary state not encoded in w (e.g., running averages, counters, prefix statistics).
	•	Any metadata describing active prefixes and slot allocation state.

RESTORE(snapshot) MUST reconstruct the Core in a state such that, when replayed on the same environment log, decisions (a_t) and any recorded task-level outputs are reproduced, up to allowed floating point differences.
	10.	Limitations and extensions

10.1 Limitations

The Core specification:
	•	Does not guarantee convergence or optimality in arbitrary environments.
	•	Does not fix concrete forms of external encoders, simulators, or reward shaping.
	•	Does not specify a unique choice of loss functions or meta tasks.
	•	Assumes the existence of a Lossless Hashed Linear implementation satisfying per-operation worst-case bounds, which may be nontrivial to realize in practice.

It defines a structure and complexity discipline in which such choices can be made and implemented with bounded per-step cost.

10.2 Possible extensions

Future versions MAY extend:
	•	Role set to include structured memories (e.g., sequence-specific roles, episodic buffers).
	•	Task class to include limited non-linear heads with bounded cost beyond linear models.
	•	More precise statistical guarantees on prefix splitting policies and their effect on U_t.
	•	Interfaces to specialized solvers (e.g., theorem provers, external planners) as oracles under the same key and task framework, with explicit cost bounds.
	•	Multi-agent extensions where multiple Cores share environment and partial state, while preserving per-agent constant-time constraints.

