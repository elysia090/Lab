Title
Search v0.0.1 – Fixed-Budget Abstract Game Search Module (Tightened)

Status
Draft, self-contained, ASCII, English only
	0.	Conventions, scope, and non-goals

0.1 Mathematical notation

R          : set of real numbers
N          : set of nonnegative integers {0,1,2,…}
N_plus     : set of positive integers {1,2,…}

For random variables X, E[X] denotes expectation.
For a function f and argument x, f(x) is the value of f at x.

Indicator of an event E is 1{E} in {0,1}.

Logarithms “log” are natural logs.

0.2 High-level purpose

Search v0.0.1 is a fixed-budget decision module that, at each environment step t,
	1.	maps the concrete state s_t to an abstract decision site d_t,
	2.	queries a reference strategy and preference for actions at d_t,
	3.	runs a bounded number of rollouts up to fixed depth using a mixture of reference, model, and local strategy,
	4.	updates local adjustments for visited site-action pairs based on observed outcomes,
	5.	returns an action distribution pi_dec_t over concrete environment actions.

Key properties:
	•	Per-step work is bounded by configuration constants (N_roll, T_h, A_max, etc.), independent of total horizon and number of sites.
	•	All randomness is derived from fixed seeds and indices, so the module is deterministic given a log and a seed.
	•	The algorithm does not name any external algorithm family; it is defined purely in terms of its own state and update equations.

0.3 Non-goals

The module does NOT attempt to:
	•	Prove convergence to an equilibrium in arbitrary multi-agent settings.
	•	Guarantee optimality in arbitrary Markov decision processes.
	•	Specify how reference parameters are trained; it only constrains their form and usage.
	•	Define the external model (value function or policy) in any detail.

	1.	Configuration and constants

A configuration C_Search for Search v0.0.1 consists of:

gamma         in (0,1]       ; discount factor
N_roll        in N_plus      ; rollouts per environment step
T_h           in N_plus      ; maximum rollout horizon (depth)
A_max         in N_plus      ; upper bound on number of actions at any site
w_ref         >= 0           ; weight for reference preference
w_adj         >= 0           ; weight for local adjustment
w_model       >= 0           ; weight for model logits
alpha_ref     >= 0           ; mixing weight for reference strategy in rollout
alpha_loc     >= 0           ; mixing weight for local strategy in rollout
alpha_model   >= 0           ; mixing weight for model policy in rollout
eta_adj       > 0            ; step size for adjustment updates
u_min         <= 0           ; lower clamp bound for adjustment weights
u_max         >= 0           ; upper clamp bound for adjustment weights
epsilon_prob  in (0,1)       ; minimum probability for numerical safety
eps_log       in (0,1)       ; lower bound inside log to avoid log(0)
seed_search   in N           ; base seed for all pseudo-random choices

With constraint:

alpha_ref + alpha_loc + alpha_model > 0

so that rollout policy is well-defined.

Some constants may be set to zero to disable the corresponding component (for example w_model = 0 if no model logits are used).

All constants are fixed for the lifetime of a run. Changing them requires publishing a new configuration and restarting or branching the run.
	2.	External interfaces

2.1 Environment simulator

The environment or its model provides:

Simulate(s, a) -> (s_next, r_env, done_flag)

where
	•	s, s_next are concrete states in some space S,
	•	a is a concrete action in some space A,
	•	r_env in R is the reward for taking action a at state s,
	•	done_flag in {True, False} indicates termination.

Simulate must be deterministic given its arguments and the environment seed. Search treats Simulate as a black box.

2.2 Abstract site mapping

Two deterministic mappings are provided:

SiteOf(s) -> d in D_site
EnvActions(d, s) -> list of actions in A

where D_site is an abstract decision-site set.

Requirements:
	1.	For any concrete state s, SiteOf(s) returns a well-defined decision site d.
	2.	For any (d,s), EnvActions(d,s) returns a finite list C(d,s) of admissible environment actions.
	3.	|C(d,s)| <= A_max for all (d,s).
	4.	SiteOf and EnvActions are deterministic functions of (s, configuration) with no hidden state.

2.3 Optional value and policy functions

If available, a model provides:

Value(s) -> scalar in R
PolicyPrior(d, s) -> dict mapping a in EnvActions(d,s) to probability in [0,1]

Requirements:
	1.	Value is a deterministic function of (s, configuration, model parameters).
	2.	PolicyPrior(d,s) returns a distribution over EnvActions(d,s) with sum 1 and nonnegative entries.
	3.	If PolicyPrior is not available, w_model and alpha_model must be set to 0.

Search does not specify how Value or PolicyPrior are computed.
	3.	Parameter storage for reference and adjustment

3.1 Logical key space

For each decision site d in D_site and each action a in A_site(d), the module conceptually maintains:

Base preference     B(d,a) in R
Reference mass      R_ref(d,a) in [0, +infinity)
Local adjustment    U(d,a) in R

A_site(d) may be larger than EnvActions(d,s); the subset C(d,s) = EnvActions(d,s) is used at runtime.

3.2 Storage interface

The implementation provides functions:

ReadBasePref(d, a) -> float      // returns B(d,a) or 0.0 if uninitialized
ReadRefMass(d, a)  -> float      // returns R_ref(d,a) or 0.0 if uninitialized
ReadAdj(d, a)      -> float      // returns U(d,a) or 0.0 if uninitialized

WriteAdj(d, a, u_new)            // sets U(d,a) := u_new

Semantics:
	1.	For any (d,a), ReadBasePref, ReadRefMass, and ReadAdj return well-defined real numbers.
	2.	If a value has never been written or set, Read* returns 0.0 by default.
	3.	WriteAdj operates in bounded time independent of total number of (d,a) pairs observed so far.

Reference parameters B and R_ref are expected to be updated rarely, e.g., only offline. The module itself only reads them at runtime.

3.3 Reference strategy induced from masses

At runtime, the reference strategy at a site d and concrete state s is induced from R_ref:

Let C = EnvActions(d,s).
Let m_a = max( ReadRefMass(d,a), 0.0 ) for a in C.
Let M = sum_{a in C} m_a.

Define:
	•	if M > 0:
pi_ref(d,s)(a) = m_a / M
	•	else (all masses 0):
pi_ref(d,s)(a) = 1 / |C| for a in C (uniform)

pi_ref(d,s) is a probability distribution over C. Note that R_ref(d,a) is a nonnegative “mass”, not necessarily normalized.
	4.	Per-site scoring and decision distribution

4.1 Model logit

At site d, state s, action a in C = EnvActions(d,s),

if PolicyPrior is available:

p_model_raw = PolicyPrior(d,s).get(a, 0.0)
p_model = max(p_model_raw, eps_log)
l_model(d,s,a) = log(p_model)

else:

l_model(d,s,a) = 0.0

4.2 Reference preference score

Base preference score is:

b(d,a) = ReadBasePref(d,a)

No constraints on b(d,a) are imposed beyond being finite when used.

4.3 Local adjustment score

Local adjustment weight is:

u(d,a) = ReadAdj(d,a)

When computing a score, only the nonnegative part is used:

u_plus(d,a) = max(u(d,a), 0.0)

This prevents negative adjustment from directly amplifying the score in the wrong direction; negative values implicitly reduce action probability through relative differences.

4.4 Combined pre-activation at site

Combined score at site d and state s for action a in C is:

z(d,s,a) = w_ref   * b(d,a)
+ w_adj   * u_plus(d,a)
+ w_model * l_model(d,s,a)

where weights w_ref, w_adj, w_model come from configuration.

4.5 Site-local decision distribution

For site d and concrete state s, define:

z_max = max_{a in C} z(d,s,a)

For each a in C, define numerically stable exponents:

e(d,s,a) = exp( z(d,s,a) - z_max )

Z_sum = sum_{a in C} e(d,s,a)

Distribution:

pi_dec(d,s)(a) = e(d,s,a) / max(Z_sum, epsilon_prob * |C|)

If Z_sum is very small, epsilon_prob * |C| provides a floor; effectively this biases toward uniform.

pi_dec(d,s) is the decision distribution used to choose the environment action at the root site.
	5.	Rollout policy and rollout distribution

5.1 Local strategy for rollouts

At site d and state s, compute z(d,s,a) as above and form:

pi_loc(d,s)(a) = exp(z(d,s,a) - z_max) / sum_{b in C} exp(z(d,s,b) - z_max)

where z_max and C = EnvActions(d,s) are as before.

If the denominator is zero (all exponents underflow), switch to uniform over C.

5.2 Reference strategy for rollouts

Use the site-level reference strategy:

pi_ref(d,s) defined from R_ref and C = EnvActions(d,s) as in section 3.3.

5.3 Model strategy for rollouts

If PolicyPrior is available, define:

pi_model(d,s)(a) = PolicyPrior(d,s)[a] restricted to C, renormalized if needed.

If not available, treat pi_model as undefined and set alpha_model = 0.

5.4 Rollout policy mixture

At site d and state s, define unnormalized mixture weights:

w_mix_ref   = alpha_ref
w_mix_loc   = alpha_loc
w_mix_model = alpha_model

If a component is unavailable (for example PolicyPrior missing), its w_mix_* must be set to 0 in configuration.

Normalize mixture weights if desired, or treat them as absolute weights. A simple normalized mixture is:

W_total = w_mix_ref + w_mix_loc + w_mix_model

For each action a in C:

pi_roll(d,s)(a) =
( w_mix_ref   * pi_ref(d,s)(a)
	•	w_mix_loc   * pi_loc(d,s)(a)
	•	w_mix_model * pi_model(d,s)(a) ) / max(W_total, epsilon_prob)

If W_total = 0, set pi_roll(d,s) to uniform over C.

pi_roll is the distribution used during rollouts to sample actions.
	6.	Rollout trajectories and return estimates

6.1 Single rollout structure

At environment step t with concrete state s_t, Search performs N_roll independent rollouts for k = 1..N_roll.

For rollout k:
	1.	Initialize

s_0 = s_t
h = 0
	2.	While h < T_h:

	•	Determine site and actions:
d_h = SiteOf(s_h)
C_h = EnvActions(d_h, s_h)
	•	If C_h is empty: break (terminal).
	•	Compute rollout policy pi_roll(d_h, s_h) over C_h as in section 5.
	•	Sample action a_h from pi_roll(d_h, s_h) using seed_search, rollout index k, and step index h to derive randomness deterministically.
	•	Call Simulate(s_h, a_h) -> (s_{h+1}, r_h, done_h).
	•	Record (d_h, C_h, a_h, r_h).
	•	If done_h == True: set h_end = h, break.
	•	Else increment h := h + 1 and continue.

If the loop ends without done_h or without empty C_h, define h_end = last h visited. Let H_k = h_end.

6.2 Terminal bootstrap

For rollout k, after reaching H_k:

If Value is available:

G_terminal = Value(s_{H_k})

Else:

G_terminal = 0.0

6.3 Backward return propagation

For rollout k, define discounted returns G_h for h = 0..H_k as:

G_{H_k} = r_{H_k} + gamma * G_terminal
For h = H_k-1 down to 0:

G_h = r_h + gamma * G_{h+1}

If there is no r_{H_k} (for example, break due to C_h empty without Simulate), treat G_{H_k} = G_terminal.

Note: this definition yields a standard discounted return starting at step h, with bootstrapped tail from Value at terminal state.

6.4 Single-trajectory baseline estimates

For each visited site h in rollout k:
	•	site d_h, candidate set C_h, chosen action a_h, return G_h from above.

Define baseline V_base(d_h) approximately as expected return under reference strategy at that site.

A simple one-sample approximation:
	1.	Compute reference strategy pi_ref(d_h,s_h) over C_h (section 3.3).
	2.	Use G_h as the value of the chosen action and approximate others by the baseline itself.

To avoid circular definition, use:

V_base(d_h) = G_h

and advantages:

A_h(a_h) = G_h - V_base(d_h) = 0
A_h(a != a_h) defined as 0 (no update).

This trivial baseline yields no signal. To get nontrivial adjustment, use the following heuristic:

Define:

V_est_root(d_h) = mean return from d_h across rollouts (maintained in diagnostics).
For a single rollout, approximate:

V_base(d_h) = V_est_root(d_h)

and advantage for chosen action:

A_h(a_h) = G_h - V_base(d_h)

All other actions a != a_h receive no adjustment in that rollout.

The spec requires:
	•	For each visited (d_h, a_h), an advantage A_h(a_h) is computed as a bounded real value.
	•	Only A_h(a_h) is used for weight updates; advantages for unchosen actions may be treated as zero.

Concrete approximation of V_base and A_h may be improved or made more precise, but must remain bounded-cost per visited site.
	7.	Local adjustment update rule

7.1 Per-visit update

For each rollout k and each step h in that rollout:
	•	site d_h, chosen action a_h, advantage A_h(a_h) from section 6.4.

Define update increment:

delta(d_h, a_h) = eta_adj * A_h(a_h)

where eta_adj > 0 is fixed.

Read current adjustment:

u_old = ReadAdj(d_h, a_h)

Propose:

u_raw = u_old + delta(d_h, a_h)

Clamp to bounds:

u_new = min( max(u_raw, u_min), u_max )

Write adjustment:

WriteAdj(d_h, a_h, u_new)

7.2 Bounded work and sparsity

At each environment step:
	•	There are N_roll rollouts.
	•	Each rollout has at most T_h visited steps.

Thus:
	•	The number of WriteAdj operations per step is at most N_roll * T_h.
	•	The number of ReadAdj operations during updates is at most N_roll * T_h.

Only chosen actions in visited sites receive updates; all other (d,a) keep their previous U(d,a) values.
	8.	Online step algorithm

8.1 Input

At environment step t:
	•	Current concrete state s_t.
	•	Access to Simulate, SiteOf, EnvActions, Value (optional), PolicyPrior (optional).
	•	Configured constants and parameter storage.

8.2 Algorithm Step_Search(s_t)

The module executes:
	1.	Rollout phase

For k in {1,..,N_roll}:
	•	Perform rollout k as in section 6.1, using pi_roll for action selection.
	•	Compute returns G_h by backward propagation as in section 6.3.
	•	For each visited step h, compute advantages A_h(a_h) (section 6.4) and perform updates (section 7.1).

	2.	Decision phase at root

	•	Compute root site and actions:
d_root = SiteOf(s_t)
C_root = EnvActions(d_root, s_t)
	•	For each a in C_root:
	•	b(d_root,a) = ReadBasePref(d_root,a)
	•	u(d_root,a) = ReadAdj(d_root,a)
	•	u_plus = max(u(d_root,a), 0.0)
	•	l_model(d_root,s_t,a) from PolicyPrior if available, else 0.0
	•	z(d_root,s_t,a) = w_ref * b(d_root,a) + w_adj * u_plus + w_model * l_model(d_root,s_t,a)
	•	Compute pi_dec(d_root,s_t) as in section 4.5.

	3.	Action choice

Option A (greedy):

a_t = argmax_{a in C_root} pi_dec(d_root,s_t)(a)

Option B (sampling):

a_t is sampled from pi_dec(d_root,s_t) using seed_search and indices (t, local counter).

The choice between Option A and B is a configuration parameter.
	4.	Output

Return:
	•	selected action a_t,
	•	optionally the distribution pi_dec(d_root,s_t) and diagnostics.

The module does not directly interact with the real environment; integration code is responsible for calling Simulate for the real step using a_t.
	9.	Offline training of reference parameters

9.1 Abstract decision process for training

Separately from online operation, an abstract decision process is defined over D_site and action sets A_site(d), with simulator:

SimulateAbs(d, a) -> (d_next, r_abs, done_abs)

This process approximates or compresses the target environment or task distribution.

9.2 Reference parameter objectives

A training procedure (not specified here) produces B(d,a) and R_ref(d,a) such that:
	•	The induced reference strategy pi_ref(d) over A_site(d) provides reasonable performance on the abstract process.
	•	Base preferences B(d,a) encode prior preferences over actions at each site, which may be used to break symmetries and bias search.
	•	R_ref(d,a) are nonnegative masses such that for each site d, the normalized masses are a probability distribution over actions.

Search v0.0.1 does not constrain the training algorithm beyond these shape constraints.

9.3 Installation into the module

After training, the learned parameters are installed by setting the backing store for ReadBasePref and ReadRefMass appropriately.

During online operation, these parameters are treated as read-only; only U(d,a) is updated by the module.
	10.	Complexity, determinism, and constraints

10.1 Per-step complexity

With configuration constants N_roll, T_h, A_max, and assuming:
	•	SiteOf and EnvActions have bounded cost per call,
	•	Read* and WriteAdj have O(1) worst-case time independent of number of keys used so far,

the per-step complexity of Step_Search is bounded by:
	•	O(N_roll * T_h * A_max) environment simulations and parameter reads,
	•	O(N_roll * T_h) parameter writes,
	•	O(N_roll * T_h * A_max) evaluations of exponentials or similar.

These bounds do not depend on:
	•	t (environment step index),
	•	number of distinct states visited,
	•	number of distinct sites d in D_site,
	•	number of distinct (d,a) pairs that have nonzero parameters.

10.2 Determinism

Search v0.0.1 must be deterministic under fixed seeds:
	•	All uses of randomness (for action sampling in rollouts or at root) derive from seed_search and a fixed, explicit indexing scheme using (t, rollout index, depth index, possibly site index).
	•	There is no data-dependent reseeding.
	•	For a fixed configuration, parameter initialization, and environment behavior, Step_Search(s_t) is a deterministic function of the log up to time t and seed_search, given deterministic Simulate and Value/PolicyPrior.

10.3 Memory usage

The number of nonzero U(d,a) entries may grow over time as new sites and actions are visited. Implementations are allowed to:
	•	Store U(d,a) in a shared key-value structure,
	•	Or use a bounded-capacity cache that stores adjustments only for recently active sites.

If a bounded cache is used:
	•	Eviction must be deterministic given the sequence of operations.
	•	When U(d,a) is evicted, ReadAdj(d,a) returns 0.0 subsequently until updated again.
	•	This behavior effectively forgets local adjustments for old or infrequently used sites.

	11.	Diagnostics and monitoring

The module may expose diagnostics per step, including but not limited to:
	•	root_site_id (abstract identifier for d_root)
	•	|C_root| (number of candidate actions at root)
	•	for each a in C_root:
	•	b(d_root,a), u(d_root,a), u_plus(d_root,a)
	•	pi_ref(d_root,s_t)(a), pi_loc(d_root,s_t)(a), pi_dec(d_root,s_t)(a)
	•	rollout statistics:
	•	N_roll used,
	•	empirical distribution of trajectory lengths H_k,
	•	sample mean and variance of G_0 across rollouts,
	•	count of adjustment updates and clamping at u_min or u_max.

Diagnostics are not used by the core algorithm, but are recommended for tuning hyperparameters (eta_adj, N_roll, T_h, mixture weights).
	12.	Summary

Search v0.0.1 defines a self-contained module that:
	•	Maps concrete states to abstract decision sites,
	•	Uses a precomputed reference strategy and base preferences over actions at each site,
	•	Performs a bounded number of rollouts per environment step using a mixture of reference, local, and model policies,
	•	Updates local adjustment weights for visited site-action pairs according to outcome-based signals,
	•	Constructs a decision distribution from reference preferences, local adjustments, and optional model logits,
	•	Returns a concrete action with fixed per-step computational and memory budgets.

