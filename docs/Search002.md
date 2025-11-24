Title
Search v0.0.2 – Fixed-Budget Abstract Game Search with Policy-Improvement Targets

Status
Draft, implementation-oriented, self-contained, constant-time per decision (episode-length independent)

Language
ASCII, English only
	0.	Scope, notation, configuration

0.1 Scope

Search v0.0.2 is a fixed-budget decision module that, at each environment decision step t,
	1.	Maps the concrete environment state s_t to an abstract decision site d_t.
	2.	Enumerates a finite set of candidate actions C_t = EnvActions(d_t, s_t).
	3.	Combines three sources of information:
(a) a reference strategy and base preference over actions at each site,
(b) a model policy prior and value function,
(c) local adjustments stored in a key–value structure.
	4.	Runs a bounded number of rollouts up to a fixed depth using a mixture of reference, local, and model policies.
	5.	Estimates action values and advantages from these rollouts.
	6.	Constructs policy-improvement targets using advantage-weighted and critic-regularized regression objectives, including a Gumbel-style completed Q operator at the root.
	7.	Updates local adjustments for visited site–action pairs based on the gap between current and target policies.
	8.	Returns a decision distribution pi_dec_t over concrete environment actions at the root and optionally a selected action a_t.

Key properties:
	1.	Per-step work is bounded by configuration constants (N_roll, T_h, A_max, etc.) and does not grow with total horizon, number of sites, or number of visited states.
	2.	The algorithm is deterministic given configuration, parameters, and a seed, assuming deterministic environment and model.
	3.	The module is defined in terms of its own state and update equations; it does not require naming specific external algorithms.

0.2 Notation

R           : set of real numbers
N           : set of nonnegative integers {0,1,2,…}
N_pos       : set of positive integers {1,2,…}

S           : environment state space (arbitrary, possibly large)
A           : environment action space (arbitrary, possibly large)
D_site      : abstract decision-site space

For random variables X, E[X] denotes expectation.

For a finite set C, |C| is its cardinality. For a distribution pi over C, pi[a] is the probability assigned to a in C.

Indicator of event E is 1{E} in {0,1}.

0.3 Configuration constants

A configuration C_Search for Search v0.0.2 consists of:

Discount and budget

gamma         in (0,1]     ; reward discount factor
N_roll        in N_pos      ; number of rollouts per environment step
T_h           in N_pos      ; maximum rollout horizon (depth, number of steps)
A_max         in N_pos      ; upper bound on number of actions at any site

Mixture and scoring weights

w_ref         >= 0          ; weight for reference base preference in decision logits
w_adj         >= 0          ; weight for local adjustment in decision logits
w_model       >= 0          ; weight for model logits in decision logits

alpha_ref     >= 0          ; weight for reference strategy in rollout policy
alpha_loc     >= 0          ; weight for local strategy in rollout policy
alpha_model   >= 0          ; weight for model policy in rollout policy

The rollout mixture must satisfy alpha_ref + alpha_loc + alpha_model > 0.

Local adjustment update

eta_adj       > 0           ; step size for adjustment updates
u_min         <= 0          ; lower clamp bound for adjustment weights
u_max         >= 0          ; upper clamp bound for adjustment weights

Numerical constants

epsilon_prob  in (0,1)      ; minimum probability scale in softmax denominators
eps_log       in (0,1)      ; minimum probability inside log to avoid log(0)

Root policy-improvement hyperparameters

beta_AWR      > 0           ; temperature for advantage-weighted regression target
beta_CRR      > 0           ; temperature for soft CRR weights
tau_CRR       in R          ; threshold for binary CRR weights
w_CRR_max     > 0           ; maximum CRR weight

beta_Q        > 0           ; scaling for completed Q transform
sigma_max     > 0           ; clipping magnitude for completed Q transform

Seed

seed_search   in N          ; base seed for all pseudo-random choices

All constants in C_Search remain fixed for the lifetime of a run.

0.4 External model interfaces

Search v0.0.2 assumes access to:
	1.	Environment simulator:
Simulate(s, a) -> (s_next, r_env, done_flag)
where
s, s_next in S,
a in A,
r_env in R,
done_flag in {True, False}.
Simulate must be deterministic given its arguments and the environment’s seed.
	2.	Abstract site mapping:
SiteOf(s) -> d in D_site
EnvActions(d, s) -> C subset of A
with the constraints:
(a) |C| >= 1 for all nonterminal states used for decisions,
(b) |C| <= A_max.
	3.	Model policy and value:
PolicyPrior(d, s) -> distribution over EnvActions(d,s)
ValueModel(s)     -> scalar in R
PolicyPrior(d,s)[a] is required to be nonnegative with sum 1 over C = EnvActions(d,s). ValueModel(s) is any scalar evaluation of state s.

The source of PolicyPrior and ValueModel is not specified (they may come from any model, such as Sera v0.0.2). Search v0.0.2 only constrains how these interfaces are used.

0.5 Parameter storage for reference and adjustments

Search maintains three conceptual parameter families:

B(d,a)      in R         base preference score
R_ref(d,a)  in [0, +infty) reference mass
U(d,a)      in R         local adjustment

for each site d in D_site and each action a in a (possibly large) action set A_site(d). At runtime, only the subset C = EnvActions(d,s) is relevant.

The implementation provides the following O(1)-time primitives:

ReadBasePref(d, a) -> float      returns B(d,a) or 0.0 if unset
ReadRefMass(d, a)  -> float      returns R_ref(d,a) or 0.0 if unset
ReadAdj(d, a)      -> float      returns U(d,a) or 0.0 if unset
WriteAdj(d, a, u_new)            sets U(d,a) = u_new

For any (d,a) never written, Read* returns 0.0. WriteAdj must run in bounded time independent of the number of keys touched so far. An implementation may use bounded caches; see section 11.3.
	1.	Reference strategy and model logit

1.1 Reference strategy pi_ref

Given site d and concrete state s, let C = EnvActions(d,s).

For each a in C:

m_a = max( ReadRefMass(d,a), 0.0 )

Let M = sum_{a in C} m_a.

The reference strategy is:

If M > 0:
pi_ref(d,s)(a) = m_a / M
Else:
pi_ref(d,s)(a) = 1 / |C|  (uniform over C)

1.2 Model logit l_model

Let PolicyPrior(d,s) be a distribution over C with probabilities PolicyPrior(d,s)[a] >= 0 and sum 1.

For a in C:

p_model_raw = PolicyPrior(d,s)[a]
p_model     = max(p_model_raw, eps_log)
l_model(d,s,a) = log(p_model)

The model policy lo gits are thus log probabilities clipped away from minus infinity.
	2.	Per-site scoring, local strategy, and decision distribution

2.1 Base preference and adjustment

Base preference:

b(d,a) = ReadBasePref(d,a)

Local adjustment:

u(d,a) = ReadAdj(d,a)

For scoring, only the nonnegative part is used:

u_plus(d,a) = max(u(d,a), 0.0)

2.2 Combined pre-activation z(d,s,a)

At site d, state s, for action a in C = EnvActions(d,s):

z(d,s,a) =
w_ref   * b(d,a)
	•	w_adj   * u_plus(d,a)
	•	w_model * l_model(d,s,a)

2.3 Local softmax policy pi_loc

Define:

z_max = max_{a in C} z(d,s,a)
e_loc(a) = exp(z(d,s,a) - z_max)
Z_loc    = sum_{a in C} e_loc(a)

If Z_loc <= 0, set pi_loc(d,s)(a) = 1/|C| for all a. Otherwise:

pi_loc(d,s)(a) = e_loc(a) / Z_loc

2.4 Decision distribution pi_dec at root

At the root site (d_root, s_root), we define the decision distribution using the same z(d,s,a) and a numerically safe denominator:

z_max_root = max_{a in C_root} z(d_root,s_root,a)
e_dec(a)   = exp(z(d_root,s_root,a) - z_max_root)
Z_dec      = sum_{a in C_root} e_dec(a)

Denominator:

Z_safe = max(Z_dec, epsilon_prob * |C_root|)

Decision distribution:

pi_dec(d_root,s_root)(a) = e_dec(a) / Z_safe
	3.	Rollout policy pi_roll

3.1 Components

At a generic site (d,s), we define three policies:
	1.	Reference policy:
pi_ref(d,s)(a) from section 1.1.
	2.	Local policy:
pi_loc(d,s)(a) from section 2.3.
	3.	Model policy:
pi_model(d,s)(a) = PolicyPrior(d,s)[a] restricted and renormalized over C.
Since PolicyPrior(d,s) is already defined over EnvActions(d,s), only a possible renormalization for numerical reasons is needed.

3.2 Mixture rollout policy

Let C = EnvActions(d,s). Use configuration weights alpha_ref, alpha_loc, alpha_model.

For each a in C:

w_roll(a) =
alpha_ref   * pi_ref(d,s)(a)
	•	alpha_loc   * pi_loc(d,s)(a)
	•	alpha_model * pi_model(d,s)(a)

Let W_total = sum_{a in C} w_roll(a).

If W_total <= 0, set pi_roll(d,s)(a) = 1/|C|. Otherwise:

pi_roll(d,s)(a) = w_roll(a) / W_total
	4.	Rollout trajectories and value estimates

4.1 Single rollout structure

At environment step t with current concrete state s_t, Search performs N_roll independent rollouts indexed by k in {1,…,N_roll}. All randomness is derived from seed_search and indexes (t, k, h) as follows:
	1.	Initialize:
s_0 = s_t
h   = 0
	2.	While h < T_h:
a. d_h = SiteOf(s_h)
b. C_h = EnvActions(d_h, s_h)
If C_h is empty, break.
c. Compute pi_roll(d_h, s_h) from section 3.
d. Sample a_h from pi_roll(d_h, s_h) using a deterministic pseudo-random generator seeded with (seed_search, t, k, h).
e. Call Simulate(s_h, a_h) -> (s_{h+1}, r_h, done_h).
f. Record the tuple (s_h, d_h, C_h, a_h, r_h).
g. If done_h is True, set h_end = h and break.
Otherwise, increment h and continue.

If the loop ends because h reached T_h - 1 without termination, set h_end = last visited index. Define rollout length H_k = h_end.

4.2 Terminal bootstrap value

For rollout k, at final state s_{H_k+1} (if last step simulated) or s_{H_k} (if no simulation was performed on last iteration), define bootstrap value:

G_terminal_k = ValueModel(s_terminal_k)

where s_terminal_k is chosen as the last concrete state in the recorded trajectory (precise choice is implementation-dependent but must be deterministic).

4.3 Backward discounted returns

For rollout k, define G_h^{(k)} for h = H_k,…,0:

If r_{H_k} exists:

G_{H_k}^{(k)} = r_{H_k} + gamma * G_terminal_k

Else:

G_{H_k}^{(k)} = G_terminal_k

For h = H_k-1 down to 0:

G_h^{(k)} = r_h + gamma * G_{h+1}^{(k)}

G_0^{(k)} is the discounted return from the root of rollout k.

4.4 Action-value estimates at root

Let (d_root, s_root) be the site and state at rollout depth h=0 for all rollouts (since they start from s_t).

For each root action a in C_root = EnvActions(d_root, s_root), collect all indices k where the rollout’s first action is a; that is, a_0^{(k)} = a.

If N_visits(a) is the number of such rollouts, define:

If N_visits(a) > 0:

Q_hat(a) = (1 / N_visits(a)) * sum_{k: a_0^{(k)} = a} G_0^{(k)}

Else:

Q_hat(a) is undefined and will be filled by completed Q (section 5.3).

The empirical root value is:

V_root = (1 / N_roll) * sum_{k=1..N_roll} G_0^{(k)}

V_root serves as a baseline for advantages and as the default value for unvisited actions.

4.5 Baseline and advantages

For the root site (d_root, s_root):

Baseline:

V_base_root = V_root

Advantage at root for any action a with Q_hat(a) defined:

A_root(a) = Q_hat(a) - V_base_root

If Q_hat(a) is not defined (no visits), set A_root(a) = 0 before Gumbel completion (the completion will lift its Q).
	5.	Policy-improvement targets at root

5.1 Behavior distribution at root

Let mu_root(a) denote the effective behavior distribution at the root, defined as the empirical fraction of rollouts that selected action a at h=0:

N_visits(a) as above,
mu_root(a) = N_visits(a) / N_roll.

If some actions have mu_root(a) = 0 (never sampled), mu_root is still a proper distribution, but those actions appear only through the model and completion.

5.2 Advantage-weighted target (AWR style)

Given advantages A_root(a), define a temperature beta_AWR > 0.

Intermediate unnormalized weights:

w_AWR(a) = mu_root(a) * exp( A_root(a) / beta_AWR )

To avoid degenerate zero weights, if sum_a w_AWR(a) = 0, set all w_AWR(a) = 1.

AWR target policy:

pi_AWR(a) = w_AWR(a) / sum_{b in C_root} w_AWR(b)

5.3 Completed Q values (Gumbel-style operator)

Define completed Q for each root action a:

If Q_hat(a) is defined:

Q_comp(a) = Q_hat(a)

Else:

Q_comp(a) = V_root

Define a monotone transform sigma_Q with temperature beta_Q > 0 and clipping sigma_max > 0:

sigma_Q(Q_comp(a)) = clip( Q_comp(a) / beta_Q, -sigma_max, sigma_max )

5.4 Gumbel-style improved policy at root

Using current decision distribution pi_dec(d_root,s_root) from section 2.4, define:

w_G(a) = pi_dec(d_root,s_root)(a) * exp( sigma_Q(Q_comp(a)) )

If sum_a w_G(a) = 0, set w_G(a) = 1 for all a.

Gumbel-style target policy:

pi_G(a) = w_G(a) / sum_{b in C_root} w_G(b)

5.5 CRR-style weights at root

Define CRR weights based on advantage A_root(a):

Soft CRR:

w_CRR_soft(a) = min( exp(A_root(a) / beta_CRR), w_CRR_max )

Binary CRR:

w_CRR_bin(a) = 1 if A_root(a) > tau_CRR, else 0

A combined CRR weight can be defined as:

w_CRR(a) = max( w_CRR_soft(a), w_CRR_bin(a) )

5.6 Combined target policy pi_tar at root

Define three nonnegative scalars alpha_AWR_root, alpha_G_root, alpha_CRR_root (they may be absorbed into normalization; the configuration may control them, but this spec treats them as fixed hyperparameters or 1).

Define preliminary weights:

w_tar(a) =
alpha_AWR_root * pi_AWR(a)
	•	alpha_G_root   * pi_G(a)
	•	alpha_CRR_root * w_CRR(a) * mu_root(a)

Normalize:

If sum_a w_tar(a) = 0, set w_tar(a) = 1 for all a.

pi_tar(a) = w_tar(a) / sum_{b in C_root} w_tar(b)

pi_tar is the root target policy used to update local adjustments and, if desired, to train upstream models.
	6.	Local adjustment updates

6.1 KL-style adjustment update at root

At the root site, we want pi_dec to move toward pi_tar under a constrained local offset. Recall:

pi_dec(d_root,s_root)(a) is induced by z(d_root,s_root,a) which includes w_adj * u_plus(d_root,a).

We treat U(d,a) as a local logit adjustment and update it proportional to the log-ratio between target and current decision distributions.

For each a in C_root:

pi_dec_a = max( pi_dec(d_root,s_root)(a), epsilon_prob / |C_root| )
pi_tar_a = max( pi_tar(a), epsilon_prob / |C_root| )

delta_log = log(pi_tar_a) - log(pi_dec_a)

Read:

u_old = ReadAdj(d_root, a)

Propose:

u_raw = u_old + eta_adj * delta_log

Clamp:

u_new = min( max(u_raw, u_min), u_max )

WriteAdj(d_root, a, u_new)

This update tends to decrease the KL divergence between pi_tar and the distribution induced by reference, model, and adjusted logits.

6.2 Non-root updates

For non-root sites (d_h, s_h) at rollout depth h > 0, Search v0.0.2 permits a simpler advantage-based update similar to v0.0.1.

For each visited step h in rollout k, with site d_h^{(k)}, action a_h^{(k)}, and return G_h^{(k)}, define a local baseline V_base(d_h^{(k)}) as the average G_h^{(k)} over visits to that site. Let A_h^{(k)} = G_h^{(k)} - V_base(d_h^{(k)}).

Then:

u_old = ReadAdj(d_h^{(k)}, a_h^{(k)})
u_raw = u_old + eta_adj * A_h^{(k)}
u_new = min( max(u_raw, u_min), u_max )
WriteAdj(d_h^{(k)}, a_h^{(k)}, u_new)

This ensures that frequently visited non-root site–action pairs receive signals, while the full KL-style policy-improvement is concentrated at the root where an external controller typically queries actions.
	7.	Per-step algorithm Step_Search

7.1 Inputs

At real environment step t:
	•	Current state s_t in S.
	•	Access to Simulate, SiteOf, EnvActions, PolicyPrior, ValueModel.
	•	Parameter storage via ReadBasePref, ReadRefMass, ReadAdj, WriteAdj.
	•	Configuration C_Search, including seed_search.

7.2 Step_Search(s_t)
	1.	Root identification
d_root = SiteOf(s_t)
C_root = EnvActions(d_root, s_t)
	2.	Rollout phase
For k in {1,…,N_roll}:
	•	Perform rollout k starting from s_t according to section 4.1 using pi_roll.
	•	Compute G_h^{(k)} values via backward propagation (section 4.3).
	3.	Root value and Q estimates
	•	From rollouts, compute V_root and Q_hat(a) for each a in C_root as in section 4.4.
	•	Compute A_root(a) and Q_comp(a) as in sections 4.5 and 5.3.
	4.	Behavior and current decision distribution
	•	Compute mu_root(a) from rollout frequencies as in section 5.1.
	•	Compute pi_dec(d_root,s_t)(a) from section 2.4.
	5.	Target policy at root
	•	Construct pi_AWR(a), pi_G(a), w_CRR(a), and combined pi_tar(a) as in sections 5.2–5.6.
	6.	Update root adjustments
	•	For each a in C_root, update U(d_root,a) via KL-style update (section 6.1).
	7.	Optional non-root updates
	•	For each (d_h^{(k)}, a_h^{(k)}) visited during rollouts, perform advantage-based updates as in section 6.2.
	8.	Compute final decision distribution and select action
	•	Recompute b(d_root,a), u(d_root,a), u_plus, l_model(d_root,s_t,a), z(d_root,s_t,a), and pi_dec(d_root,s_t)(a) using updated U.
	•	Choose action a_t by either:
Option Greedy:
a_t = argmax_{a in C_root} pi_dec(d_root,s_t)(a)
Option Sampling:
a_t sampled from pi_dec(d_root,s_t) using deterministic pseudo-randomness from seed_search, t, and a local counter.
	9.	Output
Return:
	•	selected action a_t
	•	decision distribution pi_dec(d_root,s_t)
	•	optional diagnostics (section 10).
	10.	Offline training of reference parameters

8.1 Abstract training process

Separate from online operation, a training process defines an abstract decision process over sites and actions:

SimulateAbs(d, a) -> (d_next, r_abs, done_abs)

with d_next in D_site, r_abs in R, done_abs in {True, False}.

8.2 Objectives for B(d,a) and R_ref(d,a)

The goal of offline training is to set B(d,a) and R_ref(d,a) such that:
	1.	The induced reference strategy pi_ref(d) over A_site(d) (the full action set at site d) has acceptable performance on the abstract process.
	2.	Base preferences B(d,a) encode priors or structure at each site, breaking symmetries and biasing rollouts toward promising actions.
	3.	Reference masses R_ref(d,a) are nonnegative and form reasonable base distributions when normalized.

A generic objective is:

Maximize J(B, R_ref) =
E_{d ~ D_0} [ E_{trajectory under pi_ref} [ sum_{h} gamma^h r_abs(h) ] ]

subject to R_ref(d,a) >= 0.

Any policy gradient or imitation-based algorithm that improves J can be used to fit B, R_ref. Once trained, these parameters are installed into the runtime storage used by ReadBasePref and ReadRefMass.

8.3 Interaction with online adjustments

Online adjustments U(d,a) adapt the search behavior to the actual environment, rolling horizon, and ValueModel. Offline-trained B and R_ref provide a prior; U(d,a) refines it in response to task-specific rollouts and policy-improvement targets.
	9.	Diagnostics and monitoring

Search v0.0.2 may expose diagnostics at each step t:
	1.	Root-level:
	•	root_site_id: identifier for d_root
	•	C_root_size: |C_root|
	•	For each a in C_root:
	•	b(d_root,a), u(d_root,a), u_plus(d_root,a)
	•	pi_ref(d_root,s_t)(a)
	•	pi_loc(d_root,s_t)(a)
	•	pi_dec(d_root,s_t)(a)
	•	mu_root(a), Q_hat(a), Q_comp(a), A_root(a)
	•	pi_AWR(a), pi_G(a), pi_tar(a)
	•	V_root (empirical root value)
	2.	Rollout-level:
	•	N_roll used
	•	distribution of rollout lengths H_k
	•	sample mean and variance of G_0^{(k)}
	•	count of root and non-root updates applied
	•	count of clamping events at u_min and u_max

Diagnostics are not part of the decision logic but are required for tuning N_roll, T_h, alpha_, w_, and other hyperparameters.
	10.	Complexity, memory, and determinism

10.1 Per-step complexity

Assume:
	•	SiteOf and EnvActions have O(1) cost.
	•	PolicyPrior and ValueModel have bounded cost per call independent of total state-space size.
	•	ReadBasePref, ReadRefMass, ReadAdj, WriteAdj are O(1) operations independent of the total number of keys used so far.

Then per environment step t:
	•	At most N_roll * T_h calls to SiteOf and EnvActions.
	•	At most N_roll * T_h calls to Simulate.
	•	At most N_roll * T_h evaluations of pi_roll and log/exp for sampling.
	•	At most N_roll * T_h ReadAdj calls and N_roll * T_h WriteAdj calls for non-root updates, plus |C_root| ReadAdj and WriteAdj calls at the root.

Thus total cost is O(N_roll * T_h * A_max) operations per step, independent of t, total number of visited states, or number of distinct sites.

10.2 Memory usage

Search stores parameters:
	•	Reference base preferences B(d,a) and masses R_ref(d,a): these may be large but are treated as read-only and may reside in compressed storage.
	•	Local adjustments U(d,a): these evolve online.

Implementations may use:
	•	An unbounded key–value store with O(1) amortized operations.
	•	A bounded-capacity cache:
If a bounded cache is used,
	•	eviction must be deterministic given past operations,
	•	when (d,a) is evicted, ReadAdj(d,a) resets to 0.0,
	•	the algorithm remains correct as an approximation; evicted entries effectively forget their local adjustments.

10.3 Determinism

Search v0.0.2 must be deterministic given:
	•	Configuration C_Search, including seed_search.
	•	Initial parameter storage for B, R_ref, U.
	•	Deterministic implementations of Simulate, SiteOf, EnvActions, PolicyPrior, and ValueModel.

All randomness for sampling actions in rollouts and at the root must be derived from seed_search and a fixed indexing scheme (t, rollout index k, depth index h, and any required counters). No data-dependent reseeding is allowed.
	11.	Summary

Search v0.0.2 is a fixed-budget, abstract game-search module that:
	1.	Operates at the level of abstract decision sites and candidate actions induced from concrete states.
	2.	Combines reference priors, model policy and value, and local adjustments to form a decision distribution at each site.
	3.	Executes a bounded number of rollouts per step with a mixture of reference, local, and model policies.
	4.	Uses rollout returns to estimate root action-values, advantages, and completed Q values.
	5.	Constructs policy-improvement targets using advantage-weighted regression, critic-regularized regression, and a Gumbel-style completed Q operator.
	6.	Updates local adjustment parameters via KL-style log-ratio updates at the root and advantage-based updates at non-root sites.
	7.	Maintains O(1) per-step complexity with respect to episode length and number of states.

