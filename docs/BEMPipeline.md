BEM Self-Evolving Training Pipeline v0.0.1
	0.	Scope

This document defines a self-evolving training pipeline for BEM v0.0.1 that:
	1.	Trains domain-specialized BEM instances via self-play and synthetic experiments.
	2.	Distills their behavior into a unified main BEM.
	3.	Runs mixed-domain reinforcement learning with verifier-guided updates.
	4.	Uses adaptive curriculum and task scheduling to increase generalization stepwise.
	5.	Uses PoX and verification to prioritize structurally meaningful patches.

It is written in English ASCII and assumes the BEM core specification you already have.
	1.	Components

1.1 Main BEM

A single BEM instance B_main with:
	•	fixed N, K, W, expert budget S
	•	full verification stack (SAT, Hoare, CEGIS)
	•	PoX metrics enabled
	•	benchmark suite integrated into its task generator

1.2 Specialist BEMs

For each domain d in D = {math, code, general_reasoning, agentic, safety, etc}:
	•	a BEM instance B_d with:
	•	domain-specific configuration (state layout, experts, tasks)
	•	domain-specific task generator G_d
	•	domain-specific verifier configuration (e.g. SAT focus for code, bandit safety for some domains)

1.3 Domain Scheduler

A meta-controller S_domain that:
	•	maintains per-domain statistics:
	•	validation loss or regret L_d
	•	learning progress Delta_L_d
	•	uncertainty metric U_d (for example disagreement between hypotheses, or variance of performance)
	•	stability metric Stab_d (for example KL divergence drift, violation rate)
	•	outputs domain allocation probabilities:
	•	p_d_act, p_d_synth, p_d_verify
such that for each mode m in {act, synth, verify}
	•	sum over d of p_d_m = 1

1.4 Task Generator and Curriculum Engine

For each domain d:
	•	a family of difficulty levels k in K_d (for example length, parameter range, noise, composition)
	•	a generator G_d,k(theta) that creates an environment or task instance parameterized by theta
	•	a curriculum policy C_d that:
	•	chooses difficulty levels k for each domain
	•	updates k based on pass rate and learning progress

1.5 Verifiers and Evaluators

For each domain d:
	•	a verifier V_d that:
	•	checks correctness or safety of episodes (for example SAT, Hoare, execution tests, external verifier)
	•	an evaluator E_d that:
	•	assigns scalar rewards and auxiliary metrics (pass, partial credit, reasoning quality)

1.6 Patch and Verification Scheduler

A scheduler S_patch that:
	•	maintains a queue Q_patch of structural patches Delta
	•	estimates for each Delta:
	•	expected improvement gain G(Delta)
	•	verification cost C_verif(Delta)
	•	derives a priority score:
	•	prio(Delta) = G(Delta) / max(epsilon, C_verif(Delta))
	•	decides which patches to verify next, subject to verification budget

	2.	Data Structures

2.1 Domain Statistics

For each domain d:
	•	moving window of validation metrics:
	•	L_d(t), accuracy_d(t), regret_d(t)
	•	learning progress:
	•	Delta_L_d(t) = L_d(t - h) - L_d(t) for some horizon h
	•	uncertainty:
	•	U_d(t) (for example variance of performance, or entropy of hypothesis distribution restricted to d)
	•	stability:
	•	Stab_d(t) (for example average KL drift and violation rate)

2.2 Difficulty Statistics

For each domain d and difficulty level k:
	•	pass rate PR_d,k
	•	regret R_d,k
	•	number of episodes N_d,k

2.3 Episode Metadata

For each episode tau:
	•	domain d(tau)
	•	difficulty k(tau)
	•	reward trajectory r_t
	•	old policy descriptor pi_old (for example expert weight snapshot or routing metadata)
	•	current policy descriptor pi_new at update time
	•	verifier result V_d(tau) in {OK, FAIL}
	•	computed advantage A(tau) or equivalent learning signal

2.4 Patch Metadata

For each patch Delta:
	•	domains impacted Dom(Delta)
	•	estimated gain components:
	•	Delta_R (regret reduction)
	•	Delta_S (safety improvement)
	•	Delta_C (fast path cost reduction)
	•	Delta_I (information gain)
	•	composite PoX score:
	•	score(Delta) = w_R Delta_R + w_S Delta_S + w_C Delta_C + w_I Delta_I
	•	estimate of verification cost C_verif(Delta)
	•	priority prio(Delta) = score(Delta) / max(epsilon, C_verif(Delta))

	3.	Training Phases

3.1 Phase 0: Initialization
	•	Initialize each B_d with:
	•	minimal expert set implementing simple baselines for its domain
	•	simple tasks with low difficulty levels k in K_d
	•	Initialize B_main with:
	•	slightly larger expert capacity
	•	initial experts seeded from simple hand-coded or random circuits

3.2 Phase 1: Specialist Self-Play

Loop per domain d:
	•	use S_domain to pick a domain d and mode m in {act, synth}
	•	if m = act:
	•	run real or benchmark episodes for domain d using tasks from G_d,k
	•	if m = synth:
	•	run synthetic experiments or self-play based on templates and adversarial samplers for d
	•	for each episode tau:
	•	evaluate with E_d, verify with V_d if available
	•	update expert weights and routing of B_d using BEM bandit plus mirror-descent logic
	•	log tau and relevant metrics for the domain

Repeat until each domain d reaches its early target performance (for example baseline accuracy, or low regret on small difficulties).

3.3 Phase 2: Distillation into Main BEM

Construct a distillation dataset D_distill:
	•	for each domain d:
	•	select episodes tau with:
	•	verifier V_d(tau) = OK
	•	regret below domain-dependent threshold
	•	possibly good reasoning traces if available
	•	encode these episodes in a common format for B_main:
	•	unify observation and state layouts via adapters

Train B_main in supervised mode:
	•	for each tau in D_distill:
	•	reproduce expert selection decisions
	•	reproduce outputs
	•	optionally reproduce internal state transitions when they are observable

Optionally perform structural distillation:
	•	extract frequent expert patterns or CFG fragments from B_d logs
	•	construct templates T_k and expert macros
	•	install these as initial experts or CFG fragments in B_main, subject to verification checks

3.4 Phase 3: Mixed-Domain RL on Main BEM

Now B_main is the main optimization target.

Loop:
	•	use S_domain to choose a domain d and mode m in {act, synth, verify}
	•	if m = act:
	•	sample difficulty k from C_d
	•	generate task via G_d,k
	•	run episode tau using B_main
	•	if m = synth:
	•	sample synthetic task via G_d,k or adversarial generator Adv_d
	•	run self-play episode tau
	•	if m = verify:
	•	ask S_patch for the next patch Delta with highest prio(Delta)
	•	run verification for Delta using SAT, Hoare, and CEGIS
	•	if VC(Delta) holds, apply Delta and log PoX metrics

After each batch of episodes:
	•	perform weight updates on B_main using:
	•	bandit plus mirror-descent update
	•	group-relative normalization if multiple trajectories for the same observation are sampled
	•	filter episodes with off policy masking:
	•	discard episodes where:
	•	KL(pi_old, pi_new) > KL_threshold and A(tau) is negative
	•	periodically recompute domain statistics and update S_domain and C_d

3.5 Phase 4: Maintenance and Long-Run Self-Evolution

In long-run operation:
	•	maintain the specialist B_d to explore new capabilities or cover new domains
	•	periodically re-run a reduced distillation step to sync new specialist capabilities into B_main
	•	run mixed-domain RL and structural updates continuously
	•	adjust PoX weights (w_R, w_S, w_C, w_I) and difficulty ranges to reflect new priorities

	4.	Domain Scheduling Algorithm

At meta-step t_meta:

Inputs:
	•	for each domain d:
	•	learning progress Delta_L_d
	•	uncertainty U_d
	•	stability Stab_d
	•	recent PoX yield PoX_d (for example average score(Delta) originating from d)

Compute a scalar score S_d:

S_d = a1 * normalize(Delta_L_d positive part)
+ a2 * normalize(U_d)
+ a3 * normalize(PoX_d)
- a4 * normalize(Stab_d)

where normalize is any bounded scaling to comparable ranges.

Then for each mode m in {act, synth, verify}:
	•	define mode-specific weights w_m,d (for example favor synth in high uncertainty domains, favor verify when many patches exist)
	•	compute unnormalized logits:

logit_m,d = w_m,d * S_d
	•	derive probabilities by softmax:

p_d_m = exp(logit_m,d) / sum_d exp(logit_m,d)

The pipeline uses p_d_m to allocate the next batch of episodes and verification jobs.
	5.	Episode Selection and Off-Policy Masking

For each domain d and episode tau:
	•	compute:
	•	advantage A(tau) or analogous learning signal
	•	policy divergence D_KL(pi_old || pi_new) at update time

Define thresholds:
	•	KL_max_d
	•	A_min_d (often zero or a small negative number)

Masking rule:
	•	if D_KL(pi_old || pi_new) > KL_max_d
and A(tau) < A_min_d:
	•	exclude tau from weight updates

Otherwise include tau.

Optionally:
	•	group multiple trajectories tau_1,…,tau_G for the same context (same initial observation)
	•	compute group-relative normalized returns
	•	use normalized returns for advantage estimation, similar to group relative policy optimization.

	6.	Structural Patch Scheduling with PoX

For each patch Delta in Q_patch:
	•	estimate Delta_R, Delta_S, Delta_C, Delta_I from pilot runs or analytic approximations
	•	compute score(Delta):

score(Delta) = w_R * Delta_R + w_S * Delta_S + w_C * Delta_C + w_I * Delta_I
	•	estimate verification cost C_verif(Delta) from:
	•	CNF size
	•	expected SAT depth
	•	code region size

Compute priority:

prio(Delta) = score(Delta) / max(epsilon, C_verif(Delta))

Verification policy:
	•	at each verify step:
	•	pop the Delta with highest prio from Q_patch
	•	run verification for Delta
	•	if VC(Delta) holds:
	•	apply Delta to B_main
	•	log PoX metrics and update PoX_d for all affected domains
	•	else:
	•	reject Delta and log counterexamples if available

Adjust the global verification share alpha_verify dynamically:
	•	if many high-priority patches remain unverified:
	•	increase alpha_verify
	•	if backlog is small or verification yields are low:
	•	decrease alpha_verify

	7.	Curriculum and Difficulty Control

For each domain d and difficulty k:
	•	maintain pass rate PR_d,k and regret R_d,k

Define:
	•	target pass rate range [PR_min, PR_max]
	•	regret thresholds R_min, R_max

Difficulty update heuristics:
	•	if PR_d,k > PR_max and R_d,k < R_min:
	•	reduce sampling of difficulty k
	•	increase preference for higher difficulty k_plus
	•	if PR_d,k < PR_min or R_d,k > R_max:
	•	reduce sampling of higher difficulties
	•	increase sampling of k or easier k_minus
	•	keep the distribution over k roughly centered where the model is challenged but not failing completely.

Curriculum policy C_d uses these statistics to produce a distribution over difficulties for G_d,k.
	8.	Integration with BEM Time Scales

Fast path:
	•	unchanged: observation, ANN routing, expert selection and evaluation, minimal stats

Mid path:
	•	weight updates for expert weights and bandit statistics using accepted episodes (after masking)
	•	logging and basic template extraction

Slow path:
	•	execution of self-play and synthetic experiments according to domain and difficulty schedules
	•	generation of structural patches
	•	verification and superoptimization driven by S_patch
	•	periodic recomputation of PoX metrics and domain statistics

The scheduler controls how much of the available compute goes to each layer and domain by adjusting:
	•	p_d_act, p_d_synth, p_d_verify
	•	alpha_act, alpha_synth, alpha_verify inside each domain, if you further split by mode

	9.	Expected Behavior

Under this pipeline:
	1.	Specialist BEMs B_d become strong within their domains using self-play, synthetic experiments, and domain-specific verifiers.
	2.	Main BEM B_main bootstraps from them via distillation, gaining an initial broad expert set and routing structure.
	3.	Mixed-domain RL and structural updates refine B_main, with:
	•	domain scheduler focusing compute on domains and difficulties where learning progress and uncertainty are high
	•	off-policy masking removing harmful episodes
	•	PoX focusing verification and structural updates on the patches with best improvement per verification cost
	4.	Curriculum policies C_d gradually expand length, parameter ranges, noise, and composition complexity, increasing generalization stepwise until each level saturates.

This yields a self-evolving BEM training loop where the same compute is increasingly concentrated on the most informative tasks, episodes, and patches, while verification ensures that structural changes preserve or improve safety and correctness.
