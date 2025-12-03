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

BEM Image Snapshot Model v0.0.1
	0.	Scope

This document defines how to operate BEM as a long-running process on a mini PC and periodically capture the entire runtime into Docker images.

It specifies:
	1.	Image layout and responsibilities.
	2.	How to snapshot a running container into a tagged image.
	3.	How to restore and promote images.
	4.	Rotation and retention policies.
	5.	Basic compatibility and migration rules.

The goal is to treat each tagged image as a deployable stateful artifact that can be moved to other machines and resumed.
	1.	Runtime and Image Layout

1.1 Base image

Define a stable base image that contains the BEM engine and dependencies, but no evolving runtime state.

Example:
	•	Image name:
	•	registry.example.com/bem-core:0.0.1
	•	Contents:
	•	BEM engine binaries (bem-engine, bem-cli, auxiliary tools).
	•	SAT/Hoare/ANN libraries and runtime dependencies.
	•	Default configuration templates under /etc/bem.
	•	Minimal system tooling required for diagnostics.

This image is built via a normal Dockerfile and published to a registry:
	•	docker build -t registry.example.com/bem-core:0.0.1 .
	•	docker push registry.example.com/bem-core:0.0.1

1.2 Long-lived runtime container

On the mini PC, run a long-lived container derived from the base image:
	•	Container name:
	•	bem-runner
	•	Command:
	•	launches BEM scheduler, self-play, synthetic experiments, verification, and related components.

Example:
	•	docker run -d 
–name bem-runner 
registry.example.com/bem-core:0.0.1 
bem-engine –config /etc/bem/config.yaml

Inside the container:
	•	Evolving state directory:
	•	/var/lib/bem
	•	Logs:
	•	/var/log/bem
	•	Configuration:
	•	/etc/bem (fixed for a given run, but allowed to change over time if needed).

	2.	Snapshot as Docker Image

2.1 Snapshot definition

A snapshot is a Docker image created from the current filesystem of the running bem-runner container.

Tag format:
	•	registry.example.com/bem-brain:

Examples:
	•	registry.example.com/bem-brain:20251203-2300
	•	registry.example.com/bem-brain:20251203-2300-acc062-reg003

The snapshot contains:
	•	All layers from bem-core:0.0.1.
	•	An additional top layer that includes:
	•	/var/lib/bem (runtime state, experts, ANN indices, metrics).
	•	/etc/bem if it has been edited.
	•	Any other modifications inside the container since startup.

2.2 Snapshot command sequence

On the mini PC, to create and push a snapshot:
	•	SNAP_TAG=bem-brain:$(date -u +%Y%m%d-%H%M)
	•	docker commit bem-runner registry.example.com/${SNAP_TAG}
	•	docker push registry.example.com/${SNAP_TAG}

This produces a new image that can be pulled and executed as-is on another machine.

2.3 Labels and metadata

Attach metadata at commit time using image labels:
	•	docker commit 
–change “LABEL bem.version=0.0.3 bem.acc=0.62 bem.regret=-0.03” 
bem-runner registry.example.com/${SNAP_TAG}

This allows quick inspection of key metrics and engine version using docker inspect or registry tooling.
	3.	Snapshot Scheduling and Gating

3.1 Scheduling

A snapshot agent runs on the mini PC and triggers snapshot creation at a fixed cadence or after specific conditions.

Examples:
	•	Time-based:
	•	Every 6 hours via cron or systemd timer.
	•	Event-based:
	•	After a defined amount of synthetic experiments or episodes.

Time-based example using cron:
	•	0 */6 * * * /usr/local/bin/bem-image-snapshot.sh

3.2 Gating conditions

To avoid committing low-quality or unstable states, define gating conditions that must pass before a snapshot is taken and pushed.

Inputs:
	•	Metrics file inside the container (generated by BEM), for example:
	•	/var/lib/bem/metrics/summary.json
	•	This file may expose:
	•	safety_violations_last_window
	•	accuracy_delta
	•	regret_delta
	•	verification_backlog

Snapshot is allowed only if, for example:
	•	safety_violations_last_window == 0
	•	verification_backlog below a configured threshold
	•	accuracy_delta >= epsilon or regret_delta <= -epsilon

If conditions fail:
	•	Skip snapshot creation for this interval.
	•	Wait until the next scheduled evaluation.

3.3 Snapshot script outline

bem-image-snapshot.sh (conceptual):
	1.	Read metrics via docker exec bem-runner.
	2.	Evaluate gating logic.
	3.	If conditions pass:
	•	Compute tag and labels.
	•	docker commit bem-runner registry.example.com/${SNAP_TAG}
	•	docker push registry.example.com/${SNAP_TAG}
	4.	Restore and Deployment

4.1 Restore on another machine

On a different mini PC or server:
	1.	Pull a snapshot image:
	•	docker pull registry.example.com/bem-brain:20251203-2300
	2.	Run the container:
	•	docker run -d 
–name bem-runner 
registry.example.com/bem-brain:20251203-2300 
bem-engine –resume

The /var/lib/bem directory inside the image contains the runtime state needed by BEM to resume self-play, synthetic experiments, and verification.

4.2 Stable tags and promotion

To promote a particular snapshot to a stable version:
	1.	Tag a selected snapshot:
	•	docker tag 
registry.example.com/bem-brain:20251203-2300 
registry.example.com/bem-brain:code-v2-stable
	2.	Push the stable tag:
	•	docker push registry.example.com/bem-brain:code-v2-stable

Deployment systems (Kubernetes, docker-compose, CI/CD) can then refer to the stable tag:
	•	image: registry.example.com/bem-brain:code-v2-stable

This makes the stable snapshot the desired state for production or evaluation environments.
	5.	Engine Upgrades and Migration

5.1 New core image

When upgrading the underlying BEM engine:
	1.	Build and push a new base image:
	•	registry.example.com/bem-core:0.0.2
	2.	Snapshot migration is handled by extracting and transforming /var/lib/bem from an older snapshot into a new container based on bem-core:0.0.2.

5.2 Migration procedure (conceptual)
	1.	Start a source container from an old snapshot:
	•	docker run -d 
–name bem-migrate-src 
registry.example.com/bem-brain:20251203-2300 
sleep infinity
	2.	Archive state:
	•	docker exec bem-migrate-src tar czf /tmp/bem-state.tgz /var/lib/bem
	•	docker cp bem-migrate-src:/tmp/bem-state.tgz .
	3.	Start a destination container from the new core image:
	•	docker run -d 
–name bem-migrate-dst 
-v $(pwd)/bem-state.tgz:/tmp/bem-state.tgz 
registry.example.com/bem-core:0.0.2 
sleep infinity
	4.	Inside bem-migrate-dst:
	•	tar xzf /tmp/bem-state.tgz -C /
	•	bem-migrate –from 0.0.1 –to 0.0.2 /var/lib/bem
	5.	Commit and push the migrated snapshot:
	•	docker commit bem-migrate-dst 
registry.example.com/bem-brain:20251203-2300-core0.0.2
	•	docker push registry.example.com/bem-brain:20251203-2300-core0.0.2
	6.	Rotation and Retention

6.1 Retention policy

To limit storage growth in the registry and on the mini PC, define a retention policy over snapshot tags:

Examples:
	•	Keep:
	•	Last N daily snapshots.
	•	Last M weekly snapshots.
	•	All explicitly marked stable tags.
	•	Remove:
	•	Older non-stable snapshots outside the retention window.

This can be implemented via:
	•	Scheduled registry cleanup that filters tags by date and naming convention.
	•	Local docker image pruning on the mini PC using docker image rm based on filters.

6.2 Milestone tagging

Beyond stable tags, define milestone tags for significant states:
	•	registry.example.com/bem-brain:code-hard-v1
	•	registry.example.com/bem-brain:safety-v1-verified

Use these as references for long-term comparison and for spinning up specialized evaluation or training nodes.
	7.	Integration with CI/CD

7.1 CI usage

CI pipelines can:
	1.	Pull a specific snapshot image:
	•	image: registry.example.com/bem-brain:code-v2-stable
	2.	Run BEM Benchmark Suite inside that container:
	•	bem-cli bench run –config /etc/bem/benchmarks/code.yaml
	3.	Evaluate performance against thresholds.
	4.	Decide whether to:
	•	accept a new snapshot as stable.
	•	reject and keep using the previous stable tag.

7.2 CD usage

CD systems can treat snapshot images as the source of truth for deployed BEM instances:
	•	For a production environment:
	•	deploy bem-brain: as a stateful component.
	•	For evaluation clusters:
	•	use different tags (for example latest dev snapshot vs last promoted snapshot).

This makes the image tag the primary identifier of which BEM state is running where.
	8.	Security Considerations

8.1 Registry and access
	•	Use a private container registry.
	•	Restrict push access to the snapshot agent and CI/CD systems.
	•	Restrict pull access to authorized nodes.

8.2 Content

Snapshots may contain:
	•	BEM internal state.
	•	Synthetic tasks and traces.
	•	Potentially sensitive data if external inputs are processed inside the same container.

Mitigations:
	•	Where possible, segregate external raw inputs (for example mount them as volumes that are not committed into images).
	•	Keep /var/lib/bem focused on engine state and synthetic artifacts.
	•	Document clearly whether any externally supplied data is stored in committed layers.

	9.	Summary

This model defines BEM operation on a mini PC as:
	•	A long-running container derived from a stable core image.
	•	Periodic whole-container snapshots using docker commit, producing tagged images.
	•	Snapshots pushed to a registry, with gating conditions based on metrics and safety.
	•	Other machines can pull a snapshot image and resume BEM using the contained state.
	•	Retention, tagging, and migration policies keep the image set manageable and compatible.

The result is a stateful image workflow in which each Docker image tag represents a concrete, deployable BEM instance with a specific engine version and internal state.
