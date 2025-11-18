Strategic Command-and-Control Doctrine
for National SIGINT and Cyber Defense (v0.2, Tightened)
	0.	Scope, Model, and Terminology

0.1 Scope

This document defines a defensive strategic doctrine for a national-level SIGINT and cyber defense organization.

It specifies:
	•	The protected decision loop (ping / pull / push).
	•	Asset tiers and protection classes.
	•	Structural metrics (non-asymmetry metrics).
	•	An operational cycle for measurement and improvement.
	•	Governance requirements.

The doctrine is vendor-neutral and technology-agnostic. It is intended only for defense and must be interpreted under applicable law.

0.2 System Model

The system consists of:
	•	Domains: coherent control scopes (identity, routing, cloud, SIGINT, etc.).
	•	Control planes: mechanisms by which each domain is configured and governed.
	•	Telemetry: logs, metrics, traces, and derived alerts.
	•	Operational entities: SOC, SIGINT center, network operations, cloud operations, identity operations, governance.

0.3 Terminology and Notation

Keywords:
	•	MUST, MUST NOT, REQUIRED: absolute requirements.
	•	SHOULD, SHOULD NOT: recommended behavior.
	•	MAY: optional behavior.

Notation:
	•	D: domain.
	•	c: privileged credential.
	•	S: reference attack scenario.
	•	T0, T1: Tier-Zero, Tier-One.
	•	R(c): revocation ratio.
	•	OQ(S): observability quotient.
	•	OAI: orphan alert index.
	•	ASI: analyst saturation index.

	1.	Decision Loop

1.1 Loop Definition

The decision loop is a 3-phase cycle for each domain D:

Phase ping(D):
	•	Self-verification of control planes and prerequisites within D.
	•	Examples: key status, config integrity, path availability.

Phase pull(D):
	•	Acquisition and validation of facts relevant to D.
	•	Examples: incident reports, telemetry summaries, SIGINT products, external advisories.

Phase push(D):
	•	Application of decisions within D.
	•	Examples: revocation, config changes, routing updates, access control, public or inter-agency notifications.

1.2 Loop Integrity Invariant

Invariant L1 (Loop Continuity):

For each critical domain D_critical, the following MUST hold:
	•	There exists at least one authenticated path that supports ping(D_critical) -> pull(D_critical) -> push(D_critical) with bounded latency T_max(D_critical).
	•	T_max(D_critical) is defined per domain and MUST be less than the adversary’s typical dwell time in that domain.

If L1 is violated for any D_critical for longer than a predefined duration T_violation(D_critical), a Tier-Zero incident MUST be declared.

1.3 Domains

Examples of domains:
	•	D_id: identity and access management (IAM).
	•	D_net: network perimeter and routing.
	•	D_cloud: cloud and SaaS tenancy.
	•	D_sig: SIGINT and cyber defense analysis.
	•	D_critical-infra: energy, transport, core public services.

Each domain MUST:
	•	Have a documented owner.
	•	Have a defined mapping to the global loop.
	•	Have explicit T_max(D) and T_violation(D) thresholds.

	2.	Asset Tiers

2.1 Tier-Zero (T0)

Definition:

An asset A is Tier-Zero if compromise or unavailability of A can:
	•	Break L1 for any D_critical, or
	•	Allow an attacker to impersonate the state or override its control-plane decisions at scale.

Examples:
	•	Root CAs and HSMs for state trust anchors.
	•	Global administrators for D_id and D_cloud.
	•	Core BGP/peering routers and authoritative DNS for key state domains.
	•	Primary SIGINT and SOC correlation/orchestration platforms for D_sig.

Requirements:
	•	T0 assets MUST:
	•	Use hardware-backed key protection where applicable.
	•	Enforce multi-party approval for destructive or global actions.
	•	Be monitored by independent channels (for example, out-of-band).
	•	Have documented backup and recovery procedures tested at least annually.
	•	Be enumerated in a maintained T0 inventory.

2.2 Tier-One (T1)

Definition:

An asset B is Tier-One if:
	•	Its compromise is serious but recoverable without permanently breaking L1, provided T0 remains intact.

Examples:
	•	Regional IAM components.
	•	Non-root CAs.
	•	Regional SOC infrastructure.
	•	Major but non-root application control planes.

Requirements:
	•	T1 assets SHOULD:
	•	Implement strong authentication, logging, and backup.
	•	Have documented incident response and failover plans.
	•	Be included in impact analysis for all T0 changes.

2.3 Lower Tiers (T2+)

Assets that do not affect L1 even under compromise are Tier-Two or below.
	•	T2 assets SHOULD be protected proportionally.
	•	T2+ assets MUST NOT consume T0-level resources or governance attention unless explicitly justified.

	3.	Axioms

Axiom A1 (Time as a Security Parameter):

Security is evaluated as a function of:
	•	P_success: probability that the attacker reaches objective.
	•	T_defense: time required to detect and contain.
	•	T_attack: time for the attacker to achieve persistence or irreversible changes.

For a domain to be structurally acceptable:
	•	For critical objectives: T_defense < T_attack is REQUIRED.

Axiom A2 (Loop Priority):

If a conflict arises between:
	•	Preserving arbitrary data or non-critical services, and
	•	Preserving L1 for D_critical,

then preserving L1 takes precedence.

Axiom A3 (Protocol Authority):

The effective authority over any domain D is determined by the entity whose messages are:
	•	Accepted as authoritative by the control planes of D, and
	•	Bound to keys and credentials that the system treats as legitimate.

Therefore:
	•	Key management and protocol behavior for control planes are treated as constitutional-level infrastructure.

	4.	Non-Asymmetry Metrics

The goal of non-asymmetry evaluation is to eliminate configurations where the defender is structurally forced to lose on time or cost, independent of skill or effort.

4.1 Revocation Ratio R(c)

4.1.1 Definitions

For a credential c:

RevocationLatency(c):
Elapsed time between:
	•	t_decision: timestamp of formal decision to revoke c, and
	•	t_effective: timestamp when c can no longer be used successfully anywhere in its domain.

CredentialRefreshPeriod(c):
Maximum time window in which an attacker with c can:
	•	Obtain new, functionally equivalent credentials, or
	•	Refresh existing tokens or sessions, without additional compromise or detection.

Define:
R(c) = RevocationLatency(c) / CredentialRefreshPeriod(c)

4.1.2 Targets
	•	For all T0 credentials c: R(c) MUST be <= 0.5.
	•	For all T1 credentials c: R(c) SHOULD be <= 1.0.
	•	Any credential with R(c) > 1.0 is structurally favorable to the attacker.

4.1.3 Measurement Procedure

(1) Inventory:
- Enumerate all T0 and T1 credentials.
(2) Drill:
- Select sample credentials in each class.
- Trigger revocation using normal processes.
- Log t_decision and monitor until t_effective at all relevant services.
(3) Measurement:
- Compute RevocationLatency(c) = t_effective - t_decision.
- Determine CredentialRefreshPeriod(c) from:
- Token lifetimes.
- Observed or configured refresh behavior.
- Design of identity and access APIs.
(4) Compute R(c) and record.
(5) Identify worst-case R(c) per class and prioritize remediation there.

4.2 Observability Quotient OQ(S)

4.2.1 Definitions

For a reference adversary scenario S:

TotalSteps(S):
Number of discrete attacker steps from initial foothold to objective, at defined granularity.

VisibleSteps(S):
Subset of TotalSteps(S) for which:
	•	There exists at least one event or signal ingested into the central detection platform, and
	•	The event has enough context (fields, timestamps, identifiers) to support detection or investigation.

Define:
OQ(S) = VisibleSteps(S) / TotalSteps(S)

4.2.2 Targets
	•	For any scenario S that impacts T0: OQ(S) MUST be >= 0.9.
	•	For representative organization-wide scenarios: OQ(S) SHOULD be >= 0.7.
	•	OQ(S) < 0.5 indicates a severe visibility deficit.

4.2.3 Measurement Procedure

(1) Define scenarios S1..Sn including:
- Email-driven initial access.
- VPN or SSO credential misuse.
- Cloud control-plane token theft.
- CI/CD or supply chain compromise.
(2) Execute each scenario in lab or tightly controlled environment.
(3) Capture all events produced across:
- Endpoint,
- Network,
- Identity,
- Cloud,
- Application logs.
(4) Count TotalSteps(Si) and VisibleSteps(Si) based on what reaches the central platform.
(5) Compute OQ(Si); document blind spots and map them to missing sensors or ingestion gaps.

4.3 Orphan Alert Index OAI

4.3.1 Definitions

TelemetryDomain d:
Coherent class of telemetry, such as:
	•	Email and anti-abuse logs.
	•	Cloud audit logs.
	•	Identity sign-in and token logs.
	•	VPN and firewall logs.
	•	Endpoint security logs.
	•	SIGINT-derived feeds.

Owner(d):
Team or role with:
	•	Explicit responsibility to monitor and triage d, and
	•	Authority and capability to act (access to tools, on-call, processes).

OrphanDomain:
A TelemetryDomain d for which:
	•	Owner(d) is not defined, or
	•	Owner(d) is nominal but has no practical ability to act.

Define:
	•	OAI_T0: count of OrphanDomains that contain any telemetry related to T0 assets or D_critical.
	•	OAI_total: total count of OrphanDomains.

4.3.2 Targets
	•	OAI_T0 MUST be 0.
	•	OAI_total SHOULD be <= 3.
	•	OAI_total > 3 indicates structural responsibility fragmentation.

4.3.3 Measurement Procedure

(1) Enumerate all TelemetryDomains d used by SIGINT, SOC, and operations.
(2) Build a RACI matrix:
- R (Responsible),
- A (Accountable),
- C (Consulted),
- I (Informed).
(3) Mark OrphanDomains where:
- There is no R and A, or
- R/A exist only on paper but have no access or on-call.
(4) Compute OAI_T0 and OAI_total.
(5) Address OrphanDomains by:
- Assigning or consolidating Owners,
- Adjusting mandates and access.

4.4 Analyst Saturation Index ASI

4.4.1 Definitions

For an operations center:

QueueLength(t):
Number of open actionable alerts at time t.

Saturation point N:
Minimum QueueLength such that, for a sustained interval, measurable degradation appears in:
	•	True positive detection rate during controlled exercises, or
	•	Time-to-handle for seeded high-priority alerts.

Define:
ASI = N

4.4.2 Targets
	•	Normal operations SHOULD maintain QueueLength(t) < ASI for the majority of time.
	•	During surge or crisis, automated aggregation and prioritization MUST be used to:
	•	Keep effective QueueLength for operators near or below ASI for T0/T1-related alerts.

4.4.3 Measurement Procedure

(1) Instrument SOC tooling to track QueueLength(t) and outcomes.
(2) Conduct controlled exercises:
- Inject benign noise to raise QueueLength systematically.
- Simultaneously seed known true positives.
(3) Observe threshold N where:
- Missed detections increase, or
- Handling times for critical alerts grow beyond acceptable limits.
(4) Set ASI = N and:
- Use it to design staffing and shifts.
- Limit expansion of raw alert rules; favor correlation and suppression of low-value noise.

4.5 Defensive Economic Profile (DEP)

4.5.1 Definitions

For an incident type I:
	•	TTD(I): mean time to detect.
	•	TTC(I): mean time to contain.
	•	MTTR(I): mean time to recover.
	•	CPA(I): mean cost per alert (analyst time, compute, tooling).

4.5.2 Objectives
	•	For T0/T1-relevant incident types:
	•	Adjust detection and automation so that the marginal cost to reduce TTD(I) and TTC(I) is lower than expected loss from delayed or missed cases.
	•	DEP is used to:
	•	Identify high-cost, low-value alerts for removal or redesign.
	•	Justify investment in sensors, correlation, and runbooks.

	5.	Operational Cycle

5.1 Overview

The doctrine MUST be executed as a recurring cycle with the following stages:

Stage 1: Threat and Attribution Baseline
Stage 2: Metric Measurement
Stage 3: Design and Governance Adjustments
Stage 4: Exercise and Validation
Stage 5: Policy and Communication

5.2 Stage 1: Threat and Attribution Baseline

The organization MUST:
	•	Maintain up-to-date threat actor and TTP profiles relevant to its remit.
	•	Map threat actors to domains D and reference scenarios S.
	•	Use consistent frameworks (for example, clustering by infrastructure, tooling, and victimology) for attribution.
	•	Avoid premature motive analysis before technical evidence is stabilized.

5.3 Stage 2: Metric Measurement

On a defined cadence (for example, quarterly for metrics, annually for ASI):

(1) R(c):
- Run revocation drills and recompute R(c) for representative T0/T1 credentials.
(2) OQ(S):
- Re-evaluate OQ(Si) for reference scenarios after major architectural changes.
(3) OAI:
- Rebuild the telemetry ownership map and update OAI_T0 and OAI_total.
(4) ASI:
- Re-validate ASI when staffing, tooling, or alert volumes change significantly.
(5) DEP:
- Recalculate CPA(I) and related values for key incident types.

All measurements MUST be recorded in a versioned repository.

5.4 Stage 3: Design and Governance Adjustments

For each metric that violates its target:

(1) Identify candidate interventions in:
- Architecture (sensors, logging, ingress paths).
- IAM and key management (token lifetimes, revocation mechanisms).
- Organizational ownership and escalation structures.
- Automation (playbooks, SOAR, self-service remediation).
(2) Prioritize interventions that:
- Reduce R(c) or increase OQ(S) for multiple scenarios.
- Reduce OAI and improve ASI simultaneously.
(3) Assign each intervention:
- An owner,
- A timeline,
- Expected metric deltas.

5.5 Stage 4: Exercise and Validation

(1) Implement changes in controlled or pilot domains.
(2) Conduct exercises simulating T0/T1 threats:
- Use red teams or realistic emulations.
- Confirm that:
- R(c) decreases where expected,
- OQ(S) increases,
- OAI falls,
- ASI is not worsened,
- DEP improves for relevant incident types.
(3) Only after successful validation SHOULD changes be deployed broadly.

5.6 Stage 5: Policy and Communication

Only after technical posture is measured and stable SHOULD policy steps be considered, such as:
	•	Public advisories or warnings.
	•	Diplomatic or legal responses.
	•	Strategic signaling and deterrence.

This sequencing reduces the risk of policy decisions based on incomplete or inaccurate technical understanding.
	6.	Governance

6.1 Metrics Registry

The organization MUST maintain a single authoritative registry that includes:
	•	T0/T1 asset inventory.
	•	Credential classes and latest R(c) values.
	•	Reference scenarios S and latest OQ(S).
	•	TelemetryDomains and OAI (OAI_T0, OAI_total).
	•	ASI measurements, with methods and assumptions.
	•	Summaries of DEP analysis for key incident types.

6.2 Change Control

Any change that can materially impact T0/T1 assets or metrics MUST:
	•	Include a pre-change estimate of impact on R(c), OQ(S), OAI, ASI, and DEP.
	•	Include a post-change measurement plan.
	•	Be reviewed by a governance function responsible for L1 and non-asymmetry compliance.

6.3 Role Separation

The following roles SHOULD be separated where feasible:
	•	Metric Owner:
	•	Defines metrics and methods.
	•	Ensures correctness and repeatability of measurements.
	•	System Owner:
	•	Owns systems that influence metric values.
	•	Executes changes and provides data.
	•	Governance Owner:
	•	Enforces thresholds and approves or rejects risky configurations.
	•	Escalates when metrics indicate structural failure.

	7.	Structural Acceptance Criteria

A national-level defensive posture is considered structurally acceptable if:

(1) For all T0 credentials:
- R(c) <= 0.5.

(2) For all high-priority scenarios S affecting T0:
- OQ(S) >= 0.9.

(3) Orphan domains:
- OAI_T0 = 0.
- OAI_total <= 3.

(4) Operational load:
- Under normal conditions, QueueLength(t) < ASI for most t.
- During surges, automated mechanisms ensure T0/T1-related alerts remain within effective ASI.

(5) Defensive economics:
- For T0/T1 incident types, DEP shows that incremental investments in detection/containment yield a positive expected value relative to loss from delayed response.

These criteria do not guarantee immunity from compromise, but they ensure the defender is not structurally configured to lose by design and that L1 is protected as a first-class objective.
