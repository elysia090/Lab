Strategic Command-and-Control Doctrine for National Signals Intelligence Services (Defensive-Only Edition)
v0.0.1 (Detailed)

0 Use Restriction and Status

0.1 Use restriction
The recipient shall not use, adapt, or derive this doctrine for offensive, proactive, or preemptive cyber operations under any circumstance.
This document is written for defensive command-and-control continuity, governance, auditing, incident response, resilience engineering, and lawful countermeasures within the recipient’s authority.

0.2 Status
v0.0.1 is an initial frozen edition. It defines:
	•	the command loop model (ping, pull, push)
	•	a reverse-lens method for finding structural breakpoints
	•	a measurement system that makes the doctrine executable (TU/PU accounting)
	•	four metrics that characterize structural defeat conditions (KHL, OQ_B, Orphan, ASI)
	•	governance requirements and verification drills

0.3 Normative keywords
The keywords MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY are to be interpreted as described in RFC 2119 style usage.

0.4 Audience
This document is intended for:
	•	national or enterprise incident commanders
	•	security operations leadership
	•	identity and access governance owners
	•	platform and cloud operations leaders
	•	policy, compliance, and legal stakeholders who control retention, escalation authority, and vendor boundaries

1 Foreword and Strategic Thesis

1.1 The contest of pulses
Modern command continuity is decided in the interval between operational pulses.
Permit the first sixty seconds to slip and the next three hours are typically already committed against you.
Surrender those three hours and strategic authority degrades faster than formal institutions can correct.

1.2 The three-note refrain
Strip a large organization to its marrow and what remains is a three-note refrain:
ping: self-interrogation. “What is my current state, and can I prove it?”
pull: external apprehension. “What beyond myself is true, and can I trust it?”
push: projection of will. “Apply a binding decision so the world changes, and prove that it changed.”

1.3 Core claim
The primary failure mode is not a single intrusion event. The primary failure mode is a break in lawful command:
	•	loss of reliable self-state
	•	loss of trustworthy external facts
	•	inability to execute, propagate, and confirm binding decisions within the required window

2 Definitions and Scope

2.1 Tier-Zero
Tier-Zero is the minimum set of systems, people, and procedures required to preserve the ping/pull/push loop under adversarial pressure.
Anything not required to preserve the loop MUST NOT be Tier-Zero.

2.2 Dominion
A dominion is a control artifact whose compromise allows broad command substitution or irreversible loss of governance, such as:
	•	identity super-admin roles (cloud and on-prem)
	•	signing keys and certificate authority roots
	•	privileged automation tokens for CI/CD and deployment
	•	break-glass access paths
	•	network control planes and routing authority
	•	authoritative audit and policy control planes (retention, logging, enforcement)

2.3 Control action
A control action is a bounded, lawful operation intended to restore command continuity, such as:
	•	revoke or rotate dominion credentials
	•	suspend or restrict privileged roles
	•	isolate a segment or workload
	•	roll back a configuration or deployment
	•	degrade service to a known-safe mode
	•	fail over to a trusted environment
	•	publish an attestation of known-good state
	•	initiate required notifications and legal holds

2.4 Scope
Scope is the set of assets, tenants, regions, subsidiaries, vendors, and environments to which a control action must apply for command continuity to be restored.
Scope MUST be explicit for every Tier-Zero control action.

2.5 Evidence spine
The evidence spine is a minimum set of high-integrity observations that:
	•	establish what happened (mechanism)
	•	establish what changed (state)
	•	establish what was authorized (lawful process)
	•	establish whether corrective actions applied across scope (confirmation)

3 Threat Model and Failure Conditions

3.1 Threat model (defensive)
This doctrine assumes an adversary can:
	•	exploit procedural blind spots (exceptions, unowned domains, retention gaps)
	•	exploit responsibility fragmentation (handoffs, approval gates, vendor boundaries)
	•	exploit cognitive overload (triage collapse under noise)
	•	leverage artifacts that appear lawful within existing processes (unauthorized changes that look authorized)

3.2 Defeat conditions (structural)
You are structurally defeated if any of the following become true for Tier-Zero scope:
	•	ping cannot establish a correct self-state within budget
	•	pull cannot provide trustworthy external facts within budget
	•	push cannot execute and confirm a binding decision within budget
	•	revocation and containment lag the adversary-relevant refresh window
	•	saturation forces selective blindness such that high-severity work is delayed beyond budget

4 Alignment to OODA

4.1 Mapping
OODA (Observe, Orient, Decide, Act) is aligned to ping/pull/push as follows:

Observe
	•	ping provides internal observation (state snapshots, invariants, change ledgers)
	•	pull provides external observation (telemetry, advisories, partner and supplier state, regulator and legal constraints)

Orient
	•	reconcile ping and pull into a consistent picture
	•	enforce bias controls (Section 5)

Decide
	•	select a bounded control action from a pre-defined set
	•	set scope, authorization path, and success criteria

Act
	•	push executes the control action
	•	push MUST close the loop by triggering a ping-based confirmation path

4.2 Closed-loop requirement
Every Tier-Zero push MUST have a defined confirmation path:
	•	what evidence proves effect across scope
	•	where that evidence is recorded
	•	what the deadline is for confirmation
If confirmation cannot be achieved within budget, the action MUST be treated as incomplete.

5 Why We Delay the “Why” (Bias Control)

5.1 Policy
Intent (the “why”) MUST NOT be used as a primary driver of early technical response decisions.
Early response MUST be driven by the evidence spine and the command loop budgets.

5.2 Rationale
	•	intent narratives are cheap to fake and costly to validate
	•	narratives induce confirmation bias in triage and engineering
	•	false-flag economics favor deception
	•	orientation precedes purpose in OODA; a weak orientation layer amplifies error
	•	structural seams outlive shifting motives; fix seams to blunt many possible motives
	•	policy actions require defensible attribution; premature hypotheses are operationally expensive

5.3 Allowed use of intent
Intent MAY be discussed as a bounded hypothesis only after:
	•	the evidence spine is established for mechanism and state
	•	Tier-Zero control actions are not delayed by motive debates
	•	uncertainties are explicitly recorded

6 Reversing the Lens (Method)

6.1 End-state first
Start with the worst credible end-state for Tier-Zero command continuity:
	•	you cannot revoke dominion access within window
	•	unauthorized changes appear authorized and propagate
	•	audit and retention prevent reconstruction
	•	authority cannot be exercised after-hours
	•	vendor boundaries prevent rapid action or confirmation

6.2 Cartography of absence
Map voids as first-class objects:
	•	unowned telemetry domains
	•	unmonitored subsystems and exception pathways
	•	retention gaps that erase evidence before confirmation
	•	delegated authorities not covered by standard review
	•	cross-jurisdiction areas where processes or subpoenas are slow
	•	after-hours authority gaps
	•	“shadow workflows” (manual steps, side channels, undocumented approvals)

6.3 Pre-emptive interdiction (defensive)
For each void, choose one:
	•	eliminate: remove the path or reduce scope so it cannot affect Tier-Zero
	•	constrain: reduce privilege, add guardrails, add ownership, add observability
	•	circuit-break: add a lawful, rapid, reversible-as-needed control action that stops further damage and can be confirmed
If excision is impossible, the void MUST be laced with a defensive circuit breaker and a verification drill.

7 Measurement System v0.0.1 (TU/PU Accounting)

7.1 Purpose
This doctrine is executable only if time and complexity are accounted for in a uniform way that requires minimal new measurement burden.

7.2 Time units
Transition Unit (TU)
	•	1 TU is one logged operational transition with a 15-minute budget.
	•	A transition is any recorded change that implies human or procedural movement, including:
a) ownership change (handoff)
b) workflow state change (open, investigating, mitigating, resolved, reopened)
c) severity or priority change
d) approval gate pass/fail (including legal/compliance gates)
e) queue transfer across teams, vendors, or authorities
f) rollback or reopen to an earlier workflow phase
	•	TU is counted from existing systems of record (ticketing, paging, approvals, incident comms).

Propagation Unit (PU)
	•	1 PU is a 30-minute propagation wait unit.
	•	PU captures the wall-clock time after a push is executed until the action is effective across scope and can be confirmed by ping.
	•	PU is measured by timestamps: execution time and confirmation time.

7.3 Rounding rules
	•	TU is integer count of transitions.
	•	PU is computed as ceil((confirmation_time - execution_time) / 30 minutes).
	•	If confirmation is not possible, PU MUST be recorded as UNKNOWN, and the action MUST be treated as non-confirmable.

7.4 Latency function
For any control action p:
Latency(p) = 15 minutes * TU(p) + 30 minutes * PU(p)

7.5 Budgets
Budgets MUST be defined at three levels:
	•	B60: first 60 seconds (initial command stabilization)
	•	B3H: first 3 hours (command reassertion window)
	•	BCONF: confirmation budget for each Tier-Zero control action
Budgets MAY differ by dominion type, but MUST be explicit.

7.6 Data integrity requirements
	•	Systems used to count TU MUST be authoritative and tamper-evident enough for audit.
	•	Logs required for confirmation MUST have retention that exceeds the confirmation budget plus review latency.
	•	If privacy or regulatory rules constrain logging, the doctrine MUST define alternate lawful evidence sources; silent absence is not acceptable.

8 The Four Metrics v0.0.1 (TU/PU Edition)

8.1 Key Half-Life (KHL)

8.1.1 Objective
Determine whether defenders can revoke or neutralize dominion control artifacts faster than adversary-relevant refresh windows.

8.1.2 Inputs per dominion p
	•	TU_revoke(p): transitions from decision to confirmed revocation across scope
	•	PU_revoke(p): propagation wait units to confirmation across scope
	•	RefreshWindow(p): the window in which continued adversary control remains effective, such as:
	•	credential renewal interval
	•	token lifetime and renewal cadence
	•	secret rotation cadence
	•	time-to-reissue for an unauthorized but “accepted” change

8.1.3 Definitions
KHL(p) = Latency_revoke(p) = 15m * TU_revoke(p) + 30m * PU_revoke(p)
R_KHL(p) = KHL(p) / RefreshWindow(p)

8.1.4 Interpretation
	•	If R_KHL(p) > 1, revocation lags the window; command continuity is structurally at risk.
	•	If R_KHL(p) < 1, revocation can outpace the window, subject to saturation (Section 8.4).

8.1.5 Required outputs
For each dominion p, record:
	•	KHL(p)
	•	R_KHL(p)
	•	drill evidence pointer (audit trail)
	•	worst-case (p95 or p99) KHL under after-hours conditions
Tier-Zero decisions MUST use worst-case values, not best-case.

8.2 Observability Quotient under budget (OQ_B)

8.2.1 Objective
Measure whether steps in a representative storyboard can be confirmed within a time budget, not merely whether telemetry exists.

8.2.2 Storyboards
A Tier-Zero storyboard is a sequence of steps that represent a credible command failure chain.
It MUST include at minimum:
	•	initial trigger (anomalous change or signal)
	•	execution point (where state actually changes)
	•	propagation point (where scope-wide effect appears)
	•	confirmation point (where ping can prove state)
	•	containment/revocation point (where push is applied)
Storyboards MUST be defensively framed and must not enumerate offensive techniques.

8.2.3 Inputs for each step i
	•	TU_confirm(i): transitions required to obtain and validate evidence for the step
	•	PU_confirm(i): propagation wait units before evidence is available
	•	ConfirmLatency(i) = 15m * TU_confirm(i) + 30m * PU_confirm(i)
	•	Budget B (policy-defined, e.g., 60m, 180m)

8.2.4 Definitions
Visible_B(i) is true iff ConfirmLatency(i) <= B and evidence is retained.
OQ_B = count(Visible_B(i)) / count(All storyboard steps)

8.2.5 Interpretation
Low OQ_B means your Observe/Orient layer cannot stabilize fast enough to support Decide/Act.

8.2.6 Required outputs
	•	OQ_B for at least two budgets (e.g., B=60m and B=180m)
	•	list of non-visible steps with reasons:
a) sensor absent
b) evidence delayed (PU too large)
c) evidence retention too short
d) evidence access requires too many transitions (TU too large)
	•	remediation target per step (reduce TU, reduce PU, extend retention, or change confirmation method)

8.3 Orphan Alert Map (Responsibility Fragmentation)

8.3.1 Objective
Quantify whether detection can reach authorized containment and revocation authority without structural dead zones.

8.3.2 Inputs for a representative escalation path
	•	OrphanTU: TU from detection to confirmed containment/revocation
	•	OrphanDomains: number of distinct boundaries crossed, including:
	•	separate teams or departments
	•	separate tool systems (ticketing, chat, paging, approvals)
	•	vendor or MSSP boundaries
	•	legal/compliance gates that can halt action
	•	AfterHoursPath: whether a lawful after-hours authority path exists and is trained

8.3.3 Definitions
OrphanScore may be recorded as a tuple:
OrphanScore = (OrphanTU, OrphanDomains)

8.3.4 Interpretation
	•	High OrphanDomains predicts non-linear delay under stress.
	•	High OrphanTU is direct delay and increases KHL.
	•	Missing AfterHoursPath is an immediate Tier-Zero defect.

8.3.5 Required outputs
	•	OrphanScore for each Tier-Zero dominion control action
	•	owner and backup owner
	•	authority chain (who can approve what, after-hours)
	•	evidence pointer (drill or real event)

8.4 Analyst Saturation Index (ASI_TU, optional ASI_PU)

8.4.1 Objective
Measure the point where operational throughput collapses and selective blindness begins.

8.4.2 Definitions
At a time t when triage degrades (filters tighten, backlogs spike, or escalation slows):
	•	RemainingTU(j): estimated TU to resolve queued item j
ASI_TU(t) = sum(RemainingTU(j) for all queued items j)

Optional:
	•	RemainingPU(j): if confirmation waits dominate the queue
ASI_PU(t) = sum(RemainingPU(j) for all queued items j)

8.4.3 Capacity and demand
Define:
	•	CapacityTU_rate: TU per hour the organization can process in Tier-Zero operations
	•	ArrivalTU_rate: TU per hour arriving (new incidents + required workflow transitions)
Saturation exists when ArrivalTU_rate > CapacityTU_rate for a sustained interval, causing ASI_TU to grow.

8.4.4 Interpretation
Saturation increases KHL by adding queue delay before revocation begins.
Therefore KHL MUST be reported alongside ASI, and worst-case KHL MUST assume saturation during adverse periods.

8.4.5 Required outputs
	•	ASI_TU at the moment triage degrades (with timestamp)
	•	observed triggers for triage degradation (policy change, staffing, tool failure)
	•	capacity assumptions and their evidence (staffing rosters, on-call coverage, automation)

9 Breakpoint Discovery (Finding where the loop breaks)

9.1 Principle
A breakpoint is not a “bug” in isolation. A breakpoint is a location where command continuity fails within budget.

9.2 ping breakpoints (self-state)
A ping breakpoint exists if any is true:
	•	self-state cannot be obtained within budget
	•	self-state is not uniquely defined (multiple conflicting sources)
	•	self-state changes without a corresponding authorized change record
	•	self-state cannot be confirmed across scope within budget
	•	required evidence is erased before confirmation (retention breakpoint)

Method: snapshot and reconciliation
	•	Maintain periodic Tier-Zero state snapshots of defined invariants (Appendix B).
	•	Reconcile snapshots against change ledgers (tickets, approvals, config audit, IAM audit).
	•	Any state delta without an authorized record is an UNKNOWN PATH defect.
UNKNOWN PATH defects MUST be treated as Tier-Zero priority and MUST have an owner within one business day.

9.3 pull breakpoints (external facts)
A pull breakpoint exists if:
	•	missingness is not detectable (you cannot prove you did not receive required input)
	•	provenance cannot be assessed within budget
	•	external facts do not reach the decision point within budget
	•	reconciliation across multiple sources cannot be performed within budget

Method: input SLOs and missingness-as-event
	•	For each required external input, define cadence, allowed delay, and provenance checks.
	•	Missingness MUST generate a logged transition (TU). Silent absence is forbidden.

9.4 push breakpoints (execution)
A push breakpoint exists if:
	•	authority cannot be reached lawfully within budget (after-hours, approvals, vendor boundary)
	•	propagation cannot be confirmed (PU UNKNOWN)
	•	execution affects only part of scope without a lawful record
	•	push cannot be closed back into ping confirmation within budget

Method: shortest lawful path and drills
	•	For each Tier-Zero control action, define the shortest lawful transition path and measure it by drill.
	•	Any action that exceeds its maximum tolerated window MUST be redesigned or removed from the Tier-Zero playbook.

10 Governance and Controls v0.0.1

10.1 Ownership (RACI)
Every Tier-Zero domain MUST have:
	•	a single accountable owner
	•	a backup owner
	•	an after-hours authority path
Unowned domains are forbidden in Tier-Zero scope.

10.2 Approval design
Tier-Zero emergency controls SHOULD minimize approvals.
If approvals are mandatory, the doctrine MUST define:
	•	who can approve after-hours
	•	maximum approval latency budget
	•	safe default actions when approval cannot be obtained (degrade, isolate, fail-safe)

10.3 Vendor and subsidiary boundaries
Contracts and SLAs MUST include:
	•	maximum time to execute control actions across scope
	•	maximum time to confirm propagation
	•	evidence requirements for confirmation
Vendor boundaries that cannot meet budgets MUST be treated as structural risk and MUST have compensating controls.

10.4 Retention and legal constraints
Retention policies MUST not erase evidence before confirmation and review.
If privacy law requires minimization, the doctrine MUST specify:
	•	minimal lawful fields required for confirmation
	•	lawful storage location and access controls
	•	legal hold triggers tied to Tier-Zero events

10.5 Exceptions
All exceptions MUST be:
	•	owned
	•	time-bounded
	•	logged
Exceptions that bypass change records create UNKNOWN PATH defects.

11 Verification Drills v0.0.1

11.1 Purpose
Drills convert assumptions into measured TU/PU values and produce evidence pointers for audit.

11.2 Revocation drill (KHL)
For each dominion category:
	•	execute a lawful revocation or neutralization action in a controlled manner
	•	measure TU_revoke and PU_revoke
	•	confirm across full scope with defined evidence
	•	record worst-case values (after-hours drill at least quarterly)

11.3 Confirmation drill (OQ_B)
	•	run a Tier-Zero storyboard against production-equivalent telemetry
	•	measure ConfirmLatency(i) for each step
	•	update OQ_B and remediation backlog

11.4 Saturation drill (ASI)
	•	simulate a workload increase in a controlled way that does not introduce harm
	•	observe when triage degrades
	•	record ASI_TU at degradation time
	•	update capacity planning and automation priorities

11.5 Evidence requirements
Every drill MUST record:
	•	timestamps (start, decision, execution, confirmation)
	•	the system of record for TU counting
	•	the confirmation evidence source
	•	who authorized and who executed
	•	the scope covered

12 Operating Cadence v0.0.1

12.1 Daily
	•	ping snapshot and reconciliation report
	•	open UNKNOWN PATH defects and assign owners
	•	review after-hours readiness for Tier-Zero actions

12.2 Weekly
	•	update KHL and OQ_B for Tier-Zero dominions and storyboards
	•	review OrphanScores and remove unnecessary boundaries
	•	review saturation windows and staffing

12.3 Monthly
	•	at least one dominion revocation drill per category
	•	validate retention and confirmation evidence availability

12.4 Quarterly
	•	after-hours drill for the most critical dominions
	•	saturation drill and capacity recalibration
	•	governance review of vendor boundaries and exceptions

13 Reporting and Decision Support

13.1 Required dashboards
	•	KHL and R_KHL per dominion (p50, p95, worst-case)
	•	OQ_B per storyboard with non-visible steps list
	•	OrphanScore per Tier-Zero control action
	•	ASI_TU trend and saturation windows
	•	UNKNOWN PATH defect count and aging

13.2 Risk register
Every Tier-Zero defect MUST be tracked with:
	•	defect type (ping/pull/push breakpoint, retention breakpoint, authority gap)
	•	impacted dominions and scope
	•	measured TU/PU impact
	•	remediation plan and owner
	•	deadline aligned to budgets

13.3 Decision policy
Tier-Zero actions MUST prioritize:
	1.	restoring push ability (lawful controls that stop further damage)
	2.	restoring ping certainty (provable self-state)
	3.	restoring pull reliability (trustworthy external facts)
	4.	only then refining attribution and intent narratives

Appendix A Minimal Templates (ASCII fields)

A1 Dominion Inventory Record
	•	dominion_id
	•	description
	•	owner
	•	backup_owner
	•	scope
	•	control_actions (list)
	•	RefreshWindow (time)
	•	max_tolerated_latency (time)
	•	TU_revoke_measured (p50, p95, worst)
	•	PU_revoke_measured (p50, p95, worst)
	•	KHL (p50, p95, worst)
	•	R_KHL (p50, p95, worst)
	•	after_hours_path (true/false + pointer)
	•	last_drill_timestamp
	•	drill_evidence_pointer

A2 Control Action Record
	•	action_id
	•	action_type (revoke, suspend, isolate, rollback, degrade, failover, attest, notify)
	•	lawful_authority_basis (policy reference)
	•	required_approvals (list)
	•	shortest_lawful_transition_path (enumerated)
	•	confirmation_method
	•	confirmation_evidence_source
	•	scope_definition
	•	TU_measured (p50, p95, worst)
	•	PU_measured (p50, p95, worst)
	•	max_tolerated_latency
	•	last_drill_timestamp
	•	evidence_pointer

A3 Storyboard Step Record (for OQ_B)
	•	storyboard_id
	•	step_id
	•	step_description
	•	required_evidence_source
	•	confirmation_method
	•	TU_confirm_measured
	•	PU_confirm_measured
	•	ConfirmLatency
	•	budget_B
	•	Visible_under_budget_B (true/false)
	•	failure_reason (absent, delayed, retention, access, ambiguity)
	•	remediation_target (reduce_TU, reduce_PU, extend_retention, change_method)
	•	evidence_pointer

A4 Orphan Path Record
	•	path_id
	•	detection_trigger_id
	•	control_action_id
	•	OrphanTU_measured
	•	OrphanDomains_count
	•	boundary_list
	•	after_hours_path_exists (true/false)
	•	owner
	•	evidence_pointer

A5 Saturation Record
	•	time_window
	•	ArrivalTU_rate
	•	CapacityTU_rate
	•	ASI_TU_at_degradation
	•	triage_degradation_signal (policy change, filter, backlog threshold)
	•	notes
	•	evidence_pointer

Appendix B Tier-Zero ping Snapshot (example categories)
This appendix defines categories only. Each organization MUST specify exact items.
	•	privileged role memberships and super-admin grants
	•	key and certificate authority state (issuance, revocation lists, signing policy)
	•	CI/CD signing and deployment token inventories and last-use
	•	break-glass account state and access logs
	•	configuration baselines for Tier-Zero systems
	•	retention and audit pipeline health indicators
	•	vendor boundary control status and last confirmation timestamps

Change Log
v0.0.1 Initial defensive-only edition (detailed). Implements ping/pull/push to OODA alignment, TU/PU accounting, four metrics, breakpoint discovery, governance controls, and verification cadence. Offensive operational content is intentionally excluded.

