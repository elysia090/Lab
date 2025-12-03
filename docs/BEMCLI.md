BEM Service Surfaces and Security Model v0.0.1
	0.	Scope

This document defines how to split the BEM-based service into:
	1.	A multi-tenant cloud API, covered by SOC2 / ISO27001.
	2.	An edge CLI runtime that runs inside customer infrastructure.

It specifies data boundaries, responsibilities, and allowed interactions between the two.
	1.	Surfaces

1.1 Cloud API

Logical role:
A multi-tenant SaaS endpoint that generates and evaluates synthetic tasks using BEM running in your cloud.

Main capabilities:
	1.	Task generation for pretraining and RL.
	2.	Evaluation and benchmarking of customer models.
	3.	Adversarial and safety-oriented synthetic tasks.

API examples (conceptual):
	•	POST /v1/tasks/generate
	•	POST /v1/eval/run
	•	POST /v1/benchmarks/run
	•	POST /v1/safety/adversarial

The cloud API is fully inside your SOC2 / ISO27001 scope.

1.2 Edge CLI

Logical role:
A local runner that executes BEM workloads inside the customer environment, without sending code or data to your cloud by default.

Main capabilities:
	1.	Run BEM solvers on local code and data.
	2.	Generate synthetic tasks and evaluations locally.
	3.	Optionally send limited metadata or aggregated statistics to the cloud API, if explicitly enabled.

The edge CLI is outside your operational SOC2 / ISO27001 scope for runtime data. You are responsible only for its development lifecycle; customers are responsible for its operation and data.
	2.	Security and Compliance Boundary

2.1 Boundary Definition

Cloud side:
	•	Assets in scope:
	•	Control plane (auth, tenancy, billing).
	•	BEM compute cluster in your cloud.
	•	Cloud storage for logs and metrics.
	•	API gateways and related services.
	•	Covered by:
	•	SOC2 controls.
	•	ISO27001 ISMS scope.
	•	Your documented data protection policies.

Edge side:
	•	Assets in scope for the customer:
	•	Edge CLI binaries and local configuration.
	•	Local BEM runtime instances.
	•	Local logs, caches, task outputs.
	•	Not part of your runtime data scope:
	•	You do not manage or store customer code or data that stays on the edge.

2.2 Data Categories

Define three broad data categories:
	1.	Raw customer payloads:
	•	Source code.
	•	Proprietary datasets.
	•	Internal specs and logs.
	2.	Abstracted metadata:
	•	Task types.
	•	Difficulty levels.
	•	Error and result categories.
	•	Aggregated statistics.
	3.	Service-level telemetry:
	•	Usage counts.
	•	Latencies.
	•	Version identifiers.

Rules:
	•	Cloud API may process raw payloads only in explicit opt-in modes.
	•	Default cloud usage is restricted to metadata and telemetry where feasible.
	•	Edge CLI default is to keep raw payloads entirely local.

	3.	Cloud API Design

3.1 Tenancy and Isolation

Each request is bound to a tenant:
	•	tenant_id included in auth and in all logs.
	•	Per-tenant:
	•	Separate logical namespaces.
	•	Separate encryption keys.
	•	Policy controls for which data types are allowed (metadata only or raw payloads allowed).

Multi-tenant data isolation:
	•	No cross-tenant task mixing unless explicitly configured for shared public benchmarks.
	•	Synthetic tasks generated for one tenant are not tagged or exposed as another tenant’s private data.

3.2 Data Modes

For each API family, define allowed modes.

Example: /v1/eval/run
	1.	Metadata-only mode:
	•	Inputs:
	•	Task identifiers.
	•	Difficulty and type.
	•	Short anonymized model outputs (for example classification IDs or pass/fail tags).
	•	No raw source code or long free-form outputs.
	2.	Restricted payload mode:
	•	Inputs:
	•	Code snippets or model outputs.
	•	Requires explicit tenant-level configuration and contractual agreement.

3.3 Logging

Cloud logs contain:
	•	tenant_id, endpoint, timestamp.
	•	Payload size and types (for example “metadata only”, “code snippet”).
	•	Result status and high-level metrics.

They do not contain:
	•	Full raw payloads by default, unless in restricted payload mode and explicitly documented.

All logs are retained and rotated according to your compliance policies.
	4.	Edge CLI Design

4.1 Operating Modes

The CLI supports three modes:
	1.	Offline mode (default):
	•	No outbound network usage.
	•	BEM models and tasks run entirely local.
	•	No data sent to the cloud.
	2.	Telemetry-only mode:
	•	Sends only minimal usage metrics:
	•	Counts, anonymized error codes, version information.
	•	No raw code or task contents.
	3.	Opt-in sync mode:
	•	Optionally sends selected artifacts to the cloud:
	•	For example anonymized task summaries or specific synthetic tasks for global improvement.
	•	Controlled by explicit configuration flags.

4.2 Default Behavior

Security default:
	•	Install and run with offline mode enabled.
	•	Any data sharing beyond telemetry-only must be:
	•	Explicitly configured.
	•	Visible in CLI help.
	•	Auditable via local logs.

4.3 Configuration

Examples of flags and config keys:
	•	–offline:
	•	Force no network, even if configuration suggests otherwise.
	•	–mode=telemetry-only:
	•	Permit only basic usage reporting.
	•	–mode=sync:
	•	Allow sending configured artifacts to cloud.

Configuration files:
	•	Stored under a local path controlled by the customer.
	•	Document which fields control network behavior.

	5.	Learning and Model Updates

5.1 Global BEM (Cloud)

Global models in the cloud can learn from:
	•	Aggregated statistics from all tenants.
	•	Synthetic task performance distributions.
	•	Error type frequencies and difficulty curves.

They do not require access to raw tenant payloads. Where possible, training signals should be derived from:
	•	Anonymous histograms.
	•	Difficulty versus pass rate.
	•	Class of failure modes.

5.2 Tenant-Specific Adapters (Edge)

Customers may customize:
	•	Local BEM experts.
	•	Local templates and macros.
	•	Local policies for their internal APIs and codebases.

These adaptations:
	•	Remain on the edge by default.
	•	Are not uploaded unless the customer explicitly chooses to contribute patterns upstream.

5.3 Contribution Channels (Optional)

If customers opt in, define narrow contribution channels:
	•	Abstract templates:
	•	API arity, type signatures, but no names or secrets.
	•	Task schemas:
	•	Problem families without proprietary content.

These are integrated into global BEM as generic patterns, not as raw data.
	6.	Responsibilities and Contracts

6.1 Your Responsibilities

You are responsible for:
	•	Security and compliance of:
	•	Cloud control plane.
	•	Cloud BEM compute cluster.
	•	Cloud storage of logs and metadata.
	•	Access control and auditing over the cloud API.
	•	Proper handling of whatever payload category the tenant enables:
	•	Metadata-only.
	•	Restricted payloads.

6.2 Customer Responsibilities

Customers are responsible for:
	•	Securing the environment where the edge CLI runs.
	•	Controlling which mode the CLI operates in.
	•	Ensuring that data they allow to leave their environment is permitted by their internal policies.

6.3 Documentation Requirements

You must provide:
	•	A clear description of:
	•	What data each endpoint can accept.
	•	What data each CLI mode can transmit.
	•	A matrix that maps:
	•	Data categories
	•	Surfaces (API vs CLI)
	•	Compliance scope (in-scope vs out-of-scope)

So that auditors and customers can see exactly where each type of data flows.
	7.	Summary

By cleanly separating:
	1.	Cloud API:
	•	Multi-tenant, SOC2 / ISO27001 scoped, clear payload categories and logging.
	2.	Edge CLI:
	•	Runs in customer environments, offline by default, with explicit opt-in for any data sharing.

you get:
	•	A simple and auditable security boundary.
	•	A clear data separation story.
	•	A product shape that supports both a SaaS model and on-prem style usage without mixing responsibilities.
