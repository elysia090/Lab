ACAP Specification (SSOT) v0.0.1

Status: frozen, normative. This document is the single source of truth for ACAP v0.0.1.

Table of contents
0. Conventions
	1.	Scope
	2.	Threat model
	3.	Terms
	4.	Conformance
	5.	Object model
	6.	OCI carriage profile (ACAP-OCI)
	7.	Canonical encoding profile (ACAP-CBOR)
	8.	Cryptographic profile (ACAP-COSE)
	9.	Trust, roles, and rotation (ACAP-TRUST)
	10.	Schemas (ACAP-SCHEMA)
	11.	Canonicalization rules (ACAP-CANON)
	12.	Retrieval and bounding rules (ACAP-FETCH)
	13.	Evaluation semantics (ACAP-EVAL)
	14.	Verification algorithm (ACAP-VERIFY)
	15.	Minimal HTTP API (optional)
	16.	Violation code registry (v0.0.1)
	17.	Completeness checklist (normative)
	18.	Conventions

0.1 Normative keywords
The keywords MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT, MAY are normative.

0.2 ASCII and determinism
All normative text, identifiers, and field names in ACAP v0.0.1 are ASCII.
ACAP is deterministic: given identical inputs, a conforming verifier MUST produce identical results and identical violation codes.

0.3 Fail-closed
ACAP is fail-closed by default.
Any missing required data, parsing ambiguity, unknown required field, unknown critical header, or inability to fetch required objects MUST result in DENY.

0.4 Version pinning
This specification defines v0.0.1 only.
Any object with v != “0.0.1” MUST be rejected.
	1.	Scope

ACAP v0.0.1 standardizes:
	•	How to bind Evidence and Decision to an OCI Subject by digest.
	•	How to encode payloads deterministically (CBOR) and sign them (COSE).
	•	Trust bundle format, role separation, thresholds, and rotation rules.
	•	Policy object format and evaluation semantics.
	•	Verifier pinning and policy selection pinning at admission.
	•	Revocation and exception mechanisms (signed, scoped, time-bounded).
	•	Anti-replay (target binding, TTL, optional challenge binding).
	•	Decision chaining and anchoring (rollback resistance).
	•	Bounded discovery and retrieval rules (DoS resistance).
	•	Minimal transparency log chain (optional).
	•	Minimal error output format for HTTP (optional).

ACAP v0.0.1 does not standardize:
	•	A general purpose policy language.
	•	A CI system, scheduler, or build system.
	•	A transparency service network protocol (only a local tlog object format).

	2.	Threat model

ACAP assumes:
	•	The registry, network, and storage may be malicious or inconsistent.
	•	Build and CI infrastructure may be compromised.
	•	A single signer key may be compromised.
	•	Operators may attempt ad-hoc exceptions.
	•	Old decisions may be replayed into other environments.
	•	Time sources may be skewed.

ACAP mitigations include:
	•	Subject-by-digest binding.
	•	Deterministic CBOR encoding and signed COSE messages.
	•	Threshold multi-signatures and strict role separation.
	•	Admission-pinned policy selector and verifier selector (Decisions do not choose policy/verifier).
	•	Revocation artifacts with strict precedence.
	•	Exceptions as signed artifacts with strict non-waivable invariants.
	•	Target binding, TTL, and optional challenge binding.
	•	Decision pointer requirement (no arbitrary Decision selection from the registry).
	•	Decision chaining with an anchor rule.
	•	Bounded referrers discovery and bounded fetch.
	•	Optional mirror double-fetch rule.
	•	Optional local transparency log chain.

	3.	Terms

Artifact
An OCI object (image, artifact manifest, or index) identified by digest.

Subject
The specific OCI object being gated, identified by digest and kind.

Subject digest
A string “sha256:” + 64 lowercase hex.

Evidence
A signed statement about a Subject (provenance, SBOM, tests, scans, receipts).

Decision
A signed gate result (allow or deny) produced from Policy and Evidence for a Subject, for a specific target and time window.

Policy
A signed object declaring requirements, thresholds, allowlists, anti-replay constraints, and evaluation rules.

Verifier
The exact implementation used to evaluate Policy and Evidence into a Decision, pinned by digest.

Trust Bundle
A signed set of public keys and roles, with thresholds.

Revocation
A signed object invalidating keys and/or objects by match rules.

Exception
A signed, time-bounded override scoped to a Subject and Policy, allowing only explicitly waivable violation codes.

Target
Admission context identifiers (environment, cluster, namespace, workload, etc.).

Challenge
Optional nonce issued by admission to bind a Decision to a specific request instance.

Tlog
A local append-only transparency log chain object format.
	4.	Conformance

4.1 Conforming producer
A conforming producer MUST:
	•	Produce objects that match schemas in Section 10.
	•	Encode payloads using ACAP-CBOR (Section 7).
	•	Sign payloads using ACAP-COSE (Section 8).
	•	Set OCI carriage fields per ACAP-OCI (Section 6).

4.2 Conforming verifier
A conforming verifier MUST:
	•	Implement retrieval and bounding rules (Section 12).
	•	Implement evaluation semantics (Section 13).
	•	Implement verification algorithm (Section 14).
	•	Deny on any ambiguity or missing required input.

4.3 Unknown fields
Unknown top-level keys outside “ext” MUST cause rejection.
Unknown keys inside “ext” MUST be ignored.
For Evidence predicate objects, unknown keys outside “ext” inside the predicate schema MUST cause rejection if the predicateType is required by policy.

4.4 No forward compatibility
No forward compatibility behavior is defined in v0.0.1.
A verifier MUST NOT accept objects that deviate from v0.0.1 schemas.
	5.	Object model

5.1 Object types
ACAP defines these signed payload types:
	•	acap/evidence
	•	acap/decision
	•	acap/policy
	•	acap/revocation
	•	acap/exception
	•	acap/verifier
	•	acap/trust-bundle
	•	acap/tlog

5.2 Core invariants
I1. Every Evidence and Decision MUST bind to exactly one Subject digest (the Subject digest).
I2. Every Decision MUST bind to exactly one Policy digest and exactly one Verifier digest.
I3. Every verifier MUST enforce admission-configured policy and verifier selectors; Decisions do not define intent.
I4. Revocation overrides everything: revoked keys or objects MUST be treated invalid.
I5. Absent required Evidence types MUST yield DENY.
I6. Any fetch failure for required objects MUST yield DENY.
I7. Any parsing, schema, deterministic encoding, signature, or role validation failure MUST yield DENY.

5.3 Decision pointer
Admission MUST require an explicit Decision pointer from the deployment request:
	•	Either decision.digest, or decision_id (resolving uniquely as specified).
If the pointer is absent, the admission result MUST be DENY.

5.4 Subject kinds
Subject.kind MUST be one of:
	•	“oci.manifest”
	•	“oci.index”

Index handling is defined in Section 13.11.
	6.	OCI carriage profile (ACAP-OCI)

6.1 OCI association
Evidence and Decision MUST be stored as OCI artifacts whose manifests set:
	•	subject to the referenced Subject descriptor (OCI descriptor)
	•	artifactType to an ACAP-defined type string

Discovery MUST use an OCI referrers mechanism equivalent to OCI 1.1 behavior; tag scanning MUST NOT be required.

6.2 ACAP artifactType strings (v0.0.1)
These artifactType values are reserved and MUST be used verbatim:
	•	application/vnd.acap.evidence.v0.0.1
	•	application/vnd.acap.decision.v0.0.1
	•	application/vnd.acap.policy.v0.0.1
	•	application/vnd.acap.revocation.v0.0.1
	•	application/vnd.acap.exception.v0.0.1
	•	application/vnd.acap.verifier.v0.0.1
	•	application/vnd.acap.trust-bundle.v0.0.1
	•	application/vnd.acap.tlog.v0.0.1

6.3 Payload layer
Each ACAP object MUST be stored as exactly one content-addressed blob layer containing COSE bytes.
The blob mediaType SHOULD be:
	•	application/cose

6.4 Canonical identifier
The canonical identifier of an ACAP object is the OCI blob digest of its COSE bytes.

6.5 Repository scope
ACAP retrieval is repository-scoped (Section 12).
Therefore, every ACAP payload MUST include a locator (Section 10.1).
	7.	Canonical encoding profile (ACAP-CBOR)

7.1 Deterministic CBOR
All signed payload bytes MUST be a deterministic CBOR encoding of the payload map.

7.2 Allowed data model
	•	Keys MUST be UTF-8 text strings and MUST be ASCII.
	•	Map keys MUST be unique.
	•	Integers MUST fit in signed 64-bit (no bignum).
	•	Floating point MUST NOT appear anywhere in v0.0.1 payloads.
	•	Byte strings MAY appear only where explicitly specified.
	•	Arrays and maps MAY be nested only as permitted by schemas.

7.3 Timestamp representation
All timestamps are RFC3339 text strings under specific keys (ts, valid_from, valid_to, expires_at).

7.4 Extensions
All extensibility MUST occur under a single top-level key “ext” whose value is a map.
Unknown top-level keys outside “ext” MUST cause rejection.
	8.	Cryptographic profile (ACAP-COSE)

8.1 COSE message types
	•	Evidence MAY be signed with COSE_Sign1 (single signature).
	•	Decision, Policy, Revocation, Exception, Trust Bundle, Verifier, and Tlog MUST be signed with COSE_Sign (multi-signature).

8.2 Algorithm profile
	•	Signatures MUST use Ed25519.
	•	Hashing MUST use SHA-256 where hashing is specified.
	•	Any other algorithm MUST be rejected.

8.3 Protected headers (required)
The protected header map MUST contain:
	•	alg: “EdDSA”
	•	kid: bstr
	•	typ: tstr
	•	v: “0.0.1”

The unprotected header MUST be empty.

8.4 Key id (kid)
kid MUST be a byte string equal to SHA-256(pubkey_bytes) where pubkey_bytes is the 32-byte Ed25519 public key.

8.5 typ header binding
The COSE protected header “typ” MUST equal the payload field “typ”.
Mismatch MUST be rejected.

8.6 Multi-signature rules
For COSE_Sign:
	•	The payload MUST be present exactly once and be identical for all signatures.
	•	Each signature entry MUST include its own protected header with kid and alg and typ and v.
	•	Duplicate kid values within the same COSE_Sign MUST be rejected.
	•	The verifier MUST count signatures toward a threshold only if the kid is trusted for the required role.

8.7 Critical headers
If a critical header mechanism is present, any unknown critical parameter MUST cause rejection.
	9.	Trust, roles, and rotation (ACAP-TRUST)

9.1 Roles
Trust Bundle keys MUST be assigned exactly one role:
	•	policy
	•	decision
	•	revocation
	•	exception
	•	trust-bundle
	•	verifier
	•	tlog

A key MUST NOT appear in more than one role in the same Trust Bundle.

9.2 Thresholds
Trust Bundle MUST declare threshold k for each role.
A signature set satisfies a role iff it contains at least k valid signatures from distinct kids with that role.

9.3 Trust Bundle pinning (bootstrap)
Admission configuration MUST pin an initial Trust Bundle by digest:
	•	config.trust_bundle_digest (REQUIRED)
Verifiers MUST reject any Trust Bundle not matching the pinned digest unless rotation rules (9.4) apply.

9.4 Rotation
A new Trust Bundle MUST contain:
	•	prev_trust_bundle_digest: tstr (sha256:…)
Rotation acceptance requires:
	•	The new Trust Bundle is multi-signed by the old trust-bundle role threshold AND by the new trust-bundle role threshold.
If cross-signing is missing or invalid: reject.

9.5 Trust Bundle expiry
Trust Bundle MUST include expires_at and verifiers MUST reject expired bundles (with clock skew rules in Section 11.4).

9.6 Admission selectors (intent pinning)
Admission MUST be configured with:
	•	config.required_policy_ids[] or config.required_policy_digests[]
	•	config.required_verifier_ids[] or config.required_verifier_digests[]
At least one of ids or digests MUST be provided for policy and for verifier.

Decisions referencing policy/verifier outside these selectors MUST be rejected.
	10.	Schemas (ACAP-SCHEMA)

All payloads are CBOR maps and MUST include:
	•	v: “0.0.1”
	•	typ: type string
	•	ts: RFC3339 time string
	•	locator: locator map
	•	ext: optional map

10.1 Common schemas

Locator
locator = {
“registry”: tstr / null,
“repo”: tstr,
“subject_repo”: tstr / null
}

Rules:
	•	repo is REQUIRED and identifies the repository scope used to publish this object.
	•	subject_repo, if present, identifies the repository of the Subject.
	•	If registry is null, admission configuration MUST supply the registry host for the repo.

Subject descriptor
subject = {
“digest”: tstr,
“kind”: tstr,
“name”: tstr / null,
“mediaType”: tstr / null
}

Rules:
	•	digest MUST match “sha256:” + 64 lowercase hex.
	•	kind MUST be “oci.manifest” or “oci.index”.

Target
target = {
“env”: tstr,
“cluster”: tstr / null,
“namespace”: tstr / null,
“workload”: tstr / null,
“tenant”: tstr / null,
“region”: tstr / null,
“ext”: { * tstr => any } / null
}

Violation
violation = {
“code”: tstr,
“severity”: tstr,
“msg”: tstr / null,
“loc”: tstr / null,
“hint”: tstr / null,
“ext”: { * tstr => any } / null
}

Severity values MUST be one of: “low”, “med”, “high”, “crit”.

Digest list
digest_str MUST be “sha256:” + 64 lowercase hex.

10.2 Evidence payload (typ = “acap/evidence”)
Evidence payload is an in-toto Statement v1 shaped object, encoded as deterministic CBOR, with a constrained schema.

evidence = {
“v”: “0.0.1”,
“typ”: “acap/evidence”,
“ts”: tstr,
“locator”: locator,
“_type”: “https://in-toto.io/Statement/v1”,
“subject”: [ subject ],
“predicateType”: tstr,
“predicate”: { * tstr => any },
“ext”: { * tstr => any } / null
}

Rules:
	•	subject array MUST contain exactly one element in v0.0.1.
	•	subject[0].digest is the Subject digest.
	•	predicateType MUST be a policy-recognized predicate type string.
	•	predicate MUST conform exactly to the minimal schema for its predicateType (Section 10.8) when the predicateType is required by policy.

10.3 Policy payload (typ = “acap/policy”)
policy = {
“v”: “0.0.1”,
“typ”: “acap/policy”,
“ts”: tstr,
“locator”: locator,
“policy_id”: tstr,
“prev_policy_digest”: tstr / null,
“expires_at”: tstr,
“required_predicate_types”: [ + tstr ],
“required_signers”: {
“decision”: { “threshold”: uint, “members”: [ + bstr ] },
“policy”: { “threshold”: uint, “members”: [ + bstr ] },
“revocation”: { “threshold”: uint, “members”: [ + bstr ] },
“exception”: { “threshold”: uint, “members”: [ + bstr ] },
“trust_bundle”: { “threshold”: uint, “members”: [ + bstr ] },
“verifier”: { “threshold”: uint, “members”: [ + bstr ] },
“tlog”: { “threshold”: uint, “members”: [ + bstr ] }
},
“waivable_codes”: [ * tstr ],
“nonwaivable_codes”: [ + tstr ],
“allowed_builders”: [ * tstr ],
“hermetic”: bool,
“inputs_required”: bool,
“revocation”: {
“required”: bool,
“max_age_seconds”: uint
},
“decision”: {
“validity_seconds_max”: uint,
“target_required”: bool,
“challenge_mode”: tstr,
“clock_skew_seconds_max”: uint
},
“rollback”: {
“chain_required”: bool,
“min_chain_height”: uint,
“anchor_mode”: tstr,
“anchor_decision_digest”: tstr / null,
“anchor_tlog_digest”: tstr / null
},
“index”: {
“index_mode”: tstr
},
“fetch”: {
“max_referrers_total”: uint,
“max_referrers_per_type”: uint,
“max_blob_bytes”: uint
},
“mirror”: {
“double_fetch_required”: bool
},
“transparency”: {
“tlog_required”: bool,
“tlog_anchor_required”: bool
},
“retention”: {
“min_days”: uint
},
“ext”: { * tstr => any } / null
}

Rules:
	•	challenge_mode MUST be one of “none”, “optional”, “required”.
	•	index.index_mode MUST be one of “index-only”, “covers-all-platforms”.
	•	rollback.anchor_mode MUST be one of “none”, “decision”, “tlog”.
	•	If rollback.chain_required is true, anchor_mode MUST NOT be “none”.
	•	waivable_codes MAY be empty. Default semantics if absent are defined by schema: in v0.0.1 it is present but may be empty.
	•	nonwaivable_codes MUST be non-empty and MUST include at least:
SIG.INVALID, SIG.THRESHOLD_NOT_MET, TRUST.EXPIRED, TRUST.NOT_PINNED, TRUST.KEY_NOT_ALLOWED,
REVOCATION.HIT, POLICY.EXPIRED, POLICY.NOT_SELECTED, VERIFIER.NOT_SELECTED,
SUBJECT.MISMATCH, SUBJECT.KIND_INVALID, FETCH.MISSING, FETCH.OVER_LIMIT,
TIME.INVALID_RANGE, DECISION.POINTER_MISSING, DECISION.TARGET_MISMATCH,
DECISION.CHALLENGE_REQUIRED, DECISION.CHALLENGE_REUSED

10.4 Verifier payload (typ = “acap/verifier”)
verifier = {
“v”: “0.0.1”,
“typ”: “acap/verifier”,
“ts”: tstr,
“locator”: locator,
“verifier_id”: tstr,
“artifact_digest”: tstr,
“compat”: { “acap”: “0.0.1” },
“ext”: { * tstr => any } / null
}

Rules:
	•	artifact_digest is the digest of the OCI object representing the verifier.
	•	Admission selector MUST constrain accepted verifiers (Section 9.6).

10.5 Trust Bundle payload (typ = “acap/trust-bundle”)
trust_bundle = {
“v”: “0.0.1”,
“typ”: “acap/trust-bundle”,
“ts”: tstr,
“locator”: locator,
“expires_at”: tstr,
“prev_trust_bundle_digest”: tstr / null,
“keys”: [ + { “kid”: bstr, “pub”: bstr, “role”: tstr } ],
“thresholds”: { * tstr => uint },
“ext”: { * tstr => any } / null
}

Rules:
	•	pub MUST be 32 bytes (Ed25519).
	•	kid MUST equal SHA-256(pub).
	•	Each role in thresholds MUST have a corresponding set of keys in keys[].
	•	Roles MUST be the set defined in Section 9.1.

10.6 Decision payload (typ = “acap/decision”)
decision = {
“v”: “0.0.1”,
“typ”: “acap/decision”,
“ts”: tstr,
“locator”: locator,
“decision_id”: tstr,
“subject”: subject,
“policy”: { “digest”: tstr, “policy_id”: tstr },
“verifier”: { “digest”: tstr, “verifier_id”: tstr },
“result”: tstr,
“evidence_digests”: [ * tstr ],
“target”: target / null,
“valid_from”: tstr,
“valid_to”: tstr,
“challenge_hash”: bstr / null,
“challenge_id”: tstr / null,
“prev_decision_digest”: tstr / null,
“chain_height”: uint,
“covered_subjects”: [ * tstr ],
“violations”: [ * violation ],
“ext”: { * tstr => any } / null
}

Rules:
	•	result MUST be “allow” or “deny”.
	•	If result == “deny”, violations MUST be non-empty.
	•	valid_to MUST NOT be earlier than valid_from.
	•	The time window length MUST be <= policy.decision.validity_seconds_max.
	•	If policy.decision.target_required is true, target MUST be non-null.
	•	If policy.decision.challenge_mode == “required”, challenge_hash and challenge_id MUST be non-null.
	•	If policy.rollback.chain_required is true, prev_decision_digest MUST be non-null, and chain_height MUST be >= policy.rollback.min_chain_height.
	•	If subject.kind == “oci.index” and policy.index.index_mode == “covers-all-platforms”, covered_subjects MUST list the digests of all referenced platform manifests (Section 13.11). If index_mode == “index-only”, covered_subjects MUST be empty.

10.7 Revocation payload (typ = “acap/revocation”)
revocation = {
“v”: “0.0.1”,
“typ”: “acap/revocation”,
“ts”: tstr,
“locator”: locator,
“revocation_id”: tstr,
“expires_at”: tstr,
“reason”: tstr,
“match”: {
“keys_kid”: [ * bstr ],
“object_digests”: [ * tstr ],
“policy_digests”: [ * tstr ],
“verifier_digests”: [ * tstr ],
“decision_digests”: [ * tstr ],
“builder_ids”: [ * tstr ]
},
“ext”: { * tstr => any } / null
}

Rules:
	•	expires_at MUST be enforced; expired revocation objects MUST be ignored.
	•	Any match hit MUST invalidate the matched object or key immediately regardless of timestamps.

10.8 Exception payload (typ = “acap/exception”)
exception = {
“v”: “0.0.1”,
“typ”: “acap/exception”,
“ts”: tstr,
“locator”: locator,
“exception_id”: tstr,
“expires_at”: tstr,
“subject”: subject,
“policy_digest”: tstr,
“allow_violation_codes”: [ + tstr ],
“reason”: tstr,
“ext”: { * tstr => any } / null
}

Rules:
	•	Exception applies only to the exact subject.digest and the exact policy_digest.
	•	Exception MUST be rejected if any allow_violation_codes is in policy.nonwaivable_codes.
	•	Exception MUST be rejected if any allow_violation_codes is not in policy.waivable_codes.
	•	Exception MUST NOT override signature failures, revocation hits, trust bundle pinning, policy selection, verifier selection, policy expiry, verifier pinning, subject binding, fetch limits, or time range validity.

10.9 Tlog entry payload (typ = “acap/tlog”)
tlog_entry = {
“v”: “0.0.1”,
“typ”: “acap/tlog”,
“ts”: tstr,
“locator”: locator,
“entry_id”: tstr,
“prev_hash”: bstr,
“hash”: bstr,
“hash_alg”: tstr,
“objects”: [ + tstr ],
“anchor”: { * tstr => any } / null,
“ext”: { * tstr => any } / null
}

Rules:
	•	hash_alg MUST be “sha256”.
	•	hash MUST equal SHA-256(tlog_hash_input(prev_hash, objects)).
	•	The function tlog_hash_input is defined in Section 11.6.
	•	If policy.transparency.tlog_anchor_required is true, anchor MUST be non-null.

10.10 Minimal predicate types (v0.0.1)
ACAP v0.0.1 defines three minimal predicateType values:
	•	acap/predicate/provenance-min/v0.0.1
	•	acap/predicate/sbom-min/v0.0.1
	•	acap/predicate/tests-min/v0.0.1

If a predicateType is listed in policy.required_predicate_types, the predicate MUST match exactly the corresponding schema below, with unknown keys outside ext rejected.

10.11 provenance-min predicate schema
predicate = {
“v”: “0.0.1”,
“builder”: { “id”: tstr, “digest”: tstr },
“inputs”: {
“source”: [ + tstr ],
“lockfiles”: [ * tstr ],
“dependencies”: [ * tstr ],
“toolchain”: [ * tstr ],
“base_images”: [ * tstr ]
},
“network_used”: bool,
“tool”: { “name”: tstr, “version”: tstr / null },
“ext”: { * tstr => any } / null
}

Rules:
	•	builder.id MUST be ASCII and canonicalized (Section 11.3).
	•	builder.digest MUST be digest_str.
	•	If policy.hermetic is true, network_used MUST be false.
	•	If policy.inputs_required is true, inputs.source MUST be non-empty and all listed digest strings MUST be well-formed.
	•	Empty arrays are permitted only where marked with *.

10.12 sbom-min predicate schema
predicate = {
“v”: “0.0.1”,
“sbom_type”: tstr,
“sbom_digest”: tstr,
“ext”: { * tstr => any } / null
}

Rules:
	•	sbom_digest MUST be digest_str.

10.13 tests-min predicate schema
predicate = {
“v”: “0.0.1”,
“suite_id”: tstr,
“result”: tstr,
“artifacts_digests”: [ * tstr ],
“ext”: { * tstr => any } / null
}

Rules:
	•	result MUST be “pass” or “fail”.

	11.	Canonicalization rules (ACAP-CANON)

11.1 Digest string canonical form
All digest strings MUST be lowercase hex and MUST have the prefix “sha256:”.

11.2 Identifier canonical form
The following fields MUST be ASCII and MUST match the regex:
[a-z0-9][a-z0-9._-]{0,127}
Fields:
	•	policy_id
	•	verifier_id
	•	decision_id
	•	revocation_id
	•	exception_id
	•	builder.id
	•	target.env, target.cluster, target.namespace, target.workload, target.tenant, target.region

If a system uses different native casing, it MUST transform to this canonical form before comparison and hashing.

11.3 Repository canonical form
locator.repo and locator.subject_repo MUST be ASCII and MUST match:
[a-z0-9][a-z0-9._/-]{0,255}

11.4 Time and skew
Each policy specifies clock_skew_seconds_max.
A verifier MUST treat a time interval [valid_from, valid_to] as valid if:
Now + skew >= valid_from AND Now - skew <= valid_to
where skew = policy.decision.clock_skew_seconds_max.
If valid_to < valid_from, reject with TIME.INVALID_RANGE.

11.5 Concatenation encoding for hash inputs
When computing a hash over multiple fields, the verifier MUST use length-prefixed concatenation:
encode_bytes(x) = u32be(len(x)) || x
input = encode_bytes(field1) || encode_bytes(field2) || …

11.6 tlog hash input function
tlog_hash_input(prev_hash, objects):
	•	prev_hash is bytes.
	•	objects is an array of digest_str.
Compute:
SHA-256( encode_bytes(“ACAPv0.0.1”) ||
encode_bytes(prev_hash) ||
encode_bytes(concat( encode_bytes(obj_i_bytes) for each object digest string obj_i encoded as ASCII bytes in array order )) )

	12.	Retrieval and bounding rules (ACAP-FETCH)

12.1 Admission configuration (required)
Admission MUST provide:
	•	config.trust_bundle_digest (Section 9.3)
	•	policy selector: required_policy_ids[] or required_policy_digests[]
	•	verifier selector: required_verifier_ids[] or required_verifier_digests[]
	•	registry mapping: for each repo prefix, a registry host (if locator.registry is null)
	•	decision pointer from request: decision.digest or decision_id
	•	target of the request (Tadm) or null

12.2 Repository scope and locator rules
A verifier MUST resolve objects as follows:
	•	The Subject is resolved within subject_repo if known, else within the admission-provided subject repository.
	•	Referrers discovery is performed only within the subject repository scope.
	•	Objects referenced by digest (policy digest, verifier digest, evidence digests, revocation digests, exception digests, tlog digests) MUST be fetched from:
a) the same subject repository, OR
b) the object locator.repo if present and allowed by admission config, OR
c) a configured additional repo allowlist in admission config.
If an object cannot be fetched from allowed scopes: DENY with FETCH.MISSING.

12.3 Bounded discovery
A verifier MUST enforce policy.fetch limits:
	•	max_referrers_total
	•	max_referrers_per_type
	•	max_blob_bytes

If limits are exceeded at any stage: DENY with FETCH.OVER_LIMIT.

12.4 Prefer pointer-driven fetch
A verifier MUST NOT enumerate all referrers if not required.
It MUST:
	•	Fetch the pointed Decision first.
	•	Use the Decision to fetch policy, verifier, and the listed evidence digests.
	•	Fetch revocations and exceptions as required by policy and admission configuration.
Referrers enumeration MAY be used to find revocations/exceptions/tlog only if they are not directly referenced and policy requires them, and MUST still respect bounds.

12.5 Blob size limits
If any fetched blob exceeds max_blob_bytes: DENY with FETCH.OVER_LIMIT.

12.6 Mirror double-fetch (optional)
If policy.mirror.double_fetch_required is true:
	•	Each required blob fetch MUST be performed from two independent endpoints.
	•	The resulting bytes MUST be identical, or at minimum the resulting digest MUST match identically.
If mismatch or missing: DENY with MIRROR.MISMATCH.

12.7 Confidential evidence pointers (optional)
ACAP objects are public by default.
If confidential data is needed, Evidence predicates MUST store only an opaque pointer (under ext) and MUST NOT embed secrets.
If policy requires confidential evidence, admission MUST configure the confidential store resolver; if unresolved: DENY with EVIDENCE.CONFIDENTIAL_MISSING.
	13.	Evaluation semantics (ACAP-EVAL)

13.1 Policy selection (intent)
The effective policy set is determined by admission selectors, not by Decisions.
A Decision referencing policy not in the selector MUST be rejected (POLICY.NOT_SELECTED).

13.2 Verifier selection (intent)
The effective verifier set is determined by admission selectors, not by Decisions.
A Decision referencing verifier not in the selector MUST be rejected (VERIFIER.NOT_SELECTED).

13.3 Revocation requirement and freshness
If policy.revocation.required is true:
	•	The verifier MUST obtain revocation objects with a freshness not older than policy.revocation.max_age_seconds.
Freshness means the revocation object ts MUST satisfy:
Now - ts <= max_age_seconds, within clock skew.
If required revocations cannot be fetched fresh: DENY with REVOCATION.STALE.

13.4 Revocation precedence
If a revocation match hits:
	•	A key kid is invalid immediately.
	•	An object digest is invalid immediately.
Revocation hit MUST force denial even if other checks pass.

13.5 Evidence validity
Evidence is valid iff:
	•	Deterministic CBOR decoding succeeds.
	•	Schema matches acap/evidence plus predicate schema for required predicateTypes.
	•	COSE signature verifies.
	•	kid role is allowed for evidence (role “decision” MAY be used for evidence if policy allows; otherwise a dedicated role may be added only in a future version; in v0.0.1, evidence signatures MUST use role “decision”).
	•	Evidence subject[0].digest matches the Subject digest.
	•	Evidence is not revoked.

13.6 Required predicate types
For each predicateType in policy.required_predicate_types, there MUST exist at least one valid Evidence with that predicateType, otherwise DENY with EVIDENCE.REQUIRED_MISSING.

13.7 Allowed builders
If policy.allowed_builders is non-empty:
	•	At least one valid provenance-min Evidence MUST exist whose predicate.builder.id is in allowed_builders.
Otherwise DENY with BUILD.BUILDER_NOT_ALLOWED.

13.8 Hermetic and inputs closure
If policy.hermetic is true:
	•	provenance-min predicate.network_used MUST be false, otherwise DENY with BUILD.NON_HERMETIC.
If policy.inputs_required is true:
	•	provenance-min predicate.inputs.source MUST be non-empty and every listed digest MUST be well-formed, otherwise DENY with BUILD.INPUTS_CLOSURE_MISSING.

13.9 Exceptions
An exception applies iff:
	•	exception.subject.digest == Subject digest
	•	exception.policy_digest == policy.digest
	•	exception not expired (with skew)
	•	exception not revoked
	•	exception is multi-signed with exception threshold and correct roles
Applicable exceptions can remove violations only for codes listed in exception.allow_violation_codes.
Exceptions MUST NOT remove any code in policy.nonwaivable_codes.

13.10 Decision TTL, target binding, and challenge binding
A Decision is admissible iff:
	•	Its validity window is within policy.decision.validity_seconds_max (with skew).
	•	If policy.decision.target_required is true, Decision.target matches admission target (match rules below).
	•	If policy.decision.challenge_mode == “required”, Decision.challenge_hash and Decision.challenge_id are present and valid.

Target matching:
	•	For each target field in Decision.target that is non-null, it MUST equal the corresponding admission target field after canonicalization.
	•	If Decision.target is null when required: DENY with DECISION.TARGET_MISMATCH.

Challenge binding:
If challenge is used (mode optional or required), challenge_hash MUST equal:
SHA-256( encode_bytes(“ACAPv0.0.1”) ||
encode_bytes(subject.digest as ASCII bytes) ||
encode_bytes(policy.digest as ASCII bytes) ||
encode_bytes(verifier.digest as ASCII bytes) ||
encode_bytes(canonical(target) as ASCII bytes) ||
encode_bytes(challenge_bytes) )
If mode is required and missing: DENY with DECISION.CHALLENGE_REQUIRED.
challenge_id MUST be single-use; reuse MUST be rejected with DECISION.CHALLENGE_REUSED.

13.11 Subject.kind and index handling
If subject.kind == “oci.manifest”:
	•	Decision applies to that manifest only.

If subject.kind == “oci.index”:
	•	policy.index.index_mode MUST be evaluated:
a) “index-only”: Decision applies only to the index digest; platform manifests MUST be gated separately by their own Decisions.
b) “covers-all-platforms”: Decision.covered_subjects MUST list all referenced platform manifest digests. Admission MUST verify that covered_subjects matches the actual index content; mismatch DENY with SUBJECT.MISMATCH.

13.12 Decision chaining and anchors
If policy.rollback.chain_required is true:
	•	Each Decision MUST include prev_decision_digest and chain_height.
	•	chain_height MUST be >= policy.rollback.min_chain_height.
	•	The chain MUST be validated back to an anchor specified by policy.rollback.anchor_mode:
a) “decision”: policy.rollback.anchor_decision_digest MUST be present, and the chain MUST reach it exactly.
b) “tlog”: policy.rollback.anchor_tlog_digest MUST be present, and the relevant Decisions MUST appear in a valid tlog chain rooted at that anchor digest.
If chain cannot be validated: DENY with DECISION.CHAIN_BROKEN.

13.13 Transparency log requirements
If policy.transparency.tlog_required is true:
	•	The verifier MUST validate the tlog hash chain including the objects required by evaluation.
If policy.transparency.tlog_anchor_required is true:
	•	The tlog entry MUST include a non-null anchor object and the verifier MUST validate its presence; if missing: DENY with TLOG.MISSING.

13.14 Decision result is not authoritative if violations remain
Even if Decision.result == “allow”, the verifier MUST compute violations per policy.
If any non-waived violations remain, the final result MUST be DENY.
	14.	Verification algorithm (ACAP-VERIFY)

Input:
	•	Subject digest Dsub and subject.kind
	•	Admission target Tadm (may be null)
	•	Decision pointer: decision.digest or decision_id
	•	Current time Now
	•	Admission config (Section 12.1)
	•	Pinned trust_bundle_digest
	•	Optional challenge_bytes and challenge_id tracking

Output:
	•	allow or deny
	•	violation list (stable codes)

Algorithm (ordered, normative):
	1.	Fetch pinned Trust Bundle by digest (from configured location) and validate:
	•	deterministic CBOR
	•	COSE multi-signature meets trust-bundle threshold
	•	not expired (with skew)
If not pinned or invalid: DENY with TRUST.NOT_PINNED or TRUST.EXPIRED.
	2.	Resolve Subject descriptor:
	•	Fetch Subject bytes by digest from subject repository scope.
	•	Determine subject.kind (manifest or index) and verify it matches the provided kind; otherwise DENY with SUBJECT.KIND_INVALID.
	3.	Resolve the pointed Decision:
	•	If decision.digest is provided: fetch exactly that object.
	•	If decision_id is provided: resolve it uniquely within allowed repo scopes; if not unique or missing: DENY with DECISION.MISSING.
	•	If pointer is absent: DENY with DECISION.POINTER_MISSING.
	4.	Validate Decision object:
	•	deterministic CBOR, schema, COSE signatures, role and threshold per policy.required_signers.decision (after policy is fetched), no duplicate kids, typ header matches.
	•	valid_from/valid_to range valid and within TTL (with skew).
	•	subject.digest matches Dsub and subject.kind matches.
If any validation fails: DENY with SIG.INVALID or TIME.INVALID_RANGE or SUBJECT.MISMATCH.
	5.	Fetch and validate Policy referenced by Decision.policy.digest:
	•	Policy MUST be in admission policy selector (id or digest). If not: DENY with POLICY.NOT_SELECTED.
	•	Policy MUST be multi-signed with policy threshold and correct roles.
	•	Policy MUST not be expired (with skew).
	•	Policy chaining semantics: if prev_policy_digest is present, it MUST be fetchable and valid; if fetch fails: DENY with FETCH.MISSING.
If policy missing: DENY with POLICY.MISSING.
	6.	Fetch and validate Verifier referenced by Decision.verifier.digest:
	•	Verifier MUST be in admission verifier selector. If not: DENY with VERIFIER.NOT_SELECTED.
	•	Verifier MUST be multi-signed with verifier threshold and correct roles (or treated as data object signed by role “verifier”).
If verifier missing: DENY with VERIFIER.MISSING.
	7.	Fetch required revocations:
	•	If policy.revocation.required is true, fetch revocation objects with freshness per policy.
	•	Validate each revocation signature threshold and role.
If required revocation cannot be fetched fresh: DENY with REVOCATION.STALE.
	8.	Apply revocations:
	•	If any signature kid used in Trust Bundle or in any required object is revoked: treat that signature invalid.
	•	If any required object digest is revoked: DENY with REVOCATION.HIT.
	9.	Validate anti-replay requirements:
	•	If policy.decision.target_required is true, verify Decision.target matches Tadm.
	•	If policy.decision.challenge_mode is required, verify challenge_hash against challenge_bytes and verify challenge_id is unused.
Any failure: DENY with DECISION.TARGET_MISMATCH or DECISION.CHALLENGE_REQUIRED or DECISION.CHALLENGE_REUSED.
	10.	Validate chaining and anchors if required:

	•	Fetch and validate chain parents and/or tlog per anchor_mode.
Any failure: DENY with DECISION.CHAIN_BROKEN or TLOG.INVALID_CHAIN.

	11.	Fetch Evidence digests referenced by Decision.evidence_digests and validate Evidence:

	•	Enforce fetch limits.
	•	Validate signatures and schema.
	•	Enforce required predicateType schemas.
If any required evidence cannot be fetched: DENY with FETCH.MISSING.

	12.	Evaluate policy rules:

	•	Required predicate types present (EVIDENCE.REQUIRED_MISSING).
	•	Allowed builders (BUILD.BUILDER_NOT_ALLOWED).
	•	Hermetic and inputs closure (BUILD.NON_HERMETIC, BUILD.INPUTS_CLOSURE_MISSING).
	•	Index handling if subject.kind == “oci.index” (SUBJECT.MISMATCH).
	•	Mirror rule if required (MIRROR.MISMATCH).
	•	Transparency log if required (TLOG.MISSING, TLOG.INVALID_CHAIN).

	13.	Fetch and validate applicable Exceptions (bounded):

	•	Only exceptions that match Dsub and policy.digest are considered.
	•	Reject exceptions that attempt to waive nonwaivable codes or codes not in waivable_codes.

	14.	Apply exceptions (waive only allowed codes).
Remaining violations after waiver decide final result.
	15.	Final result:

	•	If no remaining violations and Decision.result == “allow”: ALLOW.
	•	Otherwise: DENY.
DENY MUST include at least one violation code.

	15.	Minimal HTTP API (optional)

If an implementation exposes HTTP verification:
	•	Errors SHOULD be returned in a Problem Details shaped JSON object with these fields:
type, title, status, detail, instance, and an “acap.violations” array containing violation objects.
This is informational; ACAP correctness MUST NOT depend on transport.

	16.	Violation code registry (v0.0.1)

Implementations MUST emit stable codes from this baseline set at minimum:

SIG.INVALID
SIG.THRESHOLD_NOT_MET

TRUST.NOT_PINNED
TRUST.EXPIRED
TRUST.KEY_NOT_ALLOWED

POLICY.MISSING
POLICY.EXPIRED
POLICY.NOT_SELECTED

VERIFIER.MISSING
VERIFIER.NOT_SELECTED

SUBJECT.MISMATCH
SUBJECT.KIND_INVALID

DECISION.POINTER_MISSING
DECISION.MISSING
DECISION.EXPIRED
DECISION.TARGET_MISMATCH
DECISION.CHAIN_BROKEN
DECISION.CHALLENGE_REQUIRED
DECISION.CHALLENGE_REUSED

REVOCATION.HIT
REVOCATION.STALE

EVIDENCE.REQUIRED_MISSING
EVIDENCE.SUBJECT_MISMATCH
EVIDENCE.CONFIDENTIAL_MISSING

BUILD.BUILDER_NOT_ALLOWED
BUILD.NON_HERMETIC
BUILD.INPUTS_CLOSURE_MISSING

FETCH.MISSING
FETCH.OVER_LIMIT

MIRROR.MISMATCH

TLOG.MISSING
TLOG.INVALID_CHAIN

TIME.INVALID_RANGE

Codes are stable identifiers. Messages may vary, but codes MUST NOT.
	17.	Completeness checklist (normative)

A v0.0.1 conforming deployment MUST ensure:
	•	Policy intent cannot be selected by Decisions:
Admission pins policy selector; policy outside selector is rejected.
	•	Verifier intent cannot be selected by Decisions:
Admission pins verifier selector; verifier outside selector is rejected.
	•	Trust bootstrapping is unambiguous:
Admission pins trust bundle digest; rotation requires cross-signing.
	•	Replay is prevented:
Decisions have TTL and target binding; optional challenge binding can be required.
	•	Rollback is resisted:
Decision chaining is anchored (decision anchor or tlog anchor) when enabled.
	•	Single key compromise is mitigated:
Threshold multi-signatures with strict role separation; duplicate kids rejected.
	•	Emergency stop exists:
Revocation artifacts are enforced with strict precedence and freshness rules.
	•	Exceptions do not become a backdoor:
Exceptions are scoped, time-bounded, multi-signed, and cannot waive nonwaivable codes.
	•	Supply chain integrity is enforced:
Required predicate types exist; allowed builders enforced; hermetic and inputs closure enforced.
	•	Registry DoS is mitigated:
Bounded discovery and bounded fetch; pointer-driven retrieval is preferred.
	•	Optional distribution hardening can be enabled:
Mirror double-fetch.
	•	Optional auditability can be enabled:
Local tlog chain, optionally anchored, and required by policy.

End of ACAP Specification (SSOT) v0.0.1
