Title: storyed Integrated Spec v0.0.1 (Single Binary, Offline, CAS+DAG, Scene-level, Markdown, Lexo-rank, UTF-8, Pinned UI, Git/Worktree Interop)
	0.	Status

0.1 This document defines the frozen v0.0.1 integrated specification for storyed: a self-contained editor server for stories/documents and its pinned browser UI, including security headers, UI behavior, latency budgets, deterministic export/import, and Git/worktree interop.

0.2 Target: single host, offline/air-gapped capable, non-realtime collaboration via branches and merge requests (MR), with optional Git ecosystem interop via deterministic worktree export and import.

0.3 Normative keywords: MUST, MUST NOT, SHOULD, MAY.

0.4 v0.0.1 compliant implementations MUST implement exactly the ABIs and semantics in this document. No forward-compat behavior is defined in v0.0.1.
	1.	Goals

1.1 Provide a time-series reading UX (chapters -> scenes) on top of a Git-like internal model.

1.2 Provide immutable content-addressed storage (CAS) for blobs/trees/commits and a DAG history.

1.3 Provide scene-level diff/merge and first-class scene operations (create/edit/move/reorder/split/merge/delete).

1.4 Provide single-binary deployment with embedded UI assets and local storage only.

1.5 Provide deterministic export/import that restores identical repository state, including identical object IDs and identical refs and metadata state.

1.6 Provide deterministic export bytes for identical repository state (byte-identical archive for identical state) using the deterministic archive rules in Section 11.

1.7 Provide pinned UI implementation and security profile such that UI correctness does not depend on external resources.

1.8 Provide pinned UI behavior, state machines, and measurable latency budgets for core flows.

1.9 Provide deterministic worktree export/import for Git and external editor interop with strong safety guards (no silent ref updates).

1.10 Provide predictable conflict handling such that 409 conflicts are normal flows with explicit user choices.

1.11 Eliminate operational rituals by standardizing operation receipts, preflight checks, and conflict-resolution operations that remain explicit but one-step.
	2.	Non-goals

2.1 Real-time concurrent editing (OT/CRDT).

2.2 Payments/marketplace/licensing enforcement.

2.3 Multi-region HA and global-scale operations.

2.4 Git wire protocol compatibility.

2.5 Required DAG visualization as a primary workflow.

2.6 Background or implicit server-side queueing of mutations while offline (mutations must be explicit).

2.7 Silent auto-resolution of 409 conflicts.

2.8 Automatic background maintenance (GC/pack/verify) without explicit operator action.
	3.	Terminology

3.1 Repo: repository (one work/document).

3.2 CAS: content-addressed storage (immutable objects).

3.3 Blob: raw bytes object stored in CAS (identified by blob_id).

3.4 Tree: deterministic snapshot mapping absolute path -> blob_id (identified by tree_id).

3.5 Commit: DAG node referencing a tree and parent commits (identified by commit_id).

3.6 Ref: mutable pointer name -> commit_id (branch/tag).

3.7 Chapter: container of scenes for reading UX.

3.8 Scene: minimum editorial unit (diff/merge/cherry-pick granularity).

3.9 MR: merge request (base_ref + head_ref + review + merge).

3.10 Actor: authenticated user performing a request.

3.11 Draft: browser-local uncommitted edits persisted in IndexedDB (not visible to others).

3.12 Staged: browser-local change set selected for publish; stable and reviewable; derived from Draft explicitly.

3.13 Baseline: last loaded committed scene state used for dirty detection and conflict flows.

3.14 Head: current commit_id of a ref.

3.15 View id: the identifier a UI read view is rendering (either ref name or commit id).

3.16 Optimistic UI: UI updates visible state before server acknowledgement, then reconciles.

3.17 Worktree: deterministic filesystem projection of a repo/ref view for external tools (editors/Git).

3.18 Worktree guard: deterministic metadata file binding a worktree to (repo_id, ref_name, base_commit_id).

3.19 Operation receipt: deterministic response record for a mutation, describing what changed and which ref/head was affected (Section 9.4).

3.20 Canonical JSON blob: a JSON object validated and canonicalized per Section 5 before being stored as a Blob.

3.21 Mutating endpoint: any HTTP endpoint that modifies persistent server state, including ref updates, MR updates, user/ACL changes, import, worktree import, and maintenance actions.
	4.	Packaging, Runtime, and Commands (L3)

4.1 Single executable

4.1.1 The system MUST be delivered as a single executable file.

4.1.2 The executable MUST serve both:
4.1.2.1 HTTP API (JSON and selected binary downloads/uploads)
4.1.2.2 UI (static SPA assets embedded in the binary)

4.2 Offline requirement

4.2.1 The process MUST run offline (no mandatory external network calls).

4.2.2 UI assets MUST NOT require external CDNs for correctness.

4.3 Commands (normative interface)

4.3.1 The executable MUST support:

4.3.1.1 serve –data-dir  –listen addr:port￼ [–config ]
4.3.1.2 export –data-dir  –out <file.tar.zst> [–repo <repo_id>|–all]
4.3.1.3 import –data-dir  –in <file.tar.zst> [–run-verify 0|1] [–dry-run]
4.3.1.4 maintenance rebalance –data-dir  –repo <repo_id> –chapter <chapter_id> –ref <ref_name>
4.3.1.5 export-md –data-dir  –repo <repo_id> –ref <ref_name|commit_id> –out 
4.3.1.6 import-md –data-dir  –repo <repo_id> –ref <ref_name> –in  –expected-head <commit_id>
4.3.1.7 worktree add –data-dir  –repo <repo_id> –ref <ref_name> –path  –expected-head <commit_id|null>
4.3.1.8 worktree pull –data-dir  –path 
4.3.1.9 worktree push –data-dir  –path  –expected-head <commit_id>
4.3.1.10 lint –in  [–format text|json]
4.3.1.11 maintenance verify –data-dir  [–repo <repo_id>|–all]
4.3.1.12 maintenance repair-order –data-dir  –repo <repo_id> –chapter <chapter_id> –ref <ref_name> –mode derive
4.3.1.13 maintenance prune-idempotency –data-dir  –older-than 
4.3.1.14 maintenance init-admin –data-dir  –handle  –password 
4.3.1.15 maintenance reset-password –data-dir  –user-id <user_id> –password 

4.3.2 Implementations MUST NOT change the semantics of the above.

4.3.3 Implementations MAY provide additional commands, but MUST NOT change the behavior of required commands.
	5.	Canonicalization and Content Safety (SSOT)

5.1 UTF-8 and normalization

5.1.1 All user-visible text fields MUST be UTF-8 in API and storage.

5.1.2 The server MUST reject invalid UTF-8.

5.1.3 The server MUST normalize all stored text fields to Unicode NFC prior to canonicalization and hashing.

5.2 Forbidden characters

5.2.1 The server MUST reject the following characters in any stored text field:
5.2.1.1 U+0000 (NUL)
5.2.1.2 U+0001..U+001F and U+007F (control characters)
EXCEPT:
5.2.1.3 LF (U+000A) is permitted inside body_md and commit.message
5.2.1.4 TAB (U+0009) is permitted inside body_md

5.2.2 The server MUST reject bidi control characters:
5.2.2.1 U+202A..U+202E and U+2066..U+2069

5.3 Length limits (configurable)

5.3.1 The server MUST enforce configurable length limits.

5.3.2 Defaults SHOULD be:
5.3.2.1 chapter.title <= 256 code points
5.3.2.2 scene.title <= 256 code points
5.3.2.3 tags[i] <= 64 code points
5.3.2.4 entities[i] <= 128 code points
5.3.2.5 commit.message <= 2048 code points
5.3.2.6 body_md <= 5 MiB (bytes after LF normalization)
5.3.2.7 username/handle <= 64 code points

5.4 Error detail requirements for text failures

5.4.1 For any rejection due to 5.1 or 5.2, the server MUST return 400 with:
5.4.1.1 code: “TEXT_INVALID”
5.4.1.2 details: { field: , reason: , offset: <int|null> }

5.4.2 For any rejection due to empty-string rules (5.6), reason MUST be “EMPTY_STRING”.

5.5 Newline normalization

5.5.1 The server MUST normalize stored body_md newlines to LF (”\n”).

5.5.2 Incoming CRLF and CR MUST be converted to LF before canonicalization and storage.

5.5.3 commit.message MUST be normalized CRLF/CR -> LF before hashing.

5.6 Canonical JSON policy (required)

5.6.1 For any Canonical JSON blob stored in CAS (Chapter, Scene, order.json, manifest.json, ui_manifest.json, audit.details_json, worktree guard JSON, and any other JSON blobs declared canonical by this spec), canonicalization MUST be:

5.6.1.1 Validate UTF-8.
5.6.1.2 Normalize all text values to NFC.
5.6.1.3 Reject forbidden characters after NFC (5.2).
5.6.1.4 Apply LF normalization where applicable:
5.6.1.4.1 body_md
5.6.1.4.2 commit.message
5.6.1.5 Canonical JSON blobs MUST NOT contain JSON numbers.
5.6.1.5.1 All numeric values MUST be encoded as decimal strings in v0.0.1.
5.6.1.5.2 If any JSON number is present, reject with 400 code “JSON_NUMBER_FORBIDDEN” and details { path:  }.
5.6.1.6 Canonicalize set-like arrays (required) for:
5.6.1.6.1 Scene.tags, Scene.entities, constraints.flags, Chapter.tags, constraints.flags
5.6.1.6.2 Remove duplicates by exact bytewise equality of UTF-8 bytes after NFC.
5.6.1.6.3 Sort ascending by bytewise comparison of UTF-8 bytes.
5.6.1.6.4 Reject empty strings after NFC.
5.6.1.7 Apply JSON Canonicalization Scheme (JCS, RFC 8785).

5.6.2 The API MUST return canonicalized JSON when returning stored canonical JSON blobs.

5.7 Markdown safety (required)

5.7.1 body_md MUST be rendered as CommonMark-compatible Markdown.

5.7.2 Raw HTML in Markdown MUST be disabled in v0.0.1 (MUST NOT be rendered as HTML).

5.7.3 Rendered output MUST be XSS-safe. The UI MUST NOT execute scripts from rendered content.

5.7.4 The renderer MUST NOT emit links with forbidden schemes (case-insensitive):
5.7.4.1 javascript:
5.7.4.2 data:
5.7.4.3 vbscript:

5.7.5 The renderer SHOULD allow only:
5.7.5.1 http
5.7.5.2 https
5.7.5.3 mailto
5.7.5.4 relative URLs (including fragment-only)

5.7.6 Images (required alignment with CSP)
5.7.6.1 The Markdown renderer MUST NOT emit  elements for external URLs.
5.7.6.2 v0.0.1 default behavior MUST be: render images as plain links (no ), regardless of URL.
	6.	Repository Model, Identifiers, and Refs

6.1 Stable IDs (textual form)

6.1.1 repo_id, chapter_id, scene_id, mr_id, user_id, event_id MUST be UUIDv7 strings.

6.1.2 Canonical UUID string form MUST be lowercase hex with hyphens: 8-4-4-4-12.

6.2 Content IDs

6.2.1 blob_id, tree_id, commit_id MUST be lowercase hex sha256 strings of length 64.

6.2.2 Internally, tree/commit objects store raw sha256 bytes(32); externally they are represented as hex strings.

6.3 Refs

6.3.1 Branch ref names MUST use refs/heads/.

6.3.2 Tag ref names MUST use refs/tags/.

6.3.3 <ref_segment> MUST match regex: [A-Za-z0-9._-]{1,64}.

6.3.4 Ref names MUST be case-sensitive.

6.4 Repository creation

6.4.1 Creating a repo MUST:
6.4.1.1 allocate repo_id
6.4.1.2 create an initial empty tree whose entries array is empty
6.4.1.3 create an initial commit referencing that empty tree
6.4.1.4 create refs/heads/main pointing to that commit

6.5 Ref updates

6.5.1 Ref updates MUST be atomic (single SQLite transaction).

6.5.2 Ref update MUST implement compare-and-swap semantics when a caller supplies expected_old_commit_id.

6.5.3 If expected_old_commit_id is non-null and does not match the current ref value, the server MUST return 409 conflict.
	7.	CAS Objects, Tree Layout, and Canonical Encodings (SSOT)

7.1 Blob

7.1.1 Blob is raw bytes stored exactly as written (post-canonicalization if the blob is a canonical JSON blob).

7.1.2 blob_id = sha256(blob_bytes).

7.2 Tree object (Canonical CBOR map)

7.2.1 Tree represents a flat mapping from absolute path -> blob_id.

7.2.2 Tree CBOR map MUST contain exactly:
7.2.2.1 “type” : “tree”
7.2.2.2 “entries” : array

7.2.3 Each entry MUST be a CBOR map containing exactly:
7.2.3.1 “path” : text (absolute path, ASCII; see 7.4)
7.2.3.2 “id” : bytes(32) (raw sha256 of the referenced blob)

7.2.4 “entries” MUST be sorted by “path” ascending using bytewise comparison of UTF-8 bytes.

7.2.5 Duplicate “path” values MUST be rejected (400).

7.2.6 Tree MUST NOT contain any additional fields in v0.0.1.

7.3 Commit object (Canonical CBOR map)

7.3.1 Commit CBOR map MUST contain exactly:
7.3.1.1 “type” : “commit”
7.3.1.2 “tree” : bytes(32)
7.3.1.3 “parents” : array of bytes(32) (0..n)
7.3.1.4 “author” : map { “user_id”: text(UUIDv7), “handle”: text|null }
7.3.1.5 “message” : text
7.3.1.6 “created_at” : int (unix seconds, UTC)

7.3.2 Commit MUST NOT include any additional fields (metadata fields are forbidden in v0.0.1).

7.3.3 “parents” ordering MUST be deterministic: sort by raw bytes(32) ascending.

7.3.4 commit_id depends on created_at and message; identical trees can yield different commit_id values.

7.3.5 Implementations MAY store additional operational metadata in meta.db, but MUST NOT affect commit_id/tree_id derivation.

7.4 Tree paths (normative)

7.4.1 Paths MUST start with “/”.

7.4.2 Segments MUST NOT be empty.

7.4.3 Segments MUST NOT be “.” or “..”.

7.4.4 Paths MUST NOT contain backslash “".

7.4.5 Paths MUST be ASCII-only in v0.0.1.

7.4.6 Implementations MUST reject any attempt to store objects under non-conforming paths.

7.5 Repository tree layout (normative for v0.0.1)

7.5.1 The repository tree layout MUST be:
7.5.1.1 /chapters/<chapter_id>.json -> blob (Chapter JSON)
7.5.1.2 /chapters/<chapter_id>/scenes/<scene_id>.json -> blob (Scene JSON)
7.5.1.3 /chapters/<chapter_id>/order.json -> blob (Order JSON, Section 8.7)

7.5.2 <chapter_id> and <scene_id> MUST be canonical UUIDv7 strings (lowercase).

7.6 Canonical encoding rules

7.6.1 Tree and Commit objects MUST be encoded using Canonical CBOR (RFC 8949 canonical rules).

7.6.2 Canonical CBOR output bytes are hashed for tree_id/commit_id.

7.6.3 The server MUST be the source of truth for canonicalization.
	8.	Ordering (Fixed, v0.0.1) (SSOT)

8.1 Lexo-rank keys

8.1.1 order_key MUST be exactly 16 characters.

8.1.2 Characters MUST be base62: 0-9A-Za-z.

8.1.3 Comparison MUST be lexicographic by ASCII byte order over the 16 bytes.

8.2 Digit mapping (normative)

8.2.1 DIGITS: “0123456789”
8.2.2 UPPER:  “ABCDEFGHIJKLMNOPQRSTUVWXYZ”
8.2.3 LOWER:  “abcdefghijklmnopqrstuvwxyz”

8.2.4 digit(“0”)=0 .. digit(“9”)=9, digit(“A”)=10 .. digit(“Z”)=35, digit(“a”)=36 .. digit(“z”)=61.

8.3 Sentinels

8.3.1 left_sentinel  = “0000000000000000”
8.3.2 right_sentinel = “zzzzzzzzzzzzzzzz”
8.3.3 default_mid_digit = “U” (digit 30)

8.4 Between(left, right) (deterministic, fixed width)

8.4.1 Inputs:
8.4.1.1 left_key MAY be null (means -infinity -> left_sentinel)
8.4.1.2 right_key MAY be null (means +infinity -> right_sentinel)
8.4.1.3 If both non-null, left_key MUST be < right_key.

8.4.2 Output MUST be a 16-char key strictly between.

8.4.3 Algorithm:
8.4.3.1 Let L = (left_key or left_sentinel), R = (right_key or right_sentinel).
8.4.3.2 For i in 0..15:
li = digit(L[i])
ri = digit(R[i])
If li == ri: output L[i] at i and continue
If ri - li >= 2:
output[i] = value(floor((li+ri)/2))
output[i+1..15] = default_mid_digit
return output
If ri - li == 1:
output[i] = L[i]
continue
8.4.3.3 If the loop completes, there is no space at width 16:
return error ORDER_KEY_SPACE_EXHAUSTED (HTTP 409).

8.4.4 Between(null, null) MUST return “UUUUUUUUUUUUUUUU”.

8.5 Rebalance(chapter_id) (deterministic)

8.5.1 Rebalance MUST assign fresh order_key values to all scenes within a chapter and MUST create a new commit.

8.5.2 Rebalance input order is the current stable order defined by order.json at the base commit.

8.5.3 Rebalance output algorithm:
8.5.3.1 Define GAP = 62^4
8.5.3.2 For scene index i from 1..n:
key_num = i * GAP
order_key = base62_encode_fixed(key_num, width=16, pad=“0”)

8.5.4 base62_encode_fixed encodes most significant digit first; left-pad with “0” to width 16.

8.6 Stable sorting (UI)

8.6.1 Chapter order: (chapter.order_key, chapter_id).

8.6.2 Scene order: order.json order, with tie-break by (scene.order_key, scene_id) if needed for resilience.

8.7 Order JSON (normative)

8.7.1 /chapters/<chapter_id>/order.json MUST exist for each chapter that contains at least one scene.

8.7.2 order.json MUST be a Canonical JSON blob containing exactly:
8.7.2.1 “chapter_id”: string(UUIDv7)
8.7.2.2 “items”: array of object

8.7.3 Each item MUST be a JSON object containing exactly:
8.7.3.1 “scene_id”: string(UUIDv7)
8.7.3.2 “order_key”: string(16 base62)

8.7.4 items MUST NOT contain duplicate scene_id.

8.7.5 For any scene in the chapter tree, order.json MUST contain an item for that scene_id.

8.7.6 order.json MUST NOT reference any scene_id not present in the chapter tree.

8.7.7 If order.json is missing or invalid for a chapter with scenes, the server MUST return 500 with code “ORDER_CORRUPT” and MUST NOT guess.

8.8 Rebalance pressure (non-canonical)

8.8.1 Implementations SHOULD compute and expose per-chapter “rebalance_pressure” metrics derived from order_key density, without changing canonical objects.

8.8.2 Metrics MUST NOT affect commit_id/tree_id derivation.
	9.	Mutations Contract (SSOT)

9.1 Guarded ref-head semantics (required)

9.1.1 All endpoints that mutate a repo ref (including MR merge and all ops/* endpoints, publish flows, worktree import, and repair-order) MUST accept an optional expected_head_commit_id.

9.1.2 If expected_head_commit_id is provided and does not match the current head commit_id of the target ref at operation start, the server MUST return 409 with code “REF_HEAD_MISMATCH” and details at least:
9.1.2.1 { “ref”: <ref_name>, “expected”: <sha256_hex>, “actual”: <sha256_hex> }.

9.1.3 Deterministic failure rule: If a guarded mutation fails with REF_HEAD_MISMATCH, and the same request is replayed with identical inputs and identical ref head, the server MUST return the same error code and details.

9.2 One-commit rule for ops/* (required)

9.2.1 All ops/* endpoints MUST:
9.2.1.1 accept “ref” (refs/heads/* only) as the target ref
9.2.1.2 accept optional “expected_head_commit_id”
9.2.1.3 produce exactly one new commit on success
9.2.1.4 update the target ref to that commit atomically
9.2.1.5 return previous_head_commit_id
9.2.1.6 return an operation receipt (9.4)

9.2.2 If an op requires ORDER_KEY_SPACE_EXHAUSTED handling:
9.2.2.1 the server MUST compute the rebalance result and the requested op result against that rebalance in-memory
9.2.2.2 the server MUST commit the final state as a single commit
9.2.2.3 the server MUST NOT create an intermediate visible commit for the rebalance

9.3 Idempotency-Key (required)

9.3.1 All mutating requests MUST support Idempotency-Key and the server MUST enforce it.

9.3.2 For any mutating endpoint, if Idempotency-Key is missing, server MUST return 400 code “IDEMPOTENCY_REQUIRED”.

9.3.3 The server MUST store idempotency results for at least 24 hours (configurable), keyed by:
9.3.3.1 (actor_id, method, path, idempotency_key)

9.3.4 On idempotent replay, the server MUST return the stored response status code and body, and MUST NOT perform the mutation again.

9.3.5 The server MUST persist idempotency records for deterministic failures as well:
9.3.5.1 Persist: 200/201/204, 409, 429, 507
9.3.5.2 Do NOT persist: 500

9.3.6 The persisted response body MUST be exactly the response body bytes sent to the client.

9.3.7 The persisted response MUST NOT include or depend on per-request random values.

9.4 Operation receipt (required)

9.4.1 Any successful mutation that updates a ref MUST return a receipt object:
9.4.1.1 {
“op_name”: string,
“repo_id”: UUIDv7,
“ref”: string,
“expected_head_commit_id”: sha256_hex|null,
“head_before”: sha256_hex,
“head_after”: sha256_hex,
“commit_id”: sha256_hex,
“changed_paths”: [string…],
“changed_scene_ids”: [UUIDv7…],
“request_id”: string|null
}

9.4.2 changed_paths MUST be sorted ascending (bytewise ASCII).

9.4.3 changed_scene_ids MUST be sorted ascending (bytewise UUID string).

9.4.4 receipt MUST be deterministic given identical inputs and repository state.

9.4.5 receipt MUST NOT affect CAS IDs.

9.4.6 request_id MUST be null in v0.0.1 to preserve determinism for idempotency replay.

9.5 Preflight (required)

9.5.1 The server MUST provide preflight endpoints that validate inputs and predict effects without mutating state:
9.5.1.1 preflight-publish (single scene or staged set)
9.5.1.2 preflight-worktree-import

9.5.2 Preflight MUST:
9.5.2.1 validate text rules (Section 5), markdown rules (Section 5.7), and ordering consistency (Section 8.7) as applicable
9.5.2.2 validate expected_head_commit_id if provided; on mismatch return 409 REF_HEAD_MISMATCH
9.5.2.3 return predicted changed_paths and changed_scene_ids deterministically
9.5.2.4 return stable error codes and locations similar to lint (Section 12.6)

9.5.3 Preflight MUST NOT create blobs, trees, commits, audit rows, or idempotency records.
	10.	Diff, Merge Requests, and Conflicts (SSOT)

10.1 Diff semantics (scene-first)

10.1.1 Diff(base, head) MUST report:
10.1.1.1 chapters: added/deleted/modified/reordered
10.1.1.2 scenes: added/deleted/modified/moved/reordered

10.1.2 Scene classification:
10.1.2.1 modified: same scene_id exists in both trees, blob differs
10.1.2.2 added: only in head tree
10.1.2.3 deleted: only in base tree
10.1.2.4 moved: same scene_id exists in both trees, but chapter_id/path differs
10.1.2.5 reordered: order differs per order.json or order_key differs

10.1.3 Modified scene diff MUST include:
10.1.3.1 body_md line diff (LF normalized)
10.1.3.2 structured diff for: title, tags, entities, constraints, order_key, chapter_id, provenance

10.1.4 Line diff algorithm (required minimum)
10.1.4.1 The server MUST provide a stable, deterministic line-based diff.
10.1.4.2 Output MUST be deterministic given the same inputs.
10.1.4.3 If multiple minimal diffs are possible, tie-break MUST be deterministic:
10.1.4.3.1 prefer earliest (leftmost) matching anchors
10.1.4.3.2 prefer shorter edit scripts if multiple remain

10.1.5 Diff API split (required)
10.1.5.1 The server MUST provide:
10.1.5.1.1 summary diff: classification only (no body diff)
10.1.5.1.2 scene diff: per-scene body and structured diffs
10.1.5.2 summary diff MUST be O(number of changed paths) and MUST NOT compute body line diffs.

10.2 Merge Requests (MR)

10.2.1 MR record MUST include:
10.2.1.1 mr_id, repo_id
10.2.1.2 base_ref, head_ref
10.2.1.3 base_commit_id (snapshot at MR creation)
10.2.1.4 status: open|merged|closed
10.2.1.5 checks: array of { name, status, details? }

10.2.2 Merge modes:
10.2.2.1 fast-forward: ref update only
10.2.2.2 merge: 3-way merge commit (new commit with 2+ parents)
10.2.2.3 squash: single new commit capturing head changes

10.2.3 merge_base (deterministic)
10.2.3.1 merge_base MUST be computed as a common ancestor in the commit DAG.
10.2.3.2 If multiple common ancestors exist, choose deterministically:
10.2.3.2.1 Define dist(X, A) as the length of the shortest parent-edge path from commit X to ancestor A.
10.2.3.2.2 For each common ancestor A, compute tuple:
T(A) = (
max(dist(base_head, A), dist(head_head, A)),
dist(base_head, A) + dist(head_head, A),
commit_id(A)
)
10.2.3.2.3 Choose A with lexicographically smallest T(A).

10.2.4 Conflict detection (scene-level)
10.2.4.1 content conflict: body_md changed on both sides since merge_base
10.2.4.2 meta conflict: title/tags/entities/constraints/provenance changed on both sides since merge_base
10.2.4.3 order conflict: order changed on both sides since merge_base

10.2.5 Default resolution policy
10.2.5.1 order conflict defaults to head, but MUST be overrideable per scene (base/head/manual)
10.2.5.2 content/meta conflicts MUST require explicit resolution by user (MR UI/API)

10.2.6 Merge result
10.2.6.1 Conflict resolutions MUST result in a new commit (immutable history).
10.2.6.2 Merge MUST update base_ref to the resulting commit_id atomically.
10.2.6.3 MR merge MUST honor expected_head_commit_id semantics for base_ref updates (Section 9.1).
10.2.6.4 MR merge MUST return an operation receipt (Section 9.4).
	11.	Storage Layout and Database (meta.db)

11.1 data-dir contents

11.1.1 data-dir MUST contain:
11.1.1.1 /meta.db : SQLite database
11.1.1.2 /objects/ : CAS objects (immutable) and optional pack files (Section 11.7)

11.1.2 data-dir MAY contain:
11.1.2.1 /tmp/
11.1.2.2 /logs/ (optional)

11.2 SQLite mode

11.2.1 SQLite SHOULD use WAL mode for performance and crash safety.

11.2.2 The server MUST configure a finite busy timeout for SQLite operations:
11.2.2.1 busy_timeout_ms MUST be configurable, default SHOULD be 2000ms
11.2.2.2 If a mutating transaction cannot begin or commit due to SQLITE_BUSY beyond busy_timeout_ms, the server MUST fail with HTTP 503 code “DB_BUSY” (Section 16.7)

11.3 CAS object file path (loose objects)

11.3.1 Loose CAS objects MUST be stored under:
11.3.1.1 /objects/sha256//
where  is the first 2 hex chars of  and  is the full lowercase sha256 hex string.

11.3.2 The system MUST NOT overwrite an existing loose CAS object file.

11.3.3 The system MUST write loose objects atomically (write temp then rename).

11.3.4 The system MUST fsync the final object file and its parent directory, or otherwise ensure durability equivalent to atomic rename on the target platform.

11.4 Garbage collection

11.4.1 GC MAY be omitted in v0.0.1.

11.4.2 If implemented, GC MUST be explicit (manual) and MUST NOT run automatically.

11.4.3 GC MUST NOT delete any object reachable from any ref, MR base_commit_id, or audit log references.

11.5 meta.db minimum schema (required)

11.5.1 Required tables:
11.5.1.1 repos(repo_id TEXT PRIMARY KEY, name TEXT NULL, created_at INTEGER NOT NULL)
11.5.1.2 refs(repo_id TEXT NOT NULL, ref_name TEXT NOT NULL, commit_id BLOB NOT NULL, updated_at INTEGER NOT NULL, PRIMARY KEY(repo_id, ref_name))
11.5.1.3 mrs(mr_id TEXT PRIMARY KEY, repo_id TEXT NOT NULL, base_ref TEXT NOT NULL, head_ref TEXT NOT NULL, base_commit_id BLOB NOT NULL, status TEXT NOT NULL, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL)
11.5.1.4 audit(event_id TEXT PRIMARY KEY, ts INTEGER NOT NULL, actor_id TEXT NOT NULL, action TEXT NOT NULL, repo_id TEXT NULL, details_json TEXT NOT NULL)
11.5.1.5 users(user_id TEXT PRIMARY KEY, handle TEXT UNIQUE NOT NULL, created_at INTEGER NOT NULL, password_hash BLOB NOT NULL, password_params_json TEXT NOT NULL)
11.5.1.6 repo_acl(repo_id TEXT NOT NULL, user_id TEXT NOT NULL, role TEXT NOT NULL, PRIMARY KEY(repo_id, user_id))
11.5.1.7 idempotency(actor_id TEXT NOT NULL, method TEXT NOT NULL, path TEXT NOT NULL, key TEXT NOT NULL, created_at INTEGER NOT NULL, response_status INTEGER NOT NULL, response_body BLOB NOT NULL, PRIMARY KEY(actor_id, method, path, key))
11.5.1.8 sessions(session_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, created_at INTEGER NOT NULL, expires_at INTEGER NOT NULL)

11.5.2 Atomicity
11.5.2.1 Ref updates, MR status changes, idempotency records, and session changes MUST be transactionally consistent.

11.5.3 Required indexes (performance)
11.5.3.1 audit(ts, event_id)
11.5.3.2 audit(repo_id, ts, event_id)
11.5.3.3 mrs(repo_id, status, updated_at)

11.6 Audit log rules

11.6.1 Audit log MUST be append-only.

11.6.2 Audit MUST record at least:
11.6.2.1 ref updates
11.6.2.2 commit creation
11.6.2.3 MR open/merge/close
11.6.2.4 user/role changes
11.6.2.5 export/import
11.6.2.6 maintenance operations (rebalance, pack, verify, GC if any)
11.6.2.7 worktree import operations
11.6.2.8 create/delete chapter/scene

11.6.3 Audit entries MUST include:
11.6.3.1 event_id (UUIDv7)
11.6.3.2 ts (unix seconds; stored in DB as integer)
11.6.3.3 actor_id (UUIDv7)
11.6.3.4 action (string)
11.6.3.5 repo_id (UUIDv7|null)
11.6.3.6 details_json (Canonical JSON blob per Section 5.6)

11.7 CAS pack format (deterministic, v0.0.1)

11.7.1 Packs reduce filesystem overhead by grouping many objects into fewer files while preserving object immutability and IDs.

11.7.2 Pack directory layout
11.7.2.1 Pack files MUST be stored under: /objects/packs/
11.7.2.2 Pack MUST consist of:
11.7.2.2.1 pack data file: pack-<pack_id>.dat
11.7.2.2.2 pack index file: pack-<pack_id>.idx
11.7.2.3 pack_id MUST be sha256_hex of the canonical bytes of the index file.

11.7.3 Pack entries
11.7.3.1 The pack index MUST map object_id -> (kind, offset, length) into the data file.
11.7.3.2 object_id is sha256_hex of the object payload bytes for that object (Blob bytes, Tree CBOR bytes, Commit CBOR bytes).
11.7.3.3 kind MUST be a u8 with fixed meaning in v0.0.1:
11.7.3.3.1 0 = blob
11.7.3.3.2 1 = tree
11.7.3.3.3 2 = commit
11.7.3.3.4 3..255 reserved (MUST NOT appear in v0.0.1 packs)
11.7.3.4 The data file MUST contain concatenated object records, each preceded by a fixed header:
11.7.3.4.1 magic “SDPK” (4 bytes)
11.7.3.4.2 version u32 = 1
11.7.3.4.3 kind u8
11.7.3.4.4 object_id raw bytes(32)
11.7.3.4.5 length u64 (little-endian)
11.7.3.4.6 payload bytes(length)
11.7.3.5 The server MUST validate object_id matches sha256(payload) when reading, unless a verified cache is used.
11.7.3.6 For kind=1 and kind=2, the payload MUST be the canonical CBOR bytes for tree/commit respectively.

11.7.4 Pack determinism
11.7.4.1 pack entries MUST be sorted by (kind, object_id) ascending:
11.7.4.1.1 kind ascending numeric
11.7.4.1.2 object_id ascending by bytewise ordering over raw 32 bytes
11.7.4.2 The index file MUST be a canonical binary encoding:
11.7.4.2.1 magic “SDIX” (4 bytes)
11.7.4.2.2 version u32 = 1
11.7.4.2.3 count u64
11.7.4.2.4 repeated count times:
kind u8
object_id raw bytes(32)
offset u64
length u64
11.7.4.3 pack_id = sha256(index_bytes).
11.7.4.4 pack creation MUST be deterministic given identical input object set and bytes.
	12.	Export/Import (Deterministic Archive) (SSOT)

12.1 Archive format

12.1.1 Export MUST produce a single file: TAR archive compressed with Zstandard.

12.1.2 Filename extension SHOULD be .tar.zst.

12.2 Archive contents

12.2.1 Archive MUST contain:
12.2.1.1 meta.db (canonical export snapshot, Section 12.4)
12.2.1.2 objects/ (all CAS data: loose objects and pack data)
12.2.1.3 manifest.json

12.3 Deterministic manifest

12.3.1 manifest.json MUST be a Canonical JSON blob containing exactly:
12.3.1.1 “spec_version”: “0.0.1”
12.3.1.2 “created_at”: int_string (unix seconds; encoded as decimal string)
12.3.1.3 “repo_ids”: [UUIDv7…]
12.3.1.4 “files”: [{ “path”: string, “sha256_hex”: string, “size”: int_string }]

12.3.2 files MUST list every file in the archive excluding manifest.json.

12.3.3 files MUST be sorted by path ascending (bytewise ASCII).

12.3.4 sha256_hex MUST be sha256 over the exact bytes of the archived file.

12.3.5 Determinism rule: created_at MUST be derived deterministically as:
12.3.5.1 max(ts) across all audit rows included in meta.db snapshot; if audit table is empty, created_at MUST be “0”.

12.4 SQLite snapshot and canonical export meta.db (required)

12.4.1 Export MUST archive a consistent meta.db snapshot even if SQLite WAL is enabled.

12.4.2 Implementations MUST satisfy 12.4.1 by one of:
12.4.2.1 using SQLite Online Backup API to write a snapshot meta.db file, then archiving that snapshot
12.4.2.2 an equivalent mechanism that guarantees transactional consistency

12.4.3 Export MUST NOT rely on copying meta.db alone when WAL is enabled unless it also guarantees inclusion or checkpointing of all committed data.

12.4.4 To satisfy Goal 1.6, export MUST write a canonical export meta.db (called meta.db in the archive) that is byte-identical for identical logical state:

12.4.4.1 Take a consistent read snapshot of the live meta.db (via 12.4.2).
12.4.4.2 Create a new empty SQLite database file meta.db.tmp using fixed parameters:
page_size fixed (configurable; default SHOULD be 4096)
journal_mode=OFF during construction
synchronous=FULL during final checkpoint
auto_vacuum=NONE
encoding=UTF-8
user_version fixed (default 1)
12.4.4.3 Create tables in a fixed order exactly matching Section 11.5 schema.
12.4.4.4 Insert rows for each table in deterministic order:
For each table, order rows by the table PRIMARY KEY columns ascending, using bytewise ordering for TEXT (UTF-8 bytes) and raw bytes ordering for BLOB.
12.4.4.5 Create indexes in a fixed order.
12.4.4.6 Ensure the file is checkpointed and fully written:
run a full WAL checkpoint if any WAL was created during build
fsync the database file and parent directory (or equivalent durability)
12.4.4.7 Use the resulting meta.db.tmp bytes as the archived meta.db bytes.

12.4.5 Export MUST NOT include SQLite WAL or SHM files in the archive.

12.4.6 If canonical construction fails, export MUST fail with 500 code “EXPORT_CANONICAL_DB_FAILED”.

12.5 Import

12.5.1 Import MUST restore refs, MRs, ACLs, sessions, idempotency history, and audit history.

12.5.2 IDs MUST remain valid across export/import (blob/tree/commit IDs unchanged).

12.5.3 Import MUST fail if any required file checksum mismatches manifest (400, code “IMPORT_CHECKSUM_MISMATCH”, details include first mismatching path).

12.5.4 Import MUST be atomic per data-dir: on failure, the server MUST not leave a partially imported state.

12.6 Deterministic TAR and Zstandard rules

12.6.1 The TAR stream (before zstd) MUST be deterministic given identical archive file set and identical file bytes.

12.6.2 TAR entry ordering MUST be path ascending (bytewise ASCII), and MUST match manifest.files ordering.

12.6.3 TAR header normalization MUST be:
12.6.3.1 uid = 0, gid = 0
12.6.3.2 uname = “”, gname = “”
12.6.3.3 mtime = 0
12.6.3.4 devmajor = 0, devminor = 0
12.6.3.5 mode:
regular files: 0644
directories: 0755

12.6.4 TAR MUST NOT include PAX headers except those strictly required for long path support; if PAX headers are used, their content MUST be deterministic and MUST NOT include time fields (atime/ctime).

12.6.5 Zstandard compression MUST be deterministic:
12.6.5.1 compression level MUST be fixed by config default (default SHOULD be 3)
12.6.5.2 threads MUST be 1
12.6.5.3 checksum MUST be enabled or disabled deterministically (default SHOULD be enabled)

12.6.6 Implementations MUST NOT include any extra files beyond Section 12.2.1.
	13.	Worktree and Markdown Interop (Deterministic, Local)

13.1 Purpose

13.1.1 Worktree interop provides a deterministic filesystem projection for external editing tools and Git workflows.

13.2 Worktree export layout (normative)

13.2.1 A worktree directory MUST contain:
13.2.1.1 .storyed/worktree.json (guard file, Section 13.3)
13.2.1.2 chapters/<chapter_id>/chapter.meta.json
13.2.1.3 chapters/<chapter_id>/order.json
13.2.1.4 chapters/<chapter_id>/scenes/<scene_id>.md
13.2.1.5 chapters/<chapter_id>/scenes/<scene_id>.meta.json
13.2.1.6 .gitattributes (recommended defaults, Section 13.7)
13.2.1.7 .editorconfig (recommended defaults, Section 13.7)

13.3 Worktree guard file (required)

13.3.1 .storyed/worktree.json MUST be a Canonical JSON blob containing exactly:
13.3.1.1 “spec_version”: “0.0.1”
13.3.1.2 “repo_id”: UUIDv7
13.3.1.3 “ref_name”: string (refs/heads/* only)
13.3.1.4 “base_commit_id”: sha256_hex
13.3.1.5 “export_ts”: int_string; determinism rule: export_ts MUST be “0” in v0.0.1

13.3.2 worktree.json MUST be validated and canonicalized under Section 5.

13.3.3 worktree import MUST fail if worktree.json is missing or invalid.

13.4 Worktree export semantics

13.4.1 Export MUST be deterministic for the same (repo_id, view id) state.

13.4.2 Export MUST write files atomically (write temp then rename) within the output directory.

13.5 Worktree import semantics (required)

13.5.1 Worktree import MUST require explicit expected_head_commit_id and MUST enforce Section 9.1.

13.5.2 Import MUST validate:
13.5.2.1 presence of all required files for referenced items (no half-pairs)
13.5.2.2 JSON canonicalization (Section 5.6) and forbidden chars (Section 5.2)
13.5.2.3 LF normalization of .md (Section 5.5)
13.5.2.4 order.json consistency (Section 8.7)

13.5.3 Import MUST map each <scene_id>.md + <scene_id>.meta.json to the corresponding Scene JSON update.

13.5.4 Import MUST create exactly one new commit on success, updating the target ref atomically.

13.5.5 Import MUST record an audit event with action “WORKTREE_IMPORT” and details including file counts and changed scene_ids.

13.6 Extra-file rejection (required)

13.6.1 Worktree import MUST reject any file path in the archive that is not one of:
13.6.1.1 .storyed/worktree.json
13.6.1.2 chapters/<chapter_id>/chapter.meta.json
13.6.1.3 chapters/<chapter_id>/order.json
13.6.1.4 chapters/<chapter_id>/scenes/<scene_id>.md
13.6.1.5 chapters/<chapter_id>/scenes/<scene_id>.meta.json
13.6.1.6 optional hygiene files: .gitattributes, .editorconfig (ignored on import)

13.6.2 If extra files exist, import MUST fail with 400 code “WORKTREE_EXTRA_FILE” and details listing up to first 20 offending paths (sorted bytewise ASCII).

13.7 Recommended generated repo hygiene files

13.7.1 .gitattributes SHOULD include:
13.7.1.1 *.md text eol=lf
13.7.1.2 *.json text eol=lf

13.7.2 .editorconfig SHOULD include:
13.7.2.1 charset = utf-8
13.7.2.2 end_of_line = lf
13.7.2.3 insert_final_newline = true

13.7.3 worktree add SHOULD generate .gitattributes and .editorconfig by default to prevent ritual fixes.

13.8 Lint (required)

13.8.1 storyed lint MUST validate worktree directories according to Sections 13.2-13.6.

13.8.2 lint output MUST include stable error codes and locations:
13.8.2.1 { code, path, field, message, offset|null }
	14.	Canonical Chapter/Scene Objects (Normative)

14.1 Chapter JSON

14.1.1 Chapter blob MUST be a Canonical JSON blob containing exactly these members:
14.1.1.1 “chapter_id”: string(UUIDv7)
14.1.1.2 “title”: string
14.1.1.3 “summary”: string|null
14.1.1.4 “constraints”: object
14.1.1.5 “tags”: array of string
14.1.1.6 “order_key”: string (lexo-rank, Section 8)

14.1.2 constraints object (Chapter) MUST contain exactly:
14.1.2.1 “rating”: “general”|“r15”|“r18”
14.1.2.2 “flags”: array of string
14.1.2.3 “flags” MAY be empty but MUST be present
14.1.2.4 “tags” MAY be empty but MUST be present

14.1.3 Path/ID consistency: chapter_id MUST equal the <chapter_id> in its path.

14.2 Scene JSON

14.2.1 Scene blob MUST be a Canonical JSON blob containing exactly these members:
14.2.1.1 “scene_id”: string(UUIDv7)
14.2.1.2 “chapter_id”: string(UUIDv7)
14.2.1.3 “order_key”: string (lexo-rank, Section 8)
14.2.1.4 “title”: string|null
14.2.1.5 “body_md”: string (Markdown, UTF-8, LF-normalized)
14.2.1.6 “tags”: array of string
14.2.1.7 “entities”: array of string
14.2.1.8 “constraints”: object
14.2.1.9 “provenance”: object

14.2.2 constraints object (Scene) MUST contain exactly:
14.2.2.1 “rating”: “general”|“r15”|“r18”
14.2.2.2 “flags”: array of string

14.2.3 provenance object (Scene) MUST contain exactly:
14.2.3.1 “op”: “create”|“edit”|“split_from”|“merge_of”|“move”|“delete”
14.2.3.2 “parents”: array of object

14.2.4 provenance.parents MAY be empty only when op=“create”.

14.2.5 provenance.parents MUST be non-empty for op in {“edit”,“split_from”,“merge_of”,“move”,“delete”}.

14.2.6 provenance parent entry MUST contain exactly:
14.2.6.1 “scene_id”: string(UUIDv7)
14.2.6.2 “commit_id”: string(sha256_hex)

14.2.7 provenance.parents MUST NOT contain duplicates of (scene_id, commit_id).

14.2.8 Path/ID consistency:
14.2.8.1 scene_id MUST equal the <scene_id> in its path.
14.2.8.2 chapter_id MUST equal the <chapter_id> in its path.
	15.	HTTP API (Normative)

15.1 OpenAPI

15.1.1 MUST expose OpenAPI at /openapi.json.

15.2 Common headers

15.2.1 All responses SHOULD include X-Request-Id.

15.2.2 All mutating requests MUST enforce Idempotency-Key per Section 9.3.

15.2.3 Auth uses cookie-based sessions (Section 16). All mutating endpoints MUST enforce CSRF Origin checks (Section 16.3).

15.3 Error model (normative)

15.3.1 Error JSON: { “code”: string, “message”: string, “details”?: any }

15.3.2 HTTP mapping:
15.3.2.1 400 invalid input
15.3.2.2 401 unauthenticated
15.3.2.3 403 unauthorized or CSRF blocked
15.3.2.4 404 not found
15.3.2.5 409 conflict (includes ORDER_KEY_SPACE_EXHAUSTED and REF_HEAD_MISMATCH)
15.3.2.6 413 payload too large
15.3.2.7 415 unsupported media type
15.3.2.8 429 rate limited
15.3.2.9 500 internal error
15.3.2.10 503 service unavailable (DB_BUSY)
15.3.2.11 507 storage insufficient

15.4 Endpoints (required)

15.4.1 Core:
GET  /health
GET  /metrics
POST /auth/login
POST /auth/logout
GET  /auth/me

15.4.2 Repos and refs:
POST /repos
GET  /repos/{repo_id}
POST /repos/{repo_id}/refs
GET  /repos/{repo_id}/refs
GET  /repos/{repo_id}/head?ref=<ref_name>

15.4.3 CAS:
POST /blobs
GET  /blobs/{blob_id}
POST /trees
GET  /trees/{tree_id}

15.4.4 Commits and diffs:
POST /repos/{repo_id}/commits
GET  /repos/{repo_id}/commits/{commit_id}
GET  /repos/{repo_id}/diff-summary?base=<ref|commit>&head=<ref|commit>
GET  /repos/{repo_id}/diff-scene?base=<ref|commit>&head=<ref|commit>&scene_id=

15.4.5 MR:
POST /repos/{repo_id}/mrs
GET  /repos/{repo_id}/mrs
GET  /repos/{repo_id}/mrs/{mr_id}
POST /repos/{repo_id}/mrs/{mr_id}/merge
POST /repos/{repo_id}/mrs/{mr_id}/cherry-pick

15.4.6 Ops:
POST /repos/{repo_id}/rank/between
POST /repos/{repo_id}/rank/rebalance
POST /repos/{repo_id}/ops/create-chapter
POST /repos/{repo_id}/ops/create-scene
POST /repos/{repo_id}/ops/delete-scene
POST /repos/{repo_id}/ops/publish-scene
POST /repos/{repo_id}/ops/publish-staged
POST /repos/{repo_id}/ops/preflight-publish
POST /repos/{repo_id}/ops/rebase-staged
POST /repos/{repo_id}/ops/split-scene
POST /repos/{repo_id}/ops/merge-scenes
POST /repos/{repo_id}/ops/move-scene
POST /repos/{repo_id}/ops/revert-ref

15.4.7 Admin and ACL:
GET  /users
POST /users
GET  /users/{user_id}
GET  /repos/{repo_id}/acl
PUT  /repos/{repo_id}/acl

15.4.8 Audit and export/import:
GET  /repos/{repo_id}/audit?after_ts=<int|null>&limit=
GET  /export?repo_id=<UUIDv7|null>
POST /import?run_verify=0|1&dry_run=0|1

15.4.9 Worktree:
GET  /repos/{repo_id}/worktree/export?ref=<ref|commit>
POST /repos/{repo_id}/worktree/preflight-import?ref=&expected_head_commit_id=
POST /repos/{repo_id}/worktree/import?ref=&expected_head_commit_id=

15.4.10 UI serving:
GET  /ui/
GET  /ui/index.html
GET  /ui/ui_manifest.json
GET  /ui/assets/*

15.4.11 Maintenance and diagnostics:
POST /maintenance/pack?dry_run=0|1
POST /maintenance/verify
POST /maintenance/repair-order
POST /maintenance/prune-idempotency
GET  /system/diagnostics

15.5 Endpoint schemas (normative minimum)

15.5.1 GET /health
Response 200: { “status”: “ok”|“degraded”|“fail”, “spec_version”: “0.0.1”, “checks”: { … } }

15.5.2 POST /auth/login
Request: { “handle”: string, “password”: string }
Response 200: { “user_id”: UUIDv7, “handle”: string, “role_summary”: { “is_admin”: bool } }
Rules:
On success, server MUST set session cookie.
On failure, server MUST return 401 with code “AUTH_INVALID”.

15.5.3 POST /auth/logout
Response 200: { “ok”: true }
Rules: server MUST clear session cookie (expired).

15.5.4 GET /auth/me
Response 200: { “user_id”: UUIDv7, “handle”: string, “roles”: [{ “repo_id”: UUIDv7, “role”: string }], “is_admin”: bool }
Response 401 if no valid session.

15.5.5 POST /repos
Request: { “name”: string|null }
Response 201: { “repo_id”: UUIDv7, “default_ref”: “refs/heads/main”, “head_commit_id”: sha256_hex }

15.5.6 GET /repos/{repo_id}
Response 200: { “repo_id”: UUIDv7, “name”: string|null, “default_ref”: string, “head_commit_id”: sha256_hex }

15.5.7 GET /repos/{repo_id}/head?ref=…
Response 200: { “ref_name”: string, “commit_id”: sha256_hex }

15.5.8 POST /repos/{repo_id}/refs
Request:
{ “ref_name”: string, “target_commit_id”: sha256_hex, “expected_old_commit_id”: sha256_hex|null }
Rules:
ref_name MUST match: ^refs/(heads|tags)/[A-Za-z0-9._-]{1,64}$
Response 200: { “ref_name”: string, “commit_id”: sha256_hex }

15.5.9 GET /repos/{repo_id}/refs
Response 200:
{ “refs”: [{ “ref_name”: string, “commit_id”: sha256_hex, “updated_at”: int_string }] }
Rules:
refs MUST be sorted by ref_name ascending (bytewise ASCII).
updated_at MUST be encoded as a decimal string in JSON.

15.5.10 POST /blobs
Request: raw bytes, Content-Type required
Response 201: { “blob_id”: sha256_hex, “size”: int_string, “content_type”: string }
Rules:
The server MUST store a normalized content_type value in meta.db and return the normalized value:
trim leading/trailing ASCII whitespace
reject NUL or control characters
additionally lowercase type/subtype deterministically
size MUST be returned as a decimal string.

15.5.11 GET /blobs/{blob_id}
Response 200: raw bytes of the blob
Rules:
Content-Type SHOULD be the stored normalized content_type if known; otherwise application/octet-stream.

15.5.12 POST /trees
Request: { “entries”: [ { “path”: string, “blob_id”: sha256_hex } ] }
Rules:
path MUST comply with Section 7.4 and MUST be unique within request.
The server MUST reject any blob_id not present in CAS (404, code “CAS_BLOB_NOT_FOUND”).
The server MUST sort entries by path bytewise before canonical CBOR encoding.
Response 201: { “tree_id”: sha256_hex }

15.5.13 GET /trees/{tree_id}
Response 200: { “tree_id”: sha256_hex, “entries”: [ { “path”: string, “blob_id”: sha256_hex } ] }
Rules:
returned list MUST be sorted by path bytewise.

15.5.14 POST /repos/{repo_id}/commits
Request:
{ “tree_id”: sha256_hex, “parents”: [sha256_hex…], “author”: { “user_id”: UUIDv7, “handle”: string|null }, “message”: string, “created_at”: int_string }
Rules:
tree_id MUST exist (404, code “CAS_TREE_NOT_FOUND”).
each parent commit_id MUST exist (404, code “CAS_COMMIT_NOT_FOUND”).
server MUST sort parents by raw bytes ascending before canonical encoding.
created_at MUST be a decimal string in JSON; server MUST parse as int.
Response 201: { “commit_id”: sha256_hex }

15.5.15 GET /repos/{repo_id}/commits/{commit_id}
Response 200:
{ “commit_id”: sha256_hex, “tree_id”: sha256_hex, “parents”: [sha256_hex…], “author”: { “user_id”: UUIDv7, “handle”: string|null }, “message”: string, “created_at”: int_string }

15.5.16 GET /repos/{repo_id}/diff-summary
Response 200:
{
“base”: { “kind”: “ref”|“commit”, “id”: string },
“head”: { “kind”: “ref”|“commit”, “id”: string },
“chapters”: { “added”: [UUIDv7…], “deleted”: [UUIDv7…], “modified”: [UUIDv7…], “reordered”: [UUIDv7…] },
“scenes”: { “added”: [UUIDv7…], “deleted”: [UUIDv7…], “modified”: [UUIDv7…], “moved”: [UUIDv7…], “reordered”: [UUIDv7…] }
}

15.5.17 GET /repos/{repo_id}/diff-scene
Response 200:
{
“scene_id”: UUIDv7,
“base_commit_id”: sha256_hex,
“head_commit_id”: sha256_hex,
“body_diff”: { “algorithm”: string, “hunks”: any },
“meta_diff”: any,
“order_diff”: any
}

15.5.18 POST /repos/{repo_id}/mrs
Request: { “base_ref”: string, “head_ref”: string }
Response 201:
{ “mr_id”: UUIDv7, “repo_id”: UUIDv7, “base_ref”: string, “head_ref”: string, “base_commit_id”: sha256_hex, “status”: “open” }

15.5.19 POST /repos/{repo_id}/mrs/{mr_id}/merge
Request:
{
“mode”: “ff”|“merge”|“squash”,
“expected_base_head_commit_id”: sha256_hex|null,
“order_conflicts_default”: “head”|“base”,
“resolutions”: [
{
“scene_id”: UUIDv7,
“content”: { “choice”: “base”|“head”|“manual”, “body_md”?: string },
“meta”:    { “choice”: “base”|“head”|“manual”, “fields”?: object },
“order”:   { “choice”: “base”|“head”|“manual”, “chapter_id”?: UUIDv7, “order_key”?: string }
}
]
}
Rules:
expected_base_head_commit_id, if provided, MUST be enforced as expected_head_commit_id against base_ref (Section 9.1).
Response 200: { “merged_commit_id”: sha256_hex, “receipt”: object(Section 9.4) }

15.5.20 POST /repos/{repo_id}/mrs/{mr_id}/cherry-pick
Request: { “scene_ids”: [UUIDv7…], “target_ref”: string, “expected_head_commit_id”: sha256_hex|null }
Response 200: { “commit_id”: sha256_hex, “updated_ref”: string, “previous_head_commit_id”: sha256_hex, “receipt”: object(Section 9.4) }

15.5.21 POST /repos/{repo_id}/rank/between
Request: { “left_key”: string|null, “right_key”: string|null }
Response 200: { “order_key”: string }

15.5.22 POST /repos/{repo_id}/rank/rebalance
Request: { “chapter_id”: UUIDv7, “ref”: string, “expected_head_commit_id”: sha256_hex|null }
Response 200: { “commit_id”: sha256_hex, “updated_ref”: string, “previous_head_commit_id”: sha256_hex, “receipt”: object(Section 9.4) }

15.5.23 POST /repos/{repo_id}/ops/create-chapter
Request:
{ “ref”: string, “expected_head_commit_id”: sha256_hex|null, “fields”: { “title”: string, “summary”: string|null, “constraints”: { “rating”: string, “flags”: [string…] }, “tags”: [string…] }, “message”: string|null }
Response 200: { “chapter_id”: UUIDv7, “commit_id”: sha256_hex, “updated_ref”: string, “previous_head_commit_id”: sha256_hex, “receipt”: object(Section 9.4) }

15.5.24 POST /repos/{repo_id}/ops/create-scene
Request:
{ “ref”: string, “expected_head_commit_id”: sha256_hex|null, “chapter_id”: UUIDv7, “left_scene_id”: UUIDv7|null, “right_scene_id”: UUIDv7|null, “fields”: { “title”: string|null, “body_md”: string, “tags”: [string…], “entities”: [string…], “constraints”: { “rating”: string, “flags”: [string…] } }, “message”: string|null }
Response 200: { “scene_id”: UUIDv7, “new_order_key”: string, “commit_id”: sha256_hex, “updated_ref”: string, “previous_head_commit_id”: sha256_hex, “receipt”: object(Section 9.4) }

15.5.25 POST /repos/{repo_id}/ops/delete-scene
Request:
{ “ref”: string, “expected_head_commit_id”: sha256_hex|null, “scene_id”: UUIDv7, “message”: string|null }
Response 200: { “commit_id”: sha256_hex, “updated_ref”: string, “previous_head_commit_id”: sha256_hex, “receipt”: object(Section 9.4) }

15.5.26 POST /repos/{repo_id}/ops/publish-scene
Request:
{
“ref”: string,
“expected_head_commit_id”: sha256_hex|null,
“scene_id”: UUIDv7,
“chapter_id”: UUIDv7,
“fields”: {
“title”: string|null,
“body_md”: string,
“tags”: [string…],
“entities”: [string…],
“constraints”: { “rating”: string, “flags”: [string…] }
},
“message”: string|null
}
Rules:
fields MUST be validated per Sections 5.1-5.7; server is source of truth for canonicalization.
server MUST update provenance.op=“edit” and include parent (scene_id, head_before) deterministically.
Response 200: { “commit_id”: sha256_hex, “updated_ref”: string, “previous_head_commit_id”: sha256_hex, “receipt”: object(Section 9.4) }

15.5.27 POST /repos/{repo_id}/ops/publish-staged
Request:
{
“ref”: string,
“expected_head_commit_id”: sha256_hex|null,
“base_head_commit_id”: sha256_hex,
“items”: [
{ “scene_id”: UUIDv7, “chapter_id”: UUIDv7, “fields”: object, “message”: string|null }
]
}
Rules:
items MUST be applied in deterministic order:
sort by (chapter_id, order_key, scene_id) from the base_head_commit_id snapshot.
server MUST reject if base_head_commit_id does not match the committed state used to derive the order (409, code “STAGE_BASE_MISMATCH”).
Response 200: { “commit_id”: sha256_hex, “updated_ref”: string, “previous_head_commit_id”: sha256_hex, “receipt”: object(Section 9.4) }

15.5.28 POST /repos/{repo_id}/ops/preflight-publish
Request:
{
“ref”: string,
“expected_head_commit_id”: sha256_hex|null,
“mode”: “scene”|“staged”,
“scene”?: { “scene_id”: UUIDv7, “chapter_id”: UUIDv7, “fields”: object },
“staged”?: { “base_head_commit_id”: sha256_hex, “items”: [object…] }
}
Response 200:
{
“ok”: bool,
“errors”: [ { “code”: string, “path”?: string, “field”?: string, “message”: string, “offset”: int_string|null } ],
“predicted”: {
“changed_paths”: [string…],
“changed_scene_ids”: [UUIDv7…]
}
}

15.5.29 POST /repos/{repo_id}/ops/rebase-staged
Request:
{
“ref”: string,
“expected_head_commit_id”: sha256_hex,
“base_head_commit_id”: sha256_hex,
“items”: [object…]
}
Response 200: { “commit_id”: sha256_hex, “updated_ref”: string, “previous_head_commit_id”: sha256_hex, “receipt”: object(Section 9.4) }
Response 409: code “REBASE_CONFLICT” with details describing per-scene conflict types.

15.5.30 POST /repos/{repo_id}/ops/move-scene
Request:
{
“ref”: string,
“expected_head_commit_id”: sha256_hex|null,
“scene_id”: UUIDv7,
“target_chapter_id”: UUIDv7,
“left_scene_id”: UUIDv7|null,
“right_scene_id”: UUIDv7|null
}
Response 200: { “commit_id”: sha256_hex, “updated_ref”: string, “previous_head_commit_id”: sha256_hex, “new_order_key”: string, “receipt”: object(Section 9.4) }

15.5.31 POST /repos/{repo_id}/ops/split-scene
Request:
{
“ref”: string,
“expected_head_commit_id”: sha256_hex|null,
“scene_id”: UUIDv7,
“splits”: [ { “byte_offset”: int_string } ],
“titles”: [string|null] | null
}
Rules:
split offsets are expressed as byte offsets into LF-normalized UTF-8 bytes of body_md:
strictly increasing
in range 1..len(body_bytes)-1
on a valid UTF-8 codepoint boundary
evaluated after LF normalization and NFC normalization of body_md
If violated, server MUST return 400 invalid input.
If ORDER_KEY_SPACE_EXHAUSTED occurs during split, apply Section 9.2.2.
Response 200: { “commit_id”: sha256_hex, “updated_ref”: string, “previous_head_commit_id”: sha256_hex, “new_scene_ids”: [UUIDv7…], “receipt”: object(Section 9.4) }

15.5.32 POST /repos/{repo_id}/ops/merge-scenes
Request:
{
“ref”: string,
“expected_head_commit_id”: sha256_hex|null,
“scene_ids”: [UUIDv7…],
“target_chapter_id”: UUIDv7,
“left_scene_id”: UUIDv7|null,
“right_scene_id”: UUIDv7|null,
“joiner”: string
}
Rules:
joiner MUST be processed under Section 5 (UTF-8, NFC, forbidden chars, canonical JSON rules where applicable).
Response 200: { “commit_id”: sha256_hex, “updated_ref”: string, “previous_head_commit_id”: sha256_hex, “merged_scene_id”: UUIDv7, “receipt”: object(Section 9.4) }

15.5.33 POST /repos/{repo_id}/ops/revert-ref
Request:
{
“ref”: string,
“expected_head_commit_id”: sha256_hex|null,
“to_commit_id”: sha256_hex,
“message”: string
}
Rules:
MUST create a new commit whose tree equals to_commit_id.tree and update ref atomically.
Response 200: { “commit_id”: sha256_hex, “updated_ref”: string, “previous_head_commit_id”: sha256_hex, “receipt”: object(Section 9.4) }

15.5.34 GET /repos/{repo_id}/audit
Response 200:
{
“events”: [
{ “event_id”: UUIDv7, “ts”: int_string, “actor_id”: UUIDv7, “action”: string, “repo_id”: UUIDv7|null, “details_json”: object }
],
“next_after_ts”: int_string|null
}
Rules:
events MUST be ordered by (ts, event_id) ascending.
details_json MUST be canonicalized per Section 5.6 when stored; API returns canonical form.
ts MUST be encoded as decimal string in JSON.

15.5.35 GET /export
Response 200:
Content-Type: application/zstd
Body: tar.zst bytes as defined in Section 12
Rules:
Access MUST require admin.
repo_id query parameter MAY be null; if null, export all repos.

15.5.36 POST /import?run_verify=0|1&dry_run=0|1
Request:
Content-Type: application/zstd
Body: tar.zst bytes
Response 200: { “ok”: true, “imported_repo_ids”: [UUIDv7…], “verify”: object|null, “dry_run”: bool }
Rules:
Access MUST require admin.
Import MUST be atomic per data-dir.
If run_verify=1, server MUST run verify after import and return results.
In dry_run=1, server MUST NOT modify target data-dir.

15.5.37 Worktree endpoints

15.5.37.1 GET /repos/{repo_id}/worktree/export?ref=<ref|commit>
Response 200: tar.zst or zip bytes (implementation choice), containing worktree layout (Section 13.2)
Rules:
Export MUST be deterministic for the same (repo_id, view id) state.
Export MUST include .storyed/worktree.json with export_ts=0.

15.5.37.2 POST /repos/{repo_id}/worktree/preflight-import
Request: { “ref”: string, “expected_head_commit_id”: sha256_hex, “archive_bytes”: bytes }
Response 200: same schema as 15.5.28 with mode=“worktree”
Rules:
MUST NOT mutate persistent state.

15.5.37.3 POST /repos/{repo_id}/worktree/import
Request: { “ref”: string, “expected_head_commit_id”: sha256_hex, “archive_bytes”: bytes }
Response 200: { “commit_id”: sha256_hex, “updated_ref”: string, “previous_head_commit_id”: sha256_hex, “receipt”: object(Section 9.4) }

15.5.38 Users and ACL (minimal)

15.5.38.1 GET /users (admin)
Response 200: { “users”: [{ “user_id”: UUIDv7, “handle”: string, “created_at”: int_string }] }

15.5.38.2 POST /users (admin)
Request: { “handle”: string, “password”: string }
Response 201: { “user_id”: UUIDv7 }

15.5.38.3 GET /repos/{repo_id}/acl (maintainer+)
Response 200: { “entries”: [{ “user_id”: UUIDv7, “role”: “admin”|“maintainer”|“writer”|“reader” }] }

15.5.38.4 PUT /repos/{repo_id}/acl (admin or repo maintainer)
Request: { “entries”: [{ “user_id”: UUIDv7, “role”: “maintainer”|“writer”|“reader” }] }
Response 200: { “ok”: true }

15.5.39 POST /maintenance/verify
Response 200: { “ok”: bool, “checks”: [ { “name”: string, “status”: “pass”|“fail”, “details”?: any } ] }

15.5.40 POST /maintenance/pack?dry_run=0|1
Request: { “repo_id”: UUIDv7|null }
Response 200:
if dry_run=1: { “dry_run”: true, “target_object_count”: int_string, “estimated_pack_bytes”: int_string, “predicted_pack_id”: sha256_hex|null }
else: { “dry_run”: false, “pack_id”: sha256_hex, “object_count”: int_string }
Rules:
pack MUST be explicit and MUST NOT run automatically.
pack MUST NOT change any content IDs; it is a storage optimization only.

15.5.41 GET /system/diagnostics
Response 200:
{
“spec_version”: “0.0.1”,
“db”: { “page_count”: int_string, “wal_bytes”: int_string|null },
“cas”: { “loose_object_count”: int_string, “pack_count”: int_string },
“rebalance”: { “chapters”: [ { “chapter_id”: UUIDv7, “pressure”: string, “min_gap_hint”: string|null } ] },
“perf”: { “p95_publish_ms”: int_string|null, “p95_diff_scene_ms”: int_string|null }
}
Rules:
Diagnostics MUST be offline-safe and MUST NOT expose secrets.
	16.	Security, Auth, CSRF, and UI Serving

16.1 Auth and sessions (required)

16.1.1 Mutating endpoints MUST require authentication.

16.1.2 Read endpoints MAY be public depending on repo ACL, but MUST enforce ACL.

16.1.3 The server MUST support cookie-based sessions:
16.1.3.1 session cookie MUST be HttpOnly
16.1.3.2 session cookie MUST be SameSite=Lax or SameSite=Strict
16.1.3.3 session cookie SHOULD be Secure when served over HTTPS

16.1.4 The server MUST NOT require any external IdP for correctness.

16.2 Password hashing

16.2.1 Password hashing MUST use a modern KDF (Argon2id recommended).

16.3 CSRF protection (required)

16.3.1 Because authentication uses cookies, all mutating endpoints MUST enforce CSRF protections.

16.3.2 For any mutating endpoint, the server MUST require an Origin header.

16.3.3 The server MUST reject the request with 403 code “CSRF_BLOCKED” if:
16.3.3.1 Origin is missing, OR
16.3.3.2 Origin does not match the server’s own origin (scheme + host + port) as seen by the server.

16.3.4 For JSON mutating endpoints, the server MUST require Content-Type: application/json (after trimming ASCII whitespace) and MUST reject other content types with 415 code “UNSUPPORTED_MEDIA_TYPE”.

16.3.5 For binary mutating endpoints (e.g., import), the server MUST still enforce Origin matching.

16.4 UI static serving (normative)

16.4.1 UI MUST be a static SPA served by the binary.

16.4.2 Pinned UI implementation profile (normative)
16.4.2.1 TypeScript (ECMAScript 2022 target)
16.4.2.2 React 18.x
16.4.2.3 Vite 5.x
16.4.2.4 CodeMirror 6.x

16.4.3 The UI build MUST be fully self-contained:
16.4.3.1 MUST NOT require external CDNs, external fonts, or remote scripts/styles.
16.4.3.2 MUST NOT require service workers for correctness (MAY use them if fully offline and deterministic).

16.4.4 The binary MUST embed the exact UI build output bytes and serve them verbatim.

16.4.5 The server MUST serve the UI at path prefix /ui/ and MUST redirect / to /ui/ (HTTP 302) or serve the UI directly at /.

16.4.6 The server MUST set Content-Type correctly for: .html, .js, .css, .json, .svg, .png, .woff2 (if used).

16.4.7 The UI MUST be compatible with Chromium-based browsers and Firefox current stable at release time.

16.5 ui_manifest.json (required)

16.5.1 The embedded UI asset root MUST contain:
16.5.1.1 /ui/index.html
16.5.1.2 /ui/assets/
16.5.1.3 /ui/ui_manifest.json

16.5.2 ui_manifest.json MUST be a Canonical JSON blob with:
16.5.2.1 “spec_version”: “0.0.1”
16.5.2.2 “build_ts”: int_string (unix seconds, UTC; encoded as decimal string)
16.5.2.3 “files”: [{ “path”: string, “sha256_hex”: string, “size”: int_string }]

16.5.3 files MUST list every embedded UI file including index.html and all assets, excluding ui_manifest.json itself.

16.5.4 files MUST be sorted by path ascending (bytewise ASCII).

16.5.5 sha256_hex MUST be sha256 over the exact bytes served for that file.

16.5.6 Determinism rule: build_ts MUST be “0” in v0.0.1.

16.5.7 The server MUST expose the same ui_manifest.json bytes at GET /ui/ui_manifest.json.

16.6 UI security headers (required)

16.6.1 The server MUST send the following headers on all /ui/ responses:
16.6.1.1 X-Content-Type-Options: nosniff
16.6.1.2 Referrer-Policy: no-referrer
16.6.1.3 Cross-Origin-Resource-Policy: same-origin
16.6.1.4 Cross-Origin-Opener-Policy: same-origin
16.6.1.5 Cross-Origin-Embedder-Policy: require-corp

16.6.2 The server MUST send a Content-Security-Policy header for /ui/ that enforces at minimum:
16.6.2.1 default-src ‘none’
16.6.2.2 script-src ‘self’
16.6.2.3 style-src ‘self’
16.6.2.4 img-src ‘self’
16.6.2.5 font-src ‘self’
16.6.2.6 connect-src ‘self’
16.6.2.7 base-uri ‘none’
16.6.2.8 frame-ancestors ‘none’
16.6.2.9 form-action ‘none’

16.7 UI caching (required minimum)

16.7.1 /ui/assets/* SHOULD use content-hashed filenames.

16.7.2 If content-hashed, the server SHOULD set Cache-Control: public, max-age=31536000, immutable.

16.7.3 /ui/index.html and /ui/ui_manifest.json MUST be served with Cache-Control: no-store.
	17.	UI Profile (Pinned, Normative)

17.1 Global UX invariants

17.1.1 Draft vs staged vs committed separation
17.1.1.1 Draft changes MUST be local-only and MUST NOT be visible to other users until published.
17.1.1.2 Staged changes MUST be local-only and MUST represent an explicit user-selected publish set.
17.1.1.3 Committed state MUST be the server’s ref/commit state and MUST be treated as source of truth.

17.1.2 Deterministic navigation
17.1.2.1 Deep links MUST be stable across reloads.
17.1.2.2 Reload MUST restore route, repo selection, ref selection, and selected scene when present.

17.1.3 Stable identity visibility
17.1.3.1 The UI MUST provide a discoverable display of:
repo_id
current view id (ref name or commit id)
for edit views, active ref name (refs/heads/*) and its head commit id
17.1.3.2 The UI MUST show the last mutation receipt in a copyable panel for the current session.

17.1.4 Keyboard-first viability
17.1.4.1 Core flows MUST be usable with keyboard-only navigation: read, edit, stage, publish, MR review, conflict resolution, merge.

17.1.5 Error transparency
17.1.5.1 All error surfaces MUST show server error code and message when provided.
17.1.5.2 The UI MUST show X-Request-Id when present (copyable).
17.1.5.3 For any mutation failure, the UI MUST show the attempted expected_head_commit_id and current head (if known).

17.1.6 Ref targeting guard
17.1.6.1 All ref-mutating UI actions MUST display target ref name and expected head commit id prior to confirmation.
17.1.6.2 The UI MUST NOT allow ref mutations without sending expected_head_commit_id when available.

17.1.7 Ritual elimination invariants
17.1.7.1 The UI MUST implement one-click flows that internally use preflight endpoints before performing dangerous or failure-prone mutations.
17.1.7.2 The UI MUST surface explicit single-step options for 409 handling using rebase-staged when applicable.

17.2 Routes (required under /ui/)

17.2.1 /ui/
17.2.2 /ui/repos/:repo_id/read?ref=<ref_name_or_commit_id>[&scene=<scene_id>]
17.2.3 /ui/repos/:repo_id/edit?ref=<ref_name>[&scene=<scene_id>]
17.2.4 /ui/repos/:repo_id/staged?ref=<ref_name>
17.2.5 /ui/repos/:repo_id/mrs
17.2.6 /ui/repos/:repo_id/mrs/:mr_id
17.2.7 /ui/repos/:repo_id/settings
17.2.8 /ui/system
17.2.9 /ui/system/diagnostics

17.2.10 Optional: /ui/login

17.2.11 Route constraints
17.2.11.1 read route ref MAY be a ref name or commit id.
17.2.11.2 edit route ref MUST be a refs/heads/* name; the UI MUST reject editing refs/tags/* or raw commit ids in the editor.
17.2.11.3 If edit route is requested with an invalid ref, the UI MUST redirect to read route for the same repo with the same ref string (best-effort) and show an explanatory banner.

17.3 Screen model

17.3.1 Shell layout MUST provide:
17.3.1.1 top bar: identity, global status, and actions
17.3.1.2 left panel: chapter list and scene list
17.3.1.3 main panel: route content

17.3.2 top bar MUST show: repo_id, view id, active ref (if any), head commit id, and last receipt summary.

17.3.3 Global status indicators (required)
17.3.3.1 Network: ONLINE, OFFLINE, DEGRADED
17.3.3.2 Draft per scene: CLEAN, DIRTY, SAVING, SAVE_FAILED
17.3.3.3 Stage per scene: UNSTAGED, STAGED
17.3.3.4 Publish: IDLE, PUBLISHING, PUBLISH_FAILED, PUBLISHED, PUBLISH_CONFLICT
17.3.3.5 Indicators MUST be non-blocking and MUST NOT steal focus during typing.

17.3.4 Global toasts (required)
17.3.4.1 Success toasts MUST be time-limited and non-modal.
17.3.4.2 Error toasts MUST remain until dismissed or replaced by a newer error for the same operation.
17.3.4.3 Toasts MUST include: operation name, error code, and request id when present.

17.3.5 Scene list dual sorting (required)
17.3.5.1 Toggle between read order (order.json) and work order (most recently staged or edited).
17.3.5.2 Show per-scene badges: DIRTY, STAGED, CHANGED_ON_HEAD, CONFLICT.

17.4 Data fetching and caching

17.4.1 All requests MUST be cancellable via AbortController on route changes.

17.4.2 The UI MUST dedupe in-flight GET requests by (method, url).

17.4.3 The UI SHOULD maintain a bounded in-memory cache for GET responses with explicit invalidation.

17.4.4 Cache invalidation MUST occur after successful mutations that affect those resources.

17.4.5 The UI MUST rely on HttpOnly cookie-based sessions and MUST NOT store auth tokens in localStorage.

17.4.6 Mutation rules
17.4.6.1 All mutating requests MUST include Idempotency-Key and MUST reuse it across retries.
17.4.6.2 The UI MUST implement exponential backoff for retries on network errors and 429 only.
17.4.6.3 The UI MUST NOT auto-retry 409 conflicts; it MUST show a conflict flow.
17.4.6.4 The UI MUST send expected_head_commit_id for all ref-mutating operations.

17.4.7 Offline behavior
17.4.7.1 If network requests fail, the UI MUST remain usable for typing and local stage persistence.
17.4.7.2 The UI MUST display OFFLINE state.
17.4.7.3 The UI MUST NOT implicitly queue server mutations.

17.4.8 Worker offload (typing safety)
17.4.8.1 Markdown preview rendering and large diff formatting SHOULD be performed in a WebWorker.
17.4.8.2 Worker outputs MUST be deterministic for identical inputs.

17.5 Draft and stage persistence (IndexedDB)

17.5.1 Drafts MUST be persisted in IndexedDB.
17.5.2 Draft key MUST be (repo_id, ref_name, scene_id).
17.5.3 Draft value MUST include body_md, title, tags, entities, updated_at (local-only; not canonical).

17.5.4 Staged changes MUST be persisted in IndexedDB.
17.5.5 Stage key MUST be (repo_id, ref_name, scene_id).
17.5.6 Stage value MUST include staged_fields, base_head_commit_id, summary, summary_hash, updated_at.

17.5.7 Baseline MUST be loaded from committed state for the active ref when entering edit route.

17.5.8 Baseline MUST be updated after a successful publish that includes the scene.

17.5.9 Dirty MUST be computed as draft != baseline over editor-exposed fields.

17.5.10 Autosave rules
17.5.10.1 Autosave MUST be incremental and MUST NOT block typing latency budgets.
17.5.10.2 Trigger: after 1000ms inactivity and at least once every 3000ms during continuous typing.
17.5.10.3 Autosave MUST also occur on blur and before unload.
17.5.10.4 Autosave MUST write only the selected scene draft.
17.5.10.5 On failure, Draft indicator MUST become SAVE_FAILED and retry with backoff without blocking typing.

17.5.11 Multi-tab handling
17.5.11.1 Detect multi-tab conflicts on same draft key and prompt: KEEP_LOCAL, TAKE_OTHER, MANUAL_MERGE.
17.5.11.2 MANUAL_MERGE MUST present a deterministic 2-way diff for body_md and structured diffs for metadata.

17.6 Reading UX

17.6.1 Chapters displayed sorted by (chapter.order_key, chapter_id).

17.6.2 Scenes displayed in order.json order.

17.6.3 Reader MUST display title, rendered Markdown body, tags, entities.

17.6.4 Images MUST be rendered as links (no ) in v0.0.1.

17.6.5 Branch surfacing rule: UI MUST NOT show an unbounded list of forks; SHOULD show at most 3 recommended alternate routes by default.

17.6.6 Scene detail MUST provide a lineage view derived from provenance.

17.7 Editor UX

17.7.1 body_md editor MUST be CodeMirror 6.x.

17.7.2 IME composition MUST NOT be interrupted.

17.7.3 CodeMirror MUST NOT be re-mounted during normal typing.

17.7.4 UI MUST NOT steal focus from editor during active typing unless user explicitly navigates.

17.7.5 Preview SHOULD be available and MUST be safe (no script execution), incremental for large body_md, and offline.

17.8 Publish UX

17.8.1 Publish MUST be presented as PUBLISH (shared) and distinct from Draft autosave.

17.8.2 Publish MUST operate on one scene or the staged set for the active ref; batch publish produces exactly one commit.

17.8.3 UI MUST publish via:
17.8.3.1 POST /repos/{repo_id}/ops/publish-scene for single scene
17.8.3.2 POST /repos/{repo_id}/ops/publish-staged for batch
17.8.3.3 UI MUST run preflight before publish and display errors in-place.
17.8.3.4 UI MUST include expected_head_commit_id for all publish operations.

17.8.4 Publish state machine: IDLE -> PREFLIGHT -> IN_FLIGHT -> SUCCESS | FAILED | CONFLICT

17.8.5 IN_FLIGHT MUST keep the editor editable; only publish action may be disabled.

17.8.6 SUCCESS MUST update baseline and clear dirty for published scenes.

17.8.7 FAILED MUST preserve drafts and staged changes and allow RETRY with same Idempotency-Key.

17.8.8 CONFLICT (409) MUST preserve drafts/staged and enter conflict flow:
fetch latest head
compute diff head_at_stage -> latest_head
offer: REBASE_STAGED, TAKE_HEAD, MANUAL_MERGE
UI MUST require explicit user choice; MUST NOT auto-resolve.

17.9 Scene operations UX

17.9.1 Create scene navigates to created scene in edit route with editor focused.

17.9.2 Move/reorder optimistic update visible within budgets; on rejection revert and show actionable error.

17.9.3 Split-at-cursor supported; preview what stays vs moves; ORDER_KEY_SPACE_EXHAUSTED offers REBALANCE_THEN_RETRY while preserving inputs.

17.9.4 Merge scenes via multi-select; default joiner “\n\n”; on success navigate to merged scene.

17.9.5 Delete scene explicit confirmation; on success navigate to nearest scene by reading order.

17.9.6 Revert: UI MUST provide one-click revert using /ops/revert-ref for operations exposing previous_head_commit_id.

17.10 MR and diff UX

17.10.1 MR list shows mr_id, base_ref, head_ref, status, updated_at; supports filter by status.

17.10.2 MR detail shows base_ref, head_ref, base_commit_id, head_commit_id, merge_base_commit_id; scene-first grouped change list; body diffs not expanded by default.

17.10.3 Per-scene diff shows deterministic line diff and structured diffs for title/tags/entities/constraints/order/provenance.

17.10.4 Merge action requires confirmation; shows mode and effect on base_ref; on success shows merged_commit_id and receipt.

17.11 Conflict resolution UX

17.11.1 Per conflicted scene: content choice base/head/manual; meta choice base/head/manual; order choice base/head/manual.

17.11.2 Manual order MUST allow selecting placement and MUST derive order_key via rank/between.

17.11.3 Bulk flows: UI MUST provide BULK_APPLY for order conflicts with per-scene overrides.

17.11.4 If many order conflicts exist, UI SHOULD propose MERGE_THEN_REBALANCE.

17.11.5 Switching resolution choice MUST update preview within 100ms p95.

17.12 Keyboard shortcuts (required minimum)

17.12.1 Shortcuts for next/prev scene; focus list/editor; stage/unstage; publish scene; publish staged; open MR list; MR next/prev changed scene; conflict next/prev.

17.12.2 UI MUST provide a help surface listing shortcuts.

17.12.3 Shortcut design MUST avoid conflicts with IME and common browser/system shortcuts.

17.13 Performance and latency profile (pinned)

17.13.1 Measurement:
t0 is when browser receives input event; t1 is first paint after state reflecting intent is committed; latency = t1 - t0.
Percentiles p50/p95/p99 over rolling window N=200 per session.

17.13.2 Main-thread constraints: during active typing, tasks > 50ms SHOULD NOT occur; if they occur, postpone non-critical work.

17.13.3 Editor latency: p50 <= 16ms MUST, p95 <= 32ms MUST, p99 <= 50ms SHOULD. Autosave MUST NOT cause p95 to exceed 32ms.

17.13.4 Navigation latency:
route change to first usable paint p95 <= 250ms in-memory MUST, <= 800ms localhost fetch MUST, <= 1500ms LAN SHOULD.
scene selection to content visible p95 <= 100ms in-memory MUST, <= 400ms localhost fetch MUST.

17.13.5 Scene ops:
move/reorder optimistic update within 100ms p95 MUST.
split/merge pending within 100ms p95 MUST; success navigation within 1200ms p95 localhost SHOULD, 2000ms p95 LAN SHOULD.
create/delete pending within 100ms p95 MUST; success navigation within 1200ms p95 localhost SHOULD.

17.13.6 Publish:
single scene publish p95 <= 1500ms localhost SHOULD, <= 3000ms LAN SHOULD.
batch publish p95 <= 2500ms localhost SHOULD, <= 5000ms LAN SHOULD.

17.13.7 MR and diff:
MR list first usable paint p95 <= 1000ms localhost MUST, <= 2000ms LAN MUST.
MR detail change list visible p95 <= 1200ms localhost MUST, <= 2500ms LAN MUST.
open single scene diff p95 <= 300ms cached MUST, <= 1200ms compute SHOULD.

17.13.8 Instrumentation: offline-safe; MAY record local metrics; if persisted, local-only and erasable; MUST NOT violate typing budgets.

17.14 Accessibility (required)

17.14.1 Visible focus indicators MUST exist.

17.14.2 Controls MUST be reachable by Tab in logical order.

17.14.3 Icon-only controls MUST have accessible labels.

17.14.4 UI MUST not rely on color alone to convey state.
	18.	Maintenance, Operational Profile, Failure Modes, and Runbook

18.1 Maintenance and import exclusivity (required)

18.1.1 The system MUST enforce mutual exclusion for all maintenance operations and all import operations.

18.1.2 At most one of the following may run at a time:
import (dry-run or real)
maintenance verify
maintenance pack
maintenance repair-order
maintenance prune-idempotency
maintenance init-admin
maintenance reset-password

18.1.3 If a conflicting operation is requested, the server MUST return 409 with code “MAINTENANCE_BUSY” and details including the active operation name if known.

18.1.4 The mutual exclusion mechanism MUST be crash-safe and MUST avoid stale-lock deadlocks (e.g., PID plus start-time, or a lease with renewal).

18.2 Disk space guard (required)

18.2.1 The system MUST implement a free-space guard with configurable threshold min_free_bytes (default SHOULD be 1 GiB).

18.2.2 If free bytes on the filesystem containing data-dir are below min_free_bytes:
18.2.2.1 /health MUST report degraded
18.2.2.2 all mutating operations MUST fail fast with HTTP 507 code “STORAGE_FULL” and details:
{ “op”: string, “path”: string|null, “free_bytes”: int_string|null, “required_bytes_estimate”: int_string|null }

18.2.3 Implementations MUST check free space at least:
once at startup
before starting any mutation transaction
before writing any new CAS object or pack segment
before starting export canonical meta.db construction

18.3 Process shutdown contract (required)

18.3.1 On SIGTERM or equivalent graceful shutdown signal, the server MUST:
stop accepting new requests
allow in-flight requests to complete up to shutdown_timeout_secs (default SHOULD be 30)
close SQLite connections cleanly

18.3.2 The server MUST NOT attempt background mutations during shutdown beyond completing in-flight work.

18.4 Supported backup method (single source of truth)

18.4.1 The only supported backup mechanism is the built-in deterministic archive export defined in Section 12.

18.4.2 Operators MUST NOT treat a raw copy of data-dir (including rsync, filesystem snapshot without verification, or manual file copy) as a supported backup method.

18.4.3 Implementations MUST document this restriction in built-in help text for serve/export/import.

18.5 Export requirements (operational hardening)

18.5.1 export MUST produce an archive that passes internal verification before returning success.

18.5.2 export MUST perform, at minimum, these checks prior to returning success:
manifest.json exists and is valid canonical JSON per Section 12.3
all manifest file checksums match the archived bytes
refs -> commits -> trees -> blobs reachability holds
order.json consistency holds for all chapters
canonical meta.db construction succeeded

18.5.3 If any check fails, export MUST fail with 500 code “EXPORT_VERIFY_FAILED” and MUST NOT output a partial archive as success.

18.6 Import requirements (operational hardening)

18.6.1 import MUST be atomic per data-dir.

18.6.2 import MUST support a dry-run mode (CLI and HTTP). In dry-run mode, MUST NOT modify target data-dir.

18.6.3 dry-run MUST perform the same validations as a real import, including checksum validation and internal consistency verification.

18.6.4 import MUST implement atomicity using:
extract to a fresh sibling directory, verify, then rename-swap the directory
or an equivalent mechanism providing the same atomic cutover semantics

18.6.5 If atomic cutover fails after extraction, the implementation MUST leave the original data-dir intact and MUST remove or quarantine the extracted temporary directory.

18.7 Verify and pack (required)

18.7.1 verify MUST check:
manifest and archive checksums during import
internal consistency: refs -> commits -> trees -> blobs exist
order.json consistency for all chapters
verify MUST NOT modify data.

18.7.2 pack MUST be explicit and MUST NOT run automatically; MUST NOT change any content IDs; SHOULD support dry_run.

18.8 Repair order.json (required recovery path)

18.8.1 CLI and HTTP repair-order MUST exist.

18.8.2 repair-order MUST be explicit and MUST NOT run automatically.

18.8.3 repair-order mode derive MUST:
read the chapter tree at head of provided ref
enumerate scenes in that chapter
sort by (scene.order_key, scene_id) ascending
write a new canonical order.json consistent with Section 8.7

18.8.4 repair-order MUST create exactly one new commit and update the target ref atomically.

18.8.5 repair-order MUST append an audit event with action “REPAIR_ORDER” and details including chapter_id and scene count.

18.9 Prune idempotency (required)

18.9.1 prune-idempotency MUST delete rows with created_at < (now - older-than) deterministically.

18.9.2 prune-idempotency MUST be explicit and MUST NOT run automatically in v0.0.1.

18.10 Init admin (required)

18.10.1 init-admin MUST succeed only if no users exist.

18.10.2 On success, init-admin MUST create exactly one admin user and MUST append an audit event “INIT_ADMIN”.

18.10.3 init-admin MUST fail with 409 code “ADMIN_ALREADY_INITIALIZED” if users already exist.

18.11 Reset password (required)

18.11.1 reset-password MUST append an audit event “RESET_PASSWORD” including user_id.

18.11.2 reset-password MUST NOT reveal password material in logs.

18.12 Health and readiness (required)

18.12.1 GET /health response MUST be:
{ “status”: “ok”|“degraded”|“fail”, “spec_version”: “0.0.1”, “checks”: { … } }

18.12.2 checks MUST include at least:
db_rw: bool
cas_rw: bool
free_space_ok: bool
schema_ok: bool
maintenance_lock_free: bool

18.12.3 status MUST be “fail” if schema_ok is false or db_rw is false.

18.13 Metrics (required minimum, offline-safe)

18.13.1 The server MUST expose GET /metrics.

18.13.2 /metrics MUST be text/plain; charset=utf-8 and MUST NOT require authentication by default (deployers may protect it at the reverse proxy layer).

18.13.3 /metrics format MUST be stable key-value lines:
 
{k=“v”,…}  MAY be used

18.13.4 /metrics MUST include at least:
storyed_http_requests_total{code,route}
storyed_op_latency_ms_p95{op}
storyed_sqlite_busy_total
storyed_cas_read_errors_total
storyed_cas_write_errors_total
storyed_free_bytes

18.14 Failure modes (required behavior)

18.14.1 Failures MUST be surfaced as deterministic error codes and MUST NOT silently corrupt state.

18.14.2 If a mutation fails after partial work, the server MUST roll back such that the visible ref state is unchanged.

18.14.3 CAS dangling references:
If a requested commit/tree/blob cannot be fully materialized due to missing referenced objects, return 500 code “CAS_DANGLING_REFERENCE” with details listing missing items. maintenance verify MUST detect this.

18.14.4 order.json corruption:
If order.json missing/invalid for chapter with scenes, return 500 code “ORDER_CORRUPT”. maintenance verify MUST report failing chapter_id and reason.

18.14.5 DB busy:
If SQLite busy beyond busy_timeout_ms, return 503 code “DB_BUSY” with details { op, busy_timeout_ms } and increment storyed_sqlite_busy_total.

18.15 Routine schedule (normative recommendation)

18.15.1 Operators SHOULD run:
export daily
maintenance verify at least weekly
import dry-run of latest export at least monthly

18.16 Disaster recovery runbook (normative)

18.16.1 Definitions:
backup artifact: an export archive produced by storyed export
restore target: a fresh empty directory intended to become the new data-dir

18.16.2 Backup procedure (daily):
run export daily and store archive on independent storage
keep at least N=7 daily backups and N=4 weekly backups

18.16.3 Restore drill (monthly):
import –dry-run against latest backup
real import into a non-production restore target
verify:
health ok
maintenance verify ok=true
sample repo read and a no-op publish completes

18.16.4 Actual restore procedure:
create fresh restore target directory
import –dry-run against chosen backup
if passes, real import into restore target
start serve pointing at restore target
run maintenance verify and require ok=true
preserve compromised original data-dir for forensics; do not modify it

End of v0.0.1
