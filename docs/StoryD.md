Title: storyed Integrated Spec v0.0.1 (Single Binary, Offline, CAS+DAG, Scene-level, Markdown, Lexo-rank, UTF-8, Pinned UI, Git/Worktree Interop)
	0.	Status
0.1 This document defines the frozen v0.0.1 integrated specification for storyed: a self-contained editor server for stories/documents and its pinned browser UI, including security headers, UI behavior, latency budgets, deterministic export/import, and Git/worktree interop.
0.2 Target: single host, offline/air-gapped capable, non-realtime collaboration via branches and merge requests (MR), with optional Git ecosystem interop via deterministic worktree export and import.
0.3 Normative keywords: MUST, MUST NOT, SHOULD, MAY.
0.4 v0.0.1 compliant implementations MUST implement exactly the ABIs and semantics in this document. No forward-compat behavior is defined in v0.0.1.
	1.	Goals
1.1 Provide a time-series reading UX (chapters -> scenes) on top of a Git-like internal model.
1.2 Provide immutable content-addressed storage (CAS) for blobs/trees/commits and a DAG history.
1.3 Provide scene-level diff/merge and first-class scene operations (create/edit/move/reorder/split/merge).
1.4 Provide single-binary deployment with embedded UI assets and local storage only.
1.5 Provide deterministic export/import that restores identical object IDs and repository state.
1.6 Provide pinned UI implementation and security profile such that UI correctness does not depend on external resources.
1.7 Provide pinned UI behavior, state machines, and measurable latency budgets for core flows.
1.8 Provide deterministic worktree export/import for Git and external editor interop with strong safety guards (no silent ref updates).
1.9 Provide predictable conflict handling such that 409 conflicts are normal flows with explicit user choices.
1.10 Eliminate operational rituals by standardizing operation receipts, preflight checks, and conflict-resolution operations that remain explicit but one-step.
	2.	Non-goals
2.1 Real-time concurrent editing (OT/CRDT).
2.2 Payments/marketplace/licensing enforcement.
2.3 Multi-region HA and global-scale operations.
2.4 Git wire protocol compatibility.
2.5 Required DAG visualization as a primary workflow.
2.6 Background or implicit server-side queueing of mutations while offline (mutations must be explicit).
2.7 Silent auto-resolution of 409 conflicts.
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
3.19 Operation receipt: a deterministic response record for a mutation, describing what changed and which ref/head was affected (see 19.7).
	4.	Packaging and Runtime (L3)
4.1 Single executable
4.1.1 The system MUST be delivered as a single executable file.
4.1.2 The executable MUST serve both:
4.1.2.1 HTTP API (JSON and selected binary downloads/uploads)
4.1.2.2 UI (static SPA assets embedded in the binary)

4.2 Commands (normative interface)
4.2.1 The executable MUST support:
4.2.1.1 serve –data-dir  –listen addr:port￼ [–config ]
4.2.1.2 export –data-dir  –out <file.tar.zst> [–repo <repo_id>|–all]
4.2.1.3 import –data-dir  –in <file.tar.zst> [–run-verify 0|1]
4.2.1.4 maintenance rebalance –repo <repo_id> –chapter <chapter_id> –ref <ref_name>
4.2.1.5 export-md –data-dir  –repo <repo_id> –ref <ref_name|commit_id> –out 
4.2.1.6 import-md –data-dir  –repo <repo_id> –ref <ref_name> –in  –expected-head <commit_id>
4.2.1.7 worktree add –data-dir  –repo <repo_id> –ref <ref_name> –path  –expected-head <commit_id|null>
4.2.1.8 worktree pull –data-dir  –path 
4.2.1.9 worktree push –data-dir  –path  –expected-head <commit_id>
4.2.1.10 lint –in  [–format text|json]
4.2.2 Implementations MUST NOT change the semantics of the above.
4.2.3 Implementations MAY provide additional commands, but MUST NOT change the behavior of required commands.

4.3 Offline requirement
4.3.1 The process MUST run offline (no mandatory external network calls).
4.3.2 UI assets MUST NOT require external CDNs for correctness.
	5.	Storage Layout
5.1 data-dir contents
5.1.1 data-dir MUST contain:
5.1.1.1 /meta.db : SQLite database
5.1.1.2 /objects/ : CAS objects (immutable) and optional pack files (48)
5.1.2 data-dir MAY contain:
5.1.2.1 /tmp/
5.1.2.2 /logs/ (optional)

5.2 SQLite mode
5.2.1 SQLite SHOULD use WAL mode for performance and crash safety.
5.2.2 Export MUST produce a consistent snapshot regardless of WAL usage (see 40.4).

5.3 CAS object file path (loose objects)
5.3.1 Loose CAS objects MUST be stored under:
5.3.1.1 /objects/sha256//
where  is the first 2 hex chars of  and  is the full lowercase sha256 hex string.
5.3.2 The system MUST NOT overwrite an existing loose CAS object file.
5.3.3 The system MUST write loose objects atomically (write temp then rename).
5.3.4 The system MUST fsync the final object file and its parent directory, or otherwise ensure durability equivalent to atomic rename on the target platform.

5.4 Garbage collection
5.4.1 GC MAY be omitted in v0.0.1.
5.4.2 If implemented, GC MUST be explicit (manual) and MUST NOT run automatically.
5.4.3 GC MUST NOT delete any object reachable from any ref, MR base_commit_id, or audit log references.
	6.	Identifiers and Canonical Forms
6.1 Stable IDs (textual form)
6.1.1 repo_id, chapter_id, scene_id, mr_id, user_id, event_id MUST be UUIDv7 strings.
6.1.2 Canonical UUID string form MUST be lowercase hex with hyphens: 8-4-4-4-12.

6.2 Content IDs
6.2.1 blob_id, tree_id, commit_id MUST be lowercase hex sha256 strings of length 64.
6.2.2 Internally, tree/commit objects store raw sha256 bytes(32); externally they are represented as hex strings.

6.3 Refs
6.3.1 Branch ref names MUST use refs/heads/.
6.3.2 Tag ref names MUST use refs/tags/.
6.3.3  MUST match regex: [A-Za-z0-9._-]{1,64}.
6.3.4 Ref names MUST be case-sensitive.
	7.	Text and Internationalization (UTF-8)
7.1 UTF-8
7.1.1 All user-visible text fields MUST be UTF-8 in API and storage.
7.1.2 The server MUST reject invalid UTF-8.

7.2 Unicode normalization
7.2.1 The server MUST normalize all stored text fields to Unicode NFC prior to canonicalization and hashing.

7.3 Forbidden characters
7.3.1 The server MUST reject the following characters in any stored text field:
7.3.1.1 U+0000 (NUL)
7.3.1.2 U+0001..U+001F and U+007F (control characters)
EXCEPT:
7.3.1.3 LF (U+000A) is permitted inside body_md and commit.message.
7.3.1.4 TAB (U+0009) is permitted inside body_md.
7.3.2 The server MUST reject bidi control characters:
7.3.2.1 U+202A..U+202E and U+2066..U+2069

7.4 Length limits (configurable)
7.4.1 The server MUST enforce configurable length limits.
7.4.2 Default SHOULD be:
7.4.2.1 chapter.title <= 256 code points
7.4.2.2 scene.title <= 256 code points
7.4.2.3 tags[i] <= 64 code points
7.4.2.4 entities[i] <= 128 code points
7.4.2.5 commit.message <= 2048 code points
7.4.2.6 body_md <= 5 MiB (bytes after LF normalization)
7.4.2.7 username/handle <= 64 code points

7.5 Error detail requirements for normalization failures
7.5.1 For any rejection due to 7.1-7.3, the server MUST return 400 with:
7.5.1.1 code: “TEXT_INVALID”
7.5.1.2 details: { field: , reason: , offset: <int|null> }
	8.	Canonicalization and Hash Stability (Required)
8.1 Canonicalization order (normative)
For any JSON-based stored object (Chapter, Scene, manifest.json, Audit details_json, ui_manifest.json, worktree guard files, order.json, and operation receipts when persisted):
8.1.1 Validate UTF-8.
8.1.2 Normalize to NFC for all text values.
8.1.3 Reject forbidden characters (7.3) after NFC.
8.1.4 Normalize newlines to LF for fields where LF normalization applies:
8.1.4.1 body_md
8.1.4.2 commit.message (CRLF/CR -> LF)
8.1.5 Apply JSON Canonicalization Scheme (JCS, RFC 8785).
8.1.6 Hash the resulting canonical UTF-8 bytes with sha256 to obtain blob_id when stored in CAS as a JSON blob.

8.2 JSON blobs
8.2.1 The server MUST canonicalize JSON blobs prior to storage as defined in 8.1.
8.2.2 The API MUST return canonicalized JSON when returning stored Chapter/Scene/manifest/ui_manifest/audit JSON blobs.

8.3 Tree/Commit encoding
8.3.1 Tree and Commit objects MUST be encoded using Canonical CBOR (RFC 8949 canonical rules).
8.3.2 Canonical CBOR output bytes are hashed for tree_id/commit_id.
8.3.3 The server MUST be the source of truth for canonicalization.
8.3.4 commit_id depends on created_at and message; identical trees can yield different commit_id values.
	9.	CAS Object Formats (Normative)
9.1 Blob
9.1.1 Blob is raw bytes stored exactly as written (post-canonicalization if the blob is a canonical JSON blob).
9.1.2 blob_id = sha256(blob_bytes).

9.2 Tree object (Canonical CBOR map)
9.2.1 Tree represents a flat mapping from absolute path -> blob_id.
9.2.2 Tree CBOR map MUST contain exactly:
9.2.2.1 “type” : “tree”
9.2.2.2 “entries” : array
9.2.3 Each entry MUST be a CBOR map containing exactly:
9.2.3.1 “path” : text (absolute path, ASCII; see 10)
9.2.3.2 “id” : bytes(32) (raw sha256 of the referenced blob)
9.2.4 “entries” MUST be sorted by “path” ascending using bytewise comparison of UTF-8 bytes.
9.2.5 Duplicate “path” values MUST be rejected (400).
9.2.6 Tree MUST NOT contain any additional fields in v0.0.1.

9.3 Commit object (Canonical CBOR map)
9.3.1 Commit CBOR map MUST contain exactly:
9.3.1.1 “type” : “commit”
9.3.1.2 “tree” : bytes(32)
9.3.1.3 “parents” : array of bytes(32) (0..n)
9.3.1.4 “author” : map { “user_id”: text(UUIDv7), “handle”: text|null }
9.3.1.5 “message” : text
9.3.1.6 “created_at” : int (unix seconds, UTC)
9.3.2 Commit MUST NOT include any additional fields (metadata fields are forbidden in v0.0.1).
9.3.3 “parents” ordering MUST be deterministic: sort by raw bytes(32) ascending.
9.3.4 Implementations MAY store additional operational metadata in meta.db, but MUST NOT affect commit_id/tree_id derivation.
	10.	Tree Paths (Normative)
10.1 Layout (normative for v0.0.1)
10.1.1 The repository tree layout MUST be:
10.1.1.1 /chapters/<chapter_id>.json -> blob (Chapter JSON)
10.1.1.2 /chapters/<chapter_id>/scenes/<scene_id>.json -> blob (Scene JSON)
10.1.1.3 /chapters/<chapter_id>/order.json -> blob (Order JSON, 15.7)

10.2 ID segments
10.2.1 <chapter_id> and <scene_id> MUST be canonical UUIDv7 strings (lowercase).

10.3 Path restrictions
10.3.1 Paths MUST start with “/”.
10.3.2 Segments MUST NOT be empty.
10.3.3 Segments MUST NOT be “.” or “..”.
10.3.4 Paths MUST NOT contain backslash “".
10.3.5 Paths MUST be ASCII-only in v0.0.1 (by construction of the layout).
10.3.6 Implementations MUST reject any attempt to store objects under non-conforming paths.
	11.	Repository Model
11.1 Repository creation
11.1.1 Creating a repo MUST:
11.1.1.1 allocate repo_id
11.1.1.2 create an initial empty tree whose entries array is empty
11.1.1.3 create an initial commit referencing that empty tree
11.1.1.4 create refs/heads/main pointing to that commit

11.2 Ref updates
11.2.1 Ref updates MUST be atomic (single SQLite transaction).
11.2.2 Ref update MUST implement compare-and-swap semantics when a caller supplies expected_old_commit_id.
11.2.3 If expected_old_commit_id is non-null and does not match the current ref value, the server MUST return 409 conflict.

11.3 Operation head guards (required for v0.0.1 mutations)
11.3.1 All endpoints that mutate a repo ref (including MR merge and all ops/* endpoints, publish flows, and worktree import) MUST accept an optional expected_head_commit_id.
11.3.2 If expected_head_commit_id is provided and does not match the current head commit_id of the target ref at operation start, the server MUST return 409 with code “REF_HEAD_MISMATCH”.
11.3.3 The server MUST include in details at least:
11.3.3.1 { “ref”: <ref_name>, “expected”: <sha256_hex>, “actual”: <sha256_hex> }.
	12.	Chapter JSON (Normative)
12.1 Chapter blob MUST be a JCS-canonical JSON object containing exactly these members:
12.1.1 “chapter_id”: string(UUIDv7)
12.1.2 “title”: string
12.1.3 “summary”: string|null
12.1.4 “constraints”: object (12.2)
12.1.5 “tags”: array of string
12.1.6 “order_key”: string (lexo-rank, 15)

12.2 constraints object (Chapter)
12.2.1 “constraints” MUST be a JSON object containing exactly:
12.2.1.1 “rating”: “general”|“r15”|“r18”
12.2.1.2 “flags”: array of string
12.2.2 “flags” MAY be empty but MUST be present (no null).
12.2.3 “tags” MAY be empty but MUST be present (no null).

12.3 Path/ID consistency
12.3.1 chapter_id MUST equal the <chapter_id> in its path.
	13.	Scene JSON (Normative)
13.1 Scene blob MUST be a JCS-canonical JSON object containing exactly these members:
13.1.1 “scene_id”: string(UUIDv7)
13.1.2 “chapter_id”: string(UUIDv7)
13.1.3 “order_key”: string (lexo-rank, 15)
13.1.4 “title”: string|null
13.1.5 “body_md”: string (Markdown, UTF-8, LF-normalized)
13.1.6 “tags”: array of string
13.1.7 “entities”: array of string
13.1.8 “constraints”: object (13.2)
13.1.9 “provenance”: object (13.3)

13.2 constraints object (Scene)
13.2.1 “constraints” MUST be a JSON object containing exactly:
13.2.1.1 “rating”: “general”|“r15”|“r18”
13.2.1.2 “flags”: array of string

13.3 provenance object (Scene)
13.3.1 “provenance” MUST be a JSON object containing exactly:
13.3.1.1 “op”: “create”|“edit”|“split_from”|“merge_of”|“move”
13.3.1.2 “parents”: array of object (13.4)
13.3.2 provenance.parents MAY be empty only when op=“create”.
13.3.3 provenance.parents MUST be non-empty for op in {“edit”,“split_from”,“merge_of”,“move”}.

13.4 provenance parent entry
13.4.1 Each entry in provenance.parents MUST be a JSON object containing exactly:
13.4.1.1 “scene_id”: string(UUIDv7)
13.4.1.2 “commit_id”: string(sha256_hex)
13.4.2 provenance.parents MUST NOT contain duplicates of (scene_id, commit_id).

13.5 Path/ID consistency
13.5.1 scene_id MUST equal the <scene_id> in its path.
13.5.2 chapter_id MUST equal the <chapter_id> in its path.
	14.	Markdown Rules
14.1 CommonMark
14.1.1 body_md MUST be rendered as CommonMark-compatible Markdown.

14.2 Newline normalization
14.2.1 The server MUST normalize stored body_md newlines to LF (”\n”).
14.2.2 Incoming CRLF and CR MUST be converted to LF before canonicalization and storage.

14.3 Raw HTML
14.3.1 Raw HTML in Markdown MUST be disabled in v0.0.1 (MUST NOT be rendered as HTML).

14.4 XSS and link policy
14.4.1 Rendered output MUST be XSS-safe.
14.4.2 The UI MUST NOT execute scripts from rendered content.
14.4.3 The renderer MUST NOT emit links with forbidden schemes (case-insensitive):
14.4.3.1 javascript:
14.4.3.2 data:
14.4.3.3 vbscript:
14.4.4 The renderer SHOULD allow only:
14.4.4.1 http
14.4.4.2 https
14.4.4.3 mailto
14.4.4.4 relative URLs (including fragment-only)

14.5 Images (required alignment with CSP)
14.5.1 The Markdown renderer MUST NOT emit  elements for external URLs.
14.5.2 v0.0.1 default behavior MUST be: render images as plain links (no ), regardless of URL.
	15.	Ordering (Fixed, v0.0.1)
15.1 Lexo-rank keys
15.1.1 order_key MUST be exactly 16 characters.
15.1.2 Characters MUST be base62: 0-9A-Za-z.
15.1.3 Comparison MUST be lexicographic by ASCII byte order over the 16 bytes.

15.2 Digit mapping (normative)
15.2.1 The base62 alphabet is:
15.2.1.1 DIGITS: “0123456789”
15.2.1.2 UPPER:  “ABCDEFGHIJKLMNOPQRSTUVWXYZ”
15.2.1.3 LOWER:  “abcdefghijklmnopqrstuvwxyz”
15.2.2 digit(“0”)=0 .. digit(“9”)=9, digit(“A”)=10 .. digit(“Z”)=35, digit(“a”)=36 .. digit(“z”)=61.

15.3 Sentinels
15.3.1 left_sentinel  = “0000000000000000”
15.3.2 right_sentinel = “zzzzzzzzzzzzzzzz”
15.3.3 default_mid_digit = “U” (digit 30)

15.4 Between(left, right) (deterministic, fixed width)
15.4.1 Inputs:
15.4.1.1 left_key MAY be null (means -infinity -> left_sentinel)
15.4.1.2 right_key MAY be null (means +infinity -> right_sentinel)
15.4.1.3 If both non-null, left_key MUST be < right_key.
15.4.2 Output MUST be a 16-char key strictly between.
15.4.3 Algorithm:
15.4.3.1 Let L = (left_key or left_sentinel), R = (right_key or right_sentinel).
15.4.3.2 For i in 0..15:
	•	li = digit(L[i])
	•	ri = digit(R[i])
	•	If li == ri: output L[i] at i and continue
	•	If ri - li >= 2:
output[i] = value(floor((li+ri)/2))
output[i+1..15] = default_mid_digit
return output
	•	If ri - li == 1:
output[i] = L[i]
continue
15.4.3.3 If the loop completes, there is no space at width 16:
	•	return error ORDER_KEY_SPACE_EXHAUSTED (HTTP 409).
15.4.4 Between(null, null) MUST return “UUUUUUUUUUUUUUUU”.

15.5 Rebalance(chapter_id) (deterministic)
15.5.1 Rebalance MUST assign fresh order_key values to all scenes within a chapter and MUST create a new commit.
15.5.2 Rebalance input order is the current stable order defined by order.json (15.7) at the base commit.
15.5.3 Rebalance output algorithm:
15.5.3.1 Define GAP = 62^4
15.5.3.2 For scene index i from 1..n:
	•	key_num = i * GAP
	•	order_key = base62_encode_fixed(key_num, width=16, pad=“0”)
15.5.4 base62_encode_fixed encodes most significant digit first; left-pad with “0” to width 16.

15.6 Stable sorting (UI)
15.6.1 Chapter order: (chapter.order_key, chapter_id).
15.6.2 Scene order: order.json order, with tie-break by (scene.order_key, scene_id) if needed for resilience.

15.7 Order JSON (Normative)
15.7.1 /chapters/<chapter_id>/order.json MUST exist for each chapter that contains at least one scene.
15.7.2 order.json MUST be a JCS-canonical JSON object containing exactly:
15.7.2.1 “chapter_id”: string(UUIDv7)
15.7.2.2 “items”: array of object
15.7.3 Each item MUST be a JSON object containing exactly:
15.7.3.1 “scene_id”: string(UUIDv7)
15.7.3.2 “order_key”: string(16 base62)
15.7.4 items MUST NOT contain duplicate scene_id.
15.7.5 For any scene in the chapter tree, order.json MUST contain an item for that scene_id.
15.7.6 order.json MUST NOT reference any scene_id not present in the chapter tree.
15.7.7 If order.json is missing or invalid for a chapter with scenes, the server MUST return 500 with code “ORDER_CORRUPT” and MUST NOT guess.

15.8 Rebalance pressure (operational signal, non-canonical)
15.8.1 Implementations SHOULD compute and expose per-chapter “rebalance_pressure” metrics derived from order_key density, without changing canonical objects.
15.8.2 Metrics MUST NOT affect commit_id/tree_id derivation.
	16.	Diff Semantics (Scene-first)
16.1 Diff(base, head) MUST report:
16.1.1 chapters: added/deleted/modified/reordered
16.1.2 scenes: added/deleted/modified/moved/reordered

16.2 Scene classification (normative)
16.2.1 modified: same scene_id exists in both trees, blob differs
16.2.2 added: only in head tree
16.2.3 deleted: only in base tree
16.2.4 moved: same scene_id exists in both trees, but chapter_id/path differs
16.2.5 reordered: order differs per order.json or order_key differs

16.3 Modified scene diff MUST include:
16.3.1 body_md line diff (LF normalized)
16.3.2 structured diff for: title, tags, entities, constraints, order_key, chapter_id, provenance

16.4 Line diff algorithm (required minimum)
16.4.1 The server MUST provide a stable, deterministic line-based diff.
16.4.2 Implementations MAY choose the algorithm; output MUST be deterministic given the same inputs.
16.4.3 If multiple minimal diffs are possible, the implementation MUST apply a deterministic tie-break rule:
16.4.3.1 prefer earliest (leftmost) matching anchors
16.4.3.2 prefer shorter edit scripts if multiple remain

16.5 Diff API split (required)
16.5.1 The server MUST provide:
16.5.1.1 summary diff: classification only (no body diff)
16.5.1.2 scene diff: per-scene body and structured diffs
16.5.2 summary diff MUST be O(number of changed paths) and MUST NOT compute body line diffs.

16.6 Diff caching (implementation freedom)
16.6.1 Implementations MAY cache diff-scene results keyed by (base_commit_id, head_commit_id, scene_id, base_blob_id, head_blob_id).
16.6.2 Cache MUST NOT affect determinism of API outputs.
	17.	Merge Requests (MR)
17.1 MR record MUST include:
17.1.1 mr_id, repo_id
17.1.2 base_ref, head_ref
17.1.3 base_commit_id (snapshot at MR creation)
17.1.4 status: open|merged|closed
17.1.5 checks: array of { name, status, details? }

17.2 Merge modes
17.2.1 fast-forward: ref update only.
17.2.2 merge: 3-way merge commit (new commit with 2+ parents).
17.2.3 squash: single new commit capturing head changes.

17.3 merge_base (deterministic)
17.3.1 merge_base MUST be computed as a common ancestor in the commit DAG.
17.3.2 If multiple common ancestors exist, the implementation MUST choose deterministically:
17.3.2.1 Define dist(X, A) as the length of the shortest parent-edge path from commit X to ancestor A.
17.3.2.2 For each common ancestor A, compute tuple:
T(A) = (
max(dist(base_head, A), dist(head_head, A)),
dist(base_head, A) + dist(head_head, A),
commit_id(A)
)
17.3.2.3 Choose A with lexicographically smallest T(A).

17.4 Conflict detection (scene-level)
17.4.1 content conflict: body_md changed on both sides since merge_base
17.4.2 meta conflict: title/tags/entities/constraints/provenance changed on both sides since merge_base
17.4.3 order conflict: order changed on both sides since merge_base

17.5 Default resolution policy
17.5.1 order conflict defaults to head, but MUST be overrideable per scene (base/head/manual).
17.5.2 content/meta conflicts MUST require explicit resolution by user (MR UI/API).

17.6 Merge result
17.6.1 Conflict resolutions MUST result in a new commit (immutable history).
17.6.2 The merge operation MUST update base_ref to the resulting commit_id atomically.
17.6.3 MR merge MUST honor expected_head_commit_id (11.3) for base_ref updates.
17.6.4 MR merge MUST return an operation receipt (19.7).
	18.	Scene Operations (First-class)
18.0 Common rules for ops/*
18.0.1 All ops/* endpoints MUST:
18.0.1.1 accept “ref” (refs/heads/* only) as the target ref
18.0.1.2 accept optional “expected_head_commit_id” (11.3)
18.0.1.3 produce exactly one new commit on success
18.0.1.4 update the target ref to that commit atomically
18.0.1.5 return previous_head_commit_id for reversible UX (18.0.3)
18.0.1.6 return an operation receipt (19.7) describing head before/after and changed items
18.0.2 If an op requires ORDER_KEY_SPACE_EXHAUSTED handling:
18.0.2.1 the server MUST compute the rebalance result and the requested op result against that rebalance in-memory
18.0.2.2 the server MUST commit the final state as a single commit
18.0.2.3 the server MUST NOT create an intermediate visible commit for the rebalance
18.0.3 All successful ops/* responses MUST include:
18.0.3.1 { commit_id, updated_ref, previous_head_commit_id }

18.1 Edit (semantic rules for scene edits, regardless of API shape)
18.1.1 scene_id MUST be preserved.
18.1.2 provenance.op=“edit”.
18.1.3 provenance.parents MUST include prior (scene_id, commit_id) for the edited scene.

18.2 Move/Reorder
18.2.1 Move across chapters MUST update chapter_id/path.
18.2.2 Reorder MUST update order_key and update order.json accordingly.
18.2.3 provenance.op=“move”, parents include prior.

18.3 Split (A -> A + B..)
18.3.1 Source scene A remains (typically first segment).
18.3.2 New scenes B.. MUST use new scene_ids.
18.3.3 All involved scenes MUST set provenance.op=“split_from”.
18.3.4 provenance.parents MUST reference A@previous_commit.
18.3.5 order_keys MUST reflect intended reading order and order.json MUST be updated.
18.3.6 Split offsets are expressed as byte offsets into LF-normalized UTF-8 bytes of body_md:
18.3.6.1 strictly increasing
18.3.6.2 in range 1..len(body_bytes)-1
18.3.6.3 on a valid UTF-8 codepoint boundary
If violated, server MUST return 400 invalid input.
18.3.7 If ORDER_KEY_SPACE_EXHAUSTED occurs during split, apply 18.0.2.

18.4 Merge Scenes (A,B.. -> N)
18.4.1 Create new scene N with new scene_id.
18.4.2 Remove source scenes from resulting tree (no tombstones).
18.4.3 N sets provenance.op=“merge_of” with parents referencing each source scene_id and commit_id.
18.4.4 order.json MUST be updated to place N according to left/right anchors.

18.5 Publish-scene operation (required, optimized)
18.5.1 The server MUST provide an operation endpoint that performs the full publish sequence server-side:
18.5.1.1 It MUST create the scene blob (canonicalize on server), create the new tree, create the new commit, and update the ref atomically.
18.5.2 It MUST honor expected_head_commit_id (11.3).
18.5.3 It MUST produce exactly one new commit on success.

18.6 Preflight operations (ritual elimination, no mutation)
18.6.1 The server MUST provide preflight endpoints that validate inputs and predict effects without mutating state:
18.6.1.1 preflight-publish (single scene or staged set)
18.6.1.2 preflight-worktree-import
18.6.2 Preflight MUST:
18.6.2.1 validate text rules (7), markdown rules (14), and ordering consistency (15.7) as applicable
18.6.2.2 validate expected_head_commit_id if provided; on mismatch return 409 REF_HEAD_MISMATCH
18.6.2.3 return predicted changed_paths and changed_scene_ids deterministically
18.6.2.4 return stable error codes and locations similar to lint (43.6)
18.6.3 Preflight MUST NOT create blobs, trees, commits, audit rows, or idempotency records.

18.7 Explicit conflict-resolution operations (ritual elimination, still explicit)
18.7.1 The server MUST provide an explicit “rebase-staged” operation to reduce multi-step 409 handling into a single explicit call.
18.7.2 rebase-staged MUST:
18.7.2.1 require base_head_commit_id (the head at stage time)
18.7.2.2 require expected_head_commit_id (the current head expected by caller)
18.7.2.3 compute latest head; if mismatch, return 409 REF_HEAD_MISMATCH
18.7.2.4 apply staged changes in deterministic order (same as batch publish)
18.7.2.5 on conflict, return 409 with code “REBASE_CONFLICT” and include conflict descriptors per scene/field
18.7.2.6 on success, create exactly one new commit and update the ref atomically, returning a receipt
18.7.3 rebase-staged MUST NOT auto-resolve conflicts.
	19.	API (HTTP) (Normative)
19.1 OpenAPI
19.1.1 MUST expose OpenAPI at /openapi.json.

19.2 Common headers
19.2.1 All responses SHOULD include X-Request-Id.
19.2.2 All mutating requests SHOULD support Idempotency-Key.
19.2.3 Server MUST store idempotency results for at least 24 hours (configurable), keyed by:
19.2.3.1 (actor_id, method, path, idempotency_key).
19.2.4 On idempotent replay, the server MUST return the stored response status code and body, and MUST NOT perform the mutation again.

19.3 Auth and sessions (required)
19.3.1 Mutating endpoints MUST require authentication.
19.3.2 Read endpoints MAY be public depending on repo ACL, but MUST enforce ACL.
19.3.3 The server MUST support cookie-based sessions:
19.3.3.1 session cookie MUST be HttpOnly
19.3.3.2 session cookie MUST be SameSite=Lax or SameSite=Strict
19.3.3.3 session cookie SHOULD be Secure when served over HTTPS
19.3.4 The server MUST NOT require any external IdP for correctness.

19.4 Error model (normative)
19.4.1 Error JSON: { “code”: string, “message”: string, “details”?: any }
19.4.2 HTTP mapping:
19.4.2.1 400 invalid input
19.4.2.2 401 unauthenticated
19.4.2.3 403 unauthorized
19.4.2.4 404 not found
19.4.2.5 409 conflict (includes ORDER_KEY_SPACE_EXHAUSTED and REF_HEAD_MISMATCH)
19.4.2.6 413 payload too large
19.4.2.7 429 rate limited
19.4.2.8 500 internal error

19.5 Endpoints (required)
19.5.1 Core:
	•	GET  /health
	•	POST /auth/login
	•	POST /auth/logout
	•	GET  /auth/me
19.5.2 Repos and refs:
	•	POST /repos
	•	GET  /repos/{repo_id}
	•	POST /repos/{repo_id}/refs
	•	GET  /repos/{repo_id}/refs
	•	GET  /repos/{repo_id}/head?ref=<ref_name>
19.5.3 CAS:
	•	POST /blobs
	•	GET  /blobs/{blob_id}
	•	POST /trees
	•	GET  /trees/{tree_id}
19.5.4 Commits and diffs:
	•	POST /repos/{repo_id}/commits
	•	GET  /repos/{repo_id}/commits/{commit_id}
	•	GET  /repos/{repo_id}/diff-summary?base=<ref|commit>&head=<ref|commit>
	•	GET  /repos/{repo_id}/diff-scene?base=<ref|commit>&head=<ref|commit>&scene_id=
19.5.5 MR:
	•	POST /repos/{repo_id}/mrs
	•	GET  /repos/{repo_id}/mrs
	•	GET  /repos/{repo_id}/mrs/{mr_id}
	•	POST /repos/{repo_id}/mrs/{mr_id}/merge
	•	POST /repos/{repo_id}/mrs/{mr_id}/cherry-pick
19.5.6 Ops:
	•	POST /repos/{repo_id}/rank/between
	•	POST /repos/{repo_id}/rank/rebalance
	•	POST /repos/{repo_id}/ops/publish-scene
	•	POST /repos/{repo_id}/ops/publish-staged
	•	POST /repos/{repo_id}/ops/preflight-publish
	•	POST /repos/{repo_id}/ops/rebase-staged
	•	POST /repos/{repo_id}/ops/split-scene
	•	POST /repos/{repo_id}/ops/merge-scenes
	•	POST /repos/{repo_id}/ops/move-scene
	•	POST /repos/{repo_id}/ops/revert-ref
19.5.7 Admin and ACL:
	•	GET  /users
	•	POST /users
	•	GET  /users/{user_id}
	•	GET  /repos/{repo_id}/acl
	•	PUT  /repos/{repo_id}/acl
19.5.8 Audit and export/import:
	•	GET  /repos/{repo_id}/audit?after_ts=<int|null>&limit=
	•	GET  /export?repo_id=<UUIDv7|null>
	•	POST /import?run_verify=0|1
19.5.9 Worktree and MD interop:
	•	GET  /repos/{repo_id}/worktree/export?ref=<ref|commit>
	•	POST /repos/{repo_id}/worktree/preflight-import?ref=&expected_head_commit_id=
	•	POST /repos/{repo_id}/worktree/import?ref=&expected_head_commit_id=
19.5.10 UI serving:
	•	GET  /ui/
	•	GET  /ui/index.html
	•	GET  /ui/ui_manifest.json
	•	GET  /ui/assets/*
19.5.11 Maintenance and diagnostics:
	•	POST /maintenance/pack?dry_run=0|1
	•	POST /maintenance/verify
	•	GET  /system/diagnostics

19.6 Endpoint schemas (normative minimum)
19.6.1 GET /health
Response 200: { “status”: “ok”, “spec_version”: “0.0.1” }

19.6.2 POST /auth/login
Request: { “handle”: string, “password”: string }
Response 200: { “user_id”: UUIDv7, “handle”: string, “role_summary”: { “is_admin”: bool } }
Rules:
	•	On success, server MUST set session cookie.
	•	On failure, server MUST return 401 with code “AUTH_INVALID”.

19.6.3 POST /auth/logout
Response 200: { “ok”: true }
Rules: server MUST clear session cookie (expired).

19.6.4 GET /auth/me
Response 200:
{ “user_id”: UUIDv7, “handle”: string, “roles”: [{ “repo_id”: UUIDv7, “role”: string }], “is_admin”: bool }
Response 401 if no valid session.

19.6.5 POST /repos
Request: { “name”: string|null }
Response 201: { “repo_id”: UUIDv7, “default_ref”: “refs/heads/main”, “head_commit_id”: sha256_hex }

19.6.6 GET /repos/{repo_id}
Response 200: { “repo_id”: UUIDv7, “name”: string|null, “default_ref”: string, “head_commit_id”: sha256_hex }

19.6.7 GET /repos/{repo_id}/head?ref=…
Response 200: { “ref_name”: string, “commit_id”: sha256_hex }

19.6.8 POST /repos/{repo_id}/refs
Request:
{ “ref_name”: string, “target_commit_id”: sha256_hex, “expected_old_commit_id”: sha256_hex|null }
Rules:
	•	ref_name MUST match: ^refs/(heads|tags)/[A-Za-z0-9._-]{1,64}$
Response 200: { “ref_name”: string, “commit_id”: sha256_hex }

19.6.9 GET /repos/{repo_id}/refs
Response 200:
{ “refs”: [{ “ref_name”: string, “commit_id”: sha256_hex, “updated_at”: int }] }
Rules:
	•	refs MUST be sorted by ref_name ascending (bytewise ASCII).

19.6.10 POST /blobs
Request: raw bytes, Content-Type required
Response 201: { “blob_id”: sha256_hex, “size”: int, “content_type”: string }
Rules:
	•	The server MUST store a normalized content_type value in meta.db and return the normalized value:
	•	trim leading/trailing ASCII whitespace
	•	reject NUL or control characters (7.3)
	•	additionally lowercase type/subtype deterministically

19.6.11 GET /blobs/{blob_id}
Response 200: raw bytes of the blob
Rules:
	•	Content-Type SHOULD be the stored normalized content_type if known; otherwise application/octet-stream.

19.6.12 POST /trees
Request: { “entries”: [ { “path”: string, “blob_id”: sha256_hex } ] }
Rules:
	•	path MUST comply with 10.3 and MUST be unique within request.
	•	The server MUST reject any blob_id not present in CAS (404, code “CAS_BLOB_NOT_FOUND”).
	•	The server MUST sort entries by path bytewise before canonical CBOR encoding.
Response 201: { “tree_id”: sha256_hex }

19.6.13 GET /trees/{tree_id}
Response 200: { “tree_id”: sha256_hex, “entries”: [ { “path”: string, “blob_id”: sha256_hex } ] }
Rules:
	•	returned list MUST be sorted by path bytewise.

19.6.14 POST /repos/{repo_id}/commits
Request:
{ “tree_id”: sha256_hex, “parents”: [sha256_hex…], “author”: { “user_id”: UUIDv7, “handle”: string|null }, “message”: string, “created_at”: int }
Rules:
	•	tree_id MUST exist (404, code “CAS_TREE_NOT_FOUND”).
	•	each parent commit_id MUST exist (404, code “CAS_COMMIT_NOT_FOUND”).
	•	server MUST sort parents by raw bytes ascending before canonical encoding.
Response 201: { “commit_id”: sha256_hex }

19.6.15 GET /repos/{repo_id}/commits/{commit_id}
Response 200:
{ “commit_id”: sha256_hex, “tree_id”: sha256_hex, “parents”: [sha256_hex…], “author”: { “user_id”: UUIDv7, “handle”: string|null }, “message”: string, “created_at”: int }

19.6.16 GET /repos/{repo_id}/diff-summary
Response 200:
{
“base”: { “kind”: “ref”|“commit”, “id”: string },
“head”: { “kind”: “ref”|“commit”, “id”: string },
“chapters”: { “added”: [UUIDv7…], “deleted”: [UUIDv7…], “modified”: [UUIDv7…], “reordered”: [UUIDv7…] },
“scenes”: { “added”: [UUIDv7…], “deleted”: [UUIDv7…], “modified”: [UUIDv7…], “moved”: [UUIDv7…], “reordered”: [UUIDv7…] }
}

19.6.17 GET /repos/{repo_id}/diff-scene
Response 200:
{
“scene_id”: UUIDv7,
“base_commit_id”: sha256_hex,
“head_commit_id”: sha256_hex,
“body_diff”: { “algorithm”: string, “hunks”: any },
“meta_diff”: any,
“order_diff”: any
}

19.6.18 POST /repos/{repo_id}/mrs
Request: { “base_ref”: string, “head_ref”: string }
Response 201:
{ “mr_id”: UUIDv7, “repo_id”: UUIDv7, “base_ref”: string, “head_ref”: string, “base_commit_id”: sha256_hex, “status”: “open” }

19.6.19 POST /repos/{repo_id}/mrs/{mr_id}/merge
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
	•	expected_base_head_commit_id, if provided, MUST be enforced as expected_head_commit_id against base_ref (11.3).
Response 200: { “merged_commit_id”: sha256_hex, “receipt”: object(19.7) }

19.6.20 POST /repos/{repo_id}/mrs/{mr_id}/cherry-pick
Request: { “scene_ids”: [UUIDv7…], “target_ref”: string, “expected_head_commit_id”: sha256_hex|null }
Response 200: { “commit_id”: sha256_hex, “updated_ref”: string, “previous_head_commit_id”: sha256_hex, “receipt”: object(19.7) }

19.6.21 POST /repos/{repo_id}/rank/between
Request: { “left_key”: string|null, “right_key”: string|null }
Response 200: { “order_key”: string }

19.6.22 POST /repos/{repo_id}/rank/rebalance
Request: { “chapter_id”: UUIDv7, “ref”: string, “expected_head_commit_id”: sha256_hex|null }
Response 200: { “commit_id”: sha256_hex, “updated_ref”: string, “previous_head_commit_id”: sha256_hex, “receipt”: object(19.7) }

19.6.23 POST /repos/{repo_id}/ops/publish-scene
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
	•	fields MUST be validated per 7 and 14; server is source of truth for canonicalization (8.1).
	•	server MUST update provenance.op=“edit” and include parent (scene_id, head_before) deterministically.
Response 200: { “commit_id”: sha256_hex, “updated_ref”: string, “previous_head_commit_id”: sha256_hex, “receipt”: object(19.7) }

19.6.24 POST /repos/{repo_id}/ops/publish-staged
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
	•	items MUST be applied in deterministic order:
	•	sort by (chapter_id, order_key, scene_id) from the base_head_commit_id snapshot.
	•	server MUST reject if base_head_commit_id does not match the committed state used to derive the order (409, code “STAGE_BASE_MISMATCH”).
Response 200: { “commit_id”: sha256_hex, “updated_ref”: string, “previous_head_commit_id”: sha256_hex, “receipt”: object(19.7) }

19.6.25 POST /repos/{repo_id}/ops/preflight-publish
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
“errors”: [ { “code”: string, “path”?: string, “field”?: string, “message”: string, “offset”: int|null } ],
“predicted”: {
“changed_paths”: [string…],
“changed_scene_ids”: [UUIDv7…]
}
}
Rules:
	•	MUST NOT mutate any persistent state.

19.6.26 POST /repos/{repo_id}/ops/rebase-staged
Request:
{
“ref”: string,
“expected_head_commit_id”: sha256_hex,
“base_head_commit_id”: sha256_hex,
“items”: [object…]
}
Response 200: { “commit_id”: sha256_hex, “updated_ref”: string, “previous_head_commit_id”: sha256_hex, “receipt”: object(19.7) }
Response 409: code “REBASE_CONFLICT” with details describing per-scene conflict types.

19.6.27 POST /repos/{repo_id}/ops/move-scene
Request:
{
“ref”: string,
“expected_head_commit_id”: sha256_hex|null,
“scene_id”: UUIDv7,
“target_chapter_id”: UUIDv7,
“left_scene_id”: UUIDv7|null,
“right_scene_id”: UUIDv7|null
}
Response 200: { “commit_id”: sha256_hex, “updated_ref”: string, “previous_head_commit_id”: sha256_hex, “new_order_key”: string, “receipt”: object(19.7) }

19.6.28 POST /repos/{repo_id}/ops/split-scene
Request:
{
“ref”: string,
“expected_head_commit_id”: sha256_hex|null,
“scene_id”: UUIDv7,
“splits”: [ { “byte_offset”: int } ],
“titles”: [string|null] | null
}
Response 200: { “commit_id”: sha256_hex, “updated_ref”: string, “previous_head_commit_id”: sha256_hex, “new_scene_ids”: [UUIDv7…], “receipt”: object(19.7) }

19.6.29 POST /repos/{repo_id}/ops/merge-scenes
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
	•	joiner MUST be processed under 7 and 8.1 (UTF-8, NFC, forbidden chars).
Response 200: { “commit_id”: sha256_hex, “updated_ref”: string, “previous_head_commit_id”: sha256_hex, “merged_scene_id”: UUIDv7, “receipt”: object(19.7) }

19.6.30 POST /repos/{repo_id}/ops/revert-ref
Request:
{
“ref”: string,
“expected_head_commit_id”: sha256_hex|null,
“to_commit_id”: sha256_hex,
“message”: string
}
Rules:
	•	MUST create a new commit whose tree equals to_commit_id.tree and update ref atomically.
Response 200: { “commit_id”: sha256_hex, “updated_ref”: string, “previous_head_commit_id”: sha256_hex, “receipt”: object(19.7) }

19.6.31 GET /repos/{repo_id}/audit
Response 200:
{
“events”: [
{ “event_id”: UUIDv7, “ts”: int, “actor_id”: UUIDv7, “action”: string, “repo_id”: UUIDv7|null, “details_json”: object }
],
“next_after_ts”: int|null
}
Rules:
	•	events MUST be ordered by (ts, event_id) ascending.
	•	details_json MUST be canonicalized per 8.1 when stored; API returns canonical form.

19.6.32 GET /export
Response 200:
	•	Content-Type: application/zstd
	•	Body: tar.zst bytes as defined in 40
Rules:
	•	Access MUST require admin.
	•	repo_id query parameter MAY be null; if null, export all repos.
	•	export bytes MUST correspond exactly to deterministic archive rules (40.6).

19.6.33 POST /import?run_verify=0|1
Request:
	•	Content-Type: application/zstd
	•	Body: tar.zst bytes
Response 200: { “ok”: true, “imported_repo_ids”: [UUIDv7…], “verify”: object|null }
Rules:
	•	Access MUST require admin.
	•	Import MUST be atomic per data-dir (40.5.4).
	•	Import MUST fail on checksum mismatch (40.5.3).
	•	If run_verify=1, server MUST run verify after import and return results.

19.6.34 Worktree endpoints
19.6.34.1 GET /repos/{repo_id}/worktree/export?ref=<ref|commit>
Response 200: tar.zst or zip bytes (implementation choice), containing worktree layout (43.2)
Rules:
	•	Export MUST be deterministic for the same (repo_id, view id) state.
	•	Export MUST include .storyed/worktree.json with export_ts=0.

19.6.34.2 POST /repos/{repo_id}/worktree/preflight-import
Request: { “ref”: string, “expected_head_commit_id”: sha256_hex, “archive_bytes”: bytes }
Response 200: same schema as 19.6.25 with mode=“worktree”
Rules:
	•	MUST NOT mutate persistent state.

19.6.34.3 POST /repos/{repo_id}/worktree/import
Request: { “ref”: string, “expected_head_commit_id”: sha256_hex, “archive_bytes”: bytes }
Response 200: { “commit_id”: sha256_hex, “updated_ref”: string, “previous_head_commit_id”: sha256_hex, “receipt”: object(19.7) }

19.6.35 Users and ACL (minimal)
19.6.35.1 GET /users (admin)
Response 200: { “users”: [{ “user_id”: UUIDv7, “handle”: string, “created_at”: int }] }
19.6.35.2 POST /users (admin)
Request: { “handle”: string, “password”: string }
Response 201: { “user_id”: UUIDv7 }
19.6.35.3 GET /repos/{repo_id}/acl (maintainer+)
Response 200: { “entries”: [{ “user_id”: UUIDv7, “role”: “admin”|“maintainer”|“writer”|“reader” }] }
19.6.35.4 PUT /repos/{repo_id}/acl (admin or repo maintainer)
Request: { “entries”: [{ “user_id”: UUIDv7, “role”: “maintainer”|“writer”|“reader” }] }
Response 200: { “ok”: true }

19.6.36 POST /maintenance/verify
Response 200: { “ok”: bool, “checks”: [ { “name”: string, “status”: “pass”|“fail”, “details”?: any } ] }

19.6.37 POST /maintenance/pack?dry_run=0|1
Request: { “repo_id”: UUIDv7|null }
Response 200:
	•	if dry_run=1: { “dry_run”: true, “target_object_count”: int, “estimated_pack_bytes”: int, “predicted_pack_id”: sha256_hex|null }
	•	else: { “dry_run”: false, “pack_id”: sha256_hex, “object_count”: int }
Rules:
	•	pack MUST be explicit and MUST NOT run automatically.
	•	pack MUST NOT change any content IDs; it is a storage optimization only.

19.6.38 GET /system/diagnostics
Response 200:
{
“spec_version”: “0.0.1”,
“db”: { “page_count”: int, “wal_bytes”: int|null },
“cas”: { “loose_object_count”: int, “pack_count”: int },
“rebalance”: { “chapters”: [ { “chapter_id”: UUIDv7, “pressure”: string, “min_gap_hint”: string|null } ] },
“perf”: { “p95_publish_ms”: int|null, “p95_diff_scene_ms”: int|null }
}
Rules:
	•	Diagnostics MUST be offline-safe and MUST NOT expose secrets.

19.7 Operation receipt (required for mutations)
19.7.1 Any successful mutation that updates a ref MUST return a receipt object:
19.7.1.1 {
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
19.7.2 changed_paths MUST be sorted ascending (bytewise ASCII).
19.7.3 changed_scene_ids MUST be sorted ascending (bytewise UUID string).
19.7.4 receipt MUST be deterministic given identical inputs and repository state.
19.7.5 receipt MUST NOT affect CAS IDs.
	20.	UI Static Serving (Normative)
20.1 SPA embedding
20.1.1 UI MUST be a static SPA served by the binary.

20.2 Pinned implementation profile (normative)
20.2.1 The UI MUST be implemented as a static SPA built with:
20.2.1.1 TypeScript (ECMAScript 2022 target)
20.2.1.2 React 18.x
20.2.1.3 Vite 5.x
20.2.1.4 CodeMirror 6.x
20.2.2 The UI build MUST be fully self-contained:
20.2.2.1 MUST NOT require external CDNs, external fonts, or remote scripts/styles.
20.2.2.2 MUST NOT require service workers for correctness (MAY use them if fully offline and deterministic).
20.2.3 The binary MUST embed the exact UI build output bytes and serve them verbatim.
20.2.4 The server MUST serve the UI at path prefix /ui/ and MUST redirect / to /ui/ (HTTP 302) or serve the UI directly at /.
20.2.5 The server MUST set Content-Type correctly for: .html, .js, .css, .json, .svg, .png, .woff2 (if used).
20.2.6 The UI MUST be compatible with Chromium-based browsers and Firefox current stable at release time.

20.3 UI asset layout (normative)
20.3.1 The embedded UI asset root MUST contain:
20.3.1.1 /ui/index.html
20.3.1.2 /ui/assets/
20.3.1.3 /ui/ui_manifest.json
20.3.2 ui_manifest.json MUST be a JCS-canonical JSON blob with:
20.3.2.1 “spec_version”: “0.0.1”
20.3.2.2 “build_ts”: int (unix seconds, UTC)
20.3.2.3 “files”: [{ “path”: string, “sha256_hex”: string, “size”: int }]
Rules:
20.3.2.4 files MUST list every embedded UI file including index.html and all assets, excluding ui_manifest.json itself.
20.3.2.5 files MUST be sorted by path ascending (bytewise ASCII).
20.3.2.6 sha256_hex MUST be sha256 over the exact bytes served for that file.
20.3.3 Determinism rule: build_ts MUST be 0 in v0.0.1.
20.3.4 The server MUST expose the same ui_manifest.json bytes at GET /ui/ui_manifest.json.

20.4 UI security headers (required)
20.4.1 The server MUST send the following headers on all /ui/ responses:
20.4.1.1 X-Content-Type-Options: nosniff
20.4.1.2 Referrer-Policy: no-referrer
20.4.1.3 Cross-Origin-Resource-Policy: same-origin
20.4.1.4 Cross-Origin-Opener-Policy: same-origin
20.4.1.5 Cross-Origin-Embedder-Policy: require-corp
20.4.2 The server MUST send a Content-Security-Policy header for /ui/ that enforces at minimum:
20.4.2.1 default-src ‘none’
20.4.2.2 script-src ‘self’
20.4.2.3 style-src ‘self’
20.4.2.4 img-src ‘self’
20.4.2.5 font-src ‘self’
20.4.2.6 connect-src ‘self’
20.4.2.7 base-uri ‘none’
20.4.2.8 frame-ancestors ‘none’
20.4.2.9 form-action ‘none’
	21.	UI Global UX Invariants (Normative)
21.1 Draft vs staged vs committed separation
21.1.1 Draft changes MUST be local-only and MUST NOT be visible to other users until published.
21.1.2 Staged changes MUST be local-only and MUST represent an explicit user-selected publish set.
21.1.3 Committed state MUST be the server’s ref/commit state and MUST be treated as source of truth.

21.2 Deterministic navigation
21.2.1 Deep links MUST be stable across reloads.
21.2.2 Reload MUST restore route, repo selection, ref selection, and selected scene when present.

21.3 Stable identity visibility
21.3.1 The UI MUST provide a discoverable display of:
21.3.1.1 repo_id
21.3.1.2 current view id (ref name or commit id)
21.3.1.3 for edit views, active ref name (refs/heads/*) and its head commit id
21.3.2 The UI MUST show the last mutation receipt (19.7) in a copyable panel for the current session.

21.4 No remote dependencies
21.4.1 UI correctness MUST NOT require external networks or CDNs.

21.5 Keyboard-first viability
21.5.1 Core flows MUST be usable with keyboard-only navigation: read, edit, stage, publish, MR review, conflict resolution, merge.

21.6 Error transparency
21.6.1 All error surfaces MUST show server error code and message when provided.
21.6.2 The UI MUST show X-Request-Id when present (copyable).
21.6.3 The UI MUST show the operation name that failed (e.g., PUBLISH, MOVE_SCENE, SPLIT_SCENE, MERGE_MR, WORKTREE_PUSH).
21.6.4 For any mutation failure, the UI MUST show the attempted expected_head_commit_id and current head (if known).

21.7 Ref targeting guard
21.7.1 All ref-mutating UI actions MUST display target ref name and expected head commit id prior to confirmation.
21.7.2 The UI MUST NOT allow ref mutations without sending expected_head_commit_id when available.

21.8 Ritual elimination invariants
21.8.1 The UI MUST implement one-click flows that internally use preflight endpoints (18.6) before performing dangerous or failure-prone mutations.
21.8.2 The UI MUST surface explicit single-step options for 409 handling using rebase-staged (18.7) when applicable.
	22.	UI Routes (Normative)
22.1 Required routes under /ui/
22.1.1 /ui/
22.1.2 /ui/repos/:repo_id/read?ref=<ref_name_or_commit_id>[&scene=<scene_id>]
22.1.3 /ui/repos/:repo_id/edit?ref=<ref_name>[&scene=<scene_id>]
22.1.4 /ui/repos/:repo_id/staged?ref=<ref_name>
22.1.5 /ui/repos/:repo_id/mrs
22.1.6 /ui/repos/:repo_id/mrs/:mr_id
22.1.7 /ui/repos/:repo_id/settings
22.1.8 /ui/system
22.1.9 /ui/system/diagnostics

22.2 Optional route
22.2.1 The UI MAY implement /ui/login as a client-side route.

22.3 Route constraints
22.3.1 read route ref MAY be a ref name or commit id.
22.3.2 edit route ref MUST be a refs/heads/* name; the UI MUST reject editing refs/tags/* or raw commit ids in the editor.
22.3.3 If edit route is requested with an invalid ref, the UI MUST redirect to read route for the same repo with the same ref string (best-effort) and show an explanatory banner.
	23.	UI Screen Model (Normative)
23.1 Shell layout
23.1.1 The UI MUST provide a persistent shell with:
23.1.1.1 top bar: identity, global status, and actions
23.1.1.2 left panel: chapter list and scene list
23.1.1.3 main panel: route content
23.1.2 top bar MUST show: repo_id, view id, active ref (if any), head commit id, and last receipt summary.

23.2 Global status indicators (required)
23.2.1 Network indicator states: ONLINE, OFFLINE, DEGRADED.
23.2.2 Draft indicator states per selected scene: CLEAN, DIRTY, SAVING, SAVE_FAILED.
23.2.3 Stage indicator states per selected scene: UNSTAGED, STAGED.
23.2.4 Publish indicator states: IDLE, PUBLISHING, PUBLISH_FAILED, PUBLISHED, PUBLISH_CONFLICT.
23.2.5 Indicators MUST be non-blocking and MUST NOT steal focus during typing.

23.3 Global toasts (required)
23.3.1 Success toasts MUST be time-limited and non-modal.
23.3.2 Error toasts MUST remain until dismissed or replaced by a newer error for the same operation.
23.3.3 Toasts MUST include: operation name, error code, and request id when present.

23.4 Scene list dual sorting (required)
23.4.1 The UI MUST provide a toggle between:
23.4.1.1 read order (order.json)
23.4.1.2 work order (most recently staged or edited)
23.4.2 The UI MUST show per-scene badges:
23.4.2.1 DIRTY (draft differs from baseline)
23.4.2.2 STAGED
23.4.2.3 CHANGED_ON_HEAD (head advanced since baseline)
23.4.2.4 CONFLICT (publish or MR unresolved for the scene)
	24.	UI Data Fetching and Caching (Normative)
24.1 Request cancellation
24.1.1 All requests MUST be cancellable via AbortController on route changes.

24.2 GET deduplication and caching
24.2.1 The UI MUST dedupe in-flight GET requests by (method, url).
24.2.2 The UI SHOULD maintain a bounded in-memory cache for GET responses with explicit invalidation.
24.2.3 Cache invalidation MUST occur after successful mutations that affect those resources.

24.3 Session model
24.3.1 The UI MUST rely on HttpOnly cookie-based sessions.
24.3.2 The UI MUST NOT store auth tokens in localStorage.

24.4 Mutation rules and idempotency
24.4.1 All mutating requests MUST include Idempotency-Key.
24.4.2 The UI MUST generate Idempotency-Key per user intent and MUST reuse it across retries.
24.4.3 The UI MUST implement exponential backoff for retries on network errors and 429 only.
24.4.4 The UI MUST NOT auto-retry 409 conflicts; it MUST show a conflict flow.
24.4.5 The UI MUST send expected_head_commit_id (or expected_base_head_commit_id where defined) for all ref-mutating operations.

24.5 Offline behavior
24.5.1 If network requests fail, the UI MUST remain usable for typing and local stage persistence.
24.5.2 The UI MUST display OFFLINE state.
24.5.3 The UI MUST NOT implicitly queue server mutations.

24.6 Worker offload (typing safety)
24.6.1 Markdown preview rendering and large diff formatting SHOULD be performed in a WebWorker to protect typing latency budgets.
24.6.2 Worker outputs MUST be deterministic for identical inputs.
	25.	UI Draft and Stage Model and Persistence (Normative)
25.1 Draft storage
25.1.1 Drafts MUST be persisted in IndexedDB.
25.1.2 Draft key MUST be (repo_id, ref_name, scene_id).
25.1.3 Draft value MUST include:
25.1.3.1 body_md (string)
25.1.3.2 title (string|null)
25.1.3.3 tags (string[])
25.1.3.4 entities (string[])
25.1.3.5 updated_at (unix seconds)

25.2 Stage storage (required)
25.2.1 Staged changes MUST be persisted in IndexedDB.
25.2.2 Stage key MUST be (repo_id, ref_name, scene_id).
25.2.3 Stage value MUST include:
25.2.3.1 staged_fields (subset of editable fields)
25.2.3.2 base_head_commit_id (sha256_hex) captured at stage time
25.2.3.3 summary (deterministic summary for display)
25.2.3.4 summary_hash (sha256_hex of canonical staged_fields)
25.2.3.5 updated_at (unix seconds)

25.3 Baseline rules
25.3.1 Baseline MUST be loaded from committed state for the active ref when entering edit route.
25.3.2 Baseline MUST be updated after a successful publish that includes the scene.

25.4 Dirty detection
25.4.1 Dirty MUST be computed as draft != baseline over fields exposed in editor.
25.4.2 Dirty computation MUST be stable and MUST NOT depend on transient UI formatting.

25.5 Autosave rules
25.5.1 Autosave MUST be incremental and MUST NOT block typing latency budgets.
25.5.2 Autosave trigger MUST be:
25.5.2.1 after 1000ms of typing inactivity, AND
25.5.2.2 at least once every 3000ms during continuous typing.
25.5.3 Autosave MUST also occur on blur and before unload.
25.5.4 Autosave MUST write only the selected scene draft, not the entire repo.
25.5.5 On failure, Draft indicator MUST become SAVE_FAILED and the UI MUST retry with exponential backoff without blocking typing.

25.6 Multi-tab handling (required minimum)
25.6.1 If multiple tabs edit the same draft key, the UI MUST detect updated_at changes and prompt resolve.
25.6.2 The UI MUST provide actions: KEEP_LOCAL, TAKE_OTHER, MANUAL_MERGE.
25.6.3 MANUAL_MERGE MUST present a deterministic 2-way diff (line-based) over body_md and structured diffs over title/tags/entities.
	26.	UI Reading UX (Normative)
26.1 Ordering
26.1.1 Chapters displayed sorted by (chapter.order_key, chapter_id).
26.1.2 Scenes displayed in order.json order.

26.2 Reader content
26.2.1 Reader MUST display title, rendered Markdown body, tags, entities.
26.2.2 Raw HTML MUST be disabled.
26.2.3 Links MUST be sanitized; forbidden schemes MUST NOT be emitted.
26.2.4 Images MUST be rendered as links (no inline images) in v0.0.1.

26.3 Time-series first
26.3.1 Default view MUST read a selected ref (default: refs/heads/main) as ordered in 26.1.
26.3.2 Branch/DAG graph views MUST NOT be required for basic reading.

26.4 Branch surfacing rule (anti-noise)
26.4.1 UI MUST NOT show an unbounded list of forks at any reading point.
26.4.2 UI SHOULD show at most 3 recommended alternate routes, with the rest behind an explicit “more” interaction.

26.5 Scene lineage
26.5.1 Scene detail MUST provide a lineage view derived from provenance (split_from / merge_of parents).
	27.	UI Editor UX (Normative)
27.1 CodeMirror invariants
27.1.1 body_md editor MUST be CodeMirror 6.x.
27.1.2 IME composition MUST NOT be interrupted.
27.1.3 CodeMirror MUST NOT be re-mounted during normal typing.

27.2 Focus and navigation rules
27.2.1 UI MUST NOT steal focus from editor during active typing unless user explicitly navigates.
27.2.2 After navigation within edit route, UI SHOULD restore focus to editor automatically.

27.3 Preview
27.3.1 Preview SHOULD be available and MUST be safe (no script execution).
27.3.2 Preview render MUST be incremental for large body_md to protect latency budgets.
27.3.3 Preview MUST not depend on external network.
	28.	UI Publish UX (Normative)
28.1 Publish meaning and labels
28.1.1 Publish MUST be presented as PUBLISH (shared) and MUST be visually distinct from Draft autosave.
28.1.2 The UI MUST show the active ref name that will be updated by publish.

28.2 Publish scope (required)
28.2.1 v0.0.1 Publish MUST operate on:
28.2.1.1 one scene (current scene), OR
28.2.1.2 the staged set for the active ref (batch publish), producing exactly one commit for the batch.

28.3 Publish protocol (required, deterministic)
28.3.1 UI MUST publish via POST /repos/{repo_id}/ops/publish-scene for single-scene publish.
28.3.2 For batch publish, UI MUST call POST /repos/{repo_id}/ops/publish-staged.
28.3.3 UI MUST run preflight before publish using /ops/preflight-publish and display any errors in-place.
28.3.4 UI MUST include expected_head_commit_id for all publish operations.

28.4 Publish state machine (required)
28.4.1 States: IDLE -> PREFLIGHT -> IN_FLIGHT -> SUCCESS | FAILED | CONFLICT
28.4.2 PREFLIGHT MUST validate locally when feasible and MUST rely on server preflight for final validation.
28.4.3 IN_FLIGHT MUST keep the editor editable; only publish action may be disabled.
28.4.4 SUCCESS MUST update baseline for published scenes and MUST clear dirty indicator for those scenes.
28.4.5 FAILED MUST preserve drafts and staged changes and allow RETRY with same Idempotency-Key.
28.4.6 CONFLICT (409) MUST enter the conflict flow in 28.5 and MUST preserve drafts and staged.

28.5 Conflict flow for publish (409)
28.5.1 Steps:
28.5.1.1 fetch latest head for the active ref
28.5.1.2 compute diff head_at_stage -> latest_head for affected scenes
28.5.1.3 offer actions:
A) REBASE_STAGED (call /ops/rebase-staged)
B) TAKE_HEAD (discard staged and drafts for those scenes)
C) MANUAL_MERGE (editor + preview + structured merge)
28.5.2 UI MUST require explicit user choice; it MUST NOT auto-resolve conflicts.
	29.	UI Scene Operations UX (Normative)
29.1 Create scene
29.1.1 Create action MUST navigate to the created scene in edit route with editor focused.
29.1.2 Create MUST set provenance.op=“create” and parents empty.

29.2 Move and reorder
29.2.1 Move/reorder MUST be available via context menu and keyboard shortcuts.
29.2.2 Optimistic update MUST be visible within latency budgets.
29.2.3 On rejection, UI MUST revert and show actionable error.

29.3 Split
29.3.1 Split-at-cursor MUST be supported and MUST produce exactly one new scene by default.
29.3.2 UI MUST preview which text stays vs moves before confirmation.
29.3.3 ORDER_KEY_SPACE_EXHAUSTED (409) MUST offer REBALANCE_THEN_RETRY while preserving inputs.

29.4 Merge scenes
29.4.1 Merge scenes MUST be available via multi-select.
29.4.2 Default joiner MUST be “\n\n”.
29.4.3 On success, UI MUST navigate to merged scene.

29.5 Revert (required)
29.5.1 For any operation that returns previous_head_commit_id, UI MUST provide a one-click revert that creates a new commit and restores the previous tree.
	30.	UI MR and Diff UX (Normative)
30.1 MR list
30.1.1 MUST show mr_id, base_ref, head_ref, status, updated_at.
30.1.2 MUST allow filter by status.

30.2 MR detail
30.2.1 MUST show base_ref, head_ref, base_commit_id, head_commit_id, merge_base_commit_id when available.
30.2.2 Default view MUST be scene-first grouped change list.
30.2.3 Body diffs MUST NOT be expanded by default.

30.3 Per-scene diff view
30.3.1 Body diff MUST be stable deterministic line diff (LF normalized).
30.3.2 Structured diffs MUST be shown for: title, tags, entities, constraints, order, provenance.

30.4 MR merge action
30.4.1 Merge MUST require explicit confirmation.
30.4.2 Merge UI MUST show mode (ff/merge/squash) and the effect on base_ref.
30.4.3 On success, UI MUST show merged_commit_id and updated base_ref head and provide link to read.
30.4.4 UI MUST display the returned receipt.
	31.	UI Conflict Resolution UX (Normative)
31.1 Per-scene independent choices
31.1.1 For each conflicted scene, UI MUST provide:
31.1.1.1 content choice: base, head, manual
31.1.1.2 meta choice: base, head, manual
31.1.1.3 order choice: base, head, manual

31.2 Manual tools
31.2.1 Manual content MUST provide editor and preview.
31.2.2 Manual meta SHOULD provide union helpers for tags/entities.
31.2.3 Manual order MUST allow selecting placement and MUST derive order_key via rank/between.

31.3 Bulk flows (required)
31.3.1 UI MUST provide BULK_APPLY for order conflicts (apply head or base to all) with per-scene overrides.

31.4 High-volume order conflicts
31.4.1 If many order conflicts exist, UI SHOULD propose MERGE_THEN_REBALANCE.

31.5 Save and submit
31.5.1 Switching resolution choice MUST update preview within 100ms p95 MUST.
31.5.2 Merge submission MUST include all chosen resolutions.
	32.	UI Keyboard Shortcuts (Required Minimum)
32.1 The UI MUST provide shortcuts for:
32.1.1 next scene, previous scene
32.1.2 focus scene list, focus editor
32.1.3 stage/unstage current scene
32.1.4 publish current scene
32.1.5 publish staged set
32.1.6 open MR list
32.1.7 in MR detail: next changed scene, previous changed scene
32.1.8 in conflict resolution: next conflict, previous conflict
32.2 The UI MUST provide a help surface listing shortcuts.
32.3 Shortcut design MUST avoid conflicts with IME and common browser/system shortcuts.
	33.	UI Performance and Latency Profile (Pinned, Normative)
33.1 Measurement
33.1.1 Input timestamp t0 is when the browser receives the input event.
33.1.2 Output timestamp t1 is the first paint after UI state reflecting the intent is committed.
33.1.3 Latency = t1 - t0.
33.1.4 Percentiles p50/p95/p99 computed over rolling window N=200 per session.

33.2 Main-thread constraints
33.2.1 During active typing, main-thread tasks > 50ms SHOULD NOT occur.
33.2.2 If they occur, UI MUST postpone non-critical work to protect typing.

33.3 Editor latency
33.3.1 Keystroke-to-paint (body_md): p50 <= 16ms MUST, p95 <= 32ms MUST, p99 <= 50ms SHOULD.
33.3.2 Autosave MUST NOT cause p95 to exceed 32ms.

33.4 Navigation latency
33.4.1 Route change to first usable paint: p95 <= 250ms in-memory MUST, <= 800ms localhost fetch MUST, <= 1500ms LAN SHOULD.
33.4.2 Scene selection to content visible: p95 <= 100ms in-memory MUST, <= 400ms localhost fetch MUST.

33.5 Scene operations
33.5.1 Move/reorder optimistic update visible within 100ms p95 MUST.
33.5.2 Split/merge pending state within 100ms p95 MUST; success navigation within 1200ms p95 localhost SHOULD, 2000ms p95 LAN SHOULD.

33.6 Publish
33.6.1 Single-scene publish to committed state p95 <= 1500ms localhost SHOULD, <= 3000ms LAN SHOULD.
33.6.2 Batch publish (staged set) p95 <= 2500ms localhost SHOULD, <= 5000ms LAN SHOULD.

33.7 MR and diff
33.7.1 MR list first usable paint p95 <= 1000ms localhost MUST, <= 2000ms LAN MUST.
33.7.2 MR detail change list visible p95 <= 1200ms localhost MUST, <= 2500ms LAN MUST.
33.7.3 Open single scene diff p95 <= 300ms cached MUST, <= 1200ms compute SHOULD.

33.8 Conflict resolution
33.8.1 Switch choice updates preview within 100ms p95 MUST.
33.8.2 Save local resolution state within 50ms p95 MUST.

33.9 Instrumentation (offline-safe)
33.9.1 UI MAY record local performance metrics in memory.
33.9.2 If persisted, metrics MUST be local-only and erasable.
33.9.3 Instrumentation MUST NOT violate typing latency budgets.
	34.	UI Accessibility (Required)
34.1 Visible focus indicators MUST exist.
34.2 Controls MUST be reachable by Tab in logical order.
34.3 Icon-only controls MUST have accessible labels.
34.4 The UI MUST not rely on color alone to convey state.
	35.	UI Error UX (Normative)
35.1 Error surfaces
35.1.1 Fatal route errors MUST render an error screen.
35.1.2 Non-fatal errors MUST render banners or toasts without blocking typing.

35.2 Retry rules
35.2.1 The UI MUST NOT retry mutations without reusing Idempotency-Key.
35.2.2 Auto-retry is allowed only for network errors and 429, with backoff; never for 409.

35.3 Error guidance
35.3.1 413 MUST suggest splitting content.
35.3.2 429 SHOULD show retry-after if provided.
35.3.3 500 MUST display request id when present.
	36.	UI Asset and Cache Behavior (Normative)
36.1 /ui/assets/* SHOULD use content-hashed filenames.
36.2 If content-hashed, assets SHOULD be cacheable as immutable by the server.
36.3 Sensitive API responses MUST be treated as no-store beyond in-memory cache.
36.4 UI correctness MUST not depend on cache presence.
	37.	UI-side Security Expectations (Normative)
37.1 The UI MUST never execute content-derived scripts.
37.2 The UI MUST not render raw HTML from Markdown.
37.3 The UI MUST not store secrets or tokens in localStorage.
37.4 The UI MUST only connect to same-origin endpoints.
	38.	Auth, RBAC, Audit (Server-side)
38.1 Local auth
38.1.1 MUST support local authentication (no external IdP required).
38.1.2 Password hashing MUST use a modern KDF (Argon2id recommended).

38.2 Roles
38.2.1 Roles MUST include: admin, maintainer, writer, reader.
38.2.2 Permissions MUST be repo-scoped via ACL.

38.3 Audit log
38.3.1 Audit log MUST be append-only.
38.3.2 Audit MUST record at least:
38.3.2.1 ref updates
38.3.2.2 commit creation
38.3.2.3 MR open/merge/close
38.3.2.4 user/role changes
38.3.2.5 export/import
38.3.2.6 maintenance operations (rebalance, pack, verify, GC if any)
38.3.2.7 worktree import operations
38.3.3 Audit entries MUST include:
38.3.3.1 event_id (UUIDv7)
38.3.3.2 ts (unix seconds)
38.3.3.3 actor_id (UUIDv7)
38.3.3.4 action (string)
38.3.3.5 repo_id (UUIDv7|null)
38.3.3.6 details_json (JSON)
	39.	Security and Operational Limits
39.1 Limits
39.1.1 MUST implement configurable:
39.1.1.1 request size limits
39.1.1.2 upload size limits
39.1.1.3 rate limiting
39.1.1.4 max objects per import
39.1.1.5 max expanded bytes per import (bomb prevention)

39.2 Crash safety
39.2.1 DB migrations MUST be crash-safe.
39.2.2 The system MUST NOT corrupt meta.db on crash.

39.3 Logging
39.3.1 Logs SHOULD be structured JSON and include request_id.
39.3.2 Logs MUST NOT include plaintext passwords or password hashes.

39.4 Import hardening (required)
39.4.1 Import MUST reject:
39.4.1.1 absolute paths
39.4.1.2 paths containing “..” segments
39.4.1.3 duplicate paths
39.4.1.4 symlinks, hardlinks, device files
39.4.2 Import MUST fail fast on checksum mismatch (40.5.3).
	40.	Export/Import (Deterministic)
40.1 Archive format (normative)
40.1.1 Export MUST produce a single file: TAR archive compressed with Zstandard.
40.1.2 Filename extension SHOULD be .tar.zst.

40.2 Archive contents (normative)
40.2.1 Archive MUST contain:
40.2.1.1 meta.db (a consistent snapshot, see 40.4)
40.2.1.2 objects/ (all CAS data: loose objects and pack data, 48)
40.2.1.3 manifest.json

40.3 Deterministic manifest (normative)
40.3.1 manifest.json MUST be JCS-canonical JSON with:
40.3.1.1 “spec_version”: “0.0.1”
40.3.1.2 “created_at”: int (unix seconds)
40.3.1.3 “repo_ids”: [UUIDv7…]
40.3.1.4 “files”: [{ “path”: string, “sha256_hex”: string, “size”: int }]
40.3.2 files MUST list every file in the archive excluding manifest.json.
40.3.3 files MUST be sorted by path ascending (bytewise ASCII).
40.3.4 sha256_hex MUST be sha256 over the exact bytes of the archived file.
40.3.5 Determinism rule: created_at MUST be derived deterministically as:
40.3.5.1 max(ts) across all audit rows included in meta.db snapshot; if audit table is empty, created_at MUST be 0.

40.4 SQLite snapshot rule (required)
40.4.1 Export MUST archive a consistent meta.db snapshot even if SQLite WAL is enabled.
40.4.2 Implementations MUST satisfy 40.4.1 by one of:
40.4.2.1 using SQLite Online Backup API to write a snapshot meta.db file, then archiving that snapshot
40.4.2.2 an equivalent mechanism that guarantees transactional consistency
40.4.3 Export MUST NOT rely on copying meta.db alone when WAL is enabled unless it also guarantees inclusion or checkpointing of all committed data.

40.5 Import (normative)
40.5.1 Import MUST restore refs, MRs, ACLs, sessions, idempotency history, and audit history.
40.5.2 IDs MUST remain valid across export/import (blob/tree/commit IDs unchanged).
40.5.3 Import MUST fail if any required file checksum mismatches manifest.
40.5.4 Import MUST be atomic per data-dir: on failure, the server MUST not leave a partially imported state.

40.6 Deterministic TAR and Zstandard rules (required)
40.6.1 The TAR stream (before zstd) MUST be deterministic given identical archive file set and identical file bytes.
40.6.2 TAR entry ordering MUST be path ascending (bytewise ASCII), and MUST match manifest.files ordering.
40.6.3 TAR header normalization MUST be:
40.6.3.1 uid = 0, gid = 0
40.6.3.2 uname = “”, gname = “”
40.6.3.3 mtime = 0
40.6.3.4 devmajor = 0, devminor = 0
40.6.3.5 mode:
	•	regular files: 0644
	•	directories: 0755
40.6.4 TAR MUST NOT include PAX headers except those strictly required for long path support; if PAX headers are used, their content MUST be deterministic and MUST NOT include time fields (atime/ctime).
40.6.5 Zstandard compression MUST be deterministic:
40.6.5.1 compression level MUST be fixed by config default (default SHOULD be 3)
40.6.5.2 threads MUST be 1
40.6.5.3 checksum MUST be enabled or disabled deterministically (default SHOULD be enabled)
40.6.6 Implementations MUST NOT include any extra files beyond 40.2.1.

	41.	SQLite meta.db Minimum Schema (Required)
41.1 Required tables
41.1.1 repos(repo_id TEXT PRIMARY KEY, name TEXT NULL, created_at INTEGER NOT NULL)
41.1.2 refs(repo_id TEXT NOT NULL, ref_name TEXT NOT NULL, commit_id BLOB NOT NULL, updated_at INTEGER NOT NULL, PRIMARY KEY(repo_id, ref_name))
41.1.3 mrs(mr_id TEXT PRIMARY KEY, repo_id TEXT NOT NULL, base_ref TEXT NOT NULL, head_ref TEXT NOT NULL, base_commit_id BLOB NOT NULL, status TEXT NOT NULL, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL)
41.1.4 audit(event_id TEXT PRIMARY KEY, ts INTEGER NOT NULL, actor_id TEXT NOT NULL, action TEXT NOT NULL, repo_id TEXT NULL, details_json TEXT NOT NULL)
41.1.5 users(user_id TEXT PRIMARY KEY, handle TEXT UNIQUE NOT NULL, created_at INTEGER NOT NULL, password_hash BLOB NOT NULL, password_params_json TEXT NOT NULL)
41.1.6 repo_acl(repo_id TEXT NOT NULL, user_id TEXT NOT NULL, role TEXT NOT NULL, PRIMARY KEY(repo_id, user_id))
41.1.7 idempotency(actor_id TEXT NOT NULL, method TEXT NOT NULL, path TEXT NOT NULL, key TEXT NOT NULL, created_at INTEGER NOT NULL, response_status INTEGER NOT NULL, response_body BLOB NOT NULL, PRIMARY KEY(actor_id, method, path, key))
41.1.8 sessions(session_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, created_at INTEGER NOT NULL, expires_at INTEGER NOT NULL)

41.2 Atomicity
41.2.1 Ref updates, MR status changes, idempotency records, and session changes MUST be transactionally consistent.

41.3 Recommended indexes (required for performance)
41.3.1 audit(ts, event_id)
41.3.2 audit(repo_id, ts, event_id)
41.3.3 mrs(repo_id, status, updated_at)
	42.	Invariants
42.1 CAS objects are immutable.
42.2 commit/tree IDs are sha256 of canonical CBOR bytes.
42.3 Chapter/Scene/order/manifest/ui_manifest/audit.details_json/worktree guard JSON blobs are NFC-normalized, forbidden characters rejected, LF-normalized where applicable, then JCS-canonical.
42.4 body_md is LF-normalized and raw HTML disabled; links are sanitized; images are rendered as links (no ) in v0.0.1.
42.5 scene_id/chapter_id are content-independent UUIDv7.
42.6 Ordering is stable and explicit via order.json; tie-break by id for resilience.
42.7 Split produces new scene_ids; merge-of produces a new scene_id.
42.8 Rebalance and conflict resolutions always create new commits; no in-place edits of history.
42.9 Export/import restores identical object IDs and repository state; import fails on checksum mismatch.
42.10 UI is pinned to specified toolchain and is served from embedded bytes with required security headers and CSP.
42.11 All ref-mutating operations support expected_head_commit_id and return 409 REF_HEAD_MISMATCH on mismatch.
42.12 Publish uses server-side canonicalization and results in exactly one new commit on the active ref when successful.
42.13 Worktree import never updates refs without explicit expected_head_commit_id.
42.14 All ref-mutating mutations return deterministic operation receipts.
42.15 Preflight endpoints exist for publish and worktree import and do not mutate state.
42.16 A single-step explicit rebase-staged operation exists to reduce multi-step 409 rituals without auto-resolution.
	43.	Worktree and Markdown Interop (Required)
43.1 Purpose
43.1.1 Worktree interop provides a deterministic filesystem projection for external editing tools and Git workflows.

43.2 Worktree export layout (normative)
43.2.1 A worktree directory MUST contain:
43.2.1.1 .storyed/worktree.json (guard file, 43.3)
43.2.1.2 chapters/<chapter_id>/chapter.meta.json
43.2.1.3 chapters/<chapter_id>/order.json
43.2.1.4 chapters/<chapter_id>/scenes/<scene_id>.md
43.2.1.5 chapters/<chapter_id>/scenes/<scene_id>.meta.json
43.2.1.6 .gitattributes (recommended defaults, 43.7)
43.2.1.7 .editorconfig (recommended defaults, 43.7)

43.3 Worktree guard file (required)
43.3.1 .storyed/worktree.json MUST be a JCS-canonical JSON object containing exactly:
43.3.1.1 “spec_version”: “0.0.1”
43.3.1.2 “repo_id”: UUIDv7
43.3.1.3 “ref_name”: string (refs/heads/* only)
43.3.1.4 “base_commit_id”: sha256_hex (the commit exported)
43.3.1.5 “export_ts”: int (unix seconds); determinism rule: export_ts MUST be 0 in v0.0.1
43.3.2 worktree.json MUST be validated under 7 and 8.1 and treated as canonical text.
43.3.3 worktree import MUST fail if worktree.json is missing or invalid.

43.4 Worktree export semantics
43.4.1 Export MUST be deterministic for the same (repo_id, view id) state.
43.4.2 Export MUST write files atomically (write temp then rename) within the output directory.

43.5 Worktree import semantics (required)
43.5.1 Worktree import MUST require explicit expected_head_commit_id and MUST enforce 11.3.
43.5.2 Import MUST validate:
43.5.2.1 presence of all required files for referenced items (no half-pairs)
43.5.2.2 JSON canonicalization (8.1) and forbidden chars (7.3)
43.5.2.3 LF normalization of .md (14.2)
43.5.2.4 order.json consistency (15.7)
43.5.3 Import MUST map each <scene_id>.md + <scene_id>.meta.json to the corresponding Scene JSON update.
43.5.4 Import MUST create exactly one new commit on success, updating the target ref atomically.
43.5.5 Import MUST record an audit event with action “WORKTREE_IMPORT” and details including file counts and changed scene_ids.

43.6 Lint (required)
43.6.1 storyed lint MUST validate worktree directories according to 43.2-43.5.
43.6.2 lint output MUST include stable error codes and locations:
43.6.2.1 { code, path, field, message, offset|null }

43.7 Recommended generated repo hygiene files
43.7.1 .gitattributes SHOULD include:
43.7.1.1 *.md text eol=lf
43.7.1.2 *.json text eol=lf
43.7.2 .editorconfig SHOULD include:
43.7.2.1 charset = utf-8
43.7.2.2 end_of_line = lf
43.7.2.3 insert_final_newline = true
43.7.3 worktree add SHOULD generate .gitattributes and .editorconfig by default to prevent ritual fixes.
	44.	Git Interop Profile (Deterministic, Local)
44.1 Scope
44.1.1 v0.0.1 does not require Git protocol compatibility.
44.1.2 v0.0.1 requires deterministic worktree export/import (43) enabling Git workflows on the exported directory.

44.2 Git metadata policy
44.2.1 storyed MUST NOT embed Git-specific metadata inside canonical CAS objects.
44.2.2 storyed MAY export additional Git-facing files in the worktree that are not imported by storyed; if so, they MUST be placed under .storyed/ and MUST NOT affect import results.
	45.	Maintenance: Verify and Pack (Required)
45.1 Verify
45.1.1 The server MUST provide a verify operation that checks:
45.1.1.1 manifest and archive checksums during import
45.1.1.2 internal consistency: refs -> commits -> trees -> blobs exist
45.1.1.3 order.json consistency for all chapters
45.1.2 verify MUST NOT modify data; it is read-only.

45.2 Pack (required)
45.2.1 The server MUST support packing CAS loose objects into deterministic pack files (48) via POST /maintenance/pack.
45.2.2 pack MUST be explicit and MUST NOT run automatically.
45.2.3 pack MUST NOT change any content IDs; it is a storage optimization only.
45.2.4 pack SHOULD support dry_run to eliminate operational rituals.
	46.	Worktree and Publish Safety (Required)
46.1 No implicit ref updates
46.1.1 storyed MUST NOT update any ref due to filesystem changes without explicit worktree import by a user.
46.1.2 UI MUST never auto-publish; publish and worktree import require explicit user action.

46.2 Expected head enforcement
46.2.1 worktree import MUST include expected_head_commit_id and MUST fail on mismatch with 409 REF_HEAD_MISMATCH.
	47.	Deterministic Optimizations (Implementation Constraints)
47.1 Server-side publish ops
47.1.1 Implementations MUST use server-side publish operations for UI publish flows (18.5, 19.5.6).
47.2 Diff split
47.2.1 Implementations MUST implement diff-summary and diff-scene split (16.5, 19.5.4).
47.3 IndexedDB non-blocking behavior
47.3.1 Implementations MUST ensure draft/stage persistence does not violate UI typing budgets (33.3).
47.4 Tree update efficiency (implementation choice)
47.4.1 Implementations SHOULD avoid rebuilding full trees for single-scene updates and SHOULD apply sorted-entry merges to reduce CPU and allocations, without changing canonical outputs.
	48.	CAS Pack Format (Deterministic, v0.0.1)
48.1 Pack purpose
48.1.1 Packs reduce filesystem overhead by grouping many objects into fewer files while preserving object immutability and IDs.

48.2 Pack directory layout
48.2.1 Pack files MUST be stored under:
48.2.1.1 /objects/packs/
48.2.2 Pack MUST consist of:
48.2.2.1 pack data file: pack-<pack_id>.dat
48.2.2.2 pack index file: pack-<pack_id>.idx
48.2.3 pack_id MUST be sha256_hex of the canonical bytes of the index file (48.4).

48.3 Pack entries
48.3.1 The index MUST map blob_id -> (offset, length) into the data file.
48.3.2 The data file MUST contain concatenated raw blob bytes, each preceded by a fixed header:
48.3.2.1 magic “SDPK” (4 bytes)
48.3.2.2 version u32 = 1
48.3.2.3 blob_id raw bytes(32)
48.3.2.4 length u64 (little-endian)
48.3.2.5 payload bytes(length)
48.3.3 The server MUST validate blob_id matches sha256(payload) when reading, unless a verified cache is used.

48.4 Determinism
48.4.1 pack entries MUST be sorted by blob_id ascending (bytewise over raw 32 bytes).
48.4.2 The index file MUST be a canonical binary encoding:
48.4.2.1 magic “SDIX” (4 bytes)
48.4.2.2 version u32 = 1
48.4.2.3 count u64
48.4.2.4 repeated count times:
	•	blob_id raw bytes(32)
	•	offset u64
	•	length u64
48.4.3 pack_id = sha256(index_bytes).
48.4.4 pack creation MUST be deterministic given identical input object set and bytes.

	49.	End
End of v0.0.1
