Title: StoryD Editor Core Spec v0.0.1 (Single Binary, Offline, CAS+DAG, Scene-level, Markdown, Lexo-rank, UTF-8, Pinned UI)
	0.	Status
0.1 This document defines the frozen v0.0.1 specification for a self-contained editor server for stories/documents, including the pinned UI profile and security headers.
0.2 Target: single host, offline/air-gapped capable, non-realtime collaboration via branches and merge requests (MR).
0.3 Normative keywords: MUST, MUST NOT, SHOULD, MAY.
0.4 v0.0.1 compliant implementations MUST implement exactly the behavior and formats specified here.
	1.	Goals
1.1 Provide a time-series reading UX (chapters -> scenes) on top of a Git-like internal model.
1.2 Provide immutable content-addressed storage (CAS) for blobs/trees/commits and a DAG history.
1.3 Provide scene-level diff/merge and first-class scene operations (create/edit/move/reorder/split/merge).
1.4 Provide single-binary deployment with embedded UI assets and local storage only.
1.5 Provide deterministic export/import that restores identical object IDs and repository state.
1.6 Provide pinned UI implementation and security profile such that UI correctness does not depend on external resources.
	2.	Non-goals
2.1 Real-time concurrent editing (OT/CRDT).
2.2 Payments/marketplace/licensing enforcement.
2.3 Multi-region HA and global-scale operations.
2.4 Git wire protocol compatibility.
	3.	Terminology
3.1 Repo: repository (one work/document).
3.2 CAS: content-addressed storage (immutable objects).
3.3 Blob: raw bytes object.
3.4 Tree: a deterministic snapshot mapping absolute path -> blob_id.
3.5 Commit: DAG node referencing a tree and parent commits.
3.6 Ref: mutable pointer name -> commit_id (branch/tag).
3.7 Chapter: container of scenes for reading UX.
3.8 Scene: minimum editorial unit (diff/merge/cherry-pick granularity).
3.9 MR: merge request (base_ref + head_ref + review + merge).
3.10 Actor: authenticated user performing a request.
3.11 Draft: browser-local uncommitted edits (not visible to others).
	4.	Packaging and Runtime (L3)
4.1 Single executable
4.1.1 The system MUST be delivered as a single executable file.
4.1.2 The executable MUST serve both:

	•	HTTP API (JSON and selected binary downloads/uploads)
	•	UI (static SPA assets embedded in the binary)

4.2 Commands (normative interface)
4.2.1 The executable MUST support:
	•	serve --data-dir <path> --listen <addr:port> [--config <file>]
	•	export --data-dir <path> --out <file>
	•	import --data-dir <path> --in <file>
	•	maintenance rebalance --repo <repo_id> --chapter <chapter_id> --ref <ref_name>
4.2.2 Implementations MAY provide additional commands, but MUST NOT change the semantics of the above.

4.3 Offline requirement
4.3.1 The process MUST run offline (no mandatory external network calls).
4.3.2 UI assets MUST NOT require external CDNs for correctness.
	5.	Storage Layout
5.1 data-dir contents
5.1.1 data-dir MUST contain:

	•	<data-dir>/meta.db : SQLite database
	•	<data-dir>/objects/ : CAS objects (immutable)
5.1.2 data-dir MAY contain:
	•	<data-dir>/tmp/
	•	<data-dir>/logs/ (optional)

5.2 SQLite mode
5.2.1 SQLite SHOULD use WAL mode for performance and crash safety.
5.2.2 Export MUST produce a consistent snapshot regardless of WAL usage (see §23.4).

5.3 CAS object file path
5.3.1 CAS objects MUST be stored under:
	•	<data-dir>/objects/sha256/<aa>/<hex64>
where <aa> is the first 2 hex chars of <hex64> and <hex64> is the full lowercase sha256 hex string.
5.3.2 The system MUST NOT overwrite an existing CAS object file.
5.3.3 The system MUST write objects atomically (write temp then rename).
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
6.3.1 Branch ref names MUST use refs/heads/<name>.
6.3.2 Tag ref names MUST use refs/tags/<name>.
6.3.3 <name> MUST match regex: [A-Za-z0-9._-]{1,64}.
6.3.4 Ref names MUST be case-sensitive.
	7.	Text and Internationalization (UTF-8)
7.1 UTF-8
7.1.1 All user-visible text fields MUST be UTF-8 in API and storage.
7.1.2 The server MUST reject invalid UTF-8.

7.2 Unicode normalization
7.2.1 The server MUST normalize all stored text fields to Unicode NFC prior to canonicalization and hashing.

7.3 Forbidden characters
7.3.1 The server MUST reject the following characters in any stored text field:
	•	U+0000 (NUL)
	•	U+0001..U+001F and U+007F (control characters)
EXCEPT:
	•	LF (U+000A) is permitted inside body_md and commit.message.
	•	TAB (U+0009) is permitted inside body_md.
7.3.2 The server MUST reject bidi control characters:
	•	U+202A..U+202E and U+2066..U+2069

7.4 Length limits (configurable)
7.4.1 The server MUST enforce configurable length limits.
7.4.2 Default SHOULD be:
	•	chapter.title <= 256 code points
	•	scene.title <= 256 code points
	•	tags[i] <= 64 code points
	•	entities[i] <= 128 code points
	•	commit.message <= 2048 code points
	•	body_md <= 5 MiB (bytes after LF normalization)
	•	username/handle <= 64 code points

	8.	Canonicalization and Hash Stability (Required)
8.1 Canonicalization order (normative)
For any JSON-based stored object (Chapter, Scene, Manifest, Audit details_json, UI manifest):
8.1.1 Validate UTF-8.
8.1.2 Normalize to NFC for all text values.
8.1.3 Reject forbidden characters (§7.3) after NFC.
8.1.4 Normalize newlines to LF for fields where LF normalization applies:

	•	body_md
	•	commit.message (CRLF/CR -> LF)
8.1.5 Apply JSON Canonicalization Scheme (JCS, RFC 8785).
8.1.6 Hash the resulting canonical UTF-8 bytes with sha256 to obtain blob_id when stored in CAS as a JSON blob.

8.2 JSON blobs
8.2.1 The server MUST canonicalize JSON blobs prior to storage as defined in §8.1.
8.2.2 The API MUST return canonicalized JSON when returning stored chapter/scene blobs.

8.3 Tree/Commit encoding
8.3.1 Tree and Commit objects MUST be encoded using Canonical CBOR (RFC 8949 canonical rules).
8.3.2 Canonical CBOR output bytes are hashed for tree_id/commit_id.
8.3.3 The server MUST be the source of truth for canonicalization.
	9.	CAS Object Formats (Normative)
9.1 Blob
9.1.1 Blob is raw bytes stored exactly as written (post-canonicalization if the blob is a canonical JSON blob).
9.1.2 blob_id = sha256(blob_bytes).

9.2 Tree object (Canonical CBOR map)
9.2.1 Tree represents a flat mapping from absolute path -> blob_id.
9.2.2 Tree CBOR map MUST contain exactly:
	•	“type” : “tree”
	•	“entries” : array
9.2.3 Each entry MUST be a CBOR map containing exactly:
	•	“path” : text (absolute path, ASCII; see §10)
	•	“id” : bytes(32) (raw sha256 of the referenced blob)
9.2.4 “entries” MUST be sorted by “path” ascending using bytewise comparison of UTF-8 bytes.
9.2.5 Duplicate “path” values MUST be rejected (400).
9.2.6 Tree MUST NOT contain any additional fields in v0.0.1.

9.3 Commit object (Canonical CBOR map)
9.3.1 Commit CBOR map MUST contain exactly:
	•	“type” : “commit”
	•	“tree” : bytes(32)
	•	“parents” : array of bytes(32) (0..n)
	•	“author” : map { “user_id”: text(UUIDv7), “handle”: text|null }
	•	“message” : text
	•	“created_at” : int (unix seconds, UTC)
9.3.2 Commit MUST NOT include any additional fields (metadata fields are forbidden in v0.0.1).
9.3.3 “parents” ordering MUST be deterministic: sort by raw bytes(32) ascending.
9.3.4 Implementations MAY store additional operational metadata in meta.db, but MUST NOT affect commit_id/tree_id derivation.

	10.	Tree Paths (Normative)
10.1 Layout (normative for v0.0.1)
10.1.1 The repository tree layout MUST be:

	•	/chapters/<chapter_id>.json -> blob (Chapter JSON)
	•	/chapters/<chapter_id>/scenes/<scene_id>.json -> blob (Scene JSON)

10.2 ID segments
10.2.1 <chapter_id> and <scene_id> MUST be canonical UUIDv7 strings (lowercase).

10.3 Path restrictions
10.3.1 Paths MUST start with ‘/’.
10.3.2 Segments MUST NOT be empty.
10.3.3 Segments MUST NOT be “.” or “..”.
10.3.4 Paths MUST NOT contain backslash ’'.
10.3.5 Paths MUST be ASCII-only in v0.0.1 (by construction of the layout).
10.3.6 Implementations MUST reject any attempt to store objects under non-conforming paths.
	11.	Repository Model
11.1 Repository creation
11.1.1 Creating a repo MUST:

	•	allocate repo_id
	•	create an initial empty tree whose entries array is empty
	•	create an initial commit referencing that empty tree
	•	create refs/heads/main pointing to that commit

11.2 Ref updates
11.2.1 Ref updates MUST be atomic (single SQLite transaction).
11.2.2 Ref update MUST implement compare-and-swap semantics when a caller supplies expected_old_commit_id.
11.2.3 If expected_old_commit_id is non-null and does not match the current ref value, the server MUST return 409 conflict.
	12.	Chapter JSON (Normative)
12.1 Chapter blob MUST be JCS-canonical JSON with fields:

	•	chapter_id: string(UUIDv7)
	•	title: string
	•	summary: string|null
	•	constraints:
	•	rating: “general”|“r15”|“r18”
	•	flags: string[]
	•	tags: string[]
	•	order_key: string (lexo-rank, §15)
12.2 chapter_id MUST equal the <chapter_id> in its path.
12.3 tags and constraints.flags MAY be empty arrays but MUST be present (no null).

	13.	Scene JSON (Normative)
13.1 Scene blob MUST be JCS-canonical JSON with fields:

	•	scene_id: string(UUIDv7)
	•	chapter_id: string(UUIDv7)
	•	order_key: string (lexo-rank, §15)
	•	title: string|null
	•	body_md: string (Markdown, UTF-8, LF-normalized)
	•	tags: string[]
	•	entities: string[]
	•	constraints:
	•	rating: “general”|“r15”|“r18”
	•	flags: string[]
	•	provenance:
	•	op: “create”|“edit”|“split_from”|“merge_of”|“move”
	•	parents: array of { scene_id: UUIDv7, commit_id: sha256_hex }
13.2 scene_id MUST equal the <scene_id> in its path.
13.3 chapter_id MUST equal the <chapter_id> in its path.
13.4 provenance.parents MAY be empty only when op=“create”.
13.5 provenance.parents MUST NOT contain duplicates of (scene_id, commit_id).

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
	•	javascript:
	•	data:
	•	vbscript:
14.4.4 The renderer SHOULD allow only:
	•	http
	•	https
	•	mailto
	•	relative URLs (including fragment-only)

	15.	Lexo-rank Ordering (Fixed-length base62, v0.0.1)
15.1 Alphabet and validation
15.1.1 order_key MUST be exactly 16 characters.
15.1.2 Characters MUST be base62: 0-9A-Za-z.
15.1.3 Comparison MUST be lexicographic by ASCII byte order over the 16 bytes.

15.2 Digit mapping (normative)
15.2.1 The base62 alphabet is:
	•	DIGITS: “0123456789”
	•	UPPER:  “ABCDEFGHIJKLMNOPQRSTUVWXYZ”
	•	LOWER:  “abcdefghijklmnopqrstuvwxyz”
15.2.2 digit(‘0’)=0 .. digit(‘9’)=9, digit(‘A’)=10 .. digit(‘Z’)=35, digit(‘a’)=36 .. digit(‘z’)=61.

15.3 Sentinels
15.3.1 left_sentinel  = “0000000000000000”
15.3.2 right_sentinel = “zzzzzzzzzzzzzzzz”
15.3.3 default_mid_digit = “U” (digit 30)

15.4 Between(left, right) (deterministic, fixed width)
15.4.1 Inputs:
	•	left_key MAY be null (means -infinity -> left_sentinel)
	•	right_key MAY be null (means +infinity -> right_sentinel)
	•	If both non-null, left_key MUST be < right_key.
15.4.2 Output MUST be a 16-char key strictly between.
15.4.3 Algorithm:
Let L = (left_key or left_sentinel), R = (right_key or right_sentinel).
For i in 0..15:
	•	li = digit(L[i])
	•	ri = digit(R[i])
	•	If li == ri: output L[i] at i and continue
	•	If ri - li >= 2:
set output[i] = value(floor((li+ri)/2))
set output[i+1..15] = default_mid_digit
return output
	•	If ri - li == 1:
output[i] = L[i]
continue
If the loop completes, there is no space at width 16:
	•	return error ORDER_KEY_SPACE_EXHAUSTED (HTTP 409).
15.4.4 Between(null, null) MUST return “UUUUUUUUUUUUUUUU”.

15.5 Rebalance(chapter_id) (deterministic)
15.5.1 Rebalance MUST assign fresh order_key values to all scenes within a chapter and MUST create a new commit.
15.5.2 Rebalance input order is the current stable order (order_key, scene_id) at the base commit.
15.5.3 Rebalance output algorithm:
	•	Define GAP = 62^4
	•	For scene index i from 1..n:
key_num = i * GAP
order_key = base62_encode_fixed(key_num, width=16, pad=‘0’)
15.5.4 base62_encode_fixed encodes most significant digit first; left-pad with ‘0’ to width 16.

15.6 Stable sorting
15.6.1 Chapter order: (chapter.order_key, chapter_id)
15.6.2 Scene order: (scene.order_key, scene_id)
	16.	Diff Semantics (Scene-first)
16.1 Diff(base, head) MUST report:

	•	chapters: added/deleted/modified/reordered
	•	scenes: added/deleted/modified/moved/reordered

16.2 Scene classification (normative)
	•	modified: same scene_id exists in both trees, blob differs
	•	added: only in head tree
	•	deleted: only in base tree
	•	moved: same scene_id exists in both trees, but chapter_id/path differs
	•	reordered: same scene_id exists in both trees, order_key differs

16.3 Modified scene diff MUST include:
	•	body_md line diff (LF normalized)
	•	structured diff for: title, tags, entities, constraints, order_key, chapter_id, provenance

16.4 Line diff algorithm (required minimum)
16.4.1 The server MUST provide a stable, deterministic line-based diff.
16.4.2 Implementations MAY choose the algorithm; output MUST be deterministic given the same inputs.
	17.	Merge Requests (MR)
17.1 MR record MUST include:

	•	mr_id, repo_id
	•	base_ref, head_ref
	•	base_commit_id (snapshot at MR creation)
	•	status: open|merged|closed
	•	checks: array of { name, status, details? }

17.2 Merge modes
17.2.1 fast-forward: ref update only.
17.2.2 merge: 3-way merge commit (new commit with 2+ parents).
17.2.3 squash: single new commit capturing head changes.

17.3 merge_base (deterministic)
17.3.1 merge_base MUST be computed as a common ancestor in the commit DAG.
17.3.2 If multiple common ancestors exist, the implementation MUST choose deterministically:
Define dist(X, A) as the length of the shortest parent-edge path from commit X to ancestor A.
For each common ancestor A, compute tuple:
T(A) = (
max(dist(base_head, A), dist(head_head, A)),
dist(base_head, A) + dist(head_head, A),
commit_id(A)
)
Choose A with lexicographically smallest T(A).

17.4 Conflict detection (scene-level)
	•	content conflict: body_md changed on both sides since merge_base
	•	meta conflict: title/tags/entities/constraints/provenance changed on both sides since merge_base
	•	order conflict: order_key and/or chapter_id changed on both sides since merge_base

17.5 Default resolution policy
17.5.1 order conflict defaults to head, but MUST be overrideable per scene (base/head/manual).
17.5.2 content/meta conflicts MUST require explicit resolution by user (MR UI/API).

17.6 Merge result
17.6.1 Conflict resolutions MUST result in a new commit (immutable history).
17.6.2 The merge operation MUST update base_ref to the resulting commit_id atomically.
	18.	Scene Operations (First-class)
18.1 Edit
18.1.1 scene_id MUST be preserved.
18.1.2 provenance.op=“edit”.
18.1.3 provenance.parents MUST include prior (scene_id, commit_id).

18.2 Move/Reorder
18.2.1 Move across chapters MUST update chapter_id/path.
18.2.2 Reorder MUST update order_key.
18.2.3 provenance.op=“move”, parents include prior.

18.3 Split (A -> A + B..)
18.3.1 Source scene A remains (typically first segment).
18.3.2 New scenes B.. MUST use new scene_ids.
18.3.3 All involved scenes MUST set provenance.op=“split_from”.
18.3.4 provenance.parents MUST reference A@previous_commit.
18.3.5 order_keys MUST reflect intended reading order.
18.3.6 Split offsets are expressed as byte offsets into LF-normalized UTF-8 bytes of body_md:
	•	strictly increasing
	•	in range 1..len(body_bytes)-1
	•	on a valid UTF-8 codepoint boundary
If violated, server MUST return 400 invalid input.
18.3.7 If ORDER_KEY_SPACE_EXHAUSTED occurs during split:
	•	the server MUST perform Rebalance on the target chapter in a new commit, then retry the split within the operation.
	•	the operation MUST succeed or fail atomically.

18.4 Merge Scenes (A,B.. -> N)
18.4.1 Create new scene N with new scene_id.
18.4.2 Remove source scenes from resulting tree (no tombstones).
18.4.3 N sets provenance.op=“merge_of” with parents referencing each source scene_id and commit_id.
	19.	API (HTTP) (Normative)
19.1 OpenAPI
19.1.1 MUST expose OpenAPI at /openapi.json.

19.2 Common headers
19.2.1 All responses SHOULD include X-Request-Id.
19.2.2 All mutating requests SHOULD support Idempotency-Key.
19.2.3 Server MUST store idempotency results for at least 24 hours (configurable), keyed by:
(actor_id, method, path, idempotency_key).
19.2.4 On idempotent replay, the server MUST return the stored response status code and body, and MUST NOT perform the mutation again.

19.3 Auth and sessions (required)
19.3.1 Mutating endpoints MUST require authentication.
19.3.2 Read endpoints MAY be public depending on repo ACL, but MUST enforce ACL.
19.3.3 The server MUST support cookie-based sessions:
	•	session cookie MUST be HttpOnly
	•	session cookie MUST be SameSite=Lax or SameSite=Strict
	•	session cookie SHOULD be Secure when served over HTTPS
19.3.4 The server MUST NOT require any external IdP for correctness.

19.4 Error model (normative)
19.4.1 Error JSON: { “code”: string, “message”: string, “details”?: any }
19.4.2 HTTP mapping:
	•	400 invalid input
	•	401 unauthenticated
	•	403 unauthorized
	•	404 not found
	•	409 conflict (includes ORDER_KEY_SPACE_EXHAUSTED)
	•	413 payload too large
	•	429 rate limited
	•	500 internal error

19.5 Endpoints (required)
Core:
	•	GET  /health
	•	POST /auth/login
	•	POST /auth/logout
	•	GET  /auth/me
Repos and refs:
	•	POST /repos
	•	GET  /repos/{repo_id}
	•	POST /repos/{repo_id}/refs
	•	GET  /repos/{repo_id}/refs
CAS:
	•	POST /blobs
	•	GET  /blobs/{blob_id}
	•	POST /trees
	•	GET  /trees/{tree_id}
Commits and diffs:
	•	POST /repos/{repo_id}/commits
	•	GET  /repos/{repo_id}/commits/{commit_id}
	•	GET  /repos/{repo_id}/diff?base=<ref|commit>&head=<ref|commit>
MR:
	•	POST /repos/{repo_id}/mrs
	•	GET  /repos/{repo_id}/mrs
	•	GET  /repos/{repo_id}/mrs/{mr_id}
	•	POST /repos/{repo_id}/mrs/{mr_id}/merge
	•	POST /repos/{repo_id}/mrs/{mr_id}/cherry-pick
Ops:
	•	POST /repos/{repo_id}/rank/between
	•	POST /repos/{repo_id}/rank/rebalance
	•	POST /repos/{repo_id}/ops/split-scene
	•	POST /repos/{repo_id}/ops/merge-scenes
	•	POST /repos/{repo_id}/ops/move-scene
Admin and ACL:
	•	GET  /users
	•	POST /users
	•	GET  /users/{user_id}
	•	GET  /repos/{repo_id}/acl
	•	PUT  /repos/{repo_id}/acl
Audit and export/import:
	•	GET  /repos/{repo_id}/audit?after_ts=<int|null>&limit=
	•	GET  /export?repo_id=<UUIDv7|null>
	•	POST /import
UI serving:
	•	GET  /ui/
	•	GET  /ui/index.html
	•	GET  /ui/ui_manifest.json
	•	GET  /ui/assets/*

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

19.6.6 POST /repos/{repo_id}/refs
Request:
{
“ref_name”: string,
“target_commit_id”: sha256_hex,
“expected_old_commit_id”: sha256_hex|null
}
Rules:
	•	ref_name MUST match: ^refs/(heads|tags)/[A-Za-z0-9._-]{1,64}$
Response 200: { “ref_name”: string, “commit_id”: sha256_hex }

19.6.7 POST /blobs
Request: raw bytes, Content-Type required
Response 201: { “blob_id”: sha256_hex, “size”: int, “content_type”: string }
Rules:
	•	The server MUST store a normalized content_type value in meta.db and return the normalized value:
	•	trim leading/trailing ASCII whitespace
	•	reject NUL or control characters (§7.3)
	•	the server MAY additionally lowercase type/subtype deterministically

19.6.8 POST /trees
Request:
{ “entries”: [ { “path”: string, “blob_id”: sha256_hex } ] }
Rules:
	•	path MUST comply with §10.3 and MUST be unique within request.
	•	The server MUST reject any blob_id not present in CAS (404, code “CAS_BLOB_NOT_FOUND”).
	•	The server MUST sort entries by path bytewise before canonical CBOR encoding.
Response 201: { “tree_id”: sha256_hex }

19.6.9 GET /trees/{tree_id}
Response 200:
{ “tree_id”: sha256_hex, “entries”: [ { “path”: string, “blob_id”: sha256_hex } ] }
Rules:
	•	returned list MUST be sorted by path bytewise.

19.6.10 POST /repos/{repo_id}/commits
Request:
{
“tree_id”: sha256_hex,
“parents”: [sha256_hex…],
“author”: { “user_id”: UUIDv7, “handle”: string|null },
“message”: string,
“created_at”: int
}
Rules:
	•	tree_id MUST exist (404, code “CAS_TREE_NOT_FOUND”).
	•	each parent commit_id MUST exist (404, code “CAS_COMMIT_NOT_FOUND”).
	•	server MUST sort parents by raw bytes ascending before canonical encoding.
Response 201: { “commit_id”: sha256_hex }

19.6.11 GET /repos/{repo_id}/diff
Response 200:
{
“base”: { “kind”: “ref”|“commit”, “id”: string },
“head”: { “kind”: “ref”|“commit”, “id”: string },
“chapters”: { “added”: [UUIDv7…], “deleted”: [UUIDv7…], “modified”: [UUIDv7…], “reordered”: [UUIDv7…] },
“scenes”: { “added”: [UUIDv7…], “deleted”: [UUIDv7…], “modified”: [UUIDv7…], “moved”: [UUIDv7…], “reordered”: [UUIDv7…] }
}

19.6.12 POST /repos/{repo_id}/mrs
Request: { “base_ref”: string, “head_ref”: string }
Response 201:
{ “mr_id”: UUIDv7, “repo_id”: UUIDv7, “base_ref”: string, “head_ref”: string, “base_commit_id”: sha256_hex, “status”: “open” }

19.6.13 POST /repos/{repo_id}/mrs/{mr_id}/merge
Request:
{
“mode”: “ff”|“merge”|“squash”,
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
Response 200: { “merged_commit_id”: sha256_hex }

19.6.14 POST /repos/{repo_id}/mrs/{mr_id}/cherry-pick
Request: { “scene_ids”: [UUIDv7…], “target_ref”: string }
Response 200: { “commit_id”: sha256_hex, “updated_ref”: string }

19.6.15 POST /repos/{repo_id}/rank/between
Request: { “left_key”: string|null, “right_key”: string|null }
Response 200: { “order_key”: string }

19.6.16 POST /repos/{repo_id}/rank/rebalance
Request: { “chapter_id”: UUIDv7, “ref”: string }
Response 200: { “commit_id”: sha256_hex, “updated_ref”: string }

19.6.17 POST /repos/{repo_id}/ops/move-scene
Request:
{
“ref”: string,
“scene_id”: UUIDv7,
“target_chapter_id”: UUIDv7,
“left_scene_id”: UUIDv7|null,
“right_scene_id”: UUIDv7|null
}
Response 200: { “commit_id”: sha256_hex, “updated_ref”: string, “new_order_key”: string }

19.6.18 POST /repos/{repo_id}/ops/split-scene
Request:
{
“ref”: string,
“scene_id”: UUIDv7,
“splits”: [ { “byte_offset”: int } ],
“titles”: [string|null] | null
}
Response 200: { “commit_id”: sha256_hex, “updated_ref”: string, “new_scene_ids”: [UUIDv7…] }

19.6.19 POST /repos/{repo_id}/ops/merge-scenes
Request:
{
“ref”: string,
“scene_ids”: [UUIDv7…],
“target_chapter_id”: UUIDv7,
“left_scene_id”: UUIDv7|null,
“right_scene_id”: UUIDv7|null,
“joiner”: string
}
Rules:
	•	joiner MUST be processed under §7 and §8.1 (UTF-8, NFC, forbidden chars).
Response 200: { “commit_id”: sha256_hex, “updated_ref”: string, “merged_scene_id”: UUIDv7 }

19.6.20 GET /repos/{repo_id}/audit
Response 200:
{
“events”: [
{ “event_id”: UUIDv7, “ts”: int, “actor_id”: UUIDv7, “action”: string, “repo_id”: UUIDv7|null, “details_json”: object }
],
“next_after_ts”: int|null
}
Rules:
	•	events MUST be ordered by (ts, event_id) ascending.
	•	details_json MUST be canonicalized per §8.1 when stored; API returns canonical form.

19.6.21 GET /export
Response 200:
	•	Content-Type: application/zstd
	•	Body: tar.zst bytes as defined in §23
Rules:
	•	Access MUST require admin.
	•	repo_id query parameter MAY be null; if null, export all repos.
	•	export bytes MUST correspond exactly to deterministic archive rules (§23).

19.6.22 POST /import
Request:
	•	Content-Type: application/zstd
	•	Body: tar.zst bytes
Response 200: { “ok”: true, “imported_repo_ids”: [UUIDv7…] }
Rules:
	•	Access MUST require admin.
	•	Import MUST be atomic per data-dir (§23.5.4).
	•	Import MUST fail on checksum mismatch (§23.5.3).

19.6.23 Users and ACL (minimal)
19.6.23.1 GET /users (admin)
Response 200: { “users”: [{ “user_id”: UUIDv7, “handle”: string, “created_at”: int }] }

19.6.23.2 POST /users (admin)
Request: { “handle”: string, “password”: string }
Response 201: { “user_id”: UUIDv7 }

19.6.23.3 GET /repos/{repo_id}/acl (maintainer+)
Response 200: { “entries”: [{ “user_id”: UUIDv7, “role”: “admin”|“maintainer”|“writer”|“reader” }] }

19.6.23.4 PUT /repos/{repo_id}/acl (admin or repo maintainer)
Request: { “entries”: [{ “user_id”: UUIDv7, “role”: “maintainer”|“writer”|“reader” }] }
Response 200: { “ok”: true }

19.7 UI static serving is normative
19.7.1 The server MUST serve the UI from embedded bytes and MUST NOT require external assets for correctness.
19.7.2 The server MUST expose /ui/ui_manifest.json as described in §20.3 and MUST serve exactly the bytes listed there.
	20.	UI Requirements (Pinned, Normative)
20.1 SPA embedding
20.1.1 UI MUST be a static SPA served by the binary.

20.2 Implementation profile (pinned)
20.2.1 The UI MUST be implemented as a static SPA built with:
	•	TypeScript (ECMAScript 2022 target)
	•	React 18.x
	•	Vite 5.x
	•	CodeMirror 6.x
20.2.2 The UI build MUST be fully self-contained:
	•	MUST NOT require external CDNs, external fonts, or remote scripts/styles.
	•	MUST NOT require service workers for correctness (MAY use them if fully offline and deterministic).
20.2.3 The binary MUST embed the exact UI build output bytes and serve them verbatim.
20.2.4 The server MUST serve the UI at path prefix /ui/ and MUST redirect / to /ui/ (HTTP 302) or serve the UI directly at /.
20.2.5 The server MUST set Content-Type correctly for: .html, .js, .css, .json, .svg, .png, .woff2 (if used).
20.2.6 The UI MUST be compatible with Chromium-based browsers and Firefox current stable at release time.

20.3 UI asset layout (normative)
20.3.1 The embedded UI asset root MUST contain:
	•	/ui/index.html
	•	/ui/assets/
	•	/ui/ui_manifest.json
20.3.2 ui_manifest.json MUST be a JCS-canonical JSON blob with:
	•	spec_version: “0.0.1”
	•	build_ts: int (unix seconds, UTC)
	•	files: [{ path: string, sha256_hex: string, size: int }]
Rules:
	•	files MUST list every embedded UI file including index.html and all assets, excluding ui_manifest.json itself.
	•	files MUST be sorted by path ascending (bytewise ASCII).
	•	sha256_hex MUST be sha256 over the exact bytes served for that file.
20.3.3 The server MUST expose the same ui_manifest.json bytes at GET /ui/ui_manifest.json.

20.4 UI security headers (required)
20.4.1 The server MUST send the following headers on all /ui/ responses:
	•	X-Content-Type-Options: nosniff
	•	Referrer-Policy: no-referrer
	•	Cross-Origin-Resource-Policy: same-origin
	•	Cross-Origin-Opener-Policy: same-origin
	•	Cross-Origin-Embedder-Policy: require-corp
20.4.2 The server MUST send a Content-Security-Policy header for /ui/ that enforces at minimum:
	•	default-src ‘none’
	•	script-src ‘self’
	•	style-src ‘self’
	•	img-src ‘self’
	•	font-src ‘self’
	•	connect-src ‘self’
	•	base-uri ‘none’
	•	frame-ancestors ‘none’
	•	form-action ‘none’

20.5 Routing and core screens (normative)
20.5.1 The UI MUST implement client-side routing under /ui/ with routes:
	•	/ui/
	•	/ui/repos/:repo_id/read?ref=<ref_name_or_commit_id>
	•	/ui/repos/:repo_id/edit?ref=<ref_name>
	•	/ui/repos/:repo_id/mrs
	•	/ui/repos/:repo_id/mrs/:mr_id
	•	/ui/repos/:repo_id/settings
	•	/ui/system

20.6 Reading UX (time-series first)
20.6.1 Default view MUST read a selected ref (default: refs/heads/main) as:
	•	chapters sorted by (chapter.order_key, chapter_id)
	•	scenes sorted by (scene.order_key, scene_id)
20.6.2 Branch/DAG graph views MUST NOT be required for basic reading.
20.6.3 Branch surfacing rule (anti-noise):
	•	UI MUST NOT show an unbounded list of forks at any reading point.
	•	UI SHOULD show at most 3 recommended alternate routes, with the rest behind an explicit “more” interaction.
20.6.4 Scene detail MUST provide a lineage view derived from provenance (split_from / merge_of parents).

20.7 Editor UX (normative)
20.7.1 The editor MUST use CodeMirror 6.x and MUST support IME composition without interruption.
20.7.2 The editor MUST provide Draft vs Commit separation:
	•	Draft state is local to the browser and MUST NOT be visible to other users.
	•	Commit is an explicit action that produces a new commit on a ref.

20.7.3 Draft persistence
20.7.3.1 Drafts MUST be persisted locally using IndexedDB.
20.7.3.2 Draft key MUST be (repo_id, ref_name, scene_id).
20.7.3.3 Draft value MUST include:
	•	body_md (string)
	•	title (string|null)
	•	tags (string[])
	•	entities (string[])
	•	updated_at (unix seconds)
20.7.3.4 UI MUST provide a visible indicator when a draft differs from last committed version.

20.7.4 Commit UX
20.7.4.1 UI MUST allow an auto-generated default commit message (editable).
20.7.4.2 UI MUST NOT force a long message on every commit.
20.7.4.3 On commit, the UI MUST:
	•	validate inputs locally where possible (length limits, forbidden chars)
	•	send mutation requests with Idempotency-Key
	•	display resulting commit_id and updated ref head

20.8 Diff and MR UX (normative)
20.8.1 MR list page MUST show: mr_id, base_ref, head_ref, status, updated_at.
20.8.2 MR detail page MUST show:
	•	base_commit_id, head_commit_id, merge_base_commit_id
	•	per-scene change list grouped by: added/modified/moved/reordered/deleted
20.8.3 Conflict resolution UX MUST provide per-scene choices:
	•	content: base/head/manual
	•	meta: base/head/manual (manual MAY provide helpers such as union of tags/entities)
	•	order: base/head/manual (manual allows chapter_id + order_key)
20.8.4 If many order conflicts exist, UI SHOULD propose “merge then rebalance” as a one-click follow-up action.
20.8.5 UI MUST display a stable line-based diff for body_md (LF normalized).

20.9 Network and caching rules (normative)
20.9.1 UI MUST communicate with the server using fetch() to same-origin endpoints only.
20.9.2 UI MUST NOT store authentication tokens in localStorage.
20.9.3 UI MUST support cookie-based sessions (HttpOnly, SameSite=Lax or Strict).
20.9.4 Server SHOULD set Cache-Control for /ui/assets/* to immutable with long max-age if filenames are content-hashed.
20.9.5 Server MUST set Cache-Control: no-store for API responses that contain sensitive data (users, ACL, audit, auth/me).

20.10 Accessibility and keyboard (required)
20.10.1 UI MUST be usable with keyboard-only navigation for core flows (read, edit, MR resolve, merge).
20.10.2 UI MUST include visible focus indicators.
	21.	Auth, RBAC, Audit (Server-side)
21.1 Local auth
21.1.1 MUST support local authentication (no external IdP required).
21.1.2 Password hashing MUST use a modern KDF (Argon2id recommended).

21.2 Roles
21.2.1 Roles MUST include: admin, maintainer, writer, reader.
21.2.2 Permissions MUST be repo-scoped via ACL.

21.3 Audit log
21.3.1 Audit log MUST be append-only.
21.3.2 Audit MUST record at least:
	•	ref updates
	•	commit creation
	•	MR open/merge/close
	•	user/role changes
	•	export/import
	•	maintenance operations (rebalance, GC if any)
21.3.3 Audit entries MUST include:
	•	event_id (UUIDv7)
	•	ts (unix seconds)
	•	actor_id (UUIDv7)
	•	action (string)
	•	repo_id (UUIDv7|null)
	•	details_json (JSON)

	22.	Security and Operational Limits
22.1 Limits
22.1.1 MUST implement configurable:

	•	request size limits
	•	upload size limits
	•	rate limiting

22.2 Crash safety
22.2.1 DB migrations MUST be crash-safe.
22.2.2 The system MUST NOT corrupt meta.db on crash.

22.3 Logging
22.3.1 Logs SHOULD be structured JSON and include request_id.
22.3.2 Logs MUST NOT include plaintext passwords or password hashes.
	23.	Export/Import (Deterministic)
23.1 Archive format (normative)
23.1.1 Export MUST produce a single file: TAR archive compressed with Zstandard.
23.1.2 Filename extension SHOULD be .tar.zst.

23.2 Archive contents (normative)
23.2.1 Archive MUST contain:
	•	meta.db (a consistent snapshot, see §23.4)
	•	objects/sha256// (all CAS object files)
	•	manifest.json

23.3 Deterministic manifest (normative)
23.3.1 manifest.json MUST be JCS-canonical JSON with:
	•	spec_version: “0.0.1”
	•	created_at: unix seconds
	•	repo_ids: [UUIDv7…]
	•	files: [{ path: string, sha256_hex: string, size: int }]
23.3.2 files MUST list every file in the archive excluding manifest.json.
23.3.3 files MUST be sorted by path ascending (bytewise ASCII).
23.3.4 sha256_hex MUST be sha256 over the exact bytes of the archived file.

23.4 SQLite snapshot rule (required)
23.4.1 Export MUST archive a consistent meta.db snapshot even if SQLite WAL is enabled.
23.4.2 Implementations MUST satisfy §23.4.1 by one of:
	•	using SQLite Online Backup API to write a snapshot meta.db file, then archiving that snapshot
	•	an equivalent mechanism that guarantees transactional consistency
23.4.3 Export MUST NOT rely on copying meta.db alone when WAL is enabled unless it also guarantees inclusion or checkpointing of all committed data.

23.5 Import (normative)
23.5.1 Import MUST restore refs, MRs, ACLs, and audit history.
23.5.2 IDs MUST remain valid across export/import (blob/tree/commit IDs unchanged).
23.5.3 Import MUST fail if any required file checksum mismatches manifest.
23.5.4 Import MUST be atomic per data-dir: on failure, the server MUST not leave a partially imported state.
	24.	SQLite meta.db Minimum Schema (Required)
24.1 Required tables

	•	repos(repo_id TEXT PRIMARY KEY, name TEXT NULL, created_at INTEGER NOT NULL)
	•	refs(repo_id TEXT NOT NULL, ref_name TEXT NOT NULL, commit_id BLOB NOT NULL, updated_at INTEGER NOT NULL, PRIMARY KEY(repo_id, ref_name))
	•	mrs(mr_id TEXT PRIMARY KEY, repo_id TEXT NOT NULL, base_ref TEXT NOT NULL, head_ref TEXT NOT NULL, base_commit_id BLOB NOT NULL, status TEXT NOT NULL, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL)
	•	audit(event_id TEXT PRIMARY KEY, ts INTEGER NOT NULL, actor_id TEXT NOT NULL, action TEXT NOT NULL, repo_id TEXT NULL, details_json TEXT NOT NULL)
	•	users(user_id TEXT PRIMARY KEY, handle TEXT UNIQUE NOT NULL, created_at INTEGER NOT NULL, password_hash BLOB NOT NULL, password_params_json TEXT NOT NULL)
	•	repo_acl(repo_id TEXT NOT NULL, user_id TEXT NOT NULL, role TEXT NOT NULL, PRIMARY KEY(repo_id, user_id))
	•	idempotency(actor_id TEXT NOT NULL, method TEXT NOT NULL, path TEXT NOT NULL, key TEXT NOT NULL, created_at INTEGER NOT NULL, response_status INTEGER NOT NULL, response_body BLOB NOT NULL, PRIMARY KEY(actor_id, method, path, key))
	•	sessions(session_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, created_at INTEGER NOT NULL, expires_at INTEGER NOT NULL)

24.2 Atomicity
24.2.1 Ref updates, MR status changes, idempotency records, and session changes MUST be transactionally consistent.
	25.	Invariants
25.1 CAS objects are immutable.
25.2 commit/tree IDs are sha256 of canonical CBOR bytes.
25.3 chapter/scene/manifest/ui_manifest JSON blobs are NFC-normalized, forbidden characters rejected, LF-normalized where applicable, then JCS-canonical.
25.4 body_md is LF-normalized and raw HTML disabled; links are sanitized.
25.5 scene_id/chapter_id are content-independent UUIDv7.
25.6 Ordering is stable sort with tie-break by id.
25.7 Split produces new scene_ids; merge-of produces a new scene_id.
25.8 Rebalance and conflict resolutions always create new commits; no in-place edits of history.
25.9 Export/import restores identical object IDs and repository state; import fails on checksum mismatch.
25.10 UI is pinned to specified toolchain and is served from embedded bytes with minimum security headers and CSP.

End of v0.0.1
