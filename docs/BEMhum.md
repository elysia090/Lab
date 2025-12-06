BEM Human Feedback Layer v0.0.1
Normative Human-in-the-Loop Interface (Tightened Draft)
	0.	Scope

0.1 Purpose

This document defines a precise Human Feedback Layer on top of BEM v0.0.1 and the Tool and Resource Layer v0.0.1. It specifies:
	1.	Identifier classes and data structures for human-feedback tasks and responses.
	2.	A HINL_TASK table in SHARED and its lifecycle.
	3.	Binding to the Tool Bus via TOOL_HINL and TOOL_REQ.
	4.	Algorithms for injecting human feedback into:
	•	Bandit Core (expert selection).
	•	GRPO_LITE (mid-path policy updates).
	•	TEACH (task grammar and curriculum).
	•	PoX and patch scheduling.
	5.	Execution, complexity, and safety constraints.

The goal is to enable Human-in-the-Loop (HITL) while preserving BEM fast-path constraints and auditability.

0.2 Assumptions
	1.	BEM core, segments, and algorithms are as specified in BEM v0.0.1.
	2.	Tool Bus and Resource Catalog are as in Tool and Resource Layer v0.0.1.
	3.	All human feedback is asynchronous relative to STEP_FAST.

0.3 Notation
	1.	U is the 32-bit identifier domain.
	2.	STATE, SHARED, TRACE, WORK, TEACH are logical segments.
	3.	step_global is the global step counter in SHARED.
	4.	All structures are conceptual and may be laid out differently in physical memory.
	5.	Identifier Classes and Global Constants

1.1 New identifier classes

The following class tags in U are reserved for the Human Feedback Layer:
	1.	class = hinl_task
Identifiers for human feedback tasks.
	2.	class = episode_id
Identifiers for episodes (if not already defined). May reuse task_id-dependent schemes.
	3.	class = patch_id
Identifiers for structural patches Delta.

Exact numeric values for class must be fixed at configuration time and consistent with other class definitions.

1.2 Human feedback tool kind

A tool kind identifier TOOL_HINL in U with class = tool_kind is reserved for human feedback. Its TOOL_KIND entry must set a role_flags bit indicating “human tool”.

1.3 Enumerations

Define fixed-size enums as integers in SHARED:

HINL_TARGET_KIND (u8):
0 = HINL_TARGET_STEP
1 = HINL_TARGET_EPISODE
2 = HINL_TARGET_PATCH
3 = HINL_TARGET_TEMPLATE
4 = HINL_TARGET_CONFIG

HINL_KIND (u8):
0 = HINL_KIND_LABEL_STEP_ACTION
1 = HINL_KIND_SCORE_EPISODE
2 = HINL_KIND_APPROVE_PATCH
3 = HINL_KIND_RANK_TEMPLATES
4 = HINL_KIND_ADJUST_CONFIG

HINL_STATUS (u8):
0 = HINL_STATUS_EMPTY
1 = HINL_STATUS_PENDING
2 = HINL_STATUS_IN_PROGRESS
3 = HINL_STATUS_DONE
4 = HINL_STATUS_SKIPPED

HINL_PRIORITY (u8):
0 = HINL_PRIORITY_LOW
1 = HINL_PRIORITY_NORMAL
2 = HINL_PRIORITY_HIGH
3 = HINL_PRIORITY_CRITICAL

HINL_DECISION_PATCH (u8):
0 = HINL_PATCH_UNDECIDED
1 = HINL_PATCH_APPROVE
2 = HINL_PATCH_REJECT
3 = HINL_PATCH_DEFER
	2.	SHARED Structures

2.1 HINL_TASK table

The HINL_TASK table is a fixed array in SHARED:

For each index h in [0, H):

HINL_TASK[h] = (
id,             // U, class=hinl_task
kind,           // HINL_KIND
target_kind,    // HINL_TARGET_KIND
target_ref,     // U, target identifier
priority,       // HINL_PRIORITY
status,         // HINL_STATUS
ctx_ptr,        // u64, pointer to context payload
ctx_len,        // u32
answer_ptr,     // u64, pointer to answer payload
answer_cap,     // u32
deadline_step,  // u64, optional, 0 means no deadline
created_step,   // u64
last_update_step,// u64
meta_flags      // u32
)

Constraints:
	1.	HINL_TASK[h].id must be unique among active tasks (status != EMPTY).
	2.	HINL_TASK[h].ctx_ptr and answer_ptr must point into SHARED or a designated HITL buffer segment.
	3.	ctx_len and answer_cap must be within configured bounds.

2.2 HINL_CONTROL

A control record in WORK:

HINL_CONTROL = (
enable_online,          // u8 (0 or 1)
max_online_step_rate,   // u16, max percent of steps that may generate HINL tasks (0..10000 for basis 100)
max_online_episode_rate,// u16, max percent of episodes
max_pending_tasks,      // u16, cap on HINL_TASK entries with status in {PENDING,IN_PROGRESS}
safety_override_mode,   // u8, semantics below
min_confidence_for_auto,// fixed-point, threshold on bandit confidence
human_weight_bandit,    // fixed-point, scale for bandit updates
human_weight_grpo,      // fixed-point
human_weight_teacher,   // fixed-point
human_weight_pox,       // fixed-point
reserved                // padding or future fields
)

safety_override_mode (u8):
0 = HINL_SAFETY_OFF
1 = HINL_SAFETY_WARN
2 = HINL_SAFETY_REQUIRE_APPROVAL (critical patches must have human approval)
3 = HINL_SAFETY_RESTRICT_TO_WHITELIST (only human-approved patches/templates)

2.3 HINL_STATS

A statistics record in WORK to track HITL usage:

HINL_STATS = (
total_tasks_created,        // u64
total_tasks_completed,      // u64
total_tasks_skipped,        // u64
total_step_feedback_used,   // u64
total_episode_feedback_used,// u64
total_patch_feedback_used,  // u64
total_template_feedback_used,// u64
last_task_id_seq,           // u32, monotonically increasing per new HINL_TASK
reserved                    // space for more counters
)
	3.	Tool Bus Binding for Human Feedback

3.1 TOOL_REQ usage

Human feedback uses TOOL_REQ entries as defined in the Tool and Resource Layer. For TOOL_HINL, the following op codes are defined (u16):

0 = HINL_OP_ENQUEUE_TASK
1 = HINL_OP_NOTIFY (optional, for push-based UI)
2 = HINL_OP_QUERY_TASKS (optional, for pull-based UI meta queries)

3.2 HINL_OP_ENQUEUE_TASK semantics

Input payload (at in_ptr) defines a requested HINL task:

HINL_ENQUEUE_REQ = (
kind,           // HINL_KIND
target_kind,    // HINL_TARGET_KIND
target_ref,     // U
priority,       // HINL_PRIORITY
ctx_ptr,        // u64, pointer to context payload
ctx_len,        // u32
deadline_step   // u64 (0 if none)
)

Output payload (at out_ptr) is:

HINL_ENQUEUE_RES = (
hinl_task_id,   // U, class=hinl_task or null id on failure
status_code     // u16, 0=OK, non-zero=error
)

Semantics:
	1.	The HITL worker (external) may implement HINL_OP_ENQUEUE_TASK as a convenience wrapper.
However, in many deployments, BEM mid-path will write HINL_TASK entries directly without going through TOOL_REQ for internal generation.
HINL_OP_ENQUEUE_TASK is required for external tools to create HINL tasks.

3.3 HINL_OP_NOTIFY and HINL_OP_QUERY_TASKS

These are optional and non-normative. They may be used by a UI gateway or broker to receive task metadata and push updates. The Human Feedback Layer does not assume their existence.
	4.	Task Lifecycle

4.1 Allocation

Allocation is managed by BEM mid-path or slow-path code:

Algorithm HINL_ALLOC_TASK

Input:
	•	desired HINL_ENQUEUE_REQ r
	•	maximum number of pending tasks, from HINL_CONTROL.max_pending_tasks

Output:
	•	index h or bottom if allocation fails

Steps:
	1.	Count current_pending = number of HINL_TASK entries with status in {PENDING, IN_PROGRESS}.
	2.	If current_pending >= max_pending_tasks:
return bottom.
	3.	Scan HINL_TASK for first index h where status = EMPTY.
	4.	If none found:
return bottom.
	5.	Generate new id as:
task_seq = HINL_STATS.last_task_id_seq + 1
HINL_STATS.last_task_id_seq = task_seq
id = encode_hinl_task_id(task_seq) in U with class=hinl_task
	6.	Write:
HINL_TASK[h].id              = id
HINL_TASK[h].kind            = r.kind
HINL_TASK[h].target_kind     = r.target_kind
HINL_TASK[h].target_ref      = r.target_ref
HINL_TASK[h].priority        = r.priority
HINL_TASK[h].status          = HINL_STATUS_PENDING
HINL_TASK[h].ctx_ptr         = r.ctx_ptr
HINL_TASK[h].ctx_len         = r.ctx_len
HINL_TASK[h].answer_ptr      = preallocated_answer_buffer(h)
HINL_TASK[h].answer_cap      = configured_answer_cap
HINL_TASK[h].deadline_step   = r.deadline_step
HINL_TASK[h].created_step    = step_global
HINL_TASK[h].last_update_step= step_global
HINL_TASK[h].meta_flags      = 0
	7.	Increment HINL_STATS.total_tasks_created.
	8.	Return h.

Complexity: O(H) bounded by small constant H.

4.2 UI worker behavior

External UI workers periodically:
	1.	Scan HINL_TASK for entries with status = PENDING.
	2.	For each selected entry h:
a) Set status = IN_PROGRESS, last_update_step = current_step_global_or_wallclock.
b) Read context from ctx_ptr, ctx_len.
c) Render context in a UI, ask the human for input according to kind and target_kind.
d) Encode the answer into memory at answer_ptr, not exceeding answer_cap bytes.
e) Set status = DONE, last_update_step updated.

If the worker decides not to handle a task, it may leave status as PENDING, or set SKIPPED in special cases (for example expired deadlines).

4.3 Expiration and skipping

Mid-path HINL maintenance:

Algorithm HINL_MAINTENANCE

For each h in [0,H):
	1.	If status = PENDING or IN_PROGRESS:
a) If HINL_TASK[h].deadline_step != 0 and step_global > deadline_step:
i) Set status = SKIPPED.
ii) HINL_STATS.total_tasks_skipped++.
	2.	If status = DONE or SKIPPED and answer has been processed:
a) Optionally clear the entry:
HINL_TASK[h].status = EMPTY
Other fields may be zeroed or left as debugging info.
	3.	Feedback Payload Formats

5.1 Step-level action label

Context payload for kind = LABEL_STEP_ACTION, target_kind = STEP:

HINL_CTX_STEP = (
task_id,        // int, current task_id_t
context_hash,   // u64, context_hash_t
candidate_count,// u8, number of candidates in C_large
candidate_ids[],// array of candidate expert identifiers or slot indices
chosen_id,      // expert id or slot index actually chosen i_t
reward_observed,// optional fixed-point reward or 0 if not yet known
step_index,     // local step index in episode
episode_id      // U, optional
)

Answer payload:

HINL_ANS_STEP = (
version,        // u8
accept_flag,    // u8 (0 or 1)
best_id,        // expert id or slot index or null marker
score,          // fixed-point in [0,1], human rating of decision quality
comment_tag,    // small u16 tag referencing predefined comment category
reserved
)

5.2 Episode-level score

Context payload for kind = SCORE_EPISODE:

HINL_CTX_EPISODE = (
episode_id,     // U
task_id,        // int
length,         // u32
return_env,     // fixed-point return from environment
return_shaped,  // fixed-point return used in current training
regret_est,     // fixed-point regret estimate if available
flags           // u32 (BAD events, safety flags, etc)
)

Answer payload:

HINL_ANS_EPISODE = (
version,        // u8
score_h,        // fixed-point in [0,1]
keep_flag,      // u8 (0=discard episode,1=keep,2=keep for curriculum only)
tag_bits,       // u32 (safety, quality, novelty, etc)
reserved
)

5.3 Patch approval

Context payload for kind = APPROVE_PATCH:

HINL_CTX_PATCH = (
patch_id,       // U
patch_kind,     // u8 (split, merge, macro, superopt, etc)
pox_score,      // fixed-point
delta_R,        // fixed-point
delta_S,        // fixed-point
delta_C,        // fixed-point
delta_I,        // fixed-point
risk_flags      // u32 (modified modules, environment domains)
)

Answer payload:

HINL_ANS_PATCH = (
version,        // u8
decision,       // HINL_DECISION_PATCH
risk_tag_bits,  // u32
human_weight,   // fixed-point weight override (0 to up-weight or down-weight)
reserved
)

5.4 Template ranking

Context payload for kind = RANK_TEMPLATES:

HINL_CTX_TEMPLATES = (
template_count, // u8
template_ids[], // up to K template ids
stats[]         // per-template stats: usage, pass_rate, difficulty, etc
)

Answer payload:

HINL_ANS_TEMPLATES = (
version,        // u8
order[],        // permutation or ranking of templates
score[],        // optional score per template in [0,1]
reserved
)

5.5 Config adjustment

Context payload for kind = ADJUST_CONFIG:

HINL_CTX_CONFIG = (
current_HINL_CONTROL,// snapshot of control struct
current_safety_stats,// aggregated safety metrics
current_PoX_stats    // aggregated patch acceptance metrics
)

Answer payload:

HINL_ANS_CONFIG = (
version,            // u8
enable_online,      // optional override or -1 for no change
safety_override_mode,// optional override or -1
min_conf_auto_delta,// fixed-point delta to apply or 0
weight_bandit_delta,// fixed-point delta
weight_grpo_delta,  // fixed-point delta
weight_teacher_delta,// fixed-point delta
weight_pox_delta,   // fixed-point delta
reserved
)
	6.	Injection into Learning Components

6.1 Injection into Bandit Core

Algorithm HINL_APPLY_STEP_FEEDBACK

Executed in mid-path periodically:

For each HINL_TASK[h] with:
	•	kind = LABEL_STEP_ACTION
	•	status = DONE

Steps:
	1.	Decode HINL_CTX_STEP from ctx_ptr and HINL_ANS_STEP from answer_ptr.
	2.	Let tau = task_id from context.
	3.	Let C_large = candidate_ids array, and i_t = chosen_id.
	4.	If accept_flag = 0:
a) Define human_loss_penalty in [0,1], derived from score and HINL_CONTROL.human_weight_bandit.
b) For expert i_t and task tau:
i) Compute synthetic loss_h = max(loss_min, min(1.0, human_loss_penalty)).
ii) Apply BANDIT_UPDATE_STEP style update using loss_h instead of environment loss.
	5.	If best_id is valid and in C_large and best_id != i_t:
a) For expert best_id and task tau:
i) Compute synthetic loss_best = low value (for example 0).
ii) Apply a weighted update:
- treat selection as if best_id was chosen with small fractional count w_best = human_weight_bandit.
- update visits_best_tau += w_best (in fixed-point accumulator).
- update loss_sum_best_tau += w_best * loss_best.
b) Optionally, penalize i_t with a slightly higher loss for this step.
	6.	Update HINL_STATS.total_step_feedback_used++.
	7.	Mark HINL_TASK[h] as processed (either SKIPPED or EMPTY after logging).

Notes:
	1.	Fixed-point fractional counts can be implemented by maintaining separate fractional accumulators or by scaling counters.
	2.	The goal is to shift priors z_i_tau and empirical means toward human preferences without violating no-regret properties too strongly.

6.2 Injection into GRPO_LITE

Algorithm HINL_APPLY_EPISODE_FEEDBACK

For each HINL_TASK[h] with:
	•	kind = SCORE_EPISODE
	•	status = DONE

Steps:
	1.	Decode HINL_CTX_EPISODE and HINL_ANS_EPISODE.
	2.	Let R_env = return_env, R_shaped = return_shaped.
	3.	Let score_h in [0,1].
	4.	Define blended return R_blend = (1 - alpha_h) * R_shaped + alpha_h * f(score_h), where:
	•	alpha_h = clamp(HINL_CONTROL.human_weight_grpo, 0, 1) or derived.
	•	f is a monotone mapping from [0,1] to reward scale.
	5.	In GRPO_LITE_UPDATE, use R_blend for this episode instead of R_shaped.
	6.	If keep_flag indicates discard, mark the episode as excluded from training but still log it.
	7.	Update HINL_STATS.total_episode_feedback_used++.
	8.	Mark HINL_TASK[h] as processed.

6.3 Injection into TEACH

Algorithm HINL_APPLY_TEMPLATE_FEEDBACK

For each HINL_TASK[h] with:
	•	kind = RANK_TEMPLATES
	•	status = DONE

Steps:
	1.	Decode HINL_CTX_TEMPLATES and HINL_ANS_TEMPLATES.
	2.	For each template t in the context:
a) Compute human preference signal p_t from order and score fields.
b) Update TEACH bandit stats for template t:
	•	treat high p_t as low loss template.
	•	adjust z_template_t via:
z_template_t <- z_template_t + human_weight_teacher * g(p_t)
	3.	Optionally adjust difficulty estimates d_t, pass_rate_t based on human hints.
	4.	Update HINL_STATS.total_template_feedback_used++.
	5.	Mark HINL_TASK[h] as processed.

6.4 Injection into PoX and patch scheduling

Algorithm HINL_APPLY_PATCH_FEEDBACK

For each HINL_TASK[h] with:
	•	kind = APPROVE_PATCH
	•	status = DONE

Steps:
	1.	Decode HINL_CTX_PATCH and HINL_ANS_PATCH.
	2.	Retrieve patch Delta metadata by patch_id.
	3.	Let decision be the human decision, human_weight a scaling factor.
	4.	If decision = APPROVE:
a) Increase patch PoX score by:
score(Delta) <- score(Delta) + HINL_CONTROL.human_weight_pox * human_weight.
b) Optionally set a flag “human_approved” in patch metadata.
	5.	If decision = REJECT:
a) Mark patch as rejected (blacklist) and do not apply.
b) Optionally penalize any automated proposal mechanism that produced it.
	6.	If decision = DEFER:
a) Leave patch pending; scheduler may deprioritize it.
	7.	If safety_override_mode enforces human approval for critical patches:
a) Only apply patches with decision = APPROVE when they are marked as critical.
	8.	Update HINL_STATS.total_patch_feedback_used++.
	9.	Mark HINL_TASK[h] as processed.

6.5 Injection into configuration

Algorithm HINL_APPLY_CONFIG_FEEDBACK

For each HINL_TASK[h] with:
	•	kind = ADJUST_CONFIG
	•	status = DONE

Steps:
	1.	Decode HINL_ANS_CONFIG.
	2.	For each field in HINL_CONTROL:
a) If corresponding delta or override is provided:
	•	apply bounded change (for example clamp to valid ranges).
	3.	Update WORK with new HINL_CONTROL.
	4.	Mark HINL_TASK[h] as processed.
	5.	Online HITL Triggering Rules

7.1 Trigger for step-level HITL

Mid-path can create LABEL_STEP_ACTION tasks according to heuristics:

Trigger conditions:
	1.	Uncertainty threshold:
	•	bandit index gap between top-1 and top-2 experts below a threshold.
	2.	Safety-related events:
	•	BAD_t flagged.
	3.	Novel context:
	•	context_hash_t in low-frequency region.
	4.	Rate control:
	•	fraction of steps with HINL tasks in a window below HINL_CONTROL.max_online_step_rate.

Algorithm HINL_MAYBE_CREATE_STEP_TASK

Input:
	•	step context, including task_id, context_hash, candidate experts, chosen expert, reward if available.

Output:
	•	none (may create HINL_TASK).

Steps:
	1.	If HINL_CONTROL.enable_online = 0:
return.
	2.	Evaluate trigger conditions. If none met:
return.
	3.	Build HINL_CTX_STEP payload into a buffer.
	4.	Call HINL_ALLOC_TASK with:
kind = LABEL_STEP_ACTION
target_kind = HINL_TARGET_STEP
target_ref = encode_step_ref(episode_id, local_step_index)
priority = based on safety and uncertainty
ctx_ptr, ctx_len = buffer
deadline_step = step_global + configured_step_deadline
	5.	If allocation fails, silently drop or increment a dropped-HITL counter.

7.2 Trigger for episode-level HITL

Triggers:
	1.	High regret episodes.
	2.	Episodes flagged as potentially unsafe or very high/low reward.
	3.	Rate-limited by max_online_episode_rate.

Algorithm HINL_MAYBE_CREATE_EPISODE_TASK

Similar to step-level but using HINL_CTX_EPISODE and kind = SCORE_EPISODE.

7.3 Trigger for patch HITL

Triggers:
	1.	Patch Delta marked as critical by PoX (touches safety-sensitive modules).
	2.	Patch PoX score near threshold D.
	3.	safety_override_mode requires approval for given patch kind.

Algorithm HINL_MAYBE_CREATE_PATCH_TASK

Create APPROVE_PATCH tasks with high priority for patches matching criteria.
	8.	Complexity and Safety

8.1 Complexity constraints
	1.	All HITL-related work on fast-path is optional and bounded:
	•	STEP_FAST may log metadata, but must not allocate HINL_TASK or read answer_ptr directly.
	2.	Mid-path HITL operations:
	•	HINL task allocation and scanning are O(H) with small H.
	•	Application of feedback must have bounded cost independent of episode length beyond local context.
	3.	UI workers:
	•	May run at arbitrary rate; their latency does not affect fast-path guarantees.

8.2 Safety constraints
	1.	Human feedback is not trusted more than verification:
	•	Structural patches still require PROVER and PoX checks, even if human approves.
	2.	safety_override_mode can enforce:
	•	Patches without human approval are blocked for certain modules.
	•	HITL must not override hard safety invariants encoded in PROOF.

8.3 Auditability
	1.	All HINL task creation, updates, and applications should be logged in TRACE:
	•	task id, kind, target, timestamps, decisions, and hash of context and answer payloads.
	2.	LOG unit can compute hashes and append them to the log chain.
	3.	Summary

The Human Feedback Layer v0.0.1 defines:
	1.	A precise HINL_TASK table with statuses and payload pointers in SHARED.
	2.	TOOL_HINL as a tool kind on the Tool Bus for external UI workers.
	3.	Exact payload formats for step labels, episode scores, patch approvals, template rankings, and config adjustments.
	4.	Algorithms to map human feedback into bandit updates, GRPO adjustments, TEACH priors, PoX scores, and control knobs.
	5.	Trigger rules and rate controls to integrate humans into BEM’s learning loop without violating fast-path constraints.

This layer allows BEM to treat humans as a structured, asynchronous teaching and oversight signal, while keeping the core decision-making loop finite, bounded, and auditable.
