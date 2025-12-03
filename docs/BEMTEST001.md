BEM v0.0.1 – Boolean Expert Machine
Core + ISA Specification (ASCII only, tightened)

Status
Experimental, implementation-oriented, hardware-oriented model

Language
English, ASCII only
	0.	Overview

0.1 Scope

This document defines BEM v0.0.1 as
	1.	A finite-state abstract machine with bit-sliced global state and shared memory.
	2.	A small RISC-like integer core ISA extended with BEM-specific instructions.
	3.	A fixed set of co-processors (ANN, SAT/Hoare, HASH/ECC, LOG, optional WORLD).
	4.	A fast path with strictly bounded per-step cost (independent of sequence length and history).
	5.	A slow path for self-modification, gated by formal verification and optional PoX scoring.

The goal is to provide a concrete target for implementations that:
	•	Execute Boolean experts over bit-sliced state.
	•	Route between experts using bandit-style learning and ANN-based candidate retrieval.
	•	Evolve their own structure (experts, macros, invariants) under verification.
	•	Maintain auditable logs and integrity proofs.

0.2 Non-goals

BEM v0.0.1 does not define:
	•	Concrete binary encodings of instructions.
	•	Micro-architectural details (pipelines, cache hierarchies, TLBs).
	•	Performance, area, or power targets.
	•	Any host ABI, calling convention, or OS integration.
	•	Any floating-point or neural primitives on the fast path.

Floating-point or neural components may exist as untrusted proposal generators (for example for candidate experts, invariants, or patches). Their outputs must be validated by the verification kernel before they can change the trusted fast path or structural state.

0.3 Parameters and symbols

Global configuration parameters:
	•	XLEN: integer register width in bits (for example 32 or 64).
	•	W: SIMD width (number of logical lanes for bit-slicing).
	•	N: number of global state bits.
	•	K: number of shared memory bits (K >= N).
	•	P: number of fixed-point parameters.
	•	S: maximum number of expert slots (EXPERT[0..S-1]).
	•	d: feature-bit dimension for routing (for example 64..256).
	•	k_large: size of ANN candidate set (for example 16..64).
	•	k_small: size of final routing candidate set (for example 1..4).
	•	D_max: maximum ANN search depth.
	•	M_max: maximum ANN neighbor degree.

Notation:
	•	Bits are elements of {0,1}.
	•	Bitvectors of length n are elements of {0,1}^n.
	•	Integers are elements of Z, represented as fixed-width XLEN-bit words when stored in registers.
	•	Per-step complexity refers to core cycles and co-processor invocations associated with one logical step (event or token). It must be bounded by a constant that does not depend on episode length or total number of past steps.

	1.	Machine state and segments

1.1 Logical state

At logical time t the BEM state is

X_t = (s_t, M_t, Theta_t, C_t)

where
	•	s_t in {0,1}^N: global state bits (bit-sliced over W lanes).
	•	M_t in {0,1}^K: shared memory bits (scalar).
	•	Theta_t in Z^P: fixed-point parameters (for example bandit hyperparameters, PoX weights).
	•	C_t: configuration (EXPERT table, CFG/CODE, ANN index, invariants, WORK configuration).

Fast path:
	•	May read and write s_t and M_t.
	•	May read Theta_t and structural tables in C_t.
	•	Must not modify structural tables.

Slow path:
	•	May modify Theta_t and C_t via patches that have passed verification and optional PoX checks.

1.2 Memory segments

Logical memory is partitioned into disjoint segments:
	1.	STATE
	•	Holds bit-sliced global state.
	•	For each bit index k in [0, N):
	•	STATE[k] in {0,1}^W stores bit k across all W lanes.
	•	Lane l views s_t^(l)[k] as bit l of STATE[k].
	2.	SHARED
	•	Holds scalar data and mutable metadata:
	•	M_t (shared bits, packed into words).
	•	Theta_t (fixed-point parameter arrays).
	•	routing statistics and bandit stats.
	•	ANN indices and configuration.
	•	task and context metadata.
	•	co-processor arguments and results.
	3.	EXPERT
	•	EXPERT[0..S-1] descriptors for experts.
	4.	CFG / CODE
	•	Instruction array C[0..L-1].
	•	CFG node metadata (pc ranges, successor lists, invariants).
	•	Macro descriptors.
	5.	TRACE / LOG
	•	Step-level traces (optional).
	•	Episode summaries.
	•	Patch metadata.
	•	PoX records.
	6.	PROOF
	•	CNF formulas.
	•	Solver contexts.
	•	Proof objects and unsat cores.
	7.	WORK (optional PoX)
	•	PoX weights and thresholds.
	•	Moving averages and saturation metrics.

From the ISA perspective, all segments are part of a flat byte-addressed memory. Segment boundaries and protections are established by configuration registers and software. The abstract semantics assumes no aliasing between segments.

1.3 Identifier space

Identifiers are 32-bit integers:
	•	U = {0, 1, …, 2^32 - 1}.

Each identifier u in U is logically structured as:
	•	u = [class(6) | ecc(6) | shard(6) | local(14)]

where:
	•	class in {0..63}: object class (expert, cfg_node, variable, template, patch, hypothesis, etc).
	•	ecc in {0..63}: ECC parity bits over shard and local.
	•	shard in {0..63}: logical shard index.
	•	local in {0..16383}: shard-local index.

Gray encoding:
	•	g(u) = u xor (u >> 1)

Hamming distance:
	•	d_H(a, b) = number of bits where a and b differ.

Similarity function:
	•	sim(u, v) = 32 - d_H(g(u), g(v)).

Identifiers are used for experts, CFG nodes, CNF formulas, solver contexts, patches, and other objects.

1.4 Expert descriptors

For each expert slot i in [0, S):

EXPERT[i] = (id_i, R_spec_i, C_rep_i, W_spec_i, stats_i, params_i, routing_meta_i)

where:
	•	id_i in U: expert identifier.
	•	R_spec_i: input selection specification.
	•	Defines how to read a finite vector x from (s_t, M_t):
	•	bit indices into STATE and SHARED,
	•	simple address expressions (for example base + offset),
	•	dimension n_i.
	•	C_rep_i: Boolean circuit representation (ANF or ROBDD).
	•	Maps x in {0,1}^{n_i} to y in {0,1}^{m_i}.
	•	W_spec_i: write-back specification.
	•	Defines how to write y back into (s, M):
	•	target bit indices or contiguous regions,
	•	optional lane masks.
	•	stats_i:
	•	wins_total_i in Z_nonneg.
	•	visits_total_i in Z_nonneg.
	•	per-task wins_i,tau and visits_i,tau (tau = task id).
	•	last_update_step_i in Z.
	•	params_i:
	•	z_i: log-weight (fixed-point).
	•	L_i,tau, S_i,tau: cumulative loss and squared loss per task tau.
	•	routing_meta_i:
	•	priority tags.
	•	macro flags.
	•	pointer into ANN index structures.
	•	version identifiers.

Input and output sizes are bounded by configuration constants:
	•	n_i <= N_in_max.
	•	m_i <= N_out_max.

	2.	ISA model

2.1 Programmer-visible core state

Registers:
	•	x0..xR-1: general-purpose integer registers, each XLEN bits.
	•	x0 is hard-wired to zero.
	•	pc: program counter, byte address.
	•	fcsr: BEM control and status register (exception flags, modes, etc).
	•	csr_seg_*: implementation-defined CSRs for segment base and limit.

Optional vector and mask registers:
	•	v0..vV-1: W-bit vector temporaries (optional).
	•	k0..kK-1: W-bit mask or predicate registers (optional).

The abstract semantics requires only GPRs, pc, and memory. Vector and mask registers are an implementation optimization; if absent, operations on STATE are still defined.

2.2 Memory model
	•	Flat byte-addressed memory.
	•	Load and store instructions operate on bytes and XLEN-bit words.
	•	Segment layout is enforced by CSRs and software.
	•	Accesses outside valid ranges are undefined at the abstract level (implementation may trap).

2.3 Instruction classes

Instructions are grouped into classes:
	1.	Base integer RISC:
	•	integer arithmetic and logical operations,
	•	comparisons and branches,
	•	loads and stores,
	•	CSR access.
	2.	Bitwise and population-count:
	•	bitwise operations on XLEN-bit words,
	•	optional vector-level bitwise and popcount.
	3.	BEM-specific:
	•	observation (BEM_OBS),
	•	routing (BEM_ROUTE),
	•	expert evaluation (BEM_EXPERT_BATCH),
	•	stats and learning (BEM_STATS_UPDATE, BEM_LEARN),
	•	logging and Merkle updates (BEM_LOG, BEM_MERKLE_UPDATE),
	•	fast-path macro (BEM_STEP_FAST, optional).
	4.	Co-processor call:
	•	COPROC cp_id, op_id, arg_ptr, res_ptr.

Binary encodings and exact opcode layouts are implementation-defined. This specification defines only abstract semantics.
	3.	Base integer ISA

3.1 Arithmetic and logical operations

All operations take XLEN-bit inputs and produce XLEN-bit results.
	•	ADD rd, rs1, rs2
x[rd] = x[rs1] + x[rs2].
	•	ADDI rd, rs1, imm
x[rd] = x[rs1] + imm (imm sign-extended).
	•	SUB rd, rs1, rs2
x[rd] = x[rs1] - x[rs2].
	•	AND rd, rs1, rs2
x[rd] = x[rs1] bitwise_and x[rs2].
	•	ANDI rd, rs1, imm
x[rd] = x[rs1] bitwise_and imm.
	•	OR rd, rs1, rs2
x[rd] = x[rs1] bitwise_or x[rs2].
	•	ORI rd, rs1, imm
x[rd] = x[rs1] bitwise_or imm.
	•	XOR rd, rs1, rs2
x[rd] = x[rs1] bitwise_xor x[rs2].
	•	XORI rd, rs1, imm
x[rd] = x[rs1] bitwise_xor imm.

Shifts:
	•	SLL rd, rs1, rs2
x[rd] = x[rs1] << (x[rs2] low bits).
	•	SRL rd, rs1, rs2
x[rd] = logical right shift.
	•	SRA rd, rs1, rs2
x[rd] = arithmetic right shift.

Comparisons and set:
	•	SLT rd, rs1, rs2
x[rd] = 1 if x[rs1] < x[rs2] (signed), else 0.
	•	SLTI rd, rs1, imm
x[rd] = 1 if x[rs1] < imm (signed), else 0.
	•	SLTU rd, rs1, rs2
x[rd] = 1 if x[rs1] < x[rs2] (unsigned), else 0.
	•	SLTIU rd, rs1, imm
unsigned variant with immediate.

3.2 Branches and jumps
	•	BEQ rs1, rs2, offset
If x[rs1] == x[rs2], pc = pc + offset; else pc = pc + instruction_length.
	•	BNE rs1, rs2, offset
If x[rs1] != x[rs2], branch.
	•	BLT, BGE, BLTU, BGEU
Branch on signed or unsigned comparisons.
	•	JAL rd, offset
x[rd] = pc + instruction_length; pc = pc + offset.
	•	JALR rd, rs1, imm
x[rd] = pc + instruction_length; pc = (x[rs1] + imm) & alignment_mask.

3.3 Loads and stores
	•	LB rd, offset(rs1)
Load signed byte, sign-extend to XLEN.
	•	LBU rd, offset(rs1)
Load unsigned byte, zero-extend.
	•	LW rd, offset(rs1) (XLEN >= 32)
Load 32-bit word, sign-extend.
	•	LD rd, offset(rs1) (XLEN = 64)
Load XLEN-bit word.
	•	SB rs2, offset(rs1)
Store low 8 bits of x[rs2] to memory.
	•	SW rs2, offset(rs1)
Store low 32 bits of x[rs2].
	•	SD rs2, offset(rs1) (XLEN = 64)
Store XLEN bits.

Alignment behavior for misaligned accesses is implementation-defined.

3.4 CSR access
	•	CSRRW rd, csr, rs1
temp = CSR[csr]; CSR[csr] = x[rs1]; x[rd] = temp.
	•	CSRRS rd, csr, rs1
temp = CSR[csr]; CSR[csr] = temp | x[rs1]; x[rd] = temp.
	•	CSRRC rd, csr, rs1
temp = CSR[csr]; CSR[csr] = temp & ~x[rs1]; x[rd] = temp.

CSR names and layout are implementation-defined. Typical CSRs include:
	•	segment base and limit registers.
	•	PoX configuration.
	•	random seed for PRNG.
	•	flags controlling BEM-specific instructions.

	4.	Bitwise and STATE operations

4.1 Word-level bitwise

The base integer instructions already provide word-level bitwise operations:
	•	AND, OR, XOR, shifts.

These are used for hash functions, H_ctx, and generic bit manipulation.

4.2 Vector-level (optional) and STATE access

If vector registers v* are exposed:
	•	VAND vd, vs1, vs2
vd = vs1 bitwise_and vs2 (W bits).
	•	VOR, VXOR, VNOT
Analogous bitwise OR, XOR, NOT.
	•	VSHL, VSHR, VROT
Shift or rotate vd by an immediate or scalar amount.
	•	VPOPCNT rd, vs1
x[rd] = popcount(vs1) over all W bits.
	•	VPAR rd, vs1
x[rd] = popcount(vs1) mod 2.

Interaction with STATE:
	•	LSTATE vd, rs1
k = x[rs1] (bit index); vd = STATE[k].
	•	SSTATE vs1, rs1, vmask
k = x[rs1];
STATE[k] = (STATE[k] & ~vmask) | (vs1 & vmask).

If vector registers are not implemented, LSTATE and SSTATE are abstract operations implemented by the BIT-ALU using internal temporaries. Semantics remain:
	•	STATE is the only storage for bit-sliced global state.
	•	All expert circuits operate on STATE via R_spec and W_spec.

	5.	BEM-specific instructions

All BEM-specific instructions operate over STATE and SHARED, and are implemented by fixed-cost micro-sequences on the core plus co-processor calls where noted.

5.1 Observation

BEM_OBS obs_ptr
	•	Input:
	•	obs_ptr (in rs1 or rd): pointer to a struct in SHARED.
	•	Semantics:
	•	Compute obs_t = O(s_t, M_t) = (task_id_t, context_hash_t, local_bits_t, feature_bits_t).
	•	Write to memory at obs_ptr in a fixed layout:
	•	task_id_t (integer).
	•	context_hash_t (integer).
	•	local_bits_t (packed bits).
	•	feature_bits_t (packed bits, length d).

O is deterministic and integer-only. The cost of BEM_OBS is bounded by a constant C_obs.

5.2 Routing and ANN

BEM_ROUTE route_ptr
	•	route_ptr: pointer to a routing struct in SHARED.

Input struct (written partly by BEM_OBS):
	•	task_id_t.
	•	context_hash_t.
	•	feature_bits_t.
	•	k_large and other routing parameters.

Output struct:
	•	candidate_count <= k_large.
	•	candidate_ids[0..candidate_count-1]: expert slot indices or identifiers.
	•	optional scores score_i,tau for each candidate.

Semantics:
	1.	Load (task_id_t, context_hash_t, feature_bits_t) from route_ptr.
	2.	Compute q_t = F_id(task_id_t, context_hash_t), using only integer and bitwise ops.
	3.	Invoke COPROC CPID_ANN with op_id = ANN_QUERY:
	•	arguments: (q_t, feature_bits_t, k_large, config(task_id_t)).
	•	results: candidate list C_large.
	4.	Optionally compute routing scores score_i,tau based on bandit stats (visits_i,tau, wins_i,tau) and write them to route_ptr.

The cost of BEM_ROUTE is bounded by C_ANN + C_route, derived from D_max, M_max, k_large, and bandit arithmetic.

Selection of a single expert or a small set of experts (k_small) is done either inside BEM_ROUTE or by a short follow-up code sequence that reads the candidate list and scores.

5.3 Expert batch execution

BEM_EXPERT_BATCH cfg_ptr
	•	cfg_ptr: pointer to a struct in SHARED describing the chosen expert(s) and lane assignment.

Input struct:
	•	mode: all_lanes | grouped | per_lane.
	•	chosen expert slot indexes or identifiers.
	•	optional lane masks for grouped or per-lane mode.

Semantics:
	•	For each expert i actually executed at this step:
	•	Read STATE and SHARED via R_spec_i, producing x in {0,1}^{n_i}.
	•	Evaluate C_rep_i(x) bit-sliced over W lanes, using only bitwise and popcount operations.
	•	Write back y via W_spec_i into STATE and SHARED, respecting lane masks.

The total cost of BEM_EXPERT_BATCH is bounded by C_expert_batch, which depends on N_in_max, N_out_max, circuit form, and k_small, but not on episode length or history.

5.4 Statistics update

BEM_STATS_UPDATE stats_ptr
	•	stats_ptr: pointer to per-expert and per-task stats in SHARED.

Semantics:
	•	For each expert i used at step t and current task tau = task_id_t, with reward r_t:
	•	visits_i,tau += 1
	•	visits_total_i += 1
	•	wins_i,tau += r_t
	•	wins_total_i += r_t
	•	N_tau (total visits for task tau) += 1

BEM_STATS_UPDATE is algebraically simple and runs in bounded time C_stats. It may be executed immediately after BEM_EXPERT_BATCH or batched in the mid path.

5.5 Log-domain weight update

BEM_LEARN learn_ptr
	•	learn_ptr: pointer to structures holding z_i, L_i,tau, S_i,tau, hyperparameters, and scheduling flags.

Semantics (conceptual):

For each expert i and task tau marked for update:
	1.	hat_l_i,tau = L_i,tau / max(1, visits_i,tau)
	2.	hat_l_i,tau_clipped = clamp(hat_l_i,tau, -L_max, L_max)
	3.	eta_i,tau = c_eta / sqrt(S_i,tau + epsilon_eta)
	4.	z_i’ = z_i - eta_i,tau * hat_l_i,tau_clipped
	5.	If abs(z_i’ - z_i) > Delta_z_max, clip z_i’ to z_i +/- Delta_z_max
	6.	Optionally renormalize z_i across experts.

All arithmetic is fixed-point integer, implemented via ADD, MUL (if available), shifts, and table-based approximations. BEM_LEARN is a mid-path instruction and may complete its work over multiple calls.

5.6 Logging and hash chain

BEM_LOG log_ptr, size
	•	log_ptr: pointer to a serialized log entry e_k in SHARED.
	•	size: length in bytes.

Semantics:
	1.	Append e_k to TRACE / LOG at the current log position.
	2.	Compute H_{k+1} = Hash(H_k || e_k) via CPID_HASH (HASH_BLOCK).
	3.	Store H_{k+1} in a protected location (CSR or SHARED slot reserved for the chain head).

The cost of BEM_LOG is bounded and its usage frequency is controlled by software.

5.7 Merkle updates

BEM_MERKLE_UPDATE root_ptr, range_ptr
	•	root_ptr: pointer to stored Merkle roots for relevant segments.
	•	range_ptr: pointer to a description of changed blocks (for example indices and hashes).

Semantics:
	•	Invoke CPID_HASH operations to update Merkle tree nodes for the changed leaves.
	•	Update the relevant root hashes at root_ptr.

Used on the slow path when structural changes or state migrations occur.

5.8 Fast-path macro (optional)

BEM_STEP_FAST step_ptr
	•	step_ptr: pointer to a struct describing locations of obs_ptr, route_ptr, cfg_ptr, stats_ptr in SHARED and mode flags.

Semantics:
	•	Execute the fast step sequence:
	1.	BEM_OBS obs_ptr
	2.	BEM_ROUTE route_ptr
	3.	BEM_EXPERT_BATCH cfg_ptr
	4.	BEM_STATS_UPDATE stats_ptr

in a single instruction. Semantics are exactly the same as the corresponding sequence. The cost is bounded by C_step = C_obs + C_ANN + C_route + C_expert_batch + C_stats.
	6.	Co-processor interface

6.1 COPROC instruction

COPROC cp_id, op_id, arg_ptr, res_ptr
	•	cp_id: co-processor identifier (for example CPID_ANN, CPID_SAT, CPID_HASH, CPID_WORLD).
	•	op_id: operation code local to the given co-processor.
	•	arg_ptr: pointer to argument struct in SHARED.
	•	res_ptr: pointer to result struct in SHARED.

Execution model:
	•	For fast and mid-path safe operations, COPROC is synchronous:
	•	the instruction completes before the next instruction, with bounded cost.
	•	For slow-path operations, COPROC may:
	•	schedule an asynchronous job and return a handle, or
	•	run in a separate hardware resource pool, with completion indicated via SHARED.

6.2 ANN co-processor

cp_id = CPID_ANN.

Supported operation:
	•	op_id = ANN_QUERY

Arguments at arg_ptr:
	•	q_t: query key (integer).
	•	f_t: feature_bits_t (packed bits, length d).
	•	k: maximum candidate count (k <= k_large).
	•	config: task-specific configuration.

Results at res_ptr:
	•	candidate_count.
	•	candidate_ids[0..candidate_count-1]: slot indices or identifiers.

Constraints:
	•	Search depth <= D_max.
	•	Neighbor degree <= M_max.
	•	Worst-case complexity bounded and independent of episode length and history.

Implementation may use graph-based ANN over Hamming distances of Gray-coded identifiers, bucketed lists, or other structures consistent with the bounds.

6.3 SAT / Hoare co-processor

cp_id = CPID_SAT.

Representative operations:
	•	op_id = SAT_CHECK
	•	Args: baseCNF_id, deltaCNF_id.
	•	Result: result in {SAT, UNSAT, UNKNOWN}, optional unsat_core_id.
	•	op_id = PROOF_CHECK
	•	Check external proofs against stored CNF.
	•	op_id = HOARE_CHECK
	•	Check CFG annotations (pre/post conditions, invariants).
	•	op_id = CEGIS
	•	Args: phi_id, hyp_class_id.
	•	Result: candidate_id or cex_id.

These operations are used on the slow path to validate patches, synthesize experts and invariants, and extract counterexamples.

6.4 HASH / ECC co-processor

cp_id = CPID_HASH.

Representative operations:
	•	op_id = HASH_BLOCK
	•	Compute hash for a memory range.
	•	op_id = MERKLE_NODE
	•	Compute parent hash from child hashes.
	•	op_id = ECC_ENCODE / ECC_DECODE
	•	Apply ECC to identifiers or memory blocks.

Used by BEM_LOG, BEM_MERKLE_UPDATE, and structural operations.

6.5 WORLD co-processor (optional)

cp_id = CPID_WORLD.

Representative operations (informative, out of normative core):
	•	WORLD_RETRIEVE
	•	WORLD_CALL_TOOL
	•	WORLD_STREAM_READ / WORLD_STREAM_WRITE

These operations interact with external resources or tools, and may drive synthetic tasks or RAG-like retrieval. Their detailed semantics are outside the v0.0.1 core; they are constrained only by timing and isolation requirements so as not to violate fast-path guarantees.
	7.	Fast, mid, and slow path

7.1 Fast path

The fast path implements the per-step loop. Conceptually:

Loop over t:
	1.	obs_t = O(s_t, M_t).
	2.	(q_t, f_t) = (F_id(task_id_t, context_hash_t), feature_bits_t).
	3.	C_large = ANN_QUERY(q_t, f_t, k_large, config(task_id_t)).
	4.	Select expert(s) using bandit statistics and z_i.
	5.	Evaluate expert(s) over STATE and SHARED (bit-sliced).
	6.	Perform minimal stats update.

In ISA form, this is either:
	•	One BEM_STEP_FAST step_ptr per step, or
	•	The explicit sequence:
BEM_OBS obs_ptr
BEM_ROUTE route_ptr
BEM_EXPERT_BATCH cfg_ptr
BEM_STATS_UPDATE stats_ptr

The fast-path cost satisfies:

Cost_step <= C_obs + C_ANN + C_route + C_expert_batch + C_stats

where each constant is fixed by configuration and implementation, and independent of episode length and total history.

No SAT_CHECK, CEGIS, or heavy COPROC operations are allowed on the fast path.

7.2 Mid path

Mid path operations run periodically, for example at episode boundaries or every M steps. They include:
	•	BEM_LEARN for z_i, L_i,tau, S_i,tau.
	•	Aggregation and smoothing of statistics for bandits.
	•	Lightweight template extraction from recent logs.

Mid path must not violate fast-path timing guarantees. It may be implemented as short bursts interleaved with fast-path steps or as work on separate cores synchronizing via SHARED.

7.3 Slow path

Slow path operations run asynchronously or on dedicated resources. They include:
	•	Structural proposal generation (expert split, merge, macro creation, superoptimization).
	•	Verification condition construction.
	•	SAT_CHECK, HOARE_CHECK, PROOF_CHECK, CEGIS.
	•	ANN index maintenance and rebalancing.
	•	PoX scoring and scheduling.
	•	Merkle tree updates and integrity checks.

Slow path must respect resource partitioning and scheduling so that fast-path timing remains bounded.
	8.	Learning and bandits (machine-level summary)

Algorithmic behavior of bandits and learning is expressed via:
	•	BEM_STATS_UPDATE (collect loss / reward statistics).
	•	BEM_LEARN (update log-weights and learning rates).
	•	BEM_ROUTE (use stats and z_i for routing decisions).

The ISA does not fix a particular bandit algorithm. The reference intent is:
	•	Per task tau, bandit stats (visits_i,tau, wins_i,tau, L_i,tau, S_i,tau) are maintained.
	•	Routing uses scores such as:
mean_i,tau = wins_i,tau / max(1, visits_i,tau)
bonus_i,tau = c_explore * sqrt( log(max(1, N_tau)) / max(1, visits_i,tau) )
score_i,tau = mean_i,tau + bonus_i,tau
	•	Log-domain weights z_i are updated using fixed-point mirror-descent style rules.

All arithmetic is realized via the base integer ISA; BEM_LEARN and BEM_ROUTE only require additions, multiplications or approximate square roots, and clamps.
	9.	Structural updates and verification hooks

Structural updates are implemented in software using:
	•	COPROC CPID_SAT and CPID_HASH.
	•	Regular integer and memory instructions.
	•	BEM_LOG and BEM_MERKLE_UPDATE for audit and integrity.

9.1 Patch representation

A patch delta is a structured object stored in SHARED and PROOF, describing:
	•	changes to EXPERT (for example add, split, merge, deprecate, replace C_rep_i),
	•	changes to CFG / CODE,
	•	changes to ANN index,
	•	associated CNF and invariant updates.

9.2 Patch pipeline

The patch pipeline is:
	1.	Propose a patch delta (for example from template extraction, CEGIS, or operator input).
	2.	Build verification condition VC(delta) as CNF in PROOF.
	3.	Invoke COPROC CPID_SAT, op_id = SAT_CHECK, on baseCNF and deltaCNF.
	4.	If result is UNSAT:
	•	Optionally compute PoX score(delta) using evaluation runs and analysis.
	•	If PoX acceptance criteria satisfied:
	•	Apply patch: update EXPERT, CFG / CODE, ANN index, and related metadata.
	•	Update Merkle roots via CPID_HASH and BEM_MERKLE_UPDATE.
	•	Log the patch and its proof via BEM_LOG.
	5.	If result is SAT or UNKNOWN:
	•	Do not apply patch.
	•	Optionally log counterexamples or proof failures.

The ISA only requires COPROC, BEM_LOG, BEM_MERKLE_UPDATE, and regular load/store; patch semantics are specified at the abstract machine level.
	10.	Integrity and audit

Integrity and audit are based on:
	•	A hash chain over LOG.
	•	Merkle trees over structural segments (EXPERT, CFG / CODE, WORK).
	•	Strict patch application policy.

10.1 Log chain

LOG stores entries e_k. A hash chain is defined:
	•	H_0 = fixed constant.
	•	For k >= 0: H_{k+1} = Hash(H_k || e_k).

BEM_LOG maintains H_k using CPID_HASH. H_k is stored in protected state (CSR or reserved SHARED) and optionally exported externally.

Any modification to past entries will be detectable as a mismatch in H_k.

10.2 Merkle trees

Merkle trees can cover:
	•	EXPERT segment.
	•	CFG / CODE.
	•	WORK configuration.

Leaves store hash(block). Internal nodes store hash(children). BEM_MERKLE_UPDATE plus CPID_HASH maintain consistency when blocks change. Root hashes are stored in protected locations.

10.3 Patch acceptance policy

A patch delta is allowed to change structural state only if:
	1.	SAT_CHECK(VC(delta)) returns UNSAT (verification condition holds).
	2.	Optional PoX condition holds:
	•	score(delta) >= D (threshold stored in WORK).

On acceptance:
	•	The patch is applied to memory.
	•	Merkle roots are updated.
	•	A patch log entry is emitted via BEM_LOG containing:
	•	patch identifier,
	•	structural changes,
	•	identifiers of verification artifacts,
	•	PoX components and score,
	•	parent hash pointer.

This yields an auditable chain of structural changes over time.
	11.	Informative note: behavioral intent

This section is informative and does not change normative semantics.
	•	The fast path implements a contextual bandit over experts, with safety and cost encoded into rewards.
	•	The mid path adjusts log-weights and statistics so that regret against a reference policy class remains sublinear in horizon.
	•	The slow path searches over structural configurations (experts, macros, invariants) and uses verification plus PoX to decide which structural changes to admit.
	•	The ISA is designed so that:
	•	fast path is a small, analyzable loop over BEM_OBS, BEM_ROUTE, BEM_EXPERT_BATCH, BEM_STATS_UPDATE,
	•	mid and slow paths are realized by the same core+co-processor infrastructure without affecting fast-path bounds.

