Title: SDPR Spec v0.0.1 (Linux, Rust, ASCII)
	0.	Status
0.1 This document defines the frozen v0.0.1 specification for SDPR (ShmDataPlane Runtime): a shared-memory, single-host data-plane runtime that composes verified MicroVM programs over baton-style descriptors transported by SPSC futex rings, with low-overhead debug capture via dedicated SPSC debug rings.
0.2 Target: Linux userspace, single host, multi-process, high-throughput/low-latency data-plane pipelines.
0.3 Normative keywords: MUST, MUST NOT, SHOULD, MAY.
0.4 Version policy (fixed): Implementations claiming compliance with SDPR v0.0.1 MUST implement exactly the ABIs and semantics in this document. No forward-compat behavior is defined in v0.0.1.
	1.	Goals
1.1 Provide a data-plane function runtime where each stage executes a MicroVM program over a packet-like buffer object and returns an XDP-like action controlling routing.
1.2 Provide baton-style ownership transfer of buffer objects via fixed-size descriptors.
1.3 Provide zero-copy or near-zero-copy pipeline composition using shared-memory arenas and SPSC rings.
1.4 Provide strict, testable, frozen ABIs for:
1.4.1 ShmRing (SPSC futex ring) used to transport descriptors and debug events.
1.4.2 Arena object layout used to store packet bytes and scratch bytes.
1.4.3 Descriptor layout used as the baton.
1.4.4 DebugEvent layout used for low-overhead observability.
1.4.5 MicroVM bytecode format, verifier rules, and execution contract.
1.5 Provide deterministic and auditable behavior: each stage is a pure function of its input object bytes, its program bytecode, and configured read-only tables.
	2.	Non-goals
2.1 Cross-host transport.
2.2 MPMC rings, fairness guarantees, or starvation guarantees.
2.3 Crash-recovery or automatic attachment stealing.
2.4 Kernel-bypass waiting (futex requires syscalls when sleeping/waking).
2.5 Strong security isolation against a malicious peer beyond:
2.5.1 OS-level fd/mmap permissions,
2.5.2 MicroVM verifier and hardening,
2.5.3 mandatory runtime safety checks defined here.
2.6 Multi-ingress blocking wait (waiting on multiple input rings simultaneously) in v0.0.1.
2.7 Lossless logging. Debug capture is best-effort and MAY drop under pressure.
	3.	SDPR model
3.1 Objects
3.1.1 Arena: a shared-memory region partitioned into fixed-size objects.
3.1.2 Object: a fixed-size buffer containing:
(a) an object header,
(b) data bytes (wire packet bytes),
(c) scratch bytes (internal-only bytes).
3.1.3 Descriptor (baton): a fixed-size record that identifies one object and carries minimal metadata. Descriptors are transported on ShmRings.
3.1.4 Action: a 32-bit return code produced by MicroVM, interpreted by the router to decide the next hop.
3.1.5 DebugEvent: a fixed-size record emitted by stages to dedicated debug rings.

3.2 Ownership and transfer
3.2.1 At any time, exactly one stage owns a descriptor and its referenced object.
3.2.2 Ownership is transferred only by pushing the descriptor into exactly one output ShmRing, or by returning it to the reclaim path (DROP/TX completion).
3.2.3 After transfer, the previous owner MUST NOT access the object bytes referenced by that descriptor.
3.2.4 v0.0.1 provides no refcounting. Double-free and stale-descriptor use are detected best-effort via cookie checks.

3.3 Pipeline
3.3.1 A stage is a worker with:
(a) exactly one input ShmRing (consumer role),
(b) zero or more output ShmRings (producer roles),
(c) one loaded MicroVM program table (prog_id -> program),
(d) an action router mapping REDIRECT(qid) to an output ShmRing.
3.3.2 A pipeline is a DAG of stages connected by ShmRings.
3.3.3 In v0.0.1, a stage MUST block on at most one input ring at a time (single-ingress wait).
	4.	Platform requirements
4.1 OS and futex
4.1.1 Linux kernel providing futex(2) for shared futex words.
4.1.2 Inter-process use MUST use shared futex operations (MUST NOT use FUTEX_PRIVATE variants).
4.1.3 ShmRing MUST use FUTEX_WAIT and FUTEX_WAKE only in v0.0.1.
4.2 Architecture
4.2.1 64-bit only: x86_64 or aarch64.
4.2.2 All atomics used by ShmRing and object headers MUST be lock-free on the target. Implementations MUST fail create/attach if this requirement is not met.
4.3 Endianness
4.3.1 All integer fields in all SDPR ABIs are little-endian.
4.4 Cache line model
4.4.1 Cache line size is treated as 64 bytes for ABI layout and padding rules. Implementations MUST follow fixed offsets and cache-line separation requirements regardless of host microarchitecture.
	5.	Files, fds, and mappings
5.1 SDPR instance resources
5.1.1 An SDPR instance consists of:
(a) one Arena fd (memfd or shm_open),
(b) one or more ShmRing fds (one per pipeline edge),
(c) zero or more DebugRing fds (one per stage, recommended),
(d) optional read-only Table fds (route tables, program tables),
(e) optional DumpArena fd (debug-only, optional in v0.0.1).
5.2 Sharing
5.2.1 The control plane MUST share fds to worker processes via fork+inherit, unix domain socket SCM_RIGHTS, or equivalent.
5.3 Mapping rules
5.3.1 Arena and ShmRings MUST be mapped with mmap(PROT_READ|PROT_WRITE, MAP_SHARED).
5.3.2 Read-only tables (if any) SHOULD be mapped PROT_READ.
5.3.3 DumpArena (if present) MUST be mapped PROT_READ|PROT_WRITE, MAP_SHARED.
	6.	SDPR.Arena v0.0.1 (fixed object arena)
6.1 Arena creation
6.1.1 Creator obtains an fd via memfd_create or shm_open.
6.1.2 Creator MUST size the object via ftruncate(fd, arena_total_size).
6.1.3 Arena mapping size MUST equal arena_total_size.
6.2 Arena parameters (fixed constraints)
6.2.1 object_size MUST be a power of two and MUST be in [2048, 1<<20] bytes in v0.0.1.
6.2.2 object_count MUST be >= 1.
6.2.3 arena_total_size MUST equal object_size * object_count.
6.2.4 object_size MUST be a multiple of 64.
6.3 Arena object addressing
6.3.1 An object is addressed by obj_index in [0, object_count).
6.3.2 obj_off = obj_index * object_size.
6.3.3 The object base address in memory is arena_base + obj_off.
6.3.4 Descriptors MUST use obj_off (not obj_index) in v0.0.1.
6.4 Prefault and paging (recommended, non-normative)
6.4.1 Implementation SHOULD touch all mapped pages once at init to reduce page-fault latency in hot paths.
6.4.2 Implementation MAY mlock if permitted.
6.4.3 Implementation MAY apply madvise to tune paging behavior; such tuning MUST NOT change ABIs.
	7.	SDPR.Object v0.0.1 (fixed in-object layout)
7.1 Object header (fixed size)
7.1.1 Each object begins with a 64-byte header at offset 0.
7.1.2 Header fields (offsets from object base):
0x00 magic: u32 = 0x52445053 (“SDPR” little-endian)
0x04 obj_version_major: u16 = 0
0x06 obj_version_minor: u16 = 1
0x08 cookie: u64 (generation identifier)
0x10 data_len: u32 (wire-visible data length)
0x14 scratch_len: u32 (internal scratch length)
0x18 flags: u32 (bitfield, see 7.2)
0x1C reserved0: u32 (MUST be 0)
0x20 reserved1: [u8; 32] (MUST be 0)
7.1.3 All reserved bytes MUST be zeroed by the writer and MUST be validated as zero by consumers that validate objects.
7.1.4 Writers MUST write header fields with plain stores before publishing the descriptor transfer (descriptor publish is the synchronization point).
7.1.5 Readers MUST treat cookie/data_len/scratch_len as untrusted until validated.
7.2 Object flags (u32)
7.2.1 bit 0 INTERNAL_SCRATCH_PRESENT (hint only)
7.2.2 bits 1..31 reserved (MUST be 0)
7.3 Object byte regions
7.3.1 data region begins at offset 64 and spans data_len bytes.
7.3.2 scratch region begins at offset 64 + data_len and spans scratch_len bytes.
7.3.3 Remaining bytes up to object_size are unspecified and MUST be ignored.
7.3.4 data_len + scratch_len MUST satisfy:
7.3.4.1 data_len <= object_size - 64
7.3.4.2 scratch_len <= object_size - 64 - data_len
7.4 TX requirement (mandatory)
7.4.1 Before any TX action, implementations MUST set scratch_len to 0 and MUST ensure bytes beyond data_len are not transmitted.
	8.	SDPR.Descriptor v0.0.1 (baton ABI)
8.1 Descriptor size and alignment
8.1.1 Descriptor size is fixed: 64 bytes.
8.1.2 Descriptor MUST be aligned to 8 bytes within ShmRing slot payload.
8.2 Descriptor fields (little-endian, fixed offsets)
8.2.1 0x00 obj_off: u64 (arena offset of object base)
8.2.2 0x08 obj_cap: u32 (MUST equal object_size for the arena)
8.2.3 0x0C data_len: u32 (wire length)
8.2.4 0x10 scratch_len: u32 (internal scratch length)
8.2.5 0x14 route_hint: u16 (application-defined)
8.2.6 0x16 prog_id: u16 (MicroVM program id)
8.2.7 0x18 cookie: u64 (generation identifier)
8.2.8 0x20 trace_id: u64 (diagnostic only; MAY be 0)
8.2.9 0x28 reserved0: [u8; 24] (MUST be 0)
8.3 Descriptor validity rules (mandatory)
8.3.1 obj_off MUST be a multiple of obj_cap.
8.3.2 obj_off + obj_cap MUST be within arena_total_size.
8.3.3 obj_cap MUST match the configured arena object_size.
8.3.4 data_len and scratch_len MUST satisfy the constraints in 7.3.4 when applied to obj_cap.
8.3.5 cookie MUST match the object header cookie; mismatch MUST be treated as StaleDescriptor.
8.3.6 reserved0 MUST be zero; non-zero MUST be rejected as InvalidDescriptor.
	9.	SDPR.Action v0.0.1 (XDP-like return encoding)
9.1 Action is a 32-bit value returned by MicroVM.
9.2 Encoding
9.2.1 bits 0..7 action_class:
0x00 PASS
0x01 DROP
0x02 REDIRECT
0x03 TX
0x04 ABORT
0x05 TAILCALL
0x06 RESERVED (MUST NOT be returned in v0.0.1)
9.2.2 bits 8..31 action_arg (meaning depends on action_class)
9.3 Semantics
9.3.1 PASS: deliver descriptor to the stage default output ring.
9.3.2 DROP: return descriptor to reclaim path; object becomes free for reuse.
9.3.3 REDIRECT(qid): qid = action_arg; deliver descriptor to the configured output ring for qid.
9.3.4 TX: deliver descriptor to the TX subsystem; scratch MUST be removed (7.4.1).
9.3.5 ABORT(code): code = action_arg; stage MUST record code and MUST return descriptor to reclaim or quarantine (implementation-defined), but MUST NOT TX.
9.3.6 TAILCALL(prog_id): prog_id = action_arg; stage MUST run the referenced program as a tail call with the same ctx, subject to verifier-enforced tailcall depth limit.
	10.	SDPR.IPC.ShmRing v0.0.1 (SPSC futex ring)
10.1 Status
10.1.1 ShmRing v0.0.1 is a frozen ABI derived from the ShmSpscFutexQueue v0.0.1 model.
10.2 Slot payload usage in SDPR
10.2.1 DataRing: ShmRing transporting SDPR.Descriptor. Slot payload MUST contain exactly one 64-byte SDPR.Descriptor and slot_header.len MUST equal 64.
10.2.2 DebugRing: ShmRing transporting SDPR.DebugEvent (section 11). Slot payload MUST contain exactly one DebugEvent and slot_header.len MUST equal debug_event_size.
10.3 ABI, ordering, futex protocol, attach, close, shutdown
10.3.1 ShmRing MUST implement exactly the ABI layout, invariants, memory ordering contract, futex protocol, attach validation, close/shutdown semantics, and performance rules as defined by ShmSpscFutexQueue Spec v0.0.1, with the SDPR-specific slot payload requirements in 10.2.
10.3.2 In particular, ShmRing MUST enforce:
(a) fixed header_size=0x180, ring_offset=0x180, fixed cache-line separation,
(b) Acquire/Release on head/tail as required,
(c) expected-value futex waits with recheck loops and fixed-budget timeout semantics,
(d) wake-on-transition rules, and wake-all for shutdown/close.
10.4 Slot size constraints for SDPR
10.4.1 For DataRing, slot_size MUST be >= 8 + 64 = 72 and slot_size MUST be a multiple of 8.
10.4.2 For DebugRing, slot_size MUST be >= 8 + debug_event_size and slot_size MUST be a multiple of 8.
	11.	SDPR.DebugEvent v0.0.1 (debug ring event ABI)
11.1 Status
11.1.1 DebugEvent is a fixed binary record intended for low-overhead, best-effort observability.
11.1.2 DebugEvent capture MUST NOT block the data plane. If the debug ring is full, events MUST be dropped.
11.2 DebugRing topology (mandatory if DebugRing is used)
11.2.1 DebugRing MUST be SPSC: exactly one stage produces into its DebugRing, exactly one collector consumes from it.
11.2.2 Each stage SHOULD have its own DebugRing (per-stage SPSC). Sharing a DebugRing among multiple stages is non-compliant.
11.3 DebugEvent size and alignment (fixed)
11.3.1 debug_event_size is fixed: 64 bytes.
11.3.2 DebugEvent MUST be aligned to 8 bytes in the ring payload.
11.4 DebugEvent layout (little-endian, fixed offsets)
11.4.1 0x00 ts: u64 (monotonic ticks; unit is implementation-defined)
11.4.2 0x08 trace_id: u64 (copied from descriptor; MAY be 0)
11.4.3 0x10 stage_id: u16
11.4.4 0x12 prog_id: u16
11.4.5 0x14 action: u32 (SDPR.Action returned or synthesized)
11.4.6 0x18 reason: u32 (DebugReason enumeration; 0 means NONE)
11.4.7 0x1C aux: u32 (auxiliary value; meaning depends on reason; MAY be 0)
11.4.8 0x20 obj_off: u64
11.4.9 0x28 cookie: u64
11.4.10 0x30 data_len: u32
11.4.11 0x34 scratch_len: u32
11.4.12 0x38 reserved0: [u8; 8] (MUST be 0)
11.5 DebugReason enumeration (required values)
11.5.1 0 NONE
11.5.2 1 STAGE_IN (descriptor accepted)
11.5.3 2 STAGE_OUT (descriptor emitted)
11.5.4 3 VM_ABORT (MicroVM returned ABORT)
11.5.5 4 VM_TRAP (runtime trap: bounds/align/stack/helper/illegal/timeout)
11.5.6 5 STALE_DESC (cookie mismatch)
11.5.7 6 INVALID_DESC (descriptor validation failed)
11.5.8 7 ROUTE_MISS (unknown qid)
11.5.9 8 CORRUPT_RING (CorruptSlot/CorruptIndices)
11.5.10 9 DROP (dropped by policy)
11.5.11 10 TX (sent to TX path)
11.6 Sampling rules (normative behavior)
11.6.1 Debug capture is best-effort; it MUST NOT cause blocking waits on the debug ring.
11.6.2 On debug ring Full, the stage MUST drop the event and MUST increment debug_drop_count (section 15.2).
11.6.3 Implementations SHOULD support deterministic sampling using trace_id bitmasking:
emit if (trace_id & sample_mask) == 0.
11.6.4 Events with reason in {VM_ABORT, VM_TRAP, STALE_DESC, INVALID_DESC, CORRUPT_RING, ROUTE_MISS} SHOULD be emitted regardless of sampling (best-effort).
11.6.5 Sampling and trigger policies MUST NOT change data-plane semantics.
	12.	SDPR.DumpArena v0.0.1 (optional deep-dump facility)
12.1 Status
12.1.1 DumpArena is OPTIONAL in v0.0.1. If implemented, it MUST follow this ABI.
12.1.2 DumpArena is intended for deep diagnostics (register snapshots, offending offsets) and MUST be written only on triggers (e.g., VM_TRAP).
12.1.3 DumpArena writes MUST be non-blocking with bounded work per trigger.
12.2 Topology
12.2.1 Each stage that supports dumps MUST have an associated DumpArena region. The region MAY be:
(a) a separate fd per stage, or
(b) a single fd partitioned into fixed per-stage segments.
12.2.2 A stage MUST only write within its own segment.
12.3 Record format (fixed)
12.3.1 dump_record_size is fixed: 512 bytes.
12.3.2 dump_record_size MUST be aligned to 64.
12.3.3 DumpArena is a ring of dump_record_count records per stage segment. Indexing is modulo dump_record_count.
12.3.4 Overwrite is permitted. DumpArena is lossy.
12.4 DumpRecord layout (little-endian, fixed offsets)
12.4.1 0x00 magic: u32 = 0x504D5544 (“DUMP” little-endian)
12.4.2 0x04 version_major: u16 = 0
12.4.3 0x06 version_minor: u16 = 1
12.4.4 0x08 ts: u64
12.4.5 0x10 stage_id: u16
12.4.6 0x12 prog_id: u16
12.4.7 0x14 reason: u32 (DebugReason; MUST be VM_TRAP or VM_ABORT in v0.0.1 dumps)
12.4.8 0x18 vm_code: u32 (MicroVM abort/trap code)
12.4.9 0x1C pc: u32 (program counter at trap; insn index)
12.4.10 0x20 trace_id: u64
12.4.11 0x28 obj_off: u64
12.4.12 0x30 cookie: u64
12.4.13 0x38 desc: [u8; 64] (raw SDPR.Descriptor bytes)
12.4.14 0x78 regs: [u64; 16] (r0..r15 snapshot)
12.4.15 0xF8 fault_off: i64 (offending offset for bounds/align if applicable; else 0)
12.4.16 0x100 fault_addr: u64 (computed address if applicable; else 0)
12.4.17 0x108 snippet_len: u32 (0..=128)
12.4.18 0x10C reserved0: u32 (MUST be 0)
12.4.19 0x110 snippet: [u8; 128] (best-effort bytes from data region head; content is optional)
12.4.20 0x190 reserved1: [u8; 112] (MUST be 0)
12.5 Dump write protocol (normative)
12.5.1 On trigger, the stage MUST write exactly one DumpRecord using plain stores, then publish it by incrementing a per-stage dump_write_index (in private memory).
12.5.2 The stage MUST NOT block. If it cannot dump within a bounded budget, it MUST skip dumping and increment dump_drop_count.
12.5.3 Dumping MUST NOT read beyond validated data_len/scratch_len bounds.
	13.	SDPR.VM.MicroVM v0.0.1
13.1 Execution contract
13.1.1 MicroVM executes over a context (ctx) derived from a descriptor and its referenced object.
13.1.2 MicroVM MUST NOT access memory outside:
(a) the object data region [obj_base+64, obj_base+64+data_len),
(b) the object scratch region [obj_base+64+data_len, obj_base+64+data_len+scratch_len),
(c) VM private stacks and registers.
13.1.3 MicroVM MUST return an SDPR.Action (section 9).
13.1.4 MicroVM runtime MUST enforce bounds and alignment checks even if the verifier is bypassed; violations MUST trap to ABORT(VM_BOUNDS or VM_ALIGN).
13.2 Bytecode container
13.2.1 A program is a byte string containing:
(a) ProgramHeader (fixed 32 bytes),
(b) Instruction array (N * 8 bytes),
(c) Optional constant pool (implementation-defined, verifier-visible).
13.2.2 ProgramHeader layout:
0x00 magic: u32 = 0x564D5053 (“SPMV” little-endian)
0x04 vm_version_major: u16 = 0
0x06 vm_version_minor: u16 = 1
0x08 prog_id: u16
0x0A flags: u16 (reserved, MUST be 0)
0x0C insn_count: u32
0x10 entry_pc: u32 (MUST be 0 in v0.0.1)
0x14 reserved: [u8; 12] (MUST be 0)
13.2.3 Instruction encoding (fixed 8 bytes, little-endian)
byte 0 opcode
byte 1 dst_reg (0..15)
byte 2 src_reg (0..15)
byte 3 imm8
bytes 4..7 imm32 (signed)
13.2.4 Registers (fixed meaning on entry)
r0 return action
r1 ctx_base (object base address, read-only)
r2 data_len (u64)
r3 scratch_len (u64)
r4 obj_cap (u64)
13.2.5 Private stacks (fixed)
stack0 1024 bytes
stack1 1024 bytes
overflow/underflow MUST trap to ABORT(VM_STACK).
13.3 Mandatory instruction subset
13.3.1 ALU64 ADD,SUB,AND,OR,XOR,SHL,SHR,MUL (wrapping)
13.3.2 MOV MOV64 dst,src and MOVI64 dst,imm32 sign-extended
13.3.3 CMP/JMP JEQ,JNE,JLT,JLE,JGT,JGE with signed compare; relative pc offset in imm32 (insn units)
13.3.4 EXIT return r0
13.3.5 ADDRESS (single memory address formation op, mandatory)
ADDR dst, base_reg, off_reg, imm32, region
region in imm8: 0 DATA, 1 SCRATCH
Semantics:
base = ctx_base + 64
if DATA: region_base=base, region_len=data_len
if SCRATCH: region_base=base+data_len, region_len=scratch_len
off = (base_reg != 0 ? r[base_reg] : 0) + r[off_reg] + imm32
addr = region_base + off
Verifier MUST prove off range is within [0, region_len).
Runtime MUST check bounds at execution time; on violation trap to ABORT(VM_BOUNDS).
Runtime MUST check alignment for LD/ST sizes; on violation trap to ABORT(VM_ALIGN).
Note: Implementations MAY additionally mask addr within obj_cap as hardening, but MUST still enforce region_len and alignment.
13.3.6 LOAD/STORE (only via address)
LD8/16/32/64 dst, addr_reg
ST8/16/32/64 addr_reg, src
unaligned MUST trap to ABORT(VM_ALIGN).
13.3.7 HELPER (optional)
CALL id with inputs in r1..r5, return in r0
unknown helper MUST trap to ABORT(VM_HELPER).
13.4 Verifier (mandatory)
13.4.1 Verifier MUST reject any program that can:
(a) access outside allowed regions,
(b) execute unbounded loops,
(c) exceed insn_budget_max per invocation (implementation-defined but MUST be enforced at runtime),
(d) overflow stacks,
(e) use invalid encodings.
13.4.2 Pointer vs scalar typing
Verifier MUST track register types: SCALAR or PTR(region, offset_range).
ADDR produces PTR with provable offset_range.
ALU ops on PTR are restricted to bounded add/sub with scalars.
13.4.3 Control flow
Jump targets MUST be within [0, insn_count).
Loops MUST have statically bounded iteration count.
13.4.4 Tail calls
If TAILCALL is supported, verifier MUST enforce tailcall_depth_max = 32.
13.5 JIT/AOT (optional but normative if present)
13.5.1 Executable memory MUST be W^X.
13.5.2 If JIT hardening mode is present, it MUST NOT weaken verifier guarantees and MUST preserve traps and ABORT codes.
13.6 MicroVM ABORT codes (required)
VM_BOUNDS
VM_ALIGN
VM_STACK
VM_HELPER
VM_ILLEGAL
VM_TIMEOUT
	14.	Routing and stage behavior
14.1 Stage main loop (normative)
14.1.1 Pop a descriptor from input DataRing (blocking or non-blocking).
14.1.2 Validate descriptor per 8.3.1-8.3.6 (mandatory). Cookie check 8.3.5 is REQUIRED.
14.1.3 Construct ctx regs from descriptor.
14.1.4 Run MicroVM program selected by descriptor.prog_id (or stage default, implementation-defined).
14.1.5 Interpret action:
PASS push to default output ring
REDIRECT(qid) push to qid output ring
DROP return to reclaim
TX push to TX path (scratch removed)
ABORT record code, emit debug trigger, return to reclaim/quarantine, MUST NOT TX
TAILCALL run tail call (bounded), then continue
14.2 Router table
14.2.1 A stage MUST have a router mapping qid to an output ring.
14.2.2 Unknown qid MUST be treated as ABORT(VM_ILLEGAL) and SHOULD emit ROUTE_MISS.
14.3 Debug emission (normative)
14.3.1 Stages SHOULD emit STAGE_IN and STAGE_OUT under sampling policy.
14.3.2 Stages SHOULD emit 100% (best-effort) for:
VM_ABORT, VM_TRAP, STALE_DESC, INVALID_DESC, CORRUPT_RING, ROUTE_MISS.
14.3.3 If DumpArena is present, stages SHOULD write a DumpRecord on VM_TRAP (and MAY on VM_ABORT).
	15.	Reclaim and allocation
15.1 Free descriptor source
15.1.1 v0.0.1 requires a reclaim path that returns freed objects to a free pool.
15.1.2 Implementations MAY use a dedicated allocator/reclaimer stage or per-stage free pools; ownership invariants MUST hold.
15.2 Cookie (generation) rules (mandatory)
15.2.1 Each time an object is returned to the free pool, its cookie MUST be incremented (wrapping u64 permitted).
15.2.2 Newly allocated descriptors MUST carry the current object cookie.
15.2.3 Cookie mismatch MUST be treated as StaleDescriptor and MUST NOT allow object access.
	16.	Observability counters (normative minimum)
16.1 Per stage counters
16.1.1 in_count, out_count, drop_count, tx_count, abort_count
16.1.2 redirect_count[qid]
16.1.3 vm_trap_count by ABORT code
16.1.4 stale_desc_count, invalid_desc_count, route_miss_count
16.1.5 debug_emit_count, debug_drop_count
16.1.6 dump_write_count, dump_drop_count (if DumpArena supported)
16.1.7 futex_wait_ne, futex_wake_ne, futex_wait_nf, futex_wake_nf (from ShmRing)
	17.	Error model (required set)
17.1 InvalidMagic
17.2 UnsupportedVersion
17.3 InvalidLayout
17.4 InvalidDescriptor
17.5 StaleDescriptor
17.6 CorruptSlot
17.7 CorruptIndices
17.8 Full
17.9 Empty
17.10 Closed
17.11 Shutdown
17.12 Timeout
17.13 WouldBlock
17.14 AlreadyAttached
17.15 Syscall { op: SysOp, errno: i32 }
17.16 SysOp (required)
FutexWaitNe
FutexWakeNe
FutexWaitNf
FutexWakeNf
Mmap
Ftruncate
MemfdCreate
ShmOpen
CloseFd
	18.	Performance rules (mandatory for a compliant optimized build)
18.1 No syscalls on hot path when progress is possible.
18.2 No CAS and no fetch_add on ShmRing head/tail during push/pop.
18.3 head, tail, doorbells MUST be on distinct cache lines (ShmRing fixed offsets).
18.4 ShmRing futex_wake count MUST be 1 for normal operation; INT_MAX only for shutdown/close wake-all.
18.5 MicroVM MUST enforce an instruction budget and MUST trap on budget exceed (VM_TIMEOUT).
18.6 Debug emission MUST NOT block and MUST be bounded work per packet.
	19.	Test requirements (minimum)
19.1 ShmRing threaded SPSC test for at least 10 million ops.
19.2 ShmRing cross-process test mapping the same fd.
19.3 Descriptor validation tests:
obj_off alignment and bounds
data_len/scratch_len bounds
cookie mismatch -> StaleDescriptor
reserved0 non-zero -> InvalidDescriptor
19.4 Object TX stripping test: scratch_len becomes 0 before TX and transmitted bytes equal data_len.
19.5 MicroVM verifier tests:
out-of-bounds access rejected
unbounded loop rejected
stack overflow traps
illegal helper traps
19.6 Pipeline tests:
PASS and REDIRECT routing to correct rings
DROP returns descriptor to free pool
ABORT records error and does not TX
19.7 DebugRing tests:
event size is 64, slot len matches
full ring causes drops and increments debug_drop_count
VM_TRAP emits DebugEvent with reason VM_TRAP (best-effort)
19.8 DumpArena tests (if implemented):
DumpRecord layout fixed 512 bytes
VM_TRAP produces a DumpRecord with correct magic/version and copies descriptor bytes
19.9 Shutdown/close tests for ShmRing wake-all behavior.
19.10 Wrap-around test for ShmRing head/tail across u64::MAX.
	20.	Compliance checklist
20.1 Implements ShmRing ABI exactly and validates on attach; enforces SDPR slot payload rules.
20.2 Implements ShmRing Acquire/Release orderings and futex protocol exactly as required.
20.3 Enforces single-owner baton invariant for descriptors and objects.
20.4 Implements SDPR.Object and SDPR.Descriptor ABIs exactly and validates reserved bytes.
20.5 Enforces cookie generation checks to detect stale descriptors.
20.6 MicroVM verifier enforces memory safety and bounded execution; runtime enforces bounds/alignment regardless of verifier.
20.7 TX path strips scratch (scratch_len=0) before transmission.
20.8 If DebugRing is supported:
DebugEvent ABI is fixed 64 bytes
DebugRing is per-stage SPSC
debug emission is non-blocking and increments debug_drop_count on full
20.9 If DumpArena is supported:
DumpRecord ABI is fixed 512 bytes
dump writes are bounded and non-blocking

End of SDPR Spec v0.0.1
