Title: SDPR Spec v0.0.1 (Linux, Rust, ASCII)
	0.	Status
0.1 This document defines the frozen v0.0.1 specification for SDPR (ShmDataPlane Runtime): a shared-memory, single-host data-plane runtime that composes verified MicroVM programs over baton-style descriptors transported by SPSC futex rings, with low-overhead debug capture via dedicated per-stage SPSC debug rings, mandatory exception capture via dedicated per-stage SPSC exception rings, fixed per-stage statistics and policy configuration via StatsPage mappings, and a reclaim path using generation cookies.
0.2 Target: Linux userspace, single host, multi-process, high-throughput/low-latency data-plane pipelines.
0.3 Normative keywords: MUST, MUST NOT, SHOULD, MAY.
0.4 Version policy (fixed): Implementations claiming compliance with SDPR v0.0.1 MUST implement exactly the ABIs and semantics in this document. No forward-compat behavior is defined in v0.0.1.
	1.	Goals
1.1 Provide a data-plane function runtime where each stage executes a MicroVM program over a packet-like buffer object and returns an XDP-like action controlling routing.
1.2 Provide baton-style ownership transfer of buffer objects via fixed-size descriptors.
1.3 Provide zero-copy or near-zero-copy pipeline composition using shared-memory arenas and SPSC rings.
1.4 Provide strict, testable, frozen ABIs for:
1.4.1 ShmRing (SPSC futex ring) used to transport descriptors and debug/exception events.
1.4.2 Arena object layout used to store packet bytes and scratch bytes.
1.4.3 Descriptor layout used as the baton.
1.4.4 DebugEvent layout used for low-overhead observability and exception capture.
1.4.5 StatsPage layout used for per-stage counters and policy configuration.
1.4.6 MicroVM bytecode format, verifier rules, and execution contract.
1.5 Provide deterministic, auditable compute for MicroVM:
1.5.1 MicroVM program evaluation MUST be deterministic as defined by 13.7.
1.5.2 Stage outcomes MAY additionally be affected by policy gating and scheduling (e.g., rate limiting, sampling, wait strategy). Such policy gating MUST be explicit, bounded, and observable via counters and events, and MUST NOT compromise MicroVM determinism.
1.5.3 SDPR v0.0.1 defines a single default profile for correctness and performance. Implementations MUST conform to the default rules; no optional feature profiles are defined in v0.0.1.
1.6 Provide bounded, non-blocking safety controls suitable for DoS prevention, including per-stage rate limiting and per-stage scratch quotas, without blocking the data plane.
1.7 Provide an optimized build contract where steady-state hot paths have no syscalls when progress is possible, and where batching is used to amortize shared-memory and accounting costs, without changing data-plane semantics.
1.8 Eliminate implementation drift: SDPR v0.0.1 defines a single mandatory stage execution model, a single mandatory policy update protocol, and a single mandatory exception-capture path.
	2.	Non-goals
2.1 Cross-host transport.
2.2 MPMC rings, fairness guarantees, or starvation guarantees for stage input rings.
2.3 Crash-recovery or automatic attachment stealing.
2.4 Kernel-bypass waiting (futex requires syscalls when sleeping/waking).
2.5 Strong security isolation against a malicious peer beyond:
2.5.1 OS-level fd/mmap permissions,
2.5.2 MicroVM verifier and hardening,
2.5.3 mandatory runtime safety checks defined here.
2.6 Multi-ingress blocking wait (waiting on multiple input rings simultaneously) for stages in v0.0.1.
2.7 Lossless logging. Debug and exception capture are best-effort and MAY drop under pressure.
2.8 Global pipeline-wide resource accounting (e.g., total scratch bytes across the pipeline) in v0.0.1.
2.9 MicroVM helper syscalls, I/O, randomness, allocation, and time access are not supported in v0.0.1.
	3.	SDPR model
3.1 Objects
3.1.1 Arena: a shared-memory region partitioned into fixed-size objects.
3.1.2 Object: a fixed-size buffer containing:
(a) an object header,
(b) data bytes (wire packet bytes),
(c) scratch bytes (internal-only bytes).
3.1.3 Descriptor (baton): a fixed-size record that identifies one object and carries minimal metadata. Descriptors are transported on ShmRings.
3.1.4 Action: a 32-bit return code produced by MicroVM, interpreted by the router to decide the next hop.
3.1.5 Event: a fixed-size record emitted by stages to dedicated event rings. SDPR uses one Event ABI (SDPR.DebugEvent) for both debug and exception channels.
3.1.6 StatsPage: a fixed-size per-stage shared-memory page containing counters and policy configuration.

3.2 Ownership and transfer
3.2.1 At any time, exactly one stage owns a descriptor and its referenced object.
3.2.2 Ownership is transferred only by pushing the descriptor into exactly one output DataRing, or by pushing it into FreeRing (reclaim), or by pushing it into the TX input ring (TXRing) as defined in 14 and 15.
3.2.3 After transfer, the previous owner MUST NOT access the object bytes referenced by that descriptor.
3.2.4 v0.0.1 provides no refcounting. Double-free and stale-descriptor use are detected best-effort via cookie checks and reserved/len validation.
3.2.5 Ownership rules are enforced by topology constraints in 15.7 (mandatory).

3.3 Pipeline
3.3.1 A stage is a worker with:
(a) exactly one input DataRing (consumer role),
(b) zero or more output DataRings (producer roles),
(c) one loaded MicroVM program table (prog_id -> program bytes),
(d) an action router mapping REDIRECT(qid) to an output DataRing,
(e) one DebugRing (mandatory; per-stage SPSC),
(f) one ExceptionRing (mandatory; per-stage SPSC),
(g) one StatsPage (mandatory; section 16.3),
(h) one dedicated FreeRing producer for reclaim (section 15.3 and 15.7).
3.3.2 A pipeline is a directed acyclic graph (DAG) of stages connected by DataRings plus allocator/reclaimer rings per section 15, and TX rings per section 14/15 if TX is used.
3.3.3 Pipeline configurations MUST be acyclic. The control plane MUST detect and reject cycles during pipeline setup.
3.3.4 In v0.0.1, a stage MUST block on at most one input ring at a time (single-ingress wait).

3.4 Mandatory execution model: batching and amortization
3.4.1 A compliant implementation MUST process descriptors in batches (vectors) to amortize ring operations, routing, stats, and event costs, without changing data-plane semantics.
3.4.2 Batch processing MUST preserve ordering within a single input DataRing.
3.4.3 Batch processing MUST NOT reassign ownership semantics: each descriptor remains single-owner at all times.
3.4.4 Batch processing MUST NOT introduce new blocking behavior on the data plane hot path.
3.4.5 A stage MUST perform at most one push_batch per output ring per main-loop iteration (section 14.5).
	4.	Platform requirements
4.1 OS and futex
4.1.1 Linux kernel providing futex(2) for shared futex words.
4.1.2 Inter-process use MUST use shared futex operations (MUST NOT use FUTEX_PRIVATE variants).
4.1.3 ShmRing MUST use FUTEX_WAIT and FUTEX_WAKE only in v0.0.1.
4.2 Architecture
4.2.1 64-bit only: x86_64 or aarch64.
4.2.2 All atomics used by ShmRing, object headers, and StatsPage counters MUST be lock-free on the target. Implementations MUST fail create/attach if required atomics are not lock-free.
4.3 Endianness
4.3.1 All integer fields in all SDPR ABIs are little-endian.
4.4 Cache line model
4.4.1 Cache line size is treated as 64 bytes for ABI layout and padding rules. Implementations MUST follow fixed offsets and cache-line separation requirements regardless of host microarchitecture.
4.5 Time source requirements (mandatory)
4.5.1 Stages and collectors that emit timestamps (event ts) or apply rate limiting MUST obtain monotonic timestamps in nanoseconds without syscalls on steady-state hot paths.
4.5.2 An implementation MUST document its time source. Allowed sources include:
(a) vDSO clock_gettime for CLOCK_MONOTONIC_RAW or CLOCK_MONOTONIC (no syscall fallback on the hot path),
(b) calibrated cycle counters (e.g., TSC or CNTVCT) converted to nanoseconds with a monotonic mapping.
4.5.3 If the platform cannot provide syscall-free monotonic timestamps, the implementation MUST NOT claim compliance with the optimized build SLOs in section 18.
	5.	Files, fds, and mappings
5.1 SDPR instance resources
5.1.1 An SDPR instance consists of:
(a) one Arena fd (memfd or shm_open),
(b) one or more ShmRing fds (one per pipeline edge and per allocator edge and per event ring and per TX edge),
(c) one StatsPage fd per stage (mandatory; section 16.3),
(d) one read-only ProgramTable fd per stage (mandatory; section 14.2 and 13.2),
(e) one read-only RouteTable fd per stage (mandatory; section 14.2),
(f) optional read-only PolicyTable fd (not used in v0.0.1; policy is in StatsPage only).
5.2 Sharing
5.2.1 The control plane MUST share fds to worker processes via fork+inherit, unix domain socket SCM_RIGHTS, or equivalent.
5.3 Mapping rules
5.3.1 Arena and ShmRings MUST be mapped with mmap(PROT_READ|PROT_WRITE, MAP_SHARED).
5.3.2 Read-only tables MUST be mapped PROT_READ.
5.3.3 StatsPage mappings MUST be mmap(PROT_READ|PROT_WRITE, MAP_SHARED).
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
6.2.5 arena_total_size MUST be <= (1<<48) bytes in v0.0.1.
6.3 Arena object addressing
6.3.1 An object is addressed by obj_index in [0, object_count).
6.3.2 obj_off = obj_index * object_size.
6.3.3 The object base address in memory is arena_base + obj_off.
6.3.4 Descriptors MUST use obj_off (not obj_index) in v0.0.1.
6.4 Prefault and paging (mandatory for SLO measurements)
6.4.1 Implementation MUST touch all mapped pages once at init to reduce page-fault latency in hot paths for deployments claiming SLO compliance.
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
7.1.4 Writers MUST write header fields with plain stores before publishing the descriptor transfer (descriptor publish is the synchronization point as defined in 14.3).
7.1.5 Readers MUST treat cookie/data_len/scratch_len/flags as untrusted until validated.
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
7.4.1 Before any TX action is submitted to TXRing, implementations MUST set scratch_len to 0 in both descriptor and object header, and MUST ensure bytes beyond data_len are not transmitted.
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
8.3.5 reserved0 MUST be zero; non-zero MUST be rejected as InvalidDescriptor.
8.3.6 After 8.3.1-8.3.5, the stage MUST validate the referenced object header using only the first 64 bytes of the object:
(a) header.magic matches 7.1.2,
(b) header.obj_version_major == 0 and header.obj_version_minor == 1,
(c) header.flags has no reserved bits set (7.2),
(d) header.cookie matches descriptor.cookie; mismatch MUST be treated as StaleDescriptor,
(e) header.data_len equals descriptor.data_len and header.scratch_len equals descriptor.scratch_len; mismatch MUST be InvalidDescriptor,
(f) header.reserved0 == 0 and header.reserved1 all zero; otherwise InvalidDescriptor.
8.3.7 No stage MAY read or write object data/scratch bytes unless 8.3.1-8.3.6 succeed.
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
9.3.1 PASS: deliver descriptor to the stage default output DataRing.
9.3.2 DROP: deliver descriptor to the reclaim path per 15 (FreeRing); object becomes free for reuse only after reclaimer updates cookie.
9.3.3 REDIRECT(qid): qid = action_arg; deliver descriptor to the configured output DataRing for qid.
9.3.4 TX: deliver descriptor to the TXRing; scratch MUST be stripped (7.4.1).
9.3.5 ABORT(code): code = action_arg; stage MUST record code and MUST deliver descriptor to reclaim; MUST NOT TX.
9.3.6 TAILCALL(prog_id): prog_id = (u16)action_arg; bits 24..31 of action_arg MUST be zero in v0.0.1. Stage MUST run the referenced program as a tail call with the same ctx, subject to verifier-enforced tailcall depth limit. Non-zero upper bits MUST be treated as ABORT(VM_ILLEGAL).
	10.	SDPR.IPC.ShmRing v0.0.1 (SPSC futex ring)
10.1 Status
10.1.1 ShmRing is a frozen ABI for a shared-memory single-producer single-consumer (SPSC) bounded ring of fixed-size slots with futex-based blocking on empty and optional blocking on full.
10.1.2 ShmRing is used by SDPR for DataRing, DebugRing, ExceptionRing, FreeRing, AllocRing, TXRing, and TXFreeRing.
10.1.3 ShmRing correctness relies on:
(a) head/tail monotonic u64 counters with wrapping arithmetic,
(b) Acquire/Release ordering on head/tail,
(c) user-space recheck loops around futex waits,
(d) uniqueness of exactly one producer and exactly one consumer per ring mapping.

10.2 Slot payload usage in SDPR (mandatory)
10.2.1 DataRing: slot payload MUST contain exactly one 64-byte SDPR.Descriptor and slot_header.len MUST equal 64.
10.2.2 DebugRing: slot payload MUST contain exactly one 64-byte SDPR.DebugEvent and slot_header.len MUST equal 64.
10.2.3 ExceptionRing: slot payload MUST contain exactly one 64-byte SDPR.DebugEvent and slot_header.len MUST equal 64.
10.2.4 FreeRing: slot payload MUST contain exactly one 64-byte SDPR.Descriptor and slot_header.len MUST equal 64.
10.2.5 AllocRing: slot payload MUST contain exactly one 64-byte SDPR.Descriptor and slot_header.len MUST equal 64.
10.2.6 TXRing: slot payload MUST contain exactly one 64-byte SDPR.Descriptor and slot_header.len MUST equal 64.
10.2.7 TXFreeRing: slot payload MUST contain exactly one 64-byte SDPR.Descriptor and slot_header.len MUST equal 64.

10.3 Platform requirements (ShmRing)
10.3.1 Linux futex(2) for shared futex words.
10.3.2 Inter-process use MUST NOT use FUTEX_PRIVATE variants.
10.3.3 v0.0.1 MUST use FUTEX_WAIT and FUTEX_WAKE only.
10.3.4 64-bit only: x86_64 or aarch64.
10.3.5 ShmRing-required atomics MUST be lock-free; create/attach MUST fail if not.
10.3.6 Cache line size is treated as 64 bytes for layout; fixed offsets MUST be followed exactly.
10.3.7 All integer fields are little-endian.

10.4 ShmRing primitive model and invariants (mandatory)
10.4.1 cap: ring capacity in slots, cap = 1 << capacity_pow2.
10.4.2 head: AtomicU64, monotonically increasing modulo 2^64, counts produced slots.
10.4.3 tail: AtomicU64, monotonically increasing modulo 2^64, counts consumed slots.
10.4.4 used: head.wrapping_sub(tail).
10.4.5 empty: used == 0.
10.4.6 full: used == cap.
10.4.7 Fundamental correctness invariant:
10.4.7.1 Producer MUST NOT advance head such that used would exceed cap.
10.4.7.2 Consumer MUST NOT advance tail beyond head (i.e., MUST NOT consume when empty).
10.4.8 Corruption rule:
10.4.8.1 If used > cap, the ring state is CORRUPT and MUST trigger Shutdown behavior (10.12.4).

10.5 Shared memory creation and mapping (ShmRing)
10.5.1 Creator obtains an fd via memfd_create or shm_open.
10.5.2 Creator MUST size via ftruncate(fd, total_size).
10.5.3 Attach MUST mmap(fd, total_size, PROT_READ|PROT_WRITE, MAP_SHARED).
10.5.4 Attach MUST validate header fields and reserved bytes before any operations.
10.5.5 Prefault is mandatory for deployments claiming SLO compliance: implementation MUST touch mapped pages once at init.

10.6 ABI constants and layout (ShmRing fixed)
10.6.1 magic: u64 = 0x5348515350534651
10.6.2 version_major: u16 = 0
10.6.3 version_minor: u16 = 1
10.6.4 Attach MUST accept only (version_major == 0 and version_minor == 1).
10.6.5 header_size is fixed: 0x180 bytes.
10.6.6 header_size MUST be a multiple of 64.
10.6.7 ring_offset MUST be 0x180 and MUST be aligned to 64.
10.6.8 head, tail, doorbell_ne, doorbell_nf MUST start at offsets that are multiples of 64 and MUST NOT share a cache line with each other.

10.7 ShmRing header layout (frozen offsets)
10.7.1 The header is a fixed binary layout. Implementations MUST treat the mapping as bytes and use fixed offsets below.
10.7.2 Reserved bytes MUST be zero on create and MUST be validated as zero on attach.
10.7.3 Reserved flag bits (7..31) MUST be zero on create and MUST be validated as zero on attach.
10.7.4 Header offsets (from mapping base):
0x000 magic: u64
0x008 version_major: u16
0x00A version_minor: u16
0x00C header_size: u32 (MUST be 0x180)
0x010 total_size: u64
0x018 ring_offset: u64 (MUST be 0x180)
0x020 ring_bytes: u64
0x028 arena_offset: u64 (MUST be 0 in SDPR v0.0.1 ShmRing)
0x030 arena_bytes: u64 (MUST be 0 in SDPR v0.0.1 ShmRing)
0x038 capacity_pow2: u8
0x039 reserved0: [u8; 7] (MUST be 0)
0x040 slot_size: u32
0x044 reserved1: u32 (MUST be 0)
0x048 flags: AtomicU32
0x04C reserved2: u32 (MUST be 0)
0x050 producer_pid: AtomicU32 (diagnostic only)
0x054 consumer_pid: AtomicU32 (diagnostic only)
0x058 error_code: AtomicU32 (diagnostic only)
0x05C reserved3: u32 (MUST be 0)
0x060 reserved4: [u8; 32] (MUST be 0)
0x080 head: AtomicU64
0x088 reserved5: [u8; 56] (MUST be 0)
0x0C0 tail: AtomicU64
0x0C8 reserved6: [u8; 56] (MUST be 0)
0x100 doorbell_ne: AtomicI32
0x104 reserved7: [u8; 60] (MUST be 0)
0x140 doorbell_nf: AtomicI32
0x144 reserved8: [u8; 60] (MUST be 0)

10.8 ShmRing flags (AtomicU32 bitfield)
10.8.1 bit 0 INITIALIZED
10.8.2 bit 1 PRODUCER_ATTACHED
10.8.3 bit 2 CONSUMER_ATTACHED
10.8.4 bit 3 PRODUCER_CLOSED
10.8.5 bit 4 CONSUMER_CLOSED
10.8.6 bit 5 SHUTDOWN
10.8.7 bit 6 NOT_FULL_ENABLED
10.8.8 bits 7..31 reserved (MUST be 0)

10.9 Ring layout and slot encoding (ShmRing)
10.9.1 cap = 1 << capacity_pow2.
10.9.2 mask = cap - 1.
10.9.3 slot_stride = slot_size.
10.9.4 ring_bytes MUST equal cap * slot_size.
10.9.5 ring starts at ring_offset (0x180) and spans ring_bytes.
10.9.6 Slot encoding (mandatory):
10.9.6.1 Each slot begins with an 8-byte slot header followed by payload bytes.
10.9.6.2 slot_header at slot offset 0:
(a) len: u16 (0..=payload_capacity)
(b) tag: u16 (application-defined; SDPR MAY use 0)
(c) sflags: u16
(c.1) bit 0 VALID (debug-only; MAY be set)
(c.2) bits 1..15 reserved, MUST be 0
(d) reserved: u16, MUST be 0
10.9.6.3 payload_capacity = slot_size - 8.
10.9.6.4 payload_capacity MUST be <= 65535.
10.9.6.5 Payload bytes stored at slot+8..slot+8+len.
10.9.6.6 Bytes beyond len are unspecified and MUST be ignored by consumer.
10.9.6.7 VALID MUST NOT be used as the synchronization primitive. Synchronization is exclusively via head publish (Release) and head observe (Acquire).
10.9.6.8 If VALID is not used, producer MUST write sflags=0 and consumer MUST ignore VALID.

10.10 Memory ordering contract (ShmRing, mandatory)
10.10.1 Producer publish rule:
10.10.1.1 Producer MUST write slot payload and slot header using plain stores before publishing head.
10.10.1.2 Producer MUST publish head using store with Release ordering.
10.10.2 Consumer observe rule:
10.10.2.1 Consumer MUST load head using Acquire ordering before reading any slot contents that are claimed available by head.
10.10.2.2 Consumer MUST publish tail using store with Release ordering after consuming.
10.10.3 Full/empty checks:
10.10.3.1 Producer MUST load tail using Acquire ordering when checking for full.
10.10.3.2 Consumer MUST load head using Acquire ordering when checking for empty in blocking paths.
10.10.4 Doorbell ordering:
10.10.4.1 Doorbell loads/stores MAY be Relaxed.
10.10.4.2 Correctness MUST rely on Acquire/Release on head/tail plus user-space rechecks.
10.10.4.3 Doorbells are wake hints only.
10.10.5 Flags and initialization ordering:
10.10.5.1 Creator MUST initialize all header fields, reserved bytes, ring bytes, head=0, tail=0, doorbells=0.
10.10.5.2 Creator MUST set or clear NOT_FULL_ENABLED before setting INITIALIZED.
10.10.5.3 Creator MUST set INITIALIZED using a Release operation after initialization is complete.
10.10.5.4 Blocking loops MUST read flags with Acquire when checking for SHUTDOWN/CLOSED.
10.10.5.5 Attach MUST NOT perform operations until INITIALIZED is observed with Acquire.
10.10.6 Forbidden synchronization patterns (mandatory)
10.10.6.1 Implementations MUST NOT use slot_header.sflags.VALID (or any other slot byte) as a correctness synchronization mechanism.
10.10.6.2 Implementations MUST NOT publish head or tail using Relaxed ordering.
10.10.6.3 Implementations MUST NOT depend on doorbell ordering for correctness.

10.11 Ownership and attachment protocol (ShmRing, mandatory)
10.11.1 Exactly one Producer may perform push operations.
10.11.2 Exactly one Consumer may perform pop operations.
10.11.3 Cross-process uniqueness MUST be enforced via flags bits:
10.11.3.1 attach_producer MUST set PRODUCER_ATTACHED via compare_exchange on the whole flags word.
10.11.3.2 attach_consumer MUST set CONSUMER_ATTACHED via compare_exchange on the whole flags word.
10.11.3.3 CAS success ordering MUST be AcqRel and failure ordering MUST be Acquire.
10.11.3.4 CAS MUST preserve all other defined flag bits (it MUST only set the relevant ATTACHED bit).
10.11.3.5 If the relevant bit is already set, attach MUST fail with Error::AlreadyAttached.
10.11.4 v0.0.1 defines no crash-recovery or stealing of attachments. A stale ATTACHED bit requires creating a new ring.
10.11.5 Only Producer writes head. Only Consumer writes tail.
10.11.6 Only Consumer waits on doorbell_ne.
10.11.7 Only Producer waits on doorbell_nf when NOT_FULL_ENABLED.
10.11.8 PID fields are diagnostic only and MUST NOT be used for correctness.

10.12 Futex protocol (ShmRing exact, mandatory)
10.12.1 Futex word requirements:
10.12.1.1 doorbell_ne and doorbell_nf are 32-bit aligned i32 in shared memory.
10.12.1.2 Futex WAIT MUST use expected value equal to an epoch read from the doorbell word.
10.12.1.3 Waiter MUST recheck conditions in user space after any futex return (spurious wakeups allowed).
10.12.2 Timeout model (fixed-budget):
10.12.2.1 timeout is an optional relative duration and is a total budget for the entire blocking call.
10.12.2.2 Implementations MUST track a deadline and pass remaining time to each futex_wait.
10.12.2.3 Spurious returns MUST NOT extend the total wait budget.
10.12.2.4 If remaining <= 0, blocking operation MUST return Timeout without calling futex_wait.
10.12.3 Spin-then-futex (permitted and mandatory where required by SDPR stage wait strategy):
10.12.3.1 Blocking pop/push MAY spin for a configurable spin_iters before futex_wait.
10.12.3.2 Spin MUST use cpu_relax and MUST recheck head/tail conditions in user space.
10.12.4 CorruptIndices handling:
10.12.4.1 On used > cap detection, implementation MUST:
(a) set flags.SHUTDOWN best-effort (Release),
(b) wake both doorbells with wake-all (count INT_MAX),
(c) return Error::CorruptIndices.

10.13 Not-empty wait and wake (consumer/prod)
10.13.1 Consumer waits only when empty.
10.13.2 pop_blocking(timeout) loop (normative):
(a) If not empty, consume.
(b) If flags.SHUTDOWN set, return Shutdown.
(c) If flags.PRODUCER_CLOSED set and empty, return Closed.
(d) Spin phase (optional): for i in 0..spin_iters, cpu_relax and recheck.
(e) epoch = doorbell_ne.load(Relaxed).
(f) Recheck empty; if still empty, futex_wait(doorbell_ne, epoch, remaining_timeout).
(g) Treat EINTR/EAGAIN as spurious; on ETIMEDOUT return Timeout; on other errno return Syscall(FutexWaitNe, errno).
10.13.3 Producer MUST wake consumer only on empty->non-empty transition:
(a) before publishing, read tail Acquire and compute was_empty (head_local == tail),
(b) publish head with Release,
(c) if was_empty: doorbell_ne.fetch_add(1, Relaxed) then futex_wake(doorbell_ne, 1).
10.13.4 Wake discipline (mandatory)
10.13.4.1 Producer MUST NOT call futex_wake(doorbell_ne, …) unless it has observed was_empty==true immediately before the publish that made the ring non-empty.
10.13.4.2 In batch operations, producer MUST perform at most one not-empty wake per batch, and only if the batch caused empty->non-empty.

10.14 Not-full wait and wake (optional in ShmRing; SDPR does not require blocking push in v0.0.1)
10.14.1 Enabled iff flags.NOT_FULL_ENABLED is set at init and observed after INITIALIZED.
10.14.2 Producer waits only when full.
10.14.3 push_blocking(timeout) loop mirrors 10.13.2 with doorbell_nf and full check.
10.14.4 Consumer MUST wake producer only on full->not-full transition when NOT_FULL_ENABLED:
(a) before advancing tail, read head Acquire and compute was_full,
(b) publish tail Release,
(c) if was_full: doorbell_nf.fetch_add(1, Relaxed) then futex_wake(doorbell_nf, 1).
10.14.5 Wake discipline (mandatory when NOT_FULL_ENABLED)
10.14.5.1 Consumer MUST NOT call futex_wake(doorbell_nf, …) unless it has observed was_full==true immediately before the publish that made the ring not-full.
10.14.5.2 In batch operations, consumer MUST perform at most one not-full wake per batch, and only if the batch caused full->not-full.

10.15 Close and shutdown semantics (ShmRing, mandatory)
10.15.1 close_producer():
(a) set flags.PRODUCER_CLOSED via fetch_or(Release),
(b) doorbell_ne.fetch_add(1, Relaxed), futex_wake(doorbell_ne, INT_MAX),
(c) after close_producer, producer MUST NOT push.
10.15.2 close_consumer():
(a) set flags.CONSUMER_CLOSED via fetch_or(Release),
(b) when NOT_FULL_ENABLED: doorbell_nf.fetch_add(1, Relaxed), futex_wake(doorbell_nf, INT_MAX),
(c) after close_consumer, consumer MUST NOT pop.
10.15.3 shutdown():
(a) flags.fetch_or(SHUTDOWN, Release),
(b) doorbell_ne.fetch_add(1, Relaxed), futex_wake(doorbell_ne, INT_MAX),
(c) doorbell_nf.fetch_add(1, Relaxed), futex_wake(doorbell_nf, INT_MAX).

10.16 Ring operations (ShmRing exact, mandatory behavior)
10.16.1 Producer local state:
(a) Producer MUST keep head_local in private memory.
(b) Producer SHOULD keep cached_tail in private memory to reduce shared loads.
10.16.2 Consumer local state:
(a) Consumer MUST keep tail_local in private memory.
(b) Consumer SHOULD keep cached_head in private memory to reduce shared loads.
10.16.3 try_push(tag, payload):
(a) Validate payload.len <= payload_capacity.
(b) Read flags Acquire; if SHUTDOWN return Shutdown; if CONSUMER_CLOSED return Closed.
(c) If cached_tail stale or predicted full, refresh tail Acquire.
(d) If head_local.wrapping_sub(tail) == cap, return Full.
(e) Compute slot = ring_base + ((head_local & mask) * slot_stride).
(f) Copy payload into slot+8..slot+8+len; write slot_header with reserved==0 and sflags reserved bits zero.
(g) next_head = head_local + 1 (wrapping).
(h) Publish head.store(Release, next_head).
(i) If transition empty->non-empty, perform not-empty wake (10.13.3).
(j) head_local = next_head.
10.16.4 try_pop(out):
(a) If cached_head stale or predicted empty, refresh head Acquire.
(b) If cached_head == tail_local:
(b.1) Read flags Acquire; if SHUTDOWN return Shutdown; if PRODUCER_CLOSED return Closed; else return Empty.
(c) Compute slot = ring_base + ((tail_local & mask) * slot_stride).
(d) Read slot_header; if len > payload_capacity return CorruptSlot; if reserved!=0 return CorruptSlot; if sflags reserved bits non-zero return CorruptSlot.
(e) Copy payload to out; if out too small return OutputTooSmall.
(f) next_tail = tail_local + 1 (wrapping); publish tail.store(Release, next_tail).
(g) tail_local = next_tail.
10.16.5 Batch operations (mandatory for SDPR v0.0.1)
10.16.5.1 push_batch(k items) all-or-nothing:
(a) If k==0 return Ok.
(b) Read flags Acquire; handle SHUTDOWN/CLOSED.
(c) Read tail Acquire; was_empty = (head_local == tail).
(d) If used + k > cap return Full and MUST NOT publish any items.
(e) Write k slots; publish head once Release (head_local + k).
(f) If was_empty, perform exactly one not-empty wake.
(g) Update head_local.
10.16.5.2 pop_batch(up to k items) partial permitted:
(a) If k==0 return 0.
(b) Refresh head Acquire if needed; if empty, handle flags like try_pop.
(c) Pop n available items (0..=k) following slot rules; if any slot is corrupt, return CorruptSlot for that operation and MAY set SHUTDOWN (implementation-defined for ShmRing). SDPR requires fail-closed handling in section 14.
(d) Publish tail once Release.
(e) Return n.

10.17 ShmRing attach validation (mandatory)
10.17.1 Attach MUST validate:
(a) magic matches,
(b) version_major == 0 and version_minor == 1,
(c) header_size == 0x180,
(d) total_size equals mapping size,
(e) ring_offset == 0x180,
(f) ring_bytes within total_size and total_size == header_size + ring_bytes,
(g) slot_size >= 8 and slot_size % 8 == 0,
(h) payload_capacity <= 65535,
(i) capacity_pow2 in [1, 30],
(j) ring_bytes == (1 << capacity_pow2) * slot_size,
(k) arena_offset == 0 and arena_bytes == 0,
(l) all reserved bytes are 0,
(m) reserved flag bits are 0,
(n) all size computations use checked arithmetic; reject on overflow.
10.17.2 If any validation fails, attach MUST return an error and MUST NOT perform operations.
10.17.3 Attach MUST observe flags.INITIALIZED (Acquire) before starting operations; if not set, attach MAY return WouldBlock.

10.18 SDPR-required slot size constraints (mandatory)
10.18.1 For DataRing, DebugRing, ExceptionRing, FreeRing, AllocRing, TXRing, TXFreeRing:
slot_size MUST equal 128, and payload_capacity MUST equal 120.
10.18.2 Implementations MUST reject SDPR ring attachment if slot_size != 128 for any SDPR ring.
	11.	SDPR.DebugEvent v0.0.1 (event ABI for debug and exceptions)
11.1 Status
11.1.1 DebugEvent is a fixed binary record intended for low-overhead, best-effort observability and exception capture.
11.1.2 Event capture MUST NOT block the data plane. If an event ring is full, events MUST be dropped and counted.
11.2 Event ring topology (mandatory)
11.2.1 DebugRing MUST be SPSC: exactly one stage produces into its DebugRing, exactly one collector consumes from it.
11.2.2 ExceptionRing MUST be SPSC: exactly one stage produces into its ExceptionRing, exactly one collector consumes from it.
11.2.3 Each stage MUST have its own DebugRing and its own ExceptionRing. Sharing an event ring among multiple stages is non-compliant.
11.3 DebugEvent size and alignment (fixed)
11.3.1 DebugEvent size is fixed: 64 bytes.
11.3.2 DebugEvent MUST be aligned to 8 bytes in the ring payload.
11.4 DebugEvent layout (little-endian, fixed offsets)
11.4.1 0x00 ts: u64 (monotonic time in nanoseconds; source must satisfy 4.5)
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
11.4.12 0x38 extra0: u32 (reason-dependent; e.g., vm_code or pc; MAY be 0)
11.4.13 0x3C extra1: u32 (reason-dependent; MAY be 0)
11.5 DebugReason enumeration (required values)
11.5.1 0 NONE
11.5.2 1 STAGE_IN
11.5.3 2 STAGE_OUT
11.5.4 3 VM_ABORT
11.5.5 4 VM_TRAP
11.5.6 5 STALE_DESC
11.5.7 6 INVALID_DESC
11.5.8 7 ROUTE_MISS
11.5.9 8 CORRUPT_RING
11.5.10 9 DROP
11.5.11 10 TX
11.5.12 11 TX_FAIL
11.6 Aux and extra conventions (normative)
11.6.1 aux meaning is reason-dependent. Unless specified below, aux MAY be 0.
11.6.2 For reason=DROP, aux=1 means RATE_LIMIT drop, aux=2 means POLICY drop or ALLOC_STARVATION drop.
11.6.3 For reason=VM_ABORT, aux=1 means VM_TIMEOUT, aux=2 means SCRATCH_QUOTA violation (stage synthesized ABORT(VM_ILLEGAL)), aux=3 means VM_ILLEGAL, aux=4 means VM_BOUNDS, aux=5 means VM_ALIGN, aux=6 means VM_STACK, aux=7 means VM_TRAP.
11.6.4 For reason=VM_TRAP, aux MAY carry trap class (imm32 of TRAP); extra0 SHOULD carry vm_code (VM_TRAP) and extra1 SHOULD carry pc.
11.6.5 For reason=TX_FAIL, aux=1 means SOFT_FAIL, aux=2 means HARD_FAIL; extra0 MAY carry an implementation-defined NIC error code.
11.7 DebugRing emission rules (mandatory behavior)
11.7.1 DebugRing capture is best-effort; it MUST NOT cause blocking waits on the debug ring.
11.7.2 On DebugRing Full, the stage MUST drop the event and MUST increment debug_drop_count (section 16.3).
11.7.3 Stages MUST support deterministic sampling using trace_id bitmasking:
emit if (trace_id & sample_mask) == 0.
11.7.4 DebugRing MUST emit STAGE_IN and STAGE_OUT under sampling policy.
11.7.5 DebugRing MUST attempt 100% best-effort emission for reasons:
VM_ABORT, VM_TRAP, STALE_DESC, INVALID_DESC, CORRUPT_RING, ROUTE_MISS, TX_FAIL.
If DebugRing drops due to full, the corresponding exception MUST still be attempted on ExceptionRing (section 12).
11.8 Optimized profile: event emission batching (mandatory)
11.8.1 Stages MUST accumulate non-exception debug events (STAGE_IN/STAGE_OUT) in private memory and flush them in bounded bursts, provided:
(a) the flush remains best-effort and non-blocking,
(b) drops are counted in debug_drop_count.
11.8.2 ExceptionRing events MUST be attempted immediately best-effort (no buffering requirement; buffering permitted only if bounded and does not increase drop probability for exceptions).
	12.	SDPR.ExceptionRing v0.0.1 (mandatory exception channel)
12.1 Status
12.1.1 ExceptionRing is mandatory in SDPR v0.0.1 and is the single required exception-capture path.
12.1.2 ExceptionRing uses the SDPR.DebugEvent ABI (section 11) with mandatory reason semantics below.
12.2 Emission rules (mandatory)
12.2.1 ExceptionRing capture MUST NOT block the data plane. If the exception ring is full, events MUST be dropped and counted in exception_drop_count.
12.2.2 ExceptionRing MUST attempt to emit 100% best-effort for each occurrence of:
VM_ABORT, VM_TRAP, STALE_DESC, INVALID_DESC, CORRUPT_RING, ROUTE_MISS, TX_FAIL.
12.2.3 ExceptionRing MUST NOT emit sampled STAGE_IN/STAGE_OUT. It is exception-only.
12.2.4 ExceptionRing MUST include enough context for forensics using only validated or pre-validation fields:
trace_id, stage_id, prog_id, reason, aux, obj_off, cookie, data_len, scratch_len, action, extra0/extra1 as defined in 11.4 and 11.6.
12.2.5 On emission attempt (success or drop), the stage MUST increment exception_emit_count or exception_drop_count accordingly.
	13.	SDPR.VM.MicroVM v0.0.1
13.1 Execution contract
13.1.1 MicroVM executes over a context (ctx) derived from a descriptor and its referenced object.
13.1.2 MicroVM MUST NOT access memory outside:
(a) the object data region [obj_base+64, obj_base+64+data_len),
(b) the object scratch region [obj_base+64+data_len, obj_base+64+data_len+scratch_len),
(c) VM private stacks and registers.
13.1.3 MicroVM MUST return an SDPR.Action (section 9).
13.1.4 MicroVM runtime MUST enforce bounds and alignment checks even if the verifier is bypassed; violations MUST trap to ABORT(VM_BOUNDS or VM_ALIGN) or ABORT(VM_TRAP) as defined below.
13.1.5 MicroVM runtime MUST be deterministic per 13.7.
13.1.6 Length immutability (mandatory):
MicroVM MUST NOT modify descriptor.data_len or descriptor.scratch_len, and MUST NOT modify object header data_len or scratch_len. Any length changes are prohibited in v0.0.1.

13.2 Bytecode container
13.2.1 A program is a byte string containing:
(a) ProgramHeader (fixed 32 bytes),
(b) Instruction array (N * 8 bytes).
13.2.2 Constant pools are not permitted in v0.0.1:
Any bytes beyond ProgramHeader and the instruction array MUST cause verifier rejection.
13.2.3 ProgramHeader layout:
0x00 magic: u32 = 0x564D5053 (“SPMV” little-endian)
0x04 vm_version_major: u16 = 0
0x06 vm_version_minor: u16 = 1
0x08 prog_id: u16
0x0A flags: u16 (reserved, MUST be 0)
0x0C insn_count: u32
0x10 entry_pc: u32 (MUST be 0 in v0.0.1)
0x14 reserved: [u8; 12] (MUST be 0)
13.2.4 Instruction encoding (fixed 8 bytes, little-endian)
byte 0 opcode
byte 1 dst_reg (0..15)
byte 2 src_reg (0..15)
byte 3 imm8
bytes 4..7 imm32 (signed)
13.2.5 Registers (fixed meaning on entry)
r0 return action (initialized to PASS)
r1 ctx_base (object base address, read-only)
r2 data_len (u64, read-only)
r3 scratch_len (u64, read-only)
r4 obj_cap (u64, read-only)
r5 reserved (initialized to 0; MUST remain 0)
r6..r15 general purpose
13.2.6 Mandatory initialization (mandatory)
13.2.6.1 Runtime MUST initialize all registers r0..r15 to 0 before setting the mandated entry values (r0..r5 per 13.2.5).
13.2.6.2 Runtime MUST initialize all private stack bytes to 0 on entry.
13.2.6.3 Reading uninitialized registers/stack is still verifier-illegal (13.4.5); runtime zeroing is a safety backstop and must not be relied on for program validity.
13.2.7 Private stacks (fixed)
stack0 1024 bytes
stack1 1024 bytes
overflow/underflow MUST trap to ABORT(VM_STACK).
13.2.8 Instruction count limits (mandatory)
13.2.8.1 insn_count MUST be in [1, 65535] in v0.0.1.
13.2.8.2 Runtime MUST enforce insn_budget_max per invocation (13.6.1).

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
13.3.6 LOAD/STORE (only via address)
LD8/16/32/64 dst, addr_reg
ST8/16/32/64 addr_reg, src
unaligned MUST trap to ABORT(VM_ALIGN).
13.3.7 TRAP (mandatory)
TRAP imm32
Semantics:
	•	Runtime MUST trap immediately to ABORT(VM_TRAP).
	•	imm32 MUST be recorded in the ExceptionRing event as aux (best-effort) for reason=VM_TRAP.
	•	TRAP and its trap class MUST be verifier-visible and deterministic.
13.3.8 CALL is not supported in v0.0.1:
Any CALL opcode MUST be rejected by the verifier; if executed by bypassing the verifier, runtime MUST trap to ABORT(VM_ILLEGAL).

13.4 Verifier (mandatory)
13.4.1 Verifier MUST reject any program that can:
(a) access outside allowed regions,
(b) execute any backward jump (loop) in v0.0.1,
(c) exceed insn_budget_max per invocation (13.6.1),
(d) overflow stacks,
(e) use invalid encodings,
(f) use CALL.
13.4.2 Pointer vs scalar typing
Verifier MUST track register types: SCALAR or PTR(region, offset_range).
ADDR produces PTR with provable offset_range.
ALU ops on PTR are restricted to bounded add/sub with scalars, and MUST preserve provable offset_range within region_len.
13.4.3 Control flow
13.4.3.1 Jump targets MUST be within [0, insn_count).
13.4.3.2 Any jump with a negative pc delta (backward jump) MUST be rejected in v0.0.1.
13.4.4 Tail calls
13.4.4.1 If TAILCALL is supported, verifier MUST enforce tailcall_depth_max = 32.
13.4.4.2 TAILCALL target prog_id MUST be validated against the stage program table.
13.4.5 Definite initialization rules (mandatory)
13.4.5.1 Verifier MUST reject any program that reads any general-purpose register r6..r15 before it is written on all paths.
13.4.5.2 Verifier MUST reject any program that reads any stack byte before it is written on all paths.
13.4.5.3 r5 MUST remain 0: any write to r5 MUST be rejected.
13.4.6 Verifier complexity limits (mandatory)
13.4.6.1 Verifier MUST enforce bounded analysis complexity using implementation-defined limits (e.g., max_total_states and max_states_per_insn).
13.4.6.2 On limit exceed, verifier MUST reject the program with a deterministic error (e.g., VerifierTooComplex).

13.5 JIT/AOT
13.5.1 JIT and AOT compilation are not required in v0.0.1. If present:
13.5.1.1 Executable memory MUST be W^X.
13.5.1.2 JIT/AOT MUST preserve traps and ABORT codes.

13.6 Runtime budgets and traps (mandatory)
13.6.1 Runtime MUST enforce an instruction budget per invocation, insn_budget_max, and MUST trap to ABORT(VM_TIMEOUT) on budget exceed.
13.6.2 Runtime MUST trap to ABORT(VM_ILLEGAL) on illegal opcode, invalid register index, invalid encoding, or prohibited CALL execution.
13.6.3 Runtime MUST trap to ABORT(VM_STACK) on stack overflow/underflow.
13.6.4 Optimized build requirement (mandatory for SLO compliance)
insn_budget_max MUST be configured to a bounded value and MUST NOT exceed 4096.

13.7 Determinism constraints (mandatory)
13.7.1 MicroVM programs MUST be deterministic functions of:
(a) validated object bytes within [data_len, scratch_len],
(b) program bytecode bytes,
(c) read-only table bytes mapped PROT_READ and explicitly provided to the stage,
(d) initial ctx registers (r1..r4) derived from the descriptor.
13.7.2 MicroVM MUST NOT access time, randomness, thread ids, process ids, or any external state.
13.7.3 MicroVM MUST NOT perform syscalls, I/O, clock reads, or memory allocation.
13.7.4 TRAP is deterministic and MUST NOT read external state. It is intended for debugging and controlled aborts.

13.8 MicroVM ABORT codes (required)
VM_BOUNDS
VM_ALIGN
VM_STACK
VM_ILLEGAL
VM_TIMEOUT
VM_TRAP
	14.	Routing, safety controls, tables, and stage behavior
14.1 Stage main loop (mandatory single model)
14.1.1 A stage MUST operate as a single-threaded batch loop:
(a) pop_batch up to batch_max descriptors from input DataRing using non-blocking pop_batch,
(b) if none available, apply the mandatory wait strategy (14.5.6) and retry,
(c) validate each descriptor per 8.3 before any object region access,
(d) apply per-stage rate limiting (14.6). If limited, synthesize DROP without running MicroVM,
(e) construct ctx regs from descriptor and run MicroVM program selected by descriptor.prog_id,
(f) enforce per-stage scratch quota (14.7),
(g) interpret action and enqueue the descriptor into exactly one output category (default out, redirect out, free, tx, abort-to-free),
(h) perform stable routing partition: for each output ring, preserve the relative order of descriptors destined to that ring as in the input order,
(i) for each output ring, perform at most one push_batch in this iteration,
(j) flush counters using the mandatory aggregation rule (16.5).
14.1.2 A stage MUST NOT perform per-descriptor immediate push calls that interleave with processing, except as part of building per-output batches for the final push_batch steps.
14.1.3 A stage MUST NOT block on any ring except its single input DataRing wait strategy. All output pushes MUST be non-blocking; on Full, the stage MUST apply the mandatory backpressure policy described in 14.9.

14.2 Mandatory read-only tables (ProgramTable and RouteTable)
14.2.1 Each stage MUST have a ProgramTable fd mapped PROT_READ that provides the set of programs usable by prog_id.
14.2.2 Each stage MUST have a RouteTable fd mapped PROT_READ that defines:
(a) default output ring id,
(b) mapping from qid to output ring id.
14.2.3 The exact ABIs of ProgramTable and RouteTable are fixed in 14.10 and 14.11.

14.3 Publish and visibility contract (mandatory)
14.3.1 The descriptor transfer on a ShmRing is the sole publish point for object bytes for SDPR.
14.3.2 Writer contract (mandatory):
(a) Writer MUST fully initialize object header and object data/scratch bytes using plain stores.
(b) Writer MUST write descriptor fields matching object header (cookie, data_len, scratch_len).
(c) Writer MUST publish the descriptor by pushing it to a ring using a ShmRing push operation that performs a Release publish of the slot (head.store Release).
14.3.3 Reader contract (mandatory):
(a) Reader MUST pop the descriptor using a ShmRing pop operation that performs an Acquire consume of the slot (head.load Acquire before slot read).
(b) Only after successful pop, reader MAY read the object header/bytes, and MUST validate per 8.3 before dereferencing regions.
14.3.4 Implementations MUST NOT introduce any additional publish point for object bytes outside 14.3.2/14.3.3.

14.4 Event emission (mandatory)
14.4.1 Stages MUST emit STAGE_IN and STAGE_OUT on DebugRing under sampling policy.
14.4.2 Stages MUST attempt 100% best-effort emission for exceptions on ExceptionRing (section 12).
14.4.3 Stages MUST increment event counters for emits/drops.

14.5 Mandatory batching and wait strategy
14.5.1 batch_max MUST be bounded and MUST be documented; typical values are 32..256.
14.5.2 Batching MUST preserve ordering within the input ring.
14.5.3 A stage MUST implement stable partitioning into per-output batches and MUST issue at most one push_batch per output ring per iteration.
14.5.4 A stage MUST implement a two-mode wait (poll/spin and sleep) with hysteresis:
(a) poll/spin mode when recent average batch size is high,
(b) sleep mode (futex-based blocking pop) when the ring is frequently empty.
14.5.5 Poll/spin mode MUST be bounded by poll_iters_max; if exceeded without progress, the stage MUST transition to sleep mode.
14.5.6 Sleep mode MUST use spin-then-futex with bounded spin_iters and MUST use the ShmRing pop_blocking timeout budget rules.

14.6 Per-stage rate limiting (mandatory behavior)
14.6.1 Purpose: best-effort DoS prevention per stage input without blocking the data plane.
14.6.2 Configuration is provided via StatsPage fields rate_limit_pps and rate_limit_burst (16.3.5) and applied via the policy_epoch protocol (16.3.9).
14.6.3 If rate_limit_pps == 0, rate limiting is disabled.
14.6.4 A compliant implementation MUST implement a token-bucket limiter per stage input:
(a) tokens refilled at rate_limit_pps tokens per second,
(b) bucket capacity is rate_limit_burst tokens (if 0, treat as rate_limit_pps),
(c) each accepted descriptor consumes 1 token.
14.6.5 Time base:
14.6.5.1 The limiter MUST use a monotonic time base in nanoseconds consistent with 4.5.
14.6.5.2 The limiter MUST NOT introduce syscalls on steady-state hot paths.
14.6.6 On limit exceed, the stage MUST:
(a) increment rate_limit_drop_count,
(b) synthesize a policy DROP by pushing to FreeRing,
(c) emit ExceptionRing event reason=DROP aux=1 best-effort,
(d) continue.

14.7 Per-stage scratch quota (mandatory behavior)
14.7.1 Purpose: bound scratch usage per object at each stage to prevent unbounded scratch growth.
14.7.2 Configuration is provided via StatsPage field stage_scratch_max (16.3.5) and applied via the policy_epoch protocol (16.3.9).
14.7.3 If stage_scratch_max == 0, the effective maximum is (obj_cap - 64 - data_len) for that object.
14.7.4 After MicroVM execution and before routing/TX, the stage MUST validate:
scratch_len <= effective_stage_scratch_max.
14.7.5 On violation, the stage MUST:
(a) synthesize ABORT(VM_ILLEGAL),
(b) increment scratch_quota_drop_count and abort_count and vm_code_illegal,
(c) emit ExceptionRing event reason=VM_ABORT aux=2 best-effort,
(d) push the descriptor to FreeRing, and MUST NOT TX.

14.8 Routing rules
14.8.1 PASS: enqueue to default output ring.
14.8.2 REDIRECT(qid): qid MUST be mapped by RouteTable; unknown qid MUST be treated as ABORT(VM_ILLEGAL) and MUST emit ROUTE_MISS on ExceptionRing.
14.8.3 DROP: enqueue to FreeRing.
14.8.4 ABORT: enqueue to FreeRing (ABORT code recorded); MUST emit VM_ABORT on ExceptionRing.
14.8.5 TX: stage MUST enforce TX stripping (7.4.1) and enqueue to TXRing; MUST emit TX on DebugRing under sampling and MUST emit TX on ExceptionRing for any TX_FAIL produced by TX stage.

14.9 Output Full handling (mandatory, non-blocking)
14.9.1 Output push operations MUST be non-blocking. If any output ring push_batch would return Full, the stage MUST apply the following fixed policy:
(a) For PASS/REDIRECT outputs: convert those descriptors to DROP by pushing them to FreeRing, and increment drop_count and a per-reason counter route_full_drop_count.
(b) For TXRing: convert to ABORT(VM_ILLEGAL) and push to FreeRing, and emit TX_FAIL aux=1 (soft) on ExceptionRing.
14.9.2 This policy MUST be applied deterministically based only on the observed Full result and must not block.

14.10 ProgramTable v0.0.1 ABI (mandatory, read-only)
14.10.1 ProgramTable is a read-only byte array mapped PROT_READ with the following layout:
14.10.2 Header (64 bytes, little-endian):
0x00 magic: u32 = 0x4C425250 (“PRBL” little-endian)
0x04 version_major: u16 = 0
0x06 version_minor: u16 = 1
0x08 entry_count: u32
0x0C entry_stride: u32 (MUST be 32)
0x10 entries_off: u64 (MUST be 64)
0x18 bytes_len: u64 (total mapping length)
0x20 reserved: [u8; 32] (MUST be 0)
14.10.3 Entry (32 bytes, repeated entry_count times):
0x00 prog_id: u16
0x02 reserved0: u16 (MUST be 0)
0x04 byte_off: u64
0x0C byte_len: u32
0x10 code_crc32: u32 (CRC32 of program bytes)
0x14 reserved1: [u8; 12] (MUST be 0)
14.10.4 Validation (mandatory):
(a) header fields MUST match constants,
(b) entry_count MUST be > 0,
(c) each entry byte_off and byte_len MUST be within bytes_len and aligned to 8,
(d) prog_id MUST be unique,
(e) program bytes at (byte_off, byte_len) MUST contain a valid MicroVM program container (13.2) and MUST pass verifier,
(f) code_crc32 MUST match the bytes.
14.10.5 A stage MUST reject pipeline start if ProgramTable validation fails.

14.11 RouteTable v0.0.1 ABI (mandatory, read-only)
14.11.1 RouteTable is a read-only byte array mapped PROT_READ with the following layout:
14.11.2 Header (64 bytes, little-endian):
0x00 magic: u32 = 0x54425552 (“RUBT” little-endian)
0x04 version_major: u16 = 0
0x06 version_minor: u16 = 1
0x08 qid_count: u32
0x0C default_qid: u32
0x10 map_off: u64 (MUST be 64)
0x18 bytes_len: u64
0x20 reserved: [u8; 32] (MUST be 0)
14.11.3 Map (qid_count entries, each u16 ring_id):
ring_id[qid] at map_off + 2qid
14.11.4 Validation (mandatory):
(a) header fields MUST match constants,
(b) qid_count MUST be in [1, 4096],
(c) default_qid MUST be < qid_count,
(d) map_off + 2qid_count MUST be <= bytes_len,
(e) reserved bytes MUST be 0.
14.11.5 Unknown qid is not possible if table is valid; however, if MicroVM returns a qid >= qid_count, stage MUST treat as ROUTE_MISS and ABORT(VM_ILLEGAL).
	15.	Allocation, reclaim, TX stage, and generation cookies
15.1 Status
15.1.1 v0.0.1 requires a reclaim path that returns freed objects to a free pool and increments object cookies.
15.1.2 v0.0.1 defines a minimal allocator/reclaimer ring protocol using FreeRing and AllocRing ABIs. Implementations MUST implement the protocol semantics.
15.1.3 Topology constraints in 15.7 are mandatory and are intended to prevent SPSC breakage and reclaim bottlenecks.
15.1.4 v0.0.1 defines TX as a dedicated TX stage with mandatory ownership and completion semantics (15.10).

15.2 Roles and invariants (mandatory)
15.2.1 The Reclaimer is the only component permitted to increment object cookies and to publish a descriptor for reuse on AllocRing.
15.2.2 Stages MUST NOT modify object header cookie.
15.2.3 An object is considered free for reuse only after Reclaimer increments cookie and publishes a fresh descriptor on AllocRing.

15.3 FreeRing protocol (mandatory)
15.3.1 FreeRing is SPSC. Producer is exactly one stage (or exactly one TX stage). Consumer is the Reclaimer (or one Reclaimer shard worker; see 15.8).
15.3.2 On DROP or ABORT, a stage MUST push the descriptor to its FreeRing.
15.3.3 After pushing to FreeRing, producer MUST NOT access the object bytes.
15.3.4 TX stage MUST push the descriptor to its own TXFreeRing (which is also a FreeRing for reclaim purposes).

15.4 AllocRing protocol (mandatory)
15.4.1 AllocRing is SPSC. Producer is the Reclaimer (or one Reclaimer shard worker). Consumer is exactly one Ingress/Producer component that fills objects.
15.4.2 Consumers MUST treat descriptors from AllocRing as owning a free object and MAY write object bytes/header before publishing the descriptor into a DataRing.

15.5 Reclaimer behavior (mandatory)
15.5.1 Reclaimer MUST pop descriptors from each FreeRing and perform:
(a) Validate descriptor object bounds (8.3.1-8.3.5) before any object access.
(b) Read object header (first 64 bytes) and validate header.magic/version/flags/reserved (7.1.2, 7.2, 7.1.3).
(c) Compare header.cookie with descriptor.cookie. If mismatch, Reclaimer MUST count stale_desc and MUST NOT reuse based on mismatched cookie; it MAY drop the descriptor.
(d) On match, Reclaimer MUST increment header.cookie by 1 (wrapping u64 permitted).
(e) Reclaimer MUST reset object header fields for reuse:
data_len=0, scratch_len=0, flags=0, reserved0=0, reserved1=all zero.
(f) Reclaimer MUST create a fresh descriptor for allocation:
obj_off, obj_cap set, data_len=0, scratch_len=0, route_hint=0, prog_id=0, cookie=new_cookie, trace_id=0, reserved0=all zero.
(g) Reclaimer MUST publish the fresh descriptor by pushing it to an AllocRing.
15.5.2 Reclaimer MUST follow the publish and visibility contract (14.3) when publishing AllocRing descriptors.
15.5.3 Reclaimer MUST operate as a two-phase batched loop:
Phase 1 (Intake): pop_batch from FreeRings and perform descriptor-only validation.
Phase 2 (Reset+Publish): perform header cookie/reset and push_batch to AllocRings.
15.5.4 AllocRing publish MUST use push_batch with a single head publish per batch.

15.6 Cookie (generation) rules (mandatory)
15.6.1 Each time an object is returned to the free pool and accepted by the Reclaimer, its cookie MUST be incremented (wrapping u64 permitted).
15.6.2 Newly allocated descriptors MUST carry the current object cookie.
15.6.3 Cookie mismatch MUST be treated as StaleDescriptor and MUST NOT allow object access.
15.6.4 Cookie wrap handling (mandatory):
If Reclaimer increments a cookie and the result equals 0 (wrap observed), Reclaimer MUST initiate pipeline shutdown best-effort and MUST stop reusing objects, and MUST increment cookie_wrap_count.

15.7 Topology rules (mandatory; anti-pattern kill)
15.7.1 SPSC preservation rule
15.7.1.1 Any ring used as FreeRing, AllocRing, DataRing, DebugRing, ExceptionRing, TXRing, or TXFreeRing MUST remain strictly SPSC at runtime:
	•	exactly one producer attachment and exactly one consumer attachment.
15.7.1.2 Shared FreeRing with multiple stage producers is non-compliant.
15.7.1.3 Shared AllocRing with multiple allocator consumers is non-compliant.
15.7.1.4 Shared DebugRing or ExceptionRing among multiple stages is non-compliant.
15.7.2 Per-producer FreeRings (mandatory)
15.7.2.1 Each stage that can DROP/ABORT and the TX stage MUST have its own dedicated FreeRing.
15.7.2.2 A producer MUST push reclaimed descriptors only to its own FreeRing.
15.7.3 Per-consumer AllocRings (mandatory)
15.7.3.1 Each ingress/producer that requests free objects MUST have its own dedicated AllocRing.
15.7.3.2 A consumer MUST pop free descriptors only from its own AllocRing.
15.7.4 Reclaimer polling
15.7.4.1 Reclaimer MUST be permitted to poll multiple FreeRings without blocking waits on multiple rings simultaneously.
15.7.4.2 Reclaimer MUST implement a fair polling strategy (e.g., round-robin) to avoid starving a FreeRing.
15.7.5 Enforcement
15.7.5.1 The attach uniqueness rules in ShmRing (10.11) are the enforcement mechanism. Any attempt to attach more than one producer or consumer MUST fail.

15.8 Reclaim scalability rules (mandatory)
15.8.1 Sharded reclaimers are permitted. If multiple Reclaimer workers are used:
15.8.1.1 Each object MUST be assigned to exactly one Reclaimer shard for cookie increment responsibility (single-writer per object cookie).
15.8.1.2 Sharding scheme MUST be deterministic (e.g., shard = (obj_off >> K) & (shard_count-1)).
15.8.2 If sharded reclaimers are used, FreeRings and AllocRings MAY be sharded accordingly, provided each ring remains SPSC.

15.9 Alloc starvation behavior (mandatory)
15.9.1 If AllocRing is empty, an ingress MUST NOT block the entire pipeline. The ingress MUST either:
(a) backpressure locally (sleep) or
(b) drop inputs.
The chosen behavior MUST be documented.
15.9.2 Any drops due to allocator starvation MUST be observable (counter and ExceptionRing event reason=DROP aux=2).

15.10 TX stage contract (mandatory)
15.10.1 TX is implemented as a dedicated TX stage with:
(a) exactly one input TXRing (consumer),
(b) exactly one output TXFreeRing (producer) which is treated as that TX stage’s FreeRing,
(c) one StatsPage and one ExceptionRing (stage_id unique).
15.10.2 Ownership:
When an upstream stage pushes a descriptor to TXRing, ownership transfers to the TX stage. Upstream MUST NOT access the object thereafter.
15.10.3 Submission:
TX stage MUST transmit exactly data_len bytes and MUST ensure scratch is not transmitted.
15.10.4 Completion:
On TX completion, regardless of success or failure, TX stage MUST push the descriptor to TXFreeRing.
15.10.5 Failure reporting:
On TX failure, TX stage MUST:
(a) increment tx_fail_count and a classified counter (tx_fail_soft or tx_fail_hard),
(b) emit ExceptionRing event reason=TX_FAIL aux=1 for soft or aux=2 for hard.
15.10.6 Retry:
TX stage MUST NOT perform automatic retries in v0.0.1.
	16.	Observability: counters, policy configuration, and audit triggers
16.1 Per-stage counters (required minimum set)
16.1.1 in_count, out_count, drop_count, tx_count, abort_count
16.1.2 redirect_count[qid]
16.1.3 vm_code_* counters by ABORT/trap code (bounds, align, stack, illegal, timeout, trap)
16.1.4 stale_desc_count, invalid_desc_count, route_miss_count, corrupt_ring_count
16.1.5 rate_limit_drop_count, scratch_quota_drop_count, route_full_drop_count, alloc_starvation_drop_count
16.1.6 debug_emit_count, debug_drop_count
16.1.7 exception_emit_count, exception_drop_count
16.1.8 futex_wait_ne, futex_wake_ne, futex_wait_nf, futex_wake_nf (from ShmRing events)
16.1.9 tx_fail_count, tx_fail_soft, tx_fail_hard (TX stage only)
16.1.10 cookie_wrap_count (reclaimer stage only)

16.2 Counter semantics (mandatory)
16.2.1 All counters are monotonically non-decreasing u64 (wrapping permitted).
16.2.2 Counter increments MUST NOT block the data plane.
16.2.3 Counter reads by collectors are best-effort and MAY observe intermediate values.
16.2.4 Optimized builds MUST implement counter aggregation (16.5) and MUST NOT perform per-packet atomic increments for required minimum counters in steady state.
16.2.5 Writers MUST update StatsPage counters using lock-free atomic u64 operations (fetch_add or atomic store).
16.2.6 Readers MUST read counters using atomic loads (Relaxed permitted).

16.3 SDPR.StatsPage v0.0.1 ABI (mandatory)
16.3.1 Each stage MUST expose exactly one StatsPage mapping.
16.3.2 StatsPage size is fixed: 4096 bytes.
16.3.3 StatsPage MUST be aligned to 4096 bytes and mapped MAP_SHARED.
16.3.4 StatsPage fields are little-endian and fixed at the offsets below. All unspecified bytes MUST be zero.
16.3.5 StatsPage layout (offsets from page base):
0x000 magic: u32 = 0x54535453 (“SSTT” little-endian)
0x004 version_major: u16 = 0
0x006 version_minor: u16 = 1
0x008 stage_id: u16
0x00A reserved0: u16 (MUST be 0)
0x00C reserved1: u32 (MUST be 0)
0x010 config_flags: u64 (reserved, MUST be 0 in v0.0.1)
0x018 sample_mask: u64 (debug sampling mask; 11.7.3)
0x020 rate_limit_pps: u64 (0=disabled)
0x028 rate_limit_burst: u64 (0=use rate_limit_pps)
0x030 stage_scratch_max: u32 (0=no additional restriction beyond 7.3.4)
0x034 reserved2: u32 (MUST be 0)
0x038 invalid_desc_alert_per_sec: u64 (0=disabled; used by collector)
0x040 stale_desc_alert_per_sec: u64 (0=disabled; used by collector)
0x048 corrupt_ring_alert_per_min: u64 (0=disabled; used by collector)
0x050 policy_epoch: u64 (monotonic; policy update commit; 16.3.9)
0x058 uptime_ns: u64
0x060 in_count: u64
0x068 out_count: u64
0x070 drop_count: u64
0x078 tx_count: u64
0x080 abort_count: u64
0x088 stale_desc_count: u64
0x090 invalid_desc_count: u64
0x098 route_miss_count: u64
0x0A0 corrupt_ring_count: u64
0x0A8 rate_limit_drop_count: u64
0x0B0 scratch_quota_drop_count: u64
0x0B8 route_full_drop_count: u64
0x0C0 alloc_starvation_drop_count: u64
0x0C8 debug_emit_count: u64
0x0D0 debug_drop_count: u64
0x0D8 exception_emit_count: u64
0x0E0 exception_drop_count: u64
0x0E8 vm_code_bounds: u64
0x0F0 vm_code_align: u64
0x0F8 vm_code_stack: u64
0x100 vm_code_illegal: u64
0x108 vm_code_timeout: u64
0x110 vm_code_trap: u64
0x118 futex_wait_ne: u64
0x120 futex_wake_ne: u64
0x128 futex_wait_nf: u64
0x130 futex_wake_nf: u64
0x138 tx_fail_count: u64
0x140 tx_fail_soft: u64
0x148 tx_fail_hard: u64
0x150 cookie_wrap_count: u64
0x158 redirect_base: u64 (fixed to 0x200)
0x160 redirect_stride: u64 (fixed to 8)
0x168 redirect_count_len: u64
0x170 reserved3: [u8; 144] (MUST be 0)
0x200 redirect_count[0]: u64
0x208 redirect_count[1]: u64
… continues by stride=8 up to redirect_count_len
16.3.6 redirect_count_len MUST be <= 480 in v0.0.1.
16.3.7 Writers SHOULD update counters via aggregation flush (16.5). Readers MUST tolerate best-effort values.
16.3.8 Policy configuration fields MAY be written by the control plane before stage start. After start, policy updates MUST follow 16.3.9.
16.3.9 Mandatory policy update protocol (single protocol)
16.3.9.1 Control plane update:
(a) write one or more policy fields (sample_mask, rate_limit_pps, rate_limit_burst, stage_scratch_max) using plain stores or atomic stores,
(b) commit by incrementing policy_epoch using fetch_add(1, Release).
16.3.9.2 Stage consumption:
(a) stage MUST maintain a local cached copy of policy fields,
(b) stage MUST read policy_epoch with Acquire only at batch boundaries (or at a fixed bounded time interval) and MUST update the local cache only when policy_epoch changes,
(c) hot path MUST use only the local cache (no per-packet shared loads).

16.4 Audit and alert triggers (normative for collectors)
16.4.1 A collector SHOULD monitor per-stage StatsPage counters and compare rates against configured thresholds:
invalid_desc_alert_per_sec, stale_desc_alert_per_sec, corrupt_ring_alert_per_min.
16.4.2 If a threshold is non-zero and the corresponding observed rate exceeds it, the collector SHOULD emit an alert to an external audit/monitoring sink.
16.4.3 Alerts MUST NOT block the data plane.
16.4.4 Collectors SHOULD include enough context for forensics (stage_id, recent ExceptionRing events best-effort, and counts).

16.5 Counter locality and update rules (mandatory; anti-pattern kill)
16.5.1 Single-writer rule
16.5.1.1 Only the stage worker that owns the StatsPage MAY write counter fields.
16.5.1.2 Control plane and collectors MUST NOT write counters.
16.5.2 Flush-only requirement
16.5.2.1 Stages MUST maintain private shadow counters and periodically flush deltas to StatsPage using atomic fetch_add.
16.5.2.2 Flush frequency MUST be bounded by (N packets) OR (T time), both of which MUST be documented.
16.5.2.3 Stages MUST NOT perform per-packet atomic increments for required minimum counters in steady state.
16.5.3 False sharing avoidance
16.5.3.1 Implementations SHOULD arrange private shadow counters to avoid false sharing (per-thread private memory).
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
ClockGettime
	18.	Performance rules and SLOs (mandatory for claiming SLO compliance)
18.1 No syscalls on steady-state hot paths when progress is possible and no empty/full transition occurs. Syscalls are permitted only for futex_wait in blocking paths and futex_wake on mandated transition/wake-all events. Time sources must satisfy 4.5.
18.2 No CAS and no fetch_add on ShmRing head/tail during push/pop.
18.3 head, tail, doorbells MUST be on distinct cache lines (ShmRing fixed offsets).
18.4 ShmRing futex_wake count MUST be 1 for normal operation; INT_MAX only for shutdown/close wake-all.
18.5 MicroVM MUST enforce an instruction budget and MUST trap on budget exceed (VM_TIMEOUT).
18.6 Event emission MUST NOT block and MUST be bounded work per packet; DebugRing non-exception events MUST be buffered and flushed in bursts; ExceptionRing attempted immediately best-effort.
18.7 StatsPage counter updates MUST be aggregated flush (16.5) for SLO compliance.
18.8 Rate limiting MUST NOT introduce syscalls on steady-state hot paths and MUST be bounded work per packet.
18.9 Slot size MUST be 128 for all SDPR rings (10.18).
18.10 Stage loop MUST use stable partitioning and at most one push_batch per output ring per iteration (14.1, 14.5).
18.11 Fail-closed: any CorruptSlot observed on any SDPR ring MUST cause stage to emit CORRUPT_RING on ExceptionRing and initiate pipeline shutdown best-effort.

18.12 SLO measurement conditions (mandatory)
18.12.1 Benchmarks MUST pin stage and ring participants to dedicated CPU cores and MUST document CPU model, frequency policy, and NUMA placement.
18.12.2 Benchmarks MUST report packet size (descriptor-only or data bytes) and MUST specify whether debug sampling, exception capture, and rate limiting are enabled.
18.12.3 Benchmarks MUST document the time source used (4.5) and whether it is syscall-free on the hot path.

18.13 Latency measurement points (mandatory)
18.13.1 t_in is defined as the instant when the consumer has acquired (Acquire) head and has logically claimed a descriptor in its local batch (after pop_batch copies the descriptor bytes).
18.13.2 t_out is defined as the instant when the producer publishes (Release) head for the output ring push_batch that includes that descriptor.
18.13.3 stage-hop latency = t_out - t_in for each descriptor.
18.13.4 Batch measurement MUST attribute t_in per descriptor (not only per batch).

18.14 Baseline SLOs (mandatory for implementations claiming SLO compliance)
18.14.1 Empty pipeline (single stage, validate+PASS, debug sampling enabled, exception ring enabled, rate limit off, stats flush aggregated, slot_size=128):
	•	throughput >= 12 Mpps per core (64B descriptors)
	•	p50 stage-hop latency <= 1.0 us
	•	p99 stage-hop latency <= 4.0 us
	•	p99.9 stage-hop latency <= 8.0 us
18.14.2 Typical pipeline (single stage, validate + MicroVM with insn_budget_max<=4096, routing, stats aggregated, debug sampling enabled, exception ring enabled, rate limit optional):
	•	throughput >= 6 Mpps per core (64B descriptors)
	•	p50 stage-hop latency <= 2.0 us
	•	p99 stage-hop latency <= 10.0 us
	•	p99.9 stage-hop latency <= 20.0 us

18.15 Structural performance invariants (mandatory to report)
18.15.1 wake_per_packet = (futex_wake_ne + futex_wake_nf) / max(1, in_count)
SLO-compliant builds SHOULD achieve wake_per_packet <= 1/1024 under sustained load.
18.15.2 syscall_per_packet (excluding blocking waits) MUST be 0 in steady state.
18.15.3 stats_atomic_ops_per_packet SHOULD be <= 1/256 in steady state due to flush-only aggregation (16.5).
	19.	Test requirements (minimum; expanded)
19.1 ShmRing threaded SPSC test for at least 10 million ops.
19.2 ShmRing cross-process test mapping the same fd.
19.3 ShmRing attach validation tests:
magic/version mismatch reject
reserved bytes non-zero reject
layout overflow reject
slot_size constraints for SDPR rings (must reject non-128)
19.4 ShmRing corruption injection: force used > cap and assert:
CorruptIndices returned
SHUTDOWN set
both doorbells wake-all executed
19.5 Slot corruption tests:
reserved field non-zero -> CorruptSlot
sflags reserved bit set -> CorruptSlot
SDPR stage must treat as CORRUPT_RING and initiate shutdown
19.6 Descriptor and object validation tests:
obj_off alignment and bounds
data_len/scratch_len bounds
reserved0 non-zero -> InvalidDescriptor
header magic/version/flags reserved -> InvalidDescriptor
cookie mismatch -> StaleDescriptor
descriptor/header len mismatch -> InvalidDescriptor
19.7 MicroVM verifier/runtime tests:
out-of-bounds access rejected or trapped
backward jump rejected by verifier
stack overflow traps
CALL rejected by verifier and traps to VM_ILLEGAL if bypassed
insn budget traps to VM_TIMEOUT
TRAP traps to VM_TRAP and emits exception event
definite init: read-before-write of regs/stack rejected
19.8 Pipeline tests:
PASS and REDIRECT routing to correct rings
DROP pushes descriptor to FreeRing
ABORT records error and does not TX
TAILCALL uses u16 prog_id and rejects non-zero upper bits
stable partitioning preserves per-output relative order
one push_batch per output ring per iteration
19.9 Event ring tests:
event size is 64, slot len matches
full ring causes drops and increments corresponding drop counters
ExceptionRing emits 100% best-effort for required reasons
DebugRing sampling deterministic by sample_mask
19.10 StatsPage tests:
magic/version fixed
counters monotonic
redirect_count_len bounds
uptime_ns monotonic
policy_epoch protocol works (update then commit, stage observes and updates cache)
19.11 Rate limiter tests:
configured pps limits accepted rate approximately
over-limit causes drop to FreeRing and increments rate_limit_drop_count and emits exception DROP aux=1
19.12 Scratch quota tests:
stage_scratch_max enforced
violation causes ABORT(VM_ILLEGAL), increments scratch_quota_drop_count and vm_code_illegal, emits exception VM_ABORT aux=2, and does not TX
19.13 Reclaimer protocol tests:
only Reclaimer increments cookie
cookie increments on reclaim
fresh descriptors published on AllocRing carry new cookie
two-phase batching behavior
cookie wrap triggers shutdown and increments cookie_wrap_count
19.14 TX stage tests:
TX stage ownership transfer on TXRing
scratch stripped before submit
completion returns to TXFreeRing regardless of success/failure
failure emits TX_FAIL and increments counters
no automatic retry
19.15 Shutdown/close tests for ShmRing wake-all behavior (close cannot be missed).
19.16 Wrap-around test for ShmRing head/tail across u64::MAX (used <= cap maintained).
19.17 Batch operation tests:
push_batch all-or-nothing correctness
pop_batch partial correctness
single wake on transitions
19.18 ShmRing memory ordering litmus tests (mandatory)
19.18.1 Producer writes payload then head Release; consumer head Acquire then reads payload; assert no stale/uninitialized reads over at least 1e8 iterations across processes.
19.18.2 VALID misuse test: randomly flip VALID bits; correctness MUST be unaffected.
19.19 Futex adversarial tests (mandatory)
19.19.1 Inject EINTR/EAGAIN on futex_wait; operations MUST recover by user-space recheck.
19.19.2 Timeout budget test: spurious returns MUST NOT extend total timeout.
19.20 Wake discipline tests (mandatory)
19.20.1 Under continuous non-empty operation, futex_wake_ne count MUST equal the number of empty->non-empty transitions.
19.21 Attach uniqueness race tests (mandatory)
19.21.1 Two producers racing attach_producer: exactly one succeeds, one fails AlreadyAttached.
19.21.2 Two consumers racing attach_consumer: exactly one succeeds, one fails AlreadyAttached.
19.22 Topology tests (mandatory)
19.22.1 Attempt to share a FreeRing among two stage producers MUST fail attachment.
19.22.2 Attempt to share an AllocRing among two consumers MUST fail attachment.
19.22.3 Attempt to share a DebugRing or ExceptionRing among two stages MUST fail attachment.
19.22.4 Reclaimer polling multiple FreeRings MUST make progress and MUST NOT starve any ring under sustained load (best-effort fairness test).
19.23 Runtime safety with verifier bypass (mandatory)
19.23.1 Execute deliberately invalid bytecode by bypassing verifier; runtime MUST trap to ABORT(VM_ILLEGAL/VM_BOUNDS/VM_ALIGN/VM_STACK/VM_TRAP) as appropriate.
19.24 Perf regression structure tests (mandatory if claiming SLO compliance)
19.24.1 Measure wake_per_packet and syscall_per_packet (18.15) under sustained load and assert reported values meet documented thresholds.
	20.	Compliance checklist
20.1 Implements ShmRing ABI exactly and validates on attach; enforces SDPR slot payload rules for all SDPR ring types; enforces slot_size==128 for SDPR rings.
20.2 Implements ShmRing Acquire/Release orderings and futex protocol exactly as required; conforms to SDPR publish and visibility contract (14.3); does not use forbidden sync patterns (10.10.6).
20.3 Enforces single-owner baton invariant for descriptors and objects; forbids post-transfer access.
20.4 Implements SDPR.Object and SDPR.Descriptor ABIs exactly; validates reserved bytes; enforces object header magic/version/flags/reserved and descriptor/header len consistency.
20.5 Enforces cookie generation checks to detect stale descriptors and prevent object access; implements cookie wrap shutdown rule.
20.6 MicroVM verifier enforces memory safety, loop-free control flow, definite initialization rules, and rejects CALL and any bytes beyond instruction array; runtime enforces bounds/alignment regardless of verifier; enforces insn budget; TRAP instruction traps to VM_TRAP.
20.7 TX path strips scratch (scratch_len=0) before submission; TX is a dedicated TX stage with mandatory completion semantics.
20.8 DebugRing and ExceptionRing are mandatory per stage SPSC; emission is non-blocking; drops are counted; DebugRing sampling deterministic by sample_mask; ExceptionRing attempts 100% best-effort for required exception reasons.
20.9 StatsPage is mandatory:
StatsPage ABI is fixed 4096 bytes with policy_epoch protocol
counters include required minimum set
counter updates are non-blocking and use flush-only aggregation
20.10 Rate limiting conforms to 14.6 and does not block or introduce syscalls on steady-state hot paths.
20.11 Scratch quota enforcement conforms to 14.7 and never TXes on violation.
20.12 Stage setup rejects cycles (3.3.3).
20.13 Reclaim protocol is enforced:
Reclaimer is the only cookie incrementer
topology rules in 15.7 are enforced (per-producer FreeRing, per-consumer AllocRing, per-stage event rings)
fresh descriptors are published via AllocRing after header reset
20.14 Stage execution model is mandatory:
batch loop
stable partitioning per output ring
at most one push_batch per output ring per iteration
mandatory wait strategy
20.15 Implementations claiming SLO compliance satisfy section 18 measurement rules and baseline SLOs.

End of SDPR Spec v0.0.1
