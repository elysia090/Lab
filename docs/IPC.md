Title: ShmSpscFutexQueue Spec v0.0.1 (Linux, Rust, ASCII)
	0.	Status
0.1 This document defines a frozen ABI and required behavior for a shared-memory single-producer single-consumer (SPSC) bounded queue with futex-based blocking.
0.2 Target: Linux userspace, Rust implementation, inter-process IPC on one host.
0.3 Normative keywords: MUST, MUST NOT, SHOULD, MAY.
0.4 Version policy (fixed): Implementations claiming compliance with v0.0.1 MUST implement exactly the ABI and semantics in this document. No forward-compat behavior is defined in v0.0.1.
	1.	Scope
1.1 This spec defines:
1.1.1 A bounded ring of fixed-size slots in MAP_SHARED memory.
1.1.2 A slot encoding carrying variable-length payloads up to payload_capacity within each slot.
1.1.3 Futex-based blocking for empty (consumer) and optional full (producer).
1.1.4 Close and shutdown semantics.
1.2 Non-goals
1.2.1 MPMC, fairness, or starvation guarantees.
1.2.2 Cross-host transport.
1.2.3 Security isolation beyond OS-level shared-memory permissions.
1.2.4 Kernel-bypass waiting. futex requires syscalls when sleeping/waking.
1.2.5 Crash-recovery or automatic attachment stealing.
	2.	Platform requirements
2.1 OS and futex
2.1.1 Linux kernel providing futex(2) for shared futex words.
2.1.2 Inter-process use MUST use shared futex operations (MUST NOT use FUTEX_PRIVATE variants).
2.1.3 The implementation MUST use FUTEX_WAIT and FUTEX_WAKE only (no *_BITSET, no PI, no requeue) in v0.0.1.
2.2 Architecture
2.2.1 64-bit only: x86_64 or aarch64.
2.2.2 All atomics used in this ABI MUST be lock-free on the target. Implementations MUST fail create/attach if this requirement is not met.
2.3 Cache-line model
2.3.1 Cache line size is treated as 64 bytes for ABI layout and padding rules in v0.0.1. Implementations MUST follow the fixed offsets and cache-line separation requirements regardless of the host microarchitecture.
	3.	Primitive model
3.1 Data plane
3.1.1 Producer writes a slot, then publishes head.
3.1.2 Consumer observes head, reads a slot, then publishes tail.
3.2 Wait plane
3.2.1 doorbell_ne: futex word for not-empty (consumer waits here).
3.2.2 doorbell_nf: futex word for not-full (producer waits here, optional).
3.2.3 Futex waits MUST be done with an expected epoch value read from the doorbell word.
3.2.4 Waiters MUST recheck conditions in user space after any futex return (spurious wakeups allowed).
	4.	Terminology and invariants
4.1 cap: ring capacity in slots, cap = 1 << capacity_pow2.
4.2 head: AtomicU64, monotonically increasing modulo 2^64, counts produced slots.
4.3 tail: AtomicU64, monotonically increasing modulo 2^64, counts consumed slots.
4.4 used: head.wrapping_sub(tail).
4.5 empty: used == 0.
4.6 full: used == cap.
4.7 epoch: a 32-bit counter value stored in doorbell words, used as futex expected values. epoch is a wrapping counter modulo 2^32 (or modulo 2^31 if treated as i32); only equality comparison is relied upon.
4.8 Fundamental correctness invariant (required)
4.8.1 Producer MUST NOT advance head such that used would exceed cap.
4.8.2 Consumer MUST NOT advance tail beyond head (i.e., MUST NOT consume when empty).
	5.	Shared memory creation and mapping
5.1 Creation
5.1.1 Creator obtains an fd via memfd_create or shm_open.
5.1.2 Creator MUST size the object via ftruncate(fd, total_size).
5.1.3 In v0.0.1, total_size MUST equal header_size + ring_bytes, and ring_offset MUST equal header_size.
5.1.4 Creator MUST set arena_offset=0 and arena_bytes=0 in v0.0.1.
5.2 Mapping
5.2.1 Attach MUST mmap(fd, total_size, PROT_READ|PROT_WRITE, MAP_SHARED).
5.2.2 Attach MUST validate header fields before any queue operations.
5.3 Prefault (recommended)
5.3.1 Implementation SHOULD touch all mapped pages once at init to reduce page-fault latency in hot paths.
5.3.2 Implementation MAY mlock if permitted.
	6.	ABI constants (fixed)
6.1 Endianness
6.1.1 All integer fields are little-endian.
6.2 Magic and version
6.2.1 magic: u64 = 0x5348515350534651
6.2.2 version_major: u16 = 0
6.2.3 version_minor: u16 = 1
6.2.4 Attach MUST accept only (version_major == 0 and version_minor == 1).
6.3 Fixed sizes and alignment
6.3.1 header_size is fixed: 0x180 bytes (384).
6.3.2 header_size MUST be a multiple of 64.
6.3.3 ring_offset MUST be 0x180 and MUST be aligned to 64.
6.3.4 head, tail, doorbell_ne, doorbell_nf MUST start at offsets that are multiples of 64 and MUST NOT share a cache line with each other.
	7.	ABI: shared header layout (frozen offsets)
7.1 The header is a fixed binary layout. Implementations MUST treat the mapping as bytes and use fixed offsets below.
7.2 Implementations MAY additionally define a #[repr(C)] Rust struct for convenience, but MUST verify at build-time or runtime that:
7.2.1 size_of(header) == 0x180, and
7.2.2 each field offset matches section 7.4.
7.2.3 If such verification is not present, using a compiler-laid-out struct is non-compliant.
7.3 Reserved bytes and reserved flag bits
7.3.1 All reserved bytes MUST be zero on create and MUST be validated as zero on attach.
7.3.2 All reserved flag bits (7..31) MUST be zero on create and MUST be validated as zero on attach.
7.4 Header offsets (from mapping base)
7.4.1  0x000 magic: u64
7.4.2  0x008 version_major: u16
7.4.3  0x00A version_minor: u16
7.4.4  0x00C header_size: u32 (MUST be 0x180)
7.4.5  0x010 total_size: u64
7.4.6  0x018 ring_offset: u64 (MUST be 0x180)
7.4.7  0x020 ring_bytes: u64
7.4.8  0x028 arena_offset: u64 (MUST be 0 in v0.0.1)
7.4.9  0x030 arena_bytes: u64 (MUST be 0 in v0.0.1)
7.4.10 0x038 capacity_pow2: u8
7.4.11 0x039 reserved0: [u8; 7] (MUST be 0)
7.4.12 0x040 slot_size: u32
7.4.13 0x044 reserved1: u32 (MUST be 0)
7.4.14 0x048 flags: AtomicU32
7.4.15 0x04C reserved2: u32 (MUST be 0)
7.4.16 0x050 producer_pid: AtomicU32 (diagnostic only)
7.4.17 0x054 consumer_pid: AtomicU32 (diagnostic only)
7.4.18 0x058 error_code: AtomicU32 (diagnostic only)
7.4.19 0x05C reserved3: u32 (MUST be 0)
7.4.20 0x060 reserved4: [u8; 32] (MUST be 0)
7.4.21 0x080 head: AtomicU64
7.4.22 0x088 reserved5: [u8; 56] (MUST be 0)
7.4.23 0x0C0 tail: AtomicU64
7.4.24 0x0C8 reserved6: [u8; 56] (MUST be 0)
7.4.25 0x100 doorbell_ne: AtomicI32
7.4.26 0x104 reserved7: [u8; 60] (MUST be 0)
7.4.27 0x140 doorbell_nf: AtomicI32
7.4.28 0x144 reserved8: [u8; 60] (MUST be 0)
7.4.29 Total header bytes: 0x180.
7.5 Flags bitfield (AtomicU32)
7.5.1 bit 0 INITIALIZED
7.5.2 bit 1 PRODUCER_ATTACHED
7.5.3 bit 2 CONSUMER_ATTACHED
7.5.4 bit 3 PRODUCER_CLOSED
7.5.5 bit 4 CONSUMER_CLOSED
7.5.6 bit 5 SHUTDOWN
7.5.7 bit 6 NOT_FULL_ENABLED
7.5.8 bits 7..31 reserved (MUST be 0).
	8.	Ring layout and slot encoding
8.1 Ring
8.1.1 cap = 1 << capacity_pow2.
8.1.2 mask = cap - 1.
8.1.3 slot_stride = slot_size.
8.1.4 ring_bytes MUST equal cap * slot_size.
8.1.5 ring starts at ring_offset (0x180) and spans ring_bytes.
8.2 Slot encoding (mandatory)
8.2.1 Each slot begins with an 8-byte slot header followed by payload bytes.
8.2.2 slot_header at slot offset 0
8.2.2.1 len: u16 (0..=payload_capacity)
8.2.2.2 tag: u16 (application-defined)
8.2.2.3 sflags: u16
8.2.2.3.1 bit 0 VALID (debug-only; producer MAY set)
8.2.2.3.2 bits 1..15 reserved, MUST be 0
8.2.2.4 reserved: u16, MUST be 0
8.2.3 payload_capacity = slot_size - 8.
8.2.4 payload_capacity MUST be <= 65535 (because len is u16).
8.2.5 Payload bytes stored at slot+8..slot+8+len.
8.2.6 Bytes beyond len are unspecified and MUST be ignored by consumer.
8.2.7 VALID MUST NOT be used as the synchronization primitive. Synchronization is exclusively via head publish (Release) and head observe (Acquire).
8.2.8 If VALID is not used, producer MUST write sflags=0 and consumer MUST ignore VALID.
	9.	Indexing, wrap, and corruption rules
9.1 Indices are u64 counters operating modulo 2^64.
9.2 used MUST be computed as head.wrapping_sub(tail).
9.3 empty iff used == 0.
9.4 full iff used == cap.
9.5 CORRUPT if used > cap.
9.6 Wrap behavior
9.6.1 Wrap is permitted. Correctness relies only on invariant 4.8 plus the used>cap corruption rule.
9.6.2 Note: With wrapping_sub, a valid state MUST always satisfy used <= cap even across 2^64 wrap, provided invariants 4.8.1-4.8.2 hold. Therefore, used > cap is always CORRUPT and requires no special-case for wrap.
9.6.3 Implementations MUST use wrapping_add for local head/tail increments.
9.7 On CORRUPT detection, implementation MUST
9.7.1 set flags.SHUTDOWN (best-effort),
9.7.2 wake both doorbells (see 12.9),
9.7.3 return Error::CorruptIndices.
	10.	Memory ordering contract (mandatory)
10.1 Producer publish rule
10.1.1 Producer MUST write slot payload and slot header using plain stores before publishing head.
10.1.2 Producer MUST publish head using store with Release ordering.
10.2 Consumer observe rule
10.2.1 Consumer MUST load head using Acquire ordering before reading any slot contents that are claimed available by head.
10.2.2 Consumer MUST publish tail using store with Release ordering after consuming.
10.3 Full/empty check rules
10.3.1 Producer MUST load tail using Acquire ordering when checking for full (used==cap).
10.3.2 Consumer MUST load head using Acquire ordering when checking for empty (used==0) in blocking paths.
10.4 Doorbell ordering
10.4.1 Doorbell loads/stores MAY be Relaxed.
10.4.2 Correctness MUST rely on Acquire/Release on head/tail plus user-space rechecks.
10.5 Flags and initialization ordering
10.5.1 Creator MUST initialize all header fields, all reserved bytes, the ring bytes, head=0, tail=0, and doorbells=0.
10.5.2 Creator MUST set or clear NOT_FULL_ENABLED before setting INITIALIZED.
10.5.3 Creator MUST set flags.INITIALIZED using a Release operation after initialization is complete.
10.5.4 Blocking loops MUST read flags using Acquire when checking for SHUTDOWN/CLOSED to ensure prompt visibility.
10.5.5 Attach MUST NOT perform operations until INITIALIZED is observed with Acquire.
	11.	Ownership and attachment protocol (mandatory)
11.1 Exactly one Producer may perform push operations.
11.2 Exactly one Consumer may perform pop operations.
11.3 Cross-process uniqueness MUST be enforced via flags bits:
11.3.1 attach_producer MUST set PRODUCER_ATTACHED via compare_exchange on the whole flags word.
11.3.2 attach_consumer MUST set CONSUMER_ATTACHED via compare_exchange on the whole flags word.
11.3.3 CAS success ordering MUST be AcqRel and failure ordering MUST be Acquire.
11.3.4 CAS MUST preserve all other defined flag bits (it MUST only set the relevant ATTACHED bit).
11.3.5 If the relevant bit is already set, attach MUST fail with Error::AlreadyAttached.
11.3.6 v0.0.1 defines no crash-recovery or stealing of attachments. A stale ATTACHED bit requires creating a new queue.
11.4 Only Producer writes head. Only Consumer writes tail.
11.5 Only Consumer waits on doorbell_ne.
11.6 Only Producer waits on doorbell_nf when NOT_FULL_ENABLED.
11.7 PID fields
11.7.1 producer_pid and consumer_pid are diagnostic only and MUST NOT be used for correctness.
11.7.2 Implementations MAY expose them for observability, but MUST NOT implement automatic attachment stealing based on PIDs.
11.7.3 Implementations SHOULD write producer_pid/consumer_pid (Relaxed) after successfully setting the corresponding ATTACHED bit, and MAY clear them (Relaxed) on close. These fields are best-effort and MAY be stale.
	12.	Futex protocol (exact)
12.1 Futex word requirements
12.1.1 doorbell_ne and doorbell_nf are 32-bit aligned i32 in shared memory.
12.1.2 Futex WAIT MUST use expected value equal to an epoch read from the doorbell word.
12.1.3 Waiter MUST read epoch with Relaxed ordering immediately before futex_wait.
12.1.4 The futex address passed to futex_wait/futex_wake MUST be the address of the shared 32-bit doorbell word in the mapping (the same location read/written by doorbell_* operations).
12.2 Spurious wakeups and EINTR
12.2.1 Waiter MUST treat any return from futex_wait as a prompt to recheck conditions in a loop.
12.2.2 EINTR and EAGAIN MUST be handled as spurious returns.
12.3 Timeout model (normative, fixed-budget)
12.3.1 timeout is an optional relative duration. If provided, it is a total budget for the entire blocking call.
12.3.2 Implementations MUST track a deadline and pass the remaining time to each futex_wait.
12.3.3 Spurious returns MUST NOT extend the total wait time budget.
12.3.4 If remaining <= 0, blocking operation MUST return Timeout without calling futex_wait.
12.3.5 If timeout is None, futex_wait MUST be called with NULL timeout.
12.3.6 On ETIMEDOUT, blocking operation MUST return Timeout.
12.3.7 Implementations SHOULD use a monotonic clock source for deadline tracking (e.g., CLOCK_MONOTONIC).
12.4 Not-empty wait (consumer)
12.4.1 Consumer waits only when empty.
12.4.2 pop_blocking(timeout) loop
12.4.2.1 Read h = head.load(Acquire).
12.4.2.2 Read t = tail.load(Relaxed) or tail_local.
12.4.2.3 If h != t, proceed to consume.
12.4.2.4 Read f = flags.load(Acquire).
12.4.2.5 If f has SHUTDOWN, return Shutdown.
12.4.2.6 If f has PRODUCER_CLOSED and h == t, return Closed.
12.4.2.7 Spin phase: for i in 0..spin_iters
12.4.2.7.1 cpu_relax
12.4.2.7.2 recheck h=head.load(Acquire), t as above, if h!=t then proceed
12.4.2.8 epoch = doorbell_ne.load(Relaxed).
12.4.2.9 Recheck empty using fresh h=head.load(Acquire) and t.
12.4.2.10 If still empty, futex_wait(doorbell_ne, epoch, remaining_timeout).
12.4.2.11 If syscall fails (errno not EINTR/EAGAIN/ETIMEDOUT), return Syscall { op=FutexWaitNe, errno }.
12.4.2.12 If timed out, return Timeout.
12.4.2.13 Loop.
12.5 Not-empty wake (producer)
12.5.1 Producer MUST wake consumer only on empty->non-empty transition.
12.5.2 Exact rule
12.5.2.1 Before publishing, producer reads t = tail.load(Acquire).
12.5.2.2 was_empty = (head_local == t).
12.5.2.3 Publish head.store(Release, next_head).
12.5.2.4 If was_empty then
12.5.2.4.1 doorbell_ne.fetch_add(1, Relaxed)
12.5.2.4.2 futex_wake(doorbell_ne, 1); on failure return Syscall { op=FutexWakeNe, errno }.
12.5.3 False-positive wakes are permitted. Implementations MAY wake even when the consumer is not actually waiting. Correctness MUST NOT rely on wake precision.
12.6 Not-full wait (producer, optional)
12.6.1 Enabled iff flags.NOT_FULL_ENABLED is set at init and observed after INITIALIZED.
12.6.2 Producer waits only when full.
12.6.3 push_blocking(timeout) loop
12.6.3.1 Read t = tail.load(Acquire).
12.6.3.2 If head_local.wrapping_sub(t) != cap, proceed to push.
12.6.3.3 Read f = flags.load(Acquire).
12.6.3.4 If f has SHUTDOWN, return Shutdown.
12.6.3.5 If f has CONSUMER_CLOSED, return Closed.
12.6.3.6 Spin phase: for i in 0..spin_iters recheck full (refresh t with Acquire permitted).
12.6.3.7 epoch = doorbell_nf.load(Relaxed).
12.6.3.8 Recheck full using t = tail.load(Acquire).
12.6.3.9 If still full, futex_wait(doorbell_nf, epoch, remaining_timeout).
12.6.3.10 If syscall fails (errno not EINTR/EAGAIN/ETIMEDOUT), return Syscall { op=FutexWaitNf, errno }.
12.6.3.11 If timed out, return Timeout.
12.6.3.12 Loop.
12.7 Not-full wake (consumer, optional)
12.7.1 Consumer MUST wake producer only on full->not-full transition.
12.7.2 Exact rule
12.7.2.1 Before advancing tail, consumer MUST read h = head.load(Acquire).
12.7.2.2 t = tail_local.
12.7.2.3 was_full = (h.wrapping_sub(t) == cap).
12.7.2.4 next_tail = t.wrapping_add(1).
12.7.2.5 Publish tail.store(Release, next_tail).
12.7.2.6 If was_full then
12.7.2.6.1 doorbell_nf.fetch_add(1, Relaxed)
12.7.2.6.2 futex_wake(doorbell_nf, 1); on failure return Syscall { op=FutexWakeNf, errno }.
12.8 Close wake
12.8.1 close_producer MUST wake doorbell_ne with futex_wake(doorbell_ne, INT_MAX).
12.8.2 close_consumer MUST wake doorbell_nf with futex_wake(doorbell_nf, INT_MAX) when NOT_FULL_ENABLED.
12.9 Shutdown wake
12.9.1 shutdown() MUST set flags.SHUTDOWN and wake all potential waiters.
12.9.2 Exact rule
12.9.2.1 flags.fetch_or(SHUTDOWN, Release)
12.9.2.2 doorbell_ne.fetch_add(1, Relaxed); futex_wake(doorbell_ne, INT_MAX)
12.9.2.3 doorbell_nf.fetch_add(1, Relaxed); futex_wake(doorbell_nf, INT_MAX)
12.10 Futex rationale (non-normative but binding for interpretation)
12.10.1 If a wake happens before a waiter enters FUTEX_WAIT, the waiter will observe an epoch mismatch and FUTEX_WAIT will return -EAGAIN, which is treated as a spurious return and triggers a user-space recheck. Therefore, epoch increment (Relaxed) followed by FUTEX_WAKE is sufficient; no additional barriers are required beyond head/tail Acquire/Release and the mandated rechecks.
	13.	Ring operations (exact)
13.1 Producer local state
13.1.1 Producer MUST keep head_local in private memory.
13.1.2 Producer SHOULD keep cached_tail in private memory to reduce shared loads.
13.2 Consumer local state
13.2.1 Consumer MUST keep tail_local in private memory.
13.2.2 Consumer SHOULD keep cached_head in private memory to reduce shared loads.
13.3 try_push(tag, payload)
13.3.1 Preconditions
13.3.1.1 payload.len <= payload_capacity.
13.3.1.2 Producer uniqueness holds.
13.3.2 Algorithm
13.3.2.1 Read f = flags.load(Acquire). If f has SHUTDOWN, return Shutdown. If f has CONSUMER_CLOSED, return Closed.
13.3.2.2 If cached_tail is stale or predicted full, refresh t = tail.load(Acquire), update cached_tail.
13.3.2.3 If head_local.wrapping_sub(t) == cap, return Full.
13.3.2.4 slot = ring_base + ((head_local & mask) * slot_stride).
13.3.2.5 Copy payload into slot+8..slot+8+len.
13.3.2.6 Write slot_header (len, tag, sflags, reserved=0) at slot.
13.3.2.7 next_head = head_local.wrapping_add(1).
13.3.2.8 Publish head.store(Release, next_head).
13.3.2.9 Apply not-empty wake rule if transition was empty.
13.3.2.10 head_local = next_head.
13.4 try_pop(out)
13.4.1 Preconditions
13.4.1.1 Consumer uniqueness holds.
13.4.2 Algorithm
13.4.2.1 If cached_head is stale or predicted empty, refresh h = head.load(Acquire), update cached_head.
13.4.2.2 If cached_head == tail_local:
13.4.2.2.1 Read f = flags.load(Acquire). If f has SHUTDOWN, return Shutdown. If f has PRODUCER_CLOSED, return Closed. Else return Empty.
13.4.2.3 slot = ring_base + ((tail_local & mask) * slot_stride).
13.4.2.4 Read slot_header.
13.4.2.5 If len > payload_capacity, return CorruptSlot.
13.4.2.6 If out.len < len, return OutputTooSmall(required=len).
13.4.2.7 Copy payload into out[0..len].
13.4.2.8 If NOT_FULL_ENABLED is set:
13.4.2.8.1 Read h_now = head.load(Acquire) fresh.
13.4.2.8.2 was_full = (h_now.wrapping_sub(tail_local) == cap).
13.4.2.9 next_tail = tail_local.wrapping_add(1).
13.4.2.10 Publish tail.store(Release, next_tail).
13.4.2.11 Apply not-full wake rule if was_full.
13.4.2.12 tail_local = next_tail.
13.5 Batch operations (optional, semantics defined)
13.5.1 Batch operations MAY be provided.
13.5.2 Batch operations MUST preserve the same visibility guarantees as single push/pop.
13.5.3 Batch operations MUST NOT introduce CAS/RMW on head/tail.
13.5.4 Batch operations MUST NOT call futex when progress is possible (i.e., they are non-blocking).
13.5.5 push_batch(k items) (all-or-nothing)
13.5.5.1 If k==0, return Ok.
13.5.5.2 Read f = flags.load(Acquire). If f has SHUTDOWN, return Shutdown. If f has CONSUMER_CLOSED, return Closed.
13.5.5.3 Read t0 = tail.load(Acquire); was_empty = (head_local == t0).
13.5.5.4 If head_local.wrapping_sub(t0) + k > cap, return Full and MUST NOT publish any of the k items.
13.5.5.5 Write k slots.
13.5.5.6 Publish head once with Release: head.store(Release, head_local + k modulo 2^64).
13.5.5.7 If was_empty, perform exactly one not-empty wake.
13.5.5.8 head_local = head_local + k modulo 2^64.
13.5.5.9 Note: In v0.0.1 (SPSC), exactly one wake is sufficient.
13.5.6 pop_batch(up to k items) (partial success permitted)
13.5.6.1 If k==0, return 0 (or Ok(0) depending on API).
13.5.6.2 If cached_head == tail_local:
13.5.6.2.1 Read f = flags.load(Acquire). If f has SHUTDOWN, return Shutdown (or Ok(0) with Shutdown status). If f has PRODUCER_CLOSED, return Closed (or 0 with Closed status). Else return 0.
13.5.6.3 If NOT_FULL_ENABLED, read h0 = head.load(Acquire) and compute was_full = (h0.wrapping_sub(tail_local) == cap).
13.5.6.4 Pop n available items (0..=k) using the same slot decoding rules as 13.4 (including CorruptSlot and OutputTooSmall).
13.5.6.5 Publish tail once with Release: tail.store(Release, tail_local + n modulo 2^64).
13.5.6.6 If was_full and n>0, perform exactly one not-full wake.
13.5.6.7 If n>0, pop_batch MUST return n regardless of concurrent flag changes; the caller observes Shutdown/Closed on a subsequent operation if applicable. If n==0, pop_batch MUST follow 13.5.6.2 flag handling.
	14.	Close and shutdown semantics
14.1 close_producer()
14.1.1 Sets flags.PRODUCER_CLOSED via fetch_or(Release).
14.1.2 Wakes doorbell_ne with wake-all.
14.1.3 After close_producer, producer MUST NOT push.
14.2 close_consumer()
14.2.1 Sets flags.CONSUMER_CLOSED via fetch_or(Release).
14.2.2 Wakes doorbell_nf with wake-all if NOT_FULL_ENABLED.
14.2.3 After close_consumer, consumer MUST NOT pop.
14.3 Return rules for non-blocking operations
14.3.1 try_pop MUST return Closed if PRODUCER_CLOSED is set and the queue is empty.
14.3.2 try_push MUST return Closed if CONSUMER_CLOSED is set.
14.3.3 Both try_* MUST return Shutdown if SHUTDOWN is set.
14.4 Blocking return rules
14.4.1 pop_blocking returns Closed if empty and PRODUCER_CLOSED is set.
14.4.2 push_blocking returns Closed if full and CONSUMER_CLOSED is set (when NOT_FULL_ENABLED).
14.4.3 Both return Shutdown if SHUTDOWN is set.
	15.	Attach and validation (mandatory)
15.1 Attach MUST validate:
15.1.1 magic matches.
15.1.2 version_major == 0 and version_minor == 1.
15.1.3 header_size == 0x180.
15.1.4 total_size equals mapping size.
15.1.5 ring_offset == 0x180.
15.1.6 ring_bytes within total_size and total_size == header_size + ring_bytes.
15.1.7 slot_size >= 8 and slot_size % 8 == 0.
15.1.8 payload_capacity <= 65535.
15.1.9 capacity_pow2 in [1, 30].
15.1.10 ring_bytes == (1 << capacity_pow2) * slot_size.
15.1.11 arena_offset == 0 and arena_bytes == 0.
15.1.12 all reserved bytes are 0 (reserved0..reserved8).
15.1.13 reserved flag bits are 0.
15.2 If any validation fails, attach MUST return an error and MUST NOT perform operations.
15.3 Attach MUST observe flags.INITIALIZED (Acquire) before starting operations; if not set, attach MAY return WouldBlock.
	16.	Rust API contract (normative behavior, not exact names)
16.1 Types
16.1.1 ShmQueue: mapped region handle.
16.1.2 Producer: unique producer wrapper (enforced by flags CAS).
16.1.3 Consumer: unique consumer wrapper (enforced by flags CAS).
16.1.4 Config: capacity_pow2, slot_size, enable_not_full, spin_iters.
16.2 Operations
16.2.1 create(config) -> fd, total_size
16.2.2 attach(fd, total_size) -> ShmQueue
16.2.3 into_producer/into_consumer enforce uniqueness via flags compare_exchange
16.2.4 try_push, try_pop
16.2.5 push_blocking, pop_blocking with optional timeout
16.2.6 shutdown, close_producer, close_consumer
16.2.7 Optional: push_batch, pop_batch as per 13.5
16.3 Safety (mandatory)
16.3.1 Public API MUST be memory-safe.
16.3.2 Internal pointer arithmetic is unsafe but MUST be encapsulated.
16.3.3 Implementation MUST NOT create Rust references (&T / &mut T) to any non-atomic data in shared memory, including slot payload bytes and slot headers. All non-atomic access MUST be performed via raw pointers and explicit copy-in/copy-out.
16.3.4 Implementation MUST NOT create Rust references (&T / &mut T) that alias shared memory regions which may be concurrently written by another process.
16.3.5 Atomic header fields MAY be accessed via properly aligned pointers to Atomic* types at the fixed offsets. Forming &Atomic* is permitted only if: (a) the address is correctly aligned for the Atomic type, (b) the reference does not outlive the mapping, and (c) no &T/&mut T is ever formed for overlapping non-atomic regions.
16.3.6 Access to shared memory MUST use atomics for atomic fields, and explicit copies for payload.
16.3.7 All atomics MUST use core::sync::atomic and the orderings specified in section 10.
	17.	Error model (required set)
17.1 InvalidMagic
17.2 UnsupportedVersion
17.3 InvalidHeaderSize
17.4 InvalidLayout
17.5 InvalidCapacity
17.6 InvalidSlotSize
17.7 CorruptIndices
17.8 CorruptSlot
17.9 Full
17.10 Empty
17.11 Closed
17.12 Shutdown
17.13 Timeout
17.14 WouldBlock
17.15 OutputTooSmall { required: usize }
17.16 AlreadyAttached
17.17 Syscall { op: SysOp, errno: i32 }
17.18 SysOp (required enumeration values)
17.18.1 FutexWaitNe
17.18.2 FutexWakeNe
17.18.3 FutexWaitNf
17.18.4 FutexWakeNf
17.18.5 Mmap
17.18.6 Ftruncate
17.18.7 MemfdCreate
17.18.8 ShmOpen
17.18.9 CloseFd
17.19 Implementations MAY provide additional error variants, but MUST preserve a stable mapping to the required set above.
	18.	Performance rules (mandatory for a compliant optimized build)
18.1 No syscalls on hot path when progress is possible.
18.2 No CAS and no fetch_add on head or tail during push/pop.
18.3 head, tail, doorbells MUST be on distinct cache lines as per fixed offsets.
18.4 futex_wake count MUST be 1 for normal operation; INT_MAX only for shutdown/close wake-all.
18.5 Spin-then-futex is permitted; spin_iters is configurable.
18.6 When NOT_FULL_ENABLED is disabled, producer MUST NOT touch doorbell_nf or call futex on it.
	19.	Test requirements (minimum)
19.1 Threaded SPSC test (single process, two threads) for at least 10 million ops.
19.2 Cross-process test using fork or two processes mapping the same fd.
19.3 Stress tests with random sleeps, signals (EINTR), timeouts, shutdown during wait.
19.4 Corruption injection test that forces used > cap and asserts:
19.4.1 CorruptIndices is returned,
19.4.2 SHUTDOWN is set,
19.4.3 both doorbells are woken.
19.5 ABI layout test that asserts the fixed offsets in section 7 at runtime (and compile-time where possible).
19.6 Close and shutdown tests:
19.6.1 try_pop returns Closed when empty and PRODUCER_CLOSED is set.
19.6.2 try_push returns Closed when CONSUMER_CLOSED is set.
19.6.3 try_* return Shutdown when SHUTDOWN is set.
19.7 Wrap-around test: head/tail cross u64::MAX and wrap to small values while preserving correctness (used <= cap) and no false CorruptIndices.
19.8 Alignment test: all atomic fields are correctly aligned, and head/tail/doorbells do not share cache lines per section 6.3.4.
19.9 False-wake resilience: inject spurious returns (EINTR/EAGAIN) and extra futex_wake calls; correctness must hold with recheck loops.
19.10 NOT_FULL_ENABLED=false test: doorbell_nf is never waited on or woken (no futex_wait_nf/futex_wake_nf calls), and the memory location is not touched in the producer hot path.
	20.	Compliance checklist
20.1 Implements the header ABI exactly (fixed header_size, ring_offset, offsets) and validates on attach.
20.2 Implements Acquire/Release orderings exactly as in section 10.
20.3 Uses expected-value futex waits with recheck loops and fixed-budget timeout semantics.
20.4 Wakes only on empty->non-empty and full->not-full transitions (plus shutdown/close wake-all).
20.5 Enforces SPSC uniqueness across processes via flags compare_exchange with specified orderings.
20.6 For NOT_FULL_ENABLED, was_full detection MUST use a fresh head.load(Acquire).
20.7 Creator MUST set arena_offset=0 and arena_bytes=0 in v0.0.1.
20.8 Creator MUST set all reserved bytes and reserved flag bits to 0; attach MUST reject otherwise.
20.9 Futex addresses passed to syscalls are the shared 32-bit doorbell words in the mapping.

End of ShmSpscFutexQueue Spec v0.0.1
