Title: ShmSpscFutexQueue Spec v0.0.1 (Linux, Rust, ASCII)
	0.	Status
0.1 This document defines a frozen ABI and required behavior for a shared-memory SPSC queue with futex-based blocking.
0.2 Target: Linux userspace, Rust implementation, inter-process IPC on one host.
0.3 Normative keywords: MUST, MUST NOT, SHOULD, MAY.
0.4 Version policy: this spec is version-fixed. Implementations claiming compliance with v0.0.1 MUST implement exactly the ABI and semantics in this document.
	1.	Scope
1.1 Defines a single-producer single-consumer bounded ring buffer placed in MAP_SHARED memory.
1.2 Defines futex-based blocking for empty (consumer) and optional full (producer).
1.3 Defines a fixed slot encoding to carry variable-length payloads up to payload_capacity.
1.4 Non-goals
1.4.1 MPMC, multi-producer, multi-consumer, or fairness guarantees.
1.4.2 Cross-host transport.
1.4.3 Security isolation beyond OS-level shared-memory permissions.
1.4.4 Kernel-bypass claims during wait/wake. futex requires syscalls.
	2.	Platform requirements
2.1 OS
2.1.1 Linux kernel with futex(2) syscall supporting FUTEX_WAIT and FUTEX_WAKE on shared futex words.
2.1.2 For FUTEX_WAIT, timeout is interpreted as a relative duration.  ￼
2.1.3 Implementation MUST use shared futex operations (not FUTEX_PRIVATE) for inter-process use.
2.2 Architecture
2.2.1 64-bit only: x86_64 or aarch64.
2.2.2 Atomic types used MUST be lock-free on the target (Rust core atomics).
2.3 Cache line assumption
2.3.1 Cache line size assumed 64 bytes for padding and false-sharing avoidance.
2.3.2 ABI layout is defined using 64-byte boundaries; implementations MAY tune internal spinning, but MUST NOT change the ABI.
	3.	Primitive model
3.1 Data plane
3.1.1 A bounded ring of fixed-size slots in shared memory.
3.1.2 Producer writes slots and publishes head.
3.1.3 Consumer reads slots and advances tail.
3.2 Wait plane
3.2.1 Two futex words in shared memory:
3.2.1.1 doorbell_ne for not-empty (consumer waits here).
3.2.1.2 doorbell_nf for not-full (producer waits here, optional).
3.2.2 Futex waits MUST use expected-value WAIT to avoid lost wakeups.
3.2.3 All futex wakeups MUST be followed by user-space condition rechecks on the waiter side (spurious wakeups allowed).
	4.	Terminology
4.1 Producer: unique writer of head and of slots that become newly visible.
4.2 Consumer: unique writer of tail and of slots that become newly consumed.
4.3 cap: ring capacity in slots, power-of-two.
4.4 head: monotonic u64 counter of next publish position.
4.5 tail: monotonic u64 counter of next consume position.
4.6 used: head - tail, computed with wrapping subtraction for diagnostics.
4.7 empty: head == tail.
4.8 full: (head - tail) == cap (with non-wrapping arithmetic in the valid regime).
4.9 epoch: a monotonically increasing 32-bit integer stored in doorbell words used for futex expected comparisons.
	5.	Shared memory creation and mapping
5.1 Creation
5.1.1 Creator obtains an fd via memfd_create or shm_open.
5.1.2 Creator MUST size the object via ftruncate(fd, total_size).
5.1.3 total_size MUST include header + ring_bytes (+ optional arena_bytes, if used).
5.2 Mapping
5.2.1 Attach MUST mmap(fd, total_size, PROT_READ|PROT_WRITE, MAP_SHARED).
5.2.2 Attach MUST validate header fields before any queue operations.
5.2.3 Attach MUST NOT perform push/pop before INITIALIZED is observed (see 11.2.3).
5.3 Prefault (recommended)
5.3.1 Implementation SHOULD touch all pages once at init to reduce page-fault latency during hot path.
5.3.2 Implementation MAY mlock if permitted.
	6.	ABI: constants
6.1 Endianness
6.1.1 All integer fields are little-endian.
6.2 Magic and version (fixed)
6.2.1 magic: u64 = 0x5348515350534651
6.2.2 version_major: u16 = 0
6.2.3 version_minor: u16 = 1
6.2.4 Attach MUST accept only (version_major == 0 and version_minor == 1).
6.3 Cache-line alignment rules (fixed)
6.3.1 header_size MUST be 0x180 bytes (384) and MUST be a multiple of 64.
6.3.2 ring_offset MUST be aligned to 64.
6.3.3 head, tail, doorbell_ne, doorbell_nf MUST start at offsets that are multiples of 64 and MUST NOT share a cache line with each other.
	7.	ABI: shared header layout (frozen offsets)
7.1 Header is repr(C) and frozen for v0.0.1.
7.2 All atomic fields MUST be naturally aligned for their type, and MUST start at the fixed offsets below.
7.3 Header size and fixed offsets
7.3.1 header_size is fixed: 0x180 bytes.
7.3.2 Offsets are from the start of the shared mapping.
7.3.3 Fields
7.3.3.1 0x000 magic: u64
7.3.3.2 0x008 version_major: u16
7.3.3.3 0x00A version_minor: u16
7.3.3.4 0x00C header_size: u32 (MUST be 0x180)
7.3.3.5 0x010 total_size: u64
7.3.3.6 0x018 ring_offset: u64
7.3.3.7 0x020 ring_bytes: u64
7.3.3.8 0x028 arena_offset: u64
7.3.3.9 0x030 arena_bytes: u64
7.3.3.10 0x038 capacity_pow2: u8
7.3.3.11 0x039 reserved0: [u8; 7] (MUST be 0)
7.3.3.12 0x040 slot_size: u32
7.3.3.13 0x044 reserved1: u32 (MUST be 0)
7.3.3.14 0x048 flags: AtomicU32
7.3.3.15 0x04C reserved2: u32 (MUST be 0)
7.3.3.16 0x050 producer_pid: AtomicU32 (diagnostic only)
7.3.3.17 0x054 consumer_pid: AtomicU32 (diagnostic only)
7.3.3.18 0x058 error_code: AtomicU32 (diagnostic only)
7.3.3.19 0x05C reserved3: u32 (MUST be 0)
7.3.3.20 0x060 reserved4: [u8; 32] (MUST be 0)
7.3.3.21 0x080 head: AtomicU64
7.3.3.22 0x0C0 tail: AtomicU64
7.3.3.23 0x100 doorbell_ne: AtomicI32
7.3.3.24 0x140 doorbell_nf: AtomicI32
7.4 Flags bitfield (AtomicU32)
7.4.1 bit 0 INITIALIZED
7.4.2 bit 1 PRODUCER_ATTACHED
7.4.3 bit 2 CONSUMER_ATTACHED
7.4.4 bit 3 PRODUCER_CLOSED
7.4.5 bit 4 CONSUMER_CLOSED
7.4.6 bit 5 SHUTDOWN
7.4.7 bit 6 NOT_FULL_ENABLED
7.4.8 bits 7..31 reserved, MUST be 0.
	8.	Ring layout and slot encoding
8.1 Ring
8.1.1 cap = 1 << capacity_pow2.
8.1.2 mask = cap - 1.
8.1.3 slot_stride = slot_size.
8.1.4 ring_bytes MUST equal cap * slot_size.
8.1.5 ring_offset MUST satisfy ring_offset >= header_size and ring_offset % 64 == 0.
8.2 Slot encoding (mandatory)
8.2.1 Each slot begins with an 8-byte slot header followed by payload bytes.
8.2.2 slot_header layout at offset 0 within the slot
8.2.2.1 len: u16 (0..=payload_capacity)
8.2.2.2 tag: u16 (application-defined)
8.2.2.3 sflags: u16
8.2.2.3.1 bit 0 VALID (producer MAY set to 1 for debug)
8.2.2.3.2 bits 1..15 reserved, MUST be 0
8.2.2.4 reserved: u16, MUST be 0
8.2.3 payload_capacity = slot_size - 8.
8.2.4 To keep len representable, payload_capacity MUST be <= 65535.
8.2.5 Payload bytes stored at slot+8..slot+8+len.
8.2.6 Bytes beyond len are unspecified and MUST be ignored by consumer.
8.2.7 VALID MUST NOT be used as the synchronization primitive. Synchronization is exclusively via head publish (release) and head observe (acquire).
8.2.8 If VALID is omitted, producer MUST write sflags=0 and consumer MUST ignore VALID.
	9.	Indexing and corruption rules
9.1 Indices are u64 monotonic counters.
9.2 used is computed for validation as used = head.wrapping_sub(tail).
9.3 empty iff head == tail.
9.4 full iff (head - tail) == cap in the valid regime (i.e., when head and tail have not wrapped relative to each other).
9.5 CORRUPT if used > cap.
9.6 On CORRUPT detection, implementation MUST
9.6.1 set flags.SHUTDOWN (best-effort),
9.6.2 wake both doorbells (see 12.7),
9.6.3 return Error::CorruptIndices.
9.7 Wrap handling
9.7.1 If head and tail wrap such that used becomes large, 9.5 triggers and the queue enters SHUTDOWN.
9.7.2 v0.0.1 provides no wrap recovery; a new queue MUST be created.
	10.	Memory ordering contract (mandatory)
10.1 Producer publish rule
10.1.1 Producer MUST write slot payload and slot header using plain stores before publishing head.
10.1.2 Producer MUST publish head using store with Release ordering.
10.2 Consumer observe rule
10.2.1 Consumer MUST load head using Acquire ordering before reading any slot contents for indices < head.
10.2.2 Consumer MUST advance tail using store with Release ordering after consuming.
10.3 Producer full check rule
10.3.1 Producer MUST load tail using Acquire ordering when checking for full.
10.4 Doorbell ordering
10.4.1 Doorbell loads/stores MAY be Relaxed.
10.4.2 Correctness MUST rely on condition recheck using Acquire/Release on head/tail.
10.5 Flags ordering
10.5.1 Creator MUST publish INITIALIZED using flags.store(Release) after fully initializing the header, head, tail, and doorbells.
10.5.2 Attach MUST read flags with Acquire before trusting initialized content.
	11.	Ownership and attachment protocol (mandatory)
11.1 Exactly one Producer may perform push operations.
11.2 Exactly one Consumer may perform pop operations.
11.3 Safe Rust APIs MUST enforce uniqueness within a process by construction.
11.4 Cross-process uniqueness MUST be enforced via flags bits:
11.4.1 attach_producer MUST set PRODUCER_ATTACHED via compare_exchange on flags.
11.4.2 attach_consumer MUST set CONSUMER_ATTACHED via compare_exchange on flags.
11.4.3 If the relevant bit is already set, attach MUST fail with Error::AlreadyAttached (implementation-defined name; MUST map to a distinct error).
11.4.4 v0.0.1 does not define crash-recovery or stealing of attachment; a stale ATTACHED bit requires creating a new queue.
11.5 PID fields
11.5.1 producer_pid and consumer_pid are diagnostic only.
11.5.2 Writing PID MUST occur after successfully acquiring the ATTACHED bit.
11.5.3 PID MUST NOT be used for correctness.
	12.	Futex protocol (exact)
12.1 Futex word requirements
12.1.1 doorbell_ne and doorbell_nf are 32-bit aligned i32 in shared memory.
12.1.2 Futex WAIT MUST use expected value equal to an epoch read from the doorbell word.
12.2 Spurious wakeups and EINTR
12.2.1 Waiter MUST treat any return from futex_wait as a prompt to recheck conditions in a loop.
12.2.2 EINTR and EAGAIN MUST be handled as spurious returns.
12.3 Timeout model (normative)
12.3.1 timeout is an optional relative duration.
12.3.2 If timeout is None, futex_wait MUST be called with NULL timeout.
12.3.3 If timeout is Some(d), futex_wait MUST be called with a relative timespec derived from d.
12.3.4 On ETIMEDOUT, blocking operation MUST return Timeout.
12.3.5 Implementations SHOULD treat EINVAL from futex_wait with a provided timespec as Syscall(errno) (timespec invalid).
12.4 Not-empty wait (consumer)
12.4.1 Consumer waits only when empty.
12.4.2 pop_blocking(timeout) loop
12.4.2.1 Read h = head.load(Acquire), t = tail.load(Relaxed).
12.4.2.2 If h != t, proceed to consume.
12.4.2.3 If flags.SHUTDOWN set, return Shutdown.
12.4.2.4 If flags.PRODUCER_CLOSED set and h == t, return Closed.
12.4.2.5 Spin phase: for i in 0..spin_iters
12.4.2.5.1 cpu_relax
12.4.2.5.2 recheck h/t as above, break if non-empty
12.4.2.6 epoch = doorbell_ne.load(Relaxed).
12.4.2.7 Recheck empty using h = head.load(Acquire) and t (tail_local or tail.load(Relaxed)).
12.4.2.8 If still empty, futex_wait(doorbell_ne, epoch, timeout).
12.4.2.9 If timed out, return Timeout.
12.4.2.10 Loop.
12.5 Not-empty wake (producer)
12.5.1 Producer MUST wake consumer only on empty to non-empty transition.
12.5.2 Exact rule
12.5.2.1 Before publishing a push, producer reads t = tail.load(Acquire).
12.5.2.2 Let was_empty = (head_local == t).
12.5.2.3 After publishing head (Release), if was_empty then
12.5.2.3.1 doorbell_ne.fetch_add(1, Relaxed)
12.5.2.3.2 futex_wake(doorbell_ne, 1)
12.6 Not-full wait (producer, optional)
12.6.1 Enabled iff flags.NOT_FULL_ENABLED set at init and observed by producer.
12.6.2 Producer waits only when full.
12.6.3 push_blocking(timeout) loop
12.6.3.1 Read t = tail.load(Acquire).
12.6.3.2 If (head_local - t) != cap, proceed to push.
12.6.3.3 If flags.SHUTDOWN set, return Shutdown.
12.6.3.4 If flags.CONSUMER_CLOSED set, return Closed.
12.6.3.5 Spin phase: for i in 0..spin_iters recheck full (with t Acquire refresh permitted).
12.6.3.6 epoch = doorbell_nf.load(Relaxed).
12.6.3.7 Recheck full using t = tail.load(Acquire).
12.6.3.8 If still full, futex_wait(doorbell_nf, epoch, timeout).
12.6.3.9 If timed out, return Timeout.
12.6.3.10 Loop.
12.7 Not-full wake (consumer, optional)
12.7.1 Consumer MUST wake producer only on full to not-full transition.
12.7.2 Exact rule
12.7.2.1 Before advancing tail, consumer MUST read h = head.load(Acquire).
12.7.2.2 Let t = tail_local (current).
12.7.2.3 Let was_full = ((h - t) == cap).
12.7.2.4 After tail.store(Release, t+1), if was_full then
12.7.2.4.1 doorbell_nf.fetch_add(1, Relaxed)
12.7.2.4.2 futex_wake(doorbell_nf, 1)
12.8 Shutdown wake
12.8.1 shutdown() MUST set flags.SHUTDOWN and wake all potential waiters.
12.8.2 Exact rule
12.8.2.1 flags.fetch_or(SHUTDOWN, Release)
12.8.2.2 doorbell_ne.fetch_add(1, Relaxed); futex_wake(doorbell_ne, INT_MAX)
12.8.2.3 doorbell_nf.fetch_add(1, Relaxed); futex_wake(doorbell_nf, INT_MAX)
	13.	Ring operations (exact)
13.1 Producer state
13.1.1 Producer MUST keep head_local in private memory.
13.1.2 Producer SHOULD keep cached_tail in private memory to reduce shared loads.
13.2 Consumer state
13.2.1 Consumer MUST keep tail_local in private memory.
13.2.2 Consumer SHOULD keep cached_head in private memory to reduce shared loads.
13.3 try_push(tag, payload)
13.3.1 Preconditions
13.3.1.1 payload.len <= payload_capacity.
13.3.1.2 Producer uniqueness holds (11 and 12).
13.3.2 Algorithm
13.3.2.1 If cached_tail is stale or predicted full, refresh t = tail.load(Acquire), update cached_tail.
13.3.2.2 If (head_local - t) == cap, return Full.
13.3.2.3 slot = ring_base + ((head_local & mask) * slot_stride).
13.3.2.4 Write payload into slot+8..slot+8+len.
13.3.2.5 Write slot_header (len, tag, sflags, reserved) at slot.
13.3.2.6 Compute next_head = head_local + 1.
13.3.2.7 Publish head.store(Release, next_head).
13.3.2.8 Apply not-empty wake rule if transition was empty.
13.3.2.9 Set head_local = next_head.
13.3.3 Postconditions
13.3.3.1 Consumer observing head via Acquire MUST see the slot writes.
13.4 try_pop(out)
13.4.1 Preconditions
13.4.1.1 Consumer uniqueness holds (11 and 12).
13.4.2 Algorithm
13.4.2.1 If cached_head is stale or predicted empty, refresh h = head.load(Acquire), update cached_head.
13.4.2.2 If cached_head == tail_local, return Empty.
13.4.2.3 slot = ring_base + ((tail_local & mask) * slot_stride).
13.4.2.4 Read slot_header (plain loads).
13.4.2.5 If len > payload_capacity, return CorruptSlot.
13.4.2.6 If out.len < len, return OutputTooSmall(required=len).
13.4.2.7 Copy payload bytes into out[0..len].
13.4.2.8 If NOT_FULL_ENABLED is set, compute was_full using a fresh h_now = head.load(Acquire) and current tail_local.
13.4.2.9 next_tail = tail_local + 1.
13.4.2.10 Publish tail.store(Release, next_tail).
13.4.2.11 Apply not-full wake rule if was_full.
13.4.2.12 Set tail_local = next_tail.
13.5 Batch operations (recommended but not required for compliance)
13.5.1 MAY provide push_batch/pop_batch.
13.5.2 Batch operations MUST preserve the same memory ordering rules and MUST NOT introduce CAS/RMW on head/tail.
13.5.3 Batch operations MUST NOT call futex.
	14.	Close semantics
14.1 close_producer()
14.1.1 Sets flags.PRODUCER_CLOSED via fetch_or(Release or AcqRel).
14.1.2 Wakes doorbell_ne (count INT_MAX allowed).
14.1.3 After close_producer, producer MUST NOT push.
14.2 close_consumer()
14.2.1 Sets flags.CONSUMER_CLOSED via fetch_or(Release or AcqRel).
14.2.2 Wakes doorbell_nf (count INT_MAX allowed).
14.2.3 After close_consumer, consumer MUST NOT pop.
14.3 Blocking return rules
14.3.1 pop_blocking returns Closed if empty and PRODUCER_CLOSED set.
14.3.2 push_blocking returns Closed if full and CONSUMER_CLOSED set (when NOT_FULL_ENABLED).
14.3.3 Both return Shutdown if SHUTDOWN set.
	15.	Attach and validation (mandatory)
15.1 Attach MUST validate:
15.1.1 magic matches.
15.1.2 version_major == 0 and version_minor == 1.
15.1.3 header_size == 0x180.
15.1.4 total_size equals mapping size.
15.1.5 ring_offset and ring_bytes within total_size.
15.1.6 ring_offset aligned to 64 and ring_offset >= header_size.
15.1.7 slot_size >= 8 and slot_size % 8 == 0.
15.1.8 payload_capacity <= 65535.
15.1.9 capacity_pow2 in [1, 30].
15.1.10 ring_bytes == (1 << capacity_pow2) * slot_size.
15.1.11 arena_offset and arena_bytes are either both 0 or define a range within total_size.
15.1.12 reserved fields and reserved flag bits are 0.
15.2 If any validation fails, attach MUST return an error and MUST NOT perform operations.
15.3 Attach MUST observe flags.INITIALIZED (Acquire) before starting operations; if not set, attach MAY wait briefly and retry, or return WouldBlock.
	16.	Rust API contract (normative behavior, not exact names)
16.1 Types
16.1.1 ShmQueue: mapped region handle.
16.1.2 Producer: unique producer wrapper.
16.1.3 Consumer: unique consumer wrapper.
16.1.4 Config: capacity_pow2, slot_size, enable_not_full, spin_iters.
16.2 Operations
16.2.1 create(config) -> fd, total_size (initializes header and sets INITIALIZED)
16.2.2 attach(fd, total_size) -> ShmQueue (validates)
16.2.3 into_producer/into_consumer enforce cross-process uniqueness via flags CAS.
16.2.4 try_push, try_pop.
16.2.5 push_blocking, pop_blocking with optional timeout.
16.2.6 shutdown, close_producer, close_consumer.
16.3 Safety
16.3.1 Public API MUST be memory-safe.
16.3.2 Internal pointer arithmetic is unsafe but MUST be encapsulated.
16.3.3 Implementation MUST NOT create references (&T) into shared memory for slot payload; use raw pointers and copies to avoid aliasing UB.
16.3.4 All atomics MUST use core::sync::atomic and the orderings specified in section 10.
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
17.16 Syscall { errno: i32 }
17.17 AlreadyAttached
	18.	Performance rules (mandatory for a compliant optimized build)
18.1 No syscalls on hot path when progress is possible.
18.2 No CAS and no fetch_add on head or tail during push/pop.
18.3 head, tail, doorbells separated by cache lines as per ABI.
18.4 futex_wake count MUST be 1 for normal operation; INT_MAX only for shutdown/close wake-all.
18.5 Spin-then-futex is permitted; spin_iters is configurable.
18.6 NOT_FULL_ENABLED may reduce producer spinning at the cost of extra wake traffic on full->not-full transitions only.
	19.	Test requirements (minimum)
19.1 Threaded SPSC test (single process, two threads) for at least 10 million ops.
19.2 Cross-process test using fork or two processes mapping the same fd.
19.3 Stress tests with random sleeps, signals (EINTR), timeouts, shutdown during wait.
19.4 Corruption injection test for used > cap triggers CorruptIndices and wakes.
19.5 ABI layout test that asserts the fixed offsets in section 7 at runtime (or compile-time where possible).
	20.	Compliance checklist
20.1 Implements header ABI exactly (fixed offsets, header_size==0x180) and validates on attach.
20.2 Implements acquire/release orderings exactly.
20.3 Uses expected-value futex waits with recheck loops.
20.4 Wakes only on empty->nonempty and full->notfull transitions (plus shutdown/close).
20.5 Enforces SPSC uniqueness across processes via flags CAS.
20.6 NOT_FULL_ENABLED uses fresh head.load(Acquire) for was_full detection.

End of ShmSpscFutexQueue Spec v0.0.1
