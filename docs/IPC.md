Title: ShmSpscFutexQueue Spec v0.0.1 (Linux, Rust, ASCII)
	0.	Status
0.1 This document defines a frozen ABI and required behavior for a shared-memory SPSC queue with futex-based blocking.
0.2 Target: Linux userspace, Rust implementation, inter-process IPC on one host.
0.3 Normative keywords: MUST, MUST NOT, SHOULD, MAY.
	1.	Scope
1.1 Defines a single-producer single-consumer bounded ring buffer placed in MAP_SHARED memory.
1.2 Defines futex-based blocking for empty (consumer) and optional full (producer).
1.3 Defines a fixed slot encoding to carry variable-length payloads up to slot_size.
1.4 Non-goals
1.4.1 MPMC or fairness guarantees.
1.4.2 Cross-host transport.
1.4.3 Security isolation beyond OS-level shared-memory permissions.
1.4.4 Kernel-bypass claims during wait/wake. futex requires syscalls.
	2.	Platform requirements
2.1 OS
2.1.1 Linux kernel with futex(2) syscall supporting FUTEX_WAIT and FUTEX_WAKE on shared futex words.
2.1.2 Implementation MUST use shared futex operations (not FUTEX_PRIVATE) for inter-process use.
2.2 Architecture
2.2.1 64-bit only: x86_64 or aarch64.
2.2.2 Atomic types used MUST be lock-free on the target (Rust core atomics).
2.3 Cache line assumption
2.3.1 Cache line size assumed 64 bytes for padding and false-sharing avoidance.
2.3.2 Implementation MAY provide a build-time override, but the ABI layout in shared memory MUST remain compatible with the 64-byte alignment rules below.
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
4.6 used: head - tail.
4.7 empty: head == tail.
4.8 full: used == cap.
4.9 epoch: a monotonically increasing 32-bit integer stored in doorbell words used for futex expected comparisons.
	5.	Shared memory creation and mapping
5.1 Creation
5.1.1 Creator obtains an fd via memfd_create or shm_open.
5.1.2 Creator MUST size the object via ftruncate(fd, total_size).
5.1.3 total_size MUST include header + ring_bytes (+ optional arena_bytes, if used).
5.2 Mapping
5.2.1 Attach MUST mmap(fd, total_size, PROT_READ|PROT_WRITE, MAP_SHARED).
5.2.2 Attach MUST validate header fields before any queue operations.
5.3 Prefault (recommended)
5.3.1 Implementation SHOULD touch all pages once at init to reduce page-fault latency during hot path.
5.3.2 Implementation MAY mlock if permitted.
	6.	ABI: constants
6.1 Endianness
6.1.1 All integer fields are little-endian.
6.2 Magic and version
6.2.1 magic: u64 = 0x5348515350534651
6.2.2 version_major: u16 = 0
6.2.3 version_minor: u16 = 1
6.2.4 Attach MUST reject unknown version_major.
6.2.5 Attach MAY accept higher version_minor if header_size indicates compatibility and all required fields exist.
6.3 Cache-line alignment rules
6.3.1 header_size MUST be a multiple of 64.
6.3.2 ring_offset MUST be aligned to 64.
6.3.3 head, tail, doorbell_ne, doorbell_nf MUST start at offsets that are multiples of 64 and MUST NOT share a cache line with each other.
	7.	ABI: shared header layout
7.1 Header is repr(C) and frozen for v0.0.1.
7.2 All atomic fields MUST be naturally aligned for their type.
7.3 Header definition
struct ShmSpscHeader {
magic: u64
version_major: u16
version_minor: u16
header_size: u32

total_size: u64
ring_offset: u64
ring_bytes: u64
arena_offset: u64
arena_bytes: u64

capacity_pow2: u8
reserved0: [u8; 7]

slot_size: u32
reserved1: u32

flags: AtomicU32
reserved2: u32

producer_pid: AtomicU32
consumer_pid: AtomicU32

error_code: AtomicU32
reserved3: u32

pad0: [u8; 64]

head: AtomicU64
pad1: [u8; 64]

tail: AtomicU64
pad2: [u8; 64]

doorbell_ne: AtomicI32
pad3: [u8; 64]

doorbell_nf: AtomicI32
pad4: [u8; 64]
}
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
8.2 Slot encoding (mandatory)
8.2.1 Each slot begins with an 8-byte slot header followed by payload bytes.
8.2.2 slot_header layout at offset 0 within the slot
8.2.2.1 len: u16 (0..=payload_capacity)
8.2.2.2 tag: u16 (application-defined)
8.2.2.3 sflags: u16
8.2.2.3.1 bit 0 VALID (producer writes 1)
8.2.2.3.2 bits 1..15 reserved, MUST be 0
8.2.2.4 reserved: u16, MUST be 0
8.2.3 payload_capacity = slot_size - 8.
8.2.4 Payload bytes stored at slot+8..slot+8+len.
8.2.5 Bytes beyond len are unspecified and MUST be ignored by consumer.
8.2.6 VALID MUST NOT be used as the synchronization primitive. Synchronization is exclusively via head publish (release) and head observe (acquire).
8.2.7 Producer MAY omit setting VALID in release builds, but if omitted it MUST set sflags to 0 and consumer MUST ignore VALID. If enabled, VALID is a debug-only consistency check.
	9.	Indexing and corruption rules
9.1 Indices are u64 monotonic counters.
9.2 empty iff head == tail.
9.3 full iff (head - tail) == cap.
9.4 used = head - tail.
9.5 CORRUPT if used > cap.
9.6 On CORRUPT detection, implementation MUST
9.6.1 set flags.SHUTDOWN (best-effort),
9.6.2 wake both doorbells (see 12.6),
9.6.3 return Error::CorruptIndices.
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
	11.	Ownership and uniqueness rules
11.1 Exactly one Producer may perform push operations.
11.2 Exactly one Consumer may perform pop operations.
11.3 Only Producer writes head.
11.4 Only Consumer writes tail.
11.5 Only Consumer waits on doorbell_ne.
11.6 Only Producer waits on doorbell_nf (when enabled).
11.7 Safe Rust APIs MUST enforce uniqueness by construction. Any escape hatch MUST be explicit unsafe.
	12.	Futex protocol (exact)
12.1 Futex word requirements
12.1.1 doorbell_ne and doorbell_nf are 32-bit aligned i32 in shared memory.
12.1.2 Futex WAIT MUST use expected value equal to an epoch read from the doorbell word.
12.2 Spurious wakeups and EINTR
12.2.1 Waiter MUST treat any return from futex_wait as a prompt to recheck conditions in a loop.
12.2.2 EINTR and EAGAIN MUST be handled as spurious returns.
12.3 Not-empty wait (consumer)
12.3.1 Consumer waits only when empty.
12.3.2 Consumer pop_blocking(timeout) loop
12.3.2.1 Read h = head.load(Acquire), t = tail.load(Relaxed).
12.3.2.2 If h != t, proceed to consume.
12.3.2.3 If flags.SHUTDOWN set, return Shutdown.
12.3.2.4 If flags.PRODUCER_CLOSED set and h == t, return Closed.
12.3.2.5 Spin phase: for i in 0..spin_iters
12.3.2.5.1 cpu_relax
12.3.2.5.2 recheck h and t as above, break if non-empty
12.3.2.6 epoch = doorbell_ne.load(Relaxed).
12.3.2.7 Recheck empty (h/t).
12.3.2.8 If still empty, futex_wait(doorbell_ne, epoch, timeout).
12.3.2.9 On timeout expiration, return Timeout.
12.3.2.10 Loop.
12.4 Not-empty wake (producer)
12.4.1 Producer MUST wake consumer only on empty to non-empty transition.
12.4.2 Exact rule
12.4.2.1 Before publishing a push, producer reads t = tail.load(Acquire).
12.4.2.2 Let was_empty = (head_local == t).
12.4.2.3 After publishing head (Release), if was_empty then
12.4.2.3.1 doorbell_ne.fetch_add(1, Relaxed)
12.4.2.3.2 futex_wake(doorbell_ne, 1)
12.5 Not-full wait (producer, optional)
12.5.1 Enabled iff flags.NOT_FULL_ENABLED set at init.
12.5.2 Producer waits only when full.
12.5.3 Producer push_blocking(timeout) loop
12.5.3.1 Read t = tail.load(Acquire).
12.5.3.2 If (head_local - t) != cap, proceed to push.
12.5.3.3 If flags.SHUTDOWN set, return Shutdown.
12.5.3.4 If flags.CONSUMER_CLOSED set, return Closed.
12.5.3.5 Spin phase: for i in 0..spin_iters recheck full.
12.5.3.6 epoch = doorbell_nf.load(Relaxed).
12.5.3.7 Recheck full with t acquire.
12.5.3.8 If still full, futex_wait(doorbell_nf, epoch, timeout).
12.5.3.9 On timeout expiration, return Timeout.
12.5.3.10 Loop.
12.6 Not-full wake (consumer, optional)
12.6.1 Consumer MUST wake producer only on full to not-full transition.
12.6.2 Exact rule
12.6.2.1 Before advancing tail, consumer reads h = head.load(Acquire) once and uses t = tail_local (or tail.load(Relaxed)).
12.6.2.2 Let was_full = ((h - t) == cap).
12.6.2.3 After tail.store(Release, t+1), if was_full then
12.6.2.3.1 doorbell_nf.fetch_add(1, Relaxed)
12.6.2.3.2 futex_wake(doorbell_nf, 1)
12.7 Shutdown wake
12.7.1 shutdown() MUST set flags.SHUTDOWN and wake all potential waiters.
12.7.2 Exact rule
12.7.2.1 flags.fetch_or(SHUTDOWN, Release)
12.7.2.2 doorbell_ne.fetch_add(1, Relaxed); futex_wake(doorbell_ne, INT_MAX)
12.7.2.3 doorbell_nf.fetch_add(1, Relaxed); futex_wake(doorbell_nf, INT_MAX)
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
13.3.1.2 Producer uniqueness holds.
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
13.4.1.1 Consumer uniqueness holds.
13.4.2 Algorithm
13.4.2.1 If cached_head is stale or predicted empty, refresh h = head.load(Acquire), update cached_head.
13.4.2.2 If cached_head == tail_local, return Empty.
13.4.2.3 slot = ring_base + ((tail_local & mask) * slot_stride).
13.4.2.4 Read slot_header.
13.4.2.5 If len > payload_capacity, return CorruptSlot.
13.4.2.6 If out.len < len, return OutputTooSmall(required=len).
13.4.2.7 Copy payload bytes into out[0..len].
13.4.2.8 Let h_now = cached_head (or refresh once if desired).
13.4.2.9 Let was_full = ((h_now - tail_local) == cap) if NOT_FULL_ENABLED.
13.4.2.10 next_tail = tail_local + 1.
13.4.2.11 Publish tail.store(Release, next_tail).
13.4.2.12 Apply not-full wake rule if was_full.
13.4.2.13 Set tail_local = next_tail.
13.5 Batch operations (recommended but not required for compliance)
13.5.1 MAY provide push_batch/pop_batch.
13.5.2 Batch operations MUST preserve the same memory ordering rules and MUST NOT introduce CAS/RMW on head/tail.
13.5.3 Batch operations MUST NOT call futex.
	14.	Close semantics
14.1 close_producer()
14.1.1 Sets flags.PRODUCER_CLOSED.
14.1.2 Wakes doorbell_ne (count INT_MAX allowed).
14.1.3 After close_producer, producer MUST NOT push.
14.2 close_consumer()
14.2.1 Sets flags.CONSUMER_CLOSED.
14.2.2 Wakes doorbell_nf (count INT_MAX allowed).
14.2.3 After close_consumer, consumer MUST NOT pop.
14.3 Blocking return rules
14.3.1 pop_blocking returns Closed if empty and PRODUCER_CLOSED set.
14.3.2 push_blocking returns Closed if CONSUMER_CLOSED set.
14.3.3 Both return Shutdown if SHUTDOWN set.
	15.	Attach and validation (mandatory)
15.1 Attach MUST validate:
15.1.1 magic matches.
15.1.2 version_major == 0 and version_minor >= 1 (or exactly 1 if strict).
15.1.3 header_size within mapping and multiple of 64.
15.1.4 total_size equals mapping size.
15.1.5 ring_offset and ring_bytes within total_size.
15.1.6 ring_offset aligned to 64.
15.1.7 slot_size >= 8 and slot_size % 8 == 0.
15.1.8 capacity_pow2 in [1, 30].
15.1.9 ring_bytes == (1 << capacity_pow2) * slot_size.
15.1.10 arena_offset and arena_bytes are either both 0 or define a range within total_size (arena is optional and unused by core queue).
15.2 If any validation fails, attach MUST return an error and MUST NOT perform operations.
	16.	Rust API contract (normative behavior, not exact names)
16.1 Types
16.1.1 ShmQueue: mapped region handle.
16.1.2 Producer: unique producer wrapper.
16.1.3 Consumer: unique consumer wrapper.
16.1.4 Config: capacity_pow2, slot_size, enable_not_full, spin_iters.
16.2 Operations
16.2.1 create(config) -> fd, total_size (initializes header and sets INITIALIZED)
16.2.2 attach(fd, total_size) -> ShmQueue (validates)
16.2.3 into_producer/into_consumer enforce uniqueness.
16.2.4 try_push, try_pop.
16.2.5 push_blocking, pop_blocking with Option.
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
	18.	Performance rules (mandatory for a compliant optimized build)
18.1 No syscalls on hot path when progress is possible.
18.2 No CAS and no fetch_add on head or tail during push/pop.
18.3 head, tail, doorbells separated by cache lines as per ABI.
18.4 futex_wake count MUST be 1 for normal operation; INT_MAX only for shutdown/close wake-all.
18.5 Spin-then-futex is permitted; spin_iters is configurable.
	19.	Test requirements (minimum)
19.1 Threaded SPSC test (single process, two threads) for at least 10 million ops.
19.2 Cross-process test using fork or two processes mapping the same fd.
19.3 Stress tests with random sleeps, signals (EINTR), timeouts, shutdown during wait.
19.4 Corruption injection test for used > cap triggers CorruptIndices and wakes.
	20.	Compliance checklist
20.1 Implements header ABI exactly and validates on attach.
20.2 Implements acquire/release orderings exactly.
20.3 Uses expected-value futex waits with recheck loops.
20.4 Wakes only on empty->nonempty and full->notfull transitions (plus shutdown/close).
20.5 Enforces SPSC uniqueness at safe API boundary.

End of ShmSpscFutexQueue Spec v0.0.1
