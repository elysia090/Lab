Title
  GuardHV v0.0.1 – Thin Hypervisor Enforcement Kernel

Status
  Design specification, implementation-oriented, Linux-only, x86_64

Language
  ASCII, English only

0. Scope
--------

GuardHV v0.0.1 is a type-1 hypervisor that boots before a single
Linux guest and enforces a small set of non-bypassable security
policies with minimal runtime overhead.

v0.0.1 targets:

  - Commodity x86_64 servers and laptops with:
      * Intel VT-x + EPT + VT-d (or AMD SVM + NPT + IOMMU)
      * UEFI Secure Boot
  - A single Linux guest (no nested virtualization, no Windows yet)
  - A constrained set of TTPs:
      * eBPF misuse for stealth hooks / sensors
      * rsyslog/logrotate config-based persistence
      * privilege escalation via execve chains from log daemons
      * unauthorized manipulation of key log/config files
      * uncontrolled outbound L3/L4 connectivity

Non-goals for v0.0.1:

  - Full L7 protocol enforcement
  - ServiceWorker / browser / GPU introspection
  - Multi-guest support
  - Live policy updates (policy is static at boot)


1. Threat Model
---------------

1.1 Adversary

The adversary:

  - Can obtain arbitrary code execution inside the Linux guest
    at the level of an unprivileged or privileged user.
  - Can load and run arbitrary userland binaries.
  - Can attempt to:
      * load eBPF programs,
      * modify rsyslog/logrotate configuration,
      * use log daemons as execution pivots,
      * exfiltrate data via arbitrary outbound TCP/UDP connections.

The adversary:

  - Cannot (in v0.0.1) compromise the hypervisor or firmware.
  - Cannot physically bypass UEFI Secure Boot and VT-d/IOMMU setup.
  - Cannot compromise GuardHV’s keys used to sign its image.

1.2 Assets

GuardHV protects:

  - Integrity of specific configuration and log files on the guest:
      * rsyslog.conf, logrotate.d/*, journald config
      * selected log files (e.g. auth.log, secure, syslog)
  - Integrity of the kernel instrumentation surface:
      * eBPF program load / attach
  - Controlled outbound network connectivity:
      * only pre-approved destinations and ports per host label
  - Integrity and non-repudiation of GuardHV’s own event log.

GuardHV does NOT protect:

  - Confidentiality of all guest data (guest is still a normal OS).
  - Fine-grained application-layer semantics (HTTP, SMTP contents).
  - Local physical attacks on hardware.


2. Platform and Boot Requirements
---------------------------------

2.1 Hardware

GuardHV v0.0.1 assumes:

  - x86_64 CPU with:
      * Intel VT-x with EPT and VT-d, or
      * AMD SVM with NPT and IOMMU
  - UEFI firmware with Secure Boot
  - At least one Ethernet NIC with:
      * MSI-X
      * supported by a DPDK-capable driver (e.g. Intel ixgbe/i40e/ice)
  - TPM 2.0 (optional but recommended for attestation/anchoring)

2.2 Boot Chain

The boot chain is:

  UEFI → GuardHV image → GuardHV loads Linux guest

Requirements:

  - GuardHV image MUST be signed with a key trusted by Secure Boot.
  - GuardHV MUST NOT chain-load arbitrary unverified kernels.
  - The Linux guest kernel may be unmodified, but its image hash is
    measured and recorded into the GuardHV event log at load time.

For v0.0.1:

  - Exactly one guest VM is supported.
  - GuardHV itself is non-updatable at runtime (no module loading).


3. High-level Architecture
--------------------------

Components:

  - Core:
      * VMX/SVM management, EPT/NPT setup
      * IOMMU configuration
      * Minimal scheduler (single guest vCPU mapping to pCPU)
      * Timekeeping, APIC virtualization

  - Policy Engine:
      * Syscall Gate (Linux-specific)
      * FS Gate (config/log file protection)
      * Net Gate (label-based egress control)
      * eBPF Gate (program load control)

  - Telemetry + Crypto Log:
      * Event ring buffer (GuardHV internal)
      * Hash-chained, signed log output to a dedicated disk partition
        or serial/UART for collection

  - Guest Interface:
      * Minimal hypercall ABI for:
          - host identity / label registration
          - policy version query
          - one-way event push (from agent to HV, optional)

Data flow summary:

  - CPU instructions from the Linux guest run natively, except on:
      * syscalls (selected subset)
      * certain CR writes / MSR operations
      * EPT-protected pages (config/log file writes)
      * I/O to certain MMIO ranges (NIC)
  - All physical I/O devices are owned by GuardHV and exposed as
    virtual devices to the guest.


4. Guest Model (Linux v0.0.1)
-----------------------------

GuardHV v0.0.1 supports:

  - Single Linux distribution at a time (kernel 5.x/6.x)
  - Boot as a PVH-like VM:
      * GuardHV emulates a simple bootloader interface:
          - passes kernel image and initrd
          - provides a basic device tree or ACPI subset
      * Guest sees:
          - one virtio-net NIC (backed by GuardHV Net Gate)
          - one virtio-blk disk (backed by physical disk)
          - optional virtio-serial used for a GuardHV-aware agent

Guest OS requirements:

  - No special kernel patches required, but:
      * optional lightweight agent can be installed to:
          - send process signing info to GuardHV
          - expose high-level events for correlation
  - eBPF subsystem is compiled in as usual; GuardHV will gate loads.


5. Policy Model v0.0.1
----------------------

Policies are static and compiled into the GuardHV image as a
read-only table.

5.1 Subject Identity

GuardHV identifies subjects by:

  - Guest PID/PPID (from vCPU state / OS-specific helpers)
  - Binary identity:
      * image hash (SHA-256 of executable file)
      * optional signer identity (if agent reports a signature)
  - Host label:
      * a small integer label assigned at boot via a config blob
        embedded in GuardHV (e.g. LAB=1, PROD=2, DMZ=3)

No dynamic user/role mapping is performed in v0.0.1.

5.2 Policy Types

v0.0.1 supports four policy tables:

  1. eBPF Policy:
       key: image_hash
       value: {allow_load: bool}

  2. FS Protection Policy:
       key: path_prefix (for config/log files)
       value: {allow_writers: [image_hash]}

  3. Exec Chain Policy:
       key: (parent_image_hash, child_image_hash)
       value: {allow_exec: bool}

  4. Net Egress Policy:
       key: host_label
       value: {egress_rules[]}
           egress_rule:
             dst_cidr, dst_port_range, proto, action

Policies are small, fixed-size arrays; v0.0.1 does not support
runtime policy update. Policy updates require rebuilding and
reinstalling the GuardHV image.


6. Enforcement Mechanisms v0.0.1
--------------------------------

6.1 Syscall Gate (Linux)

GuardHV intercepts a subset of syscalls using VM exits. For Linux:

  - Mandatory intercept:
      * `bpf`
      * `ptrace`
      * `clone3` / `unshare` (optional for v0.0.1)
      * `execve` / `execveat`
      * `openat` / `open` (for specific path prefixes only)
      * `mount`, `umount` (optional for stricter modes)

Interception method:

  - Guest is configured with syscall instruction causing a VM exit
    for all syscalls (or for a subset based on MSR configuration).
  - GuardHV reads registers:
      * syscall number (RAX)
      * arguments (RDI, RSI, RDX, R10, R8, R9)
  - For non-targeted syscalls:
      * GuardHV quickly passes control back:
          - adjusts RIP to after syscall
          - performs the syscall using guest mode (fast path)
            or emulates minimally (implementation dependent)
  - For targeted syscalls:
      * GuardHV applies the relevant policy checks (below).
      * If allowed:
          - syscall is executed as normal
      * If denied:
          - syscall is forced to fail with `-EPERM`
          - an event is logged

6.2 eBPF Gate

For `bpf(BPF_PROG_LOAD, ...)`:

  - GuardHV obtains:
      * calling process image_hash
      * BPF program type (e.g. kprobe, tracepoint, socket, etc.)
  - Policy:
      * If image_hash is not in eBPF Policy allow list:
          - deny load (return -EPERM)
          - log event `EV_BPF_DENY`
      * If allowed:
          - pass through

Optional v0.0.1 restriction:

  - Only a dedicated, pre-approved security agent binary is allowed
    to load BPF programs.
  - All other `bpf` calls are denied.

6.3 FS Gate

GuardHV protects specific guest paths:

  - Example protected set P:
      * `/etc/rsyslog.conf`
      * `/etc/rsyslog.d/`
      * `/etc/logrotate.conf`
      * `/etc/logrotate.d/`
      * `/var/log/auth.log`
      * `/var/log/secure`
      * `/var/log/syslog`

Mechanism:

  - GuardHV translates these paths into guest physical page ranges
    at boot-time by performing a minimal scan or via a guest-side
    agent that reports the mappings once at start.
  - Those pages are mapped read-only in EPT for the guest.
  - On a guest write to a protected page:
      * EPT violation occurs → VM exit
      * GuardHV resolves:
          - identifies process image_hash
          - consults FS Protection Policy
      * If writer allowed for that path:
          - GuardHV performs a copy-on-write:
              - allocates a new page
              - updates EPT mapping for that guest page
              - logs event `EV_FS_WRITE_ALLOW`
          - write proceeds
      * If not allowed:
          - write is blocked
          - guest sees write failure (e.g. `-EACCES` or `-EROFS`)
          - event `EV_FS_WRITE_DENY` is logged

For v0.0.1, FS Gate is limited to a small set P to keep complexity low.

6.4 Exec Chain Gate

GuardHV enforces simple exec-chain constraints for:

  - `execve` where the parent image_hash is one of:
      * rsyslogd
      * logrotate
      * journald
      * other log daemons configured in policy.

Mechanism:

  - On `execve` syscall intercept:
      * Read parent image_hash (from metadata cache)
      * Determine new executable’s image_hash (via guest agent or via
        a one-time lookup recorded earlier)
  - Consult Exec Chain Policy:
      * If (parent_hash, child_hash) not allowed:
          - deny exec (e.g. return `-EACCES`)
          - log `EV_EXEC_DENY`
      * else:
          - allow

This directly prevents log-based persistence chains such as:

  rsyslog → /bin/sh → payload
  logrotate postrotate script → /bin/sh → payload

6.5 Net Gate (Egress-only, v0.0.1)

GuardHV owns the physical NIC, and guest sees a virtio-net NIC.

GuardHV:

  - Implements a simple label-based ACL per host_label:

    - host_label L has rules:
        (dst_cidr, dst_port_range, proto, action)

  - For each outbound TCP/UDP flow:
      * Extracts 5-tuple (src_ip, src_port, dst_ip, dst_port, proto)
      * Checks rules for host_label L
      * If allowed: forwards packets
      * If denied: drops and logs `EV_NET_DENY`

v0.0.1 simplifications:

  - No L7 inspection
  - No per-process or per-user egress control
  - No dynamic connection tracking beyond basic 5-tuple flows


7. Telemetry and Cryptographic Log
----------------------------------

GuardHV maintains:

  - An in-memory ring buffer of events.
  - A tamper-evident log anchored by hash chaining and signatures.

7.1 Event Format

Each event E has:

  - timestamp (monotonic counter)
  - event_id (e.g. EV_BPF_DENY, EV_FS_WRITE_DENY, EV_EXEC_DENY)
  - guest_pid, guest_ppid
  - image_hash
  - extra fields depending on event (path, syscall no, dst_ip, etc.)

Events are appended to an in-memory ring:

  - fixed size (e.g. 64k events)
  - overwrite oldest entries on overflow

7.2 Hash Chain and Signing

GuardHV maintains:

  - state H_0 = all-zero 256-bit value.
  - For each flushed log block B_i:
      * H_i = SHA-256(H_{i-1} || B_i)

Periodically (or on demand):

  - GuardHV signs (H_i, timestamp, seqno) with its private key
    stored in TPM or measured at boot.
  - The signed record is persisted to:
      * a dedicated partition (append-only) or
      * streamed out via serial/UART or a management NIC.

Security properties:

  - Tampering with past log blocks changes H_i and invalidates
    the signature chain.
  - v0.0.1 does not implement remote attestation, but the log can
    be validated offline against GuardHV’s public key and measured
    image hash.


8. Guest Interface (Hypercalls)
-------------------------------

GuardHV provides a minimal hypercall ABI exposed via:

  - CPUID leaf advertising GuardHV presence and version.
  - A specific I/O port or MSR used as hypercall trigger.

Hypercalls (v0.0.1):

  - HV_GET_VERSION:
      * returns GuardHV version, policy version

  - HV_GET_HOST_LABEL:
      * returns host_label

  - HV_PUSH_PROC_INFO (optional):
      * guest agent reports:
          - PID
          - image_hash
          - signer_id (if known)
      * GuardHV caches mapping PID → (image_hash, signer_id)

These hypercalls are optional; GuardHV can also run without a guest
agent, at the cost of reduced fidelity of image identity. In that
case, some enforcement (like Exec Chain) may use coarser heuristics.


9. Implementation Notes (Non-normative)
---------------------------------------

9.1 Language and Structure

- GuardHV core SHOULD be implemented in Rust or a similarly safe
  systems language where possible, with a minimal C/ASM shim for:
    * entry/exit
    * VMX/SVM instructions
    * page table and IOMMU setup

- No dynamic memory allocation in the core enforcement path:
    * all policy tables and event buffers allocated at boot.

9.2 Performance Targets

- Syscall intercept overhead for non-targeted syscalls MUST be
  minimized:
    * use fast-path heuristics (e.g. simple number comparisons)
    * keep VM exits low by selectively intercepting only relevant
      subsystems if hardware allows.

- Net Gate should be implemented with a small set of fixed-size
  hash tables or binary search over small rule lists.

- FS Gate page tracking:
    * v0.0.1 may accept a one-time guest-agent-assisted discovery
      of mappings at early boot, then treat them as static.

9.3 Configuration and Build

- Policies are specified in a static configuration file:

  - e.g. `guardhv_policy.hcl` or a simple custom format.

- A build tool compiles this into a C/Rust array and links it into
  the GuardHV image.

- Every GuardHV build is:
    * reproducible (same input → same image hash)
    * signed with a dedicated key

9.4 Failure Handling

- On internal GuardHV errors:
    * default to “fail closed” for enforcement:
        - deny eBPF load
        - deny FS writes to protected paths
        - deny Exec chains
        - deny outbound network
    * log an error event

- GuardHV MUST avoid guest kernel panic unless absolutely necessary
  (policy violation should be reported to admin first).

10. Security Guarantees (v0.0.1)
--------------------------------

Under the stated assumptions, GuardHV v0.0.1 guarantees:

  - Only a pre-approved binary can load eBPF programs in the guest.
  - Specific config/log files cannot be modified by unauthorized
    binaries, even with root inside the guest.
  - Log-daemon-based exec chains (rsyslog/logrotate/journald →
    arbitrary shell) can be blocked at the hypervisor level.
  - All outbound network flows comply with a host-level egress ACL;
    violations are dropped and logged.
  - Enforcement decisions and significant violations are recorded
    in a tamper-evident log independent of the guest OS.

It does NOT guarantee:

  - Protection against kernel 0-day exploits *inside* the guest that
    do not rely on the gated surfaces.
  - Protection against physical attacks or firmware-level compromise.
  - Complete prevention of all APT-grade techniques; it only makes
    specific families of TTPs structurally harder or impossible in
    environments where GuardHV policy is correctly configured.

End of GuardHV v0.0.1 specification.
