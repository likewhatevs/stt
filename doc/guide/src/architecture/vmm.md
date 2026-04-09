# VMM

stt includes a purpose-built VMM (virtual machine monitor) that boots
Linux kernels in KVM for testing.

## SttVm builder

```rust,ignore
let result = vmm::SttVm::builder()
    .kernel(&kernel_path)
    .init_binary(&stt_binary)
    .topology(sockets, cores_per_socket, threads_per_core)
    .memory_mb(4096)
    .run_args(&["run", "--stt-test-fn", "my_test"])
    .build()?
    .run()?;
```

## Topology

The VM topology is specified as `(sockets, cores_per_socket,
threads_per_core)`. The VMM creates the appropriate ACPI tables
(MADT, SRAT) and MP tables so the guest kernel sees the specified
topology.

```rust,ignore
pub struct Topology {
    pub sockets: u32,
    pub cores_per_socket: u32,
    pub threads_per_core: u32,
}
```

`total_cpus()` = sockets * cores_per_socket * threads_per_core.
`num_llcs()` = sockets (one LLC per socket).

## initramfs

The VMM builds a cpio initramfs containing:

- The test binary (as `/init`)
- Optional scheduler binary (as `/scheduler`)
- Shared library dependencies (resolved via `ldd`)

The initramfs is cached based on a `BaseKey` derived from the binary
contents. A compressed SHM segment enables COW overlay into guest
memory, sharing physical pages across concurrent VMs.

## Guest-host communication

**Serial console** -- the guest writes delimited test results to COM2
(between `===STT_TEST_RESULT_START===` / `===STT_TEST_RESULT_END===`
sentinels). The host parses results from the serial output after VM exit.

**SHM ring buffer** -- a shared memory ring buffer passes data from
guest to host. Used for profraw data (`MSG_TYPE_PROFRAW`) and stimulus
events (`MSG_TYPE_STIMULUS`). Each entry has a CRC32 for integrity
checking.

## Performance mode

When `performance_mode` is enabled on the builder, the VMM applies
optimizations across VM build, creation, and thread spawn:

1. Reads host LLC topology from sysfs.
2. Maps each virtual socket to a physical LLC group.
3. Pins each vCPU thread to a dedicated host core via
   `sched_setaffinity`.
4. Allocates guest memory with 2MB hugepages (`MAP_HUGETLB`)
   when sufficient free hugepages exist.
5. Sets KVM_HINTS_REALTIME in CPUID leaf 0x40000001 EDX,
   disabling PV spinlocks, PV TLB flush, and PV sched_yield
   in the guest, and enabling haltpoll cpuidle.
6. Disables PAUSE VM exits via `KVM_CAP_X86_DISABLE_EXITS`
   (PAUSE bit only; HLT exits remain for BSP shutdown detection).
7. Skips `KVM_CAP_HALT_POLL` (guest haltpoll cpuidle disables
   host halt polling via `MSR_KVM_POLL_CONTROL`).

Non-performance-mode VMs set `KVM_CAP_HALT_POLL` to 200µs (matching
the x86 kernel default) to reduce vCPU wakeup latency. Overcommitted
topologies (vCPUs > host CPUs) set it to 0.

Validation runs at build time. Oversubscription and unsatisfiable
topology mappings are fatal errors. Insufficient hugepages is a
warning -- the VM runs with regular pages rather than failing.

See [Performance Mode](../concepts/performance-mode.md) for usage
and prerequisites.

## Dual-role architecture

The same test binary serves two roles:

**Host side** -- manages the VM lifecycle: builds the initramfs, boots
the kernel, runs the monitor, and evaluates results.

**Guest side** -- runs inside the VM as `/init` (PID 1). The Rust init
code (`vmm::rust_init`) mounts filesystems, starts the scheduler,
dispatches the test function, then reboots.

The role is determined at runtime:

- **PID 1 detection**: when running as PID 1, the test harness
  `main()` calls `stt_guest_init()` which handles the full guest
  lifecycle. The `ctor` early dispatch also checks for PID 1.
- **`#[stt_test]` host dispatch**: a `#[ctor::ctor]` function
  (`stt_test_early_dispatch`) runs before `main()` in any binary
  that links against stt. When both `--stt-test-fn` and `--stt-topo`
  are present, it boots a VM and runs the test inside it.
- **`#[stt_test]` guest dispatch**: when only `--stt-test-fn` is
  present (no `--stt-topo`), the ctor runs the test function
  directly -- the binary is already inside a VM.

This design means one `cargo build` produces everything needed for
both host and guest execution. The initramfs embeds the same binary
that built it.

## Boot process

1. Load bzImage kernel via `linux-loader`.
2. Set up KVM vCPUs with the specified topology.
3. Build and load initramfs.
4. Set up serial devices (COM1 for console, COM2 for results).
5. Boot the kernel.
6. Kernel starts `/init` (the test binary).
7. PID 1 detected: `stt_guest_init()` mounts filesystems, starts
   the scheduler, dispatches the test function, and reboots.
