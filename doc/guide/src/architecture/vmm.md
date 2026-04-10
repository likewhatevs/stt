# VMM

scx-ktstr includes a purpose-built VMM (virtual machine monitor) that boots
Linux kernels in KVM for testing.

## KtstrVm builder

```rust,ignore
let result = vmm::KtstrVm::builder()
    .kernel(&kernel_path)
    .init_binary(&stt_binary)
    .topology(sockets, cores_per_socket, threads_per_core)
    .memory_mb(4096)
    .run_args(&["run".into(), "--ktstr-test-fn".into(), "my_test".into()])
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

**Serial console** -- COM2 carries guest stdout/stderr and serves as
a fallback result transport. Delimited test results (between
`===STT_TEST_RESULT_START===` / `===STT_TEST_RESULT_END===` sentinels)
and exit codes (`STT_EXIT=N`) are written to COM2 as fallback when
SHM is unavailable.

**SHM ring buffer** -- the primary guest-to-host data channel. A shared
memory ring buffer carries test results (`MSG_TYPE_TEST_RESULT`), exit
codes (`MSG_TYPE_EXIT`), profraw data (`MSG_TYPE_PROFRAW`), and stimulus
events (`MSG_TYPE_STIMULUS`). Each entry has a CRC32 for integrity
checking.

## Performance mode

When `performance_mode` is enabled, the VMM applies host-side
isolation (vCPU pinning, hugepages, NUMA mbind, RT scheduling),
guest-visible hints (KVM_HINTS_REALTIME CPUID), and KVM exit
suppression. Non-performance-mode VMs set `KVM_CAP_HALT_POLL` to
200us; overcommitted topologies set it to 0.

See [Performance Mode](../concepts/performance-mode.md) for the
full optimization list, prerequisites, and validation.

## Dual-role architecture

The same test binary serves two roles:

**Host side** -- manages the VM lifecycle: builds the initramfs, boots
the kernel, runs the monitor, and evaluates results.

**Guest side** -- runs inside the VM as `/init` (PID 1). The Rust init
code (`vmm::rust_init`) mounts filesystems, starts the scheduler,
dispatches the test function, then reboots.

The role is determined at runtime:

- **PID 1 detection**: when running as PID 1, the `#[ctor]` function
  `ktstr_test_early_dispatch()` calls `stt_guest_init()` which handles
  the full guest lifecycle.
- **`#[ktstr_test]` host dispatch**: a `#[ctor::ctor]` function
  (`ktstr_test_early_dispatch`) runs before `main()` in any binary
  that links against scx-ktstr. When both `--ktstr-test-fn` and `--ktstr-topo`
  are present, it boots a VM and runs the test inside it.
- **`#[ktstr_test]` guest dispatch**: when only `--ktstr-test-fn` is
  present (no `--ktstr-topo`), the ctor runs the test function
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
