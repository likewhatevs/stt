# VMM

ktstr includes a purpose-built VMM (virtual machine monitor) that boots
Linux kernels in KVM for testing.

## KtstrVm builder

```rust,ignore
let result = vmm::KtstrVm::builder()
    .kernel(&kernel_path)
    .init_binary(&ktstr_binary)
    .topology(numa_nodes, llcs, cores_per_llc, threads_per_core)
    .memory_mb(4096)
    .run_args(&["run".into(), "--ktstr-test-fn".into(), "my_test".into()])
    .build()?
    .run()?;
```

## Topology

The VM topology is specified as `(numa_nodes, llcs, cores_per_llc,
threads_per_core)`. On x86_64, the VMM creates ACPI tables (MADT,
SRAT, SLIT, and HMAT when `numa_nodes > 1`) and MP tables. On
aarch64, topology is expressed via FDT cpu nodes with MPIDR-derived
`reg` properties.

```rust,ignore
pub struct Topology {
    pub llcs: u32,
    pub cores_per_llc: u32,
    pub threads_per_core: u32,
    pub numa_nodes: u32,
    pub nodes: Option<&'static [NumaNode]>,
    pub distances: Option<&'static NumaDistance>,
}
```

`total_cpus()` = llcs * cores_per_llc * threads_per_core.
`num_llcs()` = llcs.

When `nodes` is `None` (the default), memory and LLCs are distributed
uniformly across NUMA nodes with default 10/20 distances. When
`Some`, each `NumaNode` specifies its LLC count, memory size, and
optional HMAT attributes (`latency_ns`, `bandwidth_mbs`,
`mem_side_cache`). A `NumaNode` with `llcs = 0` models a CXL
memory-only node.

`NumaDistance` is an NxN inter-node distance matrix. Diagonal entries
must be 10, off-diagonal > 10, and the matrix must be symmetric (ACPI
SLIT requirements).

Use `Topology::new(numa_nodes, llcs, cores, threads)` for uniform
topologies, or `Topology::with_nodes(cores, threads, &nodes)` for
explicit per-node configuration.

## initramfs

The VMM builds a cpio initramfs containing:

- The test binary (as `/init`)
- Optional scheduler binary (as `/scheduler`)
- Shared library dependencies (resolved via ELF DT_NEEDED parsing)

The initramfs is cached based on a cache key derived from the binary
contents. A compressed SHM segment enables COW overlay into guest
memory, sharing physical pages across concurrent VMs.

## Guest-host communication

**Serial console** -- COM2 carries guest stdout/stderr, the
canonical crash diagnostic transport, and a fallback result
transport. The guest panic hook writes `PANIC: <info>\n<bt>\n` to
COM2; the host parses it via `extract_panic_message` and surfaces
the backtrace in test failure output. Delimited test results
(between `===KTSTR_TEST_RESULT_START===` /
`===KTSTR_TEST_RESULT_END===` sentinels) and exit codes
(`KTSTR_EXIT=N`) are also written to COM2 as a fallback when the
TLV stream is unavailable.

**Virtio-console port 1 TLV stream** -- the primary guest-to-host
data channel. Carries scenario markers (`MSG_TYPE_SCENARIO_START`,
`MSG_TYPE_SCENARIO_END`), test results (`MSG_TYPE_TEST_RESULT`),
exit codes (`MSG_TYPE_EXIT`), stimulus events (`MSG_TYPE_STIMULUS`),
scheduler exit notifications (`MSG_TYPE_SCHED_EXIT`), profraw
coverage data (`MSG_TYPE_PROFRAW`), per-payload-invocation metrics
(`MSG_TYPE_PAYLOAD_METRICS`), and raw LlmExtract output
(`MSG_TYPE_RAW_PAYLOAD_OUTPUT`). Each TLV frame has a CRC32 for
integrity checking.

**SHM ring buffer** -- the early-boot fallback for the TLV stream.
When `/dev/vport0p1` is not yet available (multiport handshake in
progress), `shm_ring::write_msg` falls back to the SHM ring; the
host's bulk drain merges port-1 bytes with the SHM ring drain so
messages emitted before the port came up still reach the test
verdict. The SHM region also hosts the snapshot doorbell and
signal-slot control plane.

## Virtio devices

The VMM implements three virtio-MMIO devices in addition to the
serial console + SHM channels above. All three speak the virtio
1.x MMIO transport (virtio-v1.2 §4.2.2) with `VIRTIO_F_VERSION_1`
and use irqfd (eventfd → KVM GSI) for interrupt delivery.

- **virtio-blk** (`vmm::virtio_blk`) -- file-backed block device
  with a single request virtqueue and a token-bucket throttle.
  Used to give workloads real on-disk filesystems (per-test images
  cloned from a btrfs template). Advertises
  `VIRTIO_BLK_F_BLK_SIZE`, `VIRTIO_BLK_F_SEG_MAX`,
  `VIRTIO_BLK_F_SIZE_MAX`, `VIRTIO_BLK_F_FLUSH`, and
  `VIRTIO_RING_F_EVENT_IDX`, plus `VIRTIO_BLK_F_RO` when configured
  read-only.
- **virtio-net** (`vmm::virtio_net`) -- two-virtqueue (RX, TX) NIC
  with an in-VMM L2 loopback backend. Used by network-shaped
  workloads (TCP/UDP throughput, latency) without depending on the
  host's network stack. Advertises `VIRTIO_NET_F_MAC` so the guest
  binds a deterministic MAC.
- **virtio-console** (`vmm::virtio_console`) -- two-port multiport
  console with six virtqueues (per virtio-v1.2 §5.3.5: two console
  queues, two control queues, two bulk queues). Port 0 carries the
  interactive `/dev/hvc0` console alongside the COM1/COM2 16550
  serial ports; port 1 carries the guest-to-host TLV stream that
  delivers exit code, test result, per-payload metrics, raw payload
  outputs, profraw, and scheduler exit notifications. Advertises
  `VIRTIO_CONSOLE_F_MULTIPORT`.

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
  `ktstr_test_early_dispatch()` runs the guest init path, which handles
  the full guest lifecycle.
- **`#[ktstr_test]` host dispatch**: a `#[ctor::ctor]` function
  (`ktstr_test_early_dispatch`) runs before `main()` in any binary
  that links against ktstr. When both `--ktstr-test-fn` and `--ktstr-topo`
  are present, it boots a VM and runs the test inside it.
- **`#[ktstr_test]` guest dispatch**: when only `--ktstr-test-fn` is
  present (no `--ktstr-topo`), the ctor runs the test function
  directly -- the binary is already inside a VM.

This design means one `cargo build` produces everything needed for
both host and guest execution. The initramfs embeds the same binary
that built it.

## Boot process

1. Load kernel (bzImage on x86_64, Image on aarch64) via `linux-loader`.
2. Set up KVM vCPUs with the specified topology.
3. Build and load initramfs.
4. Set up serial devices (COM1 for console, COM2 for results).
5. Boot the kernel.
6. Kernel starts `/init` (the test binary).
7. PID 1 detected: the guest init path mounts filesystems, starts the
   scheduler, dispatches the test function, and reboots.
