//! Virtual machine monitor for booting Linux kernels in KVM to host
//! scheduler test scenarios.
//!
//! The entry point is [`KtstrVm::builder()`], which returns a
//! [`KtstrVmBuilder`] for configuring the kernel, init binary,
//! virtual topology, memory, host-side performance options, and
//! monitor thresholds. Calling `.build()?.run()?` on the result
//! boots the guest and returns a [`VmResult`] containing exit state,
//! captured console, monitor samples, and drained guest messages.
//!
//! See the [VMM architecture
//! page](https://likewhatevs.github.io/ktstr/guide/architecture/vmm.html)
//! for the boot flow and the [Performance Mode
//! page](https://likewhatevs.github.io/ktstr/guide/concepts/performance-mode.html)
//! for the isolation options the builder exposes.
//!
//! # Module layout
//!
//! `KtstrVm`'s implementation is split across several files. mod.rs
//! holds the canonical [`KtstrVm`] struct definition, [`KtstrVm::run`]
//! and [`KtstrVm::run_interactive`] entry points, and the public
//! re-exports. The remaining methods are reopened from the children
//! below via additional `impl KtstrVm` blocks:
//!
//! - [`builder`] — [`KtstrVmBuilder`], its `Default`, every setter,
//!   `build()`, and the host-resource acquisition helpers.
//! - [`setup`] — boot pipeline: virtio-blk init, KVM creation,
//!   initramfs resolution / compression / load, x86_64 + aarch64
//!   memory and FDT setup, vCPU register configuration.
//! - [`freeze_coord`] — run-loop orchestration: AP thread spawn,
//!   freeze coordinator, BPF map writer, BSP run loop, and result
//!   collection.
//! - [`contention`] — KVM EINTR retry policy, `HostResourceSnapshot`,
//!   `map_transient_to_contention`, and `create_vm_with_retry`.
//! - [`initramfs_cache`] — cross-process initramfs blob cache with
//!   POSIX SHM coordination.
//! - [`vcpu`] — vCPU thread infrastructure: `ImmediateExitHandle`,
//!   signal handler, thread pinning, RT priority, and perf capture.
//! - [`result`] — [`VmResult`], [`KvmStatsTotals`], `VmRunState`.

// `pub mod` — public sub-API surface that downstream callers may name
// directly. The arch-conditional modules (`aarch64`, `x86_64`) are
// also `pub` but live below where the cfg-gated re-exports for
// their contents are kept together.
//
// `disk_template` is `pub` for rustdoc cross-link visibility — its
// items are referenced from `disk_config`, `rust_init`, and the
// `KtstrVmBuilder::disk` doc as the canonical home for the disk
// template lifecycle. Downstream test authors do not call into it
// directly (the public path is via `KtstrVmBuilder::disk` plus the
// `Filesystem` enum), but rustdoc requires the module path to be
// reachable for the existing intra-doc-links to resolve.
pub mod cgroup_sandbox;
pub mod console;
pub mod disk_config;
pub mod disk_template;
pub mod host_topology;
pub mod initramfs;
pub(crate) mod kvm_stats;
pub mod topology;

// `pub(crate) mod` — crate-internal sub-modules.
pub(crate) mod builder;
pub(crate) mod capture_numa;
pub(crate) mod capture_scx;
pub(crate) mod capture_tasks;
pub(crate) mod contention;
pub(crate) mod exit_dispatch;
pub(crate) mod freeze_coord;
pub(crate) mod initramfs_cache;
pub(crate) mod net_config;
pub(crate) mod numa_mem;
pub(crate) mod result;
pub(crate) mod rust_init;
pub(crate) mod setup;
pub(crate) mod vcpu;
pub(crate) mod virtio_blk;
pub(crate) mod virtio_console;
pub(crate) mod virtio_net;

// Bulk transport modules. The wire format (`wire`), the host-side
// streaming assembler (`bulk`), the guest-side typed senders
// (`guest_comms`), and the host-side typed consumers (`host_comms`)
// each carry a single responsibility. Production data (STIMULUS /
// EXIT / SCHED_EXIT / PAYLOAD_METRICS / RAW_PAYLOAD_OUTPUT /
// SCENARIO_*) flows through the virtio-console port-1 TLV stream,
// and crash diagnostics travel via COM2.
pub(crate) mod bulk;
pub(crate) mod guest_comms;
pub(crate) mod host_comms;
pub mod wire;

// `mod` — file-private helpers.
mod memory_budget;
mod pi_mutex;
mod terminal;
mod vcpu_panic;
mod vmlinux;

// Re-export `VirtioBlkCounters` for users who hold a [`VmResult`]:
// `VmResult::virtio_blk_counters` exposes the device-side counter
// Arc, and the type itself must be reachable from the public path
// for a user to spell out `Arc<VirtioBlkCounters>` in their own
// signatures. The defining module stays `pub(crate)` because the
// device implementation is internal — this is the single public
// surface for the counters type.
// Re-export `NetConfig` and `VirtioNetCounters` for the same reason
// as `VirtioBlkCounters` below: user-facing test code holds a
// `VmResult` whose `virtio_net_counters` field carries the device's
// counter Arc, and `NetConfig` is the builder-side configuration
// type, so both names must be reachable from the public path. The
// in-tree readers go through the prelude path
// `crate::vmm::net_config::NetConfig`, so the lib build sees no
// direct readers of these names; allow unused-imports locally to
// keep `cargo check` quiet while preserving the public re-export.
#[allow(unused_imports)]
pub use net_config::NetConfig;
pub use virtio_blk::VirtioBlkCounters;
#[allow(unused_imports)]
pub use virtio_net::VirtioNetCounters;

// Re-export public result types from the new submodule.
// `KVM_INTERESTING_STATS` is part of the public surface for stats
// tooling — sidecar consumers reference these names by content, not
// by importing the constant, so the lib build sees no in-tree
// readers. Allow the unused-import lint locally to keep cargo check
// quiet while preserving the public re-export.
pub use builder::KtstrVmBuilder;
#[allow(unused_imports)]
pub use result::KVM_INTERESTING_STATS;
pub use result::{KvmStatsTotals, VmResult};

pub(crate) use contention::{
    create_vm_with_retry, host_resource_snapshot, map_transient_to_contention,
};
pub(crate) use pi_mutex::PiMutex;
pub(crate) use terminal::TerminalRawGuard;
pub(crate) use vcpu::{
    BpfMapWriteParams, ImmediateExitHandle, register_vcpu_signal_handler, set_thread_cpumask,
    vcpu_signal,
};
pub(crate) use vmlinux::find_vmlinux;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;
#[cfg(target_arch = "x86_64")]
pub mod x86_64;

// `acpi`, `boot`, `mptable` are re-exported as part of the public arch
// surface for downstream tooling. mod.rs itself does not consume them
// directly (boot/setup pipeline lives in `setup.rs` and reaches them
// via `super::x86_64::{...}`), so `unused_imports` would otherwise fire.
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
pub use x86_64::acpi;
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
pub use x86_64::boot;
#[cfg(target_arch = "x86_64")]
pub use x86_64::kvm;
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
pub use x86_64::mptable;

#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)]
pub use aarch64::boot;
#[cfg(target_arch = "aarch64")]
pub use aarch64::kvm;

pub use topology::Topology;

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Start of the guest physical address space used for RAM.
/// x86_64: PA 0 (sub-1MB legacy regions share the same PA space).
/// aarch64: device MMIO below DRAM_START, RAM above.
#[cfg(target_arch = "x86_64")]
const DRAM_BASE: u64 = 0;

#[cfg(target_arch = "aarch64")]
const DRAM_BASE: u64 = kvm::DRAM_START;

// ---------------------------------------------------------------------------
// KtstrVm — builder + run
// ---------------------------------------------------------------------------

/// Builder for creating and running VMs with custom topologies.
///
/// Methods are split across multiple files via additional
/// `impl KtstrVm` blocks: the boot pipeline lives in [`setup`]
/// (init_virtio_blk, setup_memory, setup_vcpus, plus aarch64
/// counterparts), and the run-loop orchestration lives in
/// [`freeze_coord`] (run_vm, spawn_ap_threads, start_monitor,
/// start_bpf_map_write, run_bsp_loop, collect_results).
pub struct KtstrVm {
    pub(crate) kernel: PathBuf,
    pub(crate) init_binary: Option<PathBuf>,
    pub(crate) scheduler_binary: Option<PathBuf>,
    pub(crate) run_args: Vec<String>,
    pub(crate) sched_args: Vec<String>,
    pub(crate) topology: Topology,
    /// Guest memory in MB. `None` = deferred: computed from actual
    /// initramfs size after the initramfs build completes.
    pub(crate) memory_mb: Option<u32>,
    /// Minimum memory in MB for deferred allocation. When non-zero,
    /// the deferred path uses `max(computed, memory_min_mb)` so topology
    /// configs that need more memory than the initramfs floor are honored.
    pub(crate) memory_min_mb: u32,
    pub(crate) cmdline_extra: String,
    pub(crate) timeout: Duration,
    /// Thresholds for reactive SysRq-D dump. When set and the monitor
    /// detects a sustained violation, it writes the dump flag to guest SHM.
    pub(crate) monitor_thresholds: Option<crate::monitor::MonitorThresholds>,
    /// Override for `scx_sched.watchdog_timeout` in the guest kernel.
    /// Converted to jiffies via CONFIG_HZ at monitor start time and
    /// written at each monitor iteration after the scheduler attaches.
    pub(crate) watchdog_timeout: Option<Duration>,
    /// Host-side BPF map writes. Empty slice disables the thread.
    /// When non-empty, a thread polls for BPF map discoverability,
    /// waits for scenario start via SHM ring, then writes each
    /// `u32` value at its specified map/offset. All writes complete
    /// before the guest is signaled via SHM slot 0, so the guest
    /// sees a single unblock regardless of how many writes ran.
    pub(crate) bpf_map_writes: Vec<BpfMapWriteParams>,
    /// Performance mode: vCPU pinning to host LLCs, hugepage-backed
    /// guest memory, NUMA mbind, and RT scheduling on both
    /// architectures. On x86_64, additionally: KVM_HINTS_REALTIME
    /// CPUID hint, PAUSE and HLT VM exit disabling via
    /// KVM_CAP_X86_DISABLE_EXITS, and KVM_CAP_HALT_POLL skipped
    /// (guest haltpoll cpuidle disables host halt polling via
    /// MSR_KVM_POLL_CONTROL). Oversubscription validation at build
    /// time on both architectures.
    pub(crate) performance_mode: bool,
    /// Pinning plan computed during build() when performance_mode is enabled.
    /// Stored so topology is read once and the plan is reused at VM start.
    pub(crate) pinning_plan: Option<host_topology::PinningPlan>,
    /// Per-guest-NUMA-node host NUMA nodes for mbind. Indexed by guest
    /// node ID. Each entry is the set of host NUMA nodes that the guest
    /// node's vCPUs are pinned to. Empty when performance_mode is off.
    pub(crate) mbind_node_map: Vec<Vec<usize>>,
    /// CPU flock fds for non-perf VMs. Held for the VM's lifetime to
    /// prevent other VMs from double-booking the same CPUs.
    #[allow(dead_code)]
    pub(crate) cpu_locks: Vec<std::os::fd::OwnedFd>,
    /// No-perf-mode resource plan. Populated for every no-perf-mode
    /// VM — either the operator-set CPU count
    /// (`--cpu-cap N` / `KTSTR_CPU_CAP=N`) or the 30%-of-allowed
    /// default when neither is present. Holds the flat CPU list +
    /// RAII flock fds returned by
    /// [`host_topology::acquire_llc_plan`]. `run_vm` reads the CPU
    /// list to `sched_setaffinity` every vCPU thread onto the
    /// reserved host CPUs, and `Drop` releases the LLC flocks with
    /// the VM.
    ///
    /// `None` only in the degraded-sysfs case (no-perf-mode on a
    /// host whose `/sys/devices/system/cpu` cannot be read AND no
    /// explicit cap was set — the build bails with an error when
    /// a cap IS set under the same sysfs failure), and for
    /// perf-mode (which uses `pinning_plan`). The two paths are
    /// orthogonal — perf-mode hard-pins single CPUs, --cpu-cap
    /// soft-masks a pool.
    #[allow(dead_code)]
    pub(crate) no_perf_plan: Option<host_topology::LlcPlan>,
    /// Shell commands to run in the guest to enable a kernel-built scheduler.
    pub(crate) sched_enable_cmds: Vec<String>,
    /// Shell commands to run in the guest to disable a kernel-built scheduler.
    pub(crate) sched_disable_cmds: Vec<String>,
    /// Files to include in the guest initramfs at their archive paths.
    /// Each entry is (archive_path, host_path).
    pub(crate) include_files: Vec<(String, PathBuf)>,
    /// v0 holds at most one DiskConfig; rendered as `/dev/vda`.
    /// Vec retained for future multi-disk expansion. The backing
    /// file is produced by the template-VM lifecycle (one-time
    /// guest-side `mkfs.<fstype>` against a sparse image, cached
    /// alongside the kernel; per-test reflink-copy at fan-out).
    /// Per-test boots populate the backing via the `Raw` tempfile
    /// or `Btrfs` cache-clone branches in
    /// [`KtstrVm::init_virtio_blk`]; the disk-template-build VM
    /// driver overrides both branches via
    /// [`Self::template_staging_image`] so it can format a
    /// host-staged image without re-entering its own cache.
    pub(crate) disks: Vec<disk_config::DiskConfig>,
    /// Optional network device. `None` skips virtio-net entirely:
    /// no FDT node, no MMIO range, no IRQ. `Some(_)` attaches one
    /// virtio-net device whose backend is the in-VMM loopback (TX
    /// bytes echoed back into RX). v0 supports a single device.
    pub(crate) network: Option<net_config::NetConfig>,
    /// Internal-only override for `init_virtio_blk`'s per-test
    /// backing-file allocation. `Some(path)` makes the device open
    /// `path` directly instead of allocating a fresh `tempfile()`
    /// or invoking [`disk_template::ensure_template`]. Set
    /// exclusively by [`KtstrVmBuilder::template_staging_image`] for
    /// the disk-template-build VM driver in
    /// [`disk_template::build_template_via_vm`]; `None` for every
    /// other code path. See the builder field's doc for the full
    /// recursion-break rationale.
    pub(crate) template_staging_image: Option<PathBuf>,
    /// Embed busybox in the initramfs for shell mode.
    pub(crate) busybox: bool,
    /// Forward COM1 (kernel console) to stderr in real-time during
    /// interactive shell mode. Useful for watching virtio probe and
    /// kernel messages alongside the shell session.
    pub(crate) dmesg: bool,
    /// Command to execute non-interactively in shell mode (--exec).
    /// Passed to the guest via /exec_cmd in the initramfs.
    pub(crate) exec_cmd: Option<String>,
    /// Optional host path to `ktstr-jemalloc-probe`. When `Some`, the
    /// probe is packed into the guest initramfs as an extra binary at
    /// `bin/ktstr-jemalloc-probe`. Consumed by `spawn_initramfs_resolve`.
    pub(crate) jemalloc_probe_binary: Option<PathBuf>,
    /// Optional host path to `ktstr-jemalloc-alloc-worker`. When
    /// `Some`, the worker is packed alongside the probe as an
    /// extra. The cross-process closed-loop test in
    /// `tests/jemalloc_probe_tests.rs` spawns it as a background
    /// payload and probes its pid.
    pub(crate) jemalloc_alloc_worker_binary: Option<PathBuf>,
    /// Where the freeze coordinator writes the JSON-pretty
    /// [`monitor::dump::FailureDumpReport`] when an error-class
    /// SCX exit fires. `None` disables the file sink (the dump
    /// still goes to `tracing::error` regardless). The test
    /// framework sets this to a per-test path under the run's
    /// sidecar directory so operators find the structured JSON
    /// alongside `*.ktstr.json` without needing an env var; CLI /
    /// library callers that want the dump on disk set the path
    /// explicitly via [`KtstrVmBuilder::failure_dump_path`].
    pub(crate) failure_dump_path: Option<PathBuf>,
    /// Capture two BPF-state snapshots per VM run: an early one when
    /// the host-side `runnable_at` scanner observes any task with
    /// `jiffies - p->scx.runnable_at > watchdog_timeout/2`
    /// (mirrors the kernel's `check_rq_for_timeouts`), and a late
    /// one at the same `ktstr_err_exit_detected` latch as the
    /// single-snapshot path. Emits
    /// [`monitor::dump::DualFailureDumpReport`] instead of the
    /// single-snapshot `FailureDumpReport`. Only the late snapshot
    /// is required — the early one is `None` when the stall fires
    /// before the half-way threshold trips, and the file is not
    /// written at all when only the early snapshot is captured (the
    /// run completed without a stall, so the early snapshot is not
    /// useful as a standalone artifact).
    ///
    /// Set by [`crate::test_support::probe::attempt_auto_repro`] for
    /// the repro VM only. Primary VMs leave this `false`; their
    /// freeze coordinator emits a [`monitor::dump::FailureDumpReport`]
    /// directly, matching the existing single-snapshot behaviour.
    pub(crate) dual_snapshot: bool,
}

impl KtstrVm {
    pub fn builder() -> KtstrVmBuilder {
        KtstrVmBuilder::default()
    }

    /// Borrow this VM's per-invocation initramfs-suffix inputs into an
    /// [`initramfs::SuffixParams`]. Centralizes the `run_args` /
    /// `sched_args` / sched-enable / sched-disable / `exec_cmd`
    /// bundling so both x86_64 and aarch64 paths construct the suffix
    /// from the same source of truth.
    fn suffix_params(&self) -> initramfs::SuffixParams<'_> {
        initramfs::SuffixParams {
            args: &self.run_args,
            sched_args: &self.sched_args,
            sched_enable: &self.sched_enable_cmds,
            sched_disable: &self.sched_disable_cmds,
            exec_cmd: self.exec_cmd.as_deref(),
        }
    }

    /// Boot the VM, run until shutdown/timeout, return captured output.
    pub fn run(&self) -> Result<VmResult> {
        let start = Instant::now();

        let initramfs_handle = self.spawn_initramfs_resolve();
        let (mut vm, kernel_result) = self.create_vm_and_load_kernel()?;

        #[cfg(target_arch = "x86_64")]
        let _kernel_result = {
            let kr = self.setup_memory(&mut vm, kernel_result, initramfs_handle)?;
            self.setup_vcpus(&vm, kr.entry)?;
            kr
        };
        #[cfg(target_arch = "aarch64")]
        let _kernel_result = {
            let kr = self.setup_memory_aarch64(&mut vm, kernel_result, initramfs_handle)?;
            self.setup_vcpus_aarch64(&vm, kr.entry)?;
            kr
        };

        // Open persistent stats fds before vCPUs move to threads.
        // Stats fds hold kernel references independent of VcpuFd ownership.
        // Read once after VM exit to capture cumulative totals.
        // KVM_GET_STATS_FD is generic uapi (include/uapi/linux/kvm.h),
        // so this works on every KVM-supported architecture.
        let stats_ctx = kvm_stats::open_stats_context(&vm.vcpus);
        if stats_ctx.is_none() {
            tracing::debug!("KVM_GET_STATS_FD not supported, skipping stats collection");
        }

        tracing::debug!(elapsed_us = start.elapsed().as_micros(), "total_setup");

        // Run-phase clock approximates the watchdog's hard_deadline
        // (both post-setup; the watchdog computes its deadline slightly
        // later, inside the spawned thread) so the BSP loop and monitor
        // thread don't charge VM setup overhead against the guest's
        // timeout budget.
        let run_start = Instant::now();

        let run = self.run_vm(run_start, vm)?;

        let mut result = self.collect_results(start, run)?;

        // Read cumulative KVM stats after VM exit.
        if let Some(ctx) = stats_ctx {
            result.kvm_stats = Some(ctx.read_stats());
        }

        Ok(result)
    }

    /// Boot the VM with bidirectional stdin/stdout forwarding via virtio-console.
    ///
    /// Sets the host terminal to raw mode, spawns threads for stdin->hvc0
    /// and hvc0->stdout forwarding, and runs until the guest shuts down.
    /// Terminal state is restored on all exit paths including panic and
    /// process-killing signals (SIGINT, SIGTERM, SIGQUIT).
    ///
    /// Builder settings ignored in interactive mode: `monitor_thresholds`,
    /// `watchdog_timeout`, `bpf_map_write`, `performance_mode` pinning,
    /// and KVM stats collection. These are test-specific features that
    /// do not apply to interactive shell sessions.
    pub fn run_interactive(&self) -> Result<()> {
        let start = Instant::now();

        let initramfs_handle = self.spawn_initramfs_resolve();
        let (mut vm, kernel_result) = self.create_vm_and_load_kernel()?;

        #[cfg(target_arch = "x86_64")]
        {
            let kr = self.setup_memory(&mut vm, kernel_result, initramfs_handle)?;
            self.setup_vcpus(&vm, kr.entry)?;
        }
        #[cfg(target_arch = "aarch64")]
        {
            let kr = self.setup_memory_aarch64(&mut vm, kernel_result, initramfs_handle)?;
            self.setup_vcpus_aarch64(&vm, kr.entry)?;
        }

        let com1 = Arc::new(PiMutex::new(console::Serial::new(console::COM1_BASE)));
        let com2 = Arc::new(PiMutex::new(console::Serial::new(console::COM2_BASE)));

        // Virtio-console for shell I/O via /dev/hvc0.
        let mut vc = virtio_console::VirtioConsole::new();
        vc.set_mem((*vm.guest_mem).clone());
        let virtio_con = Arc::new(PiMutex::new(vc));

        // Split-irqchip rejection: see freeze_coord.rs run_vm for the
        // full rationale. Without an IOAPIC, COM1/COM2/virtio-console
        // have no IRQ delivery path and the guest hangs on first I/O.
        #[cfg(target_arch = "x86_64")]
        if vm.split_irqchip {
            anyhow::bail!(
                "interactive shell requires irqfd; split-irqchip mode \
                 has no IOAPIC and the guest's serial / virtio-console \
                 drivers have no polling fallback — reduce topology \
                 so all APIC IDs are at or below 254 (MAX_XAPIC_ID)",
            );
        }
        #[cfg(target_arch = "x86_64")]
        {
            vm.vm_fd
                .register_irqfd(com1.lock().irq_evt(), console::COM1_IRQ)
                .context("register COM1 irqfd")?;
            vm.vm_fd
                .register_irqfd(com2.lock().irq_evt(), console::COM2_IRQ)
                .context("register COM2 irqfd")?;
            vm.vm_fd
                .register_irqfd(virtio_con.lock().irq_evt(), kvm::VIRTIO_CONSOLE_IRQ)
                .context("register virtio-console irqfd")?;
        }
        #[cfg(target_arch = "aarch64")]
        {
            vm.vm_fd
                .register_irqfd(com1.lock().irq_evt(), kvm::SERIAL_IRQ)
                .context("register serial irqfd")?;
            vm.vm_fd
                .register_irqfd(com2.lock().irq_evt(), kvm::SERIAL2_IRQ)
                .context("register serial2 irqfd")?;
            vm.vm_fd
                .register_irqfd(virtio_con.lock().irq_evt(), kvm::VIRTIO_CONSOLE_IRQ)
                .context("register virtio-console irqfd")?;
        }

        // Optional virtio-blk for shell mode. `None` when the builder
        // has no disks attached.
        let virtio_blk = self.init_virtio_blk(&vm)?;

        // Optional virtio-net for shell mode. `None` when the builder
        // has no `NetConfig` attached.
        let virtio_net = self.init_virtio_net(&vm)?;

        // Non-interactive exec mode (--exec) does not need a TTY.
        let exec_mode = self.exec_cmd.is_some();

        // Pre-flight: verify stdin is a tty, enter raw mode, and create
        // the wakeup pipe before spawning threads. Failing after thread
        // spawn would abandon AP threads.
        if !exec_mode {
            use std::os::unix::io::AsRawFd;
            let stdin_fd = std::io::stdin().as_raw_fd();
            let borrowed = unsafe { std::os::unix::io::BorrowedFd::borrow_raw(stdin_fd) };
            anyhow::ensure!(
                nix::unistd::isatty(borrowed).unwrap_or(false),
                "stdin must be a terminal for interactive shell mode",
            );
        }

        // Set host terminal to raw mode. TerminalRawGuard restores on drop
        // and installs signal handlers for SIGINT, SIGTERM, SIGQUIT,
        // SIGABRT, and SIGFPE so every terminating signal routes through
        // the terminal-restore path before the process exits (see
        // `src/terminal.rs`). Skip for exec mode — no interactive
        // terminal needed.
        let _raw_guard = if exec_mode {
            None
        } else {
            Some(TerminalRawGuard::enter().context("failed to set terminal to raw mode")?)
        };

        // Wakeup pipe: write end signals the stdin reader to exit when
        // the kill flag is set, avoiding a blocking read that prevents join.
        let (wakeup_r, wakeup_w) = nix::unistd::pipe().context("create stdin wakeup pipe")?;

        let kill = Arc::new(AtomicBool::new(false));
        // Companion eventfd for `kill`. The interactive shell has no
        // epoll consumer for it (kicks land via `pthread_kill` +
        // `immediate_exit`), but spawn_ap_threads requires a non-None
        // eventfd in its signature; allocate a sentinel and let it
        // drop with the function frame.
        let kill_evt = Arc::new(
            vmm_sys_util::eventfd::EventFd::new(libc::EFD_NONBLOCK)
                .context("create shell kill eventfd")?,
        );
        // Interactive shell does not arm the failure-dump freeze
        // pipeline (no monitor thread requesting freezes). Construct
        // sentinel flags that stay false for the lifetime of the
        // session so vcpu_run_loop_unified / run_bsp_loop see a stable
        // freeze=false on every iteration and never enter the park
        // path.
        let freeze = Arc::new(AtomicBool::new(false));
        // Interactive shell never runs the freeze coordinator, so
        // `request_kva` stays 0 and `self_arm_watchpoint` is a no-op
        // on every iteration. Allocated only to satisfy the
        // spawn_ap_threads / run_bsp_loop signatures shared with the
        // failure-dump path.
        let watchpoint =
            Arc::new(vcpu::WatchpointArm::new().context("create WatchpointArm.hit_evt EventFd")?);
        let bsp_parked = Arc::new(AtomicBool::new(false));
        let bsp_regs: Arc<std::sync::Mutex<Option<exit_dispatch::VcpuRegSnapshot>>> =
            Arc::new(std::sync::Mutex::new(None));
        let has_immediate_exit = vm.has_immediate_exit;
        let mut vcpus = std::mem::take(&mut vm.vcpus);
        let mut bsp = vcpus.remove(0);

        let ap_pins = vec![None; vcpus.len()];
        // Shell/interactive path mirrors run_vm: no-perf + --cpu-cap
        // applies the LlcPlan's CPU list as a sched_setaffinity mask
        // on every vCPU thread. Perf-mode's pin_targets doesn't
        // apply here — interactive shell runs under no-perf by
        // convention, and `pin_targets` is empty in this branch.
        let no_perf_mask: Option<&[usize]> = self.no_perf_plan.as_ref().map(|p| p.cpus.as_slice());
        // Interactive shell does not run a freeze coordinator, so
        // discard the freeze-handle Vecs. Interactive mode also skips
        // the perf-counter capture path; allocate empty TID slots so
        // the spawn signature is honored without producing values
        // anything reads.
        let n_aps = vcpus.len();
        let ap_tid_slots: Vec<(Arc<AtomicI32>, Arc<crate::sync::Latch>)> = (0..n_aps)
            .map(|_| {
                (
                    Arc::new(AtomicI32::new(0)),
                    Arc::new(crate::sync::Latch::new()),
                )
            })
            .collect();
        let (ap_threads, _ap_freeze) = self.spawn_ap_threads(
            vcpus,
            has_immediate_exit,
            &com1,
            &com2,
            Some(&virtio_con),
            virtio_blk.as_ref(),
            virtio_net.as_ref(),
            &kill,
            &kill_evt,
            &freeze,
            &watchpoint,
            &ap_pins,
            no_perf_mask,
            &ap_tid_slots,
            // Interactive shell does not run a freeze coordinator,
            // so no parked_evt / thaw_evt to plumb. The
            // `vcpu_run_loop_unified` honours `freeze` only when it
            // flips, which never happens in this path; the
            // eventfds remain unused.
            None,
            None,
        )?;

        // BSP kick handles for the stdin escape sequence. The stdin thread
        // needs to force the BSP out of KVM_RUN when Ctrl+A X is pressed.
        let bsp_ie_for_stdin = if has_immediate_exit {
            Some(ImmediateExitHandle::from_vcpu(&mut bsp))
        } else {
            None
        };
        let bsp_tid = unsafe { libc::pthread_self() };

        // Stdin reader thread: host stdin -> virtio-console RX queue.
        // The guest reads stdin from /dev/hvc0 (virtio-console), never
        // from COM2. pending_rx buffers input until the guest activates
        // the RX queue. Uses poll() on both stdin and the wakeup pipe
        // so the thread can be cleanly joined on shutdown.
        //
        // Escape sequence: Ctrl+A X (0x01 followed by 'x' or 'X') triggers
        // host-side VM teardown without guest cooperation.
        let vc_for_stdin = virtio_con.clone();
        let kill_for_stdin = kill.clone();
        let stdin_thread = std::thread::Builder::new()
            .name("interactive-stdin".into())
            .spawn(move || {
                use std::io::Read;
                use std::os::unix::io::{AsFd, AsRawFd};

                // wakeup_r is an OwnedFd moved into this closure; closed on exit.
                let wakeup_fd = wakeup_r;
                let stdin_fd = std::io::stdin().as_raw_fd();
                let mut buf = [0u8; 4096];
                let mut saw_ctrl_a = false;

                loop {
                    if kill_for_stdin.load(Ordering::Acquire) {
                        break;
                    }

                    // Poll stdin and wakeup fd with 100ms timeout.
                    let stdin_borrowed =
                        unsafe { std::os::unix::io::BorrowedFd::borrow_raw(stdin_fd) };
                    let wakeup_borrowed = wakeup_fd.as_fd();
                    let mut fds = [
                        nix::poll::PollFd::new(stdin_borrowed, nix::poll::PollFlags::POLLIN),
                        nix::poll::PollFd::new(wakeup_borrowed, nix::poll::PollFlags::POLLIN),
                    ];
                    match nix::poll::poll(&mut fds, 100u16) {
                        Ok(0) => continue, // timeout
                        Err(nix::errno::Errno::EINTR) => continue,
                        Err(_) => break,
                        Ok(_) => {}
                    }

                    // Wakeup fd readable means shutdown requested.
                    if fds[1]
                        .revents()
                        .is_some_and(|r| r.intersects(nix::poll::PollFlags::POLLIN))
                    {
                        break;
                    }

                    // Stdin readable.
                    if fds[0]
                        .revents()
                        .is_some_and(|r| r.intersects(nix::poll::PollFlags::POLLIN))
                    {
                        let mut stdin = std::io::stdin().lock();
                        match stdin.read(&mut buf) {
                            Ok(0) => break,
                            Ok(n) => {
                                // Scan for Ctrl+A X escape sequence. Filter
                                // escape bytes from the forwarded input so
                                // neither the 0x01 nor 'x'/'X' reaches the
                                // guest.
                                let mut forward_start = 0usize;
                                for i in 0..n {
                                    if saw_ctrl_a {
                                        saw_ctrl_a = false;
                                        if buf[i] == b'x' || buf[i] == b'X' {
                                            // Trigger host-side teardown. Bytes
                                            // before the Ctrl+A were already
                                            // flushed when saw_ctrl_a was set.
                                            eprintln!("\r\nTerminated.");
                                            kill_for_stdin.store(true, Ordering::Release);
                                            if let Some(ref ie) = bsp_ie_for_stdin {
                                                ie.set(1);
                                                std::sync::atomic::fence(Ordering::Release);
                                            }
                                            unsafe {
                                                libc::pthread_kill(bsp_tid, vcpu_signal());
                                            }
                                            return;
                                        }
                                        // Not 'x'/'X' after Ctrl+A: the 0x01
                                        // was a real keystroke. Flush any
                                        // unflushed bytes preceding this point
                                        // first so the deferred 0x01 lands in
                                        // chronological order, then queue the
                                        // 0x01, then continue processing from
                                        // `i` onward (current byte may itself
                                        // be 0x01).
                                        if forward_start < i {
                                            vc_for_stdin.lock().queue_input(&buf[forward_start..i]);
                                            forward_start = i;
                                        }
                                        vc_for_stdin.lock().queue_input(&[0x01]);
                                    }
                                    if buf[i] == 0x01 {
                                        // Flush bytes before the Ctrl+A.
                                        if forward_start < i {
                                            vc_for_stdin.lock().queue_input(&buf[forward_start..i]);
                                        }
                                        saw_ctrl_a = true;
                                        forward_start = i + 1;
                                        continue;
                                    }
                                }
                                // Forward remaining bytes.
                                if forward_start < n {
                                    vc_for_stdin.lock().queue_input(&buf[forward_start..n]);
                                }
                            }
                            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                            Err(_) => break,
                        }
                    }
                }
            })
            .context("spawn stdin reader thread")?;

        // Stdout writer thread: virtio-console TX -> host stdout.
        // Polls tx_evt for zero-latency wakeup when guest writes data.
        // On write errors (including BrokenPipe), sets kill flag and exits
        // to stop the VM rather than polling a dead pipe until timeout.
        let vc_for_stdout = virtio_con.clone();
        let kill_for_stdout = kill.clone();
        let stdout_thread: JoinHandle<bool> = std::thread::Builder::new()
            .name("interactive-stdout".into())
            .spawn(move || {
                use std::io::Write;

                let mut wrote_any = false;

                // Cache the raw fd for poll. The eventfd lives as long as
                // VirtioConsole which is behind Arc<PiMutex> — valid for
                // the thread's lifetime.
                let tx_evt_raw_fd = {
                    let guard = vc_for_stdout.lock();
                    std::os::unix::io::AsRawFd::as_raw_fd(guard.tx_evt())
                };
                let mut stdout = std::io::stdout().lock();
                loop {
                    if kill_for_stdout.load(Ordering::Acquire) {
                        break;
                    }
                    let borrowed =
                        unsafe { std::os::unix::io::BorrowedFd::borrow_raw(tx_evt_raw_fd) };
                    let mut fds = [nix::poll::PollFd::new(
                        borrowed,
                        nix::poll::PollFlags::POLLIN,
                    )];
                    match nix::poll::poll(&mut fds, 50u16) {
                        Ok(0) => continue,
                        Err(nix::errno::Errno::EINTR) => continue,
                        Err(_) => break,
                        Ok(_) => {
                            // Consume eventfd counter.
                            let _ = vc_for_stdout.lock().tx_evt().read();
                        }
                    }
                    // Re-check kill after poll. During shutdown the
                    // dying guest may enqueue a stray byte into the
                    // virtio TX queue (from kernel hvc_close flushing
                    // n_outbuf via tty_wait_until_sent → hvc_push →
                    // put_chars). That byte passes from_utf8 (valid
                    // single-byte UTF-8) but is unprintable, producing
                    // a garbled character on the terminal.
                    if kill_for_stdout.load(Ordering::Acquire) {
                        break;
                    }
                    let data = vc_for_stdout.lock().drain_output();
                    if !data.is_empty() {
                        // Write only valid UTF-8 prefix. Trailing
                        // incomplete sequences (from guest shutdown
                        // mid-write) are dropped to prevent garbled
                        // output.
                        let valid_len = match std::str::from_utf8(&data) {
                            Ok(_) => data.len(),
                            Err(e) => e.valid_up_to(),
                        };
                        if valid_len > 0 {
                            if stdout.write_all(&data[..valid_len]).is_err()
                                || stdout.flush().is_err()
                            {
                                kill_for_stdout.store(true, Ordering::Release);
                                break;
                            }
                            wrote_any = true;
                        }
                    }
                }
                // Final drain: the guest may have flushed output just
                // before shutdown that hasn't been polled yet.
                let data = vc_for_stdout.lock().drain_output();
                if !data.is_empty() {
                    let valid_len = match std::str::from_utf8(&data) {
                        Ok(_) => data.len(),
                        Err(e) => e.valid_up_to(),
                    };
                    if valid_len > 0 {
                        let _ = stdout.write_all(&data[..valid_len]);
                        let _ = stdout.flush();
                        wrote_any = true;
                    }
                }
                wrote_any
            })
            .context("spawn stdout writer thread")?;

        // Optional dmesg thread: COM1 -> stderr in real-time.
        // Only spawned when --dmesg is active. Gives the user kernel
        // messages (including virtio probe results) alongside the shell.
        //
        // The thread blocks in `epoll_wait` on two fds:
        //   * `data_evt` — bumped by `Serial::handle_out` whenever a
        //     guest port write appends a byte to COM1's captured-output
        //     buffer (see `Serial::install_data_evt`). Fires on every
        //     guest-side console write.
        //   * `dmesg_wakeup_evt` — a shutdown wakeup the BSP-cleanup
        //     code below pulses after flipping `kill` so the thread
        //     exits the wait promptly without sleep-polling.
        // Replaces a 50ms sleep+poll loop on `drain_output`.
        let (dmesg_thread, dmesg_wakeup_evt) = if self.dmesg {
            use std::os::unix::io::AsRawFd;
            use vmm_sys_util::epoll::{ControlOperation, Epoll, EpollEvent, EventSet};
            use vmm_sys_util::eventfd::{EFD_NONBLOCK, EventFd};

            let data_evt = com1
                .lock()
                .install_data_evt()
                .context("install COM1 dmesg data eventfd")?;
            let wakeup_evt =
                Arc::new(EventFd::new(EFD_NONBLOCK).context("create dmesg wakeup eventfd")?);
            let com1_for_dmesg = com1.clone();
            let kill_for_dmesg = kill.clone();
            let wakeup_for_thread = wakeup_evt.clone();
            const DATA_TOKEN: u64 = 0;
            const WAKEUP_TOKEN: u64 = 1;
            let handle = std::thread::Builder::new()
                .name("interactive-dmesg".into())
                .spawn(move || {
                    use std::io::Write;
                    let epoll = match Epoll::new() {
                        Ok(e) => e,
                        Err(e) => {
                            tracing::warn!(%e, "interactive-dmesg: epoll_create1 failed");
                            return;
                        }
                    };
                    if let Err(e) = epoll.ctl(
                        ControlOperation::Add,
                        data_evt.as_raw_fd(),
                        EpollEvent::new(EventSet::IN, DATA_TOKEN),
                    ) {
                        tracing::warn!(%e, "interactive-dmesg: add data_evt to epoll");
                        return;
                    }
                    if let Err(e) = epoll.ctl(
                        ControlOperation::Add,
                        wakeup_for_thread.as_raw_fd(),
                        EpollEvent::new(EventSet::IN, WAKEUP_TOKEN),
                    ) {
                        tracing::warn!(%e, "interactive-dmesg: add wakeup to epoll");
                        return;
                    }
                    let mut events = [EpollEvent::default(); 2];
                    // Lock stderr per-write, not for the whole loop.
                    // Holding the lock blocks Ctrl+A X's eprintln.
                    loop {
                        if kill_for_dmesg.load(Ordering::Acquire) {
                            break;
                        }
                        match epoll.wait(-1, &mut events) {
                            Ok(_) => {}
                            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                            Err(e) => {
                                tracing::warn!(%e, "interactive-dmesg: epoll_wait failed");
                                break;
                            }
                        }
                        // Drain both eventfd counters (counter mode —
                        // a single read returns the accumulated count
                        // and resets it; spurious EAGAIN from a racing
                        // refill is harmless).
                        let _ = data_evt.read();
                        let _ = wakeup_for_thread.read();
                        let data = com1_for_dmesg.lock().drain_output();
                        if !data.is_empty() {
                            let mut stderr = std::io::stderr().lock();
                            let _ = stderr.write_all(&data);
                            let _ = stderr.flush();
                        }
                    }
                    // Final drain.
                    let data = com1_for_dmesg.lock().drain_output();
                    if !data.is_empty() {
                        let mut stderr = std::io::stderr().lock();
                        let _ = stderr.write_all(&data);
                        let _ = stderr.flush();
                    }
                })
                .context("spawn dmesg thread")?;
            (Some(handle), Some(wakeup_evt))
        } else {
            (None, None)
        };

        // BSP run loop (same shutdown detection as run()).
        // Interactive sessions are user-controlled; the builder's timeout
        // (default 60s) must not kill the shell. Use 24 hours as a
        // practical upper bound.
        //
        // Apply the no-perf + --cpu-cap mask to the BSP thread so
        // interactive `ktstr shell --no-perf-mode --cpu-cap N` runs
        // inside the reserved LLCs just like run_vm's BSP. No pin
        // here — perf-mode doesn't apply to interactive shell:
        // `--cpu-cap` requires `--no-perf-mode` on Shell (clap
        // `requires` attribute on the cpu_cap field).
        if let Some(mask) = self.no_perf_plan.as_ref().map(|p| p.cpus.as_slice()) {
            set_thread_cpumask(mask, "BSP (shell)");
        }
        register_vcpu_signal_handler();
        let interactive_timeout = Duration::from_secs(24 * 60 * 60);
        self.run_bsp_loop(
            &mut bsp,
            &com1,
            &com2,
            Some(&virtio_con),
            virtio_blk.as_ref(),
            virtio_net.as_ref(),
            &kill,
            &freeze,
            &watchpoint,
            &bsp_parked,
            &bsp_regs,
            has_immediate_exit,
            start,
            interactive_timeout,
            // Interactive shell never sets `freeze`, so the
            // handle_freeze branch is unreachable in this path.
            // Pass None for the wake-fd handles — the legacy
            // park_timeout cadence is the safe-by-construction
            // fallback.
            None,
            None,
            None,
            // Interactive shell does not construct a GuestKernel
            // for monitor / BPF map writes, so no TCR_EL1 cache
            // is needed.
            None,
            // CR3 cache: unused in interactive shell (no monitor
            // thread, no phys_base resolution).
            &std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        );

        // Shutdown.
        kill.store(true, Ordering::Release);

        // Wake the stdin reader so it exits poll() and can be joined.
        let _ = nix::unistd::write(&wakeup_w, &[0u8]);
        drop(wakeup_w);

        // Wake the dmesg thread so it exits epoll_wait promptly and
        // can be joined. The kill load above the loop short-circuits
        // any pending iteration; this bump ensures the wait returns
        // immediately rather than blocking on the next byte from the
        // guest after teardown.
        if let Some(ref evt) = dmesg_wakeup_evt {
            let _ = evt.write(1);
        }

        for vt in &ap_threads {
            if !vt.exited.load(Ordering::Acquire) {
                vt.kick();
            }
        }
        for vt in ap_threads {
            vt.wait_for_exit(Duration::from_secs(5));
            let _ = vt.handle.join();
        }

        let stdout_wrote = stdout_thread.join().unwrap_or(false);
        let _ = stdin_thread.join();
        if let Some(dt) = dmesg_thread {
            let _ = dt.join();
        }
        drop(dmesg_wakeup_evt);

        // _raw_guard drops here, restoring terminal and signal handlers.
        drop(_raw_guard);

        // Exec mode fallback: if virtio-console produced no output
        // (kernel lacks CONFIG_VIRTIO_CONSOLE, guest fell back to
        // COM2), print COM2 output to stdout so the caller sees it.
        // Filter out the KTSTR_EXEC_EXIT sentinel which the guest
        // writes to stderr (also COM2 in the fallback case).
        if exec_mode && !stdout_wrote {
            let app_output = com2.lock().output();
            if !app_output.is_empty() {
                use std::io::Write;
                let mut stdout = std::io::stdout().lock();
                // Pre-bincode-migration the guest emitted a
                // `KTSTR_EXEC_EXIT=N` sentinel line on COM2 that
                // needed filtering out of this stdout copy. The
                // exec exit is now a typed `MSG_TYPE_EXEC_EXIT`
                // frame on the bulk data port (see
                // `crate::vmm::guest_comms::send_exec_exit`), so
                // the sentinel never appears in COM2 — no filter
                // needed. Write the captured bytes verbatim.
                let _ = stdout.write_all(app_output.as_bytes());
                let _ = stdout.flush();
            }
        }

        // Print kernel console output (COM1) to stderr if non-empty.
        // Skip when --dmesg was active (already streamed to stderr).
        if !self.dmesg {
            let console_output = com1.lock().output();
            if !console_output.is_empty() {
                eprintln!("{console_output}");
            }
        }

        if !exec_mode {
            eprintln!("Connection to VM closed.");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn ap_mp_state_set_correctly() {
        let topo = Topology {
            llcs: 2,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let vm = kvm::KtstrKvm::new(topo, 128, false).unwrap();
        for vcpu in &vm.vcpus[1..] {
            let state = vcpu.get_mp_state().unwrap();
            assert_eq!(
                state.mp_state,
                kvm_bindings::KVM_MP_STATE_UNINITIALIZED,
                "AP should default to UNINITIALIZED"
            );
        }
    }
    /// Boot a real kernel and verify it produces console output.
    /// No initramfs — the kernel boots to panic, which is enough to
    /// confirm KVM, kernel loading, and serial console all work.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn boot_kernel_produces_output() {
        let kernel = crate::test_support::require_kernel();

        let vm = skip_on_contention!(
            KtstrVm::builder()
                .kernel(&kernel)
                .topology(1, 1, 1, 1)
                .memory_mb(256)
                .timeout(Duration::from_secs(10))
                .cmdline("loglevel=7")
                .build()
        );
        let result = vm.run().unwrap();
        assert!(
            result.stderr.contains("Linux") || result.stderr.contains("Booting"),
            "kernel console should contain boot messages"
        );
    }

    /// Boot with SMP topology and verify kernel detects multiple CPUs.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn boot_kernel_smp_topology() {
        let kernel = crate::test_support::require_kernel();

        let vm = skip_on_contention!(
            KtstrVm::builder()
                .kernel(&kernel)
                .topology(1, 2, 2, 1) // 4 CPUs
                .memory_mb(256)
                .timeout(Duration::from_secs(10))
                .cmdline("loglevel=7")
                .build()
        );
        let result = vm.run().unwrap();
        assert!(!result.stderr.is_empty(), "no console output from SMP boot");
    }

    /// Benchmark: measure VM boot time to kernel panic (no init = fastest path).
    /// The kernel boots, finds no initramfs, panics. The panic timestamp
    /// IS the boot time. With `panic=-1`, the kernel calls
    /// `emergency_restart()` which triggers an I8042 reset (port 0x64,
    /// 0xFE via `reboot=k`), returning to userspace.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn bench_boot_time() {
        let kernel = crate::test_support::require_kernel();

        for (label, llcs, cores, threads, mem) in [("1cpu", 1, 1, 1, 256), ("4cpu", 2, 2, 1, 512)] {
            let start = Instant::now();
            let vm = match KtstrVm::builder()
                .kernel(&kernel)
                .topology(1, llcs, cores, threads)
                .memory_mb(mem)
                .timeout(Duration::from_secs(10))
                .build()
            {
                Ok(vm) => vm,
                Err(e)
                    if e.downcast_ref::<host_topology::ResourceContention>()
                        .is_some() =>
                {
                    crate::report::test_skip(format_args!("{label}: resource contention: {e}"));
                    continue;
                }
                Err(e) => panic!("{e:#}"),
            };
            let setup = start.elapsed();
            let result = vm.run().unwrap();
            // Extract kernel timestamp from last line (e.g. "[    0.189300] Kernel panic")
            let boot_ms = result
                .stderr
                .lines()
                .rev()
                .find(|l| l.contains("Kernel panic") || l.contains("end Kernel panic"))
                .and_then(|l| {
                    l.trim()
                        .strip_prefix('[')
                        .and_then(|s| s.split(']').next())
                        .and_then(|s| s.trim().parse::<f64>().ok())
                })
                .map(|s| (s * 1000.0) as u64)
                .unwrap_or(0);
            eprintln!(
                "BENCH {label}: setup={:.0}ms kernel_boot={boot_ms}ms wall={:.0}ms timed_out={}",
                setup.as_millis(),
                result.duration.as_millis(),
                result.timed_out,
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn kvm_has_immediate_exit_cap() {
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let vm = kvm::KtstrKvm::new(topo, 64, false).unwrap();
        // KVM_CAP_IMMEDIATE_EXIT has been available since Linux 4.12.
        assert!(
            vm.has_immediate_exit,
            "KVM_CAP_IMMEDIATE_EXIT should be available on modern kernels"
        );
    }
    /// Boot a kernel with vmlinux available and verify the monitor
    /// produces samples with meaningful runqueue data and degrades
    /// gracefully for scx_root-gated paths.
    ///
    /// No scheduler is loaded. Event counters (gated on scx_root)
    /// must be None. Watchdog observation may be Some on kernels
    /// with a static watchdog_timeout symbol (pre-7.1); if present,
    /// the write/read roundtrip must match.
    #[test]
    fn boot_kernel_with_monitor() {
        let kernel = crate::test_support::require_kernel();
        let _vmlinux = crate::test_support::require_vmlinux(&kernel);

        // 5s timeout, 2s watchdog: monitor-only test. The in-monitor
        // 5s sys_rdy ceiling caps the worst case; 2s watchdog gives
        // the host-write override a tight observable value while
        // staying well above the kernel's per-tick granularity.
        let vm = skip_on_contention!(
            KtstrVm::builder()
                .kernel(&kernel)
                .topology(1, 1, 2, 1)
                .memory_mb(256)
                .timeout(Duration::from_secs(5))
                .watchdog_timeout(Duration::from_secs(2))
                .build()
        );
        let result = vm.run().unwrap();
        let Some(ref report) = result.monitor else {
            return;
        };
        assert!(
            report.summary.total_samples > 0,
            "monitor should have collected at least one sample"
        );

        // Scan samples in reverse for the first one where ANY CPU
        // reports rq_clock past the early-boot noise floor.
        let populated = report
            .samples
            .iter()
            .rev()
            .find(|s| s.cpus.iter().any(|c| c.rq_clock > 1_000_000))
            .expect(
                "no monitor sample showed populated runqueue data — every sample \
                 had all CPUs at rq_clock <= 1ms, \
                 or the monitor is reading the wrong rq offsets",
            );
        assert_eq!(
            populated.cpus.len(),
            2,
            "topology requested 2 CPUs but monitor saw {}",
            populated.cpus.len()
        );
        for (i, cpu) in populated.cpus.iter().enumerate() {
            if cpu.rq_clock <= 1_000_000 {
                continue;
            }
            assert!(
                cpu.rq_clock < 300_000_000_000,
                "cpu {i}: rq_clock must be < 300s (ns), got {}",
                cpu.rq_clock
            );
        }
        if let Some(ref obs) = report.watchdog_observation {
            assert_eq!(
                obs.expected_jiffies, obs.observed_jiffies,
                "watchdog write/read roundtrip mismatch: expected={} observed={}",
                obs.expected_jiffies, obs.observed_jiffies
            );
        }
        for (i, cpu) in populated.cpus.iter().enumerate() {
            assert!(
                cpu.event_counters.is_none(),
                "cpu {i}: event_counters must be None when no scheduler is loaded"
            );
        }
    }

    /// Asserts the monitor's `DATA_VALID` latch fires before the run
    /// ends and records the live KASLR-randomized `page_offset`. The
    /// per-iteration refresh in `monitor_loop` reads
    /// `page_offset_base` from guest memory once the guest BSP has
    /// completed `setup_per_cpu_areas` and KASLR randomization, then
    /// latches `page_offset` for every subsequent KVA→PA translation.
    /// This test fails if the latch never fires (`page_offset == 0`),
    /// proving the boot signal + refresh pipeline reaches the
    /// `__per_cpu_offset[0]` populated && `page_offset_resolved`
    /// AND condition before the run closes.
    ///
    /// Rationale: the same wrong `page_offset` would make every
    /// `kva_to_pa` translation off by the KASLR delta and zero out
    /// every monitor read. `boot_kernel_with_monitor`'s
    /// `rq_clock > 1ms` assertion only fires when the read landed in
    /// DRAM — but the test does not distinguish "latch never fired"
    /// (page_offset stays at 0 here) from "latch fired but data still
    /// pre-boot." Probing the latched value directly closes that gap.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn monitor_data_valid_latch_records_live_page_offset() {
        let kernel = crate::test_support::require_kernel();
        let _vmlinux = crate::test_support::require_vmlinux(&kernel);

        let vm = skip_on_contention!(
            KtstrVm::builder()
                .kernel(&kernel)
                .topology(1, 1, 2, 1)
                .memory_mb(256)
                .timeout(Duration::from_secs(5))
                .watchdog_timeout(Duration::from_secs(2))
                .build()
        );
        let result = vm.run().unwrap();
        let Some(ref report) = result.monitor else {
            return;
        };
        assert!(
            report.summary.total_samples > 0,
            "monitor produced no samples — DATA_VALID latch \
             observability cannot be evaluated"
        );

        // x86_64: DATA_VALID requires page_offset_resolved (bit 63 +
        // 4 KiB alignment + stability gate) AND
        // __per_cpu_offset[0] != 0. A non-zero `report.page_offset`
        // proves both conjuncts held during at least one iteration.
        assert_ne!(
            report.page_offset, 0,
            "DATA_VALID latch never fired during the run — \
             monitor.page_offset stayed at the initial 0 sentinel. \
             page_offset_base was never resolved or \
             __per_cpu_offset[0] never became non-zero before the \
             run closed",
        );

        // Bit 63 set: kernel half on x86_64 (canonical addresses
        // with VA_BITS=48 occupy 0xffff_8000_0000_0000 and above).
        // The latch's own gate enforces this same bit, so any
        // value here that lacks bit 63 means the assertion suite
        // is reading garbage rather than a live latch capture.
        assert!(
            report.page_offset & (1u64 << 63) != 0,
            "monitor.page_offset {:#x} is not in the canonical \
             upper half — page_offset_resolved gate accepted a \
             user-space address",
            report.page_offset,
        );

        // 4 KiB page alignment: kernel PAGE_OFFSET is page-aligned
        // by construction. The latch gate also enforces this; a
        // misaligned value here would be a regression in either
        // the gate or the field plumbing.
        assert_eq!(
            report.page_offset & 0xFFF,
            0,
            "monitor.page_offset {:#x} is not 4 KiB aligned",
            report.page_offset,
        );
    }

    /// End-to-end check that the SYS_RDY eventfd actually unblocks
    /// the monitor's pre-sample boot wait — the failure mode that
    /// motivated this test. With sys_rdy wired correctly, the guest
    /// publishes [`crate::vmm::wire::MSG_TYPE_SYS_RDY`] after
    /// `mount_filesystems()` and the monitor advances into the
    /// sample loop within seconds of boot — well under the
    /// in-monitor 5 s `boot_epoll.wait` ceiling.
    ///
    /// The first emitted sample's `elapsed_ms` therefore must land
    /// well below the in-monitor 5 s ceiling — anything ≥ 4 s
    /// here means a sys_rdy regression let the wait time out and
    /// the sample loop only started after the fall-through, which
    /// is exactly the regression this test is meant to surface.
    ///
    /// Returns silently (test-skip-equivalent) when the host has
    /// no kernel / no vmlinux / no scx_root etc.; the assertions
    /// only fire on a real run that produced a `MonitorReport`.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn sys_rdy_releases_monitor_before_5s_timeout() {
        let kernel = crate::test_support::require_kernel();
        let _vmlinux = crate::test_support::require_vmlinux(&kernel);

        let vm = skip_on_contention!(
            KtstrVm::builder()
                .kernel(&kernel)
                .topology(1, 1, 2, 1)
                .memory_mb(256)
                .timeout(Duration::from_secs(5))
                .build()
        );
        let result = vm.run().unwrap();
        let Some(ref report) = result.monitor else {
            return;
        };
        assert!(
            report.summary.total_samples > 0,
            "monitor produced no samples within 5 s — sys_rdy never \
             unblocked the boot wait, or the boot wait never woke on \
             kill_evt either. Run wall time: {:?}",
            result.duration,
        );
        let first = report
            .samples
            .first()
            .expect("total_samples > 0 but samples list empty");
        assert!(
            first.elapsed_ms < 4_000,
            "first monitor sample landed at {} ms — that is past the \
             4 s budget and within the in-monitor 5 s sys_rdy \
             timeout window. The sys_rdy eventfd is not actually \
             unblocking the boot wait; the loop fell through on the \
             5 s ceiling. Total samples: {}, run duration: {:?}",
            first.elapsed_ms,
            report.summary.total_samples,
            result.duration,
        );
    }

    /// Pins the monitor's clean-exit path when the guest never
    /// reaches `send_sys_rdy`. With `init=/nonexistent` and
    /// `panic=-1`, the kernel panics on its `run_init_process`
    /// failure, the guest reboots immediately, and the host VM
    /// loop sees the reboot and shuts down. The monitor's
    /// pre-sample boot wait MUST observe the kill eventfd and
    /// fall through — not block until the 5 s sys_rdy ceiling.
    ///
    /// Wallclock budget: 8 s. The path to a kill_evt-driven
    /// monitor wakeup is "kernel panic → reboot exit → BSP loop
    /// sets kill → freeze coordinator writes kill_evt → monitor
    /// boot wait wakes". A regression that left the monitor
    /// blocked on sys_rdy alone (no kill_evt registration) would
    /// hold the VM open for the full 5 s ceiling — still under
    /// the 8 s budget, but a kill_evt regression that blocks
    /// indefinitely on a different fd would still surface here.
    ///
    /// `init=/nonexistent` rides on the kernel cmdline ahead of
    /// the builder's own `rdinit=/init` token; the kernel's
    /// `init/main.c::run_init_process` tries every `init=` path
    /// in order and panics when none succeeds, regardless of
    /// `rdinit` (which only fires for ramdisk-style discovery).
    /// `panic=-1` is the existing default in
    /// `KtstrVm::setup_memory`'s cmdline composition; setting it
    /// again via `cmdline_extra` is a no-op for the kernel parser
    /// (last token wins, and both tokens specify the same value).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn monitor_exits_cleanly_when_guest_panics_before_sys_rdy() {
        let kernel = crate::test_support::require_kernel();
        let _vmlinux = crate::test_support::require_vmlinux(&kernel);

        let vm = skip_on_contention!(
            KtstrVm::builder()
                .kernel(&kernel)
                .topology(1, 1, 2, 1)
                .memory_mb(256)
                .timeout(Duration::from_secs(10))
                .cmdline("init=/nonexistent panic=-1")
                .build()
        );
        let result = vm.run().unwrap();
        // The VM loop must shut down via the kernel's reboot exit
        // path, not via the builder's 10 s timeout.
        assert!(
            !result.timed_out,
            "guest never panicked / rebooted within 10 s — the test's \
             premise (panic-before-sys_rdy → kernel reboot → VM exit) \
             is not holding. Stderr tail: {:?}",
            result.stderr.lines().rev().take(5).collect::<Vec<_>>(),
        );
        // Wallclock budget: 8 s. With the monitor's 5 s
        // sys_rdy ceiling, a healthy run should finish well
        // under this. A regression that blocks the boot wait
        // indefinitely (e.g. kill_evt unregistered, sys_rdy
        // not promoted to the eventfd) would blow the budget.
        assert!(
            result.duration < Duration::from_secs(8),
            "VM ran for {:?} — past the 8 s budget. The monitor's \
             boot wait did not wake on kill_evt; the loop sat on the \
             sys_rdy ceiling instead. timed_out={}, exit_code={}",
            result.duration,
            result.timed_out,
            result.exit_code,
        );
    }

    /// Asserts the FIRST monitor sample (no reverse scan) has
    /// `rq_clock > 1ms` on at least one CPU. This pins the SYS_RDY
    /// → DATA_VALID pipeline's load-bearing semantics: when
    /// `send_sys_rdy` fires, the guest BSP has already completed
    /// `setup_per_cpu_areas` AND KASLR randomization AND
    /// `mount_filesystems()`, so the first per-iteration refresh in
    /// `monitor_loop` produces in-DRAM PAs and `read_rq_stats`
    /// returns live counters — no zero-pad sentinel period and no
    /// reverse scan needed to find a populated sample.
    ///
    /// Distinct from `boot_kernel_with_monitor`'s reverse-scan
    /// assertion: that test passes if ANY sample (even the last
    /// one, after seconds of pre-boot zeros) is populated. This
    /// test fails if the FIRST sample is empty — which would
    /// indicate the monitor started sampling before the guest had
    /// the rq fields written, defeating the whole point of the
    /// SYS_RDY gate.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn first_sample_has_valid_rq_clock_thanks_to_sys_rdy() {
        let kernel = crate::test_support::require_kernel();
        let _vmlinux = crate::test_support::require_vmlinux(&kernel);

        let vm = skip_on_contention!(
            KtstrVm::builder()
                .kernel(&kernel)
                .topology(1, 1, 2, 1)
                .memory_mb(256)
                .timeout(Duration::from_secs(5))
                .watchdog_timeout(Duration::from_secs(2))
                .build()
        );
        let result = vm.run().unwrap();
        let Some(ref report) = result.monitor else {
            return;
        };
        assert!(
            report.summary.total_samples > 0,
            "monitor produced no samples — cannot evaluate \
             FIRST-sample semantics"
        );
        let first = report
            .samples
            .first()
            .expect("total_samples > 0 but samples list empty");
        let any_populated = first.cpus.iter().any(|c| c.rq_clock > 1_000_000);
        assert!(
            any_populated,
            "FIRST monitor sample at elapsed_ms={} had every CPU at \
             rq_clock <= 1ms — SYS_RDY did not actually wait for the \
             guest's runqueue fields to be populated, or the \
             per-iteration refresh ran against pre-boot zeros. \
             cpus.rq_clock: {:?}, total_samples: {}, run duration: {:?}",
            first.elapsed_ms,
            first.cpus.iter().map(|c| c.rq_clock).collect::<Vec<_>>(),
            report.summary.total_samples,
            result.duration,
        );
    }

    /// Regression guard for the `scx_sched.watchdog_timeout` host-write
    /// mechanism. Boots a VM with scx-ktstr loaded plus a distinctive
    /// 2-second watchdog override, then asserts the monitor loop
    /// observed the expected jiffies value in guest memory.
    ///
    /// Skips gracefully when: no host kernel available, no vmlinux for
    /// BTF, `scx_root` symbol or `scx_sched.watchdog_timeout` BTF field
    /// missing, or the scheduler failed to attach. Real failure
    /// requires the override path to silently stop writing — which is
    /// exactly what we want to catch.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn watchdog_timeout_override_lands_in_guest_memory() {
        let kernel = crate::test_support::require_kernel();
        let vmlinux = crate::test_support::require_vmlinux(&kernel);

        // Version-dependent skips, in order of check cost. scx_root
        // is a 6.16+ symbol; its absence means either the kernel
        // predates the 6.16 scx_sched refactor (sched_ext still
        // present via the older scx_ops API, e.g. 6.14) or sched_ext
        // was not compiled in. Either way this test has nothing to
        // verify — skip. watchdog_offsets depends on BTF field layout
        // that only exists on 7.1+ kernels where
        // `scx_sched.watchdog_timeout` is a struct field.
        let syms = crate::test_support::require_kernel_symbols(&vmlinux);
        if syms.scx_root.is_none() {
            skip!("scx_root not present (needs Linux 6.16+ with sched_ext enabled)");
        }
        let offsets = crate::test_support::require_kernel_offsets(&vmlinux);
        if offsets.watchdog_offsets.is_none() {
            skip!(
                "scx_sched.watchdog_timeout field not in BTF \
                 (needs Linux 7.1+; pre-7.1 exposes watchdog timeout as a file-scope \
                 scx_watchdog_timeout symbol handled separately)"
            );
        }

        const TIMEOUT_SECS: u64 = 2;
        let hz = crate::monitor::guest_kernel_hz(Some(&kernel));
        let expected_jiffies = TIMEOUT_SECS * hz;

        let sched_bin = crate::test_support::require_binary("scx-ktstr");

        let vm = skip_on_contention!(
            KtstrVm::builder()
                .kernel(&kernel)
                .topology(1, 1, 1, 1)
                .memory_mb(256)
                .timeout(Duration::from_secs(5))
                .scheduler_binary(&sched_bin)
                .watchdog_timeout(Duration::from_secs(TIMEOUT_SECS))
                .build()
        );
        let result = vm.run().unwrap();
        let report = result.monitor.as_ref().expect(
            "ktstr: monitor report missing — require_kernel_offsets, scx_root, and \
             watchdog_offsets all resolved at setup, so monitor initialization must \
             have succeeded. A None report here is a bug in monitor startup",
        );
        let Some(obs) = &report.watchdog_observation else {
            // scx_root remained null for the whole run — the scheduler
            // never attached. Not a watchdog regression — skip.
            skip!(
                "watchdog observation missing — the scheduler did not attach \
                 (scx_root remained null throughout the run)"
            );
        };
        assert_eq!(
            obs.expected_jiffies, expected_jiffies,
            "expected_jiffies recorded by monitor ({}) does not match {} * HZ {} = {}",
            obs.expected_jiffies, TIMEOUT_SECS, hz, expected_jiffies,
        );
        assert_eq!(
            obs.observed_jiffies, obs.expected_jiffies,
            "host wrote {} jiffies to scx_sched.watchdog_timeout but guest memory holds {} — host-write mechanism broken",
            obs.expected_jiffies, obs.observed_jiffies,
        );
    }

    /// Prove the kernel uses the host-written watchdog timeout.
    ///
    /// Sets a 300-second watchdog and runs the scheduler for 15s.
    /// If the host write is effective, the kernel's watchdog timer
    /// uses 300s and no stall exit occurs. If the write were
    /// ineffective (kernel ignoring the value), the default timeout
    /// would apply and could spuriously fire on a slow guest.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn watchdog_override_prevents_stall_exit() {
        let kernel = crate::test_support::require_kernel();
        let _vmlinux = crate::test_support::require_vmlinux(&kernel);

        let sched_bin = crate::test_support::require_binary("scx-ktstr");

        let vm = skip_on_contention!(
            KtstrVm::builder()
                .kernel(&kernel)
                .topology(1, 1, 2, 1)
                .memory_mb(256)
                .timeout(Duration::from_secs(30))
                .scheduler_binary(&sched_bin)
                .watchdog_timeout(Duration::from_secs(300))
                .build()
        );
        let result = vm.run().unwrap();
        // Prior versions asserted `result.success` here. That's the
        // conjunction `!timed_out && exit_code == 0`, which depends
        // on init writing MSG_TYPE_EXIT to SHM before the AP-triggered
        // reboot propagates through the watchdog-kicks-BSP path. When
        // init is slightly slow (cold host cache, contended CPU),
        // exit_code lands at -1 (BSP run-loop default) and the
        // assertion fires even though the thing under test — scx
        // stall-exit behavior — is unaffected. Assert the actual
        // invariants instead: no guest crash, no scheduler
        // stall-exit markers in guest output. These are what would
        // change if the 300s watchdog override had failed.
        assert!(
            result.crash_message.is_none(),
            "no crash expected with 300s watchdog: {:?}",
            result.crash_message
        );
        // SchedulerDied / SchedulerNotAttached lifecycle frames are
        // written by start_scheduler in rust_init on attach failure
        // or scheduler exit (now via `send_lifecycle` on the bulk
        // data port — pre-bincode-migration these were COM2
        // sentinel strings). "sched_ext: disabled" is the kernel's
        // own disable message when scx tears down a scheduler (e.g.
        // on watchdog stall). Any of these appearing proves the
        // watchdog either fired or the scheduler exited for another
        // reason — either way the test's "no stall exit" invariant
        // is broken.
        let output = &result.output;
        let stderr = &result.stderr;
        let lifecycle_phase_seen = |phase: crate::vmm::wire::LifecyclePhase| -> bool {
            let Some(ref drain) = result.guest_messages else {
                return false;
            };
            drain.entries.iter().any(|e| {
                e.msg_type == crate::vmm::wire::MSG_TYPE_LIFECYCLE
                    && e.crc_ok
                    && !e.payload.is_empty()
                    && crate::vmm::wire::LifecyclePhase::from_wire(e.payload[0]) == Some(phase)
            })
        };
        assert!(
            !lifecycle_phase_seen(crate::vmm::wire::LifecyclePhase::SchedulerDied),
            "scheduler no longer running after 15s — either the watchdog fired or the \
             scheduler exited for another reason. output: {output:?}, stderr: {stderr:?}",
        );
        assert!(
            !lifecycle_phase_seen(crate::vmm::wire::LifecyclePhase::SchedulerNotAttached),
            "scheduler did not attach — no watchdog override to evaluate. \
             output: {output:?}, stderr: {stderr:?}",
        );
        assert!(
            !output.contains("sched_ext: disabled") && !stderr.contains("sched_ext: disabled"),
            "kernel disabled sched_ext during run — a watchdog stall or ops \
             error fired. output: {output:?}, stderr: {stderr:?}",
        );
        if let Some(ref report) = result.monitor
            && let Some(ref obs) = report.watchdog_observation
        {
            let hz = crate::monitor::guest_kernel_hz(Some(&kernel));
            let expected_jiffies = 300 * hz;
            assert_eq!(
                obs.expected_jiffies, expected_jiffies,
                "watchdog override should be 300s * HZ={hz}"
            );
            assert_eq!(
                obs.observed_jiffies, obs.expected_jiffies,
                "write/read roundtrip mismatch"
            );
        }
    }

    /// Validate that the core monitoring path reads meaningful
    /// runqueue data when a scheduler is loaded.
    ///
    /// Boots a VM with scx-ktstr, then asserts per-CPU snapshots
    /// contain plausible values. When schedstat data is present
    /// (CONFIG_SCHEDSTATS enabled), asserts sched_count is in a
    /// plausible range.
    #[test]
    fn monitor_reads_runqueue_data_with_scheduler() {
        let kernel = crate::test_support::require_kernel();
        let vmlinux = crate::test_support::require_vmlinux(&kernel);

        // Monitor-reads-runqueue asserts on cpu.rq_clock and cpu.schedstat,
        // which resolve through the non-optional rq offsets inside
        // KernelOffsets. Gating these at setup turns a silently-skipped
        // test (on BTF parse failure) into a loud infrastructure error.
        let _offsets = crate::test_support::require_kernel_offsets(&vmlinux);

        let sched_bin = crate::test_support::require_binary("scx-ktstr");

        let vm = skip_on_contention!(
            KtstrVm::builder()
                .kernel(&kernel)
                .topology(1, 1, 2, 1)
                .memory_mb(256)
                .timeout(Duration::from_secs(12))
                .watchdog_timeout(Duration::from_secs(5))
                .scheduler_binary(&sched_bin)
                .build()
        );
        let result = vm.run().unwrap();
        let report = result.monitor.as_ref().expect(
            "ktstr: monitor report missing — require_kernel_offsets resolved at \
             setup, so monitor initialization must have succeeded. A None report \
             here is a bug in monitor startup",
        );

        assert!(
            report.summary.total_samples >= 2,
            "need at least 2 monitor samples, got {}",
            report.summary.total_samples
        );

        // Scan samples in reverse chronological order looking for the
        // first sample where ANY CPU reports a rq_clock past the
        // early-boot noise floor (1 ms in ns). `all` flaked on CI
        // where one vCPU can remain near rq_clock=0 throughout
        // a short run. The 12s timeout covers boot + scheduler
        // attach (~1-2s) + several monitor samples; a single
        // populated CPU proves the monitor reads real rq data —
        // the code path is identical per-CPU.
        let populated = report
            .samples
            .iter()
            .rev()
            .find(|s| s.cpus.iter().any(|c| c.rq_clock > 1_000_000))
            .expect(
                "no monitor sample showed populated runqueue data — every sample \
                 had all CPUs at rq_clock <= 1ms, \
                 or the monitor is reading the wrong rq offsets",
            );
        for (i, cpu) in populated.cpus.iter().enumerate() {
            if cpu.rq_clock <= 1_000_000 {
                continue;
            }
            assert!(
                cpu.rq_clock < 300_000_000_000,
                "cpu {i}: rq_clock must be < 300s (ns), got {}",
                cpu.rq_clock
            );
        }

        for (i, cpu) in populated.cpus.iter().enumerate() {
            if let Some(ref ss) = cpu.schedstat {
                assert!(
                    ss.sched_count < 100_000_000,
                    "cpu {i}: sched_count {} exceeds plausible range — offset may be wrong",
                    ss.sched_count
                );
            }
        }
    }

    /// Validate that scx event counters populate on scx_sched kernels
    /// (Linux 6.16+). `event_offsets` resolves via either the 6.18+
    /// `scx_sched.pcpu → scx_sched_pcpu.event_stats` path or the
    /// 6.16–6.17 `scx_sched.event_stats_cpu` fallback; see
    /// `resolve_event_offsets` in `crate::monitor::btf_offsets` for
    /// the resolver that tries both.
    ///
    /// Gates on scx_root symbol presence and event_offsets BTF
    /// resolution. On pre-6.16 kernels (no scx_sched struct) or when
    /// neither BTF path resolves, event_offsets is None and this test
    /// skips.
    ///
    /// Event-counter physical-address resolution happens once at
    /// monitor start. If the scheduler has not attached by then
    /// (scx_root is still null), the monitor skips event counters
    /// for the entire run. The test skips in that case rather than
    /// asserting, matching the watchdog test's approach to
    /// scheduler-attach timing.
    #[test]
    fn event_counters_populated_with_scheduler() {
        let kernel = crate::test_support::require_kernel();
        let vmlinux = crate::test_support::require_vmlinux(&kernel);

        let syms = crate::test_support::require_kernel_symbols(&vmlinux);
        if syms.scx_root.is_none() {
            skip!("scx_root not present (needs Linux 6.16+ with sched_ext enabled)");
        }
        let offsets = crate::test_support::require_kernel_offsets(&vmlinux);
        if offsets.event_offsets.is_none() {
            skip!(
                "scx event-counter BTF fields not found \
                 (need either scx_sched.pcpu→scx_sched_pcpu.event_stats [Linux 6.18+] \
                 or scx_sched.event_stats_cpu [Linux 6.16–6.17])"
            );
        }

        let sched_bin = crate::test_support::require_binary("scx-ktstr");

        let vm = skip_on_contention!(
            KtstrVm::builder()
                .kernel(&kernel)
                .topology(1, 1, 2, 1)
                .memory_mb(256)
                .timeout(Duration::from_secs(12))
                .watchdog_timeout(Duration::from_secs(5))
                .scheduler_binary(&sched_bin)
                .build()
        );
        let result = vm.run().unwrap();
        let report = result.monitor.as_ref().expect(
            "ktstr: monitor report missing — require_kernel_offsets, scx_root, and \
             event_offsets all resolved at setup, so monitor initialization must \
             have succeeded. A None report here is a bug in monitor startup",
        );

        assert!(
            report.summary.total_samples > 0,
            "monitor should have collected at least one sample"
        );

        let last = report.samples.last().unwrap();
        let has_event_data = last.cpus.iter().any(|c| c.event_counters.is_some());
        if !has_event_data {
            skip!(
                "event counters remained None despite resolved offsets — \
                 the scheduler may not have attached before the monitor \
                 resolved event-counter physical addresses"
            );
        }

        let any_nonzero = last.cpus.iter().any(|c| {
            c.event_counters.as_ref().is_some_and(|ev| {
                ev.select_cpu_fallback != 0
                    || ev.dispatch_local_dsq_offline != 0
                    || ev.dispatch_keep_last != 0
                    || ev.enq_skip_exiting != 0
                    || ev.enq_skip_migration_disabled != 0
            })
        });
        assert!(
            any_nonzero,
            "event counters present but all zero — offset resolution may \
             have produced addresses that read uninitialized memory"
        );
        for (i, cpu) in last.cpus.iter().enumerate() {
            if let Some(ref ev) = cpu.event_counters {
                assert!(
                    ev.select_cpu_fallback >= 0 && ev.select_cpu_fallback < 1_000_000_000,
                    "cpu {i}: select_cpu_fallback {} outside plausible range",
                    ev.select_cpu_fallback
                );
                assert!(
                    ev.dispatch_local_dsq_offline >= 0
                        && ev.dispatch_local_dsq_offline < 1_000_000_000,
                    "cpu {i}: dispatch_local_dsq_offline {} outside plausible range",
                    ev.dispatch_local_dsq_offline
                );
                assert!(
                    ev.dispatch_keep_last >= 0 && ev.dispatch_keep_last < 1_000_000_000,
                    "cpu {i}: dispatch_keep_last {} outside plausible range",
                    ev.dispatch_keep_last
                );
                assert!(
                    ev.enq_skip_exiting >= 0 && ev.enq_skip_exiting < 1_000_000_000,
                    "cpu {i}: enq_skip_exiting {} outside plausible range",
                    ev.enq_skip_exiting
                );
                assert!(
                    ev.enq_skip_migration_disabled >= 0
                        && ev.enq_skip_migration_disabled < 1_000_000_000,
                    "cpu {i}: enq_skip_migration_disabled {} outside plausible range",
                    ev.enq_skip_migration_disabled
                );
            }
        }
    }

    /// Validate that sched_domain data is populated when BTF offsets
    /// resolve. Domains are kernel-built at boot and do not require a
    /// scheduler.
    ///
    /// Gates on sched_domain_offsets BTF availability. Uses a 2-CPU
    /// topology so the domain tree spans multiple CPUs.
    #[test]
    fn sched_domain_data_populated() {
        let kernel = crate::test_support::require_kernel();
        let vmlinux = crate::test_support::require_vmlinux(&kernel);

        let offsets = crate::test_support::require_kernel_offsets(&vmlinux);
        if offsets.sched_domain_offsets.is_none() {
            skip!(
                "sched_domain BTF fields not found (likely CONFIG_SMP=n; \
                 struct sched_domain is absent or incomplete in BTF on UP kernels, \
                 and on pre-6.17 kernels the rq.sd field is also compiled out)"
            );
        }

        // 5s timeout, 2s watchdog: monitor-only test, no scheduler.
        // Kernel threads (`sched_setup_smp` work) build the sched
        // domain tree during boot and finish before sys_rdy fires;
        // the first valid sample after boot already sees the
        // populated tree.
        let vm = skip_on_contention!(
            KtstrVm::builder()
                .kernel(&kernel)
                .topology(1, 1, 2, 1)
                .memory_mb(256)
                .timeout(Duration::from_secs(5))
                .watchdog_timeout(Duration::from_secs(2))
                .build()
        );
        let result = vm.run().unwrap();
        let report = result.monitor.as_ref().expect(
            "ktstr: monitor report missing — require_kernel_offsets and \
             sched_domain_offsets resolved at setup, so monitor initialization \
             must have succeeded. A None report here is a bug in monitor startup",
        );

        assert!(
            report.summary.total_samples > 0,
            "monitor should have collected at least one sample"
        );

        // Scan samples in reverse chronological order for the first
        // one where at least one CPU reports a non-empty sched_domains
        // list. `.last()` alone flaked on slow hosts where the final
        // sample was captured before the kernel finished building the
        // domain tree — sched_domains is populated via kernel threads
        // at boot, and the per-CPU `rq.sd` pointer lags the first rq
        // samples. Reverse-searching guards against that boot race:
        // if ANY sample in the run carries populated domains, the
        // kernel path works and the assertion passes.
        let populated = report
            .samples
            .iter()
            .rev()
            .find(|s| {
                s.cpus.iter().any(|c| {
                    c.sched_domains
                        .as_ref()
                        .is_some_and(|doms| !doms.is_empty())
                })
            })
            .unwrap_or_else(|| {
                panic!(
                    "no sample had any CPU with non-empty sched_domains across \
                     {} collected samples — monitor samples may be racing boot-time \
                     kernel thread that builds the domain tree, or `rq.sd` offsets \
                     are wrong",
                    report.samples.len(),
                );
            });

        for cpu in &populated.cpus {
            if let Some(ref doms) = cpu.sched_domains {
                if doms.is_empty() {
                    continue;
                }
                for w in doms.windows(2) {
                    assert!(
                        w[1].level > w[0].level,
                        "domain levels must be strictly increasing: {} -> {}",
                        w[0].level,
                        w[1].level
                    );
                }
                assert!(
                    doms[0].span_weight >= 2,
                    "lowest domain span_weight must be >= 2 for a 2-CPU topology, got {}",
                    doms[0].span_weight
                );
                for dom in doms {
                    assert!(
                        dom.span_weight > 0,
                        "domain level {} span_weight must be > 0",
                        dom.level
                    );
                }
            }
        }
    }
    #[test]
    fn builder_performance_mode_false_no_validation() {
        // performance_mode=false should not trigger validation, even with
        // a topology that exceeds host capacity.
        let exe = crate::resolve_current_exe().unwrap();
        let result = KtstrVmBuilder::default()
            .kernel(&exe)
            .topology(1, 1, 1, 1)
            .performance_mode(false)
            .build();
        match result {
            Ok(_) => {}
            Err(e)
                if e.downcast_ref::<host_topology::ResourceContention>()
                    .is_some() =>
            {
                skip!("resource contention: {e}");
            }
            Err(e) => panic!("performance_mode=false should not validate host topology: {e:#}",),
        }
    }

    #[test]
    fn builder_performance_mode_oversubscribed_fails() {
        let exe = crate::resolve_current_exe().unwrap();
        let host_topo = host_topology::HostTopology::from_sysfs().unwrap();
        let too_many = host_topo.total_cpus() as u32 + 1;
        let result = KtstrVmBuilder::default()
            .kernel(&exe)
            .topology(1, 1, too_many, 1)
            .performance_mode(true)
            .build();
        match result {
            Ok(_) => panic!("oversubscribed topology should fail"),
            Err(e) => {
                let msg = format!("{e}");
                assert!(
                    msg.contains("performance_mode"),
                    "error should mention performance_mode: {msg}",
                );
            }
        }
    }

    #[test]
    fn builder_performance_mode_too_many_llcs_fails() {
        let exe = crate::resolve_current_exe().unwrap();
        let host_topo = host_topology::HostTopology::from_sysfs().unwrap();
        let too_many_llcs = host_topo.llc_groups.len() as u32 + 1;
        // Need total vCPUs + 1 service CPU to fit without oversubscription.
        if (too_many_llcs as usize + 1) <= host_topo.total_cpus() {
            let result = KtstrVmBuilder::default()
                .kernel(&exe)
                .topology(1, too_many_llcs, 1, 1)
                .performance_mode(true)
                .build();
            assert!(
                result.is_err(),
                "more virtual LLCs than host LLCs should fail",
            );
        }
    }

    #[test]
    fn builder_performance_mode_valid_succeeds() {
        let exe = crate::resolve_current_exe().unwrap();
        let host_topo = host_topology::HostTopology::from_sysfs().unwrap();
        if host_topo.total_cpus() < 3 {
            skip!("need >= 3 host CPUs for performance_mode test");
        }
        let result = KtstrVmBuilder::default()
            .kernel(&exe)
            .topology(1, 1, 2, 1)
            .performance_mode(true)
            .build();
        match result {
            Ok(_) => {}
            Err(e)
                if e.downcast_ref::<host_topology::ResourceContention>()
                    .is_some() =>
            {
                skip!("resource contention: {e}");
            }
            Err(e) => panic!("valid topology with performance_mode should build: {e:#}",),
        }
    }

    #[test]
    fn builder_performance_mode_preserves_in_vm() {
        let exe = crate::resolve_current_exe().unwrap();
        let host_topo = host_topology::HostTopology::from_sysfs().unwrap();
        if host_topo.total_cpus() < 3 {
            skip!("need >= 3 host CPUs for performance_mode test");
        }
        let vm = skip_on_contention!(
            KtstrVmBuilder::default()
                .kernel(&exe)
                .topology(1, 1, 2, 1)
                .performance_mode(true)
                .build()
        );
        assert!(vm.performance_mode);
    }

    #[test]
    fn builder_performance_mode_false_preserves_in_vm() {
        let exe = crate::resolve_current_exe().unwrap();
        let vm = skip_on_contention!(
            KtstrVmBuilder::default()
                .kernel(&exe)
                .topology(1, 1, 1, 1)
                .performance_mode(false)
                .build()
        );
        assert!(!vm.performance_mode);
    }

    #[test]
    fn builder_performance_mode_mbind_nodes_populated() {
        let exe = crate::resolve_current_exe().unwrap();
        let host_topo = host_topology::HostTopology::from_sysfs().unwrap();
        if host_topo.total_cpus() < 3 {
            skip!("need >= 3 host CPUs for performance_mode test");
        }
        let vm = KtstrVmBuilder::default()
            .kernel(&exe)
            .topology(1, 1, 2, 1)
            .performance_mode(true)
            .build();
        if let Ok(vm) = vm {
            assert!(
                !vm.mbind_node_map.is_empty(),
                "mbind_node_map should be populated for performance_mode",
            );
        }
    }
    // -- aarch64 boot tests --

    /// Find an aarch64 kernel suitable for boot tests.
    /// Accepts both raw Image and gzip-compressed vmlinuz — load_kernel
    /// decompresses transparently.
    #[cfg(target_arch = "aarch64")]
    fn find_aarch64_image() -> Option<std::path::PathBuf> {
        crate::find_kernel().unwrap()
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn boot_kernel_produces_output_aarch64() {
        let Some(kernel) = find_aarch64_image() else {
            skip!("no aarch64 kernel image found");
        };

        let vm = skip_on_contention!(
            KtstrVm::builder()
                .kernel(&kernel)
                .topology(1, 1, 1, 1)
                .memory_mb(256)
                .timeout(Duration::from_secs(10))
                .cmdline("loglevel=7")
                .build()
        );
        let result = vm.run().unwrap();
        assert!(
            result.stderr.contains("Linux") || result.stderr.contains("Booting"),
            "kernel console should contain boot messages, got: {}",
            &result.stderr[..result.stderr.len().min(200)],
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn boot_kernel_smp_topology_aarch64() {
        let Some(kernel) = find_aarch64_image() else {
            skip!("no aarch64 kernel image found");
        };

        let vm = skip_on_contention!(
            KtstrVm::builder()
                .kernel(&kernel)
                .topology(1, 2, 2, 1) // 4 CPUs
                .memory_mb(256)
                .timeout(Duration::from_secs(10))
                .cmdline("loglevel=7")
                .build()
        );
        let result = vm.run().unwrap();
        assert!(!result.stderr.is_empty(), "no console output from SMP boot");
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn aarch64_kvm_has_immediate_exit() {
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let vm = kvm::KtstrKvm::new(topo, 64, false).unwrap();
        assert!(
            vm.has_immediate_exit,
            "KVM_CAP_IMMEDIATE_EXIT should be available on modern kernels"
        );
    }
}
