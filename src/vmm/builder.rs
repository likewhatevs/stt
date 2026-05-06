//! [`KtstrVmBuilder`] — public configuration surface for [`super::KtstrVm`].
//!
//! Test authors compose a VM by chaining the setters defined here, then
//! call [`KtstrVmBuilder::build`] to produce a runnable [`super::KtstrVm`].
//! The builder is the only path that constructs a VM — every field on
//! the runtime [`super::KtstrVm`] struct flows through one of the setters
//! plus the `build()` validator, which performs host-resource gating
//! (LLC reservation, hugepage probe, memory_mb sanity check) before
//! handing the VM back to the caller.
//!
//! Helpers `build_per_node_map` and `acquire_slot_with_locks` live next
//! to `build()` because they execute as part of the build pipeline:
//! both are called only from `build()` and `validate_performance_mode`,
//! and they cooperate with the [`super::host_topology`] flock primitives
//! to reserve the LLC slots the resulting VM will pin against.

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::time::Duration;

use super::host_topology;
use super::net_config;
use super::topology::{self, Topology};
use super::vcpu::BpfMapWriteParams;
use super::{KtstrVm, disk_config};

/// Builder for [`super::KtstrVm`].
///
/// Obtain via [`super::KtstrVm::builder()`], configure with the chained
/// setters below, then call [`build`](Self::build) to validate the
/// configuration and materialise a `KtstrVm`. Required inputs are a
/// `kernel` source directory or image, an `init_binary`, and either
/// a `run_args` payload (for test runs) or an `exec_cmd` / shell
/// configuration (for `ktstr shell`). Everything else is optional.
///
/// # Defaults
///
/// Field defaults applied by [`Default::default`]:
/// - `memory_mb` — 256 MB (overridden by [`memory_mb`](Self::memory_mb))
/// - `timeout` — 12 s (overridden by [`timeout`](Self::timeout))
/// - `watchdog_timeout` — 5 s (overridden by [`watchdog_timeout`](Self::watchdog_timeout))
/// - `topology` — 1 NUMA node × 1 LLC × 1 core × 1 thread (overridden
///   by [`topology`](Self::topology) or [`with_topology`](Self::with_topology))
/// - `performance_mode` — `false` (operator opts in via
///   [`performance_mode`](Self::performance_mode))
pub struct KtstrVmBuilder {
    kernel: Option<PathBuf>,
    init_binary: Option<PathBuf>,
    scheduler_binary: Option<PathBuf>,
    run_args: Vec<String>,
    sched_args: Vec<String>,
    pub(crate) topology: Topology,
    pub(crate) memory_mb: Option<u32>,
    memory_min_mb: u32,
    pub(crate) cmdline_extra: String,
    pub(crate) timeout: Duration,
    pub(crate) monitor_thresholds: Option<crate::monitor::MonitorThresholds>,
    pub(crate) watchdog_timeout: Option<Duration>,
    bpf_map_writes: Vec<BpfMapWriteParams>,
    pub(crate) performance_mode: bool,
    no_perf_mode: bool,
    sched_enable_cmds: Vec<String>,
    sched_disable_cmds: Vec<String>,
    include_files: Vec<(String, PathBuf)>,
    /// v0 holds at most one DiskConfig; rendered as `/dev/vda`.
    /// Vec retained for future multi-disk expansion. See
    /// [`super::KtstrVm::disks`].
    disks: Vec<disk_config::DiskConfig>,
    /// Optional network device. `None` skips virtio-net entirely
    /// (no FDT node, no MMIO range, no IRQ). `Some(_)` attaches one
    /// virtio-net device with the given config; the in-VMM loopback
    /// backend echoes TX bytes back to RX. v0 supports a single
    /// device. See [`super::KtstrVm::network`].
    network: Option<net_config::NetConfig>,
    busybox: bool,
    dmesg: bool,
    exec_cmd: Option<String>,
    /// Optional host path to the `ktstr-jemalloc-probe` binary.
    /// When `Some`, the probe is packed into the guest initramfs at
    /// `bin/ktstr-jemalloc-probe` and becomes spawnable by bare name
    /// inside the guest — used by the closed-loop probe tests in
    /// `tests/jemalloc_probe_tests.rs`.
    jemalloc_probe_binary: Option<PathBuf>,
    /// Optional host path to `ktstr-jemalloc-alloc-worker`. When
    /// `Some`, packed into the initramfs at `bin/ktstr-jemalloc-
    /// alloc-worker`. Used together with `jemalloc_probe_binary` for the
    /// cross-process closed-loop test.
    jemalloc_alloc_worker_binary: Option<PathBuf>,
    /// File path where the freeze coordinator writes the
    /// JSON-pretty failure-dump report. `None` disables the file
    /// sink — the dump still emits via `tracing::error`. See
    /// [`Self::failure_dump_path`].
    failure_dump_path: Option<PathBuf>,
    /// Capture two BPF-state snapshots per VM run instead of one.
    /// See the runtime field of the same name on [`super::KtstrVm`] for
    /// the full contract; the builder field flows through `build`
    /// unchanged.
    dual_snapshot: bool,
    /// When set, [`super::KtstrVm::init_virtio_blk`] opens this path
    /// directly as the virtio-blk backing file instead of allocating
    /// a fresh `tempfile()` (Raw branch) or invoking
    /// [`super::disk_template::ensure_template`] (Btrfs branch). The
    /// path-supplied backing exists exclusively for the
    /// disk-template-build VM driver in
    /// [`super::disk_template::build_template_via_vm`]: that driver
    /// materialises a sparse staging image, points the template VM
    /// at it via this field, and recovers the now-formatted file
    /// after VM exit for [`super::disk_template::store_atomic`] to
    /// publish. Setting this from any other code path bypasses the
    /// template cache and is ALMOST CERTAINLY a mistake —
    /// per-test runs want the `Raw` tempfile or `Btrfs` cache
    /// branches in `init_virtio_blk`. `None` is the production
    /// default.
    template_staging_image: Option<PathBuf>,
}

impl Default for KtstrVmBuilder {
    fn default() -> Self {
        KtstrVmBuilder {
            kernel: None,
            init_binary: None,
            scheduler_binary: None,
            run_args: Vec::new(),
            sched_args: Vec::new(),
            topology: Topology {
                llcs: 1,
                cores_per_llc: 1,
                threads_per_core: 1,
                numa_nodes: 1,
                nodes: None,
                distances: None,
            },
            memory_mb: Some(256),
            memory_min_mb: 0,
            cmdline_extra: String::new(),
            timeout: Duration::from_secs(12),
            monitor_thresholds: None,
            watchdog_timeout: Some(Duration::from_secs(5)),
            bpf_map_writes: Vec::new(),
            performance_mode: false,
            no_perf_mode: false,
            sched_enable_cmds: Vec::new(),
            sched_disable_cmds: Vec::new(),
            include_files: Vec::new(),
            disks: Vec::new(),
            network: None,
            busybox: false,
            dmesg: false,
            exec_cmd: None,
            jemalloc_probe_binary: None,
            jemalloc_alloc_worker_binary: None,
            failure_dump_path: None,
            dual_snapshot: false,
            template_staging_image: None,
        }
    }
}

impl KtstrVmBuilder {
    /// Path to the guest kernel: either a source directory (the VMM
    /// extracts `arch/*/boot/{bzImage,Image}`) or a prebuilt image.
    pub fn kernel(mut self, path: impl Into<PathBuf>) -> Self {
        self.kernel = Some(path.into());
        self
    }

    /// Path to the userspace init binary run as PID 1 inside the
    /// guest (typically the current test binary).
    pub fn init_binary(mut self, path: impl Into<PathBuf>) -> Self {
        self.init_binary = Some(path.into());
        self
    }

    /// Path to an optional scheduler binary loaded alongside the
    /// init binary; the init spawns it before dispatching the test.
    pub fn scheduler_binary(mut self, path: impl Into<PathBuf>) -> Self {
        self.scheduler_binary = Some(path.into());
        self
    }

    /// CLI argv passed to the init binary inside the guest (typically
    /// the per-test dispatch string like `--ktstr-test-fn NAME`).
    pub fn run_args(mut self, args: &[String]) -> Self {
        self.run_args = args.to_vec();
        self
    }

    /// Extra CLI arguments appended to the scheduler binary invocation.
    #[allow(dead_code)]
    pub fn sched_args(mut self, args: &[String]) -> Self {
        self.sched_args = args.to_vec();
        self
    }

    /// Resolve the kernel image from a source-tree root (sets
    /// `kernel` to `arch/<arch>/boot/<image>`).
    #[allow(dead_code)]
    pub fn kernel_dir(mut self, path: impl Into<PathBuf>) -> Self {
        let dir: PathBuf = path.into();
        #[cfg(target_arch = "x86_64")]
        {
            self.kernel = Some(dir.join("arch/x86/boot/bzImage"));
        }
        #[cfg(target_arch = "aarch64")]
        {
            self.kernel = Some(dir.join("arch/arm64/boot/Image"));
        }
        self
    }

    /// Set a uniform virtual CPU topology (big-to-little:
    /// `numa_nodes, llcs, cores_per_llc, threads_per_core`).
    ///
    /// Produces a topology with uniform LLC/memory distribution and
    /// default 10/20 NUMA distances. For per-node configuration
    /// (asymmetric memory, CXL nodes, custom distances), use
    /// [`with_topology`](Self::with_topology).
    pub fn topology(mut self, numa_nodes: u32, llcs: u32, cores: u32, threads: u32) -> Self {
        self.topology = Topology::new(numa_nodes, llcs, cores, threads);
        self
    }

    /// Set a pre-constructed topology with full per-node configuration.
    ///
    /// Accepts a [`Topology`] built via [`Topology::with_nodes`] and
    /// optionally [`Topology::with_distances`], preserving per-node
    /// memory sizes, CXL memory-only nodes, and custom distance matrices.
    pub fn with_topology(mut self, topo: Topology) -> Self {
        self.topology = topo;
        self
    }

    /// Pin guest memory to an explicit MB value and clear the
    /// deferred-sizing hint. Use `memory_deferred` when the payload
    /// size should drive the allocation.
    pub fn memory_mb(mut self, mb: u32) -> Self {
        self.memory_mb = Some(mb);
        self.memory_min_mb = 0;
        self
    }

    /// Defer memory allocation until after the initramfs is built.
    ///
    /// Memory will be computed from the actual initramfs size. Use this
    /// when no explicit `--memory` override is provided.
    pub fn memory_deferred(mut self) -> Self {
        self.memory_mb = None;
        self.memory_min_mb = 0;
        self
    }

    /// Defer memory allocation with a minimum floor. The deferred path
    /// computes memory from actual initramfs size, then takes the max
    /// of that and `min_mb`. Use when the topology needs more memory
    /// than the initramfs alone requires (e.g. NUMA tests with 4096 MB).
    pub fn memory_deferred_min(mut self, min_mb: u32) -> Self {
        self.memory_mb = None;
        self.memory_min_mb = min_mb;
        self
    }

    /// Append extra tokens to the guest kernel command line. Useful
    /// for one-off debug knobs (e.g. enabling extra subsystem
    /// verbosity) that shouldn't live in `ktstr.kconfig`.
    #[allow(dead_code)]
    pub fn cmdline(mut self, extra: &str) -> Self {
        self.cmdline_extra = extra.to_string();
        self
    }

    /// Alias for [`Self::cmdline`]. The field is named
    /// `cmdline_extra` internally; the alias matches the field name
    /// for callers that prefer the longer form.
    #[allow(dead_code)]
    pub fn cmdline_extra(self, extra: &str) -> Self {
        self.cmdline(extra)
    }

    /// Host-side watchdog timeout. The VM is killed if it has not
    /// exited on its own within this duration; the `VmResult`
    /// returned will have `timed_out = true`.
    pub fn timeout(mut self, t: Duration) -> Self {
        self.timeout = t;
        self
    }

    /// Override the `MonitorThresholds` used for stall detection and
    /// verdict rendering. Defaults to `MonitorThresholds::DEFAULT`.
    #[allow(dead_code)]
    pub fn monitor_thresholds(mut self, thresholds: crate::monitor::MonitorThresholds) -> Self {
        self.monitor_thresholds = Some(thresholds);
        self
    }

    /// File path where the freeze coordinator writes the JSON-pretty
    /// [`crate::monitor::dump::FailureDumpReport`] when an
    /// error-class SCX exit fires. `None` (the default) disables
    /// the file sink — the dump still emits via `tracing::error`
    /// regardless. The test framework's primary dispatch path in
    /// `test_support::eval` sets this per-test under the run's
    /// sidecar directory so structured failure data sits alongside
    /// `*.ktstr.json`; the auto-repro path in
    /// `test_support::probe::attempt_auto_repro` overrides it to a
    /// `.repro.failure-dump.json` sibling; CLI / library callers
    /// that want the dump on disk set it explicitly here.
    ///
    /// Pure setter — no filesystem side effects. Stale-file
    /// pre-clear is the dispatch layer's responsibility (primary:
    /// `test_support::eval`, which clears BOTH the primary path
    /// AND the repro path on every dispatch so a passing rerun
    /// is not masked by either of the prior failure's leftovers;
    /// auto-repro: `test_support::probe::attempt_auto_repro`
    /// implicitly relies on the primary dispatch's pre-clear of
    /// the repro path before falling into the repro VM build).
    pub fn failure_dump_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.failure_dump_path = Some(path.into());
        self
    }

    /// Enable the dual-snapshot freeze-coordinator path. With
    /// `enabled = true` the coordinator runs an additional per-CPU
    /// `runnable_at` scanner alongside the existing
    /// `ktstr_err_exit_detected` poll: when any task crosses the
    /// `watchdog_timeout/2` half-way mark it triggers an extra
    /// freeze + dump cycle. Both snapshots are emitted as a single
    /// [`crate::monitor::dump::DualFailureDumpReport`] file at
    /// [`Self::failure_dump_path`] (the late snapshot at the same
    /// trigger as the single-snapshot path; the early snapshot is
    /// optional). Used by the auto-repro path to capture BPF state
    /// deltas across a stall window.
    ///
    /// Default off — two reasons:
    /// 1. **Scanner cost.** The early-trigger path walks the
    ///    kernel's global `scx_tasks` list AND every per-CPU
    ///    `rq->scx.runnable_list` once per scan tick (250 ms),
    ///    reading each task's `task_struct.scx.runnable_at` via
    ///    direct-mapped guest memory. On a 64-vCPU host with
    ///    hundreds of runnable tasks the steady-state cost is
    ///    non-negligible — a primary VM doesn't pay it unless
    ///    the run already failed and an auto-repro is being
    ///    attempted.
    /// 2. **Consumer compatibility.** The on-disk shape changes
    ///    from [`crate::monitor::dump::FailureDumpReport`] to
    ///    [`crate::monitor::dump::DualFailureDumpReport`], a
    ///    different JSON schema. Any consumer reading the dump
    ///    file must handle both schemas (gated on the `schema`
    ///    field). Keeping the primary path on the single-snapshot
    ///    shape means existing consumers (e.g.
    ///    `tests/failure_dump_e2e.rs`) keep working without
    ///    awareness of the dual-snapshot wrapper.
    pub fn dual_snapshot(mut self, enabled: bool) -> Self {
        self.dual_snapshot = enabled;
        self
    }

    /// Override the guest scx watchdog timeout. Applied via
    /// `scx_sched.watchdog_timeout` (7.1+) or the static
    /// `scx_watchdog_timeout` symbol (pre-7.1); silently no-ops on
    /// kernels where neither path is available.
    #[allow(dead_code)]
    pub fn watchdog_timeout(mut self, timeout: Duration) -> Self {
        self.watchdog_timeout = Some(timeout);
        self
    }

    /// Schedule a host-side write into a named BPF map after the
    /// scheduler is loaded. `map_name_suffix` is matched against
    /// `bpf_map.name` (kernel truncates to 15 chars); `offset` is
    /// the byte offset within the array-map value region; `value`
    /// is a `u32` written in native byte order.
    ///
    /// Repeated calls queue additional writes; all queued writes run
    /// sequentially on the same `BpfMapAccessor` after the scheduler
    /// attaches, with a single guest-side unblock once every write
    /// completes. Order of calls is preserved.
    #[allow(dead_code)]
    pub fn bpf_map_write(mut self, map_name_suffix: &str, offset: usize, value: u32) -> Self {
        self.bpf_map_writes.push(BpfMapWriteParams {
            map_name_suffix: map_name_suffix.to_string(),
            offset,
            value,
        });
        self
    }

    /// Enable performance mode: vCPU pinning to host LLCs,
    /// hugepage-backed guest memory, NUMA mbind, and RT scheduling
    /// on both architectures. On x86_64, additionally:
    /// KVM_HINTS_REALTIME CPUID hint (disables PV spinlocks, PV TLB
    /// flush, PV sched_yield; enables haltpoll cpuidle), PAUSE + HLT
    /// VM exit disabling via KVM_CAP_X86_DISABLE_EXITS (HLT falls
    /// back to PAUSE-only when mitigate_smt_rsb is active), and
    /// KVM_CAP_HALT_POLL skipped (guest haltpoll cpuidle disables
    /// host halt polling via MSR_KVM_POLL_CONTROL). On aarch64, KVM
    /// exit suppression and CPUID hints are not available. Validated
    /// at build time -- oversubscription returns `ResourceContention`,
    /// insufficient hugepages is a warning.
    #[allow(dead_code)]
    pub fn performance_mode(mut self, enabled: bool) -> Self {
        self.performance_mode = enabled;
        self
    }

    /// Skip flock topology reservation and force `performance_mode=false`
    /// (disables pinning, RT scheduling, hugepages, NUMA mbind, KVM exit
    /// suppression). For shared runners or unprivileged containers.
    pub fn no_perf_mode(mut self, enabled: bool) -> Self {
        self.no_perf_mode = enabled;
        self
    }

    /// Shell commands run inside the guest before the scenario to
    /// switch on a kernel-builtin scheduler (mirrors
    /// `SchedulerSpec::KernelBuiltin::enable`).
    pub fn sched_enable_cmds(mut self, cmds: &[&str]) -> Self {
        self.sched_enable_cmds = cmds.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Shell commands run inside the guest after the scenario to
    /// revert a kernel-builtin scheduler change (mirrors
    /// `SchedulerSpec::KernelBuiltin::disable`).
    pub fn sched_disable_cmds(mut self, cmds: &[&str]) -> Self {
        self.sched_disable_cmds = cmds.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Add files to include in the guest initramfs.
    /// Each entry is `(archive_path, host_path)`.
    pub fn include_files(mut self, files: Vec<(String, PathBuf)>) -> Self {
        self.include_files = files;
        self
    }

    /// Attach a disk to the VM. Each call replaces any previously
    /// attached disk; the framework reserves a single MMIO + IRQ
    /// pair, so today the VM exposes at most one virtio-blk device
    /// at `/dev/vda`.
    ///
    /// Per-test backing is allocated by
    /// [`super::KtstrVm::init_virtio_blk`]:
    /// - `Filesystem::Raw` (the default): a fresh sparse
    ///   `tempfile()` per test, the kernel reclaims storage on
    ///   device drop.
    /// - `Filesystem::Btrfs`: a host-cached, guest-formatted
    ///   template image produced by a one-shot template VM
    ///   ([`super::disk_template::build_template_via_vm`]) is
    ///   reflink-cloned via `FICLONE` for the per-test backing.
    ///   The host never execs mkfs against a real backing file;
    ///   the kernel inside the template VM is the on-disk-format
    ///   authority.
    ///
    /// # Visible cache + per-test fan-out
    ///
    /// For `Filesystem::Btrfs`, the cache is a real on-disk
    /// directory under the ktstr cache root (resolved via
    /// `KTSTR_CACHE_DIR` / `XDG_CACHE_HOME` / `$HOME/.cache`; see
    /// [`super::disk_template::cache_root`]) so operators can
    /// inspect what's been built, GC stale entries by hand, and warm
    /// the cache out-of-band by running a Btrfs test once. The cache
    /// is keyed by `(filesystem_tag, capacity_mib)` and the
    /// directory layout is `<cache>/disk_templates/<key>/template.img`
    /// — see [`super::disk_template`] module docs for the full encoding.
    ///
    /// Per-test fan-out goes through
    /// [`super::disk_template::clone_to_per_test`], which uses the
    /// `FICLONE` ioctl to reflink-copy the cached template image
    /// into a tempfile for the test VM. `FICLONE` is `O(metadata)`
    /// and copy-on-write at the extent level: per-test fan-out is
    /// independent of disk capacity and per-test writes never
    /// modify the cached template. The cache directory MUST live
    /// on a btrfs or xfs filesystem;
    /// [`super::disk_template::verify_cache_dir_supports_reflink`]
    /// checks `statfs.f_type` up front and bails with an actionable
    /// diagnostic when the cache filesystem cannot reflink, so
    /// operators see the constraint at first use rather than
    /// debugging a cryptic ioctl errno.
    pub fn disk(mut self, disk: disk_config::DiskConfig) -> Self {
        self.disks = vec![disk];
        self
    }

    /// Attach one virtio-net device with the given configuration. The
    /// v0 backend is in-VMM loopback: TX bytes are echoed back into
    /// the RX queue inside the VMM, generating real virtio TX kicks
    /// and real `vring_interrupt` → `NET_RX_SOFTIRQ` activity that
    /// scheduler-test scenarios can observe. There is no host
    /// networking — IP-layer self-traffic is intercepted by the
    /// guest kernel's `RTN_LOCAL` route onto `lo`, so AF_PACKET raw
    /// sockets bound by `ifindex` are the path that exercises the
    /// virtio device.
    ///
    /// v0 supports a single device; calling this method twice
    /// overwrites the prior `NetConfig`.
    ///
    /// `dead_code` allow: kept as the public builder entry point
    /// for attaching a virtio-net device; the production VM-bring-up
    /// path in [`super::setup`] currently never enables networking
    /// for a test, but the device, builder field, and config type
    /// are all wired so a scenario can opt in.
    #[allow(dead_code)]
    pub fn network(mut self, config: net_config::NetConfig) -> Self {
        self.network = Some(config);
        self
    }

    /// Override [`super::KtstrVm::init_virtio_blk`]'s per-test
    /// backing-file allocation with `path`. Internal-only: this is
    /// the seam the disk-template-build VM driver
    /// ([`super::disk_template::build_template_via_vm`]) uses to
    /// point a template-build guest at a host-staged sparse image,
    /// run `mkfs.<fstype>` against it inside the guest, and recover
    /// the now-formatted bytes after VM exit.
    ///
    /// When set, `init_virtio_blk` opens `path` for read+write and
    /// hands the resulting [`std::fs::File`] to the device — neither
    /// the `Raw` tempfile branch nor the `Btrfs` ensure_template
    /// branch executes, so a template-build VM cannot recursively
    /// re-enter the disk-template cache it is itself populating.
    /// The first attached disk's
    /// [`super::disk_config::DiskConfig::capacity_bytes`] still
    /// drives the device's advertised capacity; the staging image
    /// must already be sized to match.
    ///
    /// Production test paths leave this `None`. Setting it from a
    /// per-test build silently disables the template cache and would
    /// surface as a wrong-content backing file — the `Raw`/`Btrfs`
    /// branches in `init_virtio_blk` exist exactly to satisfy
    /// per-test isolation.
    pub(crate) fn template_staging_image(mut self, path: PathBuf) -> Self {
        self.template_staging_image = Some(path);
        self
    }

    /// Host path to `ktstr-jemalloc-probe`. When set, the probe is
    /// packed into the guest initramfs as an extra binary under
    /// `bin/` and resolves by bare name on the guest `PATH`. Tests
    /// that target the jemalloc TLS probe from a guest-side
    /// `ctx.payload(&PROBE)` invocation must set this to the host
    /// path obtained via `env!("CARGO_BIN_EXE_ktstr-jemalloc-probe")`.
    ///
    /// The probe attaches to a separately-spawned
    /// `ktstr-jemalloc-alloc-worker` via `--pid <worker_pid>`; the
    /// worker ships with DWARF, which is what the probe resolves
    /// offsets against, so the init binary does NOT need to retain
    /// DWARF. An earlier
    /// design attempted to preserve DWARF on the init binary so the
    /// probe could resolve offsets against the running init; that
    /// inflated the initramfs past practical VM memory budgets (the
    /// unstripped test binary is ~1 GB) and was abandoned in favor
    /// of routing DWARF through the probe and worker binaries.
    pub fn jemalloc_probe_binary(mut self, path: impl Into<PathBuf>) -> Self {
        self.jemalloc_probe_binary = Some(path.into());
        self
    }

    /// Host path to `ktstr-jemalloc-alloc-worker`. When set, the
    /// worker is packed alongside the probe in the guest initramfs
    /// as `/bin/ktstr-jemalloc-alloc-worker`. Used by the
    /// cross-process closed-loop test — spawned as a background
    /// payload that allocates a known number of bytes on the
    /// huge-size path (the jemalloc code path that unconditionally
    /// updates `thread_allocated` regardless of tcache state), then
    /// probed externally. The worker is much smaller than the full
    /// ktstr test binary (a single `fn main` linked against
    /// tikv-jemallocator) so shipping it keeps the initramfs well
    /// inside VM memory budgets — the init-DWARF approach that
    /// inflated the archive past those budgets was abandoned in
    /// favor of per-binary DWARF on the probe and worker.
    pub fn jemalloc_alloc_worker_binary(mut self, path: impl Into<PathBuf>) -> Self {
        self.jemalloc_alloc_worker_binary = Some(path.into());
        self
    }

    /// Embed busybox in the initramfs for shell mode.
    #[allow(dead_code)]
    pub fn busybox(mut self, enabled: bool) -> Self {
        self.busybox = enabled;
        self
    }

    /// Stream the guest kernel console (COM1/dmesg) to stderr in
    /// real time. Also bumps `loglevel=7` for verbose kernel output.
    pub fn dmesg(mut self, enabled: bool) -> Self {
        self.dmesg = enabled;
        self
    }

    /// Run a single command inside the guest instead of an
    /// interactive shell; the VM exits when the command completes.
    /// Requires `busybox(true)` and is typically paired with
    /// `KtstrVm::new_shell`.
    #[allow(dead_code)]
    pub fn exec_cmd(mut self, cmd: String) -> Self {
        self.exec_cmd = Some(cmd);
        self
    }

    /// Validate the builder configuration and materialise a [`super::KtstrVm`].
    ///
    /// Returns `Err` for missing required inputs (kernel, init binary),
    /// invalid topology, or host resources insufficient to satisfy
    /// `performance_mode` requirements (the last surfaces as
    /// `ResourceContention`, which callers typically treat as a
    /// skip rather than a failure).
    pub fn build(mut self) -> Result<KtstrVm> {
        let no_perf_mode = self.no_perf_mode;
        if no_perf_mode {
            self.performance_mode = false;
        }

        // `host_topo` is cached on KtstrVm so `KtstrVm::run`'s
        // default-else branch (neither perf-mode nor no-perf-mode)
        // can call `acquire_cpu_locks` without re-reading sysfs.
        // The no-perf-mode and perf-mode branches reuse their
        // stored plans' `locked_llcs` / `llc_indices` directly
        // through `acquire_resource_locks` and do not need the
        // topology at run time.
        let mut cached_host_topo: Option<host_topology::HostTopology> = None;

        let (pinning_plan, mbind_node_map, no_perf_plan) = if no_perf_mode {
            // No-perf-mode VMs would otherwise have unrestricted vCPU
            // affinity — the host kernel places their threads on any
            // online CPU, including ones a perf-mode peer has flocked
            // and bound its RT-FIFO vCPUs to. Injecting that thread
            // competition destroys perf-mode's measurement contract.
            // The coordination mechanism is an LLC-level flock set
            // (same as `kernel_build_pipeline`) so perf-mode's required
            // `LOCK_EX` blocks on any of them and fails over cleanly.
            //
            // `--cpu-cap` (or `KTSTR_CPU_CAP`) is a CPU-count budget:
            // the planner walks whole LLCs in contention- / NUMA-aware
            // order, filtered to the calling process's allowed cpuset
            // (sched_getaffinity), and accumulates until N CPUs are
            // reserved. `acquire_llc_plan` returns the selected LLC
            // list + flat `cpus` (intersection with allowed) + RAII
            // flock fds. The `cpus` are threaded into `no_perf_plan`
            // so `run_vm` can `sched_setaffinity` every vCPU thread
            // onto that pool. `KtstrVm::run` re-acquires fresh
            // flocks just before vCPU spawn — `build()` does not
            // hold flocks across the post-build setup window so
            // concurrent peers see the LLCs free until the run
            // actually starts.
            //
            // When the cap is absent (`CpuCap::resolve(None) ==
            // Ok(None)`), the planner applies the 30%-of-allowed
            // default (`default_cpu_budget`). The resulting plan
            // reserves a subset of host LLCs, not "every LLC" as the
            // 15ee285 path did — so no-perf-mode VMs never fight
            // concurrent builds or other no-perf peers for the full
            // host, regardless of whether the user set the flag.
            //
            // `from_sysfs` returning `Err` (non-Linux, sysfs absent)
            // still forces the no-cap branch; `acquire_llc_plan` is
            // skipped, no coordination is possible, but the VM still
            // runs. `KTSTR_BYPASS_LLC_LOCKS=1` bypasses both paths.
            //
            // The CLI binaries reject `--cpu-cap` + bypass at parse
            // time (see `ktstr::cli::CPU_CAP_HELP` and the Shell/
            // kernel-build dispatch checks in bin/ktstr.rs and
            // bin/cargo-ktstr.rs), but library consumers building
            // a `KtstrVmBuilder` directly with both env vars set
            // would silently lose the cap under a bare `if bypass
            // { return None-plan }`. Mirror the CLI check here so
            // the enforcement contract holds for every entry point,
            // not just the ones that go through the binaries.
            let bypass = std::env::var("KTSTR_BYPASS_LLC_LOCKS")
                .ok()
                .is_some_and(|v| !v.is_empty());
            let cpu_cap = host_topology::CpuCap::resolve(None)?;
            if bypass {
                if cpu_cap.is_some() {
                    anyhow::bail!(
                        "no-perf-mode: KTSTR_CPU_CAP conflicts with \
                         KTSTR_BYPASS_LLC_LOCKS=1; unset one of them. \
                         KTSTR_CPU_CAP is a resource contract; bypass \
                         disables the contract entirely."
                    );
                }
                (None, Vec::new(), None)
            } else if let Ok(host_topo) = host_topology::HostTopology::from_sysfs() {
                let test_topo = crate::topology::TestTopology::from_system()?;
                // Compute the plan and immediately drop the flocks:
                // we want the plan SHAPE on KtstrVm but not the
                // RAII fds. `run()` re-takes fresh `LOCK_SH` on
                // `plan.locked_llcs` via `acquire_resource_locks`
                // just before vCPU spawn so the build-to-run
                // setup window holds no flocks.
                let mut plan = host_topology::acquire_llc_plan(&host_topo, &test_topo, cpu_cap)?;
                host_topology::warn_if_cross_node_spill(&plan, &host_topo);
                // Strip the flock fds — they release on drop. The
                // plan's `cpus` / `locked_llcs` / `mems` fields
                // stay populated for build-time setup paths
                // (no_perf_cpus on virtio-blk worker, mask
                // computation in run_vm/freeze_coord).
                drop(std::mem::take(&mut plan.locks));
                cached_host_topo = Some(host_topo);
                (None, Vec::new(), Some(plan))
            } else {
                if cpu_cap.is_some() {
                    anyhow::bail!(
                        "--cpu-cap set but host LLC topology unreadable from \
                         sysfs — cannot enforce the resource budget. Run on a \
                         host with /sys/devices/system/cpu populated, or drop \
                         --cpu-cap to run without enforcement."
                    );
                }
                tracing::warn!(
                    "no-perf-mode: could not read host LLC topology from sysfs; \
                     skipping CPU-budget LLC reservation. Concurrent perf-mode \
                     runs on this host will NOT be serialized against this VM"
                );
                (None, Vec::new(), None)
            }
        } else if self.performance_mode {
            let (mut plan, host_topo) = self.validate_performance_mode()?;
            let node_map = build_per_node_map(&plan, &host_topo, &self.topology);
            // Strip the flock fds — `run()` re-acquires via
            // `acquire_resource_locks` using `plan.llc_indices`.
            // The build-time setup paths read `assignments` /
            // `service_cpu` / `llc_indices`, which all stay
            // populated.
            drop(std::mem::take(&mut plan.locks));
            cached_host_topo = Some(host_topo);
            (Some(plan), node_map, None)
        } else {
            // Default else: no perf-mode and no no-perf-mode. The
            // legacy path acquired a per-CPU flock window via
            // `acquire_cpu_locks` for the VM's lifetime; the
            // deferred-lock contract pushes that into `KtstrVm::run`
            // so the build-to-run setup window holds no flocks.
            // Cache `host_topo` so `run()` can pass it to
            // `acquire_cpu_locks` without re-reading sysfs.
            cached_host_topo = host_topology::HostTopology::from_sysfs().ok();
            (None, Vec::new(), None)
        };

        let kernel = self.kernel.context("kernel path required")?;
        anyhow::ensure!(kernel.exists(), "kernel not found: {}", kernel.display());
        let t = &self.topology;
        anyhow::ensure!(t.llcs > 0, "llcs must be > 0");
        anyhow::ensure!(t.cores_per_llc > 0, "cores_per_llc must be > 0");
        anyhow::ensure!(t.threads_per_core > 0, "threads_per_core must be > 0");
        anyhow::ensure!(t.numa_nodes > 0, "numa_nodes must be > 0");
        // `memory_mb == Some(0)` would forward a literal `-m 0` to the
        // VMM backend (KVM rejects it at ioctl time with an opaque
        // error). Catch it here with a clear message so the caller
        // learns they set 0 explicitly rather than seeing a generic
        // kvm failure later. `None` falls back to the default (256 MB).
        if matches!(self.memory_mb, Some(0)) {
            anyhow::bail!(
                "memory_mb must be > 0 (a VM with zero memory cannot boot); \
                 omit `.memory_mb(...)` to use the builder default"
            );
        }
        if let Some(ref bin) = self.init_binary
            && !bin.starts_with("/proc/")
        {
            anyhow::ensure!(bin.exists(), "init binary not found: {}", bin.display());
        }
        if let Some(ref bin) = self.scheduler_binary {
            anyhow::ensure!(
                bin.exists(),
                "scheduler binary not found: {}",
                bin.display()
            );
        }

        Ok(KtstrVm {
            kernel,
            init_binary: self.init_binary,
            scheduler_binary: self.scheduler_binary,
            run_args: self.run_args,
            sched_args: self.sched_args,
            topology: self.topology,
            memory_mb: self.memory_mb,
            memory_min_mb: self.memory_min_mb,
            cmdline_extra: self.cmdline_extra,
            timeout: self.timeout,
            monitor_thresholds: self.monitor_thresholds,
            watchdog_timeout: self.watchdog_timeout,
            bpf_map_writes: self.bpf_map_writes,
            performance_mode: self.performance_mode,
            no_perf_mode,
            pinning_plan,
            mbind_node_map,
            no_perf_plan,
            host_topo: cached_host_topo,
            sched_enable_cmds: self.sched_enable_cmds,
            sched_disable_cmds: self.sched_disable_cmds,
            include_files: self.include_files,
            disks: self.disks,
            network: self.network,
            busybox: self.busybox,
            dmesg: self.dmesg,
            exec_cmd: self.exec_cmd,
            jemalloc_probe_binary: self.jemalloc_probe_binary,
            jemalloc_alloc_worker_binary: self.jemalloc_alloc_worker_binary,
            failure_dump_path: self.failure_dump_path,
            dual_snapshot: self.dual_snapshot,
            template_staging_image: self.template_staging_image,
        })
    }

    /// Validate host resources for performance_mode and compute the
    /// pinning plan. Returns both the plan and the host topology (needed
    /// for NUMA node discovery). Returns `ResourceContention` when the
    /// host lacks CPUs or LLC slots. Warnings are printed for degraded
    /// conditions (hugepages, host load).
    fn validate_performance_mode(
        &mut self,
    ) -> Result<(host_topology::PinningPlan, host_topology::HostTopology)> {
        let host_topo = host_topology::HostTopology::from_sysfs()
            .context("performance_mode: read host topology")?;

        let t = &self.topology;
        let total_vcpus = t.total_cpus();

        // Validate LLC exclusivity: each virtual LLC should map to
        // its own physical LLC group. Sum actual per-group CPU counts
        // to handle asymmetric LLCs.
        let llcs_needed = t.llcs as usize;
        let reserved: usize = host_topo
            .llc_groups
            .iter()
            .take(llcs_needed)
            .map(|g| g.cpus.len())
            .sum();
        let total_reserved = reserved + 1; // +1 for service CPU
        if total_reserved > host_topo.total_cpus() {
            return Err(anyhow::Error::new(host_topology::ResourceContention {
                reason: format!(
                    "performance_mode: need {} CPUs ({} across {} LLCs + 1 service) \
                     but only {} host CPUs available\n  \
                     hint: pass --no-perf-mode or set KTSTR_NO_PERF_MODE=1 to run without CPU reservation",
                    total_reserved,
                    reserved,
                    llcs_needed,
                    host_topo.total_cpus(),
                ),
            }));
        }

        let plan = acquire_slot_with_locks(&host_topo, t)?;

        // WARN: hugepages (only when memory is known upfront).
        if let Some(mb) = self.memory_mb {
            let free = host_topology::hugepages_free();
            let needed = host_topology::hugepages_needed(mb);
            if free == 0 {
                eprintln!(
                    "performance_mode: WARNING: no 2MB hugepages available, \
                     guest memory will use regular pages",
                );
            } else if free < needed {
                eprintln!(
                    "performance_mode: WARNING: need {} 2MB hugepages, \
                     only {} free — falling back to regular pages",
                    needed, free,
                );
            }
        }

        // WARN: host load.
        if let Some((running, total)) = host_topology::host_load_estimate() {
            let threshold = (total_vcpus as f64 * 0.5) as usize;
            if running > threshold {
                eprintln!(
                    "performance_mode: WARNING: {} processes running on {} CPUs \
                     (threshold {} for {} vCPUs) — results may be noisy",
                    running, total, threshold, total_vcpus,
                );
            }
        }

        Ok((plan, host_topo))
    }
}

/// Build per-guest-NUMA-node host NUMA node mapping from a pinning plan.
fn build_per_node_map(
    plan: &host_topology::PinningPlan,
    host_topo: &host_topology::HostTopology,
    topo: &crate::vmm::topology::Topology,
) -> Vec<Vec<usize>> {
    let n = topo.numa_nodes as usize;
    let mut map: Vec<std::collections::BTreeSet<usize>> =
        vec![std::collections::BTreeSet::new(); n];
    let cpus_per_llc = topo.cores_per_llc * topo.threads_per_core;
    for &(vcpu_id, host_cpu) in &plan.assignments {
        let llc_id = vcpu_id / cpus_per_llc;
        let guest_node = topo.numa_node_of(llc_id) as usize;
        let host_node = host_topo.cpu_to_node.get(&host_cpu).copied().unwrap_or(0);
        if guest_node < n {
            map[guest_node].insert(host_node);
        }
    }
    map.into_iter().map(|s| s.into_iter().collect()).collect()
}

/// Try each LLC slot, compute a pinning plan, and acquire resource
/// locks (non-blocking). Single pass through all available slots.
/// Returns `ResourceContention` when all slots are busy; callers
/// rely on nextest retry backoff for contention resolution.
fn acquire_slot_with_locks(
    host_topo: &host_topology::HostTopology,
    topo: &topology::Topology,
) -> Result<host_topology::PinningPlan> {
    let num_llcs = host_topo.llc_groups.len();
    let llcs_needed = topo.llcs as usize;
    let max_slots = num_llcs.checked_div(llcs_needed).unwrap_or(num_llcs).max(1);
    let llc_mode = host_topology::LlcLockMode::Exclusive;

    for slot in 0..max_slots {
        let offset = slot * llcs_needed;

        let candidate = host_topo
            .compute_pinning(topo, true, offset)
            .context("performance_mode: topology mapping")?;

        match host_topology::acquire_resource_locks(&candidate, &candidate.llc_indices, llc_mode)? {
            host_topology::LockOutcome::Acquired { locks, .. } => {
                let mut plan = candidate;
                plan.locks = locks;
                eprintln!(
                    "performance_mode: reserved LLC slot {} (offset {}, max {})",
                    slot, offset, max_slots,
                );
                return Ok(plan);
            }
            host_topology::LockOutcome::Unavailable(_) => continue,
        }
    }

    Err(anyhow::Error::new(host_topology::ResourceContention {
        reason: format!(
            "all {max_slots} LLC slots busy\n  \
             hint: pass --no-perf-mode or set KTSTR_NO_PERF_MODE=1 to run without CPU reservation"
        ),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_default() {
        let b = KtstrVmBuilder::default();
        assert_eq!(b.memory_mb, Some(256));
        assert_eq!(b.topology.total_cpus(), 1);
    }

    /// Explicit `memory_mb(0)` must be rejected at build time rather
    /// than surfacing as an opaque KVM ioctl failure later. The
    /// builder default (None→256) passes.
    #[test]
    fn builder_rejects_explicit_zero_memory() {
        // Point at a real file so the kernel-existence check
        // (which runs before the memory_mb guard) does not short-
        // circuit. /bin/true exists on every host the tests care
        // about; its contents don't matter for this check.
        let kernel = std::path::PathBuf::from("/bin/true");
        let result = KtstrVmBuilder::default()
            .kernel(&kernel)
            .memory_mb(0)
            .no_perf_mode(true)
            .build();
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("build() must reject memory_mb(0)"),
        };
        let msg = format!("{err:#}");
        assert!(
            msg.contains("memory_mb") && msg.contains("> 0"),
            "error must name the field and constraint: {msg}"
        );
    }

    #[test]
    fn builder_topology() {
        let b = KtstrVmBuilder::default().topology(1, 2, 4, 2);
        assert_eq!(b.topology.total_cpus(), 16);
        assert_eq!(b.topology.llcs, 2);
    }

    #[test]
    fn builder_requires_kernel() {
        let result = KtstrVmBuilder::default().build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_rejects_missing_kernel() {
        let result = KtstrVmBuilder::default()
            .kernel("/nonexistent/vmlinuz")
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_chain() {
        let b = KtstrVmBuilder::default()
            .topology(1, 2, 2, 2)
            .memory_mb(4096)
            .cmdline("root=/dev/sda")
            .timeout(Duration::from_secs(300));
        assert_eq!(b.memory_mb, Some(4096));
        assert_eq!(b.topology.total_cpus(), 8);
        assert_eq!(b.cmdline_extra, "root=/dev/sda");
        assert_eq!(b.timeout, Duration::from_secs(300));
    }

    #[test]
    fn builder_with_init_binary() {
        let exe = crate::resolve_current_exe().unwrap();
        let b = KtstrVmBuilder::default().init_binary(&exe);
        assert_eq!(b.init_binary.as_deref(), Some(exe.as_path()));
    }

    #[test]
    fn builder_rejects_missing_init_binary() {
        let result = KtstrVmBuilder::default()
            .kernel("/nonexistent/vmlinuz")
            .init_binary("/nonexistent/binary")
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_rejects_missing_scheduler_binary() {
        let exe = crate::resolve_current_exe().unwrap();
        let result = KtstrVmBuilder::default()
            .kernel(&exe)
            .scheduler_binary("/nonexistent/scheduler")
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_run_args() {
        let b = KtstrVmBuilder::default().run_args(&["run".into(), "--json".into()]);
        assert_eq!(b.run_args, vec!["run", "--json"]);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn builder_kernel_dir_resolves_bzimage() {
        let b = KtstrVmBuilder::default().kernel_dir("/some/linux");
        assert_eq!(
            b.kernel.as_deref(),
            Some(std::path::Path::new("/some/linux/arch/x86/boot/bzImage"))
        );
    }

    #[test]
    #[should_panic(expected = "invalid Topology")]
    fn builder_rejects_zero_llcs() {
        KtstrVmBuilder::default().topology(1, 0, 2, 2);
    }

    #[test]
    #[should_panic(expected = "invalid Topology")]
    fn builder_rejects_zero_cores() {
        KtstrVmBuilder::default().topology(1, 2, 0, 2);
    }

    #[test]
    #[should_panic(expected = "invalid Topology")]
    fn builder_rejects_zero_threads() {
        KtstrVmBuilder::default().topology(1, 2, 2, 0);
    }

    #[test]
    fn builder_watchdog_timeout_default() {
        let b = KtstrVmBuilder::default();
        assert_eq!(b.watchdog_timeout, Some(Duration::from_secs(5)));
    }

    #[test]
    fn builder_watchdog_timeout_override() {
        let b = KtstrVmBuilder::default().watchdog_timeout(Duration::from_secs(5));
        assert_eq!(b.watchdog_timeout, Some(Duration::from_secs(5)));
    }

    #[test]
    fn builder_monitor_thresholds_sets() {
        let t = crate::monitor::MonitorThresholds {
            max_imbalance_ratio: 2.0,
            ..Default::default()
        };
        let b = KtstrVmBuilder::default().monitor_thresholds(t);
        assert!(b.monitor_thresholds.is_some());
    }

    #[test]
    fn builder_sched_args() {
        let b = KtstrVmBuilder::default().sched_args(&["--enable-borrow".into()]);
        assert_eq!(b.sched_args, vec!["--enable-borrow"]);
    }

    #[test]
    fn builder_performance_mode_default_false() {
        let b = KtstrVmBuilder::default();
        assert!(!b.performance_mode);
    }

    #[test]
    fn builder_performance_mode_set() {
        let b = KtstrVmBuilder::default().performance_mode(true);
        assert!(b.performance_mode);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn builder_kernel_dir_resolves_image() {
        let b = KtstrVmBuilder::default().kernel_dir("/some/linux");
        assert_eq!(
            b.kernel.as_deref(),
            Some(std::path::Path::new("/some/linux/arch/arm64/boot/Image"))
        );
    }
}
