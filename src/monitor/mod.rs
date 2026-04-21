//! Host-side guest memory monitor and BPF map introspection.
//!
//! Reads per-CPU runqueue structures from guest VM memory via BTF-resolved
//! offsets. Observes scheduler state without instrumenting the guest
//! kernel or the scheduler under test.
//!
//! The [`bpf_map`] module provides host-side discovery and read/write
//! access to BPF maps in guest memory. The [`bpf_prog`] module provides
//! host-side enumeration of BPF programs and their verifier/runtime stats.
//! Both locate kernel objects by walking IDR xarrays (shared infrastructure
//! in the [`idr`] module) through page table translation. No guest
//! cooperation is needed.
//!
//! See the [Monitor](https://likewhatevs.github.io/ktstr/guide/architecture/monitor.html)
//! chapter of the guide.

pub mod bpf_map;
pub mod bpf_prog;
pub mod btf_offsets;
pub mod guest;
pub mod idr;
pub mod reader;
pub mod symbols;

/// Guest physical address of the top-level page-table page (CR3 on x86,
/// TTBR1 on aarch64). Newtype around `u64` so address kinds can't
/// accidentally mix — passing a `PageOffset` where a `Cr3Pa` is
/// expected fails to compile instead of silently walking the wrong
/// tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Cr3Pa(pub u64);

/// Kernel direct-map base (x86-64 `PAGE_OFFSET`, aarch64 linear map
/// base). Adding this to a DRAM offset yields a KVA; subtracting it
/// from a KVA yields the DRAM offset that `GuestMem` reads use.
/// Newtype around `u64`; see [`Cr3Pa`] for the rationale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PageOffset(pub u64);

/// DSQ depth above this value indicates uninitialized guest memory.
/// Real kernels never queue this many tasks on a single CPU's local DSQ.
pub const DSQ_PLAUSIBILITY_CEILING: u32 = 10_000;

/// Non-negative delta between two samples of a monotonic counter.
///
/// BPF / kernel counters can reset mid-run (scheduler restart, counter
/// re-init) so the raw `last - first` can be negative. Callers want a
/// rate-computation-safe value, so clamp to zero.
///
/// Shared with [`crate::timeline`] so both derivation paths agree on
/// the clamp semantics instead of each reinventing the same closure.
pub(crate) fn counter_delta(last: i64, first: i64) -> i64 {
    (last - first).max(0)
}

/// Number of tick periods a vCPU must have run before rq_clock is expected
/// to have advanced. 10 ticks gives the scheduler tick multiple chances
/// to fire and update rq_clock.
const PREEMPTION_TICK_MULTIPLE: u64 = 10;

/// Default HZ when CONFIG_HZ cannot be determined from the kernel.
/// 250 is the most conservative common value (longest tick period =
/// highest threshold), avoiding false stall detection.
const DEFAULT_HZ: u64 = 250;

/// Compute the vCPU preemption threshold for a given kernel.
///
/// Tries to determine CONFIG_HZ from (in order):
/// 1. Embedded IKCONFIG in the vmlinux (gzip blob after `IKCFG_ST` marker)
/// 2. `.config` next to the kernel image (build directory)
/// 3. Host `/boot/config-$(uname -r)` (covers virtme default: host == guest)
/// 4. Falls back to 250 (40ms threshold)
///
/// Returns the threshold in nanoseconds: `(1e9 / HZ) * 10`.
pub(crate) fn vcpu_preemption_threshold_ns(kernel_path: Option<&std::path::Path>) -> u64 {
    let hz = guest_kernel_hz(kernel_path);
    let tick_ns = 1_000_000_000u64 / hz;
    tick_ns * PREEMPTION_TICK_MULTIPLE
}

/// Determine CONFIG_HZ for the guest kernel.
///
/// The host's `/boot/config-$(uname -r)` is only consulted when the
/// caller passed no explicit `kernel_path` (virtme default: host ==
/// guest). A cached kernel whose IKCONFIG was stripped and whose
/// build `.config` is unavailable must NOT inherit the host's
/// CONFIG_HZ — the cached kernel was built with its own config, and
/// using the host's CONFIG_HZ would silently misscale every
/// tick-dependent threshold. Instead, such kernels fall through to
/// [`DEFAULT_HZ`], which is the conservative default.
pub(crate) fn guest_kernel_hz(kernel_path: Option<&std::path::Path>) -> u64 {
    if let Some(kp) = kernel_path {
        // Try embedded IKCONFIG in the vmlinux.
        if let Some(vmlinux) = find_vmlinux(kp)
            && let Some(hz) = read_hz_from_ikconfig(&vmlinux)
        {
            return hz;
        }

        // Try .config next to the kernel image.
        if let Some(hz) = read_hz_from_kernel_dir(kp) {
            return hz;
        }

        // Explicit kernel_path whose config can't be recovered —
        // don't fall through to the host's /boot/config; host HZ
        // is unrelated to the cached/built guest kernel's HZ.
        tracing::warn!(
            kernel = %kp.display(),
            default_hz = DEFAULT_HZ,
            "guest_kernel_hz: no IKCONFIG or .config alongside \
             kernel; falling back to DEFAULT_HZ rather than host \
             /boot/config (tick-dependent thresholds may be \
             conservative)"
        );
        return DEFAULT_HZ;
    }

    // No kernel path given → virtme-style run where guest kernel ==
    // host kernel. Host boot config is authoritative.
    if let Some(hz) = read_hz_from_boot_config() {
        return hz;
    }

    DEFAULT_HZ
}

use crate::vmm::find_vmlinux;

/// IKCONFIG marker: gzip data starts immediately after this 8-byte sequence.
const IKCONFIG_MAGIC: &[u8] = b"IKCFG_ST";

/// ELF sections [`read_hz_from_ikconfig`] reads.
///
/// The cached-vmlinux strip pipeline
/// ([`crate::cache::strip_vmlinux_debug`]) preserves these bytes
/// verbatim via its keep-list predicate.
pub(crate) const VMLINUX_KEEP_SECTIONS: &[&[u8]] = &[
    b".rodata", // IKCONFIG gzip blob, bracketed by IKCFG_ST / IKCFG_ED markers
];

/// Extract CONFIG_HZ from the embedded IKCONFIG blob in a vmlinux ELF.
///
/// The kernel (when built with CONFIG_IKCONFIG) embeds a gzip-compressed
/// copy of `.config` in `.rodata`, bracketed by `IKCFG_ST` / `IKCFG_ED`
/// markers. This function scans the raw bytes for the marker, decompresses
/// the gzip data, and parses CONFIG_HZ from the result.
fn read_hz_from_ikconfig(vmlinux_path: &std::path::Path) -> Option<u64> {
    let data = std::fs::read(vmlinux_path).ok()?;
    // vmlinux images are tens of MB; the old
    // `windows(8).position(|w| w == IKCFG_ST)` was a naive O(n)
    // byte-wise scan. memchr's two-way matcher uses the available
    // SIMD path (x86_64 AVX2 / aarch64 Neon) and cuts scan time by
    // a constant factor on every host we care about.
    let pos = memchr::memmem::find(&data, IKCONFIG_MAGIC)?;
    let gz_start = pos + IKCONFIG_MAGIC.len();
    if gz_start >= data.len() {
        return None;
    }
    let cursor = std::io::Cursor::new(&data[gz_start..]);
    let mut decoder = flate2::read::GzDecoder::new(cursor);
    let mut config = String::new();
    std::io::Read::read_to_string(&mut decoder, &mut config).ok()?;
    parse_config_hz(&config)
}

/// Look for a `.config` file in the kernel's build directory and parse
/// CONFIG_HZ from it. Walks up from the kernel image path (e.g.
/// `<root>/arch/x86/boot/bzImage`) toward the build root.
fn read_hz_from_kernel_dir(kernel_path: &std::path::Path) -> Option<u64> {
    let mut dir = kernel_path.parent()?;
    // Walk up at most 4 levels (arch/x86/boot/bzImage -> root).
    for _ in 0..4 {
        let config = dir.join(".config");
        if config.exists() {
            let contents = std::fs::read_to_string(&config).ok()?;
            return parse_config_hz(&contents);
        }
        dir = dir.parent()?;
    }
    None
}

/// Parse CONFIG_HZ=N from `/boot/config-$(uname -r)`.
fn read_hz_from_boot_config() -> Option<u64> {
    let uname = rustix::system::uname();
    let release = uname.release().to_str().ok()?;
    let path = format!("/boot/config-{release}");
    let contents = std::fs::read_to_string(path).ok()?;
    parse_config_hz(&contents)
}

/// Extract `CONFIG_HZ=N` from kernel config text.
fn parse_config_hz(config: &str) -> Option<u64> {
    for line in config.lines() {
        let line = line.trim();
        if let Some(val) = line.strip_prefix("CONFIG_HZ=") {
            return val.parse().ok();
        }
    }
    None
}

/// Check whether a single monitor sample contains plausible data.
///
/// Returns false when any CPU's local_dsq_depth exceeds the plausibility
/// ceiling, indicating uninitialized guest memory rather than real
/// scheduler state.
pub fn sample_looks_valid(sample: &MonitorSample) -> bool {
    sample
        .cpus
        .iter()
        .all(|cpu| cpu.local_dsq_depth <= DSQ_PLAUSIBILITY_CEILING)
}

/// Find a vmlinux for tests.
///
/// Reads `KTSTR_KERNEL` for an explicit directory override and
/// delegates the remaining search to [`crate::kernel_path::resolve_btf`]
/// so tests pick the same kernel the rest of ktstr does. See that
/// function for the exact resolution order.
#[cfg(test)]
pub fn find_test_vmlinux() -> Option<std::path::PathBuf> {
    let kernel_dir = std::env::var("KTSTR_KERNEL").ok();
    let result = crate::kernel_path::resolve_btf(kernel_dir.as_deref());
    if result.is_none() {
        eprintln!("ktstr: SKIP: no vmlinux found (set KTSTR_KERNEL or place vmlinux in ./linux)");
    }
    result
}

/// Collected monitor data from a VM run.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct MonitorReport {
    /// Periodic snapshots of per-CPU state.
    pub samples: Vec<MonitorSample>,
    /// Aggregated summary statistics.
    pub summary: MonitorSummary,
    /// vCPU preemption threshold (ns) derived from the guest kernel's
    /// CONFIG_HZ at the time the VM ran. Used by evaluate() to gate
    /// stall detection. 0 means use a default.
    pub preemption_threshold_ns: u64,
    /// Post-write readback of the scx_sched.watchdog_timeout field.
    /// Framework-internal regression guard that the host-side override
    /// actually lands in guest memory; populated once per VM run after
    /// the first successful deref.
    #[doc(hidden)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub watchdog_observation: Option<WatchdogObservation>,
}

/// Observation of the `scx_sched.watchdog_timeout` override,
/// recorded once by the monitor loop after the first successful
/// write to the runtime-allocated scx_sched struct.
///
/// Regression guard for the host-write mechanism: when the kernel
/// refactors the location of watchdog_timeout (as happened before
/// the runtime scx_root deref was introduced), `observed_jiffies`
/// will diverge from `expected_jiffies` and the test will fail.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WatchdogObservation {
    /// Jiffies value the override was configured to write.
    pub expected_jiffies: u64,
    /// Jiffies value read back from guest memory after the write.
    pub observed_jiffies: u64,
}

/// Tracks consecutive threshold violations and records the worst run.
///
/// Used by `MonitorThresholds::evaluate` (post-hoc, from the collected
/// sample vector) and the reactive `monitor_loop` dump path. Both paths
/// share the tracker so "sustained for N samples" means exactly the
/// same thing to the inline SysRq-D trigger and the after-the-fact
/// verdict. Call `record(true)` on violation, `record(false)` on pass.
#[derive(Debug, Clone, Default)]
pub(crate) struct SustainedViolationTracker {
    consecutive: usize,
    worst_run: usize,
    worst_value: f64,
    worst_at: usize,
}

impl SustainedViolationTracker {
    /// Record a sample. `violated`: whether the threshold was exceeded.
    /// `value`: the metric value for this sample. `at`: sample index.
    pub(crate) fn record(&mut self, violated: bool, value: f64, at: usize) {
        if violated {
            self.consecutive += 1;
            if self.consecutive > self.worst_run {
                self.worst_run = self.consecutive;
                self.worst_value = value;
                self.worst_at = at;
            }
        } else {
            self.consecutive = 0;
        }
    }

    /// Whether the worst run met or exceeded the sustained threshold.
    pub(crate) fn sustained(&self, threshold: usize) -> bool {
        self.worst_run >= threshold
    }
}

/// Point-in-time snapshot of all CPUs.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct MonitorSample {
    /// Milliseconds since VM start.
    pub elapsed_ms: u64,
    /// Per-CPU state at this instant.
    pub cpus: Vec<CpuSnapshot>,
    /// Per-program BPF runtime stats (summed across CPUs).
    /// None when no struct_ops programs are loaded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prog_stats: Option<Vec<bpf_prog::ProgRuntimeStats>>,
}

impl MonitorSample {
    /// Create a sample with no prog_stats.
    pub fn new(elapsed_ms: u64, cpus: Vec<CpuSnapshot>) -> Self {
        Self {
            elapsed_ms,
            cpus,
            prog_stats: None,
        }
    }

    /// Compute the imbalance ratio for this sample: max(nr_running) / max(1, min(nr_running)).
    /// Returns 1.0 for empty samples, 0.0 when all CPUs have nr_running=0.
    pub fn imbalance_ratio(&self) -> f64 {
        if self.cpus.is_empty() {
            return 1.0;
        }
        let mut min_nr = u32::MAX;
        let mut max_nr = 0u32;
        for cpu in &self.cpus {
            min_nr = min_nr.min(cpu.nr_running);
            max_nr = max_nr.max(cpu.nr_running);
        }
        max_nr as f64 / min_nr.max(1) as f64
    }

    /// Sum a field from event counters across all CPUs.
    /// Returns `None` if no CPU has event counters.
    pub fn sum_event_field(&self, f: fn(&ScxEventCounters) -> i64) -> Option<i64> {
        let mut total = 0i64;
        let mut any = false;
        for cpu in &self.cpus {
            if let Some(ev) = &cpu.event_counters {
                total += f(ev);
                any = true;
            }
        }
        any.then_some(total)
    }
}

/// Per-CPU state read from guest VM memory.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct CpuSnapshot {
    /// Total runnable tasks on this CPU (`rq.nr_running`).
    pub nr_running: u32,
    /// Tasks managed by the sched_ext scheduler (`scx_rq.nr_running`).
    pub scx_nr_running: u32,
    /// Depth of the scx local dispatch queue (`scx_rq.local_dsq.nr`).
    pub local_dsq_depth: u32,
    /// Runqueue clock value (`rq.clock`). Non-advancing clock indicates a stall.
    pub rq_clock: u64,
    /// sched_ext flags for this CPU (`scx_rq.flags`).
    pub scx_flags: u32,
    /// scx event counters (cumulative). None when event counter
    /// offsets are unavailable or scx_root is not set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_counters: Option<ScxEventCounters>,
    /// Runqueue schedstat fields (cumulative). None when CONFIG_SCHEDSTATS
    /// is not enabled (schedstat offsets unavailable).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schedstat: Option<RqSchedstat>,
    /// Cumulative CPU time (ns) of the vCPU thread hosting this CPU.
    /// Used by evaluate() to distinguish real stalls from host preemption.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vcpu_cpu_time_ns: Option<u64>,
    /// Sched domain tree for this CPU. Each entry is one domain level,
    /// ordered from lowest (e.g. SMT) to highest (e.g. NUMA). None when
    /// sched_domain offsets are unavailable or `rq->sd` is null.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sched_domains: Option<Vec<SchedDomainSnapshot>>,
}

/// Per-CPU runqueue schedstat fields read from guest memory.
///
/// Matches kernel `struct rq` schedstat fields (guarded by CONFIG_SCHEDSTATS).
/// `run_delay` and `pcount` come from the embedded `struct sched_info`;
/// the remaining fields are direct members of `struct rq`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct RqSchedstat {
    /// Cumulative scheduling delay (ns) on this CPU (`rq.rq_sched_info.run_delay`).
    pub run_delay: u64,
    /// Count of non-idle task arrivals on this CPU (`rq.rq_sched_info.pcount`).
    pub pcount: u64,
    /// Yield count (`rq.yld_count`).
    pub yld_count: u32,
    /// Context switch count (`rq.sched_count`).
    pub sched_count: u32,
    /// Go-idle count (`rq.sched_goidle`).
    pub sched_goidle: u32,
    /// Try-to-wake-up count (`rq.ttwu_count`).
    pub ttwu_count: u32,
    /// Try-to-wake-up local count (`rq.ttwu_local`).
    pub ttwu_local: u32,
}

/// Snapshot of one `struct sched_domain` level for a single CPU.
///
/// Domains are ordered from lowest (e.g. SMT, level 0) to highest
/// (e.g. NUMA, level N) following the kernel's `sd->parent` chain.
/// `newidle_call`, `newidle_success`, and `newidle_ratio` are `None`
/// on 6.16+ where the kernel removed these fields.
/// CONFIG_SCHEDSTATS load balancing stats are in the optional `stats`
/// field.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct SchedDomainSnapshot {
    /// Domain level number (`sd->level`). 0 = innermost (e.g. SMT).
    pub level: i32,
    /// Domain name from `sd->name` (e.g. "SMT", "MC", "DIE", "NUMA").
    pub name: String,
    /// Domain flags (`sd->flags`). SD_* values.
    pub flags: i32,
    /// Number of CPUs in this domain's span (`sd->span_weight`).
    pub span_weight: u32,

    // -- Runtime fields --
    /// Current balance interval in ms (`sd->balance_interval`).
    pub balance_interval: u32,
    /// Consecutive load balance failures (`sd->nr_balance_failed`).
    pub nr_balance_failed: u32,
    /// Number of newidle balance calls (`sd->newidle_call`).
    /// None when BTF lacks this field (added in 7.0; backported to
    /// 6.18.5+, 6.12.65+). Not present on 6.16-6.18.4.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub newidle_call: Option<u32>,
    /// Successful newidle balance calls (`sd->newidle_success`).
    /// None when BTF lacks this field (added in 7.0; backported to
    /// 6.18.5+, 6.12.65+). Not present on 6.16-6.18.4.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub newidle_success: Option<u32>,
    /// Newidle balance ratio (`sd->newidle_ratio`).
    /// None when BTF lacks this field (added in 7.0; backported to
    /// 6.18.5+, 6.12.65+). Not present on 6.16-6.18.4.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub newidle_ratio: Option<u32>,
    /// Max cost of newidle load balancing in ns (`sd->max_newidle_lb_cost`).
    pub max_newidle_lb_cost: u64,

    /// CONFIG_SCHEDSTATS load balancing stats. None when
    /// CONFIG_SCHEDSTATS is not enabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stats: Option<SchedDomainStats>,
}

/// CONFIG_SCHEDSTATS load balancing stats for one `struct sched_domain`.
///
/// Array fields have `CPU_MAX_IDLE_TYPES` (3) elements indexed by
/// `cpu_idle_type`: \[0\] = CPU_NOT_IDLE, \[1\] = CPU_IDLE,
/// \[2\] = CPU_NEWLY_IDLE. All counters are cumulative — compute
/// deltas between samples to get rates.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct SchedDomainStats {
    /// `sd->lb_count[CPU_MAX_IDLE_TYPES]`: number of load balance calls.
    pub lb_count: [u32; btf_offsets::CPU_MAX_IDLE_TYPES],
    /// `sd->lb_failed[CPU_MAX_IDLE_TYPES]`: load balance calls that found
    /// imbalance but failed to move any task.
    pub lb_failed: [u32; btf_offsets::CPU_MAX_IDLE_TYPES],
    /// `sd->lb_balanced[CPU_MAX_IDLE_TYPES]`: load balance calls that
    /// found no imbalance.
    pub lb_balanced: [u32; btf_offsets::CPU_MAX_IDLE_TYPES],
    /// `sd->lb_imbalance_load[CPU_MAX_IDLE_TYPES]`: times imbalance was
    /// load-based.
    pub lb_imbalance_load: [u32; btf_offsets::CPU_MAX_IDLE_TYPES],
    /// `sd->lb_imbalance_util[CPU_MAX_IDLE_TYPES]`: times imbalance was
    /// utilization-based.
    pub lb_imbalance_util: [u32; btf_offsets::CPU_MAX_IDLE_TYPES],
    /// `sd->lb_imbalance_task[CPU_MAX_IDLE_TYPES]`: times imbalance was
    /// task-count-based.
    pub lb_imbalance_task: [u32; btf_offsets::CPU_MAX_IDLE_TYPES],
    /// `sd->lb_imbalance_misfit[CPU_MAX_IDLE_TYPES]`: times imbalance was
    /// due to misfit task.
    pub lb_imbalance_misfit: [u32; btf_offsets::CPU_MAX_IDLE_TYPES],
    /// `sd->lb_gained[CPU_MAX_IDLE_TYPES]`: tasks pulled during load balance.
    pub lb_gained: [u32; btf_offsets::CPU_MAX_IDLE_TYPES],
    /// `sd->lb_hot_gained[CPU_MAX_IDLE_TYPES]`: cache-hot tasks pulled.
    pub lb_hot_gained: [u32; btf_offsets::CPU_MAX_IDLE_TYPES],
    /// `sd->lb_nobusyg[CPU_MAX_IDLE_TYPES]`: times no busy group was found.
    pub lb_nobusyg: [u32; btf_offsets::CPU_MAX_IDLE_TYPES],
    /// `sd->lb_nobusyq[CPU_MAX_IDLE_TYPES]`: times no busy queue was found.
    pub lb_nobusyq: [u32; btf_offsets::CPU_MAX_IDLE_TYPES],

    /// `sd->alb_count`: active load balance attempts.
    pub alb_count: u32,
    /// `sd->alb_failed`: active load balance failures.
    pub alb_failed: u32,
    /// `sd->alb_pushed`: tasks pushed via active load balancing.
    pub alb_pushed: u32,

    /// `sd->sbe_count`: exec balance attempts.
    pub sbe_count: u32,
    /// `sd->sbe_balanced`: exec balance found no imbalance.
    pub sbe_balanced: u32,
    /// `sd->sbe_pushed`: tasks pushed via exec balancing.
    pub sbe_pushed: u32,

    /// `sd->sbf_count`: fork balance attempts.
    pub sbf_count: u32,
    /// `sd->sbf_balanced`: fork balance found no imbalance.
    pub sbf_balanced: u32,
    /// `sd->sbf_pushed`: tasks pushed via fork balancing.
    pub sbf_pushed: u32,

    /// `sd->ttwu_wake_remote`: wakeups targeting a remote CPU.
    pub ttwu_wake_remote: u32,
    /// `sd->ttwu_move_affine`: wakeups moved to an affine CPU.
    pub ttwu_move_affine: u32,
    /// `sd->ttwu_move_balance`: wakeups moved for load balance.
    pub ttwu_move_balance: u32,
}

/// Cumulative scx event counter values for a single CPU.
/// These are s64 in the kernel but always non-negative; stored as i64.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ScxEventCounters {
    /// `SCX_EV_SELECT_CPU_FALLBACK`: scheduler's `ops.select_cpu()` failed to find a CPU.
    pub select_cpu_fallback: i64,
    /// `SCX_EV_DISPATCH_LOCAL_DSQ_OFFLINE`: dispatch to an offline CPU's local DSQ.
    pub dispatch_local_dsq_offline: i64,
    /// `SCX_EV_DISPATCH_KEEP_LAST`: CPU re-dispatched the previously running task.
    pub dispatch_keep_last: i64,
    /// `SCX_EV_ENQ_SKIP_EXITING`: enqueue skipped because the task is exiting.
    pub enq_skip_exiting: i64,
    /// `SCX_EV_ENQ_SKIP_MIGRATION_DISABLED`: enqueue skipped because migration is disabled.
    pub enq_skip_migration_disabled: i64,
    /// `SCX_EV_REENQ_IMMED`: task re-enqueued because CPU was unavailable for immediate execution.
    pub reenq_immed: i64,
    /// `SCX_EV_REENQ_LOCAL_REPEAT`: recursive local DSQ re-enqueue from `SCX_ENQ_IMMED` race.
    pub reenq_local_repeat: i64,
    /// `SCX_EV_REFILL_SLICE_DFL`: time slice refilled with `SCX_SLICE_DFL`.
    pub refill_slice_dfl: i64,
    /// `SCX_EV_BYPASS_DURATION`: total bypass mode duration in nanoseconds.
    pub bypass_duration: i64,
    /// `SCX_EV_BYPASS_DISPATCH`: tasks dispatched during bypass mode.
    pub bypass_dispatch: i64,
    /// `SCX_EV_BYPASS_ACTIVATE`: bypass mode activations.
    pub bypass_activate: i64,
    /// `SCX_EV_INSERT_NOT_OWNED`: attempts to insert a non-owned task into a DSQ.
    pub insert_not_owned: i64,
    /// `SCX_EV_SUB_BYPASS_DISPATCH`: tasks from bypassing descendants scheduled from sub_bypass_dsq.
    pub sub_bypass_dispatch: i64,
}

/// Aggregated monitor statistics from a set of samples.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct MonitorSummary {
    /// Number of samples collected.
    pub total_samples: usize,
    /// Peak imbalance ratio across all samples: `max(nr_running) / max(1, min(nr_running))`.
    pub max_imbalance_ratio: f64,
    /// Peak local DSQ depth across all CPUs and samples.
    pub max_local_dsq_depth: u32,
    /// Whether any CPU's `rq_clock` failed to advance between consecutive
    /// samples. Idle CPUs (`nr_running == 0` in both samples) are exempt.
    pub stall_detected: bool,
    /// Average imbalance ratio across valid samples.
    pub avg_imbalance_ratio: f64,
    /// Average nr_running per CPU across valid samples.
    pub avg_nr_running: f64,
    /// Average local DSQ depth per CPU across valid samples.
    pub avg_local_dsq_depth: f64,
    /// Aggregate event counter deltas over the monitoring window.
    /// None when event counters are not available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_deltas: Option<ScxEventDeltas>,
    /// Aggregate schedstat deltas over the monitoring window.
    /// None when CONFIG_SCHEDSTATS is not enabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schedstat_deltas: Option<SchedstatDeltas>,
    /// Per-program BPF callback profile over the monitoring window.
    /// None when no struct_ops programs are loaded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prog_stats_deltas: Option<Vec<ProgStatsDelta>>,
}

/// Per-program BPF callback profile computed from first/last monitor samples.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ProgStatsDelta {
    /// Program name.
    pub name: String,
    /// Total invocations over the monitoring window.
    pub cnt: u64,
    /// Total CPU time in nanoseconds over the monitoring window.
    pub nsecs: u64,
    /// Average nanoseconds per call (nsecs / cnt). 0 when cnt is 0.
    pub nsecs_per_call: f64,
}

/// Aggregate schedstat deltas computed from first/last monitor samples.
///
/// All values are summed across CPUs and represent the delta over the
/// monitoring window. Rates are per second.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct SchedstatDeltas {
    /// Total scheduling delay increase (ns) across all CPUs.
    pub total_run_delay: u64,
    /// Run delay per second (ns/s) across all CPUs.
    pub run_delay_rate: f64,
    /// Total pcount increase across all CPUs.
    pub total_pcount: u64,
    /// Total context switch increase across all CPUs.
    pub total_sched_count: u64,
    /// Context switches per second across all CPUs.
    pub sched_count_rate: f64,
    /// Total yield count increase across all CPUs.
    pub total_yld_count: u64,
    /// Total go-idle count increase across all CPUs.
    pub total_sched_goidle: u64,
    /// Total ttwu count increase across all CPUs.
    pub total_ttwu_count: u64,
    /// Total ttwu_local count increase across all CPUs.
    pub total_ttwu_local: u64,
}

/// Aggregate event counter statistics computed from first/last samples.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ScxEventDeltas {
    /// Total select_cpu_fallback events across all CPUs over the window.
    pub total_fallback: i64,
    /// Fallback events per second (total_fallback / duration_secs).
    pub fallback_rate: f64,
    /// Max single-sample delta of fallback across all CPUs.
    pub max_fallback_burst: i64,
    /// Total dispatch_local_dsq_offline events.
    pub total_dispatch_offline: i64,
    /// Total dispatch_keep_last events.
    pub total_dispatch_keep_last: i64,
    /// Keep-last events per second (total_dispatch_keep_last / duration_secs).
    pub keep_last_rate: f64,
    /// Total enq_skip_exiting events.
    pub total_enq_skip_exiting: i64,
    /// Total enq_skip_migration_disabled events.
    pub total_enq_skip_migration_disabled: i64,
    /// Total reenq_immed events.
    pub total_reenq_immed: i64,
    /// Total reenq_local_repeat events.
    pub total_reenq_local_repeat: i64,
    /// Total refill_slice_dfl events.
    pub total_refill_slice_dfl: i64,
    /// Total bypass_duration in nanoseconds.
    pub total_bypass_duration: i64,
    /// Total bypass_dispatch events.
    pub total_bypass_dispatch: i64,
    /// Total bypass_activate events.
    pub total_bypass_activate: i64,
    /// Total insert_not_owned events.
    pub total_insert_not_owned: i64,
    /// Total sub_bypass_dispatch events.
    pub total_sub_bypass_dispatch: i64,
}

impl MonitorSummary {
    /// Summarize a run's monitor samples using the derived default
    /// preemption threshold (equivalent to
    /// [`from_samples_with_threshold`](Self::from_samples_with_threshold)
    /// with `0`).
    pub fn from_samples(samples: &[MonitorSample]) -> Self {
        Self::from_samples_with_threshold(samples, 0)
    }

    /// Like [`from_samples`](Self::from_samples) but uses an explicit
    /// preemption threshold (ns) for stall detection. Pass 0 to derive
    /// the threshold from the guest kernel's `CONFIG_HZ` by calling
    /// [`vcpu_preemption_threshold_ns`], which tries (in order) the
    /// embedded IKCONFIG in the guest `vmlinux`, a `.config` beside
    /// the kernel image, the host's `/boot/config-$(uname -r)`, and
    /// finally the built-in `DEFAULT_HZ`.
    pub fn from_samples_with_threshold(
        samples: &[MonitorSample],
        preemption_threshold_ns: u64,
    ) -> Self {
        if samples.is_empty() {
            return Self::default();
        }

        let mut max_imbalance_ratio: f64 = 1.0;
        let mut max_local_dsq_depth: u32 = 0;
        let mut sum_imbalance_ratio: f64 = 0.0;
        let mut sum_nr_running: f64 = 0.0;
        let mut sum_local_dsq_depth: f64 = 0.0;
        let mut valid_sample_count: usize = 0;
        let mut total_cpu_readings: usize = 0;

        for sample in samples {
            if sample.cpus.is_empty() || !sample_looks_valid(sample) {
                continue;
            }
            valid_sample_count += 1;
            for cpu in &sample.cpus {
                max_local_dsq_depth = max_local_dsq_depth.max(cpu.local_dsq_depth);
                sum_nr_running += cpu.nr_running as f64;
                sum_local_dsq_depth += cpu.local_dsq_depth as f64;
                total_cpu_readings += 1;
            }
            let ratio = sample.imbalance_ratio();
            sum_imbalance_ratio += ratio;
            if ratio > max_imbalance_ratio {
                max_imbalance_ratio = ratio;
            }
        }

        let avg_imbalance_ratio = if valid_sample_count > 0 {
            sum_imbalance_ratio / valid_sample_count as f64
        } else {
            0.0
        };
        let avg_nr_running = if total_cpu_readings > 0 {
            sum_nr_running / total_cpu_readings as f64
        } else {
            0.0
        };
        let avg_local_dsq_depth = if total_cpu_readings > 0 {
            sum_local_dsq_depth / total_cpu_readings as f64
        } else {
            0.0
        };

        // Stall detection: any CPU whose rq_clock did not advance between
        // consecutive samples. Skip invalid samples.
        // Exempt idle CPUs: nr_running==0 in both samples means the tick
        // is stopped (NOHZ) and rq_clock legitimately does not advance.
        // Exempt preempted vCPUs: vcpu_cpu_time_ns didn't advance enough
        // means the host preempted the vCPU thread.
        let threshold = if preemption_threshold_ns > 0 {
            preemption_threshold_ns
        } else {
            vcpu_preemption_threshold_ns(None)
        };
        let mut stall_detected = false;
        let valid_samples: Vec<&MonitorSample> = samples
            .iter()
            .filter(|s| !s.cpus.is_empty() && sample_looks_valid(s))
            .collect();
        for w in valid_samples.windows(2) {
            let prev = w[0];
            let curr = w[1];
            let cpu_count = prev.cpus.len().min(curr.cpus.len());
            for cpu in 0..cpu_count {
                if reader::is_cpu_stalled(&prev.cpus[cpu], &curr.cpus[cpu], threshold) {
                    stall_detected = true;
                    break;
                }
            }
            if stall_detected {
                break;
            }
        }

        let event_deltas = Self::compute_event_deltas(samples);
        let schedstat_deltas = Self::compute_schedstat_deltas(samples);
        let prog_stats_deltas = Self::compute_prog_stats_deltas(samples);

        Self {
            total_samples: samples.len(),
            max_imbalance_ratio,
            max_local_dsq_depth,
            stall_detected,
            avg_imbalance_ratio,
            avg_nr_running,
            avg_local_dsq_depth,
            event_deltas,
            schedstat_deltas,
            prog_stats_deltas,
        }
    }

    /// Compute event counter deltas from the sample series.
    /// Returns None if no samples have event counters.
    fn compute_event_deltas(samples: &[MonitorSample]) -> Option<ScxEventDeltas> {
        // Find first and last samples that have event counters on any CPU.
        let has_events = |s: &MonitorSample| s.cpus.iter().any(|c| c.event_counters.is_some());
        let first = samples.iter().find(|s| has_events(s))?;
        let last = samples.iter().rev().find(|s| has_events(s))?;

        let total_fallback = counter_delta(
            last.sum_event_field(|e| e.select_cpu_fallback).unwrap_or(0),
            first
                .sum_event_field(|e| e.select_cpu_fallback)
                .unwrap_or(0),
        );
        let total_keep_last = counter_delta(
            last.sum_event_field(|e| e.dispatch_keep_last).unwrap_or(0),
            first.sum_event_field(|e| e.dispatch_keep_last).unwrap_or(0),
        );

        // Compute rates.
        let duration_ms = last.elapsed_ms.saturating_sub(first.elapsed_ms);
        let duration_secs = duration_ms as f64 / 1000.0;
        let fallback_rate = if duration_secs > 0.0 {
            total_fallback as f64 / duration_secs
        } else {
            0.0
        };
        let keep_last_rate = if duration_secs > 0.0 {
            total_keep_last as f64 / duration_secs
        } else {
            0.0
        };

        // Max per-sample fallback burst: largest delta between consecutive
        // samples, summed across all CPUs. A counter reset between
        // samples yields a negative raw delta — ignore it rather than
        // letting it decrease the running max.
        let mut max_fallback_burst: i64 = 0;
        for w in samples.windows(2) {
            let prev_sum = w[0].sum_event_field(|e| e.select_cpu_fallback).unwrap_or(0);
            let curr_sum = w[1].sum_event_field(|e| e.select_cpu_fallback).unwrap_or(0);
            let delta = counter_delta(curr_sum, prev_sum);
            if delta > max_fallback_burst {
                max_fallback_burst = delta;
            }
        }

        let delta = |f: fn(&ScxEventCounters) -> i64| -> i64 {
            counter_delta(
                last.sum_event_field(f).unwrap_or(0),
                first.sum_event_field(f).unwrap_or(0),
            )
        };

        Some(ScxEventDeltas {
            total_fallback,
            fallback_rate,
            max_fallback_burst,
            total_dispatch_offline: delta(|e| e.dispatch_local_dsq_offline),
            total_dispatch_keep_last: total_keep_last,
            keep_last_rate,
            total_enq_skip_exiting: delta(|e| e.enq_skip_exiting),
            total_enq_skip_migration_disabled: delta(|e| e.enq_skip_migration_disabled),
            total_reenq_immed: delta(|e| e.reenq_immed),
            total_reenq_local_repeat: delta(|e| e.reenq_local_repeat),
            total_refill_slice_dfl: delta(|e| e.refill_slice_dfl),
            total_bypass_duration: delta(|e| e.bypass_duration),
            total_bypass_dispatch: delta(|e| e.bypass_dispatch),
            total_bypass_activate: delta(|e| e.bypass_activate),
            total_insert_not_owned: delta(|e| e.insert_not_owned),
            total_sub_bypass_dispatch: delta(|e| e.sub_bypass_dispatch),
        })
    }

    /// Compute schedstat deltas from the sample series.
    /// Returns None if no samples have schedstat data on any CPU.
    fn compute_schedstat_deltas(samples: &[MonitorSample]) -> Option<SchedstatDeltas> {
        let has_schedstat = |s: &MonitorSample| s.cpus.iter().any(|c| c.schedstat.is_some());
        let first = samples.iter().find(|s| has_schedstat(s))?;
        let last = samples.iter().rev().find(|s| has_schedstat(s))?;

        let sum_field = |s: &MonitorSample, f: fn(&RqSchedstat) -> u64| -> u64 {
            s.cpus
                .iter()
                .filter_map(|c| c.schedstat.as_ref().map(&f))
                .sum()
        };
        let sum_field_u32 = |s: &MonitorSample, f: fn(&RqSchedstat) -> u32| -> u64 {
            s.cpus
                .iter()
                .filter_map(|c| c.schedstat.as_ref().map(|ss| f(ss) as u64))
                .sum()
        };

        let total_run_delay =
            sum_field(last, |ss| ss.run_delay).saturating_sub(sum_field(first, |ss| ss.run_delay));
        let total_pcount =
            sum_field(last, |ss| ss.pcount).saturating_sub(sum_field(first, |ss| ss.pcount));
        let total_sched_count = sum_field_u32(last, |ss| ss.sched_count)
            .saturating_sub(sum_field_u32(first, |ss| ss.sched_count));
        let total_yld_count = sum_field_u32(last, |ss| ss.yld_count)
            .saturating_sub(sum_field_u32(first, |ss| ss.yld_count));
        let total_sched_goidle = sum_field_u32(last, |ss| ss.sched_goidle)
            .saturating_sub(sum_field_u32(first, |ss| ss.sched_goidle));
        let total_ttwu_count = sum_field_u32(last, |ss| ss.ttwu_count)
            .saturating_sub(sum_field_u32(first, |ss| ss.ttwu_count));
        let total_ttwu_local = sum_field_u32(last, |ss| ss.ttwu_local)
            .saturating_sub(sum_field_u32(first, |ss| ss.ttwu_local));

        let duration_ms = last.elapsed_ms.saturating_sub(first.elapsed_ms);
        let duration_secs = duration_ms as f64 / 1000.0;
        let run_delay_rate = if duration_secs > 0.0 {
            total_run_delay as f64 / duration_secs
        } else {
            0.0
        };
        let sched_count_rate = if duration_secs > 0.0 {
            total_sched_count as f64 / duration_secs
        } else {
            0.0
        };

        Some(SchedstatDeltas {
            total_run_delay,
            run_delay_rate,
            total_pcount,
            total_sched_count,
            sched_count_rate,
            total_yld_count,
            total_sched_goidle,
            total_ttwu_count,
            total_ttwu_local,
        })
    }

    /// Compute per-program callback profile from first/last samples
    /// that contain prog_stats.
    fn compute_prog_stats_deltas(samples: &[MonitorSample]) -> Option<Vec<ProgStatsDelta>> {
        let first = samples.iter().find(|s| s.prog_stats.is_some())?;
        let last = samples.iter().rev().find(|s| s.prog_stats.is_some())?;

        let first_progs = first.prog_stats.as_ref()?;
        let last_progs = last.prog_stats.as_ref()?;

        let deltas: Vec<ProgStatsDelta> = last_progs
            .iter()
            .map(|lp| {
                let fp = first_progs.iter().find(|p| p.name == lp.name);
                let cnt = lp.cnt.saturating_sub(fp.map_or(0, |p| p.cnt));
                let nsecs = lp.nsecs.saturating_sub(fp.map_or(0, |p| p.nsecs));
                let nsecs_per_call = if cnt > 0 {
                    nsecs as f64 / cnt as f64
                } else {
                    0.0
                };
                ProgStatsDelta {
                    name: lp.name.clone(),
                    cnt,
                    nsecs,
                    nsecs_per_call,
                }
            })
            .collect();

        if deltas.is_empty() {
            None
        } else {
            Some(deltas)
        }
    }
}

/// Configurable thresholds for monitor-based pass/fail verdicts.
#[derive(Debug, Clone, Copy)]
pub struct MonitorThresholds {
    /// Max allowed imbalance ratio (max_nr_running / max(1, min_nr_running)).
    pub max_imbalance_ratio: f64,
    /// Max allowed local DSQ depth on any CPU in any sample.
    pub max_local_dsq_depth: u32,
    /// Fail when any CPU's rq_clock does not advance between consecutive samples.
    pub fail_on_stall: bool,
    /// Number of consecutive samples that must violate a threshold before failing.
    pub sustained_samples: usize,
    /// Max sustained select_cpu_fallback events/s across all CPUs.
    pub max_fallback_rate: f64,
    /// Max sustained dispatch_keep_last events/s across all CPUs.
    pub max_keep_last_rate: f64,
}

impl MonitorThresholds {
    /// Default thresholds, usable in const context.
    ///
    /// - imbalance 4.0: a scheduler that can't keep CPUs within 4x
    ///   load for `sustained_samples` consecutive reads has a real
    ///   balancing problem. Lower ratios (2-3) false-positive during
    ///   cpuset transitions when cgroups are being created/destroyed.
    /// - DSQ depth 50: local DSQ is a per-CPU overflow queue. Sustained
    ///   depth > 50 means the scheduler is not consuming dispatched tasks.
    ///   Transient spikes during cpuset changes are filtered by the
    ///   sustained_samples window.
    /// - fail_on_stall true: rq_clock not advancing on a CPU with
    ///   runnable tasks means the scheduler stalled. Idle CPUs
    ///   (nr_running==0 in both samples) are exempt because NOHZ
    ///   stops the tick. Preempted vCPUs are exempt when the vCPU
    ///   thread's CPU time didn't advance past the preemption
    ///   threshold. Uses the sustained_samples window.
    /// - sustained_samples 5: at ~100ms sample interval, requires ~500ms
    ///   of sustained violation. Filters transient spikes from cpuset
    ///   reconfiguration, cgroup creation, and scheduler restart.
    /// - max_fallback_rate 200.0: select_cpu_fallback fires when the
    ///   scheduler's ops.select_cpu() fails to find a CPU. Sustained
    ///   200/s across all CPUs indicates systematic select_cpu failure.
    /// - max_keep_last_rate 100.0: dispatch_keep_last fires when a CPU
    ///   re-dispatches the previously running task because the scheduler
    ///   provided nothing. Sustained 100/s indicates dispatch starvation.
    pub const DEFAULT: MonitorThresholds = MonitorThresholds {
        max_imbalance_ratio: 4.0,
        max_local_dsq_depth: 50,
        fail_on_stall: true,
        sustained_samples: 5,
        max_fallback_rate: 200.0,
        max_keep_last_rate: 100.0,
    };
}

impl Default for MonitorThresholds {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Verdict from evaluating monitor data against thresholds.
#[derive(Debug, Clone)]
pub struct MonitorVerdict {
    /// `true` if all thresholds were met.
    pub passed: bool,
    /// Per-violation detail messages (empty when `passed` is true).
    pub details: Vec<String>,
    /// One-line summary: "monitor OK" or "monitor FAILED: N violation(s)".
    pub summary: String,
}

impl MonitorThresholds {
    /// Evaluate a MonitorReport against these thresholds.
    ///
    /// Returns a passing verdict when samples are empty or when the monitor
    /// data appears to be uninitialized guest memory (all rq_clocks identical
    /// across every CPU and sample, or DSQ depths above a plausibility
    /// ceiling). The monitor thread reads raw guest memory via BTF offsets;
    /// in short-lived VMs the kernel may not have populated the per-CPU
    /// runqueue structures before the monitor starts sampling.
    pub fn evaluate(&self, report: &MonitorReport) -> MonitorVerdict {
        let mut details = Vec::new();

        if report.samples.is_empty() {
            return MonitorVerdict {
                passed: true,
                details: vec![],
                summary: "no monitor samples".into(),
            };
        }

        // Validity check: detect uninitialized guest memory.
        // If all rq_clock values across every CPU in every sample are
        // identical, the kernel never wrote to these fields — the monitor
        // was reading zeroed or garbage memory.
        if !Self::data_looks_valid(&report.samples) {
            return MonitorVerdict {
                passed: true,
                details: vec![],
                summary: "monitor data not yet initialized".into(),
            };
        }

        let mut imbalance = SustainedViolationTracker::default();
        let mut dsq = SustainedViolationTracker::default();
        let mut worst_dsq_cpu = 0usize;

        for (i, sample) in report.samples.iter().enumerate() {
            if sample.cpus.is_empty() {
                imbalance.record(false, 0.0, i);
                dsq.record(false, 0.0, i);
                continue;
            }

            // Imbalance check.
            let ratio = sample.imbalance_ratio();
            imbalance.record(ratio > self.max_imbalance_ratio, ratio, i);

            // DSQ depth check.
            let mut dsq_violated = false;
            let mut sample_worst_depth = 0u32;
            let mut sample_worst_cpu = 0usize;
            for (cpu_idx, cpu) in sample.cpus.iter().enumerate() {
                if cpu.local_dsq_depth > self.max_local_dsq_depth
                    && cpu.local_dsq_depth > sample_worst_depth
                {
                    dsq_violated = true;
                    sample_worst_depth = cpu.local_dsq_depth;
                    sample_worst_cpu = cpu_idx;
                }
            }
            dsq.record(dsq_violated, sample_worst_depth as f64, i);
            if dsq_violated && dsq.worst_value == sample_worst_depth as f64 {
                worst_dsq_cpu = sample_worst_cpu;
            }
        }

        let mut failed = false;

        if imbalance.sustained(self.sustained_samples) {
            failed = true;
            details.push(format!(
                "imbalance ratio {:.1} exceeded threshold {:.1} for {} consecutive samples (ending at sample {})",
                imbalance.worst_value,
                self.max_imbalance_ratio,
                imbalance.worst_run,
                imbalance.worst_at,
            ));
        }

        if dsq.sustained(self.sustained_samples) {
            failed = true;
            details.push(format!(
                "local DSQ depth {} on cpu{} exceeded threshold {} for {} consecutive samples (ending at sample {})",
                dsq.worst_value as u32,
                worst_dsq_cpu,
                self.max_local_dsq_depth,
                dsq.worst_run,
                dsq.worst_at,
            ));
        }

        // Stall detection: any CPU whose rq_clock did not advance between
        // consecutive samples. Uses the sustained_samples window like
        // imbalance and DSQ checks. Exempt idle CPUs (NOHZ stopped the
        // tick so rq_clock legitimately doesn't advance) and preempted
        // vCPUs (host stole the core, so the vCPU couldn't tick the
        // clock). See `reader::is_cpu_stalled` for the predicate.
        if self.fail_on_stall {
            let threshold = if report.preemption_threshold_ns > 0 {
                report.preemption_threshold_ns
            } else {
                vcpu_preemption_threshold_ns(None)
            };

            let num_cpus = report
                .samples
                .iter()
                .map(|s| s.cpus.len())
                .max()
                .unwrap_or(0);
            let mut stall: Vec<SustainedViolationTracker> =
                vec![SustainedViolationTracker::default(); num_cpus];

            for i in 1..report.samples.len() {
                let prev = &report.samples[i - 1];
                let curr = &report.samples[i];
                let cpu_count = prev.cpus.len().min(curr.cpus.len());
                #[allow(clippy::needless_range_loop)]
                // indexes stall[cpu], prev.cpus[cpu], curr.cpus[cpu]
                for cpu in 0..cpu_count {
                    let is_stall =
                        reader::is_cpu_stalled(&prev.cpus[cpu], &curr.cpus[cpu], threshold);
                    stall[cpu].record(is_stall, curr.cpus[cpu].rq_clock as f64, i);
                }
            }

            #[allow(clippy::needless_range_loop)] // cpu index used in format string
            for cpu in 0..num_cpus {
                if stall[cpu].sustained(self.sustained_samples) {
                    failed = true;
                    details.push(format!(
                        "rq_clock stall on cpu{} for {} consecutive samples (ending at sample {}, clock={})",
                        cpu,
                        stall[cpu].worst_run,
                        stall[cpu].worst_at,
                        stall[cpu].worst_value as u64,
                    ));
                }
            }
        }

        // Event counter rate checks: compute per-sample-interval rates
        // and track sustained violations like imbalance.
        let mut fallback_rate = SustainedViolationTracker::default();
        let mut keep_last_rate = SustainedViolationTracker::default();

        for i in 1..report.samples.len() {
            let prev = &report.samples[i - 1];
            let curr = &report.samples[i];
            let interval_s = curr.elapsed_ms.saturating_sub(prev.elapsed_ms) as f64 / 1000.0;
            if interval_s <= 0.0 {
                fallback_rate.record(false, 0.0, i);
                keep_last_rate.record(false, 0.0, i);
                continue;
            }

            // Fallback rate.
            if let (Some(prev_fb), Some(curr_fb)) = (
                prev.sum_event_field(|e| e.select_cpu_fallback),
                curr.sum_event_field(|e| e.select_cpu_fallback),
            ) {
                let rate = (curr_fb - prev_fb) as f64 / interval_s;
                fallback_rate.record(rate > self.max_fallback_rate, rate, i);
            } else {
                fallback_rate.record(false, 0.0, i);
            }

            // Keep-last rate.
            if let (Some(prev_kl), Some(curr_kl)) = (
                prev.sum_event_field(|e| e.dispatch_keep_last),
                curr.sum_event_field(|e| e.dispatch_keep_last),
            ) {
                let rate = (curr_kl - prev_kl) as f64 / interval_s;
                keep_last_rate.record(rate > self.max_keep_last_rate, rate, i);
            } else {
                keep_last_rate.record(false, 0.0, i);
            }
        }

        if fallback_rate.sustained(self.sustained_samples) {
            failed = true;
            details.push(format!(
                "fallback rate {:.1}/s exceeded threshold {:.1}/s for {} consecutive intervals (ending at sample {})",
                fallback_rate.worst_value,
                self.max_fallback_rate,
                fallback_rate.worst_run,
                fallback_rate.worst_at,
            ));
        }

        if keep_last_rate.sustained(self.sustained_samples) {
            failed = true;
            details.push(format!(
                "keep_last rate {:.1}/s exceeded threshold {:.1}/s for {} consecutive intervals (ending at sample {})",
                keep_last_rate.worst_value,
                self.max_keep_last_rate,
                keep_last_rate.worst_run,
                keep_last_rate.worst_at,
            ));
        }

        let summary = if failed {
            format!("monitor FAILED: {} violation(s)", details.len())
        } else {
            "monitor OK".into()
        };

        MonitorVerdict {
            passed: !failed,
            details,
            summary,
        }
    }

    /// Check whether the monitor samples contain plausible data.
    ///
    /// Returns false when the data looks like uninitialized guest memory:
    /// - All rq_clock values across every CPU in every sample are identical
    ///   (the kernel never wrote to these fields).
    /// - Any local_dsq_depth exceeds a plausibility ceiling (real kernels
    ///   never queue millions of tasks on a single CPU's local DSQ).
    fn data_looks_valid(samples: &[MonitorSample]) -> bool {
        let mut first_clock: Option<u64> = None;
        let mut all_clocks_same = true;

        for sample in samples {
            if !sample_looks_valid(sample) {
                return false;
            }
            for cpu in &sample.cpus {
                match first_clock {
                    None => first_clock = Some(cpu.rq_clock),
                    Some(fc) => {
                        if cpu.rq_clock != fc {
                            all_clocks_same = false;
                        }
                    }
                }
            }
        }

        // If we saw at least 2 clock readings and they were all identical,
        // the data is uninitialized.
        if first_clock.is_some() && all_clocks_same {
            // Check we actually had multiple readings to compare.
            let total_readings: usize = samples.iter().map(|s| s.cpus.len()).sum();
            if total_readings > 1 {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- parse_config_hz / vcpu_preemption_threshold_ns tests --

    #[test]
    fn parse_config_hz_standard() {
        let config = "# comment\nCONFIG_HZ_1000=y\nCONFIG_HZ=1000\n";
        assert_eq!(parse_config_hz(config), Some(1000));
    }

    #[test]
    fn parse_config_hz_250() {
        let config = "CONFIG_HZ=250\n";
        assert_eq!(parse_config_hz(config), Some(250));
    }

    #[test]
    fn parse_config_hz_100() {
        let config = "CONFIG_HZ=100\n";
        assert_eq!(parse_config_hz(config), Some(100));
    }

    #[test]
    fn parse_config_hz_missing() {
        let config = "CONFIG_PREEMPT=y\nCONFIG_HZ_1000=y\n";
        assert_eq!(parse_config_hz(config), None);
    }

    #[test]
    fn parse_config_hz_garbage_value() {
        let config = "CONFIG_HZ=abc\n";
        assert_eq!(parse_config_hz(config), None);
    }

    #[test]
    fn parse_config_hz_whitespace() {
        let config = "  CONFIG_HZ=1000  \n";
        assert_eq!(parse_config_hz(config), Some(1000));
    }

    #[test]
    fn parse_config_hz_commented_out() {
        let config = "# CONFIG_HZ=1000\nCONFIG_HZ_1000=y\n";
        assert_eq!(parse_config_hz(config), None);
    }

    #[test]
    fn vcpu_threshold_reasonable_range() {
        // With no kernel path, falls back to host config or DEFAULT_HZ=250.
        // Threshold should be between 10ms (HZ=1000) and 100ms (HZ=100).
        let t = vcpu_preemption_threshold_ns(None);
        assert!(
            (10_000_000..=100_000_000).contains(&t),
            "threshold {t} ns outside expected range 10ms-100ms"
        );
    }

    #[test]
    fn vcpu_threshold_default_hz_fallback() {
        // Nonexistent kernel path -> falls back to host config or default.
        let t = vcpu_preemption_threshold_ns(Some(std::path::Path::new("/nonexistent/bzImage")));
        assert!(
            (10_000_000..=100_000_000).contains(&t),
            "fallback threshold {t} ns outside expected range"
        );
    }

    /// Regression for the "host config leaks into guest HZ" bug:
    /// when `kernel_path` is `Some`, `guest_kernel_hz` must not fall
    /// back to `/boot/config-$(uname -r)`. A cached/built guest
    /// kernel's HZ is independent of the host's HZ, so silently
    /// picking up host HZ would yield wrong tick-dependent thresholds
    /// on any mismatch.
    ///
    /// This test points `kernel_path` at a nonexistent file. The
    /// IKCONFIG and `.config` lookups both fail, and the function
    /// must return exactly [`DEFAULT_HZ`] — NOT whatever the host's
    /// `/boot/config` happens to contain.
    #[test]
    fn guest_kernel_hz_gated_on_kernel_path() {
        let bogus = std::path::Path::new("/nonexistent/ktstr-kernel/bzImage");
        let hz = guest_kernel_hz(Some(bogus));
        assert_eq!(
            hz, DEFAULT_HZ,
            "kernel_path=Some with no IKCONFIG/.config must fall back \
             to DEFAULT_HZ, not host /boot/config; got {hz}"
        );
    }

    /// Complement: with `kernel_path=None` (virtme-style run), the
    /// host config IS authoritative and may legitimately override
    /// `DEFAULT_HZ`. Verify the returned value is a plausible HZ
    /// value — i.e., the code path still works when we explicitly
    /// want host fallback.
    #[test]
    fn guest_kernel_hz_none_consults_host_config() {
        let hz = guest_kernel_hz(None);
        // Accept any known Linux HZ value (DEFAULT_HZ=250 is in this set).
        assert!(
            matches!(hz, 100 | 250 | 300 | 1000),
            "guest_kernel_hz(None) = {hz} outside plausible HZ set"
        );
    }

    // -- IKCONFIG extraction tests --

    /// Build a synthetic blob: padding + IKCFG_ST marker + gzip(config_text) + IKCFG_ED marker.
    fn make_ikconfig_blob(config_text: &str) -> Vec<u8> {
        use flate2::Compression;
        use flate2::write::GzEncoder;
        use std::io::Write;

        let mut blob = vec![0u8; 64]; // padding
        blob.extend_from_slice(IKCONFIG_MAGIC);
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(config_text.as_bytes()).unwrap();
        blob.extend(encoder.finish().unwrap());
        blob.extend_from_slice(b"IKCFG_ED");
        blob
    }

    #[test]
    fn ikconfig_extracts_hz_1000() {
        let blob = make_ikconfig_blob("CONFIG_HZ=1000\nCONFIG_PREEMPT=y\n");
        let dir = std::env::temp_dir().join("ktstr-ikconfig-test-1000");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("vmlinux");
        std::fs::write(&path, &blob).unwrap();
        assert_eq!(read_hz_from_ikconfig(&path), Some(1000));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ikconfig_extracts_hz_250() {
        let blob = make_ikconfig_blob("CONFIG_HZ=250\n");
        let dir = std::env::temp_dir().join("ktstr-ikconfig-test-250");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("vmlinux");
        std::fs::write(&path, &blob).unwrap();
        assert_eq!(read_hz_from_ikconfig(&path), Some(250));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ikconfig_no_marker_returns_none() {
        let dir = std::env::temp_dir().join("ktstr-ikconfig-test-none");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("vmlinux");
        std::fs::write(&path, b"no marker here").unwrap();
        assert_eq!(read_hz_from_ikconfig(&path), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ikconfig_missing_config_hz_returns_none() {
        let blob = make_ikconfig_blob("CONFIG_PREEMPT=y\n");
        let dir = std::env::temp_dir().join("ktstr-ikconfig-test-nohz");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("vmlinux");
        std::fs::write(&path, &blob).unwrap();
        assert_eq!(read_hz_from_ikconfig(&path), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_samples_default_summary() {
        let summary = MonitorSummary::from_samples(&[]);
        assert_eq!(summary.total_samples, 0);
        assert_eq!(summary.max_imbalance_ratio, 0.0);
        assert_eq!(summary.max_local_dsq_depth, 0);
        assert!(!summary.stall_detected);
        assert_eq!(summary.avg_imbalance_ratio, 0.0);
        assert_eq!(summary.avg_nr_running, 0.0);
        assert_eq!(summary.avg_local_dsq_depth, 0.0);
    }

    #[test]
    fn single_sample_imbalanced_cpus() {
        let sample = MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    local_dsq_depth: 3,
                    rq_clock: 1000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 4,
                    local_dsq_depth: 1,
                    rq_clock: 2000,
                    ..Default::default()
                },
            ],
        };
        let summary = MonitorSummary::from_samples(&[sample]);
        assert_eq!(summary.total_samples, 1);
        assert!((summary.max_imbalance_ratio - 4.0).abs() < f64::EPSILON);
        assert_eq!(summary.max_local_dsq_depth, 3);
        assert!(!summary.stall_detected);
        // avg fields: single sample with cpus [nr_running=1, nr_running=4]
        assert!((summary.avg_imbalance_ratio - 4.0).abs() < f64::EPSILON);
        assert!((summary.avg_nr_running - 2.5).abs() < f64::EPSILON);
        assert!((summary.avg_local_dsq_depth - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn stall_detected_when_clock_stuck() {
        let s1 = MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 6000,
                    ..Default::default()
                },
            ],
        };
        let s2 = MonitorSample {
            prog_stats: None,
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000, // stuck
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 7000,
                    ..Default::default()
                },
            ],
        };
        let summary = MonitorSummary::from_samples(&[s1, s2]);
        assert!(summary.stall_detected);
    }

    #[test]
    fn balanced_cpus_ratio_one() {
        let sample = MonitorSample {
            prog_stats: None,
            elapsed_ms: 50,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 3,
                    rq_clock: 100,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 3,
                    rq_clock: 200,
                    ..Default::default()
                },
            ],
        };
        let summary = MonitorSummary::from_samples(&[sample]);
        assert!((summary.max_imbalance_ratio - 1.0).abs() < f64::EPSILON);
        assert!(!summary.stall_detected);
        assert!((summary.avg_imbalance_ratio - 1.0).abs() < f64::EPSILON);
        assert!((summary.avg_nr_running - 3.0).abs() < f64::EPSILON);
        assert!((summary.avg_local_dsq_depth - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn single_cpu_no_division_by_zero() {
        let sample = MonitorSample {
            prog_stats: None,
            elapsed_ms: 10,
            cpus: vec![CpuSnapshot {
                nr_running: 5,
                local_dsq_depth: 2,
                rq_clock: 1000,
                ..Default::default()
            }],
        };
        let summary = MonitorSummary::from_samples(&[sample]);
        assert_eq!(summary.total_samples, 1);
        // Single CPU: min == max, ratio = 1.0
        assert!((summary.max_imbalance_ratio - 1.0).abs() < f64::EPSILON);
        assert_eq!(summary.max_local_dsq_depth, 2);
        assert!(!summary.stall_detected);
    }

    #[test]
    fn all_zero_snapshots() {
        let sample = MonitorSample {
            prog_stats: None,
            elapsed_ms: 0,
            cpus: vec![CpuSnapshot::default(), CpuSnapshot::default()],
        };
        let summary = MonitorSummary::from_samples(&[sample]);
        assert_eq!(summary.total_samples, 1);
        // nr_running=0 for all CPUs: max/max(min,1) = 0/1 = 0.0, but
        // initial max_imbalance_ratio is 1.0 and 0.0 < 1.0, so stays 1.0.
        assert!((summary.max_imbalance_ratio - 1.0).abs() < f64::EPSILON);
        assert_eq!(summary.max_local_dsq_depth, 0);
        // rq_clock=0 is excluded from stall detection
        assert!(!summary.stall_detected);
        // avg: valid sample with 2 all-zero CPUs
        assert_eq!(summary.avg_imbalance_ratio, 0.0);
        assert_eq!(summary.avg_nr_running, 0.0);
        assert_eq!(summary.avg_local_dsq_depth, 0.0);
    }

    #[test]
    fn empty_cpus_in_sample() {
        let sample = MonitorSample {
            prog_stats: None,
            elapsed_ms: 10,
            cpus: vec![],
        };
        let summary = MonitorSummary::from_samples(&[sample]);
        assert_eq!(summary.total_samples, 1);
        // Empty cpus slice is skipped via `continue`
        assert!((summary.max_imbalance_ratio - 1.0).abs() < f64::EPSILON);
        // avg: sample skipped (empty cpus), no valid readings
        assert_eq!(summary.avg_imbalance_ratio, 0.0);
        assert_eq!(summary.avg_nr_running, 0.0);
        assert_eq!(summary.avg_local_dsq_depth, 0.0);
    }

    #[test]
    fn min_nr_zero_division_guard() {
        // All CPUs have nr_running=0. The code uses min_nr.max(1) as
        // divisor, so ratio = 0/1 = 0.0, which is < initial 1.0.
        let sample = MonitorSample {
            prog_stats: None,
            elapsed_ms: 10,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 0,
                    rq_clock: 100,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 0,
                    rq_clock: 200,
                    ..Default::default()
                },
            ],
        };
        let summary = MonitorSummary::from_samples(&[sample]);
        // Should not panic from division by zero.
        // max_imbalance_ratio stays at initial 1.0 since 0/1=0 < 1.0.
        assert!((summary.max_imbalance_ratio - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn min_nr_zero_max_nr_nonzero() {
        // min_nr=0, max_nr=5: ratio = 5/max(0,1) = 5.0
        let sample = MonitorSample {
            prog_stats: None,
            elapsed_ms: 10,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 0,
                    rq_clock: 100,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 5,
                    rq_clock: 200,
                    ..Default::default()
                },
            ],
        };
        let summary = MonitorSummary::from_samples(&[sample]);
        assert!((summary.max_imbalance_ratio - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn advancing_clocks_no_stall() {
        let s1 = MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 1000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 2000,
                    ..Default::default()
                },
            ],
        };
        let s2 = MonitorSample {
            prog_stats: None,
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 1500,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 2500,
                    ..Default::default()
                },
            ],
        };
        let s3 = MonitorSample {
            prog_stats: None,
            elapsed_ms: 300,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 2000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 3000,
                    ..Default::default()
                },
            ],
        };
        let summary = MonitorSummary::from_samples(&[s1, s2, s3]);
        assert!(!summary.stall_detected);
        assert_eq!(summary.total_samples, 3);
    }

    #[test]
    fn different_length_cpu_vecs() {
        // First sample has 2 CPUs, second has 3. Stall detection uses
        // min(prev.len, curr.len) = 2, so only CPUs 0-1 are compared.
        let s1 = MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 1000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 2000,
                    ..Default::default()
                },
            ],
        };
        let s2 = MonitorSample {
            prog_stats: None,
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 1500,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 2500,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 3000,
                    ..Default::default()
                },
            ],
        };
        let summary = MonitorSummary::from_samples(&[s1, s2]);
        assert!(!summary.stall_detected);
        assert_eq!(summary.total_samples, 2);
        // max_local_dsq_depth comes from all CPUs in all samples.
        assert_eq!(summary.max_local_dsq_depth, 0);
    }

    // -- MonitorThresholds tests --

    fn balanced_sample(elapsed_ms: u64, clock_base: u64) -> MonitorSample {
        MonitorSample {
            prog_stats: None,
            elapsed_ms,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 2,
                    rq_clock: clock_base,
                    local_dsq_depth: 3,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 2,
                    rq_clock: clock_base + 100,
                    local_dsq_depth: 2,
                    ..Default::default()
                },
            ],
        }
    }

    #[test]
    fn thresholds_default_values() {
        // Regression guard for `MonitorThresholds::DEFAULT`. Every
        // field is asserted: changing a default silently shifts what
        // "passes by default" across every test that inherits
        // defaults via `Assert::default_checks()` + per-scheduler
        // merge. If a default moves, the rationale belongs in the
        // doc comment on `DEFAULT` first; the test failure then
        // prompts the rationale update.
        let t = MonitorThresholds::default();
        assert!(
            (t.max_imbalance_ratio - 4.0).abs() < f64::EPSILON,
            "default max_imbalance_ratio drifted: {}",
            t.max_imbalance_ratio,
        );
        assert_eq!(
            t.max_local_dsq_depth, 50,
            "default max_local_dsq_depth drifted",
        );
        assert!(t.fail_on_stall, "default fail_on_stall drifted");
        assert_eq!(t.sustained_samples, 5, "default sustained_samples drifted");
        assert!(
            (t.max_fallback_rate - 200.0).abs() < f64::EPSILON,
            "default max_fallback_rate drifted: {}",
            t.max_fallback_rate,
        );
        assert!(
            (t.max_keep_last_rate - 100.0).abs() < f64::EPSILON,
            "default max_keep_last_rate drifted: {}",
            t.max_keep_last_rate,
        );
    }

    #[test]
    fn thresholds_default_matches_const() {
        // `Default::default()` and `DEFAULT` must agree — the impl
        // forwards, but the forward is a single expression that a
        // drive-by refactor could break.
        let a = MonitorThresholds::default();
        let b = MonitorThresholds::DEFAULT;
        assert!((a.max_imbalance_ratio - b.max_imbalance_ratio).abs() < f64::EPSILON);
        assert_eq!(a.max_local_dsq_depth, b.max_local_dsq_depth);
        assert_eq!(a.fail_on_stall, b.fail_on_stall);
        assert_eq!(a.sustained_samples, b.sustained_samples);
        assert!((a.max_fallback_rate - b.max_fallback_rate).abs() < f64::EPSILON);
        assert!((a.max_keep_last_rate - b.max_keep_last_rate).abs() < f64::EPSILON);
    }

    #[test]
    fn thresholds_empty_report_passes() {
        let t = MonitorThresholds::default();
        let report = MonitorReport {
            samples: vec![],
            summary: MonitorSummary::default(),
            ..Default::default()
        };
        let v = t.evaluate(&report);
        assert!(v.passed);
        assert!(v.details.is_empty());
    }

    #[test]
    fn thresholds_balanced_samples_pass() {
        let t = MonitorThresholds::default();
        let samples: Vec<_> = (0..10)
            .map(|i| balanced_sample(i * 100, 1000 + i * 500))
            .collect();
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let v = t.evaluate(&report);
        assert!(v.passed, "balanced samples should pass: {:?}", v.details);
    }

    #[test]
    fn thresholds_imbalance_below_sustained_passes() {
        let t = MonitorThresholds {
            sustained_samples: 5,
            max_imbalance_ratio: 4.0,
            ..Default::default()
        };
        // 4 consecutive imbalanced samples (below sustained_samples=5).
        let mut samples = Vec::new();
        for i in 0..4 {
            samples.push(MonitorSample {
                prog_stats: None,
                elapsed_ms: i * 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 1000 + i * 500,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 10,
                        rq_clock: 1100 + i * 500,
                        ..Default::default()
                    },
                ],
            });
        }
        // Then a balanced one to break the streak.
        samples.push(balanced_sample(400, 3000));
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let v = t.evaluate(&report);
        assert!(
            v.passed,
            "4 imbalanced < sustained_samples=5: {:?}",
            v.details
        );
    }

    #[test]
    fn thresholds_imbalance_at_sustained_fails() {
        let t = MonitorThresholds {
            sustained_samples: 5,
            max_imbalance_ratio: 4.0,
            ..Default::default()
        };
        // 5 consecutive imbalanced samples (ratio=10, threshold=4).
        let mut samples = Vec::new();
        for i in 0..5u64 {
            samples.push(MonitorSample {
                prog_stats: None,
                elapsed_ms: i * 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 1000 + i * 500,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 10,
                        rq_clock: 1100 + i * 500,
                        ..Default::default()
                    },
                ],
            });
        }
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let v = t.evaluate(&report);
        assert!(!v.passed);
        assert!(v.details.iter().any(|d| d.contains("imbalance")));
    }

    #[test]
    fn thresholds_dsq_depth_sustained_fails() {
        let t = MonitorThresholds {
            sustained_samples: 3,
            max_local_dsq_depth: 10,
            fail_on_stall: false,
            ..Default::default()
        };
        let mut samples = Vec::new();
        for i in 0..3u64 {
            samples.push(MonitorSample {
                prog_stats: None,
                elapsed_ms: i * 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 2,
                        local_dsq_depth: 20,
                        rq_clock: 1000 + i * 500,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 2,
                        local_dsq_depth: 5,
                        rq_clock: 1100 + i * 500,
                        ..Default::default()
                    },
                ],
            });
        }
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let v = t.evaluate(&report);
        assert!(!v.passed);
        assert!(v.details.iter().any(|d| d.contains("DSQ depth")));
    }

    #[test]
    fn thresholds_dsq_depth_below_sustained_passes() {
        let t = MonitorThresholds {
            sustained_samples: 3,
            max_local_dsq_depth: 10,
            fail_on_stall: false,
            ..Default::default()
        };
        // Only 2 consecutive DSQ violations, then a clean sample.
        let mut samples = Vec::new();
        for i in 0..2u64 {
            samples.push(MonitorSample {
                prog_stats: None,
                elapsed_ms: i * 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 2,
                        local_dsq_depth: 20,
                        rq_clock: 1000 + i * 500,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 2,
                        local_dsq_depth: 5,
                        rq_clock: 1100 + i * 500,
                        ..Default::default()
                    },
                ],
            });
        }
        samples.push(balanced_sample(200, 2000));
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let v = t.evaluate(&report);
        assert!(v.passed, "2 DSQ violations < sustained=3: {:?}", v.details);
    }

    #[test]
    fn thresholds_stall_detected_fails() {
        // Stalls use the sustained_samples window. With sustained_samples=1,
        // a single stall pair triggers failure.
        let t = MonitorThresholds {
            fail_on_stall: true,
            sustained_samples: 1,
            ..Default::default()
        };
        let samples = vec![
            MonitorSample {
                prog_stats: None,
                elapsed_ms: 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 5000,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 6000,
                        ..Default::default()
                    },
                ],
            },
            MonitorSample {
                prog_stats: None,
                elapsed_ms: 200,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 5000,
                        ..Default::default()
                    }, // stuck
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 7000,
                        ..Default::default()
                    },
                ],
            },
        ];
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let v = t.evaluate(&report);
        assert!(!v.passed);
        assert!(v.details.iter().any(|d| d.contains("rq_clock stall")));
    }

    #[test]
    fn thresholds_stall_disabled_passes() {
        let t = MonitorThresholds {
            fail_on_stall: false,
            sustained_samples: 100,
            ..Default::default()
        };
        let samples = vec![
            MonitorSample {
                prog_stats: None,
                elapsed_ms: 100,
                cpus: vec![CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,
                    ..Default::default()
                }],
            },
            MonitorSample {
                prog_stats: None,
                elapsed_ms: 200,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 5000,
                        ..Default::default()
                    }, // stuck but stall check disabled
                ],
            },
        ];
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let v = t.evaluate(&report);
        assert!(v.passed, "stall disabled should pass: {:?}", v.details);
    }

    #[test]
    fn thresholds_imbalance_interrupted_by_balanced_resets() {
        // 3 imbalanced, 1 balanced, 3 imbalanced — never reaches sustained=5.
        let t = MonitorThresholds {
            sustained_samples: 5,
            max_imbalance_ratio: 4.0,
            fail_on_stall: false,
            ..Default::default()
        };
        let mut samples = Vec::new();
        for i in 0..3u64 {
            samples.push(MonitorSample {
                prog_stats: None,
                elapsed_ms: i * 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 1000 + i * 500,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 10,
                        rq_clock: 1100 + i * 500,
                        ..Default::default()
                    },
                ],
            });
        }
        samples.push(balanced_sample(300, 2500));
        for i in 4..7u64 {
            samples.push(MonitorSample {
                prog_stats: None,
                elapsed_ms: i * 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 3000 + i * 500,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 10,
                        rq_clock: 3100 + i * 500,
                        ..Default::default()
                    },
                ],
            });
        }
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let v = t.evaluate(&report);
        assert!(
            v.passed,
            "interrupted imbalance should pass: {:?}",
            v.details
        );
    }

    #[test]
    fn thresholds_multiple_violations() {
        // Both imbalance and stall in the same report. Both need to
        // reach sustained_samples to trigger. 3 samples = 2 consecutive
        // stall pairs for cpu0 (clock stuck at 1000), 2 consecutive
        // imbalance violations (ratio=5.0 > 2.0).
        let t = MonitorThresholds {
            sustained_samples: 2,
            max_imbalance_ratio: 2.0,
            fail_on_stall: true,
            ..Default::default()
        };
        let samples = vec![
            MonitorSample {
                prog_stats: None,
                elapsed_ms: 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 1000,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 5,
                        rq_clock: 2000,
                        ..Default::default()
                    },
                ],
            },
            MonitorSample {
                prog_stats: None,
                elapsed_ms: 200,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 1000,
                        ..Default::default()
                    }, // stall + imbalance
                    CpuSnapshot {
                        nr_running: 5,
                        rq_clock: 3000,
                        ..Default::default()
                    },
                ],
            },
            MonitorSample {
                prog_stats: None,
                elapsed_ms: 300,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 1000,
                        ..Default::default()
                    }, // stall continues
                    CpuSnapshot {
                        nr_running: 5,
                        rq_clock: 4000,
                        ..Default::default()
                    },
                ],
            },
        ];
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let v = t.evaluate(&report);
        assert!(!v.passed);
        assert!(v.details.iter().any(|d| d.contains("imbalance")));
        assert!(v.details.iter().any(|d| d.contains("rq_clock stall")));
    }

    #[test]
    fn thresholds_empty_cpus_samples_pass() {
        let t = MonitorThresholds::default();
        let samples = vec![
            MonitorSample {
                prog_stats: None,
                elapsed_ms: 100,
                cpus: vec![],
            },
            MonitorSample {
                prog_stats: None,
                elapsed_ms: 200,
                cpus: vec![],
            },
        ];
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let v = t.evaluate(&report);
        assert!(v.passed);
    }

    #[test]
    fn thresholds_uninitialized_memory_passes() {
        // Simulates what happens when monitor reads guest memory before
        // kernel initialization: all rq_clocks identical, DSQ depths garbage.
        let t = MonitorThresholds::default();
        let garbage_clock = 10314579376562252011u64;
        let samples: Vec<_> = (0..10)
            .map(|i| MonitorSample {
                prog_stats: None,
                elapsed_ms: i * 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 0,
                        rq_clock: garbage_clock,
                        local_dsq_depth: 1550435906,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 0,
                        rq_clock: garbage_clock,
                        local_dsq_depth: 1550435906,
                        ..Default::default()
                    },
                ],
            })
            .collect();
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let v = t.evaluate(&report);
        assert!(
            v.passed,
            "uninitialized guest memory should be skipped: {:?}",
            v.details
        );
    }

    #[test]
    fn thresholds_all_same_clocks_passes() {
        // All clocks identical across all CPUs and samples = uninitialized.
        let t = MonitorThresholds {
            fail_on_stall: true,
            ..Default::default()
        };
        let samples = vec![
            MonitorSample {
                prog_stats: None,
                elapsed_ms: 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 5000,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 5000,
                        ..Default::default()
                    },
                ],
            },
            MonitorSample {
                prog_stats: None,
                elapsed_ms: 200,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 5000,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 5000,
                        ..Default::default()
                    },
                ],
            },
        ];
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let v = t.evaluate(&report);
        assert!(
            v.passed,
            "all-same clocks should be treated as uninitialized: {:?}",
            v.details
        );
    }

    #[test]
    fn thresholds_dsq_over_plausibility_ceiling_passes() {
        let t = MonitorThresholds::default();
        let samples = vec![MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 1000,
                    local_dsq_depth: 50000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 2000,
                    local_dsq_depth: 5,
                    ..Default::default()
                },
            ],
        }];
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let v = t.evaluate(&report);
        assert!(
            v.passed,
            "implausible DSQ depth should skip evaluation: {:?}",
            v.details
        );
    }

    #[test]
    fn thresholds_single_cpu_single_sample_valid() {
        // A single reading cannot be compared, so all_clocks_same with
        // total_readings=1 should still be treated as valid.
        let t = MonitorThresholds {
            fail_on_stall: true,
            sustained_samples: 1,
            ..Default::default()
        };
        let samples = vec![MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![CpuSnapshot {
                nr_running: 1,
                rq_clock: 5000,
                ..Default::default()
            }],
        }];
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let v = t.evaluate(&report);
        assert!(v.passed, "single reading should be valid: {:?}", v.details);
    }

    // -- Event counter rate threshold tests --

    /// Build a sample with event counters. Each CPU gets the same counter
    /// values so the total across CPUs = ncpus * per_cpu_value.
    fn sample_with_events(
        elapsed_ms: u64,
        clock_base: u64,
        fallback: i64,
        keep_last: i64,
    ) -> MonitorSample {
        MonitorSample {
            prog_stats: None,
            elapsed_ms,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 2,
                    rq_clock: clock_base,
                    event_counters: Some(ScxEventCounters {
                        select_cpu_fallback: fallback,
                        dispatch_keep_last: keep_last,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 2,
                    rq_clock: clock_base + 100,
                    event_counters: Some(ScxEventCounters {
                        select_cpu_fallback: fallback,
                        dispatch_keep_last: keep_last,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ],
        }
    }

    #[test]
    fn thresholds_fallback_rate_sustained_fails() {
        // sustained_samples=3, max_fallback_rate=10.0.
        // 100ms intervals, 2 CPUs. Each CPU increments fallback by 10
        // per sample -> delta = 20 total per interval / 0.1s = 200/s > 10.
        let t = MonitorThresholds {
            sustained_samples: 3,
            max_fallback_rate: 10.0,
            fail_on_stall: false,
            ..Default::default()
        };
        let samples: Vec<_> = (0..4)
            .map(|i| sample_with_events(i * 100, 1000 + i * 500, i as i64 * 10, 0))
            .collect();
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let v = t.evaluate(&report);
        assert!(!v.passed);
        assert!(v.details.iter().any(|d| d.contains("fallback rate")));
    }

    #[test]
    fn thresholds_fallback_rate_below_sustained_passes() {
        // 2 violating intervals then a clean one — below sustained=3.
        let t = MonitorThresholds {
            sustained_samples: 3,
            max_fallback_rate: 10.0,
            fail_on_stall: false,
            ..Default::default()
        };
        let mut samples: Vec<_> = (0..3)
            .map(|i| sample_with_events(i * 100, 1000 + i * 500, i as i64 * 10, 0))
            .collect();
        // 4th sample: same fallback as 3rd -> rate = 0.
        samples.push(sample_with_events(300, 2500, 20, 0));
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let v = t.evaluate(&report);
        assert!(v.passed, "2 violations < sustained=3: {:?}", v.details);
    }

    #[test]
    fn thresholds_keep_last_rate_sustained_fails() {
        let t = MonitorThresholds {
            sustained_samples: 3,
            max_keep_last_rate: 10.0,
            fail_on_stall: false,
            ..Default::default()
        };
        let samples: Vec<_> = (0..4)
            .map(|i| sample_with_events(i * 100, 1000 + i * 500, 0, i as i64 * 10))
            .collect();
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let v = t.evaluate(&report);
        assert!(!v.passed);
        assert!(v.details.iter().any(|d| d.contains("keep_last rate")));
    }

    #[test]
    fn thresholds_keep_last_rate_below_sustained_passes() {
        let t = MonitorThresholds {
            sustained_samples: 3,
            max_keep_last_rate: 10.0,
            fail_on_stall: false,
            ..Default::default()
        };
        let mut samples: Vec<_> = (0..3)
            .map(|i| sample_with_events(i * 100, 1000 + i * 500, 0, i as i64 * 10))
            .collect();
        // Reset: same keep_last as previous -> rate = 0.
        samples.push(sample_with_events(300, 2500, 0, 20));
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let v = t.evaluate(&report);
        assert!(v.passed, "2 violations < sustained=3: {:?}", v.details);
    }

    #[test]
    fn thresholds_event_rate_interrupted_resets() {
        // 2 violating intervals, 1 clean, 2 violating — never reaches sustained=3.
        let t = MonitorThresholds {
            sustained_samples: 3,
            max_fallback_rate: 10.0,
            fail_on_stall: false,
            ..Default::default()
        };
        let mut samples = Vec::new();
        // 3 samples = 2 intervals of high fallback rate.
        for i in 0..3u64 {
            samples.push(sample_with_events(
                i * 100,
                1000 + i * 500,
                i as i64 * 10,
                0,
            ));
        }
        // Clean interval: same fallback -> rate = 0.
        samples.push(sample_with_events(300, 2500, 20, 0));
        // 3 more samples = 2 intervals of high fallback rate (not 3).
        // The fallback delta for the first interval covers sample 3->4,
        // which is (30-20)/0.1 = 100/s (violating), then 4->5 is also
        // violating. That's 2 intervals, below sustained=3.
        for i in 0..2u64 {
            samples.push(sample_with_events(
                400 + i * 100,
                3000 + i * 500,
                30 + (i + 1) as i64 * 10,
                0,
            ));
        }
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let v = t.evaluate(&report);
        assert!(
            v.passed,
            "interrupted rate violations should pass: {:?}",
            v.details
        );
    }

    #[test]
    fn thresholds_no_event_counters_skips_rate_check() {
        // Samples without event counters should not trigger rate violations.
        let t = MonitorThresholds {
            sustained_samples: 1,
            max_fallback_rate: 0.0, // any rate would fail
            max_keep_last_rate: 0.0,
            fail_on_stall: false,
            ..Default::default()
        };
        let samples: Vec<_> = (0..5)
            .map(|i| balanced_sample(i * 100, 1000 + i * 500))
            .collect();
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let v = t.evaluate(&report);
        assert!(
            v.passed,
            "no event counters should skip rate check: {:?}",
            v.details
        );
    }

    #[test]
    fn thresholds_default_event_rate_values() {
        let t = MonitorThresholds::default();
        assert!((t.max_fallback_rate - 200.0).abs() < f64::EPSILON);
        assert!((t.max_keep_last_rate - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn summary_keep_last_rate_computed() {
        // 2 CPUs, each with keep_last incrementing by 5 per sample.
        // 3 samples over 200ms -> total delta = 2*10 = 20, rate = 20/0.2 = 100.
        let samples = vec![
            sample_with_events(0, 1000, 0, 0),
            sample_with_events(100, 1500, 0, 5),
            sample_with_events(200, 2000, 0, 10),
        ];
        let summary = MonitorSummary::from_samples(&samples);
        let deltas = summary.event_deltas.unwrap();
        assert!((deltas.keep_last_rate - 100.0).abs() < f64::EPSILON);
    }

    // -- compute_event_deltas edge cases --

    #[test]
    fn event_deltas_none_without_counters() {
        let samples = vec![balanced_sample(100, 1000), balanced_sample(200, 1500)];
        let summary = MonitorSummary::from_samples(&samples);
        assert!(summary.event_deltas.is_none());
    }

    #[test]
    fn event_deltas_single_sample() {
        // Only one sample with events -> first == last, duration=0, rates=0.
        let samples = vec![sample_with_events(100, 1000, 50, 25)];
        let summary = MonitorSummary::from_samples(&samples);
        let deltas = summary.event_deltas.unwrap();
        assert_eq!(deltas.fallback_rate, 0.0);
        assert_eq!(deltas.keep_last_rate, 0.0);
    }

    #[test]
    fn event_deltas_max_fallback_burst() {
        // 3 samples: burst between samples 1 and 2.
        let samples = vec![
            sample_with_events(0, 1000, 0, 0),
            sample_with_events(100, 1500, 5, 0),
            sample_with_events(200, 2000, 100, 0),
        ];
        let summary = MonitorSummary::from_samples(&samples);
        let deltas = summary.event_deltas.unwrap();
        // Per-CPU: burst is (100-5)*2 = 190 across 2 CPUs.
        assert!(deltas.max_fallback_burst > 0);
    }

    #[test]
    fn event_deltas_counter_reset_clamps_to_zero() {
        // A scheduler restart between samples resets the per-CPU
        // counters to smaller (or zero) values. The raw delta
        // `last - first` is then negative — which would flow through
        // as a negative fallback_rate / negative total. Clamp to zero
        // so the downstream rate is sane.
        //
        // Sample 0 at t=0ms has high counters (pre-restart).
        // Sample 1 at t=1000ms has low counters (post-restart).
        let samples = vec![
            sample_with_events(0, 1000, 1000, 500),
            sample_with_events(1000, 2000, 5, 2),
        ];
        let summary = MonitorSummary::from_samples(&samples);
        let deltas = summary.event_deltas.unwrap();
        assert!(
            deltas.total_fallback >= 0,
            "reset must not produce negative total_fallback, got {}",
            deltas.total_fallback
        );
        assert!(
            deltas.fallback_rate >= 0.0,
            "reset must not produce negative fallback_rate, got {}",
            deltas.fallback_rate
        );
        assert!(
            deltas.total_dispatch_keep_last >= 0,
            "reset must not produce negative keep_last total, got {}",
            deltas.total_dispatch_keep_last
        );
        assert!(
            deltas.keep_last_rate >= 0.0,
            "reset must not produce negative keep_last_rate, got {}",
            deltas.keep_last_rate
        );
    }

    #[test]
    fn event_deltas_all_counters_computed() {
        let make = |elapsed_ms, fb, kl, dsq_off, exit, migdis| MonitorSample {
            prog_stats: None,
            elapsed_ms,
            cpus: vec![CpuSnapshot {
                nr_running: 1,
                rq_clock: elapsed_ms * 10,
                event_counters: Some(ScxEventCounters {
                    select_cpu_fallback: fb,
                    dispatch_local_dsq_offline: dsq_off,
                    dispatch_keep_last: kl,
                    enq_skip_exiting: exit,
                    enq_skip_migration_disabled: migdis,
                    ..Default::default()
                }),
                ..Default::default()
            }],
        };
        let samples = vec![
            make(100, 10, 20, 30, 40, 50),
            make(200, 110, 120, 130, 140, 150),
        ];
        let summary = MonitorSummary::from_samples(&samples);
        let d = summary.event_deltas.unwrap();
        assert_eq!(d.total_fallback, 100);
        assert_eq!(d.total_dispatch_keep_last, 100);
        assert_eq!(d.total_dispatch_offline, 100);
        assert_eq!(d.total_enq_skip_exiting, 100);
        assert_eq!(d.total_enq_skip_migration_disabled, 100);
    }

    // -- data_looks_valid tests --

    #[test]
    fn data_looks_valid_empty() {
        assert!(MonitorThresholds::data_looks_valid(&[]));
    }

    #[test]
    fn data_looks_valid_normal() {
        let samples = vec![balanced_sample(100, 1000), balanced_sample(200, 2000)];
        assert!(MonitorThresholds::data_looks_valid(&samples));
    }

    #[test]
    fn data_looks_valid_all_same_clocks() {
        let samples = vec![
            MonitorSample {
                prog_stats: None,
                elapsed_ms: 100,
                cpus: vec![
                    CpuSnapshot {
                        rq_clock: 5000,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        rq_clock: 5000,
                        ..Default::default()
                    },
                ],
            },
            MonitorSample {
                prog_stats: None,
                elapsed_ms: 200,
                cpus: vec![
                    CpuSnapshot {
                        rq_clock: 5000,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        rq_clock: 5000,
                        ..Default::default()
                    },
                ],
            },
        ];
        assert!(!MonitorThresholds::data_looks_valid(&samples));
    }

    #[test]
    fn data_looks_valid_dsq_over_ceiling() {
        let samples = vec![MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![CpuSnapshot {
                local_dsq_depth: 50000,
                rq_clock: 1000,
                ..Default::default()
            }],
        }];
        assert!(!MonitorThresholds::data_looks_valid(&samples));
    }

    // -- MonitorSample::imbalance_ratio tests --

    #[test]
    fn imbalance_ratio_empty_cpus() {
        let s = MonitorSample {
            prog_stats: None,
            elapsed_ms: 0,
            cpus: vec![],
        };
        assert!((s.imbalance_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn imbalance_ratio_single_cpu() {
        let s = MonitorSample {
            prog_stats: None,
            elapsed_ms: 0,
            cpus: vec![CpuSnapshot {
                nr_running: 5,
                ..Default::default()
            }],
        };
        assert!((s.imbalance_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn imbalance_ratio_balanced() {
        let s = MonitorSample {
            prog_stats: None,
            elapsed_ms: 0,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 3,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 3,
                    ..Default::default()
                },
            ],
        };
        assert!((s.imbalance_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn imbalance_ratio_imbalanced() {
        let s = MonitorSample {
            prog_stats: None,
            elapsed_ms: 0,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 2,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 8,
                    ..Default::default()
                },
            ],
        };
        assert!((s.imbalance_ratio() - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn imbalance_ratio_zero_min() {
        let s = MonitorSample {
            prog_stats: None,
            elapsed_ms: 0,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 0,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 5,
                    ..Default::default()
                },
            ],
        };
        // min=0, max(0,1)=1, ratio=5/1=5.0
        assert!((s.imbalance_ratio() - 5.0).abs() < f64::EPSILON);
    }

    // -- MonitorSample::sum_event_field tests --

    #[test]
    fn sum_event_field_none_when_no_counters() {
        let s = MonitorSample {
            prog_stats: None,
            elapsed_ms: 0,
            cpus: vec![CpuSnapshot::default(), CpuSnapshot::default()],
        };
        assert!(s.sum_event_field(|e| e.select_cpu_fallback).is_none());
    }

    #[test]
    fn sum_event_field_sums_across_cpus() {
        let s = MonitorSample {
            prog_stats: None,
            elapsed_ms: 0,
            cpus: vec![
                CpuSnapshot {
                    event_counters: Some(ScxEventCounters {
                        select_cpu_fallback: 10,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                CpuSnapshot {
                    event_counters: Some(ScxEventCounters {
                        select_cpu_fallback: 20,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ],
        };
        assert_eq!(s.sum_event_field(|e| e.select_cpu_fallback), Some(30));
    }

    #[test]
    fn sum_event_field_mixed_some_none() {
        let s = MonitorSample {
            prog_stats: None,
            elapsed_ms: 0,
            cpus: vec![
                CpuSnapshot {
                    event_counters: Some(ScxEventCounters {
                        dispatch_keep_last: 7,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                CpuSnapshot::default(),
            ],
        };
        assert_eq!(s.sum_event_field(|e| e.dispatch_keep_last), Some(7));
    }

    // -- sample_looks_valid tests --

    #[test]
    fn sample_looks_valid_normal() {
        let s = MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![CpuSnapshot {
                local_dsq_depth: 5,
                ..Default::default()
            }],
        };
        assert!(sample_looks_valid(&s));
    }

    #[test]
    fn sample_looks_valid_at_ceiling() {
        let s = MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![CpuSnapshot {
                local_dsq_depth: DSQ_PLAUSIBILITY_CEILING,
                ..Default::default()
            }],
        };
        assert!(sample_looks_valid(&s));
    }

    #[test]
    fn sample_looks_valid_over_ceiling() {
        let s = MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![CpuSnapshot {
                local_dsq_depth: DSQ_PLAUSIBILITY_CEILING + 1,
                ..Default::default()
            }],
        };
        assert!(!sample_looks_valid(&s));
    }

    #[test]
    fn sample_looks_valid_empty_cpus() {
        let s = MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![],
        };
        assert!(sample_looks_valid(&s));
    }

    #[test]
    fn sample_looks_valid_zero_initialized() {
        let s = MonitorSample {
            prog_stats: None,
            elapsed_ms: 0,
            cpus: vec![CpuSnapshot::default(), CpuSnapshot::default()],
        };
        // All fields zero, local_dsq_depth=0 <= DSQ_PLAUSIBILITY_CEILING.
        assert!(sample_looks_valid(&s));
    }

    #[test]
    fn sample_looks_valid_multiple_cpus_one_over() {
        let s = MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    local_dsq_depth: 5,
                    ..Default::default()
                },
                CpuSnapshot {
                    local_dsq_depth: DSQ_PLAUSIBILITY_CEILING + 1,
                    ..Default::default()
                },
            ],
        };
        // One CPU over ceiling invalidates the entire sample.
        assert!(!sample_looks_valid(&s));
    }

    // -- MonitorSummary field value assertions --

    #[test]
    fn from_samples_fields_sane_values() {
        let samples: Vec<_> = (0..5u64)
            .map(|i| MonitorSample {
                prog_stats: None,
                elapsed_ms: i * 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: (i as u32 + 1),
                        scx_nr_running: i as u32,
                        local_dsq_depth: (i as u32) % 3,
                        rq_clock: 1000 + i * 500,
                        scx_flags: 0,
                        event_counters: Some(ScxEventCounters {
                            select_cpu_fallback: i as i64 * 2,
                            dispatch_keep_last: i as i64,
                            ..Default::default()
                        }),
                        schedstat: None,
                        vcpu_cpu_time_ns: None,
                        sched_domains: None,
                    },
                    CpuSnapshot {
                        nr_running: (i as u32 + 2),
                        scx_nr_running: i as u32 + 1,
                        local_dsq_depth: 0,
                        rq_clock: 1100 + i * 600,
                        scx_flags: 0,
                        event_counters: Some(ScxEventCounters {
                            select_cpu_fallback: i as i64 * 3,
                            dispatch_keep_last: i as i64 * 2,
                            ..Default::default()
                        }),
                        schedstat: None,
                        vcpu_cpu_time_ns: None,
                        sched_domains: None,
                    },
                ],
            })
            .collect();
        let summary = MonitorSummary::from_samples(&samples);
        // total_samples matches input count
        assert_eq!(summary.total_samples, 5);
        // max_imbalance_ratio: all samples have nr_running differing by 1,
        // worst case is sample 0: nr_running=[1,2] -> ratio=2.0
        assert!(
            summary.max_imbalance_ratio >= 1.0,
            "ratio must be >= 1.0: {}",
            summary.max_imbalance_ratio
        );
        assert!(
            summary.max_imbalance_ratio <= 10.0,
            "ratio must be reasonable: {}",
            summary.max_imbalance_ratio
        );
        // max_local_dsq_depth: worst is (4 % 3) = 1 on cpu0 at i=4, or (3 % 3)=0 at i=3, (2%3)=2 at i=2
        assert!(
            summary.max_local_dsq_depth <= DSQ_PLAUSIBILITY_CEILING,
            "dsq depth must be below plausibility ceiling: {}",
            summary.max_local_dsq_depth
        );
        assert!(
            summary.max_local_dsq_depth <= 10,
            "dsq depth must be small in this controlled test: {}",
            summary.max_local_dsq_depth
        );
        // stall_detected: rq_clock advances each sample, so no stall
        assert!(
            !summary.stall_detected,
            "no stall expected with advancing rq_clock"
        );
        // event_deltas: should be computed
        let deltas = summary
            .event_deltas
            .as_ref()
            .expect("event deltas must be present");
        assert!(
            deltas.total_fallback >= 0,
            "fallback count must be non-negative"
        );
        assert!(
            deltas.total_dispatch_keep_last >= 0,
            "keep_last count must be non-negative"
        );
        assert!(
            deltas.fallback_rate >= 0.0,
            "fallback rate must be non-negative"
        );
        assert!(
            deltas.keep_last_rate >= 0.0,
            "keep_last rate must be non-negative"
        );
        // avg fields: must be positive with non-zero nr_running input
        assert!(
            summary.avg_imbalance_ratio >= 1.0,
            "avg imbalance must be >= 1.0: {}",
            summary.avg_imbalance_ratio,
        );
        assert!(
            summary.avg_nr_running > 0.0,
            "avg nr_running must be positive: {}",
            summary.avg_nr_running,
        );
        assert!(
            summary.avg_local_dsq_depth >= 0.0,
            "avg dsq_depth must be non-negative: {}",
            summary.avg_local_dsq_depth,
        );
    }

    #[test]
    fn from_samples_empty_all_defaults() {
        // Verify every field of MonitorSummary defaults correctly for empty input,
        // including event_deltas which empty_samples_default_summary does not check.
        let summary = MonitorSummary::from_samples(&[]);
        assert_eq!(summary.total_samples, 0);
        assert_eq!(summary.max_imbalance_ratio, 0.0);
        assert_eq!(summary.max_local_dsq_depth, 0);
        assert!(!summary.stall_detected);
        assert_eq!(summary.avg_imbalance_ratio, 0.0);
        assert_eq!(summary.avg_nr_running, 0.0);
        assert_eq!(summary.avg_local_dsq_depth, 0.0);
        assert!(
            summary.event_deltas.is_none(),
            "empty input must not produce event deltas"
        );
    }

    // ---------------------------------------------------------------
    // Negative tests: verify monitor diagnostics catch controlled failures
    // ---------------------------------------------------------------

    #[test]
    fn neg_tight_imbalance_threshold_catches_mild_imbalance() {
        let t = MonitorThresholds {
            max_imbalance_ratio: 1.0,
            sustained_samples: 2,
            fail_on_stall: false,
            ..Default::default()
        };
        let samples: Vec<_> = (0..3u64)
            .map(|i| MonitorSample {
                prog_stats: None,
                elapsed_ms: i * 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 2,
                        rq_clock: 1000 + i * 500,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 3,
                        rq_clock: 1100 + i * 500,
                        ..Default::default()
                    },
                ],
            })
            .collect();
        let summary = MonitorSummary::from_samples(&samples);
        assert!(
            summary.max_imbalance_ratio >= 1.5,
            "summary must capture ratio"
        );
        assert!(!summary.stall_detected, "no stall in this scenario");
        assert_eq!(summary.total_samples, 3);
        let report = MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let v = t.evaluate(&report);
        assert!(!v.passed, "imbalance=1.5 must fail threshold=1.0");
        // Format: "imbalance ratio 1.5 exceeded threshold 1.0 for 2 consecutive samples (ending at sample 2)"
        let detail = v.details.iter().find(|d| d.contains("imbalance")).unwrap();
        assert!(detail.contains("ratio"), "must include 'ratio': {detail}");
        assert!(
            detail.contains("exceeded threshold"),
            "must include threshold: {detail}"
        );
        assert!(
            detail.contains("1.0"),
            "must show threshold value: {detail}"
        );
        assert!(
            detail.contains("consecutive samples"),
            "must show sustained count: {detail}"
        );
        assert!(
            detail.contains("ending at sample"),
            "must show sample index: {detail}"
        );
        assert!(
            v.summary.contains("FAILED"),
            "summary must say FAILED: {}",
            v.summary
        );
    }

    #[test]
    fn neg_tight_dsq_threshold_catches_small_depth() {
        let t = MonitorThresholds {
            max_local_dsq_depth: 1,
            sustained_samples: 2,
            fail_on_stall: false,
            ..Default::default()
        };
        let samples: Vec<_> = (0..3u64)
            .map(|i| MonitorSample {
                prog_stats: None,
                elapsed_ms: i * 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        local_dsq_depth: 3,
                        rq_clock: 1000 + i * 500,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 1,
                        local_dsq_depth: 0,
                        rq_clock: 1100 + i * 500,
                        ..Default::default()
                    },
                ],
            })
            .collect();
        let summary = MonitorSummary::from_samples(&samples);
        assert_eq!(
            summary.max_local_dsq_depth, 3,
            "summary must capture max depth"
        );
        assert!(
            summary.max_local_dsq_depth <= DSQ_PLAUSIBILITY_CEILING,
            "depth must be plausible"
        );
        let report = MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let v = t.evaluate(&report);
        assert!(!v.passed, "dsq_depth=3 must fail threshold=1");
        // Format: "local DSQ depth 3 on cpu0 exceeded threshold 1 for 2 consecutive samples (ending at sample 2)"
        let detail = v.details.iter().find(|d| d.contains("DSQ depth")).unwrap();
        assert!(detail.contains("3"), "must show depth value: {detail}");
        assert!(detail.contains("cpu0"), "must show CPU number: {detail}");
        assert!(
            detail.contains("threshold 1"),
            "must show threshold: {detail}"
        );
        assert!(
            detail.contains("consecutive samples"),
            "must show count: {detail}"
        );
    }

    #[test]
    fn neg_stall_detection_catches_frozen_rq_clock() {
        // Stalls use sustained_samples window. sustained_samples=1 means
        // a single stall pair triggers failure.
        let t = MonitorThresholds {
            fail_on_stall: true,
            sustained_samples: 1,
            ..Default::default()
        };
        let samples = vec![
            MonitorSample {
                prog_stats: None,
                elapsed_ms: 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 5000,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 6000,
                        ..Default::default()
                    },
                ],
            },
            MonitorSample {
                prog_stats: None,
                elapsed_ms: 200,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 5000,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 7000,
                        ..Default::default()
                    },
                ],
            },
        ];
        let summary = MonitorSummary::from_samples(&samples);
        assert!(
            summary.stall_detected,
            "summary.stall_detected must be true"
        );
        let report = MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let v = t.evaluate(&report);
        assert!(!v.passed, "frozen rq_clock must be detected");
        let detail = v
            .details
            .iter()
            .find(|d| d.contains("rq_clock stall"))
            .unwrap();
        assert!(detail.contains("cpu0"), "must name frozen CPU: {detail}");
        assert!(
            detail.contains("consecutive samples"),
            "must show sustained count: {detail}"
        );
        assert!(
            detail.contains("clock=5000"),
            "must include frozen clock value: {detail}"
        );
    }

    #[test]
    fn neg_combined_imbalance_and_stall_both_reported() {
        let t = MonitorThresholds {
            max_imbalance_ratio: 2.0,
            sustained_samples: 1,
            fail_on_stall: true,
            ..Default::default()
        };
        let samples = vec![
            MonitorSample {
                prog_stats: None,
                elapsed_ms: 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 1000,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 10,
                        rq_clock: 2000,
                        ..Default::default()
                    },
                ],
            },
            MonitorSample {
                prog_stats: None,
                elapsed_ms: 200,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 1000,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 10,
                        rq_clock: 3000,
                        ..Default::default()
                    },
                ],
            },
        ];
        let summary = MonitorSummary::from_samples(&samples);
        assert!(summary.stall_detected);
        assert!(summary.max_imbalance_ratio >= 10.0);
        let report = MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let v = t.evaluate(&report);
        assert!(!v.passed);
        let imb = v.details.iter().find(|d| d.contains("imbalance")).unwrap();
        assert!(
            imb.contains("exceeded threshold 2.0"),
            "imbalance format: {imb}"
        );
        let stall = v
            .details
            .iter()
            .find(|d| d.contains("rq_clock stall"))
            .unwrap();
        assert!(stall.contains("cpu0"), "stall format: {stall}");
        assert!(
            v.details.len() >= 2,
            "both violations must be reported, got {}",
            v.details.len()
        );
        assert!(v.summary.contains("FAILED"), "summary: {}", v.summary);
    }

    #[test]
    fn stall_idle_cpu_exempt() {
        // nr_running==0 on both samples: idle CPU, NOHZ tick stopped.
        // rq_clock not advancing is expected, not a stall.
        let t = MonitorThresholds {
            fail_on_stall: true,
            sustained_samples: 1,
            ..Default::default()
        };
        let samples = vec![
            MonitorSample {
                prog_stats: None,
                elapsed_ms: 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 0,
                        rq_clock: 5000,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 6000,
                        ..Default::default()
                    },
                ],
            },
            MonitorSample {
                prog_stats: None,
                elapsed_ms: 200,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 0,
                        rq_clock: 5000, // stuck but idle
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 7000,
                        ..Default::default()
                    },
                ],
            },
        ];
        let summary = MonitorSummary::from_samples(&samples);
        assert!(
            !summary.stall_detected,
            "idle CPU should not trigger stall in summary"
        );
        let report = MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let v = t.evaluate(&report);
        assert!(
            v.passed,
            "idle CPU should not trigger stall: {:?}",
            v.details
        );
    }

    #[test]
    fn stall_idle_to_busy_not_exempt() {
        // nr_running transitions from 0 to 1 — the CPU woke up but
        // rq_clock didn't advance. This IS a stall (the CPU is now
        // busy but the scheduler tick hasn't fired).
        // Second CPU has a different clock value so data_looks_valid passes.
        let t = MonitorThresholds {
            fail_on_stall: true,
            sustained_samples: 1,
            ..Default::default()
        };
        let samples = vec![
            MonitorSample {
                prog_stats: None,
                elapsed_ms: 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 0,
                        rq_clock: 5000,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 6000,
                        ..Default::default()
                    },
                ],
            },
            MonitorSample {
                prog_stats: None,
                elapsed_ms: 200,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 5000, // stuck, but now busy
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 7000,
                        ..Default::default()
                    },
                ],
            },
        ];
        let summary = MonitorSummary::from_samples(&samples);
        assert!(
            summary.stall_detected,
            "busy CPU with frozen clock is a stall"
        );
        let report = MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let v = t.evaluate(&report);
        assert!(
            !v.passed,
            "busy CPU with frozen clock must fail: {:?}",
            v.details
        );
    }

    #[test]
    fn stall_sustained_window_filters_transient() {
        // With sustained_samples=3, a 2-sample stall doesn't trigger.
        // Second CPU has a different clock value so data_looks_valid passes.
        let t = MonitorThresholds {
            fail_on_stall: true,
            sustained_samples: 3,
            ..Default::default()
        };
        let mut samples = Vec::new();
        // 3 samples: 2 consecutive stall pairs for cpu0, then clock advances.
        for i in 0..3u64 {
            samples.push(MonitorSample {
                prog_stats: None,
                elapsed_ms: i * 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 5000, // stuck for all 3
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 6000 + i * 500, // advancing
                        ..Default::default()
                    },
                ],
            });
        }
        // Break the streak: clock advances in 4th sample.
        samples.push(MonitorSample {
            prog_stats: None,
            elapsed_ms: 300,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 6000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 7500,
                    ..Default::default()
                },
            ],
        });
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let v = t.evaluate(&report);
        // 2 consecutive stall pairs < sustained_samples=3
        assert!(v.passed, "2 stall pairs < sustained=3: {:?}", v.details);
    }

    #[test]
    fn stall_sustained_window_catches_real_stall() {
        // With sustained_samples=3, 3+ consecutive stall pairs trigger.
        // Second CPU has a different clock value so data_looks_valid passes.
        let t = MonitorThresholds {
            fail_on_stall: true,
            sustained_samples: 3,
            ..Default::default()
        };
        // 4 samples = 3 consecutive stall pairs for cpu0. cpu1 advances.
        let samples: Vec<_> = (0..4u64)
            .map(|i| MonitorSample {
                prog_stats: None,
                elapsed_ms: i * 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 5000, // stuck
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 6000 + i * 500, // advancing
                        ..Default::default()
                    },
                ],
            })
            .collect();
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let v = t.evaluate(&report);
        assert!(!v.passed, "3 consecutive stall pairs must fail");
        assert!(v.details.iter().any(|d| d.contains("rq_clock stall")));
    }

    #[test]
    fn from_samples_idle_cpu_no_stall() {
        // from_samples should not flag stall when both samples have
        // nr_running==0 on the stuck CPU.
        let s1 = MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 0,
                    rq_clock: 5000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 6000,
                    ..Default::default()
                },
            ],
        };
        let s2 = MonitorSample {
            prog_stats: None,
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 0,
                    rq_clock: 5000, // stuck but idle
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 7000,
                    ..Default::default()
                },
            ],
        };
        let summary = MonitorSummary::from_samples(&[s1, s2]);
        assert!(!summary.stall_detected);
    }

    #[test]
    fn stall_below_sustained_passes() {
        // 1 stall pair with sustained_samples=5 should pass.
        // Second CPU has a different clock value so data_looks_valid passes.
        let t = MonitorThresholds {
            fail_on_stall: true,
            sustained_samples: 5,
            ..Default::default()
        };
        let samples = vec![
            MonitorSample {
                prog_stats: None,
                elapsed_ms: 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 5000,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 6000,
                        ..Default::default()
                    },
                ],
            },
            MonitorSample {
                prog_stats: None,
                elapsed_ms: 200,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 5000,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 7000,
                        ..Default::default()
                    },
                ],
            },
            // Clock recovers.
            MonitorSample {
                prog_stats: None,
                elapsed_ms: 300,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 6000,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 8000,
                        ..Default::default()
                    },
                ],
            },
        ];
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let v = t.evaluate(&report);
        assert!(v.passed, "1 stall < sustained=5: {:?}", v.details);
    }

    #[test]
    fn neg_fallback_rate_threshold_fires() {
        let t = MonitorThresholds {
            sustained_samples: 2,
            max_fallback_rate: 5.0,
            fail_on_stall: false,
            ..Default::default()
        };
        let samples: Vec<_> = (0..3u64)
            .map(|i| sample_with_events(i * 100, 1000 + i * 500, i as i64 * 10, 0))
            .collect();
        let summary = MonitorSummary::from_samples(&samples);
        assert!(
            summary.event_deltas.is_some(),
            "event deltas must be computed"
        );
        let report = MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let v = t.evaluate(&report);
        assert!(!v.passed, "fallback rate must be caught");
        // Format: "fallback rate 200.0/s exceeded threshold 5.0/s for 2 consecutive intervals (ending at sample 2)"
        let detail = v
            .details
            .iter()
            .find(|d| d.contains("fallback rate"))
            .unwrap();
        assert!(detail.contains("/s"), "must include rate unit: {detail}");
        assert!(
            detail.contains("exceeded threshold"),
            "must state threshold: {detail}"
        );
        assert!(
            detail.contains("5.0/s"),
            "must show threshold value: {detail}"
        );
        assert!(
            detail.contains("consecutive intervals"),
            "must show sustained count: {detail}"
        );
    }

    #[test]
    fn neg_keep_last_rate_threshold_fires() {
        let t = MonitorThresholds {
            sustained_samples: 2,
            max_keep_last_rate: 5.0,
            fail_on_stall: false,
            ..Default::default()
        };
        let samples: Vec<_> = (0..3u64)
            .map(|i| sample_with_events(i * 100, 1000 + i * 500, 0, i as i64 * 10))
            .collect();
        let summary = MonitorSummary::from_samples(&samples);
        assert!(summary.event_deltas.is_some());
        let report = MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let v = t.evaluate(&report);
        assert!(!v.passed, "keep_last rate must be caught");
        // Format: "keep_last rate .../s exceeded threshold 5.0/s for 2 consecutive intervals ..."
        let detail = v
            .details
            .iter()
            .find(|d| d.contains("keep_last rate"))
            .unwrap();
        assert!(detail.contains("/s"), "must include rate unit: {detail}");
        assert!(
            detail.contains("exceeded threshold"),
            "must state threshold: {detail}"
        );
        assert!(
            detail.contains("5.0/s"),
            "must show threshold value: {detail}"
        );
    }

    // -- vCPU CPU time gating tests --

    #[test]
    fn evaluate_suppresses_stall_when_vcpu_preempted() {
        // vcpu_cpu_time_ns shows < threshold advancement -> vCPU was
        // preempted, stall should be suppressed. Use explicit threshold
        // (10ms) to avoid host CONFIG_HZ dependency.
        let t = MonitorThresholds {
            fail_on_stall: true,
            sustained_samples: 1,
            ..Default::default()
        };
        let samples = vec![
            MonitorSample {
                prog_stats: None,
                elapsed_ms: 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 5000,
                        vcpu_cpu_time_ns: Some(1_000_000_000),
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 6000,
                        vcpu_cpu_time_ns: Some(1_000_000_000),
                        ..Default::default()
                    },
                ],
            },
            MonitorSample {
                prog_stats: None,
                elapsed_ms: 200,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 5000,                        // stuck
                        vcpu_cpu_time_ns: Some(1_000_500_000), // 0.5ms < 10ms threshold
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 7000,
                        vcpu_cpu_time_ns: Some(1_010_000_000),
                        ..Default::default()
                    },
                ],
            },
        ];
        let summary = MonitorSummary::from_samples_with_threshold(&samples, 10_000_000);
        assert!(
            !summary.stall_detected,
            "preempted vCPU should not flag stall in summary"
        );
        let report = MonitorReport {
            samples,
            summary,
            preemption_threshold_ns: 10_000_000,
            watchdog_observation: None,
        };
        let v = t.evaluate(&report);
        assert!(
            v.passed,
            "preempted vCPU should suppress stall: {:?}",
            v.details
        );
    }

    #[test]
    fn evaluate_catches_stall_when_vcpu_running() {
        // vcpu_cpu_time_ns shows advancement >= threshold -> vCPU was
        // running, stall is real. Use explicit threshold (10ms) to avoid
        // host CONFIG_HZ dependency (DEFAULT_HZ=250 gives 40ms threshold,
        // which would mask the 10ms advance).
        let t = MonitorThresholds {
            fail_on_stall: true,
            sustained_samples: 1,
            ..Default::default()
        };
        let samples = vec![
            MonitorSample {
                prog_stats: None,
                elapsed_ms: 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 5000,
                        vcpu_cpu_time_ns: Some(1_000_000_000),
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 6000,
                        vcpu_cpu_time_ns: Some(1_000_000_000),
                        ..Default::default()
                    },
                ],
            },
            MonitorSample {
                prog_stats: None,
                elapsed_ms: 200,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 5000,                        // stuck
                        vcpu_cpu_time_ns: Some(1_010_000_000), // 10ms advance
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 7000,
                        vcpu_cpu_time_ns: Some(1_010_000_000),
                        ..Default::default()
                    },
                ],
            },
        ];
        let summary = MonitorSummary::from_samples_with_threshold(&samples, 10_000_000);
        assert!(
            summary.stall_detected,
            "running vCPU with stuck clock is a stall"
        );
        let report = MonitorReport {
            samples,
            summary,
            preemption_threshold_ns: 10_000_000,
            watchdog_observation: None,
        };
        let v = t.evaluate(&report);
        assert!(!v.passed, "running vCPU stall must fail: {:?}", v.details);
        assert!(v.details.iter().any(|d| d.contains("rq_clock stall")));
    }

    #[test]
    fn evaluate_stall_none_vcpu_time_falls_back_to_current_behavior() {
        // vcpu_cpu_time_ns is None -> assume vCPU was running (don't suppress).
        let t = MonitorThresholds {
            fail_on_stall: true,
            sustained_samples: 1,
            ..Default::default()
        };
        let samples = vec![
            MonitorSample {
                prog_stats: None,
                elapsed_ms: 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 5000,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 6000,
                        ..Default::default()
                    },
                ],
            },
            MonitorSample {
                prog_stats: None,
                elapsed_ms: 200,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 5000, // stuck, no vcpu_cpu_time_ns
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 7000,
                        ..Default::default()
                    },
                ],
            },
        ];
        let summary = MonitorSummary::from_samples(&samples);
        assert!(
            summary.stall_detected,
            "None vcpu time should not suppress stall"
        );
        let report = MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let v = t.evaluate(&report);
        assert!(
            !v.passed,
            "None vcpu time should detect stall: {:?}",
            v.details
        );
    }

    #[test]
    fn from_samples_suppresses_stall_when_vcpu_preempted() {
        // from_samples_with_threshold should respect vcpu_cpu_time_ns
        // gating. Use explicit threshold to avoid host CONFIG_HZ dependency.
        let s1 = MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,
                    vcpu_cpu_time_ns: Some(1_000_000_000),
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 6000,
                    vcpu_cpu_time_ns: Some(1_000_000_000),
                    ..Default::default()
                },
            ],
        };
        let s2 = MonitorSample {
            prog_stats: None,
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,                        // stuck
                    vcpu_cpu_time_ns: Some(1_000_100_000), // 0.1ms < 10ms threshold
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 7000,
                    vcpu_cpu_time_ns: Some(1_010_000_000),
                    ..Default::default()
                },
            ],
        };
        let summary = MonitorSummary::from_samples_with_threshold(&[s1, s2], 10_000_000);
        assert!(
            !summary.stall_detected,
            "preempted vCPU should not flag stall"
        );
    }

    // -- SchedstatDeltas tests --

    fn sample_with_schedstat(
        elapsed_ms: u64,
        clock_base: u64,
        run_delay: u64,
        pcount: u64,
        sched_count: u32,
        ttwu_count: u32,
    ) -> MonitorSample {
        MonitorSample {
            prog_stats: None,
            elapsed_ms,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 2,
                    rq_clock: clock_base,
                    schedstat: Some(RqSchedstat {
                        run_delay,
                        pcount,
                        sched_count,
                        ttwu_count,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 2,
                    rq_clock: clock_base + 100,
                    schedstat: Some(RqSchedstat {
                        run_delay,
                        pcount,
                        sched_count,
                        ttwu_count,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ],
        }
    }

    #[test]
    fn schedstat_deltas_computed_from_samples() {
        // 2 CPUs, each starting at run_delay=1000, ending at 5000.
        // Total delta = 2 * (5000 - 1000) = 8000.
        let samples = vec![
            sample_with_schedstat(0, 1000, 1000, 10, 50, 30),
            sample_with_schedstat(1000, 2000, 5000, 20, 100, 60),
        ];
        let summary = MonitorSummary::from_samples(&samples);
        let d = summary.schedstat_deltas.unwrap();
        assert_eq!(d.total_run_delay, 8000);
        assert_eq!(d.total_pcount, 20);
        assert_eq!(d.total_sched_count, 100);
        assert_eq!(d.total_ttwu_count, 60);
        // Rate: 8000 ns / 1.0 s = 8000.0 ns/s.
        assert!((d.run_delay_rate - 8000.0).abs() < f64::EPSILON);
        assert!((d.sched_count_rate - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn schedstat_deltas_none_without_schedstat() {
        let samples = vec![balanced_sample(100, 1000), balanced_sample(200, 1500)];
        let summary = MonitorSummary::from_samples(&samples);
        assert!(summary.schedstat_deltas.is_none());
    }

    #[test]
    fn schedstat_deltas_single_sample() {
        // Single sample -> first == last, duration=0, rates=0.
        let samples = vec![sample_with_schedstat(100, 1000, 5000, 10, 50, 30)];
        let summary = MonitorSummary::from_samples(&samples);
        let d = summary.schedstat_deltas.unwrap();
        assert_eq!(d.run_delay_rate, 0.0);
        assert_eq!(d.sched_count_rate, 0.0);
        assert_eq!(d.total_run_delay, 0);
    }

    #[test]
    fn schedstat_deltas_rates() {
        // 1 CPU, 500ms window. run_delay increases by 2000, sched_count by 40.
        // run_delay_rate = 2000 / 0.5 = 4000.0 ns/s.
        // sched_count_rate = 40 / 0.5 = 80.0 /s.
        let samples = vec![
            sample_with_schedstat(0, 1000, 1000, 5, 10, 20),
            sample_with_schedstat(500, 2000, 3000, 15, 50, 40),
        ];
        let summary = MonitorSummary::from_samples(&samples);
        let d = summary.schedstat_deltas.unwrap();
        // 2 CPUs, each delta = 2000, total = 4000.
        assert_eq!(d.total_run_delay, 4000);
        // rate = 4000 / 0.5s = 8000.0
        assert!((d.run_delay_rate - 8000.0).abs() < f64::EPSILON);
        // 2 CPUs, each sched_count delta = 40, total = 80.
        assert_eq!(d.total_sched_count, 80);
        // rate = 80 / 0.5s = 160.0
        assert!((d.sched_count_rate - 160.0).abs() < f64::EPSILON);
    }

    #[test]
    fn schedstat_deltas_all_fields() {
        let make = |elapsed_ms, rd, pc, yc, sc, sg, tc, tl| MonitorSample {
            prog_stats: None,
            elapsed_ms,
            cpus: vec![CpuSnapshot {
                nr_running: 1,
                rq_clock: elapsed_ms * 10,
                schedstat: Some(RqSchedstat {
                    run_delay: rd,
                    pcount: pc,
                    yld_count: yc,
                    sched_count: sc,
                    sched_goidle: sg,
                    ttwu_count: tc,
                    ttwu_local: tl,
                }),
                ..Default::default()
            }],
        };
        let samples = vec![
            make(100, 100, 10, 1, 20, 5, 30, 15),
            make(200, 500, 25, 4, 50, 12, 70, 35),
        ];
        let summary = MonitorSummary::from_samples(&samples);
        let d = summary.schedstat_deltas.unwrap();
        assert_eq!(d.total_run_delay, 400);
        assert_eq!(d.total_pcount, 15);
        assert_eq!(d.total_yld_count, 3);
        assert_eq!(d.total_sched_count, 30);
        assert_eq!(d.total_sched_goidle, 7);
        assert_eq!(d.total_ttwu_count, 40);
        assert_eq!(d.total_ttwu_local, 20);
    }

    // -- SustainedViolationTracker direct tests --

    #[test]
    fn sustained_tracker_no_violations() {
        let t = SustainedViolationTracker::default();
        assert!(!t.sustained(3));
        assert_eq!(t.worst_run, 0);
    }

    #[test]
    fn sustained_tracker_single_violation_not_sustained() {
        let mut t = SustainedViolationTracker::default();
        t.record(true, 5.0, 0);
        assert!(!t.sustained(3));
        assert_eq!(t.worst_run, 1);
        assert_eq!(t.worst_at, 0);
        assert!((t.worst_value - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn sustained_tracker_meets_threshold() {
        let mut t = SustainedViolationTracker::default();
        t.record(true, 2.0, 0);
        t.record(true, 3.0, 1);
        t.record(true, 4.0, 2);
        assert!(t.sustained(3));
        assert_eq!(t.worst_run, 3);
        assert_eq!(t.worst_at, 2);
        assert!((t.worst_value - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn sustained_tracker_reset_on_non_violation() {
        let mut t = SustainedViolationTracker::default();
        t.record(true, 1.0, 0);
        t.record(true, 2.0, 1);
        t.record(false, 0.0, 2); // reset
        t.record(true, 3.0, 3);
        assert!(!t.sustained(3));
        assert_eq!(t.worst_run, 2); // longest consecutive run was 2
        assert_eq!(t.consecutive, 1); // current run is 1
    }

    #[test]
    fn sustained_tracker_worst_run_preserved_after_reset() {
        let mut t = SustainedViolationTracker::default();
        for i in 0..5 {
            t.record(true, i as f64, i);
        }
        t.record(false, 0.0, 5);
        t.record(true, 99.0, 6);
        t.record(true, 100.0, 7);
        // Worst run is 5 from the first sequence.
        assert_eq!(t.worst_run, 5);
        assert!(t.sustained(5));
        assert!(!t.sustained(6));
    }
}
