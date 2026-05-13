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

pub mod arena;
pub mod bpf_map;
pub mod bpf_prog;
pub mod bpf_syscall;
pub mod btf_offsets;
pub mod btf_render;
pub(crate) mod cast_analysis;
pub mod debug_capture;
pub mod dmesg_scx;
pub mod dump;
pub mod guest;
pub mod idr;
pub mod live_host_kernel;
pub mod perf_counters;
pub mod reader;
pub mod reproducer_gen;
pub mod runnable_scan;
pub mod scx_static_alloc;
pub mod scx_walker;
pub mod sdt_alloc;
pub mod symbols;
pub mod task_enrichment;
pub mod timeline;

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) mod test_util;

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

/// Kernel virtual address. Newtype around `u64` so a KVA can't be
/// mistaken for a guest DRAM offset (`PA`) or for the base values
/// [`Cr3Pa`]/[`PageOffset`] at any page-walk call site. `Display`
/// renders as `0x<hex>` for tracing output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Kva(pub u64);

impl std::fmt::Display for Kva {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#x}", self.0)
    }
}

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
/// Routes the `KTSTR_KERNEL` read through [`crate::ktstr_kernel_env`]
/// so the empty/whitespace normalization matches every other reader.
/// When the env value is a [`crate::kernel_path::KernelId::Version`]
/// or [`KernelId::CacheKey`], the cache is consulted via
/// [`crate::cli::resolve_cached_kernel`] and the resolved entry dir is
/// passed to [`crate::kernel_path::resolve_btf`]; when it is a
/// [`KernelId::Path`], the path flows through `resolve_btf` directly
/// as before. Unset env or unresolved Version/CacheKey falls through
/// to `resolve_btf(None)` so the local-tree / sysfs fallbacks still
/// apply. See `resolve_btf` for the full resolution order.
#[cfg(test)]
pub fn find_test_vmlinux() -> Option<std::path::PathBuf> {
    use crate::kernel_path::KernelId;
    let raw = crate::ktstr_kernel_env();
    let resolved_dir: Option<String> = match raw.as_deref().map(KernelId::parse) {
        Some(KernelId::Path(p)) => p.into_os_string().into_string().ok(),
        Some(id @ (KernelId::Version(_) | KernelId::CacheKey(_))) => {
            // Cache lookup. On failure, fall through to `None` so
            // `resolve_btf(None)` still tries the local-tree / sysfs
            // fallbacks — a test running with a stale env pointer
            // shouldn't be any worse off than a test with no env set.
            crate::cli::resolve_cached_kernel(&id, "ktstr test")
                .ok()
                .and_then(|p| p.into_os_string().into_string().ok())
        }
        // Multi-kernel specs (`A..B` ranges, `git+URL#REF`) cannot
        // resolve to a single BTF source — there is no dispatch
        // loop here, just a one-shot lookup feeding `resolve_btf`.
        // Treat as "no env hint" and let the local-tree / sysfs
        // fallbacks pick a vmlinux if one exists; the env value
        // would have surfaced a hard error at the actual VM-boot
        // entry point instead.
        Some(KernelId::Range { .. }) | Some(KernelId::Git { .. }) => None,
        None => None,
    };
    let result = crate::kernel_path::resolve_btf(resolved_dir.as_deref());
    if result.is_none() {
        crate::report::test_skip(format!("no vmlinux found; {}", crate::KTSTR_KERNEL_HINT));
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
    /// Live `PAGE_OFFSET` value used by the monitor for KVA→PA
    /// translation, captured at the moment the per-iteration
    /// `DATA_VALID` latch fired. On KASLR-randomized kernels
    /// (`CONFIG_RANDOMIZE_MEMORY`) the guest publishes a
    /// randomized base into `page_offset_base` only after early
    /// boot completes, so this records the value the monitor
    /// actually used to read `struct rq` rather than the static
    /// fallback (`0xffff_8880_0000_0000` on x86_64). 0 means the
    /// latch never fired (guest never finished boot, or the
    /// monitor was not started).
    #[doc(hidden)]
    #[serde(default)]
    pub page_offset: u64,
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
    /// Host-side hardware perf counters for the vCPU thread that owns
    /// this guest CPU. Captured via `perf_event_open(2)` with
    /// `exclude_host=1` so the PMU only ticks while the vCPU is
    /// running guest code. `None` when perf is unavailable on the
    /// host (paranoid policy, missing CAP_PERFMON, hardware lacks
    /// the requested counter), when no TID is registered for this
    /// vCPU, or before the per-vCPU counter set was opened on the
    /// first sample. See [`perf_counters`](crate::monitor::perf_counters).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vcpu_perf: Option<perf_counters::VcpuPerfSample>,
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
    pub stuck_detected: bool,
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

        // Stuck detection: any CPU whose rq_clock did not advance between
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
        let mut stuck_detected = false;
        let valid_samples: Vec<&MonitorSample> = samples
            .iter()
            .filter(|s| !s.cpus.is_empty() && sample_looks_valid(s))
            .collect();
        for w in valid_samples.windows(2) {
            let prev = w[0];
            let curr = w[1];
            let cpu_count = prev.cpus.len().min(curr.cpus.len());
            for cpu in 0..cpu_count {
                if reader::is_cpu_stuck(&prev.cpus[cpu], &curr.cpus[cpu], threshold) {
                    stuck_detected = true;
                    break;
                }
            }
            if stuck_detected {
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
            stuck_detected,
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

        let first_by_name: std::collections::HashMap<&str, &bpf_prog::ProgRuntimeStats> =
            first_progs.iter().map(|p| (p.name.as_str(), p)).collect();

        let deltas: Vec<ProgStatsDelta> = last_progs
            .iter()
            .map(|lp| {
                let fp = first_by_name.get(lp.name.as_str()).copied();
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
///
/// Default behaviour is REPORT-ONLY: violations populate
/// [`MonitorVerdict::details`] but [`MonitorVerdict::passed`] stays
/// `true`. To make violations fail the test, opt into enforcement via
/// [`crate::assert::Assert::with_monitor_defaults`] (which sets
/// `enforce = true`) or by constructing `MonitorThresholds` with
/// `enforce: true` explicitly.
///
/// The two-mode design lets a test attach monitor coverage for
/// diagnostic purposes without inheriting a five-axis failure
/// surface the test author did not opt into.
#[derive(Debug, Clone, Copy)]
pub struct MonitorThresholds {
    /// Max allowed imbalance ratio (max_nr_running / max(1, min_nr_running)).
    pub max_imbalance_ratio: f64,
    /// Max allowed local DSQ depth on any CPU in any sample.
    pub max_local_dsq_depth: u32,
    /// Flag when any CPU's rq_clock does not advance between consecutive samples.
    pub fail_on_stall: bool,
    /// Number of consecutive samples that must violate a threshold before flagging.
    pub sustained_samples: usize,
    /// Max sustained select_cpu_fallback events/s across all CPUs.
    pub max_fallback_rate: f64,
    /// Max sustained dispatch_keep_last events/s across all CPUs.
    pub max_keep_last_rate: f64,
    /// Promote violations from report-only to pass/fail. When `false`
    /// (the default), [`MonitorThresholds::evaluate`] still walks every
    /// sample and records every violation in the verdict's `details`,
    /// but returns `passed: true` regardless. When `true`, any
    /// recorded violation also fails the verdict.
    pub enforce: bool,
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
        enforce: false,
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

        // Stuck detection: any CPU whose rq_clock did not advance between
        // consecutive samples. Uses the sustained_samples window like
        // imbalance and DSQ checks. Exempt idle CPUs (NOHZ stopped the
        // tick so rq_clock legitimately doesn't advance) and preempted
        // vCPUs (host stole the core, so the vCPU couldn't tick the
        // clock). See `reader::is_cpu_stuck` for the predicate.
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
                        reader::is_cpu_stuck(&prev.cpus[cpu], &curr.cpus[cpu], threshold);
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
            if self.enforce {
                format!("monitor FAILED: {} violation(s)", details.len())
            } else {
                format!(
                    "monitor flagged {} violation(s) (report-only; pass `Assert::with_monitor_defaults` to enforce)",
                    details.len()
                )
            }
        } else {
            "monitor OK".into()
        };

        MonitorVerdict {
            passed: !failed || !self.enforce,
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
