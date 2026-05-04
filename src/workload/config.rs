//! Declarative configuration types for the workload pipeline.
//!
//! Holds every type a test author writes (or that round-trips through
//! serde) without crossing the kernel boundary itself: [`WorkloadConfig`]
//! and its [`WorkSpec`] composed entries, the per-knob enums
//! ([`SchedPolicy`], [`SchedClass`], [`MemPolicy`], [`MpolFlags`],
//! [`CloneMode`], [`FutexLockMode`], [`WakeMechanism`], [`AluWidth`]),
//! the [`defaults`] constants `WorkType::from_name` consults, the
//! [`humantime_serde_helper`] module the duration fields cite, and the
//! [`resolve_work_type`] selector. The corresponding kernel-call
//! helpers live in the [`spawn`](super::spawn) submodule
//! (`apply_mempolicy_with_flags`, `apply_nice`, `build_nodemask`)
//! and the [`worker`](super::worker) submodule
//! (`set_sched_policy` in `worker/sched.rs`).
//!
//! Types are re-exported from the parent module via `pub use config::*`,
//! so existing `crate::workload::WorkloadConfig` paths continue to
//! resolve.

use std::collections::BTreeSet;
use std::time::Duration;

use super::{AffinityIntent, WorkType};

/// Serde helper for [`std::time::Duration`] using human-readable
/// strings (`"100ms"`, `"5s"`, `"1h30m"`) instead of the default
/// `{secs, nanos}` object.
///
/// Wire format chosen so persisted [`WorkSpec`] / [`WorkloadConfig`]
/// values are operator-readable: a test author who exports a config
/// can edit `"work_per_hop": "100us"` directly without translating
/// from `{secs: 0, nanos: 100_000}`.
///
/// Reuses the [`humantime`] crate already pulled in for CLI flag
/// parsing — no new dependency. Use via `#[serde(with =
/// "humantime_serde_helper")]` on `Duration` fields.
pub(crate) mod humantime_serde_helper {
    use std::time::Duration;

    pub fn serialize<S: serde::Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&humantime::format_duration(*d).to_string())
    }

    pub fn deserialize<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        let s = <String as serde::Deserialize>::deserialize(d)?;
        humantime::parse_duration(&s).map_err(serde::de::Error::custom)
    }
}

/// Named defaults for the parametric [`WorkType`] variants, used by
/// [`WorkType::from_name`]. Extracting the magic numbers here
/// provides a named home for the default values so tests and docs
/// (e.g. `doc/guide/src/architecture/workers.md`) can cite them by
/// constant name instead of each tracking a scattered integer
/// literal. Every value carries a single-line comment naming the
/// knob and its unit; the const names mirror the
/// `{variant_snake}_{field}` convention so renames show up as
/// compile errors in both sites.
pub mod defaults {
    // Bursty
    pub const BURSTY_BURST_DURATION: std::time::Duration = std::time::Duration::from_millis(50);
    pub const BURSTY_SLEEP_DURATION: std::time::Duration = std::time::Duration::from_millis(100);
    // PipeIo
    pub const PIPE_IO_BURST_ITERS: u64 = 1024;
    // FutexPingPong
    pub const FUTEX_PING_PONG_SPIN_ITERS: u64 = 1024;
    // CachePressure / CacheYield / CachePipe share buffer shape
    pub const CACHE_PRESSURE_SIZE_KB: usize = 32;
    pub const CACHE_PRESSURE_STRIDE: usize = 64;
    pub const CACHE_YIELD_SIZE_KB: usize = 32;
    pub const CACHE_YIELD_STRIDE: usize = 64;
    pub const CACHE_PIPE_SIZE_KB: usize = 32;
    pub const CACHE_PIPE_BURST_ITERS: u64 = 1024;
    // FutexFanOut
    pub const FUTEX_FAN_OUT_FAN_OUT: usize = 4;
    pub const FUTEX_FAN_OUT_SPIN_ITERS: u64 = 1024;
    // AffinityChurn
    pub const AFFINITY_CHURN_SPIN_ITERS: u64 = 1024;
    // PolicyChurn
    pub const POLICY_CHURN_SPIN_ITERS: u64 = 1024;
    // FanOutCompute
    pub const FAN_OUT_COMPUTE_FAN_OUT: usize = 4;
    pub const FAN_OUT_COMPUTE_CACHE_FOOTPRINT_KB: usize = 256;
    pub const FAN_OUT_COMPUTE_OPERATIONS: usize = 5;
    pub const FAN_OUT_COMPUTE_SLEEP_USEC: u64 = 100;
    // PageFaultChurn
    pub const PAGE_FAULT_CHURN_REGION_KB: usize = 4096;
    pub const PAGE_FAULT_CHURN_TOUCHES_PER_CYCLE: usize = 256;
    pub const PAGE_FAULT_CHURN_SPIN_ITERS: u64 = 64;
    // MutexContention
    pub const MUTEX_CONTENTION_CONTENDERS: usize = 4;
    pub const MUTEX_CONTENTION_HOLD_ITERS: u64 = 256;
    pub const MUTEX_CONTENTION_WORK_ITERS: u64 = 1024;
    // ThunderingHerd
    pub const THUNDERING_HERD_WAITERS: usize = 7;
    pub const THUNDERING_HERD_BATCHES: u64 = 1_000;
    pub const THUNDERING_HERD_INTER_BATCH_MS: u64 = 5;
    // PriorityInversion
    pub const PRIORITY_INVERSION_HIGH_COUNT: usize = 1;
    pub const PRIORITY_INVERSION_MEDIUM_COUNT: usize = 1;
    pub const PRIORITY_INVERSION_LOW_COUNT: usize = 1;
    pub const PRIORITY_INVERSION_HOLD_ITERS: u64 = 4096;
    pub const PRIORITY_INVERSION_WORK_ITERS: u64 = 1024;
    pub const PRIORITY_INVERSION_PI_MODE: super::FutexLockMode = super::FutexLockMode::Plain;
    // ProducerConsumerImbalance
    pub const PRODUCER_CONSUMER_PRODUCERS: usize = 2;
    pub const PRODUCER_CONSUMER_CONSUMERS: usize = 1;
    pub const PRODUCER_CONSUMER_PRODUCE_RATE_HZ: u64 = 1_000;
    pub const PRODUCER_CONSUMER_CONSUME_ITERS: u64 = 4_096;
    pub const PRODUCER_CONSUMER_QUEUE_DEPTH_TARGET: u64 = 1024;
    // RtStarvation
    pub const RT_STARVATION_RT_WORKERS: usize = 1;
    pub const RT_STARVATION_CFS_WORKERS: usize = 1;
    pub const RT_STARVATION_RT_PRIORITY: i32 = 50;
    pub const RT_STARVATION_BURST_ITERS: u64 = 1024;
    // AsymmetricWaker
    pub const ASYMMETRIC_WAKER_BURST_ITERS: u64 = 1024;
    // WakeChain
    pub const WAKE_CHAIN_DEPTH: usize = 4;
    pub const WAKE_CHAIN_WAKE: super::WakeMechanism = super::WakeMechanism::Pipe;
    pub const WAKE_CHAIN_WORK_PER_HOP: std::time::Duration = std::time::Duration::from_micros(100);
    // NumaWorkingSetSweep
    pub const NUMA_WORKING_SET_SWEEP_REGION_KB: usize = 4_096;
    pub const NUMA_WORKING_SET_SWEEP_SWEEP_PERIOD_MS: u64 = 100;
    // CgroupChurn
    pub const CGROUP_CHURN_GROUPS: usize = 2;
    pub const CGROUP_CHURN_CYCLE_MS: u64 = 100;
    // SignalStorm
    pub const SIGNAL_STORM_SIGNALS_PER_ITER: u64 = 16;
    pub const SIGNAL_STORM_WORK_ITERS: u64 = 1024;
    // PreemptStorm
    pub const PREEMPT_STORM_CFS_WORKERS: usize = 2;
    pub const PREEMPT_STORM_RT_BURST_ITERS: u64 = 1024;
    pub const PREEMPT_STORM_RT_SLEEP_US: u64 = 1_000;
    // EpollStorm
    pub const EPOLL_STORM_PRODUCERS: usize = 1;
    pub const EPOLL_STORM_CONSUMERS: usize = 2;
    pub const EPOLL_STORM_EVENTS_PER_BURST: u64 = 32;
    // NumaMigrationChurn
    pub const NUMA_MIGRATION_CHURN_PERIOD_MS: u64 = 100;
    // IdleChurn
    pub const IDLE_CHURN_BURST_DURATION: std::time::Duration = std::time::Duration::from_millis(1);
    pub const IDLE_CHURN_SLEEP_DURATION: std::time::Duration = std::time::Duration::from_millis(5);
    /// Default for [`WorkType::IdleChurn`]'s `precise_timing` field.
    /// `false` keeps the inherited 50µs `current->timer_slack_ns`
    /// the variant doc describes; opt-in callers set the field to
    /// `true` directly to call `prctl(PR_SET_TIMERSLACK, 1)`.
    pub const IDLE_CHURN_PRECISE_TIMING: bool = false;
    // AluHot
    /// Default for [`WorkType::AluHot`]'s `width` field. `Widest`
    /// resolves to the widest data-path the host supports at
    /// worker entry — see [`super::AluWidth`] for the resolution
    /// order.
    pub const ALU_HOT_WIDTH: super::AluWidth = super::AluWidth::Widest;
    // IpcVariance
    /// Multiply-chain steps per hot phase in [`WorkType::IpcVariance`].
    /// At IPC 2.0 / 2 GHz this spans ~50µs — long enough that the
    /// scheduler's IPC-window observer sees a steady high-IPC
    /// signal before the cold phase flips it.
    pub const IPC_VARIANCE_HOT_ITERS: u64 = 100_000;
    /// Random cache-line touches per cold phase in
    /// [`WorkType::IpcVariance`]. 1024 touches across a 512KB
    /// working set on a typical x86 core takes ~100µs (LLC) to
    /// ~1ms (DRAM-spill).
    pub const IPC_VARIANCE_COLD_ITERS: u64 = 1024;
    /// Hot+cold pair iterations per outer loop in
    /// [`WorkType::IpcVariance`]. 64 keeps per-stop-check
    /// overhead at <2% while bounding shutdown latency to one
    /// outer iteration (~10ms with the defaults above).
    pub const IPC_VARIANCE_PERIOD_ITERS: u64 = 64;
}

/// Resolve a work type with an optional override.
///
/// Returns a clone of `override_wt` when `swappable` is true, an
/// override is provided, and the override's group size (if any)
/// divides `num_workers`. Otherwise returns a clone of `base`. When
/// `override_wt` is `None`, always returns `base` regardless of
/// `swappable`.
pub(crate) fn resolve_work_type(
    base: &WorkType,
    override_wt: Option<&WorkType>,
    swappable: bool,
    num_workers: usize,
) -> WorkType {
    if !swappable {
        return base.clone();
    }
    match override_wt {
        Some(wt) => {
            if let Some(gs) = wt.worker_group_size()
                && !num_workers.is_multiple_of(gs)
            {
                return base.clone();
            }
            wt.clone()
        }
        None => base.clone(),
    }
}

/// Linux scheduling policy for a worker process.
///
/// `Fifo`, `RoundRobin`, and `Deadline` all require `CAP_SYS_NICE`
/// (`user_check_sched_setscheduler` in `kernel/sched/syscalls.c`
/// routes rt_policy and dl_policy through `req_priv`). `Normal`,
/// `Batch`, and (entering) `Idle` are unprivileged transitions for
/// fair-policy tasks. Priority values for `Fifo`/`RoundRobin` are
/// clamped to 1-99.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchedPolicy {
    /// `SCHED_NORMAL` (CFS/EEVDF).
    Normal,
    /// `SCHED_BATCH`.
    Batch,
    /// `SCHED_IDLE`.
    Idle,
    /// `SCHED_FIFO` with the given priority (1-99).
    Fifo(u32),
    /// `SCHED_RR` with the given priority (1-99).
    RoundRobin(u32),
    /// `SCHED_DEADLINE` with explicit `runtime`, `deadline`, and
    /// `period`. Applied via `sched_setattr(2)`.
    ///
    /// Each field is a [`Duration`] — the nanosecond representation
    /// the kernel requires is materialised at the syscall site, so
    /// callers express intent in idiomatic Rust units
    /// (`Duration::from_micros(100)`, `Duration::from_millis(1)`,
    /// etc.) and don't have to thread integer-nanosecond literals
    /// through their test fixtures.
    ///
    /// Constraints (from `__checkparam_dl` in
    /// `kernel/sched/deadline.c`):
    /// - `deadline != Duration::ZERO`.
    /// - `runtime` must be at least 1024 ns (the kernel's
    ///   `DL_SCALE` floor); shorter runtimes are silently truncated
    ///   inside the kernel and break bandwidth accounting.
    /// - `runtime <= deadline`.
    /// - `period == Duration::ZERO` is legal — the kernel
    ///   substitutes `deadline` for the period when zero. When
    ///   non-zero, `deadline <= period`.
    /// - The effective period (`period` if non-zero, else
    ///   `deadline`) is checked against
    ///   `/proc/sys/kernel/sched_deadline_period_min_us` (default
    ///   100us = 100_000 ns) and
    ///   `/proc/sys/kernel/sched_deadline_period_max_us` (default
    ///   `1 << 22` us = 4_194_304_000 ns), inclusive. Both sysctls
    ///   are runtime-tunable; this crate does not pre-validate the
    ///   sysctl range and lets the kernel surface out-of-range
    ///   values as `EINVAL`.
    /// - The nanosecond count of `deadline` and `period` must each
    ///   fit in 63 bits (`< 1 << 63`, i.e. `<= i64::MAX` ns ≈ 292
    ///   years) — the kernel uses bit 63 internally. Any longer
    ///   `Duration` is rejected at the syscall site.
    ///
    /// Transitions to/from `Deadline` always require `CAP_SYS_NICE`.
    /// Tasks set to `Deadline` get exclusive bandwidth on the
    /// admission-controlled root domain; oversubscription returns
    /// `EBUSY` (see `sched_dl_overflow` in `kernel/sched/deadline.c`).
    ///
    /// `set_sched_policy` validates the structural constraints
    /// (zero-deadline, DL_SCALE floor, ordering, top-bit) before
    /// invoking `sched_setattr` so a malformed `Deadline` fails
    /// fast in user space rather than tunneling an `EINVAL`
    /// through the syscall.
    Deadline {
        /// Runtime budget per period.
        #[serde(with = "humantime_serde_helper")]
        runtime: Duration,
        /// Relative deadline from period start.
        #[serde(with = "humantime_serde_helper")]
        deadline: Duration,
        /// Period. `Duration::ZERO` means "use `deadline` as the
        /// period" per the kernel's `__checkparam_dl` substitution.
        #[serde(with = "humantime_serde_helper")]
        period: Duration,
    },
}

impl SchedPolicy {
    /// `SCHED_FIFO` with the given priority (1-99).
    pub fn fifo(priority: u32) -> Self {
        SchedPolicy::Fifo(priority)
    }

    /// `SCHED_RR` with the given priority (1-99).
    pub fn round_robin(priority: u32) -> Self {
        SchedPolicy::RoundRobin(priority)
    }

    /// `SCHED_DEADLINE` with the given runtime / deadline / period.
    /// See [`SchedPolicy::Deadline`] for parameter constraints.
    ///
    /// All three arguments share the same [`Duration`] type. The
    /// canonical order is `(runtime, deadline, period)` — runtime
    /// budget first, then the relative deadline, then the period.
    /// For tests that need to make the order obvious at the call
    /// site, prefer the struct-literal form
    /// `SchedPolicy::Deadline { runtime: ..., deadline: ...,
    /// period: ... }` which carries the field names through the
    /// reader's eye.
    ///
    /// ```
    /// # use std::time::Duration;
    /// # use ktstr::workload::SchedPolicy;
    /// // Convenience constructor — canonical (runtime, deadline, period) order.
    /// let p = SchedPolicy::deadline(
    ///     Duration::from_micros(500), // runtime
    ///     Duration::from_millis(1),   // deadline
    ///     Duration::from_millis(10),  // period
    /// );
    /// // Struct-literal form — names elide positional confusion.
    /// let q = SchedPolicy::Deadline {
    ///     runtime: Duration::from_micros(500),
    ///     deadline: Duration::from_millis(1),
    ///     period: Duration::from_millis(10),
    /// };
    /// assert!(matches!(p, SchedPolicy::Deadline { .. }));
    /// assert!(matches!(q, SchedPolicy::Deadline { .. }));
    /// ```
    pub fn deadline(runtime: Duration, deadline: Duration, period: Duration) -> Self {
        SchedPolicy::Deadline {
            runtime,
            deadline,
            period,
        }
    }
}

/// Whether [`WorkType::PriorityInversion`] uses a PI-aware mutex
/// or a plain futex.
///
/// `Pi` exercises `FUTEX_LOCK_PI` and the rt_mutex priority-boost
/// chain (`kernel/futex/pi.c`). When the low-priority lock holder
/// is preempted by a medium-priority worker, the kernel boosts
/// the holder to the high-priority waiter's priority for the
/// duration of the hold — both unblocking `high` and pinning
/// `medium` from preempting it. `Plain` uses a non-PI futex so
/// the inversion is left unrepaired and the scheduler must
/// surface the stall.
///
/// Carried as a typed wrapper rather than a `bool` to avoid
/// positional-argument confusion at call sites and so the
/// failure-dump diagnostic names the choice explicitly
/// ("pi_mode = Pi" vs "pi_mode = Plain") instead of a bare
/// boolean.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FutexLockMode {
    /// `FUTEX_LOCK_PI` with rt_mutex PI chain.
    Pi,
    /// Plain futex (no PI boost). The default — exercises the
    /// uncorrected inversion the workload exists to surface.
    #[default]
    Plain,
}

/// Wake mechanism between stages of a [`WorkType::WakeChain`].
///
/// Carried as a typed enum rather than a `bool` so call sites
/// name the choice explicitly (`Pipe` / `Futex`) instead of a
/// bare `sync: true` / `sync: false`. The serde wire format is
/// `"pipe"` / `"futex"` (snake_case).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WakeMechanism {
    /// Anon-pipe ring (`depth` pipes per chain). Wakes carry
    /// `WF_SYNC` via `wake_up_interruptible_sync_poll`, biasing
    /// scheduler placement against migration. Tests the
    /// `SCX_WAKE_SYNC` path that scx variants must respect. The
    /// default — see [`WakeChain`](WorkType::WakeChain) for the
    /// kernel call-chain citations.
    #[default]
    Pipe,
    /// Single shared futex word per chain. The active stage
    /// advances the word and `FUTEX_WAKE`s; the stage whose
    /// `pos` matches runs, others re-park. No `WF_SYNC`.
    Futex,
}

/// ALU/SIMD execution width for [`WorkType::AluHot`].
///
/// Selects the widest data-path the worker exercises per
/// multiply chain. Today every variant executes the same scalar
/// four-stream multiply chain — the width selector is preserved
/// on the wire so a downstream classifier can distinguish runs
/// that requested SIMD from runs that requested scalar even
/// though the dispatch is uniform. Wider variants WILL drive
/// more functional-unit pressure and (for AVX-512 / AMX) draw
/// the package into a frequency-throttled mode the kernel
/// scheduler must observe once SIMD intrinsics land per-arm.
/// The serde wire form is snake_case (`"scalar"`, `"vec128"`,
/// `"vec256"`, `"vec512"`, `"amx"`, `"widest"`).
///
/// # Current behaviour
///
/// All widths run the same four-stream scalar multiply path;
/// the width selector is preserved on the wire and on
/// [`WorkerReport`](crate::workload::WorkerReport) so a
/// downstream classifier can distinguish runs that requested
/// SIMD from runs that requested scalar even though the
/// dispatch is uniform.
///
/// # Default semantics
///
/// `Scalar` is the type-level Rust default (the
/// `#[derive(Default)]` fallback that serde uses when an
/// `AluWidth` field is missing on the wire — keeps backward-
/// compat for older capture data). `Widest` is the
/// workload-level default the
/// [`defaults::ALU_HOT_WIDTH`] constant resolves at runtime
/// via [`resolve_alu_width`]: tests that take
/// `WorkType::from_name("AluHot")` get the host's widest
/// available data-path, not the type-level scalar fallback.
/// The asymmetry is deliberate — type-level Default favours
/// "always available everywhere"; workload-level default
/// favours "stress the host as hard as it can run."
///
/// # Resolution rules
///
/// `Widest` is a runtime-resolved sentinel: at worker entry the
/// dispatch arm probes the host CPU via
/// [`std::is_x86_feature_detected!`] (x86_64) and picks the
/// widest available variant in the order
/// `Amx > Vec512 > Vec256 > Vec128 > Scalar`. On `aarch64` only
/// `Scalar` and `Vec128` (NEON) are available; `Vec256` /
/// `Vec512` / `Amx` are absent and `Widest` resolves to NEON
/// when present, falling back to `Scalar`. A configured value
/// that the host cannot run is downgraded to the next-widest
/// available variant with a one-shot `tracing::warn!` so the
/// test still produces useful telemetry rather than
/// hard-failing — silent downgrade without the warn would
/// mask the host capability gap.
///
/// # Frequency throttle on x86_64
///
/// On Intel client / server SKUs the AVX-512 license raises the
/// per-core voltage and lowers the all-core turbo for the
/// package; running [`Vec512`](Self::Vec512) workers under one
/// scheduler while other workers run under another biases the
/// comparison because the throttle is package-wide, not
/// per-task. Tests that A/B-compare schedulers under
/// [`Vec512`](Self::Vec512) or [`Amx`](Self::Amx) need the
/// runs serialized on the same package — the framework does
/// not currently coordinate this serialization across worker
/// groups.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AluWidth {
    /// 64-bit scalar integer multiply chain. Drives the integer
    /// pipeline only; no SIMD or AVX licensing involved.
    /// Available on every supported architecture.
    #[default]
    Scalar,
    /// 128-bit vector integer multiply chain (SSE2 on x86_64,
    /// NEON on aarch64). The widest baseline both architectures
    /// support; a reasonable default when the test cares about
    /// "vectorized ALU" without architecture-specific tuning.
    Vec128,
    /// 256-bit vector integer multiply chain (AVX2 on x86_64).
    /// Not available on aarch64 — falls back to `Vec128`
    /// (NEON) at worker entry with a one-shot warn.
    Vec256,
    /// 512-bit vector integer multiply chain (AVX-512F on
    /// x86_64). Triggers the package-wide frequency throttle
    /// described above. Not available on aarch64 — falls back
    /// to `Vec128` (NEON) at worker entry.
    Vec512,
    /// AMX tile multiply chain (x86_64 server SKUs with AMX-INT8
    /// or AMX-BF16). The widest data-path on x86_64; uses XFD
    /// gating in the kernel
    /// (`arch/x86/kernel/traps.c::handle_xfd_event` raises the
    /// #NM trap, then
    /// `arch/x86/kernel/fpu/xstate.c::__xfd_enable_feature`
    /// allocates the dynamic XSAVE area) so the first AMX
    /// instruction triggers a #NM fault and the kernel allocates
    /// the dynamic XSAVE area lazily — adds a one-time per-task
    /// latency spike on first use.
    ///
    /// AMX additionally requires
    /// `prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILE_DATA)` per
    /// process before the first AMX instruction; the framework
    /// does NOT issue this prctl, so AMX is not yet runnable.
    /// `resolve_alu_width` therefore downgrades `AluWidth::Amx`
    /// to the host's widest stable-detectable variant; AMX is
    /// not currently runnable end-to-end on this framework.
    ///
    /// Not available on aarch64 — falls back to `Vec128`.
    Amx,
    /// Resolve to the widest variant the host supports at
    /// worker entry. See the type-level doc for the resolution
    /// order. Useful as a default when the test author wants
    /// "as much ALU pressure as the host can sustain" without
    /// hardcoding an architecture or feature level.
    Widest,
}

/// Coarse Linux scheduling class identifier.
///
/// Maps to one of the kernel's six core scheduler classes:
/// `fair_sched_class` (CFS / EEVDF — covers `SCHED_NORMAL`,
/// `SCHED_BATCH`, `SCHED_IDLE`), `rt_sched_class` (covers
/// `SCHED_FIFO` and `SCHED_RR`), `dl_sched_class` (covers
/// `SCHED_DEADLINE`), and `ext_sched_class` (covers `SCHED_EXT`
/// when sched_ext is loaded). The class is a coarser concept
/// than [`SchedPolicy`] — `Cfs` covers Normal/Batch/Idle, `Rt`
/// covers Fifo/RoundRobin — and is what
/// [`WorkType::AsymmetricWaker`] consumes when it wants to
/// describe a waker / wakee pair without specifying priority
/// values. When a per-worker class is applied,
/// [`apply_sched_class`] maps the variant to the equivalent
/// [`SchedPolicy`] (using a default priority where applicable)
/// and routes through `set_sched_policy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchedClass {
    /// `fair_sched_class` — `SCHED_NORMAL` (CFS / EEVDF). The
    /// default; matches a freshly-forked task before any policy
    /// override.
    #[default]
    Cfs,
    /// `fair_sched_class` — `SCHED_BATCH` (background-friendly
    /// fair task with longer wakeup latency targets).
    Batch,
    /// `fair_sched_class` — `SCHED_IDLE` (lowest fair-class
    /// weight; runs only when nothing else is runnable).
    Idle,
    /// `rt_sched_class` — `SCHED_FIFO` at default priority
    /// `RT_DEFAULT_PRIO`. Requires `CAP_SYS_NICE`. For explicit
    /// priority control use [`SchedPolicy::Fifo`] directly.
    Rt,
    /// `dl_sched_class` — `SCHED_DEADLINE`. Maps to a
    /// minimum-bandwidth deadline reservation
    /// ([`SchedClass::default_deadline_reservation`]) so
    /// `SchedClass::Deadline` is constructible without picking
    /// runtime/deadline/period. Callers needing precise
    /// reservations should use [`SchedPolicy::Deadline`]
    /// directly.
    Deadline,
    /// `ext_sched_class` — `SCHED_EXT`. Routes the worker
    /// through the loaded sched_ext BPF scheduler. Under
    /// switch-all (the default scx-ktstr regime), this is the
    /// same effective class as `Cfs` because every fair-policy
    /// task already reroutes to ext via `task_should_scx` (see
    /// kernel/sched/ext.c). `Cfs` is preserved as the explicit
    /// "I want fair semantics" knob the user expresses; `Ext`
    /// is preserved for tests that explicitly want
    /// `policy == SCHED_EXT` set on the task_struct.
    Ext,
}

/// Default `RT_DEFAULT_PRIO` for [`SchedClass::Rt`] when mapped to
/// a [`SchedPolicy`]. Picked at the middle of the 1..=99 valid range
/// so the worker neither preempts every other RT task in the system
/// nor sits at the floor; tests that need a specific RT priority
/// must construct [`SchedPolicy::Fifo`] directly.
const RT_DEFAULT_PRIO: u32 = 50;

impl SchedClass {
    /// Resolve to an equivalent [`SchedPolicy`]. `Rt` uses
    /// [`RT_DEFAULT_PRIO`]; `Deadline` uses the minimum-bandwidth
    /// reservation (1us runtime over 1ms period — passes
    /// `__checkparam_dl` and the default sysctl bounds).
    /// `Ext` maps to `SchedPolicy::Normal` because there is no
    /// userspace `SCHED_EXT` constant in libc; tests that want
    /// the kernel to read `policy == SCHED_EXT` (which
    /// requires sched_ext-aware userspace) cannot be expressed
    /// via this helper and must call the raw syscall path.
    pub fn to_policy(self) -> SchedPolicy {
        match self {
            SchedClass::Cfs | SchedClass::Ext => SchedPolicy::Normal,
            SchedClass::Batch => SchedPolicy::Batch,
            SchedClass::Idle => SchedPolicy::Idle,
            SchedClass::Rt => SchedPolicy::Fifo(RT_DEFAULT_PRIO),
            SchedClass::Deadline => Self::default_deadline_reservation(),
        }
    }

    /// Minimum-bandwidth `SCHED_DEADLINE` reservation that passes
    /// `__checkparam_dl`'s `runtime >= DL_SCALE` floor and the
    /// kernel's default `sched_deadline_period_min_us` (100us).
    /// 1us runtime, 1ms deadline, 10ms period — bandwidth fraction
    /// 0.0001, well below admission-control limits.
    pub fn default_deadline_reservation() -> SchedPolicy {
        SchedPolicy::Deadline {
            runtime: Duration::from_micros(1),
            deadline: Duration::from_millis(1),
            period: Duration::from_millis(10),
        }
    }
}

/// NUMA memory placement policy for worker processes.
///
/// Applied via `set_mempolicy(2)` after fork, before the work loop.
/// Maps to Linux `MPOL_*` constants. When `Default`, no syscall is
/// made (inherits the parent's policy).
///
/// Optional [`MpolFlags`] modify behavior (e.g. `STATIC_NODES` to
/// keep the nodemask absolute across cpuset changes).
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemPolicy {
    /// Inherit the parent process's memory policy (no syscall).
    #[default]
    Default,
    /// Allocate only from the specified NUMA nodes (`MPOL_BIND`).
    Bind(BTreeSet<usize>),
    /// Prefer allocations from the specified node, falling back to
    /// others when the preferred node is full (`MPOL_PREFERRED`).
    Preferred(usize),
    /// Interleave allocations round-robin across the specified nodes
    /// (`MPOL_INTERLEAVE`).
    Interleave(BTreeSet<usize>),
    /// Prefer the nearest node to the CPU where the allocation occurs
    /// (`MPOL_LOCAL`). No nodemask.
    Local,
    /// Prefer allocations from any of the specified nodes, falling back
    /// to others when all preferred nodes are full
    /// (`MPOL_PREFERRED_MANY`, kernel 5.15+).
    PreferredMany(BTreeSet<usize>),
    /// Weighted interleave across the specified nodes. Page distribution
    /// is proportional to per-node weights set via
    /// `/sys/kernel/mm/mempolicy/weighted_interleave/nodeN`
    /// (`MPOL_WEIGHTED_INTERLEAVE`, kernel 6.9+).
    WeightedInterleave(BTreeSet<usize>),
}

/// Optional mode flags for `set_mempolicy(2)`.
///
/// OR'd into the mode argument. See `MPOL_F_*` in
/// `include/uapi/linux/mempolicy.h`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct MpolFlags(u32);

impl MpolFlags {
    /// No flags.
    pub const NONE: Self = Self(0);
    /// `MPOL_F_STATIC_NODES` (1 << 15): nodemask is absolute, not
    /// remapped when the task's cpuset changes.
    pub const STATIC_NODES: Self = Self(1 << 15);
    /// `MPOL_F_RELATIVE_NODES` (1 << 14): nodemask is relative to
    /// the task's current cpuset.
    pub const RELATIVE_NODES: Self = Self(1 << 14);
    /// `MPOL_F_NUMA_BALANCING` (1 << 13): enable NUMA balancing
    /// optimization for this policy.
    pub const NUMA_BALANCING: Self = Self(1 << 13);

    /// Test-only raw-bit constructor. Lets unknown-bit guards
    /// (e.g. `validate_mempolicy_cpuset` in src/scenario/ops.rs)
    /// be tested against bit patterns that are not reachable via
    /// the documented `STATIC_NODES | RELATIVE_NODES |
    /// NUMA_BALANCING` constants. Production callers must use the
    /// named constants + `union` / `BitOr` so the model stays in
    /// sync with the validator's known-bits mask.
    #[cfg(test)]
    pub(crate) const fn from_bits_for_test(bits: u32) -> Self {
        Self(bits)
    }

    /// Combine two flag sets.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Raw flag bits for passing to the syscall.
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Whether every bit in `other` is set in `self`.
    ///
    /// Set-theoretic, not syntactic: `contains(NONE)` returns `true`
    /// for any `self` (vacuous truth — the empty set is a subset of
    /// everything). Callers who want "has a non-empty intersection
    /// with `other`" must compare `self.bits() & other.bits() != 0`
    /// explicitly; using `contains` for that query silently returns
    /// `true` when the operand is `NONE` regardless of `self`.
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl std::ops::BitOr for MpolFlags {
    type Output = Self;
    /// Delegates to [`MpolFlags::union`] so the bitwise-OR logic
    /// lives in one place. `union` is `const fn` (usable in
    /// const contexts like `const` initializers); `BitOr::bitor`
    /// cannot currently be `const` on stable, so keeping both
    /// entry points is necessary, but they must never diverge.
    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

/// How [`WorkloadHandle::spawn`] creates worker tasks.
///
/// `Fork` is the default — the existing [`fork(2)`] path with
/// separate address space, separate thread group, and `waitpid`
/// reaping. `Thread` switches to [`std::thread::spawn`] for workers
/// that share the test runner's tgid.
///
/// # `WorkType` × `CloneMode` compatibility
///
/// Most [`WorkType`] variants compose with both clone modes. The
/// only exception is surfaced at spawn time by
/// [`WorkloadHandle::spawn`]:
///
/// | WorkType                | Fork | Thread |
/// |-------------------------|------|--------|
/// | All variants (default)  | OK   | OK     |
/// | [`WorkType::ForkExit`]  | OK   | reject |
///
/// `ForkExit + Thread` is rejected because the worker body calls
/// `libc::fork()` from inside a thread of the parent's tgid; the
/// child then calls `_exit(0)`, which the kernel routes through
/// `do_exit`, tearing down the entire tgid (every sibling thread
/// dies). Use [`CloneMode::Fork`] for [`WorkType::ForkExit`].
///
/// Other Thread-mode interactions worth knowing:
///
/// - [`WorkType::NiceSweep`]: `setpriority(PRIO_PROCESS, 0, …)`
///   targets the calling task only (`kernel/sys.c::sys_setpriority`
///   `case PRIO_PROCESS: if (who == 0) p = current`), so each
///   sibling thread independently sweeps its own nice. Allowed.
/// - [`WorkType::AffinityChurn`]: `sched_setaffinity(0, …)`
///   addresses the calling thread by kernel rule
///   (`kernel/sched/syscalls.c::sched_setaffinity`). Allowed; no
///   cross-thread interference.
/// - [`WorkType::PolicyChurn`]: `sched_setscheduler(0, …)` is also
///   per-task. Allowed.
/// - [`WorkType::AsymmetricWaker`] with an RT class: legal but
///   the harness still runs as its original (likely SCHED_NORMAL)
///   policy; only the worker thread is RT.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CloneMode {
    /// Plain `fork(2)`: separate address space, separate thread
    /// group (`p->tgid = p->pid`), reaped via `waitpid`. The default
    /// — preserves existing [`WorkloadHandle::spawn`] behavior.
    #[default]
    Fork,
    /// Same thread group as the spawning process. Implementation
    /// uses [`std::thread::spawn`]; the Rust thread runtime owns
    /// all clone-flag selection internally. Reaped via
    /// [`std::thread::JoinHandle`]. Workers share `tgid`,
    /// signal-handler table, and address space with the parent —
    /// observers like `task_struct->group_leader`, `tgid`,
    /// `real_parent` all match the parent's.
    Thread,
}

impl MemPolicy {
    /// Construct a `Bind` policy from any iterator of NUMA node IDs.
    ///
    /// Accepts arrays, ranges, `Vec`, `BTreeSet`, or any `IntoIterator<Item = usize>`.
    pub fn bind(nodes: impl IntoIterator<Item = usize>) -> Self {
        MemPolicy::Bind(nodes.into_iter().collect())
    }

    /// Construct a `Preferred` policy for a single NUMA node.
    pub fn preferred(node: usize) -> Self {
        MemPolicy::Preferred(node)
    }

    /// Construct an `Interleave` policy from any iterator of NUMA node IDs.
    ///
    /// Accepts arrays, ranges, `Vec`, `BTreeSet`, or any `IntoIterator<Item = usize>`.
    pub fn interleave(nodes: impl IntoIterator<Item = usize>) -> Self {
        MemPolicy::Interleave(nodes.into_iter().collect())
    }

    /// Construct a `PreferredMany` policy from any iterator of NUMA node IDs.
    pub fn preferred_many(nodes: impl IntoIterator<Item = usize>) -> Self {
        MemPolicy::PreferredMany(nodes.into_iter().collect())
    }

    /// Construct a `WeightedInterleave` policy from any iterator of NUMA node IDs.
    pub fn weighted_interleave(nodes: impl IntoIterator<Item = usize>) -> Self {
        MemPolicy::WeightedInterleave(nodes.into_iter().collect())
    }

    /// NUMA node IDs referenced by this policy.
    ///
    /// Returns the node set for `Bind`, `Interleave`, `PreferredMany`,
    /// and `WeightedInterleave`, a single-element set for `Preferred`,
    /// and an empty set for `Default`/`Local`.
    pub fn node_set(&self) -> BTreeSet<usize> {
        match self {
            MemPolicy::Default | MemPolicy::Local => BTreeSet::new(),
            MemPolicy::Bind(nodes)
            | MemPolicy::Interleave(nodes)
            | MemPolicy::PreferredMany(nodes)
            | MemPolicy::WeightedInterleave(nodes) => nodes.clone(),
            MemPolicy::Preferred(node) => [*node].into_iter().collect(),
        }
    }

    /// Validate that this policy's node set is non-empty where required.
    ///
    /// Returns `Err` with a description when a node-set-bearing policy
    /// has an empty set.
    pub fn validate(&self) -> std::result::Result<(), String> {
        match self {
            MemPolicy::Default | MemPolicy::Local => Ok(()),
            MemPolicy::Preferred(_) => Ok(()),
            MemPolicy::Bind(nodes) if nodes.is_empty() => {
                Err("Bind policy requires at least one NUMA node".into())
            }
            MemPolicy::Interleave(nodes) if nodes.is_empty() => {
                Err("Interleave policy requires at least one NUMA node".into())
            }
            MemPolicy::PreferredMany(nodes) if nodes.is_empty() => {
                Err("PreferredMany policy requires at least one NUMA node".into())
            }
            MemPolicy::WeightedInterleave(nodes) if nodes.is_empty() => {
                Err("WeightedInterleave policy requires at least one NUMA node".into())
            }
            _ => Ok(()),
        }
    }
}

/// Configuration for spawning a group of worker processes.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
// See [`WorkType`]'s `#[serde(bound(...))]` comment — embedding
// `WorkType` propagates the same lifetime-bound issue, so we pass
// through the same explicit empty bound.
#[serde(bound(deserialize = ""))]
pub struct WorkloadConfig {
    /// Number of worker processes to fork.
    pub num_workers: usize,
    /// Per-worker affinity intent. Resolved at spawn time via the
    /// same gate as composed entries (see [`Self::composed`]):
    /// [`AffinityIntent::Inherit`] (resolved to
    /// [`ResolvedAffinity::None`]),
    /// [`AffinityIntent::Exact`] (resolved to
    /// [`ResolvedAffinity::Fixed`]), and
    /// [`AffinityIntent::RandomSubset`] (resolved to
    /// [`ResolvedAffinity::Random`] — sampling deferred per-worker
    /// at spawn time) are accepted at [`WorkloadHandle::spawn`].
    /// Topology-aware variants (`SingleCpu`, `LlcAligned`,
    /// `CrossCgroup`, `SmtSiblingPair`) require scenario context
    /// and are rejected with an actionable diagnostic.
    /// Type-unified with [`WorkSpec::affinity`] so a test author
    /// writes the same affinity expression at the top level and
    /// inside `composed` entries.
    pub affinity: AffinityIntent,
    /// What each worker does.
    pub work_type: WorkType,
    /// Linux scheduling policy.
    pub sched_policy: SchedPolicy,
    /// NUMA memory placement policy.
    pub mem_policy: MemPolicy,
    /// Optional mode flags for `set_mempolicy(2)`.
    pub mpol_flags: MpolFlags,
    /// Per-worker nice value applied via `setpriority(2)` after
    /// fork, before the work loop. Range `-20..=19` per `MIN_NICE`
    /// / `MAX_NICE` in `kernel/sys.c`'s `setpriority` syscall;
    /// values outside this window are clamped kernel-side. `0` (the
    /// default) skips the syscall entirely so the worker inherits
    /// the parent's nice value.
    ///
    /// Negative values require `CAP_SYS_NICE` (the `set_one_prio`
    /// → `can_nice` path returns `EACCES` to unprivileged callers
    /// trying to lower nice below the parent's). Failures are
    /// logged once via stderr and do not abort the worker — the
    /// scheduling-policy and affinity sites use the same idiom.
    pub nice: i32,
    /// How to create each worker. Defaults to [`CloneMode::Fork`].
    pub clone_mode: CloneMode,
    /// Secondary worker groups spawned alongside the primary group
    /// described by the top-level fields. Each entry is a
    /// [`WorkSpec`] with its own `work_type`, `num_workers`,
    /// `sched_policy`, `affinity`, etc. Composed groups are spawned
    /// in declaration order after the primary group; their workers
    /// run concurrently with the primary's for the lifetime of the
    /// [`WorkloadHandle`]. The default (an empty vec) skips the
    /// composed pass and behaves exactly as the pre-composition
    /// spawn.
    ///
    /// All groups share the same stop signal —
    /// [`WorkloadHandle::stop_and_collect`] terminates primary plus
    /// every composed group atomically. Per-group stop is not
    /// supported.
    ///
    /// Reports carry [`WorkerReport::group_idx`] = 0 for the primary
    /// group and 1..=N for composed entries in declaration order.
    ///
    /// # Worked example
    ///
    /// Build a multi-group workload — primary `SpinWait(2)` plus
    /// one `PipeIo(2)` composed group plus one `YieldHeavy(1)`
    /// composed group — using either the replacing
    /// [`composed`](Self::composed) setter or the appending
    /// [`with_composed`](Self::with_composed) chain:
    ///
    /// ```
    /// use ktstr::workload::{WorkSpec, WorkType, WorkloadConfig};
    ///
    /// // Append style: each call adds one group to the existing list.
    /// let cfg = WorkloadConfig::default()
    ///     .work_type(WorkType::SpinWait)
    ///     .workers(2)
    ///     .with_composed(
    ///         WorkSpec::default()
    ///             .work_type(WorkType::pipe_io(64))
    ///             .workers(2),
    ///     )
    ///     .with_composed(
    ///         WorkSpec::default()
    ///             .work_type(WorkType::YieldHeavy)
    ///             .workers(1),
    ///     );
    /// assert_eq!(cfg.composed.len(), 2);
    ///
    /// // Replace style: one call passes every composed group at once.
    /// let cfg2 = WorkloadConfig::default()
    ///     .work_type(WorkType::SpinWait)
    ///     .workers(2)
    ///     .composed([
    ///         WorkSpec::default().work_type(WorkType::pipe_io(64)).workers(2),
    ///         WorkSpec::default().work_type(WorkType::YieldHeavy).workers(1),
    ///     ]);
    /// assert_eq!(cfg2.composed.len(), 2);
    /// ```
    ///
    /// # Resolution rules at spawn time
    ///
    /// Composed [`WorkSpec`] entries must specify
    /// [`WorkSpec::num_workers`] (`Some(n)`); the `None` default
    /// resolved by the scenario engine via
    /// `Ctx::workers_per_cgroup` is unreachable from
    /// [`WorkloadHandle::spawn`] and is rejected with an actionable
    /// diagnostic.
    ///
    /// Composed [`WorkSpec::affinity`] accepts the no-context
    /// variants [`AffinityIntent::Inherit`] (resolved to
    /// [`ResolvedAffinity::None`]), [`AffinityIntent::Exact`]
    /// (resolved to [`ResolvedAffinity::Fixed`]), and
    /// [`AffinityIntent::RandomSubset`] (resolved to
    /// [`ResolvedAffinity::Random`] — sampling deferred per-worker
    /// at spawn time). The topology-aware variants (`SingleCpu`,
    /// `LlcAligned`, `CrossCgroup`, `SmtSiblingPair`) are rejected
    /// because spawn() has no access to the
    /// [`crate::topology::TestTopology`] / cpuset state that the
    /// scenario engine threads in.
    ///
    /// Composed entries inherit the parent
    /// [`WorkloadConfig::clone_mode`] — the dispatch path
    /// (fork vs thread) is a workload-wide property, so
    /// [`WorkSpec`] carries no `clone_mode` field of its own.
    ///
    /// Composition is single-level — a [`WorkSpec`] inside
    /// `composed` has no `composed` field of its own.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub composed: Vec<WorkSpec>,
}

impl Default for WorkloadConfig {
    fn default() -> Self {
        Self {
            num_workers: 1,
            affinity: AffinityIntent::Inherit,
            work_type: WorkType::SpinWait,
            sched_policy: SchedPolicy::Normal,
            mem_policy: MemPolicy::Default,
            mpol_flags: MpolFlags::NONE,
            nice: 0,
            clone_mode: CloneMode::Fork,
            composed: Vec::new(),
        }
    }
}

impl WorkloadConfig {
    /// Set the number of worker processes.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn workers(mut self, n: usize) -> Self {
        self.num_workers = n;
        self
    }

    /// Set the per-worker affinity intent.
    ///
    /// At [`WorkloadHandle::spawn`], [`AffinityIntent::Inherit`],
    /// [`AffinityIntent::Exact`], and [`AffinityIntent::RandomSubset`]
    /// are accepted; topology-aware variants (`SingleCpu`,
    /// `LlcAligned`, `CrossCgroup`, `SmtSiblingPair`) require
    /// scenario context and are rejected.
    ///
    /// Idiomatic short form for an exact CPU set:
    /// `cfg.affinity(AffinityIntent::exact([0, 1]))`.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn affinity(mut self, a: AffinityIntent) -> Self {
        self.affinity = a;
        self
    }

    /// Set the work type.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn work_type(mut self, wt: WorkType) -> Self {
        self.work_type = wt;
        self
    }

    /// Set the Linux scheduling policy.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn sched_policy(mut self, p: SchedPolicy) -> Self {
        self.sched_policy = p;
        self
    }

    /// Set the NUMA memory placement policy.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn mem_policy(mut self, p: MemPolicy) -> Self {
        self.mem_policy = p;
        self
    }

    /// Set the NUMA memory policy mode flags.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn mpol_flags(mut self, f: MpolFlags) -> Self {
        self.mpol_flags = f;
        self
    }

    /// Set the per-worker nice value applied via `setpriority(2)`.
    ///
    /// `0` (the default) skips the syscall and inherits the
    /// parent's nice. Negative values require `CAP_SYS_NICE`.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn nice(mut self, n: i32) -> Self {
        self.nice = n;
        self
    }

    /// Set the clone mode used when spawning each worker.
    ///
    /// [`CloneMode::Fork`] (the default) preserves historical
    /// behavior. See [`CloneMode`] for the full menu and dispatch
    /// status.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn clone_mode(mut self, m: CloneMode) -> Self {
        self.clone_mode = m;
        self
    }

    /// Replace the composed worker groups (replacing setter).
    ///
    /// Pass an iterator of [`WorkSpec`] entries; the existing
    /// `composed` vec is REPLACED with the supplied entries. Each
    /// will be spawned as an independent group alongside the
    /// primary described by the top-level fields. Pass an empty
    /// iterator to clear any previously-set composed groups.
    ///
    /// Use this when you have all groups in hand at once. To add
    /// one group at a time to an existing list, use the appending
    /// [`with_composed`](Self::with_composed) instead.
    ///
    /// See [`Self::composed`] for the resolution rules applied to
    /// each entry's `num_workers` / `affinity` fields at spawn time.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn composed(mut self, specs: impl IntoIterator<Item = WorkSpec>) -> Self {
        self.composed = specs.into_iter().collect();
        self
    }

    /// Append a single composed worker group to the existing list
    /// (appending setter).
    ///
    /// The supplied [`WorkSpec`] is PUSHED onto the existing
    /// `composed` vec; previously-set groups are preserved.
    /// Convenience for chained construction:
    /// `cfg.with_composed(a).with_composed(b)` produces
    /// `composed: [a, b]`.
    ///
    /// Use this when building the group list incrementally. To
    /// replace the entire list in one call, use the replacing
    /// [`composed`](Self::composed) instead.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn with_composed(mut self, spec: WorkSpec) -> Self {
        self.composed.push(spec);
        self
    }
}

/// Workload definition for a single group of workers within a cgroup.
///
/// Extracted from [`CgroupDef`](crate::scenario::ops::CgroupDef) to allow
/// multiple concurrent work groups per cgroup. Each `WorkSpec` spawns its own
/// set of worker processes.
///
/// ```
/// # use ktstr::workload::{WorkSpec, WorkType, SchedPolicy, MemPolicy};
/// # use std::time::Duration;
/// let w = WorkSpec::default()
///     .workers(4)
///     .work_type(WorkType::bursty(
///         Duration::from_millis(50),
///         Duration::from_millis(100),
///     ))
///     .sched_policy(SchedPolicy::Batch)
///     .mem_policy(MemPolicy::bind([0, 1]));
/// assert_eq!(w.num_workers, Some(4));
/// ```
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
// See [`WorkType`]'s `#[serde(bound(...))]` comment — embedding
// `WorkType` here propagates the same lifetime-bound issue, so we
// pass through the same explicit empty bound.
#[serde(bound(deserialize = ""))]
pub struct WorkSpec {
    /// What each worker does.
    pub work_type: WorkType,
    /// Linux scheduling policy.
    pub sched_policy: SchedPolicy,
    /// Number of workers. `None` means use `Ctx::workers_per_cgroup`.
    pub num_workers: Option<usize>,
    /// Per-worker affinity intent. Resolved to [`ResolvedAffinity`] at
    /// runtime via [`resolve_affinity_for_cgroup()`](crate::scenario::resolve_affinity_for_cgroup).
    pub affinity: AffinityIntent,
    /// NUMA memory placement policy. Applied via `set_mempolicy(2)`
    /// after fork, before the work loop.
    pub mem_policy: MemPolicy,
    /// Optional mode flags for `set_mempolicy(2)`.
    pub mpol_flags: MpolFlags,
    /// Per-worker nice value applied via `setpriority(2)` after
    /// fork, before the work loop. See [`WorkloadConfig::nice`]
    /// for range, default-zero skip semantics, and `CAP_SYS_NICE`
    /// rules.
    pub nice: i32,
}

impl Default for WorkSpec {
    fn default() -> Self {
        Self {
            work_type: WorkType::SpinWait,
            sched_policy: SchedPolicy::Normal,
            num_workers: None,
            affinity: AffinityIntent::Inherit,
            mem_policy: MemPolicy::Default,
            mpol_flags: MpolFlags::NONE,
            nice: 0,
        }
    }
}

impl WorkSpec {
    /// Set the number of workers.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn workers(mut self, n: usize) -> Self {
        self.num_workers = Some(n);
        self
    }

    /// Set the work type.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn work_type(mut self, wt: WorkType) -> Self {
        self.work_type = wt;
        self
    }

    /// Set the Linux scheduling policy.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn sched_policy(mut self, p: SchedPolicy) -> Self {
        self.sched_policy = p;
        self
    }

    /// Set the per-worker affinity intent.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn affinity(mut self, a: AffinityIntent) -> Self {
        self.affinity = a;
        self
    }

    /// Set the NUMA memory placement policy.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn mem_policy(mut self, p: MemPolicy) -> Self {
        self.mem_policy = p;
        self
    }

    /// Set the NUMA memory policy mode flags.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn mpol_flags(mut self, f: MpolFlags) -> Self {
        self.mpol_flags = f;
        self
    }

    /// Set the per-worker nice value applied via `setpriority(2)`.
    ///
    /// `0` (the default) skips the syscall and inherits the
    /// parent's nice. Negative values require `CAP_SYS_NICE`.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn nice(mut self, n: i32) -> Self {
        self.nice = n;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::WorkType;
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn sched_policy_debug_shows_variant_and_priority() {
        let s = format!("{:?}", SchedPolicy::Fifo(50));
        assert!(s.contains("Fifo"), "must show variant name");
        assert!(s.contains("50"), "must show priority value");
        let s = format!("{:?}", SchedPolicy::RoundRobin(99));
        assert!(s.contains("RoundRobin"), "must show variant name");
        assert!(s.contains("99"), "must show priority value");
        // Ensure different priorities produce different output.
        let s1 = format!("{:?}", SchedPolicy::Fifo(1));
        let s10 = format!("{:?}", SchedPolicy::Fifo(10));
        assert_ne!(
            s1, s10,
            "different priorities must produce different debug output"
        );
    }
    #[test]
    fn sched_policy_copy_preserves_priority() {
        let a = SchedPolicy::Fifo(42);
        let b = a; // Copy
        match b {
            SchedPolicy::Fifo(p) => assert_eq!(p, 42),
            _ => panic!("copy must preserve variant and priority"),
        }
    }
    // -- SchedPolicy constructors --

    #[test]
    fn sched_policy_fifo_constructor() {
        match SchedPolicy::fifo(50) {
            SchedPolicy::Fifo(p) => assert_eq!(p, 50),
            _ => panic!("expected Fifo"),
        }
    }
    #[test]
    fn sched_policy_rr_constructor() {
        match SchedPolicy::round_robin(25) {
            SchedPolicy::RoundRobin(p) => assert_eq!(p, 25),
            _ => panic!("expected RoundRobin"),
        }
    }
    // -- MemPolicy tests --

    #[test]
    fn mempolicy_default_node_set_empty() {
        assert!(MemPolicy::Default.node_set().is_empty());
    }
    #[test]
    fn mempolicy_local_node_set_empty() {
        assert!(MemPolicy::Local.node_set().is_empty());
    }
    #[test]
    fn mempolicy_bind_node_set() {
        let p = MemPolicy::Bind([0, 2].into_iter().collect());
        assert_eq!(p.node_set(), [0, 2].into_iter().collect());
    }
    #[test]
    fn mempolicy_preferred_node_set() {
        let p = MemPolicy::Preferred(1);
        assert_eq!(p.node_set(), [1].into_iter().collect());
    }
    #[test]
    fn mempolicy_interleave_node_set() {
        let p = MemPolicy::Interleave([0, 1, 3].into_iter().collect());
        assert_eq!(p.node_set(), [0, 1, 3].into_iter().collect());
    }
    #[test]
    fn mempolicy_preferred_many_node_set() {
        let p = MemPolicy::preferred_many([0, 2]);
        assert_eq!(p.node_set(), [0, 2].into_iter().collect());
    }
    #[test]
    fn mempolicy_weighted_interleave_node_set() {
        let p = MemPolicy::weighted_interleave([1, 3]);
        assert_eq!(p.node_set(), [1, 3].into_iter().collect());
    }
    #[test]
    fn mempolicy_validate_preferred_many_empty() {
        assert!(
            MemPolicy::PreferredMany(BTreeSet::new())
                .validate()
                .is_err()
        );
    }
    #[test]
    fn mempolicy_validate_weighted_interleave_empty() {
        assert!(
            MemPolicy::WeightedInterleave(BTreeSet::new())
                .validate()
                .is_err()
        );
    }
    #[test]
    fn mempolicy_validate_preferred_many_ok() {
        assert!(MemPolicy::preferred_many([0]).validate().is_ok());
    }
    #[test]
    fn mempolicy_validate_weighted_interleave_ok() {
        assert!(MemPolicy::weighted_interleave([0, 1]).validate().is_ok());
    }
    #[test]
    fn mpol_flags_union() {
        let f = MpolFlags::STATIC_NODES | MpolFlags::NUMA_BALANCING;
        assert_eq!(f.bits(), (1 << 15) | (1 << 13));
    }
    #[test]
    fn mpol_flags_none_is_zero() {
        assert_eq!(MpolFlags::NONE.bits(), 0);
    }
    #[test]
    fn work_mpol_flags_builder() {
        let w = WorkSpec::default().mpol_flags(MpolFlags::STATIC_NODES);
        assert_eq!(w.mpol_flags, MpolFlags::STATIC_NODES);
    }
    #[test]
    fn mpol_flags_contains_identity() {
        assert!(MpolFlags::NONE.contains(MpolFlags::NONE));
        assert!(MpolFlags::STATIC_NODES.contains(MpolFlags::STATIC_NODES));
        let composite = MpolFlags::STATIC_NODES | MpolFlags::NUMA_BALANCING;
        assert!(composite.contains(composite));
    }
    #[test]
    fn mpol_flags_contains_superset_is_true_for_subset() {
        let composite = MpolFlags::STATIC_NODES | MpolFlags::NUMA_BALANCING;
        assert!(composite.contains(MpolFlags::STATIC_NODES));
        assert!(composite.contains(MpolFlags::NUMA_BALANCING));
    }
    #[test]
    fn mpol_flags_contains_subset_is_false_for_superset() {
        let composite = MpolFlags::STATIC_NODES | MpolFlags::NUMA_BALANCING;
        assert!(!MpolFlags::STATIC_NODES.contains(composite));
        assert!(!MpolFlags::NUMA_BALANCING.contains(composite));
    }
    #[test]
    fn mpol_flags_contains_empty_is_always_true() {
        // `(x & 0) == 0` holds for every x, so every MpolFlags
        // value — including NONE itself — is a superset of NONE.
        assert!(MpolFlags::NONE.contains(MpolFlags::NONE));
        assert!(MpolFlags::STATIC_NODES.contains(MpolFlags::NONE));
        let composite = MpolFlags::STATIC_NODES | MpolFlags::NUMA_BALANCING;
        assert!(composite.contains(MpolFlags::NONE));
    }
    #[test]
    fn mpol_flags_none_does_not_contain_any_set_flag() {
        assert!(!MpolFlags::NONE.contains(MpolFlags::STATIC_NODES));
        assert!(!MpolFlags::NONE.contains(MpolFlags::RELATIVE_NODES));
        assert!(!MpolFlags::NONE.contains(MpolFlags::NUMA_BALANCING));
    }
    #[test]
    fn mpol_flags_contains_rejects_disjoint_flag() {
        // Single-flag values that share no bits must not satisfy
        // `contains` in either direction.
        assert!(!MpolFlags::STATIC_NODES.contains(MpolFlags::NUMA_BALANCING));
        assert!(!MpolFlags::NUMA_BALANCING.contains(MpolFlags::STATIC_NODES));
    }
    #[test]
    fn mpol_flags_contains_rejects_partial_overlap() {
        // Partial bit overlap must not satisfy `contains` — every
        // bit of `other` must be set in `self`, not merely some.
        let a = MpolFlags::STATIC_NODES | MpolFlags::NUMA_BALANCING;
        let b = MpolFlags::RELATIVE_NODES | MpolFlags::NUMA_BALANCING;
        assert!(!a.contains(b));
        assert!(!b.contains(a));
    }
    // -- CloneMode tests --

    #[test]
    fn clone_mode_default_is_fork() {
        // Preserves historical fork-based behavior — anything else
        // would silently change every existing caller's spawn path.
        assert!(matches!(CloneMode::default(), CloneMode::Fork));
    }
    #[test]
    fn workload_config_default_clone_mode_is_fork() {
        let c = WorkloadConfig::default();
        assert!(matches!(c.clone_mode, CloneMode::Fork));
    }
    #[test]
    fn workload_config_clone_mode_builder() {
        let cfg = WorkloadConfig::default().clone_mode(CloneMode::Thread);
        assert!(matches!(cfg.clone_mode, CloneMode::Thread));
    }
    #[test]
    fn work_mem_policy_builder() {
        let w = WorkSpec::default().mem_policy(MemPolicy::Bind([0].into_iter().collect()));
        assert!(matches!(w.mem_policy, MemPolicy::Bind(_)));
    }
    #[test]
    fn work_default_mempolicy_is_default() {
        let w = WorkSpec::default();
        assert!(matches!(w.mem_policy, MemPolicy::Default));
    }
    #[test]
    fn workload_config_default_mempolicy() {
        let wl = WorkloadConfig::default();
        assert!(matches!(wl.mem_policy, MemPolicy::Default));
    }
    /// Full `WorkloadConfig` round-trip with `Default` ensures every
    /// field handles serde correctly together — no field is silently
    /// missing a derive.
    #[test]
    fn workload_config_default_roundtrips() {
        let cfg = WorkloadConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let back: WorkloadConfig = serde_json::from_str(&json).unwrap();
        // Compare via re-serialization since WorkloadConfig has no PartialEq.
        let json2 = serde_json::to_string(&back).unwrap();
        assert_eq!(json, json2);
    }

    // -- resolve_work_type --

    #[test]
    fn resolve_work_type_not_swappable() {
        let base = WorkType::SpinWait;
        let over = WorkType::YieldHeavy;
        let result = resolve_work_type(&base, Some(&over), false, 4);
        assert!(matches!(result, WorkType::SpinWait));
    }
    #[test]
    fn resolve_work_type_swappable_applies_override() {
        let base = WorkType::SpinWait;
        let over = WorkType::YieldHeavy;
        let result = resolve_work_type(&base, Some(&over), true, 4);
        assert!(matches!(result, WorkType::YieldHeavy));
    }
    #[test]
    fn resolve_work_type_swappable_no_override() {
        let base = WorkType::SpinWait;
        let result = resolve_work_type(&base, None, true, 4);
        assert!(matches!(result, WorkType::SpinWait));
    }
    #[test]
    fn resolve_work_type_group_size_mismatch() {
        let base = WorkType::SpinWait;
        let over = WorkType::pipe_io(100); // group_size = 2
        let result = resolve_work_type(&base, Some(&over), true, 3); // 3 not divisible by 2
        assert!(matches!(result, WorkType::SpinWait));
    }
    #[test]
    fn resolve_work_type_group_size_match() {
        let base = WorkType::SpinWait;
        let over = WorkType::pipe_io(100); // group_size = 2
        let result = resolve_work_type(&base, Some(&over), true, 4); // 4 divisible by 2
        assert!(matches!(result, WorkType::PipeIo { .. }));
    }

    // -- WorkSpec builder --

    #[test]
    fn work_builder_chain() {
        let w = WorkSpec::default()
            .workers(8)
            .work_type(WorkType::bursty(
                Duration::from_millis(10),
                Duration::from_millis(20),
            ))
            .sched_policy(SchedPolicy::Batch)
            .affinity(AffinityIntent::SingleCpu)
            .nice(7);
        assert_eq!(w.num_workers, Some(8));
        if let WorkType::Bursty {
            burst_duration,
            sleep_duration,
        } = w.work_type
        {
            assert_eq!(burst_duration, Duration::from_millis(10));
            assert_eq!(sleep_duration, Duration::from_millis(20));
        } else {
            panic!("expected Bursty variant; got {:?}", w.work_type);
        }
        assert!(matches!(w.sched_policy, SchedPolicy::Batch));
        assert!(matches!(w.affinity, AffinityIntent::SingleCpu));
        assert_eq!(w.nice, 7);
    }
    #[test]
    fn work_default_values() {
        let w = WorkSpec::default();
        assert_eq!(w.num_workers, None);
        assert!(matches!(w.work_type, WorkType::SpinWait));
        assert!(matches!(w.sched_policy, SchedPolicy::Normal));
        assert!(matches!(w.affinity, AffinityIntent::Inherit));
        // Default nice is 0 — same skip semantics as
        // [`WorkloadConfig::nice`].
        assert_eq!(w.nice, 0);
    }
}
