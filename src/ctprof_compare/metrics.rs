//! Metric registry — the master catalog the comparison pipeline
//! parameterizes itself over.
//!
//! Two registries live here:
//!
//! 1. [`CTPROF_METRICS`] — array of [`CtprofMetricDef`] entries,
//!    one per primary metric. Each entry pairs a name with an
//!    [`super::AggRule`] (typed reduction over a thread bucket),
//!    scheduler-class scope, kernel CONFIG gating, dead-counter
//!    flag, operator-facing description, and rendered-section tag.
//!    Order of entries IS load-bearing — it's the default
//!    display order for rows that have no numeric delta to sort by.
//!
//! 2. [`CTPROF_DERIVED_METRICS`] — array of [`DerivedMetricDef`]
//!    entries, one per derived metric (ratio, average, signed
//!    difference). Each entry consumes already-aggregated input
//!    metrics from a group's metrics map and produces a single
//!    [`DerivedValue`] scalar with its own scale ladder. The
//!    helpers [`input_scalar`], [`ratio_compute`], and
//!    [`ratio_of_sum_compute`] are private to this module and
//!    feed the closures stored in each entry's `compute` field.
//!
//! [`metric_display_name`] and [`metric_tags`] are pure formatters
//! for the metric-list rendering path; they take a [`CtprofMetricDef`]
//! and return the user-visible name + bracketed tag suffix.
//!
//! **PSI is intentionally NOT in this registry.** Each
//! [`super::AggRule`] variant's accessor takes
//! `&crate::ctprof::ThreadState` and returns a
//! [`crate::metric_types`] newtype (or a primitive the dispatch
//! coerces via `to_string()` for `ModeChar` / `ModeBool`); only
//! per-thread data fits that signature, while Pressure Stall
//! Information is per-snapshot (host-level) and per-cgroup. PSI
//! surfaces in dedicated secondary tables under
//! `## Host pressure / ...` and `## Pressure / ...` headers,
//! rendered by [`super::write_diff`] / `write_show` directly
//! rather than via [`super::AggRule`].

use std::collections::BTreeMap;

use super::{AggRule, Aggregated, ScaleLadder, Section};

/// One metric exposed by the comparison pipeline.
///
/// The auto-scale ladder for the rendered cell is derived from
/// [`AggRule::ladder`] at render time — there is no separate
/// `unit` tag on the metric def. A registry entry that pairs an
/// AggRule variant with a category-mismatched ladder fails at
/// compile time (the ladder mapping is a closed match on the
/// variant, not a free-form string).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct CtprofMetricDef {
    pub name: &'static str,
    pub rule: AggRule,
    /// Scheduler-class scope for the metric. `None` means
    /// class-agnostic — every task class accumulates the value
    /// (e.g. `nr_migrations`). Concrete spellings:
    /// - `"cfs-only"` — incremented strictly inside CFS-class
    ///   call paths (`kernel/sched/fair.c`), zero under
    ///   SCHED_EXT / SCHED_FIFO / SCHED_RR / SCHED_DEADLINE /
    ///   SCHED_IDLE. Examples: `nr_wakeups_affine`,
    ///   `nr_wakeups_affine_attempts`, `nr_failed_migrations_*`,
    ///   `nr_forced_migrations`, `slice_max`.
    /// - `"fair-policy"` — emitted only when
    ///   `fair_policy(p->policy)` returns true. Per
    ///   `kernel/sched/sched.h:194,203`, that admits
    ///   SCHED_NORMAL, SCHED_BATCH, AND SCHED_EXT (under
    ///   CONFIG_SCHED_CLASS_EXT). Zero under SCHED_FIFO/RR/DL/IDLE.
    ///   Example: `fair_slice_ns`.
    /// - `"non-ext"` — written by the schedstats sleep/wait
    ///   family wrappers `__update_stats_enqueue_sleeper`
    ///   (kernel/sched/stats.c:48) and `__update_stats_wait_end`
    ///   (kernel/sched/stats.c:21), called from fair.c, rt.c,
    ///   deadline.c but NOT ext.c — i.e. CFS/RT/DL accumulate,
    ///   sched_ext bypasses. Examples: `wait_sum`, `wait_count`,
    ///   `wait_max`, `voluntary_sleep_ns`, `sleep_max`,
    ///   `block_sum`, `block_max`, `iowait_sum`, `iowait_count`.
    pub sched_class: Option<&'static str>,
    /// Kernel CONFIG options that gate the metric. `&[]` means
    /// no gating (always populated when the source path runs).
    /// One element typically; multi-element when more than one
    /// gate is required (e.g. `core_forceidle_sum` requires
    /// CONFIG_SCHED_CORE AND CONFIG_SCHEDSTATS). Concrete
    /// spellings match the literal `Kconfig` symbol so an
    /// operator can `grep CONFIG_X /boot/config-$(uname -r)` to
    /// confirm. Verified gates:
    /// - `"CONFIG_SCHEDSTATS"` — gates every `__schedstat_*` /
    ///   `schedstat_*` macro call. Off → the macro is
    ///   `do { } while (0)` per `kernel/sched/stats.h:75-82`.
    /// - `"CONFIG_SCHED_INFO"` — gates the lighter-weight
    ///   `sched_info_*` accounting (`run_time_ns`,
    ///   `wait_time_ns`, `timeslices`); the schedstat file is
    ///   gated by `sched_info_on()` at
    ///   `proc_pid_schedstat` (fs/proc/base.c:511-523).
    /// - `"CONFIG_SCHED_CORE"` — gates the core-scheduling
    ///   subsystem (`__account_forceidle_time`).
    /// - `"CONFIG_SCHED_CLASS_EXT"` — gates the sched_ext
    ///   class. When off, no task can land on ext, so
    ///   `ext_enabled` reads false uniformly.
    /// - `"CONFIG_TASK_DELAY_ACCT"` — gates the delayacct
    ///   accounting path that populates the taskstats genetlink
    ///   delay-family fields (`cpu_delay_*`, `blkio_delay_*`,
    ///   etc.).
    /// - `"CONFIG_TASK_IO_ACCOUNTING"` — gates the per-task
    ///   I/O accounting fields exposed by `/proc/<tid>/io`
    ///   (`rchar`, `wchar`, `syscr`, `syscw`, `read_bytes`,
    ///   `write_bytes`, `cancelled_write_bytes`). The kernel
    ///   emits all 7 fields under one `do_io_accounting` call,
    ///   and CONFIG_TASK_IO_ACCOUNTING `depends on`
    ///   CONFIG_TASK_XACCT in `init/Kconfig` — so from the
    ///   procfs-reader perspective the file is all-or-nothing.
    pub config_gates: &'static [&'static str],
    /// True for kernel counters that are exposed in `/proc`
    /// but never incremented anywhere in the kernel tree —
    /// always reads zero. Operators reading the rendered table
    /// see the `[dead]` flag and stop chasing the always-zero
    /// cell. The registry is currently empty of `is_dead: true`
    /// entries: the previously-registered dead counters
    /// (`nr_wakeups_idle`, `nr_wakeups_passive`,
    /// `nr_migrations_cold`) were dropped from `ThreadState`
    /// and the registry; the kernel still emits the lines so
    /// the parser silently ignores them. The flag remains as
    /// infrastructure: a future kernel that resurrects a dead
    /// counter (or exposes a new always-zero one) registers
    /// with `is_dead: true` and the `[dead]` rendering path
    /// fires.
    pub is_dead: bool,
    /// One-line operator-facing description of what this metric
    /// counts. Surfaced by the `ctprof metric-list`
    /// subcommand alongside the bracketed tag suffix so an
    /// operator scanning a rendered table can map an unfamiliar
    /// metric name to its semantics without leaving the CLI.
    /// Plain ASCII. "Cumulative" is load-bearing — use it to
    /// distinguish counters from gauges; the [`AggRule`] only
    /// names the per-group reduction, not the per-thread
    /// counter shape.
    pub description: &'static str,
    /// Section this metric belongs to for the `--sections`
    /// per-row filter. Most rows tag [`Section::Primary`];
    /// taskstats-sourced rows (the eight delay-accounting
    /// categories plus the two memory watermarks) carry
    /// [`Section::TaskstatsDelay`] so an operator can scope
    /// the rendered table down to (or away from) the taskstats
    /// rows. The primary-table emitter checks
    /// [`DisplayOptions::is_section_enabled`] per row before
    /// rendering — `--sections taskstats-delay` keeps only
    /// taskstats rows, `--sections primary` excludes them, and
    /// either alone keeps the primary table open. The default
    /// (empty filter) renders every row regardless of section.
    pub section: Section,
}

/// Registry of per-thread metrics. Order here is the default
/// display order for rows that have no numeric delta to sort by
/// (ties fall back to registry order). Names are the ASCII
/// short-form used in capture code; long-form display is the
/// same — no translation layer.
///
/// **PSI is intentionally not in this registry.** Each
/// [`AggRule`] variant's accessor takes `&ThreadState` and
/// returns a [`crate::metric_types`] newtype (or a primitive
/// the dispatch coerces via `to_string()` for `ModeChar` /
/// `ModeBool`); only per-thread data fits that signature, while
/// Pressure Stall Information is per-snapshot (host-level) and
/// per-cgroup. PSI surfaces in dedicated secondary tables
/// under "## Host pressure / ..." and "## Pressure / ..."
/// headers, rendered by [`write_diff`] / `write_show` directly
/// rather than via [`AggRule`]. See [`Psi`] / [`PsiResource`] /
/// [`PsiHalf`] for the data model.
pub static CTPROF_METRICS: &[CtprofMetricDef] = &[
    // structural: group population count
    CtprofMetricDef {
        name: "thread_count",
        rule: AggRule::SumCount(|_| crate::metric_types::MonotonicCount(1)),
        sched_class: None,
        config_gates: &[],
        is_dead: false,
        description: "Number of threads in this group. Each thread contributes 1; the sum is the group population. Useful for --sort-by thread_count:desc to find groups where thread count changed the most.",
        section: Section::Primary,
    },
    // identity / structural (non-numeric aggregation)
    CtprofMetricDef {
        name: "policy",
        rule: AggRule::Mode(|t| t.policy.clone()),
        sched_class: None,
        config_gates: &[],
        is_dead: false,
        description: "Scheduling policy (SCHED_OTHER, SCHED_FIFO, SCHED_RR, SCHED_BATCH, SCHED_IDLE, SCHED_DEADLINE, SCHED_EXT).",
        section: Section::Primary,
    },
    CtprofMetricDef {
        name: "nice",
        rule: AggRule::RangeI32(|t| t.nice),
        sched_class: None,
        config_gates: &[],
        is_dead: false,
        description: "Nice value (-20..19); CFS priority knob.",
        section: Section::Primary,
    },
    // `task_prio()` value from `/proc/<tid>/stat` field 18.
    // Per-thread ordinal — aggregate as OrdinalRange (mirrors
    // `nice` directly above), not Sum. Kernel ranges per
    // `task_prio()` at `kernel/sched/syscalls.c:170`:
    // CFS=[0..39], RT=[-2..-100], DL=-101 — see the field
    // doc on [`ThreadState::priority`].
    CtprofMetricDef {
        name: "priority",
        rule: AggRule::RangeI32(|t| t.priority),
        sched_class: None,
        config_gates: &[],
        is_dead: false,
        description: "Kernel task priority from /proc/<tid>/stat field 18 (CFS=[0..39], RT=[-2..-100], DL=-101).",
        section: Section::Primary,
    },
    // Real-time scheduler priority from `/proc/<tid>/stat`
    // field 40. Bounded 0..99 in practice (SCHED_FIFO /
    // SCHED_RR range); zero for CFS tasks. OrdinalRange to
    // surface the spread across a group, like `nice` and
    // `priority`.
    CtprofMetricDef {
        name: "rt_priority",
        rule: AggRule::RangeU32(|t| t.rt_priority),
        sched_class: None,
        config_gates: &[],
        is_dead: false,
        description: "Real-time scheduler priority (0..99); 0 for non-RT tasks.",
        section: Section::Primary,
    },
    CtprofMetricDef {
        name: "cpu_affinity",
        rule: AggRule::Affinity(|t| t.cpu_affinity.clone()),
        sched_class: None,
        config_gates: &[],
        is_dead: false,
        description: "Set of CPUs the task is allowed to run on (sched_getaffinity result).",
        section: Section::Primary,
    },
    CtprofMetricDef {
        name: "processor",
        rule: AggRule::RangeI32(|t| t.processor),
        sched_class: None,
        config_gates: &[],
        is_dead: false,
        description: "Last CPU the task ran on.",
        section: Section::Primary,
    },
    CtprofMetricDef {
        name: "state",
        rule: AggRule::ModeChar(|t| t.state),
        sched_class: None,
        config_gates: &[],
        is_dead: false,
        description: "Task state letter (R running, S sleeping, D uninterruptible, Z zombie, T stopped).",
        section: Section::Primary,
    },
    // `ext_enabled` reflects whether the task is currently on
    // the sched_ext class. Gated by CONFIG_SCHED_CLASS_EXT —
    // when off, no task can land on ext, so the field reads
    // `false` uniformly across every thread.
    CtprofMetricDef {
        name: "ext_enabled",
        rule: AggRule::ModeBool(|t| t.ext_enabled),
        sched_class: None,
        config_gates: &["CONFIG_SCHED_CLASS_EXT"],
        is_dead: false,
        description: "Whether the task is currently dispatched on the sched_ext class.",
        section: Section::Primary,
    },
    // Process-wide thread count (`signal_struct->nr_threads`)
    // from `/proc/<tid>/status` `Threads:`. Capture-side
    // populates only on tid == tgid threads (leader dedup), so
    // every non-leader thread carries 0 — Sum across a group
    // would render 0 for any bucket whose leader is not part of
    // the bucket (e.g. `--group-by comm` puts non-leader threads
    // in their own comm bucket). `Max` answers "largest process
    // represented in this bucket"; the row count already covers
    // "how many threads are here". Identity/structural rather
    // than counter — placement here mirrors `state` and
    // `ext_enabled` (per-thread snapshots, not deltas).
    CtprofMetricDef {
        name: "nr_threads",
        rule: AggRule::MaxGaugeCount(|t| t.nr_threads),
        sched_class: None,
        config_gates: &[],
        is_dead: false,
        description: "Process-wide thread count (signal_struct->nr_threads); leader-only.",
        section: Section::Primary,
    },
    // scheduling
    // `run_time_ns` from `/proc/<tid>/schedstat` field 1 —
    // gated by CONFIG_SCHED_INFO via `sched_info_on()` at
    // `proc_pid_schedstat` (fs/proc/base.c:511-523).
    CtprofMetricDef {
        name: "run_time_ns",
        rule: AggRule::SumNs(|t| t.run_time_ns),
        sched_class: None,
        config_gates: &["CONFIG_SCHED_INFO"],
        is_dead: false,
        description: "Cumulative on-CPU time, ns; /proc/<tid>/schedstat field 1.",
        section: Section::Primary,
    },
    // `wait_time_ns` from `/proc/<tid>/schedstat` field 2 —
    // gated by CONFIG_SCHED_INFO via `sched_info_on()` at
    // `proc_pid_schedstat` (fs/proc/base.c:511-523).
    CtprofMetricDef {
        name: "wait_time_ns",
        rule: AggRule::SumNs(|t| t.wait_time_ns),
        sched_class: None,
        config_gates: &["CONFIG_SCHED_INFO"],
        is_dead: false,
        description: "Cumulative time waiting on the runqueue, ns; schedstat field 2.",
        section: Section::Primary,
    },
    // `timeslices` from `/proc/<tid>/schedstat` field 3 —
    // same gate as `wait_time_ns`.
    CtprofMetricDef {
        name: "timeslices",
        rule: AggRule::SumCount(|t| t.timeslices),
        sched_class: None,
        config_gates: &["CONFIG_SCHED_INFO"],
        is_dead: false,
        description: "Number of times the task was run on a CPU; schedstat field 3.",
        section: Section::Primary,
    },
    CtprofMetricDef {
        name: "voluntary_csw",
        rule: AggRule::SumCount(|t| t.voluntary_csw),
        sched_class: None,
        config_gates: &[],
        is_dead: false,
        description: "Voluntary context switches (task gave up the CPU itself).",
        section: Section::Primary,
    },
    CtprofMetricDef {
        name: "nonvoluntary_csw",
        rule: AggRule::SumCount(|t| t.nonvoluntary_csw),
        sched_class: None,
        config_gates: &[],
        is_dead: false,
        description: "Involuntary context switches (task was preempted).",
        section: Section::Primary,
    },
    // `nr_wakeups`, `_local`, `_remote`, `_sync`, `_migrate`
    // are class-agnostic — `__schedstat_inc` from
    // `kernel/sched/core.c::ttwu_stat` (e.g. line 3614 for the
    // base counter) fires for every task class. The macro
    // expands to `do { } while (0)` under !CONFIG_SCHEDSTATS
    // per `kernel/sched/stats.h:75-82`.
    CtprofMetricDef {
        name: "nr_wakeups",
        rule: AggRule::SumCount(|t| t.nr_wakeups),
        sched_class: None,
        config_gates: &["CONFIG_SCHEDSTATS"],
        is_dead: false,
        description: "Total wakeups via try_to_wake_up().",
        section: Section::Primary,
    },
    CtprofMetricDef {
        name: "nr_wakeups_local",
        rule: AggRule::SumCount(|t| t.nr_wakeups_local),
        sched_class: None,
        config_gates: &["CONFIG_SCHEDSTATS"],
        is_dead: false,
        description: "Wakeups landed on the same CPU as the waker.",
        section: Section::Primary,
    },
    CtprofMetricDef {
        name: "nr_wakeups_remote",
        rule: AggRule::SumCount(|t| t.nr_wakeups_remote),
        sched_class: None,
        config_gates: &["CONFIG_SCHEDSTATS"],
        is_dead: false,
        description: "Wakeups landed on a different CPU than the waker.",
        section: Section::Primary,
    },
    CtprofMetricDef {
        name: "nr_wakeups_sync",
        rule: AggRule::SumCount(|t| t.nr_wakeups_sync),
        sched_class: None,
        config_gates: &["CONFIG_SCHEDSTATS"],
        is_dead: false,
        description: "WF_SYNC wakeups (synchronous wakeup hint to scheduler).",
        section: Section::Primary,
    },
    CtprofMetricDef {
        name: "nr_wakeups_migrate",
        rule: AggRule::SumCount(|t| t.nr_wakeups_migrate),
        sched_class: None,
        config_gates: &["CONFIG_SCHEDSTATS"],
        is_dead: false,
        description: "Wakeups where the task migrated to a different CPU than its prior one (WF_MIGRATED); distinct from nr_wakeups_remote (waker CPU != target CPU).",
        section: Section::Primary,
    },
    // `nr_wakeups_affine`, `_attempts` are CFS-only —
    // `kernel/sched/fair.c::wake_affine` calls
    // `schedstat_inc(p->stats.nr_wakeups_affine_attempts)` at
    // line 7604 and the matching `_affine` increment at line
    // 7609. Both expand only under CFS task lifetime, so a
    // task on SCHED_EXT / SCHED_FIFO / SCHED_RR / SCHED_DL
    // never accumulates them.
    CtprofMetricDef {
        name: "nr_wakeups_affine",
        rule: AggRule::SumCount(|t| t.nr_wakeups_affine),
        sched_class: Some("cfs-only"),
        config_gates: &["CONFIG_SCHEDSTATS"],
        is_dead: false,
        description: "Wakeups that succeeded under the wake_affine() heuristic.",
        section: Section::Primary,
    },
    CtprofMetricDef {
        name: "nr_wakeups_affine_attempts",
        rule: AggRule::SumCount(|t| t.nr_wakeups_affine_attempts),
        sched_class: Some("cfs-only"),
        config_gates: &["CONFIG_SCHEDSTATS"],
        is_dead: false,
        description: "wake_affine() attempts; success rate = nr_wakeups_affine / attempts.",
        section: Section::Primary,
    },
    // `nr_migrations` is incremented unconditionally at
    // `kernel/sched/core.c:3283` (`p->se.nr_migrations++`) — no
    // schedstat macro, no class gating. Always populated.
    CtprofMetricDef {
        name: "nr_migrations",
        rule: AggRule::SumCount(|t| t.nr_migrations),
        sched_class: None,
        config_gates: &[],
        is_dead: false,
        description: "Cumulative cross-CPU migrations of the task.",
        section: Section::Primary,
    },
    // `nr_forced_migrations` is set by
    // `kernel/sched/fair.c:9775` (`schedstat_inc`) inside
    // CFS-only load-balancing.
    CtprofMetricDef {
        name: "nr_forced_migrations",
        rule: AggRule::SumCount(|t| t.nr_forced_migrations),
        sched_class: Some("cfs-only"),
        config_gates: &["CONFIG_SCHEDSTATS"],
        is_dead: false,
        description: "Migrations forced by the CFS load balancer.",
        section: Section::Primary,
    },
    // `nr_failed_migrations_*` family — all CFS-only,
    // incremented in `kernel/sched/fair.c::can_migrate_task`
    // (lines 9701, 9735, 9761, 9942).
    CtprofMetricDef {
        name: "nr_failed_migrations_affine",
        rule: AggRule::SumCount(|t| t.nr_failed_migrations_affine),
        sched_class: Some("cfs-only"),
        config_gates: &["CONFIG_SCHEDSTATS"],
        is_dead: false,
        description: "Load-balancer migrations rejected for cpu-affinity reasons.",
        section: Section::Primary,
    },
    CtprofMetricDef {
        name: "nr_failed_migrations_running",
        rule: AggRule::SumCount(|t| t.nr_failed_migrations_running),
        sched_class: Some("cfs-only"),
        config_gates: &["CONFIG_SCHEDSTATS"],
        is_dead: false,
        description: "Load-balancer migrations rejected because the task was running.",
        section: Section::Primary,
    },
    CtprofMetricDef {
        name: "nr_failed_migrations_hot",
        rule: AggRule::SumCount(|t| t.nr_failed_migrations_hot),
        sched_class: Some("cfs-only"),
        config_gates: &["CONFIG_SCHEDSTATS"],
        is_dead: false,
        description: "Load-balancer migrations rejected because the task was cache-hot.",
        section: Section::Primary,
    },
    // `wait_sum` / `wait_count` / `wait_max` — written by
    // `__update_stats_wait_end` (`kernel/sched/stats.c:21`),
    // which is called from `update_stats_wait_end_fair`
    // (kernel/sched/fair.c:1426), `update_stats_wait_end_dl`
    // (kernel/sched/deadline.c:2114), and
    // `update_stats_wait_end_rt` (kernel/sched/rt.c:1282) —
    // i.e. CFS, RT, AND DL classes accumulate. Sched_ext bypasses
    // these wrappers, so the counters stay at zero for SCHED_EXT
    // tasks. Tagged `non-ext`. Expanded to a no-op under
    // !CONFIG_SCHEDSTATS via the schedstat macros at
    // `kernel/sched/stats.h:75-82`.
    CtprofMetricDef {
        name: "wait_sum",
        rule: AggRule::SumNs(|t| t.wait_sum),
        sched_class: Some("non-ext"),
        config_gates: &["CONFIG_SCHEDSTATS"],
        is_dead: false,
        description: "Cumulative time the task waited on the runqueue, ns.",
        section: Section::Primary,
    },
    CtprofMetricDef {
        name: "wait_count",
        rule: AggRule::SumCount(|t| t.wait_count),
        sched_class: Some("non-ext"),
        config_gates: &["CONFIG_SCHEDSTATS"],
        is_dead: false,
        description: "Number of distinct runqueue-wait intervals the task accumulated.",
        section: Section::Primary,
    },
    CtprofMetricDef {
        name: "wait_max",
        rule: AggRule::MaxPeak(|t| t.wait_max),
        sched_class: Some("non-ext"),
        config_gates: &["CONFIG_SCHEDSTATS"],
        is_dead: false,
        description: "Longest single runqueue-wait interval observed, ns.",
        section: Section::Primary,
    },
    // `voluntary_sleep_ns` / `sleep_max` / `block_sum` /
    // `block_max` / `iowait_sum` / `iowait_count` — written by
    // `__update_stats_enqueue_sleeper` (kernel/sched/stats.c:48),
    // which is called from `update_stats_enqueue_sleeper_fair`
    // (kernel/sched/fair.c:1452),
    // `update_stats_enqueue_sleeper_dl`
    // (kernel/sched/deadline.c:2122), and
    // `update_stats_enqueue_sleeper_rt`
    // (kernel/sched/rt.c:1252). Same shape as the wait_* family
    // above: CFS+RT+DL accumulate, sched_ext bypasses, so the
    // counters stay at zero for SCHED_EXT tasks. Tagged `non-ext`.
    // Expanded to a no-op under !CONFIG_SCHEDSTATS via the
    // schedstat macros at `kernel/sched/stats.h:75-82`.
    // `voluntary_sleep_ns` is the capture-side normalization of
    // the kernel's `sum_sleep_runtime` — the raw value
    // double-counts block under sleep, so capture subtracts
    // `sum_block_runtime` before storing.
    CtprofMetricDef {
        name: "voluntary_sleep_ns",
        rule: AggRule::SumNs(|t| t.voluntary_sleep_ns),
        sched_class: Some("non-ext"),
        config_gates: &["CONFIG_SCHEDSTATS"],
        is_dead: false,
        description: "Pure voluntary sleep time (TASK_INTERRUPTIBLE only), ns; capture-side normalized as sum_sleep_runtime - sum_block_runtime so the kernel's sleep/block double-count is stripped before delta math.",
        section: Section::Primary,
    },
    CtprofMetricDef {
        name: "sleep_max",
        rule: AggRule::MaxPeak(|t| t.sleep_max),
        sched_class: Some("non-ext"),
        config_gates: &["CONFIG_SCHEDSTATS"],
        is_dead: false,
        description: "Longest single sleep interval observed, ns.",
        section: Section::Primary,
    },
    // No `sleep_count` metric: the kernel does not emit that
    // counter — the wake-side tally is captured by `nr_wakeups`
    // already.
    CtprofMetricDef {
        name: "block_sum",
        rule: AggRule::SumNs(|t| t.block_sum),
        sched_class: Some("non-ext"),
        config_gates: &["CONFIG_SCHEDSTATS"],
        is_dead: false,
        description: "Cumulative time the task spent blocked (TASK_UNINTERRUPTIBLE), ns.",
        section: Section::Primary,
    },
    CtprofMetricDef {
        name: "block_max",
        rule: AggRule::MaxPeak(|t| t.block_max),
        sched_class: Some("non-ext"),
        config_gates: &["CONFIG_SCHEDSTATS"],
        is_dead: false,
        description: "Longest single uninterruptible-block interval observed, ns.",
        section: Section::Primary,
    },
    // No `block_count` metric: the kernel emits no per-event
    // counter for `sum_block_runtime` (unlike `wait_sum/wait_count`
    // and `iowait_sum/iowait_count` pairs).
    CtprofMetricDef {
        name: "iowait_sum",
        rule: AggRule::SumNs(|t| t.iowait_sum),
        sched_class: Some("non-ext"),
        config_gates: &["CONFIG_SCHEDSTATS"],
        is_dead: false,
        description: "Cumulative time the task spent in iowait, ns.",
        section: Section::Primary,
    },
    CtprofMetricDef {
        name: "iowait_count",
        rule: AggRule::SumCount(|t| t.iowait_count),
        sched_class: Some("non-ext"),
        config_gates: &["CONFIG_SCHEDSTATS"],
        is_dead: false,
        description: "Number of distinct iowait intervals the task accumulated.",
        section: Section::Primary,
    },
    // delayacct_blkio_ticks (the procfs USER_HZ-ticks delivery
    // of the same delay-accounting block-I/O bucket) was removed
    // because `blkio_delay_total_ns` from the taskstats genetlink
    // path supersedes it: same kernel data via the same
    // CONFIG_TASK_DELAY_ACCT gate, but ns precision instead of
    // USER_HZ truncation, no procfs round-trip, and one row in
    // the rendered registry instead of two. ktstr always runs as
    // root (CAP_NET_ADMIN is implicit), so the procfs fallback
    // bought no extra coverage.
    // `exec_max` is set inside `update_se`
    // (`kernel/sched/fair.c:1335`), guarded by
    // `if (schedstat_enabled())`. Reachable from sched_ext via
    // `update_curr_common` (`kernel/sched/ext.c:1355`), so
    // class-agnostic at runtime, gated only by CONFIG_SCHEDSTATS.
    CtprofMetricDef {
        name: "exec_max",
        rule: AggRule::MaxPeak(|t| t.exec_max),
        sched_class: None,
        config_gates: &["CONFIG_SCHEDSTATS"],
        is_dead: false,
        description: "Longest single uninterrupted on-CPU run observed, ns.",
        section: Section::Primary,
    },
    // `slice_max` is part of the CFS-class statistics struct.
    // Per the kernel-field-semantics audit, zero under
    // sched_ext / RT / DL because the populating call sites
    // live in CFS-class entry points.
    CtprofMetricDef {
        name: "slice_max",
        rule: AggRule::MaxPeak(|t| t.slice_max),
        sched_class: Some("cfs-only"),
        config_gates: &["CONFIG_SCHEDSTATS"],
        is_dead: false,
        description: "Longest CFS slice the task was granted, ns.",
        section: Section::Primary,
    },
    // Cumulative core-scheduling forced-idle time, ns. Counter
    // (Sum). Increment is class-agnostic: `__account_forceidle_time()`
    // at `kernel/sched/cputime.c:244` does a plain
    // `__schedstat_add(p->stats.core_forceidle_sum, delta)` on
    // whichever task is running on each SMT sibling, called
    // from `__sched_core_account_forceidle()` in
    // `kernel/sched/core_sched.c:287`. Real gating is at
    // build/rq level: CONFIG_SCHED_CORE + CONFIG_SCHEDSTATS +
    // `core_forceidle_count > 0`. See [`ThreadState::core_forceidle_sum`]
    // for the full caller chain.
    // Auto_scale ns ladder takes ns → µs → ms → s. Lives next
    // to `slice_max` because both relate to scheduler-decision
    // moments rather than wait/sleep accumulation.
    CtprofMetricDef {
        name: "core_forceidle_sum",
        rule: AggRule::SumNs(|t| t.core_forceidle_sum),
        sched_class: None,
        config_gates: &["CONFIG_SCHED_CORE", "CONFIG_SCHEDSTATS"],
        is_dead: false,
        description: "Cumulative time this task forced its SMT sibling idle, ns (core scheduling).",
        section: Section::Primary,
    },
    // Current scheduler slice in ns (stale under SCHED_EXT —
    // see field doc) from `/proc/<tid>/sched`'s `slice` line.
    // Per-thread instantaneous gauge (NOT a high-water counter
    // — `slice_max` directly above is the historical max).
    // Aggregating across a group via Max surfaces the longest
    // current slice any thread is running with — Sum would
    // multiply a near-identical value across the group and
    // obscure the signal. Name `fair_slice_ns` mirrors the
    // kernel emission gate `fair_policy(p->policy)` at
    // `kernel/sched/debug.c:1363`, which (per
    // `kernel/sched/sched.h:194,203`) accepts SCHED_NORMAL,
    // SCHED_BATCH, AND SCHED_EXT under CONFIG_SCHED_CLASS_EXT.
    CtprofMetricDef {
        name: "fair_slice_ns",
        rule: AggRule::MaxGaugeNs(|t| t.fair_slice_ns),
        sched_class: Some("fair-policy"),
        config_gates: &[],
        is_dead: false,
        description: "Current scheduler slice, ns; snapshot from /proc/<tid>/sched (stale under sched_ext).",
        section: Section::Primary,
    },
    // memory
    CtprofMetricDef {
        name: "allocated_bytes",
        rule: AggRule::SumBytes(|t| t.allocated_bytes),
        sched_class: None,
        config_gates: &[],
        is_dead: false,
        description: "jemalloc per-thread allocated bytes (TSD thread_allocated counter).",
        section: Section::Primary,
    },
    CtprofMetricDef {
        name: "deallocated_bytes",
        rule: AggRule::SumBytes(|t| t.deallocated_bytes),
        sched_class: None,
        config_gates: &[],
        is_dead: false,
        description: "jemalloc per-thread deallocated bytes (TSD thread_deallocated counter).",
        section: Section::Primary,
    },
    CtprofMetricDef {
        name: "minflt",
        rule: AggRule::SumCount(|t| t.minflt),
        sched_class: None,
        config_gates: &[],
        is_dead: false,
        description: "Minor page faults (resolved without I/O).",
        section: Section::Primary,
    },
    CtprofMetricDef {
        name: "majflt",
        rule: AggRule::SumCount(|t| t.majflt),
        sched_class: None,
        config_gates: &[],
        is_dead: false,
        description: "Major page faults (required disk I/O to resolve).",
        section: Section::Primary,
    },
    CtprofMetricDef {
        name: "utime_clock_ticks",
        rule: AggRule::SumTicks(|t| t.utime_clock_ticks),
        sched_class: None,
        config_gates: &[],
        is_dead: false,
        description: "User-mode CPU time, USER_HZ ticks; /proc/<tid>/stat field 14.",
        section: Section::Primary,
    },
    CtprofMetricDef {
        name: "stime_clock_ticks",
        rule: AggRule::SumTicks(|t| t.stime_clock_ticks),
        sched_class: None,
        config_gates: &[],
        is_dead: false,
        description: "Kernel-mode CPU time, USER_HZ ticks; /proc/<tid>/stat field 15.",
        section: Section::Primary,
    },
    // I/O — `/proc/<tid>/io` is emitted by
    // `do_io_accounting` (`fs/proc/base.c`) under a single
    // `CONFIG_TASK_IO_ACCOUNTING` gate, and CONFIG_TASK_IO_ACCOUNTING
    // `depends on` CONFIG_TASK_XACCT in init/Kconfig — so from
    // the capture-pipeline perspective the file is
    // all-or-nothing. All 6 fields share the same
    // `CONFIG_TASK_IO_ACCOUNTING` gate.
    CtprofMetricDef {
        name: "rchar",
        rule: AggRule::SumBytes(|t| t.rchar),
        sched_class: None,
        config_gates: &["CONFIG_TASK_IO_ACCOUNTING"],
        is_dead: false,
        description: "Bytes read at the read syscall layer (incl. cached / pagecache hits).",
        section: Section::Primary,
    },
    CtprofMetricDef {
        name: "wchar",
        rule: AggRule::SumBytes(|t| t.wchar),
        sched_class: None,
        config_gates: &["CONFIG_TASK_IO_ACCOUNTING"],
        is_dead: false,
        description: "Bytes written at the write syscall layer (incl. pagecache / writeback).",
        section: Section::Primary,
    },
    CtprofMetricDef {
        name: "syscr",
        rule: AggRule::SumCount(|t| t.syscr),
        sched_class: None,
        config_gates: &["CONFIG_TASK_IO_ACCOUNTING"],
        is_dead: false,
        description: "Number of read syscalls.",
        section: Section::Primary,
    },
    CtprofMetricDef {
        name: "syscw",
        rule: AggRule::SumCount(|t| t.syscw),
        sched_class: None,
        config_gates: &["CONFIG_TASK_IO_ACCOUNTING"],
        is_dead: false,
        description: "Number of write syscalls.",
        section: Section::Primary,
    },
    CtprofMetricDef {
        name: "read_bytes",
        rule: AggRule::SumBytes(|t| t.read_bytes),
        sched_class: None,
        config_gates: &["CONFIG_TASK_IO_ACCOUNTING"],
        is_dead: false,
        description: "Bytes that hit the storage device on read (excludes pagecache hits).",
        section: Section::Primary,
    },
    CtprofMetricDef {
        name: "write_bytes",
        rule: AggRule::SumBytes(|t| t.write_bytes),
        sched_class: None,
        config_gates: &["CONFIG_TASK_IO_ACCOUNTING"],
        is_dead: false,
        description: "Bytes that hit the storage device on write (post-writeback).",
        section: Section::Primary,
    },
    // `cancelled_write_bytes` from `/proc/<tid>/io` 7th line.
    // `task_io_account_cancelled_write` (kernel
    // include/linux/task_io_accounting_ops.h:39-42) increments
    // `current->ioac.cancelled_write_bytes` from
    // `folio_account_cleaned` (mm/page-writeback.c:2628) when a
    // dirty folio is reclaimed without writeback (truncate /
    // inode invalidation), so the per-thread value records on
    // the truncating task — not necessarily the original writer.
    // Group-level Sum is meaningful (total cancelled-write
    // bytes for the bucket); per-thread `write_bytes -
    // cancelled_write_bytes` is NOT a derived metric because
    // the two counters track distinct parties — see the field
    // doc on ThreadState::cancelled_write_bytes.
    CtprofMetricDef {
        name: "cancelled_write_bytes",
        rule: AggRule::SumBytes(|t| t.cancelled_write_bytes),
        sched_class: None,
        config_gates: &["CONFIG_TASK_IO_ACCOUNTING"],
        is_dead: false,
        description: "Bytes the kernel deaccounted from a prior dirty-write because the page was reclaimed without writeback (truncate / inode invalidation); recorded on the truncating task, not the writer. Per-thread `write_bytes - cancelled_write_bytes` is NOT a valid derivation — see field doc.",
        section: Section::Primary,
    },
    // taskstats — captured via the kernel's genetlink TASKSTATS
    // family ([`crate::taskstats`]). Two field families share the
    // CONFIG_TASKSTATS netlink-family gate but differ in the
    // per-family kconfig:
    //
    //   - delay-accounting fields (cpu/blkio/swapin/freepages/
    //     thrashing/compact/wpcopy/irq × count/total/max/min,
    //     32 entries) are gated on CONFIG_TASKSTATS +
    //     CONFIG_TASK_DELAY_ACCT (the per-task counters in
    //     `kernel/delayacct.c`); the runtime `delayacct=on` toggle
    //     (sysctl `kernel.task_delayacct` or boot param
    //     `delayacct`) is a separate condition that must hold for
    //     the counters to actually update.
    //   - memory-watermark fields (hiwater_rss_bytes,
    //     hiwater_vm_bytes) are gated on CONFIG_TASKSTATS +
    //     CONFIG_TASK_XACCT (the extended-accounting path in
    //     `kernel/tsacct.c::xacct_add_tsk`); they do NOT respond
    //     to the `delayacct=on` toggle.
    //
    // Calling the netlink family additionally requires
    // `CAP_NET_ADMIN`. Any failed gate / missing cap collapses
    // the affected fields to zero per the best-effort capture
    // contract.
    //
    // CPU-delay block: cpu_count + cpu_delay_total are RACY —
    // updated by the sched_info path without a lock, so a reader
    // may observe count or total advance ahead of the other.
    // (cpu_delay_max / cpu_delay_min are PeakNs lifetime
    // watermarks updated at delayacct path entries; same race
    // window in principle, but the watermark semantics already
    // mask brief skew.)
    CtprofMetricDef {
        name: "cpu_delay_count",
        rule: AggRule::SumCount(|t| t.cpu_delay_count),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
        is_dead: false,
        description: "Number of off-CPU windows the task waited for the runqueue to schedule it (taskstats cpu_count). RACY: count + total are not updated atomically.",
        section: Section::TaskstatsDelay,
    },
    CtprofMetricDef {
        name: "cpu_delay_total_ns",
        rule: AggRule::SumNs(|t| t.cpu_delay_total_ns),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
        is_dead: false,
        description: "Cumulative ns the task waited on the runqueue (taskstats cpu_delay_total). Distinct from `wait_sum` (schedstat) which captures the same wait-for-CPU bucket via a different code path. RACY (see cpu_delay_count).",
        section: Section::TaskstatsDelay,
    },
    CtprofMetricDef {
        name: "cpu_delay_max_ns",
        rule: AggRule::MaxPeak(|t| t.cpu_delay_max_ns),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
        is_dead: false,
        description: "Longest single CPU-wait window observed, ns (taskstats cpu_delay_max).",
        section: Section::TaskstatsDelay,
    },
    CtprofMetricDef {
        name: "cpu_delay_min_ns",
        rule: AggRule::MaxPeak(|t| t.cpu_delay_min_ns),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
        is_dead: false,
        description: "Shortest non-zero CPU-wait window observed, ns (taskstats cpu_delay_min). Sentinel 0 means \"no events observed\" — compare against cpu_delay_count.",
        section: Section::TaskstatsDelay,
    },
    // Block-I/O delay block: serializes through `task->delays->lock`
    // so count + total are atomic (unlike cpu_*).
    CtprofMetricDef {
        name: "blkio_delay_count",
        rule: AggRule::SumCount(|t| t.blkio_delay_count),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
        is_dead: false,
        description: "Number of synchronous block-I/O wait windows (taskstats blkio_count).",
        section: Section::TaskstatsDelay,
    },
    CtprofMetricDef {
        name: "blkio_delay_total_ns",
        rule: AggRule::SumNs(|t| t.blkio_delay_total_ns),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
        is_dead: false,
        description: "Cumulative ns waiting on synchronous block I/O (taskstats blkio_delay_total). Distinct from `iowait_sum` (schedstat).",
        section: Section::TaskstatsDelay,
    },
    CtprofMetricDef {
        name: "blkio_delay_max_ns",
        rule: AggRule::MaxPeak(|t| t.blkio_delay_max_ns),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
        is_dead: false,
        description: "Longest single block-I/O wait observed, ns (taskstats blkio_delay_max).",
        section: Section::TaskstatsDelay,
    },
    CtprofMetricDef {
        name: "blkio_delay_min_ns",
        rule: AggRule::MaxPeak(|t| t.blkio_delay_min_ns),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
        is_dead: false,
        description: "Shortest non-zero block-I/O wait observed, ns (taskstats blkio_delay_min). Sentinel 0 means \"no events observed\".",
        section: Section::TaskstatsDelay,
    },
    // Swap-in delay block: OVERLAPS with thrashing_* — every
    // thrashing event is also a swapin event from the syscall
    // layer. Do not sum swapin and thrashing.
    CtprofMetricDef {
        name: "swapin_delay_count",
        rule: AggRule::SumCount(|t| t.swapin_delay_count),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
        is_dead: false,
        description: "Number of swap-in wait windows (taskstats swapin_count). OVERLAPS with thrashing_delay_count — do not sum.",
        section: Section::TaskstatsDelay,
    },
    CtprofMetricDef {
        name: "swapin_delay_total_ns",
        rule: AggRule::SumNs(|t| t.swapin_delay_total_ns),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
        is_dead: false,
        description: "Cumulative ns waiting for swap-in to complete (taskstats swapin_delay_total).",
        section: Section::TaskstatsDelay,
    },
    CtprofMetricDef {
        name: "swapin_delay_max_ns",
        rule: AggRule::MaxPeak(|t| t.swapin_delay_max_ns),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
        is_dead: false,
        description: "Longest single swap-in wait observed, ns (taskstats swapin_delay_max).",
        section: Section::TaskstatsDelay,
    },
    CtprofMetricDef {
        name: "swapin_delay_min_ns",
        rule: AggRule::MaxPeak(|t| t.swapin_delay_min_ns),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
        is_dead: false,
        description: "Shortest non-zero swap-in wait observed, ns (taskstats swapin_delay_min). Sentinel 0 means \"no events observed\".",
        section: Section::TaskstatsDelay,
    },
    // Direct memory reclaim (free-pages) block.
    CtprofMetricDef {
        name: "freepages_delay_count",
        rule: AggRule::SumCount(|t| t.freepages_delay_count),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
        is_dead: false,
        description: "Number of direct-reclaim wait windows (taskstats freepages_count).",
        section: Section::TaskstatsDelay,
    },
    CtprofMetricDef {
        name: "freepages_delay_total_ns",
        rule: AggRule::SumNs(|t| t.freepages_delay_total_ns),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
        is_dead: false,
        description: "Cumulative ns waiting in direct memory reclaim (taskstats freepages_delay_total).",
        section: Section::TaskstatsDelay,
    },
    CtprofMetricDef {
        name: "freepages_delay_max_ns",
        rule: AggRule::MaxPeak(|t| t.freepages_delay_max_ns),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
        is_dead: false,
        description: "Longest single direct-reclaim wait observed, ns (taskstats freepages_delay_max).",
        section: Section::TaskstatsDelay,
    },
    CtprofMetricDef {
        name: "freepages_delay_min_ns",
        rule: AggRule::MaxPeak(|t| t.freepages_delay_min_ns),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
        is_dead: false,
        description: "Shortest non-zero direct-reclaim wait observed, ns (taskstats freepages_delay_min). Sentinel 0 means \"no events observed\".",
        section: Section::TaskstatsDelay,
    },
    // Thrashing block: OVERLAPS with swapin_* (see above).
    CtprofMetricDef {
        name: "thrashing_delay_count",
        rule: AggRule::SumCount(|t| t.thrashing_delay_count),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
        is_dead: false,
        description: "Number of thrashing wait windows (taskstats thrashing_count). OVERLAPS with swapin_delay_count — do not sum.",
        section: Section::TaskstatsDelay,
    },
    CtprofMetricDef {
        name: "thrashing_delay_total_ns",
        rule: AggRule::SumNs(|t| t.thrashing_delay_total_ns),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
        is_dead: false,
        description: "Cumulative ns waiting under thrashing pressure (taskstats thrashing_delay_total).",
        section: Section::TaskstatsDelay,
    },
    CtprofMetricDef {
        name: "thrashing_delay_max_ns",
        rule: AggRule::MaxPeak(|t| t.thrashing_delay_max_ns),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
        is_dead: false,
        description: "Longest single thrashing wait observed, ns (taskstats thrashing_delay_max).",
        section: Section::TaskstatsDelay,
    },
    CtprofMetricDef {
        name: "thrashing_delay_min_ns",
        rule: AggRule::MaxPeak(|t| t.thrashing_delay_min_ns),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
        is_dead: false,
        description: "Shortest non-zero thrashing wait observed, ns (taskstats thrashing_delay_min). Sentinel 0 means \"no events observed\".",
        section: Section::TaskstatsDelay,
    },
    // Memory compaction block.
    CtprofMetricDef {
        name: "compact_delay_count",
        rule: AggRule::SumCount(|t| t.compact_delay_count),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
        is_dead: false,
        description: "Number of memory-compaction wait windows (taskstats compact_count).",
        section: Section::TaskstatsDelay,
    },
    CtprofMetricDef {
        name: "compact_delay_total_ns",
        rule: AggRule::SumNs(|t| t.compact_delay_total_ns),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
        is_dead: false,
        description: "Cumulative ns waiting on memory compaction (taskstats compact_delay_total).",
        section: Section::TaskstatsDelay,
    },
    CtprofMetricDef {
        name: "compact_delay_max_ns",
        rule: AggRule::MaxPeak(|t| t.compact_delay_max_ns),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
        is_dead: false,
        description: "Longest single compaction wait observed, ns (taskstats compact_delay_max).",
        section: Section::TaskstatsDelay,
    },
    CtprofMetricDef {
        name: "compact_delay_min_ns",
        rule: AggRule::MaxPeak(|t| t.compact_delay_min_ns),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
        is_dead: false,
        description: "Shortest non-zero compaction wait observed, ns (taskstats compact_delay_min). Sentinel 0 means \"no events observed\".",
        section: Section::TaskstatsDelay,
    },
    // Write-protect-copy (CoW) fault block.
    CtprofMetricDef {
        name: "wpcopy_delay_count",
        rule: AggRule::SumCount(|t| t.wpcopy_delay_count),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
        is_dead: false,
        description: "Number of write-protect-copy (CoW) fault wait windows (taskstats wpcopy_count).",
        section: Section::TaskstatsDelay,
    },
    CtprofMetricDef {
        name: "wpcopy_delay_total_ns",
        rule: AggRule::SumNs(|t| t.wpcopy_delay_total_ns),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
        is_dead: false,
        description: "Cumulative ns waiting on write-protect-copy faults (taskstats wpcopy_delay_total).",
        section: Section::TaskstatsDelay,
    },
    CtprofMetricDef {
        name: "wpcopy_delay_max_ns",
        rule: AggRule::MaxPeak(|t| t.wpcopy_delay_max_ns),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
        is_dead: false,
        description: "Longest single write-protect-copy fault wait observed, ns (taskstats wpcopy_delay_max).",
        section: Section::TaskstatsDelay,
    },
    CtprofMetricDef {
        name: "wpcopy_delay_min_ns",
        rule: AggRule::MaxPeak(|t| t.wpcopy_delay_min_ns),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
        is_dead: false,
        description: "Shortest non-zero write-protect-copy fault wait observed, ns (taskstats wpcopy_delay_min). Sentinel 0 means \"no events observed\".",
        section: Section::TaskstatsDelay,
    },
    // IRQ-handler delay block. Updates from `delayacct_irq` in
    // `kernel/delayacct.c` — counts kernel-IRQ time charged to
    // the task.
    CtprofMetricDef {
        name: "irq_delay_count",
        rule: AggRule::SumCount(|t| t.irq_delay_count),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
        is_dead: false,
        description: "Number of IRQ-handler windows charged to the task (taskstats irq_count).",
        section: Section::TaskstatsDelay,
    },
    CtprofMetricDef {
        name: "irq_delay_total_ns",
        rule: AggRule::SumNs(|t| t.irq_delay_total_ns),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
        is_dead: false,
        description: "Cumulative ns of IRQ handling charged to the task (taskstats irq_delay_total).",
        section: Section::TaskstatsDelay,
    },
    CtprofMetricDef {
        name: "irq_delay_max_ns",
        rule: AggRule::MaxPeak(|t| t.irq_delay_max_ns),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
        is_dead: false,
        description: "Longest single IRQ-handler window observed, ns (taskstats irq_delay_max).",
        section: Section::TaskstatsDelay,
    },
    CtprofMetricDef {
        name: "irq_delay_min_ns",
        rule: AggRule::MaxPeak(|t| t.irq_delay_min_ns),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
        is_dead: false,
        description: "Shortest non-zero IRQ-handler window observed, ns (taskstats irq_delay_min). Sentinel 0 means \"no events observed\".",
        section: Section::TaskstatsDelay,
    },
    // Lifetime memory watermarks. Updates from `xacct_add_tsk` in
    // `kernel/tsacct.c` — kB → bytes conversion happens at parse
    // time in `crate::taskstats::parse_taskstats_payload`. Gated
    // on CONFIG_TASK_XACCT (the "extended accounting" path), NOT
    // CONFIG_TASK_DELAY_ACCT — `xacct_add_tsk` lives behind
    // `CONFIG_TASK_XACCT` while delayacct is the parallel
    // `CONFIG_TASK_DELAY_ACCT` subsystem; the two are
    // independently selectable.
    CtprofMetricDef {
        name: "hiwater_rss_bytes",
        rule: AggRule::MaxPeakBytes(|t| t.hiwater_rss_bytes),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_XACCT"],
        is_dead: false,
        description: "Lifetime high-watermark of resident-set size, bytes (taskstats hiwater_rss). Distinct from smaps_rollup_kb[\"Rss\"] which is the CURRENT RSS.",
        section: Section::TaskstatsDelay,
    },
    CtprofMetricDef {
        name: "hiwater_vm_bytes",
        rule: AggRule::MaxPeakBytes(|t| t.hiwater_vm_bytes),
        sched_class: None,
        config_gates: &["CONFIG_TASKSTATS", "CONFIG_TASK_XACCT"],
        is_dead: false,
        description: "Lifetime high-watermark of virtual-memory size, bytes (taskstats hiwater_vm).",
        section: Section::TaskstatsDelay,
    },
];

// ---------------------------------------------------------------------------
// Derived metrics
// ---------------------------------------------------------------------------

/// Output value of a derived metric.
///
/// Derived metrics carry an `f64` scalar. The `f64` carrier is
/// chosen because the value range varies across derivations:
/// - `[0, 1]` ratios: `cpu_efficiency`, `affine_success_ratio`,
///   `involuntary_csw_ratio`.
/// - `[0, ∞)` ratios: `disk_io_fraction` (readahead can pull more
///   block-device bytes than the syscall requested, so the ratio
///   exceeds 1.0 in practice).
/// - `[0, ∞)` per-event means: `avg_wait_ns`, `avg_slice_ns`,
///   `avg_iowait_ns` — sum over count, both non-negative.
/// - `(-∞, ∞)` signed differences: `live_heap_estimate` =
///   `allocated_bytes - deallocated_bytes` can go negative when
///   the deallocation total exceeds the allocation total (a
///   freelist drains memory allocated before capture began, or
///   the per-thread TSD counters were sampled mid-update on a
///   thread that has just released a large arena).
///
/// All four shapes flow through the same `f64` carrier. The
/// per-derivation auto-scale ladder lives on
/// [`DerivedMetricDef::ladder`] (not on the value type) so the
/// renderer picks the right magnitude (ns / Bytes / unitless)
/// per row regardless of whether the value is positive, zero,
/// negative, fractional, or in the millions. The `is_ratio`
/// flag on [`DerivedMetricDef`] toggles between the auto-scaled
/// path (e.g. `1.500ms`, `7.500GiB`) and the raw three-decimal
/// path (`0.873` for ratios).
///
/// Sign preservation: the [`auto_scale`] step uses `abs()` for
/// the threshold check but propagates the original signed value
/// through the scaled output, and [`format_derived_value_cell`]
/// / [`format_derived_delta_cell`] both render with `{value:.2}`
/// or `{value:.3}` formatters that preserve the explicit `-` for
/// negatives. The [`auto_scale_preserves_sign_on_negative_input`]
/// regression test pins this for the Bytes and ns ladders.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum DerivedValue {
    /// Floating-point value. Render via the
    /// [`DerivedMetricDef::ladder`] + [`DerivedMetricDef::is_ratio`]
    /// pair: ratios format with three decimals (`0.873`,
    /// `+0.100`); ladder-bearing values
    /// ([`ScaleLadder::Ns`] / [`ScaleLadder::Bytes`] / etc.)
    /// route through the same auto-scale ladders the main table
    /// uses.
    Scalar(f64),
}

impl DerivedValue {
    /// Return the underlying `f64`. Helper for delta math
    /// downstream of [`DerivedRow`] consumers.
    pub fn as_f64(&self) -> f64 {
        match self {
            DerivedValue::Scalar(v) => *v,
        }
    }
}

/// Definition of a derived metric: a function that consumes the
/// already-aggregated input metrics for a group and produces a
/// single scalar (with its own unit and operator-facing
/// description).
///
/// The compute fn returns `None` when an input metric is missing
/// from the group's metrics map (capture-side gated by a kernel
/// CONFIG that wasn't enabled, or jemalloc not linked) OR when
/// the formula would divide by zero. The renderer surfaces a
/// `None` cell as `-` so the operator can distinguish "not
/// computable" from "computed as zero".
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct DerivedMetricDef {
    pub name: &'static str,
    /// Auto-scale ladder for the cell. [`ScaleLadder::None`] for
    /// ratio rows (renders as a bare three-decimal scalar with
    /// no suffix), [`ScaleLadder::Ns`] / [`ScaleLadder::Bytes`] /
    /// etc. for unit-bearing derivations. The same closed-match
    /// dispatch [`AggRule::ladder`] feeds.
    pub ladder: ScaleLadder,
    /// Operator-facing one-line description; surfaced by the
    /// `ctprof metric-list` subcommand.
    pub description: &'static str,
    /// Names of input metrics from [`CTPROF_METRICS`]. Pure
    /// documentation — surfaces in the `metric-list` output so
    /// the operator sees what each derivation depends on.
    pub inputs: &'static [&'static str],
    /// Render-shape flag for dimensionless quantities. When true,
    /// the renderer (1) suppresses the `%` (delta_pct) column,
    /// (2) renders the value as `N.NNN` with three decimals
    /// instead of routing through the auto-scale ladder, and
    /// (3) renders the delta as `+/-N.NNN` (no scaled unit
    /// suffix).
    ///
    /// The `[0, 1]` interval is the common case where this flag
    /// applies: `cpu_efficiency`, `affine_success_ratio`, and
    /// `involuntary_csw_ratio` all live in `[0, 1]`. Delta on a
    /// `[0, 1]` ratio reads as percentage points
    /// (0.5 → 0.6 = +0.100 = +10pp), and `delta / baseline` as
    /// a fraction (the `%` column) becomes confusing — `+20%` on
    /// a `[0, 1]` ratio is already in percentage points, so a
    /// percentage-of-percentage readout double-encodes the
    /// signal.
    ///
    /// `disk_io_fraction` (range `[0, ∞)`) carries `is_ratio: true`
    /// for the rendering shape but does NOT satisfy the
    /// percentage-points interpretation: a value of 1.5 is
    /// possible (readahead pulls more block-device bytes than
    /// the syscall requested), so a delta of +0.100 reads as
    /// "ratio rose by 0.1" rather than "ratio rose by 10
    /// percentage points." The render shape is still correct
    /// (suppress `%`, three decimals, no auto-scale) — only the
    /// pp interpretation is invalid.
    pub is_ratio: bool,
    /// The computation. Pulls input scalars from the group's
    /// metrics map via `Aggregated::numeric()` and produces the
    /// derived scalar.
    pub compute: fn(&BTreeMap<String, Aggregated>) -> Option<DerivedValue>,
    /// Section this derived metric belongs to for the
    /// `--sections` per-row filter, mirroring
    /// [`CtprofMetricDef::section`]. Most derivations tag
    /// [`Section::Derived`]; the 9 derivations whose inputs are
    /// taskstats fields (the eight `avg_*_delay_ns` averages
    /// plus `total_offcpu_delay_ns`) tag
    /// [`Section::TaskstatsDelay`] so an operator running
    /// `--sections taskstats-delay` gets a full taskstats view
    /// — the 34 raw rows AND the 9 derivations that depend on
    /// them — without dragging in unrelated derived metrics.
    /// The `## Derived metrics` table emitter checks
    /// [`DisplayOptions::is_section_enabled`] per row before
    /// rendering, and the outer-table gate opens whenever EITHER
    /// section in the rendered set is enabled.
    pub section: Section,
}

/// Helper: pull an input metric's `Aggregated::numeric()`
/// projection out of the group's metrics map.
fn input_scalar(metrics: &BTreeMap<String, Aggregated>, name: &str) -> Option<f64> {
    metrics.get(name).and_then(|a| a.numeric())
}

/// Helper: compute `num / den` for a simple ratio. Returns
/// `None` when either input is missing OR `den == 0` (so the
/// renderer surfaces `-` rather than NaN/inf). Used by the
/// majority of derived metrics whose formula is a plain
/// quotient over two registry inputs.
fn ratio_compute(
    metrics: &BTreeMap<String, Aggregated>,
    numerator: &str,
    denominator: &str,
) -> Option<DerivedValue> {
    let num = input_scalar(metrics, numerator)?;
    let den = input_scalar(metrics, denominator)?;
    if den == 0.0 {
        return None;
    }
    Some(DerivedValue::Scalar(num / den))
}

/// Helper: compute `num / (num + addend)` for ratios whose
/// denominator is a sum of two registry inputs. Returns `None`
/// when either input is missing OR the synthesized denominator
/// is zero. Used by `cpu_efficiency` (run / (run + wait)) and
/// `involuntary_csw_ratio` (nvcsw / (vcsw + nvcsw)).
fn ratio_of_sum_compute(
    metrics: &BTreeMap<String, Aggregated>,
    numerator: &str,
    addend: &str,
) -> Option<DerivedValue> {
    let num = input_scalar(metrics, numerator)?;
    let other = input_scalar(metrics, addend)?;
    let den = num + other;
    if den == 0.0 {
        return None;
    }
    Some(DerivedValue::Scalar(num / den))
}

/// Registry of derived metrics. Each entry consumes one or more
/// already-aggregated input metrics from
/// [`CTPROF_METRICS`] and produces a single scalar with its
/// own unit. See the per-entry doc strings for the formula and
/// kernel-source rationale.
pub static CTPROF_DERIVED_METRICS: &[DerivedMetricDef] = &[
    DerivedMetricDef {
        name: "affine_success_ratio",
        ladder: ScaleLadder::None,
        description: "wake_affine() success ratio: nr_wakeups_affine / nr_wakeups_affine_attempts.",
        inputs: &["nr_wakeups_affine", "nr_wakeups_affine_attempts"],
        is_ratio: true,
        compute: |m| ratio_compute(m, "nr_wakeups_affine", "nr_wakeups_affine_attempts"),
        section: Section::Derived,
    },
    DerivedMetricDef {
        name: "avg_wait_ns",
        ladder: ScaleLadder::Ns,
        description: "Average runqueue-wait duration per scheduling event: wait_sum / wait_count (ns/event).",
        inputs: &["wait_sum", "wait_count"],
        is_ratio: false,
        compute: |m| ratio_compute(m, "wait_sum", "wait_count"),
        section: Section::Derived,
    },
    // `voluntary_sleep_sum` derived metric was removed when
    // `voluntary_sleep_ns` became a first-class capture field.
    // The kernel's `sum_sleep_runtime - sum_block_runtime`
    // computation now happens at capture time inside
    // `capture_thread_at_with_tally` so every consumer reads the
    // pre-normalized value without re-deriving.
    DerivedMetricDef {
        name: "cpu_efficiency",
        ladder: ScaleLadder::None,
        description: "Fraction of total scheduler-tracked time spent on-CPU: run_time_ns / (run_time_ns + wait_time_ns).",
        inputs: &["run_time_ns", "wait_time_ns"],
        is_ratio: true,
        compute: |m| ratio_of_sum_compute(m, "run_time_ns", "wait_time_ns"),
        section: Section::Derived,
    },
    DerivedMetricDef {
        name: "avg_slice_ns",
        ladder: ScaleLadder::Ns,
        description: "Average on-CPU slice length per timeslice: run_time_ns / timeslices (ns/timeslice).",
        inputs: &["run_time_ns", "timeslices"],
        is_ratio: false,
        compute: |m| ratio_compute(m, "run_time_ns", "timeslices"),
        section: Section::Derived,
    },
    DerivedMetricDef {
        name: "involuntary_csw_ratio",
        ladder: ScaleLadder::None,
        description: "Fraction of context switches that were preemptions: nonvoluntary_csw / (voluntary_csw + nonvoluntary_csw).",
        inputs: &["nonvoluntary_csw", "voluntary_csw"],
        is_ratio: true,
        compute: |m| ratio_of_sum_compute(m, "nonvoluntary_csw", "voluntary_csw"),
        section: Section::Derived,
    },
    DerivedMetricDef {
        name: "disk_io_fraction",
        ladder: ScaleLadder::None,
        description: "Fraction of read syscall bytes that hit storage: read_bytes / rchar. Typically <= 1.0 but can exceed when readahead pulls more block-device bytes than the syscall requested.",
        inputs: &["read_bytes", "rchar"],
        is_ratio: true,
        compute: |m| ratio_compute(m, "read_bytes", "rchar"),
        section: Section::Derived,
    },
    DerivedMetricDef {
        name: "live_heap_estimate",
        ladder: ScaleLadder::Bytes,
        description: "jemalloc live-heap estimate: allocated_bytes - deallocated_bytes. Signed: negative when deallocations dominate (freelist drains memory allocated before capture, or sampled mid-update on a thread that just released a large arena). Renders with explicit `-` and the IEC binary suffix (e.g. `-1.907MiB`).",
        inputs: &["allocated_bytes", "deallocated_bytes"],
        is_ratio: false,
        compute: |m| {
            let alloc = input_scalar(m, "allocated_bytes")?;
            let dealloc = input_scalar(m, "deallocated_bytes")?;
            Some(DerivedValue::Scalar(alloc - dealloc))
        },
        section: Section::Derived,
    },
    DerivedMetricDef {
        name: "avg_iowait_ns",
        ladder: ScaleLadder::Ns,
        description: "Average iowait interval per iowait event: iowait_sum / iowait_count (ns/event).",
        inputs: &["iowait_sum", "iowait_count"],
        is_ratio: false,
        compute: |m| ratio_compute(m, "iowait_sum", "iowait_count"),
        section: Section::Derived,
    },
    // -- taskstats per-category averages (delay_total / count) --
    //
    // One average per delay-accounting category. Same shape as
    // avg_wait_ns / avg_iowait_ns above (sum-over-count quotient,
    // ns ladder, non-ratio). The category-specific caveats from
    // the registry (cpu RACY, swapin/thrashing OVERLAP, sentinel
    // semantics) carry forward into the description so an operator
    // reading `metric-list` for the derived row sees the same
    // gating discipline they get for the raw count/total fields.
    DerivedMetricDef {
        name: "avg_cpu_delay_ns",
        ladder: ScaleLadder::Ns,
        description: "Average CPU-wait per scheduling event: cpu_delay_total_ns / cpu_delay_count (ns/event). RACY: the kernel updates count + total via the lockless sched_info path, so a concurrent reader may observe one ahead of the other; the quotient is approximate at the sub-event scale and stable at the integrated scale.",
        inputs: &["cpu_delay_total_ns", "cpu_delay_count"],
        is_ratio: false,
        compute: |m| ratio_compute(m, "cpu_delay_total_ns", "cpu_delay_count"),
        section: Section::TaskstatsDelay,
    },
    DerivedMetricDef {
        name: "avg_blkio_delay_ns",
        ladder: ScaleLadder::Ns,
        description: "Average synchronous block-I/O wait per event: blkio_delay_total_ns / blkio_delay_count (ns/event). Distinct from avg_iowait_ns (schedstat) — this travels through the delayacct path and is the canonical delay-accounting block-I/O reading.",
        inputs: &["blkio_delay_total_ns", "blkio_delay_count"],
        is_ratio: false,
        compute: |m| ratio_compute(m, "blkio_delay_total_ns", "blkio_delay_count"),
        section: Section::TaskstatsDelay,
    },
    DerivedMetricDef {
        name: "avg_swapin_delay_ns",
        ladder: ScaleLadder::Ns,
        description: "Average swap-in wait per event: swapin_delay_total_ns / swapin_delay_count (ns/event). OVERLAPS with thrashing — every thrashing event is also a swapin event from the syscall layer; do not sum the two averages or the underlying totals directly.",
        inputs: &["swapin_delay_total_ns", "swapin_delay_count"],
        is_ratio: false,
        compute: |m| ratio_compute(m, "swapin_delay_total_ns", "swapin_delay_count"),
        section: Section::TaskstatsDelay,
    },
    DerivedMetricDef {
        name: "avg_freepages_delay_ns",
        ladder: ScaleLadder::Ns,
        description: "Average direct-reclaim wait per event: freepages_delay_total_ns / freepages_delay_count (ns/event).",
        inputs: &["freepages_delay_total_ns", "freepages_delay_count"],
        is_ratio: false,
        compute: |m| ratio_compute(m, "freepages_delay_total_ns", "freepages_delay_count"),
        section: Section::TaskstatsDelay,
    },
    DerivedMetricDef {
        name: "avg_thrashing_delay_ns",
        ladder: ScaleLadder::Ns,
        description: "Average thrashing wait per event: thrashing_delay_total_ns / thrashing_delay_count (ns/event). OVERLAPS with swapin (see avg_swapin_delay_ns).",
        inputs: &["thrashing_delay_total_ns", "thrashing_delay_count"],
        is_ratio: false,
        compute: |m| ratio_compute(m, "thrashing_delay_total_ns", "thrashing_delay_count"),
        section: Section::TaskstatsDelay,
    },
    DerivedMetricDef {
        name: "avg_compact_delay_ns",
        ladder: ScaleLadder::Ns,
        description: "Average memory-compaction wait per event: compact_delay_total_ns / compact_delay_count (ns/event).",
        inputs: &["compact_delay_total_ns", "compact_delay_count"],
        is_ratio: false,
        compute: |m| ratio_compute(m, "compact_delay_total_ns", "compact_delay_count"),
        section: Section::TaskstatsDelay,
    },
    DerivedMetricDef {
        name: "avg_wpcopy_delay_ns",
        ladder: ScaleLadder::Ns,
        description: "Average write-protect-copy fault wait per event: wpcopy_delay_total_ns / wpcopy_delay_count (ns/event).",
        inputs: &["wpcopy_delay_total_ns", "wpcopy_delay_count"],
        is_ratio: false,
        compute: |m| ratio_compute(m, "wpcopy_delay_total_ns", "wpcopy_delay_count"),
        section: Section::TaskstatsDelay,
    },
    DerivedMetricDef {
        name: "avg_irq_delay_ns",
        ladder: ScaleLadder::Ns,
        description: "Average IRQ-handler window per event: irq_delay_total_ns / irq_delay_count (ns/event).",
        inputs: &["irq_delay_total_ns", "irq_delay_count"],
        is_ratio: false,
        compute: |m| ratio_compute(m, "irq_delay_total_ns", "irq_delay_count"),
        section: Section::TaskstatsDelay,
    },
    // -- taskstats off-CPU rollup --
    //
    // Sum of every meaningful off-CPU delay category. Combines
    // cpu (runqueue wait), blkio (sync I/O wait), freepages
    // (direct reclaim), compact (compaction), wpcopy (CoW fault),
    // irq (IRQ-handler windows), and the LARGER of (swapin,
    // thrashing) — the two share the same syscall-layer event,
    // so summing both would double-count a thrashing-induced
    // swapin. `?` propagates None when any input is missing
    // (gating off, kernel pre-v14, etc.); `.max()` over the
    // overlap pair picks the dominant signal.
    DerivedMetricDef {
        name: "total_offcpu_delay_ns",
        ladder: ScaleLadder::Ns,
        description: "Sum of all off-CPU delay-accounting buckets, ns: cpu + blkio + freepages + compact + wpcopy + irq + max(swapin, thrashing). The swapin/thrashing pair is OR'd with .max() rather than summed because the two share syscall-layer events (every thrashing event is also a swapin). Returns `-` when any input is missing (CONFIG_TASK_DELAY_ACCT off, runtime toggle off, or kernel older than the bucket's introduction version).",
        inputs: &[
            "cpu_delay_total_ns",
            "blkio_delay_total_ns",
            "swapin_delay_total_ns",
            "freepages_delay_total_ns",
            "thrashing_delay_total_ns",
            "compact_delay_total_ns",
            "wpcopy_delay_total_ns",
            "irq_delay_total_ns",
        ],
        is_ratio: false,
        compute: |m| {
            let cpu = input_scalar(m, "cpu_delay_total_ns")?;
            let blkio = input_scalar(m, "blkio_delay_total_ns")?;
            let swapin = input_scalar(m, "swapin_delay_total_ns")?;
            let freepages = input_scalar(m, "freepages_delay_total_ns")?;
            let thrashing = input_scalar(m, "thrashing_delay_total_ns")?;
            let compact = input_scalar(m, "compact_delay_total_ns")?;
            let wpcopy = input_scalar(m, "wpcopy_delay_total_ns")?;
            let irq = input_scalar(m, "irq_delay_total_ns")?;
            let mem_overlap = swapin.max(thrashing);
            Some(DerivedValue::Scalar(
                cpu + blkio + freepages + compact + wpcopy + irq + mem_overlap,
            ))
        },
        section: Section::TaskstatsDelay,
    },
];

/// Borrow the metric's bare name from the registry. The
/// `&'static str` lifetime piggybacks on
/// [`CtprofMetricDef::name`]'s static-string storage —
/// callers may borrow the static name without allocation;
/// render sites that need owned `String`s allocate at the
/// table-cell boundary (see [`super::render`] at the
/// `metric_display_name(metric_def).to_string()` call site
/// and [`super::runner::write_metric_list`]).
///
/// Companion to [`metric_tags`], which renders the bracketed
/// `[<class>] [<tag>] ...` suffix separately. Render sites
/// concatenate the two into the final display column.
pub fn metric_display_name(metric: &CtprofMetricDef) -> &'static str {
    metric.name
}

/// Render a metric's bracketed gating tags as a single
/// space-separated string. Returns the empty string when
/// `sched_class` is `None`, `is_dead` is false, AND
/// `config_gates` is empty.
///
/// Tag emission order: `[<sched_class>]` first when
/// `sched_class` is `Some`, then `[dead]` when `is_dead`, then
/// each `config_gate` in registry-declared order. Examples:
/// - `nr_wakeups_affine` → `[cfs-only] [SCHEDSTATS]`
/// - `core_forceidle_sum` → `[SCHED_CORE] [SCHEDSTATS]`
/// - `fair_slice_ns` → `[fair-policy]`
///
/// Compact rendering: each `config_gate` is stripped of its
/// `CONFIG_` prefix before emission so the rendered cell stays
/// scannable in narrow tables. The data field
/// [`CtprofMetricDef::config_gates`] keeps the full `CONFIG_X`
/// spelling so an operator can grep their kconfig directly.
/// `sched_class` tags are rendered as-is (already short, e.g.
/// `[cfs-only]`, `[fair-policy]`, `[non-ext]`).
///
/// Pure formatting layer — does not interpret tag values; the
/// metric's own [`CtprofMetricDef::sched_class`] /
/// [`CtprofMetricDef::config_gates`] / [`CtprofMetricDef::is_dead`]
/// docs are the source of truth for what each spelling means.
pub fn metric_tags(metric: &CtprofMetricDef) -> String {
    let mut out = String::new();
    if let Some(class) = metric.sched_class {
        out.push('[');
        out.push_str(class);
        out.push(']');
    }
    if metric.is_dead {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str("[dead]");
    }
    for gate in metric.config_gates {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push('[');
        let short = gate.strip_prefix("CONFIG_").unwrap_or(gate);
        out.push_str(short);
        out.push(']');
    }
    out
}
