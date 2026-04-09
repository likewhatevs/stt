//! Pass/fail evaluation of scenario results.
//!
//! Key types:
//! - [`AssertResult`] -- pass/fail status with diagnostics and statistics
//! - [`Assert`] -- composable assertion config (worker + monitor checks)
//! - `AssertPlan` -- worker-side check configuration (crate-internal)
//! - [`ScenarioStats`] / [`CgroupStats`] -- aggregated telemetry
//!
//! Assertion uses a three-layer merge: [`Assert::default_checks()`] ->
//! `Scheduler.assert` -> per-test `assert`.
//!
//! See the [Verification](https://sched-ext.github.io/scx/stt/concepts/verification.html)
//! chapter of the guide.

use crate::workload::WorkerReport;
use std::collections::{BTreeMap, BTreeSet};

fn gap_threshold_ms() -> u64 {
    // Unoptimized debug builds have higher scheduling overhead.
    if cfg!(debug_assertions) { 3000 } else { 2000 }
}

fn spread_threshold_pct() -> f64 {
    // Debug builds in small VMs (especially under EEVDF) show higher
    // spread than optimized builds under sched_ext schedulers.
    if cfg!(debug_assertions) { 35.0 } else { 15.0 }
}

/// Result of checking a scenario run.
///
/// Contains pass/fail status, human-readable detail messages, and
/// aggregated statistics. Multiple results can be combined with
/// [`merge()`](AssertResult::merge).
///
/// ```
/// # use stt::assert::AssertResult;
/// let mut a = AssertResult::pass();
/// assert!(a.passed);
///
/// let mut b = AssertResult::pass();
/// b.passed = false;
/// b.details.push("worker starved".into());
///
/// a.merge(b);
/// assert!(!a.passed);
/// assert!(a.details.iter().any(|d| d.contains("starved")));
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AssertResult {
    /// Whether all checks passed.
    pub passed: bool,
    /// Human-readable diagnostic messages (failures, warnings).
    pub details: Vec<String>,
    /// Aggregated stats from all workers in this scenario.
    #[serde(default)]
    pub stats: ScenarioStats,
}

/// Per-cgroup statistics from worker telemetry.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct CgroupStats {
    pub num_workers: usize,
    pub num_cpus: usize,
    pub avg_runnable_pct: f64,
    pub min_runnable_pct: f64,
    pub max_runnable_pct: f64,
    pub spread: f64,
    pub max_gap_ms: u64,
    pub max_gap_cpu: usize,
    pub total_migrations: u64,
    /// Migrations per iteration (total_migrations / total_iterations).
    #[serde(default)]
    pub migration_ratio: f64,
    #[serde(default)]
    pub p99_wake_latency_us: f64,
    #[serde(default)]
    pub median_wake_latency_us: f64,
    #[serde(default)]
    pub wake_latency_cv: f64,
    #[serde(default)]
    pub total_iterations: u64,
    #[serde(default)]
    pub mean_run_delay_us: f64,
    #[serde(default)]
    pub worst_run_delay_us: f64,
    /// Extensible metrics for the generic comparison pipeline.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub ext_metrics: BTreeMap<String, f64>,
}

/// Aggregated statistics across all cgroups in a scenario.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ScenarioStats {
    pub cgroups: Vec<CgroupStats>,
    pub total_workers: usize,
    pub total_cpus: usize,
    pub total_migrations: u64,
    /// Worst spread across any cgroup.
    pub worst_spread: f64,
    /// Worst gap across any cgroup (ms).
    pub worst_gap_ms: u64,
    pub worst_gap_cpu: usize,
    /// Worst migration ratio across any cgroup.
    #[serde(default)]
    pub worst_migration_ratio: f64,
    #[serde(default)]
    pub p99_wake_latency_us: f64,
    #[serde(default)]
    pub median_wake_latency_us: f64,
    #[serde(default)]
    pub wake_latency_cv: f64,
    #[serde(default)]
    pub total_iterations: u64,
    #[serde(default)]
    pub mean_run_delay_us: f64,
    #[serde(default)]
    pub worst_run_delay_us: f64,
    /// Extensible metrics for the generic comparison pipeline.
    /// Populated from per-cgroup ext_metrics (worst value across cgroups).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub ext_metrics: BTreeMap<String, f64>,
}

impl AssertResult {
    pub fn pass() -> Self {
        Self {
            passed: true,
            details: vec![],
            stats: Default::default(),
        }
    }
    pub fn merge(&mut self, other: AssertResult) {
        if !other.passed {
            self.passed = false;
        }
        self.details.extend(other.details);
        self.stats.cgroups.extend(other.stats.cgroups);
        self.stats.total_workers += other.stats.total_workers;
        self.stats.total_cpus += other.stats.total_cpus;
        self.stats.total_migrations += other.stats.total_migrations;
        if other.stats.worst_spread > self.stats.worst_spread {
            self.stats.worst_spread = other.stats.worst_spread;
        }
        if other.stats.worst_gap_ms > self.stats.worst_gap_ms {
            self.stats.worst_gap_ms = other.stats.worst_gap_ms;
            self.stats.worst_gap_cpu = other.stats.worst_gap_cpu;
        }
        if other.stats.worst_migration_ratio > self.stats.worst_migration_ratio {
            self.stats.worst_migration_ratio = other.stats.worst_migration_ratio;
        }
        if other.stats.p99_wake_latency_us > self.stats.p99_wake_latency_us {
            self.stats.p99_wake_latency_us = other.stats.p99_wake_latency_us;
        }
        if other.stats.median_wake_latency_us > self.stats.median_wake_latency_us {
            self.stats.median_wake_latency_us = other.stats.median_wake_latency_us;
        }
        if other.stats.wake_latency_cv > self.stats.wake_latency_cv {
            self.stats.wake_latency_cv = other.stats.wake_latency_cv;
        }
        self.stats.total_iterations += other.stats.total_iterations;
        if other.stats.worst_run_delay_us > self.stats.worst_run_delay_us {
            self.stats.worst_run_delay_us = other.stats.worst_run_delay_us;
        }
        // mean_run_delay: take worst across cgroups for scenario-level stat.
        if other.stats.mean_run_delay_us > self.stats.mean_run_delay_us {
            self.stats.mean_run_delay_us = other.stats.mean_run_delay_us;
        }
        // Merge extensible metrics: take worst (max) value per key.
        for (k, v) in &other.stats.ext_metrics {
            let entry = self.stats.ext_metrics.entry(k.clone()).or_insert(0.0);
            if *v > *entry {
                *entry = *v;
            }
        }
    }
}

/// Worker-side assertion plan (crate-internal). Specifies which checks
/// to run on worker reports after collection.
///
/// External users should use [`Assert`] and its `assert_cgroup()` method
/// instead.
#[derive(Clone, Debug)]
pub(crate) struct AssertPlan {
    pub(crate) not_starved: bool,
    pub(crate) isolation: bool,
    pub(crate) max_gap_ms: Option<u64>,
    pub(crate) max_spread_pct: Option<f64>,
    pub(crate) max_throughput_cv: Option<f64>,
    pub(crate) min_work_rate: Option<f64>,
    pub(crate) max_p99_wake_latency_ns: Option<u64>,
    pub(crate) max_wake_latency_cv: Option<f64>,
    pub(crate) min_iteration_rate: Option<f64>,
    pub(crate) max_migration_ratio: Option<f64>,
}

impl AssertPlan {
    pub(crate) fn new() -> Self {
        Self {
            not_starved: false,
            isolation: false,
            max_gap_ms: None,
            max_spread_pct: None,
            max_throughput_cv: None,
            min_work_rate: None,
            max_p99_wake_latency_ns: None,
            max_wake_latency_cv: None,
            min_iteration_rate: None,
            max_migration_ratio: None,
        }
    }

    /// Run all configured checks against one cgroup's reports.
    ///
    /// `cpuset` is the expected CPU set for isolation checks. Pass `None`
    /// when there is no cpuset constraint (isolation check is skipped).
    ///
    pub(crate) fn assert_cgroup(
        &self,
        reports: &[WorkerReport],
        cpuset: Option<&BTreeSet<usize>>,
    ) -> AssertResult {
        let mut r = AssertResult::pass();
        if self.not_starved {
            let mut cgroup_result = assert_not_starved(reports);
            // Apply custom spread threshold if set.
            if let Some(spread_limit) = self.max_spread_pct {
                // Re-check spread against custom threshold. The default
                // assert_not_starved uses spread_threshold_pct(); clear
                // those failures and re-evaluate.
                cgroup_result.details.retain(|d| !d.contains("unfair"));
                if let Some(cg) = cgroup_result.stats.cgroups.first() {
                    if cg.spread > spread_limit && cg.num_workers >= 2 {
                        cgroup_result.passed = false;
                        cgroup_result.details.push(format!(
                            "unfair cgroup: spread={:.0}% ({:.0}-{:.0}%) {} workers on {} cpus (threshold {:.0}%)",
                            cg.spread, cg.min_runnable_pct, cg.max_runnable_pct,
                            cg.num_workers, cg.num_cpus, spread_limit
                        ));
                    } else {
                        // Re-derive passed: only non-spread failures matter.
                        cgroup_result.passed = !cgroup_result
                            .details
                            .iter()
                            .any(|d| d.contains("starved") || d.contains("stuck"));
                    }
                }
            }
            // Apply custom gap threshold if set.
            if let Some(threshold) = self.max_gap_ms {
                // Re-check gaps against custom threshold. The default
                // assert_not_starved uses 2000ms; clear those failures
                // and re-evaluate.
                cgroup_result.details.retain(|d| !d.contains("stuck"));
                let had_gap_failure = reports.iter().any(|w| w.max_gap_ms > threshold);
                if had_gap_failure {
                    cgroup_result.passed = false;
                    for w in reports {
                        if w.max_gap_ms > threshold {
                            cgroup_result.details.push(format!(
                                "stuck {}ms on cpu{} at +{}ms (threshold {}ms)",
                                w.max_gap_ms, w.max_gap_cpu, w.max_gap_at_ms, threshold
                            ));
                        }
                    }
                } else {
                    // Re-derive passed: only non-gap failures matter.
                    cgroup_result.passed = !cgroup_result
                        .details
                        .iter()
                        .any(|d| d.contains("starved") || d.contains("unfair"));
                }
            }
            r.merge(cgroup_result);
        }
        if self.isolation
            && let Some(cs) = cpuset
        {
            r.merge(assert_isolation(reports, cs));
        }
        if self.max_throughput_cv.is_some() || self.min_work_rate.is_some() {
            r.merge(assert_throughput_parity(
                reports,
                self.max_throughput_cv,
                self.min_work_rate,
            ));
        }
        if self.max_p99_wake_latency_ns.is_some()
            || self.max_wake_latency_cv.is_some()
            || self.min_iteration_rate.is_some()
        {
            r.merge(assert_benchmarks(
                reports,
                self.max_p99_wake_latency_ns,
                self.max_wake_latency_cv,
                self.min_iteration_rate,
            ));
        }
        if let Some(max_ratio) = self.max_migration_ratio {
            let total_mig: u64 = reports.iter().map(|w| w.migration_count).sum();
            let total_iters: u64 = reports.iter().map(|w| w.iterations).sum();
            let ratio = if total_iters > 0 {
                total_mig as f64 / total_iters as f64
            } else {
                0.0
            };
            if ratio > max_ratio {
                r.passed = false;
                r.details.push(format!(
                    "migration ratio {:.4} exceeds threshold {:.4} ({} migrations / {} iterations)",
                    ratio, max_ratio, total_mig, total_iters,
                ));
            }
        }
        r
    }
}

impl Default for AssertPlan {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
impl AssertPlan {
    fn check_not_starved(mut self) -> Self {
        self.not_starved = true;
        self
    }

    fn check_isolation(mut self) -> Self {
        self.isolation = true;
        self
    }

    fn max_gap_ms(mut self, ms: u64) -> Self {
        self.max_gap_ms = Some(ms);
        self
    }
}

/// Unified assertion configuration. Carries both worker checks and
/// monitor thresholds as a single composable type. Each `Option` field
/// acts as an override — `None` means "inherit from parent layer".
///
/// Merge order: `Assert::default_checks()` -> `Scheduler.assert` -> per-test `assert`.
///
/// ```
/// # use stt::assert::Assert;
/// // Start from defaults, override imbalance threshold.
/// let sched_assert = Assert::NONE.max_imbalance_ratio(5.0);
///
/// // Merge: defaults <- scheduler <- test.
/// let merged = Assert::default_checks()
///     .merge(&sched_assert)
///     .merge(&Assert::NONE.max_gap_ms(5000));
///
/// assert_eq!(merged.not_starved, Some(true));   // from default_checks
/// assert_eq!(merged.max_imbalance_ratio, Some(5.0)); // from sched
/// assert_eq!(merged.max_gap_ms, Some(5000));    // from test
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Assert {
    // Worker checks
    pub not_starved: Option<bool>,
    pub isolation: Option<bool>,
    pub max_gap_ms: Option<u64>,
    pub max_spread_pct: Option<f64>,

    // Throughput checks
    /// Max coefficient of variation for work_units/cpu_time across workers.
    /// Catches placement unfairness where some workers get less CPU than others.
    pub max_throughput_cv: Option<f64>,
    /// Minimum work_units per CPU-second. Catches cases where all workers
    /// are equally slow (CV passes but absolute throughput is too low).
    pub min_work_rate: Option<f64>,

    // Benchmarking checks
    /// Max p99 wake latency (ns). Fails if any cgroup's p99 exceeds this.
    pub max_p99_wake_latency_ns: Option<u64>,
    /// Max wake latency coefficient of variation. Fails if CV exceeds this.
    pub max_wake_latency_cv: Option<f64>,
    /// Minimum iterations per wall-clock second. Fails if any worker is below.
    pub min_iteration_rate: Option<f64>,
    /// Max migration ratio (migrations/iterations). Fails if any cgroup exceeds this.
    pub max_migration_ratio: Option<f64>,

    // Monitor checks
    pub max_imbalance_ratio: Option<f64>,
    pub max_local_dsq_depth: Option<u32>,
    pub fail_on_stall: Option<bool>,
    pub sustained_samples: Option<usize>,
    pub max_fallback_rate: Option<f64>,
    pub max_keep_last_rate: Option<f64>,
}

impl Assert {
    /// Empty assert — no checks enabled, all overrides None.
    pub const NONE: Assert = Assert {
        not_starved: None,
        isolation: None,
        max_gap_ms: None,
        max_spread_pct: None,
        max_throughput_cv: None,
        min_work_rate: None,
        max_p99_wake_latency_ns: None,
        max_wake_latency_cv: None,
        min_iteration_rate: None,
        max_migration_ratio: None,
        max_imbalance_ratio: None,
        max_local_dsq_depth: None,
        fail_on_stall: None,
        sustained_samples: None,
        max_fallback_rate: None,
        max_keep_last_rate: None,
    };

    /// Default checks: not_starved enabled, monitor thresholds
    /// (imbalance 4.0, dsq_depth 50, stall on, sustained 5,
    /// fallback 200.0, keep_last 100.0).
    pub const fn default_checks() -> Assert {
        use crate::monitor::MonitorThresholds;
        Assert {
            not_starved: Some(true),
            isolation: None,
            max_gap_ms: None,
            max_spread_pct: None,
            max_throughput_cv: None,
            min_work_rate: None,
            max_p99_wake_latency_ns: None,
            max_wake_latency_cv: None,
            min_iteration_rate: None,
            max_migration_ratio: None,
            max_imbalance_ratio: Some(MonitorThresholds::DEFAULT.max_imbalance_ratio),
            max_local_dsq_depth: Some(MonitorThresholds::DEFAULT.max_local_dsq_depth),
            fail_on_stall: Some(MonitorThresholds::DEFAULT.fail_on_stall),
            sustained_samples: Some(MonitorThresholds::DEFAULT.sustained_samples),
            max_fallback_rate: Some(MonitorThresholds::DEFAULT.max_fallback_rate),
            max_keep_last_rate: Some(MonitorThresholds::DEFAULT.max_keep_last_rate),
        }
    }

    pub const fn check_not_starved(mut self) -> Self {
        self.not_starved = Some(true);
        self
    }

    pub const fn check_isolation(mut self) -> Self {
        self.isolation = Some(true);
        self
    }

    pub const fn max_gap_ms(mut self, ms: u64) -> Self {
        self.max_gap_ms = Some(ms);
        self
    }

    pub const fn max_spread_pct(mut self, pct: f64) -> Self {
        self.max_spread_pct = Some(pct);
        self
    }

    pub const fn max_throughput_cv(mut self, v: f64) -> Self {
        self.max_throughput_cv = Some(v);
        self
    }

    pub const fn min_work_rate(mut self, v: f64) -> Self {
        self.min_work_rate = Some(v);
        self
    }

    pub const fn max_p99_wake_latency_ns(mut self, v: u64) -> Self {
        self.max_p99_wake_latency_ns = Some(v);
        self
    }

    pub const fn max_wake_latency_cv(mut self, v: f64) -> Self {
        self.max_wake_latency_cv = Some(v);
        self
    }

    pub const fn min_iteration_rate(mut self, v: f64) -> Self {
        self.min_iteration_rate = Some(v);
        self
    }

    pub const fn max_migration_ratio(mut self, v: f64) -> Self {
        self.max_migration_ratio = Some(v);
        self
    }

    pub const fn max_imbalance_ratio(mut self, v: f64) -> Self {
        self.max_imbalance_ratio = Some(v);
        self
    }

    pub const fn max_local_dsq_depth(mut self, v: u32) -> Self {
        self.max_local_dsq_depth = Some(v);
        self
    }

    pub const fn fail_on_stall(mut self, v: bool) -> Self {
        self.fail_on_stall = Some(v);
        self
    }

    pub const fn sustained_samples(mut self, v: usize) -> Self {
        self.sustained_samples = Some(v);
        self
    }

    pub const fn max_fallback_rate(mut self, v: f64) -> Self {
        self.max_fallback_rate = Some(v);
        self
    }

    pub const fn max_keep_last_rate(mut self, v: f64) -> Self {
        self.max_keep_last_rate = Some(v);
        self
    }

    /// True when any worker-level check field is `Some`.
    pub const fn has_worker_checks(&self) -> bool {
        self.not_starved.is_some()
            || self.isolation.is_some()
            || self.max_gap_ms.is_some()
            || self.max_spread_pct.is_some()
            || self.max_throughput_cv.is_some()
            || self.min_work_rate.is_some()
            || self.max_p99_wake_latency_ns.is_some()
            || self.max_wake_latency_cv.is_some()
            || self.min_iteration_rate.is_some()
            || self.max_migration_ratio.is_some()
    }

    /// Merge `other` on top of `self`. Each `Some` field in `other`
    /// overrides the corresponding field in `self`; `None` fields
    /// inherit from `self`.
    ///
    /// Use when composing scheduler-level and test-level overrides:
    /// `Assert::default_checks().merge(&scheduler.assert).merge(&test.assert)`.
    pub const fn merge(&self, other: &Assert) -> Assert {
        Assert {
            not_starved: match other.not_starved {
                Some(v) => Some(v),
                None => self.not_starved,
            },
            isolation: match other.isolation {
                Some(v) => Some(v),
                None => self.isolation,
            },
            max_gap_ms: match other.max_gap_ms {
                Some(v) => Some(v),
                None => self.max_gap_ms,
            },
            max_spread_pct: match other.max_spread_pct {
                Some(v) => Some(v),
                None => self.max_spread_pct,
            },
            max_throughput_cv: match other.max_throughput_cv {
                Some(v) => Some(v),
                None => self.max_throughput_cv,
            },
            min_work_rate: match other.min_work_rate {
                Some(v) => Some(v),
                None => self.min_work_rate,
            },
            max_p99_wake_latency_ns: match other.max_p99_wake_latency_ns {
                Some(v) => Some(v),
                None => self.max_p99_wake_latency_ns,
            },
            max_wake_latency_cv: match other.max_wake_latency_cv {
                Some(v) => Some(v),
                None => self.max_wake_latency_cv,
            },
            min_iteration_rate: match other.min_iteration_rate {
                Some(v) => Some(v),
                None => self.min_iteration_rate,
            },
            max_migration_ratio: match other.max_migration_ratio {
                Some(v) => Some(v),
                None => self.max_migration_ratio,
            },
            max_imbalance_ratio: match other.max_imbalance_ratio {
                Some(v) => Some(v),
                None => self.max_imbalance_ratio,
            },
            max_local_dsq_depth: match other.max_local_dsq_depth {
                Some(v) => Some(v),
                None => self.max_local_dsq_depth,
            },
            fail_on_stall: match other.fail_on_stall {
                Some(v) => Some(v),
                None => self.fail_on_stall,
            },
            sustained_samples: match other.sustained_samples {
                Some(v) => Some(v),
                None => self.sustained_samples,
            },
            max_fallback_rate: match other.max_fallback_rate {
                Some(v) => Some(v),
                None => self.max_fallback_rate,
            },
            max_keep_last_rate: match other.max_keep_last_rate {
                Some(v) => Some(v),
                None => self.max_keep_last_rate,
            },
        }
    }

    /// Extract an `AssertPlan` for worker-side checks.
    pub(crate) fn worker_plan(&self) -> AssertPlan {
        AssertPlan {
            not_starved: self.not_starved.unwrap_or(false),
            isolation: self.isolation.unwrap_or(false),
            max_gap_ms: self.max_gap_ms,
            max_spread_pct: self.max_spread_pct,
            max_throughput_cv: self.max_throughput_cv,
            min_work_rate: self.min_work_rate,
            max_p99_wake_latency_ns: self.max_p99_wake_latency_ns,
            max_wake_latency_cv: self.max_wake_latency_cv,
            min_iteration_rate: self.min_iteration_rate,
            max_migration_ratio: self.max_migration_ratio,
        }
    }

    /// Run the configured worker checks against one cgroup's reports.
    pub fn assert_cgroup(
        &self,
        reports: &[crate::workload::WorkerReport],
        cpuset: Option<&BTreeSet<usize>>,
    ) -> AssertResult {
        self.worker_plan().assert_cgroup(reports, cpuset)
    }

    /// Extract `MonitorThresholds` for monitor-side evaluation.
    pub(crate) fn monitor_thresholds(&self) -> crate::monitor::MonitorThresholds {
        use crate::monitor::MonitorThresholds;
        let d = MonitorThresholds::DEFAULT;
        MonitorThresholds {
            max_imbalance_ratio: self.max_imbalance_ratio.unwrap_or(d.max_imbalance_ratio),
            max_local_dsq_depth: self.max_local_dsq_depth.unwrap_or(d.max_local_dsq_depth),
            fail_on_stall: self.fail_on_stall.unwrap_or(d.fail_on_stall),
            sustained_samples: self.sustained_samples.unwrap_or(d.sustained_samples),
            max_fallback_rate: self.max_fallback_rate.unwrap_or(d.max_fallback_rate),
            max_keep_last_rate: self.max_keep_last_rate.unwrap_or(d.max_keep_last_rate),
        }
    }
}

/// Check that workers only ran on CPUs in `expected`.
///
/// Any worker that used a CPU outside the expected set produces a
/// failure with the unexpected CPU IDs listed.
///
/// ```
/// # use stt::assert::assert_isolation;
/// # use stt::workload::WorkerReport;
/// # use std::collections::BTreeSet;
/// # let report = WorkerReport {
/// #     tid: 1, cpus_used: [0, 1].into_iter().collect(),
/// #     work_units: 100, cpu_time_ns: 1_000_000, wall_time_ns: 2_000_000,
/// #     runnable_ns: 1_000_000, migration_count: 0, migrations: vec![],
/// #     max_gap_ms: 0, max_gap_cpu: 0, max_gap_at_ms: 0,
/// #     wake_latencies_ns: vec![], iterations: 0,
/// #     schedstat_run_delay_ns: 0, schedstat_ctx_switches: 0,
/// #     schedstat_cpu_time_ns: 0,
/// # };
/// let expected: BTreeSet<usize> = [0, 1, 2].into_iter().collect();
/// assert!(assert_isolation(&[report], &expected).passed);
/// ```
pub fn assert_isolation(reports: &[WorkerReport], expected: &BTreeSet<usize>) -> AssertResult {
    let mut r = AssertResult::pass();
    for w in reports {
        let bad: BTreeSet<usize> = w.cpus_used.difference(expected).copied().collect();
        if !bad.is_empty() {
            r.passed = false;
            r.details
                .push(format!("tid {} ran on unexpected CPUs {:?}", w.tid, bad));
        }
    }
    r
}

/// Check one cgroup's workers. Returns per-cgroup stats.
///
/// ```
/// # use stt::assert::assert_not_starved;
/// # use stt::workload::WorkerReport;
/// # let report = WorkerReport {
/// #     tid: 1, cpus_used: [0].into_iter().collect(),
/// #     work_units: 100, cpu_time_ns: 1_000_000, wall_time_ns: 5_000_000_000,
/// #     runnable_ns: 500_000_000, migration_count: 0, migrations: vec![],
/// #     max_gap_ms: 50, max_gap_cpu: 0, max_gap_at_ms: 1000,
/// #     wake_latencies_ns: vec![], iterations: 0,
/// #     schedstat_run_delay_ns: 0, schedstat_ctx_switches: 0,
/// #     schedstat_cpu_time_ns: 0,
/// # };
/// let r = assert_not_starved(&[report]);
/// assert!(r.passed);
/// assert_eq!(r.stats.total_workers, 1);
/// ```
pub fn assert_not_starved(reports: &[WorkerReport]) -> AssertResult {
    let mut r = AssertResult::pass();
    if reports.is_empty() {
        return r;
    }

    let cpus: BTreeSet<usize> = reports
        .iter()
        .flat_map(|w| w.cpus_used.iter().copied())
        .collect();
    let mut pcts: Vec<f64> = Vec::new();

    for w in reports {
        if w.work_units == 0 {
            r.passed = false;
            r.details
                .push(format!("tid {} starved (0 work units)", w.tid));
        }
        if w.wall_time_ns > 0 {
            pcts.push(w.runnable_ns as f64 / w.wall_time_ns as f64 * 100.0);
        }
    }

    let min = pcts.iter().cloned().reduce(f64::min).unwrap_or(0.0);
    let max = pcts.iter().cloned().reduce(f64::max).unwrap_or(0.0);
    let avg = if pcts.is_empty() {
        0.0
    } else {
        pcts.iter().sum::<f64>() / pcts.len() as f64
    };
    let spread = max - min;

    let worst_gap = reports.iter().max_by_key(|w| w.max_gap_ms);
    let (gap_ms, gap_cpu) = worst_gap
        .map(|w| (w.max_gap_ms, w.max_gap_cpu))
        .unwrap_or((0, 0));

    // Compute benchmarking stats from worker reports.
    let all_latencies: Vec<u64> = reports
        .iter()
        .flat_map(|w| w.wake_latencies_ns.iter().copied())
        .collect();
    let (p99_us, median_us, lat_cv) = if all_latencies.is_empty() {
        (0.0, 0.0, 0.0)
    } else {
        let mut sorted = all_latencies.clone();
        sorted.sort_unstable();
        let p99_idx = (sorted.len() as f64 * 0.99).ceil() as usize;
        let p99 = sorted[p99_idx.min(sorted.len() - 1)] as f64 / 1000.0;
        let median = sorted[sorted.len() / 2] as f64 / 1000.0;
        let n = all_latencies.len() as f64;
        let mean_ns = all_latencies.iter().sum::<u64>() as f64 / n;
        let cv = if mean_ns > 0.0 {
            let variance = all_latencies
                .iter()
                .map(|&v| (v as f64 - mean_ns).powi(2))
                .sum::<f64>()
                / n;
            variance.sqrt() / mean_ns
        } else {
            0.0
        };
        (p99, median, cv)
    };

    let total_iters: u64 = reports.iter().map(|w| w.iterations).sum();
    let run_delays: Vec<f64> = reports
        .iter()
        .map(|w| w.schedstat_run_delay_ns as f64 / 1000.0)
        .collect();
    let mean_run_delay = if run_delays.is_empty() {
        0.0
    } else {
        run_delays.iter().sum::<f64>() / run_delays.len() as f64
    };
    let worst_run_delay = run_delays.iter().cloned().reduce(f64::max).unwrap_or(0.0);

    let total_mig: u64 = reports.iter().map(|w| w.migration_count).sum();
    let mig_ratio = if total_iters > 0 {
        total_mig as f64 / total_iters as f64
    } else {
        0.0
    };

    let cg = CgroupStats {
        num_workers: reports.len(),
        num_cpus: cpus.len(),
        avg_runnable_pct: avg,
        min_runnable_pct: min,
        max_runnable_pct: max,
        spread,
        max_gap_ms: gap_ms,
        max_gap_cpu: gap_cpu,
        total_migrations: total_mig,
        migration_ratio: mig_ratio,
        p99_wake_latency_us: p99_us,
        median_wake_latency_us: median_us,
        wake_latency_cv: lat_cv,
        total_iterations: total_iters,
        mean_run_delay_us: mean_run_delay,
        worst_run_delay_us: worst_run_delay,
        ext_metrics: BTreeMap::new(),
    };

    // Per-cgroup fairness: spread above threshold means unequal scheduling within a cgroup
    let spread_limit = spread_threshold_pct();
    if spread > spread_limit && pcts.len() >= 2 {
        r.passed = false;
        r.details.push(format!(
            "unfair cgroup: spread={:.0}% ({:.0}-{:.0}%) {} workers on {} cpus",
            spread,
            min,
            max,
            reports.len(),
            cpus.len(),
        ));
    }

    // Scheduling gap: >threshold = dispatch failure
    let gap_limit = gap_threshold_ms();
    for w in reports {
        if w.max_gap_ms > gap_limit {
            r.passed = false;
            r.details.push(format!(
                "stuck {}ms on cpu{} at +{}ms",
                w.max_gap_ms, w.max_gap_cpu, w.max_gap_at_ms
            ));
        }
    }

    // Store this cgroup's stats - merge accumulates cgroups
    r.stats = ScenarioStats {
        total_workers: reports.len(),
        total_cpus: cpus.len(),
        total_migrations: reports.iter().map(|w| w.migration_count).sum(),
        worst_spread: spread,
        worst_gap_ms: gap_ms,
        worst_gap_cpu: gap_cpu,
        worst_migration_ratio: cg.migration_ratio,
        p99_wake_latency_us: cg.p99_wake_latency_us,
        median_wake_latency_us: cg.median_wake_latency_us,
        wake_latency_cv: cg.wake_latency_cv,
        total_iterations: cg.total_iterations,
        mean_run_delay_us: cg.mean_run_delay_us,
        worst_run_delay_us: cg.worst_run_delay_us,
        ext_metrics: cg.ext_metrics.clone(),
        cgroups: vec![cg],
    };

    r
}

/// Check throughput parity across workers: coefficient of variation and
/// minimum work rate.
///
/// `max_cv`: maximum allowed coefficient of variation (stddev/mean) for
/// work_units / cpu_time_ns across workers. `None` skips the CV check.
///
/// `min_rate`: minimum work_units per CPU-second. `None` skips the floor check.
///
/// ```
/// # use stt::assert::assert_throughput_parity;
/// # use stt::workload::WorkerReport;
/// # let mk = |units, cpu_ns| WorkerReport {
/// #     tid: 1, cpus_used: [0].into_iter().collect(),
/// #     work_units: units, cpu_time_ns: cpu_ns, wall_time_ns: cpu_ns,
/// #     runnable_ns: cpu_ns, migration_count: 0, migrations: vec![],
/// #     max_gap_ms: 0, max_gap_cpu: 0, max_gap_at_ms: 0,
/// #     wake_latencies_ns: vec![], iterations: 0,
/// #     schedstat_run_delay_ns: 0, schedstat_ctx_switches: 0,
/// #     schedstat_cpu_time_ns: 0,
/// # };
/// // Equal throughput -> low CV -> passes.
/// let reports = [mk(1000, 1_000_000_000), mk(1000, 1_000_000_000)];
/// assert!(assert_throughput_parity(&reports, Some(0.5), None).passed);
/// ```
pub fn assert_throughput_parity(
    reports: &[WorkerReport],
    max_cv: Option<f64>,
    min_rate: Option<f64>,
) -> AssertResult {
    let mut r = AssertResult::pass();
    if reports.is_empty() {
        return r;
    }

    // Compute per-worker throughput: work_units / cpu_seconds
    let rates: Vec<f64> = reports
        .iter()
        .map(|w| {
            if w.cpu_time_ns == 0 {
                0.0
            } else {
                w.work_units as f64 / (w.cpu_time_ns as f64 / 1e9)
            }
        })
        .collect();

    let n = rates.len() as f64;
    let mean = rates.iter().sum::<f64>() / n;

    if let Some(cv_limit) = max_cv
        && mean > 0.0
        && rates.len() >= 2
    {
        let variance = rates.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let stddev = variance.sqrt();
        let cv = stddev / mean;
        if cv > cv_limit {
            r.passed = false;
            r.details.push(format!(
                "throughput CV {cv:.3} exceeds limit {cv_limit:.3} (mean={mean:.0} work/cpu_s)"
            ));
        }
    }

    if let Some(floor) = min_rate {
        for (i, &rate) in rates.iter().enumerate() {
            if rate < floor {
                r.passed = false;
                r.details.push(format!(
                    "worker {} throughput {rate:.0} work/cpu_s below floor {floor:.0}",
                    reports[i].tid
                ));
            }
        }
    }

    r
}

/// Check benchmarking metrics: p99 wake latency, wake latency CV,
/// and minimum iteration rate.
///
/// ```
/// # use stt::assert::assert_benchmarks;
/// # use stt::workload::WorkerReport;
/// # let report = WorkerReport {
/// #     tid: 1, cpus_used: [0].into_iter().collect(),
/// #     work_units: 1000, cpu_time_ns: 2_500_000_000,
/// #     wall_time_ns: 5_000_000_000, runnable_ns: 2_500_000_000,
/// #     migration_count: 0, migrations: vec![],
/// #     max_gap_ms: 50, max_gap_cpu: 0, max_gap_at_ms: 1000,
/// #     wake_latencies_ns: vec![100, 200, 300, 400, 500],
/// #     iterations: 1000,
/// #     schedstat_run_delay_ns: 0, schedstat_ctx_switches: 0,
/// #     schedstat_cpu_time_ns: 0,
/// # };
/// // p99 = 500ns, well under 10000ns limit.
/// assert!(assert_benchmarks(&[report], Some(10000), None, None).passed);
/// ```
pub fn assert_benchmarks(
    reports: &[WorkerReport],
    max_p99_ns: Option<u64>,
    max_cv: Option<f64>,
    min_iter_rate: Option<f64>,
) -> AssertResult {
    let mut r = AssertResult::pass();
    if reports.is_empty() {
        return r;
    }

    // Collect all wake latencies across workers.
    let all_latencies: Vec<u64> = reports
        .iter()
        .flat_map(|w| w.wake_latencies_ns.iter().copied())
        .collect();

    if let Some(p99_limit) = max_p99_ns
        && !all_latencies.is_empty()
    {
        let mut sorted = all_latencies.clone();
        sorted.sort_unstable();
        let p99_idx = (sorted.len() as f64 * 0.99).ceil() as usize;
        let p99 = sorted[p99_idx.min(sorted.len() - 1)];
        if p99 > p99_limit {
            r.passed = false;
            r.details.push(format!(
                "p99 wake latency {p99}ns exceeds limit {p99_limit}ns ({} samples)",
                sorted.len()
            ));
        }
    }

    if let Some(cv_limit) = max_cv
        && all_latencies.len() >= 2
    {
        let n = all_latencies.len() as f64;
        let mean = all_latencies.iter().sum::<u64>() as f64 / n;
        if mean > 0.0 {
            let variance = all_latencies
                .iter()
                .map(|&v| (v as f64 - mean).powi(2))
                .sum::<f64>()
                / n;
            let cv = variance.sqrt() / mean;
            if cv > cv_limit {
                r.passed = false;
                r.details.push(format!(
                    "wake latency CV {cv:.3} exceeds limit {cv_limit:.3} (mean={mean:.0}ns)"
                ));
            }
        }
    }

    if let Some(rate_floor) = min_iter_rate {
        for w in reports {
            if w.wall_time_ns == 0 {
                continue;
            }
            let rate = w.iterations as f64 / (w.wall_time_ns as f64 / 1e9);
            if rate < rate_floor {
                r.passed = false;
                r.details.push(format!(
                    "worker {} iteration rate {rate:.1}/s below floor {rate_floor:.1}/s",
                    w.tid
                ));
            }
        }
    }

    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workload::WorkerReport;

    fn rpt(
        tid: u32,
        work: u64,
        wall_ns: u64,
        run_ns: u64,
        cpus: &[usize],
        gap_ms: u64,
    ) -> WorkerReport {
        WorkerReport {
            tid,
            work_units: work,
            cpu_time_ns: wall_ns.saturating_sub(run_ns),
            wall_time_ns: wall_ns,
            runnable_ns: run_ns,
            migration_count: 0,
            cpus_used: cpus.iter().copied().collect(),
            migrations: vec![],
            max_gap_ms: gap_ms,
            max_gap_cpu: cpus.first().copied().unwrap_or(0),
            max_gap_at_ms: 1000,
            wake_latencies_ns: vec![],
            iterations: 0,
            schedstat_run_delay_ns: 0,
            schedstat_ctx_switches: 0,
            schedstat_cpu_time_ns: 0,
        }
    }

    #[test]
    fn healthy_pass() {
        let r = assert_not_starved(&[
            rpt(1, 1000, 5_000_000_000, 500_000_000, &[0, 1], 50),
            rpt(2, 1000, 5_000_000_000, 600_000_000, &[0, 1], 60),
            rpt(3, 1000, 5_000_000_000, 550_000_000, &[0, 1], 45),
        ]);
        assert!(r.passed, "{:?}", r.details);
    }

    #[test]
    fn starved_fail() {
        let r = assert_not_starved(&[
            rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50),
            rpt(2, 0, 5e9 as u64, 5e9 as u64, &[0], 50),
        ]);
        assert!(!r.passed);
        assert!(r.details.iter().any(|d| d.contains("starved")));
    }

    #[test]
    fn unfair_spread_fail() {
        let r = assert_not_starved(&[
            rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50), // 10%
            rpt(2, 500, 5e9 as u64, 4e9 as u64, &[0, 1], 50),  // 80%
            rpt(3, 800, 5e9 as u64, 2e9 as u64, &[0, 1], 50),  // 40%
        ]);
        assert!(!r.passed);
        assert!(r.details.iter().any(|d| d.contains("unfair")));
    }

    #[test]
    fn fair_oversubscribed_pass() {
        let r = assert_not_starved(&[
            rpt(1, 100, 5e9 as u64, (3.75e9) as u64, &[0], 50),
            rpt(2, 100, 5e9 as u64, (3.70e9) as u64, &[0], 50),
            rpt(3, 100, 5e9 as u64, (3.80e9) as u64, &[0], 50),
            rpt(4, 100, 5e9 as u64, (3.75e9) as u64, &[0], 50),
        ]);
        assert!(r.passed, "{:?}", r.details);
    }

    #[test]
    fn stuck_fail() {
        let threshold = gap_threshold_ms();
        let r = assert_not_starved(&[
            rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50),
            rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[0], threshold + 500),
        ]);
        assert!(!r.passed);
        assert!(r.details.iter().any(|d| d.contains("stuck")));
    }

    #[test]
    fn isolation_pass() {
        let expected: BTreeSet<usize> = [0, 1, 2, 3].into_iter().collect();
        let r = assert_isolation(
            &[
                rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50),
                rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[2, 3], 50),
            ],
            &expected,
        );
        assert!(r.passed);
    }

    #[test]
    fn isolation_fail() {
        let expected: BTreeSet<usize> = [0, 1].into_iter().collect();
        let r = assert_isolation(
            &[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1, 4], 50)],
            &expected,
        );
        assert!(!r.passed);
        assert!(r.details.iter().any(|d| d.contains("unexpected")));
    }

    #[test]
    fn merge_cgroups() {
        let r1 = assert_not_starved(&[
            rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50),
            rpt(2, 1000, 5e9 as u64, 6e8 as u64, &[0, 1], 60),
        ]);
        let r2 = assert_not_starved(&[
            rpt(3, 1000, 5e9 as u64, 25e8 as u64, &[2, 3], 50),
            rpt(4, 1000, 5e9 as u64, 26e8 as u64, &[2, 3], 50),
        ]);
        let mut m = r1;
        m.merge(r2);
        assert_eq!(m.stats.cgroups.len(), 2);
        assert_eq!(m.stats.total_workers, 4);
        assert!(m.passed, "diff cgroups diff runnable should pass");
    }

    #[test]
    fn spread_boundary() {
        let threshold = spread_threshold_pct();
        // At threshold exactly - pass
        // Worker 1: 10% runnable, Worker 2: 10%+threshold runnable
        let at_threshold_ns = ((10.0 + threshold) / 100.0 * 5e9) as u64;
        let r = assert_not_starved(&[
            rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50), // 10%
            rpt(2, 1000, 5e9 as u64, at_threshold_ns, &[0], 50), // 10% + threshold
        ]);
        assert!(
            r.passed,
            "{threshold}% spread at threshold: {:?}",
            r.details
        );
        // Above threshold - fail
        let above_ns = ((15.0 + threshold) / 100.0 * 5e9) as u64;
        let r = assert_not_starved(&[
            rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50), // 10%
            rpt(2, 1000, 5e9 as u64, above_ns, &[0], 50),   // 10% + threshold + 5%
        ]);
        assert!(!r.passed, "spread above {threshold}% should fail");
    }

    #[test]
    fn empty_pass() {
        assert!(assert_not_starved(&[]).passed);
    }

    #[test]
    fn zero_wall_time() {
        let r = assert_not_starved(&[
            rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50),
            rpt(2, 0, 0, 0, &[], 0),
        ]);
        assert!(!r.passed);
        assert!(r.details.iter().any(|d| d.contains("starved")));
    }

    #[test]
    fn single_worker_always_pass() {
        let r = assert_not_starved(&[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50)]);
        assert!(r.passed);
        assert_eq!(r.stats.total_workers, 1);
        assert_eq!(r.stats.cgroups.len(), 1);
    }

    #[test]
    fn stats_accuracy() {
        let r = assert_not_starved(&[
            rpt(1, 1000, 5e9 as u64, 1e9 as u64, &[0], 50),  // 20%
            rpt(2, 1000, 5e9 as u64, 15e8 as u64, &[1], 60), // 30%
        ]);
        assert!(r.passed); // spread = 10% < 15%
        let c = &r.stats.cgroups[0];
        assert_eq!(c.num_workers, 2);
        assert_eq!(c.num_cpus, 2);
        assert!((c.min_runnable_pct - 20.0).abs() < 0.1);
        assert!((c.max_runnable_pct - 30.0).abs() < 0.1);
        assert!((c.spread - 10.0).abs() < 0.1);
        assert!((c.avg_runnable_pct - 25.0).abs() < 0.1);
    }

    #[test]
    fn merge_takes_worst_gap() {
        let r1 = assert_not_starved(&[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 100)]);
        let r2 = assert_not_starved(&[rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[1], 500)]);
        let mut m = r1;
        m.merge(r2);
        assert_eq!(m.stats.worst_gap_ms, 500);
        assert_eq!(m.stats.worst_gap_cpu, 1);
    }

    #[test]
    fn merge_takes_worst_spread() {
        let r1 = assert_not_starved(&[
            rpt(1, 1000, 5e9 as u64, 1e9 as u64, &[0], 50),
            rpt(2, 1000, 5e9 as u64, 12e8 as u64, &[0], 50),
        ]); // spread = 4%
        let r2 = assert_not_starved(&[
            rpt(3, 1000, 5e9 as u64, 1e9 as u64, &[1], 50),
            rpt(4, 1000, 5e9 as u64, 15e8 as u64, &[1], 50),
        ]); // spread = 10%
        let mut m = r1;
        m.merge(r2);
        assert!((m.stats.worst_spread - 10.0).abs() < 0.1);
    }

    #[test]
    fn merge_accumulates_totals() {
        let r1 = assert_not_starved(&[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50)]);
        let r2 = assert_not_starved(&[rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[1], 50)]);
        let mut m = r1;
        m.merge(r2);
        assert_eq!(m.stats.total_workers, 2);
        assert_eq!(m.stats.total_cpus, 2);
    }

    #[test]
    fn isolation_empty_reports() {
        let expected: BTreeSet<usize> = [0, 1].into_iter().collect();
        assert!(assert_isolation(&[], &expected).passed);
    }

    #[test]
    fn gap_boundary_at_threshold_pass() {
        let threshold = gap_threshold_ms();
        let r = assert_not_starved(&[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], threshold)]);
        assert!(r.passed, "gap at threshold should pass: {:?}", r.details);
    }

    #[test]
    fn gap_boundary_above_threshold_fail() {
        let threshold = gap_threshold_ms();
        let r = assert_not_starved(&[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], threshold + 1)]);
        assert!(!r.passed);
        assert!(r.details.iter().any(|d| d.contains("stuck")));
    }

    #[test]
    fn scenario_stats_serde_roundtrip() {
        let s = ScenarioStats {
            cgroups: vec![CgroupStats {
                num_workers: 4,
                num_cpus: 2,
                avg_runnable_pct: 50.0,
                min_runnable_pct: 40.0,
                max_runnable_pct: 60.0,
                spread: 20.0,
                max_gap_ms: 150,
                max_gap_cpu: 3,
                total_migrations: 10,
                ..Default::default()
            }],
            total_workers: 4,
            total_cpus: 2,
            total_migrations: 10,
            worst_spread: 20.0,
            worst_gap_ms: 150,
            worst_gap_cpu: 3,
            ..Default::default()
        };
        let json = serde_json::to_string(&s).unwrap();
        let s2: ScenarioStats = serde_json::from_str(&json).unwrap();
        assert_eq!(s.total_workers, s2.total_workers);
        assert_eq!(s.worst_gap_ms, s2.worst_gap_ms);
        assert_eq!(s.cgroups.len(), s2.cgroups.len());
        assert_eq!(s.cgroups[0].num_workers, s2.cgroups[0].num_workers);
    }

    #[test]
    fn assert_result_serde_roundtrip() {
        let r = AssertResult {
            passed: false,
            details: vec!["test".into()],
            stats: Default::default(),
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: AssertResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r.passed, r2.passed);
        assert_eq!(r.details, r2.details);
    }

    #[test]
    fn multiple_stuck_workers() {
        let threshold = gap_threshold_ms();
        let r = assert_not_starved(&[
            rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], threshold + 500),
            rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[1], threshold + 1500),
        ]);
        assert!(!r.passed);
        let stuck_count = r.details.iter().filter(|d| d.contains("stuck")).count();
        assert_eq!(stuck_count, 2, "both workers should be flagged stuck");
    }

    #[test]
    fn migration_tracking() {
        let mut report = rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1, 2], 50);
        report.migration_count = 5;
        let r = assert_not_starved(&[report]);
        assert_eq!(r.stats.total_migrations, 5);
    }

    // AssertPlan tests

    #[test]
    fn plan_default_empty() {
        let plan = AssertPlan::new();
        assert!(!plan.not_starved);
        assert!(!plan.isolation);
        assert!(plan.max_gap_ms.is_none());
        assert!(plan.max_spread_pct.is_none());
    }

    #[test]
    fn plan_check_not_starved() {
        let plan = AssertPlan::new().check_not_starved();
        let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50)];
        let r = plan.assert_cgroup(&reports, None);
        assert!(r.passed);
        assert_eq!(r.stats.total_workers, 1);
    }

    #[test]
    fn plan_check_isolation_with_cpuset() {
        let plan = AssertPlan::new().check_not_starved().check_isolation();
        let expected: BTreeSet<usize> = [0, 1].into_iter().collect();
        let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1, 4], 50)];
        let r = plan.assert_cgroup(&reports, Some(&expected));
        assert!(!r.passed);
        assert!(r.details.iter().any(|d| d.contains("unexpected")));
    }

    #[test]
    fn plan_isolation_skipped_without_cpuset() {
        let plan = AssertPlan::new().check_isolation();
        let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1, 4], 50)];
        // No cpuset provided -- isolation check is skipped.
        let r = plan.assert_cgroup(&reports, None);
        assert!(r.passed);
    }

    #[test]
    fn plan_custom_gap_threshold_pass() {
        let plan = AssertPlan::new().check_not_starved().max_gap_ms(3000);
        // 2500ms gap: passes with 3000ms threshold.
        let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 2500)];
        let r = plan.assert_cgroup(&reports, None);
        assert!(r.passed, "2500ms < 3000ms threshold: {:?}", r.details);
    }

    #[test]
    fn plan_custom_gap_threshold_fail() {
        let plan = AssertPlan::new().check_not_starved().max_gap_ms(1500);
        // 2000ms gap: fails with 1500ms threshold.
        let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 2000)];
        let r = plan.assert_cgroup(&reports, None);
        assert!(!r.passed);
        assert!(r.details.iter().any(|d| d.contains("stuck")));
        assert!(r.details.iter().any(|d| d.contains("threshold 1500ms")));
    }

    #[test]
    fn plan_no_checks_always_passes() {
        let plan = AssertPlan::new();
        let reports = [rpt(1, 0, 0, 0, &[], 5000)]; // starved + stuck
        let r = plan.assert_cgroup(&reports, None);
        assert!(r.passed, "no checks enabled should pass");
    }

    #[test]
    fn plan_default_all_checks_disabled() {
        // Default::default() must produce the same state as new() —
        // all checks disabled, no gap override.
        let plan = AssertPlan::default();
        assert!(!plan.not_starved, "default must not enable not_starved");
        assert!(!plan.isolation, "default must not enable isolation");
        assert!(
            plan.max_gap_ms.is_none(),
            "default must not set gap override"
        );
        assert!(
            plan.max_spread_pct.is_none(),
            "default must not set spread override"
        );
        // A plan with all checks disabled must pass even pathological input.
        let reports = [rpt(1, 0, 0, 0, &[], 99999)];
        let r = plan.assert_cgroup(&reports, None);
        assert!(r.passed, "all-disabled plan must pass any input");
    }

    #[test]
    fn assert_plan_default_equals_new() {
        // Default impl calls new(). Check field-by-field equivalence
        // and that both produce identical assert_cgroup results.
        let d = AssertPlan::default();
        let n = AssertPlan::new();
        assert_eq!(d.not_starved, n.not_starved);
        assert_eq!(d.isolation, n.isolation);
        assert_eq!(d.max_gap_ms, n.max_gap_ms);
        assert_eq!(d.max_spread_pct, n.max_spread_pct);
        // Both should produce identical pass/fail on the same input.
        let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50)];
        let rd = d.assert_cgroup(&reports, None);
        let rn = n.assert_cgroup(&reports, None);
        assert_eq!(rd.passed, rn.passed);
    }

    #[test]
    fn single_worker_spread_zero() {
        let r = assert_not_starved(&[rpt(1, 500, 5e9 as u64, 25e8 as u64, &[0, 1], 50)]);
        assert!(r.passed);
        let c = &r.stats.cgroups[0];
        assert!((c.spread - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn zero_wall_time_nonzero_work() {
        // wall_time=0 but work_units>0: the worker did work but the timer
        // didn't advance. Should not produce a starved failure since work was done.
        // The runnable_pct computation skips this worker (no pcts entry).
        let r = assert_not_starved(&[rpt(1, 100, 0, 0, &[0], 0)]);
        assert!(
            r.passed,
            "nonzero work with zero wall_time: {:?}",
            r.details
        );
    }

    #[test]
    fn isolation_empty_expected_set() {
        // Empty expected set means no CPUs are "expected", so any CPU
        // used by the worker is unexpected. difference(empty) == worker's set.
        let expected: BTreeSet<usize> = BTreeSet::new();
        let r = assert_isolation(
            &[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50)],
            &expected,
        );
        // Worker used CPUs {0,1}, expected is empty, so all are unexpected.
        assert!(!r.passed);
        assert!(r.details.iter().any(|d| d.contains("unexpected")));
    }

    #[test]
    fn isolation_worker_used_no_cpus() {
        // Worker used no CPUs -- difference with expected is empty, so passes.
        let expected: BTreeSet<usize> = [0, 1].into_iter().collect();
        let r = assert_isolation(&[rpt(1, 0, 0, 0, &[], 0)], &expected);
        assert!(r.passed);
    }

    #[test]
    fn isolation_all_unexpected_cpus() {
        let expected: BTreeSet<usize> = [0, 1].into_iter().collect();
        let r = assert_isolation(
            &[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[4, 5, 6], 50)],
            &expected,
        );
        assert!(!r.passed);
        assert!(r.details.iter().any(|d| d.contains("unexpected")));
    }

    #[test]
    fn merge_pass_and_fail() {
        let pass = AssertResult::pass();
        let mut fail = AssertResult::pass();
        fail.passed = false;
        fail.details.push("something failed".into());

        let mut merged = pass;
        merged.merge(fail);
        assert!(!merged.passed, "merging pass+fail must produce fail");
        assert!(
            merged
                .details
                .iter()
                .any(|d| d.contains("something failed"))
        );
    }

    #[test]
    fn merge_fail_and_pass() {
        let mut fail = AssertResult::pass();
        fail.passed = false;
        fail.details.push("first failed".into());
        let pass = AssertResult::pass();

        let mut merged = fail;
        merged.merge(pass);
        assert!(!merged.passed, "merging fail+pass must produce fail");
    }

    #[test]
    fn plan_starved_still_fails_with_custom_gap() {
        // A starved worker (work_units=0) must still cause failure even
        // when the custom max_gap_ms threshold is high enough that the
        // gap check passes.
        let plan = AssertPlan::new().check_not_starved().max_gap_ms(5000);
        let reports = [
            rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 100), // healthy
            rpt(2, 0, 5e9 as u64, 0, &[1], 1500),            // starved, gap < threshold
        ];
        let r = plan.assert_cgroup(&reports, None);
        assert!(
            !r.passed,
            "starved worker must fail even with relaxed gap threshold"
        );
        assert!(r.details.iter().any(|d| d.contains("starved")));
        // The gap (1500ms) is below the 5000ms threshold, so no "stuck" detail.
        assert!(!r.details.iter().any(|d| d.contains("stuck")));
    }

    // -- Assert merge tests --

    #[test]
    fn assert_none_has_no_checks() {
        let v = Assert::NONE;
        assert!(v.not_starved.is_none());
        assert!(v.isolation.is_none());
        assert!(v.max_gap_ms.is_none());
        assert!(v.max_spread_pct.is_none());
        assert!(v.max_imbalance_ratio.is_none());
    }

    #[test]
    fn assert_default_checks_enables_not_starved() {
        let v = Assert::default_checks();
        assert_eq!(v.not_starved, Some(true));
        assert!(v.isolation.is_none());
        assert!(v.max_imbalance_ratio.is_some());
        assert!(v.max_local_dsq_depth.is_some());
        assert!(v.fail_on_stall.is_some());
        assert!(v.sustained_samples.is_some());
        assert!(v.max_fallback_rate.is_some());
        assert!(v.max_keep_last_rate.is_some());
    }

    #[test]
    fn assert_merge_other_overrides_self() {
        let base = Assert::NONE;
        let other = Assert::NONE
            .check_not_starved()
            .max_gap_ms(5000)
            .max_imbalance_ratio(2.0);
        let merged = base.merge(&other);
        assert_eq!(merged.not_starved, Some(true));
        assert_eq!(merged.max_gap_ms, Some(5000));
        assert_eq!(merged.max_imbalance_ratio, Some(2.0));
    }

    #[test]
    fn assert_merge_preserves_self_when_other_is_none() {
        let base = Assert::default_checks();
        let merged = base.merge(&Assert::NONE);
        assert_eq!(merged.not_starved, Some(true));
        assert!(merged.max_imbalance_ratio.is_some());
        assert!(merged.max_local_dsq_depth.is_some());
    }

    #[test]
    fn assert_merge_other_takes_precedence() {
        let base = Assert::NONE.max_imbalance_ratio(4.0);
        let other = Assert::NONE.max_imbalance_ratio(2.0);
        let merged = base.merge(&other);
        assert_eq!(merged.max_imbalance_ratio, Some(2.0));
    }

    #[test]
    fn assert_merge_last_some_wins() {
        let base = Assert::NONE.check_not_starved();
        let other = Assert::NONE.check_isolation();
        let merged = base.merge(&other);
        assert_eq!(merged.not_starved, Some(true));
        assert_eq!(merged.isolation, Some(true));
    }

    #[test]
    fn assert_merge_child_disables_not_starved() {
        let base = Assert::default_checks(); // not_starved = Some(true)
        let other = Assert {
            not_starved: Some(false),
            ..Assert::NONE
        };
        let merged = base.merge(&other);
        assert_eq!(merged.not_starved, Some(false));
        assert!(!merged.worker_plan().not_starved);
    }

    #[test]
    fn assert_merge_child_disables_isolation() {
        let base = Assert::NONE.check_isolation(); // isolation = Some(true)
        let other = Assert {
            isolation: Some(false),
            ..Assert::NONE
        };
        let merged = base.merge(&other);
        assert_eq!(merged.isolation, Some(false));
        assert!(!merged.worker_plan().isolation);
    }

    #[test]
    fn assert_worker_plan_extraction() {
        let v = Assert::NONE
            .check_not_starved()
            .check_isolation()
            .max_gap_ms(3000)
            .max_spread_pct(25.0);
        assert_eq!(v.not_starved, Some(true));
        assert_eq!(v.isolation, Some(true));
        let plan = v.worker_plan();
        assert!(plan.not_starved);
        assert!(plan.isolation);
        assert_eq!(plan.max_gap_ms, Some(3000));
        assert_eq!(plan.max_spread_pct, Some(25.0));
    }

    #[test]
    fn assert_cgroup_delegates_to_plan() {
        let v = Assert::NONE.check_not_starved();
        let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50)];
        let r = v.assert_cgroup(&reports, None);
        assert!(r.passed);
        assert_eq!(r.stats.total_workers, 1);
    }

    #[test]
    fn assert_monitor_thresholds_extraction() {
        let v = Assert::NONE
            .max_imbalance_ratio(2.5)
            .max_local_dsq_depth(100)
            .fail_on_stall(false)
            .sustained_samples(10)
            .max_fallback_rate(50.0)
            .max_keep_last_rate(25.0);
        let t = v.monitor_thresholds();
        assert!((t.max_imbalance_ratio - 2.5).abs() < f64::EPSILON);
        assert_eq!(t.max_local_dsq_depth, 100);
        assert!(!t.fail_on_stall);
        assert_eq!(t.sustained_samples, 10);
        assert!((t.max_fallback_rate - 50.0).abs() < f64::EPSILON);
        assert!((t.max_keep_last_rate - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn assert_monitor_thresholds_defaults_when_none() {
        let v = Assert::NONE;
        let t = v.monitor_thresholds();
        let d = crate::monitor::MonitorThresholds::DEFAULT;
        assert!((t.max_imbalance_ratio - d.max_imbalance_ratio).abs() < f64::EPSILON);
        assert_eq!(t.max_local_dsq_depth, d.max_local_dsq_depth);
    }

    #[test]
    fn assert_chain_all_setters() {
        let v = Assert::NONE
            .check_not_starved()
            .check_isolation()
            .max_gap_ms(1000)
            .max_spread_pct(5.0)
            .max_imbalance_ratio(3.0)
            .max_local_dsq_depth(20)
            .fail_on_stall(true)
            .sustained_samples(3)
            .max_fallback_rate(100.0)
            .max_keep_last_rate(50.0);
        assert_eq!(v.not_starved, Some(true));
        assert_eq!(v.isolation, Some(true));
        assert_eq!(v.max_gap_ms, Some(1000));
        assert_eq!(v.max_spread_pct, Some(5.0));
        assert_eq!(v.max_imbalance_ratio, Some(3.0));
        assert_eq!(v.max_local_dsq_depth, Some(20));
        assert_eq!(v.fail_on_stall, Some(true));
        assert_eq!(v.sustained_samples, Some(3));
        assert_eq!(v.max_fallback_rate, Some(100.0));
        assert_eq!(v.max_keep_last_rate, Some(50.0));
    }

    // -- gap_threshold_ms tests --

    #[test]
    fn gap_threshold_default() {
        let t = gap_threshold_ms();
        if cfg!(debug_assertions) {
            assert_eq!(t, 3000);
        } else {
            assert_eq!(t, 2000);
        }
    }

    #[test]
    fn assert_result_pass_defaults() {
        let r = AssertResult::pass();
        assert!(r.passed);
        assert!(r.details.is_empty());
        assert_eq!(r.stats.total_workers, 0);
    }

    // -- Assert::merge per-field tests --

    #[test]
    fn assert_merge_max_spread_pct() {
        let base = Assert::NONE.max_spread_pct(10.0);
        let other = Assert::NONE.max_spread_pct(5.0);
        assert_eq!(base.merge(&other).max_spread_pct, Some(5.0));
        assert_eq!(base.merge(&Assert::NONE).max_spread_pct, Some(10.0));
    }

    #[test]
    fn assert_merge_fail_on_stall() {
        let base = Assert::NONE.fail_on_stall(true);
        let other = Assert::NONE.fail_on_stall(false);
        assert_eq!(base.merge(&other).fail_on_stall, Some(false));
        assert_eq!(base.merge(&Assert::NONE).fail_on_stall, Some(true));
    }

    #[test]
    fn assert_merge_sustained_samples() {
        let base = Assert::NONE.sustained_samples(5);
        let other = Assert::NONE.sustained_samples(10);
        assert_eq!(base.merge(&other).sustained_samples, Some(10));
        assert_eq!(base.merge(&Assert::NONE).sustained_samples, Some(5));
    }

    #[test]
    fn assert_merge_max_fallback_rate() {
        let base = Assert::NONE.max_fallback_rate(200.0);
        let other = Assert::NONE.max_fallback_rate(50.0);
        assert_eq!(base.merge(&other).max_fallback_rate, Some(50.0));
        assert_eq!(base.merge(&Assert::NONE).max_fallback_rate, Some(200.0));
    }

    #[test]
    fn assert_merge_max_keep_last_rate() {
        let base = Assert::NONE.max_keep_last_rate(100.0);
        let other = Assert::NONE.max_keep_last_rate(25.0);
        assert_eq!(base.merge(&other).max_keep_last_rate, Some(25.0));
        assert_eq!(base.merge(&Assert::NONE).max_keep_last_rate, Some(100.0));
    }

    #[test]
    fn assert_merge_max_local_dsq_depth() {
        let base = Assert::NONE.max_local_dsq_depth(50);
        let other = Assert::NONE.max_local_dsq_depth(100);
        assert_eq!(base.merge(&other).max_local_dsq_depth, Some(100));
        assert_eq!(base.merge(&Assert::NONE).max_local_dsq_depth, Some(50));
    }

    #[test]
    fn assert_merge_max_gap_ms() {
        let base = Assert::NONE.max_gap_ms(2000);
        let other = Assert::NONE.max_gap_ms(5000);
        assert_eq!(base.merge(&other).max_gap_ms, Some(5000));
        assert_eq!(base.merge(&Assert::NONE).max_gap_ms, Some(2000));
    }

    #[test]
    fn assert_merge_three_layers() {
        let defaults = Assert::default_checks();
        let sched = Assert::NONE
            .max_imbalance_ratio(2.0)
            .max_fallback_rate(50.0);
        let test = Assert::NONE.max_gap_ms(5000);
        let merged = defaults.merge(&sched).merge(&test);
        assert_eq!(merged.not_starved, Some(true));
        assert_eq!(merged.max_imbalance_ratio, Some(2.0));
        assert_eq!(merged.max_fallback_rate, Some(50.0));
        assert_eq!(merged.max_gap_ms, Some(5000));
        assert_eq!(merged.sustained_samples, Some(5));
    }

    // ---------------------------------------------------------------
    // Negative tests: check that diagnostics catch controlled failures
    // ---------------------------------------------------------------

    #[test]
    fn neg_starvation_zero_work_detected() {
        let r = assert_not_starved(&[
            rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50),
            rpt(2, 0, 5e9 as u64, 0, &[0], 0), // starved
            rpt(3, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50),
        ]);
        assert!(!r.passed, "starvation must be caught");
        let starved = r.details.iter().filter(|d| d.contains("starved")).count();
        assert_eq!(starved, 1, "exactly one starved worker expected");
        // Format: "tid 2 starved (0 work units)"
        let detail = r.details.iter().find(|d| d.contains("starved")).unwrap();
        assert!(
            detail.contains("tid 2"),
            "must name the starved tid: {detail}"
        );
        assert!(
            detail.contains("0 work units"),
            "must state zero work: {detail}"
        );
    }

    #[test]
    fn neg_isolation_violation_outside_cpuset() {
        let expected: BTreeSet<usize> = [0, 1].into_iter().collect();
        let reports = [
            rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50),
            rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[0, 1, 2, 3], 50),
        ];
        let r = assert_isolation(&reports, &expected);
        assert!(!r.passed, "isolation violation must be caught");
        // Format: "tid 2 ran on unexpected CPUs {2, 3}"
        let detail = r
            .details
            .iter()
            .find(|d| d.contains("unexpected CPUs"))
            .unwrap();
        assert!(
            detail.contains("tid 2"),
            "must name violating tid: {detail}"
        );
        assert!(detail.contains("2"), "must list out-of-set CPU 2: {detail}");
        assert!(detail.contains("3"), "must list out-of-set CPU 3: {detail}");
        // Worker 1 ran only on {0,1} which is within expected — no violation.
        assert_eq!(r.details.len(), 1, "only tid 2 should violate");
    }

    #[test]
    fn neg_unfairness_extreme_spread_detected() {
        let r = assert_not_starved(&[
            rpt(1, 100, 5e9 as u64, 25e7 as u64, &[0, 1], 50), // 5%
            rpt(2, 5000, 5e9 as u64, 475e7 as u64, &[0, 1], 50), // 95%
        ]);
        assert!(!r.passed, "extreme unfairness must be caught");
        // Format: "unfair cgroup: spread=90% (5-95%) 2 workers on 2 cpus"
        let detail = r.details.iter().find(|d| d.contains("unfair")).unwrap();
        assert!(
            detail.contains("spread="),
            "must include spread value: {detail}"
        );
        assert!(
            detail.contains("workers"),
            "must include worker count: {detail}"
        );
        assert!(detail.contains("cpus"), "must include cpu count: {detail}");
        let c = &r.stats.cgroups[0];
        assert!(
            c.spread > 80.0,
            "spread should be >80%, got {:.1}",
            c.spread
        );
        assert_eq!(c.num_workers, 2);
        assert_eq!(c.num_cpus, 2);
        assert!(
            c.min_runnable_pct < 10.0,
            "min pct should be ~5%: {:.1}",
            c.min_runnable_pct
        );
        assert!(
            c.max_runnable_pct > 90.0,
            "max pct should be ~95%: {:.1}",
            c.max_runnable_pct
        );
    }

    #[test]
    fn neg_scheduling_gap_exceeds_threshold() {
        let threshold = gap_threshold_ms();
        let gap = threshold + 2000;
        let r = assert_not_starved(&[
            rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50),
            rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[1], gap),
        ]);
        assert!(!r.passed, "scheduling gap must be caught");
        // Format: "stuck {gap}ms on cpu1 at +1000ms"
        let detail = r.details.iter().find(|d| d.contains("stuck")).unwrap();
        assert!(
            detail.contains(&format!("{}ms", gap)),
            "must include gap duration: {detail}"
        );
        assert!(
            detail.contains("on cpu"),
            "must include CPU number: {detail}"
        );
        assert!(
            detail.contains("at +"),
            "must include timing offset: {detail}"
        );
        assert!(detail.contains("cpu1"), "gap is on cpu1: {detail}");
        // Stats must reflect the gap.
        assert_eq!(r.stats.worst_gap_ms, gap);
        assert_eq!(r.stats.worst_gap_cpu, 1);
    }

    #[test]
    fn neg_plan_custom_gap_catches_lower_threshold() {
        let plan = AssertPlan::new().check_not_starved().max_gap_ms(500);
        let reports = [
            rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50),
            rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[1], 1000),
        ];
        let r = plan.assert_cgroup(&reports, None);
        assert!(!r.passed, "custom 500ms threshold must catch 1000ms gap");
        // Format: "stuck 1000ms on cpu1 at +1000ms (threshold 500ms)"
        let detail = r.details.iter().find(|d| d.contains("stuck")).unwrap();
        assert!(
            detail.contains("1000ms"),
            "must include gap duration: {detail}"
        );
        assert!(detail.contains("cpu1"), "must include CPU: {detail}");
        assert!(
            detail.contains("threshold 500ms"),
            "must include custom threshold: {detail}"
        );
    }

    #[test]
    fn neg_isolation_plus_starvation_both_reported() {
        let plan = AssertPlan::new().check_not_starved().check_isolation();
        let expected: BTreeSet<usize> = [0, 1].into_iter().collect();
        let reports = [
            rpt(1, 0, 5e9 as u64, 0, &[0], 0),
            rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[4, 5], 50),
        ];
        let r = plan.assert_cgroup(&reports, Some(&expected));
        assert!(!r.passed);
        // Starvation detail must name tid 1 with "0 work units".
        let starved_detail = r.details.iter().find(|d| d.contains("starved")).unwrap();
        assert!(
            starved_detail.contains("tid 1"),
            "starved tid: {starved_detail}"
        );
        assert!(
            starved_detail.contains("0 work units"),
            "format: {starved_detail}"
        );
        // Isolation detail must name tid 2 with CPUs {4, 5}.
        let iso_detail = r.details.iter().find(|d| d.contains("unexpected")).unwrap();
        assert!(iso_detail.contains("tid 2"), "isolation tid: {iso_detail}");
        assert!(iso_detail.contains("4"), "must list CPU 4: {iso_detail}");
        assert!(iso_detail.contains("5"), "must list CPU 5: {iso_detail}");
    }

    #[test]
    fn neg_assert_cgroup_via_assert_struct() {
        let v = Assert::NONE.check_not_starved().check_isolation();
        let expected: BTreeSet<usize> = [0].into_iter().collect();
        let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1, 2], 50)];
        let r = v.assert_cgroup(&reports, Some(&expected));
        assert!(
            !r.passed,
            "Assert.assert_cgroup must catch isolation failure"
        );
        let detail = r.details.iter().find(|d| d.contains("unexpected")).unwrap();
        assert!(detail.contains("tid 1"), "must name tid: {detail}");
        assert!(detail.contains("1"), "must list CPU 1: {detail}");
        assert!(detail.contains("2"), "must list CPU 2: {detail}");
    }

    #[test]
    fn assert_merge_none_preserves_base() {
        let base = Assert::default_checks();
        let merged = base.merge(&Assert::NONE);
        assert_eq!(merged.not_starved, Some(true));
        assert!(merged.max_imbalance_ratio.is_some());
        assert!(merged.fail_on_stall.is_some());
    }

    #[test]
    fn assert_merge_overrides_fields() {
        let base = Assert::NONE;
        let overrides = Assert::NONE
            .max_imbalance_ratio(5.0)
            .max_gap_ms(1000)
            .check_not_starved();
        let merged = base.merge(&overrides);
        assert_eq!(merged.not_starved, Some(true));
        assert_eq!(merged.max_imbalance_ratio, Some(5.0));
        assert_eq!(merged.max_gap_ms, Some(1000));
    }

    #[test]
    fn assert_merge_later_overrides_earlier() {
        let a = Assert::NONE.max_imbalance_ratio(2.0);
        let b = Assert::NONE.max_imbalance_ratio(10.0);
        let merged = a.merge(&b);
        assert_eq!(merged.max_imbalance_ratio, Some(10.0));
    }

    #[test]
    fn assert_worker_plan_extracts_fields() {
        let v = Assert::NONE
            .check_not_starved()
            .check_isolation()
            .max_gap_ms(500)
            .max_spread_pct(10.0);
        assert_eq!(v.not_starved, Some(true));
        assert_eq!(v.isolation, Some(true));
        let plan = v.worker_plan();
        assert!(plan.not_starved);
        assert!(plan.isolation);
        assert_eq!(plan.max_gap_ms, Some(500));
        assert_eq!(plan.max_spread_pct, Some(10.0));
    }

    #[test]
    fn assert_monitor_thresholds_defaults() {
        let v = Assert::NONE;
        let t = v.monitor_thresholds();
        // Should use MonitorThresholds::DEFAULT values.
        let d = crate::monitor::MonitorThresholds::DEFAULT;
        assert_eq!(t.max_imbalance_ratio, d.max_imbalance_ratio);
        assert_eq!(t.max_local_dsq_depth, d.max_local_dsq_depth);
    }

    #[test]
    fn assert_monitor_thresholds_overridden() {
        let v = Assert::NONE
            .max_imbalance_ratio(99.0)
            .max_local_dsq_depth(42)
            .fail_on_stall(false)
            .sustained_samples(10)
            .max_fallback_rate(0.5)
            .max_keep_last_rate(0.3);
        let t = v.monitor_thresholds();
        assert_eq!(t.max_imbalance_ratio, 99.0);
        assert_eq!(t.max_local_dsq_depth, 42);
        assert!(!t.fail_on_stall);
        assert_eq!(t.sustained_samples, 10);
        assert_eq!(t.max_fallback_rate, 0.5);
        assert_eq!(t.max_keep_last_rate, 0.3);
    }

    #[test]
    fn assert_max_spread_pct() {
        let v = Assert::NONE.max_spread_pct(25.0);
        assert_eq!(v.max_spread_pct, Some(25.0));
    }

    #[test]
    fn gap_threshold_debug_vs_release() {
        let t = gap_threshold_ms();
        // In test builds (debug_assertions=true), threshold is 3000.
        assert!(t >= 2000, "threshold should be at least 2000ms: {t}");
    }

    #[test]
    fn assert_result_merge_combines_stats() {
        let mut a = AssertResult {
            passed: true,
            details: vec!["a".into()],
            stats: ScenarioStats {
                cgroups: vec![],
                total_workers: 2,
                total_cpus: 4,
                total_migrations: 10,
                worst_spread: 5.0,
                worst_gap_ms: 100,
                worst_gap_cpu: 0,
                ..Default::default()
            },
        };
        let b = AssertResult {
            passed: false,
            details: vec!["b".into()],
            stats: ScenarioStats {
                cgroups: vec![],
                total_workers: 3,
                total_cpus: 6,
                total_migrations: 20,
                worst_spread: 15.0,
                worst_gap_ms: 500,
                worst_gap_cpu: 2,
                ..Default::default()
            },
        };
        a.merge(b);
        assert!(!a.passed);
        assert_eq!(a.details, vec!["a", "b"]);
        assert_eq!(a.stats.total_workers, 5);
        assert_eq!(a.stats.total_cpus, 10);
        assert_eq!(a.stats.total_migrations, 30);
        assert_eq!(a.stats.worst_spread, 15.0);
        assert_eq!(a.stats.worst_gap_ms, 500);
        assert_eq!(a.stats.worst_gap_cpu, 2);
    }

    #[test]
    fn neg_plan_custom_gap_passes_below_threshold() {
        let plan = AssertPlan::new().check_not_starved().max_gap_ms(5000);
        let reports = [
            rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50),
            rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[1], 1000),
        ];
        let r = plan.assert_cgroup(&reports, None);
        // 1000ms gap < 5000ms threshold, so it passes.
        let has_stuck = r.details.iter().any(|d| d.contains("stuck"));
        assert!(!has_stuck, "1000ms gap should pass 5000ms threshold");
    }

    // -- assert_benchmarks tests --

    fn rpt_with_latencies(
        tid: u32,
        latencies: Vec<u64>,
        iterations: u64,
        wall_ns: u64,
    ) -> WorkerReport {
        WorkerReport {
            tid,
            work_units: 1000,
            cpu_time_ns: wall_ns / 2,
            wall_time_ns: wall_ns,
            runnable_ns: wall_ns / 2,
            migration_count: 0,
            cpus_used: [0].into_iter().collect(),
            migrations: vec![],
            max_gap_ms: 50,
            max_gap_cpu: 0,
            max_gap_at_ms: 1000,
            wake_latencies_ns: latencies,
            iterations,
            schedstat_run_delay_ns: 0,
            schedstat_ctx_switches: 0,
            schedstat_cpu_time_ns: 0,
        }
    }

    #[test]
    fn assert_benchmarks_empty_reports() {
        let r = assert_benchmarks(&[], Some(1000), Some(0.5), Some(100.0));
        assert!(r.passed);
    }

    #[test]
    fn assert_benchmarks_no_thresholds() {
        let reports = [rpt_with_latencies(
            1,
            vec![1000, 2000, 3000],
            10,
            5_000_000_000,
        )];
        let r = assert_benchmarks(&reports, None, None, None);
        assert!(r.passed);
    }

    #[test]
    fn assert_benchmarks_p99_pass() {
        let reports = [rpt_with_latencies(
            1,
            vec![100, 200, 300, 400, 500],
            10,
            5_000_000_000,
        )];
        let r = assert_benchmarks(&reports, Some(1000), None, None);
        assert!(r.passed, "p99 500ns < 1000ns limit: {:?}", r.details);
    }

    #[test]
    fn assert_benchmarks_p99_fail() {
        let reports = [rpt_with_latencies(
            1,
            vec![100, 200, 300, 400, 2000],
            10,
            5_000_000_000,
        )];
        let r = assert_benchmarks(&reports, Some(1000), None, None);
        assert!(!r.passed);
        assert!(r.details.iter().any(|d| d.contains("p99 wake latency")));
    }

    #[test]
    fn assert_benchmarks_cv_pass() {
        // All same latency -> CV = 0.
        let reports = [rpt_with_latencies(
            1,
            vec![1000, 1000, 1000, 1000],
            10,
            5_000_000_000,
        )];
        let r = assert_benchmarks(&reports, None, Some(0.5), None);
        assert!(r.passed, "uniform latencies CV=0: {:?}", r.details);
    }

    #[test]
    fn assert_benchmarks_cv_fail() {
        // High variance latencies.
        let reports = [rpt_with_latencies(
            1,
            vec![100, 100, 100, 100000],
            10,
            5_000_000_000,
        )];
        let r = assert_benchmarks(&reports, None, Some(0.5), None);
        assert!(!r.passed);
        assert!(r.details.iter().any(|d| d.contains("wake latency CV")));
    }

    #[test]
    fn assert_benchmarks_iteration_rate_pass() {
        // 1000 iterations in 5 seconds = 200/s, above 100/s floor.
        let reports = [rpt_with_latencies(1, vec![], 1000, 5_000_000_000)];
        let r = assert_benchmarks(&reports, None, None, Some(100.0));
        assert!(r.passed, "200/s > 100/s floor: {:?}", r.details);
    }

    #[test]
    fn assert_benchmarks_iteration_rate_fail() {
        // 10 iterations in 5 seconds = 2/s, below 100/s floor.
        let reports = [rpt_with_latencies(1, vec![], 10, 5_000_000_000)];
        let r = assert_benchmarks(&reports, None, None, Some(100.0));
        assert!(!r.passed);
        assert!(r.details.iter().any(|d| d.contains("iteration rate")));
    }

    #[test]
    fn assert_benchmarks_zero_wall_time_skips_rate() {
        let reports = [rpt_with_latencies(1, vec![], 10, 0)];
        let r = assert_benchmarks(&reports, None, None, Some(100.0));
        assert!(r.passed, "zero wall_time should skip rate check");
    }

    #[test]
    fn assert_benchmarks_no_latencies_skips_p99() {
        let reports = [rpt_with_latencies(1, vec![], 10, 5_000_000_000)];
        let r = assert_benchmarks(&reports, Some(1000), None, None);
        assert!(r.passed, "empty latencies should skip p99 check");
    }

    #[test]
    fn assert_benchmarks_single_latency_cv_skipped() {
        // Single sample -> len < 2, CV check skipped.
        let reports = [rpt_with_latencies(1, vec![1000], 10, 5_000_000_000)];
        let r = assert_benchmarks(&reports, None, Some(0.1), None);
        assert!(r.passed, "single sample should skip CV check");
    }

    // -- wake latency stats in assert_not_starved --

    #[test]
    fn not_starved_wake_latency_stats() {
        let reports = [
            rpt_with_latencies(1, vec![1000, 2000, 3000, 4000, 5000], 100, 5_000_000_000),
            rpt_with_latencies(2, vec![6000, 7000, 8000, 9000, 10000], 200, 5_000_000_000),
        ];
        let r = assert_not_starved(&reports);
        assert!(r.passed, "{:?}", r.details);
        let s = &r.stats;
        // p99 of [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000] in us:
        // sorted, p99_idx = ceil(10*0.99) = 10, clamped to 9 -> 10000ns = 10.0us
        assert!(
            s.p99_wake_latency_us > 9.0,
            "p99: {}",
            s.p99_wake_latency_us
        );
        // median of 10 samples: index 5 -> 6000ns = 6.0us
        assert!(
            (s.median_wake_latency_us - 6.0).abs() < 0.1,
            "median: {}",
            s.median_wake_latency_us
        );
        assert!(s.wake_latency_cv > 0.0, "cv: {}", s.wake_latency_cv);
        assert_eq!(s.total_iterations, 300);
    }

    #[test]
    fn not_starved_empty_latencies_zero_stats() {
        let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50)];
        let r = assert_not_starved(&reports);
        assert!(r.passed);
        assert_eq!(r.stats.p99_wake_latency_us, 0.0);
        assert_eq!(r.stats.median_wake_latency_us, 0.0);
        assert_eq!(r.stats.wake_latency_cv, 0.0);
    }

    #[test]
    fn not_starved_run_delay_stats() {
        let mut w1 = rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50);
        w1.schedstat_run_delay_ns = 100_000; // 100us
        let mut w2 = rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[1], 50);
        w2.schedstat_run_delay_ns = 300_000; // 300us
        let r = assert_not_starved(&[w1, w2]);
        assert!(r.passed, "{:?}", r.details);
        // mean_run_delay = (100 + 300) / 2 = 200us
        assert!(
            (r.stats.mean_run_delay_us - 200.0).abs() < 0.1,
            "mean: {}",
            r.stats.mean_run_delay_us
        );
        // worst_run_delay = 300us
        assert!(
            (r.stats.worst_run_delay_us - 300.0).abs() < 0.1,
            "worst: {}",
            r.stats.worst_run_delay_us
        );
    }

    // -- AssertPlan benchmarking integration --

    #[test]
    fn plan_benchmarks_p99_via_assert_cgroup() {
        let plan = AssertPlan {
            not_starved: false,
            isolation: false,
            max_gap_ms: None,
            max_spread_pct: None,
            max_throughput_cv: None,
            min_work_rate: None,
            max_p99_wake_latency_ns: Some(500),
            max_wake_latency_cv: None,
            min_iteration_rate: None,
            max_migration_ratio: None,
        };
        let reports = [rpt_with_latencies(
            1,
            vec![100, 200, 300, 400, 1000],
            10,
            5_000_000_000,
        )];
        let r = plan.assert_cgroup(&reports, None);
        assert!(!r.passed, "p99 1000ns > 500ns limit");
        assert!(r.details.iter().any(|d| d.contains("p99 wake latency")));
    }

    #[test]
    fn plan_migration_ratio_gate() {
        let mut w = rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50);
        w.migration_count = 10;
        w.iterations = 100;
        // ratio = 10/100 = 0.10, threshold 0.05 → fail
        let plan = AssertPlan {
            not_starved: false,
            isolation: false,
            max_gap_ms: None,
            max_spread_pct: None,
            max_throughput_cv: None,
            min_work_rate: None,
            max_p99_wake_latency_ns: None,
            max_wake_latency_cv: None,
            min_iteration_rate: None,
            max_migration_ratio: Some(0.05),
        };
        let r = plan.assert_cgroup(&[w], None);
        assert!(!r.passed);
        assert!(r.details.iter().any(|d| d.contains("migration ratio")));
    }

    #[test]
    fn plan_migration_ratio_gate_pass() {
        let mut w = rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50);
        w.migration_count = 2;
        w.iterations = 100;
        // ratio = 2/100 = 0.02, threshold 0.05 → pass
        let plan = AssertPlan {
            not_starved: false,
            isolation: false,
            max_gap_ms: None,
            max_spread_pct: None,
            max_throughput_cv: None,
            min_work_rate: None,
            max_p99_wake_latency_ns: None,
            max_wake_latency_cv: None,
            min_iteration_rate: None,
            max_migration_ratio: Some(0.05),
        };
        let r = plan.assert_cgroup(&[w], None);
        assert!(r.passed, "{:?}", r.details);
    }

    #[test]
    fn plan_benchmarks_iteration_rate_via_assert_cgroup() {
        let plan = AssertPlan {
            not_starved: false,
            isolation: false,
            max_gap_ms: None,
            max_spread_pct: None,
            max_throughput_cv: None,
            min_work_rate: None,
            max_p99_wake_latency_ns: None,
            max_wake_latency_cv: None,
            min_iteration_rate: Some(1000.0),
            max_migration_ratio: None,
        };
        let reports = [rpt_with_latencies(1, vec![], 10, 5_000_000_000)];
        let r = plan.assert_cgroup(&reports, None);
        assert!(!r.passed, "2/s < 1000/s floor");
        assert!(r.details.iter().any(|d| d.contains("iteration rate")));
    }
}
