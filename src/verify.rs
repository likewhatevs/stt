//! Pass/fail evaluation of scenario results.
//!
//! Key types:
//! - [`VerifyResult`] -- pass/fail status with diagnostics and statistics
//! - [`Verify`] -- composable verification config (worker + monitor checks)
//! - [`VerificationPlan`] -- worker-side check configuration
//! - [`ScenarioStats`] / [`CgroupStats`] -- aggregated telemetry
//!
//! Verification uses a three-layer merge: [`Verify::default_checks()`] ->
//! `Scheduler.verify` -> per-test `verify`.
//!
//! See the [Verification](https://sched-ext.github.io/scx/stt/concepts/verification.html)
//! chapter of the guide.

use crate::workload::WorkerReport;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static WARN_UNFAIR: AtomicBool = AtomicBool::new(false);

/// Override for the default 2000ms gap threshold. 0 means use default.
static COVERAGE_GAP_MS: AtomicU64 = AtomicU64::new(0);

/// When true, unfair spread produces a warning detail but does not fail the result.
///
/// `pub` because main.rs is a separate binary crate.
#[doc(hidden)]
pub fn set_warn_unfair(v: bool) {
    WARN_UNFAIR.store(v, Ordering::Relaxed);
}

/// Override the default scheduling gap threshold (ms).
///
/// Set to a higher value for coverage-instrumented runs where
/// instrumentation overhead increases scheduling latency.
/// 0 means use the default (2000ms release, 3000ms debug).
///
/// `pub` because main.rs is a separate binary crate.
#[doc(hidden)]
pub fn set_coverage_gap_ms(ms: u64) {
    COVERAGE_GAP_MS.store(ms, Ordering::Relaxed);
}

fn gap_threshold_ms() -> u64 {
    let v = COVERAGE_GAP_MS.load(Ordering::Relaxed);
    if v > 0 {
        return v;
    }
    // Unoptimized debug builds have higher scheduling overhead.
    if cfg!(debug_assertions) { 3000 } else { 2000 }
}

fn spread_threshold_pct() -> f64 {
    // Debug builds in small VMs (especially under EEVDF) show higher
    // spread than optimized builds under sched_ext schedulers.
    if cfg!(debug_assertions) { 35.0 } else { 15.0 }
}

/// Result of verifying a scenario run.
///
/// Contains pass/fail status, human-readable detail messages, and
/// aggregated statistics. Multiple results can be combined with
/// [`merge()`](VerifyResult::merge).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VerifyResult {
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
}

impl VerifyResult {
    pub fn pass() -> Self {
        Self {
            passed: true,
            details: vec![],
            stats: Default::default(),
        }
    }
    pub fn merge(&mut self, other: VerifyResult) {
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
    }
}

/// Composable verification plan. Specifies which checks to run on worker
/// reports after collection.
#[derive(Clone, Debug)]
pub struct VerificationPlan {
    pub not_starved: bool,
    pub isolation: bool,
    pub max_gap_ms: Option<u64>,
    pub max_spread_pct: Option<f64>,
}

impl VerificationPlan {
    pub fn new() -> Self {
        Self {
            not_starved: false,
            isolation: false,
            max_gap_ms: None,
            max_spread_pct: None,
        }
    }

    /// Enable the not-starved check (zero work units, spread, scheduling gaps).
    pub fn check_not_starved(mut self) -> Self {
        self.not_starved = true;
        self
    }

    /// Enable cpuset isolation check. Only applied when a cpuset is provided
    /// to `verify_cell`.
    pub fn check_isolation(mut self) -> Self {
        self.isolation = true;
        self
    }

    /// Override the default max scheduling gap threshold.
    pub fn max_gap_ms(mut self, ms: u64) -> Self {
        self.max_gap_ms = Some(ms);
        self
    }

    /// Override the default max spread threshold (%).
    pub fn max_spread_pct(mut self, pct: f64) -> Self {
        self.max_spread_pct = Some(pct);
        self
    }

    /// Run all configured checks against one cgroup's reports.
    ///
    /// `cpuset` is the expected CPU set for isolation checks. Pass `None`
    /// when there is no cpuset constraint (isolation check is skipped).
    pub fn verify_cell(
        &self,
        reports: &[WorkerReport],
        cpuset: Option<&BTreeSet<usize>>,
    ) -> VerifyResult {
        let mut r = VerifyResult::pass();
        if self.not_starved {
            let mut cell_result = verify_not_starved(reports);
            // Apply custom spread threshold if set.
            if let Some(spread_limit) = self.max_spread_pct {
                // Re-check spread against custom threshold. The default
                // verify_not_starved uses spread_threshold_pct(); clear
                // those failures and re-evaluate.
                cell_result.details.retain(|d| !d.contains("unfair"));
                if let Some(cg) = cell_result.stats.cgroups.first() {
                    if cg.spread > spread_limit && cg.num_workers >= 2 {
                        cell_result.passed = false;
                        cell_result.details.push(format!(
                            "unfair cgroup: spread={:.0}% ({:.0}-{:.0}%) {} workers on {} cpus (threshold {:.0}%)",
                            cg.spread, cg.min_runnable_pct, cg.max_runnable_pct,
                            cg.num_workers, cg.num_cpus, spread_limit
                        ));
                    } else {
                        // Re-derive passed: only non-spread failures matter.
                        cell_result.passed = !cell_result
                            .details
                            .iter()
                            .any(|d| d.contains("starved") || d.contains("stuck"));
                    }
                }
            }
            // Apply custom gap threshold if set.
            if let Some(threshold) = self.max_gap_ms {
                // Re-check gaps against custom threshold. The default
                // verify_not_starved uses 2000ms; clear those failures
                // and re-evaluate.
                cell_result.details.retain(|d| !d.contains("stuck"));
                let had_gap_failure = reports.iter().any(|w| w.max_gap_ms > threshold);
                if had_gap_failure {
                    cell_result.passed = false;
                    for w in reports {
                        if w.max_gap_ms > threshold {
                            cell_result.details.push(format!(
                                "stuck {}ms on cpu{} at +{}ms (threshold {}ms)",
                                w.max_gap_ms, w.max_gap_cpu, w.max_gap_at_ms, threshold
                            ));
                        }
                    }
                } else {
                    // Re-derive passed: only non-gap failures matter.
                    cell_result.passed = !cell_result
                        .details
                        .iter()
                        .any(|d| d.contains("starved") || d.contains("unfair"));
                }
            }
            r.merge(cell_result);
        }
        if self.isolation
            && let Some(cs) = cpuset
        {
            r.merge(verify_isolation(reports, cs));
        }
        r
    }
}

impl Default for VerificationPlan {
    fn default() -> Self {
        Self::new()
    }
}

/// Unified verification configuration. Carries both worker checks and
/// monitor thresholds as a single composable type. Each `Option` field
/// acts as an override — `None` means "inherit from parent layer".
///
/// Merge order: `Verify::default_checks()` -> `Scheduler.verify` -> per-test `verify`.
#[derive(Clone, Copy, Debug)]
pub struct Verify {
    // Worker checks
    pub not_starved: Option<bool>,
    pub isolation: Option<bool>,
    pub max_gap_ms: Option<u64>,
    pub max_spread_pct: Option<f64>,

    // Monitor checks
    pub max_imbalance_ratio: Option<f64>,
    pub max_local_dsq_depth: Option<u32>,
    pub fail_on_stall: Option<bool>,
    pub sustained_samples: Option<usize>,
    pub max_fallback_rate: Option<f64>,
    pub max_keep_last_rate: Option<f64>,
}

impl Verify {
    /// Empty verify — no checks enabled, all overrides None.
    pub const NONE: Verify = Verify {
        not_starved: None,
        isolation: None,
        max_gap_ms: None,
        max_spread_pct: None,
        max_imbalance_ratio: None,
        max_local_dsq_depth: None,
        fail_on_stall: None,
        sustained_samples: None,
        max_fallback_rate: None,
        max_keep_last_rate: None,
    };

    /// Default checks: not_starved enabled, monitor thresholds from
    /// `MonitorThresholds::DEFAULT`.
    pub const fn default_checks() -> Verify {
        use crate::monitor::MonitorThresholds;
        Verify {
            not_starved: Some(true),
            isolation: None,
            max_gap_ms: None,
            max_spread_pct: None,
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

    /// Merge `other` on top of `self`. Each `Some` field in `other`
    /// overrides the corresponding field in `self`; `None` fields
    /// inherit from `self`.
    ///
    /// Use when composing scheduler-level and test-level overrides:
    /// `Verify::default_checks().merge(&scheduler.verify).merge(&test.verify)`.
    pub const fn merge(&self, other: &Verify) -> Verify {
        Verify {
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

    /// Extract a `VerificationPlan` for worker-side checks.
    pub fn worker_plan(&self) -> VerificationPlan {
        VerificationPlan {
            not_starved: self.not_starved.unwrap_or(false),
            isolation: self.isolation.unwrap_or(false),
            max_gap_ms: self.max_gap_ms,
            max_spread_pct: self.max_spread_pct,
        }
    }

    /// Run worker checks against one cgroup's reports.
    ///
    /// Equivalent to `self.worker_plan().verify_cell(reports, cpuset)`.
    pub fn verify_cell(
        &self,
        reports: &[crate::workload::WorkerReport],
        cpuset: Option<&BTreeSet<usize>>,
    ) -> VerifyResult {
        self.worker_plan().verify_cell(reports, cpuset)
    }

    /// Extract `MonitorThresholds` for monitor-side evaluation.
    pub fn monitor_thresholds(&self) -> crate::monitor::MonitorThresholds {
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
pub fn verify_isolation(reports: &[WorkerReport], expected: &BTreeSet<usize>) -> VerifyResult {
    let mut r = VerifyResult::pass();
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

/// Verify one cgroup's workers. Returns per-cgroup stats.
pub fn verify_not_starved(reports: &[WorkerReport]) -> VerifyResult {
    let mut r = VerifyResult::pass();
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

    let cg = CgroupStats {
        num_workers: reports.len(),
        num_cpus: cpus.len(),
        avg_runnable_pct: avg,
        min_runnable_pct: min,
        max_runnable_pct: max,
        spread,
        max_gap_ms: gap_ms,
        max_gap_cpu: gap_cpu,
        total_migrations: reports.iter().map(|w| w.migration_count).sum(),
    };

    // Per-cgroup fairness: spread above threshold means unequal scheduling within a cgroup
    let spread_limit = spread_threshold_pct();
    if spread > spread_limit && pcts.len() >= 2 {
        if !WARN_UNFAIR.load(Ordering::Relaxed) {
            r.passed = false;
        }
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
        cgroups: vec![cg],
        total_workers: reports.len(),
        total_cpus: cpus.len(),
        total_migrations: reports.iter().map(|w| w.migration_count).sum(),
        worst_spread: spread,
        worst_gap_ms: gap_ms,
        worst_gap_cpu: gap_cpu,
    };

    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workload::WorkerReport;
    use std::sync::Mutex;

    /// Serializes tests that mutate the global WARN_UNFAIR flag.
    static WARN_UNFAIR_LOCK: Mutex<()> = Mutex::new(());

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
        }
    }

    #[test]
    fn healthy_pass() {
        let r = verify_not_starved(&[
            rpt(1, 1000, 5_000_000_000, 500_000_000, &[0, 1], 50),
            rpt(2, 1000, 5_000_000_000, 600_000_000, &[0, 1], 60),
            rpt(3, 1000, 5_000_000_000, 550_000_000, &[0, 1], 45),
        ]);
        assert!(r.passed, "{:?}", r.details);
    }

    #[test]
    fn starved_fail() {
        let r = verify_not_starved(&[
            rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50),
            rpt(2, 0, 5e9 as u64, 5e9 as u64, &[0], 50),
        ]);
        assert!(!r.passed);
        assert!(r.details.iter().any(|d| d.contains("starved")));
    }

    #[test]
    fn unfair_spread_fail() {
        let _guard = WARN_UNFAIR_LOCK.lock().unwrap();
        let r = verify_not_starved(&[
            rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50), // 10%
            rpt(2, 500, 5e9 as u64, 4e9 as u64, &[0, 1], 50),  // 80%
            rpt(3, 800, 5e9 as u64, 2e9 as u64, &[0, 1], 50),  // 40%
        ]);
        assert!(!r.passed);
        assert!(r.details.iter().any(|d| d.contains("unfair")));
    }

    #[test]
    fn fair_oversubscribed_pass() {
        let r = verify_not_starved(&[
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
        let r = verify_not_starved(&[
            rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50),
            rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[0], threshold + 500),
        ]);
        assert!(!r.passed);
        assert!(r.details.iter().any(|d| d.contains("stuck")));
    }

    #[test]
    fn isolation_pass() {
        let expected: BTreeSet<usize> = [0, 1, 2, 3].into_iter().collect();
        let r = verify_isolation(
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
        let r = verify_isolation(
            &[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1, 4], 50)],
            &expected,
        );
        assert!(!r.passed);
        assert!(r.details.iter().any(|d| d.contains("unexpected")));
    }

    #[test]
    fn merge_cgroups() {
        let r1 = verify_not_starved(&[
            rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50),
            rpt(2, 1000, 5e9 as u64, 6e8 as u64, &[0, 1], 60),
        ]);
        let r2 = verify_not_starved(&[
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
        let _guard = WARN_UNFAIR_LOCK.lock().unwrap();
        let threshold = spread_threshold_pct();
        // At threshold exactly - pass
        // Worker 1: 10% runnable, Worker 2: 10%+threshold runnable
        let at_threshold_ns = ((10.0 + threshold) / 100.0 * 5e9) as u64;
        let r = verify_not_starved(&[
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
        let r = verify_not_starved(&[
            rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50), // 10%
            rpt(2, 1000, 5e9 as u64, above_ns, &[0], 50),   // 10% + threshold + 5%
        ]);
        assert!(!r.passed, "spread above {threshold}% should fail");
    }

    #[test]
    fn empty_pass() {
        assert!(verify_not_starved(&[]).passed);
    }

    #[test]
    fn zero_wall_time() {
        let r = verify_not_starved(&[
            rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50),
            rpt(2, 0, 0, 0, &[], 0),
        ]);
        assert!(!r.passed);
        assert!(r.details.iter().any(|d| d.contains("starved")));
    }

    #[test]
    fn single_worker_always_pass() {
        let r = verify_not_starved(&[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50)]);
        assert!(r.passed);
        assert_eq!(r.stats.total_workers, 1);
        assert_eq!(r.stats.cgroups.len(), 1);
    }

    #[test]
    fn stats_accuracy() {
        let r = verify_not_starved(&[
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
        let r1 = verify_not_starved(&[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 100)]);
        let r2 = verify_not_starved(&[rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[1], 500)]);
        let mut m = r1;
        m.merge(r2);
        assert_eq!(m.stats.worst_gap_ms, 500);
        assert_eq!(m.stats.worst_gap_cpu, 1);
    }

    #[test]
    fn merge_takes_worst_spread() {
        let r1 = verify_not_starved(&[
            rpt(1, 1000, 5e9 as u64, 1e9 as u64, &[0], 50),
            rpt(2, 1000, 5e9 as u64, 12e8 as u64, &[0], 50),
        ]); // spread = 4%
        let r2 = verify_not_starved(&[
            rpt(3, 1000, 5e9 as u64, 1e9 as u64, &[1], 50),
            rpt(4, 1000, 5e9 as u64, 15e8 as u64, &[1], 50),
        ]); // spread = 10%
        let mut m = r1;
        m.merge(r2);
        assert!((m.stats.worst_spread - 10.0).abs() < 0.1);
    }

    #[test]
    fn merge_accumulates_totals() {
        let r1 = verify_not_starved(&[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50)]);
        let r2 = verify_not_starved(&[rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[1], 50)]);
        let mut m = r1;
        m.merge(r2);
        assert_eq!(m.stats.total_workers, 2);
        assert_eq!(m.stats.total_cpus, 2);
    }

    #[test]
    fn isolation_empty_reports() {
        let expected: BTreeSet<usize> = [0, 1].into_iter().collect();
        assert!(verify_isolation(&[], &expected).passed);
    }

    #[test]
    fn gap_boundary_at_threshold_pass() {
        let threshold = gap_threshold_ms();
        let r = verify_not_starved(&[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], threshold)]);
        assert!(r.passed, "gap at threshold should pass: {:?}", r.details);
    }

    #[test]
    fn gap_boundary_above_threshold_fail() {
        let threshold = gap_threshold_ms();
        let r = verify_not_starved(&[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], threshold + 1)]);
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
            }],
            total_workers: 4,
            total_cpus: 2,
            total_migrations: 10,
            worst_spread: 20.0,
            worst_gap_ms: 150,
            worst_gap_cpu: 3,
        };
        let json = serde_json::to_string(&s).unwrap();
        let s2: ScenarioStats = serde_json::from_str(&json).unwrap();
        assert_eq!(s.total_workers, s2.total_workers);
        assert_eq!(s.worst_gap_ms, s2.worst_gap_ms);
        assert_eq!(s.cgroups.len(), s2.cgroups.len());
        assert_eq!(s.cgroups[0].num_workers, s2.cgroups[0].num_workers);
    }

    #[test]
    fn verify_result_serde_roundtrip() {
        let r = VerifyResult {
            passed: false,
            details: vec!["test".into()],
            stats: Default::default(),
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: VerifyResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r.passed, r2.passed);
        assert_eq!(r.details, r2.details);
    }

    #[test]
    fn multiple_stuck_workers() {
        let threshold = gap_threshold_ms();
        let r = verify_not_starved(&[
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
        let r = verify_not_starved(&[report]);
        assert_eq!(r.stats.total_migrations, 5);
    }

    // VerificationPlan tests

    #[test]
    fn plan_default_empty() {
        let plan = VerificationPlan::new();
        assert!(!plan.not_starved);
        assert!(!plan.isolation);
        assert!(plan.max_gap_ms.is_none());
        assert!(plan.max_spread_pct.is_none());
    }

    #[test]
    fn plan_check_not_starved() {
        let plan = VerificationPlan::new().check_not_starved();
        let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50)];
        let r = plan.verify_cell(&reports, None);
        assert!(r.passed);
        assert_eq!(r.stats.total_workers, 1);
    }

    #[test]
    fn plan_check_isolation_with_cpuset() {
        let plan = VerificationPlan::new()
            .check_not_starved()
            .check_isolation();
        let expected: BTreeSet<usize> = [0, 1].into_iter().collect();
        let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1, 4], 50)];
        let r = plan.verify_cell(&reports, Some(&expected));
        assert!(!r.passed);
        assert!(r.details.iter().any(|d| d.contains("unexpected")));
    }

    #[test]
    fn plan_isolation_skipped_without_cpuset() {
        let plan = VerificationPlan::new().check_isolation();
        let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1, 4], 50)];
        // No cpuset provided -- isolation check is skipped.
        let r = plan.verify_cell(&reports, None);
        assert!(r.passed);
    }

    #[test]
    fn plan_custom_gap_threshold_pass() {
        let plan = VerificationPlan::new().check_not_starved().max_gap_ms(3000);
        // 2500ms gap: passes with 3000ms threshold.
        let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 2500)];
        let r = plan.verify_cell(&reports, None);
        assert!(r.passed, "2500ms < 3000ms threshold: {:?}", r.details);
    }

    #[test]
    fn plan_custom_gap_threshold_fail() {
        let plan = VerificationPlan::new().check_not_starved().max_gap_ms(1500);
        // 2000ms gap: fails with 1500ms threshold.
        let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 2000)];
        let r = plan.verify_cell(&reports, None);
        assert!(!r.passed);
        assert!(r.details.iter().any(|d| d.contains("stuck")));
        assert!(r.details.iter().any(|d| d.contains("threshold 1500ms")));
    }

    #[test]
    fn plan_no_checks_always_passes() {
        let plan = VerificationPlan::new();
        let reports = [rpt(1, 0, 0, 0, &[], 5000)]; // starved + stuck
        let r = plan.verify_cell(&reports, None);
        assert!(r.passed, "no checks enabled should pass");
    }

    #[test]
    fn plan_default_all_checks_disabled() {
        // Default::default() must produce the same state as new() —
        // all checks disabled, no gap override.
        let plan = VerificationPlan::default();
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
        let r = plan.verify_cell(&reports, None);
        assert!(r.passed, "all-disabled plan must pass any input");
    }

    #[test]
    fn verification_plan_default_equals_new() {
        // Default impl calls new(). Verify field-by-field equivalence
        // and that both produce identical verify_cell results.
        let d = VerificationPlan::default();
        let n = VerificationPlan::new();
        assert_eq!(d.not_starved, n.not_starved);
        assert_eq!(d.isolation, n.isolation);
        assert_eq!(d.max_gap_ms, n.max_gap_ms);
        assert_eq!(d.max_spread_pct, n.max_spread_pct);
        // Both should produce identical pass/fail on the same input.
        let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50)];
        let rd = d.verify_cell(&reports, None);
        let rn = n.verify_cell(&reports, None);
        assert_eq!(rd.passed, rn.passed);
    }

    #[test]
    fn single_worker_spread_zero() {
        let r = verify_not_starved(&[rpt(1, 500, 5e9 as u64, 25e8 as u64, &[0, 1], 50)]);
        assert!(r.passed);
        let c = &r.stats.cgroups[0];
        assert!((c.spread - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn zero_wall_time_nonzero_work() {
        // wall_time=0 but work_units>0: the worker did work but the timer
        // didn't advance. Should not produce a starved failure since work was done.
        // The runnable_pct computation skips this worker (no pcts entry).
        let r = verify_not_starved(&[rpt(1, 100, 0, 0, &[0], 0)]);
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
        let r = verify_isolation(
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
        let r = verify_isolation(&[rpt(1, 0, 0, 0, &[], 0)], &expected);
        assert!(r.passed);
    }

    #[test]
    fn isolation_all_unexpected_cpus() {
        let expected: BTreeSet<usize> = [0, 1].into_iter().collect();
        let r = verify_isolation(
            &[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[4, 5, 6], 50)],
            &expected,
        );
        assert!(!r.passed);
        assert!(r.details.iter().any(|d| d.contains("unexpected")));
    }

    #[test]
    fn merge_pass_and_fail() {
        let pass = VerifyResult::pass();
        let mut fail = VerifyResult::pass();
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
        let mut fail = VerifyResult::pass();
        fail.passed = false;
        fail.details.push("first failed".into());
        let pass = VerifyResult::pass();

        let mut merged = fail;
        merged.merge(pass);
        assert!(!merged.passed, "merging fail+pass must produce fail");
    }

    #[test]
    fn warn_unfair_downgrades_to_warning() {
        let _guard = WARN_UNFAIR_LOCK.lock().unwrap();
        // With WARN_UNFAIR=true, unfair spread should NOT fail the result
        // (it still adds the detail string but passed stays true).
        set_warn_unfair(true);
        let r = verify_not_starved(&[
            rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50), // 10%
            rpt(2, 500, 5e9 as u64, 4e9 as u64, &[0, 1], 50),  // 80%
        ]);
        // Reset before assertions so other tests are unaffected.
        set_warn_unfair(false);
        // The detail about unfair should still be present.
        assert!(r.details.iter().any(|d| d.contains("unfair")));
        // But passed should be true because WARN_UNFAIR was set.
        assert!(r.passed, "with WARN_UNFAIR=true, unfair should not fail");
    }

    #[test]
    fn plan_starved_still_fails_with_custom_gap() {
        // A starved worker (work_units=0) must still cause failure even
        // when the custom max_gap_ms threshold is high enough that the
        // gap check passes.
        let plan = VerificationPlan::new().check_not_starved().max_gap_ms(5000);
        let reports = [
            rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 100), // healthy
            rpt(2, 0, 5e9 as u64, 0, &[1], 1500),            // starved, gap < threshold
        ];
        let r = plan.verify_cell(&reports, None);
        assert!(
            !r.passed,
            "starved worker must fail even with relaxed gap threshold"
        );
        assert!(r.details.iter().any(|d| d.contains("starved")));
        // The gap (1500ms) is below the 5000ms threshold, so no "stuck" detail.
        assert!(!r.details.iter().any(|d| d.contains("stuck")));
    }

    #[test]
    fn warn_unfair_false_fails() {
        let _guard = WARN_UNFAIR_LOCK.lock().unwrap();
        // With WARN_UNFAIR=false (default), unfair spread fails.
        set_warn_unfair(false);
        let r = verify_not_starved(&[
            rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50), // 10%
            rpt(2, 500, 5e9 as u64, 4e9 as u64, &[0, 1], 50),  // 80%
        ]);
        assert!(!r.passed);
        assert!(r.details.iter().any(|d| d.contains("unfair")));
    }

    // -- Verify merge tests --

    #[test]
    fn verify_none_has_no_checks() {
        let v = Verify::NONE;
        assert!(v.not_starved.is_none());
        assert!(v.isolation.is_none());
        assert!(v.max_gap_ms.is_none());
        assert!(v.max_spread_pct.is_none());
        assert!(v.max_imbalance_ratio.is_none());
    }

    #[test]
    fn verify_default_checks_enables_not_starved() {
        let v = Verify::default_checks();
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
    fn verify_merge_other_overrides_self() {
        let base = Verify::NONE;
        let other = Verify::NONE
            .check_not_starved()
            .max_gap_ms(5000)
            .max_imbalance_ratio(2.0);
        let merged = base.merge(&other);
        assert_eq!(merged.not_starved, Some(true));
        assert_eq!(merged.max_gap_ms, Some(5000));
        assert_eq!(merged.max_imbalance_ratio, Some(2.0));
    }

    #[test]
    fn verify_merge_preserves_self_when_other_is_none() {
        let base = Verify::default_checks();
        let merged = base.merge(&Verify::NONE);
        assert_eq!(merged.not_starved, Some(true));
        assert!(merged.max_imbalance_ratio.is_some());
        assert!(merged.max_local_dsq_depth.is_some());
    }

    #[test]
    fn verify_merge_other_takes_precedence() {
        let base = Verify::NONE.max_imbalance_ratio(4.0);
        let other = Verify::NONE.max_imbalance_ratio(2.0);
        let merged = base.merge(&other);
        assert_eq!(merged.max_imbalance_ratio, Some(2.0));
    }

    #[test]
    fn verify_merge_last_some_wins() {
        let base = Verify::NONE.check_not_starved();
        let other = Verify::NONE.check_isolation();
        let merged = base.merge(&other);
        assert_eq!(merged.not_starved, Some(true));
        assert_eq!(merged.isolation, Some(true));
    }

    #[test]
    fn verify_merge_child_disables_not_starved() {
        let base = Verify::default_checks(); // not_starved = Some(true)
        let other = Verify {
            not_starved: Some(false),
            ..Verify::NONE
        };
        let merged = base.merge(&other);
        assert_eq!(merged.not_starved, Some(false));
        assert!(!merged.worker_plan().not_starved);
    }

    #[test]
    fn verify_merge_child_disables_isolation() {
        let base = Verify::NONE.check_isolation(); // isolation = Some(true)
        let other = Verify {
            isolation: Some(false),
            ..Verify::NONE
        };
        let merged = base.merge(&other);
        assert_eq!(merged.isolation, Some(false));
        assert!(!merged.worker_plan().isolation);
    }

    #[test]
    fn verify_worker_plan_extraction() {
        let v = Verify::NONE
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
    fn verify_verify_cell_delegates_to_plan() {
        let v = Verify::NONE.check_not_starved();
        let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50)];
        let r = v.verify_cell(&reports, None);
        assert!(r.passed);
        assert_eq!(r.stats.total_workers, 1);
    }

    #[test]
    fn verify_monitor_thresholds_extraction() {
        let v = Verify::NONE
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
    fn verify_monitor_thresholds_defaults_when_none() {
        let v = Verify::NONE;
        let t = v.monitor_thresholds();
        let d = crate::monitor::MonitorThresholds::DEFAULT;
        assert!((t.max_imbalance_ratio - d.max_imbalance_ratio).abs() < f64::EPSILON);
        assert_eq!(t.max_local_dsq_depth, d.max_local_dsq_depth);
    }

    #[test]
    fn verify_chain_all_setters() {
        let v = Verify::NONE
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

    // -- gap_threshold_ms / set_coverage_gap_ms tests --

    /// Serializes tests that mutate COVERAGE_GAP_MS.
    static GAP_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn gap_threshold_default() {
        let _guard = GAP_LOCK.lock().unwrap();
        set_coverage_gap_ms(0);
        let t = gap_threshold_ms();
        if cfg!(debug_assertions) {
            assert_eq!(t, 3000);
        } else {
            assert_eq!(t, 2000);
        }
    }

    #[test]
    fn gap_threshold_custom() {
        let _guard = GAP_LOCK.lock().unwrap();
        set_coverage_gap_ms(5000);
        assert_eq!(gap_threshold_ms(), 5000);
        set_coverage_gap_ms(0);
    }

    #[test]
    fn verify_result_pass_defaults() {
        let r = VerifyResult::pass();
        assert!(r.passed);
        assert!(r.details.is_empty());
        assert_eq!(r.stats.total_workers, 0);
    }

    // -- Verify::merge per-field tests --

    #[test]
    fn verify_merge_max_spread_pct() {
        let base = Verify::NONE.max_spread_pct(10.0);
        let other = Verify::NONE.max_spread_pct(5.0);
        assert_eq!(base.merge(&other).max_spread_pct, Some(5.0));
        assert_eq!(base.merge(&Verify::NONE).max_spread_pct, Some(10.0));
    }

    #[test]
    fn verify_merge_fail_on_stall() {
        let base = Verify::NONE.fail_on_stall(true);
        let other = Verify::NONE.fail_on_stall(false);
        assert_eq!(base.merge(&other).fail_on_stall, Some(false));
        assert_eq!(base.merge(&Verify::NONE).fail_on_stall, Some(true));
    }

    #[test]
    fn verify_merge_sustained_samples() {
        let base = Verify::NONE.sustained_samples(5);
        let other = Verify::NONE.sustained_samples(10);
        assert_eq!(base.merge(&other).sustained_samples, Some(10));
        assert_eq!(base.merge(&Verify::NONE).sustained_samples, Some(5));
    }

    #[test]
    fn verify_merge_max_fallback_rate() {
        let base = Verify::NONE.max_fallback_rate(200.0);
        let other = Verify::NONE.max_fallback_rate(50.0);
        assert_eq!(base.merge(&other).max_fallback_rate, Some(50.0));
        assert_eq!(base.merge(&Verify::NONE).max_fallback_rate, Some(200.0));
    }

    #[test]
    fn verify_merge_max_keep_last_rate() {
        let base = Verify::NONE.max_keep_last_rate(100.0);
        let other = Verify::NONE.max_keep_last_rate(25.0);
        assert_eq!(base.merge(&other).max_keep_last_rate, Some(25.0));
        assert_eq!(base.merge(&Verify::NONE).max_keep_last_rate, Some(100.0));
    }

    #[test]
    fn verify_merge_max_local_dsq_depth() {
        let base = Verify::NONE.max_local_dsq_depth(50);
        let other = Verify::NONE.max_local_dsq_depth(100);
        assert_eq!(base.merge(&other).max_local_dsq_depth, Some(100));
        assert_eq!(base.merge(&Verify::NONE).max_local_dsq_depth, Some(50));
    }

    #[test]
    fn verify_merge_max_gap_ms() {
        let base = Verify::NONE.max_gap_ms(2000);
        let other = Verify::NONE.max_gap_ms(5000);
        assert_eq!(base.merge(&other).max_gap_ms, Some(5000));
        assert_eq!(base.merge(&Verify::NONE).max_gap_ms, Some(2000));
    }

    #[test]
    fn verify_merge_three_layers() {
        let defaults = Verify::default_checks();
        let sched = Verify::NONE
            .max_imbalance_ratio(2.0)
            .max_fallback_rate(50.0);
        let test = Verify::NONE.max_gap_ms(5000);
        let merged = defaults.merge(&sched).merge(&test);
        assert_eq!(merged.not_starved, Some(true));
        assert_eq!(merged.max_imbalance_ratio, Some(2.0));
        assert_eq!(merged.max_fallback_rate, Some(50.0));
        assert_eq!(merged.max_gap_ms, Some(5000));
        assert_eq!(merged.sustained_samples, Some(5));
    }

    // ---------------------------------------------------------------
    // Negative tests: verify that diagnostics catch controlled failures
    // ---------------------------------------------------------------

    #[test]
    fn neg_starvation_zero_work_detected() {
        let r = verify_not_starved(&[
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
        let r = verify_isolation(&reports, &expected);
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
        let _guard = WARN_UNFAIR_LOCK.lock().unwrap();
        set_warn_unfair(false);
        let r = verify_not_starved(&[
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
        let r = verify_not_starved(&[
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
        let plan = VerificationPlan::new().check_not_starved().max_gap_ms(500);
        let reports = [
            rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50),
            rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[1], 1000),
        ];
        let r = plan.verify_cell(&reports, None);
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
        let _guard = WARN_UNFAIR_LOCK.lock().unwrap();
        set_warn_unfair(false);
        let plan = VerificationPlan::new()
            .check_not_starved()
            .check_isolation();
        let expected: BTreeSet<usize> = [0, 1].into_iter().collect();
        let reports = [
            rpt(1, 0, 5e9 as u64, 0, &[0], 0),
            rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[4, 5], 50),
        ];
        let r = plan.verify_cell(&reports, Some(&expected));
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
    fn neg_verify_cell_via_verify_struct() {
        let v = Verify::NONE.check_not_starved().check_isolation();
        let expected: BTreeSet<usize> = [0].into_iter().collect();
        let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1, 2], 50)];
        let r = v.verify_cell(&reports, Some(&expected));
        assert!(!r.passed, "Verify.verify_cell must catch isolation failure");
        let detail = r.details.iter().find(|d| d.contains("unexpected")).unwrap();
        assert!(detail.contains("tid 1"), "must name tid: {detail}");
        assert!(detail.contains("1"), "must list CPU 1: {detail}");
        assert!(detail.contains("2"), "must list CPU 2: {detail}");
    }

    #[test]
    fn verify_merge_none_preserves_base() {
        let base = Verify::default_checks();
        let merged = base.merge(&Verify::NONE);
        assert_eq!(merged.not_starved, Some(true));
        assert!(merged.max_imbalance_ratio.is_some());
        assert!(merged.fail_on_stall.is_some());
    }

    #[test]
    fn verify_merge_overrides_fields() {
        let base = Verify::NONE;
        let overrides = Verify::NONE
            .max_imbalance_ratio(5.0)
            .max_gap_ms(1000)
            .check_not_starved();
        let merged = base.merge(&overrides);
        assert_eq!(merged.not_starved, Some(true));
        assert_eq!(merged.max_imbalance_ratio, Some(5.0));
        assert_eq!(merged.max_gap_ms, Some(1000));
    }

    #[test]
    fn verify_merge_later_overrides_earlier() {
        let a = Verify::NONE.max_imbalance_ratio(2.0);
        let b = Verify::NONE.max_imbalance_ratio(10.0);
        let merged = a.merge(&b);
        assert_eq!(merged.max_imbalance_ratio, Some(10.0));
    }

    #[test]
    fn verify_worker_plan_extracts_fields() {
        let v = Verify::NONE
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
    fn verify_monitor_thresholds_defaults() {
        let v = Verify::NONE;
        let t = v.monitor_thresholds();
        // Should use MonitorThresholds::DEFAULT values.
        let d = crate::monitor::MonitorThresholds::DEFAULT;
        assert_eq!(t.max_imbalance_ratio, d.max_imbalance_ratio);
        assert_eq!(t.max_local_dsq_depth, d.max_local_dsq_depth);
    }

    #[test]
    fn verify_monitor_thresholds_overridden() {
        let v = Verify::NONE
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
    fn verify_max_spread_pct() {
        let v = Verify::NONE.max_spread_pct(25.0);
        assert_eq!(v.max_spread_pct, Some(25.0));
    }

    #[test]
    fn gap_threshold_debug_vs_release() {
        let t = gap_threshold_ms();
        // In test builds (debug_assertions=true), threshold is 3000.
        assert!(t >= 2000, "threshold should be at least 2000ms: {t}");
    }

    #[test]
    fn set_coverage_gap_ms_overrides_default() {
        let prev = COVERAGE_GAP_MS.load(Ordering::Relaxed);
        set_coverage_gap_ms(9999);
        assert_eq!(gap_threshold_ms(), 9999);
        set_coverage_gap_ms(prev);
    }

    #[test]
    fn verify_result_merge_combines_stats() {
        let mut a = VerifyResult {
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
            },
        };
        let b = VerifyResult {
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
        let plan = VerificationPlan::new().check_not_starved().max_gap_ms(5000);
        let reports = [
            rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50),
            rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[1], 1000),
        ];
        let r = plan.verify_cell(&reports, None);
        // 1000ms gap < 5000ms threshold, so it passes.
        let has_stuck = r.details.iter().any(|d| d.contains("stuck"));
        assert!(!has_stuck, "1000ms gap should pass 5000ms threshold");
    }
}
