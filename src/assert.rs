//! Pass/fail evaluation of scenario results.
//!
//! Key types:
//! - [`AssertResult`] -- pass/fail status with diagnostics and statistics
//! - [`Assert`] -- composable assertion config (worker + monitor checks)
//! - [`ScenarioStats`] / [`CgroupStats`] -- aggregated telemetry
//! - [`NumaMapsEntry`] -- parsed `/proc/self/numa_maps` VMA entry
//!
//! NUMA assertion functions:
//! - [`parse_numa_maps`] -- parse numa_maps content into per-VMA entries
//! - [`page_locality`] -- compute page locality fraction from entries
//! - [`parse_vmstat_numa_pages_migrated`] -- extract vmstat migration counter
//! - [`assert_page_locality`] / [`assert_cross_node_migration`] -- threshold checks
//!
//! Assertion uses a three-layer merge: [`Assert::default_checks()`] ->
//! `Scheduler.assert` -> per-test `assert`.
//!
//! See the [Verification](https://likewhatevs.github.io/ktstr/guide/concepts/verification.html)
//! chapter of the guide.

use crate::workload::WorkerReport;
use std::collections::{BTreeMap, BTreeSet};

/// Per-VMA entry parsed from `/proc/self/numa_maps`.
#[derive(Debug, Clone, Default)]
pub struct NumaMapsEntry {
    /// Virtual address of the VMA.
    pub addr: u64,
    /// Per-node page counts (node_id -> page_count).
    pub node_pages: BTreeMap<usize, u64>,
}

/// Parse `/proc/self/numa_maps` content into per-VMA entries.
///
/// Each line has the format:
///   `<hex_addr> <policy> [key=val ...]`
/// where per-node page counts appear as `N<node>=<count>`.
pub fn parse_numa_maps(content: &str) -> Vec<NumaMapsEntry> {
    let mut entries = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let addr = match parts.next().and_then(|s| u64::from_str_radix(s, 16).ok()) {
            Some(a) => a,
            None => continue,
        };
        // Skip policy field.
        let _ = parts.next();

        let mut entry = NumaMapsEntry {
            addr,
            ..Default::default()
        };

        for token in parts {
            if let Some(rest) = token.strip_prefix('N')
                && let Some((node_str, count_str)) = rest.split_once('=')
                && let (Ok(node), Ok(count)) = (node_str.parse::<usize>(), count_str.parse::<u64>())
            {
                *entry.node_pages.entry(node).or_insert(0) += count;
            }
        }

        if !entry.node_pages.is_empty() {
            entries.push(entry);
        }
    }
    entries
}

/// Compute page locality fraction from parsed numa_maps entries.
///
/// Returns the fraction of pages residing on any node in
/// `expected_nodes` (0.0-1.0). Returns 1.0 when no pages are observed
/// (vacuously local). The expected node set is derived from the
/// worker's [`MemPolicy`](crate::workload::MemPolicy) at evaluation
/// time.
pub fn page_locality(entries: &[NumaMapsEntry], expected_nodes: &BTreeSet<usize>) -> f64 {
    let mut total: u64 = 0;
    let mut local: u64 = 0;
    for entry in entries {
        for (&node, &count) in &entry.node_pages {
            total += count;
            if expected_nodes.contains(&node) {
                local += count;
            }
        }
    }
    if total > 0 {
        local as f64 / total as f64
    } else {
        1.0
    }
}

/// Extract `numa_pages_migrated` from `/proc/vmstat` content.
///
/// Returns `None` if the counter is not present. The counter is
/// cumulative; callers diff pre- and post-workload snapshots to
/// get migration count during the test.
pub fn parse_vmstat_numa_pages_migrated(content: &str) -> Option<u64> {
    for line in content.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("numa_pages_migrated") {
            let rest = rest.trim();
            if let Ok(v) = rest.parse::<u64>() {
                return Some(v);
            }
        }
    }
    None
}

fn gap_threshold_ms() -> u64 {
    // Unoptimized debug builds have higher scheduling overhead.
    if cfg!(debug_assertions) { 3000 } else { 2000 }
}

fn spread_threshold_pct() -> f64 {
    // Debug builds in small VMs (especially under EEVDF) show higher
    // spread than optimized builds under sched_ext schedulers.
    if cfg!(debug_assertions) { 35.0 } else { 15.0 }
}

/// Category tag for an [`AssertDetail`]. Enables structural filtering
/// (e.g. by [`AssertPlan`]) without matching on substrings of
/// human-readable messages, which is fragile if wording changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DetailKind {
    /// A worker made zero progress.
    Starved,
    /// A worker was stuck off-CPU longer than the gap threshold.
    Stuck,
    /// Spread between best and worst worker exceeded the fairness threshold.
    Unfair,
    /// A worker ran on a CPU outside its expected cpuset.
    Isolation,
    /// Throughput / benchmarking threshold failure (p99, CV, rate).
    Benchmark,
    /// Migration-ratio threshold failure (migrations per iteration).
    Migration,
    /// NUMA page locality threshold failure.
    PageLocality,
    /// Cross-node migration threshold failure.
    CrossNodeMigration,
    /// Slow-tier (memory tier) threshold failure.
    SlowTier,
    /// Scheduler-health diagnostic — includes monitor-subsystem anomalies
    /// (imbalance, DSQ depth, rq_clock stall) and scheduler-liveness
    /// detection (process crashes).
    Monitor,
    /// Skip notification (scenario could not run under this topology/flags).
    Skip,
    /// Uncategorized — falls through when a detail has no specific kind.
    Other,
}

/// A single diagnostic message from an assertion, paired with a
/// structural [`DetailKind`] so filtering is robust to wording changes.
///
/// `Deref<Target = str>` and `Display` forward to `message` so existing
/// string-based probes (`d.contains("...")`, `format!("{d}")`) keep
/// working; new code that needs to filter by category should match on
/// `kind`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AssertDetail {
    pub kind: DetailKind,
    pub message: String,
}

impl PartialEq<&str> for AssertDetail {
    fn eq(&self, other: &&str) -> bool {
        self.message == *other
    }
}

impl PartialEq<str> for AssertDetail {
    fn eq(&self, other: &str) -> bool {
        self.message == *other
    }
}

impl PartialEq<String> for AssertDetail {
    fn eq(&self, other: &String) -> bool {
        self.message == *other
    }
}

impl AsRef<str> for AssertDetail {
    fn as_ref(&self) -> &str {
        &self.message
    }
}

impl AssertDetail {
    pub fn new(kind: DetailKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl From<String> for AssertDetail {
    /// Conversion for uncategorized messages; defaults `kind` to
    /// [`DetailKind::Other`]. Prefer [`AssertDetail::new`] when the
    /// detail has a meaningful category.
    fn from(message: String) -> Self {
        Self {
            kind: DetailKind::Other,
            message,
        }
    }
}

impl From<&str> for AssertDetail {
    fn from(s: &str) -> Self {
        Self {
            kind: DetailKind::Other,
            message: s.to_string(),
        }
    }
}

impl std::ops::Deref for AssertDetail {
    type Target = str;
    fn deref(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for AssertDetail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// Result of checking a scenario run.
///
/// Contains pass/fail status, human-readable detail messages, and
/// aggregated statistics. Multiple results can be combined with
/// [`merge()`](AssertResult::merge).
///
/// ```
/// # use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
/// let mut a = AssertResult::pass();
/// assert!(a.passed);
///
/// let mut b = AssertResult::pass();
/// b.passed = false;
/// b.details.push(AssertDetail::new(DetailKind::Starved, "worker starved"));
///
/// a.merge(b);
/// assert!(!a.passed);
/// assert!(a.details.iter().any(|d| d.kind == DetailKind::Starved));
/// ```
#[must_use = "test verdict is lost if not checked"]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AssertResult {
    /// Whether all checks passed.
    pub passed: bool,
    /// True when the scenario was skipped (e.g. topology mismatch,
    /// missing resource). `passed` stays `true` for backward compat
    /// with callers that treat skip as "not a failure"; stats tooling
    /// must subtract skipped runs from pass counts so they don't
    /// count as successful executions.
    pub skipped: bool,
    /// Human-readable diagnostic messages (failures, warnings), each
    /// tagged with a [`DetailKind`] for structural filtering.
    pub details: Vec<AssertDetail>,
    /// Aggregated stats from all workers in this scenario.
    #[serde(default)]
    pub stats: ScenarioStats,
}

/// Per-cgroup statistics from worker telemetry.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct CgroupStats {
    /// Number of workers in this cgroup.
    pub num_workers: usize,
    /// Distinct CPUs used across all workers in this cgroup.
    pub num_cpus: usize,
    /// Mean off-CPU percentage across workers (off_cpu_ns / wall_time_ns * 100).
    pub avg_off_cpu_pct: f64,
    /// Minimum off-CPU percentage across workers.
    pub min_off_cpu_pct: f64,
    /// Maximum off-CPU percentage across workers.
    pub max_off_cpu_pct: f64,
    /// max_off_cpu_pct - min_off_cpu_pct. Measures scheduling fairness within the cgroup.
    pub spread: f64,
    /// Longest scheduling gap across all workers (ms).
    pub max_gap_ms: u64,
    /// CPU where the longest scheduling gap occurred.
    pub max_gap_cpu: usize,
    /// Sum of CPU migration counts across all workers.
    pub total_migrations: u64,
    /// Migrations per iteration (total_migrations / total_iterations).
    #[serde(default)]
    pub migration_ratio: f64,
    /// 99th percentile wake latency across all workers (microseconds).
    #[serde(default)]
    pub p99_wake_latency_us: f64,
    /// Median wake latency across all workers (microseconds).
    #[serde(default)]
    pub median_wake_latency_us: f64,
    /// Coefficient of variation (stddev / mean) of wake latencies.
    #[serde(default)]
    pub wake_latency_cv: f64,
    /// Sum of iteration counts across all workers.
    #[serde(default)]
    pub total_iterations: u64,
    /// Mean schedstat run delay across workers (microseconds).
    #[serde(default)]
    pub mean_run_delay_us: f64,
    /// Worst schedstat run delay across workers (microseconds).
    #[serde(default)]
    pub worst_run_delay_us: f64,
    /// Fraction of pages on the expected NUMA node(s) (0.0-1.0).
    /// Derived from `/proc/self/numa_maps` and the worker's
    /// [`MemPolicy`](crate::workload::MemPolicy).
    #[serde(default)]
    pub page_locality: f64,
    /// Cross-node page migration ratio from `/proc/vmstat`
    /// `numa_pages_migrated` delta divided by total allocated pages.
    #[serde(default)]
    pub cross_node_migration_ratio: f64,
    /// Extensible metrics for the generic comparison pipeline.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub ext_metrics: BTreeMap<String, f64>,
}

/// Aggregated statistics across all cgroups in a scenario.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ScenarioStats {
    /// Per-cgroup stats, one entry per cgroup.
    pub cgroups: Vec<CgroupStats>,
    /// Sum of workers across all cgroups.
    pub total_workers: usize,
    /// Sum of per-cgroup distinct CPU counts (not deduplicated across cgroups).
    pub total_cpus: usize,
    /// Sum of migration counts across all cgroups.
    pub total_migrations: u64,
    /// Worst spread across any cgroup.
    pub worst_spread: f64,
    /// Worst gap across any cgroup (ms).
    pub worst_gap_ms: u64,
    /// CPU where the worst gap occurred across all cgroups.
    pub worst_gap_cpu: usize,
    /// Worst migration ratio across any cgroup.
    #[serde(default)]
    pub worst_migration_ratio: f64,
    /// Worst p99 wake latency across all cgroups (microseconds).
    #[serde(default)]
    pub p99_wake_latency_us: f64,
    /// Worst median wake latency across all cgroups (microseconds).
    #[serde(default)]
    pub median_wake_latency_us: f64,
    /// Worst wake latency coefficient of variation across all cgroups.
    #[serde(default)]
    pub wake_latency_cv: f64,
    /// Sum of iteration counts across all cgroups.
    #[serde(default)]
    pub total_iterations: u64,
    /// Worst mean schedstat run delay across all cgroups (microseconds).
    #[serde(default)]
    pub mean_run_delay_us: f64,
    /// Worst schedstat run delay across all cgroups (microseconds).
    #[serde(default)]
    pub worst_run_delay_us: f64,
    /// Worst (lowest) page locality fraction across cgroups.
    #[serde(default)]
    pub worst_page_locality: f64,
    /// Worst (highest) cross-node migration ratio across cgroups.
    #[serde(default)]
    pub worst_cross_node_migration_ratio: f64,
    /// Extensible metrics for the generic comparison pipeline.
    /// Populated from per-cgroup ext_metrics (worst value across cgroups).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub ext_metrics: BTreeMap<String, f64>,
}

impl AssertResult {
    /// Empty passing result with no details and default stats. Use
    /// when a scenario completed successfully with nothing interesting
    /// to report.
    pub fn pass() -> Self {
        Self {
            passed: true,
            skipped: false,
            details: vec![],
            stats: Default::default(),
        }
    }
    /// Pass result with a skip reason. Used when a scenario cannot run
    /// under the current topology or flag combination but is not a failure.
    pub fn skip(reason: impl Into<String>) -> Self {
        Self {
            passed: true,
            skipped: true,
            details: vec![AssertDetail::new(DetailKind::Skip, reason)],
            stats: Default::default(),
        }
    }
    /// Convenience accessor returning [`Self::skipped`]. Stats tooling
    /// uses this to subtract non-executions from pass counts so
    /// "topology mismatch" runs don't inflate the pass rate.
    pub fn is_skipped(&self) -> bool {
        self.skipped
    }
    /// Fold `other` into `self`. `passed` is conjoined (any failure
    /// wins), `details` concatenate, and aggregate stats adopt the
    /// worst-case value per dimension so the merged result represents
    /// the union of all checks applied.
    pub fn merge(&mut self, other: AssertResult) {
        if !other.passed {
            self.passed = false;
        }
        // skip + skip = skipped (nothing executed); skip + pass/fail =
        // NOT skipped (real work ran). Equivalent to logical AND of
        // the two `skipped` flags.
        self.skipped = self.skipped && other.skipped;
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
        // NUMA: worst page locality is the lowest non-zero value.
        if other.stats.worst_page_locality > 0.0
            && (self.stats.worst_page_locality == 0.0
                || other.stats.worst_page_locality < self.stats.worst_page_locality)
        {
            self.stats.worst_page_locality = other.stats.worst_page_locality;
        }
        if other.stats.worst_cross_node_migration_ratio
            > self.stats.worst_cross_node_migration_ratio
        {
            self.stats.worst_cross_node_migration_ratio =
                other.stats.worst_cross_node_migration_ratio;
        }
        // Merge extensible metrics: take worst per key according to
        // each metric's polarity in the MetricDef registry. For
        // `higher_is_worse: true` the worst is max; for
        // `higher_is_worse: false` the worst is min. Unknown metrics
        // default to max (treat them as higher-is-worse until the
        // caller registers a MetricDef — conservative for regressions).
        //
        // `or_insert(*v)` rather than `or_insert(0.0)`: the old sentinel
        // clobbered real-but-small values for min-polarity metrics on
        // first merge, making the subsequent min comparison meaningless.
        for (k, v) in &other.stats.ext_metrics {
            let higher_is_worse = crate::stats::metric_def(k)
                .map(|m| m.higher_is_worse)
                .unwrap_or(true);
            let entry = self.stats.ext_metrics.entry(k.clone()).or_insert(*v);
            *entry = if higher_is_worse {
                entry.max(*v)
            } else {
                entry.min(*v)
            };
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
    pub(crate) min_page_locality: Option<f64>,
    pub(crate) max_cross_node_migration_ratio: Option<f64>,
    pub(crate) max_slow_tier_ratio: Option<f64>,
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
            min_page_locality: None,
            max_cross_node_migration_ratio: None,
            max_slow_tier_ratio: None,
        }
    }

    /// Run all configured checks against one cgroup's reports.
    ///
    /// `cpuset` is the expected CPU set for isolation checks. Pass `None`
    /// when there is no cpuset constraint (isolation check is skipped).
    ///
    /// `numa_nodes` is the NUMA node IDs covered by the cpuset (derived
    /// via `TestTopology::numa_nodes_for_cpuset`). Used for page locality
    /// and slow-tier ratio checks. Pass `None` when NUMA checks are not
    /// applicable.
    pub(crate) fn assert_cgroup(
        &self,
        reports: &[WorkerReport],
        cpuset: Option<&BTreeSet<usize>>,
        numa_nodes: Option<&BTreeSet<usize>>,
    ) -> AssertResult {
        let mut r = AssertResult::pass();
        if self.not_starved {
            let mut cgroup_result = assert_not_starved(reports);
            // Apply custom spread threshold if set.
            if let Some(spread_limit) = self.max_spread_pct {
                // Re-check spread against custom threshold. The default
                // assert_not_starved uses spread_threshold_pct(); clear
                // those failures and re-evaluate.
                cgroup_result
                    .details
                    .retain(|d| d.kind != DetailKind::Unfair);
                if let Some(cg) = cgroup_result.stats.cgroups.first() {
                    if cg.spread > spread_limit && cg.num_workers >= 2 {
                        cgroup_result.passed = false;
                        cgroup_result.details.push(AssertDetail::new(
                            DetailKind::Unfair,
                            format!(
                                "unfair cgroup: spread={:.0}% ({:.0}-{:.0}%) {} workers on {} cpus (threshold {:.0}%)",
                                cg.spread, cg.min_off_cpu_pct, cg.max_off_cpu_pct,
                                cg.num_workers, cg.num_cpus, spread_limit
                            ),
                        ));
                    } else {
                        // Re-derive passed: only non-spread failures matter.
                        cgroup_result.passed = !cgroup_result
                            .details
                            .iter()
                            .any(|d| matches!(d.kind, DetailKind::Starved | DetailKind::Stuck));
                    }
                }
            }
            // Apply custom gap threshold if set.
            if let Some(threshold) = self.max_gap_ms {
                // Re-check gaps against custom threshold. The default
                // assert_not_starved uses gap_threshold_ms() (2000ms
                // release, 3000ms debug); clear those failures and
                // re-evaluate.
                cgroup_result
                    .details
                    .retain(|d| d.kind != DetailKind::Stuck);
                let had_gap_failure = reports.iter().any(|w| w.max_gap_ms > threshold);
                if had_gap_failure {
                    cgroup_result.passed = false;
                    for w in reports {
                        if w.max_gap_ms > threshold {
                            cgroup_result.details.push(AssertDetail::new(
                                DetailKind::Stuck,
                                format!(
                                    "stuck {}ms on cpu{} at +{}ms (threshold {}ms)",
                                    w.max_gap_ms, w.max_gap_cpu, w.max_gap_at_ms, threshold
                                ),
                            ));
                        }
                    }
                } else {
                    // Re-derive passed: only non-gap failures matter.
                    cgroup_result.passed = !cgroup_result
                        .details
                        .iter()
                        .any(|d| matches!(d.kind, DetailKind::Starved | DetailKind::Unfair));
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
                r.details.push(AssertDetail::new(
                    DetailKind::Migration,
                    format!(
                        "migration ratio {:.4} exceeds threshold {:.4} ({} migrations / {} iterations)",
                        ratio, max_ratio, total_mig, total_iters,
                    ),
                ));
            }
        }
        if let Some(min_locality) = self.min_page_locality
            && let Some(nodes) = numa_nodes
        {
            for w in reports {
                if w.numa_pages.is_empty() {
                    continue;
                }
                let total: u64 = w.numa_pages.values().sum();
                let local: u64 = w
                    .numa_pages
                    .iter()
                    .filter(|(node, _)| nodes.contains(node))
                    .map(|(_, count)| count)
                    .sum();
                if total > 0 {
                    let locality = local as f64 / total as f64;
                    r.merge(assert_page_locality(
                        locality,
                        Some(min_locality),
                        total,
                        local,
                    ));
                }
            }
        }
        if let Some(max_ratio) = self.max_cross_node_migration_ratio {
            for w in reports {
                let total: u64 = w.numa_pages.values().sum();
                if total > 0 {
                    r.merge(assert_cross_node_migration(
                        w.vmstat_numa_pages_migrated,
                        total,
                        Some(max_ratio),
                    ));
                }
            }
        }
        if let Some(max_ratio) = self.max_slow_tier_ratio
            && numa_nodes.is_some()
        {
            for w in reports {
                if w.numa_pages.is_empty() {
                    continue;
                }
                let total: u64 = w.numa_pages.values().sum();
                if total > 0 {
                    r.merge(assert_slow_tier_ratio(
                        &w.numa_pages,
                        max_ratio,
                        total,
                        numa_nodes,
                    ));
                }
            }
        }
        r
    }
}

/// Check slow-tier page ratio against threshold.
///
/// "Slow tier" nodes are NUMA nodes NOT in the cpuset's NUMA node set.
/// For CXL memory-only nodes, these are the nodes without CPUs.
fn assert_slow_tier_ratio(
    numa_pages: &BTreeMap<usize, u64>,
    max_ratio: f64,
    total_pages: u64,
    numa_nodes: Option<&BTreeSet<usize>>,
) -> AssertResult {
    let mut r = AssertResult::pass();
    let Some(cpu_nodes) = numa_nodes else {
        return r;
    };
    let slow_pages: u64 = numa_pages
        .iter()
        .filter(|(node, _)| !cpu_nodes.contains(node))
        .map(|(_, count)| count)
        .sum();
    let ratio = slow_pages as f64 / total_pages as f64;
    if ratio > max_ratio {
        r.passed = false;
        r.details.push(AssertDetail::new(
            DetailKind::SlowTier,
            format!(
                "slow-tier page ratio {ratio:.4} exceeds threshold {max_ratio:.4} \
                 ({slow_pages}/{total_pages} pages on non-CPU nodes)",
            ),
        ));
    }
    r
}

/// Check NUMA page locality against threshold.
///
/// `observed` is the fraction of pages on expected nodes (0.0-1.0).
/// `total_pages` and `local_pages` are included in diagnostics.
pub fn assert_page_locality(
    observed: f64,
    min_locality: Option<f64>,
    total_pages: u64,
    local_pages: u64,
) -> AssertResult {
    let mut r = AssertResult::pass();
    if let Some(threshold) = min_locality
        && observed < threshold
    {
        r.passed = false;
        r.details.push(AssertDetail::new(
            DetailKind::PageLocality,
            format!(
                "page locality {observed:.4} below threshold {threshold:.4} ({local_pages}/{total_pages} pages local)",
            ),
        ));
    }
    r
}

/// Check cross-node page migration ratio against threshold.
///
/// `migrated_pages` is the delta of `/proc/vmstat` `numa_pages_migrated`
/// between pre- and post-workload snapshots. `total_pages` is the total
/// allocated pages from numa_maps.
pub fn assert_cross_node_migration(
    migrated_pages: u64,
    total_pages: u64,
    max_ratio: Option<f64>,
) -> AssertResult {
    let mut r = AssertResult::pass();
    if let Some(threshold) = max_ratio {
        let ratio = if total_pages > 0 {
            migrated_pages as f64 / total_pages as f64
        } else {
            0.0
        };
        if ratio > threshold {
            r.passed = false;
            r.details.push(AssertDetail::new(
                DetailKind::CrossNodeMigration,
                format!(
                    "cross-node migration ratio {ratio:.4} exceeds threshold {threshold:.4} ({migrated_pages}/{total_pages} pages migrated)",
                ),
            ));
        }
    }
    r
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
/// # use ktstr::assert::Assert;
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
#[must_use = "builder methods return a new Assert; discard means config is lost"]
#[derive(Clone, Copy, Debug)]
pub struct Assert {
    // Worker checks
    /// Enable starvation, fairness spread, and gap checks across
    /// worker reports. `Some(true)` enables, `Some(false)` explicitly
    /// disables (overriding any enabling merge from a lower layer),
    /// `None` inherits from the merge parent.
    pub not_starved: Option<bool>,
    /// Enable per-worker CPU isolation checks (ensure workers remain
    /// within their assigned cpuset). Same tri-state semantics as
    /// `not_starved`.
    pub isolation: Option<bool>,
    /// Max per-worker scheduling gap in milliseconds. Fails the
    /// assertion if any worker's longest off-CPU stretch exceeds this.
    pub max_gap_ms: Option<u64>,
    /// Max per-cgroup fairness spread as a percentage. Fails if the
    /// range between the most- and least-served workers exceeds this
    /// fraction of their mean.
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
    /// Max `nr_running` / LLC imbalance ratio observed by the monitor.
    /// Fails if the worst sample's imbalance exceeds this.
    pub max_imbalance_ratio: Option<f64>,
    /// Max local DSQ depth observed by the monitor. Fails if any
    /// sampled CPU's local DSQ grew beyond this.
    pub max_local_dsq_depth: Option<u32>,
    /// Treat a stall verdict from the monitor as a hard failure. Same
    /// tri-state semantics as `not_starved`.
    pub fail_on_stall: Option<bool>,
    /// Minimum number of consecutive samples that must exceed the
    /// monitor threshold before a verdict is raised. Smooths out
    /// single-sample spikes.
    pub sustained_samples: Option<usize>,
    /// Max `select_cpu_fallback` rate (events/sec). Fails if the
    /// scx event counter delta over the run exceeds this rate.
    pub max_fallback_rate: Option<f64>,
    /// Max `keep_last` rate (events/sec). Fails if the scx event
    /// counter delta over the run exceeds this rate.
    pub max_keep_last_rate: Option<f64>,

    // NUMA checks
    /// Minimum fraction of pages on the expected NUMA node(s) (0.0-1.0).
    /// Expected nodes are derived from the worker's
    /// [`MemPolicy`](crate::workload::MemPolicy) at evaluation time.
    /// Fails if the observed locality fraction falls below this.
    pub min_page_locality: Option<f64>,
    /// Maximum ratio of NUMA-node-migrated pages to total allocated
    /// pages (0.0-1.0). Distinct from [`max_migration_ratio`](Self::max_migration_ratio)
    /// which measures CPU migrations per iteration. Fails if the
    /// observed migration ratio exceeds this.
    pub max_cross_node_migration_ratio: Option<f64>,
    /// Maximum fraction of pages on slow-tier (memory-only) NUMA nodes
    /// (0.0-1.0). For CXL memory tiering tests: fails if more than
    /// this fraction of pages land on memory-only nodes. Requires
    /// `slow_tier_nodes` to be set at evaluation time.
    pub max_slow_tier_ratio: Option<f64>,
}

impl Assert {
    /// Identity element for [`Assert::merge`]: every field is `None`,
    /// so neither side of a merge with `NONE` is altered.
    ///
    /// `NONE` is "no overrides," not "no checks." When used as a
    /// per-test or per-scheduler value (`entry.assert`,
    /// `scheduler.assert`), the runtime merge chain
    /// `default_checks().merge(&scheduler.assert).merge(&entry.assert)`
    /// still leaves [`default_checks`](Self::default_checks) intact,
    /// so `not_starved` and the monitor thresholds keep firing. To
    /// turn a default check off, override it explicitly with the
    /// builder method (e.g. `not_starved = Some(false)` via
    /// struct-update syntax) rather than reaching for `NONE`.
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
        min_page_locality: None,
        max_cross_node_migration_ratio: None,
        max_slow_tier_ratio: None,
    };

    /// Baseline of the runtime merge chain
    /// `default_checks().merge(&scheduler.assert).merge(&entry.assert)`:
    /// `not_starved` enabled and monitor thresholds populated from
    /// [`MonitorThresholds::DEFAULT`] (imbalance 4.0, dsq_depth 50,
    /// stall on, sustained 5, fallback 200.0/s, keep_last 100.0/s).
    ///
    /// Because [`Assert::NONE`] is the merge identity, scheduler- or
    /// per-test asserts that leave a default field as `None` inherit
    /// it. To suppress a default check, override the field explicitly
    /// (e.g. `not_starved: Some(false)`), not by switching to `NONE`.
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
            min_page_locality: None,
            max_cross_node_migration_ratio: None,
            max_slow_tier_ratio: None,
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

    /// Control whether a monitor stall verdict fails the assertion.
    pub const fn fail_on_stall(mut self, v: bool) -> Self {
        self.fail_on_stall = Some(v);
        self
    }

    /// Set the number of consecutive over-threshold samples required
    /// before the monitor raises a verdict.
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

    pub const fn min_page_locality(mut self, v: f64) -> Self {
        self.min_page_locality = Some(v);
        self
    }

    pub const fn max_cross_node_migration_ratio(mut self, v: f64) -> Self {
        self.max_cross_node_migration_ratio = Some(v);
        self
    }

    pub const fn max_slow_tier_ratio(mut self, v: f64) -> Self {
        self.max_slow_tier_ratio = Some(v);
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
            || self.min_page_locality.is_some()
            || self.max_cross_node_migration_ratio.is_some()
            || self.max_slow_tier_ratio.is_some()
    }

    /// Merge `other` on top of `self`. Each `Some` field in `other`
    /// overrides the corresponding field in `self`; `None` fields
    /// inherit from `self`.
    ///
    /// [`Assert::NONE`] is the two-sided identity: `x.merge(&NONE)`
    /// and `NONE.merge(&x)` both yield `x`. The runtime composes
    /// scheduler- and test-level overrides as
    /// `Assert::default_checks().merge(&scheduler.assert).merge(&test.assert)`,
    /// so a `NONE` at either override layer leaves the defaults
    /// untouched -- which means "no override," not "no checks."
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
            min_page_locality: match other.min_page_locality {
                Some(v) => Some(v),
                None => self.min_page_locality,
            },
            max_cross_node_migration_ratio: match other.max_cross_node_migration_ratio {
                Some(v) => Some(v),
                None => self.max_cross_node_migration_ratio,
            },
            max_slow_tier_ratio: match other.max_slow_tier_ratio {
                Some(v) => Some(v),
                None => self.max_slow_tier_ratio,
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
            min_page_locality: self.min_page_locality,
            max_cross_node_migration_ratio: self.max_cross_node_migration_ratio,
            max_slow_tier_ratio: self.max_slow_tier_ratio,
        }
    }

    /// Run the configured worker checks against one cgroup's reports.
    ///
    /// `cpuset` is the CPU set for isolation checks. `numa_nodes` is
    /// the NUMA node IDs covered by the cpuset (for page locality and
    /// slow-tier checks). Derive via
    /// [`TestTopology::numa_nodes_for_cpuset`](crate::topology::TestTopology::numa_nodes_for_cpuset).
    pub fn assert_cgroup(
        &self,
        reports: &[crate::workload::WorkerReport],
        cpuset: Option<&BTreeSet<usize>>,
    ) -> AssertResult {
        self.worker_plan().assert_cgroup(reports, cpuset, None)
    }

    /// Run worker checks with explicit NUMA node set for page locality.
    pub fn assert_cgroup_with_numa(
        &self,
        reports: &[crate::workload::WorkerReport],
        cpuset: Option<&BTreeSet<usize>>,
        numa_nodes: Option<&BTreeSet<usize>>,
    ) -> AssertResult {
        self.worker_plan()
            .assert_cgroup(reports, cpuset, numa_nodes)
    }

    /// Run NUMA page locality check.
    ///
    /// `observed` is the fraction of pages on expected nodes (0.0-1.0).
    /// `total_pages` and `local_pages` are for diagnostics.
    pub fn assert_page_locality(
        &self,
        observed: f64,
        total_pages: u64,
        local_pages: u64,
    ) -> AssertResult {
        assert_page_locality(observed, self.min_page_locality, total_pages, local_pages)
    }

    /// Run cross-node migration ratio check.
    ///
    /// `migrated_pages` is the `/proc/vmstat` `numa_pages_migrated` delta.
    /// `total_pages` is total allocated pages from numa_maps.
    pub fn assert_cross_node_migration(
        &self,
        migrated_pages: u64,
        total_pages: u64,
    ) -> AssertResult {
        assert_cross_node_migration(
            migrated_pages,
            total_pages,
            self.max_cross_node_migration_ratio,
        )
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
/// # use ktstr::assert::assert_isolation;
/// # use ktstr::workload::WorkerReport;
/// # use std::collections::BTreeSet;
/// # let report = WorkerReport {
/// #     tid: 1, cpus_used: [0, 1].into_iter().collect(),
/// #     work_units: 100, cpu_time_ns: 1_000_000, wall_time_ns: 2_000_000,
/// #     off_cpu_ns: 1_000_000, migration_count: 0, migrations: vec![],
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
            r.details.push(AssertDetail::new(
                DetailKind::Isolation,
                format!("tid {} ran on unexpected CPUs {:?}", w.tid, bad),
            ));
        }
    }
    r
}

/// Check one cgroup's workers. Returns per-cgroup stats.
///
/// ```
/// # use ktstr::assert::assert_not_starved;
/// # use ktstr::workload::WorkerReport;
/// # let report = WorkerReport {
/// #     tid: 1, cpus_used: [0].into_iter().collect(),
/// #     work_units: 100, cpu_time_ns: 1_000_000, wall_time_ns: 5_000_000_000,
/// #     off_cpu_ns: 500_000_000, migration_count: 0, migrations: vec![],
/// #     max_gap_ms: 50, max_gap_cpu: 0, max_gap_at_ms: 1000,
/// #     wake_latencies_ns: vec![], iterations: 0,
/// #     schedstat_run_delay_ns: 0, schedstat_ctx_switches: 0,
/// #     schedstat_cpu_time_ns: 0,
/// # };
/// let r = assert_not_starved(&[report]);
/// assert!(r.passed);
/// assert_eq!(r.stats.total_workers, 1);
/// ```
/// Nearest-rank percentile of a sorted slice (`p` in `[0.0, 1.0]`).
///
/// Returns the value at index `ceil(n * p) - 1`, clamped into
/// `[0, n-1]`. For `n = 100` and `p = 0.99` this is `sorted[98]` (the
/// 99th element in 1-indexed order), not `sorted[99]` (the max). The
/// previous formulation, `ceil(n * 0.99)` without the `-1`, was
/// off-by-one and returned the max for `n = 100`.
///
/// Callers must pass a sorted non-empty slice; an empty slice yields
/// `0` (the caller should short-circuit before invoking).
fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let n = sorted.len();
    let idx = ((n as f64 * p).ceil() as usize)
        .saturating_sub(1)
        .min(n - 1);
    sorted[idx]
}

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
            r.details.push(AssertDetail::new(
                DetailKind::Starved,
                format!("tid {} starved (0 work units)", w.tid),
            ));
        }
        if w.wall_time_ns > 0 {
            pcts.push(w.off_cpu_ns as f64 / w.wall_time_ns as f64 * 100.0);
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
        let p99 = percentile(&sorted, 0.99) as f64 / 1000.0;
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
        avg_off_cpu_pct: avg,
        min_off_cpu_pct: min,
        max_off_cpu_pct: max,
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
        page_locality: 0.0,
        cross_node_migration_ratio: 0.0,
        ext_metrics: BTreeMap::new(),
    };

    // Per-cgroup fairness: spread above threshold means unequal scheduling within a cgroup
    let spread_limit = spread_threshold_pct();
    if spread > spread_limit && pcts.len() >= 2 {
        r.passed = false;
        r.details.push(AssertDetail::new(
            DetailKind::Unfair,
            format!(
                "unfair cgroup: spread={:.0}% ({:.0}-{:.0}%) {} workers on {} cpus",
                spread,
                min,
                max,
                reports.len(),
                cpus.len(),
            ),
        ));
    }

    // Scheduling gap: >threshold = dispatch failure
    let gap_limit = gap_threshold_ms();
    for w in reports {
        if w.max_gap_ms > gap_limit {
            r.passed = false;
            r.details.push(AssertDetail::new(
                DetailKind::Stuck,
                format!(
                    "stuck {}ms on cpu{} at +{}ms",
                    w.max_gap_ms, w.max_gap_cpu, w.max_gap_at_ms
                ),
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
        worst_page_locality: 0.0,
        worst_cross_node_migration_ratio: 0.0,
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
/// # use ktstr::assert::assert_throughput_parity;
/// # use ktstr::workload::WorkerReport;
/// # let mk = |units, cpu_ns| WorkerReport {
/// #     tid: 1, cpus_used: [0].into_iter().collect(),
/// #     work_units: units, cpu_time_ns: cpu_ns, wall_time_ns: cpu_ns,
/// #     off_cpu_ns: cpu_ns, migration_count: 0, migrations: vec![],
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
            r.details.push(AssertDetail::new(
                DetailKind::Benchmark,
                format!(
                    "throughput CV {cv:.3} exceeds limit {cv_limit:.3} (mean={mean:.0} work/cpu_s)"
                ),
            ));
        }
    }

    if let Some(floor) = min_rate {
        for (i, &rate) in rates.iter().enumerate() {
            if rate < floor {
                r.passed = false;
                r.details.push(AssertDetail::new(
                    DetailKind::Benchmark,
                    format!(
                        "worker {} throughput {rate:.0} work/cpu_s below floor {floor:.0}",
                        reports[i].tid
                    ),
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
/// # use ktstr::assert::assert_benchmarks;
/// # use ktstr::workload::WorkerReport;
/// # let report = WorkerReport {
/// #     tid: 1, cpus_used: [0].into_iter().collect(),
/// #     work_units: 1000, cpu_time_ns: 2_500_000_000,
/// #     wall_time_ns: 5_000_000_000, off_cpu_ns: 2_500_000_000,
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
        let p99 = percentile(&sorted, 0.99);
        if p99 > p99_limit {
            r.passed = false;
            r.details.push(AssertDetail::new(
                DetailKind::Benchmark,
                format!(
                    "p99 wake latency {p99}ns exceeds limit {p99_limit}ns ({} samples)",
                    sorted.len()
                ),
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
                r.details.push(AssertDetail::new(
                    DetailKind::Benchmark,
                    format!(
                        "wake latency CV {cv:.3} exceeds limit {cv_limit:.3} (mean={mean:.0}ns)"
                    ),
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
                r.details.push(AssertDetail::new(
                    DetailKind::Benchmark,
                    format!(
                        "worker {} iteration rate {rate:.1}/s below floor {rate_floor:.1}/s",
                        w.tid
                    ),
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
        off_cpu_ns: u64,
        cpus: &[usize],
        gap_ms: u64,
    ) -> WorkerReport {
        WorkerReport {
            tid,
            work_units: work,
            cpu_time_ns: wall_ns.saturating_sub(off_cpu_ns),
            wall_time_ns: wall_ns,
            off_cpu_ns,
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
            numa_pages: BTreeMap::new(),
            vmstat_numa_pages_migrated: 0,
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
        assert!(m.passed, "diff cgroups diff off_cpu should pass");
    }

    #[test]
    fn spread_boundary() {
        let threshold = spread_threshold_pct();
        // At threshold exactly - pass
        // Worker 1: 10% off-CPU, Worker 2: 10%+threshold off-CPU
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
        assert!((c.min_off_cpu_pct - 20.0).abs() < 0.1);
        assert!((c.max_off_cpu_pct - 30.0).abs() < 0.1);
        assert!((c.spread - 10.0).abs() < 0.1);
        assert!((c.avg_off_cpu_pct - 25.0).abs() < 0.1);
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
    fn is_skipped_true_for_skip_result() {
        // Regression for #36: skip results must be distinguishable
        // from pass results so stats tooling can subtract them from
        // pass counts (a skipped test is not a successful execution).
        let r = AssertResult::skip("no LLC available");
        assert!(r.passed, "skip keeps passed=true for simple gate");
        assert!(r.is_skipped(), "skip must report is_skipped");
    }

    #[test]
    fn is_skipped_false_for_pass_result() {
        let r = AssertResult::pass();
        assert!(r.passed);
        assert!(!r.is_skipped(), "pass is not a skip");
    }

    #[test]
    fn is_skipped_false_for_fail_result() {
        let mut r = AssertResult::pass();
        r.passed = false;
        r.details
            .push(AssertDetail::new(DetailKind::Starved, "worker starved"));
        assert!(
            !r.is_skipped(),
            "fail is not a skip even with non-skip details"
        );
    }

    #[test]
    fn merge_skip_plus_pass_demotes_skip() {
        let mut a = AssertResult::skip("optional");
        let b = AssertResult::pass();
        a.merge(b);
        assert!(!a.skipped);
        assert!(a.passed);
    }

    #[test]
    fn merge_skip_plus_fail_is_fail_not_skip() {
        let mut a = AssertResult::skip("topo missing");
        let mut b = AssertResult::pass();
        b.passed = false;
        a.merge(b);
        assert!(!a.passed);
        assert!(!a.skipped);
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
    fn merge_ext_metrics_higher_is_worse_takes_max() {
        // "worst_spread" is registered with higher_is_worse=true → merge max.
        let mut a = AssertResult::pass();
        a.stats.ext_metrics.insert("worst_spread".into(), 10.0);
        let mut b = AssertResult::pass();
        b.stats.ext_metrics.insert("worst_spread".into(), 42.0);
        a.merge(b);
        assert_eq!(a.stats.ext_metrics["worst_spread"], 42.0);
    }

    #[test]
    fn merge_ext_metrics_higher_is_better_takes_min() {
        // Regression: "total_iterations" is registered with
        // higher_is_worse=false. Merge must take min (worst case)
        // rather than max (best case). Previously returned 42.0.
        let mut a = AssertResult::pass();
        a.stats.ext_metrics.insert("total_iterations".into(), 10.0);
        let mut b = AssertResult::pass();
        b.stats.ext_metrics.insert("total_iterations".into(), 42.0);
        a.merge(b);
        assert_eq!(
            a.stats.ext_metrics["total_iterations"], 10.0,
            "higher_is_worse=false must take min on merge"
        );
    }

    #[test]
    fn merge_ext_metrics_unknown_metric_defaults_to_max() {
        // Unregistered metric names fall back to max (conservative —
        // treat as higher-is-worse until a MetricDef is registered).
        let mut a = AssertResult::pass();
        a.stats.ext_metrics.insert("unknown_metric".into(), 10.0);
        let mut b = AssertResult::pass();
        b.stats.ext_metrics.insert("unknown_metric".into(), 42.0);
        a.merge(b);
        assert_eq!(a.stats.ext_metrics["unknown_metric"], 42.0);
    }

    #[test]
    fn merge_ext_metrics_first_insert_uses_other_value() {
        // When the key is absent on self, insert other's value verbatim
        // regardless of polarity (no prior value to compare against).
        let mut a = AssertResult::pass();
        let mut b = AssertResult::pass();
        b.stats.ext_metrics.insert("total_iterations".into(), 77.0);
        a.merge(b);
        assert_eq!(a.stats.ext_metrics["total_iterations"], 77.0);
    }

    // -- percentile: nearest-rank without off-by-one --

    #[test]
    fn percentile_empty_slice_is_zero() {
        assert_eq!(percentile(&[], 0.99), 0);
    }

    #[test]
    fn percentile_single_element() {
        assert_eq!(percentile(&[42], 0.99), 42);
    }

    #[test]
    fn percentile_p99_of_100_samples_is_element_98() {
        // Regression: previous formulation `ceil(n * 0.99)` returned
        // index 99 (the max) for n=100. The correct nearest-rank p99
        // of [0, 1, 2, ..., 99] is 98 — the 99th element 1-indexed.
        let sorted: Vec<u64> = (0..100).collect();
        assert_eq!(percentile(&sorted, 0.99), 98);
    }

    #[test]
    fn percentile_p99_of_1000_samples_is_element_989() {
        let sorted: Vec<u64> = (0..1000).collect();
        assert_eq!(percentile(&sorted, 0.99), 989);
    }

    #[test]
    fn percentile_saturates_into_bounds_for_small_n() {
        // For very small n, ceil(n * 0.99) may equal n, so the helper
        // must saturating_sub(1) and clamp to n-1 to stay in bounds.
        for n in 1u64..=10 {
            let sorted: Vec<u64> = (0..n).collect();
            let v = percentile(&sorted, 0.99);
            assert!(v < n, "percentile({sorted:?}, 0.99)={v} must be < n ({n})");
        }
    }

    #[test]
    fn percentile_p50_on_odd_count_is_middle() {
        // p50 of [0..9] at nearest-rank: ceil(9 * 0.5) - 1 = 4.
        let sorted: Vec<u64> = (0..9).collect();
        assert_eq!(percentile(&sorted, 0.50), 4);
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
                avg_off_cpu_pct: 50.0,
                min_off_cpu_pct: 40.0,
                max_off_cpu_pct: 60.0,
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
            skipped: false,
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
        let r = plan.assert_cgroup(&reports, None, None);
        assert!(r.passed);
        assert_eq!(r.stats.total_workers, 1);
    }

    #[test]
    fn plan_check_isolation_with_cpuset() {
        let plan = AssertPlan::new().check_not_starved().check_isolation();
        let expected: BTreeSet<usize> = [0, 1].into_iter().collect();
        let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1, 4], 50)];
        let r = plan.assert_cgroup(&reports, Some(&expected), None);
        assert!(!r.passed);
        assert!(r.details.iter().any(|d| d.contains("unexpected")));
    }

    #[test]
    fn plan_isolation_skipped_without_cpuset() {
        let plan = AssertPlan::new().check_isolation();
        let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1, 4], 50)];
        // No cpuset provided -- isolation check is skipped.
        let r = plan.assert_cgroup(&reports, None, None);
        assert!(r.passed);
    }

    #[test]
    fn plan_custom_gap_threshold_pass() {
        let plan = AssertPlan::new().check_not_starved().max_gap_ms(3000);
        // 2500ms gap: passes with 3000ms threshold.
        let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 2500)];
        let r = plan.assert_cgroup(&reports, None, None);
        assert!(r.passed, "2500ms < 3000ms threshold: {:?}", r.details);
    }

    #[test]
    fn plan_custom_gap_threshold_fail() {
        let plan = AssertPlan::new().check_not_starved().max_gap_ms(1500);
        // 2000ms gap: fails with 1500ms threshold.
        let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 2000)];
        let r = plan.assert_cgroup(&reports, None, None);
        assert!(!r.passed);
        assert!(r.details.iter().any(|d| d.contains("stuck")));
        assert!(r.details.iter().any(|d| d.contains("threshold 1500ms")));
    }

    #[test]
    fn plan_custom_gap_threshold_produces_stuck_kind() {
        // Regression for #22: AssertPlan's custom-threshold stuck
        // re-emission must tag DetailKind::Stuck so downstream kind
        // filters (and any test expecting structural categorization)
        // see it.
        let plan = AssertPlan::new().check_not_starved().max_gap_ms(1500);
        let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 2000)];
        let r = plan.assert_cgroup(&reports, None, None);
        assert!(!r.passed);
        assert!(
            r.details.iter().any(|d| d.kind == DetailKind::Stuck),
            "custom gap override must produce a Stuck-kind detail: {:?}",
            r.details
        );
    }

    #[test]
    fn plan_permissive_overrides_clear_unfair_and_stuck_preserve_starved() {
        // Regression for #22: when custom spread + gap thresholds are
        // permissive enough to absorb the default-threshold failures,
        // AssertPlan must strip the Unfair/Stuck details it generated
        // but keep the Starved detail (kind-based filtering, not
        // substring match).
        //
        // Worker 1: 10% off-CPU, 500ms gap — fair, not stuck.
        // Worker 2: work=0 — starved (kind=Starved).
        // Worker 3: 80% off-CPU — would trigger default Unfair; absorbed
        //                         by permissive max_spread_pct.
        // Worker 4: 4000ms gap — would trigger default Stuck; absorbed
        //                        by permissive max_gap_ms.
        let reports = [
            rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 500),
            rpt(2, 0, 5e9 as u64, 0, &[0], 500),
            rpt(3, 500, 5e9 as u64, 4e9 as u64, &[0], 500),
            rpt(4, 1000, 5e9 as u64, 5e8 as u64, &[0], 4000),
        ];
        let mut plan = AssertPlan::new();
        plan.not_starved = true;
        plan.max_spread_pct = Some(100.0);
        plan.max_gap_ms = Some(5000);
        let r = plan.assert_cgroup(&reports, None, None);
        assert!(
            r.details.iter().any(|d| d.kind == DetailKind::Starved),
            "starved detail must survive permissive overrides: {:?}",
            r.details
        );
        assert!(
            !r.details.iter().any(|d| d.kind == DetailKind::Unfair),
            "unfair detail must be cleared by permissive spread: {:?}",
            r.details
        );
        assert!(
            !r.details.iter().any(|d| d.kind == DetailKind::Stuck),
            "stuck detail must be cleared by permissive gap: {:?}",
            r.details
        );
        assert!(!r.passed, "starved alone is still a failure");
    }

    #[test]
    fn plan_no_checks_always_passes() {
        let plan = AssertPlan::new();
        let reports = [rpt(1, 0, 0, 0, &[], 5000)]; // starved + stuck
        let r = plan.assert_cgroup(&reports, None, None);
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
        let r = plan.assert_cgroup(&reports, None, None);
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
        let rd = d.assert_cgroup(&reports, None, None);
        let rn = n.assert_cgroup(&reports, None, None);
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
        // The off_cpu_pct computation skips this worker (no pcts entry).
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
        let r = plan.assert_cgroup(&reports, None, None);
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
            c.min_off_cpu_pct < 10.0,
            "min pct should be ~5%: {:.1}",
            c.min_off_cpu_pct
        );
        assert!(
            c.max_off_cpu_pct > 90.0,
            "max pct should be ~95%: {:.1}",
            c.max_off_cpu_pct
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
        let r = plan.assert_cgroup(&reports, None, None);
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
        let r = plan.assert_cgroup(&reports, Some(&expected), None);
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

    /// `Assert::NONE` is the two-sided identity for `merge`. The
    /// right-identity case is covered above; this locks the
    /// left-identity case so a `NONE.merge(&default_checks())` at
    /// either order in the runtime chain produces the same defaults.
    #[test]
    fn assert_merge_none_is_left_identity() {
        let merged = Assert::NONE.merge(&Assert::default_checks());
        let baseline = Assert::default_checks();
        assert_eq!(merged.not_starved, baseline.not_starved);
        assert_eq!(merged.max_imbalance_ratio, baseline.max_imbalance_ratio);
        assert_eq!(merged.max_local_dsq_depth, baseline.max_local_dsq_depth);
        assert_eq!(merged.fail_on_stall, baseline.fail_on_stall);
        assert_eq!(merged.sustained_samples, baseline.sustained_samples);
        assert_eq!(merged.max_fallback_rate, baseline.max_fallback_rate);
        assert_eq!(merged.max_keep_last_rate, baseline.max_keep_last_rate);
        // Fields that default_checks leaves None remain None.
        assert!(merged.max_gap_ms.is_none());
        assert!(merged.isolation.is_none());
    }

    /// The runtime three-layer chain
    /// `default_checks -> scheduler -> test` collapses to
    /// `default_checks` when both override layers are `NONE`. This
    /// proves the documented "NONE means no override, not no checks"
    /// invariant end-to-end.
    #[test]
    fn assert_merge_runtime_chain_with_none_overrides_yields_defaults() {
        let scheduler_assert = Assert::NONE;
        let test_assert = Assert::NONE;
        let merged = Assert::default_checks()
            .merge(&scheduler_assert)
            .merge(&test_assert);
        let baseline = Assert::default_checks();
        assert_eq!(merged.not_starved, baseline.not_starved);
        assert_eq!(merged.max_imbalance_ratio, baseline.max_imbalance_ratio);
        assert_eq!(merged.max_local_dsq_depth, baseline.max_local_dsq_depth);
        assert_eq!(merged.fail_on_stall, baseline.fail_on_stall);
        assert_eq!(merged.sustained_samples, baseline.sustained_samples);
        assert_eq!(merged.max_fallback_rate, baseline.max_fallback_rate);
        assert_eq!(merged.max_keep_last_rate, baseline.max_keep_last_rate);
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
            skipped: false,
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
            skipped: false,
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
        let r = plan.assert_cgroup(&reports, None, None);
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
            off_cpu_ns: wall_ns / 2,
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
            numa_pages: BTreeMap::new(),
            vmstat_numa_pages_migrated: 0,
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
    fn assert_benchmarks_p99_n100_at_limit_passes() {
        // Regression for #23: with samples [0..100], the corrected p99
        // is 98 (nearest-rank: sorted[ceil(100*0.99) - 1] = sorted[98]).
        // Setting the limit to 99 must pass (98 <= 99). Under the old
        // off-by-one formulation, the returned p99 was 99 (the max),
        // which would have been exactly at the limit (99 <= 99) — the
        // same pass, but for the wrong reason. Pairing this with the
        // _fail test below pins down the correct index.
        let latencies: Vec<u64> = (0..100).collect();
        let reports = [rpt_with_latencies(1, latencies, 100, 5_000_000_000)];
        let r = assert_benchmarks(&reports, Some(99), None, None);
        assert!(
            r.passed,
            "p99 should be 98, under limit 99: {:?}",
            r.details
        );
    }

    #[test]
    fn assert_benchmarks_p99_n100_below_old_p100_passes() {
        // Tighter regression: with samples [0..100], set the limit to
        // 98. Correct p99 (98) equals the limit and passes (strict
        // `p99 > p99_limit` comparison). The old off-by-one returned
        // 99, which would have FAILED (99 > 98). This test therefore
        // only passes with the corrected index.
        let latencies: Vec<u64> = (0..100).collect();
        let reports = [rpt_with_latencies(1, latencies, 100, 5_000_000_000)];
        let r = assert_benchmarks(&reports, Some(98), None, None);
        assert!(
            r.passed,
            "corrected p99 (98) must equal limit 98 and pass: {:?}",
            r.details
        );
    }

    #[test]
    fn assert_not_starved_p99_n100_is_99_microseconds() {
        // Regression for #23: assert_not_starved exposes p99 as
        // microseconds via ScenarioStats. Samples = [1000, 2000, ...,
        // 100_000] ns (100 values at kilo-ns spacing) so the reported
        // p99 is exactly 99.0us with the corrected index
        // (sorted[ceil(100*0.99) - 1] = sorted[98] = 99_000ns = 99us).
        // The old off-by-one returned sorted[99] = 100_000ns = 100us.
        let latencies: Vec<u64> = (1..=100).map(|v: u64| v * 1000).collect();
        let reports = [rpt_with_latencies(1, latencies, 100, 5_000_000_000)];
        let r = assert_not_starved(&reports);
        assert_eq!(
            r.stats.p99_wake_latency_us, 99.0,
            "p99 must equal 99.0us (sorted[98] = 99_000ns), got {}us",
            r.stats.p99_wake_latency_us
        );
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
        // sorted, percentile index = ceil(10*0.99) - 1 = 9 -> sorted[9] = 10000ns = 10.0us
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
            min_page_locality: None,
            max_cross_node_migration_ratio: None,
            max_slow_tier_ratio: None,
        };
        let reports = [rpt_with_latencies(
            1,
            vec![100, 200, 300, 400, 1000],
            10,
            5_000_000_000,
        )];
        let r = plan.assert_cgroup(&reports, None, None);
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
            min_page_locality: None,
            max_cross_node_migration_ratio: None,
            max_slow_tier_ratio: None,
        };
        let r = plan.assert_cgroup(&[w], None, None);
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
            min_page_locality: None,
            max_cross_node_migration_ratio: None,
            max_slow_tier_ratio: None,
        };
        let r = plan.assert_cgroup(&[w], None, None);
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
            min_page_locality: None,
            max_cross_node_migration_ratio: None,
            max_slow_tier_ratio: None,
        };
        let reports = [rpt_with_latencies(1, vec![], 10, 5_000_000_000)];
        let r = plan.assert_cgroup(&reports, None, None);
        assert!(!r.passed, "2/s < 1000/s floor");
        assert!(r.details.iter().any(|d| d.contains("iteration rate")));
    }

    // -- AssertResult::skip --

    #[test]
    fn assert_result_skip_is_pass_with_reason() {
        let r = AssertResult::skip("topology too small");
        assert!(r.passed);
        assert_eq!(r.details.len(), 1);
        assert_eq!(r.details[0], "topology too small");
    }

    #[test]
    fn assert_result_skip_default_stats() {
        let r = AssertResult::skip("skipped");
        assert_eq!(r.stats.total_workers, 0);
        assert!(r.stats.cgroups.is_empty());
    }

    // -- Assert::has_worker_checks --

    #[test]
    fn assert_none_has_no_worker_checks() {
        assert!(!Assert::NONE.has_worker_checks());
    }

    #[test]
    fn assert_default_checks_has_worker_checks() {
        assert!(Assert::default_checks().has_worker_checks());
    }

    #[test]
    fn assert_single_field_has_worker_checks() {
        assert!(Assert::NONE.max_gap_ms(5000).has_worker_checks());
        assert!(Assert::NONE.check_isolation().has_worker_checks());
        assert!(Assert::NONE.max_spread_pct(10.0).has_worker_checks());
        assert!(Assert::NONE.max_throughput_cv(0.5).has_worker_checks());
        assert!(Assert::NONE.min_work_rate(100.0).has_worker_checks());
        assert!(
            Assert::NONE
                .max_p99_wake_latency_ns(1000)
                .has_worker_checks()
        );
        assert!(Assert::NONE.max_wake_latency_cv(0.5).has_worker_checks());
        assert!(Assert::NONE.min_iteration_rate(10.0).has_worker_checks());
        assert!(Assert::NONE.max_migration_ratio(0.5).has_worker_checks());
    }

    #[test]
    fn assert_monitor_only_no_worker_checks() {
        let a = Assert::NONE.max_imbalance_ratio(5.0).fail_on_stall(true);
        assert!(!a.has_worker_checks());
    }

    // -- AssertResult::merge ext_metrics --

    #[test]
    fn assert_result_merge_ext_metrics_max_value() {
        let mut a = AssertResult::pass();
        a.stats.ext_metrics.insert("latency".into(), 10.0);
        a.stats.ext_metrics.insert("throughput".into(), 100.0);

        let mut b = AssertResult::pass();
        b.stats.ext_metrics.insert("latency".into(), 20.0);
        b.stats.ext_metrics.insert("jitter".into(), 5.0);

        a.merge(b);
        assert_eq!(a.stats.ext_metrics["latency"], 20.0);
        assert_eq!(a.stats.ext_metrics["throughput"], 100.0);
        assert_eq!(a.stats.ext_metrics["jitter"], 5.0);
    }

    #[test]
    fn assert_result_merge_ext_metrics_keeps_larger() {
        let mut a = AssertResult::pass();
        a.stats.ext_metrics.insert("x".into(), 50.0);

        let mut b = AssertResult::pass();
        b.stats.ext_metrics.insert("x".into(), 30.0);

        a.merge(b);
        assert_eq!(a.stats.ext_metrics["x"], 50.0);
    }

    // -- Assert::merge worker + benchmark + monitor fields --

    #[test]
    fn assert_merge_all_field_categories() {
        // Layer 1: defaults (worker + monitor fields).
        let defaults = Assert::default_checks();

        // Layer 2: scheduler sets worker and benchmark fields.
        let sched = Assert::NONE
            .max_spread_pct(50.0)
            .max_p99_wake_latency_ns(100_000)
            .max_migration_ratio(0.5);

        // Layer 3: test overrides a worker field and sets isolation.
        let test = Assert::NONE.check_isolation().max_spread_pct(80.0);

        let merged = defaults.merge(&sched).merge(&test);

        // test overrides sched's spread.
        assert_eq!(merged.max_spread_pct, Some(80.0));
        // sched's benchmark fields survive (test didn't set them).
        assert_eq!(merged.max_p99_wake_latency_ns, Some(100_000));
        assert_eq!(merged.max_migration_ratio, Some(0.5));
        // test sets isolation.
        assert_eq!(merged.isolation, Some(true));
        // defaults: monitor fields survive all layers.
        assert_eq!(merged.fail_on_stall, Some(true));
    }

    // -- numa_maps parsing tests --

    #[test]
    fn parse_numa_maps_basic() {
        let content = "\
00400000 default file=/bin/cat mapped=10 N0=8 N1=2
00600000 default anon=5 N0=3 N1=2";
        let entries = parse_numa_maps(content);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].addr, 0x00400000);
        assert_eq!(entries[0].node_pages[&0], 8);
        assert_eq!(entries[0].node_pages[&1], 2);
        assert_eq!(entries[1].addr, 0x00600000);
        assert_eq!(entries[1].node_pages[&0], 3);
        assert_eq!(entries[1].node_pages[&1], 2);
    }

    #[test]
    fn parse_numa_maps_empty() {
        assert!(parse_numa_maps("").is_empty());
    }

    #[test]
    fn parse_numa_maps_no_node_fields() {
        let content = "00400000 default file=/bin/cat mapped=10";
        let entries = parse_numa_maps(content);
        assert!(entries.is_empty());
    }

    #[test]
    fn parse_numa_maps_single_node() {
        let content = "7f000000 default anon=100 N0=100";
        let entries = parse_numa_maps(content);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].node_pages[&0], 100);
        assert_eq!(entries[0].node_pages.len(), 1);
    }

    #[test]
    fn parse_numa_maps_high_node_ids() {
        let content = "7f000000 default N0=10 N3=20 N7=5";
        let entries = parse_numa_maps(content);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].node_pages[&0], 10);
        assert_eq!(entries[0].node_pages[&3], 20);
        assert_eq!(entries[0].node_pages[&7], 5);
    }

    #[test]
    fn parse_numa_maps_malformed_lines() {
        let content = "\
not_hex default N0=10
00400000 default N0=10
 default N0=5";
        let entries = parse_numa_maps(content);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].addr, 0x00400000);
    }

    // -- page_locality tests --

    #[test]
    fn page_locality_all_local() {
        let entries = vec![NumaMapsEntry {
            addr: 0x1000,
            node_pages: [(0, 100)].into_iter().collect(),
        }];
        let expected: BTreeSet<usize> = [0].into_iter().collect();
        let loc = page_locality(&entries, &expected);
        assert!((loc - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn page_locality_mixed_nodes() {
        let entries = vec![NumaMapsEntry {
            addr: 0x1000,
            node_pages: [(0, 80), (1, 20)].into_iter().collect(),
        }];
        let expected: BTreeSet<usize> = [0].into_iter().collect();
        let loc = page_locality(&entries, &expected);
        assert!((loc - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn page_locality_multi_expected_nodes() {
        let entries = vec![NumaMapsEntry {
            addr: 0x1000,
            node_pages: [(0, 40), (1, 40), (2, 20)].into_iter().collect(),
        }];
        let expected: BTreeSet<usize> = [0, 1].into_iter().collect();
        let loc = page_locality(&entries, &expected);
        assert!((loc - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn page_locality_empty_entries() {
        let expected: BTreeSet<usize> = [0].into_iter().collect();
        let loc = page_locality(&[], &expected);
        assert!((loc - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn page_locality_no_local_pages() {
        let entries = vec![NumaMapsEntry {
            addr: 0x1000,
            node_pages: [(1, 50)].into_iter().collect(),
        }];
        let expected: BTreeSet<usize> = [0].into_iter().collect();
        let loc = page_locality(&entries, &expected);
        assert!((loc - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn page_locality_empty_expected_set() {
        let entries = vec![NumaMapsEntry {
            addr: 0x1000,
            node_pages: [(0, 50)].into_iter().collect(),
        }];
        let loc = page_locality(&entries, &BTreeSet::new());
        assert!((loc - 0.0).abs() < f64::EPSILON);
    }

    // -- assert_page_locality tests --

    #[test]
    fn assert_page_locality_pass() {
        let r = assert_page_locality(0.9, Some(0.8), 100, 90);
        assert!(r.passed, "{:?}", r.details);
    }

    #[test]
    fn assert_page_locality_fail() {
        let r = assert_page_locality(0.5, Some(0.8), 100, 50);
        assert!(!r.passed);
        assert!(r.details.iter().any(|d| d.contains("page locality")));
    }

    #[test]
    fn assert_page_locality_no_threshold() {
        let r = assert_page_locality(0.1, None, 100, 10);
        assert!(r.passed);
    }

    #[test]
    fn assert_page_locality_exact_threshold() {
        let r = assert_page_locality(0.8, Some(0.8), 100, 80);
        assert!(r.passed, "{:?}", r.details);
    }

    // -- assert_slow_tier_ratio tests --

    #[test]
    fn assert_slow_tier_ratio_pass() {
        let mut pages = BTreeMap::new();
        pages.insert(0, 90);
        pages.insert(1, 10);
        let nodes: BTreeSet<usize> = [0, 1].into_iter().collect();
        let r = assert_slow_tier_ratio(&pages, 0.5, 100, Some(&nodes));
        assert!(r.passed, "{:?}", r.details);
    }

    #[test]
    fn assert_slow_tier_ratio_fail() {
        let mut pages = BTreeMap::new();
        pages.insert(0, 40);
        pages.insert(2, 60);
        let nodes: BTreeSet<usize> = [0].into_iter().collect();
        let r = assert_slow_tier_ratio(&pages, 0.5, 100, Some(&nodes));
        assert!(!r.passed);
        assert!(r.details.iter().any(|d| d.contains("slow-tier")));
    }

    #[test]
    fn assert_slow_tier_ratio_none_numa_nodes() {
        let mut pages = BTreeMap::new();
        pages.insert(0, 100);
        let r = assert_slow_tier_ratio(&pages, 0.1, 100, None);
        assert!(r.passed);
    }

    #[test]
    fn assert_slow_tier_ratio_zero_pages() {
        let pages = BTreeMap::new();
        let nodes: BTreeSet<usize> = [0].into_iter().collect();
        let r = assert_slow_tier_ratio(&pages, 0.5, 0, Some(&nodes));
        assert!(r.passed);
    }

    #[test]
    fn assert_slow_tier_ratio_all_local() {
        let mut pages = BTreeMap::new();
        pages.insert(0, 100);
        let nodes: BTreeSet<usize> = [0].into_iter().collect();
        let r = assert_slow_tier_ratio(&pages, 0.0, 100, Some(&nodes));
        assert!(r.passed, "{:?}", r.details);
    }

    // -- Assert NUMA builder and merge tests --

    #[test]
    fn assert_min_page_locality_setter() {
        let v = Assert::NONE.min_page_locality(0.9);
        assert_eq!(v.min_page_locality, Some(0.9));
    }

    #[test]
    fn assert_merge_numa_fields() {
        let base = Assert::NONE.min_page_locality(0.9);
        let merged = base.merge(&Assert::NONE);
        assert_eq!(merged.min_page_locality, Some(0.9));
    }

    #[test]
    fn assert_merge_numa_override() {
        let base = Assert::NONE.min_page_locality(0.9);
        let other = Assert::NONE.min_page_locality(0.5);
        assert_eq!(base.merge(&other).min_page_locality, Some(0.5));
    }

    #[test]
    fn assert_numa_has_worker_checks() {
        assert!(Assert::NONE.min_page_locality(0.8).has_worker_checks());
    }

    #[test]
    fn assert_page_locality_method_pass() {
        let a = Assert::NONE.min_page_locality(0.8);
        let r = a.assert_page_locality(0.9, 100, 90);
        assert!(r.passed, "{:?}", r.details);
    }

    #[test]
    fn assert_page_locality_method_fail() {
        let a = Assert::NONE.min_page_locality(0.95);
        let r = a.assert_page_locality(0.8, 100, 80);
        assert!(!r.passed);
    }

    // -- ScenarioStats NUMA merge tests --

    #[test]
    fn assert_result_merge_numa_worst_page_locality() {
        let mut a = AssertResult::pass();
        a.stats.worst_page_locality = 0.9;
        let mut b = AssertResult::pass();
        b.stats.worst_page_locality = 0.7;
        a.merge(b);
        assert!((a.stats.worst_page_locality - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn assert_result_merge_numa_zero_locality_ignored() {
        let mut a = AssertResult::pass();
        a.stats.worst_page_locality = 0.9;
        let b = AssertResult::pass();
        a.merge(b);
        assert!((a.stats.worst_page_locality - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn cgroup_stats_numa_defaults() {
        let c = CgroupStats::default();
        assert_eq!(c.page_locality, 0.0);
        assert_eq!(c.cross_node_migration_ratio, 0.0);
    }

    #[test]
    fn scenario_stats_numa_defaults() {
        let s = ScenarioStats::default();
        assert_eq!(s.worst_page_locality, 0.0);
        assert_eq!(s.worst_cross_node_migration_ratio, 0.0);
    }

    // -- parse_vmstat_numa_pages_migrated tests --

    #[test]
    fn parse_vmstat_present() {
        let content = "\
nr_free_pages 12345
numa_hit 100
numa_pages_migrated 42
numa_miss 5";
        assert_eq!(parse_vmstat_numa_pages_migrated(content), Some(42));
    }

    #[test]
    fn parse_vmstat_absent() {
        let content = "nr_free_pages 12345\nnuma_hit 100";
        assert_eq!(parse_vmstat_numa_pages_migrated(content), None);
    }

    #[test]
    fn parse_vmstat_zero() {
        let content = "numa_pages_migrated 0";
        assert_eq!(parse_vmstat_numa_pages_migrated(content), Some(0));
    }

    #[test]
    fn parse_vmstat_large_value() {
        let content = "numa_pages_migrated 9999999999";
        assert_eq!(parse_vmstat_numa_pages_migrated(content), Some(9999999999));
    }

    #[test]
    fn parse_vmstat_empty() {
        assert_eq!(parse_vmstat_numa_pages_migrated(""), None);
    }

    #[test]
    fn parse_vmstat_malformed_value() {
        let content = "numa_pages_migrated abc";
        assert_eq!(parse_vmstat_numa_pages_migrated(content), None);
    }

    // -- assert_cross_node_migration tests --

    #[test]
    fn assert_cross_node_migration_pass() {
        let r = assert_cross_node_migration(5, 100, Some(0.1));
        assert!(r.passed, "{:?}", r.details);
    }

    #[test]
    fn assert_cross_node_migration_fail() {
        let r = assert_cross_node_migration(20, 100, Some(0.1));
        assert!(!r.passed);
        assert!(r.details.iter().any(|d| d.contains("cross-node migration")));
    }

    #[test]
    fn assert_cross_node_migration_no_threshold() {
        let r = assert_cross_node_migration(50, 100, None);
        assert!(r.passed);
    }

    #[test]
    fn assert_cross_node_migration_exact_threshold() {
        let r = assert_cross_node_migration(10, 100, Some(0.1));
        assert!(r.passed, "{:?}", r.details);
    }

    #[test]
    fn assert_cross_node_migration_zero_pages() {
        let r = assert_cross_node_migration(0, 0, Some(0.1));
        assert!(r.passed, "zero total pages should pass");
    }

    // -- Assert cross-node migration builder/merge --

    #[test]
    fn assert_max_cross_node_migration_ratio_setter() {
        let v = Assert::NONE.max_cross_node_migration_ratio(0.05);
        assert_eq!(v.max_cross_node_migration_ratio, Some(0.05));
    }

    #[test]
    fn assert_merge_cross_node_migration() {
        let base = Assert::NONE.max_cross_node_migration_ratio(0.1);
        let other = Assert::NONE.max_cross_node_migration_ratio(0.05);
        assert_eq!(
            base.merge(&other).max_cross_node_migration_ratio,
            Some(0.05)
        );
    }

    #[test]
    fn assert_merge_cross_node_migration_preserves() {
        let base = Assert::NONE.max_cross_node_migration_ratio(0.1);
        assert_eq!(
            base.merge(&Assert::NONE).max_cross_node_migration_ratio,
            Some(0.1)
        );
    }

    #[test]
    fn assert_cross_node_migration_has_worker_checks() {
        assert!(
            Assert::NONE
                .max_cross_node_migration_ratio(0.1)
                .has_worker_checks()
        );
    }

    #[test]
    fn assert_cross_node_migration_method_pass() {
        let a = Assert::NONE.max_cross_node_migration_ratio(0.1);
        let r = a.assert_cross_node_migration(5, 100);
        assert!(r.passed, "{:?}", r.details);
    }

    #[test]
    fn assert_cross_node_migration_method_fail() {
        let a = Assert::NONE.max_cross_node_migration_ratio(0.05);
        let r = a.assert_cross_node_migration(20, 100);
        assert!(!r.passed);
    }

    // -- ScenarioStats cross-node migration merge --

    #[test]
    fn assert_result_merge_worst_cross_node_migration() {
        let mut a = AssertResult::pass();
        a.stats.worst_cross_node_migration_ratio = 0.05;
        let mut b = AssertResult::pass();
        b.stats.worst_cross_node_migration_ratio = 0.15;
        a.merge(b);
        assert!((a.stats.worst_cross_node_migration_ratio - 0.15).abs() < f64::EPSILON);
    }
}
