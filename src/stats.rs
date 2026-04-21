//! Gauntlet analysis and run-to-run comparison.
//!
//! Collects per-scenario results into a [`polars`] DataFrame for
//! statistical analysis, regression detection, and run-to-run compare
//! workflows.

use std::collections::BTreeMap;

use polars::prelude::*;

/// Definition of a metric for the comparison pipeline.
///
/// Each entry describes polarity (`higher_is_worse`), dual-gate
/// significance thresholds (`default_abs`, `default_rel`), a
/// display unit string for formatted output, and a row accessor
/// (`accessor`) that returns the metric's value from a
/// [`GauntletRow`] without a hand-maintained name→field match.
///
/// The `accessor` field is skipped in serde output because `fn`
/// pointers are not serializable. If a `Deserialize` impl is
/// added later, callers must re-hydrate the accessor by looking
/// up `name` via [`metric_def`] — the static `METRICS` table is
/// the authoritative source of the function identity.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MetricDef {
    pub name: &'static str,
    /// Regression direction for this metric. A metric that
    /// previously used `higher_is_worse: true` maps to
    /// [`Polarity::LowerBetter`](crate::test_support::Polarity::LowerBetter)
    /// (bigger values are regressions, so smaller is better);
    /// `false` maps to
    /// [`Polarity::HigherBetter`](crate::test_support::Polarity::HigherBetter).
    /// The sense is INVERSE: the old bool answered "does growing
    /// this value mean worse?" while the enum answers "what
    /// direction do we want this to move?".
    pub polarity: crate::test_support::Polarity,
    pub default_abs: f64,
    pub default_rel: f64,
    pub display_unit: &'static str,
    #[serde(skip)]
    pub accessor: fn(&GauntletRow) -> Option<f64>,
}

impl MetricDef {
    /// Read this metric's value from `row`. Consults the
    /// accessor first (for built-in `GauntletRow` fields) and
    /// falls back to `row.ext_metrics[self.name]` when the
    /// accessor returns `None`.
    pub fn read(&self, row: &GauntletRow) -> Option<f64> {
        (self.accessor)(row).or_else(|| row.ext_metrics.get(self.name).copied())
    }

    /// Returns `true` for [`Polarity::LowerBetter`], `false` for
    /// [`Polarity::HigherBetter`]. [`Polarity::TargetValue`] and
    /// [`Polarity::Unknown`] branches keep the match total; they
    /// are unreachable for the current [`METRICS`] entries (guarded
    /// by the `metric_def_polarity_covers_all_entries` test).
    pub const fn higher_is_worse(&self) -> bool {
        use crate::test_support::Polarity;
        matches!(
            self.polarity,
            Polarity::LowerBetter | Polarity::TargetValue(_) | Polarity::Unknown
        )
    }
}

/// Unified metric registry covering all built-in and extensible metrics.
///
/// The comparison pipeline uses `higher_is_worse` to determine regression
/// direction, `default_abs`/`default_rel` for dual-gate significance
/// thresholds, and `display_unit` for formatted output. Per-test
/// assertion overrides can still use their own thresholds; this registry
/// is the source of truth for polarity and display.
///
/// `AssertResult::merge` consults `higher_is_worse` via [`metric_def`]
/// when folding per-cgroup `ext_metrics` into the scenario-level worst
/// case: `true` takes max, `false` takes min. Unknown names (not in
/// this registry) default to max; register a `MetricDef` here before
/// relying on min-polarity merge. The comparison system
/// ([`compare_runs`]) uses `higher_is_worse` for delta direction.
pub static METRICS: &[MetricDef] = &[
    MetricDef {
        name: "worst_spread",
        polarity: crate::test_support::Polarity::LowerBetter,
        default_abs: 5.0,
        default_rel: 0.25,
        display_unit: "%",
        accessor: |r| Some(r.spread),
    },
    MetricDef {
        name: "worst_gap_ms",
        polarity: crate::test_support::Polarity::LowerBetter,
        default_abs: 500.0,
        default_rel: 0.50,
        display_unit: "ms",
        accessor: |r| Some(r.gap_ms as f64),
    },
    MetricDef {
        name: "total_migrations",
        polarity: crate::test_support::Polarity::LowerBetter,
        default_abs: 10.0,
        default_rel: 0.30,
        display_unit: "",
        accessor: |r| Some(r.migrations as f64),
    },
    MetricDef {
        name: "worst_migration_ratio",
        polarity: crate::test_support::Polarity::LowerBetter,
        default_abs: 0.05,
        default_rel: 0.20,
        display_unit: "",
        accessor: |r| Some(r.migration_ratio),
    },
    MetricDef {
        name: "max_imbalance_ratio",
        polarity: crate::test_support::Polarity::LowerBetter,
        default_abs: 1.0,
        default_rel: 0.25,
        display_unit: "x",
        accessor: |r| Some(r.imbalance_ratio),
    },
    MetricDef {
        name: "max_dsq_depth",
        polarity: crate::test_support::Polarity::LowerBetter,
        default_abs: 10.0,
        default_rel: 0.50,
        display_unit: "",
        accessor: |r| Some(r.max_dsq_depth as f64),
    },
    MetricDef {
        name: "stall_count",
        polarity: crate::test_support::Polarity::LowerBetter,
        default_abs: 1.0,
        default_rel: 0.50,
        display_unit: "",
        accessor: |r| Some(r.stall_count as f64),
    },
    MetricDef {
        name: "total_fallback",
        polarity: crate::test_support::Polarity::LowerBetter,
        default_abs: 5.0,
        default_rel: 0.30,
        display_unit: "/s",
        accessor: |r| Some(r.fallback_count as f64),
    },
    MetricDef {
        name: "total_keep_last",
        polarity: crate::test_support::Polarity::LowerBetter,
        default_abs: 5.0,
        default_rel: 0.30,
        display_unit: "/s",
        accessor: |r| Some(r.keep_last_count as f64),
    },
    MetricDef {
        name: "p99_wake_latency_us",
        polarity: crate::test_support::Polarity::LowerBetter,
        default_abs: 50.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |r| Some(r.p99_wake_latency_us),
    },
    MetricDef {
        name: "median_wake_latency_us",
        polarity: crate::test_support::Polarity::LowerBetter,
        default_abs: 20.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |r| Some(r.median_wake_latency_us),
    },
    MetricDef {
        name: "wake_latency_cv",
        polarity: crate::test_support::Polarity::LowerBetter,
        default_abs: 0.10,
        default_rel: 0.25,
        display_unit: "",
        accessor: |r| Some(r.wake_latency_cv),
    },
    MetricDef {
        name: "total_iterations",
        polarity: crate::test_support::Polarity::HigherBetter,
        default_abs: 100.0,
        default_rel: 0.10,
        display_unit: "",
        accessor: |r| Some(r.total_iterations as f64),
    },
    MetricDef {
        name: "mean_run_delay_us",
        polarity: crate::test_support::Polarity::LowerBetter,
        default_abs: 50.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |r| Some(r.mean_run_delay_us),
    },
    MetricDef {
        name: "worst_run_delay_us",
        polarity: crate::test_support::Polarity::LowerBetter,
        default_abs: 100.0,
        default_rel: 0.50,
        display_unit: "\u{00b5}s",
        accessor: |r| Some(r.worst_run_delay_us),
    },
    MetricDef {
        name: "worst_page_locality",
        polarity: crate::test_support::Polarity::HigherBetter,
        default_abs: 0.05,
        default_rel: 0.10,
        display_unit: "",
        accessor: |r| Some(r.page_locality),
    },
    MetricDef {
        name: "worst_cross_node_migration_ratio",
        polarity: crate::test_support::Polarity::LowerBetter,
        default_abs: 0.05,
        default_rel: 0.20,
        display_unit: "",
        accessor: |r| Some(r.cross_node_migration_ratio),
    },
];

/// Look up a metric definition by name.
pub fn metric_def(name: &str) -> Option<&'static MetricDef> {
    METRICS.iter().find(|m| m.name == name)
}

/// Per-scenario result row for gauntlet analysis and run-to-run comparison.
///
/// Populated by [`sidecar_to_row`] from on-disk `SidecarResult`s. The
/// comparison pipeline reads metric values through [`MetricDef::read`]
/// / [`METRICS`] rather than dereferencing fields directly so new
/// metrics can land through the registry without touching every
/// reader.
///
/// The `worst_degradation_*` and `degradation_count` fields are read by
/// [`build_dataframe`] but are always zero when populated via
/// [`sidecar_to_row`]: sidecars carry the aggregate `MonitorSummary`
/// but not the per-sample trace that would populate the degradation
/// fields.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GauntletRow {
    pub scenario: String,
    pub flags: String,
    pub topology: String,
    pub work_type: String,
    /// Scheduler binary name carried from the source sidecar
    /// (`SidecarResult::scheduler`). Surfaced through the substring
    /// filter in [`compare_rows`] so users can narrow A/B comparisons
    /// by scheduler name.
    pub scheduler: String,
    pub replica: u32,
    pub passed: bool,
    /// True when the run was skipped (topology mismatch, missing
    /// resource). `passed` stays `true` for gate-compat; `skipped`
    /// lets stats tooling exclude these from pass counts so skipped
    /// runs don't inflate the apparent pass rate.
    pub skipped: bool,
    pub spread: f64,
    pub gap_ms: u64,
    pub migrations: u64,
    pub migration_ratio: f64,
    // Monitor fields (host-side telemetry from guest memory reads).
    pub imbalance_ratio: f64,
    pub max_dsq_depth: u32,
    pub stall_count: usize,
    pub fallback_count: i64,
    pub keep_last_count: i64,
    // Benchmarking fields.
    pub p99_wake_latency_us: f64,
    pub median_wake_latency_us: f64,
    pub wake_latency_cv: f64,
    pub total_iterations: u64,
    pub mean_run_delay_us: f64,
    pub worst_run_delay_us: f64,
    // NUMA fields.
    pub page_locality: f64,
    pub cross_node_migration_ratio: f64,
    pub worst_degradation_op: String,
    pub worst_imbalance_delta: f64,
    pub worst_dsq_delta: f64,
    pub worst_fallback_delta: f64,
    pub worst_keep_last_delta: f64,
    pub degradation_count: u32,
    /// Extensible metrics populated by scenarios and processed by the
    /// comparison pipeline. Keyed by metric name; looked up via
    /// [`metric_def`] when a matching entry exists in [`METRICS`].
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub ext_metrics: BTreeMap<String, f64>,
}

/// Convert a SidecarResult to a GauntletRow for run-to-run comparison.
pub fn sidecar_to_row(sc: &crate::test_support::SidecarResult) -> GauntletRow {
    GauntletRow {
        scenario: sc.test_name.clone(),
        flags: String::new(),
        topology: sc.topology.clone(),
        work_type: sc.work_type.clone(),
        scheduler: sc.scheduler.clone(),
        replica: 1,
        passed: sc.passed,
        skipped: sc.skipped,
        spread: sc.stats.worst_spread,
        gap_ms: sc.stats.worst_gap_ms,
        migrations: sc.stats.total_migrations,
        migration_ratio: sc.stats.worst_migration_ratio,
        imbalance_ratio: sc
            .monitor
            .as_ref()
            .map(|m| m.max_imbalance_ratio)
            .unwrap_or(0.0),
        max_dsq_depth: sc
            .monitor
            .as_ref()
            .map(|m| m.max_local_dsq_depth)
            .unwrap_or(0),
        stall_count: if sc.monitor.as_ref().is_some_and(|m| m.stall_detected) {
            1
        } else {
            0
        },
        fallback_count: sc
            .monitor
            .as_ref()
            .and_then(|m| m.event_deltas.as_ref())
            .map(|e| e.total_fallback)
            .unwrap_or(0),
        keep_last_count: sc
            .monitor
            .as_ref()
            .and_then(|m| m.event_deltas.as_ref())
            .map(|e| e.total_dispatch_keep_last)
            .unwrap_or(0),
        p99_wake_latency_us: sc.stats.p99_wake_latency_us,
        median_wake_latency_us: sc.stats.median_wake_latency_us,
        wake_latency_cv: sc.stats.wake_latency_cv,
        total_iterations: sc.stats.total_iterations,
        mean_run_delay_us: sc.stats.mean_run_delay_us,
        worst_run_delay_us: sc.stats.worst_run_delay_us,
        page_locality: sc.stats.worst_page_locality,
        cross_node_migration_ratio: sc.stats.worst_cross_node_migration_ratio,
        worst_degradation_op: String::new(),
        worst_imbalance_delta: 0.0,
        worst_dsq_delta: 0.0,
        worst_fallback_delta: 0.0,
        worst_keep_last_delta: 0.0,
        degradation_count: 0,
        ext_metrics: sc.stats.ext_metrics.clone(),
    }
}

/// Build a polars DataFrame from gauntlet rows.
fn build_dataframe(rows: &[GauntletRow]) -> PolarsResult<DataFrame> {
    let scenario: Vec<&str> = rows.iter().map(|r| r.scenario.as_str()).collect();
    let flags: Vec<&str> = rows.iter().map(|r| r.flags.as_str()).collect();
    let topology: Vec<&str> = rows.iter().map(|r| r.topology.as_str()).collect();
    let work_type: Vec<&str> = rows.iter().map(|r| r.work_type.as_str()).collect();
    let replica: Vec<u32> = rows.iter().map(|r| r.replica).collect();
    let passed: Vec<bool> = rows.iter().map(|r| r.passed).collect();
    let skipped: Vec<bool> = rows.iter().map(|r| r.skipped).collect();
    let spread: Vec<f64> = rows.iter().map(|r| r.spread).collect();
    let gap_ms: Vec<f64> = rows.iter().map(|r| r.gap_ms as f64).collect();
    let migrations: Vec<f64> = rows.iter().map(|r| r.migrations as f64).collect();
    let migration_ratio: Vec<f64> = rows.iter().map(|r| r.migration_ratio).collect();
    let imbalance: Vec<f64> = rows.iter().map(|r| r.imbalance_ratio).collect();
    let dsq_depth: Vec<f64> = rows.iter().map(|r| r.max_dsq_depth as f64).collect();
    let stalls: Vec<f64> = rows.iter().map(|r| r.stall_count as f64).collect();
    let fallback: Vec<f64> = rows.iter().map(|r| r.fallback_count as f64).collect();
    let keep_last: Vec<f64> = rows.iter().map(|r| r.keep_last_count as f64).collect();
    let p99_wake_lat: Vec<f64> = rows.iter().map(|r| r.p99_wake_latency_us).collect();
    let median_wake_lat: Vec<f64> = rows.iter().map(|r| r.median_wake_latency_us).collect();
    let wake_cv: Vec<f64> = rows.iter().map(|r| r.wake_latency_cv).collect();
    let total_iters: Vec<f64> = rows.iter().map(|r| r.total_iterations as f64).collect();
    let mean_run_delay: Vec<f64> = rows.iter().map(|r| r.mean_run_delay_us).collect();
    let worst_run_delay: Vec<f64> = rows.iter().map(|r| r.worst_run_delay_us).collect();
    let page_locality: Vec<f64> = rows.iter().map(|r| r.page_locality).collect();
    let cross_node_mig: Vec<f64> = rows.iter().map(|r| r.cross_node_migration_ratio).collect();
    let worst_deg_op: Vec<&str> = rows
        .iter()
        .map(|r| r.worst_degradation_op.as_str())
        .collect();
    let imbalance_delta: Vec<f64> = rows.iter().map(|r| r.worst_imbalance_delta).collect();
    let dsq_delta: Vec<f64> = rows.iter().map(|r| r.worst_dsq_delta).collect();
    let fallback_delta: Vec<f64> = rows.iter().map(|r| r.worst_fallback_delta).collect();
    let keep_last_delta: Vec<f64> = rows.iter().map(|r| r.worst_keep_last_delta).collect();
    let degradation_count: Vec<u32> = rows.iter().map(|r| r.degradation_count).collect();

    df!(
        "scenario" => &scenario,
        "flags" => &flags,
        "topology" => &topology,
        "work_type" => &work_type,
        "replica" => &replica,
        "passed" => &passed,
        "skipped" => &skipped,
        "spread" => &spread,
        "gap_ms" => &gap_ms,
        "migrations" => &migrations,
        "migration_ratio" => &migration_ratio,
        "imbalance" => &imbalance,
        "dsq_depth" => &dsq_depth,
        "stalls" => &stalls,
        "fallback" => &fallback,
        "keep_last" => &keep_last,
        "p99_wake_lat_us" => &p99_wake_lat,
        "median_wake_lat_us" => &median_wake_lat,
        "wake_latency_cv" => &wake_cv,
        "total_iterations" => &total_iters,
        "mean_run_delay_us" => &mean_run_delay,
        "worst_run_delay_us" => &worst_run_delay,
        "page_locality" => &page_locality,
        "cross_node_migration_ratio" => &cross_node_mig,
        "worst_deg_op" => &worst_deg_op,
        "imbalance_delta" => &imbalance_delta,
        "dsq_delta" => &dsq_delta,
        "fallback_delta" => &fallback_delta,
        "keep_last_delta" => &keep_last_delta,
        "degradation_count" => &degradation_count
    )
}

/// Detected outlier: a (scenario, flags) pair with an anomalous stat.
struct Outlier {
    scenario: String,
    flags: String,
    metric: &'static str,
    value: f64,
    overall_mean: f64,
    sigma: f64,
    worst_topos: Vec<String>,
}

impl std::fmt::Display for Outlier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} + {}: {} {:.1} (overall avg {:.1}, +{:.1}\u{03c3})",
            self.scenario, self.flags, self.metric, self.value, self.overall_mean, self.sigma
        )?;
        if !self.worst_topos.is_empty() {
            write!(f, "\n    worst on: {}", self.worst_topos.join(", "))?;
        }
        Ok(())
    }
}

/// Extract a column as a `ChunkedArray<Float64Type>`.
fn col_f64(df: &DataFrame, name: &str) -> Option<ChunkedArray<Float64Type>> {
    df.column(name)
        .ok()
        .and_then(|c| c.as_materialized_series().f64().ok().cloned())
}

/// Extract a column as a `ChunkedArray<UInt32Type>`.
fn col_u32(df: &DataFrame, name: &str) -> Option<ChunkedArray<UInt32Type>> {
    df.column(name)
        .ok()
        .and_then(|c| c.as_materialized_series().u32().ok().cloned())
}

/// Extract a column as a `ChunkedArray<Utf8Type>`.
fn col_str(df: &DataFrame, name: &str) -> Option<StringChunked> {
    df.column(name)
        .ok()
        .and_then(|c| c.as_materialized_series().str().ok().cloned())
}

/// Compute mean and stddev for a column, returning (mean, std).
fn col_mean_std(df: &DataFrame, name: &str) -> (f64, f64) {
    match col_f64(df, name) {
        Some(ca) => {
            let mean = ca.mean().unwrap_or(0.0);
            let std = ca.std(1).unwrap_or(0.0);
            (mean, std)
        }
        None => (0.0, 0.0),
    }
}

/// Find outlier (scenario, flags) pairs where a metric exceeds 2 sigma.
fn find_outliers(df: &DataFrame) -> Vec<Outlier> {
    let metrics: &[&str] = &[
        "spread",
        "gap_ms",
        "migrations",
        "migration_ratio",
        "imbalance",
        "dsq_depth",
        "stalls",
        "fallback",
        "keep_last",
        "p99_wake_lat_us",
        "wake_latency_cv",
        "mean_run_delay_us",
        "worst_run_delay_us",
    ];
    let mut outliers = Vec::new();

    for &metric in metrics {
        let (overall_mean, overall_std) = col_mean_std(df, metric);
        if overall_std < f64::EPSILON {
            continue;
        }
        let threshold = overall_mean + 2.0 * overall_std;

        // Group by (scenario, flags), compute mean of metric across topologies.
        let grouped = df
            .clone()
            .lazy()
            .group_by([col("scenario"), col("flags")])
            .agg([
                col(metric).mean().alias("metric_mean"),
                col(metric).max().alias("metric_max"),
            ])
            .collect();

        let grouped = match grouped {
            Ok(g) => g,
            Err(_) => continue,
        };

        let scenarios = col_str(&grouped, "scenario");
        let flags_col = col_str(&grouped, "flags");
        let means = col_f64(&grouped, "metric_mean");

        let (scenarios, flags_col, means) = match (scenarios, flags_col, means) {
            (Some(s), Some(f), Some(m)) => (s, f, m),
            _ => continue,
        };

        for i in 0..grouped.height() {
            let mean_val = means.get(i).unwrap_or(0.0);
            if mean_val <= threshold {
                continue;
            }
            let sigma = (mean_val - overall_mean) / overall_std;
            let sc = scenarios.get(i).unwrap_or("");
            let fl = flags_col.get(i).unwrap_or("");

            // Find worst topologies for this pair.
            let worst = find_worst_topos(df, sc, fl, metric, threshold);

            outliers.push(Outlier {
                scenario: sc.to_string(),
                flags: fl.to_string(),
                metric,
                value: mean_val,
                overall_mean,
                sigma,
                worst_topos: worst,
            });
        }
    }

    outliers.sort_by(|a, b| {
        b.sigma
            .partial_cmp(&a.sigma)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    outliers
}

/// Find topology names where a (scenario, flags) pair exceeds the threshold.
fn find_worst_topos(
    df: &DataFrame,
    scenario: &str,
    flags: &str,
    metric: &str,
    threshold: f64,
) -> Vec<String> {
    let filtered = df
        .clone()
        .lazy()
        .filter(
            col("scenario")
                .eq(lit(scenario))
                .and(col("flags").eq(lit(flags)))
                .and(col(metric).gt(lit(threshold))),
        )
        .select([col("topology")])
        .collect();

    match filtered {
        Ok(f) => col_str(&f, "topology")
            .map(|ca| {
                ca.into_iter()
                    .filter_map(|v| v.map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default(),
        Err(_) => vec![],
    }
}

/// Format a group-by summary for one dimension.
fn format_dimension_summary(df: &DataFrame, group_col: &str) -> String {
    let grouped = df
        .clone()
        .lazy()
        .group_by([col(group_col)])
        .agg([
            // pass_count excludes skipped rows — a skipped run is not
            // a successful execution and must not inflate pass rate.
            (col("passed").and(col("skipped").not()))
                .cast(DataType::UInt32)
                .sum()
                .alias("pass_count"),
            col("skipped")
                .cast(DataType::UInt32)
                .sum()
                .alias("skip_count"),
            col("passed").count().cast(DataType::UInt32).alias("total"),
            col("spread").mean().alias("avg_spread"),
            col("gap_ms").mean().alias("avg_gap_ms"),
            col("imbalance").mean().alias("avg_imbalance"),
            col("dsq_depth").mean().alias("avg_dsq_depth"),
            col("stalls").sum().alias("total_stalls"),
            col("fallback").mean().alias("avg_fallback"),
        ])
        .sort(
            ["avg_spread"],
            SortMultipleOptions::new().with_order_descending(true),
        )
        .collect();

    let grouped = match grouped {
        Ok(g) => g,
        Err(_) => return String::new(),
    };

    let mut out = String::new();
    let names = col_str(&grouped, group_col);
    let pass_counts = col_u32(&grouped, "pass_count");
    let skip_counts = col_u32(&grouped, "skip_count");
    let totals = col_u32(&grouped, "total");
    let spreads = col_f64(&grouped, "avg_spread");
    let gaps = col_f64(&grouped, "avg_gap_ms");

    let imbalances = col_f64(&grouped, "avg_imbalance");
    let dsq_depths = col_f64(&grouped, "avg_dsq_depth");
    let stall_totals = col_f64(&grouped, "total_stalls");
    let fallbacks = col_f64(&grouped, "avg_fallback");

    let (names, pass_counts, totals, spreads, gaps) =
        match (names, pass_counts, totals, spreads, gaps) {
            (Some(n), Some(p), Some(t), Some(s), Some(g)) => (n, p, t, s, g),
            _ => return out,
        };

    for i in 0..grouped.height() {
        let name = names.get(i).unwrap_or("?");
        let pass = pass_counts.get(i).unwrap_or(0);
        let skip = skip_counts.as_ref().and_then(|s| s.get(i)).unwrap_or(0);
        let total = totals.get(i).unwrap_or(0);
        let fail = total.saturating_sub(pass).saturating_sub(skip);
        let spread = spreads.get(i).unwrap_or(0.0);
        let gap = gaps.get(i).unwrap_or(0.0);
        let mut line = format!(
            "  {:<25} {}/{} passed ({} skipped, {} failed)  avg_spread={:.1}%  avg_gap={:.0}ms",
            name, pass, total, skip, fail, spread, gap
        );
        if let Some(ref imb) = imbalances {
            let v = imb.get(i).unwrap_or(0.0);
            if v > 1.0 {
                line.push_str(&format!("  imbal={:.1}", v));
            }
        }
        if let Some(ref dsq) = dsq_depths {
            let v = dsq.get(i).unwrap_or(0.0);
            if v > 0.0 {
                line.push_str(&format!("  dsq={:.0}", v));
            }
        }
        if let Some(ref st) = stall_totals {
            let v = st.get(i).unwrap_or(0.0) as u64;
            if v > 0 {
                line.push_str(&format!("  stalls={}", v));
            }
        }
        if let Some(ref fb) = fallbacks {
            let v = fb.get(i).unwrap_or(0.0);
            if v > 0.0 {
                line.push_str(&format!("  fallback={:.0}", v));
            }
        }
        line.push('\n');
        out.push_str(&line);
    }
    out
}

/// Classify a topology name into a CPU-count bucket.
///
/// CPU counts are derived from [`crate::vm::gauntlet_presets()`] at
/// first call and cached. Preset names not found in the cache return
/// `"unknown"`.
fn topo_bucket(topo: &str) -> &'static str {
    use std::collections::HashMap;
    use std::sync::OnceLock;

    static MAP: OnceLock<HashMap<String, u32>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        crate::vm::gauntlet_presets()
            .into_iter()
            .map(|p| (p.name.to_string(), p.topology.total_cpus()))
            .collect()
    });

    let cpus = match map.get(topo) {
        Some(&c) => c,
        None => return "unknown",
    };
    match cpus {
        0..=8 => "<=8cpu",
        9..=32 => "9-32cpu",
        33..=128 => "33-128cpu",
        _ => ">128cpu",
    }
}

/// Decompose a combo flag string like "borrow+rebal" into individual flags.
fn decompose_flags(flags: &str) -> Vec<&str> {
    if flags == "default" || flags.is_empty() {
        return vec![];
    }
    flags.split('+').collect()
}

/// Format the stimulus cross-tab analysis from the DataFrame.
fn format_stimulus_crosstab(df: &DataFrame) -> String {
    let mut out = String::new();

    // Check if any rows have degradation data.
    let has_data = col_u32(df, "degradation_count")
        .map(|ca| ca.sum().unwrap_or(0) > 0)
        .unwrap_or(false);

    if !has_data {
        return out;
    }

    out.push_str("\n=== STIMULUS CROSS-TAB ===\n");

    // --- Part 1: Worst stimulus by metric ---
    out.push_str("\nWorst stimulus by metric:\n");
    let delta_metrics: &[(&str, &str)] = &[
        ("imbalance", "imbalance_delta"),
        ("dsq_depth", "dsq_delta"),
        ("fallback", "fallback_delta"),
        ("keep_last", "keep_last_delta"),
    ];
    for &(metric, col_name) in delta_metrics {
        let grouped = df
            .clone()
            .lazy()
            .filter(col(col_name).gt(lit(0.0)))
            .group_by([col("worst_deg_op"), col("flags")])
            .agg([
                col(col_name).mean().alias("avg_delta"),
                col(col_name).count().alias("hit_count"),
            ])
            .sort(
                ["avg_delta"],
                SortMultipleOptions::new().with_order_descending(true),
            )
            .limit(1)
            .collect();

        match grouped {
            Ok(g) if g.height() > 0 => {
                let op = col_str(&g, "worst_deg_op")
                    .and_then(|ca| ca.get(0).map(|s| s.to_string()))
                    .unwrap_or_else(|| "?".to_string());
                let fl = col_str(&g, "flags")
                    .and_then(|ca| ca.get(0).map(|s| s.to_string()))
                    .unwrap_or_else(|| "?".to_string());
                let avg = col_f64(&g, "avg_delta")
                    .and_then(|ca| ca.get(0))
                    .unwrap_or(0.0);
                let hits = col_u32(&g, "hit_count")
                    .and_then(|ca| ca.get(0))
                    .unwrap_or(0);
                out.push_str(&format!(
                    "  {:<12} {} + {:<12} avg_delta={:+.1}  in {} runs\n",
                    metric, &op, &fl, avg, hits
                ));
            }
            _ => {
                out.push_str(&format!("  {:<12} (no significant degradation)\n", metric));
            }
        }
    }

    // Stalls: count runs with stalls > 0, grouped by worst_deg_op + flags.
    let stall_grouped = df
        .clone()
        .lazy()
        .filter(col("stalls").gt(lit(0.0)))
        .group_by([col("worst_deg_op"), col("flags")])
        .agg([
            col("stalls").sum().alias("total_stalls"),
            col("stalls").count().alias("run_count"),
        ])
        .sort(
            ["total_stalls"],
            SortMultipleOptions::new().with_order_descending(true),
        )
        .limit(1)
        .collect();

    match stall_grouped {
        Ok(g) if g.height() > 0 => {
            let op = col_str(&g, "worst_deg_op")
                .and_then(|ca| ca.get(0).map(|s| s.to_string()))
                .unwrap_or_else(|| "?".to_string());
            let fl = col_str(&g, "flags")
                .and_then(|ca| ca.get(0).map(|s| s.to_string()))
                .unwrap_or_else(|| "?".to_string());
            let total = col_f64(&g, "total_stalls")
                .and_then(|ca| ca.get(0))
                .unwrap_or(0.0) as u64;
            let runs = col_u32(&g, "run_count")
                .and_then(|ca| ca.get(0))
                .unwrap_or(0);
            if total > 0 {
                out.push_str(&format!(
                    "  {:<12} {} + {:<12} {} stalls in {} runs\n",
                    "stalls", &op, &fl, total, runs
                ));
            }
        }
        _ => {}
    }

    // --- Part 2: Flag decomposition ---
    // Decompose combo flags and compute avg metric delta per individual flag.
    // Done via row iteration since the polars `strings` feature is not enabled.
    let flags_ca = col_str(df, "flags");

    let all_flags_set: std::collections::BTreeSet<String> = flags_ca
        .as_ref()
        .map(|ca| {
            let mut s = std::collections::BTreeSet::new();
            for v in ca.into_iter().flatten() {
                for f in decompose_flags(v) {
                    s.insert(f.to_string());
                }
            }
            s
        })
        .unwrap_or_default();

    if !all_flags_set.is_empty() {
        let metric_cols = &["imbalance_delta", "dsq_delta", "fallback_delta"];
        let metric_labels = &["imbalance", "dsq_depth", "fallback"];

        let overall_means: Vec<f64> = metric_cols.iter().map(|m| col_mean_std(df, m).0).collect();

        out.push_str("\nFlag decomposition (avg metric delta when flag present vs absent):\n");
        out.push_str(&format!("  {:<14}", ""));
        for label in metric_labels {
            out.push_str(&format!("{:<14}", label));
        }
        out.push_str("stalls\n");

        // Build row-level flag membership mask and compute per-flag averages.
        let flags_ca = flags_ca.unwrap();
        let n = df.height();

        for flag in &all_flags_set {
            out.push_str(&format!("  {:<14}", flag));

            // Build boolean mask: rows where flags contains this individual flag.
            let mask: Vec<bool> = (0..n)
                .map(|i| {
                    flags_ca
                        .get(i)
                        .is_some_and(|v| decompose_flags(v).contains(&flag.as_str()))
                })
                .collect();

            for (mi, mc) in metric_cols.iter().enumerate() {
                let vals = col_f64(df, mc);
                let avg = vals
                    .as_ref()
                    .map(|ca| {
                        let masked: Vec<f64> = mask
                            .iter()
                            .enumerate()
                            .filter(|&(_, &m)| m)
                            .map(|(i, _)| ca.get(i).unwrap_or(0.0))
                            .collect();
                        if masked.is_empty() {
                            0.0
                        } else {
                            masked.iter().sum::<f64>() / masked.len() as f64
                        }
                    })
                    .unwrap_or(0.0);
                let delta = avg - overall_means[mi];
                out.push_str(&format!("{:+.1}{:<8}", delta, ""));
            }

            // Stalls for this flag.
            let stall_vals = col_f64(df, "stalls");
            let stall_avg = stall_vals
                .as_ref()
                .map(|ca| {
                    let masked: Vec<f64> = mask
                        .iter()
                        .enumerate()
                        .filter(|&(_, &m)| m)
                        .map(|(i, _)| ca.get(i).unwrap_or(0.0))
                        .collect();
                    if masked.is_empty() {
                        0.0
                    } else {
                        masked.iter().sum::<f64>() / masked.len() as f64
                    }
                })
                .unwrap_or(0.0);
            out.push_str(&format!("{:.1}\n", stall_avg));
        }
    }

    // --- Part 3: Flag x topology bucket ---
    if !all_flags_set.is_empty() {
        let buckets = &["<=8cpu", "9-32cpu", "33-128cpu", ">128cpu"];
        let n = df.height();

        let topo_ca = col_str(df, "topology");
        let flags_ca = col_str(df, "flags");
        let imb_ca = col_f64(df, "imbalance_delta");

        if let (Some(topo_ca), Some(flags_ca), Some(imb_ca)) = (topo_ca, flags_ca, imb_ca) {
            out.push_str("\nFlag x topology (imbalance delta):\n");
            out.push_str(&format!("  {:<14}", ""));
            for b in buckets {
                out.push_str(&format!("{:<12}", b));
            }
            out.push('\n');

            for flag in &all_flags_set {
                out.push_str(&format!("  {:<14}", flag));
                for &bucket in buckets {
                    let masked: Vec<f64> = (0..n)
                        .filter(|&i| {
                            flags_ca
                                .get(i)
                                .is_some_and(|v| decompose_flags(v).contains(&flag.as_str()))
                                && topo_ca.get(i).is_some_and(|v| topo_bucket(v) == bucket)
                        })
                        .map(|i| imb_ca.get(i).unwrap_or(0.0))
                        .collect();
                    let avg = if masked.is_empty() {
                        0.0
                    } else {
                        masked.iter().sum::<f64>() / masked.len() as f64
                    };
                    out.push_str(&format!("{:+.1}{:<6}", avg, ""));
                }
                out.push('\n');
            }
        }
    }

    out
}

/// Format per-cgroup pass rates, flagging cgroups below 100%.
fn format_cgroup_pass_rates(df: &DataFrame) -> String {
    let grouped = df
        .clone()
        .lazy()
        .group_by([
            col("scenario"),
            col("flags"),
            col("topology"),
            col("work_type"),
        ])
        .agg([
            // pass_count excludes skipped rows.
            (col("passed").and(col("skipped").not()))
                .cast(DataType::UInt32)
                .sum()
                .alias("pass_count"),
            col("skipped")
                .cast(DataType::UInt32)
                .sum()
                .alias("skip_count"),
            col("passed").count().cast(DataType::UInt32).alias("total"),
            col("spread").mean().alias("avg_spread"),
            col("spread").std(1).alias("std_spread"),
            col("gap_ms").mean().alias("avg_gap_ms"),
            col("gap_ms").min().alias("min_gap_ms"),
            col("gap_ms").max().alias("max_gap_ms"),
            col("imbalance").mean().alias("avg_imbalance"),
            col("dsq_depth").mean().alias("avg_dsq_depth"),
            col("fallback").mean().alias("avg_fallback"),
        ])
        .sort(
            ["pass_count"],
            SortMultipleOptions::new().with_order_descending(false),
        )
        .collect();

    let grouped = match grouped {
        Ok(g) => g,
        Err(_) => return String::new(),
    };

    let scenarios = col_str(&grouped, "scenario");
    let flags_col = col_str(&grouped, "flags");
    let topos = col_str(&grouped, "topology");
    let pass_counts = col_u32(&grouped, "pass_count");
    let skip_counts = col_u32(&grouped, "skip_count");
    let totals = col_u32(&grouped, "total");
    let spreads = col_f64(&grouped, "avg_spread");
    let std_spreads = col_f64(&grouped, "std_spread");
    let gaps = col_f64(&grouped, "avg_gap_ms");
    let min_gaps = col_f64(&grouped, "min_gap_ms");
    let max_gaps = col_f64(&grouped, "max_gap_ms");

    let (scenarios, flags_col, topos, pass_counts, totals) =
        match (scenarios, flags_col, topos, pass_counts, totals) {
            (Some(s), Some(f), Some(t), Some(p), Some(n)) => (s, f, t, p, n),
            _ => return String::new(),
        };

    let mut flaky = String::new();
    let mut all_pass = true;

    for i in 0..grouped.height() {
        let pass = pass_counts.get(i).unwrap_or(0);
        let skip = skip_counts.as_ref().and_then(|s| s.get(i)).unwrap_or(0);
        let total = totals.get(i).unwrap_or(0);
        // A group is "clean" when every executed (non-skipped) row
        // passed. Skipped rows are neither passes nor flakes.
        if total == 0 || pass + skip == total {
            continue;
        }
        all_pass = false;
        let sc = scenarios.get(i).unwrap_or("?");
        let fl = flags_col.get(i).unwrap_or("?");
        let tp = topos.get(i).unwrap_or("?");
        let fail = total.saturating_sub(pass).saturating_sub(skip);
        let mut line = if skip > 0 {
            format!("  {tp}/{sc}/{fl}  {pass}/{total} ({skip} skipped, {fail} failed)")
        } else {
            format!("  {tp}/{sc}/{fl}  {pass}/{total}")
        };
        if let Some(ref sp) = spreads {
            let avg = sp.get(i).unwrap_or(0.0);
            line.push_str(&format!("  spread={:.1}", avg));
        }
        if let Some(ref sp) = std_spreads {
            let std = sp.get(i).unwrap_or(0.0);
            if std > 0.0 {
                line.push_str(&format!("\u{00b1}{:.1}", std));
            }
        }
        if let Some(ref g) = gaps {
            let avg = g.get(i).unwrap_or(0.0);
            line.push_str(&format!("  gap={:.0}", avg));
        }
        if let (Some(mn), Some(mx)) = (&min_gaps, &max_gaps) {
            let min_v = mn.get(i).unwrap_or(0.0);
            let max_v = mx.get(i).unwrap_or(0.0);
            if max_v > min_v {
                line.push_str(&format!("[{:.0}-{:.0}]", min_v, max_v));
            }
        }
        line.push('\n');
        flaky.push_str(&line);
    }

    if all_pass {
        "All cgroups passed across all replicas.\n\n".to_string()
    } else {
        format!("Cgroups with <100% pass rate:\n{flaky}\n")
    }
}

/// Analyze pre-built gauntlet rows and return a formatted report.
pub fn analyze_rows(rows: &[GauntletRow]) -> String {
    if rows.is_empty() {
        return String::new();
    }

    let df = match build_dataframe(rows) {
        Ok(d) => d,
        Err(_) => return String::new(),
    };

    let mut report = String::from("\n=== GAUNTLET ANALYSIS ===\n\n");

    let has_replicas = col_u32(&df, "replica")
        .map(|ca| ca.max().unwrap_or(1) > 1)
        .unwrap_or(false);

    if has_replicas {
        report.push_str(&format_cgroup_pass_rates(&df));
    }

    let outliers = find_outliers(&df);
    if outliers.is_empty() {
        report.push_str("No outliers detected.\n");
    } else {
        report.push_str("Outliers detected:\n");
        for o in &outliers {
            report.push_str(&format!("  {o}\n"));
        }
    }

    report.push_str("\nBy scenario (worst first):\n");
    report.push_str(&format_dimension_summary(&df, "scenario"));

    report.push_str("\nBy flags:\n");
    report.push_str(&format_dimension_summary(&df, "flags"));

    report.push_str("\nBy topology:\n");
    report.push_str(&format_dimension_summary(&df, "topology"));

    let has_work_types = col_str(&df, "work_type")
        .map(|ca| ca.n_unique().unwrap_or(1) > 1)
        .unwrap_or(false);
    if has_work_types {
        report.push_str("\nBy work_type:\n");
        report.push_str(&format_dimension_summary(&df, "work_type"));
    }

    report.push_str(&format_stimulus_crosstab(&df));

    report
}

// ---------------------------------------------------------------------------
// Test-run enumeration and A/B comparison
// ---------------------------------------------------------------------------

/// List the test-run directories under
/// `{CARGO_TARGET_DIR or "target"}/ktstr/`.
///
/// Each subdirectory is one run keyed `{kernel}-{git_short}`. The
/// sidecar JSON files inside it are the run's results -- there is no
/// separate "baselines" cache; runs ARE baselines.
pub fn list_runs() -> anyhow::Result<()> {
    use std::fs;
    let root = crate::test_support::runs_root();
    if !root.exists() {
        eprintln!("no runs found at {}", root.display());
        return Ok(());
    }
    let mut entries: Vec<_> = fs::read_dir(&root)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();
    entries.sort_by_key(|e| e.file_name());

    println!("{:<40} {:>6}  DATE", "RUN", "TESTS");
    println!("{}", "-".repeat(60));
    for entry in &entries {
        let key = entry.file_name();
        let key_str = key.to_string_lossy();
        let sidecars = crate::test_support::collect_sidecars(&entry.path());
        let count = sidecars.len();
        let date = sidecars
            .iter()
            .map(|s| s.timestamp.as_str())
            .filter(|t| !t.is_empty())
            .min()
            .unwrap_or("-")
            .to_string();
        println!("{:<40} {:>6}  {}", key_str, count, date);
    }
    Ok(())
}

/// One significant per-metric finding produced by [`compare_rows`].
///
/// Each finding represents a single (scenario, topology, work_type,
/// metric) tuple whose A/B delta cleared both the absolute and
/// relative gates. The pairing key inside [`compare_rows`] is
/// `(scenario, topology, work_type)`; carrying `work_type` here lets
/// consumers disambiguate two findings that share scenario+topology
/// but were measured under different workloads. `metric` is the
/// registry entry the comparison ran against; consumers read
/// polarity, display unit, and name through it directly without
/// re-looking up [`metric_def`].
#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct Finding {
    pub scenario: String,
    pub topology: String,
    pub work_type: String,
    pub metric: &'static MetricDef,
    pub val_a: f64,
    pub val_b: f64,
    pub delta: f64,
    pub is_regression: bool,
}

/// Aggregate result of comparing two row sets via [`compare_rows`].
///
/// `regressions` and `improvements` count significant entries in
/// `findings`; `unchanged` counts metrics that fell below the dual
/// gate; `skipped_failed` counts paired (scenario, topology, work_type)
/// row pairs where either side has `passed=false`. `new_in_b`
/// counts B-side rows whose key has no match on the A side; the
/// converse is `removed_from_a`. The filter (when set) applies to
/// every counter, so excluded rows do not contribute.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub(crate) struct CompareReport {
    pub regressions: u32,
    pub improvements: u32,
    pub unchanged: u32,
    pub skipped_failed: u32,
    pub new_in_b: u32,
    pub removed_from_a: u32,
    pub findings: Vec<Finding>,
}

/// Compare two row sets metric-by-metric.
///
/// Pure function: no I/O, no globals. Pairs `rows_a` and `rows_b` by
/// `(scenario, topology, work_type)`. When `filter` is `Some(s)`, a
/// row is included only if `s` appears as a substring of the joined
/// `"scenario topology scheduler work_type"` string. The scheduler
/// is searchable via the filter but is not part of the pairing key,
/// so the same scenario+topology+work_type pair compares correctly
/// across different scheduler binaries when the filter does not
/// constrain it.
///
/// Row-pair accounting:
/// - B-side rows with no A-side match are counted in `new_in_b`.
/// - A-side rows with no B-side match are counted in `removed_from_a`
///   (a separate pass over `rows_a`).
/// - Paired rows where either side has `passed=false` are dropped
///   from the regression math and counted in `skipped_failed`: a
///   failed scenario's metrics reflect the failure mode (short run,
///   stalled workload, missing samples), not the scheduler's
///   behavior.
///
/// The filter (when set) applies to every counter -- excluded rows
/// never reach the matching, pass, or metric stages.
///
/// `threshold` is a relative percentage (e.g. `Some(10.0)` for 10%).
/// Deltas whose relative magnitude is below `threshold / 100.0` are
/// treated as unchanged. When `None`, each metric's built-in
/// `default_rel` is used. The absolute gate always uses the metric's
/// `default_abs`. A delta must clear both gates to count as
/// significant.
pub(crate) fn compare_rows(
    rows_a: &[GauntletRow],
    rows_b: &[GauntletRow],
    filter: Option<&str>,
    threshold: Option<f64>,
) -> CompareReport {
    let mut report = CompareReport::default();

    for row_b in rows_b {
        let key_b = (&row_b.scenario, &row_b.topology, &row_b.work_type);
        if let Some(f) = filter {
            let joined = format!(
                "{} {} {} {}",
                row_b.scenario, row_b.topology, row_b.scheduler, row_b.work_type,
            );
            if !joined.contains(f) {
                continue;
            }
        }
        let row_a = rows_a
            .iter()
            .find(|r| (&r.scenario, &r.topology, &r.work_type) == key_b);
        let Some(row_a) = row_a else {
            report.new_in_b += 1;
            continue;
        };

        // Drop from regression math when either side is a skip or a
        // failure. Skips carry no executed metrics (the run didn't
        // happen); failures carry telemetry dominated by the failure
        // mode (short run, stalled workload), not the scheduler's
        // behavior — comparing either against a real run produces
        // meaningless deltas.
        if !row_a.passed || !row_b.passed || row_a.skipped || row_b.skipped {
            report.skipped_failed += 1;
            continue;
        }

        for m in METRICS {
            let val_a = m.read(row_a).unwrap_or(0.0);
            let val_b = m.read(row_b).unwrap_or(0.0);
            if val_a.abs() < f64::EPSILON && val_b.abs() < f64::EPSILON {
                continue;
            }

            let rel_thresh = match threshold {
                Some(t) => t / 100.0,
                None => m.default_rel,
            };

            let delta = val_b - val_a;
            let rel_delta = if val_a.abs() > f64::EPSILON {
                (delta / val_a).abs()
            } else {
                0.0
            };

            if delta.abs() < m.default_abs || rel_delta < rel_thresh {
                report.unchanged += 1;
                continue;
            }

            let is_regression = if m.higher_is_worse() {
                delta > 0.0
            } else {
                delta < 0.0
            };
            if is_regression {
                report.regressions += 1;
            } else {
                report.improvements += 1;
            }
            report.findings.push(Finding {
                scenario: row_b.scenario.clone(),
                topology: row_b.topology.clone(),
                work_type: row_b.work_type.clone(),
                metric: m,
                val_a,
                val_b,
                delta,
                is_regression,
            });
        }
    }

    // Second pass: A-side rows whose key has no match on the B side.
    // Filter applies here too, so rows excluded by the filter never
    // count as removed.
    for row_a in rows_a {
        let key_a = (&row_a.scenario, &row_a.topology, &row_a.work_type);
        if let Some(f) = filter {
            let joined = format!(
                "{} {} {} {}",
                row_a.scenario, row_a.topology, row_a.scheduler, row_a.work_type,
            );
            if !joined.contains(f) {
                continue;
            }
        }
        let exists_in_b = rows_b
            .iter()
            .any(|r| (&r.scenario, &r.topology, &r.work_type) == key_a);
        if !exists_in_b {
            report.removed_from_a += 1;
        }
    }

    report
}

/// Compare two test runs and report regressions.
///
/// `a` and `b` are run keys (subdirectory names under
/// `{CARGO_TARGET_DIR or "target"}/ktstr/`) -- the same keys printed by
/// [`list_runs`]. Resolves run directories, loads sidecars, converts
/// to rows, and delegates dual-gate comparison to [`compare_rows`].
/// Prints a per-delta table and a summary line.
///
/// Returns 0 on no regressions, 1 if regressions detected.
pub fn compare_runs(
    a: &str,
    b: &str,
    filter: Option<&str>,
    threshold: Option<f64>,
) -> anyhow::Result<i32> {
    let root = crate::test_support::runs_root();
    let dir_a = root.join(a);
    let dir_b = root.join(b);
    if !dir_a.exists() {
        anyhow::bail!("run '{a}' not found under {}", root.display());
    }
    if !dir_b.exists() {
        anyhow::bail!("run '{b}' not found under {}", root.display());
    }
    let sidecars_a = crate::test_support::collect_sidecars(&dir_a);
    let sidecars_b = crate::test_support::collect_sidecars(&dir_b);
    if sidecars_a.is_empty() {
        anyhow::bail!("run '{a}' has no sidecar data");
    }
    if sidecars_b.is_empty() {
        anyhow::bail!("run '{b}' has no sidecar data");
    }

    let rows_a: Vec<GauntletRow> = sidecars_a.iter().map(sidecar_to_row).collect();
    let rows_b: Vec<GauntletRow> = sidecars_b.iter().map(sidecar_to_row).collect();

    let report = compare_rows(&rows_a, &rows_b, filter, threshold);

    println!(
        "{:<30} {:<34} {:>10} {:>10} {:>10}  VERDICT",
        "TEST", "METRIC", a, b, "DELTA"
    );
    println!("{}", "-".repeat(112));
    for f in &report.findings {
        let verdict = if f.is_regression {
            "\x1b[31mREGRESSION\x1b[0m"
        } else {
            "\x1b[32mimprovement\x1b[0m"
        };
        let label = format!("{}/{}/{}", f.scenario, f.topology, f.work_type);
        println!(
            "{:<30} {:<34} {:>10.2} {:>10.2} {:>+10.2}{:<2} {}",
            label, f.metric.name, f.val_a, f.val_b, f.delta, f.metric.display_unit, verdict,
        );
    }

    println!();
    println!(
        "summary: {} regressions, {} improvements, {} unchanged",
        report.regressions, report.improvements, report.unchanged,
    );
    if report.skipped_failed > 0 {
        println!(
            "  {} (scenario, topology, work_type) row pair(s) skipped \
             because one or both runs failed",
            report.skipped_failed,
        );
    }
    if report.new_in_b > 0 {
        println!(
            "  {} row(s) new in '{}' (no matching key in '{}')",
            report.new_in_b, b, a,
        );
    }
    if report.removed_from_a > 0 {
        println!(
            "  {} row(s) removed from '{}' (no matching key in '{}')",
            report.removed_from_a, a, b,
        );
    }

    Ok(if report.regressions > 0 { 1 } else { 0 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert::ScenarioStats;

    #[test]
    fn col_mean_std_basic() {
        let df = df!(
            "x" => &[1.0, 2.0, 3.0, 4.0, 5.0]
        )
        .unwrap();
        let (mean, std) = col_mean_std(&df, "x");
        assert!((mean - 3.0).abs() < 0.01);
        assert!(std > 1.0);
    }

    #[test]
    fn col_mean_std_missing_column() {
        let df = df!(
            "x" => &[1.0, 2.0, 3.0]
        )
        .unwrap();
        let (mean, std) = col_mean_std(&df, "nonexistent");
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
    }

    fn make_row(scenario: &str, flags: &str, topo: &str, passed: bool, spread: f64) -> GauntletRow {
        GauntletRow {
            scenario: scenario.into(),
            flags: flags.into(),
            topology: topo.into(),
            work_type: "CpuSpin".into(),
            scheduler: String::new(),
            skipped: false,
            replica: 1,
            passed,
            spread,
            gap_ms: 50,
            migrations: 10,
            migration_ratio: 0.0,
            imbalance_ratio: 1.0,
            max_dsq_depth: 2,
            stall_count: 0,
            fallback_count: 0,
            keep_last_count: 0,
            p99_wake_latency_us: 0.0,
            median_wake_latency_us: 0.0,
            wake_latency_cv: 0.0,
            total_iterations: 0,
            mean_run_delay_us: 0.0,
            worst_run_delay_us: 0.0,
            page_locality: 0.0,
            cross_node_migration_ratio: 0.0,
            worst_degradation_op: String::new(),
            worst_imbalance_delta: 0.0,
            worst_dsq_delta: 0.0,
            worst_fallback_delta: 0.0,
            worst_keep_last_delta: 0.0,
            degradation_count: 0,
            ext_metrics: BTreeMap::new(),
        }
    }

    // -- topo_bucket tests --

    #[test]
    fn topo_bucket_tiny() {
        assert_eq!(topo_bucket("tiny-1llc"), "<=8cpu");
        assert_eq!(topo_bucket("tiny-2llc"), "<=8cpu");
        #[cfg(not(target_arch = "aarch64"))]
        assert_eq!(topo_bucket("smt-2llc"), "<=8cpu");
    }

    #[test]
    fn topo_bucket_small() {
        assert_eq!(topo_bucket("odd-3llc"), "9-32cpu");
        assert_eq!(topo_bucket("odd-5llc"), "9-32cpu");
        assert_eq!(topo_bucket("odd-7llc"), "9-32cpu");
        #[cfg(not(target_arch = "aarch64"))]
        assert_eq!(topo_bucket("smt-3llc"), "9-32cpu");
        #[cfg(not(target_arch = "aarch64"))]
        assert_eq!(topo_bucket("medium-4llc"), "9-32cpu");
        assert_eq!(topo_bucket("medium-4llc-nosmt"), "9-32cpu");
        assert_eq!(topo_bucket("numa2-4llc"), "9-32cpu");
        assert_eq!(topo_bucket("numa4-8llc"), "9-32cpu");
    }

    #[test]
    fn topo_bucket_medium() {
        #[cfg(not(target_arch = "aarch64"))]
        assert_eq!(topo_bucket("medium-8llc"), "33-128cpu");
        #[cfg(not(target_arch = "aarch64"))]
        assert_eq!(topo_bucket("large-4llc"), "33-128cpu");
        #[cfg(not(target_arch = "aarch64"))]
        assert_eq!(topo_bucket("large-8llc"), "33-128cpu");
        assert_eq!(topo_bucket("medium-8llc-nosmt"), "33-128cpu");
        assert_eq!(topo_bucket("large-4llc-nosmt"), "33-128cpu");
        assert_eq!(topo_bucket("large-8llc-nosmt"), "33-128cpu");
        #[cfg(not(target_arch = "aarch64"))]
        assert_eq!(topo_bucket("numa2-8llc"), "33-128cpu");
        assert_eq!(topo_bucket("numa2-8llc-nosmt"), "33-128cpu");
    }

    #[test]
    fn topo_bucket_large() {
        #[cfg(not(target_arch = "aarch64"))]
        assert_eq!(topo_bucket("near-max-llc"), ">128cpu");
        #[cfg(not(target_arch = "aarch64"))]
        assert_eq!(topo_bucket("max-cpu"), ">128cpu");
        assert_eq!(topo_bucket("near-max-llc-nosmt"), ">128cpu");
        assert_eq!(topo_bucket("max-cpu-nosmt"), ">128cpu");
        #[cfg(not(target_arch = "aarch64"))]
        assert_eq!(topo_bucket("numa4-12llc"), ">128cpu");
    }

    #[test]
    fn topo_bucket_unknown() {
        assert_eq!(topo_bucket("not-a-preset"), "unknown");
        assert_eq!(topo_bucket(""), "unknown");
    }

    // -- decompose_flags tests --

    #[test]
    fn decompose_flags_single() {
        assert_eq!(decompose_flags("borrow"), vec!["borrow"]);
    }

    #[test]
    fn decompose_flags_multiple() {
        assert_eq!(decompose_flags("borrow+rebal"), vec!["borrow", "rebal"]);
    }

    #[test]
    fn decompose_flags_default() {
        assert!(decompose_flags("default").is_empty());
    }

    #[test]
    fn decompose_flags_empty() {
        assert!(decompose_flags("").is_empty());
    }

    // -- format_stimulus_crosstab tests --

    #[test]
    fn format_stimulus_crosstab_no_degradation_data() {
        let rows = vec![
            make_row("a", "borrow+rebal", "tiny-1llc", true, 5.0),
            make_row("b", "rebal", "medium-4llc", true, 8.0),
        ];
        let df = build_dataframe(&rows).unwrap();
        let out = format_stimulus_crosstab(&df);
        assert!(
            out.is_empty(),
            "no degradation data should produce empty output"
        );
    }

    #[test]
    fn format_stimulus_crosstab_computes_avg_delta() {
        // Two rows with imbalance_delta > 0 for the same (op, flags) group.
        // avg_delta should be (2.0 + 4.0) / 2 = 3.0, formatted as "+3.0".
        let mut r1 = make_row("a", "borrow", "tiny-1llc", true, 5.0);
        r1.worst_degradation_op = "SetCpuset".into();
        r1.worst_imbalance_delta = 2.0;
        r1.degradation_count = 1;
        let mut r2 = make_row("a", "borrow", "medium-4llc", true, 5.0);
        r2.worst_degradation_op = "SetCpuset".into();
        r2.worst_imbalance_delta = 4.0;
        r2.degradation_count = 1;
        let rows = vec![r1, r2];
        let df = build_dataframe(&rows).unwrap();
        let out = format_stimulus_crosstab(&df);
        // "imbalance    SetCpuset + borrow       avg_delta=+3.0  in 2 runs"
        assert!(
            out.contains("avg_delta=+3.0"),
            "expected avg_delta=+3.0, got:\n{out}"
        );
        assert!(
            out.contains("in 2 runs"),
            "expected 'in 2 runs', got:\n{out}"
        );
        assert!(
            out.contains("SetCpuset"),
            "expected op name SetCpuset, got:\n{out}"
        );
    }

    #[test]
    fn format_stimulus_crosstab_stalls_section() {
        // Row with stall_count=3 should produce "3 stalls in 1 runs".
        let mut r1 = make_row("a", "borrow", "tiny-1llc", true, 5.0);
        r1.worst_degradation_op = "Spawn".into();
        r1.stall_count = 3;
        r1.degradation_count = 1;
        let rows = vec![r1];
        let df = build_dataframe(&rows).unwrap();
        let out = format_stimulus_crosstab(&df);
        assert!(
            out.contains("3 stalls in 1 runs"),
            "expected stall count, got:\n{out}"
        );
        assert!(out.contains("Spawn"), "expected op Spawn, got:\n{out}");
    }

    #[test]
    fn format_stimulus_crosstab_flag_decomposition_values() {
        // Two flags: "borrow" with imbalance_delta=4.0, "rebal" with imbalance_delta=2.0.
        // Overall mean = (4.0 + 2.0) / 2 = 3.0.
        // borrow delta from mean = 4.0 - 3.0 = +1.0
        // rebal delta from mean = 2.0 - 3.0 = -1.0
        let mut r1 = make_row("a", "borrow", "tiny-1llc", true, 5.0);
        r1.worst_degradation_op = "SetCpuset".into();
        r1.worst_imbalance_delta = 4.0;
        r1.degradation_count = 1;
        let mut r2 = make_row("b", "rebal", "medium-4llc", true, 5.0);
        r2.worst_degradation_op = "Spawn".into();
        r2.worst_imbalance_delta = 2.0;
        r2.degradation_count = 1;
        let rows = vec![r1, r2];
        let df = build_dataframe(&rows).unwrap();
        let out = format_stimulus_crosstab(&df);
        // Flag decomposition should show "+1.0" for borrow and "-1.0" for rebal
        // in the imbalance column.
        assert!(
            out.contains("+1.0"),
            "borrow should be +1.0 above mean, got:\n{out}"
        );
        assert!(
            out.contains("-1.0"),
            "rebal should be -1.0 below mean, got:\n{out}"
        );
    }

    #[test]
    fn format_stimulus_crosstab_flag_x_topo_values() {
        // borrow on tiny-1llc (<=8cpu bucket) with imbalance_delta=3.0.
        // borrow on medium-4llc-nosmt (9-32cpu bucket) with imbalance_delta=7.0.
        // Use nosmt preset so the topology exists on aarch64 (which filters SMT presets).
        // The flag x topo table should show +3.0 for <=8cpu and +7.0 for 9-32cpu.
        let mut r1 = make_row("a", "borrow", "tiny-1llc", true, 5.0);
        r1.worst_degradation_op = "SetCpuset".into();
        r1.worst_imbalance_delta = 3.0;
        r1.degradation_count = 1;
        let mut r2 = make_row("a", "borrow", "medium-4llc-nosmt", true, 5.0);
        r2.worst_degradation_op = "SetCpuset".into();
        r2.worst_imbalance_delta = 7.0;
        r2.degradation_count = 1;
        let rows = vec![r1, r2];
        let df = build_dataframe(&rows).unwrap();
        let out = format_stimulus_crosstab(&df);
        assert!(
            out.contains("+3.0"),
            "<=8cpu bucket should show +3.0, got:\n{out}"
        );
        assert!(
            out.contains("+7.0"),
            "9-32cpu bucket should show +7.0, got:\n{out}"
        );
    }

    #[test]
    fn format_stimulus_crosstab_no_significant_metric() {
        // dsq_delta=0 for all rows -> dsq_depth line should say "(no significant degradation)".
        let mut r1 = make_row("a", "borrow", "tiny-1llc", true, 5.0);
        r1.worst_degradation_op = "SetCpuset".into();
        r1.worst_imbalance_delta = 2.0;
        r1.worst_dsq_delta = 0.0;
        r1.degradation_count = 1;
        let rows = vec![r1];
        let df = build_dataframe(&rows).unwrap();
        let out = format_stimulus_crosstab(&df);
        assert!(
            out.contains("dsq_depth") && out.contains("(no significant degradation)"),
            "dsq_depth should show no degradation, got:\n{out}"
        );
    }

    // -- format_cgroup_pass_rates tests --

    #[test]
    fn format_cgroup_pass_rates_all_pass() {
        let rows = vec![
            make_row("a", "f1", "t1", true, 5.0),
            make_row("a", "f1", "t1", true, 6.0),
        ];
        let df = build_dataframe(&rows).unwrap();
        let out = format_cgroup_pass_rates(&df);
        assert_eq!(out, "All cgroups passed across all replicas.\n\n");
    }

    #[test]
    fn format_cgroup_pass_rates_computed_values() {
        // 3 replicas: spread 10.0, 20.0, 30.0 -> avg=20.0, std>0.
        // gap_ms: 100, 200, 300 -> avg=200, min=100, max=300.
        // 2 pass, 1 fail.
        let mut rows = vec![
            make_row("scenario_a", "flags_x", "topo_y", true, 10.0),
            make_row("scenario_a", "flags_x", "topo_y", false, 20.0),
            make_row("scenario_a", "flags_x", "topo_y", true, 30.0),
        ];
        rows[0].gap_ms = 100;
        rows[1].gap_ms = 200;
        rows[2].gap_ms = 300;
        let df = build_dataframe(&rows).unwrap();
        let out = format_cgroup_pass_rates(&df);
        // Format: "  {tp}/{sc}/{fl}  {pass}/{total}  spread={avg}  gap={avg}[{min}-{max}]"
        assert!(out.contains("2/3"), "2/3 pass rate, got:\n{out}");
        assert!(out.contains("spread=20.0"), "avg spread=20.0, got:\n{out}");
        assert!(out.contains("gap=200"), "avg gap=200, got:\n{out}");
        assert!(
            out.contains("[100-300]"),
            "gap range [100-300], got:\n{out}"
        );
        // std_spread of [10,20,30] is 10.0 -> shows ±10.0
        assert!(out.contains("±10.0"), "std spread ±10.0, got:\n{out}");
    }

    #[test]
    fn format_cgroup_pass_rates_no_gap_range_when_equal() {
        // All same gap_ms -> no range shown.
        let rows = vec![
            make_row("a", "f1", "t1", true, 5.0),
            make_row("a", "f1", "t1", false, 5.0),
        ];
        let df = build_dataframe(&rows).unwrap();
        let out = format_cgroup_pass_rates(&df);
        assert!(out.contains("1/2"), "1/2, got:\n{out}");
        // gap_ms=50 for both via make_row -> no range brackets
        assert!(
            !out.contains("[50-50]"),
            "equal gaps should not show range, got:\n{out}"
        );
    }

    // -- format_dimension_summary tests --

    #[test]
    fn format_dimension_summary_computed_values() {
        // Two scenarios: "fast" with spread=4.0, gap=40, and "slow" with spread=20.0, gap=200.
        // Each has 1 row. format_dimension_summary sorts by avg_spread descending.
        let mut r1 = make_row("slow", "default", "tiny-1llc", false, 20.0);
        r1.gap_ms = 200;
        r1.imbalance_ratio = 2.5; // > 1.0, should show imbal=2.5
        r1.max_dsq_depth = 8; // > 0, should show dsq=8
        r1.stall_count = 2; // > 0, should show stalls=2
        r1.fallback_count = 15; // > 0, should show fallback=15
        let r2 = make_row("fast", "default", "tiny-1llc", true, 4.0);
        let rows = vec![r1, r2];
        let df = build_dataframe(&rows).unwrap();
        let out = format_dimension_summary(&df, "scenario");
        // "slow" has higher spread, should appear first (sorted descending).
        let slow_pos = out.find("slow").unwrap();
        let fast_pos = out.find("fast").unwrap();
        assert!(
            slow_pos < fast_pos,
            "slow should sort before fast, got:\n{out}"
        );
        // Check computed values for "slow"
        assert!(out.contains("0/1 passed"), "slow: 0/1 passed, got:\n{out}");
        assert!(
            out.contains("avg_spread=20.0%"),
            "slow: avg_spread=20.0%, got:\n{out}"
        );
        assert!(
            out.contains("avg_gap=200ms"),
            "slow: avg_gap=200ms, got:\n{out}"
        );
        assert!(out.contains("imbal=2.5"), "slow: imbal=2.5, got:\n{out}");
        assert!(out.contains("dsq=8"), "slow: dsq=8, got:\n{out}");
        assert!(out.contains("stalls=2"), "slow: stalls=2, got:\n{out}");
        assert!(
            out.contains("fallback=15"),
            "slow: fallback=15, got:\n{out}"
        );
        // "fast" should show 1/1 passed
        assert!(out.contains("1/1 passed"), "fast: 1/1 passed, got:\n{out}");
    }

    // -- analyze_rows tests --

    #[test]
    fn analyze_rows_empty() {
        assert!(analyze_rows(&[]).is_empty());
    }

    #[test]
    fn analyze_rows_with_work_type_diversity() {
        let mut rows = vec![
            make_row("a", "f1", "t1", true, 5.0),
            make_row("a", "f1", "t1", true, 6.0),
        ];
        rows[0].work_type = "CpuSpin".into();
        rows[1].work_type = "Bursty".into();
        let report = analyze_rows(&rows);
        assert!(
            report.contains("By work_type"),
            "should show work_type section when diverse"
        );
        assert!(report.contains("CpuSpin"), "should list CpuSpin");
        assert!(report.contains("Bursty"), "should list Bursty");
    }

    #[test]
    fn analyze_rows_no_work_type_section_when_uniform() {
        let rows = vec![
            make_row("a", "f1", "t1", true, 5.0),
            make_row("b", "f2", "t2", true, 8.0),
        ];
        let report = analyze_rows(&rows);
        assert!(
            !report.contains("By work_type"),
            "should not show work_type when uniform"
        );
    }

    // -- sidecar_to_row tests --

    #[test]
    fn sidecar_to_row_basic() {
        use crate::monitor;
        use crate::test_support;
        let sc = test_support::SidecarResult {
            test_name: "my_test".to_string(),
            topology: "1n2l4c2t".to_string(),
            scheduler: "scx_mitosis".to_string(),
            payload: None,
            metrics: vec![],
            passed: true,
            skipped: false,
            stats: ScenarioStats {
                cgroups: vec![],
                total_workers: 4,
                total_cpus: 8,
                total_migrations: 12,
                worst_spread: 15.0,
                worst_gap_ms: 200,
                worst_gap_cpu: 3,
                ..Default::default()
            },
            monitor: Some(monitor::MonitorSummary {
                total_samples: 10,
                max_imbalance_ratio: 2.5,
                max_local_dsq_depth: 4,
                stall_detected: true,
                event_deltas: Some(monitor::ScxEventDeltas {
                    total_fallback: 7,
                    fallback_rate: 0.5,
                    max_fallback_burst: 2,
                    total_dispatch_offline: 0,
                    total_dispatch_keep_last: 3,
                    keep_last_rate: 0.2,
                    total_enq_skip_exiting: 0,
                    total_enq_skip_migration_disabled: 0,
                    ..Default::default()
                }),
                schedstat_deltas: None,
                prog_stats_deltas: None,
                ..Default::default()
            }),
            stimulus_events: vec![],
            work_type: "CpuSpin".to_string(),
            active_flags: Vec::new(),
            verifier_stats: vec![],
            kvm_stats: None,
            sysctls: vec![],
            kargs: vec![],
            kernel_version: None,
            timestamp: String::new(),
            run_id: String::new(),
        };
        let row = sidecar_to_row(&sc);
        assert_eq!(row.scenario, "my_test");
        assert_eq!(row.topology, "1n2l4c2t");
        assert!(row.passed);
        assert_eq!(row.spread, 15.0);
        assert_eq!(row.gap_ms, 200);
        assert_eq!(row.migrations, 12);
        assert_eq!(row.imbalance_ratio, 2.5);
        assert_eq!(row.max_dsq_depth, 4);
        assert_eq!(row.stall_count, 1);
        assert_eq!(row.fallback_count, 7);
        assert_eq!(row.keep_last_count, 3);
        assert!(row.flags.is_empty());
        assert_eq!(row.replica, 1);
    }

    #[test]
    fn sidecar_to_row_no_monitor() {
        use crate::test_support;
        let sc = test_support::SidecarResult {
            test_name: "eevdf_test".to_string(),
            topology: "1n1l2c1t".to_string(),
            scheduler: "eevdf".to_string(),
            payload: None,
            metrics: vec![],
            skipped: false,
            passed: false,
            stats: Default::default(),
            monitor: None,
            stimulus_events: vec![],
            work_type: "CpuSpin".to_string(),
            active_flags: Vec::new(),
            verifier_stats: vec![],
            kvm_stats: None,
            sysctls: vec![],
            kargs: vec![],
            kernel_version: None,
            timestamp: String::new(),
            run_id: String::new(),
        };
        let row = sidecar_to_row(&sc);
        assert_eq!(row.scenario, "eevdf_test");
        assert!(!row.passed);
        assert_eq!(row.imbalance_ratio, 0.0);
        assert_eq!(row.max_dsq_depth, 0);
        assert_eq!(row.stall_count, 0);
        assert_eq!(row.fallback_count, 0);
        assert_eq!(row.keep_last_count, 0);
    }

    #[test]
    fn sidecar_to_row_no_stall() {
        use crate::monitor;
        use crate::test_support;
        let sc = test_support::SidecarResult {
            test_name: "t".to_string(),
            topology: "1n1l1c1t".to_string(),
            skipped: false,
            scheduler: "test".to_string(),
            payload: None,
            metrics: vec![],
            passed: true,
            stats: Default::default(),
            monitor: Some(monitor::MonitorSummary {
                prog_stats_deltas: None,
                total_samples: 5,
                max_imbalance_ratio: 1.0,
                max_local_dsq_depth: 0,
                stall_detected: false,
                event_deltas: None,
                schedstat_deltas: None,
                ..Default::default()
            }),
            stimulus_events: vec![],
            work_type: "CpuSpin".to_string(),
            active_flags: Vec::new(),
            verifier_stats: vec![],
            kvm_stats: None,
            sysctls: vec![],
            kargs: vec![],
            kernel_version: None,
            timestamp: String::new(),
            run_id: String::new(),
        };
        let row = sidecar_to_row(&sc);
        assert_eq!(row.stall_count, 0);
        assert_eq!(row.fallback_count, 0);
        assert_eq!(row.keep_last_count, 0);
    }

    // -- metric_def tests --

    #[test]
    fn metric_def_known() {
        let d = metric_def("worst_spread").unwrap();
        assert_eq!(d.name, "worst_spread");
        assert!(d.higher_is_worse());
        assert_eq!(d.display_unit, "%");
    }

    #[test]
    fn metric_def_not_higher_is_worse() {
        let d = metric_def("total_iterations").unwrap();
        assert!(!d.higher_is_worse());
    }

    #[test]
    fn metric_def_unknown() {
        assert!(metric_def("nonexistent").is_none());
    }

    #[test]
    fn metric_def_polarity_inverse_sense() {
        use crate::test_support::Polarity;
        // higher_is_worse=true means growing = regression; the
        // Polarity for "what do we want it to move toward?" is
        // LowerBetter.
        let d = metric_def("worst_spread").unwrap();
        assert!(d.higher_is_worse());
        assert_eq!(d.polarity, Polarity::LowerBetter);
        // higher_is_worse=false means growing = improvement; the
        // Polarity is HigherBetter.
        let d = metric_def("total_iterations").unwrap();
        assert!(!d.higher_is_worse());
        assert_eq!(d.polarity, Polarity::HigherBetter);
    }

    #[test]
    fn metric_def_polarity_covers_all_entries() {
        use crate::test_support::Polarity;
        // Every METRICS entry must map cleanly to HigherBetter or
        // LowerBetter; no entry should produce TargetValue or Unknown
        // from the bool->Polarity adaptor.
        for m in METRICS.iter() {
            assert!(
                matches!(m.polarity, Polarity::HigherBetter | Polarity::LowerBetter),
                "metric {} produced non-binary polarity {:?}",
                m.name,
                m.polarity
            );
        }
    }

    #[test]
    fn metric_def_all_entries_unique() {
        let mut names: Vec<&str> = METRICS.iter().map(|m| m.name).collect();
        let len = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), len);
    }

    // -- MetricDef::read tests --

    fn read_metric(row: &GauntletRow, name: &str) -> Option<f64> {
        metric_def(name).expect("metric name").read(row)
    }

    #[test]
    fn metric_def_read_named_fields() {
        let mut row = make_row("a", "f", "t", true, 42.0);
        row.gap_ms = 100;
        row.migrations = 7;
        row.migration_ratio = 0.3;
        row.imbalance_ratio = 2.0;
        row.max_dsq_depth = 5;
        row.stall_count = 3;
        row.fallback_count = 11;
        row.keep_last_count = 4;
        row.p99_wake_latency_us = 99.0;
        row.median_wake_latency_us = 50.0;
        row.wake_latency_cv = 0.5;
        row.total_iterations = 1000;
        row.mean_run_delay_us = 25.0;
        row.worst_run_delay_us = 200.0;
        row.page_locality = 0.8;
        row.cross_node_migration_ratio = 0.1;
        assert_eq!(read_metric(&row, "worst_spread"), Some(42.0));
        assert_eq!(read_metric(&row, "worst_gap_ms"), Some(100.0));
        assert_eq!(read_metric(&row, "total_migrations"), Some(7.0));
        assert_eq!(read_metric(&row, "worst_migration_ratio"), Some(0.3));
        assert_eq!(read_metric(&row, "max_imbalance_ratio"), Some(2.0));
        assert_eq!(read_metric(&row, "max_dsq_depth"), Some(5.0));
        assert_eq!(read_metric(&row, "stall_count"), Some(3.0));
        assert_eq!(read_metric(&row, "total_fallback"), Some(11.0));
        assert_eq!(read_metric(&row, "total_keep_last"), Some(4.0));
        assert_eq!(read_metric(&row, "p99_wake_latency_us"), Some(99.0));
        assert_eq!(read_metric(&row, "median_wake_latency_us"), Some(50.0));
        assert_eq!(read_metric(&row, "wake_latency_cv"), Some(0.5));
        assert_eq!(read_metric(&row, "total_iterations"), Some(1000.0));
        assert_eq!(read_metric(&row, "mean_run_delay_us"), Some(25.0));
        assert_eq!(read_metric(&row, "worst_run_delay_us"), Some(200.0));
        assert_eq!(read_metric(&row, "worst_page_locality"), Some(0.8));
        assert_eq!(
            read_metric(&row, "worst_cross_node_migration_ratio"),
            Some(0.1)
        );
    }

    #[test]
    fn metric_def_read_prefers_accessor_over_ext_metrics() {
        // When a name is in METRICS, the built-in accessor wins.
        // Even if ext_metrics carries a colliding entry for the
        // same name, MetricDef::read returns the accessor's value
        // — built-in fields are the authoritative source.
        let mut row = make_row("a", "f", "t", true, 5.0);
        row.ext_metrics.insert("worst_spread".into(), 999.0);
        assert_eq!(read_metric(&row, "worst_spread"), Some(5.0));

        // User ext_metrics with no matching MetricDef are reachable
        // via the direct ext_metrics map; metric_def returns None
        // for unregistered names.
        row.ext_metrics.insert("custom_metric".into(), 77.0);
        assert!(metric_def("custom_metric").is_none());
        assert_eq!(row.ext_metrics.get("custom_metric").copied(), Some(77.0));
    }

    // -- compare_rows tests --

    /// Build a row matching the sidecar-derived schema: empty `flags`,
    /// `replica = 1`, `work_type = "CpuSpin"`, all metrics zeroed
    /// except `spread` and `total_iterations`.
    fn cmp_row(scenario: &str, topo: &str, passed: bool, spread: f64, iters: u64) -> GauntletRow {
        let mut r = make_row(scenario, "", topo, passed, spread);
        r.gap_ms = 0;
        r.migrations = 0;
        r.imbalance_ratio = 0.0;
        r.max_dsq_depth = 0;
        r.total_iterations = iters;
        r
    }

    #[test]
    fn compare_rows_dual_gate_both_must_trigger() {
        // worst_spread default_abs=5.0, default_rel=0.25.
        // 10 -> 12: abs delta 2.0 < 5.0 (abs gate fails); rel 0.20 < 0.25
        // (rel gate also fails). Result: 0 regressions, 0 improvements,
        // unchanged for worst_spread.
        let rows_a = vec![cmp_row("test_a", "tiny-1llc", true, 10.0, 0)];
        let rows_b = vec![cmp_row("test_a", "tiny-1llc", true, 12.0, 0)];
        let res = compare_rows(&rows_a, &rows_b, None, None);
        assert_eq!(res.regressions, 0, "abs gate must block 2.0 < 5.0");
        assert_eq!(res.improvements, 0);
        assert_eq!(
            res.unchanged, 1,
            "worst_spread should be classified unchanged"
        );
        assert!(res.findings.is_empty());

        // Confirm the rel gate alone is not enough: spread 10 -> 14 has
        // rel 0.40 (>= 0.25) but abs delta 4.0 (< 5.0), still unchanged.
        let rows_b2 = vec![cmp_row("test_a", "tiny-1llc", true, 14.0, 0)];
        let res2 = compare_rows(&rows_a, &rows_b2, None, None);
        assert_eq!(
            res2.regressions, 0,
            "rel-only is insufficient: abs gate must also fire"
        );
        assert_eq!(res2.unchanged, 1);
    }

    #[test]
    fn compare_rows_synthetic_regression_and_improvement() {
        // spread 10 -> 30: abs delta 20.0 >= 5.0, rel 2.0 >= 0.10 →
        // regression (higher_is_worse).
        // total_iterations 1000 -> 500: abs delta 500 >= 100, rel 0.5
        // >= 0.10, higher_is_worse=false so decrease is a regression.
        // Net: 2 regressions, 0 improvements; one Finding per
        // significant metric.
        let rows_a = vec![cmp_row("test1", "tiny-1llc", true, 10.0, 1000)];
        let rows_b = vec![cmp_row("test1", "tiny-1llc", true, 30.0, 500)];
        let res = compare_rows(&rows_a, &rows_b, None, Some(10.0));
        assert_eq!(
            res.regressions, 2,
            "spread up + iterations down both regress"
        );
        assert_eq!(res.improvements, 0);
        assert_eq!(res.skipped_failed, 0);
        let metrics: Vec<&str> = res.findings.iter().map(|d| d.metric.name).collect();
        assert!(metrics.contains(&"worst_spread"));
        assert!(metrics.contains(&"total_iterations"));
        for d in &res.findings {
            assert!(d.is_regression, "all reported deltas should be regressions");
            assert_eq!(d.scenario, "test1");
            assert_eq!(d.topology, "tiny-1llc");
        }

        // Reverse direction: improvements should also surface.
        let res_imp = compare_rows(&rows_b, &rows_a, None, Some(10.0));
        assert_eq!(res_imp.regressions, 0);
        assert_eq!(res_imp.improvements, 2);
        for d in &res_imp.findings {
            assert!(!d.is_regression);
        }
    }

    #[test]
    fn compare_rows_higher_is_worse_inversion() {
        // total_iterations is higher_is_worse=false. A drop of 1000 ->
        // 500 must be reported as a regression, not an improvement.
        let rows_a = vec![cmp_row("t", "tiny-1llc", true, 0.0, 1000)];
        let rows_b = vec![cmp_row("t", "tiny-1llc", true, 0.0, 500)];
        let res = compare_rows(&rows_a, &rows_b, None, None);
        let iters_delta = res
            .findings
            .iter()
            .find(|d| d.metric.name == "total_iterations")
            .expect("total_iterations should produce a delta");
        assert!(
            iters_delta.is_regression,
            "iterations decrease is a regression"
        );
        assert_eq!(iters_delta.delta, -500.0);
        assert_eq!(res.regressions, 1);
        assert_eq!(res.improvements, 0);

        // worst_spread is higher_is_worse=true. An increase must be a
        // regression; a decrease must be an improvement.
        let rows_a2 = vec![cmp_row("t", "tiny-1llc", true, 10.0, 0)];
        let rows_b2 = vec![cmp_row("t", "tiny-1llc", true, 30.0, 0)];
        let res_up = compare_rows(&rows_a2, &rows_b2, None, None);
        let spread_up = res_up
            .findings
            .iter()
            .find(|d| d.metric.name == "worst_spread")
            .expect("worst_spread should produce a delta");
        assert!(spread_up.is_regression, "spread increase is a regression");
        assert_eq!(spread_up.delta, 20.0);

        let res_down = compare_rows(&rows_b2, &rows_a2, None, None);
        let spread_down = res_down
            .findings
            .iter()
            .find(|d| d.metric.name == "worst_spread")
            .expect("worst_spread should produce a delta");
        assert!(
            !spread_down.is_regression,
            "spread decrease is an improvement"
        );
        assert_eq!(spread_down.delta, -20.0);
    }

    #[test]
    fn compare_rows_skipped_side_drops_pair_into_skipped_failed() {
        // A skipped row on either side of the comparison must not
        // contribute to regressions/improvements — a skipped run
        // carries no executed metrics. `row.passed == true` for skips
        // would otherwise let the pair through the regression math,
        // producing meaningless deltas against default-zero metric
        // values.
        let mut row_a = cmp_row("t", "tiny-1llc", true, 10.0, 100);
        let mut row_b = cmp_row("t", "tiny-1llc", true, 10.0, 100);
        row_a.skipped = true; // A side was skipped
        let res = compare_rows(&[row_a.clone()], &[row_b.clone()], None, None);
        assert_eq!(res.regressions, 0);
        assert_eq!(res.improvements, 0);
        assert_eq!(
            res.skipped_failed, 1,
            "skipped side must count as skipped_failed, not produce deltas"
        );

        // Symmetrically on the B side.
        row_a.skipped = false;
        row_b.skipped = true;
        let res = compare_rows(&[row_a], &[row_b], None, None);
        assert_eq!(res.regressions, 0);
        assert_eq!(res.improvements, 0);
        assert_eq!(res.skipped_failed, 1);
    }

    /// Rows where either side has `passed=false` are dropped from the
    /// regression math. A failed scenario's metrics reflect the failure
    /// mode (short run, stalled workload, missing samples), not
    /// scheduler behavior.
    #[test]
    fn compare_rows_skips_failed_scenarios() {
        // Three scenarios, all with the same metric movement. Only
        // test_ok (passed on both sides) should be eligible for the
        // regression math; the other two are counted as skipped_failed.
        let rows_a = vec![
            cmp_row("test_ok", "tiny-1llc", true, 10.0, 1000),
            cmp_row("test_failed_b", "tiny-1llc", true, 10.0, 1000),
            cmp_row("test_failed_a", "tiny-1llc", false, 10.0, 1000),
        ];
        let rows_b = vec![
            cmp_row("test_ok", "tiny-1llc", true, 30.0, 500),
            cmp_row("test_failed_b", "tiny-1llc", false, 30.0, 500),
            cmp_row("test_failed_a", "tiny-1llc", true, 30.0, 500),
        ];
        let res = compare_rows(&rows_a, &rows_b, None, Some(10.0));
        assert_eq!(
            res.skipped_failed, 2,
            "test_failed_a and test_failed_b skip"
        );
        // test_ok regresses on worst_spread and total_iterations only.
        assert_eq!(res.regressions, 2);
        assert_eq!(res.improvements, 0);
        for d in &res.findings {
            assert_eq!(d.scenario, "test_ok");
        }
    }

    #[test]
    fn compare_rows_filter_substring() {
        // Two scenarios in each run. Filter "alpha" must match the
        // alpha row (substring of the joined "scenario topology
        // scheduler work_type" string) and exclude the beta row.
        let rows_a = vec![
            cmp_row("alpha", "tiny-1llc", true, 10.0, 0),
            cmp_row("beta", "tiny-1llc", true, 10.0, 0),
        ];
        let rows_b = vec![
            cmp_row("alpha", "tiny-1llc", true, 30.0, 0),
            cmp_row("beta", "tiny-1llc", true, 30.0, 0),
        ];
        let res = compare_rows(&rows_a, &rows_b, Some("alpha"), None);
        assert_eq!(res.regressions, 1, "only alpha row should compare");
        assert_eq!(res.findings.len(), 1);
        assert_eq!(res.findings[0].scenario, "alpha");
        // Finding carries work_type so two findings sharing
        // scenario+topology under different workloads stay
        // distinguishable.
        assert_eq!(res.findings[0].work_type, "CpuSpin");

        // Filter on topology substring is also honored. Both rows
        // share the "tiny-1llc" topology and only worst_spread crosses
        // both gates (10 -> 30 with default_abs=5.0, default_rel=0.25),
        // so each row contributes exactly one finding.
        let res_topo = compare_rows(&rows_a, &rows_b, Some("tiny"), None);
        assert_eq!(res_topo.regressions, 2, "both rows match 'tiny' topology");
        assert_eq!(res_topo.findings.len(), 2);

        // Non-matching filter yields no comparisons at all.
        let res_none = compare_rows(&rows_a, &rows_b, Some("nomatch"), None);
        assert_eq!(res_none.regressions, 0);
        assert_eq!(res_none.improvements, 0);
        assert_eq!(res_none.unchanged, 0);
        assert_eq!(res_none.skipped_failed, 0);
    }

    #[test]
    fn compare_rows_threshold_override() {
        // worst_spread default_rel=0.25, default_abs=5.0. Move 100 ->
        // 106: abs delta 6.0 >= 5.0 (abs gate passes); rel 0.06 < 0.25
        // (default rel fails) → unchanged with default thresholds.
        let rows_a = vec![cmp_row("t", "tiny-1llc", true, 100.0, 0)];
        let rows_b = vec![cmp_row("t", "tiny-1llc", true, 106.0, 0)];
        let res_default = compare_rows(&rows_a, &rows_b, None, None);
        let spread_default = res_default
            .findings
            .iter()
            .find(|d| d.metric.name == "worst_spread");
        assert!(
            spread_default.is_none(),
            "default rel 0.25 must classify 6% change as unchanged"
        );

        // Override threshold to 5% (Some(5.0) → rel_thresh 0.05). Now
        // rel 0.06 >= 0.05, both gates fire → regression.
        let res_override = compare_rows(&rows_a, &rows_b, None, Some(5.0));
        let spread_override = res_override
            .findings
            .iter()
            .find(|d| d.metric.name == "worst_spread")
            .expect("override 5% must surface 6% spread change");
        assert!(spread_override.is_regression);
        assert_eq!(spread_override.delta, 6.0);

        // The override does NOT loosen the abs gate. Move 1.0 -> 1.5:
        // abs delta 0.5 < 5.0; even threshold=1% (rel_thresh 0.01)
        // can't promote it to significant.
        let rows_a_small = vec![cmp_row("t", "tiny-1llc", true, 1.0, 0)];
        let rows_b_small = vec![cmp_row("t", "tiny-1llc", true, 1.5, 0)];
        let res_small = compare_rows(&rows_a_small, &rows_b_small, None, Some(1.0));
        assert!(
            !res_small
                .findings
                .iter()
                .any(|d| d.metric.name == "worst_spread"),
            "abs gate must still block tiny absolute moves"
        );
    }

    /// `compare_rows` uses `Iterator::find` to locate the A-side
    /// match for each B-side row, so when `rows_a` contains two
    /// entries with the same `(scenario, topology, work_type)` key
    /// the first one wins. Lock that contract in: the second
    /// duplicate must be ignored even though it would change the
    /// verdict.
    #[test]
    fn compare_rows_duplicate_key_first_match_wins() {
        // First A-side entry has spread=10 (would yield a regression
        // against B's 30). Second has spread=29 (would be unchanged).
        // The result must reflect the first entry only.
        let rows_a = vec![
            cmp_row("t", "tiny-1llc", true, 10.0, 0),
            cmp_row("t", "tiny-1llc", true, 29.0, 0),
        ];
        let rows_b = vec![cmp_row("t", "tiny-1llc", true, 30.0, 0)];
        let res = compare_rows(&rows_a, &rows_b, None, None);
        assert_eq!(res.regressions, 1, "first match (spread=10) must win");
        let spread = res
            .findings
            .iter()
            .find(|d| d.metric.name == "worst_spread")
            .expect("worst_spread regression should fire");
        assert_eq!(
            spread.val_a, 10.0,
            "val_a must come from the first matching row"
        );
        assert_eq!(spread.delta, 20.0);
    }

    /// Filtering is applied before the failed-row gate. A failed row
    /// that the filter excludes never reaches the `passed` check, so
    /// `skipped_failed` stays at zero -- the failure on the filtered
    /// row is invisible by design.
    #[test]
    fn compare_rows_filter_excludes_failed_from_skip_count() {
        let rows_a = vec![
            cmp_row("alpha", "tiny-1llc", true, 10.0, 0),
            cmp_row("beta", "tiny-1llc", false, 10.0, 0),
        ];
        let rows_b = vec![
            cmp_row("alpha", "tiny-1llc", true, 30.0, 0),
            cmp_row("beta", "tiny-1llc", true, 30.0, 0),
        ];
        // Without a filter, beta's failed row contributes
        // skipped_failed=1.
        let unfiltered = compare_rows(&rows_a, &rows_b, None, None);
        assert_eq!(unfiltered.skipped_failed, 1);
        assert_eq!(unfiltered.regressions, 1, "alpha still regresses");

        // Filtering to "alpha" excludes beta entirely; the failed row
        // is filtered out before the passed gate runs, so
        // skipped_failed=0.
        let filtered = compare_rows(&rows_a, &rows_b, Some("alpha"), None);
        assert_eq!(filtered.skipped_failed, 0);
        assert_eq!(filtered.regressions, 1);
        assert_eq!(filtered.findings.len(), 1);
        assert_eq!(filtered.findings[0].scenario, "alpha");
    }

    /// The substring filter searches the joined "scenario topology
    /// scheduler work_type" string, so a scheduler name uniquely
    /// scopes the comparison even when scenarios and topologies
    /// overlap. Without scheduler in the join string this would
    /// require a less-precise substring (e.g. a scenario name).
    #[test]
    fn compare_rows_filter_substring_matches_scheduler() {
        let mut a1 = cmp_row("test1", "tiny-1llc", true, 10.0, 0);
        a1.scheduler = "scx_alpha".into();
        let mut a2 = cmp_row("test2", "tiny-1llc", true, 10.0, 0);
        a2.scheduler = "scx_beta".into();
        let mut b1 = cmp_row("test1", "tiny-1llc", true, 30.0, 0);
        b1.scheduler = "scx_alpha".into();
        let mut b2 = cmp_row("test2", "tiny-1llc", true, 30.0, 0);
        b2.scheduler = "scx_beta".into();

        let res = compare_rows(&[a1, a2], &[b1, b2], Some("scx_alpha"), None);
        assert_eq!(res.regressions, 1, "only the scx_alpha row compares");
        assert_eq!(res.findings.len(), 1);
        assert_eq!(res.findings[0].scenario, "test1");
        // scx_beta rows are filtered out, not counted as new/removed.
        assert_eq!(res.new_in_b, 0);
        assert_eq!(res.removed_from_a, 0);
    }

    /// `new_in_b` counts B-side rows whose key has no match on the A
    /// side; `removed_from_a` counts the converse. Both are needed so
    /// schema drift between two runs (a renamed scenario, an added
    /// topology preset, a removed work_type) is visible in the
    /// summary instead of silently dropped.
    #[test]
    fn compare_rows_tracks_new_and_removed_rows() {
        // alpha exists in both -> regression.
        // beta exists only in B -> new_in_b=1.
        // gamma exists only in A -> removed_from_a=1.
        let rows_a = vec![
            cmp_row("alpha", "tiny-1llc", true, 10.0, 0),
            cmp_row("gamma", "tiny-1llc", true, 10.0, 0),
        ];
        let rows_b = vec![
            cmp_row("alpha", "tiny-1llc", true, 30.0, 0),
            cmp_row("beta", "tiny-1llc", true, 30.0, 0),
        ];
        let res = compare_rows(&rows_a, &rows_b, None, None);
        assert_eq!(res.regressions, 1, "alpha regresses on worst_spread");
        assert_eq!(res.new_in_b, 1, "beta is new on B side");
        assert_eq!(res.removed_from_a, 1, "gamma is removed on B side");
        assert_eq!(res.skipped_failed, 0);
    }

    /// The filter applies to every counter, including `new_in_b` and
    /// `removed_from_a`. An excluded row never reaches matching, so
    /// it contributes to no counter at all.
    #[test]
    fn compare_rows_filter_applies_to_new_and_removed_counters() {
        let rows_a = vec![
            cmp_row("alpha", "tiny-1llc", true, 10.0, 0),
            cmp_row("gamma", "tiny-1llc", true, 10.0, 0),
        ];
        let rows_b = vec![
            cmp_row("alpha", "tiny-1llc", true, 30.0, 0),
            cmp_row("beta", "tiny-1llc", true, 30.0, 0),
        ];

        // Filter to "alpha" -- beta and gamma are excluded by the
        // substring filter on both passes.
        let res = compare_rows(&rows_a, &rows_b, Some("alpha"), None);
        assert_eq!(res.regressions, 1);
        assert_eq!(res.new_in_b, 0, "beta is filtered out, not new");
        assert_eq!(res.removed_from_a, 0, "gamma is filtered out, not removed");
    }
}
