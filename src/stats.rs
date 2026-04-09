//! Gauntlet analysis and baseline comparison.
//!
//! Collects per-scenario results into a [`polars`] DataFrame for
//! statistical analysis, regression detection, and baseline save/compare
//! workflows.

use std::collections::BTreeMap;

use polars::prelude::*;

use crate::timeline::Timeline;
use crate::vmm::shm_ring;

/// Definition of an extensible metric for the generic comparison pipeline.
///
/// Each entry describes a metric that can be populated via `ext_metrics`
/// on stats types. The pipeline uses `higher_is_worse` to determine
/// regression direction, and `default_abs`/`default_rel` for dual-gate
/// significance thresholds.
pub struct MetricDef {
    pub name: &'static str,
    pub higher_is_worse: bool,
    pub default_abs: f64,
    pub default_rel: f64,
}

/// Registry of extensible metrics processed by the comparison pipeline.
///
/// Typed fields (spread, gap_ms, etc.) remain as primary metrics. Entries
/// here are processed for any metric populated in `ext_metrics`.
///
/// All entries must have `higher_is_worse: true` until
/// `AssertResult::merge` supports per-metric merge direction.
pub static EXTENSIBLE_METRICS: &[MetricDef] = &[MetricDef {
    name: "migration_ratio",
    higher_is_worse: true,
    default_abs: 0.05,
    default_rel: 0.20,
}];

/// Monitor data preserved from a gauntlet VM run for timeline analysis.
#[derive(Debug, Clone)]
pub struct GauntletMonitorData {
    pub summary: crate::monitor::MonitorSummary,
    pub samples: Vec<crate::monitor::MonitorSample>,
    pub stimulus_events: Vec<shm_ring::StimulusEvent>,
}

/// Result from a single gauntlet VM run.
#[derive(Debug, Clone)]
pub struct VmRunResult {
    pub label: String,
    pub passed: bool,
    pub duration_s: f64,
    pub detail: String,
    pub scenario_results: Vec<crate::runner::ScenarioResult>,
    pub monitor_data: Option<GauntletMonitorData>,
}

/// Default work type name for serde deserialization.
pub fn default_work_type() -> String {
    "CpuSpin".to_string()
}

/// Per-scenario result row for gauntlet analysis and baseline comparison.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GauntletRow {
    pub scenario: String,
    pub flags: String,
    pub topology: String,
    #[serde(default = "default_work_type")]
    pub work_type: String,
    pub replica: u32,
    pub passed: bool,
    pub spread: f64,
    pub gap_ms: u64,
    pub migrations: u64,
    #[serde(default)]
    pub migration_ratio: f64,
    // Monitor fields (host-side telemetry from guest memory reads).
    pub imbalance_ratio: f64,
    pub max_dsq_depth: u32,
    pub stall_count: usize,
    pub fallback_count: i64,
    pub keep_last_count: i64,
    // Benchmarking fields.
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
    // Timeline degradation fields.
    pub worst_degradation_op: String,
    pub worst_imbalance_delta: f64,
    pub worst_dsq_delta: f64,
    pub worst_fallback_delta: f64,
    pub worst_keep_last_delta: f64,
    pub degradation_count: u32,
    /// Extensible metrics populated by scenarios and processed by the
    /// generic comparison pipeline. Keyed by metric name (matching
    /// `MetricDef::name` in `EXTENSIBLE_METRICS`).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub ext_metrics: BTreeMap<String, f64>,
}

/// Convert a SidecarResult to a GauntletRow for baseline comparison.
pub fn sidecar_to_row(sc: &crate::test_support::SidecarResult) -> GauntletRow {
    GauntletRow {
        scenario: sc.test_name.clone(),
        flags: String::new(),
        topology: sc.topology.clone(),
        work_type: sc.work_type.clone(),
        replica: 1,
        passed: sc.passed,
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
        worst_degradation_op: String::new(),
        worst_imbalance_delta: 0.0,
        worst_dsq_delta: 0.0,
        worst_fallback_delta: 0.0,
        worst_keep_last_delta: 0.0,
        degradation_count: 0,
        ext_metrics: sc.stats.ext_metrics.clone(),
    }
}

/// Parse a gauntlet label "topology/scenario/flags[/work_type][#replica]" into
/// (topology, scenario, flags, work_type, replica).
fn parse_label(label: &str) -> (&str, &str, &str, &str, u32) {
    // Strip optional "#N" replica suffix.
    let (base, replica) = match label.rfind('#') {
        Some(pos) => {
            let tail = &label[pos + 1..];
            match tail.parse::<u32>() {
                Ok(r) => (&label[..pos], r),
                Err(_) => (label, 1),
            }
        }
        None => (label, 1),
    };
    let mut parts = base.splitn(4, '/');
    let topo = parts.next().unwrap_or("");
    let scenario = parts.next().unwrap_or("");
    let flags = parts.next().unwrap_or("default");
    let work_type = parts.next().unwrap_or("CpuSpin");
    (topo, scenario, flags, work_type, replica)
}

/// Map op_kinds bitmask to the name of the dominant op variant.
fn op_kinds_to_name(op_kinds: u32) -> &'static str {
    // Return the name of the highest-priority op present.
    // Priority: the op most likely to cause observable scheduler changes.
    const NAMES: &[&str] = &[
        "AddCgroup",         // 0
        "RemoveCgroup",      // 1
        "SetCpuset",         // 2
        "ClearCpuset",       // 3
        "SwapCpusets",       // 4
        "Spawn",             // 5
        "StopCgroup",        // 6
        "RandomizeAffinity", // 7
        "SetAffinity",       // 8
        "SpawnHost",         // 9
        "MoveAllTasks",      // 10
        "MoveTasks",         // 11
    ];
    // Pick the first set bit as the representative op.
    for (i, name) in NAMES.iter().enumerate() {
        if op_kinds & (1 << i) != 0 {
            return name;
        }
    }
    "unknown"
}

/// Convert shm_ring::StimulusEvent to timeline::StimulusEvent.
fn shm_stim_to_timeline(events: &[shm_ring::StimulusEvent]) -> Vec<crate::timeline::StimulusEvent> {
    let mut out = vec![crate::timeline::StimulusEvent {
        elapsed_ms: 0,
        label: "ScenarioStart".to_string(),
        op_kind: None,
        detail: None,
        total_iterations: None,
    }];
    for e in events {
        out.push(crate::timeline::StimulusEvent {
            elapsed_ms: e.elapsed_ms as u64,
            label: format!("StepStart[{}]", e.step_index),
            op_kind: Some(op_kinds_to_name(e.op_kinds).to_string()),
            detail: Some(format!("{} ops", e.op_count)),
            total_iterations: Some(e.total_iterations),
        });
    }
    out
}

/// Extract worst degradation per metric from a timeline.
/// Returns (worst_op, imbalance_delta, dsq_delta, fallback_delta, keep_last_delta, count).
fn extract_worst_degradation(timeline: Option<&Timeline>) -> (String, f64, f64, f64, f64, u32) {
    let timeline = match timeline {
        Some(t) => t,
        None => return (String::new(), 0.0, 0.0, 0.0, 0.0, 0),
    };

    let mut worst_op = String::new();
    let mut worst_imb = 0.0f64;
    let mut worst_dsq = 0.0f64;
    let mut worst_fb = 0.0f64;
    let mut worst_kl = 0.0f64;
    let mut worst_max_delta = 0.0f64;
    let mut count = 0u32;

    for (phase, change) in timeline.degradations() {
        count += 1;
        let delta = change.after - change.before;
        let op = phase
            .stimulus
            .as_ref()
            .and_then(|s| s.op_kind.clone())
            .unwrap_or_default();

        match change.metric.as_str() {
            "imbalance" if delta > worst_imb => {
                worst_imb = delta;
            }
            "dsq_depth" if delta > worst_dsq => {
                worst_dsq = delta;
            }
            "fallback" if delta > worst_fb => {
                worst_fb = delta;
            }
            "keep_last" if delta > worst_kl => {
                worst_kl = delta;
            }
            _ => {}
        }

        if delta.abs() > worst_max_delta {
            worst_max_delta = delta.abs();
            worst_op = op;
        }
    }

    (worst_op, worst_imb, worst_dsq, worst_fb, worst_kl, count)
}

/// Extract analysis rows from gauntlet results.
pub fn extract_rows(results: &[VmRunResult]) -> Vec<GauntletRow> {
    let mut rows = Vec::new();
    for r in results {
        let (topo, scenario, flags, work_type, replica) = parse_label(&r.label);
        let stats = r.scenario_results.first().map(|r| &r.stats);
        let summary = r.monitor_data.as_ref().map(|m| &m.summary);

        // Build timeline from monitor samples + stimulus events.
        let timeline = r.monitor_data.as_ref().map(|m| {
            let stim_events: Vec<crate::timeline::StimulusEvent> =
                shm_stim_to_timeline(&m.stimulus_events);
            Timeline::build(&stim_events, &m.samples)
        });

        // Extract worst degradation per metric from timeline.
        let (
            worst_deg_op,
            worst_imb_delta,
            worst_dsq_delta,
            worst_fb_delta,
            worst_kl_delta,
            deg_count,
        ) = extract_worst_degradation(timeline.as_ref());

        rows.push(GauntletRow {
            scenario: scenario.to_string(),
            flags: flags.to_string(),
            topology: topo.to_string(),
            work_type: work_type.to_string(),
            replica,
            passed: r.passed,
            spread: stats.map(|s| s.worst_spread).unwrap_or(0.0),
            gap_ms: stats.map(|s| s.worst_gap_ms).unwrap_or(0),
            migrations: stats.map(|s| s.total_migrations).unwrap_or(0),
            migration_ratio: stats.map(|s| s.worst_migration_ratio).unwrap_or(0.0),
            imbalance_ratio: summary.map(|m| m.max_imbalance_ratio).unwrap_or(0.0),
            max_dsq_depth: summary.map(|m| m.max_local_dsq_depth).unwrap_or(0),
            stall_count: if summary.map(|m| m.stall_detected).unwrap_or(false) {
                1
            } else {
                0
            },
            fallback_count: summary
                .and_then(|m| m.event_deltas.as_ref())
                .map(|e| e.total_fallback)
                .unwrap_or(0),
            keep_last_count: summary
                .and_then(|m| m.event_deltas.as_ref())
                .map(|e| e.total_dispatch_keep_last)
                .unwrap_or(0),
            p99_wake_latency_us: stats.map(|s| s.p99_wake_latency_us).unwrap_or(0.0),
            median_wake_latency_us: stats.map(|s| s.median_wake_latency_us).unwrap_or(0.0),
            wake_latency_cv: stats.map(|s| s.wake_latency_cv).unwrap_or(0.0),
            total_iterations: stats.map(|s| s.total_iterations).unwrap_or(0),
            mean_run_delay_us: stats.map(|s| s.mean_run_delay_us).unwrap_or(0.0),
            worst_run_delay_us: stats.map(|s| s.worst_run_delay_us).unwrap_or(0.0),
            worst_degradation_op: worst_deg_op,
            worst_imbalance_delta: worst_imb_delta,
            worst_dsq_delta,
            worst_fallback_delta: worst_fb_delta,
            worst_keep_last_delta: worst_kl_delta,
            degradation_count: deg_count,
            ext_metrics: stats.map(|s| s.ext_metrics.clone()).unwrap_or_default(),
        });
    }
    rows
}

/// Build a polars DataFrame from gauntlet rows.
fn build_dataframe(rows: &[GauntletRow]) -> PolarsResult<DataFrame> {
    let scenario: Vec<&str> = rows.iter().map(|r| r.scenario.as_str()).collect();
    let flags: Vec<&str> = rows.iter().map(|r| r.flags.as_str()).collect();
    let topology: Vec<&str> = rows.iter().map(|r| r.topology.as_str()).collect();
    let work_type: Vec<&str> = rows.iter().map(|r| r.work_type.as_str()).collect();
    let replica: Vec<u32> = rows.iter().map(|r| r.replica).collect();
    let passed: Vec<bool> = rows.iter().map(|r| r.passed).collect();
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
            col("passed")
                .cast(DataType::UInt32)
                .sum()
                .alias("pass_count"),
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
        let total = totals.get(i).unwrap_or(0);
        let spread = spreads.get(i).unwrap_or(0.0);
        let gap = gaps.get(i).unwrap_or(0.0);
        let mut line = format!(
            "  {:<25} {}/{} passed  avg_spread={:.1}%  avg_gap={:.0}ms",
            name, pass, total, spread, gap
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
fn topo_bucket(topo: &str) -> &'static str {
    // Parse CPU count from topology preset names like "tiny-1llc" (4 CPUs),
    // "medium-4llc" (32 CPUs) etc. Fall back to the name prefix.
    let cpus = match topo {
        "tiny-1llc" | "tiny-2llc" => 4,
        "odd-3llc" => 9,
        "odd-5llc" => 15,
        "odd-7llc" => 14,
        "smt-2llc" => 8,
        "smt-3llc" => 12,
        "medium-4llc" => 32,
        "medium-8llc" => 64,
        "large-4llc" | "large-8llc" => 128,
        "near-max-llc" => 240,
        "max-cpu" => 252,
        _ => return "unknown",
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
            col("passed")
                .cast(DataType::UInt32)
                .sum()
                .alias("pass_count"),
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
        let total = totals.get(i).unwrap_or(0);
        if total == 0 || pass == total {
            continue;
        }
        all_pass = false;
        let sc = scenarios.get(i).unwrap_or("?");
        let fl = flags_col.get(i).unwrap_or("?");
        let tp = topos.get(i).unwrap_or("?");
        let mut line = format!("  {tp}/{sc}/{fl}  {pass}/{total}");
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

/// Analyze gauntlet results and return a formatted report.
pub fn analyze_gauntlet(results: &[VmRunResult]) -> String {
    if results.is_empty() {
        return String::new();
    }
    let rows = extract_rows(results);
    analyze_rows(&rows)
}

// ---------------------------------------------------------------------------
// Baseline serialization and A/B comparison
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert::ScenarioStats;
    use crate::runner::ScenarioResult;

    fn make_result(
        label: &str,
        passed: bool,
        spread: f64,
        gap_ms: u64,
        migrations: u64,
    ) -> VmRunResult {
        let sr = ScenarioResult {
            scenario_name: label.to_string(),
            passed,
            duration_s: 20.0,
            details: vec![],
            stats: ScenarioStats {
                cgroups: vec![],
                total_workers: 4,
                total_cpus: 4,
                total_migrations: migrations,
                worst_spread: spread,
                worst_gap_ms: gap_ms,
                worst_gap_cpu: 0,
                ..Default::default()
            },
        };
        VmRunResult {
            label: label.to_string(),
            passed,
            duration_s: 20.0,
            detail: String::new(),
            scenario_results: vec![sr],
            monitor_data: None,
        }
    }

    #[test]
    fn replicated_cgroup_pass_rate() {
        // 3 replicas of a cgroup, 2 pass 1 fails.
        let results = vec![
            make_result("tiny/a/flags#1", true, 5.0, 50, 10),
            make_result("tiny/a/flags#2", false, 25.0, 3000, 5),
            make_result("tiny/a/flags#3", true, 8.0, 100, 12),
        ];
        let report = analyze_gauntlet(&results);
        assert!(report.contains("Cgroups with <100% pass rate"));
        assert!(report.contains("2/3"));
    }

    #[test]
    fn replicated_all_pass() {
        let results = vec![
            make_result("tiny/a/flags#1", true, 5.0, 50, 10),
            make_result("tiny/a/flags#2", true, 6.0, 60, 11),
            make_result("tiny/a/flags#3", true, 7.0, 55, 9),
        ];
        let report = analyze_gauntlet(&results);
        assert!(report.contains("All cgroups passed across all replicas"));
    }

    #[test]
    fn build_dataframe_basic() {
        let results = vec![
            make_result("tiny/a/flags1", true, 5.0, 50, 10),
            make_result("tiny/b/flags2", false, 20.0, 3000, 5),
        ];
        let rows = extract_rows(&results);
        let df = build_dataframe(&rows).unwrap();
        assert_eq!(df.height(), 2);
        assert_eq!(df.width(), 27);
    }

    #[test]
    fn analyze_empty() {
        let report = analyze_gauntlet(&[]);
        assert!(report.is_empty());
    }

    #[test]
    fn analyze_no_outliers() {
        // All results similar — no outliers expected.
        let results: Vec<VmRunResult> = (0..5)
            .map(|i| make_result(&format!("topo{i}/scenario/flags"), true, 5.0, 50, 10))
            .collect();
        let report = analyze_gauntlet(&results);
        assert!(report.contains("GAUNTLET ANALYSIS"));
        assert!(report.contains("No outliers detected"));
    }

    #[test]
    fn analyze_with_outlier() {
        // Many normal results to anchor the mean low; a few extreme outliers
        // to exceed the 2-sigma threshold.
        let mut results: Vec<VmRunResult> = (0..20)
            .map(|i| make_result(&format!("topo{}/normal/flags", i % 5), true, 5.0, 50, 10))
            .collect();
        results.push(make_result("topo0/outlier/flags", true, 200.0, 50, 10));
        results.push(make_result("topo1/outlier/flags", true, 195.0, 50, 10));
        results.push(make_result("topo2/outlier/flags", true, 190.0, 50, 10));
        let report = analyze_gauntlet(&results);
        assert!(report.contains("GAUNTLET ANALYSIS"));
        assert!(report.contains("Outliers detected"), "report: {report}");
        assert!(report.contains("outlier"));
        assert!(report.contains("spread"));
    }

    #[test]
    fn analyze_dimension_summaries() {
        let results = vec![
            make_result("tiny/a/f1", true, 5.0, 50, 10),
            make_result("large/a/f1", false, 25.0, 3000, 5),
            make_result("tiny/b/f2", true, 3.0, 30, 8),
            make_result("large/b/f2", true, 8.0, 100, 12),
        ];
        let report = analyze_gauntlet(&results);
        assert!(report.contains("By scenario"));
        assert!(report.contains("By flags"));
        assert!(report.contains("By topology"));
    }

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
        assert_eq!(topo_bucket("smt-2llc"), "<=8cpu");
    }

    #[test]
    fn topo_bucket_small() {
        assert_eq!(topo_bucket("odd-3llc"), "9-32cpu");
        assert_eq!(topo_bucket("odd-5llc"), "9-32cpu");
        assert_eq!(topo_bucket("odd-7llc"), "9-32cpu");
        assert_eq!(topo_bucket("smt-3llc"), "9-32cpu");
        assert_eq!(topo_bucket("medium-4llc"), "9-32cpu");
    }

    #[test]
    fn topo_bucket_medium() {
        assert_eq!(topo_bucket("medium-8llc"), "33-128cpu");
        assert_eq!(topo_bucket("large-4llc"), "33-128cpu");
        assert_eq!(topo_bucket("large-8llc"), "33-128cpu");
    }

    #[test]
    fn topo_bucket_large() {
        assert_eq!(topo_bucket("near-max-llc"), ">128cpu");
        assert_eq!(topo_bucket("max-cpu"), ">128cpu");
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

    // -- op_kinds_to_name tests --

    #[test]
    fn op_kinds_to_name_single_bits() {
        assert_eq!(op_kinds_to_name(1 << 0), "AddCgroup");
        assert_eq!(op_kinds_to_name(1 << 1), "RemoveCgroup");
        assert_eq!(op_kinds_to_name(1 << 2), "SetCpuset");
        assert_eq!(op_kinds_to_name(1 << 3), "ClearCpuset");
        assert_eq!(op_kinds_to_name(1 << 4), "SwapCpusets");
        assert_eq!(op_kinds_to_name(1 << 5), "Spawn");
        assert_eq!(op_kinds_to_name(1 << 6), "StopCgroup");
        assert_eq!(op_kinds_to_name(1 << 7), "RandomizeAffinity");
        assert_eq!(op_kinds_to_name(1 << 8), "SetAffinity");
    }

    #[test]
    fn op_kinds_to_name_multiple_returns_lowest_set() {
        // With bits 2 and 5 set, returns the first (lowest) set bit match.
        assert_eq!(op_kinds_to_name((1 << 2) | (1 << 5)), "SetCpuset");
    }

    #[test]
    fn op_kinds_to_name_zero() {
        assert_eq!(op_kinds_to_name(0), "unknown");
    }

    #[test]
    fn op_kinds_to_name_high_bit() {
        // Bit 9+ is beyond the NAMES array.
        assert_eq!(op_kinds_to_name(1 << 15), "unknown");
    }

    // -- extract_worst_degradation tests --

    #[test]
    fn extract_worst_degradation_none() {
        let (op, imb, dsq, fb, kl, count) = extract_worst_degradation(None);
        assert!(op.is_empty());
        assert_eq!(imb, 0.0);
        assert_eq!(dsq, 0.0);
        assert_eq!(fb, 0.0);
        assert_eq!(kl, 0.0);
        assert_eq!(count, 0);
    }

    #[test]
    fn extract_worst_degradation_no_degradations() {
        use crate::timeline::Timeline;
        let t = Timeline { phases: vec![] };
        let (op, imb, dsq, fb, kl, count) = extract_worst_degradation(Some(&t));
        assert!(op.is_empty());
        assert_eq!(imb, 0.0);
        assert_eq!(dsq, 0.0);
        assert_eq!(fb, 0.0);
        assert_eq!(kl, 0.0);
        assert_eq!(count, 0);
    }

    // -- shm_stim_to_timeline tests --

    #[test]
    fn shm_stim_to_timeline_empty() {
        let result = shm_stim_to_timeline(&[]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].label, "ScenarioStart");
    }

    #[test]
    fn shm_stim_to_timeline_with_events() {
        let events = vec![crate::vmm::shm_ring::StimulusEvent {
            elapsed_ms: 1000,
            step_index: 0,
            op_count: 3,
            op_kinds: 1 << 2,
            cgroup_count: 2,
            worker_count: 4,
            total_iterations: 50000,
        }];
        let result = shm_stim_to_timeline(&events);
        assert_eq!(result.len(), 2);
        assert_eq!(result[1].label, "StepStart[0]");
        assert_eq!(result[1].op_kind.as_deref(), Some("SetCpuset"));
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
        // borrow on medium-4llc (9-32cpu bucket) with imbalance_delta=7.0.
        // The flag x topo table should show +3.0 for <=8cpu and +7.0 for 9-32cpu.
        let mut r1 = make_row("a", "borrow", "tiny-1llc", true, 5.0);
        r1.worst_degradation_op = "SetCpuset".into();
        r1.worst_imbalance_delta = 3.0;
        r1.degradation_count = 1;
        let mut r2 = make_row("a", "borrow", "medium-4llc", true, 5.0);
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

    // -- default_work_type test --

    #[test]
    fn default_work_type_is_cpuspin() {
        assert_eq!(default_work_type(), "CpuSpin");
    }

    // -- sidecar_to_row tests --

    #[test]
    fn sidecar_to_row_basic() {
        use crate::monitor;
        use crate::test_support;
        let sc = test_support::SidecarResult {
            test_name: "my_test".to_string(),
            topology: "2s4c2t".to_string(),
            scheduler: "scx_mitosis".to_string(),
            passed: true,
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
                }),
                schedstat_deltas: None,
                prog_stats_deltas: None,
            }),
            stimulus_events: vec![],
            work_type: "CpuSpin".to_string(),
            verifier_stats: vec![],
            kvm_stats: None,
        };
        let row = sidecar_to_row(&sc);
        assert_eq!(row.scenario, "my_test");
        assert_eq!(row.topology, "2s4c2t");
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
            topology: "1s2c1t".to_string(),
            scheduler: "eevdf".to_string(),
            passed: false,
            stats: Default::default(),
            monitor: None,
            stimulus_events: vec![],
            work_type: "CpuSpin".to_string(),
            verifier_stats: vec![],
            kvm_stats: None,
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
            topology: "1s1c1t".to_string(),
            scheduler: "test".to_string(),
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
            }),
            stimulus_events: vec![],
            work_type: "CpuSpin".to_string(),
            verifier_stats: vec![],
            kvm_stats: None,
        };
        let row = sidecar_to_row(&sc);
        assert_eq!(row.stall_count, 0);
        assert_eq!(row.fallback_count, 0);
        assert_eq!(row.keep_last_count, 0);
    }
}
