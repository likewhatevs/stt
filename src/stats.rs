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
/// pointers are not serializable. A future `Deserialize` impl
/// would need callers to re-hydrate the accessor by looking up
/// `name` via [`metric_def`] — the static [`METRICS`] table is
/// the authoritative source of the function identity. No such
/// impl exists today; the note is a forward-conditional so that
/// if one is added, the migration path is spelled out rather
/// than reinvented per site.
///
/// # Registered vs unregistered metrics
///
/// The static [`METRICS`] registry is the "core metric" set with
/// hand-authored accessors, hand-tuned dual-gate thresholds
/// (`default_abs` / `default_rel`), and display units. Each
/// registered `MetricDef.accessor` reads a typed field on
/// `GauntletRow` directly (e.g. `r.spread`, `r.gap_ms`).
///
/// Metrics that fall OUTSIDE this registry are carried on
/// `GauntletRow.ext_metrics: BTreeMap<String, f64>`. Registered
/// metrics never flow through `ext_metrics`; unregistered metrics
/// never flow through the typed fields. [`MetricDef::read`] and
/// [`read_metric`] check the registered-field accessor first and
/// fall back to an `ext_metrics.get(name)` lookup — a name that
/// matches neither returns `None`. Consumers that want to
/// distinguish "registered-but-null" from "unregistered-and-
/// absent" must inspect the registry directly rather than rely
/// on the fallback.
///
/// # `#[non_exhaustive]` migration note
///
/// Downstream code that pattern-matches an instance of `MetricDef`
/// must end the match with `..` so a future field addition does
/// not become a breaking change. Prefer reading values through
/// the static [`METRICS`] registry and [`metric_def`] lookup
/// rather than constructing `MetricDef` values by hand.
#[derive(Debug, Clone, serde::Serialize)]
#[non_exhaustive]
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
///
/// # Metric-name triples (registry / field / DataFrame column)
///
/// Each metric is referenced by three names across the pipeline.
/// The registry name is the stable surface — sidecars, CI gates,
/// and `cargo ktstr stats compare` output all quote it verbatim —
/// and cannot be renamed without silently invalidating downstream
/// consumers. The field name on [`GauntletRow`] and the polars
/// DataFrame column name are internal; they are kept terse and
/// match each other, but diverge from the registry name where
/// the domain-level wording adds context (`worst_*`, `total_*`,
/// `max_*`) that would be noise on an already-qualified field.
/// Eleven divergent triples:
///
/// | Registry (`MetricDef.name`) | `GauntletRow` field | DataFrame column |
/// |---|---|---|
/// | `worst_spread` | `spread` | `spread` |
/// | `worst_gap_ms` | `gap_ms` | `gap_ms` |
/// | `total_migrations` | `migrations` | `migrations` |
/// | `worst_migration_ratio` | `migration_ratio` | `migration_ratio` |
/// | `max_imbalance_ratio` | `imbalance_ratio` | `imbalance` |
/// | `max_dsq_depth` | `max_dsq_depth` | `dsq_depth` |
/// | `stall_count` | `stall_count` | `stalls` |
/// | `total_fallback` | `fallback_count` | `fallback` |
/// | `total_keep_last` | `keep_last_count` | `keep_last` |
/// | `worst_page_locality` | `page_locality` | `page_locality` |
/// | `worst_cross_node_migration_ratio` | `cross_node_migration_ratio` | `cross_node_migration_ratio` |
///
/// Metrics with matching names (`worst_p99_wake_latency_us`,
/// `worst_median_wake_latency_us`, `worst_wake_latency_cv`,
/// `total_iterations`, `worst_mean_run_delay_us`,
/// `worst_run_delay_us`) are not listed — the registry name,
/// field, and DataFrame column are all identical, so there is no
/// translation to document.
///
/// Consumers that cross the registry / DataFrame boundary should
/// go through [`MetricDef::read`] / the accessor closure rather
/// than hand-translating by string. The four-name mapping for
/// `worst_spread` specifically is documented in detail on the
/// [`GauntletRow::spread`] field (adds the
/// [`ScenarioStats::worst_spread`](crate::assert::ScenarioStats::worst_spread)
/// upstream source as a fourth name).
pub static METRICS: &[MetricDef] = &[
    MetricDef {
        // `"worst_spread"` is the wire/surface name — emitted in
        // sidecars, referenced by CI gates, and printed by
        // `cargo ktstr stats compare`. Internally the field on
        // `GauntletRow` is named `spread` and the polars DataFrame
        // column keeps that shorter name; see the doc on
        // `GauntletRow.spread` for the rationale (rename-of-
        // registry-name is not safe because existing gate configs
        // match this string by value).
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
        name: "worst_p99_wake_latency_us",
        polarity: crate::test_support::Polarity::LowerBetter,
        default_abs: 50.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |r| Some(r.worst_p99_wake_latency_us),
    },
    MetricDef {
        name: "worst_median_wake_latency_us",
        polarity: crate::test_support::Polarity::LowerBetter,
        default_abs: 20.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |r| Some(r.worst_median_wake_latency_us),
    },
    MetricDef {
        name: "worst_wake_latency_cv",
        polarity: crate::test_support::Polarity::LowerBetter,
        default_abs: 0.10,
        default_rel: 0.25,
        display_unit: "",
        accessor: |r| Some(r.worst_wake_latency_cv),
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
        name: "worst_mean_run_delay_us",
        polarity: crate::test_support::Polarity::LowerBetter,
        default_abs: 50.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |r| Some(r.worst_mean_run_delay_us),
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
/// # NaN-ambiguity on direct f64 fields
///
/// All direct f64 fields on this struct are sanitized via
/// `finite_or_zero` at [`sidecar_to_row`] ingress. A `0.0` on any
/// direct f64 field may represent either a genuine zero measurement
/// or a sanitized non-finite upstream value (NaN / ±Infinity). See
/// [`sidecar_to_row`]'s NaN-ambiguity doc for the full policy;
/// `tracing::warn!` is the disambiguation channel — the sanitizer
/// warns on every non-finite it rewrites to zero, so the log
/// timeline tells you which run's zeroes were real. Consumers that
/// cannot accept the ambiguity should prefer metric paths that
/// flow through `ext_metrics` (which keep non-finite values as
/// explicit `Option::None` rather than collapsing to zero).
///
/// # `#[non_exhaustive]` migration note
///
/// Downstream code that pattern-matches a `GauntletRow` must end
/// the match with `..`; future fields added alongside new metrics
/// otherwise break every matcher. Prefer reading values via
/// [`MetricDef::read`] / the registry — the point of the
/// registry indirection is that new metrics do not touch
/// existing readers.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct GauntletRow {
    pub scenario: String,
    pub topology: String,
    pub work_type: String,
    /// Scheduler binary name carried from the source sidecar
    /// (`SidecarResult::scheduler`). Surfaced through the substring
    /// filter in [`compare_rows`] so users can narrow A/B comparisons
    /// by scheduler name.
    pub scheduler: String,
    /// Active scheduler flags carried from
    /// `SidecarResult::active_flags`. Previously this field did not
    /// exist on the row; every A/B comparison therefore ignored
    /// flag-profile identity and treated two rows whose only
    /// difference was the flag set as the same row (causing
    /// same-key collisions in `compare_rows` and pointer-latching
    /// on whichever sidecar happened to be scanned first).
    /// Carrying the full flag list here keeps the (scenario,
    /// topology, work_type, flags) identity tuple unambiguous and
    /// lets the substring filter match on flag names too.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub flags: Vec<String>,
    pub passed: bool,
    /// True when the run was skipped (topology mismatch, missing
    /// resource). `passed` stays `true` for gate-compat; `skipped`
    /// lets stats tooling exclude these from pass counts so skipped
    /// runs don't inflate the apparent pass rate.
    pub skipped: bool,
    /// Worst-case per-cgroup spread across the run. Four names
    /// describe the same quantity across the pipeline:
    /// - [`ScenarioStats::worst_spread`](crate::assert::ScenarioStats::worst_spread)
    ///   — the upstream source. `sidecar_to_row` reads it and
    ///   writes the value into this field via `finite_or_zero`.
    /// - `GauntletRow.spread` (this field) — the Rust-side
    ///   struct access path inside the comparison pipeline.
    /// - `MetricDef.name == "worst_spread"` — the [`METRICS`]
    ///   registry key, which is the domain-level name that appears
    ///   in sidecars, CI gates, and `cargo ktstr stats compare`
    ///   output.
    /// - DataFrame column `"spread"` — the polars column name used
    ///   when the rows are projected into a DataFrame for group /
    ///   aggregate operations.
    ///
    /// The registry name is not renamed to match the field name
    /// because existing sidecars and CI regression gates reference
    /// `"worst_spread"` by string and a rename would silently
    /// invalidate them. The DataFrame column stays `"spread"` for
    /// terseness and to match the field; consumers that cross
    /// the registry / DataFrame boundary translate via
    /// [`MetricDef::read`] rather than by string comparison.
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
    pub worst_p99_wake_latency_us: f64,
    pub worst_median_wake_latency_us: f64,
    pub worst_wake_latency_cv: f64,
    pub total_iterations: u64,
    pub worst_mean_run_delay_us: f64,
    pub worst_run_delay_us: f64,
    // NUMA fields.
    pub page_locality: f64,
    pub cross_node_migration_ratio: f64,
    /// Extensible metrics populated by scenarios and processed by the
    /// comparison pipeline. Keyed by metric name; looked up via
    /// [`metric_def`] when a matching entry exists in [`METRICS`].
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub ext_metrics: BTreeMap<String, f64>,
}

/// Convert a SidecarResult to a GauntletRow for run-to-run comparison.
///
/// Non-finite f64 values (NaN, ±Infinity) are sanitized to 0.0 with a
/// warn before they reach the row. `serde_json::to_string` rejects
/// non-finite, so a single poisoned metric would otherwise halt every
/// downstream JSON write. Sanitizing at the ingress boundary keeps the
/// serializer happy without silencing the upstream data quality issue.
///
/// # NaN → 0.0 ambiguity for zero-meaningful metrics
///
/// The 0.0 substitution is indistinguishable from a legitimate 0.0
/// measurement for metrics whose natural zero carries its own signal.
/// Three fields are especially affected — note in-tree producers
/// already guard the typical divide-by-zero path (`assert.rs` emits
/// `0.0` for migration_ratio when `total_iters == 0` and `1.0` for
/// page_locality when `total == 0`), so a NaN reaching this boundary
/// indicates an upstream producer outside those guards (e.g. an
/// external `ext_metrics` contributor, or a schedstat arithmetic
/// edge that slipped past a guard):
///
/// - `migration_ratio`: lower-better. A real 0.0 means "no task was
///   migrated" (ideal locality). A sanitized NaN collapses to the
///   same value and reads as *falsely good* — a downstream regression
///   gate sees "perfect locality" where the truth is "no data".
/// - `page_locality`: higher-better. A real 0.0 means "no local-node
///   accesses". A sanitized NaN collapses to the same value and
///   reads as *falsely bad* — a downstream regression gate sees
///   "everything cross-node" where the truth is "no data". The
///   polarity is opposite to `migration_ratio`: the two failure
///   modes push the comparison in opposite directions.
/// - `worst_wake_latency_cv`: lower-better. A real 0.0 means
///   "wake-latency samples were perfectly uniform" (ideal jitter).
///   A sanitized NaN collapses to the same value and reads as
///   *falsely good* — same direction as `migration_ratio`.
///
/// The accompanying `tracing::warn!` is the only signal that
/// separates a sanitized NaN from a real 0.0; downstream aggregation
/// by value alone cannot distinguish them.
pub fn sidecar_to_row(sc: &crate::test_support::SidecarResult) -> GauntletRow {
    // Local closure so the warn can carry the scenario name as
    // context — keyed by field so the operator can pinpoint which
    // metric produced the bad value.
    let finite_or_zero = |field: &str, v: f64| -> f64 {
        if v.is_finite() {
            v
        } else {
            tracing::warn!(
                test = %sc.test_name,
                field,
                value = v,
                "non-finite f64 in GauntletRow field; substituting 0.0",
            );
            0.0
        }
    };

    GauntletRow {
        scenario: sc.test_name.clone(),
        topology: sc.topology.clone(),
        work_type: sc.work_type.clone(),
        scheduler: sc.scheduler.clone(),
        flags: sc.active_flags.clone(),
        passed: sc.passed,
        skipped: sc.skipped,
        spread: finite_or_zero("spread", sc.stats.worst_spread),
        gap_ms: sc.stats.worst_gap_ms,
        migrations: sc.stats.total_migrations,
        migration_ratio: finite_or_zero("migration_ratio", sc.stats.worst_migration_ratio),
        imbalance_ratio: finite_or_zero(
            "imbalance_ratio",
            sc.monitor
                .as_ref()
                .map(|m| m.max_imbalance_ratio)
                .unwrap_or(0.0),
        ),
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
        worst_p99_wake_latency_us: finite_or_zero(
            "worst_p99_wake_latency_us",
            sc.stats.worst_p99_wake_latency_us,
        ),
        worst_median_wake_latency_us: finite_or_zero(
            "worst_median_wake_latency_us",
            sc.stats.worst_median_wake_latency_us,
        ),
        worst_wake_latency_cv: finite_or_zero(
            "worst_wake_latency_cv",
            sc.stats.worst_wake_latency_cv,
        ),
        total_iterations: sc.stats.total_iterations,
        worst_mean_run_delay_us: finite_or_zero(
            "worst_mean_run_delay_us",
            sc.stats.worst_mean_run_delay_us,
        ),
        worst_run_delay_us: finite_or_zero("worst_run_delay_us", sc.stats.worst_run_delay_us),
        page_locality: finite_or_zero("page_locality", sc.stats.worst_page_locality),
        cross_node_migration_ratio: finite_or_zero(
            "cross_node_migration_ratio",
            sc.stats.worst_cross_node_migration_ratio,
        ),
        // Non-finite entries would also break `serde_json::to_string`,
        // but the map shape makes "substitute 0.0" ambiguous (the entry
        // might legitimately be 0.0 for a different scenario). Drop the
        // entry entirely so the non-finite value can't be confused with
        // a real zero datapoint.
        //
        // Also drop the walk-depth truncation sentinel
        // [`crate::test_support::WALK_TRUNCATION_SENTINEL_NAME`]:
        // it is diagnostic metadata from the JSON-walker depth cap,
        // not a scenario metric, and must not participate in A/B
        // comparison output.
        ext_metrics: sc
            .stats
            .ext_metrics
            .iter()
            .filter_map(|(k, &v)| {
                if k == crate::test_support::WALK_TRUNCATION_SENTINEL_NAME {
                    return None;
                }
                if v.is_finite() {
                    Some((k.clone(), v))
                } else {
                    tracing::warn!(
                        test = %sc.test_name,
                        metric = %k,
                        value = v,
                        "dropping non-finite ext_metric; serde_json rejects NaN/Infinity",
                    );
                    None
                }
            })
            .collect(),
    }
}

/// Build a polars DataFrame from gauntlet rows.
fn build_dataframe(rows: &[GauntletRow]) -> PolarsResult<DataFrame> {
    let scenario: Vec<&str> = rows.iter().map(|r| r.scenario.as_str()).collect();
    let topology: Vec<&str> = rows.iter().map(|r| r.topology.as_str()).collect();
    let work_type: Vec<&str> = rows.iter().map(|r| r.work_type.as_str()).collect();
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
    let p99_wake_lat: Vec<f64> = rows.iter().map(|r| r.worst_p99_wake_latency_us).collect();
    let median_wake_lat: Vec<f64> = rows.iter().map(|r| r.worst_median_wake_latency_us).collect();
    let wake_cv: Vec<f64> = rows.iter().map(|r| r.worst_wake_latency_cv).collect();
    let total_iters: Vec<f64> = rows.iter().map(|r| r.total_iterations as f64).collect();
    let mean_run_delay: Vec<f64> = rows.iter().map(|r| r.worst_mean_run_delay_us).collect();
    let worst_run_delay: Vec<f64> = rows.iter().map(|r| r.worst_run_delay_us).collect();
    let page_locality: Vec<f64> = rows.iter().map(|r| r.page_locality).collect();
    let cross_node_mig: Vec<f64> = rows.iter().map(|r| r.cross_node_migration_ratio).collect();

    df!(
        "scenario" => &scenario,
        "topology" => &topology,
        "work_type" => &work_type,
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
        "worst_p99_wake_latency_us" => &p99_wake_lat,
        "worst_median_wake_latency_us" => &median_wake_lat,
        "worst_wake_latency_cv" => &wake_cv,
        "total_iterations" => &total_iters,
        "worst_mean_run_delay_us" => &mean_run_delay,
        "worst_run_delay_us" => &worst_run_delay,
        "page_locality" => &page_locality,
        "cross_node_migration_ratio" => &cross_node_mig,
    )
}

/// Detected outlier: a scenario with an anomalous stat.
struct Outlier {
    scenario: String,
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
            "{}: {} {:.1} (overall avg {:.1}, +{:.1}\u{03c3})",
            self.scenario, self.metric, self.value, self.overall_mean, self.sigma
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
        "worst_p99_wake_latency_us",
        "worst_wake_latency_cv",
        "worst_mean_run_delay_us",
        "worst_run_delay_us",
    ];
    let mut outliers = Vec::new();

    for &metric in metrics {
        let (overall_mean, overall_std) = col_mean_std(df, metric);
        if overall_std < f64::EPSILON {
            continue;
        }
        let threshold = overall_mean + 2.0 * overall_std;

        // Group by scenario, compute mean of metric across topologies.
        let grouped = df
            .clone()
            .lazy()
            .group_by([col("scenario")])
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
        let means = col_f64(&grouped, "metric_mean");

        let (scenarios, means) = match (scenarios, means) {
            (Some(s), Some(m)) => (s, m),
            _ => continue,
        };

        for i in 0..grouped.height() {
            let mean_val = means.get(i).unwrap_or(0.0);
            if mean_val <= threshold {
                continue;
            }
            let sigma = (mean_val - overall_mean) / overall_std;
            let sc = scenarios.get(i).unwrap_or("");

            // Find worst topologies for this scenario.
            let worst = find_worst_topos(df, sc, metric, threshold);

            outliers.push(Outlier {
                scenario: sc.to_string(),
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

/// Find topology names where a scenario exceeds the threshold.
fn find_worst_topos(
    df: &DataFrame,
    scenario: &str,
    metric: &str,
    threshold: f64,
) -> Vec<String> {
    let filtered = df
        .clone()
        .lazy()
        .filter(
            col("scenario")
                .eq(lit(scenario))
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

    report.push_str("\nBy topology:\n");
    report.push_str(&format_dimension_summary(&df, "topology"));

    let has_work_types = col_str(&df, "work_type")
        .map(|ca| ca.n_unique().unwrap_or(1) > 1)
        .unwrap_or(false);
    if has_work_types {
        report.push_str("\nBy work_type:\n");
        report.push_str(&format_dimension_summary(&df, "work_type"));
    }

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

    let mut table = crate::cli::new_table();
    table.set_header(vec!["RUN", "TESTS", "DATE"]);
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
        table.add_row(vec![key_str.to_string(), count.to_string(), date]);
    }
    println!("{table}");
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
        // Identity key includes `flags` so two rows that share
        // (scenario, topology, work_type) but run under different
        // flag sets do not collide into the same A/B pair. Earlier
        // the key was a 3-tuple, so a scheduler with N flag profiles
        // produced N rows per (scenario, topology, work_type) and
        // compare_rows would pick arbitrarily whichever rows_a entry
        // happened to be first — making regression math match a
        // baseline against an unrelated flag profile.
        let key_b = (
            &row_b.scenario,
            &row_b.topology,
            &row_b.work_type,
            &row_b.flags,
        );
        if let Some(f) = filter {
            // Include `flags` in the filterable join so the substring
            // filter can narrow by flag name (e.g. `-E llc`).
            let joined = format!(
                "{} {} {} {} {}",
                row_b.scenario,
                row_b.topology,
                row_b.scheduler,
                row_b.work_type,
                row_b.flags.join(","),
            );
            if !joined.contains(f) {
                continue;
            }
        }
        let row_a = rows_a.iter().find(|r| {
            (&r.scenario, &r.topology, &r.work_type, &r.flags) == key_b
        });
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
        // Same 4-tuple identity key as the first pass — see that
        // loop's comment for the flag-profile collision rationale.
        let key_a = (
            &row_a.scenario,
            &row_a.topology,
            &row_a.work_type,
            &row_a.flags,
        );
        if let Some(f) = filter {
            let joined = format!(
                "{} {} {} {} {}",
                row_a.scenario,
                row_a.topology,
                row_a.scheduler,
                row_a.work_type,
                row_a.flags.join(","),
            );
            if !joined.contains(f) {
                continue;
            }
        }
        let exists_in_b = rows_b.iter().any(|r| {
            (&r.scenario, &r.topology, &r.work_type, &r.flags) == key_a
        });
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
    dir: Option<&std::path::Path>,
) -> anyhow::Result<i32> {
    // `--dir` overrides the default runs root. Earlier versions of
    // this function accepted the flag through the CLI but never
    // threaded it through to the sidecar lookup, so the value was
    // silently ignored and every comparison ran against
    // `runs_root()` regardless — the user could see their runs via
    // `cargo ktstr stats list --dir X` but `compare` quietly looked
    // in the default location. Accepting `Option<&Path>` here keeps
    // `--dir` load-bearing.
    let root: std::path::PathBuf = match dir {
        Some(d) => d.to_path_buf(),
        None => crate::test_support::runs_root(),
    };
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

    use comfy_table::{Cell, Color};
    let mut table = crate::cli::new_table();
    table.set_header(vec!["TEST", "METRIC", a, b, "DELTA", "VERDICT"]);
    for f in &report.findings {
        let (verdict_text, verdict_color) = if f.is_regression {
            ("REGRESSION", Color::Red)
        } else {
            ("improvement", Color::Green)
        };
        let label = format!("{}/{}/{}", f.scenario, f.topology, f.work_type);
        table.add_row(vec![
            Cell::new(label),
            Cell::new(f.metric.name),
            Cell::new(format!("{:.2}", f.val_a)),
            Cell::new(format!("{:.2}", f.val_b)),
            Cell::new(format!("{:+.2}{}", f.delta, f.metric.display_unit)),
            Cell::new(verdict_text).fg(verdict_color),
        ]);
    }
    println!("{table}");

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

    // Host-context delta. Static fields (uname triple, CPU
    // identity, total memory, hugepage size, NUMA count) are
    // memoized once per process in [`host_context`]'s
    // `STATIC_HOST_INFO`, so every sidecar in a run carries
    // identical values for them. Dynamic fields (sched_tunables,
    // hugepages_{total,free}, thp_enabled, thp_defrag, kernel_cmdline)
    // are re-read on every `collect_host_context` call, so an
    // operator who flips a sysctl or reserves hugepages
    // mid-run will see drift across sidecars within the same
    // run. Picking the first `Some(host)` we encounter is a
    // representative baseline, not a replay of every sample.
    // For full timeseries, inspect individual sidecar JSON files.
    let host_a = sidecars_a.iter().find_map(|s| s.host.as_ref());
    let host_b = sidecars_b.iter().find_map(|s| s.host.as_ref());
    print!("{}", format_host_delta(host_a, host_b, a, b));

    Ok(if report.regressions > 0 { 1 } else { 0 })
}

/// Render the host-context delta section of `stats compare --runs`
/// as a block of text ready to `print!`. Extracted as a pure
/// function of `(Option<&HostContext>, Option<&HostContext>, &str,
/// &str)` so the five match arms can be unit-tested without
/// fixturing a real run directory.
///
/// The returned string is either empty (when both sides have no
/// host data — nothing to print) or ends with a newline so callers
/// can chain further output. Single-side cases print a clear
/// "captured in X only, delta unavailable" message rather than
/// silently suppressing the section — a mixed-tooling-version run
/// comparison should surface the asymmetry.
pub(crate) fn format_host_delta(
    host_a: Option<&crate::host_context::HostContext>,
    host_b: Option<&crate::host_context::HostContext>,
    a: &str,
    b: &str,
) -> String {
    match (host_a, host_b) {
        (Some(ha), Some(hb)) => {
            let delta = ha.diff(hb);
            if delta.is_empty() {
                format!("\nhost: identical between '{a}' and '{b}'\n")
            } else {
                format!("\nhost delta ('{a}' → '{b}'):\n{delta}")
            }
        }
        (Some(_), None) => {
            format!("\nhost: captured in '{a}' only, delta unavailable\n")
        }
        (None, Some(_)) => {
            format!("\nhost: captured in '{b}' only, delta unavailable\n")
        }
        (None, None) => String::new(),
    }
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

    fn make_row(scenario: &str, topo: &str, passed: bool, spread: f64) -> GauntletRow {
        GauntletRow {
            scenario: scenario.into(),
            topology: topo.into(),
            work_type: "CpuSpin".into(),
            scheduler: String::new(),
            flags: Vec::new(),
            skipped: false,
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
            worst_p99_wake_latency_us: 0.0,
            worst_median_wake_latency_us: 0.0,
            worst_wake_latency_cv: 0.0,
            total_iterations: 0,
            worst_mean_run_delay_us: 0.0,
            worst_run_delay_us: 0.0,
            page_locality: 0.0,
            cross_node_migration_ratio: 0.0,
            ext_metrics: BTreeMap::new(),
        }
    }

    // -- format_dimension_summary tests --

    #[test]
    fn format_dimension_summary_computed_values() {
        // Two scenarios: "fast" with spread=4.0, gap=40, and "slow" with spread=20.0, gap=200.
        // Each has 1 row. format_dimension_summary sorts by avg_spread descending.
        let mut r1 = make_row("slow", "tiny-1llc", false, 20.0);
        r1.gap_ms = 200;
        r1.imbalance_ratio = 2.5; // > 1.0, should show imbal=2.5
        r1.max_dsq_depth = 8; // > 0, should show dsq=8
        r1.stall_count = 2; // > 0, should show stalls=2
        r1.fallback_count = 15; // > 0, should show fallback=15
        let r2 = make_row("fast", "tiny-1llc", true, 4.0);
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
            make_row("a", "t1", true, 5.0),
            make_row("a", "t1", true, 6.0),
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
            make_row("a", "t1", true, 5.0),
            make_row("b", "t2", true, 8.0),
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
            ..test_support::SidecarResult::test_fixture()
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
    }

    #[test]
    fn sidecar_to_row_no_monitor() {
        use crate::test_support;
        let sc = test_support::SidecarResult {
            test_name: "eevdf_test".to_string(),
            topology: "1n1l2c1t".to_string(),
            passed: false,
            ..test_support::SidecarResult::test_fixture()
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
            ..test_support::SidecarResult::test_fixture()
        };
        let row = sidecar_to_row(&sc);
        assert_eq!(row.stall_count, 0);
        assert_eq!(row.fallback_count, 0);
        assert_eq!(row.keep_last_count, 0);
    }

    /// Drive every direct f64 field on [`GauntletRow`] through
    /// `finite_or_zero` with `non_finite` planted in the source
    /// [`SidecarResult`], then assert each lands as 0.0 on the row.
    ///
    /// Covers all ten `finite_or_zero` call sites in `sidecar_to_row`:
    /// nine fields drawn from [`ScenarioStats`] plus `imbalance_ratio`
    /// which is read from [`MonitorSummary`]. A missed call site would
    /// leave one of the asserts comparing the non-finite input to 0.0
    /// (NaN != 0.0, ±Infinity != 0.0) and fail the test.
    fn assert_all_direct_f64_fields_sanitized(non_finite: f64) {
        use crate::assert::ScenarioStats;
        use crate::monitor::MonitorSummary;
        use crate::test_support;
        let sc = test_support::SidecarResult {
            stats: ScenarioStats {
                worst_spread: non_finite,
                worst_migration_ratio: non_finite,
                worst_p99_wake_latency_us: non_finite,
                worst_median_wake_latency_us: non_finite,
                worst_wake_latency_cv: non_finite,
                worst_mean_run_delay_us: non_finite,
                worst_run_delay_us: non_finite,
                worst_page_locality: non_finite,
                worst_cross_node_migration_ratio: non_finite,
                ..Default::default()
            },
            monitor: Some(MonitorSummary {
                max_imbalance_ratio: non_finite,
                ..Default::default()
            }),
            ..test_support::SidecarResult::test_fixture()
        };
        let row = sidecar_to_row(&sc);
        for (name, val) in [
            ("spread", row.spread),
            ("migration_ratio", row.migration_ratio),
            ("imbalance_ratio", row.imbalance_ratio),
            ("worst_p99_wake_latency_us", row.worst_p99_wake_latency_us),
            (
                "worst_median_wake_latency_us",
                row.worst_median_wake_latency_us,
            ),
            ("worst_wake_latency_cv", row.worst_wake_latency_cv),
            ("worst_mean_run_delay_us", row.worst_mean_run_delay_us),
            ("worst_run_delay_us", row.worst_run_delay_us),
            ("page_locality", row.page_locality),
            ("cross_node_migration_ratio", row.cross_node_migration_ratio),
        ] {
            assert_eq!(
                val, 0.0,
                "{name} must collapse to 0.0 for non-finite input {non_finite:?}",
            );
        }
        // Motivation check: the sanitized row serializes. Without the
        // `finite_or_zero` wraps, serde_json::to_string would return
        // Err because NaN / Infinity have no JSON representation.
        serde_json::to_string(&row).expect("sanitized row must serialize cleanly");
    }

    /// `sidecar_to_row` must sanitize NaN in every direct f64 field
    /// (both [`ScenarioStats`]-sourced and the
    /// [`MonitorSummary`]-sourced `imbalance_ratio`), not just a
    /// representative sample — same `serde_json` rejects-NaN
    /// motivation. Unlike `ext_metrics`, direct fields can't be
    /// dropped (the row schema is fixed), so non-finite collapses to
    /// 0.0 with a warn.
    #[test]
    fn sidecar_to_row_zeros_nan_in_every_direct_f64_field() {
        assert_all_direct_f64_fields_sanitized(f64::NAN);
    }

    /// Companion to `sidecar_to_row_zeros_nan_in_every_direct_f64_field`
    /// pinning the `+Infinity` branch of `finite_or_zero` for every
    /// direct f64 field on the row.
    #[test]
    fn sidecar_to_row_zeros_pos_infinity_in_every_direct_f64_field() {
        assert_all_direct_f64_fields_sanitized(f64::INFINITY);
    }

    /// Companion to `sidecar_to_row_zeros_nan_in_every_direct_f64_field`
    /// pinning the `-Infinity` branch of `finite_or_zero` for every
    /// direct f64 field on the row.
    #[test]
    fn sidecar_to_row_zeros_neg_infinity_in_every_direct_f64_field() {
        assert_all_direct_f64_fields_sanitized(f64::NEG_INFINITY);
    }

    /// Subnormal f64 values (IEEE 754 denormals) are finite —
    /// `is_finite()` returns `true` for them — and must pass through
    /// `finite_or_zero` unchanged. Guards against a future refactor
    /// that reaches for `is_normal()` instead of `is_finite()`,
    /// which would incorrectly collapse subnormals to 0.0 and erase
    /// very-small legitimate measurements. `f64::MIN_POSITIVE` is the
    /// smallest normal positive; `/ 2.0` lands in the subnormal
    /// range.
    #[test]
    fn sidecar_to_row_preserves_subnormal_f64_in_direct_fields() {
        use crate::assert::ScenarioStats;
        use crate::test_support;
        let subnormal = f64::MIN_POSITIVE / 2.0;
        assert!(subnormal.is_finite(), "subnormal must still be finite");
        assert!(!subnormal.is_normal(), "subnormal must not be normal");
        assert!(subnormal > 0.0, "subnormal is positive");
        let sc = test_support::SidecarResult {
            stats: ScenarioStats {
                worst_spread: subnormal,
                worst_page_locality: -subnormal,
                worst_wake_latency_cv: subnormal,
                ..Default::default()
            },
            ..test_support::SidecarResult::test_fixture()
        };
        let row = sidecar_to_row(&sc);
        assert_eq!(
            row.spread, subnormal,
            "positive subnormal must pass through finite_or_zero unchanged",
        );
        assert_eq!(
            row.page_locality, -subnormal,
            "negative subnormal must pass through finite_or_zero unchanged",
        );
        assert_eq!(
            row.worst_wake_latency_cv, subnormal,
            "subnormal on a second direct-f64 field must also pass through",
        );
        // Motivation check: subnormals serialize (unlike NaN / ±Inf,
        // serde_json emits them as standard decimal literals).
        serde_json::to_string(&row).expect("subnormals serialize cleanly");
    }

    /// Pins that the direct-field NaN sanitization in
    /// `sidecar_to_row` does NOT reach into `ext_metrics`. Finite
    /// `ext_metrics` entries must survive untouched even when every
    /// direct f64 field collapses to 0.0, and the `ext_metrics` map
    /// must not grow a sanitization-synthesized entry. Complements
    /// [`sidecar_to_row_drops_non_finite_ext_metrics`] (which pins
    /// that non-finite `ext_metrics` entries are DROPPED) by pinning
    /// the orthogonal claim: direct-field sanitization never writes
    /// into `ext_metrics` regardless of the direct values.
    #[test]
    fn sidecar_to_row_direct_field_nan_does_not_touch_ext_metrics() {
        use crate::assert::ScenarioStats;
        use crate::test_support;
        let mut ext = BTreeMap::new();
        ext.insert("finite_nonzero".to_string(), 2.5);
        ext.insert("finite_zero".to_string(), 0.0);
        ext.insert("finite_negative".to_string(), -7.25);
        let sc = test_support::SidecarResult {
            stats: ScenarioStats {
                // Every direct f64 field non-finite.
                worst_spread: f64::NAN,
                worst_migration_ratio: f64::INFINITY,
                worst_p99_wake_latency_us: f64::NEG_INFINITY,
                worst_median_wake_latency_us: f64::NAN,
                worst_wake_latency_cv: f64::INFINITY,
                worst_mean_run_delay_us: f64::NEG_INFINITY,
                worst_run_delay_us: f64::NAN,
                worst_page_locality: f64::INFINITY,
                worst_cross_node_migration_ratio: f64::NEG_INFINITY,
                ext_metrics: ext.clone(),
                ..Default::default()
            },
            ..test_support::SidecarResult::test_fixture()
        };
        let row = sidecar_to_row(&sc);

        // Direct-field collapse still works.
        assert_eq!(row.spread, 0.0);
        assert_eq!(row.migration_ratio, 0.0);
        assert_eq!(row.page_locality, 0.0);

        // ext_metrics survives unchanged — same length, same keys,
        // same values.
        assert_eq!(
            row.ext_metrics.len(),
            ext.len(),
            "direct-field sanitization must not add or drop ext_metrics entries",
        );
        for (k, v) in &ext {
            assert_eq!(
                row.ext_metrics.get(k),
                Some(v),
                "ext_metrics entry {k:?} must pass through unchanged",
            );
        }

        // Motivation check: the full row still serializes.
        serde_json::to_string(&row).expect("sanitized row must serialize cleanly");
    }

    /// `sidecar_to_row` must drop NaN / +Infinity / -Infinity from
    /// `ext_metrics` because `serde_json::to_string` rejects non-finite
    /// f64 values — without this guard a single malformed scenario
    /// metric would poison every sidecar write on its batch. Finite
    /// entries must pass through unchanged. Also checks that the
    /// post-filter row serializes cleanly (the motivation for the
    /// filter).
    #[test]
    fn sidecar_to_row_drops_non_finite_ext_metrics() {
        use crate::assert::ScenarioStats;
        use crate::test_support;
        let mut ext = BTreeMap::new();
        ext.insert("good".to_string(), 1.0);
        ext.insert("nan".to_string(), f64::NAN);
        ext.insert("inf".to_string(), f64::INFINITY);
        ext.insert("neg_inf".to_string(), f64::NEG_INFINITY);
        let sc = test_support::SidecarResult {
            stats: ScenarioStats {
                ext_metrics: ext,
                ..Default::default()
            },
            ..test_support::SidecarResult::test_fixture()
        };
        let row = sidecar_to_row(&sc);
        assert_eq!(
            row.ext_metrics.len(),
            1,
            "only the finite entry should survive: {:?}",
            row.ext_metrics
        );
        assert_eq!(row.ext_metrics.get("good"), Some(&1.0));
        assert!(!row.ext_metrics.contains_key("nan"));
        assert!(!row.ext_metrics.contains_key("inf"));
        assert!(!row.ext_metrics.contains_key("neg_inf"));
        // Motivation check: the post-filter row serializes. Without the
        // filter, serde_json::to_string would return Err because NaN /
        // Infinity have no JSON representation.
        serde_json::to_string(&row).expect("filtered row must serialize cleanly");
    }

    /// `sidecar_to_row` must drop the JSON-walker depth-cap sentinel
    /// [`crate::test_support::WALK_TRUNCATION_SENTINEL_NAME`] from
    /// `ext_metrics`. The sentinel is diagnostic metadata about the
    /// extraction pass (depth cap hit), not a scenario metric, so it
    /// must not leak into A/B comparison output where it would be
    /// mistaken for a real measurement and skew filter / aggregation
    /// logic. Sibling finite entries must survive untouched.
    #[test]
    fn sidecar_to_row_drops_walk_truncation_sentinel() {
        use crate::assert::ScenarioStats;
        use crate::test_support;
        let mut ext = BTreeMap::new();
        ext.insert("good".to_string(), 1.0);
        ext.insert(
            test_support::WALK_TRUNCATION_SENTINEL_NAME.to_string(),
            72.0,
        );
        let sc = test_support::SidecarResult {
            stats: ScenarioStats {
                ext_metrics: ext,
                ..Default::default()
            },
            ..test_support::SidecarResult::test_fixture()
        };
        let row = sidecar_to_row(&sc);
        assert_eq!(
            row.ext_metrics.len(),
            1,
            "only the real metric should survive: {:?}",
            row.ext_metrics,
        );
        assert_eq!(row.ext_metrics.get("good"), Some(&1.0));
        assert!(
            !row.ext_metrics
                .contains_key(test_support::WALK_TRUNCATION_SENTINEL_NAME),
            "sentinel must not appear in the row's ext_metrics",
        );
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
        let mut row = make_row("a", "t", true, 42.0);
        row.gap_ms = 100;
        row.migrations = 7;
        row.migration_ratio = 0.3;
        row.imbalance_ratio = 2.0;
        row.max_dsq_depth = 5;
        row.stall_count = 3;
        row.fallback_count = 11;
        row.keep_last_count = 4;
        row.worst_p99_wake_latency_us = 99.0;
        row.worst_median_wake_latency_us = 50.0;
        row.worst_wake_latency_cv = 0.5;
        row.total_iterations = 1000;
        row.worst_mean_run_delay_us = 25.0;
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
        assert_eq!(read_metric(&row, "worst_p99_wake_latency_us"), Some(99.0));
        assert_eq!(read_metric(&row, "worst_median_wake_latency_us"), Some(50.0));
        assert_eq!(read_metric(&row, "worst_wake_latency_cv"), Some(0.5));
        assert_eq!(read_metric(&row, "total_iterations"), Some(1000.0));
        assert_eq!(read_metric(&row, "worst_mean_run_delay_us"), Some(25.0));
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
        let mut row = make_row("a", "t", true, 5.0);
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

    /// Build a row matching the sidecar-derived schema:
    /// `work_type = "CpuSpin"`, all metrics zeroed except `spread`
    /// and `total_iterations`.
    fn cmp_row(scenario: &str, topo: &str, passed: bool, spread: f64, iters: u64) -> GauntletRow {
        let mut r = make_row(scenario, topo, passed, spread);
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

    /// Flag-profile collision regression pin. Two rows that share
    /// `(scenario, topology, work_type)` but run under different
    /// flag sets must NOT collide in the A/B join. Before `flags`
    /// was part of the identity key, `compare_rows` would match
    /// rows_b's `llc` variant against whichever rows_a variant came
    /// first — typically `borrow` — and silently produce a diff
    /// across two unrelated flag profiles.
    ///
    /// Construction: rows_a carries `llc` at spread=10 and `borrow`
    /// at spread=100; rows_b mirrors the 3-tuple but swaps the flag
    /// order (`borrow` at 10, `llc` at 100). A 3-tuple join would
    /// pair `(llc, 10)` vs `(borrow, 10)` → zero spread delta, zero
    /// regressions. The 4-tuple join pairs same-flag-set rows:
    /// `(llc, 10)` vs `(llc, 100)` and `(borrow, 100)` vs
    /// `(borrow, 10)` — one regression, one improvement on
    /// worst_spread.
    #[test]
    fn compare_rows_same_key_different_flags_do_not_collide() {
        let mut a_llc = cmp_row("t", "tiny-1llc", true, 10.0, 0);
        a_llc.flags = vec!["llc".to_string()];
        let mut a_borrow = cmp_row("t", "tiny-1llc", true, 100.0, 0);
        a_borrow.flags = vec!["borrow".to_string()];
        let mut b_borrow = cmp_row("t", "tiny-1llc", true, 10.0, 0);
        b_borrow.flags = vec!["borrow".to_string()];
        let mut b_llc = cmp_row("t", "tiny-1llc", true, 100.0, 0);
        b_llc.flags = vec!["llc".to_string()];

        let rows_a = vec![a_llc, a_borrow];
        let rows_b = vec![b_borrow, b_llc];
        let res = compare_rows(&rows_a, &rows_b, None, None);

        // Each flag profile's spread moved by 90 → one regression
        // (llc 10→100) and one improvement (borrow 100→10).
        assert_eq!(
            res.regressions, 1,
            "llc regression should fire (10 → 100)",
        );
        assert_eq!(
            res.improvements, 1,
            "borrow improvement should fire (100 → 10)",
        );
        // Neither side should be treated as new / removed — both
        // keys match across A and B when flags are part of the key.
        assert_eq!(res.new_in_b, 0);
        assert_eq!(res.removed_from_a, 0);
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

    // -- format_host_delta: the 5 match arms of the host-delta
    //    section emitted under `stats compare --runs a b`. --

    /// Builder for a `HostContext` with enough populated fields to
    /// exercise `HostContext::diff`. Leaves everything else at its
    /// `Default` so each test varies only the field under study.
    fn host_ctx(release: &str, kernel_cmdline: Option<&str>) -> crate::host_context::HostContext {
        crate::host_context::HostContext {
            kernel_name: Some("Linux".to_string()),
            kernel_release: Some(release.to_string()),
            kernel_cmdline: kernel_cmdline.map(str::to_string),
            ..Default::default()
        }
    }

    /// `(Some, Some)` identical: the helper emits a one-line
    /// confirmation so users running `stats compare` can distinguish
    /// "same host" from "captured but unused" without inspecting
    /// individual sidecars.
    #[test]
    fn format_host_delta_both_present_identical() {
        let ctx = host_ctx("6.14.0", Some("preempt=lazy"));
        let out = format_host_delta(Some(&ctx), Some(&ctx), "a-run", "b-run");
        assert_eq!(out, "\nhost: identical between 'a-run' and 'b-run'\n");
    }

    /// `(Some, Some)` differing: the helper emits the header line
    /// followed by whatever `HostContext::diff` produced. Asserts
    /// the structural shape (header present, delta body present)
    /// rather than the exact diff formatting so this test stays
    /// robust to future tweaks to the diff renderer.
    #[test]
    fn format_host_delta_both_present_differ() {
        let ha = host_ctx("6.14.0", Some("preempt=lazy"));
        let hb = host_ctx("6.15.1", Some("preempt=lazy"));
        let out = format_host_delta(Some(&ha), Some(&hb), "a", "b");
        assert!(out.starts_with("\nhost delta ('a' → 'b'):\n"), "got: {out:?}");
        // `kernel_release` differs between the two contexts so the
        // diff body must be non-empty — confirms we routed through
        // the `else` arm and not the `identical` arm.
        let body = &out["\nhost delta ('a' → 'b'):\n".len()..];
        assert!(!body.is_empty(), "differing contexts must produce a diff body");
        // Pin the trailing-newline contract: the other three arms
        // (`identical`, left-only, right-only) all end with '\n'; the
        // differ arm delegates to `HostContext::diff()` whose output
        // must also terminate with a newline so caller-side
        // concatenation with subsequent sections doesn't butt headers
        // against the last diff line. A regression that trimmed the
        // trailing newline in `HostContext::diff` would produce
        // run-on output only in the differ case — this assertion
        // catches that asymmetry.
        assert!(
            out.ends_with('\n'),
            "differ arm must end with a newline for contiguous-section output: {out:?}",
        );
    }

    /// `(Some, None)` left-only: one run captured host data, the
    /// other did not (mixed tooling version, partial migration
    /// window). Surface the asymmetry explicitly so the missing
    /// side is diagnosable.
    #[test]
    fn format_host_delta_left_only() {
        let ctx = host_ctx("6.14.0", Some("preempt=lazy"));
        let out = format_host_delta(Some(&ctx), None, "a-run", "b-run");
        assert_eq!(out, "\nhost: captured in 'a-run' only, delta unavailable\n");
    }

    /// `(None, Some)` right-only: symmetric complement to
    /// `left_only`. The `b`-name must appear (not `a`) — guards
    /// against a future copy-paste typo that swaps the names.
    #[test]
    fn format_host_delta_right_only() {
        let ctx = host_ctx("6.14.0", Some("preempt=lazy"));
        let out = format_host_delta(None, Some(&ctx), "a-run", "b-run");
        assert_eq!(out, "\nhost: captured in 'b-run' only, delta unavailable\n");
    }

    /// `(None, None)`: neither side carries host data. The section
    /// is fully suppressed — no blank line, no header, nothing.
    /// Pinning this prevents a regression that introduces a
    /// spurious "host: none" footer on legacy runs.
    #[test]
    fn format_host_delta_both_absent_emits_nothing() {
        assert_eq!(format_host_delta(None, None, "a", "b"), "");
    }

    // -- GauntletRow serde round-trip tests --
    //
    // Both `flags: Vec<String>` and `ext_metrics: BTreeMap<String, f64>`
    // carry `#[serde(default, skip_serializing_if = "…::is_empty")]`.
    // These tests pin that symmetric contract: the keys disappear from
    // JSON when the collection is empty, round-trip through from_str
    // reconstructs an equivalent row, and a non-empty payload emits
    // its contents verbatim.

    /// Empty collections are elided on serialize. Regression guard for
    /// the `skip_serializing_if` half — dropping it would make the
    /// writer emit `"flags":[]` / `"ext_metrics":{}` noise on every
    /// row (the `default` half is guarded by the sibling round-trip
    /// test).
    #[test]
    fn gauntlet_row_empty_collections_omit_keys() {
        let row = make_row("scn", "topo", true, 0.0);
        assert!(row.flags.is_empty());
        assert!(row.ext_metrics.is_empty());
        let json = serde_json::to_string(&row).unwrap();
        assert!(
            !json.contains("\"flags\""),
            "empty flags must be omitted from JSON: {json}"
        );
        assert!(
            !json.contains("\"ext_metrics\""),
            "empty ext_metrics must be omitted from JSON: {json}"
        );
    }

    /// Non-empty collections appear with their full payload. Locks in
    /// that `skip_serializing_if` only fires on empty, not on "has
    /// content". A false positive here would silently drop flags and
    /// extensible metrics from sidecar files.
    #[test]
    fn gauntlet_row_non_empty_collections_emit_payload() {
        let mut row = make_row("scn", "topo", true, 0.0);
        row.flags = vec!["flag_a".into(), "flag_b".into()];
        row.ext_metrics.insert("custom_metric".into(), 42.5);
        let json = serde_json::to_string(&row).unwrap();
        assert!(
            json.contains("\"flags\":[\"flag_a\",\"flag_b\"]"),
            "flags payload missing: {json}"
        );
        assert!(
            json.contains("\"custom_metric\":42.5"),
            "ext_metrics payload missing: {json}"
        );
    }

    /// Round-trip with empty collections: the writer omits the keys
    /// (via `skip_serializing_if`), so the reader must default them
    /// back to empty for the round-trip to close. Regression guard
    /// for the `default` half of the symmetric pair — removing it
    /// would make deserialize fail on JSON this same process just
    /// produced.
    #[test]
    fn gauntlet_row_round_trip_empty_collections() {
        let row = make_row("scn", "topo", true, 1.5);
        let json = serde_json::to_string(&row).unwrap();
        let back: GauntletRow = serde_json::from_str(&json).unwrap();
        assert_eq!(back, row);
        assert!(back.flags.is_empty());
        assert!(back.ext_metrics.is_empty());
    }

    /// Round-trip with populated collections: every entry survives the
    /// to_string → from_str cycle. Guards against any future field-level
    /// serde attribute (e.g. a rename or custom serializer) accidentally
    /// shearing content on one side of the cycle.
    #[test]
    fn gauntlet_row_round_trip_non_empty_collections() {
        let mut row = make_row("scn", "topo", false, 3.14);
        row.flags = vec!["a".into(), "b".into(), "c".into()];
        row.ext_metrics.insert("m1".into(), 1.0);
        row.ext_metrics.insert("m2".into(), 2.5);
        let json = serde_json::to_string(&row).unwrap();
        let back: GauntletRow = serde_json::from_str(&json).unwrap();
        assert_eq!(back, row);
    }
}
