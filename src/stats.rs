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
/// ([`compare_partitions`]) uses `higher_is_worse` for delta direction.
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
/// Six of the remaining metrics in [`METRICS`] have matching
/// registry / field / DataFrame column names (`worst_p99_wake_latency_us`,
/// `worst_median_wake_latency_us`, `worst_wake_latency_cv`,
/// `total_iterations`, `worst_mean_run_delay_us`,
/// `worst_run_delay_us`) and are not listed — no translation
/// to document.
///
/// Two further entries in [`METRICS`] —
/// `worst_wake_latency_tail_ratio` and
/// `worst_iterations_per_worker` — are registered and
/// populated on [`GauntletRow`] but have NO DataFrame column
/// in [`build_dataframe`]. Consumers that reach for them via
/// polars receive "column not found" — go through the
/// registry accessor closure instead. A follow-up task (see
/// comments on the two [`GauntletRow`] fields) wires them into
/// the DataFrame once the comparison pipeline accounts for the
/// two new dimensions.
///
/// Quoting the matching list instead of a bare count avoids
/// the prior silent-drift failure mode: the old "remaining
/// eight metrics" sentence was wrong (two of the eight have
/// no DataFrame column) and it would have stayed wrong on any
/// future matching-name rename.
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
        // Integer event count, not a rate — the source field on
        // `MonitorSummary::event_deltas.total_fallback` is a cumulative
        // delta across the run, not per-second. Empty unit matches the
        // other counter metrics (`stall_count`, `total_iterations`,
        // `total_migrations`).
        display_unit: "",
        accessor: |r| Some(r.fallback_count as f64),
    },
    MetricDef {
        name: "total_keep_last",
        polarity: crate::test_support::Polarity::LowerBetter,
        default_abs: 5.0,
        default_rel: 0.30,
        // Integer event count, not a rate — see `total_fallback`
        // rationale above. Source field is
        // `MonitorSummary::event_deltas.total_dispatch_keep_last`.
        display_unit: "",
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
        // Ratio of p99 / median wake latency, worst-case across
        // cgroups. `LowerBetter` because a higher ratio signals a
        // stretched long tail. Unitless; baseline is 1.0 (p99 == median
        // is the perfect-uniform floor set by order-statistic
        // ordering). `default_abs = 0.5` guards against trivially
        // small deltas that percent-only gates would flag; `default_rel
        // = 0.25` matches the wake-latency metrics' percent gate.
        //
        // Samples-required noise gate: the accessor returns `None` when
        // the run completed fewer than
        // [`WAKE_LATENCY_TAIL_RATIO_MIN_ITERATIONS`] iterations; with
        // few samples the p99 estimate is effectively the observed
        // maximum and the tail ratio is dominated by a single
        // outlier rather than a distributional signal. Routing
        // through `None` lets `compare_rows` fall through to the
        // `ext_metrics` lookup (which is also empty for a sub-
        // threshold run), then to the `unwrap_or(0.0)` default, so
        // both A- and B-side rows collapse to 0.0 and the subsequent
        // `abs() < EPSILON` short-circuit silently skips the metric
        // for that row. See [`WAKE_LATENCY_TAIL_RATIO_MIN_ITERATIONS`]
        // for the threshold-value rationale.
        name: "worst_wake_latency_tail_ratio",
        polarity: crate::test_support::Polarity::LowerBetter,
        default_abs: 0.5,
        default_rel: 0.25,
        display_unit: "x",
        accessor: |r| {
            if r.total_iterations < WAKE_LATENCY_TAIL_RATIO_MIN_ITERATIONS {
                None
            } else {
                Some(r.worst_wake_latency_tail_ratio)
            }
        },
    },
    MetricDef {
        // Per-worker iteration throughput, worst (lowest) cgroup.
        // `HigherBetter` mirrors [`total_iterations`]: a cgroup that
        // fell behind regresses this downward, and a cross-variant
        // improvement raises it. `default_abs = 10.0` is the absolute
        // iteration-count floor below which deltas are noise;
        // `default_rel = 0.10` mirrors the `total_iterations` gate.
        //
        // Derivation of `abs = 10`: this metric is PER-WORKER. In-tree
        // fixtures span `workers_per_cgroup` from 1 through 8 (see
        // the KtstrTestEntry declarations under src/scenario/*.rs and
        // tests/*.rs); `KtstrTestEntry::DEFAULT.workers_per_cgroup`
        // is 2, with scenario-level overrides commonly picking 4 or
        // 8. A per-worker floor of 10 therefore corresponds to
        // aggregate regressions of 10-80 total iterations across the
        // supported worker counts — high enough that a lightly-
        // loaded scheduler's jitter does not flag a regression, low
        // enough that a genuine drop (e.g. a cgroup that fell behind
        // by 10 iterations at workers=1, or 80 at workers=8) still
        // trips the gate. Going below 10 would flag normal cross-run
        // jitter on single-worker configs; going above 10 would mask
        // regressions on low-worker-count tests. The `rel=0.10`
        // companion gate handles larger throughputs proportionally,
        // so the `abs=10` floor only binds in the small-count regime
        // where rel-only would let single-digit losses slip through.
        name: "worst_iterations_per_worker",
        polarity: crate::test_support::Polarity::HigherBetter,
        default_abs: 10.0,
        default_rel: 0.10,
        display_unit: "",
        accessor: |r| Some(r.worst_iterations_per_worker),
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

/// Minimum total iterations a run must have accumulated before the
/// `worst_wake_latency_tail_ratio` metric participates in regression
/// math.
///
/// Below this threshold the p99 / median ratio is dominated by a
/// handful of outlier samples rather than a distributional signal:
/// p99 on an N-sample set where `N < 100` collapses to approximately
/// `samples.max()` (the empirical p99 sits at the Nth item of a
/// sorted set, rounded down, so with N=10 every "p99" is in fact the
/// maximum), and the ratio `max/median` swings by order of magnitude
/// across runs that differ only in which worker happened to hit a
/// scheduling stall. `compare_rows` would report those swings as
/// regressions / improvements, burying real signal under low-N noise.
///
/// 100 is the threshold of interest because percentile estimation
/// stabilizes when the sample count crosses `1 / (1 - target_p)` —
/// i.e. 100 samples for a p99 — which is the point at which at least
/// one sample is expected in the 99th-percentile tail by pigeonhole.
/// Below this floor the p99 estimator degenerates to the observed
/// maximum (`samples[99]` when N is exactly 100, and a still-sparse
/// tail at N just above 100). Above 100 the ratio begins to reflect
/// actual tail behavior rather than single-sample extrema.
///
/// The gate uses `total_iterations` (scenario-wide sum across every
/// cgroup in the run) as a coarse floor, not an exact per-cgroup
/// sample count. That sum OVERESTIMATES the per-cgroup iteration
/// count when the scenario has multiple cgroups sharing load, so a
/// scenario whose total just clears the floor may still have
/// individual cgroups with fewer than 100 iterations and therefore
/// noisy per-cgroup tail ratios. The floor is a minimum-viable
/// filter against the lowest-N degeneracy, not a guarantee that
/// every cgroup in a passing row has a stable p99.
///
/// The gate is applied in the metric's accessor closure in [`METRICS`]:
/// a row with `total_iterations < WAKE_LATENCY_TAIL_RATIO_MIN_ITERATIONS`
/// returns `None`, which `compare_rows` short-circuits to 0.0 against
/// both A- and B-side rows, which then falls under the
/// `abs() < EPSILON` "unchanged" guard and emits no finding.
pub const WAKE_LATENCY_TAIL_RATIO_MIN_ITERATIONS: u64 = 100;

/// Look up a metric definition by name.
pub fn metric_def(name: &str) -> Option<&'static MetricDef> {
    METRICS.iter().find(|m| m.name == name)
}

/// Render the [`METRICS`] registry for `cargo ktstr stats list-metrics`.
///
/// `json=false` renders a comfy-table with one row per registered
/// metric and columns NAME / POLARITY / DEFAULT_ABS / DEFAULT_REL
/// / UNIT. `json=true` emits `serde_json::to_string_pretty`
/// on the whole [`METRICS`] slice — the `accessor` fn-pointer is
/// `#[serde(skip)]` so the array carries only wire-stable fields.
///
/// Iteration order equals [`METRICS`] declaration order (the
/// canonical surface order for sidecar / CI-gate consumers).
///
/// The return is owned `String` rather than a print-direct helper so
/// callers can pin output via `assert_eq!` in tests; the cargo-ktstr
/// dispatch arm at `run_stats` writes it to stdout verbatim.
pub fn list_metrics(json: bool) -> anyhow::Result<String> {
    if json {
        return serde_json::to_string_pretty(METRICS)
            .map_err(|e| anyhow::anyhow!("serialize METRICS to JSON: {e}"));
    }

    let mut table = crate::cli::new_table();
    table.set_header(vec![
        "NAME",
        "POLARITY",
        "DEFAULT_ABS",
        "DEFAULT_REL",
        "UNIT",
    ]);
    for m in METRICS {
        table.add_row(vec![
            m.name.to_string(),
            polarity_label(m.polarity),
            format!("{}", m.default_abs),
            format!("{}", m.default_rel),
            m.display_unit.to_string(),
        ]);
    }
    Ok(format!("{table}\n"))
}

/// Short human label for a [`Polarity`](crate::test_support::Polarity)
/// variant in the list-metrics table.
///
/// `HigherBetter` → `higher`, `LowerBetter` → `lower`,
/// `TargetValue(t)` → `target(t)`, `Unknown` → `unknown`. Match is
/// total; adding a new `Polarity` variant without extending this
/// rendering surfaces as a compile error.
fn polarity_label(p: crate::test_support::Polarity) -> String {
    use crate::test_support::Polarity;
    match p {
        Polarity::HigherBetter => "higher".to_string(),
        Polarity::LowerBetter => "lower".to_string(),
        Polarity::TargetValue(t) => format!("target({t})"),
        Polarity::Unknown => "unknown".to_string(),
    }
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
/// flow through `ext_metrics` (a `BTreeMap<String, f64>` — see the
/// field definition below): non-finite entries are DROPPED at
/// [`sidecar_to_row`] ingress rather than stored. A subsequent
/// `ext_metrics.get(name)` returns `None` because the key is
/// absent, not because an `Option::None` sentinel is stored — the
/// map's value type is `f64`, which cannot represent "missing".
/// Absent-key and zero-valued metrics therefore remain distinguishable
/// for downstream consumers.
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
    /// filter in [`compare_rows`] and the typed
    /// [`RowFilter::scheduler`] so users can narrow A/B comparisons
    /// by scheduler name.
    pub scheduler: String,
    /// Kernel version carried from the source sidecar
    /// (`SidecarResult::kernel_version`). `None` when the sidecar
    /// writer could not extract a version (e.g. a raw kernel image
    /// path with no metadata.json sibling, or a dirty source tree
    /// where HEAD does not describe the build). Surfaced via the
    /// typed [`RowFilter::kernels`] for narrowing — when the user
    /// passes `--kernel 6.14.2` (repeatable), rows with `None` are
    /// dropped to preserve the operator's intent ("only these
    /// kernels"); a `None`-as-wildcard would silently dilute the
    /// filtered set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel_version: Option<String>,
    /// ktstr project git commit carried from the source sidecar
    /// (`SidecarResult::project_commit`). Short hex with optional
    /// `-dirty` suffix (e.g. `"abcdef1"` or `"abcdef1-dirty"`).
    /// `None` when the sidecar writer could not probe a git repo
    /// at write time (cwd not inside a checkout, or
    /// [`crate::test_support::sidecar::detect_project_commit`]
    /// failed for any reason). Surfaced via the typed
    /// [`RowFilter::project_commits`] for narrowing — when the
    /// user passes `--project-commit abcdef1` (repeatable), rows
    /// with `None` are dropped to preserve the operator's intent
    /// ("only these commits"); a `None`-as-wildcard would silently
    /// dilute the filtered set, mirroring the [`RowFilter::kernels`]
    /// policy.
    ///
    /// Sourced from `SidecarResult::project_commit`; shortened to
    /// `commit` on the row because the project commit is the
    /// most-frequently-narrowed-on of the three commit dimensions
    /// on `SidecarResult`. The other two commit fields —
    /// `SidecarResult::scheduler_commit` and
    /// `SidecarResult::kernel_commit` — get fully-qualified names
    /// here (`scheduler_commit` is reserved and not yet exposed,
    /// `kernel_commit` is the typed filter `RowFilter::kernel_commits`
    /// applies). The bare `commit` shortening is internal to
    /// `GauntletRow`; the CLI flag is the disambiguated
    /// `--project-commit` form so an operator never has to guess
    /// which "commit" dimension a bare `--commit` would have meant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
    /// Kernel SOURCE TREE git commit carried from the source
    /// sidecar (`SidecarResult::kernel_commit`). Short hex with
    /// optional `-dirty` suffix (e.g. `"abcdef1"` or
    /// `"abcdef1-dirty"`). `None` when the sidecar writer could
    /// not probe a git repo for the kernel directory at write
    /// time (KTSTR_KERNEL points at a non-git path, the
    /// underlying source is `Tarball` / `Git` rather than
    /// `Local`, or
    /// [`crate::test_support::sidecar::detect_kernel_commit`]
    /// failed for any reason).
    ///
    /// Distinct from [`GauntletRow::commit`]: that field tracks
    /// the ktstr framework HEAD ("which version of the harness
    /// produced this sidecar?"); this field tracks the kernel
    /// tree HEAD ("which kernel commit did this run boot?"). Two
    /// runs with the same `commit` but different `kernel_commit`
    /// values are typical when the kernel under test is updated
    /// without re-checking out the harness; two runs with the
    /// same `kernel_commit` but different `commit` values are
    /// typical when the harness is bumped without rebuilding the
    /// kernel.
    ///
    /// Surfaced via the typed [`RowFilter::kernel_commits`] for
    /// narrowing — same opt-in policy as [`RowFilter::project_commits`]:
    /// rows with `None` never match a populated filter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel_commit: Option<String>,
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
    /// Run-environment provenance tag carried from
    /// `SidecarResult::run_source` (`"local"` for developer runs,
    /// `"ci"` when [`KTSTR_CI_ENV`](crate::test_support::sidecar::KTSTR_CI_ENV)
    /// was set at write time, `"archive"` when the consumer pulled
    /// the pool from a non-default `--dir`). `None` for sidecars
    /// produced before the field existed (pre-1.0 disposable
    /// schema; re-running the test regenerates the entry).
    /// Surfaced via the typed [`RowFilter::run_sources`] for
    /// narrowing — when the user passes `--run-source local`
    /// (repeatable), rows with `None` are dropped to preserve the
    /// operator's intent ("only these environments"); a
    /// `None`-as-wildcard would silently dilute the filtered set,
    /// mirroring the [`RowFilter::kernels`] /
    /// [`RowFilter::project_commits`] / [`RowFilter::kernel_commits`]
    /// policy.
    ///
    /// Field name `run_source` (renamed from `source`) disambiguates
    /// from [`crate::cache::KernelSource`] / `KernelMetadata.source`
    /// — those describe the kernel build's input (tarball / git /
    /// local), this describes the run-environment provenance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_source: Option<String>,
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
    /// Worst-case per-cgroup scheduling gap (ms). Surfaced in
    /// [`METRICS`] under registry name `worst_gap_ms`; the
    /// field / registry / DataFrame-column divergence is catalogued
    /// in the triples table on [`METRICS`].
    pub gap_ms: u64,
    /// Total CPU migrations across the run. Surfaced in [`METRICS`]
    /// under registry name `total_migrations`; see the triples
    /// table on [`METRICS`] for the rationale behind the
    /// field / registry / DataFrame-column divergence.
    pub migrations: u64,
    /// Worst-case per-cgroup migrations-per-iteration ratio.
    /// Surfaced in [`METRICS`] under registry name
    /// `worst_migration_ratio`; see the triples table on
    /// [`METRICS`] for the field / registry / DataFrame-column
    /// divergence.
    pub migration_ratio: f64,
    // Monitor fields (host-side telemetry from guest memory reads).
    /// Worst per-sample cgroup imbalance ratio. Surfaced in
    /// [`METRICS`] under registry name `max_imbalance_ratio`
    /// (DataFrame column `imbalance`); see the triples table on
    /// [`METRICS`] for the registry/field/column rationale.
    pub imbalance_ratio: f64,
    /// Worst observed DSQ queue depth. Registry and field names
    /// match (`max_dsq_depth`) but the DataFrame column is
    /// `dsq_depth`; see the triples table on [`METRICS`] for the
    /// column-level rename rationale.
    pub max_dsq_depth: u32,
    /// Stalled-sample count across the run. Registry and field
    /// names match (`stall_count`) but the DataFrame column is
    /// `stalls`; see the triples table on [`METRICS`] for the
    /// column-level rename rationale.
    pub stall_count: usize,
    /// Fallback-dispatch count across the run. Carried as-is from
    /// `MonitorSummary::event_deltas.total_fallback` — an integer
    /// event count, NOT a rate. Surfaced in [`METRICS`] under
    /// registry name `total_fallback` (DataFrame column `fallback`);
    /// see the triples table on [`METRICS`] for the registry / field /
    /// column rationale.
    pub fallback_count: i64,
    /// Keep-last dispatch count across the run. Carried as-is from
    /// `MonitorSummary::event_deltas.total_dispatch_keep_last` — an
    /// integer event count, NOT a rate. Surfaced in [`METRICS`] under
    /// registry name `total_keep_last` (DataFrame column `keep_last`);
    /// see the triples table on [`METRICS`] for the registry / field /
    /// column rationale.
    pub keep_last_count: i64,
    // Benchmarking fields.
    pub worst_p99_wake_latency_us: f64,
    pub worst_median_wake_latency_us: f64,
    pub worst_wake_latency_cv: f64,
    pub total_iterations: u64,
    pub worst_mean_run_delay_us: f64,
    pub worst_run_delay_us: f64,
    /// Worst-case ratio of p99 / median wake latency across cgroups.
    /// Higher values indicate a stretched long tail. Registry name
    /// matches the field name; see the triples table on [`METRICS`]
    /// for the full registry / field / DataFrame-column mapping.
    /// Noise-suppressed when the scenario produced fewer than
    /// [`WAKE_LATENCY_TAIL_RATIO_MIN_ITERATIONS`] iterations — see
    /// the constant's doc for the rationale.
    pub worst_wake_latency_tail_ratio: f64,
    /// Worst-case per-worker iteration count across cgroups (LOWEST
    /// across cgroups — lower is worse). Registry name matches the
    /// field name; see the triples table on [`METRICS`] for the
    /// field / registry / DataFrame-column mapping.
    ///
    /// # `worst_` vs `lowest_` naming evaluation
    ///
    /// A `lowest_iterations_per_worker` rename was considered — it
    /// would describe the merge direction (min across cgroups) more
    /// literally than `worst_`, which semantically maps "worst" to
    /// different merge operations depending on polarity (max for
    /// lower-better metrics, min for higher-better). Rejected
    /// because `worst_` is the codebase-wide prefix for
    /// cross-cgroup roll-ups regardless of polarity — see
    /// `worst_page_locality` (`HigherBetter` → the merge takes the
    /// LOWEST non-zero value) and `worst_spread` (`LowerBetter` →
    /// the merge takes the HIGHEST). Breaking that convention for
    /// one metric would require either (a) renaming every existing
    /// `HigherBetter` worst_* metric to `lowest_*` for consistency,
    /// or (b) accepting a mixed naming scheme where readers have to
    /// cross-reference each metric's polarity to understand the
    /// prefix. Option (a) is a high-churn rename across
    /// sidecars / DataFrames / CI gates; option (b) degrades
    /// readability. The current convention — `worst_` = "the
    /// cross-cgroup roll-up that surfaces the most problematic
    /// cgroup, direction determined by the metric's polarity" —
    /// is documented on [`METRICS`] and applies here.
    pub worst_iterations_per_worker: f64,
    // NUMA fields.
    /// Worst-case per-cgroup NUMA page-locality fraction (lowest
    /// non-zero). Surfaced in [`METRICS`] under registry name
    /// `worst_page_locality`; see the triples table on
    /// [`METRICS`] for the registry/field/column rationale.
    pub page_locality: f64,
    /// Worst-case cross-node migration ratio. Surfaced in
    /// [`METRICS`] under registry name
    /// `worst_cross_node_migration_ratio`; see the triples table
    /// on [`METRICS`] for the registry/field/column rationale.
    pub cross_node_migration_ratio: f64,
    /// Extensible metrics populated by scenarios and processed by the
    /// comparison pipeline. Keyed by metric name; looked up via
    /// [`metric_def`] when a matching entry exists in [`METRICS`].
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub ext_metrics: BTreeMap<String, f64>,
}

/// Typed-field filter set for narrowing [`GauntletRow`] sets in the
/// `cargo ktstr stats compare` pipeline. Every field is `None` /
/// empty by default; populated fields are AND-combined ACROSS
/// fields, with field-internal OR/AND semantics described per-field
/// below. Applied via [`apply_row_filters`] in `compare_partitions`
/// before the rows reach `compare_rows`.
///
/// Match semantics:
/// - `scheduler` / `topology` / `work_type` — STRICT EQUALITY against
///   the row's corresponding field. The sibling substring filter on
///   [`compare_rows`] (`-E`) stays as the only fuzzy-match knob;
///   typed fields are exact so a `--scheduler scx_rusty` filter does
///   NOT spuriously match `scx_rusty_alt`.
/// - `kernels` — repeatable, OR-combined: a row matches iff its
///   `kernel_version` equals ANY entry in `kernels`. Mirrors the
///   `--kernel` flag on `cargo ktstr test`/`coverage`/`llvm-cov`
///   so the same flag name carries the same multi-value semantic
///   across every subcommand.
/// - `project_commits` — repeatable, OR-combined: a row matches
///   iff its `commit` equals ANY entry in `project_commits`. Same
///   multi-value semantic as `kernels`, applied to the ktstr
///   project commit recorded by `detect_project_commit` at
///   sidecar-write time. Surfaced as the `--project-commit` CLI
///   flag.
/// - `kernel_commits` — repeatable, OR-combined: a row matches
///   iff its `kernel_commit` equals ANY entry in `kernel_commits`.
///   Same multi-value semantic as `project_commits`, applied to
///   the kernel source-tree commit recorded by
///   [`crate::test_support::sidecar::detect_kernel_commit`] at
///   sidecar-write time. Filters on the kernel HEAD, NOT on the
///   kernel release version (`kernels` is the version filter).
/// - `run_sources` — repeatable, OR-combined: a row matches iff
///   its `run_source` equals ANY entry in `run_sources`. Same
///   multi-value semantic as `kernels` / `project_commits` /
///   `kernel_commits`, applied to the run-environment provenance
///   tag (`"local"`, `"ci"`, `"archive"`) recorded by
///   [`crate::test_support::sidecar::detect_run_source`] at
///   sidecar-write time, or rewritten to `"archive"` at load
///   time when the consumer pulled the pool from a non-default
///   `--dir`. Surfaced as the `--run-source` CLI flag.
/// - `flags` — AND-combined: every entry in the filter must appear
///   somewhere in the row's `flags` vec. The row may carry
///   additional flags beyond the filter set; the filter pins
///   "at-least-these-flags-are-active", not "exactly-these".
/// - A `kernels`-populated filter against a row whose
///   `kernel_version` is `None` ALWAYS fails (no wildcard semantic)
///   — the operator wrote specific versions and a `None`-row would
///   silently dilute the set. The same opt-in policy applies to
///   `project_commits` against rows with `commit == None`, to
///   `kernel_commits` against rows with `kernel_commit == None`,
///   and to `run_sources` against rows with `run_source == None`.
///
/// Empty `RowFilter` (every field `None`/empty) is the no-op default
/// and matches every row. Use [`RowFilter::default()`] to build it.
#[derive(Clone, Debug, Default)]
pub struct RowFilter {
    /// Repeatable kernel-version filter, OR-combined: a row matches
    /// iff its `GauntletRow::kernel_version` equals ANY entry. Empty
    /// vec disables the filter ("do not filter on kernel"). A row
    /// whose `kernel_version` is itself `None` never matches a
    /// non-empty filter.
    pub kernels: Vec<String>,
    /// Repeatable project-commit filter, OR-combined: a row matches
    /// iff its `GauntletRow::commit` equals ANY entry. Empty vec
    /// disables the filter ("do not filter on commit"). A row whose
    /// `commit` is itself `None` never matches a non-empty filter
    /// — same opt-in semantic as `kernels`.
    ///
    /// Field name `project_commits` (renamed from `commits`)
    /// disambiguates from the sibling `kernel_commits` field — both
    /// describe commit dimensions, so the prefix makes "which
    /// repository's commit?" obvious at every call site.
    pub project_commits: Vec<String>,
    /// Repeatable kernel-source-commit filter, OR-combined: a row
    /// matches iff its `GauntletRow::kernel_commit` equals ANY
    /// entry. Empty vec disables the filter ("do not filter on
    /// kernel commit"). A row whose `kernel_commit` is itself
    /// `None` never matches a non-empty filter — same opt-in
    /// semantic as `project_commits`.
    ///
    /// Distinct from `project_commits` (the ktstr framework commit)
    /// and from `kernels` (the kernel release version): two runs
    /// with the same `kernel_version` but different `kernel_commit`
    /// values represent the same release rebuilt from different
    /// trees (e.g. WIP patches on top, a different remote ref).
    pub kernel_commits: Vec<String>,
    /// Repeatable run-environment-source filter, OR-combined: a row
    /// matches iff its `GauntletRow::run_source` equals ANY entry.
    /// Empty vec disables the filter ("do not filter on
    /// run_source"). A row whose `run_source` is itself `None`
    /// (sidecar pre-dates the field) never matches a non-empty
    /// filter — same opt-in semantic as `kernels` /
    /// `project_commits` / `kernel_commits`.
    /// Typical values: `"local"`, `"ci"`, `"archive"`. The schema
    /// is open: any string is acceptable so a future producer can
    /// introduce a new tag without a version bump.
    ///
    /// Field name `run_sources` (renamed from `sources`)
    /// disambiguates from `KernelMetadata.source` /
    /// [`crate::cache::KernelSource`] — those describe the kernel
    /// build's input, this describes the run-environment provenance.
    pub run_sources: Vec<String>,
    /// Repeatable scheduler-name filter, OR-combined: a row matches
    /// iff its `GauntletRow::scheduler` equals ANY entry. Empty vec
    /// disables the filter ("do not filter on scheduler"). Strict
    /// equality on each entry — the substring `-E` filter is the
    /// only fuzzy-match knob; typed flags exact-match. Mirrors the
    /// shape of `kernels` / `project_commits` / `kernel_commits` /
    /// `run_sources` so every typed dimension supports the same
    /// repeatable OR-combined idiom.
    pub schedulers: Vec<String>,
    /// Repeatable topology filter, OR-combined: a row matches iff
    /// its `GauntletRow::topology` equals ANY entry. The filter
    /// values are the rendered form (e.g. `"1n2l4c2t"`) that
    /// `Topology::Display` emits and `cargo ktstr stats list`
    /// shows. Empty vec disables the filter.
    pub topologies: Vec<String>,
    /// Repeatable work-type filter, OR-combined: a row matches iff
    /// its `GauntletRow::work_type` equals ANY entry. Valid names
    /// are the PascalCase variants of `WorkType::ALL_NAMES`. Empty
    /// vec disables the filter.
    pub work_types: Vec<String>,
    /// Repeatable flag filter, AND-combined: every entry must appear
    /// in `GauntletRow::flags`. Empty vec disables the filter.
    pub flags: Vec<String>,
}

impl RowFilter {
    /// Returns true when every populated filter field matches the
    /// row. The empty `RowFilter` (default) returns true for every
    /// row — it's the identity filter.
    pub fn matches(&self, row: &GauntletRow) -> bool {
        if !self.kernels.is_empty() {
            // OR-combined: the row matches iff its kernel version
            // matches ANY listed kernel. A row with `None`
            // kernel_version never satisfies a non-empty filter —
            // same opt-in semantic the original `Option<String>`
            // field carried.
            //
            // Match shape: a filter value with two dot-separated
            // segments (e.g. `6.12`) is a major.minor PREFIX —
            // the row matches if its `kernel_version` starts with
            // `6.12.` OR equals `6.12` exactly. A filter with three
            // or more segments (e.g. `6.14.2`, `6.15-rc3`) is
            // strict equality. The two-segment cutoff matches the
            // shape of `MAJOR.MINOR` versus `MAJOR.MINOR.PATCH` /
            // `MAJOR.MINOR-rcN` — there is no shorter form on the
            // sidecar producer side worth treating as a prefix
            // (`6` alone would match every 6.x release, which is
            // a less useful cohort than the per-stable-series
            // narrowing the operator usually wants).
            let row_kernel = row.kernel_version.as_deref();
            let any = self.kernels.iter().any(|want| match row_kernel {
                Some(rk) => kernel_filter_matches(want, rk),
                None => false,
            });
            if !any {
                return false;
            }
        }
        if !self.project_commits.is_empty() {
            // OR-combined match against `GauntletRow::commit`,
            // mirroring the `kernels` policy: a row whose `commit`
            // is `None` (the sidecar writer's gix probe failed or
            // cwd was outside any git repo) never matches a
            // populated filter, so a `--project-commit` argument is opt-in
            // to "only rows with this commit" rather than a wildcard.
            let row_commit = row.commit.as_deref();
            let any = self
                .project_commits
                .iter()
                .any(|want| row_commit == Some(want.as_str()));
            if !any {
                return false;
            }
        }
        if !self.kernel_commits.is_empty() {
            // OR-combined match against `GauntletRow::kernel_commit`,
            // mirroring the `project_commits` policy: a row whose
            // `kernel_commit` is `None` (the sidecar writer's
            // `detect_kernel_commit` probe failed, or `KTSTR_KERNEL`
            // pointed at a non-git source) never matches a populated
            // filter — same opt-in semantic as `--project-commit` /
            // `--kernel`.
            let row_kc = row.kernel_commit.as_deref();
            let any = self
                .kernel_commits
                .iter()
                .any(|want| row_kc == Some(want.as_str()));
            if !any {
                return false;
            }
        }
        if !self.run_sources.is_empty() {
            // OR-combined match against `GauntletRow::run_source`,
            // mirroring the `kernels` / `project_commits` /
            // `kernel_commits` opt-in policy: a row whose
            // `run_source` is `None` (sidecar pre-dates the field)
            // never matches a populated filter, so a `--run-source`
            // argument demands a tagged row rather than acting as a
            // wildcard.
            let row_run_source = row.run_source.as_deref();
            let any = self
                .run_sources
                .iter()
                .any(|want| row_run_source == Some(want.as_str()));
            if !any {
                return false;
            }
        }
        if !self.schedulers.is_empty() {
            // OR-combined match against `GauntletRow::scheduler`
            // (a `String`, never `None`). Strict equality on each
            // entry — same shape as the other repeatable typed
            // filters above.
            let any = self.schedulers.contains(&row.scheduler);
            if !any {
                return false;
            }
        }
        if !self.topologies.is_empty() {
            // OR-combined match against `GauntletRow::topology`.
            let any = self.topologies.contains(&row.topology);
            if !any {
                return false;
            }
        }
        if !self.work_types.is_empty() {
            // OR-combined match against `GauntletRow::work_type`.
            let any = self.work_types.contains(&row.work_type);
            if !any {
                return false;
            }
        }
        for required in &self.flags {
            if !row.flags.iter().any(|f| f == required) {
                return false;
            }
        }
        true
    }
}

/// Drop rows from `rows` that do not match every populated filter
/// field on `filter`. Returns the surviving rows in their original
/// order. The caller is responsible for any further dedup or
/// aggregation; this helper preserves duplicates as written.
///
/// Used by [`compare_partitions`] before the surviving rows reach
/// [`compare_rows`], so the substring-`-E` filter and the typed
/// filters compose: typed narrows happen first, substring runs over
/// the surviving set.
pub fn apply_row_filters(rows: &[GauntletRow], filter: &RowFilter) -> Vec<GauntletRow> {
    rows.iter().filter(|r| filter.matches(r)).cloned().collect()
}

/// Match a single `--kernel` filter value against a row's
/// `kernel_version`. Major.minor (two-segment) filter values match
/// any patch release in that series via prefix; longer filter
/// values use strict equality.
///
/// `want` is the user-supplied filter value (e.g. `6.12`,
/// `6.14.2`, `6.15-rc3`). `row_kernel` is the sidecar-recorded
/// kernel version (e.g. `6.12.5`). The two-segment cutoff matches
/// the natural shape of `MAJOR.MINOR` versus
/// `MAJOR.MINOR.PATCH` / `MAJOR.MINOR-rcN` — `6.12.` is a
/// stable-series prefix; `6.14.2` is one specific release.
///
/// Examples:
/// - `kernel_filter_matches("6.12", "6.12.5")` → true (prefix)
/// - `kernel_filter_matches("6.12", "6.12")` → true (exact equal)
/// - `kernel_filter_matches("6.12", "6.13.0")` → false
/// - `kernel_filter_matches("6.14.2", "6.14.2")` → true
/// - `kernel_filter_matches("6.14.2", "6.14.20")` → false
///   (strict equality on three-segment filter — without the
///   strict path, `6.14.2` would also match `6.14.20`,
///   `6.14.21`, ..., which is not what the operator asked for)
pub(crate) fn kernel_filter_matches(want: &str, row_kernel: &str) -> bool {
    if is_major_minor_prefix(want) {
        // Exact match OR prefix-with-trailing-dot match. The
        // trailing dot prevents `6.1` from spuriously matching
        // `6.10.0` (`6.10.0`.starts_with("6.1") is true; the
        // trailing-dot variant rejects it because `6.10` does not
        // start with `6.1.`). The exact-equal arm covers the case
        // where the row's recorded version IS the major.minor
        // string itself (no patch component).
        row_kernel == want || row_kernel.starts_with(&format!("{want}."))
    } else {
        row_kernel == want
    }
}

/// Whether a filter value looks like a major.minor PREFIX. Two
/// non-empty dot-separated digit segments and nothing else
/// (no `-rcN`, no third dot). Conservative: anything outside the
/// `MAJOR.MINOR` shape falls through to strict equality so a typo
/// like `6.14.2.` or `6.14-something` does not silently turn into
/// a wildcard.
fn is_major_minor_prefix(s: &str) -> bool {
    let parts: Vec<&str> = s.split('.').collect();
    parts.len() == 2
        && parts
            .iter()
            .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
}

/// One of the eight dimensions that compose a [`GauntletRow`]'s
/// identity in the comparison pipeline: `kernel`, `scheduler`,
/// `topology`, `work_type`, `commit`, `kernel_commit`, `source`,
/// `flags`. Each maps to the corresponding `RowFilter` field and
/// `GauntletRow` field; the dimension model lets
/// [`compare_partitions`] derive its slicing dims and dynamic
/// pairing key without hardcoding the dimension list at every
/// call site.
///
/// `scenario` is NOT a dimension — it is the test name and is
/// always part of the pairing key (you can't compare scenario A
/// against scenario B; that would compare unrelated tests).
///
/// Iteration order via [`Dimension::ALL`] is deterministic and
/// matches the order operators read in the CLI flags
/// (`--kernel` / `--scheduler` / `--topology` / `--work-type` /
/// `--project-commit` / `--kernel-commit` / `--run-source` / `--flag`), so
/// generated labels and error messages list dims in a stable,
/// predictable order.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Dimension {
    Kernel,
    Scheduler,
    Topology,
    WorkType,
    Commit,
    KernelCommit,
    Source,
    Flags,
}

impl Dimension {
    /// Every dimension in CLI-flag order. Used by
    /// [`derive_slicing_dims`] to walk the dimension space and by
    /// [`compare_partitions`] to compute the pairing-dim
    /// complement set (all dims minus slicing dims).
    pub const ALL: &'static [Dimension] = &[
        Dimension::Kernel,
        Dimension::Scheduler,
        Dimension::Topology,
        Dimension::WorkType,
        Dimension::Commit,
        Dimension::KernelCommit,
        Dimension::Source,
        Dimension::Flags,
    ];

    /// Compute pairing dims from a slicing-dim set: every
    /// dimension in [`Dimension::ALL`] that is NOT in `slicing`,
    /// in canonical order. This is the dynamic key derivation the
    /// comparison pipeline uses everywhere — slicing dims define
    /// the contrast (different on A vs B), pairing dims define
    /// the join (same across A and B).
    pub fn pairing_dims(slicing: &[Dimension]) -> Vec<Dimension> {
        Self::ALL
            .iter()
            .copied()
            .filter(|d| !slicing.contains(d))
            .collect()
    }

    /// Operator-readable name for diagnostic and table output.
    /// Matches the CLI flag suffix (e.g. `--kernel` →
    /// `"kernel"`, `--work-type` → `"work-type"`). Used in the
    /// "slicing dimensions: ..." / "pairing on: ..." header
    /// lines and in the "A and B select identical rows" error.
    pub fn name(self) -> &'static str {
        match self {
            Dimension::Kernel => "kernel",
            Dimension::Scheduler => "scheduler",
            Dimension::Topology => "topology",
            Dimension::WorkType => "work-type",
            Dimension::Commit => "commit",
            Dimension::KernelCommit => "kernel-commit",
            Dimension::Source => "source",
            Dimension::Flags => "flags",
        }
    }
}

/// Legacy pairing-dim set used by tests that pre-date the
/// dimensional-slicing refactor. Equivalent to the historical
/// hardcoded 4-tuple `(scenario, topology, work_type, flags)` —
/// scenario is always implicit in [`PairingKey::from_row`] and
/// the remaining three dimensions are listed here. Production
/// callers (`compare_partitions`) compute pairing dims via
/// [`Dimension::pairing_dims`] from the slicing-dim derivation;
/// only test fixtures use this constant directly.
pub(crate) const LEGACY_PAIRING_DIMS: &[Dimension] =
    &[Dimension::Topology, Dimension::WorkType, Dimension::Flags];

/// Derive the set of dimensions on which `filter_a` and
/// `filter_b` differ. These are the SLICING dimensions —
/// dimensions on which the two sides select disjoint cohorts and
/// therefore form the A/B contrast. The complement (every other
/// dimension) is the PAIRING-key dimension set used by
/// [`compare_rows`] to join A-side rows against B-side rows.
///
/// Comparison shape per dimension: every dim uses the same
/// SORTED-DEDUPED `Vec<&str>` comparison — order and multiplicity
/// don't matter (`--a-kernel 6.14 --a-kernel 6.15` and
/// `--b-kernel 6.15 --b-kernel 6.14` are NOT a slice). All eight
/// dimensions are now repeatable Vec filters; the previously
/// `Option<String>`-typed `scheduler` / `topology` / `work_type`
/// dims were promoted to `Vec<String>` so the operator-visible
/// shape is uniform across every dimension.
///
/// Returns dimensions in [`Dimension::ALL`] order so callers
/// (header lines, error messages, side labels) get a stable
/// presentation.
pub fn derive_slicing_dims(filter_a: &RowFilter, filter_b: &RowFilter) -> Vec<Dimension> {
    let mut out = Vec::new();
    for &dim in Dimension::ALL {
        let differs = match dim {
            Dimension::Kernel => sorted_dedup(&filter_a.kernels) != sorted_dedup(&filter_b.kernels),
            Dimension::Scheduler => {
                sorted_dedup(&filter_a.schedulers) != sorted_dedup(&filter_b.schedulers)
            }
            Dimension::Topology => {
                sorted_dedup(&filter_a.topologies) != sorted_dedup(&filter_b.topologies)
            }
            Dimension::WorkType => {
                sorted_dedup(&filter_a.work_types) != sorted_dedup(&filter_b.work_types)
            }
            Dimension::Commit => {
                sorted_dedup(&filter_a.project_commits) != sorted_dedup(&filter_b.project_commits)
            }
            Dimension::KernelCommit => {
                sorted_dedup(&filter_a.kernel_commits) != sorted_dedup(&filter_b.kernel_commits)
            }
            Dimension::Source => {
                sorted_dedup(&filter_a.run_sources) != sorted_dedup(&filter_b.run_sources)
            }
            Dimension::Flags => sorted_dedup(&filter_a.flags) != sorted_dedup(&filter_b.flags),
        };
        if differs {
            out.push(dim);
        }
    }
    out
}

fn sorted_dedup(v: &[String]) -> Vec<&str> {
    let mut s: Vec<&str> = v.iter().map(String::as_str).collect();
    s.sort_unstable();
    s.dedup();
    s
}

/// Render a side's filter values into a column-header label for
/// the comparison table. `dims` is the slicing-dimension set —
/// the only dims whose values vary between A and B. The label
/// concatenates each dim's per-side filter value(s) with `:`
/// between dim values (e.g. `"6.14.2:scx_rusty"` when both
/// `kernel` and `scheduler` slice). For multi-value Vec filters
/// (kernels, commits, flags) the values join with `|` when there
/// are ≤3; longer lists collapse to `"A"` or `"B"` (the bare
/// side label) to keep the column header readable.
///
/// `bare_label` is `"A"` / `"B"`, used as the fallback when a
/// slicing dim's filter has more than 3 values OR the slicing
/// dim's filter is empty on this side (the slice exists because
/// the OTHER side populated the filter — the empty-side label is
/// the bare letter).
pub(crate) fn render_side_label(
    filter: &RowFilter,
    dims: &[Dimension],
    bare_label: &str,
) -> String {
    if dims.is_empty() {
        return bare_label.to_string();
    }
    let mut parts: Vec<String> = Vec::new();
    for &dim in dims {
        let part = match dim {
            Dimension::Kernel => render_vec_dim(&filter.kernels, bare_label),
            Dimension::Scheduler => render_vec_dim(&filter.schedulers, bare_label),
            Dimension::Topology => render_vec_dim(&filter.topologies, bare_label),
            Dimension::WorkType => render_vec_dim(&filter.work_types, bare_label),
            Dimension::Commit => render_vec_dim(&filter.project_commits, bare_label),
            Dimension::KernelCommit => render_vec_dim(&filter.kernel_commits, bare_label),
            Dimension::Source => render_vec_dim(&filter.run_sources, bare_label),
            Dimension::Flags => render_vec_dim(&filter.flags, bare_label),
        };
        parts.push(part);
    }
    parts.join(":")
}

/// `≤3` values: join with `|`. `>3` values: collapse to
/// `bare_label`. Empty Vec: also bare label (slicing exists
/// because the OTHER side populated the same dim).
fn render_vec_dim(values: &[String], bare_label: &str) -> String {
    if values.is_empty() || values.len() > 3 {
        bare_label.to_string()
    } else {
        let mut sorted: Vec<&str> = values.iter().map(String::as_str).collect();
        sorted.sort_unstable();
        sorted.join("|")
    }
}

/// Dynamic pairing key for [`compare_rows`] — the tuple of
/// values on every NON-slicing dimension, plus the always-pinned
/// `scenario`. Two rows pair iff their dynamic keys match.
///
/// Stored as a `Vec<String>` so the same struct shape works for
/// any `pairing_dims` slice (the alternative — a tuple of
/// `Option<&str>` per dim — would force every consumer to know
/// the dim list at compile time, defeating the point of
/// dimension-set parametrisation).
///
/// First element is always `scenario`; subsequent elements
/// follow `pairing_dims` order (which is itself
/// [`Dimension::ALL`] order minus the slicing dims).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize)]
pub(crate) struct PairingKey(pub Vec<String>);

impl PairingKey {
    /// Extract the pairing key for `row` given the list of
    /// dimensions to include. The scenario is ALWAYS the first
    /// component; the `pairing_dims` list controls the rest.
    /// Each non-scenario dim contributes a single string slot:
    /// `Option<String>` fields render `None` as the empty
    /// string, `Vec<String>` fields render as a sorted-deduped
    /// `|`-joined string so the same set produces the same key
    /// regardless of input order.
    ///
    /// Commit dimensions (`Commit`, `KernelCommit`) strip the
    /// trailing `-dirty` suffix before contributing to the key.
    /// Without the strip, a clean run at HEAD `abc1234` and a
    /// dirty run at the same HEAD (`abc1234-dirty`) would shatter
    /// into two separate pairing buckets, defeating
    /// [`group_and_average_by`]'s `+mixed` cohort detection — that
    /// helper can only surface "this aggregate has both clean and
    /// dirty contributors" when the two contributors actually land
    /// in the same group. Stripping at the key level pairs them by
    /// canonical hex; the per-row `-dirty` distinction is preserved
    /// downstream in the aggregate's `commit` / `kernel_commit`
    /// field via the `+mixed` marker in
    /// [`group_and_average_by::render_mixed_dirty`].
    pub fn from_row(row: &GauntletRow, pairing_dims: &[Dimension]) -> Self {
        let mut parts = Vec::with_capacity(1 + pairing_dims.len());
        parts.push(row.scenario.clone());
        for &dim in pairing_dims {
            parts.push(match dim {
                Dimension::Kernel => row.kernel_version.clone().unwrap_or_default(),
                Dimension::Scheduler => row.scheduler.clone(),
                Dimension::Topology => row.topology.clone(),
                Dimension::WorkType => row.work_type.clone(),
                Dimension::Commit => commit_pairing_key_part(&row.commit),
                Dimension::KernelCommit => commit_pairing_key_part(&row.kernel_commit),
                Dimension::Source => row.run_source.clone().unwrap_or_default(),
                Dimension::Flags => {
                    let mut sorted: Vec<&str> = row.flags.iter().map(String::as_str).collect();
                    sorted.sort_unstable();
                    sorted.join("|")
                }
            });
        }
        PairingKey(parts)
    }
}

/// Strip the trailing `-dirty` suffix from a commit dimension's
/// value before it contributes to a [`PairingKey`]. `None` and
/// already-clean values pass through unchanged (`None` → empty
/// string; `Some("abc1234")` → `"abc1234"`); a dirty value
/// (`Some("abc1234-dirty")`) is canonicalized to `"abc1234"` so
/// it pairs with its clean sibling.
///
/// Used by [`PairingKey::from_row`] for both the `Commit` and
/// `KernelCommit` arms; the per-row `-dirty` distinction is
/// preserved separately by [`group_and_average_by`] via its
/// dirty-tracking accumulator and `+mixed` marker.
fn commit_pairing_key_part(value: &Option<String>) -> String {
    let Some(s) = value.as_deref() else {
        return String::new();
    };
    s.strip_suffix("-dirty").unwrap_or(s).to_string()
}

/// One aggregated [`GauntletRow`] produced by [`group_and_average`],
/// plus the pass-bookkeeping needed to render `N/M` in the per-group
/// summary block.
///
/// `row` carries arithmetic-mean metric values across every
/// non-failing, non-skipped contributor in the group; the
/// (`scenario`, `topology`, `work_type`, `flags`, `scheduler`,
/// `kernel_version`) identity is taken verbatim from the first
/// contributor in iteration order — every contributor in the group
/// shares the identity tuple by construction (it IS the group key
/// for the first four fields, and `scheduler` / `kernel_version`
/// are typed-filter-narrowed at the call site, so they can only
/// vary if the operator passed no `--scheduler` / `--kernel`
/// filter).
///
/// `passed` on `row` is the AND across every contributor: a single
/// failing contributor in the group flips the aggregated row to
/// `passed = false`, which routes the pair through
/// [`compare_rows`]' `skipped_failed` gate. `skipped` follows an
/// OR rule — any skipped contributor flips the aggregate to
/// skipped.
///
/// `passes_observed` and `total_observed` count the contributors:
/// `total_observed = group.len()`, `passes_observed` counts entries
/// where both `passed && !skipped`. Failing/skipped contributors do
/// NOT participate in the metric mean (they would carry
/// failure-mode telemetry, not scheduler behaviour); only passing
/// non-skipped contributors feed the running sums. When no
/// contributor passed cleanly the running sum is zero and the
/// resulting `row` carries default-zero metric values plus
/// `passed = false` — the downstream `skipped_failed` gate then
/// drops the pair from the regression math.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct AveragedGroup {
    /// Aggregated row carrying arithmetic-mean metric values plus
    /// the AND-of-contributors `passed` / OR-of-contributors
    /// `skipped` flags. Fed
    /// directly into [`compare_rows`] when `--average` is active.
    pub row: GauntletRow,
    /// Number of contributors where both `passed && !skipped`.
    /// Renders as the numerator of the per-group `N/M` summary.
    pub passes_observed: u32,
    /// Total contributors in the group (`= group.len()`). Renders
    /// as the denominator of the per-group `N/M` summary.
    pub total_observed: u32,
}

/// Per-row dirty-status update used by [`group_and_average_by`] to
/// detect when a group's contributors disagree on the `-dirty`
/// suffix for a commit dimension. `value` is `Some(hex)` /
/// `Some(hex-dirty)` / `None`; the function flips `any_clean` if
/// the value lacks the `-dirty` suffix and `any_dirty` if it
/// carries one. `first_base` records the first un-suffixed form
/// seen (used to render the `+mixed` marker against a canonical
/// hex even when `acc.first` happens to be the dirty form).
///
/// Per-row scope spans EVERY contributor (passing, failing,
/// skipped). Mixed-dirty is metadata about the cohort's working-
/// tree state, not about which contributors succeeded — surfacing
/// it only across passes would hide WIP-vs-committed disagreement
/// that the operator needs to know about. `None` values do not
/// flip either flag and do not seed `first_base`.
fn update_dirty_tracking(
    value: &Option<String>,
    any_clean: &mut bool,
    any_dirty: &mut bool,
    first_base: &mut Option<String>,
) {
    let Some(s) = value.as_deref() else { return };
    let (base, is_dirty) = match s.strip_suffix("-dirty") {
        Some(base) => (base, true),
        None => (s, false),
    };
    if is_dirty {
        *any_dirty = true;
    } else {
        *any_clean = true;
    }
    if first_base.is_none() {
        *first_base = Some(base.to_string());
    }
}

/// Render the aggregate's commit string for one dimension
/// (project_commit or kernel_commit) given the cohort-wide
/// dirty/clean tracking state. When `any_clean && any_dirty` for
/// the same un-suffixed hex, the rendered form is
/// `Some("{first_base}+mixed")`; otherwise the function returns
/// `acc.first.commit` (or `acc.first.kernel_commit`) verbatim,
/// preserving the existing first-seen behaviour for homogeneous
/// cohorts (every contributor clean, every contributor dirty, or
/// every contributor `None`).
///
/// `first_base` is the canonical un-suffixed hex captured by
/// [`update_dirty_tracking`]; using it (rather than stripping
/// `acc.first.commit`) ensures the rendered form is `abc1234+mixed`
/// regardless of whether the first contributor was clean or dirty.
fn render_mixed_dirty(
    any_clean: bool,
    any_dirty: bool,
    first_base: &Option<String>,
    first_commit: &Option<String>,
) -> Option<String> {
    if any_clean
        && any_dirty
        && let Some(base) = first_base
    {
        return Some(format!("{base}+mixed"));
    }
    first_commit.clone()
}

/// Group `rows` by `(scenario, topology, work_type, flags)` and
/// arithmetic-mean their metric fields, returning one
/// [`AveragedGroup`] per distinct key.
///
/// Group key matches [`compare_rows`]' pairing key so the post-
/// aggregation row vec joins cleanly across A/B sides under the
/// same identity contract — different flag profiles do not
/// collide.
///
/// Aggregation rules:
/// - `passed` aggregates as a logical AND across every contributor.
///   A single fail flips the aggregate to `passed = false`.
/// - `skipped` aggregates as a logical OR across every contributor:
///   any single skipped contributor flips the aggregate to
///   `skipped = true`. The OR aligns with [`compare_rows`]' skip
///   gate (one skipped side drops the pair) — averaging across a
///   mixed pass-and-skip set would silently dilute the metric mean
///   with rows that didn't run.
/// - Metrics (`f64` / `u64` / `i64` fields, plus `ext_metrics`
///   entries) are summed only across contributors where
///   `passed && !skipped`, then divided by that count to yield an
///   arithmetic mean. Failing/skipped contributors carry telemetry
///   dominated by the failure mode, NOT scheduler behaviour, and
///   are therefore excluded from the mean. When no contributor
///   passed cleanly, every metric defaults to zero and the
///   aggregate's `passed = false` routes the pair to
///   [`compare_rows`]' `skipped_failed` gate.
/// - `u64` / `i64` fields take the rounded mean
///   (`(sum / count).round() as u64`). The 0.5-unit rounding error
///   is well below every integer metric's `default_abs` gate (the
///   smallest is `stall_count = 1.0`).
/// - `ext_metrics` keys are unioned across passing contributors;
///   each key's mean is computed only across contributors that
///   carried it. A key present in some passing rows and absent
///   from others uses the present-only count as its denominator —
///   absent-and-zero are not equivalent (the `BTreeMap<String,
///   f64>` shape cannot represent "absent" with a stored zero).
/// - Identity fields (`scenario`, `topology`, `work_type`, `flags`,
///   `scheduler`, `kernel_version`) come from the first contributor
///   in iteration order. Every contributor in the group shares the
///   first four by construction (group key); `scheduler` and
///   `kernel_version` may vary across the group if the operator did
///   not narrow via typed filters first, but the aggregated row
///   carries the first contributor's value in any case — the join
///   downstream uses the four-tuple, so scheduler/version on the
///   aggregate is metadata, not a join key.
/// - Commit dimensions (`commit`, `kernel_commit`) follow a
///   first-seen rule with one exception: when contributors disagree
///   on the `-dirty` suffix for the same canonical hex (some clean,
///   some dirty), the rendered form becomes `{hex}+mixed` so the
///   working-tree disagreement is surfaced rather than hidden by
///   first-seen. `+mixed` (not `-mixed`) is intentional —
///   `-dirty` is a per-record property of one sidecar, `+mixed`
///   is a cohort-level property of the average. Mixed-dirty
///   tracking spans EVERY contributor (passing, failing, skipped)
///   because the cohort's WIP state is metadata, not a metric.
///
/// Group iteration order matches the order of FIRST appearance of
/// each key in `rows`; `BTreeMap` ordering is by key (not iteration
/// order) so we maintain a parallel `Vec<key>` to preserve
/// first-seen ordering. Stable order keeps test fixtures
/// deterministic across runs.
/// Backward-compatible wrapper that uses [`LEGACY_PAIRING_DIMS`]
/// (topology, work-type, flags). New code that participates in
/// the dimensional-slicing pipeline should call
/// [`group_and_average_by`] directly with the dynamic pairing
/// dims derived from `derive_slicing_dims`.
pub fn group_and_average(rows: &[GauntletRow]) -> Vec<AveragedGroup> {
    group_and_average_by(rows, LEGACY_PAIRING_DIMS)
}

/// Group rows by the dynamic pairing key (`scenario` plus every
/// dimension in `pairing_dims`) and return one [`AveragedGroup`]
/// per distinct key. The pairing-dim model lets the comparison
/// pipeline parametrise grouping without hardcoding a fixed
/// tuple — slicing dims are EXCLUDED from the key (rows on the
/// A/B sides differ on them by design), pairing dims are
/// INCLUDED.
pub fn group_and_average_by(
    rows: &[GauntletRow],
    pairing_dims: &[Dimension],
) -> Vec<AveragedGroup> {
    // Dynamic pairing key — scenario + every NON-slicing
    // dimension's value, in [`Dimension::ALL`] order. The
    // `PairingKey` newtype is owned (`Vec<String>`) so the
    // BTreeMap can hold keys without lifetime gymnastics; the
    // alternative — borrowing slices into `rows` — would force
    // every consumer to keep `rows` alive for the duration of
    // the map.
    type Key = PairingKey;

    struct Accumulator<'a> {
        first: &'a GauntletRow,
        total_observed: u32,
        passes_observed: u32,
        any_skipped: bool,
        any_failed: bool,
        // Tracks whether contributors disagree on the `-dirty`
        // suffix for the project_commit / kernel_commit dimensions.
        // `any_*_clean` is true if any contributor's value is the
        // un-suffixed form; `any_*_dirty` is true if any contributor
        // ends in `-dirty`. When BOTH are true the aggregate is
        // mixed-dirty and the rendered `commit` / `kernel_commit`
        // gets a `+mixed` marker so downstream readers don't see a
        // single arbitrary contributor's status. Tracked across
        // EVERY contributor (passing, failing, skipped) — a mixed
        // working-tree state is metadata about the cohort, not
        // about the metric mean. Empty / `None` values are ignored
        // and do not flip either flag.
        any_project_clean: bool,
        any_project_dirty: bool,
        any_kernel_clean: bool,
        any_kernel_dirty: bool,
        // First-seen un-suffixed (clean-form) project / kernel
        // commit string. Held separately from `first` because
        // `first.commit` may be `Some("abc1234-dirty")` when the
        // first contributor was dirty but later contributors carry
        // the clean form — the rendered `+mixed` marker should
        // still attach to the canonical un-suffixed hex so the
        // operator sees `abc1234+mixed` not `abc1234-dirty+mixed`.
        first_project_base: Option<String>,
        first_kernel_base: Option<String>,
        // Sums across passing+non-skipped contributors only.
        // Counts are tracked per ext_metric key separately because
        // a key may be absent from some contributors.
        sum_spread: f64,
        sum_gap_ms: u64,
        sum_migrations: u64,
        sum_migration_ratio: f64,
        sum_imbalance_ratio: f64,
        sum_max_dsq_depth: u64,
        sum_stall_count: usize,
        sum_fallback_count: i64,
        sum_keep_last_count: i64,
        sum_p99_wake: f64,
        sum_median_wake: f64,
        sum_wake_cv: f64,
        sum_total_iterations: u64,
        sum_mean_run_delay: f64,
        sum_run_delay: f64,
        sum_tail_ratio: f64,
        sum_iters_per_worker: f64,
        sum_page_locality: f64,
        sum_cross_node_mig: f64,
        // Per-ext-metric (sum, count) so a key absent from some
        // contributors averages only over those that carried it.
        ext_sums: BTreeMap<String, (f64, u32)>,
    }

    let mut order: Vec<Key> = Vec::new();
    let mut groups: BTreeMap<Key, Accumulator<'_>> = BTreeMap::new();

    for row in rows {
        let key = PairingKey::from_row(row, pairing_dims);
        let acc = groups.entry(key.clone()).or_insert_with(|| {
            order.push(key);
            Accumulator {
                first: row,
                total_observed: 0,
                passes_observed: 0,
                any_skipped: false,
                any_failed: false,
                any_project_clean: false,
                any_project_dirty: false,
                any_kernel_clean: false,
                any_kernel_dirty: false,
                first_project_base: None,
                first_kernel_base: None,
                sum_spread: 0.0,
                sum_gap_ms: 0,
                sum_migrations: 0,
                sum_migration_ratio: 0.0,
                sum_imbalance_ratio: 0.0,
                sum_max_dsq_depth: 0,
                sum_stall_count: 0,
                sum_fallback_count: 0,
                sum_keep_last_count: 0,
                sum_p99_wake: 0.0,
                sum_median_wake: 0.0,
                sum_wake_cv: 0.0,
                sum_total_iterations: 0,
                sum_mean_run_delay: 0.0,
                sum_run_delay: 0.0,
                sum_tail_ratio: 0.0,
                sum_iters_per_worker: 0.0,
                sum_page_locality: 0.0,
                sum_cross_node_mig: 0.0,
                ext_sums: BTreeMap::new(),
            }
        });
        acc.total_observed += 1;
        // Dirty-status tracking spans ALL contributors. Same hex
        // with mixed dirty/clean across the cohort is the case the
        // `+mixed` marker exists to surface — the per-row scope
        // (passing, failing, skipped) is irrelevant since the
        // marker describes WIP-vs-committed disagreement among the
        // contributors, not their metric outcomes.
        update_dirty_tracking(
            &row.commit,
            &mut acc.any_project_clean,
            &mut acc.any_project_dirty,
            &mut acc.first_project_base,
        );
        update_dirty_tracking(
            &row.kernel_commit,
            &mut acc.any_kernel_clean,
            &mut acc.any_kernel_dirty,
            &mut acc.first_kernel_base,
        );
        if row.skipped {
            acc.any_skipped = true;
            continue;
        }
        if !row.passed {
            acc.any_failed = true;
            continue;
        }
        acc.passes_observed += 1;
        acc.sum_spread += row.spread;
        acc.sum_gap_ms += row.gap_ms;
        acc.sum_migrations += row.migrations;
        acc.sum_migration_ratio += row.migration_ratio;
        acc.sum_imbalance_ratio += row.imbalance_ratio;
        acc.sum_max_dsq_depth += u64::from(row.max_dsq_depth);
        acc.sum_stall_count += row.stall_count;
        acc.sum_fallback_count += row.fallback_count;
        acc.sum_keep_last_count += row.keep_last_count;
        acc.sum_p99_wake += row.worst_p99_wake_latency_us;
        acc.sum_median_wake += row.worst_median_wake_latency_us;
        acc.sum_wake_cv += row.worst_wake_latency_cv;
        acc.sum_total_iterations += row.total_iterations;
        acc.sum_mean_run_delay += row.worst_mean_run_delay_us;
        acc.sum_run_delay += row.worst_run_delay_us;
        acc.sum_tail_ratio += row.worst_wake_latency_tail_ratio;
        acc.sum_iters_per_worker += row.worst_iterations_per_worker;
        acc.sum_page_locality += row.page_locality;
        acc.sum_cross_node_mig += row.cross_node_migration_ratio;
        for (k, v) in &row.ext_metrics {
            let entry = acc.ext_sums.entry(k.clone()).or_insert((0.0, 0));
            entry.0 += *v;
            entry.1 += 1;
        }
    }

    let mut out = Vec::with_capacity(order.len());
    for key in order {
        let acc = groups
            .remove(&key)
            .expect("first-seen key must still be in groups map");
        let n = acc.passes_observed;
        let denom = if n == 0 { 1.0 } else { f64::from(n) };
        // Rounded mean for integer-typed fields. When n == 0 the
        // sums are all zero, so dividing by 1.0 still yields 0 —
        // the aggregate's passed=false routes the pair through
        // skipped_failed downstream and the metrics are never
        // consulted.
        let round_u32 = |sum: u64| -> u32 {
            (sum as f64 / denom).round().clamp(0.0, f64::from(u32::MAX)) as u32
        };
        let round_u64 = |sum: u64| -> u64 { (sum as f64 / denom).round() as u64 };
        let round_i64 = |sum: i64| -> i64 { (sum as f64 / denom).round() as i64 };
        let round_usize = |sum: usize| -> usize { (sum as f64 / denom).round() as usize };

        // Mixed-dirty markers. When the cohort contains both a
        // clean-form and dirty-form contributor for the same hex
        // (e.g. some sidecars from a clean tree, others from a
        // -dirty WIP), the rendered commit field carries `+mixed`
        // appended to the canonical un-suffixed hex. The
        // alternative — taking `acc.first.commit` verbatim — would
        // hide WIP-vs-committed disagreement, presenting `abc1234`
        // when half the contributors actually came from a dirty
        // tree (or `abc1234-dirty` when half came from a clean
        // tree). Operators reading averaged stats need to know the
        // cohort spanned a working-tree state change, since that
        // changes the meaning of the metric mean. `+mixed` is the
        // chosen separator (not `-mixed`) so it cannot be confused
        // with the existing `-dirty` suffix grammar — `dirty` is a
        // per-record property, `mixed` is a cohort-level property.
        let project_commit_rendered = render_mixed_dirty(
            acc.any_project_clean,
            acc.any_project_dirty,
            &acc.first_project_base,
            &acc.first.commit,
        );
        let kernel_commit_rendered = render_mixed_dirty(
            acc.any_kernel_clean,
            acc.any_kernel_dirty,
            &acc.first_kernel_base,
            &acc.first.kernel_commit,
        );
        let aggregated = GauntletRow {
            scenario: acc.first.scenario.clone(),
            topology: acc.first.topology.clone(),
            work_type: acc.first.work_type.clone(),
            scheduler: acc.first.scheduler.clone(),
            kernel_version: acc.first.kernel_version.clone(),
            commit: project_commit_rendered,
            kernel_commit: kernel_commit_rendered,
            run_source: acc.first.run_source.clone(),
            flags: acc.first.flags.clone(),
            // ALL must pass: any failed or skipped contributor
            // flips the aggregate. A group with zero
            // passes_observed (every contributor failed or was
            // skipped) collapses to passed=false here.
            passed: !acc.any_failed && !acc.any_skipped && n > 0,
            skipped: acc.any_skipped,
            spread: acc.sum_spread / denom,
            gap_ms: round_u64(acc.sum_gap_ms),
            migrations: round_u64(acc.sum_migrations),
            migration_ratio: acc.sum_migration_ratio / denom,
            imbalance_ratio: acc.sum_imbalance_ratio / denom,
            max_dsq_depth: round_u32(acc.sum_max_dsq_depth),
            stall_count: round_usize(acc.sum_stall_count),
            fallback_count: round_i64(acc.sum_fallback_count),
            keep_last_count: round_i64(acc.sum_keep_last_count),
            worst_p99_wake_latency_us: acc.sum_p99_wake / denom,
            worst_median_wake_latency_us: acc.sum_median_wake / denom,
            worst_wake_latency_cv: acc.sum_wake_cv / denom,
            total_iterations: round_u64(acc.sum_total_iterations),
            worst_mean_run_delay_us: acc.sum_mean_run_delay / denom,
            worst_run_delay_us: acc.sum_run_delay / denom,
            worst_wake_latency_tail_ratio: acc.sum_tail_ratio / denom,
            worst_iterations_per_worker: acc.sum_iters_per_worker / denom,
            page_locality: acc.sum_page_locality / denom,
            cross_node_migration_ratio: acc.sum_cross_node_mig / denom,
            ext_metrics: acc
                .ext_sums
                .into_iter()
                .map(|(k, (sum, count))| (k, sum / f64::from(count)))
                .collect(),
        };
        out.push(AveragedGroup {
            row: aggregated,
            passes_observed: acc.passes_observed,
            total_observed: acc.total_observed,
        });
    }
    out
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
        kernel_version: sc.kernel_version.clone(),
        commit: sc.project_commit.clone(),
        kernel_commit: sc.kernel_commit.clone(),
        flags: sc.active_flags.clone(),
        run_source: sc.run_source.clone(),
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
        worst_wake_latency_tail_ratio: finite_or_zero(
            "worst_wake_latency_tail_ratio",
            sc.stats.worst_wake_latency_tail_ratio,
        ),
        worst_iterations_per_worker: finite_or_zero(
            "worst_iterations_per_worker",
            sc.stats.worst_iterations_per_worker,
        ),
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
                if crate::test_support::is_truncation_sentinel_name(k) {
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
    let median_wake_lat: Vec<f64> = rows
        .iter()
        .map(|r| r.worst_median_wake_latency_us)
        .collect();
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
fn find_worst_topos(df: &DataFrame, scenario: &str, metric: &str, threshold: f64) -> Vec<String> {
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
/// Each subdirectory is one run keyed `{kernel}-{timestamp}`. The
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

/// Pool every sidecar under the runs root (or `dir` when set) and
/// emit the distinct values present on each filterable dimension.
///
/// Eight dimensions are reported, matching the eight fields on
/// [`RowFilter`]: `kernel` (from `SidecarResult::kernel_version`),
/// `scheduler`, `topology`, `work_type`, `commit`
/// (from `SidecarResult::project_commit`), `kernel_commit` (from
/// `SidecarResult::kernel_commit`), `source` (from
/// `SidecarResult::run_source`), and `flags` (individual flag
/// names, exploded from each row's `active_flags`). The dimension
/// catalogue here matches what `cargo ktstr stats compare`
/// accepts as `--X` and `--a-X` / `--b-X` filter flags — the
/// command exists so an operator can answer "what kernel versions
/// are in the pool?" before crafting a compare invocation. The
/// JSON keys `commit` and `source` are the wire contract; the
/// corresponding per-side filter flags spell `--project-commit`
/// and `--run-source`.
///
/// `kernel_version`, `project_commit`, `kernel_commit`, and
/// `run_source` are `Option<String>` on the source sidecar;
/// absence is reported as a literal JSON `null` in the JSON
/// shape and the textual sentinel `unknown` in the table shape.
/// The set is sorted by the type's natural ordering (`BTreeSet`);
/// `None` collates before any populated value in `Option<String>`
/// ordering, so `null` / `unknown` always lands at the top of the
/// per-dimension listing.
///
/// `flags` is exploded: each entry in any row's `active_flags`
/// vector becomes a single value in the flag set. The
/// `--flag NAME` filter on `compare` matches individual flag
/// names so the discovery output mirrors the filter's input shape.
/// A scheduler that activates `["llc", "rusty_balance"]` therefore
/// contributes two distinct entries to this dimension's set.
///
/// `json=true` emits a JSON object keyed by dimension name with
/// arrays of values (with `null` interleaved for absent
/// `kernel`, `commit`, `kernel_commit`, or `source` entries —
/// the four optional dimensions); `json=false` emits a
/// per-dimension human-readable block with the values one per
/// line.
///
/// `dir` mirrors `compare_partitions` / `show_run_host` semantics:
/// when `Some(d)`, `d` replaces `runs_root()` as the pool source;
/// when `None`, `runs_root()` is used.
pub fn list_values(json: bool, dir: Option<&std::path::Path>) -> anyhow::Result<String> {
    use std::collections::BTreeSet;

    let (root, override_archive) = match dir {
        Some(d) => (d.to_path_buf(), true),
        None => (crate::test_support::runs_root(), false),
    };
    let mut pool = crate::test_support::collect_pool(&root);
    if override_archive {
        // `--dir` points at a non-default pool root. Stats tooling
        // treats those sidecars as `"archive"` regardless of the
        // tag they were written with — see
        // `apply_archive_source_override` for the rewrite contract.
        crate::test_support::apply_archive_source_override(&mut pool);
    }

    let mut kernels: BTreeSet<Option<String>> = BTreeSet::new();
    let mut project_commits: BTreeSet<Option<String>> = BTreeSet::new();
    let mut kernel_commits: BTreeSet<Option<String>> = BTreeSet::new();
    let mut run_sources: BTreeSet<Option<String>> = BTreeSet::new();
    let mut schedulers: BTreeSet<String> = BTreeSet::new();
    let mut topologies: BTreeSet<String> = BTreeSet::new();
    let mut work_types: BTreeSet<String> = BTreeSet::new();
    let mut flags: BTreeSet<String> = BTreeSet::new();

    for sc in &pool {
        kernels.insert(sc.kernel_version.clone());
        project_commits.insert(sc.project_commit.clone());
        kernel_commits.insert(sc.kernel_commit.clone());
        run_sources.insert(sc.run_source.clone());
        schedulers.insert(sc.scheduler.clone());
        topologies.insert(sc.topology.clone());
        work_types.insert(sc.work_type.clone());
        for f in &sc.active_flags {
            flags.insert(f.clone());
        }
    }

    if json {
        let kernels_json: Vec<serde_json::Value> = kernels
            .iter()
            .map(|opt| match opt {
                Some(s) => serde_json::Value::String(s.clone()),
                None => serde_json::Value::Null,
            })
            .collect();
        let project_commits_json: Vec<serde_json::Value> = project_commits
            .iter()
            .map(|opt| match opt {
                Some(s) => serde_json::Value::String(s.clone()),
                None => serde_json::Value::Null,
            })
            .collect();
        let kernel_commits_json: Vec<serde_json::Value> = kernel_commits
            .iter()
            .map(|opt| match opt {
                Some(s) => serde_json::Value::String(s.clone()),
                None => serde_json::Value::Null,
            })
            .collect();
        let run_sources_json: Vec<serde_json::Value> = run_sources
            .iter()
            .map(|opt| match opt {
                Some(s) => serde_json::Value::String(s.clone()),
                None => serde_json::Value::Null,
            })
            .collect();
        // JSON keys stay as `commit` / `source` — operator-visible
        // wire contract for `cargo ktstr stats list-values --json`
        // does not rename when the internal field/variable does.
        // Note: the per-side filter flags on `compare` spell as
        // `--project-commit` / `--run-source` (longer-form
        // disambiguating names), so the JSON keys here intentionally
        // diverge from the CLI flag names. The wire contract is the
        // shorter form because that's what every external consumer
        // (CI scripts, archive readers) has been parsing since the
        // sidecar format was first introduced.
        let payload = serde_json::json!({
            "kernel": kernels_json,
            "commit": project_commits_json,
            "kernel_commit": kernel_commits_json,
            "source": run_sources_json,
            "scheduler": schedulers.iter().collect::<Vec<_>>(),
            "topology": topologies.iter().collect::<Vec<_>>(),
            "work_type": work_types.iter().collect::<Vec<_>>(),
            "flags": flags.iter().collect::<Vec<_>>(),
        });
        return serde_json::to_string_pretty(&payload)
            .map(|mut s| {
                s.push('\n');
                s
            })
            .map_err(|e| anyhow::anyhow!("serialize list-values JSON: {e}"));
    }

    let mut out = String::new();
    let render_opt_set = |out: &mut String, label: &str, set: &BTreeSet<Option<String>>| {
        out.push_str(label);
        out.push('\n');
        if set.is_empty() {
            out.push_str("  (no sidecars in pool)\n");
        } else {
            for opt in set {
                match opt {
                    Some(s) => {
                        out.push_str("  ");
                        out.push_str(s);
                        out.push('\n');
                    }
                    None => out.push_str("  unknown\n"),
                }
            }
        }
        out.push('\n');
    };
    let render_str_set = |out: &mut String, label: &str, set: &BTreeSet<String>| {
        out.push_str(label);
        out.push('\n');
        if set.is_empty() {
            out.push_str("  (no sidecars in pool)\n");
        } else {
            for s in set {
                out.push_str("  ");
                out.push_str(s);
                out.push('\n');
            }
        }
        out.push('\n');
    };
    render_opt_set(&mut out, "kernel:", &kernels);
    render_opt_set(&mut out, "commit:", &project_commits);
    render_opt_set(&mut out, "kernel_commit:", &kernel_commits);
    render_opt_set(&mut out, "source:", &run_sources);
    render_str_set(&mut out, "scheduler:", &schedulers);
    render_str_set(&mut out, "topology:", &topologies);
    render_str_set(&mut out, "work_type:", &work_types);
    render_str_set(&mut out, "flags:", &flags);
    Ok(out)
}

/// One significant per-metric finding produced by [`compare_rows`].
///
/// `pairing_key` carries the dynamic identity the row pair joined
/// on — `scenario` plus every NON-slicing dimension's value. The
/// table renderer in [`compare_partitions`] decodes the key against
/// the slicing-dim list to produce a label like
/// `scenario/topology/work_type` (when topology + work_type are
/// pairing dims) or just `scenario` (when every other dim slices).
///
/// The `scenario` / `topology` / `work_type` fields carry the
/// matched row's values verbatim for legacy-shape consumers and
/// test fixtures that pre-date the dimensional-slicing refactor.
/// New code should read [`Finding::pairing_key`] directly so the
/// slicing-dim variation stays visible.
///
/// `metric` is the registry entry the comparison ran against;
/// consumers read polarity, display unit, and name through it
/// directly without re-looking up [`metric_def`].
#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct Finding {
    pub pairing_key: PairingKey,
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

/// Per-metric threshold policy driving [`compare_rows`] /
/// [`compare_partitions`].
///
/// Resolution priority for a given metric's relative significance
/// threshold, highest first:
///
/// 1. `per_metric_percent[metric_name]` — explicit override for
///    this metric.
/// 2. `default_percent` — uniform override across every metric
///    not listed in the map (equivalent to the old `--threshold N`
///    CLI flag).
/// 3. The metric's built-in `default_rel` from the [`METRICS`]
///    registry — the "no policy" fallback.
///
/// Values in the struct are stored as PERCENT (e.g. `10.0` meaning
/// 10%), NOT fractions. [`Self::rel_threshold`] does the `/100.0`
/// conversion so every caller inside `compare_rows` reads a
/// fraction without re-deriving the division.
///
/// Note on the registry-fallback branch: the `default_rel` field
/// on `MetricDef` is already a FRACTION (e.g. `0.25` for 25%),
/// not a percent. `rel_threshold` returns it verbatim — it
/// does NOT divide by 100. Only the override branches
/// (per-metric map, `default_percent`) do the percent-to-fraction
/// conversion because their inputs are percents. This asymmetry
/// is deliberate so callers supplying CLI/file-based overrides
/// work in human-intuitive percent units while the registry
/// defaults (which already ship in fraction form) pass through
/// unchanged.
///
/// The struct is `serde::Serialize` / `serde::Deserialize` so
/// `cargo ktstr stats compare --policy <path>` can load a
/// JSON-persisted policy file. Default construction produces an
/// empty policy that uses every registry default; [`Self::uniform`]
/// reproduces the old `--threshold N` behaviour without any
/// per-metric override plumbing at the call site.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ComparisonPolicy {
    /// Uniform override: when `Some(p)`, every metric whose name is
    /// NOT in [`Self::per_metric_percent`] uses `p / 100.0` as its
    /// relative threshold. `None` falls through to the registry
    /// `default_rel`. Stored as percent (e.g. `10.0` for 10%).
    pub default_percent: Option<f64>,
    /// Per-metric overrides keyed by metric name. Each value is a
    /// percent (e.g. `15.0` → 15%). An entry here takes precedence
    /// over both [`Self::default_percent`] and the registry
    /// `default_rel`.
    pub per_metric_percent: BTreeMap<String, f64>,
}

impl ComparisonPolicy {
    /// Empty policy — every metric uses its [`METRICS`] registry
    /// default. Equivalent to the old `--threshold None` CLI path.
    pub fn new() -> Self {
        Self::default()
    }

    /// Uniform override: every metric uses `percent / 100.0`.
    /// Mirrors the old `--threshold N` CLI behaviour; the CLI
    /// dispatch at `cargo-ktstr stats compare --threshold N`
    /// constructs a policy via this constructor.
    pub fn uniform(percent: f64) -> Self {
        Self {
            default_percent: Some(percent),
            per_metric_percent: BTreeMap::new(),
        }
    }

    /// Load a JSON-persisted policy from a file. Errors propagate
    /// the read / parse reason as an `anyhow::Error` with the file
    /// path in the context chain so a malformed `--policy path.json`
    /// surfaces an actionable message rather than a generic
    /// "invalid JSON."
    ///
    /// Validates after parsing via [`Self::validate`]: rejects
    /// negative thresholds (a misconfigured 10 vs -10 would
    /// invert the dual-gate logic at the `.abs() >= rel_thresh`
    /// check and silently classify every metric as significant)
    /// and rejects per-metric keys not registered in [`METRICS`]
    /// (a typo like `"wrost_spread"` would otherwise be silently
    /// ignored — the key simply never matches during resolution
    /// and the metric falls through to `default_percent`).
    pub fn load_json(path: &std::path::Path) -> anyhow::Result<Self> {
        use anyhow::Context;
        let data = std::fs::read_to_string(path)
            .with_context(|| format!("read comparison policy from {}", path.display()))?;
        let policy: ComparisonPolicy = serde_json::from_str(&data)
            .with_context(|| format!("parse comparison policy from {}", path.display()))?;
        policy
            .validate()
            .with_context(|| format!("validate comparison policy from {}", path.display()))?;
        Ok(policy)
    }

    /// Structural validation separate from parsing so both the
    /// `load_json` path and programmatic constructors (after
    /// [`Self::uniform`] with a user-supplied percent) can share
    /// one set of invariants without re-implementing checks at
    /// each call site. Called automatically by [`Self::load_json`];
    /// CLI dispatch should call it after constructing via
    /// [`Self::uniform`] to catch `--threshold -10` at the
    /// entry point rather than deep inside `compare_rows` where
    /// the dual-gate math silently misbehaves.
    ///
    /// Rejects:
    /// - Negative `default_percent` (nonsensical — thresholds are
    ///   absolute-value comparisons).
    /// - Negative entries in `per_metric_percent`.
    /// - Per-metric keys not in the [`METRICS`] registry (silent
    ///   typos would otherwise fall through to `default_percent`
    ///   unnoticed).
    pub fn validate(&self) -> anyhow::Result<()> {
        if let Some(p) = self.default_percent
            && p < 0.0
        {
            anyhow::bail!(
                "ComparisonPolicy: default_percent must be non-negative; got {p}. \
                 Thresholds are absolute-value comparisons — a negative value \
                 would invert the dual-gate logic and silently classify every \
                 delta as significant."
            );
        }
        for (name, p) in &self.per_metric_percent {
            if !METRICS.iter().any(|m| m.name == name) {
                let known: Vec<&str> = METRICS.iter().map(|m| m.name).collect();
                anyhow::bail!(
                    "ComparisonPolicy: per_metric_percent contains unknown \
                     metric `{name}`. A typo in the key would silently fall \
                     through to default_percent. Registered metrics: {}",
                    known.join(", "),
                );
            }
            if *p < 0.0 {
                anyhow::bail!(
                    "ComparisonPolicy: per_metric_percent[{name:?}] must be \
                     non-negative; got {p}",
                );
            }
        }
        Ok(())
    }

    /// Resolve the relative threshold (as a fraction, e.g. `0.10`
    /// for 10%) for `metric_name` with `default_rel` as the
    /// registry-level fallback. Handles the percent→fraction
    /// conversion so [`compare_rows`] does not need to re-derive
    /// `p / 100.0` at every call site.
    pub fn rel_threshold(&self, metric_name: &str, default_rel: f64) -> f64 {
        if let Some(p) = self.per_metric_percent.get(metric_name) {
            p / 100.0
        } else if let Some(p) = self.default_percent {
            p / 100.0
        } else {
            default_rel
        }
    }
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
/// `policy` carries the comparison thresholds. See
/// [`ComparisonPolicy`] for the resolution rules — per-metric
/// override → `default_percent` → registry `default_rel`. The
/// absolute gate always uses the metric's `default_abs`. A delta
/// must clear both gates to count as significant.
/// Backward-compatible wrapper that uses [`LEGACY_PAIRING_DIMS`]
/// (topology, work-type, flags) for pairing. New code that
/// participates in the dimensional-slicing pipeline should call
/// [`compare_rows_by`] with the dynamic pairing dims derived
/// from `derive_slicing_dims`. Production callers all route
/// through [`compare_partitions`] which calls `_by` directly;
/// only test fixtures still call this wrapper, and they keep it
/// alive without a `cfg(test)` gate so the API surface stays
/// uniform between debug and release builds.
#[allow(dead_code)]
pub(crate) fn compare_rows(
    rows_a: &[GauntletRow],
    rows_b: &[GauntletRow],
    filter: Option<&str>,
    policy: &ComparisonPolicy,
) -> CompareReport {
    compare_rows_by(rows_a, rows_b, LEGACY_PAIRING_DIMS, filter, policy)
}

/// Pair-by-key comparison parametrised on `pairing_dims`. Two
/// rows pair iff their [`PairingKey`] (scenario + every value
/// for each dimension in `pairing_dims`) is equal. This is the
/// dimensional-slicing pipeline's join primitive — the slicing
/// dims are EXCLUDED from `pairing_dims` so rows on the A/B
/// sides that differ on those dims still pair as long as they
/// agree on every non-slicing dim.
pub(crate) fn compare_rows_by(
    rows_a: &[GauntletRow],
    rows_b: &[GauntletRow],
    pairing_dims: &[Dimension],
    filter: Option<&str>,
    policy: &ComparisonPolicy,
) -> CompareReport {
    let mut report = CompareReport::default();

    for row_b in rows_b {
        // Dynamic pairing key: scenario + every NON-slicing
        // dimension's value. Two rows pair iff their dynamic keys
        // match. The flag-set component is sorted-deduped inside
        // `PairingKey::from_row` so order-of-accumulation noise
        // doesn't shatter pairs (matching the canonicalize-on-write
        // contract documented on `canonicalize_active_flags`).
        let key_b = PairingKey::from_row(row_b, pairing_dims);
        if let Some(f) = filter {
            // Substring filter joins all identity-bearing fields —
            // including the SLICING dim values — so an operator
            // can narrow by any visible field via `-E`. The
            // canonical `flags` rendering matches what the table
            // shows.
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
        let row_a = rows_a
            .iter()
            .find(|r| PairingKey::from_row(r, pairing_dims) == key_b);
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

            let rel_thresh = policy.rel_threshold(m.name, m.default_rel);

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
                pairing_key: key_b.clone(),
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
        let key_a = PairingKey::from_row(row_a, pairing_dims);
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
        let exists_in_b = rows_b
            .iter()
            .any(|r| PairingKey::from_row(r, pairing_dims) == key_a);
        if !exists_in_b {
            report.removed_from_a += 1;
        }
    }

    report
}

/// Emit a stderr warning naming any `-dirty` commit values present
/// in the partitioned rows so the operator knows the comparison
/// includes builds whose source tree may not match the recorded
/// HEAD.
///
/// Scans `commit` (project HEAD) and `kernel_commit` (kernel source
/// tree HEAD) on both sides' rows, dedupes the surviving values,
/// and emits one warning block listing each distinct dirty value
/// per dimension. Emits at most one block — silent when no row
/// carries a `-dirty` suffix on either dimension.
///
/// Dirty runs reuse the same sidecar filename as their clean HEAD
/// (the variant hash excludes `commit` / `kernel_commit` per
/// [`crate::test_support::sidecar`]), so re-running the same test
/// from a dirty tree overwrites the previous record. The warning
/// surfaces this so an operator can decide whether to commit the
/// working tree before re-running for a reproducible comparison.
///
/// Splits collection from emission via [`render_dirty_warning`] so
/// unit tests can pin the rendered text without trapping `stderr`.
fn warn_on_dirty_builds(rows_a: &[GauntletRow], rows_b: &[GauntletRow]) {
    if let Some(text) = render_dirty_warning(rows_a, rows_b) {
        eprint!("{text}");
    }
}

/// Build the dirty-builds warning block from row data.
///
/// Returns `None` when no row on either side carries a `-dirty`
/// suffix on either `commit` or `kernel_commit`. Otherwise returns
/// the full multi-line warning text — the body emitted to stderr by
/// [`warn_on_dirty_builds`] — terminated with a trailing newline so
/// the caller can `eprint!` it without further formatting.
///
/// Dimensions render in fixed order ("kernel source" before
/// "project") so the same dirty hashes always produce byte-identical
/// output across runs; values within each dimension are
/// `BTreeSet`-deduped so multiple rows sharing one dirty hash list
/// it once, and multiple distinct dirty hashes on one dimension list
/// in lex order.
fn render_dirty_warning(rows_a: &[GauntletRow], rows_b: &[GauntletRow]) -> Option<String> {
    use std::collections::BTreeSet;
    use std::fmt::Write;

    let mut dirty_kernel: BTreeSet<&str> = BTreeSet::new();
    let mut dirty_project: BTreeSet<&str> = BTreeSet::new();
    for row in rows_a.iter().chain(rows_b.iter()) {
        // `ends_with` matches the producer contract: `detect_kernel_commit`
        // and `detect_project_commit` (sidecar.rs:851, :983) append
        // `-dirty` as a SUFFIX to the 7-char hex via
        // `format!("{short_hash}-dirty")`, so the dirty marker is
        // always tail-positioned. `contains` would also match a
        // hex hash that legitimately contains the substring `-dirty`
        // somewhere in the middle (impossible for the current
        // 7-char hex prefix, but a future commit-ish format change
        // would let a non-dirty value flag itself dirty under
        // `contains`).
        if let Some(c) = row.kernel_commit.as_deref()
            && c.ends_with("-dirty")
        {
            dirty_kernel.insert(c);
        }
        if let Some(c) = row.commit.as_deref()
            && c.ends_with("-dirty")
        {
            dirty_project.insert(c);
        }
    }

    if dirty_kernel.is_empty() && dirty_project.is_empty() {
        return None;
    }

    let mut out = String::new();
    writeln!(out, "warning: comparison includes dirty builds:").unwrap();
    for v in &dirty_kernel {
        writeln!(
            out,
            "  - kernel source: {v} (working tree may have changed since this run)"
        )
        .unwrap();
    }
    for v in &dirty_project {
        writeln!(
            out,
            "  - project: {v} (working tree may have changed since this run)"
        )
        .unwrap();
    }
    writeln!(
        out,
        "  Dirty runs overwrite previous results with the same HEAD."
    )
    .unwrap();
    writeln!(out, "  Commit changes for reproducible-ish comparisons.").unwrap();
    Some(out)
}

/// Render the actionable bail message emitted when one side's filter
/// matches zero sidecars in the pool.
///
/// Beyond the generic "check filters / run `cargo ktstr stats list`"
/// redirect, this helper inspects WHY the filter matched nothing and
/// adds three operator-actionable hints when applicable:
///
/// 1. **Dirty-form hint**: when the user passed
///    `--project-commit X` (or per-side / kernel-commit equivalent)
///    and the pool contains a row whose `commit` (or `kernel_commit`)
///    is `X-dirty`, append "Did you mean `--project-commit X-dirty`?".
///    A clean-vs-dirty mismatch is the single most common cause of a
///    false-zero on the commit dims — `detect_project_commit` /
///    `detect_kernel_commit` append `-dirty` whenever HEAD-vs-index
///    or index-vs-worktree changes are observed, so an operator who
///    expected `abcdef1` but the recorded value is `abcdef1-dirty`
///    sees no rows match without realizing why.
///
/// 2. **Unknown run-source hint**: when the user passed
///    `--run-source X` (or per-side equivalent) and `X` is NOT
///    among the distinct `run_source` values present in the pool,
///    append a hint listing the actual values seen. The schema is
///    deliberately extensible (`"benchmark"` and other future tags
///    are valid), so this is a hint rather than a hard validator —
///    but a typo (`--run-source loca` for `local`, or `--run-source CI`
///    for `ci` since the values are case-sensitive) is the most
///    common cause of a false-zero on the source dim, and listing
///    the distinct values present is more actionable than asking
///    the operator to consult the schema doc.
///
/// 3. **list-values redirect for commit dims**: when the user
///    populated any commit dimension (`project_commits` /
///    `kernel_commits`), suggest `cargo ktstr stats list-values`
///    specifically — that command emits the exact distinct values
///    present per dimension, which is more actionable than the
///    generic `stats list` which only shows top-level run keys.
///
/// `side` is `"A"` or `"B"` for diagnostic context. `filter` is the
/// per-side `RowFilter`. `rows` is the sidecar-derived row vec
/// (post-`sidecar_to_row` mapping, pre-filtering). `pool_len` is
/// the raw pool count for the "(N pooled)" diagnostic context.
fn zero_match_diagnostic(
    side: &str,
    filter: &RowFilter,
    rows: &[GauntletRow],
    pool_len: usize,
) -> String {
    let mut msg = format!(
        "stats compare: {side} side filter matched 0 sidecars in \
         pool ({pool_len} pooled). Check the per-side filters or \
         confirm the runs exist with `cargo ktstr stats list`."
    );

    // Dirty-form hint per commit dimension. Only fires when a
    // populated filter value's `-dirty` form is in the pool.
    let mut dirty_hints: Vec<String> = Vec::new();
    for want in &filter.project_commits {
        let dirty = format!("{want}-dirty");
        let found = rows
            .iter()
            .any(|r| r.commit.as_deref() == Some(dirty.as_str()));
        if found {
            dirty_hints.push(format!(
                "no rows match `--project-commit {want}` but `{dirty}` exists in the pool — \
                 did you mean `--project-commit {dirty}`?"
            ));
        }
    }
    for want in &filter.kernel_commits {
        let dirty = format!("{want}-dirty");
        let found = rows
            .iter()
            .any(|r| r.kernel_commit.as_deref() == Some(dirty.as_str()));
        if found {
            dirty_hints.push(format!(
                "no rows match `--kernel-commit {want}` but `{dirty}` exists in the pool — \
                 did you mean `--kernel-commit {dirty}`?"
            ));
        }
    }
    for hint in dirty_hints {
        msg.push_str("\nhint: ");
        msg.push_str(&hint);
    }

    // Unknown-run-source hint. Fires when a `--run-source X` value
    // is not present in the pool — typo / wrong casing is the most
    // common cause. Schema is intentionally extensible (operators
    // can write `"benchmark"` etc.), so this is a hint not a hard
    // validator: the bail still fires, the operator still sees the
    // distinct values present, and the producer side is free to
    // emit any tag.
    if !filter.run_sources.is_empty() {
        let pool_run_sources: std::collections::BTreeSet<&str> = rows
            .iter()
            .filter_map(|r| r.run_source.as_deref())
            .collect();
        let unknowns: Vec<&str> = filter
            .run_sources
            .iter()
            .map(String::as_str)
            .filter(|want| !pool_run_sources.contains(*want))
            .collect();
        if !unknowns.is_empty() {
            let mut present: Vec<&str> = pool_run_sources.iter().copied().collect();
            present.sort_unstable();
            let unknown_list = unknowns
                .iter()
                .map(|s| format!("`{s}`"))
                .collect::<Vec<_>>()
                .join(", ");
            let present_list = if present.is_empty() {
                "(none — every row has `run_source: null`)".to_string()
            } else {
                present
                    .iter()
                    .map(|s| format!("`{s}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            msg.push_str(&format!(
                "\nhint: --run-source {unknown_list} not found in pool; \
                 distinct values present: {present_list}. Values are \
                 case-sensitive (`ci` ≠ `CI`)."
            ));
        }
    }

    // list-values redirect: only fires when the operator narrowed
    // on a commit dimension. Generic case (no commit filter) keeps
    // the existing `stats list` redirect at the top of the message
    // — `list-values` would emit a long per-dimension dump that
    // isn't more actionable than `stats list` for a kernel/scheduler
    // /topology miss.
    let touched_commit_dim =
        !filter.project_commits.is_empty() || !filter.kernel_commits.is_empty();
    if touched_commit_dim {
        msg.push_str(
            "\nhint: run `cargo ktstr stats list-values` to see every \
             distinct commit value present in the pool — the specific \
             value the filter expected may not have a sidecar yet, or \
             may differ from what was recorded by \
             `detect_project_commit` / `detect_kernel_commit`.",
        );
    }
    msg
}

/// Compare two filter-defined partitions of the sidecar pool and
/// report regressions across slicing dimensions.
///
/// `filter_a` and `filter_b` are the per-side row filters that
/// define the A/B contrast. The dimensions on which the two
/// filters DIFFER are the SLICING dimensions; the dimensions on
/// which they AGREE (or on which both are unconstrained) are the
/// PAIRING dimensions. Two rows pair across the A/B sides iff
/// their dynamic [`PairingKey`] (scenario plus every pairing-dim
/// value) is equal — so the comparison naturally ignores
/// differences on the slicing axes (those ARE the contrast) and
/// joins on everything else.
///
/// `dir` overrides the default `runs_root()` for pool collection.
/// Pass `Some(path)` to compare archived sidecar trees copied off
/// a CI host; pass `None` to walk `target/ktstr/` (or
/// `CARGO_TARGET_DIR/ktstr/`).
///
/// Validation:
/// - Empty slicing-dim set (every dimension is identical between
///   A and B): bail with "specify at least one --a-X / --b-X to
///   define what to compare". This includes the no-flags-at-all
///   case (both filters are the empty default).
/// - Identical effective filters with at least one slicing dim is
///   a contradiction caught by clap-level construction; the
///   downstream check is "every value in filter_a appears in
///   filter_b on the same dim and vice versa." We catch that as
///   "A and B select identical rows" — symmetric to the empty
///   case.
/// - More than one slicing dimension prints a warning to stderr
///   ("warning: slicing on N dimensions; results compress
///   multiple axes into a single A/B contrast") but does NOT
///   bail — multi-dim slicing is a deliberate feature for
///   comparing e.g. (kernel A + scheduler A) against (kernel B +
///   scheduler B).
///
/// `no_average = false` (the default) groups every matching
/// sidecar within each side by pairing key and averages the
/// metrics across the group. `no_average = true` keeps each
/// sidecar row distinct; if multiple rows on one side share the
/// same pairing key the function bails with an actionable
/// "duplicate pairing keys" error rather than picking one
/// arbitrarily.
///
/// Returns 0 on no regressions, 1 if regressions detected.
pub fn compare_partitions(
    filter_a: &RowFilter,
    filter_b: &RowFilter,
    filter: Option<&str>,
    policy: &ComparisonPolicy,
    dir: Option<&std::path::Path>,
    no_average: bool,
) -> anyhow::Result<i32> {
    // Validation gate 1: there must be at least one dimension
    // on which filter_a differs from filter_b — otherwise the
    // operator hasn't expressed a contrast and the function has
    // nothing to compare. Empty slicing dims OR identical filters
    // are both rejected here with actionable diagnostics so the
    // user knows which knob to turn.
    let slicing_dims = derive_slicing_dims(filter_a, filter_b);
    if slicing_dims.is_empty() {
        anyhow::bail!(
            "stats compare: A and B select identical rows. \
             Specify at least one per-side filter (e.g. \
             --a-kernel 6.14 --b-kernel 6.15) to define what \
             dimension separates the two sides."
        );
    }

    // Validation gate 2: warn (not error) when slicing on
    // multiple dimensions. The result is still well-defined —
    // the comparison joins on remaining pairing dims and
    // collapses the slicing-dim cross-product into a single
    // A/B contrast — but the operator is asking for a multi-axis
    // delta which is harder to interpret. The warning surfaces
    // the dim list so they can confirm the cohort shape.
    if slicing_dims.len() > 1 {
        let dim_names: Vec<&str> = slicing_dims.iter().map(|d| d.name()).collect();
        eprintln!(
            "warning: stats compare: slicing on {n} dimensions [{dims}]; \
             results compress multiple axes into a single A/B contrast.",
            n = slicing_dims.len(),
            dims = dim_names.join(", "),
        );
    }

    // Pairing dims = every dimension NOT in the slicing-dim set,
    // in canonical [`Dimension::ALL`] order. The dynamic key
    // shape `(scenario, *pairing_dims)` matches whatever
    // dimensions are currently NOT being contrasted across A
    // and B.
    let pairing_dims = Dimension::pairing_dims(&slicing_dims);

    // Pool every sidecar under the runs root (or the operator's
    // --dir override) and convert to rows. The full-scan cost
    // is acceptable for the single-comparison-per-session
    // workflow.
    //
    // `--dir`-loaded sidecars get their `source` field rewritten
    // to `"archive"` via `apply_archive_source_override` before
    // row conversion. The producer-side `"local"` / `"ci"`
    // distinction is meaningful on the host that wrote the
    // sidecars; once the files have been copied off, the only
    // useful classification is "this came from elsewhere", which
    // is what `--run-source archive` queries for. Operators who need
    // to retain the producer-side distinction read from the
    // default root (no `--dir`) so values pass through untouched.
    let (root, override_archive) = match dir {
        Some(d) => (d.to_path_buf(), true),
        None => (crate::test_support::runs_root(), false),
    };
    let mut pool = crate::test_support::collect_pool(&root);
    if override_archive {
        crate::test_support::apply_archive_source_override(&mut pool);
    }
    if pool.is_empty() {
        anyhow::bail!(
            "stats compare: no sidecar data found under {}. \
             Run `cargo ktstr test` to generate runs, or pass \
             --dir to point at an archived sidecar tree.",
            root.display(),
        );
    }
    let rows: Vec<GauntletRow> = pool.iter().map(sidecar_to_row).collect();

    // Partition: apply each side's filter to the same pool. A
    // row may match both sides (e.g. when scheduler is the
    // slicing dim and kernel is unconstrained on both, a row
    // whose `scheduler` is in `filter_a.schedulers` matches A
    // but NOT B unless `filter_b.schedulers` also contains it —
    // typically not when scheduler is the slicing axis).
    let rows_a = apply_row_filters(&rows, filter_a);
    let rows_b = apply_row_filters(&rows, filter_b);
    if rows_a.is_empty() {
        anyhow::bail!(
            "{}",
            zero_match_diagnostic("A", filter_a, &rows, pool.len()),
        );
    }
    if rows_b.is_empty() {
        anyhow::bail!(
            "{}",
            zero_match_diagnostic("B", filter_b, &rows, pool.len()),
        );
    }

    warn_on_dirty_builds(&rows_a, &rows_b);

    let pre_agg_a = rows_a.len();
    let pre_agg_b = rows_b.len();

    // Average by default: fold same-pairing-key rows on each
    // side into one mean row. `--no-average` keeps every
    // sidecar distinct but still rejects duplicate pairing keys
    // because compare_rows can't pair an A-row against multiple
    // B-rows with the same key.
    let (rows_a_for_compare, rows_b_for_compare, avg_a, avg_b) = if !no_average {
        let avg_a = group_and_average_by(&rows_a, &pairing_dims);
        let avg_b = group_and_average_by(&rows_b, &pairing_dims);
        let a_rows: Vec<GauntletRow> = avg_a.iter().map(|r| r.row.clone()).collect();
        let b_rows: Vec<GauntletRow> = avg_b.iter().map(|r| r.row.clone()).collect();
        (a_rows, b_rows, Some(avg_a), Some(avg_b))
    } else {
        // Detect duplicates manually so the error names the key
        // rather than letting compare_rows silently latch onto
        // the first match.
        check_no_duplicate_pairing_keys(&rows_a, &pairing_dims, "A")?;
        check_no_duplicate_pairing_keys(&rows_b, &pairing_dims, "B")?;
        (rows_a, rows_b, None, None)
    };

    let report = compare_rows_by(
        &rows_a_for_compare,
        &rows_b_for_compare,
        &pairing_dims,
        filter,
        policy,
    );

    // Side labels derive from the slicing dims' filter values.
    // Single slicing dim: e.g. "6.14.2" / "6.15.0". Multi: e.g.
    // "6.14.2:scx_rusty" / "6.15.0:scx_alpha". >3 values per dim:
    // collapse to "A"/"B" to keep column headers readable.
    let label_a = render_side_label(filter_a, &slicing_dims, "A");
    let label_b = render_side_label(filter_b, &slicing_dims, "B");

    // Header lines: name the slicing and pairing axes so the
    // operator can confirm the comparison shape at a glance.
    let slice_names: Vec<&str> = slicing_dims.iter().map(|d| d.name()).collect();
    let pair_names: Vec<&str> = pairing_dims.iter().map(|d| d.name()).collect();
    println!("slicing dimensions: {}", slice_names.join(", "));
    println!(
        "pairing on: scenario{}{}",
        if pair_names.is_empty() { "" } else { ", " },
        pair_names.join(", "),
    );

    if !no_average {
        println!(
            "{}",
            format_average_header(pre_agg_a, pre_agg_b, &label_a, &label_b)
        );
    }

    use comfy_table::{Cell, Color};
    let mut table = crate::cli::new_table();
    table.set_header(vec![
        "TEST", "METRIC", &label_a, &label_b, "DELTA", "VERDICT",
    ]);
    for f in &report.findings {
        let (verdict_text, verdict_color) = if f.is_regression {
            ("REGRESSION", Color::Red)
        } else {
            ("improvement", Color::Green)
        };
        // PairingKey's first slot is scenario; subsequent slots
        // are the pairing-dim values in canonical order. Joining
        // with `/` produces a label whose shape mirrors the
        // pairing-dim count — so a comparison that pairs on
        // (topology, work_type, flags) renders the historical
        // `scenario/topology/work_type/flags` label, while a
        // comparison that slices on most dims renders a shorter
        // identifier. The operator can always cross-reference
        // the "pairing on:" header line above to see what each
        // segment means.
        let label = f.pairing_key.0.join("/");
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
            "  {} pairing-key row pair(s) skipped because one or both sides failed",
            report.skipped_failed,
        );
    }
    if let (Some(avg_a), Some(avg_b)) = (&avg_a, &avg_b) {
        let block = format_per_group_pass_counts(avg_a, avg_b, &label_a, &label_b);
        if !block.is_empty() {
            print!("{block}");
        }
    }
    if report.new_in_b > 0 {
        println!(
            "  {} row(s) new in '{}' (no matching key in '{}')",
            report.new_in_b, label_b, label_a,
        );
    }
    if report.removed_from_a > 0 {
        println!(
            "  {} row(s) removed from '{}' (no matching key in '{}')",
            report.removed_from_a, label_a, label_b,
        );
    }

    // Host-context delta. Same first-Some(host) baseline
    // `compare_partitions` uses — picking representative hosts
    // off the partitioned sidecars rather than the full pool so
    // the delta reflects what actually fed the comparison.
    let sidecars_a: Vec<&crate::test_support::SidecarResult> = pool
        .iter()
        .filter(|s| filter_a.matches(&sidecar_to_row(s)))
        .collect();
    let sidecars_b: Vec<&crate::test_support::SidecarResult> = pool
        .iter()
        .filter(|s| filter_b.matches(&sidecar_to_row(s)))
        .collect();
    let host_a = sidecars_a.iter().find_map(|s| s.host.as_ref());
    let host_b = sidecars_b.iter().find_map(|s| s.host.as_ref());
    print!("{}", format_host_delta(host_a, host_b, &label_a, &label_b));

    Ok(if report.regressions > 0 { 1 } else { 0 })
}

/// Bail when `rows` contains two or more entries with the same
/// pairing key — only relevant under `--no-average`, where each
/// sidecar row stays distinct and `compare_rows_by` would
/// silently latch onto whichever entry happened to be first in
/// iteration order. Names the offending key in the diagnostic
/// so the operator can choose to either drop `--no-average` or
/// add another per-side filter to disambiguate.
fn check_no_duplicate_pairing_keys(
    rows: &[GauntletRow],
    pairing_dims: &[Dimension],
    side_label: &str,
) -> anyhow::Result<()> {
    let mut seen: BTreeMap<PairingKey, usize> = BTreeMap::new();
    for row in rows {
        let key = PairingKey::from_row(row, pairing_dims);
        *seen.entry(key).or_insert(0) += 1;
    }
    if let Some((dup_key, count)) = seen.iter().find(|&(_, &c)| c > 1) {
        anyhow::bail!(
            "stats compare --no-average: side {side_label} has {count} \
             sidecars with the same pairing key {key:?}. Either drop \
             --no-average to average them, or add another --{side}-X \
             filter to disambiguate.",
            key = dup_key.0,
            side = side_label.to_lowercase(),
        );
    }
    Ok(())
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
/// Format the one-line averaging-mode header that prints above
/// the comparison table when `--average` is active.
///
/// Pure function of (`pre_agg_a`, `pre_agg_b`, `a`, `b`) so the
/// exact-string contract — the operator-visible "averaged across
/// N runs (A) and M runs (B)" surface — can be unit-tested
/// without capturing stdout from `compare_partitions`.
///
/// `pre_agg_a` / `pre_agg_b` are the post-typed-filter contributor
/// row counts (i.e. the number of sidecar rows that fed
/// [`group_and_average`]), NOT the post-aggregation unique-key
/// counts. The two answer different operator questions; the
/// header surfaces the contributor count because that's the
/// "how many trials got folded?" intuition the `--average` flag
/// is actually delivering.
pub(crate) fn format_average_header(
    pre_agg_a: usize,
    pre_agg_b: usize,
    a: &str,
    b: &str,
) -> String {
    format!("averaged across {pre_agg_a} runs ({a}) and {pre_agg_b} runs ({b})")
}

/// Format the per-group `passes_observed/total_observed` block
/// that prints below the summary line when `--average` is active.
///
/// Pure function of (`avg_a`, `avg_b`, `a`, `b`) so the rendered
/// surface — one line per (scenario, topology, work_type, flags)
/// group present on either side, with `N/M` per side and `-` for
/// any side that lacks the group — can be unit-tested without
/// capturing stdout. Returns the trailing-newline-terminated
/// block, or empty string when neither side has groups.
///
/// Line shape:
/// `  scenario/topology/work_type: {a}=N/M {b}=N/M`
///
/// The leading two-space indent matches the sibling
/// `summary:` block's continuation lines (e.g.
/// `"  N (scenario, topology, work_type) row pair(s) skipped..."`)
/// so the per-group block reads as a continuation of the same
/// summary section. A blank line separates this block from the
/// preceding `summary:` line for readability.
///
/// Groups present on only one side render `-` for the missing
/// side (also counted in `compare_rows`' `new_in_b` /
/// `removed_from_a` upstream — the per-group block surfaces the
/// asymmetry by name so the operator can see *which* groups went
/// missing without cross-referencing the summary counters).
pub(crate) fn format_per_group_pass_counts(
    avg_a: &[AveragedGroup],
    avg_b: &[AveragedGroup],
    a: &str,
    b: &str,
) -> String {
    type SummaryKey<'a> = (&'a str, &'a str, &'a str, &'a [String]);
    type SummaryValue<'a> = (Option<&'a AveragedGroup>, Option<&'a AveragedGroup>);
    let mut keys: BTreeMap<SummaryKey<'_>, SummaryValue<'_>> = BTreeMap::new();
    for ar in avg_a {
        let k = (
            ar.row.scenario.as_str(),
            ar.row.topology.as_str(),
            ar.row.work_type.as_str(),
            ar.row.flags.as_slice(),
        );
        keys.entry(k).or_insert((None, None)).0 = Some(ar);
    }
    for br in avg_b {
        let k = (
            br.row.scenario.as_str(),
            br.row.topology.as_str(),
            br.row.work_type.as_str(),
            br.row.flags.as_slice(),
        );
        keys.entry(k).or_insert((None, None)).1 = Some(br);
    }
    if keys.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    out.push('\n');
    out.push_str("per-group pass counts (passes_observed/total_observed):\n");
    for ((scn, topo, wt, _flags), (ka, kb)) in keys.into_iter() {
        let fmt_side = |r: Option<&AveragedGroup>| -> String {
            r.map(|x| format!("{}/{}", x.passes_observed, x.total_observed))
                .unwrap_or_else(|| "-".to_string())
        };
        out.push_str(&format!(
            "  {scn}/{topo}/{wt}: {a}={pa} {b}={pb}\n",
            pa = fmt_side(ka),
            pb = fmt_side(kb),
        ));
    }
    out
}

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
            kernel_version: None,
            commit: None,
            kernel_commit: None,
            run_source: None,
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
            worst_wake_latency_tail_ratio: 0.0,
            worst_iterations_per_worker: 0.0,
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

    /// `sidecar_to_row` must copy `SidecarResult::project_commit`
    /// into `GauntletRow::commit` verbatim so the typed
    /// `--project-commit` filter and the upcoming `--a-project-commit` /
    /// `--b-project-commit` slicers see the value the sidecar writer
    /// recorded. A regression that left the field at the
    /// `Option::default()` (`None`) would silently drop the
    /// commit dimension from every comparison even when the
    /// sidecar had a populated value. Pinned for `None`, clean
    /// `Some` (no suffix), and dirty `Some` (`-dirty` suffix) to
    /// catch a regression that special-cases one shape and not
    /// the others — e.g. one that stripped the suffix when copying.
    #[test]
    fn sidecar_to_row_propagates_project_commit() {
        use crate::test_support;
        let sc_dirty = test_support::SidecarResult {
            test_name: "commit_dirty_test".to_string(),
            topology: "1n1l2c1t".to_string(),
            project_commit: Some("abcdef1-dirty".to_string()),
            ..test_support::SidecarResult::test_fixture()
        };
        let row_dirty = sidecar_to_row(&sc_dirty);
        assert_eq!(
            row_dirty.commit.as_deref(),
            Some("abcdef1-dirty"),
            "populated dirty project_commit must propagate \
             verbatim, including the `-dirty` suffix",
        );

        let sc_clean = test_support::SidecarResult {
            test_name: "commit_clean_test".to_string(),
            topology: "1n1l2c1t".to_string(),
            project_commit: Some("abcdef1".to_string()),
            ..test_support::SidecarResult::test_fixture()
        };
        let row_clean = sidecar_to_row(&sc_clean);
        assert_eq!(
            row_clean.commit.as_deref(),
            Some("abcdef1"),
            "populated clean project_commit (no `-dirty` suffix) \
             must propagate verbatim — a regression that always \
             appended `-dirty` or always stripped a tail would \
             surface here independently of the dirty case above",
        );

        let sc_none = test_support::SidecarResult {
            test_name: "no_commit_test".to_string(),
            topology: "1n1l2c1t".to_string(),
            project_commit: None,
            ..test_support::SidecarResult::test_fixture()
        };
        let row_none = sidecar_to_row(&sc_none);
        assert!(
            row_none.commit.is_none(),
            "absent project_commit must propagate as None — a \
             regression substituting an empty string would dilute \
             every `--project-commit` filter into matching all None rows",
        );
    }

    /// `sidecar_to_row` must copy `SidecarResult::kernel_commit`
    /// into `GauntletRow::kernel_commit` verbatim so the typed
    /// `--kernel-commit` filter and per-side
    /// `--a-kernel-commit` / `--b-kernel-commit` slicers see the
    /// value the sidecar writer recorded. A regression that left
    /// the field at the `Option::default()` (`None`) would
    /// silently drop the kernel-commit dimension from every
    /// comparison even when the sidecar had a populated value.
    /// Mirrors `sidecar_to_row_propagates_project_commit` for
    /// the kernel_commit field; pinned for `None`, clean `Some`
    /// (no suffix), and dirty `Some` (`-dirty` suffix) to catch
    /// a regression that special-cases one shape and not the
    /// others.
    #[test]
    fn sidecar_to_row_propagates_kernel_commit() {
        use crate::test_support;
        let sc_dirty = test_support::SidecarResult {
            test_name: "kc_dirty_test".to_string(),
            topology: "1n1l2c1t".to_string(),
            kernel_commit: Some("kabcde7-dirty".to_string()),
            ..test_support::SidecarResult::test_fixture()
        };
        let row_dirty = sidecar_to_row(&sc_dirty);
        assert_eq!(
            row_dirty.kernel_commit.as_deref(),
            Some("kabcde7-dirty"),
            "populated dirty kernel_commit must propagate \
             verbatim, including the `-dirty` suffix",
        );

        let sc_clean = test_support::SidecarResult {
            test_name: "kc_clean_test".to_string(),
            topology: "1n1l2c1t".to_string(),
            kernel_commit: Some("kabcde7".to_string()),
            ..test_support::SidecarResult::test_fixture()
        };
        let row_clean = sidecar_to_row(&sc_clean);
        assert_eq!(
            row_clean.kernel_commit.as_deref(),
            Some("kabcde7"),
            "populated clean kernel_commit (no `-dirty` suffix) \
             must propagate verbatim — a regression that always \
             appended `-dirty` or always stripped a tail would \
             surface here independently of the dirty case above",
        );

        let sc_none = test_support::SidecarResult {
            test_name: "no_kc_test".to_string(),
            topology: "1n1l2c1t".to_string(),
            kernel_commit: None,
            ..test_support::SidecarResult::test_fixture()
        };
        let row_none = sidecar_to_row(&sc_none);
        assert!(
            row_none.kernel_commit.is_none(),
            "absent kernel_commit must propagate as None — a \
             regression substituting an empty string would dilute \
             every `--kernel-commit` filter into matching all \
             None rows",
        );

        // Field non-aliasing pin: kernel_commit and commit must
        // route to distinct row fields. A regression that
        // accidentally cross-wired the two (e.g. `commit:
        // sc.kernel_commit.clone()` instead of
        // `sc.project_commit.clone()`) would hide behind the
        // populated tests above unless the values differ — which
        // they do here. Distinct tokens make the swap obvious.
        let sc_both = test_support::SidecarResult {
            test_name: "both_test".to_string(),
            topology: "1n1l2c1t".to_string(),
            project_commit: Some("project1".to_string()),
            kernel_commit: Some("kernel1".to_string()),
            ..test_support::SidecarResult::test_fixture()
        };
        let row_both = sidecar_to_row(&sc_both);
        assert_eq!(
            row_both.commit.as_deref(),
            Some("project1"),
            "row.commit must come from project_commit, not kernel_commit",
        );
        assert_eq!(
            row_both.kernel_commit.as_deref(),
            Some("kernel1"),
            "row.kernel_commit must come from kernel_commit, not project_commit",
        );
    }

    /// `sidecar_to_row` must copy `SidecarResult::run_source` into
    /// `GauntletRow::run_source` verbatim so the typed `--run-source`
    /// filter and per-side `--a-run-source` / `--b-run-source` slicers
    /// see the run-environment provenance tag the sidecar writer
    /// recorded. A regression that left the field at the
    /// `Option::default()` (`None`) would silently drop the
    /// run-source dimension from every comparison even when the
    /// sidecar had a populated value. Mirrors
    /// `sidecar_to_row_propagates_kernel_commit` for the
    /// `run_source` field; pinned for `None` and the canonical
    /// `Some("local")` / `Some("ci")` / `Some("archive")`
    /// values so a regression that special-cased one tag and
    /// not the others surfaces here. A non-aliasing pin
    /// confirms `run_source` reads from `sc.run_source` rather
    /// than being cross-wired to the visually-similar
    /// `kernel_commit` / `project_commit` fields.
    #[test]
    fn sidecar_to_row_propagates_run_source() {
        use crate::test_support;
        for tag in ["local", "ci", "archive"] {
            let sc = test_support::SidecarResult {
                test_name: format!("run_source_{tag}_test"),
                topology: "1n1l2c1t".to_string(),
                run_source: Some(tag.to_string()),
                ..test_support::SidecarResult::test_fixture()
            };
            let row = sidecar_to_row(&sc);
            assert_eq!(
                row.run_source.as_deref(),
                Some(tag),
                "populated run_source `{tag}` must propagate verbatim",
            );
        }

        let sc_none = test_support::SidecarResult {
            test_name: "no_run_source_test".to_string(),
            topology: "1n1l2c1t".to_string(),
            run_source: None,
            ..test_support::SidecarResult::test_fixture()
        };
        let row_none = sidecar_to_row(&sc_none);
        assert!(
            row_none.run_source.is_none(),
            "absent run_source must propagate as None — a regression \
             substituting an empty string would dilute every \
             `--run-source` filter into matching all None rows",
        );

        // Field non-aliasing pin: `run_source` must route to its
        // own row field. A regression that cross-wired
        // `run_source` to `kernel_commit` (or vice versa) would
        // hide behind the populated tests above unless the values
        // are visibly different. Distinct tokens make the swap
        // obvious.
        let sc_distinct = test_support::SidecarResult {
            test_name: "run_source_distinct_test".to_string(),
            topology: "1n1l2c1t".to_string(),
            run_source: Some("local".to_string()),
            kernel_commit: Some("kabcde7".to_string()),
            project_commit: Some("pabcde7".to_string()),
            ..test_support::SidecarResult::test_fixture()
        };
        let row_distinct = sidecar_to_row(&sc_distinct);
        assert_eq!(
            row_distinct.run_source.as_deref(),
            Some("local"),
            "row.run_source must come from sc.run_source, not from \
             kernel_commit or project_commit",
        );
        assert_eq!(
            row_distinct.kernel_commit.as_deref(),
            Some("kabcde7"),
            "row.kernel_commit must remain sourced from sc.kernel_commit",
        );
        assert_eq!(
            row_distinct.commit.as_deref(),
            Some("pabcde7"),
            "row.commit must remain sourced from sc.project_commit",
        );
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
    /// Covers all twelve `finite_or_zero` call sites in `sidecar_to_row`:
    /// eleven fields drawn from [`ScenarioStats`] plus `imbalance_ratio`
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
                worst_wake_latency_tail_ratio: non_finite,
                worst_iterations_per_worker: non_finite,
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
            (
                "worst_wake_latency_tail_ratio",
                row.worst_wake_latency_tail_ratio,
            ),
            (
                "worst_iterations_per_worker",
                row.worst_iterations_per_worker,
            ),
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

    // -- list_metrics tests --

    /// Text-mode [`list_metrics`] emits a table that names every
    /// registered metric at least once. Uses substring contains
    /// rather than column-exact equality so a future comfy-table
    /// preset rename (NOTHING → other) that rewraps whitespace
    /// does not false-fail — the surface contract is "every metric
    /// name appears somewhere in the rendered output", not a
    /// column-width pin.
    #[test]
    fn list_metrics_text_names_every_metric() {
        let out = list_metrics(false).expect("text render must succeed");
        assert!(!out.is_empty(), "text output must be non-empty");
        for m in METRICS {
            assert!(
                out.contains(m.name),
                "list_metrics(false) output missing metric name {}: {out}",
                m.name,
            );
        }
    }

    /// Text-mode [`list_metrics`] header row names every column. Pins
    /// the header contract so a column rename in
    /// `list_metrics` lands here instead of silently in downstream CI
    /// scripts that grep the output.
    #[test]
    fn list_metrics_text_header_pins_column_names() {
        let out = list_metrics(false).expect("text render must succeed");
        for header in ["NAME", "POLARITY", "DEFAULT_ABS", "DEFAULT_REL", "UNIT"] {
            assert!(
                out.contains(header),
                "list_metrics(false) output missing column header {header}: {out}",
            );
        }
    }

    /// JSON-mode [`list_metrics`] parses back to a `Vec<MetricDef>`-
    /// shaped structure with one entry per registry member. `MetricDef`
    /// itself does not derive `Deserialize` (the `accessor` fn-pointer
    /// is unserializable), so we deserialize into a minimal struct
    /// that captures the fields the wire contract promises.
    #[test]
    fn list_metrics_json_round_trips_via_minimal_schema() {
        #[derive(serde::Deserialize)]
        struct MetricEntry {
            name: String,
            default_abs: f64,
            default_rel: f64,
            display_unit: String,
            // polarity is serialized as an enum tag string by serde
            // (Polarity derives Serialize with the default
            // externally-tagged representation). Deserialize into a
            // serde_json::Value to avoid a cross-crate enum
            // dependency in the test-private schema.
            polarity: serde_json::Value,
        }

        let out = list_metrics(true).expect("json render must succeed");
        let parsed: Vec<MetricEntry> = serde_json::from_str(&out).expect("json output must parse");
        assert_eq!(
            parsed.len(),
            METRICS.len(),
            "json entry count must match METRICS.len()",
        );
        for (parsed_m, registry_m) in parsed.iter().zip(METRICS.iter()) {
            assert_eq!(parsed_m.name, registry_m.name);
            assert_eq!(parsed_m.default_abs, registry_m.default_abs);
            assert_eq!(parsed_m.default_rel, registry_m.default_rel);
            assert_eq!(parsed_m.display_unit, registry_m.display_unit);
            assert!(
                !parsed_m.polarity.is_null(),
                "polarity for {} must serialize as a non-null value",
                registry_m.name,
            );
        }
    }

    /// JSON-mode [`list_metrics`] must NOT expose the `accessor`
    /// fn-pointer field. The `#[serde(skip)]` attribute on
    /// `MetricDef::accessor` carries that contract; a regression that
    /// dropped the attribute would surface here as the emitted JSON
    /// gaining an "accessor" key. Pins the wire surface.
    #[test]
    fn list_metrics_json_omits_accessor_field() {
        let out = list_metrics(true).expect("json render must succeed");
        assert!(
            !out.contains("\"accessor\""),
            "list_metrics(true) must not emit the accessor field — \
             fn-pointers are not serializable and the field carries \
             #[serde(skip)]: {out}",
        );
    }

    /// Iteration order of [`list_metrics`] matches [`METRICS`]
    /// declaration order. Registry order is the canonical surface
    /// order for sidecar / CI-gate consumers; a renderer that sorted
    /// by name or polarity would silently break scripts that key on
    /// the first row.
    #[test]
    fn list_metrics_text_preserves_registry_order() {
        let out = list_metrics(false).expect("text render must succeed");
        let mut last_pos = 0usize;
        for m in METRICS {
            let pos = out
                .find(m.name)
                .unwrap_or_else(|| panic!("metric {} must appear in text output", m.name));
            assert!(
                pos >= last_pos,
                "metric {} appears before a prior metric — text output must \
                 preserve METRICS declaration order",
                m.name,
            );
            last_pos = pos;
        }
    }

    // -- list_values --

    /// Helper that writes N sidecars to `{root}/{run_key}/{run_key}.ktstr.json`.
    /// Each sidecar overrides only the fields the test wants to vary;
    /// the rest come from `SidecarResult::test_fixture()`. Used by the
    /// `list_values_*` tests to build pool fixtures isolated from
    /// `runs_root()`.
    fn write_listvalues_fixture(
        root: &std::path::Path,
        sidecars: &[crate::test_support::SidecarResult],
    ) {
        for (i, sc) in sidecars.iter().enumerate() {
            let run_key = format!("__lv_fixture_{i}__");
            let run_dir = root.join(&run_key);
            std::fs::create_dir_all(&run_dir).expect("create run dir");
            let json = serde_json::to_string(sc).expect("serialize fixture sidecar");
            let path = run_dir.join(format!("{run_key}.ktstr.json"));
            std::fs::write(&path, json).expect("write fixture sidecar");
        }
    }

    /// Empty pool (no run subdirs) must produce a well-formed text
    /// shape with the "(no sidecars in pool)" sentinel under each
    /// dimension heading. The function does NOT bail — discovery on
    /// an empty pool is a valid query that should answer "nothing"
    /// rather than fail.
    #[test]
    fn list_values_empty_pool_text_has_sentinel_per_dim() {
        let alt = tempfile::TempDir::new().expect("tempdir");
        let out = list_values(false, Some(alt.path())).expect("text render must succeed");
        for dim in [
            "kernel:",
            "commit:",
            "kernel_commit:",
            "source:",
            "scheduler:",
            "topology:",
            "work_type:",
            "flags:",
        ] {
            assert!(
                out.contains(dim),
                "text output must include heading for {dim}: {out}",
            );
        }
        // Each dim should report the empty-pool sentinel exactly eight
        // times — one per dim — so a regression that dropped the
        // sentinel for one dim falls out as a count mismatch.
        let sentinel_count = out.matches("(no sidecars in pool)").count();
        assert_eq!(
            sentinel_count, 8,
            "empty pool must surface the no-sidecars sentinel under every \
             one of the 8 dims (kernel/commit/kernel_commit/source/\
             scheduler/topology/work_type/flags); got {sentinel_count} \
             occurrences in:\n{out}",
        );
    }

    /// Empty pool → JSON object with empty arrays for every dim.
    /// Pins the JSON shape so a regression that dropped a key (e.g.
    /// "scheduler") on the empty-pool branch surfaces here.
    #[test]
    fn list_values_empty_pool_json_emits_empty_arrays() {
        let alt = tempfile::TempDir::new().expect("tempdir");
        let out = list_values(true, Some(alt.path())).expect("json render must succeed");
        let parsed: serde_json::Value = serde_json::from_str(&out).expect("json output must parse");
        for dim in [
            "kernel",
            "commit",
            "kernel_commit",
            "source",
            "scheduler",
            "topology",
            "work_type",
            "flags",
        ] {
            let arr = parsed
                .get(dim)
                .unwrap_or_else(|| panic!("missing key {dim}"));
            assert!(arr.is_array(), "key {dim} must serialize as an array");
            assert_eq!(
                arr.as_array().unwrap().len(),
                0,
                "key {dim} must be an empty array on empty pool",
            );
        }
    }

    /// Populated pool: distinct values per dim are deduplicated and
    /// sorted; flags are exploded (each entry of `active_flags`
    /// becomes one set member); `kernel_version: None` and
    /// `project_commit: None` produce a `null` entry (JSON) and
    /// `unknown` line (text).
    #[test]
    fn list_values_text_dedupes_and_sorts_per_dim() {
        use crate::test_support::SidecarResult;

        let alt = tempfile::TempDir::new().expect("tempdir");
        let sidecars = vec![
            SidecarResult {
                test_name: "t_a".to_string(),
                topology: "1n2l4c1t".to_string(),
                scheduler: "scx_rusty".to_string(),
                work_type: "CpuSpin".to_string(),
                active_flags: vec!["llc".to_string(), "rusty_balance".to_string()],
                kernel_version: Some("6.14.2".to_string()),
                project_commit: Some("abcdef1".to_string()),
                ..SidecarResult::test_fixture()
            },
            SidecarResult {
                test_name: "t_b".to_string(),
                topology: "1n4l2c1t".to_string(),
                scheduler: "eevdf".to_string(),
                work_type: "PageFaultChurn".to_string(),
                active_flags: vec!["llc".to_string()],
                kernel_version: None,
                project_commit: None,
                ..SidecarResult::test_fixture()
            },
            // Duplicate of the first sidecar's identity-fields; the
            // BTreeSet must dedupe so each value lands once in the
            // rendered output.
            SidecarResult {
                test_name: "t_c".to_string(),
                topology: "1n2l4c1t".to_string(),
                scheduler: "scx_rusty".to_string(),
                work_type: "CpuSpin".to_string(),
                active_flags: vec!["rusty_balance".to_string()],
                kernel_version: Some("6.14.2".to_string()),
                project_commit: Some("abcdef1".to_string()),
                ..SidecarResult::test_fixture()
            },
        ];
        write_listvalues_fixture(alt.path(), &sidecars);

        let out = list_values(false, Some(alt.path())).expect("text render must succeed");

        // Dedupe: each distinct VALUE appears EXACTLY once per
        // dim (set semantics) even though "scx_rusty" / "1n2l4c1t"
        // / "CpuSpin" / "abcdef1" / "6.14.2" come from two of the
        // three fixtures. Each value below is unique to its dim
        // so it should appear once across the rendered text. The
        // `unknown` sentinel is checked separately because both
        // `kernel` and `commit` are optional dims and each emits
        // its own `unknown` line.
        for value in [
            "6.14.2",
            "abcdef1",
            "scx_rusty",
            "eevdf",
            "1n2l4c1t",
            "1n4l2c1t",
            "CpuSpin",
            "PageFaultChurn",
            "llc",
            "rusty_balance",
        ] {
            let count = out.matches(value).count();
            assert_eq!(
                count, 1,
                "value {value} must appear exactly once in text output (BTreeSet dedup); \
                 got {count} in:\n{out}",
            );
        }
        // `unknown` appears once per Optional dim that has a None
        // entry: kernel, commit, and kernel_commit. The second
        // fixture has `kernel_version: None` and `project_commit:
        // None`; every fixture in this test leaves `kernel_commit`
        // at its `test_fixture` default (None), so the
        // kernel_commit set's None bucket renders one `unknown`
        // line as well. Total: 3 occurrences.
        //
        // `run_source` is the fourth optional dim but does NOT
        // contribute an `unknown` here: `list_values(_, Some(dir))`
        // calls `apply_archive_source_override` on the loaded pool
        // (the `--dir` flag treats the supplied root as an archive),
        // which rewrites every `run_source: None` to
        // `Some("archive")` BEFORE the dimension-set is built. Every
        // fixture above leaves `run_source` at its `test_fixture`
        // default (None), but they all surface as `archive` after
        // the override — the run_source set never holds a None
        // entry on this code path, so no `unknown` line is emitted
        // for it.
        let unknown_count = out.matches("unknown").count();
        assert_eq!(
            unknown_count, 3,
            "`unknown` must render once per optional dim with a None \
             entry (kernel + commit + kernel_commit = 3); got \
             {unknown_count} in:\n{out}",
        );

        // Sort: both schedulers in ascending lex order means
        // "eevdf" appears BEFORE "scx_rusty" in the rendered text.
        let pos_eevdf = out.find("eevdf").expect("eevdf in output");
        let pos_rusty = out.find("scx_rusty").expect("scx_rusty in output");
        assert!(
            pos_eevdf < pos_rusty,
            "values within a dim must render sorted (BTreeSet iter order); \
             expected 'eevdf' before 'scx_rusty' in:\n{out}",
        );
    }

    /// JSON shape: `kernel` and `commit` arrays carry `null` for
    /// absent values, `Value::String` for present values; the other
    /// four dims are bare `String` arrays. `flags` is exploded —
    /// individual flag names, not the joined-set string.
    #[test]
    fn list_values_json_carries_null_for_optional_dims() {
        use crate::test_support::SidecarResult;

        let alt = tempfile::TempDir::new().expect("tempdir");
        let sidecars = vec![
            SidecarResult {
                test_name: "t_known".to_string(),
                kernel_version: Some("6.14.2".to_string()),
                project_commit: Some("abcdef1".to_string()),
                active_flags: vec!["llc".to_string(), "rusty_balance".to_string()],
                ..SidecarResult::test_fixture()
            },
            SidecarResult {
                test_name: "t_unknown".to_string(),
                kernel_version: None,
                project_commit: None,
                active_flags: vec![],
                ..SidecarResult::test_fixture()
            },
        ];
        write_listvalues_fixture(alt.path(), &sidecars);

        let out = list_values(true, Some(alt.path())).expect("json render must succeed");
        let parsed: serde_json::Value = serde_json::from_str(&out).expect("json output must parse");

        let kernel = parsed
            .get("kernel")
            .expect("kernel key")
            .as_array()
            .unwrap();
        assert!(
            kernel.iter().any(|v| v.is_null()),
            "kernel array must include a literal null for the None entry; got {kernel:?}",
        );
        assert!(
            kernel.iter().any(|v| v.as_str() == Some("6.14.2")),
            "kernel array must include the populated value 6.14.2; got {kernel:?}",
        );

        let commit = parsed
            .get("commit")
            .expect("commit key")
            .as_array()
            .unwrap();
        assert!(
            commit.iter().any(|v| v.is_null()),
            "commit array must include a literal null for the None entry; got {commit:?}",
        );
        assert!(
            commit.iter().any(|v| v.as_str() == Some("abcdef1")),
            "commit array must include the populated value abcdef1; got {commit:?}",
        );

        // Flags: exploded — both "llc" and "rusty_balance" appear
        // as DISTINCT entries, NOT a single "llc|rusty_balance"
        // string.
        let flags = parsed.get("flags").expect("flags key").as_array().unwrap();
        assert_eq!(
            flags.len(),
            2,
            "flags must explode to individual names — expected 2 \
             entries (llc, rusty_balance), got {flags:?}",
        );
        let flag_names: Vec<&str> = flags
            .iter()
            .map(|v| v.as_str().expect("flag is string"))
            .collect();
        assert!(flag_names.contains(&"llc"));
        assert!(flag_names.contains(&"rusty_balance"));
    }

    /// `dir = None` resolves against `runs_root()`; if `runs_root()`
    /// does not exist, the function returns Ok with empty arrays /
    /// per-dim sentinel rather than bailing. Pins the no-bail
    /// contract on missing-root.
    #[test]
    fn list_values_none_dir_does_not_bail_on_missing_root() {
        // We cannot reliably wipe `runs_root()` from a unit test, but
        // we can pin the "Some(nonexistent_path)" branch which
        // exercises the same `collect_pool -> empty Vec` codepath
        // (`fs::read_dir` returns Err on a missing root, and
        // `collect_pool` swallows that into an empty pool).
        let alt = tempfile::TempDir::new().expect("tempdir");
        let nonexistent = alt.path().join("definitely_does_not_exist");
        let out = list_values(false, Some(&nonexistent)).expect("must not bail on missing root");
        assert!(
            out.contains("(no sidecars in pool)"),
            "missing root must render the no-sidecars sentinel: {out}",
        );
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
        assert_eq!(
            read_metric(&row, "worst_median_wake_latency_us"),
            Some(50.0)
        );
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
        let res = compare_rows(&rows_a, &rows_b, None, &ComparisonPolicy::default());
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
        let res2 = compare_rows(&rows_a, &rows_b2, None, &ComparisonPolicy::default());
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
        let res = compare_rows(&rows_a, &rows_b, None, &ComparisonPolicy::uniform(10.0));
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
        let res_imp = compare_rows(&rows_b, &rows_a, None, &ComparisonPolicy::uniform(10.0));
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
        let res = compare_rows(&rows_a, &rows_b, None, &ComparisonPolicy::default());
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
        let res_up = compare_rows(&rows_a2, &rows_b2, None, &ComparisonPolicy::default());
        let spread_up = res_up
            .findings
            .iter()
            .find(|d| d.metric.name == "worst_spread")
            .expect("worst_spread should produce a delta");
        assert!(spread_up.is_regression, "spread increase is a regression");
        assert_eq!(spread_up.delta, 20.0);

        let res_down = compare_rows(&rows_b2, &rows_a2, None, &ComparisonPolicy::default());
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
        let res = compare_rows(
            &[row_a.clone()],
            &[row_b.clone()],
            None,
            &ComparisonPolicy::default(),
        );
        assert_eq!(res.regressions, 0);
        assert_eq!(res.improvements, 0);
        assert_eq!(
            res.skipped_failed, 1,
            "skipped side must count as skipped_failed, not produce deltas"
        );

        // Symmetrically on the B side.
        row_a.skipped = false;
        row_b.skipped = true;
        let res = compare_rows(&[row_a], &[row_b], None, &ComparisonPolicy::default());
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
        let res = compare_rows(&rows_a, &rows_b, None, &ComparisonPolicy::uniform(10.0));
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
        let res = compare_rows(
            &rows_a,
            &rows_b,
            Some("alpha"),
            &ComparisonPolicy::default(),
        );
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
        let res_topo = compare_rows(&rows_a, &rows_b, Some("tiny"), &ComparisonPolicy::default());
        assert_eq!(res_topo.regressions, 2, "both rows match 'tiny' topology");
        assert_eq!(res_topo.findings.len(), 2);

        // Non-matching filter yields no comparisons at all.
        let res_none = compare_rows(
            &rows_a,
            &rows_b,
            Some("nomatch"),
            &ComparisonPolicy::default(),
        );
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
        let res_default = compare_rows(&rows_a, &rows_b, None, &ComparisonPolicy::default());
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
        let res_override = compare_rows(&rows_a, &rows_b, None, &ComparisonPolicy::uniform(5.0));
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
        let res_small = compare_rows(
            &rows_a_small,
            &rows_b_small,
            None,
            &ComparisonPolicy::uniform(1.0),
        );
        assert!(
            !res_small
                .findings
                .iter()
                .any(|d| d.metric.name == "worst_spread"),
            "abs gate must still block tiny absolute moves"
        );
    }

    /// `ComparisonPolicy::rel_threshold` resolution priority pinned
    /// by exhaustive enumeration: per-metric override wins over
    /// `default_percent`, which wins over the registry fallback.
    /// A regression that inverted the priority or shortcut the
    /// fallback (e.g. always returning `default_percent` even when
    /// a per-metric override exists) surfaces here, not as subtly-
    /// wrong thresholds inside `compare_rows`.
    #[test]
    fn comparison_policy_rel_threshold_resolution_priority() {
        // Empty policy → registry fallback. `default_rel` is
        // passed by the caller (compare_rows supplies it from
        // `m.default_rel`), so we pick an arbitrary fallback here
        // and check it's returned verbatim.
        let empty = ComparisonPolicy::default();
        assert_eq!(
            empty.rel_threshold("worst_spread", 0.25),
            0.25,
            "empty policy must fall through to the registry default_rel",
        );

        // Uniform override → default_percent / 100 wins over
        // the registry default.
        let uniform = ComparisonPolicy::uniform(10.0);
        assert_eq!(
            uniform.rel_threshold("worst_spread", 0.25),
            0.10,
            "uniform(10.0) must override the registry default_rel \
             with 10.0 / 100.0 = 0.10",
        );

        // Per-metric override wins over both `default_percent` and
        // the registry default. Use two metric names so the test
        // also proves other metrics still see `default_percent`
        // when no per-metric entry matches.
        let mut per_metric = ComparisonPolicy::uniform(10.0);
        per_metric
            .per_metric_percent
            .insert("worst_spread".to_string(), 5.0);
        assert_eq!(
            per_metric.rel_threshold("worst_spread", 0.25),
            0.05,
            "per-metric override (5.0) must win over default_percent \
             (10.0) and the registry default (0.25)",
        );
        assert_eq!(
            per_metric.rel_threshold("worst_gap_ms", 0.25),
            0.10,
            "metrics not in the per-metric map must still see the \
             default_percent (10.0 → 0.10), not the registry default",
        );
    }

    /// `worst_wake_latency_tail_ratio` must be suppressed below the
    /// [`WAKE_LATENCY_TAIL_RATIO_MIN_ITERATIONS`] sample floor. Low-N
    /// runs produce p99/median ratios dominated by a single outlier;
    /// the metric accessor must return `None` in that regime so
    /// [`compare_rows`] short-circuits and emits no finding.
    ///
    /// Positive side: above the floor, the same delta that was
    /// suppressed below must produce a finding. This proves the
    /// None-vs-Some branching is the gate that's firing — not an
    /// unrelated threshold somewhere else in the comparison math.
    #[test]
    fn wake_latency_tail_ratio_is_suppressed_below_min_iteration_floor() {
        use crate::stats::WAKE_LATENCY_TAIL_RATIO_MIN_ITERATIONS as MIN;
        let metric = metric_def("worst_wake_latency_tail_ratio")
            .expect("worst_wake_latency_tail_ratio must be registered in METRICS");

        // Below the floor: accessor returns None. Both sides collapse
        // to 0.0 via unwrap_or(0.0); the EPSILON-guard then classifies
        // the delta as unchanged.
        let mut low_a = make_row("tail_low", "tiny-1llc", true, 0.0);
        let mut low_b = make_row("tail_low", "tiny-1llc", true, 0.0);
        low_a.total_iterations = MIN - 1;
        low_b.total_iterations = MIN - 1;
        low_a.worst_wake_latency_tail_ratio = 2.0;
        low_b.worst_wake_latency_tail_ratio = 20.0;
        assert!(
            metric.read(&low_a).is_none(),
            "below-floor A accessor must return None so the regression \
             math cannot see a value",
        );
        assert!(
            metric.read(&low_b).is_none(),
            "below-floor B accessor must return None even when the \
             raw field would have carried a suspicious value",
        );
        let below = compare_rows(
            std::slice::from_ref(&low_a),
            std::slice::from_ref(&low_b),
            None,
            &ComparisonPolicy::default(),
        );
        assert_eq!(
            below.regressions, 0,
            "below-floor comparison must not surface a regression — \
             low-N ratios are noise, not signal",
        );
        assert!(
            below.findings.is_empty(),
            "below-floor comparison must emit no findings",
        );

        // At and above the floor: accessor returns Some and the same
        // delta now produces a finding.
        let mut hi_a = make_row("tail_hi", "tiny-1llc", true, 0.0);
        let mut hi_b = make_row("tail_hi", "tiny-1llc", true, 0.0);
        hi_a.total_iterations = MIN;
        hi_b.total_iterations = MIN;
        hi_a.worst_wake_latency_tail_ratio = 2.0;
        hi_b.worst_wake_latency_tail_ratio = 20.0;
        assert_eq!(
            metric.read(&hi_a),
            Some(2.0),
            "at-floor accessor must return Some",
        );
        let above = compare_rows(
            std::slice::from_ref(&hi_a),
            std::slice::from_ref(&hi_b),
            None,
            &ComparisonPolicy::default(),
        );
        assert_eq!(
            above.regressions, 1,
            "at-floor comparison with a 10x tail blow-up must surface \
             as a regression; threshold wiring has a gap otherwise",
        );
    }

    /// Explicit None-branch pin on the compare_rows accessor contract.
    ///
    /// `compare_rows` calls `m.read(row)` for every metric and
    /// falls through `unwrap_or(0.0)` to the EPSILON-guard when the
    /// accessor returns `None`. The `wake_latency_tail_ratio_is_suppressed_below_*`
    /// sibling exercises this path EMBEDDED in the full comparison
    /// flow (via the tail-ratio accessor's iteration-count gate),
    /// but does NOT directly prove that `compare_rows` handles a
    /// None result; a regression that removed the `unwrap_or(0.0)`
    /// and panicked on None would fail the sibling only through
    /// the indirect "compare_rows panicked" route, which could be
    /// mistaken for a test infrastructure problem.
    ///
    /// This test synthesizes the None condition explicitly — a
    /// below-floor iterations count with distinctly-different
    /// stored `worst_wake_latency_tail_ratio` values on each side
    /// — and asserts the three observable consequences:
    /// 1. `metric.read(&row)` returns `None` on both sides.
    /// 2. `compare_rows` does NOT panic.
    /// 3. The resulting `CompareReport` classifies the pair as
    ///    `unchanged` (EPSILON guard swallowed the 0.0/0.0 delta).
    ///
    /// A panic or a regression/improvement count > 0 here would
    /// indicate the `unwrap_or(0.0)` in `compare_rows` has drifted.
    #[test]
    fn compare_rows_handles_none_from_accessor_as_zero() {
        use crate::stats::WAKE_LATENCY_TAIL_RATIO_MIN_ITERATIONS as MIN;
        let metric = metric_def("worst_wake_latency_tail_ratio")
            .expect("tail ratio metric must be registered");

        let mut row_a = make_row("none_branch", "tiny-1llc", true, 0.0);
        let mut row_b = make_row("none_branch", "tiny-1llc", true, 0.0);
        row_a.total_iterations = MIN - 1;
        row_b.total_iterations = MIN - 1;
        // Stored fields are distinctly non-zero so a regression that
        // short-circuited the accessor (returned the stored value
        // directly) would produce a 1000x delta that would fail
        // both the "unchanged" classification AND the regression
        // count assertion.
        row_a.worst_wake_latency_tail_ratio = 1.0;
        row_b.worst_wake_latency_tail_ratio = 1000.0;

        assert!(
            metric.read(&row_a).is_none(),
            "accessor must return None for below-floor A input — \
             otherwise this test is not actually exercising the \
             None branch of compare_rows",
        );
        assert!(
            metric.read(&row_b).is_none(),
            "accessor must return None for below-floor B input",
        );

        // The call must not panic (a regression that dropped the
        // `unwrap_or` would trip here), and the result must
        // classify the pair as unchanged — both sides collapse to
        // 0.0 via unwrap_or, then the `abs() < EPSILON` guard
        // short-circuits without producing a finding.
        let report = compare_rows(
            std::slice::from_ref(&row_a),
            std::slice::from_ref(&row_b),
            None,
            &ComparisonPolicy::default(),
        );
        assert_eq!(
            report.regressions, 0,
            "None accessor result must land as unchanged, not a regression",
        );
        assert_eq!(
            report.improvements, 0,
            "None accessor result must land as unchanged, not an improvement",
        );
        assert!(
            report.findings.is_empty(),
            "no findings must be emitted when the accessor returns None; \
             got: {:?}",
            report.findings,
        );
    }

    /// `ComparisonPolicy::load_json` round-trips a policy file: a
    /// policy constructed in memory, serialized, and reloaded must
    /// yield the same thresholds end-to-end. Pins the wire format
    /// for the `--policy <path>` CLI flag.
    #[test]
    fn comparison_policy_load_json_round_trip() {
        let mut original = ComparisonPolicy::uniform(10.0);
        original
            .per_metric_percent
            .insert("worst_spread".to_string(), 5.0);
        original
            .per_metric_percent
            .insert("worst_p99_wake_latency_us".to_string(), 20.0);

        let json = serde_json::to_string(&original).expect("serialize policy");

        let tmp = tempfile::NamedTempFile::new().expect("create tempfile");
        std::fs::write(tmp.path(), json).expect("write policy file");

        let loaded = ComparisonPolicy::load_json(tmp.path()).expect("load policy");

        assert_eq!(
            loaded.default_percent,
            Some(10.0),
            "default_percent must round-trip",
        );
        assert_eq!(
            loaded.per_metric_percent.get("worst_spread"),
            Some(&5.0),
            "per-metric worst_spread override must round-trip",
        );
        assert_eq!(
            loaded.per_metric_percent.get("worst_p99_wake_latency_us"),
            Some(&20.0),
            "per-metric worst_p99 override must round-trip",
        );
        // Resolution-path equivalence: the loaded policy resolves
        // every metric identically to the original.
        for metric_name in ["worst_spread", "worst_p99_wake_latency_us", "worst_gap_ms"] {
            assert_eq!(
                loaded.rel_threshold(metric_name, 0.25),
                original.rel_threshold(metric_name, 0.25),
                "load_json round-trip must preserve threshold \
                 resolution for {metric_name}",
            );
        }
    }

    /// `ComparisonPolicy::load_json` on a nonexistent path must
    /// surface an actionable error naming the path (not a generic
    /// "no such file"). Pins the `with_context` chain — a
    /// regression that dropped the context would collapse a
    /// user-facing `--policy missing.json` invocation into a
    /// bare `No such file or directory` with no clue about where
    /// the missing file was expected.
    #[test]
    fn comparison_policy_load_json_nonexistent_path_surfaces_path() {
        let path = std::path::Path::new("/nonexistent/ktstr/policy-DOES-NOT-EXIST.json");
        let err = ComparisonPolicy::load_json(path).expect_err("nonexistent path must fail");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains(&path.display().to_string()),
            "error must name the missing path so a user can see \
             which file was expected; got: {rendered}",
        );
        assert!(
            rendered.to_ascii_lowercase().contains("read")
                || rendered.to_ascii_lowercase().contains("no such"),
            "error must describe the read failure (either the \
             `with_context` \"read comparison policy from ...\" \
             prefix or std's underlying \"No such file...\" \
             reason); got: {rendered}",
        );
    }

    /// `ComparisonPolicy::load_json` on a malformed JSON body
    /// must include both the path (for locating) AND the parse
    /// context (for understanding the failure shape). A
    /// `serde_json::Error` on its own gives line/column but no
    /// file identity; the `with_context` adds the path. Pins
    /// both halves.
    #[test]
    fn comparison_policy_load_json_malformed_json_surfaces_path_and_parse_context() {
        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        // Not JSON — clearly malformed.
        std::fs::write(tmp.path(), "this is not json at all {{{").expect("write");
        let err = ComparisonPolicy::load_json(tmp.path()).expect_err("malformed JSON must fail");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains(&tmp.path().display().to_string()),
            "malformed-JSON error must name the path; got: {rendered}",
        );
        assert!(
            rendered.to_ascii_lowercase().contains("parse")
                || rendered.to_ascii_lowercase().contains("expected"),
            "malformed-JSON error must include a parse-context \
             hint (either the `with_context` \"parse comparison \
             policy from ...\" prefix, or serde_json's \"expected \
             ...\" reason); got: {rendered}",
        );
    }

    /// `load_json` rejects unknown top-level fields per
    /// `deny_unknown_fields`. A misspelled field (e.g.
    /// `default_percentage` vs `default_percent`) must surface as
    /// a parse error, not silently drop the value and fall back
    /// to the default.
    #[test]
    fn comparison_policy_load_json_rejects_unknown_fields() {
        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        std::fs::write(tmp.path(), r#"{"default_percentage": 10.0}"#).expect("write");
        let err = ComparisonPolicy::load_json(tmp.path()).expect_err("unknown field must fail");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("default_percentage")
                || rendered.to_ascii_lowercase().contains("unknown"),
            "unknown-field error must name the typo so a user \
             can fix the policy file; got: {rendered}",
        );
    }

    /// `validate` rejects negative `default_percent`. A regression
    /// that lost the sign check would let `--threshold -10`
    /// through to `compare_rows`' dual-gate `.abs()` comparison,
    /// where a negative `rel_thresh` makes every delta (including
    /// zero) significant — silently inverting the comparison.
    #[test]
    fn comparison_policy_validate_rejects_negative_default_percent() {
        let policy = ComparisonPolicy::uniform(-10.0);
        let err = policy
            .validate()
            .expect_err("negative default_percent must fail validation");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("default_percent"),
            "validation error must name the field; got: {rendered}",
        );
        assert!(
            rendered.contains("-10"),
            "validation error must echo the rejected value; got: {rendered}",
        );
    }

    /// `validate` rejects unknown per-metric keys. A typo in the
    /// policy file would otherwise silently fall through to
    /// `default_percent` — a user debugging a regression with
    /// `--policy typo.json` would see the uniform threshold
    /// applied instead of the expected override and have no way
    /// to know why.
    #[test]
    fn comparison_policy_validate_rejects_unknown_per_metric_keys() {
        let mut policy = ComparisonPolicy::default();
        policy
            .per_metric_percent
            .insert("wrost_spread".to_string(), 5.0); // typo
        let err = policy
            .validate()
            .expect_err("unknown per-metric key must fail validation");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("wrost_spread"),
            "validation error must echo the unknown key so a user \
             can see the typo; got: {rendered}",
        );
        // Known-metric list should appear so the user can pick the
        // right spelling. Registered metric names include
        // `worst_spread` — a hint toward the correct key.
        assert!(
            rendered.contains("worst_spread"),
            "validation error should include the registered \
             metric list so users can find the right spelling; \
             got: {rendered}",
        );
    }

    /// `validate` rejects negative per-metric overrides. Covers
    /// the sibling case of the default_percent sign check above.
    #[test]
    fn comparison_policy_validate_rejects_negative_per_metric_value() {
        let mut policy = ComparisonPolicy::default();
        policy
            .per_metric_percent
            .insert("worst_spread".to_string(), -5.0);
        let err = policy
            .validate()
            .expect_err("negative per-metric percent must fail");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("worst_spread") && rendered.contains("-5"),
            "validation error must name both the key and the \
             rejected value; got: {rendered}",
        );
    }

    /// Defence-in-depth against an on-disk policy missing fields
    /// (e.g. older wire format, hand-edited JSON). The struct uses
    /// `#[serde(default)]` on every field so a partial JSON
    /// (`{}`, `{"default_percent": 5}`) deserializes to a policy
    /// with the missing field at its `Default` value. A regression
    /// that dropped the `#[serde(default)]` attribute would make
    /// `load_json` reject otherwise-valid partial policies.
    #[test]
    fn comparison_policy_load_json_accepts_partial_fields() {
        let tmp = tempfile::NamedTempFile::new().expect("create tempfile");
        // Empty object → policy with every default.
        std::fs::write(tmp.path(), "{}").expect("write empty policy");
        let loaded = ComparisonPolicy::load_json(tmp.path()).expect("load empty policy");
        assert_eq!(loaded.default_percent, None);
        assert!(loaded.per_metric_percent.is_empty());

        // Only default_percent set → empty per_metric.
        std::fs::write(tmp.path(), r#"{"default_percent": 7.5}"#).expect("write partial policy");
        let loaded = ComparisonPolicy::load_json(tmp.path()).expect("load partial policy");
        assert_eq!(loaded.default_percent, Some(7.5));
        assert!(loaded.per_metric_percent.is_empty());

        // Only per_metric_percent set → default_percent None.
        std::fs::write(
            tmp.path(),
            r#"{"per_metric_percent": {"worst_spread": 3.0}}"#,
        )
        .expect("write per-metric-only policy");
        let loaded = ComparisonPolicy::load_json(tmp.path()).expect("load per-metric-only policy");
        assert_eq!(loaded.default_percent, None);
        assert_eq!(loaded.per_metric_percent.get("worst_spread"), Some(&3.0),);
    }

    /// End-to-end pin: `compare_rows` with a per-metric policy
    /// must apply the override for the matching metric AND fall
    /// through to `default_percent` for every other metric. The
    /// unit-level `comparison_policy_rel_threshold_resolution_priority`
    /// test above pins the resolution function in isolation; this
    /// test runs it through the actual compare_rows pipeline with
    /// rows that trigger distinct deltas on two metrics, proving
    /// that `compare_rows` reads `m.name` correctly and hands it
    /// to `policy.rel_threshold`. A regression that hard-coded a
    /// single metric name, or passed the wrong name to the
    /// resolver, would surface here as the wrong regression count.
    ///
    /// Fixture:
    /// - A: `worst_spread = 100`, `worst_median_wake_latency_us = 100`
    /// - B: `worst_spread = 106` (6% delta, passes the abs gate
    ///   at 5.0), `worst_median_wake_latency_us = 110` (10%
    ///   delta).
    /// - Policy: `default_percent = 20%`, per_metric
    ///   `worst_spread = 5%`.
    ///
    /// Expected: `worst_spread`'s 6% delta beats the 5%
    /// per-metric override → regression. `worst_median_wake_latency_us`'s
    /// 10% delta falls under the 20% default → unchanged. Total
    /// regressions = 1.
    #[test]
    fn compare_rows_per_metric_policy_resolves_each_metric_independently() {
        // Construct rows with both metrics non-default so we can
        // trigger per-metric and default_percent branches in one
        // row pair.
        let mut row_a = cmp_row("t", "tiny-1llc", true, 100.0, 0);
        row_a.worst_median_wake_latency_us = 100.0;
        let mut row_b = cmp_row("t", "tiny-1llc", true, 106.0, 0);
        row_b.worst_median_wake_latency_us = 110.0;

        let mut policy = ComparisonPolicy::uniform(20.0);
        policy
            .per_metric_percent
            .insert("worst_spread".to_string(), 5.0);

        let res = compare_rows(&[row_a], &[row_b], None, &policy);

        let spread_finding = res
            .findings
            .iter()
            .find(|f| f.metric.name == "worst_spread");
        assert!(
            spread_finding.is_some(),
            "worst_spread per-metric override (5%) must fire on 6% \
             delta; got findings: {:?}",
            res.findings
                .iter()
                .map(|f| f.metric.name)
                .collect::<Vec<_>>(),
        );
        let spread_finding = spread_finding.unwrap();
        assert!(spread_finding.is_regression, "6% > 5% → regression");

        // worst_median_wake_latency_us has a 10% delta; under
        // default_percent = 20%, it must be unchanged (not in
        // findings).
        let wake_finding = res
            .findings
            .iter()
            .find(|f| f.metric.name == "worst_median_wake_latency_us");
        assert!(
            wake_finding.is_none(),
            "worst_median_wake_latency_us 10% delta must fall \
             under default_percent 20% and be unchanged. The \
             regression would indicate `compare_rows` ignored \
             default_percent for non-per-metric entries; got \
             finding: {wake_finding:?}",
        );

        assert_eq!(
            res.regressions, 1,
            "exactly one regression expected — the per-metric \
             spread override should win on spread, and the \
             default_percent should suppress wake latency. Got: \
             regressions={}, improvements={}, unchanged={}",
            res.regressions, res.improvements, res.unchanged,
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
        let res = compare_rows(&rows_a, &rows_b, None, &ComparisonPolicy::default());
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
        let res = compare_rows(&rows_a, &rows_b, None, &ComparisonPolicy::default());

        // Each flag profile's spread moved by 90 → one regression
        // (llc 10→100) and one improvement (borrow 100→10).
        assert_eq!(res.regressions, 1, "llc regression should fire (10 → 100)",);
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
        let unfiltered = compare_rows(&rows_a, &rows_b, None, &ComparisonPolicy::default());
        assert_eq!(unfiltered.skipped_failed, 1);
        assert_eq!(unfiltered.regressions, 1, "alpha still regresses");

        // Filtering to "alpha" excludes beta entirely; the failed row
        // is filtered out before the passed gate runs, so
        // skipped_failed=0.
        let filtered = compare_rows(
            &rows_a,
            &rows_b,
            Some("alpha"),
            &ComparisonPolicy::default(),
        );
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

        let res = compare_rows(
            &[a1, a2],
            &[b1, b2],
            Some("scx_alpha"),
            &ComparisonPolicy::default(),
        );
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
        let res = compare_rows(&rows_a, &rows_b, None, &ComparisonPolicy::default());
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
        let res = compare_rows(
            &rows_a,
            &rows_b,
            Some("alpha"),
            &ComparisonPolicy::default(),
        );
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
        assert!(
            out.starts_with("\nhost delta ('a' → 'b'):\n"),
            "got: {out:?}"
        );
        // `kernel_release` differs between the two contexts so the
        // diff body must be non-empty — confirms we routed through
        // the `else` arm and not the `identical` arm.
        let body = &out["\nhost delta ('a' → 'b'):\n".len()..];
        assert!(
            !body.is_empty(),
            "differing contexts must produce a diff body"
        );
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
        let mut row = make_row("scn", "topo", false, std::f64::consts::PI);
        row.flags = vec!["a".into(), "b".into(), "c".into()];
        row.ext_metrics.insert("m1".into(), 1.0);
        row.ext_metrics.insert("m2".into(), 2.5);
        let json = serde_json::to_string(&row).unwrap();
        let back: GauntletRow = serde_json::from_str(&json).unwrap();
        assert_eq!(back, row);
    }

    /// `compare_partitions` honours the `--dir` override —
    /// pool-collection walks the override path rather than the
    /// default [`crate::test_support::runs_root`]. Pool source-of-
    /// truth threading regressed silently in earlier versions
    /// (`--dir` was parsed but ignored), so this test pins the
    /// load-bearing wire from CLI arg through `compare_partitions`
    /// down to `collect_pool`.
    ///
    /// Fixture: a tempdir alt-root with two run subdirectories,
    /// each holding one sidecar. The two sidecars differ on
    /// `scheduler` so the slicing-dim is `scheduler` and
    /// `compare_partitions` has a well-defined contrast. Calling
    /// `compare_partitions` with `dir = Some(alt_root)` finds the
    /// pooled fixtures and returns Ok; calling without `--dir`
    /// against runs_root (which doesn't contain these private
    /// fixtures) fails with a "no sidecar data" diagnostic.
    #[test]
    fn compare_partitions_threads_dir_through_to_pool_collection() {
        use crate::test_support::SidecarResult;

        let alt_root = tempfile::TempDir::new().expect("create alt-root tempdir");
        // Two run subdirs; each holds one sidecar. The sidecars
        // differ on scheduler so the slicing-dim derivation has
        // a non-empty result.
        for (run_key, sched) in [
            ("__dir_thread_a__", "scx_alpha"),
            ("__dir_thread_b__", "scx_beta"),
        ] {
            let run_dir = alt_root.path().join(run_key);
            std::fs::create_dir_all(&run_dir).expect("create run dir");
            let sidecar = SidecarResult {
                test_name: "dir_thread_fixture".to_string(),
                scheduler: sched.to_string(),
                ..SidecarResult::test_fixture()
            };
            let json = serde_json::to_string(&sidecar).expect("serialize fixture sidecar");
            let sidecar_path = run_dir.join(format!("{run_key}.ktstr.json"));
            std::fs::write(&sidecar_path, json).expect("write fixture sidecar");
        }

        let filter_a = RowFilter {
            schedulers: vec!["scx_alpha".to_string()],
            ..RowFilter::default()
        };
        let filter_b = RowFilter {
            schedulers: vec!["scx_beta".to_string()],
            ..RowFilter::default()
        };

        // Positive: --dir threads to collect_pool; the two
        // partitions resolve and the comparison runs without
        // bailing. Identical metric values mean exit 0 (no
        // regressions); we only care that the call succeeds.
        let exit = compare_partitions(
            &filter_a,
            &filter_b,
            None,
            &ComparisonPolicy::default(),
            Some(alt_root.path()),
            false,
        )
        .expect("compare_partitions must pool sidecars under --dir override");
        assert_eq!(
            exit, 0,
            "byte-identical metrics across the two scheduler \
             partitions must yield zero regressions (exit 0). \
             A non-zero exit means either the partitions loaded \
             different data than written above or compare_rows \
             regressed on identical inputs.",
        );
    }

    // -- render_dirty_warning --

    /// No `-dirty` commit values on either side returns `None` so
    /// the caller emits no banner. Pins the silent-when-clean
    /// contract that lets `warn_on_dirty_builds` be a no-op for
    /// release-quality runs.
    #[test]
    fn render_dirty_warning_silent_when_no_dirty_commits() {
        let mut row = make_row("scn", "topo", true, 1.0);
        row.commit = Some("abcdef1".into());
        row.kernel_commit = Some("0123456".into());
        let other = row.clone();
        assert!(
            super::render_dirty_warning(&[row], &[other]).is_none(),
            "clean rows on both sides must yield no warning"
        );
    }

    /// Empty input on both sides is silent — `compare_partitions`
    /// bails before the call when either side is empty, but the
    /// helper itself must still degrade cleanly.
    #[test]
    fn render_dirty_warning_silent_on_empty_inputs() {
        assert!(
            super::render_dirty_warning(&[], &[]).is_none(),
            "empty inputs must yield no warning"
        );
    }

    /// Dirty `kernel_commit` values across both sides are deduped
    /// into one block under "kernel source", with each distinct
    /// value listed once and `commit` (project) absent because
    /// none of the rows are dirty on that dimension.
    #[test]
    fn render_dirty_warning_kernel_only_dedupes_values_across_sides() {
        let mut a = make_row("scn", "topo", true, 1.0);
        a.kernel_commit = Some("aaaaaaa-dirty".into());
        a.commit = Some("clean01".into());
        let mut a2 = make_row("scn2", "topo", true, 1.0);
        a2.kernel_commit = Some("aaaaaaa-dirty".into()); // same as a
        let mut b = make_row("scn", "topo", true, 1.0);
        b.kernel_commit = Some("bbbbbbb-dirty".into());
        let text = super::render_dirty_warning(&[a, a2], &[b])
            .expect("dirty kernel_commit must yield warning");
        assert!(
            text.contains("warning: comparison includes dirty builds:"),
            "missing header in {text:?}"
        );
        assert_eq!(
            text.matches("kernel source: aaaaaaa-dirty").count(),
            1,
            "duplicate kernel_commit must be deduped, got {text:?}"
        );
        assert!(
            text.contains("kernel source: bbbbbbb-dirty"),
            "second distinct dirty kernel_commit must be listed, got {text:?}"
        );
        assert!(
            !text.contains("project:"),
            "no -dirty project commit; the project line must not appear: {text:?}"
        );
        assert!(
            text.contains("Dirty runs overwrite previous results with the same HEAD."),
            "missing trailer line 1 in {text:?}"
        );
        assert!(
            text.contains("Commit changes for reproducible-ish comparisons."),
            "missing trailer line 2 in {text:?}"
        );
    }

    /// Dirty `commit` (project) values are listed under "project"
    /// when no `kernel_commit` is dirty, so each dimension renders
    /// only when populated.
    #[test]
    fn render_dirty_warning_project_only_omits_kernel_section() {
        let mut a = make_row("scn", "topo", true, 1.0);
        a.commit = Some("ccccccc-dirty".into());
        let text = super::render_dirty_warning(&[a], &[]).expect("dirty commit must yield warning");
        assert!(
            text.contains("project: ccccccc-dirty"),
            "expected project line in {text:?}"
        );
        assert!(
            !text.contains("kernel source:"),
            "kernel section must not appear when only project is dirty: {text:?}"
        );
    }

    /// Both dimensions dirty: the warning lists "kernel source"
    /// before "project" in stable order so byte-identical inputs
    /// always render byte-identically. BTreeSet ordering of distinct
    /// hashes within each dimension is also pinned (lex order).
    #[test]
    fn render_dirty_warning_both_dimensions_in_stable_order() {
        let mut a = make_row("scn", "topo", true, 1.0);
        a.kernel_commit = Some("kkkkk22-dirty".into());
        a.commit = Some("pppp222-dirty".into());
        let mut b = make_row("scn", "topo", true, 1.0);
        b.kernel_commit = Some("kkkkk11-dirty".into());
        b.commit = Some("pppp111-dirty".into());
        let text = super::render_dirty_warning(&[a], &[b])
            .expect("both dimensions dirty must yield warning");
        let kernel11 = text
            .find("kernel source: kkkkk11-dirty")
            .expect("kernel11 line absent");
        let kernel22 = text
            .find("kernel source: kkkkk22-dirty")
            .expect("kernel22 line absent");
        let project11 = text
            .find("project: pppp111-dirty")
            .expect("project11 line absent");
        let project22 = text
            .find("project: pppp222-dirty")
            .expect("project22 line absent");
        assert!(
            kernel11 < kernel22,
            "kernel section must list values in lex order: {text:?}"
        );
        assert!(
            project11 < project22,
            "project section must list values in lex order: {text:?}"
        );
        assert!(
            kernel22 < project11,
            "kernel section must precede project section: {text:?}"
        );
    }

    /// `None` commit fields and clean (suffix-free) values on the
    /// other rows do not contribute to either set, so the warning
    /// only mentions the actually-dirty hash.
    #[test]
    fn render_dirty_warning_skips_none_and_clean_values() {
        let mut clean_a = make_row("a", "topo", true, 1.0);
        clean_a.commit = Some("clean01".into());
        clean_a.kernel_commit = None;
        let mut dirty_b = make_row("b", "topo", true, 1.0);
        dirty_b.commit = None;
        dirty_b.kernel_commit = Some("dddddd1-dirty".into());
        let text = super::render_dirty_warning(&[clean_a], &[dirty_b])
            .expect("at least one dirty value must yield warning");
        assert!(
            text.contains("kernel source: dddddd1-dirty"),
            "dirty kernel_commit must surface in {text:?}"
        );
        assert!(
            !text.contains("project:"),
            "no dirty project commit; project section must be absent in {text:?}"
        );
        assert!(
            !text.contains("clean01"),
            "clean commit values must not appear in {text:?}"
        );
    }

    // -- RowFilter / apply_row_filters --

    /// Helper that builds a `GauntletRow` with controllable
    /// scheduler / topology / work_type / kernel_version / flags
    /// for the filter tests. The metric fields default to harmless
    /// passing values; tests are interested in identity-field
    /// matching, not metrics.
    fn make_filter_row(
        scenario: &str,
        scheduler: &str,
        topology: &str,
        work_type: &str,
        kernel_version: Option<&str>,
        flags: &[&str],
    ) -> GauntletRow {
        GauntletRow {
            scenario: scenario.into(),
            topology: topology.into(),
            work_type: work_type.into(),
            scheduler: scheduler.into(),
            kernel_version: kernel_version.map(str::to_owned),
            commit: None,
            kernel_commit: None,
            run_source: None,
            flags: flags.iter().map(|s| (*s).to_owned()).collect(),
            passed: true,
            skipped: false,
            spread: 0.0,
            gap_ms: 0,
            migrations: 0,
            migration_ratio: 0.0,
            imbalance_ratio: 0.0,
            max_dsq_depth: 0,
            stall_count: 0,
            fallback_count: 0,
            keep_last_count: 0,
            worst_p99_wake_latency_us: 0.0,
            worst_median_wake_latency_us: 0.0,
            worst_wake_latency_cv: 0.0,
            total_iterations: 0,
            worst_mean_run_delay_us: 0.0,
            worst_run_delay_us: 0.0,
            worst_wake_latency_tail_ratio: 0.0,
            worst_iterations_per_worker: 0.0,
            page_locality: 0.0,
            cross_node_migration_ratio: 0.0,
            ext_metrics: BTreeMap::new(),
        }
    }

    /// Default `RowFilter` (every field None/empty) matches every
    /// row — it's the identity filter. Pins the no-op contract so a
    /// future regression that flipped the default to a "match
    /// nothing" semantic lands here.
    #[test]
    fn row_filter_default_matches_every_row() {
        let row = make_filter_row("t", "scx_a", "1n2l4c1t", "CpuSpin", Some("6.14.2"), &[]);
        let filter = RowFilter::default();
        assert!(filter.matches(&row), "empty filter must match every row");
    }

    /// `--scheduler` is strict equality, NOT substring. A filter of
    /// `"scx"` does not match a row with scheduler `"scx_rusty"`.
    /// Pins the typed-vs-substring asymmetry: -E stays as the
    /// substring knob; typed flags exact-match.
    #[test]
    fn row_filter_scheduler_strict_equality_rejects_prefix() {
        let row = make_filter_row("t", "scx_rusty", "1n2l4c1t", "CpuSpin", None, &[]);
        let filter = RowFilter {
            schedulers: vec!["scx".to_string()],
            ..RowFilter::default()
        };
        assert!(
            !filter.matches(&row),
            "strict-equality scheduler filter must NOT match a prefix; \
             got match for scheduler=`scx_rusty` against filter=`scx`",
        );
    }

    /// Exact scheduler match passes; the strict-equality contract's
    /// happy path.
    #[test]
    fn row_filter_scheduler_strict_equality_matches_exact() {
        let row = make_filter_row("t", "scx_rusty", "1n2l4c1t", "CpuSpin", None, &[]);
        let filter = RowFilter {
            schedulers: vec!["scx_rusty".to_string()],
            ..RowFilter::default()
        };
        assert!(filter.matches(&row));
    }

    /// `--kernel 6.14.2` against a row whose `kernel_version` is
    /// `None` must NOT match — the operator opted in to a specific
    /// kernel and a None-row would silently dilute the filtered set.
    #[test]
    fn row_filter_kernel_none_row_never_matches_populated_filter() {
        let row = make_filter_row("t", "scx_a", "1n2l4c1t", "CpuSpin", None, &[]);
        let filter = RowFilter {
            kernels: vec!["6.14.2".to_string()],
            ..RowFilter::default()
        };
        assert!(
            !filter.matches(&row),
            "None-row must not match populated filter; got dilution",
        );
    }

    /// `--kernel 6.14.2` against a row whose `kernel_version` is
    /// `Some("6.14.2")` matches.
    #[test]
    fn row_filter_kernel_exact_match() {
        let row = make_filter_row("t", "scx_a", "1n2l4c1t", "CpuSpin", Some("6.14.2"), &[]);
        let filter = RowFilter {
            kernels: vec!["6.14.2".to_string()],
            ..RowFilter::default()
        };
        assert!(filter.matches(&row));
    }

    /// `--kernel 6.14.2` against a row whose `kernel_version` is
    /// `Some("6.14.3")` rejects.
    #[test]
    fn row_filter_kernel_mismatch_rejects() {
        let row = make_filter_row("t", "scx_a", "1n2l4c1t", "CpuSpin", Some("6.14.3"), &[]);
        let filter = RowFilter {
            kernels: vec!["6.14.2".to_string()],
            ..RowFilter::default()
        };
        assert!(!filter.matches(&row));
    }

    /// Repeatable `--kernel A --kernel B` is OR-combined: a row
    /// matches iff its `kernel_version` equals ANY listed entry.
    /// Pins the multi-value semantic.
    #[test]
    fn row_filter_kernels_or_combined_matches_any_listed() {
        let row_a = make_filter_row("t", "scx_a", "1n2l4c1t", "CpuSpin", Some("6.14.2"), &[]);
        let row_b = make_filter_row("t", "scx_a", "1n2l4c1t", "CpuSpin", Some("6.15.0"), &[]);
        let row_c = make_filter_row("t", "scx_a", "1n2l4c1t", "CpuSpin", Some("6.16.0"), &[]);
        let filter = RowFilter {
            kernels: vec!["6.14.2".to_string(), "6.15.0".to_string()],
            ..RowFilter::default()
        };
        assert!(filter.matches(&row_a), "first listed kernel must match");
        assert!(filter.matches(&row_b), "second listed kernel must match");
        assert!(
            !filter.matches(&row_c),
            "kernel outside the listed set must reject",
        );
    }

    /// Repeatable `--scheduler A --scheduler B` is OR-combined:
    /// a row matches iff its `scheduler` equals ANY listed entry.
    /// Pins the multi-value semantic for the
    /// post-Vec-promotion `schedulers` field; before promotion
    /// `--scheduler` was a single-value `Option<String>` and the
    /// OR semantic did not exist.
    #[test]
    fn row_filter_schedulers_or_combined_matches_any_listed() {
        let row_a = make_filter_row("t", "scx_alpha", "1n2l4c1t", "CpuSpin", None, &[]);
        let row_b = make_filter_row("t", "scx_beta", "1n2l4c1t", "CpuSpin", None, &[]);
        let row_c = make_filter_row("t", "scx_gamma", "1n2l4c1t", "CpuSpin", None, &[]);
        let filter = RowFilter {
            schedulers: vec!["scx_alpha".to_string(), "scx_beta".to_string()],
            ..RowFilter::default()
        };
        assert!(filter.matches(&row_a), "first listed scheduler must match",);
        assert!(filter.matches(&row_b), "second listed scheduler must match",);
        assert!(
            !filter.matches(&row_c),
            "scheduler outside the listed set must reject",
        );
    }

    /// Repeatable `--topology A --topology B` is OR-combined:
    /// a row matches iff its `topology` equals ANY listed entry.
    /// Mirror of
    /// `row_filter_schedulers_or_combined_matches_any_listed`
    /// for the topologies field.
    #[test]
    fn row_filter_topologies_or_combined_matches_any_listed() {
        let row_a = make_filter_row("t", "scx_a", "1n2l4c1t", "CpuSpin", None, &[]);
        let row_b = make_filter_row("t", "scx_a", "1n2l4c2t", "CpuSpin", None, &[]);
        let row_c = make_filter_row("t", "scx_a", "1n4l8c1t", "CpuSpin", None, &[]);
        let filter = RowFilter {
            topologies: vec!["1n2l4c1t".to_string(), "1n2l4c2t".to_string()],
            ..RowFilter::default()
        };
        assert!(filter.matches(&row_a), "first listed topology must match",);
        assert!(filter.matches(&row_b), "second listed topology must match",);
        assert!(
            !filter.matches(&row_c),
            "topology outside the listed set must reject",
        );
    }

    /// Repeatable `--work-type A --work-type B` is OR-combined:
    /// a row matches iff its `work_type` equals ANY listed
    /// entry. Mirror of
    /// `row_filter_schedulers_or_combined_matches_any_listed`
    /// for the work_types field.
    #[test]
    fn row_filter_work_types_or_combined_matches_any_listed() {
        let row_a = make_filter_row("t", "scx_a", "1n2l4c1t", "CpuSpin", None, &[]);
        let row_b = make_filter_row("t", "scx_a", "1n2l4c1t", "PageFaultChurn", None, &[]);
        let row_c = make_filter_row("t", "scx_a", "1n2l4c1t", "MutexContention", None, &[]);
        let filter = RowFilter {
            work_types: vec!["CpuSpin".to_string(), "PageFaultChurn".to_string()],
            ..RowFilter::default()
        };
        assert!(filter.matches(&row_a), "first listed work_type must match",);
        assert!(filter.matches(&row_b), "second listed work_type must match",);
        assert!(
            !filter.matches(&row_c),
            "work_type outside the listed set must reject",
        );
    }

    /// `--project-commit abcdef1` against a row whose `commit` is `None`
    /// must NOT match — same opt-in policy as `--kernel`. Mirror
    /// of `row_filter_kernel_none_row_never_matches_populated_filter`
    /// for the project-commit field.
    #[test]
    fn row_filter_commit_none_row_never_matches_populated_filter() {
        let row = make_filter_row("t", "scx_a", "1n2l4c1t", "CpuSpin", None, &[]);
        let filter = RowFilter {
            project_commits: vec!["abcdef1".to_string()],
            ..RowFilter::default()
        };
        assert!(
            !filter.matches(&row),
            "None-commit row must not match populated filter; \
             got dilution",
        );
    }

    /// `--project-commit abcdef1` against a row whose `commit` is
    /// `Some("abcdef1")` matches; `Some("other")` rejects.
    /// Pins the strict-equality contract for commit, including
    /// the OR-combined multi-value semantic and the `-dirty`
    /// suffix's contribution to identity (a clean and dirty run
    /// of the same HEAD bucket separately).
    #[test]
    fn row_filter_commit_exact_match_and_or_combined() {
        let mut row_clean = make_filter_row("t", "scx_a", "1n2l4c1t", "CpuSpin", None, &[]);
        row_clean.commit = Some("abcdef1".to_string());
        let mut row_dirty = make_filter_row("t", "scx_a", "1n2l4c1t", "CpuSpin", None, &[]);
        row_dirty.commit = Some("abcdef1-dirty".to_string());
        let mut row_other = make_filter_row("t", "scx_a", "1n2l4c1t", "CpuSpin", None, &[]);
        row_other.commit = Some("fedcba2".to_string());

        let filter_single = RowFilter {
            project_commits: vec!["abcdef1".to_string()],
            ..RowFilter::default()
        };
        assert!(
            filter_single.matches(&row_clean),
            "exact commit match must succeed",
        );
        assert!(
            !filter_single.matches(&row_dirty),
            "`abcdef1-dirty` must NOT match a filter for `abcdef1` — \
             the suffix is part of identity, so the dirty run buckets \
             separately from the clean run of the same HEAD",
        );
        assert!(
            !filter_single.matches(&row_other),
            "different commit must reject",
        );

        let filter_or = RowFilter {
            project_commits: vec!["abcdef1".to_string(), "fedcba2".to_string()],
            ..RowFilter::default()
        };
        assert!(
            filter_or.matches(&row_clean),
            "first listed commit must match in OR-combined filter",
        );
        assert!(
            filter_or.matches(&row_other),
            "second listed commit must match in OR-combined filter",
        );
        assert!(
            !filter_or.matches(&row_dirty),
            "`abcdef1-dirty` must still reject — the suffix-bearing \
             form is its own identity even in OR-combined mode",
        );
    }

    /// `--kernel-commit kabcde7` against a row whose
    /// `kernel_commit` is `None` must NOT match — same opt-in
    /// policy as `--project-commit` and `--kernel`. Mirror of
    /// `row_filter_commit_none_row_never_matches_populated_filter`
    /// for the kernel-commit field.
    #[test]
    fn row_filter_kernel_commit_none_row_never_matches_populated_filter() {
        let row = make_filter_row("t", "scx_a", "1n2l4c1t", "CpuSpin", None, &[]);
        let filter = RowFilter {
            kernel_commits: vec!["kabcde7".to_string()],
            ..RowFilter::default()
        };
        assert!(
            !filter.matches(&row),
            "None-kernel-commit row must not match populated filter; \
             got dilution",
        );
    }

    /// `--kernel-commit kabcde7` against a row whose
    /// `kernel_commit` is `Some("kabcde7")` matches;
    /// `Some("other")` rejects. Pins the strict-equality
    /// contract for kernel_commit, including the OR-combined
    /// multi-value semantic and the `-dirty` suffix's
    /// contribution to identity (a clean and dirty run of the
    /// same kernel HEAD bucket separately).
    #[test]
    fn row_filter_kernel_commit_exact_match_and_or_combined() {
        let mut row_clean = make_filter_row("t", "scx_a", "1n2l4c1t", "CpuSpin", None, &[]);
        row_clean.kernel_commit = Some("kabcde7".to_string());
        let mut row_dirty = make_filter_row("t", "scx_a", "1n2l4c1t", "CpuSpin", None, &[]);
        row_dirty.kernel_commit = Some("kabcde7-dirty".to_string());
        let mut row_other = make_filter_row("t", "scx_a", "1n2l4c1t", "CpuSpin", None, &[]);
        row_other.kernel_commit = Some("fedcba2".to_string());

        let filter_single = RowFilter {
            kernel_commits: vec!["kabcde7".to_string()],
            ..RowFilter::default()
        };
        assert!(
            filter_single.matches(&row_clean),
            "exact kernel_commit match must succeed",
        );
        assert!(
            !filter_single.matches(&row_dirty),
            "`kabcde7-dirty` must NOT match a filter for `kabcde7` — \
             the suffix is part of identity, so the dirty run buckets \
             separately from the clean run of the same kernel HEAD",
        );
        assert!(
            !filter_single.matches(&row_other),
            "different kernel_commit must reject",
        );

        let filter_or = RowFilter {
            kernel_commits: vec!["kabcde7".to_string(), "fedcba2".to_string()],
            ..RowFilter::default()
        };
        assert!(
            filter_or.matches(&row_clean),
            "first listed kernel_commit must match in OR-combined filter",
        );
        assert!(
            filter_or.matches(&row_other),
            "second listed kernel_commit must match in OR-combined filter",
        );
        assert!(
            !filter_or.matches(&row_dirty),
            "`kabcde7-dirty` must still reject — the suffix-bearing \
             form is its own identity even in OR-combined mode",
        );
    }

    /// `--kernel-commit` and `--project-commit` filter on DISTINCT row
    /// fields. Pins the field non-aliasing: a row whose
    /// `kernel_commit` matches but whose `commit` does not (or
    /// vice versa) must reject. A regression that cross-wired
    /// the `matches()` arms (e.g. `kernel_commits` checked
    /// against `row.commit`) would silently dilute filtered
    /// sets.
    #[test]
    fn row_filter_kernel_commit_and_commit_filter_distinct_fields() {
        let mut row = make_filter_row("t", "scx_a", "1n2l4c1t", "CpuSpin", None, &[]);
        row.commit = Some("project1".to_string());
        row.kernel_commit = Some("kernel1".to_string());

        // Filter on kernel_commit only — commit dimension is unconstrained.
        let kc_only = RowFilter {
            kernel_commits: vec!["kernel1".to_string()],
            ..RowFilter::default()
        };
        assert!(
            kc_only.matches(&row),
            "kernel_commit match with no commit filter must accept",
        );

        let kc_mismatch = RowFilter {
            kernel_commits: vec!["project1".to_string()],
            ..RowFilter::default()
        };
        assert!(
            !kc_mismatch.matches(&row),
            "kernel_commits filter must check `kernel_commit` not `commit` — \
             a regression that cross-wired the fields would accept here",
        );

        // Filter on commit only — kernel_commit dimension is unconstrained.
        let commit_mismatch = RowFilter {
            project_commits: vec!["kernel1".to_string()],
            ..RowFilter::default()
        };
        assert!(
            !commit_mismatch.matches(&row),
            "project_commits filter must check `commit` not `kernel_commit` — \
             a regression that cross-wired the fields would accept here",
        );
    }

    /// `--run-source local` against a row whose `run_source` is
    /// `None` must NOT match — same opt-in policy as `--kernel`,
    /// `--project-commit`, and `--kernel-commit`. The operator wrote
    /// specific tags and a None-row would silently dilute the
    /// filtered set. Mirror of
    /// `row_filter_kernel_commit_none_row_never_matches_populated_filter`
    /// for the `run_source` field.
    #[test]
    fn row_filter_run_source_none_row_never_matches_populated_filter() {
        let row = make_filter_row("t", "scx_a", "1n2l4c1t", "CpuSpin", None, &[]);
        let filter = RowFilter {
            run_sources: vec!["local".to_string()],
            ..RowFilter::default()
        };
        assert!(
            !filter.matches(&row),
            "None-run_source row must not match populated filter; \
             got dilution",
        );
    }

    /// Repeatable `--run-source A --run-source B` is OR-combined: a row
    /// matches iff its `run_source` equals ANY listed entry.
    /// Mirror of `row_filter_kernels_or_combined_matches_any_listed`
    /// for the `run_source` dimension.
    #[test]
    fn row_filter_run_sources_or_combined_matches_any_listed() {
        let mut row_local = make_filter_row("t", "scx_a", "1n2l4c1t", "CpuSpin", None, &[]);
        row_local.run_source = Some("local".to_string());
        let mut row_ci = make_filter_row("t", "scx_a", "1n2l4c1t", "CpuSpin", None, &[]);
        row_ci.run_source = Some("ci".to_string());
        let mut row_archive = make_filter_row("t", "scx_a", "1n2l4c1t", "CpuSpin", None, &[]);
        row_archive.run_source = Some("archive".to_string());
        let filter = RowFilter {
            run_sources: vec!["local".to_string(), "ci".to_string()],
            ..RowFilter::default()
        };
        assert!(
            filter.matches(&row_local),
            "first listed run_source must match",
        );
        assert!(
            filter.matches(&row_ci),
            "second listed run_source must match",
        );
        assert!(
            !filter.matches(&row_archive),
            "run_source outside the listed set must reject",
        );
    }

    /// `--run-source` and `--kernel-commit` filter on DISTINCT row
    /// fields. Pins the field non-aliasing: a row whose
    /// `run_source` matches but whose `kernel_commit` does not
    /// (or vice versa) must reject. A regression that cross-wired
    /// the `matches()` arms (e.g. `run_sources` checked against
    /// `row.kernel_commit`) would silently dilute filtered sets.
    /// Mirror of
    /// `row_filter_kernel_commit_and_commit_filter_distinct_fields`
    /// for the `run_source` × `kernel_commit` cross-wire surface.
    #[test]
    fn row_filter_run_sources_and_kernel_commits_are_distinct_fields() {
        let mut row = make_filter_row("t", "scx_a", "1n2l4c1t", "CpuSpin", None, &[]);
        row.run_source = Some("local".to_string());
        row.kernel_commit = None;
        let filter = RowFilter {
            run_sources: vec!["local".to_string()],
            kernel_commits: vec!["abc1234".to_string()],
            ..RowFilter::default()
        };
        assert!(
            !filter.matches(&row),
            "AND composition must reject when kernel_commit gate \
             fails (row's kernel_commit is None) even though the \
             run_source gate matches; a regression that cross-wired \
             run_sources against `row.kernel_commit` would accept here",
        );

        // Symmetric arm: run_source mismatches but kernel_commit
        // matches. Whole filter must still reject.
        let mut row2 = make_filter_row("t", "scx_a", "1n2l4c1t", "CpuSpin", None, &[]);
        row2.run_source = Some("ci".to_string());
        row2.kernel_commit = Some("abc1234".to_string());
        let filter2 = RowFilter {
            run_sources: vec!["local".to_string()],
            kernel_commits: vec!["abc1234".to_string()],
            ..RowFilter::default()
        };
        assert!(
            !filter2.matches(&row2),
            "AND composition must reject when run_source gate \
             fails even though kernel_commit gate passes; a \
             regression that cross-wired kernel_commits against \
             `row.run_source` would accept here",
        );
    }

    /// `--project-commit` and `--kernel` compose with AND semantics: a
    /// populated commit filter and a populated kernel filter must
    /// BOTH match for the row to survive. Pins the cross-field
    /// composition rule for the new commit field, mirroring the
    /// existing multi-field test for scheduler+topology+kernel.
    #[test]
    fn row_filter_commit_and_kernel_compose_and() {
        let mut row = make_filter_row("t", "scx_a", "1n2l4c1t", "CpuSpin", Some("6.14.2"), &[]);
        row.commit = Some("abcdef1".to_string());
        let filter_both_match = RowFilter {
            kernels: vec!["6.14.2".to_string()],
            project_commits: vec!["abcdef1".to_string()],
            ..RowFilter::default()
        };
        assert!(
            filter_both_match.matches(&row),
            "both filters matching must accept the row",
        );
        let filter_kernel_only_match = RowFilter {
            kernels: vec!["6.14.2".to_string()],
            project_commits: vec!["fedcba2".to_string()],
            ..RowFilter::default()
        };
        assert!(
            !filter_kernel_only_match.matches(&row),
            "AND composition must reject when commit mismatches even \
             though kernel matches",
        );
    }

    /// `--topology 1n2l4c1t` strict-equal against the row's
    /// rendered topology. The filter is the same string the
    /// `Topology::Display` impl emits and `cargo ktstr stats list`
    /// shows; passing the exact form that appears in the listing
    /// is the operator's expected workflow.
    #[test]
    fn row_filter_topology_strict_equality() {
        let row = make_filter_row("t", "scx_a", "1n2l4c1t", "CpuSpin", None, &[]);
        let filter_match = RowFilter {
            topologies: vec!["1n2l4c1t".to_string()],
            ..RowFilter::default()
        };
        assert!(filter_match.matches(&row));
        let filter_miss = RowFilter {
            topologies: vec!["1n2l4c2t".to_string()],
            ..RowFilter::default()
        };
        assert!(!filter_miss.matches(&row));
    }

    /// Repeatable `--flag` is AND-combined: every entry in the
    /// filter must appear in the row's flags vec. The row may
    /// carry additional flags (the filter is at-least-these, not
    /// exactly-these) — pinned here by adding `extra` to the row
    /// and confirming it doesn't break the match.
    #[test]
    fn row_filter_flags_and_combined_subset() {
        let row = make_filter_row(
            "t",
            "scx_a",
            "1n2l4c1t",
            "CpuSpin",
            None,
            &["llc", "rusty_balance", "extra"],
        );
        let filter = RowFilter {
            flags: vec!["llc".to_string(), "rusty_balance".to_string()],
            ..RowFilter::default()
        };
        assert!(
            filter.matches(&row),
            "AND-combined flags must match when row has all required \
             entries (extra flags are fine); got rejection",
        );
    }

    /// AND-combined: a single missing required flag rejects the
    /// whole match, even when other required flags are present.
    #[test]
    fn row_filter_flags_missing_required_rejects() {
        let row = make_filter_row("t", "scx_a", "1n2l4c1t", "CpuSpin", None, &["llc"]);
        let filter = RowFilter {
            flags: vec!["llc".to_string(), "rusty_balance".to_string()],
            ..RowFilter::default()
        };
        assert!(
            !filter.matches(&row),
            "missing single required flag must reject the whole match",
        );
    }

    /// Multiple typed filters compose with AND semantics: every
    /// populated field must match. A mismatch on any one field
    /// rejects the whole match. Pinned via a row that matches 3
    /// of 4 filter fields and assertion that it still rejects.
    #[test]
    fn row_filter_multi_field_and_composes() {
        let row = make_filter_row(
            "t",
            "scx_a",
            "1n2l4c1t",
            "CpuSpin",
            Some("6.14.2"),
            &["llc"],
        );
        // 3 of 4 typed fields match (scheduler, topology, kernels);
        // work_type mismatches. Whole filter must reject.
        let filter = RowFilter {
            schedulers: vec!["scx_a".to_string()],
            topologies: vec!["1n2l4c1t".to_string()],
            kernels: vec!["6.14.2".to_string()],
            work_types: vec!["YieldHeavy".to_string()],
            ..RowFilter::default()
        };
        assert!(
            !filter.matches(&row),
            "AND composition must reject when any single field mismatches; \
             got match despite work_type divergence",
        );
    }

    /// `apply_row_filters` preserves the original row order and
    /// drops only non-matching rows. Pinned by feeding a 3-row
    /// vec where row 1 of 3 matches; result must be a 1-element
    /// vec with the original middle row.
    #[test]
    fn apply_row_filters_preserves_order_drops_mismatch() {
        let rows = vec![
            make_filter_row("t1", "scx_a", "1n2l4c1t", "CpuSpin", None, &[]),
            make_filter_row("t2", "scx_b", "1n2l4c1t", "CpuSpin", None, &[]),
            make_filter_row("t3", "scx_a", "1n2l4c1t", "CpuSpin", None, &[]),
        ];
        let filter = RowFilter {
            schedulers: vec!["scx_b".to_string()],
            ..RowFilter::default()
        };
        let kept = apply_row_filters(&rows, &filter);
        assert_eq!(kept.len(), 1, "expected 1 surviving row, got {kept:?}");
        assert_eq!(kept[0].scenario, "t2");
    }

    /// `apply_row_filters` with the default filter is the identity
    /// — every row survives in original order.
    #[test]
    fn apply_row_filters_default_is_identity() {
        let rows = vec![
            make_filter_row("t1", "scx_a", "1n2l4c1t", "CpuSpin", None, &[]),
            make_filter_row(
                "t2",
                "scx_b",
                "1n2l4c2t",
                "YieldHeavy",
                Some("6.14.2"),
                &["llc"],
            ),
        ];
        let kept = apply_row_filters(&rows, &RowFilter::default());
        assert_eq!(kept.len(), rows.len());
        for (a, b) in kept.iter().zip(rows.iter()) {
            assert_eq!(a.scenario, b.scenario);
        }
    }

    // -- group_and_average / AveragedGroup --

    /// Mutate a row's metric fields away from defaults so
    /// aggregation has a non-zero signal to average. Returns the
    /// row reference for chaining.
    fn paint_metrics(row: &mut GauntletRow, spread: f64, gap_ms: u64, migrations: u64, iters: u64) {
        row.spread = spread;
        row.gap_ms = gap_ms;
        row.migrations = migrations;
        row.migration_ratio = spread / 100.0;
        row.imbalance_ratio = spread / 10.0;
        row.max_dsq_depth = (gap_ms / 10) as u32;
        row.stall_count = (migrations / 10) as usize;
        row.fallback_count = migrations as i64;
        row.keep_last_count = -(migrations as i64);
        row.worst_p99_wake_latency_us = spread * 2.0;
        row.worst_median_wake_latency_us = spread;
        row.worst_wake_latency_cv = spread / 50.0;
        row.total_iterations = iters;
        row.worst_mean_run_delay_us = gap_ms as f64;
        row.worst_run_delay_us = (gap_ms * 2) as f64;
        row.worst_wake_latency_tail_ratio = spread / 25.0;
        row.worst_iterations_per_worker = iters as f64 / 10.0;
        row.page_locality = 1.0 - spread / 100.0;
        row.cross_node_migration_ratio = spread / 200.0;
    }

    /// Empty input produces zero aggregated rows. Pins the empty-
    /// vec edge case so callers iterating over the result vector
    /// don't need to special-case the `--average` path on empty
    /// run directories.
    #[test]
    fn group_and_average_empty_input_yields_empty_output() {
        let out = group_and_average(&[]);
        assert!(out.is_empty());
    }

    /// Single passing contributor: aggregate is a faithful copy
    /// of the input, with `passes_observed = total_observed = 1`.
    /// Pins the trivial pass-through path so a regression in the
    /// `denom` math (e.g. division by `total_observed` instead of
    /// `passes_observed`) lands here.
    #[test]
    fn group_and_average_single_pass_passes_through_metrics() {
        let mut row = make_row("t", "tiny-1llc", true, 0.0);
        paint_metrics(&mut row, 12.0, 200, 50, 1000);
        let out = group_and_average(std::slice::from_ref(&row));
        assert_eq!(out.len(), 1);
        let ar = &out[0];
        assert_eq!(ar.passes_observed, 1);
        assert_eq!(ar.total_observed, 1);
        assert!(ar.row.passed);
        assert!(!ar.row.skipped);
        assert_eq!(ar.row.spread, 12.0);
        assert_eq!(ar.row.gap_ms, 200);
        assert_eq!(ar.row.migrations, 50);
        assert_eq!(ar.row.total_iterations, 1000);
        assert_eq!(ar.row.fallback_count, 50);
        assert_eq!(ar.row.keep_last_count, -50);
        assert_eq!(ar.row.worst_p99_wake_latency_us, 24.0);
    }

    /// Three passing contributors with the same key are folded
    /// into a single aggregate carrying the arithmetic mean of
    /// every metric field. f64 means are exact (modulo IEEE
    /// rounding); u64/i64 means are rounded to nearest.
    #[test]
    fn group_and_average_multi_pass_arithmetic_mean() {
        let mut a = make_row("t", "tiny-1llc", true, 0.0);
        paint_metrics(&mut a, 10.0, 100, 30, 900);
        let mut b = make_row("t", "tiny-1llc", true, 0.0);
        paint_metrics(&mut b, 20.0, 200, 60, 1100);
        let mut c = make_row("t", "tiny-1llc", true, 0.0);
        paint_metrics(&mut c, 30.0, 300, 90, 1000);
        let out = group_and_average(&[a, b, c]);
        assert_eq!(out.len(), 1);
        let ar = &out[0];
        assert_eq!(ar.passes_observed, 3);
        assert_eq!(ar.total_observed, 3);
        assert!(ar.row.passed);
        assert!(!ar.row.skipped);
        // f64 mean: (10 + 20 + 30) / 3 = 20.0 exactly.
        assert_eq!(ar.row.spread, 20.0);
        // u64 rounded mean: (100 + 200 + 300) / 3 = 200.0 exactly.
        assert_eq!(ar.row.gap_ms, 200);
        // u64 rounded mean: (30 + 60 + 90) / 3 = 60.
        assert_eq!(ar.row.migrations, 60);
        // u64 rounded mean: (900 + 1100 + 1000) / 3 = 1000.
        assert_eq!(ar.row.total_iterations, 1000);
        // i64 mean for fallback_count: (30 + 60 + 90)/3 = 60.
        assert_eq!(ar.row.fallback_count, 60);
        // i64 mean for keep_last_count: (-30 + -60 + -90)/3 = -60.
        assert_eq!(ar.row.keep_last_count, -60);
        // f64 mean for derived field
        // worst_p99_wake_latency_us: (20 + 40 + 60)/3 = 40.
        assert_eq!(ar.row.worst_p99_wake_latency_us, 40.0);
    }

    /// Different (scenario, topology, work_type, flags) groups
    /// produce distinct aggregates — the four-tuple is the join
    /// key. Pins the group-key contract so a regression that
    /// dropped flags from the key would land here as a collision.
    #[test]
    fn group_and_average_distinct_groups_stay_separate() {
        let mut a = make_row("alpha", "tiny-1llc", true, 0.0);
        paint_metrics(&mut a, 10.0, 100, 30, 1000);
        let mut b = make_row("beta", "tiny-1llc", true, 0.0);
        paint_metrics(&mut b, 50.0, 500, 100, 2000);
        let out = group_and_average(&[a, b]);
        assert_eq!(out.len(), 2);
        // First-seen iteration order preserved (alpha before beta).
        assert_eq!(out[0].row.scenario, "alpha");
        assert_eq!(out[1].row.scenario, "beta");
    }

    /// Different `flags` profiles for the same (scenario,
    /// topology, work_type) tuple yield distinct aggregates.
    /// Mirrors the `compare_rows_same_key_different_flags_do_not_collide`
    /// pin for the join key — averaging must respect the same
    /// four-tuple.
    #[test]
    fn group_and_average_different_flags_stay_separate() {
        let mut llc1 = make_row("t", "tiny-1llc", true, 0.0);
        llc1.flags = vec!["llc".to_string()];
        paint_metrics(&mut llc1, 10.0, 100, 30, 1000);
        let mut llc2 = make_row("t", "tiny-1llc", true, 0.0);
        llc2.flags = vec!["llc".to_string()];
        paint_metrics(&mut llc2, 14.0, 140, 50, 1200);
        let mut borrow1 = make_row("t", "tiny-1llc", true, 0.0);
        borrow1.flags = vec!["borrow".to_string()];
        paint_metrics(&mut borrow1, 80.0, 800, 200, 5000);
        let out = group_and_average(&[llc1, llc2, borrow1]);
        assert_eq!(out.len(), 2);
        let llc_ar = out
            .iter()
            .find(|r| r.row.flags == vec!["llc".to_string()])
            .expect("llc aggregate must exist");
        assert_eq!(llc_ar.passes_observed, 2);
        assert_eq!(llc_ar.row.spread, 12.0);
        let borrow_ar = out
            .iter()
            .find(|r| r.row.flags == vec!["borrow".to_string()])
            .expect("borrow aggregate must exist");
        assert_eq!(borrow_ar.passes_observed, 1);
        assert_eq!(borrow_ar.row.spread, 80.0);
    }

    /// Failing contributors are excluded from the metric mean and
    /// flip the aggregate's `passed` to false. The aggregate's
    /// `total_observed` still counts every contributor;
    /// `passes_observed` counts only the clean ones.
    #[test]
    fn group_and_average_failed_contributors_excluded_from_mean_and_flag_aggregate() {
        let mut pass1 = make_row("t", "tiny-1llc", true, 0.0);
        paint_metrics(&mut pass1, 10.0, 100, 30, 1000);
        let mut fail = make_row("t", "tiny-1llc", false, 0.0);
        // The failing row's metrics are pathologically large —
        // if they leaked into the mean, the aggregate's `spread`
        // would explode upward.
        paint_metrics(&mut fail, 10000.0, 99999, 99999, 99999);
        let mut pass2 = make_row("t", "tiny-1llc", true, 0.0);
        paint_metrics(&mut pass2, 30.0, 300, 90, 1000);
        let out = group_and_average(&[pass1, fail, pass2]);
        assert_eq!(out.len(), 1);
        let ar = &out[0];
        assert_eq!(ar.passes_observed, 2);
        assert_eq!(ar.total_observed, 3);
        // ALL-must-pass: a single failure flips the aggregate.
        assert!(
            !ar.row.passed,
            "any failing contributor must flip the aggregate to passed=false",
        );
        // Mean of only the passing entries: (10 + 30) / 2 = 20.0.
        // If the failing row leaked in, this would be ~3346.
        assert_eq!(ar.row.spread, 20.0);
        assert_eq!(ar.row.gap_ms, 200);
    }

    /// Skipped contributors are excluded from the metric mean
    /// and flip the aggregate's `skipped` to true (any-skipped
    /// OR rule). `passes_observed` does not count them; the
    /// passing-only entries still feed the mean cleanly.
    #[test]
    fn group_and_average_skipped_contributors_excluded_from_mean_and_flag_aggregate() {
        let mut pass1 = make_row("t", "tiny-1llc", true, 0.0);
        paint_metrics(&mut pass1, 10.0, 100, 30, 1000);
        let mut skip = make_row("t", "tiny-1llc", true, 0.0);
        skip.skipped = true;
        // Pathological metrics on the skipped row to prove the
        // exclusion is real.
        paint_metrics(&mut skip, 9999.0, 99999, 99999, 99999);
        let mut pass2 = make_row("t", "tiny-1llc", true, 0.0);
        paint_metrics(&mut pass2, 50.0, 500, 70, 2000);
        let out = group_and_average(&[pass1, skip, pass2]);
        assert_eq!(out.len(), 1);
        let ar = &out[0];
        assert_eq!(ar.passes_observed, 2);
        assert_eq!(ar.total_observed, 3);
        assert!(
            ar.row.skipped,
            "any skipped contributor must flip the aggregate to skipped=true",
        );
        assert!(
            !ar.row.passed,
            "skipped aggregate must collapse `passed` to false so compare_rows \
             routes the pair through the skipped_failed gate",
        );
        // Mean of (pass1, pass2): (10 + 50)/2 = 30.0.
        assert_eq!(ar.row.spread, 30.0);
        assert_eq!(ar.row.gap_ms, 300);
    }

    /// All contributors fail: aggregate has `passes_observed = 0`,
    /// `passed = false`, and zero metric values (no contributor
    /// fed the running sums). Pins the divide-by-zero guard:
    /// `denom` must default to 1.0 when `passes_observed = 0`.
    #[test]
    fn group_and_average_all_failed_collapses_to_default_zero_metrics_and_failed_flag() {
        let mut fail1 = make_row("t", "tiny-1llc", false, 0.0);
        paint_metrics(&mut fail1, 99.0, 999, 99, 999);
        let mut fail2 = make_row("t", "tiny-1llc", false, 0.0);
        paint_metrics(&mut fail2, 88.0, 888, 88, 888);
        let out = group_and_average(&[fail1, fail2]);
        assert_eq!(out.len(), 1);
        let ar = &out[0];
        assert_eq!(ar.passes_observed, 0);
        assert_eq!(ar.total_observed, 2);
        assert!(!ar.row.passed);
        // Failed-only group: every metric collapses to its zero
        // default. The aggregate's `passed=false` then routes the
        // pair through compare_rows' skipped_failed gate.
        assert_eq!(ar.row.spread, 0.0);
        assert_eq!(ar.row.gap_ms, 0);
        assert_eq!(ar.row.migrations, 0);
    }

    /// `ext_metrics` keys are unioned across passing
    /// contributors; each key averages over the contributors
    /// that carried it. A key absent on some passing rows is
    /// NOT treated as a stored zero — its denominator is the
    /// present-only count.
    #[test]
    fn group_and_average_ext_metrics_average_per_key_present_count() {
        let mut a = make_row("t", "tiny-1llc", true, 0.0);
        a.ext_metrics.insert("shared".into(), 10.0);
        a.ext_metrics.insert("a_only".into(), 100.0);
        let mut b = make_row("t", "tiny-1llc", true, 0.0);
        b.ext_metrics.insert("shared".into(), 30.0);
        b.ext_metrics.insert("b_only".into(), 200.0);
        let out = group_and_average(&[a, b]);
        assert_eq!(out.len(), 1);
        let ar = &out[0];
        // shared: (10 + 30) / 2 = 20.
        assert_eq!(ar.row.ext_metrics.get("shared"), Some(&20.0));
        // a_only: present only in a → mean over 1 entry = 100.
        assert_eq!(ar.row.ext_metrics.get("a_only"), Some(&100.0));
        // b_only: present only in b → mean over 1 entry = 200.
        assert_eq!(ar.row.ext_metrics.get("b_only"), Some(&200.0));
    }

    /// `group_and_average` preserves first-seen iteration order so
    /// downstream tests against the result remain deterministic
    /// even though the internal map uses BTreeMap (key-sorted)
    /// for storage. Pinned by feeding keys in z→a order and
    /// asserting the output keeps that order.
    #[test]
    fn group_and_average_preserves_first_seen_order() {
        let zebra = make_row("zebra", "tiny-1llc", true, 0.0);
        let alpha = make_row("alpha", "tiny-1llc", true, 0.0);
        let mango = make_row("mango", "tiny-1llc", true, 0.0);
        let out = group_and_average(&[zebra, alpha, mango]);
        let names: Vec<&str> = out.iter().map(|r| r.row.scenario.as_str()).collect();
        assert_eq!(
            names,
            vec!["zebra", "alpha", "mango"],
            "output must follow first-seen iteration order, not key sort",
        );
    }

    /// Cohort with mixed clean/dirty `commit` values (same hex)
    /// renders with `+mixed` appended to the canonical
    /// un-suffixed hex. First contributor is dirty; the second
    /// is clean. Pinning the rendered form catches a regression
    /// where averaging silently kept first-seen behaviour and
    /// hid the WIP-vs-committed disagreement.
    #[test]
    fn group_and_average_mixed_dirty_project_commit_renders_plus_mixed() {
        let mut dirty = make_row("t", "tiny-1llc", true, 0.0);
        dirty.commit = Some("abc1234-dirty".to_string());
        let mut clean = make_row("t", "tiny-1llc", true, 0.0);
        clean.commit = Some("abc1234".to_string());

        let out = group_and_average(&[dirty, clean]);
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].row.commit.as_deref(),
            Some("abc1234+mixed"),
            "mixed clean+dirty must render as `{{hex}}+mixed`, not first-seen",
        );
    }

    /// Same shape on `kernel_commit`. Pins the second commit
    /// dimension separately because the production code uses
    /// two parallel accumulator-state pairs and a regression
    /// could miss one.
    #[test]
    fn group_and_average_mixed_dirty_kernel_commit_renders_plus_mixed() {
        let mut clean = make_row("t", "tiny-1llc", true, 0.0);
        clean.kernel_commit = Some("def5678".to_string());
        let mut dirty = make_row("t", "tiny-1llc", true, 0.0);
        dirty.kernel_commit = Some("def5678-dirty".to_string());

        let out = group_and_average(&[clean, dirty]);
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].row.kernel_commit.as_deref(),
            Some("def5678+mixed"),
            "mixed clean+dirty kernel_commit must render as `{{hex}}+mixed`",
        );
    }

    /// Homogeneous-dirty cohort (every contributor has `-dirty`)
    /// must NOT receive the `+mixed` marker — the cohort agrees
    /// on the working-tree state. Pinning this guards against a
    /// regression where the marker fires on every dirty value
    /// regardless of clean siblings.
    #[test]
    fn group_and_average_all_dirty_keeps_dirty_suffix_no_mixed() {
        let mut a = make_row("t", "tiny-1llc", true, 0.0);
        a.commit = Some("abc1234-dirty".to_string());
        let mut b = make_row("t", "tiny-1llc", true, 0.0);
        b.commit = Some("abc1234-dirty".to_string());

        let out = group_and_average(&[a, b]);
        assert_eq!(
            out[0].row.commit.as_deref(),
            Some("abc1234-dirty"),
            "homogeneous-dirty cohort must keep first-seen `-dirty`, no `+mixed`",
        );
    }

    /// Homogeneous-clean cohort (every contributor lacks
    /// `-dirty`) keeps the un-suffixed first-seen value, no
    /// marker.
    #[test]
    fn group_and_average_all_clean_keeps_value_no_mixed() {
        let mut a = make_row("t", "tiny-1llc", true, 0.0);
        a.commit = Some("abc1234".to_string());
        let mut b = make_row("t", "tiny-1llc", true, 0.0);
        b.commit = Some("abc1234".to_string());

        let out = group_and_average(&[a, b]);
        assert_eq!(
            out[0].row.commit.as_deref(),
            Some("abc1234"),
            "homogeneous-clean cohort must keep first-seen value, no `+mixed`",
        );
    }

    /// Failing and skipped contributors participate in mixed-
    /// dirty tracking. The cohort's WIP state is metadata
    /// independent of metric outcome — a skipped sidecar from a
    /// dirty tree still counts toward the dirty-flag because it
    /// records the producer's working-tree state at run time.
    /// Pin: one passing-clean + one skipped-dirty contributor
    /// renders `+mixed`.
    #[test]
    fn group_and_average_mixed_dirty_tracking_includes_skipped_and_failed() {
        let mut clean_pass = make_row("t", "tiny-1llc", true, 0.0);
        clean_pass.commit = Some("abc1234".to_string());
        let mut dirty_skip = make_row("t", "tiny-1llc", true, 0.0);
        dirty_skip.skipped = true;
        dirty_skip.commit = Some("abc1234-dirty".to_string());

        let out = group_and_average(&[clean_pass, dirty_skip]);
        assert_eq!(
            out[0].row.commit.as_deref(),
            Some("abc1234+mixed"),
            "skipped contributors still flip the dirty flag — \
             cohort metadata is independent of metric outcome",
        );
    }

    /// Failed contributor pin: a passing-clean row paired with a
    /// FAILING-dirty row (`passed=false`, `skipped=false`) must
    /// still flip the cohort's mixed-dirty flag and render
    /// `+mixed` on the aggregate's commit field. The
    /// `update_dirty_tracking` call site executes BEFORE the
    /// `if !row.passed { continue; }` short-circuit, which is
    /// the load-bearing ordering: dirty-status is per-row
    /// metadata about the producer's working tree, NOT a metric
    /// outcome, so failed contributors must carry their dirty
    /// flag forward even though their metrics are excluded from
    /// the mean. A regression that moved `update_dirty_tracking`
    /// below the failed-skip continue would silently drop the
    /// failed row's dirty status and the cohort would render the
    /// clean form — hiding WIP-vs-committed disagreement that
    /// the operator needs to see.
    ///
    /// Distinct from
    /// `group_and_average_mixed_dirty_tracking_includes_skipped_and_failed`
    /// which exercises the SKIPPED arm only (`passed=true,
    /// skipped=true`). The two arms have separate `continue`
    /// statements and one could regress without the other; this
    /// test pins the FAILED arm specifically.
    #[test]
    fn group_and_average_mixed_dirty_tracking_includes_failed_contributors() {
        let mut clean_pass = make_row("t", "tiny-1llc", true, 0.0);
        clean_pass.commit = Some("abc1234".to_string());
        let mut dirty_fail = make_row("t", "tiny-1llc", false, 0.0);
        dirty_fail.commit = Some("abc1234-dirty".to_string());

        let out = group_and_average(&[clean_pass, dirty_fail]);
        assert_eq!(out.len(), 1, "single cohort key must produce one aggregate");
        assert_eq!(
            out[0].row.commit.as_deref(),
            Some("abc1234+mixed"),
            "failed contributor's `-dirty` flag must still flip the \
             cohort's dirty-tracking — cohort metadata is independent \
             of metric outcome. A regression moving update_dirty_tracking \
             below the `if !row.passed` continue would drop the failed \
             row's dirty status and render `abc1234` instead",
        );
        // Symmetric arm: passing-dirty + failing-clean. The
        // dirty-tracking flip on the failing contributor's clean
        // form must register as well — `any_clean` is the
        // counterpart flag, and the same code path executes for
        // both `Some(hex)` and `Some(hex-dirty)` values.
        let mut dirty_pass = make_row("t", "tiny-1llc", true, 0.0);
        dirty_pass.commit = Some("def5678-dirty".to_string());
        let mut clean_fail = make_row("t", "tiny-1llc", false, 0.0);
        clean_fail.commit = Some("def5678".to_string());

        let out = group_and_average(&[dirty_pass, clean_fail]);
        assert_eq!(
            out[0].row.commit.as_deref(),
            Some("def5678+mixed"),
            "failed contributor's CLEAN form must also flip the \
             cohort's any_clean flag — symmetric to the dirty arm",
        );
        // Failed contributor's `passed=false` still flips the
        // aggregate's `passed` flag (logical-AND across all
        // contributors). This sanity-checks that the new test
        // doesn't accidentally exercise an aggregate-passes path
        // — failed rows are correctly being excluded from the
        // metric mean while contributing to dirty tracking.
        assert!(
            !out[0].row.passed,
            "any failing contributor must flip the aggregate to \
             passed=false, regardless of dirty-tracking semantics",
        );
    }

    /// Mixed-dirty marker uses canonical un-suffixed hex even
    /// when `acc.first` is the dirty form. Pin: first contributor
    /// is `abc1234-dirty`, second is `abc1234`; rendered form is
    /// `abc1234+mixed`, NOT `abc1234-dirty+mixed`. Guards against
    /// a stripping bug in `render_mixed_dirty`.
    #[test]
    fn group_and_average_mixed_dirty_strips_dirty_from_first_seen() {
        let mut dirty_first = make_row("t", "tiny-1llc", true, 0.0);
        dirty_first.commit = Some("abc1234-dirty".to_string());
        let mut clean_second = make_row("t", "tiny-1llc", true, 0.0);
        clean_second.commit = Some("abc1234".to_string());

        let out = group_and_average(&[dirty_first, clean_second]);
        let rendered = out[0].row.commit.as_deref().expect("commit must render");
        assert_eq!(rendered, "abc1234+mixed");
        assert!(
            !rendered.contains("-dirty"),
            "rendered form must drop `-dirty` even when first contributor was dirty; got: {rendered}",
        );
    }

    /// `None`-only cohort keeps `None`. Sanity check that the
    /// dirty-tracking does not synthesize a marker when no
    /// contributor has a commit value.
    #[test]
    fn group_and_average_all_none_commits_keeps_none_no_mixed() {
        let a = make_row("t", "tiny-1llc", true, 0.0);
        let b = make_row("t", "tiny-1llc", true, 0.0);

        let out = group_and_average(&[a, b]);
        assert!(
            out[0].row.commit.is_none(),
            "None-only cohort must keep None — no synthesized `+mixed`",
        );
    }

    /// End-to-end: aggregated rows feed `compare_rows` cleanly.
    /// Side A has [10, 12, 14] (mean 12); side B has [28, 30, 32]
    /// (mean 30). The 18-unit delta on `worst_spread`
    /// (default_abs=5.0, default_rel=0.25) clears both gates,
    /// producing a regression. Pins the full averaging pipeline.
    #[test]
    fn group_and_average_then_compare_rows_yields_regression_on_means() {
        let mut a1 = make_row("t", "tiny-1llc", true, 0.0);
        paint_metrics(&mut a1, 10.0, 100, 30, 1000);
        let mut a2 = make_row("t", "tiny-1llc", true, 0.0);
        paint_metrics(&mut a2, 12.0, 120, 35, 1000);
        let mut a3 = make_row("t", "tiny-1llc", true, 0.0);
        paint_metrics(&mut a3, 14.0, 140, 40, 1000);
        let mut b1 = make_row("t", "tiny-1llc", true, 0.0);
        paint_metrics(&mut b1, 28.0, 280, 70, 1000);
        let mut b2 = make_row("t", "tiny-1llc", true, 0.0);
        paint_metrics(&mut b2, 30.0, 300, 75, 1000);
        let mut b3 = make_row("t", "tiny-1llc", true, 0.0);
        paint_metrics(&mut b3, 32.0, 320, 80, 1000);

        let agg_a = group_and_average(&[a1, a2, a3]);
        let agg_b = group_and_average(&[b1, b2, b3]);
        let rows_a: Vec<GauntletRow> = agg_a.iter().map(|r| r.row.clone()).collect();
        let rows_b: Vec<GauntletRow> = agg_b.iter().map(|r| r.row.clone()).collect();
        let res = compare_rows(&rows_a, &rows_b, None, &ComparisonPolicy::default());
        let spread = res
            .findings
            .iter()
            .find(|f| f.metric.name == "worst_spread")
            .expect("worst_spread must regress on aggregated means");
        assert!(spread.is_regression);
        assert_eq!(spread.val_a, 12.0, "mean of [10, 12, 14] = 12");
        assert_eq!(spread.val_b, 30.0, "mean of [28, 30, 32] = 30");
        assert_eq!(spread.delta, 18.0);
    }

    /// `compare_partitions` with the default (averaging-on)
    /// path must aggregate every matching sidecar within each
    /// side and detect regressions on the aggregated means.
    /// End-to-end pin against on-disk fixtures so a regression
    /// in the aggregation → compare wiring lands here.
    ///
    /// Fixture: two runs each carrying three sidecars that
    /// differ on `scheduler` (the slicing dim). Side A's three
    /// trials cluster around `worst_spread = 10` (mean 12);
    /// side B's three cluster around `worst_spread = 30` (mean
    /// 30). The 18-unit delta clears the default dual gate, so
    /// `compare_partitions` returns exit code 1 (regressions
    /// detected).
    #[test]
    fn compare_partitions_with_average_default_produces_regression_on_aggregated_means() {
        use crate::test_support::SidecarResult;

        let alt_root = tempfile::TempDir::new().expect("create alt-root tempdir");
        let run_a = "__avg_thread_a__";
        let run_b = "__avg_thread_b__";

        // Three trials per side, same (scenario, topology,
        // work_type) so they aggregate into a single key. Vary
        // the per-trial spread so the mean is non-degenerate
        // (regression flags would also fire if the values were
        // identical, but the average path is exercised either way).
        let trials_a = [(10.0, 100), (12.0, 120), (14.0, 140)];
        let trials_b = [(28.0, 280), (30.0, 300), (32.0, 320)];

        // Scheduler is the slicing dim: side A's three trials
        // run under "scx_alpha", side B's under "scx_beta". The
        // pairing dims are everything else (kernel/topology/
        // work_type/commit/flags) which match across both runs,
        // so the three trials on each side aggregate into one
        // mean row keyed by `(scenario, topology, work_type,
        // flags)` plus the matching kernel/commit values.
        for (run_key, trials, sched) in [
            (run_a, &trials_a, "scx_alpha"),
            (run_b, &trials_b, "scx_beta"),
        ] {
            let run_dir = alt_root.path().join(run_key);
            std::fs::create_dir_all(&run_dir).expect("create run dir");
            for (i, (spread, gap_ms)) in trials.iter().enumerate() {
                let trial_name = format!("avg_trial_{run_key}_{i}");
                let mut sidecar = SidecarResult {
                    test_name: "avg_test".to_string(),
                    topology: "1n2l4c1t".to_string(),
                    scheduler: sched.to_string(),
                    work_type: "CpuSpin".to_string(),
                    ..SidecarResult::test_fixture()
                };
                sidecar.stats.worst_spread = *spread;
                sidecar.stats.worst_gap_ms = *gap_ms;
                sidecar.passed = true;
                sidecar.skipped = false;
                let json = serde_json::to_string(&sidecar).expect("serialize fixture sidecar");
                let sidecar_path = run_dir.join(format!("{trial_name}.ktstr.json"));
                std::fs::write(&sidecar_path, json).expect("write fixture sidecar");
            }
        }

        let filter_a = RowFilter {
            schedulers: vec!["scx_alpha".to_string()],
            ..RowFilter::default()
        };
        let filter_b = RowFilter {
            schedulers: vec!["scx_beta".to_string()],
            ..RowFilter::default()
        };

        // Default (averaging-on) path: three sidecars per side
        // share one pairing key, so each side aggregates to a
        // single mean row. The 18-unit worst_spread delta on
        // those means (12 vs 30) clears the default dual gate
        // and surfaces exit code 1.
        let exit = compare_partitions(
            &filter_a,
            &filter_b,
            None,
            &ComparisonPolicy::default(),
            Some(alt_root.path()),
            false, // no_average=false → averaging is ON
        )
        .expect("compare_partitions must succeed against valid fixtures");
        assert_eq!(
            exit, 1,
            "an 18-unit worst_spread regression on the aggregated mean \
             (a=12 → b=30) must clear the default dual gate and surface \
             exit code 1; got {exit}",
        );
    }

    // -- format_average_header / format_per_group_pass_counts --

    /// `format_average_header` renders the exact header line that
    /// `compare_partitions` prints above the comparison table when
    /// `--average` is active. Pins the operator-visible surface
    /// (the "averaged across N runs (A) and M runs (B)" string)
    /// so a regression that reworded the header without
    /// updating downstream parsers / scripts lands here.
    #[test]
    fn format_average_header_exact_string() {
        let out = format_average_header(5, 3, "kernel-6.14", "kernel-6.15");
        assert_eq!(
            out,
            "averaged across 5 runs (kernel-6.14) and 3 runs (kernel-6.15)",
        );
    }

    /// Zero-contributor sides are surfaced verbatim — operator
    /// will see `0 runs` for an empty side. Pins the empty-side
    /// edge case so a regression that special-cased `pre_agg = 0`
    /// (e.g. omitted the side, said "no contributors") would
    /// fail here. The companion empty-rows path is already
    /// guarded upstream by `compare_partitions`' `sidecars_*.is_empty()`
    /// bail; this test guards the formatter itself in case it's
    /// reused outside the compare path.
    #[test]
    fn format_average_header_zero_contributor_sides_render_verbatim() {
        assert_eq!(
            format_average_header(0, 0, "a", "b"),
            "averaged across 0 runs (a) and 0 runs (b)",
        );
    }

    /// Helper for the per-group-block tests: build an
    /// `AveragedGroup` with the named identity and pass counters
    /// while leaving every metric field at zero. Metrics aren't
    /// observed by [`format_per_group_pass_counts`] — only the
    /// identity tuple and pass counters drive the output.
    fn group(
        scenario: &str,
        topology: &str,
        work_type: &str,
        flags: &[&str],
        passes_observed: u32,
        total_observed: u32,
    ) -> AveragedGroup {
        let mut row = make_row(scenario, topology, true, 0.0);
        row.work_type = work_type.into();
        row.flags = flags.iter().map(|s| (*s).to_string()).collect();
        AveragedGroup {
            row,
            passes_observed,
            total_observed,
        }
    }

    /// Empty input: no groups on either side. The formatter
    /// returns an empty string so the caller can suppress the
    /// block entirely (no header, no body, no separator).
    #[test]
    fn format_per_group_pass_counts_empty_returns_empty_string() {
        let out = format_per_group_pass_counts(&[], &[], "a", "b");
        assert!(
            out.is_empty(),
            "empty input must yield empty output, got: {out:?}",
        );
    }

    /// Both-sides-present: every (scenario, topology, work_type,
    /// flags) group renders one line. Healthy 5/5 groups appear
    /// alongside unhealthy 3/5 groups — the spec is "show every
    /// group", not "show only the broken ones".
    #[test]
    fn format_per_group_pass_counts_renders_every_group_with_n_over_m() {
        let avg_a = vec![
            group("alpha", "tiny-1llc", "CpuSpin", &[], 5, 5),
            group("beta", "tiny-1llc", "CpuSpin", &[], 3, 5),
        ];
        let avg_b = vec![
            group("alpha", "tiny-1llc", "CpuSpin", &[], 4, 5),
            group("beta", "tiny-1llc", "CpuSpin", &[], 5, 5),
        ];
        let out = format_per_group_pass_counts(&avg_a, &avg_b, "a", "b");
        // Header line present.
        assert!(
            out.contains("per-group pass counts"),
            "header line must appear, got: {out:?}",
        );
        // Both groups render with their per-side N/M counters.
        assert!(
            out.contains("alpha/tiny-1llc/CpuSpin: a=5/5 b=4/5"),
            "alpha group line missing; got: {out:?}",
        );
        assert!(
            out.contains("beta/tiny-1llc/CpuSpin: a=3/5 b=5/5"),
            "beta group line missing; got: {out:?}",
        );
        // Trailing newline so the next section reads cleanly.
        assert!(
            out.ends_with('\n'),
            "block must end with newline, got: {out:?}",
        );
    }

    /// One-side-only group renders `-` for the missing side.
    /// Pins the asymmetric-key path: a B-side row that has no
    /// A-side match gets `a=-`; symmetric for A-only / B-side.
    /// The block surfaces the asymmetry by name so the operator
    /// doesn't have to cross-reference the summary's `new_in_b`
    /// / `removed_from_a` counters to know which groups went
    /// missing.
    #[test]
    fn format_per_group_pass_counts_one_side_missing_renders_dash() {
        let avg_a = vec![group("only_a", "tiny-1llc", "CpuSpin", &[], 5, 5)];
        let avg_b = vec![group("only_b", "tiny-1llc", "CpuSpin", &[], 3, 5)];
        let out = format_per_group_pass_counts(&avg_a, &avg_b, "a", "b");
        assert!(
            out.contains("only_a/tiny-1llc/CpuSpin: a=5/5 b=-"),
            "A-only group must render b=-; got: {out:?}",
        );
        assert!(
            out.contains("only_b/tiny-1llc/CpuSpin: a=- b=3/5"),
            "B-only group must render a=-; got: {out:?}",
        );
    }

    /// Different `flags` profiles for the same (scenario,
    /// topology, work_type) tuple render as separate lines. The
    /// flag tuple is part of the join key; treating two flag
    /// profiles as the same group would silently merge their
    /// pass counts and hide flag-specific failures.
    #[test]
    fn format_per_group_pass_counts_distinct_flags_render_separately() {
        let avg_a = vec![
            group("t", "tiny-1llc", "CpuSpin", &["llc"], 5, 5),
            group("t", "tiny-1llc", "CpuSpin", &["borrow"], 4, 5),
        ];
        let avg_b = vec![
            group("t", "tiny-1llc", "CpuSpin", &["llc"], 3, 5),
            group("t", "tiny-1llc", "CpuSpin", &["borrow"], 5, 5),
        ];
        let out = format_per_group_pass_counts(&avg_a, &avg_b, "a", "b");
        // Both lines must appear separately — not collapsed.
        // Count occurrences of "t/tiny-1llc/CpuSpin" — there
        // should be exactly 2 (one per flag profile).
        let occurrences = out.matches("t/tiny-1llc/CpuSpin").count();
        assert_eq!(
            occurrences, 2,
            "two flag profiles must render as two separate lines; got: {out:?}",
        );
    }

    // -- Dimension / derive_slicing_dims / pairing dims --

    /// `Dimension::ALL` lists all eight dims in canonical order.
    /// Order matters for [`PairingKey::from_row`] and for header
    /// rendering — a regression that reordered the slice would
    /// silently shift every dynamic key, splitting previously-
    /// paired rows. Pin the literal order.
    #[test]
    fn dimension_all_canonical_order() {
        assert_eq!(
            Dimension::ALL,
            &[
                Dimension::Kernel,
                Dimension::Scheduler,
                Dimension::Topology,
                Dimension::WorkType,
                Dimension::Commit,
                Dimension::KernelCommit,
                Dimension::Source,
                Dimension::Flags,
            ],
        );
    }

    /// `Dimension::pairing_dims` returns every dim NOT in the
    /// slicing set, preserving canonical order. Two slicing
    /// orderings produce the same pairing-dim list (the function
    /// iterates `ALL`, not `slicing`).
    #[test]
    fn dimension_pairing_dims_complements_slicing() {
        let pair = Dimension::pairing_dims(&[Dimension::Kernel, Dimension::Commit]);
        assert_eq!(
            pair,
            vec![
                Dimension::Scheduler,
                Dimension::Topology,
                Dimension::WorkType,
                Dimension::KernelCommit,
                Dimension::Source,
                Dimension::Flags,
            ],
        );
        // Order of slicing input doesn't change the output —
        // the function iterates ALL and filters.
        let pair_reversed = Dimension::pairing_dims(&[Dimension::Commit, Dimension::Kernel]);
        assert_eq!(pair, pair_reversed);
    }

    /// Empty slicing set → every dim is a pairing dim.
    #[test]
    fn dimension_pairing_dims_empty_slicing_yields_all() {
        let pair = Dimension::pairing_dims(&[]);
        assert_eq!(pair, Dimension::ALL.to_vec());
    }

    /// `derive_slicing_dims` returns every dimension on which
    /// filter_a and filter_b differ. Equal filters → empty
    /// slicing.
    #[test]
    fn derive_slicing_dims_identical_filters_yields_empty() {
        let f = RowFilter {
            schedulers: vec!["scx_alpha".to_string()],
            ..RowFilter::default()
        };
        assert!(derive_slicing_dims(&f, &f).is_empty());
    }

    /// One-dim diff: only the differing dimension is reported.
    #[test]
    fn derive_slicing_dims_single_dim_diff() {
        let f_a = RowFilter {
            schedulers: vec!["scx_alpha".to_string()],
            ..RowFilter::default()
        };
        let f_b = RowFilter {
            schedulers: vec!["scx_beta".to_string()],
            ..RowFilter::default()
        };
        assert_eq!(derive_slicing_dims(&f_a, &f_b), vec![Dimension::Scheduler]);
    }

    /// Vec dims (kernels/commits/flags) compare as sorted-deduped
    /// sets — order and duplicates inside the filter don't shift
    /// the slicing-dim derivation.
    #[test]
    fn derive_slicing_dims_vec_compares_as_set() {
        let f_a = RowFilter {
            kernels: vec!["6.14".to_string(), "6.15".to_string()],
            ..RowFilter::default()
        };
        let f_b = RowFilter {
            kernels: vec!["6.15".to_string(), "6.14".to_string(), "6.14".to_string()],
            ..RowFilter::default()
        };
        assert!(
            derive_slicing_dims(&f_a, &f_b).is_empty(),
            "same set in different order/multiplicity must NOT slice",
        );
    }

    /// Multi-dim diff: every differing dimension is reported, in
    /// canonical [`Dimension::ALL`] order.
    #[test]
    fn derive_slicing_dims_multi_dim_diff_in_canonical_order() {
        let f_a = RowFilter {
            kernels: vec!["6.14".to_string()],
            schedulers: vec!["scx_alpha".to_string()],
            ..RowFilter::default()
        };
        let f_b = RowFilter {
            kernels: vec!["6.15".to_string()],
            schedulers: vec!["scx_beta".to_string()],
            ..RowFilter::default()
        };
        assert_eq!(
            derive_slicing_dims(&f_a, &f_b),
            vec![Dimension::Kernel, Dimension::Scheduler],
        );
    }

    /// Source-only diff: filters that disagree on `run_sources`
    /// and agree on every other dimension produce a slicing-dim
    /// set containing exactly `Dimension::Source`. Pins the
    /// Source arm of the per-dimension comparison switch in
    /// [`derive_slicing_dims`] — a regression that omitted the
    /// arm or compared the wrong field would surface here as an
    /// empty slicing-dim set (and downstream as a `compare`
    /// command that mistakenly bails with "A and B select
    /// identical rows" on a legitimate source contrast).
    #[test]
    fn derive_slicing_dims_source_only_diff() {
        let f_a = RowFilter {
            run_sources: vec!["local".to_string()],
            ..RowFilter::default()
        };
        let f_b = RowFilter {
            run_sources: vec!["ci".to_string()],
            ..RowFilter::default()
        };
        assert_eq!(
            derive_slicing_dims(&f_a, &f_b),
            vec![Dimension::Source],
            "differing `run_sources` must surface Source as a slicing dim",
        );

        // Sorted-deduped Vec semantics also apply on the Source
        // dim — same set in different order/multiplicity must NOT
        // slice. Mirrors the `derive_slicing_dims_vec_compares_as_set`
        // contract for the Source arm.
        let f_c = RowFilter {
            run_sources: vec!["local".to_string(), "ci".to_string()],
            ..RowFilter::default()
        };
        let f_d = RowFilter {
            run_sources: vec!["ci".to_string(), "local".to_string(), "local".to_string()],
            ..RowFilter::default()
        };
        assert!(
            derive_slicing_dims(&f_c, &f_d).is_empty(),
            "same run_source set in different order/multiplicity must NOT slice",
        );
    }

    /// Topology-only diff: filters that disagree on `topologies`
    /// and agree on every other dimension produce a slicing-dim
    /// set containing exactly `Dimension::Topology`. Pins the
    /// Topology arm of the per-dimension comparison switch in
    /// [`derive_slicing_dims`] for the post-Vec-promotion
    /// `topologies` field; before promotion `--topology` was a
    /// single-value `Option<String>` and the per-arm comparison
    /// shape was `Option<String> != Option<String>`. Mirror of
    /// `derive_slicing_dims_source_only_diff` for the Topology
    /// arm.
    #[test]
    fn derive_slicing_dims_topology_only_diff() {
        let f_a = RowFilter {
            topologies: vec!["1n2l4c1t".to_string()],
            ..RowFilter::default()
        };
        let f_b = RowFilter {
            topologies: vec!["1n2l4c2t".to_string()],
            ..RowFilter::default()
        };
        assert_eq!(
            derive_slicing_dims(&f_a, &f_b),
            vec![Dimension::Topology],
            "differing `topologies` must surface Topology as a slicing dim",
        );

        // Sorted-deduped Vec semantics: same set in different
        // order/multiplicity must NOT slice.
        let f_c = RowFilter {
            topologies: vec!["1n2l4c1t".to_string(), "1n2l4c2t".to_string()],
            ..RowFilter::default()
        };
        let f_d = RowFilter {
            topologies: vec![
                "1n2l4c2t".to_string(),
                "1n2l4c1t".to_string(),
                "1n2l4c1t".to_string(),
            ],
            ..RowFilter::default()
        };
        assert!(
            derive_slicing_dims(&f_c, &f_d).is_empty(),
            "same topology set in different order/multiplicity must NOT slice",
        );
    }

    /// WorkType-only diff: filters that disagree on `work_types`
    /// and agree on every other dimension produce a slicing-dim
    /// set containing exactly `Dimension::WorkType`. Mirror of
    /// `derive_slicing_dims_topology_only_diff` for the WorkType
    /// arm.
    #[test]
    fn derive_slicing_dims_work_type_only_diff() {
        let f_a = RowFilter {
            work_types: vec!["CpuSpin".to_string()],
            ..RowFilter::default()
        };
        let f_b = RowFilter {
            work_types: vec!["PageFaultChurn".to_string()],
            ..RowFilter::default()
        };
        assert_eq!(
            derive_slicing_dims(&f_a, &f_b),
            vec![Dimension::WorkType],
            "differing `work_types` must surface WorkType as a slicing dim",
        );

        // Sorted-deduped Vec semantics: same set in different
        // order/multiplicity must NOT slice.
        let f_c = RowFilter {
            work_types: vec!["CpuSpin".to_string(), "PageFaultChurn".to_string()],
            ..RowFilter::default()
        };
        let f_d = RowFilter {
            work_types: vec![
                "PageFaultChurn".to_string(),
                "CpuSpin".to_string(),
                "CpuSpin".to_string(),
            ],
            ..RowFilter::default()
        };
        assert!(
            derive_slicing_dims(&f_c, &f_d).is_empty(),
            "same work_type set in different order/multiplicity must NOT slice",
        );
    }

    /// `kernel_filter_matches`: major.minor (`6.12`) prefix
    /// matches every patch in the series via the
    /// `starts_with("6.12.")` arm, and ALSO matches `6.12`
    /// exactly. Three-segment-or-longer filters are strict.
    #[test]
    fn kernel_filter_matches_major_minor_prefix() {
        // Two-segment filter: prefix matches.
        assert!(kernel_filter_matches("6.12", "6.12"));
        assert!(kernel_filter_matches("6.12", "6.12.0"));
        assert!(kernel_filter_matches("6.12", "6.12.5"));
        assert!(!kernel_filter_matches("6.12", "6.13.0"));
        // Critically: `6.1` must not match `6.10.0` — the
        // trailing-dot in the prefix path prevents the
        // accidental wildcard.
        assert!(!kernel_filter_matches("6.1", "6.10.0"));
    }

    /// `kernel_filter_matches`: three-segment+ filters are strict
    /// equality.
    #[test]
    fn kernel_filter_matches_strict_for_three_plus_segments() {
        assert!(kernel_filter_matches("6.14.2", "6.14.2"));
        // Critically: `6.14.2` must NOT match `6.14.20` — the
        // strict-equality arm prevents the patch-level prefix
        // wildcarding.
        assert!(!kernel_filter_matches("6.14.2", "6.14.20"));
        assert!(!kernel_filter_matches("6.14.2", "6.14.21"));
        // RC suffixes are also strict.
        assert!(kernel_filter_matches("6.15-rc3", "6.15-rc3"));
        assert!(!kernel_filter_matches("6.15-rc3", "6.15-rc30"));
    }

    /// `RowFilter::matches` with a major.minor `--kernel` filter
    /// admits the row whose `kernel_version` is a patch in that
    /// series.
    #[test]
    fn row_filter_kernel_major_minor_prefix_admits_patch_version() {
        let row = make_filter_row("t", "scx_a", "1n2l4c1t", "CpuSpin", Some("6.12.5"), &[]);
        let filter = RowFilter {
            kernels: vec!["6.12".to_string()],
            ..RowFilter::default()
        };
        assert!(
            filter.matches(&row),
            "major.minor filter `6.12` must admit row with kernel_version `6.12.5`",
        );
    }

    // -- PairingKey --

    /// `PairingKey::from_row` always puts `scenario` first, then
    /// the requested dims in canonical order. Two rows with the
    /// same scenario+dims agree; one with a different topology
    /// (when topology IS a pairing dim) does not.
    #[test]
    fn pairing_key_from_row_basic() {
        let row_a = make_filter_row("scenA", "scx_a", "1n1l", "CpuSpin", Some("6.14"), &[]);
        let row_b = make_filter_row("scenA", "scx_a", "1n1l", "CpuSpin", Some("6.14"), &[]);
        let row_c = make_filter_row("scenA", "scx_a", "2n2l", "CpuSpin", Some("6.14"), &[]);
        let dims = &[Dimension::Topology, Dimension::WorkType];
        assert_eq!(
            PairingKey::from_row(&row_a, dims),
            PairingKey::from_row(&row_b, dims),
        );
        assert_ne!(
            PairingKey::from_row(&row_a, dims),
            PairingKey::from_row(&row_c, dims),
            "different topology must distinguish the keys when topology is a pairing dim",
        );
    }

    /// Slicing on topology means topology is NOT in the pairing
    /// dim set — so two rows that differ ONLY on topology pair
    /// to the same key, allowing the comparison to contrast
    /// them across A/B sides.
    #[test]
    fn pairing_key_excludes_slicing_dim() {
        let row_a = make_filter_row("scenA", "scx_a", "1n1l", "CpuSpin", Some("6.14"), &[]);
        let row_b = make_filter_row("scenA", "scx_a", "2n2l", "CpuSpin", Some("6.14"), &[]);
        // Pairing dims = ALL minus Topology. So these two rows
        // pair iff they agree on everything BUT topology.
        let pair_dims = Dimension::pairing_dims(&[Dimension::Topology]);
        assert_eq!(
            PairingKey::from_row(&row_a, &pair_dims),
            PairingKey::from_row(&row_b, &pair_dims),
            "rows differing only on a slicing dim must produce equal pairing keys",
        );
    }

    /// `PairingKey::from_row` first slot is always scenario;
    /// rendering via `parts.join("/")` reproduces the historical
    /// `scenario/topology/work_type/flags` shape when those dims
    /// are pairing dims.
    #[test]
    fn pairing_key_join_renders_legacy_shape() {
        let mut row = make_filter_row("test_a", "scx_a", "1n2l", "CpuSpin", Some("6.14"), &[]);
        row.flags = vec!["llc".to_string(), "steal".to_string()];
        let key = PairingKey::from_row(&row, LEGACY_PAIRING_DIMS);
        assert_eq!(
            key.0.join("/"),
            "test_a/1n2l/CpuSpin/llc|steal",
            "legacy-shape join must render the four-segment label \
             with flags sorted+deduped via `|`",
        );
    }

    /// `PairingKey::from_row` includes the row's `kernel_commit`
    /// when `KernelCommit` is in the pairing-dim list, and
    /// excludes it when `KernelCommit` is the slicing dim. Pins
    /// the [`Dimension::KernelCommit`] arm of the from_row match
    /// — a regression that omitted the arm or substituted the
    /// wrong row field would surface here as either a missing
    /// key slot or a slot carrying the wrong value.
    ///
    /// `None` kernel_commit renders as the empty string slot per
    /// the `unwrap_or_default()` policy on Option dims; that
    /// shape is shared across every Option-typed dim arm.
    #[test]
    fn pairing_key_from_row_includes_kernel_commit_when_pairing() {
        let mut row_some = make_filter_row("scn", "scx_a", "1n1l", "CpuSpin", Some("6.14"), &[]);
        row_some.kernel_commit = Some("kabcde7".to_string());
        let mut row_none = make_filter_row("scn", "scx_a", "1n1l", "CpuSpin", Some("6.14"), &[]);
        row_none.kernel_commit = None;

        // KernelCommit in pairing dims → key carries the commit
        // value (or the empty slot for None). The two rows
        // therefore produce DIFFERENT keys because their
        // kernel_commit values disagree.
        let pair_dims = &[Dimension::KernelCommit];
        let key_some = PairingKey::from_row(&row_some, pair_dims);
        let key_none = PairingKey::from_row(&row_none, pair_dims);
        assert_eq!(
            key_some.0,
            vec!["scn".to_string(), "kabcde7".to_string()],
            "Some(kernel_commit) must occupy the second slot verbatim",
        );
        assert_eq!(
            key_none.0,
            vec!["scn".to_string(), String::new()],
            "None kernel_commit must collapse to an empty slot per \
             unwrap_or_default policy",
        );
        assert_ne!(
            key_some, key_none,
            "two rows differing on kernel_commit must produce \
             distinct pairing keys when KernelCommit is a pairing dim",
        );

        // KernelCommit excluded (slicing) → the two rows pair to
        // the same key because the dim is dropped. Pins the
        // dimensional-slicing semantic for the new arm.
        let slice_dims = Dimension::pairing_dims(&[Dimension::KernelCommit]);
        assert_eq!(
            PairingKey::from_row(&row_some, &slice_dims),
            PairingKey::from_row(&row_none, &slice_dims),
            "rows differing only on the slicing dim (KernelCommit) \
             must produce equal pairing keys",
        );
    }

    /// `PairingKey::from_row` includes the row's `run_source`
    /// when `Source` is in the pairing-dim list, and excludes it
    /// when `Source` is the slicing dim. Pins the
    /// [`Dimension::Source`] arm of the from_row match — same
    /// shape and motivation as
    /// `pairing_key_from_row_includes_kernel_commit_when_pairing`
    /// but for the run_source arm. A regression that omitted the
    /// arm or substituted `row.kernel_commit` for
    /// `row.run_source` would surface here.
    #[test]
    fn pairing_key_from_row_includes_run_source_when_pairing() {
        let mut row_local = make_filter_row("scn", "scx_a", "1n1l", "CpuSpin", Some("6.14"), &[]);
        row_local.run_source = Some("local".to_string());
        let mut row_ci = make_filter_row("scn", "scx_a", "1n1l", "CpuSpin", Some("6.14"), &[]);
        row_ci.run_source = Some("ci".to_string());
        let mut row_none = make_filter_row("scn", "scx_a", "1n1l", "CpuSpin", Some("6.14"), &[]);
        row_none.run_source = None;

        let pair_dims = &[Dimension::Source];
        let key_local = PairingKey::from_row(&row_local, pair_dims);
        let key_ci = PairingKey::from_row(&row_ci, pair_dims);
        let key_none = PairingKey::from_row(&row_none, pair_dims);
        assert_eq!(
            key_local.0,
            vec!["scn".to_string(), "local".to_string()],
            "Some(run_source) must occupy the second slot verbatim",
        );
        assert_eq!(key_ci.0, vec!["scn".to_string(), "ci".to_string()]);
        assert_eq!(
            key_none.0,
            vec!["scn".to_string(), String::new()],
            "None run_source must collapse to an empty slot per \
             unwrap_or_default policy",
        );
        assert_ne!(
            key_local, key_ci,
            "two rows differing on run_source must produce \
             distinct pairing keys when Source is a pairing dim",
        );

        // Source excluded (slicing) → the differing-run_source
        // rows pair to the same key.
        let slice_dims = Dimension::pairing_dims(&[Dimension::Source]);
        assert_eq!(
            PairingKey::from_row(&row_local, &slice_dims),
            PairingKey::from_row(&row_ci, &slice_dims),
            "rows differing only on the slicing dim (Source) must \
             produce equal pairing keys",
        );
    }

    /// Clean and dirty contributors at the same canonical hex
    /// must land in the same pairing bucket. Without the
    /// `-dirty` strip in `commit_pairing_key_part`, `abc1234`
    /// and `abc1234-dirty` shatter into separate groups,
    /// defeating `group_and_average_by`'s `+mixed` cohort
    /// detection (which can only fire when the two contributors
    /// land in ONE group).
    #[test]
    fn pairing_key_from_row_strips_dirty_suffix_on_commit() {
        let mut row_clean = make_filter_row("scn", "scx_a", "1n1l", "CpuSpin", Some("6.14"), &[]);
        row_clean.commit = Some("abc1234".to_string());
        let mut row_dirty = make_filter_row("scn", "scx_a", "1n1l", "CpuSpin", Some("6.14"), &[]);
        row_dirty.commit = Some("abc1234-dirty".to_string());

        let pair_dims = &[Dimension::Commit];
        let key_clean = PairingKey::from_row(&row_clean, pair_dims);
        let key_dirty = PairingKey::from_row(&row_dirty, pair_dims);

        assert_eq!(
            key_clean, key_dirty,
            "clean `abc1234` and dirty `abc1234-dirty` must produce \
             EQUAL pairing keys so the +mixed cohort machinery in \
             group_and_average_by can surface their disagreement",
        );
        assert_eq!(
            key_clean.0,
            vec!["scn".to_string(), "abc1234".to_string()],
            "key part must be the canonical un-suffixed hex",
        );
    }

    /// Same shape on the kernel_commit dimension. Pins the
    /// second commit dim's strip independently because
    /// `from_row` uses two parallel arms; a regression could
    /// strip one but not the other.
    #[test]
    fn pairing_key_from_row_strips_dirty_suffix_on_kernel_commit() {
        let mut row_clean = make_filter_row("scn", "scx_a", "1n1l", "CpuSpin", Some("6.14"), &[]);
        row_clean.kernel_commit = Some("def5678".to_string());
        let mut row_dirty = make_filter_row("scn", "scx_a", "1n1l", "CpuSpin", Some("6.14"), &[]);
        row_dirty.kernel_commit = Some("def5678-dirty".to_string());

        let pair_dims = &[Dimension::KernelCommit];
        let key_clean = PairingKey::from_row(&row_clean, pair_dims);
        let key_dirty = PairingKey::from_row(&row_dirty, pair_dims);

        assert_eq!(
            key_clean, key_dirty,
            "clean and dirty kernel_commit at the same canonical \
             hex must pair together",
        );
        assert_eq!(key_clean.0, vec!["scn".to_string(), "def5678".to_string()],);
    }

    /// Distinct hexes still differentiate even when one carries
    /// `-dirty`. Pins that the strip operates ONLY on the
    /// suffix, not on the entire value — `aaa1111-dirty` and
    /// `bbb2222` remain distinct.
    #[test]
    fn pairing_key_from_row_distinct_hexes_remain_distinct_under_strip() {
        let mut row_a = make_filter_row("scn", "scx_a", "1n1l", "CpuSpin", Some("6.14"), &[]);
        row_a.commit = Some("aaa1111-dirty".to_string());
        let mut row_b = make_filter_row("scn", "scx_a", "1n1l", "CpuSpin", Some("6.14"), &[]);
        row_b.commit = Some("bbb2222".to_string());

        let pair_dims = &[Dimension::Commit];
        let key_a = PairingKey::from_row(&row_a, pair_dims);
        let key_b = PairingKey::from_row(&row_b, pair_dims);

        assert_ne!(
            key_a, key_b,
            "distinct canonical hexes must remain distinct after the \
             -dirty strip — only the suffix is stripped",
        );
        assert_eq!(key_a.0[1], "aaa1111");
        assert_eq!(key_b.0[1], "bbb2222");
    }

    /// `None` commit values still collapse to the empty slot
    /// (the strip is a no-op on `None`). Pins the absence path
    /// against a regression that special-cased the strip and
    /// inadvertently changed the unwrap_or_default behavior.
    #[test]
    fn pairing_key_from_row_none_commit_unchanged_under_strip() {
        let mut row = make_filter_row("scn", "scx_a", "1n1l", "CpuSpin", Some("6.14"), &[]);
        row.commit = None;
        row.kernel_commit = None;
        let pair_dims = &[Dimension::Commit, Dimension::KernelCommit];
        let key = PairingKey::from_row(&row, pair_dims);
        assert_eq!(
            key.0,
            vec!["scn".to_string(), String::new(), String::new()],
            "None commit and None kernel_commit must collapse to empty slots",
        );
    }

    // -- render_side_label --

    /// Empty slicing dims → the bare label is returned.
    #[test]
    fn render_side_label_empty_dims_yields_bare() {
        let f = RowFilter::default();
        assert_eq!(render_side_label(&f, &[], "A"), "A");
    }

    /// Single-dim single-value scheduler renders the value
    /// verbatim. After the Vec promotion of `--scheduler` the
    /// scheduler arm goes through `render_vec_dim` like every
    /// other Vec dim; a single entry still surfaces the bare
    /// string.
    #[test]
    fn render_side_label_single_value_dim() {
        let f = RowFilter {
            schedulers: vec!["scx_rusty".to_string()],
            ..RowFilter::default()
        };
        assert_eq!(
            render_side_label(&f, &[Dimension::Scheduler], "A"),
            "scx_rusty",
        );
    }

    /// Vec dim with ≤3 entries joins with `|` (sorted).
    #[test]
    fn render_side_label_vec_dim_short_joins_with_pipe() {
        let f = RowFilter {
            kernels: vec!["6.15".to_string(), "6.14".to_string()],
            ..RowFilter::default()
        };
        assert_eq!(
            render_side_label(&f, &[Dimension::Kernel], "A"),
            "6.14|6.15",
            "≤3 values must join sorted with `|`",
        );
    }

    /// Vec dim with >3 entries collapses to the bare label.
    #[test]
    fn render_side_label_vec_dim_long_collapses_to_bare() {
        let f = RowFilter {
            kernels: vec![
                "6.10".to_string(),
                "6.11".to_string(),
                "6.12".to_string(),
                "6.13".to_string(),
                "6.14".to_string(),
            ],
            ..RowFilter::default()
        };
        assert_eq!(
            render_side_label(&f, &[Dimension::Kernel], "A"),
            "A",
            ">3 values must collapse to the bare letter so the \
             column header stays readable",
        );
    }

    /// Multi-dim slicing joins per-dim parts with `:`.
    #[test]
    fn render_side_label_multi_dim_joins_with_colon() {
        let f = RowFilter {
            kernels: vec!["6.14".to_string()],
            schedulers: vec!["scx_rusty".to_string()],
            ..RowFilter::default()
        };
        assert_eq!(
            render_side_label(&f, &[Dimension::Kernel, Dimension::Scheduler], "A"),
            "6.14:scx_rusty",
        );
    }

    /// Empty per-side filter on a slicing dim falls back to the
    /// bare label (the slice exists because the OTHER side
    /// populated the dim).
    #[test]
    fn render_side_label_empty_dim_value_uses_bare() {
        let f = RowFilter::default();
        assert_eq!(
            render_side_label(&f, &[Dimension::Kernel], "B"),
            "B",
            "empty Vec dim must fall back to the bare letter",
        );
        assert_eq!(
            render_side_label(&f, &[Dimension::Scheduler], "B"),
            "B",
            "None Option dim must fall back to the bare letter",
        );
    }

    /// `Dimension::KernelCommit` arm of [`render_side_label`] reads
    /// `filter.kernel_commits` (a Vec) and routes through the same
    /// `render_vec_dim` path as `Kernel` / `Commit` / `Flags`. Pins
    /// the arm so a regression that omitted it (or substituted the
    /// wrong field, e.g. `filter.project_commits`) surfaces here
    /// instead of silently rendering the bare label even when the
    /// filter is populated.
    ///
    /// Single-value: emits the value verbatim. Two-value: joins
    /// sorted with `|` per `render_vec_dim`'s ≤3 rule. >3 values:
    /// collapse to bare. Empty Vec: bare. Same shape as the
    /// `Kernel` arm pinned above; a regression in the
    /// `KernelCommit` arm specifically would NOT be caught by the
    /// existing `render_side_label_vec_dim_*` tests because those
    /// only exercise the `Kernel` field.
    #[test]
    fn render_side_label_kernel_commit_arm_renders_filter_value() {
        let f_one = RowFilter {
            kernel_commits: vec!["kabcde7".to_string()],
            ..RowFilter::default()
        };
        assert_eq!(
            render_side_label(&f_one, &[Dimension::KernelCommit], "A"),
            "kabcde7",
            "single kernel_commit value must render verbatim — \
             a regression that read `filter.project_commits` instead of \
             `filter.kernel_commits` would render `A` here because \
             the project-commit field is empty",
        );

        let f_two = RowFilter {
            kernel_commits: vec!["kbbb222".to_string(), "kaaa111".to_string()],
            ..RowFilter::default()
        };
        assert_eq!(
            render_side_label(&f_two, &[Dimension::KernelCommit], "A"),
            "kaaa111|kbbb222",
            "≤3 kernel_commit values must join sorted with `|`",
        );

        let f_long = RowFilter {
            kernel_commits: vec![
                "k111".to_string(),
                "k222".to_string(),
                "k333".to_string(),
                "k444".to_string(),
            ],
            ..RowFilter::default()
        };
        assert_eq!(
            render_side_label(&f_long, &[Dimension::KernelCommit], "A"),
            "A",
            ">3 kernel_commit values must collapse to the bare letter",
        );

        let f_empty = RowFilter::default();
        assert_eq!(
            render_side_label(&f_empty, &[Dimension::KernelCommit], "B"),
            "B",
            "empty kernel_commits Vec must fall back to the bare letter",
        );
    }

    /// `Dimension::Source` arm of [`render_side_label`] reads
    /// `filter.run_sources` (a Vec) and routes through the same
    /// `render_vec_dim` path as the other Vec dims. Mirror of
    /// `render_side_label_kernel_commit_arm_renders_filter_value`
    /// for the Source arm. A regression that omitted the Source
    /// arm or substituted the wrong field would surface here
    /// instead of silently rendering the bare label even when
    /// the filter is populated.
    #[test]
    fn render_side_label_source_arm_renders_filter_value() {
        let f_one = RowFilter {
            run_sources: vec!["local".to_string()],
            ..RowFilter::default()
        };
        assert_eq!(
            render_side_label(&f_one, &[Dimension::Source], "A"),
            "local",
            "single run_source value must render verbatim — a \
             regression that read another field would render `A` here",
        );

        let f_two = RowFilter {
            run_sources: vec!["local".to_string(), "ci".to_string()],
            ..RowFilter::default()
        };
        assert_eq!(
            render_side_label(&f_two, &[Dimension::Source], "A"),
            "ci|local",
            "≤3 run_source values must join sorted with `|`",
        );

        let f_long = RowFilter {
            run_sources: vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ],
            ..RowFilter::default()
        };
        assert_eq!(
            render_side_label(&f_long, &[Dimension::Source], "A"),
            "A",
            ">3 run_source values must collapse to the bare letter",
        );

        let f_empty = RowFilter::default();
        assert_eq!(
            render_side_label(&f_empty, &[Dimension::Source], "B"),
            "B",
            "empty run_sources Vec must fall back to the bare letter",
        );
    }

    /// `zero_match_diagnostic` flags a `--run-source` value that is
    /// not present in the pool, naming the unknown value AND the
    /// distinct values actually seen. Guards against the
    /// typo-class miss (e.g. `--run-source loca` for `local`,
    /// `--run-source CI` for `ci`) that produces a silent
    /// zero-match in `compare_partitions`.
    #[test]
    fn zero_match_diagnostic_unknown_run_source_lists_present_values() {
        let mut row_local = make_row("scn", "1n1l1c1t", true, 1.0);
        row_local.run_source = Some("local".to_string());
        let mut row_ci = make_row("scn", "1n1l1c1t", true, 1.0);
        row_ci.run_source = Some("ci".to_string());
        let rows = vec![row_local, row_ci];
        let filter = RowFilter {
            run_sources: vec!["loca".to_string()],
            ..Default::default()
        };

        let msg = zero_match_diagnostic("A", &filter, &rows, rows.len());

        assert!(
            msg.contains("--run-source `loca` not found"),
            "must name the unknown value verbatim; got:\n{msg}",
        );
        assert!(
            msg.contains("`ci`") && msg.contains("`local`"),
            "must list distinct values present in the pool so the \
             operator can correct the typo; got:\n{msg}",
        );
        assert!(
            msg.contains("case-sensitive"),
            "must mention case sensitivity (`ci` ≠ `CI`); got:\n{msg}",
        );
    }

    /// When every row has `run_source: None`, the hint surfaces the
    /// "(none — every row has `run_source: null`)" form rather than
    /// an empty list. This is the post-`apply_archive_source_override`
    /// path with a pool that pre-dates the run_source field, so
    /// distinguishing "unknown value, no values present" from
    /// "unknown value, here's what's there" is operator-actionable.
    #[test]
    fn zero_match_diagnostic_unknown_run_source_with_empty_pool_explains_absence() {
        let row = make_row("scn", "1n1l1c1t", true, 1.0);
        let rows = vec![row];
        let filter = RowFilter {
            run_sources: vec!["ci".to_string()],
            ..Default::default()
        };

        let msg = zero_match_diagnostic("A", &filter, &rows, rows.len());

        assert!(
            msg.contains("--run-source `ci` not found"),
            "must name the unknown value; got:\n{msg}",
        );
        assert!(
            msg.contains("none — every row has `run_source: null`"),
            "must explain the empty-distinct-values case rather than \
             listing nothing; got:\n{msg}",
        );
    }

    /// A `--run-source` value that DOES match a row in the pool
    /// must NOT trigger the unknown-value hint, even when the
    /// filter still matches zero rows due to other dimension
    /// mismatches (e.g. scenario filter zeroes the set first).
    /// Pinning this guards against a regression where the hint
    /// fires for every populated `--run-source` regardless of
    /// pool membership.
    #[test]
    fn zero_match_diagnostic_known_run_source_does_not_fire_unknown_hint() {
        let mut row = make_row("scn", "1n1l1c1t", true, 1.0);
        row.run_source = Some("local".to_string());
        let rows = vec![row];
        let filter = RowFilter {
            run_sources: vec!["local".to_string()],
            ..Default::default()
        };

        let msg = zero_match_diagnostic("A", &filter, &rows, rows.len());

        assert!(
            !msg.contains("--run-source") || !msg.contains("not found"),
            "must NOT fire the unknown-source hint when the value is \
             present in the pool; got:\n{msg}",
        );
    }

    /// `zero_match_diagnostic` fires the dirty-form hint for a
    /// `--project-commit X` filter when the pool contains a
    /// matching `X-dirty` row — pointing the operator at the
    /// dirty form so they don't have to manually scan
    /// `stats list-values`. The hint must name the original
    /// value, the dirty form, and the suggested replacement
    /// flag form.
    #[test]
    fn zero_match_diagnostic_project_commit_dirty_hint_fires() {
        let mut row = make_row("scn", "1n1l1c1t", true, 1.0);
        row.commit = Some("abcdef1-dirty".to_string());
        let rows = vec![row];
        let filter = RowFilter {
            project_commits: vec!["abcdef1".to_string()],
            ..Default::default()
        };

        let msg = zero_match_diagnostic("A", &filter, &rows, rows.len());

        assert!(
            msg.contains("no rows match `--project-commit abcdef1`"),
            "hint must name the unmatched filter value verbatim; \
             got:\n{msg}",
        );
        assert!(
            msg.contains("`abcdef1-dirty` exists in the pool"),
            "hint must surface the dirty form found in the pool; \
             got:\n{msg}",
        );
        assert!(
            msg.contains("did you mean `--project-commit abcdef1-dirty`"),
            "hint must propose the dirty form as the corrected flag; \
             got:\n{msg}",
        );
    }

    /// Companion to
    /// `zero_match_diagnostic_project_commit_dirty_hint_fires`
    /// for the `kernel_commits` arm. Same shape: hint names the
    /// unmatched value, the matching `-dirty` form found in the
    /// pool, and the suggested `--kernel-commit` replacement.
    /// A regression that wired the kernel_commits arm to scan
    /// `row.commit` (or never wired it at all) would surface
    /// here as a missing hint.
    #[test]
    fn zero_match_diagnostic_kernel_commit_dirty_hint_fires() {
        let mut row = make_row("scn", "1n1l1c1t", true, 1.0);
        row.kernel_commit = Some("kabcde7-dirty".to_string());
        let rows = vec![row];
        let filter = RowFilter {
            kernel_commits: vec!["kabcde7".to_string()],
            ..Default::default()
        };

        let msg = zero_match_diagnostic("A", &filter, &rows, rows.len());

        assert!(
            msg.contains("no rows match `--kernel-commit kabcde7`"),
            "hint must name the unmatched kernel_commit value verbatim; \
             got:\n{msg}",
        );
        assert!(
            msg.contains("`kabcde7-dirty` exists in the pool"),
            "hint must surface the dirty form found in the pool; \
             got:\n{msg}",
        );
        assert!(
            msg.contains("did you mean `--kernel-commit kabcde7-dirty`"),
            "hint must propose the dirty form as the corrected flag; \
             got:\n{msg}",
        );
    }

    /// `zero_match_diagnostic` appends the `stats list-values`
    /// redirect when the operator narrowed on a commit
    /// dimension (project_commits OR kernel_commits populated)
    /// — that redirect points at the per-dimension dump where
    /// the commit values can be cross-referenced. Without a
    /// commit-dim filter the redirect is suppressed because
    /// `list-values` would dump every dimension, which is no
    /// more actionable than the existing `stats list` redirect
    /// at the top of the message for a kernel / scheduler /
    /// topology miss.
    #[test]
    fn zero_match_diagnostic_list_values_redirect_when_commit_dim_populated() {
        let row = make_row("scn", "1n1l1c1t", true, 1.0);
        let rows = vec![row];
        let filter = RowFilter {
            project_commits: vec!["abcdef1".to_string()],
            ..Default::default()
        };

        let msg = zero_match_diagnostic("A", &filter, &rows, rows.len());

        assert!(
            msg.contains("cargo ktstr stats list-values"),
            "must include the list-values redirect when commit \
             dim filter is populated; got:\n{msg}",
        );

        // Same redirect when only kernel_commits is populated.
        let filter_kc = RowFilter {
            kernel_commits: vec!["kabcde7".to_string()],
            ..Default::default()
        };
        let msg_kc = zero_match_diagnostic("A", &filter_kc, &rows, rows.len());
        assert!(
            msg_kc.contains("cargo ktstr stats list-values"),
            "list-values redirect must also fire on the \
             kernel_commits arm; got:\n{msg_kc}",
        );
    }

    /// Without a commit-dim filter populated, the list-values
    /// redirect must NOT fire — generic kernel / scheduler /
    /// topology / work-type misses already get the `stats list`
    /// redirect, and a list-values dump would be noise rather
    /// than signal. Pins the suppression so a regression that
    /// always emitted the redirect (or omitted the touched-
    /// commit-dim guard) surfaces here.
    #[test]
    fn zero_match_diagnostic_no_list_values_redirect_when_no_commit_dim() {
        let row = make_row("scn", "1n1l1c1t", true, 1.0);
        let rows = vec![row];
        // Filter narrowed on a non-commit dim only — the
        // redirect must stay quiet.
        let filter = RowFilter {
            schedulers: vec!["scx_alpha".to_string()],
            ..Default::default()
        };

        let msg = zero_match_diagnostic("A", &filter, &rows, rows.len());

        assert!(
            !msg.contains("cargo ktstr stats list-values"),
            "list-values redirect must NOT fire when no commit-dim \
             filter is populated; got:\n{msg}",
        );
    }
}
