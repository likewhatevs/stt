//! Per-row data carriers for the comparison output.
//!
//! Three layers, each consumed by the renderer in the parent
//! module's `write_diff` / `write_show` paths:
//!
//! 1. [`ThreadGroup`] — the per-axis aggregation result. One
//!    instance per group key (pcomm / cgroup / comm / comm-exact
//!    skeleton); carries the per-metric [`super::Aggregated`] map,
//!    the cgroup v2 enrichment counters when grouping by cgroup,
//!    the union of `comm` / `pcomm` member literals for
//!    pattern-aware rendering, and the average start-time tick
//!    used for the uptime% column.
//!
//! 2. [`DiffRow`] / [`DerivedRow`] — the per-`(group, metric)`
//!    diff carriers. Each owns its baseline + candidate aggregate
//!    plus the precomputed delta and pct cells. The
//!    [`DiffRow::sort_key`] / [`DerivedRow::sort_key`] helpers
//!    project to the `|delta_pct|`-descending ordering the
//!    renderer uses by default; the multi-key sort in compare.rs
//!    overrides this ordering when `--sort-by` is set.
//!
//! 3. [`CtprofDiff`] — the full comparison result. Aggregates the
//!    per-thread diff rows ([`Self::rows`]), the derived-metric
//!    rows ([`Self::derived_rows`]), the unmatched-keys lists
//!    ([`Self::only_baseline`] / [`Self::only_candidate`]) AFTER
//!    fudging removes pairs joined via thread-population overlap,
//!    the [`FudgedPair`] entries documenting matched cgroup
//!    renames, the host PSI snapshots, the per-cgroup smaps_rollup
//!    maps, and the global sched_ext sysfs snapshot. Consumed by
//!    `write_diff` directly; the pointer-hash identity used to
//!    deduplicate rows downstream still works because every
//!    consumer passes `&CtprofDiff` by reference, never by value.
//!
//! The types in this module are pure data carriers — no rendering
//! logic, no aggregation logic. Aggregation lives next door in
//! [`super::aggregate`]; rendering lives in mod.rs (Phase D-E
//! moves it into render.rs / report.rs).

use std::collections::BTreeMap;

use crate::ctprof::{CgroupStats, Psi};

use super::{Aggregated, DerivedValue, ScaleLadder};

/// Aggregated metrics for every thread matched by one group key.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ThreadGroup {
    pub key: String,
    pub thread_count: usize,
    /// Metric name → aggregated value. Entries are created for
    /// every registered metric; absent keys signal a missed
    /// aggregation step, not a skip.
    pub metrics: BTreeMap<String, Aggregated>,
    /// Only populated when grouping by cgroup — carries the cgroup
    /// v2 enrichment counters (cpu.stat, memory.current) for that
    /// path. Nested here so the renderer can surface them
    /// alongside the thread-metric rows without a second lookup.
    pub cgroup_stats: Option<CgroupStats>,
    /// Distinct member literals contained in this bucket, sorted
    /// ascending. The field carries `comm` literals under
    /// [`super::GroupBy::Comm`] and `pcomm` literals under
    /// [`super::GroupBy::Pcomm`] — both groupings feed the grex
    /// display-label path the same way (each pattern-aware bucket
    /// renders a regex over the union of its members across
    /// baseline + candidate). Empty Vec for groupings that
    /// render the join key directly: [`super::GroupBy::Cgroup`],
    /// [`super::GroupBy::CommExact`], or pattern-aware groupings under
    /// [`super::CompareOptions::no_thread_normalize`] where the join key
    /// IS the literal name and there is nothing to expand into a
    /// regex.
    pub members: Vec<String>,
    /// Average start_time_clock_ticks across group members.
    /// Lower = older = the group has been alive longer on average.
    pub avg_start_ticks: u64,
}

/// One row in the comparison table: `(group, metric)` pair with
/// aggregated values from both sides.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DiffRow {
    /// Internal join key — deterministic across snapshots and
    /// stable for tests / programmatic consumers. For pattern-
    /// aggregated rows ([`super::GroupBy::Comm`] or [`super::GroupBy::Pcomm`]
    /// with bucket size ≥ 2 under default normalization), this is
    /// the token-normalized skeleton the bucket clusters on (e.g.
    /// `kworker/{N}:{N}-mm_percpu_wq` for Comm,
    /// `worker-{N}` for Pcomm); for every other grouping
    /// (CommExact, Cgroup, or pattern-aware grouping under
    /// [`super::CompareOptions::no_thread_normalize`]) it equals the
    /// rendered display key.
    pub group_key: String,
    pub thread_count_a: usize,
    pub thread_count_b: usize,
    /// Relative uptime % for this group (candidate side).
    /// 100% = as long-lived as the oldest group, 0% = just spawned.
    pub uptime_pct: Option<f64>,
    /// Sort-by metric cell: "baseline → candidate (delta%)" for
    /// the metric specified by --sort-by. Same value for every
    /// row in a group. None when no --sort-by is set.
    pub sort_by_cell: Option<String>,
    /// Sort metric's delta for this group (for coloring the SortBy column).
    pub sort_by_delta: Option<f64>,
    pub metric_name: &'static str,
    /// Auto-scale ladder for the row's value/delta cells. Sourced
    /// from `metric.rule.ladder()` at build time so the format
    /// dispatch stays a closed match (no string-keyed
    /// pass-through branch).
    pub metric_ladder: ScaleLadder,
    pub baseline: Aggregated,
    pub candidate: Aggregated,
    /// Signed candidate − baseline for numeric-capable rules.
    pub delta: Option<f64>,
    /// `delta / baseline` as a fraction. `None` when baseline is
    /// zero or the row has no numeric projection.
    pub delta_pct: Option<f64>,
    /// Operator-facing rendering of the group key. Equals
    /// `group_key` for non-pattern groupings; for [`super::GroupBy::Comm`]
    /// or [`super::GroupBy::Pcomm`] pattern buckets containing ≥ 2
    /// distinct member literals, this carries a grex-generated
    /// regex over the union of baseline+candidate members so the
    /// operator sees exactly which names landed in the bucket.
    pub display_key: String,
}

impl DiffRow {
    /// Sort key for "biggest absolute delta %". Numeric rows
    /// with a non-zero baseline sort by `|delta_pct|`; numeric
    /// rows with a zero baseline sort by `|delta|` scaled by a
    /// large constant so any non-zero candidate dominates
    /// percent-based rows; non-numeric rows sink to the bottom.
    pub(super) fn sort_key(&self) -> f64 {
        if let Some(p) = self.delta_pct {
            p.abs()
        } else if let Some(d) = self.delta {
            // Baseline was zero (delta_pct undefined) but candidate
            // is some value — still a visible change. Inflate so it
            // beats percent-only rows in the sort.
            d.abs() * 1e9
        } else {
            f64::NEG_INFINITY
        }
    }
}

/// A pair of cgroup groups fudged together by thread population
/// overlap. Fudging joins a baseline cgroup to a candidate cgroup
/// when their per-cgroup thread-type sets share enough population
/// (Jaccard similarity ≥ 0.90) — a renamed-but-otherwise-identical
/// scope under a shifted path is rejoined for diffing instead of
/// surfacing as separate orphans.
///
/// Fields are role-prefixed: `baseline_*` and `candidate_*` track
/// the two sides of the pair; `overlap` / `jaccard` /
/// `cascaded_children` are pair-level metrics.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct FudgedPair {
    /// Baseline cgroup path of the matched pair — full join key
    /// from the baseline-side bucket. Format: an absolute cgroup
    /// path (e.g. `/system.slice/foo.service`). The form mirrors
    /// what [`super::build_groups`] writes for `super::GroupBy::Cgroup`.
    pub baseline_cgroup: String,
    /// Candidate cgroup path of the matched pair — full join key
    /// from the candidate-side bucket. Same format as
    /// [`Self::baseline_cgroup`].
    pub candidate_cgroup: String,
    /// Number of (pcomm, comm) thread types in the intersection
    /// of the two sides' thread-type sets. Higher = stronger
    /// match.
    pub overlap: usize,
    /// Jaccard similarity coefficient: `|A ∩ B| / |A ∪ B|` over
    /// the thread-type sets. Range `[0.0, 1.0]`. Matching gate
    /// is `jaccard >= 0.90`.
    pub jaccard: f64,
    /// Thread types present in baseline but missing from the
    /// UNION of every candidate matched against this baseline
    /// (per-bcg dedup; see N:1 fudge merge). Each entry is
    /// `pcomm:comm` formatted.
    pub baseline_residual: Vec<String>,
    /// Thread types present in candidate but missing from the
    /// UNION of every baseline matched against this candidate.
    /// Same format as [`Self::baseline_residual`].
    pub candidate_residual: Vec<String>,
    /// Count of cgroup descendants joined via cascade matching
    /// under the shared longest-common-suffix root. Cascade
    /// extends the fudge from the named pair down to children
    /// that share the same suffix relative to their roots.
    pub cascaded_children: usize,
    /// Cascade root on the baseline side: longest common
    /// path-segment suffix stripped from
    /// [`Self::baseline_cgroup`]. Equal to
    /// [`Self::baseline_cgroup`] when no suffix is shared.
    /// Smaps remap re-keys candidate-side smaps data under this
    /// root.
    pub baseline_root: String,
    /// Cascade root on the candidate side: longest common
    /// path-segment suffix stripped from
    /// [`Self::candidate_cgroup`]. Equal to
    /// [`Self::candidate_cgroup`] when no suffix is shared.
    pub candidate_root: String,
}

/// Full comparison result.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct CtprofDiff {
    pub sort_metric_name: Option<&'static str>,
    pub rows: Vec<DiffRow>,
    /// Group keys that appeared in the baseline snapshot but not
    /// in the candidate, AFTER fudging removes pairs that joined
    /// via thread-population overlap. Post-fudge survivors only
    /// — keys that were rejoined to a candidate counterpart
    /// move into [`Self::fudged_pairs`] and drop out of this
    /// list.
    pub only_baseline: Vec<String>,
    /// Group keys that appeared in the candidate snapshot but not
    /// in the baseline, AFTER fudging. Post-fudge survivors only;
    /// same semantics as [`Self::only_baseline`] for the
    /// candidate side.
    pub only_candidate: Vec<String>,
    /// Cgroup pairs joined together via thread-population overlap
    /// (Jaccard ≥ 0.90 over (pcomm, comm) thread-type sets).
    /// Each entry is one matched (baseline, candidate) cgroup
    /// pair plus its overlap / Jaccard / residuals / cascade
    /// metadata. Pairs are emitted by the fudge stage of
    /// [`super::compare`] and consumed by the renderer's "Fudged cgroup
    /// matches" section. Empty under non-cgroup `super::GroupBy` modes
    /// (fudge applies only when keys are cgroup paths).
    pub fudged_pairs: Vec<FudgedPair>,
    /// Baseline-only cgroup-level enrichment rows, keyed by the
    /// cgroup path (after flatten). Populated only for
    /// [`super::GroupBy::Cgroup`].
    pub cgroup_stats_a: BTreeMap<String, CgroupStats>,
    /// Candidate-only cgroup-level enrichment rows, same shape.
    pub cgroup_stats_b: BTreeMap<String, CgroupStats>,
    /// Baseline host-level Pressure Stall Information snapshot.
    /// Always populated (independent of `super::GroupBy`) — host-level
    /// PSI surfaces above the per-thread table for any compare,
    /// not just cgroup-grouped ones.
    pub host_psi_a: Psi,
    /// Candidate host-level PSI snapshot.
    pub host_psi_b: Psi,
    /// Baseline per-process smaps_rollup maps. Default
    /// normalization keys by the token-normalized pcomm
    /// (`pattern_key(&t.pcomm)`) — ephemeral PIDs across snapshots
    /// collapse into one bucket per pcomm pattern (e.g.
    /// `worker-{N}`), and the tgid is intentionally NOT part of
    /// the key (every PID for a given pcomm pattern shares a
    /// bucket; per-field byte counts SUM at
    /// [`super::collect_smaps_rollup`] when multiple PIDs collapse).
    /// Keys match the primary-table Pcomm group keys WHEN ≥2
    /// processes share the same pattern (`firefox`,
    /// `kworker/{N}:{N}`, `worker-{N}`, …). Singleton digit
    /// pcomms diverge intentionally: the primary table reverts
    /// the bucket key to the literal pcomm (e.g. `worker-7`)
    /// when only one process matches the skeleton — see
    /// [`super::build_groups`]'s singleton-revert gate — while smaps
    /// stays normalized (`worker-{N}`) regardless of bucket
    /// size, so cross-snapshot rows still join when PIDs are
    /// ephemeral. The asymmetry is documented on
    /// [`super::collect_smaps_rollup`] and is load-bearing for
    /// memory-leak diffing across reboots; correlation between
    /// the smaps row and the primary table happens via the
    /// shared pcomm pattern, not always via byte-identical keys.
    ///
    /// With [`super::CompareOptions::no_thread_normalize`] set, keys
    /// preserve the literal `pcomm[tgid]` shape so each PID stays
    /// attributable to its specific process instance — the
    /// `[tgid]` is preserved precisely so two distinct PIDs
    /// sharing a pcomm don't collide within a snapshot. Rows
    /// only join across snapshots when the same process instance
    /// ran on both sides, which is the price of literal mode.
    ///
    /// Populated from the per-thread leader rows of the
    /// snapshot (tid == tgid; see [`crate::ctprof::ThreadState::smaps_rollup_kb`]).
    pub smaps_rollup_a: BTreeMap<String, BTreeMap<String, u64>>,
    /// Candidate per-process smaps_rollup maps, same shape and
    /// normalization rules as [`Self::smaps_rollup_a`].
    pub smaps_rollup_b: BTreeMap<String, BTreeMap<String, u64>>,
    /// Baseline global sched_ext sysfs snapshot. `None` when
    /// the baseline kernel had no `/sys/kernel/sched_ext/`
    /// directory (CONFIG_SCHED_CLASS_EXT=n build).
    pub sched_ext_a: Option<crate::ctprof::SchedExtSysfs>,
    /// Candidate global sched_ext sysfs snapshot, same shape.
    pub sched_ext_b: Option<crate::ctprof::SchedExtSysfs>,
    /// One row per `(matched group, derived metric)` pair. Each
    /// derivation in [`super::CTPROF_DERIVED_METRICS`] consumes
    /// already-aggregated input metrics from the group's
    /// metrics map (see [`ThreadGroup::metrics`]) and produces a
    /// scalar `f64` with its own unit. `None`-valued sides
    /// signal "not computable" — either the input metric was
    /// missing on that side (capture-time CONFIG gate not set,
    /// jemalloc not linked) or the formula's denominator was
    /// zero. Surfaced by [`super::write_diff`] in the dedicated
    /// `## Derived metrics` section after the main table.
    pub derived_rows: Vec<DerivedRow>,
}

/// One row in the derived-metrics table: `(matched group,
/// derivation)` with the computed scalar from both sides.
///
/// Mirrors [`DiffRow`] in shape so the renderer can reuse the
/// same `(group | threads | metric | baseline | candidate |
/// delta | %)` column layout. The `%` column is suppressed for
/// rows whose derivation is a ratio
/// ([`super::DerivedMetricDef::is_ratio`] true) — absolute delta on a
/// `[0, 1]` ratio is already in percentage points so a delta_pct
/// readout would be confusing.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DerivedRow {
    pub group_key: String,
    pub display_key: String,
    pub thread_count_a: usize,
    pub thread_count_b: usize,
    pub metric_name: &'static str,
    /// Auto-scale ladder for the row's value/delta cells. Mirrors
    /// [`DiffRow::metric_ladder`]; sourced from
    /// [`super::DerivedMetricDef::ladder`] at build time.
    pub metric_ladder: ScaleLadder,
    /// True when the derivation produces a ratio. Renderer
    /// suppresses the `%` column for ratio rows.
    pub is_ratio: bool,
    /// `None` when the input metric was missing on this side or
    /// the formula divides by zero.
    pub baseline: Option<DerivedValue>,
    /// `None` with the same semantics as [`Self::baseline`].
    pub candidate: Option<DerivedValue>,
    /// Signed candidate − baseline; `None` when either side is
    /// `None`.
    pub delta: Option<f64>,
    /// `delta / baseline`; `None` when baseline is zero, either
    /// side is `None`, OR the row is a ratio (suppressed for
    /// ratios so a `0.5 → 0.6` row doesn't render as
    /// `+20%` when the natural read is `+10pp`).
    pub delta_pct: Option<f64>,
    /// Pre-rendered cell string for the SortBy column under
    /// `--sort-by`. Same value for every row in a group; `None`
    /// when no `--sort-by` is set. Mirrors
    /// [`DiffRow::sort_by_cell`].
    pub sort_by_cell: Option<String>,
    /// Sort metric's delta for this group, used to color the
    /// SortBy column. Mirrors [`DiffRow::sort_by_delta`].
    pub sort_by_delta: Option<f64>,
}

impl DerivedRow {
    /// Sort key mirroring [`DiffRow::sort_key`] for default
    /// `|delta_pct|`-descending ordering. Ratio rows have
    /// `delta_pct == None` by design so they sort by their
    /// absolute delta, scaled by `1e9` so a non-zero ratio
    /// movement dominates a percent-based row whose baseline
    /// happens to be zero.
    pub(super) fn sort_key(&self) -> f64 {
        if let Some(p) = self.delta_pct {
            p.abs()
        } else if let Some(d) = self.delta {
            d.abs() * 1e9
        } else {
            f64::NEG_INFINITY
        }
    }
}
