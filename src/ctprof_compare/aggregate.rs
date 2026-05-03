//! Per-group aggregate value carrier and the merge primitives
//! that compose it.
//!
//! Two layers:
//!
//! 1. [`Aggregated`] — closed enum of the per-metric aggregate
//!    shapes a thread group produces: [`Aggregated::Sum`] /
//!    [`Aggregated::Max`] for the numeric reduction rules,
//!    [`Aggregated::OrdinalRange`] for `[min, max]` interval rules
//!    over signed kernel ordinals, [`Aggregated::Mode`] for the
//!    categorical mode-with-tally aggregation, and
//!    [`Aggregated::Affinity`] for the cpuset-cardinality summary.
//!    Each variant carries the shape its rule's accessor produced
//!    after the cross-thread reduction. `numeric()` projects to
//!    `Option<f64>` for the delta-math / sort path; categorical
//!    rules return `None`. The [`std::fmt::Display`] impl is
//!    consumed by [`super::scale::format_value_cell`] when the
//!    rule's ladder is [`super::ScaleLadder::None`].
//!
//! 2. [`merge_aggregated_into`] — the per-variant merge primitive
//!    that composes summaries when N candidate groups collapse
//!    onto one baseline-keyed row (the N:1 fudge merge): Sum
//!    counters add, Max peaks take max-of-maxes (NOT a sum —
//!    summing peaks invents peaks taller than any thread
//!    observed), OrdinalRange unions the bounds, Mode unions the
//!    per-value tally maps, Affinity widens the cardinality bounds
//!    and collapses the exact CPU list to `None` on any mismatch.
//!    Variant mismatches are silently dropped — every group's
//!    metrics map shares the same [`super::CTPROF_METRICS`] keys,
//!    so the fall-through arm is defense-in-depth.
//!
//! The [`AffinitySummary`] CPU-set cardinality carrier and the
//! [`format_cpu_range`] runs-collapsing renderer travel with
//! [`Aggregated::Affinity`] because the renderer is consumed only
//! by the Affinity arm of the [`std::fmt::Display`] impl.

use std::collections::BTreeMap;
use std::fmt;

/// Aggregated metric value for a single [`super::ThreadGroup`].
///
/// Carries both a numeric projection (used for delta math and
/// sort order) and a display form. Not every rule produces a
/// numeric — the categorical rules
/// ([`super::AggRule::Mode`] / [`super::AggRule::ModeChar`] /
/// [`super::AggRule::ModeBool`]) aggregate to a string, which has no
/// scalar — so the numeric is optional and rows without one
/// fall to the bottom of the default sort.
#[derive(Debug, Clone)]
pub enum Aggregated {
    /// Group-wide sum produced by the
    /// [`super::AggRule::SumCount`] / [`super::AggRule::SumNs`] /
    /// [`super::AggRule::SumTicks`] / [`super::AggRule::SumBytes`] rules. The
    /// dispatch unwraps the typed newtype's inner `u64` after
    /// the [`crate::metric_types::Summable::sum_across`]
    /// reduction; storage stays u64 to preserve full precision
    /// across the entire schedstats / byte / tick range with
    /// no lossy cast at aggregation time. Phase 4 will read
    /// the registry's `unit` tag (not the wrapper type) at
    /// render time to pick the auto-scale ladder.
    Sum(u64),
    /// Group-wide maximum produced by the
    /// [`super::AggRule::MaxPeak`] / [`super::AggRule::MaxGaugeNs`] /
    /// [`super::AggRule::MaxGaugeCount`] rules. Distinct variant from
    /// `Sum` so a downstream consumer that wants to surface
    /// "the worst single thread" rather than "the
    /// summed-across-threads value" can match without name-
    /// matching against the metric registry. Storage is u64 to
    /// preserve full ns precision across the entire schedstats
    /// range (no `as f64` lossy cast at aggregation time).
    Max(u64),
    /// Group-wide `[min, max]` interval produced by the
    /// [`super::AggRule::RangeI32`] / [`super::AggRule::RangeU32`] rules.
    /// Both bounds widen to `i64` at the dispatch boundary
    /// (`i64::from(OrdinalI32.0)` / `i64::from(OrdinalU32.0)`)
    /// — `OrdinalI32` carries a signed kernel-side range
    /// (`nice` includes negative values) and `OrdinalU32` fits
    /// into `i64` losslessly, so a single signed scalar
    /// represents both ordinal widths without losing the sign
    /// from `OrdinalI32` or wrapping the magnitude from
    /// `OrdinalU32`. Delta math takes the midpoint
    /// (`(min + max) / 2`) so a one-sided shift surfaces in
    /// the rendered delta column.
    OrdinalRange {
        min: i64,
        max: i64,
    },
    /// Categorical aggregate carrying per-value counts across
    /// the bucket. `tallies` maps each observed value to its
    /// occurrence count; `total` is the bucket size (count of
    /// every contributor, including any empty-string fallbacks).
    /// The mode (most-frequent value) is derived on demand via
    /// [`Aggregated::mode_value`] / [`Aggregated::mode_count`]
    /// so cross-bucket merges (N:1 fudge) compose correctly:
    /// unioning two buckets' tallies gives the true cross-bucket
    /// frequency for each value, not just the per-bucket
    /// max-count. Empty buckets surface as `tallies: empty,
    /// total: bucket_size`; the renderer handles the empty case
    /// by emitting `~` in place of the mode value.
    Mode {
        tallies: BTreeMap<String, usize>,
        total: usize,
    },
    Affinity(AffinitySummary),
}

/// CPU-affinity aggregation result.
///
/// `uniform` is `Some(cpus)` when every thread in the group shared
/// the same allowed set; otherwise heterogeneous and the renderer
/// emits "N-M cpus (mixed)".
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AffinitySummary {
    pub min_cpus: usize,
    pub max_cpus: usize,
    pub uniform: Option<Vec<u32>>,
}

impl Aggregated {
    /// Scalar projection for delta math. `None` when the rule
    /// produces no meaningful scalar (categorical mode, affinity
    /// with heterogeneous cpusets).
    pub fn numeric(&self) -> Option<f64> {
        match self {
            Aggregated::Sum(v) => Some(*v as f64),
            Aggregated::Max(v) => Some(*v as f64),
            Aggregated::OrdinalRange { min, max } => {
                // Midpoint: keeps a min→max shift on one end visible
                // in the delta without privileging either bound.
                Some((*min as f64 + *max as f64) / 2.0)
            }
            Aggregated::Mode { .. } => None,
            Aggregated::Affinity(s) => {
                // Number of allowed CPUs is the natural scalar. When
                // the group is uniform, `min_cpus == max_cpus`; when
                // heterogeneous, midpoint parallels OrdinalRange.
                Some((s.min_cpus as f64 + s.max_cpus as f64) / 2.0)
            }
        }
    }

    /// Construct an `Aggregated::Mode` from a single
    /// (value, count, total) triple, common in test fixtures
    /// and the single-bucket aggregate path. Equivalent to
    /// inserting `(value, count)` into a fresh tally map.
    /// `count == 0` produces an empty tally (the empty-bucket
    /// shape), preserving the historical contract that an
    /// empty Mode renders as `~` while still carrying a
    /// well-defined `total` for the rendered "(count/total)"
    /// suffix.
    pub fn mode_single(value: String, count: usize, total: usize) -> Aggregated {
        let mut tallies = BTreeMap::new();
        if count > 0 {
            tallies.insert(value, count);
        }
        Aggregated::Mode { tallies, total }
    }

    /// The most-frequent value tracked by an `Aggregated::Mode`,
    /// or the empty string when the tallies map is empty (an
    /// empty bucket whose `total` may still be non-zero, e.g.
    /// when every contributor produced no categorical sample).
    /// Ties on count are broken by lexicographic order on the
    /// VALUE — the lex-smallest key wins. Deterministic across
    /// runs and matches the historical Mode contract under the
    /// old `value, count` shape.
    pub fn mode_value(&self) -> &str {
        match self {
            Aggregated::Mode { tallies, .. } => {
                // BTreeMap iterates keys in lex order. Use a
                // strict-greater fold so on a count tie the
                // first key encountered (lex-smallest) wins;
                // `max_by_key` returns the last-seen at a tie,
                // which would invert the lex order.
                let mut best: Option<(&str, usize)> = None;
                for (k, c) in tallies {
                    match best {
                        None => best = Some((k.as_str(), *c)),
                        Some((_, bc)) if *c > bc => best = Some((k.as_str(), *c)),
                        _ => {}
                    }
                }
                best.map(|(v, _)| v).unwrap_or("")
            }
            _ => "",
        }
    }

    /// The frequency count of the mode value. Zero when the
    /// tallies map is empty (no contributor produced a
    /// categorical sample). Same tie-break as
    /// [`Self::mode_value`].
    pub fn mode_count(&self) -> usize {
        match self {
            Aggregated::Mode { tallies, .. } => tallies.values().copied().max().unwrap_or(0),
            _ => 0,
        }
    }
}

impl fmt::Display for Aggregated {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Aggregated::Sum(v) => write!(f, "{v}"),
            Aggregated::Max(v) => write!(f, "{v}"),
            Aggregated::OrdinalRange { min, max } => {
                if min == max {
                    write!(f, "{min}")
                } else {
                    write!(f, "{min}..{max}")
                }
            }
            Aggregated::Mode { tallies, total } => {
                // Tie-break: lex-smallest value wins on count
                // ties. Mirrors `Aggregated::mode_value`'s
                // strict-greater fold.
                let mut best: Option<(&str, usize)> = None;
                for (k, c) in tallies {
                    match best {
                        None => best = Some((k.as_str(), *c)),
                        Some((_, bc)) if *c > bc => best = Some((k.as_str(), *c)),
                        _ => {}
                    }
                }
                let value = best.map(|(v, _)| v).unwrap_or("~");
                let count = best.map(|(_, c)| c).unwrap_or(0);
                if count == *total && count > 0 {
                    write!(f, "{value}")
                } else {
                    write!(f, "{value} ({count}/{total})")
                }
            }
            Aggregated::Affinity(s) => {
                if let Some(cpus) = &s.uniform {
                    let n = cpus.len();
                    let range = format_cpu_range(cpus);
                    write!(f, "{n} cpus ({range})")
                } else if s.min_cpus == s.max_cpus {
                    write!(f, "{} cpus (mixed)", s.min_cpus)
                } else {
                    write!(f, "{}-{} cpus (mixed)", s.min_cpus, s.max_cpus)
                }
            }
        }
    }
}

/// Merge `val` into `existing` for the N:1 fudge aggregation
/// (multiple candidate groups collapsed into one baseline-keyed
/// row). Mirrors the canonical [`super::aggregate`] semantics on the
/// merged-set rather than re-aggregating the per-thread inputs:
/// per-group summaries are all the merge has access to, so the
/// merge composes summaries directly.
///
/// Per variant:
/// - [`Aggregated::Sum`]: monotone counters add.
/// - [`Aggregated::Max`]: peaks take max-of-maxes (NOT a sum —
///   wait_max / sleep_max / exec_max are per-thread peaks; summing
///   would invent peaks taller than any thread observed).
/// - [`Aggregated::OrdinalRange`]: union the bounds.
/// - [`Aggregated::Mode`]: union the per-value tally maps and
///   sum the totals. The mode (most-frequent value) is derived
///   on demand from the merged tallies, so cross-bucket
///   frequencies stay accurate — a value that appears in N
///   buckets accumulates its true total count, not just the
///   largest single-bucket count.
/// - [`Aggregated::Affinity`]: cardinality bounds widen —
///   `min_cpus` becomes the smallest of any merged summary's
///   `min_cpus`, `max_cpus` the largest. The exact CPU list
///   (`uniform`) survives only when every merged summary
///   already carried the same `Some(cpus)` list; mismatch or
///   any `None` collapses to `None`.
///
/// Variant mismatches between `existing` and `val` are silently
/// dropped — a fudged pair must agree on the metric's typed
/// rule because every group's metrics map shares the same
/// [`super::CTPROF_METRICS`] keys, so the fall-through arm is
/// defense-in-depth rather than a runtime-reachable case.
pub(super) fn merge_aggregated_into(existing: &mut Aggregated, val: &Aggregated) {
    match (existing, val) {
        (Aggregated::Sum(s), Aggregated::Sum(v)) => {
            *s += v;
        }
        (Aggregated::Max(m), Aggregated::Max(v)) => {
            *m = (*m).max(*v);
        }
        (
            Aggregated::OrdinalRange { min, max },
            Aggregated::OrdinalRange {
                min: vmin,
                max: vmax,
            },
        ) => {
            *min = (*min).min(*vmin);
            *max = (*max).max(*vmax);
        }
        (
            Aggregated::Mode { tallies: et, total },
            Aggregated::Mode {
                tallies: vt,
                total: vtot,
            },
        ) => {
            *total += vtot;
            for (k, c) in vt {
                *et.entry(k.clone()).or_insert(0) += c;
            }
        }
        (Aggregated::Affinity(es), Aggregated::Affinity(vs)) => {
            es.min_cpus = es.min_cpus.min(vs.min_cpus);
            es.max_cpus = es.max_cpus.max(vs.max_cpus);
            es.uniform = match (&es.uniform, &vs.uniform) {
                (Some(a), Some(b)) if a == b => Some(a.clone()),
                _ => None,
            };
        }
        _ => {}
    }
}

/// Render a CPU set as a comma-separated list of contiguous-run
/// ranges (`a-b`), matching the kernel's cpuset display
/// convention (`cat cpuset.cpus` emits `0-3,8`). Assumes the
/// input is sorted ascending — capture layer
/// ([`crate::ctprof::ThreadState::allowed_cpus`]) stores
/// sorted cpusets. Empty input returns an empty string. Used
/// only by [`Aggregated::fmt`]'s Affinity arm; kept here so the
/// renderer and its helper travel together.
pub(super) fn format_cpu_range(cpus: &[u32]) -> String {
    // Collapse contiguous runs to `a-b`, join with commas. Assumes
    // sorted ascending; capture layer stores sorted cpusets.
    if cpus.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    let mut start = cpus[0];
    let mut prev = cpus[0];
    for &c in &cpus[1..] {
        if c == prev + 1 {
            prev = c;
            continue;
        }
        if !out.is_empty() {
            out.push(',');
        }
        if start == prev {
            out.push_str(&start.to_string());
        } else {
            out.push_str(&format!("{start}-{prev}"));
        }
        start = c;
        prev = c;
    }
    if !out.is_empty() {
        out.push(',');
    }
    if start == prev {
        out.push_str(&start.to_string());
    } else {
        out.push_str(&format!("{start}-{prev}"));
    }
    out
}
