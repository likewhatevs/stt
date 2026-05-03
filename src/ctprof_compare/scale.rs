//! Auto-scale ladder dispatch and the cell-formatting helpers that
//! consume it.
//!
//! Two layers:
//!
//! 1. [`ScaleLadder`] — closed enumeration of unit families
//!    (ns, µs, Bytes, Ticks, Unitless, None) and the [`auto_scale`]
//!    free function that maps an `(f64, ladder)` pair to a
//!    `(scaled_value, scaled_unit)` pair. The ladder choice flows
//!    from [`super::AggRule::ladder`] for primary metrics and from
//!    [`super::DerivedMetricDef::ladder`] for derived metrics; the
//!    cgroup-stats render path passes a ladder directly. A
//!    type-system-mismatch between an `AggRule` variant and its
//!    declared ladder is a compile error rather than a silent
//!    pass-through, because the dispatch is a closed match.
//!
//! 2. The `format_*` helpers (`format_value_cell`,
//!    `format_scaled_u64`, `format_derived_value_cell`,
//!    `format_derived_delta_cell`, `format_optional_limit`,
//!    `format_cpu_max`, `cgroup_optional_limit_cell`,
//!    `cgroup_limits_cell`, `format_delta_cell`) — render-only
//!    entry points that consume an [`super::Aggregated`] / scalar
//!    plus a ladder and produce the `String` cell that feeds
//!    `comfy_table` rows in the parent module's `write_diff` /
//!    `write_show` paths.
//!
//! All of this is pure formatting; no underlying numeric values
//! used for sort order or delta math are mutated here.

use super::{Aggregated, DerivedValue};

/// Closed enumeration of auto-scale ladders driven by phase 4
/// format dispatch.
///
/// Picks the unit family up the type system rather than a free-form
/// `&'static str` tag. Each [`AggRule`] variant maps to exactly one
/// ladder via [`AggRule::ladder`]; each [`super::DerivedMetricDef`] entry
/// carries a ladder via [`super::DerivedMetricDef::ladder`]; the cgroup-
/// level render path passes a ladder directly. A registry typo or
/// drift between accessor newtype and ladder choice fails to compile
/// at the registry edit site rather than silently routing through
/// an "unknown unit" pass-through arm at render time.
///
/// The six ladder variants and their step-up rules:
/// - [`Ns`](Self::Ns): ns → µs (×1e3) → ms (×1e6) → s (×1e9).
///   Decimal prefixes — SI time, not binary. Used for
///   [`AggRule::SumNs`] (cumulative ns counters),
///   [`AggRule::MaxPeak`] (lifetime ns high-water marks),
///   [`AggRule::MaxGaugeNs`] (instantaneous ns gauges), and
///   the `"ns"` derived-metric ladder.
/// - [`Us`](Self::Us): µs → ms (×1e3) → s (×1e6). Decimal SI
///   prefixes. The cgroup `cpu_usage_usec` and `throttled_usec`
///   fields are reported by the kernel in microseconds; this
///   ladder scales them up the same way the `Ns` ladder scales
///   nanoseconds.
/// - [`Bytes`](Self::Bytes): B → KiB → MiB → GiB → TiB. IEC binary
///   prefixes (×1024) for byte counts. Used for
///   [`AggRule::SumBytes`] and any byte-typed derived metric.
/// - [`Ticks`](Self::Ticks): ticks → Kticks (×1e3) → Mticks (×1e6).
///   Decimal prefixes for clock-tick counts
///   (`utime_clock_ticks`, `stime_clock_ticks`); the unit
///   itself is opaque (the kernel's `USER_HZ` rate is
///   host-dependent), so an SI prefix is the most we can
///   promise.
/// - [`Unitless`](Self::Unitless): "" → K → M → G. Decimal
///   prefixes for non-dimensional counters (wakeups, migrations,
///   csw, syscall counts). Used for [`AggRule::SumCount`] and
///   [`AggRule::MaxGaugeCount`].
/// - [`None`](Self::None): no ladder — values render as the bare
///   integer with no unit suffix and no scaling. Used for
///   [`AggRule::Mode`] / [`AggRule::ModeChar`] /
///   [`AggRule::ModeBool`] (categorical strings),
///   [`AggRule::RangeI32`] / [`AggRule::RangeU32`] (bounded
///   ordinals), and [`AggRule::Affinity`] (cpuset summaries) —
///   the [`Aggregated`] [`std::fmt::Display`] impl handles render for
///   these directly.
///
/// The threshold for stepping up is `|value| >= next_scale`.
/// Sign is preserved through scaling (negative deltas pass
/// through). Zero stays at base unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScaleLadder {
    Ns,
    Us,
    Bytes,
    Ticks,
    Unitless,
    None,
}

impl ScaleLadder {
    /// Base unit string for this ladder — what [`auto_scale`]
    /// returns for a value at the bottom of the ladder. Used by
    /// the format helpers to detect whether a value stepped up
    /// (`auto_scale(v).1 != ladder.base_unit()` ⇒ stepped up,
    /// render with the scaled unit; equal ⇒ no step-up, render
    /// the bare integer with the base unit suffix).
    pub fn base_unit(&self) -> &'static str {
        match self {
            ScaleLadder::Ns => "ns",
            ScaleLadder::Us => "µs",
            ScaleLadder::Bytes => "B",
            ScaleLadder::Ticks => "ticks",
            ScaleLadder::Unitless | ScaleLadder::None => "",
        }
    }
}

/// Auto-scale a numeric value to a more readable magnitude based
/// on its [`ScaleLadder`]. Returns the scaled value paired with
/// the scaled unit string.
///
/// This is render-only; the underlying numeric values used for
/// sort order and delta math are untouched.
///
/// Phase 4: dispatches on a closed [`ScaleLadder`] enum rather
/// than a free-form unit string. The mapping from
/// [`AggRule`] / [`super::DerivedMetricDef`] / cgroup-render call site
/// to [`ScaleLadder`] lives at the type level — see
/// [`AggRule::ladder`] and [`super::DerivedMetricDef::ladder`] — so a
/// registry typo can no longer fall through an `other =>
/// pass-through` arm and silently render the unscaled value.
pub(super) fn auto_scale(value: f64, ladder: ScaleLadder) -> (f64, &'static str) {
    let abs = value.abs();
    match ladder {
        ScaleLadder::Ns => {
            if abs >= 1e9 {
                (value / 1e9, "s")
            } else if abs >= 1e6 {
                (value / 1e6, "ms")
            } else if abs >= 1e3 {
                (value / 1e3, "µs")
            } else {
                (value, "ns")
            }
        }
        ScaleLadder::Us => {
            if abs >= 1e6 {
                (value / 1e6, "s")
            } else if abs >= 1e3 {
                (value / 1e3, "ms")
            } else {
                (value, "µs")
            }
        }
        ScaleLadder::Bytes => {
            const KIB: f64 = 1024.0;
            const MIB: f64 = 1024.0 * KIB;
            const GIB: f64 = 1024.0 * MIB;
            const TIB: f64 = 1024.0 * GIB;
            if abs >= TIB {
                (value / TIB, "TiB")
            } else if abs >= GIB {
                (value / GIB, "GiB")
            } else if abs >= MIB {
                (value / MIB, "MiB")
            } else if abs >= KIB {
                (value / KIB, "KiB")
            } else {
                (value, "B")
            }
        }
        ScaleLadder::Ticks => {
            if abs >= 1e6 {
                (value / 1e6, "Mticks")
            } else if abs >= 1e3 {
                (value / 1e3, "Kticks")
            } else {
                (value, "ticks")
            }
        }
        ScaleLadder::Unitless => {
            if abs >= 1e9 {
                (value / 1e9, "G")
            } else if abs >= 1e6 {
                (value / 1e6, "M")
            } else if abs >= 1e3 {
                (value / 1e3, "K")
            } else {
                (value, "")
            }
        }
        ScaleLadder::None => (value, ""),
    }
}

/// Format a per-row baseline / candidate cell for [`super::write_diff`].
/// Numeric aggregates ([`Aggregated::Sum`] / [`Aggregated::Max`])
/// run through [`auto_scale`] so large values render in a
/// readable magnitude (`1.235ms` instead of `1234567ns`). When
/// the scaled unit equals the ladder's base unit (no step-up was
/// triggered), the original integer value is rendered verbatim
/// — this avoids polluting small numbers with a `.000` suffix.
/// Non-numeric aggregates (`OrdinalRange`, `Mode`, `Affinity`)
/// fall through to the [`Aggregated`] [`std::fmt::Display`] impl
/// unchanged because no scaling applies; the ladder is
/// [`ScaleLadder::None`] for these and the suffix is empty.
pub fn format_value_cell(agg: &Aggregated, ladder: ScaleLadder) -> String {
    match agg {
        Aggregated::Sum(v) => format_scaled_u64(*v, ladder),
        Aggregated::Max(v) => format_scaled_u64(*v, ladder),
        _ => format!("{agg}{}", ladder.base_unit()),
    }
}

/// Auto-scale a `u64` value at the given ladder and render it as
/// a cell. Helper for [`format_value_cell`] — the Sum and Max
/// arms share this exact logic. Also used by the `ctprof
/// show` renderer for the cgroup-stats secondary table, where
/// each scalar stands alone (no baseline/candidate pair to fold
/// into a delta cell).
pub fn format_scaled_u64(v: u64, ladder: ScaleLadder) -> String {
    let (scaled, scaled_unit) = auto_scale(v as f64, ladder);
    if scaled_unit == ladder.base_unit() {
        // No step-up — render the original integer to preserve
        // exact precision (auto_scale's f64 round-trip is
        // identity below the threshold, but the integer form is
        // shorter and avoids the `.000` suffix).
        format!("{v}{}", ladder.base_unit())
    } else {
        format!("{scaled:.3}{scaled_unit}")
    }
}

/// Format a derived-metric value cell for the `## Derived metrics`
/// table. Ratio rows (`is_ratio: true`, [`ScaleLadder::None`])
/// render with three decimals (`0.873`); ns / B / ticks ladders
/// route through the same auto-scale ladder as the main table.
/// Negative values (e.g. a negative `live_heap_estimate`) carry
/// their explicit minus sign through the format.
pub fn format_derived_value_cell(v: DerivedValue, ladder: ScaleLadder, is_ratio: bool) -> String {
    let value = v.as_f64();
    if is_ratio {
        return format!("{value:.3}");
    }
    let (scaled, scaled_unit) = auto_scale(value, ladder);
    if scaled_unit == ladder.base_unit() {
        // No ladder step-up — render two decimals to preserve
        // the fractional precision derived averages carry (e.g.
        // wait_sum=1234 ns / wait_count=10 = 123.40 ns). The
        // primary-table integer formatter (format_scaled_u64)
        // strips fractions because its inputs ARE integers; the
        // derived path's inputs are `f64` divisions, so two
        // decimals keep the signal intact.
        format!("{value:.2}{}", ladder.base_unit())
    } else {
        format!("{scaled:.3}{scaled_unit}")
    }
}

/// Format the signed delta cell for a derived row. Mirrors
/// [`format_derived_value_cell`] but always carries an explicit
/// `+`/`-` sign so the operator can read directionality at a
/// glance. Ratios render with three decimals (`+0.100` is +10pp);
/// other ladders route through `auto_scale` and pick up the
/// scaled unit suffix.
pub fn format_derived_delta_cell(d: f64, ladder: ScaleLadder, is_ratio: bool) -> String {
    if is_ratio {
        return format!("{d:+.3}");
    }
    let (scaled, scaled_unit) = auto_scale(d, ladder);
    if scaled_unit == ladder.base_unit() {
        format!("{d:+.2}{}", ladder.base_unit())
    } else {
        format!("{scaled:+.3}{scaled_unit}")
    }
}

/// Render an `Option<u64>` cgroup limit as either `max` (no
/// limit / kernel emitted the literal `max` token) or the
/// auto-scaled value. Used for `memory.max`, `memory.high`,
/// `memory.low`, `memory.min`, `pids.max`, `cpu.max` quota.
/// Mirrors the kernel's own display: `cat memory.max` prints
/// `max` when no cap is set, a u64 byte count otherwise.
pub fn format_optional_limit(v: Option<u64>, ladder: ScaleLadder) -> String {
    match v {
        Some(n) => format_scaled_u64(n, ladder),
        None => "max".to_string(),
    }
}

/// Render a `cpu.max` pair as `<quota>/<period>` where quota is
/// either `max` (no cap) or the auto-scaled µs value. Period is
/// always present (default 100_000 µs per
/// `default_bw_period_us()` at `kernel/sched/sched.h:441`). The
/// `<quota>/<period>` separator is THIS crate's display
/// convention — the kernel itself emits raw integers in
/// `cat cpu.max` (space-separated, no auto-scale); we
/// auto-scale via [`format_scaled_u64`] for human-friendly
/// output, which also widens the visual delimiter from the
/// kernel's space to a slash.
pub fn format_cpu_max(quota: Option<u64>, period_us: u64) -> String {
    let q = match quota {
        Some(q) => format_scaled_u64(q, ScaleLadder::Us),
        None => "max".to_string(),
    };
    let p = format_scaled_u64(period_us, ScaleLadder::Us);
    format!("{q}/{p}")
}

/// Render a baseline → candidate cell for an `Option<u64>`
/// LIMIT (e.g. `memory.max`, `memory.high`, `pids.max`). `None`
/// reads as `max` (no limit) per [`format_optional_limit`]; a
/// step from concrete to `max` between snapshots renders as
/// `<value> → max`.
pub fn cgroup_optional_limit_cell(
    baseline: Option<u64>,
    candidate: Option<u64>,
    ladder: ScaleLadder,
) -> String {
    let bl = format_optional_limit(baseline, ladder);
    let cd = format_optional_limit(candidate, ladder);
    if baseline == candidate {
        // No diff — render once. Avoids the `max → max` redundancy
        // and keeps the limits column scannable when nothing
        // changed.
        return bl;
    }
    format!("{bl} → {cd}")
}

/// Render a baseline → candidate cell for `cpu.max`
/// `(quota, period)` pairs. When both pairs are equal, renders
/// once via [`format_cpu_max`]; otherwise renders as
/// `<a> → <b>`. Mirrors [`cgroup_optional_limit_cell`]'s
/// equality-collapse policy.
pub fn cgroup_limits_cell(
    baseline: Option<(Option<u64>, u64)>,
    candidate: Option<(Option<u64>, u64)>,
) -> String {
    let render = |pair: Option<(Option<u64>, u64)>| match pair {
        Some((q, p)) => format_cpu_max(q, p),
        None => "-".to_string(),
    };
    let bl = render(baseline);
    let cd = render(candidate);
    if bl == cd {
        return bl;
    }
    format!("{bl} → {cd}")
}

/// Format a per-row delta cell for [`super::write_diff`]. Routes the
/// signed numeric delta through [`auto_scale`] so a large delta
/// renders in a readable magnitude with the matching prefix
/// applied to the ladder's base unit. Sign is preserved (rendered
/// with `+` or `-`). When no step-up was triggered AND the delta
/// is integer-valued, the cell renders as the bare signed integer
/// to match [`format_value_cell`]'s short-circuit (so `+5ns`
/// instead of `+5.000ns`); otherwise the scaled f64 renders with
/// 3 decimals.
pub(super) fn format_delta_cell(delta: f64, ladder: ScaleLadder) -> String {
    let (scaled, scaled_unit) = auto_scale(delta, ladder);
    if scaled_unit == ladder.base_unit() && delta.fract() == 0.0 {
        format!("{:+}{scaled_unit}", delta as i64)
    } else {
        format!("{scaled:+.3}{scaled_unit}")
    }
}
