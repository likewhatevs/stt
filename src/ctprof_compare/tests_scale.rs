//! Tests for `super::scale` (Phase F.2 per-module redistribution).

#![allow(unused_imports)]
#![allow(clippy::field_reassign_with_default)]

use std::collections::BTreeMap;
use std::path::Path;

use super::*;
use super::aggregate::{format_cpu_range, merge_aggregated_into};
use super::cgroup_merge::{
    merge_cgroup_cpu, merge_cgroup_memory, merge_cgroup_pids, merge_kv_counters,
    merge_max_option, merge_memory_stat, merge_min_option, merge_psi,
};
use super::columns::{compare_columns_for, format_cgroup_only_section_warning};
use super::compare::sort_diff_rows_by_keys;
use super::groups::build_row;
use super::pattern::{
    Segment, apply_systemd_template, cgroup_normalize_skeleton, cgroup_skeleton_tokens,
    classify_token, is_token_separator, pattern_counts_union, pattern_key, split_into_segments,
    tighten_group,
};
use super::render::psi_pair_has_data;
use super::scale::{auto_scale, format_delta_cell};
use super::tests_fixtures::*;
use crate::ctprof::{CgroupStats, CtprofSnapshot, Psi, ThreadState};
use crate::metric_types::{
    Bytes, CategoricalString, CpuSet, MonotonicCount, MonotonicNs, OrdinalI32, PeakNs,
};
use regex::Regex;

/// `format_derived_value_cell` renders a ratio with three
/// decimals (`0.873`); ns/B values route through auto-scale.
#[test]
fn format_derived_value_cell_ratio_three_decimals() {
    let v = DerivedValue::Scalar(0.873_5);
    let cell = format_derived_value_cell(v, ScaleLadder::None, true);
    assert_eq!(cell, "0.874");
}

/// `format_derived_value_cell` auto-scales ns to ms above
/// the threshold.
#[test]
fn format_derived_value_cell_ns_auto_scales() {
    let v = DerivedValue::Scalar(2_500_000.0);
    let cell = format_derived_value_cell(v, ScaleLadder::Ns, false);
    // 2.5e6 ns → 2.500ms via the existing auto_scale ladder.
    assert_eq!(cell, "2.500ms");
}

/// `format_derived_value_cell` preserves fractional precision
/// for derived averages below the auto-scale threshold.
/// avg_wait_ns = 1234 ns / 10 events = 123.4 ns; the
/// formatter renders 123.40ns (two decimals). Without the
/// fractional precision, this would round to "123ns" and
/// the operator would lose the post-decimal signal.
#[test]
fn format_derived_value_cell_ns_preserves_fractional_precision() {
    let v = DerivedValue::Scalar(123.4);
    let cell = format_derived_value_cell(v, ScaleLadder::Ns, false);
    assert_eq!(cell, "123.40ns");
}

/// `format_derived_value_cell` renders a negative B value
/// with the explicit minus sign (live_heap_estimate that
/// went negative).
#[test]
fn format_derived_value_cell_negative_bytes_signed() {
    let two_kib_neg = -(2.0 * 1024.0);
    let v = DerivedValue::Scalar(two_kib_neg);
    let cell = format_derived_value_cell(v, ScaleLadder::Bytes, false);
    assert_eq!(cell, "-2.000KiB");
}

/// `format_derived_delta_cell` carries explicit `+` for
/// positive deltas (mirrors format_delta_cell). Pin the
/// sign carry on a ratio delta of +0.100 = +10pp.
#[test]
fn format_derived_delta_cell_ratio_carries_sign() {
    let cell = format_derived_delta_cell(0.1, ScaleLadder::None, true);
    assert_eq!(cell, "+0.100");
}

/// `live_heap_estimate` can go negative when deallocations
/// dominate — the renderer must preserve the sign through
/// the auto-scale ladder step (here: MiB step-up). Pins the
/// signed-Bytes path that f64 carries. Mirrors the
/// existing KiB-scale test but exercises the MiB threshold
/// so a future regression that drops the sign at a
/// higher rung of the ladder still fails.
#[test]
fn format_derived_value_cell_negative_bytes_at_mib_step() {
    // -2_000_000 bytes: |abs| = 2_000_000 ≥ 1 MiB (1_048_576),
    // < 1 GiB (1_073_741_824) → step to MiB.
    // -2_000_000 / 1_048_576 ≈ -1.907.
    let v = DerivedValue::Scalar(-2_000_000.0);
    let cell = format_derived_value_cell(v, ScaleLadder::Bytes, false);
    assert_eq!(cell, "-1.907MiB");
}

/// `disk_io_fraction` is `is_ratio: true` for the rendering
/// shape (three decimals, no `%` column, no auto-scale) but
/// can exceed 1.0 in practice — readahead pulls more
/// block-device bytes than the syscall requested, pushing
/// `read_bytes / rchar` above 1. Pin that the renderer
/// emits the value verbatim with three decimals when it
/// crosses 1.0 — no clamp, no truncation, no exponent.
#[test]
fn format_derived_value_cell_ratio_above_one_renders_verbatim() {
    let v = DerivedValue::Scalar(1.5);
    let cell = format_derived_value_cell(v, ScaleLadder::None, true);
    assert_eq!(cell, "1.500");
}

/// Boundary: 999 ns stays at the base unit; 1000 ns steps up
/// to µs. Pins the threshold at exactly the prefix transition.
#[test]
fn auto_scale_ns_boundary_stays_at_base_below_threshold() {
    assert_eq!(auto_scale(0.0, ScaleLadder::Ns), (0.0, "ns"));
    assert_eq!(auto_scale(999.0, ScaleLadder::Ns), (999.0, "ns"));
    assert_eq!(auto_scale(1000.0, ScaleLadder::Ns), (1.0, "µs"));
}

/// ns ladder: ns → µs (1e3) → ms (1e6) → s (1e9). Pins each
/// step. Decimal SI prefixes (NOT IEC binary).
#[test]
fn auto_scale_ns_ladder_steps_up_at_powers_of_ten() {
    let (v, u) = auto_scale(1_500.0, ScaleLadder::Ns);
    assert_eq!(u, "µs");
    assert!((v - 1.5).abs() < 1e-9);
    let (v, u) = auto_scale(1_500_000.0, ScaleLadder::Ns);
    assert_eq!(u, "ms");
    assert!((v - 1.5).abs() < 1e-9);
    let (v, u) = auto_scale(1_500_000_000.0, ScaleLadder::Ns);
    assert_eq!(u, "s");
    assert!((v - 1.5).abs() < 1e-9);
}

/// Byte ladder uses IEC binary prefixes (×1024). 1024 B → 1
/// KiB, 1 MiB at 1024², 1 GiB at 1024³. Pin both the
/// threshold and the divisor.
#[test]
fn auto_scale_byte_iec_ladder_uses_1024() {
    assert_eq!(auto_scale(1023.0, ScaleLadder::Bytes), (1023.0, "B"));
    let (v, u) = auto_scale(1024.0, ScaleLadder::Bytes);
    assert_eq!(u, "KiB");
    assert!((v - 1.0).abs() < 1e-9);
    let (v, u) = auto_scale(1024.0 * 1024.0, ScaleLadder::Bytes);
    assert_eq!(u, "MiB");
    assert!((v - 1.0).abs() < 1e-9);
    let (v, u) = auto_scale(1024.0 * 1024.0 * 1024.0, ScaleLadder::Bytes);
    assert_eq!(u, "GiB");
    assert!((v - 1.0).abs() < 1e-9);
}

/// Ticks ladder: ticks → Kticks (×1e3) → Mticks (×1e6).
/// Decimal prefixes — clock-tick rate is host-dependent.
#[test]
fn auto_scale_ticks_ladder_uses_decimal_prefixes() {
    assert_eq!(auto_scale(999.0, ScaleLadder::Ticks), (999.0, "ticks"));
    let (v, u) = auto_scale(1_500.0, ScaleLadder::Ticks);
    assert_eq!(u, "Kticks");
    assert!((v - 1.5).abs() < 1e-9);
    let (v, u) = auto_scale(2_000_000.0, ScaleLadder::Ticks);
    assert_eq!(u, "Mticks");
    assert!((v - 2.0).abs() < 1e-9);
}

/// Unitless (large counts) ladder: "" → K → M → G. Decimal
/// SI prefixes for non-dimensional counts (wakeups,
/// migrations, etc.).
#[test]
fn auto_scale_unitless_ladder_uses_si_prefixes() {
    assert_eq!(auto_scale(999.0, ScaleLadder::Unitless), (999.0, ""));
    let (v, u) = auto_scale(1_500.0, ScaleLadder::Unitless);
    assert_eq!(u, "K");
    assert!((v - 1.5).abs() < 1e-9);
    let (v, u) = auto_scale(2_500_000.0, ScaleLadder::Unitless);
    assert_eq!(u, "M");
    assert!((v - 2.5).abs() < 1e-9);
    let (v, u) = auto_scale(3_000_000_000.0, ScaleLadder::Unitless);
    assert_eq!(u, "G");
    assert!((v - 3.0).abs() < 1e-9);
}

/// Negative values pass through scaling with sign preserved.
/// A delta cell with `-2,000,000 ns` should scale to
/// `-2.000 ms` (NOT `+2 ms` or `2 ms`).
#[test]
fn auto_scale_preserves_sign_on_negative_input() {
    let (v, u) = auto_scale(-2_000_000.0, ScaleLadder::Ns);
    assert_eq!(u, "ms");
    assert!((v - (-2.0)).abs() < 1e-9);
    let (v, u) = auto_scale(-5_000.0, ScaleLadder::Bytes);
    // -5000 < -1024 in absolute value, but value is signed.
    // |-5000| = 5000 ≥ 1024, so step to KiB.
    assert_eq!(u, "KiB");
    assert!((v - (-5000.0 / 1024.0)).abs() < 1e-9);
}

/// Phase 4: the unknown-unit pass-through behavior was
/// removed when `auto_scale` migrated from a free-form
/// `&'static str` unit tag to the closed [`ScaleLadder`]
/// enum. A registry typo can no longer slip through an
/// `other => pass-through` arm at render time — every
/// ladder is named at the type level. The corresponding
/// `auto_scale_unknown_unit_passes_through` test
/// disappeared with that change.
///
/// `format_value_cell` for a Sum aggregate with the Ns ladder:
/// values below the µs threshold render as integers; values
/// at/above the threshold render as scaled f64 with 3
/// decimals.
#[test]
fn format_value_cell_renders_sum_at_appropriate_scale() {
    // Below threshold → integer + base unit, no decimals.
    assert_eq!(
        format_value_cell(&Aggregated::Sum(50), ScaleLadder::Ns),
        "50ns"
    );
    assert_eq!(
        format_value_cell(&Aggregated::Sum(999), ScaleLadder::Ns),
        "999ns"
    );
    // At/above threshold → scaled f64 with 3 decimals.
    assert_eq!(
        format_value_cell(&Aggregated::Sum(1_500), ScaleLadder::Ns),
        "1.500µs",
    );
    assert_eq!(
        format_value_cell(&Aggregated::Sum(2_000_000), ScaleLadder::Ns),
        "2.000ms",
    );
}

/// `format_value_cell` for a Max aggregate: same scaling
/// behavior as Sum (the *_max kernel fields use ns just like
/// the *_sum fields).
#[test]
fn format_value_cell_renders_max_at_appropriate_scale() {
    assert_eq!(
        format_value_cell(&Aggregated::Max(100), ScaleLadder::Ns),
        "100ns"
    );
    assert_eq!(
        format_value_cell(&Aggregated::Max(7_500_000), ScaleLadder::Ns),
        "7.500ms",
    );
}

/// Non-numeric aggregates (Mode, OrdinalRange, Affinity) fall
/// through to the [`Aggregated`] [`fmt::Display`] impl
/// unchanged. No scaling because the values aren't scalar
/// counts.
#[test]
fn format_value_cell_passes_non_numeric_aggregates_through() {
    let m = Aggregated::mode_single("SCHED_OTHER".into(), 4, 4);
    assert_eq!(format_value_cell(&m, ScaleLadder::None), "SCHED_OTHER");
    let r = Aggregated::OrdinalRange { min: -5, max: 10 };
    assert_eq!(format_value_cell(&r, ScaleLadder::None), "-5..10");
}

/// `format_delta_cell` renders the signed delta with the
/// scaled unit. Sign is preserved (with explicit `+` for
/// positive). When no step-up was triggered AND the delta is
/// integer-valued, the cell renders as a bare signed integer
/// (no `.000` noise) to match
/// [`format_value_cell`]'s short-circuit; otherwise 3-decimal
/// precision applies.
#[test]
fn format_delta_cell_renders_signed_scaled_value() {
    // Below threshold, integer delta — short-circuit to bare
    // signed integer.
    assert_eq!(format_delta_cell(-50.0, ScaleLadder::Ns), "-50ns");
    assert_eq!(format_delta_cell(50.0, ScaleLadder::Ns), "+50ns");
    assert_eq!(format_delta_cell(0.0, ScaleLadder::Ns), "+0ns");
    // Below threshold, non-integer delta — keep 3 decimals so
    // sub-unit precision survives (rare in practice — counters
    // are u64-sourced — but possible after delta math on
    // ordinal-range midpoints).
    assert_eq!(format_delta_cell(50.5, ScaleLadder::Ns), "+50.500ns");
    // Above threshold — step up. Always 3 decimals because
    // the scale-up path can produce fractional values
    // (`2_000_001 / 1e6 = 2.000001`).
    assert_eq!(format_delta_cell(2_000_000.0, ScaleLadder::Ns), "+2.000ms");
    assert_eq!(format_delta_cell(-2_000_000.0, ScaleLadder::Ns), "-2.000ms");
}

/// `compare`'s sort order is unaffected by render-time
/// scaling: the underlying `delta_pct` and `delta` fields
/// hold the raw numeric values regardless of how cells are
/// rendered. Pin two rows whose deltas differ in scale (one
/// in ns range, one in ms-equivalent range) and verify sort
/// is by raw |delta_pct|, not by rendered string.
#[test]
fn auto_scale_does_not_affect_sort_order() {
    let mut a_small = make_thread("small", "w");
    a_small.run_time_ns = MonotonicNs(100);
    let mut a_big = make_thread("big", "w");
    a_big.run_time_ns = MonotonicNs(1_000_000);
    let mut b_small = make_thread("small", "w");
    b_small.run_time_ns = MonotonicNs(110);
    let mut b_big = make_thread("big", "w");
    b_big.run_time_ns = MonotonicNs(2_000_000);
    let diff = compare(
        &snap_with(vec![a_small, a_big]),
        &snap_with(vec![b_small, b_big]),
        &CompareOptions::default(),
    );
    // big: +100% (1M → 2M) vs small: +10% (100 → 110). Big
    // should sort first regardless of which scale the cells
    // render at.
    let run_rows: Vec<&DiffRow> = diff
        .rows
        .iter()
        .filter(|r| r.metric_name == "run_time_ns")
        .collect();
    assert_eq!(run_rows[0].group_key, "big");
    assert_eq!(run_rows[1].group_key, "small");
}

/// Integration test: a snapshot pair whose run_time_ns sums
/// fall in the ms range renders as `*ms` cells via
/// [`write_diff`]. Pins that the new auto-scale call sites
/// at the baseline / candidate / delta cells take effect end-
/// to-end.
#[test]
fn write_diff_renders_auto_scaled_cells_for_ns_metric() {
    let mut ta = make_thread("p", "w");
    ta.run_time_ns = MonotonicNs(5_000_000); // 5 ms
    let mut tb = make_thread("p", "w");
    tb.run_time_ns = MonotonicNs(8_000_000); // 8 ms
    let diff = compare(
        &snap_with(vec![ta]),
        &snap_with(vec![tb]),
        &CompareOptions::default(),
    );
    let mut out = String::new();
    write_diff(
        &mut out,
        &diff,
        Path::new("a"),
        Path::new("b"),
        GroupBy::Pcomm,
        &DisplayOptions::default(),
    )
    .unwrap();
    // Baseline cell: 5 ms with the ms unit.
    assert!(out.contains("5.000ms"), "missing baseline ms:\n{out}");
    // Candidate cell.
    assert!(out.contains("8.000ms"), "missing candidate ms:\n{out}");
    // Delta cell: +3 ms.
    assert!(out.contains("+3.000ms"), "missing delta ms:\n{out}");
}

/// Registry pin: the utime/stime clock-tick metrics carry
/// the `"ticks"` unit so they pick up the ticks ladder under
/// auto-scaling. Defends against a regression that flips
/// either entry's unit back to `""` (which would route them
/// through the unitless ladder and produce `K` / `M` /
/// `G`-prefix cells).
#[test]
fn registry_utime_stime_carry_ticks_unit() {
    let utime = CTPROF_METRICS
        .iter()
        .find(|m| m.name == "utime_clock_ticks")
        .expect("utime_clock_ticks in registry");
    let stime = CTPROF_METRICS
        .iter()
        .find(|m| m.name == "stime_clock_ticks")
        .expect("stime_clock_ticks in registry");
    assert_eq!(utime.rule.ladder(), ScaleLadder::Ticks);
    assert_eq!(stime.rule.ladder(), ScaleLadder::Ticks);
}

/// Auto-scale edge case: zero values render as bare
/// `0<unit>` across all five unit families. Pin that the
/// `abs() >= threshold` chain short-circuits to "no
/// step-up" at zero and the integer fast-path renders
/// `0ns`, `0µs`, `0B`, `0ticks`, and `0` (the empty-unit
/// case). A regression that flipped the threshold to `>`
/// (so `abs >= 0` matches and the chain over-steps to the
/// largest unit) would surface here.
#[test]
fn format_scaled_u64_zero_renders_at_base_unit_for_all_families() {
    assert_eq!(format_scaled_u64(0, ScaleLadder::Ns), "0ns");
    assert_eq!(format_scaled_u64(0, ScaleLadder::Us), "0µs");
    assert_eq!(format_scaled_u64(0, ScaleLadder::Bytes), "0B");
    assert_eq!(format_scaled_u64(0, ScaleLadder::Ticks), "0ticks");
    // Empty unit: format prints just the integer with no
    // suffix. This is the canonical unitless render path.
    assert_eq!(format_scaled_u64(0, ScaleLadder::Unitless), "0");
}

/// `format_delta_cell` on a negative µs delta auto-scales
/// AND keeps the explicit minus sign. Pin both sides:
/// magnitude is reported in seconds (`-1.500s`, not
/// `-1500000µs`), and the leading `-` survives the scale
/// step.
#[test]
fn format_delta_cell_negative_microseconds_scales_to_seconds() {
    let cell = format_delta_cell(-1_500_000.0, ScaleLadder::Us);
    assert_eq!(cell, "-1.500s");
}

/// `format_delta_cell` on a negative byte delta auto-scales
/// AND keeps the explicit minus sign. Pin the IEC binary
/// path on the negative side; the existing positive-byte
/// path is exercised by other tests but the negative-byte
/// branch was unpinned.
#[test]
fn format_delta_cell_negative_bytes_scales_to_gib() {
    let two_gib_neg = -(2.0 * 1024.0 * 1024.0 * 1024.0);
    let cell = format_delta_cell(two_gib_neg, ScaleLadder::Bytes);
    assert_eq!(cell, "-2.000GiB");
}

