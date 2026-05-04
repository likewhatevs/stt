//! Tests for `super::render` (Phase F.2 per-module redistribution).

#![allow(unused_imports)]
#![allow(clippy::field_reassign_with_default)]

use std::collections::BTreeMap;
use std::path::Path;

use super::aggregate::{format_cpu_range, merge_aggregated_into};
use super::cgroup_merge::{
    merge_cgroup_cpu, merge_cgroup_memory, merge_cgroup_pids, merge_kv_counters, merge_max_option,
    merge_memory_stat, merge_min_option, merge_psi,
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
use super::*;
use crate::ctprof::{CgroupStats, CtprofSnapshot, Psi, ThreadState};
use crate::metric_types::{
    Bytes, CategoricalString, CpuSet, MonotonicCount, MonotonicNs, OrdinalI32, PeakNs,
};
use regex::Regex;

/// Pin all four branches of `cgroup_cell` directly with the
/// dimensionless ("") unit so values render verbatim (no
/// scaling). Auto-scaling per-unit is exercised separately by
/// `cgroup_cell_renders_scaled_*`. Existing higher-level
/// tests only exercise the (Some, Some) path transitively via
/// `write_diff_cgroup_enrichment_section_for_cgroup_mode`; the
/// other three branches (baseline-only, candidate-only,
/// both-missing) are rendering-critical for the one-sided
/// enrichment row path (`all_keys` union at the enrichment
/// table site) and have no current pin.
#[test]
fn cgroup_cell_renders_all_four_branches() {
    // (Some, Some) → "a → b (+d)" where d = b - a (signed).
    assert_eq!(
        cgroup_cell(Some(10), Some(42), ScaleLadder::Unitless),
        "10 → 42 (+32)"
    );
    // Negative delta uses the signed formatter to keep the
    // sign explicit.
    assert_eq!(
        cgroup_cell(Some(50), Some(5), ScaleLadder::Unitless),
        "50 → 5 (-45)"
    );
    // (Some, None) → baseline value then en-dash placeholder.
    assert_eq!(cgroup_cell(Some(7), None, ScaleLadder::Unitless), "7 → -");
    // (None, Some) → leading en-dash placeholder.
    assert_eq!(cgroup_cell(None, Some(99), ScaleLadder::Unitless), "- → 99");
    // (None, None) → single en-dash (both sides absent).
    assert_eq!(cgroup_cell(None, None, ScaleLadder::Unitless), "-");
}

/// Pin all four branches of `format_psi_avg_cell`. Mirrors
/// the [`cgroup_cell_renders_all_four_branches`] discipline
/// for the centi-percent display path.
#[test]
fn format_psi_avg_cell_renders_all_four_branches() {
    // (Some, Some) — both halves render N.NN% with a signed
    // (+|-D.DD%) delta. 1859 centi-percent = 18.59%, 2431 =
    // 24.31%, delta = 5.72%.
    assert_eq!(
        format_psi_avg_cell(Some(1859), Some(2431)),
        "18.59% → 24.31% (+5.72%)",
    );
    // Negative delta uses an explicit minus sign.
    assert_eq!(
        format_psi_avg_cell(Some(2431), Some(1859)),
        "24.31% → 18.59% (-5.72%)",
    );
    // (Some, None) → baseline value then en-dash placeholder.
    assert_eq!(format_psi_avg_cell(Some(750), None), "7.50% → -");
    // (None, Some) → leading en-dash placeholder.
    assert_eq!(format_psi_avg_cell(None, Some(50)), "- → 0.50%");
    // (None, None) → single en-dash (both sides absent).
    assert_eq!(format_psi_avg_cell(None, None), "-");
}

/// `format_psi_avg_centi_percent` renders the kernel's
/// 2-decimal-digit fixed-point representation. Pins the
/// zero-padding boundary explicitly (`5` centi-percent must
/// render as `0.05%`, not `0.5%`) — a regression dropping
/// the zero-pad would round-trip through display only on
/// the integer-percent path.
#[test]
fn format_psi_avg_centi_percent_zero_pads_fraction() {
    assert_eq!(format_psi_avg_centi_percent(0), "0.00%");
    assert_eq!(format_psi_avg_centi_percent(5), "0.05%");
    assert_eq!(format_psi_avg_centi_percent(50), "0.50%");
    assert_eq!(format_psi_avg_centi_percent(100), "1.00%");
    assert_eq!(format_psi_avg_centi_percent(101), "1.01%");
    assert_eq!(format_psi_avg_centi_percent(10000), "100.00%");
    // Kernel EWMA rounding ceiling
    // (include/linux/sched/loadavg.h:35).
    assert_eq!(format_psi_avg_centi_percent(10099), "100.99%");
}

/// `psi_pair_has_data` returns false only when BOTH sides of
/// the pair are entirely zero. Pins the gating used in
/// `write_diff` to suppress the host-pressure block.
#[test]
fn psi_pair_has_data_returns_false_when_both_sides_zero() {
    let zero = Psi::default();
    assert!(!psi_pair_has_data(&zero, &zero));
}

#[test]
fn psi_pair_has_data_returns_true_when_one_side_nonzero() {
    let zero = Psi::default();
    let mut nonzero = Psi::default();
    nonzero.cpu.some.avg10 = 1;
    // Either order: the helper checks both sides.
    assert!(psi_pair_has_data(&zero, &nonzero));
    assert!(psi_pair_has_data(&nonzero, &zero));
}

/// Boundary: `total_usec` set to a non-zero value with every
/// avg-field still at zero counts as "has data". The avg
/// fields can lag on a low-pressure system that still
/// accumulated cumulative stall time, so a regression that
/// only checked avg10/60/300 (omitting total) would render
/// a misleading empty section here.
#[test]
fn psi_pair_has_data_detects_total_usec_only_data() {
    let zero = Psi::default();
    let mut total_only = Psi::default();
    total_only.io.full.total_usec = 1;
    assert!(psi_pair_has_data(&zero, &total_only));
    assert!(psi_pair_has_data(&total_only, &zero));
}

/// Auto-scale on the cgroup_cell µs family: a cpu_usage_usec
/// row with raw values in the millions of microseconds (i.e.
/// seconds-of-CPU range) renders with `s` / `ms` prefixes
/// rather than raw 7-digit µs counts. Each cell scales
/// independently.
#[test]
fn cgroup_cell_scales_microseconds_to_ms_or_s() {
    // 1_500_000 µs = 1.5 s; 3_000_000 µs = 3.0 s; delta 1.5 s.
    assert_eq!(
        cgroup_cell(Some(1_500_000), Some(3_000_000), ScaleLadder::Us),
        "1.500s → 3.000s (+1.500s)",
    );
    // Below the ms threshold — no step-up; integer below the
    // delta's short-circuit so the bare integer renders.
    assert_eq!(
        cgroup_cell(Some(500), Some(900), ScaleLadder::Us),
        "500µs → 900µs (+400µs)",
    );
}

/// Auto-scale on the cgroup_cell B family: a memory_current
/// row in the GiB range renders with the `GiB` prefix on each
/// scalar. Same IEC binary divisor (1024) as the per-thread
/// allocated_bytes / read_bytes columns.
#[test]
fn cgroup_cell_scales_bytes_to_iec_prefix() {
    let one_gib: u64 = 1024 * 1024 * 1024;
    let two_gib: u64 = 2 * one_gib;
    assert_eq!(
        cgroup_cell(Some(one_gib), Some(two_gib), ScaleLadder::Bytes),
        "1.000GiB → 2.000GiB (+1.000GiB)",
    );
}

/// Auto-scale on the dimensionless cgroup_cell column
/// (`nr_throttled`): large counts render with `K` / `M` /
/// `G` SI prefixes per the empty-unit ladder. Exercises each
/// step of the ladder so a regression that flips any
/// threshold (1e3 / 1e6 / 1e9) surfaces here.
#[test]
fn cgroup_cell_scales_unitless_count_to_k_m_g() {
    // K step: values in the 1e3..1e6 range pick up a `K`
    // suffix and divide by 1e3.
    assert_eq!(
        cgroup_cell(Some(1_500), Some(2_500), ScaleLadder::Unitless),
        "1.500K → 2.500K (+1.000K)",
    );
    // M step: values in the 1e6..1e9 range pick up `M` and
    // divide by 1e6.
    assert_eq!(
        cgroup_cell(Some(1_500_000), Some(2_500_000), ScaleLadder::Unitless),
        "1.500M → 2.500M (+1.000M)",
    );
    // G step: values >= 1e9 pick up `G` and divide by 1e9.
    assert_eq!(
        cgroup_cell(
            Some(1_500_000_000),
            Some(2_500_000_000),
            ScaleLadder::Unitless
        ),
        "1.500G → 2.500G (+1.000G)",
    );
}

/// Asymmetric threshold-crossing: each cell of a
/// `cgroup_cell` triple scales independently. A baseline
/// just below the µs→ms threshold renders as bare µs while
/// the candidate (just above) jumps to ms — and the delta
/// (their difference) picks its own scale based on its own
/// magnitude. Pin that the three cells don't bleed scales
/// into each other.
#[test]
fn cgroup_cell_each_cell_scales_independently() {
    // Baseline 999 µs (below 1000-µs ms threshold) →
    // renders as `999µs`. Candidate 2000 µs (above) → `2.000ms`.
    // Delta +1001 µs (above) → `+1.001ms`.
    let cell = cgroup_cell(Some(999), Some(2000), ScaleLadder::Us);
    assert_eq!(
        cell, "999µs → 2.000ms (+1.001ms)",
        "asymmetric scaling: each cell must pick its own prefix",
    );
}
