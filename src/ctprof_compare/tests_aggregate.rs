//! Tests for `super::aggregate` (Phase F.2 per-module redistribution).

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

#[test]
fn sum_aggregation_totals_across_group() {
    let mut a = make_thread("app", "w1");
    a.run_time_ns = MonotonicNs(1_000);
    let mut b = make_thread("app", "w2");
    b.run_time_ns = MonotonicNs(3_000);
    let v = aggregate(AggRule::SumNs(|t| t.run_time_ns), &[&a, &b]);
    match v {
        Aggregated::Sum(s) => assert_eq!(s, 4_000),
        other => panic!("expected Sum, got {other:?}"),
    }
}

#[test]
fn sum_saturates_on_overflow() {
    let mut a = make_thread("app", "w1");
    a.run_time_ns = MonotonicNs(u64::MAX);
    let mut b = make_thread("app", "w2");
    b.run_time_ns = MonotonicNs(5);
    let v = aggregate(AggRule::SumNs(|t| t.run_time_ns), &[&a, &b]);
    match v {
        Aggregated::Sum(s) => assert_eq!(s, u64::MAX),
        other => panic!("expected Sum, got {other:?}"),
    }
}

#[test]
fn ordinal_range_picks_extremes() {
    let mut a = make_thread("app", "w1");
    a.nice = OrdinalI32(-5);
    let mut b = make_thread("app", "w2");
    b.nice = OrdinalI32(10);
    let v = aggregate(AggRule::RangeI32(|t| t.nice), &[&a, &b]);
    match v {
        Aggregated::OrdinalRange { min, max } => {
            assert_eq!(min, -5);
            assert_eq!(max, 10);
        }
        other => panic!("expected OrdinalRange, got {other:?}"),
    }
}

#[test]
fn mode_aggregation_picks_most_frequent() {
    let mut a = make_thread("app", "w1");
    a.policy = "SCHED_OTHER".into();
    let mut b = make_thread("app", "w2");
    b.policy = "SCHED_OTHER".into();
    let mut c = make_thread("app", "w3");
    c.policy = "SCHED_FIFO".into();
    let v = aggregate(AggRule::Mode(|t| t.policy.clone()), &[&a, &b, &c]);
    match v {
        Aggregated::Mode { ref tallies, total } => {
            assert_eq!(v.mode_value(), "SCHED_OTHER");
            assert_eq!(v.mode_count(), 2);
            assert_eq!(total, 3);
            // Tally pin: SCHED_OTHER counted twice,
            // SCHED_FIFO counted once.
            assert_eq!(tallies.get("SCHED_OTHER").copied(), Some(2));
            assert_eq!(tallies.get("SCHED_FIFO").copied(), Some(1));
        }
        other => panic!("expected Mode, got {other:?}"),
    }
}

#[test]
fn affinity_uniform_preserves_cpuset() {
    let a = make_thread("app", "w1");
    let b = make_thread("app", "w2");
    let v = aggregate(AggRule::Affinity(|t| t.cpu_affinity.clone()), &[&a, &b]);
    match v {
        Aggregated::Affinity(s) => {
            assert_eq!(s.min_cpus, 4);
            assert_eq!(s.max_cpus, 4);
            assert_eq!(s.uniform, Some(vec![0, 1, 2, 3]));
        }
        other => panic!("expected Affinity, got {other:?}"),
    }
}

#[test]
fn affinity_heterogeneous_drops_uniform() {
    let a = make_thread("app", "w1");
    let mut b = make_thread("app", "w2");
    b.cpu_affinity = CpuSet(vec![4, 5]);
    let v = aggregate(AggRule::Affinity(|t| t.cpu_affinity.clone()), &[&a, &b]);
    match v {
        Aggregated::Affinity(s) => {
            assert_eq!(s.min_cpus, 2);
            assert_eq!(s.max_cpus, 4);
            assert!(s.uniform.is_none());
        }
        other => panic!("expected Affinity, got {other:?}"),
    }
}

#[test]
fn format_cpu_range_collapses_contiguous_runs() {
    assert_eq!(format_cpu_range(&[0, 1, 2, 3]), "0-3");
    assert_eq!(format_cpu_range(&[0, 1, 4, 5, 7]), "0-1,4-5,7");
    assert_eq!(format_cpu_range(&[3]), "3");
    assert_eq!(format_cpu_range(&[]), "");
}

/// Ordinal range collapses `min == max` to a single number
/// in display. Defends against `nice=0` single-thread
/// groups rendering as `0..0`.
#[test]
fn ordinal_display_collapses_degenerate_range() {
    let r = Aggregated::OrdinalRange { min: 0, max: 0 };
    assert_eq!(r.to_string(), "0");
    let r = Aggregated::OrdinalRange { min: -5, max: 10 };
    assert_eq!(r.to_string(), "-5..10");
}

/// Mode display omits the minority ratio when the mode is
/// unanimous (count == total). Keeps the table compact for
/// homogeneous groups.
#[test]
fn mode_display_hides_ratio_when_unanimous() {
    let m = Aggregated::mode_single("SCHED_OTHER".into(), 4, 4);
    assert_eq!(m.to_string(), "SCHED_OTHER");
    let m = Aggregated::mode_single("SCHED_OTHER".into(), 3, 5);
    assert_eq!(m.to_string(), "SCHED_OTHER (3/5)");
}

/// `aggregate(OrdinalRange, &[])` returns `OrdinalRange {
/// min: 0, max: 0 }` via the `unwrap_or(0)` in the first-value
/// init. Sibling to the empty-affinity test.
#[test]
fn aggregate_ordinal_range_on_empty_threads_is_zero() {
    let empty: Vec<&ThreadState> = vec![];
    let v = aggregate(AggRule::RangeI32(|t| t.nice), &empty);
    match v {
        Aggregated::OrdinalRange { min, max } => {
            assert_eq!(min, 0);
            assert_eq!(max, 0);
        }
        other => panic!("expected OrdinalRange, got {other:?}"),
    }
}

/// `Aggregated::Max` projects to f64 via `numeric()` so the
/// delta-math pipeline in `build_row` handles Max rows the
/// same way it handles Sum rows. Display renders the bare u64
/// (same shape as Sum). Pins both the numeric and Display
/// arms so a regression that dropped one of them surfaces.
#[test]
fn aggregated_max_numeric_and_display() {
    let m = Aggregated::Max(7_500_000);
    assert_eq!(m.numeric(), Some(7_500_000.0));
    assert_eq!(format!("{m}"), "7500000");
}

/// `Aggregated::numeric` returns `None` for `Mode` — a
/// policy name has no scalar projection. Pin the contract
/// directly rather than via the diff pipeline because the
/// pipeline only reads numeric through `build_row`'s `(a.numeric(),
/// b.numeric())` pair and a regression could silently flip the
/// return to `Some(0.0)` without any currently-visible symptom.
#[test]
fn numeric_returns_none_for_mode() {
    let m = Aggregated::mode_single("SCHED_OTHER".into(), 4, 4);
    assert!(m.numeric().is_none());
}

/// `Aggregated::numeric` for a heterogeneous `Affinity`
/// returns `(min_cpus + max_cpus) / 2.0` — the midpoint
/// projection. Existing affinity tests only exercise uniform
/// cpusets where `min == max`, so the arithmetic path is
/// unpinned.
#[test]
fn numeric_returns_midpoint_for_affinity_heterogeneous() {
    let a = Aggregated::Affinity(AffinitySummary {
        min_cpus: 2,
        max_cpus: 8,
        uniform: None,
    });
    assert_eq!(a.numeric(), Some(5.0));
    // Single-element (uniform) heterogeneous check is the
    // degenerate case where the midpoint equals either bound.
    let b = Aggregated::Affinity(AffinitySummary {
        min_cpus: 4,
        max_cpus: 4,
        uniform: None,
    });
    assert_eq!(b.numeric(), Some(4.0));
}

/// Uniform non-contiguous cpuset `[0, 2]` renders as
/// `"2 cpus (0,2)"` — exercises the comma-separated branch of
/// `format_cpu_range` from the Affinity display impl. Existing
/// uniform test uses `[0,1,2,3]` which collapses to a single
/// range token.
#[test]
fn affinity_display_uniform_noncontiguous_renders_comma_separated() {
    let a = Aggregated::Affinity(AffinitySummary {
        min_cpus: 2,
        max_cpus: 2,
        uniform: Some(vec![0, 2]),
    });
    assert_eq!(a.to_string(), "2 cpus (0,2)");
}

/// Heterogeneous affinity where `min_cpus == max_cpus` (every
/// thread has the same cpuset SIZE but different SETS) renders
/// as `"N cpus (mixed)"` — pins the specific branch in the
/// display impl. Current heterogeneous test has min != max so
/// this branch was unpinned.
#[test]
fn affinity_display_heterogeneous_same_count_renders_mixed() {
    let a = Aggregated::Affinity(AffinitySummary {
        min_cpus: 3,
        max_cpus: 3,
        uniform: None,
    });
    assert_eq!(a.to_string(), "3 cpus (mixed)");
}

/// P1: N:1 merge unions Aggregated::OrdinalRange bounds.
/// Fudge two candidates with disjoint nice ranges; the
/// merged range spans both.
#[test]
fn fudge_n_to_one_merge_unions_ordinal_range() {
    let threads_a = fudge_threads_with("/svc", 10, |t| {
        t.nice = OrdinalI32(0);
    });
    let mut threads_b = fudge_threads_with("/svc-a", 10, |t| {
        t.nice = OrdinalI32(-5);
    });
    threads_b.extend(fudge_threads_with("/svc-b", 10, |t| {
        t.nice = OrdinalI32(5);
    }));
    let diff = fudge_compare(&snap_with(threads_a), &snap_with(threads_b));
    let r = diff
        .rows
        .iter()
        .find(|r| r.metric_name == "nice" && r.group_key.contains("alpha"))
        .expect("nice alpha row must surface");
    match &r.candidate {
        Aggregated::OrdinalRange { min, max } => {
            assert_eq!(*min, -5, "merged candidate nice range min must be -5");
            assert_eq!(*max, 5, "merged candidate nice range max must be 5");
        }
        other => panic!("expected OrdinalRange for nice; got {other:?}"),
    }
}
