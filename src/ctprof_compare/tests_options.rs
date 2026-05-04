//! Tests for `super::options` (Phase F.2 per-module redistribution).

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

/// `affine_success_ratio` = nr_wakeups_affine /
/// nr_wakeups_affine_attempts. Pin the formula on a
/// deterministic 7/10 input.
#[test]
fn derived_affine_success_ratio_formula() {
    let mut t = make_thread("p", "w");
    t.nr_wakeups_affine = MonotonicCount(7);
    t.nr_wakeups_affine_attempts = MonotonicCount(10);
    let diff = compare(
        &snap_with(vec![t.clone()]),
        &snap_with(vec![t]),
        &CompareOptions::default(),
    );
    let row = diff
        .derived_rows
        .iter()
        .find(|r| r.metric_name == "affine_success_ratio")
        .expect("affine_success_ratio row present");
    assert_eq!(row.baseline, Some(DerivedValue::Scalar(0.7)));
    assert_eq!(row.candidate, Some(DerivedValue::Scalar(0.7)));
    assert!(row.is_ratio, "affine_success_ratio is a ratio");
}

/// `avg_wait_ns` = wait_sum / wait_count. Pin formula on
/// 1000ns / 4 events = 250ns.
#[test]
fn derived_avg_wait_ns_formula() {
    let mut t = make_thread("p", "w");
    t.wait_sum = MonotonicNs(1000);
    t.wait_count = MonotonicCount(4);
    let diff = compare(
        &snap_with(vec![t.clone()]),
        &snap_with(vec![t]),
        &CompareOptions::default(),
    );
    let row = diff
        .derived_rows
        .iter()
        .find(|r| r.metric_name == "avg_wait_ns")
        .expect("avg_wait_ns row present");
    assert_eq!(row.baseline, Some(DerivedValue::Scalar(250.0)));
}

/// `voluntary_sleep_ns` is now a first-class capture field
/// — the normalization (`sum_sleep_runtime -
/// sum_block_runtime`) happens at capture time inside
/// `capture_thread_at_with_tally`, so the derived metric of
/// the same shape was removed. The compare/show path simply
/// sums `voluntary_sleep_ns` like any other Sum metric.
/// Pin a 1000ns thread renders as 1000ns through the
/// SumNs aggregation path.
#[test]
fn voluntary_sleep_ns_sums_through_registry() {
    let mut t = make_thread("p", "w");
    t.voluntary_sleep_ns = MonotonicNs(1000);
    let diff = compare(
        &snap_with(vec![t.clone()]),
        &snap_with(vec![t]),
        &CompareOptions::default(),
    );
    let row = diff
        .rows
        .iter()
        .find(|r| r.metric_name == "voluntary_sleep_ns")
        .expect("voluntary_sleep_ns row in diff");
    assert_eq!(
        row.baseline.numeric(),
        Some(1000.0),
        "voluntary_sleep_ns flows through SumNs aggregation \
         carrying the capture-side normalized value verbatim",
    );
}

/// `cpu_efficiency` = run / (run + wait). Pin on
/// 100 / (100 + 100) = 0.5.
#[test]
fn derived_cpu_efficiency_formula() {
    let mut t = make_thread("p", "w");
    t.run_time_ns = MonotonicNs(100);
    t.wait_time_ns = MonotonicNs(100);
    let diff = compare(
        &snap_with(vec![t.clone()]),
        &snap_with(vec![t]),
        &CompareOptions::default(),
    );
    let row = diff
        .derived_rows
        .iter()
        .find(|r| r.metric_name == "cpu_efficiency")
        .expect("cpu_efficiency row present");
    assert_eq!(row.baseline, Some(DerivedValue::Scalar(0.5)));
    assert!(row.is_ratio);
}

/// `avg_slice_ns` = run_time_ns / timeslices.
#[test]
fn derived_avg_slice_ns_formula() {
    let mut t = make_thread("p", "w");
    t.run_time_ns = MonotonicNs(4000);
    t.timeslices = MonotonicCount(8);
    let diff = compare(
        &snap_with(vec![t.clone()]),
        &snap_with(vec![t]),
        &CompareOptions::default(),
    );
    let row = diff
        .derived_rows
        .iter()
        .find(|r| r.metric_name == "avg_slice_ns")
        .expect("avg_slice_ns row present");
    assert_eq!(row.baseline, Some(DerivedValue::Scalar(500.0)));
}

/// `involuntary_csw_ratio` = nvcsw / (vcsw + nvcsw).
#[test]
fn derived_involuntary_csw_ratio_formula() {
    let mut t = make_thread("p", "w");
    t.voluntary_csw = MonotonicCount(75);
    t.nonvoluntary_csw = MonotonicCount(25);
    let diff = compare(
        &snap_with(vec![t.clone()]),
        &snap_with(vec![t]),
        &CompareOptions::default(),
    );
    let row = diff
        .derived_rows
        .iter()
        .find(|r| r.metric_name == "involuntary_csw_ratio")
        .expect("involuntary_csw_ratio row present");
    assert_eq!(row.baseline, Some(DerivedValue::Scalar(0.25)));
    assert!(row.is_ratio);
}

/// `disk_io_fraction` = read_bytes / rchar.
#[test]
fn derived_disk_io_fraction_formula() {
    let mut t = make_thread("p", "w");
    t.rchar = Bytes(10_000);
    t.read_bytes = Bytes(2_500);
    let diff = compare(
        &snap_with(vec![t.clone()]),
        &snap_with(vec![t]),
        &CompareOptions::default(),
    );
    let row = diff
        .derived_rows
        .iter()
        .find(|r| r.metric_name == "disk_io_fraction")
        .expect("disk_io_fraction row present");
    assert_eq!(row.baseline, Some(DerivedValue::Scalar(0.25)));
    assert!(row.is_ratio);
}

/// `live_heap_estimate` = allocated - deallocated. Pin
/// signed: 1000 alloc - 1500 dealloc = -500 (drained).
#[test]
fn derived_live_heap_estimate_signed() {
    let mut t = make_thread("p", "w");
    t.allocated_bytes = Bytes(1000);
    t.deallocated_bytes = Bytes(1500);
    let diff = compare(
        &snap_with(vec![t.clone()]),
        &snap_with(vec![t]),
        &CompareOptions::default(),
    );
    let row = diff
        .derived_rows
        .iter()
        .find(|r| r.metric_name == "live_heap_estimate")
        .expect("live_heap_estimate row present");
    assert_eq!(row.baseline, Some(DerivedValue::Scalar(-500.0)));
    assert!(!row.is_ratio, "live_heap_estimate is a B-unit, not ratio");
}

/// `avg_iowait_ns` = iowait_sum / iowait_count.
#[test]
fn derived_avg_iowait_ns_formula() {
    let mut t = make_thread("p", "w");
    t.iowait_sum = MonotonicNs(9000);
    t.iowait_count = MonotonicCount(3);
    let diff = compare(
        &snap_with(vec![t.clone()]),
        &snap_with(vec![t]),
        &CompareOptions::default(),
    );
    let row = diff
        .derived_rows
        .iter()
        .find(|r| r.metric_name == "avg_iowait_ns")
        .expect("avg_iowait_ns row present");
    assert_eq!(row.baseline, Some(DerivedValue::Scalar(3000.0)));
}

/// Every per-category `avg_<bucket>_delay_ns` row
/// computes `total / count` correctly. One thread, distinct
/// (count, total) pair per bucket so a row that mixed up
/// numerator and denominator (or pulled from the wrong
/// bucket's count) would surface as an off-by-bucket
/// equality failure here.
#[test]
fn derived_avg_delay_ns_formulas_match_manual_division() {
    let mut t = make_thread("p", "w");
    // Distinct (count, total) per bucket so a wrong-bucket
    // crosswire produces a wrong quotient rather than a
    // collision that hides the bug.
    t.cpu_delay_count = MonotonicCount(3);
    t.cpu_delay_total_ns = MonotonicNs(9_000);
    t.blkio_delay_count = MonotonicCount(4);
    t.blkio_delay_total_ns = MonotonicNs(20_000);
    t.swapin_delay_count = MonotonicCount(5);
    t.swapin_delay_total_ns = MonotonicNs(35_000);
    t.freepages_delay_count = MonotonicCount(6);
    t.freepages_delay_total_ns = MonotonicNs(54_000);
    t.thrashing_delay_count = MonotonicCount(7);
    t.thrashing_delay_total_ns = MonotonicNs(77_000);
    t.compact_delay_count = MonotonicCount(8);
    t.compact_delay_total_ns = MonotonicNs(104_000);
    t.wpcopy_delay_count = MonotonicCount(9);
    t.wpcopy_delay_total_ns = MonotonicNs(135_000);
    t.irq_delay_count = MonotonicCount(10);
    t.irq_delay_total_ns = MonotonicNs(170_000);
    let diff = compare(
        &snap_with(vec![t.clone()]),
        &snap_with(vec![t]),
        &CompareOptions::default(),
    );
    // (name, expected_avg)
    for (name, expected) in [
        ("avg_cpu_delay_ns", 3_000.0),
        ("avg_blkio_delay_ns", 5_000.0),
        ("avg_swapin_delay_ns", 7_000.0),
        ("avg_freepages_delay_ns", 9_000.0),
        ("avg_thrashing_delay_ns", 11_000.0),
        ("avg_compact_delay_ns", 13_000.0),
        ("avg_wpcopy_delay_ns", 15_000.0),
        ("avg_irq_delay_ns", 17_000.0),
    ] {
        let row = diff
            .derived_rows
            .iter()
            .find(|r| r.metric_name == name)
            .unwrap_or_else(|| panic!("{name} row present"));
        assert_eq!(
            row.baseline,
            Some(DerivedValue::Scalar(expected)),
            "{name} formula mismatch — expected {expected}",
        );
    }
}

/// `total_offcpu_delay_ns` sums every bucket and OR's
/// (swapin, thrashing) via `.max()`. Two test cases to pin
/// the .max() behavior in both directions:
///
/// (a) swapin > thrashing → swapin contributes.
/// (b) thrashing > swapin → thrashing contributes.
///
/// A regression that summed swapin + thrashing (instead of
/// max-ing) would double-count the overlap and the rollup
/// would be off by `min(swapin, thrashing)` in both cases.
#[test]
fn derived_total_offcpu_delay_ns_sums_with_max_overlap() {
    // Case (a): swapin (200) > thrashing (50). Rollup picks
    // swapin via .max().
    let mut t_a = make_thread("p", "w");
    t_a.cpu_delay_total_ns = MonotonicNs(10);
    t_a.blkio_delay_total_ns = MonotonicNs(20);
    t_a.swapin_delay_total_ns = MonotonicNs(200);
    t_a.freepages_delay_total_ns = MonotonicNs(30);
    t_a.thrashing_delay_total_ns = MonotonicNs(50);
    t_a.compact_delay_total_ns = MonotonicNs(40);
    t_a.wpcopy_delay_total_ns = MonotonicNs(60);
    t_a.irq_delay_total_ns = MonotonicNs(70);
    // Expected: 10 + 20 + 30 + 40 + 60 + 70 + max(200,50) = 430
    let diff_a = compare(
        &snap_with(vec![t_a.clone()]),
        &snap_with(vec![t_a]),
        &CompareOptions::default(),
    );
    let row_a = diff_a
        .derived_rows
        .iter()
        .find(|r| r.metric_name == "total_offcpu_delay_ns")
        .expect("total_offcpu_delay_ns row present (case a)");
    assert_eq!(
        row_a.baseline,
        Some(DerivedValue::Scalar(430.0)),
        "case (a) swapin>thrashing: expected 430, got {:?}",
        row_a.baseline,
    );

    // Case (b): thrashing (300) > swapin (75). Rollup picks
    // thrashing via .max().
    let mut t_b = make_thread("p", "w");
    t_b.cpu_delay_total_ns = MonotonicNs(10);
    t_b.blkio_delay_total_ns = MonotonicNs(20);
    t_b.swapin_delay_total_ns = MonotonicNs(75);
    t_b.freepages_delay_total_ns = MonotonicNs(30);
    t_b.thrashing_delay_total_ns = MonotonicNs(300);
    t_b.compact_delay_total_ns = MonotonicNs(40);
    t_b.wpcopy_delay_total_ns = MonotonicNs(60);
    t_b.irq_delay_total_ns = MonotonicNs(70);
    // Expected: 10 + 20 + 30 + 40 + 60 + 70 + max(75,300) = 530
    let diff_b = compare(
        &snap_with(vec![t_b.clone()]),
        &snap_with(vec![t_b]),
        &CompareOptions::default(),
    );
    let row_b = diff_b
        .derived_rows
        .iter()
        .find(|r| r.metric_name == "total_offcpu_delay_ns")
        .expect("total_offcpu_delay_ns row present (case b)");
    assert_eq!(
        row_b.baseline,
        Some(DerivedValue::Scalar(530.0)),
        "case (b) thrashing>swapin: expected 530, got {:?}",
        row_b.baseline,
    );
}

/// Division by zero in any ratio derivation produces `None`,
/// not NaN or zero. Operator-actionable as `-` in the
/// rendered cell.
#[test]
fn derived_division_by_zero_returns_none() {
    let mut t = make_thread("p", "w");
    // affine_attempts == 0 → ratio is None
    t.nr_wakeups_affine = MonotonicCount(0);
    t.nr_wakeups_affine_attempts = MonotonicCount(0);
    // wait_count == 0 → avg_wait_ns is None
    t.wait_sum = MonotonicNs(0);
    t.wait_count = MonotonicCount(0);
    // run + wait == 0 → cpu_efficiency is None
    t.run_time_ns = MonotonicNs(0);
    t.wait_time_ns = MonotonicNs(0);
    // timeslices == 0 → avg_slice_ns is None
    t.timeslices = MonotonicCount(0);
    // vcsw + nvcsw == 0 → involuntary_csw_ratio is None
    t.voluntary_csw = MonotonicCount(0);
    t.nonvoluntary_csw = MonotonicCount(0);
    // rchar == 0 → disk_io_fraction is None
    t.rchar = Bytes(0);
    t.read_bytes = Bytes(0);
    // iowait_count == 0 → avg_iowait_ns is None
    t.iowait_sum = MonotonicNs(0);
    t.iowait_count = MonotonicCount(0);
    // Every taskstats avg_*_delay_ns: count == 0 → None
    // (all default to MonotonicCount(0) / MonotonicNs(0)
    // from `..ThreadState::default()` so no explicit
    // assignment is needed; pinning the assertion below is
    // the load-bearing check). The 8 buckets follow the
    // ratio_compute pattern of avg_wait_ns / avg_iowait_ns,
    // so the same division-by-zero contract applies.
    let diff = compare(
        &snap_with(vec![t.clone()]),
        &snap_with(vec![t]),
        &CompareOptions::default(),
    );
    for name in [
        "affine_success_ratio",
        "avg_wait_ns",
        "cpu_efficiency",
        "avg_slice_ns",
        "involuntary_csw_ratio",
        "disk_io_fraction",
        "avg_iowait_ns",
        "avg_cpu_delay_ns",
        "avg_blkio_delay_ns",
        "avg_swapin_delay_ns",
        "avg_freepages_delay_ns",
        "avg_thrashing_delay_ns",
        "avg_compact_delay_ns",
        "avg_wpcopy_delay_ns",
        "avg_irq_delay_ns",
    ] {
        let row = diff
            .derived_rows
            .iter()
            .find(|r| r.metric_name == name)
            .unwrap_or_else(|| panic!("{name} row present"));
        assert!(
            row.baseline.is_none(),
            "{name} divides by zero — baseline must be None, got {:?}",
            row.baseline
        );
        assert!(
            row.delta.is_none(),
            "{name} delta must be None when inputs are zero"
        );
    }

    // total_offcpu_delay_ns is a SUM, not a quotient. With
    // every input present and all-zero, the formula evaluates
    // cleanly to 0.0 — `Some(Scalar(0.0))`, not `None`.
    // Genuine zero is meaningful here (the task accumulated
    // zero off-CPU delay across every bucket, which is a
    // real signal — e.g. an idle-since-fork bookkeeping
    // thread); collapsing it to None would conflate "no
    // delay observed" with "missing input". The
    // missing-input case is covered separately by
    // `derived_avg_delay_ns_returns_none_on_missing_input`.
    let total_row = diff
        .derived_rows
        .iter()
        .find(|r| r.metric_name == "total_offcpu_delay_ns")
        .expect("total_offcpu_delay_ns row present");
    assert_eq!(
        total_row.baseline,
        Some(DerivedValue::Scalar(0.0)),
        "total_offcpu_delay_ns with all-zero inputs must be \
         Some(0.0), not None — genuine zero is meaningful for a sum",
    );
}

/// Mode rule with a deterministic tie-break: when two
/// values share the top count, the lexicographically
/// smaller one wins. Pin the rule so the rendered output
/// is reproducible across runs.
#[test]
fn mode_rule_tie_break_is_lexicographic() {
    let mut a = make_thread("app", "w1");
    a.policy = "SCHED_FIFO".into();
    let mut b = make_thread("app", "w2");
    b.policy = "SCHED_OTHER".into();
    let v = aggregate(AggRule::Mode(|t| t.policy.clone()), &[&a, &b]);
    match v {
        Aggregated::Mode { .. } => {
            // Tie at count=1 for both values; BTreeMap
            // iteration order picks the lexicographically
            // first key, which is "SCHED_FIFO".
            assert_eq!(v.mode_value(), "SCHED_FIFO");
            assert_eq!(v.mode_count(), 1);
        }
        other => panic!("expected Mode, got {other:?}"),
    }
}

/// Affinity aggregate on an empty thread slice returns
/// `min_cpus == max_cpus == 0` and no uniform cpuset — the
/// compare engine cannot produce an empty group today, but
/// this defends against an upstream refactor that permits
/// one.
#[test]
fn affinity_aggregate_on_empty_threads_is_zero() {
    let empty: Vec<&ThreadState> = vec![];
    let v = aggregate(AggRule::Affinity(|t| t.cpu_affinity.clone()), &empty);
    match v {
        Aggregated::Affinity(s) => {
            assert_eq!(s.min_cpus, 0);
            assert_eq!(s.max_cpus, 0);
            assert!(s.uniform.is_none());
        }
        other => panic!("expected Affinity, got {other:?}"),
    }
}

/// `sort_key` inflates the zero-baseline-nonzero-candidate
/// branch (delta=Some, delta_pct=None) by 1e9 so it sorts
/// above pure zero-delta rows but still below any nonzero
/// percentage row. Two rows: one zero-delta (delta_pct=0.0),
/// one zero-baseline (delta=100, delta_pct=None) — the zero-
/// baseline row must sort FIRST.
#[test]
fn sort_key_zero_delta_rows_sink_below_nonzero() {
    // Group "calm": identical values → delta 0, pct 0.0.
    let mut a1 = make_thread("calm", "w");
    a1.run_time_ns = MonotonicNs(500);
    let mut b1 = make_thread("calm", "w");
    b1.run_time_ns = MonotonicNs(500);
    // Group "birth": baseline 0 → candidate 100 → delta 100,
    // pct undefined (None). sort_key inflates to 100 * 1e9.
    let a2 = make_thread("birth", "w");
    let mut b2 = make_thread("birth", "w");
    b2.run_time_ns = MonotonicNs(100);
    let diff = compare(
        &snap_with(vec![a1, a2]),
        &snap_with(vec![b1, b2]),
        &CompareOptions::default(),
    );
    let run_rows: Vec<&DiffRow> = diff
        .rows
        .iter()
        .filter(|r| r.metric_name == "run_time_ns")
        .collect();
    // "birth" row (zero-baseline branch of sort_key) sorts
    // ahead of "calm" (zero-delta branch).
    assert_eq!(run_rows[0].group_key, "birth");
    assert_eq!(run_rows[1].group_key, "calm");
    // Pin the exact shape each branch is meant to carry, so a
    // regression that swapped the inflation with the zero
    // arm surfaces here with a precise diagnostic rather than
    // just "wrong order".
    assert_eq!(run_rows[0].delta, Some(100.0));
    assert!(run_rows[0].delta_pct.is_none());
    assert_eq!(run_rows[1].delta, Some(0.0));
    assert_eq!(run_rows[1].delta_pct, Some(0.0));
}

/// Rows with no numeric delta (categorical Mode) sort to the
/// bottom via `sort_key`'s `f64::NEG_INFINITY` arm. Pin that a
/// nonzero numeric row sorts ahead of a Mode row whose inputs
/// differ, and that the Mode row still appears (sinks, not
/// dropped).
#[test]
fn sort_key_none_delta_rows_sink_to_bottom() {
    let mut a = make_thread("app", "w");
    a.run_time_ns = MonotonicNs(100);
    a.policy = "SCHED_OTHER".into();
    let mut b = make_thread("app", "w");
    b.run_time_ns = MonotonicNs(200);
    b.policy = "SCHED_FIFO".into();
    let diff = compare(
        &snap_with(vec![a]),
        &snap_with(vec![b]),
        &CompareOptions::default(),
    );
    // Locate the positions of run_time_ns (numeric) and
    // policy (Mode, delta=None) in the sorted rows.
    let run_idx = diff
        .rows
        .iter()
        .position(|r| r.metric_name == "run_time_ns")
        .expect("run_time_ns row");
    let policy_idx = diff
        .rows
        .iter()
        .position(|r| r.metric_name == "policy")
        .expect("policy row");
    assert!(
        run_idx < policy_idx,
        "numeric row at {run_idx} must sort above Mode row at {policy_idx}",
    );
    // Mode row really is None-delta — otherwise the ordering
    // wouldn't prove the NEG_INFINITY branch.
    assert!(diff.rows[policy_idx].delta.is_none());
}

/// `aggregate(Mode, &[])` returns `Mode { value: "", count:
/// 0, total: 0 }` via the empty-iterator tail of
/// `Modeable::mode_across`.
#[test]
fn aggregate_mode_on_empty_threads_is_empty() {
    let empty: Vec<&ThreadState> = vec![];
    let v = aggregate(AggRule::Mode(|t| t.policy.clone()), &empty);
    match v {
        Aggregated::Mode { ref tallies, total } => {
            assert!(tallies.is_empty());
            assert_eq!(v.mode_value(), "");
            assert_eq!(v.mode_count(), 0);
            assert_eq!(total, 0);
        }
        other => panic!("expected Mode, got {other:?}"),
    }
}

/// All three Mode-family arms — `Mode`, `ModeChar`,
/// `ModeBool` — route through the same `mode_aggregate`
/// helper. Drive each arm with a deterministic 3-thread
/// fixture and assert all three produce
/// `Aggregated::Mode { value, count, total }` with `total
/// == 3` and a count >= 1, pinning the helper's projection
/// shape (value+count+total triple from `mode_across`, then
/// total override from the supplied `threads.len()`).
#[test]
fn mode_aggregate_helper_dispatches_all_three_arms() {
    use crate::metric_types::CategoricalString;
    let mut t1 = make_thread("p", "w");
    let mut t2 = make_thread("p", "w");
    let mut t3 = make_thread("p", "w");
    // Mode: policy field — three distinct values, lex-tie
    // resolves to alphabetically-smallest unique winner.
    t1.policy = CategoricalString::from("SCHED_OTHER");
    t2.policy = CategoricalString::from("SCHED_OTHER");
    t3.policy = CategoricalString::from("SCHED_FIFO");
    // ModeChar: state is char.
    t1.state = 'R';
    t2.state = 'R';
    t3.state = 'S';
    // ModeBool: ext_enabled is bool.
    t1.ext_enabled = true;
    t2.ext_enabled = true;
    t3.ext_enabled = false;
    let threads: Vec<&ThreadState> = vec![&t1, &t2, &t3];

    // Mode arm: SCHED_OTHER wins 2/3.
    let v = aggregate(AggRule::Mode(|t| t.policy.clone()), &threads);
    match v {
        Aggregated::Mode { total, .. } => {
            assert_eq!(v.mode_value(), "SCHED_OTHER");
            assert_eq!(v.mode_count(), 2);
            assert_eq!(total, 3);
        }
        other => panic!("expected Mode for AggRule::Mode, got {other:?}"),
    }
    // ModeChar arm: 'R' wins 2/3 — coerced through
    // CategoricalString::to_string() via the helper.
    let v = aggregate(AggRule::ModeChar(|t| t.state), &threads);
    match v {
        Aggregated::Mode { total, .. } => {
            assert_eq!(v.mode_value(), "R");
            assert_eq!(v.mode_count(), 2);
            assert_eq!(total, 3);
        }
        other => panic!("expected Mode for AggRule::ModeChar, got {other:?}"),
    }
    // ModeBool arm: true wins 2/3 — coerced through
    // bool::Display.
    let v = aggregate(AggRule::ModeBool(|t| t.ext_enabled), &threads);
    match v {
        Aggregated::Mode { total, .. } => {
            assert_eq!(v.mode_value(), "true");
            assert_eq!(v.mode_count(), 2);
            assert_eq!(total, 3);
        }
        other => panic!("expected Mode for AggRule::ModeBool, got {other:?}"),
    }
}

/// `aggregate(SumNs, &[])` returns `Sum(0)` via the
/// identity-element seed of `Summable::sum_across`.
/// Completes empty-slice coverage across the reduction
/// families (Sum*/Max*/Range*/Mode*).
#[test]
fn aggregate_sum_on_empty_threads_is_zero() {
    let empty: Vec<&ThreadState> = vec![];
    let v = aggregate(AggRule::SumNs(|t| t.run_time_ns), &empty);
    match v {
        Aggregated::Sum(s) => assert_eq!(s, 0),
        other => panic!("expected Sum, got {other:?}"),
    }
}

/// Three threads with different `wait_max` values aggregate to
/// the GROUP MAX, not the sum. Pins the core semantic of
/// `AggRule::MaxPeak` — the kernel's `*_max` schedstats fields
/// are already per-thread maxes, and the group-level reduction
/// should surface the worst single thread's worst window, not
/// conflate a single 1s tail-latency spike with 1000 routine
/// 1ms windows.
#[test]
fn aggregate_max_picks_group_maximum_not_sum() {
    let mut a = make_thread("p", "w");
    let mut b = make_thread("p", "w");
    let mut c = make_thread("p", "w");
    a.wait_max = PeakNs(100);
    b.wait_max = PeakNs(999_999_999); // The clear group-wide tail.
    c.wait_max = PeakNs(50);
    let v = aggregate(AggRule::MaxPeak(|t| t.wait_max), &[&a, &b, &c]);
    match v {
        Aggregated::Max(m) => {
            assert_eq!(
                m, 999_999_999,
                "Max must pick the largest value, not sum (sum \
                 would be 1_000_000_149)"
            );
        }
        other => panic!("expected Max, got {other:?}"),
    }
}

/// `aggregate(MaxPeak, &[])` returns `Max(0)` via the
/// dispatch's None-to-Max(0) collapse at the call boundary —
/// `Maxable::max_across` itself returns `Option<Self>`
/// (`None` on empty input), and the `MaxPeak` arm in
/// `aggregate()` collapses `None` to `Aggregated::Max(0)` so
/// the historical empty-bucket contract on this code path is
/// preserved. Mirrors the empty-Sum contract so downstream
/// delta math works the same way for both rules when one
/// side has no threads under the join key.
#[test]
fn aggregate_max_on_empty_threads_is_zero() {
    let empty: Vec<&ThreadState> = vec![];
    let v = aggregate(AggRule::MaxPeak(|t| t.wait_max), &empty);
    match v {
        Aggregated::Max(m) => assert_eq!(m, 0),
        other => panic!("expected Max, got {other:?}"),
    }
}

/// Single-thread group: `MaxPeak` returns the single thread's
/// value verbatim. Pins that the dispatch's None-to-Max(0)
/// collapse does not override a real reading — the trait's
/// `Some(...)` arm fires for any non-empty input regardless
/// of value.
#[test]
fn aggregate_max_single_thread_returns_thread_value() {
    let mut t = make_thread("p", "w");
    t.sleep_max = PeakNs(12_345_678_901);
    let v = aggregate(AggRule::MaxPeak(|t| t.sleep_max), &[&t]);
    match v {
        Aggregated::Max(m) => assert_eq!(m, 12_345_678_901),
        other => panic!("expected Max, got {other:?}"),
    }
}

/// `--no-thread-normalize` (mirrored at API level by
/// `CompareOptions::no_thread_normalize = true`) bypasses the
/// token normalizer for thread-name grouping. Two threads
/// with names that share a normalized skeleton but differ
/// literally (e.g. `worker-0` and `worker-1`) end up in
/// SEPARATE buckets — same effect as `GroupBy::CommExact`.
#[test]
fn no_thread_normalize_uses_literal_comm() {
    let snap_a = snap_with(vec![
        make_thread("p", "worker-0"),
        make_thread("p", "worker-1"),
    ]);
    let snap_b = snap_with(vec![
        make_thread("p", "worker-0"),
        make_thread("p", "worker-1"),
    ]);
    let diff = compare(
        &snap_a,
        &snap_b,
        &CompareOptions {
            group_by: GroupBy::Comm.into(),
            cgroup_flatten: vec![],
            no_thread_normalize: true,
            no_cg_normalize: false,
            sort_by: Vec::new(),
        },
    );
    // Two distinct buckets — no collapse to `worker-{N}`.
    let group_keys: std::collections::BTreeSet<&str> =
        diff.rows.iter().map(|r| r.group_key.as_str()).collect();
    assert!(
        group_keys.contains("worker-0"),
        "literal worker-0 missing: {group_keys:?}",
    );
    assert!(
        group_keys.contains("worker-1"),
        "literal worker-1 missing: {group_keys:?}",
    );
    assert!(
        !group_keys.contains("worker-{N}"),
        "no normalized bucket under no_thread_normalize: {group_keys:?}",
    );
}

/// Cgroup grouping: spec test data verbatim. Two
/// `user@{I}.service` paths with different opaque IDs
/// collapse into one bucket; `launcher@<structured>.service`
/// paths stay as singletons; sprocket leaves (run_17 +
/// run_22) share a skeleton, gizmo leaves are singletons
/// (different words: gizmo vs sprocket).
#[test]
fn spec_cgroup_grouping_verbatim() {
    let cgroups: &[&str] = &[
        "/",
        "/boot.scope",
        "/critical.slice/emitd.service",
        "/critical.slice/remoted.service",
        "/critical.slice/launcher@foo.bar.baz.service",
        "/critical.slice/launcher@foo.qux.quux.service",
        "/critical.slice/launcher@foo.waldo.grault.service",
        "/system.slice/crond.service",
        "/system.slice/ntpd.service",
        "/system.slice/tpl.slice/launcher@foo.garply.plugh.service",
        "/system.slice/tpl.slice/launcher@foo.corge.xyzzy.service",
        "/system.slice/tpl.slice/launcher@foo.thud.fred.service",
        "/user.slice/user-0.slice/session-a1234.scope",
        "/user.slice/user-0.slice/user@0.service/boot.scope",
        "/user.slice/user-1001.slice/session-b5678.scope",
        "/user.slice/user-1001.slice/user@1001.service/boot.scope",
        // Sprocket app variants (run_17, run_22) — share skeleton.
        // Each variant has 4 leaves.
        "/apps.slice/wl-foo.slice/wl-foo-abc123def456.7890ab.alloc.slice/v2_acme.prod_widget_sprocket_run_17.400_fluxcap9000.01.zz3_650ab12cd34ef_1a2.run.yy._650ab34ef56cd_1b3.run.exec.service/helper-logs",
        "/apps.slice/wl-foo.slice/wl-foo-abc123def456.7890ab.alloc.slice/v2_acme.prod_widget_sprocket_run_17.400_fluxcap9000.01.zz3_650ab12cd34ef_1a2.run.yy._650ab34ef56cd_1b3.run.exec.service/nested/boot.scope",
        "/apps.slice/wl-foo.slice/wl-foo-abc123def456.7890ab.alloc.slice/v2_acme.prod_widget_sprocket_run_17.400_fluxcap9000.01.zz3_650ab12cd34ef_1a2.run.yy._650ab34ef56cd_1b3.run.exec.service/nested/system.slice/remoted.service",
        "/apps.slice/wl-foo.slice/wl-foo-abc123def456.7890ab.alloc.slice/v2_acme.prod_widget_sprocket_run_17.400_fluxcap9000.01.zz3_650ab12cd34ef_1a2.run.yy._650ab34ef56cd_1b3.run.exec.service/nested/system.slice/emitd.service",
        "/apps.slice/wl-foo.slice/wl-foo-def789abc012.3456cd.alloc.slice/v2_acme.prod_widget_sprocket_run_22.401_fluxcap9000.01.zz3_650ab12cd78ef_1a3.run.yy._650ab34ef90cd_1b4.run.exec.service/helper-logs",
        "/apps.slice/wl-foo.slice/wl-foo-def789abc012.3456cd.alloc.slice/v2_acme.prod_widget_sprocket_run_22.401_fluxcap9000.01.zz3_650ab12cd78ef_1a3.run.yy._650ab34ef90cd_1b4.run.exec.service/nested/boot.scope",
        "/apps.slice/wl-foo.slice/wl-foo-def789abc012.3456cd.alloc.slice/v2_acme.prod_widget_sprocket_run_22.401_fluxcap9000.01.zz3_650ab12cd78ef_1a3.run.yy._650ab34ef90cd_1b4.run.exec.service/nested/system.slice/remoted.service",
        "/apps.slice/wl-foo.slice/wl-foo-def789abc012.3456cd.alloc.slice/v2_acme.prod_widget_sprocket_run_22.401_fluxcap9000.01.zz3_650ab12cd78ef_1a3.run.yy._650ab34ef90cd_1b4.run.exec.service/nested/system.slice/emitd.service",
        // Gizmo app variant — different words (gizmo,
        // fluxcap2000, zz7), so its skeleton differs from
        // sprocket's; each gizmo leaf is a singleton.
        "/apps.slice/wl-foo.slice/wl-foo-fedcba987654.abcdef.alloc.slice/v2_acme.prod_widget_gizmo_run_5.399_fluxcap2000.03.zz7_650ab12cdaaef_2c1.run.yy._650ab34efbbcd_2c2.run.exec.service/helper-logs",
        "/apps.slice/wl-foo.slice/wl-foo-fedcba987654.abcdef.alloc.slice/v2_acme.prod_widget_gizmo_run_5.399_fluxcap2000.03.zz7_650ab12cdaaef_2c1.run.yy._650ab34efbbcd_2c2.run.exec.service/nested/boot.scope",
        "/apps.slice/wl-foo.slice/wl-foo-fedcba987654.abcdef.alloc.slice/v2_acme.prod_widget_gizmo_run_5.399_fluxcap2000.03.zz7_650ab12cdaaef_2c1.run.yy._650ab34efbbcd_2c2.run.exec.service/nested/system.slice/remoted.service",
        "/apps.slice/wl-foo.slice/wl-foo-fedcba987654.abcdef.alloc.slice/v2_acme.prod_widget_gizmo_run_5.399_fluxcap2000.03.zz7_650ab12cdaaef_2c1.run.yy._650ab34efbbcd_2c2.run.exec.service/nested/system.slice/emitd.service",
        "/apps.slice/wl-bar.slice/relay.service",
        "/apps.slice/wl-bar.slice/cache.service",
    ];

    // Build a thread per cgroup, then group by Cgroup.
    let threads: Vec<_> = cgroups
        .iter()
        .enumerate()
        .map(|(i, cg)| {
            let mut t = make_thread("p", &format!("t{i}"));
            t.cgroup = (*cg).into();
            t
        })
        .collect();
    let snap_a = snap_with(threads.clone());
    let snap_b = snap_with(threads);
    let diff = compare(
        &snap_a,
        &snap_b,
        &CompareOptions {
            group_by: GroupBy::Cgroup.into(),
            cgroup_flatten: vec![],
            no_thread_normalize: false,
            no_cg_normalize: false,
            sort_by: Vec::new(),
        },
    );

    let group_keys: std::collections::BTreeSet<String> =
        diff.rows.iter().map(|r| r.group_key.clone()).collect();

    // user session bucket: the two `session-{H}.scope` paths
    // collapse — `a1234` and `b5678` are both hex tokens.
    let user_session_skel = "/user.slice/user-{N}.slice/session-{H}.scope";
    assert!(
        group_keys.contains(user_session_skel),
        "missing user-session bucket; got {group_keys:?}",
    );

    // user@{I}.service bucket: two paths collapse via Layer 1.
    let user_service_skel = "/user.slice/user-{N}.slice/user@{I}.service/boot.scope";
    assert!(
        group_keys.contains(user_service_skel),
        "missing user@.service bucket; got {group_keys:?}",
    );

    // Singletons stay literal under `build_groups` gate.
    for singleton in &[
        "/",
        "/boot.scope",
        "/critical.slice/emitd.service",
        "/critical.slice/remoted.service",
        "/critical.slice/launcher@foo.bar.baz.service",
        "/critical.slice/launcher@foo.qux.quux.service",
        "/critical.slice/launcher@foo.waldo.grault.service",
        "/system.slice/crond.service",
        "/system.slice/ntpd.service",
        "/system.slice/tpl.slice/launcher@foo.garply.plugh.service",
        "/system.slice/tpl.slice/launcher@foo.corge.xyzzy.service",
        "/system.slice/tpl.slice/launcher@foo.thud.fred.service",
        "/apps.slice/wl-bar.slice/relay.service",
        "/apps.slice/wl-bar.slice/cache.service",
    ] {
        assert!(
            group_keys.contains(*singleton),
            "missing singleton bucket {singleton}; got {group_keys:?}",
        );
    }
}

/// `--no-cg-normalize` (mirrored at API level by
/// `CompareOptions::no_cg_normalize = true`) bypasses
/// Layer 1 / 2 / 3 entirely. Two cgroup paths that would
/// collapse under auto-normalize stay as separate literal
/// buckets. Explicit `cgroup_flatten` glob patterns still
/// apply; this flag only disables the auto-normalizer.
#[test]
fn no_cg_normalize_uses_literal_post_flatten_path() {
    let mut ta = make_thread("p", "ta");
    ta.cgroup = "/user.slice/user-0.slice/user@0.service/boot.scope".into();
    let mut tb = make_thread("p", "tb");
    tb.cgroup = "/user.slice/user-1001.slice/user@1001.service/boot.scope".into();
    let snap_a = snap_with(vec![ta]);
    let snap_b = snap_with(vec![tb]);

    // With auto-normalize ON (default): both paths collapse
    // into one bucket via Layer 1 + Layer 2.
    let diff_on = compare(
        &snap_a,
        &snap_b,
        &CompareOptions {
            group_by: GroupBy::Cgroup.into(),
            cgroup_flatten: vec![],
            no_thread_normalize: false,
            no_cg_normalize: false,
            sort_by: Vec::new(),
        },
    );
    let normalized_key = "/user.slice/user-{N}.slice/user@{I}.service/boot.scope";
    assert!(
        diff_on.rows.iter().any(|r| r.group_key == normalized_key),
        "expected normalized key {normalized_key:?} when no_cg_normalize=false",
    );

    // With no_cg_normalize ON: paths stay separate as
    // singletons, surfacing as only-baseline / only-candidate.
    let diff_off = compare(
        &snap_a,
        &snap_b,
        &CompareOptions {
            group_by: GroupBy::Cgroup.into(),
            cgroup_flatten: vec![],
            no_thread_normalize: false,
            no_cg_normalize: true,
            sort_by: Vec::new(),
        },
    );
    assert!(
        diff_off
            .only_baseline
            .contains(&"/user.slice/user-0.slice/user@0.service/boot.scope".to_string()),
        "literal baseline path missing under no_cg_normalize: only_baseline={:?}",
        diff_off.only_baseline,
    );
    assert!(
        diff_off
            .only_candidate
            .contains(&"/user.slice/user-1001.slice/user@1001.service/boot.scope".to_string()),
        "literal candidate path missing under no_cg_normalize: only_candidate={:?}",
        diff_off.only_candidate,
    );
}

/// `--no-thread-normalize` under [`GroupBy::Pcomm`] preserves
/// literal pcomm grouping — `worker-7` and `worker-15` stay in
/// distinct buckets. Mirrors
/// [`no_thread_normalize_uses_literal_comm`] for the Pcomm axis.
#[test]
fn no_thread_normalize_uses_literal_pcomm() {
    let snap_a = snap_with(vec![
        make_thread("worker-7", "t0"),
        make_thread("worker-15", "t1"),
    ]);
    let snap_b = snap_with(vec![
        make_thread("worker-7", "t0"),
        make_thread("worker-15", "t1"),
    ]);
    let diff = compare(
        &snap_a,
        &snap_b,
        &CompareOptions {
            group_by: GroupBy::Pcomm.into(),
            cgroup_flatten: vec![],
            no_thread_normalize: true,
            no_cg_normalize: false,
            sort_by: Vec::new(),
        },
    );
    let group_keys: std::collections::BTreeSet<&str> =
        diff.rows.iter().map(|r| r.group_key.as_str()).collect();
    assert!(
        group_keys.contains("worker-7"),
        "literal worker-7 missing under no_thread_normalize: {group_keys:?}",
    );
    assert!(
        group_keys.contains("worker-15"),
        "literal worker-15 missing under no_thread_normalize: {group_keys:?}",
    );
    assert!(
        !group_keys.contains("worker-{N}"),
        "no normalized bucket under no_thread_normalize on Pcomm: {group_keys:?}",
    );
}

/// Smaps render primary sort key: absolute Rss delta
/// (descending). When two processes have different absolute
/// Rss deltas, the larger-delta process must render first
/// regardless of max-Rss or alphabetical order. Pin the
/// primary-key contract by constructing a case where
/// max-Rss and alpha both prefer the WRONG order, isolating
/// the abs-delta primary sort.
#[test]
fn write_diff_smaps_abs_rss_delta_is_primary_sort_key() {
    let mut diff = CtprofDiff::default();
    // alpha_proc: small delta, BIG max-Rss.
    // The alpha-sort fallback would put alpha_proc first
    // (alphabetically before zoomed); the max-Rss tiebreak
    // would also prefer alpha_proc (240 MiB > 60 MiB max).
    // Only the abs-delta primary sort places zoomed first.
    let mut a = BTreeMap::new();
    a.insert("Rss".to_string(), 200 * 1024 * 1024);
    let mut a_b = BTreeMap::new();
    a_b.insert("Rss".to_string(), 240 * 1024 * 1024);
    // zoomed: HUGE delta (+50 MiB), small max-Rss (60 MiB).
    let mut z = BTreeMap::new();
    z.insert("Rss".to_string(), 10 * 1024 * 1024);
    let mut z_b = BTreeMap::new();
    z_b.insert("Rss".to_string(), 60 * 1024 * 1024);
    diff.smaps_rollup_a.insert("alpha_proc".into(), a);
    diff.smaps_rollup_b.insert("alpha_proc".into(), a_b);
    diff.smaps_rollup_a.insert("zoomed".into(), z);
    diff.smaps_rollup_b.insert("zoomed".into(), z_b);

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
    let smaps_at = out
        .find("## smaps_rollup")
        .expect("smaps section must render");
    let after = &out[smaps_at..];
    let zoomed_pos = after.find("zoomed").expect("zoomed key must appear");
    let alpha_pos = after
        .find("alpha_proc")
        .expect("alpha_proc key must appear");
    assert!(
        zoomed_pos < alpha_pos,
        "abs-delta primary sort must place larger-delta process (zoomed: \
         +50 MiB) ahead of smaller-delta process (alpha_proc: +40 MiB), \
         regardless of max-Rss or alpha; got zoomed@{zoomed_pos} \
         alpha_proc@{alpha_pos}",
    );
}
