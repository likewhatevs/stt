//! Tests for `super::diff_types` (Phase F.2 per-module redistribution).

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

/// Single-thread group: registry emits exactly one row per
/// registered metric. Defends against a future "skip if
/// only one thread" short-circuit sneaking into
/// `aggregate`.
#[test]
fn single_thread_group_yields_one_row_per_metric() {
    let a = make_thread("solo", "t");
    let mut b = make_thread("solo", "t");
    b.run_time_ns = MonotonicNs(1);
    let diff = compare(
        &snap_with(vec![a]),
        &snap_with(vec![b]),
        &CompareOptions::default(),
    );
    let solo_rows: Vec<&DiffRow> = diff.rows.iter().filter(|r| r.group_key == "solo").collect();
    assert_eq!(solo_rows.len(), CTPROF_METRICS.len());
}

/// Reference test data: 14 multi-member buckets and 13
/// singletons covering every classifier rule. Every input
/// thread name produces the exact expected normalized form;
/// every bucket has the exact expected member count.
#[test]
fn spec_thread_grouping_verbatim() {
    let inputs: &[&str] = &[
        // Bucket 1: whirly-gig-{N} (4 members)
        "whirly-gig-0",
        "whirly-gig-1",
        "whirly-gig-2",
        "whirly-gig-15",
        // Bucket 2: plonk_zap_{N} (3)
        "plonk_zap_0",
        "plonk_zap_1",
        "plonk_zap_7",
        // Bucket 3: ksoftirqd/{N} (4)
        "ksoftirqd/0",
        "ksoftirqd/1",
        "ksoftirqd/2",
        "ksoftirqd/99",
        // Bucket 4: kworker/{N}:{N} (4) — bare bound
        "kworker/0:0",
        "kworker/0:1",
        "kworker/1:0",
        "kworker/3:2",
        // Bucket 5: kworker/{N}:{N}-wq_reclaim (3)
        "kworker/0:0-wq_reclaim",
        "kworker/1:0-wq_reclaim",
        "kworker/47:2-wq_reclaim",
        // Bucket 6: kworker/u{N}:{N} (3) — bare unbound
        "kworker/u8:3",
        "kworker/u8:7",
        "kworker/u16:0",
        // Bucket 6b: kworker/{N}:{N}H-wq_prio (3) —
        // high-priority bound workers; rule 4 normalizes
        // `<id>H` tokens.
        "kworker/0:1H-wq_prio",
        "kworker/1:0H-wq_prio",
        "kworker/2:1H-wq_prio",
        // Bucket 7: FooBar{N} (4)
        "FooBar0",
        "FooBar1",
        "FooBar2",
        "FooBar175",
        // Bucket 8: BazQux{N} (3)
        "BazQux0",
        "BazQux1",
        "BazQux42",
        // Bucket 9: wonk{N} (3)
        "wonk0",
        "wonk1",
        "wonk9",
        // Bucket 10: Grommet.Z{N} (3)
        "Grommet.Z0",
        "Grommet.Z1",
        "Grommet.Z999",
        // Bucket 11: fizz-buzz-wham{N} (3)
        "fizz-buzz-wham0",
        "fizz-buzz-wham1",
        "fizz-buzz-wham7",
        // Bucket 12: rcu_exp_par_gp_kthread_worker/{N} (2)
        "rcu_exp_par_gp_kthread_worker/0",
        "rcu_exp_par_gp_kthread_worker/1",
        // Bucket 13: migration/{N} (2)
        "migration/0",
        "migration/1",
        // Singletons:
        "bloop-tangler",
        "narf-bonker",
        "spork-wrangler",
        "hamster",
        "zilch",
        "gadget-v2",
        "thingo-r2",
        "cpu0",
        "blip0",
        "snorf0",
        "ptp0",
        "BPF_CUBIC",
        "AUTO_FLOWLABEL",
    ];

    // Per-input expected pattern_key.
    let expected_keys: &[(&str, &str)] = &[
        ("whirly-gig-0", "whirly-gig-{N}"),
        ("whirly-gig-1", "whirly-gig-{N}"),
        ("whirly-gig-2", "whirly-gig-{N}"),
        ("whirly-gig-15", "whirly-gig-{N}"),
        ("plonk_zap_0", "plonk_zap_{N}"),
        ("plonk_zap_1", "plonk_zap_{N}"),
        ("plonk_zap_7", "plonk_zap_{N}"),
        ("ksoftirqd/0", "ksoftirqd/{N}"),
        ("ksoftirqd/1", "ksoftirqd/{N}"),
        ("ksoftirqd/2", "ksoftirqd/{N}"),
        ("ksoftirqd/99", "ksoftirqd/{N}"),
        ("kworker/0:0", "kworker/{N}:{N}"),
        ("kworker/0:1", "kworker/{N}:{N}"),
        ("kworker/1:0", "kworker/{N}:{N}"),
        ("kworker/3:2", "kworker/{N}:{N}"),
        ("kworker/0:0-wq_reclaim", "kworker/{N}:{N}-wq_reclaim"),
        ("kworker/1:0-wq_reclaim", "kworker/{N}:{N}-wq_reclaim"),
        ("kworker/47:2-wq_reclaim", "kworker/{N}:{N}-wq_reclaim"),
        ("kworker/u8:3", "kworker/u{N}:{N}"),
        ("kworker/u8:7", "kworker/u{N}:{N}"),
        ("kworker/u16:0", "kworker/u{N}:{N}"),
        ("kworker/0:1H-wq_prio", "kworker/{N}:{N}H-wq_prio"),
        ("kworker/1:0H-wq_prio", "kworker/{N}:{N}H-wq_prio"),
        ("kworker/2:1H-wq_prio", "kworker/{N}:{N}H-wq_prio"),
        ("FooBar0", "FooBar{N}"),
        ("FooBar1", "FooBar{N}"),
        ("FooBar2", "FooBar{N}"),
        ("FooBar175", "FooBar{N}"),
        ("BazQux0", "BazQux{N}"),
        ("BazQux1", "BazQux{N}"),
        ("BazQux42", "BazQux{N}"),
        ("wonk0", "wonk{N}"),
        ("wonk1", "wonk{N}"),
        ("wonk9", "wonk{N}"),
        ("Grommet.Z0", "Grommet.Z{N}"),
        ("Grommet.Z1", "Grommet.Z{N}"),
        ("Grommet.Z999", "Grommet.Z{N}"),
        ("fizz-buzz-wham0", "fizz-buzz-wham{N}"),
        ("fizz-buzz-wham1", "fizz-buzz-wham{N}"),
        ("fizz-buzz-wham7", "fizz-buzz-wham{N}"),
        (
            "rcu_exp_par_gp_kthread_worker/0",
            "rcu_exp_par_gp_kthread_worker/{N}",
        ),
        (
            "rcu_exp_par_gp_kthread_worker/1",
            "rcu_exp_par_gp_kthread_worker/{N}",
        ),
        ("migration/0", "migration/{N}"),
        ("migration/1", "migration/{N}"),
        // Singletons (skeleton form per algorithm).
        ("bloop-tangler", "bloop-tangler"),
        ("narf-bonker", "narf-bonker"),
        ("spork-wrangler", "spork-wrangler"),
        ("hamster", "hamster"),
        ("zilch", "zilch"),
        ("gadget-v2", "gadget-v{N}"),
        ("thingo-r2", "thingo-r{N}"),
        ("cpu0", "cpu{N}"),
        ("blip0", "blip{N}"),
        ("snorf0", "snorf{N}"),
        ("ptp0", "ptp{N}"),
        ("BPF_CUBIC", "BPF_CUBIC"),
        ("AUTO_FLOWLABEL", "AUTO_FLOWLABEL"),
    ];

    for (input, expected) in expected_keys {
        assert_eq!(
            pattern_key(input),
            *expected,
            "pattern_key({input:?}) skeleton mismatch",
        );
    }

    // Build groups via `build_groups` and assert bucket
    // membership counts. Singletons revert to the literal
    // input under `build_groups`'s gate, so the bucket key
    // for a singleton is the input string, not the
    // skeleton.
    let threads: Vec<_> = inputs.iter().map(|c| make_thread("p", c)).collect();
    let snap = snap_with(threads);
    let groups = build_groups(&snap, GroupBy::Comm, &[], None, None, false);

    let expected_buckets: &[(&str, usize)] = &[
        ("whirly-gig-{N}", 4),
        ("plonk_zap_{N}", 3),
        ("ksoftirqd/{N}", 4),
        ("kworker/{N}:{N}", 4),
        ("kworker/{N}:{N}-wq_reclaim", 3),
        ("kworker/u{N}:{N}", 3),
        ("kworker/{N}:{N}H-wq_prio", 3),
        ("FooBar{N}", 4),
        ("BazQux{N}", 3),
        ("wonk{N}", 3),
        ("Grommet.Z{N}", 3),
        ("fizz-buzz-wham{N}", 3),
        ("rcu_exp_par_gp_kthread_worker/{N}", 2),
        ("migration/{N}", 2),
    ];
    for (key, count) in expected_buckets {
        let g = groups
            .get(*key)
            .unwrap_or_else(|| panic!("missing bucket {key:?}"));
        assert_eq!(
            g.thread_count, *count,
            "bucket {key:?} expected {count} members, got {}",
            g.thread_count,
        );
    }

    // Singletons keep their literal input as the bucket key
    // (the gate at `build_groups` reverts singletons to the
    // input).
    for singleton in &[
        "bloop-tangler",
        "narf-bonker",
        "spork-wrangler",
        "hamster",
        "zilch",
        "gadget-v2",
        "thingo-r2",
        "cpu0",
        "blip0",
        "snorf0",
        "ptp0",
        "BPF_CUBIC",
        "AUTO_FLOWLABEL",
    ] {
        let g = groups
            .get(*singleton)
            .unwrap_or_else(|| panic!("missing singleton bucket {singleton:?}"));
        assert_eq!(
            g.thread_count, 1,
            "singleton {singleton:?} should have 1 member",
        );
    }

    // Total bucket count: 14 multi-member + 13 singletons.
    assert_eq!(groups.len(), 14 + 13, "expected 27 buckets total");
}

/// Multi-key sort: groups rank by the requested metrics'
/// deltas in tuple order. Big regression on the FIRST key
/// dominates regardless of the second key.
///
/// Exercises `sort_diff_rows_by_keys` directly on synthetic
/// `DiffRow` values rather than driving through `compare()`
/// — the function under test is the sort, not the diff
/// pipeline; building the diff via `compare(empty, full)`
/// would route every group into `only_baseline` /
/// `only_candidate` rather than producing the matched-group
/// rows the sort consumes.
#[test]
fn sort_diff_rows_by_keys_ranks_by_first_key_first() {
    // Build synthetic rows: 3 groups × 2 metrics = 6 rows.
    let mk_row = |group: &str, metric: &'static str, delta: f64| DiffRow {
        group_key: group.into(),
        thread_count_a: 1,
        thread_count_b: 1,
        metric_name: metric,
        metric_ladder: ScaleLadder::None,
        baseline: Aggregated::Sum(0),
        candidate: Aggregated::Sum(0),
        delta: Some(delta),
        delta_pct: None,
        display_key: group.into(),
        uptime_pct: None,
        sort_by_cell: None,
        sort_by_delta: None,
    };
    let mut rows = vec![
        mk_row("A", "run_time_ns", 1000.0),
        mk_row("A", "wait_sum", 100.0),
        mk_row("B", "run_time_ns", 100.0),
        mk_row("B", "wait_sum", 1000.0),
        mk_row("C", "run_time_ns", 50.0),
        mk_row("C", "wait_sum", 50.0),
    ];
    sort_diff_rows_by_keys(
        &mut rows,
        &mut Vec::new(),
        &[SortKey {
            metric: "run_time_ns",
            descending: true,
        }],
    );
    let groups_in_order: Vec<&str> = rows.iter().map(|r| r.group_key.as_str()).collect();
    // A has run_time_ns 1000 → first. B has 100 → second. C has 50 → third.
    // Each group's two rows cluster together in registry
    // order (run_time_ns before wait_sum).
    assert_eq!(
        groups_in_order,
        vec!["A", "A", "B", "B", "C", "C"],
        "groups should rank by run_time_ns delta desc",
    );
    // Within each group: run_time_ns row comes first
    // (registry index lower than wait_sum).
    let metrics_first_two: Vec<&str> = rows.iter().take(2).map(|r| r.metric_name).collect();
    assert_eq!(metrics_first_two, vec!["run_time_ns", "wait_sum"]);
}

/// Multi-key sort tie-break: when the first key value is
/// equal across groups, the second key disambiguates. Two
/// groups with the same run_time_ns delta but different
/// wait_sum deltas: the one with the larger wait_sum delta
/// sorts first (under desc,desc).
#[test]
fn sort_diff_rows_by_keys_breaks_ties_with_second_key() {
    let mk_row = |group: &str, metric: &'static str, delta: f64| DiffRow {
        group_key: group.into(),
        thread_count_a: 1,
        thread_count_b: 1,
        metric_name: metric,
        metric_ladder: ScaleLadder::None,
        baseline: Aggregated::Sum(0),
        candidate: Aggregated::Sum(0),
        delta: Some(delta),
        delta_pct: None,
        display_key: group.into(),
        uptime_pct: None,
        sort_by_cell: None,
        sort_by_delta: None,
    };
    let mut rows = vec![
        // A and B tie on run_time_ns (both 500). Use wait_sum
        // to break: A.wait_sum delta is 100, B.wait_sum delta
        // is 200. Under desc,desc → B first.
        mk_row("A", "run_time_ns", 500.0),
        mk_row("A", "wait_sum", 100.0),
        mk_row("B", "run_time_ns", 500.0),
        mk_row("B", "wait_sum", 200.0),
    ];
    sort_diff_rows_by_keys(
        &mut rows,
        &mut Vec::new(),
        &[
            SortKey {
                metric: "run_time_ns",
                descending: true,
            },
            SortKey {
                metric: "wait_sum",
                descending: true,
            },
        ],
    );
    let groups_in_order: Vec<&str> = rows.iter().map(|r| r.group_key.as_str()).collect();
    assert_eq!(groups_in_order, vec!["B", "B", "A", "A"]);
}

/// Ascending direction reverses the sort. Group with the
/// SMALLEST delta should sort first under `:asc`.
#[test]
fn sort_diff_rows_by_keys_respects_ascending_direction() {
    let mk_row = |group: &str, metric: &'static str, delta: f64| DiffRow {
        group_key: group.into(),
        thread_count_a: 1,
        thread_count_b: 1,
        metric_name: metric,
        metric_ladder: ScaleLadder::None,
        baseline: Aggregated::Sum(0),
        candidate: Aggregated::Sum(0),
        delta: Some(delta),
        delta_pct: None,
        display_key: group.into(),
        uptime_pct: None,
        sort_by_cell: None,
        sort_by_delta: None,
    };
    let mut rows = vec![
        mk_row("A", "run_time_ns", 1000.0),
        mk_row("B", "run_time_ns", 100.0),
        mk_row("C", "run_time_ns", 500.0),
    ];
    sort_diff_rows_by_keys(
        &mut rows,
        &mut Vec::new(),
        &[SortKey {
            metric: "run_time_ns",
            descending: false, // asc
        }],
    );
    let groups_in_order: Vec<&str> = rows.iter().map(|r| r.group_key.as_str()).collect();
    // B (100) < C (500) < A (1000) under asc.
    assert_eq!(groups_in_order, vec!["B", "C", "A"]);
}

/// Final tie-break: when every sort-key value matches across
/// groups, `sort_diff_rows_by_keys` falls through to ascending
/// `group_key` ordering for deterministic output. Pins the
/// last branch in the comparator (`a.cmp(b)`) — without it,
/// equal-delta groups would emerge in BTreeMap-iteration order
/// dependent on hash, which would produce flaky test output.
#[test]
fn sort_diff_rows_by_keys_falls_back_to_ascending_group_key_on_full_tie() {
    let mk_row = |group: &str, metric: &'static str, delta: f64| DiffRow {
        group_key: group.into(),
        thread_count_a: 1,
        thread_count_b: 1,
        metric_name: metric,
        metric_ladder: ScaleLadder::None,
        baseline: Aggregated::Sum(0),
        candidate: Aggregated::Sum(0),
        delta: Some(delta),
        delta_pct: None,
        display_key: group.into(),
        uptime_pct: None,
        sort_by_cell: None,
        sort_by_delta: None,
    };
    // Three groups with IDENTICAL deltas — only the
    // group_key tie-break can deterministically order them.
    // Insert in reverse-alphabetical order so the test fails
    // if the tie-break is dropped (BTreeMap iteration would
    // already produce ascending — distinguishable only via
    // explicit reverse-input ordering).
    let mut rows = vec![
        mk_row("charlie", "run_time_ns", 100.0),
        mk_row("bravo", "run_time_ns", 100.0),
        mk_row("alpha", "run_time_ns", 100.0),
    ];
    sort_diff_rows_by_keys(
        &mut rows,
        &mut Vec::new(),
        &[SortKey {
            metric: "run_time_ns",
            descending: true,
        }],
    );
    let order: Vec<&str> = rows.iter().map(|r| r.group_key.as_str()).collect();
    assert_eq!(
        order,
        vec!["alpha", "bravo", "charlie"],
        "full sort-key tie must fall back to ascending group_key",
    );
}

/// Categorical-only group: every row's `delta` is `None`
/// (the group's metric is Mode and delta math doesn't
/// apply), but the group still appears in `rows`.
/// `sort_diff_rows_by_keys` must surface the group with
/// the missing-metric fallback applied — no panic, no row
/// dropped. This guards the second loop in the function
/// that adds groups present in `rows` but absent from
/// `group_metrics`.
#[test]
fn sort_diff_rows_by_keys_categorical_only_group_does_not_panic() {
    let mk_row = |group: &str, metric: &'static str| DiffRow {
        group_key: group.into(),
        thread_count_a: 1,
        thread_count_b: 1,
        metric_name: metric,
        metric_ladder: ScaleLadder::None,
        baseline: Aggregated::mode_single("SCHED_OTHER".into(), 1, 1),
        candidate: Aggregated::mode_single("SCHED_OTHER".into(), 1, 1),
        // `Mode` rows carry `delta: None` because mode
        // metrics have no scalar projection — see
        // `Aggregated::numeric()`.
        delta: None,
        delta_pct: None,
        display_key: group.into(),
        uptime_pct: None,
        sort_by_cell: None,
        sort_by_delta: None,
    };
    let mut rows = vec![mk_row("alpha", "policy"), mk_row("bravo", "policy")];
    // Sort by run_time_ns — neither group has it, both fall
    // through to the missing-metric fallback. Final tie-break
    // (`a.cmp(b)`) breaks the tie ascending.
    sort_diff_rows_by_keys(
        &mut rows,
        &mut Vec::new(),
        &[SortKey {
            metric: "run_time_ns",
            descending: true,
        }],
    );
    let order: Vec<&str> = rows.iter().map(|r| r.group_key.as_str()).collect();
    assert_eq!(
        order,
        vec!["alpha", "bravo"],
        "categorical-only groups must survive the sort and fall to ascending group_key",
    );
}

/// Within a group, rows appear in `CTPROF_METRICS`
/// registry order regardless of input order or sort spec.
/// Pins the documented "rows within a group keep registry
/// order" contract — a regression that ordered metric rows
/// by `metric_name` lexicographically (or by sort_key
/// position) would produce non-deterministic per-bucket
/// layouts.
#[test]
fn sort_diff_rows_by_keys_within_group_uses_registry_order() {
    let mk_row = |group: &str, metric: &'static str, delta: f64| DiffRow {
        group_key: group.into(),
        thread_count_a: 1,
        thread_count_b: 1,
        metric_name: metric,
        metric_ladder: ScaleLadder::None,
        baseline: Aggregated::Sum(0),
        candidate: Aggregated::Sum(0),
        delta: Some(delta),
        delta_pct: None,
        display_key: group.into(),
        uptime_pct: None,
        sort_by_cell: None,
        sort_by_delta: None,
    };
    // Use four metrics from the scheduling block in their
    // registry order: run_time_ns (idx 6), wait_time_ns (7),
    // timeslices (8), nr_wakeups (11). Insert in
    // REVERSE-registry order so a regression that orders by
    // input/sort-spec/lexicographic would surface as a
    // visibly wrong metric_order assertion.
    let mut rows = vec![
        mk_row("alpha", "nr_wakeups", 4.0),
        mk_row("alpha", "timeslices", 3.0),
        mk_row("alpha", "wait_time_ns", 999.0),
        mk_row("alpha", "run_time_ns", 1.0),
    ];
    sort_diff_rows_by_keys(
        &mut rows,
        &mut Vec::new(),
        &[SortKey {
            // Sort by wait_time_ns to verify the metric
            // rows still emerge in REGISTRY order, not
            // sort-spec order (which would put wait_time_ns
            // first).
            metric: "wait_time_ns",
            descending: true,
        }],
    );
    let metric_order: Vec<&str> = rows.iter().map(|r| r.metric_name).collect();
    assert_eq!(
        metric_order,
        vec!["run_time_ns", "wait_time_ns", "timeslices", "nr_wakeups"],
        "within-group order must be registry, not sort-spec, order",
    );
}

/// NaN-safe partial_cmp: a `delta` that's NaN must not
/// panic the sort. `partial_cmp` returns `None` for NaN,
/// which the comparator maps to `Ordering::Equal` so the
/// remaining keys (or the group_key tie-break) decide. Pin
/// that the function survives the NaN input — without the
/// `unwrap_or(Equal)` in both arms, the sort would panic on
/// the implicit `unwrap()` of an arithmetic NaN result.
#[test]
fn sort_diff_rows_by_keys_nan_delta_does_not_panic() {
    let mk_row = |group: &str, metric: &'static str, delta: f64| DiffRow {
        group_key: group.into(),
        thread_count_a: 1,
        thread_count_b: 1,
        metric_name: metric,
        metric_ladder: ScaleLadder::None,
        baseline: Aggregated::Sum(0),
        candidate: Aggregated::Sum(0),
        delta: Some(delta),
        delta_pct: None,
        display_key: group.into(),
        uptime_pct: None,
        sort_by_cell: None,
        sort_by_delta: None,
    };
    let mut rows = vec![
        mk_row("alpha", "run_time_ns", f64::NAN),
        mk_row("bravo", "run_time_ns", 100.0),
        mk_row("charlie", "run_time_ns", f64::NAN),
    ];
    // The function call must not panic; output ordering is
    // unspecified for NaN-vs-NaN beyond the group_key
    // tie-break, so we only assert that all three groups
    // survive the sort.
    sort_diff_rows_by_keys(
        &mut rows,
        &mut Vec::new(),
        &[SortKey {
            metric: "run_time_ns",
            descending: true,
        }],
    );
    let mut groups: Vec<&str> = rows.iter().map(|r| r.group_key.as_str()).collect();
    groups.sort();
    groups.dedup();
    assert_eq!(
        groups,
        vec!["alpha", "bravo", "charlie"],
        "NaN delta must not drop or duplicate any group",
    );
}

