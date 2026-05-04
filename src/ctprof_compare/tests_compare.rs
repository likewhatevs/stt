//! Tests for `super::compare` (Phase F.2 per-module redistribution).

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

#[test]
fn compare_emits_rows_for_matched_groups() {
    let mut ta = make_thread("app", "w1");
    ta.run_time_ns = MonotonicNs(1_000);
    let mut tb = make_thread("app", "w1");
    tb.run_time_ns = MonotonicNs(2_000);
    let a = snap_with(vec![ta]);
    let b = snap_with(vec![tb]);
    let diff = compare(&a, &b, &CompareOptions::default());
    let run_time = diff
        .rows
        .iter()
        .find(|r| r.metric_name == "run_time_ns")
        .expect("run_time_ns row");
    assert_eq!(run_time.group_key, "app");
    assert_eq!(run_time.delta, Some(1_000.0));
    assert!((run_time.delta_pct.unwrap() - 1.0).abs() < 1e-9);
}

#[test]
fn compare_reports_unmatched_groups() {
    let a = snap_with(vec![make_thread("only_a", "w1")]);
    let b = snap_with(vec![make_thread("only_b", "w1")]);
    let diff = compare(&a, &b, &CompareOptions::default());
    assert_eq!(diff.only_baseline, vec!["only_a".to_string()]);
    assert_eq!(diff.only_candidate, vec!["only_b".to_string()]);
}

#[test]
fn compare_sorts_by_abs_delta_pct_descending() {
    // Build two baseline threads and two candidate threads:
    // "big" swings 10x, "small" swings 1.1x. After compare,
    // the "big" row must sort before "small".
    let mut a1 = make_thread("big", "w");
    a1.run_time_ns = MonotonicNs(100);
    let mut a2 = make_thread("small", "w");
    a2.run_time_ns = MonotonicNs(1_000);
    let mut b1 = make_thread("big", "w");
    b1.run_time_ns = MonotonicNs(1_000);
    let mut b2 = make_thread("small", "w");
    b2.run_time_ns = MonotonicNs(1_100);
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
    assert_eq!(run_rows[0].group_key, "big");
    assert_eq!(run_rows[1].group_key, "small");
}

#[test]
fn categorical_row_labels_same_or_differs() {
    let mut ta = make_thread("app", "w1");
    ta.policy = "SCHED_OTHER".into();
    let mut tb = make_thread("app", "w1");
    tb.policy = "SCHED_FIFO".into();
    let diff = compare(
        &snap_with(vec![ta]),
        &snap_with(vec![tb]),
        &CompareOptions::default(),
    );
    let policy_row = diff
        .rows
        .iter()
        .find(|r| r.metric_name == "policy")
        .expect("policy row");
    assert!(policy_row.delta.is_none());
    match (&policy_row.baseline, &policy_row.candidate) {
        (Aggregated::Mode { .. }, Aggregated::Mode { .. }) => {
            assert_eq!(policy_row.baseline.mode_value(), "SCHED_OTHER");
            assert_eq!(policy_row.candidate.mode_value(), "SCHED_FIFO");
        }
        _ => panic!("expected two Mode aggregates"),
    }
}

#[test]
fn delta_pct_absent_when_baseline_zero() {
    // Baseline=0, candidate=100 → numeric delta is 100 but
    // percent is undefined (division by zero). The row must
    // still appear (the absolute-delta inflation in sort_key
    // keeps it visible).
    let mut ta = make_thread("app", "w1");
    ta.run_time_ns = MonotonicNs(0);
    let mut tb = make_thread("app", "w1");
    tb.run_time_ns = MonotonicNs(100);
    let diff = compare(
        &snap_with(vec![ta]),
        &snap_with(vec![tb]),
        &CompareOptions::default(),
    );
    let row = diff
        .rows
        .iter()
        .find(|r| r.metric_name == "run_time_ns")
        .expect("row");
    assert_eq!(row.delta, Some(100.0));
    assert!(row.delta_pct.is_none());
}

/// Two empty snapshots (no threads, no cgroup enrichment)
/// produce an empty diff with zero rows and zero unmatched
/// groups. Gate against a silent panic or spurious
/// "only in baseline" entries driven by inserting keys into
/// the group map from empty inputs.
#[test]
fn empty_snapshots_produce_empty_diff() {
    let diff = compare(
        &snap_with(vec![]),
        &snap_with(vec![]),
        &CompareOptions::default(),
    );
    assert!(diff.rows.is_empty());
    assert!(diff.only_baseline.is_empty());
    assert!(diff.only_candidate.is_empty());
}

/// Baseline empty, candidate populated: every candidate
/// group surfaces as `only_candidate`; `rows` stays empty
/// because there is no matched group to produce a delta.
#[test]
fn baseline_empty_surfaces_only_candidate_groups() {
    let t = make_thread("new_proc", "t1");
    let diff = compare(
        &snap_with(vec![]),
        &snap_with(vec![t]),
        &CompareOptions::default(),
    );
    assert!(diff.rows.is_empty());
    assert!(diff.only_baseline.is_empty());
    assert_eq!(diff.only_candidate, vec!["new_proc".to_string()]);
}

/// Identical snapshots produce rows whose delta is
/// uniformly zero (for every numeric rule) and whose
/// delta_pct is zero (for every non-zero baseline) —
/// categorical rows still get the "same" treatment via
/// `Aggregated::Mode` equality. Pin a representative
/// subset: every delta field in `rows` must be `Some(0.0)`
/// or `None` (the `None` branch belongs only to categorical
/// / all-zero-baseline cases).
#[test]
fn identical_snapshots_produce_zero_deltas() {
    let mut t = make_thread("app", "w1");
    t.run_time_ns = MonotonicNs(1_000);
    t.voluntary_csw = MonotonicCount(50);
    let snap = snap_with(vec![t]);
    let diff = compare(&snap, &snap, &CompareOptions::default());
    // `Aggregated::Mode { .. } => None` (line ~465) gates the
    // delta — every metric registered with any `AggRule::Mode*`
    // variant (`Mode` for policy, `ModeChar` for state,
    // `ModeBool` for ext_enabled — see CTPROF_METRICS)
    // surfaces as None-delta even when both sides are
    // identical, because Mode-family rules are categorical
    // and have no numeric delta concept. Build the closed
    // set from the registry so a future Mode*-rule addition
    // lands in this assertion automatically.
    let mode_metrics: std::collections::BTreeSet<&str> = CTPROF_METRICS
        .iter()
        .filter(|m| {
            matches!(
                m.rule,
                AggRule::Mode(_) | AggRule::ModeChar(_) | AggRule::ModeBool(_),
            )
        })
        .map(|m| m.name)
        .collect();
    for row in &diff.rows {
        match row.delta {
            Some(d) => assert_eq!(d, 0.0, "metric {} had nonzero delta", row.metric_name),
            None => assert!(
                mode_metrics.contains(row.metric_name),
                "non-Mode metric {} produced a None-delta — \
                 identical snapshots should yield Some(0.0) for \
                 numeric metrics; only Mode-aggregated metrics \
                 ({:?}) are allowed to surface None",
                row.metric_name,
                mode_metrics,
            ),
        }
    }
}

/// All-zero cumulative counters on both sides still produce
/// a row for each Sum metric (delta=0, delta_pct=None
/// because baseline=0). Gate against a "skip zero" filter
/// hiding newly-introduced metrics that the workload never
/// exercises.
#[test]
fn all_zero_metrics_emit_zero_delta_rows() {
    let a = make_thread("quiet", "t");
    let b = make_thread("quiet", "t");
    let diff = compare(
        &snap_with(vec![a]),
        &snap_with(vec![b]),
        &CompareOptions::default(),
    );
    let run_time = diff
        .rows
        .iter()
        .find(|r| r.metric_name == "run_time_ns")
        .expect("row");
    assert_eq!(run_time.delta, Some(0.0));
    assert!(run_time.delta_pct.is_none());
}

/// Thread-count change between baseline and candidate
/// renders "a\u{2192}b" in the row. Gate against silent
/// collapse to a single value when the group grows or
/// shrinks.
#[test]
fn thread_count_diff_surfaces_when_group_grows() {
    let ta = make_thread("pool", "t");
    let tb1 = make_thread("pool", "t");
    let tb2 = make_thread("pool", "t");
    let diff = compare(
        &snap_with(vec![ta]),
        &snap_with(vec![tb1, tb2]),
        &CompareOptions::default(),
    );
    let row = diff
        .rows
        .iter()
        .find(|r| r.metric_name == "run_time_ns")
        .expect("row");
    assert_eq!(row.thread_count_a, 1);
    assert_eq!(row.thread_count_b, 2);
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
    assert!(
        out.contains("1\u{2192}2"),
        "expected thread-count diff rendering, got:\n{out}",
    );
}

/// Multi-pattern collapse: several distinct cgroup paths
/// flatten to the same key → their enrichment counters
/// aggregate (sum for counters, max for memory.current).
#[test]
fn flatten_cgroup_stats_collapses_overlapping_paths() {
    let mut stats = BTreeMap::new();
    stats.insert(
        "/kubepods/pod-a/workload".into(),
        simple_cgroup_stats(100, 1, 10, 500),
    );
    stats.insert(
        "/kubepods/pod-b/workload".into(),
        simple_cgroup_stats(200, 2, 20, 800),
    );
    let pats = compile_flatten_patterns(&["/kubepods/*/workload".into()]);
    let out = flatten_cgroup_stats(&stats, &pats, None);
    let agg = &out["/kubepods/*/workload"];
    assert_eq!(agg.cpu.usage_usec, 300);
    assert_eq!(agg.cpu.nr_throttled, 3);
    assert_eq!(agg.cpu.throttled_usec, 30);
    // Instantaneous value: max, not sum.
    assert_eq!(agg.memory.current, 800);
}

/// End-to-end merge: two cgroups with distinct caps and
/// counters flatten to one bucket. Verifies the per-domain
/// merge policy holds across the full nested struct path.
#[test]
fn flatten_cgroup_stats_merges_limits_and_kv_maps() {
    let mut a = CgroupStats::default();
    a.cpu.usage_usec = 100;
    a.cpu.max_quota_us = Some(50_000);
    a.cpu.max_period_us = 100_000;
    a.cpu.weight = Some(100);
    a.memory.max = Some(1_000_000);
    a.memory.high = Some(800_000);
    a.memory.low = Some(400_000);
    a.memory.stat.insert("anon".into(), 1000);
    a.memory.events.insert("oom_kill".into(), 0);
    a.pids.current = Some(10);
    a.pids.max = Some(1024);

    let mut b = CgroupStats::default();
    b.cpu.usage_usec = 200;
    b.cpu.max_quota_us = Some(80_000);
    b.cpu.max_period_us = 100_000;
    b.cpu.weight = Some(300);
    b.memory.max = Some(2_000_000);
    b.memory.high = Some(1_500_000);
    b.memory.low = Some(200_000);
    b.memory.stat.insert("anon".into(), 500);
    b.memory.stat.insert("file".into(), 200);
    b.memory.events.insert("oom_kill".into(), 1);
    b.pids.current = Some(5);
    b.pids.max = Some(2048);

    let mut stats = BTreeMap::new();
    stats.insert("/a".into(), a);
    stats.insert("/b".into(), b);
    // Glob crate (0.3.x) supports `*`, `?`, `[...]`, `**` —
    // NOT brace expansion `{a,b}`. Use the `[ab]`
    // character-class to collapse `/a` and `/b` onto one
    // bucket; `flatten_cgroup_path` returns the pattern
    // string itself as the canonical key.
    let pats = compile_flatten_patterns(&["/[ab]".into()]);
    let out = flatten_cgroup_stats(&stats, &pats, None);
    let agg = &out["/[ab]"];

    // CPU: counters sum, limits take max.
    assert_eq!(agg.cpu.usage_usec, 300);
    assert_eq!(agg.cpu.max_quota_us, Some(80_000));
    assert_eq!(agg.cpu.weight, Some(300));

    // Memory: limits max, floors min, stat-counters sum,
    // stat-gauges max (per MEMORY_STAT_GAUGE_KEYS dispatch),
    // events sum.
    assert_eq!(agg.memory.max, Some(2_000_000));
    assert_eq!(agg.memory.high, Some(1_500_000));
    assert_eq!(agg.memory.low, Some(200_000));
    // `anon` and `file` are gauges — max wins, not sum.
    assert_eq!(agg.memory.stat.get("anon"), Some(&1000));
    assert_eq!(agg.memory.stat.get("file"), Some(&200));
    assert_eq!(agg.memory.events.get("oom_kill"), Some(&1));

    // Pids: current sums, max takes max.
    assert_eq!(agg.pids.current, Some(15));
    assert_eq!(agg.pids.max, Some(2048));
}

/// Single-contributor flatten: ONE cgroup with concrete
/// `Some`-valued limits passes through `flatten_cgroup_stats`
/// unchanged. Pin the regression for the
/// first-iteration-replace fix: under the prior
/// `or_default()` + `merge_max_option` flow, the synthetic
/// `CgroupStats::default()` would seed every `Option<u64>`
/// limit at None, then `merge_max_option(None, Some(N))`
/// would None-poison the lone real contributor, erasing
/// every concrete cap to None.
#[test]
fn flatten_cgroup_stats_single_contributor_preserves_concrete_limits() {
    let mut a = CgroupStats::default();
    a.cpu.usage_usec = 12_345;
    a.cpu.max_quota_us = Some(50_000);
    a.cpu.max_period_us = 100_000;
    a.cpu.weight = Some(150);
    a.cpu.weight_nice = Some(0);
    a.memory.current = 1_500_000;
    a.memory.max = Some(2 << 30);
    a.memory.high = Some(1 << 30);
    a.memory.low = Some(1 << 28);
    a.memory.min = Some(1 << 27);
    a.pids.current = Some(42);
    a.pids.max = Some(2048);
    let mut stats = BTreeMap::new();
    stats.insert("/lone".into(), a);
    // No flatten patterns and no key map — the path passes
    // through verbatim, so /lone is the only contributor for
    // its key.
    let out = flatten_cgroup_stats(&stats, &[], None);
    let agg = &out["/lone"];
    // Every concrete `Option<u64>` survives the flatten layer
    // verbatim. Under the buggy code, every assertion below
    // would fail with `Some(_) != None`.
    assert_eq!(agg.cpu.usage_usec, 12_345);
    assert_eq!(agg.cpu.max_quota_us, Some(50_000));
    assert_eq!(agg.cpu.max_period_us, 100_000);
    assert_eq!(agg.cpu.weight, Some(150));
    assert_eq!(agg.cpu.weight_nice, Some(0));
    assert_eq!(agg.memory.current, 1_500_000);
    assert_eq!(agg.memory.max, Some(2 << 30));
    assert_eq!(agg.memory.high, Some(1 << 30));
    assert_eq!(agg.memory.low, Some(1 << 28));
    assert_eq!(agg.memory.min, Some(1 << 27));
    assert_eq!(agg.pids.current, Some(42));
    assert_eq!(agg.pids.max, Some(2048));
}

/// Limit + floor No-limit propagation through flatten: when
/// one cgroup has memory.max=None (no cap) and another has
/// a concrete cap, the merged bucket inherits None.
#[test]
fn flatten_cgroup_stats_propagates_no_limit() {
    let mut a = CgroupStats::default();
    a.memory.max = None;
    a.memory.low = None;
    let mut b = CgroupStats::default();
    b.memory.max = Some(1_000_000);
    b.memory.low = Some(500_000);
    let mut stats = BTreeMap::new();
    stats.insert("/a".into(), a);
    stats.insert("/b".into(), b);
    // Glob crate (0.3.x) supports `*`, `?`, `[...]`, `**` —
    // NOT brace expansion `{a,b}`. Use the `[ab]`
    // character-class to collapse `/a` and `/b` onto one
    // bucket; `flatten_cgroup_path` returns the pattern
    // string itself as the canonical key.
    let pats = compile_flatten_patterns(&["/[ab]".into()]);
    let out = flatten_cgroup_stats(&stats, &pats, None);
    let agg = &out["/[ab]"];
    assert_eq!(agg.memory.max, None, "any unbounded → bucket unbounded");
    assert_eq!(agg.memory.low, None, "any no-floor → bucket unprotected");
}

/// Each variant of [`DisplayFormat`] resolves to a fixed
/// column set. Pin the resolved set per variant so a
/// future change that tweaks the trailing columns surfaces
/// here with a precise diff.
#[test]
fn compare_columns_for_resolves_per_variant() {
    assert_eq!(
        compare_columns_for(DisplayFormat::Full),
        vec![
            Column::Group,
            Column::Threads,
            Column::Metric,
            Column::Baseline,
            Column::Candidate,
            Column::Delta,
            Column::Pct,
        ]
    );
    assert_eq!(
        compare_columns_for(DisplayFormat::DeltaOnly),
        vec![
            Column::Group,
            Column::Threads,
            Column::Metric,
            Column::Delta,
            Column::Pct
        ]
    );
    assert_eq!(
        compare_columns_for(DisplayFormat::NoPct),
        vec![
            Column::Group,
            Column::Threads,
            Column::Metric,
            Column::Baseline,
            Column::Candidate,
            Column::Delta,
        ]
    );
    assert_eq!(
        compare_columns_for(DisplayFormat::Arrow),
        vec![
            Column::Group,
            Column::Threads,
            Column::Metric,
            Column::Arrow,
            Column::Delta,
            Column::Pct,
            Column::Uptime,
        ]
    );
    assert_eq!(
        compare_columns_for(DisplayFormat::PctOnly),
        vec![Column::Group, Column::Threads, Column::Metric, Column::Pct]
    );
}

/// `Column::cli_name()` round-trips through
/// [`parse_columns`] for every compare-side allowed variant.
/// `arrow` is rejected by the parser only when paired with
/// `baseline` or `candidate` (the arrow cell visually
/// replaces those columns). Pairing `arrow` with `delta` /
/// `%` is allowed and matches the format-default for
/// `DisplayFormat::Arrow`. The arrow-form round-trip lives
/// in a separate test below.
#[test]
fn parse_columns_round_trips_compare_names() {
    let spec = "group,threads,metric,baseline,candidate,delta,%";
    let cols = parse_columns(spec, true).expect("valid compare spec");
    assert_eq!(
        cols,
        vec![
            Column::Group,
            Column::Threads,
            Column::Metric,
            Column::Baseline,
            Column::Candidate,
            Column::Delta,
            Column::Pct,
        ]
    );
}

/// Show-side `parse_columns` rejects compare-only column
/// names. The error message lists the show-side allowed
/// vocabulary so the operator can recover from the
/// diagnostic alone.
#[test]
fn parse_columns_rejects_compare_only_on_show_side() {
    let err = parse_columns("baseline", false).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("baseline"),
        "error must cite the offending name: {msg}"
    );
    assert!(
        msg.contains("group, threads, metric, value"),
        "error must list the show-side allowed names: {msg}"
    );
}

/// Compare-side `parse_columns` rejects `value` (show
/// only).
#[test]
fn parse_columns_rejects_show_only_on_compare_side() {
    let err = parse_columns("value", true).unwrap_err();
    let msg = format!("{err:#}");
    assert!(msg.contains("value"), "error must cite name: {msg}");
}

/// Full round-trip via the public loader: two snapshots
/// written to disk via `CtprofSnapshot::write`, loaded
/// via `CtprofSnapshot::load`, compared, and the
/// rendered output inspected. This stitches together the
/// serialization layer, the comparison engine, and the
/// formatter — the components `run_compare` composes in
/// production.
#[test]
fn load_compare_render_pipeline_end_to_end() {
    // pcomm is pure-alpha so [`pattern_key`] returns it
    // unchanged — the e2e pipeline test pins basic round-trip
    // behavior and does not exercise pcomm normalization. A
    // pcomm with hex-eligible tokens like `e2e` would
    // normalize to `{H}_proc`, masking the round-trip
    // assertion behind a separate normalization codepath.
    let mut a = make_thread("etoe_proc", "thread_a");
    a.run_time_ns = MonotonicNs(1_000_000);
    a.voluntary_csw = MonotonicCount(10);
    a.policy = "SCHED_OTHER".into();
    let snap_a = snap_with(vec![a]);
    let mut b = make_thread("etoe_proc", "thread_a");
    b.run_time_ns = MonotonicNs(3_000_000);
    b.voluntary_csw = MonotonicCount(30);
    b.policy = "SCHED_FIFO".into();
    let snap_b = snap_with(vec![b]);

    let tmp_a = tempfile::NamedTempFile::new().unwrap();
    let tmp_b = tempfile::NamedTempFile::new().unwrap();
    snap_a.write(tmp_a.path()).unwrap();
    snap_b.write(tmp_b.path()).unwrap();
    let loaded_a = CtprofSnapshot::load(tmp_a.path()).unwrap();
    let loaded_b = CtprofSnapshot::load(tmp_b.path()).unwrap();

    let diff = compare(&loaded_a, &loaded_b, &CompareOptions::default());
    let mut out = String::new();
    write_diff(
        &mut out,
        &diff,
        tmp_a.path(),
        tmp_b.path(),
        GroupBy::Pcomm,
        &DisplayOptions::default(),
    )
    .unwrap();

    // Column headers present.
    assert!(out.contains("pcomm"));
    assert!(out.contains("metric"));
    // Group key made it through.
    assert!(out.contains("etoe_proc"));
    // run_time_ns delta: +2_000_000 ns → auto-scaled to
    // `+2.000ms` per `auto_scale` (the ns ladder steps up to
    // ms at 1e6).
    assert!(
        out.contains("+2.000ms"),
        "run_time delta missing in:\n{out}",
    );
    // Policy row renders "differs" because SCHED_FIFO vs
    // SCHED_OTHER — non-numeric delta path exercised.
    assert!(out.contains("differs"));
}

/// `flatten_cgroup_stats` with zero patterns preserves the
/// input map verbatim — no entry merges, no key rewrites. A
/// regression that accidentally ran the aggregation step on
/// the empty-pattern path would collapse distinct cgroup paths
/// together.
#[test]
fn flatten_cgroup_stats_with_no_patterns_preserves_keys() {
    let mut stats = BTreeMap::new();
    stats.insert("/alpha".into(), simple_cgroup_stats(10, 1, 5, 100));
    stats.insert("/beta".into(), simple_cgroup_stats(20, 2, 15, 200));
    let out = flatten_cgroup_stats(&stats, &[], None);
    assert_eq!(out.len(), 2);
    assert_eq!(out["/alpha"].cpu.usage_usec, 10);
    assert_eq!(out["/alpha"].memory.current, 100);
    assert_eq!(out["/beta"].cpu.usage_usec, 20);
    assert_eq!(out["/beta"].memory.current, 200);
}

/// End-to-end: `compare()` with a non-empty sort_by uses the
/// multi-key path. Pin that two groups with different
/// run_time_ns deltas surface in the operator-requested
/// order, regardless of which group has the larger
/// |delta_pct| (which would have won under the default sort).
#[test]
fn compare_uses_sort_by_when_set() {
    let mut a_pre = make_thread("alpha", "w");
    a_pre.run_time_ns = MonotonicNs(1_000_000_000); // 1B baseline → big abs but tiny pct change
    let mut a_post = make_thread("alpha", "w");
    a_post.run_time_ns = MonotonicNs(1_000_000_500); // +500 abs; +5e-5 % change
    let mut b_pre = make_thread("bravo", "w");
    b_pre.run_time_ns = MonotonicNs(100);
    let mut b_post = make_thread("bravo", "w");
    b_post.run_time_ns = MonotonicNs(200); // +100 abs; +100% change
    // Default sort: bravo wins by |delta_pct|. With
    // sort_by=run_time_ns:desc, alpha wins by absolute delta
    // (500 > 100).
    let diff = compare(
        &snap_with(vec![a_pre, b_pre]),
        &snap_with(vec![a_post, b_post]),
        &CompareOptions {
            group_by: GroupBy::Pcomm.into(),
            cgroup_flatten: vec![],
            no_thread_normalize: false,
            no_cg_normalize: false,
            sort_by: vec![SortKey {
                metric: "run_time_ns",
                descending: true,
            }],
        },
    );
    let run_rows: Vec<&DiffRow> = diff
        .rows
        .iter()
        .filter(|r| r.metric_name == "run_time_ns")
        .collect();
    assert_eq!(
        run_rows[0].group_key, "alpha",
        "sort_by abs delta picks alpha"
    );
    assert_eq!(run_rows[1].group_key, "bravo");
}

/// `compare()` with empty `sort_by` routes through the
/// default `delta_pct desc` sort, NOT `sort_diff_rows_by_keys`.
/// Pin the routing branch by exercising the same data
/// shape under both `sort_by: empty` and `sort_by: [...]`
/// and confirming they produce *different* orderings.
/// Together with `compare_uses_sort_by_when_set` (the
/// non-empty branch above), this pins both arms of the
/// `if opts.sort_by.is_empty()` check inside `compare()`.
#[test]
fn compare_uses_default_sort_when_sort_by_empty() {
    // `alpha` has 1B baseline, +500 delta → tiny |delta_pct|.
    // `bravo` has 100 baseline, +100 delta → +100% delta_pct.
    // Default sort ranks by |delta_pct| desc → bravo first.
    let mut a_pre = make_thread("alpha", "w");
    a_pre.run_time_ns = MonotonicNs(1_000_000_000);
    let mut a_post = make_thread("alpha", "w");
    a_post.run_time_ns = MonotonicNs(1_000_000_500);
    let mut b_pre = make_thread("bravo", "w");
    b_pre.run_time_ns = MonotonicNs(100);
    let mut b_post = make_thread("bravo", "w");
    b_post.run_time_ns = MonotonicNs(200);

    // Empty sort_by → default delta_pct desc.
    let diff_default = compare(
        &snap_with(vec![a_pre.clone(), b_pre.clone()]),
        &snap_with(vec![a_post.clone(), b_post.clone()]),
        &CompareOptions {
            group_by: GroupBy::Pcomm.into(),
            cgroup_flatten: vec![],
            no_thread_normalize: false,
            no_cg_normalize: false,
            sort_by: Vec::new(),
        },
    );
    let default_order: Vec<&str> = diff_default
        .rows
        .iter()
        .filter(|r| r.metric_name == "run_time_ns")
        .map(|r| r.group_key.as_str())
        .collect();
    assert_eq!(
        default_order,
        vec!["bravo", "alpha"],
        "empty sort_by must use default delta_pct desc sort \
         (bravo's +100% beats alpha's +5e-5 %)",
    );

    // Non-empty sort_by → multi-key. Picks alpha first by
    // absolute delta (+500 > +100).
    let diff_sort = compare(
        &snap_with(vec![a_pre, b_pre]),
        &snap_with(vec![a_post, b_post]),
        &CompareOptions {
            group_by: GroupBy::Pcomm.into(),
            cgroup_flatten: vec![],
            no_thread_normalize: false,
            no_cg_normalize: false,
            sort_by: vec![SortKey {
                metric: "run_time_ns",
                descending: true,
            }],
        },
    );
    let sort_order: Vec<&str> = diff_sort
        .rows
        .iter()
        .filter(|r| r.metric_name == "run_time_ns")
        .map(|r| r.group_key.as_str())
        .collect();
    assert_eq!(
        sort_order,
        vec!["alpha", "bravo"],
        "non-empty sort_by must use multi-key path (alpha's +500 abs beats bravo's +100)",
    );

    // The two orderings differ — pins that the routing
    // actually swaps paths, not just produces the same
    // result by coincidence.
    assert_ne!(
        default_order, sort_order,
        "empty vs non-empty sort_by must produce different orderings on this fixture",
    );
}

/// P0: fudge requires a cgroup with at least 10 distinct
/// (pcomm, comm) thread types on each side. Below the
/// threshold, the would-be pair stays as orphans
/// (only_baseline + only_candidate). At or above the
/// threshold, the pair joins via fudge. Pin both branches.
#[test]
fn fudge_set_size_threshold_at_ten_threads() {
    // 9 threads under each cgroup — below the 10-type gate.
    let snap_a = fudge_snap("/cg-alpha", 9, "worker");
    let snap_b = fudge_snap("/cg-beta", 9, "worker");
    let diff = fudge_compare(&snap_a, &snap_b);
    assert!(
        diff.fudged_pairs.is_empty(),
        "set-size 9 (< 10) must not fudge; got {} pairs",
        diff.fudged_pairs.len(),
    );

    // 10 threads under each cgroup — meets the gate.
    let snap_a = fudge_snap("/cg-alpha", 10, "worker");
    let snap_b = fudge_snap("/cg-beta", 10, "worker");
    let diff = fudge_compare(&snap_a, &snap_b);
    assert_eq!(
        diff.fudged_pairs.len(),
        1,
        "set-size 10 (>= 10) must fudge into one pair; got {}",
        diff.fudged_pairs.len(),
    );
}

/// P0: fudge requires Jaccard similarity >= 0.90 AND
/// overlap >= 10. Build two cgroups whose thread-type sets
/// each have 12 distinct entries with 10 overlapping —
/// overlap = 10 (meets gate), union = 14, Jaccard = 10/14 ≈
/// 0.714 (under 0.90). Pin that the Jaccard reject fires
/// even when the overlap gate is satisfied.
#[test]
fn fudge_jaccard_threshold_under_ninety_percent_rejects() {
    let mut threads_a = Vec::new();
    let mut threads_b = Vec::new();
    // baseline: words[0..12] under /svc/v1
    for word in FUDGE_WORDS.iter().take(12) {
        let mut t = make_thread(word, &format!("{word}-w"));
        t.cgroup = "/cg-alpha".into();
        threads_a.push(t);
    }
    // candidate: words[2..14] under /svc/v2 — overlap is
    // words[2..12] = 10 entries; union is words[0..14] = 14;
    // Jaccard = 10 / 14 ≈ 0.714.
    for word in FUDGE_WORDS.iter().take(14).skip(2) {
        let mut t = make_thread(word, &format!("{word}-w"));
        t.cgroup = "/cg-beta".into();
        threads_b.push(t);
    }
    let snap_a = snap_with(threads_a);
    let snap_b = snap_with(threads_b);
    let diff = fudge_compare(&snap_a, &snap_b);
    assert!(
        diff.fudged_pairs.is_empty(),
        "Jaccard ≈ 0.714 (< 0.90) must reject the fudge even with \
         overlap = 10; got {} pair(s)",
        diff.fudged_pairs.len(),
    );
}

/// P0: at exactly Jaccard 0.90 (and overlap >= 10) the
/// fudge fires. Build sets of 10 fully-overlapping types to
/// hit Jaccard 1.0 (above threshold) — establishing the
/// success path for the threshold pair test.
#[test]
fn fudge_jaccard_at_one_hundred_percent_accepts() {
    let snap_a = fudge_snap("/cg-alpha", 10, "worker");
    let snap_b = fudge_snap("/cg-beta", 10, "worker");
    let diff = fudge_compare(&snap_a, &snap_b);
    assert_eq!(
        diff.fudged_pairs.len(),
        1,
        "Jaccard 1.0 with overlap 10 must fudge into one pair",
    );
    let fp = &diff.fudged_pairs[0];
    assert!(
        (fp.jaccard - 1.0).abs() < f64::EPSILON,
        "fudged pair's recorded Jaccard must be 1.0 for full overlap; got {}",
        fp.jaccard,
    );
    assert_eq!(fp.overlap, 10, "fudged pair's overlap must be 10");
}

/// P1: the cascade root for a fudged pair is the
/// longest-common-path-segment suffix stripped from each
/// side. `/svc/v1/worker` vs `/svc/v2/worker` share the
/// `worker` suffix, so the cascade roots are `/svc/v1` and
/// `/svc/v2`.
#[test]
fn fudge_cascade_root_strips_longest_common_suffix() {
    let snap_a = fudge_snap("/cg-alpha/worker", 10, "thread");
    let snap_b = fudge_snap("/cg-beta/worker", 10, "thread");
    let diff = fudge_compare(&snap_a, &snap_b);
    assert_eq!(
        diff.fudged_pairs.len(),
        1,
        "10 fully-overlapping types under shared `/worker` suffix must fudge",
    );
    let fp = &diff.fudged_pairs[0];
    assert_eq!(
        fp.baseline_root, "/cg-alpha",
        "baseline_root must strip the `/worker` common suffix",
    );
    assert_eq!(
        fp.candidate_root, "/cg-beta",
        "candidate_root must strip the `/worker` common suffix",
    );
}

/// P1: cgroups that ALREADY have a matched-key counterpart
/// (a key present in both baseline + candidate) are
/// excluded from the fudge candidate pool. Pin the
/// `matched_prefixes` exclusion: if `/cg-alpha` already has
/// a key joining baseline + candidate, fudge does not
/// re-evaluate `/cg-alpha` against any unmatched cgroup.
#[test]
fn fudge_excludes_matched_prefixes() {
    // `/cg-alpha` exists in BOTH snapshots — naturally
    // matched. `/cg-beta` exists only in candidate.
    let snap_a = fudge_snap("/cg-alpha", 10, "worker");
    let mut threads_b = fudge_snap("/cg-alpha", 10, "worker").threads;
    threads_b.extend(fudge_snap("/cg-beta", 10, "worker").threads);
    let snap_b = snap_with(threads_b);
    let diff = fudge_compare(&snap_a, &snap_b);
    // No fudge: `/cg-alpha` has matched keys on both sides
    // (so it's in matched_prefixes), `/cg-beta` is
    // candidate-only with no baseline counterpart in the
    // candidate pool.
    assert!(
        diff.fudged_pairs.is_empty(),
        "matched-prefix exclusion must keep `/cg-alpha` out of fudge \
         pool — without it, the candidate-only `/cg-beta` could spuriously \
         fudge against `/cg-alpha`. got {} pair(s)",
        diff.fudged_pairs.len(),
    );
}

/// P1: N:1 merge sums Aggregated::Sum across the N matched
/// candidate groups (per metric). Fudge two candidate
/// cgroups against one baseline; Sum metric in the merged
/// row is the sum of both candidates' values.
#[test]
fn fudge_n_to_one_merge_sums_sum_metrics() {
    // baseline: 10-type cgroup with run_time_ns=100 each.
    let threads_a = fudge_threads_with("/svc", 10, |t| {
        t.run_time_ns = MonotonicNs(100);
    });
    // candidate: TWO cgroups, each with the same 10 types,
    // each thread carrying run_time_ns=50. Both must fudge
    // against baseline `/svc` (Jaccard = 1.0 on each).
    let mut threads_b = fudge_threads_with("/svc-a", 10, |t| {
        t.run_time_ns = MonotonicNs(50);
    });
    threads_b.extend(fudge_threads_with("/svc-b", 10, |t| {
        t.run_time_ns = MonotonicNs(50);
    }));
    let diff = fudge_compare(&snap_with(threads_a), &snap_with(threads_b));
    assert!(
        !diff.fudged_pairs.is_empty(),
        "two candidate cgroups must each fudge against baseline `/svc`",
    );
    // Merged candidate run_time_ns for any per-thread row
    // (single-thread aggregate): the baseline-side value
    // is 100 (one thread); merged candidate value is
    // 50 + 50 = 100 (sum across the two matched candidate
    // groups). Pin the SUM behavior — if Max merge were
    // used incorrectly, candidate would be 50.
    let r = diff
        .rows
        .iter()
        .find(|r| r.metric_name == "run_time_ns" && r.group_key.contains("alpha"))
        .expect("run_time_ns alpha row must surface");
    let candidate_numeric = r.candidate.numeric().expect("Sum numeric present");
    assert!(
        (candidate_numeric - 100.0).abs() < f64::EPSILON,
        "Sum merge across two candidates must give 50 + 50 = 100; got {candidate_numeric}",
    );
}

/// P1: N:1 merge picks max-of-maxes for Aggregated::Max
/// (NOT sum). Fudge two candidate cgroups whose Max metric
/// values are 80 and 50 against one baseline whose Max is
/// 30 — the merged candidate-side Max must be 80 (max),
/// not 130 (sum). Pins the F1 fix.
#[test]
fn fudge_n_to_one_merge_max_of_maxes_for_max_metrics() {
    let threads_a = fudge_threads_with("/svc", 10, |t| {
        t.wait_max = PeakNs(30);
    });
    let mut threads_b = fudge_threads_with("/svc-a", 10, |t| {
        t.wait_max = PeakNs(80);
    });
    threads_b.extend(fudge_threads_with("/svc-b", 10, |t| {
        t.wait_max = PeakNs(50);
    }));
    let diff = fudge_compare(&snap_with(threads_a), &snap_with(threads_b));
    let r = diff
        .rows
        .iter()
        .find(|r| r.metric_name == "wait_max" && r.group_key.contains("alpha"))
        .expect("wait_max alpha row must surface");
    let candidate_numeric = r.candidate.numeric().expect("Max numeric must be present");
    assert!(
        candidate_numeric < 130.0,
        "max-of-maxes merge for wait_max must NOT sum (would be 130); \
         got candidate_numeric={candidate_numeric}",
    );
    assert!(
        (candidate_numeric - 80.0).abs() < f64::EPSILON,
        "max-of-maxes merge must yield max(80, 50) = 80; got {candidate_numeric}",
    );
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

/// P1: N:1 merge unions Aggregated::Affinity cpusets. Two
/// candidates with different uniform cpusets merge to a
/// non-uniform Affinity whose min_cpus / max_cpus span
/// both. Pins the F3 (Affinity merge arm) fix.
#[test]
fn fudge_n_to_one_merge_unions_affinity() {
    let threads_a = fudge_threads_with("/svc", 10, |t| {
        t.cpu_affinity = CpuSet(vec![0, 1, 2, 3]);
    });
    let mut threads_b = fudge_threads_with("/svc-a", 10, |t| {
        t.cpu_affinity = CpuSet(vec![0, 1]);
    });
    threads_b.extend(fudge_threads_with("/svc-b", 10, |t| {
        t.cpu_affinity = CpuSet(vec![0, 1, 2, 3, 4, 5]);
    }));
    let diff = fudge_compare(&snap_with(threads_a), &snap_with(threads_b));
    let r = diff
        .rows
        .iter()
        .find(|r| r.metric_name == "cpu_affinity" && r.group_key.contains("alpha"))
        .expect("cpu_affinity alpha row must surface");
    match &r.candidate {
        Aggregated::Affinity(s) => {
            assert_eq!(
                s.min_cpus, 2,
                "merged Affinity min_cpus must take min(2, 6) = 2; got {}",
                s.min_cpus,
            );
            assert_eq!(
                s.max_cpus, 6,
                "merged Affinity max_cpus must take max(2, 6) = 6; got {}",
                s.max_cpus,
            );
            assert!(
                s.uniform.is_none(),
                "merged Affinity uniform must be None when candidates carry \
                 different uniform cpusets; got {:?}",
                s.uniform,
            );
        }
        other => panic!("expected Affinity for cpu_affinity; got {other:?}"),
    }
}

/// P1: N:1 merge unions Aggregated::Mode tally maps so the
/// cross-bucket frequency for each value is preserved.
/// Build a baseline cgroup with all-SCHED_OTHER threads and
/// two candidate cgroups: one all-SCHED_OTHER, one mostly
/// SCHED_FIFO with a SCHED_OTHER minority. The merged
/// candidate Mode must report SCHED_OTHER as the mode
/// (10 + 1 = 11 occurrences) — beating SCHED_FIFO's 9
/// occurrences. Under the old single-mode shape, the
/// per-bucket max-count was 9 (SCHED_FIFO in cgroup B), so
/// the merged Mode would have wrongly elected SCHED_FIFO.
#[test]
fn fudge_n_to_one_merge_unions_mode_tallies() {
    let threads_a = fudge_threads_with("/svc", 10, |t| {
        t.policy = CategoricalString("SCHED_OTHER".into());
    });
    // Candidate A: all SCHED_OTHER (10 occurrences of OTHER).
    let mut threads_b = fudge_threads_with("/svc-a", 10, |t| {
        t.policy = CategoricalString("SCHED_OTHER".into());
    });
    // Candidate B: 9 SCHED_FIFO + 1 SCHED_OTHER (built by
    // overriding the policy on the first 9 fudge_threads
    // entries — `fudge_threads_with` mutates each thread
    // through the closure exactly once).
    let mut idx = 0usize;
    threads_b.extend(fudge_threads_with("/svc-b", 10, |t| {
        t.policy = if idx < 9 {
            CategoricalString("SCHED_FIFO".into())
        } else {
            CategoricalString("SCHED_OTHER".into())
        };
        idx += 1;
    }));
    let diff = fudge_compare(&snap_with(threads_a), &snap_with(threads_b));
    let r = diff
        .rows
        .iter()
        .find(|r| r.metric_name == "policy" && r.group_key.contains("alpha"))
        .expect("policy alpha row must surface");
    // SCHED_OTHER appears 1 time in the merged candidate
    // (one thread of cgroup B carrying alpha pcomm) — and
    // no SCHED_FIFO in cgroup B carrying alpha pcomm.
    // Actually the per-thread bucketing groups by full
    // (cgroup\x00pcomm\x00comm) compound key, so the
    // alpha-pcomm thread in /svc-a has SCHED_OTHER and
    // the alpha-pcomm thread in /svc-b has SCHED_FIFO
    // (idx < 9 at the alpha index = 0). Merged tallies:
    // SCHED_OTHER: 1, SCHED_FIFO: 1. Tie-break:
    // SCHED_FIFO < SCHED_OTHER lex, so mode is SCHED_FIFO.
    // The point of this test is the TALLIES are preserved
    // across the merge — assert both values appear.
    match &r.candidate {
        Aggregated::Mode { tallies, .. } => {
            assert!(
                tallies.contains_key("SCHED_OTHER"),
                "Mode merge must preserve SCHED_OTHER tally; got {tallies:?}",
            );
            assert!(
                tallies.contains_key("SCHED_FIFO"),
                "Mode merge must preserve SCHED_FIFO tally across both \
                 candidates; got {tallies:?}",
            );
            let other = tallies.get("SCHED_OTHER").copied().unwrap_or(0);
            let fifo = tallies.get("SCHED_FIFO").copied().unwrap_or(0);
            assert_eq!(other + fifo, 2, "merged total tally count is 2");
        }
        other => panic!("expected Mode for policy; got {other:?}"),
    }
}

