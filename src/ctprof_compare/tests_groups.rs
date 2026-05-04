//! Tests for `super::groups` (Phase F.2 per-module redistribution).

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
fn flatten_cgroup_path_collapses_via_pattern() {
    let pats = compile_flatten_patterns(&["/kubepods/*/workload".into()]);
    let out = flatten_cgroup_path("/kubepods/pod-abc-123/workload", &pats);
    assert_eq!(out, "/kubepods/*/workload");
}

#[test]
fn flatten_cgroup_path_falls_through_unmatched() {
    let pats = compile_flatten_patterns(&["/kubepods/*/workload".into()]);
    assert_eq!(
        flatten_cgroup_path("/system.slice/sshd.service", &pats),
        "/system.slice/sshd.service",
    );
}

#[test]
fn group_by_cgroup_applies_flatten_patterns() {
    let mut ta = make_thread("app", "w1");
    ta.cgroup = "/kubepods/pod-xxx/workload".into();
    ta.run_time_ns = MonotonicNs(1_000);
    let mut tb = make_thread("app", "w1");
    tb.cgroup = "/kubepods/pod-yyy/workload".into();
    tb.run_time_ns = MonotonicNs(2_000);
    let opts = CompareOptions {
        group_by: GroupBy::Cgroup.into(),
        cgroup_flatten: vec!["/kubepods/*/workload".into()],
        no_thread_normalize: false,
        no_cg_normalize: false,
        sort_by: Vec::new(),
    };
    let diff = compare(&snap_with(vec![ta]), &snap_with(vec![tb]), &opts);
    assert!(diff.only_baseline.is_empty(), "{:?}", diff.only_baseline);
    assert!(diff.only_candidate.is_empty(), "{:?}", diff.only_candidate,);
    assert!(
        diff.rows
            .iter()
            .any(|r| r.group_key == "/kubepods/*/workload"),
        "rows={:?}",
        diff.rows.iter().map(|r| &r.group_key).collect::<Vec<_>>(),
    );
}

#[test]
fn group_by_cgroup_surfaces_enrichment_on_diff() {
    let mut ta = make_thread("app", "w1");
    ta.cgroup = "/app".into();
    let mut snap_a = snap_with(vec![ta]);
    snap_a
        .cgroup_stats
        .insert("/app".into(), simple_cgroup_stats(100, 1, 50, 1 << 20));
    let mut tb = make_thread("app", "w1");
    tb.cgroup = "/app".into();
    let mut snap_b = snap_with(vec![tb]);
    snap_b
        .cgroup_stats
        .insert("/app".into(), simple_cgroup_stats(500, 3, 250, 2 << 20));
    let opts = CompareOptions {
        group_by: GroupBy::Cgroup.into(),
        cgroup_flatten: vec![],
        no_thread_normalize: false,
        no_cg_normalize: false,
        sort_by: Vec::new(),
    };
    let diff = compare(&snap_a, &snap_b, &opts);
    assert_eq!(diff.cgroup_stats_a["/app"].cpu.usage_usec, 100);
    assert_eq!(diff.cgroup_stats_b["/app"].cpu.usage_usec, 500);
}

/// `GroupBy::Comm` lumps threads with the same thread name
/// across processes.
#[test]
fn group_by_comm_aggregates_across_processes() {
    let mut ta = make_thread("procA", "worker");
    ta.run_time_ns = MonotonicNs(100);
    let mut tb = make_thread("procB", "worker");
    tb.run_time_ns = MonotonicNs(200);
    let mut candidate = make_thread("procA", "worker");
    candidate.run_time_ns = MonotonicNs(500);
    let mut candidate2 = make_thread("procB", "worker");
    candidate2.run_time_ns = MonotonicNs(500);
    let diff = compare(
        &snap_with(vec![ta, tb]),
        &snap_with(vec![candidate, candidate2]),
        &CompareOptions {
            group_by: GroupBy::Comm.into(),
            cgroup_flatten: vec![],
            no_thread_normalize: false,
            no_cg_normalize: false,
            sort_by: Vec::new(),
        },
    );
    let row = diff
        .rows
        .iter()
        .find(|r| r.metric_name == "run_time_ns" && r.group_key == "worker")
        .expect("worker row");
    // Summed across both processes: baseline=300, candidate=1000, delta=700.
    assert_eq!(row.thread_count_a, 2);
    assert_eq!(row.thread_count_b, 2);
    assert_eq!(row.delta, Some(700.0));
}

/// Earlier flatten pattern wins when multiple patterns
/// match the same path. Gate against a later pattern
/// silently stealing the collapse when an operator layers
/// broad and narrow patterns.
#[test]
fn flatten_first_match_wins_over_later_pattern() {
    let pats =
        compile_flatten_patterns(&["/kubepods/*/workload".into(), "/kubepods/**".into()]);
    assert_eq!(
        flatten_cgroup_path("/kubepods/pod-abc/workload", &pats),
        "/kubepods/*/workload",
    );
}

/// Malformed glob patterns are silently dropped by the
/// compiler (they never match so they never collapse
/// anything). Gate against a future change that accidentally
/// starts rejecting valid-looking patterns.
#[test]
fn compile_flatten_patterns_skips_malformed() {
    let pats = compile_flatten_patterns(&["[invalid".into(), "/ok/*".into()]);
    assert_eq!(pats.len(), 1);
    assert_eq!(pats[0].as_str(), "/ok/*");
}

#[test]
fn write_diff_enrichment_section_absent_when_group_by_pcomm() {
    let mut diff = CtprofDiff::default();
    // Populate enrichment; renderer must ignore it under
    // GroupBy::Pcomm.
    diff.cgroup_stats_a
        .insert("/app".into(), simple_cgroup_stats(10, 0, 0, 0));
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
    assert!(!out.contains("cpu_usage_usec"), "enrichment leaked:\n{out}");
}

/// A lone `worker-0` (no peer to share the prefix) reverts to
/// the literal comm so the operator does not see a fake
/// `worker-{N}` pattern matching only one thread.
#[test]
fn build_groups_comm_singleton_reverts_to_literal() {
    let snap = snap_with(vec![make_thread("app", "worker-0")]);
    let groups = build_groups(&snap, GroupBy::Comm, &[], None, None, false);
    assert!(
        groups.contains_key("worker-0"),
        "lone worker-0 stays literal",
    );
    assert!(
        !groups.contains_key("worker-{N}"),
        "no `worker-{{N}}` pattern key for a singleton",
    );
    assert_eq!(groups.len(), 1);
}

/// Different prefixes do not merge: `worker-0`, `worker-1`,
/// `worker-large-0`, `worker-large-1` produce two distinct
/// pattern buckets (`worker-{N}` and `worker-large-{N}`).
#[test]
fn build_groups_comm_distinct_prefixes_do_not_merge() {
    let snap = snap_with(vec![
        make_thread("app", "worker-0"),
        make_thread("app", "worker-1"),
        make_thread("app", "worker-large-0"),
        make_thread("app", "worker-large-1"),
    ]);
    let groups = build_groups(&snap, GroupBy::Comm, &[], None, None, false);
    assert_eq!(groups["worker-{N}"].thread_count, 2);
    assert_eq!(groups["worker-large-{N}"].thread_count, 2);
    assert_eq!(groups.len(), 2);
}

/// AlphaPrefix grouping (no separator before trailing digits)
/// clusters CamelCase names that share a prefix. 176
/// `CamelCaseWord{0..175}` threads (one per CPU) collapse
/// into one bucket — pin the bucket count and exact member
/// count to defend against a regression that reintroduces
/// the separator gate.
#[test]
fn build_groups_comm_alpha_prefix_clusters_camelcase() {
    let mut threads = Vec::new();
    for i in 0..6 {
        threads.push(make_thread("app", &format!("CamelCaseWord{i}")));
    }
    let snap = snap_with(threads);
    let groups = build_groups(&snap, GroupBy::Comm, &[], None, None, false);
    assert!(
        groups.contains_key("CamelCaseWord{N}"),
        "CamelCaseWord{{N}} bucket",
    );
    assert_eq!(groups["CamelCaseWord{N}"].thread_count, 6);
    assert_eq!(groups.len(), 1);
}

/// kworker workqueue grouping: workqueue-bearing kworkers
/// collapse across CPUs to one `kworker/{N}:{N}-<wq>` bucket
/// per workqueue. Different workqueues do NOT merge —
/// `wq_reclaim` and `mm_percpu_wq` each get their own bucket.
/// The workqueue suffix is whatever pure-alpha tokens form
/// (e.g. `wq_reclaim` tokenizes to `wq` + `_` + `reclaim`,
/// both literal).
#[test]
fn build_groups_comm_kworker_workqueue_collapses_per_cpu() {
    let snap = snap_with(vec![
        make_thread("kworker", "kworker/42:7-mm_percpu_wq"),
        make_thread("kworker", "kworker/43:8-mm_percpu_wq"),
        make_thread("kworker", "kworker/44:9-mm_percpu_wq"),
        make_thread("kworker", "kworker/0:0-wq_reclaim"),
        make_thread("kworker", "kworker/1:0-wq_reclaim"),
    ]);
    let groups = build_groups(&snap, GroupBy::Comm, &[], None, None, false);
    assert_eq!(groups["kworker/{N}:{N}-mm_percpu_wq"].thread_count, 3);
    assert_eq!(groups["kworker/{N}:{N}-wq_reclaim"].thread_count, 2);
    assert_eq!(groups.len(), 2);
}

/// Bare kworker (no `-<wq>` suffix) collapses across CPUs
/// under the token normalizer: `kworker/0:0`, `kworker/0:1`,
/// `kworker/1:0`, `kworker/3:2` all produce
/// `kworker/{N}:{N}` and join one bucket. This is the new
/// spec behavior — both `<cpu>` and `<id>` tokens normalize to
/// `{N}`.
#[test]
fn build_groups_comm_kworker_bare_collapses_across_cpus() {
    let snap = snap_with(vec![
        make_thread("kworker", "kworker/0:0"),
        make_thread("kworker", "kworker/0:1"),
        make_thread("kworker", "kworker/1:0"),
        make_thread("kworker", "kworker/3:2"),
    ]);
    let groups = build_groups(&snap, GroupBy::Comm, &[], None, None, false);
    assert_eq!(groups["kworker/{N}:{N}"].thread_count, 4);
    assert_eq!(groups.len(), 1);
}

/// Unbound kworker (`u<pool_id>:<id>`) and bound kworker
/// (`<cpu>:<id>`) skeletons differ — unbound has the `u`
/// prefix, bound does not. They group into separate buckets:
/// `kworker/u{N}:{N}` and `kworker/{N}:{N}`.
#[test]
fn build_groups_comm_kworker_unbound_separate_from_bound() {
    let snap = snap_with(vec![
        make_thread("kworker", "kworker/0:0"),
        make_thread("kworker", "kworker/3:2"),
        make_thread("kworker", "kworker/u8:3"),
        make_thread("kworker", "kworker/u8:7"),
        make_thread("kworker", "kworker/u16:0"),
    ]);
    let groups = build_groups(&snap, GroupBy::Comm, &[], None, None, false);
    assert_eq!(groups["kworker/{N}:{N}"].thread_count, 2);
    assert_eq!(groups["kworker/u{N}:{N}"].thread_count, 3);
    assert_eq!(groups.len(), 2);
}

/// Empty comm strings group together as the empty literal —
/// no panic, no special handling.
#[test]
fn build_groups_comm_empty_comm_does_not_panic() {
    let snap = snap_with(vec![make_thread("app", ""), make_thread("app", "")]);
    let groups = build_groups(&snap, GroupBy::Comm, &[], None, None, false);
    assert_eq!(groups[""].thread_count, 2);
}

/// TASK_COMM_LEN truncation: identical truncated comms group
/// together via the literal-comm branch (no separator before
/// trailing chars). Pin the all-too-common case where Linux
/// truncates a long thread name to 15 chars and two threads
/// land on the same truncated literal.
#[test]
fn build_groups_comm_truncated_comms_group_via_exact_match() {
    // Both threads share the same truncated 15-char comm.
    let snap = snap_with(vec![
        make_thread("app", "tokio-runtime-w"),
        make_thread("app", "tokio-runtime-w"),
    ]);
    let groups = build_groups(&snap, GroupBy::Comm, &[], None, None, false);
    // No trailing digits → pattern_key returns input unchanged
    // → both threads land in the same literal-comm bucket.
    assert_eq!(groups["tokio-runtime-w"].thread_count, 2);
    assert_eq!(groups.len(), 1);
}

/// Conservation: the sum of an aggregated counter across every
/// pattern bucket equals the sum across every input thread.
/// Pattern-aggregation must be bookkeeping-neutral.
#[test]
fn build_groups_comm_sum_conservation_across_buckets() {
    let mut threads = Vec::new();
    for i in 0..5 {
        let mut t = make_thread("app", &format!("worker-{i}"));
        t.run_time_ns = MonotonicNs(100 * (i as u64 + 1));
        threads.push(t);
    }
    for i in 0..3 {
        let mut t = make_thread("app", &format!("redis-bg-{i}"));
        t.run_time_ns = MonotonicNs(50 * (i as u64 + 1));
        threads.push(t);
    }
    let mut single = make_thread("app", "main");
    single.run_time_ns = MonotonicNs(999);
    threads.push(single);

    let input_total: u64 = threads.iter().map(|t| t.run_time_ns.0).sum();
    let snap = snap_with(threads);
    let groups = build_groups(&snap, GroupBy::Comm, &[], None, None, false);

    let aggregated_total: u64 = groups
        .values()
        .map(|g| match g.metrics.get("run_time_ns") {
            Some(Aggregated::Sum(n)) => *n,
            _ => 0,
        })
        .sum();
    assert_eq!(
        aggregated_total, input_total,
        "pattern-aggregated sum must equal input sum",
    );
}

/// `GroupBy::CommExact` preserves the old literal semantics —
/// `worker-0` and `worker-1` stay in distinct buckets.
#[test]
fn build_groups_comm_exact_preserves_literal_semantics() {
    let snap = snap_with(vec![
        make_thread("app", "worker-0"),
        make_thread("app", "worker-1"),
        make_thread("app", "worker-1"),
    ]);
    let groups = build_groups(&snap, GroupBy::CommExact, &[], None, None, false);
    assert_eq!(groups["worker-0"].thread_count, 1);
    assert_eq!(groups["worker-1"].thread_count, 2);
    assert_eq!(groups.len(), 2);
}

/// kworker-style parent processes collapse into one bucket
/// when grouped by pcomm under default normalization.
/// `kworker/0:0`, `kworker/1:0`, `kworker/2:1` all produce
/// the skeleton `kworker/{N}:{N}` so a 3-process fleet
/// clusters into one bucket. Mirrors
/// [`build_groups_comm_kworker_bare_collapses_across_cpus`]
/// for the Pcomm axis.
#[test]
fn build_groups_pcomm_kworker_collapses_across_cpus() {
    let snap = snap_with(vec![
        make_thread("kworker/0:0", "t0"),
        make_thread("kworker/1:0", "t1"),
        make_thread("kworker/3:2", "t2"),
    ]);
    let groups = build_groups(&snap, GroupBy::Pcomm, &[], None, None, false);
    assert_eq!(groups["kworker/{N}:{N}"].thread_count, 3);
    assert_eq!(groups.len(), 1);
}

/// Singleton pcomm reverts to literal so a lone parent
/// process does not advertise a `worker-{N}` pattern that
/// no other process shares. Mirrors
/// [`build_groups_comm_singleton_reverts_to_literal`] for the
/// Pcomm axis.
#[test]
fn build_groups_pcomm_singleton_reverts_to_literal() {
    let snap = snap_with(vec![make_thread("worker-7", "t0")]);
    let groups = build_groups(&snap, GroupBy::Pcomm, &[], None, None, false);
    assert!(
        groups.contains_key("worker-7"),
        "lone worker-7 stays literal under Pcomm normalization",
    );
    assert!(
        !groups.contains_key("worker-{N}"),
        "no `worker-{{N}}` pattern key for a singleton pcomm",
    );
    assert_eq!(groups.len(), 1);
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

/// Conservation: the sum of an aggregated counter across every
/// pattern bucket equals the sum across every input thread.
/// Pcomm pattern-aggregation must be bookkeeping-neutral —
/// mirrors [`build_groups_comm_sum_conservation_across_buckets`]
/// for the Pcomm axis.
#[test]
fn build_groups_pcomm_sum_conservation_across_buckets() {
    let mut threads = Vec::new();
    for i in 0..5 {
        let mut t = make_thread(&format!("worker-{i}"), "t");
        t.run_time_ns = MonotonicNs(100 * (i as u64 + 1));
        threads.push(t);
    }
    for i in 0..3 {
        let mut t = make_thread(&format!("redis-bg-{i}"), "t");
        t.run_time_ns = MonotonicNs(50 * (i as u64 + 1));
        threads.push(t);
    }
    let mut single = make_thread("init", "t");
    single.run_time_ns = MonotonicNs(999);
    threads.push(single);

    let input_total: u64 = threads.iter().map(|t| t.run_time_ns.0).sum();
    let snap = snap_with(threads);
    let groups = build_groups(&snap, GroupBy::Pcomm, &[], None, None, false);

    let aggregated_total: u64 = groups
        .values()
        .map(|g| match g.metrics.get("run_time_ns") {
            Some(Aggregated::Sum(n)) => *n,
            _ => 0,
        })
        .sum();
    assert_eq!(
        aggregated_total, input_total,
        "Pcomm pattern-aggregated sum must equal input sum",
    );
}

/// Default normalization collapses ephemeral PIDs into one
/// bucket per pcomm pattern. Three `worker-{0,1,2}` parents
/// (each with its own ephemeral tgid) all key as `worker-{N}`
/// — the tgid is dropped, so the join key matches the primary-
/// table Pcomm group key exactly. Per-field byte counts SUM
/// across the three collapsed PIDs.
#[test]
fn collect_smaps_rollup_normalizes_and_sums_across_pids() {
    let snap = snap_with(vec![
        smaps_thread("worker-0", 100, 1024, 512),
        smaps_thread("worker-1", 200, 2048, 1024),
        smaps_thread("worker-2", 300, 4096, 2048),
    ]);
    let out = collect_smaps_rollup(&snap, false);
    assert_eq!(out.len(), 1, "three PIDs collapse into one bucket: {out:?}");
    let bucket = out
        .get("worker-{N}")
        .expect("bucket key is pattern_key(pcomm) — no `[tgid]` suffix");
    // Values are bytes (kB * 1024).
    assert_eq!(
        bucket.get("Rss").copied(),
        Some((1024 + 2048 + 4096) * 1024),
        "Rss SUMs across the three collapsed PIDs",
    );
    assert_eq!(
        bucket.get("Pss").copied(),
        Some((512 + 1024 + 2048) * 1024),
        "Pss SUMs across the three collapsed PIDs",
    );
}

/// Default normalization always produces a normalized key —
/// no singleton revert. A lone `worker-7` parent process
/// still keys as `worker-{N}` so a literal-PID baseline like
/// `worker[7]` joins a literal-PID candidate `worker[1234]`
/// across snapshots, which is the load-bearing invariant
/// behind dropping the tgid suffix. The primary-table Pcomm
/// axis DOES revert singletons; smaps does NOT — the design
/// asymmetry is documented on `collect_smaps_rollup` and is
/// the bug-fix the normalization exists for.
#[test]
fn collect_smaps_rollup_no_singleton_revert_when_normalizing() {
    // Single leader for `worker-7` — under primary Pcomm
    // grouping this would revert to literal `worker-7`
    // because the bucket has count 1. smaps does NOT revert:
    // the join across baseline/candidate would otherwise fail
    // when the PID changes between snapshots.
    let snap = snap_with(vec![smaps_thread("worker-7", 99, 1024, 512)]);
    let out = collect_smaps_rollup(&snap, false);
    assert_eq!(out.len(), 1);
    assert!(
        out.contains_key("worker-{N}"),
        "lone worker-7 must STILL normalize to worker-{{N}} for smaps; \
         singleton-revert is intentionally skipped on the smaps axis: \
         got {:?}",
        out.keys().collect::<Vec<_>>(),
    );
    assert!(
        !out.contains_key("worker-7"),
        "literal singleton key must NOT appear under default smaps \
         normalization: got {:?}",
        out.keys().collect::<Vec<_>>(),
    );
}

/// `no_thread_normalize: true` preserves the literal
/// `pcomm[tgid]` key — each PID stays attributable to its
/// specific instance. Three workers produce three buckets
/// with their per-PID values verbatim, no summation.
#[test]
fn collect_smaps_rollup_no_normalize_preserves_literal_pid_keys() {
    let snap = snap_with(vec![
        smaps_thread("worker-0", 100, 1024, 512),
        smaps_thread("worker-1", 200, 2048, 1024),
        smaps_thread("worker-2", 300, 4096, 2048),
    ]);
    let out = collect_smaps_rollup(&snap, true);
    assert_eq!(
        out.len(),
        3,
        "no_normalize keeps three distinct PID buckets"
    );
    assert_eq!(out["worker-0[100]"]["Rss"], 1024 * 1024);
    assert_eq!(out["worker-1[200]"]["Rss"], 2048 * 1024);
    assert_eq!(out["worker-2[300]"]["Rss"], 4096 * 1024);
}

/// Empty snapshot produces an empty rollup map under both
/// modes (no panic, no synthesized entries). Boundary case.
#[test]
fn collect_smaps_rollup_empty_snapshot_returns_empty_map() {
    let snap = snap_with(vec![]);
    assert!(collect_smaps_rollup(&snap, false).is_empty());
    assert!(collect_smaps_rollup(&snap, true).is_empty());
}

/// Non-leader threads (tid != tgid) carry empty smaps_rollup
/// maps per the leader-dedup contract. The `is_empty()` skip
/// at the head of `collect_smaps_rollup` filters them — pin
/// that they don't synthesize ghost buckets under either
/// normalization mode.
#[test]
fn collect_smaps_rollup_skips_non_leader_threads() {
    let leader = smaps_thread("worker-0", 100, 1024, 512);
    let mut non_leader = ThreadState {
        tid: 101,
        tgid: 100,
        pcomm: "worker-0".into(),
        comm: "worker-0".into(),
        cgroup: "/".into(),
        ..ThreadState::default()
    };
    // non_leader.smaps_rollup_kb stays empty (default) — the
    // capture-side dedup contract means non-leader threads
    // never carry a populated map.
    assert!(non_leader.smaps_rollup_kb.is_empty());
    // Reassure: clearing is the no-op the contract assumes.
    non_leader.smaps_rollup_kb.clear();
    let snap = snap_with(vec![leader, non_leader]);
    // Default normalize: one bucket from the leader keyed by
    // `pattern_key(pcomm)`; no ghost entry from the
    // non-leader's empty map.
    let out_norm = collect_smaps_rollup(&snap, false);
    assert_eq!(out_norm.len(), 1);
    assert!(out_norm.contains_key("worker-{N}"));
    // No-normalize: one bucket keyed at the leader's literal
    // pcomm[tgid].
    let out_lit = collect_smaps_rollup(&snap, true);
    assert_eq!(out_lit.len(), 1);
    assert!(out_lit.contains_key("worker-0[100]"));
}

/// Multiple PIDs with the same pcomm pattern but disjoint
/// smaps_rollup field sets (e.g. one snapshot has Rss only,
/// another has Pss only) merge into one bucket whose map
/// carries every field that any contributor reported. Pin
/// that absent fields don't shadow present ones at the merge
/// boundary.
#[test]
fn collect_smaps_rollup_merge_carries_every_field_seen() {
    let t1 = smaps_thread("worker-0", 100, 1024, 512);
    let mut t2 = ThreadState {
        tid: 200,
        tgid: 200,
        pcomm: "worker-1".into(),
        comm: "worker-1".into(),
        cgroup: "/".into(),
        ..ThreadState::default()
    };
    // t1 has Rss + Pss. t2 has Rss + Private_Clean only.
    t2.smaps_rollup_kb.insert("Rss".into(), 2048);
    t2.smaps_rollup_kb.insert("Private_Clean".into(), 256);
    // t1 keeps its Rss + Pss from the helper, no Private_Clean.
    assert!(!t1.smaps_rollup_kb.contains_key("Private_Clean"));

    let snap = snap_with(vec![t1, t2]);
    let out = collect_smaps_rollup(&snap, false);
    let bucket = out.get("worker-{N}").expect("merged bucket");
    // Rss: 1024 + 2048 (both contribute).
    assert_eq!(bucket.get("Rss").copied(), Some((1024 + 2048) * 1024));
    // Pss: only t1 contributed → t1's value alone.
    assert_eq!(bucket.get("Pss").copied(), Some(512 * 1024));
    // Private_Clean: only t2 contributed → t2's value alone.
    assert_eq!(bucket.get("Private_Clean").copied(), Some(256 * 1024));
}

/// Saturating overflow: two leader threads each reporting
/// `Rss = u64::MAX kB` (impossible in practice, defensive
/// pin). Sum via `saturating_add` must not panic; the
/// merged Rss caps at `u64::MAX` bytes after the kB→B
/// conversion. Without `saturating_add`, the addition would
/// overflow and panic in debug builds.
#[test]
fn collect_smaps_rollup_saturating_add_does_not_panic_on_overflow() {
    let snap = snap_with(vec![
        smaps_thread("worker-0", 100, u64::MAX, 1),
        smaps_thread("worker-1", 200, u64::MAX, 1),
    ]);
    let out = collect_smaps_rollup(&snap, false);
    let bucket = out.get("worker-{N}").expect("merged bucket");
    // `smaps_rollup_bytes` converts kB → B by multiplying by
    // 1024; with kB at u64::MAX the conversion itself
    // saturates inside `smaps_rollup_bytes`. The
    // post-conversion sum then saturates again on the
    // second contributor. Either way, the merge never
    // panics — pin the well-defined output value.
    let v = bucket
        .get("Rss")
        .copied()
        .expect("Rss key present after overflow");
    assert_eq!(
        v,
        u64::MAX,
        "saturating_add must clamp to u64::MAX, not panic",
    );
}

/// Distinct prefixes do not merge under Pcomm: `worker-0/1`
/// and `worker-large-0/1` produce two normalized buckets
/// (`worker-{N}` and `worker-large-{N}`), not one. Pcomm-
/// axis sibling to [`build_groups_comm_distinct_prefixes_do_not_merge`].
#[test]
fn build_groups_pcomm_distinct_prefixes_do_not_merge() {
    let snap = snap_with(vec![
        make_thread("worker-0", "t"),
        make_thread("worker-1", "t"),
        make_thread("worker-large-0", "t"),
        make_thread("worker-large-1", "t"),
    ]);
    let groups = build_groups(&snap, GroupBy::Pcomm, &[], None, None, false);
    assert_eq!(groups["worker-{N}"].thread_count, 2);
    assert_eq!(groups["worker-large-{N}"].thread_count, 2);
    assert_eq!(groups.len(), 2);
}

/// Singleton PID smaps pin: a single leader thread with
/// pcomm `bash` (pure-alpha, no normalizer rule fires)
/// produces ONE bucket keyed at `pattern_key("bash") =
/// "bash"`. The key drops the tgid suffix even with one
/// PID, matching what the primary table's Pcomm bucket
/// would render. Without dropping the suffix, smaps would
/// emit `bash[42]` while the primary table shows `bash` —
/// the very mismatch the design fix was here to eliminate.
#[test]
fn collect_smaps_rollup_singleton_drops_tgid_suffix() {
    let snap = snap_with(vec![smaps_thread("bash", 42, 4096, 1024)]);
    let out = collect_smaps_rollup(&snap, false);
    assert_eq!(out.len(), 1);
    assert!(
        out.contains_key("bash"),
        "singleton bash key must equal pattern_key(\"bash\") = \"bash\"; \
         got {:?}",
        out.keys().collect::<Vec<_>>(),
    );
    assert!(
        !out.contains_key("bash[42]"),
        "singleton must NOT carry the tgid suffix under \
         default normalization: got {:?}",
        out.keys().collect::<Vec<_>>(),
    );
}

/// Empty pcomm threads collapse together under default
/// normalization. `pattern_key("")` returns the empty
/// string; two threads with empty pcomm both key as `""`
/// and merge into one bucket. Defensive pin: the
/// `pattern_key` empty-input arm and the merge path both
/// have to survive the empty key without panic. Real-world
/// hits include kernel threads whose comm read failed
/// during capture — capture-side default for an unreadable
/// comm is the empty string.
#[test]
fn build_groups_pcomm_empty_pcomm_collapses_under_normalization() {
    let snap = snap_with(vec![make_thread("", "t0"), make_thread("", "t1")]);
    let groups = build_groups(&snap, GroupBy::Pcomm, &[], None, None, false);
    assert_eq!(groups[""].thread_count, 2);
    assert_eq!(groups.len(), 1);
}

/// Bracketed pcomms collapse under Pcomm normalization. Three
/// processes with pcomms `[stress-ng-0]`, `[stress-ng-1]`,
/// `[stress-ng-2]` all key as `[stress-ng-{N}]` after
/// `pattern_key` runs (brackets are separators, the digit
/// suffix normalizes to `{N}`). Pins the integration of
/// bracket-as-separator with the build_groups Pcomm path.
#[test]
fn build_groups_pcomm_bracketed_pcomms_collapse() {
    let snap = snap_with(vec![
        make_thread("[stress-ng-0]", "t0"),
        make_thread("[stress-ng-1]", "t1"),
        make_thread("[stress-ng-2]", "t2"),
    ]);
    let groups = build_groups(&snap, GroupBy::Pcomm, &[], None, None, false);
    assert_eq!(
        groups["[stress-ng-{N}]"].thread_count,
        3,
        "all three bracketed pcomms must collapse into one bucket; got {:?}",
        groups.keys().collect::<Vec<_>>(),
    );
    assert_eq!(groups.len(), 1);
}

/// TASK_COMM_LEN truncation under Pcomm: identical truncated
/// pcomms group together via the literal-pcomm branch (no
/// normalization fires when the tail tokens are pure alpha).
/// Mirror of `build_groups_comm_truncated_comms_group_via_exact_match`
/// for the Pcomm axis. Two processes share the same 15-char
/// truncated pcomm and merge into one bucket.
#[test]
fn build_groups_pcomm_truncated_pcomms_group_via_exact_match() {
    // Both processes share the same truncated 15-char pcomm.
    // `tokio-runtime-w` has tokens `tokio`, `runtime`, `w` —
    // all pure alpha → literal → bucket key matches input.
    let snap = snap_with(vec![
        make_thread("tokio-runtime-w", "t0"),
        make_thread("tokio-runtime-w", "t1"),
    ]);
    let groups = build_groups(&snap, GroupBy::Pcomm, &[], None, None, false);
    assert_eq!(
        groups["tokio-runtime-w"].thread_count, 2,
        "identical truncated pcomms collapse via literal-pcomm branch",
    );
    assert_eq!(groups.len(), 1);
}

/// `collect_smaps_rollup` normalizes pcomm independently of
/// the primary group_by axis. Build a snapshot with worker
/// processes whose threads carry mixed cgroups; even if a
/// caller groups primary metrics by cgroup, smaps keys still
/// flow through `pattern_key(&t.pcomm)` and merge across
/// PIDs. Pins the design property that smaps keying is
/// orthogonal to `--group-by` — the smaps section reads
/// pcomm directly off each leader thread, not the
/// post-grouping bucket key.
#[test]
fn collect_smaps_rollup_independent_of_group_by_axis() {
    let mut t0 = smaps_thread("worker-0", 100, 1024, 512);
    t0.cgroup = "/cg-a".into();
    let mut t1 = smaps_thread("worker-1", 200, 2048, 1024);
    t1.cgroup = "/cg-b".into();
    let mut t2 = smaps_thread("worker-2", 300, 4096, 2048);
    t2.cgroup = "/cg-c".into();
    let snap = snap_with(vec![t0, t1, t2]);
    // Drive collect_smaps_rollup directly with the
    // normalize-on path. The function takes `(snap,
    // no_thread_normalize)` only — group_by is not in its
    // signature, which is the load-bearing fact this test
    // pins. Three distinct cgroup paths but one normalized
    // pcomm bucket.
    let out = collect_smaps_rollup(&snap, false);
    assert_eq!(
        out.len(),
        1,
        "smaps keying must collapse pcomm-pattern siblings \
         regardless of cgroup distribution: got {:?}",
        out.keys().collect::<Vec<_>>(),
    );
    let bucket = out.get("worker-{N}").expect("merged worker bucket");
    assert_eq!(
        bucket.get("Rss").copied(),
        Some((1024 + 2048 + 4096) * 1024)
    );
}

/// Empty pcomm under default smaps normalization: a leader
/// thread whose pcomm is the empty string (kernel-thread
/// capture race, unreadable comm fallback) keys at
/// `pattern_key("") = ""`. Two such leaders aggregate into
/// one bucket whose key is the empty string. Defensive pin —
/// the empty-key path through `or_default()` and the
/// per-field saturating_add merge must survive without
/// panic. Mirrors `build_groups_pcomm_empty_pcomm_collapses_under_normalization`
/// for the smaps axis.
#[test]
fn collect_smaps_rollup_empty_pcomm_collapses_under_normalization() {
    let snap = snap_with(vec![
        smaps_thread("", 100, 1024, 512),
        smaps_thread("", 200, 2048, 1024),
    ]);
    let out = collect_smaps_rollup(&snap, false);
    assert_eq!(
        out.len(),
        1,
        "two empty-pcomm leaders must merge into one bucket; got {:?}",
        out.keys().collect::<Vec<_>>(),
    );
    let bucket = out.get("").expect("empty-key bucket");
    assert_eq!(bucket.get("Rss").copied(), Some((1024 + 2048) * 1024));
    assert_eq!(bucket.get("Pss").copied(), Some((512 + 1024) * 1024));
}

/// P0: fudge is gated on `GroupBy::All`. Other group_by
/// modes do not activate the fudge stage even when the
/// cgroups would otherwise qualify. Pin the GroupBy guard.
#[test]
fn fudge_only_runs_under_group_by_all() {
    let snap_a = fudge_snap("/cg-alpha", 10, "worker");
    let snap_b = fudge_snap("/cg-beta", 10, "worker");
    // GroupBy::Cgroup would also produce two distinct
    // baseline/candidate buckets keyed by the cgroup paths
    // — without fudge, they stay as orphans.
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
    assert!(
        diff.fudged_pairs.is_empty(),
        "GroupBy::Cgroup must not activate fudge; got {} pair(s)",
        diff.fudged_pairs.len(),
    );
}

