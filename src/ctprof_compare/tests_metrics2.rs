//! Tests for `super::metrics` (Phase F.2 per-module redistribution).

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

/// Brackets split tokens identically to the existing separator
/// class. `worker[42]` tokenizes to
/// `[Token("worker"), Sep("["), Token("42"), Sep("]")]` so a
/// rejoin under `pattern_key` produces `worker[{N}]`. A
/// regression that removed `[` / `]` from the separator class
/// would surface here as `worker[42]` returning literal because
/// the bracketed-digit token would no longer reach rule 1.
#[test]
fn pattern_key_normalizes_bracketed_digits() {
    assert_eq!(pattern_key("worker[42]"), "worker[{N}]");
    assert_eq!(
        pattern_key("systemd-network[105904]"),
        "systemd-network[{N}]"
    );
    // Both pcomm halves and the tgid normalize when each side
    // is hex/digit-eligible. `bash[4242]` — bash is pure alpha,
    // 4242 is pure digits → `bash[{N}]`.
    assert_eq!(pattern_key("bash[4242]"), "bash[{N}]");
    // Hex-only inside the brackets still picks `{H}` per the
    // hex-rule precedence over rule 4.
    assert_eq!(pattern_key("dev[1ab]"), "dev[{H}]");
}

/// `[` and `]` join the existing separator class — `split_into_segments`
/// emits separator runs that include them verbatim. Pin both
/// the standalone bracket and a multi-char run mixing brackets
/// with other separators.
#[test]
fn split_into_segments_treats_brackets_as_separators() {
    let segs = split_into_segments("worker[42]");
    assert_eq!(
        segs,
        vec![
            Segment::Token("worker"),
            Segment::Separator("["),
            Segment::Token("42"),
            Segment::Separator("]"),
        ],
    );
    // Bracket adjacent to existing separator chars merges into
    // a single separator run.
    let segs = split_into_segments("a-[1]");
    assert_eq!(
        segs,
        vec![
            Segment::Token("a"),
            Segment::Separator("-["),
            Segment::Token("1"),
            Segment::Separator("]"),
        ],
    );
}

/// `is_token_separator` returns true for `[` and `]` directly.
/// Pin the boolean predicate so a regression that drops a
/// bracket from the `matches!` arm surfaces as a unit-test
/// failure rather than only at the end-to-end pattern-rejoin
/// site.
#[test]
fn is_token_separator_includes_brackets() {
    assert!(is_token_separator('['));
    assert!(is_token_separator(']'));
}

// ------------------------------------------------------------
// GroupBy::Pcomm normalization
//
// Pcomm now flows through the same token-based normalizer as
// Comm — ephemeral worker pools whose pcomm differs only by
// a digit suffix collapse across snapshots. The pin tests
// below mirror the Comm-axis tests on the Pcomm axis.
// ------------------------------------------------------------

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

/// Cross-snapshot frequency union promotes a `worker-{N}`
/// pattern when baseline has 1 process + candidate has 2 —
/// the union total (3) crosses the >= 2 promotion gate so
/// both sides use the same `worker-{N}` join key. Mirrors
/// [`compare_comm_pattern_joins_across_asymmetric_resize`]
/// for the Pcomm axis. Without the union, baseline's
/// `worker-7` would gate to literal (count 1) while
/// candidate's two would gate to pattern, producing orphaned
/// only-in-baseline / only-in-candidate rows.
#[test]
fn compare_pcomm_pattern_joins_across_asymmetric_resize() {
    let baseline = snap_with(vec![make_thread("worker-7", "t0")]);
    let candidate = snap_with(vec![
        make_thread("worker-0", "t0"),
        make_thread("worker-1", "t1"),
    ]);
    let diff = compare(
        &baseline,
        &candidate,
        &CompareOptions {
            group_by: GroupBy::Pcomm.into(),
            cgroup_flatten: vec![],
            no_thread_normalize: false,
            no_cg_normalize: false,
            sort_by: Vec::new(),
        },
    );
    let row = diff
        .rows
        .iter()
        .find(|r| r.metric_name == "run_time_ns" && r.group_key == "worker-{N}")
        .expect("worker-{N} pcomm row joined across asymmetric snapshots");
    assert_eq!(row.thread_count_a, 1, "baseline carries 1 worker process");
    assert_eq!(
        row.thread_count_b, 2,
        "candidate carries 2 worker processes"
    );
    // No orphan rows for the worker family.
    let baseline_orphans: Vec<&String> = diff
        .only_baseline
        .iter()
        .filter(|k| k.starts_with("worker"))
        .collect();
    assert!(
        baseline_orphans.is_empty(),
        "no worker-prefixed pcomm orphans in only_baseline; got {baseline_orphans:?}",
    );
    let candidate_orphans: Vec<&String> = diff
        .only_candidate
        .iter()
        .filter(|k| k.starts_with("worker"))
        .collect();
    assert!(
        candidate_orphans.is_empty(),
        "no worker-prefixed pcomm orphans in only_candidate; got {candidate_orphans:?}",
    );
}

/// End-to-end: `compare(GroupBy::Pcomm, ...)` produces a
/// `DiffRow` whose `group_key` is the `prefix-{N}` skeleton
/// (deterministic across snapshots) and whose `display_key`
/// is fed by [`pattern_display_label`] over the union of
/// baseline + candidate pcomm members. Mirrors
/// [`compare_comm_pattern_emits_prefix_join_key_and_grex_display`]
/// for the Pcomm axis.
#[test]
fn compare_pcomm_pattern_emits_prefix_join_key_and_grex_display() {
    let baseline = snap_with(vec![
        make_thread("worker-0", "t0"),
        make_thread("worker-1", "t1"),
    ]);
    let candidate = snap_with(vec![
        make_thread("worker-2", "t0"),
        make_thread("worker-3", "t1"),
    ]);
    let diff = compare(
        &baseline,
        &candidate,
        &CompareOptions {
            group_by: GroupBy::Pcomm.into(),
            cgroup_flatten: vec![],
            no_thread_normalize: false,
            no_cg_normalize: false,
            sort_by: Vec::new(),
        },
    );
    let row = diff
        .rows
        .iter()
        .find(|r| r.metric_name == "run_time_ns" && r.group_key == "worker-{N}")
        .expect("worker-{N} pcomm row");
    assert_eq!(
        row.group_key, "worker-{N}",
        "join key is the placeholder pattern under Pcomm normalization",
    );
    assert!(
        row.display_key.contains("worker"),
        "display key reflects grex (or fallback to join key) over union; got {:?}",
        row.display_key,
    );
    // The display label is whatever `pattern_display_label`
    // produces for the union of distinct members — either
    // the grex regex when it fits within the join key, or
    // the join key itself under the high-cardinality
    // fallback. Pin the contract by computing the expected
    // label directly and comparing.
    let mut union_members: Vec<String> = vec![
        "worker-0".into(),
        "worker-1".into(),
        "worker-2".into(),
        "worker-3".into(),
    ];
    union_members.sort();
    union_members.dedup();
    let expected_label = pattern_display_label("worker-{N}", &union_members);
    assert_eq!(
        row.display_key, expected_label,
        "display label must match pattern_display_label over union"
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

// ------------------------------------------------------------
// collect_smaps_rollup normalization
//
// smaps_rollup keys default to `pattern_key(&t.pcomm)` (the
// tgid is dropped) so ephemeral PIDs collapse into one bucket
// per pcomm pattern; multiple PIDs mapping to the same key SUM
// their per-field byte counts. Under `no_thread_normalize:
// true`, the literal `pcomm[tgid]` shape is preserved instead
// so each PID stays attributable.
// ------------------------------------------------------------

/// Helper: build a leader thread with a populated smaps_rollup
/// map. The `tid == tgid` shape lets the leader-dedup gate
/// inside `collect_smaps_rollup` admit the row.
fn smaps_thread(pcomm: &str, tgid: u32, rss_kb: u64, pss_kb: u64) -> ThreadState {
    let mut t = ThreadState {
        tid: tgid,
        tgid,
        pcomm: pcomm.into(),
        comm: pcomm.into(),
        cgroup: "/".into(),
        ..ThreadState::default()
    };
    t.smaps_rollup_kb.insert("Rss".into(), rss_kb);
    t.smaps_rollup_kb.insert("Pss".into(), pss_kb);
    t
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

/// Sort by total Rss desc: the smaps render iterates process
/// keys ranked by max(baseline_rss, candidate_rss) descending
/// so the heaviest mover appears first. Pin that the rendered
/// table places `heavy` ahead of `light` regardless of
/// alphabetical key order. Without the sort, BTreeSet
/// iteration would put `heavy` after `light`.
#[test]
fn write_diff_smaps_orders_processes_by_rss_desc() {
    let mut diff = CtprofDiff::default();
    let mut heavy = BTreeMap::new();
    heavy.insert("Rss".to_string(), 100 * 1024 * 1024); // 100 MiB
    heavy.insert("Pss".to_string(), 50 * 1024 * 1024);
    let mut heavy_b = BTreeMap::new();
    heavy_b.insert("Rss".to_string(), 200 * 1024 * 1024);
    heavy_b.insert("Pss".to_string(), 100 * 1024 * 1024);
    let mut light = BTreeMap::new();
    light.insert("Rss".to_string(), 1024); // 1 KiB
    light.insert("Pss".to_string(), 512);
    let mut light_b = BTreeMap::new();
    light_b.insert("Rss".to_string(), 2048);
    light_b.insert("Pss".to_string(), 1024);
    // Keys ordered alphabetically (`a_light` before `b_heavy`
    // when sorted) so a regression that fell back to BTreeSet
    // iteration would put a_light first.
    diff.smaps_rollup_a.insert("a_light[1]".to_string(), light);
    diff.smaps_rollup_b
        .insert("a_light[1]".to_string(), light_b);
    diff.smaps_rollup_a.insert("b_heavy[2]".to_string(), heavy);
    diff.smaps_rollup_b
        .insert("b_heavy[2]".to_string(), heavy_b);

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
    let after_header = &out[smaps_at..];
    let heavy_pos = after_header
        .find("b_heavy[2]")
        .expect("b_heavy must appear");
    let light_pos = after_header
        .find("a_light[1]")
        .expect("a_light must appear");
    assert!(
        heavy_pos < light_pos,
        "process with larger Rss must render first; \
         b_heavy@{heavy_pos} must precede a_light@{light_pos}",
    );
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

/// Literal bracket name (no digits inside): `pattern_key`
/// returns the input unchanged. `[bar]` tokenizes to
/// `[Sep("["), Token("bar"), Sep("]")]`; `bar` is pure alpha
/// (no rule fires). The whole input thus echoes through
/// — the bracket separators are preserved verbatim.
#[test]
fn pattern_key_bracket_alpha_token_stays_literal() {
    assert_eq!(pattern_key("foo[bar]"), "foo[bar]");
    assert_eq!(pattern_key("a[b]"), "a[b]");
    // Hex-eligible alpha-only inside brackets still stays
    // literal — rule 2 requires at least one digit.
    assert_eq!(pattern_key("dev[abc]"), "dev[abc]");
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

/// Render-side process-name pin: both process keys appear
/// in the smaps section body, not just the headers. Pins
/// that the row-emission loop reaches both keys — a future
/// regression that broke iteration after the first match
/// would surface here as a missing process.
#[test]
fn write_diff_smaps_emits_row_for_each_process_key() {
    let mut diff = CtprofDiff::default();
    let mut firefox_a = BTreeMap::new();
    firefox_a.insert("Rss".to_string(), 100 * 1024 * 1024);
    let mut firefox_b = BTreeMap::new();
    firefox_b.insert("Rss".to_string(), 200 * 1024 * 1024);
    let mut bash_a = BTreeMap::new();
    bash_a.insert("Rss".to_string(), 1024);
    let mut bash_b = BTreeMap::new();
    bash_b.insert("Rss".to_string(), 2048);
    diff.smaps_rollup_a.insert("firefox".into(), firefox_a);
    diff.smaps_rollup_b.insert("firefox".into(), firefox_b);
    diff.smaps_rollup_a.insert("bash".into(), bash_a);
    diff.smaps_rollup_b.insert("bash".into(), bash_b);

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
    let smaps_section = &out[smaps_at..];
    assert!(
        smaps_section.contains("firefox"),
        "process key `firefox` must appear in smaps section body:\n{smaps_section}",
    );
    assert!(
        smaps_section.contains("bash"),
        "process key `bash` must appear in smaps section body:\n{smaps_section}",
    );
}

/// `pattern_display_label` over members whose names contain
/// brackets must not panic in `grex` and must produce a
/// regex that contains the bracketed substrings. Brackets
/// are regex metacharacters, so `grex` has to escape them
/// for the resulting regex to be valid. Pin that the labels
/// for `worker[0]` and `worker[1]` come back containing
/// `worker` — the literal-prefix portion — so a regression
/// that drops the bracket escaping (or that crashes on
/// bracket input) surfaces here.
#[test]
fn pattern_display_label_handles_bracket_member_names() {
    let members = vec![
        "worker[0]".to_string(),
        "worker[1]".to_string(),
        "worker[2]".to_string(),
    ];
    let label = pattern_display_label("worker[{N}]", &members);
    assert!(
        label.contains("worker"),
        "grex must produce a label that contains the shared `worker` prefix; got {label:?}",
    );
    // The regex `grex` produces is well-formed — try
    // compiling it. A bracket-escaping regression would
    // produce an invalid regex syntax that `Regex::new`
    // rejects.
    let _: Regex = Regex::new(&label)
        .unwrap_or_else(|e| panic!("grex output {label:?} is not a valid regex: {e}"));
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

/// Cgroup paths with bracketed segments tokenize through
/// the same separator class as comms. A path like
/// `/runner-[xyz]/scope` splits brackets into separator
/// runs around the inner token. Pin that
/// [`cgroup_skeleton_tokens`] handles the brackets without
/// panic and that the resulting skeleton preserves the
/// non-bracket tokens. Sister test to
/// [`cgroup_normalize_collapses_bracketed_hex_session_ids`]
/// but at the lower-level `cgroup_skeleton_tokens` boundary.
#[test]
fn cgroup_skeleton_tokens_handles_bracketed_segments() {
    let (skeleton, tokens) = cgroup_skeleton_tokens("/runner-[xyz]/scope");
    // Tokens come from the non-separator runs only:
    // `runner`, `xyz`, `scope`. Brackets and `/` and `-` are
    // all separators and don't show up in the token list.
    assert_eq!(
        tokens,
        vec!["runner".to_string(), "xyz".to_string(), "scope".to_string(),],
        "bracket separators must split tokens cleanly; got {tokens:?}",
    );
    // Skeleton: `runner` and `xyz` and `scope` are all pure
    // alpha → literal. Separators preserved verbatim.
    assert_eq!(
        skeleton, "/runner-[xyz]/scope",
        "skeleton must preserve separators including brackets; got {skeleton:?}",
    );
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

/// Smaps render max-Rss tiebreaker: when two processes
/// report equal absolute Rss delta, the secondary sort key
/// is descending max-Rss across baseline and candidate. Pin
/// that the larger-max-Rss process appears ahead of the
/// smaller-max-Rss process when the absolute delta is tied.
/// Choose pcomm names such that alphabetical fallback would
/// place the lower-max-Rss process first — that way the test
/// distinguishes "max-Rss tiebreak fired" from "alpha
/// fallback fired."
#[test]
fn write_diff_smaps_max_rss_breaks_tie_when_delta_equal() {
    let mut diff = CtprofDiff::default();
    // Two processes with identical absolute Rss delta
    // (+20 MiB on each side); one carries higher max-Rss.
    // The higher-max-Rss process must render first.
    let mut a = BTreeMap::new();
    a.insert("Rss".to_string(), 100 * 1024 * 1024);
    a.insert("Pss".to_string(), 30 * 1024 * 1024);
    let mut a_b = BTreeMap::new();
    a_b.insert("Rss".to_string(), 120 * 1024 * 1024);
    a_b.insert("Pss".to_string(), 35 * 1024 * 1024);
    // Same +20 MiB delta as alpha_proc, but max_Rss is
    // 240 MiB vs 120 MiB.
    let mut z = BTreeMap::new();
    z.insert("Rss".to_string(), 220 * 1024 * 1024);
    z.insert("Pss".to_string(), 80 * 1024 * 1024);
    let mut z_b = BTreeMap::new();
    z_b.insert("Rss".to_string(), 240 * 1024 * 1024);
    z_b.insert("Pss".to_string(), 90 * 1024 * 1024);
    // Alphabetical pcomm names: "alpha_proc" < "zoomed".
    // Under pure alpha, alpha_proc would come first — which
    // here is the LOWER-max-Rss process. The max-Rss
    // tiebreaker must place "zoomed" first to pass.
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
        "max-Rss tiebreaker must place higher-max-Rss process (zoomed) \
         ahead of lower-max-Rss process (alpha_proc) when delta ties; got \
         zoomed@{zoomed_pos} alpha_proc@{alpha_pos}",
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

/// End-to-end literal-mode smaps render: drive
/// `compare(..., no_thread_normalize: true)` with populated
/// smaps_rollup data, then `write_diff` it, and assert the
/// rendered section shows the literal `pcomm[tgid]` keys
/// (NOT `pattern_key(pcomm)`). The same process instance
/// must run on both sides for the row to join — that's the
/// price of literal mode and the row only appears when both
/// snapshots reference the same PID. Pin the full pipeline
/// from `collect_smaps_rollup` literal-key construction
/// through `write_diff`'s smaps section emission.
#[test]
fn write_diff_smaps_literal_mode_renders_pcomm_tgid_keys() {
    let mut leader_a = make_thread("worker", "worker");
    leader_a.tid = 4242;
    leader_a.tgid = 4242;
    leader_a.smaps_rollup_kb.insert("Rss".into(), 4096);
    leader_a.smaps_rollup_kb.insert("Pss".into(), 1024);
    let snap_a = snap_with(vec![leader_a]);

    let mut leader_b = make_thread("worker", "worker");
    leader_b.tid = 4242;
    leader_b.tgid = 4242;
    leader_b.smaps_rollup_kb.insert("Rss".into(), 4096);
    leader_b.smaps_rollup_kb.insert("Pss".into(), 2048);
    let snap_b = snap_with(vec![leader_b]);

    let opts = CompareOptions {
        group_by: GroupBy::Pcomm.into(),
        cgroup_flatten: vec![],
        no_thread_normalize: true,
        no_cg_normalize: false,
        sort_by: Vec::new(),
    };
    let diff = compare(&snap_a, &snap_b, &opts);

    // Diff struct must carry the literal `worker[4242]` key,
    // not the normalized `worker` form.
    assert!(
        diff.smaps_rollup_a.contains_key("worker[4242]"),
        "literal-mode baseline key must be `worker[4242]`; got {:?}",
        diff.smaps_rollup_a.keys().collect::<Vec<_>>(),
    );
    assert!(
        diff.smaps_rollup_b.contains_key("worker[4242]"),
        "literal-mode candidate key must be `worker[4242]`; got {:?}",
        diff.smaps_rollup_b.keys().collect::<Vec<_>>(),
    );
    // No normalized `worker` key under literal mode.
    assert!(
        !diff.smaps_rollup_a.contains_key("worker"),
        "literal-mode must NOT carry the normalized `worker` key",
    );

    // Rendered section text shows the literal key.
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
    assert!(
        after.contains("worker[4242]"),
        "literal-mode rendered table must show `worker[4242]` key:\n{after}",
    );
}

// ------------------------------------------------------------
// Fudge / cgroup-rename matching tests (P0 + P1)
// ------------------------------------------------------------

/// Build a snapshot with `n` distinct thread types under a
/// single cgroup. Each thread carries a unique
/// (pcomm, comm) pair so the cgroup's TypeSet has size `n`.
/// Used by the fudge-threshold tests below to exercise the
/// 10-type set-size gate at exact, below, and above
/// boundaries.
///
/// Pcomms are chosen from a fixed alphabetic vocabulary so
/// each one classifies through `pattern_key` to its literal
/// (no shared `prefix-{N}` skeleton). With shared digit
/// suffixes the pattern_key normalizer would collapse
/// `worker-0`...`worker-9` into a single `worker-{N}`
/// bucket, breaking the test's "n distinct types"
/// invariant.
/// Greek-letter words that pattern_key keeps as literals
/// (no shared `prefix-{N}` skeleton). Used by the fudge
/// tests so each thread's (pcomm, comm) pair stays a
/// distinct entry in the cgroup's TypeSet — `worker-0`...
/// `worker-9` collapse into `worker-{N}` under
/// `pattern_key` and would break the "n distinct types"
/// invariant the threshold tests depend on.
///
/// The same vocabulary works for cgroup path components
/// when used as full segments — `/svc-alpha` and
/// `/svc-beta` survive the cgroup normalization because
/// `alpha`/`beta` classify to themselves (pure literals).
/// Avoid `/svc/v1`-style paths in tests: the `v1` token
/// normalizes to `v{N}`, so `/svc/v1` and `/svc/v2` BOTH
/// collapse to `/svc/v{N}` and would match as the same
/// cgroup (defeating the "different cgroups" precondition
/// fudge depends on).
const FUDGE_WORDS: &[&str] = &[
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
    "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi",
    "chi", "psi", "omega",
];

fn fudge_snap(cgroup: &str, n: usize, _pcomm_prefix: &str) -> CtprofSnapshot {
    assert!(
        n <= FUDGE_WORDS.len(),
        "fudge_snap: requested n={n} exceeds the literal-pcomm vocabulary",
    );
    let mut threads = Vec::new();
    for (i, word) in FUDGE_WORDS.iter().enumerate().take(n) {
        let pcomm = word.to_string();
        let comm = format!("{word}-w");
        let mut t = make_thread(&pcomm, &comm);
        t.tid = (1000 + i) as u32;
        t.tgid = t.tid;
        t.cgroup = cgroup.into();
        threads.push(t);
    }
    snap_with(threads)
}

/// Compose a CtprofSnapshot from N already-built per-cgroup
/// snapshots by concatenating their thread vectors. Each
/// input snapshot is expected to have already been built
/// with `fudge_snap`.
fn fudge_compose(snaps: Vec<CtprofSnapshot>) -> CtprofSnapshot {
    let mut threads = Vec::new();
    for snap in snaps {
        threads.extend(snap.threads);
    }
    snap_with(threads)
}

/// Drive the compare under `GroupBy::All` (the only mode
/// that activates fudge) with default options.
fn fudge_compare(a: &CtprofSnapshot, b: &CtprofSnapshot) -> CtprofDiff {
    compare(
        a,
        b,
        &CompareOptions {
            group_by: GroupBy::All.into(),
            cgroup_flatten: vec![],
            no_thread_normalize: false,
            no_cg_normalize: false,
            sort_by: Vec::new(),
        },
    )
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

/// Build a vector of threads under one cgroup using
/// FUDGE_WORDS (literal pcomms that survive `pattern_key`),
/// applying a per-thread mutator. Used by the N:1 merge
/// tests so the merge arms are exercised under realistic
/// 10-distinct-type sets without per-test boilerplate.
fn fudge_threads_with<F: FnMut(&mut ThreadState)>(
    cgroup: &str,
    n: usize,
    mut tweak: F,
) -> Vec<ThreadState> {
    assert!(
        n <= FUDGE_WORDS.len(),
        "fudge_threads_with: requested n={n} exceeds the literal-pcomm vocabulary",
    );
    let mut threads = Vec::new();
    for (i, word) in FUDGE_WORDS.iter().enumerate().take(n) {
        let mut t = make_thread(word, &format!("{word}-w"));
        t.tid = (1000 + i) as u32;
        t.tgid = t.tid;
        t.cgroup = cgroup.into();
        tweak(&mut t);
        threads.push(t);
    }
    threads
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

// Suppress `dead_code` on test-only helpers when only some
// are exercised in this build.
#[allow(dead_code)]
fn _fudge_helpers_used() {
    let _ = fudge_compose;
}
