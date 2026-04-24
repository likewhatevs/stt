//! Integration tests for the host-state comparison pipeline.
//!
//! These tests exercise the public surface of
//! [`ktstr::host_state`] and [`ktstr::host_state_compare`] as a
//! single unit: write two `HostStateSnapshot`s to disk, load
//! them back, run the compare, render the result, and verify
//! that the rendered output contains the expected columns and
//! non-zero deltas.
//!
//! The fixtures are synthetic (no VM) because the capture-side
//! implementation lands in parallel. Once `ktstr host-state -o`
//! is wired end-to-end, a VM-backed integration test will run
//! the same toy workload twice through real capture and
//! compare the real snapshots; that test is tracked separately.
//! This file establishes the compare-only pipeline so the
//! downstream addition is a drop-in extension rather than a
//! rewrite.

use std::collections::BTreeMap;
use std::path::Path;

use ktstr::host_state::{CgroupStats, HostStateSnapshot, ThreadState};
use ktstr::host_state_compare::{
    CompareOptions, GroupBy, HOST_STATE_METRICS, HostStateCompareArgs, compare, run_compare,
    write_diff,
};

fn make_thread(pcomm: &str, comm: &str) -> ThreadState {
    let mut t = ThreadState::default();
    t.tid = 1;
    t.tgid = 1;
    t.pcomm = pcomm.into();
    t.comm = comm.into();
    t.cgroup = "/".into();
    t.policy = "SCHED_OTHER".into();
    t.cpu_affinity = vec![0, 1, 2, 3];
    t
}

fn snapshot(
    threads: Vec<ThreadState>,
    cgroup_stats: BTreeMap<String, CgroupStats>,
) -> HostStateSnapshot {
    // `HostStateSnapshot` is `#[non_exhaustive]` so external
    // consumers cannot use struct-literal construction. Start
    // from `Default::default()` and mutate public fields — the
    // pattern documented in `lib.rs` / [`ktstr::non_exhaustive`].
    let mut snap = HostStateSnapshot::default();
    snap.threads = threads;
    snap.cgroup_stats = cgroup_stats;
    snap
}

fn cgroup_stats(cpu: u64, throttled: u64, throttled_usec: u64, memory: u64) -> CgroupStats {
    let mut s = CgroupStats::default();
    s.cpu_usage_usec = cpu;
    s.nr_throttled = throttled;
    s.throttled_usec = throttled_usec;
    s.memory_current = memory;
    s
}

fn compare_options(group_by: GroupBy, flatten: Vec<String>) -> CompareOptions {
    let mut opts = CompareOptions::default();
    opts.group_by = group_by.into();
    opts.cgroup_flatten = flatten;
    opts
}

/// Write two snapshots to disk under a shared tempdir, load
/// them via the public loader, run compare, render the result,
/// and verify that every stage composes correctly.
///
/// Baseline and candidate differ in `run_time_ns` and
/// `voluntary_csw` so the renderer's delta path is exercised;
/// they share `(pcomm, comm)` so the join matches them.
#[test]
fn full_pipeline_with_disk_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let baseline_path = tmp.path().join("baseline.hst.zst");
    let candidate_path = tmp.path().join("candidate.hst.zst");

    let mut ta = make_thread("integration_proc", "worker");
    ta.run_time_ns = 1_000_000;
    ta.voluntary_csw = 10;
    ta.nr_wakeups = 100;
    let mut tb = make_thread("integration_proc", "worker");
    tb.run_time_ns = 3_500_000;
    tb.voluntary_csw = 40;
    tb.nr_wakeups = 350;

    snapshot(vec![ta], BTreeMap::new())
        .write(&baseline_path)
        .unwrap();
    snapshot(vec![tb], BTreeMap::new())
        .write(&candidate_path)
        .unwrap();

    let loaded_a = HostStateSnapshot::load(&baseline_path).unwrap();
    let loaded_b = HostStateSnapshot::load(&candidate_path).unwrap();
    assert_eq!(loaded_a.threads.len(), 1);
    assert_eq!(loaded_b.threads.len(), 1);

    let diff = compare(&loaded_a, &loaded_b, &CompareOptions::default());
    assert!(diff.only_baseline.is_empty());
    assert!(diff.only_candidate.is_empty());

    // One row per registered metric for the single matched group.
    let proc_rows: Vec<_> = diff
        .rows
        .iter()
        .filter(|r| r.group_key == "integration_proc")
        .collect();
    assert_eq!(proc_rows.len(), HOST_STATE_METRICS.len());

    // Non-zero deltas survive the full pipeline.
    let run_time = proc_rows
        .iter()
        .find(|r| r.metric_name == "run_time_ns")
        .unwrap();
    assert_eq!(run_time.delta, Some(2_500_000.0));
    let csw = proc_rows
        .iter()
        .find(|r| r.metric_name == "voluntary_csw")
        .unwrap();
    assert_eq!(csw.delta, Some(30.0));
    let wakeups = proc_rows
        .iter()
        .find(|r| r.metric_name == "nr_wakeups")
        .unwrap();
    assert_eq!(wakeups.delta, Some(250.0));

    // Rendered output carries the expected columns and values.
    let mut out = String::new();
    write_diff(
        &mut out,
        &diff,
        &baseline_path,
        &candidate_path,
        GroupBy::Pcomm,
    )
    .unwrap();
    for col in [
        "pcomm",
        "threads",
        "metric",
        "baseline",
        "candidate",
        "delta",
        "%",
    ] {
        assert!(out.contains(col), "missing column {col}:\n{out}");
    }
    assert!(
        out.contains("integration_proc"),
        "missing group key in output:\n{out}",
    );
    assert!(
        out.contains("+2500000.000ns"),
        "missing run_time_ns delta in output:\n{out}",
    );
    assert!(
        out.contains("+250.0%"),
        "missing run_time_ns pct in output:\n{out}",
    );
}

/// Grouping by cgroup + flatten patterns collapses pods with
/// different IDs into one group. Validates the flatten
/// pipeline against a realistic kubepods-style cgroup path.
#[test]
fn cgroup_flatten_joins_pods_with_different_ids() {
    let tmp = tempfile::tempdir().unwrap();
    let a_path = tmp.path().join("a.hst.zst");
    let b_path = tmp.path().join("b.hst.zst");

    let mut ta = make_thread("app", "worker");
    ta.cgroup = "/kubepods/burstable/pod-AAA/container".into();
    ta.run_time_ns = 1_000;
    let mut cgroup_stats_a = BTreeMap::new();
    cgroup_stats_a.insert(
        "/kubepods/burstable/pod-AAA/container".into(),
        cgroup_stats(100, 0, 0, 1 << 20),
    );

    let mut tb = make_thread("app", "worker");
    tb.cgroup = "/kubepods/burstable/pod-BBB/container".into();
    tb.run_time_ns = 4_000;
    let mut cgroup_stats_b = BTreeMap::new();
    cgroup_stats_b.insert(
        "/kubepods/burstable/pod-BBB/container".into(),
        cgroup_stats(400, 0, 0, 2 << 20),
    );

    snapshot(vec![ta], cgroup_stats_a).write(&a_path).unwrap();
    snapshot(vec![tb], cgroup_stats_b).write(&b_path).unwrap();

    let loaded_a = HostStateSnapshot::load(&a_path).unwrap();
    let loaded_b = HostStateSnapshot::load(&b_path).unwrap();

    let opts = compare_options(
        GroupBy::Cgroup,
        vec!["/kubepods/burstable/*/container".into()],
    );
    let diff = compare(&loaded_a, &loaded_b, &opts);
    assert!(
        diff.only_baseline.is_empty() && diff.only_candidate.is_empty(),
        "flatten failed to join: only_a={:?} only_b={:?}",
        diff.only_baseline,
        diff.only_candidate,
    );
    let flat_key = "/kubepods/burstable/*/container";
    assert!(
        diff.rows.iter().any(|r| r.group_key == flat_key),
        "missing flattened key {flat_key}",
    );

    let mut out = String::new();
    write_diff(&mut out, &diff, &a_path, &b_path, GroupBy::Cgroup).unwrap();
    // Enrichment section renders under GroupBy::Cgroup and
    // carries the cgroup-level cpu.stat delta.
    assert!(
        out.contains("cpu_usage_usec"),
        "missing enrichment header:\n{out}",
    );
    assert!(out.contains("100"), "missing baseline cpu_usage:\n{out}");
    assert!(out.contains("400"), "missing candidate cpu_usage:\n{out}");
    assert!(out.contains("+300"), "missing cpu_usage delta:\n{out}");
}

/// Unmatched groups on one side produce an `only_baseline` /
/// `only_candidate` section in the rendered output with the
/// source-file path in the header. Validates that a
/// short-lived process that existed in one run but not the
/// other surfaces as a structural entry rather than a
/// silently-ignored row.
#[test]
fn unmatched_groups_render_with_source_path() {
    let tmp = tempfile::tempdir().unwrap();
    let a_path = tmp.path().join("a.hst.zst");
    let b_path = tmp.path().join("b.hst.zst");

    snapshot(vec![make_thread("only_a", "w")], BTreeMap::new())
        .write(&a_path)
        .unwrap();
    snapshot(vec![make_thread("only_b", "w")], BTreeMap::new())
        .write(&b_path)
        .unwrap();

    let loaded_a = HostStateSnapshot::load(&a_path).unwrap();
    let loaded_b = HostStateSnapshot::load(&b_path).unwrap();
    let diff = compare(&loaded_a, &loaded_b, &CompareOptions::default());
    assert_eq!(diff.only_baseline, vec!["only_a".to_string()]);
    assert_eq!(diff.only_candidate, vec!["only_b".to_string()]);

    let mut out = String::new();
    write_diff(&mut out, &diff, &a_path, &b_path, GroupBy::Pcomm).unwrap();
    assert!(out.contains("only in baseline"));
    assert!(out.contains("only in candidate"));
    assert!(out.contains(Path::new(&a_path).file_name().unwrap().to_str().unwrap()));
    assert!(out.contains(Path::new(&b_path).file_name().unwrap().to_str().unwrap()));
}

/// Malformed snapshot: the loader surfaces an error rather
/// than silently deserializing to an empty snapshot. This
/// check lives at the integration layer because the error
/// path has to survive the `std::fs::read` + `zstd::decode` +
/// `serde_json::from_slice` pipeline as a single flow.
#[test]
fn load_surfaces_error_on_malformed_snapshot() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("bad.hst.zst");
    std::fs::write(&path, b"not even zstd").unwrap();
    let err = HostStateSnapshot::load(&path).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("host-state") || msg.contains("zstd"),
        "error context missing source hint:\n{msg}",
    );
}

/// `run_compare` returns `Ok(0)` even when the comparison
/// produces a non-empty diff. Pin the doc contract at the top
/// of the function ("a non-empty diff is data, not a failure")
/// — a silent regression that made `Ok(0)` / `Ok(1)` depend on
/// whether `diff.rows.is_empty()` would turn routine comparison
/// into a spurious CI failure.
///
/// Two distinct snapshots are written to disk, `run_compare` is
/// driven through the full CLI surface (load → compare → print),
/// and the exit code is asserted against the documented contract.
/// Also covers the `--group-by=Pcomm` default path without
/// flatten patterns, mirroring the typical invocation shape.
#[test]
fn run_compare_returns_ok_zero_regardless_of_diff_emptiness() {
    let tmp = tempfile::tempdir().unwrap();
    let baseline_path = tmp.path().join("baseline.hst.zst");
    let candidate_path = tmp.path().join("candidate.hst.zst");

    let mut ta = make_thread("run_compare_proc", "w");
    ta.run_time_ns = 10_000;
    let mut tb = make_thread("run_compare_proc", "w");
    tb.run_time_ns = 50_000;

    snapshot(vec![ta], BTreeMap::new())
        .write(&baseline_path)
        .unwrap();
    snapshot(vec![tb], BTreeMap::new())
        .write(&candidate_path)
        .unwrap();

    let args = HostStateCompareArgs {
        baseline: baseline_path,
        candidate: candidate_path,
        group_by: GroupBy::Pcomm,
        cgroup_flatten: vec![],
    };
    // `run_compare` returns `anyhow::Result<i32>`. The contract
    // says the `i32` is always 0 for a successful load+compare,
    // regardless of whether any rows changed — interpretation is
    // left to the caller.
    let rc = run_compare(&args).expect("run_compare must succeed on valid snapshots");
    assert_eq!(
        rc, 0,
        "run_compare must return Ok(0) on a non-empty diff \
         (doc contract: 'a non-empty diff is data, not a failure'); \
         got Ok({rc})",
    );
}
