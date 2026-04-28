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
//! implementation lands in parallel. Once `ktstr host-state capture -o`
//! is wired end-to-end, a VM-backed integration test will run
//! the same toy workload twice through real capture and
//! compare the real snapshots; that test is tracked separately.
//! This file establishes the compare-only pipeline so the
//! downstream addition is a drop-in extension rather than a
//! rewrite.

// Shared host-state test helpers live under `tests/common/`
// so the same `make_thread` / `snapshot` / `cgroup_stats_entry`
// builders feed both this file and `tests/host_state_show.rs`
// without duplicating bodies. The `mod common;` declaration
// pulls the subdirectory in as a non-test module.
mod common;

use std::collections::BTreeMap;
use std::path::Path;

use ktstr::host_state::{HostStateSnapshot, ThreadState};
use ktstr::host_state_compare::{
    AggRule, Aggregated, CompareOptions, GroupBy, HOST_STATE_METRICS, HostStateCompareArgs,
    aggregate, compare, run_compare, write_diff,
};

use common::host_state::{cgroup_stats_entry, make_thread, snapshot};

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
/// they share both pcomm and comm so they group together
/// under any GroupBy axis.
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
    // run_time_ns delta: +2_500_000 ns → auto-scaled to
    // `+2.500ms` per the ns ladder in `auto_scale`.
    assert!(
        out.contains("+2.500ms"),
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
        cgroup_stats_entry(100, 0, 0, 1 << 20),
    );

    let mut tb = make_thread("app", "worker");
    tb.cgroup = "/kubepods/burstable/pod-BBB/container".into();
    tb.run_time_ns = 4_000;
    let mut cgroup_stats_b = BTreeMap::new();
    cgroup_stats_b.insert(
        "/kubepods/burstable/pod-BBB/container".into(),
        cgroup_stats_entry(400, 0, 0, 2 << 20),
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
        "flatten failed to group: only_a={:?} only_b={:?}",
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
    // Pin the contiguous scaled triple `cgroup_cell` emits for
    // cpu_usage_usec under the µs unit family. Both 100 and 400
    // sit below the 1000-µs ms-step threshold so each cell keeps
    // its base unit; delta +300 likewise. Bare `out.contains("100")`
    // would silently pass even if the baseline cell were dropped
    // entirely (the substring "100" appears elsewhere in the
    // surrounding format).
    assert!(
        out.contains("100µs → 400µs (+300µs)"),
        "missing contiguous scaled triple `100µs → 400µs (+300µs)`:\n{out}",
    );
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
        no_thread_normalize: false,
        no_cg_normalize: false,
        sort_by: String::new(),
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

/// `run_compare` with a non-empty `--sort-by` spec routes
/// through the multi-key sort path end-to-end: load → parse_sort_by
/// → compare → print. Pin that the integration surface (not just
/// the unit-level sort tested in `compare_uses_sort_by_when_set`)
/// accepts the spec and returns `Ok(0)` without bubbling a
/// parse error.
#[test]
fn run_compare_with_valid_sort_by_succeeds() {
    let tmp = tempfile::tempdir().unwrap();
    let baseline_path = tmp.path().join("baseline.hst.zst");
    let candidate_path = tmp.path().join("candidate.hst.zst");

    // Two distinct buckets so the multi-key sort actually has
    // groups to rank — single-bucket would short-circuit
    // through degenerate ordering.
    let mut a1 = make_thread("alpha", "w");
    a1.run_time_ns = 1_000;
    let mut a2 = make_thread("alpha", "w");
    a2.run_time_ns = 2_000;
    let mut b1 = make_thread("bravo", "w");
    b1.run_time_ns = 100;
    let mut b2 = make_thread("bravo", "w");
    b2.run_time_ns = 500;

    snapshot(vec![a1, b1], BTreeMap::new())
        .write(&baseline_path)
        .unwrap();
    snapshot(vec![a2, b2], BTreeMap::new())
        .write(&candidate_path)
        .unwrap();

    let args = HostStateCompareArgs {
        baseline: baseline_path,
        candidate: candidate_path,
        group_by: GroupBy::Pcomm,
        cgroup_flatten: vec![],
        no_thread_normalize: false,
        no_cg_normalize: false,
        // Multi-key spec exercising every parser feature
        // (case-insensitive direction, whitespace tolerance,
        // mixed asc/desc) so a regression in any of those
        // reaches the integration boundary.
        sort_by: "run_time_ns:DESC, wait_time_ns:asc".into(),
    };
    let rc = run_compare(&args).expect("run_compare must accept a valid --sort-by spec end-to-end");
    assert_eq!(rc, 0, "run_compare must return Ok(0) on success");
}

/// `run_compare` with an INVALID `--sort-by` spec returns `Err`
/// (not a panic, not silent fallthrough). Pins that
/// `parse_sort_by`'s rejection bubbles all the way out through
/// `run_compare`'s `with_context` chain — operator who typos a
/// metric name gets an actionable error rather than a confusing
/// no-op sort.
#[test]
fn run_compare_with_invalid_sort_by_returns_err() {
    let tmp = tempfile::tempdir().unwrap();
    let baseline_path = tmp.path().join("baseline.hst.zst");
    let candidate_path = tmp.path().join("candidate.hst.zst");

    let ta = make_thread("p", "w");
    let tb = make_thread("p", "w");
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
        no_thread_normalize: false,
        no_cg_normalize: false,
        // Unknown metric name — must surface the parser error,
        // not a silent best-effort sort.
        sort_by: "not_a_real_metric".into(),
    };
    let err = run_compare(&args).expect_err("invalid --sort-by must produce Err");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("not_a_real_metric"),
        "error must name the offending metric, got: {msg}",
    );
    // The `with_context` wrapper in `run_compare` adds the
    // `parse --sort-by ...` preamble — pin that the integration
    // path actually wraps the underlying anyhow error rather
    // than leaking the bare parser bail.
    assert!(
        msg.contains("parse --sort-by"),
        "error must carry the run_compare context preamble, got: {msg}",
    );
}

/// Exhaustive accessor pin for the entire HOST_STATE_METRICS
/// registry: every entry, regardless of its [`AggRule`] variant
/// (Sum, Max, OrdinalRange, Mode, Affinity), must read back
/// the exact field its accessor closure names.
///
/// The unit-level test
/// `sum_metric_accessors_read_expected_field` (in
/// src/host_state_compare.rs) only covers Sum metrics. This
/// integration-level pin extends the same idea to every variant
/// AND verifies (via set-equality against the registry below)
/// that no metric is missing from the local case table — the
/// table cannot drift relative to the registry without
/// surfacing.
///
/// Method per metric:
/// 1. Construct a single `ThreadState` populated with a UNIQUE,
///    distinguishable value for the metric's source field
///    (each Sum/Max metric gets a different u64 so an accessor
///    cross-wired to a sibling field reads the wrong value).
/// 2. Call `aggregate(rule, &[&t])`.
/// 3. Match the returned `Aggregated` variant against the rule
///    family and assert the value flowed through the accessor.
///
/// For Mode metrics (`policy`, `state`, `ext_enabled`) we set
/// the field to a distinct value and assert the mode is the
/// populated value with count 1. For Affinity (`cpu_affinity`)
/// we set a 5-element uniform cpuset distinct from
/// `make_thread`'s default and assert the aggregate carries
/// `min_cpus == max_cpus == 5` plus the `uniform` cpuset. For
/// OrdinalRange (`nice`, `processor`) we set the value and
/// assert both `min` and `max` equal it. For Sum and Max we
/// set the field to a unique u64 and assert the returned
/// scalar equals it.
#[test]
fn host_state_metrics_accessors_read_every_variant() {
    // Every registry name in this list paired with a setter
    // closure that populates the field the accessor reads.
    // Each Sum/Max setter writes a UNIQUE value so an accessor
    // cross-wired to a sibling field would read a value that
    // doesn't match the assertion. Set-equality below catches
    // missing or duplicate entries against the registry.
    //
    // Sum/Max value choice: each metric gets `100 + index_in_table`,
    // hand-paired with the per-metric expected value below.
    // We don't compute it programmatically because the closure
    // can't see its index — but the constant-per-metric pattern
    // means a future drift surfaces as a wrong-value assertion
    // rather than a false pass.
    type MetricSetter = fn(&mut ThreadState);
    let cases: &[(&str, MetricSetter)] = &[
        // Mode rules — distinct strings per metric.
        ("policy", |t| t.policy = "SCHED_RR".into()),
        ("state", |t| t.state = 'R'),
        ("ext_enabled", |t| t.ext_enabled = true),
        // OrdinalRange rules — distinct integers.
        ("nice", |t| t.nice = 7),
        ("processor", |t| t.processor = 5),
        // Affinity rule — 5-element uniform cpuset distinct
        // from `make_thread`'s default `vec![0, 1, 2, 3]`.
        ("cpu_affinity", |t| {
            t.cpu_affinity = vec![10, 11, 12, 13, 14]
        }),
        // Sum rules — each gets a unique u64. The expected
        // value table below mirrors these so a swap between
        // two accessors (run_time_ns ↔ wait_time_ns) would
        // surface as a numeric mismatch citing the metric.
        ("run_time_ns", |t| t.run_time_ns = 100),
        ("wait_time_ns", |t| t.wait_time_ns = 101),
        ("timeslices", |t| t.timeslices = 102),
        ("voluntary_csw", |t| t.voluntary_csw = 103),
        ("nonvoluntary_csw", |t| t.nonvoluntary_csw = 104),
        ("nr_wakeups", |t| t.nr_wakeups = 105),
        ("nr_wakeups_local", |t| t.nr_wakeups_local = 106),
        ("nr_wakeups_remote", |t| t.nr_wakeups_remote = 107),
        ("nr_wakeups_sync", |t| t.nr_wakeups_sync = 108),
        ("nr_wakeups_migrate", |t| t.nr_wakeups_migrate = 109),
        ("nr_wakeups_idle", |t| t.nr_wakeups_idle = 110),
        ("nr_wakeups_affine", |t| t.nr_wakeups_affine = 111),
        ("nr_wakeups_affine_attempts", |t| {
            t.nr_wakeups_affine_attempts = 112
        }),
        ("nr_migrations", |t| t.nr_migrations = 113),
        ("nr_migrations_cold", |t| t.nr_migrations_cold = 114),
        ("nr_forced_migrations", |t| t.nr_forced_migrations = 115),
        ("nr_failed_migrations_affine", |t| {
            t.nr_failed_migrations_affine = 116
        }),
        ("nr_failed_migrations_running", |t| {
            t.nr_failed_migrations_running = 117
        }),
        ("nr_failed_migrations_hot", |t| {
            t.nr_failed_migrations_hot = 118
        }),
        ("wait_sum", |t| t.wait_sum = 119),
        ("wait_count", |t| t.wait_count = 120),
        ("sleep_sum", |t| t.sleep_sum = 121),
        ("block_sum", |t| t.block_sum = 122),
        ("iowait_sum", |t| t.iowait_sum = 123),
        ("iowait_count", |t| t.iowait_count = 124),
        ("allocated_bytes", |t| t.allocated_bytes = 125),
        ("deallocated_bytes", |t| t.deallocated_bytes = 126),
        ("minflt", |t| t.minflt = 127),
        ("majflt", |t| t.majflt = 128),
        ("utime_clock_ticks", |t| t.utime_clock_ticks = 129),
        ("stime_clock_ticks", |t| t.stime_clock_ticks = 130),
        ("rchar", |t| t.rchar = 131),
        ("wchar", |t| t.wchar = 132),
        ("syscr", |t| t.syscr = 133),
        ("syscw", |t| t.syscw = 134),
        ("read_bytes", |t| t.read_bytes = 135),
        ("write_bytes", |t| t.write_bytes = 136),
        // Max rules — each gets a unique u64.
        ("wait_max", |t| t.wait_max = 200),
        ("sleep_max", |t| t.sleep_max = 201),
        ("block_max", |t| t.block_max = 202),
        ("exec_max", |t| t.exec_max = 203),
        ("slice_max", |t| t.slice_max = 204),
        // /proc/<tid>/stat additions (parse_stat).
        // priority: kernel-internal scheduler priority (signed,
        // OrdinalRange).
        ("priority", |t| t.priority = 25),
        // rt_priority: bounded real-time priority (OrdinalRange
        // — non-Sum because it's a bounded ordinal, not a
        // counter).
        ("rt_priority", |t| t.rt_priority = 50),
        // delayacct_blkio_ticks: cumulative counter (Sum).
        ("delayacct_blkio_ticks", |t| t.delayacct_blkio_ticks = 137),
        // /proc/<tid>/sched additions (parse_sched).
        // nr_wakeups_passive: counter (Sum) — known dead on
        // mainline but registered for forward compat.
        ("nr_wakeups_passive", |t| t.nr_wakeups_passive = 138),
        // core_forceidle_sum: counter (Sum, ns).
        ("core_forceidle_sum", |t| t.core_forceidle_sum = 139),
        // fair_slice_ns: current scheduler slice (Max, ns) —
        // distinct from slice_max which is the schedstat
        // high-water. Name mirrors the kernel `fair_policy()`
        // gate, which accepts SCHED_NORMAL/BATCH/EXT.
        ("fair_slice_ns", |t| t.fair_slice_ns = 250),
        // /proc/<tid>/status addition (parse_status).
        // nr_threads: tgid thread count, leader-only dedup. Max
        // surfaces "the largest process represented in this
        // bucket"; the row count already covers thread totals.
        ("nr_threads", |t| t.nr_threads = 140),
    ];

    // Hand-paired expected scalar value for each Sum/Max
    // metric — must match the setter's u64 above. A drift
    // between this map and the setter table surfaces as a
    // wrong-value assertion failure naming the metric.
    let expected_scalar: std::collections::BTreeMap<&str, u64> = [
        // Sum
        ("run_time_ns", 100),
        ("wait_time_ns", 101),
        ("timeslices", 102),
        ("voluntary_csw", 103),
        ("nonvoluntary_csw", 104),
        ("nr_wakeups", 105),
        ("nr_wakeups_local", 106),
        ("nr_wakeups_remote", 107),
        ("nr_wakeups_sync", 108),
        ("nr_wakeups_migrate", 109),
        ("nr_wakeups_idle", 110),
        ("nr_wakeups_affine", 111),
        ("nr_wakeups_affine_attempts", 112),
        ("nr_migrations", 113),
        ("nr_migrations_cold", 114),
        ("nr_forced_migrations", 115),
        ("nr_failed_migrations_affine", 116),
        ("nr_failed_migrations_running", 117),
        ("nr_failed_migrations_hot", 118),
        ("wait_sum", 119),
        ("wait_count", 120),
        ("sleep_sum", 121),
        ("block_sum", 122),
        ("iowait_sum", 123),
        ("iowait_count", 124),
        ("allocated_bytes", 125),
        ("deallocated_bytes", 126),
        ("minflt", 127),
        ("majflt", 128),
        ("utime_clock_ticks", 129),
        ("stime_clock_ticks", 130),
        ("rchar", 131),
        ("wchar", 132),
        ("syscr", 133),
        ("syscw", 134),
        ("read_bytes", 135),
        ("write_bytes", 136),
        ("delayacct_blkio_ticks", 137),
        ("nr_wakeups_passive", 138),
        ("core_forceidle_sum", 139),
        // Max
        ("wait_max", 200),
        ("sleep_max", 201),
        ("block_max", 202),
        ("exec_max", 203),
        ("slice_max", 204),
        ("fair_slice_ns", 250),
        ("nr_threads", 140),
    ]
    .into_iter()
    .collect();

    // Drive every case through aggregate() and assert the
    // accessor surfaced the populated field. The match is
    // structured per-variant so a regression that swapped a
    // metric's rule (e.g. Sum → Max) would surface as a panic
    // citing the metric name.
    for (name, set) in cases {
        let mut t = make_thread("p", "w");
        set(&mut t);
        let def = HOST_STATE_METRICS
            .iter()
            .find(|m| m.name == *name)
            .unwrap_or_else(|| panic!("metric {name} not in registry"));
        let agg = aggregate(def.rule, &[&t]);
        match (def.rule, &agg) {
            (AggRule::Sum(_), Aggregated::Sum(v)) => {
                let expected = *expected_scalar.get(name).unwrap_or_else(|| {
                    panic!("Sum metric {name} missing from expected_scalar table")
                });
                assert_eq!(
                    *v, expected,
                    "Sum accessor for {name} read the wrong field — \
                     got {v}, want {expected}",
                );
            }
            (AggRule::Max(_), Aggregated::Max(v)) => {
                let expected = *expected_scalar.get(name).unwrap_or_else(|| {
                    panic!("Max metric {name} missing from expected_scalar table")
                });
                assert_eq!(
                    *v, expected,
                    "Max accessor for {name} read the wrong field — \
                     got {v}, want {expected}",
                );
            }
            (AggRule::OrdinalRange(_), Aggregated::OrdinalRange { min, max }) => {
                let expected: i64 = match *name {
                    "nice" => 7,
                    "processor" => 5,
                    "priority" => 25,
                    "rt_priority" => 50,
                    _ => panic!("unexpected OrdinalRange metric {name}"),
                };
                assert_eq!(
                    *min, expected,
                    "OrdinalRange min for {name} did not read the populated value",
                );
                assert_eq!(
                    *max, expected,
                    "OrdinalRange max for {name} did not read the populated value",
                );
            }
            (
                AggRule::Mode(_),
                Aggregated::Mode {
                    value,
                    count,
                    total,
                },
            ) => {
                let expected: &str = match *name {
                    "policy" => "SCHED_RR",
                    "state" => "R",
                    "ext_enabled" => "true",
                    _ => panic!("unexpected Mode metric {name}"),
                };
                assert_eq!(
                    value, expected,
                    "Mode accessor for {name} did not read the populated value",
                );
                assert_eq!(
                    *count, 1,
                    "Mode count for single-thread aggregate must be 1"
                );
                assert_eq!(
                    *total, 1,
                    "Mode total for single-thread aggregate must be 1"
                );
            }
            (AggRule::Affinity(_), Aggregated::Affinity(s)) => {
                // The setter populates a 5-element cpuset
                // (10..=14) — distinct from `make_thread`'s
                // default 4-element default. A cross-wire to a
                // different field would yield a different
                // length OR cpuset.
                assert_eq!(
                    s.min_cpus, 5,
                    "Affinity min_cpus did not match populated cpuset len (5)",
                );
                assert_eq!(
                    s.max_cpus, 5,
                    "Affinity max_cpus did not match populated cpuset len (5)",
                );
                let cpus = s
                    .uniform
                    .as_ref()
                    .expect("single-thread Affinity must be uniform");
                assert_eq!(cpus, &vec![10, 11, 12, 13, 14]);
            }
            (rule, agg) => {
                panic!("rule/aggregate variant mismatch for {name}: rule={rule:?}, agg={agg:?}",)
            }
        }
    }

    // Exhaustiveness pin via SET-equality. count-equality
    // (cases.len() == HOST_STATE_METRICS.len()) would silently
    // pass on a duplicate test entry that displaced a missing
    // metric. Building name sets and asserting equality
    // catches both directions: a registry-added metric without
    // a test entry, AND a test-entry typo that doesn't exist
    // in the registry.
    let case_names: std::collections::BTreeSet<&str> =
        cases.iter().map(|(name, _)| *name).collect();
    let registry_names: std::collections::BTreeSet<&str> =
        HOST_STATE_METRICS.iter().map(|m| m.name).collect();
    assert_eq!(
        case_names.len(),
        cases.len(),
        "duplicate metric in test case table — set-vs-vec length mismatch",
    );
    assert_eq!(
        case_names, registry_names,
        "test cases must mirror HOST_STATE_METRICS exactly; \
         missing-from-cases or extra-in-cases delta surfaces here",
    );
}

/// Pin the `nr_threads` leader-dedup contract.
///
/// `capture_thread_at_with_tally` populates
/// [`ThreadState::nr_threads`] only when `tid == tgid` (the
/// thread leader); every non-leader thread of the same tgid
/// lands at zero. The registry pairs the field with
/// [`AggRule::Max`] so the rendered cell answers "the largest
/// process represented in this bucket" regardless of grouping
/// axis — the row count already covers thread totals.
///
/// This test fixes the contract empirically: build three
/// threads with the same tgid, populate `nr_threads = 3` ONLY
/// on the leader, and check that `aggregate` over the three
/// returns `Aggregated::Max(3)` — proving Max reads through to
/// the leader's value rather than to a follower's zero. A
/// regression that
///   - changed the registry rule back to `Sum` would surface
///     here as `Aggregated::Sum(3)` (wrong variant; assertion
///     fails on enum match), and
///   - silently broke the registry accessor (e.g. swapping
///     `t.nr_threads` for some other field) would surface as a
///     `Max(0)` because the followers' zero would dominate the
///     un-populated leader value.
///
/// The test is grouping-axis-independent: it calls `aggregate`
/// directly against `[&leader, &follower_a, &follower_b]` so
/// it pins the AGGREGATOR contract rather than the bucketing
/// path. Bucketing-path coverage lives in the rendered-output
/// integration tests.
#[test]
fn nr_threads_leader_dedup_aggregates_via_max_on_leader_value() {
    let mut leader = make_thread("server", "server-leader");
    leader.tid = 4242;
    leader.tgid = 4242;
    leader.nr_threads = 3;

    let mut follower_a = make_thread("server", "server-worker-1");
    follower_a.tid = 4243;
    follower_a.tgid = 4242;
    // Capture-side dedup: non-leader threads of the same tgid
    // land at zero. Setting it explicitly here makes the
    // contract obvious; relying on the Default would be
    // implicit and could mask a regression that flipped the
    // dedup direction (leader=0, followers=N).
    follower_a.nr_threads = 0;

    let mut follower_b = make_thread("server", "server-worker-2");
    follower_b.tid = 4244;
    follower_b.tgid = 4242;
    follower_b.nr_threads = 0;

    let def = HOST_STATE_METRICS
        .iter()
        .find(|m| m.name == "nr_threads")
        .expect("nr_threads metric must be in HOST_STATE_METRICS");

    // Sanity-pin the registry rule shape itself — a regression
    // that flipped Max → Sum would surface here BEFORE the
    // aggregate call, with a clearer message than the variant
    // mismatch below.
    assert!(
        matches!(def.rule, AggRule::Max(_)),
        "nr_threads must be registered as AggRule::Max — Sum is \
         wrong because non-leader threads contribute 0 (capture-\
         side leader-dedup), so a comm/cgroup bucket whose leader \
         lives elsewhere would render 0 under Sum",
    );

    let agg = aggregate(def.rule, &[&leader, &follower_a, &follower_b]);

    match agg {
        Aggregated::Max(v) => {
            assert_eq!(
                v, 3,
                "Max across [leader=3, follower=0, follower=0] must \
                 read the leader's value (3); a result of 0 means the \
                 followers' zeros displaced the leader OR the accessor \
                 read the wrong field",
            );
        }
        other => panic!(
            "nr_threads aggregator returned wrong variant — \
             expected Aggregated::Max(3), got {other:?}. A Sum-\
             shaped aggregate would silently produce Sum(3) here, \
             which is the regression this test guards against.",
        ),
    }
}

/// End-to-end smaps_rollup compare: stage two snapshots, each
/// with a leader thread carrying a populated `smaps_rollup_kb`
/// map. Run compare → write_diff and assert the rendered
/// output carries the `## smaps_rollup` header AND the
/// scaled-byte values for the keys that changed.
///
/// Pins the full pipeline: per-thread map → diff struct via
/// `collect_smaps_rollup` → `pcomm[tgid]` keying → kB→B
/// auto-scale → baseline → candidate cell rendering.
#[test]
fn compare_smaps_rollup_renders_header_and_scaled_byte_values() {
    use ktstr::host_state_compare::{CompareOptions, GroupBy, compare, write_diff};
    use std::path::Path;

    // Baseline thread carries Pss = 1024 kB (= 1 MiB);
    // candidate carries Pss = 2048 kB (= 2 MiB). Same tgid +
    // pcomm so the row collapses to ONE process key
    // `worker[4242]`.
    let mut baseline_leader = make_thread("worker", "worker");
    baseline_leader.tid = 4242;
    baseline_leader.tgid = 4242;
    baseline_leader.smaps_rollup_kb.insert("Rss".into(), 4096);
    baseline_leader.smaps_rollup_kb.insert("Pss".into(), 1024);

    let mut candidate_leader = make_thread("worker", "worker");
    candidate_leader.tid = 4242;
    candidate_leader.tgid = 4242;
    candidate_leader.smaps_rollup_kb.insert("Rss".into(), 4096);
    candidate_leader.smaps_rollup_kb.insert("Pss".into(), 2048);

    let baseline = snapshot(vec![baseline_leader], BTreeMap::new());
    let candidate = snapshot(vec![candidate_leader], BTreeMap::new());

    let opts = CompareOptions::default();
    let diff = compare(&baseline, &candidate, &opts);

    // Diff struct carries the per-process maps in BYTES (not kB
    // — the bytes conversion happens up-front in
    // collect_smaps_rollup so the renderer gets to skip it).
    assert_eq!(
        diff.smaps_rollup_a
            .get("worker[4242]")
            .and_then(|m| m.get("Pss").copied()),
        Some(1024 * 1024),
        "baseline Pss kB-to-bytes conversion landed in diff",
    );
    assert_eq!(
        diff.smaps_rollup_b
            .get("worker[4242]")
            .and_then(|m| m.get("Pss").copied()),
        Some(2048 * 1024),
        "candidate Pss kB-to-bytes conversion landed in diff",
    );

    let mut out = String::new();
    write_diff(
        &mut out,
        &diff,
        Path::new("a"),
        Path::new("b"),
        GroupBy::Pcomm,
    )
    .unwrap();

    assert!(
        out.contains("## smaps_rollup"),
        "smaps_rollup section header missing:\n{out}",
    );
    assert!(
        out.contains("worker[4242]"),
        "process key missing from rendered table:\n{out}",
    );
    assert!(
        out.contains("Pss"),
        "Pss row (changed) must appear in rendered table:\n{out}",
    );
    // Pss changed: 1 MiB → 2 MiB. The auto_scale "B" ladder
    // promotes a u64 byte count of 1 MiB or larger to MiB.
    assert!(
        out.contains("1.000MiB") && out.contains("2.000MiB"),
        "expected baseline 1 MiB and candidate 2 MiB cells:\n{out}",
    );
    // Rss didn't change (4096 kB on both sides) — must NOT
    // appear in the rendered table per the per-row gate.
    let header_pos = out.find("## smaps_rollup").unwrap();
    let after = &out[header_pos..];
    let next_section = after.find("\n## ").map(|p| p + 1).unwrap_or(after.len());
    let section = &after[..next_section];
    assert!(
        !section.contains("Rss"),
        "unchanged key (Rss: 4096 kB on both) must be suppressed in section:\n{section}",
    );
}

/// Section-gate test (F8): when every (process, key) pair has
/// equal baseline and candidate values, the entire
/// `## smaps_rollup` header is suppressed (not just per-row
/// rows). Pins the `any_delta` precheck.
#[test]
fn compare_smaps_rollup_suppresses_section_when_all_unchanged() {
    use ktstr::host_state_compare::{CompareOptions, GroupBy, compare, write_diff};
    use std::path::Path;

    let mut leader = make_thread("worker", "worker");
    leader.tid = 4242;
    leader.tgid = 4242;
    leader.smaps_rollup_kb.insert("Rss".into(), 4096);
    leader.smaps_rollup_kb.insert("Pss".into(), 1024);

    // Baseline AND candidate both carry IDENTICAL smaps_rollup
    // values — every (process, key) pair is unchanged.
    let baseline = snapshot(vec![leader.clone()], BTreeMap::new());
    let candidate = snapshot(vec![leader], BTreeMap::new());

    let opts = CompareOptions::default();
    let diff = compare(&baseline, &candidate, &opts);

    let mut out = String::new();
    write_diff(
        &mut out,
        &diff,
        Path::new("a"),
        Path::new("b"),
        GroupBy::Pcomm,
    )
    .unwrap();

    assert!(
        !out.contains("## smaps_rollup"),
        "section header must be suppressed when no key changed:\n{out}",
    );
}
