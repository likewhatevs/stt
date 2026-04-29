//! Integration tests for the `ktstr host-state show` CLI
//! subcommand. Drives the binary entry point end-to-end:
//! write a synthetic snapshot to disk, invoke
//! `ktstr host-state show <path> [--group-by ...]` via
//! `assert_cmd`, and assert the rendered stdout carries the
//! expected columns / scaled values.
//!
//! The unit-level `write_show_*` tests in `src/bin/ktstr.rs`
//! exercise the renderer in isolation against an in-memory
//! `HostStateSnapshot`. This file covers the missing
//! disk-roundtrip + binary-spawn boundary: clap parses the argv,
//! `run_show` loads the snapshot from disk, the renderer emits
//! the table to stdout. A regression in any layer (clap shape,
//! load path, render path) surfaces here without requiring a
//! VM-backed capture.

// Shared host-state test helpers (`make_thread`, `snapshot`,
// `cgroup_stats_entry`) live under `tests/common/` so this
// file and `tests/host_state_compare.rs` share a single
// definition. The `mod common;` declaration pulls the
// subdirectory in as a non-test module.
mod common;

use std::collections::BTreeMap;

use assert_cmd::Command;

use common::host_state::{cgroup_stats_entry, make_thread, snapshot};
use ktstr::metric_types::{MonotonicCount, MonotonicNs};

fn ktstr() -> Command {
    Command::cargo_bin("ktstr").unwrap()
}

/// `ktstr host-state show <path>` with default flags renders
/// the standard `pcomm | threads | metric | value` columns and
/// surfaces the populated group key. Pins the binary entry
/// point's happy path through clap → run_show → write_show.
#[test]
fn show_default_renders_pcomm_grouping_columns() {
    let tmp = tempfile::tempdir().unwrap();
    let snap_path = tmp.path().join("snap.hst.zst");

    let mut t = make_thread("integration_proc", "worker");
    t.run_time_ns = MonotonicNs(5_000_000);
    t.nr_wakeups = MonotonicCount(200);
    snapshot(vec![t], BTreeMap::new())
        .write(&snap_path)
        .unwrap();

    let assert = ktstr()
        .args([
            "host-state",
            "show",
            snap_path.to_str().expect("ascii temp path"),
        ])
        .assert()
        .success();
    let out = String::from_utf8_lossy(&assert.get_output().stdout).to_string();

    // The default grouping (pcomm) renders the four-column
    // header and the populated group key.
    for col in ["pcomm", "threads", "metric", "value"] {
        assert!(out.contains(col), "missing column {col}:\n{out}");
    }
    assert!(
        out.contains("integration_proc"),
        "missing group key in output:\n{out}",
    );
    // run_time_ns: 5_000_000 ns auto-scales to 5.000ms via the
    // ns ladder (>= 1e6 → ms step). Pin the scaled cell to
    // confirm the renderer's value-cell formatter ran.
    assert!(
        out.contains("5.000ms"),
        "missing scaled run_time_ns cell:\n{out}",
    );
}

/// `ktstr host-state show <path> --group-by cgroup` renders the
/// secondary cgroup-stats enrichment table with auto-scaled
/// single-value cells (no `→` arrow, no `(+0)` zero-delta tail
/// — distinguishing show from compare). Pins both the show
/// entry point's `--group-by cgroup` routing AND the secondary
/// table's `format_scaled_u64` rendering at the disk-roundtrip
/// boundary.
#[test]
fn show_cgroup_grouping_renders_scaled_cgroup_stats() {
    let tmp = tempfile::tempdir().unwrap();
    let snap_path = tmp.path().join("snap.hst.zst");

    // Single thread under `/app` so the primary table has a
    // bucket; cgroup_stats populates the secondary table.
    let mut t = make_thread("worker", "w");
    t.cgroup = "/app".to_string();
    snapshot(
        vec![t],
        BTreeMap::from([(
            "/app".to_string(),
            // 1.5 s of CPU usage (1_500_000 µs → "1.500s") +
            // 1 GiB memory (1_073_741_824 → "1.000GiB").
            cgroup_stats_entry(1_500_000, 0, 0, 1024 * 1024 * 1024),
        )]),
    )
    .write(&snap_path)
    .unwrap();

    let assert = ktstr()
        .args([
            "host-state",
            "show",
            snap_path.to_str().expect("ascii temp path"),
            "--group-by",
            "cgroup",
        ])
        .assert()
        .success();
    let out = String::from_utf8_lossy(&assert.get_output().stdout).to_string();

    // Cgroup-stats secondary table headers.
    for col in [
        "cgroup",
        "cpu_usage_usec",
        "nr_throttled",
        "throttled_usec",
        "memory_current",
    ] {
        assert!(out.contains(col), "missing column {col}:\n{out}");
    }

    // Auto-scaled single-value cells. Distinguishes show from
    // compare's `a → b (+d)` triples.
    assert!(
        out.contains("1.500s"),
        "cpu_usage_usec 1_500_000µs must scale to '1.500s', got:\n{out}",
    );
    assert!(
        out.contains("1.000GiB"),
        "memory_current 1 GiB must scale to '1.000GiB', got:\n{out}",
    );
    // No arrow — show carries one snapshot, not a delta.
    assert!(
        !out.contains("→"),
        "show cgroup-stats must not emit a `→` arrow:\n{out}",
    );
    // No `(+0` tail — show does not pretend to carry a delta.
    assert!(
        !out.contains("(+0"),
        "show cgroup-stats must not carry a `(+0…)` zero-delta tail:\n{out}",
    );
}

/// `ktstr host-state show <path> --sort-by run_time_ns` ranks
/// groups by absolute aggregated value, descending. With three
/// pcomm buckets carrying different run_time_ns values, the
/// largest-value bucket appears first in stdout — distinct from
/// the alphabetical default. Pins the --sort-by clap arg
/// → parse_sort_by → write_show sort branch end-to-end.
#[test]
fn show_sort_by_orders_groups_by_metric_descending() {
    let tmp = tempfile::tempdir().unwrap();
    let snap_path = tmp.path().join("snap.hst.zst");

    let mut t_alpha = make_thread("alpha", "alpha-w");
    t_alpha.run_time_ns = MonotonicNs(100);
    let mut t_bravo = make_thread("bravo", "bravo-w");
    t_bravo.run_time_ns = MonotonicNs(500);
    let mut t_charlie = make_thread("charlie", "charlie-w");
    t_charlie.run_time_ns = MonotonicNs(250);
    snapshot(vec![t_alpha, t_bravo, t_charlie], BTreeMap::new())
        .write(&snap_path)
        .unwrap();

    let assert = ktstr()
        .args([
            "host-state",
            "show",
            snap_path.to_str().expect("ascii temp path"),
            "--sort-by",
            "run_time_ns",
        ])
        .assert()
        .success();
    let out = String::from_utf8_lossy(&assert.get_output().stdout).to_string();

    // Find the first occurrence of each group key. Under
    // descending run_time_ns sort: bravo (500) → charlie (250)
    // → alpha (100). Alphabetical default would be alpha first.
    let bravo_at = out.find("bravo").expect("bravo must surface in output");
    let charlie_at = out.find("charlie").expect("charlie must surface in output");
    let alpha_at = out.find("alpha").expect("alpha must surface in output");
    assert!(
        bravo_at < charlie_at,
        "sort_by run_time_ns must place bravo (500) before charlie (250); \
         alpha={alpha_at} bravo={bravo_at} charlie={charlie_at}\n{out}",
    );
    assert!(
        charlie_at < alpha_at,
        "sort_by run_time_ns must place charlie (250) before alpha (100); \
         alpha={alpha_at} bravo={bravo_at} charlie={charlie_at}\n{out}",
    );
}

/// `ktstr host-state show <path> --sort-by <bad>` exits non-zero
/// with a diagnostic naming the bad metric. Pins that
/// `parse_sort_by`'s rejection bubbles all the way out through
/// the binary's exit code — operator typos surface, not silent
/// fallthrough.
#[test]
fn show_invalid_sort_by_returns_error() {
    let tmp = tempfile::tempdir().unwrap();
    let snap_path = tmp.path().join("snap.hst.zst");
    snapshot(vec![make_thread("p", "w")], BTreeMap::new())
        .write(&snap_path)
        .unwrap();

    let assert = ktstr()
        .args([
            "host-state",
            "show",
            snap_path.to_str().expect("ascii temp path"),
            "--sort-by",
            "not_a_real_metric",
        ])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&assert.get_output().stderr).to_string();
    assert!(
        stderr.contains("not_a_real_metric"),
        "stderr must name the offending metric:\n{stderr}",
    );
}

/// `--sections primary` suppresses the derived-metrics
/// sub-table even when the snapshot would otherwise produce
/// derived rows. Pins the section-filter clap arg →
/// parse_sections → write_show sub-table-gating path
/// end-to-end. The default rendering carries `## Derived
/// metrics`; the filtered rendering must not.
#[test]
fn show_sections_primary_suppresses_derived_subtable() {
    let tmp = tempfile::tempdir().unwrap();
    let snap_path = tmp.path().join("snap.hst.zst");

    let mut t = make_thread("integration_proc", "worker");
    t.run_time_ns = MonotonicNs(5_000_000);
    t.wait_count = MonotonicCount(4);
    t.wait_sum = MonotonicNs(1_000_000);
    snapshot(vec![t], BTreeMap::new())
        .write(&snap_path)
        .unwrap();

    // Default rendering carries `## Derived metrics`.
    let default_assert = ktstr()
        .args([
            "host-state",
            "show",
            snap_path.to_str().expect("ascii temp path"),
        ])
        .assert()
        .success();
    let default_out = String::from_utf8_lossy(&default_assert.get_output().stdout).to_string();
    assert!(
        default_out.contains("## Derived metrics"),
        "default rendering must include the derived-metrics \
         heading; got:\n{default_out}",
    );

    // `--sections primary` suppresses the derived-metrics
    // sub-table.
    let filtered_assert = ktstr()
        .args([
            "host-state",
            "show",
            snap_path.to_str().expect("ascii temp path"),
            "--sections",
            "primary",
        ])
        .assert()
        .success();
    let filtered_out = String::from_utf8_lossy(&filtered_assert.get_output().stdout).to_string();
    assert!(
        !filtered_out.contains("## Derived metrics"),
        "--sections primary must suppress the derived-metrics \
         heading; got:\n{filtered_out}",
    );
    // The primary table still renders — `pcomm` is the
    // group-header column for the default GroupBy::Pcomm
    // grouping.
    assert!(
        filtered_out.contains("pcomm"),
        "--sections primary must still render the primary \
         table; got:\n{filtered_out}",
    );
}

/// `--sections derived` is the converse filter — keep ONLY
/// the derived sub-table, hide the primary one. Pins both
/// directions of the filter so a future regression that
/// inverted the gate (e.g. always rendered primary regardless
/// of filter) surfaces here.
#[test]
fn show_sections_derived_suppresses_primary_subtable() {
    let tmp = tempfile::tempdir().unwrap();
    let snap_path = tmp.path().join("snap.hst.zst");

    let mut t = make_thread("integration_proc", "worker");
    t.run_time_ns = MonotonicNs(5_000_000);
    t.wait_time_ns = MonotonicNs(1_000_000);
    t.wait_count = MonotonicCount(4);
    t.wait_sum = MonotonicNs(1_000_000);
    snapshot(vec![t], BTreeMap::new())
        .write(&snap_path)
        .unwrap();

    let assert = ktstr()
        .args([
            "host-state",
            "show",
            snap_path.to_str().expect("ascii temp path"),
            "--sections",
            "derived",
        ])
        .assert()
        .success();
    let out = String::from_utf8_lossy(&assert.get_output().stdout).to_string();
    // Derived heading present.
    assert!(
        out.contains("## Derived metrics"),
        "--sections derived must keep the derived heading; \
         got:\n{out}",
    );
    // Primary metric `run_time_ns` (registry-only, NOT a
    // derived metric name) absent — the derived sub-table
    // shares the same `pcomm` group-header column so
    // checking for `pcomm` does not distinguish "primary
    // suppressed" from "derived rendered". Instead, check
    // that no PRIMARY metric name surfaces. `run_time_ns` is
    // populated on the test thread, so its absence proves
    // the primary metric table is suppressed.
    assert!(
        !out.contains("run_time_ns"),
        "--sections derived must suppress the primary table \
         and its rows — `run_time_ns` is a primary-only \
         registry name, so its presence would prove the \
         primary table still rendered; got:\n{out}",
    );
    // Sanity: a derived metric IS rendered. cpu_efficiency
    // computes from run_time_ns + wait_time_ns, both
    // populated above, so the derived sub-table must surface
    // its name — pins that the gate didn't accidentally
    // suppress everything.
    assert!(
        out.contains("cpu_efficiency"),
        "--sections derived must keep the derived rows; \
         cpu_efficiency should surface (run_time_ns + \
         wait_time_ns are both populated): got:\n{out}",
    );
}

/// `--sections <unknown>` exits non-zero with a diagnostic
/// naming the bad section. Pins the `parse_sections` rejection
/// path end-to-end through clap → run_show → context.
#[test]
fn show_invalid_sections_returns_error() {
    let tmp = tempfile::tempdir().unwrap();
    let snap_path = tmp.path().join("snap.hst.zst");
    snapshot(vec![make_thread("p", "w")], BTreeMap::new())
        .write(&snap_path)
        .unwrap();

    let assert = ktstr()
        .args([
            "host-state",
            "show",
            snap_path.to_str().expect("ascii temp path"),
            "--sections",
            "not_a_real_section",
        ])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&assert.get_output().stderr).to_string();
    assert!(
        stderr.contains("not_a_real_section"),
        "stderr must name the offending section:\n{stderr}",
    );
}

/// `--wrap` is silently dropped when stdout is not a tty
/// (the binary-spawn case where stdout is captured into a
/// pipe), so the rendered byte sequence MUST match the
/// no-wrap default. Pins the documented contract that
/// awk/grep pipelines see the same output regardless of
/// whether `--wrap` was passed.
///
/// This is a behavioral pin, not a feature test for the
/// wrap renderer — that's exercised in the unit-level
/// `write_show_*` tests where the output goes into a String.
#[test]
fn show_wrap_does_not_change_byte_output_when_not_tty() {
    let tmp = tempfile::tempdir().unwrap();
    let snap_path = tmp.path().join("snap.hst.zst");

    let mut t = make_thread("integration_proc", "worker");
    t.run_time_ns = MonotonicNs(5_000_000);
    snapshot(vec![t], BTreeMap::new())
        .write(&snap_path)
        .unwrap();

    let no_wrap_out = ktstr()
        .args([
            "host-state",
            "show",
            snap_path.to_str().expect("ascii temp path"),
        ])
        .output()
        .expect("show without --wrap must execute");
    let with_wrap_out = ktstr()
        .args([
            "host-state",
            "show",
            snap_path.to_str().expect("ascii temp path"),
            "--wrap",
        ])
        .output()
        .expect("show with --wrap must execute");

    // Both invocations must succeed.
    assert!(
        no_wrap_out.status.success(),
        "no-wrap show must succeed; stderr:\n{}",
        String::from_utf8_lossy(&no_wrap_out.stderr),
    );
    assert!(
        with_wrap_out.status.success(),
        "with-wrap show must succeed; stderr:\n{}",
        String::from_utf8_lossy(&with_wrap_out.stderr),
    );
    // Bytes match — `--wrap` is a no-op when stdout is a pipe.
    assert_eq!(
        no_wrap_out.stdout, with_wrap_out.stdout,
        "--wrap must produce byte-identical output when stdout \
         is captured into a pipe (not a tty) — pins the \
         awk/grep-friendly contract documented on the flag",
    );
}

/// `--sections cgroup-stats` under `--group-by pcomm` (the
/// default) emits a stderr warning naming the requested
/// section AND the active group-by axis. The warning surfaces
/// at run_show time before the snapshot load, so the operator
/// sees it before any disk I/O completes. Pins the
/// `warn_cgroup_only_sections_under_non_cgroup` end-to-end
/// path.
#[test]
fn show_sections_cgroup_stats_under_pcomm_warns() {
    let tmp = tempfile::tempdir().unwrap();
    let snap_path = tmp.path().join("snap.hst.zst");
    snapshot(vec![make_thread("p", "w")], BTreeMap::new())
        .write(&snap_path)
        .unwrap();

    let assert = ktstr()
        .args([
            "host-state",
            "show",
            snap_path.to_str().expect("ascii temp path"),
            "--sections",
            "cgroup-stats",
        ])
        .assert()
        .success();
    let stderr = String::from_utf8_lossy(&assert.get_output().stderr).to_string();
    assert!(
        stderr.contains("cgroup-stats"),
        "stderr must name the cgroup-only section the \
         operator requested:\n{stderr}",
    );
    assert!(
        stderr.contains("--group-by"),
        "stderr must reference --group-by so the operator \
         understands the gate:\n{stderr}",
    );
}

/// `--metrics run_time_ns` keeps the named row and suppresses
/// every other primary row. Pins the row-level filter clap
/// arg → parse_metrics → write_show row-iteration gate
/// end-to-end. The default rendering surfaces multiple metric
/// rows (run_time_ns, wait_sum, voluntary_csw, etc.); the
/// filtered rendering must keep only the one named on the
/// flag.
#[test]
fn show_metrics_filter_keeps_named_row() {
    let tmp = tempfile::tempdir().unwrap();
    let snap_path = tmp.path().join("snap.hst.zst");

    let mut t = make_thread("integration_proc", "worker");
    t.run_time_ns = MonotonicNs(5_000_000);
    t.wait_sum = MonotonicNs(2_000_000);
    t.voluntary_csw = MonotonicCount(42);
    t.nr_wakeups = MonotonicCount(100);
    snapshot(vec![t], BTreeMap::new())
        .write(&snap_path)
        .unwrap();

    // Default rendering surfaces every metric in the registry
    // — including the named one and several others.
    let default_assert = ktstr()
        .args([
            "host-state",
            "show",
            snap_path.to_str().expect("ascii temp path"),
        ])
        .assert()
        .success();
    let default_out = String::from_utf8_lossy(&default_assert.get_output().stdout).to_string();
    assert!(
        default_out.contains("run_time_ns"),
        "default rendering must surface run_time_ns:\n{default_out}",
    );
    assert!(
        default_out.contains("wait_sum"),
        "default rendering must surface wait_sum (proves the \
         filter has work to do): \n{default_out}",
    );
    assert!(
        default_out.contains("voluntary_csw"),
        "default rendering must surface voluntary_csw:\n{default_out}",
    );

    // `--metrics run_time_ns` keeps the named row, suppresses
    // every other primary row.
    let filtered_assert = ktstr()
        .args([
            "host-state",
            "show",
            snap_path.to_str().expect("ascii temp path"),
            "--metrics",
            "run_time_ns",
        ])
        .assert()
        .success();
    let filtered_out = String::from_utf8_lossy(&filtered_assert.get_output().stdout).to_string();
    assert!(
        filtered_out.contains("run_time_ns"),
        "--metrics run_time_ns must keep the named row; \
         got:\n{filtered_out}",
    );
    assert!(
        !filtered_out.contains("wait_sum"),
        "--metrics run_time_ns must suppress wait_sum; \
         got:\n{filtered_out}",
    );
    assert!(
        !filtered_out.contains("voluntary_csw"),
        "--metrics run_time_ns must suppress voluntary_csw; \
         got:\n{filtered_out}",
    );
    assert!(
        !filtered_out.contains("nr_wakeups"),
        "--metrics run_time_ns must suppress nr_wakeups; \
         got:\n{filtered_out}",
    );
}

/// `--metrics <unknown>` exits non-zero with a diagnostic
/// naming the bad metric. Pins the `parse_metrics` rejection
/// path end-to-end through clap → run_show → context.
#[test]
fn show_invalid_metrics_returns_error() {
    let tmp = tempfile::tempdir().unwrap();
    let snap_path = tmp.path().join("snap.hst.zst");
    snapshot(vec![make_thread("p", "w")], BTreeMap::new())
        .write(&snap_path)
        .unwrap();

    let assert = ktstr()
        .args([
            "host-state",
            "show",
            snap_path.to_str().expect("ascii temp path"),
            "--metrics",
            "not_a_real_metric",
        ])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&assert.get_output().stderr).to_string();
    assert!(
        stderr.contains("not_a_real_metric"),
        "stderr must name the offending metric token:\n{stderr}",
    );
}
