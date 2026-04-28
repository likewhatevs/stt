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
