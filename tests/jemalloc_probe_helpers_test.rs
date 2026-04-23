//! Host-side unit tests for the `PayloadMetrics`-scanning helpers
//! used in `tests/jemalloc_probe_tests.rs`.
//!
//! These tests DO NOT boot a VM — they exercise the pure-data
//! flat-metric lookup helpers (`find_metric`, `has_metric`,
//! `count_indexed_metrics`) plus the `ThreadLookup::ExceedsCap`
//! saturated-scan case. The helpers live inline in
//! `jemalloc_probe_tests.rs` because they close over file-private
//! names (`PayloadMetrics`, `Metric`) and the probe-specific key
//! conventions; duplicating the small helper bodies here lets the
//! unit tests live in a clean-slate integration-test file without
//! the ktstr early-dispatch ctor hiding `#[test]` fns behind the
//! `--list` intercept (see the module doc on
//! `jemalloc_probe_tests.rs` for the intercept rationale).

use ktstr::test_support::{Metric, MetricSource, MetricStream, PayloadMetrics, Polarity};

// --- Helpers (mirrors of the inline helpers in `jemalloc_probe_tests.rs`) ---

fn find_metric<'a>(metrics: &'a PayloadMetrics, key: &str) -> Option<&'a Metric> {
    metrics.metrics.iter().find(|m| m.name == key)
}

fn has_metric(metrics: &PayloadMetrics, key: &str) -> bool {
    find_metric(metrics, key).is_some()
}

fn count_indexed_metrics<F>(metrics: &PayloadMetrics, cap: usize, key_fn: F) -> usize
where
    F: Fn(usize) -> String,
{
    let mut n = 0;
    for i in 0..cap {
        if has_metric(metrics, &key_fn(i)) {
            n += 1;
        } else {
            break;
        }
    }
    n
}

// --- Mirror of the ThreadLookup shape + lookup used in probe tests ---

#[derive(Debug, PartialEq, Eq)]
enum ThreadLookup {
    Found { allocated_bytes: u64 },
    MissingAllocatedBytes,
    TidAbsent,
    ExceedsCap,
}

fn lookup_thread(metrics: &PayloadMetrics, worker_tid: i32, cap: usize) -> ThreadLookup {
    let worker_tid_f64 = worker_tid as f64;
    for i in 0..cap {
        let tid_key = format!("snapshots.0.threads.{i}.tid");
        let tid_m = match find_metric(metrics, &tid_key) {
            Some(m) => m,
            None => return ThreadLookup::TidAbsent,
        };
        if tid_m.value == worker_tid_f64 {
            let alloc_key = format!("snapshots.0.threads.{i}.allocated_bytes");
            return match find_metric(metrics, &alloc_key).map(|m| m.value as u64) {
                Some(v) => ThreadLookup::Found { allocated_bytes: v },
                None => ThreadLookup::MissingAllocatedBytes,
            };
        }
    }
    ThreadLookup::ExceedsCap
}

// --- Fixture builders ---

fn metric(name: &str, value: f64) -> Metric {
    Metric {
        name: name.to_string(),
        value,
        polarity: Polarity::Unknown,
        unit: String::new(),
        source: MetricSource::Json,
        stream: MetricStream::Stdout,
    }
}

fn payload_with(metrics: Vec<Metric>) -> PayloadMetrics {
    PayloadMetrics {
        metrics,
        exit_code: 0,
    }
}

// --- find_metric tests ---

#[test]
fn find_metric_returns_matching_entry_by_exact_name() {
    let p = payload_with(vec![
        metric("snapshots.0.threads.0.tid", 1234.0),
        metric("snapshots.0.threads.0.allocated_bytes", 4096.0),
    ]);
    let m = find_metric(&p, "snapshots.0.threads.0.allocated_bytes")
        .expect("metric must be found by exact name");
    assert_eq!(m.value, 4096.0);
}

#[test]
fn find_metric_returns_none_for_absent_key() {
    let p = payload_with(vec![metric("snapshots.0.threads.0.tid", 1234.0)]);
    assert!(find_metric(&p, "snapshots.0.threads.0.not_a_real_key").is_none());
}

#[test]
fn find_metric_is_exact_match_not_prefix() {
    // `tid` should not match `tid_extra` or similar. A prefix-match
    // bug would cause `find_metric` to return the wrong metric and
    // silently confuse downstream lookups.
    let p = payload_with(vec![metric("snapshots.0.threads.0.tid_extra", 42.0)]);
    assert!(find_metric(&p, "snapshots.0.threads.0.tid").is_none());
}

// --- has_metric tests ---

#[test]
fn has_metric_true_on_present_key() {
    let p = payload_with(vec![metric("snapshots.0.threads.0.tid", 1234.0)]);
    assert!(has_metric(&p, "snapshots.0.threads.0.tid"));
}

#[test]
fn has_metric_false_on_absent_key() {
    let p = payload_with(vec![metric("snapshots.0.threads.0.tid", 1234.0)]);
    assert!(!has_metric(&p, "snapshots.99.threads.99.tid"));
}

#[test]
fn has_metric_false_on_empty_payload() {
    let p = payload_with(vec![]);
    assert!(!has_metric(&p, "any_key"));
}

// --- count_indexed_metrics tests ---

#[test]
fn count_indexed_metrics_stops_at_first_miss() {
    // Indices 0, 1, 2 present; index 3 missing. Walk should return
    // 3 and not continue past the gap — matches the probe's
    // `walk_json_leaves` contiguous-index contract.
    let p = payload_with(vec![
        metric("snapshots.0.threads.0.tid", 1.0),
        metric("snapshots.0.threads.1.tid", 2.0),
        metric("snapshots.0.threads.2.tid", 3.0),
        // No index-3 entry.
        metric("snapshots.0.threads.4.tid", 5.0), // Gap + later entry
    ]);
    let n = count_indexed_metrics(&p, 1024, |i| format!("snapshots.0.threads.{i}.tid"));
    assert_eq!(
        n, 3,
        "count must stop at the first gap; a later-indexed entry \
         past the gap must not be counted",
    );
}

#[test]
fn count_indexed_metrics_saturates_at_cap() {
    // Every index 0..cap present — function returns cap, not cap+1
    // (the cap is exclusive upper bound via `for i in 0..cap`).
    let metrics: Vec<Metric> = (0..10)
        .map(|i| metric(&format!("x.{i}"), i as f64))
        .collect();
    let p = payload_with(metrics);
    assert_eq!(
        count_indexed_metrics(&p, 10, |i| format!("x.{i}")),
        10,
        "cap is exclusive and every index below it is present",
    );
}

#[test]
fn count_indexed_metrics_returns_zero_when_first_index_absent() {
    // A completely-missing array yields 0 without walking the cap.
    let p = payload_with(vec![metric("unrelated.key", 7.0)]);
    assert_eq!(
        count_indexed_metrics(&p, 1024, |i| format!("missing.{i}.tid")),
        0,
    );
}

// --- ThreadLookup::ExceedsCap regression guard ---

#[test]
fn lookup_thread_returns_exceeds_cap_on_saturated_scan() {
    // Scenario: the scan runs every index 0..CAP and finds a tid
    // entry at each one, but none matches the worker_tid. The loop
    // exits normally at the end of the range, and the function must
    // return `ExceedsCap` — NOT `TidAbsent`. The two outcomes are
    // semantically distinct: `TidAbsent` means "the contiguous tid
    // array terminated, and worker_tid was not in it", while
    // `ExceedsCap` means "the scan ran out of cap before seeing a
    // terminator, so worker_tid may live at a later index and the
    // lookup is inconclusive".
    const CAP: usize = 4;
    const WORKER_TID: i32 = 999;
    let metrics: Vec<Metric> = (0..CAP)
        .flat_map(|i| {
            vec![
                metric(&format!("snapshots.0.threads.{i}.tid"), (i as f64) + 100.0),
                metric(
                    &format!("snapshots.0.threads.{i}.allocated_bytes"),
                    1024.0,
                ),
            ]
        })
        .collect();
    let p = payload_with(metrics);
    let result = lookup_thread(&p, WORKER_TID, CAP);
    assert_eq!(
        result,
        ThreadLookup::ExceedsCap,
        "saturated scan must return ExceedsCap, not TidAbsent",
    );
}

#[test]
fn lookup_thread_returns_tid_absent_when_array_terminates_early() {
    // Companion to the ExceedsCap test: when the tid array
    // terminates BEFORE the cap, a missing worker_tid surfaces as
    // `TidAbsent`. Pins the distinction between the two outcomes.
    const CAP: usize = 10;
    let p = payload_with(vec![
        metric("snapshots.0.threads.0.tid", 100.0),
        metric("snapshots.0.threads.0.allocated_bytes", 512.0),
        metric("snapshots.0.threads.1.tid", 200.0),
        metric("snapshots.0.threads.1.allocated_bytes", 512.0),
        // No snapshots.0.threads.2.* — terminator at index 2.
    ]);
    assert_eq!(lookup_thread(&p, 999, CAP), ThreadLookup::TidAbsent);
}

#[test]
fn lookup_thread_returns_missing_allocated_bytes_on_err_thread() {
    // Probe's Err arm emits `tid` without a sibling
    // `allocated_bytes` — `walk_json_leaves` drops the string
    // `error` field. lookup_thread must surface
    // `MissingAllocatedBytes` so the caller can distinguish a probe
    // error on this thread from a clean absent tid.
    const CAP: usize = 10;
    let p = payload_with(vec![metric("snapshots.0.threads.0.tid", 999.0)]);
    assert_eq!(
        lookup_thread(&p, 999, CAP),
        ThreadLookup::MissingAllocatedBytes,
    );
}

#[test]
fn lookup_thread_returns_found_on_happy_path() {
    const CAP: usize = 10;
    let p = payload_with(vec![
        metric("snapshots.0.threads.0.tid", 999.0),
        metric("snapshots.0.threads.0.allocated_bytes", 16_777_216.0),
    ]);
    assert_eq!(
        lookup_thread(&p, 999, CAP),
        ThreadLookup::Found {
            allocated_bytes: 16_777_216,
        },
    );
}
