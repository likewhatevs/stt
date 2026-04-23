//! Flat-metric lookup helpers for the jemalloc-probe integration
//! tests.
//!
//! Lives in the crate (not inline in `tests/jemalloc_probe_tests.rs`)
//! so the logic is reachable from `#[cfg(test)]` unit tests. The
//! probe test binary registers `#[ktstr_test]` entries, which
//! activates the early-dispatch ctor's `--list` intercept — any
//! plain `#[test]` fn declared in that binary is hidden from
//! nextest's discovery (see the long comment at the head of
//! `tests/jemalloc_alloc_worker_exit_codes.rs`). Moving the
//! helpers here lets the ExceedsCap branch and friends be
//! pinned by lib-crate unit tests that run under `cargo nextest
//! run --lib` without the ctor path.
//!
//! All items are `pub` so the integration test file at
//! `tests/jemalloc_probe_tests.rs` can import them through the
//! `test_support` re-export surface.

use super::payload::{Metric, PayloadMetrics};

/// Outcome of scanning the flat metric list for a tid-keyed thread
/// entry. Distinguishes "tid not present" from "tid present but
/// `allocated_bytes` missing" AND from "probe emitted more than
/// [`MAX_SCAN_INDEX`] contiguous threads without the caller's
/// tid appearing in the prefix" — so a caller can issue a precise
/// diagnostic instead of a blanket "not found".
pub enum ThreadLookup {
    /// `snapshots.{snap_idx}.threads.N.tid == worker_tid` and
    /// `snapshots.{snap_idx}.threads.N.allocated_bytes` are both
    /// present. Returns the observed counter plus the companion
    /// `deallocated_bytes` (if emitted).
    Found {
        allocated_bytes: u64,
        deallocated_bytes: Option<u64>,
    },
    /// Probe emitted a `snapshots.{snap_idx}.threads.N.tid` matching
    /// `worker_tid`, but no `snapshots.{snap_idx}.threads.N.allocated_bytes`
    /// sibling. The probe hit an error on that thread — typically
    /// an `error` entry replaces the counter fields.
    MissingAllocatedBytes,
    /// No `snapshots.{snap_idx}.threads.N.tid == worker_tid` entry in
    /// the flat metric list. Probe did not visit the worker at all.
    TidAbsent,
    /// The flat metric list contained at least [`MAX_SCAN_INDEX`]
    /// contiguous `snapshots.{snap_idx}.threads.N.tid` entries, none
    /// of which matched `worker_tid`, and the scan hit the cap
    /// before reaching the array terminator. The worker's tid may
    /// exist at a later index and be invisible to the scan.
    /// Distinct from `TidAbsent` — this outcome means the lookup is
    /// inconclusive, not that the probe definitively skipped the
    /// worker.
    ExceedsCap,
}

/// Safety bound on the `snapshots.*.threads.N.tid` scan in
/// [`lookup_thread`], [`snapshot_worker_allocated`], [`thread_count`],
/// and [`snapshot_count`]. Realistic probe runs see at most a few
/// dozen threads in a single-allocator worker process; hitting this
/// cap indicates either an unexpectedly wide target or a flat-metric
/// schema change that broke the terminator convention.
pub const MAX_SCAN_INDEX: usize = 1024;

/// Find a metric by exact name. Returns `None` if absent.
pub fn find_metric<'a>(metrics: &'a PayloadMetrics, key: &str) -> Option<&'a Metric> {
    metrics.metrics.iter().find(|m| m.name == key)
}

/// Does the flat metric list contain a metric with this exact name?
/// Thin wrapper around [`find_metric`] for the common existence
/// check — avoids forcing every call site to spell `.is_some()`.
pub fn has_metric(metrics: &PayloadMetrics, key: &str) -> bool {
    find_metric(metrics, key).is_some()
}

/// Fetch a metric by exact name and return its numeric value as a
/// `u64`. Returns `None` if the metric is absent. Thin wrapper around
/// [`find_metric`] + `value as u64` for the common numeric-lookup
/// shape. Integer metrics that originated as JSON numbers round-trip
/// through `f64` without loss up to 2^53 — every counter in the
/// probe's output sits well below that bound.
pub fn metric_u64(metrics: &PayloadMetrics, key: &str) -> Option<u64> {
    find_metric(metrics, key).map(|m| m.value as u64)
}

/// Walk `0..cap` applying `key_fn(i)` to form a metric name and
/// count how many consecutive indices yield a present metric.
/// Stops at the first miss — the probe's `walk_json_leaves`
/// flattening yields indices 0..N contiguously, so the first gap is
/// the array terminator. Returns the count, which may be `cap` if
/// every index below the bound is present (inconclusive — the
/// caller should treat `cap` as "saturated scan, real count may be
/// larger").
pub fn count_indexed_metrics<F>(metrics: &PayloadMetrics, cap: usize, key_fn: F) -> usize
where
    F: Fn(usize) -> String,
{
    let mut n = 0;
    for i in 0..cap {
        if find_metric(metrics, &key_fn(i)).is_some() {
            n += 1;
        } else {
            break;
        }
    }
    n
}

/// Extract the `allocated_bytes` / `deallocated_bytes` values for
/// `worker_tid` from snapshot 0 in the flat metric list produced by
/// `walk_json_leaves` over the probe's JSON output.
///
/// The probe emits
/// `{"pid":P,"snapshots":[{"timestamp_unix_sec":T,"threads":[{"tid":T,"allocated_bytes":A,"deallocated_bytes":D,...}, ...]}, ...]}`
/// which `walk_json_leaves` flattens per array index into contiguous
/// keys `snapshots.0.threads.0.tid`, `snapshots.0.threads.1.tid`, …
/// with no gaps. The scan below stops at the first
/// `snapshots.0.threads.N.tid` miss, which is the natural array
/// terminator, and returns [`ThreadLookup::TidAbsent`] in that case.
/// If the scan instead runs the full [`MAX_SCAN_INDEX`]
/// iterations without hitting the terminator AND without matching
/// `worker_tid`, it returns [`ThreadLookup::ExceedsCap`] to make the
/// inconclusive outcome visible to the caller (the tid may exist
/// past the cap).
pub fn lookup_thread(metrics: &PayloadMetrics, worker_tid: i32) -> ThreadLookup {
    let worker_tid_f64 = worker_tid as f64;
    for i in 0..MAX_SCAN_INDEX {
        let tid_key = format!("snapshots.0.threads.{i}.tid");
        let tid_m = match find_metric(metrics, &tid_key) {
            Some(m) => m,
            None => return ThreadLookup::TidAbsent,
        };
        if tid_m.value == worker_tid_f64 {
            let alloc_key = format!("snapshots.0.threads.{i}.allocated_bytes");
            let dealloc_key = format!("snapshots.0.threads.{i}.deallocated_bytes");
            let allocated_bytes = match find_metric(metrics, &alloc_key).map(|m| m.value as u64) {
                Some(v) => v,
                None => return ThreadLookup::MissingAllocatedBytes,
            };
            let deallocated_bytes = find_metric(metrics, &dealloc_key).map(|m| m.value as u64);
            return ThreadLookup::Found {
                allocated_bytes,
                deallocated_bytes,
            };
        }
    }
    // Loop ran to completion — every one of 0..MAX_SCAN_INDEX
    // had a tid entry, and none matched. A contiguous-array
    // terminator would have early-returned `TidAbsent`, so the cap
    // was hit with data remaining. Surface the inconclusive outcome
    // distinctly from genuine absence.
    ThreadLookup::ExceedsCap
}

/// Extract `snapshots.{snap_idx}.threads[*].allocated_bytes` for the
/// thread whose tid matches `worker_tid`. Returns [`ThreadLookup`]
/// so callers distinguish "tid absent" from "cap hit before tid
/// seen" from "allocated_bytes sibling missing" — parallel to
/// [`lookup_thread`], which covers the single-snapshot path.
pub fn snapshot_worker_allocated(
    metrics: &PayloadMetrics,
    snap_idx: usize,
    worker_tid: i32,
) -> ThreadLookup {
    let worker_tid_f64 = worker_tid as f64;
    for j in 0..MAX_SCAN_INDEX {
        let tid_key = format!("snapshots.{snap_idx}.threads.{j}.tid");
        let tid_m = match find_metric(metrics, &tid_key) {
            Some(m) => m,
            None => return ThreadLookup::TidAbsent,
        };
        if tid_m.value == worker_tid_f64 {
            let alloc_key = format!("snapshots.{snap_idx}.threads.{j}.allocated_bytes");
            let dealloc_key = format!("snapshots.{snap_idx}.threads.{j}.deallocated_bytes");
            let allocated_bytes = match find_metric(metrics, &alloc_key).map(|m| m.value as u64) {
                Some(v) => v,
                None => return ThreadLookup::MissingAllocatedBytes,
            };
            let deallocated_bytes = find_metric(metrics, &dealloc_key).map(|m| m.value as u64);
            return ThreadLookup::Found {
                allocated_bytes,
                deallocated_bytes,
            };
        }
    }
    ThreadLookup::ExceedsCap
}

/// Count the number of `snapshots.0.threads.N.tid` entries in the
/// flat metric list, capped at [`MAX_SCAN_INDEX`].
pub fn thread_count(metrics: &PayloadMetrics) -> usize {
    count_indexed_metrics(metrics, MAX_SCAN_INDEX, |i| {
        format!("snapshots.0.threads.{i}.tid")
    })
}

/// Count the number of `snapshots.N.timestamp_unix_sec` entries in
/// the flat metric list, capped at [`MAX_SCAN_INDEX`].
pub fn snapshot_count(metrics: &PayloadMetrics) -> usize {
    count_indexed_metrics(metrics, MAX_SCAN_INDEX, |i| {
        format!("snapshots.{i}.timestamp_unix_sec")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metric(name: &str, value: f64) -> Metric {
        use super::super::payload::{MetricSource, MetricStream, Polarity};
        Metric {
            name: name.to_owned(),
            value,
            polarity: Polarity::Unknown,
            unit: String::new(),
            source: MetricSource::Json,
            stream: MetricStream::Stdout,
        }
    }

    fn empty_payload() -> PayloadMetrics {
        PayloadMetrics {
            metrics: Vec::new(),
            exit_code: 0,
        }
    }

    fn push_tid(metrics: &mut PayloadMetrics, idx: usize, tid: f64) {
        metrics
            .metrics
            .push(metric(&format!("snapshots.0.threads.{idx}.tid"), tid));
    }

    fn push_alloc(metrics: &mut PayloadMetrics, idx: usize, alloc: f64) {
        metrics.metrics.push(metric(
            &format!("snapshots.0.threads.{idx}.allocated_bytes"),
            alloc,
        ));
    }

    /// Empty flat-metric list → no tid entries at all → terminator
    /// at index 0 → `TidAbsent`.
    #[test]
    fn lookup_thread_empty_metrics_returns_tid_absent() {
        let m = empty_payload();
        assert!(matches!(lookup_thread(&m, 42), ThreadLookup::TidAbsent));
    }

    /// Matching tid with an `allocated_bytes` sibling → `Found`
    /// carrying the observed counter.
    #[test]
    fn lookup_thread_matching_tid_returns_found() {
        let mut m = empty_payload();
        push_tid(&mut m, 0, 42.0);
        push_alloc(&mut m, 0, 1_048_576.0);
        match lookup_thread(&m, 42) {
            ThreadLookup::Found {
                allocated_bytes,
                deallocated_bytes,
            } => {
                assert_eq!(allocated_bytes, 1_048_576);
                assert_eq!(deallocated_bytes, None);
            }
            _ => panic!("expected ThreadLookup::Found"),
        }
    }

    /// Matching tid but no `allocated_bytes` sibling → the probe hit
    /// an error on that thread → `MissingAllocatedBytes`.
    #[test]
    fn lookup_thread_missing_allocated_bytes_returns_missing_variant() {
        let mut m = empty_payload();
        push_tid(&mut m, 0, 42.0);
        // no matching `.allocated_bytes`
        assert!(matches!(
            lookup_thread(&m, 42),
            ThreadLookup::MissingAllocatedBytes
        ));
    }

    /// A contiguous run of tids that does NOT include the caller's
    /// tid, but terminates BEFORE the cap → natural-terminator path
    /// → `TidAbsent` (not `ExceedsCap`).
    #[test]
    fn lookup_thread_contiguous_prefix_without_match_returns_tid_absent() {
        let mut m = empty_payload();
        for i in 0..10 {
            push_tid(&mut m, i, (1000 + i) as f64);
        }
        assert!(matches!(lookup_thread(&m, 42), ThreadLookup::TidAbsent));
    }

    /// The full-cap case: fill indices `0..MAX_SCAN_INDEX`
    /// with non-matching tids, then call lookup_thread with a tid
    /// that isn't in the list. The scan runs all 1024 iterations,
    /// never hits a terminator, never matches, and therefore must
    /// return `ExceedsCap` — distinct from `TidAbsent`.
    #[test]
    fn lookup_thread_saturated_scan_without_match_returns_exceeds_cap() {
        let mut m = empty_payload();
        for i in 0..MAX_SCAN_INDEX {
            // tids chosen so none is equal to the probe tid below.
            push_tid(&mut m, i, (1_000_000 + i) as f64);
        }
        let target_tid: i32 = 42;
        let outcome = lookup_thread(&m, target_tid);
        assert!(
            matches!(outcome, ThreadLookup::ExceedsCap),
            "saturated scan without match must return ExceedsCap; got other variant"
        );
    }

    /// Same invariant for `snapshot_worker_allocated` (the
    /// multi-snapshot path): fill 1024 tid entries for snapshot
    /// index 0, call with a non-matching tid, assert `ExceedsCap`.
    #[test]
    fn snapshot_worker_allocated_saturated_scan_returns_exceeds_cap() {
        let mut m = empty_payload();
        for i in 0..MAX_SCAN_INDEX {
            push_tid(&mut m, i, (1_000_000 + i) as f64);
        }
        let outcome = snapshot_worker_allocated(&m, 0, 42);
        assert!(
            matches!(outcome, ThreadLookup::ExceedsCap),
            "saturated multi-snapshot scan without match must return ExceedsCap"
        );
    }

    /// `snapshot_worker_allocated` with an empty metric list must
    /// return `TidAbsent` — parallel to the single-snapshot path.
    #[test]
    fn snapshot_worker_allocated_empty_returns_tid_absent() {
        let m = empty_payload();
        assert!(matches!(
            snapshot_worker_allocated(&m, 0, 42),
            ThreadLookup::TidAbsent
        ));
    }
}
