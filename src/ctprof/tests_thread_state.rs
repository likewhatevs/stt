//! ThreadState defaults, Mode tie-break, wire-format identity, type pins.
//!
//! Co-located with `super::mod.rs`; one of the topic-grouped
//! split files that replace the monolithic `tests.rs`.

#![cfg(test)]

use super::*;


/// `ThreadState::default()` produces `'~'` (not `'\0'`) for
/// the `state` char so the absent-value sentinel matches the
/// capture-time `unwrap_or_else(default_state_char)`
/// discipline. The bare `char` Default of `'\0'` (U+0000)
/// lex-compares SMALLER than every real kernel state letter
/// (`R`/`S`/`D`/`T`/`t`/`X`/`Z`/`P`/`I`); a Mode-tie-break
/// that picks the lex-smallest would silently elect `'\0'`
/// whenever a default-built thread sat alongside a real one
/// in a group, dragging the cell from a meaningful state
/// letter down to the absent sentinel. The manual
/// [`Default`] impl on [`ThreadState`] pairs with the
/// `serde(default = "default_state_char")` attribute on the
/// field so both construction paths land on `'~'`.
#[test]
fn default_threadstate_state_is_sentinel_tilde() {
    let t = ThreadState::default();
    assert_eq!(
        t.state, '~',
        "ThreadState::default().state must be '~' (the \
         absent-value sentinel chosen to lex-sort AFTER \
         every real kernel state letter), not '\\0' (the \
         bare char Default); see field doc on \
         ThreadState::state"
    );
}

/// Mode tie-break regression: a default-constructed
/// `ThreadState` must NOT lex-beat a real kernel state
/// letter when both contribute to the same Mode aggregation
/// at equal frequency. The kernel's
/// [`crate::ctprof_compare::aggregate`] closure
/// `a.1.cmp(&b.1).then(b.0.cmp(&a.0))` selects
/// LEX-SMALLEST on count-ties, so the sentinel must be
/// LARGER than every real letter to keep the real letter
/// winning. `'~'` (U+007E = 126) is larger than every
/// kernel state letter (`R`=82, `S`=83, `D`=68, `T`=84,
/// `t`=116, `X`=88, `Z`=90, `P`=80, `I`=73), so the
/// tiebreak picks `R`. The original `'?'` (U+003F = 63)
/// sentinel was SMALLER than every real letter, which
/// would have made this test fail.
#[test]
fn mode_tiebreak_against_default_state_picks_real_letter() {
    use crate::ctprof_compare::{AggRule, Aggregated, aggregate};
    let default_thread = ThreadState::default();
    let real_thread = ThreadState {
        state: 'R',
        ..ThreadState::default()
    };
    let agg = aggregate(
        AggRule::ModeChar(|t| t.state),
        &[&default_thread, &real_thread],
    );
    match &agg {
        Aggregated::Mode { .. } => assert_eq!(
            agg.mode_value(),
            "R",
            "Mode tiebreak between '~' (default sentinel) \
             and 'R' (real kernel state) must elect 'R'; \
             got {:?}",
            agg.mode_value(),
        ),
        other => panic!("expected Mode, got {other:?}"),
    }
}

/// Wire-format identity: hand-written JSON with raw
/// primitive values at every newtype-wrapped field position
/// must deserialize cleanly into a post-phase-2
/// `ThreadState` with the wrapper fields holding the
/// expected values. Covers one representative field per
/// newtype family — MonotonicCount, MonotonicNs, ClockTicks,
/// Bytes, PeakNs, GaugeNs, GaugeCount, OrdinalI32,
/// OrdinalU32, CategoricalString, CpuSet — so a regression
/// that breaks `serde(transparent)` on any wrapper would
/// surface here without needing a real .ctprof.zst file from
/// pre-phase-2 capture. Pre-phase-2 snapshot files (raw
/// `u64`/`i32`/`String`/`Vec<u32>` at every position)
/// continue to deserialize identically.
#[test]
fn wire_format_identity_raw_primitives_deserialize_into_wrapped_thread_state() {
    let json = r#"{
        "tid": 1234,
        "tgid": 1234,
        "pcomm": "demo",
        "comm": "demo-w",
        "cgroup": "/app",
        "start_time_clock_ticks": 555000,
        "policy": "SCHED_OTHER",
        "nice": -5,
        "cpu_affinity": [0, 1, 2, 3],
        "processor": 7,
        "state": "R",
        "ext_enabled": false,
        "run_time_ns": 1000000,
        "wait_time_ns": 0,
        "timeslices": 50,
        "voluntary_csw": 100,
        "nonvoluntary_csw": 25,
        "nr_wakeups": 200,
        "nr_wakeups_local": 80,
        "nr_wakeups_remote": 30,
        "nr_wakeups_sync": 10,
        "nr_wakeups_migrate": 5,
        "nr_wakeups_affine": 60,
        "nr_wakeups_affine_attempts": 100,
        "nr_migrations": 8,
        "nr_forced_migrations": 1,
        "nr_failed_migrations_affine": 0,
        "nr_failed_migrations_running": 0,
        "nr_failed_migrations_hot": 0,
        "wait_sum": 5000000,
        "wait_count": 15,
        "wait_max": 250000,
        "voluntary_sleep_ns": 3200000,
        "sleep_max": 180000,
        "block_sum": 1100000,
        "block_max": 60000,
        "iowait_sum": 77000,
        "iowait_count": 18,
        "exec_max": 90000,
        "slice_max": 400000,
        "allocated_bytes": 16777216,
        "deallocated_bytes": 8388608,
        "minflt": 7777,
        "majflt": 8888,
        "utime_clock_ticks": 10,
        "stime_clock_ticks": 11,
        "priority": 25,
        "rt_priority": 99,
        "core_forceidle_sum": 0,
        "fair_slice_ns": 250000,
        "nr_threads": 4,
        "smaps_rollup_kb": {},
        "rchar": 100,
        "wchar": 200,
        "syscr": 10,
        "syscw": 20,
        "read_bytes": 4096,
        "write_bytes": 8192,
        "cancelled_write_bytes": 1024,
        "cpu_delay_count": 0,
        "cpu_delay_total_ns": 0,
        "cpu_delay_max_ns": 0,
        "cpu_delay_min_ns": 0,
        "blkio_delay_count": 0,
        "blkio_delay_total_ns": 0,
        "blkio_delay_max_ns": 0,
        "blkio_delay_min_ns": 0,
        "swapin_delay_count": 0,
        "swapin_delay_total_ns": 0,
        "swapin_delay_max_ns": 0,
        "swapin_delay_min_ns": 0,
        "freepages_delay_count": 0,
        "freepages_delay_total_ns": 0,
        "freepages_delay_max_ns": 0,
        "freepages_delay_min_ns": 0,
        "thrashing_delay_count": 0,
        "thrashing_delay_total_ns": 0,
        "thrashing_delay_max_ns": 0,
        "thrashing_delay_min_ns": 0,
        "compact_delay_count": 0,
        "compact_delay_total_ns": 0,
        "compact_delay_max_ns": 0,
        "compact_delay_min_ns": 0,
        "wpcopy_delay_count": 0,
        "wpcopy_delay_total_ns": 0,
        "wpcopy_delay_max_ns": 0,
        "wpcopy_delay_min_ns": 0,
        "irq_delay_count": 0,
        "irq_delay_total_ns": 0,
        "irq_delay_max_ns": 0,
        "irq_delay_min_ns": 0,
        "hiwater_rss_bytes": 0,
        "hiwater_vm_bytes": 0
    }"#;
    let t: ThreadState = serde_json::from_str(json).expect("deserialize");
    // One representative field per newtype family proves
    // serde(transparent) works post-migration.
    assert_eq!(t.run_time_ns, crate::metric_types::MonotonicNs(1_000_000));
    assert_eq!(t.timeslices, crate::metric_types::MonotonicCount(50));
    assert_eq!(t.utime_clock_ticks, crate::metric_types::ClockTicks(10));
    assert_eq!(t.allocated_bytes, crate::metric_types::Bytes(16_777_216));
    assert_eq!(
        t.cancelled_write_bytes,
        crate::metric_types::Bytes(1024),
        "cancelled_write_bytes round-trips through the JSON \
         wire format alongside the other Bytes-typed fields",
    );
    assert_eq!(t.wait_max, crate::metric_types::PeakNs(250_000));
    assert_eq!(t.fair_slice_ns, crate::metric_types::GaugeNs(250_000));
    assert_eq!(t.nr_threads, crate::metric_types::GaugeCount(4));
    assert_eq!(t.nice, crate::metric_types::OrdinalI32(-5));
    assert_eq!(t.rt_priority, crate::metric_types::OrdinalU32(99));
    assert_eq!(
        t.policy,
        crate::metric_types::CategoricalString::from("SCHED_OTHER")
    );
    assert_eq!(
        t.cpu_affinity,
        crate::metric_types::CpuSet(vec![0, 1, 2, 3])
    );
}

/// Type-pin: nr_threads MUST be `GaugeCount`. A future
/// refactor that flips it to a different newtype (e.g.
/// `MonotonicCount`, which would silently re-enable Summable
/// and let `--group-by comm`/`--group-by cgroup` over-count
/// the parent process N-fold) would break this single
/// `let _: GaugeCount = ...;` assignment. The test compiles
/// only when the type is exactly `GaugeCount`.
#[test]
fn nr_threads_field_pinned_to_gauge_count() {
    let t = ThreadState::default();
    let _: crate::metric_types::GaugeCount = t.nr_threads;
}

/// Type-pin: cancelled_write_bytes MUST be `Bytes`. A future
/// refactor that flipped it to a non-byte type (e.g. plain
/// `MonotonicCount`, dropping the IEC-binary auto-scale
/// ladder and the registry's `unit: "B"` rendering) would
/// break this single `let _: Bytes = ...;` assignment. The
/// test compiles only when the type is exactly `Bytes`.
#[test]
fn cancelled_write_bytes_field_pinned_to_bytes() {
    let t = ThreadState::default();
    let _: crate::metric_types::Bytes = t.cancelled_write_bytes;
}
