//! Spawn-pipeline tests — idle_churn group.

#![cfg(test)]
#![allow(unused_imports)]

use super::super::affinity::*;
use super::super::config::*;
use super::super::types::*;
use super::super::worker::*;
use super::testing::*;
use super::*;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// `WorkType::IdleChurn` smoke test. burst=1ms + sleep=5ms
/// matches the variant's defaults; a 200ms run gives ~30
/// iterations per worker (timer_slack adds ~50µs to each
/// 5ms sleep). Asserts every worker iterates — the variant
/// is dead if `nanosleep` returns immediately, the timespec
/// is malformed, or the spawn-side validation rejects the
/// non-zero defaults.
#[test]
fn pathology_idle_churn_iterates() {
    let cfg = WorkloadConfig {
        num_workers: 2,
        work_type: WorkType::IdleChurn {
            burst_duration: Duration::from_millis(1),
            sleep_duration: Duration::from_millis(5),
            precise_timing: false,
        },
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&cfg).expect("IdleChurn must spawn");
    h.start();
    std::thread::sleep(Duration::from_millis(200));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 2);
    for r in &reports {
        assert!(r.iterations > 0, "IdleChurn worker must iterate: {r:?}");
    }
}
// -- IdleChurn spawn-side validation coverage --
//
// Pin the bail messages emitted by the IdleChurn arm of
// `WorkloadHandle::spawn`'s per-group validation: zero
// sleep_duration collapses to SpinWait, zero burst_duration
// produces pure-sleep no-runtime workers, both-zero must fire
// the burst check first (validation order), and minimal
// non-zero values must pass through. Composed groups must
// tag their group_idx in the diagnostic so multi-group test
// scenarios can locate the offending entry.

/// Zero `sleep_duration` collapses IdleChurn to SpinWait — the
/// bail must name `sleep_duration` and steer the caller to
/// SpinWait directly. Also pins the typed-error downcast to
/// [`WorkTypeValidationError::ZeroSleepDuration`] so callers
/// can program against the variant rather than the message text.
#[test]
fn idle_churn_zero_sleep_rejects() {
    let cfg = WorkloadConfig {
        num_workers: 1,
        work_type: WorkType::IdleChurn {
            burst_duration: Duration::from_millis(1),
            sleep_duration: Duration::ZERO,
            precise_timing: false,
        },
        ..Default::default()
    };
    let err = WorkloadHandle::spawn(&cfg)
        .err()
        .expect("IdleChurn with sleep_duration=ZERO must be rejected at spawn");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("sleep_duration must be > 0"),
        "diagnostic must name the rejected field; got: {msg}",
    );
    assert!(
        msg.contains("SpinWait"),
        "diagnostic must steer the caller to SpinWait; got: {msg}",
    );
    let typed = err
        .downcast_ref::<WorkTypeValidationError>()
        .expect("error must downcast to WorkTypeValidationError");
    assert!(
        matches!(
            typed,
            WorkTypeValidationError::ZeroSleepDuration { group_idx: 0 }
        ),
        "expected ZeroSleepDuration {{ group_idx: 0 }}, got: {typed:?}",
    );
}
/// Zero `burst_duration` makes the loop pure sleep — the bail
/// must name `burst_duration` and explain "pure sleep". Also
/// pins the typed-error downcast to
/// [`WorkTypeValidationError::ZeroBurstDuration`].
#[test]
fn idle_churn_zero_burst_rejects() {
    let cfg = WorkloadConfig {
        num_workers: 1,
        work_type: WorkType::IdleChurn {
            burst_duration: Duration::ZERO,
            sleep_duration: Duration::from_millis(5),
            precise_timing: false,
        },
        ..Default::default()
    };
    let err = WorkloadHandle::spawn(&cfg)
        .err()
        .expect("IdleChurn with burst_duration=ZERO must be rejected at spawn");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("burst_duration must be > 0"),
        "diagnostic must name the rejected field; got: {msg}",
    );
    assert!(
        msg.contains("pure sleep"),
        "diagnostic must explain the pure-sleep degeneracy; got: {msg}",
    );
    let typed = err
        .downcast_ref::<WorkTypeValidationError>()
        .expect("error must downcast to WorkTypeValidationError");
    assert!(
        matches!(
            typed,
            WorkTypeValidationError::ZeroBurstDuration { group_idx: 0 }
        ),
        "expected ZeroBurstDuration {{ group_idx: 0 }}, got: {typed:?}",
    );
}
/// When both fields are zero, the burst check must fire first
/// (validation order). The diagnostic must name
/// `burst_duration` and must NOT name `sleep_duration` —
/// proving the spawn returns on the first failed check rather
/// than concatenating both messages. The typed-error variant
/// must be [`WorkTypeValidationError::ZeroBurstDuration`] for
/// the same reason.
#[test]
fn idle_churn_both_zero_rejects_burst_first() {
    let cfg = WorkloadConfig {
        num_workers: 1,
        work_type: WorkType::IdleChurn {
            burst_duration: Duration::ZERO,
            sleep_duration: Duration::ZERO,
            precise_timing: false,
        },
        ..Default::default()
    };
    let err = WorkloadHandle::spawn(&cfg)
        .err()
        .expect("IdleChurn with both fields ZERO must be rejected at spawn");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("burst_duration must be > 0"),
        "burst check fires first; diagnostic must name burst_duration: {msg}",
    );
    assert!(
        !msg.contains("sleep_duration must be > 0"),
        "sleep check must NOT fire when burst already failed; got: {msg}",
    );
    let typed = err
        .downcast_ref::<WorkTypeValidationError>()
        .expect("error must downcast to WorkTypeValidationError");
    assert!(
        matches!(
            typed,
            WorkTypeValidationError::ZeroBurstDuration { group_idx: 0 }
        ),
        "expected ZeroBurstDuration {{ group_idx: 0 }} (burst fires first), got: {typed:?}",
    );
}
/// Minimum-valid Durations (1ns each) must pass spawn-time
/// validation. The runtime semantics of 1ns burst+sleep are
/// degenerate (timer slack dominates the sleep, the burst
/// completes in a single iter check), but spawn must accept
/// any non-zero Duration — only `Duration::ZERO` is rejected.
/// Start+stop immediately to bound test runtime.
#[test]
fn idle_churn_min_valid_durations_pass_validation() {
    let cfg = WorkloadConfig {
        num_workers: 1,
        work_type: WorkType::IdleChurn {
            burst_duration: Duration::from_nanos(1),
            sleep_duration: Duration::from_nanos(1),
            precise_timing: false,
        },
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&cfg)
        .expect("IdleChurn with 1ns durations must pass spawn-side validation");
    h.start();
    let _reports = h.stop_and_collect();
}
/// `precise_timing: true` is the IdleChurn opt-in for shrinking
/// `current->timer_slack_ns` from the inherited 50µs default to
/// 1ns. This smoke test exercises spawn + run with the flag on
/// to prove the dispatch arm's `prctl(PR_SET_TIMERSLACK, 1)`
/// branch survives end-to-end (worker doesn't crash, iterations
/// still accrue). It does NOT measure the slack itself —
/// `prctl(PR_GET_TIMERSLACK)` would be required for an
/// observable assertion, and slack effects only become visible
/// in wake-latency distributions for sub-50µs sleeps which
/// require longer runs than this smoke test allocates.
/// `default_timer_slack_ns` per
/// `init/init_task.c:172` is 50_000ns, so a 5ms sleep already
/// drowns the slack signal in the dispatch arm's
/// `resume_overhead = elapsed.saturating_sub(sleep_duration)`
/// computation; this test only proves the prctl call doesn't
/// fail-stop the worker.
#[test]
fn idle_churn_precise_timing_iterates() {
    let cfg = WorkloadConfig {
        num_workers: 1,
        work_type: WorkType::IdleChurn {
            burst_duration: Duration::from_millis(1),
            sleep_duration: Duration::from_millis(5),
            precise_timing: true,
        },
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&cfg).expect("IdleChurn precise_timing=true must spawn");
    h.start();
    std::thread::sleep(Duration::from_millis(100));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 1);
    assert!(
        reports[0].iterations > 0,
        "IdleChurn precise_timing=true worker must iterate: {:?}",
        reports[0],
    );
}
