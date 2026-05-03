//! Spawn-pipeline tests — fan_out group.

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

// -- FutexFanOut tests --

#[test]
fn spawn_futex_fan_out_produces_work() {
    let reports = spawn_and_collect_after(
        WorkType::FutexFanOut {
            fan_out: 4,
            spin_iters: 1024,
        },
        5, // 1 messenger + 4 receivers
        500,
    );
    assert_eq!(reports.len(), 5);
    for r in &reports {
        assert!(r.work_units > 0, "FutexFanOut worker {} did no work", r.tid);
    }
}
#[test]
fn spawn_futex_fan_out_receivers_record_wake_latency() {
    let config = WorkloadConfig {
        num_workers: 5,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::FutexFanOut {
            fan_out: 4,
            spin_iters: 512,
        },
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    h.start();
    std::thread::sleep(std::time::Duration::from_millis(500));
    let reports = h.stop_and_collect();
    // At least one receiver should have wake latency samples.
    let has_latencies = reports.iter().any(|r| !r.resume_latencies_ns.is_empty());
    assert!(has_latencies, "receivers should record wake latencies");
}
#[test]
fn spawn_futex_fan_out_bad_worker_count_fails() {
    let config = WorkloadConfig {
        num_workers: 3, // not divisible by 5
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::FutexFanOut {
            fan_out: 4,
            spin_iters: 1024,
        },
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let result = WorkloadHandle::spawn(&config);
    assert!(result.is_err());
    let msg = format!("{:#}", result.err().unwrap());
    assert!(
        msg.contains("divisible by 5"),
        "expected divisibility error: {msg}"
    );
}
#[test]
fn spawn_futex_fan_out_two_groups() {
    let config = WorkloadConfig {
        num_workers: 10, // 2 groups of (1+4)
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::FutexFanOut {
            fan_out: 4,
            spin_iters: 512,
        },
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    assert_eq!(h.worker_pids().len(), 10);
    h.start();
    std::thread::sleep(std::time::Duration::from_millis(500));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 10);
    for r in &reports {
        assert!(r.work_units > 0, "worker {} did no work", r.tid);
    }
}
#[test]
fn spawn_futex_fan_out_single_receiver() {
    // Minimal fan-out: 1 messenger + 1 receiver per group (like ping-pong).
    let config = WorkloadConfig {
        num_workers: 2,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::FutexFanOut {
            fan_out: 1,
            spin_iters: 1024,
        },
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    h.start();
    std::thread::sleep(std::time::Duration::from_millis(300));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 2);
    for r in &reports {
        assert!(r.work_units > 0, "worker {} did no work", r.tid);
    }
}
#[test]
fn work_type_futex_fan_out_name() {
    let wt = WorkType::FutexFanOut {
        fan_out: 4,
        spin_iters: 1024,
    };
    assert_eq!(wt.name(), "FutexFanOut");
}
#[test]
fn work_type_futex_fan_out_from_name() {
    let wt = WorkType::from_name("FutexFanOut").unwrap();
    match wt {
        WorkType::FutexFanOut {
            fan_out,
            spin_iters,
        } => {
            assert_eq!(fan_out, 4);
            assert_eq!(spin_iters, 1024);
        }
        _ => panic!("expected FutexFanOut"),
    }
}
#[test]
fn work_type_futex_fan_out_group_size() {
    let wt = WorkType::FutexFanOut {
        fan_out: 4,
        spin_iters: 1024,
    };
    assert_eq!(wt.worker_group_size(), Some(5));
}
#[test]
fn resolve_work_type_fan_out_group_size() {
    let base = WorkType::SpinWait;
    let over = WorkType::futex_fan_out(3, 100); // group_size = 4
    let result = resolve_work_type(&base, Some(&over), true, 8); // 8 divisible by 4
    assert!(matches!(result, WorkType::FutexFanOut { .. }));
    let fail = resolve_work_type(&base, Some(&over), true, 6); // 6 not divisible by 4
    assert!(matches!(fail, WorkType::SpinWait));
}
/// Guards two invariants of [`WorkType::FanOutCompute`]:
///
/// 1. Every spawned worker produces non-zero `work_units`, and at
///    least one records a wake latency into `resume_latencies_ns`.
/// 2. The Release/Acquire ordering between the messenger's
///    `wake_ns` store and its generation advance prevents workers
///    from pairing a fresh generation with a stale or zero-init
///    `wake_ns` — the 10 s latency bound below detects only the
///    zero-init arm of that failure mode (see comment on the
///    bound).
///
/// Platform coverage: x86-64 is TSO (store-store and load-load
/// reordering are hardware-prohibited), so on x86 CI this test
/// cannot reproduce a weak-memory regression of the messenger-
/// side store reorder or the worker-side load speculation that
/// the Release/Acquire on aarch64 guards against — the hardware
/// masks the bug. It still catches implementation bugs that
/// surface on any platform, most notably a missing or
/// misordered `wake_ns` store that leaves workers reading
/// zero-init memory (the 10 s bound trips on `now_ns - 0`).
/// Round-over-round reordering cannot be detected by this
/// assertion on any platform. Meaningful weak-memory
/// regression protection requires running this test on an
/// aarch64 runner in CI.
#[test]
fn spawn_fan_out_compute_produces_work() {
    let config = WorkloadConfig {
        num_workers: 5, // 1 messenger + 4 workers
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::FanOutCompute {
            fan_out: 4,
            cache_footprint_kb: 256,
            operations: 5,
            sleep_usec: 100,
        },
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    h.start();
    std::thread::sleep(std::time::Duration::from_millis(500));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 5);
    for r in &reports {
        assert!(
            r.work_units > 0,
            "FanOutCompute worker {} did no work",
            r.tid
        );
    }
    // Every non-messenger worker (receiver) must record at
    // least one wake-latency sample — the messenger advances
    // the generation and never waits, so its latency vec is
    // legitimately empty. Asserting the stronger per-receiver
    // contract (previously `reports.iter().any(...)`) catches
    // a regression that leaves one group of receivers parked
    // on futex_wait without ever seeing the generation advance.
    assert!(
        reports
            .iter()
            .filter(|r| !r.is_messenger)
            .all(|r| !r.resume_latencies_ns.is_empty()),
        "every FanOutCompute receiver must record at least one \
         wake latency sample; got {:?}",
        reports
            .iter()
            .map(|r| (r.tid, r.is_messenger, r.resume_latencies_ns.len()))
            .collect::<Vec<_>>(),
    );
    // The 10 s bound catches the zero-init arm of a missing
    // Release/Acquire pairing: a worker that reads `wake_ns`
    // before the messenger's first store sees 0, so
    // `now_ns.saturating_sub(0)` surfaces `CLOCK_MONOTONIC`
    // (seconds-to-days of monotonic uptime) >> 10 s on any
    // live machine. It does NOT catch round-over-round
    // mispairing — a fresh generation paired with the
    // immediately-prior round's `wake_ns` yields a sub-second
    // delta that is indistinguishable from a correctly-paired
    // fast wake. This is a coarse guard against the easy
    // failure mode, not a full verification of the ordering.
    const MAX_PLAUSIBLE_LATENCY_NS: u64 = 10_000_000_000;
    for r in &reports {
        for &lat in &r.resume_latencies_ns {
            assert!(
                lat < MAX_PLAUSIBLE_LATENCY_NS,
                "worker {} recorded implausible wake latency {} ns \
                 (expected < {} ns); indicates wake_ns/generation \
                 ordering race. NB: lat==0 is LEGITIMATE under \
                 correct ordering — a Relaxed `wake_atom.load` \
                 paired with an Acquire gen load can see a wake_ns \
                 from a LATER round (gen+1's store becomes visible \
                 ahead of gen+1's wake_ns re-load), making \
                 now_ns < wake_ns and `saturating_sub` = 0. The \
                 reservoir-sampling of real latencies is dominated \
                 by positive values; a stray zero from this race \
                 is not a bug, so no lower bound is asserted.",
                r.tid,
                lat,
                MAX_PLAUSIBLE_LATENCY_NS
            );
        }
    }
}
#[test]
fn spawn_fan_out_compute_bad_worker_count_fails() {
    let config = WorkloadConfig {
        num_workers: 3,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::FanOutCompute {
            fan_out: 4,
            cache_footprint_kb: 256,
            operations: 5,
            sleep_usec: 100,
        },
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let result = WorkloadHandle::spawn(&config);
    assert!(result.is_err());
    let msg = format!("{:#}", result.err().unwrap());
    assert!(
        msg.contains("divisible by 5"),
        "expected divisibility error: {msg}"
    );
}
/// Two-messenger-group variant of the invariants guarded by
/// [`spawn_fan_out_compute_produces_work`] — see that test's
/// doc for the full Release/Acquire rationale and platform
/// coverage notes.
#[test]
fn spawn_fan_out_compute_two_groups() {
    let config = WorkloadConfig {
        num_workers: 10, // 2 groups of (1 messenger + 4 workers)
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::FanOutCompute {
            fan_out: 4,
            cache_footprint_kb: 256,
            operations: 5,
            sleep_usec: 100,
        },
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    assert_eq!(h.worker_pids().len(), 10);
    h.start();
    std::thread::sleep(std::time::Duration::from_millis(500));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 10);
    for r in &reports {
        assert!(
            r.work_units > 0,
            "FanOutCompute worker {} did no work",
            r.tid
        );
    }
    // Every non-messenger worker (receiver) in each group must
    // record at least one wake-latency sample — mirror of the
    // per-receiver contract asserted in the single-group test
    // at `spawn_fan_out_compute_produces_work`. With 10 workers
    // and 2 groups (1 messenger + 4 receivers each), this means
    // 8 receivers must all report non-empty latency vectors.
    assert!(
        reports
            .iter()
            .filter(|r| !r.is_messenger)
            .all(|r| !r.resume_latencies_ns.is_empty()),
        "every FanOutCompute receiver in both groups must record \
         at least one wake latency sample; got {:?}",
        reports
            .iter()
            .map(|r| (r.tid, r.is_messenger, r.resume_latencies_ns.len()))
            .collect::<Vec<_>>(),
    );
    // Mirror of the single-group test's latency sanity check —
    // see `spawn_fan_out_compute_produces_work` for rationale.
    // The 10 s bound catches the zero-init arm of a missing
    // Release/Acquire pairing but not round-over-round
    // mispairing; with two messenger groups running
    // independently it still provides a coarse smoke test per
    // group.
    const MAX_PLAUSIBLE_LATENCY_NS: u64 = 10_000_000_000;
    for r in &reports {
        for &lat in &r.resume_latencies_ns {
            assert!(
                lat < MAX_PLAUSIBLE_LATENCY_NS,
                "worker {} recorded implausible wake latency {} ns \
                 (expected < {} ns); indicates wake_ns/generation \
                 ordering race. NB: lat==0 is LEGITIMATE under \
                 correct ordering — a Relaxed `wake_atom.load` \
                 paired with an Acquire gen load can see a wake_ns \
                 from a LATER round (gen+1's store becomes visible \
                 ahead of gen+1's wake_ns re-load), making \
                 now_ns < wake_ns and `saturating_sub` = 0. The \
                 reservoir-sampling of real latencies is dominated \
                 by positive values; a stray zero from this race \
                 is not a bug, so no lower bound is asserted.",
                r.tid,
                lat,
                MAX_PLAUSIBLE_LATENCY_NS
            );
        }
    }
}
