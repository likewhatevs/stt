//! Spawn-pipeline tests — misc group.

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

#[test]
fn set_affinity_via_handle() {
    let config = WorkloadConfig {
        num_workers: 1,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::SpinWait,
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    h.start();
    let cpus: BTreeSet<usize> = [0].into_iter().collect();
    let result = h.set_affinity(0, &cpus);
    assert!(result.is_ok());
    std::thread::sleep(std::time::Duration::from_millis(100));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 1);
}
#[test]
fn start_idempotent() {
    let config = WorkloadConfig {
        num_workers: 1,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::SpinWait,
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    h.start();
    h.start(); // Second call should be a no-op (started flag is true).
    std::thread::sleep(std::time::Duration::from_millis(100));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 1);
    assert!(reports[0].work_units > 0);
}
/// Overflow-path pin: when `region_kb * 1024` overflows `usize`
/// (the configured value is so large that the page-fault region
/// size cannot be represented), the worker's outer loop hits
/// the `checked_mul` None arm, emits the `tracing::warn!`, and
/// `break`s without doing any page-fault work. The worker
/// still terminates cleanly and reports 0 iterations — no
/// mmap, no segfault, no hang.
///
/// Spawns a single worker with `region_kb = usize::MAX` so the
/// multiplication overflows on every pointer width we support
/// (32-bit: MAX*1024 overflows immediately; 64-bit: MAX*1024
/// also overflows). Runs briefly, asserts the worker's
/// `iterations` is 0 — proof the outer loop broke out before
/// the first page-fault cycle ran. The worker report still
/// arrives (proving `stop_and_collect` sees a graceful exit
/// on this path, not a signal kill).
///
/// Pairs with [`page_fault_churn_from_name_defaults`] which
/// pins the happy path — together they pin both ends of the
/// region_size validity domain.
#[test]
fn page_fault_churn_region_kb_overflow_worker_exits_cleanly() {
    let config = WorkloadConfig {
        num_workers: 1,
        affinity: AffinityIntent::Inherit,
        // `region_kb = usize::MAX` — `usize::MAX * 1024`
        // overflows on both 32-bit and 64-bit usize, so
        // `checked_mul` returns None and the outer loop
        // `break`s immediately. `touches_per_cycle` and
        // `spin_iters` are ignored by that path.
        work_type: WorkType::PageFaultChurn {
            region_kb: usize::MAX,
            touches_per_cycle: 16,
            spin_iters: 32,
        },
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    h.start();
    // Give the worker a short window to spin through its
    // spawn handshake + outer-loop entry + break. 100 ms is
    // comfortably more than the sub-millisecond path the
    // overflow arm runs, while keeping the test fast.
    std::thread::sleep(Duration::from_millis(100));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 1, "exactly one worker was spawned");
    let r = &reports[0];
    // `iterations` is the outer-loop counter: 0 means the
    // worker hit the `break` BEFORE any page-fault cycle
    // completed, which is the overflow path.
    assert_eq!(
        r.iterations, 0,
        "worker with overflowing region_kb must break out of the outer loop \
         without completing any page-fault cycle; got iterations={}",
        r.iterations,
    );
    // `work_units` may be 0 (spin_burst inside the overflow
    // arm never ran) OR a tiny positive value if the worker
    // took an unrelated iteration through the outer loop —
    // but under this config only PageFaultChurn is selected
    // so spin_burst before the overflow break is not
    // reachable. Assert exact zero to pin the overflow path's
    // no-op guarantee.
    assert_eq!(
        r.work_units, 0,
        "overflow path must not increment work_units; got {}",
        r.work_units,
    );
}
#[test]
fn mutex_contention_records_wake_latency() {
    let config = WorkloadConfig {
        num_workers: 4,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::MutexContention {
            contenders: 4,
            hold_iters: 64,
            work_iters: 256,
        },
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    h.start();
    std::thread::sleep(std::time::Duration::from_millis(500));
    let reports = h.stop_and_collect();
    let has_latencies = reports.iter().any(|r| !r.resume_latencies_ns.is_empty());
    assert!(has_latencies, "contenders should record wake latencies");
}
// -- pathology WorkType smoke tests --
//
// Each pathology variant added in #25 was implemented and
// wired into name registries but had no runtime call site.
// These smoke tests exercise the worker body of every variant
// for ~200ms with the minimum legal worker count and assert
// that workers actually iterated. Catches MAP_SHARED layout
// regressions, futex-word offset mistakes, and worker-group
// partitioning bugs that would surface as zero-iteration
// reports or panics inside `worker_main`.

/// `WorkType::PageFaultChurn` smoke test. Per-iteration cycle:
/// mmap → touch random pages → MADV_DONTNEED → repeat.
#[test]
fn pathology_page_fault_churn_iterates() {
    let cfg = WorkloadConfig {
        num_workers: 2,
        work_type: WorkType::PageFaultChurn {
            region_kb: 256,
            touches_per_cycle: 16,
            spin_iters: 32,
        },
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&cfg).expect("PageFaultChurn must spawn");
    h.start();
    std::thread::sleep(Duration::from_millis(200));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 2);
    for r in &reports {
        assert!(
            r.iterations > 0,
            "PageFaultChurn worker must iterate: {r:?}"
        );
    }
}
/// `WorkType::MutexContention` smoke test. 2 contenders share a
/// MAP_SHARED region; group_size=2 so num_workers=2 fits.
#[test]
fn pathology_mutex_contention_iterates() {
    let cfg = WorkloadConfig {
        num_workers: 2,
        work_type: WorkType::MutexContention {
            contenders: 2,
            hold_iters: 64,
            work_iters: 128,
        },
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&cfg).expect("MutexContention must spawn");
    h.start();
    std::thread::sleep(Duration::from_millis(200));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 2);
    for r in &reports {
        assert!(
            r.iterations > 0,
            "MutexContention worker must iterate: {r:?}"
        );
    }
}
/// `WorkType::ThunderingHerd` smoke test. Minimal herd:
/// waiters=1 → group_size=2, num_workers=2.
#[test]
fn pathology_thundering_herd_iterates() {
    let cfg = WorkloadConfig {
        num_workers: 2,
        work_type: WorkType::ThunderingHerd {
            waiters: 1,
            batches: 50,
            inter_batch_ms: 1,
        },
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&cfg).expect("ThunderingHerd must spawn");
    h.start();
    std::thread::sleep(Duration::from_millis(200));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 2);
    // Waker iterates per batch; waiters iterate per wake. At
    // least one worker should have done some iterations within
    // 200ms even on a contended host.
    let total: u64 = reports.iter().map(|r| r.iterations).sum();
    assert!(total > 0, "ThunderingHerd cohort must iterate: {reports:?}");
}
/// `WorkType::PriorityInversion` smoke test. 1+1+1 = 3 workers
/// (smallest group satisfying high+medium+low constraint).
#[test]
fn pathology_priority_inversion_iterates() {
    let cfg = WorkloadConfig {
        num_workers: 3,
        work_type: WorkType::PriorityInversion {
            high_count: 1,
            medium_count: 1,
            low_count: 1,
            hold_iters: 256,
            work_iters: 128,
            pi_mode: FutexLockMode::Plain,
        },
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&cfg).expect("PriorityInversion must spawn");
    h.start();
    std::thread::sleep(Duration::from_millis(200));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 3);
    let total: u64 = reports.iter().map(|r| r.iterations).sum();
    assert!(
        total > 0,
        "PriorityInversion cohort must iterate: {reports:?}"
    );
}
/// `WorkType::ProducerConsumerImbalance` smoke test. Minimal
/// 1+1 producers/consumers, low rate so the queue doesn't
/// instantly saturate.
#[test]
fn pathology_producer_consumer_imbalance_iterates() {
    let cfg = WorkloadConfig {
        num_workers: 2,
        work_type: WorkType::ProducerConsumerImbalance {
            producers: 1,
            consumers: 1,
            produce_rate_hz: 200,
            consume_iters: 64,
            queue_depth_target: 16,
        },
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&cfg).expect("ProducerConsumerImbalance must spawn");
    h.start();
    std::thread::sleep(Duration::from_millis(200));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 2);
    let total: u64 = reports.iter().map(|r| r.iterations).sum();
    assert!(
        total > 0,
        "Producer/Consumer cohort must iterate: {reports:?}"
    );
}
/// `WorkType::RtStarvation` smoke test. 1 RT + 1 CFS worker.
/// Requires CAP_SYS_NICE for `sched_setscheduler(SCHED_FIFO)`
/// (ktstr always runs as root per project memory).
#[test]
fn pathology_rt_starvation_iterates() {
    let cfg = WorkloadConfig {
        num_workers: 2,
        work_type: WorkType::RtStarvation {
            rt_workers: 1,
            cfs_workers: 1,
            rt_priority: 50,
            burst_iters: 64,
        },
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&cfg).expect("RtStarvation must spawn");
    h.start();
    std::thread::sleep(Duration::from_millis(200));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 2);
    // RT worker pegs the CPU; CFS may or may not iterate
    // depending on starvation. At least one must have run.
    let total: u64 = reports.iter().map(|r| r.iterations).sum();
    assert!(total > 0, "RtStarvation cohort must iterate: {reports:?}");
}
/// `WorkType::AsymmetricWaker` smoke test. Default classes
/// (Cfs/Cfs) so no privilege required at the kernel layer.
#[test]
fn pathology_asymmetric_waker_iterates() {
    let cfg = WorkloadConfig {
        num_workers: 2,
        work_type: WorkType::AsymmetricWaker {
            waker_class: SchedClass::Cfs,
            wakee_class: SchedClass::Cfs,
            burst_iters: 128,
        },
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&cfg).expect("AsymmetricWaker must spawn");
    h.start();
    std::thread::sleep(Duration::from_millis(200));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 2);
    let total: u64 = reports.iter().map(|r| r.iterations).sum();
    assert!(total > 0, "AsymmetricWaker pair must iterate: {reports:?}");
}
// -- Compute WorkType variants smoke tests --

/// `WorkType::AluHot` smoke test at the default
/// (`AluWidth::Widest`) width. Workers must iterate within
/// the test window — a regression that breaks the multiply
/// chain (e.g. by removing the per-step `black_box`) would
/// either fold the loop to nothing or produce zero
/// iterations.
#[test]
fn pathology_alu_hot_iterates() {
    let cfg = WorkloadConfig {
        num_workers: 2,
        work_type: WorkType::AluHot {
            width: AluWidth::Widest,
        },
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&cfg).expect("AluHot must spawn");
    h.start();
    std::thread::sleep(Duration::from_millis(200));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 2);
    for r in &reports {
        assert!(r.iterations > 0, "AluHot worker must iterate: {r:?}");
    }
}
/// `WorkType::AluHot` at `AluWidth::Scalar` exercises the
/// fall-through path that runs on every architecture. Pins
/// the architecture-independent dispatch arm so a regression
/// to the SIMD branches doesn't silently break the
/// scalar default.
#[test]
fn pathology_alu_hot_scalar_iterates() {
    let cfg = WorkloadConfig {
        num_workers: 1,
        work_type: WorkType::AluHot {
            width: AluWidth::Scalar,
        },
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&cfg).expect("AluHot Scalar must spawn");
    h.start();
    std::thread::sleep(Duration::from_millis(100));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 1);
    assert!(
        reports[0].iterations > 0,
        "AluHot Scalar worker must iterate: {:?}",
        reports[0]
    );
}
/// AluHot must populate `iteration_costs_ns` and bump
/// `iteration_cost_sample_total`. Pins the per-iteration
/// reservoir-sampling path so a regression that drops the
/// `reservoir_push` call (or wires the wrong counter) is
/// caught at the WorkerReport boundary, not just at the
/// downstream consumer. AluHot is the simplest of the three
/// pure-compute variants that populate the reservoir; if it
/// stops sampling, SmtSiblingSpin and IpcVariance almost
/// certainly stop too.
#[test]
fn pathology_alu_hot_populates_iteration_costs() {
    let cfg = WorkloadConfig {
        num_workers: 1,
        work_type: WorkType::AluHot {
            width: AluWidth::Scalar,
        },
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&cfg).expect("AluHot must spawn");
    h.start();
    std::thread::sleep(Duration::from_millis(200));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 1);
    let r = &reports[0];
    assert!(
        !r.iteration_costs_ns.is_empty(),
        "AluHot must populate iteration_costs_ns: {r:?}",
    );
    assert!(
        r.iteration_cost_sample_total >= 1,
        "AluHot must record at least one iteration-cost sample: {r:?}",
    );
}
/// `WorkType::SmtSiblingSpin` smoke test. Pairs require
/// `num_workers == 2 * k` (k pairs); with k=1 the workers
/// run independently because the test doesn't pin to SMT
/// siblings — but they still iterate. Pinning to actual SMT
/// siblings is the test author's responsibility (see
/// follow-up #311 for the framework helper).
#[test]
fn pathology_smt_sibling_spin_iterates() {
    let cfg = WorkloadConfig {
        num_workers: 2,
        work_type: WorkType::SmtSiblingSpin,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&cfg).expect("SmtSiblingSpin must spawn");
    h.start();
    std::thread::sleep(Duration::from_millis(200));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 2);
    for r in &reports {
        assert!(
            r.iterations > 0,
            "SmtSiblingSpin worker must iterate: {r:?}"
        );
    }
}
/// `WorkType::SmtSiblingSpin` requires
/// `num_workers % 2 == 0`. Spawn-side rejection at odd
/// counts pins the [`worker_group_size`](WorkType::worker_group_size)
/// `== Some(2)` contract through to the typed-error variant.
#[test]
fn smt_sibling_spin_odd_workers_rejects() {
    let cfg = WorkloadConfig {
        num_workers: 3,
        work_type: WorkType::SmtSiblingSpin,
        ..Default::default()
    };
    let err = WorkloadHandle::spawn(&cfg)
        .err()
        .expect("SmtSiblingSpin with odd num_workers must be rejected");
    let typed = err
        .downcast_ref::<WorkTypeValidationError>()
        .expect("error must downcast to WorkTypeValidationError");
    assert!(
        matches!(
            typed,
            WorkTypeValidationError::NonDivisibleWorkerCount {
                name,
                group_idx: 0,
                group_size: 2,
                num_workers: 3,
            } if name == "SmtSiblingSpin"
        ),
        "expected NonDivisibleWorkerCount for SmtSiblingSpin; got: {typed:?}",
    );
}
/// `WorkType::IpcVariance` smoke test at the defaults from
/// [`defaults`]. Workers alternate hot/cold phases internally
/// — a 200ms run completes a few outer iterations.
#[test]
fn pathology_ipc_variance_iterates() {
    let cfg = WorkloadConfig {
        num_workers: 2,
        work_type: WorkType::IpcVariance {
            hot_iters: 1024,
            cold_iters: 64,
            period_iters: 4,
        },
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&cfg).expect("IpcVariance must spawn");
    h.start();
    std::thread::sleep(Duration::from_millis(200));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 2);
    for r in &reports {
        assert!(r.iterations > 0, "IpcVariance worker must iterate: {r:?}");
    }
}
