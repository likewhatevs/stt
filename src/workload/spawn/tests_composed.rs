//! Spawn-pipeline tests — composed group.

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

/// Composed groups spawn alongside the primary group and tag
/// every produced [`WorkerReport`] with the spawning group's
/// `group_idx`. The brief specifies SpinWait(2) primary +
/// composed=[PipeIo(2)] → 4 reports total with group_idxs
/// `[0, 0, 1, 1]` in spawn order.
#[test]
fn spawn_with_composed_tags_group_idx() {
    let config = WorkloadConfig::default()
        .work_type(WorkType::SpinWait)
        .workers(2)
        .with_composed(
            WorkSpec::default()
                .work_type(WorkType::pipe_io(64))
                .workers(2),
        );
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    assert_eq!(
        h.worker_pids().len(),
        4,
        "primary(2) + composed[0](2) = 4 worker pids",
    );
    h.start();
    std::thread::sleep(std::time::Duration::from_millis(200));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 4, "every spawned worker emits a report");

    // Per-group counts: 2 reports from group 0, 2 from group 1.
    let group_idxs: Vec<usize> = reports.iter().map(|r| r.group_idx).collect();
    let n_primary = group_idxs.iter().filter(|&&g| g == 0).count();
    let n_composed_0 = group_idxs.iter().filter(|&&g| g == 1).count();
    assert_eq!(
        n_primary, 2,
        "group_idx==0 (primary) must produce exactly num_workers reports; got {group_idxs:?}",
    );
    assert_eq!(
        n_composed_0, 2,
        "group_idx==1 (composed[0]) must produce exactly num_workers reports; got {group_idxs:?}",
    );

    // Every report must come from one of the declared groups —
    // a group_idx outside `0..=1` would mean a sentinel/leak
    // path is forging a tag.
    for r in &reports {
        assert!(
            r.group_idx <= 1,
            "report carries group_idx={}, exceeds composed-list \
             cardinality (1 primary + 1 composed = max group_idx 1)",
            r.group_idx,
        );
    }
}
/// Composed [`WorkSpec::num_workers`] = `None` is rejected at
/// spawn time. The scenario engine resolves `None` against
/// `Ctx::workers_per_cgroup` before reaching
/// [`WorkloadHandle::spawn`]; bare callers of `spawn()` must
/// supply a concrete count.
#[test]
fn spawn_with_composed_rejects_none_num_workers() {
    let config = WorkloadConfig::default()
        .work_type(WorkType::SpinWait)
        .workers(1)
        .with_composed(WorkSpec::default().work_type(WorkType::SpinWait));
    let result = WorkloadHandle::spawn(&config);
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("composed entry with num_workers=None must be rejected at spawn"),
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains("num_workers must be set"),
        "diagnostic must name the failure cause; got: {msg}",
    );
}
/// Composed [`WorkSpec::affinity`] resolution: only `Inherit`
/// and `Exact` are reachable from spawn() — topology-aware
/// variants need scenario-level state (TestTopology, cpuset)
/// that bare `spawn()` does not have.
#[test]
fn spawn_with_composed_rejects_topology_affinity() {
    let config = WorkloadConfig::default()
        .work_type(WorkType::SpinWait)
        .workers(1)
        .with_composed(
            WorkSpec::default()
                .work_type(WorkType::SpinWait)
                .workers(1)
                .affinity(AffinityIntent::LlcAligned),
        );
    let result = WorkloadHandle::spawn(&config);
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("composed entry with topology-aware affinity must be rejected"),
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains("requires scenario context"),
        "diagnostic must point at the missing scenario context; got: {msg}",
    );
    assert!(
        msg.contains("composed[0].affinity"),
        "diagnostic must name the composed-entry site; got: {msg}",
    );
}
/// Composed [`WorkSpec::affinity`] = `Exact(set)` is accepted
/// — it carries its own resolved CPU set and needs no
/// scenario context. Confirms the no-context path remains
/// reachable from bare `spawn()`.
#[test]
fn spawn_with_composed_accepts_exact_affinity() {
    let config = WorkloadConfig::default()
        .work_type(WorkType::SpinWait)
        .workers(1)
        .with_composed(
            WorkSpec::default()
                .work_type(WorkType::SpinWait)
                .workers(1)
                .affinity(AffinityIntent::exact([0])),
        );
    let mut h = WorkloadHandle::spawn(&config).expect("Exact affinity must accept");
    h.start();
    std::thread::sleep(std::time::Duration::from_millis(100));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 2, "1 primary + 1 composed = 2 reports");
}
/// Composed entries inherit the parent
/// [`WorkloadConfig::clone_mode`] — [`WorkSpec`] carries no
/// `clone_mode` field of its own, so there's nothing to
/// disagree with the parent. The dispatch path is a
/// workload-wide property; mixing `fork` and `thread` workers
/// inside one [`SpawnGuard`] would route teardown through the
/// wrong code path. Composed Thread-mode workloads still spawn
/// correctly because every group inherits the same mode.
#[test]
fn spawn_with_composed_inherits_parent_clone_mode() {
    let config = WorkloadConfig::default()
        .work_type(WorkType::SpinWait)
        .workers(1)
        .clone_mode(CloneMode::Thread)
        .with_composed(WorkSpec::default().work_type(WorkType::SpinWait).workers(1));
    let mut h = WorkloadHandle::spawn(&config)
        .expect("composed entry must inherit Thread mode without diagnostic");
    h.start();
    std::thread::sleep(std::time::Duration::from_millis(100));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 2, "1 primary + 1 composed = 2 reports");
}
/// Three composed entries plus the primary = 4 groups total.
/// Catches off-by-one in the group_idx assignment loop: if the
/// primary mistakenly used group_idx=1 (or composed[k] used
/// k instead of k+1), the count-by-group_idx asserts surface
/// the drift immediately.
#[test]
fn spawn_with_three_composed_tags_each_group_idx() {
    let config = WorkloadConfig::default()
        .work_type(WorkType::SpinWait)
        .workers(1)
        .with_composed(WorkSpec::default().work_type(WorkType::SpinWait).workers(2))
        .with_composed(WorkSpec::default().work_type(WorkType::SpinWait).workers(3))
        .with_composed(WorkSpec::default().work_type(WorkType::SpinWait).workers(4));
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    assert_eq!(
        h.worker_pids().len(),
        1 + 2 + 3 + 4,
        "primary(1) + composed[0](2) + composed[1](3) + composed[2](4) = 10 pids",
    );
    h.start();
    std::thread::sleep(std::time::Duration::from_millis(200));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 10);
    let group_idxs: Vec<usize> = reports.iter().map(|r| r.group_idx).collect();
    // Per-group counts: 1 primary, 2 in composed[0], 3 in
    // composed[1], 4 in composed[2].
    let count_for = |g: usize| group_idxs.iter().filter(|&&x| x == g).count();
    assert_eq!(
        count_for(0),
        1,
        "primary (group_idx=0) must produce 1 report; got {group_idxs:?}",
    );
    assert_eq!(
        count_for(1),
        2,
        "composed[0] (group_idx=1) must produce 2 reports; got {group_idxs:?}",
    );
    assert_eq!(
        count_for(2),
        3,
        "composed[1] (group_idx=2) must produce 3 reports; got {group_idxs:?}",
    );
    assert_eq!(
        count_for(3),
        4,
        "composed[2] (group_idx=3) must produce 4 reports; got {group_idxs:?}",
    );
    // group_idx must never exceed declared cardinality (4 groups → max 3).
    for r in &reports {
        assert!(
            r.group_idx <= 3,
            "report carries group_idx={}, exceeds composed list cardinality",
            r.group_idx,
        );
    }
}
/// Composed = empty Vec spawns the primary group only — the
/// composed iteration is a no-op when the vec is empty.
#[test]
fn spawn_with_empty_composed_runs_primary_only() {
    let config = WorkloadConfig::default()
        .work_type(WorkType::SpinWait)
        .workers(2)
        .composed(std::iter::empty());
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    assert_eq!(h.worker_pids().len(), 2);
    h.start();
    std::thread::sleep(std::time::Duration::from_millis(100));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 2);
    for r in &reports {
        assert_eq!(r.group_idx, 0, "empty composed: every report is primary");
    }
}
#[test]
fn spawn_with_fixed_affinity() {
    let config = WorkloadConfig {
        num_workers: 1,
        affinity: AffinityIntent::Exact([0].into_iter().collect()),
        work_type: WorkType::SpinWait,
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    h.start();
    std::thread::sleep(std::time::Duration::from_millis(200));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 1);
    assert!(reports[0].cpus_used.contains(&0));
    assert_eq!(reports[0].cpus_used.len(), 1, "should only use pinned CPU");
}
/// JSON roundtrip for `AffinityIntent::RandomSubset` — the `from`
/// pool and `count` survive serialize → deserialize unchanged.
/// Pins the serde shape so a future variant rename or field
/// reorder surfaces here rather than in user-visible config files.
#[test]
fn affinity_intent_random_subset_serde_roundtrip() {
    let original = WorkloadConfig::default()
        .work_type(WorkType::SpinWait)
        .affinity(AffinityIntent::RandomSubset {
            from: BTreeSet::from([0usize, 1, 2, 3]),
            count: 2,
        });
    let json =
        serde_json::to_string(&original).expect("WorkloadConfig with RandomSubset must serialize");
    let deserialized: WorkloadConfig =
        serde_json::from_str(&json).expect("WorkloadConfig RandomSubset JSON must deserialize");
    match deserialized.affinity {
        AffinityIntent::RandomSubset { from, count } => {
            assert_eq!(
                from,
                BTreeSet::from([0usize, 1, 2, 3]),
                "from set must roundtrip unchanged"
            );
            assert_eq!(count, 2, "count must roundtrip unchanged");
        }
        other => panic!("expected RandomSubset after roundtrip, got {:?}", other),
    }
}
/// Serde JSON roundtrip for `AffinityIntent::RandomSubset`
/// embedded in `WorkloadConfig`. Mirrors the table-driven serde
/// pattern used for `WorkType` (#307): asserts the pool's CPU set
/// and `count` field survive serialize → JSON → deserialize.
/// Empty-pool / zero-count cases are excluded — those are
/// rejected at the spawn-time gate (see
/// `resolve_spawn_affinity_rejects_empty_random_pool` and
/// `resolve_spawn_affinity_rejects_zero_count_random`).
#[test]
fn workload_config_random_subset_serde_roundtrip() {
    let original = WorkloadConfig::default()
        .work_type(WorkType::SpinWait)
        .workers(4)
        .affinity(AffinityIntent::random_subset([0usize, 2, 4, 6], 2));
    let json = serde_json::to_string(&original).expect("serialize");
    let restored: WorkloadConfig = serde_json::from_str(&json).expect("deserialize");
    match restored.affinity {
        AffinityIntent::RandomSubset { from, count } => {
            assert_eq!(count, 2, "count must roundtrip");
            let cpus: Vec<usize> = from.into_iter().collect();
            assert_eq!(cpus, vec![0, 2, 4, 6], "pool must roundtrip");
        }
        other => panic!("expected RandomSubset, got {other:?}"),
    }
}
/// Composed groups must tag their `group_idx` in the bail
/// diagnostic so multi-group scenarios can locate the
/// offending entry. Primary IdleChurn is valid; the composed
/// group at index 1 has zero burst_duration. The error must
/// mention "group 1" so the caller knows WHICH composed entry
/// is malformed. The typed-error variant carries the same
/// `group_idx` field so callers can program against it without
/// parsing the message.
#[test]
fn idle_churn_zero_in_composed_group_rejects_with_group_idx() {
    let cfg = WorkloadConfig::default()
        .work_type(WorkType::IdleChurn {
            burst_duration: Duration::from_millis(1),
            sleep_duration: Duration::from_millis(5),
            precise_timing: false,
        })
        .workers(1)
        .with_composed(
            WorkSpec::default()
                .work_type(WorkType::IdleChurn {
                    burst_duration: Duration::ZERO,
                    sleep_duration: Duration::from_millis(5),
                    precise_timing: false,
                })
                .workers(1),
        );
    let err = WorkloadHandle::spawn(&cfg)
        .err()
        .expect("composed IdleChurn with burst_duration=ZERO must be rejected");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("burst_duration must be > 0"),
        "diagnostic must name the rejected field; got: {msg}",
    );
    assert!(
        msg.contains("group 1"),
        "diagnostic must tag the composed group_idx (1); got: {msg}",
    );
    let typed = err
        .downcast_ref::<WorkTypeValidationError>()
        .expect("error must downcast to WorkTypeValidationError");
    assert!(
        matches!(
            typed,
            WorkTypeValidationError::ZeroBurstDuration { group_idx: 1 }
        ),
        "expected ZeroBurstDuration {{ group_idx: 1 }} (composed group), got: {typed:?}",
    );
}
/// Composed-group rejection for `IpcVariance`: a primary
/// `Bursty` group is valid; the composed group at index 1
/// has zero `cold_iters`. The error must mention "group 1"
/// so the caller knows WHICH composed entry is malformed,
/// and the typed `ZeroIpcVarianceParam` variant must
/// carry `group_idx: 1`. Mirrors the IdleChurn composed
/// test pattern at
/// `idle_churn_zero_in_composed_group_rejects_with_group_idx`.
#[test]
fn ipc_variance_zero_in_composed_group_rejects_with_group_idx() {
    let cfg = WorkloadConfig::default()
        .work_type(WorkType::Bursty {
            burst_duration: Duration::from_millis(50),
            sleep_duration: Duration::from_millis(100),
        })
        .workers(1)
        .with_composed(
            WorkSpec::default()
                .work_type(WorkType::IpcVariance {
                    hot_iters: 1,
                    cold_iters: 0,
                    period_iters: 1,
                })
                .workers(1),
        );
    let err = WorkloadHandle::spawn(&cfg)
        .err()
        .expect("composed IpcVariance with cold_iters=0 must be rejected");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("group 1"),
        "diagnostic must tag the composed group_idx (1); got: {msg}",
    );
    let typed = err
        .downcast_ref::<WorkTypeValidationError>()
        .expect("error must downcast to WorkTypeValidationError");
    assert!(
        matches!(
            typed,
            WorkTypeValidationError::ZeroIpcVarianceParam {
                field: "cold_iters",
                group_idx: 1,
            }
        ),
        "expected ZeroIpcVarianceParam {{ cold_iters, group_idx: 1 }}; got: {typed:?}",
    );
}
