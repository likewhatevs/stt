//! Spawn-pipeline tests — integration group.

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
fn workload_config_default() {
    let c = WorkloadConfig::default();
    assert_eq!(c.num_workers, 1);
    assert!(matches!(c.work_type, WorkType::SpinWait));
    assert!(matches!(c.sched_policy, SchedPolicy::Normal));
    assert!(matches!(c.affinity, AffinityIntent::Inherit));
    // Default nice is 0 — `apply_nice(0)` short-circuits before
    // the syscall, preserving inherit-from-parent semantics.
    assert_eq!(c.nice, 0);
}
#[test]
fn workload_config_builder_setters_chain() {
    let cfg = WorkloadConfig::default()
        .workers(7)
        .work_type(WorkType::SpinWait)
        .sched_policy(SchedPolicy::Batch)
        .nice(5);
    assert_eq!(cfg.num_workers, 7);
    assert!(matches!(cfg.work_type, WorkType::SpinWait));
    assert!(matches!(cfg.sched_policy, SchedPolicy::Batch));
    assert_eq!(cfg.nice, 5);
}
#[test]
fn worker_report_serde_roundtrip() {
    let r = WorkerReport {
        tid: 42,
        work_units: 1000,
        cpu_time_ns: 5_000_000_000,
        wall_time_ns: 10_000_000_000,
        off_cpu_ns: 5_000_000_000,
        migration_count: 3,
        cpus_used: [0, 1, 2].into_iter().collect(),
        migrations: vec![Migration {
            at_ns: 100,
            from_cpu: 0,
            to_cpu: 1,
        }],
        max_gap_ms: 50,
        max_gap_cpu: 1,
        max_gap_at_ms: 500,
        resume_latencies_ns: vec![1000, 2000],
        wake_sample_total: 2,
        iteration_costs_ns: vec![3000, 4000, 5000],
        iteration_cost_sample_total: 3,
        iterations: 10,
        schedstat_run_delay_ns: 500_000,
        schedstat_run_count: 20,
        schedstat_cpu_time_ns: 4_000_000_000,
        completed: true,
        numa_pages: BTreeMap::new(),
        vmstat_numa_pages_migrated: 0,
        exit_info: None,
        // Non-default so the serde roundtrip proves the field
        // survives, not just that Default's value matches on
        // both sides.
        is_messenger: true,
        // Non-zero so the serde roundtrip proves group_idx
        // serializes/deserializes correctly. The composed
        // dispatch path tags reports with their group_idx; a
        // silent default-zero on serde would lose that tag.
        group_idx: 7,
        affinity_error: None,
    };
    let json = serde_json::to_string(&r).unwrap();
    let r2: WorkerReport = serde_json::from_str(&json).unwrap();
    assert_eq!(r.tid, r2.tid);
    assert_eq!(r.work_units, r2.work_units);
    assert_eq!(r.migration_count, r2.migration_count);
    assert_eq!(r.cpus_used, r2.cpus_used);
    assert_eq!(r.max_gap_ms, r2.max_gap_ms);
    assert_eq!(r.wake_sample_total, r2.wake_sample_total);
    assert_eq!(r.iteration_costs_ns, r2.iteration_costs_ns);
    assert_eq!(
        r.iteration_cost_sample_total,
        r2.iteration_cost_sample_total
    );
    assert_eq!(r.completed, r2.completed);
    assert_eq!(r.is_messenger, r2.is_messenger);
    assert_eq!(r.group_idx, r2.group_idx);
}
#[test]
fn migration_serde() {
    let m = Migration {
        at_ns: 12345,
        from_cpu: 0,
        to_cpu: 3,
    };
    let json = serde_json::to_string(&m).unwrap();
    let m2: Migration = serde_json::from_str(&json).unwrap();
    assert_eq!(m.at_ns, m2.at_ns);
    assert_eq!(m.from_cpu, m2.from_cpu);
    assert_eq!(m.to_cpu, m2.to_cpu);
}
#[test]
fn spawn_start_collect_integration() {
    let config = WorkloadConfig {
        num_workers: 2,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::SpinWait,
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    assert_eq!(h.worker_pids().len(), 2);
    h.start();
    std::thread::sleep(std::time::Duration::from_millis(200));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 2);
    for r in &reports {
        assert!(r.work_units > 0, "worker {} did no work", r.tid);
        assert!(r.wall_time_ns > 0);
        assert!(!r.cpus_used.is_empty());
    }
}
#[test]
fn spawn_auto_start_on_collect() {
    let config = WorkloadConfig {
        num_workers: 1,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::SpinWait,
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let h = WorkloadHandle::spawn(&config).unwrap();
    // Don't call start() - collect should auto-start
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 1);
}
#[test]
fn spawn_yield_heavy_produces_work() {
    let reports = spawn_and_collect_after(WorkType::YieldHeavy, 1, 200);
    assert_eq!(reports.len(), 1);
    assert!(reports[0].work_units > 0);
}
#[test]
fn spawn_mixed_produces_work() {
    let reports = spawn_and_collect_after(WorkType::Mixed, 1, 200);
    assert_eq!(reports.len(), 1);
    assert!(reports[0].work_units > 0);
}
/// Regression guard for the sign-cast bug: every pid returned
/// from `worker_pids()` must be a positive, live `pid_t` that
/// round-trips through `Pid::from_raw` + `kill(_, None)` (the
/// "exists" probe). A negative pid would silently broadcast
/// SIGKILL to a process group; a stale/reaped pid would fail the
/// probe with ESRCH. Either indicates storage upstream
/// re-introduced the u32 wraparound or dropped a child on the
/// floor.
#[test]
fn spawn_pids_fit_in_pid_t() {
    let config = WorkloadConfig {
        num_workers: 4,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::SpinWait,
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let h = WorkloadHandle::spawn(&config).unwrap();
    for pid in h.worker_pids() {
        assert!(pid > 0, "child pid must be positive, got {pid}");
        // Signal 0 (None) only checks existence; it does not
        // deliver anything. Proves the pid is a real, live
        // process we can address — not a negative-cast bomb.
        nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None)
            .unwrap_or_else(|e| panic!("spawned child pid {pid} not addressable: {e}"));
    }
}
#[test]
fn spawn_multiple_workers_distinct_pids() {
    let config = WorkloadConfig {
        num_workers: 4,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::SpinWait,
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    let pids = h.worker_pids();
    assert_eq!(pids.len(), 4);
    let unique: std::collections::HashSet<libc::pid_t> = pids.iter().copied().collect();
    assert_eq!(unique.len(), 4, "all worker PIDs should be distinct");
    h.start();
    std::thread::sleep(std::time::Duration::from_millis(500));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 4);
}
/// Spawn-time affinity gate: every accepted variant resolves to
/// the matching [`ResolvedAffinity`] shape, every rejected variant
/// bails with an actionable diagnostic. Pins the gate's accept /
/// reject contract so adding a new [`AffinityIntent`] variant
/// forces a deliberate decision here.
#[test]
fn resolve_spawn_affinity_accepts_no_context_variants() {
    // Inherit -> ResolvedAffinity::None
    let r =
        GroupParams::resolve_spawn_affinity(&AffinityIntent::Inherit, "WorkloadConfig::affinity")
            .expect("Inherit must resolve");
    assert!(matches!(r, ResolvedAffinity::None));

    // Exact -> ResolvedAffinity::Fixed (set preserved)
    let r = GroupParams::resolve_spawn_affinity(
        &AffinityIntent::exact([0, 2, 4]),
        "WorkloadConfig::affinity",
    )
    .expect("Exact must resolve");
    match r {
        ResolvedAffinity::Fixed(set) => {
            assert_eq!(set.len(), 3);
            assert!(set.contains(&0) && set.contains(&2) && set.contains(&4));
        }
        other => panic!("expected Fixed, got {:?}", other),
    }

    // RandomSubset -> ResolvedAffinity::Random (pool + count preserved)
    let r = GroupParams::resolve_spawn_affinity(
        &AffinityIntent::random_subset([0usize, 1, 2, 3], 2),
        "WorkloadConfig::affinity",
    )
    .expect("RandomSubset must resolve");
    match r {
        ResolvedAffinity::Random { from, count } => {
            assert_eq!(from.len(), 4);
            assert_eq!(count, 2);
        }
        other => panic!("expected Random, got {:?}", other),
    }
}
#[test]
fn resolve_spawn_affinity_rejects_topology_variants() {
    for variant in [
        AffinityIntent::SingleCpu,
        AffinityIntent::LlcAligned,
        AffinityIntent::CrossCgroup,
        AffinityIntent::SmtSiblingPair,
    ] {
        let err = GroupParams::resolve_spawn_affinity(&variant, "WorkloadConfig::affinity")
            .expect_err("topology-aware variant must bail at gate");
        let msg = err.to_string();
        assert!(
            msg.contains("requires scenario"),
            "diagnostic must mention scenario context, got: {msg}"
        );
        assert!(
            msg.contains("WorkloadConfig::affinity"),
            "diagnostic must include site, got: {msg}"
        );
    }
}
/// Empty `Exact` would yield a zero-mask `sched_setaffinity` call
/// that the kernel rejects with EINVAL. The gate bails with an
/// actionable diagnostic pointing the caller at `Inherit`.
#[test]
fn resolve_spawn_affinity_rejects_empty_exact() {
    let err = GroupParams::resolve_spawn_affinity(
        &AffinityIntent::Exact(BTreeSet::new()),
        "WorkloadConfig::affinity",
    )
    .expect_err("empty Exact must bail at gate");
    let msg = err.to_string();
    assert!(
        msg.contains("empty CPU set"),
        "diagnostic must name the empty-set condition, got: {msg}"
    );
    assert!(
        msg.contains("Inherit"),
        "diagnostic must point caller at Inherit, got: {msg}"
    );
    assert!(
        msg.contains("WorkloadConfig::affinity"),
        "diagnostic must include site, got: {msg}"
    );
}
/// `RandomSubset` with an empty pool leaves the spawn-time gate
/// nothing to sample from. The gate bails rather than silently
/// resolving to no affinity.
#[test]
fn resolve_spawn_affinity_rejects_empty_random_pool() {
    let err = GroupParams::resolve_spawn_affinity(
        &AffinityIntent::RandomSubset {
            from: BTreeSet::new(),
            count: 2,
        },
        "WorkloadConfig::affinity",
    )
    .expect_err("empty RandomSubset pool must bail at gate");
    let msg = err.to_string();
    assert!(
        msg.contains("empty pool"),
        "diagnostic must name the empty-pool condition, got: {msg}"
    );
    assert!(
        msg.contains("Inherit"),
        "diagnostic must point caller at Inherit, got: {msg}"
    );
    assert!(
        msg.contains("WorkloadConfig::affinity"),
        "diagnostic must include site, got: {msg}"
    );
}
/// `RandomSubset { count: 0 }` would draw zero CPUs per worker —
/// equivalent to no constraint. The gate bails rather than
/// silently resolving to no affinity.
#[test]
fn resolve_spawn_affinity_rejects_zero_count_random() {
    let err = GroupParams::resolve_spawn_affinity(
        &AffinityIntent::RandomSubset {
            from: BTreeSet::from([0usize, 1, 2]),
            count: 0,
        },
        "WorkloadConfig::affinity",
    )
    .expect_err("RandomSubset count=0 must bail at gate");
    let msg = err.to_string();
    assert!(
        msg.contains("count=0"),
        "diagnostic must name the zero-count condition, got: {msg}"
    );
    assert!(
        msg.contains("Inherit"),
        "diagnostic must point caller at Inherit, got: {msg}"
    );
    assert!(
        msg.contains("WorkloadConfig::affinity"),
        "diagnostic must include site, got: {msg}"
    );
}
/// Direct `WorkloadHandle::spawn` rejects each topology-aware
/// variant the gate guards. Verifies the bail propagates through
/// the spawn pipeline and the error message identifies the
/// offending field.
#[test]
fn spawn_rejects_topology_aware_variants_at_primary() {
    for variant in [
        AffinityIntent::SingleCpu,
        AffinityIntent::LlcAligned,
        AffinityIntent::CrossCgroup,
        AffinityIntent::SmtSiblingPair,
    ] {
        let label = format!("{variant:?}");
        let config = WorkloadConfig::default()
            .work_type(WorkType::SpinWait)
            .affinity(variant);
        // WorkloadHandle does not impl Debug, so expect_err is
        // unavailable — match on the Result directly.
        let err = match WorkloadHandle::spawn(&config) {
            Ok(_) => panic!(
                "topology-aware variant {label} must reject at \
                 WorkloadHandle::spawn"
            ),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("WorkloadConfig::affinity"),
            "diagnostic must name the field for {label}, got: {msg}"
        );
        assert!(
            msg.contains("requires scenario"),
            "diagnostic must mention scenario context for {label}, got: {msg}"
        );
    }
}
/// Direct `WorkloadHandle::spawn` accepts `RandomSubset` because
/// the caller supplies the `from` pool, so the gate has every
/// resolved field it needs without scenario context. Each worker
/// gets an independent draw at spawn time; this test verifies the
/// resolved affinity falls inside the pool.
#[test]
fn spawn_accepts_random_subset_directly() {
    let pool: Vec<usize> = (0..2).collect();
    let config = WorkloadConfig::default()
        .work_type(WorkType::SpinWait)
        .workers(2)
        .affinity(AffinityIntent::random_subset(pool.iter().copied(), 1));
    let mut h = WorkloadHandle::spawn(&config).expect("RandomSubset with explicit pool must spawn");
    h.start();
    std::thread::sleep(std::time::Duration::from_millis(200));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 2);
    for r in &reports {
        assert!(
            !r.cpus_used.is_empty(),
            "RandomSubset worker must run somewhere"
        );
        for cpu in &r.cpus_used {
            assert!(
                pool.contains(cpu),
                "worker used CPU {cpu} outside pool {pool:?}"
            );
        }
    }
}
#[test]
fn spawn_io_sync_write_produces_work() {
    let reports = spawn_and_collect_after(WorkType::IoSyncWrite, 1, 200);
    assert_eq!(reports.len(), 1);
    assert!(
        reports[0].work_units > 0,
        "IoSyncWrite worker {} did no work",
        reports[0].tid
    );
}
#[test]
fn spawn_io_rand_read_produces_work() {
    let reports = spawn_and_collect_after(WorkType::IoRandRead, 1, 200);
    assert_eq!(reports.len(), 1);
    assert!(
        reports[0].work_units > 0,
        "IoRandRead worker {} did no work",
        reports[0].tid
    );
}
#[test]
fn spawn_io_convoy_produces_work() {
    let reports = spawn_and_collect_after(WorkType::IoConvoy, 1, 200);
    assert_eq!(reports.len(), 1);
    assert!(
        reports[0].work_units > 0,
        "IoConvoy worker {} did no work",
        reports[0].tid
    );
}
#[test]
fn spawn_bursty_produces_work() {
    let reports = spawn_and_collect_after(
        WorkType::Bursty {
            burst_duration: Duration::from_millis(50),
            sleep_duration: Duration::from_millis(50),
        },
        1,
        300,
    );
    assert_eq!(reports.len(), 1);
    assert!(reports[0].work_units > 0);
}
#[test]
fn spawn_pipeio_produces_work() {
    let reports = spawn_and_collect_after(WorkType::PipeIo { burst_iters: 1024 }, 2, 300);
    assert_eq!(reports.len(), 2);
    for r in &reports {
        assert!(r.work_units > 0, "PipeIo worker {} did no work", r.tid);
    }
}
#[test]
fn spawn_pipeio_odd_workers_fails() {
    let config = WorkloadConfig {
        num_workers: 3,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::PipeIo { burst_iters: 1024 },
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let result = WorkloadHandle::spawn(&config);
    assert!(result.is_err(), "PipeIo with odd workers should fail");
    let msg = format!("{:#}", result.err().unwrap());
    assert!(
        msg.contains("divisible by 2"),
        "expected divisibility error: {msg}"
    );
}
#[test]
fn spawn_zero_workers() {
    let config = WorkloadConfig {
        num_workers: 0,
        ..Default::default()
    };
    let h = WorkloadHandle::spawn(&config).unwrap();
    assert!(h.worker_pids().is_empty());
    let reports = h.stop_and_collect();
    assert!(reports.is_empty());
}
#[test]
fn worker_pids_count_matches_num_workers() {
    for n in [1, 3, 5] {
        let config = WorkloadConfig {
            num_workers: n,
            ..Default::default()
        };
        let h = WorkloadHandle::spawn(&config).unwrap();
        assert_eq!(
            h.worker_pids().len(),
            n,
            "worker_pids().len() should match num_workers={n}"
        );
        drop(h);
    }
}
#[test]
fn worker_report_serde_edge_cases() {
    // Empty migrations and cpus_used
    let r = WorkerReport {
        tid: 0,
        work_units: 0,
        cpu_time_ns: 0,
        wall_time_ns: 0,
        off_cpu_ns: 0,
        migration_count: 0,
        cpus_used: BTreeSet::new(),
        migrations: vec![],
        max_gap_ms: 0,
        max_gap_cpu: 0,
        max_gap_at_ms: 0,
        resume_latencies_ns: vec![],
        wake_sample_total: 0,
        iteration_costs_ns: vec![],
        iteration_cost_sample_total: 0,
        iterations: 0,
        schedstat_run_delay_ns: 0,
        schedstat_run_count: 0,
        schedstat_cpu_time_ns: 0,
        completed: true,
        numa_pages: BTreeMap::new(),
        vmstat_numa_pages_migrated: 0,
        exit_info: None,
        is_messenger: false,
        group_idx: 0,
        affinity_error: None,
    };
    let json = serde_json::to_string(&r).unwrap();
    let r2: WorkerReport = serde_json::from_str(&json).unwrap();
    assert_eq!(r2.tid, 0);
    assert!(r2.cpus_used.is_empty());
    assert!(r2.migrations.is_empty());

    // Max u64 values
    let r = WorkerReport {
        tid: i32::MAX,
        work_units: u64::MAX,
        cpu_time_ns: u64::MAX,
        wall_time_ns: u64::MAX,
        off_cpu_ns: u64::MAX,
        migration_count: u64::MAX,
        cpus_used: [0, usize::MAX].into_iter().collect(),
        migrations: vec![],
        max_gap_ms: u64::MAX,
        max_gap_cpu: usize::MAX,
        max_gap_at_ms: u64::MAX,
        resume_latencies_ns: vec![],
        wake_sample_total: u64::MAX,
        iteration_costs_ns: vec![],
        iteration_cost_sample_total: u64::MAX,
        iterations: u64::MAX,
        schedstat_run_delay_ns: u64::MAX,
        schedstat_run_count: u64::MAX,
        schedstat_cpu_time_ns: u64::MAX,
        completed: true,
        numa_pages: BTreeMap::new(),
        vmstat_numa_pages_migrated: 0,
        exit_info: None,
        is_messenger: false,
        group_idx: usize::MAX,
        affinity_error: None,
    };
    let json = serde_json::to_string(&r).unwrap();
    let r2: WorkerReport = serde_json::from_str(&json).unwrap();
    assert_eq!(r2.work_units, u64::MAX);
    assert_eq!(r2.tid, i32::MAX);
}
#[test]
fn spawn_pipeio_four_workers() {
    let config = WorkloadConfig {
        num_workers: 4,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::PipeIo { burst_iters: 512 },
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    assert_eq!(h.worker_pids().len(), 4);
    h.start();
    std::thread::sleep(std::time::Duration::from_millis(300));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 4);
    for r in &reports {
        assert!(
            r.work_units > 0,
            "PipeIo 4-worker worker {} did no work",
            r.tid
        );
    }
}
#[test]
fn workload_config_debug_shows_field_values() {
    let c = WorkloadConfig {
        num_workers: 7,
        affinity: AffinityIntent::Exact([3].into_iter().collect()),
        work_type: WorkType::YieldHeavy,
        sched_policy: SchedPolicy::Batch,
        ..Default::default()
    };
    let s = format!("{:?}", c);
    assert!(s.contains("7"), "must show num_workers value");
    assert!(s.contains("Exact"), "must show affinity variant");
    assert!(s.contains("3"), "must show affinity CPU set");
    assert!(s.contains("YieldHeavy"), "must show work_type variant");
    assert!(s.contains("Batch"), "must show sched_policy variant");
}
#[test]
fn migration_debug_shows_field_values() {
    let m = Migration {
        at_ns: 99999,
        from_cpu: 3,
        to_cpu: 7,
    };
    let s = format!("{:?}", m);
    assert!(s.contains("99999"), "must show at_ns value");
    assert!(s.contains("3"), "must show from_cpu value");
    assert!(s.contains("7"), "must show to_cpu value");
    let m2 = Migration {
        at_ns: 1,
        from_cpu: 0,
        to_cpu: 1,
    };
    let s2 = format!("{:?}", m2);
    assert_ne!(
        s, s2,
        "different field values must produce different debug output"
    );
}
#[test]
fn worker_report_debug_shows_field_values() {
    let r = WorkerReport {
        tid: 42,
        work_units: 12345,
        cpu_time_ns: 1000,
        wall_time_ns: 2000,
        off_cpu_ns: 1000,
        migration_count: 3,
        cpus_used: [0, 5].into_iter().collect(),
        migrations: vec![],
        max_gap_ms: 77,
        max_gap_cpu: 5,
        max_gap_at_ms: 500,
        resume_latencies_ns: vec![],
        wake_sample_total: 0,
        iteration_costs_ns: vec![],
        iteration_cost_sample_total: 0,
        iterations: 0,
        schedstat_run_delay_ns: 0,
        schedstat_run_count: 0,
        schedstat_cpu_time_ns: 0,
        completed: true,
        numa_pages: BTreeMap::new(),
        vmstat_numa_pages_migrated: 0,
        exit_info: None,
        is_messenger: false,
        group_idx: 0,
        affinity_error: None,
    };
    let s = format!("{:?}", r);
    assert!(s.contains("42"), "must show tid value");
    assert!(s.contains("12345"), "must show work_units value");
    assert!(s.contains("77"), "must show max_gap_ms value");
    assert!(s.contains("5"), "must show max_gap_cpu value");
}
// -- WorkerReport edge cases --

#[test]
fn worker_report_off_cpu_ns_calculation() {
    // off_cpu_ns = wall_time_ns - cpu_time_ns
    let r = WorkerReport {
        tid: 1,
        work_units: 100,
        cpu_time_ns: 3_000_000_000,
        wall_time_ns: 5_000_000_000,
        off_cpu_ns: 2_000_000_000,
        migration_count: 0,
        cpus_used: [0].into_iter().collect(),
        migrations: vec![],
        max_gap_ms: 0,
        max_gap_cpu: 0,
        max_gap_at_ms: 0,
        resume_latencies_ns: vec![],
        wake_sample_total: 0,
        iteration_costs_ns: vec![],
        iteration_cost_sample_total: 0,
        iterations: 0,
        schedstat_run_delay_ns: 0,
        schedstat_run_count: 0,
        schedstat_cpu_time_ns: 0,
        completed: true,
        numa_pages: BTreeMap::new(),
        vmstat_numa_pages_migrated: 0,
        exit_info: None,
        is_messenger: false,
        group_idx: 0,
        affinity_error: None,
    };
    assert_eq!(r.off_cpu_ns, r.wall_time_ns - r.cpu_time_ns);
}
#[test]
fn migration_serde_multiple() {
    let migrations = vec![
        Migration {
            at_ns: 100,
            from_cpu: 0,
            to_cpu: 1,
        },
        Migration {
            at_ns: 200,
            from_cpu: 1,
            to_cpu: 2,
        },
        Migration {
            at_ns: 300,
            from_cpu: 2,
            to_cpu: 0,
        },
    ];
    let json = serde_json::to_string(&migrations).unwrap();
    let m2: Vec<Migration> = serde_json::from_str(&json).unwrap();
    assert_eq!(m2.len(), 3);
    assert_eq!(m2[0].from_cpu, 0);
    assert_eq!(m2[2].to_cpu, 0);
}
// -- snapshot_iterations tests --

#[test]
fn snapshot_iterations_empty_handle() {
    let config = WorkloadConfig {
        num_workers: 0,
        ..Default::default()
    };
    let h = WorkloadHandle::spawn(&config).unwrap();
    assert!(h.snapshot_iterations().is_empty());
    drop(h);
}
#[test]
fn snapshot_iterations_running_workers() {
    let config = WorkloadConfig {
        num_workers: 2,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::SpinWait,
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    h.start();
    std::thread::sleep(std::time::Duration::from_millis(200));
    let iters = h.snapshot_iterations();
    assert_eq!(iters.len(), 2);
    // After 200ms of SpinWait, workers should have done iterations.
    for (i, &v) in iters.iter().enumerate() {
        assert!(v > 0, "worker {i} should have iterations > 0, got {v}");
    }
    drop(h);
}
#[test]
fn spawn_cache_pressure_produces_work() {
    let reports = spawn_and_collect_after(
        WorkType::CachePressure {
            size_kb: 32,
            stride: 64,
        },
        1,
        200,
    );
    assert_eq!(reports.len(), 1);
    assert!(reports[0].work_units > 0);
}
#[test]
fn spawn_cache_yield_produces_work() {
    let reports = spawn_and_collect_after(
        WorkType::CacheYield {
            size_kb: 32,
            stride: 64,
        },
        1,
        200,
    );
    assert_eq!(reports.len(), 1);
    assert!(reports[0].work_units > 0);
}
#[test]
fn spawn_cache_pipe_produces_work() {
    let reports = spawn_and_collect_after(
        WorkType::CachePipe {
            size_kb: 32,
            burst_iters: 1024,
        },
        2,
        300,
    );
    assert_eq!(reports.len(), 2);
    for r in &reports {
        assert!(r.work_units > 0, "CachePipe worker {} did no work", r.tid);
    }
}
#[test]
fn spawn_sequence_produces_work() {
    let reports = spawn_and_collect_after(
        WorkType::Sequence {
            first: Phase::Spin(Duration::from_millis(10)),
            rest: vec![Phase::Yield(Duration::from_millis(10))],
        },
        1,
        200,
    );
    assert_eq!(reports.len(), 1);
    assert!(reports[0].work_units > 0);
}
#[test]
fn spawn_custom_produces_work() {
    let config = WorkloadConfig {
        num_workers: 1,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::custom("test_spin", custom_spin_fn),
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    h.start();
    std::thread::sleep(std::time::Duration::from_millis(200));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 1);
    assert!(
        reports[0].work_units > 0,
        "Custom worker did no work: work_units={}",
        reports[0].work_units
    );
    assert!(reports[0].wall_time_ns > 0);
    assert!(
        reports.iter().all(|r| r.completed),
        "every worker report on the live / non-sentinel path \
         must carry completed=true — pairs with the
         completed=false assertion in \
         stop_and_collect_reaps_grandchild_from_panicking_custom_closure",
    );
}
/// `CloneMode::Fork + WorkType::ForkExit` is the well-tested
/// pair (existing test
/// `stop_and_collect_reaps_grandchild_from_panicking_custom_closure`
/// pins the fork mode's panic shape). This regression guard
/// proves the new D5 incompatibility check does NOT also reject
/// the legitimate Fork+ForkExit combination.
#[test]
fn spawn_fork_with_forkexit_succeeds() {
    let config = WorkloadConfig {
        num_workers: 1,
        clone_mode: CloneMode::Fork,
        work_type: WorkType::ForkExit,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).expect("Fork + ForkExit must remain valid");
    h.start();
    std::thread::sleep(std::time::Duration::from_millis(100));
    let _ = h.stop_and_collect();
}
/// Guards three invariants of [`WorkType::PageFaultChurn`]:
///
/// 1. Every spawned worker produces non-zero `work_units` and
///    `iterations` (sanity — holds under the pre-fix bug too,
///    so it's a basic progress check, not a regression guard).
/// 2. `iter_slot` (host-side iteration sampling, read via
///    [`WorkloadHandle::snapshot_iterations`]) ADVANCES during
///    the run. Asserted as a positive delta between two
///    snapshots taken at 100 ms and 250 ms. A delta is
///    insensitive to worker start-up latency (the test would
///    otherwise race against workers whose first outer iter
///    lands after the first snapshot). Pre-fix, PageFaultChurn
///    used an inner `while !STOP` loop that bypassed the
///    iter_slot publish in the outer `worker_main` loop, so
///    both snapshots were pinned at 0 and the delta would be 0.
/// 3. On multi-CPU hosts, at least one worker records ≥ 1
///    migration. With `num_workers = available_parallelism() + 1`
///    the workload oversubscribes by one, forcing at least one
///    context switch and CPU re-dispatch in any realistic
///    scheduler; combined with the migration check in the
///    outer `worker_main` loop (gated on
///    `work_units.is_multiple_of(1024)`) firing every 64 outer
///    iters for this test's parameters (touches_per_cycle=16 +
///    spin_iters=32 = 48 work_units/iter,
///    gcd(48, 1024) = 16, period = 1024/16 = 64; the default
///    16-iter period documented in
///    doc/guide/src/architecture/workers.md assumes
///    default params 256+64=320 instead), this puts the
///    assertion well above the flake threshold. Gated on
///    `available_parallelism() > 1` because single-CPU
///    sandboxes legitimately report 0 migrations.
#[test]
fn spawn_page_fault_churn_produces_work() {
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    // Oversubscribe by one to force CPU sharing even on fully
    // idle hosts, so the migration-count assertion below has
    // a reliable signal.
    let num_workers = num_cpus + 1;
    let config = WorkloadConfig {
        num_workers,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::PageFaultChurn {
            region_kb: 64,
            touches_per_cycle: 16,
            spin_iters: 32,
        },
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    h.start();
    // Delta-based iter_slot assertion. Pre-fix these snapshots
    // were both 0 for PageFaultChurn (inner `while !STOP`
    // blocked the iter_slot publish in the outer `worker_main`
    // loop). Post-fix the outer loop
    // updates iter_slot every iteration, so the 150 ms gap
    // between snap1 and snap2 observes many iterations'
    // worth of progress.
    std::thread::sleep(std::time::Duration::from_millis(100));
    let snap1 = h.snapshot_iterations();
    std::thread::sleep(std::time::Duration::from_millis(150));
    let snap2 = h.snapshot_iterations();
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), num_workers);
    assert_eq!(snap1.len(), num_workers);
    assert_eq!(snap2.len(), num_workers);
    for i in 0..num_workers {
        let delta = snap2[i].saturating_sub(snap1[i]);
        assert!(
            delta > 0,
            "worker {i} iter_slot delta between 100 ms and 250 ms \
             was 0 (snap1={}, snap2={}); outer loop is not \
             advancing, indicating a regression that restores \
             the inner-`while !STOP` bug",
            snap1[i],
            snap2[i],
        );
    }
    // Basic progress sanity — holds even under the pre-fix
    // bug (inner loop still incremented work_units and
    // iterations), so this is not a regression guard for the
    // inner-while bug. Delta assertion above covers that.
    for r in &reports {
        assert!(
            r.work_units > 0,
            "PageFaultChurn worker {} did no work",
            r.tid
        );
        assert!(
            r.iterations > 0,
            "PageFaultChurn worker {} final iterations = 0",
            r.tid
        );
    }
    if num_cpus > 1 {
        let total_migrations: u64 = reports.iter().map(|r| r.migration_count).sum();
        assert!(
            total_migrations > 0,
            "expected ≥ 1 migration across {num_workers} \
             oversubscribed workers on {num_cpus}-cpu host; 0 \
             total migrations suggests the outer migration \
             check at work_units.is_multiple_of(1024) isn't \
             firing, indicating a regression that restores the \
             inner-`while !STOP` bug"
        );
    }
}
#[test]
fn spawn_mutex_contention_produces_work() {
    let reports = spawn_and_collect_after(
        WorkType::MutexContention {
            contenders: 4,
            hold_iters: 64,
            work_iters: 256,
        },
        4,
        500,
    );
    assert_eq!(reports.len(), 4);
    for r in &reports {
        assert!(
            r.work_units > 0,
            "MutexContention worker {} did no work",
            r.tid
        );
    }
}
#[test]
fn spawn_mutex_contention_bad_worker_count_fails() {
    let config = WorkloadConfig {
        num_workers: 3,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::MutexContention {
            contenders: 4,
            hold_iters: 256,
            work_iters: 1024,
        },
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let result = WorkloadHandle::spawn(&config);
    assert!(result.is_err());
    let msg = format!("{:#}", result.err().unwrap());
    assert!(
        msg.contains("divisible by 4"),
        "expected divisibility error: {msg}"
    );
}
/// `WorkType::IpcVariance` spawn-side rejection mirrors the
/// constructor: a struct-literal with zero `hot_iters`
/// fails at [`WorkloadHandle::spawn`] with the typed error.
#[test]
fn ipc_variance_spawn_rejects_zero_hot_iters() {
    let cfg = WorkloadConfig {
        num_workers: 1,
        work_type: WorkType::IpcVariance {
            hot_iters: 0,
            cold_iters: 1,
            period_iters: 1,
        },
        ..Default::default()
    };
    let err = WorkloadHandle::spawn(&cfg)
        .err()
        .expect("IpcVariance hot_iters=0 must be rejected at spawn");
    let typed = err
        .downcast_ref::<WorkTypeValidationError>()
        .expect("error must downcast to WorkTypeValidationError");
    assert!(
        matches!(
            typed,
            WorkTypeValidationError::ZeroIpcVarianceParam {
                field: "hot_iters",
                group_idx: 0,
            }
        ),
        "expected ZeroIpcVarianceParam {{ hot_iters }} at spawn; got: {typed:?}",
    );
}
/// `WorkType::IpcVariance` spawn-side rejection mirrors the
/// constructor for `cold_iters`. Zero `cold_iters` would
/// produce a cold phase that does no memory work — the
/// scheduler-observable IPC variance the variant is named
/// for would not exist. The same `ZeroIpcVarianceParam`
/// variant fires from both the constructor and the spawn
/// gate.
#[test]
fn ipc_variance_spawn_rejects_zero_cold_iters() {
    let cfg = WorkloadConfig {
        num_workers: 1,
        work_type: WorkType::IpcVariance {
            hot_iters: 1,
            cold_iters: 0,
            period_iters: 1,
        },
        ..Default::default()
    };
    let err = WorkloadHandle::spawn(&cfg)
        .err()
        .expect("IpcVariance cold_iters=0 must be rejected at spawn");
    let typed = err
        .downcast_ref::<WorkTypeValidationError>()
        .expect("error must downcast to WorkTypeValidationError");
    assert!(
        matches!(
            typed,
            WorkTypeValidationError::ZeroIpcVarianceParam {
                field: "cold_iters",
                group_idx: 0,
            }
        ),
        "expected ZeroIpcVarianceParam {{ cold_iters }} at spawn; got: {typed:?}",
    );
}
/// `WorkType::IpcVariance` spawn-side rejection mirrors the
/// constructor for `period_iters`. Zero `period_iters`
/// would skip the inner loop entirely so the variant
/// produces no hot/cold alternation — the worker still
/// iterates the outer loop but performs no work. Pinning
/// the rejection prevents that silent degeneration.
#[test]
fn ipc_variance_spawn_rejects_zero_period_iters() {
    let cfg = WorkloadConfig {
        num_workers: 1,
        work_type: WorkType::IpcVariance {
            hot_iters: 1,
            cold_iters: 1,
            period_iters: 0,
        },
        ..Default::default()
    };
    let err = WorkloadHandle::spawn(&cfg)
        .err()
        .expect("IpcVariance period_iters=0 must be rejected at spawn");
    let typed = err
        .downcast_ref::<WorkTypeValidationError>()
        .expect("error must downcast to WorkTypeValidationError");
    assert!(
        matches!(
            typed,
            WorkTypeValidationError::ZeroIpcVarianceParam {
                field: "period_iters",
                group_idx: 0,
            }
        ),
        "expected ZeroIpcVarianceParam {{ period_iters }} at spawn; got: {typed:?}",
    );
}

/// Build a fully populated `WorkerReport` with non-default values
/// in every field. Anchoring tests on this shape proves the wire
/// format carries every byte the worker writes — a missing field on
/// either side would shift the positional bincode decoder onto the
/// next field's bytes (silent corruption per the doc on
/// `exit_info` / `affinity_error`).
fn fully_populated_report() -> WorkerReport {
    WorkerReport {
        tid: 12345,
        work_units: 7_777_777,
        cpu_time_ns: 3_141_592_653,
        wall_time_ns: 6_283_185_307,
        off_cpu_ns: 3_141_592_654,
        migration_count: 9,
        cpus_used: [0usize, 3, 5, 7].into_iter().collect(),
        migrations: vec![
            Migration {
                at_ns: 100,
                from_cpu: 0,
                to_cpu: 3,
            },
            Migration {
                at_ns: 250,
                from_cpu: 3,
                to_cpu: 5,
            },
        ],
        max_gap_ms: 42,
        max_gap_cpu: 5,
        max_gap_at_ms: 999,
        resume_latencies_ns: vec![1_000, 2_000, 3_000, 4_000],
        wake_sample_total: 4,
        iteration_costs_ns: vec![10, 20, 30],
        iteration_cost_sample_total: 3,
        iterations: 1024,
        schedstat_run_delay_ns: 555_000,
        schedstat_run_count: 73,
        schedstat_cpu_time_ns: 8_000_000_000,
        completed: true,
        numa_pages: [(0usize, 100u64), (1usize, 200u64)].into_iter().collect(),
        vmstat_numa_pages_migrated: 17,
        exit_info: None,
        is_messenger: true,
        group_idx: 4,
        affinity_error: None,
    }
}

/// Compare two `WorkerReport`s field-by-field. `WorkerReport` does
/// not derive `PartialEq`, so the roundtrip tests must check every
/// field explicitly. A missing assertion would silently let a
/// mismatched field through — the same hazard the production
/// bincode pipe avoids by emitting every field on every call.
fn assert_worker_report_eq(a: &WorkerReport, b: &WorkerReport) {
    assert_eq!(a.tid, b.tid, "tid");
    assert_eq!(a.work_units, b.work_units, "work_units");
    assert_eq!(a.cpu_time_ns, b.cpu_time_ns, "cpu_time_ns");
    assert_eq!(a.wall_time_ns, b.wall_time_ns, "wall_time_ns");
    assert_eq!(a.off_cpu_ns, b.off_cpu_ns, "off_cpu_ns");
    assert_eq!(a.migration_count, b.migration_count, "migration_count");
    assert_eq!(a.cpus_used, b.cpus_used, "cpus_used");
    assert_eq!(a.migrations.len(), b.migrations.len(), "migrations.len");
    for (i, (am, bm)) in a.migrations.iter().zip(b.migrations.iter()).enumerate() {
        assert_eq!(am.at_ns, bm.at_ns, "migrations[{i}].at_ns");
        assert_eq!(am.from_cpu, bm.from_cpu, "migrations[{i}].from_cpu");
        assert_eq!(am.to_cpu, bm.to_cpu, "migrations[{i}].to_cpu");
    }
    assert_eq!(a.max_gap_ms, b.max_gap_ms, "max_gap_ms");
    assert_eq!(a.max_gap_cpu, b.max_gap_cpu, "max_gap_cpu");
    assert_eq!(a.max_gap_at_ms, b.max_gap_at_ms, "max_gap_at_ms");
    assert_eq!(
        a.resume_latencies_ns, b.resume_latencies_ns,
        "resume_latencies_ns"
    );
    assert_eq!(
        a.wake_sample_total, b.wake_sample_total,
        "wake_sample_total"
    );
    assert_eq!(
        a.iteration_costs_ns, b.iteration_costs_ns,
        "iteration_costs_ns"
    );
    assert_eq!(
        a.iteration_cost_sample_total, b.iteration_cost_sample_total,
        "iteration_cost_sample_total"
    );
    assert_eq!(a.iterations, b.iterations, "iterations");
    assert_eq!(
        a.schedstat_run_delay_ns, b.schedstat_run_delay_ns,
        "schedstat_run_delay_ns"
    );
    assert_eq!(
        a.schedstat_run_count, b.schedstat_run_count,
        "schedstat_run_count"
    );
    assert_eq!(
        a.schedstat_cpu_time_ns, b.schedstat_cpu_time_ns,
        "schedstat_cpu_time_ns"
    );
    assert_eq!(a.completed, b.completed, "completed");
    assert_eq!(a.numa_pages, b.numa_pages, "numa_pages");
    assert_eq!(
        a.vmstat_numa_pages_migrated, b.vmstat_numa_pages_migrated,
        "vmstat_numa_pages_migrated"
    );
    match (&a.exit_info, &b.exit_info) {
        (None, None) => {}
        (Some(WorkerExitInfo::Exited(x)), Some(WorkerExitInfo::Exited(y))) => {
            assert_eq!(x, y, "exit_info Exited code");
        }
        (Some(WorkerExitInfo::Signaled(x)), Some(WorkerExitInfo::Signaled(y))) => {
            assert_eq!(x, y, "exit_info Signaled signum");
        }
        (Some(WorkerExitInfo::TimedOut), Some(WorkerExitInfo::TimedOut)) => {}
        (Some(WorkerExitInfo::WaitFailed(x)), Some(WorkerExitInfo::WaitFailed(y))) => {
            assert_eq!(x, y, "exit_info WaitFailed message");
        }
        (Some(WorkerExitInfo::Panicked(x)), Some(WorkerExitInfo::Panicked(y))) => {
            assert_eq!(x, y, "exit_info Panicked message");
        }
        (other_a, other_b) => {
            panic!("exit_info variant mismatch: a={other_a:?} b={other_b:?}");
        }
    }
    assert_eq!(a.is_messenger, b.is_messenger, "is_messenger");
    assert_eq!(a.group_idx, b.group_idx, "group_idx");
    assert_eq!(a.affinity_error, b.affinity_error, "affinity_error");
}

/// Roundtrip a fully populated `WorkerReport` (`exit_info=None`)
/// through `bincode::serde::encode_to_vec` →
/// `bincode::serde::decode_from_slice` with
/// `bincode::config::standard()` — the exact codec the
/// worker→parent report pipe uses (mod.rs: `encode_to_vec` at the
/// worker child, `decode_from_slice` at `stop_and_collect`). Every
/// field is asserted equal post-decode; a missing field on either
/// side would corrupt subsequent fields silently per the
/// `exit_info` / `affinity_error` doc warnings.
#[test]
fn worker_report_bincode_roundtrip() {
    let report = fully_populated_report();
    let bytes =
        bincode::serde::encode_to_vec(&report, bincode::config::standard()).expect("encode");
    let (decoded, consumed): (WorkerReport, usize) =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).expect("decode");
    assert_eq!(consumed, bytes.len(), "decoder must consume entire frame");
    assert_worker_report_eq(&report, &decoded);
}

/// Roundtrip a sentinel-shaped `WorkerReport`: `exit_info =
/// Some(Exited(1))` (the catch_unwind panic-arm shape per
/// stop_and_collect's sentinel doc) and `affinity_error =
/// Some("EINVAL")` (the EINVAL-from-cpuset shape per the
/// `affinity_error` doc). Confirms the `Option<…>` tag bytes
/// round-trip through bincode without losing the inner payload —
/// the bincode positional encoding emits the tag whether or not
/// the option is populated, so a sentinel and a live-worker frame
/// must each decode with their original shape.
#[test]
fn worker_report_bincode_sentinel_roundtrip() {
    let mut report = fully_populated_report();
    report.exit_info = Some(WorkerExitInfo::Exited(1));
    report.affinity_error = Some("EINVAL".to_string());
    let bytes =
        bincode::serde::encode_to_vec(&report, bincode::config::standard()).expect("encode");
    let (decoded, consumed): (WorkerReport, usize) =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).expect("decode");
    assert_eq!(consumed, bytes.len(), "decoder must consume entire frame");
    assert!(
        matches!(decoded.exit_info, Some(WorkerExitInfo::Exited(1))),
        "exit_info must roundtrip as Exited(1); got {:?}",
        decoded.exit_info
    );
    assert_eq!(decoded.affinity_error.as_deref(), Some("EINVAL"));
    assert_worker_report_eq(&report, &decoded);
}

/// Roundtrip a `Vec<WorkerReport>` — the pcomm-container shape
/// the leader process writes to the report pipe before exit. The
/// container holds one report per worker thread; per #6 the
/// container will move from serde_json to bincode so a single
/// codec governs both the fork-mode single-report and pcomm-mode
/// multi-report payloads. This test pins that wire format.
#[test]
fn vec_worker_report_bincode_roundtrip() {
    let mut second = fully_populated_report();
    second.tid = 67890;
    second.group_idx = 5;
    second.is_messenger = false;
    second.exit_info = Some(WorkerExitInfo::Signaled(9));
    let reports: Vec<WorkerReport> = vec![fully_populated_report(), second];
    let bytes =
        bincode::serde::encode_to_vec(&reports, bincode::config::standard()).expect("encode");
    let (decoded, consumed): (Vec<WorkerReport>, usize) =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).expect("decode");
    assert_eq!(consumed, bytes.len(), "decoder must consume entire frame");
    assert_eq!(decoded.len(), reports.len(), "vec length must roundtrip");
    for (i, (a, b)) in reports.iter().zip(decoded.iter()).enumerate() {
        assert_worker_report_eq(a, b);
        assert_eq!(a.tid, b.tid, "report[{i}] tid");
    }
}

/// A truncated bincode frame must surface as a decode error, not a
/// silent partial decode. The parent's pipe-drain code at
/// `stop_and_collect` relies on this to detect short writes (worker
/// died mid-flush) and synthesize a sentinel report. If
/// `decode_from_slice` returned `Ok` on a partial buffer the
/// parent would publish a half-populated report with garbage in
/// the trailing fields.
#[test]
fn truncated_frame_decodes_to_err() {
    let report = fully_populated_report();
    let bytes =
        bincode::serde::encode_to_vec(&report, bincode::config::standard()).expect("encode");
    assert!(
        bytes.len() >= 2,
        "encoded report must be at least 2 bytes; got {}",
        bytes.len()
    );
    let truncated = &bytes[..bytes.len() / 2];
    let result: Result<(WorkerReport, usize), _> =
        bincode::serde::decode_from_slice(truncated, bincode::config::standard());
    assert!(
        result.is_err(),
        "truncated frame must decode to Err; got Ok({:?})",
        result.ok()
    );
}

/// An empty payload must surface as a decode error. The parent's
/// pipe-drain code observes an empty buffer when the worker dies
/// before writing anything (post-fork crash, OOM kill, or a
/// SIGKILL that fired before bincode encoding) and the sentinel
/// path depends on the empty-slice decode failing — otherwise a
/// zero-byte buffer would round-trip as a default `WorkerReport`
/// and mask the real failure.
#[test]
fn empty_payload_decodes_to_err() {
    let result: Result<(WorkerReport, usize), _> =
        bincode::serde::decode_from_slice(&[], bincode::config::standard());
    assert!(
        result.is_err(),
        "empty payload must decode to Err; got Ok({:?})",
        result.ok()
    );
}
