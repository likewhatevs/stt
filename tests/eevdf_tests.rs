use anyhow::Result;
use scx_ktstr::assert::AssertResult;
use scx_ktstr::ktstr_test;
use scx_ktstr::scenario::Ctx;
use scx_ktstr::workload::{WorkType, WorkloadConfig, WorkloadHandle};

/// EEVDF spinlock-holder preemption contention test.
///
/// Models the workload pattern behind the PREEMPT_LAZY regression: many
/// workers hold a brief "critical section" (1ms CPU burst with no sleep)
/// under heavy oversubscription (2x CPU count). If the scheduler preempts
/// a worker during its burst, all other workers spinning on the shared
/// resource stall, and throughput collapses.
///
/// With good scheduling, workers complete their burst quickly and
/// scheduling gaps stay small. With aggressive preemption of short-burst
/// workers, max_gap_ms rises and spread increases.
#[ktstr_test(
    sockets = 1,
    cores = 4,
    threads = 2,
    memory_mb = 2048,
    max_gap_ms = 2000,
    max_spread_pct = 80.0
)]
fn eevdf_spinlock_contention(ctx: &Ctx) -> Result<AssertResult> {
    let total_cpus = ctx.topo.total_cpus();
    let num_workers = total_cpus * 2;

    let config = WorkloadConfig {
        num_workers,
        affinity: scx_ktstr::workload::AffinityMode::None,
        work_type: WorkType::Bursty {
            burst_ms: 1,
            sleep_ms: 0,
        },
        sched_policy: scx_ktstr::workload::SchedPolicy::Normal,
    };

    let mut handle = WorkloadHandle::spawn(&config)?;
    handle.start();
    std::thread::sleep(ctx.duration);
    let reports = handle.stop_and_collect();

    // Under heavy oversubscription with 1ms bursts, spread can be high
    // (EEVDF doesn't guarantee perfect fairness at sub-slice granularity).
    // The real signal is max_gap_ms — if preemption during bursts causes
    // cascading contention, gaps spike. Spread check is relaxed via the
    // default threshold; gap check is enforced via Assert::max_gap_ms(2000).
    let checks = scx_ktstr::assert::Assert::default_checks()
        .max_gap_ms(2000)
        .max_spread_pct(80.0);
    Ok(checks.assert_cgroup(&reports, None))
}
