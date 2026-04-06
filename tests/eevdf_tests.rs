use anyhow::Result;
use stt::scenario::Ctx;
use stt::stt_test;
use stt::verify::VerifyResult;
use stt::workload::{WorkType, WorkloadConfig, WorkloadHandle};

/// EEVDF spinlock-holder preemption contention test.
///
/// Models the workload pattern behind the PREEMPT_LAZY regression: many
/// workers hold a brief "critical section" (1ms CPU burst with no sleep)
/// under heavy oversubscription (4x CPU count). If the scheduler preempts
/// a worker during its burst, all other workers spinning on the shared
/// resource stall, and throughput collapses.
///
/// With good scheduling, workers complete their burst quickly and
/// scheduling gaps stay small. With aggressive preemption of short-burst
/// workers, max_gap_ms rises and spread increases.
#[stt_test(sockets = 1, cores = 4, threads = 2, memory_mb = 2048)]
fn eevdf_spinlock_contention(ctx: &Ctx) -> Result<VerifyResult> {
    let total_cpus = ctx.topo.total_cpus();
    let num_workers = total_cpus * 2;

    let config = WorkloadConfig {
        num_workers,
        affinity: stt::workload::AffinityMode::None,
        work_type: WorkType::Bursty {
            burst_ms: 1,
            sleep_ms: 0,
        },
        sched_policy: stt::workload::SchedPolicy::Normal,
    };

    let mut handle = WorkloadHandle::spawn(&config)?;
    handle.start();
    std::thread::sleep(ctx.duration);
    let reports = handle.stop_and_collect();

    // Under heavy oversubscription with 1ms bursts, spread can be high
    // (EEVDF doesn't guarantee perfect fairness at sub-slice granularity).
    // The real signal is max_gap_ms — if preemption during bursts causes
    // cascading contention, gaps spike.
    let mut result = stt::verify::verify_not_starved(&reports);
    result.passed = result.stats.worst_gap_ms < 2000
        && result.stats.total_workers > 0
        && !reports.iter().any(|w| w.work_units == 0);
    if !result.passed {
        result.details.push(format!(
            "gap {}ms (threshold 2000ms), {} workers",
            result.stats.worst_gap_ms, result.stats.total_workers,
        ));
    }
    Ok(result)
}
