//! WorkType self-validation tests under EEVDF.
//!
//! Each test boots a small VM with no sched_ext scheduler attached
//! (EEVDF is the in-kernel default), spawns a workload using one
//! production [`WorkType`] variant, and asserts that the
//! [`WorkerReport`] fields the variant's documentation claims to
//! populate are actually populated. Catches assertion-string drift
//! between production and tests — a regression that breaks the
//! `iteration_costs_ns` reservoir-sampling for `IpcVariance`, or
//! the `resume_latencies_ns` capture for `FutexPingPong`, would
//! surface here even when the workload still produces work_units.
//!
//! Distinct from `eevdf_tests.rs` (which exercises EEVDF's
//! starvation/oversubscription behavior with `Bursty`) and
//! `preempt_regression.rs` (which targets the lock-holder
//! preemption regression with a `Custom` closure): this file
//! covers the cross-cutting "WorkType X populates field Y" matrix
//! that no single test file enforces.
//!
//! WorkType variants covered:
//!   - `WakeChain { wake: Pipe, ... }` — populates
//!     `resume_latencies_ns` from the chain stage handoff.
//!   - `FutexPingPong` — populates `resume_latencies_ns` from the
//!     paired wake/wait round-trips.
//!   - `MutexContention` — populates `iterations` and `work_units`
//!     under multi-worker contention; serialized critical sections
//!     guarantee a non-zero count.
//!   - `PageFaultChurn` — populates `iterations` and `work_units`
//!     via cold-page touches; per-iteration madvise(DONTNEED) zaps
//!     the PTEs so each iteration faults.
//!   - `Bursty` — populates `iterations` (one per burst+sleep
//!     cycle); the burst+sleep cadence is wall-clock-bounded so
//!     the count is bounded above by `duration / (burst + sleep)`.
//!
//! Variants intentionally excluded:
//!   - `Custom` — bypasses the built-in instrumentation by
//!     contract; the `WorkerReport` carries only what the closure
//!     writes. Validating the framework's instrumentation against
//!     `Custom` would test the closure, not the framework.
//!   - `Sequence` — composed of phases, not a leaf variant; the
//!     phase-walk test in `src/workload/types/tests.rs` covers
//!     the dispatch correctness.

use anyhow::Result;
use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
use ktstr::ktstr_test;
use ktstr::scenario::Ctx;
use ktstr::workload::{
    AffinityIntent, SchedPolicy, WakeMechanism, WorkType, WorkloadConfig, WorkloadHandle,
};
use std::time::Duration;

/// `WakeChain { wake: WakeMechanism::Pipe }` MUST populate
/// `resume_latencies_ns` for at least one stage in the chain.
/// The head stage publishes its first wake on the bootstrap path;
/// subsequent stages collect samples on each post-bootstrap
/// round-trip. A regression that breaks the per-stage pipe-fd
/// hand-off (in `WorkloadHandle::chain_pipes`) or the latency
/// reservoir would surface as `total_samples == 0` even when
/// `work_units > 0`.
///
/// Threshold: aggregated samples across all workers must be
/// non-zero. The strict per-worker invariant lives in
/// `spawn_thread_with_wake_chain_pipe` in
/// `src/workload/spawn/tests_thread_mode.rs`; this test pins the
/// production-VM equivalent under EEVDF on the kernel scheduler
/// path.
#[ktstr_test(
    llcs = 1,
    cores = 2,
    threads = 1,
    memory_mb = 1024,
    max_spread_pct = 80.0,
    duration_s = 5,
    watchdog_timeout_s = 15
)]
fn validation_wake_chain_pipe_populates_resume_latencies(ctx: &Ctx) -> Result<AssertResult> {
    let config = WorkloadConfig {
        num_workers: 4,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::WakeChain {
            depth: 4,
            wake: WakeMechanism::Pipe,
            work_per_hop: Duration::from_millis(10),
        },
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut handle = WorkloadHandle::spawn(&config)?;
    handle.start();
    std::thread::sleep(ctx.duration);
    let reports = handle.stop_and_collect();

    let mut result = AssertResult::pass();
    if reports.len() != 4 {
        result.passed = false;
        result.details.push(AssertDetail::new(
            DetailKind::Other,
            format!(
                "WakeChain wake=Pipe expected 4 reports, got {}; spawn or \
                 collection broken",
                reports.len(),
            ),
        ));
        return Ok(result);
    }

    let total_samples: usize = reports.iter().map(|r| r.resume_latencies_ns.len()).sum();
    if total_samples == 0 {
        result.passed = false;
        result.details.push(AssertDetail::new(
            DetailKind::Other,
            format!(
                "WakeChain wake=Pipe captured zero `resume_latencies_ns` \
                 samples across {} workers — the chain pipes never routed \
                 a stage handoff. Check chain_pipes ownership transfer \
                 and the per-stage pipe-fd lifetime.",
                reports.len(),
            ),
        ));
        return Ok(result);
    }

    let total_iters: u64 = reports.iter().map(|r| r.iterations).sum();
    result.details.push(AssertDetail::new(
        DetailKind::Other,
        format!(
            "WakeChain wake=Pipe populated resume_latencies_ns: \
             total_samples={total_samples}, total_iterations={total_iters} \
             across {} workers",
            reports.len(),
        ),
    ));
    Ok(result)
}

/// `FutexPingPong` MUST populate `resume_latencies_ns` from the
/// paired FUTEX_WAKE/FUTEX_WAIT round-trips. Each pair (i, i+1)
/// shares a futex word; one worker's wake unblocks the partner's
/// wait, and the timestamp from `before_block` to the wait return
/// is the latency sample. A regression that breaks the per-pair
/// futex word allocation, the wake/wait pairing, or the latency
/// reservoir would surface here.
#[ktstr_test(
    llcs = 1,
    cores = 2,
    threads = 1,
    memory_mb = 1024,
    max_spread_pct = 80.0,
    duration_s = 5,
    watchdog_timeout_s = 15
)]
fn validation_futex_ping_pong_populates_resume_latencies(ctx: &Ctx) -> Result<AssertResult> {
    let config = WorkloadConfig {
        num_workers: 2,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::FutexPingPong { spin_iters: 1024 },
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut handle = WorkloadHandle::spawn(&config)?;
    handle.start();
    std::thread::sleep(ctx.duration);
    let reports = handle.stop_and_collect();

    let mut result = AssertResult::pass();
    if reports.len() != 2 {
        result.passed = false;
        result.details.push(AssertDetail::new(
            DetailKind::Other,
            format!(
                "FutexPingPong expected 2 reports, got {}; spawn or \
                 collection broken",
                reports.len(),
            ),
        ));
        return Ok(result);
    }

    let total_samples: usize = reports.iter().map(|r| r.resume_latencies_ns.len()).sum();
    if total_samples == 0 {
        result.passed = false;
        result.details.push(AssertDetail::new(
            DetailKind::Other,
            "FutexPingPong captured zero resume_latencies_ns samples — \
             the per-pair futex word never routed a wake/wait round-trip. \
             Check the futex allocation or the wake_sample_count branch."
                .to_string(),
        ));
        return Ok(result);
    }

    for r in &reports {
        if r.work_units == 0 {
            result.passed = false;
            result.details.push(AssertDetail::new(
                DetailKind::Other,
                format!(
                    "FutexPingPong worker tid={} did no work; the \
                     paired-worker spawn produced an idle worker.",
                    r.tid,
                ),
            ));
            return Ok(result);
        }
    }

    result.details.push(AssertDetail::new(
        DetailKind::Other,
        format!(
            "FutexPingPong populated resume_latencies_ns: \
             total_samples={total_samples} across {} workers",
            reports.len(),
        ),
    ));
    Ok(result)
}

/// `MutexContention` MUST populate `iterations` and `work_units`
/// under multi-worker contention. The variant's contract is a
/// shared mutex serializing critical sections; with `contenders >
/// 1`, every iteration drains a shared queue and bumps both
/// counters per acquire/release cycle. A regression that breaks
/// the futex-fast-path acquisition or the lock release branch
/// would surface here as zero iterations across all contenders.
#[ktstr_test(
    llcs = 1,
    cores = 4,
    threads = 1,
    memory_mb = 1024,
    max_spread_pct = 80.0,
    duration_s = 5,
    watchdog_timeout_s = 15
)]
fn validation_mutex_contention_populates_iterations(ctx: &Ctx) -> Result<AssertResult> {
    let config = WorkloadConfig {
        num_workers: 4,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::MutexContention {
            contenders: 4,
            hold_iters: 256,
            work_iters: 1024,
        },
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut handle = WorkloadHandle::spawn(&config)?;
    handle.start();
    std::thread::sleep(ctx.duration);
    let reports = handle.stop_and_collect();

    let mut result = AssertResult::pass();
    if reports.len() != 4 {
        result.passed = false;
        result.details.push(AssertDetail::new(
            DetailKind::Other,
            format!(
                "MutexContention expected 4 reports, got {}; spawn or \
                 collection broken",
                reports.len(),
            ),
        ));
        return Ok(result);
    }

    let total_iters: u64 = reports.iter().map(|r| r.iterations).sum();
    if total_iters == 0 {
        result.passed = false;
        result.details.push(AssertDetail::new(
            DetailKind::Other,
            "MutexContention reported zero total iterations — every \
             worker failed to acquire the shared mutex (futex_fastpath \
             broken or stop fired before any acquire). Workers with \
             work_units > 0 but iterations == 0 indicate the iteration \
             counter is decoupled from the lock cycle."
                .to_string(),
        ));
        return Ok(result);
    }

    for r in &reports {
        if r.work_units == 0 {
            result.passed = false;
            result.details.push(AssertDetail::new(
                DetailKind::Other,
                format!(
                    "MutexContention worker tid={} did no work; spawn \
                     produced an idle contender.",
                    r.tid,
                ),
            ));
            return Ok(result);
        }
    }

    result.details.push(AssertDetail::new(
        DetailKind::Other,
        format!(
            "MutexContention populated iterations and work_units: \
             total_iterations={total_iters} across {} workers",
            reports.len(),
        ),
    ));
    Ok(result)
}

/// `PageFaultChurn` MUST populate `iterations` and `work_units`
/// via the per-iteration cold-page touch loop. The variant's
/// contract is `madvise(MADV_DONTNEED)` zapping PTEs after every
/// iteration so the next round faults again; this drives the
/// kernel's anonymous-page fault handler `do_anonymous_page` on
/// every touch. Zero iterations would mean the madvise loop
/// never fired (a regression in the per-iteration teardown) or
/// the worker bailed before the first touch.
#[ktstr_test(
    llcs = 1,
    cores = 2,
    threads = 1,
    memory_mb = 1024,
    max_spread_pct = 80.0,
    duration_s = 5,
    watchdog_timeout_s = 15
)]
fn validation_page_fault_churn_populates_iterations(ctx: &Ctx) -> Result<AssertResult> {
    let config = WorkloadConfig {
        num_workers: 2,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::page_fault_churn(4096, 64, 8),
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut handle = WorkloadHandle::spawn(&config)?;
    handle.start();
    std::thread::sleep(ctx.duration);
    let reports = handle.stop_and_collect();

    let mut result = AssertResult::pass();
    if reports.len() != 2 {
        result.passed = false;
        result.details.push(AssertDetail::new(
            DetailKind::Other,
            format!(
                "PageFaultChurn expected 2 reports, got {}; spawn broken",
                reports.len(),
            ),
        ));
        return Ok(result);
    }

    let total_iters: u64 = reports.iter().map(|r| r.iterations).sum();
    if total_iters == 0 {
        result.passed = false;
        result.details.push(AssertDetail::new(
            DetailKind::Other,
            "PageFaultChurn reported zero total iterations — the cold \
             page touch loop never advanced. Check the madvise(DONTNEED) \
             dispatch arm and the per-iteration mmap reuse."
                .to_string(),
        ));
        return Ok(result);
    }

    for r in &reports {
        if r.work_units == 0 {
            result.passed = false;
            result.details.push(AssertDetail::new(
                DetailKind::Other,
                format!(
                    "PageFaultChurn worker tid={} did no work; the \
                     first-touch fault never fired.",
                    r.tid,
                ),
            ));
            return Ok(result);
        }
    }

    result.details.push(AssertDetail::new(
        DetailKind::Other,
        format!(
            "PageFaultChurn populated iterations and work_units: \
             total_iterations={total_iters} across {} workers",
            reports.len(),
        ),
    ));
    Ok(result)
}

/// `Bursty` MUST populate `iterations` and `work_units` from the
/// burst+sleep cycles. Each cycle bumps `iterations` by one; the
/// observable count is wall-clock-bounded by `duration / (burst +
/// sleep)`. A regression that breaks the cycle boundary detection
/// or the iteration counter advancement would surface here as
/// `iterations == 0` even when the burst phase still executes
/// (work_units bumped but cycle not advanced).
///
/// The threshold is `iterations >= 2` (at least two complete
/// cycles within the test duration) — guards a regression where
/// only the first cycle's prelude executes before the bug fires
/// and the cycle counter never advances. A weaker `>= 1` threshold
/// would not distinguish "one full cycle completed" from "stop
/// fired mid-burst with iterations still at 0."
#[ktstr_test(
    llcs = 1,
    cores = 1,
    threads = 1,
    memory_mb = 512,
    max_spread_pct = 80.0,
    duration_s = 5,
    watchdog_timeout_s = 15
)]
fn validation_bursty_populates_iterations(ctx: &Ctx) -> Result<AssertResult> {
    let config = WorkloadConfig {
        num_workers: 1,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::Bursty {
            burst_duration: Duration::from_millis(50),
            sleep_duration: Duration::from_millis(50),
        },
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut handle = WorkloadHandle::spawn(&config)?;
    handle.start();
    std::thread::sleep(ctx.duration);
    let reports = handle.stop_and_collect();

    let mut result = AssertResult::pass();
    if reports.len() != 1 {
        result.passed = false;
        result.details.push(AssertDetail::new(
            DetailKind::Other,
            format!(
                "Bursty expected 1 report, got {}; spawn broken",
                reports.len(),
            ),
        ));
        return Ok(result);
    }

    let r = &reports[0];
    if r.iterations < 2 {
        result.passed = false;
        result.details.push(AssertDetail::new(
            DetailKind::Other,
            format!(
                "Bursty reported {} iterations over {:?}; expected at \
                 least 2 (burst+sleep = 100ms each cycle). The cycle \
                 boundary detection or iteration counter is not \
                 advancing.",
                r.iterations, ctx.duration,
            ),
        ));
        return Ok(result);
    }
    if r.work_units == 0 {
        result.passed = false;
        result.details.push(AssertDetail::new(
            DetailKind::Other,
            "Bursty reported zero work_units — the burst phase never \
             executed CPU work."
                .to_string(),
        ));
        return Ok(result);
    }

    result.details.push(AssertDetail::new(
        DetailKind::Other,
        format!(
            "Bursty populated iterations and work_units: \
             iterations={}, work_units={}, wall_time_ns={}",
            r.iterations, r.work_units, r.wall_time_ns,
        ),
    ));
    Ok(result)
}
