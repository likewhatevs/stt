//! Production worker-thread integration tests.
//!
//! End-to-end tests for [`CloneMode::Thread`] running real
//! production [`WorkType`] variants under a real Linux kernel
//! (in-VM via `#[ktstr_test]`). The unit-level Thread-mode
//! coverage in `src/workload/spawn/tests_thread_mode.rs` runs
//! on the host's `cargo nextest run` harness; this file pins
//! the same workloads under the guest VM so the kernel-side
//! pthreads, futex, mmap, and madvise paths are exercised on
//! the same kernel image the rest of ktstr targets.
//!
//! Why integration-tier coverage matters:
//!   - Host-side unit tests run under `cargo nextest`'s parent
//!     process. The futex / pipe / mmap behavior depends on the
//!     host kernel, NOT the guest kernel under test.
//!   - The guest kernel may have a different version,
//!     CONFIG_FUTEX2 setting, or PREEMPT/RT configuration. Thread
//!     mode produces siblings inside the guest's `init` tgid; a
//!     regression in CONFIG_PREEMPT setting that only surfaces
//!     under high contention would not show up in host-side
//!     unit tests.
//!   - The KTSTR_TESTS distributed_slice dispatch path inside
//!     the guest is what runs in production. The host-side
//!     `#[test]` path runs the workload but not the surrounding
//!     dispatch infrastructure.
//!
//! WorkType variants covered (each as a Thread-mode worker):
//!   - `SpinWait` — pure CPU; pins the basic Thread dispatch
//!     and the per-thread `gettid()` publish path under VM
//!     conditions.
//!   - `FutexPingPong` — futex wake/wait; pins the shared-fd
//!     futex word allocation and the wait/wake pairing under
//!     the guest kernel's futex implementation.
//!   - `PageFaultChurn` — anonymous-page faults; pins the per-
//!     iteration madvise(DONTNEED) → fault loop under the
//!     guest kernel's anon-fault path.
//!   - `MutexContention` — multi-worker contention; pins the
//!     futex-fast-path acquire and the lock-release wake under
//!     guest-kernel scheduling.
//!
//! Variants intentionally excluded:
//!   - `ForkExit` — bails at spawn under Thread mode (the worker
//!     calls `_exit` which tears down the whole tgid). Coverage
//!     for the rejection lives in
//!     `spawn_thread_with_forkexit_rejected_at_spawn_time`.
//!   - `WakeChain { wake: Pipe }` — covered by the standalone
//!     thread-mode test `wake_chain_pipe_thread_mode_bootstrap_throughput`
//!     which already runs on the host harness.

use anyhow::Result;
use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
use ktstr::ktstr_test;
use ktstr::scenario::Ctx;
use ktstr::workload::{
    AffinityIntent, CloneMode, SchedPolicy, WorkType, WorkloadConfig, WorkloadHandle,
};

/// Thread-mode `SpinWait` MUST run to completion in-VM. Pins the
/// basic Thread dispatch path under guest-kernel conditions:
/// every worker publishes a non-zero gettid(), produces non-zero
/// work_units, and reports completed=true. A regression in
/// `spawn_thread_worker` (e.g. start-rendezvous deadlock under a
/// low-CPU guest topology) would surface here even when the
/// host-side `spawn_thread_clone_mode_runs_to_completion` passes.
#[ktstr_test(llcs = 1, cores = 2, threads = 1, memory_mb = 1024)]
fn thread_integration_spin_wait(ctx: &Ctx) -> Result<AssertResult> {
    let config = WorkloadConfig {
        num_workers: 2,
        clone_mode: CloneMode::Thread,
        work_type: WorkType::SpinWait,
        affinity: AffinityIntent::Inherit,
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut handle = WorkloadHandle::spawn(&config)?;
    let pids = handle.worker_pids();
    if pids.len() != 2 {
        return Ok(failing_result(format!(
            "Thread SpinWait expected 2 workers, got {}; spawn broken",
            pids.len(),
        )));
    }
    for tid in &pids {
        if *tid <= 0 {
            return Ok(failing_result(format!(
                "Thread SpinWait worker reported non-positive tid={tid}; \
                 the gettid() publish path is broken under VM conditions",
            )));
        }
    }

    handle.start();
    std::thread::sleep(ctx.duration);
    let reports = handle.stop_and_collect();

    let mut result = AssertResult::pass();
    if reports.len() != 2 {
        result.passed = false;
        result.details.push(AssertDetail::new(
            DetailKind::Other,
            format!(
                "Thread SpinWait expected 2 reports, got {}; collection \
                 broken",
                reports.len(),
            ),
        ));
        return Ok(result);
    }
    for r in &reports {
        if !r.completed {
            result.passed = false;
            result.details.push(AssertDetail::new(
                DetailKind::Other,
                format!(
                    "Thread SpinWait worker tid={} did not complete; \
                     stop signaling broken under VM. exit_info={:?}",
                    r.tid, r.exit_info,
                ),
            ));
            return Ok(result);
        }
        if r.work_units == 0 {
            result.passed = false;
            result.details.push(AssertDetail::new(
                DetailKind::Other,
                format!(
                    "Thread SpinWait worker tid={} did no work; the \
                     spin loop never advanced under guest-kernel \
                     scheduling",
                    r.tid,
                ),
            ));
            return Ok(result);
        }
    }

    result.details.push(AssertDetail::new(
        DetailKind::Other,
        format!(
            "Thread SpinWait completed cleanly across {} workers; \
             total work_units={}",
            reports.len(),
            reports.iter().map(|r| r.work_units).sum::<u64>(),
        ),
    ));
    Ok(result)
}

/// Thread-mode `FutexPingPong` MUST exchange real wake/wait
/// signals through the per-pair futex word. Pins the shared
/// address-space futex semantics under the guest kernel: both
/// workers must report `resume_latencies_ns` non-empty and
/// `work_units > 0`. Because thread workers share the parent's
/// mm, the futex word is automatically shared without explicit
/// MAP_SHARED — a regression that breaks the per-pair allocation
/// would surface as zero latency samples.
#[ktstr_test(llcs = 1, cores = 2, threads = 1, memory_mb = 1024)]
fn thread_integration_futex_ping_pong(ctx: &Ctx) -> Result<AssertResult> {
    let config = WorkloadConfig {
        num_workers: 2,
        clone_mode: CloneMode::Thread,
        work_type: WorkType::FutexPingPong { spin_iters: 256 },
        affinity: AffinityIntent::Inherit,
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
                "Thread FutexPingPong expected 2 reports, got {}; \
                 spawn broken",
                reports.len(),
            ),
        ));
        return Ok(result);
    }

    for r in &reports {
        if r.resume_latencies_ns.is_empty() {
            result.passed = false;
            result.details.push(AssertDetail::new(
                DetailKind::Other,
                format!(
                    "Thread FutexPingPong worker tid={} captured zero \
                     wake-latency samples — the partner's futex wake \
                     never arrived. Under shared-mm semantics this \
                     means the futex word allocation is broken or the \
                     pair (0, 1) routing is mis-wired.",
                    r.tid,
                ),
            ));
            return Ok(result);
        }
        if r.work_units == 0 {
            result.passed = false;
            result.details.push(AssertDetail::new(
                DetailKind::Other,
                format!(
                    "Thread FutexPingPong worker tid={} did no work; \
                     the spin loop between wakes never advanced.",
                    r.tid,
                ),
            ));
            return Ok(result);
        }
    }

    let total_samples: usize = reports.iter().map(|r| r.resume_latencies_ns.len()).sum();
    result.details.push(AssertDetail::new(
        DetailKind::Other,
        format!(
            "Thread FutexPingPong populated resume_latencies_ns: \
             total_samples={total_samples} across {} workers",
            reports.len(),
        ),
    ));
    Ok(result)
}

/// Thread-mode `PageFaultChurn` MUST drive per-iteration anon
/// faults via the madvise(DONTNEED) loop under the guest kernel.
/// Pins the kernel's `do_anonymous_page` path against shared-mm
/// thread workers: every worker faults its own private region
/// (per-worker mmap, not shared), so the per-iteration zap and
/// re-fault must occur regardless of address-space sharing. A
/// regression where the madvise call returns -EINVAL under shared
/// mm would surface here as zero iterations.
#[ktstr_test(llcs = 1, cores = 2, threads = 1, memory_mb = 1024)]
fn thread_integration_page_fault_churn(ctx: &Ctx) -> Result<AssertResult> {
    let config = WorkloadConfig {
        num_workers: 2,
        clone_mode: CloneMode::Thread,
        work_type: WorkType::page_fault_churn(4096, 64, 8),
        affinity: AffinityIntent::Inherit,
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
                "Thread PageFaultChurn expected 2 reports, got {}; \
                 spawn broken",
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
            "Thread PageFaultChurn reported zero total iterations — \
             the madvise(DONTNEED) → fault loop never advanced. \
             Under shared-mm thread workers, each must still hold \
             its own per-worker mmap region; check the per-iteration \
             dispatch arm."
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
                    "Thread PageFaultChurn worker tid={} did no \
                     work; the first-touch fault never fired.",
                    r.tid,
                ),
            ));
            return Ok(result);
        }
    }

    result.details.push(AssertDetail::new(
        DetailKind::Other,
        format!(
            "Thread PageFaultChurn populated iterations: \
             total_iterations={total_iters} across {} workers",
            reports.len(),
        ),
    ));
    Ok(result)
}

/// Thread-mode `MutexContention` MUST serialize through the
/// shared mutex under the guest kernel. Pins the futex_fastpath
/// + slow-path fallback under contention with the guest kernel
/// (which may differ from the host in CONFIG_FUTEX2 / PREEMPT).
/// All four contenders must produce work_units, and total
/// iterations must be non-zero.
#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mb = 1024)]
fn thread_integration_mutex_contention(ctx: &Ctx) -> Result<AssertResult> {
    let config = WorkloadConfig {
        num_workers: 4,
        clone_mode: CloneMode::Thread,
        work_type: WorkType::MutexContention {
            contenders: 4,
            hold_iters: 256,
            work_iters: 1024,
        },
        affinity: AffinityIntent::Inherit,
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
                "Thread MutexContention expected 4 reports, got {}; \
                 spawn broken",
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
            "Thread MutexContention reported zero total iterations — \
             every worker failed to acquire the shared mutex. The \
             futex_fastpath or its contention fallback is broken \
             under VM."
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
                    "Thread MutexContention worker tid={} did no \
                     work; spawn produced an idle contender.",
                    r.tid,
                ),
            ));
            return Ok(result);
        }
    }

    result.details.push(AssertDetail::new(
        DetailKind::Other,
        format!(
            "Thread MutexContention populated iterations: \
             total_iterations={total_iters} across {} workers",
            reports.len(),
        ),
    ));
    Ok(result)
}

/// Builds an [`AssertResult`] with `passed = false` and a single
/// [`AssertDetail`] containing `msg`. Helper to reduce repetition
/// across the early-bail branches in the tests above.
fn failing_result(msg: String) -> AssertResult {
    let mut r = AssertResult::pass();
    r.passed = false;
    r.details.push(AssertDetail::new(DetailKind::Other, msg));
    r
}
