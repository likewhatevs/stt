use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::scenario::Ctx;
use ktstr::workload::{AffinityMode, SchedPolicy, WorkType, WorkloadConfig, WorkloadHandle};

/// Boots a VM under EEVDF (no `sched_ext` scheduler attached) and
/// exits without running any workload. Guards against the
/// `trace_pipe` cleanup hang that previously made every no-scheduler
/// VM teardown wait the full host watchdog: `start_trace_pipe`
/// enables the `sched_ext_dump` tracepoint and spawns a reader thread
/// regardless of whether a scheduler is present, so teardown's
/// `handle.join()` blocks on the kernel's `tracing_wait_pipe` if the
/// reader is parked in `wait_on_pipe` with `iter->pos == 0`. With the
/// non-blocking + `poll` design in `start_trace_pipe`, the reader
/// exits within one poll cycle of the stop signal.
///
/// The 60-second host-side watchdog (`KTSTR_VM_TIMEOUT`, compile-time
/// const at `src/test_support/runtime.rs`) bounds the run; the
/// `cleanup_budget_ms = 5000` attribute below tightens that bound for
/// the host-side teardown window specifically, comparing
/// [`ktstr::vmm::VmResult::cleanup_duration`] against
/// [`KtstrTestEntry::cleanup_budget`](`ktstr::test_support::KtstrTestEntry::cleanup_budget`)
/// in `evaluate_vm_result`. A regression that re-introduces a partial
/// cleanup wait (a 30s teardown that the 60s watchdog would silently
/// absorb) flags here as a budget overshoot. The empty body keeps the
/// test cheap so it can run on every PR. The cleanup duration is
/// also persisted to the sidecar, so stats tooling can spot drift
/// across runs even when the budget passes.
#[ktstr_test(
    llcs = 1,
    cores = 1,
    threads = 1,
    memory_mb = 256,
    cleanup_budget_ms = 5000
)]
fn eevdf_empty_run_exits_under_watchdog(_ctx: &Ctx) -> Result<AssertResult> {
    Ok(AssertResult::pass())
}

/// EEVDF CPU-oversubscription gap test.
///
/// Spawns `2x total_cpus` workers each running independent
/// [`WorkType::Bursty`] cycles (1ms `spin_burst` followed by a 0ms
/// sleep — see `workload.rs` `WorkType::Bursty` arm). The workers
/// share no lock and do not coordinate; the contention is purely for
/// CPU time on an oversubscribed run queue. With sane scheduling,
/// EEVDF rotates workers fairly enough that each thread's longest gap
/// between completed bursts stays bounded by `max_gap_ms`. Aggressive
/// preemption of in-progress 1ms bursts breaks that signal: gaps
/// spike when a worker waits for a runqueue slot.
///
/// `max_gap_ms = 2000` is the empirical baseline for the configured
/// topology (4 cores × 2 SMT threads = 8 logical CPUs, 16 workers, 1ms
/// bursts, 0ms sleep). Healthy EEVDF with this load typically holds
/// `max_gap_ms` well under 1s; the 2s threshold leaves margin for boot
/// jitter, page-fault stalls during initial ramp, and per-host
/// timer-tick scheduling noise without flagging a benign hiccup as a
/// regression. Lowering it risks flakes; raising it past ~3s would
/// hide the PREEMPT_LAZY-class regressions this test guards.
///
/// `max_spread_pct = 80.0` is a RELAXATION that overrides the default
/// 15% starvation-spread threshold from `Assert::default_checks()`
/// (which enables `not_starved=true`, running `assert_not_starved`
/// with `spread_threshold_pct() = 15%` in release builds — see
/// `spread_threshold_pct()` and the spread-vs-limit comparison in
/// `assert_not_starved` in `src/assert.rs`). With 16 workers
/// oversubscribing 8 CPUs and 1ms bursts, EEVDF spread at sub-slice
/// granularity routinely exceeds 15% on healthy runs; 80% is wide
/// enough to absorb that variance while still catching a fully
/// starved worker, leaving `max_gap_ms` as the primary regression
/// signal.
///
/// Models the regression surface from the PREEMPT_LAZY thread without
/// reproducing its lock-holder-preemption mechanic; this test stresses
/// the runqueue-fairness side, not lock contention.
///
/// The body asserts via [`Ctx::assert`], which the in-VM dispatch
/// path populates as `Assert::default_checks() +
/// scheduler.assert + entry.assert` (the macro attributes flow into
/// `entry.assert`). The `#[ktstr_test]` attributes above are therefore
/// the single source of truth for the thresholds — the body does not
/// rebuild them.
#[ktstr_test(
    llcs = 1,
    cores = 4,
    threads = 2,
    memory_mb = 2048,
    max_gap_ms = 2000,
    max_spread_pct = 80.0
)]
fn eevdf_burst_oversubscription(ctx: &Ctx) -> Result<AssertResult> {
    let total_cpus = ctx.topo.total_cpus();
    let num_workers = total_cpus * 2;

    let config = WorkloadConfig {
        num_workers,
        affinity: AffinityMode::None,
        work_type: WorkType::Bursty {
            burst_ms: 1,
            sleep_ms: 0,
        },
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };

    let mut handle = WorkloadHandle::spawn(&config)?;
    handle.start();
    std::thread::sleep(ctx.duration);
    let reports = handle.stop_and_collect();

    Ok(ctx.assert.assert_cgroup(&reports, None))
}
