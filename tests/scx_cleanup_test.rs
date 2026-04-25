use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::scenario::Ctx;
use ktstr::test_support::{Payload, Scheduler, SchedulerSpec};

const SCX_CLEANUP_SCHED: Scheduler =
    Scheduler::new("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));
const SCX_CLEANUP_SCHED_PAYLOAD: Payload = Payload::from_scheduler(&SCX_CLEANUP_SCHED);

/// Boots a VM with the scx-ktstr scheduler attached, runs no
/// workload, and exits cleanly. Counterpart to
/// `eevdf_empty_run_exits_under_watchdog` in `tests/eevdf_tests.rs`.
///
/// Where the EEVDF version exercises the no-scheduler-attached
/// trace_pipe drain path (`iter->pos == 0` in
/// `start_trace_pipe`'s reader), this version exercises the
/// scheduler-attached drain path: scx-ktstr's `sched_ext_dump`
/// tracepoint emits events as the scheduler attaches and the BSP
/// boots, so by the time the host-side teardown runs `iter->pos
/// > 0` and the reader takes the second branch in the cleanup
/// loop. A regression that breaks either branch lands either as
/// a `cleanup_budget_ms = 5000` overshoot (caught by
/// `evaluate_vm_result` against
/// [`ktstr::vmm::VmResult::cleanup_duration`]) or, in the
/// catastrophic case, as a host watchdog timeout (60 s,
/// `KTSTR_VM_TIMEOUT` in `src/test_support/runtime.rs`). The
/// cleanup duration is also persisted to the sidecar so stats
/// tooling can flag drift across runs.
///
/// Body returns `Ok(AssertResult::pass())` because the assertion
/// of interest — that VM teardown completes within the cleanup
/// budget — is enforced host-side after the body returns; if the
/// budget is exceeded the framework folds a failing detail into
/// the verdict.
#[ktstr_test(
    scheduler = SCX_CLEANUP_SCHED_PAYLOAD,
    llcs = 1,
    cores = 1,
    threads = 1,
    memory_mb = 256,
    cleanup_budget_ms = 5000,
)]
fn scx_empty_run_exits_under_watchdog(_ctx: &Ctx) -> Result<AssertResult> {
    Ok(AssertResult::pass())
}
