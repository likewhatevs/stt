//! End-to-end coverage for the pcomm fork-then-thread spawn path
//! via [`CgroupDef::pcomm`].
//!
//! `CgroupDef::pcomm("chrome")` writes `pcomm = Some("chrome")` into
//! every [`WorkSpec`] in the cgroup's `works` slice. `apply_setup`
//! routes every WorkSpec sharing that pcomm value through
//! [`crate::workload::WorkloadHandle::spawn_pcomm_cgroup`]: ONE
//! forked thread-group leader whose `task->comm` is `"chrome"`,
//! hosting every requested worker as a thread under that leader.
//! Each worker thread additionally sets its own `task->comm` from
//! `WorkSpec::comm` — modelling a real workload like `chrome`
//! (pcomm) hosting `ThreadPool` worker threads.
//!
//! The `post_vm` callback gates on the same host-visible signals
//! used in [`worker_properties_e2e`](super::worker_properties_e2e):
//! a regression in the partitioner, container fork, leader's
//! `prctl(PR_SET_NAME)`, per-thread spawn, or bincode report stream
//! collapses one of `timed_out` / `crash_message` / `exit_code` /
//! `success`.

use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::VmResult;
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps};
use ktstr::test_support::{Payload, Scheduler, SchedulerSpec};
use ktstr::workload::config::WorkSpec;

const KTSTR_SCHED: Scheduler =
    Scheduler::new("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));
const KTSTR_SCHED_PAYLOAD: Payload = Payload::from_scheduler(&KTSTR_SCHED);

/// Host-side gate: a clean run means the in-VM dispatch routed every
/// pcomm-tagged WorkSpec through `spawn_pcomm_cgroup`, the container
/// leader forked successfully, threads spawned, the work loop ran to
/// the hold deadline, and the per-worker reports streamed back
/// through the bincode pipe to the parent without faulting. Any
/// regression in the pcomm path collapses one of these signals.
fn assert_pcomm_ran_clean(result: &VmResult) -> Result<()> {
    anyhow::ensure!(
        !result.timed_out,
        "guest timed out under the watchdog — the pcomm container \
         spawn or hold never completed; the fork-then-thread dispatch \
         in `apply_setup` likely faulted before reaching the hold"
    );
    anyhow::ensure!(
        result.crash_message.is_none(),
        "guest panicked during the run — `crash_message` = {:?}; a \
         regression in `spawn_pcomm_cgroup` (leader fork, \
         `prctl(PR_SET_NAME)`, per-thread spawn, or bincode report \
         stream) would surface here",
        result.crash_message,
    );
    anyhow::ensure!(
        result.exit_code == 0,
        "guest exit_code = {} (expected 0) — non-zero typically means \
         the in-guest scenario bailed before completing the hold (e.g. \
         the pcomm partitioner rejected the WorkSpec slice or the \
         container leader's report-collection path returned an error)",
        result.exit_code,
    );
    anyhow::ensure!(
        result.success,
        "VM run reported success = false (timed_out = {}, exit_code = \
         {}, crash_message = {:?}); the pcomm fork-then-thread \
         dispatch did not produce a clean run",
        result.timed_out,
        result.exit_code,
        result.crash_message,
    );
    Ok(())
}

/// Boots a real guest, declares one cgroup with `.pcomm("chrome")`
/// and a [`WorkSpec`] of two workers whose per-thread `comm` is
/// `"ThreadPool"`. The cgroup-level `.pcomm("chrome")` builder
/// writes `pcomm = Some("chrome")` into every WorkSpec in `works`,
/// which causes `apply_setup` to coalesce them into a single
/// `spawn_pcomm_cgroup` call: ONE forked leader whose `task->comm`
/// is `"chrome"`, hosting two worker threads whose own `task->comm`
/// is `"ThreadPool"`. Completion alone — gated by
/// `assert_pcomm_ran_clean` — is the assertion: any regression in
/// the fork-then-thread spawn pipeline collapses one of the
/// host-visible signals checked there.
#[ktstr_test(
    scheduler = KTSTR_SCHED_PAYLOAD,
    duration_s = 3,
    watchdog_timeout_s = 15,
    workers_per_cgroup = 2,
    auto_repro = false,
    post_vm = assert_pcomm_ran_clean,
)]
fn pcomm_fork_then_thread_e2e(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cg_chrome")
                .pcomm("chrome")
                .work(WorkSpec::default().workers(2).comm("ThreadPool")),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}
