//! End-to-end coverage for [`CgroupDef::comm`] and [`CgroupDef::nice`]
//! propagation through `apply_setup` into the worker-spawn pipeline.
//!
//! `CgroupDef::comm` and `CgroupDef::nice` set cgroup-level defaults
//! that merge into every [`WorkSpec`] whose own `comm` / `nice` is
//! unset. The spawned worker then issues `prctl(PR_SET_NAME)` and
//! `setpriority(PRIO_PROCESS, 0, n)` from inside `worker_main` (see
//! `src/workload/worker/mod.rs`), so a propagation regression
//! surfaces as one of three observable host-side failure shapes:
//!
//! 1. spawn aborts in-guest, the test body bails before the hold
//!    completes, and `vm.run()` returns `success == false`.
//! 2. the guest hits the watchdog (`timed_out == true`) because the
//!    in-VM scenario never reached `MSG_TYPE_EXIT`.
//! 3. the guest panics on the propagation path, surfacing a
//!    `PANIC:` line on COM2 that lands in `crash_message`.
//!
//! The `post_vm` callback runs on the host after `vm.run()` returns
//! and gates on those three host-visible signals — the strongest
//! proof available to the host that the worker dispatch actually
//! applied the cgroup-level defaults.

use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::VmResult;
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps};
use ktstr::test_support::{Scheduler, SchedulerSpec};

const KTSTR_SCHED: Scheduler =
    Scheduler::new("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Host-side gate: a clean run means the in-VM spawn pipeline
/// processed the cgroup-level `comm`/`nice` defaults without
/// crashing. Any propagation regression collapses one of these
/// signals.
fn assert_workers_ran_clean(result: &VmResult) -> Result<()> {
    anyhow::ensure!(
        !result.timed_out,
        "guest timed out under the watchdog — the worker spawn or \
         hold never completed"
    );
    anyhow::ensure!(
        result.crash_message.is_none(),
        "guest panicked during the run — `crash_message` = {:?}; \
         a panic in the worker spawn path would surface here",
        result.crash_message,
    );
    anyhow::ensure!(
        result.exit_code == 0,
        "guest exit_code = {} (expected 0) — non-zero typically \
         means the in-guest scenario bailed before completing the \
         hold (e.g. `apply_setup` rejected the merged WorkSpec)",
        result.exit_code,
    );
    anyhow::ensure!(
        result.success,
        "VM run reported success = false (timed_out = {}, exit_code = \
         {}, crash_message = {:?}); the merged comm/nice defaults \
         did not produce a clean worker dispatch",
        result.timed_out,
        result.exit_code,
        result.crash_message,
    );
    Ok(())
}

/// Boots a real guest, declares one cgroup with `comm("test_comm")`
/// and `nice(5)` set as cgroup-level defaults, and holds for the
/// full step duration. With no per-WorkSpec `comm`/`nice` overrides,
/// the cgroup defaults must propagate through `apply_setup` into the
/// implicit default `WorkSpec` so every worker calls
/// `prctl(PR_SET_NAME, "test_comm")` and `setpriority(PRIO_PROCESS,
/// 0, 5)` at startup. Completion alone — gated by `assert_workers_ran_clean`
/// — is the assertion: any propagation regression collapses one of
/// the host-visible signals checked there.
#[ktstr_test(
    scheduler = KTSTR_SCHED,
    duration_s = 3,
    watchdog_timeout_s = 15,
    workers_per_cgroup = 2,
    auto_repro = false,
    post_vm = assert_workers_ran_clean,
)]
fn worker_properties_e2e(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cg_props")
                .comm("test_comm")
                .nice(5)
                .workers(2),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}
