//! End-to-end coverage for the `no_perf_mode = true` test property.
//!
//! `no_perf_mode = true` decouples the requested virtual topology from
//! the host hardware. With it set, the dispatch path in
//! `KtstrTestEntry::accepts_no_perf_mode` (see
//! `src/test_support/entry.rs`) relaxes the per-LLC and per-NUMA
//! shape checks while still requiring the host to provide enough
//! total CPUs for the requested topology — so the test runs on
//! basic hardware that would otherwise hardware-skip.
//!
//! A regression that re-tightens the constraints surfaces as a
//! hardware-skip in dispatch and the in-VM scenario never executes,
//! so `vm.run()` returns without `success`. A regression that
//! breaks the `no_perf_mode` plumbing through the macro (see the
//! `no_perf_mode` arm in `ktstr-macros/src/lib.rs` and
//! `KtstrTestEntry::no_perf_mode` in `entry.rs`) collapses one of
//! the same host-visible signals checked below.
//!
//! The `post_vm` callback runs on the host after `vm.run()` returns
//! and gates on `timed_out`, `crash_message`, `exit_code`, and
//! `success` — the strongest proof available to the host that the
//! relaxed-topology dispatch admitted this test and the in-VM run
//! completed cleanly.
//!
//! The default cgroup with two workers exercises the standard
//! worker-spawn path under the relaxed-topology dispatch; the value
//! of `no_perf_mode` is that the test RUNS on hardware that would
//! otherwise skip, so a clean completion is the assertion.

use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::VmResult;
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps};
use ktstr::test_support::{Payload, Scheduler, SchedulerSpec};

const KTSTR_SCHED: Scheduler =
    Scheduler::new("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));
const KTSTR_SCHED_PAYLOAD: Payload = Payload::from_scheduler(&KTSTR_SCHED);

/// Host-side gate: a clean run means the relaxed-topology dispatch
/// admitted the test on the available hardware and the in-VM
/// scenario completed without faulting. Any regression in
/// `no_perf_mode` plumbing or in
/// `KtstrTestEntry::accepts_no_perf_mode` collapses one of these
/// signals.
fn assert_no_perf_mode_ran_clean(result: &VmResult) -> Result<()> {
    anyhow::ensure!(
        !result.timed_out,
        "guest timed out under the watchdog — the in-VM scenario \
         never completed; a regression in no_perf_mode dispatch or \
         the relaxed topology checks would surface here"
    );
    anyhow::ensure!(
        result.crash_message.is_none(),
        "guest panicked during the run — `crash_message` = {:?}; \
         a regression in the no_perf_mode plumbing through the \
         macro or entry struct would surface here",
        result.crash_message,
    );
    anyhow::ensure!(
        result.exit_code == 0,
        "guest exit_code = {} (expected 0) — non-zero typically \
         means the in-guest scenario bailed before completing the \
         hold under the relaxed-topology dispatch",
        result.exit_code,
    );
    anyhow::ensure!(
        result.success,
        "VM run reported success = false (timed_out = {}, exit_code = \
         {}, crash_message = {:?}); the no_perf_mode relaxed-topology \
         dispatch did not produce a clean run",
        result.timed_out,
        result.exit_code,
        result.crash_message,
    );
    Ok(())
}

/// Boots a real guest with `no_perf_mode = true`, declaring a wild
/// virtual topology (8 LLCs × 4 cores × 2 threads = 64 CPUs) that
/// would normally only dispatch on hardware whose per-LLC and
/// per-NUMA shape matches. With `no_perf_mode = true`,
/// `KtstrTestEntry::accepts_no_perf_mode` relaxes those shape checks
/// while still requiring enough total CPUs, so the test runs on
/// basic hardware. Completion alone — gated by
/// `assert_no_perf_mode_ran_clean` — is the assertion.
#[ktstr_test(
    scheduler = KTSTR_SCHED_PAYLOAD,
    no_perf_mode = true,
    llcs = 8,
    cores = 4,
    threads = 2,
    duration_s = 3,
    watchdog_timeout_s = 15,
    workers_per_cgroup = 2,
    auto_repro = false,
    post_vm = assert_no_perf_mode_ran_clean,
)]
fn no_perf_mode_e2e(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_no_perf").workers(2)].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}
