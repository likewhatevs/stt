//! End-to-end tests for the snapshot capture pipeline.
//!
//! Each `#[ktstr_test]` scenario fires a snapshot op
//! (`Op::snapshot` or `Op::watch_snapshot`) from inside a real
//! guest VM and verifies the SHM reply status via `execute_steps`.
//! The guest cannot read the captured `FailureDumpReport` because
//! the bridge that owns it lives in HOST memory, populated by the
//! freeze coordinator's doorbell handler. Content assertions
//! against the captured bridge therefore live host-side in
//! [`crate::test_support::eval::run_ktstr_test_inner_impl`] (in
//! `src/test_support/eval.rs`), which drains
//! [`VmResult::snapshot_bridge`] after `vm.run()` and walks every
//! captured report — verifying the scheduler's `.bss` map is
//! present and that `ktstr_enabled` (the always-present probe gate
//! variable) appears in the BTF render's `RenderedMember` list.

use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Op, Step, execute_steps};
use ktstr::test_support::{Payload, Scheduler, SchedulerSpec};

const KTSTR_SCHED: Scheduler =
    Scheduler::new("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));
const KTSTR_SCHED_PAYLOAD: Payload = Payload::from_scheduler(&KTSTR_SCHED);

// --------------------------------------------------------------------
// In-VM scenarios: verify the SHM/doorbell pipeline fires without
// error. The guest can't read host files, so content assertions
// happen on the host side below.
// --------------------------------------------------------------------

#[ktstr_test(
    scheduler = KTSTR_SCHED_PAYLOAD,
    duration_s = 4,
    watchdog_timeout_s = 15,
    workers_per_cgroup = 2,
    auto_repro = false,
)]
fn snapshot_real_capture_op_snapshot(ctx: &ktstr::scenario::Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![Op::snapshot("mid_run")],
        hold: HoldSpec::FULL,
    }];
    let mut result = execute_steps(ctx, steps)?;
    result.details.push(ktstr::assert::AssertDetail::new(
        ktstr::assert::DetailKind::Other,
        "Op::snapshot('mid_run') SHM request succeeded".to_string(),
    ));
    Ok(result)
}

#[ktstr_test(
    scheduler = KTSTR_SCHED_PAYLOAD,
    duration_s = 2,
    watchdog_timeout_s = 15,
    workers_per_cgroup = 2,
    auto_repro = false,
)]
fn snapshot_real_capture_op_watch_snapshot(ctx: &ktstr::scenario::Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![Op::watch_snapshot("jiffies_64")],
        hold: HoldSpec::FULL,
    }];
    let mut result = execute_steps(ctx, steps)?;
    result.details.push(ktstr::assert::AssertDetail::new(
        ktstr::assert::DetailKind::Other,
        "Op::watch_snapshot('jiffies_64') SHM request succeeded".to_string(),
    ));
    Ok(result)
}
