//! End-to-end tests for the snapshot capture pipeline.
//!
//! Each `#[ktstr_test]` scenario fires a snapshot op from inside a
//! real guest VM. The guest verifies the SHM round-trip succeeded
//! (Op returned Ok). The `post_vm` callback runs on the HOST after
//! `vm.run()` returns and asserts the captured `FailureDumpReport`
//! on the `SnapshotBridge` contains real BTF-rendered BPF state.

use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::{RenderedValue, VmResult};
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Op, Step, execute_steps};
use ktstr::test_support::{Payload, Scheduler, SchedulerSpec};

const KTSTR_SCHED: Scheduler =
    Scheduler::new("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));
const KTSTR_SCHED_PAYLOAD: Payload = Payload::from_scheduler(&KTSTR_SCHED);

/// Host-side content assertion: verify the bridge has a capture
/// with the scheduler's .bss containing real BTF-rendered globals.
fn assert_bridge_has_real_capture(result: &VmResult) -> Result<()> {
    let captured = result.snapshot_bridge.drain();
    anyhow::ensure!(
        !captured.is_empty(),
        "snapshot bridge is empty — no captures reached the host"
    );
    for (tag, report) in &captured {
        anyhow::ensure!(
            !report.maps.is_empty(),
            "snapshot '{tag}' has 0 maps — capture produced nothing"
        );
        let bss = report.maps.iter().find(|m| m.name.ends_with(".bss"));
        anyhow::ensure!(
            bss.is_some(),
            "snapshot '{tag}' has {} maps but no .bss map. maps: {:?}",
            report.maps.len(),
            report
                .maps
                .iter()
                .map(|m| m.name.as_str())
                .collect::<Vec<_>>()
        );
        let bss = bss.unwrap();
        let has_real_members = bss
            .value
            .as_ref()
            .and_then(|v| match v {
                RenderedValue::Struct { members, .. } => Some(members.len() >= 3),
                _ => None,
            })
            .unwrap_or(false);
        anyhow::ensure!(
            has_real_members,
            "snapshot '{tag}' .bss '{}' has no BTF-rendered members — \
             capture did not produce real scheduler state",
            bss.name
        );
    }
    Ok(())
}

#[ktstr_test(
    scheduler = KTSTR_SCHED_PAYLOAD,
    duration_s = 10,
    watchdog_timeout_s = 15,
    workers_per_cgroup = 2,
    auto_repro = false,
    post_vm = assert_bridge_has_real_capture,
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
    duration_s = 10,
    watchdog_timeout_s = 30,
    workers_per_cgroup = 2,
    auto_repro = false,
    post_vm = assert_bridge_has_real_capture,
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
