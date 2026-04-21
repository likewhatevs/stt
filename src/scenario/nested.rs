//! Nested cgroup hierarchy scenario implementations.

use super::backdrop::Backdrop;
use super::ops::{CgroupDef, HoldSpec, Op, Step, execute_scenario, execute_steps};
use super::{CgroupGroup, Ctx, collect_all, dfl_wl, setup_cgroups};
use crate::assert::{self, AssertResult};
use crate::workload::*;
use anyhow::Result;
use std::collections::BTreeSet;
use std::thread;
use std::time::{Duration, Instant};

/// Four nested sub-cgroups up to three levels deep with steady workload.
pub fn custom_nested_cgroup_steady(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![
        Step::with_defs(
            vec![
                CgroupDef::named("cg_0/sub_a"),
                CgroupDef::named("cg_0/sub_b"),
                CgroupDef::named("cg_1/sub_b"),
                CgroupDef::named("cg_1/sub_a/deep"),
            ],
            HoldSpec::Fixed(Duration::from_secs(2) + ctx.duration),
        )
        .set_ops(vec![
            Op::add_cgroup("cg_0"),
            Op::add_cgroup("cg_1"),
            Op::add_cgroup("cg_1/sub_a"),
        ]),
    ];

    execute_steps(ctx, steps)
}

/// Move workers through nested hierarchy: sub -> parent ->
/// cross-hierarchy sub -> parent.
///
/// The four cgroups (`cg_0/sub` with workers; `cg_0`, `cg_1`, and
/// `cg_1/sub` as empty move targets) all persist for the full
/// scenario — declaring them on the Backdrop is what lets every
/// Step's `MoveAllTasks` resolve the target cgroup without the
/// previous Step's teardown rmdir'ing it. Workers spawn inside
/// `cg_0/sub` via [`CgroupDef`]; the empty peer cgroups go through
/// [`Backdrop::with_ops`] so no implicit worker spawn happens
/// there.
pub fn custom_nested_cgroup_task_move(ctx: &Ctx) -> Result<AssertResult> {
    let backdrop = Backdrop::new()
        .with_cgroup(CgroupDef::named("cg_0/sub"))
        .with_ops(vec![
            Op::add_cgroup("cg_0"),
            Op::add_cgroup("cg_1"),
            Op::add_cgroup("cg_1/sub"),
        ]);
    let steps = vec![
        // Settle: hold once so workers run inside cg_0/sub before
        // the first MoveAllTasks. Matches the legacy 2s + duration/4
        // budget the pre-refactor single-Step version used.
        Step::new(
            vec![],
            HoldSpec::Fixed(Duration::from_secs(2) + ctx.duration / 4),
        ),
        Step::new(
            vec![Op::move_all_tasks("cg_0/sub", "cg_0")],
            HoldSpec::Frac(0.25),
        ),
        Step::new(
            vec![Op::move_all_tasks("cg_0", "cg_1/sub")],
            HoldSpec::Frac(0.25),
        ),
        Step::new(
            vec![Op::move_all_tasks("cg_1/sub", "cg_1")],
            HoldSpec::Frac(0.25),
        ),
    ];

    execute_scenario(ctx, backdrop, steps)
}

/// Rapid nested cgroup create/destroy with dynamic names. Custom logic
/// for dynamic naming.
pub fn custom_nested_cgroup_rapid_churn(ctx: &Ctx) -> Result<AssertResult> {
    let (handles, _guard) = setup_cgroups(ctx, 2, &dfl_wl(ctx))?;
    let deadline = Instant::now() + ctx.duration;
    let mut i = 0;
    while Instant::now() < deadline {
        let path = format!("cg_0/churn_{i}");
        ctx.cgroups.create_cgroup(&path)?;
        if i % 3 == 0 {
            let deep = format!("{path}/deep");
            ctx.cgroups.create_cgroup(&deep)?;
            thread::sleep(Duration::from_millis(50));
            let _ = ctx.cgroups.remove_cgroup(&deep);
        }
        thread::sleep(Duration::from_millis(50));
        let _ = ctx.cgroups.remove_cgroup(&path);
        i += 1;
    }
    Ok(collect_all(handles, &ctx.assert))
}

/// Nested cgroups with cpusets. `create_cgroup` auto-enables
/// controllers on intermediate cgroup `subtree_control` for
/// nested paths.
pub fn custom_nested_cgroup_cpuset(ctx: &Ctx) -> Result<AssertResult> {
    let all = ctx.topo.all_cpus();
    if all.len() < 4 {
        return Ok(AssertResult::skip("skipped: need >=4 CPUs"));
    }
    let mid = all.len() / 2;
    let set_a: BTreeSet<usize> = all[..mid].iter().copied().collect();

    let mut _guard = CgroupGroup::new(ctx.cgroups);
    _guard.add_cgroup("cg_0", &set_a)?;
    thread::sleep(Duration::from_secs(2));

    let sub_set: BTreeSet<usize> = all[..mid / 2].iter().copied().collect();
    _guard.add_cgroup("cg_0/narrow", &sub_set)?;

    let wl = WorkloadConfig {
        num_workers: ctx.workers_per_cgroup,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&wl)?;
    ctx.cgroups.move_tasks("cg_0/narrow", &h.tids())?;
    h.start();

    thread::sleep(ctx.duration);
    let reports = h.stop_and_collect();
    let mut r = AssertResult::pass();
    r.merge(assert::assert_not_starved(&reports));
    r.merge(assert::assert_isolation(&reports, &sub_set));
    Ok(r)
}

/// Nested sub-cgroups with heavy CpuSpin vs light Bursty load imbalance.
pub fn custom_nested_cgroup_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![
        Step::with_defs(
            vec![
                CgroupDef::named("cg_0/sub_a").workers(8),
                CgroupDef::named("cg_1/sub_b")
                    .workers(2)
                    .work_type(WorkType::bursty(50, 100)),
            ],
            HoldSpec::Fixed(ctx.settle + ctx.duration),
        )
        .set_ops(vec![Op::add_cgroup("cg_0"), Op::add_cgroup("cg_1")]),
    ];

    execute_steps(ctx, steps)
}

/// Three-level nested hierarchy with workers at leaf cgroups.
pub fn custom_nested_cgroup_no_ctrl(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![
        Step::with_defs(
            vec![
                CgroupDef::named("cg_0/sub_a/deep"),
                CgroupDef::named("cg_1/sub_b"),
            ],
            HoldSpec::Fixed(ctx.settle + ctx.duration),
        )
        .set_ops(vec![
            Op::add_cgroup("cg_0"),
            Op::add_cgroup("cg_0/sub_a"),
            Op::add_cgroup("cg_1"),
        ]),
    ];

    execute_steps(ctx, steps)
}
