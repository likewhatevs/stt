//! Dynamic cgroup add/remove scenario implementations.

use super::backdrop::Backdrop;
use super::ops::{CgroupDef, CpusetSpec, HoldSpec, Step, execute_scenario};
use super::{Ctx, collect_all, dfl_wl, setup_cgroups};
use crate::assert::AssertResult;
use crate::workload::*;
use anyhow::Result;
use std::thread;
use std::time::{Duration, Instant};

/// Add up to two cgroups mid-run alongside two steady cgroups.
///
/// `cg_0` and `cg_1` run for the whole scenario; `cg_2` / `cg_3`
/// appear mid-run as a step-local CgroupDef set and tear down at
/// the step boundary. Steady cgroups go on the Backdrop;
/// mid-run additions stay step-local.
pub fn custom_cgroup_add_midrun(ctx: &Ctx) -> Result<AssertResult> {
    let max_new = ctx.topo.total_cpus().saturating_sub(3).min(2);
    if max_new == 0 {
        return Ok(AssertResult::skip("skipped: need >=4 CPUs"));
    }

    let extra_names: &[&str] = &["cg_2", "cg_3"];
    let phase2_setup: Vec<CgroupDef> = extra_names[..max_new]
        .iter()
        .map(|&name| CgroupDef::named(name))
        .collect();

    let backdrop = Backdrop::new()
        .with_cgroup(CgroupDef::named("cg_0"))
        .with_cgroup(CgroupDef::named("cg_1"));
    let steps = vec![
        // Phase 1: settle with just the two steady cgroups.
        Step::new(vec![], HoldSpec::Fixed(ctx.settle + ctx.duration / 2)),
        // Phase 2: add the step-local extras.
        Step::with_defs(phase2_setup, HoldSpec::Frac(0.5)),
    ];

    execute_scenario(ctx, backdrop, steps)
}

/// Remove half of up to four cgroups mid-run.
///
/// The kept half (`cg_0`, `cg_1`) lives on the Backdrop and
/// persists across both Steps. The ephemeral half (`cg_2` / `cg_3`,
/// up to the topology limit) is a step-local Step-0 CgroupDef set
/// whose automatic per-Step teardown IS the "remove mid-run"
/// event — no explicit `Op::stop_cgroup` / `Op::remove_cgroup`
/// ops required. Step 1 is then a pure hold with only the
/// Backdrop cgroups still present.
pub fn custom_cgroup_remove_midrun(ctx: &Ctx) -> Result<AssertResult> {
    let n = 4.min(ctx.topo.total_cpus().saturating_sub(1));
    if n < 2 {
        return Ok(AssertResult::skip("skipped: need >=3 CPUs"));
    }
    let half = n / 2;

    let cgroup_names: &[&str] = &["cg_0", "cg_1", "cg_2", "cg_3"];

    let mut backdrop = Backdrop::new();
    for &name in &cgroup_names[..half] {
        backdrop = backdrop.with_cgroup(CgroupDef::named(name));
    }

    let step0_defs: Vec<CgroupDef> = cgroup_names[half..n]
        .iter()
        .map(|&name| CgroupDef::named(name))
        .collect();

    let steps = vec![
        Step::with_defs(step0_defs, HoldSpec::Fixed(ctx.settle + ctx.duration / 2)),
        Step::new(vec![], HoldSpec::Frac(0.5)),
    ];

    execute_scenario(ctx, backdrop, steps)
}

/// Rapid create/destroy cycling. Custom logic for dynamic naming.
pub fn custom_cgroup_rapid_churn(ctx: &Ctx) -> Result<AssertResult> {
    let (handles, _guard) = setup_cgroups(ctx, 2, &dfl_wl(ctx))?;
    let deadline = Instant::now() + ctx.duration;
    let mut i = 0;
    while Instant::now() < deadline {
        let n = format!("ephemeral_{i}");
        ctx.cgroups.create_cgroup(&n)?;
        thread::sleep(Duration::from_millis(100));
        let _ = ctx.cgroups.remove_cgroup(&n);
        i += 1;
    }
    Ok(collect_all(handles, &ctx.assert))
}

/// Add a third cpuset-partitioned cgroup mid-run; tear it down
/// via automatic step boundary.
///
/// `cg_0` / `cg_1` hold their cpusets for the full scenario on the
/// Backdrop. `cg_2` lives only in the middle Step — the automatic
/// step-boundary teardown removes it before the final hold runs,
/// replacing the pre-refactor explicit stop + remove ops.
pub fn custom_cgroup_cpuset_add_remove(ctx: &Ctx) -> Result<AssertResult> {
    if ctx.topo.all_cpus().len() < 4 {
        return Ok(AssertResult::skip("skipped: need >=4 CPUs"));
    }

    let backdrop = Backdrop::new().with_cgroups([
        CgroupDef::named("cg_0").with_cpuset(CpusetSpec::disjoint(0, 3)),
        CgroupDef::named("cg_1").with_cpuset(CpusetSpec::disjoint(1, 3)),
    ]);
    let steps = vec![
        // Phase 1: settle the two steady cgroups.
        Step::new(vec![], HoldSpec::Fixed(ctx.settle + ctx.duration / 3)),
        // Phase 2: add cg_2; auto teardown at step end removes it.
        Step::with_defs(
            vec![CgroupDef::named("cg_2").with_cpuset(CpusetSpec::disjoint(2, 3))],
            HoldSpec::Frac(1.0 / 3.0),
        ),
        // Phase 3: only cg_0 / cg_1 continue — cg_2 is gone.
        Step::new(vec![], HoldSpec::Frac(1.0 / 3.0)),
    ];

    execute_scenario(ctx, backdrop, steps)
}

/// Add a third cgroup under load alongside heavy and bursty cgroups.
///
/// The heavy `cg_0` and bursty `cg_1` run for the full scenario on
/// the Backdrop. The mid-run `cg_2` appears in the second Step and
/// tears down at the step boundary.
pub fn custom_cgroup_add_during_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    let backdrop = Backdrop::new().with_cgroups([
        CgroupDef::named("cg_0").workers(8),
        CgroupDef::named("cg_1")
            .workers(2)
            .work_type(WorkType::bursty(50, 100)),
    ]);
    let steps = vec![
        // Phase 1: settle with cg_0 and cg_1 alone.
        Step::new(vec![], HoldSpec::Fixed(ctx.settle + ctx.duration / 2)),
        // Phase 2: add cg_2 as step-local.
        Step::with_defs(
            vec![CgroupDef::named("cg_2").workers(4)],
            HoldSpec::Frac(0.5),
        ),
    ];

    execute_scenario(ctx, backdrop, steps)
}
