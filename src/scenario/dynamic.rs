//! Dynamic cgroup add/remove scenario implementations.

use super::ops::{CgroupDef, CpusetSpec, HoldSpec, Op, Step, execute_steps};
use super::{Ctx, collect_all, dfl_wl, setup_cgroups};
use crate::assert::AssertResult;
use crate::workload::*;
use anyhow::Result;
use std::thread;
use std::time::{Duration, Instant};

/// Add up to two cgroups mid-run alongside two steady cgroups.
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

    let steps = vec![
        Step::with_defs(
            vec![CgroupDef::named("cg_0"), CgroupDef::named("cg_1")],
            HoldSpec::Fixed(ctx.settle + ctx.duration / 2),
        ),
        Step::with_defs(phase2_setup, HoldSpec::Frac(0.5)),
    ];

    execute_steps(ctx, steps)
}

/// Remove half of up to four cgroups mid-run.
pub fn custom_cgroup_remove_midrun(ctx: &Ctx) -> Result<AssertResult> {
    let n = 4.min(ctx.topo.total_cpus().saturating_sub(1));
    if n < 2 {
        return Ok(AssertResult::skip("skipped: need >=3 CPUs"));
    }
    let half = n / 2;

    let cgroup_names: &[&str] = &["cg_0", "cg_1", "cg_2", "cg_3"];
    let phase1_setup: Vec<CgroupDef> = cgroup_names[..n]
        .iter()
        .map(|&name| CgroupDef::named(name))
        .collect();

    let mut phase2_ops = Vec::new();
    for &name in &cgroup_names[half..n] {
        phase2_ops.push(Op::stop_cgroup(name));
        phase2_ops.push(Op::remove_cgroup(name));
    }

    let steps = vec![
        Step::with_defs(phase1_setup, HoldSpec::Fixed(ctx.settle + ctx.duration / 2)),
        Step::new(phase2_ops, HoldSpec::Frac(0.5)),
    ];

    execute_steps(ctx, steps)
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

/// Add a third cpuset-partitioned cgroup mid-run, then remove it.
pub fn custom_cgroup_cpuset_add_remove(ctx: &Ctx) -> Result<AssertResult> {
    if ctx.topo.all_cpus().len() < 4 {
        return Ok(AssertResult::skip("skipped: need >=4 CPUs"));
    }

    let steps = vec![
        Step::with_defs(
            vec![
                CgroupDef::named("cg_0").with_cpuset(CpusetSpec::Disjoint { index: 0, of: 3 }),
                CgroupDef::named("cg_1").with_cpuset(CpusetSpec::Disjoint { index: 1, of: 3 }),
            ],
            HoldSpec::Fixed(ctx.settle + ctx.duration / 3),
        ),
        Step::with_defs(
            vec![CgroupDef::named("cg_2").with_cpuset(CpusetSpec::Disjoint { index: 2, of: 3 })],
            HoldSpec::Frac(1.0 / 3.0),
        ),
        Step::new(
            vec![Op::stop_cgroup("cg_2"), Op::remove_cgroup("cg_2")],
            HoldSpec::Frac(1.0 / 3.0),
        ),
    ];

    execute_steps(ctx, steps)
}

/// Add a third cgroup under load alongside heavy and bursty cgroups.
pub fn custom_cgroup_add_during_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![
        Step::with_defs(
            vec![
                CgroupDef::named("cg_0").workers(8),
                CgroupDef::named("cg_1")
                    .workers(2)
                    .work_type(WorkType::Bursty {
                        burst_ms: 50,
                        sleep_ms: 100,
                    }),
            ],
            HoldSpec::Fixed(ctx.settle + ctx.duration / 2),
        ),
        Step::with_defs(
            vec![CgroupDef::named("cg_2").workers(4)],
            HoldSpec::Frac(0.5),
        ),
    ];

    execute_steps(ctx, steps)
}
