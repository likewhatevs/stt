use super::ops::{CgroupDef, CpusetSpec, HoldSpec, Op, Step, execute_steps};
use super::{Ctx, collect_all, dfl_wl, setup_cgroups};
use crate::verify::VerifyResult;
use crate::workload::*;
use anyhow::Result;
use std::thread;
use std::time::{Duration, Instant};

pub fn custom_cgroup_add_midrun(ctx: &Ctx) -> Result<VerifyResult> {
    let max_new = ctx.topo.total_cpus().saturating_sub(3).min(2);
    if max_new == 0 {
        return Ok(VerifyResult {
            passed: true,
            details: vec!["skipped: need >=4 CPUs".into()],
            stats: Default::default(),
        });
    }

    let extra_names: &[&str] = &["cg_2", "cg_3"];
    let phase2_setup: Vec<CgroupDef> = extra_names[..max_new]
        .iter()
        .map(|&name| CgroupDef::named(name))
        .collect();

    let steps = vec![
        Step {
            setup: vec![CgroupDef::named("cg_0"), CgroupDef::named("cg_1")].into(),
            ops: vec![],
            hold: HoldSpec::Fixed(Duration::from_millis(ctx.settle_ms) + ctx.duration / 2),
        },
        Step {
            setup: phase2_setup.into(),
            ops: vec![],
            hold: HoldSpec::Frac(0.5),
        },
    ];

    execute_steps(ctx, steps)
}

pub fn custom_cgroup_remove_midrun(ctx: &Ctx) -> Result<VerifyResult> {
    let n = 4.min(ctx.topo.total_cpus().saturating_sub(1));
    if n < 2 {
        return Ok(VerifyResult {
            passed: true,
            details: vec!["skipped: need >=3 CPUs".into()],
            stats: Default::default(),
        });
    }
    let half = n / 2;

    let cgroup_names: &[&str] = &["cg_0", "cg_1", "cg_2", "cg_3"];
    let phase1_setup: Vec<CgroupDef> = cgroup_names[..n]
        .iter()
        .map(|&name| CgroupDef::named(name))
        .collect();

    let mut phase2_ops = Vec::new();
    for &name in &cgroup_names[half..n] {
        phase2_ops.push(Op::StopCgroup {
            cgroup: name.into(),
        });
        phase2_ops.push(Op::RemoveCgroup { name: name.into() });
    }

    let steps = vec![
        Step {
            setup: phase1_setup.into(),
            ops: vec![],
            hold: HoldSpec::Fixed(Duration::from_millis(ctx.settle_ms) + ctx.duration / 2),
        },
        Step {
            setup: vec![].into(),
            ops: phase2_ops,
            hold: HoldSpec::Frac(0.5),
        },
    ];

    execute_steps(ctx, steps)
}

/// Rapid create/destroy cycling. Custom logic for dynamic naming.
pub fn custom_cgroup_rapid_churn(ctx: &Ctx) -> Result<VerifyResult> {
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
    Ok(collect_all(handles))
}

pub fn custom_cgroup_cpuset_add_remove(ctx: &Ctx) -> Result<VerifyResult> {
    if ctx.topo.all_cpus().len() < 4 {
        return Ok(VerifyResult {
            passed: true,
            details: vec!["skipped: need >=4 CPUs".into()],
            stats: Default::default(),
        });
    }

    let steps = vec![
        Step {
            setup: vec![
                CgroupDef::named("cg_0").with_cpuset(CpusetSpec::Disjoint { index: 0, of: 3 }),
                CgroupDef::named("cg_1").with_cpuset(CpusetSpec::Disjoint { index: 1, of: 3 }),
            ]
            .into(),
            ops: vec![],
            hold: HoldSpec::Fixed(Duration::from_secs(3) + ctx.duration / 3),
        },
        Step {
            setup: vec![
                CgroupDef::named("cg_2").with_cpuset(CpusetSpec::Disjoint { index: 2, of: 3 }),
            ]
            .into(),
            ops: vec![],
            hold: HoldSpec::Frac(1.0 / 3.0),
        },
        Step {
            setup: vec![].into(),
            ops: vec![
                Op::StopCgroup {
                    cgroup: "cg_2".into(),
                },
                Op::RemoveCgroup {
                    name: "cg_2".into(),
                },
            ],
            hold: HoldSpec::Frac(1.0 / 3.0),
        },
    ];

    execute_steps(ctx, steps)
}

pub fn custom_cgroup_add_during_imbalance(ctx: &Ctx) -> Result<VerifyResult> {
    let steps = vec![
        Step {
            setup: vec![
                CgroupDef::named("cg_0").workers(8),
                CgroupDef::named("cg_1")
                    .workers(2)
                    .work_type(WorkType::Bursty {
                        burst_ms: 50,
                        sleep_ms: 100,
                    }),
            ]
            .into(),
            ops: vec![],
            hold: HoldSpec::Fixed(Duration::from_secs(3) + ctx.duration / 2),
        },
        Step {
            setup: vec![CgroupDef::named("cg_2").workers(4)].into(),
            ops: vec![],
            hold: HoldSpec::Frac(0.5),
        },
    ];

    execute_steps(ctx, steps)
}
