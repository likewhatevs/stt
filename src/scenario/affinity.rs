//! CPU affinity scenario implementations.

use super::Ctx;
use super::ops::{CgroupDef, CpusetSpec, HoldSpec, Op, Step, execute_steps};
use crate::assert::AssertResult;
use anyhow::Result;
use std::collections::BTreeSet;
use std::time::Duration;

pub fn custom_cgroup_affinity_change(ctx: &Ctx) -> Result<AssertResult> {
    let mut steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0"), CgroupDef::named("cg_1")].into(),
        ops: vec![],
        hold: HoldSpec::Fixed(Duration::from_millis(ctx.settle_ms) + ctx.duration / 5),
    }];

    for _ in 0..4 {
        steps.push(Step {
            setup: vec![].into(),
            ops: vec![
                Op::RandomizeAffinity {
                    cgroup: "cg_0".into(),
                },
                Op::RandomizeAffinity {
                    cgroup: "cg_1".into(),
                },
            ],
            hold: HoldSpec::Frac(0.2),
        });
    }

    execute_steps(ctx, steps)
}

pub fn custom_cgroup_multicpu_pin(ctx: &Ctx) -> Result<AssertResult> {
    let all = ctx.topo.all_cpus();
    let pin_cpus: BTreeSet<usize> = if all.len() >= 2 {
        all[..2].iter().copied().collect()
    } else {
        all.iter().copied().collect()
    };

    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0"), CgroupDef::named("cg_1")].into(),
        ops: vec![
            Op::SetAffinity {
                cgroup: "cg_0".into(),
                cpus: pin_cpus.clone(),
            },
            Op::SetAffinity {
                cgroup: "cg_1".into(),
                cpus: pin_cpus,
            },
        ],
        hold: HoldSpec::Fixed(Duration::from_millis(ctx.settle_ms) + ctx.duration),
    }];

    execute_steps(ctx, steps)
}

pub fn custom_cgroup_cpuset_multicpu_pin(ctx: &Ctx) -> Result<AssertResult> {
    let usable = ctx.topo.usable_cpus();
    let mid = usable.len() / 2;
    let a: BTreeSet<usize> = usable[..mid].iter().copied().collect();
    let b: BTreeSet<usize> = usable[mid..].iter().copied().collect();

    let pin_a: BTreeSet<usize> = a.iter().copied().take(2.min(a.len())).collect();
    let pin_b: BTreeSet<usize> = b.iter().copied().take(2.min(b.len())).collect();

    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cg_0").with_cpuset(CpusetSpec::Disjoint { index: 0, of: 2 }),
            CgroupDef::named("cg_1").with_cpuset(CpusetSpec::Disjoint { index: 1, of: 2 }),
        ]
        .into(),
        ops: vec![
            Op::SetAffinity {
                cgroup: "cg_0".into(),
                cpus: pin_a,
            },
            Op::SetAffinity {
                cgroup: "cg_1".into(),
                cpus: pin_b,
            },
        ],
        hold: HoldSpec::Fixed(Duration::from_millis(ctx.settle_ms) + ctx.duration),
    }];

    execute_steps(ctx, steps)
}
