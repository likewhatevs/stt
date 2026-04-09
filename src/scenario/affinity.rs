//! CPU affinity scenario implementations.

use super::Ctx;
use super::ops::{CgroupDef, CpusetSpec, HoldSpec, Op, Step, execute_steps, execute_steps_with};
use crate::assert::{Assert, AssertResult};
use anyhow::Result;
use std::collections::BTreeSet;

pub fn custom_cgroup_affinity_change(ctx: &Ctx) -> Result<AssertResult> {
    let mut steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0"), CgroupDef::named("cg_1")].into(),
        ops: vec![],
        hold: HoldSpec::Fixed(ctx.settle + ctx.duration / 5),
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

    // Pinning all workers to 2 CPUs concentrates load and increases
    // spread under EEVDF; relax the default 35% threshold.
    let checks = Assert::default_checks().max_spread_pct(75.0);

    let steps = vec![
        Step {
            setup: vec![CgroupDef::named("cg_0"), CgroupDef::named("cg_1")].into(),
            ops: vec![],
            hold: HoldSpec::Fixed(ctx.settle),
        },
        Step {
            setup: vec![].into(),
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
            hold: HoldSpec::Fixed(ctx.duration),
        },
    ];

    execute_steps_with(ctx, steps, Some(&checks))
}

pub fn custom_cgroup_cpuset_multicpu_pin(ctx: &Ctx) -> Result<AssertResult> {
    let usable = ctx.topo.usable_cpus();
    let mid = usable.len() / 2;
    let a: BTreeSet<usize> = usable[..mid].iter().copied().collect();
    let b: BTreeSet<usize> = usable[mid..].iter().copied().collect();

    let pin_a: BTreeSet<usize> = a.iter().copied().take(2.min(a.len())).collect();
    let pin_b: BTreeSet<usize> = b.iter().copied().take(2.min(b.len())).collect();

    // Pinning workers to 2 CPUs within each cpuset partition
    // concentrates load and increases spread; relax the threshold.
    let checks = Assert::default_checks().max_spread_pct(75.0);

    let steps = vec![
        Step {
            setup: vec![
                CgroupDef::named("cg_0").with_cpuset(CpusetSpec::Disjoint { index: 0, of: 2 }),
                CgroupDef::named("cg_1").with_cpuset(CpusetSpec::Disjoint { index: 1, of: 2 }),
            ]
            .into(),
            ops: vec![],
            hold: HoldSpec::Fixed(ctx.settle),
        },
        Step {
            setup: vec![].into(),
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
            hold: HoldSpec::Fixed(ctx.duration),
        },
    ];

    execute_steps_with(ctx, steps, Some(&checks))
}
