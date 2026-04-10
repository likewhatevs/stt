//! CPU affinity scenario implementations.

use super::Ctx;
use super::ops::{CgroupDef, CpusetSpec, HoldSpec, Op, Step, execute_steps, execute_steps_with};
use crate::assert::{Assert, AssertResult};
use crate::workload::AffinityKind;
use anyhow::Result;
use std::collections::BTreeSet;

/// Two cgroups with worker affinities randomized four times during the
/// run. Each randomization assigns half the available CPUs to each worker.
pub fn custom_cgroup_affinity_change(ctx: &Ctx) -> Result<AssertResult> {
    let mut steps = vec![Step::with_defs(
        vec![CgroupDef::named("cg_0"), CgroupDef::named("cg_1")],
        HoldSpec::Fixed(ctx.settle + ctx.duration / 5),
    )];

    for _ in 0..4 {
        steps.push(Step::new(
            vec![
                Op::set_affinity("cg_0", AffinityKind::RandomSubset),
                Op::set_affinity("cg_1", AffinityKind::RandomSubset),
            ],
            HoldSpec::Frac(0.2),
        ));
    }

    execute_steps(ctx, steps)
}

/// Two cgroups with all workers pinned to the same 2-CPU subset.
/// Uses a relaxed 75% spread threshold since concentrated pinning
/// increases work-unit spread under EEVDF.
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
        Step::with_defs(
            vec![CgroupDef::named("cg_0"), CgroupDef::named("cg_1")],
            HoldSpec::Fixed(ctx.settle),
        ),
        Step::new(
            vec![
                Op::set_affinity("cg_0", AffinityKind::Exact(pin_cpus.clone())),
                Op::set_affinity("cg_1", AffinityKind::Exact(pin_cpus)),
            ],
            HoldSpec::Fixed(ctx.duration),
        ),
    ];

    execute_steps_with(ctx, steps, Some(&checks))
}

/// Two cgroups with disjoint cpusets, workers pinned to 2 CPUs within
/// each partition. Verifies pinning interacts correctly with cpuset
/// constraints. Uses a relaxed 75% spread threshold.
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
        Step::with_defs(
            vec![
                CgroupDef::named("cg_0").with_cpuset(CpusetSpec::disjoint(0, 2)),
                CgroupDef::named("cg_1").with_cpuset(CpusetSpec::disjoint(1, 2)),
            ],
            HoldSpec::Fixed(ctx.settle),
        ),
        Step::new(
            vec![
                Op::set_affinity("cg_0", AffinityKind::Exact(pin_a)),
                Op::set_affinity("cg_1", AffinityKind::Exact(pin_b)),
            ],
            HoldSpec::Fixed(ctx.duration),
        ),
    ];

    execute_steps_with(ctx, steps, Some(&checks))
}
