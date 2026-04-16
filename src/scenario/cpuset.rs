//! Cpuset mutation scenario implementations.

use super::Ctx;
use super::ops::{CgroupDef, CpusetSpec, HoldSpec, Op, Step, execute_steps};
use crate::assert::AssertResult;
use crate::workload::*;
use anyhow::Result;

/// Apply disjoint cpusets to two initially unconstrained cgroups mid-run.
pub fn custom_cgroup_cpuset_apply_midrun(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![
        Step::with_defs(
            vec![CgroupDef::named("cg_0"), CgroupDef::named("cg_1")],
            HoldSpec::Fixed(ctx.settle + ctx.duration / 2),
        ),
        Step::new(
            vec![
                Op::set_cpuset("cg_0", CpusetSpec::disjoint(0, 2)),
                Op::set_cpuset("cg_1", CpusetSpec::disjoint(1, 2)),
            ],
            HoldSpec::Frac(0.5),
        ),
    ];

    execute_steps(ctx, steps)
}

/// Clear disjoint cpusets from two cgroups mid-run.
pub fn custom_cgroup_cpuset_clear_midrun(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![
        Step::with_defs(
            vec![
                CgroupDef::named("cg_0").with_cpuset(CpusetSpec::disjoint(0, 2)),
                CgroupDef::named("cg_1").with_cpuset(CpusetSpec::disjoint(1, 2)),
            ],
            HoldSpec::Fixed(ctx.settle + ctx.duration / 2),
        ),
        Step::new(
            vec![Op::clear_cpuset("cg_0"), Op::clear_cpuset("cg_1")],
            HoldSpec::Frac(0.5),
        ),
    ];

    execute_steps(ctx, steps)
}

/// Three-phase cpuset resize: 50/50, 25/75, 75/25.
pub fn custom_cgroup_cpuset_resize(ctx: &Ctx) -> Result<AssertResult> {
    if ctx.topo.all_cpus().len() < 4 {
        return Ok(AssertResult::skip("skipped: need >=4 CPUs"));
    }

    let steps = vec![
        Step::with_defs(
            vec![
                CgroupDef::named("cg_0").with_cpuset(CpusetSpec::range(0.0, 0.5)),
                CgroupDef::named("cg_1").with_cpuset(CpusetSpec::range(0.5, 1.0)),
            ],
            HoldSpec::Fixed(ctx.settle + ctx.duration / 3),
        ),
        Step::new(
            vec![
                Op::set_cpuset("cg_0", CpusetSpec::range(0.0, 0.25)),
                Op::set_cpuset("cg_1", CpusetSpec::range(0.25, 1.0)),
            ],
            HoldSpec::Frac(1.0 / 3.0),
        ),
        Step::new(
            vec![
                Op::set_cpuset("cg_0", CpusetSpec::range(0.0, 0.75)),
                Op::set_cpuset("cg_1", CpusetSpec::range(0.75, 1.0)),
            ],
            HoldSpec::Frac(1.0 / 3.0),
        ),
    ];

    execute_steps(ctx, steps)
}

/// Swap disjoint cpuset assignments between two cgroups twice.
pub fn custom_cgroup_cpuset_swap_disjoint(ctx: &Ctx) -> Result<AssertResult> {
    if ctx.topo.all_cpus().len() < 8 {
        return Ok(AssertResult::skip("skipped: need >=8 CPUs"));
    }

    let steps = vec![
        Step::with_defs(
            vec![
                CgroupDef::named("cg_0").with_cpuset(CpusetSpec::range(0.0, 0.5)),
                CgroupDef::named("cg_1").with_cpuset(CpusetSpec::range(0.5, 1.0)),
            ],
            HoldSpec::Fixed(ctx.settle + ctx.duration / 3),
        ),
        Step::new(
            vec![
                Op::set_cpuset("cg_0", CpusetSpec::range(0.5, 1.0)),
                Op::set_cpuset("cg_1", CpusetSpec::range(0.0, 0.5)),
            ],
            HoldSpec::Frac(1.0 / 3.0),
        ),
        Step::new(
            vec![
                Op::set_cpuset("cg_0", CpusetSpec::range(0.0, 0.5)),
                Op::set_cpuset("cg_1", CpusetSpec::range(0.5, 1.0)),
            ],
            HoldSpec::Frac(1.0 / 3.0),
        ),
    ];

    execute_steps(ctx, steps)
}

/// Disjoint cpusets with oversubscribed CpuSpin vs bursty workers.
pub fn custom_cgroup_cpuset_workload_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    let mid = ctx.topo.usable_cpus().len() / 2;

    let steps = vec![Step::with_defs(
        vec![
            CgroupDef::named("cg_0")
                .with_cpuset(CpusetSpec::disjoint(0, 2))
                .workers(mid * 2),
            CgroupDef::named("cg_1")
                .with_cpuset(CpusetSpec::disjoint(1, 2))
                .work_type(WorkType::bursty(50, 100)),
        ],
        HoldSpec::Fixed(ctx.settle + ctx.duration),
    )];

    execute_steps(ctx, steps)
}

/// Oversubscribed and bursty cgroups with cpuset narrowing and widening.
pub fn custom_cgroup_cpuset_change_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    if ctx.topo.all_cpus().len() < 4 {
        return Ok(AssertResult::skip("skipped: need >=4 CPUs"));
    }

    let all = ctx.topo.all_cpus();
    let last = all.len() - 1;
    let mid = last / 2;

    let narrow = CpusetSpec::exact([all[mid]]);

    let steps = vec![
        Step::with_defs(
            vec![
                CgroupDef::named("cg_0")
                    .with_cpuset(CpusetSpec::range(0.0, 0.5))
                    .workers(mid * 2),
                CgroupDef::named("cg_1")
                    .with_cpuset(CpusetSpec::range(0.5, 1.0))
                    .workers(2)
                    .work_type(WorkType::bursty(30, 100)),
            ],
            HoldSpec::Fixed(ctx.settle + ctx.duration / 3),
        ),
        Step::new(
            vec![Op::set_cpuset("cg_1", narrow)],
            HoldSpec::Frac(1.0 / 3.0),
        ),
        Step::new(
            vec![Op::set_cpuset("cg_1", CpusetSpec::range(0.5, 1.0))],
            HoldSpec::Frac(1.0 / 3.0),
        ),
    ];

    execute_steps(ctx, steps)
}

/// NUMA-scoped cpusets: one cgroup per NUMA node, then swap mid-run.
///
/// Requires a 2+ NUMA node topology. Each cgroup is constrained to a
/// single NUMA node's CPUs, then cpusets are swapped to force cross-NUMA
/// migration.
pub fn custom_cgroup_cpuset_numa_swap(ctx: &Ctx) -> Result<AssertResult> {
    if ctx.topo.num_numa_nodes() < 2 {
        return Ok(AssertResult::skip("skipped: need >=2 NUMA nodes"));
    }

    let steps = vec![
        Step::with_defs(
            vec![
                CgroupDef::named("cg_0").with_cpuset(CpusetSpec::numa(0)),
                CgroupDef::named("cg_1").with_cpuset(CpusetSpec::numa(1)),
            ],
            HoldSpec::Fixed(ctx.settle + ctx.duration / 2),
        ),
        Step::new(
            vec![
                Op::set_cpuset("cg_0", CpusetSpec::numa(1)),
                Op::set_cpuset("cg_1", CpusetSpec::numa(0)),
            ],
            HoldSpec::Frac(0.5),
        ),
    ];

    execute_steps(ctx, steps)
}

/// Disjoint cpusets where a light cgroup gets heavy load added mid-run.
pub fn custom_cgroup_cpuset_load_shift(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![
        Step::with_defs(
            vec![
                CgroupDef::named("cg_0")
                    .with_cpuset(CpusetSpec::disjoint(0, 2))
                    .workers(16),
                CgroupDef::named("cg_1")
                    .with_cpuset(CpusetSpec::disjoint(1, 2))
                    .workers(1)
                    .work_type(WorkType::YieldHeavy),
            ],
            HoldSpec::Fixed(ctx.settle + ctx.duration / 2),
        ),
        // Phase 2: add heavy load to cg_1
        Step::new(
            vec![Op::spawn("cg_1", Work::default().workers(16))],
            HoldSpec::Frac(0.5),
        ),
    ];

    execute_steps(ctx, steps)
}
