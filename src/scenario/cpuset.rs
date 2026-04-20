//! Cpuset mutation scenario implementations.

use super::Ctx;
use super::backdrop::Backdrop;
use super::ops::{CgroupDef, CpusetSpec, HoldSpec, Op, Step, execute_scenario, execute_steps};
use crate::assert::AssertResult;
use crate::workload::*;
use anyhow::Result;

fn cgroup_cpuset_apply_midrun_backdrop() -> Backdrop {
    Backdrop::new()
        .with_cgroup(CgroupDef::named("cg_0"))
        .with_cgroup(CgroupDef::named("cg_1"))
}

fn cgroup_cpuset_apply_midrun_steps(ctx: &Ctx) -> Vec<Step> {
    vec![
        Step::new(vec![], HoldSpec::Fixed(ctx.settle + ctx.duration / 2)),
        Step::new(
            vec![
                Op::set_cpuset("cg_0", CpusetSpec::disjoint(0, 2)),
                Op::set_cpuset("cg_1", CpusetSpec::disjoint(1, 2)),
            ],
            HoldSpec::Frac(0.5),
        ),
    ]
}

/// Apply disjoint cpusets to two initially unconstrained cgroups mid-run.
pub fn custom_cgroup_cpuset_apply_midrun(ctx: &Ctx) -> Result<AssertResult> {
    execute_scenario(
        ctx,
        cgroup_cpuset_apply_midrun_backdrop(),
        cgroup_cpuset_apply_midrun_steps(ctx),
    )
}

/// Clear disjoint cpusets from two cgroups mid-run.
pub fn custom_cgroup_cpuset_clear_midrun(ctx: &Ctx) -> Result<AssertResult> {
    let backdrop = Backdrop::new()
        .with_cgroup(CgroupDef::named("cg_0").with_cpuset(CpusetSpec::disjoint(0, 2)))
        .with_cgroup(CgroupDef::named("cg_1").with_cpuset(CpusetSpec::disjoint(1, 2)));

    let steps = vec![
        Step::new(vec![], HoldSpec::Fixed(ctx.settle + ctx.duration / 2)),
        Step::new(
            vec![Op::clear_cpuset("cg_0"), Op::clear_cpuset("cg_1")],
            HoldSpec::Frac(0.5),
        ),
    ];

    execute_scenario(ctx, backdrop, steps)
}

fn cgroup_cpuset_resize_backdrop() -> Backdrop {
    Backdrop::new()
        .with_cgroup(CgroupDef::named("cg_0").with_cpuset(CpusetSpec::range(0.0, 0.5)))
        .with_cgroup(CgroupDef::named("cg_1").with_cpuset(CpusetSpec::range(0.5, 1.0)))
}

fn cgroup_cpuset_resize_steps(ctx: &Ctx) -> Vec<Step> {
    vec![
        Step::new(vec![], HoldSpec::Fixed(ctx.settle + ctx.duration / 3)),
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
    ]
}

/// Three-phase cpuset resize: 50/50, 25/75, 75/25.
pub fn custom_cgroup_cpuset_resize(ctx: &Ctx) -> Result<AssertResult> {
    if ctx.topo.all_cpus().len() < 4 {
        return Ok(AssertResult::skip("skipped: need >=4 CPUs"));
    }
    execute_scenario(
        ctx,
        cgroup_cpuset_resize_backdrop(),
        cgroup_cpuset_resize_steps(ctx),
    )
}

/// Swap disjoint cpuset assignments between two cgroups twice.
pub fn custom_cgroup_cpuset_swap_disjoint(ctx: &Ctx) -> Result<AssertResult> {
    if ctx.topo.all_cpus().len() < 8 {
        return Ok(AssertResult::skip("skipped: need >=8 CPUs"));
    }

    let backdrop = Backdrop::new()
        .with_cgroup(CgroupDef::named("cg_0").with_cpuset(CpusetSpec::range(0.0, 0.5)))
        .with_cgroup(CgroupDef::named("cg_1").with_cpuset(CpusetSpec::range(0.5, 1.0)));

    let steps = vec![
        Step::new(vec![], HoldSpec::Fixed(ctx.settle + ctx.duration / 3)),
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

    execute_scenario(ctx, backdrop, steps)
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

    let backdrop = Backdrop::new()
        .with_cgroup(
            CgroupDef::named("cg_0")
                .with_cpuset(CpusetSpec::range(0.0, 0.5))
                .workers(mid * 2),
        )
        .with_cgroup(
            CgroupDef::named("cg_1")
                .with_cpuset(CpusetSpec::range(0.5, 1.0))
                .workers(2)
                .work_type(WorkType::bursty(30, 100)),
        );

    let steps = vec![
        Step::new(vec![], HoldSpec::Fixed(ctx.settle + ctx.duration / 3)),
        Step::new(
            vec![Op::set_cpuset("cg_1", narrow)],
            HoldSpec::Frac(1.0 / 3.0),
        ),
        Step::new(
            vec![Op::set_cpuset("cg_1", CpusetSpec::range(0.5, 1.0))],
            HoldSpec::Frac(1.0 / 3.0),
        ),
    ];

    execute_scenario(ctx, backdrop, steps)
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

    let backdrop = Backdrop::new()
        .with_cgroup(CgroupDef::named("cg_0").with_cpuset(CpusetSpec::numa(0)))
        .with_cgroup(CgroupDef::named("cg_1").with_cpuset(CpusetSpec::numa(1)));

    let steps = vec![
        Step::new(vec![], HoldSpec::Fixed(ctx.settle + ctx.duration / 2)),
        Step::new(
            vec![
                Op::set_cpuset("cg_0", CpusetSpec::numa(1)),
                Op::set_cpuset("cg_1", CpusetSpec::numa(0)),
            ],
            HoldSpec::Frac(0.5),
        ),
    ];

    execute_scenario(ctx, backdrop, steps)
}

/// Disjoint cpusets where a light cgroup gets heavy load added mid-run.
pub fn custom_cgroup_cpuset_load_shift(ctx: &Ctx) -> Result<AssertResult> {
    let backdrop = Backdrop::new()
        .with_cgroup(
            CgroupDef::named("cg_0")
                .with_cpuset(CpusetSpec::disjoint(0, 2))
                .workers(16),
        )
        .with_cgroup(
            CgroupDef::named("cg_1")
                .with_cpuset(CpusetSpec::disjoint(1, 2))
                .workers(1)
                .work_type(WorkType::YieldHeavy),
        );

    let steps = vec![
        Step::new(vec![], HoldSpec::Fixed(ctx.settle + ctx.duration / 2)),
        // Phase 2: add heavy step-local load to cg_1. The new workers
        // die at step teardown — which is what the prior
        // execute_steps behavior eventually did at scenario end too.
        Step::new(
            vec![Op::spawn("cg_1", Work::default().workers(16))],
            HoldSpec::Frac(0.5),
        ),
    ];

    execute_scenario(ctx, backdrop, steps)
}

#[cfg(test)]
mod tests {
    use super::super::ops::Setup;
    use super::*;
    use crate::cgroup::CgroupManager;
    use crate::topology::TestTopology;
    use std::time::Duration;

    fn ctx_for_test<'a>(cgroups: &'a CgroupManager, topo: &'a TestTopology) -> Ctx<'a> {
        Ctx {
            cgroups,
            topo,
            duration: Duration::from_secs(6),
            workers_per_cgroup: 2,
            sched_pid: 1,
            settle: Duration::from_millis(100),
            work_type_override: None,
            assert: crate::assert::Assert::default_checks(),
            wait_for_map_write: false,
        }
    }

    #[test]
    fn apply_midrun_backdrop_declares_two_cgroups() {
        let backdrop = cgroup_cpuset_apply_midrun_backdrop();
        assert_eq!(
            backdrop.cgroups.len(),
            2,
            "Backdrop declares cg_0 and cg_1 as persistent"
        );
        assert_eq!(backdrop.cgroups[0].name.as_ref(), "cg_0");
        assert_eq!(backdrop.cgroups[1].name.as_ref(), "cg_1");
    }

    #[test]
    fn apply_midrun_builds_two_phase_steps() {
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::from_vm_topology(&crate::vmm::topology::Topology::new(1, 1, 4, 1));
        let ctx = ctx_for_test(&cgroups, &topo);

        let steps = cgroup_cpuset_apply_midrun_steps(&ctx);
        assert_eq!(steps.len(), 2, "settle + apply phases");

        assert!(
            matches!(&steps[0].setup, Setup::Defs(defs) if defs.is_empty()),
            "phase 1 has no step-local CgroupDefs (cgroups live in the Backdrop)",
        );
        assert!(steps[0].ops.is_empty(), "phase 1 is a pure settle — no ops");

        assert!(matches!(steps[0].hold, HoldSpec::Fixed(_)));
        let phase2_ops = &steps[1].ops;
        assert_eq!(phase2_ops.len(), 2, "set_cpuset once per cgroup");
        for op in phase2_ops {
            assert!(matches!(op, Op::SetCpuset { .. }));
        }
        assert!(matches!(steps[1].hold, HoldSpec::Frac(f) if (f - 0.5).abs() < f64::EPSILON));
    }

    #[test]
    fn resize_backdrop_declares_two_cgroups_with_cpusets() {
        let backdrop = cgroup_cpuset_resize_backdrop();
        assert_eq!(backdrop.cgroups.len(), 2);
        assert!(backdrop.cgroups[0].cpuset.is_some());
        assert!(backdrop.cgroups[1].cpuset.is_some());
    }

    #[test]
    fn resize_builds_three_phase_range_progression() {
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::from_vm_topology(&crate::vmm::topology::Topology::new(1, 1, 4, 1));
        let ctx = ctx_for_test(&cgroups, &topo);

        let steps = cgroup_cpuset_resize_steps(&ctx);
        assert_eq!(steps.len(), 3);
        assert!(
            matches!(&steps[0].setup, Setup::Defs(defs) if defs.is_empty()),
            "phase 1 is a settle step — cgroups live in the Backdrop",
        );
        assert!(steps[0].ops.is_empty(), "phase 1 has no ops");
        // Phases 2 and 3: each reassigns cpusets on both cgroups.
        for step in &steps[1..] {
            assert_eq!(step.ops.len(), 2);
            for op in &step.ops {
                assert!(matches!(op, Op::SetCpuset { .. }));
            }
        }
    }
}
