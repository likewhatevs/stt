//! Cross-cgroup interaction scenario implementations.

use super::Ctx;
use super::ops::{CgroupDef, CpusetSpec, HoldSpec, Op, Step, execute_steps};
use crate::assert::AssertResult;
use crate::workload::*;
use anyhow::Result;
use std::time::Duration;

/// Add a heavy 16-worker cgroup mid-run alongside two light YieldHeavy
/// cgroups.
pub fn custom_cgroup_add_load_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![
        Step::with_defs(
            vec![
                CgroupDef::named("cg_0")
                    .workers(1)
                    .work_type(WorkType::YieldHeavy),
                CgroupDef::named("cg_1")
                    .workers(1)
                    .work_type(WorkType::YieldHeavy),
            ],
            HoldSpec::Fixed(ctx.settle + ctx.duration / 2),
        ),
        Step::with_defs(
            vec![CgroupDef::named("cg_2").workers(16)],
            HoldSpec::Frac(0.5),
        ),
    ];

    execute_steps(ctx, steps)
}

/// Three cgroups with CpuSpin, Bursty, and IoSync workloads.
pub fn custom_cgroup_imbalance_mixed_workload(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step::with_defs(
        vec![
            CgroupDef::named("cg_0").workers(8),
            CgroupDef::named("cg_1")
                .workers(ctx.workers_per_cgroup)
                .work_type(WorkType::Bursty {
                    burst_ms: 100,
                    sleep_ms: 50,
                }),
            CgroupDef::named("cg_2")
                .workers(ctx.workers_per_cgroup)
                .work_type(WorkType::IoSync),
        ],
        HoldSpec::Fixed(ctx.settle + ctx.duration),
    )];

    execute_steps(ctx, steps)
}

/// Oscillate load between two cgroups across four phases.
pub fn custom_cgroup_load_oscillation(ctx: &Ctx) -> Result<AssertResult> {
    let heavy = Work::default().workers(ctx.workers_per_cgroup * 2);
    let light = Work::default().workers(1).work_type(WorkType::YieldHeavy);

    let mut steps = vec![Step::with_defs(
        vec![
            CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup * 2),
            CgroupDef::named("cg_1")
                .workers(1)
                .work_type(WorkType::YieldHeavy),
        ],
        HoldSpec::Fixed(ctx.settle + ctx.duration / 4),
    )];

    // Phases 1-3: swap load by stopping and respawning.
    for i in 1..4 {
        let (heavy_cgroup, light_cgroup): (&str, &str) = if i % 2 == 0 {
            ("cg_0", "cg_1")
        } else {
            ("cg_1", "cg_0")
        };
        steps.push(Step::new(
            vec![
                Op::stop_cgroup("cg_0"),
                Op::stop_cgroup("cg_1"),
                Op::spawn(heavy_cgroup, heavy.clone()),
                Op::spawn(light_cgroup, light.clone()),
            ],
            HoldSpec::Frac(0.25),
        ));
    }

    execute_steps(ctx, steps)
}

/// Four cgroups with 16/1/8/4 workers testing multi-cell rebalancing.
pub fn custom_cgroup_4way_load_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    if ctx.topo.all_cpus().len() < 5 {
        return Ok(AssertResult::skip("skipped: need >=5 CPUs for 4 cgroups"));
    }

    let steps = vec![Step::with_defs(
        vec![
            CgroupDef::named("cg_0").workers(16),
            CgroupDef::named("cg_1")
                .workers(1)
                .work_type(WorkType::YieldHeavy),
            CgroupDef::named("cg_2").workers(8),
            CgroupDef::named("cg_3").workers(4),
        ],
        HoldSpec::Fixed(ctx.settle + ctx.duration),
    )];

    execute_steps(ctx, steps)
}

/// Disjoint cpusets with oversubscribed CpuSpin vs light Bursty workers.
pub fn custom_cgroup_cpuset_imbalance_combined(ctx: &Ctx) -> Result<AssertResult> {
    let mid = ctx.topo.usable_cpus().len() / 2;

    let steps = vec![Step::with_defs(
        vec![
            CgroupDef::named("cg_0")
                .with_cpuset(CpusetSpec::Disjoint { index: 0, of: 2 })
                .workers(mid * 2),
            CgroupDef::named("cg_1")
                .with_cpuset(CpusetSpec::Disjoint { index: 1, of: 2 })
                .workers(2)
                .work_type(WorkType::Bursty {
                    burst_ms: 50,
                    sleep_ms: 150,
                }),
        ],
        HoldSpec::Fixed(ctx.settle + ctx.duration),
    )];

    execute_steps(ctx, steps)
}

/// Three overlapping cpusets with heavy, bursty, and yield-heavy workers.
pub fn custom_cgroup_cpuset_overlap_imbalance_combined(ctx: &Ctx) -> Result<AssertResult> {
    let sets = ctx.topo.overlapping_cpusets(3, 0.5);
    if sets.iter().any(|s| s.is_empty()) {
        return Ok(AssertResult::skip("skipped: not enough CPUs"));
    }

    let steps = vec![Step::with_defs(
        vec![
            CgroupDef::named("cg_0")
                .with_cpuset(CpusetSpec::Exact(sets[0].clone()))
                .workers(12),
            CgroupDef::named("cg_1")
                .with_cpuset(CpusetSpec::Exact(sets[1].clone()))
                .workers(2)
                .work_type(WorkType::Bursty {
                    burst_ms: 50,
                    sleep_ms: 100,
                }),
            CgroupDef::named("cg_2")
                .with_cpuset(CpusetSpec::Exact(sets[2].clone()))
                .workers(1)
                .work_type(WorkType::YieldHeavy),
        ],
        HoldSpec::Fixed(ctx.settle + ctx.duration),
    )];

    execute_steps(ctx, steps)
}

/// Workers ping-pong between cg_mobile and cg_1 across 9 MoveAllTasks
/// phases.
pub fn custom_cgroup_noctrl_task_migration(ctx: &Ctx) -> Result<AssertResult> {
    // cg_0: permanent residents. cg_mobile: workers that ping-pong to cg_1.
    // Each cgroup has exactly one handle so MoveAllTasks tracks correctly.
    let mut move_steps: Vec<Step> = (0..9)
        .map(|i| {
            let (from, to) = if i % 2 == 0 {
                ("cg_mobile", "cg_1")
            } else {
                ("cg_1", "cg_mobile")
            };
            Step::new(vec![Op::move_all_tasks(from, to)], HoldSpec::Frac(0.1))
        })
        .collect();

    let mut steps = vec![
        Step::with_defs(
            vec![
                CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup),
                CgroupDef::named("cg_mobile").workers(ctx.workers_per_cgroup),
            ],
            HoldSpec::Fixed(Duration::from_secs(2)),
        )
        .with_ops(vec![Op::add_cgroup("cg_1")]),
    ];
    steps.append(&mut move_steps);
    steps.push(Step::new(vec![], HoldSpec::Frac(0.1)));

    execute_steps(ctx, steps)
}

/// Heavy, light, and mobile workers with tasks ping-ponging to overflow
/// cgroup.
pub fn custom_cgroup_noctrl_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    // cg_heavy: 6 permanent CPU-spin workers.
    // cg_light: 2 permanent bursty workers.
    // cg_mobile: 2 workers that ping-pong to cg_overflow.
    // Each cgroup has at most one handle so MoveAllTasks tracks correctly.
    let mut move_steps: Vec<Step> = (0..5)
        .map(|i| {
            let (from, to) = if i % 2 == 0 {
                ("cg_mobile", "cg_overflow")
            } else {
                ("cg_overflow", "cg_mobile")
            };
            Step::new(
                vec![Op::move_all_tasks(from, to)],
                HoldSpec::Frac(1.0 / 6.0),
            )
        })
        .collect();

    let mut steps = vec![
        Step::with_defs(
            vec![
                CgroupDef::named("cg_heavy").workers(6),
                CgroupDef::named("cg_mobile").workers(2),
                CgroupDef::named("cg_light")
                    .workers(2)
                    .work_type(WorkType::Bursty {
                        burst_ms: 50,
                        sleep_ms: 100,
                    }),
            ],
            HoldSpec::Fixed(ctx.settle),
        )
        .with_ops(vec![Op::add_cgroup("cg_overflow")]),
    ];
    steps.append(&mut move_steps);
    steps.push(Step::new(vec![], HoldSpec::Frac(1.0 / 6.0)));

    execute_steps(ctx, steps)
}

/// Disjoint cpusets cleared mid-run with cpu-controller disabled.
pub fn custom_cgroup_noctrl_cpuset_change(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![
        Step::with_defs(
            vec![
                CgroupDef::named("cg_0").with_cpuset(CpusetSpec::Disjoint { index: 0, of: 2 }),
                CgroupDef::named("cg_1").with_cpuset(CpusetSpec::Disjoint { index: 1, of: 2 }),
            ],
            HoldSpec::Fixed(ctx.settle + ctx.duration / 2),
        ),
        // Phase 2: clear cpusets, hold remaining half.
        Step::new(
            vec![Op::clear_cpuset("cg_0"), Op::clear_cpuset("cg_1")],
            HoldSpec::Frac(0.5),
        ),
    ];

    execute_steps(ctx, steps)
}

/// Heavy CpuSpin vs light YieldHeavy cgroups with cpu-controller disabled.
pub fn custom_cgroup_noctrl_load_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step::with_defs(
        vec![
            CgroupDef::named("cg_0").workers(16),
            CgroupDef::named("cg_1")
                .workers(1)
                .work_type(WorkType::YieldHeavy),
        ],
        HoldSpec::Fixed(ctx.settle + ctx.duration),
    )];

    execute_steps(ctx, steps)
}

/// IoSync cgroup vs fully-subscribed CpuSpin cgroup.
pub fn custom_cgroup_io_compute_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step::with_defs(
        vec![
            CgroupDef::named("cg_0")
                .workers(ctx.workers_per_cgroup)
                .work_type(WorkType::IoSync),
            CgroupDef::named("cg_1").workers(ctx.topo.total_cpus()),
        ],
        HoldSpec::Fixed(ctx.settle + ctx.duration),
    )];

    execute_steps(ctx, steps)
}
