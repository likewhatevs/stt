//! Cross-cgroup interaction scenario implementations.

use super::Ctx;
use super::ops::{CgroupDef, CpusetSpec, HoldSpec, Op, Step, execute_steps};
use crate::assert::AssertResult;
use crate::workload::*;
use anyhow::Result;
use std::time::Duration;

pub fn custom_cgroup_add_load_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![
        Step {
            setup: vec![
                CgroupDef::named("cg_0")
                    .workers(1)
                    .work_type(WorkType::YieldHeavy),
                CgroupDef::named("cg_1")
                    .workers(1)
                    .work_type(WorkType::YieldHeavy),
            ]
            .into(),
            ops: vec![],
            hold: HoldSpec::Fixed(ctx.settle + ctx.duration / 2),
        },
        Step {
            setup: vec![CgroupDef::named("cg_2").workers(16)].into(),
            ops: vec![],
            hold: HoldSpec::Frac(0.5),
        },
    ];

    execute_steps(ctx, steps)
}

pub fn custom_cgroup_imbalance_mixed_workload(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![
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
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::Fixed(ctx.settle + ctx.duration),
    }];

    execute_steps(ctx, steps)
}

pub fn custom_cgroup_load_oscillation(ctx: &Ctx) -> Result<AssertResult> {
    let heavy = Work::default().workers(ctx.workers_per_cgroup * 2);
    let light = Work::default().workers(1).work_type(WorkType::YieldHeavy);

    let mut steps = vec![Step {
        setup: vec![
            CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup * 2),
            CgroupDef::named("cg_1")
                .workers(1)
                .work_type(WorkType::YieldHeavy),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::Fixed(ctx.settle + ctx.duration / 4),
    }];

    // Phases 1-3: swap load by stopping and respawning.
    for i in 1..4 {
        let (heavy_cgroup, light_cgroup): (&str, &str) = if i % 2 == 0 {
            ("cg_0", "cg_1")
        } else {
            ("cg_1", "cg_0")
        };
        steps.push(Step {
            setup: vec![].into(),
            ops: vec![
                Op::StopCgroup {
                    cgroup: "cg_0".into(),
                },
                Op::StopCgroup {
                    cgroup: "cg_1".into(),
                },
                Op::Spawn {
                    cgroup: heavy_cgroup.into(),
                    work: heavy.clone(),
                },
                Op::Spawn {
                    cgroup: light_cgroup.into(),
                    work: light.clone(),
                },
            ],
            hold: HoldSpec::Frac(0.25),
        });
    }

    execute_steps(ctx, steps)
}

pub fn custom_cgroup_4way_load_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    if ctx.topo.all_cpus().len() < 5 {
        return Ok(AssertResult::skip("skipped: need >=5 CPUs for 4 cgroups"));
    }

    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cg_0").workers(16),
            CgroupDef::named("cg_1")
                .workers(1)
                .work_type(WorkType::YieldHeavy),
            CgroupDef::named("cg_2").workers(8),
            CgroupDef::named("cg_3").workers(4),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::Fixed(ctx.settle + ctx.duration),
    }];

    execute_steps(ctx, steps)
}

pub fn custom_cgroup_cpuset_imbalance_combined(ctx: &Ctx) -> Result<AssertResult> {
    let mid = ctx.topo.usable_cpus().len() / 2;

    let steps = vec![Step {
        setup: vec![
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
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::Fixed(ctx.settle + ctx.duration),
    }];

    execute_steps(ctx, steps)
}

pub fn custom_cgroup_cpuset_overlap_imbalance_combined(ctx: &Ctx) -> Result<AssertResult> {
    let sets = ctx.topo.overlapping_cpusets(3, 0.5);
    if sets.iter().any(|s| s.is_empty()) {
        return Ok(AssertResult::skip("skipped: not enough CPUs"));
    }

    let steps = vec![Step {
        setup: vec![
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
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::Fixed(ctx.settle + ctx.duration),
    }];

    execute_steps(ctx, steps)
}

pub fn custom_cgroup_noctrl_task_migration(ctx: &Ctx) -> Result<AssertResult> {
    let half = ctx.workers_per_cgroup;

    let mut move_steps: Vec<Step> = (0..9)
        .map(|i| {
            let target = if i % 2 == 0 { "cg_1" } else { "cg_0" };
            let from = if i % 2 == 0 { "cg_0" } else { "cg_1" };
            Step {
                setup: vec![].into(),
                ops: vec![Op::MoveTasks {
                    from: from.into(),
                    to: target.into(),
                    count: half,
                }],
                hold: HoldSpec::Frac(0.1),
            }
        })
        .collect();

    let mut steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup * 2)].into(),
        ops: vec![Op::AddCgroup {
            name: "cg_1".into(),
        }],
        hold: HoldSpec::Fixed(Duration::from_secs(2)),
    }];
    steps.append(&mut move_steps);
    // Final hold for remaining time.
    steps.push(Step {
        setup: vec![].into(),
        ops: vec![],
        hold: HoldSpec::Frac(0.1),
    });

    execute_steps(ctx, steps)
}

pub fn custom_cgroup_noctrl_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    let mut move_steps: Vec<Step> = (0..5)
        .map(|i| {
            let (from, to) = if i % 2 == 0 {
                ("cg_0", "cg_1")
            } else {
                ("cg_1", "cg_0")
            };
            Step {
                setup: vec![].into(),
                ops: vec![Op::MoveTasks {
                    from: from.into(),
                    to: to.into(),
                    count: 2,
                }],
                hold: HoldSpec::Frac(1.0 / 6.0),
            }
        })
        .collect();

    let mut steps = vec![Step {
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
        hold: HoldSpec::Fixed(ctx.settle),
    }];
    steps.append(&mut move_steps);
    steps.push(Step {
        setup: vec![].into(),
        ops: vec![],
        hold: HoldSpec::Frac(1.0 / 6.0),
    });

    execute_steps(ctx, steps)
}

pub fn custom_cgroup_noctrl_cpuset_change(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![
        Step {
            setup: vec![
                CgroupDef::named("cg_0").with_cpuset(CpusetSpec::Disjoint { index: 0, of: 2 }),
                CgroupDef::named("cg_1").with_cpuset(CpusetSpec::Disjoint { index: 1, of: 2 }),
            ]
            .into(),
            ops: vec![],
            hold: HoldSpec::Fixed(ctx.settle + ctx.duration / 2),
        },
        // Phase 2: clear cpusets, hold remaining half.
        Step {
            setup: vec![].into(),
            ops: vec![
                Op::ClearCpuset {
                    cgroup: "cg_0".into(),
                },
                Op::ClearCpuset {
                    cgroup: "cg_1".into(),
                },
            ],
            hold: HoldSpec::Frac(0.5),
        },
    ];

    execute_steps(ctx, steps)
}

pub fn custom_cgroup_noctrl_load_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cg_0").workers(16),
            CgroupDef::named("cg_1")
                .workers(1)
                .work_type(WorkType::YieldHeavy),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::Fixed(ctx.settle + ctx.duration),
    }];

    execute_steps(ctx, steps)
}

pub fn custom_cgroup_io_compute_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cg_0")
                .workers(ctx.workers_per_cgroup)
                .work_type(WorkType::IoSync),
            CgroupDef::named("cg_1").workers(ctx.topo.total_cpus()),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::Fixed(ctx.settle + ctx.duration),
    }];

    execute_steps(ctx, steps)
}
