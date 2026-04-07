use super::Ctx;
use super::ops::{CgroupDef, CpusetSpec, HoldSpec, Op, Step, execute_steps};
use crate::assert::AssertResult;
use crate::workload::*;
use anyhow::Result;
use std::collections::BTreeSet;
use std::time::Duration;

pub fn custom_cgroup_cpuset_apply_midrun(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![
        Step {
            setup: vec![CgroupDef::named("cg_0"), CgroupDef::named("cg_1")].into(),
            ops: vec![],
            hold: HoldSpec::Fixed(Duration::from_millis(ctx.settle_ms) + ctx.duration / 2),
        },
        Step {
            setup: vec![].into(),
            ops: vec![
                Op::SetCpuset {
                    cgroup: "cg_0".into(),
                    cpus: CpusetSpec::Disjoint { index: 0, of: 2 },
                },
                Op::SetCpuset {
                    cgroup: "cg_1".into(),
                    cpus: CpusetSpec::Disjoint { index: 1, of: 2 },
                },
            ],
            hold: HoldSpec::Frac(0.5),
        },
    ];

    execute_steps(ctx, steps)
}

pub fn custom_cgroup_cpuset_clear_midrun(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![
        Step {
            setup: vec![
                CgroupDef::named("cg_0").with_cpuset(CpusetSpec::Disjoint { index: 0, of: 2 }),
                CgroupDef::named("cg_1").with_cpuset(CpusetSpec::Disjoint { index: 1, of: 2 }),
            ]
            .into(),
            ops: vec![],
            hold: HoldSpec::Fixed(Duration::from_secs(3) + ctx.duration / 2),
        },
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

pub fn custom_cgroup_cpuset_resize(ctx: &Ctx) -> Result<AssertResult> {
    if ctx.topo.all_cpus().len() < 4 {
        return Ok(AssertResult {
            passed: true,
            details: vec!["skipped: need >=4 CPUs".into()],
            stats: Default::default(),
        });
    }

    let steps = vec![
        Step {
            setup: vec![
                CgroupDef::named("cg_0").with_cpuset(CpusetSpec::Range {
                    start_frac: 0.0,
                    end_frac: 0.5,
                }),
                CgroupDef::named("cg_1").with_cpuset(CpusetSpec::Range {
                    start_frac: 0.5,
                    end_frac: 1.0,
                }),
            ]
            .into(),
            ops: vec![],
            hold: HoldSpec::Fixed(Duration::from_secs(3) + ctx.duration / 3),
        },
        Step {
            setup: vec![].into(),
            ops: vec![
                Op::SetCpuset {
                    cgroup: "cg_0".into(),
                    cpus: CpusetSpec::Range {
                        start_frac: 0.0,
                        end_frac: 0.25,
                    },
                },
                Op::SetCpuset {
                    cgroup: "cg_1".into(),
                    cpus: CpusetSpec::Range {
                        start_frac: 0.25,
                        end_frac: 1.0,
                    },
                },
            ],
            hold: HoldSpec::Frac(1.0 / 3.0),
        },
        Step {
            setup: vec![].into(),
            ops: vec![
                Op::SetCpuset {
                    cgroup: "cg_0".into(),
                    cpus: CpusetSpec::Range {
                        start_frac: 0.0,
                        end_frac: 0.75,
                    },
                },
                Op::SetCpuset {
                    cgroup: "cg_1".into(),
                    cpus: CpusetSpec::Range {
                        start_frac: 0.75,
                        end_frac: 1.0,
                    },
                },
            ],
            hold: HoldSpec::Frac(1.0 / 3.0),
        },
    ];

    execute_steps(ctx, steps)
}

pub fn custom_cgroup_cpuset_swap_disjoint(ctx: &Ctx) -> Result<AssertResult> {
    if ctx.topo.all_cpus().len() < 8 {
        return Ok(AssertResult {
            passed: true,
            details: vec!["skipped: need >=8 CPUs".into()],
            stats: Default::default(),
        });
    }

    let steps = vec![
        Step {
            setup: vec![
                CgroupDef::named("cg_0").with_cpuset(CpusetSpec::Range {
                    start_frac: 0.0,
                    end_frac: 0.5,
                }),
                CgroupDef::named("cg_1").with_cpuset(CpusetSpec::Range {
                    start_frac: 0.5,
                    end_frac: 1.0,
                }),
            ]
            .into(),
            ops: vec![],
            hold: HoldSpec::Fixed(Duration::from_millis(ctx.settle_ms) + ctx.duration / 3),
        },
        Step {
            setup: vec![].into(),
            ops: vec![
                Op::SetCpuset {
                    cgroup: "cg_0".into(),
                    cpus: CpusetSpec::Range {
                        start_frac: 0.5,
                        end_frac: 1.0,
                    },
                },
                Op::SetCpuset {
                    cgroup: "cg_1".into(),
                    cpus: CpusetSpec::Range {
                        start_frac: 0.0,
                        end_frac: 0.5,
                    },
                },
            ],
            hold: HoldSpec::Frac(1.0 / 3.0),
        },
        Step {
            setup: vec![].into(),
            ops: vec![
                Op::SetCpuset {
                    cgroup: "cg_0".into(),
                    cpus: CpusetSpec::Range {
                        start_frac: 0.0,
                        end_frac: 0.5,
                    },
                },
                Op::SetCpuset {
                    cgroup: "cg_1".into(),
                    cpus: CpusetSpec::Range {
                        start_frac: 0.5,
                        end_frac: 1.0,
                    },
                },
            ],
            hold: HoldSpec::Frac(1.0 / 3.0),
        },
    ];

    execute_steps(ctx, steps)
}

pub fn custom_cgroup_cpuset_workload_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    let mid = ctx.topo.usable_cpus().len() / 2;

    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cg_0")
                .with_cpuset(CpusetSpec::Disjoint { index: 0, of: 2 })
                .workers(mid * 2),
            CgroupDef::named("cg_1")
                .with_cpuset(CpusetSpec::Disjoint { index: 1, of: 2 })
                .work_type(WorkType::Bursty {
                    burst_ms: 50,
                    sleep_ms: 100,
                }),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::Fixed(Duration::from_secs(3) + ctx.duration),
    }];

    execute_steps(ctx, steps)
}

pub fn custom_cgroup_cpuset_change_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    if ctx.topo.all_cpus().len() < 4 {
        return Ok(AssertResult {
            passed: true,
            details: vec!["skipped: need >=4 CPUs".into()],
            stats: Default::default(),
        });
    }

    let all = ctx.topo.all_cpus();
    let last = all.len() - 1;
    let mid = last / 2;

    let narrow: BTreeSet<usize> = all[mid..mid + 1].iter().copied().collect();

    let steps = vec![
        Step {
            setup: vec![
                CgroupDef::named("cg_0")
                    .with_cpuset(CpusetSpec::Range {
                        start_frac: 0.0,
                        end_frac: 0.5,
                    })
                    .workers(mid * 2),
                CgroupDef::named("cg_1")
                    .with_cpuset(CpusetSpec::Range {
                        start_frac: 0.5,
                        end_frac: 1.0,
                    })
                    .workers(2)
                    .work_type(WorkType::Bursty {
                        burst_ms: 30,
                        sleep_ms: 100,
                    }),
            ]
            .into(),
            ops: vec![],
            hold: HoldSpec::Fixed(Duration::from_secs(3) + ctx.duration / 3),
        },
        Step {
            setup: vec![].into(),
            ops: vec![Op::SetCpuset {
                cgroup: "cg_1".into(),
                cpus: CpusetSpec::Exact(narrow),
            }],
            hold: HoldSpec::Frac(1.0 / 3.0),
        },
        Step {
            setup: vec![].into(),
            ops: vec![Op::SetCpuset {
                cgroup: "cg_1".into(),
                cpus: CpusetSpec::Range {
                    start_frac: 0.5,
                    end_frac: 1.0,
                },
            }],
            hold: HoldSpec::Frac(1.0 / 3.0),
        },
    ];

    execute_steps(ctx, steps)
}

pub fn custom_cgroup_cpuset_load_shift(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![
        Step {
            setup: vec![
                CgroupDef::named("cg_0")
                    .with_cpuset(CpusetSpec::Disjoint { index: 0, of: 2 })
                    .workers(16),
                CgroupDef::named("cg_1")
                    .with_cpuset(CpusetSpec::Disjoint { index: 1, of: 2 })
                    .workers(1)
                    .work_type(WorkType::YieldHeavy),
            ]
            .into(),
            ops: vec![],
            hold: HoldSpec::Fixed(Duration::from_secs(3) + ctx.duration / 2),
        },
        // Phase 2: add heavy load to cg_1
        Step {
            setup: vec![].into(),
            ops: vec![Op::Spawn {
                cgroup: "cg_1".into(),
                workload: WorkloadConfig {
                    num_workers: 16,
                    ..Default::default()
                },
            }],
            hold: HoldSpec::Frac(0.5),
        },
    ];

    execute_steps(ctx, steps)
}
