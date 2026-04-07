use super::Ctx;
use super::ops::{CgroupDef, HoldSpec, Op, Step, execute_steps};
use crate::assert::AssertResult;
use crate::workload::*;
use anyhow::Result;
use std::time::Duration;

pub fn custom_host_cgroup_contention(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0"), CgroupDef::named("cg_1")].into(),
        ops: vec![Op::SpawnHost {
            workload: WorkloadConfig {
                num_workers: ctx.topo.total_cpus(),
                ..Default::default()
            },
        }],
        hold: HoldSpec::Fixed(Duration::from_millis(ctx.settle_ms) + ctx.duration),
    }];

    execute_steps(ctx, steps)
}

pub fn custom_sched_mixed(ctx: &Ctx) -> Result<AssertResult> {
    let configs = [
        (SchedPolicy::Normal, WorkType::CpuSpin),
        (SchedPolicy::Batch, WorkType::CpuSpin),
        (SchedPolicy::Idle, WorkType::CpuSpin),
        (
            SchedPolicy::Fifo(1),
            WorkType::Bursty {
                burst_ms: 500,
                sleep_ms: 250,
            },
        ),
    ];

    let mut ops = vec![
        Op::AddCgroup {
            name: "cg_0".into(),
        },
        Op::AddCgroup {
            name: "cg_1".into(),
        },
    ];
    for name in ["cg_0", "cg_1"] {
        for &(policy, wtype) in &configs {
            ops.push(Op::Spawn {
                cgroup: name.into(),
                workload: WorkloadConfig {
                    num_workers: 2,
                    sched_policy: policy,
                    work_type: wtype,
                    ..Default::default()
                },
            });
        }
    }

    let steps = vec![Step {
        setup: vec![].into(),
        ops,
        hold: HoldSpec::Fixed(Duration::from_millis(ctx.settle_ms) + ctx.duration),
    }];

    execute_steps(ctx, steps)
}

pub fn custom_cgroup_pipe_io(ctx: &Ctx) -> Result<AssertResult> {
    let mut ops = vec![
        Op::AddCgroup {
            name: "cg_0".into(),
        },
        Op::AddCgroup {
            name: "cg_1".into(),
        },
    ];
    for name in ["cg_0", "cg_1"] {
        ops.push(Op::Spawn {
            cgroup: name.into(),
            workload: WorkloadConfig {
                num_workers: 2,
                work_type: WorkType::PipeIo { burst_iters: 1024 },
                ..Default::default()
            },
        });
        ops.push(Op::Spawn {
            cgroup: name.into(),
            workload: WorkloadConfig {
                num_workers: ctx.workers_per_cgroup,
                ..Default::default()
            },
        });
    }

    let steps = vec![Step {
        setup: vec![].into(),
        ops,
        hold: HoldSpec::Fixed(Duration::from_millis(ctx.settle_ms) + ctx.duration),
    }];

    execute_steps(ctx, steps)
}
