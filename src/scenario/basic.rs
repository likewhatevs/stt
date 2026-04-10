//! Basic and mixed-workload scenario implementations.

use super::Ctx;
use super::ops::{CgroupDef, HoldSpec, Op, Step, execute_steps};
use crate::assert::AssertResult;
use crate::workload::*;
use anyhow::Result;

/// Two managed cgroups with host-level contention from workers in the
/// parent cgroup. Spawns `total_cpus` workers outside any managed cgroup
/// alongside two default cgroups.
pub fn custom_host_cgroup_contention(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![
        Step::with_defs(
            vec![CgroupDef::named("cg_0"), CgroupDef::named("cg_1")],
            HoldSpec::Fixed(ctx.settle + ctx.duration),
        )
        .with_ops(vec![Op::spawn_host(
            Work::default().workers(ctx.topo.total_cpus()),
        )]),
    ];

    execute_steps(ctx, steps)
}

/// Two cgroups each running Normal, Batch, Idle, and FIFO(1) workers
/// concurrently. FIFO workers use bursty workloads to avoid monopolizing
/// CPUs.
pub fn custom_sched_mixed(ctx: &Ctx) -> Result<AssertResult> {
    let configs = [
        (SchedPolicy::Normal, WorkType::CpuSpin),
        (SchedPolicy::Batch, WorkType::CpuSpin),
        (SchedPolicy::Idle, WorkType::CpuSpin),
        (SchedPolicy::Fifo(1), WorkType::bursty(500, 250)),
    ];

    let mut ops = vec![Op::add_cgroup("cg_0"), Op::add_cgroup("cg_1")];
    for name in ["cg_0", "cg_1"] {
        for &(policy, ref wtype) in &configs {
            ops.push(Op::spawn(
                name,
                Work::default()
                    .workers(2)
                    .sched_policy(policy)
                    .work_type(wtype.clone()),
            ));
        }
    }

    let steps = vec![Step::new(ops, HoldSpec::Fixed(ctx.settle + ctx.duration))];

    execute_steps(ctx, steps)
}

/// Two cgroups each with paired PipeIo workers and CpuSpin workers.
/// Exercises cross-CPU wake placement from pipe I/O under CPU load.
pub fn custom_cgroup_pipe_io(ctx: &Ctx) -> Result<AssertResult> {
    let mut ops = vec![Op::add_cgroup("cg_0"), Op::add_cgroup("cg_1")];
    for name in ["cg_0", "cg_1"] {
        ops.push(Op::spawn(
            name,
            Work::default()
                .workers(2)
                .work_type(WorkType::pipe_io(1024)),
        ));
        ops.push(Op::spawn(
            name,
            Work::default().workers(ctx.workers_per_cgroup),
        ));
    }

    let steps = vec![Step::new(ops, HoldSpec::Fixed(ctx.settle + ctx.duration))];

    execute_steps(ctx, steps)
}
