//! Basic and mixed-workload scenario implementations.

use super::Ctx;
use super::ops::{CgroupDef, HoldSpec, Op, Step, execute_steps};
use crate::assert::AssertResult;
use crate::workload::*;
use anyhow::Result;

fn host_cgroup_contention_steps(ctx: &Ctx) -> Vec<Step> {
    vec![
        Step::with_defs(
            vec![CgroupDef::named("cg_0"), CgroupDef::named("cg_1")],
            HoldSpec::Fixed(ctx.settle + ctx.duration),
        )
        .with_ops(vec![Op::spawn_host(
            Work::default().workers(ctx.topo.total_cpus()),
        )]),
    ]
}

/// Two managed cgroups with host-level contention from workers in the
/// parent cgroup. Spawns `total_cpus` workers outside any managed cgroup
/// alongside two default cgroups.
pub fn custom_host_cgroup_contention(ctx: &Ctx) -> Result<AssertResult> {
    execute_steps(ctx, host_cgroup_contention_steps(ctx))
}

fn sched_mixed_steps(ctx: &Ctx) -> Vec<Step> {
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

    vec![Step::new(ops, HoldSpec::Fixed(ctx.settle + ctx.duration))]
}

/// Two cgroups each running Normal, Batch, Idle, and FIFO(1) workers
/// concurrently. FIFO workers use bursty workloads to avoid monopolizing
/// CPUs.
pub fn custom_sched_mixed(ctx: &Ctx) -> Result<AssertResult> {
    execute_steps(ctx, sched_mixed_steps(ctx))
}

fn cgroup_pipe_io_steps(ctx: &Ctx) -> Vec<Step> {
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

    vec![Step::new(ops, HoldSpec::Fixed(ctx.settle + ctx.duration))]
}

/// Two cgroups each with paired PipeIo workers and CpuSpin workers.
/// Exercises cross-CPU wake placement from pipe I/O under CPU load.
pub fn custom_cgroup_pipe_io(ctx: &Ctx) -> Result<AssertResult> {
    execute_steps(ctx, cgroup_pipe_io_steps(ctx))
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
            duration: Duration::from_secs(1),
            workers_per_cgroup: 3,
            sched_pid: 1,
            settle: Duration::from_millis(100),
            work_type_override: None,
            assert: crate::assert::Assert::default_checks(),
            wait_for_map_write: false,
        }
    }

    fn def_names(step: &Step) -> Vec<String> {
        match &step.setup {
            Setup::Defs(defs) => defs.iter().map(|d| d.name.to_string()).collect(),
            Setup::Factory(_) => Vec::new(),
        }
    }

    #[test]
    fn host_cgroup_contention_builds_two_defs_and_host_spawn() {
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::from_spec(1, 1, 4, 1);
        let ctx = ctx_for_test(&cgroups, &topo);

        let steps = host_cgroup_contention_steps(&ctx);
        assert_eq!(steps.len(), 1);
        assert_eq!(def_names(&steps[0]), ["cg_0", "cg_1"]);
        assert_eq!(steps[0].ops.len(), 1);
        match &steps[0].ops[0] {
            Op::SpawnHost { work } => {
                assert_eq!(work.num_workers, Some(topo.total_cpus()));
            }
            other => panic!("expected SpawnHost, got {other:?}"),
        }
    }

    #[test]
    fn sched_mixed_builds_two_add_cgroups_and_eight_spawns() {
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::from_spec(1, 1, 4, 1);
        let ctx = ctx_for_test(&cgroups, &topo);

        let steps = sched_mixed_steps(&ctx);
        assert_eq!(steps.len(), 1);
        let ops = &steps[0].ops;
        let adds = ops
            .iter()
            .filter(|o| matches!(o, Op::AddCgroup { .. }))
            .count();
        let spawns = ops.iter().filter(|o| matches!(o, Op::Spawn { .. })).count();
        assert_eq!(adds, 2, "two cgroups added");
        assert_eq!(spawns, 8, "4 policies × 2 cgroups = 8 spawns");
        for op in ops {
            if let Op::Spawn { work, .. } = op {
                assert_eq!(work.num_workers, Some(2));
            }
        }
    }

    #[test]
    fn cgroup_pipe_io_spawn_counts_follow_workers_per_cgroup() {
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::from_spec(1, 1, 4, 1);
        let ctx = ctx_for_test(&cgroups, &topo);

        let steps = cgroup_pipe_io_steps(&ctx);
        assert_eq!(steps.len(), 1);
        let ops = &steps[0].ops;
        let spawns: Vec<_> = ops
            .iter()
            .filter_map(|o| match o {
                Op::Spawn { cgroup, work } => Some((cgroup.to_string(), work.num_workers)),
                _ => None,
            })
            .collect();
        assert_eq!(spawns.len(), 4, "pipe_io spawn + cpuspin spawn per cgroup");
        let cpuspin_workers: Vec<_> = spawns
            .iter()
            .filter(|(_, n)| *n == Some(ctx.workers_per_cgroup))
            .collect();
        assert_eq!(cpuspin_workers.len(), 2);
        let pipe_workers: Vec<_> = spawns.iter().filter(|(_, n)| *n == Some(2)).collect();
        assert_eq!(pipe_workers.len(), 2);
    }
}
