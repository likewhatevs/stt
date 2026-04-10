//! Performance and benchmarking scenario implementations.

use super::Ctx;
use super::ops::{CgroupDef, CpusetSpec, HoldSpec, Step, execute_steps_with};
use crate::assert::{Assert, AssertResult};
use crate::workload::*;
use anyhow::Result;

/// CachePressure vs CpuSpin cgroups under work conservation.
///
/// One cgroup runs CachePressure workers (L1-strided RMW, cache-hot) and
/// the other runs CpuSpin workers (cache-cold). Checks throughput
/// fairness across workers (CV < 1.0) to catch gross placement imbalance.
pub fn custom_cache_pressure_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    let checks = Assert::default_checks().max_throughput_cv(1.0);

    let steps = vec![Step::with_defs(
        vec![
            CgroupDef::named("cg_0")
                .workers(ctx.workers_per_cgroup)
                .work_type(WorkType::cache_pressure(32, 64)),
            CgroupDef::named("cg_1").workers(ctx.topo.total_cpus()),
        ],
        HoldSpec::Fixed(ctx.settle + ctx.duration),
    )];

    execute_steps_with(ctx, steps, Some(&checks))
}

/// CacheYield workers testing wake-affine placement after voluntary preemption.
///
/// All workers run CacheYield (strided RMW then sched_yield). After yield,
/// the scheduler must decide where to place the waking task. Two cgroups on
/// LLC-aligned cpusets make cross-LLC migration observable. Checks wake
/// latency CV (consistent placement) and throughput fairness.
pub fn custom_cache_yield_wake_affine(ctx: &Ctx) -> Result<AssertResult> {
    if ctx.topo.num_llcs() < 2 {
        return Ok(AssertResult::skip("skipped: need >=2 LLCs"));
    }

    let checks = Assert::default_checks()
        .max_wake_latency_cv(3.0)
        .max_throughput_cv(1.0);

    let steps = vec![Step::with_defs(
        vec![
            CgroupDef::named("cg_0")
                .with_cpuset(CpusetSpec::llc(0))
                .workers(ctx.workers_per_cgroup)
                .work_type(WorkType::cache_yield(32, 64)),
            CgroupDef::named("cg_1")
                .with_cpuset(CpusetSpec::llc(1))
                .workers(ctx.workers_per_cgroup)
                .work_type(WorkType::cache_yield(32, 64)),
        ],
        HoldSpec::Fixed(ctx.settle + ctx.duration),
    )];

    execute_steps_with(ctx, steps, Some(&checks))
}

/// CachePipe vs CpuSpin cgroups under work conservation.
///
/// One cgroup runs CachePipe workers (cache-hot burst then pipe exchange,
/// combining cache pressure with cross-CPU wake placement). The other runs
/// CpuSpin at full CPU count. Checks wake latency CV to catch erratic
/// pipe wake placement.
pub fn custom_cache_pipe_io_compute_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    let n_pipe = ctx.workers_per_cgroup;
    // CachePipe requires even workers.
    let n_pipe = if !n_pipe.is_multiple_of(2) {
        n_pipe + 1
    } else {
        n_pipe
    };

    let checks = Assert::default_checks().max_wake_latency_cv(15.0);

    let steps = vec![Step::with_defs(
        vec![
            CgroupDef::named("cg_0")
                .workers(n_pipe)
                .work_type(WorkType::cache_pipe(32, 1024)),
            CgroupDef::named("cg_1").workers(ctx.topo.total_cpus()),
        ],
        HoldSpec::Fixed(ctx.settle + ctx.duration),
    )];

    execute_steps_with(ctx, steps, Some(&checks))
}

/// 1:N fan-out wake pattern (schbench-style).
///
/// One cgroup runs FutexFanOut workers: each group has 1 messenger that
/// does CPU work then wakes 4 receivers via FUTEX_WAKE. Receivers measure
/// wake-to-run latency. A second cgroup runs CpuSpin workers to create
/// CPU contention. Checks wake latency CV to catch inconsistent
/// receiver placement.
pub fn custom_fanout_wake(ctx: &Ctx) -> Result<AssertResult> {
    let fan_out = 4usize;
    let group_size = fan_out + 1;
    // Round down to nearest multiple of group_size, at least one group.
    let n_fanout = (ctx.workers_per_cgroup / group_size).max(1) * group_size;

    let checks = Assert::default_checks()
        .max_wake_latency_cv(10.0)
        .max_spread_pct(50.0);

    let steps = vec![Step::with_defs(
        vec![
            CgroupDef::named("cg_0")
                .workers(n_fanout)
                .work_type(WorkType::futex_fan_out(fan_out, 1024)),
            CgroupDef::named("cg_1").workers(ctx.topo.total_cpus()),
        ],
        HoldSpec::Fixed(ctx.settle + ctx.duration),
    )];

    execute_steps_with(ctx, steps, Some(&checks))
}
