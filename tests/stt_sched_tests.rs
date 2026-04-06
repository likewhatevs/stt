use anyhow::Result;
use stt::scenario::Ctx;
use stt::scenario::ops::{CgroupDef, CpusetSpec, HoldSpec, Step, execute_steps};
use stt::stt_test;
use stt::test_support::{Scheduler, SchedulerSpec};
use stt::verify::VerifyResult;

const STT_SCHED: Scheduler = Scheduler::new("stt_sched").binary(SchedulerSpec::Name("stt-sched"));

#[stt_test(scheduler = STT_SCHED, sockets = 1, cores = 2, threads = 1)]
fn sched_basic_proportional(ctx: &Ctx) -> Result<VerifyResult> {
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cell_0").workers(ctx.workers_per_cell),
            CgroupDef::named("cell_1").workers(ctx.workers_per_cell),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::Frac(1.0),
    }];
    execute_steps(ctx, steps)
}

#[stt_test(scheduler = STT_SCHED, sockets = 1, cores = 4, threads = 1)]
fn sched_cpuset_split(ctx: &Ctx) -> Result<VerifyResult> {
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cell_0").with_cpuset(CpusetSpec::Disjoint { index: 0, of: 2 }),
            CgroupDef::named("cell_1").with_cpuset(CpusetSpec::Disjoint { index: 1, of: 2 }),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::Frac(1.0),
    }];
    execute_steps(ctx, steps)
}

#[stt_test(scheduler = STT_SCHED, sockets = 1, cores = 2, threads = 1)]
fn sched_dynamic_add(ctx: &Ctx) -> Result<VerifyResult> {
    let steps = vec![
        Step {
            setup: vec![CgroupDef::named("cell_0")].into(),
            ops: vec![],
            hold: HoldSpec::Frac(0.5),
        },
        Step {
            setup: vec![CgroupDef::named("cell_1")].into(),
            ops: vec![],
            hold: HoldSpec::Frac(0.5),
        },
    ];
    execute_steps(ctx, steps)
}
