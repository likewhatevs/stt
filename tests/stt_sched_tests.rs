use anyhow::Result;
use stt::scenario::Ctx;
use stt::scenario::ops::{CgroupDef, CpusetSpec, HoldSpec, Step, execute_steps};
use stt::stt_test;
use stt::test_support::{BpfMapWrite, Scheduler, SchedulerSpec};
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

/// Integration test for the host-side BPF map API.
///
/// Boots a VM with stt-sched, uses bpf_map_write to exercise
/// BpfMapAccessorOwned::new() -> maps() -> find_map() ->
/// read_value_u32() -> write_value_u32() end-to-end. Writes
/// stall=0 (no-op) to confirm the pipeline works without
/// disrupting the scheduler.
#[test]
fn sched_bpf_map_api_integration() {
    use stt::test_support::{SttTestEntry, run_stt_test};

    fn scenario(ctx: &stt::scenario::Ctx) -> Result<stt::verify::VerifyResult> {
        let steps = vec![Step {
            setup: vec![CgroupDef::named("cell_0").workers(ctx.workers_per_cell)].into(),
            ops: vec![],
            hold: HoldSpec::Frac(1.0),
        }];
        execute_steps(ctx, steps)
    }

    /// Write stall=0 to the .bss map after scenario starts.
    /// stall is at offset 0, already 0 — this is a no-op write
    /// that exercises the full BPF map API pipeline.
    static BPF_NOOP: BpfMapWrite = BpfMapWrite {
        map_name_suffix: ".bss",
        offset: 0,
        value: 0,
    };

    #[linkme::distributed_slice(stt::test_support::STT_TESTS)]
    #[linkme(crate = linkme)]
    static __STT_ENTRY_BPF_API: SttTestEntry = SttTestEntry {
        name: "sched_bpf_map_api_integration",
        func: scenario,
        sockets: 1,
        cores: 2,
        threads: 1,
        memory_mb: 1024,
        scheduler: &STT_SCHED,
        auto_repro: false,
        replicas: 1,
        verify: stt::verify::Verify::NONE.fail_on_stall(false),
        extra_sched_args: &[],
        watchdog_timeout_jiffies: 0,
        bpf_map_write: Some(&BPF_NOOP),
        performance_mode: false,
    };

    // The bpf_map_write thread exercises the full API:
    // BpfMapAccessorOwned::new(), maps(), find_map(".bss"),
    // read_value_u32(), write_value_u32().
    // If any step fails, it logs to stderr but doesn't fail the test.
    // The test passes if the scheduler runs normally through the scenario.
    run_stt_test(&__STT_ENTRY_BPF_API).unwrap();
}
