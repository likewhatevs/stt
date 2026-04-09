use anyhow::Result;
use stt::assert::AssertResult;
use stt::scenario::Ctx;
use stt::scenario::ops::{CgroupDef, CpusetSpec, HoldSpec, Step, execute_steps};
use stt::stt_test;
use stt::test_support::{BpfMapWrite, Scheduler, SchedulerSpec};

fn main() {
    if stt::test_support::is_pid1() {
        stt::test_support::stt_guest_init();
    }
    let args = libtest_mimic::Arguments::from_args();
    let trials = stt::test_support::build_stt_trials();
    let conclusion = libtest_mimic::run(&args, trials);
    stt::test_support::collect_and_print_sidecar_stats();
    conclusion.exit();
}

const STT_SCHED: Scheduler = Scheduler::new("stt_sched").binary(SchedulerSpec::Name("stt-sched"));

#[stt_test(scheduler = STT_SCHED, sockets = 1, cores = 2, threads = 1, sustained_samples = 15)]
fn sched_basic_proportional(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup),
            CgroupDef::named("cg_1").workers(ctx.workers_per_cgroup),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::Frac(1.0),
    }];
    execute_steps(ctx, steps)
}

#[stt_test(scheduler = STT_SCHED, sockets = 1, cores = 4, threads = 1, sustained_samples = 15)]
fn sched_cpuset_split(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cg_0").with_cpuset(CpusetSpec::Disjoint { index: 0, of: 2 }),
            CgroupDef::named("cg_1").with_cpuset(CpusetSpec::Disjoint { index: 1, of: 2 }),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::Frac(1.0),
    }];
    execute_steps(ctx, steps)
}

#[stt_test(scheduler = STT_SCHED, sockets = 1, cores = 2, threads = 1, sustained_samples = 15)]
fn sched_dynamic_add(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![
        Step {
            setup: vec![CgroupDef::named("cg_0")].into(),
            ops: vec![],
            hold: HoldSpec::Frac(0.5),
        },
        Step {
            setup: vec![CgroupDef::named("cg_1")].into(),
            ops: vec![],
            hold: HoldSpec::Frac(0.5),
        },
    ];
    execute_steps(ctx, steps)
}

fn scenario_bpf_api(ctx: &stt::scenario::Ctx) -> Result<stt::assert::AssertResult> {
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
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

#[stt::__linkme::distributed_slice(stt::test_support::STT_TESTS)]
#[linkme(crate = stt::__linkme)]
static __STT_ENTRY_BPF_API: stt::test_support::SttTestEntry = stt::test_support::SttTestEntry {
    name: "sched_bpf_map_api_integration",
    func: scenario_bpf_api,
    scheduler: &STT_SCHED,
    auto_repro: false,
    assert: stt::assert::Assert::NONE.fail_on_stall(false),
    bpf_map_write: Some(&BPF_NOOP),
    duration_s: 10,
    ..stt::test_support::SttTestEntry::DEFAULT
};

/// Positive benchmarking test: stt-sched under performance_mode passes
/// min_iteration_rate and max_gap_ms gates.
#[stt_test(
    scheduler = STT_SCHED,
    sockets = 1,
    cores = 2,
    threads = 1,
    performance_mode = true,
    duration_s = 3,
    workers_per_cgroup = 2,
    sustained_samples = 15,
)]
fn sched_perf_positive(ctx: &Ctx) -> Result<AssertResult> {
    use stt::scenario::ops::execute_steps_with;
    let checks = stt::assert::Assert::default_checks()
        .min_iteration_rate(5000.0)
        .max_gap_ms(500);
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::Frac(1.0),
    }];
    execute_steps_with(ctx, steps, Some(&checks))
}

fn scenario_perf_negative(ctx: &stt::scenario::Ctx) -> Result<stt::assert::AssertResult> {
    use stt::scenario::ops::execute_steps_with;
    let checks = stt::assert::Assert::default_checks().max_gap_ms(50);
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::Frac(1.0),
    }];
    execute_steps_with(ctx, steps, Some(&checks))
}

#[stt::__linkme::distributed_slice(stt::test_support::STT_TESTS)]
#[linkme(crate = stt::__linkme)]
static __STT_ENTRY_PERF_NEG: stt::test_support::SttTestEntry = stt::test_support::SttTestEntry {
    name: "sched_perf_negative",
    func: scenario_perf_negative,
    scheduler: &STT_SCHED,
    auto_repro: false,
    extra_sched_args: &["--degrade"],
    performance_mode: true,
    duration_s: 5,
    workers_per_cgroup: 4,
    expect_err: true,
    ..stt::test_support::SttTestEntry::DEFAULT
};

fn scenario_scattershot(ctx: &stt::scenario::Ctx) -> Result<stt::assert::AssertResult> {
    use stt::scenario::ops::execute_steps_with;
    let checks = stt::assert::Assert::default_checks()
        .max_gap_ms(10000)
        .max_spread_pct(80.0);
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::Frac(1.0),
    }];
    execute_steps_with(ctx, steps, Some(&checks))
}

const SCATTER_SCHED: Scheduler =
    Scheduler::new("stt_sched").binary(SchedulerSpec::Name("stt-sched"));

#[stt::__linkme::distributed_slice(stt::test_support::STT_TESTS)]
#[linkme(crate = stt::__linkme)]
static __STT_ENTRY_SCATTER: stt::test_support::SttTestEntry = stt::test_support::SttTestEntry {
    name: "demo_scattershot_migration",
    func: scenario_scattershot,
    topology: stt::test_support::Topology {
        sockets: 2,
        cores_per_socket: 2,
        threads_per_core: 1,
    },
    scheduler: &SCATTER_SCHED,
    extra_sched_args: &["--scattershot"],
    performance_mode: true,
    duration_s: 5,
    workers_per_cgroup: 4,
    ..stt::test_support::SttTestEntry::DEFAULT
};

fn scenario_throughput_regression(ctx: &stt::scenario::Ctx) -> Result<stt::assert::AssertResult> {
    use stt::scenario::ops::execute_steps_with;
    let checks = stt::assert::Assert::default_checks()
        .min_iteration_rate(5000.0)
        .max_gap_ms(500);
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::Frac(1.0),
    }];
    execute_steps_with(ctx, steps, Some(&checks))
}

const SLOW_SCHED: Scheduler = Scheduler::new("stt_sched").binary(SchedulerSpec::Name("stt-sched"));

#[stt::__linkme::distributed_slice(stt::test_support::STT_TESTS)]
#[linkme(crate = stt::__linkme)]
static __STT_ENTRY_SLOW: stt::test_support::SttTestEntry = stt::test_support::SttTestEntry {
    name: "demo_throughput_regression",
    func: scenario_throughput_regression,
    scheduler: &SLOW_SCHED,
    extra_sched_args: &["--slow"],
    performance_mode: true,
    duration_s: 5,
    workers_per_cgroup: 4,
    expect_err: true,
    ..stt::test_support::SttTestEntry::DEFAULT
};

fn scenario_auto_repro(ctx: &stt::scenario::Ctx) -> Result<stt::assert::AssertResult> {
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::Frac(1.0),
    }];
    execute_steps(ctx, steps)
}

const STALL_SCHED: Scheduler = Scheduler::new("stt_sched").binary(SchedulerSpec::Name("stt-sched"));

#[stt::__linkme::distributed_slice(stt::test_support::STT_TESTS)]
#[linkme(crate = stt::__linkme)]
static __STT_ENTRY_AUTO_REPRO: stt::test_support::SttTestEntry = stt::test_support::SttTestEntry {
    name: "demo_auto_repro",
    func: scenario_auto_repro,
    scheduler: &STALL_SCHED,
    extra_sched_args: &["--stall-after=1"],
    watchdog_timeout_s: 3,
    duration_s: 10,
    workers_per_cgroup: 2,
    expect_err: true,
    ..stt::test_support::SttTestEntry::DEFAULT
};

fn scenario_baseline(ctx: &stt::scenario::Ctx) -> Result<stt::assert::AssertResult> {
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::Frac(1.0),
    }];
    execute_steps(ctx, steps)
}

#[stt::__linkme::distributed_slice(stt::test_support::STT_TESTS)]
#[linkme(crate = stt::__linkme)]
static __STT_ENTRY_EEVDF: stt::test_support::SttTestEntry = stt::test_support::SttTestEntry {
    name: "demo_baseline_eevdf",
    func: scenario_baseline,
    auto_repro: false,
    performance_mode: true,
    duration_s: 3,
    workers_per_cgroup: 4,
    ..stt::test_support::SttTestEntry::DEFAULT
};

#[stt::__linkme::distributed_slice(stt::test_support::STT_TESTS)]
#[linkme(crate = stt::__linkme)]
static __STT_ENTRY_SCX: stt::test_support::SttTestEntry = stt::test_support::SttTestEntry {
    name: "demo_baseline_scx",
    func: scenario_baseline,
    scheduler: &STT_SCHED,
    performance_mode: true,
    duration_s: 3,
    workers_per_cgroup: 4,
    ..stt::test_support::SttTestEntry::DEFAULT
};

/// Minimal scheduler test that exercises host-side BPF program enumeration.
/// The framework warns when verifier_stats is empty for scheduler tests.
#[stt_test(scheduler = STT_SCHED, sockets = 1, cores = 2, threads = 1, duration_s = 3)]
fn sched_verifier_stats_populated(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::Frac(1.0),
    }];
    execute_steps(ctx, steps)
}

fn scenario_mid_degrade(ctx: &stt::scenario::Ctx) -> Result<stt::assert::AssertResult> {
    use stt::scenario::ops::execute_steps_with;
    let checks = stt::assert::Assert::default_checks().max_gap_ms(50);
    let steps = vec![
        Step {
            setup: vec![
                CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup),
                CgroupDef::named("cg_1").workers(ctx.workers_per_cgroup),
            ]
            .into(),
            ops: vec![],
            hold: HoldSpec::Fixed(std::time::Duration::from_secs(3)),
        },
        Step {
            setup: vec![].into(),
            ops: vec![],
            hold: HoldSpec::Fixed(std::time::Duration::from_secs(5)),
        },
    ];
    execute_steps_with(ctx, steps, Some(&checks))
}

#[stt::__linkme::distributed_slice(stt::test_support::STT_TESTS)]
#[linkme(crate = stt::__linkme)]
static __STT_ENTRY_MID_DEGRADE: stt::test_support::SttTestEntry = stt::test_support::SttTestEntry {
    name: "demo_mid_run_degrade",
    func: scenario_mid_degrade,
    scheduler: &STT_SCHED,
    extra_sched_args: &["--degrade-after=3"],
    performance_mode: true,
    duration_s: 10,
    workers_per_cgroup: 4,
    watchdog_timeout_s: 60,
    expect_err: true,
    ..stt::test_support::SttTestEntry::DEFAULT
};
