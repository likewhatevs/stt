use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{CgroupDef, CpusetSpec, HoldSpec, Step, execute_steps};
use ktstr::test_support::{BpfMapWrite, Payload, Scheduler, SchedulerSpec};

const KTSTR_SCHED: Scheduler =
    Scheduler::new("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));
const KTSTR_SCHED_PAYLOAD: Payload = Payload::from_scheduler(&KTSTR_SCHED);

#[ktstr_test(scheduler = KTSTR_SCHED_PAYLOAD, llcs = 1, cores = 2, threads = 1, sustained_samples = 15)]
fn sched_basic_proportional(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup),
            CgroupDef::named("cg_1").workers(ctx.workers_per_cgroup),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}

#[ktstr_test(scheduler = KTSTR_SCHED_PAYLOAD, llcs = 1, cores = 4, threads = 1, sustained_samples = 15)]
fn sched_cpuset_split(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cg_0").with_cpuset(CpusetSpec::Disjoint { index: 0, of: 2 }),
            CgroupDef::named("cg_1").with_cpuset(CpusetSpec::Disjoint { index: 1, of: 2 }),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}

#[ktstr_test(scheduler = KTSTR_SCHED_PAYLOAD, llcs = 1, cores = 2, threads = 1, sustained_samples = 15)]
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

fn scenario_bpf_api(ctx: &ktstr::scenario::Ctx) -> Result<ktstr::assert::AssertResult> {
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
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

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_BPF_API: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "sched_bpf_map_api_integration",
        func: scenario_bpf_api,
        scheduler: &KTSTR_SCHED_PAYLOAD,
        auto_repro: false,
        assert: ktstr::assert::Assert::NO_OVERRIDES.fail_on_stall(false),
        bpf_map_write: &[&BPF_NOOP],
        duration: std::time::Duration::from_secs(10),
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

// Host-to-guest observable-action integration test.
//
// Exercises the full loop:
//   1. HOST writes into the guest's BPF `.bss` map via the KVM
//      memslot path (`BpfMapAccessor::write_value_u32` dispatched
//      from `vmm::bpf_map_write_thread`).
//   2. GUEST scheduler's BPF dispatcher reads the new `stall` value
//      from its `.bss` section on the next dispatch entry (line 93
//      of main.bpf.c: `if (stall) return;`).
//   3. GUEST ACTS: the scheduler stops inserting tasks into DSQs,
//      every CPU sits idle, the scx watchdog observes no progress
//      within its budget and tears the scheduler down (emitted
//      via the `SchedulerDied` assert detail the runtime records).
//   4. HOST CONFIRMS: the scenario returns a failing AssertResult
//      carrying the scheduler-died signal; `expect_err: true`
//      inverts the verdict so "fails as expected" is the PASS
//      state.
//
// Differs from the existing BPF-NOOP test (value=0 over a field
// already 0) — that proves the API pipeline LINKS, this proves
// the pipeline's WRITE is OBSERVED by the guest and produces
// distinct guest behaviour. Differs from `cover_watchdog_forced_stall`
// which achieves the same stall via the scheduler's
// `--stall-after` CLI flag (a scheduler-internal timer, no host
// write): that path tests the scheduler's self-stall plumbing,
// this path tests the host→guest map-write plumbing.
//
// `watchdog_timeout` is set short (2 s) so the stall-detection
// fires quickly; `duration` is longer so the watchdog has room
// to fire inside the scenario window rather than racing the
// natural scenario end.
static BPF_STALL_HOST_WRITE: BpfMapWrite = BpfMapWrite {
    map_name_suffix: ".bss",
    offset: 0, // stall @ main.bpf.c line 14
    value: 1,
};

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_BPF_HOST_WRITE_STALLS: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "sched_host_bpf_map_write_stalls_scheduler",
        func: scenario_bpf_api,
        scheduler: &KTSTR_SCHED_PAYLOAD,
        auto_repro: false,
        bpf_map_write: &[&BPF_STALL_HOST_WRITE],
        watchdog_timeout: std::time::Duration::from_secs(2),
        duration: std::time::Duration::from_secs(10),
        performance_mode: true,
        expect_err: true,
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

/// Positive benchmarking test: scx-ktstr under performance_mode passes
/// min_iteration_rate and max_gap_ms gates.
#[ktstr_test(
    scheduler = KTSTR_SCHED_PAYLOAD,
    llcs = 1,
    cores = 2,
    threads = 1,
    performance_mode = true,
    duration_s = 3,
    workers_per_cgroup = 2,
    sustained_samples = 15,
)]
fn sched_perf_positive(ctx: &Ctx) -> Result<AssertResult> {
    use ktstr::scenario::ops::execute_steps_with;
    let checks = ktstr::assert::Assert::default_checks()
        .min_iteration_rate(5000.0)
        .max_gap_ms(500);
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps_with(ctx, steps, Some(&checks))
}

fn scenario_perf_negative(ctx: &ktstr::scenario::Ctx) -> Result<ktstr::assert::AssertResult> {
    use ktstr::scenario::ops::execute_steps_with;
    let checks = ktstr::assert::Assert::default_checks().max_gap_ms(50);
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps_with(ctx, steps, Some(&checks))
}

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_PERF_NEG: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "sched_perf_negative",
        func: scenario_perf_negative,
        scheduler: &KTSTR_SCHED_PAYLOAD,
        auto_repro: false,
        extra_sched_args: &["--degrade"],
        performance_mode: true,
        duration: std::time::Duration::from_secs(5),
        workers_per_cgroup: 4,
        expect_err: true,
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

fn scenario_scattershot(ctx: &ktstr::scenario::Ctx) -> Result<ktstr::assert::AssertResult> {
    use ktstr::scenario::ops::execute_steps_with;
    let checks = ktstr::assert::Assert::default_checks()
        .max_gap_ms(10000)
        .max_spread_pct(80.0);
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps_with(ctx, steps, Some(&checks))
}

const SCATTER_SCHED: Scheduler =
    Scheduler::new("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));
const SCATTER_SCHED_PAYLOAD: Payload = Payload::from_scheduler(&SCATTER_SCHED);

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_SCATTER: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "demo_scattershot_migration",
        func: scenario_scattershot,
        topology: ktstr::test_support::Topology {
            llcs: 2,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        },
        scheduler: &SCATTER_SCHED_PAYLOAD,
        extra_sched_args: &["--scattershot"],
        performance_mode: true,
        duration: std::time::Duration::from_secs(5),
        workers_per_cgroup: 4,
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

fn scenario_throughput_regression(
    ctx: &ktstr::scenario::Ctx,
) -> Result<ktstr::assert::AssertResult> {
    use ktstr::scenario::ops::execute_steps_with;
    let checks = ktstr::assert::Assert::default_checks()
        .min_iteration_rate(5000.0)
        .max_gap_ms(500);
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps_with(ctx, steps, Some(&checks))
}

const SLOW_SCHED: Scheduler =
    Scheduler::new("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));
const SLOW_SCHED_PAYLOAD: Payload = Payload::from_scheduler(&SLOW_SCHED);

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_SLOW: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "demo_throughput_regression",
        func: scenario_throughput_regression,
        scheduler: &SLOW_SCHED_PAYLOAD,
        extra_sched_args: &["--slow"],
        performance_mode: true,
        duration: std::time::Duration::from_secs(5),
        workers_per_cgroup: 4,
        expect_err: true,
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

fn scenario_auto_repro(ctx: &ktstr::scenario::Ctx) -> Result<ktstr::assert::AssertResult> {
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}

const STALL_SCHED: Scheduler =
    Scheduler::new("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));
const STALL_SCHED_PAYLOAD: Payload = Payload::from_scheduler(&STALL_SCHED);

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_AUTO_REPRO: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "demo_auto_repro",
        func: scenario_auto_repro,
        scheduler: &STALL_SCHED_PAYLOAD,
        extra_sched_args: &["--stall-after=1"],
        watchdog_timeout: std::time::Duration::from_secs(3),
        duration: std::time::Duration::from_secs(10),
        workers_per_cgroup: 2,
        expect_err: true,
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

fn scenario_baseline(ctx: &ktstr::scenario::Ctx) -> Result<ktstr::assert::AssertResult> {
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_EEVDF: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "demo_baseline_eevdf",
        func: scenario_baseline,
        auto_repro: false,
        performance_mode: true,
        duration: std::time::Duration::from_secs(3),
        workers_per_cgroup: 4,
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_SCX: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "demo_baseline_scx",
        func: scenario_baseline,
        scheduler: &KTSTR_SCHED_PAYLOAD,
        performance_mode: true,
        duration: std::time::Duration::from_secs(3),
        workers_per_cgroup: 4,
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

/// Minimal scheduler test that exercises host-side BPF program enumeration.
/// The framework warns when verifier_stats is empty for scheduler tests.
#[ktstr_test(scheduler = KTSTR_SCHED_PAYLOAD, llcs = 1, cores = 2, threads = 1, duration_s = 3)]
fn sched_verifier_stats_populated(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}

fn scenario_mid_degrade(ctx: &ktstr::scenario::Ctx) -> Result<ktstr::assert::AssertResult> {
    use ktstr::scenario::ops::execute_steps_with;
    let checks = ktstr::assert::Assert::default_checks().max_gap_ms(50);
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

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_MID_DEGRADE: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "demo_mid_run_degrade",
        func: scenario_mid_degrade,
        scheduler: &KTSTR_SCHED_PAYLOAD,
        extra_sched_args: &["--degrade-after=3"],
        performance_mode: true,
        duration: std::time::Duration::from_secs(10),
        workers_per_cgroup: 4,
        watchdog_timeout: std::time::Duration::from_secs(60),
        expect_err: true,
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };
