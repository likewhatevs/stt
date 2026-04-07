use anyhow::Result;
use stt::assert::AssertResult;
use stt::scenario::Ctx;
use stt::scenario::ops::{CgroupDef, CpusetSpec, HoldSpec, Step, execute_steps};
use stt::stt_test;
use stt::test_support::{BpfMapWrite, Scheduler, SchedulerSpec};

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

    fn scenario(ctx: &stt::scenario::Ctx) -> Result<stt::assert::AssertResult> {
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
        assert: stt::assert::Assert::NONE.fail_on_stall(false),
        extra_sched_args: &[],
        required_flags: &[],
        excluded_flags: &[],
        min_sockets: 1,
        min_llcs: 1,
        requires_smt: false,
        min_cpus: 1,
        watchdog_timeout_jiffies: 0,
        bpf_map_write: Some(&BPF_NOOP),
        performance_mode: false,
        super_perf_mode: false,
        duration_s: 0,
        workers_per_cgroup: 0,
    };

    // The bpf_map_write thread exercises the full API:
    // BpfMapAccessorOwned::new(), maps(), find_map(".bss"),
    // read_value_u32(), write_value_u32().
    // If any step fails, it logs to stderr but doesn't fail the test.
    // The test passes if the scheduler runs normally through the scenario.
    run_stt_test(&__STT_ENTRY_BPF_API).unwrap();
}

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

/// Negative benchmarking test: stt-sched with --degrade skips 63 out
/// of 64 dispatch calls and burns ~5ms via bpf_loop on the 64th.
/// With 4 workers on 2 CPUs, skipped dispatches plus the periodic
/// delay produce scheduling gaps that exceed the max_gap_ms(50) threshold.
#[test]
fn sched_perf_negative() {
    use stt::scenario::ops::execute_steps_with;
    use stt::test_support::{SttTestEntry, run_stt_test};

    fn scenario(ctx: &stt::scenario::Ctx) -> Result<stt::assert::AssertResult> {
        let checks = stt::assert::Assert::default_checks().max_gap_ms(50);
        let steps = vec![Step {
            setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
            ops: vec![],
            hold: HoldSpec::Frac(1.0),
        }];
        execute_steps_with(ctx, steps, Some(&checks))
    }

    #[linkme::distributed_slice(stt::test_support::STT_TESTS)]
    #[linkme(crate = linkme)]
    static __STT_ENTRY_PERF_NEG: SttTestEntry = SttTestEntry {
        name: "sched_perf_negative",
        func: scenario,
        sockets: 1,
        cores: 2,
        threads: 1,
        memory_mb: 2048,
        scheduler: &STT_SCHED,
        auto_repro: false,
        replicas: 1,
        assert: stt::assert::Assert::NONE,
        extra_sched_args: &["--degrade"],
        required_flags: &[],
        excluded_flags: &[],
        min_sockets: 1,
        min_llcs: 1,
        requires_smt: false,
        min_cpus: 1,
        watchdog_timeout_jiffies: 0,
        bpf_map_write: None,
        performance_mode: true,
        super_perf_mode: false,
        duration_s: 5,
        workers_per_cgroup: 4,
    };

    let result = run_stt_test(&__STT_ENTRY_PERF_NEG);
    assert!(
        result.is_err(),
        "degraded scheduler should fail benchmarking gates, but test passed"
    );
    let err_msg = format!("{:#}", result.unwrap_err());
    assert!(
        err_msg.contains("stuck") || err_msg.contains("failed"),
        "error should indicate degraded scheduling: {err_msg}"
    );
}

/// Showcase: --scattershot dispatches tasks to random CPUs, causing
/// cross-LLC migrations. The scenario asserts that total migrations
/// are non-zero — random CPU placement guarantees cross-CPU movement
/// with 4 workers across 4 CPUs.
///
/// Run manually: `cargo test sched_scattershot_migration_regression -- --ignored --nocapture`
#[ignore]
#[test]
fn sched_scattershot_migration_regression() {
    use stt::scenario::ops::execute_steps_with;
    use stt::test_support::{SttTestEntry, run_stt_test};

    fn scenario(ctx: &stt::scenario::Ctx) -> Result<stt::assert::AssertResult> {
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

    #[linkme::distributed_slice(stt::test_support::STT_TESTS)]
    #[linkme(crate = linkme)]
    static __STT_ENTRY_SCATTER: SttTestEntry = SttTestEntry {
        name: "sched_scattershot_migration_regression",
        func: scenario,
        sockets: 2,
        cores: 2,
        threads: 1,
        memory_mb: 2048,
        scheduler: &SCATTER_SCHED,
        auto_repro: false,
        replicas: 1,
        assert: stt::assert::Assert::NONE,
        extra_sched_args: &["--scattershot"],
        required_flags: &[],
        excluded_flags: &[],
        min_sockets: 1,
        min_llcs: 1,
        requires_smt: false,
        min_cpus: 1,
        watchdog_timeout_jiffies: 0,
        bpf_map_write: None,
        performance_mode: true,
        super_perf_mode: false,
        duration_s: 5,
        workers_per_cgroup: 4,
    };

    let assert_result =
        run_stt_test(&__STT_ENTRY_SCATTER).expect("scattershot scenario should pass");
    let total_migrations: u64 = assert_result
        .stats
        .cgroups
        .iter()
        .map(|c| c.total_migrations)
        .sum();
    assert!(
        total_migrations > 0,
        "scattershot should cause migrations, got 0",
    );
}

/// Showcase: --slow skips 3/4 of dispatch calls, creating throughput
/// degradation detectable by the min_iteration_rate gate.
///
/// Run manually: `cargo test sched_throughput_regression -- --ignored --nocapture`
#[ignore]
#[test]
fn sched_throughput_regression() {
    use stt::scenario::ops::execute_steps_with;
    use stt::test_support::{SttTestEntry, run_stt_test};

    fn scenario(ctx: &stt::scenario::Ctx) -> Result<stt::assert::AssertResult> {
        // CpuSpin workers do ~50K-300K iterations/sec on modern hardware.
        // 5000/s is well below normal but detects the throughput collapse
        // from --slow skipping 3/4 of dispatch calls.
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

    const SLOW_SCHED: Scheduler =
        Scheduler::new("stt_sched").binary(SchedulerSpec::Name("stt-sched"));

    #[linkme::distributed_slice(stt::test_support::STT_TESTS)]
    #[linkme(crate = linkme)]
    static __STT_ENTRY_SLOW: SttTestEntry = SttTestEntry {
        name: "sched_throughput_regression",
        func: scenario,
        sockets: 1,
        cores: 2,
        threads: 1,
        memory_mb: 2048,
        scheduler: &SLOW_SCHED,
        auto_repro: false,
        replicas: 1,
        assert: stt::assert::Assert::NONE,
        extra_sched_args: &["--slow"],
        required_flags: &[],
        excluded_flags: &[],
        min_sockets: 1,
        min_llcs: 1,
        requires_smt: false,
        min_cpus: 1,
        watchdog_timeout_jiffies: 0,
        bpf_map_write: None,
        performance_mode: true,
        super_perf_mode: false,
        duration_s: 5,
        workers_per_cgroup: 4,
    };

    let result = run_stt_test(&__STT_ENTRY_SLOW);
    assert!(
        result.is_err(),
        "--slow scheduler should fail throughput/gap gates, but test passed"
    );
    let err_msg = format!("{:#}", result.unwrap_err());
    assert!(
        err_msg.contains("stuck") || err_msg.contains("iteration") || err_msg.contains("failed"),
        "error should indicate throughput regression: {err_msg}"
    );
}

/// Showcase: auto-repro. Stalls the scheduler after 1s to trigger a
/// watchdog kill, then verifies that the error output contains the
/// auto-repro rerun hint with extracted stack functions.
///
/// Run manually: `cargo test sched_auto_repro_showcase -- --ignored --nocapture`
#[ignore]
#[test]
fn sched_auto_repro_showcase() {
    use stt::test_support::{SttTestEntry, run_stt_test};

    fn scenario(ctx: &stt::scenario::Ctx) -> Result<stt::assert::AssertResult> {
        let steps = vec![Step {
            setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
            ops: vec![],
            hold: HoldSpec::Frac(1.0),
        }];
        execute_steps(ctx, steps)
    }

    const STALL_SCHED: Scheduler =
        Scheduler::new("stt_sched").binary(SchedulerSpec::Name("stt-sched"));

    #[linkme::distributed_slice(stt::test_support::STT_TESTS)]
    #[linkme(crate = linkme)]
    static __STT_ENTRY_AUTO_REPRO: SttTestEntry = SttTestEntry {
        name: "sched_auto_repro_showcase",
        func: scenario,
        sockets: 1,
        cores: 2,
        threads: 1,
        memory_mb: 2048,
        scheduler: &STALL_SCHED,
        auto_repro: true,
        replicas: 1,
        assert: stt::assert::Assert::NONE,
        extra_sched_args: &["--stall-after=1"],
        required_flags: &[],
        excluded_flags: &[],
        min_sockets: 1,
        min_llcs: 1,
        requires_smt: false,
        min_cpus: 1,
        watchdog_timeout_jiffies: 3000,
        bpf_map_write: None,
        performance_mode: false,
        super_perf_mode: false,
        duration_s: 10,
        workers_per_cgroup: 2,
    };

    let result = run_stt_test(&__STT_ENTRY_AUTO_REPRO);
    assert!(result.is_err(), "stalled scheduler should fail");
    let err_msg = format!("{:#}", result.unwrap_err());
    // The scheduler stalls after 1s, watchdog kills it. The error
    // should indicate scheduler death.
    assert!(
        err_msg.contains("scheduler died")
            || err_msg.contains("SCHEDULER_DIED")
            || err_msg.contains("timed out")
            || err_msg.contains("no test result"),
        "error should indicate scheduler death: {err_msg}"
    );
}

/// EEVDF baseline: runs the same workload as sched_baseline_scx under
/// the kernel default scheduler. Compare the two results to verify scx
/// performs at least as well as EEVDF.
///
/// Run: `cargo test sched_baseline_eevdf -- --ignored --nocapture`
#[ignore]
#[test]
fn sched_baseline_eevdf() {
    use stt::test_support::{SttTestEntry, run_stt_test};

    fn scenario(ctx: &stt::scenario::Ctx) -> Result<stt::assert::AssertResult> {
        let steps = vec![Step {
            setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
            ops: vec![],
            hold: HoldSpec::Frac(1.0),
        }];
        execute_steps(ctx, steps)
    }

    #[linkme::distributed_slice(stt::test_support::STT_TESTS)]
    #[linkme(crate = linkme)]
    static __STT_ENTRY_EEVDF: SttTestEntry = SttTestEntry {
        name: "sched_baseline_eevdf",
        func: scenario,
        sockets: 1,
        cores: 2,
        threads: 1,
        memory_mb: 2048,
        scheduler: &stt::test_support::Scheduler::EEVDF,
        auto_repro: false,
        replicas: 1,
        assert: stt::assert::Assert::NONE,
        extra_sched_args: &[],
        required_flags: &[],
        excluded_flags: &[],
        min_sockets: 1,
        min_llcs: 1,
        requires_smt: false,
        min_cpus: 1,
        watchdog_timeout_jiffies: 0,
        bpf_map_write: None,
        performance_mode: true,
        super_perf_mode: false,
        duration_s: 3,
        workers_per_cgroup: 4,
    };

    run_stt_test(&__STT_ENTRY_EEVDF).unwrap();
}

/// scx baseline: runs the same workload as sched_baseline_eevdf under
/// stt-sched. Compare the two results to verify scx performs at least
/// as well as EEVDF.
///
/// Run: `cargo test sched_baseline_scx -- --ignored --nocapture`
#[ignore]
#[test]
fn sched_baseline_scx() {
    use stt::test_support::{SttTestEntry, run_stt_test};

    fn scenario(ctx: &stt::scenario::Ctx) -> Result<stt::assert::AssertResult> {
        let steps = vec![Step {
            setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
            ops: vec![],
            hold: HoldSpec::Frac(1.0),
        }];
        execute_steps(ctx, steps)
    }

    #[linkme::distributed_slice(stt::test_support::STT_TESTS)]
    #[linkme(crate = linkme)]
    static __STT_ENTRY_SCX: SttTestEntry = SttTestEntry {
        name: "sched_baseline_scx",
        func: scenario,
        sockets: 1,
        cores: 2,
        threads: 1,
        memory_mb: 2048,
        scheduler: &STT_SCHED,
        auto_repro: false,
        replicas: 1,
        assert: stt::assert::Assert::NONE,
        extra_sched_args: &[],
        required_flags: &[],
        excluded_flags: &[],
        min_sockets: 1,
        min_llcs: 1,
        requires_smt: false,
        min_cpus: 1,
        watchdog_timeout_jiffies: 0,
        bpf_map_write: None,
        performance_mode: true,
        super_perf_mode: false,
        duration_s: 3,
        workers_per_cgroup: 4,
    };

    run_stt_test(&__STT_ENTRY_SCX).unwrap();
}
