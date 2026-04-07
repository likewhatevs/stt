use anyhow::Result;
use stt::assert::{Assert, AssertResult};
use stt::scenario::Ctx;
use stt::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps_with};
use stt::test_support::{Scheduler, SchedulerSpec, SttTestEntry, run_stt_test};

const STT_SCHED: Scheduler = Scheduler::new("stt_sched").binary(SchedulerSpec::Name("stt-sched"));

fn scenario_with_checks(ctx: &Ctx, checks: &Assert) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::Frac(1.0),
    }];
    execute_steps_with(ctx, steps, Some(checks))
}

// ---------------------------------------------------------------------------
// Macro for perf-mode positive tests (normal scheduler, threshold should pass)
// ---------------------------------------------------------------------------

macro_rules! perf_positive_test {
    ($name:ident, $checks:expr) => {
        #[ignore]
        #[test]
        fn $name() {
            fn scenario(ctx: &Ctx) -> Result<AssertResult> {
                let checks = $checks;
                scenario_with_checks(ctx, &checks)
            }

            #[linkme::distributed_slice(stt::test_support::STT_TESTS)]
            #[linkme(crate = linkme)]
            static __STT_ENTRY: SttTestEntry = SttTestEntry {
                name: stringify!($name),
                func: scenario,
                sockets: 1,
                cores: 2,
                threads: 1,
                memory_mb: 2048,
                scheduler: &STT_SCHED,
                auto_repro: false,
                replicas: 1,
                assert: Assert::NONE,
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
                duration_s: 5,
                workers_per_cgroup: 2,
            };

            let result = run_stt_test(&__STT_ENTRY);
            assert!(
                result.is_ok(),
                "normal scheduler should pass {} gate: {:#}",
                stringify!($name),
                result.unwrap_err()
            );
        }
    };
}

// ---------------------------------------------------------------------------
// Macro for perf-mode negative tests (--degrade scheduler, tight threshold)
// ---------------------------------------------------------------------------

macro_rules! perf_negative_test {
    ($name:ident, $checks:expr, $err_pattern:expr) => {
        #[ignore]
        #[test]
        fn $name() {
            fn scenario(ctx: &Ctx) -> Result<AssertResult> {
                let checks = $checks;
                scenario_with_checks(ctx, &checks)
            }

            #[linkme::distributed_slice(stt::test_support::STT_TESTS)]
            #[linkme(crate = linkme)]
            static __STT_ENTRY: SttTestEntry = SttTestEntry {
                name: stringify!($name),
                func: scenario,
                sockets: 1,
                cores: 2,
                threads: 1,
                memory_mb: 2048,
                scheduler: &STT_SCHED,
                auto_repro: false,
                replicas: 1,
                assert: Assert::NONE,
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

            let result = run_stt_test(&__STT_ENTRY);
            assert!(
                result.is_err(),
                "degraded scheduler should fail {} gate",
                stringify!($name),
            );
            let err_msg = format!("{:#}", result.unwrap_err());
            assert!(
                err_msg.contains($err_pattern),
                "{} error should contain '{}': {err_msg}",
                stringify!($name),
                $err_pattern,
            );
        }
    };
}

// ---------------------------------------------------------------------------
// Macro for no-perf positive tests (normal scheduler, threshold should pass)
// ---------------------------------------------------------------------------

macro_rules! noperf_positive_test {
    ($name:ident, $checks:expr) => {
        #[ignore]
        #[test]
        fn $name() {
            fn scenario(ctx: &Ctx) -> Result<AssertResult> {
                let checks = $checks;
                scenario_with_checks(ctx, &checks)
            }

            #[linkme::distributed_slice(stt::test_support::STT_TESTS)]
            #[linkme(crate = linkme)]
            static __STT_ENTRY: SttTestEntry = SttTestEntry {
                name: stringify!($name),
                func: scenario,
                sockets: 1,
                cores: 2,
                threads: 1,
                memory_mb: 2048,
                scheduler: &STT_SCHED,
                auto_repro: false,
                replicas: 1,
                assert: Assert::NONE,
                extra_sched_args: &[],
                required_flags: &[],
                excluded_flags: &[],
                min_sockets: 1,
                min_llcs: 1,
                requires_smt: false,
                min_cpus: 1,
                watchdog_timeout_jiffies: 0,
                bpf_map_write: None,
                performance_mode: false,
                super_perf_mode: false,
                duration_s: 5,
                workers_per_cgroup: 2,
            };

            let result = run_stt_test(&__STT_ENTRY);
            assert!(
                result.is_ok(),
                "normal scheduler should pass {} gate: {:#}",
                stringify!($name),
                result.unwrap_err()
            );
        }
    };
}

// ---------------------------------------------------------------------------
// Macro for no-perf negative tests (--degrade, tight threshold)
// ---------------------------------------------------------------------------

macro_rules! noperf_negative_test {
    ($name:ident, $checks:expr, $err_pattern:expr) => {
        #[ignore]
        #[test]
        fn $name() {
            fn scenario(ctx: &Ctx) -> Result<AssertResult> {
                let checks = $checks;
                scenario_with_checks(ctx, &checks)
            }

            #[linkme::distributed_slice(stt::test_support::STT_TESTS)]
            #[linkme(crate = linkme)]
            static __STT_ENTRY: SttTestEntry = SttTestEntry {
                name: stringify!($name),
                func: scenario,
                sockets: 1,
                cores: 2,
                threads: 1,
                memory_mb: 2048,
                scheduler: &STT_SCHED,
                auto_repro: false,
                replicas: 1,
                assert: Assert::NONE,
                extra_sched_args: &["--degrade"],
                required_flags: &[],
                excluded_flags: &[],
                min_sockets: 1,
                min_llcs: 1,
                requires_smt: false,
                min_cpus: 1,
                watchdog_timeout_jiffies: 0,
                bpf_map_write: None,
                performance_mode: false,
                super_perf_mode: false,
                duration_s: 5,
                workers_per_cgroup: 4,
            };

            let result = run_stt_test(&__STT_ENTRY);
            assert!(
                result.is_err(),
                "degraded scheduler should fail {} gate",
                stringify!($name),
            );
            let err_msg = format!("{:#}", result.unwrap_err());
            assert!(
                err_msg.contains($err_pattern),
                "{} error should contain '{}': {err_msg}",
                stringify!($name),
                $err_pattern,
            );
        }
    };
}

// ===========================================================================
// max_p99_wake_latency_ns
// ===========================================================================

perf_positive_test!(
    gate_p99_wake_perf_on_positive,
    Assert::default_checks().max_p99_wake_latency_ns(100_000_000)
);

perf_negative_test!(
    gate_p99_wake_perf_on_negative,
    Assert::default_checks().max_p99_wake_latency_ns(1),
    "p99 wake latency"
);

noperf_positive_test!(
    gate_p99_wake_perf_off_positive,
    Assert::default_checks().max_p99_wake_latency_ns(100_000_000)
);

noperf_negative_test!(
    gate_p99_wake_perf_off_negative,
    Assert::default_checks().max_p99_wake_latency_ns(1),
    "p99 wake latency"
);

// ===========================================================================
// max_wake_latency_cv
// ===========================================================================

perf_positive_test!(
    gate_wake_cv_perf_on_positive,
    Assert::default_checks().max_wake_latency_cv(100.0)
);

perf_negative_test!(
    gate_wake_cv_perf_on_negative,
    Assert::default_checks().max_wake_latency_cv(0.0001),
    "wake latency CV"
);

noperf_positive_test!(
    gate_wake_cv_perf_off_positive,
    Assert::default_checks().max_wake_latency_cv(100.0)
);

noperf_negative_test!(
    gate_wake_cv_perf_off_negative,
    Assert::default_checks().max_wake_latency_cv(0.0001),
    "wake latency CV"
);

// ===========================================================================
// min_iteration_rate
// ===========================================================================

perf_positive_test!(
    gate_iter_rate_perf_on_positive,
    Assert::default_checks().min_iteration_rate(1.0)
);

perf_negative_test!(
    gate_iter_rate_perf_on_negative,
    Assert::default_checks().min_iteration_rate(1_000_000_000.0),
    "iteration rate"
);

noperf_positive_test!(
    gate_iter_rate_perf_off_positive,
    Assert::default_checks().min_iteration_rate(1.0)
);

noperf_negative_test!(
    gate_iter_rate_perf_off_negative,
    Assert::default_checks().min_iteration_rate(1_000_000_000.0),
    "iteration rate"
);

// ===========================================================================
// max_gap_ms
// ===========================================================================

perf_positive_test!(
    gate_gap_ms_perf_on_positive,
    Assert::default_checks().max_gap_ms(10_000)
);

perf_negative_test!(
    gate_gap_ms_perf_on_negative,
    Assert::default_checks().max_gap_ms(50),
    "stuck"
);

noperf_positive_test!(
    gate_gap_ms_perf_off_positive,
    Assert::default_checks().max_gap_ms(10_000)
);

noperf_negative_test!(
    gate_gap_ms_perf_off_negative,
    Assert::default_checks().max_gap_ms(50),
    "stuck"
);

// ===========================================================================
// max_spread_pct
// ===========================================================================

perf_positive_test!(
    gate_spread_perf_on_positive,
    Assert::default_checks().max_spread_pct(99.0)
);

perf_negative_test!(
    gate_spread_perf_on_negative,
    Assert::default_checks().max_spread_pct(0.01),
    "unfair"
);

noperf_positive_test!(
    gate_spread_perf_off_positive,
    Assert::default_checks().max_spread_pct(99.0)
);

noperf_negative_test!(
    gate_spread_perf_off_negative,
    Assert::default_checks().max_spread_pct(0.01),
    "unfair"
);

// ===========================================================================
// max_throughput_cv
// ===========================================================================

perf_positive_test!(
    gate_throughput_cv_perf_on_positive,
    Assert::default_checks().max_throughput_cv(100.0)
);

perf_negative_test!(
    gate_throughput_cv_perf_on_negative,
    Assert::default_checks().max_throughput_cv(0.0001),
    "throughput CV"
);

noperf_positive_test!(
    gate_throughput_cv_perf_off_positive,
    Assert::default_checks().max_throughput_cv(100.0)
);

noperf_negative_test!(
    gate_throughput_cv_perf_off_negative,
    Assert::default_checks().max_throughput_cv(0.0001),
    "throughput CV"
);

// ===========================================================================
// min_work_rate
// ===========================================================================

perf_positive_test!(
    gate_work_rate_perf_on_positive,
    Assert::default_checks().min_work_rate(1.0)
);

perf_negative_test!(
    gate_work_rate_perf_on_negative,
    Assert::default_checks().min_work_rate(1_000_000_000_000.0),
    "below floor"
);

noperf_positive_test!(
    gate_work_rate_perf_off_positive,
    Assert::default_checks().min_work_rate(1.0)
);

noperf_negative_test!(
    gate_work_rate_perf_off_negative,
    Assert::default_checks().min_work_rate(1_000_000_000_000.0),
    "below floor"
);
