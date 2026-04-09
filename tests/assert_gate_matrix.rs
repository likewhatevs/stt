use anyhow::Result;
use stt::assert::{Assert, AssertResult};
use stt::scenario::Ctx;
use stt::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps_with};
use stt::test_support::{Scheduler, SchedulerSpec, SttTestEntry};

const STT_SCHED: Scheduler = Scheduler::new("stt_sched").binary(SchedulerSpec::Name("stt-sched"));

fn scenario_with_checks(ctx: &Ctx, checks: &Assert) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::Frac(1.0),
    }];
    execute_steps_with(ctx, steps, Some(checks))
}

// Macros emit module-scope distributed_slice entries. Each test gets a
// scenario function that captures its Assert checks, and a static
// SttTestEntry registered in STT_TESTS.

macro_rules! perf_positive_test {
    ($name:ident, $checks:expr) => {
        mod $name {
            use super::*;
            pub(super) fn scenario(ctx: &Ctx) -> Result<AssertResult> {
                let checks = $checks;
                scenario_with_checks(ctx, &checks)
            }
        }

        #[allow(non_upper_case_globals)]
        #[stt::__linkme::distributed_slice(stt::test_support::STT_TESTS)]
        #[linkme(crate = stt::__linkme)]
        static $name: SttTestEntry = SttTestEntry {
            name: stringify!($name),
            func: $name::scenario,
            scheduler: &STT_SCHED,
            auto_repro: false,
            performance_mode: true,
            duration_s: 5,
            workers_per_cgroup: 2,
            ..SttTestEntry::DEFAULT
        };
    };
}

macro_rules! perf_negative_test {
    ($name:ident, $checks:expr) => {
        mod $name {
            use super::*;
            pub(super) fn scenario(ctx: &Ctx) -> Result<AssertResult> {
                let checks = $checks;
                scenario_with_checks(ctx, &checks)
            }
        }

        #[allow(non_upper_case_globals)]
        #[stt::__linkme::distributed_slice(stt::test_support::STT_TESTS)]
        #[linkme(crate = stt::__linkme)]
        static $name: SttTestEntry = SttTestEntry {
            name: stringify!($name),
            func: $name::scenario,
            scheduler: &STT_SCHED,
            auto_repro: false,
            extra_sched_args: &["--degrade"],
            performance_mode: true,
            duration_s: 5,
            workers_per_cgroup: 4,
            expect_err: true,
            ..SttTestEntry::DEFAULT
        };
    };
}

macro_rules! noperf_positive_test {
    ($name:ident, $checks:expr) => {
        mod $name {
            use super::*;
            pub(super) fn scenario(ctx: &Ctx) -> Result<AssertResult> {
                let checks = $checks;
                scenario_with_checks(ctx, &checks)
            }
        }

        #[allow(non_upper_case_globals)]
        #[stt::__linkme::distributed_slice(stt::test_support::STT_TESTS)]
        #[linkme(crate = stt::__linkme)]
        static $name: SttTestEntry = SttTestEntry {
            name: stringify!($name),
            func: $name::scenario,
            scheduler: &STT_SCHED,
            auto_repro: false,
            duration_s: 5,
            workers_per_cgroup: 2,
            ..SttTestEntry::DEFAULT
        };
    };
}

macro_rules! noperf_negative_test {
    ($name:ident, $checks:expr) => {
        mod $name {
            use super::*;
            pub(super) fn scenario(ctx: &Ctx) -> Result<AssertResult> {
                let checks = $checks;
                scenario_with_checks(ctx, &checks)
            }
        }

        #[allow(non_upper_case_globals)]
        #[stt::__linkme::distributed_slice(stt::test_support::STT_TESTS)]
        #[linkme(crate = stt::__linkme)]
        static $name: SttTestEntry = SttTestEntry {
            name: stringify!($name),
            func: $name::scenario,
            scheduler: &STT_SCHED,
            auto_repro: false,
            extra_sched_args: &["--degrade"],
            duration_s: 5,
            workers_per_cgroup: 4,
            expect_err: true,
            ..SttTestEntry::DEFAULT
        };
    };
}

// ===========================================================================
// max_p99_wake_latency_ns
// ===========================================================================

perf_positive_test!(
    demo_gate_p99_wake_perf_on_positive,
    Assert::default_checks().max_p99_wake_latency_ns(100_000_000)
);

perf_negative_test!(
    demo_gate_p99_wake_perf_on_negative,
    Assert::default_checks().max_p99_wake_latency_ns(1)
);

noperf_positive_test!(
    demo_gate_p99_wake_perf_off_positive,
    Assert::default_checks().max_p99_wake_latency_ns(100_000_000)
);

noperf_negative_test!(
    demo_gate_p99_wake_perf_off_negative,
    Assert::default_checks().max_p99_wake_latency_ns(1)
);

// ===========================================================================
// max_wake_latency_cv
// ===========================================================================

perf_positive_test!(
    demo_gate_wake_cv_perf_on_positive,
    Assert::default_checks().max_wake_latency_cv(100.0)
);

perf_negative_test!(
    demo_gate_wake_cv_perf_on_negative,
    Assert::default_checks().max_wake_latency_cv(0.0001)
);

noperf_positive_test!(
    demo_gate_wake_cv_perf_off_positive,
    Assert::default_checks().max_wake_latency_cv(100.0)
);

noperf_negative_test!(
    demo_gate_wake_cv_perf_off_negative,
    Assert::default_checks().max_wake_latency_cv(0.0001)
);

// ===========================================================================
// min_iteration_rate
// ===========================================================================

perf_positive_test!(
    demo_gate_iter_rate_perf_on_positive,
    Assert::default_checks().min_iteration_rate(1.0)
);

perf_negative_test!(
    demo_gate_iter_rate_perf_on_negative,
    Assert::default_checks().min_iteration_rate(1_000_000_000.0)
);

noperf_positive_test!(
    demo_gate_iter_rate_perf_off_positive,
    Assert::default_checks().min_iteration_rate(1.0)
);

noperf_negative_test!(
    demo_gate_iter_rate_perf_off_negative,
    Assert::default_checks().min_iteration_rate(1_000_000_000.0)
);

// ===========================================================================
// max_gap_ms
// ===========================================================================

perf_positive_test!(
    demo_gate_gap_ms_perf_on_positive,
    Assert::default_checks().max_gap_ms(10_000)
);

perf_negative_test!(
    demo_gate_gap_ms_perf_on_negative,
    Assert::default_checks().max_gap_ms(50)
);

noperf_positive_test!(
    demo_gate_gap_ms_perf_off_positive,
    Assert::default_checks().max_gap_ms(10_000)
);

noperf_negative_test!(
    demo_gate_gap_ms_perf_off_negative,
    Assert::default_checks().max_gap_ms(50)
);

// ===========================================================================
// max_spread_pct
// ===========================================================================

perf_positive_test!(
    demo_gate_spread_perf_on_positive,
    Assert::default_checks().max_spread_pct(99.0)
);

perf_negative_test!(
    demo_gate_spread_perf_on_negative,
    Assert::default_checks().max_spread_pct(0.01)
);

noperf_positive_test!(
    demo_gate_spread_perf_off_positive,
    Assert::default_checks().max_spread_pct(99.0)
);

noperf_negative_test!(
    demo_gate_spread_perf_off_negative,
    Assert::default_checks().max_spread_pct(0.01)
);

// ===========================================================================
// max_throughput_cv
// ===========================================================================

perf_positive_test!(
    demo_gate_throughput_cv_perf_on_positive,
    Assert::default_checks().max_throughput_cv(100.0)
);

perf_negative_test!(
    demo_gate_throughput_cv_perf_on_negative,
    Assert::default_checks().max_throughput_cv(0.0001)
);

noperf_positive_test!(
    demo_gate_throughput_cv_perf_off_positive,
    Assert::default_checks().max_throughput_cv(100.0)
);

noperf_negative_test!(
    demo_gate_throughput_cv_perf_off_negative,
    Assert::default_checks().max_throughput_cv(0.0001)
);

// ===========================================================================
// min_work_rate
// ===========================================================================

perf_positive_test!(
    demo_gate_work_rate_perf_on_positive,
    Assert::default_checks().min_work_rate(1.0)
);

perf_negative_test!(
    demo_gate_work_rate_perf_on_negative,
    Assert::default_checks().min_work_rate(1_000_000_000_000.0)
);

noperf_positive_test!(
    demo_gate_work_rate_perf_off_positive,
    Assert::default_checks().min_work_rate(1.0)
);

noperf_negative_test!(
    demo_gate_work_rate_perf_off_negative,
    Assert::default_checks().min_work_rate(1_000_000_000_000.0)
);
