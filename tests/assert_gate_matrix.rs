use anyhow::Result;
use ktstr::assert::{Assert, AssertResult};
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps_with};
use ktstr::test_support::{KtstrTestEntry, Scheduler, SchedulerSpec};

const KTSTR_SCHED: Scheduler =
    Scheduler::new("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

fn scenario_with_checks(ctx: &Ctx, checks: &Assert) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps_with(ctx, steps, Some(checks))
}

// Macro emits a module-scope distributed_slice entry. Each test gets a
// scenario function that captures its Assert checks, and a static
// KtstrTestEntry registered in KTSTR_TESTS.
//
// perf: performance_mode value (true/false)
// negative: when true, passes --degrade, expects failure, uses 4 workers

macro_rules! gate_test {
    ($name:ident, perf: $perf:expr, negative: $neg:expr, $checks:expr) => {
        mod $name {
            use super::*;
            pub(super) fn scenario(ctx: &Ctx) -> Result<AssertResult> {
                let checks = $checks;
                scenario_with_checks(ctx, &checks)
            }
        }

        #[allow(non_upper_case_globals)]
        #[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
        #[linkme(crate = ktstr::__private::linkme)]
        static $name: KtstrTestEntry = KtstrTestEntry {
            name: stringify!($name),
            func: $name::scenario,
            scheduler: &KTSTR_SCHED,
            auto_repro: false,
            performance_mode: $perf,
            extra_sched_args: if $neg { &["--degrade"] } else { &[] },
            expect_err: $neg,
            workers_per_cgroup: if $neg { 4 } else { 2 },
            duration: std::time::Duration::from_secs(5),
            ..KtstrTestEntry::DEFAULT
        };
    };
}

// ===========================================================================
// max_p99_wake_latency_ns
// ===========================================================================

gate_test!(demo_gate_p99_wake_perf_on_positive, perf: true, negative: false,
    Assert::default_checks().max_p99_wake_latency_ns(100_000_000));
gate_test!(demo_gate_p99_wake_perf_on_negative, perf: true, negative: true,
    Assert::default_checks().max_p99_wake_latency_ns(1));
gate_test!(demo_gate_p99_wake_perf_off_positive, perf: false, negative: false,
    Assert::default_checks().max_p99_wake_latency_ns(100_000_000));
gate_test!(demo_gate_p99_wake_perf_off_negative, perf: false, negative: true,
    Assert::default_checks().max_p99_wake_latency_ns(1));

// ===========================================================================
// max_wake_latency_cv
// ===========================================================================

gate_test!(demo_gate_wake_cv_perf_on_positive, perf: true, negative: false,
    Assert::default_checks().max_wake_latency_cv(100.0));
gate_test!(demo_gate_wake_cv_perf_on_negative, perf: true, negative: true,
    Assert::default_checks().max_wake_latency_cv(0.0001));
gate_test!(demo_gate_wake_cv_perf_off_positive, perf: false, negative: false,
    Assert::default_checks().max_wake_latency_cv(100.0));
gate_test!(demo_gate_wake_cv_perf_off_negative, perf: false, negative: true,
    Assert::default_checks().max_wake_latency_cv(0.0001));

// ===========================================================================
// min_iteration_rate
// ===========================================================================

gate_test!(demo_gate_iter_rate_perf_on_positive, perf: true, negative: false,
    Assert::default_checks().min_iteration_rate(1.0));
gate_test!(demo_gate_iter_rate_perf_on_negative, perf: true, negative: true,
    Assert::default_checks().min_iteration_rate(1_000_000_000.0));
gate_test!(demo_gate_iter_rate_perf_off_positive, perf: false, negative: false,
    Assert::default_checks().min_iteration_rate(1.0));
gate_test!(demo_gate_iter_rate_perf_off_negative, perf: false, negative: true,
    Assert::default_checks().min_iteration_rate(1_000_000_000.0));

// ===========================================================================
// max_gap_ms
// ===========================================================================

gate_test!(demo_gate_gap_ms_perf_on_positive, perf: true, negative: false,
    Assert::default_checks().max_gap_ms(10_000));
gate_test!(demo_gate_gap_ms_perf_on_negative, perf: true, negative: true,
    Assert::default_checks().max_gap_ms(50));
gate_test!(demo_gate_gap_ms_perf_off_positive, perf: false, negative: false,
    Assert::default_checks().max_gap_ms(10_000));
gate_test!(demo_gate_gap_ms_perf_off_negative, perf: false, negative: true,
    Assert::default_checks().max_gap_ms(50));

// ===========================================================================
// max_spread_pct
// ===========================================================================

gate_test!(demo_gate_spread_perf_on_positive, perf: true, negative: false,
    Assert::default_checks().max_spread_pct(99.0));
gate_test!(demo_gate_spread_perf_on_negative, perf: true, negative: true,
    Assert::default_checks().max_spread_pct(0.01));
gate_test!(demo_gate_spread_perf_off_positive, perf: false, negative: false,
    Assert::default_checks().max_spread_pct(99.0));
gate_test!(demo_gate_spread_perf_off_negative, perf: false, negative: true,
    Assert::default_checks().max_spread_pct(0.01));

// ===========================================================================
// max_throughput_cv
// ===========================================================================

gate_test!(demo_gate_throughput_cv_perf_on_positive, perf: true, negative: false,
    Assert::default_checks().max_throughput_cv(100.0));
gate_test!(demo_gate_throughput_cv_perf_on_negative, perf: true, negative: true,
    Assert::default_checks().max_throughput_cv(0.0001));
gate_test!(demo_gate_throughput_cv_perf_off_positive, perf: false, negative: false,
    Assert::default_checks().max_throughput_cv(100.0));
gate_test!(demo_gate_throughput_cv_perf_off_negative, perf: false, negative: true,
    Assert::default_checks().max_throughput_cv(0.0001));

// ===========================================================================
// min_work_rate
// ===========================================================================

gate_test!(demo_gate_work_rate_perf_on_positive, perf: true, negative: false,
    Assert::default_checks().min_work_rate(1.0));
gate_test!(demo_gate_work_rate_perf_on_negative, perf: true, negative: true,
    Assert::default_checks().min_work_rate(1_000_000_000_000.0));
gate_test!(demo_gate_work_rate_perf_off_positive, perf: false, negative: false,
    Assert::default_checks().min_work_rate(1.0));
gate_test!(demo_gate_work_rate_perf_off_negative, perf: false, negative: true,
    Assert::default_checks().min_work_rate(1_000_000_000_000.0));

// ===========================================================================
// max_migration_ratio
// ===========================================================================

gate_test!(demo_gate_migration_ratio_perf_on_positive, perf: true, negative: false,
    Assert::default_checks().max_migration_ratio(100.0));
gate_test!(demo_gate_migration_ratio_perf_on_negative, perf: true, negative: true,
    Assert::default_checks().max_migration_ratio(0.0));
gate_test!(demo_gate_migration_ratio_perf_off_positive, perf: false, negative: false,
    Assert::default_checks().max_migration_ratio(100.0));
gate_test!(demo_gate_migration_ratio_perf_off_negative, perf: false, negative: true,
    Assert::default_checks().max_migration_ratio(0.0));
