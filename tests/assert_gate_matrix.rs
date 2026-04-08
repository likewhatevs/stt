use anyhow::Result;
use stt::assert::{Assert, AssertResult};
use stt::scenario::Ctx;
use stt::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps_with};
use stt::test_support::{Scheduler, SchedulerSpec, SttTestEntry};

fn main() {
    let args = libtest_mimic::Arguments::from_args();
    let presets = stt::vm::gauntlet_presets();
    let host_llcs = stt::test_support::host_llc_count();
    let host_topo = stt::vmm::host_topology::HostTopology::from_sysfs().ok();
    let mut trials = Vec::new();

    for entry in stt::test_support::STT_TESTS.iter() {
        let profiles = entry
            .scheduler
            .generate_profiles(entry.required_flags, entry.excluded_flags);

        // Base trial: runs with the entry's own topology, not ignored
        // unless it's a demo_ test.
        let expect_err = entry.expect_err;
        trials.push(
            libtest_mimic::Trial::test(entry.name.to_string(), move || {
                match stt::test_support::run_stt_test(entry) {
                    Ok(_) if expect_err => Err("expected error but test passed".into()),
                    Ok(_) => Ok(()),
                    Err(_) if expect_err => Ok(()),
                    Err(e) => Err(format!("{e:#}").into()),
                }
            })
            .with_ignored_flag(entry.name.starts_with("demo_")),
        );

        // Gauntlet variants: topology x flags, always ignored by default.
        for preset in &presets {
            if !stt::test_support::preset_matches(preset, entry, host_llcs) {
                continue;
            }
            if let Some(ref host) = host_topo
                && !stt::test_support::host_preset_compatible(preset, host)
            {
                continue;
            }
            let t = &preset.topology;
            let topo_str = format!(
                "{}s{}c{}t",
                t.sockets, t.cores_per_socket, t.threads_per_core,
            );
            let cpus = t.total_cpus();
            let memory_mb = (cpus * 64).max(256).max(entry.memory_mb);
            let preset_name = preset.name;

            for profile in &profiles {
                let pname = profile.name();
                let name = format!("gauntlet/{}/{}/{}", entry.name, preset_name, pname);
                let topo_str = topo_str.clone();
                let flags: Vec<String> = profile.flags.iter().map(|s| s.to_string()).collect();

                let expect_err = entry.expect_err;
                trials.push(
                    libtest_mimic::Trial::test(name, move || {
                        let (sockets, cores, threads) =
                            stt::test_support::parse_topo_string(&topo_str)
                                .expect("invalid topo string");
                        let topo = stt::test_support::TopoOverride {
                            sockets,
                            cores,
                            threads,
                            memory_mb,
                        };
                        match stt::test_support::run_stt_test_with_topo_and_flags(
                            entry, &topo, &flags,
                        ) {
                            Ok(_) if expect_err => Err("expected error but test passed".into()),
                            Ok(_) => Ok(()),
                            Err(_) if expect_err => Ok(()),
                            Err(e) => Err(format!("{e:#}").into()),
                        }
                    })
                    .with_ignored_flag(true),
                );
            }
        }
    }

    let conclusion = libtest_mimic::run(&args, trials);
    if let Ok(dir) = std::env::var("STT_SIDECAR_DIR") {
        let sidecars = stt::test_support::collect_sidecars(std::path::Path::new(&dir));
        let rows: Vec<_> = sidecars.iter().map(stt::stats::sidecar_to_row).collect();
        if !rows.is_empty() {
            eprintln!("{}", stt::stats::analyze_rows(&rows));
        }
    }
    conclusion.exit();
}

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
        #[linkme::distributed_slice(stt::test_support::STT_TESTS)]
        #[linkme(crate = linkme)]
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
        #[linkme::distributed_slice(stt::test_support::STT_TESTS)]
        #[linkme(crate = linkme)]
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
        #[linkme::distributed_slice(stt::test_support::STT_TESTS)]
        #[linkme(crate = linkme)]
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
        #[linkme::distributed_slice(stt::test_support::STT_TESTS)]
        #[linkme(crate = linkme)]
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
