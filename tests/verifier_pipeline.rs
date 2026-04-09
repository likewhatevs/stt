use anyhow::Result;
use stt::assert::AssertResult;
use stt::scenario::Ctx;
use stt::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps_with};
use stt::test_support::{Scheduler, SchedulerSpec, SttTestEntry};

fn main() {
    if stt::test_support::is_pid1() {
        stt::test_support::stt_guest_init();
    }
    let args = libtest_mimic::Arguments::from_args();
    let mut trials = stt::test_support::build_stt_trials();

    // Verifier tests: build scheduler per test, call library with paths.
    trials.push(
        libtest_mimic::Trial::test("demo_verifier_brief", || {
            let (sched_bin, stt_bin, kernel) = resolve_verifier_paths("stt-sched")?;
            let result = stt::verifier::collect_verifier_output(&sched_bin, &stt_bin, &kernel, &[])
                .map_err(|e| format!("{e:#}"))?;
            let output = stt::verifier::format_verifier_output("stt-sched", &result, false);
            if !output.contains("stt_enqueue") {
                return Err("output should list stt_enqueue".into());
            }
            if !output.contains("stt_dispatch") {
                return Err("output should list stt_dispatch".into());
            }
            if !output.contains("insns=") {
                return Err("output should contain insns=".into());
            }
            if !output.contains("processed=") {
                return Err("output should contain processed=".into());
            }
            Ok(())
        })
        .with_ignored_flag(true),
    );

    trials.push(
        libtest_mimic::Trial::test("demo_verifier_diff", || {
            let (sched_bin, stt_bin, kernel) = resolve_verifier_paths("stt-sched")?;
            let result_a =
                stt::verifier::collect_verifier_output(&sched_bin, &stt_bin, &kernel, &[])
                    .map_err(|e| format!("{e:#}"))?;
            let result_b =
                stt::verifier::collect_verifier_output(&sched_bin, &stt_bin, &kernel, &[])
                    .map_err(|e| format!("{e:#}"))?;
            let output = stt::verifier::format_verifier_diff(
                "stt-sched",
                &result_a.stats,
                "stt-sched",
                &result_b.stats,
            );
            if !output.contains("delta") {
                return Err("diff output should contain 'delta' header".into());
            }
            if !output.contains("program") {
                return Err("diff output should contain 'program' column".into());
            }
            if !output.contains("+0") {
                return Err("self-comparison deltas should be 0".into());
            }
            Ok(())
        })
        .with_ignored_flag(true),
    );

    // Non-ignored: programmatic check that cycle collapse works.
    trials.push(libtest_mimic::Trial::test(
        "verifier_cycle_collapse",
        || {
            let (sched_bin, stt_bin, kernel) = resolve_verifier_paths("stt-sched")?;
            let sched_args = vec!["--verify-loop".to_string()];
            let result =
                stt::verifier::collect_verifier_output(&sched_bin, &stt_bin, &kernel, &sched_args)
                    .map_err(|e| format!("{e:#}"))?;
            let output = stt::verifier::format_verifier_output("stt-sched", &result, false);
            if !output.contains("stt_dispatch") {
                return Err("output should contain stt_dispatch program".into());
            }
            if !output.contains("scheduler log") {
                return Err("output should contain scheduler log section".into());
            }
            if !output.contains("identical iterations omitted") {
                return Err("cycle collapse should compress verifier loop traces".into());
            }
            Ok(())
        },
    ));

    let conclusion = libtest_mimic::run(&args, trials);
    stt::test_support::collect_and_print_sidecar_stats();
    conclusion.exit();
}

/// Build a scheduler package and resolve paths for verifier tests.
fn resolve_verifier_paths(
    package: &str,
) -> std::result::Result<
    (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf),
    libtest_mimic::Failed,
> {
    let sched_bin =
        stt::build_and_find_binary(package).map_err(|e| format!("build {package}: {e:#}"))?;
    let stt_bin = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let kernel =
        stt::find_kernel().ok_or_else(|| libtest_mimic::Failed::from("no kernel found"))?;
    Ok((sched_bin, stt_bin, kernel))
}

// -- demo_verifier_fail_verify: BPF load rejection via --fail-verify --

const FAIL_SCHED: Scheduler = Scheduler::new("stt_sched").binary(SchedulerSpec::Name("stt-sched"));

fn scenario_fail_verify(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(2)].into(),
        ops: vec![],
        hold: HoldSpec::Frac(1.0),
    }];
    execute_steps_with(ctx, steps, None)
}

#[linkme::distributed_slice(stt::test_support::STT_TESTS)]
#[linkme(crate = linkme)]
static __STT_ENTRY_FAIL_VERIFY: SttTestEntry = SttTestEntry {
    name: "demo_verifier_fail_verify",
    func: scenario_fail_verify,
    scheduler: &FAIL_SCHED,
    extra_sched_args: &["--fail-verify"],
    expect_err: true,
    duration_s: 5,
    workers_per_cgroup: 2,
    ..SttTestEntry::DEFAULT
};

#[linkme::distributed_slice(stt::test_support::STT_TESTS)]
#[linkme(crate = linkme)]
static __STT_ENTRY_VERIFY_REJECT: SttTestEntry = SttTestEntry {
    name: "demo_verifier_cycle_collapse",
    func: scenario_fail_verify,
    scheduler: &FAIL_SCHED,
    extra_sched_args: &["--verify-loop"],
    duration_s: 5,
    workers_per_cgroup: 2,
    expect_err: true,
    ..SttTestEntry::DEFAULT
};
