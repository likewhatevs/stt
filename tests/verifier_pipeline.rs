use anyhow::Result;
use stt::assert::AssertResult;
use stt::scenario::Ctx;
use stt::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps_with};
use stt::test_support::{Scheduler, SchedulerSpec, SttTestEntry};

/// Build a scheduler package and resolve paths for verifier tests.
fn resolve_verifier_paths(
    package: &str,
) -> Result<(std::path::PathBuf, std::path::PathBuf, std::path::PathBuf)> {
    let sched_bin = stt::build_and_find_binary(package)?;
    let stt_bin = std::env::current_exe()?;
    let kernel = stt::find_kernel().ok_or_else(|| anyhow::anyhow!("no kernel found"))?;
    Ok((sched_bin, stt_bin, kernel))
}

fn __stt_inner_demo_verifier_brief(_ctx: &Ctx) -> Result<AssertResult> {
    let (sched_bin, stt_bin, kernel) = resolve_verifier_paths("stt-sched")?;
    let result = stt::verifier::collect_verifier_output(&sched_bin, &stt_bin, &kernel, &[])?;
    let output = stt::verifier::format_verifier_output("stt-sched", &result, false);
    anyhow::ensure!(
        output.contains("stt_enqueue"),
        "output should list stt_enqueue"
    );
    anyhow::ensure!(
        output.contains("stt_dispatch"),
        "output should list stt_dispatch"
    );
    anyhow::ensure!(
        output.contains("verified_insns="),
        "output should contain verified_insns="
    );
    Ok(AssertResult::pass())
}

#[stt::__linkme::distributed_slice(stt::test_support::STT_TESTS)]
#[linkme(crate = stt::__linkme)]
static __STT_ENTRY_VERIFIER_BRIEF: SttTestEntry = SttTestEntry {
    name: "demo_verifier_brief",
    func: __stt_inner_demo_verifier_brief,
    auto_repro: false,
    host_only: true,
    ..SttTestEntry::DEFAULT
};

fn __stt_inner_demo_verifier_diff(_ctx: &Ctx) -> Result<AssertResult> {
    let (sched_bin, stt_bin, kernel) = resolve_verifier_paths("stt-sched")?;
    let result_a = stt::verifier::collect_verifier_output(&sched_bin, &stt_bin, &kernel, &[])?;
    let result_b = stt::verifier::collect_verifier_output(&sched_bin, &stt_bin, &kernel, &[])?;
    let output = stt::verifier::format_verifier_diff(
        "stt-sched",
        &result_a.stats,
        "stt-sched",
        &result_b.stats,
    );
    anyhow::ensure!(
        output.contains("delta"),
        "diff output should contain 'delta' header"
    );
    anyhow::ensure!(
        output.contains("program"),
        "diff output should contain 'program' column"
    );
    anyhow::ensure!(output.contains("+0"), "self-comparison deltas should be 0");
    Ok(AssertResult::pass())
}

#[stt::__linkme::distributed_slice(stt::test_support::STT_TESTS)]
#[linkme(crate = stt::__linkme)]
static __STT_ENTRY_VERIFIER_DIFF: SttTestEntry = SttTestEntry {
    name: "demo_verifier_diff",
    func: __stt_inner_demo_verifier_diff,
    auto_repro: false,
    host_only: true,
    ..SttTestEntry::DEFAULT
};

fn __stt_inner_verifier_cycle_collapse(_ctx: &Ctx) -> Result<AssertResult> {
    let (sched_bin, stt_bin, kernel) = resolve_verifier_paths("stt-sched")?;
    let sched_args = vec!["--verify-loop".to_string()];
    let result =
        stt::verifier::collect_verifier_output(&sched_bin, &stt_bin, &kernel, &sched_args)?;
    let output = stt::verifier::format_verifier_output("stt-sched", &result, false);
    anyhow::ensure!(
        output.contains("scheduler log"),
        "output should contain scheduler log section"
    );
    anyhow::ensure!(
        output.contains("identical iterations omitted"),
        "cycle collapse should compress verifier loop traces"
    );
    Ok(AssertResult::pass())
}

#[stt::__linkme::distributed_slice(stt::test_support::STT_TESTS)]
#[linkme(crate = stt::__linkme)]
static __STT_ENTRY_CYCLE_COLLAPSE: SttTestEntry = SttTestEntry {
    name: "verifier_cycle_collapse",
    func: __stt_inner_verifier_cycle_collapse,
    auto_repro: false,
    host_only: true,
    ..SttTestEntry::DEFAULT
};

// -- demo_verifier_fail_verify: BPF load rejection via --fail-verify --

const FAIL_SCHED: Scheduler = Scheduler::new("stt_sched").binary(SchedulerSpec::Name("stt-sched"));

fn scenario_fail_verify(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(2)].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps_with(ctx, steps, None)
}

#[stt::__linkme::distributed_slice(stt::test_support::STT_TESTS)]
#[linkme(crate = stt::__linkme)]
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

#[stt::__linkme::distributed_slice(stt::test_support::STT_TESTS)]
#[linkme(crate = stt::__linkme)]
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
