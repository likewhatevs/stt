use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps_with};
use ktstr::test_support::{KtstrTestEntry, Scheduler, SchedulerSpec};

/// Build a scheduler package and resolve paths for verifier tests.
/// Returns `Ok(None)` when no kernel is available (CI without a custom
/// kernel) — callers should skip the test, not fail.
fn resolve_verifier_paths(
    package: &str,
) -> Result<Option<(std::path::PathBuf, std::path::PathBuf, std::path::PathBuf)>> {
    let Some(kernel) = ktstr::find_kernel()? else {
        return Ok(None);
    };
    let sched_bin = ktstr::build_and_find_binary(package)?;
    let ktstr_bin = std::env::current_exe()?;
    Ok(Some((sched_bin, ktstr_bin, kernel)))
}

fn __ktstr_inner_demo_verifier_brief(_ctx: &Ctx) -> Result<AssertResult> {
    let Some((sched_bin, ktstr_bin, kernel)) = resolve_verifier_paths("scx-ktstr")? else {
        return Ok(AssertResult::pass());
    };
    let result = ktstr::verifier::collect_verifier_output(&sched_bin, &ktstr_bin, &kernel, &[])?;
    let output = ktstr::verifier::format_verifier_output("scx-ktstr", &result, false);
    anyhow::ensure!(
        output.contains("ktstr_enqueue"),
        "output should list ktstr_enqueue"
    );
    anyhow::ensure!(
        output.contains("ktstr_dispatch"),
        "output should list ktstr_dispatch"
    );
    anyhow::ensure!(
        output.contains("verified_insns="),
        "output should contain verified_insns="
    );
    Ok(AssertResult::pass())
}

#[ktstr::__linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__linkme)]
static __KTSTR_ENTRY_VERIFIER_BRIEF: KtstrTestEntry = KtstrTestEntry {
    name: "demo_verifier_brief",
    func: __ktstr_inner_demo_verifier_brief,
    auto_repro: false,
    host_only: true,
    ..KtstrTestEntry::DEFAULT
};

fn __ktstr_inner_demo_verifier_diff(_ctx: &Ctx) -> Result<AssertResult> {
    let Some((sched_bin, ktstr_bin, kernel)) = resolve_verifier_paths("scx-ktstr")? else {
        return Ok(AssertResult::pass());
    };
    let result_a = ktstr::verifier::collect_verifier_output(&sched_bin, &ktstr_bin, &kernel, &[])?;
    let result_b = ktstr::verifier::collect_verifier_output(&sched_bin, &ktstr_bin, &kernel, &[])?;
    let output = ktstr::verifier::format_verifier_diff(
        "scx-ktstr",
        &result_a.stats,
        "scx-ktstr",
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

#[ktstr::__linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__linkme)]
static __KTSTR_ENTRY_VERIFIER_DIFF: KtstrTestEntry = KtstrTestEntry {
    name: "demo_verifier_diff",
    func: __ktstr_inner_demo_verifier_diff,
    auto_repro: false,
    host_only: true,
    ..KtstrTestEntry::DEFAULT
};

fn __ktstr_inner_verifier_cycle_collapse(_ctx: &Ctx) -> Result<AssertResult> {
    let Some((sched_bin, ktstr_bin, kernel)) = resolve_verifier_paths("scx-ktstr")? else {
        return Ok(AssertResult::pass());
    };
    let sched_args = vec!["--verify-loop".to_string()];
    let result =
        ktstr::verifier::collect_verifier_output(&sched_bin, &ktstr_bin, &kernel, &sched_args)?;
    let output = ktstr::verifier::format_verifier_output("scx-ktstr", &result, false);
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

#[ktstr::__linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__linkme)]
static __KTSTR_ENTRY_CYCLE_COLLAPSE: KtstrTestEntry = KtstrTestEntry {
    name: "verifier_cycle_collapse",
    func: __ktstr_inner_verifier_cycle_collapse,
    auto_repro: false,
    host_only: true,
    ..KtstrTestEntry::DEFAULT
};

// -- demo_verifier_fail_verify: BPF load rejection via --fail-verify --

const FAIL_SCHED: Scheduler =
    Scheduler::new("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

fn scenario_fail_verify(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(2)].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps_with(ctx, steps, None)
}

#[ktstr::__linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__linkme)]
static __KTSTR_ENTRY_FAIL_VERIFY: KtstrTestEntry = KtstrTestEntry {
    name: "demo_verifier_fail_verify",
    func: scenario_fail_verify,
    scheduler: &FAIL_SCHED,
    extra_sched_args: &["--fail-verify"],
    duration: std::time::Duration::from_secs(5),
    workers_per_cgroup: 2,
    ..KtstrTestEntry::DEFAULT
};

#[ktstr::__linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__linkme)]
static __KTSTR_ENTRY_VERIFY_REJECT: KtstrTestEntry = KtstrTestEntry {
    name: "demo_verifier_cycle_collapse",
    func: scenario_fail_verify,
    scheduler: &FAIL_SCHED,
    extra_sched_args: &["--verify-loop"],
    duration: std::time::Duration::from_secs(5),
    workers_per_cgroup: 2,
    ..KtstrTestEntry::DEFAULT
};
