use anyhow::Result;
use stt::scenario::Ctx;
use stt::stt_test;
use stt::test_support::{Scheduler, SchedulerSpec};
use stt::verify::VerifyResult;

const STT_SCHED: Scheduler = Scheduler::new("stt_sched").binary(SchedulerSpec::Name("stt-sched"));

// -- basic --

#[stt_test(sockets = 1, cores = 2, threads = 1, memory_mb = 2048)]
fn cover_cgroup_pipe_io(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::basic::custom_cgroup_pipe_io(ctx)
}

#[stt_test(scheduler = STT_SCHED, sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_sched_mixed(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::basic::custom_sched_mixed(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_host_cgroup_contention(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::basic::custom_host_cgroup_contention(ctx)
}

// -- affinity --

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_affinity_change(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::affinity::custom_cgroup_affinity_change(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_multicpu_pin(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::affinity::custom_cgroup_multicpu_pin(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_cpuset_multicpu_pin(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::affinity::custom_cgroup_cpuset_multicpu_pin(ctx)
}

// -- cpuset --

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_cpuset_apply_midrun(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::cpuset::custom_cgroup_cpuset_apply_midrun(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_cpuset_clear_midrun(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::cpuset::custom_cgroup_cpuset_clear_midrun(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_cpuset_resize(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::cpuset::custom_cgroup_cpuset_resize(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 2, memory_mb = 2048)]
fn cover_cgroup_cpuset_swap_disjoint(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::cpuset::custom_cgroup_cpuset_swap_disjoint(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_cpuset_workload_imbalance(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::cpuset::custom_cgroup_cpuset_workload_imbalance(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_cpuset_change_imbalance(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::cpuset::custom_cgroup_cpuset_change_imbalance(ctx)
}

#[stt_test(scheduler = STT_SCHED, sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_cpuset_load_shift(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::cpuset::custom_cgroup_cpuset_load_shift(ctx)
}

// -- dynamic --

#[stt_test(scheduler = STT_SCHED, sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_add_midrun(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::dynamic::custom_cgroup_add_midrun(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_remove_midrun(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::dynamic::custom_cgroup_remove_midrun(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_rapid_churn(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::dynamic::custom_cgroup_rapid_churn(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_cpuset_add_remove(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::dynamic::custom_cgroup_cpuset_add_remove(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_add_during_imbalance(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::dynamic::custom_cgroup_add_during_imbalance(ctx)
}

// -- interaction --

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_imbalance_mixed_workload(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::interaction::custom_cgroup_imbalance_mixed_workload(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_add_load_imbalance(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::interaction::custom_cgroup_add_load_imbalance(ctx)
}

#[stt_test(scheduler = STT_SCHED, sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_load_oscillation(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::interaction::custom_cgroup_load_oscillation(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 2, memory_mb = 2048)]
fn cover_cgroup_4way_load_imbalance(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::interaction::custom_cgroup_4way_load_imbalance(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_cpuset_imbalance_combined(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::interaction::custom_cgroup_cpuset_imbalance_combined(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_cpuset_overlap_imbalance_combined(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::interaction::custom_cgroup_cpuset_overlap_imbalance_combined(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_noctrl_task_migration(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::interaction::custom_cgroup_noctrl_task_migration(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_noctrl_imbalance(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::interaction::custom_cgroup_noctrl_imbalance(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_noctrl_cpuset_change(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::interaction::custom_cgroup_noctrl_cpuset_change(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_noctrl_load_imbalance(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::interaction::custom_cgroup_noctrl_load_imbalance(ctx)
}

#[stt_test(scheduler = STT_SCHED, sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_io_compute_imbalance(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::interaction::custom_cgroup_io_compute_imbalance(ctx)
}

// -- nested --

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_nested_cgroup_steady(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::nested::custom_nested_cgroup_steady(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_nested_cgroup_task_move(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::nested::custom_nested_cgroup_task_move(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_nested_cgroup_rapid_churn(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::nested::custom_nested_cgroup_rapid_churn(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_nested_cgroup_cpuset(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::nested::custom_nested_cgroup_cpuset(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_nested_cgroup_imbalance(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::nested::custom_nested_cgroup_imbalance(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_nested_cgroup_noctrl(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::nested::custom_nested_cgroup_noctrl(ctx)
}

// -- stress --

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_per_cpu(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::stress::custom_cgroup_per_cpu(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_exhaust_reuse(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::stress::custom_cgroup_exhaust_reuse(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_dsq_contention(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::stress::custom_cgroup_dsq_contention(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 2, memory_mb = 2048)]
fn cover_cgroup_workload_variety(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::stress::custom_cgroup_workload_variety(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 2, memory_mb = 2048)]
fn cover_cgroup_cpuset_workload_variety(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::stress::custom_cgroup_cpuset_workload_variety(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 2, memory_mb = 2048)]
fn cover_cgroup_dynamic_workload_variety(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::stress::custom_cgroup_dynamic_workload_variety(ctx)
}

#[stt_test(sockets = 2, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_cpuset_crossllc_race(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::stress::custom_cgroup_cpuset_crossllc_race(ctx)
}

// -- watchdog timeout overwrite --

#[stt_test(
    scheduler = STT_SCHED,
    sockets = 1, cores = 4, threads = 1, memory_mb = 2048,
    watchdog_timeout_jiffies = 60000,
)]
fn cover_watchdog_long_timeout_survives(ctx: &Ctx) -> Result<VerifyResult> {
    stt::scenario::basic::custom_sched_mixed(ctx)
}

#[test]
fn cover_watchdog_forced_stall() {
    use stt::test_support::{SttTestEntry, run_stt_test};

    fn scenario(_ctx: &stt::scenario::Ctx) -> Result<stt::verify::VerifyResult> {
        stt::scenario::basic::custom_sched_mixed(_ctx)
    }

    static STALL_SCHED: Scheduler =
        Scheduler::new("stt_sched").binary(SchedulerSpec::Name("stt-sched"));

    #[linkme::distributed_slice(stt::test_support::STT_TESTS)]
    #[linkme(crate = linkme)]
    static __STT_ENTRY_FORCED_STALL: SttTestEntry = SttTestEntry {
        name: "cover_watchdog_forced_stall",
        func: scenario,
        sockets: 1,
        cores: 4,
        threads: 1,
        memory_mb: 2048,
        scheduler: &STALL_SCHED,
        auto_repro: true,
        replicas: 1,
        verify: stt::verify::Verify::NONE,
        extra_sched_args: &["--stall-after", "1"],
        watchdog_timeout_jiffies: 2000,
        bpf_map_write: None,
    };

    let result = run_stt_test(&__STT_ENTRY_FORCED_STALL);
    assert!(
        result.is_err(),
        "expected error from watchdog-killed scheduler"
    );
    let err = format!("{:#}", result.unwrap_err());
    assert!(
        err.contains("scheduler died")
            || err.contains("SCHEDULER_DIED")
            || err.contains("timed out"),
        "expected scheduler death or timeout, got: {err}"
    );
    // The error must reference the test name in the stt_test error wrapper.
    assert!(
        err.contains("cover_watchdog_forced_stall"),
        "error must reference the test name: {err}"
    );
    // With watchdog_timeout_jiffies=2000 (2s) and --stall-after 1,
    // the stall fires at ~1s and watchdog kills at ~2s. The error
    // should NOT reference default 5s watchdog behavior.
    assert!(
        !err.contains("5.0s") && !err.contains("5000ms"),
        "error should reflect lowered watchdog timeout, not default 5s: {err}"
    );
}

// -- negative: stall detection with SCX_EXIT_ERROR_STALL --

#[test]
fn neg_stall_detection_scx_exit() {
    use stt::test_support::{SttTestEntry, run_stt_test};

    fn scenario(_ctx: &stt::scenario::Ctx) -> Result<stt::verify::VerifyResult> {
        stt::scenario::basic::custom_sched_mixed(_ctx)
    }

    static STALL_SCHED: Scheduler =
        Scheduler::new("stt_sched").binary(SchedulerSpec::Name("stt-sched"));

    #[linkme::distributed_slice(stt::test_support::STT_TESTS)]
    #[linkme(crate = linkme)]
    static __STT_ENTRY_STALL_DETECT: SttTestEntry = SttTestEntry {
        name: "neg_stall_detection_scx_exit",
        func: scenario,
        sockets: 1,
        cores: 4,
        threads: 1,
        memory_mb: 2048,
        scheduler: &STALL_SCHED,
        auto_repro: false,
        replicas: 1,
        verify: stt::verify::Verify::NONE,
        extra_sched_args: &["--stall-after", "1"],
        watchdog_timeout_jiffies: 3000,
        bpf_map_write: None,
    };

    let result = run_stt_test(&__STT_ENTRY_STALL_DETECT);
    assert!(result.is_err(), "stalled scheduler must cause test failure");
    let err = format!("{:#}", result.unwrap_err());
    assert!(
        err.contains("scheduler died")
            || err.contains("SCHEDULER_DIED")
            || err.contains("timed out")
            || err.contains("stall"),
        "expected stall-related error, got: {err}"
    );
    // Error must reference the test name.
    assert!(
        err.contains("neg_stall_detection_scx_exit"),
        "error must reference the test name: {err}"
    );
}

// -- negative: scheduler death without VerifyResult --

#[test]
fn neg_sched_death_no_verify_result() {
    use stt::test_support::{SttTestEntry, run_stt_test};

    fn scenario(_ctx: &stt::scenario::Ctx) -> Result<stt::verify::VerifyResult> {
        stt::scenario::basic::custom_sched_mixed(_ctx)
    }

    static DEATH_SCHED: Scheduler =
        Scheduler::new("stt_sched").binary(SchedulerSpec::Name("stt-sched"));

    #[linkme::distributed_slice(stt::test_support::STT_TESTS)]
    #[linkme(crate = linkme)]
    static __STT_ENTRY_SCHED_DEATH: SttTestEntry = SttTestEntry {
        name: "neg_sched_death_no_verify_result",
        func: scenario,
        sockets: 1,
        cores: 4,
        threads: 1,
        memory_mb: 2048,
        scheduler: &DEATH_SCHED,
        auto_repro: true,
        replicas: 1,
        verify: stt::verify::Verify::NONE,
        extra_sched_args: &["--stall-after", "1"],
        watchdog_timeout_jiffies: 3000,
        bpf_map_write: None,
    };

    let result = run_stt_test(&__STT_ENTRY_SCHED_DEATH);
    // Scheduler stalls after 1s, kernel watchdog kills it.
    // No VerifyResult is produced — the error path handles this.
    assert!(result.is_err(), "stalled scheduler must cause failure");
    let err = format!("{:#}", result.unwrap_err());
    assert!(
        err.contains("scheduler died")
            || err.contains("SCHEDULER_DIED")
            || err.contains("timed out"),
        "expected scheduler death, got: {err}"
    );
    // Error must reference the test name.
    assert!(
        err.contains("neg_sched_death_no_verify_result"),
        "error must reference the test name: {err}"
    );
    // The error message format from run_stt_test_inner includes
    // "[sched=stt-sched]" or "[sched=stt_sched]" when a scheduler is set.
    assert!(
        err.contains("[sched="),
        "error must include scheduler label: {err}"
    );
}

// -- negative: auto-repro exercises attempt_auto_repro path --

#[test]
fn neg_auto_repro_on_verify_failure() {
    use stt::test_support::{SttTestEntry, run_stt_test};

    // Scenario completes normally but forces passed=false, triggering
    // the verify-failure branch with auto_repro=true. attempt_auto_repro
    // extracts functions from the scheduler log; a normal exit has no
    // stack traces, so it returns None. No --stall-after: the scheduler
    // must stay alive so the VerifyResult reaches COM2.
    fn scenario(_ctx: &stt::scenario::Ctx) -> Result<stt::verify::VerifyResult> {
        let mut r = stt::scenario::basic::custom_sched_mixed(_ctx)?;
        r.passed = false;
        r.details.push("forced failure for auto-repro test".into());
        Ok(r)
    }

    static REPRO_SCHED: Scheduler =
        Scheduler::new("stt_sched").binary(SchedulerSpec::Name("stt-sched"));

    #[linkme::distributed_slice(stt::test_support::STT_TESTS)]
    #[linkme(crate = linkme)]
    static __STT_ENTRY_AUTO_REPRO_VERIFY: SttTestEntry = SttTestEntry {
        name: "neg_auto_repro_on_verify_failure",
        func: scenario,
        sockets: 1,
        cores: 4,
        threads: 1,
        memory_mb: 2048,
        scheduler: &REPRO_SCHED,
        auto_repro: true,
        replicas: 1,
        verify: stt::verify::Verify::NONE,
        extra_sched_args: &[],
        watchdog_timeout_jiffies: 0,
        bpf_map_write: None,
    };

    let result = run_stt_test(&__STT_ENTRY_AUTO_REPRO_VERIFY);
    assert!(result.is_err(), "forced failure must propagate");
    let err = format!("{:#}", result.unwrap_err());
    // The error must contain the forced failure detail string.
    assert!(
        err.contains("forced failure for auto-repro test"),
        "error must contain the forced failure detail, got: {err}"
    );
    // The error wrapper format is: "stt_test 'NAME' [sched=X] failed:\n  DETAILS"
    assert!(
        err.contains("neg_auto_repro_on_verify_failure"),
        "error must reference the test name: {err}"
    );
    assert!(
        err.contains("failed"),
        "error must contain 'failed' from the stt_test wrapper: {err}"
    );
    assert!(
        err.contains("[sched="),
        "error must include scheduler label: {err}"
    );
    // Auto-repro was attempted but returned None (no stack traces
    // in a normal scheduler exit). Verify "auto-repro" section is
    // NOT present — confirming attempt_auto_repro returned None.
    assert!(
        !err.contains("--- auto-repro ---"),
        "auto-repro section should be absent (no crash stack in normal exit): {err}"
    );
}

// -- negative: host-side BPF map write triggers scheduler crash --

#[test]
fn neg_crash_after_auto_repro() {
    use stt::test_support::{BpfMapWrite, SttTestEntry, run_stt_test};

    fn scenario(_ctx: &stt::scenario::Ctx) -> Result<stt::verify::VerifyResult> {
        stt::scenario::basic::custom_sched_mixed(_ctx)
    }

    static CRASH_SCHED: Scheduler =
        Scheduler::new("stt_sched").binary(SchedulerSpec::Name("stt-sched"));

    /// Write crash=1 to the scheduler's .bss map after scenario starts.
    /// `crash` is at offset 4 (after `stall`, both volatile int).
    static BPF_CRASH: BpfMapWrite = BpfMapWrite {
        map_name_suffix: ".bss",
        offset: 4,
        value: 1,
    };

    #[linkme::distributed_slice(stt::test_support::STT_TESTS)]
    #[linkme(crate = linkme)]
    static __STT_ENTRY_CRASH_AFTER: SttTestEntry = SttTestEntry {
        name: "neg_crash_after_auto_repro",
        func: scenario,
        sockets: 1,
        cores: 4,
        threads: 1,
        memory_mb: 2048,
        scheduler: &CRASH_SCHED,
        auto_repro: true,
        replicas: 1,
        verify: stt::verify::Verify::NONE,
        extra_sched_args: &[],
        watchdog_timeout_jiffies: 0,
        bpf_map_write: Some(&BPF_CRASH),
    };

    let result = run_stt_test(&__STT_ENTRY_CRASH_AFTER);
    assert!(result.is_err(), "scheduler crash must cause test failure");
    let err = format!("{:#}", result.unwrap_err());
    // A scheduler crash can manifest as:
    // - "scheduler died" — COM2 never received a result
    // - "SCHEDULER_DIED" — exit info in guest output
    // - "timed out" — VM timed out waiting for completion
    // - "monitor failed" — scenario completed but post-crash stalls
    //   triggered monitor threshold violations
    assert!(
        err.contains("scheduler died")
            || err.contains("SCHEDULER_DIED")
            || err.contains("timed out")
            || err.contains("monitor failed"),
        "expected scheduler death, timeout, or monitor failure, got: {err}"
    );
    assert!(
        err.contains("neg_crash_after_auto_repro"),
        "error must reference the test name: {err}"
    );
    assert!(
        err.contains("[sched="),
        "error must include scheduler label: {err}"
    );
}

// -- demo: host-triggered crash with auto-repro pipeline --

/// Demo test: triggers a scheduler crash via host-side BPF map write,
/// then verifies auto-repro extracts stack traces and attaches probes.
/// Run manually with `cargo test demo_bpf_crash_auto_repro -- --ignored`.
#[ignore]
#[test]
fn demo_bpf_crash_auto_repro() {
    use stt::test_support::{BpfMapWrite, SttTestEntry, run_stt_test};

    fn scenario(_ctx: &stt::scenario::Ctx) -> Result<stt::verify::VerifyResult> {
        stt::scenario::basic::custom_sched_mixed(_ctx)
    }

    static CRASH_SCHED: Scheduler =
        Scheduler::new("stt_sched").binary(SchedulerSpec::Name("stt-sched"));

    /// Write crash=1 to the scheduler's .bss map after scenario starts.
    /// `crash` is at offset 4 (after `stall`, both volatile int).
    static BPF_CRASH: BpfMapWrite = BpfMapWrite {
        map_name_suffix: ".bss",
        offset: 4,
        value: 1,
    };

    #[linkme::distributed_slice(stt::test_support::STT_TESTS)]
    #[linkme(crate = linkme)]
    static __STT_ENTRY_DEMO_CRASH: SttTestEntry = SttTestEntry {
        name: "demo_bpf_crash_auto_repro",
        func: scenario,
        sockets: 1,
        cores: 4,
        threads: 1,
        memory_mb: 2048,
        scheduler: &CRASH_SCHED,
        auto_repro: true,
        replicas: 1,
        verify: stt::verify::Verify::NONE,
        extra_sched_args: &[],
        watchdog_timeout_jiffies: 0,
        bpf_map_write: Some(&BPF_CRASH),
    };

    let result = run_stt_test(&__STT_ENTRY_DEMO_CRASH);
    assert!(result.is_err(), "scheduler crash must cause test failure");
    let err = format!("{:#}", result.unwrap_err());

    // The scheduler must die from the host-triggered crash.
    assert!(
        err.contains("scheduler died")
            || err.contains("SCHEDULER_DIED")
            || err.contains("timed out"),
        "expected scheduler death or timeout, got: {err}"
    );
    // Auto-repro should have found stack traces and attached probes.
    assert!(
        err.contains("--- auto-repro ---"),
        "auto-repro section must be present: {err}"
    );
}

// -- demo: host-triggered crash with auto-repro (inline scenario) --

/// Demo test: defines an inline yield-heavy scenario, triggers a
/// scheduler crash via host-side BPF map write after 3 seconds, then
/// verifies auto-repro extracts stack traces and attaches probes.
///
// -- host-triggered crash: validation (runs in CI) --

#[test]
fn neg_host_crash_auto_repro() {
    use std::time::Duration;
    use stt::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps};
    use stt::test_support::{BpfMapWrite, SttTestEntry, run_stt_test};
    use stt::workload::WorkType;

    fn scenario(ctx: &stt::scenario::Ctx) -> Result<stt::verify::VerifyResult> {
        let steps = vec![Step {
            setup: vec![
                CgroupDef::named("demo_workers")
                    .work_type(WorkType::YieldHeavy)
                    .workers(4),
            ]
            .into(),
            ops: vec![],
            hold: HoldSpec::Fixed(Duration::from_secs(8)),
        }];
        execute_steps(ctx, steps)
    }

    static CRASH_SCHED: Scheduler =
        Scheduler::new("stt_sched").binary(SchedulerSpec::Name("stt-sched"));

    static BPF_CRASH: BpfMapWrite = BpfMapWrite {
        map_name_suffix: ".bss",
        offset: 4,
        value: 1,
    };

    #[linkme::distributed_slice(stt::test_support::STT_TESTS)]
    #[linkme(crate = linkme)]
    static __STT_ENTRY_HOST_CRASH: SttTestEntry = SttTestEntry {
        name: "neg_host_crash_auto_repro",
        func: scenario,
        sockets: 1,
        cores: 4,
        threads: 1,
        memory_mb: 2048,
        scheduler: &CRASH_SCHED,
        auto_repro: true,
        replicas: 1,
        verify: stt::verify::Verify::NONE,
        extra_sched_args: &[],
        watchdog_timeout_jiffies: 0,
        bpf_map_write: Some(&BPF_CRASH),
    };

    let result = run_stt_test(&__STT_ENTRY_HOST_CRASH);
    assert!(result.is_err(), "scheduler crash must cause test failure");
    let err = format!("{:#}", result.unwrap_err());
    assert!(
        err.contains("scheduler died")
            || err.contains("SCHEDULER_DIED")
            || err.contains("timed out"),
        "expected scheduler death or timeout, got: {err}"
    );
    assert!(
        err.contains("--- auto-repro ---"),
        "auto-repro section must be present: {err}"
    );
}

// -- host-triggered crash: demo (run manually to see full output) --

/// Run manually to see the full auto-repro output:
///   cargo test demo_host_crash_auto_repro -- --ignored --nocapture
#[ignore]
#[test]
fn demo_host_crash_auto_repro() {
    use std::time::Duration;
    use stt::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps};
    use stt::test_support::{BpfMapWrite, SttTestEntry, run_stt_test};
    use stt::workload::WorkType;

    fn scenario(ctx: &stt::scenario::Ctx) -> Result<stt::verify::VerifyResult> {
        let steps = vec![Step {
            setup: vec![
                CgroupDef::named("demo_workers")
                    .work_type(WorkType::YieldHeavy)
                    .workers(4),
            ]
            .into(),
            ops: vec![],
            hold: HoldSpec::Fixed(Duration::from_secs(8)),
        }];
        execute_steps(ctx, steps)
    }

    static CRASH_SCHED: Scheduler =
        Scheduler::new("stt_sched").binary(SchedulerSpec::Name("stt-sched"));

    static BPF_CRASH: BpfMapWrite = BpfMapWrite {
        map_name_suffix: ".bss",
        offset: 4,
        value: 1,
    };

    #[linkme::distributed_slice(stt::test_support::STT_TESTS)]
    #[linkme(crate = linkme)]
    static __STT_ENTRY_DEMO_CRASH: SttTestEntry = SttTestEntry {
        name: "demo_host_crash_auto_repro",
        func: scenario,
        sockets: 1,
        cores: 4,
        threads: 1,
        memory_mb: 2048,
        scheduler: &CRASH_SCHED,
        auto_repro: true,
        replicas: 1,
        verify: stt::verify::Verify::NONE,
        extra_sched_args: &[],
        watchdog_timeout_jiffies: 0,
        bpf_map_write: Some(&BPF_CRASH),
    };

    let result = run_stt_test(&__STT_ENTRY_DEMO_CRASH);
    let err = format!("{:#}", result.unwrap_err());
    // Print the full error so the user sees the auto-repro output.
    panic!("{err}");
}

// -- monitor evaluation path with default thresholds --

#[stt_test(
    scheduler = STT_SCHED,
    sockets = 1, cores = 4, threads = 1, memory_mb = 2048,
    watchdog_timeout_jiffies = 60000,
)]
fn cover_monitor_evaluation(ctx: &Ctx) -> Result<VerifyResult> {
    // Exercises the host-side monitor threshold evaluation path in
    // run_stt_test_inner (lines 550-571). The scenario passes, then
    // the host evaluates monitor samples against merged thresholds.
    stt::scenario::basic::custom_sched_mixed(ctx)
}
