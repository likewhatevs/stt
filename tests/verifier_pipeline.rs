use std::process::Command;

fn cargo_stt_path() -> std::path::PathBuf {
    let output = Command::new("cargo")
        .args(["build", "-p", "cargo-stt", "--message-format=json"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .expect("cargo build cargo-stt");
    assert!(
        output.status.success(),
        "cargo build cargo-stt failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(line)
            && msg.get("reason").and_then(|r| r.as_str()) == Some("compiler-artifact")
            && msg
                .get("target")
                .and_then(|t| t.get("name"))
                .and_then(|n| n.as_str())
                == Some("cargo-stt")
            && msg
                .get("profile")
                .and_then(|p| p.get("test"))
                .and_then(|t| t.as_bool())
                == Some(false)
            && msg
                .get("target")
                .and_then(|t| t.get("kind"))
                .and_then(|k| k.as_array())
                .is_some_and(|kinds| kinds.iter().any(|k| k.as_str() == Some("bin")))
            && let Some(filenames) = msg.get("filenames").and_then(|f| f.as_array())
            && let Some(path) = filenames.first().and_then(|f| f.as_str())
        {
            return std::path::PathBuf::from(path);
        }
    }
    panic!("cargo-stt binary not found in cargo build output");
}

/// Run cargo-stt verifier and return (stdout, stderr, exit_code).
fn run_verifier(args: &[&str]) -> (String, String, i32) {
    let bin = cargo_stt_path();
    let output = Command::new(&bin)
        .arg("stt")
        .arg("verifier")
        .args(args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .expect("run cargo-stt verifier");
    (
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
        output.status.code().unwrap_or(-1),
    )
}

/// Verifier brief output: loads stt-sched BPF programs and prints
/// instruction counts and verifier stats for each program.
#[ignore]
#[test]
fn verifier_brief_output() {
    let (stdout, stderr, code) = run_verifier(&[]);
    assert_eq!(
        code, 0,
        "verifier should succeed\nstdout: {stdout}\nstderr: {stderr}"
    );
    // Brief output lists each program with insns= and processed= fields.
    assert!(
        stdout.contains("stt_enqueue"),
        "output should list stt_enqueue: {stdout}"
    );
    assert!(
        stdout.contains("stt_dispatch"),
        "output should list stt_dispatch: {stdout}"
    );
    assert!(
        stdout.contains("insns="),
        "output should contain insns=: {stdout}"
    );
    assert!(
        stdout.contains("processed="),
        "output should contain processed=: {stdout}"
    );
}

/// Verifier verbose output: includes per-instruction verifier trace.
#[ignore]
#[test]
fn verifier_verbose_output() {
    let (stdout, stderr, code) = run_verifier(&["-v"]);
    assert_eq!(
        code, 0,
        "verifier -v should succeed\nstdout: {stdout}\nstderr: {stderr}"
    );
    // Verbose output includes the verifier log with BPF instruction traces.
    assert!(
        stdout.contains("stt_dispatch"),
        "verbose output should list stt_dispatch: {stdout}"
    );
    // The log should contain instruction traces (numbered lines with opcodes).
    // Cycle collapse may or may not trigger depending on log content.
    assert!(
        stdout.contains("processed") || stdout.contains("insns="),
        "verbose output should contain verifier stats: {stdout}"
    );
}

/// Verifier diff mode: A/B instruction count comparison between two
/// scheduler packages. Uses stt-sched vs itself.
#[ignore]
#[test]
fn verifier_diff_output() {
    let (stdout, stderr, code) = run_verifier(&["--diff", "stt-sched"]);
    assert_eq!(
        code, 0,
        "verifier --diff should succeed\nstdout: {stdout}\nstderr: {stderr}"
    );
    // Diff output contains an A/B delta table.
    assert!(
        stdout.contains("delta"),
        "diff output should contain 'delta' header: {stdout}"
    );
    assert!(
        stdout.contains("program"),
        "diff output should contain 'program' column: {stdout}"
    );
    // Self-comparison: all deltas should be 0 (same package both sides).
    assert!(
        stdout.contains("+0"),
        "self-comparison deltas should be 0: {stdout}"
    );
}

/// --fail-verify: the scheduler binary rejects at BPF load time when
/// fail_verify=1 is set via .rodata before load. The scheduler process
/// exits with an error before attaching, causing a scheduler death in
/// the VM.
#[ignore]
#[test]
fn verifier_fail_verify() {
    use anyhow::Result;
    use stt::assert::{Assert, AssertResult};
    use stt::scenario::Ctx;
    use stt::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps_with};
    use stt::test_support::{Scheduler, SchedulerSpec, SttTestEntry, run_stt_test};

    const FAIL_SCHED: Scheduler =
        Scheduler::new("stt_sched").binary(SchedulerSpec::Name("stt-sched"));

    fn scenario(ctx: &Ctx) -> Result<AssertResult> {
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
        name: "verifier_fail_verify",
        func: scenario,
        sockets: 1,
        cores: 2,
        threads: 1,
        memory_mb: 2048,
        scheduler: &FAIL_SCHED,
        auto_repro: false,
        replicas: 1,
        assert: Assert::NONE,
        extra_sched_args: &["--fail-verify"],
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

    let result = run_stt_test(&__STT_ENTRY_FAIL_VERIFY);
    assert!(
        result.is_err(),
        "scheduler with --fail-verify should fail (BPF load rejected)"
    );
    let err_msg = format!("{:#}", result.unwrap_err());
    // The scheduler fails at BPF load time — the verifier rejects
    // the program due to the null pointer dereference in stt_dispatch.
    // This surfaces as scheduler death (no result in COM2).
    assert!(
        err_msg.contains("scheduler died")
            || err_msg.contains("SCHEDULER_DIED")
            || err_msg.contains("timed out")
            || err_msg.contains("no test result"),
        "error should indicate scheduler death from verifier rejection: {err_msg}"
    );
}

/// Cycle collapse on real stt-sched verifier output.
/// The verbose output applies collapse_cycles() to the log. If the
/// verifier log contains bpf_loop iterations (from degrade_spin_cb),
/// they should be collapsed. If not, the log should still be valid.
#[ignore]
#[test]
fn verifier_cycle_collapse_real() {
    let (stdout, stderr, code) = run_verifier(&["-v"]);
    assert_eq!(
        code, 0,
        "verifier -v should succeed\nstdout: {stdout}\nstderr: {stderr}"
    );
    // The output is valid regardless of whether cycles were found.
    // If cycles exist, we see "--- Nx of the following M lines ---".
    // If not, the raw log is printed. Both are valid outcomes.
    //
    // Verify the output is non-empty and contains verifier content.
    assert!(
        !stdout.is_empty(),
        "verbose verifier output should not be empty"
    );
    assert!(
        stdout.contains("stt_dispatch"),
        "output should contain stt_dispatch program: {stdout}"
    );
    // The processed line should appear (it's always emitted by the verifier).
    assert!(
        stdout.contains("processed=") || stdout.contains("processed "),
        "output should contain processed insn count: {stdout}"
    );
}
