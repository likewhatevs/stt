//! Host-side signal-handling integration tests for
//! `ktstr-jemalloc-probe`.
//!
//! These tests spawn the probe binary directly via `Command::new`
//! (no VM) and verify the SIGINT-mid-multi-snapshot contract:
//! the probe responds to SIGINT while sleeping between snapshots,
//! emits a partial ProbeOutput with `interrupted: true`, and exits
//! with status `0` (successful completion of a truncated run is not
//! a failure — the operator got the snapshots they got).
//!
//! The alloc-worker is spawned as the probe's target. YAMA
//! `kernel.yama.ptrace_scope` > 0 rejects attach even under same-uid
//! when the target is not a descendant of the probe; this test
//! SKIPs cleanly on the EPERM/permission error path so CI hosts with
//! YAMA pinned do not flake.
//!
//! Same clean-slate-file rationale as
//! `jemalloc_alloc_worker_exit_codes.rs`: no `#[ktstr_test]` entries
//! here, so the early-dispatch ctor does not hide `#[test]` fns
//! behind the `--list` intercept.

use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

/// Drop-guard around a spawned worker process so a failed test
/// never leaks an orphan parked on `pause()`.
struct WorkerGuard(Option<Child>);
impl Drop for WorkerGuard {
    fn drop(&mut self) {
        if let Some(mut c) = self.0.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

/// Poll the worker's pid-scoped ready marker. The alloc-worker
/// writes `/tmp/ktstr-worker-ready-$PID` via
/// [`ktstr::worker_ready::worker_ready_marker_path`] after its
/// allocation + `black_box` triple completes and before parking on
/// `pause()` — same handshake the in-VM probe tests use.
fn wait_for_worker_ready(pid: i32, timeout: Duration) -> Result<(), String> {
    let marker = ktstr::worker_ready::worker_ready_marker_path(pid as u32);
    let deadline = std::time::Instant::now() + timeout;
    while std::time::Instant::now() < deadline {
        if std::path::Path::new(&marker).exists() {
            // Best-effort cleanup so a future test run targeting
            // the same pid (unlikely but possible under pid reuse)
            // does not consume a stale marker.
            let _ = std::fs::remove_file(&marker);
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    Err(format!("worker pid {pid} never wrote ready marker {marker}"))
}

/// Multi-snapshot run interrupted mid-sleep by SIGINT must exit
/// cleanly with a partial ProbeOutput carrying `interrupted: true`.
///
/// Protocol:
/// 1. Spawn the alloc-worker with 16 MiB, waits for ready marker.
/// 2. Spawn the probe with `--snapshots 20 --interval-ms 300 --json`
///    so the run takes ~6s of wall-clock.
/// 3. Sleep 500ms — roughly one snapshot + start of a sleep.
/// 4. Send SIGINT to the probe. The probe's signal handler sets
///    `CLEANUP_REQUESTED` and `sleep_with_cancel` returns within
///    one poll tick (10ms).
/// 5. Wait for probe exit + parse JSON. Assert:
///    - exit code 0 (success, not signaled)
///    - `interrupted: true`
///    - `snapshots.len() < 20` (interrupt landed before completion)
///    - at least one snapshot emitted (interrupt landed AFTER first)
#[test]
fn probe_sigint_mid_multi_snapshot_produces_partial_output() {
    let worker_bin = env!("CARGO_BIN_EXE_ktstr-jemalloc-alloc-worker");
    let probe_bin = env!("CARGO_BIN_EXE_ktstr-jemalloc-probe");

    let worker = Command::new(worker_bin)
        .arg(format!("{}", 16 * 1024 * 1024))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn alloc-worker");
    let worker_pid = worker.id() as i32;
    let _guard = WorkerGuard(Some(worker));

    if let Err(e) = wait_for_worker_ready(worker_pid, Duration::from_secs(5)) {
        panic!(
            "worker ready marker wait failed: {e}. The worker may have \
             crashed before writing the marker — see jemalloc_alloc_worker \
             exit codes for diagnostics"
        );
    }

    let mut probe = Command::new(probe_bin)
        .arg("--pid")
        .arg(worker_pid.to_string())
        .arg("--snapshots")
        .arg("20")
        .arg("--interval-ms")
        .arg("300")
        .arg("--json")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn probe");
    let probe_pid = probe.id() as i32;

    // Give the probe time to complete 1-2 snapshots and enter an
    // inter-snapshot sleep. Sending SIGINT before the first snapshot
    // would produce snapshots-empty output; this test specifically
    // pins the "interrupt between snapshots" behavior.
    std::thread::sleep(Duration::from_millis(500));

    // Send SIGINT. The probe's handler sets CLEANUP_REQUESTED and
    // sleep_with_cancel observes the flag within its 10ms poll tick.
    unsafe {
        if libc::kill(probe_pid, libc::SIGINT) != 0 {
            panic!(
                "failed to SIGINT probe pid {probe_pid}: {}",
                std::io::Error::last_os_error(),
            );
        }
    }

    let mut stdout_buf = String::new();
    probe
        .stdout
        .as_mut()
        .expect("probe stdout piped")
        .read_to_string(&mut stdout_buf)
        .expect("read probe stdout");
    let mut stderr_buf = String::new();
    let _ = probe
        .stderr
        .as_mut()
        .expect("probe stderr piped")
        .read_to_string(&mut stderr_buf);
    let status = probe.wait().expect("probe wait");

    // Permission gate: PTRACE_SEIZE errors under YAMA ptrace_scope
    // >= 1 produce a Fatal RunOutcome and exit code 1 — NOT the
    // SIGINT-interrupted path. Skip cleanly so CI hosts with YAMA
    // pinned do not flake.
    if !status.success()
        && (stderr_buf.contains("Operation not permitted")
            || stderr_buf.contains("permission")
            || stderr_buf.contains("ptrace")
            || stderr_buf.contains("PTRACE_SEIZE"))
    {
        eprintln!(
            "SKIP: probe could not attach to worker — likely YAMA \
             ptrace_scope > 0 or missing CAP_SYS_PTRACE. stderr:\n{stderr_buf}"
        );
        return;
    }

    assert!(
        status.success(),
        "probe exited non-zero after SIGINT; interrupt path should \
         exit 0 with partial output. status={:?}, stderr:\n{stderr_buf}",
        status.code(),
    );

    let out: serde_json::Value = serde_json::from_str(&stdout_buf).unwrap_or_else(|e| {
        panic!(
            "probe stdout is not valid JSON after SIGINT: {e}. \
             stdout:\n{stdout_buf}\nstderr:\n{stderr_buf}",
        )
    });

    assert_eq!(
        out.get("interrupted").and_then(|v| v.as_bool()),
        Some(true),
        "SIGINT mid-run must set interrupted:true. stdout:\n{stdout_buf}",
    );

    let snapshots = out
        .get("snapshots")
        .and_then(|v| v.as_array())
        .unwrap_or_else(|| panic!("probe output missing snapshots array: {stdout_buf}"));
    assert!(
        !snapshots.is_empty(),
        "at least one snapshot must land before the 500ms SIGINT; got zero. \
         stdout:\n{stdout_buf}",
    );
    assert!(
        snapshots.len() < 20,
        "SIGINT at 500ms into a 6s run must produce fewer than the \
         requested 20 snapshots; got {}. stdout:\n{stdout_buf}",
        snapshots.len(),
    );
}
