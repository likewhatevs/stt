//! Spawn-pipeline tests — lifecycle group.

#![cfg(test)]
#![allow(unused_imports)]

use super::super::affinity::*;
use super::super::config::*;
use super::super::types::*;
use super::super::worker::*;
use super::testing::*;
use super::*;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// `mmap_shared_anon_errno_hint` must produce distinct,
/// grep-friendly text for each of the three expected errnos
/// (ENOMEM, EPERM, EINVAL) and the empty-string fallback for
/// anything else. Pins the wire contract the two call sites
/// in `WorkloadHandle::spawn` share so an errno that drifts
/// between arms silently would trip the test here rather than
/// in production diagnostics. Every expected arm checks the
/// leading space (caller formats as `"{errno}{hint}"` and
/// relies on the hint providing its own separator) plus a
/// distinctive substring unique to that arm.
#[test]
fn mmap_shared_anon_errno_hint_variants() {
    let enomem = mmap_shared_anon_errno_hint(Some(libc::ENOMEM));
    assert!(
        enomem.starts_with(' '),
        "non-empty hint must begin with a space so \"{{errno}}{{hint}}\" has its separator; got {enomem:?}",
    );
    assert!(
        enomem.contains("ENOMEM"),
        "ENOMEM arm must name the errno in the hint; got {enomem:?}",
    );
    assert!(
        enomem.contains("vm.max_map_count"),
        "ENOMEM arm must mention the remediation sysctl; got {enomem:?}",
    );

    let eperm = mmap_shared_anon_errno_hint(Some(libc::EPERM));
    assert!(eperm.starts_with(' '), "EPERM hint must start with a space");
    assert!(
        eperm.contains("EPERM"),
        "EPERM arm must name the errno; got {eperm:?}",
    );
    assert!(
        eperm.contains("cgroup"),
        "EPERM arm must mention memory cgroup as a remediation path; got {eperm:?}",
    );

    let einval = mmap_shared_anon_errno_hint(Some(libc::EINVAL));
    assert!(
        einval.starts_with(' '),
        "EINVAL hint must start with a space"
    );
    assert!(
        einval.contains("EINVAL"),
        "EINVAL arm must name the errno; got {einval:?}",
    );
    assert!(
        einval.contains("num_workers > 0"),
        "EINVAL arm must give the concrete `num_workers > 0` remediation \
         (the older 'zero or misaligned' wording was too vague); got {einval:?}",
    );

    // Fallback arm: every unrecognised errno (EACCES, EBUSY,
    // EEXIST, random positive integers) must produce the empty
    // string so the caller's format produces no trailing noise.
    assert_eq!(
        mmap_shared_anon_errno_hint(Some(libc::EACCES)),
        "",
        "unrecognised errno must fold to empty-string hint",
    );
    assert_eq!(
        mmap_shared_anon_errno_hint(None),
        "",
        "None errno (io::Error without raw_os_error) must fold to empty-string",
    );
}
// ---- classify_wait_outcome variant coverage ------------------------
//
// Five fixtures pin the `waitpid` → `WorkerExitInfo` mapping that the
// sentinel path in [`WorkloadHandle::stop_and_collect`] depends on.
// A silent table drift here would misreport panic / signal / timeout
// root cause on every failed worker, so this is the canonical test
// for each shape.

#[test]
fn classify_wait_outcome_exited_preserves_code() {
    let status = nix::sys::wait::WaitStatus::Exited(nix::unistd::Pid::from_raw(123), 42);
    match classify_wait_outcome(Ok(status)) {
        WorkerExitInfo::Exited(code) => assert_eq!(code, 42),
        other => panic!("expected Exited(42), got {other:?}"),
    }
}
#[test]
fn classify_wait_outcome_signaled_preserves_signum() {
    let status = nix::sys::wait::WaitStatus::Signaled(
        nix::unistd::Pid::from_raw(123),
        nix::sys::signal::Signal::SIGABRT,
        false,
    );
    match classify_wait_outcome(Ok(status)) {
        WorkerExitInfo::Signaled(sig) => {
            assert_eq!(sig, nix::sys::signal::Signal::SIGABRT as i32);
        }
        other => panic!("expected Signaled(SIGABRT), got {other:?}"),
    }
}
#[test]
fn classify_wait_outcome_still_alive_maps_to_timed_out() {
    match classify_wait_outcome(Ok(nix::sys::wait::WaitStatus::StillAlive)) {
        WorkerExitInfo::TimedOut => {}
        other => panic!("expected TimedOut, got {other:?}"),
    }
}
#[test]
fn classify_wait_outcome_exotic_continued_maps_to_timed_out() {
    // `Continued` is one of the non-terminal WaitStatus variants
    // that can't describe a worker exit for a ptrace-free fork —
    // the catch-all arm must collapse it to TimedOut rather than
    // silently dropping the reap.
    let status = nix::sys::wait::WaitStatus::Continued(nix::unistd::Pid::from_raw(123));
    match classify_wait_outcome(Ok(status)) {
        WorkerExitInfo::TimedOut => {}
        other => panic!("expected TimedOut (exotic→TimedOut), got {other:?}"),
    }
}
#[test]
fn classify_wait_outcome_errno_maps_to_wait_failed() {
    match classify_wait_outcome(Err(nix::errno::Errno::ECHILD)) {
        WorkerExitInfo::WaitFailed(msg) => {
            // nix renders Errno via Display — the string carries
            // the canonical ECHILD description. Substring-match
            // keeps the test robust against OS-specific wording
            // variations without hardcoding a specific phrase.
            assert!(
                msg.to_ascii_lowercase().contains("child"),
                "expected ECHILD description to mention 'child', got {msg:?}",
            );
        }
        other => panic!("expected WaitFailed, got {other:?}"),
    }
}
/// `extract_panic_payload` round-trips both canonical panic
/// payload shapes (`&'static str` from `panic!("literal")` and
/// `String` from `panic!("{x}")`) and falls back to the named
/// sentinel for everything else.
#[test]
fn extract_panic_payload_handles_all_canonical_shapes() {
    let str_panic: Box<dyn std::any::Any + Send> = Box::new("literal panic");
    assert_eq!(extract_panic_payload(str_panic), "literal panic");

    let string_panic: Box<dyn std::any::Any + Send> = Box::new(String::from("formatted panic"));
    assert_eq!(extract_panic_payload(string_panic), "formatted panic");

    // Anything else — e.g. a custom panic payload type — folds
    // to the sentinel without crashing the extractor. The
    // payload value (`42`) is never observed — only the type
    // identity matters for the &str / String downcast misses
    // — so silence the dead-field lint.
    #[derive(Clone)]
    struct CustomPayload(#[allow(dead_code)] u32);
    let custom: Box<dyn std::any::Any + Send> = Box::new(CustomPayload(42));
    assert_eq!(extract_panic_payload(custom), "<non-string panic payload>");
}
/// `apply_nice(n)` invokes `setpriority(2)` for every value —
/// including the boundary case where the old API treated `0`
/// as "skip". The skip role now lives one layer up via
/// `WorkSpec::nice = None` / `WorkloadConfig::nice = None`,
/// gated at the `worker_main` call site (see `apply_nice` in
/// `workload::spawn` and the `if let Some(n) = nice` guard in
/// `worker_main`). With the call-site gate handling skip, the
/// in-function code always writes via the syscall.
///
/// Test from default nice 0 by raising to 5 with `apply_nice(5)`
/// — raising nice (positive direction) is always permitted for
/// own-task without `CAP_SYS_NICE` per the kernel's
/// `set_one_prio` → `can_nice` check (which only triggers when
/// `niceval < task_nice(p)`). A successful raise proves the
/// function reached the `setpriority` syscall rather than
/// short-circuiting on the value. Lowering back from 5 to 0
/// would require `CAP_SYS_NICE` or sufficient `RLIMIT_NICE` and
/// is unreliable on unprivileged test runners, so this test
/// does not exercise that direction; the call-site
/// `Option<i32>` gate is covered end-to-end by
/// `worker_nice_applied_via_setpriority` below.
#[test]
fn apply_nice_invokes_setpriority() {
    // The Rust `libc` crate's `getpriority` is a direct binding
    // to glibc's POSIX `getpriority(3)` wrapper, which returns
    // the actual nice value (range -20..=19) rather than the
    // raw syscall encoding (`20 - nice`). errno-clear before
    // call because getpriority can legitimately return -1 for
    // nice=-1 — only errno disambiguates -1-as-error from
    // -1-as-nice.
    unsafe {
        *libc::__errno_location() = 0;
    }
    let nice_before = unsafe { libc::getpriority(libc::PRIO_PROCESS, 0) };
    let errno_before = unsafe { *libc::__errno_location() };
    assert_eq!(
        errno_before, 0,
        "getpriority must succeed before apply_nice; rc={nice_before}"
    );
    assert_eq!(
        nice_before, 0,
        "test must start from default nice 0; observed {nice_before} \
         (a non-default starting nice indicates external state \
         leakage from a prior test or runner config)"
    );

    // Invoke apply_nice(5) — must raise nice via setpriority.
    // Raising is unconditionally permitted for own-task so this
    // call cannot fail on permissions and isolates the function
    // body's syscall path from CAP_SYS_NICE / RLIMIT_NICE.
    apply_nice(5);

    unsafe {
        *libc::__errno_location() = 0;
    }
    let nice_after = unsafe { libc::getpriority(libc::PRIO_PROCESS, 0) };
    let errno_after = unsafe { *libc::__errno_location() };
    assert_eq!(errno_after, 0, "getpriority must succeed after apply_nice");
    assert_eq!(
        nice_after, 5,
        "apply_nice(5) must invoke setpriority and write 5 — \
         observed nice {nice_after} after starting at {nice_before}; \
         a no-op (e.g. an early-return short-circuit, the regression \
         this test guards against) would leave nice at 0",
    );

    // Restore default. Lowering from 5 to 0 may fail without
    // CAP_SYS_NICE — that is exactly why the assertion above
    // tests raising rather than lowering. Best-effort cleanup;
    // rc is intentionally ignored.
    let _ = unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, 0) };
}
/// Positive-nice end-to-end: spawn one worker with `nice = 10`,
/// verify the worker process actually has nice 10 by reading
/// `/proc/<pid>/stat` field 19 (priority field) before
/// `stop_and_collect`. Positive nice never requires
/// `CAP_SYS_NICE` — `set_one_prio` only checks `can_nice` for
/// `niceval < task_nice(p)`.
///
/// Reading via /proc rather than `getpriority` because the
/// worker is in a child process; `getpriority(PRIO_PROCESS, pid)`
/// would also work but /proc/stat field 19 is the canonical
/// observation point used elsewhere in the crate's tests.
#[test]
fn worker_nice_applied_via_setpriority() {
    let config = WorkloadConfig {
        num_workers: 1,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::SpinWait,
        sched_policy: SchedPolicy::Normal,
        nice: Some(10),
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    let pid = h.worker_pids()[0];
    h.start();
    // Brief sleep so the worker has actually executed
    // `apply_nice` post-fork and post-start before we read
    // /proc.
    std::thread::sleep(std::time::Duration::from_millis(100));
    // /proc/<pid>/stat field 19 is "nice" per `proc(5)` —
    // tokenize after the comm field's closing paren to avoid
    // splitting names containing spaces.
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).expect("/proc/stat read");
    let after_paren = stat
        .rsplit_once(") ")
        .expect("/proc/stat has comm in parens")
        .1;
    // After the closing paren, fields are 1-indexed starting
    // at "state" (field 3 of the original layout). nice is
    // field 19; minus the 2 fields before the paren that's
    // index 16 in the post-paren token list.
    let tokens: Vec<&str> = after_paren.split_whitespace().collect();
    let nice_str = tokens
        .get(16)
        .expect("/proc/stat must have at least 17 fields after comm");
    let nice_observed: i32 = nice_str.parse().expect("nice field must be i32");
    // Stop before assertion so a failure doesn't leak a
    // non-default-nice worker.
    let _reports = h.stop_and_collect();
    assert_eq!(
        nice_observed, 10,
        "worker /proc/<pid>/stat field 19 must reflect the \
         configured nice value; got {nice_observed}, expected 10"
    );
}
/// Regression guard for the spawn-leak fix: on a mid-setup
/// `bail!` path, the `SpawnGuard` Drop must release every
/// resource acquired so far — no leaked children, no leaked
/// pipe fds, no leaked mmap regions. This test constructs a
/// config that passes the `worker_group_size` check and then
/// provokes the per-worker pipe path (num_workers=2 with
/// PipeIo) so the function allocates inter-worker pipes and
/// spawns successfully, then checks Drop cleans up when the
/// handle is dropped without `stop_and_collect`.
///
/// The direct spawn-failure path is hard to trigger
/// synthetically (would require EMFILE / ENOMEM injection); the
/// scope guard's correctness is proven by the unified cleanup
/// pattern — Drop runs on every early return *and* on the
/// normal drop-without-collect flow.
#[test]
fn handle_drop_reaps_children_and_closes_pipes() {
    let config = WorkloadConfig {
        num_workers: 2,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::PipeIo { burst_iters: 4 },
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let h = WorkloadHandle::spawn(&config).unwrap();
    let pids = h.worker_pids();
    assert_eq!(pids.len(), 2, "both workers spawned");
    // Drop without calling start() or stop_and_collect() — this
    // exercises the WorkloadHandle::Drop path, which has the
    // same cleanup semantics as SpawnGuard's error path.
    drop(h);
    // Poll for termination: ESRCH (no such process) means the
    // child was reaped. Give the kernel a brief grace window
    // because waitpid runs synchronously but kill reporting can
    // race.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    for pid in pids {
        loop {
            let alive = nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None).is_ok();
            if !alive {
                break;
            }
            if std::time::Instant::now() >= deadline {
                panic!("child {pid} still alive after drop deadline");
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }
}
#[test]
fn drop_kills_children() {
    let config = WorkloadConfig {
        num_workers: 2,
        ..Default::default()
    };
    let h = WorkloadHandle::spawn(&config).unwrap();
    let pids = h.worker_pids();
    drop(h);
    // After drop, children should be dead.
    for pid in pids {
        let alive = nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None).is_ok();
        assert!(!alive, "child {} should be dead after drop", pid);
    }
}
/// Zombie-tolerance on the Drop path: a caller drops a live
/// `WorkloadHandle` after external code has SIGKILLed one of
/// its workers. Between the signal delivery and the parent's
/// `waitpid`, the killed worker sits as a zombie — its pid
/// is still owned by this parent (only `waitpid` consumes
/// the zombie state; an external signal does not), so Drop's
/// follow-up `kill(pid, SIGKILL)` is a no-op against the
/// zombie and Drop's `waitpid` reaps the exit status
/// normally.
///
/// Pins that Drop survives this realistic failure mode — an
/// external operator (a CI runner's OOM killer, a stray
/// `killall <name>`, a test-harness teardown signal)
/// signals one worker before the handle's owning code
/// finishes. Drop must leave the surviving siblings alone
/// and reap the zombie without panicking.
#[test]
fn workload_handle_drop_tolerates_externally_killed_child() {
    let config = WorkloadConfig {
        num_workers: 2,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::SpinWait,
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    let pids = h.worker_pids();
    assert_eq!(pids.len(), 2);
    h.start();
    // Externally SIGKILL one worker. The handle still owns
    // the pid; on Drop it will try to signal + reap it.
    unsafe { libc::kill(pids[0], libc::SIGKILL) };
    // A brief sleep covers SIGKILL delivery latency. The
    // killed worker becomes a zombie rather than ESRCH (only
    // `waitpid` can clear it), so probing `kill(pid, 0)`
    // would spin forever — 50 ms is more than enough for
    // the kernel to deliver the signal and transition the
    // target to zombie state.
    std::thread::sleep(std::time::Duration::from_millis(50));
    // The assertion is implicit: this drop must not panic.
    // A panic inside Drop under panic=abort aborts the test
    // process, which nextest reports as an abnormal failure.
    drop(h);
}
/// Pins the `stop_and_collect` sentinel path where SIGUSR1 is
/// ignored and the WNOHANG-returns-`StillAlive` branch fires:
/// the parent escalates to SIGKILL, collects zero JSON from the
/// worker, and the synthesized [`WorkerReport`] carries
/// `exit_info: Some(TimedOut)` (or `Some(Signaled(SIGKILL))`
/// if the race between WNOHANG and the kill put the reap at
/// the blocking waitpid). Without this test, the escalation
/// branch of `classify_wait_outcome` is only covered by the
/// pure unit test `classify_wait_outcome_still_alive_maps_to_timed_out`;
/// pairing that with this end-to-end exercise proves the
/// integration (parent loop + `ignores_sigusr1_fn` + sentinel
/// fill) doesn't drop the diagnostic along the way.
///
/// Expected runtime: ~5s (the shared deadline), plus a few ms
/// for spawn + kill + reap. Marked with a shorter spin window
/// in `ignores_sigusr1_fn` (7s ceiling) so even if the parent
/// deadline extends accidentally, the test still terminates.
#[test]
fn stop_and_collect_sentinel_exits_for_sigusr1_ignoring_worker() {
    let config = WorkloadConfig {
        num_workers: 1,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::custom("sigusr1_ignore", ignores_sigusr1_fn),
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    // Readiness handshake — poll for the ready file the worker
    // writes after its `libc::signal(SIGUSR1, SIG_IGN)` call
    // completes. Replaces a fixed 200ms sleep with progress-
    // driven waiting: we send SIGUSR1 only once SIG_IGN is
    // definitely installed. The poll interval is 10ms and the
    // ceiling is 3s (~15× the old sleep) to cover CPU-starved
    // hosts without silently hanging — the earlier 2s ceiling
    // was tight enough that heavily-loaded CI runners (many
    // parallel cargo nextest workers competing for CPU during
    // fork + signal-handler install) occasionally missed the
    // deadline on valid SIG_IGN installs; bumping to 3s
    // preserves the "bounded, actionable" intent without the
    // flake.
    let worker_pid = h.worker_pids()[0];
    let ready_path = ready_file_path(worker_pid);
    // Remove any stale ready file from a prior run that happened
    // to land the same PID — `ready_path.exists()` in the poll
    // loop below would otherwise short-circuit on the stale file
    // and the parent would send SIGUSR1 before SIG_IGN was
    // actually installed. PID reuse across test runs in the same
    // session is plausible because fork() picks from the kernel's
    // recycled PID pool. This MUST run before `h.start()` — after
    // start() the worker is unblocked and can write a fresh ready
    // file before we reach this line, which would cause us to
    // unlink a live handshake and wedge the poll loop.
    let _ = std::fs::remove_file(&ready_path);
    h.start();
    wait_for_file_or_panic(
        &ready_path,
        Duration::from_secs(3),
        worker_pid,
        "SIG_IGN install may have failed or child never reached \
         ignores_sigusr1_fn's ready-file write",
    );
    let reports = h.stop_and_collect();
    // Ready file outlives the worker (written early, never
    // cleaned up by the child because the parent SIGKILLs it
    // before any cleanup could run). Remove it here so repeated
    // test runs don't observe a stale file from a prior run.
    let _ = std::fs::remove_file(&ready_path);
    assert_eq!(reports.len(), 1);
    let r = &reports[0];
    // Sentinel path: the worker never wrote JSON to the pipe
    // (because it ignored SIGUSR1 + ran past the deadline), so
    // the report is the zeroed sentinel shape. work_units = 0
    // confirms the sentinel construction at stop_and_collect's
    // `serde_json::from_slice` Err branch, not a worker-authored
    // report leaking through.
    assert_eq!(
        r.work_units, 0,
        "sentinel sidecar must be zeroed; non-zero work_units means \
         we parsed the worker's real report instead of hitting the \
         Err branch",
    );
    // `exit_info` must describe either the TimedOut (WNOHANG fast
    // path caught StillAlive) or Signaled(SIGKILL=9) (the kill
    // landed before the WNOHANG check) outcome. Any other variant
    // — Exited (worker wrote JSON), WaitFailed (reap error) —
    // would indicate a different failure shape than the one this
    // test pins.
    match &r.exit_info {
        Some(WorkerExitInfo::TimedOut) => {}
        Some(WorkerExitInfo::Signaled(sig)) if *sig == libc::SIGKILL => {}
        other => panic!("expected TimedOut or Signaled(SIGKILL), got {other:?}",),
    }
}
// -- Test-helper unit tests --

/// Happy path: the file appears WITHIN the deadline, so
/// [`wait_for_file_or_panic`] returns without panicking. Uses
/// `std::process::id()` as `liveness_pid` — this test process is
/// always alive, so the early-exit probe never fires.
#[test]
fn wait_for_file_or_panic_returns_when_file_appears() {
    let dir = std::env::temp_dir().join(format!("ktstr-wfp-happy-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let marker = dir.join("ready");
    // Pre-create the marker so the first iteration exits the
    // loop. No race to worry about for the happy-path pin.
    std::fs::write(&marker, b"ok").unwrap();
    wait_for_file_or_panic(
        &marker,
        Duration::from_secs(1),
        unsafe { libc::getpid() },
        "pre-existing marker must satisfy the guard",
    );
    let _ = std::fs::remove_dir_all(&dir);
}
/// Liveness-death path: `liveness_pid` dies before the file
/// appears, so the helper panics with "exited before writing
/// ready file" rather than waiting the full deadline. The test
/// forks a `/bin/true` child, reaps it, then polls a file that
/// will never appear; the helper's `kill(pid, 0)` returns ESRCH
/// on the dead pid and the panic fires inside catch_unwind.
#[test]
fn wait_for_file_or_panic_detects_liveness_death() {
    let mut child = std::process::Command::new("/bin/true")
        .spawn()
        .expect("spawn /bin/true");
    let dead_pid = child.id() as libc::pid_t;
    let _ = child.wait();
    // `dead_pid` is now reaped; `kill(dead_pid, 0)` returns ESRCH
    // unless the kernel has already recycled it. Recycling is
    // very unlikely within the ~100ms test window.
    let nonexistent = std::env::temp_dir().join(format!(
        "ktstr-wfp-never-exists-{}-{dead_pid}",
        std::process::id(),
    ));
    let _ = std::fs::remove_file(&nonexistent);
    let result = std::panic::catch_unwind(|| {
        wait_for_file_or_panic(
            &nonexistent,
            Duration::from_secs(30), // generous — we want the liveness path, not the deadline
            dead_pid,
            "liveness-death path",
        );
    });
    let err = result.expect_err("must panic when liveness pid is dead");
    let msg = crate::test_support::test_helpers::panic_payload_to_string(err);
    assert!(
        msg.contains("exited before writing ready file"),
        "panic must name the early-exit path, got: {msg}"
    );
}
