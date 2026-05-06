//! Shared test fixtures for the spawn-pipeline test files. Holds
//! grandchild-reaping helpers (PidfileCleanup, forks_grandchild_*),
//! the SIGUSR1-ignore worker, lifecycle helpers (`wait_for_deadline`,
//! `wait_for_file_or_panic`), and the per-test `spawn_and_collect_after`
//! collector. Imported from each `tests_*.rs` sibling via
//! `use super::testing::*;`.

#![cfg(test)]
// `tests_*.rs` siblings glob-import these fixtures via
// `use super::testing::*;` and each file uses only a topical
// subset. Without the allow, fixtures unused by the importing
// file would warn even though every fixture is used by at
// least one test file in this directory. The audit alternative
// (per-fixture imports in every tests_*.rs) trades 13 file
// churn points for the warning, with no behavioral payoff.
#![allow(dead_code)]

use super::super::affinity::*;
use super::super::config::*;
use super::*;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

pub(super) fn spawn_and_collect_after(
    work_type: WorkType,
    num_workers: usize,
    sleep_ms: u64,
) -> Vec<WorkerReport> {
    let config = WorkloadConfig {
        num_workers,
        affinity: AffinityIntent::Inherit,
        work_type,
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    h.start();
    std::thread::sleep(std::time::Duration::from_millis(sleep_ms));
    h.stop_and_collect()
}
// -- SpawnGuard failure-injection tests --
//
// These exercise the error-path cleanup that the unified
// `handle_drop_reaps_children_and_closes_pipes` test explicitly
// noted it could not cover: the mid-spawn bail paths reached when
// a syscall inside `WorkloadHandle::spawn` fails with EMFILE
// (RLIMIT_NOFILE) or EAGAIN (RLIMIT_NPROC). Each case forks a
// helper subprocess so `setrlimit` scope is confined to that
// child and the parent test binary's limits stay intact.
//
// Cleanup check strategy:
//   - Count open fds via `/proc/self/fd/` before and after the
//     failed `spawn`. After SpawnGuard::Drop, the fd count must
//     return to baseline (all pipe pairs, report pipes, and start
//     pipes released).
//   - Poll `waitpid(-1, WNOHANG)` to prove no zombie worker
//     children were left behind by a partial fork.
//
// Child exit code convention:
//   0  = success (spawn returned Err AND cleanup is clean)
//   10 = spawn unexpectedly returned Ok (failure not triggered)
//   11 = fd leak detected after SpawnGuard::Drop
//   12 = zombie worker process detected after SpawnGuard::Drop
//   13 = setrlimit itself failed (harness issue, not a test
//        failure of the guard)
//   14 = bail arrived via an unexpected branch (test picks the
//        wrong failure path)
//   15 = post-bail setrlimit raise failed (harness issue; would
//        mask a genuine fd leak as a false positive)
//   other nonzero = unrelated failure (panic, assertion miss)
//
// `libc::_exit` is used instead of `std::process::exit` in the
// child so Rust's global destructors — shared with the parent
// test binary through the fork's copied state — do not fire.

/// Count open file descriptors for the calling process by
/// listing `/proc/self/fd/`. The directory iterator itself holds
/// one fd while open; the snapshot is taken after the iterator
/// drops, so the count reflects steady state.
pub(super) fn count_open_fds() -> usize {
    std::fs::read_dir("/proc/self/fd")
        .map(|d| d.count())
        .unwrap_or(0)
}
/// Non-blocking reap of any exited children. Returns true when a
/// child reported via waitpid(-1, WNOHANG), indicating an
/// orphaned-but-not-reaped zombie remained after `spawn`'s error
/// path. SpawnGuard::Drop reaps everything it forked; any
/// positive return here is a guard bug.
pub(super) fn any_zombie_child() -> bool {
    let mut status = 0i32;
    let ret = unsafe { libc::waitpid(-1, &mut status, libc::WNOHANG) };
    ret > 0
}
/// Lower RLIMIT_NPROC to the current process count so any `fork`
/// in this child returns -1 with EAGAIN. Returns true on success.
pub(super) fn set_rlimit_nproc_zero_headroom() -> bool {
    // Setting rlim_cur to 1 would block even our own existing
    // thread spawns; setting it to the current process's uid
    // usage is what reliably triggers EAGAIN on the next fork.
    // getrusage does not expose that counter; instead use a
    // small value just high enough for the ktstr test binary's
    // baseline and no more. Empirically, setting rlim_cur == 0
    // causes fork to return EAGAIN because the kernel rejects
    // the new-process creation against the per-uid cap.
    let rl = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    unsafe { libc::setrlimit(libc::RLIMIT_NPROC, &rl) == 0 }
}
/// Fork a helper subprocess that lowers its own rlimits, runs
/// the provided test body, and exits with the body's result
/// code. Parent waits for child and returns the child's exit
/// code. Any nonzero code from the child indicates a guard
/// cleanup defect or harness issue — see exit-code convention
/// comment above.
pub(super) fn run_in_forked_child<F: FnOnce() -> i32>(body: F) -> i32 {
    let pid = unsafe { libc::fork() };
    assert!(pid >= 0, "fork failed: {}", std::io::Error::last_os_error());
    if pid == 0 {
        // Child: install a silent panic hook so an assertion
        // failure inside the body doesn't multiplex stderr with
        // the parent's test output. Then run the body, which
        // returns an exit code. `_exit` skips Rust destructors
        // so the parent's resources copied via fork are not
        // double-closed.
        //
        // `catch_unwind` + `unwrap_or(99)` is effective here
        // because this helper is gated under `#[cfg(test)]` and
        // the dev/test profile inherits default unwind
        // semantics. Under `[profile.release]`'s `panic =
        // "abort"` the catch_unwind would be a no-op and a panic
        // in `body` would SIGABRT the child — which the parent's
        // signal-code path (`100 + WTERMSIG`) still surfaces
        // distinctly from the 99 fallback, so the exit-code
        // convention above remains self-consistent either way.
        let _ = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let code = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)).unwrap_or(99);
        unsafe { libc::_exit(code) };
    }
    let mut status: libc::c_int = 0;
    let waited = unsafe { libc::waitpid(pid, &mut status, 0) };
    assert_eq!(
        waited,
        pid,
        "waitpid({pid}) failed: {}",
        std::io::Error::last_os_error()
    );
    if libc::WIFEXITED(status) {
        libc::WEXITSTATUS(status)
    } else {
        // Terminated by signal — surface the signal number
        // as a large exit code so the parent's assertion can
        // distinguish it from the body's own codes.
        100 + libc::WTERMSIG(status)
    }
}
// -- Custom work type tests --

pub(super) fn stub_custom_fn(_stop: &AtomicBool) -> WorkerReport {
    WorkerReport {
        tid: 0,
        work_units: 0,
        cpu_time_ns: 0,
        wall_time_ns: 0,
        off_cpu_ns: 0,
        migration_count: 0,
        cpus_used: BTreeSet::new(),
        migrations: vec![],
        max_gap_ms: 0,
        max_gap_cpu: 0,
        max_gap_at_ms: 0,
        resume_latencies_ns: vec![],
        wake_sample_total: 0,
        iteration_costs_ns: vec![],
        iteration_cost_sample_total: 0,
        iterations: 0,
        schedstat_run_delay_ns: 0,
        schedstat_run_count: 0,
        schedstat_cpu_time_ns: 0,
        completed: true,
        numa_pages: BTreeMap::new(),
        vmstat_numa_pages_migrated: 0,
        exit_info: None,
        is_messenger: false,
        group_idx: 0,
        affinity_error: None,
    }
}
pub(super) fn custom_spin_fn(stop: &AtomicBool) -> WorkerReport {
    let tid: libc::pid_t = unsafe { libc::getpid() };
    let start = Instant::now();
    let mut work_units = 0u64;
    while !stop_requested(stop) {
        work_units = std::hint::black_box(work_units.wrapping_add(1));
        std::hint::spin_loop();
    }
    let wall_time_ns = start.elapsed().as_nanos() as u64;
    WorkerReport {
        tid,
        work_units,
        cpu_time_ns: 0,
        wall_time_ns,
        off_cpu_ns: 0,
        migration_count: 0,
        cpus_used: BTreeSet::new(),
        migrations: vec![],
        max_gap_ms: 0,
        max_gap_cpu: 0,
        max_gap_at_ms: 0,
        resume_latencies_ns: vec![],
        wake_sample_total: 0,
        iteration_costs_ns: vec![],
        iteration_cost_sample_total: 0,
        iterations: work_units,
        schedstat_run_delay_ns: 0,
        schedstat_run_count: 0,
        schedstat_cpu_time_ns: 0,
        completed: true,
        numa_pages: BTreeMap::new(),
        vmstat_numa_pages_migrated: 0,
        exit_info: None,
        is_messenger: false,
        group_idx: 0,
        affinity_error: None,
    }
}
/// Ready-file path shared between [`ignores_sigusr1_fn`] and
/// `stop_and_collect_sentinel_exits_for_sigusr1_ignoring_worker`.
/// The worker writes a zero-byte file at this path after
/// installing `SIG_IGN` for SIGUSR1; the parent polls for the
/// file's appearance before sending SIGUSR1, eliminating the
/// race the old 200ms sleep papered over.
pub(super) fn ready_file_path(pid: libc::pid_t) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("ktstr-sigusr1-ignore-ready-{pid}"))
}
/// Shared post-fork prologue for test WorkType closures: installs
/// `SIG_IGN` for SIGUSR1 so stop_and_collect cannot flip STOP via
/// the signal path, then returns the current pid (which doubles as
/// the worker's tid on Linux because [`WorkloadHandle::spawn`]
/// forks one process per worker). Factored out of the two custom
/// closures that share this opening; both forks land in a
/// single-threaded child where `libc::signal` is safe.
pub(super) fn ignore_sigusr1_and_get_pid() -> libc::pid_t {
    unsafe {
        libc::signal(libc::SIGUSR1, libc::SIG_IGN);
    }
    unsafe { libc::getpid() }
}
/// Sleep-based deadline loop shared by the SIGUSR1-ignoring test
/// closures. Returns when either `stop` flips (SIGUSR1 handler
/// path, never fires under SIG_IGN — kept honest) or `timeout`
/// elapses. Takes a [`Duration`] to match
/// [`wait_for_file_or_panic`]'s signature; callers that want to
/// spell the value as "seven seconds" still write
/// `Duration::from_secs(7)`.
///
/// Uses `thread::sleep(10ms)` rather than `spin_loop()`: the
/// closures' purpose is to outlive stop_and_collect's 5s
/// collection deadline, not to respond to cache-coherent store
/// visibility at CPU speed, so a ~100x lower CPU footprint is
/// strictly better under CI contention.
pub(super) fn wait_for_deadline(stop: &AtomicBool, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    while !stop_requested(stop) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
}
/// Poll for `path`'s appearance with a deadline, aborting early if
/// `liveness_pid` dies before the file is written. `kill(pid, 0)` is
/// the POSIX existence probe — Err means the pid is gone (or the
/// caller is not permitted to signal it, which for a pid owned by
/// this test process implies the pid has already been reaped).
/// Panics with an actionable message on either early-death or
/// deadline. `context` is appended to the panic text so the caller
/// can pin the failure to a specific test scenario.
pub(super) fn wait_for_file_or_panic(
    path: &std::path::Path,
    timeout: Duration,
    liveness_pid: libc::pid_t,
    context: &str,
) {
    let deadline = Instant::now() + timeout;
    while !path.exists() {
        if nix::sys::signal::kill(nix::unistd::Pid::from_raw(liveness_pid), None).is_err() {
            panic!("pid {liveness_pid} exited before writing ready file {path:?} — {context}",);
        }
        if Instant::now() >= deadline {
            panic!(
                "pid {liveness_pid} did not write ready file {path:?} within {timeout:?} — {context}",
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}
/// Worker function that installs `SIG_IGN` for SIGUSR1 — overriding
/// the `sigusr1_handler` the child set up post-fork — and spins
/// for long enough to outlive the parent's 5s collection deadline.
/// Used by the sigusr1-ignored path test below.
///
/// `libc::signal(SIGUSR1, SIG_IGN)` replaces the handler on the
/// child's process-wide disposition table, so the parent's
/// `kill(pid, SIGUSR1)` arrives as a no-op — STOP never flips to
/// true via the handler, and even code that checks STOP spins
/// past the deadline.
pub(super) fn ignores_sigusr1_fn(stop: &AtomicBool) -> WorkerReport {
    let tid = ignore_sigusr1_and_get_pid();
    // SIG_IGN is now installed. Clear any STOP set by the
    // framework's handler during the handshake window (between
    // mask unblock and this point). This worker deliberately
    // ignores SIGUSR1 — the parent must escalate to SIGKILL.
    stop.store(false, Ordering::Relaxed);
    // Readiness handshake: after SIG_IGN is installed, write a
    // zero-byte ready file so the parent can proceed without
    // waiting on a fixed-duration sleep. Without the handshake
    // the parent had to guess a safe delay (200ms) covering
    // fork + signal(2) syscalls plus CPU contention —
    // too short and the parent's SIGUSR1 races the handler
    // replacement and the test fails spuriously. See
    // `stop_and_collect_sentinel_exits_for_sigusr1_ignoring_worker`
    // below for the reader side.
    let ready_path = ready_file_path(tid);
    let _ = std::fs::write(&ready_path, []);
    // Wait 7s — well past stop_and_collect's 5s shared deadline.
    // The `!stop.load` check is kept honest inside
    // `wait_for_deadline` (no infinite loop) but is only
    // observed via the fallback timeout: with SIG_IGN in place,
    // the parent's SIGUSR1 doesn't flip STOP.
    wait_for_deadline(stop, Duration::from_secs(7));
    // Report body is never observed — the parent SIGKILLs the
    // worker before any `f.write_all(&json)` could run. Per the
    // `WorkerReport` doc, sentinel-shape constructions use
    // `..Default::default()` so a future field addition doesn't
    // silently drift the test.
    WorkerReport {
        tid,
        ..WorkerReport::default()
    }
}
/// Shared path helper for [`forks_grandchild_sleep_fn`] and the
/// grandchild reaping tests below. Workers write their forked-
/// grandchild pid here so the test can observe it without fragile
/// pipe-based IPC.
pub(super) fn grandchild_pidfile_path(worker_pid: libc::pid_t) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("ktstr-grandchild-pid-{worker_pid}"))
}
/// Path to the grandchild exec target used by every reaping test.
/// Pinned here (rather than inlined in the `execv` call sites) so
/// the test-side existence guard
/// [`require_grandchild_sleep_binary`] and the worker-side
/// `execv(prog, argv)` cannot drift.
pub(super) const GRANDCHILD_SLEEP_BINARY: &str = "/bin/sleep";
/// Panic with an actionable message if `GRANDCHILD_SLEEP_BINARY`
/// is missing or not marked executable (any of the user / group /
/// other x-bits set). Every grandchild reaping test
/// `execv(/bin/sleep, …)` after fork; a missing or non-executable
/// binary causes the exec to fail and the grandchild to
/// `_exit(127)` before the parent can read the pidfile, which then
/// trips [`wait_for_file_or_panic`] with a generic timeout that
/// buries the real cause. Failing here first keeps the diagnostic
/// specific.
pub(super) fn require_grandchild_sleep_binary() {
    use std::os::unix::fs::PermissionsExt;
    let path = std::path::Path::new(GRANDCHILD_SLEEP_BINARY);
    let meta = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(e) => panic!(
            "grandchild reaping tests require {GRANDCHILD_SLEEP_BINARY} to \
             exist; stat failed: {e}. Install coreutils (or adjust the \
             test's exec target + update GRANDCHILD_SLEEP_BINARY)."
        ),
    };
    // 0o111 covers all three x-bits (user / group / other). execv(2)
    // only requires one of them to be set AND match the caller's
    // effective uid / gid / other, but a file with zero x-bits
    // cannot be executed by anyone; catch that clear case here.
    // A finer-grained check would need `faccessat(X_OK)`; the
    // coarse check is sufficient for the "coreutils forgot to
    // mark /bin/sleep executable" failure mode this guard exists
    // to catch.
    if meta.permissions().mode() & 0o111 == 0 {
        panic!(
            "grandchild reaping tests require {GRANDCHILD_SLEEP_BINARY} to \
             have at least one execute bit set; mode = {:o}. Fix the \
             file's permissions or adjust the test's exec target.",
            meta.permissions().mode() & 0o7777,
        );
    }
}
/// Block on `pidfile` until it holds a parseable `libc::pid_t` and
/// return it. Combines [`wait_for_file_or_panic`] + the
/// retry-on-empty reader used by every grandchild reaping test
/// (tempfile + rename write-atomicity sometimes races reads on
/// slower filesystems or under heavy contention, so the reader
/// guards anyway). Panics with an actionable message on timeout,
/// empty-file stall, or parse failure.
pub(super) fn read_grandchild_gpid_from_pidfile(
    worker_pid: libc::pid_t,
    pidfile: &std::path::Path,
) -> libc::pid_t {
    wait_for_file_or_panic(
        pidfile,
        Duration::from_secs(3),
        worker_pid,
        "fork+exec path likely broken — check /bin/sleep exists and is executable",
    );
    let read_deadline = Instant::now() + Duration::from_secs(3);
    let gpid_str = loop {
        let s = std::fs::read_to_string(pidfile).expect("pidfile readable once exists");
        if !s.trim().is_empty() {
            break s;
        }
        if Instant::now() >= read_deadline {
            panic!(
                "pidfile {pidfile:?} stayed empty for 3s after exists() \
                 returned true — writer may have crashed between O_TRUNC \
                 and write",
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let gpid: libc::pid_t = gpid_str
        .trim()
        .parse()
        .expect("pidfile holds a valid pid_t");
    assert!(gpid > 0, "grandchild pid must be positive: {gpid}");
    gpid
}
/// Poll for `gpid` death with a bounded deadline. Returns `Ok(())`
/// when the pid is gone (ESRCH on the existence probe) and
/// `Err(())` on timeout. The waitpid + WNOHANG inside the loop
/// reaps a zombie if the caller inherited the grandchild under
/// `PR_SET_CHILD_SUBREAPER` (systemd-run scopes, some CI
/// runners). Shared by
/// [`stop_and_collect_reaps_custom_grandchild_via_process_group`]
/// and the new multi-worker / panic-path / Drop-path tests.
pub(super) fn wait_for_grandchild_reap(gpid: libc::pid_t, timeout: Duration) -> Result<(), ()> {
    let deadline = Instant::now() + timeout;
    loop {
        match nix::sys::signal::kill(nix::unistd::Pid::from_raw(gpid), None) {
            Err(nix::errno::Errno::ESRCH) => return Ok(()),
            Err(e) => panic!(
                "unexpected errno from existence probe: {e} \
                 (common non-ESRCH errnos: EPERM = caller may not \
                 signal this process despite it existing; EINVAL = \
                 invalid signal number, which cannot happen here \
                 since we pass None / signal 0)",
            ),
            Ok(()) => {
                match nix::sys::wait::waitpid(
                    nix::unistd::Pid::from_raw(gpid),
                    Some(nix::sys::wait::WaitPidFlag::WNOHANG),
                ) {
                    Ok(nix::sys::wait::WaitStatus::Exited(_, _))
                    | Ok(nix::sys::wait::WaitStatus::Signaled(_, _, _)) => return Ok(()),
                    _ => {}
                }
                if Instant::now() >= deadline {
                    return Err(());
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    }
}
/// Last-resort SIGKILL + assertion-panic wrapper around
/// [`wait_for_grandchild_reap`]. Ensures a test failure never
/// leaks a live grandchild into the host.
pub(super) fn assert_grandchild_reaped_within(gpid: libc::pid_t, timeout: Duration, context: &str) {
    if wait_for_grandchild_reap(gpid, timeout).is_err() {
        let _ = nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(gpid),
            nix::sys::signal::Signal::SIGKILL,
        );
        panic!(
            "grandchild {gpid} still alive {:?} after {context} — \
             setpgid/killpg path broken",
            timeout,
        );
    }
}
/// RAII pidfile cleanup: removes the file on Drop so a panicking
/// test doesn't leak a `/tmp/ktstr-grandchild-pid-*` stub into
/// the host. Manual impl rather than `scopeguard` to keep the
/// crate out of the workspace dep graph.
pub(super) struct PidfileCleanup(pub(super) Vec<std::path::PathBuf>);
impl Drop for PidfileCleanup {
    fn drop(&mut self) {
        for p in &self.0 {
            let _ = std::fs::remove_file(p);
        }
    }
}
/// Shared post-fork-and-exec helper used by every grandchild
/// reaping test closure. In the parent-worker: forks a
/// [`GRANDCHILD_SLEEP_BINARY`] 60 grandchild via `execv`, publishes
/// the gpid atomically via tempfile + rename, and returns the
/// worker's own pid. In the child: `execv(prog, [prog, "60", NULL])`
/// followed by `_exit(127)` on exec failure — `execv` requires
/// `argv[0]` to carry the program name by convention so the
/// exec'd `/bin/sleep` sees its usual `argv[0]`. Never returns on the
/// child side.
///
/// Does NOT install any SIGUSR1 disposition — callers pick the
/// policy (SIG_IGN to force StillAlive escalation, or the
/// inherited SIGUSR1→STOP handler for graceful-exit). CString
/// construction runs pre-fork so a hypothetical NulError fires in
/// the parent where it's debuggable. The tempfile + rename
/// protocol closes the exists()→read() race the reader-side
/// retry loop also defends against.
pub(super) fn fork_and_exec_grandchild_and_publish_pidfile() -> libc::pid_t {
    let exec_path = std::ffi::CString::new(GRANDCHILD_SLEEP_BINARY)
        .expect("GRANDCHILD_SLEEP_BINARY must have no interior NUL");
    let exec_arg = std::ffi::CString::new("60").expect("literal has no NUL");
    let worker_pid = unsafe { libc::getpid() };
    let gpid = unsafe { libc::fork() };
    if gpid < 0 {
        // _exit is async-signal-safe; eprintln goes to the
        // harness-captured test log.
        eprintln!("fork failed: {}", std::io::Error::last_os_error());
        unsafe {
            libc::_exit(127);
        }
    }
    if gpid == 0 {
        // Close every inherited fd above stdio BEFORE exec so
        // the grandchild does not keep the parent-worker's
        // pipes open. The worker's report-pipe write end is
        // especially load-bearing: if the grandchild inherits
        // it, the test's parent-side `read_to_end` in
        // `stop_and_collect` blocks on EOF until the
        // grandchild itself dies, turning a fast graceful-exit
        // test into a /bin/sleep-wall-clock-long run
        // (observed: 60s).
        //
        // `close_range(3, u32::MAX, 0)` is the one-syscall form
        // (Linux 5.9+) and is the fast path. BUT this code
        // runs on the HOST, not inside the ktstr guest VM —
        // ktstr's 6.16+ kernel floor applies to the sched_ext
        // guest kernel, not to the host running the tests. A
        // host kernel predating 5.9 returns ENOSYS from
        // `close_range`, leaving every inherited fd open and
        // re-introducing the 60s hang. Fall back to the
        // bounded `3..=256` close loop on any non-zero return
        // so pre-5.9 hosts still close the load-bearing
        // report-pipe write end.
        let rc = unsafe { libc::close_range(3, u32::MAX, 0) };
        if rc != 0 {
            for fd in 3..=256 {
                unsafe {
                    libc::close(fd);
                }
            }
        }
        // Grandchild: exec immediately. `execv` returns only on
        // failure; any return is a setup error → _exit(127).
        // CStrings live on the child's CoW'd heap from the
        // parent; pointers stay valid until execv replaces the
        // address space.
        let argv: [*const libc::c_char; 3] =
            [exec_path.as_ptr(), exec_arg.as_ptr(), std::ptr::null()];
        unsafe {
            libc::execv(exec_path.as_ptr(), argv.as_ptr());
            libc::_exit(127);
        }
    }
    // Parent-worker: publish gpid. A failure here leaves the test
    // hanging on a file that never appears — surface the errno
    // and exit so the test gets an actionable diagnostic instead
    // of a poll-timeout panic.
    let pidfile = grandchild_pidfile_path(worker_pid);
    let pidfile_tmp = std::env::temp_dir().join(format!("ktstr-grandchild-pid-{worker_pid}.tmp"));
    if let Err(e) = std::fs::write(&pidfile_tmp, gpid.to_string()) {
        eprintln!("failed to write grandchild pidfile tmp {pidfile_tmp:?}: {e}");
        unsafe {
            libc::_exit(127);
        }
    }
    if let Err(e) = std::fs::rename(&pidfile_tmp, &pidfile) {
        eprintln!("failed to rename grandchild pidfile {pidfile_tmp:?} → {pidfile:?}: {e}");
        unsafe {
            libc::_exit(127);
        }
    }
    worker_pid
}
/// Custom WorkType closure that forks a long-running grandchild
/// and ignores `SIGUSR1` on the parent-worker side so
/// stop_and_collect is forced into its StillAlive escalation
/// branch. Pairs with
/// [`stop_and_collect_reaps_custom_grandchild_via_process_group`].
pub(super) fn forks_grandchild_sleep_fn(stop: &AtomicBool) -> WorkerReport {
    // Ignore SIGUSR1 so stop_and_collect escalates — matches
    // ignores_sigusr1_fn's rationale.
    let worker_pid = ignore_sigusr1_and_get_pid();
    fork_and_exec_grandchild_and_publish_pidfile();
    // Wait past the 5s collection deadline so stop_and_collect
    // escalates to SIGKILL → killpg. The `!stop.load` check is
    // kept honest inside `wait_for_deadline` even though SIG_IGN
    // prevents SIGUSR1 from flipping STOP; the 7s deadline is
    // the real terminator.
    wait_for_deadline(stop, Duration::from_secs(7));
    WorkerReport {
        tid: worker_pid,
        ..WorkerReport::default()
    }
}
/// Graceful-exit variant: forks the grandchild and then waits on
/// the `stop` flag via [`wait_for_deadline`]. Does NOT install
/// SIG_IGN — the worker's inherited `SIGUSR1 → STOP` handler
/// fires on stop_and_collect's signal and flips `stop`, letting
/// this closure return cleanly BEFORE the 5s collection deadline.
/// stop_and_collect therefore hits its graceful-exit branch;
/// killpg on that branch must still reap the grandchild.
///
/// 10s upper bound on the wait is purely a liveness sentinel —
/// stop_and_collect sends SIGUSR1 within milliseconds of its
/// own invocation, so in practice `stop` flips well before 10s
/// elapses.
pub(super) fn forks_grandchild_and_exits_cleanly_fn(stop: &AtomicBool) -> WorkerReport {
    let worker_pid = fork_and_exec_grandchild_and_publish_pidfile();
    wait_for_deadline(stop, Duration::from_secs(10));
    WorkerReport {
        tid: worker_pid,
        ..WorkerReport::default()
    }
}
/// Custom closure that forks a grandchild exactly like
/// [`forks_grandchild_sleep_fn`], publishes the gpid via the
/// same pidfile protocol, then deliberately panics. Exercises the
/// Custom-closure panic path — the worker process unwinds /
/// aborts without a clean `WorkerReport` return, but the
/// `setpgid(0, 0)` it installed at fork time still applies, so
/// `stop_and_collect`'s unconditional killpg must still reap the
/// grandchild.
pub(super) fn forks_grandchild_and_panics_fn(_stop: &AtomicBool) -> WorkerReport {
    // SIG_IGN so a racing SIGUSR1 from stop_and_collect cannot
    // trip the default worker handler before the panic fires;
    // the panic + catch_unwind → _exit(1) path is what this
    // closure exists to exercise, not the graceful SIGUSR1 flow.
    let _worker_pid = ignore_sigusr1_and_get_pid();
    fork_and_exec_grandchild_and_publish_pidfile();
    panic!(
        "intentional panic after grandchild fork to exercise the \
         Custom-closure panic path in stop_and_collect"
    );
}
/// Skip a multi-stage WakeChain test when the host advertises
/// fewer than `min_cpus` parallel execution units. Bootstrap
/// throughput tests below pin per-stage rates against
/// `work_per_hop`-bounded ceilings; if the host serialises
/// stages onto a single CPU, scheduler jitter dominates and
/// the per-stage throughput collapses below the lower bound,
/// flaking the test on contended runners. `available_parallelism`
/// reads `sched_getaffinity` (per `std::thread` docs), so a
/// nextest invocation with `--test-threads` or a cpuset-pinned
/// runner reports its constrained budget — exactly the signal
/// these tests need.
///
/// Returns `true` to indicate the caller should `return`
/// immediately without running the test body. Uses `eprintln!`
/// to surface the skip in nextest output (matches the
/// `set_mempolicy: ... skipping` precedent at sites like
/// `apply_mempolicy_with_flags`); `panic!` would fail the
/// test rather than skip it, contradicting the "skip on
/// insufficient CPUs" contract.
pub(super) fn require_isolated_cpus(min_cpus: usize, test_name: &str) -> bool {
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    if available < min_cpus {
        eprintln!(
            "ktstr: {test_name}: skipping — host reports \
             available_parallelism={available}, test requires \
             ≥ {min_cpus} CPUs to keep stages on independent \
             execution units"
        );
        return true;
    }
    false
}
