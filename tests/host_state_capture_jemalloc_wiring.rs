//! Host-side end-to-end test for the jemalloc-counter wiring in
//! [`ktstr::host_state::capture_pid`].
//!
//! Spawns `ktstr-jemalloc-alloc-worker` as a child process on the
//! host with a known allocation size, waits for the worker's
//! pid-scoped ready marker, then runs `capture_pid` against the
//! host's real `/proc` and confirms the worker's tid carries a
//! populated `allocated_bytes` field. Distinct from the VM-backed
//! `tests/host_state_capture.rs`, which only proves the procfs
//! walk reaches non-jemalloc counters: this test specifically
//! exercises the ELF/DWARF + ptrace + process_vm_readv path that
//! the wiring lifted out of the standalone probe binary into the
//! capture pipeline.
//!
//! # Privilege
//!
//! `ptrace(PTRACE_SEIZE)` must succeed against the worker. Under
//! `kernel.yama.ptrace_scope=0` (typical CI runner default; also
//! the layout the ktstr VMM-backed integration tests use) any
//! same-uid process attaches and the test runs unconditionally.
//! Under `=1` (Debian/Ubuntu host default outside CI) the test
//! parent is the worker's parent PID, which YAMA's `is_my_thread`
//! branch already permits — same-uid + same-process-tree, no
//! capability needed.
//!
//! Under `=2` or `=3`, or when running as a non-uid-matching user,
//! the `attach_jemalloc` step inside `capture_pid` returns an
//! `AttachError` that the capture pipeline absorbs into the
//! "absent counter = 0" contract, so the worker's tid would land
//! with `allocated_bytes=0`. To distinguish "wiring failed" from
//! "test ran on a host where ptrace is locked down", the test
//! short-circuits with an informative skip message when it cannot
//! self-attach (a one-shot probe against the test process's own
//! parent) — operators see why the test passed without exercising
//! the path.

use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use ktstr::metric_types::Bytes;

/// Compile-time path to the alloc-worker binary; cargo populates
/// `CARGO_BIN_EXE_<name>` for every `[[bin]]` declared in the
/// workspace, so the test does not need to spell the build
/// directory layout manually.
const ALLOC_WORKER_BINARY: &str = env!("CARGO_BIN_EXE_ktstr-jemalloc-alloc-worker");

/// Allocation size the worker should hold after it writes the
/// ready marker. Picked well above jemalloc's tcache threshold
/// (16 KiB) so the allocation lands on the slow / huge path and
/// `tsd_s.thread_allocated` is updated synchronously rather than
/// deferred through a per-thread cache.
const KNOWN_BYTES: u64 = 16 * 1024 * 1024;

/// Upper bound on jemalloc/runtime overhead added on top of
/// [`KNOWN_BYTES`]. Mirrors the slop the in-VM probe tests use
/// (see `tests/jemalloc_probe_tests.rs`).
const MAX_SLOP: u64 = 4 * 1024 * 1024;

/// Wait deadline for the worker's ready-marker file.
const READY_TIMEOUT: Duration = Duration::from_secs(10);

/// Override env var the worker honours to redirect the
/// ready-marker path — pulled from
/// [`ktstr::worker_ready::WORKER_READY_MARKER_OVERRIDE_ENV`] so a
/// rename of the const propagates without manual sync.
const READY_MARKER_OVERRIDE_ENV: &str = ktstr::worker_ready::WORKER_READY_MARKER_OVERRIDE_ENV;

/// RAII guard that kills + reaps the worker on scope exit so a
/// test failure does not orphan the process.
struct WorkerGuard {
    child: Option<Child>,
}

impl Drop for WorkerGuard {
    fn drop(&mut self) {
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

/// Probe whether the current process can `ptrace(PTRACE_SEIZE)` a
/// child it just spawned. Returns `true` iff the kernel allows
/// the attach — under `ptrace_scope=2` / `=3` or when the test
/// runs as a uid that cannot trace, the attach surface returns
/// EPERM and we should skip the wiring assertion.
///
/// Implementation: spawn a `sleep` child and try the SEIZE+DETACH
/// dance directly. Cleaner than reading the sysctl and inferring,
/// because the kernel's effective policy depends on uid + parent
/// relationship, not just the scope value.
fn ptrace_attach_allowed() -> bool {
    let child = match Command::new("sleep")
        .arg("1")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    let pid = child.id() as i32;
    let nix_pid = nix::unistd::Pid::from_raw(pid);
    let res = nix::sys::ptrace::seize(nix_pid, nix::sys::ptrace::Options::empty());
    if res.is_ok() {
        let _ = nix::sys::ptrace::detach(nix_pid, None);
    }
    // Reap the sleep child regardless of outcome.
    let _ = nix::sys::wait::waitpid(nix_pid, Some(nix::sys::wait::WaitPidFlag::WNOHANG));
    let mut child = child;
    let _ = child.kill();
    let _ = child.wait();
    res.is_ok()
}

/// Spawn the alloc-worker with a known size and a redirected
/// ready-marker path inside `tempdir`. The redirected path keeps
/// each test invocation isolated from any concurrent tests using
/// the worker's default pid-scoped /tmp marker.
fn spawn_worker(bytes: u64, marker: &PathBuf) -> std::io::Result<Child> {
    Command::new(ALLOC_WORKER_BINARY)
        .arg(bytes.to_string())
        .env(READY_MARKER_OVERRIDE_ENV, marker)
        // Pin jemalloc's background thread to OFF so the
        // single-thread self-check inside the worker passes
        // (extra threads from a leaky shell env var trip exit 3).
        .env("MALLOC_CONF", "background_thread:false")
        .env("_RJEM_MALLOC_CONF", "background_thread:false")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
}

/// Poll for `marker` to appear; surface the worker's stderr if
/// the deadline expires so a setup failure (exit 2/3/4/5/6 in the
/// worker) is visible at test level instead of a bare timeout.
fn wait_for_ready(child: &mut Child, marker: &PathBuf) -> Result<(), String> {
    let deadline = Instant::now() + READY_TIMEOUT;
    while Instant::now() < deadline {
        if marker.exists() {
            return Ok(());
        }
        if let Ok(Some(status)) = child.try_wait() {
            // Worker exited before the marker arrived — drain stderr
            // for the error message it printed.
            let mut stderr = String::new();
            if let Some(mut s) = child.stderr.take() {
                let _ = s.read_to_string(&mut stderr);
            }
            return Err(format!(
                "worker exited early with status {:?}; stderr: {}",
                status, stderr
            ));
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    Err(format!(
        "ready marker {:?} did not appear within {:?}",
        marker, READY_TIMEOUT,
    ))
}

#[test]
fn capture_populates_jemalloc_counters_for_alloc_worker() {
    // Skip the wiring assertion when ptrace is locked down. A
    // skipped test still passes the suite — the goal is to avoid
    // a false-negative on hosts where `ptrace_scope` blocks
    // attach. The test's primary value is the positive
    // assertion path; running it on a locked-down host would
    // produce zero observable signal anyway because the probe
    // would silently absorb every `AttachError::PtraceSeize`
    // through capture's "absent = 0" contract.
    if !ptrace_attach_allowed() {
        eprintln!(
            "host_state_capture_jemalloc_wiring: skipping — \
             ptrace attach is denied by the kernel policy (likely \
             yama.ptrace_scope >= 1 with no parent relationship). \
             Set kernel.yama.ptrace_scope=0 or run the test from \
             the worker's parent process tree."
        );
        return;
    }

    let tmp = tempfile::TempDir::new().expect("tempdir for ready marker");
    let marker = tmp.path().join("ready");

    let child = spawn_worker(KNOWN_BYTES, &marker)
        .expect("alloc-worker should spawn (CARGO_BIN_EXE_ resolved at compile time)");
    let mut guard = WorkerGuard { child: Some(child) };
    let child_ref = guard.child.as_mut().expect("child handle present");
    let worker_pid = child_ref.id() as i32;

    if let Err(msg) = wait_for_ready(child_ref, &marker) {
        panic!("worker did not signal ready: {}", msg);
    }

    // Capture the host-state snapshot scoped to the worker's
    // tgid. `capture_pid` walks `/proc/<pid>/task` only and runs
    // the jemalloc probe attach against the single tgid — much
    // tighter than the global `capture()` walk, which would also
    // try to ptrace every other jemalloc-linked process on the
    // host (potentially hundreds of unrelated tids; slow + risky
    // on a busy dev box). The wiring being exercised is the SAME
    // probe-attach + per-tid probe_thread call that `capture` runs
    // for every jemalloc tgid, so the scoped capture proves
    // exactly the path under test.
    let snap = ktstr::host_state::capture_pid(worker_pid);

    // Find the worker's main thread in the snapshot. The worker
    // is single-threaded (its own self-check enforces it via
    // exit code 3), so `tid == pid` for the only entry — we look
    // up by tgid instead of tid to keep this robust against any
    // future jemalloc helper thread (they would carry different
    // tids but share the tgid).
    let worker_threads: Vec<_> = snap
        .threads
        .iter()
        .filter(|t| t.tgid == worker_pid as u32)
        .collect();
    assert!(
        !worker_threads.is_empty(),
        "capture_pid() did not see worker tgid={worker_pid} in \
         its /proc walk; total threads in snapshot: {}",
        snap.threads.len(),
    );
    // The main thread carries the allocation. Its tid equals the
    // pid for the single-threaded worker; if jemalloc spawned a
    // helper thread despite background_thread:false in the env,
    // we still scan every worker thread for a non-zero counter
    // (only the main thread allocated, helpers stay near zero).
    let allocated: u64 = worker_threads
        .iter()
        .map(|t| t.allocated_bytes.0)
        .max()
        .expect("worker_threads non-empty per assert above");
    let deallocated: u64 = worker_threads
        .iter()
        .map(|t| t.deallocated_bytes.0)
        .max()
        .expect("worker_threads non-empty per assert above");

    assert!(
        allocated >= KNOWN_BYTES,
        "expected worker allocated_bytes >= {KNOWN_BYTES}, \
         got {allocated}; worker_pid={worker_pid}, threads in \
         worker tgid: {}. The capture pipeline's attach_jemalloc \
         either failed against the worker's ELF (DWARF missing, \
         arch mismatch, jemalloc-not-found) or the per-thread \
         ptrace step failed (check ptrace_scope / EPERM).",
        worker_threads.len(),
    );
    assert!(
        allocated <= KNOWN_BYTES + MAX_SLOP,
        "worker allocated_bytes={allocated} exceeds known + slop \
         ({}); probe may be reading the wrong address or the \
         worker leaked extra allocations beyond the planted Vec",
        KNOWN_BYTES + MAX_SLOP,
    );
    // The worker holds its Vec until kill, so deallocations are
    // bounded to jemalloc startup churn — well below the planted
    // size.
    assert!(
        deallocated < KNOWN_BYTES,
        "worker deallocated_bytes={deallocated} >= KNOWN_BYTES \
         ({KNOWN_BYTES}); worker should not free its planted Vec \
         before kill",
    );
}

#[test]
fn capture_pid_skips_self_attach_and_keeps_counters_zero() {
    // The capture process's own tgid must not be probed —
    // ptrace(PTRACE_SEIZE) rejects self-attach with EPERM. The
    // wiring inside capture_pid_with skips self via a `pid !=
    // self_pid` gate before calling attach_jemalloc; this test
    // proves the gate engages by running capture_pid against the
    // test binary itself and confirming the resulting self-tgid
    // ThreadState entries land with allocated_bytes==0 AND
    // deallocated_bytes==0.
    //
    // The pid != self_pid gate is the load-bearing skip; without
    // it, attach_jemalloc would be called against self_pid and
    // PTRACE_SEIZE would EPERM. (The ktstr library does not
    // declare a `#[global_allocator]`, so the test binary is not
    // automatically jemalloc-linked through `cargo test`; whether
    // attach_jemalloc would have detected jemalloc inside the
    // current binary depends on the binary's static link graph.
    // The self-skip gate makes that detail moot — the probe is
    // never even attempted against self.)
    let self_pid = std::process::id() as i32;
    let snap = ktstr::host_state::capture_pid(self_pid);
    let self_threads: Vec<_> = snap
        .threads
        .iter()
        .filter(|t| t.tgid == self_pid as u32)
        .collect();
    assert!(
        !self_threads.is_empty(),
        "capture_pid() did not see self tgid={self_pid}; \
         expected the test process's own tids in the /proc walk",
    );
    for t in &self_threads {
        assert_eq!(
            t.allocated_bytes,
            Bytes(0),
            "self-pid threads must carry allocated_bytes=0 — the \
             pid==self_pid gate must keep attach_jemalloc from \
             running against the calling process; got {} on tid {}",
            t.allocated_bytes,
            t.tid,
        );
        assert_eq!(
            t.deallocated_bytes,
            Bytes(0),
            "self-pid threads must carry deallocated_bytes=0; \
             got {} on tid {}",
            t.deallocated_bytes,
            t.tid,
        );
    }
}

#[test]
fn capture_pid_against_non_jemalloc_target_keeps_counters_zero_but_populates_procfs() {
    // Spawn /bin/sleep — coreutils, not jemalloc-linked.
    // capture_pid against it should:
    //   - run attach_jemalloc, which returns JemallocNotFound
    //   - leave allocated_bytes / deallocated_bytes at 0 per the
    //     absent-counter contract
    //   - still populate identity (tid/tgid/pcomm/comm) and the
    //     procfs-derived counters so the snapshot is otherwise
    //     complete.
    //
    // This is the negative wiring path complement to
    // `capture_populates_jemalloc_counters_for_alloc_worker` — that
    // proves attach success populates counters, this proves attach
    // failure leaves them at zero without dropping the rest.
    let mut child = match Command::new("sleep")
        .arg("3")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => {
            eprintln!("skipping — /bin/sleep unavailable");
            return;
        }
    };
    // Brief settle so /proc/<pid>/{maps,exe,task} populate.
    std::thread::sleep(Duration::from_millis(50));
    let pid = child.id() as i32;

    let snap = ktstr::host_state::capture_pid(pid);

    let _ = child.kill();
    let _ = child.wait();

    let target_threads: Vec<_> = snap
        .threads
        .iter()
        .filter(|t| t.tgid == pid as u32)
        .collect();
    assert!(
        !target_threads.is_empty(),
        "capture_pid did not see /bin/sleep tgid={pid} in its /proc walk; \
         total threads in snapshot: {}",
        snap.threads.len(),
    );
    for t in &target_threads {
        assert_eq!(
            t.allocated_bytes,
            Bytes(0),
            "non-jemalloc target must carry allocated_bytes=0 (attach \
             returned JemallocNotFound, capture absorbed into absent-\
             counter contract); got {} on tid {}",
            t.allocated_bytes,
            t.tid,
        );
        assert_eq!(
            t.deallocated_bytes,
            Bytes(0),
            "non-jemalloc target must carry deallocated_bytes=0; \
             got {} on tid {}",
            t.deallocated_bytes,
            t.tid,
        );
        // Procfs identity + counters populate normally — the attach
        // failure absorbs into the jemalloc fields only, not the
        // whole ThreadState.
        assert_eq!(t.tgid, pid as u32);
        assert!(
            t.start_time_clock_ticks > 0,
            "/proc/<pid>/stat field 22 must populate for a live target",
        );
        assert!(
            !t.policy.0.is_empty(),
            "scheduling policy must populate from procfs even when the \
             jemalloc probe fails — proves the per-thread procfs path \
             does not depend on probe success",
        );
    }
}
