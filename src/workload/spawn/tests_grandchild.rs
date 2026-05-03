//! Spawn-pipeline tests — grandchild group.

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

/// Proves the `setpgid(0, 0)` + `killpg` path works end-to-end:
/// a long-running grandchild forked from a Custom worker's
/// closure dies when stop_and_collect runs. Without setpgid +
/// killpg, the grandchild would orphan onto init and survive the
/// test — which this test catches via `kill(gpid, 0)` returning
/// ESRCH after collection.
///
/// The SIGUSR1 ignore forces stop_and_collect into its StillAlive
/// escalation branch. This test pins the StillAlive path. The
/// graceful-exit branch (stop_and_collect's `waited` arm where the
/// worker exits before the 5s deadline) is pinned by TWO variants
/// covering the disjoint shapes a worker can die in before the
/// parent reaps it:
///   - [`stop_and_collect_reaps_grandchild_from_panicking_custom_closure`]
///     — worker panics → process dies via `_exit(1)` (under
///     `panic = "unwind"`) or SIGABRT (under `panic = "abort"`)
///     BEFORE stop_and_collect even signals it. The graceful
///     branch's `waited` result is `Exited(1)` / `Signaled(SIGABRT)`
///     on that path; the unconditional killpg must still reach
///     the grandchild.
///   - [`stop_and_collect_reaps_grandchild_from_graceful_custom_closure`]
///     — worker's inherited SIGUSR1 handler fires and flips STOP,
///     the closure returns a clean WorkerReport, the worker
///     `_exit(0)`s WITHIN the deadline. The graceful branch's
///     `waited` is `Exited(0)`; the same unconditional killpg
///     must still reap the grandchild.
///
/// The Drop branch is pinned by
/// [`drop_reaps_custom_grandchild_via_process_group`] (handle is
/// dropped with no stop_and_collect call → `impl Drop`'s killpg
/// sweeps). The multi-worker variant is
/// [`stop_and_collect_reaps_grandchildren_from_multiple_workers`].
#[test]
fn stop_and_collect_reaps_custom_grandchild_via_process_group() {
    require_grandchild_sleep_binary();
    let config = WorkloadConfig {
        num_workers: 1,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::custom("grandchild_sleep", forks_grandchild_sleep_fn),
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    let worker_pid = h.worker_pids()[0];
    let pidfile = grandchild_pidfile_path(worker_pid);
    let _ = std::fs::remove_file(&pidfile);
    // Pidfile cleanup fires via the module-level PidfileCleanup
    // helper — Drop removes the stub even if later assertions
    // panic.
    let _pidfile_cleanup = PidfileCleanup(vec![pidfile.clone()]);
    h.start();
    let gpid = read_grandchild_gpid_from_pidfile(worker_pid, &pidfile);
    // Confirm grandchild is alive before stop_and_collect.
    assert!(
        nix::sys::signal::kill(nix::unistd::Pid::from_raw(gpid), None).is_ok(),
        "grandchild {gpid} must be alive before stop_and_collect",
    );
    // Trigger the teardown that should also reap the grandchild.
    let _reports = h.stop_and_collect();
    assert_grandchild_reaped_within(gpid, Duration::from_secs(5), "stop_and_collect");
}
/// Multi-worker variant of
/// [`stop_and_collect_reaps_custom_grandchild_via_process_group`]:
/// `num_workers = 3`, each worker forks its own grandchild, and
/// `stop_and_collect` must reap all three process groups. Guards
/// against a future refactor that accidentally single-target's
/// killpg (e.g. only the first child in
/// `WorkloadHandle::children`).
#[test]
fn stop_and_collect_reaps_grandchildren_from_multiple_workers() {
    require_grandchild_sleep_binary();
    const N: usize = 3;
    let config = WorkloadConfig {
        num_workers: N,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::custom("grandchild_sleep", forks_grandchild_sleep_fn),
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    let worker_pids = h.worker_pids();
    assert_eq!(
        worker_pids.len(),
        N,
        "WorkloadHandle::worker_pids should report {N} workers",
    );
    // Pin uniqueness: every worker must have a distinct pid. A
    // repeated pid would mean the spawn logic conflated two
    // workers (or the pidfile scheme collides across workers,
    // which would also break this multi-worker reaping test).
    let unique: std::collections::HashSet<libc::pid_t> = worker_pids.iter().copied().collect();
    assert_eq!(
        unique.len(),
        worker_pids.len(),
        "WorkloadHandle::worker_pids returned duplicates: {worker_pids:?}",
    );
    let pidfiles: Vec<std::path::PathBuf> = worker_pids
        .iter()
        .map(|&p| grandchild_pidfile_path(p))
        .collect();
    for p in &pidfiles {
        let _ = std::fs::remove_file(p);
    }
    let _pidfile_cleanup = PidfileCleanup(pidfiles.clone());
    h.start();
    // Collect every grandchild pid; any pidfile miss panics with
    // the worker_pid context embedded so the failure names the
    // offending worker.
    let gpids: Vec<libc::pid_t> = worker_pids
        .iter()
        .zip(pidfiles.iter())
        .map(|(&wp, pf)| read_grandchild_gpid_from_pidfile(wp, pf))
        .collect();
    for &gpid in &gpids {
        assert!(
            nix::sys::signal::kill(nix::unistd::Pid::from_raw(gpid), None).is_ok(),
            "grandchild {gpid} must be alive before stop_and_collect",
        );
    }
    let _reports = h.stop_and_collect();
    for &gpid in &gpids {
        assert_grandchild_reaped_within(
            gpid,
            Duration::from_secs(5),
            "stop_and_collect (multi-worker)",
        );
    }
}
/// Panic-path variant: the Custom closure panics after forking
/// its grandchild. Under `panic = "unwind"` the worker's
/// `std::panic::catch_unwind` (around the child body in the
/// forked-child path of `WorkloadHandle::spawn`) catches the
/// panic and the child hits `libc::_exit(1)` directly — no
/// abort. Under `panic = "abort"`
/// SIGABRT fires before catch_unwind runs. Either way the
/// parent-worker process exits BEFORE `stop_and_collect` is
/// called; stop_and_collect's graceful-exit branch must still
/// issue killpg to reach the grandchild. Pins the unconditional
/// killpg in the graceful branch — without it, the grandchild
/// would orphan onto init.
#[test]
fn stop_and_collect_reaps_grandchild_from_panicking_custom_closure() {
    require_grandchild_sleep_binary();
    let config = WorkloadConfig {
        num_workers: 1,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::custom("grandchild_panic", forks_grandchild_and_panics_fn),
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    let worker_pid = h.worker_pids()[0];
    let pidfile = grandchild_pidfile_path(worker_pid);
    let _ = std::fs::remove_file(&pidfile);
    let _pidfile_cleanup = PidfileCleanup(vec![pidfile.clone()]);
    h.start();
    // The worker panics immediately after publishing the gpid;
    // read_grandchild_gpid_from_pidfile observes the file before
    // the worker process finishes exiting because fork + panic
    // is slower than the tempfile + rename write.
    let gpid = read_grandchild_gpid_from_pidfile(worker_pid, &pidfile);
    assert!(
        nix::sys::signal::kill(nix::unistd::Pid::from_raw(gpid), None).is_ok(),
        "grandchild {gpid} must be alive before stop_and_collect",
    );
    let reports = h.stop_and_collect();
    assert_grandchild_reaped_within(
        gpid,
        Duration::from_secs(5),
        "stop_and_collect (panic-path)",
    );
    // Sentinel-mapping audit: the panicking worker cannot
    // serialize a WorkerReport to the pipe, so
    // `stop_and_collect`'s JSON-parse branch must fall into
    // the sentinel path. The `exit_info` carried on the
    // sentinel depends on the compile-time panic strategy:
    //   - Under `panic = "abort"` (release profile), the
    //     panic raises SIGABRT before the worker's
    //     `catch_unwind` can run → `Signaled(SIGABRT)`.
    //   - Under `panic = "unwind"` (dev/test profile, which
    //     this test runs under), the worker's `catch_unwind`
    //     intercepts the panic and calls `libc::_exit(1)` →
    //     `Exited(1)`.
    // Both paths produce a sentinel with `work_units == 0`;
    // the match below accepts either.
    assert_eq!(reports.len(), 1, "one worker spawned");
    let r = &reports[0];
    assert_eq!(
        r.work_units, 0,
        "sentinel must be zeroed; non-zero work_units would mean \
         a worker-authored report leaked through the JSON-parse \
         branch despite the panic",
    );
    assert!(
        !r.completed,
        "sentinel must carry completed=false so downstream \
         consumers distinguish '0 iterations by design / fast \
         exit' (completed=true) from '0 iterations because the \
         worker crashed before producing a report' (this case); \
         `..WorkerReport::default()` gives the bool-default \
         `false` at the sentinel construction site in \
         `stop_and_collect`",
    );
    match &r.exit_info {
        Some(WorkerExitInfo::Signaled(sig)) if *sig == libc::SIGABRT => {}
        Some(WorkerExitInfo::Exited(1)) => {}
        other => panic!(
            "expected sentinel with Signaled(SIGABRT) (panic=abort) \
             or Exited(1) (panic=unwind + catch_unwind) for a \
             panicking Custom closure; got {other:?}",
        ),
    }
}
/// Drop-path variant: the caller drops the handle WITHOUT calling
/// `stop_and_collect`. The `impl Drop for WorkloadHandle`
/// (src/workload.rs) is responsible for killpg'ing every worker
/// process group, then SIGKILLing each leader and waitpid'ing it.
/// Without the Drop-path killpg, any long-running grandchild
/// would orphan onto init and leak past the test. Pins the
/// Drop-path sweep.
#[test]
fn drop_reaps_custom_grandchild_via_process_group() {
    require_grandchild_sleep_binary();
    let config = WorkloadConfig {
        num_workers: 1,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::custom("grandchild_sleep", forks_grandchild_sleep_fn),
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    let worker_pid = h.worker_pids()[0];
    let pidfile = grandchild_pidfile_path(worker_pid);
    let _ = std::fs::remove_file(&pidfile);
    let _pidfile_cleanup = PidfileCleanup(vec![pidfile.clone()]);
    h.start();
    let gpid = read_grandchild_gpid_from_pidfile(worker_pid, &pidfile);
    assert!(
        nix::sys::signal::kill(nix::unistd::Pid::from_raw(gpid), None).is_ok(),
        "grandchild {gpid} must be alive before Drop",
    );
    // No stop_and_collect call — Drop is the sole teardown path
    // under test here. `drop(h)` triggers the impl Drop killpg +
    // kill + waitpid sweep.
    drop(h);
    assert_grandchild_reaped_within(
        gpid,
        Duration::from_secs(5),
        "handle Drop (no stop_and_collect)",
    );
}
/// Graceful-exit variant: the Custom closure forks a grandchild,
/// publishes the pidfile, and waits on `stop` at 10ms granularity
/// — no SIG_IGN, no panic. The worker's inherited `SIGUSR1 → STOP`
/// handler fires when `stop_and_collect` signals us, the closure
/// returns a clean `WorkerReport`, and the worker exits cleanly
/// WITHIN the 5s collection deadline. That routes stop_and_collect
/// into its `waited` / graceful-exit branch (not StillAlive, not
/// Drop). The unconditional killpg on THAT branch is the path
/// under test — without it, the grandchild would orphan onto
/// init.
#[test]
fn stop_and_collect_reaps_grandchild_from_graceful_custom_closure() {
    require_grandchild_sleep_binary();
    let config = WorkloadConfig {
        num_workers: 1,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::custom("grandchild_graceful", forks_grandchild_and_exits_cleanly_fn),
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    let worker_pid = h.worker_pids()[0];
    let pidfile = grandchild_pidfile_path(worker_pid);
    let _ = std::fs::remove_file(&pidfile);
    let _pidfile_cleanup = PidfileCleanup(vec![pidfile.clone()]);
    h.start();
    let gpid = read_grandchild_gpid_from_pidfile(worker_pid, &pidfile);
    assert!(
        nix::sys::signal::kill(nix::unistd::Pid::from_raw(gpid), None).is_ok(),
        "grandchild {gpid} must be alive before stop_and_collect",
    );
    // Pin which `stop_and_collect` branch fires. The graceful path
    // — worker's SIGUSR1 handler flips STOP, the closure returns
    // cleanly via `wait_for_deadline`'s stop-observed early-exit,
    // the worker `_exit(0)`s well within the 5s collection
    // deadline — completes in a few hundred milliseconds
    // (500ms auto-start sleep + SIGUSR1 + 10ms wait_for_deadline
    // poll + worker serialize/_exit + WNOHANG reap). The
    // StillAlive escalation branch, by contrast, waits the full
    // 5s deadline before SIGKILL. A <2s ceiling rules out
    // StillAlive escalation (~5s+) while leaving generous slack
    // for CI contention on the graceful path.
    let t0 = Instant::now();
    let _reports = h.stop_and_collect();
    let elapsed = t0.elapsed();
    assert!(
        elapsed < Duration::from_secs(2),
        "stop_and_collect must hit the graceful-exit branch \
         (<2s), not StillAlive escalation (~5s). elapsed={elapsed:?} \
         — a value near the 5s deadline means SIGUSR1 failed to \
         reach the worker or wait_for_deadline did not observe \
         STOP in time",
    );
    assert_grandchild_reaped_within(
        gpid,
        Duration::from_secs(5),
        "stop_and_collect (graceful-exit)",
    );
}
