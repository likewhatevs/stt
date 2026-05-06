//! Spawn-pipeline tests — thread_mode group.

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

/// `join_thread_with_timeout` returns the join result when the
/// thread completes within the deadline. The exit eventfd is
/// bumped from inside the closure to mirror production's
/// `WorkerExitSignal` Drop guard.
#[test]
fn join_thread_with_timeout_returns_result_on_quick_completion() {
    use std::sync::Arc;
    let exit_evt = Arc::new(vmm_sys_util::eventfd::EventFd::new(libc::EFD_NONBLOCK).unwrap());
    let exit_evt_thread = Arc::clone(&exit_evt);
    let join = std::thread::spawn(move || {
        let _ = exit_evt_thread.write(1);
        WorkerReport {
            tid: 7,
            ..WorkerReport::default()
        }
    });
    let r = join_thread_with_timeout(join, &exit_evt, Duration::from_secs(2));
    match r {
        Some(Ok(report)) => assert_eq!(report.tid, 7),
        Some(Err(_)) => panic!("clean thread must not produce join Err"),
        None => panic!("clean thread must not time out within 2s"),
    }
}
/// `join_thread_with_timeout` returns `None` when the thread is
/// still running past the deadline. The thread itself leaks for
/// the rest of the test process — acceptable in a `#[test]`
/// because the test harness terminates after the thread's
/// upper-bound sleep.
#[test]
fn join_thread_with_timeout_returns_none_on_timeout() {
    use std::sync::Arc;
    let exit_evt = Arc::new(vmm_sys_util::eventfd::EventFd::new(libc::EFD_NONBLOCK).unwrap());
    let join = std::thread::spawn(|| {
        // Sleep WELL past the 100ms timeout so the polling
        // helper definitely observes is_finished()==false.
        std::thread::sleep(Duration::from_millis(800));
        WorkerReport::default()
    });
    let r = join_thread_with_timeout(join, &exit_evt, Duration::from_millis(100));
    assert!(r.is_none(), "100ms timeout vs 800ms thread must time out");
}
/// Defense-in-depth: `ThreadWorker::drop` MUST join its
/// `JoinHandle`. Rust's std `JoinHandle::drop` detaches by
/// default — the bug class this test exists to catch is a
/// future refactor that lets a `ThreadWorker` fall out of
/// scope without going through the `WorkloadHandle::drop`
/// / `stop_and_collect` / `SpawnGuard::drop` paths that
/// already explicitly take + join.
///
/// The test constructs a `ThreadWorker` whose worker writes a
/// shared flag and waits on a stop signal, drops the
/// `ThreadWorker` directly (NOT via any of the explicit Drop
/// paths), and verifies the worker observed `stop=true` and
/// completed before the drop returned. If `ThreadWorker::drop`
/// detached, the worker would still be running when the test
/// returns — the spin-loop on the shared flag confirms a
/// successful join.
#[test]
fn thread_worker_drop_joins_handle() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
    use std::sync::mpsc;

    let stop = Arc::new(AtomicBool::new(false));
    let observed = Arc::new(AtomicBool::new(false));
    let stop_thread = Arc::clone(&stop);
    let observed_thread = Arc::clone(&observed);
    let (start_tx, start_rx) = mpsc::sync_channel::<()>(0);
    let tid = Arc::new(AtomicI32::new(0));
    let tid_thread = Arc::clone(&tid);
    let exit_evt = Arc::new(vmm_sys_util::eventfd::EventFd::new(libc::EFD_NONBLOCK).unwrap());
    let exit_evt_thread = Arc::clone(&exit_evt);

    let join = std::thread::spawn(move || {
        tid_thread.store(
            unsafe { libc::syscall(libc::SYS_gettid) as libc::pid_t },
            Ordering::Relaxed,
        );
        // Block on start so the worker is guaranteed to be
        // running (not just dispatched) by the time we drop.
        let _ = start_rx.recv();
        // Spin on stop with the same 100ms poll cadence the
        // production worker uses.
        while !stop_thread.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(20));
        }
        observed_thread.store(true, Ordering::Relaxed);
        let _ = exit_evt_thread.write(1);
        WorkerReport::default()
    });

    let tw = ThreadWorker {
        tid,
        stop,
        start_tx: Some(start_tx),
        join: Some(join),
        exit_evt,
    };
    // Send the start signal so the worker proceeds to its
    // stop-check loop. (The Drop will also drop start_tx but
    // that comes after recv() has consumed our send.)
    if let Some(ref tx) = tw.start_tx {
        let _ = tx.send(());
    }
    // Tiny sleep so the worker definitely observes the start
    // and enters the spin loop before Drop runs.
    std::thread::sleep(Duration::from_millis(50));

    // Drop the ThreadWorker directly — this is the path under
    // test. ThreadWorker::drop must flip stop and join.
    drop(tw);

    // Assertion: by the time drop returns, the worker has
    // observed stop and completed. If drop detached, observed
    // would still be false because the worker would either
    // still be sleeping or already gone without a join.
    assert!(
        observed.load(Ordering::Relaxed),
        "ThreadWorker::drop must join its JoinHandle — observed=false \
         means the drop returned without waiting for the worker, which \
         would mean the worker was detached (Rust's default for \
         JoinHandle::drop) instead of explicitly joined"
    );
}
// -- spawn dispatch tests (Fork / Thread) --

/// Thread mode: the worker runs in-process via std::thread, the
/// JoinHandle returns a real WorkerReport, and worker_pids()
/// reports a non-zero gettid() after start.
#[test]
fn spawn_thread_clone_mode_runs_to_completion() {
    let config = WorkloadConfig {
        num_workers: 2,
        clone_mode: CloneMode::Thread,
        work_type: WorkType::SpinWait,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).expect("Thread mode must spawn");
    h.start();
    std::thread::sleep(std::time::Duration::from_millis(150));
    let pids = h.worker_pids();
    assert_eq!(pids.len(), 2, "worker_pids must reflect both threads");
    for tid in &pids {
        assert!(*tid > 0, "thread tid must be a real gettid() value: {tid}");
    }
    // Sibling threads in the same tgid must report distinct
    // gettid()s — duplicates would mean the publish step is
    // broken or only one thread actually ran.
    assert_ne!(pids[0], pids[1], "sibling thread tids must differ");
    let reports = h.stop_and_collect();
    assert_eq!(
        reports.len(),
        2,
        "thread mode collects one report per worker"
    );
    for r in &reports {
        assert!(r.completed, "thread worker must complete: {:?}", r);
        assert!(
            r.work_units > 0,
            "thread worker must do work: {}",
            r.work_units
        );
    }
}
/// `CloneMode::Thread + WorkType::ForkExit` MUST bail at spawn
/// time. Pin the diagnostic message names both the variant and
/// the structural reason (forked child's `_exit` tears down the
/// whole tgid via `do_exit`).
#[test]
fn spawn_thread_with_forkexit_rejected_at_spawn_time() {
    let config = WorkloadConfig {
        num_workers: 1,
        clone_mode: CloneMode::Thread,
        work_type: WorkType::ForkExit,
        ..Default::default()
    };
    let result = WorkloadHandle::spawn(&config);
    let err = match result {
        Ok(_) => panic!("Thread + ForkExit must bail at spawn"),
        Err(e) => e,
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains("CloneMode::Thread")
            && msg.contains("WorkType::ForkExit")
            && msg.contains("CloneMode::Fork"),
        "diagnostic must name both incompatible variants and the safe \
         alternative: {msg}"
    );
}
/// `CloneMode::Thread + WorkType::CgroupChurn` MUST bail at spawn
/// time. CgroupChurn writes the worker tid to `cgroup.procs`,
/// which the kernel resolves to the whole tgid and migrates every
/// sibling thread to the target cgroup; under Thread mode the
/// "tgid" includes the test harness itself. Pin the diagnostic so
/// a future change to the admission gate cannot silently regress
/// to letting the rejection through. Mirrors
/// `spawn_thread_with_forkexit_rejected_at_spawn_time` — both
/// tests guard CloneMode/WorkType pair rejections at the same
/// admission site.
#[test]
fn spawn_thread_with_cgroupchurn_rejected_at_spawn_time() {
    let config = WorkloadConfig {
        num_workers: 1,
        clone_mode: CloneMode::Thread,
        work_type: WorkType::CgroupChurn {
            groups: 2,
            cycle_ms: 100,
        },
        ..Default::default()
    };
    let result = WorkloadHandle::spawn(&config);
    let err = match result {
        Ok(_) => panic!("Thread + CgroupChurn must bail at spawn"),
        Err(e) => e,
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains("CloneMode::Thread")
            && msg.contains("WorkType::CgroupChurn")
            && msg.contains("CloneMode::Fork"),
        "diagnostic must name both incompatible variants and the safe \
         alternative: {msg}"
    );
}
/// `CloneMode::Fork + WorkType::EpollStorm` MUST bail at spawn
/// time. EpollStorm publishes eventfd / epoll fd numbers through
/// a shared mmap region for siblings to consume, but forked
/// children hold independent fd tables that never contain those
/// post-fork descriptors. Pin the diagnostic. Sibling of the two
/// rejection tests above — kept here so the entire CloneMode /
/// WorkType admission matrix is exercised in one cluster.
#[test]
fn spawn_fork_with_epollstorm_rejected_at_spawn_time() {
    let config = WorkloadConfig {
        num_workers: 2,
        clone_mode: CloneMode::Fork,
        work_type: WorkType::EpollStorm {
            producers: 1,
            consumers: 1,
            events_per_burst: 1,
        },
        ..Default::default()
    };
    let result = WorkloadHandle::spawn(&config);
    let err = match result {
        Ok(_) => panic!("Fork + EpollStorm must bail at spawn"),
        Err(e) => e,
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains("CloneMode::Fork")
            && msg.contains("WorkType::EpollStorm")
            && msg.contains("CloneMode::Thread"),
        "diagnostic must name both incompatible variants and the safe \
         alternative: {msg}"
    );
}
/// Thread-mode worker that panics on first iteration must
/// surface a [`WorkerExitInfo::Panicked`] sentinel with the
/// panic message extracted from the join Err payload. Uses a
/// `WorkType::Custom` closure so the panic path is reproducible
/// without depending on a buggy work-type implementation.
#[test]
fn spawn_thread_panic_yields_panicked_exit_info() {
    // Custom closure that panics immediately. Returns
    // `WorkerReport` to satisfy the signature; the panic fires
    // before `return` is reached.
    fn panic_immediately(_stop: &AtomicBool) -> WorkerReport {
        panic!("test panic from thread worker");
    }
    let config = WorkloadConfig {
        num_workers: 1,
        clone_mode: CloneMode::Thread,
        work_type: WorkType::custom("panic_immediately", panic_immediately),
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).expect("Thread spawn must succeed");
    h.start();
    // Tight: the panic fires synchronously after the start
    // rendezvous; no sleep needed beyond the start handshake.
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 1);
    let r = &reports[0];
    assert!(
        !r.completed,
        "panicked worker must NOT report completed=true"
    );
    match &r.exit_info {
        Some(WorkerExitInfo::Panicked(msg)) => {
            assert!(
                msg.contains("test panic from thread worker"),
                "panic message must round-trip from panic!() to exit_info: {msg}"
            );
        }
        other => panic!("expected Panicked(_) exit_info on thread panic, got {other:?}",),
    }
}
/// Thread-mode `Custom` closure that loops on its `stop` arg
/// MUST terminate via `stop_and_collect` flipping the per-worker
/// flag, AND `stop_and_collect` MUST NOT touch the global
/// [`STOP`] (that signal-flag belongs exclusively to Fork mode;
/// flipping it from Thread mode would inadvertently reach any
/// concurrently-running fork-mode workers and any fork-child of
/// the test harness itself). The test snapshots the global
/// [`STOP`] before/after `stop_and_collect` and asserts no
/// change.
#[test]
fn spawn_thread_custom_stop_does_not_touch_global_stop() {
    // Custom closure that spins on the per-worker stop arg.
    // Returns a non-default WorkerReport with completed=true so
    // the test can pin "the stop loop saw stop=true and exited
    // cleanly" instead of "the worker crashed before reading
    // its arg."
    fn spin_until_stop(stop: &AtomicBool) -> WorkerReport {
        let tid: libc::pid_t = unsafe { libc::syscall(libc::SYS_gettid) as libc::pid_t };
        while !stop_requested(stop) {
            std::thread::sleep(Duration::from_millis(10));
        }
        WorkerReport {
            tid,
            completed: true,
            ..WorkerReport::default()
        }
    }

    // Snapshot the global STOP before spawning. This MUST be
    // false (no concurrent workload running in the test
    // harness) and remain false across the whole call sequence.
    STOP.store(false, Ordering::Relaxed);
    let stop_before = STOP.load(Ordering::Relaxed);
    assert!(
        !stop_before,
        "global STOP must be false before the test runs — \
         a stale true from a prior test would mask the assertion"
    );

    let config = WorkloadConfig {
        num_workers: 1,
        clone_mode: CloneMode::Thread,
        work_type: WorkType::custom("spin_until_stop", spin_until_stop),
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).expect("Thread spawn must succeed");
    h.start();
    // Brief sleep so the worker definitely enters its spin loop
    // before we ask stop_and_collect to flip its flag.
    std::thread::sleep(Duration::from_millis(50));

    let reports = h.stop_and_collect();
    // Worker observed its per-worker stop and returned a clean
    // report — proves the stop signal reached the closure.
    assert_eq!(reports.len(), 1);
    assert!(
        reports[0].completed,
        "Custom thread worker must observe per-worker stop and \
         return completed=true: got {:?}",
        reports[0]
    );

    // Critical assertion: stop_and_collect MUST NOT have flipped
    // the global STOP. Thread-mode stop is per-worker
    // Arc<AtomicBool>; the global STOP is reserved for the
    // SIGUSR1-driven Fork-mode path. Touching it from Thread
    // mode would leak shutdown signals into unrelated workers.
    let stop_after = STOP.load(Ordering::Relaxed);
    assert!(
        !stop_after,
        "global STOP must remain false after Thread-mode \
         stop_and_collect — Thread mode flips per-worker flags \
         only, never the global signal-handler flag"
    );
}
/// Thread-mode workers MUST share the parent's tgid (kernel
/// `getpid()` returns the tgid because `SYS_getpid` is
/// `task_tgid_vnr`) while reporting distinct kernel TIDs from
/// `gettid()`. Pin both halves: every worker's `getpid()` matches
/// the parent's, AND every worker's `gettid()` differs from the
/// parent's. Sibling-distinct gettids are pinned by
/// `spawn_thread_clone_mode_runs_to_completion`; this test pins
/// the parent-vs-worker relationship that flows from
/// `std::thread::spawn` reusing the parent's mm/files/sighand
/// (no new tgid created). A regression to a fork-like dispatch
/// for `CloneMode::Thread` would surface here as worker
/// `getpid() != parent_getpid()`.
#[test]
fn spawn_thread_workers_share_tgid() {
    use std::sync::Mutex;
    // Static collector: each worker pushes its (getpid, gettid)
    // pair before spinning. nextest runs each #[test] in its own
    // process so the static is fresh per-test.
    static WORKER_PIDTIDS: Mutex<Vec<(libc::pid_t, libc::pid_t)>> = Mutex::new(Vec::new());

    fn record_pid_tid_then_spin(stop: &AtomicBool) -> WorkerReport {
        let pid: libc::pid_t = unsafe { libc::getpid() };
        let tid: libc::pid_t = unsafe { libc::syscall(libc::SYS_gettid) as libc::pid_t };
        WORKER_PIDTIDS.lock().unwrap().push((pid, tid));
        while !stop_requested(stop) {
            std::thread::sleep(Duration::from_millis(10));
        }
        WorkerReport {
            tid,
            completed: true,
            ..WorkerReport::default()
        }
    }

    let parent_pid: libc::pid_t = unsafe { libc::getpid() };
    let parent_tid: libc::pid_t = unsafe { libc::syscall(libc::SYS_gettid) as libc::pid_t };

    let config = WorkloadConfig {
        num_workers: 2,
        clone_mode: CloneMode::Thread,
        work_type: WorkType::custom("record_pid_tid_then_spin", record_pid_tid_then_spin),
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).expect("Thread spawn must succeed");
    h.start();
    // Brief sleep so both workers reach the record-and-spin point
    // before stop_and_collect flips their stop flags.
    std::thread::sleep(Duration::from_millis(50));
    let _reports = h.stop_and_collect();

    let captured = WORKER_PIDTIDS.lock().unwrap().clone();
    assert_eq!(
        captured.len(),
        2,
        "both workers must record their (pid, tid) before stop: got {captured:?}"
    );
    for (worker_pid, worker_tid) in &captured {
        assert_eq!(
            *worker_pid, parent_pid,
            "Thread worker getpid()={worker_pid} must match parent \
             getpid()={parent_pid} — std::thread shares the tgid",
        );
        assert_ne!(
            *worker_tid, parent_tid,
            "Thread worker gettid()={worker_tid} must differ from parent \
             gettid()={parent_tid} — each std::thread is a distinct \
             kernel task",
        );
    }
}
/// `CloneMode::Thread + WorkType::NiceSweep` MUST spawn cleanly.
/// NiceSweep cycles `setpriority(PRIO_PROCESS, 0, niceval)` per
/// iteration (see `kernel/sys.c::sys_setpriority` /
/// `set_one_prio`); under Thread mode `0` resolves to the
/// calling task's tid (per-thread credential tweak), not the
/// whole tgid, so it is safe to share with the harness. Pin
/// that the spawn succeeds and the worker produces a
/// non-default report — a regression that bails on Thread +
/// NiceSweep at spawn time, or one that crashes the worker
/// before it returns, would trip this guard.
#[test]
fn spawn_thread_with_nicesweep_succeeds() {
    let config = WorkloadConfig {
        num_workers: 1,
        clone_mode: CloneMode::Thread,
        work_type: WorkType::NiceSweep,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config)
        .expect("Thread + NiceSweep spawn must succeed (no incompatibility)");
    h.start();
    std::thread::sleep(Duration::from_millis(150));
    let reports = h.stop_and_collect();
    assert_eq!(
        reports.len(),
        1,
        "Thread + NiceSweep must collect one report"
    );
    assert!(
        reports[0].completed,
        "Thread + NiceSweep worker must complete cleanly: {:?}",
        reports[0]
    );
}
/// `WorkloadHandle` dropped without `stop_and_collect` MUST
/// drive every Thread worker to completion via Drop's
/// stop-flag-then-join path
/// (`WorkloadHandle::drop`'s `tw.stop.store(true)` →
/// `join_thread_with_timeout`). Pin via a static counter the
/// closures bump just before returning: post-`drop(h)` the
/// counter MUST equal the worker count, proving every worker
/// exited inside the join window — not abandoned, not timed
/// out (5s `THREAD_JOIN_TIMEOUT` would surface as a missing
/// increment).
#[test]
fn spawn_thread_drop_cleanup() {
    use std::sync::atomic::AtomicUsize;
    static EXITED_COUNT: AtomicUsize = AtomicUsize::new(0);

    fn spin_then_record_exit(stop: &AtomicBool) -> WorkerReport {
        let tid: libc::pid_t = unsafe { libc::syscall(libc::SYS_gettid) as libc::pid_t };
        while !stop_requested(stop) {
            std::thread::sleep(Duration::from_millis(5));
        }
        // Bump AFTER the spin loop so the count grows only on
        // a genuine clean exit. SeqCst because the post-Drop
        // load on the parent must observe every increment that
        // happened-before the join — Release/Acquire on the
        // JoinHandle's join already provides the cross-thread
        // edge, but SeqCst keeps the audit trail trivial.
        EXITED_COUNT.fetch_add(1, Ordering::SeqCst);
        WorkerReport {
            tid,
            completed: true,
            ..WorkerReport::default()
        }
    }

    let config = WorkloadConfig {
        num_workers: 2,
        clone_mode: CloneMode::Thread,
        work_type: WorkType::custom("spin_then_record_exit", spin_then_record_exit),
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).expect("Thread spawn must succeed");
    h.start();
    // Brief sleep so workers definitely enter the spin loop
    // before drop flips their stop flags. Without this, drop
    // could race the first stop_requested check and exercise
    // a degenerate "exit before any work" path that doesn't
    // pin the join semantics.
    std::thread::sleep(Duration::from_millis(50));
    // Drop without stop_and_collect — the Drop impl is the
    // sole teardown path under test here.
    drop(h);
    // Drop blocks on join_thread_with_timeout (5s budget); by
    // the time it returns, every joined worker's exit
    // happens-before this load (Release on the JoinHandle's
    // store-pair-with-thread-exit, Acquire on join()).
    let count = EXITED_COUNT.load(Ordering::SeqCst);
    assert_eq!(
        count, 2,
        "both Thread workers must run to completion under \
         WorkloadHandle::drop's join path (got {count}); a count \
         below 2 indicates Drop timed out or abandoned a thread \
         instead of joining it",
    );
}
/// `CloneMode::Thread + WorkType::PipeIo` MUST exchange real
/// bytes through the inter-worker pipe pair. Thread workers
/// share the parent's fd table, so the pipe fds the workers
/// receive are the same kernel-side `pipe2(O_CLOEXEC)`-backed
/// objects the parent allocated. Workers exchange 1-byte
/// messages via `pipe_exchange` (one `write(byte)` then a
/// `poll(POLLIN)` + `read(byte)` round-trip per iteration);
/// each successful round-trip pushes a sample into the
/// reservoir-sampled `resume_latencies_ns` Vec, so a worker
/// reporting an empty `resume_latencies_ns` after the run
/// window means its pipe ops never observed a real wake.
///
/// Asserting `work_units > 0` would NOT prove pipe routing —
/// `pipe_exchange` ignores `libc::write`/`libc::poll` return
/// values, and the surrounding worker loop bumps work_units
/// per iteration regardless of pipe success. A pipe with
/// closed fds returns -1/EBADF and `pipe_exchange` short-
/// circuits via `if ret < 0 { break; }` (skipping the latency
/// push) but the outer iteration counter advances. Hence the
/// invariant the test pins is `resume_latencies_ns.len() > 0`,
/// not `work_units > 0`.
///
/// Pin two stronger checks alongside the latency-sample
/// requirement:
///   - both workers in the (0, 1) pair produce samples (no
///     half-routed pair where only one direction works)
///   - work_units > 0 stays as a smoke check that the worker
///     loop ran at all
#[test]
fn spawn_thread_with_pipe_io() {
    let config = WorkloadConfig {
        num_workers: 2,
        clone_mode: CloneMode::Thread,
        work_type: WorkType::PipeIo { burst_iters: 1024 },
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).expect("Thread + PipeIo spawn must succeed");
    h.start();
    std::thread::sleep(Duration::from_millis(200));
    let reports = h.stop_and_collect();
    assert_eq!(
        reports.len(),
        2,
        "Thread + PipeIo collects one report per worker"
    );
    for r in &reports {
        assert!(
            r.work_units > 0,
            "Thread + PipeIo worker tid={} ran zero iterations: {:?}",
            r.tid,
            r,
        );
        assert!(
            !r.resume_latencies_ns.is_empty(),
            "Thread + PipeIo worker tid={} captured zero wake-latency \
             samples — its pipe ops never observed a partner write, \
             which under shared-fd-table semantics means the pipe fds \
             were closed before the worker reached pipe_exchange. \
             work_units={} (bumped regardless of pipe success). Full \
             report: {:?}",
            r.tid,
            r.work_units,
            r,
        );
    }
}
/// `WakeChain { wake: WakeMechanism::Pipe }` bootstrap-once
/// invariant under [`CloneMode::Thread`]. The shared fd table
/// makes the bug behavior identical to Fork mode (the same
/// repeat-bootstrap regression queues bytes on the same
/// pipe), but the spawn path differs: thread workers route
/// through `spawn_thread_worker` rather than `fork`, and the
/// pipe-fd ownership transfer goes through
/// `WorkloadHandle::chain_pipes` rather than the post-fork
/// close. This test pins the throughput contract under the
/// Thread spawn path so a regression that breaks Thread-mode
/// pipe-fd lifetime (e.g. closes fds before workers reach
/// the chain handoff) trips the bootstrap-once invariant
/// here too.
///
/// Identical thresholds to the Fork-mode test
/// (`wake_chain_pipe_bootstrap_once_invariant`): depth=4,
/// work_per_hop=50ms, 1s window, total ≤ 40. Throughput is
/// wall-clock-bounded by `work_per_hop`, not clone-mode-
/// bounded — both Fork and Thread workers spend ~50ms in
/// the CPU burst per stage handoff, so the per-stage rate
/// ceiling and the buggy 4× upper expectation match
/// exactly.
#[test]
fn wake_chain_pipe_thread_mode_bootstrap_throughput() {
    const DEPTH: usize = 4;
    const WORK_PER_HOP_MS: u64 = 50;
    const TEST_WINDOW_MS: u64 = 1000;
    const TOTAL_ITER_THRESHOLD: u64 = 40;

    if require_isolated_cpus(DEPTH, "wake_chain_pipe_thread_mode_bootstrap_throughput") {
        return;
    }

    let config = WorkloadConfig {
        num_workers: DEPTH,
        clone_mode: CloneMode::Thread,
        work_type: WorkType::WakeChain {
            depth: DEPTH,
            wake: WakeMechanism::Pipe,
            work_per_hop: Duration::from_millis(WORK_PER_HOP_MS),
        },
        ..Default::default()
    };
    let mut h =
        WorkloadHandle::spawn(&config).expect("Thread + WakeChain wake=Pipe spawn must succeed");
    h.start();
    std::thread::sleep(Duration::from_millis(TEST_WINDOW_MS));
    let reports = h.stop_and_collect();
    assert_eq!(
        reports.len(),
        DEPTH,
        "Thread + WakeChain wake=Pipe collects one report per worker"
    );
    let total_iters: u64 = reports.iter().map(|r| r.iterations).sum();
    assert!(
        total_iters <= TOTAL_ITER_THRESHOLD,
        "Thread + WakeChain wake=Pipe total iterations across {DEPTH} \
         stages exceeded {TOTAL_ITER_THRESHOLD} over {TEST_WINDOW_MS}ms \
         with work_per_hop={WORK_PER_HOP_MS}ms (got {total_iters}). \
         Throughput is wall-clock-bounded; the bootstrap-once invariant \
         holds identically under Thread mode. Expected correct total \
         ~{}; expected buggy total ~{}. Per-worker reports: {:?}",
        TEST_WINDOW_MS / WORK_PER_HOP_MS,
        (TEST_WINDOW_MS / WORK_PER_HOP_MS) * (DEPTH as u64),
        reports,
    );
    assert!(
        total_iters >= 4,
        "Thread + WakeChain wake=Pipe made fewer than one ring \
         round-trip over {TEST_WINDOW_MS}ms (got {total_iters}, \
         expected ≥ 4) — the bootstrap byte never completed a full \
         lap. Under Thread mode this typically means the pipe fds \
         were closed before the workers reached the chain handoff \
         site (a regression in `WorkloadHandle::chain_pipes` \
         ownership transfer). Per-worker reports: {:?}",
        reports,
    );
}
/// `CloneMode::Thread + WorkType::WakeChain { wake: Pipe }`
/// MUST run the chain pipes to completion. After the pipe-fd
/// ownership fix (chain_pipes now transferred to
/// [`WorkloadHandle`] and closed only at handle Drop), a
/// Thread-mode WakeChain wake=Pipe workload must observe each
/// stage's predecessor write — verified via
/// `resume_latencies_ns` non-empty for at least one stage in
/// the chain (the head stage publishes its first wake on the
/// bootstrap path; subsequent stages collect samples on the
/// post-bootstrap rounds).
///
/// Before the fix, this configuration was rejected at spawn
/// time with the diagnostic "WakeChain wake=Pipe is not
/// supported under CloneMode::Thread"; the rejection arm has
/// been deleted, so spawn now succeeds and the workers must
/// route real bytes through the chain pipes.
#[test]
fn spawn_thread_with_wake_chain_pipe() {
    let config = WorkloadConfig {
        num_workers: 2,
        clone_mode: CloneMode::Thread,
        work_type: WorkType::WakeChain {
            depth: 2,
            wake: WakeMechanism::Pipe,
            work_per_hop: Duration::from_micros(100),
        },
        ..Default::default()
    };
    let mut h =
        WorkloadHandle::spawn(&config).expect("Thread + WakeChain wake=Pipe spawn must succeed");
    h.start();
    std::thread::sleep(Duration::from_millis(200));
    let reports = h.stop_and_collect();
    assert_eq!(
        reports.len(),
        2,
        "Thread + WakeChain wake=Pipe collects one report per worker"
    );
    let total_samples: usize = reports.iter().map(|r| r.resume_latencies_ns.len()).sum();
    assert!(
        total_samples > 0,
        "Thread + WakeChain wake=Pipe captured zero wake-latency \
         samples across {} workers — the chain pipes never routed a \
         stage handoff, which under shared-fd-table semantics means \
         the pipe fds were closed before the workers reached the \
         chain handoff site. Per-worker reports: {:?}",
        reports.len(),
        reports,
    );
    for r in &reports {
        assert!(
            r.work_units > 0,
            "Thread + WakeChain wake=Pipe worker tid={} ran zero \
             iterations: {:?}",
            r.tid,
            r,
        );
    }
}
/// `CloneMode::Thread + WorkType::FutexPingPong` MUST run to
/// completion. FutexPingPong allocates a per-pair shared
/// `u32` futex word and exchanges `FUTEX_WAKE` / `FUTEX_WAIT`
/// across the pair — under Thread mode every worker shares
/// the harness's address space, so the existing per-pair
/// futex plumbing must still pair (0,1) correctly. Both
/// workers must produce work_units > 0; a regression that
/// binds the futex word to a fork-only allocation site would
/// surface as one or both workers reporting zero work.
#[test]
fn spawn_thread_with_futex_ping_pong() {
    let config = WorkloadConfig {
        num_workers: 2,
        clone_mode: CloneMode::Thread,
        work_type: WorkType::FutexPingPong { spin_iters: 1024 },
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).expect("Thread + FutexPingPong spawn must succeed");
    h.start();
    std::thread::sleep(Duration::from_millis(200));
    let reports = h.stop_and_collect();
    assert_eq!(
        reports.len(),
        2,
        "Thread + FutexPingPong collects one report per worker",
    );
    for r in &reports {
        assert!(
            r.work_units > 0,
            "Thread + FutexPingPong worker tid={} did no work: {:?}",
            r.tid,
            r,
        );
    }
}
/// `WorkloadHandle::set_affinity` MUST succeed for a Thread
/// worker once the worker has published its `gettid()` — the
/// `Acquire` load on `tw.tid` returns a non-zero kernel task
/// id, and `sched_setaffinity(tid, ...)` accepts the per-task
/// pid_t. The publish happens on the worker thread's first
/// instructions (see `spawn_thread_worker`'s `tid_thread.store`
/// before the start rendezvous); calling `start()` plus a
/// brief sleep guarantees the publish is observable, matching
/// the doc's "call start() first" guidance. Pinning the
/// Ok-on-CPU-0 path here guards the post-start affinity
/// surface against a regression that re-introduces the
/// pre-publish bail (`tid == 0`) for live threads.
#[test]
fn spawn_thread_set_affinity_works_post_start() {
    let config = WorkloadConfig {
        num_workers: 1,
        clone_mode: CloneMode::Thread,
        work_type: WorkType::SpinWait,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).expect("Thread spawn must succeed");
    h.start();
    // Give the worker a moment to publish its tid past the
    // Release store. Without this the Acquire load races the
    // store and could observe the AtomicI32's initial 0 — the
    // bail branch we explicitly do not want to test here.
    std::thread::sleep(Duration::from_millis(50));
    let cpus: BTreeSet<usize> = [0].into_iter().collect();
    let result = h.set_affinity(0, &cpus);
    assert!(
        result.is_ok(),
        "set_affinity(0, {{0}}) on a started Thread worker must succeed; \
         got {:?}",
        result.err(),
    );
    let _reports = h.stop_and_collect();
}
// -- Thread-mode dispatch coverage expansion --
//
// These tests pin Thread-mode worker contracts the initial
// dispatch tests didn't cover: thread/tgid identity, bounded
// stop latency, multi-worker panic isolation, drop cleanup,
// affinity, and paired-WorkType compatibility.

/// All Thread-mode workers share the same tgid (kernel
/// "process") because they live inside the test harness's own
/// process. Distinct gettid()s but a single getpid() — pinning
/// this proves the Thread variant really creates std::thread
/// kernel tasks, not hidden subprocess-style isolation. The
/// tgid invariant is what makes the cgroup.procs hazard at
/// `worker_pids` real.
#[test]
fn thread_workers_share_tgid_with_harness() {
    let config = WorkloadConfig {
        num_workers: 3,
        clone_mode: CloneMode::Thread,
        work_type: WorkType::SpinWait,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).expect("Thread spawn must succeed");
    h.start();
    std::thread::sleep(Duration::from_millis(100));
    let pids = h.worker_pids();
    assert_eq!(pids.len(), 3);
    let harness_pid = unsafe { libc::getpid() };
    for &tid in &pids {
        let status = std::fs::read_to_string(format!("/proc/{tid}/status"))
            .expect("must read /proc/<tid>/status for thread worker");
        let tgid_line = status
            .lines()
            .find(|l| l.starts_with("Tgid:"))
            .expect("status must include Tgid line");
        let tgid: i32 = tgid_line
            .trim_start_matches("Tgid:")
            .trim()
            .parse()
            .expect("Tgid must be a parseable integer");
        assert_eq!(
            tgid, harness_pid,
            "Thread worker tid={tid} must share tgid with test harness pid={harness_pid}; \
             found Tgid={tgid}. Thread workers run inside the harness process — a \
             distinct tgid would mean the dispatch silently forked instead."
        );
    }
    let _ = h.stop_and_collect();
}
/// Thread-mode `stop_and_collect` must return inside a bounded
/// deadline once the per-worker stop flag is flipped. Pin a 5s
/// upper bound: workers that don't poll their stop flag would
/// hang the harness, and this test would fail at the deadline.
#[test]
fn thread_stop_and_collect_returns_within_bounded_deadline() {
    fn spin_until_stop(stop: &AtomicBool) -> WorkerReport {
        let tid: libc::pid_t = unsafe { libc::syscall(libc::SYS_gettid) as libc::pid_t };
        while !stop_requested(stop) {
            std::thread::sleep(Duration::from_millis(10));
        }
        WorkerReport {
            tid,
            completed: true,
            ..WorkerReport::default()
        }
    }
    let config = WorkloadConfig {
        num_workers: 4,
        clone_mode: CloneMode::Thread,
        work_type: WorkType::custom("spin_until_stop", spin_until_stop),
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).expect("Thread spawn must succeed");
    h.start();
    std::thread::sleep(Duration::from_millis(50));
    let started = std::time::Instant::now();
    let reports = h.stop_and_collect();
    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_secs(5),
        "stop_and_collect must return inside 5s for cooperating workers; took {elapsed:?}"
    );
    assert_eq!(reports.len(), 4);
    for r in &reports {
        assert!(
            r.completed,
            "every worker must observe stop and return: {r:?}"
        );
    }
}
