//! Spawn-pipeline tests — spawn_guard group.

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

/// EMFILE on the inter-worker pipe loop: with num_workers=4 and
/// PipeIo (which needs 2 pipe pairs = 4 pipe() calls = 8 fds),
/// cap RLIMIT_NOFILE at baseline+5 so the first pair allocates
/// cleanly (ab+ba = 4 fds) and the second pair's first `pipe(ab)`
/// call fails with EMFILE (needs 2 fds, only 1 slot remains).
/// At bail time `guard.pipe_pairs` holds the first pair;
/// SpawnGuard::Drop must close all 4 fds so the child's fd
/// count returns to baseline.
///
/// Assumes a dense fd table (no gaps below the current baseline).
/// If the child inherits a sparse table (e.g. a coordinator that
/// closed fd 2 but left fd 3 open), RLIMIT_NOFILE gating yields
/// different triggering semantics and the test may report 10
/// (failure did not trigger) instead of 0. Also assumes
/// `RUST_BACKTRACE` is unset — when set, a panic inside the body
/// triggers backtrace capture which itself opens fds, shifting
/// the effective baseline mid-run.
#[test]
fn spawn_guard_cleans_up_on_interworker_pipe_emfile() {
    let code = run_in_forked_child(|| {
        let baseline = count_open_fds();
        // Capture the inherited RLIMIT_NOFILE so the post-bail
        // restore uses a value the kernel will accept. The
        // lowering path below touches only `rlim_cur` and leaves
        // `rlim_max` at the original value, so an unprivileged
        // process can still raise `rlim_cur` back up after the
        // bail (without CAP_SYS_RESOURCE, which would be needed
        // to raise a previously-lowered `rlim_max`).
        let mut original_rlimit = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut original_rlimit) } != 0 {
            return 13;
        }
        // RLIMIT_NOFILE is a hard limit on the highest fd
        // number + 1, not a headroom value — we need to pass a
        // value slightly above baseline so the first pipe pair
        // succeeds but the second pair's first `pipe(ab)` does
        // not. baseline + 5 permits 5 new fds: 4 for the first
        // pipe pair (ab+ba) and 1 leftover. The second pair's
        // `pipe(ab)` needs 2 fds against that 1 slot and fails
        // with EMFILE.
        let target_cur = (baseline + 5) as u64;
        let lowered = libc::rlimit {
            rlim_cur: target_cur,
            rlim_max: original_rlimit.rlim_max,
        };
        if unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &lowered) } != 0 {
            return 13;
        }
        let config = WorkloadConfig {
            num_workers: 4,
            affinity: AffinityIntent::Inherit,
            work_type: WorkType::PipeIo { burst_iters: 1 },
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let result = WorkloadHandle::spawn(&config);
        if result.is_ok() {
            return 10; // Failure did not trigger.
        }
        // SpawnGuard::Drop has already run on the `?`/`bail!`
        // exit. Raise rlim_cur back to its original value so
        // reading /proc/self/fd for the post-check does not
        // itself fail with EMFILE. Silent ignore here would mask
        // an EMFILE in `count_open_fds` below as a fd leak;
        // return code 15 distinguishes the harness issue from a
        // guard defect.
        let err_msg = format!("{:#}", result.as_ref().err().unwrap());
        if unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &original_rlimit) } != 0 {
            return 15;
        }
        // Prove the bail arrived via the pipe branch, not a
        // later mmap or fork. Both inter-worker pipe-failure
        // paths bail with "pipe2 failed: ..." (the
        // `libc::pipe2` call site at the top of `spawn_group`).
        if !err_msg.contains("pipe2 failed") {
            return 14;
        }
        let after = count_open_fds();
        if after > baseline {
            return 11; // Fd leak.
        }
        if any_zombie_child() {
            return 12;
        }
        0
    });
    assert_eq!(
        code, 0,
        "child reported cleanup defect (code {code}): see exit-code table above \
         spawn_guard_cleans_up_on_interworker_pipe_emfile"
    );
}
/// EMFILE during a WakeChain `wake = WakeMechanism::Pipe` spawn:
/// with num_workers=4 and depth=4, the spawn path needs 8 fds
/// for the chain-pipe ring (1 chain × 4 pipes × 2 fds), plus 4
/// fds per worker for the report+start pipe pair (16 more for
/// 4 workers), all allocated inside [`WorkloadHandle::spawn`]'s
/// spawn_group routine via `libc::pipe2`. Cap RLIMIT_NOFILE at
/// baseline+5 so the kernel rejects one of those `pipe2` calls
/// with EMFILE before every worker has been forked. The exact
/// allocation that hits the limit depends on transient fd state
/// (CI runner, coverage instrumentation, prior tests in the
/// same nextest worker) because a sparse fd table — fds
/// inherited above the new `rlim_cur` — leaves more low-fd
/// slots free for new allocations than a dense table does. The
/// test therefore accepts any pipe2-related bail; the
/// SpawnGuard cleanup contract (no fd leak, no zombie children
/// after Drop) is independent of which pipe2 site fired first.
///
/// SpawnGuard::Drop must close everything that was successfully
/// allocated by the time the bail fires: any chain pipes
/// pushed into `guard.chain_pipes`, the iter_counters mmap, the
/// futex region mmap, and any per-worker pipes the local
/// fork-loop cleanup didn't already release. The fd count must
/// return to baseline.
///
/// This test mirrors `spawn_guard_cleans_up_on_interworker_pipe_emfile`
/// for the chain-pipe path. WakeChain wake=Pipe is the only
/// other allocation site in WorkloadHandle::spawn that calls
/// `libc::pipe2` per-stage at allocation time (PipeIo /
/// CachePipe go through the inter-worker pipe-pair loop tested
/// separately above).
///
/// Inherits the same harness assumptions:
/// - `RUST_BACKTRACE` unset (panic-time fd churn would shift
///   the effective baseline mid-run)
#[test]
fn spawn_guard_cleans_up_on_wake_chain_pipe_emfile() {
    let code = run_in_forked_child(|| {
        let baseline = count_open_fds();
        // Capture the inherited RLIMIT_NOFILE so the post-bail
        // restore uses a value the kernel will accept. Same
        // pattern as spawn_guard_cleans_up_on_interworker_pipe_emfile.
        let mut original_rlimit = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut original_rlimit) } != 0 {
            return 13;
        }
        // baseline + 5 caps the new-fd budget tight enough that
        // at least one `pipe2` call inside spawn_group (chain
        // ring, per-worker report, or per-worker start) hits
        // EMFILE before every worker has been forked. Which
        // specific call fails depends on the inherited fd
        // density (see the per-test doc comment).
        let target_cur = (baseline + 5) as u64;
        let lowered = libc::rlimit {
            rlim_cur: target_cur,
            rlim_max: original_rlimit.rlim_max,
        };
        if unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &lowered) } != 0 {
            return 13;
        }
        let config = WorkloadConfig {
            num_workers: 4,
            affinity: AffinityIntent::Inherit,
            work_type: WorkType::WakeChain {
                depth: 4,
                wake: WakeMechanism::Pipe,
                work_per_hop: Duration::from_micros(100),
            },
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let result = WorkloadHandle::spawn(&config);
        if result.is_ok() {
            return 10; // Failure did not trigger.
        }
        let err_msg = format!("{:#}", result.as_ref().err().unwrap());
        if unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &original_rlimit) } != 0 {
            return 15;
        }
        // Prove the bail came from a pipe2 EMFILE on the spawn
        // path. The exact allocation that hits the limit depends
        // on transient fd state (coverage instrumentation, CI
        // environment), so accept any pipe2-related bail.
        if !err_msg.contains("pipe2 ") {
            eprintln!("unexpected spawn error (exit 14): {err_msg}");
            return 14;
        }
        let after = count_open_fds();
        if after > baseline {
            return 11; // Fd leak.
        }
        if any_zombie_child() {
            return 12;
        }
        0
    });
    assert_eq!(
        code, 0,
        "child reported cleanup defect (code {code}): see exit-code table above \
         spawn_guard_cleans_up_on_wake_chain_pipe_emfile"
    );
}
/// EAGAIN on `fork`: with num_workers=1 and SpinWait (no pipe
/// pairs, no futex), cap RLIMIT_NPROC to 0 so the very first
/// `libc::fork` inside the per-worker loop returns -1. At bail
/// time the local cleanup (in the per-worker fork dispatch in
/// `WorkloadHandle::spawn`) has closed the report+start pipes, so
/// the guard carries only its empty `pipe_pairs`, zero children,
/// and the iter_counters mmap. The Drop munmaps the iter_counters
/// region (no-op for the fd count but proves the guard path
/// fires) and returns cleanly. No zombies, no fd leak.
#[test]
fn spawn_guard_cleans_up_on_fork_eagain() {
    let code = run_in_forked_child(|| {
        let baseline = count_open_fds();
        if !set_rlimit_nproc_zero_headroom() {
            return 13;
        }
        let config = WorkloadConfig {
            num_workers: 1,
            affinity: AffinityIntent::Inherit,
            work_type: WorkType::SpinWait,
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let result = WorkloadHandle::spawn(&config);
        if result.is_ok() {
            // CAP_SYS_RESOURCE bypasses RLIMIT_NPROC — skip.
            return 0;
        }
        let msg = format!("{:#}", result.err().unwrap());
        // RLIMIT_NPROC denies fork with EAGAIN; prove the bail
        // arrived via the fork branch, not an earlier pipe
        // allocation.
        if !msg.contains("fork failed") {
            return 14;
        }
        let after = count_open_fds();
        if after > baseline {
            return 11;
        }
        if any_zombie_child() {
            return 12;
        }
        0
    });
    assert_eq!(
        code, 0,
        "child reported cleanup defect (code {code}): see exit-code table above \
         spawn_guard_cleans_up_on_fork_eagain"
    );
}
/// IoSyncWrite uses /dev/vda when available (block device, no
/// path to clean up) and falls back to a per-worker tempfile
/// `ktstr_iodev_{tid}` on host machines where /dev/vda is
/// absent. The cleanup contract: when the fallback was used,
/// the tempfile must be unlinked when the worker exits.
/// Skipped when running inside a VM where /dev/vda exists —
/// no fallback path to assert on.
#[test]
fn io_sync_write_cleans_up_tempfile_fallback() {
    if std::path::Path::new("/dev/vda").exists() {
        // Running inside a VM with a real virtio-blk: the
        // workload uses /dev/vda directly, no host-side
        // tempfile to clean up.
        return;
    }
    let config = WorkloadConfig {
        num_workers: 1,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::IoSyncWrite,
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    h.start();
    std::thread::sleep(std::time::Duration::from_millis(200));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 1);
    let tid = reports[0].tid;
    let path = std::env::temp_dir()
        .join(format!("ktstr_iodev_{tid}"))
        .to_string_lossy()
        .to_string();
    assert!(
        !std::path::Path::new(&path).exists(),
        "tempfile fallback {path} should be cleaned up"
    );
}
