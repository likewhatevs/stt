//! Synthetic-tree capture_with / capture_pid coverage. Calls stage_synthetic_proc from `super::tests_helpers`.
//!
//! Co-located with `super::mod.rs`; one of the topic-grouped
//! split files that replace the monolithic `tests.rs`.

#![cfg(test)]

use super::*;
use crate::metric_types::Bytes;
use super::tests_helpers::stage_synthetic_proc;


/// Ghost-thread filter: a tid whose directory exists but
/// carries ZERO readable procfs files (classic mid-capture
/// exit — readdir races the reap) assembles an all-Default
/// `ThreadState` and must NOT land in the snapshot. Stages
/// one live thread with real content and one empty-directory
/// ghost tid under the same tgid, calls `capture_with`, and
/// asserts the output contains only the live thread.
///
/// Without the filter, the ghost would land as `{ tid: 202,
/// comm: "", cgroup: "", start_time_clock_ticks: 0, ...all
/// counters zero }` and pollute downstream comparisons — a
/// baseline run captures some number of ghosts, the candidate
/// captures a different number, and the diff surfaces spurious
/// "thread vanished" signal on every report.
#[test]
fn capture_with_filters_ghost_threads_with_empty_comm_and_zero_start() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 42;
    let live_tid: i32 = 101;
    let ghost_tid: i32 = 202;

    // Stage the live thread in full.
    stage_synthetic_proc(proc_tmp.path(), tgid, live_tid, "pcomm-proc", "live-thread");

    // Stage a ghost tid directory with NO inner files —
    // simulates the "readdir saw it, per-file reads all
    // ENOENT'd" race window. `iter_task_ids_at` enumerates
    // it (the numeric dir name parses), every capture read
    // returns the default, and the filter rejects the
    // resulting all-zero entry.
    let ghost_dir = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(ghost_tid.to_string());
    std::fs::create_dir_all(&ghost_dir).unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);

    // Exactly one thread — the live one. The ghost is gone.
    assert_eq!(
        snap.threads.len(),
        1,
        "ghost tid with empty comm + zero start must be filtered; \
         got threads: {:?}",
        snap.threads
            .iter()
            .map(|t| (t.tid, &t.comm))
            .collect::<Vec<_>>(),
    );
    assert_eq!(snap.threads[0].tid, live_tid as u32);
    assert_eq!(snap.threads[0].comm, "live-thread");
}

/// H1 + H2 — `capture_with` against a synthetic procfs:
/// staging every file the capture walks and asserting the
/// assembled `ThreadState` carries the planted values.
#[test]
fn capture_with_synthetic_tree_assembles_thread_state() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 42;
    let tid: i32 = 101;

    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "pcomm-proc", "worker-thread");

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);

    // Exactly one thread — the one we planted.
    assert_eq!(snap.threads.len(), 1, "synthetic proc has one tid");
    let t = &snap.threads[0];

    // Identity fields (round-trip from /proc/<tgid>/comm +
    // /proc/<tgid>/task/<tid>/comm).
    assert_eq!(t.tid, tid as u32);
    assert_eq!(t.tgid, tgid as u32);
    assert_eq!(t.pcomm, "pcomm-proc");
    assert_eq!(t.comm, "worker-thread");
    assert_eq!(t.cgroup, "/ktstr.slice/worker0");

    use crate::metric_types::{
        Bytes, CategoricalString, ClockTicks, CpuSet, MonotonicCount, MonotonicNs, OrdinalI32,
        PeakNs,
    };

    // /proc/<tid>/stat fields parsed out of the paren-comm
    // tail: nice, utime, stime, starttime, processor, policy,
    // minflt, majflt.
    assert_eq!(t.nice, OrdinalI32(-10));
    assert_eq!(t.start_time_clock_ticks, 555_555);
    assert_eq!(t.policy, CategoricalString::from("SCHED_OTHER"));
    assert_eq!(t.minflt, MonotonicCount(7777));
    assert_eq!(t.majflt, MonotonicCount(8888));
    assert_eq!(
        t.utime_clock_ticks,
        ClockTicks(10),
        "tail[11] of stat fixture lands at utime_clock_ticks",
    );
    assert_eq!(
        t.stime_clock_ticks,
        ClockTicks(11),
        "tail[12] of stat fixture lands at stime_clock_ticks",
    );
    assert_eq!(
        t.processor,
        OrdinalI32(1700),
        "tail[36] of stat fixture (the 17th post-starttime \
         token, value 100*17=1700) lands at processor. 1700 is \
         a synthetic test value; on real hosts `processor` is \
         bounded by the online CPU count and would not exceed \
         `nproc - 1`. The synthetic fixture exercises the wire \
         format without conforming to that bound, since no real \
         /proc is involved.",
    );

    // schedstat — three-tuple of run/wait/slices.
    assert_eq!(t.run_time_ns, MonotonicNs(1_000_000));
    assert_eq!(t.wait_time_ns, MonotonicNs(200_000));
    assert_eq!(t.timeslices, MonotonicCount(50));

    // status — state + csw + Cpus_allowed_list. With
    // `use_syscall_affinity=false`, the capture path reads
    // cpu_affinity from status only.
    assert_eq!(
        t.state, 'R',
        "first non-whitespace char of `State:\tR (running)` is \
         the single-letter code R",
    );
    assert_eq!(t.voluntary_csw, MonotonicCount(42));
    assert_eq!(t.nonvoluntary_csw, MonotonicCount(7));
    assert_eq!(t.cpu_affinity, CpuSet(vec![0, 1, 2, 3]));

    // io — seven cumulative counters.
    assert_eq!(t.rchar, Bytes(100));
    assert_eq!(t.wchar, Bytes(200));
    assert_eq!(t.syscr, MonotonicCount(10));
    assert_eq!(t.syscw, MonotonicCount(20));
    assert_eq!(t.read_bytes, Bytes(4096));
    assert_eq!(t.write_bytes, Bytes(8192));
    assert_eq!(
        t.cancelled_write_bytes,
        Bytes(512),
        "cancelled_write_bytes round-trips from the 7th line of \
         /proc/<tid>/io",
    );

    // sched — every wakeup field, migrations (live counters
    // only; the dead-counter fields nr_wakeups_idle /
    // nr_migrations_cold / nr_wakeups_passive are no longer
    // surfaced on ThreadState — the kernel never increments
    // them so the registry was the wrong place for them; the
    // synthetic fixture still emits the lines to exercise the
    // parser's silent-drop on unknown keys), the four *_sum
    // fractional-parse fields, the five *_max fractional-parse
    // fields, and the ext.enabled bool.
    assert_eq!(t.nr_wakeups, MonotonicCount(11));
    assert_eq!(t.nr_wakeups_local, MonotonicCount(8));
    assert_eq!(t.nr_wakeups_remote, MonotonicCount(3));
    assert_eq!(t.nr_wakeups_sync, MonotonicCount(2));
    assert_eq!(t.nr_wakeups_migrate, MonotonicCount(1));
    assert_eq!(t.nr_wakeups_affine, MonotonicCount(12));
    assert_eq!(
        t.nr_wakeups_affine_attempts,
        MonotonicCount(20),
        "denominator for the affine-wake success ratio \
         (nr_wakeups_affine / nr_wakeups_affine_attempts = 12/20)",
    );
    assert_eq!(t.nr_migrations, MonotonicCount(9));
    assert_eq!(t.nr_forced_migrations, MonotonicCount(7));
    assert_eq!(t.nr_failed_migrations_affine, MonotonicCount(1));
    assert_eq!(t.nr_failed_migrations_running, MonotonicCount(2));
    assert_eq!(t.nr_failed_migrations_hot, MonotonicCount(3));
    // PN_SCHEDSTAT format is ms.ns_remainder. Reconstructed
    // ns = ms_part * 1_000_000 + zero-right-padded ns_part.
    // `5000.25` → `.25` pads to `.250000` (=250_000 ns) +
    // 5000ms × 1_000_000 = 5_000_250_000 ns total.
    assert_eq!(
        t.wait_sum,
        MonotonicNs(5_000_250_000),
        "PN_SCHEDSTAT 5000.25 reconstructs to 5_000_250_000 ns \
         (5000ms + 250_000ns)",
    );
    assert_eq!(t.wait_count, MonotonicCount(15));
    assert_eq!(
        t.wait_max,
        PeakNs(250_500_000),
        "PN_SCHEDSTAT 250.5 reconstructs to 250_500_000 ns",
    );
    // voluntary_sleep_ns = sum_sleep_runtime - sum_block_runtime,
    // computed at capture: 3_200_500_000 - 1_100_750_000 =
    // 2_099_750_000 ns. The kernel's sum_sleep_runtime
    // double-counts block under sleep, so the normalized
    // voluntary-only residual is what surfaces on ThreadState.
    assert_eq!(
        t.voluntary_sleep_ns,
        MonotonicNs(2_099_750_000),
        "voluntary_sleep_ns = sum_sleep_runtime (3_200_500_000) \
         minus sum_block_runtime (1_100_750_000) = \
         2_099_750_000 ns; capture-side normalization strips \
         the kernel's sleep/block double-count",
    );
    assert_eq!(
        t.sleep_max,
        PeakNs(180_250_000),
        "PN_SCHEDSTAT 180.25 reconstructs to 180_250_000 ns",
    );
    assert_eq!(
        t.block_sum,
        MonotonicNs(1_100_750_000),
        "PN_SCHEDSTAT 1100.75 reconstructs to 1_100_750_000 ns; \
         block_sum is populated from the kernel's `sum_block_runtime` key",
    );
    assert_eq!(
        t.block_max,
        PeakNs(60_750_000),
        "PN_SCHEDSTAT 60.75 reconstructs to 60_750_000 ns",
    );
    assert_eq!(
        t.iowait_sum,
        MonotonicNs(77_000_000),
        "PN_SCHEDSTAT 77.0 reconstructs to 77_000_000 ns",
    );
    assert_eq!(t.iowait_count, MonotonicCount(18));
    assert_eq!(
        t.exec_max,
        PeakNs(90_000_000),
        "PN_SCHEDSTAT 90.0 reconstructs to 90_000_000 ns",
    );
    assert_eq!(
        t.slice_max,
        PeakNs(400_500_000),
        "PN_SCHEDSTAT 400.5 reconstructs to 400_500_000 ns",
    );
    assert!(
        t.ext_enabled,
        "ext.enabled = 1 round-trips through the full-key gate \
         to ThreadState::ext_enabled true",
    );

    // jemalloc TSD counters: synthetic procfs has no real ELF
    // behind /proc/<tgid>/exe, so the probe attach is gated off
    // (use_syscall_affinity=false). Both fields land at the
    // absent-counter default of 0. Pins this so a future
    // regression that always-probes (ignoring use_syscall_affinity)
    // would either crash on the synthetic /proc or surface garbage
    // here.
    assert_eq!(
        t.allocated_bytes,
        Bytes(0),
        "synthetic-tree capture must not probe — allocated_bytes \
         collapses to absent-counter zero",
    );
    assert_eq!(
        t.deallocated_bytes,
        Bytes(0),
        "synthetic-tree capture must not probe — deallocated_bytes \
         collapses to absent-counter zero",
    );
}

/// Capture against an empty `proc_root` (no tgid subdirs at
/// all) must complete without panic and produce an empty
/// snapshot. Pins the rayon parallel-probe phase's empty-input
/// handling: `iter_tgids_at` returns an empty Vec, `par_iter`
/// over zero elements collects to an empty HashMap, and the
/// sequential phase 2 loop runs zero iterations. `use_syscall_affinity=true`
/// is required to enter the rayon block at all (the `false`
/// branch skips probe-attach entirely and assigns an empty
/// HashMap directly). Without this gate test, the rayon
/// par_iter over empty input has zero coverage.
#[test]
fn capture_with_empty_proc_root_produces_empty_snapshot() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();

    // Stage `/proc/loadavg` so the parallelism-clamp read at
    // <proc_root>/loadavg succeeds rather than falling back to
    // the 0.0 default. Empty `proc_root` otherwise — no tgid
    // subdirs, so `iter_tgids_at` returns Vec::new().
    std::fs::write(proc_tmp.path().join("loadavg"), "0.0 0.0 0.0 1/1 1\n").unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), true);
    assert!(
        snap.threads.is_empty(),
        "empty proc_root must produce empty snapshot; got {} threads",
        snap.threads.len(),
    );
}

/// Exercises the cache-lookup and insert code path in the
/// rayon probe loop. Two tgids whose `/proc/<tgid>/exe`
/// symlinks resolve to the same underlying inode trigger
/// cache interaction: both attach calls fail with
/// AttachError::MapsReadFailure (the synthetic tree has no
/// `/proc/<tgid>/maps`), and the absent-counter contract
/// holds — both threads land in the snapshot with
/// allocated_bytes==0.
#[test]
fn capture_with_inode_cache_collapses_duplicate_binaries() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();

    // Required by the parallelism-clamp logic in capture_with.
    std::fs::write(proc_tmp.path().join("loadavg"), "0.0 0.0 0.0 1/1 1\n").unwrap();

    // One real file, two symlinks pointing at it. Both tgids'
    // exe metadata calls return the same (dev, ino) tuple, so
    // the cache_key matches across them.
    let shared_exe = proc_tmp.path().join("shared-exe");
    std::fs::write(&shared_exe, b"\x7fELFsynthetic\n").unwrap();

    for tgid in [4242, 4243] {
        stage_synthetic_proc(
            proc_tmp.path(),
            tgid,
            tgid + 1,
            "shared-pcomm",
            "shared-comm",
        );
        // `/proc/<tgid>/exe` symlink points at the shared file.
        // `attach_jemalloc_at` will read_link this successfully
        // and then fail on the absent `/proc/<tgid>/maps` →
        // AttachError::MapsReadFailure. The cache stores None
        // keyed by (dev, ino) of the shared file.
        let exe_link = proc_tmp.path().join(tgid.to_string()).join("exe");
        std::os::unix::fs::symlink(&shared_exe, &exe_link).unwrap();
    }

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), true);

    // Both threads still land in the snapshot — the failed
    // attach just leaves allocated_bytes at the absent-counter
    // default of zero. If the cache-hit branch panicked
    // (poisoned mutex, key collision logic, etc.), the rayon
    // worker would crash and `capture_with` would not return.
    assert_eq!(
        snap.threads.len(),
        2,
        "both staged threads must land in the snapshot",
    );
    for thread in &snap.threads {
        assert_eq!(
            thread.allocated_bytes,
            Bytes(0),
            "synthetic /proc has no maps; attach fails, allocated_bytes \
             collapses to absent-counter zero — cache-hit branch must not \
             fabricate a non-zero counter",
        );
    }
}

// ------------------------------------------------------------
// Capture-pipeline error paths (Batch A + B)
//
// The synthetic-tree happy path is covered by
// capture_with_synthetic_tree_assembles_thread_state above.
// The tests below pin the pipeline's behavior against
// adversarial inputs:
// - missing/empty proc_root and tgid dirs (Batch A)
// - non-numeric junk under proc_root (Batch A)
// - capture_pid_with against pids that don't exist or are
//   ghost (Batch A + B)
// - selectively malformed/corrupted procfs files leaving
//   the matching ThreadState fields zero-defaulted (Batch B)
//
// Each test uses stage_synthetic_proc to lay down a known-
// good baseline, then mutates one specific axis. Assertions
// include observed value, expected value, and likely root
// cause so a regression points the reader at the failure
// mode without re-derivation.
// ------------------------------------------------------------

/// G1 — proc_root pointing at a directory that does NOT
/// exist must NOT panic. Pipeline collapses to an empty
/// snapshot via `iter_tgids_at`'s read_dir-fail-→-empty-Vec
/// guard. Defends against a future change that bubbled the
/// I/O error to the caller.
#[test]
fn capture_with_nonexistent_proc_root_produces_empty_snapshot() {
    let scratch = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    // A path inside a fresh tempdir that we never create —
    // guaranteed to not exist within this test's scope.
    // io::read_dir returns ENOENT, iter_tgids_at returns
    // Vec::new(). Use false for use_syscall_affinity so the
    // parallel probe phase is fully skipped. Reuse the same
    // nonexistent path for sys_root: this test exercises the
    // ENOENT-collapses-cleanly invariant uniformly.
    let nonexistent = scratch.path().join("does-not-exist");
    let snap = capture_with(&nonexistent, cgroup_tmp.path(), &nonexistent, false);
    assert!(
        snap.threads.is_empty(),
        "nonexistent proc_root must produce empty snapshot; got \
         {} threads — iter_tgids_at must collapse ENOENT to empty",
        snap.threads.len(),
    );
}

/// G2 — tgid directory present but missing the inner
/// `task/` subdirectory. `iter_task_ids_at` returns an
/// empty vec, so the per-tid loop runs zero iterations and
/// the tgid contributes no threads. Pins that the missing
/// `task/` does not crash or fabricate a synthetic tid.
#[test]
fn capture_with_tgid_missing_task_dir_yields_no_threads_for_that_tgid() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();

    // tgid 4242: has `task/` and one tid (live thread).
    // tgid 4243: numeric directory but NO `task/` subdir.
    let live_tgid: i32 = 4242;
    let live_tid: i32 = 101;
    stage_synthetic_proc(
        proc_tmp.path(),
        live_tgid,
        live_tid,
        "live-pcomm",
        "live-comm",
    );

    let bare_tgid: i32 = 4243;
    std::fs::create_dir_all(proc_tmp.path().join(bare_tgid.to_string())).unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);

    assert_eq!(
        snap.threads.len(),
        1,
        "tgid 4243 has no `task/` subdir → contributes zero threads; \
         only live tgid 4242's tid should land. got {} threads, expected 1",
        snap.threads.len(),
    );
    assert_eq!(snap.threads[0].tgid, live_tgid as u32);
    assert_eq!(snap.threads[0].tid, live_tid as u32);
}

/// G3 — non-numeric directory entries under proc_root
/// (real procfs has `self`, `thread-self`, `sys`, `kpageflags`,
/// etc.) MUST be filtered by the parse-as-i32 step in
/// `iter_tgids_at`. Pins the filter so a future refactor
/// that loosened it (e.g. accepted any digit-prefix) does
/// not surface kernel pseudo-files as fake tgids.
#[test]
fn capture_with_non_numeric_proc_entries_are_filtered() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();

    // Stage one valid numeric tgid plus several non-numeric
    // names that mimic real procfs entries.
    let live_tgid: i32 = 5151;
    let live_tid: i32 = 5152;
    stage_synthetic_proc(proc_tmp.path(), live_tgid, live_tid, "real", "real-thread");

    for junk in &["self", "thread-self", "sys", "version", "12abc", "abc"] {
        std::fs::create_dir_all(proc_tmp.path().join(junk)).unwrap();
    }
    // Negative or zero are filtered by `> 0` predicate.
    std::fs::create_dir_all(proc_tmp.path().join("0")).unwrap();
    std::fs::create_dir_all(proc_tmp.path().join("-1")).unwrap();

    // Direct check on the parse filter — pins iter_tgids_at
    // independently of the rest of the pipeline. Without this,
    // a loosened parse that accepted "12" from "12abc" would
    // still produce 1 thread downstream (the "12" dir has no
    // task/ subdir → contributes zero threads regardless), so
    // the snap.threads.len()==1 assertion alone wouldn't catch
    // the regression.
    assert_eq!(
        iter_tgids_at(proc_tmp.path()),
        vec![live_tgid],
        "iter_tgids_at must return only the real numeric tgid; \
         non-numeric and `0`/`-1` entries must be filtered by \
         parse::<i32>().ok() + `> 0` predicates",
    );

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);

    assert_eq!(
        snap.threads.len(),
        1,
        "non-numeric proc_root entries (`self`, `12abc`, etc.) and \
         `0`/`-1` must be filtered by iter_tgids_at; got {} threads, \
         expected 1 (only the real tgid {live_tgid})",
        snap.threads.len(),
    );
    assert_eq!(snap.threads[0].tgid, live_tgid as u32);
}

/// G7 — `capture_pid_with` against a pid whose `/proc/<pid>`
/// directory does not exist must NOT panic. `iter_task_ids_at`
/// returns empty, the loop iterates zero times, and the
/// snapshot's `threads` is empty. Pins that the per-pid
/// capture path tolerates the same exit-mid-capture race the
/// global path does.
#[test]
fn capture_pid_with_nonexistent_pid_produces_empty_snapshot() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    // pid 99999 is not staged — `proc_tmp/99999` does not exist.
    let snap = capture_pid_with(
        proc_tmp.path(),
        cgroup_tmp.path(),
        sys_tmp.path(),
        99999,
        false,
    );
    assert!(
        snap.threads.is_empty(),
        "capture_pid_with against nonexistent pid must produce empty \
         snapshot; got {} threads — iter_task_ids_at must collapse \
         ENOENT to empty",
        snap.threads.len(),
    );
}

/// G4a — corrupt the `stat` file so `parse_stat` returns
/// all-None defaults (write a single non-paren token, so
/// `rfind(')')` returns None and `parse_stat`
/// short-circuits to `StatFields::default()`). With `comm`
/// intact, the ghost-filter clause does NOT fire, so the
/// thread lands with stat-derived fields at zero (nice,
/// start_time, policy, processor, utime, stime) while
/// comm + status + io still populate from their intact
/// files.
#[test]
fn capture_with_corrupt_stat_file_zeroes_stat_fields_only() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 6161;
    let tid: i32 = 6162;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
    // Corrupt /proc/<tgid>/task/<tid>/stat — write a single
    // non-paren token so rfind(')') fails.
    let stat_path = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string())
        .join("stat");
    std::fs::write(&stat_path, "garbage no parens here\n").unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);

    assert_eq!(
        snap.threads.len(),
        1,
        "corrupt stat does not block thread landing — comm + status \
         + io still populate; ghost filter only fires when comm AND \
         start_time are both empty/zero. got {} threads",
        snap.threads.len(),
    );
    let t = &snap.threads[0];
    // stat-derived fields collapse to zero/default.
    assert_eq!(
        t.start_time_clock_ticks, 0,
        "corrupt stat → start_time_clock_ticks default 0; got {}",
        t.start_time_clock_ticks
    );
    use crate::metric_types::{
        Bytes, CategoricalString, ClockTicks, MonotonicCount, OrdinalI32,
    };
    assert_eq!(
        t.nice,
        OrdinalI32(0),
        "corrupt stat → nice default 0; got {}",
        t.nice.0,
    );
    assert_eq!(
        t.policy,
        CategoricalString::from(""),
        "corrupt stat → policy default empty; got {:?}",
        t.policy
    );
    assert_eq!(t.utime_clock_ticks, ClockTicks(0));
    assert_eq!(t.stime_clock_ticks, ClockTicks(0));
    assert_eq!(t.processor, OrdinalI32(0));
    // status-derived fields still populate.
    assert_eq!(
        t.voluntary_csw,
        MonotonicCount(42),
        "status file is intact → voluntary_csw still populates"
    );
    // io-derived fields still populate.
    assert_eq!(
        t.rchar,
        Bytes(100),
        "io file is intact → rchar still populates"
    );
}

/// G4b — missing `schedstat` file (kernel without
/// CONFIG_SCHEDSTATS) leaves run_time_ns / wait_time_ns /
/// timeslices at zero. The thread still lands because
/// stat/comm are intact.
#[test]
fn capture_with_missing_schedstat_zeroes_schedstat_fields() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 7171;
    let tid: i32 = 7172;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
    // Remove /proc/<tgid>/task/<tid>/schedstat.
    let schedstat_path = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string())
        .join("schedstat");
    std::fs::remove_file(&schedstat_path).unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
    assert_eq!(
        snap.threads.len(),
        1,
        "thread still lands with schedstat absent"
    );
    let t = &snap.threads[0];
    use crate::metric_types::{MonotonicCount, MonotonicNs};
    assert_eq!(
        t.run_time_ns,
        MonotonicNs(0),
        "missing schedstat → run_time_ns default 0; got {}",
        t.run_time_ns.0
    );
    assert_eq!(t.wait_time_ns, MonotonicNs(0));
    assert_eq!(t.timeslices, MonotonicCount(0));
    // start_time still populates from intact stat.
    assert_eq!(t.start_time_clock_ticks, 555_555);
}

/// G4c — malformed `status` file (random text, no recognized
/// keys) leaves status-derived fields (voluntary_csw,
/// nonvoluntary_csw, state, cpu_affinity) at default. With
/// `use_syscall_affinity=false`, cpu_affinity comes from
/// status only — so this also pins that absent
/// Cpus_allowed_list defaults to empty Vec, NOT to the
/// caller process's real affinity.
#[test]
fn capture_with_corrupt_status_zeroes_status_fields_and_empty_affinity() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 8181;
    let tid: i32 = 8182;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
    let status_path = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string())
        .join("status");
    // No `:` separators → split_once(':') returns None for
    // every line → no field populates.
    std::fs::write(&status_path, "totally malformed garbage no colons here\n").unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
    assert_eq!(snap.threads.len(), 1);
    let t = &snap.threads[0];
    use crate::metric_types::MonotonicCount;
    assert_eq!(
        t.voluntary_csw,
        MonotonicCount(0),
        "corrupt status → voluntary_csw default 0; got {}",
        t.voluntary_csw.0
    );
    assert_eq!(t.nonvoluntary_csw, MonotonicCount(0));
    assert_eq!(
        t.state, '~',
        "corrupt status → state collapses to '~' (capture-time \
         unwrap_or_else(default_state_char)); got {:?}",
        t.state
    );
    assert!(
        t.cpu_affinity.0.is_empty(),
        "use_syscall_affinity=false + corrupt status → cpu_affinity \
         must be empty Vec, NOT inherit caller's real affinity; got {:?}",
        t.cpu_affinity,
    );
}

/// G4d — missing `io` file (CONFIG_TASK_IO_ACCOUNTING off
/// at kernel build) leaves all 6 byte counters at zero.
/// Pins that the capture continues without io data rather
/// than failing the whole snapshot.
#[test]
fn capture_with_missing_io_zeroes_io_fields() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 9191;
    let tid: i32 = 9192;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
    let io_path = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string())
        .join("io");
    std::fs::remove_file(&io_path).unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
    assert_eq!(snap.threads.len(), 1);
    let t = &snap.threads[0];
    use crate::metric_types::{Bytes, MonotonicCount};
    assert_eq!(
        t.rchar,
        Bytes(0),
        "missing io → rchar default 0; got {}",
        t.rchar.0,
    );
    assert_eq!(t.wchar, Bytes(0));
    assert_eq!(t.syscr, MonotonicCount(0));
    assert_eq!(t.syscw, MonotonicCount(0));
    assert_eq!(t.read_bytes, Bytes(0));
    assert_eq!(t.write_bytes, Bytes(0));
    assert_eq!(t.cancelled_write_bytes, Bytes(0));
    // stat-derived fields still populate.
    assert_eq!(t.start_time_clock_ticks, 555_555);
}

/// G4e — missing `sched` file leaves every sched-derived
/// field at zero (nr_wakeups family, *_sum, *_max,
/// migrations, ext_enabled). The thread still lands because
/// stat is intact.
#[test]
fn capture_with_missing_sched_zeroes_sched_fields() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 1010;
    let tid: i32 = 1011;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
    let sched_path = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string())
        .join("sched");
    std::fs::remove_file(&sched_path).unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
    assert_eq!(snap.threads.len(), 1);
    let t = &snap.threads[0];
    use crate::metric_types::{MonotonicCount, MonotonicNs, PeakNs};
    assert_eq!(
        t.nr_wakeups,
        MonotonicCount(0),
        "missing sched → nr_wakeups default 0; got {}",
        t.nr_wakeups.0,
    );
    assert_eq!(t.nr_migrations, MonotonicCount(0));
    assert_eq!(t.wait_sum, MonotonicNs(0));
    assert_eq!(t.wait_max, PeakNs(0));
    assert_eq!(t.voluntary_sleep_ns, MonotonicNs(0));
    assert_eq!(t.block_sum, MonotonicNs(0));
    assert_eq!(t.iowait_sum, MonotonicNs(0));
    assert_eq!(t.exec_max, PeakNs(0));
    assert_eq!(t.slice_max, PeakNs(0));
    assert!(
        !t.ext_enabled,
        "missing sched → ext.enabled key absent → ext_enabled false; \
         got {}",
        t.ext_enabled
    );
}

/// G5 — selectively delete EVERY non-comm file under one tid
/// to simulate a partial mid-capture race (readdir saw the
/// dir, then the kernel completed exit cleanup before our
/// per-file reads). With comm intact, the thread still
/// lands but every counter is zero. Pins the absent-=-zero
/// contract under the worst plausible mid-capture race.
#[test]
fn capture_with_partial_mid_capture_race_lands_zero_thread() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 1212;
    let tid: i32 = 1213;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "racy-pcomm", "racy-comm");
    let task_dir = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string());
    // Remove every per-tid file EXCEPT comm. comm is the
    // ghost filter's anchor — keeping it preserves the
    // thread's identity so the test exercises the
    // counters-zero path rather than the ghost-drop path.
    for f in &["stat", "schedstat", "status", "io", "sched", "cgroup"] {
        std::fs::remove_file(task_dir.join(f)).unwrap();
    }

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
    assert_eq!(snap.threads.len(), 1, "comm intact → thread still lands");
    let t = &snap.threads[0];
    use crate::metric_types::{Bytes, MonotonicCount, MonotonicNs};
    assert_eq!(t.comm, "racy-comm", "comm survives the racy partial reads");
    // Every counter zeros.
    assert_eq!(t.start_time_clock_ticks, 0);
    assert_eq!(t.nr_wakeups, MonotonicCount(0));
    assert_eq!(t.run_time_ns, MonotonicNs(0));
    assert_eq!(t.voluntary_csw, MonotonicCount(0));
    assert_eq!(t.rchar, Bytes(0));
    assert_eq!(t.minflt, MonotonicCount(0));
    assert_eq!(t.cgroup, "");
    assert!(
        snap.cgroup_stats.is_empty(),
        "all threads have empty cgroup → enrichment loop skips → \
         cgroup_stats stays empty",
    );
}

/// G6 — `capture_pid_with` ghost filter: a tid directory
/// under the target pid exists but carries zero readable
/// files (mid-capture exit). `capture_pid_with`'s
/// terminal ghost-filter check — same shape as the global
/// `capture_with` path's filter — must drop the
/// all-Default ThreadState. Pins the per-pid path's filter
/// independently of the global path.
#[test]
fn capture_pid_with_filters_ghost_threads() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 1313;
    let live_tid: i32 = 1314;
    let ghost_tid: i32 = 1315;

    stage_synthetic_proc(proc_tmp.path(), tgid, live_tid, "p", "live");

    // Ghost tid: directory exists but empty (no files).
    let ghost_dir = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(ghost_tid.to_string());
    std::fs::create_dir_all(&ghost_dir).unwrap();

    let snap = capture_pid_with(
        proc_tmp.path(),
        cgroup_tmp.path(),
        sys_tmp.path(),
        tgid,
        false,
    );

    assert_eq!(
        snap.threads.len(),
        1,
        "capture_pid_with must filter ghost tid {ghost_tid}; got {} \
         threads, expected 1 (only live tid {live_tid})",
        snap.threads.len(),
    );
    assert_eq!(snap.threads[0].tid, live_tid as u32);
}

/// G8 — malformed `Cpus_allowed_list:` value (a reversed
/// range like `5-3`) routes through `parse_cpu_list` which
/// returns `None`. With `use_syscall_affinity=false`, the
/// capture site has no fallback and `cpu_affinity` stays
/// at the default empty Vec. Pins that a malformed cpulist
/// does NOT crash the parse and does NOT silently fabricate
/// a partial range.
#[test]
fn capture_with_malformed_cpus_allowed_list_yields_empty_affinity() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 1414;
    let tid: i32 = 1415;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");

    let status_path = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string())
        .join("status");
    // Reversed range — parse_cpu_list rejects (returns None).
    let status = "Name:\tfoo\n\
         State:\tR (running)\n\
         voluntary_ctxt_switches:\t1\n\
         nonvoluntary_ctxt_switches:\t1\n\
         Cpus_allowed_list:\t5-3\n";
    std::fs::write(&status_path, status).unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
    assert_eq!(snap.threads.len(), 1);
    let t = &snap.threads[0];
    use crate::metric_types::MonotonicCount;
    assert!(
        t.cpu_affinity.0.is_empty(),
        "malformed Cpus_allowed_list `5-3` → parse_cpu_list returns \
         None → cpu_affinity defaults to empty Vec (NOT a partial \
         range, NOT the caller's affinity); got {:?}",
        t.cpu_affinity,
    );
    // Other status fields still populate (the malformed
    // line failed only the cpulist arm of parse_status).
    assert_eq!(
        t.voluntary_csw,
        MonotonicCount(1),
        "malformed cpulist must NOT corrupt csw fields on the same \
         status file — per-arm Option isolation"
    );
}

/// G11 — huge `Cpus_allowed_list:` range (above the
/// MAX_CPU_RANGE_EXPANSION cap at 64 Ki CPUs) routes
/// through the `parse_cpu_list` cap and returns `None`.
/// Same observable effect as G8 (empty Vec) but pins a
/// distinct adversarial input — a hostile /proc with a
/// `0-4294967295` cpulist must NOT allocate gigabytes.
#[test]
fn capture_with_huge_cpu_range_in_status_yields_empty_affinity() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 1515;
    let tid: i32 = 1516;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");

    let status_path = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string())
        .join("status");
    // u32::MAX-spanning range — well above the 64 Ki cap;
    // parse_cpu_list rejects without expansion.
    let status = "Cpus_allowed_list:\t0-4294967295\n\
         voluntary_ctxt_switches:\t1\n\
         nonvoluntary_ctxt_switches:\t1\n";
    std::fs::write(&status_path, status).unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
    assert_eq!(snap.threads.len(), 1);
    let t = &snap.threads[0];
    use crate::metric_types::MonotonicCount;
    assert!(
        t.cpu_affinity.0.is_empty(),
        "huge cpulist range `0-4294967295` exceeds the 64 Ki \
         expansion cap → parse_cpu_list returns None → cpu_affinity \
         empty (NOT a 4-billion-element Vec, NOT a partial range); \
         got {} elements",
        t.cpu_affinity.0.len(),
    );
    // Per-arm isolation: the cap-rejected cpulist must NOT
    // crash the rest of parse_status. csw fields on the same
    // file still populate. Mirrors G8's isolation check.
    assert_eq!(
        t.voluntary_csw,
        MonotonicCount(1),
        "huge cpulist rejection must not break csw parsing on the \
         same status file — per-arm Option isolation"
    );
}

/// G9 — non-numeric directory entries under `<proc_root>/<tgid>/task/`
/// MUST be filtered by the parse-as-i32 step in
/// `iter_task_ids_at`. Mirrors G3 for the per-tgid `task/` subdir
/// (G3 covers `<proc_root>` itself). Real procfs has only numeric
/// task entries, but a hostile or malformed test fixture could
/// stage non-numeric names; the filter must drop them rather
/// than surface garbage tids.
#[test]
fn capture_with_non_numeric_task_entries_are_filtered() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();

    let live_tgid: i32 = 8181;
    let live_tid: i32 = 8182;
    stage_synthetic_proc(proc_tmp.path(), live_tgid, live_tid, "real", "real-thread");

    // Stage non-numeric entries alongside the real tid under
    // <tgid>/task/. iter_task_ids_at must filter on parse::<i32>().
    let task_dir = proc_tmp.path().join(live_tgid.to_string()).join("task");
    for junk in &["status", "self", "12abc", "abc"] {
        std::fs::create_dir_all(task_dir.join(junk)).unwrap();
    }
    std::fs::create_dir_all(task_dir.join("0")).unwrap();
    std::fs::create_dir_all(task_dir.join("-1")).unwrap();

    // Direct check on the parse filter — pins iter_task_ids_at
    // independently of the rest of the pipeline.
    assert_eq!(
        iter_task_ids_at(proc_tmp.path(), live_tgid),
        vec![live_tid],
        "iter_task_ids_at must return only the real numeric tid; \
         non-numeric and `0`/`-1` entries must be filtered by \
         parse::<i32>().ok() + `> 0` predicates",
    );

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
    assert_eq!(
        snap.threads.len(),
        1,
        "non-numeric `task/` entries must be filtered by \
         iter_task_ids_at; got {} threads, expected 1",
        snap.threads.len(),
    );
    assert_eq!(snap.threads[0].tid, live_tid as u32);
}

/// G10 — a tgid emitting a v1-only `cgroup` file (legacy
/// hierarchy entries, no `0::` unified line) lands the thread
/// with `cgroup` defaulting to "". The ghost filter does NOT
/// fire because comm + start_time are intact. The empty cgroup
/// is a legitimate observable signal — `capture_with`'s
/// cgroup_stats enrichment loop skips entries with empty
/// `cgroup` so no synthetic stats land for the missing path.
#[test]
fn capture_with_v1_only_cgroup_yields_empty_cgroup_string() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 9191;
    let tid: i32 = 9192;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");

    // Overwrite the cgroup file with only legacy v1 lines —
    // parse_cgroup_v2 returns None, read_cgroup_at returns
    // None, ThreadState.cgroup defaults to "".
    let cgroup_path = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string())
        .join("cgroup");
    let v1_only = "12:cpuset:/legacy/cpuset/path\n\
         5:freezer:/legacy/freezer\n\
         3:blkio:/\n";
    std::fs::write(&cgroup_path, v1_only).unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);

    assert_eq!(
        snap.threads.len(),
        1,
        "v1-only cgroup does not block thread landing — comm + \
         start_time are intact, ghost filter does not fire; \
         got {} threads",
        snap.threads.len(),
    );
    let t = &snap.threads[0];
    assert_eq!(
        t.cgroup, "",
        "v1-only cgroup file → parse_cgroup_v2 returns None → \
         ThreadState.cgroup defaults to empty; got {:?}",
        t.cgroup,
    );
    // cgroup_stats enrichment skips empty-cgroup threads. The
    // map must not carry an entry keyed on "" (would otherwise
    // accumulate a meaningless aggregate row in the snapshot).
    assert!(
        !snap.cgroup_stats.contains_key(""),
        "empty-cgroup thread must NOT seed an empty-key entry in \
         cgroup_stats — the enrichment loop's `!is_empty()` guard \
         pins the skip; got keys: {:?}",
        snap.cgroup_stats.keys().collect::<Vec<_>>(),
    );
}

/// `capture_to` propagates write errors through anyhow with the
/// destination path in the context chain so an operator who
/// passed an unwritable target sees the path in the diagnostic
/// rather than a bare I/O error. Pins the `with_context` wrapper
/// at the public-API boundary; without it, the error message
/// loses the path and operators can't tell which target failed.
#[test]
fn capture_to_returns_err_on_unwritable_path() {
    // A path under a directory that does not exist — std::fs::write
    // returns ENOENT for the parent; capture_to's with_context
    // wraps it with the destination path.
    let scratch = tempfile::TempDir::new().unwrap();
    let unwritable = scratch.path().join("missing-dir").join("snap.ctprof.zst");
    let err = capture_to(&unwritable).unwrap_err();
    let chain = format!("{err:#}");
    assert!(
        chain.contains(unwritable.to_string_lossy().as_ref()),
        "error chain must name the unwritable target path; got: {chain}",
    );
}

/// `read_cgroup_stats_at` reads from the path string verbatim;
/// when the path names a cgroup directory that does not exist
/// (the thread's cgroup string was captured but the cgroup has
/// since been rmdir'd, or the cgroup_root differs from the live
/// host), every cpu.stat / memory.current read fails with
/// ENOENT and the resulting `CgroupStats` is all-zero. Pins the
/// "absent = 0" contract for the enrichment loop's stale-string
/// race.
#[test]
fn capture_with_stale_cgroup_path_yields_all_zero_stats() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 7373;
    let tid: i32 = 7374;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
    // stage_synthetic_proc writes "0::/ktstr.slice/worker0" into
    // the cgroup file but does NOT create the matching directory
    // under cgroup_root. The enrichment loop calls
    // read_cgroup_stats_at("/ktstr.slice/worker0"), which
    // resolves to a non-existent dir and returns all-zero stats.

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
    assert_eq!(snap.threads.len(), 1);
    let stats = snap
        .cgroup_stats
        .get("/ktstr.slice/worker0")
        .expect("non-empty cgroup string must seed the stats map");
    assert_eq!(stats.cpu.usage_usec, 0, "stale cgroup → cpu_usage_usec 0");
    assert_eq!(stats.cpu.nr_throttled, 0, "stale cgroup → nr_throttled 0");
    assert_eq!(
        stats.cpu.throttled_usec, 0,
        "stale cgroup → throttled_usec 0"
    );
    assert_eq!(stats.memory.current, 0, "stale cgroup → memory_current 0");
}

/// `read_cgroup_at` returns `None` when the cgroup file is
/// present but contains only v1 hierarchy lines (no `0::`
/// unified prefix). Pins the "v1-only → None" path of
/// `parse_cgroup_v2` from the file-read entry point — distinct
/// from `parse_cgroup_v2_none_when_only_legacy_present` which
/// pins the parse function in isolation.
#[test]
fn read_cgroup_at_v1_only_cgroup_returns_none() {
    let tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 4242;
    let tid: i32 = 4243;
    let task_dir = tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string());
    std::fs::create_dir_all(&task_dir).unwrap();
    let v1_only = "12:cpuset:/legacy/cpuset/path\n\
         5:freezer:/legacy/freezer\n";
    std::fs::write(task_dir.join("cgroup"), v1_only).unwrap();

    assert_eq!(
        read_cgroup_at(tmp.path(), tgid, tid),
        None,
        "v1-only cgroup file → read_cgroup_at returns None (no 0:: line)",
    );

    // Symmetric missing-file branch: no cgroup file → None.
    assert_eq!(
        read_cgroup_at(tmp.path(), tgid, 9999),
        None,
        "missing cgroup file → read_cgroup_at returns None",
    );
}

/// `parse_cgroup_v2` accepts the degenerate "/" root path. A
/// process cgrouped at the unified root emits "0::/" and the
/// parser returns Some("/"). Pins the boundary distinct from
/// `parse_cgroup_v2_empty_path_and_multiple_unified_lines`
/// (which covers "0::" with empty-string-after-prefix); this
/// test pins that "/" alone is treated as a valid path, not
/// folded into the empty-string rejection.
#[test]
fn parse_cgroup_v2_root_only_path_returns_slash() {
    // Single "0::/" line — the trim + non-empty guard accepts
    // "/" as a valid path.
    assert_eq!(parse_cgroup_v2("0::/\n"), Some("/".to_string()));
    // Same with trailing whitespace — trim absorbs it but "/"
    // survives as the post-trim value.
    assert_eq!(parse_cgroup_v2("0::/  \n"), Some("/".to_string()));
    // Mixed alongside legacy v1 lines — unified picks "/".
    let raw = "12:cpuset:/legacy/path\n0::/\n5:freezer:/legacy\n";
    assert_eq!(parse_cgroup_v2(raw), Some("/".to_string()));
}

/// `nr_threads` leader-dedup: only the thread leader
/// (tid == tgid) carries the `signal_struct->nr_threads`
/// snapshot; non-leader threads zero the field. Pin the
/// `tid == tgid` gate at the [`ThreadState`] construction
/// site so a regression that populated nr_threads on every
/// thread would surface here as a Sum-rule aggregator
/// multiplying the count by itself across the bucket.
///
/// Stages two threads under one tgid: the leader (tid ==
/// tgid) and a worker (tid != tgid). `/proc/<tid>/status`
/// for both carries `Threads: 2` (every thread of the
/// process sees the same kernel-emitted value), but only
/// the leader's [`ThreadState::nr_threads`] reads 2; the
/// worker reads 0.
#[test]
fn capture_with_nr_threads_dedup_populates_leader_only() {
    use crate::metric_types::GaugeCount;
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 7000;
    let leader_tid: i32 = 7000;
    let worker_tid: i32 = 7001;

    // Stage the leader thread fully — `Threads: 2` lands on
    // the leader's status file.
    stage_synthetic_proc(proc_tmp.path(), tgid, leader_tid, "leader-pcomm", "leader-comm");
    let leader_status_path = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(leader_tid.to_string())
        .join("status");
    let leader_status = "Name:\tfoo\n\
        State:\tR (running)\n\
        voluntary_ctxt_switches:\t1\n\
        nonvoluntary_ctxt_switches:\t1\n\
        Cpus_allowed_list:\t0\n\
        Threads:\t2\n";
    std::fs::write(&leader_status_path, leader_status).unwrap();

    // Stage the worker thread under the same tgid. Same
    // `Threads: 2` value (every thread of the process reads
    // the kernel-shared field). Distinct comm so the test can
    // tell the two ThreadState entries apart in the output
    // vector.
    stage_synthetic_proc(proc_tmp.path(), tgid, worker_tid, "leader-pcomm", "worker-comm");
    let worker_status_path = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(worker_tid.to_string())
        .join("status");
    let worker_status = "Name:\tfoo\n\
        State:\tR (running)\n\
        voluntary_ctxt_switches:\t1\n\
        nonvoluntary_ctxt_switches:\t1\n\
        Cpus_allowed_list:\t0\n\
        Threads:\t2\n";
    std::fs::write(&worker_status_path, worker_status).unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
    assert_eq!(snap.threads.len(), 2, "two threads under tgid {tgid}");

    let leader = snap
        .threads
        .iter()
        .find(|t| t.tid == leader_tid as u32)
        .expect("leader thread present");
    let worker = snap
        .threads
        .iter()
        .find(|t| t.tid == worker_tid as u32)
        .expect("worker thread present");

    assert_eq!(leader.tid, leader.tgid, "leader: tid == tgid");
    assert_ne!(worker.tid, worker.tgid, "worker: tid != tgid");

    // Leader carries the kernel-emitted count.
    assert_eq!(
        leader.nr_threads,
        GaugeCount(2),
        "leader.nr_threads must carry the parsed Threads: value (2); \
         got {:?}",
        leader.nr_threads,
    );
    // Worker zeros the field.
    assert_eq!(
        worker.nr_threads,
        GaugeCount(0),
        "worker.nr_threads must zero out under leader-dedup; \
         got {:?} — populating non-leaders would let any Sum-style \
         aggregator multiply the count by itself across the bucket",
        worker.nr_threads,
    );
}

/// Ghost filter four-corner matrix: pin each of the four
/// (comm, start_time) corners. The filter at the capture
/// pipeline tail reads:
///
/// ```ignore
/// if t.comm.is_empty() && t.start_time_clock_ticks == 0 { drop }
/// ```
///
/// AND-semantics — both halves must be empty/zero for the
/// filter to fire. The other three corners must surface in
/// the output.
///
/// Existing coverage:
/// - (empty, 0) is pinned by
///   `capture_with_filters_ghost_threads_with_empty_comm_and_zero_start`
/// - (empty, nonzero) is pinned by
///   `capture_with_empty_comm_nonzero_start_time_keeps_thread`
///
/// This test fills the remaining corners:
/// - (nonempty, 0): a thread with a parseable comm but a
///   missing/zero start_time field (e.g. malformed stat
///   line that the parser couldn't lift `starttime` out of)
///   must NOT be filtered — comm alone keeps it alive.
/// - (nonempty, nonzero): the canonical live thread; pin
///   that the standard well-formed-procfs path lands a
///   thread with both halves populated.
#[test]
fn capture_with_ghost_filter_four_corner_keeps_nonzero_halves() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();

    // Corner (nonempty comm, nonzero start) — canonical live
    // thread under tgid 8000. Ghost filter must not fire.
    let tgid_alive: i32 = 8000;
    let tid_alive: i32 = 8001;
    stage_synthetic_proc(
        proc_tmp.path(),
        tgid_alive,
        tid_alive,
        "alive-pcomm",
        "alive-comm",
    );

    // Corner (nonempty comm, zero start) — comm reads
    // `parseable-comm` but `/proc/<tid>/stat` is missing so
    // every parsed field defaults to zero, including
    // start_time_clock_ticks. Ghost filter must NOT fire on
    // this corner — comm alone is non-empty.
    let tgid_no_stat: i32 = 8002;
    let tid_no_stat: i32 = 8003;
    let task_dir = proc_tmp
        .path()
        .join(tgid_no_stat.to_string())
        .join("task")
        .join(tid_no_stat.to_string());
    std::fs::create_dir_all(&task_dir).unwrap();
    // tgid /comm
    std::fs::write(
        proc_tmp
            .path()
            .join(tgid_no_stat.to_string())
            .join("comm"),
        "parseable-pcomm\n",
    )
    .unwrap();
    // task/<tid>/comm — non-empty so the AND-clause's first
    // half is false and the filter must skip the thread.
    std::fs::write(task_dir.join("comm"), "parseable-comm\n").unwrap();
    // No stat / schedstat / status / sched / io / cgroup
    // files: every read returns the parser's default (zero
    // for numeric fields, empty for strings). start_time
    // therefore lands at 0.

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);

    // The `(nonempty, nonzero)` corner — alive thread.
    let alive = snap
        .threads
        .iter()
        .find(|t| t.tid == tid_alive as u32)
        .expect("alive thread (nonempty comm, nonzero start) must surface");
    assert!(
        !alive.comm.is_empty(),
        "alive thread carries non-empty comm",
    );
    assert_ne!(
        alive.start_time_clock_ticks, 0,
        "alive thread carries non-zero start_time",
    );

    // The `(nonempty, 0)` corner — comm alone keeps the
    // thread alive even when start_time is zero.
    let comm_only = snap
        .threads
        .iter()
        .find(|t| t.tid == tid_no_stat as u32)
        .expect(
            "comm-only thread must surface; ghost filter is AND-gated, \
             so non-empty comm with zero start_time must NOT be filtered",
        );
    assert_eq!(
        comm_only.comm, "parseable-comm",
        "comm-only thread surfaces with the parsed non-empty comm",
    );
    assert_eq!(
        comm_only.start_time_clock_ticks, 0,
        "comm-only thread has zero start_time (no stat file staged)",
    );
}

/// `ThreadState::smaps_rollup_bytes` saturating boundary:
/// a kilobyte value of `u64::MAX` clamps to `u64::MAX` bytes
/// after the `* 1024` conversion rather than overflowing. Pin
/// the `saturating_mul(1024)` arm at the upper boundary so a
/// regression that flipped to `wrapping_mul` or unchecked
/// multiplication would silently corrupt the rendered byte
/// count (a snapshot file with `u64::MAX` Rss kB would render
/// as a tiny number after wraparound).
///
/// Also pin the lower boundary: a 0 kB value converts to 0
/// bytes (no saturation needed, but the field's expected zero
/// behaviour is part of the contract).
#[test]
fn smaps_rollup_bytes_saturating_mul_clamps_at_u64_max() {
    use crate::metric_types::Bytes;
    let mut t = ThreadState {
        tid: 1,
        tgid: 1,
        ..ThreadState::default()
    };
    // Boundary: u64::MAX kB. saturating_mul(1024) clamps to
    // u64::MAX rather than wrapping. The rendered Bytes
    // wrapper carries u64::MAX so the auto-scale ladder caps
    // at the largest representable byte count.
    t.smaps_rollup_kb.insert("Rss".into(), u64::MAX);
    // Below-saturation reference value: 4 KiB → 4 * 1024 =
    // 4096 bytes, no saturation.
    t.smaps_rollup_kb.insert("Pss".into(), 4);
    // Lower boundary: 0 kB → 0 bytes.
    t.smaps_rollup_kb.insert("Shared_Clean".into(), 0);
    // Just below saturation: floor((u64::MAX - 1) / 1024)
    // multiplied by 1024 must NOT saturate. Pick the largest
    // value whose * 1024 conversion fits in u64.
    let near_max_kb = u64::MAX / 1024;
    t.smaps_rollup_kb.insert("Anonymous".into(), near_max_kb);

    let map: std::collections::BTreeMap<&String, Bytes> =
        t.smaps_rollup_bytes().collect();

    let rss_key = "Rss".to_string();
    let pss_key = "Pss".to_string();
    let shared_key = "Shared_Clean".to_string();
    let anon_key = "Anonymous".to_string();

    assert_eq!(
        map[&rss_key],
        Bytes(u64::MAX),
        "u64::MAX kB must saturate at u64::MAX bytes; got {:?}",
        map[&rss_key],
    );
    assert_eq!(
        map[&pss_key],
        Bytes(4 * 1024),
        "4 kB must convert to 4096 bytes; got {:?}",
        map[&pss_key],
    );
    assert_eq!(
        map[&shared_key],
        Bytes(0),
        "0 kB must convert to 0 bytes; got {:?}",
        map[&shared_key],
    );
    // near_max_kb * 1024 fits inside u64 — no saturation
    // (saturating_mul returns the exact product when it
    // doesn't overflow).
    let expected_anon = near_max_kb.saturating_mul(1024);
    assert_eq!(
        map[&anon_key],
        Bytes(expected_anon),
        "below-saturation kB value must convert exactly; \
         got {:?}, expected {:?}",
        map[&anon_key],
        Bytes(expected_anon),
    );
    assert!(
        expected_anon < u64::MAX,
        "test fixture: near_max_kb * 1024 must NOT saturate \
         so the boundary distinction is meaningful; got {expected_anon}",
    );
}
