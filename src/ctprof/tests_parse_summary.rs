//! ParseSummary aggregation across synthetic /proc trees + capture-with phase1 edge cases.
//!
//! Co-located with `super::mod.rs`; one of the topic-grouped
//! split files that replace the monolithic `tests.rs`.

#![cfg(test)]

use super::*;
use crate::metric_types::MonotonicCount;
use std::path::Path;
use super::tests_helpers::stage_synthetic_proc;

// ------------------------------------------------------------
// T28 — CtprofParseSummary: per-file read-failure tally
// ------------------------------------------------------------

/// Stage a synthetic procfs tree for parse-summary tests:
/// a single live tgid + tid with `comm` and `stat` populated
/// so the ghost filter does NOT fire (start_time is parseable
/// from `stat`). The caller then deletes the specific
/// per-file targets they want to fail. `cgroup` and other
/// non-asserted files are populated so the surrounding reads
/// succeed and the tally only counts the targeted failures.
fn stage_minimal_proc_for_parse(root: &Path, tgid: i32, tid: i32) {
    use std::fs;
    let tgid_dir = root.join(tgid.to_string());
    let task_dir = tgid_dir.join("task").join(tid.to_string());
    fs::create_dir_all(&task_dir).unwrap();
    fs::write(tgid_dir.join("comm"), "p\n").unwrap();
    fs::write(task_dir.join("comm"), "live\n").unwrap();
    // Non-zero start_time keeps the ghost filter from firing
    // even when other files vanish.
    let stat_line = format!(
        "{tid} (live) R 1 2 3 4 5 6 7 0 8 0 10 11 12 13 14 0 1 0 \
         555555 100 200 300 400 500 600 700 800 900 1000 1100 \
         1200 1300 1400 1500 1600 1700 1800 0\n"
    );
    fs::write(task_dir.join("stat"), stat_line).unwrap();
    fs::write(task_dir.join("schedstat"), "0 0 0\n").unwrap();
    fs::write(
        task_dir.join("status"),
        "voluntary_ctxt_switches:\t0\n\
         nonvoluntary_ctxt_switches:\t0\n",
    )
    .unwrap();
    fs::write(task_dir.join("io"), "rchar: 0\n").unwrap();
    fs::write(task_dir.join("sched"), "").unwrap();
    fs::write(task_dir.join("cgroup"), "0::/\n").unwrap();
}

/// Per-file-kind tally: deleting `schedstat` lands a single
/// `"schedstat"` failure in the summary's per-file map. Other
/// categories stay at zero (key absent from the map).
#[test]
fn parse_summary_records_schedstat_failure() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 5050;
    let tid: i32 = 5051;
    stage_minimal_proc_for_parse(proc_tmp.path(), tgid, tid);
    // Delete schedstat so the read fails.
    std::fs::remove_file(
        proc_tmp
            .path()
            .join(tgid.to_string())
            .join("task")
            .join(tid.to_string())
            .join("schedstat"),
    )
    .unwrap();

    // capture_with(_, _, false) skips the production gate so
    // parse_summary is None; use true and stage a /proc tree
    // that the host_context probe absorbs without panicking.
    // For the synthetic-tree pattern, stage a tally directly.
    let mut tally = ParseTally::default();
    let mut tally_opt: Option<&mut ParseTally> = Some(&mut tally);
    tally_opt.as_mut().unwrap().tids_walked += 1;
    let _ = capture_thread_at_with_tally(
        proc_tmp.path(),
        tgid,
        tid,
        "p",
        "live",
        false,
        &mut tally_opt,
    );
    tally_opt.as_mut().unwrap().commit_pending();

    let summary = tally.to_public();
    assert_eq!(summary.tids_walked, 1);
    assert_eq!(summary.read_failures, 1);
    assert_eq!(summary.read_failures_by_file.get("schedstat"), Some(&1));
    assert!(!summary.read_failures_by_file.contains_key("stat"));
    assert!(!summary.read_failures_by_file.contains_key("io"));
}

/// Per-file-kind tally: deleting `io` lands an `"io"` failure.
#[test]
fn parse_summary_records_io_failure() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 5060;
    let tid: i32 = 5061;
    stage_minimal_proc_for_parse(proc_tmp.path(), tgid, tid);
    std::fs::remove_file(
        proc_tmp
            .path()
            .join(tgid.to_string())
            .join("task")
            .join(tid.to_string())
            .join("io"),
    )
    .unwrap();

    let mut tally = ParseTally::default();
    let mut tally_opt: Option<&mut ParseTally> = Some(&mut tally);
    tally_opt.as_mut().unwrap().tids_walked += 1;
    let _ = capture_thread_at_with_tally(
        proc_tmp.path(),
        tgid,
        tid,
        "p",
        "live",
        false,
        &mut tally_opt,
    );
    tally_opt.as_mut().unwrap().commit_pending();

    let summary = tally.to_public();
    assert_eq!(summary.read_failures_by_file.get("io"), Some(&1));
}

/// Per-file-kind tally: a fully populated synthetic /proc
/// (every reader succeeds) lands an empty map and zero
/// `read_failures`. Pins the "absent key = zero" contract.
#[test]
fn parse_summary_clean_proc_yields_empty_map() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 5070;
    let tid: i32 = 5071;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");

    let mut tally = ParseTally::default();
    let mut tally_opt: Option<&mut ParseTally> = Some(&mut tally);
    tally_opt.as_mut().unwrap().tids_walked += 1;
    let _ = capture_thread_at_with_tally(
        proc_tmp.path(),
        tgid,
        tid,
        "p",
        "live",
        false,
        &mut tally_opt,
    );
    tally_opt.as_mut().unwrap().commit_pending();

    let summary = tally.to_public();
    assert_eq!(summary.tids_walked, 1);
    assert_eq!(summary.read_failures, 0);
    assert!(
        summary.read_failures_by_file.is_empty(),
        "clean procfs must yield an empty map, got {:?}",
        summary.read_failures_by_file,
    );
    assert!(summary.dominant_read_failure.is_none());
    assert!(!summary.kernel_config_dominant);
}

/// Ghost filter discipline (T28.2): a tid that exits between
/// readdir and the per-file reads (every read fails with
/// ENOENT, comm is empty, ghost filter rejects the tid) must
/// NOT contribute to the parse-summary tally. Otherwise a
/// busy host with mid-capture exits would inflate
/// `read_failures` with bumps that correspond to threads the
/// snapshot doesn't even contain.
#[test]
fn parse_summary_excludes_ghost_filtered_tids() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 5080;
    let tid: i32 = 5081;
    // Stage only the empty task directory (no comm, no stat,
    // no other files) so every read fails AND the ghost filter
    // fires (empty comm + zero start_time).
    let task_dir = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string());
    std::fs::create_dir_all(&task_dir).unwrap();

    let mut tally = ParseTally::default();
    let mut tally_opt: Option<&mut ParseTally> = Some(&mut tally);
    tally_opt.as_mut().unwrap().tids_walked += 1;
    let t =
        capture_thread_at_with_tally(proc_tmp.path(), tgid, tid, "", "", false, &mut tally_opt);
    // Ghost filter: empty comm + zero start_time → discard.
    if t.comm.is_empty() && t.start_time_clock_ticks == 0 {
        tally_opt.as_mut().unwrap().discard_pending();
    } else {
        tally_opt.as_mut().unwrap().commit_pending();
    }

    let summary = tally.to_public();
    assert_eq!(
        summary.read_failures, 0,
        "ghost-filtered tid must NOT contribute to read_failures; \
         got {} failures (the discard_pending unwind is broken)",
        summary.read_failures,
    );
    assert!(summary.read_failures_by_file.is_empty());
    // tids_walked still incremented — the tid was attempted.
    assert_eq!(summary.tids_walked, 1);
}

/// Serde round-trip: a populated `CtprofParseSummary`
/// preserves every field through JSON.
#[test]
fn parse_summary_serde_round_trip() {
    let mut by_file = BTreeMap::new();
    by_file.insert("schedstat".to_string(), 100);
    by_file.insert("io".to_string(), 50);
    let summary = CtprofParseSummary {
        tids_walked: 1000,
        read_failures: 150,
        read_failures_by_file: by_file,
        dominant_read_failure: Some("schedstat".to_string()),
        kernel_config_dominant: true,
        negative_dotted_values: 7,
    };
    let json = serde_json::to_string(&summary).unwrap();
    let back: CtprofParseSummary = serde_json::from_str(&json).unwrap();
    assert_eq!(back.tids_walked, 1000);
    assert_eq!(back.read_failures, 150);
    assert_eq!(back.read_failures_by_file.get("schedstat"), Some(&100));
    assert_eq!(back.read_failures_by_file.get("io"), Some(&50));
    assert_eq!(back.dominant_read_failure.as_deref(), Some("schedstat"));
    assert!(back.kernel_config_dominant);
    assert_eq!(
        back.negative_dotted_values, 7,
        "negative_dotted_values surfaces in the public surface \
         and round-trips through JSON",
    );
}

/// `dominant_read_failure` picks the file kind with the most
/// failures. Ties resolve REVERSE-alphabetically (mirrors the
/// probe-summary comparator) — alphabetically-EARLIER tag
/// wins.
#[test]
fn parse_summary_dominant_picks_max_file_kind() {
    let mut tally = ParseTally::default();
    // schedstat: 10 failures, io: 5, status: 5. schedstat wins.
    for _ in 0..10 {
        tally.record_failure("schedstat");
    }
    for _ in 0..5 {
        tally.record_failure("io");
    }
    for _ in 0..5 {
        tally.record_failure("status");
    }
    tally.commit_pending();
    let summary = tally.to_public();
    assert_eq!(summary.dominant_read_failure.as_deref(), Some("schedstat"));

    // Tie between io and status (same count) — io wins (earlier
    // alphabetical, matches the reverse-alphabetical comparator).
    let mut tally2 = ParseTally::default();
    for _ in 0..3 {
        tally2.record_failure("io");
    }
    for _ in 0..3 {
        tally2.record_failure("status");
    }
    tally2.commit_pending();
    let summary2 = tally2.to_public();
    assert_eq!(
        summary2.dominant_read_failure.as_deref(),
        Some("io"),
        "tie must resolve to alphabetically-earlier tag — \
         `io` beats `status`",
    );
}

/// `kernel_config_hint` returns Some(_) when ≥ 50% of failures
/// land in `schedstat`/`io`. Pins the gate equality at the
/// boundary.
#[test]
fn parse_summary_kernel_config_hint_gate() {
    // 50/50 split: 5 schedstat + 5 status. Kconfig share = 50%.
    let mut tally = ParseTally::default();
    for _ in 0..5 {
        tally.record_failure("schedstat");
    }
    for _ in 0..5 {
        tally.record_failure("status");
    }
    tally.commit_pending();
    let summary = tally.to_public();
    assert!(
        summary.kernel_config_dominant,
        "50% kconfig share must hit the gate (>= 50% boundary inclusive)",
    );
    assert!(summary.kernel_config_hint().is_some());

    // Below threshold: 1 schedstat, 9 status. Kconfig share 10%.
    let mut tally2 = ParseTally::default();
    tally2.record_failure("schedstat");
    for _ in 0..9 {
        tally2.record_failure("status");
    }
    tally2.commit_pending();
    let summary2 = tally2.to_public();
    assert!(!summary2.kernel_config_dominant);
    assert!(summary2.kernel_config_hint().is_none());

    // Zero failures: kconfig_dominant must be false (no failures
    // to dominate), hint is None.
    let summary3 = ParseTally::default().to_public();
    assert!(!summary3.kernel_config_dominant);
    assert!(summary3.kernel_config_hint().is_none());
}

/// `dominant_read_failure` is None when zero failures landed,
/// even though the tally was constructed.
#[test]
fn parse_summary_dominant_none_when_zero_failures() {
    let summary = ParseTally::default().to_public();
    assert_eq!(summary.read_failures, 0);
    assert!(summary.dominant_read_failure.is_none());
}

/// `capture_with(_, _, false)` skips the production gate so
/// `parse_summary` stays `None` on the assembled snapshot —
/// mirrors the `probe_summary` discipline. Synthetic-tree
/// tests must not see a populated parse summary.
#[test]
fn capture_with_synthetic_tree_yields_no_parse_summary() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 5090;
    let tid: i32 = 5091;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
    assert!(
        snap.parse_summary.is_none(),
        "use_syscall_affinity=false must skip parse_summary; \
         got Some — production-gate discipline is broken",
    );
}

// ------------------------------------------------------------
// T43 — Additional capture-pipeline error-path tests
// ------------------------------------------------------------

/// Phase-1 loadavg missing: capture_with must not panic when
/// the parallelism-clamp `proc_root/loadavg` read fails. The
/// reader's `.ok().and_then(...).unwrap_or(0.0)` chain folds
/// the missing-file branch into the 0.0 default, so the
/// headroom calculation continues to clamp at
/// `[1, num_cpus/2 + 1]`. Pins the missing-loadavg branch.
#[test]
fn capture_with_phase1_loadavg_missing_does_not_panic() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    // No loadavg file. iter_tgids_at returns Vec::new() so the
    // probe-attach loop iterates zero times — but the clamp
    // computation runs unconditionally inside the
    // use_syscall_affinity=true branch, exercising the
    // missing-file path.
    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), true);
    assert!(
        snap.threads.is_empty(),
        "missing loadavg + empty proc_root → empty snapshot, \
         got {} threads",
        snap.threads.len(),
    );
}

/// Phase-1 loadavg malformed: a non-float first token must
/// fold into the 0.0 default via the `.parse::<f64>().ok()`
/// step. Pins that a hostile `proc_root/loadavg` cannot crash
/// the parallelism-clamp computation.
#[test]
fn capture_with_phase1_loadavg_malformed_does_not_panic() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    std::fs::write(proc_tmp.path().join("loadavg"), "not_a_number\n").unwrap();
    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), true);
    assert!(
        snap.threads.is_empty(),
        "malformed loadavg → 0.0 default, empty proc_root → empty \
         snapshot; got {} threads",
        snap.threads.len(),
    );
}

/// Non-UTF-8 bytes in `comm`: `fs::read_to_string` returns Err
/// on invalid UTF-8, so [`read_thread_comm_at`] yields None
/// and the caller defaults to "". With `start_time` non-zero
/// (intact `stat`), the ghost filter does NOT fire and the
/// thread lands with empty comm.
#[test]
fn capture_with_non_utf8_comm_treated_as_absent() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 6161;
    let tid: i32 = 6162;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
    // Overwrite tid/comm with non-UTF-8 bytes (lone 0xFF, then
    // 0xFE — never valid UTF-8 lead bytes).
    let comm_path = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string())
        .join("comm");
    std::fs::write(&comm_path, [0xFF, 0xFE]).unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
    assert_eq!(
        snap.threads.len(),
        1,
        "non-UTF-8 comm folds to empty; ghost filter does NOT \
         fire because start_time is intact; thread still lands. \
         got {} threads",
        snap.threads.len(),
    );
    assert_eq!(
        snap.threads[0].comm, "",
        "non-UTF-8 comm must collapse to empty (read_to_string \
         returns Err on invalid UTF-8)",
    );
    assert_ne!(
        snap.threads[0].start_time_clock_ticks, 0,
        "start_time must be intact for the ghost filter NOT to fire",
    );
}

/// Cgroup path traversal: a `0::/../escape` payload in the
/// per-tid cgroup file lands in `ThreadState.cgroup` verbatim
/// (no sanitization at parse time), and the cgroup_stats
/// enrichment loop calls `read_cgroup_stats_at` with the same
/// string. The current behaviour bounds the read inside the
/// configured `cgroup_root` via `Path::join` — which DOES NOT
/// reject `..` components. Pin that the path-traversal string
/// round-trips through the snapshot but does not surface
/// out-of-tree cgroup data: the stats land at the all-zero
/// default because no matching cgroup directory exists under
/// `cgroup_root`.
#[test]
fn capture_with_cgroup_path_traversal_yields_zero_stats() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 6262;
    let tid: i32 = 6263;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
    // Overwrite cgroup with a traversal string.
    let cgroup_path = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string())
        .join("cgroup");
    std::fs::write(&cgroup_path, "0::/../escape\n").unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
    assert_eq!(snap.threads.len(), 1);
    assert_eq!(
        snap.threads[0].cgroup, "/../escape",
        "traversal string round-trips verbatim through ThreadState.cgroup",
    );
    let stats = snap
        .cgroup_stats
        .get("/../escape")
        .expect("non-empty cgroup string must seed the stats map");
    assert_eq!(
        stats.cpu.usage_usec, 0,
        "no matching cgroup dir under cgroup_root → all-zero stats; \
         a traversal that escaped the cgroup_root would have \
         non-zero values from the parent directory",
    );
}

/// Empty `Cpus_allowed_list:` value: `parse_cpu_list("")`
/// returns None at the empty-input guard, so `cpu_affinity`
/// lands as the empty Vec. Same observable effect as a
/// malformed range (G8) but pins the empty-string branch
/// distinctly.
#[test]
fn capture_with_empty_cpus_allowed_yields_empty_affinity() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 6363;
    let tid: i32 = 6364;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
    let status_path = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string())
        .join("status");
    let status = "Cpus_allowed_list:\t\n\
         voluntary_ctxt_switches:\t1\n\
         nonvoluntary_ctxt_switches:\t1\n";
    std::fs::write(&status_path, status).unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
    assert_eq!(snap.threads.len(), 1);
    let t = &snap.threads[0];
    assert!(
        t.cpu_affinity.0.is_empty(),
        "empty Cpus_allowed_list value → parse_cpu_list returns \
         None at the empty-input guard → cpu_affinity empty; \
         got {} elements",
        t.cpu_affinity.0.len(),
    );
    assert_eq!(
        t.voluntary_csw,
        MonotonicCount(1),
        "empty cpulist must not break csw parsing on the same \
         status file",
    );
}

/// Ghost filter AND-semantics: an empty `comm` paired with a
/// NON-zero `start_time_clock_ticks` does NOT fire the filter.
/// The clause requires BOTH conditions (see
/// `t.comm.is_empty() && t.start_time_clock_ticks == 0`). Pins
/// the AND so a future refactor that flipped to OR would
/// surface here rather than hiding legitimate threads with
/// transient empty comms.
#[test]
fn capture_with_empty_comm_nonzero_start_time_keeps_thread() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 6464;
    let tid: i32 = 6465;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
    // Overwrite comm with whitespace so read_thread_comm_at
    // returns None → comm defaults to "". start_time stays
    // intact at 555_555 (the value stage_synthetic_proc writes).
    let comm_path = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string())
        .join("comm");
    std::fs::write(&comm_path, "   \n").unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
    assert_eq!(
        snap.threads.len(),
        1,
        "empty comm + nonzero start_time MUST NOT fire ghost filter \
         (AND-semantics requires both empty); got {} threads",
        snap.threads.len(),
    );
    let t = &snap.threads[0];
    assert_eq!(t.comm, "", "empty-comm thread surfaces with empty comm");
    assert_ne!(
        t.start_time_clock_ticks, 0,
        "start_time must be non-zero so the AND-clause has a `false` half",
    );
}

// ------------------------------------------------------------
// T45 — Additional parse_summary + capture-pipeline coverage
// ------------------------------------------------------------

/// W2: every tid is ghost-filtered. With N empty task dirs the
/// ghost filter rejects every tid, so each tid's pending failure
/// bumps unwind via `discard_pending`. `tids_walked` is bumped
/// at the call site BEFORE the discard, so it still reads N.
/// `read_failures` lands at zero (every bump unwound), the per-
/// file map is empty, and `dominant_read_failure` is None. Pins
/// the "tids_walked counts attempts; failure tallies count only
/// committed bumps" split end-to-end through `capture_with`.
#[test]
fn parse_summary_all_ghosts_yields_nonzero_tids_walked_zero_failures() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 7070;
    let n: u64 = 4;
    // Stage one tgid with N empty task dirs (no comm, no stat,
    // no other files). Every read fails; ghost filter fires for
    // every tid; every pending tally is unwound.
    let tgid_dir = proc_tmp.path().join(tgid.to_string());
    for k in 0..n {
        let tid = (tgid as u64 + 1 + k) as i32;
        std::fs::create_dir_all(tgid_dir.join("task").join(tid.to_string())).unwrap();
    }
    // Stage `loadavg` so the parallelism-clamp read in phase 1
    // resolves cleanly (the missing-file fallback is exercised
    // by capture_with_phase1_loadavg_missing_does_not_panic).
    std::fs::write(proc_tmp.path().join("loadavg"), "0.10 0.05 0.01 1/1 1\n").unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), true);
    assert!(
        snap.threads.is_empty(),
        "every tid is ghost-filtered → threads must be empty, got {}",
        snap.threads.len(),
    );
    let summary = snap
        .parse_summary
        .expect("use_syscall_affinity=true must populate parse_summary");
    assert_eq!(
        summary.tids_walked, n,
        "tids_walked counts every walk attempt, not committed reads — \
         got {}, want {n}",
        summary.tids_walked,
    );
    assert_eq!(
        summary.read_failures, 0,
        "ghost-filtered tids' failures unwind via discard_pending — \
         got {} failures, want 0",
        summary.read_failures,
    );
    assert!(
        summary.read_failures_by_file.is_empty(),
        "no failure bucket survives the ghost-filter unwind, got {:?}",
        summary.read_failures_by_file,
    );
    assert!(
        summary.dominant_read_failure.is_none(),
        "zero failures → dominant_read_failure is None, got {:?}",
        summary.dominant_read_failure,
    );
    assert!(
        !summary.kernel_config_dominant,
        "zero failures → kernel_config_dominant is false, got true",
    );
}

/// W3: pin which file-kind tokens count as kernel-config-gated.
/// `kernel_config_dominates` filters on `matches!(t, "schedstat"
/// | "io")`. Iterate every recognised kebab token solo (one
/// failure of that kind, no others) and assert the gate flips
/// the way the implementation says it should — schedstat/io
/// land 100% kconfig and the gate fires; stat/status/sched/cgroup
/// land 0% kconfig and the gate stays false. A future refactor
/// that added or removed a token from the kconfig set without
/// updating the docs would surface here.
#[test]
fn parse_summary_kernel_config_token_list_pinned() {
    let kconfig_tokens: &[&'static str] = &["schedstat", "io"];
    for tag in kconfig_tokens {
        let mut tally = ParseTally::default();
        tally.record_failure(tag);
        tally.commit_pending();
        let summary = tally.to_public();
        assert!(
            summary.kernel_config_dominant,
            "solo `{tag}` failure must flip kernel_config_dominant true \
             (kconfig share = 100%); got false — token dropped from the \
             kconfig set",
        );
    }

    let non_kconfig_tokens: &[&'static str] = &["stat", "status", "sched", "cgroup"];
    for tag in non_kconfig_tokens {
        let mut tally = ParseTally::default();
        tally.record_failure(tag);
        tally.commit_pending();
        let summary = tally.to_public();
        assert!(
            !summary.kernel_config_dominant,
            "solo `{tag}` failure must keep kernel_config_dominant false \
             (kconfig share = 0%); got true — token incorrectly added to \
             the kconfig set",
        );
    }
}

/// W5: tally aggregates across multiple tids. Stage 2 tids
/// where each fails a different file (one missing io, one
/// missing schedstat). Both bumps must commit (neither tid is
/// ghost-filtered) and the per-file map carries one entry per
/// failure kind with count 1, total `read_failures` = 2.
#[test]
fn parse_summary_aggregates_across_multiple_tids() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 7080;
    let tid_a: i32 = 7081;
    let tid_b: i32 = 7082;
    stage_minimal_proc_for_parse(proc_tmp.path(), tgid, tid_a);
    // Second tid under the same tgid: write a fresh task dir.
    let tgid_dir = proc_tmp.path().join(tgid.to_string());
    let task_b = tgid_dir.join("task").join(tid_b.to_string());
    std::fs::create_dir_all(&task_b).unwrap();
    std::fs::write(task_b.join("comm"), "live\n").unwrap();
    let stat_line = format!(
        "{tid_b} (live) R 1 2 3 4 5 6 7 0 8 0 10 11 12 13 14 0 1 0 \
         555555 100 200 300 400 500 600 700 800 900 1000 1100 \
         1200 1300 1400 1500 1600 1700 1800 0\n"
    );
    std::fs::write(task_b.join("stat"), stat_line).unwrap();
    std::fs::write(task_b.join("schedstat"), "0 0 0\n").unwrap();
    std::fs::write(
        task_b.join("status"),
        "voluntary_ctxt_switches:\t0\n\
         nonvoluntary_ctxt_switches:\t0\n",
    )
    .unwrap();
    std::fs::write(task_b.join("io"), "rchar: 0\n").unwrap();
    std::fs::write(task_b.join("sched"), "").unwrap();
    std::fs::write(task_b.join("cgroup"), "0::/\n").unwrap();

    // tid_a: delete io. tid_b: delete schedstat.
    std::fs::remove_file(tgid_dir.join("task").join(tid_a.to_string()).join("io")).unwrap();
    std::fs::remove_file(task_b.join("schedstat")).unwrap();

    let mut tally = ParseTally::default();
    let mut tally_opt: Option<&mut ParseTally> = Some(&mut tally);
    for tid in [tid_a, tid_b] {
        tally_opt.as_mut().unwrap().tids_walked += 1;
        let _ = capture_thread_at_with_tally(
            proc_tmp.path(),
            tgid,
            tid,
            "p",
            "live",
            false,
            &mut tally_opt,
        );
        tally_opt.as_mut().unwrap().commit_pending();
    }
    let summary = tally.to_public();
    assert_eq!(summary.tids_walked, 2);
    assert_eq!(
        summary.read_failures, 2,
        "two tids, one failure each → 2 total; got {}",
        summary.read_failures,
    );
    assert_eq!(
        summary.read_failures_by_file.get("io"),
        Some(&1),
        "tid_a missing io → io bucket = 1; got {:?}",
        summary.read_failures_by_file.get("io"),
    );
    assert_eq!(
        summary.read_failures_by_file.get("schedstat"),
        Some(&1),
        "tid_b missing schedstat → schedstat bucket = 1; got {:?}",
        summary.read_failures_by_file.get("schedstat"),
    );
}

/// W7: deleting cgroup lands a `"cgroup"` failure. Mirrors the
/// schedstat/io single-failure tests so the cgroup-read tally
/// path is exercised explicitly — `read_cgroup_at_with_tally`
/// is the only producer of the `"cgroup"` tag and a future
/// refactor that bypassed the tally would surface here.
#[test]
fn parse_summary_records_cgroup_failure() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 7090;
    let tid: i32 = 7091;
    stage_minimal_proc_for_parse(proc_tmp.path(), tgid, tid);
    std::fs::remove_file(
        proc_tmp
            .path()
            .join(tgid.to_string())
            .join("task")
            .join(tid.to_string())
            .join("cgroup"),
    )
    .unwrap();

    let mut tally = ParseTally::default();
    let mut tally_opt: Option<&mut ParseTally> = Some(&mut tally);
    tally_opt.as_mut().unwrap().tids_walked += 1;
    let _ = capture_thread_at_with_tally(
        proc_tmp.path(),
        tgid,
        tid,
        "p",
        "live",
        false,
        &mut tally_opt,
    );
    tally_opt.as_mut().unwrap().commit_pending();

    let summary = tally.to_public();
    assert_eq!(
        summary.read_failures_by_file.get("cgroup"),
        Some(&1),
        "missing cgroup file → cgroup bucket = 1; got {:?}",
        summary.read_failures_by_file.get("cgroup"),
    );
}

/// W6: the production gate (`use_syscall_affinity=true`)
/// populates `parse_summary` end-to-end. Mirror of
/// `capture_with_synthetic_tree_yields_no_parse_summary` but
/// with the gate flipped — pins that the production-path
/// assignment is wired through.
#[test]
fn capture_with_production_gate_populates_parse_summary() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 7100;
    let tid: i32 = 7101;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
    // loadavg lets the parallelism-clamp read resolve cleanly.
    std::fs::write(proc_tmp.path().join("loadavg"), "0.10 0.05 0.01 1/1 1\n").unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), true);
    assert!(
        snap.parse_summary.is_some(),
        "use_syscall_affinity=true must populate parse_summary on \
         the assembled snapshot — production-gate wiring is broken",
    );
}

/// X2: non-UTF-8 bytes in `<tgid>/comm` (the pcomm path).
/// `read_process_comm_at` calls `fs::read_to_string`, which
/// returns Err on invalid UTF-8; `.ok()?` propagates None and
/// the caller defaults `pcomm` to "" via `.unwrap_or_default()`.
/// Pin that capture does not panic and the per-thread `pcomm`
/// surfaces empty. Mirror of
/// `capture_with_non_utf8_comm_treated_as_absent` but for the
/// process-level (`<tgid>/comm`) read rather than the per-tid
/// (`<tgid>/task/<tid>/comm`) read.
#[test]
fn capture_with_non_utf8_pcomm_treated_as_absent() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 7110;
    let tid: i32 = 7111;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
    // Overwrite the pcomm path (`<tgid>/comm`) with non-UTF-8
    // lead bytes (0xFF and 0xFE — never valid UTF-8 starts).
    let pcomm_path = proc_tmp.path().join(tgid.to_string()).join("comm");
    std::fs::write(&pcomm_path, [0xFF, 0xFE]).unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
    assert_eq!(
        snap.threads.len(),
        1,
        "non-UTF-8 pcomm must not break the capture — the thread still \
         lands; got {} threads",
        snap.threads.len(),
    );
    assert_eq!(
        snap.threads[0].pcomm, "",
        "non-UTF-8 pcomm collapses to empty (read_to_string returns Err \
         on invalid UTF-8 and unwrap_or_default → \"\")",
    );
}

/// Y1: panic-injection harness for rayon worker panics.
///
/// `attach_jemalloc_at` reads `/proc/<pid>/exe`, opens the ELF
/// file, and walks DWARF — every step can panic under fd
/// exhaustion or OOM. Without the `catch_unwind` guard in
/// `capture_with`'s phase-1 worker closure, a single panicking
/// tgid would propagate through `pool.install` and tear down
/// the whole snapshot. No realistic synthetic input can force
/// the underlying readers to panic, so this test installs an
/// explicit injection seam (`PANIC_INJECT_TGID`) that fires
/// inside `attach_probe_for_tgid_at` for the matching tgid and
/// drives the rayon worker into a panic. The capture pipeline
/// must absorb it, surface it as a `worker-panic` attach tag,
/// and still walk the surviving tgid's threads.
///
/// Asserts:
///   - `capture_with(.., true)` returns rather than unwinding,
///   - the surviving tgid's thread lands in the snapshot,
///   - `probe_summary.failed >= 1` (the panic is counted),
///   - `dominant_failure == Some("worker-panic")` (the new tag
///     surfaces in the curated public surface).
#[test]
fn capture_with_rayon_worker_panic_is_caught_and_surfaced() {
    // Serialize panic-hook test against any future test that
    // also installs a custom hook, so the silenced hook below
    // is not clobbered. `Mutex<()>` is enough — the lock is
    // only held for the duration of the capture call.
    static PANIC_INJECT_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = PANIC_INJECT_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    // Required by the parallelism-clamp logic in capture_with.
    std::fs::write(proc_tmp.path().join("loadavg"), "0.0 0.0 0.0 1/1 1\n").unwrap();

    // Two tgids: the survivor (clean attach attempt → fails
    // benignly with `readlink-failure` because the synthetic
    // /proc has no `<tgid>/exe` symlink — the dominant-tag
    // filter suppresses this, leaving worker-panic as the
    // sole dominant candidate) and the panic target (the
    // sentinel tgid the seam matches against). Sentinel value
    // 99001 is intentionally outside any other test's range so
    // a parallel run cannot cross-fire.
    let survivor_tgid: i32 = 99000;
    let survivor_tid: i32 = 99002;
    let panic_tgid: i32 = 99001;
    let panic_tid: i32 = 99003;
    stage_synthetic_proc(
        proc_tmp.path(),
        survivor_tgid,
        survivor_tid,
        "ok-pcomm",
        "ok-comm",
    );
    stage_synthetic_proc(
        proc_tmp.path(),
        panic_tgid,
        panic_tid,
        "panic-pcomm",
        "panic-comm",
    );

    // Silence the default panic hook: rayon's worker panic
    // would otherwise dump a stack trace to stderr and pollute
    // the test output. Restore the hook before the lock
    // releases so subsequent tests see the real hook again.
    let saved_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_info| {}));

    // Arm the seam, run capture, then disarm BEFORE restoring
    // the hook so a panic during disarm (none expected) still
    // hits the silenced hook rather than the real one.
    PANIC_INJECT_TGID.store(panic_tgid, std::sync::atomic::Ordering::Release);
    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), true);
    PANIC_INJECT_TGID.store(0, std::sync::atomic::Ordering::Release);

    std::panic::set_hook(saved_hook);

    // Survivor thread must land. The panicking tgid's threads
    // are walked too (phase 2 still iterates every tgid in
    // `tgids`), so total threads is 2.
    assert_eq!(
        snap.threads.len(),
        2,
        "rayon worker panic must not block phase 2 — both staged tgids \
         walk their threads; got {} threads",
        snap.threads.len(),
    );

    let summary = snap
        .probe_summary
        .expect("use_syscall_affinity=true must populate probe_summary");
    assert!(
        summary.failed >= 1,
        "worker-panic must count as a failure; got failed={}",
        summary.failed,
    );
    assert_eq!(
        summary.dominant_failure.as_deref(),
        Some("worker-panic"),
        "worker-panic is the only ACTIONABLE failure tag in this \
         scenario. The survivor's synthetic /proc has no `exe` \
         symlink, so attach short-circuits with `readlink-failure` \
         — the dominant-tag comparator filters that benign tag out \
         (same `matches!` arm `record_attach_outcome` uses to log it \
         at debug rather than warn), leaving worker-panic as the \
         sole candidate. A regression that demoted worker-panic \
         out of the dominant set, or that miscounted the panic, \
         would fail here. Got {:?}",
        summary.dominant_failure,
    );
}

/// Non-string panic payload: a rayon worker that panics with
/// a typed payload (not `&'static str` or `String`) must
/// still be absorbed by the `catch_unwind` guard, surface as
/// a `worker-panic` attach tag, and let the surviving tgid's
/// threads land in the snapshot. Pins the
/// `unwrap_or("<non-string panic payload>")` fallback arm in
/// `capture_with`'s panic-handling block — without it, an
/// unwrap on the `downcast_ref` chain would re-panic and tear
/// down the worker.
///
/// The injection seam (`PANIC_INJECT_NON_STRING`) routes the
/// panic through `std::panic::panic_any(0xDEADBEEFu64)` so
/// the payload is a typed `u64` whose `Box<dyn Any + Send>`
/// downcasts to neither `&str` nor `String` — exactly the
/// case the fallback arm guards against.
///
/// Asserts that the snapshot lands cleanly, the worker-panic
/// tag bumps once, and the survivor tgid's threads still walk.
/// Companion to
/// `capture_with_rayon_worker_panic_is_caught_and_surfaced`,
/// which exercises the formatted-message (String) panic path.
#[test]
fn capture_with_rayon_worker_panic_non_string_payload_falls_back() {
    // Serialize against any other panic-hook test in the crate.
    static PANIC_INJECT_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = PANIC_INJECT_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    std::fs::write(proc_tmp.path().join("loadavg"), "0.0 0.0 0.0 1/1 1\n").unwrap();

    let survivor_tgid: i32 = 99100;
    let survivor_tid: i32 = 99102;
    let panic_tgid: i32 = 99101;
    let panic_tid: i32 = 99103;
    stage_synthetic_proc(
        proc_tmp.path(),
        survivor_tgid,
        survivor_tid,
        "ok-pcomm",
        "ok-comm",
    );
    stage_synthetic_proc(
        proc_tmp.path(),
        panic_tgid,
        panic_tid,
        "panic-pcomm",
        "panic-comm",
    );

    let saved_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_info| {}));

    PANIC_INJECT_TGID.store(panic_tgid, std::sync::atomic::Ordering::Release);
    PANIC_INJECT_NON_STRING.store(true, std::sync::atomic::Ordering::Release);
    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), true);
    PANIC_INJECT_NON_STRING.store(false, std::sync::atomic::Ordering::Release);
    PANIC_INJECT_TGID.store(0, std::sync::atomic::Ordering::Release);

    std::panic::set_hook(saved_hook);

    // Both tgids' threads walked through phase 2 — the panic
    // in phase 1 must not block phase 2 even when the payload
    // is non-string.
    assert_eq!(
        snap.threads.len(),
        2,
        "non-string-payload panic must not block phase 2; got {} threads",
        snap.threads.len(),
    );

    let summary = snap
        .probe_summary
        .expect("use_syscall_affinity=true populates probe_summary");
    assert!(
        summary.failed >= 1,
        "non-string-payload worker-panic must count as a failure; got failed={}",
        summary.failed,
    );
    assert_eq!(
        summary.dominant_failure.as_deref(),
        Some("worker-panic"),
        "non-string-payload panic must still surface as a \
         `worker-panic` tag — the fallback arm produced the \
         placeholder string but the bookkeeping path is \
         unchanged. Got {:?}",
        summary.dominant_failure,
    );
}
