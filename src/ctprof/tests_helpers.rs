//! Shared `/proc` synthetic-tree builder for the capture and
//! parse-summary test groups. `stage_synthetic_proc` writes every
//! procfs file that capture walks so the helpers in
//! `tests_capture.rs` and `tests_parse_summary.rs` can drive
//! capture without touching the real procfs. Lifted to a sibling
//! module so both consumers can call it without duplication.

#![cfg(test)]

use std::path::Path;


// ------------------------------------------------------------
// Synthetic-tree tests (H1-H5)
//
// Stage a tempdir shaped like `/proc/<tgid>/{comm,
// task/<tid>/{stat,schedstat,status,io,sched,comm,cgroup}}`
// so every capture helper can be driven without touching the
// real procfs. Mirrors the compare-side pattern in
// tests/ctprof_compare.rs but against the capture side.
// ------------------------------------------------------------

/// Build a synthetic `/proc` under `root` carrying exactly one
/// thread. Writes every file capture walks so every counter
/// on `ThreadState` round-trips with a known value. `cpus` is
/// the `Cpus_allowed_list` value (a range string the
/// `parse_cpu_list` helper decodes).
pub(super) fn stage_synthetic_proc(root: &Path, tgid: i32, tid: i32, pcomm: &str, comm: &str) {
    use std::fs;
    let tgid_dir = root.join(tgid.to_string());
    let task_dir = tgid_dir.join("task").join(tid.to_string());
    fs::create_dir_all(&task_dir).unwrap();

    // /proc/<tgid>/comm
    fs::write(tgid_dir.join("comm"), format!("{pcomm}\n")).unwrap();
    // /proc/<tgid>/task/<tid>/comm
    fs::write(task_dir.join("comm"), format!("{comm}\n")).unwrap();

    // stat: paren-safe comm, fields 1..41. Comm inserted with
    // parens inside so the rfind(')') anchor has to find the
    // LAST close-paren, not the first. Fields past comm start
    // at index 0 in `tail` (tail[0] is `state`, per procfs
    // field-index-minus-three convention that parse_stat uses).
    // Field indices (post-comm):
    //   [0]=state [1]=ppid [2]=pgrp [3]=session [4]=tty
    //   [5]=tpgid [6]=flags [7]=minflt(field 10)
    //   [8]=cminflt [9]=majflt(field 12) [10]=cmajflt
    //   [11..16]=utime/stime/cutime/cstime/priority
    //   [16]=nice (field 19) [17]=num_threads [18]=itrealvalue
    //   [19]=starttime (field 22) [20..37]=vsize/rss/...
    //   [38]=policy (field 41).
    let stat_line = format!(
        "{tid} (proc (with) parens) R 1 2 3 4 5 6 \
         7777 0 8888 0 10 11 12 13 14 {nice} 1 0 \
         {starttime} 100 200 300 400 500 600 700 800 \
         900 1000 1100 1200 1300 1400 1500 1600 1700 1800 {policy}\n",
        tid = tid,
        nice = -10_i32,
        starttime = 555_555u64,
        policy = 0, // SCHED_OTHER
    );
    fs::write(task_dir.join("stat"), stat_line).unwrap();

    // schedstat: run_time_ns wait_time_ns timeslices
    fs::write(task_dir.join("schedstat"), "1000000 200000 50\n").unwrap();

    // status: State + voluntary/nonvoluntary csw + Cpus_allowed_list.
    // parse_status matches the lowercase csw keys verbatim;
    // `State` and `Cpus_allowed_list` use the capitalised
    // leading char of the procfs file.
    let status = "Name:\tfoo\n\
         State:\tR (running)\n\
         voluntary_ctxt_switches:\t42\n\
         nonvoluntary_ctxt_switches:\t7\n\
         Cpus_allowed_list:\t0-3\n";
    fs::write(task_dir.join("status"), status).unwrap();

    // io: cumulative byte counters
    let io = "rchar: 100\n\
         wchar: 200\n\
         syscr: 10\n\
         syscw: 20\n\
         read_bytes: 4096\n\
         write_bytes: 8192\n\
         cancelled_write_bytes: 512\n";
    fs::write(task_dir.join("io"), io).unwrap();

    // sched: every parse_sched-matched key, with the
    // `se.statistics.` prefix for the wakeup family to
    // exercise the rsplit('.') short-key logic. `ext.enabled`
    // is unprefixed (literal kernel key) and tests the
    // full-key gate.
    let sched = "\
         se.statistics.nr_wakeups                       :         11\n\
         se.statistics.nr_wakeups_local                 :          8\n\
         se.statistics.nr_wakeups_remote                :          3\n\
         se.statistics.nr_wakeups_sync                  :          2\n\
         se.statistics.nr_wakeups_migrate               :          1\n\
         se.statistics.nr_wakeups_idle                  :          4\n\
         se.statistics.nr_wakeups_affine                :         12\n\
         se.statistics.nr_wakeups_affine_attempts       :         20\n\
         nr_migrations                                  :          9\n\
         se.statistics.nr_migrations_cold               :          5\n\
         se.statistics.nr_forced_migrations             :          7\n\
         se.statistics.nr_failed_migrations_affine      :          1\n\
         se.statistics.nr_failed_migrations_running     :          2\n\
         se.statistics.nr_failed_migrations_hot         :          3\n\
         wait_sum                                       :    5000.25\n\
         wait_count                                     :         15\n\
         se.statistics.wait_max                         :     250.5\n\
         sum_sleep_runtime                              :    3200.50\n\
         se.statistics.sleep_max                        :     180.25\n\
         sum_block_runtime                              :    1100.75\n\
         se.statistics.block_max                        :      60.75\n\
         iowait_sum                                     :       77.0\n\
         iowait_count                                   :         18\n\
         se.statistics.exec_max                         :      90.0\n\
         se.statistics.slice_max                        :     400.5\n\
         ext.enabled                                    :          1\n";
    fs::write(task_dir.join("sched"), sched).unwrap();

    // cgroup: v2-style single entry (0::path). read_cgroup_at
    // parses the `0::` prefix.
    fs::write(task_dir.join("cgroup"), "0::/ktstr.slice/worker0\n").unwrap();
}
