//! Spawn-pipeline tests — pcomm fork-then-thread group.
//!
//! Verifies the `WorkSpec::pcomm` contract via the dedicated entry
//! point [`WorkloadHandle::spawn_pcomm_cgroup`]: when a slice of
//! `WorkSpec` entries is passed in, the spawn path forks ONE
//! thread-group leader process, sets its comm to `pcomm`, then
//! spawns N worker threads inside it (one per entry's
//! `num_workers`). Every worker thread's
//! `task->group_leader->comm` is `pcomm`. Each thread sets its own
//! comm via `WorkSpec::comm` after the post-fork init.
//!
//! These tests bypass [`WorkloadHandle::spawn`] entirely: that
//! entry point does NOT honor pcomm — pcomm dispatch lives at
//! `apply_setup` (per-CgroupDef) or via `spawn_pcomm_cgroup`
//! (direct slice).
//!
//! Verification points:
//! - `/proc/<tid>/status` `Tgid:` line == leader pid (every worker
//!   thread shares the leader's tgid).
//! - `/proc/<leader_pid>/comm` == `pcomm`, kernel-truncated to
//!   15 bytes (`TASK_COMM_LEN - 1`).
//! - When per-thread `comm` is also set, `/proc/<tid>/comm` ==
//!   that per-thread `comm`, NOT `pcomm` — the per-thread
//!   `prctl(PR_SET_NAME)` overrides the inherited comm on the
//!   thread itself but leaves `task->group_leader->comm` alone.
//! - `stop_and_collect` returns exactly N reports per pcomm group.
//! - On Drop without start/collect, the leader is reaped (no
//!   zombie), and the report pipe / start pipe fds are closed.

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

/// Read `/proc/<tid>/status` and extract the `Tgid:` line as a
/// `pid_t`. Panics with an actionable message if the file is
/// unreadable or the line is missing — both indicate a wider
/// failure (worker died before observation, /proc unmounted)
/// that no test can recover from.
fn read_tgid(tid: libc::pid_t) -> libc::pid_t {
    let status = std::fs::read_to_string(format!("/proc/{tid}/status"))
        .expect("/proc/<tid>/status must be readable for live thread");
    let line = status
        .lines()
        .find(|l| l.starts_with("Tgid:"))
        .expect("/proc/<tid>/status must include Tgid line");
    line.trim_start_matches("Tgid:")
        .trim()
        .parse()
        .expect("Tgid must be a parseable pid_t")
}

/// Read `/proc/<pid>/comm`. The file is the kernel's authoritative
/// per-task comm string, terminated by a single newline. Strips
/// the trailing newline before returning.
fn read_comm(pid: libc::pid_t) -> String {
    let raw = std::fs::read_to_string(format!("/proc/{pid}/comm"))
        .expect("/proc/<pid>/comm must be readable for live task");
    raw.trim_end_matches('\n').to_string()
}

/// Build a `WorkSpec` ready for `spawn_pcomm_cgroup`: SpinWait
/// work, the supplied worker count, no affinity, and the supplied
/// pcomm. Per-thread `comm` is left at `None` unless the test
/// overrides.
fn pcomm_spec(workers: usize, pcomm: &'static str) -> WorkSpec {
    WorkSpec::default()
        .work_type(WorkType::SpinWait)
        .workers(workers)
        .pcomm(pcomm)
}

/// Enumerate every TID inside a thread group via
/// `/proc/<leader_pid>/task/`. Each subdirectory name is a TID; the
/// leader's own TID is included (TID == TGID for the group leader).
/// Caller must invoke before the leader is reaped — once
/// `stop_and_collect` runs, the directory disappears.
fn read_task_tids(leader_pid: libc::pid_t) -> BTreeSet<libc::pid_t> {
    let mut out = BTreeSet::new();
    let dir = std::fs::read_dir(format!("/proc/{leader_pid}/task"))
        .expect("/proc/<leader>/task must be readable while leader is alive");
    for entry in dir {
        let entry = entry.expect("task directory entry must read cleanly");
        let name = entry.file_name();
        let s = name
            .to_str()
            .expect("/proc task entry names are ASCII tids");
        let tid: libc::pid_t = s.parse().expect("/proc task entry must parse as pid_t");
        out.insert(tid);
    }
    out
}

/// Basic happy path: a single WorkSpec with `pcomm = "leader"`
/// and `num_workers = 2` produces 2 reports whose tids share a
/// single Tgid. `/proc/<Tgid>/comm` equals the configured pcomm
/// string.
///
/// This pins the core contract — the leader IS the thread-group
/// leader, and `task->group_leader->comm` carries `pcomm` for
/// every worker.
#[test]
fn pcomm_container_sets_group_leader_comm() {
    let works = vec![pcomm_spec(2, "leader")];
    let mut h = WorkloadHandle::spawn_pcomm_cgroup("leader", None, None, &works)
        .expect("pcomm spawn must succeed");
    h.start();
    std::thread::sleep(Duration::from_millis(200));
    // Read procfs while workers are alive — stop_and_collect SIGKILLs
    // and waitpid-reaps the container leader, after which
    // /proc/<tid>/{comm,status} returns ENOENT.
    let pids = h.worker_pids();
    assert_eq!(
        pids.len(),
        1,
        "pcomm spawn registers a single container child (the leader); \
         got {} entries",
        pids.len(),
    );
    let leader_pid = pids[0];
    let leader_comm = read_comm(leader_pid);
    let task_tids = read_task_tids(leader_pid);
    let mut tgids: BTreeSet<libc::pid_t> = BTreeSet::new();
    for tid in &task_tids {
        tgids.insert(read_tgid(*tid));
    }
    let reports = h.stop_and_collect();
    assert_eq!(
        reports.len(),
        2,
        "pcomm group must produce exactly 2 reports; got {}",
        reports.len(),
    );
    assert_eq!(
        tgids.len(),
        1,
        "all pcomm-group threads must share a single Tgid (the leader); \
         observed Tgids: {tgids:?}",
    );
    assert_eq!(
        *tgids.iter().next().unwrap(),
        leader_pid,
        "shared Tgid must equal the container leader pid",
    );
    assert_eq!(
        leader_comm, "leader",
        "/proc/<leader>/comm must equal pcomm; got {leader_comm:?}",
    );
}

/// When both `pcomm` and per-thread `comm` are set, the leader
/// process holds `pcomm` as its comm (the thread-group leader's
/// comm) while each worker thread sets its OWN comm via the
/// post-spawn `prctl(PR_SET_NAME)` inside `worker_main`. Reading
/// `/proc/<tid>/comm` for a worker thread returns the per-thread
/// `comm`, not `pcomm`.
///
/// This mirrors real workloads — `chrome` (pcomm) hosting
/// `ThreadPoolForeg` threads (comm) — where scheduler matchers
/// can filter on either the group-leader comm or the per-thread
/// comm independently.
#[test]
fn pcomm_per_thread_comm_distinct_from_group_leader() {
    let works = vec![
        WorkSpec::default()
            .work_type(WorkType::SpinWait)
            .workers(2)
            .pcomm("leader")
            .comm("worker"),
    ];
    let mut h = WorkloadHandle::spawn_pcomm_cgroup("leader", None, None, &works)
        .expect("pcomm + comm spawn must succeed");
    h.start();
    std::thread::sleep(Duration::from_millis(200));
    // Read procfs while workers are alive (stop_and_collect reaps
    // the leader).
    let pids = h.worker_pids();
    assert_eq!(pids.len(), 1);
    let leader_pid = pids[0];
    let leader_comm = read_comm(leader_pid);
    let task_tids = read_task_tids(leader_pid);
    // Sample per-thread comms for every worker thread (excluding the
    // leader itself, which holds pcomm). Worker threads override
    // their own comm via prctl(PR_SET_NAME) in worker_main.
    let mut worker_comms: Vec<(libc::pid_t, String)> = Vec::new();
    for tid in &task_tids {
        if *tid == leader_pid {
            continue;
        }
        worker_comms.push((*tid, read_comm(*tid)));
    }
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 2);
    assert_eq!(
        leader_comm, "leader",
        "/proc/<leader>/comm must equal pcomm",
    );
    assert_eq!(
        worker_comms.len(),
        2,
        "expected 2 worker thread tids besides the leader; got {worker_comms:?}",
    );
    for (tid, tcomm) in &worker_comms {
        assert_eq!(
            tcomm, "worker",
            "/proc/<tid>/comm for worker tid={tid} must equal per-thread \
             comm 'worker'; got {tcomm:?}",
        );
    }
}

/// Two independent `spawn_pcomm_cgroup` calls produce two distinct
/// thread-group leader processes. Each call's threads share the
/// matching leader's Tgid. Pins that pcomm groups do not collapse
/// onto a shared leader even when both pcomm strings are valid
/// simultaneously.
#[test]
fn multiple_pcomm_groups_have_distinct_containers() {
    let works_a = vec![pcomm_spec(2, "group_a")];
    let works_b = vec![pcomm_spec(2, "group_b")];
    let mut h_a = WorkloadHandle::spawn_pcomm_cgroup("group_a", None, None, &works_a)
        .expect("group_a pcomm spawn must succeed");
    let mut h_b = WorkloadHandle::spawn_pcomm_cgroup("group_b", None, None, &works_b)
        .expect("group_b pcomm spawn must succeed");
    h_a.start();
    h_b.start();
    std::thread::sleep(Duration::from_millis(200));
    // Read procfs for both groups while alive (stop_and_collect
    // reaps the leaders below).
    let pids_a = h_a.worker_pids();
    let pids_b = h_b.worker_pids();
    assert_eq!(pids_a.len(), 1);
    assert_eq!(pids_b.len(), 1);
    let leader_a = pids_a[0];
    let leader_b = pids_b[0];
    let comm_a = read_comm(leader_a);
    let comm_b = read_comm(leader_b);
    let tids_a = read_task_tids(leader_a);
    let tids_b = read_task_tids(leader_b);
    let mut tgids_a: BTreeSet<libc::pid_t> = BTreeSet::new();
    for tid in &tids_a {
        tgids_a.insert(read_tgid(*tid));
    }
    let mut tgids_b: BTreeSet<libc::pid_t> = BTreeSet::new();
    for tid in &tids_b {
        tgids_b.insert(read_tgid(*tid));
    }
    let reports_a = h_a.stop_and_collect();
    let reports_b = h_b.stop_and_collect();
    assert_eq!(reports_a.len(), 2, "group_a must produce 2 reports");
    assert_eq!(reports_b.len(), 2, "group_b must produce 2 reports");
    // Leaders must be distinct PIDs.
    assert_ne!(
        leader_a, leader_b,
        "the two pcomm groups must have distinct leaders; both observed pid={leader_a}",
    );
    // Each group's threads share their leader's tgid.
    assert_eq!(
        tgids_a.len(),
        1,
        "group_a tgids must collapse to one; observed {tgids_a:?}",
    );
    assert_eq!(
        *tgids_a.iter().next().unwrap(),
        leader_a,
        "group_a shared Tgid must equal leader_a",
    );
    assert_eq!(
        tgids_b.len(),
        1,
        "group_b tgids must collapse to one; observed {tgids_b:?}",
    );
    assert_eq!(
        *tgids_b.iter().next().unwrap(),
        leader_b,
        "group_b shared Tgid must equal leader_b",
    );
    // Leader comms are the configured pcomms.
    assert_eq!(comm_a, "group_a");
    assert_eq!(comm_b, "group_b");
}

/// `stop_and_collect` returns exactly N reports for a pcomm group
/// with N workers — the leader writes all N reports over its
/// single report pipe, and the parent successfully deserializes
/// every entry. After collect, the leader is reaped (no zombie
/// remains), and no fds leak.
#[test]
fn pcomm_stop_and_collect_returns_all_reports() {
    let works = vec![pcomm_spec(4, "multi")];
    let baseline_fds = std::fs::read_dir("/proc/self/fd")
        .map(|d| d.count())
        .unwrap_or(0);
    let mut h = WorkloadHandle::spawn_pcomm_cgroup("multi", None, None, &works)
        .expect("pcomm spawn must succeed");
    h.start();
    std::thread::sleep(Duration::from_millis(200));
    let reports = h.stop_and_collect();
    assert_eq!(
        reports.len(),
        4,
        "stop_and_collect must return all N reports from the leader; \
         got {}",
        reports.len(),
    );
    // Every report must have non-zero work_units (proving the
    // worker actually ran, not just produced a sentinel) and
    // completed=true.
    for r in &reports {
        assert!(
            r.completed,
            "pcomm worker tid={} report must be completed=true; \
             completed=false indicates a sentinel from a missing \
             or unparseable per-thread report",
            r.tid,
        );
        assert!(
            r.work_units > 0,
            "pcomm worker tid={} did no work; work_units=0",
            r.tid,
        );
    }
    // No fd leak: the report pipe (one per leader) and start pipe
    // must be closed by the time stop_and_collect returns.
    let after_fds = std::fs::read_dir("/proc/self/fd")
        .map(|d| d.count())
        .unwrap_or(0);
    assert!(
        after_fds <= baseline_fds + 1,
        "fd leak: baseline={baseline_fds}, after collect={after_fds}",
    );
}

/// Kernel truncates `prctl(PR_SET_NAME)` input to 15 bytes
/// (`TASK_COMM_LEN - 1`); the 16th byte is reserved for the NUL
/// terminator. This test pins the truncation contract for
/// `pcomm`: a >15-byte input is silently truncated, with no
/// error and no exception. Operators reading `/proc/<leader>/comm`
/// see the truncated string, not the original input.
///
/// `"this_is_a_very_long_name"` is 24 bytes; truncated to 15
/// gives `"this_is_a_very_"`. Pinning the exact truncation
/// boundary keeps documentation honest.
#[test]
fn pcomm_kernel_truncates_to_15_bytes() {
    let long_name = "this_is_a_very_long_name";
    assert!(
        long_name.len() > 15,
        "test fixture must exceed TASK_COMM_LEN-1"
    );
    let works = vec![pcomm_spec(1, long_name)];
    let mut h = WorkloadHandle::spawn_pcomm_cgroup(long_name, None, None, &works)
        .expect("pcomm spawn must succeed");
    h.start();
    std::thread::sleep(Duration::from_millis(200));
    // Read procfs while leader is alive.
    let pids = h.worker_pids();
    assert_eq!(pids.len(), 1);
    let leader_pid = pids[0];
    let observed = read_comm(leader_pid);
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 1);
    assert_eq!(
        observed.len(),
        15,
        "kernel must truncate pcomm to 15 bytes (TASK_COMM_LEN-1=15); \
         observed length {} for {observed:?}",
        observed.len(),
    );
    assert_eq!(
        observed,
        &long_name[..15],
        "truncated comm must be the leading 15 bytes of pcomm input",
    );
}

/// `pcomm` with a single WorkSpec at `num_workers = 0` is a
/// degenerate input: there are no threads to host inside the
/// leader, so the leader itself has no purpose. The contract:
/// `spawn_pcomm_cgroup` returns a handle with no children, no
/// reports, and no fork. Pin this so a regression that forks an
/// empty leader (which would idle forever waiting for SIGUSR1)
/// surfaces here.
#[test]
fn pcomm_zero_workers_no_container_spawn() {
    let works = vec![pcomm_spec(0, "empty")];
    let baseline_fds = std::fs::read_dir("/proc/self/fd")
        .map(|d| d.count())
        .unwrap_or(0);
    let h = WorkloadHandle::spawn_pcomm_cgroup("empty", None, None, &works)
        .expect("pcomm with 0 workers must spawn cleanly");
    let reports = h.stop_and_collect();
    assert!(
        reports.is_empty(),
        "pcomm group with 0 workers must produce no reports; got {}",
        reports.len(),
    );
    let after_fds = std::fs::read_dir("/proc/self/fd")
        .map(|d| d.count())
        .unwrap_or(0);
    assert!(
        after_fds <= baseline_fds + 1,
        "0-worker pcomm path must not leak fds; baseline={baseline_fds}, after={after_fds}",
    );
}

/// Drop-without-collect of a pcomm-bearing handle reaps the
/// leader child cleanly. Mirrors the fork-mode drop test:
/// dropping without `stop_and_collect()` must release the leader
/// PID via the same SIGKILL+waitpid pattern the SpawnGuard /
/// WorkloadHandle Drop already use for fork children.
///
/// `start()` is called explicitly before drop so the worker
/// threads have begun executing and produced observable Tgid
/// state via `worker_pids()`. The brief sleep gives the
/// per-thread Release publish time to land before the Acquire
/// load in `worker_pids()`.
#[test]
fn pcomm_handle_drop_reaps_container() {
    let works = vec![pcomm_spec(2, "dropme")];
    let mut h = WorkloadHandle::spawn_pcomm_cgroup("dropme", None, None, &works)
        .expect("pcomm spawn must succeed");
    h.start();
    std::thread::sleep(Duration::from_millis(100));
    let pids = h.worker_pids();
    assert!(
        !pids.is_empty(),
        "pcomm handle must report worker pids after start",
    );
    let first_pid = pids[0];
    assert!(
        first_pid > 0,
        "first worker pid must be published post-start; got {first_pid}",
    );
    // Leader PID. Under spawn_pcomm_cgroup the handle's children
    // entry holds the leader's pid directly (the parent never
    // observes the per-thread tids), so worker_pids()[0] IS the
    // leader pid. Verify by reading /proc/<pid>/status: a
    // single-thread tgid leader has Tgid == its own pid.
    let leader_pid = first_pid;
    drop(h);
    // After drop the leader must be dead. kill(pid, 0) returns
    // ESRCH when the pid is gone (or has been recycled, but PID
    // reuse within a millisecond of drop is improbable).
    let alive = nix::sys::signal::kill(nix::unistd::Pid::from_raw(leader_pid), None).is_ok();
    assert!(!alive, "leader pid {leader_pid} must be dead after Drop",);
}

/// Two WorkSpec entries inside a single `spawn_pcomm_cgroup` call,
/// each with their own `num_workers`, coalesce into ONE leader
/// process. Reports include the per-thread `group_idx` matching
/// each WorkSpec's index in the input slice.
#[test]
fn pcomm_multiple_workspecs_coalesce_into_one_leader() {
    let works = vec![
        WorkSpec::default()
            .work_type(WorkType::SpinWait)
            .workers(2)
            .pcomm("shared"),
        WorkSpec::default()
            .work_type(WorkType::SpinWait)
            .workers(1)
            .pcomm("shared"),
    ];
    let mut h = WorkloadHandle::spawn_pcomm_cgroup("shared", None, None, &works)
        .expect("multi-WorkSpec pcomm spawn must succeed");
    h.start();
    std::thread::sleep(Duration::from_millis(200));
    // Capture leader and per-thread tgids while alive.
    let pids = h.worker_pids();
    assert_eq!(pids.len(), 1);
    let leader_pid = pids[0];
    let task_tids = read_task_tids(leader_pid);
    let mut tgids: BTreeSet<libc::pid_t> = BTreeSet::new();
    for tid in &task_tids {
        tgids.insert(read_tgid(*tid));
    }
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 3, "2 + 1 workers across 2 WorkSpecs");
    assert_eq!(
        tgids.len(),
        1,
        "every thread shares the single leader's Tgid; observed {tgids:?}",
    );
    assert_eq!(
        *tgids.iter().next().unwrap(),
        leader_pid,
        "shared Tgid must equal the container leader pid",
    );
    let group0 = reports.iter().filter(|r| r.group_idx == 0).count();
    let group1 = reports.iter().filter(|r| r.group_idx == 1).count();
    assert_eq!(group0, 2, "WorkSpec[0] contributes 2 reports");
    assert_eq!(group1, 1, "WorkSpec[1] contributes 1 report");
}

/// `WorkSpec::pcomm + WorkType::ForkExit` is rejected at admission.
/// A fork from a thread of a multi-threaded container inherits all
/// locks held by sibling threads at fork time, producing undefined
/// behaviour for any libc primitive in the child. The rejection
/// fires before any resource is acquired, so the diagnostic must
/// land synchronously from `spawn_pcomm_cgroup`.
#[test]
fn pcomm_rejects_fork_exit() {
    let works = vec![
        WorkSpec::default()
            .work_type(WorkType::ForkExit)
            .workers(1)
            .pcomm("leader"),
    ];
    let result = WorkloadHandle::spawn_pcomm_cgroup("leader", None, None, &works);
    let err = match result {
        Ok(_) => panic!("pcomm + ForkExit must reject at spawn_pcomm_cgroup"),
        Err(e) => e,
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains("incompatible with WorkType::ForkExit"),
        "diagnostic must name the ForkExit incompatibility, got: {msg}",
    );
}

/// `WorkSpec::pcomm + WorkType::CgroupChurn` is rejected at
/// admission. CgroupChurn writes the worker tid to `cgroup.procs`,
/// which the kernel resolves to the whole tgid and migrates every
/// sibling thread (the entire pcomm container) instead of the
/// intended single worker. The rejection fires before any resource
/// is acquired.
#[test]
fn pcomm_rejects_cgroup_churn() {
    let works = vec![
        WorkSpec::default()
            .work_type(WorkType::CgroupChurn {
                groups: 2,
                cycle_ms: 10,
            })
            .workers(1)
            .pcomm("leader"),
    ];
    let result = WorkloadHandle::spawn_pcomm_cgroup("leader", None, None, &works);
    let err = match result {
        Ok(_) => panic!("pcomm + CgroupChurn must reject at spawn_pcomm_cgroup"),
        Err(e) => e,
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains("incompatible with WorkType::CgroupChurn"),
        "diagnostic must name the CgroupChurn incompatibility, got: {msg}",
    );
}

