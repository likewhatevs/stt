//! VM-backed integration test for [`ktstr::host_state::capture`].
//!
//! Boots a minimal KVM guest via the `#[ktstr_test]` harness,
//! runs a short CPU-spinning workload, then invokes `capture()`
//! INSIDE the guest to read its own `/proc`. The assertion
//! verifies that the returned snapshot carries at least some
//! threads with non-zero scheduling activity — the one visible
//! end-to-end signal that proves the capture layer's
//! procfs/cgroup walk is wired through to the harness.
//!
//! Distinct from `tests/host_state_compare.rs`, which exercises
//! the compare pipeline against SYNTHETIC snapshots (no VM, no
//! real procfs). This file is the counterpart: real capture,
//! real procfs, VM-booted guest kernel — the two tests together
//! cover the full "host-state capture → compare" end-to-end
//! surface once both sides are in place.

use anyhow::Result;
use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
use ktstr::ktstr_test;
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps};

/// Run a short CPU-spinning workload inside the guest, then call
/// [`ktstr::host_state::capture`] against the guest's `/proc` and
/// cgroup v2 mount. The assertion proves that at least one
/// thread in the snapshot has observable scheduling activity —
/// the cross-kernel-config signal that survives whether
/// `CONFIG_SCHED_DEBUG` is enabled or not (page faults on
/// `/proc/<tid>/stat` field 10 are populated unconditionally).
///
/// Why schedstat-OR-minflt: `CONFIG_SCHEDSTATS` is compiled in
/// (see ktstr.kconfig) but runtime-disabled until `sysctl
/// kernel.sched_schedstats=1` fires. A guest that doesn't
/// enable it sees every `run_time_ns` / `voluntary_csw` / etc.
/// field as zero. `minflt` (page faults) does not depend on
/// any compile-time flag — any thread that dirtied a COW page
/// to print its own argv has a non-zero count. Using
/// `run_time_ns > 0 || voluntary_csw > 0 || nr_wakeups > 0 ||
/// minflt > 0` accepts the common case AND the schedstat-off
/// case without softening the "did ANY counter land" invariant.
///
/// Topology: 1 LLC / 2 cores / 1 thread — minimal. The test
/// cares about the capture surface, not about scheduler-level
/// behaviour; a larger topology just lengthens the run for no
/// added signal.
///
/// Duration: 3 s — enough wall-clock for the workers to rack
/// up meaningful schedstat / minflt counters before the capture
/// fires. Shorter windows (< 1 s) risk the workers not having
/// faulted their stacks yet on slow CI runners.
#[ktstr_test(llcs = 1, cores = 2, threads = 1, duration_s = 3)]
fn host_state_capture_returns_threads_with_nonzero_counters(
    ctx: &Ctx,
) -> Result<AssertResult> {
    // Simple CpuSpin workload — workers hit the dispatcher,
    // accrue run_time_ns / voluntary_csw / minflt. One cgroup,
    // default workers_per_cgroup so the test doesn't need to
    // pick a specific count.
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    let workload_result = execute_steps(ctx, steps)?;

    // Capture the guest's host-state after the workload has had
    // a chance to generate activity. `capture()` walks `/proc`
    // and `/sys/fs/cgroup` against the guest's own mount points
    // — inside the VM, those resolve to the guest kernel's live
    // procfs, not the outer host's.
    let snap = ktstr::host_state::capture();

    // First-level sanity: the walk visited SOMETHING. An empty
    // snapshot would mean `iter_tgids_at(/proc)` returned no
    // entries — procfs not mounted, cgroup root unreadable, or
    // a similar plumbing regression.
    if snap.threads.is_empty() {
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            "host_state::capture() returned zero threads — procfs \
             walk produced no entries, indicating the capture layer \
             is not reading /proc successfully inside the guest",
        )));
    }

    // Count threads with ANY non-zero activity counter. The OR
    // across schedstat + page-fault + wakeup fields keeps the
    // assertion robust against kernel configs where SCHEDSTATS
    // or SCHED_DEBUG is compiled but runtime-off.
    let active_threads = snap
        .threads
        .iter()
        .filter(|t| {
            t.run_time_ns > 0
                || t.voluntary_csw > 0
                || t.nonvoluntary_csw > 0
                || t.nr_wakeups > 0
                || t.timeslices > 0
                || t.minflt > 0
        })
        .count();

    if active_threads == 0 {
        // Dump a short diagnostic: which fields ARE populated
        // across the snapshot at all? Helps a reviewer chasing
        // a regression tell "capture is working but schedstat
        // fields are all zero" from "capture returned bogus
        // data" without re-running with tracing.
        let any_comm_nonempty = snap.threads.iter().any(|t| !t.comm.is_empty());
        let any_tgid_nonzero = snap.threads.iter().any(|t| t.tgid > 0);
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "host_state::capture() returned {} threads but NONE \
                 had a non-zero scheduling or page-fault counter; \
                 any_comm_nonempty={any_comm_nonempty}, \
                 any_tgid_nonzero={any_tgid_nonzero}. Suggests the \
                 capture layer reached procfs but every counter \
                 read collapsed to Default — likely a
                 /proc/<tid>/{{sched,stat}} parse regression.",
                snap.threads.len(),
            ),
        )));
    }

    // Capture itself works and saw activity. Return the
    // scenario's own AssertResult so any scheduling failure
    // surfaces alongside the host-state check — the two signals
    // are orthogonal.
    Ok(workload_result)
}
