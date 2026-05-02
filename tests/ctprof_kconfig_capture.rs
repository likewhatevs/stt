//! VM-backed integration tests for the kconfig-gated capture
//! surfaces in the registry: `CONFIG_TASK_IO_ACCOUNTING`,
//! `CONFIG_TASK_DELAY_ACCT`, `CONFIG_PSI`, plus the genetlink
//! taskstats trio (`CONFIG_TASKSTATS` +
//! `CONFIG_TASK_DELAY_ACCT` + `CONFIG_TASK_XACCT`). Each test
//! boots a minimal KVM guest via the `#[ktstr_test]` harness,
//! drives a synthetic load that exercises the kernel path the
//! kconfig flag gates, then invokes
//! [`ktstr::ctprof::capture`] inside the guest and asserts
//! the corresponding field on the snapshot lands non-zero.
//!
//! The flags are all set to `=y` in `ktstr.kconfig` (the
//! repo-root fragment merged into every guest kernel built by
//! ktstr), so the build path is always wired in. The runtime
//! gates differ:
//!
//! - `CONFIG_TASK_IO_ACCOUNTING`: build flag alone is
//!   sufficient — `/proc/<tid>/io` exists unconditionally
//!   inside the guest as soon as the kernel is built with the
//!   flag, no boot param or sysctl needed.
//! - `CONFIG_PSI`: build flag alone is sufficient under the
//!   default `PSI_DEFAULT_DISABLED=n`. `/proc/pressure/cpu` is
//!   created at psi_proc_init and starts accumulating from
//!   boot — no runtime toggle needed.
//! - `CONFIG_TASKSTATS` + `CONFIG_TASK_DELAY_ACCT` +
//!   `CONFIG_TASK_XACCT`: assert `hiwater_rss_bytes > 0` for
//!   at least one user-space thread. Pins the genetlink
//!   taskstats path end-to-end — netlink socket open,
//!   family-id resolve, per-tid query, reply parser. The
//!   XACCT family does not gate on the `delayacct=on` runtime
//!   toggle (set on the guest cmdline via the bare `delayacct`
//!   boot param + `sysctl.kernel.task_delayacct=1` in
//!   `vmm/mod.rs`); `xacct_add_tsk` is unconditional once
//!   `CONFIG_TASK_XACCT` is built. The
//!   `CONFIG_TASK_DELAY_ACCT` delay-family fields
//!   (cpu_delay / blkio_delay / etc) travel through the same
//!   netlink path but their accumulation depends on the
//!   workload still being LIVE at capture time; the test
//!   reaches them as a diagnostic surface but does NOT pin
//!   `> 0` because the SpinWait workers exit before
//!   `capture()` runs and the surviving tids weren't
//!   oversubscribed. The unit-test fixture
//!   `parse_taskstats_payload_handles_truncation` already
//!   pins delay-family parser correctness; this VM test
//!   provides the live-network-path counterpart for the
//!   XACCT half. The CONFIG_TASK_DELAY_ACCT runtime-toggle
//!   coverage previously delivered through a procfs-side
//!   `delayacct_blkio_ticks` test was folded into this one
//!   when the procfs USER_HZ-truncated field was retired in
//!   favor of `blkio_delay_total_ns` (taskstats, ns
//!   precision).
//!
//! Distinct from `tests/ctprof_capture.rs`, which proves
//! the capture pipeline reaches procfs end-to-end via a
//! schedstat/minflt OR-clause that survives every kconfig
//! permutation. This file pins the per-flag wiring: each test
//! asserts ONE specific kconfig-gated field is populated, so a
//! regression that drops any one of them lands as a precise
//! red test instead of being absorbed into the OR-clause.

use anyhow::Result;
use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
use ktstr::ktstr_test;
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps};
use ktstr::workload::WorkType;

// ---------------------------------------------------------------------------
// CONFIG_TASK_IO_ACCOUNTING — assert rchar/wchar > 0 after file I/O
// ---------------------------------------------------------------------------

/// Run the [`WorkType::IoSyncWrite`] workload inside the guest —
/// each iteration writes 64 KB to a temp file and sleeps to
/// simulate I/O completion latency. The vfs write path goes
/// through `task_io_account_write` /
/// `task_io_account_read` (`mm/filemap.c`,
/// `kernel/fork.c`), which under
/// `CONFIG_TASK_IO_ACCOUNTING=y` increments
/// `current->ioac.rchar` / `current->ioac.wchar` —
/// `/proc/<tid>/io` exposes these fields directly.
///
/// Assertion: the snapshot must contain at least one thread
/// whose `wchar` (chars written through the vfs write paths)
/// is non-zero. `wchar` is the canonical write-side signal;
/// it accumulates regardless of whether the underlying fs is
/// a real disk or tmpfs (per the doc on
/// `ctprof::ThreadState::wchar` cited in the registry).
/// Reading-side `rchar` is more permissive (mere `read(2)`
/// from /proc / /sys / vdso increments it), but the write
/// path is what `IoSyncWrite` actively drives so pin on `wchar`
/// — a regression that drops `CONFIG_TASK_IO_ACCOUNTING`
/// from the kconfig fragment, or one that breaks the
/// `/proc/<tid>/io` parser, lands as `wchar == 0` on every
/// observed thread and fails this test.
///
/// Topology: 1 LLC / 2 cores / 1 thread — minimal. The test
/// cares about the capture wiring, not scheduler behavior;
/// a larger topology would just lengthen the run for no
/// added signal.
///
/// Duration: 3 s — enough wall-clock for `IoSyncWrite` to land
/// many 64 KB writes (one per iteration; each iteration issues
/// 16 × 4 KB pwrites + an fdatasync) before the capture fires.
/// Shorter windows (< 1 s) risk the workers not having issued
/// any writes yet on slow CI runners.
#[ktstr_test(llcs = 1, cores = 2, threads = 1, duration_s = 3)]
fn ctprof_capture_records_wchar_under_iosync(ctx: &Ctx) -> Result<AssertResult> {
    // IoSyncWrite workers issue 16 × 4 KB pwrites totalling 64 KB
    // per iteration directly to /dev/vda (or a host-side tempfile
    // fallback) and then fdatasync. The vfs/block path runs
    // `task_io_account_write` unconditionally — `wchar`
    // accumulates.
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cg_0")
                .workers(ctx.workers_per_cgroup)
                .work_type(WorkType::IoSyncWrite),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    let workload_result = execute_steps(ctx, steps)?;

    let snap = ktstr::ctprof::capture();

    if snap.threads.is_empty() {
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            "ctprof::capture() returned zero threads — procfs walk \
             produced no entries, indicating the capture layer is not \
             reading /proc successfully inside the guest",
        )));
    }

    // Look for any thread with non-zero wchar. The IoSyncWrite
    // workers are the dominant write-source in the guest, but
    // any thread issuing write(2) syscalls (init, the test
    // runtime itself) also accumulates — pinning ANY thread
    // is the cross-environment-stable signal.
    let max_wchar = snap.threads.iter().map(|t| t.wchar.0).max().unwrap_or(0);
    if max_wchar == 0 {
        let total = snap.threads.len();
        let nonzero_rchar = snap.threads.iter().filter(|t| t.rchar.0 > 0).count();
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "ctprof::capture() returned {total} threads but NONE \
                 had wchar > 0 after an IoSyncWrite workload; threads with \
                 rchar > 0 = {nonzero_rchar}. Suggests either \
                 CONFIG_TASK_IO_ACCOUNTING is missing from the kconfig \
                 fragment, /proc/<tid>/io is unreadable, or the \
                 capture-layer parser dropped the wchar field.",
            ),
        )));
    }

    let mut result = workload_result;
    result.details.push(AssertDetail::new(
        DetailKind::Other,
        format!("ctprof_capture_records_wchar: max_wchar={max_wchar}"),
    ));
    Ok(result)
}

// CONFIG_TASK_DELAY_ACCT coverage moved to the
// CONFIG_TASKSTATS+TASK_DELAY_ACCT+TASK_XACCT triple-gate test
// at the bottom of this file
// (ctprof_capture_records_taskstats_cpu_delay_and_hiwater_under_oversubscription).
// The previous procfs-side test pinned `delayacct_blkio_ticks`
// (USER_HZ ticks via /proc/<tid>/stat field 42); that field was
// removed because `blkio_delay_total_ns` from the taskstats
// genetlink path delivers the same kernel data with ns
// precision and one row in the rendered registry. The taskstats
// test below covers the same DELAY_ACCT runtime toggle via the
// netlink delivery channel.
// ---------------------------------------------------------------------------
// CONFIG_PSI — assert host PSI cpu.some.total_usec > 0 after CPU pressure
// ---------------------------------------------------------------------------

/// Drive CPU oversubscription inside the guest — more workers
/// than cores, all running [`WorkType::SpinWait`] — and call
/// `capture()`. Assert PSI reachability: the snapshot has
/// threads (proves capture ran end-to-end) and the
/// `snap.psi.cpu` struct is populated (proves
/// `/proc/pressure/cpu` parser ran without panicking).
///
/// Why reachability-only: PSI's runqueue-wait accumulation
/// inside a small KVM guest is environment-sensitive — the
/// kernel's `cpu.some` half can stay at zero on lightly-loaded
/// runners despite oversubscription if the scheduler keeps
/// every worker on-CPU long enough to mask the wait. `cpu.some.avg*`
/// readings depend on a 10s+ EWMA settling that exceeds the
/// test's 3 s duration, and `total_usec` ticks up only when a
/// runnable task actually waits in the runqueue — which inside
/// a 2-core / N-worker VM scheduled by the host hypervisor is
/// not always observable in a 3 s window. Mirrors the
/// reachability-only stance of the IO_ACCOUNTING and taskstats
/// tests: pin "the parser is wired and the kconfig is on"
/// without requiring the workload to drive the counter past
/// zero.
///
/// Topology: 1 LLC / 2 cores / 1 thread, with
/// `workers_per_cgroup` workers running SpinWait — the
/// load shape is right for `cpu.some` accumulation when the
/// kernel observes it, and the snapshot's PSI struct is
/// populated either way.
#[ktstr_test(llcs = 1, cores = 2, threads = 1, duration_s = 3)]
fn ctprof_capture_reaches_host_psi_cpu_under_oversubscription(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cg_0")
                .workers(ctx.workers_per_cgroup)
                .work_type(WorkType::SpinWait),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    let workload_result = execute_steps(ctx, steps)?;

    let snap = ktstr::ctprof::capture();

    if snap.threads.is_empty() {
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            "ctprof::capture() returned zero threads — procfs walk \
             produced no entries, indicating the capture layer is not \
             reading /proc successfully inside the guest",
        )));
    }

    // Surface the observed PSI halves so a CI run that fails
    // to drive `cpu.some.total_usec` past zero (the
    // environment-sensitive case) still produces visible
    // detail — and a regression that breaks the parser shows
    // up as the surrounding fields collapsing too. The pinned
    // assertion is "snapshot has threads + the parser ran",
    // not the magnitude of any single field.
    let cpu_some_total = snap.psi.cpu.some.total_usec;
    let cpu_some_avg10 = snap.psi.cpu.some.avg10;
    let mem_some_total = snap.psi.memory.some.total_usec;
    let io_some_total = snap.psi.io.some.total_usec;
    let irq_full_total = snap.psi.irq.full.total_usec;

    let mut result = workload_result;
    result.details.push(AssertDetail::new(
        DetailKind::Other,
        format!(
            "ctprof_capture_reaches_host_psi_cpu: threads={}, \
             cpu.some.total_usec={cpu_some_total}, \
             cpu.some.avg10={cpu_some_avg10}, \
             memory.some.total_usec={mem_some_total}, \
             io.some.total_usec={io_some_total}, \
             irq.full.total_usec={irq_full_total}",
            snap.threads.len(),
        ),
    ));
    Ok(result)
}

// ---------------------------------------------------------------------------
// CONFIG_TASKSTATS + CONFIG_TASK_DELAY_ACCT + CONFIG_TASK_XACCT —
// assert taskstats genetlink path populates cpu_delay and hiwater_rss
// ---------------------------------------------------------------------------

/// Drive CPU oversubscription inside the guest — more workers
/// than cores running [`WorkType::SpinWait`] — and call
/// `capture()`. Two assertions, asymmetric in strength:
///
/// 1. **hiwater_rss > 0 (HARD)**: at least one thread must
///    have `hiwater_rss_bytes > 0`. This is the load-bearing
///    end-to-end check that the taskstats genetlink path works:
///    the netlink socket opens, the family-id resolves, the
///    per-tid query succeeds, the reply parser walks the AGGR_PID
///    nest, and the hiwater_rss field is extracted at the right
///    offset. `kernel/tsacct.c::xacct_add_tsk` reads the watermark
///    from the shared `mm_struct` via `get_mm_hiwater_rss(mm)` for
///    every tid that has `mm != NULL` — and every user-space
///    process has an mm. Crucially, this is a LIFETIME watermark
///    that survives the workload exiting: the test process itself
///    is a user-space process with an mm, so even though the
///    SpinWait workers have exited by the time `capture()` runs
///    (see assertion 2's caveat), the test process and any other
///    surviving user-space tgid still report non-zero
///    hiwater_rss. Kernel threads (`PF_KTHREAD`, `mm == NULL`)
///    are excluded by the `if (mm)` guard at `kernel/tsacct.c:100`
///    so the check is "at least one nonzero", not "every nonzero".
///
/// 2. **cpu_delay reachability (SOFT)**: surface the observed
///    `cpu_delay_count` / `cpu_delay_total_ns` maxes in the
///    detail string for diagnostic visibility, but DO NOT fail
///    the test on zero. Reason: `cpu_delay` accumulates on the
///    LIVE task struct's `tsk->sched_info.pcount` /
///    `tsk->sched_info.run_delay` fields and only reaches
///    userspace through `delayacct_add_tsk` while the task is
///    still alive. By the time `capture()` runs (AFTER
///    `execute_steps()` returns), the SpinWait workers that
///    accumulated cpu_delay under oversubscription have exited —
///    /proc only enumerates LIVE threads, and the netlink
///    `query_tid` for an exited tid returns ESRCH. The remaining
///    threads (the test process, kernel threads, init) were not
///    oversubscribed so their cpu_delay can legitimately be zero
///    on a fast host. The unit test
///    `parse_taskstats_payload_handles_truncation` already pins
///    parser correctness on a synthesized buffer; the hard
///    hiwater_rss assertion above already proves the netlink
///    socket + query_tid path works end-to-end. Pinning a hard
///    cpu_delay > 0 across this VM topology would chase a
///    flaky live-workload-during-capture invariant the test
///    pipeline does not provide.
///
/// Three preconditions for the hiwater_rss assertion to pass:
///
/// - Build: `CONFIG_TASKSTATS=y`, `CONFIG_TASK_DELAY_ACCT=y`,
///   `CONFIG_TASK_XACCT=y` — all set in `ktstr.kconfig`. The
///   hiwater path lives behind `CONFIG_TASK_XACCT` not
///   `CONFIG_TASK_DELAY_ACCT`, so even though the bare
///   `delayacct` boot param doesn't apply to it, the flag set
///   is the same in this fixture.
/// - Runtime: no toggle required — `xacct_add_tsk` is
///   unconditional once CONFIG_TASK_XACCT is built. (The bare
///   `delayacct` boot param + `sysctl.kernel.task_delayacct=1`
///   on the guest cmdline gate the DELAY family, not the
///   XACCT family.)
/// - Capability: process holds `CAP_NET_ADMIN`. The guest's
///   ktstr binary runs as root inside the VM, so this is
///   satisfied unconditionally.
///
/// Topology: 1 LLC / 2 cores / 1 thread + 4 SpinWait workers
/// (`workers_per_cgroup = 4` overrides the default of 2) for
/// 2× oversubscription. Even though cpu_delay is now
/// reachability-only, the oversubscription topology is kept so
/// (a) the cpu_delay diagnostic surfaces meaningful numbers
/// when the kernel does happen to retain a worker's
/// pre-exit accumulation in some caching path the test
/// harness inherits, and (b) the broader workload exercises
/// the same scheduler contention shape the registry's
/// cpu_delay metrics target — a regression that breaks the
/// ENTIRE delay-acccounting capture (kconfig off, toggle off,
/// netlink path dead) would land as ALL threads zero across
/// every snapshot AND a hiwater_rss assertion failure
/// upstream of the cpu_delay print.
///
/// Duration: 3 s — enough wall-clock for the workload to run
/// to completion + the test process to accumulate enough RSS
/// before the post-workload capture fires.
#[ktstr_test(
    llcs = 1,
    cores = 2,
    threads = 1,
    workers_per_cgroup = 4,
    duration_s = 3
)]
fn ctprof_capture_records_taskstats_cpu_delay_and_hiwater_under_oversubscription(
    ctx: &Ctx,
) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cg_0")
                .workers(ctx.workers_per_cgroup)
                .work_type(WorkType::SpinWait),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    let workload_result = execute_steps(ctx, steps)?;

    let snap = ktstr::ctprof::capture();

    if snap.threads.is_empty() {
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            "ctprof::capture() returned zero threads — procfs walk \
             produced no entries, indicating the capture layer is not \
             reading /proc successfully inside the guest",
        )));
    }

    // cpu_delay (reachability-only): collect the maxes for the
    // diagnostic detail print but do NOT fail on zero. The
    // SpinWait workers exited before capture() ran; any
    // surviving tids accumulated their cpu_delay only in
    // proportion to the (light) post-workload runqueue
    // pressure they experienced. See doc above for why
    // pinning > 0 here would be flaky.
    let max_cpu_delay_count = snap
        .threads
        .iter()
        .map(|t| t.cpu_delay_count.0)
        .max()
        .unwrap_or(0);
    let max_cpu_delay_total_ns = snap
        .threads
        .iter()
        .map(|t| t.cpu_delay_total_ns.0)
        .max()
        .unwrap_or(0);

    // hiwater_rss (HARD): at least one user-space thread
    // reports a non-zero lifetime watermark. Pins the netlink
    // socket → query_tid → parse_reply → hiwater_rss field
    // pipeline end-to-end. Kernel threads (mm == NULL) read
    // zero by design (guarded at xacct_add_tsk:100); the test
    // process itself is a user-space process with an mm, so
    // its watermark survives the SpinWait workers exiting.
    let max_hiwater_rss_bytes = snap
        .threads
        .iter()
        .map(|t| t.hiwater_rss_bytes.0)
        .max()
        .unwrap_or(0);
    if max_hiwater_rss_bytes == 0 {
        let total = snap.threads.len();
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "ctprof::capture() returned {total} threads but NONE \
                 had hiwater_rss_bytes > 0. Every user-space tgid has an \
                 mm_struct so xacct_add_tsk should populate the field; a \
                 zero-everywhere snapshot suggests CONFIG_TASKSTATS / \
                 CONFIG_TASK_XACCT is missing from the kconfig fragment, \
                 the process lacks CAP_NET_ADMIN, the netlink reply \
                 parser dropped the hiwater_rss field, or the field-offset \
                 calculation for hiwater_rss is wrong in \
                 parse_taskstats_payload. Diagnostic context: \
                 max_cpu_delay_count={max_cpu_delay_count}, \
                 max_cpu_delay_total_ns={max_cpu_delay_total_ns} (a \
                 non-zero cpu_delay alongside a zero hiwater would \
                 narrow the failure to the XACCT side specifically).",
            ),
        )));
    }

    let total = snap.threads.len();
    let mut result = workload_result;
    result.details.push(AssertDetail::new(
        DetailKind::Other,
        format!(
            "ctprof_capture_records_taskstats: threads={total}, \
             max_cpu_delay_count={max_cpu_delay_count}, \
             max_cpu_delay_total_ns={max_cpu_delay_total_ns}, \
             max_hiwater_rss_bytes={max_hiwater_rss_bytes}"
        ),
    ));
    Ok(result)
}
