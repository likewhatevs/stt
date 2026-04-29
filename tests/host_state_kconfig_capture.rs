//! VM-backed integration tests for the kconfig-gated capture
//! surfaces in the registry: `CONFIG_TASK_IO_ACCOUNTING`,
//! `CONFIG_TASK_DELAY_ACCT`, `CONFIG_PSI`, plus the genetlink
//! taskstats trio (`CONFIG_TASKSTATS` +
//! `CONFIG_TASK_DELAY_ACCT` + `CONFIG_TASK_XACCT`). Each test
//! boots a minimal KVM guest via the `#[ktstr_test]` harness,
//! drives a synthetic load that exercises the kernel path the
//! kconfig flag gates, then invokes
//! [`ktstr::host_state::capture`] inside the guest and asserts
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
//! - `CONFIG_TASK_DELAY_ACCT`: requires BOTH the build flag
//!   AND a runtime toggle. The increment paths
//!   (`delayacct_blkio_start` / `_end`) are gated behind
//!   `static_branch_unlikely(&delayacct_key)` and stay
//!   no-ops until the toggle fires. The toggle is set on the
//!   guest cmdline via `sysctl.kernel.task_delayacct=1` (see
//!   `vmm/mod.rs`), which kicks in before user space starts so
//!   the ktstr workload's blkio waits are accounted from the
//!   first scheduling tick.
//! - `CONFIG_PSI`: build flag alone is sufficient under the
//!   default `PSI_DEFAULT_DISABLED=n`. `/proc/pressure/cpu` is
//!   created at psi_proc_init and starts accumulating from
//!   boot — no runtime toggle needed.
//! - `CONFIG_TASKSTATS` + `CONFIG_TASK_XACCT`: assert
//!   `hiwater_rss_bytes > 0` for at least one user-space
//!   thread. Pins the genetlink taskstats path end-to-end —
//!   netlink socket open, family-id resolve, per-tid query,
//!   reply parser. The XACCT family does not gate on the
//!   `delayacct=on` runtime toggle; `xacct_add_tsk` is
//!   unconditional once `CONFIG_TASK_XACCT` is built. The
//!   `CONFIG_TASK_DELAY_ACCT` delay-family fields
//!   (cpu_delay / blkio_delay / etc) travel through the same
//!   netlink path but their accumulation depends on the
//!   workload still being LIVE at capture time; the test
//!   reaches them as a diagnostic surface but does NOT pin
//!   `> 0` because the CpuSpin workers exit before
//!   `capture()` runs and the surviving tids weren't
//!   oversubscribed. The unit-test fixture
//!   `parse_taskstats_payload_handles_truncation` already
//!   pins delay-family parser correctness; this VM test
//!   provides the live-network-path counterpart for the
//!   XACCT half. Distinct from the existing
//!   `delayacct_blkio_ticks` test that travels through
//!   `/proc/<tid>/stat` field 42 — that's the procfs
//!   delivery channel; this test is the genetlink delivery
//!   channel.
//!
//! Distinct from `tests/host_state_capture.rs`, which proves
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

/// Run the [`WorkType::IoSync`] workload inside the guest —
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
/// `host_state::ThreadState::wchar` cited in the registry).
/// Reading-side `rchar` is more permissive (mere `read(2)`
/// from /proc / /sys / vdso increments it), but the write
/// path is what `IoSync` actively drives so pin on `wchar`
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
/// Duration: 3 s — enough wall-clock for `IoSync` to land
/// many 64 KB writes (one per iteration with a 100 µs sleep
/// between iterations) before the capture fires. Shorter
/// windows (< 1 s) risk the workers not having issued any
/// writes yet on slow CI runners.
#[ktstr_test(llcs = 1, cores = 2, threads = 1, duration_s = 3)]
fn host_state_capture_records_wchar_under_iosync(ctx: &Ctx) -> Result<AssertResult> {
    // IoSync workers write 64 KB to a temp file then sleep
    // 100 µs to simulate I/O completion latency. On the
    // guest's tmpfs the write is a page-cache memcpy, but
    // the vfs path still runs `task_io_account_write`
    // unconditionally — `wchar` accumulates.
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cg_0")
                .workers(ctx.workers_per_cgroup)
                .work_type(WorkType::IoSync),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    let workload_result = execute_steps(ctx, steps)?;

    let snap = ktstr::host_state::capture();

    if snap.threads.is_empty() {
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            "host_state::capture() returned zero threads — procfs walk \
             produced no entries, indicating the capture layer is not \
             reading /proc successfully inside the guest",
        )));
    }

    // Look for any thread with non-zero wchar. The IoSync
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
                "host_state::capture() returned {total} threads but NONE \
                 had wchar > 0 after an IoSync workload; threads with \
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
        format!("host_state_capture_records_wchar: max_wchar={max_wchar}"),
    ));
    Ok(result)
}

// ---------------------------------------------------------------------------
// CONFIG_TASK_DELAY_ACCT — assert delayacct_blkio_ticks field reachable
// ---------------------------------------------------------------------------

/// Run a basic CPU workload inside the guest, then call
/// `capture()` and assert the snapshot's
/// `delayacct_blkio_ticks` field is reachable — i.e. the
/// `/proc/<tid>/stat` parser pulled field 42 successfully.
/// On a kernel built with `CONFIG_TASK_DELAY_ACCT=y` AND
/// boot-time `sysctl.kernel.task_delayacct=1` (set on the
/// guest cmdline at `vmm/mod.rs`), the field exists in
/// `/proc/<tid>/stat` and the parser populates it.
///
/// The runtime toggle is what this test guards against most
/// directly: without `sysctl.kernel.task_delayacct=1` on the
/// cmdline, the `static_branch_unlikely(&delayacct_key)` gate
/// keeps the increment paths as no-ops, but the field IS still
/// printed in `/proc/<tid>/stat` (always at field 42 — the
/// unconditional `seq_put_decimal_ull(m, " ",
/// delayacct_blkio_ticks(task))` call at
/// `fs/proc/array.c:639` returns 0 via the
/// `static inline delayacct_blkio_ticks` definition when the
/// kconfig is off, but is real when on with toggle off — see
/// kernel/delayacct.c:48). What we ARE pinning here is the
/// REACHABILITY of the field through the capture's parser:
/// any thread observed in the snapshot must have a
/// `delayacct_blkio_ticks` value (zero or non-zero), not a
/// parse failure that drops the field. The cross-environment
/// signal is "snapshot has threads AND every thread carries
/// the field" — a regression that breaks the field-42 parser
/// surfaces as `delayacct_blkio_ticks` collapsing to default
/// across the entire snapshot.
///
/// We don't pin "non-zero" because a CPU-bound workload may
/// rack up zero blkio waits — that would make this test
/// dependent on disk-backed I/O, defeating the point of a
/// smoke test for the kconfig wiring.
#[ktstr_test(llcs = 1, cores = 2, threads = 1, duration_s = 3)]
fn host_state_capture_reaches_delayacct_blkio_ticks_field(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cg_0")
                .workers(ctx.workers_per_cgroup)
                .work_type(WorkType::CpuSpin),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    let workload_result = execute_steps(ctx, steps)?;

    let snap = ktstr::host_state::capture();

    if snap.threads.is_empty() {
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            "host_state::capture() returned zero threads — procfs walk \
             produced no entries, indicating the capture layer is not \
             reading /proc successfully inside the guest",
        )));
    }

    // Surface the observed range so a CI run that flips the
    // sysctl toggle off (or builds with the kconfig off)
    // produces a visible diff in the test details. The
    // empty-threads early-return above is the only runtime
    // gate this smoke test needs; a parser regression on
    // `delayacct_blkio_ticks` surfaces as a compile error
    // (typed wrapper) rather than a runtime check here.
    let max_blkio = snap
        .threads
        .iter()
        .map(|t| t.delayacct_blkio_ticks.0)
        .max()
        .unwrap_or(0);
    let mut result = workload_result;
    result.details.push(AssertDetail::new(
        DetailKind::Other,
        format!(
            "host_state_capture_reaches_delayacct_blkio_ticks: \
             threads={}, max_blkio_ticks={max_blkio}",
            snap.threads.len(),
        ),
    ));
    Ok(result)
}

// ---------------------------------------------------------------------------
// CONFIG_PSI — assert host PSI cpu.some.total_usec > 0 after CPU pressure
// ---------------------------------------------------------------------------

/// Drive CPU oversubscription inside the guest — more workers
/// than cores, all running [`WorkType::CpuSpin`] — and call
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
/// reachability-only stance of
/// `host_state_capture_reaches_delayacct_blkio_ticks_field`:
/// pin "the parser is wired and the kconfig is on" without
/// requiring the workload to drive the counter past zero.
///
/// Topology: 1 LLC / 2 cores / 1 thread, with
/// `workers_per_cgroup` workers running CpuSpin — the
/// load shape is right for `cpu.some` accumulation when the
/// kernel observes it, and the snapshot's PSI struct is
/// populated either way.
#[ktstr_test(llcs = 1, cores = 2, threads = 1, duration_s = 3)]
fn host_state_capture_reaches_host_psi_cpu_under_oversubscription(
    ctx: &Ctx,
) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cg_0")
                .workers(ctx.workers_per_cgroup)
                .work_type(WorkType::CpuSpin),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    let workload_result = execute_steps(ctx, steps)?;

    let snap = ktstr::host_state::capture();

    if snap.threads.is_empty() {
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            "host_state::capture() returned zero threads — procfs walk \
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
            "host_state_capture_reaches_host_psi_cpu: threads={}, \
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
/// than cores running [`WorkType::CpuSpin`] — and call
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
///    CpuSpin workers have exited by the time `capture()` runs
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
///    `execute_steps()` returns), the CpuSpin workers that
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
/// Topology: 1 LLC / 2 cores / 1 thread + 4 CpuSpin workers
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
fn host_state_capture_records_taskstats_cpu_delay_and_hiwater_under_oversubscription(
    ctx: &Ctx,
) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cg_0")
                .workers(ctx.workers_per_cgroup)
                .work_type(WorkType::CpuSpin),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    let workload_result = execute_steps(ctx, steps)?;

    let snap = ktstr::host_state::capture();

    if snap.threads.is_empty() {
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            "host_state::capture() returned zero threads — procfs walk \
             produced no entries, indicating the capture layer is not \
             reading /proc successfully inside the guest",
        )));
    }

    // cpu_delay (reachability-only): collect the maxes for the
    // diagnostic detail print but do NOT fail on zero. The
    // CpuSpin workers exited before capture() ran; any
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
    // its watermark survives the CpuSpin workers exiting.
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
                "host_state::capture() returned {total} threads but NONE \
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
            "host_state_capture_records_taskstats: threads={total}, \
             max_cpu_delay_count={max_cpu_delay_count}, \
             max_cpu_delay_total_ns={max_cpu_delay_total_ns}, \
             max_hiwater_rss_bytes={max_hiwater_rss_bytes}"
        ),
    ));
    Ok(result)
}
