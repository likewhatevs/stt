//! Scheduler / clock / metric helpers used by `worker_main`. Holds
//! the schedstat parser and reader, NUMA / vmstat readers, the
//! `clock_gettime_ns` wrapper, and `set_sched_policy` (which lowers
//! [`SchedPolicy`] to `sched_setattr` / `sched_setscheduler`).
//! Extracted from `worker/mod.rs` so the production file stays
//! under the per-file line budget.

use anyhow::Result;
use std::collections::BTreeMap;
use std::time::Duration;

use super::super::config::SchedPolicy;

/// Read schedstat for the calling worker and return
/// `(cpu_time_ns, run_delay_ns, timeslices)`.
///
/// `tid` selects which `/proc` path is read:
/// - `None` → `/proc/self/schedstat`. `/proc/self` resolves to
///   `/proc/<TGID>` (the thread-group leader's task_struct), which
///   is correct for [`CloneMode::Fork`] workers because each fork
///   worker IS its own thread-group leader (`gettid() == getpid()`).
/// - `Some(tid)` → `/proc/self/task/<tid>/schedstat`. Required for
///   [`CloneMode::Thread`] workers: every thread in the parent
///   tgid sees the same `/proc/self/schedstat` (the parent's
///   leader stats), so reading it from a thread worker reports
///   the test runner's stats, not the worker's. The
///   `/proc/self/task/<tid>` path returns the per-task
///   schedstat stored on `task->sched_info`. Available on Linux
///   2.6+; ktstr's 6.16 kernel floor guarantees it.
///
/// Returns `None` when the file cannot be opened (kernel built
/// without `CONFIG_SCHEDSTATS`, or `/proc` unavailable) or when any
/// of the first three whitespace-separated fields is missing or not
/// parseable as `u64`. Callers must distinguish "unavailable" from
/// "zero observed" — the previous `(0, 0, 0)`-on-failure return was
/// silently ambiguous across "schedstats disabled", "I/O error",
/// and "worker genuinely did no work yet", which caused
/// `assert_not_starved`-style checks to ratify the wrong invariant
/// on kernels without schedstats.
///
/// Emits a process-wide one-shot warning to stderr the first time
/// the file cannot be opened so the test log records the cause
/// without flooding on every per-worker read. The parse-failure
/// branches return `None` silently — a schedstat line that opens
/// but fails `u64::parse` indicates a kernel-ABI drift that should
/// not occur on any production kernel and warrants investigation by
/// the maintainer rather than per-run log noise.
pub(super) fn read_schedstat(tid: Option<libc::pid_t>) -> Option<(u64, u64, u64)> {
    let path: std::borrow::Cow<'static, str> = match tid {
        None => std::borrow::Cow::Borrowed("/proc/self/schedstat"),
        Some(t) => std::borrow::Cow::Owned(format!("/proc/self/task/{t}/schedstat")),
    };
    let data = match std::fs::read_to_string(&*path) {
        Ok(d) => d,
        Err(_) => {
            warn_schedstat_unavailable_once();
            return None;
        }
    };
    parse_schedstat_line(&data)
}

/// Pure parser split from [`read_schedstat`] for unit testability.
/// Parses the first three whitespace-separated fields of a
/// `/proc/self/schedstat` line as `(cpu_time_ns, run_delay_ns,
/// timeslices)`. Returns `None` when any of the three tokens is
/// missing or not parseable as `u64` — matches the silent-failure
/// contract described on `read_schedstat`. Synthetic fixtures can
/// exercise the parse-failure branches (truncated line, non-u64
/// token, empty input, trailing garbage) without standing up a
/// real `/proc/self/schedstat`.
pub(super) fn parse_schedstat_line(data: &str) -> Option<(u64, u64, u64)> {
    let mut parts = data.split_whitespace();
    let cpu_time = parts.next()?.parse::<u64>().ok()?;
    let run_delay = parts.next()?.parse::<u64>().ok()?;
    let timeslices = parts.next()?.parse::<u64>().ok()?;
    Some((cpu_time, run_delay, timeslices))
}

/// Print a single "schedstat unavailable" warning for the lifetime
/// of the process. The workload spawns `N_WORKERS` threads, each of
/// which calls `read_schedstat` twice; without a gate this would
/// emit up to `2N` duplicate lines on a kernel without
/// `CONFIG_SCHEDSTATS`.
pub(super) fn warn_schedstat_unavailable_once() {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        eprintln!(
            "workload: /proc/self/schedstat unavailable (CONFIG_SCHEDSTATS off?); \
             schedstat_* fields in WorkerReport will be zero"
        );
    });
}

/// Aggregate per-node page counts from `/proc/self/numa_maps`.
/// Returns empty map on failure.
pub(super) fn read_numa_maps_pages() -> BTreeMap<usize, u64> {
    let content = match std::fs::read_to_string("/proc/self/numa_maps") {
        Ok(c) => c,
        Err(_) => return BTreeMap::new(),
    };
    let entries = crate::assert::parse_numa_maps(&content);
    let mut totals: BTreeMap<usize, u64> = BTreeMap::new();
    for entry in &entries {
        for (&node, &count) in &entry.node_pages {
            *totals.entry(node).or_insert(0) += count;
        }
    }
    totals
}

/// Read `numa_pages_migrated` from `/proc/vmstat`. Returns 0 on failure.
pub(super) fn read_vmstat_numa_pages_migrated() -> u64 {
    let content = match std::fs::read_to_string("/proc/vmstat") {
        Ok(c) => c,
        Err(_) => return 0,
    };
    crate::assert::parse_vmstat_numa_pages_migrated(&content).unwrap_or(0)
}

/// Read `clk` via `clock_gettime` and return the raw timespec packed
/// as `tv_sec * 1e9 + tv_nsec` (ns units), or `None` if the syscall
/// fails. The semantics of the returned value depend on `clk`:
/// `CLOCK_MONOTONIC` is nanoseconds since an unspecified boot epoch,
/// `CLOCK_THREAD_CPUTIME_ID` is nanoseconds of CPU time charged to
/// the calling thread. Centralizes the error check that previously
/// was either discarded entirely (producing garbage timespec readings
/// that fed into wake-latency reservoirs) or collapsed to a 0
/// sentinel indistinguishable from "clock read zero".
pub(super) fn clock_gettime_ns(clk: libc::clockid_t) -> Option<u64> {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    let rc = unsafe { libc::clock_gettime(clk, &mut ts) };
    if rc != 0 {
        warn_clock_gettime_failed_once(clk);
        return None;
    }
    Some((ts.tv_sec as u64) * 1_000_000_000 + (ts.tv_nsec as u64))
}

/// Print a single `clock_gettime` failure warning per clock id for
/// the lifetime of the process. Same rationale as
/// `warn_schedstat_unavailable_once`: dozens of workers will fail
/// once each on a misconfigured host. Only `CLOCK_THREAD_CPUTIME_ID`
/// and `CLOCK_MONOTONIC` are ever passed in by this file; any other
/// clock id is a programming error and should panic in development
/// rather than silently falling through to a speculative catch-all.
pub(super) fn warn_clock_gettime_failed_once(clk: libc::clockid_t) {
    static WARNED_THREAD: std::sync::Once = std::sync::Once::new();
    static WARNED_MONO: std::sync::Once = std::sync::Once::new();
    let once = match clk {
        libc::CLOCK_THREAD_CPUTIME_ID => &WARNED_THREAD,
        libc::CLOCK_MONOTONIC => &WARNED_MONO,
        _ => unreachable!("unexpected clockid {clk}"),
    };
    once.call_once(|| {
        // Capture errno INSIDE `call_once` — on every subsequent
        // call the `Once` has already run and the computation is
        // short-circuited, so there is no point paying the syscall
        // cost to read `last_os_error` again just to drop it.
        let errno = std::io::Error::last_os_error();
        eprintln!(
            "workload: clock_gettime(clk={clk}) failed: {errno}; affected samples will be zero or skipped"
        );
    });
}

/// Read the calling thread's CPU-time counter. Returns 0 on syscall
/// failure after emitting a one-shot stderr warning — callers treat
/// the value as a per-thread cumulative counter and cannot usefully
/// distinguish "zero ns" from "clock failed" at the counter's
/// granularity (nanoseconds), so the 0 fallback is an acceptable
/// compromise. The failure path is near-impossible on Linux (kernel
/// must support `CLOCK_THREAD_CPUTIME_ID`, which has been default
/// since 2.6.12). If this lands in a hostile environment where
/// failure is real, callers should migrate to `clock_gettime_ns`
/// directly and handle `None`.
pub(super) fn thread_cpu_time_ns() -> u64 {
    clock_gettime_ns(libc::CLOCK_THREAD_CPUTIME_ID).unwrap_or(0)
}

/// Convert a [`Duration`] to the kernel's `u64` nanosecond
/// representation for `sched_setattr(2)` while enforcing the
/// bit-63-clear constraint `__checkparam_dl` imposes on
/// `sched_deadline` and `sched_period`.
///
/// `Duration::as_nanos()` returns `u128`; the kernel's UAPI struct
/// fields are `u64`. Any duration longer than `i64::MAX` ns
/// (~292 years) either flips bit 63 of the truncated `u64` (kernel
/// reserved) or wraps on the cast entirely. Both outcomes are
/// rejected here so the user sees a named-field error rather than
/// a kernel `EINVAL` after a silent truncation.
///
/// `field` is the human-readable field label embedded in the
/// error message ("runtime", "deadline", "period") so a
/// rejection points at the offending input.
pub(super) fn duration_to_kernel_ns(d: Duration, field: &str) -> Result<u64> {
    let ns_u128 = d.as_nanos();
    if ns_u128 > i64::MAX as u128 {
        anyhow::bail!(
            "sched_setattr: {field} duration ({ns_u128} ns) exceeds i64::MAX — \
             nanosecond count must fit in 63 bits (kernel reserves bit 63)"
        );
    }
    Ok(ns_u128 as u64)
}

/// Lower a [`SchedPolicy`] to a `sched_setscheduler` / `sched_setattr`
/// syscall and apply it to `pid`. `Normal` is a no-op (the kernel
/// default); `Batch`/`Idle` use `SCHED_BATCH`/`SCHED_IDLE` with prio
/// 0; `Fifo`/`RoundRobin` clamp the static priority to `1..=99` and
/// use `sched_setscheduler`; `Deadline` runs the kernel's
/// `__checkparam_dl` structural checks (zero-deadline rejection,
/// `runtime <= deadline <= period` ordering when period is
/// non-zero, DL_SCALE 1024 ns floor, bit-63-clear overflow guard
/// via `duration_to_kernel_ns`) before issuing
/// `syscall(SYS_sched_setattr, ...)` directly because glibc does
/// not wrap that syscall.
///
/// Returns `Err` on any pre-flight rejection or kernel-side
/// EINVAL/EPERM with a named-field error message; the caller
/// treats failure as fatal (worker bails to its cleanup tail).
pub(super) fn set_sched_policy(pid: libc::pid_t, policy: SchedPolicy) -> Result<()> {
    // Reject pid <= 0: pid 0 means "calling process" to the syscall,
    // pid -1 means "every process in the session," and pid < -1
    // targets a process group. None are valid inputs from within
    // this crate, which only ever stores real worker pids. Mirrors
    // `process_alive` in scenario/mod.rs.
    if pid <= 0 {
        anyhow::bail!("sched_setscheduler: invalid pid {pid} (must be > 0)");
    }
    let (pol, prio) = match policy {
        SchedPolicy::Normal => return Ok(()),
        SchedPolicy::Batch => (libc::SCHED_BATCH, 0),
        SchedPolicy::Idle => (libc::SCHED_IDLE, 0),
        SchedPolicy::Fifo(p) => (libc::SCHED_FIFO, p.clamp(1, 99) as i32),
        SchedPolicy::RoundRobin(p) => (libc::SCHED_RR, p.clamp(1, 99) as i32),
        SchedPolicy::Deadline {
            runtime,
            deadline,
            period,
        } => {
            // SCHED_DEADLINE has no `sched_param` representation —
            // the kernel only accepts it through `sched_setattr(2)`.
            // glibc does not wrap that syscall, so we issue it
            // directly via `syscall(SYS_sched_setattr, ...)`.
            //
            // `__checkparam_dl` (kernel/sched/deadline.c) rejects
            // anything that violates `sched_deadline != 0`,
            // `runtime >= 1024 ns`, the bit-63-clear requirement on
            // `deadline`/`period`, the `runtime <= deadline <=
            // effective_period` ordering (where `effective_period`
            // is `sched_deadline` when `sched_period == 0`), and
            // the sysctl-controlled period bounds. The sysctl
            // values are runtime-tunable via
            // `/proc/sys/kernel/sched_deadline_period_{min,max}_us`,
            // so this pre-validation only mirrors the structural
            // checks (zero-deadline, ordering, top-bit, DL_SCALE
            // floor) — the sysctl bound check happens kernel-side
            // and surfaces as a syscall EINVAL.
            //
            // The Duration → u64 ns conversions ALSO enforce the
            // kernel's bit-63-clear constraint as a single
            // i64::MAX overflow check in `duration_to_kernel_ns`:
            // `Duration::as_nanos()` returns `u128`, and a value
            // exceeding `i64::MAX` would either flip bit 63 of the
            // truncated u64 (kernel reserved) or wrap on the cast
            // entirely. Doing the conversion here keeps the
            // top-bit check and the syscall arg in lockstep.
            if deadline.is_zero() {
                anyhow::bail!(
                    "sched_setattr: deadline must be > 0 (kernel `__checkparam_dl` rejects zero deadline)"
                );
            }
            let runtime_ns = duration_to_kernel_ns(runtime, "runtime")?;
            let deadline_ns = duration_to_kernel_ns(deadline, "deadline")?;
            let period_ns = duration_to_kernel_ns(period, "period")?;
            if runtime_ns < 1024 {
                anyhow::bail!(
                    "sched_setattr: runtime ({runtime_ns} ns) below kernel DL_SCALE floor (1024 ns)"
                );
            }
            if runtime_ns > deadline_ns {
                anyhow::bail!(
                    "sched_setattr: runtime ({runtime_ns} ns) > deadline ({deadline_ns} ns)"
                );
            }
            // `period == Duration::ZERO` is legal: the kernel
            // substitutes `sched_deadline` for the period in that
            // case (see `if (!period) period = attr->sched_deadline;`
            // in `__checkparam_dl`). Only enforce `deadline <=
            // period` when period is non-zero.
            if period_ns != 0 && deadline_ns > period_ns {
                anyhow::bail!(
                    "sched_setattr: deadline ({deadline_ns} ns) > period ({period_ns} ns)"
                );
            }
            // SAFETY: `sched_attr` is a UAPI struct of plain
            // integer fields (no padding bytes affect kernel
            // behavior; the kernel reads `size` and treats unknown
            // tail as zero). Zero-initializing is the canonical
            // way to construct it because libc's `s!` macro
            // derives only `Clone, Copy, Debug` — no `Default`.
            let mut attr: libc::sched_attr = unsafe { std::mem::zeroed() };
            attr.size = std::mem::size_of::<libc::sched_attr>() as u32;
            attr.sched_policy = libc::SCHED_DEADLINE as u32;
            attr.sched_runtime = runtime_ns;
            attr.sched_deadline = deadline_ns;
            attr.sched_period = period_ns;
            // sched_setattr(pid_t pid, struct sched_attr *attr,
            //               unsigned int flags). flags=0 — the
            // kernel reserves them for future use.
            //
            // SAFETY:
            // - `pid` is validated > 0 at the top of
            //   `set_sched_policy`, so the kernel cannot interpret
            //   it as the broadcast / process-group target encoded
            //   by 0 / negative pid_t values.
            // - `&attr` is a borrow of a stack local that lives
            //   for the entire syscall — we do not move or drop
            //   `attr` between the borrow and the syscall return.
            //   `libc::sched_attr` is `#[repr(C)]` (UAPI) and was
            //   zeroed via `std::mem::zeroed()` then field-
            //   initialized, so the bytes the kernel reads are
            //   either the values explicitly set above or zero
            //   (the kernel-defined unset value for every
            //   remaining field).
            // - `attr.size` is the actual `size_of::<libc::sched_attr>()`
            //   the kernel ABI expects for `sched_setattr(2)`'s
            //   forward-compat protocol: the kernel uses `size`
            //   to gate which fields it reads and ignores tail
            //   bytes beyond its own struct definition. Sending
            //   our struct's size and zeroing the body cleanly
            //   covers older AND newer kernels.
            // - `flags = 0u32` is the only currently-defined
            //   value; the kernel rejects unknown flag bits with
            //   EINVAL.
            // - The kernel copies `attr` into kernel space inside
            //   the syscall (`copy_struct_from_user` in
            //   kernel/sched/syscalls.c) and does not retain a
            //   reference to our stack memory after the syscall
            //   returns, so the borrow only needs to outlive the
            //   single syscall.
            let ret = unsafe {
                libc::syscall(
                    libc::SYS_sched_setattr,
                    pid,
                    &attr as *const libc::sched_attr,
                    0u32,
                )
            };
            if ret != 0 {
                anyhow::bail!("sched_setattr: {}", std::io::Error::last_os_error());
            }
            return Ok(());
        }
    };
    let param = libc::sched_param {
        sched_priority: prio,
    };
    if unsafe { libc::sched_setscheduler(pid, pol, &param) } != 0 {
        anyhow::bail!("sched_setscheduler: {}", std::io::Error::last_os_error());
    }
    Ok(())
}
