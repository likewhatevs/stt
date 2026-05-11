//! Spawn pipeline: `WorkloadHandle`, `SpawnGuard`, `GroupParams`,
//! `ThreadWorker`, the report shapes (`WorkerReport`, `WorkerExitInfo`,
//! `Migration`), and the helpers that thread workers through fork or
//! `std::thread::spawn`. Split out of `workload/mod.rs` to keep the
//! production code path under 3500 lines per file. Tests are
//! co-located with the production code in topic-grouped sibling
//! files (`tests_lifecycle`, `tests_grandchild`, `tests_composed`,
//! ...) that import shared fixtures from `testing.rs` via
//! `use super::testing::*;`.

use anyhow::{Context, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::os::unix::io::FromRawFd;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use super::affinity::{AffinityIntent, ResolvedAffinity, resolve_affinity, set_thread_affinity};
use super::config::{CloneMode, MemPolicy, MpolFlags, SchedPolicy, WorkSpec, WorkloadConfig};
use super::types::*;
use super::worker::worker_main;

pub(super) static STOP: AtomicBool = AtomicBool::new(false);

/// A single CPU migration event observed by a worker.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Migration {
    /// Nanoseconds since worker start.
    pub at_ns: u64,
    /// CPU before migration.
    pub from_cpu: usize,
    /// CPU after migration.
    pub to_cpu: usize,
}

/// Build a nodemask bitmask and maxnode value for `set_mempolicy(2)`
/// and `mbind(2)`.
///
/// Returns `(nodemask_vec, maxnode)`. The nodemask is a bitmask of
/// `c_ulong` words where bit N corresponds to NUMA node N. `maxnode`
/// must be `max_node + 2` because the kernel's `get_nodes()` does
/// `--maxnode` before reading the bitmask.
pub fn build_nodemask(nodes: &BTreeSet<usize>) -> (Vec<libc::c_ulong>, libc::c_ulong) {
    if nodes.is_empty() {
        return (vec![], 0);
    }
    let max_node = nodes.iter().copied().max().unwrap_or(0);
    let mask_bits = max_node + 2;
    let bits_per_word = std::mem::size_of::<libc::c_ulong>() * 8;
    let mask_words = mask_bits.div_ceil(bits_per_word);
    let mut nodemask = vec![0 as libc::c_ulong; mask_words];
    for &node in nodes {
        nodemask[node / bits_per_word] |= 1 << (node % bits_per_word);
    }
    (nodemask, mask_bits as libc::c_ulong)
}

const MPOL_PREFERRED_MANY: i32 = 5;
const MPOL_WEIGHTED_INTERLEAVE: i32 = 6;

/// Worker-side `futex_wait` timeout for STOP-signal polling across
/// every blocking workload primitive (FutexPingPong, FutexFanOut,
/// FanOutCompute, MutexContention). Workers block inside the
/// per-variant futex with this timespec; on wake (or timeout) they
/// re-check [`STOP`] and either continue working or exit cleanly.
/// At 100ms the worst-case shutdown latency a `stop_and_collect`
/// caller must budget for is ~100ms above the flush/IO cost; see
/// [`WorkloadHandle::stop_and_collect`]'s "Shutdown latency"
/// paragraph for the caller-facing contract.
pub(super) const WORKER_STOP_POLL_NS: libc::c_long = 100_000_000;

/// Packaged [`libc::timespec`] for every worker-side `futex_wait`
/// across the blocking workload primitives. Duplicating the struct
/// literal per call site drifted the `tv_nsec` field between variants
/// during earlier edits; a single const keeps the shutdown-latency
/// budget documented on [`WORKER_STOP_POLL_NS`] authoritative.
pub(super) const FUTEX_WAIT_TIMEOUT: libc::timespec = libc::timespec {
    tv_sec: 0,
    tv_nsec: WORKER_STOP_POLL_NS,
};

/// Post-wake spin count used by the fan-out messenger variants
/// ([`WorkType::FutexFanOut`] and [`WorkType::FanOutCompute`]) AFTER
/// each broadcast wake. Gives receivers a short uncontended window
/// to run to their reservoir-push before the next wake cycle
/// arrives. Threaded through [`spin_burst`] rather than a raw
/// `std::hint::spin_loop` so the messenger also contributes to
/// `work_units` — matching FanOutCompute's existing pattern so
/// both variants' messengers report comparable throughput to
/// downstream assertions.
pub(super) const FAN_OUT_POST_WAKE_SPIN_ITERS: u64 = 256;

/// Call `set_mempolicy(2)` for the current process with mode flags.
///
/// No-op for `MemPolicy::Default`. Logs a warning on syscall failure.
pub(super) fn apply_mempolicy_with_flags(policy: &MemPolicy, flags: MpolFlags) {
    let (mode, node_set): (i32, BTreeSet<usize>) = match policy {
        MemPolicy::Default => return,
        MemPolicy::Bind(nodes) => (libc::MPOL_BIND, nodes.clone()),
        MemPolicy::Preferred(node) => (libc::MPOL_PREFERRED, [*node].into_iter().collect()),
        MemPolicy::Interleave(nodes) => (libc::MPOL_INTERLEAVE, nodes.clone()),
        MemPolicy::PreferredMany(nodes) => (MPOL_PREFERRED_MANY, nodes.clone()),
        MemPolicy::WeightedInterleave(nodes) => (MPOL_WEIGHTED_INTERLEAVE, nodes.clone()),
        MemPolicy::Local => {
            let rc = unsafe {
                libc::syscall(
                    libc::SYS_set_mempolicy,
                    libc::MPOL_LOCAL | flags.bits() as i32,
                    std::ptr::null::<libc::c_ulong>(),
                    0 as libc::c_ulong,
                )
            };
            if rc != 0 {
                eprintln!(
                    "ktstr: set_mempolicy(MPOL_LOCAL) failed: {}",
                    std::io::Error::last_os_error(),
                );
            }
            return;
        }
    };
    if node_set.is_empty() {
        eprintln!("ktstr: set_mempolicy: empty node set, skipping");
        return;
    }
    let (mask, maxnode) = build_nodemask(&node_set);
    let effective_mode = mode | flags.bits() as i32;
    let rc = unsafe {
        libc::syscall(
            libc::SYS_set_mempolicy,
            effective_mode,
            mask.as_ptr(),
            maxnode,
        )
    };
    if rc != 0 {
        eprintln!(
            "ktstr: set_mempolicy(mode={}, nodes={:?}) failed: {}",
            mode,
            node_set,
            std::io::Error::last_os_error(),
        );
    }
}

/// Apply `nice` to the calling worker via `setpriority(2)`.
///
/// Always invokes `setpriority(PRIO_PROCESS, 0, nice)` — including
/// for `nice == 0`, which writes 0 explicitly rather than
/// inheriting. The "skip the syscall, inherit the parent's nice"
/// state lives one layer up: callers gate the call on
/// [`WorkSpec::nice`](crate::workload::WorkSpec::nice) /
/// [`WorkloadConfig::nice`](crate::workload::WorkloadConfig::nice)
/// being `Some(_)` and pass through the inner value (so `Some(0)`
/// becomes a real `setpriority(0)` and `None` skips the call
/// entirely). The kernel clamps `niceval` to
/// `[MIN_NICE, MAX_NICE]` (-20..19) inside `setpriority`, so any
/// out-of-range input is normalised by the syscall itself rather
/// than rejected.
///
/// Failures are logged once via stderr and do not abort the
/// worker — matches the [`apply_mempolicy_with_flags`] /
/// [`set_thread_affinity`] / [`set_sched_policy`] error idiom in
/// `worker_main`. The expected failure mode is `EACCES` from
/// `set_one_prio` → `can_nice` when an unprivileged worker tries
/// to lower nice (negative niceval) without `CAP_SYS_NICE`.
pub(super) fn apply_nice(nice: i32) {
    let rc = unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, nice) };
    if rc != 0 {
        warn_setpriority_failed_once();
    }
}

/// Print a single `setpriority` failure warning for the lifetime
/// of the process. Same rationale as
/// `warn_schedstat_unavailable_once`: dozens of workers will fail
/// once each on an unprivileged host that requested negative nice,
/// and a per-worker line floods the test log.
pub(super) fn warn_setpriority_failed_once() {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        let errno = std::io::Error::last_os_error();
        eprintln!(
            "workload: setpriority(PRIO_PROCESS) failed: {errno}; nice value not applied (CAP_SYS_NICE may be required for negative nice)"
        );
    });
}

/// Telemetry collected from a worker process after it stops.
///
/// Normal reports: each field is populated by the worker itself
/// (inside the VM) and serialized via a pipe to the parent process.
/// Sentinel reports: sentinel reports synthesized by
/// [`WorkloadHandle::stop_and_collect`] on worker-exit carry
/// parent-populated `exit_info` with the remaining fields at their
/// [`Default`] values (the worker never emitted on the pipe, so
/// the parent is the sole source of truth for the surfaced
/// outcome).
///
/// # Default trade-off
///
/// [`Default`] produces a zero/empty report. The trade-off:
///
/// - **Pro:** sentinel/test code can spread `..WorkerReport::default()`
///   so adding a field does not require touching every sentinel site.
/// - **Con:** zero-valued fields are valid report outputs (e.g. a
///   worker that never blocked has `resume_latencies_ns: vec![]`), so
///   a missing field cannot be distinguished from a real-zero field at
///   the reader. Consumers that need "was this field actually set"
///   must track presence out-of-band (e.g. whether the work type
///   populates the field per [`resume_latencies_ns`]'s doc).
///
/// Decision: keep the `Default` impl. Sentinel ergonomics outweigh
/// the distinguishability cost — every real consumer already knows
/// which fields a given `WorkType` populates, and the alternative
/// (removing `Default` and hand-listing every field at sentinel
/// sites) introduces a worse drift problem that silently skips new
/// telemetry instead of reporting it as zero.
///
/// Callers building a sentinel report should spread
/// `..WorkerReport::default()` rather than listing every field by hand
/// -- the sentinel drifts silently when a field is added.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, crate::Claim)]
pub struct WorkerReport {
    /// Kernel TID from `gettid(2)`. For [`CloneMode::Fork`] each
    /// worker is its own thread-group leader so `gettid() == getpid()
    /// == tgid`; the report's tid is interchangeable with the
    /// worker's pid in libc / cgroup APIs. For [`CloneMode::Thread`]
    /// every worker shares the parent's tgid and `gettid()` is the
    /// only identifier that discriminates per-task identity, so the
    /// report's tid is what feeds `sched_setaffinity(tid, ...)` and
    /// `cgroup.threads` writes (NOT `cgroup.procs` — see the warning
    /// on [`WorkloadHandle::worker_pids`]). Stored as `pid_t` (i32)
    /// to match the kernel's native type and avoid the silent
    /// u32→i32 sign-cast wraparound at libc boundaries
    /// (kill/waitpid/Pid::from_raw).
    pub tid: i32,
    /// Cumulative work iterations (incremented by `spin_burst` or I/O loops).
    pub work_units: u64,
    /// Thread CPU time from `CLOCK_THREAD_CPUTIME_ID` (ns).
    pub cpu_time_ns: u64,
    /// Wall-clock time from worker-start to stop flag (ns).
    /// Measured from the worker's first `Instant::now()` in
    /// `worker_main` (immediately after the start handshake) to the
    /// outer-loop exit (when the per-worker `stop` flag is observed
    /// `true`); covers both Fork-mode workers (signal-driven flag)
    /// and Thread-mode workers (parent-driven flag).
    pub wall_time_ns: u64,
    /// `wall_time_ns - cpu_time_ns`: total off-CPU time (ns).
    ///
    /// Includes all time the worker was not executing on a CPU: runnable
    /// queue wait, voluntary sleep, I/O wait, futex wait, etc.
    pub off_cpu_ns: u64,
    /// Number of observed CPU migrations (checked every 1024 work units).
    pub migration_count: u64,
    /// Set of all CPUs this worker ran on.
    pub cpus_used: BTreeSet<usize>,
    /// Ordered list of CPU migration events with timestamps.
    pub migrations: Vec<Migration>,
    /// Longest wall-clock gap observed at 1024-work-unit checkpoints
    /// (ms). High values indicate the task was preempted or descheduled
    /// near a checkpoint boundary.
    pub max_gap_ms: u64,
    /// CPU where the longest gap happened.
    pub max_gap_cpu: usize,
    /// When the longest gap happened (ms from start).
    pub max_gap_at_ms: u64,
    /// Per-wakeup latency samples (ns). Measures off-CPU time
    /// between the call that blocks (any blocking primitive — pipe
    /// `read`, futex wait, `poll`, `sched_yield`, `nanosleep`, etc.)
    /// and the wakeup that resumes execution; not a yield-specific
    /// measure.
    /// Populated for blocking work types: Bursty, PipeIo, FutexPingPong,
    /// FutexFanOut, FanOutCompute, CacheYield, CachePipe, IoSyncWrite,
    /// IoRandRead, IoConvoy, NiceSweep,
    /// AffinityChurn, PolicyChurn, MutexContention, ForkExit (parent's
    /// waitpid wait), Sequence with Sleep/Yield/Io phases.
    ///
    /// Distinct from [`iteration_costs_ns`](Self::iteration_costs_ns):
    /// this field measures the OFF-CPU gap between blocks (scheduler
    /// resume latency); `iteration_costs_ns` measures the wall-clock
    /// duration of a single compute iteration. The three pure-compute
    /// variants that populate `iteration_costs_ns` —
    /// [`WorkType::AluHot`], [`WorkType::SmtSiblingSpin`], and
    /// [`WorkType::IpcVariance`] — never block and report
    /// `resume_latencies_ns: vec![]`. Other compute variants
    /// (e.g. SpinWait, YieldHeavy, Mixed) populate neither
    /// reservoir.
    pub resume_latencies_ns: Vec<u64>,
    /// Total number of wake-latency observations the worker
    /// recorded, INCLUDING any that were dropped by the reservoir
    /// sampler. `resume_latencies_ns` is reservoir-clamped to at
    /// most `MAX_WAKE_SAMPLES` (100_000) entries; on a long run
    /// that accumulates more than that many wake events, the
    /// vector stays at its cap while this counter keeps climbing.
    /// Host-side consumers that want to report "total wakeups
    /// observed" (vs. "entries in the sample") read this field;
    /// percentile / CV computations read `resume_latencies_ns`.
    pub wake_sample_total: u64,
    /// Per-iteration wall-clock duration of one compute iteration (ns),
    /// including any scheduler preemption. Measured via
    /// `Instant::now()` (CLOCK_MONOTONIC), so a sample includes any
    /// off-CPU time the kernel inserted mid-iteration. The variance
    /// across iterations is the load-bearing scheduler signal —
    /// preemption inflates samples and that inflation is the
    /// observable.
    ///
    /// Reservoir-sampled at the same cap (`MAX_WAKE_SAMPLES` =
    /// 100_000) as [`resume_latencies_ns`](Self::resume_latencies_ns),
    /// using the same Algorithm-R sampler.
    ///
    /// Populated for pure compute work types where the worker
    /// never blocks: [`WorkType::AluHot`], [`WorkType::SmtSiblingSpin`],
    /// and [`WorkType::IpcVariance`]. Each sample is the elapsed
    /// time from the start to the end of one outer-loop iteration's
    /// compute burst.
    ///
    /// Distinct from [`resume_latencies_ns`](Self::resume_latencies_ns):
    /// the wake-latency reservoir captures off-CPU time (futex /
    /// pipe / nanosleep wakeups); this reservoir captures the
    /// wall-clock duration of one compute iteration (which
    /// includes any scheduler preemption inside the iteration).
    /// The two are NOT comparable across variants — a
    /// scheduler-A/B test that wants iteration cost for a compute
    /// variant reads this field; a test that wants wake latency
    /// for a blocking variant reads `resume_latencies_ns`.
    pub iteration_costs_ns: Vec<u64>,
    /// Total number of iteration-cost observations the worker
    /// recorded, INCLUDING any that were dropped by the reservoir
    /// sampler. Mirrors [`wake_sample_total`](Self::wake_sample_total)
    /// but for [`iteration_costs_ns`](Self::iteration_costs_ns):
    /// host-side consumers that want "total compute iterations
    /// observed" read this field; distribution computations read
    /// `iteration_costs_ns` directly.
    pub iteration_cost_sample_total: u64,
    /// Outer-loop iteration count.
    pub iterations: u64,
    /// Delta of /proc/self/schedstat field 2 (run_delay) over the work loop.
    pub schedstat_run_delay_ns: u64,
    /// Delta of /proc/self/schedstat field 3 (pcount — number of
    /// times the task was scheduled in over the work loop). This is
    /// NOT a context-switch count; /proc/<pid>/status's
    /// `voluntary_ctxt_switches` / `nonvoluntary_ctxt_switches` are
    /// the true context-switch counters and are not read here.
    pub schedstat_run_count: u64,
    /// Delta of /proc/self/schedstat field 1 (cpu_time) over the work loop.
    pub schedstat_cpu_time_ns: u64,
    /// `true` when the worker reached its natural end — either the
    /// outer work loop observed STOP and exited cleanly, or a
    /// custom-closure payload returned from its `run` function. A
    /// sentinel report synthesised by
    /// [`WorkloadHandle::stop_and_collect`]'s decode-failure
    /// fallback (see `exit_info` below) carries `false`. Lets downstream
    /// consumers distinguish "worker ran to completion and
    /// observed zero iterations" (`completed: true, iterations: 0`
    /// — legitimate for pathologically short test windows) from
    /// "worker died / timed out before recording anything"
    /// (`completed: false, iterations: 0` — the sentinel shape).
    #[serde(default)]
    pub completed: bool,
    /// Per-NUMA-node page counts from `/proc/self/numa_maps` after workload.
    /// Keyed by node ID. Empty when numa_maps is unavailable.
    #[serde(default)]
    pub numa_pages: BTreeMap<usize, u64>,
    /// Delta of `/proc/vmstat` `numa_pages_migrated` over the work loop.
    pub vmstat_numa_pages_migrated: u64,
    /// Diagnostic attached only to sentinel reports — populated when
    /// `stop_and_collect` synthesized the entry because no (or
    /// unparseable) bincode payload came back on the report pipe.
    /// `None` on every real worker-produced report. Lets operators
    /// distinguish the four failure shapes that all collapse to
    /// "empty pipe + no report":
    ///
    /// - [`WorkerExitInfo::Exited`] with a non-zero code: worker
    ///   reached `_exit(code)` without writing the report —
    ///   typically the `catch_unwind` Err arm in the worker-child
    ///   closure (panic under `panic = "unwind"`) or the 30s
    ///   poll-start timeout's early `_exit(1)`.
    /// - [`WorkerExitInfo::Signaled`]: worker was killed — SIGABRT
    ///   under `panic = "abort"`, SIGKILL from the still-alive
    ///   escalation in `stop_and_collect`, or an external signal
    ///   (OOM killer, operator SIGKILL).
    /// - [`WorkerExitInfo::TimedOut`]: worker never exited within the
    ///   5s collection deadline and the WNOHANG reap observed
    ///   `StillAlive` — escalated via SIGKILL + `waitpid(None)`.
    /// - [`WorkerExitInfo::WaitFailed`]: `waitpid` itself returned an
    ///   error (ECHILD / EINTR). Typically a plumbing bug — the child
    ///   was reaped by an external signal handler, a double-reap
    ///   regression, or the pid was recycled.
    ///
    /// No `skip_serializing_if`: bincode is a positional, schemaless
    /// format — every Serialize call must emit every field in the
    /// same order or the decoder reads the next field's bytes off
    /// the wire (silent data corruption). The Option<…> tag itself
    /// (one byte) is the only overhead on the live-worker path.
    pub exit_info: Option<WorkerExitInfo>,
    /// `true` when this worker served as the messenger for a
    /// wake-fanout work type ([`WorkType::FutexFanOut`] or
    /// [`WorkType::FanOutCompute`]) — the single writer that
    /// advances the shared generation and issues `futex_wake` for
    /// its group. `false` for receivers and for every non-fanout
    /// work type.
    ///
    /// Populated from the `is_messenger` flag on the
    /// `futex: Option<(*mut u32, bool)>` parameter threaded into
    /// `worker_main`. A sentinel report synthesized by the
    /// decode-failure fallback in
    /// [`WorkloadHandle::stop_and_collect`] carries `false` via
    /// [`Default`], matching its `completed: false` shape.
    ///
    /// Enables per-worker latency-participation assertions in
    /// tests — a receiver worker produces `resume_latencies_ns`
    /// entries while its messenger pair records wake-side work but
    /// no resume latency. Without this field, tests had to
    /// cross-reference per-group indexing or guess from the empty
    /// vector — ambiguous on groups where the messenger legitimately
    /// exits before producing a report.
    #[serde(default)]
    pub is_messenger: bool,
    /// Index of the worker group this report belongs to.
    ///
    /// `0` denotes the primary group described by
    /// [`WorkloadConfig`]'s top-level `work_type` / `num_workers` /
    /// `affinity` / `sched_policy` fields. `1..=N` denotes
    /// composed groups in the order they appear in
    /// [`WorkloadConfig::composed`]. Reports collected by
    /// [`WorkloadHandle::stop_and_collect`] are tagged with the
    /// `group_idx` of the spawning [`WorkSpec`] (or `0` for the
    /// primary), so per-group filtering in test assertions can
    /// cleanly partition the vector.
    ///
    /// Sentinel reports (synthesized on missing or undecodable
    /// payload / panic / timeout) carry the `group_idx` of the
    /// worker whose pid the sentinel replaces, so a "this composed
    /// group failed" assertion still works on an outright crash.
    ///
    /// `#[serde(default)]` so reports persisted before `group_idx`
    /// existed (or written by a worker on a non-composed config)
    /// deserialize cleanly with `group_idx == 0` — the primary
    /// group, which is also the only group such reports could
    /// possibly belong to.
    #[serde(default)]
    pub group_idx: usize,
    /// Rendered error from the worker's `set_thread_affinity`
    /// call, or `None` when affinity setup succeeded (or the
    /// worker had no affinity to apply). Populated by
    /// `worker_main` when the pre-loop
    /// `set_thread_affinity(tid, cpus)` returns `Err` — the
    /// worker continues with the inherited (or kernel-default)
    /// cpumask so the test still produces an observable outcome,
    /// but the failure is now surfaced in the report instead of
    /// being silently dropped via `let _ = …`. The expected
    /// failure shape is EINVAL from a requested cpu that is
    /// outside the cpuset cgroup's `cpus.allowed` mask or the
    /// kernel's online mask; EPERM is reachable when a more
    /// privileged tracer set the worker's cpus_allowed and a
    /// container policy denies further widening. Sentinel
    /// reports synthesised by
    /// [`WorkloadHandle::stop_and_collect`] leave this field at
    /// its default `None` — a worker that died before
    /// `worker_main` ran has no affinity-error observation.
    ///
    /// No `skip_serializing_if`: bincode is positional and
    /// schemaless, so every Serialize call must emit every field
    /// in the same order — skipping a field shifts the decoder
    /// onto the next field's bytes (silent corruption). The
    /// Option<…> tag (one byte) is the only overhead on the
    /// success path.
    #[serde(default)]
    pub affinity_error: Option<String>,
}

/// Reason a sentinel [`WorkerReport`] was synthesized — attached to
/// the report's `exit_info` field so operators can triage a missing
/// or undecodable bincode payload without cross-referencing
/// parent-side logs.
///
/// Invariant: every variant carries the `waitpid`-derived status for
/// the worker PID as of the end of `stop_and_collect`. Ordered from
/// most-informative (exit code) to least (plumbing failure).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum WorkerExitInfo {
    /// `WIFEXITED=true` with the given exit code. Non-zero under
    /// `panic = "unwind"` means catch_unwind caught a panic in the
    /// worker-child closure and `_exit(1)` fired, or the 30s
    /// parent-ready poll timed out. Zero means the worker ran to
    /// completion but failed to write / serialize the report — a
    /// bincode encode or pipe-write failure that didn't panic.
    Exited(i32),
    /// `WIFSIGNALED=true` with the given signal number. Under
    /// `panic = "abort"` a worker panic raises SIGABRT (signal 6);
    /// other values indicate external kill, OOM killer, or the
    /// still-alive-escalation SIGKILL (signal 9) from this function.
    Signaled(i32),
    /// Worker was still running after the 5s shared collection
    /// deadline; escalated via SIGKILL + blocking `waitpid`. The
    /// child's final status is not retained — the reap happened past
    /// the point where operator diagnostics would differ between a
    /// clean timeout and a signal storm.
    TimedOut,
    /// `waitpid` itself returned `Err` — typically ECHILD (child
    /// already reaped by an external signal handler or a double-reap
    /// regression) or EINTR. Message is the rendered `errno` string.
    WaitFailed(String),
    /// Thread-mode worker panicked. `JoinHandle::join()` returned
    /// `Err`; the inner payload is downcast to a `&str` / `String`
    /// (the canonical `panic!` payload shapes) and recorded here so
    /// the operator can triage without scraping the test log. This
    /// variant is exclusive to [`CloneMode::Thread`] — fork workers
    /// surface panics via `Exited(1)` or `Signaled(SIGABRT)`
    /// depending on the panic strategy.
    Panicked(String),
}

/// Pure mapping from a `waitpid` outcome to the diagnostic
/// [`WorkerExitInfo`] attached to a sentinel [`WorkerReport`].
///
/// Split out of [`WorkloadHandle::stop_and_collect`] so the four
/// shapes each resolve to a `WorkerExitInfo` variant without pulling
/// in the full collection loop's state (pipe drain, SIGKILL
/// escalation, pid lifetime). Pure input → output means the variant
/// matrix is directly testable without spawning children.
///
/// Shape → variant:
/// - `Ok(Exited(_, code))` → [`WorkerExitInfo::Exited`]
/// - `Ok(Signaled(_, sig, _))` → [`WorkerExitInfo::Signaled`]
/// - `Ok(StillAlive)` → [`WorkerExitInfo::TimedOut`]
/// - `Ok(_ exotic)` → [`WorkerExitInfo::TimedOut`] (Stopped /
///   PtraceEvent / PtraceSyscall / Continued; not reachable for a
///   plain forked worker with no ptrace parent, but collapsed rather
///   than silently dropped so coverage stays exhaustive)
/// - `Err(errno)` → [`WorkerExitInfo::WaitFailed`]
pub(super) fn classify_wait_outcome(
    source: Result<nix::sys::wait::WaitStatus, nix::errno::Errno>,
) -> WorkerExitInfo {
    match source {
        Ok(nix::sys::wait::WaitStatus::Exited(_, code)) => WorkerExitInfo::Exited(code),
        Ok(nix::sys::wait::WaitStatus::Signaled(_, sig, _)) => WorkerExitInfo::Signaled(sig as i32),
        Ok(nix::sys::wait::WaitStatus::StillAlive) => WorkerExitInfo::TimedOut,
        Ok(_) => WorkerExitInfo::TimedOut,
        Err(e) => WorkerExitInfo::WaitFailed(e.to_string()),
    }
}

/// Extract a human-readable panic payload from a
/// [`std::thread::Result`] `Err` value. The two canonical shapes
/// are `&'static str` (`panic!("literal")`) and `String`
/// (`panic!("{x}")` post-formatting); anything else falls back to
/// a fixed sentinel.
///
/// Pure mapping (no IO, no allocation past `String::clone`) so the
/// stop_and_collect path can call it on every joined-and-panicked
/// thread without performance cliffs.
pub(super) fn extract_panic_payload(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

/// Wall-clock time budget for joining a thread-mode worker after
/// its per-task `stop` has been flipped. Mirrors the fork-mode
/// `stop_and_collect` 5s shared deadline so neither dispatch path
/// can serially exhaust the test runtime by hanging on a single
/// stuck worker. The 100ms `FUTEX_WAIT_TIMEOUT` inside
/// `worker_main`'s blocking primitives means a well-behaved worker
/// observes `stop=true` within 100ms of the parent's flip; the 5s
/// budget covers IO drain, scheduling delays under contention, and
/// post-loop cleanup (NUMA stat reads, schedstat snapshots).
pub(super) const THREAD_JOIN_TIMEOUT: Duration = Duration::from_secs(5);

/// Block until `join` reports finished or `timeout` elapses.
/// Returns `Some(thread_result)` on successful join, `None` on
/// timeout.
///
/// Implementation: wait on `exit_evt` (the worker's "I'm about to
/// return" eventfd, bumped from a Drop guard inside the thread
/// closure) via `epoll_wait` with a `timerfd` for the safety
/// deadline. A spurious wake (e.g. EINTR or a stale eventfd-counter
/// drain) loops back into the wait without orphaning the worker —
/// the timerfd carries the absolute deadline.
///
/// Std lacks a native timed-join API; an alternative side-thread
/// "joiner + channel" pattern would orphan the joiner on timeout
/// (joining is non-cancellable in std), which keeps the thread
/// alive past `WorkloadHandle::drop` and prevents process exit.
/// The eventfd path replaces the previous 10ms sleep-poll loop
/// without that orphan cost.
pub(super) fn join_thread_with_timeout(
    join: std::thread::JoinHandle<WorkerReport>,
    exit_evt: &vmm_sys_util::eventfd::EventFd,
    timeout: Duration,
) -> Option<std::thread::Result<WorkerReport>> {
    use std::os::unix::io::AsRawFd;
    use vmm_sys_util::epoll::{ControlOperation, Epoll, EpollEvent, EventSet};
    use vmm_sys_util::timerfd::TimerFd;

    if join.is_finished() {
        return Some(join.join());
    }

    let epoll = match Epoll::new() {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!(%e, "join_thread_with_timeout: epoll_create1 failed");
            return None;
        }
    };
    if let Err(e) = epoll.ctl(
        ControlOperation::Add,
        exit_evt.as_raw_fd(),
        EpollEvent::new(EventSet::IN, 0),
    ) {
        tracing::warn!(%e, "join_thread_with_timeout: add exit_evt to epoll");
        return None;
    }
    let mut timer = match TimerFd::new() {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!(%e, "join_thread_with_timeout: timerfd_create failed");
            return None;
        }
    };
    if let Err(e) = timer.reset(timeout, None) {
        tracing::warn!(%e, "join_thread_with_timeout: timerfd_settime failed");
        return None;
    }
    if let Err(e) = epoll.ctl(
        ControlOperation::Add,
        timer.as_raw_fd(),
        EpollEvent::new(EventSet::IN, 1),
    ) {
        tracing::warn!(%e, "join_thread_with_timeout: add timerfd to epoll");
        return None;
    }

    let deadline = Instant::now() + timeout;
    let mut events = [EpollEvent::default(); 2];
    loop {
        if join.is_finished() {
            return Some(join.join());
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return None;
        }
        match epoll.wait(-1, &mut events) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => {
                tracing::warn!(%e, "join_thread_with_timeout: epoll_wait failed");
                return None;
            }
        }
    }
}

/// `worker_main`'s loop-check predicate: returns `true` when the
/// worker should stop iterating. Reads BOTH the per-worker `stop`
/// flag and the global [`STOP`] flag; either set request causes
/// exit.
///
/// Why both:
/// - Per-worker `stop` is what `WorkloadHandle::stop_and_collect`
///   flips for graceful shutdown. For Fork mode the per-worker
///   `stop` IS the global [`STOP`] (the SIGUSR1 handler flips it).
///   For Thread mode each worker has its own `Arc<AtomicBool>`
///   passed via `&AtomicBool`.
/// - The global [`STOP`] is what the SIGUSR1 handler sets. For
///   Fork mode the worker's per-process [`STOP`] is the same
///   AtomicBool the handler writes. For Thread mode every thread
///   shares the parent process's address space, so a SIGUSR1
///   delivered to the parent (e.g. Ctrl-C / a test harness signal)
///   flips the shared global [`STOP`] but NOT the per-worker
///   `stop` Arcs. Without this disjunction, Thread workers would
///   silently keep running through a parent-level shutdown
///   request.
///
/// `#[inline]` because the call site is two atomic loads + an OR.
/// Relaxed ordering on both reads matches every existing site —
/// no cross-field happens-before edge to establish.
#[inline]
pub(super) fn stop_requested(stop: &AtomicBool) -> bool {
    stop.load(Ordering::Relaxed) || STOP.load(Ordering::Relaxed)
}

/// Per-thread worker state for [`CloneMode::Thread`] dispatch.
///
/// Thread workers cannot be reaped via `waitpid` (they share a tgid
/// with the parent), so the lifecycle uses Rust's [`std::thread`]
/// primitives instead of pid-based syscalls:
///
/// - `tid` is published by the worker thread post-spawn via
///   `gettid()` so the parent can address the kernel task for
///   `sched_setaffinity(tid, ...)` and report it from
///   [`WorkloadHandle::worker_pids`]. `Arc<AtomicI32>` because the
///   thread closure owns the publisher and the parent reads it
///   without joining.
/// - `stop` replaces the global [`STOP`] signal-flag for thread
///   mode: the parent flips it from
///   [`WorkloadHandle::stop_and_collect`], the worker observes it
///   inside `worker_main`'s `stop.load(Relaxed)` checks. SIGUSR1 is
///   process-wide and useless for per-thread stop control.
/// - `start_tx` is the rendezvous channel: the parent calls
///   `send(())` from [`WorkloadHandle::start`]; the thread blocks
///   in `recv()` until then. `Option` so `start` can take it and
///   drop it (idempotent re-call is a no-op when `None`).
/// - `join` holds the [`std::thread::JoinHandle`] returned by
///   `thread::spawn`; `stop_and_collect` joins each handle to
///   retrieve the [`WorkerReport`]. `Option` so `stop_and_collect`
///   can take ownership and `Drop` does not double-join.
pub(super) struct ThreadWorker {
    tid: std::sync::Arc<std::sync::atomic::AtomicI32>,
    stop: std::sync::Arc<AtomicBool>,
    pub(super) start_tx: Option<std::sync::mpsc::SyncSender<()>>,
    join: Option<std::thread::JoinHandle<WorkerReport>>,
    /// Eventfd bumped by the worker thread's `WorkerExitSignal` Drop
    /// guard before the thread returns from its closure. Lets
    /// [`join_thread_with_timeout`] block in `epoll_wait` instead of
    /// sleep-polling [`std::thread::JoinHandle::is_finished`]. Counter
    /// mode (not semaphore) — the value never matters; only the edge
    /// from 0 to non-zero does. The Arc is cloned into the closure
    /// for the Drop guard; the parent retains the original here.
    exit_evt: std::sync::Arc<vmm_sys_util::eventfd::EventFd>,
}

/// Defense-in-depth Drop for [`ThreadWorker`]. Rust's
/// [`std::thread::JoinHandle`] does NOT join its thread on drop —
/// it detaches, and the thread continues running until completion.
/// `WorkloadHandle::drop`, `WorkloadHandle::stop_and_collect`, and
/// `SpawnGuard::drop` already explicitly `take()` the JoinHandle and
/// route it through [`join_thread_with_timeout`]; this impl exists
/// for the case where some future refactor lets a `ThreadWorker`
/// fall out of scope without going through one of those paths.
///
/// Behavior: if `join` is still `Some` when this Drop fires, flip
/// `stop` (so the worker exits cleanly), drop `start_tx` (in case
/// the worker is still parked on `recv()`), and join with the
/// shared 5s budget. Errors / timeouts are swallowed because Drop
/// has nothing to assert against; the upstream paths produce the
/// auditable diagnostics.
impl Drop for ThreadWorker {
    fn drop(&mut self) {
        if let Some(j) = self.join.take() {
            self.stop.store(true, Ordering::Relaxed);
            self.start_tx.take();
            let _ = join_thread_with_timeout(j, &self.exit_evt, THREAD_JOIN_TIMEOUT);
        }
    }
}

/// Per-fork-child report-decoding shape and bookkeeping recorded
/// alongside each entry in [`SpawnGuard::children`] and
/// [`WorkloadHandle::children`].
///
/// Two variants:
///
/// - [`ForkedChildKind::Worker`]: the conventional fork-mode worker
///   (one process per worker, single bincode `WorkerReport`).
///   Carries the worker's `group_idx` so a sentinel report on a
///   missing payload can be tagged correctly.
/// - [`ForkedChildKind::PcommContainer`]: a single thread-group
///   leader that owns `num_workers` worker threads across one or
///   more logical groups. The leader sets its own `comm` to `pcomm`
///   via `prctl(PR_SET_NAME)` before spawning the threads, so every
///   worker thread's `task->group_leader->comm` is `pcomm` (the
///   kernel truncates to `TASK_COMM_LEN - 1 = 15` bytes inside
///   `__set_task_comm`). Reports are a single `serde_json`
///   `Vec<WorkerReport>` (one entry per worker thread; per-thread
///   `group_idx` lives inside each `WorkerReport`).
///
/// Note the per-variant wire-format split: `Worker` uses bincode and
/// `PcommContainer` uses serde_json. The encodings are dispatched by
/// `ForkedChildKind` at decode time on the parent.
#[derive(Clone, Debug)]
pub(super) enum ForkedChildKind {
    /// Conventional fork-mode worker (one process per worker).
    /// Report wire format: bare `b'r'` ready byte followed by a
    /// `bincode::serde::encode_to_vec(&WorkerReport, …)` document.
    Worker { group_idx: usize },
    /// pcomm thread-group leader hosting worker threads across one
    /// or more logical groups. `groups` records the per-group
    /// `(group_idx, num_workers)` layout in the order threads were
    /// spawned: groups[0] contributes the first
    /// `groups[0].1` worker reports, groups[1] the next
    /// `groups[1].1`, and so on. Total expected reports =
    /// `groups.iter().map(|(_, n)| n).sum()`. The layout drives
    /// sentinel distribution: when the JSON payload is missing or
    /// short, the parent emits sentinels tagged with the right
    /// per-group `group_idx` so per-group filters partition
    /// correctly.
    ///
    /// Report wire format: bare `b'r'` ready byte followed by a
    /// `serde_json::to_vec(&Vec<WorkerReport>)` document, one entry
    /// per worker thread (in `(group_idx, within-group index)`
    /// traversal order matching the `groups` layout).
    PcommContainer { groups: Vec<(usize, usize)> },
}

/// Bookkeeping for a single forked-child process owned by the spawn
/// pipeline. Replaces the prior `(pid, report_fd, start_fd)` tuple so
/// per-child decoding metadata ([`ForkedChildKind`]) lives alongside
/// the pipe fds and pid.
#[derive(Clone, Debug)]
pub(super) struct ForkedChild {
    /// Forked child's pid (== tgid for both `Worker` and
    /// `PcommContainer`: the worker is its own thread-group leader,
    /// the container is the thread-group leader of its inner threads).
    pub pid: libc::pid_t,
    /// Parent-side read end of the report pipe. The child writes one
    /// `b'r'` ready byte followed by either a bincode-encoded single
    /// `WorkerReport` (Worker) or a `serde_json` `Vec<WorkerReport>`
    /// (PcommContainer) per the [`ForkedChildKind`] tag.
    pub report_fd: std::os::unix::io::RawFd,
    /// Parent-side write end of the start pipe. Closed by [`Self::start`]
    /// after writing the start byte; set to `-1` thereafter.
    pub start_fd: std::os::unix::io::RawFd,
    /// Per-child decoding shape. See [`ForkedChildKind`].
    pub kind: ForkedChildKind,
}

/// Handle to spawned worker tasks. Workers block until
/// [`start()`](Self::start) is called.
///
/// The [`CloneMode`] in the [`WorkloadConfig`] selects how each
/// worker is created. Within one [`WorkloadHandle`] every worker
/// uses the same mode, so exactly one of `children` or `threads`
/// is populated; the other is empty. This avoids per-worker mode
/// dispatch on the hot path and keeps each vec's per-mode
/// invariants (pid-based vs JoinHandle-based reaping) cohesive.
///
/// - [`CloneMode::Fork`] populates `children` — separate process
///   per worker, reaped via `waitpid`, signaled via SIGUSR1.
/// - [`CloneMode::Thread`] populates `threads` — separate kernel
///   task in the parent's thread group via [`std::thread::spawn`],
///   joined via `JoinHandle`. Workers share the parent's tgid;
///   per-worker cgroup placement requires `cgroup.threads`
///   (cgroup v2 thread mode), which ktstr scenarios do not
///   currently configure — Thread-mode workers inherit the
///   parent's cgroup.
#[must_use = "dropping a WorkloadHandle immediately tears down all worker tasks"]
pub struct WorkloadHandle {
    /// Forked-child processes owned by this handle. Each entry is a
    /// [`ForkedChild`] carrying pid, parent-side pipe fds, and a
    /// [`ForkedChildKind`] tag that drives report decoding (single
    /// bincode `WorkerReport` for `Worker`; `serde_json`
    /// `Vec<WorkerReport>` for `PcommContainer`). Empty when every
    /// group used [`CloneMode::Thread`] without `pcomm`.
    children: Vec<ForkedChild>,
    /// Thread-mode workers. Empty when `clone_mode` is not
    /// [`CloneMode::Thread`].
    threads: Vec<ThreadWorker>,
    started: bool,
    /// Shared mmap regions for futex-based work types (one per worker group). Unmapped on drop.
    futex_ptrs: Vec<*mut u32>,
    /// Per-region byte length, parallel to `futex_ptrs`. Each
    /// region was sized at spawn time to its source group's
    /// natural width (4 for FutexPingPong / FutexFanOut /
    /// MutexContention / etc., 16 for FanOutCompute, 32 + Q*8 for
    /// ProducerConsumerImbalance — see [`futex_region_size_for`]).
    /// `futex_ptrs[i]` and `futex_region_sizes[i]` describe the
    /// same region; both are consumed pairwise on `Drop` so each
    /// `munmap` call receives the matching length.
    futex_region_sizes: Vec<usize>,
    /// MAP_SHARED region of per-worker iteration counters. Workers
    /// atomically store their iteration count; parent reads via
    /// `snapshot_iterations()`. Pointer to the first element; length
    /// is the active worker collection's len. Typed as
    /// `*mut AtomicU64` rather than `*mut u64` so the 8-byte
    /// alignment guarantee (inherited from the page-aligned
    /// iter_counters mmap site in `WorkloadHandle::spawn`) and the
    /// atomic-only-access invariant are encoded in the type system
    /// instead of prose. `AtomicU64` is layout-compatible with `u64`:
    /// `std::mem::size_of::<AtomicU64>() == std::mem::align_of::<AtomicU64>() == 8`
    /// on every supported target, so casting the `*mut c_void`
    /// returned by `mmap` to `*mut AtomicU64` is sound.
    iter_counters: *mut AtomicU64,
    /// Number of AtomicU64 slots in iter_counters (== num_workers at spawn time).
    iter_counter_len: usize,
    /// Inter-worker paired pipes `(ab, ba)` for PipeIo / CachePipe.
    /// Transferred from [`SpawnGuard`] on success; closed by
    /// [`WorkloadHandle::drop`] AFTER worker shutdown so Thread-mode
    /// workers (which share the parent's fd table) can finish their
    /// pipe ops before the close. Under Fork mode each child holds
    /// its own fd-table copy via `fork()`, so the parent's late
    /// close is a no-op for the children. Empty when `work_type`
    /// is neither PipeIo nor CachePipe.
    pipe_pairs: Vec<([i32; 2], [i32; 2])>,
    /// Per-chain pipe rings for `WakeChain { wake: WakeMechanism::Pipe }`.
    /// Outer Vec is one entry per chain (= `num_workers / depth`);
    /// inner Vec is `depth` pipes per chain. Same ownership rule as
    /// `pipe_pairs`: transferred from [`SpawnGuard`] on success,
    /// closed by [`WorkloadHandle::drop`] AFTER worker shutdown so
    /// Thread-mode chain workers don't observe `EBADF` mid-run.
    /// Empty when `work_type` is not `WakeChain { wake: Pipe }`.
    chain_pipes: Vec<Vec<[i32; 2]>>,
}

/// Per-variant byte length for the MAP_SHARED futex region.
///
/// Each WorkType that needs a shared region has a fixed natural
/// size:
///
/// - [`WorkType::FanOutCompute`] needs 16 bytes — futex `u32` at
///   offset 0, wake-timestamp `u64` at offset 8.
/// - [`WorkType::ProducerConsumerImbalance`] needs a ring buffer:
///   reserve-head `u64` @ 0, tail `u64` @ 8, producer-wake `u32`
///   @ 16, consumer-wake `u32` @ 20, publish-head `u64` @ 24,
///   then `Q` × `u64` ring slots starting at offset 32. Total
///   bytes = `32 + Q*8`. The two head counters split the MPMC
///   protocol: producers CAS-advance the reserve head to claim a
///   unique slot index, write the slot, then in slot-index FIFO
///   order release-store the publish head; consumers
///   acquire-load the publish head so a slot is visible to a
///   reader only after its producer's data write
///   synchronizes-with the consumer through the publish-head
///   release. `queue_depth_target` is `u64` to match the variant;
///   an `as usize` truncation on a 32-bit host could silently
///   produce a sub-page region with a malformed queue, so the
///   conversion is clamped at `usize::MAX/8 - 4` to keep the
///   layout well-defined (one fewer slot than before to make
///   room for `pub_head`). Realistic configs use Q in the
///   hundreds-to-thousands; the clamp only triggers on a
///   degenerate input that itself fails admission control
///   elsewhere (the queue is far larger than RAM).
/// - Everything else: `u32` (4 bytes).
///
/// Returning the same byte count for every WorkType variant lets
/// the caller mmap exactly what's needed for THIS group rather
/// than the MAX across all groups, so a small-variant group
/// composed alongside a large `ProducerConsumerImbalance` no
/// longer pays the large group's per-region overhead.
pub(super) fn futex_region_size_for(work_type: &WorkType) -> usize {
    match work_type {
        WorkType::FanOutCompute { .. } => 16,
        WorkType::ProducerConsumerImbalance {
            queue_depth_target, ..
        } => {
            let q = std::cmp::min(*queue_depth_target as usize, usize::MAX / 8 - 4);
            32 + q * 8
        }
        _ => std::mem::size_of::<u32>(),
    }
}

/// Scope guard that owns every resource acquired during
/// [`WorkloadHandle::spawn`]'s partial setup. If `spawn` returns
/// early (via `?` or `bail!`), the guard's `Drop` kills and reaps any
/// already-forked children, closes every open pipe fd, and munmaps
/// every shared region — so a mid-setup failure never leaks fds,
/// zombie processes, or anonymous-shared pages.
///
/// On success, [`SpawnGuard::into_handle`] moves every live
/// resource — children/threads, futex regions, iter-counter
/// region, AND `pipe_pairs` / `chain_pipes` — into the returned
/// [`WorkloadHandle`]. The guard's subsequent `Drop` runs against
/// empty Vecs/null pointers and is a no-op on the success path.
/// On the early-bail path (an `?` inside `WorkloadHandle::spawn`)
/// the guard still owns whatever it allocated and `Drop` cleans
/// it all up — fds, processes, threads, mmaps. Pipe fds are
/// closed by the handle (not the guard) because Thread-mode
/// workers share the parent's fd table; closing the fds before
/// worker shutdown would surface as `EBADF` on every pipe op a
/// thread runs after spawn returns.
pub(super) struct SpawnGuard {
    /// Inter-worker paired pipes `(ab, ba)` for PipeIo/CachePipe.
    /// Transferred to [`WorkloadHandle`] on success; closed by the
    /// guard only on the early-bail path. Under Fork mode each
    /// child holds its own fd-table copy via `fork()`; under
    /// Thread mode every worker thread shares these fds with the
    /// parent.
    pipe_pairs: Vec<([i32; 2], [i32; 2])>,
    /// Per-chain pipe rings for `WakeChain { wake: WakeMechanism::Pipe }`. Outer
    /// Vec is one entry per chain (= `num_workers / depth`); inner
    /// Vec is `depth` pipes per chain. Pipe `i` connects stage `i`
    /// (writer) to stage `(i + 1) % depth` (reader). Same ownership
    /// shape as `pipe_pairs`: transferred to the handle on success,
    /// closed by the guard only on the early-bail path.
    chain_pipes: Vec<Vec<[i32; 2]>>,
    /// Shared-memory futex regions (transferred to handle on success).
    futex_ptrs: Vec<*mut u32>,
    /// Per-region byte length, parallel to `futex_ptrs`. Each
    /// region is sized to its source group's natural width
    /// (4 / 16 / 32+Q*8 — see [`futex_region_size_for`]) and
    /// recorded here at `spawn_group` time so munmap on Drop
    /// can call `libc::munmap(ptr, len)` with the matching length
    /// even when groups with different natural sizes co-exist.
    /// `futex_ptrs[i]` and `futex_region_sizes[i]` describe the
    /// same region.
    futex_region_sizes: Vec<usize>,
    /// Per-worker iteration counter region (transferred on success).
    /// Typed matches the handle field; see `WorkloadHandle::iter_counters`.
    iter_counters: *mut AtomicU64,
    iter_counter_bytes: usize,
    /// Already-forked children (either conventional workers or
    /// pcomm containers) with their parent-side pipe fds and
    /// per-child decoding shape (transferred to handle on success).
    children: Vec<ForkedChild>,
    /// Already-spawned thread workers (transferred on success).
    /// Cleanup on early-exit flips each `stop` and joins each
    /// thread, since threads share the parent's address space and
    /// must be drained cooperatively (no `kill` equivalent).
    threads: Vec<ThreadWorker>,
}

impl SpawnGuard {
    fn new() -> Self {
        Self {
            pipe_pairs: Vec::new(),
            chain_pipes: Vec::new(),
            futex_ptrs: Vec::new(),
            futex_region_sizes: Vec::new(),
            iter_counters: std::ptr::null_mut(),
            iter_counter_bytes: 0,
            children: Vec::new(),
            threads: Vec::new(),
        }
    }

    /// Transfer live resources into a [`WorkloadHandle`]. Leaves the
    /// guard's `children`, `threads`, `futex_ptrs`,
    /// `futex_region_sizes`, `iter_counters`, `pipe_pairs`, and
    /// `chain_pipes` empty, so the guard's subsequent `Drop` is a
    /// no-op on the success path. The handle is now the sole owner
    /// of every resource — its own `Drop` closes the pipe fds
    /// AFTER worker shutdown completes, which is the ordering
    /// Thread mode requires (workers share the parent's fd table;
    /// closing pre-shutdown would surface as `EBADF` on every
    /// worker's pipe op). Fork mode is unaffected either way: each
    /// child holds its own fd-table copy via `fork()`, so the
    /// parent's close timing is invisible to the child.
    fn into_handle(mut self) -> WorkloadHandle {
        let children = std::mem::take(&mut self.children);
        let threads = std::mem::take(&mut self.threads);
        let futex_ptrs = std::mem::take(&mut self.futex_ptrs);
        let futex_region_sizes = std::mem::take(&mut self.futex_region_sizes);
        let iter_counters = std::mem::replace(&mut self.iter_counters, std::ptr::null_mut());
        let iter_counter_bytes = std::mem::replace(&mut self.iter_counter_bytes, 0);
        let iter_counter_len = iter_counter_bytes / std::mem::size_of::<AtomicU64>();
        let pipe_pairs = std::mem::take(&mut self.pipe_pairs);
        let chain_pipes = std::mem::take(&mut self.chain_pipes);
        WorkloadHandle {
            children,
            threads,
            started: false,
            futex_ptrs,
            futex_region_sizes,
            iter_counters,
            iter_counter_len,
            pipe_pairs,
            chain_pipes,
        }
    }
}

impl Drop for SpawnGuard {
    fn drop(&mut self) {
        // Kill and reap any already-forked children first, so their
        // pipe ends are not left blocked when we close the parent
        // side. `nix` wrappers replace the raw libc calls — kill
        // returns `Result<()>` (we swallow ECHILD/ESRCH in the
        // already-exited case), waitpid returns `Result<WaitStatus>`
        // (we discard the status in the cleanup path), close returns
        // `Result<()>` (we swallow EBADF for fds an earlier arm may
        // have already closed).
        for child in &self.children {
            let npid = nix::unistd::Pid::from_raw(child.pid);
            let _ = nix::sys::signal::kill(npid, nix::sys::signal::Signal::SIGKILL);
            let _ = nix::sys::wait::waitpid(npid, None);
        }
        // Close each child's parent-side report/start fds.
        for child in &self.children {
            for fd in [child.report_fd, child.start_fd] {
                if fd >= 0 {
                    let _ = nix::unistd::close(fd);
                }
            }
        }
        // Stop and join any partially-spawned threads. Threads
        // share our address space, so `kill` does not reach them
        // and the only safe teardown is "flip stop, drop the start
        // channel (in case worker is still parked on `recv`), then
        // join". Dropping `start_tx` causes `recv` on the worker
        // side to return `Err(Disconnected)`, unblocking a thread
        // that has not yet been signaled. After both signals
        // (stop=true and start_tx dropped), `worker_main`'s outer
        // loop exits at the next `stop.load(Relaxed)` check (max
        // ~100ms latency from the `FUTEX_WAIT_TIMEOUT` poll
        // cadence) and the thread completes. `join` returns the
        // partial `WorkerReport` (or `Err` on panic, which we
        // swallow because mid-spawn cleanup has nothing to assert).
        for tw in &mut self.threads {
            tw.stop.store(true, Ordering::Relaxed);
            // Drop start_tx FIRST so a worker still parked on
            // recv() unblocks via Disconnected.
            tw.start_tx.take();
            if let Some(j) = tw.join.take() {
                // SpawnGuard cleanup uses the same `THREAD_JOIN_TIMEOUT`
                // budget as `stop_and_collect` and `WorkloadHandle::drop`
                // so a stuck worker can't pin mid-spawn error recovery.
                // Errors (panic / timeout) are silently dropped — the
                // mid-spawn path has nothing to assert against beyond
                // not leaking, and the spawn-side bail message has
                // already named the failure mode that triggered cleanup.
                let _ = join_thread_with_timeout(j, &tw.exit_evt, THREAD_JOIN_TIMEOUT);
            }
        }
        // Early-bail pipe close. On the success path, into_handle
        // moved both `pipe_pairs` and `chain_pipes` into the handle,
        // so these Vecs are empty here and these loops iterate
        // nothing. On the early-bail path the guard still owns the
        // partially-allocated pipes and must close them now — the
        // child arm of each fork already closed any inherited
        // copies it held, and Thread-mode early-bail joined any
        // partially-spawned threads above before this loop runs.
        for (ab, ba) in &self.pipe_pairs {
            for fd in [ab[0], ab[1], ba[0], ba[1]] {
                let _ = nix::unistd::close(fd);
            }
        }
        for chain in &self.chain_pipes {
            for pipe in chain {
                let _ = nix::unistd::close(pipe[0]);
                let _ = nix::unistd::close(pipe[1]);
            }
        }
        // Munmap shared regions. `futex_ptrs[i]` and
        // `futex_region_sizes[i]` describe the same region, so each
        // munmap receives the exact length used for the matching
        // mmap. The two vectors are appended in lockstep inside
        // `spawn_group`, so they have identical lengths in every
        // observable state.
        for (&ptr, &size) in self.futex_ptrs.iter().zip(self.futex_region_sizes.iter()) {
            unsafe {
                libc::munmap(ptr as *mut libc::c_void, size);
            }
        }
        if !self.iter_counters.is_null() && self.iter_counter_bytes > 0 {
            unsafe {
                libc::munmap(
                    self.iter_counters as *mut libc::c_void,
                    self.iter_counter_bytes,
                );
            }
        }
    }
}

// SAFETY: futex_ptrs and iter_counters are MAP_SHARED anonymous pages
// created before fork, so every forked child inherits a pointer copy
// of the same underlying kernel object. Children read/write their own
// futex word — via `std::ptr::read_volatile`/`write_volatile` for
// most WorkType variants, or via `AtomicU32`/`AtomicU64` references
// re-derived from the raw pointer for FanOutCompute, which needs
// release-acquire ordering to publish `wake_ns` alongside the
// generation advance — and atomically store into their dedicated
// iter_counters slot (via a shared `&AtomicU64` reference derived
// from the `*mut AtomicU64` region pointer); the parent reads
// all slots via `snapshot_iterations` and is the sole process that
// munmaps the region, on WorkloadHandle::drop after every child has
// been reaped.
//
// Per-mode aliasing rationale:
//
// - Fork mode: each forked child constructs its own process-local
//   `&AtomicU32`/`&AtomicU64` shared reference into the MAP_SHARED
//   page from the inherited raw pointer. No reference value ever
//   crosses a process boundary — each process synthesises its own
//   reference from the same underlying kernel object. Interior
//   mutation through a shared atomic reference is permitted by
//   Rust's aliasing model because `AtomicU32`/`AtomicU64` wrap an
//   `UnsafeCell`; the post-fork alias relation is therefore not an
//   aliasing-rule violation.
//
// - Thread mode: under [`CloneMode::Thread`] every worker thread
//   shares the parent process's single address space — the same
//   raw `*mut AtomicU32`/`*mut AtomicU64` pointer is dereferenced
//   from multiple threads concurrently, and the resulting
//   `&AtomicU32`/`&AtomicU64` shared references coexist for
//   overlapping lifetimes. This is sound for the same reason
//   `Arc<AtomicU64>` is sound: atomic types' `UnsafeCell`-wrapped
//   storage permits concurrent shared-reference access by design,
//   and the underlying load/store instructions are by construction
//   non-tearing on every supported target. No `&mut` reference is
//   ever materialised; every access is via the atomic API. The
//   MAP_SHARED region is allocated once before any worker spawns
//   and `munmap`ped after every worker has been joined, so the
//   underlying kernel object outlives every alias.
unsafe impl Send for WorkloadHandle {}
unsafe impl Sync for WorkloadHandle {}

/// Pointer-sized addresses passed across a thread-spawn boundary.
///
/// Rust's auto-`Send` inference on closures conservatively treats
/// `*mut T` as `!Send` even inside a wrapper struct destructured in
/// the closure body — the destructured field type leaks into the
/// closure's auto-trait check. The simplest workaround is to round-
/// trip the pointers through `usize` (Send + Copy) and re-cast on
/// the receiver side. Soundness is identical: thread-mode workers
/// share the parent's address space, so the addresses retain
/// meaning across the thread boundary, and the underlying
/// MAP_SHARED regions are owned by the guard / handle for the full
/// duration of every worker.
///
/// `SendFutexPtr` carries a (futex_address, pos) tuple wrapped in
/// `Option`; `None` is the "no futex required" case for work types
/// that don't need shared memory. `SendIterSlotPtr` carries a single
/// address (zero ⇒ no iter_slot publish).
#[derive(Clone, Copy)]
pub(super) struct SendFutexPtr(Option<(usize, usize)>);

#[derive(Clone, Copy)]
pub(super) struct SendIterSlotPtr(usize);

impl SendFutexPtr {
    fn new(p: Option<(*mut u32, usize)>) -> Self {
        SendFutexPtr(p.map(|(ptr, pos)| (ptr as usize, pos)))
    }

    /// Re-cast back into the `*mut u32` + `pos` tuple `worker_main`
    /// expects.
    fn into_raw(self) -> Option<(*mut u32, usize)> {
        self.0.map(|(addr, pos)| (addr as *mut u32, pos))
    }
}

impl SendIterSlotPtr {
    fn new(p: *mut AtomicU64) -> Self {
        SendIterSlotPtr(p as usize)
    }

    fn into_raw(self) -> *mut AtomicU64 {
        self.0 as *mut AtomicU64
    }
}

/// Per-group resolved view of [`WorkloadConfig`] used by the
/// spawn pipeline.
///
/// [`WorkloadHandle::spawn`] iterates one `GroupParams` per group
/// it spawns: the primary group (`group_idx == 0`) is built from
/// the top-level [`WorkloadConfig`] fields via
/// [`Self::primary`], and each composed [`WorkSpec`] entry is
/// resolved into its own `GroupParams` (with `group_idx ==
/// 1..=N`) via [`Self::from_composed`]. Both paths funnel through
/// [`Self::from_work_spec`] for the actual field copy.
///
/// `GroupParams` is the post-resolution shape — `num_workers` is a
/// concrete `usize` (not the `Option<usize>` that [`WorkSpec`]
/// carries), `affinity` is a concrete [`ResolvedAffinity`] (not
/// the [`AffinityIntent`] that [`WorkSpec`] carries). The spawn
/// pipeline operates on `GroupParams` exclusively so it never has
/// to deal with the unresolved intent/optional shapes that the
/// user-facing types expose.
///
/// `clone_mode` is shared across every group — the top-level
/// [`WorkloadConfig::clone_mode`] selects fork vs thread dispatch
/// for the entire workload, and [`WorkSpec`] carries no
/// `clone_mode` field of its own (composed entries inherit the
/// parent's mode; the [`SpawnGuard`]'s lifecycle assumes a single
/// dispatch path).
#[derive(Clone)]
pub(super) struct GroupParams {
    work_type: WorkType,
    sched_policy: SchedPolicy,
    mem_policy: MemPolicy,
    mpol_flags: MpolFlags,
    nice: Option<i32>,
    comm: Option<String>,
    uid: Option<u32>,
    gid: Option<u32>,
    numa_node: Option<u32>,
    affinity: ResolvedAffinity,
    num_workers: usize,
    group_idx: usize,
}

impl GroupParams {
    /// Extract a [`GroupParams`] from a [`WorkSpec`] given the
    /// resolved sibling values. This is the single field-extraction
    /// site — both [`Self::primary`] and [`Self::from_composed`]
    /// funnel through here, so the field-by-field copy lives in one
    /// place.
    ///
    /// The caller is responsible for resolving the
    /// [`WorkSpec::num_workers`] `Option<usize>` to a concrete
    /// `usize` and the [`WorkSpec::affinity`] [`AffinityIntent`] to
    /// a concrete [`ResolvedAffinity`]. The remaining fields
    /// (`work_type`, `sched_policy`, `mem_policy`, `mpol_flags`,
    /// `nice`) are copied verbatim — they need no resolution
    /// because both [`WorkSpec`] and [`GroupParams`] carry them in
    /// their final runtime form (`nice` round-trips as
    /// `Option<i32>` so `None` continues to mean "skip the
    /// `setpriority(2)` call" at the spawn-time gate).
    fn from_work_spec(
        spec: &WorkSpec,
        group_idx: usize,
        resolved_affinity: ResolvedAffinity,
        resolved_num_workers: usize,
    ) -> Self {
        Self {
            work_type: spec.work_type.clone(),
            sched_policy: spec.sched_policy,
            mem_policy: spec.mem_policy.clone(),
            mpol_flags: spec.mpol_flags,
            nice: spec.nice,
            comm: spec.comm.as_ref().map(|c| c.to_string()),
            uid: spec.uid,
            gid: spec.gid,
            numa_node: spec.numa_node,
            affinity: resolved_affinity,
            num_workers: resolved_num_workers,
            group_idx,
        }
    }

    /// Resolve an [`AffinityIntent`] to a [`ResolvedAffinity`] under
    /// the spawn-time gate: only `Inherit`, `Exact`, and
    /// `RandomSubset` carry enough information to resolve without
    /// scenario context (the caller supplies the `from` pool for
    /// `RandomSubset`, so per-worker sampling stays self-contained).
    /// Topology-aware variants (`SingleCpu`, `LlcAligned`,
    /// `CrossCgroup`) require a [`crate::topology::TestTopology`] /
    /// cpuset state that [`WorkloadHandle::spawn`] does not have, so
    /// they bail with an actionable diagnostic.
    ///
    /// `site` names the location of the affinity field for the bail
    /// message — `"WorkloadConfig::affinity"` for the primary group,
    /// `"composed[N].affinity"` for entries inside `composed`. Pinned
    /// across both call sites so the gate matches exactly and a
    /// future variant addition is rejected uniformly.
    pub(super) fn resolve_spawn_affinity(
        intent: &AffinityIntent,
        site: &str,
    ) -> Result<ResolvedAffinity> {
        match intent {
            AffinityIntent::Inherit => Ok(ResolvedAffinity::None),
            AffinityIntent::Exact(cpus) => {
                if cpus.is_empty() {
                    anyhow::bail!(
                        "{site} = AffinityIntent::Exact with empty CPU set \
                         would produce EINVAL from sched_setaffinity; \
                         use AffinityIntent::Inherit for no affinity \
                         constraint",
                    );
                }
                Ok(ResolvedAffinity::Fixed(cpus.clone()))
            }
            AffinityIntent::RandomSubset { from, count } => {
                if from.is_empty() {
                    anyhow::bail!(
                        "{site} = AffinityIntent::RandomSubset with empty \
                         pool; use AffinityIntent::Inherit for no affinity \
                         constraint",
                    );
                }
                if *count == 0 {
                    anyhow::bail!(
                        "{site} = AffinityIntent::RandomSubset with \
                         count=0; use AffinityIntent::Inherit for no \
                         affinity constraint",
                    );
                }
                Ok(ResolvedAffinity::Random {
                    from: from.clone(),
                    count: *count,
                })
            }
            AffinityIntent::SingleCpu
            | AffinityIntent::LlcAligned
            | AffinityIntent::CrossCgroup
            | AffinityIntent::SmtSiblingPair => {
                anyhow::bail!(
                    "{site} = {:?} requires scenario context; use \
                     AffinityIntent::Exact(set), \
                     AffinityIntent::RandomSubset {{ from, count }}, \
                     or AffinityIntent::Inherit when spawning directly \
                     via WorkloadHandle::spawn. Topology-aware variants \
                     resolve automatically inside #[ktstr_test] \
                     scenarios.",
                    intent,
                );
            }
        }
    }

    /// Build the primary group's parameters from the top-level
    /// [`WorkloadConfig`] fields. `group_idx` is fixed to `0`.
    ///
    /// Synthesises a [`WorkSpec`] view of the top-level config
    /// fields and funnels through [`Self::from_work_spec`] so the
    /// field-by-field copy lives in exactly one place. The
    /// synthesised spec mirrors the resolved sibling values
    /// (`num_workers: Some(n)`, `affinity: Inherit`) — the spawn
    /// pipeline never reads it.
    ///
    /// `WorkloadConfig::affinity` is an [`AffinityIntent`]
    /// (type-unified with [`WorkSpec::affinity`]); resolution to
    /// [`ResolvedAffinity`] runs through
    /// [`Self::resolve_spawn_affinity`] under the same gate as
    /// [`Self::from_composed`]. Topology-aware variants
    /// (`SingleCpu`, `LlcAligned`, `CrossCgroup`) require scenario
    /// context; the scenario engine pre-resolves them via
    /// `crate::scenario::intent_for_spawn` (which round-trips
    /// `RandomSubset` verbatim and flattens topology-aware variants
    /// to `Exact`) before building [`WorkloadConfig`], so the gate
    /// only ever sees `Inherit`, `Exact`, or `RandomSubset` from
    /// this path.
    fn primary(config: &WorkloadConfig) -> Result<Self> {
        let resolved_affinity =
            Self::resolve_spawn_affinity(&config.affinity, "WorkloadConfig::affinity")?;
        let spec = WorkSpec {
            work_type: config.work_type.clone(),
            sched_policy: config.sched_policy,
            num_workers: Some(config.num_workers),
            affinity: AffinityIntent::Inherit,
            mem_policy: config.mem_policy.clone(),
            mpol_flags: config.mpol_flags,
            nice: config.nice,
            comm: config.comm.clone(),
            uid: config.uid,
            gid: config.gid,
            numa_node: config.numa_node,
            pcomm: None,
        };
        Ok(Self::from_work_spec(
            &spec,
            0,
            resolved_affinity,
            config.num_workers,
        ))
    }

    /// Resolve a composed [`WorkSpec`] into per-group parameters,
    /// applying the spawn-time rules documented on
    /// [`WorkloadConfig::composed`]:
    ///
    /// - `num_workers` must be `Some(n)`; the `None` default
    ///   resolved by the scenario engine via
    ///   `Ctx::workers_per_cgroup` is unreachable here. A `None`
    ///   value is rejected with an actionable diagnostic.
    /// - `affinity` resolution runs through
    ///   [`Self::resolve_spawn_affinity`] —
    ///   [`AffinityIntent::Inherit`] (mapped to
    ///   [`ResolvedAffinity::None`]),
    ///   [`AffinityIntent::Exact`] (mapped to
    ///   [`ResolvedAffinity::Fixed`]), and
    ///   [`AffinityIntent::RandomSubset`] (mapped to
    ///   [`ResolvedAffinity::Random`]) are accepted; topology-aware
    ///   variants are rejected.
    ///
    /// Composed entries inherit the parent
    /// [`WorkloadConfig::clone_mode`]; [`WorkSpec`] has no
    /// `clone_mode` field of its own.
    fn from_composed(spec: &WorkSpec, group_idx: usize) -> Result<Self> {
        if spec.pcomm.is_some() {
            anyhow::bail!(
                "composed[{}].pcomm: pcomm via WorkloadHandle::spawn is not supported; \
                 use WorkloadHandle::spawn_pcomm_cgroup or CgroupDef (apply_setup) — \
                 spawn always forks one process per worker and never coalesces into \
                 a thread-group leader",
                group_idx - 1,
            );
        }
        let num_workers = spec.num_workers.ok_or_else(|| {
            anyhow::anyhow!(
                "composed[{}].num_workers must be set explicitly at spawn time \
                 (the Some/None resolution via Ctx::workers_per_cgroup is only \
                 available through the scenario engine; \
                 WorkloadHandle::spawn requires a concrete count)",
                group_idx - 1,
            )
        })?;
        let site = format!("composed[{}].affinity", group_idx - 1);
        let affinity = Self::resolve_spawn_affinity(&spec.affinity, &site)?;
        Ok(Self::from_work_spec(spec, group_idx, affinity, num_workers))
    }
}

/// Shared per-group admission rules common to every spawn entry
/// point. Validates only the rules that are dispatch-agnostic —
/// `worker_group_size` divisibility, `chain_pipe_depth >= 2`,
/// `IdleChurn` zero-duration rejections, `IpcVariance` zero-knob
/// rejections. Dispatch-specific compatibility checks (CloneMode
/// vs WorkType, pcomm vs WorkType) stay at each entry point's
/// admission block since their reasoning differs per dispatch.
///
/// Called from [`WorkloadHandle::spawn`] (per-group inside
/// `groups`) and [`WorkloadHandle::spawn_pcomm_cgroup`] (per-group
/// inside its own `groups`). Centralises the rules so a future
/// addition (e.g. a new `WorkType` zero-rejection) lives in one
/// place.
pub(super) fn validate_workload_admission(group: &GroupParams) -> Result<()> {
    if let Some(group_size) = group.work_type.worker_group_size()
        && (group.num_workers == 0 || !group.num_workers.is_multiple_of(group_size))
    {
        return Err(WorkTypeValidationError::NonDivisibleWorkerCount {
            name: group.work_type.name().to_string(),
            group_idx: group.group_idx,
            group_size,
            num_workers: group.num_workers,
        }
        .into());
    }
    if let Some(depth) = group.work_type.chain_pipe_depth()
        && depth < 2
    {
        return Err(WorkTypeValidationError::InsufficientWakeChainDepth {
            depth,
            group_idx: group.group_idx,
        }
        .into());
    }
    if let WorkType::IdleChurn {
        burst_duration,
        sleep_duration,
        ..
    } = group.work_type
    {
        if burst_duration.is_zero() {
            return Err(WorkTypeValidationError::ZeroBurstDuration {
                group_idx: group.group_idx,
            }
            .into());
        }
        if sleep_duration.is_zero() {
            return Err(WorkTypeValidationError::ZeroSleepDuration {
                group_idx: group.group_idx,
            }
            .into());
        }
    }
    if let WorkType::IpcVariance {
        hot_iters,
        cold_iters,
        period_iters,
    } = group.work_type
    {
        if hot_iters == 0 {
            return Err(WorkTypeValidationError::ZeroIpcVarianceParam {
                field: "hot_iters",
                group_idx: group.group_idx,
            }
            .into());
        }
        if cold_iters == 0 {
            return Err(WorkTypeValidationError::ZeroIpcVarianceParam {
                field: "cold_iters",
                group_idx: group.group_idx,
            }
            .into());
        }
        if period_iters == 0 {
            return Err(WorkTypeValidationError::ZeroIpcVarianceParam {
                field: "period_iters",
                group_idx: group.group_idx,
            }
            .into());
        }
    }
    Ok(())
}

/// Spawn a single thread-mode worker via [`std::thread::Builder`].
///
/// The thread closure runs `worker_main` directly with the same
/// per-worker arguments the fork dispatch passes, except `stop` is
/// a per-worker `Arc<AtomicBool>` instead of the global [`STOP`].
/// Start rendezvous uses an `mpsc::sync_channel(0)` because every
/// worker needs to block until the parent calls
/// [`WorkloadHandle::start`]; the parent then sends `()` to each
/// worker's `start_tx` to unblock them in order.
///
/// `tid` is published from inside the closure via `gettid()` after
/// the start handshake completes, so [`WorkloadHandle::worker_pids`]
/// reads it post-`start`. A pre-start read returns `0`, which is
/// the documented sentinel for "not yet running".
///
/// SIGUSR1 is process-wide and useless for per-thread stop control,
/// so this path does not install a signal handler. The parent flips
/// `stop` directly from [`WorkloadHandle::stop_and_collect`].
#[allow(clippy::too_many_arguments)]
pub(super) fn spawn_thread_worker(
    guard: &mut SpawnGuard,
    group: &GroupParams,
    affinity: Option<BTreeSet<usize>>,
    worker_pipe_fds: Option<(i32, i32)>,
    worker_futex: Option<(*mut u32, usize)>,
    iter_slot: *mut AtomicU64,
) -> Result<()> {
    use std::sync::Arc;
    use std::sync::atomic::AtomicI32;
    use std::sync::mpsc;

    // SyncSender(0) — bounded rendezvous channel. The thread blocks
    // in `recv()` until the parent sends `()`; if the parent drops
    // the sender first (mid-spawn cleanup or early bail), `recv()`
    // returns `Err(Disconnected)` and the closure exits cleanly.
    let (start_tx, start_rx) = mpsc::sync_channel::<()>(0);
    let stop = Arc::new(AtomicBool::new(false));
    let tid = Arc::new(AtomicI32::new(0));
    // Per-worker exit eventfd: bumped by a Drop guard inside the
    // closure right before the thread returns its `WorkerReport`. The
    // parent's `join_thread_with_timeout` blocks in `epoll_wait` on
    // this fd instead of sleep-polling `is_finished`. Created with
    // `EFD_NONBLOCK` so the Drop-time `write` cannot block; counter
    // mode so a missed read just accumulates without losing the edge.
    let exit_evt = Arc::new(
        vmm_sys_util::eventfd::EventFd::new(libc::EFD_NONBLOCK)
            .context("create thread-worker exit eventfd")?,
    );

    // Clone Arcs for the closure. The thread takes ownership of the
    // closure-side handles; the parent retains the originals via
    // ThreadWorker for stop signaling and tid reading.
    let stop_thread = Arc::clone(&stop);
    let tid_thread = Arc::clone(&tid);
    let exit_evt_thread = Arc::clone(&exit_evt);
    let work_type = group.work_type.clone();
    let sched_policy = group.sched_policy;
    let mem_policy = group.mem_policy.clone();
    let mpol_flags = group.mpol_flags;
    let nice = group.nice;
    let comm = group.comm.clone();
    let uid = group.uid;
    let gid = group.gid;
    let numa_node = group.numa_node;
    let group_idx = group.group_idx;
    let num_workers = group.num_workers;

    // The closure must be `Send` to cross the thread boundary.
    // `worker_pipe_fds` is `Option<(i32, i32)>` (Copy + Send), but
    // `worker_futex` and `iter_slot` are raw pointers and not
    // `Send` by default. The module-level `SendFutexPtr` and
    // `SendIterSlotPtr` newtypes round-trip the addresses through
    // `usize` so the closure's capture set is genuinely Send (no
    // raw-pointer field appears in the closure type).
    let futex_send = SendFutexPtr::new(worker_futex);
    let iter_slot_send = SendIterSlotPtr::new(iter_slot);

    let join = std::thread::Builder::new()
        .name(format!("ktstr-worker-g{group_idx}-{}", guard.threads.len()))
        .spawn(move || {
            // Drop guard: signal the exit eventfd as the closure
            // unwinds, regardless of whether `worker_main` returned
            // normally or panicked. The parent's
            // `join_thread_with_timeout` blocks in `epoll_wait` on
            // this fd; a panic that bypassed the explicit signal
            // would otherwise leave the parent waiting until the
            // safety timerfd fires. Drop runs even under unwinding,
            // so this guard captures both the normal and panic
            // paths.
            struct WorkerExitSignal(std::sync::Arc<vmm_sys_util::eventfd::EventFd>);
            impl Drop for WorkerExitSignal {
                fn drop(&mut self) {
                    let _ = self.0.write(1);
                }
            }
            let _exit_signal = WorkerExitSignal(exit_evt_thread);

            // Publish gettid() so the parent can address this task
            // for sched_setaffinity and report it from worker_pids.
            // gettid() is the kernel TID; getpid() would return the
            // shared tgid, which collides across threads.
            let my_tid: libc::pid_t = unsafe { libc::syscall(libc::SYS_gettid) as libc::pid_t };
            // Release pairs with Acquire on the parent's
            // `tid.load()` sites so any reader observing a non-zero
            // tid also sees the worker's post-start state. Cheap on
            // every supported target (release-store on the Arc's
            // underlying AtomicI32 is a single instruction).
            tid_thread.store(my_tid, Ordering::Release);

            // Block on start rendezvous. `Err(_)` means the parent
            // dropped start_tx before sending — return a sentinel
            // WorkerReport without doing any work.
            if start_rx.recv().is_err() {
                return WorkerReport {
                    tid: my_tid,
                    completed: false,
                    group_idx,
                    ..WorkerReport::default()
                };
            }

            // Re-cast usize addresses back into raw pointers for
            // worker_main. SAFETY: the ownership and lifetime
            // arguments documented on `SendFutexPtr` /
            // `SendIterSlotPtr` ensure these pointers are still
            // live when worker_main dereferences them.
            let futex = futex_send.into_raw();
            let slot = iter_slot_send.into_raw();

            worker_main(
                affinity,
                work_type,
                sched_policy,
                mem_policy,
                mpol_flags,
                nice,
                comm.as_deref(),
                uid,
                gid,
                numa_node,
                worker_pipe_fds,
                futex,
                slot,
                &stop_thread,
                group_idx,
            )
        })
        .with_context(|| {
            format!(
                "thread::spawn for worker {}/{} (group {}) failed",
                guard.threads.len() + 1,
                num_workers,
                group_idx,
            )
        })?;

    guard.threads.push(ThreadWorker {
        tid,
        stop,
        start_tx: Some(start_tx),
        join: Some(join),
        exit_evt,
    });
    Ok(())
}

/// Per-group resource indices into the [`SpawnGuard`]'s flat
/// vectors used by [`spawn_pcomm_container`] to compute per-thread
/// pipe / chain / futex / iter-slot addresses inside the forked
/// container. Built by [`WorkloadHandle::spawn_pcomm_cgroup`] in
/// lockstep with the per-group resource allocation pass so the
/// container's threads can address their own group's resources from
/// the inherited shared regions.
#[derive(Clone, Copy, Debug)]
pub(super) struct PcommGroupResources {
    /// Offset into the iter_counters MAP_SHARED region for this
    /// group's first worker. Subsequent workers occupy
    /// `iter_offset + i` for `i in 0..num_workers`.
    pub iter_offset: usize,
    /// Index of this group's first pipe pair in `guard.pipe_pairs`
    /// (only meaningful when `needs_pipes` is true).
    pub pipe_pair_base: usize,
    /// Index of this group's first chain pipe in `guard.chain_pipes`
    /// (only meaningful when `chain_depth` is `Some(_)`).
    pub chain_pipes_base: usize,
    /// Index of this group's first futex region in `guard.futex_ptrs`
    /// (only meaningful when `needs_futex` is true).
    pub futex_ptrs_base: usize,
    /// True when the group's `WorkType` requires inter-worker pipe
    /// pairs (PipeIo / CachePipe).
    pub needs_pipes: bool,
    /// `Some(depth)` when the group's `WorkType` requires a chain
    /// pipe ring (WakeChain { wake: Pipe }); `None` otherwise.
    pub chain_depth: Option<usize>,
    /// True when the group's `WorkType` requires a MAP_SHARED futex
    /// region (FutexPingPong / FutexFanOut / FanOutCompute /
    /// MutexContention / etc.).
    pub needs_futex: bool,
    /// Worker-group-size for per-worker `pos` computation inside
    /// the futex region (e.g. 2 for FutexPingPong, the messenger /
    /// receiver split for FutexFanOut). Defaulted to 2 when the
    /// `WorkType` does not declare a group_size.
    pub futex_group_size: usize,
}

/// Fork one thread-group leader hosting every worker thread for the
/// supplied groups: fork the leader, set its `comm` to `pcomm` via
/// `prctl(PR_SET_NAME)` (the kernel truncates to `TASK_COMM_LEN - 1
/// = 15` bytes inside `__set_task_comm`), then spawn
/// `groups[k].num_workers` worker threads per group inside it. Every
/// worker thread shares the leader's tgid, so
/// `task->group_leader->comm == pcomm` for every worker. Models real
/// workloads like `chrome` (pcomm) hosting `ThreadPoolForeg` and
/// `GPU Process` (per-thread comm via [`WorkSpec::comm`]) or `java`
/// (pcomm) hosting `GC Thread` and `C2 CompilerThre`.
///
/// `groups` carries per-group [`GroupParams`] and `resources` carries
/// the parallel [`PcommGroupResources`] pre-built by
/// [`WorkloadHandle::spawn_pcomm_cgroup`] from the surrounding
/// per-group resource-allocation pass; the two slices have identical
/// length and `resources[k]` describes the indices owned by
/// `groups[k]`.
///
/// `container_uid` / `container_gid` are applied via `setresuid` /
/// `setresgid` once on the leader before spawning threads — the
/// leader takes the configured process credentials, and each worker
/// thread additionally re-applies its own merged credentials inside
/// `worker_main` at thread creation time (idempotent when the merge
/// produced the same value, which is the common case for a
/// CgroupDef-level default flowing through `merged_works`).
///
/// # Wire format
///
/// The parent collects via the same report pipe used by conventional
/// fork workers:
///
/// 1. After the start handshake, the leader writes one `b'r'` ready
///    byte to the report pipe after all threads are spawned, a
///    stronger guarantee than the conventional fork worker's
///    pre-loop byte.
/// 2. After all worker threads have joined, the leader serializes
///    `Vec<WorkerReport>` (one entry per worker thread, in
///    `(group_idx, within-group order)` traversal order) via
///    `serde_json::to_vec` and writes the bytes to the report pipe
///    before `_exit(0)`. The parent decodes via
///    `serde_json::from_slice::<Vec<WorkerReport>>`. The report pipe
///    is sized at 8 MiB; parent `read_to_end` drains concurrently
///    with the leader's `write_all` so payloads exceeding the pipe
///    size still drain correctly.
///
/// On a decode-failure or short payload, the parent emits one
/// sentinel report per expected worker so per-group filtering and
/// `assert_not_starved` see the correct cardinality.
///
/// # Lifecycle
///
/// - `PR_SET_PDEATHSIG(SIGKILL)`: when the parent dies, the kernel
///   sends SIGKILL to the LEADER thread (the calling task at fork
///   time). SIGKILL on any thread is fatal-to-tgid: the kernel
///   routes it through `complete_signal` → `do_group_exit` →
///   `zap_other_threads`, which sets `SIGNAL_GROUP_EXIT` on every
///   sibling thread and walks the thread list signalling each. So
///   the cascade to worker threads happens via
///   `zap_other_threads`, NOT via per-task PDEATHSIG inheritance
///   (PDEATHSIG itself is per-task and is cleared on clone). Net:
///   parent death → leader's SIGKILL → all worker threads die,
///   so the container cannot outlive the harness.
/// - `setpgid(0, 0)`: the leader becomes its own process group
///   leader so the parent's stop/kill `killpg` reaches any
///   descendants the workers spawn.
/// - SIGUSR1 handler installed pre-thread-spawn: any
///   `kill(leader_pid, SIGUSR1)` from the parent flips the leader's
///   STOP, which every worker thread observes via
///   `stop_requested(&STOP)` on its next loop check.
/// - `prctl(PR_SET_NAME, pcomm)`: sets the leader's `task->comm`,
///   which becomes `task->group_leader->comm` for every spawned
///   thread (kernel truncates to 15 bytes).
/// - `setresuid(container_uid, ...)` / `setresgid(container_gid,
///   ...)` (if specified) apply once before the start-byte poll.
/// - Threads spawn AFTER the start byte arrives, so `start()`'s
///   serial start-byte writes still gate work entry (the parent
///   moves the leader to its cgroup before sending start).
#[allow(clippy::too_many_arguments)]
pub(super) fn spawn_pcomm_container(
    guard: &mut SpawnGuard,
    pcomm: &str,
    container_uid: Option<u32>,
    container_gid: Option<u32>,
    groups: &[GroupParams],
    resources: &[PcommGroupResources],
) -> Result<()> {
    debug_assert_eq!(
        groups.len(),
        resources.len(),
        "spawn_pcomm_container: groups / resources must have the same length",
    );

    // Allocate the container's report and start pipes. Parent holds
    // report_fds[0] (read end) and start_fds[1] (write end); the
    // container holds the inverse ends post-fork. O_CLOEXEC matches
    // the defense-in-depth posture used by every other pipe in the
    // spawn pipeline — defends against future exec paths that could
    // otherwise leak the fd into a helper.
    let mut report_fds = [0i32; 2];
    if unsafe { libc::pipe2(report_fds.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
        anyhow::bail!(
            "pcomm container (pcomm={pcomm:?}): report pipe2 failed: {}",
            std::io::Error::last_os_error(),
        );
    }
    // Grow the report pipe to 8 MiB so the container's `write_all` of
    // the JSON-encoded `Vec<WorkerReport>` issues without blocking
    // for moderate worker counts. For larger payloads the parent's
    // `read_to_end` drains concurrently with the container's
    // `write_all`, so a pipe overflow degrades to extra wakeups
    // rather than truncation. Best-effort failure (older kernel /
    // EPERM) leaves the default size in place — large payloads
    // still drain via the concurrent flow but with more wake cycles.
    const REPORT_PIPE_SIZE: libc::c_int = 8 * 1024 * 1024;
    let prev_size = unsafe { libc::fcntl(report_fds[1], libc::F_SETPIPE_SZ, REPORT_PIPE_SIZE) };
    if prev_size < 0 {
        let err = std::io::Error::last_os_error();
        tracing::warn!(
            pcomm,
            requested_size = REPORT_PIPE_SIZE,
            %err,
            "F_SETPIPE_SZ on pcomm container report pipe failed; falling back \
             to default pipe capacity",
        );
    }
    let mut start_fds = [0i32; 2];
    if unsafe { libc::pipe2(start_fds.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
        unsafe {
            libc::close(report_fds[0]);
            libc::close(report_fds[1]);
        }
        anyhow::bail!(
            "pcomm container (pcomm={pcomm:?}): start pipe2 failed: {}",
            std::io::Error::last_os_error(),
        );
    }

    // Block SIGUSR1 across fork so the container inherits a blocked
    // mask; the post-fork install + restore pattern matches the
    // conventional fork worker (see `spawn_group`'s fork dispatch
    // for the full rationale).
    let mut old_mask: libc::sigset_t = unsafe { std::mem::zeroed() };
    let mut block_mask: libc::sigset_t = unsafe { std::mem::zeroed() };
    let psm_block_rc = unsafe {
        libc::sigemptyset(&mut block_mask);
        libc::sigaddset(&mut block_mask, libc::SIGUSR1);
        libc::pthread_sigmask(libc::SIG_BLOCK, &block_mask, &mut old_mask)
    };
    if psm_block_rc != 0 {
        tracing::warn!(
            rc = psm_block_rc,
            pcomm,
            "pthread_sigmask(SIG_BLOCK, SIGUSR1) failed pre-fork for pcomm container; \
             container inherits unblocked SIGUSR1 and may terminate on default \
             action before installing handler",
        );
    }

    // Sum of all groups' workers — matches `WorkerReport` cardinality
    // the parent expects.
    let total_workers: usize = groups.iter().map(|g| g.num_workers).sum();

    let pid = unsafe { libc::fork() };
    match pid {
        -1 => {
            let psm_restore_rc = unsafe {
                libc::pthread_sigmask(libc::SIG_SETMASK, &old_mask, std::ptr::null_mut())
            };
            if psm_restore_rc != 0 {
                tracing::warn!(
                    rc = psm_restore_rc,
                    "pthread_sigmask(SIG_SETMASK) failed restoring mask after pcomm \
                     container fork failure; SIGUSR1 may remain blocked in this thread",
                );
            }
            unsafe {
                libc::close(report_fds[0]);
                libc::close(report_fds[1]);
                libc::close(start_fds[0]);
                libc::close(start_fds[1]);
            }
            anyhow::bail!(
                "pcomm container (pcomm={pcomm:?}): fork failed: {}",
                std::io::Error::last_os_error(),
            );
        }
        0 => {
            // Container child. Mirror the conventional fork worker's
            // post-fork init: PDEATHSIG, getppid orphan check,
            // setpgid, SIGUSR1 handler install + mask restore. This
            // is the exact same defensive sequence; the only
            // structural difference is what happens AFTER the
            // start-byte handshake (thread spawn instead of inline
            // worker_main).
            unsafe {
                libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL);
            }
            if std::env::var_os("KTSTR_GUEST_INIT").is_none() && unsafe { libc::getppid() } == 1 {
                unsafe {
                    libc::_exit(0);
                }
            }
            unsafe {
                libc::setpgid(0, 0);
            }
            STOP.store(false, Ordering::Relaxed);
            let sig_prev = unsafe {
                libc::signal(
                    libc::SIGUSR1,
                    sigusr1_handler as *const () as libc::sighandler_t,
                )
            };
            if sig_prev == libc::SIG_ERR {
                let errno = std::io::Error::last_os_error();
                eprintln!(
                    "ktstr: signal(SIGUSR1) install failed in pcomm container: {errno}; \
                     graceful stop unavailable, killpg escalation will reap"
                );
            }
            let psm_unblock_rc = unsafe {
                libc::pthread_sigmask(libc::SIG_SETMASK, &old_mask, std::ptr::null_mut())
            };
            if psm_unblock_rc != 0 {
                eprintln!(
                    "ktstr: pthread_sigmask(SIG_SETMASK) unblock failed in pcomm \
                     container: rc={psm_unblock_rc}; SIGUSR1 stays blocked, killpg \
                     escalation will reap"
                );
            }
            // Close the parent's ends. Keep report_fds[1] (the
            // container's write end) and start_fds[0] (the
            // container's read end).
            unsafe {
                libc::close(report_fds[0]);
                libc::close(start_fds[1]);
            }

            // Inherited-fd sweep. The container forked from the
            // harness inherits every fd in the parent's table —
            // including pipe fds from OTHER live `WorkloadHandle`s
            // and any harness-only fds (tracing subscribers etc.).
            // Close everything not in the keep-set:
            //   - 0 / 1 / 2 (stdio).
            //   - `report_fds[1]` (this container's write end).
            //   - `start_fds[0]` (this container's read end).
            //   - This spawn's `pipe_pairs` (per-pair inter-worker
            //     fds) and `chain_pipes` (per-chain ring fds) —
            //     worker threads will use them once they start.
            //
            // MAP_SHARED regions (futex / iter_counters) are mmap
            // memory inherited via CLONE_VM, not fd entries; nothing
            // to keep on that axis. The sweep walks
            // `/proc/self/fd` once and uses raw `libc::close` on
            // each non-kept fd; failure to close (EBADF) is benign
            // — the entry is gone by the time we get to it. Heap
            // allocation here is acceptable post-fork (PhD-approved
            // pattern; the conventional fork worker also heap-
            // allocates in `worker_main`).
            let mut keep_fds: std::collections::BTreeSet<i32> = std::collections::BTreeSet::new();
            keep_fds.insert(0);
            keep_fds.insert(1);
            keep_fds.insert(2);
            keep_fds.insert(report_fds[1]);
            keep_fds.insert(start_fds[0]);
            for (ab, ba) in &guard.pipe_pairs {
                keep_fds.insert(ab[0]);
                keep_fds.insert(ab[1]);
                keep_fds.insert(ba[0]);
                keep_fds.insert(ba[1]);
            }
            for chain in &guard.chain_pipes {
                for pipe in chain {
                    keep_fds.insert(pipe[0]);
                    keep_fds.insert(pipe[1]);
                }
            }
            if let Ok(entries) = std::fs::read_dir("/proc/self/fd") {
                let mut to_close: Vec<i32> = Vec::new();
                for entry in entries.flatten() {
                    if let Some(name) = entry.file_name().to_str()
                        && let Ok(fd) = name.parse::<i32>()
                        && !keep_fds.contains(&fd)
                    {
                        to_close.push(fd);
                    }
                }
                for fd in to_close {
                    unsafe {
                        libc::close(fd);
                    }
                }
            } else {
                eprintln!(
                    "ktstr: pcomm container (pcomm={pcomm:?}): /proc/self/fd \
                     read_dir failed; inherited-fd sweep skipped",
                );
            }

            // prctl(PR_SET_NAME, pcomm). The kernel truncates to 15
            // bytes (TASK_COMM_LEN - 1) inside `__set_task_comm`
            // and accepts any byte string up to that limit including
            // the empty string. Setting it on the container's tgid
            // leader makes `task->group_leader->comm == pcomm` for
            // every spawned thread (kernel/sys.c::prctl_set_name).
            // CString construction can only fail on an interior
            // NUL byte; the WorkSpec::pcomm builder rejects those at
            // declaration time, so the unwrap path is unreachable
            // for any value that arrives here. `unwrap_or_default`
            // is a defensive fall-through to an empty CString —
            // the kernel accepts it; the leader's comm is set to
            // the empty string and the failure is surfaced to
            // stderr.
            let c_pcomm = std::ffi::CString::new(pcomm).unwrap_or_default();
            let prctl_rc = unsafe { libc::prctl(libc::PR_SET_NAME, c_pcomm.as_ptr()) };
            if prctl_rc != 0 {
                let errno = std::io::Error::last_os_error();
                eprintln!(
                    "ktstr: prctl(PR_SET_NAME, {pcomm:?}) failed on pcomm container: {errno}",
                );
            }

            // Apply container-level credentials. Order matters:
            // setresgid first (while still uid=0 we can change the
            // primary group), setresuid second (after the uid drop
            // the process loses CAP_SETGID and a later gid change
            // would EPERM). Failures are surfaced to stderr but do
            // not halt the container — the threads inside still
            // run with the inherited credentials, and worker_main's
            // own setresuid / setresgid calls (which run after this)
            // act as a second-line defense for any per-thread merge
            // that produced a different value than the container
            // default.
            if let Some(gid) = container_gid {
                let rc = unsafe { libc::setresgid(gid, gid, gid) };
                if rc != 0 {
                    let errno = std::io::Error::last_os_error();
                    eprintln!("ktstr: setresgid({gid}) failed on pcomm container: {errno}");
                }
            }
            if let Some(uid) = container_uid {
                let rc = unsafe { libc::setresuid(uid, uid, uid) };
                if rc != 0 {
                    let errno = std::io::Error::last_os_error();
                    eprintln!("ktstr: setresuid({uid}) failed on pcomm container: {errno}");
                }
            }

            // Wait for the parent's start byte (poll with 30s
            // timeout — same budget as the conventional fork
            // worker). Match the conventional worker's behaviour:
            // on `poll <= 0` _exit(1) so the parent's collect path
            // observes a missing report and emits sentinels.
            let mut pfd = libc::pollfd {
                fd: start_fds[0],
                events: libc::POLLIN,
                revents: 0,
            };
            let ret = unsafe { libc::poll(&mut pfd, 1, 30_000) };
            if ret <= 0 {
                unsafe {
                    libc::_exit(1);
                }
            }
            let mut buf = [0u8; 1];
            {
                let mut f = unsafe { std::fs::File::from_raw_fd(start_fds[0]) };
                let _ = f.read_exact(&mut buf);
                drop(f);
            }

            // Unlike fork workers, the pcomm container spawns threads after
            // the start byte. A latched STOP from a SIGUSR1 race during mask
            // restore would produce N zero-work threads. Reset so threads
            // get a fair start.
            STOP.store(false, Ordering::Relaxed);

            // Spawn worker threads for every group. Each thread
            // computes its own affinity / pipe-fd / futex /
            // iter_slot from its (group_idx, within-group index)
            // pair, matching the conventional fork loop's per-worker
            // computation but inside the container's address space
            // (so threads share the parent-allocated MAP_SHARED
            // regions and inter-worker pipes inherited via fork).
            // Each spawned thread gets its own `EventFd` cloned into
            // the closure. A `WorkerExitSignal` Drop guard inside the
            // closure writes 1 to that fd as the closure unwinds —
            // covering both the normal-return and panic paths. The
            // container's join phase below `epoll_wait`s on every
            // eventfd, so a thread's exit edge is delivered without
            // any polling sleep. This mirrors `join_thread_with_timeout`'s
            // pattern but without a timer fd: the parent's
            // `stop_and_collect` SIGKILL is the only timeout authority,
            // per the user's "event-driven only, no sleeps or
            // timeouts" rule.
            let mut joins: Vec<(
                std::thread::JoinHandle<WorkerReport>,
                std::sync::Arc<vmm_sys_util::eventfd::EventFd>,
            )> = Vec::with_capacity(total_workers);
            // Snapshot the guard's resource bookkeeping we need
            // inside each thread closure. The guard is borrowed
            // mutably by the surrounding `spawn_pcomm_cgroup` and
            // dropped on the parent side after the function returns;
            // after fork the container has its own copy of every
            // field, and these snapshots are clones owned by the
            // container's process so the closures can capture them
            // without aliasing the guard.
            let pipe_pairs_snapshot: Vec<([i32; 2], [i32; 2])> = guard.pipe_pairs.clone();
            let chain_pipes_snapshot: Vec<Vec<[i32; 2]>> = guard.chain_pipes.clone();
            let futex_ptrs_snapshot: Vec<*mut u32> = guard.futex_ptrs.clone();
            let iter_counters_base = guard.iter_counters;

            for (group, res) in groups.iter().zip(resources.iter()) {
                let num_workers = group.num_workers;
                for i in 0..num_workers {
                    // Per-thread affinity resolution. Random-subset
                    // affinity samples a fresh subset per thread, so
                    // each call produces an independent BTreeSet.
                    // Fixed affinity returns the same BTreeSet for
                    // every thread.
                    let affinity = match resolve_affinity(&group.affinity) {
                        Ok(a) => a,
                        Err(e) => {
                            eprintln!(
                                "ktstr: pcomm container (group {}): resolve_affinity \
                                 for thread {i}/{num_workers} failed: {e:#}",
                                group.group_idx,
                            );
                            None
                        }
                    };

                    let worker_pipe_fds: Option<(i32, i32)> = if res.needs_pipes {
                        let pair_idx = res.pipe_pair_base + i / 2;
                        let (ab, ba) = &pipe_pairs_snapshot[pair_idx];
                        if i % 2 == 0 {
                            Some((ba[0], ab[1]))
                        } else {
                            Some((ab[0], ba[1]))
                        }
                    } else if let Some(depth) = res.chain_depth
                        && depth > 0
                    {
                        let chain_idx = res.chain_pipes_base + i / depth;
                        let stage = i % depth;
                        let prev_stage = (stage + depth - 1) % depth;
                        let chain = &chain_pipes_snapshot[chain_idx];
                        Some((chain[prev_stage][0], chain[stage][1]))
                    } else {
                        None
                    };

                    let worker_futex: Option<(*mut u32, usize)> = if res.needs_futex {
                        let futex_group_idx = res.futex_ptrs_base + i / res.futex_group_size;
                        let pos = i % res.futex_group_size;
                        Some((futex_ptrs_snapshot[futex_group_idx], pos))
                    } else {
                        None
                    };

                    let iter_slot: *mut AtomicU64 = if !iter_counters_base.is_null() {
                        unsafe { iter_counters_base.add(res.iter_offset + i) }
                    } else {
                        std::ptr::null_mut()
                    };

                    // Round raw pointers through Send-newtypes so
                    // the closure's auto-Send check passes (same
                    // wrapper pattern `spawn_thread_worker` uses).
                    let futex_send = SendFutexPtr::new(worker_futex);
                    let iter_slot_send = SendIterSlotPtr::new(iter_slot);

                    let work_type = group.work_type.clone();
                    let sched_policy = group.sched_policy;
                    let mem_policy = group.mem_policy.clone();
                    let mpol_flags = group.mpol_flags;
                    let nice = group.nice;
                    let comm = group.comm.clone();
                    let uid = group.uid;
                    let gid = group.gid;
                    let numa_node = group.numa_node;
                    let group_idx = group.group_idx;

                    // Per-thread exit eventfd. Cloned into the
                    // closure as `Arc<EventFd>`; the closure-local
                    // `WorkerExitSignal` Drop guard writes 1 to the
                    // fd on every exit path (normal return AND
                    // panic). The container's join phase below
                    // `epoll_wait`s on every eventfd to deliver the
                    // exit edge without polling. EFD_NONBLOCK keeps
                    // the Drop-time `write` from blocking — counter
                    // mode means the writer just bumps the counter
                    // without losing the edge.
                    let exit_evt = match vmm_sys_util::eventfd::EventFd::new(libc::EFD_NONBLOCK) {
                        Ok(efd) => std::sync::Arc::new(efd),
                        Err(e) => {
                            // Hard fail: continuing with fewer
                            // threads silently breaks the
                            // workload's worker count and the
                            // parent's report count. _exit(1) so
                            // the parent's sentinel path observes
                            // a missing payload and emits one
                            // sentinel per expected report. Reuses
                            // the same fail-stop contract as the
                            // start-byte poll timeout (`ret <=
                            // 0`).
                            eprintln!(
                                "ktstr: pcomm container (group {}): EventFd::new for \
                                 worker {}/{num_workers} failed: {e}; aborting container",
                                group.group_idx,
                                i + 1,
                            );
                            unsafe {
                                libc::_exit(1);
                            }
                        }
                    };
                    let exit_evt_thread = std::sync::Arc::clone(&exit_evt);

                    let join = std::thread::Builder::new()
                        .name(format!("ktstr-pcomm-g{group_idx}-{i}"))
                        .spawn(move || {
                            // Drop guard: bumps the eventfd as the
                            // closure unwinds. The write is
                            // non-blocking (EFD_NONBLOCK) and the
                            // counter accumulates, so a missed
                            // read just queues the next read's
                            // value — no edge loss. Drop runs
                            // under both normal return and
                            // unwinding (panic), so the container's
                            // join phase observes EVERY thread's
                            // exit, including panicked ones.
                            struct WorkerExitSignal(std::sync::Arc<vmm_sys_util::eventfd::EventFd>);
                            impl Drop for WorkerExitSignal {
                                fn drop(&mut self) {
                                    let _ = self.0.write(1);
                                }
                            }
                            let _exit_signal = WorkerExitSignal(exit_evt_thread);

                            let futex = futex_send.into_raw();
                            let slot = iter_slot_send.into_raw();
                            worker_main(
                                affinity,
                                work_type,
                                sched_policy,
                                mem_policy,
                                mpol_flags,
                                nice,
                                comm.as_deref(),
                                uid,
                                gid,
                                numa_node,
                                worker_pipe_fds,
                                futex,
                                slot,
                                &STOP,
                                group_idx,
                            )
                        });
                    match join {
                        Ok(j) => joins.push((j, exit_evt)),
                        Err(e) => {
                            // Hard fail: see EventFd path above.
                            // A partially-spawned container leaves
                            // the parent guessing about which
                            // reports are real vs missing; an
                            // _exit(1) collapses to the parent's
                            // sentinel path with deterministic
                            // cardinality.
                            eprintln!(
                                "ktstr: pcomm container (group {}): thread::spawn for \
                                 worker {}/{num_workers} failed: {e}; aborting container",
                                group.group_idx,
                                i + 1,
                            );
                            unsafe {
                                libc::_exit(1);
                            }
                        }
                    }
                }
            }

            // Publish the ready byte AFTER every worker thread is
            // alive. The parent's barrier polls every report fd for
            // POLLIN with a bounded deadline; the byte's correct
            // semantic is "the container has spawned every worker
            // thread and they are now running" — earlier (pre-spawn)
            // is observable to the parent before threads exist, which
            // races with `set_affinity(idx)` / `worker_pids()` calls
            // the harness may issue between the parent's barrier wake
            // and the work loop. Each thread starts work immediately
            // upon spawn (no per-thread handshake gate) since the
            // CONTAINER's start-byte poll above is the workload's
            // single start gate; once the container observes the
            // start byte, every thread it spawns is by construction
            // intended to run. Done as a raw `libc::write` to avoid
            // taking ownership of `report_fds[1]` (we still need it
            // for the JSON write below).
            let ready_byte: u8 = b'r';
            unsafe {
                libc::write(
                    report_fds[1],
                    &ready_byte as *const u8 as *const libc::c_void,
                    1,
                );
            }

            // Join every thread via `epoll_wait` on per-thread exit
            // eventfds. Each thread's `WorkerExitSignal` Drop guard
            // writes 1 to its fd as the closure unwinds (normal
            // return AND panic), so the container's `epoll_wait`
            // observes EVERY thread's exit edge without polling.
            // No timeout: per the user's "event-driven only, no
            // sleeps or timeouts" rule, the container blocks in
            // `epoll_wait(-1)` until each fd fires. The parent's
            // `stop_and_collect` 5s collect deadline + SIGKILL is
            // the only timeout authority; if a thread hangs (no
            // Drop guard fires — no realistic path to that under
            // the closures we spawn), the parent SIGKILLs the
            // container, the kernel tears down every thread, and
            // the container exits via the signal.
            //
            // Setup-failure fallback: if `Epoll::new` or any
            // `epoll.ctl(Add)` fails, fall back to blocking
            // `JoinHandle::join` per thread — that itself is
            // event-driven from the kernel's perspective (the
            // syscall blocks on the thread's exit signal, no
            // userspace polling). The fallback path is rare
            // (epoll_create1 failure means out of fds or out of
            // memory, both of which are pre-existing failure modes
            // the conventional fork worker also has).
            use std::os::unix::io::AsRawFd;
            use vmm_sys_util::epoll::{ControlOperation, Epoll, EpollEvent, EventSet};
            let mut indexed_joins: Vec<(
                usize,
                std::thread::JoinHandle<WorkerReport>,
                std::sync::Arc<vmm_sys_util::eventfd::EventFd>,
            )> = joins
                .into_iter()
                .enumerate()
                .map(|(i, (j, evt))| (i, j, evt))
                .collect();
            let mut reports: Vec<WorkerReport> = Vec::with_capacity(indexed_joins.len());

            let epoll_setup: Option<Epoll> = match Epoll::new() {
                Ok(ep) => {
                    let mut ok = true;
                    for (idx, _, evt) in &indexed_joins {
                        if let Err(e) = ep.ctl(
                            ControlOperation::Add,
                            evt.as_raw_fd(),
                            EpollEvent::new(EventSet::IN, *idx as u64),
                        ) {
                            eprintln!(
                                "ktstr: pcomm container (pcomm={pcomm:?}): epoll.ctl(Add) \
                                 for thread {idx} failed: {e}; falling back to blocking \
                                 join per thread",
                            );
                            ok = false;
                            break;
                        }
                    }
                    if ok { Some(ep) } else { None }
                }
                Err(e) => {
                    eprintln!(
                        "ktstr: pcomm container (pcomm={pcomm:?}): Epoll::new failed: {e}; \
                         falling back to blocking join per thread",
                    );
                    None
                }
            };

            if let Some(epoll) = epoll_setup {
                // Event-driven join loop. `epoll_wait(-1, …)` blocks
                // until at least one eventfd fires, which only
                // happens when a thread's `WorkerExitSignal` Drop
                // guard runs. The wake delivers `events[k].data()`
                // as the index of the finished thread; we
                // deregister, locate the entry, and `join()` it
                // (the `join()` call may briefly wait for the OS
                // thread to fully exit after the Drop guard, but
                // that wait is itself event-driven inside the
                // kernel — no userspace polling).
                let mut events_buf: Vec<EpollEvent> =
                    vec![EpollEvent::default(); indexed_joins.len().max(1)];
                while !indexed_joins.is_empty() {
                    let n = match epoll.wait(-1, &mut events_buf) {
                        Ok(n) => n,
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                        Err(e) => {
                            eprintln!(
                                "ktstr: pcomm container (pcomm={pcomm:?}): epoll.wait \
                                 failed: {e}; falling back to blocking join for the \
                                 remaining {} thread(s)",
                                indexed_joins.len(),
                            );
                            break;
                        }
                    };
                    for ev in &events_buf[..n] {
                        let target_idx = ev.data() as usize;
                        let pos = match indexed_joins
                            .iter()
                            .position(|(idx, _, _)| *idx == target_idx)
                        {
                            Some(p) => p,
                            None => continue,
                        };
                        let (idx, j, evt) = indexed_joins.swap_remove(pos);
                        if let Err(e) = epoll.ctl(
                            ControlOperation::Delete,
                            evt.as_raw_fd(),
                            EpollEvent::default(),
                        ) {
                            eprintln!(
                                "ktstr: pcomm container (pcomm={pcomm:?}): epoll.ctl(Del) \
                                 for thread {idx} failed: {e}",
                            );
                        }
                        let _ = evt.read();
                        match j.join() {
                            Ok(r) => reports.push(r),
                            Err(payload) => {
                                let msg = extract_panic_payload(payload);
                                eprintln!(
                                    "ktstr: pcomm container (pcomm={pcomm:?}): thread {idx} \
                                     panicked: {msg}",
                                );
                                reports.push(WorkerReport {
                                    completed: false,
                                    exit_info: Some(WorkerExitInfo::Panicked(msg)),
                                    ..WorkerReport::default()
                                });
                            }
                        }
                    }
                }
            }
            // Either the epoll path completed (drained
            // `indexed_joins`) or it bailed mid-flight; either way,
            // any remaining handles are joined by the kernel-blocking
            // `JoinHandle::join` below. `join` is event-driven inside
            // the kernel (futex on the thread's TID); no userspace
            // polling.
            for (idx, j, _evt) in indexed_joins {
                match j.join() {
                    Ok(r) => reports.push(r),
                    Err(payload) => {
                        let msg = extract_panic_payload(payload);
                        eprintln!(
                            "ktstr: pcomm container (pcomm={pcomm:?}): thread {idx} \
                             panicked: {msg}",
                        );
                        reports.push(WorkerReport {
                            completed: false,
                            exit_info: Some(WorkerExitInfo::Panicked(msg)),
                            ..WorkerReport::default()
                        });
                    }
                }
            }

            // Encode and write the report stream as a single
            // `Vec<WorkerReport>` JSON document via `serde_json`.
            // serde_json for pcomm Vec<WorkerReport>
            // design ruling; fork-mode workers use bincode for
            // single WorkerReport. The pcomm container is a
            // fork-mode child and its payload sits on the same
            // per-child pipe used by every other forked worker.
            // The parent decodes via
            // `serde_json::from_slice::<Vec<WorkerReport>>`. Encode
            // failures fall through to `_exit(0)` — the parent's
            // sentinel path handles missing / truncated payloads.
            // tracing is unsafe in a forked child (the parent's
            // subscriber may hold a lock); use eprintln + raw fd.
            let bytes = match serde_json::to_vec(&reports) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!(
                        "ktstr: pcomm container (pcomm={pcomm:?}): serde_json encode \
                         of {} reports failed: {e}",
                        reports.len(),
                    );
                    Vec::new()
                }
            };
            {
                let mut f = unsafe { std::fs::File::from_raw_fd(report_fds[1]) };
                if let Err(e) = f.write_all(&bytes) {
                    eprintln!(
                        "ktstr: pcomm container (pcomm={pcomm:?}): report write_all \
                         of {} bytes failed: {e}",
                        bytes.len(),
                    );
                }
                drop(f);
            }
            unsafe {
                libc::_exit(0);
            }
        }
        child_pid => {
            // Parent. Restore signal mask, close the wrong ends,
            // record the container in `guard.children` so its
            // lifecycle (kill on bail / collect on stop) is shared
            // with conventional workers.
            let psm_parent_restore_rc = unsafe {
                libc::pthread_sigmask(libc::SIG_SETMASK, &old_mask, std::ptr::null_mut())
            };
            if psm_parent_restore_rc != 0 {
                tracing::warn!(
                    rc = psm_parent_restore_rc,
                    "pthread_sigmask(SIG_SETMASK) failed restoring mask in parent \
                     post-pcomm-fork; SIGUSR1 stays blocked in this thread for \
                     the lifetime of the workload",
                );
            }
            unsafe {
                libc::close(report_fds[1]);
                libc::close(start_fds[0]);
            }
            // Per-group layout for sentinel distribution. Each
            // entry is `(group_idx, num_workers)`; total report
            // count is the sum of `num_workers` across the layout
            // (== `total_workers` invariant). When the parent
            // emits sentinels for a missing JSON payload, the
            // layout drives per-group_idx tagging so per-group
            // filters partition correctly even on a failure.
            let group_layout: Vec<(usize, usize)> = groups
                .iter()
                .map(|g| (g.group_idx, g.num_workers))
                .collect();
            debug_assert_eq!(
                group_layout.iter().map(|(_, n)| n).sum::<usize>(),
                total_workers,
                "spawn_pcomm_container: group_layout total must match total_workers",
            );
            guard.children.push(ForkedChild {
                pid: child_pid,
                report_fd: report_fds[0],
                start_fd: start_fds[1],
                kind: ForkedChildKind::PcommContainer {
                    groups: group_layout,
                },
            });
            Ok(())
        }
    }
}

/// Internal dispatch shape resolved from
/// [`WorkloadConfig::clone_mode`] inside [`WorkloadHandle::spawn`].
pub(super) enum Dispatch {
    Fork,
    Thread,
}

impl WorkloadHandle {
    /// Fork one thread-group leader hosting every worker in `works`
    /// as worker threads inside a single forked process. Used by
    /// `apply_setup` when a `CgroupDef` declares `pcomm` — every
    /// `WorkSpec` in the same `CgroupDef` is coalesced into one
    /// thread-group leader whose `task->comm` carries `pcomm` (the
    /// kernel truncates to `TASK_COMM_LEN - 1 = 15` bytes inside
    /// `__set_task_comm`). Every spawned thread reads its
    /// `task->group_leader->comm` as `pcomm` for the leader's
    /// lifetime.
    ///
    /// `works` must already be fully resolved: each entry's
    /// `num_workers` must be `Some(_)` and `affinity` must be
    /// non-topology-aware (Inherit / Exact / RandomSubset).
    /// `apply_setup` runs the standard scenario-engine resolution
    /// (`resolve_num_workers` + `intent_for_spawn`) before calling
    /// in.
    ///
    /// `container_uid` / `container_gid` apply to the leader's
    /// process credentials via `setresuid` / `setresgid` once,
    /// inside the forked leader, before threads spawn. Each worker
    /// thread additionally re-applies its merged uid/gid inside
    /// `worker_main` at thread creation time; for the common case
    /// of a single CgroupDef-level default flowing through
    /// `merged_works`, the per-thread call is idempotent.
    ///
    /// On `works.is_empty()` the function returns a handle with no
    /// children — no fork, no resource allocation. Same for the
    /// empty-thread case across ALL groups (every `num_workers ==
    /// 0`): the leader is skipped because there is no work to host.
    /// This matches the `pcomm_with_zero_workers_skips_container`
    /// contract.
    ///
    /// Group-level admission rejects WorkType variants that conflict
    /// with the threaded shape (every worker shares the leader's
    /// tgid):
    ///
    /// - [`WorkType::ForkExit`]: a fork from a thread of a
    ///   multi-threaded process inherits all locks held by other
    ///   threads at fork time, which the child cannot release;
    ///   safe-but-degraded use cases would still need
    ///   `CloneMode::Fork` for clean lock-free child state.
    /// - [`WorkType::CgroupChurn`]: writing the worker tid to
    ///   `cgroup.procs` migrates the entire leader tgid (every
    ///   sibling thread) — the test loses control of its own
    ///   cgroup placement.
    pub fn spawn_pcomm_cgroup(
        pcomm: &str,
        container_uid: Option<u32>,
        container_gid: Option<u32>,
        works: &[WorkSpec],
    ) -> Result<Self> {
        // Empty pcomm input is treated as "no pcomm" by the
        // apply_setup caller; reject here as a safety belt — every
        // production caller has already filtered empty strings.
        if pcomm.is_empty() {
            anyhow::bail!(
                "spawn_pcomm_cgroup: pcomm must be a non-empty string; \
                 the caller (apply_setup) treats `Some(\"\")` as \
                 `None` and falls through to the conventional fork \
                 spawn — empty here is a programmer error.",
            );
        }

        // Build a `GroupParams` per `WorkSpec` using the same
        // resolver `WorkloadHandle::spawn` uses for composed
        // entries. group_idx is the entry's position in `works` (0-
        // based); the parent has no notion of "primary" inside a
        // pcomm container — every entry is a peer.
        let mut groups: Vec<GroupParams> = Vec::with_capacity(works.len());
        for (i, spec) in works.iter().enumerate() {
            let num_workers = spec.num_workers.ok_or_else(|| {
                anyhow::anyhow!(
                    "spawn_pcomm_cgroup: works[{i}].num_workers must be set explicitly \
                     (apply_setup runs resolve_num_workers before calling in)",
                )
            })?;
            let site = format!("works[{i}].affinity");
            let affinity = GroupParams::resolve_spawn_affinity(&spec.affinity, &site)?;
            groups.push(GroupParams::from_work_spec(spec, i, affinity, num_workers));
        }

        // Per-group admission. Mirrors `WorkloadHandle::spawn` for
        // shared rules (worker_group_size divisibility,
        // chain-depth-< 2, IdleChurn / IpcVariance zero rejection)
        // and adds pcomm-specific rejections (ForkExit,
        // CgroupChurn) since the container's threaded shape —
        // every worker is a thread of one forked tgid sharing one
        // fd table — introduces hazards those variants cannot
        // tolerate.
        for group in &groups {
            if matches!(group.work_type, WorkType::ForkExit) {
                anyhow::bail!(
                    "WorkSpec::pcomm is incompatible with WorkType::ForkExit \
                     (works[{}]): a fork from a thread of a multi-threaded \
                     container inherits all locks held by sibling threads at \
                     fork time, producing undefined behaviour for any libc \
                     primitive in the child. Drop pcomm or pick a different \
                     work type.",
                    group.group_idx,
                );
            }
            if matches!(group.work_type, WorkType::CgroupChurn { .. }) {
                anyhow::bail!(
                    "WorkSpec::pcomm is incompatible with WorkType::CgroupChurn \
                     (works[{}]): CgroupChurn writes the worker tid to \
                     `cgroup.procs`, which the kernel resolves to the whole \
                     tgid and migrates the entire pcomm container (every \
                     sibling thread). Drop pcomm or pick a different work type.",
                    group.group_idx,
                );
            }
            validate_workload_admission(group)?;
        }

        // No-work shortcut: every group has zero workers (or the
        // works list itself is empty). Fork would produce a
        // container that blocks on the start byte forever and
        // outlives the test handle; skip it. The handle still
        // has a sensible Drop (empty children/threads/regions).
        let total_workers: usize = groups.iter().map(|g| g.num_workers).sum();
        if total_workers == 0 {
            return Ok(SpawnGuard::new().into_handle());
        }

        // All failable acquisitions route through `guard`. If any
        // `?` returns early, the guard's Drop SIGKILLs+reaps any
        // already-forked container, closes open pipe fds, and
        // munmaps the shared regions — so no leak on a mid-spawn
        // error path.
        let mut guard = SpawnGuard::new();

        // Per-worker iteration counter region (MAP_SHARED). Sized
        // for ALL groups' workers laid out contiguously; matches
        // the `WorkloadHandle::spawn` allocation but with a
        // pcomm-specific diagnostic prefix.
        let size = total_workers * std::mem::size_of::<AtomicU64>();
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            let errno = std::io::Error::last_os_error();
            let hint = mmap_shared_anon_errno_hint(errno.raw_os_error());
            anyhow::bail!(
                "mmap(MAP_SHARED|MAP_ANONYMOUS, {size} bytes) for the \
                 per-worker iter_counters region failed: {errno}{hint}; \
                 this region holds one AtomicU64 per worker thread \
                 ({total_workers} thread(s) inside the pcomm={pcomm:?} \
                 container) so the parent can snapshot iteration \
                 counts via `snapshot_iterations()`.",
            );
        }
        guard.iter_counters = ptr as *mut AtomicU64;
        guard.iter_counter_bytes = size;

        // Per-group resource allocation (pipe pairs, chain pipes,
        // futex regions). Same shape as `WorkloadHandle::spawn_group`'s
        // prologue but inlined here so the per-group bookkeeping
        // (`PcommGroupResources`) is built directly for
        // `spawn_pcomm_container`. `iter_offset` runs across groups
        // and tracks each group's first iter_slot.
        let mut resources: Vec<PcommGroupResources> = Vec::with_capacity(groups.len());
        let mut iter_offset: usize = 0;
        for group in &groups {
            let needs_pipes = matches!(
                group.work_type,
                WorkType::PipeIo { .. } | WorkType::CachePipe { .. }
            );
            let chain_depth = group.work_type.chain_pipe_depth();
            let needs_futex = group.work_type.needs_shared_mem();
            let pipe_pair_base = guard.pipe_pairs.len();
            let chain_pipes_base = guard.chain_pipes.len();
            let futex_ptrs_base = guard.futex_ptrs.len();
            let futex_region_size = futex_region_size_for(&group.work_type);
            let futex_group_size = group.work_type.worker_group_size().unwrap_or(2);

            if needs_pipes {
                for _ in 0..group.num_workers / 2 {
                    let mut ab = [0i32; 2];
                    if unsafe { libc::pipe2(ab.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
                        anyhow::bail!(
                            "pipe2 (pcomm={pcomm:?}, group {}) failed: {}",
                            group.group_idx,
                            std::io::Error::last_os_error(),
                        );
                    }
                    let mut ba = [0i32; 2];
                    if unsafe { libc::pipe2(ba.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
                        unsafe {
                            libc::close(ab[0]);
                            libc::close(ab[1]);
                        }
                        anyhow::bail!(
                            "pipe2 (pcomm={pcomm:?}, group {}) failed: {}",
                            group.group_idx,
                            std::io::Error::last_os_error(),
                        );
                    }
                    guard.pipe_pairs.push((ab, ba));
                }
            }

            if let Some(depth) = chain_depth
                && depth > 0
                && group.num_workers >= depth
            {
                let chains = group.num_workers / depth;
                for _ in 0..chains {
                    let mut chain: Vec<[i32; 2]> = Vec::with_capacity(depth);
                    let mut alloc_ok = true;
                    for _ in 0..depth {
                        let mut p = [0i32; 2];
                        if unsafe { libc::pipe2(p.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
                            alloc_ok = false;
                            break;
                        }
                        chain.push(p);
                    }
                    if !alloc_ok {
                        for p in &chain {
                            unsafe {
                                libc::close(p[0]);
                                libc::close(p[1]);
                            }
                        }
                        anyhow::bail!(
                            "WakeChain pipe2 (pcomm={pcomm:?}, group {}) failed: {}",
                            group.group_idx,
                            std::io::Error::last_os_error(),
                        );
                    }
                    guard.chain_pipes.push(chain);
                }
            }

            if needs_futex {
                for _ in 0..group.num_workers / futex_group_size {
                    let region = unsafe {
                        libc::mmap(
                            std::ptr::null_mut(),
                            futex_region_size,
                            libc::PROT_READ | libc::PROT_WRITE,
                            libc::MAP_SHARED | libc::MAP_ANONYMOUS,
                            -1,
                            0,
                        )
                    };
                    if region == libc::MAP_FAILED {
                        let errno = std::io::Error::last_os_error();
                        let hint = mmap_shared_anon_errno_hint(errno.raw_os_error());
                        anyhow::bail!(
                            "mmap(MAP_SHARED|MAP_ANONYMOUS, {futex_region_size} bytes) \
                             for a futex shared-memory region (pcomm={pcomm:?}, group {}) \
                             failed: {errno}{hint}",
                            group.group_idx,
                        );
                    }
                    unsafe {
                        std::ptr::write_bytes(region as *mut u8, 0, futex_region_size);
                    }
                    guard.futex_ptrs.push(region as *mut u32);
                    guard.futex_region_sizes.push(futex_region_size);
                }
            }

            resources.push(PcommGroupResources {
                iter_offset,
                pipe_pair_base,
                chain_pipes_base,
                futex_ptrs_base,
                needs_pipes,
                chain_depth,
                needs_futex,
                futex_group_size,
            });
            iter_offset += group.num_workers;
        }

        // Fork the container and spawn its threads. On success the
        // container is registered in `guard.children`; on failure the
        // guard's Drop reaps it.
        spawn_pcomm_container(
            &mut guard,
            pcomm,
            container_uid,
            container_gid,
            &groups,
            &resources,
        )?;

        Ok(guard.into_handle())
    }

    /// Spawn worker tasks. Workers block until
    /// [`start()`](Self::start) is called, allowing the caller to
    /// move fork-mode workers into cgroups first. The worker creation
    /// primitive (`fork` or `std::thread::spawn`) is selected by
    /// [`WorkloadConfig::clone_mode`].
    pub fn spawn(config: &WorkloadConfig) -> Result<Self> {
        let dispatch = match &config.clone_mode {
            CloneMode::Fork => Dispatch::Fork,
            CloneMode::Thread => Dispatch::Thread,
        };

        // Build the per-group params list: primary first
        // (group_idx == 0), then composed[k] resolved into
        // group_idx == k+1. The resolver enforces the "spawn-time
        // resolution rules" documented on
        // [`WorkloadConfig::composed`] (Q1: num_workers must be
        // explicit; Q2: only Inherit/Exact affinity reachable from
        // spawn() — topology-aware variants need the scenario
        // engine).
        //
        // Every group inherits the parent
        // [`WorkloadConfig::clone_mode`]: SpawnGuard's lifecycle
        // assumes a single dispatch path (every guard.children
        // entry is a fork-mode child reaped via waitpid; every
        // guard.threads entry is a thread-mode worker joined via
        // JoinHandle). Mixing modes inside one guard would route
        // teardown through the wrong code path, so [`WorkSpec`]
        // carries no `clone_mode` field — it is a workload-wide
        // property fixed by [`WorkloadConfig::clone_mode`].
        let mut groups: Vec<GroupParams> = Vec::with_capacity(1 + config.composed.len());
        groups.push(GroupParams::primary(config)?);
        for (i, spec) in config.composed.iter().enumerate() {
            groups.push(GroupParams::from_composed(spec, i + 1)?);
        }

        // Per-group admission. Each group's work_type is checked
        // independently — a malformed composed entry bails the
        // whole workload before any resources are acquired.
        for group in &groups {
            // Thread mode + ForkExit is incompatible. ForkExit's worker
            // body calls `libc::fork()` from inside `worker_main` to
            // exercise wake_up_new_task / do_exit / wait_task_zombie;
            // under [`CloneMode::Thread`] the worker is a thread inside
            // the parent's tgid, so its `fork()` produces a child that
            // shares tgid with the parent and every sibling thread. The
            // child then calls `libc::_exit(0)` which the kernel routes
            // through `do_exit` — and `do_exit` for a thread-group leader
            // tears down the whole tgid (every worker thread dies). This
            // converts the workload into a fratricidal sibling kill on
            // the very first ForkExit iteration. Reject at spawn time
            // with an actionable diagnostic; CloneMode::Fork is the
            // correct choice for ForkExit and will continue to work.
            if matches!(dispatch, Dispatch::Thread) && matches!(group.work_type, WorkType::ForkExit)
            {
                anyhow::bail!(
                    "CloneMode::Thread is incompatible with WorkType::ForkExit \
                     (group {}) — ForkExit forks inside the worker, which under \
                     a thread-group worker tears down every sibling thread on \
                     the child's _exit. Use CloneMode::Fork for ForkExit workloads.",
                    group.group_idx,
                );
            }
            // Fork mode + EpollStorm is incompatible. EpollStorm
            // creates an eventfd + epoll fd inside worker pos 0 and
            // publishes their integer fd numbers through the per-group
            // shared mmap region (`efd_slot` / `epfd_slot`); siblings
            // load those numbers and operate on them as if they
            // referred to the same kernel objects. Under
            // [`CloneMode::Fork`] each forked child holds its own copy
            // of the parent's fd table at fork time, but the eventfd
            // and epoll fd are created AFTER the fork on worker pos 0
            // — so sibling children's fd tables never contain those
            // descriptors. The integer numbers they read from the
            // shared region either resolve to unrelated fds the child
            // happened to have at the same slot or fail with EBADF.
            // The fd table is genuinely shared only under
            // [`CloneMode::Thread`]. Reject at spawn time with an
            // actionable diagnostic; CloneMode::Thread is the correct
            // choice for EpollStorm.
            if matches!(dispatch, Dispatch::Fork)
                && matches!(group.work_type, WorkType::EpollStorm { .. })
            {
                anyhow::bail!(
                    "CloneMode::Fork is incompatible with WorkType::EpollStorm \
                     (group {}) — EpollStorm publishes eventfd/epoll fd numbers \
                     through a shared mmap region for siblings to consume, but \
                     forked children hold independent fd tables that never \
                     contain those post-fork descriptors. Use CloneMode::Thread \
                     for EpollStorm workloads.",
                    group.group_idx,
                );
            }
            // Thread mode + CgroupChurn is incompatible. CgroupChurn's
            // worker body writes its own tid (`SYS_gettid`) to a
            // sibling cgroup's `cgroup.procs` file. The kernel's
            // `__cgroup_procs_write` resolves the tid to its
            // task_struct, then migrates the *entire* thread group
            // leader's task and every member of its tgid to the
            // target cgroup (see `cgroup_attach_task` /
            // `cgroup_migrate` in kernel/cgroup/cgroup.c — procs-file
            // semantics are tgid-wide, contrast `cgroup.threads`
            // which is per-thread under the threaded controller).
            // Under [`CloneMode::Thread`] every worker is a member of
            // the test harness's tgid, so the first CgroupChurn write
            // migrates the harness itself and every sibling worker
            // thread out from under the host. Reject at spawn time
            // with an actionable diagnostic; CloneMode::Fork gives
            // each worker its own tgid so a procs-file write moves
            // only that worker.
            if matches!(dispatch, Dispatch::Thread)
                && matches!(group.work_type, WorkType::CgroupChurn { .. })
            {
                anyhow::bail!(
                    "CloneMode::Thread is incompatible with WorkType::CgroupChurn \
                     (group {}) — CgroupChurn writes the worker tid to \
                     `cgroup.procs`, which the kernel resolves to the whole tgid \
                     and migrates every sibling thread (including the harness) \
                     to the target cgroup. Use CloneMode::Fork for CgroupChurn \
                     workloads so each worker is a separate tgid.",
                    group.group_idx,
                );
            }
            // Dispatch-agnostic admission rules
            // (worker_group_size divisibility, chain depth >= 2,
            // IdleChurn / IpcVariance zero-rejections) live in
            // [`validate_workload_admission`] and are shared with
            // [`Self::spawn_pcomm_cgroup`]. The
            // CloneMode-vs-WorkType compat checks above stay
            // dispatch-specific because their reasoning differs
            // per dispatch (Thread+ForkExit vs pcomm+ForkExit are
            // rejected for different reasons; Fork+EpollStorm has
            // no pcomm equivalent).
            validate_workload_admission(group)?;
        }

        // futex region sizing is per-group, not MAX'd across all
        // groups. Each group's futex region has its own natural
        // size determined by [`futex_region_size_for`] (FanOutCompute
        // = 16, ProducerConsumerImbalance = 32 + Q*8, everything
        // else = 4). Storing the size alongside each pointer in
        // `SpawnGuard::futex_region_sizes` lets Drop munmap each
        // region with its own length, so a small-variant group
        // composed alongside a large ProducerConsumerImbalance group
        // is no longer inflated to the large size.
        //
        // The kernel rounds munmap length up to PAGE_SIZE, so the
        // per-region waste for sub-page allocations is bounded at
        // one page; the previous MAX-across-groups approach could
        // waste many pages per small-variant group when paired with
        // a large queue_depth_target.

        // All failable acquisitions in this function route through
        // `guard`. If any `?`/`bail!` returns early, the guard's Drop
        // SIGKILLs+reaps forked children, closes open pipe fds, and
        // munmaps the shared regions — so no leak on a mid-spawn
        // error path.
        let mut guard = SpawnGuard::new();

        // Per-worker iteration counter region (MAP_SHARED). Sized
        // for ALL groups' workers laid out contiguously: primary
        // group occupies slots `[0, primary.num_workers)`, composed
        // group `k` occupies slots starting at the running offset
        // tracked by `iter_offset` in the per-group spawn loop
        // below. Each worker atomically stores its iteration count
        // to its assigned slot; the parent reads all slots via
        // `snapshot_iterations()`. The mmap base is page-aligned
        // (kernel guarantee), so casting to `*mut AtomicU64` is
        // sound: page alignment (≥ 4096) ≥ AtomicU64 alignment (8),
        // and the region size is an exact multiple of
        // `size_of::<AtomicU64>()` (== 8). Each `.add(i)` moves by
        // `i * 8` bytes, preserving the 8-byte alignment invariant.
        // No non-atomic access to the region exists anywhere in the
        // crate, so the atomic-only aliasing rule (workers + parent
        // share `&AtomicU64` references derived from the raw
        // pointer) holds.
        let total_workers: usize = groups.iter().map(|g| g.num_workers).sum();
        if total_workers > 0 {
            let size = total_workers * std::mem::size_of::<AtomicU64>();
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            if ptr == libc::MAP_FAILED {
                let errno = std::io::Error::last_os_error();
                let hint = mmap_shared_anon_errno_hint(errno.raw_os_error());
                anyhow::bail!(
                    "mmap(MAP_SHARED|MAP_ANONYMOUS, {size} bytes) for the \
                     per-worker iter_counters region failed: {errno}{hint}; \
                     this region holds one AtomicU64 per worker \
                     ({total_workers} slots across {} group(s)) so the parent \
                     can snapshot iteration counts via \
                     `snapshot_iterations()`. Remediation: reduce num_workers \
                     (each worker consumes 8 bytes of this region, rounded up \
                     to a page) or raise `vm.max_map_count` / the memory \
                     cgroup limit.",
                    groups.len(),
                );
            }
            guard.iter_counters = ptr as *mut AtomicU64;
            guard.iter_counter_bytes = size;
        }

        // Spawn each group in declaration order. `iter_offset`
        // tracks the running offset into the iter_counters mmap
        // (slot allocation per the layout commented above). Each
        // group's pipes / chain_pipes / futex_ptrs are appended to
        // the guard's flat vectors; we record per-group base
        // offsets so the per-worker fork loop can compute
        // global-vector indices from per-group worker indices and
        // the close-other-fds child path can iterate the full
        // guard while still identifying its own group's resources.
        let mut iter_offset: usize = 0;
        for group in &groups {
            Self::spawn_group(&mut guard, group, &dispatch, iter_offset)?;
            iter_offset += group.num_workers;
        }

        // Success: transfer every live resource (children, threads,
        // futex_ptrs, iter_counters, pipe_pairs, chain_pipes) into
        // the handle. The guard's subsequent Drop sees empty Vecs
        // and a null iter_counters pointer — it is a no-op on this
        // path. Pipe fds are closed by `WorkloadHandle::drop` AFTER
        // worker shutdown so Thread-mode workers (which share the
        // parent's fd table) finish their pipe ops before the close.
        Ok(guard.into_handle())
    }

    /// Spawn a single worker group's resources and per-worker
    /// tasks, appending each into the shared [`SpawnGuard`].
    ///
    /// Each group records its own base offsets into the guard's
    /// flat vectors at entry time, then uses those offsets when
    /// computing per-worker `pair_idx` / `chain_idx` /
    /// `futex_group_idx`. The fork-child close-other-fds block
    /// iterates the FULL guard so it sweeps fds belonging to
    /// other groups too — without that sweep, a composed-group
    /// worker would inherit (and never close) every primary-group
    /// pipe fd.
    ///
    /// Resource ownership is uniform across groups: every
    /// allocated pipe / mmap region lives in the guard's flat
    /// vectors and is freed by `SpawnGuard::Drop` on early-bail or
    /// transferred to [`WorkloadHandle`] on success via
    /// [`SpawnGuard::into_handle`].
    #[allow(clippy::too_many_arguments)]
    fn spawn_group(
        guard: &mut SpawnGuard,
        group: &GroupParams,
        dispatch: &Dispatch,
        iter_offset: usize,
    ) -> Result<()> {
        let needs_pipes = matches!(
            group.work_type,
            WorkType::PipeIo { .. } | WorkType::CachePipe { .. }
        );
        let chain_depth = group.work_type.chain_pipe_depth();
        let needs_futex = group.work_type.needs_shared_mem();

        // Record the bases into the guard's flat vectors BEFORE
        // appending this group's allocations. The base values
        // identify "where this group's resources start" — the
        // per-worker fork loop combines `pipe_pair_base + i / 2`
        // (and analogous for chain_idx / futex_group_idx) to
        // address its own resources without colliding with another
        // group's range.
        let pipe_pair_base = guard.pipe_pairs.len();
        let chain_pipes_base = guard.chain_pipes.len();
        let futex_ptrs_base = guard.futex_ptrs.len();
        // Per-group natural size for the futex MAP_SHARED region —
        // each region in this group's range gets exactly this many
        // bytes (rather than the previous global MAX across every
        // group). See `futex_region_size_for` for the per-variant
        // sizing rules.
        let futex_region_size = futex_region_size_for(&group.work_type);

        // For paired work types, create one pipe per worker pair before forking.
        // pipe_pairs[pair_idx] = (read_fd, write_fd) for the A->B direction,
        // and a second pipe for B->A. Use `pipe2(O_CLOEXEC)` instead
        // of bare `pipe(2)`: O_CLOEXEC is the correct default for
        // any kernel fd in long-running processes — fds without
        // O_CLOEXEC silently leak into any exec path.
        if needs_pipes {
            for _ in 0..group.num_workers / 2 {
                let mut ab = [0i32; 2]; // A writes, B reads
                if unsafe { libc::pipe2(ab.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
                    anyhow::bail!("pipe2 failed: {}", std::io::Error::last_os_error());
                }
                let mut ba = [0i32; 2]; // B writes, A reads
                if unsafe { libc::pipe2(ba.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
                    // Close the ab half we just created: it is not
                    // yet owned by the guard, so its Drop won't
                    // otherwise reach it.
                    unsafe {
                        libc::close(ab[0]);
                        libc::close(ab[1]);
                    }
                    anyhow::bail!("pipe2 failed: {}", std::io::Error::last_os_error());
                }
                guard.pipe_pairs.push((ab, ba));
            }
        }

        // For WakeChain { wake: WakeMechanism::Pipe }, allocate `depth` pipes per
        // chain (one pipe per stage). Pipe `i` connects stage `i`
        // (writer) to stage `(i + 1) % depth` (reader). On any
        // `pipe2()` failure mid-allocation, close the half-built
        // chain's pipes before bailing — the chain is not yet
        // pushed onto `guard.chain_pipes`, so its Drop won't
        // otherwise reach those fds. `O_CLOEXEC` matches the
        // defense-in-depth posture documented above on the
        // pipe-pair allocation.
        if let Some(depth) = chain_depth
            && depth > 0
            && group.num_workers >= depth
        {
            let chains = group.num_workers / depth;
            for _ in 0..chains {
                let mut chain: Vec<[i32; 2]> = Vec::with_capacity(depth);
                let mut alloc_ok = true;
                for _ in 0..depth {
                    let mut p = [0i32; 2];
                    if unsafe { libc::pipe2(p.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
                        alloc_ok = false;
                        break;
                    }
                    chain.push(p);
                }
                if !alloc_ok {
                    for p in &chain {
                        unsafe {
                            libc::close(p[0]);
                            libc::close(p[1]);
                        }
                    }
                    anyhow::bail!(
                        "WakeChain pipe2 allocation failed: {}",
                        std::io::Error::last_os_error()
                    );
                }
                guard.chain_pipes.push(chain);
            }
        }

        // For FutexPingPong/FutexFanOut/FanOutCompute/MutexContention, allocate
        // one shared region per worker group via MAP_SHARED|MAP_ANONYMOUS
        // so all members of the fork see the same physical page.
        // Each region is sized exactly to the variant's natural
        // need (see [`futex_region_size_for`]) — the per-region
        // size is recorded in `guard.futex_region_sizes` parallel
        // to `guard.futex_ptrs`, so munmap on Drop receives the
        // correct length even when groups with different natural
        // sizes co-exist in the same workload.
        let futex_group_size = group.work_type.worker_group_size().unwrap_or(2);
        if needs_futex {
            for _ in 0..group.num_workers / futex_group_size {
                let ptr = unsafe {
                    libc::mmap(
                        std::ptr::null_mut(),
                        futex_region_size,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_SHARED | libc::MAP_ANONYMOUS,
                        -1,
                        0,
                    )
                };
                if ptr == libc::MAP_FAILED {
                    let errno = std::io::Error::last_os_error();
                    let hint = mmap_shared_anon_errno_hint(errno.raw_os_error());
                    anyhow::bail!(
                        "mmap(MAP_SHARED|MAP_ANONYMOUS, {futex_region_size} bytes) \
                         for a futex shared-memory region failed: {errno}{hint}; \
                         this region backs the {:?} worker-group's (group {}) \
                         inter-process futex word and is allocated \
                         before fork so every child inherits the same \
                         mapping. Remediation: reduce num_workers (each \
                         futex group consumes one shared page) or raise \
                         `vm.max_map_count` / the memory cgroup limit.",
                        group.work_type.name(),
                        group.group_idx,
                    );
                }
                unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, futex_region_size) };
                guard.futex_ptrs.push(ptr as *mut u32);
                guard.futex_region_sizes.push(futex_region_size);
            }
        }

        for i in 0..group.num_workers {
            let affinity = resolve_affinity(&group.affinity)?;

            // Determine pipe fds for this worker.
            //
            // Three shapes use the same `Option<(read_fd, write_fd)>`
            // parameter:
            // - PipeIo / CachePipe (paired): worker A reads `ba[0]`,
            //   writes `ab[1]`; worker B reads `ab[0]`, writes
            //   `ba[1]`.
            // - WakeChain { wake: WakeMechanism::Pipe } (chain ring): stage `s`
            //   reads from pipe `(s + depth - 1) % depth` (its
            //   predecessor's write end's matching read end) and
            //   writes to pipe `s` (its own pipe's write end, which
            //   stage `s + 1` reads from).
            // - Everything else: `None`.
            //
            // Indices are computed in the GLOBAL `guard.pipe_pairs`
            // / `guard.chain_pipes` space by adding the per-group
            // base recorded at the top of `spawn_group`. A composed
            // group's pipe-pair-base, for example, equals the sum
            // of every prior group's pipe-pair count, so its first
            // worker pair is allocated immediately after the
            // primary's last entry — no collision, no aliasing.
            let worker_pipe_fds: Option<(i32, i32)> = if needs_pipes {
                let pair_idx = pipe_pair_base + i / 2;
                let (ref ab, ref ba) = guard.pipe_pairs[pair_idx];
                if i % 2 == 0 {
                    // Worker A: writes to ab[1], reads from ba[0]
                    Some((ba[0], ab[1]))
                } else {
                    // Worker B: writes to ba[1], reads from ab[0]
                    Some((ab[0], ba[1]))
                }
            } else if let Some(depth) = chain_depth
                && depth > 0
            {
                let chain_idx = chain_pipes_base + i / depth;
                let stage = i % depth;
                let prev_stage = (stage + depth - 1) % depth;
                let chain = &guard.chain_pipes[chain_idx];
                // Read end of predecessor's pipe; write end of own
                // pipe. The kernel pipe pair is `[read_end,
                // write_end]` per `libc::pipe`'s manpage.
                Some((chain[prev_stage][0], chain[stage][1]))
            } else {
                None
            };

            // Futex pointer for this worker. The `pos` is the
            // worker's index inside its futex group: `pos == 0`
            // is the group's "first" worker (the role that varies
            // per-variant — pair-A for FutexPingPong, messenger for
            // FutexFanOut/FanOutCompute, waker for ThunderingHerd/
            // AsymmetricWaker, chain-head for WakeChain). Variants
            // that need finer-grained per-worker positioning
            // (PriorityInversion's 3 tiers, ProducerConsumerImbalance's
            // producer/consumer split, RtStarvation's RT/CFS split,
            // WakeChain's stage index) consume `pos` directly.
            let worker_futex: Option<(*mut u32, usize)> = if needs_futex {
                let futex_group_idx = futex_ptrs_base + i / futex_group_size;
                let pos = i % futex_group_size;
                Some((guard.futex_ptrs[futex_group_idx], pos))
            } else {
                None
            };

            // Shared iteration counter slot for this worker. The
            // group-local index `i` is added to the spawn-time
            // `iter_offset` so each group's slot range is disjoint
            // from every other group's.
            let iter_slot: *mut AtomicU64 = if !guard.iter_counters.is_null() {
                unsafe { guard.iter_counters.add(iter_offset + i) }
            } else {
                std::ptr::null_mut()
            };

            // Per-mode dispatch. Thread-mode workers do not need
            // pipes — the rendezvous and report channels are
            // in-process Rust primitives (`mpsc::sync_channel(0)` +
            // `JoinHandle`). Fork mode uses the pipe-based
            // scaffolding below.
            match dispatch {
                Dispatch::Thread => {
                    spawn_thread_worker(
                        guard,
                        group,
                        affinity,
                        worker_pipe_fds,
                        worker_futex,
                        iter_slot,
                    )?;
                    continue;
                }
                Dispatch::Fork => {
                    // fall through to the pipe-based dispatch below
                }
            }

            // Create pipe for report and a second pipe for "start" signal.
            // Local cleanup on second-pipe failure: the guard has no
            // per-worker tracking of half-allocated pipes, so the first
            // half closes here before the bail. `O_CLOEXEC` matches
            // the defense-in-depth posture above on the inter-worker
            // pipe pairs and chain pipes.
            let mut report_fds = [0i32; 2];
            if unsafe { libc::pipe2(report_fds.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
                anyhow::bail!(
                    "worker {}/{} (group {}): report pipe2 failed: {}",
                    i + 1,
                    group.num_workers,
                    group.group_idx,
                    std::io::Error::last_os_error(),
                );
            }
            // Grow the report pipe so a worker that produces a worst-
            // case report (two `Vec<u64>` reservoirs capped at
            // `MAX_WAKE_SAMPLES` (100_000) entries each, plus
            // smaller fields) finishes `write_all` without blocking
            // for the parent to drain. The default pipe capacity on
            // Linux is 16 pages (64 KiB on 4 KiB pages, 256 KiB on
            // 16 KiB pages); a bincode-encoded `WorkerReport` with
            // both reservoirs full is at most ~1.8 MiB (u64 varint
            // is up to 9 bytes per entry, 200_000 entries total).
            // Without this grow, the worker blocks inside
            // `f.write_all(&bytes)` waiting for the parent's
            // `read_to_end`. The parent reads workers serially with
            // a shared 5 s deadline (`stop_and_collect`); if the
            // first worker's drain consumes the budget, every
            // subsequent worker's `poll` budget is 0, the parent
            // closes the read fd without reading, and the child's
            // blocked write returns EPIPE — the report is lost and
            // the failure surfaces as a sentinel `WorkerReport`.
            // Sizing the pipe to 8 MiB lets the child write the
            // full payload in one non-blocking burst, exit, and
            // close its write end; when the parent later polls the
            // read fd, all data is already buffered in the kernel
            // and `read_to_end` returns immediately with EOF.
            //
            // F_SETPIPE_SZ rounds the requested size up to the next
            // power of two; unprivileged callers are clamped to
            // `/proc/sys/fs/pipe-max-size` (typically 1 MiB).
            // ktstr always runs as root inside the guest, so the
            // grow succeeds. A best-effort failure (older kernel
            // without F_SETPIPE_SZ, or an unexpected EPERM) leaves
            // the default size in place — workers with small
            // reports still complete; workers with large reports
            // fall back to the historical blocking-write behaviour.
            // Logging the failure surfaces the regression without
            // failing the whole spawn.
            const REPORT_PIPE_SIZE: libc::c_int = 8 * 1024 * 1024;
            // SAFETY: `report_fds[1]` is the freshly opened write
            // end returned by `pipe2` above; F_SETPIPE_SZ accepts
            // any pipe fd (read or write end — both refer to the
            // same kernel `struct pipe_inode_info`).
            let prev_size =
                unsafe { libc::fcntl(report_fds[1], libc::F_SETPIPE_SZ, REPORT_PIPE_SIZE) };
            if prev_size < 0 {
                let err = std::io::Error::last_os_error();
                tracing::warn!(
                    worker = i + 1,
                    num_workers = group.num_workers,
                    group_idx = group.group_idx,
                    requested_size = REPORT_PIPE_SIZE,
                    %err,
                    "F_SETPIPE_SZ on report pipe failed; falling back to default \
                     pipe capacity. Workers producing >64 KiB reports may block \
                     in write_all and have their reports truncated by the \
                     parent's deadline-driven fd close."
                );
            }
            let mut start_fds = [0i32; 2];
            if unsafe { libc::pipe2(start_fds.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
                unsafe {
                    libc::close(report_fds[0]);
                    libc::close(report_fds[1]);
                }
                anyhow::bail!(
                    "worker {}/{} (group {}): start pipe2 failed: {}",
                    i + 1,
                    group.num_workers,
                    group.group_idx,
                    std::io::Error::last_os_error(),
                );
            }

            // Block SIGUSR1 across fork so the child inherits a
            // SIGUSR1-blocked mask. Without this, there is a
            // window between `fork()` returning in the child and
            // the child installing `sigusr1_handler` below where
            // SIGUSR1's default disposition (terminate) is in
            // effect. A `stop_and_collect` call that races against
            // a worker still in mid-init — e.g. if a sibling
            // worker spawned just before this one already pushed
            // its (pid, ...) into `guard.children` and a parent
            // thread initiated stop signaling — could deliver
            // SIGUSR1 to this child before its handler is
            // installed and terminate it via the default action.
            // The block + post-install unblock pattern queues any
            // SIGUSR1 sent in the window so the handler observes
            // it on the unblock; the worker then sets `STOP` and
            // exits gracefully.
            //
            // pthread_sigmask is the multi-threaded equivalent of
            // sigprocmask: ktstr's parent process is genuinely
            // multi-threaded (test runner threads, tracing
            // threads), and `sigprocmask` semantics are unspecified
            // in MT programs per signal-safety(7) /
            // pthread_sigmask(3) — pthread_sigmask is the correct
            // primitive to use here. The fork()-inherited mask is
            // the calling thread's mask at fork time, exactly what
            // we want.
            //
            // Old mask is captured so the parent path can restore
            // its prior signal mask after fork — the block must
            // be transient on the parent side; the parent's
            // SIGUSR1 stays under whatever disposition the host
            // application configured (typically the default;
            // ktstr does not handle SIGUSR1 in the parent
            // process).
            let mut old_mask: libc::sigset_t = unsafe { std::mem::zeroed() };
            let mut block_mask: libc::sigset_t = unsafe { std::mem::zeroed() };
            // pthread_sigmask returns 0 on success, a positive errno
            // on failure (per POSIX — does NOT set errno). A non-zero
            // return here means the SIGUSR1 mask did not actually get
            // installed: the child would inherit an unblocked SIGUSR1
            // and the post-fork race window between fork() and the
            // child's signal() install would terminate the worker on
            // its default action. Log so the failure surfaces.
            let psm_block_rc = unsafe {
                libc::sigemptyset(&mut block_mask);
                libc::sigaddset(&mut block_mask, libc::SIGUSR1);
                libc::pthread_sigmask(libc::SIG_BLOCK, &block_mask, &mut old_mask)
            };
            if psm_block_rc != 0 {
                tracing::warn!(
                    rc = psm_block_rc,
                    "pthread_sigmask(SIG_BLOCK, SIGUSR1) failed pre-fork; child inherits unblocked SIGUSR1 and may terminate on default action before installing handler"
                );
            }

            let pid = unsafe { libc::fork() };
            match pid {
                -1 => {
                    // Fork failed: restore the parent's previous
                    // signal mask before bailing so this failure
                    // doesn't leave SIGUSR1 blocked in the calling
                    // thread for the rest of the process lifetime.
                    let psm_restore_rc = unsafe {
                        libc::pthread_sigmask(libc::SIG_SETMASK, &old_mask, std::ptr::null_mut())
                    };
                    if psm_restore_rc != 0 {
                        tracing::warn!(
                            rc = psm_restore_rc,
                            "pthread_sigmask(SIG_SETMASK) failed restoring mask after fork failure; SIGUSR1 may remain blocked in this thread"
                        );
                    }
                    // Close both fresh pipes so they don't leak
                    // before the guard reaps the already-forked
                    // siblings.
                    unsafe {
                        libc::close(report_fds[0]);
                        libc::close(report_fds[1]);
                        libc::close(start_fds[0]);
                        libc::close(start_fds[1]);
                    }
                    anyhow::bail!(
                        "worker {}/{} (group {}): fork failed: {}",
                        i + 1,
                        group.num_workers,
                        group.group_idx,
                        std::io::Error::last_os_error(),
                    );
                }
                0 => {
                    // Child: set parent-death signal BEFORE any other
                    // post-fork setup so the kernel SIGKILLs this worker
                    // immediately if the parent dies during the remaining
                    // init (close fd loops, signal handler install, start-
                    // pipe wait, worker_main). Without PR_SET_PDEATHSIG,
                    // a parent crash between fork and start leaves workers
                    // reparented to init and spinning indefinitely —
                    // they'd outlive the test run, consume the cgroup's
                    // CPU, and block the next scenario's cgroup teardown
                    // with EBUSY. SIGKILL is the only safe choice: it
                    // cannot be masked and runs before any of this child's
                    // destructors execute (good — those destructors still
                    // reference the parent's guard). prctl is NOT listed
                    // as async-signal-safe by signal-safety(7); safe to
                    // call here because this is a single-threaded
                    // post-fork child before any signal handlers are
                    // installed, so no interleaving can observe partial
                    // state.
                    unsafe {
                        libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL);
                    }
                    // Fork-race close: if the parent died between fork()
                    // return and the prctl above, this child was already
                    // reparented (typically to pid 1) before PDEATHSIG
                    // was armed — the death signal is keyed on the CURRENT
                    // parent, not the parent-at-fork-time, so the signal
                    // will never fire. getppid() == 1 means we are already
                    // orphaned; exit now instead of running the full
                    // worker loop only to leak into init. Using `_exit`
                    // (async-signal-safe) rather than `exit` so Rust
                    // destructors that reference the parent's now-dead
                    // guard don't run on the fork stack.
                    //
                    // PR_SET_CHILD_SUBREAPER exception: when an ancestor
                    // of the ktstr process has called
                    // `prctl(PR_SET_CHILD_SUBREAPER, 1)` (systemd user
                    // scopes, some container runtimes, certain CI
                    // harnesses), orphaned descendants reparent to the
                    // nearest live subreaper rather than pid 1. In that
                    // case `getppid() == 1` is false after an orphan-
                    // race even though the original parent is dead —
                    // the check is a best-effort fast-path for the
                    // common "pid 1 catches orphans" case and does NOT
                    // fire under subreaper ancestry. PDEATHSIG still
                    // fires correctly in that scenario (the signal is
                    // triggered when the CURRENT parent dies, and the
                    // subreaper then inherits us), so the guard is a
                    // narrowing of the leak window, not an elimination.
                    //
                    // Guest-init exception: inside a ktstr guest VM the
                    // test driver IS pid 1 (it runs as /init), so every
                    // worker forked by a scenario legitimately has
                    // `getppid() == 1` even though the parent is alive
                    // and well. Firing the orphan guard there would kill
                    // every worker on startup and produce sentinel
                    // "0 cpus, 0 iterations" reports. `ktstr_guest_init`
                    // sets `KTSTR_GUEST_INIT=1` before dispatch; that
                    // variable is inherited by every descendant process,
                    // so its presence is a reliable signal that pid 1 is
                    // the legitimate parent. Host-side workloads leave
                    // the variable unset and retain the orphan detection.
                    if std::env::var_os("KTSTR_GUEST_INIT").is_none()
                        && unsafe { libc::getppid() } == 1
                    {
                        unsafe {
                            libc::_exit(0);
                        }
                    }
                    // Make this worker its own process-group leader so
                    // any descendants it spawns inherit `pgid == worker_pid`.
                    // `stop_and_collect` / Drop issue `killpg(worker_pid,
                    // SIGKILL)` alongside the direct `kill` — without a
                    // private pgid, descendants forked by a
                    // [`WorkType::Custom`] body (or any future workload
                    // that spawns helpers) stay in the parent Rust
                    // process's pgid, survive the worker's SIGKILL, and
                    // orphan onto init. PR_SET_PDEATHSIG handles the
                    // "parent crashes" case but is per-task and cleared
                    // on fork, so grandchildren don't inherit it — the
                    // pgid route is the only safe reach for them when
                    // teardown is explicit. Failure is silently ignored:
                    // the only reachable failure mode for setpgid(0, 0)
                    // in a just-forked child is EPERM from an earlier
                    // setsid (we never call it), so a return of -1 here
                    // means the kernel invariant changed and the reach
                    // degrades to "leader only" — same as the pre-batch
                    // behavior. Async-signal-safe per signal-safety(7).
                    unsafe {
                        libc::setpgid(0, 0);
                    }
                    // Install signal handler BEFORE unblocking
                    // SIGUSR1: the parent blocked SIGUSR1 across
                    // fork so this child inherits a blocked mask;
                    // any SIGUSR1 sent by `stop_and_collect` while
                    // the mask is in effect queues until we
                    // unblock. Once the handler is registered we
                    // restore the original (pre-block) mask via
                    // `pthread_sigmask(SIG_SETMASK, …)`, which
                    // delivers any pending SIGUSR1 directly into
                    // `sigusr1_handler` and sets `STOP` cleanly —
                    // the worker proceeds into the start-wait
                    // poll() with handler armed and any pending
                    // signal already consumed. Without this two-
                    // step, SIGUSR1 would fire under its default
                    // disposition (terminate) during the window
                    // between `fork()` and `signal()`.
                    STOP.store(false, Ordering::Relaxed);
                    // libc::signal returns SIG_ERR (cast as
                    // sighandler_t) on failure with errno set.
                    // pthread_sigmask returns 0 on success, errno
                    // value on failure. Both failures here are
                    // visible defects: signal() fail means SIGUSR1
                    // keeps default disposition (terminate) and
                    // stop_and_collect cannot stop the worker
                    // gracefully — it falls through to the
                    // killpg/SIGKILL escalation; pthread_sigmask()
                    // fail means the SIGUSR1 we blocked across
                    // fork stays blocked, so the parent's stop
                    // SIGUSR1 queues forever in the child and the
                    // child runs the full work loop until the
                    // killpg escalation.
                    //
                    // tracing is unsafe in a post-fork child
                    // before _exit (the parent's tracing
                    // subscriber may be locked elsewhere); use
                    // eprintln! which is fork-safe per
                    // signal-safety(7) (write(2)).
                    let sig_prev = unsafe {
                        libc::signal(
                            libc::SIGUSR1,
                            sigusr1_handler as *const () as libc::sighandler_t,
                        )
                    };
                    if sig_prev == libc::SIG_ERR {
                        let errno = std::io::Error::last_os_error();
                        eprintln!(
                            "ktstr: signal(SIGUSR1) install failed in worker child: {errno}; graceful stop unavailable, killpg escalation will reap"
                        );
                    }
                    let psm_unblock_rc = unsafe {
                        libc::pthread_sigmask(libc::SIG_SETMASK, &old_mask, std::ptr::null_mut())
                    };
                    if psm_unblock_rc != 0 {
                        eprintln!(
                            "ktstr: pthread_sigmask(SIG_SETMASK) unblock failed in worker child: rc={psm_unblock_rc}; SIGUSR1 stays blocked, killpg escalation will reap"
                        );
                    }
                    // Close unused pipe ends
                    unsafe {
                        libc::close(report_fds[0]);
                        libc::close(start_fds[1]);
                    }
                    // Close pipe ends belonging to other workers
                    // in this pair, AND every pipe fd that belongs
                    // to any other pair anywhere in the workload —
                    // including pairs owned by other groups, since
                    // every pre-fork allocation lives in
                    // `guard.pipe_pairs` regardless of which group
                    // declared it. The fork inherits the parent's
                    // entire fd table; without this sweep, a
                    // composed-group worker would hold open every
                    // primary-group pipe fd for its lifetime,
                    // producing fd leaks and (for chain-shaped
                    // workloads) keeping reader-side blocks live
                    // when the writer-side closes.
                    if needs_pipes {
                        let pair_idx = pipe_pair_base + i / 2;
                        let (ref ab, ref ba) = guard.pipe_pairs[pair_idx];
                        if i % 2 == 0 {
                            // Worker A keeps ba[0] (read) and ab[1] (write).
                            // Close ab[0] and ba[1].
                            unsafe {
                                libc::close(ab[0]);
                                libc::close(ba[1]);
                            }
                        } else {
                            // Worker B keeps ab[0] (read) and ba[1] (write).
                            // Close ab[1] and ba[0].
                            unsafe {
                                libc::close(ab[1]);
                                libc::close(ba[0]);
                            }
                        }
                        // Close all pipe fds from other pairs (any group).
                        for (j, (ab2, ba2)) in guard.pipe_pairs.iter().enumerate() {
                            if j != pair_idx {
                                unsafe {
                                    libc::close(ab2[0]);
                                    libc::close(ab2[1]);
                                    libc::close(ba2[0]);
                                    libc::close(ba2[1]);
                                }
                            }
                        }
                    } else {
                        // Worker doesn't own any pipe pair, but
                        // other groups' pipe pairs are still in the
                        // child's fd table — close them all.
                        for (ab2, ba2) in guard.pipe_pairs.iter() {
                            unsafe {
                                libc::close(ab2[0]);
                                libc::close(ab2[1]);
                                libc::close(ba2[0]);
                                libc::close(ba2[1]);
                            }
                        }
                    }
                    if let Some(depth) = chain_depth
                        && depth > 0
                    {
                        let chain_idx = chain_pipes_base + i / depth;
                        let stage = i % depth;
                        let prev_stage = (stage + depth - 1) % depth;
                        // Close every fd in the chain that this
                        // stage does not own. Owned fds (kept open):
                        //   - chain[prev_stage][0]: read end of the
                        //     pipe predecessor writes to.
                        //   - chain[stage][1]: write end of the
                        //     pipe successor reads from.
                        // Everything else is the inverse end of an
                        // owned pipe or fully unrelated.
                        for (s, pipe) in guard.chain_pipes[chain_idx].iter().enumerate() {
                            // Always close the write end of the
                            // predecessor's pipe (we only need its
                            // read end).
                            if s == prev_stage {
                                unsafe {
                                    libc::close(pipe[1]);
                                }
                            // Always close the read end of our own
                            // pipe (we only need its write end).
                            } else if s == stage {
                                unsafe {
                                    libc::close(pipe[0]);
                                }
                            // Pipes belonging to neither this stage
                            // nor its predecessor: close both ends.
                            } else {
                                unsafe {
                                    libc::close(pipe[0]);
                                    libc::close(pipe[1]);
                                }
                            }
                        }
                        // Close every fd from other chains (any group).
                        for (cj, chain) in guard.chain_pipes.iter().enumerate() {
                            if cj != chain_idx {
                                for pipe in chain {
                                    unsafe {
                                        libc::close(pipe[0]);
                                        libc::close(pipe[1]);
                                    }
                                }
                            }
                        }
                    } else {
                        // This group has no chain pipes, but other
                        // groups may. Close every chain-pipe fd
                        // inherited via fork — leaving a primary
                        // group's chain pipe open in a composed
                        // worker would prevent the chain from ever
                        // observing EOF on its read ends.
                        for chain in guard.chain_pipes.iter() {
                            for pipe in chain {
                                unsafe {
                                    libc::close(pipe[0]);
                                    libc::close(pipe[1]);
                                }
                            }
                        }
                    }
                    // Layered defense against child-side unwinding
                    // reaching the forked-from-parent drops:
                    //
                    // 1. No-op panic hook — the default hook prints a
                    //    multi-line backtrace to stderr, which is a
                    //    shared fd with the parent post-fork. A panic
                    //    in the child would interleave garbled output
                    //    with the parent's tracing log and confuse
                    //    downstream parsers. Install a silent hook
                    //    before catch_unwind ONLY under
                    //    `panic = "unwind"` — under
                    //    `panic = "abort"` (release / nextest
                    //    profile) the silent hook would suppress the
                    //    panic message before SIGABRT fires, leaving
                    //    operators with zero diagnostic from a
                    //    crashed worker. Letting the default hook
                    //    write a backtrace to stderr is the lesser
                    //    evil under abort: the message may interleave
                    //    with parent tracing output but at least
                    //    surfaces the panic site. catch_unwind itself
                    //    is a no-op under abort (the closure is
                    //    invoked normally and the abort terminates
                    //    the child before any Err return), so gating
                    //    only the hook installation — not the
                    //    catch_unwind call — keeps the unwind-build
                    //    fast-path unchanged while restoring
                    //    release-build observability.
                    //
                    // 2. `_exit(0|1)` after the closure (success or
                    //    catch_unwind Err) — the child never returns
                    //    to a frame whose `SpawnGuard` Drop could
                    //    run. `_exit(2)` bypasses Rust's stack-unwind
                    //    drops and the static-destructor table both,
                    //    so the parent-owned `SpawnGuard` whose
                    //    storage was duplicated by `fork()` cannot
                    //    SIGKILL its siblings (fratricide) from the
                    //    child path.
                    //
                    // 3. `panic::catch_unwind` — catches any panic
                    //    before it escapes this arm. Belt-and-braces
                    //    against (a) additional Drops on this
                    //    frame's stack (e.g. future refactors that
                    //    add more RAII) and (b) alloc/OOM panics
                    //    during worker_main / bincode encode.
                    //
                    //    Caveat: catch_unwind is a no-op under
                    //    `panic = "abort"`, which ktstr's Cargo.toml
                    //    DOES set in `[profile.release]`. In release
                    //    builds a panic inside this closure aborts
                    //    the child immediately (SIGABRT); the
                    //    `catch_unwind` call compiles but never
                    //    returns `Err`, and neither the
                    //    `f.write_all(&bytes)` nor the `_exit(1)`
                    //    below runs on the panic path. The parent's
                    //    `stop_and_collect` therefore observes a
                    //    missing WorkerReport and fills in a
                    //    sentinel — that sentinel fallback IS the
                    //    release-build correctness mechanism. Under
                    //    abort the silent hook is NOT installed (see
                    //    item 1) so the default panic handler writes
                    //    the panic location and message to stderr
                    //    just before SIGABRT, giving operators the
                    //    diagnostic they would otherwise have lost.
                    //    Defense (2) still applies unchanged: abort
                    //    skips Drops (matching the `_exit` path's
                    //    no-Drop guarantee). Dev/test builds (cargo
                    //    test, cargo nextest run — dev profile
                    //    inherits default unwind semantics) still
                    //    get a real `catch_unwind` Err → `_exit(1)`
                    //    fast-path with a silent hook so the
                    //    backtrace doesn't pollute test output.
                    //
                    //    Global-state safety under unwind, scoped
                    //    to `worker_main`'s reachable code path —
                    //    the `fork()` child's observable set. Two
                    //    items: `STOP: AtomicBool` and
                    //    `STATIC_HOST_INFO: OnceLock<_>`. Neither
                    //    of them carries a Drop whose body touches
                    //    the inherited MAP_SHARED regions or the
                    //    parent-owned pipe fds. Under a
                    //    hypothetical unwind that escaped
                    //    `catch_unwind` (a double-panic that
                    //    bypasses the landing pad), the only
                    //    fork-child Drops that actually matter are
                    //    the parent-owned `SpawnGuard` (covered by
                    //    the `_exit` no-Drop guarantee above) and
                    //    the child-local `resume_latencies_ns` /
                    //    `migrations` `Vec<T>` (per-process heap,
                    //    no cross-process impact). `STATIC_HOST_INFO`'s
                    //    inner Drop frees a handful of
                    //    `Option<String>`s and is safe on either
                    //    side of fork. Crate-wide statics outside
                    //    this set (fetch, probe, vmm, …) are out
                    //    of scope — this audit pins only what the
                    //    fork-child can reach from `worker_main`.
                    //
                    // 4. `_exit(1)` on catch_unwind Err, `_exit(0)`
                    //    on Ok — bypasses Rust's global static
                    //    destructors that a plain `return` would
                    //    run.
                    //
                    // `AssertUnwindSafe` is justified: the child
                    // unconditionally _exits after this block, so no
                    // post-unwind invariant can be observed.
                    #[cfg(panic = "unwind")]
                    {
                        let _ = std::panic::take_hook();
                        std::panic::set_hook(Box::new(|_| {}));
                    }
                    let child_result =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            // Wait for parent to move us to cgroup before starting work.
                            // Use poll() with a 30s timeout — signal-safe after fork,
                            // prevents hanging forever if the parent stalls.
                            let mut pfd = libc::pollfd {
                                fd: start_fds[0],
                                events: libc::POLLIN,
                                revents: 0,
                            };
                            let ret = unsafe { libc::poll(&mut pfd, 1, 30_000) };
                            if ret <= 0 {
                                unsafe {
                                    libc::_exit(1);
                                }
                            }
                            let mut buf = [0u8; 1];
                            let mut f = unsafe { std::fs::File::from_raw_fd(start_fds[0]) };
                            let _ = f.read_exact(&mut buf);
                            drop(f);
                            // Do NOT reset STOP here. STOP is initialized to
                            // false at the pre-handshake site above (after the
                            // signal-handler install) and a SIGUSR1 delivered
                            // between the mask unblock and this point reflects
                            // a legitimate parent-issued stop request — the
                            // explicit-start path in `stop_and_collect` skips
                            // the readiness barrier and races
                            // `kill(pid, SIGUSR1)` against the worker's start-
                            // byte read here, so any STOP=true observed at
                            // this point must propagate into the work loop.
                            // Clobbering it would lose the stop and produce a
                            // worker that loops until the killpg/SIGKILL
                            // escalation.
                            // Publish a single "ready" byte on the report pipe
                            // BEFORE entering `worker_main`. The parent's
                            // auto-start barrier in `stop_and_collect` polls
                            // every worker's report fd for POLLIN with a
                            // bounded deadline and consumes this byte as the
                            // explicit signal that the worker has finished
                            // its post-fork init (cgroup placement, SIGUSR1
                            // unblock) and is about to enter the
                            // work loop. Replaces the prior 500 ms blind
                            // sleep — under host CPU contention, real worker
                            // start latency can exceed that sleep, dropping
                            // a worker into its loop AFTER `stop_and_collect`
                            // has already signalled stop, which surfaces as
                            // a starvation false-positive.
                            //
                            // Done as a raw `libc::write` (not `File::write`)
                            // to keep the report-fd's ownership inside its
                            // existing `std::fs::File::from_raw_fd` block
                            // below — wrapping it here would close the fd
                            // when the local goes out of scope, before the
                            // post-`worker_main` write_all path. A 1-byte
                            // write to a freshly-grown 8 MiB pipe with no
                            // reader contention completes synchronously
                            // without blocking; signal-safe after fork.
                            //
                            // The byte is `b'r'`. The parent's collect path
                            // strips a leading `b'r'` before
                            // `bincode::serde::decode_from_slice` so the
                            // explicit-start call site (which skips the
                            // barrier) still parses correctly. A zero return or short write
                            // is treated as best-effort: if the kernel
                            // refuses the write (EFAULT cannot occur with a
                            // stack pointer; EBADF cannot occur on a fresh
                            // fd; EPIPE only fires if the reader closed,
                            // which the parent does not do mid-spawn), the
                            // parent's barrier-deadline path falls back to
                            // the same "skip this worker" branch as a
                            // genuinely-dead worker.
                            let ready_byte: u8 = b'r';
                            unsafe {
                                libc::write(
                                    report_fds[1],
                                    &ready_byte as *const u8 as *const libc::c_void,
                                    1,
                                );
                            }
                            // Now run. Fork-mode workers thread the global
                            // STOP through `worker_main` — the SIGUSR1 handler
                            // is process-wide, so flipping `STOP` from
                            // `sigusr1_handler` is what reaches the loop's
                            // `stop.load(Relaxed)` checks.
                            let report = worker_main(
                                affinity,
                                group.work_type.clone(),
                                group.sched_policy,
                                group.mem_policy.clone(),
                                group.mpol_flags,
                                group.nice,
                                group.comm.as_deref(),
                                group.uid,
                                group.gid,
                                group.numa_node,
                                worker_pipe_fds,
                                worker_futex,
                                iter_slot,
                                &STOP,
                                group.group_idx,
                            );
                            let bytes =
                                bincode::serde::encode_to_vec(&report, bincode::config::standard())
                                    .unwrap_or_default();
                            let mut f = unsafe { std::fs::File::from_raw_fd(report_fds[1]) };
                            if let Err(e) = f.write_all(&bytes) {
                                // tracing is unsafe in a post-fork child
                                // (the global subscriber may be holding a
                                // lock taken in another thread of the
                                // parent process before fork); eprintln
                                // goes straight to stderr without locking
                                // a tracing subscriber. The parent
                                // observes a partial / empty pipe payload
                                // and emits a sentinel WorkerReport with
                                // exit_info populated, but without this
                                // log line the underlying I/O error
                                // (EPIPE if the parent closed the read
                                // end early, EFBIG / ENOSPC on a backing-
                                // file pipe, EIO on a transient kernel
                                // failure) is invisible.
                                eprintln!("ktstr: worker report write_all failed: {e}");
                            }
                            drop(f);
                        }));
                    let code = if child_result.is_ok() { 0 } else { 1 };
                    unsafe {
                        libc::_exit(code);
                    }
                }
                child_pid => {
                    // Parent: restore the pre-fork signal mask
                    // (the block was transient — only the child
                    // inherits the blocked SIGUSR1, and that
                    // child unblocks immediately after installing
                    // its handler). Without this restore, SIGUSR1
                    // would stay blocked in the calling thread for
                    // the lifetime of the workload, swallowing any
                    // signal directed at the parent process.
                    let psm_parent_restore_rc = unsafe {
                        libc::pthread_sigmask(libc::SIG_SETMASK, &old_mask, std::ptr::null_mut())
                    };
                    if psm_parent_restore_rc != 0 {
                        tracing::warn!(
                            rc = psm_parent_restore_rc,
                            "pthread_sigmask(SIG_SETMASK) failed restoring mask in parent post-fork; SIGUSR1 stays blocked in this thread for the lifetime of the workload"
                        );
                    }
                    // Close unused pipe ends.
                    unsafe {
                        libc::close(report_fds[1]);
                        libc::close(start_fds[0]);
                    }
                    // child_pid is positive by the -1 arm above, so
                    // fits in pid_t directly — store as pid_t so
                    // every downstream libc call avoids the u32→i32
                    // sign-cast wraparound bug.
                    guard.children.push(ForkedChild {
                        pid: child_pid,
                        report_fd: report_fds[0],
                        start_fd: start_fds[1],
                        kind: ForkedChildKind::Worker {
                            group_idx: group.group_idx,
                        },
                    });
                }
            }
        }

        Ok(())
    }

    /// Kernel TIDs of all worker tasks, in spawn order.
    ///
    /// Returned as `libc::pid_t` — the kernel's native type — so
    /// callers feed them directly into `kill`, `waitpid`,
    /// `Pid::from_raw`, and `sched_setaffinity` writes without any
    /// sign-cast at the libc boundary.
    ///
    /// # WARNING — `cgroup.procs` for `CloneMode::Thread`
    ///
    /// **For `CloneMode::Thread`, passing these TIDs to a
    /// `cgroup.procs` write migrates the ENTIRE test-runner process
    /// into that cgroup**: cgroup.procs writes are tgid-scoped, and
    /// every Thread worker shares the test runner's tgid. The first
    /// such write moves the test harness, every parent thread, and
    /// every sibling worker into the destination cgroup; subsequent
    /// writes are no-ops because they all point at the same tgid.
    /// Use cgroup v2 threaded-mode cgroups with `cgroup.threads`
    /// for per-thread placement. `CloneMode::Fork` is the right
    /// choice when each worker needs its own cgroup.
    ///
    /// # Per-mode interpretation
    ///
    /// - [`CloneMode::Fork`]: each entry is the worker's pid
    ///   (== tgid == kernel tid because the worker is its own
    ///   thread-group leader). Safe to feed into `cgroup.procs`.
    /// - [`CloneMode::Thread`]: each entry is the worker's
    ///   `gettid()` value — distinct kernel tasks inside the
    ///   parent's tgid. Safe for `sched_setaffinity(tid, ...)`;
    ///   safe for `cgroup.threads` writes under a threaded-mode
    ///   cgroup; **not** safe for `cgroup.procs` (see warning above).
    ///
    /// # Thread tid publish ordering
    ///
    /// Thread workers publish their `gettid()` via an
    /// `Arc<AtomicI32>` after the start handshake. The publish uses
    /// `Release`; this reader uses `Acquire`, pairing release-
    /// acquire so that any reader who observes a non-zero tid is
    /// also guaranteed to observe the worker's full post-start
    /// state. If the caller invokes `worker_pids()` before
    /// [`start()`](Self::start) returns, the worker may not yet
    /// have stored its tid and `0` (the `AtomicI32` initial value)
    /// is reported in those slots. Callers that require post-start
    /// tids must call `start()` before `worker_pids()`.
    ///
    /// # `pcomm` containers
    ///
    /// Groups configured with [`WorkSpec::pcomm`] yield a single
    /// container process whose tgid leader pid is what the parent
    /// holds; the per-thread `gettid()` values of the workers running
    /// inside the container are not exported across the process
    /// boundary. Each pcomm group contributes ONE entry to the
    /// returned vector — the container pid — regardless of how many
    /// thread workers it hosts. This pid is correct for `cgroup.procs`
    /// migration (the container's whole tgid moves) but is NOT
    /// suitable as a target for per-thread `sched_setaffinity` —
    /// see [`Self::set_affinity`] for the matching error path.
    ///
    /// Order: forked children (conventional workers + pcomm
    /// containers, in spawn order) followed by Thread-mode workers
    /// (in spawn order). A workload that mixes pcomm groups with
    /// non-pcomm Thread-mode groups produces both populated
    /// collections; a workload using only one dispatch path produces
    /// only one populated collection.
    pub fn worker_pids(&self) -> Vec<libc::pid_t> {
        let mut out = Vec::with_capacity(self.children.len() + self.threads.len());
        out.extend(self.children.iter().map(|c| c.pid));
        out.extend(self.threads.iter().map(|tw| tw.tid.load(Ordering::Acquire)));
        out
    }

    /// Worker pids suitable for `cgroup.procs` migration.
    ///
    /// `cgroup.procs` is **tgid-scoped** in the kernel: writing a
    /// tid migrates the entire thread group containing that tid
    /// (`kernel/cgroup/cgroup.c::__cgroup_procs_write` resolves the
    /// passed pid to its leader via `find_lock_task_mm` /
    /// `cgroup_procs_write_start`). Under [`CloneMode::Thread`]
    /// every worker shares the test harness's tgid, so feeding
    /// [`Self::worker_pids`] to `cgroup.procs` would migrate the
    /// harness itself — catastrophic.
    ///
    /// Returns the per-worker pids when the spawn used
    /// [`CloneMode::Fork`] (each worker has its own tgid). Bails
    /// for [`CloneMode::Thread`] with an actionable diagnostic
    /// pointing at `cgroup.threads` (the thread-scoped sibling) as
    /// the right migration sink for thread workers.
    ///
    /// Callers that integrate with `cgroup.procs` writes — e.g.
    /// [`crate::cgroup::CgroupManager::move_tasks`] — should call
    /// this in place of [`Self::worker_pids`] so a misconfigured
    /// Thread-mode test fails at the migration step rather than
    /// silently moving the harness into the per-test cgroup.
    pub fn worker_pids_for_cgroup_procs(&self) -> Result<Vec<libc::pid_t>> {
        if !self.threads.is_empty() {
            anyhow::bail!(
                "WorkloadHandle::worker_pids_for_cgroup_procs: workers were \
                 spawned with CloneMode::Thread; their pids share the test \
                 harness's tgid and a `cgroup.procs` write would migrate the \
                 harness. Use `cgroup.threads` (thread-scoped) for Thread-mode \
                 workers, or switch to CloneMode::Fork."
            );
        }
        Ok(self.worker_pids())
    }

    /// Signal all workers to start working (after they've been
    /// placed in cgroups, if applicable).
    ///
    /// Idempotent — subsequent calls after the first are no-ops.
    pub fn start(&mut self) {
        if self.started {
            return;
        }
        self.started = true;
        // Fork-mode: write a byte to each child's start pipe (covers
        // both conventional workers and pcomm containers — both wait
        // on the same start-byte handshake before entering their work
        // path).
        for child in &mut self.children {
            unsafe {
                libc::write(child.start_fd, b"s".as_ptr() as *const _, 1);
                libc::close(child.start_fd);
            }
            child.start_fd = -1;
        }
        // Thread-mode: send `()` on each worker's start_tx. The
        // SyncSender(0) rendezvous means each send blocks until the
        // worker calls recv(); if the worker has been joined or has
        // panicked before reaching recv, send returns Err which we
        // swallow (the join in stop_and_collect surfaces the real
        // exit). Take ownership so a future start() call (illegal
        // by the idempotence guard above) can't re-send.
        for tw in &mut self.threads {
            if let Some(tx) = tw.start_tx.take() {
                let _ = tx.send(());
            }
        }
    }

    /// Set CPU affinity for worker at `idx`.
    ///
    /// For [`CloneMode::Fork`] the per-worker pid addresses a
    /// distinct kernel task. For [`CloneMode::Thread`] the worker's
    /// `gettid()` is what `sched_setaffinity(tid, ...)` accepts;
    /// this method reads the tid from the worker's
    /// `Arc<AtomicI32>` (with `Acquire` ordering, paired with the
    /// `Release` publish on the worker thread). Returns an error
    /// if the thread has not yet published its tid — call
    /// [`start()`](Self::start) first so the worker reaches its
    /// `gettid()` publish before reading.
    ///
    /// Bails for [`ForkedChildKind::PcommContainer`] entries: the
    /// container's tgid leader pid is the only kernel handle the
    /// parent holds for that group, but `sched_setaffinity` against
    /// that pid pins only the leader thread (a placeholder thread
    /// that never enters the work loop), not the worker threads
    /// running the workload. The container's worker tids are not
    /// published across the process boundary. Bake the affinity into
    /// [`WorkSpec::affinity`] at spawn time instead — `worker_main`
    /// applies it per-thread inside the container.
    ///
    /// Index space: `[0, children.len())` addresses forked children
    /// (conventional workers and pcomm containers), and
    /// `[children.len(), children.len() + threads.len())` addresses
    /// Thread-mode workers, matching the ordering of
    /// [`Self::worker_pids`].
    pub fn set_affinity(&self, idx: usize, cpus: &BTreeSet<usize>) -> Result<()> {
        let pid = if idx < self.children.len() {
            let child = &self.children[idx];
            if let ForkedChildKind::PcommContainer { groups } = &child.kind {
                let total: usize = groups.iter().map(|(_, n)| n).sum();
                anyhow::bail!(
                    "set_affinity: child {idx} is a pcomm container \
                     hosting {total} thread workers; per-thread \
                     tids are not exported across the process boundary. \
                     Set affinity via WorkSpec::affinity at spawn time \
                     so worker_main applies it inside the container."
                );
            }
            child.pid
        } else {
            let thread_idx = idx - self.children.len();
            let tid = self.threads[thread_idx].tid.load(Ordering::Acquire);
            if tid == 0 {
                anyhow::bail!(
                    "set_affinity: thread worker {thread_idx} has not yet \
                     published gettid() (call start() first)"
                );
            }
            tid
        };
        set_thread_affinity(pid, cpus)
    }

    /// Read all workers' current iteration counts from shared memory.
    ///
    /// Each element is the monotonically increasing iteration count for
    /// that worker, read with Relaxed ordering. Returns an empty vec
    /// if no workers were spawned.
    ///
    /// # Ordering rationale — why Relaxed is sound
    ///
    /// Every producer (the worker-side store at the
    /// `worker_main` publish sites) writes its slot with Relaxed
    /// ordering, and this reader loads with Relaxed too. No
    /// happens-before edge is needed because no host-side consumer
    /// pairs the iteration count with OTHER shared state: the
    /// parent samples these counters to answer "is this worker
    /// still making progress?" and feeds deltas into gap
    /// detection, not into any data-dependent follow-up read from
    /// a different shared memory location. A stale value on one
    /// sample is self-correcting — the next snapshot picks up the
    /// newer count without any cross-field invariant to break.
    ///
    /// The per-slot single-producer / multi-sampler shape is
    /// inherently non-tearing on every supported target
    /// (AtomicU64 is architecture-primitive on x86_64 and aarch64
    /// LSE with 8-byte alignment enforced by the type). The only
    /// question is ordering, and the audit above concludes Relaxed
    /// is load-bearingly correct — promoting either side to
    /// Acquire/Release would add a barrier with no corresponding
    /// paired operation to synchronise with.
    pub fn snapshot_iterations(&self) -> Vec<u64> {
        if self.iter_counters.is_null() || self.iter_counter_len == 0 {
            return Vec::new();
        }
        (0..self.iter_counter_len)
            .map(|i| {
                // SAFETY: alignment + atomic-only-access invariant
                // established at the iter_counters mmap site in
                // `WorkloadHandle::spawn` and carried by the
                // `*mut AtomicU64` type. Relaxed ordering: see the
                // rationale in the outer doc comment.
                unsafe { &*self.iter_counters.add(i) }.load(Ordering::Relaxed)
            })
            .collect()
    }

    /// Stop all workers, collect their reports, and wait for exit.
    ///
    /// Auto-starts workers if [`start()`](Self::start) was not called,
    /// then waits on an event-driven barrier — each fork worker
    /// writes a single `b'r'` byte to its report pipe immediately
    /// after the start handshake completes, and the parent polls
    /// every report fd for `POLLIN` with a 5 s deadline. The
    /// barrier wakes the moment the slowest worker finishes its
    /// post-fork init, replacing the prior unconditional 500 ms
    /// sleep that under-waited under host CPU contention and
    /// over-waited on idle hosts. Thread-mode workers are pre-
    /// synchronised by `start()`'s `mpsc::sync_channel(0)` rendezvous,
    /// so the barrier is a no-op when no fork children were spawned.
    /// Consumes `self` -- workers cannot be restarted.
    ///
    /// Workers that fail to produce a report (died, timed out, or wrote
    /// corrupt data) get a zeroed-out sentinel report with `work_units: 0`.
    /// This ensures `assert_not_starved` catches dead workers as starvation
    /// failures.
    ///
    /// # Shutdown latency
    ///
    /// Workers spend their steady-state time blocked inside a
    /// `futex_wait` with timeout [`WORKER_STOP_POLL_NS`] (~100 ms).
    /// The "stop signal" is a per-mode flag the worker checks on
    /// every futex-wait wake; the wake interval bounds shutdown
    /// latency.
    ///
    /// _Fork mode_ — `stop_and_collect` sends SIGUSR1 to each
    /// worker pid; the per-process `sigusr1_handler` flips the
    /// global [`STOP`] in that worker's CoW address space, and the
    /// worker observes it on the NEXT futex wake (partner-writes
    /// or the 100 ms timeout, whichever comes first). The signal
    /// handler is process-wide and reaches one worker per kill().
    ///
    /// _Thread mode_ — `stop_and_collect` calls
    /// `worker.stop.store(true, Relaxed)` directly on each
    /// worker's `Arc<AtomicBool>`. SIGUSR1 is process-wide and
    /// useless for per-thread stop control, so no signal is sent;
    /// the worker observes the flag flip on its next futex-wait
    /// wake at the same 100 ms cadence.
    ///
    /// Callers that budget a graceful-shutdown window should
    /// allow at least one [`WORKER_STOP_POLL_NS`] tick (~100 ms)
    /// between flag flip and final collect, over and above any
    /// report-flush / IO latency. Tighter windows can race the
    /// worker's pre-stop iteration and surface as a missing
    /// report, which is then mapped to the sentinel path above.
    ///
    /// # Exit-shape invariance
    ///
    /// Collection discriminates purely on the presence and validity of
    /// the worker's pipe-delivered bincode payload — **not** on
    /// `waitpid` exit status. Under `panic = "unwind"` (dev/test
    /// profile) the worker's
    /// `catch_unwind` arm calls `_exit(1)` so the parent sees
    /// `WIFEXITED=true`, `WEXITSTATUS=1`; under `panic = "abort"`
    /// (release profile) the worker aborts with `SIGABRT` so the parent
    /// sees `WIFEXITED=false`, `WTERMSIG=6`. Either way, a panicking
    /// worker never finishes `f.write_all(&bytes)` on the report pipe,
    /// so `poll` + `read_to_end` hands back an empty (or truncated)
    /// buffer, `bincode::serde::decode_from_slice` fails, and the
    /// sentinel path fires. Partial writes from a panic between successful
    /// `write_all` and `_exit(0)` are not reachable — the write is the
    /// last non-trivial statement inside the catch_unwind closure.
    /// The `waitpid` call later in this function exists solely for
    /// reaping zombies; its return value feeds only the "still alive
    /// → SIGKILL escalate" branch and is never mapped to report
    /// state (the sentinel path DOES now read it to populate
    /// [`WorkerExitInfo`] on the attached diagnostic, but the
    /// correctness discrimination — sentinel vs real report — still
    /// happens purely on pipe payload presence).
    pub fn stop_and_collect(mut self) -> Vec<WorkerReport> {
        // Auto-start if not explicitly started (workers in parent cgroup)
        let was_started = self.started;
        self.start();

        // Event-driven worker-started barrier. Each fork worker writes
        // a single `b'r'` byte to its report pipe immediately after
        // the start-pipe handshake completes (see the matching write
        // inside `worker_main`'s catch_unwind closure). Polling every
        // worker's report fd for `POLLIN` with a bounded deadline
        // wakes the moment the slowest worker has finished its
        // post-fork init — replacing the prior 500 ms blind sleep
        // that could under-wait on a CPU-contended host (false
        // starvation if the worker entered its loop after stop was
        // signalled) and over-wait on an idle host (~500 ms wasted
        // per `stop_and_collect`).
        //
        // Thread-mode workers do not need a barrier here: `start()`
        // above sent on a `mpsc::sync_channel(0)` SyncSender(0)
        // rendezvous, which blocks the parent until the worker's
        // matching `recv()` returns — by the time `start()` returns,
        // every thread worker has crossed its start handshake. Only
        // the fork-mode pipe-based start signal is fire-and-forget,
        // so the barrier is gated on `!self.children.is_empty()`.
        //
        // Deadline budget: 5 s mirrors the existing collect deadline
        // below. Each iteration polls every still-pending fd with
        // the remaining budget; a worker that returns `POLLHUP` /
        // `POLLERR` (died before writing the ready byte — fork-race
        // close, panic during early init, or the kernel killed it)
        // is dropped from the pending set and the surrounding
        // collect path's sentinel-report logic surfaces it. The
        // worst case (every worker hits the deadline without a
        // ready byte) bounds the wait at the same 5 s the legacy
        // sleep + collect path budgeted for the entire stop_and_collect.
        if !was_started && !self.children.is_empty() {
            let barrier_deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            // Pending = (index_into_children, read_fd). We track the
            // index so a `POLLHUP` worker can be removed from
            // pending without disturbing the collect loop's
            // ordering below.
            let mut pending: Vec<(usize, i32)> = self
                .children
                .iter()
                .enumerate()
                .map(|(i, c)| (i, c.report_fd))
                .collect();
            while !pending.is_empty() {
                let remaining =
                    barrier_deadline.saturating_duration_since(std::time::Instant::now());
                if remaining.is_zero() {
                    // Barrier deadline expired with workers still
                    // pending. Stop waiting and proceed to signal
                    // stop. The collect path below treats any
                    // worker whose pipe never produced data as a
                    // sentinel report — same outcome as a worker
                    // that started in time but produced an
                    // unparseable report.
                    break;
                }
                let ms = remaining.as_millis().min(i32::MAX as u128) as i32;
                let mut pfds: Vec<libc::pollfd> = pending
                    .iter()
                    .map(|&(_, fd)| libc::pollfd {
                        fd,
                        events: libc::POLLIN,
                        revents: 0,
                    })
                    .collect();
                // SAFETY: `pfds` is a non-empty owned Vec; nfds
                // matches its length; `ms >= 0` since the
                // remaining-zero branch above bails first. Return
                // codes are interpreted below.
                let ret = unsafe { libc::poll(pfds.as_mut_ptr(), pfds.len() as libc::nfds_t, ms) };
                if ret < 0 {
                    let err = std::io::Error::last_os_error();
                    if err.kind() == std::io::ErrorKind::Interrupted {
                        // EINTR — re-poll with the remaining budget
                        // on the next iteration. Common during
                        // teardown when a sibling thread sends a
                        // signal; the loop's deadline guard bounds
                        // total wait time regardless of EINTR
                        // frequency.
                        continue;
                    }
                    // Hard poll failure (EFAULT impossible with an
                    // owned Vec; ENOMEM extreme). Bail out of the
                    // barrier and let the collect path's per-fd
                    // poll handle each worker individually.
                    tracing::warn!(
                        %err,
                        pending = pending.len(),
                        "WorkloadHandle::stop_and_collect: barrier poll failed; falling \
                         through to per-worker collect"
                    );
                    break;
                }
                // ret == 0 means the per-iteration timeout fired
                // without any fd ready — but we measured this
                // against `remaining`, so the next iteration's
                // saturating_duration_since will be zero and the
                // top-of-loop guard exits. Don't break here: a
                // future cycle could still be useful if the system
                // clock jumped backward, and the cost is one extra
                // iteration that immediately bails.
                if ret > 0 {
                    pending.retain(|&(_, fd)| {
                        // Find this fd's pollfd entry. Linear scan
                        // is fine: typical worker counts are <100
                        // and the alternative (HashMap) costs more
                        // in setup than the linear scan saves.
                        let pfd = pfds.iter().find(|p| p.fd == fd);
                        let revents = pfd.map(|p| p.revents).unwrap_or(0);
                        if revents & libc::POLLIN != 0 {
                            // Ready byte arrived. Consume exactly
                            // 1 byte and remove from pending. The
                            // raw `libc::read` does not take
                            // ownership of the fd — the collect
                            // path below still owns it via the
                            // `children` Vec and will read the
                            // bincode tail on its own deadline.
                            let mut byte: u8 = 0;
                            // SAFETY: `&mut byte` is a valid
                            // 1-byte buffer; `fd` is the report
                            // read end the parent owns until
                            // collect drains it. A 0 / -1 return
                            // is treated as not-yet-ready; the
                            // next iteration retries.
                            let n = unsafe {
                                libc::read(fd, &mut byte as *mut u8 as *mut libc::c_void, 1)
                            };
                            // n == 1 → ready byte consumed; drop
                            // from pending.
                            // n == 0 → POLLIN with zero-byte read
                            // means EOF (writer closed) without
                            // sending the byte — worker died
                            // pre-write. Drop from pending; the
                            // collect path emits a sentinel report.
                            // n < 0 → transient error (EAGAIN
                            // shouldn't happen since POLLIN
                            // signalled readability, but on EINTR
                            // the next iteration re-polls).
                            if n >= 0 {
                                return false;
                            }
                            // Negative return: re-check kind.
                            let err = std::io::Error::last_os_error();
                            if err.kind() == std::io::ErrorKind::Interrupted {
                                return true;
                            }
                            tracing::warn!(
                                %err,
                                fd,
                                "WorkloadHandle::stop_and_collect: barrier byte read \
                                 failed; treating worker as ready"
                            );
                            return false;
                        }
                        if revents & (libc::POLLHUP | libc::POLLERR | libc::POLLNVAL) != 0 {
                            // Worker closed the write end without
                            // sending the ready byte (panic during
                            // post-fork init, or kernel-killed
                            // before reaching the write). Drop
                            // from pending; the collect path's
                            // `read_to_end` returns 0 bytes and the
                            // sentinel-report branch fires.
                            return false;
                        }
                        // No POLLIN, no hangup — keep waiting.
                        true
                    });
                }
            }
        }

        let mut reports = Vec::new();
        let children = std::mem::take(&mut self.children);
        let threads = std::mem::take(&mut self.threads);

        // Drain the 'r' ready byte from each child's report pipe
        // so the collect poll below waits for the report payload, not
        // the already-present ready byte. The barrier section above
        // consumes it for auto-started workers; explicitly-started
        // workers (was_started=true) skip the barrier.
        if was_started {
            for child in &children {
                let mut byte: u8 = 0;
                unsafe {
                    libc::read(
                        child.report_fd,
                        &mut byte as *mut u8 as *mut libc::c_void,
                        1,
                    );
                }
            }
        }

        // Signal all fork-mode children to stop via SIGUSR1; the
        // signal handler flips that child's process-local STOP, which
        // worker_main's `stop_requested` checks read. For pcomm
        // containers the SIGUSR1 flips the container's STOP, which
        // every thread inside the container observes via `&STOP`
        // passed to `worker_main` — one signal stops the whole
        // group cleanly. `pid` is `libc::pid_t`, so it flows to
        // `Pid::from_raw` without the u32→i32 sign-cast wraparound
        // that produced `kill(-1, ...)` session-wide reaps when the
        // old u32 pid exceeded i32::MAX.
        for child in &children {
            let _ = nix::sys::signal::kill(
                nix::unistd::Pid::from_raw(child.pid),
                nix::sys::signal::Signal::SIGUSR1,
            );
        }
        // Signal all thread-mode workers by flipping each worker's
        // per-task `stop`. SIGUSR1 is process-wide and useless for
        // per-thread stop; the Arc<AtomicBool> threaded through
        // worker_main is the only path that reaches an individual
        // thread without affecting siblings.
        for tw in &threads {
            tw.stop.store(true, Ordering::Relaxed);
        }

        // Collect reports with a shared 5s deadline across all workers.
        // Each worker gets the remaining budget, so starved workers
        // (e.g. under degrade mode) don't serially exhaust the VM
        // timeout.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        for child in children {
            let mut buf = Vec::new();
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            let ms = remaining.as_millis().min(i32::MAX as u128) as i32;
            let mut poll_ready = false;
            if ms > 0 {
                let mut pfd = libc::pollfd {
                    fd: child.report_fd,
                    events: libc::POLLIN,
                    revents: 0,
                };
                let ready = unsafe { libc::poll(&mut pfd, 1, ms) };
                poll_ready = ready > 0;
            }

            let npid = nix::unistd::Pid::from_raw(child.pid);
            if !poll_ready {
                // Deadline expired — child didn't write a report.
                // Kill first so read_to_end doesn't block on the
                // pipe's write end (the child may have written
                // only the 'r' ready byte but no payload).
                let _ = nix::sys::signal::kill(npid, nix::sys::signal::Signal::SIGKILL);
                let _ = nix::sys::signal::killpg(npid, nix::sys::signal::Signal::SIGKILL);
                let _ = nix::unistd::close(child.report_fd);
            } else {
                // Child responded — read the report, then kill.
                let mut f = unsafe { std::fs::File::from_raw_fd(child.report_fd) };
                let _ = f.read_to_end(&mut buf);
                drop(f);
                let _ = nix::sys::signal::kill(npid, nix::sys::signal::Signal::SIGKILL);
                let _ = nix::sys::signal::killpg(npid, nix::sys::signal::Signal::SIGKILL);
            }
            let waited = nix::sys::wait::waitpid(npid, Some(nix::sys::wait::WaitPidFlag::WNOHANG));
            let still_running = matches!(waited, Ok(nix::sys::wait::WaitStatus::StillAlive));
            let exit_info_source: Result<nix::sys::wait::WaitStatus, nix::errno::Errno> =
                if still_running {
                    let _ = nix::sys::wait::waitpid(npid, None);
                    Ok(nix::sys::wait::WaitStatus::StillAlive)
                } else {
                    waited
                };

            // Strip the leading `b'r'` ready byte if the auto-start
            // barrier above did not consume it. Children write this
            // byte unconditionally on every success path right after
            // the start handshake; the barrier polls + reads it when
            // `stop_and_collect` auto-started children, but
            // explicit-start callers (`start()` invoked before
            // `stop_and_collect`) bypass the barrier and the byte
            // sits in the pipe ahead of the payload. Strip exactly 1
            // byte if present so the per-kind decoder sees a clean
            // payload either way.
            let report_slice: &[u8] = if buf.first() == Some(&b'r') {
                &buf[1..]
            } else {
                &buf[..]
            };

            // Per-kind decoding. `Worker` carries a single bincode
            // `WorkerReport` (`bincode::config::standard()`,
            // little-endian, variable int). `PcommContainer` carries
            // a `serde_json` `Vec<WorkerReport>` — serde_json for
            // pcomm per task #6 design ruling; fork-mode workers
            // use bincode for single WorkerReport. Both
            // payloads ride the EOF-terminated pipe with no length
            // prefix; the parent's `read_to_end` provides framing.
            // On a decode failure, emit one sentinel per expected
            // report so downstream consumers (per-group filtering,
            // assert_not_starved) see the correct cardinality.
            match child.kind {
                ForkedChildKind::Worker { group_idx } => {
                    let decoded: Result<(WorkerReport, usize), _> =
                        bincode::serde::decode_from_slice(
                            report_slice,
                            bincode::config::standard(),
                        );
                    if let Ok((report, _)) = decoded {
                        reports.push(report);
                    } else {
                        let exit_info = classify_wait_outcome(exit_info_source);
                        eprintln!(
                            "ktstr: worker pid={} returned no report ({} bytes read, exit={exit_info:?})",
                            child.pid,
                            buf.len(),
                        );
                        reports.push(WorkerReport {
                            tid: child.pid,
                            group_idx,
                            exit_info: Some(exit_info),
                            ..WorkerReport::default()
                        });
                    }
                }
                ForkedChildKind::PcommContainer { groups } => {
                    let total_workers: usize = groups.iter().map(|(_, n)| n).sum();
                    let decoded: Result<Vec<WorkerReport>, _> =
                        serde_json::from_slice(report_slice);
                    match decoded {
                        Ok(mut decoded_reports) if decoded_reports.len() == total_workers => {
                            reports.append(&mut decoded_reports);
                        }
                        Ok(mut decoded_reports) => {
                            // Cardinality mismatch — surface the
                            // partial set we did get and pad with
                            // sentinels so the total report count
                            // still equals `total_workers`. A short
                            // payload typically signals the
                            // thread-group leader died mid-encode
                            // (panic in `serde_json::to_vec`, OOM
                            // during Vec growth, or a write_all
                            // truncated by SIGKILL escalation).
                            // Sentinels are tagged with the right
                            // per-group `group_idx` using the
                            // `groups` layout: groups[0] consumed
                            // the first groups[0].1 slots,
                            // groups[1] the next groups[1].1, etc.
                            // We emit sentinels for the trailing
                            // missing slots, computing each
                            // sentinel's group_idx by walking the
                            // layout from `got` forward.
                            let exit_info = classify_wait_outcome(exit_info_source);
                            eprintln!(
                                "ktstr: pcomm thread-group leader pid={} returned {} of {} reports ({} bytes read, exit={exit_info:?})",
                                child.pid,
                                decoded_reports.len(),
                                total_workers,
                                buf.len(),
                            );
                            // Surplus reports (decoded > total_workers)
                            // must not leak into the parent's report
                            // stream — downstream cardinality assertions
                            // (per-group filtering, assert_not_starved)
                            // assume exactly `total_workers` entries.
                            // Truncate to the layout's total before the
                            // `got..total_workers` loop runs (which is
                            // a no-op for the surplus case after this).
                            decoded_reports.truncate(total_workers);
                            let got = decoded_reports.len();
                            for r in decoded_reports {
                                reports.push(r);
                            }
                            // Compute group_idx for each missing
                            // slot by walking the layout. `slot` is
                            // a 0-based global index from `got` to
                            // `total_workers - 1`; we accumulate
                            // per-group counts to find which group
                            // owns each slot.
                            for slot in got..total_workers {
                                let mut acc = 0usize;
                                let mut g_idx = 0usize;
                                for &(gi, n) in &groups {
                                    if slot < acc + n {
                                        g_idx = gi;
                                        break;
                                    }
                                    acc += n;
                                }
                                reports.push(WorkerReport {
                                    tid: child.pid,
                                    group_idx: g_idx,
                                    exit_info: Some(exit_info.clone()),
                                    ..WorkerReport::default()
                                });
                            }
                        }
                        Err(_) => {
                            let exit_info = classify_wait_outcome(exit_info_source);
                            eprintln!(
                                "ktstr: pcomm thread-group leader pid={} returned no decodable report ({} bytes read, exit={exit_info:?})",
                                child.pid,
                                buf.len(),
                            );
                            // Total decode failure — emit one
                            // sentinel per slot, tagging each with
                            // the correct group_idx.
                            for &(gi, n) in &groups {
                                for _ in 0..n {
                                    reports.push(WorkerReport {
                                        tid: child.pid,
                                        group_idx: gi,
                                        exit_info: Some(exit_info.clone()),
                                        ..WorkerReport::default()
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        // Thread-mode collection: join each worker's JoinHandle
        // (with the [`THREAD_JOIN_TIMEOUT`] budget) and adopt the
        // returned [`WorkerReport`]. Per-worker `stop` was flipped
        // above; the worker observes it in worker_main's
        // `stop.load(Relaxed)` checks (max ~100ms latency from the
        // FUTEX_WAIT_TIMEOUT poll cadence). Three outcomes:
        //
        //   1. Ok(report): join returned the worker's WorkerReport.
        //      Push as-is.
        //   2. Err(payload): the thread panicked. Build a sentinel
        //      report and attach
        //      `exit_info: Some(WorkerExitInfo::Panicked(msg))`
        //      where `msg` comes from `extract_panic_payload`.
        //   3. Timeout (5s elapsed without is_finished): emit a
        //      tracing::warn and push a sentinel with
        //      `exit_info: Some(WorkerExitInfo::TimedOut)` —
        //      `worker_main` should have observed the per-worker
        //      `stop` within 100ms, so a 5s no-show signals a
        //      genuinely stuck worker (deadlock, infinite spin,
        //      blocking syscall the runtime can't interrupt).
        //      stop_and_collect does NOT process::exit on timeout —
        //      the orphan thread keeps running until the test
        //      harness exits, but any subsequent worker uses a
        //      fresh per-worker `stop` so the orphan can't pollute
        //      later runs.
        for mut tw in threads {
            // Drop start_tx (idempotent — `start()` may have already
            // taken it). If start() ran first, `start_tx` is
            // already `None` and the take is a no-op; if the caller
            // skipped start() entirely, dropping start_tx here
            // signals the worker via `Disconnected` so it exits
            // cleanly without the rendezvous send.
            tw.start_tx.take();
            let tid = tw.tid.load(Ordering::Acquire);
            if let Some(j) = tw.join.take() {
                match join_thread_with_timeout(j, &tw.exit_evt, THREAD_JOIN_TIMEOUT) {
                    Some(Ok(report)) => reports.push(report),
                    Some(Err(payload)) => {
                        let msg = extract_panic_payload(payload);
                        eprintln!("ktstr: thread worker tid={tid} panicked: {msg}");
                        reports.push(WorkerReport {
                            tid,
                            completed: false,
                            exit_info: Some(WorkerExitInfo::Panicked(msg)),
                            ..WorkerReport::default()
                        });
                    }
                    None => {
                        tracing::warn!(
                            tid,
                            timeout_secs = THREAD_JOIN_TIMEOUT.as_secs(),
                            "thread worker did not join within timeout — leaking the \
                             thread; sentinel report attached with TimedOut exit_info"
                        );
                        reports.push(WorkerReport {
                            tid,
                            completed: false,
                            exit_info: Some(WorkerExitInfo::TimedOut),
                            ..WorkerReport::default()
                        });
                    }
                }
            }
        }

        reports
    }
}

impl Drop for WorkloadHandle {
    fn drop(&mut self) {
        use nix::sys::signal::{Signal, kill};
        use nix::sys::wait::waitpid;
        use nix::unistd::{Pid, close};

        // Forked children (conventional workers AND pcomm
        // containers). `pid` is `libc::pid_t` — stored as i32 so
        // `Pid::from_raw` receives the kernel's native representation
        // directly, not the sign-cast of a u32 that could alias
        // negative values (including -1, i.e. every process in the
        // session).
        for child in &self.children {
            let pid = child.pid;
            let nix_pid = Pid::from_raw(pid);
            // killpg first: reach descendants the child may have
            // forked (Custom workloads, ForkExit caught mid-fork).
            // pgid == child pid because every forked child (worker
            // or pcomm container) calls `setpgid(0, 0)` at fork time.
            // ESRCH (group gone / no members) is expected and not a
            // warning-worthy failure; swallow it to keep the log
            // clean when the common no-descendants case drops.
            if let Err(e) = nix::sys::signal::killpg(nix_pid, Signal::SIGKILL)
                && e != nix::errno::Errno::ESRCH
            {
                tracing::warn!(pid, %e, "killpg failed in WorkloadHandle::drop");
            }
            if let Err(e) = kill(nix_pid, Signal::SIGKILL) {
                tracing::warn!(pid, %e, "kill failed in WorkloadHandle::drop");
            }
            if let Err(e) = waitpid(nix_pid, None) {
                tracing::warn!(pid, %e, "waitpid failed in WorkloadHandle::drop");
            }
            for fd in [child.report_fd, child.start_fd] {
                if fd >= 0
                    && let Err(e) = close(fd)
                {
                    tracing::warn!(fd, %e, "close failed in WorkloadHandle::drop");
                }
            }
        }
        // Thread-mode workers: flip stop, drop start_tx (in case
        // worker hasn't yet recv'd), join with the same 5s budget
        // `stop_and_collect` uses. Threads share the parent's
        // address space — there is no `kill` equivalent and no
        // MAP_SHARED ownership to give back. Drop still applies
        // the timeout so a stuck worker doesn't pin
        // `WorkloadHandle::drop` indefinitely; on timeout we log
        // the leak via `tracing::warn!` and proceed.
        let threads = std::mem::take(&mut self.threads);
        for mut tw in threads {
            tw.stop.store(true, Ordering::Relaxed);
            tw.start_tx.take();
            if let Some(j) = tw.join.take() {
                let tid = tw.tid.load(Ordering::Acquire);
                match join_thread_with_timeout(j, &tw.exit_evt, THREAD_JOIN_TIMEOUT) {
                    Some(Ok(_)) => {}
                    Some(Err(e)) => {
                        let payload = extract_panic_payload(e);
                        tracing::warn!(
                            tid,
                            payload,
                            "thread worker panicked in WorkloadHandle::drop"
                        );
                    }
                    None => {
                        tracing::warn!(
                            tid,
                            timeout_secs = THREAD_JOIN_TIMEOUT.as_secs(),
                            "thread worker failed to join within timeout in \
                             WorkloadHandle::drop — leaking the thread"
                        );
                    }
                }
            }
        }
        // Close inter-worker pipe pairs and chain pipes AFTER worker
        // shutdown. Ordering matters for Thread mode: every worker
        // thread shares the parent's fd table, so closing a pipe fd
        // before its using thread joins would surface to that thread
        // as `EBADF` on the next read/write/poll syscall. The
        // children-reap loop above and the threads-join loop above
        // both block until their worker is reaped or joined; only
        // then do these closes run, which is when the workers are
        // guaranteed to no longer touch their fds. Fork mode is
        // unaffected either way: each child held its own fd-table
        // copy via `fork()`, so this close is a no-op for the
        // child's view (its own copy was closed by the post-fork
        // close-other-fds block in spawn_group).
        //
        // Errors from `close` are logged via `tracing::warn!` rather
        // than swallowed — `EBADF` here would indicate a double-close
        // (an aliased ownership bug) and is more diagnostic than the
        // SpawnGuard early-bail path's silent close. SpawnGuard's
        // Drop swallows EBADF deliberately because mid-spawn the
        // guard may share fd ownership with already-closed
        // half-allocated state; the handle on the other hand has
        // sole ownership at this point.
        for (ab, ba) in &self.pipe_pairs {
            for fd in [ab[0], ab[1], ba[0], ba[1]] {
                if let Err(e) = close(fd) {
                    tracing::warn!(fd, %e, "close failed for pipe_pair fd in WorkloadHandle::drop");
                }
            }
        }
        for chain in &self.chain_pipes {
            for pipe in chain {
                for fd in [pipe[0], pipe[1]] {
                    if let Err(e) = close(fd) {
                        tracing::warn!(fd, %e, "close failed for chain_pipe fd in WorkloadHandle::drop");
                    }
                }
            }
        }
        for (&ptr, &size) in self.futex_ptrs.iter().zip(self.futex_region_sizes.iter()) {
            unsafe {
                libc::munmap(ptr as *mut libc::c_void, size);
            }
        }
        if !self.iter_counters.is_null() && self.iter_counter_len > 0 {
            unsafe {
                libc::munmap(
                    self.iter_counters as *mut libc::c_void,
                    self.iter_counter_len * std::mem::size_of::<u64>(),
                );
            }
        }
    }
}

/// SIGUSR1 handler installed in the fork-mode child post-fork. Flips
/// the per-process global [`STOP`] so `worker_main`'s outer loop
/// exits at the next `stop_requested` check.
pub(super) extern "C" fn sigusr1_handler(_: libc::c_int) {
    STOP.store(true, Ordering::Relaxed);
}

/// Render an actionable hint for a failed
/// `mmap(MAP_SHARED | MAP_ANONYMOUS)` call based on the observed
/// `errno`. Shared between the futex-region mmap and the
/// iter_counters mmap in [`WorkloadHandle::spawn`] so the two
/// sites emit identical hint text per errno — a drift would mean
/// two related failures produce inconsistent remediation advice.
///
/// Takes `Option<i32>` (the output of `std::io::Error::raw_os_error`)
/// so an unrecognised errno folds cleanly through the `_ => ""`
/// arm without forcing callers to `unwrap`.
///
/// The leading space on every non-empty arm lets callers format
/// as `"...failed: {errno}{hint};"` without having to add a
/// conditional separator — an empty hint disappears cleanly.
pub(super) fn mmap_shared_anon_errno_hint(errno: Option<i32>) -> &'static str {
    match errno {
        Some(libc::ENOMEM) => {
            " (ENOMEM: host is out of memory \
             or /proc/sys/vm/max_map_count is too low — \
             check `sysctl vm.max_map_count` and `free -h`)"
        }
        Some(libc::EPERM) => {
            " (EPERM: MAP_SHARED|MAP_ANONYMOUS \
             rejected by the kernel — check memory cgroup \
             limits and container seccomp policy)"
        }
        Some(libc::EINVAL) => {
            " (EINVAL: invalid length or \
             flag combination — verify num_workers > 0 so the \
             region size is non-zero, and that the total size \
             does not overflow usize)"
        }
        _ => "",
    }
}

#[cfg(test)]
mod testing;
#[cfg(test)]
mod tests_composed;
#[cfg(test)]
mod tests_fan_out;
#[cfg(test)]
mod tests_futex;
#[cfg(test)]
mod tests_grandchild;
#[cfg(test)]
mod tests_idle_churn;
#[cfg(test)]
mod tests_integration;
#[cfg(test)]
mod tests_lifecycle;
#[cfg(test)]
mod tests_mempolicy;
#[cfg(test)]
mod tests_misc;
#[cfg(test)]
mod tests_pcomm;
#[cfg(test)]
mod tests_sched_policy;
#[cfg(test)]
mod tests_spawn_guard;
#[cfg(test)]
mod tests_thread_mode;
#[cfg(test)]
mod tests_wake_chain;
