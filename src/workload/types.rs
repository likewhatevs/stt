// SPDX-License-Identifier: Apache-2.0
//! WorkType, Phase, and WorkTypeValidationError — pure data types
//! and pure-self methods extracted from the parent workload module.
//!
//! Re-exported by the parent module so external paths remain
//! `crate::workload::WorkType` etc. — the split is internal.

use std::sync::atomic::AtomicBool;
use std::time::Duration;

use super::WorkerReport;
use super::{AluWidth, FutexLockMode, SchedClass, WakeMechanism};
use super::{defaults, humantime_serde_helper};

/// A single phase in a [`WorkType::Sequence`] compound work pattern.
///
/// Workers loop through all phases in order, then repeat. Each phase
/// runs for its specified duration before advancing to the next.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// CPU spin for the given duration.
    Spin(#[serde(with = "humantime_serde_helper")] Duration),
    /// Sleep (thread::sleep) for the given duration.
    Sleep(#[serde(with = "humantime_serde_helper")] Duration),
    /// Yield (sched_yield) repeatedly for the given duration.
    Yield(#[serde(with = "humantime_serde_helper")] Duration),
    /// Simulated I/O (write 64 KB to a tempfile + 100 us sleep) for
    /// the given duration. The tempfile lives on whatever filesystem
    /// `std::env::temp_dir()` returns; on the ktstr guest's tmpfs the
    /// write is a page-cache memcpy and the sleep provides the
    /// blocking behavior that real disk fsync would cause.
    /// `WorkType::IoSyncWrite` (the standalone variant) is the disk-IO
    /// counterpart that opens `/dev/vda` directly.
    Io(#[serde(with = "humantime_serde_helper")] Duration),
}


/// What each worker process does during a scenario.
///
/// Different work types exercise different scheduler code paths:
/// CPU-bound, yield-heavy, I/O, bursty, or inter-process communication.
///
/// Variants ending in `Churn` cycle their target setting WITHOUT
/// ordering (random per-iteration); variants ending in `Sweep`
/// rotate through an ordered list or range deterministically. See
/// the module-level "Churn vs Sweep" section for the convention's
/// rationale and the runtime contract for each suffix.
///
/// # Migration: `IoSync` was replaced
///
/// `IoSync` was replaced by [`IoSyncWrite`](Self::IoSyncWrite),
/// [`IoRandRead`](Self::IoRandRead), and [`IoConvoy`](Self::IoConvoy).
/// The old `IoSync` simulated IO via tmpfs+sleep — write 64 KB to
/// a temp file (page-cache memcpy on tmpfs) then sleep 100 µs to
/// imitate disk-fsync latency. The new variants do real
/// block-device IO on `/dev/vda` with `O_SYNC`/`O_DIRECT`
/// (sector-aligned 4 KiB pread/pwrite, optional `fdatasync`),
/// so the kernel paths under stress are the actual virtio-blk
/// submit/complete + BIO routing paths rather than a synthetic
/// page-cache + nanosleep loop. Tests that depended on the old
/// page-cache + sleep behavior should use a [`Sequence`](Self::Sequence)
/// with [`Phase::Sleep`] (and an arbitrary CPU phase) to model
/// the simulated-IO-completion pause without doing real disk IO.
///
/// ```
/// # use ktstr::workload::WorkType;
/// let wt = WorkType::from_name("SpinWait").unwrap();
/// assert!(matches!(wt, WorkType::SpinWait));
///
/// let bursty = WorkType::bursty(
///     std::time::Duration::from_millis(10),
///     std::time::Duration::from_millis(5),
/// );
/// assert!(matches!(bursty, WorkType::Bursty { .. }));
///
/// assert!(WorkType::from_name("nonexistent").is_none());
/// ```
///
/// IO variants share the [`IoBacking`] open path but differ in the
/// open flag + IO shape used to detect them:
///
/// - [`IoSyncWrite`](Self::IoSyncWrite): `O_SYNC` + sequential
///   `pwrite` bursts followed by `fdatasync`.
/// - [`IoRandRead`](Self::IoRandRead): `O_DIRECT` + random `pread`
///   to a logical-block-aligned scratch buffer.
/// - [`IoConvoy`](Self::IoConvoy): `O_DIRECT` + interleaved
///   sequential `pwrite` and random `pread`, with an `fdatasync`
///   every 16 iterations (the pathology cadence).
///
/// ```
/// # use ktstr::workload::{WorkType, WorkloadConfig};
/// let cfg = WorkloadConfig {
///     work_type: WorkType::IoConvoy,
///     ..Default::default()
/// };
/// assert!(matches!(cfg.work_type, WorkType::IoConvoy));
/// ```
///
/// The `VariantNames` derive generates `WorkType::VARIANTS: &[&str]`
/// at compile time from the enum arm names, which this module
/// re-exposes as [`WorkType::ALL_NAMES`] so a new variant is picked
/// up automatically without editing a parallel list.
#[derive(Debug, Clone, strum::VariantNames, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
// `#[serde(bound(...))]` overrides the auto-derived lifetime bound. The
// `Custom` variant is `#[serde(skip)]` (it carries a non-portable `fn`
// pointer + a `&'static str` name), but the derive still walks every
// variant's field types to compute the implied `Deserialize<'de>`
// bound. Without this override the compiler infers `'de: 'static`
// from `Custom { name: &'static str }`, which makes the type
// unusable from any non-`'static` deserializer (i.e. all real
// deserializers). Explicit empty bounds tell serde to skip the
// auto-inference; the skipped variant is never deserialized so no
// `&'static str` ever needs to be reconstructed.
#[serde(bound(deserialize = ""))]
pub enum WorkType {
    /// Tight CPU spin loop (1024 iterations per cycle).
    SpinWait,
    /// Repeated sched_yield with minimal CPU work.
    YieldHeavy,
    /// CPU spin burst followed by sched_yield.
    Mixed,
    /// Synchronous write workload against a real block device. Each
    /// iteration issues 16 × 4 KB pwrites totaling 64 KB at the
    /// worker's stripe offset (per-worker striping prevents fdatasync
    /// from coalescing across writers), then `fdatasync()`s. Drives
    /// fsync-heavy D-state cycles. Opens `/dev/vda` with `O_SYNC` once
    /// per worker; if `/dev/vda` is absent (host-side unit tests), a
    /// per-worker tempfile is opened with the same flags and used as
    /// the backing.
    IoSyncWrite,
    /// Random-read workload against a real block device. Each
    /// iteration issues a single 4 KB pread at a sector-aligned random
    /// offset within the device capacity. Opens `/dev/vda` with
    /// `O_DIRECT` once per worker; if `/dev/vda` is absent, a
    /// per-worker tempfile is opened with the same flags and used as
    /// the backing. Drives high-IOPS short-D-state cycles. Offsets
    /// come from a per-worker xorshift PRNG seeded from `tid`; no
    /// crate dependency on `rand`.
    IoRandRead,
    /// Interleaved sequential `pwrite` and random `pread` with
    /// periodic `fdatasync` via `O_DIRECT`. Each iteration alternates
    /// between a 4 KB pwrite at the worker's monotonic sequential
    /// cursor and a 4 KB pread at a random offset; `fdatasync()`
    /// runs every 16 iterations. Opens `/dev/vda` (or tempfile
    /// fallback) with `O_DIRECT` once per worker.
    ///
    /// The convoy pathology (writes batching behind a flush
    /// barrier) requires buffered writes; v0 uses direct IO and so
    /// does not yet exhibit the full pathology — see the
    /// follow-up tracked in the project queue for the buffered-IO
    /// variant.
    IoConvoy,
    /// Work hard for `burst_duration`, sleep for `sleep_duration`,
    /// repeat. Frees CPUs during sleep for borrowing. Both fields
    /// use [`Duration`] (humantime-serialised) so call sites and
    /// captured configs carry units explicitly, matching
    /// [`WakeChain`](Self::WakeChain) and
    /// [`IdleChurn`](Self::IdleChurn).
    Bursty {
        /// Wall-clock duration of CPU work between sleeps.
        /// Default 50ms (see [`defaults::BURSTY_BURST_DURATION`]).
        #[serde(with = "humantime_serde_helper")]
        burst_duration: Duration,
        /// Wall-clock duration of each sleep period; the worker
        /// off-CPUs via `thread::sleep`. Default 100ms (see
        /// [`defaults::BURSTY_SLEEP_DURATION`]).
        #[serde(with = "humantime_serde_helper")]
        sleep_duration: Duration,
    },
    /// CPU burst then 1-byte pipe exchange with a partner worker. Sleep
    /// duration depends on partner scheduling, exercising cross-CPU wake
    /// placement. Requires even num_workers; workers are paired (0,1), (2,3), etc.
    PipeIo { burst_iters: u64 },
    /// Paired futex wait/wake between partner workers. Each iteration does
    /// `spin_iters` of CPU work then wakes the partner and waits on the
    /// shared futex word. Exercises the non-WF_SYNC wake path.
    /// Requires even num_workers.
    FutexPingPong { spin_iters: u64 },
    /// Strided read-modify-write over a buffer, sized to pressure the L1
    /// cache. Each worker allocates its own buffer post-fork.
    CachePressure { size_kb: usize, stride: usize },
    /// Cache pressure burst followed by sched_yield(). Tests scheduler
    /// re-placement after voluntary yield with a cache-hot working set.
    CacheYield { size_kb: usize, stride: usize },
    /// Cache pressure burst then 1-byte pipe exchange with a partner
    /// worker. Combines cache-hot working set with cross-CPU wake
    /// placement. Requires even num_workers.
    CachePipe { size_kb: usize, burst_iters: u64 },
    /// 1:N fan-out wake pattern without cache pressure. One messenger per
    /// group does CPU spin work then wakes N receivers via FUTEX_WAKE.
    /// Receivers measure wake-to-run latency as the interval from
    /// stamping `before_block = Instant::now()` just before the wait
    /// loop to observing the futex generation advance. Unlike
    /// [`FanOutCompute`](Self::FanOutCompute), there is no shared messenger
    /// timestamp — the measurement is receiver-local and excludes the
    /// messenger's pre-wake delay. For cache-aware fan-out with matrix
    /// multiply work, see `FanOutCompute`. Requires num_workers divisible
    /// by (fan_out + 1).
    FutexFanOut { fan_out: usize, spin_iters: u64 },
    /// Compound work pattern: loop through phases in order, repeat.
    /// Each phase runs for its duration before the next starts.
    Sequence { first: Phase, rest: Vec<Phase> },
    /// Rapid fork+_exit cycling. Each iteration forks a child that
    /// immediately calls _exit(0). Parent waitpid's then repeats.
    /// Exercises wake_up_new_task, do_exit, wait_task_zombie.
    ForkExit,
    /// Cycle nice level from -20 to 19 across iterations. Each
    /// iteration: spin_burst → setpriority → yield. Exercises
    /// reweight_task and dynamic priority reweighting. Skips negative
    /// nice values when CAP_SYS_NICE is absent.
    NiceSweep,
    /// Rapid self-directed sched_setaffinity to random CPUs from the
    /// effective cpuset. Each iteration: spin_burst → pick random CPU
    /// → sched_setaffinity → yield. Exercises affine_move_task and
    /// migration_cpu_stop.
    AffinityChurn { spin_iters: u64 },
    /// Cycle through scheduling policies each iteration. Each iteration:
    /// spin_burst → sched_setscheduler to next policy → yield. Cycles
    /// SCHED_OTHER → SCHED_BATCH → SCHED_IDLE (and SCHED_FIFO/SCHED_RR
    /// when CAP_SYS_NICE is available). Exercises __sched_setscheduler
    /// and scheduling class transitions.
    PolicyChurn { spin_iters: u64 },
    /// Messenger/worker fan-out with compute work. One messenger per group
    /// wakes `fan_out` workers via shared futex. After recording the
    /// wake-to-run latency, each worker sleeps for `sleep_usec`
    /// microseconds (simulating think time), then does `operations`
    /// matrix multiplications over a `cache_footprint_kb`-sized working
    /// set. Wake-to-run latency is the interval from the messenger's
    /// timestamp to the worker observing the generation advance.
    /// Requires num_workers divisible by (fan_out + 1).
    FanOutCompute {
        fan_out: usize,
        cache_footprint_kb: usize,
        operations: usize,
        sleep_usec: u64,
    },
    /// Rapid page fault cycling. Workers mmap a `region_kb` KB region with
    /// `MADV_NOHUGEPAGE` (forcing 4 KB pages), touch `touches_per_cycle`
    /// random pages via write faults, then `MADV_DONTNEED` to zap PTEs and
    /// repeat. Exercises `do_anonymous_page`, page allocator contention,
    /// and TLB pressure on migration.
    PageFaultChurn {
        region_kb: usize,
        touches_per_cycle: usize,
        spin_iters: u64,
    },
    /// N-way futex mutex contention. `contenders` workers per group contend
    /// on a shared `AtomicU32` via CAS acquire / `FUTEX_WAIT` on failure.
    /// Loop: `spin_burst(work_iters)` → CAS acquire → `spin_burst(hold_iters)`
    /// → store 0 + `FUTEX_WAKE(1)`. Exercises convoy effect, lock-holder
    /// preemption cascading stalls, and futex wait/wake contention paths.
    MutexContention {
        contenders: usize,
        hold_iters: u64,
        work_iters: u64,
    },
    /// User-supplied work function. The function receives a reference to
    /// the stop flag and returns a [`WorkerReport`] when signaled.
    /// Function pointers are fork-safe (`Copy`), so `Custom` works with
    /// the fork-based worker model without serialization.
    ///
    /// `name` identifies this work type in logs and sidecar metadata.
    /// [`from_name`](Self::from_name) returns `None` for custom names.
    ///
    /// **Telemetry contract:** `Custom` runs the user closure to
    /// completion and returns its `WorkerReport` verbatim. None of the
    /// built-in per-iteration instrumentation runs for this variant —
    /// neither the reservoir-sampled wake latencies, the shared-memory
    /// `iter_slot` publish that host sampling reads, nor the periodic
    /// max-gap tracking. The custom closure owns its own telemetry and
    /// must populate the [`WorkerReport`] fields it wants measured
    /// (`iterations`, `resume_latencies_ns`, `max_gap_ns`, etc.); any
    /// field left at `WorkerReport::default()` is reported as zero by
    /// downstream evaluation. Assertions like
    /// [`assert_not_starved`](crate::assert::assert_not_starved) that
    /// compute wake-latency percentiles will produce zero/degenerate
    /// numbers against a `Custom` report that did not record them.
    ///
    /// **Process-group lifecycle (per [`CloneMode`]):**
    ///
    /// _Fork mode_ — every worker calls `setpgid(0, 0)` immediately
    /// after fork, giving the worker its own process group
    /// (`pgid == worker_pid`). Any child processes the custom
    /// closure forks (a helper binary via `execv`, a subshell via
    /// `sh -c`, etc.) inherit that pgid unless they explicitly
    /// change it. On teardown, `stop_and_collect` issues
    /// `killpg(worker_pid, SIGKILL)` unconditionally (on both the
    /// graceful-exit and StillAlive-escalation paths) and
    /// [`WorkloadHandle::drop`] issues another `killpg` on handle
    /// teardown, so **every descendant a `Custom` closure spawns
    /// will be SIGKILLed at worker teardown** — there is no opt-out.
    /// Closures that need children to outlive the worker must
    /// either detach them from the worker's pgid
    /// (`setpgid(child_pid, 0)` after fork) or wait on them
    /// explicitly before returning the [`WorkerReport`]. The
    /// grandchild reaping tests in this module pin this sweep
    /// end-to-end.
    ///
    /// _Thread mode_ — `setpgid(0, 0)` does NOT run; thread workers
    /// share the test runner's pgid and cannot have one of their
    /// own (pgid is per-process / per-tgid). `killpg`-based cleanup
    /// is therefore unavailable: if a Thread-mode `Custom` closure
    /// forks helpers (e.g. via `Command::spawn`), those helpers
    /// inherit the test runner's pgid and will not be reaped on
    /// worker teardown. **You own teardown for any helpers a
    /// Thread-mode `Custom` closure spawns** — wait on them before
    /// returning, or arrange explicit kill/wait before returning
    /// the [`WorkerReport`].
    ///
    /// **Thread-mode prohibition on process-scoping syscalls:**
    /// under Thread mode, the closure runs as a thread inside the
    /// parent (test-runner) process, sharing pid/tgid, the signal-
    /// disposition table, the file descriptor table, cwd, and
    /// every other process-scoped attribute with every sibling
    /// worker AND with the test harness. Do NOT call
    /// `_exit()`/`exit()`, `fork()`/`vfork()`/`clone()`,
    /// `setpgid()`/`setsid()`, `execve()`, `chdir()`/`chroot()`,
    /// `setresuid()`/`setresgid()`, `prctl(PR_SET_*)` or any other
    /// process-scoping syscall — these affect the entire process,
    /// including all sibling workers and the test runner itself,
    /// and will produce silent cross-worker corruption,
    /// unexpected test-harness exits, or both. The supported
    /// shutdown contract is: observe the `&AtomicBool` argument's
    /// `stop.load()` flag and return the [`WorkerReport`] when it
    /// flips. This is a runtime contract, not a static check —
    /// `Custom` closures are arbitrary user code and the framework
    /// cannot detect violations at spawn time. If your workload
    /// genuinely needs `_exit`/`fork`/etc., use [`CloneMode::Fork`]
    /// where each worker IS its own process. The
    /// [`WorkType::ForkExit`] + [`CloneMode::Thread`] combination
    /// is rejected at spawn time precisely because of this — see
    /// [`WorkloadHandle::spawn`].
    ///
    /// **Serde:** the `Custom` variant is `#[serde(skip)]` because
    /// the `run` field is a `fn` pointer that has no portable wire
    /// format. Serializing a `WorkloadConfig` with `WorkType::Custom`
    /// emits an error; persisted configs (e.g. captured via
    /// `cargo ktstr export`) must use a built-in variant. Test
    /// authors who want a custom worker should keep `WorkType::Custom`
    /// inline in the test body and not roundtrip the config.
    #[serde(skip)]
    Custom {
        name: String,
        run: fn(&AtomicBool) -> WorkerReport,
    },
    /// One waker, N waiters on a SINGLE global futex word, repeated
    /// in batches with a sleep gap. Distinct from
    /// [`FutexFanOut`](Self::FutexFanOut) which uses one futex per
    /// fan-out group: ThunderingHerd parks every worker on the same
    /// queue, so a single `FUTEX_WAKE` rouses the entire herd
    /// simultaneously. Exercises the broadcast-wake path through
    /// `try_to_wake_up` and the scheduler's ability to spread the
    /// woken cohort across CPUs without convoying.
    ///
    /// The first worker (index 0) is the waker; the remaining
    /// `num_workers - 1` are waiters. Pick `waiters >= 5` so the
    /// herd (5) + waker (1) = 6 tasks saturates a 4-core host,
    /// making convoy effects observable; scale up further on
    /// larger hosts so the runnable cohort exceeds the cgroup's
    /// CPU budget. `worker_group_size = num_workers` so every
    /// worker shares the same shared-memory region; reuses the
    /// existing futex MAP_SHARED allocator.
    ThunderingHerd {
        /// Number of waiter workers (the herd). Must satisfy
        /// `num_workers == waiters + 1` (1 waker + waiters).
        waiters: usize,
        /// Total batches of wake-and-sleep cycles before the work
        /// loop ends. The waker emits `FUTEX_WAKE(INT_MAX)` once
        /// per batch.
        batches: u64,
        /// Inter-batch sleep on the waker (ms). Gives waiters a
        /// chance to re-park before the next thundering wake.
        inter_batch_ms: u64,
    },
    /// Three priority tiers contending for one shared lock. `low`
    /// workers acquire the lock and hold it while doing CPU work;
    /// `medium` workers do non-blocking CPU work (no lock) at a
    /// higher priority so they can preempt `low`; `high` workers
    /// try to acquire the lock at top priority. When `medium` keeps
    /// preempting `low`, `high` waits on the lock indefinitely —
    /// classic priority inversion.
    ///
    /// `pi_mode = FutexLockMode::Pi` uses `FUTEX_LOCK_PI` (PI-aware
    /// mutex); kernel boosts `low` to `high`'s priority for the
    /// duration of the hold, which both unblocks `high` and pins
    /// `medium` from preempting. `FutexLockMode::Plain` uses a plain
    /// futex with no boost — the inversion goes uncorrected.
    /// Tests both halves of the rt_mutex PI chain under the same
    /// workload shape.
    ///
    /// Requires same-CPU pinning (e.g. [`AffinityIntent::SingleCpu`])
    /// for `medium` to actually preempt `low`. Without pinning, the
    /// scheduler distributes the priorities across CPUs and the
    /// inversion never materialises.
    ///
    /// `worker_group_size = high_count + medium_count + low_count`
    /// so all three tiers share one futex region.
    PriorityInversion {
        /// Number of high-priority workers. Each acquires the
        /// shared lock at top priority.
        high_count: usize,
        /// Number of medium-priority workers. Run at a priority
        /// above `low_count` so they preempt the lock holder.
        medium_count: usize,
        /// Number of low-priority workers. Each holds the shared
        /// lock during its `hold_iters` CPU burst.
        low_count: usize,
        /// CPU-spin iterations a `low` worker burns while holding
        /// the lock.
        hold_iters: u64,
        /// CPU-spin iterations every worker burns between
        /// lock-acquire attempts (`high`/`low`) or between
        /// non-blocking work cycles (`medium`).
        work_iters: u64,
        /// Whether the workload uses a PI-aware futex (`Pi`,
        /// invokes `FUTEX_LOCK_PI` and the rt_mutex PI boost
        /// chain in `kernel/futex/pi.c`) or a plain non-PI futex
        /// (`Plain`, uncorrected inversion). See [`FutexLockMode`].
        pi_mode: FutexLockMode,
    },
    /// Producer / consumer pipeline with deliberately-unbalanced
    /// rates. `producers` workers push items at `produce_rate_hz`;
    /// `consumers` workers pop items and burn `consume_iters` of
    /// CPU work per pop. When `producers * produce_rate_hz`
    /// exceeds `consumers * (1 / consume_time)`, the queue grows
    /// monotonically toward `queue_depth_target`, exercising
    /// scheduler unfairness under sustained backpressure.
    ///
    /// The shared queue is an SPSC/MPSC ring buffer in MAP_SHARED
    /// memory sized to `queue_depth_target * 8 bytes` (u64 slots).
    /// Worker indices `[0, producers)` are producers; indices
    /// `[producers, producers + consumers)` are consumers.
    /// `worker_group_size = producers + consumers`.
    ProducerConsumerImbalance {
        /// Number of producer workers feeding the shared queue.
        producers: usize,
        /// Number of consumer workers draining the shared queue.
        consumers: usize,
        /// Target rate per producer (items per second). Producers
        /// pace themselves with `nanosleep` between pushes.
        produce_rate_hz: u64,
        /// CPU-spin iterations a consumer burns per popped item.
        /// Sets the implicit consume rate as
        /// `1 / spin_time(consume_iters)`.
        consume_iters: u64,
        /// Queue capacity (number of u64 slots). Determines the
        /// shared-memory region size and the producer's drop /
        /// stall behaviour when the queue fills.
        queue_depth_target: u64,
    },
    /// `rt_workers` workers run as `SCHED_FIFO` at `rt_priority`
    /// burning 100% CPU with `burst_iters` CPU work per iteration
    /// (no yields). `cfs_workers` workers run as `SCHED_NORMAL` and
    /// try to do work in the same scheduling domain. Without DL
    /// server protection (sched_ext does not have one — see the
    /// scx_ext docs), the SCHED_NORMAL workers starve.
    ///
    /// Reproducer setup: pin both groups to the same CPU set
    /// (e.g. via [`AffinityIntent::SingleCpu`]), and on the host set
    /// `sysctl_sched_rt_runtime_us=-1` for unlimited RT bandwidth
    /// (otherwise the kernel rt_period throttle unstuck things
    /// after 0.95s).
    ///
    /// Worker indices `[0, rt_workers)` get `SCHED_FIFO` applied
    /// post-fork via `sched_setscheduler`; the remainder stay on
    /// `SCHED_NORMAL`. `worker_group_size = rt_workers + cfs_workers`.
    RtStarvation {
        /// Number of SCHED_FIFO workers. Each runs at `rt_priority`.
        rt_workers: usize,
        /// Number of SCHED_NORMAL (CFS) workers competing on the
        /// same CPU set. Expected to starve.
        cfs_workers: usize,
        /// SCHED_FIFO priority for the RT workers. Must be in
        /// `1..=99`; clamped at the apply site.
        rt_priority: i32,
        /// CPU-spin iterations every worker (RT and CFS) burns per
        /// iteration. RT workers don't yield — they monopolise
        /// the CPU until kernel-side preemption.
        burst_iters: u64,
    },
    /// Paired workers with mismatched scheduling classes share a
    /// single futex word for hand-off. The waker (worker index 0)
    /// runs as [`waker_class`](Self::AsymmetricWaker::waker_class);
    /// the wakee (worker index 1) runs as
    /// [`wakee_class`](Self::AsymmetricWaker::wakee_class). After
    /// `burst_iters` of CPU work the waker advances the futex word
    /// and `FUTEX_WAKE`s the wakee; the wakee blocks in
    /// `FUTEX_WAIT` between turns. Tests wake-affine placement
    /// when waker and wakee live in different scheduling classes
    /// (e.g. an RT waker waking an EXT wakee — does the scheduler
    /// place the wakee on the waker's CPU, the wakee's last CPU,
    /// or somewhere else?).
    ///
    /// `worker_group_size = 2`. Wake latency is recorded into the
    /// wakee's `resume_latencies_ns` reservoir using the same
    /// `before_block` → `cur != expected` measurement as
    /// [`FutexPingPong`](Self::FutexPingPong).
    AsymmetricWaker {
        /// Scheduling class for the waker (worker index 0).
        waker_class: SchedClass,
        /// Scheduling class for the wakee (worker index 1).
        wakee_class: SchedClass,
        /// CPU-spin iterations the waker burns before each wake.
        burst_iters: u64,
    },
    /// Pipeline of waker-wakee hops forming a ring of `depth` stages.
    /// Two wake mechanisms gated by the `wake` field — see
    /// [`WakeMechanism`] for kernel citations:
    ///
    /// - [`WakeMechanism::Pipe`] — anon-pipe ring (`depth` pipes
    ///   per chain). Wakes carry `WF_SYNC` via
    ///   `wake_up_interruptible_sync_poll`, biasing scheduler
    ///   placement against migration. Tests the `SCX_WAKE_SYNC`
    ///   path that scx variants must respect.
    ///
    /// - [`WakeMechanism::Futex`] — single shared futex word per
    ///   chain. The active stage advances the word and
    ///   `FUTEX_WAKE`s; the stage whose `pos` matches runs, others
    ///   re-park. No `WF_SYNC`.
    ///
    /// Worker indices are partitioned into `num_workers / depth`
    /// chains of `depth` workers each. `worker_group_size = depth`
    /// so the spawn-side allocates one independent futex region
    /// per chain. At the end of the chain the last worker loops
    /// back to the first, forming a ring so the work pattern can
    /// run for a long test window.
    ///
    /// To run multiple parallel chains, set `num_workers` to a
    /// multiple of `depth` greater than `depth` itself — the
    /// spawn-side derives the chain count from the ratio.
    ///
    /// When `wake == WakeMechanism::Pipe`, the spawn-side
    /// additionally allocates `depth` pipes per chain — see
    /// [`chain_pipe_depth`](Self::chain_pipe_depth) and the
    /// `chain_pipes` field on `SpawnGuard` (early-bail path) and
    /// [`WorkloadHandle`] (success path).
    ///
    /// Both [`CloneMode::Fork`] and [`CloneMode::Thread`] are
    /// supported for `WakeMechanism::Pipe`. On a successful spawn
    /// the chain-pipe fds transfer from the guard into
    /// [`WorkloadHandle`], and `WorkloadHandle::drop` closes them
    /// only after every worker is reaped (Fork) or joined (Thread).
    /// Under Thread mode each worker thread shares the parent's fd
    /// table, so the post-shutdown close is what guarantees workers
    /// finish their `read` / `write` ops before the fds become
    /// invalid.
    WakeChain {
        /// Number of workers per chain. Each worker waits for its
        /// predecessor's signal, does `work_per_hop` of CPU work,
        /// signals the next worker, and repeats.
        depth: usize,
        /// Selects the wake mechanism between stages — see
        /// [`WakeMechanism`].
        ///
        /// [`WakeMechanism::Pipe`] allocates one anonymous pipe
        /// per stage (a chain ring of `depth` pipes) and uses
        /// `write(1 byte)` / `read(1 byte)` (poll-stop-pollable)
        /// for stage handoffs. The kernel raises `WF_SYNC` on the
        /// wake because `anon_pipe_write` (`fs/pipe.c`) calls
        /// `wake_up_interruptible_sync_poll` (`include/linux/wait.h`)
        /// which expands to `__wake_up_sync_key` (`kernel/sched/wait.c`)
        /// and that passes `WF_SYNC` through `__wake_up_common_lock`
        /// to `try_to_wake_up`. `WF_SYNC` biases scheduler placement
        /// away from migrating the woken stage off the waker's CPU
        /// — testing the wake-affine cohabitation that scx variants
        /// must respect.
        ///
        /// [`WakeMechanism::Futex`] uses the existing futex-word
        /// ring: `FUTEX_WAKE` fans out to every parked worker on
        /// the same word, the active stage proceeds, the rest
        /// re-park. No `WF_SYNC`; the scheduler is free to
        /// migrate the woken stage.
        ///
        /// The `Pipe` path needs `depth` pipes per chain — see
        /// [`chain_pipe_depth`](Self::chain_pipe_depth) — and
        /// closes the inverse ends of every other stage's pipe in
        /// the worker post-fork. The kernel-side `WF_SYNC` raise
        /// is verified by reading the call chain:
        /// `anon_pipe_write` at `fs/pipe.c:431-601`,
        /// `wake_up_interruptible_sync_poll` at
        /// `include/linux/wait.h:246-247`, and
        /// `__wake_up_sync_key` at `kernel/sched/wait.c:186-193`.
        wake: WakeMechanism,
        /// Wall-clock CPU work each worker performs per stage
        /// before signalling the next. Use [`Duration`] to keep
        /// the unit visible at the call site (consistent with
        /// [`SchedPolicy::Deadline`]'s switch to `Duration`).
        #[serde(with = "humantime_serde_helper")]
        work_per_hop: Duration,
    },
    /// Workers allocate a `region_kb` KB region with `set_mempolicy`
    /// pinned to one node, touch every page in that region, then
    /// `mbind(MPOL_BIND)` the region to the next node in
    /// `target_nodes` and re-touch — moving the working set across
    /// NUMA nodes every `sweep_period_ms`. Exercises page migration
    /// (`migrate_pages` / `move_pages`), the kernel's NUMA-balancing
    /// path (`task_numa_work`), and scheduler placement decisions
    /// under sustained working-set churn.
    ///
    /// Each worker rotates independently through the same
    /// `target_nodes` list with a per-worker phase offset so the
    /// cohort doesn't bind every region to the same node at the
    /// same instant. `worker_group_size = None` (any worker count
    /// is valid; each worker mbinds its own region without shared
    /// state).
    NumaWorkingSetSweep {
        /// Size of the working-set region per worker (KB). Each
        /// worker allocates this much anonymous memory and re-binds
        /// it across NUMA nodes.
        region_kb: usize,
        /// Wall-clock interval between binds. After every
        /// `sweep_period_ms`, the worker rotates to the next node
        /// in `target_nodes` and `mbind`s the region.
        sweep_period_ms: u64,
        /// Ordered list of NUMA node IDs the working set rotates
        /// through. Empty list disables binding (the worker still
        /// touches the region every iteration; no migration is
        /// triggered). Single-node lists pin the region to one
        /// node permanently — useful as an A/B baseline against a
        /// rotating sweep.
        target_nodes: Vec<usize>,
    },
    /// Workers cycle their cgroup membership between sibling cgroups
    /// every `cycle_ms`, rewriting `cgroup.procs` to drive
    /// `sched_move_task` (`kernel/sched/core.c`) and the registered
    /// `scx_cgroup_move_task` ops callback. Distinct from
    /// [`AffinityChurn`](Self::AffinityChurn): that variant rotates
    /// `task_struct->cpus_ptr` (cpuset membership) and never moves
    /// the task between cgroup containers; `CgroupChurn` rotates
    /// the cgroup itself, which takes the cgroup_threadgroup_rwsem
    /// write lock and exercises the per-class `sched_move_task` /
    /// `task_change_group` callbacks. Zero coverage today.
    ///
    /// Sibling cgroups must already exist under the worker's parent
    /// cgroup with names `wt-cgroup-churn-<i>` for `i in
    /// 0..groups`; the test harness or scenario setup creates them.
    /// Each iteration the worker writes its tid to the next sibling
    /// in rotation. `worker_group_size = None` (any worker count
    /// valid; each worker rotates independently). Per-iteration
    /// budget is one `write` syscall to `cgroup.procs`.
    CgroupChurn {
        /// Number of sibling cgroups to rotate through. The harness
        /// creates `wt-cgroup-churn-0` … `wt-cgroup-churn-(groups-1)`
        /// before spawn.
        groups: usize,
        /// Wall-clock interval between cgroup rewrites (ms). Lower
        /// values increase contention on `cgroup_threadgroup_rwsem`
        /// and the per-class `task_change_group` paths.
        cycle_ms: u64,
    },
    /// Paired workers signal each other with `kill(partner,
    /// SIGUSR1)`. Each worker installs a SIGUSR1 handler via
    /// `sigaction`, then alternates: do `work_iters` of CPU work,
    /// fire `signals_per_iter` signals at the partner, repeat.
    /// Exercises `signal_wake_up_state` (`kernel/signal.c`) and the
    /// per-task `sighand->siglock`, which is distinct from the
    /// futex `pi_lock` path. The wake itself goes through
    /// `kick_process` / `smp_send_reschedule`, not
    /// `ttwu_queue_wakelist`.
    ///
    /// Workers are paired (0,1), (2,3), … so `worker_group_size = 2`
    /// and `num_workers` must be even. Partner tids are exchanged
    /// via the existing pair shared-memory region. The signal
    /// handler is a no-op SA_RESTART handler; its only purpose is
    /// to trip `TIF_SIGPENDING` on the partner and force the
    /// scheduler through the signal-delivery wake path.
    SignalStorm {
        /// Number of `kill(partner, SIGUSR1)` calls per iteration.
        signals_per_iter: u64,
        /// CPU-spin iterations between bursts of signals.
        work_iters: u64,
    },
    /// Mixed RT + CFS preemption pressure. One worker per group runs
    /// as `SCHED_FIFO` doing `rt_burst_iters` of CPU work followed
    /// by `clock_nanosleep(rt_sleep_us)`; the remaining
    /// `cfs_workers` workers run as `SCHED_NORMAL` and spin
    /// continuously. Each RT wake (post-`nanosleep`) hits
    /// `wakeup_preempt` (`kernel/sched/core.c`) → `resched_curr`,
    /// preempting the CFS worker on the same CPU. Drives sustained
    /// `nonvoluntary_ctxt_switches` on the CFS workers.
    ///
    /// Distinct from [`RtStarvation`](Self::RtStarvation) which
    /// monopolises the CPU at 100% RT (and relies on
    /// `sysctl_sched_rt_runtime_us=-1`) and from
    /// [`PriorityInversion`](Self::PriorityInversion) which uses a
    /// PI-aware lock chain. `PreemptStorm` is the
    /// "RT-flickers-and-preempts" pathology: short bursts at high
    /// frequency, no monopolisation.
    ///
    /// `worker_group_size = cfs_workers + 1`. Worker index 0 in
    /// each group is the RT worker; indices 1..=cfs_workers are
    /// CFS spinners. RT priority defaults to 1 (lowest above
    /// `SCHED_NORMAL`); raise the priority via the host
    /// `RLIMIT_RTPRIO` and `CAP_SYS_NICE` are present.
    PreemptStorm {
        /// Number of CFS spinners per group. Set to the host CPU
        /// count for full preemption coverage.
        cfs_workers: usize,
        /// CPU-spin iterations the RT worker burns between
        /// nanosleep gaps.
        rt_burst_iters: u64,
        /// `clock_nanosleep` interval between RT bursts (us). 1000
        /// gives ~1 kHz RT preemption rate.
        rt_sleep_us: u64,
    },
    /// Producers / consumers connected by a single eventfd +
    /// epoll_wait pair. Producers `write(eventfd, &1u64)` in a
    /// burst loop; consumers wait in `epoll_wait(maxevents=1)`,
    /// `read` the counter, and burn one CPU-burst before re-arming
    /// the wait. Exercises `__wake_up_common` (`kernel/sched/wait.c`)
    /// with exclusive autoremove — ONE wake per event, distinct
    /// from [`ThunderingHerd`](Self::ThunderingHerd)'s broadcast
    /// futex wake. Hits `scx_select_cpu_dfl` WITHOUT the
    /// `SCX_WAKE_SYNC` fast-path because `epoll_wait` is not a
    /// sync wakeup primitive.
    ///
    /// `worker_group_size = producers + consumers`; needs shared
    /// memory for the eventfd / epoll fd handoff between sibling
    /// workers. Producers' `events_per_burst` controls how many
    /// `write`s they issue back-to-back before one `nanosleep` gap
    /// (paces production rate without per-event sleep overhead).
    EpollStorm {
        /// Number of producer workers per group. Each writes
        /// `events_per_burst` events per cycle.
        producers: usize,
        /// Number of consumer workers per group. Each does one
        /// `epoll_wait` + `read` + spin-burst per event.
        consumers: usize,
        /// Producer burst size (events per write loop).
        events_per_burst: u64,
    },
    /// Workers rotate `sched_setaffinity` across NUMA nodes every
    /// `period_ms`. Reads online NUMA nodes from
    /// `/sys/devices/system/node/online` at startup, then cycles
    /// the worker through one node's CPUs per period. Exercises
    /// task migration via `select_task_rq` (`kernel/sched/core.c`)
    /// with the `WF_MIGRATED` flag and, on sched_ext, the
    /// `SCX_OPS_BUILTIN_IDLE_PER_NODE` branch of `scx_select_cpu_dfl`.
    ///
    /// Distinct from [`NumaWorkingSetSweep`](Self::NumaWorkingSetSweep)
    /// which moves the working-set MEMORY across nodes via `mbind`;
    /// `NumaMigrationChurn` moves the TASK across nodes via
    /// `sched_setaffinity`. `worker_group_size = None`. On hosts
    /// with one NUMA node, the variant degenerates to a no-op
    /// (every iteration re-pins to the same node).
    NumaMigrationChurn {
        /// Wall-clock interval between affinity rotations (ms).
        period_ms: u64,
    },
    /// CPU burst for `burst_duration` followed by `nanosleep` for
    /// `sleep_duration`, repeated. Exercises task off-CPU/on-CPU
    /// transitions: `nanosleep` dequeues the worker into
    /// TASK_INTERRUPTIBLE; on the pinned CPU, when no other tasks
    /// are runnable, `__pick_next_task` selects the idle class
    /// (`pick_task_idle` at `kernel/sched/idle.c:480`); on
    /// `nanosleep` expiry the hrtimer callback `hrtimer_wakeup`
    /// calls `wake_up_process` → `try_to_wake_up`.
    ///
    /// # When to use IdleChurn
    ///
    /// Reach for IdleChurn when the test needs the kernel's
    /// hrtimer + idle-class scheduling path — exercising the
    /// nanosleep → schedule → idle → hrtimer-wakeup loop that
    /// the idle thread itself observes. Concrete pickers:
    ///
    /// - You need to measure scheduler wake placement after a
    ///   `TASK_INTERRUPTIBLE` dequeue — IdleChurn blocks via
    ///   `nanosleep` directly, the same hrtimer path the idle
    ///   thread enters when no work is runnable.
    /// - You need to drive the tick-stop / C-state boundary on
    ///   the pinned CPU — sleeps > 1ms exercise the full idle
    ///   path including the tickless branch (`tick_nohz_idle_enter`).
    /// - You're A/B-testing scheduler behavior on the idle-class
    ///   transition specifically (e.g. scx_lavd's idle-CPU
    ///   selection vs scx_simple's), and need a reproducible
    ///   workload that passes through the kernel idle path.
    ///
    /// Choose [`Bursty`](Self::Bursty) instead when:
    ///
    /// - The test measures THROUGHPUT under burst-then-sleep
    ///   patterns at the millisecond regime — Bursty uses
    ///   `thread::sleep` (which is itself nanosleep-backed but
    ///   coarser-grained in libc) and matches the existing
    ///   pthread/std-lib timing model most application
    ///   benchmarks assume.
    /// - The test needs >1 ms sleeps without caring about the
    ///   idle-class transition specifically — Bursty is the
    ///   simpler variant and has fewer caveats below.
    ///
    /// IdleChurn is distinct from variants that block on
    /// futex/pipe (FutexPingPong, PipeIo, WakeChain) — those
    /// route the wake through `futex_wake` /
    /// `wake_up_interruptible_sync_poll`, exercising
    /// inter-task-coordination paths. IdleChurn's blocking
    /// primitive is the hrtimer expiry, not a peer's wake call.
    ///
    /// # Caveat impacts at a glance
    ///
    /// NB: the five bullets below mirror the detailed
    /// sections that follow — keep both in sync when
    /// editing.
    ///
    /// The five sections below detail the kernel-side mechanisms.
    /// For test authors picking thresholds, the practical
    /// per-iteration impact is:
    ///
    /// - **Timer slack** — observed sleep is
    ///   `sleep_duration + current->timer_slack_ns`. Default
    ///   slack is 50µs, so a `sleep_duration` of 80µs produces
    ///   ~130µs actual sleep. For `sleep_duration` ≥ 1ms the
    ///   slack is < 5% noise; for sub-100µs sleeps the slack
    ///   floor dominates.
    /// - **Task off-CPU vs CPU idle** — the worker off-CPUs
    ///   every iteration regardless of placement, but the CPU
    ///   only enters the idle class under exclusive pinning.
    ///   Without `AffinityIntent::SingleCpu` the CPU runs
    ///   another runnable task during the sleep window — the
    ///   variant tests TASK transitions, not CPU-idle.
    /// - **Degenerate-input rejection** — spawn-side rejects
    ///   `Duration::ZERO` for either field with an actionable
    ///   bail message. `burst_duration=0` collapses the loop to
    ///   pure nanosleep (worker accrues no runtime);
    ///   `sleep_duration=0` overlaps with two existing variants
    ///   — [`SpinWait`](Self::SpinWait) is the bail message's
    ///   forwarding target (no idle path exercised, pure spin
    ///   loop), but the kernel-level semantic is closer to
    ///   [`YieldHeavy`](Self::YieldHeavy) since `nanosleep(0)`
    ///   still calls `set_current_state(TASK_INTERRUPTIBLE)` +
    ///   `schedule()` (sched_yield-equivalent).
    /// - **NO_HZ_FULL** — workers pinned to a CPU in the
    ///   `nohz_full=` mask see LOWER median
    ///   `resume_latencies_ns` (tick re-arm is skipped) but
    ///   heavier high-percentile tail (deferred jiffy-driven
    ///   work catchup). Mixing pinned-vs-unpinned workers
    ///   across the mask boundary produces a bimodal
    ///   distribution.
    /// - **vCPU-in-KVM** — wake latency aggregates guest +
    ///   host scheduler costs. `performance_mode=true` disables
    ///   HLT vmexits so the test measures guest scheduling in
    ///   isolation; `performance_mode=false` exercises the
    ///   cross-VM idle path but adds host-scheduler jitter
    ///   bounded by one host scheduler tick.
    ///
    /// # Task off-CPU is guaranteed; CPU idle is conditional
    ///
    /// IdleChurn exercises the **TASK off-CPU/back-on-CPU
    /// transition** on every iteration — NOT necessarily the CPU
    /// idle/exit transition. The two are distinct paths in the
    /// scheduler and a test must pick the one the design
    /// requires:
    ///
    /// - `do_nanosleep` at `kernel/time/hrtimer.c:2115-2148` calls
    ///   `set_current_state(TASK_INTERRUPTIBLE | TASK_FREEZABLE)`
    ///   then `schedule()`. The current task IS dequeued and goes
    ///   off-CPU on every iteration regardless of what else is
    ///   runnable. `nr_voluntary_ctxt_switches` ticks per
    ///   iteration unconditionally.
    /// - Whether the CPU enters the **idle class**
    ///   (`__pick_next_task` selecting `pick_task_idle`) depends
    ///   on what else is on the runqueue. If any other task is
    ///   runnable on the pinned CPU, `schedule()` picks it and
    ///   the CPU never idles for that iteration.
    ///
    /// Three concrete adversary scenarios where the CPU does NOT
    /// enter the idle class even though IdleChurn fired:
    ///
    /// 1. **Multi-worker on a single CPU** — IdleChurn with
    ///    `num_workers=2` and overlapping affinity runs A and B on
    ///    the same CPU. When A nanosleeps, B is runnable; CPU
    ///    runs B, never idles. The variant tests "worker churn"
    ///    rather than "CPU idle/exit transitions".
    /// 2. **Co-scheduled kernel threads** — kworker, ksoftirqd,
    ///    rcu_* kthreads (kthread_run on the same CPU) and
    ///    deferred-work softirqs run on every CPU. ksoftirqd is
    ///    woken from `wakeup_softirqd` (kernel/softirq.c) when
    ///    `irq_exit` observes pending softirqs after inline
    ///    processing — its wake frequency tracks irq load, not a
    ///    fixed cadence. Sleep durations short enough to overlap
    ///    with steady-state softirq backlog (e.g. NIC interrupt
    ///    pressure) may observe ksoftirqd preempting the
    ///    IdleChurn worker between iterations — diluting the
    ///    idle-transition signal.
    /// 3. **Sibling test workloads in the same LLC** — a peer
    ///    test pinned to a different CPU within the same LLC can
    ///    spawn kernel threads that get migrated onto IdleChurn's
    ///    CPU by the kernel's load balancer. The migration is
    ///    invisible to the IdleChurn worker but breaks the
    ///    "CPU is exclusive" assumption.
    ///
    /// **For TASK-off-CPU testing** (the default and the variant's
    /// guaranteed semantic): no special pinning required — every
    /// iteration off-CPUs the worker.
    ///
    /// **For CPU-idle-class testing**: ensure the worker has
    /// exclusive CPU affinity AND no co-scheduled kernel threads.
    /// Concrete recipe:
    ///
    /// - Use [`AffinityIntent::SingleCpu`] or a one-CPU `Exact`
    ///   mask so only this worker is pinned to the CPU.
    /// - Run under `performance_mode=true` so the CPU lock budget
    ///   reserves the CPU for this test.
    /// - Set `num_workers=1` (multiple IdleChurn workers on the
    ///   same CPU break the assumption — see scenario 1 above).
    /// - Be aware that kernel-side periodic work (RCU callbacks,
    ///   vmstat updates, watchdog ticks) still runs on every CPU
    ///   regardless of affinity — sub-millisecond sleeps will
    ///   sometimes observe a non-idle iteration even with
    ///   exclusive pinning.
    ///
    /// This is a runtime contract, not a static one. The
    /// spawn-side does not check the affinity policy because
    /// "exclusive" depends on the rest of the host's load,
    /// which the framework cannot observe at spawn time.
    ///
    /// # Timer slack expands the requested sleep
    ///
    /// The kernel adds `current->timer_slack_ns` to the requested
    /// `sleep_duration` inside `hrtimer_nanosleep` at
    /// `kernel/time/hrtimer.c:2162-2188`, specifically the
    /// `hrtimer_set_expires_range_ns(&t.timer, rqtp,
    /// current->timer_slack_ns)` call at L2170.
    /// `timer_slack_ns` is inherited from the parent at fork; the
    /// kernel default propagated from `init_task` is 50000ns
    /// (50µs, set at `init/init_task.c:172`). So:
    ///
    /// - `sleep_duration` is a **lower bound** on the observed
    ///   idle interval — actual sleep extends by up to
    ///   `current->timer_slack_ns` to let the kernel coalesce
    ///   timer wakeups.
    /// - Sub-50µs `sleep_duration` values do not produce sub-50µs
    ///   idle periods — the slack floor dominates.
    /// - **RT workers bypass slack.** Under
    ///   [`SchedPolicy::Fifo`] or [`SchedPolicy::RoundRobin`] the
    ///   kernel forces `timer_slack_ns` to 0
    ///   (`kernel/sched/syscalls.c:258`), so RT IdleChurn workers
    ///   get exact wake timing. CFS / SCHED_NORMAL workers
    ///   inherit the 50µs default.
    /// - IdleChurn calls `prctl(PR_SET_TIMERSLACK, 1)` ONLY when
    ///   the variant's
    ///   [`precise_timing`](Self::IdleChurn::precise_timing)
    ///   field is `true`. The default is `false`, preserving the
    ///   inherited 50µs slack for CFS workers. Set
    ///   `precise_timing: true` (or use the struct-literal form
    ///   directly — the [`idle_churn`](Self::idle_churn)
    ///   constructor leaves the field at its default) to shrink
    ///   slack to 1ns for sub-50µs `sleep_duration` measurements.
    ///   See the field's doc for the kernel-source citation that
    ///   explains why `1` (not `0`) is the value that narrows
    ///   slack.
    ///
    /// # Tick-stop boundary
    ///
    /// Sleeps > 1ms exercise the full idle path including tick
    /// stop and (on configured platforms) C-state entry —
    /// `tick_nohz_idle_enter`, `cpuidle_idle_call`, governor
    /// selection. Sub-millisecond sleeps still produce
    /// `sched_switch` transitions but skip the tick-stop branch
    /// because the tick is reprogrammed for the imminent
    /// expiry rather than stopped entirely.
    ///
    /// # NO_HZ_FULL alters wake observation
    ///
    /// Three NO_HZ kernel configurations affect wake latency
    /// differently:
    ///
    /// - `CONFIG_HZ_PERIODIC` — the periodic timer tick fires
    ///   every `1/CONFIG_HZ` seconds regardless of CPU state.
    ///   Wake-from-idle latency is bounded above by the tick
    ///   period; the kernel may choose to delay wakes to the
    ///   next tick. Most predictable wake population, useful for
    ///   strict-bound assertions.
    /// - `CONFIG_NO_HZ_IDLE` — tick stops when a CPU goes idle
    ///   but resumes immediately on any wake event. Wake latency
    ///   reflects the `TASK_INTERRUPTIBLE → TASK_RUNNING`
    ///   transition cost plus tick re-arming. This is the
    ///   default on modern x86_64 / arm64 distro kernels and the
    ///   posture ktstr's bundled `ktstr.kconfig` inherits (the
    ///   fragment does not override NO_HZ_*).
    /// - `CONFIG_NO_HZ_FULL` — for CPUs in the `nohz_full=` boot
    ///   parameter mask, the tick stays stopped even when one
    ///   task is runnable. Wake delivery routes through hrtimer
    ///   expiry alone; the kernel skips tick re-arm on wake when
    ///   no tick-dependent subsystem demands it
    ///   (`tick_nohz_idle_enter` at `kernel/time/tick-sched.c`),
    ///   so steady-state `resume_latencies_ns` reads LOWER on
    ///   nohz_full CPUs than on `NO_HZ_IDLE` CPUs.
    ///   The catch: deferred jiffy-driven work (RCU callbacks,
    ///   vmstat updates, watchdog ticks) accumulates while the
    ///   tick is stopped and produces visible long-tail jitter
    ///   when it eventually runs — manifesting as occasional
    ///   high-percentile spikes in the wake-latency distribution
    ///   even though the median drops.
    ///
    /// IdleChurn behavior is consistent under
    /// `CONFIG_NO_HZ_IDLE` (the default). On hosts with
    /// `CONFIG_NO_HZ_FULL`, samples from workers whose CPU is in
    /// the nohz_full mask are NOT directly comparable to samples
    /// from CPUs outside that mask — the populations differ in
    /// both the median (lower on nohz_full) and the tail (heavier
    /// on nohz_full from deferred-work catchup). Tests asserting
    /// precise idle-duration scheduler decisions (e.g. "tasks
    /// idle <1ms get latency-sensitive treatment") must either:
    ///
    /// - require NO_HZ_FULL on (and pin the worker into the mask),
    /// - require NO_HZ_FULL off (CONFIG_HZ_PERIODIC or
    ///   CONFIG_NO_HZ_IDLE), or
    /// - tolerate both populations with looser thresholds.
    ///
    /// The active mask is readable at runtime via
    /// `/sys/devices/system/cpu/nohz_full`. The file only
    /// exists when the kernel was built with
    /// `CONFIG_NO_HZ_FULL=y`; on a `CONFIG_NO_HZ_IDLE`-only
    /// kernel (the typical distro default) the file is absent
    /// and the test author can assume no nohz_full effects.
    /// IdleChurn does not adjust the mask itself, and mixing
    /// pinned-vs-unpinned workers in the same scenario produces
    /// a bimodal latency distribution if the host is configured
    /// for nohz_full.
    ///
    /// # vCPU-in-KVM amplifies wake latency
    ///
    /// ktstr tests run inside KVM guests. IdleChurn's
    /// `nanosleep` inside a guest vCPU has a layered cost:
    ///
    /// 1. Guest task calls `nanosleep` → guest kernel arms a
    ///    guest-side hrtimer.
    /// 2. Guest task off-CPUs (`TASK_INTERRUPTIBLE` →
    ///    `schedule()`).
    /// 3. Guest CPU idles → guest kernel issues `HLT` (or
    ///    `MWAIT` on x86, `WFI` on arm64).
    /// 4. The HLT either vmexits to host KVM or spins in-guest
    ///    (see perf-mode interaction below).
    /// 5. On vmexit: host KVM blocks the vCPU thread on a wait
    ///    queue.
    /// 6. Guest-side timer expires (in guest time) → host KVM
    ///    injects a timer interrupt → vCPU thread wakes →
    ///    vmenter back to guest.
    /// 7. Guest kernel's hrtimer ISR fires →
    ///    `wake_up_process` → guest scheduler reruns the
    ///    IdleChurn task.
    ///
    /// `resume_latencies_ns` (the dispatch arm subtracts
    /// `sleep_duration` to isolate scheduler-resume overhead)
    /// captures the SUM of guest scheduling cost +
    /// vmexit-vmenter round-trip + host scheduling cost. The
    /// SCHEDULER-UNDER-TEST is the GUEST scheduler, but the
    /// host's contribution can dominate under load.
    ///
    /// **Strict bound on host preemption.** The guest's hrtimer
    /// expiry routes through the emulated LAPIC (x86) or arch
    /// timer (arm64), both backed by host timers. If the host
    /// has descheduled the vCPU thread (PLE-induced eviction
    /// from a busy guest spinlock, host-side preemption by
    /// higher-priority work, or simple oversubscription), the
    /// guest's hrtimer CANNOT fire until the host re-runs the
    /// vCPU thread. This is a hard additional latency bound
    /// added on top of guest-side scheduling cost — the guest
    /// scheduler under test cannot be observed through
    /// IdleChurn while the host has preempted its vCPU.
    ///
    /// **Performance-mode interaction.** This subsection
    /// describes x86_64 only. ktstr's x86_64 VMM disables HLT
    /// vmexits when `performance_mode=true` (see
    /// `src/vmm/x86_64/kvm.rs::Vm::new` around the
    /// `KVM_X86_DISABLE_EXITS_HLT` enable_cap call). The
    /// aarch64 VMM accepts the `performance_mode` flag but does
    /// NOT configure WFI trap behavior (no HCR_EL2.TWI tweak in
    /// `src/vmm/aarch64/kvm.rs::Vm::new`), so on aarch64 every
    /// guest WFI exits to host regardless of `performance_mode`
    /// — IdleChurn always exercises the cross-VM idle path
    /// there. With HLT exits disabled (x86_64 only):
    ///
    /// - Step 4 stays in-guest: the vCPU spins on HLT without
    ///   vmexit, consuming its assigned host CPU slot. The
    ///   guest kernel still sees the CPU as idle, but the host
    ///   never blocks the vCPU thread.
    /// - Steps 5-6 collapse: no host wait queue, no
    ///   guest-time-aware injection. The host runs the vCPU
    ///   thread continuously, and the guest hrtimer expiry is
    ///   handled inside the running vCPU.
    /// - IdleChurn under `performance_mode=true` therefore
    ///   tests ONLY the guest's idle path. It does NOT
    ///   exercise the cross-VM idle / host-scheduler
    ///   interaction. This is the right config for measuring
    ///   guest scheduler decisions in isolation.
    ///
    /// With `performance_mode=false`, HLT vmexits fire and
    /// IdleChurn DOES test the cross-VM idle path — but the
    /// host scheduler's contribution to wake latency
    /// interferes with timing-sensitive guest measurements.
    ///
    /// **Test-author guidance:**
    ///
    /// - For tests measuring GUEST scheduler decisions in
    ///   isolation (e.g. `scx_lavd` idle-CPU selection): set
    ///   `performance_mode=true` so the host doesn't perturb
    ///   the measurement.
    /// - For tests measuring CROSS-VM idle (e.g. how the host
    ///   schedules a vCPU thread after a guest HLT): set
    ///   `performance_mode=false`, run on a dedicated host (no
    ///   noisy neighbors), and budget for host-scheduler-
    ///   contributed jitter.
    /// - On a heavily-loaded host (concurrent ktstr tests, or
    ///   noisy neighbors), `resume_latencies_ns` reflects host
    ///   contention even under `performance_mode=true` because
    ///   the vCPU thread itself can be preempted on the host
    ///   (the guest sees this as "the worker just took longer
    ///   than it should").
    ///
    /// Distinguishing host vs guest contribution requires
    /// host-side observation — e.g. `perf sched` on the vCPU
    /// thread, or comparing
    /// `/proc/<vcpu_tid>/status::voluntary_ctxt_switches`
    /// before vs after the test window.
    ///
    /// # Wake-latency interpretation
    ///
    /// `resume_latencies_ns` samples for IdleChurn capture the
    /// **scheduler-resume overhead** — the time the kernel spent
    /// scheduling the worker back on-CPU after the requested
    /// `sleep_duration` elapsed. The dispatch arm subtracts
    /// `sleep_duration` from the measured nanosleep elapsed time,
    /// leaving timer slack (default 50µs) plus
    /// `try_to_wake_up` → on-CPU latency. This isolates the
    /// signal a scheduler A/B test cares about: comparing
    /// `resume_latencies_ns` distributions across schedulers
    /// directly measures their idle-class → run-class transition
    /// behavior without the requested-sleep duration dominating
    /// the measurement.
    ///
    /// `saturating_sub` guards against the rare case where
    /// `elapsed < sleep_duration`. That can happen on early-EINTR
    /// returns or sub-tick measurement windows; saturating to 0
    /// matches the "no observable resume overhead" interpretation.
    ///
    /// Samples are comparable in DIRECTION to
    /// `resume_latencies_ns` from FutexPingPong, FutexFanOut,
    /// and other wake-pair variants (lower = better scheduler
    /// resume), but the IdleChurn distribution carries a
    /// ~50µs floor from `current->timer_slack_ns` that
    /// event-driven futex variants don't. Cross-variant
    /// absolute comparisons must subtract the slack floor or
    /// limit the comparison to the > P50 percentile where the
    /// slack contribution is dwarfed by tail latency.
    ///
    /// # Spawn-time validation
    ///
    /// The spawn path rejects `burst_duration == Duration::ZERO`
    /// (loop collapses to pure nanosleep, no runtime accrued)
    /// and `sleep_duration == Duration::ZERO` (loop degenerates
    /// to [`SpinWait`](Self::SpinWait), making the variant
    /// useless as an idle-path test).
    ///
    /// The `sleep_duration == 0` rejection deserves an
    /// implementation-rationale note: `nanosleep(0)` is NOT a
    /// no-op — the kernel still calls
    /// `set_current_state(TASK_INTERRUPTIBLE)` followed by
    /// `schedule()`, which produces sched_yield-equivalent
    /// semantics (yield to the next runnable task on the
    /// runqueue, return immediately). That overlaps with
    /// [`YieldHeavy`](Self::YieldHeavy) and provides no idle-path
    /// signal, so the rejection sends the caller to the variant
    /// that already covers the yield case. Both rejections
    /// produce actionable bail messages naming the field and
    /// the degenerate semantics — see the spawn-side check in
    /// `WorkloadHandle::spawn`.
    ///
    /// `worker_group_size = None` — every worker operates
    /// independently with no shared-memory group; see
    /// [`Self::worker_group_size`] for the framework-wide
    /// semantics.
    IdleChurn {
        /// Wall-clock duration of CPU work between idle
        /// periods. Use [`Duration`] to keep the unit visible at
        /// the call site, matching
        /// [`WakeChain`](Self::WakeChain)'s `work_per_hop`.
        /// Default 1ms (see
        /// [`defaults::IDLE_CHURN_BURST_DURATION`]). Short
        /// bursts (< 1ms) maximise idle-cycle frequency.
        #[serde(with = "humantime_serde_helper")]
        burst_duration: Duration,
        /// Wall-clock duration of each idle period. Lower bound
        /// — the kernel adds `timer_slack_ns` (~50µs) to the
        /// requested duration. Default 5ms (see
        /// [`defaults::IDLE_CHURN_SLEEP_DURATION`]). Sub-1ms
        /// values produce `sched_switch` transitions but skip
        /// tick-stop / C-state entry.
        #[serde(with = "humantime_serde_helper")]
        sleep_duration: Duration,
        /// Opt-in: shrink `current->timer_slack_ns` from the
        /// inherited 50µs default to 1ns at worker entry via
        /// `prctl(PR_SET_TIMERSLACK, 1)`. Default `false` so
        /// existing callers see the inherited slack the variant
        /// doc describes.
        ///
        /// When `true`, the IdleChurn dispatch arm calls
        /// `prctl(PR_SET_TIMERSLACK, 1)` once before the work
        /// loop. The kernel's PR_SET_TIMERSLACK arm at
        /// `kernel/sys.c:2645` sets `current->timer_slack_ns =
        /// arg2` when `arg2 > 0`; passing `0` is a RESET to
        /// `default_timer_slack_ns` (the inherited 50µs), so
        /// `1` is the smallest value that actually shrinks the
        /// slack. After the call, `hrtimer_nanosleep`
        /// (`kernel/time/hrtimer.c:2162-2188`) coalesces
        /// expiries within a 1ns window instead of the default
        /// 50µs, exposing the scheduler's true wake-resume
        /// latency for sub-100µs `sleep_duration` values.
        ///
        /// This setting is most useful when the test measures
        /// wake-latency distributions for sub-50µs sleeps,
        /// where the inherited slack would otherwise dominate
        /// the observed sleep time. For `sleep_duration` ≥ 1ms
        /// the slack contribution is < 5% noise and
        /// `precise_timing=true` makes no observable
        /// difference.
        ///
        /// **RT/DL workers ignore this setting.** The kernel
        /// guard at `kernel/sys.c:2646`
        /// (`if (rt_or_dl_task_policy(current)) break;`)
        /// makes `prctl(PR_SET_TIMERSLACK, ...)` a no-op for
        /// RT/DL tasks; their slack is independently forced to
        /// 0 at sched-class entry by
        /// `kernel/sched/syscalls.c:258`. Setting
        /// `precise_timing=true` for an RT IdleChurn worker is
        /// harmless but redundant.
        ///
        /// Field defaults to `false` so existing
        /// [`from_name("IdleChurn")`](Self::from_name) callers
        /// see the historical (inherited-slack) behaviour. Opt
        /// in via the struct-literal form
        /// `WorkType::IdleChurn { ..., precise_timing: true }`.
        precise_timing: bool,
    },
    /// Sustained high-IPC ALU workload. Each worker runs four
    /// independent multiply chains in parallel, with
    /// [`std::hint::black_box`] wrapping every step to prevent
    /// the optimizer from collapsing the chain into a closed-form
    /// expression. Distinct from [`SpinWait`](Self::SpinWait) —
    /// `SpinWait` issues `PAUSE` (`std::hint::spin_loop`) whose
    /// per-iteration retire is a single fused micro-op and which
    /// signals the front-end to back off, depressing the IPC the
    /// scheduler observes. `AluHot` retires real arithmetic at
    /// IPC ≥ 2.0 on every modern x86_64 / aarch64 core, so
    /// scheduler decisions that respond to per-task runtime
    /// characteristics (lavd's `lat_cri` per-task
    /// latency-criticality scoring) see a meaningfully different
    /// signal.
    ///
    /// The [`width`](Self::AluHot::width) field selects the
    /// data-path width — see [`AluWidth`] for the resolution
    /// rules and the AVX-512 / AMX caveats. Workers do NOT
    /// adjust frequency or voltage state themselves; the
    /// package-wide frequency throttle on x86_64 is a kernel-
    /// observable effect of running AVX-512 / AMX instructions.
    ///
    /// **v0:** runs scalar four-stream multiplies for ALL
    /// widths; the width selector shapes the dispatch for future
    /// SIMD intrinsics (follow-up #309-#314). No observable
    /// behavioral difference between widths today.
    ///
    /// `worker_group_size = None` (any worker count is valid;
    /// each worker runs an independent multiply chain). No
    /// shared-memory region; no per-iteration syscall overhead.
    AluHot {
        /// SIMD / scalar width selector for the multiply chain.
        /// See [`AluWidth`] for the per-variant data-path width
        /// and the runtime resolution rules.
        width: AluWidth,
    },
    /// Tight `PAUSE`-spin from a paired worker, intended to be
    /// pinned to two SMT siblings of the same physical core so
    /// the spinning thread contends for the core's shared
    /// front-end / execution resources with its sibling. Distinct
    /// from [`SpinWait`](Self::SpinWait) which is a single-
    /// position spin: `SmtSiblingSpin` requires
    /// [`worker_group_size`](Self::worker_group_size) `== 2` and
    /// is paired with an SMT-aware affinity that pins both
    /// workers to the two siblings of one physical core.
    ///
    /// The framework provides
    /// [`AffinityIntent::SmtSiblingPair`] for this purpose: the
    /// scenario engine resolves it against the host topology
    /// (using sysfs's
    /// `/sys/devices/system/cpu/cpu_a/topology/thread_siblings_list`
    /// when the topology was built from sysfs) and produces a
    /// 2-CPU [`AffinityIntent::Exact`] for the spawn pipeline.
    /// Resolving on a non-SMT host (`threads_per_core == 1`)
    /// returns an explicit error rather than silently degrading.
    /// Test authors who want exact CPU IDs (e.g. comparing
    /// same-core vs. cross-core behaviour on a known topology)
    /// can still hand-pick via [`AffinityIntent::Exact`].
    ///
    /// Without one of those affinity intents the variant
    /// degenerates to two independent [`SpinWait`](Self::SpinWait)
    /// workers and exercises no SMT contention.
    ///
    /// `worker_group_size = Some(2)` so paired workers share
    /// the position metadata the dispatch arm uses to assert
    /// the partner exists; the variant carries no shared-memory
    /// region itself.
    SmtSiblingSpin,
    /// Per-thread alternating high-IPC / low-IPC workload. Each
    /// worker runs `hot_iters` of dependent integer multiplies
    /// (high IPC, ALU-bound) followed by `cold_iters` of random
    /// cache-line touches over a working-set region (low IPC,
    /// memory-bound), repeating the alternation
    /// `period_iters` times before checking
    /// [`stop_requested`](crate::workload). The phase split
    /// is deterministic per worker — no shared state — so two
    /// workers iterate at offset cadences only if they are
    /// scheduled differently.
    ///
    /// Drives task-level runtime variance between phases: any
    /// scheduler that estimates a task's "bursty" or
    /// "memory-stall" character from a windowed runtime sample
    /// (lavd's `lat_cri` per-task latency-criticality field on
    /// `task_ctx`) sees this task switch character every
    /// `hot_iters + cold_iters` boundary. Tests scheduler
    /// adaptation latency: how quickly does the scheduler
    /// re-classify the task as the phase changes?
    ///
    /// Field semantics:
    ///
    /// - `hot_iters`: number of multiply-chain steps per hot
    ///   phase. Chosen to span ~tens of microseconds on a
    ///   modern core; e.g. 100_000 ≈ 50µs at IPC 2.0 / 2 GHz.
    /// - `cold_iters`: number of random cache-line touches per
    ///   cold phase. The cold phase reads a 512KB region (LLC
    ///   pressure on most desktop hosts; spills to DRAM on
    ///   workloads with smaller LLCs) at random offsets.
    /// - `period_iters`: hot/cold pair count per outer
    ///   iteration. Higher values reduce the per-stop-check
    ///   overhead but increase shutdown latency.
    ///
    /// All three must be `> 0`; both the
    /// [`ipc_variance`](Self::ipc_variance) constructor and
    /// [`WorkloadHandle::spawn`] reject zeros with
    /// [`WorkTypeValidationError::ZeroIpcVarianceParam`].
    ///
    /// **Stop responsiveness.** The hot and cold inner loops do
    /// not poll [`stop`](crate::workload). The outer
    /// `period_iters` loop checks `stop_requested` between each
    /// hot/cold pair, so worst-case shutdown latency is one
    /// hot-phase + one cold-phase. Large `hot_iters` /
    /// `cold_iters` increase the shutdown-latency floor
    /// proportionally; pick values that keep a single phase
    /// under the test author's tolerance for stop lag.
    ///
    /// **`iterations` counter semantics.** Each completed outer
    /// loop bumps the per-worker `iterations` counter by ONE,
    /// regardless of how many `period_iters` the inner loop
    /// actually completed before `stop_requested` fired. The
    /// counter records ENTERED outer cycles, not completed
    /// inner periods; the per-multiply / per-touch progress
    /// flows through `work_units` instead. A worker that exits
    /// during the inner `period_iters` loop still bumps
    /// `iterations` by 1 for that outer cycle — the
    /// `iterations += 1` at the end of the dispatch arm is
    /// unconditional.
    ///
    /// `worker_group_size = None`. No shared memory; no
    /// per-iteration syscall.
    IpcVariance {
        /// Multiply-chain steps per hot phase. Must be `> 0`.
        /// Larger values increase shutdown latency
        /// proportionally — the inner hot loop does not poll
        /// `stop` between steps, so a worker mid-hot-phase
        /// finishes the phase before the outer loop sees the
        /// stop signal.
        hot_iters: u64,
        /// Random cache-line touches per cold phase. Must be
        /// `> 0`. Larger values increase shutdown latency
        /// proportionally — the inner cold loop does not poll
        /// `stop` between touches, so a worker mid-cold-phase
        /// finishes the phase before the outer loop sees the
        /// stop signal.
        cold_iters: u64,
        /// Hot+cold pair iterations per outer loop. Must be
        /// `> 0`. Higher values reduce per-stop-check overhead
        /// but increase shutdown latency.
        period_iters: u64,
    },
}

impl WorkType {
    /// PascalCase names for all built-in variants, matching the enum arm names.
    ///
    /// Generated by `strum::VariantNames` at compile time from the
    /// `WorkType` enum definition, so a new variant appears here
    /// automatically. Includes `"Sequence"` and `"Custom"` even though
    /// [`from_name`](Self::from_name) cannot construct them (sequences
    /// require explicit phases; custom requires a function pointer).
    pub const ALL_NAMES: &'static [&'static str] = <Self as strum::VariantNames>::VARIANTS;

    /// PascalCase name of this variant, matching [`ALL_NAMES`](Self::ALL_NAMES).
    /// For [`Custom`](Self::Custom), returns the user-provided `name`
    /// field instead.
    pub fn name(&self) -> &str {
        match self {
            WorkType::SpinWait => "SpinWait",
            WorkType::YieldHeavy => "YieldHeavy",
            WorkType::Mixed => "Mixed",
            WorkType::IoSyncWrite => "IoSyncWrite",
            WorkType::IoRandRead => "IoRandRead",
            WorkType::IoConvoy => "IoConvoy",
            WorkType::Bursty { .. } => "Bursty",
            WorkType::PipeIo { .. } => "PipeIo",
            WorkType::FutexPingPong { .. } => "FutexPingPong",
            WorkType::CachePressure { .. } => "CachePressure",
            WorkType::CacheYield { .. } => "CacheYield",
            WorkType::CachePipe { .. } => "CachePipe",
            WorkType::FutexFanOut { .. } => "FutexFanOut",
            WorkType::Sequence { .. } => "Sequence",
            WorkType::ForkExit => "ForkExit",
            WorkType::NiceSweep => "NiceSweep",
            WorkType::AffinityChurn { .. } => "AffinityChurn",
            WorkType::PolicyChurn { .. } => "PolicyChurn",
            WorkType::FanOutCompute { .. } => "FanOutCompute",
            WorkType::PageFaultChurn { .. } => "PageFaultChurn",
            WorkType::MutexContention { .. } => "MutexContention",
            WorkType::Custom { name, .. } => name.as_str(),
            WorkType::ThunderingHerd { .. } => "ThunderingHerd",
            WorkType::PriorityInversion { .. } => "PriorityInversion",
            WorkType::ProducerConsumerImbalance { .. } => "ProducerConsumerImbalance",
            WorkType::RtStarvation { .. } => "RtStarvation",
            WorkType::AsymmetricWaker { .. } => "AsymmetricWaker",
            WorkType::WakeChain { .. } => "WakeChain",
            WorkType::NumaWorkingSetSweep { .. } => "NumaWorkingSetSweep",
            WorkType::CgroupChurn { .. } => "CgroupChurn",
            WorkType::SignalStorm { .. } => "SignalStorm",
            WorkType::PreemptStorm { .. } => "PreemptStorm",
            WorkType::EpollStorm { .. } => "EpollStorm",
            WorkType::NumaMigrationChurn { .. } => "NumaMigrationChurn",
            WorkType::IdleChurn { .. } => "IdleChurn",
            WorkType::AluHot { .. } => "AluHot",
            WorkType::SmtSiblingSpin => "SmtSiblingSpin",
            WorkType::IpcVariance { .. } => "IpcVariance",
        }
    }

    /// Look up a variant by PascalCase name and return it with default
    /// parameters. Returns `None` for unknown names, `"Sequence"`
    /// (requires explicit phases), and `"Custom"` (requires a function
    /// pointer).
    pub fn from_name(s: &str) -> Option<WorkType> {
        match s {
            "SpinWait" => Some(WorkType::SpinWait),
            "YieldHeavy" => Some(WorkType::YieldHeavy),
            "Mixed" => Some(WorkType::Mixed),
            "IoSyncWrite" => Some(WorkType::IoSyncWrite),
            "IoRandRead" => Some(WorkType::IoRandRead),
            "IoConvoy" => Some(WorkType::IoConvoy),
            "Bursty" => Some(WorkType::Bursty {
                burst_duration: defaults::BURSTY_BURST_DURATION,
                sleep_duration: defaults::BURSTY_SLEEP_DURATION,
            }),
            "PipeIo" => Some(WorkType::PipeIo {
                burst_iters: defaults::PIPE_IO_BURST_ITERS,
            }),
            "FutexPingPong" => Some(WorkType::FutexPingPong {
                spin_iters: defaults::FUTEX_PING_PONG_SPIN_ITERS,
            }),
            "CachePressure" => Some(WorkType::CachePressure {
                size_kb: defaults::CACHE_PRESSURE_SIZE_KB,
                stride: defaults::CACHE_PRESSURE_STRIDE,
            }),
            "CacheYield" => Some(WorkType::CacheYield {
                size_kb: defaults::CACHE_YIELD_SIZE_KB,
                stride: defaults::CACHE_YIELD_STRIDE,
            }),
            "CachePipe" => Some(WorkType::CachePipe {
                size_kb: defaults::CACHE_PIPE_SIZE_KB,
                burst_iters: defaults::CACHE_PIPE_BURST_ITERS,
            }),
            "FutexFanOut" => Some(WorkType::FutexFanOut {
                fan_out: defaults::FUTEX_FAN_OUT_FAN_OUT,
                spin_iters: defaults::FUTEX_FAN_OUT_SPIN_ITERS,
            }),
            "ForkExit" => Some(WorkType::ForkExit),
            "NiceSweep" => Some(WorkType::NiceSweep),
            "AffinityChurn" => Some(WorkType::AffinityChurn {
                spin_iters: defaults::AFFINITY_CHURN_SPIN_ITERS,
            }),
            "PolicyChurn" => Some(WorkType::PolicyChurn {
                spin_iters: defaults::POLICY_CHURN_SPIN_ITERS,
            }),
            "FanOutCompute" => Some(WorkType::FanOutCompute {
                fan_out: defaults::FAN_OUT_COMPUTE_FAN_OUT,
                cache_footprint_kb: defaults::FAN_OUT_COMPUTE_CACHE_FOOTPRINT_KB,
                operations: defaults::FAN_OUT_COMPUTE_OPERATIONS,
                sleep_usec: defaults::FAN_OUT_COMPUTE_SLEEP_USEC,
            }),
            "PageFaultChurn" => Some(WorkType::PageFaultChurn {
                region_kb: defaults::PAGE_FAULT_CHURN_REGION_KB,
                touches_per_cycle: defaults::PAGE_FAULT_CHURN_TOUCHES_PER_CYCLE,
                spin_iters: defaults::PAGE_FAULT_CHURN_SPIN_ITERS,
            }),
            "MutexContention" => Some(WorkType::MutexContention {
                contenders: defaults::MUTEX_CONTENTION_CONTENDERS,
                hold_iters: defaults::MUTEX_CONTENTION_HOLD_ITERS,
                work_iters: defaults::MUTEX_CONTENTION_WORK_ITERS,
            }),
            "ThunderingHerd" => Some(WorkType::ThunderingHerd {
                waiters: defaults::THUNDERING_HERD_WAITERS,
                batches: defaults::THUNDERING_HERD_BATCHES,
                inter_batch_ms: defaults::THUNDERING_HERD_INTER_BATCH_MS,
            }),
            "PriorityInversion" => Some(WorkType::PriorityInversion {
                high_count: defaults::PRIORITY_INVERSION_HIGH_COUNT,
                medium_count: defaults::PRIORITY_INVERSION_MEDIUM_COUNT,
                low_count: defaults::PRIORITY_INVERSION_LOW_COUNT,
                hold_iters: defaults::PRIORITY_INVERSION_HOLD_ITERS,
                work_iters: defaults::PRIORITY_INVERSION_WORK_ITERS,
                pi_mode: defaults::PRIORITY_INVERSION_PI_MODE,
            }),
            "ProducerConsumerImbalance" => Some(WorkType::ProducerConsumerImbalance {
                producers: defaults::PRODUCER_CONSUMER_PRODUCERS,
                consumers: defaults::PRODUCER_CONSUMER_CONSUMERS,
                produce_rate_hz: defaults::PRODUCER_CONSUMER_PRODUCE_RATE_HZ,
                consume_iters: defaults::PRODUCER_CONSUMER_CONSUME_ITERS,
                queue_depth_target: defaults::PRODUCER_CONSUMER_QUEUE_DEPTH_TARGET,
            }),
            "RtStarvation" => Some(WorkType::RtStarvation {
                rt_workers: defaults::RT_STARVATION_RT_WORKERS,
                cfs_workers: defaults::RT_STARVATION_CFS_WORKERS,
                rt_priority: defaults::RT_STARVATION_RT_PRIORITY,
                burst_iters: defaults::RT_STARVATION_BURST_ITERS,
            }),
            "AsymmetricWaker" => Some(WorkType::AsymmetricWaker {
                waker_class: SchedClass::default(),
                wakee_class: SchedClass::default(),
                burst_iters: defaults::ASYMMETRIC_WAKER_BURST_ITERS,
            }),
            "WakeChain" => Some(WorkType::WakeChain {
                depth: defaults::WAKE_CHAIN_DEPTH,
                wake: defaults::WAKE_CHAIN_WAKE,
                work_per_hop: defaults::WAKE_CHAIN_WORK_PER_HOP,
            }),
            "NumaWorkingSetSweep" => Some(WorkType::NumaWorkingSetSweep {
                region_kb: defaults::NUMA_WORKING_SET_SWEEP_REGION_KB,
                sweep_period_ms: defaults::NUMA_WORKING_SET_SWEEP_SWEEP_PERIOD_MS,
                // Empty list — single-node default leaves binding
                // disabled, matching `node_set()` defaults from
                // `MemPolicy::Default`. Users opt-in with a node
                // list via the constructor.
                target_nodes: Vec::new(),
            }),
            "CgroupChurn" => Some(WorkType::CgroupChurn {
                groups: defaults::CGROUP_CHURN_GROUPS,
                cycle_ms: defaults::CGROUP_CHURN_CYCLE_MS,
            }),
            "SignalStorm" => Some(WorkType::SignalStorm {
                signals_per_iter: defaults::SIGNAL_STORM_SIGNALS_PER_ITER,
                work_iters: defaults::SIGNAL_STORM_WORK_ITERS,
            }),
            "PreemptStorm" => Some(WorkType::PreemptStorm {
                cfs_workers: defaults::PREEMPT_STORM_CFS_WORKERS,
                rt_burst_iters: defaults::PREEMPT_STORM_RT_BURST_ITERS,
                rt_sleep_us: defaults::PREEMPT_STORM_RT_SLEEP_US,
            }),
            "EpollStorm" => Some(WorkType::EpollStorm {
                producers: defaults::EPOLL_STORM_PRODUCERS,
                consumers: defaults::EPOLL_STORM_CONSUMERS,
                events_per_burst: defaults::EPOLL_STORM_EVENTS_PER_BURST,
            }),
            "NumaMigrationChurn" => Some(WorkType::NumaMigrationChurn {
                period_ms: defaults::NUMA_MIGRATION_CHURN_PERIOD_MS,
            }),
            "IdleChurn" => Some(WorkType::IdleChurn {
                burst_duration: defaults::IDLE_CHURN_BURST_DURATION,
                sleep_duration: defaults::IDLE_CHURN_SLEEP_DURATION,
                precise_timing: defaults::IDLE_CHURN_PRECISE_TIMING,
            }),
            "AluHot" => Some(WorkType::AluHot {
                width: defaults::ALU_HOT_WIDTH,
            }),
            "SmtSiblingSpin" => Some(WorkType::SmtSiblingSpin),
            "IpcVariance" => Some(WorkType::IpcVariance {
                hot_iters: defaults::IPC_VARIANCE_HOT_ITERS,
                cold_iters: defaults::IPC_VARIANCE_COLD_ITERS,
                period_iters: defaults::IPC_VARIANCE_PERIOD_ITERS,
            }),
            // Sequence requires explicit phases; no default from_name.
            _ => None,
        }
    }

    /// Case-insensitive lookup that returns the canonical PascalCase
    /// entry from [`ALL_NAMES`](Self::ALL_NAMES) matching the input,
    /// or `None` when no entry matches.
    ///
    /// Distinct from [`from_name`](Self::from_name) in two ways:
    ///
    /// 1. It matches case-insensitively, so `"spinwait"` / `"SPINWAIT"`
    ///    / `"SpinWait"` all map to the same canonical `"SpinWait"`.
    /// 2. It returns the name string rather than a default-parameter
    ///    [`WorkType`] value, so callers can quote the canonical
    ///    spelling in error messages without also instantiating the
    ///    variant.
    ///
    /// Intended as a CLI / config-parser helper: when `from_name`
    /// returns `None` for the user's input, pass the same string
    /// here to recover the canonical spelling (if any) for a
    /// friendlier "did you mean `SpinWait`?" diagnostic. Includes
    /// `"Sequence"` and `"Custom"` in the match space even though
    /// `from_name` refuses to construct them — the point of
    /// [`suggest`](Self::suggest) is naming, not construction.
    ///
    /// Whitespace handling: the match uses `eq_ignore_ascii_case`
    /// without trimming, so surrounding whitespace in `s`
    /// (`" SpinWait"`, `"SpinWait\n"`) suppresses a match. Callers
    /// that accept user input with possible surrounding whitespace
    /// must `s.trim()` before calling — the same convention
    /// [`from_name`] follows. Keeping the predicate strict here
    /// avoids confusing "suggested canonical spelling" reports for
    /// inputs that were already nearly correct save for stray
    /// whitespace the caller should have already normalized.
    pub fn suggest(s: &str) -> Option<&'static str> {
        Self::ALL_NAMES
            .iter()
            .copied()
            .find(|n| n.eq_ignore_ascii_case(s))
    }

    /// Worker group size for this work type, or None if ungrouped.
    ///
    /// `num_workers` must be divisible by this value. Paired types return 2,
    /// fan-out returns fan_out + 1 (1 messenger + N receivers), and
    /// MutexContention returns `contenders`.
    pub fn worker_group_size(&self) -> Option<usize> {
        match self {
            WorkType::PipeIo { .. }
            | WorkType::FutexPingPong { .. }
            | WorkType::CachePipe { .. } => Some(2),
            WorkType::FutexFanOut { fan_out, .. } => Some(fan_out + 1),
            WorkType::FanOutCompute { fan_out, .. } => Some(fan_out + 1),
            WorkType::MutexContention { contenders, .. } => Some(*contenders),
            // ThunderingHerd uses a single global futex shared by
            // every worker — group size must equal num_workers so
            // the per-group futex allocator yields exactly one
            // shared region for the whole herd.
            WorkType::ThunderingHerd { waiters, .. } => Some(waiters + 1),
            // PriorityInversion: all 3 tiers share the same futex
            // word, so the group covers every worker.
            WorkType::PriorityInversion {
                high_count,
                medium_count,
                low_count,
                ..
            } => Some(high_count + medium_count + low_count),
            // ProducerConsumerImbalance: producers + consumers
            // share the queue mmap.
            WorkType::ProducerConsumerImbalance {
                producers,
                consumers,
                ..
            } => Some(producers + consumers),
            // RtStarvation: rt + cfs workers share the same
            // affinity-constrained scheduling domain.
            WorkType::RtStarvation {
                rt_workers,
                cfs_workers,
                ..
            } => Some(rt_workers + cfs_workers),
            // AsymmetricWaker: paired waker + wakee (group of 2),
            // matching FutexPingPong's shape.
            WorkType::AsymmetricWaker { .. } => Some(2),
            // WakeChain: each chain has `depth` workers. Group
            // size is the per-chain size; num_workers must be a
            // positive multiple of depth, and the spawn-side
            // derives the parallel-chain count from
            // `num_workers / depth`.
            WorkType::WakeChain { depth, .. } => Some(*depth),
            // SignalStorm: paired (waker / wakee), num_workers
            // must be even. Each pair shares the partner-tid
            // exchange region.
            WorkType::SignalStorm { .. } => Some(2),
            // PreemptStorm: 1 RT worker + cfs_workers CFS spinners
            // share the same affinity-constrained scheduling
            // domain so the RT preempts on the same CPU.
            WorkType::PreemptStorm { cfs_workers, .. } => Some(cfs_workers + 1),
            // EpollStorm: producers + consumers share the eventfd
            // / epoll fd handoff. One group per (producers,
            // consumers) tuple.
            WorkType::EpollStorm {
                producers,
                consumers,
                ..
            } => Some(producers + consumers),
            // SmtSiblingSpin: paired workers intended to be
            // pinned to the two SMT siblings of one physical
            // core. The variant doesn't allocate shared memory;
            // the group of 2 is what binds the AffinityIntent
            // resolution to a sibling pair. Pair via
            // [`AffinityIntent::SmtSiblingPair`] (auto-resolved
            // from host topology) or [`AffinityIntent::Exact`]
            // (caller-supplied CPU IDs).
            WorkType::SmtSiblingSpin => Some(2),
            _ => None,
        }
    }

    /// Whether this work type needs a pre-fork shared memory region (MAP_SHARED mmap).
    ///
    /// `RtStarvation` opts in even though its body never reads or
    /// writes the futex word: the spawn-side `(futex_ptr, pos)`
    /// tuple is the only mechanism that hands the worker its
    /// per-position index, which `RtStarvation` consumes to
    /// classify itself as RT or CFS. Allocating a single 4-byte
    /// MAP_SHARED region per group is the cheapest way to get
    /// `pos` plumbed through worker_main without a wider dispatch
    /// contract change.
    pub fn needs_shared_mem(&self) -> bool {
        matches!(
            self,
            WorkType::FutexPingPong { .. }
                | WorkType::FutexFanOut { .. }
                | WorkType::FanOutCompute { .. }
                | WorkType::MutexContention { .. }
                | WorkType::ThunderingHerd { .. }
                | WorkType::PriorityInversion { .. }
                | WorkType::ProducerConsumerImbalance { .. }
                | WorkType::AsymmetricWaker { .. }
                | WorkType::WakeChain { .. }
                | WorkType::RtStarvation { .. }
                | WorkType::SignalStorm { .. }
                | WorkType::PreemptStorm { .. }
                | WorkType::EpollStorm { .. }
        )
    }

    /// Number of pipes per chain that the spawn-side must allocate
    /// for this work type, or `None` when no per-stage pipe ring is
    /// needed. The returned `depth` matches the variant's `depth`
    /// field for `WakeChain { wake: WakeMechanism::Pipe, .. }`;
    /// every other variant (and `WakeChain` with
    /// `wake: WakeMechanism::Futex`) returns `None`.
    ///
    /// When this returns `Some(depth)`, the spawn-side allocates
    /// `depth` pipes per chain so stage `i` holds
    /// `pipe[i].write_end` (to wake stage `i + 1`) and
    /// `pipe[(i + depth - 1) % depth].read_end` (predecessor's
    /// wake). `WakeMechanism::Futex` keeps the existing futex-word
    /// ring and returns `None`.
    pub fn chain_pipe_depth(&self) -> Option<usize> {
        match self {
            WorkType::WakeChain {
                wake: WakeMechanism::Pipe,
                depth,
                ..
            } => Some(*depth),
            _ => None,
        }
    }

    /// Whether this work type allocates a per-worker cache buffer post-fork.
    pub fn needs_cache_buf(&self) -> bool {
        matches!(
            self,
            WorkType::CachePressure { .. }
                | WorkType::CacheYield { .. }
                | WorkType::CachePipe { .. }
                | WorkType::FanOutCompute { .. }
        )
    }

    /// Bursty work: CPU burst for `burst_duration`, sleep for
    /// `sleep_duration`, repeat.
    ///
    /// Validation fires at spawn time, not construction time; see
    /// [`WorkType::Bursty`] variant doc for preconditions.
    pub fn bursty(burst_duration: Duration, sleep_duration: Duration) -> Self {
        WorkType::Bursty {
            burst_duration,
            sleep_duration,
        }
    }

    /// Paired pipe I/O with CPU burst between exchanges.
    ///
    /// Validation fires at spawn time, not construction time; see
    /// [`WorkType::PipeIo`] variant doc for preconditions.
    pub fn pipe_io(burst_iters: u64) -> Self {
        WorkType::PipeIo { burst_iters }
    }

    /// Paired futex ping-pong with CPU spin between wakes.
    ///
    /// Validation fires at spawn time, not construction time; see
    /// [`WorkType::FutexPingPong`] variant doc for preconditions.
    pub fn futex_ping_pong(spin_iters: u64) -> Self {
        WorkType::FutexPingPong { spin_iters }
    }

    /// Strided read-modify-write over a `size_kb` KB buffer.
    ///
    /// Validation fires at spawn time, not construction time; see
    /// [`WorkType::CachePressure`] variant doc for preconditions.
    pub fn cache_pressure(size_kb: usize, stride: usize) -> Self {
        WorkType::CachePressure { size_kb, stride }
    }

    /// Cache pressure burst followed by sched_yield().
    ///
    /// Validation fires at spawn time, not construction time; see
    /// [`WorkType::CacheYield`] variant doc for preconditions.
    pub fn cache_yield(size_kb: usize, stride: usize) -> Self {
        WorkType::CacheYield { size_kb, stride }
    }

    /// Cache pressure burst then pipe exchange with a partner worker.
    ///
    /// Validation fires at spawn time, not construction time; see
    /// [`WorkType::CachePipe`] variant doc for preconditions.
    pub fn cache_pipe(size_kb: usize, burst_iters: u64) -> Self {
        WorkType::CachePipe {
            size_kb,
            burst_iters,
        }
    }

    /// 1:N fan-out wake pattern with CPU spin between wakes.
    ///
    /// Validation fires at spawn time, not construction time; see
    /// [`WorkType::FutexFanOut`] variant doc for preconditions.
    pub fn futex_fan_out(fan_out: usize, spin_iters: u64) -> Self {
        WorkType::FutexFanOut {
            fan_out,
            spin_iters,
        }
    }

    /// Rapid self-directed affinity changes with `spin_iters` CPU work between.
    ///
    /// Validation fires at spawn time, not construction time; see
    /// [`WorkType::AffinityChurn`] variant doc for preconditions.
    pub fn affinity_churn(spin_iters: u64) -> Self {
        WorkType::AffinityChurn { spin_iters }
    }

    /// Cycle scheduling policies with `spin_iters` CPU work between switches.
    ///
    /// Validation fires at spawn time, not construction time; see
    /// [`WorkType::PolicyChurn`] variant doc for preconditions.
    pub fn policy_churn(spin_iters: u64) -> Self {
        WorkType::PolicyChurn { spin_iters }
    }

    /// Messenger/worker fan-out with compute work using the given parameters.
    ///
    /// `fan_out` is passed to `futex_wake(ptr, N)` where `N: i32` is
    /// the number of waiters to wake. Realistic values are tens of
    /// workers; sched-test topologies that need more than `i32::MAX`
    /// (~2.1B) receivers per messenger are not expressible.
    /// [`WorkloadHandle::spawn`] clamps the cast to `i32::MAX` so a
    /// pathological `usize` input wakes all-available instead of
    /// wrapping to a negative (FUTEX_WAKE broadcasts when passed a
    /// negative N on some kernels, which would wake every waiter on
    /// the futex rather than just this messenger's receivers).
    ///
    /// Validation fires at spawn time, not construction time; see
    /// [`WorkType::FanOutCompute`] variant doc for preconditions.
    pub fn fan_out_compute(
        fan_out: usize,
        cache_footprint_kb: usize,
        operations: usize,
        sleep_usec: u64,
    ) -> Self {
        WorkType::FanOutCompute {
            fan_out,
            cache_footprint_kb,
            operations,
            sleep_usec,
        }
    }

    /// Rapid page fault cycling with `spin_iters` CPU work between cycles.
    ///
    /// Validation fires at spawn time, not construction time; see
    /// [`WorkType::PageFaultChurn`] variant doc for preconditions.
    pub fn page_fault_churn(region_kb: usize, touches_per_cycle: usize, spin_iters: u64) -> Self {
        WorkType::PageFaultChurn {
            region_kb,
            touches_per_cycle,
            spin_iters,
        }
    }

    /// N-way futex mutex contention with `contenders` workers per group.
    ///
    /// Validation fires at spawn time, not construction time; see
    /// [`WorkType::MutexContention`] variant doc for preconditions.
    pub fn mutex_contention(contenders: usize, hold_iters: u64, work_iters: u64) -> Self {
        WorkType::MutexContention {
            contenders,
            hold_iters,
            work_iters,
        }
    }

    /// One waker, N waiters on a single global futex; broadcasts via
    /// `FUTEX_WAKE` per batch. Pairs with
    /// [`WorkType::ThunderingHerd`].
    ///
    /// Validation fires at spawn time, not construction time; see
    /// [`WorkType::ThunderingHerd`] variant doc for preconditions.
    pub fn thundering_herd(waiters: usize, batches: u64, inter_batch_ms: u64) -> Self {
        WorkType::ThunderingHerd {
            waiters,
            batches,
            inter_batch_ms,
        }
    }

    /// Three priority tiers contending for one shared lock. See
    /// [`WorkType::PriorityInversion`] for behavior; pass
    /// [`FutexLockMode::Pi`] to invoke `FUTEX_LOCK_PI` or
    /// [`FutexLockMode::Plain`] for a non-PI futex.
    ///
    /// Validation fires at spawn time, not construction time; see
    /// [`WorkType::PriorityInversion`] variant doc for preconditions.
    pub fn priority_inversion(
        high_count: usize,
        medium_count: usize,
        low_count: usize,
        hold_iters: u64,
        work_iters: u64,
        pi_mode: FutexLockMode,
    ) -> Self {
        WorkType::PriorityInversion {
            high_count,
            medium_count,
            low_count,
            hold_iters,
            work_iters,
            pi_mode,
        }
    }

    /// Producer/consumer pipeline with deliberately unbalanced
    /// rates. See [`WorkType::ProducerConsumerImbalance`].
    ///
    /// Validation fires at spawn time, not construction time; see
    /// [`WorkType::ProducerConsumerImbalance`] variant doc for preconditions.
    pub fn producer_consumer_imbalance(
        producers: usize,
        consumers: usize,
        produce_rate_hz: u64,
        consume_iters: u64,
        queue_depth_target: u64,
    ) -> Self {
        WorkType::ProducerConsumerImbalance {
            producers,
            consumers,
            produce_rate_hz,
            consume_iters,
            queue_depth_target,
        }
    }

    /// `rt_workers` SCHED_FIFO workers vs. `cfs_workers` SCHED_NORMAL
    /// workers competing on the same CPU set. See
    /// [`WorkType::RtStarvation`].
    ///
    /// Validation fires at spawn time, not construction time; see
    /// [`WorkType::RtStarvation`] variant doc for preconditions.
    pub fn rt_starvation(
        rt_workers: usize,
        cfs_workers: usize,
        rt_priority: i32,
        burst_iters: u64,
    ) -> Self {
        WorkType::RtStarvation {
            rt_workers,
            cfs_workers,
            rt_priority,
            burst_iters,
        }
    }

    /// Paired workers in mismatched scheduling classes. See
    /// [`WorkType::AsymmetricWaker`].
    ///
    /// Validation fires at spawn time, not construction time; see
    /// [`WorkType::AsymmetricWaker`] variant doc for preconditions.
    pub fn asymmetric_waker(
        waker_class: SchedClass,
        wakee_class: SchedClass,
        burst_iters: u64,
    ) -> Self {
        WorkType::AsymmetricWaker {
            waker_class,
            wakee_class,
            burst_iters,
        }
    }

    /// Pipeline of waker-wakee hops with optional `WF_SYNC`. See
    /// [`WorkType::WakeChain`].
    ///
    /// Validation fires at spawn time, not construction time; see
    /// [`WorkType::WakeChain`] variant doc for preconditions
    /// (`depth >= 2`, `num_workers` divisible by `depth`, etc.).
    pub fn wake_chain(
        depth: usize,
        wake: WakeMechanism,
        work_per_hop: Duration,
    ) -> Self {
        WorkType::WakeChain {
            depth,
            wake,
            work_per_hop,
        }
    }

    /// NUMA working-set sweep with periodic `mbind` rotation. See
    /// [`WorkType::NumaWorkingSetSweep`]. `target_nodes` accepts
    /// any `IntoIterator<Item = usize>` for ergonomic call sites
    /// (`[0, 1, 2]`, `0..node_count`, `BTreeSet`, etc.).
    ///
    /// Validation fires at spawn time, not construction time; see
    /// [`WorkType::NumaWorkingSetSweep`] variant doc for preconditions.
    pub fn numa_working_set_sweep(
        region_kb: usize,
        sweep_period_ms: u64,
        target_nodes: impl IntoIterator<Item = usize>,
    ) -> Self {
        WorkType::NumaWorkingSetSweep {
            region_kb,
            sweep_period_ms,
            target_nodes: target_nodes.into_iter().collect(),
        }
    }

    /// Construct a [`WorkType::Sequence`] from a head phase and an
    /// iterator of follow-on phases.
    ///
    /// The `Sequence` variant cannot use [`from_name`](Self::from_name)
    /// because phases require explicit construction; this constructor
    /// is the only typed entry point. Accepts any `IntoIterator<Item =
    /// Phase>` for `rest` so callers can pass arrays, `Vec`, or
    /// builder-style chains.
    ///
    /// Validation fires at spawn time, not construction time; see
    /// [`WorkType::Sequence`] variant doc for preconditions.
    pub fn sequence(first: Phase, rest: impl IntoIterator<Item = Phase>) -> Self {
        WorkType::Sequence {
            first,
            rest: rest.into_iter().collect(),
        }
    }

    /// Construct a [`WorkType::CgroupChurn`].
    ///
    /// Validation fires at spawn time, not construction time; see
    /// [`WorkType::CgroupChurn`] variant doc for preconditions.
    pub fn cgroup_churn(groups: usize, cycle_ms: u64) -> Self {
        WorkType::CgroupChurn { groups, cycle_ms }
    }

    /// Construct a [`WorkType::SignalStorm`].
    ///
    /// Validation fires at spawn time, not construction time; see
    /// [`WorkType::SignalStorm`] variant doc for preconditions.
    pub fn signal_storm(signals_per_iter: u64, work_iters: u64) -> Self {
        WorkType::SignalStorm {
            signals_per_iter,
            work_iters,
        }
    }

    /// Construct a [`WorkType::PreemptStorm`].
    ///
    /// Validation fires at spawn time, not construction time; see
    /// [`WorkType::PreemptStorm`] variant doc for preconditions.
    pub fn preempt_storm(cfs_workers: usize, rt_burst_iters: u64, rt_sleep_us: u64) -> Self {
        WorkType::PreemptStorm {
            cfs_workers,
            rt_burst_iters,
            rt_sleep_us,
        }
    }

    /// Construct a [`WorkType::EpollStorm`].
    ///
    /// Validation fires at spawn time, not construction time; see
    /// [`WorkType::EpollStorm`] variant doc for preconditions.
    pub fn epoll_storm(producers: usize, consumers: usize, events_per_burst: u64) -> Self {
        WorkType::EpollStorm {
            producers,
            consumers,
            events_per_burst,
        }
    }

    /// Construct a [`WorkType::NumaMigrationChurn`].
    ///
    /// Validation fires at spawn time, not construction time; see
    /// [`WorkType::NumaMigrationChurn`] variant doc for preconditions.
    pub fn numa_migration_churn(period_ms: u64) -> Self {
        WorkType::NumaMigrationChurn { period_ms }
    }

    /// Construct a [`WorkType::IdleChurn`] with the default
    /// `precise_timing = false`.
    ///
    /// # Spawn-time precondition
    ///
    /// `burst_duration` and `sleep_duration` must both be
    /// strictly greater than `Duration::ZERO`. The constructor
    /// itself accepts any value (no early validation); the
    /// rejection fires at [`WorkloadHandle::spawn`] time with an
    /// actionable bail message naming the offending field. See
    /// [`WorkType::IdleChurn`] variant doc for the rationale and
    /// the kernel-source citation.
    ///
    /// # `precise_timing`
    ///
    /// This constructor sets `precise_timing` to
    /// [`defaults::IDLE_CHURN_PRECISE_TIMING`] (`false`),
    /// preserving the inherited `current->timer_slack_ns`
    /// (~50µs default). To opt into 1ns timer slack, build the
    /// variant directly via the struct-literal form:
    /// `WorkType::IdleChurn { burst_duration, sleep_duration,
    /// precise_timing: true }`. See the variant's
    /// `precise_timing` field doc for the kernel-side
    /// mechanism.
    pub fn idle_churn(burst_duration: Duration, sleep_duration: Duration) -> Self {
        WorkType::IdleChurn {
            burst_duration,
            sleep_duration,
            precise_timing: defaults::IDLE_CHURN_PRECISE_TIMING,
        }
    }

    /// Construct a [`WorkType::AluHot`] at the given execution
    /// width.
    ///
    /// `AluWidth::Widest` resolves to the widest data-path the
    /// host supports at worker entry. See [`AluWidth`] for the
    /// per-variant data-path width and the runtime resolution
    /// rules.
    ///
    /// Validation fires at spawn time, not construction time;
    /// see [`WorkType::AluHot`] variant doc for preconditions.
    pub fn alu_hot(width: AluWidth) -> Self {
        WorkType::AluHot { width }
    }

    /// Construct a [`WorkType::IpcVariance`] with explicit hot,
    /// cold, and period iteration counts.
    ///
    /// Returns [`WorkTypeValidationError::ZeroIpcVarianceParam`]
    /// when any of `hot_iters`, `cold_iters`, or `period_iters`
    /// is `0`. Construction-time validation matches the
    /// spawn-time check so callers get immediate feedback at
    /// the call site rather than discovering the rejection
    /// only at [`WorkloadHandle::spawn`] time.
    pub fn ipc_variance(
        hot_iters: u64,
        cold_iters: u64,
        period_iters: u64,
    ) -> std::result::Result<Self, WorkTypeValidationError> {
        if hot_iters == 0 {
            return Err(WorkTypeValidationError::ZeroIpcVarianceParam {
                field: "hot_iters",
                group_idx: 0,
            });
        }
        if cold_iters == 0 {
            return Err(WorkTypeValidationError::ZeroIpcVarianceParam {
                field: "cold_iters",
                group_idx: 0,
            });
        }
        if period_iters == 0 {
            return Err(WorkTypeValidationError::ZeroIpcVarianceParam {
                field: "period_iters",
                group_idx: 0,
            });
        }
        Ok(WorkType::IpcVariance {
            hot_iters,
            cold_iters,
            period_iters,
        })
    }

    /// User-supplied work function with a display name.
    ///
    /// `run` receives a reference to the stop flag (flipped per-mode:
    /// the SIGUSR1 handler for [`CloneMode::Fork`], a per-worker
    /// `AtomicBool` for [`CloneMode::Thread`]) and must return a
    /// [`WorkerReport`] when the flag becomes `true`. The framework
    /// handles fork / thread spawn, cgroup placement, affinity,
    /// scheduling policy, and signal setup (Fork mode only); `run`
    /// owns only the work loop.
    ///
    /// The per-iteration built-in instrumentation (wake-latency samples,
    /// `iter_slot` publish, gap tracking) runs only for built-in variants
    /// and is bypassed for `Custom`. See the [`Custom`](Self::Custom)
    /// variant doc for the full telemetry contract and what `run` must
    /// populate on [`WorkerReport`] to keep downstream assertions honest.
    pub fn custom(name: impl Into<String>, run: fn(&AtomicBool) -> WorkerReport) -> Self {
        WorkType::Custom {
            name: name.into(),
            run,
        }
    }
}

/// Spawn-time validation failures for [`WorkType`] preconditions.
///
/// Returned (boxed inside [`anyhow::Error`]) by
/// [`WorkloadHandle::spawn`] when a per-group [`WorkSpec`] violates a
/// runtime invariant the variant doc declares as a precondition.
/// Tests that need to assert on a specific variant downcast via
/// `err.downcast_ref::<WorkTypeValidationError>()`; the
/// `Display` impl carries the same human-readable text the previous
/// `anyhow::bail!` strings did so call sites that match on the
/// rendered message keep working.
///
/// Each variant carries `group_idx` (the position of the offending
/// [`WorkSpec`] inside [`WorkloadConfig::composed`]; the primary
/// group is index 0) so multi-group scenarios can locate the
/// offending entry without re-parsing the message string. Variants
/// with multiple constraint inputs (depth, divisor, observed count)
/// expose those values as named fields to the same end.
#[derive(Debug, thiserror::Error)]
pub enum WorkTypeValidationError {
    /// [`WorkType::IdleChurn`] with `burst_duration == Duration::ZERO`.
    /// Collapses the per-iteration loop to pure nanosleep so the
    /// worker accrues no runtime — useless as a scheduler test. See
    /// the variant doc's "Spawn-time validation" section for the
    /// full rationale.
    #[error(
        "IdleChurn burst_duration must be > 0 (group {group_idx}); a zero \
         burst makes the loop pure sleep and the worker accrues \
         no runtime (see [`WorkType::IdleChurn`] variant doc)"
    )]
    ZeroBurstDuration {
        /// Index of the offending group in
        /// [`WorkloadConfig::composed`] (primary group = 0).
        group_idx: usize,
    },
    /// [`WorkType::IdleChurn`] with `sleep_duration == Duration::ZERO`.
    /// Collapses the per-iteration loop to a CPU-bound burst with
    /// no idle path; the kernel's `nanosleep(0)` is yield-like
    /// rather than idle-like. The diagnostic steers the caller to
    /// [`WorkType::SpinWait`] (pure CPU spin) or
    /// [`WorkType::YieldHeavy`] (the closer overlap).
    #[error(
        "IdleChurn sleep_duration must be > 0 (group {group_idx}); a zero \
         sleep collapses the loop to a CPU-bound burst. \
         Use WorkType::SpinWait for pure CPU spin, or \
         WorkType::YieldHeavy for the closer overlap \
         (nanosleep(0) is yield-like — see the variant \
         doc rationale in [`WorkType::IdleChurn`])."
    )]
    ZeroSleepDuration {
        /// Index of the offending group in
        /// [`WorkloadConfig::composed`] (primary group = 0).
        group_idx: usize,
    },
    /// [`WorkType::WakeChain`] with `depth < 2`. A 1-stage chain has
    /// no successor to wake, and the post-fork close-other-fds
    /// block would close the worker's own write end (deadlock).
    #[error(
        "WakeChain depth must be >= 2 (got {depth}, group {group_idx}); a 1-stage \
         chain has no successor to wake and the post-fork fd close \
         logic would close the worker's own write end \
         (see [`WorkType::WakeChain`] variant doc)"
    )]
    InsufficientWakeChainDepth {
        /// The offending `depth` value the caller supplied.
        depth: usize,
        /// Index of the offending group in
        /// [`WorkloadConfig::composed`] (primary group = 0).
        group_idx: usize,
    },
    /// `num_workers` is not a positive multiple of the variant's
    /// [`worker_group_size`](WorkType::worker_group_size). Affects
    /// every grouped variant (paired, fan-out, herd, contention,
    /// chain). The diagnostic names the variant via [`WorkType::name`].
    #[error(
        "{name} (group {group_idx}) requires num_workers divisible by {group_size}, got {num_workers}"
    )]
    NonDivisibleWorkerCount {
        /// PascalCase variant name from [`WorkType::name`].
        name: String,
        /// Index of the offending group in
        /// [`WorkloadConfig::composed`] (primary group = 0).
        group_idx: usize,
        /// Required group size (the variant's
        /// [`worker_group_size`](WorkType::worker_group_size)).
        group_size: usize,
        /// The `num_workers` count the caller supplied.
        num_workers: usize,
    },
    /// [`WorkType::IpcVariance`] with one of `hot_iters`,
    /// `cold_iters`, or `period_iters` equal to `0`. A zero in
    /// any of the three collapses the alternation: zero
    /// `hot_iters` produces a pure cold-phase memory loop, zero
    /// `cold_iters` produces a pure ALU loop (use
    /// [`WorkType::AluHot`] directly for that), and zero
    /// `period_iters` produces a worker that never advances
    /// past the first stop check. Each rejection names the
    /// offending field so the caller knows which to fix.
    #[error(
        "IpcVariance {field} must be > 0 (group {group_idx}); a zero value \
         collapses the hot/cold alternation and produces a degenerate \
         workload (see [`WorkType::IpcVariance`] variant doc)"
    )]
    ZeroIpcVarianceParam {
        /// Static name of the offending field —
        /// `"hot_iters"`, `"cold_iters"`, or `"period_iters"`.
        field: &'static str,
        /// Index of the offending group in
        /// [`WorkloadConfig::composed`] (primary group = 0).
        group_idx: usize,
    },
}
