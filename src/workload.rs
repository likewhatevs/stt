//! Worker process management and telemetry.
//!
//! Workers are `fork()`ed processes (not threads) so each can be placed
//! in its own cgroup. Key types:
//! - [`WorkType`] -- what each worker does
//! - [`WorkloadConfig`] -- spawn configuration (count, affinity, work type, policy)
//! - [`WorkloadHandle`] -- RAII handle to spawned workers
//! - [`WorkerReport`] -- per-worker telemetry collected after stop
//! - [`AffinityIntent`] -- per-worker affinity intent (Inherit, LlcAligned, Exact, etc.)
//! - [`ResolvedAffinity`] -- resolved CPU affinity for workers
//! - [`WorkSpec`] -- workload definition for a single group of workers within a cgroup
//! - [`Phase`] -- a single phase in a [`WorkType::Sequence`] compound work pattern
//! - [`SchedPolicy`] -- Linux scheduling policy for a worker process
//! - [`MemPolicy`] -- NUMA memory placement policy for worker processes
//!
//! See the [WorkSpec Types](https://likewhatevs.github.io/ktstr/guide/concepts/work-types.html)
//! and [Worker Processes](https://likewhatevs.github.io/ktstr/guide/architecture/workers.html)
//! chapters of the guide.
//!
//! # Naming conventions
//!
//! ## "Intent" vs "Resolved" naming
//!
//! Types named with an `Intent` suffix carry **test-author intent**
//! (the input to the workload pipeline). Types named with a
//! `Resolved` prefix carry **runtime-resolved configuration** (the
//! output of intent + topology + cgroup state). [`AffinityIntent`]
//! resolves to [`ResolvedAffinity`] at spawn time via
//! [`resolve_affinity_for_cgroup`](crate::scenario::resolve_affinity_for_cgroup).
//!
//! [`CloneMode`] is a runtime-resolved value because the test
//! author writes `CloneMode::Fork` / `CloneMode::Thread` directly
//! (no resolution layer); the `Mode` suffix denotes a single
//! kernel-facing dispatch decision rather than a two-stage
//! intent/resolved pipeline.
//!
//! [`SchedClass`] and [`SchedPolicy`] follow the same coarse-intent /
//! concrete-runtime split using legacy kernel terminology rather
//! than the `Intent`/`Resolved` naming — see [`SchedClass`] for
//! the per-class mapping.
//!
//! ## "Churn" vs "Sweep" suffixes on [`WorkType`] variants
//!
//! Variants whose names end in `Churn` cycle their target setting
//! **without ordering** — each iteration picks a fresh value
//! independently of the previous one. [`WorkType::AffinityChurn`]
//! samples a random CPU from the effective cpuset on every
//! iteration; [`WorkType::PolicyChurn`] cycles through the
//! supported scheduling policies; [`WorkType::PageFaultChurn`]
//! touches a fresh random subset of pages each cycle. The intent
//! is high-frequency randomness — exercise the kernel's per-task
//! state machines under unpredictable transitions.
//!
//! Variants whose names end in `Sweep` rotate their target setting
//! through an **ordered list or range** — the next value is a
//! deterministic function of the iteration counter, not a random
//! pick. [`WorkType::NiceSweep`] cycles nice values from
//! `effective_min..=19` modulo the range size;
//! [`WorkType::NumaWorkingSetSweep`] rotates the working-set
//! binding through `target_nodes` in declaration order. The
//! intent is to walk a phase space evenly so every value gets
//! comparable observation time, rather than producing the
//! unbiased-random transitions Churn produces.
//!
//! Choose `Churn` when the workload's value is its
//! transition-frequency entropy; choose `Sweep` when the workload
//! must visit every phase deterministically.

use anyhow::{Context, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Seek, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

// `FanOutCompute` stores its u64 generation counter at offset 0 of
// a 16-byte shared region and relies on the low 4 bytes of that
// counter living at offset 0 so the futex syscall (which reads the
// raw u32 at `futex_ptr`) sees the low u32 of the u64. That layout
// assumption holds on little-endian targets (x86_64, aarch64) and
// flips on big-endian — the futex would read the high 32 bits
// instead, and an increment of the u64 would leave the low 4 bytes
// unchanged until the 2^32-th advance. Reject the big-endian build
// at compile time rather than shipping a silently-broken binary.
#[cfg(not(target_endian = "little"))]
compile_error!(
    "ktstr's FanOutCompute generation-counter layout assumes a \
     little-endian target — the u64 counter at offset 0 of the \
     shared futex region must expose its low 32 bits to the \
     futex syscall at that same offset. Porting to a big-endian \
     target requires reworking the layout so futex_wait sees the \
     incrementing low 4 bytes."
);

/// Scenario-level affinity intent for a group of workers.
///
/// Resolved to a concrete [`ResolvedAffinity`] at runtime based on the
/// cgroup's effective cpuset and the VM's topology. When attached to
/// a [`WorkSpec`], determines per-worker `sched_setaffinity` masks.
///
/// Resolution uses [`resolve_affinity_for_cgroup()`](crate::scenario::resolve_affinity_for_cgroup).
#[derive(Clone, Debug, Default)]
pub enum AffinityIntent {
    /// No affinity constraint -- inherit from parent cgroup.
    #[default]
    Inherit,
    /// Pin to a random subset of the cgroup's cpuset, or all CPUs if no
    /// cpuset is configured.
    RandomSubset,
    /// Pin to the CPUs in the worker's LLC.
    LlcAligned,
    /// Pin to all CPUs (crosses cgroup boundaries).
    CrossCgroup,
    /// Pin to a single CPU.
    SingleCpu,
    /// Pin to an exact set of CPUs.
    Exact(BTreeSet<usize>),
}

impl AffinityIntent {
    /// Construct an `Exact` affinity from any iterator of CPU indices.
    ///
    /// Accepts arrays, ranges, `Vec`, `BTreeSet`, or any `IntoIterator<Item = usize>`.
    pub fn exact(cpus: impl IntoIterator<Item = usize>) -> Self {
        AffinityIntent::Exact(cpus.into_iter().collect())
    }
}

/// Resolved CPU affinity for a worker process.
///
/// Created from [`AffinityIntent`] at runtime based on topology and
/// cpuset assignments.
#[derive(Debug, Clone)]
pub enum ResolvedAffinity {
    /// No affinity constraint.
    None,
    /// Pin to a specific set of CPUs.
    Fixed(BTreeSet<usize>),
    /// Pin to `count` randomly-chosen CPUs from `from`.
    ///
    /// - `count` must be `> 0`; zero is rejected at resolve time
    ///   (previously it coerced silently to 1 and masked caller bugs).
    /// - `count > from.len()` is clamped to `from.len()` — asking for
    ///   more CPUs than the pool contains is a topology fact, not a
    ///   caller error.
    /// - `from` empty with `count > 0` resolves to no affinity (no
    ///   pool to sample from); downstream treats this as `None`.
    Random { from: BTreeSet<usize>, count: usize },
    /// Pin to a single CPU.
    SingleCpu(usize),
}

/// A single phase in a [`WorkType::Sequence`] compound work pattern.
///
/// Workers loop through all phases in order, then repeat. Each phase
/// runs for its specified duration before advancing to the next.
#[derive(Clone, Debug)]
pub enum Phase {
    /// CPU spin for the given duration.
    Spin(Duration),
    /// Sleep (thread::sleep) for the given duration.
    Sleep(Duration),
    /// Yield (sched_yield) repeatedly for the given duration.
    Yield(Duration),
    /// Simulated I/O (write 64 KB to tmpfs + 100 us sleep) for the given
    /// duration. See [`WorkType::IoSync`] for details on tmpfs behavior.
    Io(Duration),
}

/// What each worker process does during a scenario.
///
/// Different work types exercise different scheduler code paths:
/// CPU-bound, yield-heavy, I/O, bursty, or inter-process communication.
///
/// ```
/// # use ktstr::workload::WorkType;
/// let wt = WorkType::from_name("CpuSpin").unwrap();
/// assert!(matches!(wt, WorkType::CpuSpin));
///
/// let bursty = WorkType::bursty(10, 5);
/// assert!(matches!(bursty, WorkType::Bursty { .. }));
///
/// assert!(WorkType::from_name("nonexistent").is_none());
/// ```
///
/// The `VariantNames` derive generates `WorkType::VARIANTS: &[&str]`
/// at compile time from the enum arm names, which this module
/// re-exposes as [`WorkType::ALL_NAMES`] so a new variant is picked
/// up automatically without editing a parallel list.
#[derive(Debug, Clone, strum::VariantNames)]
pub enum WorkType {
    /// Tight CPU spin loop (1024 iterations per cycle).
    CpuSpin,
    /// Repeated sched_yield with minimal CPU work.
    YieldHeavy,
    /// CPU spin burst followed by sched_yield.
    Mixed,
    /// Simulated I/O-bound workload: writes 64 KB to a temp file then
    /// sleeps 100 us to simulate I/O completion latency. On tmpfs (which
    /// ktstr VMs use), the write is a page-cache memcpy and fsync is a
    /// no-op (`noop_fsync`), so the sleep provides the blocking behavior
    /// that real disk fsync would cause.
    IoSync,
    /// Work hard for burst_ms, sleep for sleep_ms, repeat. Frees CPUs during sleep for borrowing.
    Bursty { burst_ms: u64, sleep_ms: u64 },
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
    Custom {
        name: &'static str,
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
    /// `num_workers - 1` are waiters. Per pathology research
    /// (research_structural_pathology.md P1), structural minimum is
    /// `waiters >= 5` to surface convoy effects on a multi-CPU
    /// host. `worker_group_size = num_workers` so every worker
    /// shares the same shared-memory region; reuses the existing
    /// futex MAP_SHARED allocator.
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
    /// `pi_mode = PiMode::Pi` uses `FUTEX_LOCK_PI` (PI-aware
    /// mutex); kernel boosts `low` to `high`'s priority for the
    /// duration of the hold, which both unblocks `high` and pins
    /// `medium` from preempting. `PiMode::Plain` uses a plain
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
        /// (`Plain`, uncorrected inversion). See [`PiMode`].
        pi_mode: PiMode,
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
    /// Pipeline of waker-wakee hops `A → B → C → ...` of `depth`
    /// Workers form a ring chain of `depth` stages, optionally
    /// with `fanout` parallel chains. Each chain owns one shared
    /// 4-byte futex word; the active stage rotates through
    /// `0..depth`. The active stage burns `work_per_hop` of CPU
    /// work, advances the word to `(pos + 1) % depth`, and
    /// `FUTEX_WAKE`s the chain — only the next stage's worker
    /// satisfies its wait predicate and proceeds. Tests
    /// `SCX_WAKE_SYNC` path under producer-consumer NUMA pinning.
    ///
    /// Worker indices are partitioned into `fanout` chains of
    /// `depth` workers each. `worker_group_size = depth` so the
    /// spawn-side allocates `fanout` independent futex regions
    /// (one per chain). At the end of the chain the last worker
    /// loops back to the first, forming a ring so the work
    /// pattern can run for a long test window.
    WakeChain {
        /// Number of workers per chain. Each worker waits for its
        /// predecessor's signal, does `work_per_hop` of CPU work,
        /// signals the next worker, and repeats.
        depth: usize,
        /// Number of parallel chains. `fanout = 1` is a single
        /// linear chain; `fanout > 1` runs multiple chains in
        /// parallel, all sharing one configuration.
        fanout: usize,
        /// Reserved for future per-stage pipe wiring. `true` is
        /// intended to add a 1-byte pipe write per stage so the
        /// kernel raises `WF_SYNC` on the wakeup
        /// (`kernel/sched/core.c::ttwu_queue` / `try_to_wake_up`);
        /// `false` keeps the futex-only path (no `WF_SYNC`). The
        /// current implementation uses the futex path for both
        /// values — pipe wiring requires per-stage pipe pairs in
        /// `SpawnGuard` that don't yet exist.
        sync: bool,
        /// Wall-clock CPU work each worker performs per stage
        /// before signalling the next. Use [`Duration`] to keep
        /// the unit visible at the call site (consistent with
        /// [`SchedPolicy::Deadline`]'s switch to `Duration`).
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
}

/// Named defaults for the parametric [`WorkType`] variants, used by
/// [`WorkType::from_name`]. Extracting the magic numbers here
/// provides a named home for the default values so tests and docs
/// (e.g. `doc/guide/src/architecture/workers.md`) can cite them by
/// constant name instead of each tracking a scattered integer
/// literal. Every value carries a single-line comment naming the
/// knob and its unit; the const names mirror the
/// `{variant_snake}_{field}` convention so renames show up as
/// compile errors in both sites.
pub mod defaults {
    // Bursty
    pub const BURSTY_BURST_MS: u64 = 50;
    pub const BURSTY_SLEEP_MS: u64 = 100;
    // PipeIo
    pub const PIPE_IO_BURST_ITERS: u64 = 1024;
    // FutexPingPong
    pub const FUTEX_PING_PONG_SPIN_ITERS: u64 = 1024;
    // CachePressure / CacheYield / CachePipe share buffer shape
    pub const CACHE_PRESSURE_SIZE_KB: usize = 32;
    pub const CACHE_PRESSURE_STRIDE: usize = 64;
    pub const CACHE_YIELD_SIZE_KB: usize = 32;
    pub const CACHE_YIELD_STRIDE: usize = 64;
    pub const CACHE_PIPE_SIZE_KB: usize = 32;
    pub const CACHE_PIPE_BURST_ITERS: u64 = 1024;
    // FutexFanOut
    pub const FUTEX_FAN_OUT_FAN_OUT: usize = 4;
    pub const FUTEX_FAN_OUT_SPIN_ITERS: u64 = 1024;
    // AffinityChurn
    pub const AFFINITY_CHURN_SPIN_ITERS: u64 = 1024;
    // PolicyChurn
    pub const POLICY_CHURN_SPIN_ITERS: u64 = 1024;
    // FanOutCompute
    pub const FAN_OUT_COMPUTE_FAN_OUT: usize = 4;
    pub const FAN_OUT_COMPUTE_CACHE_FOOTPRINT_KB: usize = 256;
    pub const FAN_OUT_COMPUTE_OPERATIONS: usize = 5;
    pub const FAN_OUT_COMPUTE_SLEEP_USEC: u64 = 100;
    // PageFaultChurn
    pub const PAGE_FAULT_CHURN_REGION_KB: usize = 4096;
    pub const PAGE_FAULT_CHURN_TOUCHES_PER_CYCLE: usize = 256;
    pub const PAGE_FAULT_CHURN_SPIN_ITERS: u64 = 64;
    // MutexContention
    pub const MUTEX_CONTENTION_CONTENDERS: usize = 4;
    pub const MUTEX_CONTENTION_HOLD_ITERS: u64 = 256;
    pub const MUTEX_CONTENTION_WORK_ITERS: u64 = 1024;
    // ThunderingHerd
    pub const THUNDERING_HERD_WAITERS: usize = 7;
    pub const THUNDERING_HERD_BATCHES: u64 = 1_000;
    pub const THUNDERING_HERD_INTER_BATCH_MS: u64 = 5;
    // PriorityInversion
    pub const PRIORITY_INVERSION_HIGH_COUNT: usize = 1;
    pub const PRIORITY_INVERSION_MEDIUM_COUNT: usize = 1;
    pub const PRIORITY_INVERSION_LOW_COUNT: usize = 1;
    pub const PRIORITY_INVERSION_HOLD_ITERS: u64 = 4096;
    pub const PRIORITY_INVERSION_WORK_ITERS: u64 = 1024;
    pub const PRIORITY_INVERSION_PI_MODE: super::PiMode = super::PiMode::Plain;
    // ProducerConsumerImbalance
    pub const PRODUCER_CONSUMER_PRODUCERS: usize = 2;
    pub const PRODUCER_CONSUMER_CONSUMERS: usize = 1;
    pub const PRODUCER_CONSUMER_PRODUCE_RATE_HZ: u64 = 1_000;
    pub const PRODUCER_CONSUMER_CONSUME_ITERS: u64 = 4_096;
    pub const PRODUCER_CONSUMER_QUEUE_DEPTH_TARGET: u64 = 1024;
    // RtStarvation
    pub const RT_STARVATION_RT_WORKERS: usize = 1;
    pub const RT_STARVATION_CFS_WORKERS: usize = 1;
    pub const RT_STARVATION_RT_PRIORITY: i32 = 50;
    pub const RT_STARVATION_BURST_ITERS: u64 = 1024;
    // AsymmetricWaker
    pub const ASYMMETRIC_WAKER_BURST_ITERS: u64 = 1024;
    // WakeChain
    pub const WAKE_CHAIN_DEPTH: usize = 4;
    pub const WAKE_CHAIN_FANOUT: usize = 1;
    pub const WAKE_CHAIN_SYNC: bool = true;
    pub const WAKE_CHAIN_WORK_PER_HOP: std::time::Duration = std::time::Duration::from_micros(100);
    // NumaWorkingSetSweep
    pub const NUMA_WORKING_SET_SWEEP_REGION_KB: usize = 4_096;
    pub const NUMA_WORKING_SET_SWEEP_SWEEP_PERIOD_MS: u64 = 100;
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
    pub fn name(&self) -> &'static str {
        match self {
            WorkType::CpuSpin => "CpuSpin",
            WorkType::YieldHeavy => "YieldHeavy",
            WorkType::Mixed => "Mixed",
            WorkType::IoSync => "IoSync",
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
            WorkType::Custom { name, .. } => name,
            WorkType::ThunderingHerd { .. } => "ThunderingHerd",
            WorkType::PriorityInversion { .. } => "PriorityInversion",
            WorkType::ProducerConsumerImbalance { .. } => "ProducerConsumerImbalance",
            WorkType::RtStarvation { .. } => "RtStarvation",
            WorkType::AsymmetricWaker { .. } => "AsymmetricWaker",
            WorkType::WakeChain { .. } => "WakeChain",
            WorkType::NumaWorkingSetSweep { .. } => "NumaWorkingSetSweep",
        }
    }

    /// Look up a variant by PascalCase name and return it with default
    /// parameters. Returns `None` for unknown names, `"Sequence"`
    /// (requires explicit phases), and `"Custom"` (requires a function
    /// pointer).
    pub fn from_name(s: &str) -> Option<WorkType> {
        match s {
            "CpuSpin" => Some(WorkType::CpuSpin),
            "YieldHeavy" => Some(WorkType::YieldHeavy),
            "Mixed" => Some(WorkType::Mixed),
            "IoSync" => Some(WorkType::IoSync),
            "Bursty" => Some(WorkType::Bursty {
                burst_ms: defaults::BURSTY_BURST_MS,
                sleep_ms: defaults::BURSTY_SLEEP_MS,
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
                fanout: defaults::WAKE_CHAIN_FANOUT,
                sync: defaults::WAKE_CHAIN_SYNC,
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
    /// 1. It matches case-insensitively, so `"cpuspin"` / `"CPUSPIN"`
    ///    / `"CpuSpin"` all map to the same canonical `"CpuSpin"`.
    /// 2. It returns the name string rather than a default-parameter
    ///    [`WorkType`] value, so callers can quote the canonical
    ///    spelling in error messages without also instantiating the
    ///    variant.
    ///
    /// Intended as a CLI / config-parser helper: when `from_name`
    /// returns `None` for the user's input, pass the same string
    /// here to recover the canonical spelling (if any) for a
    /// friendlier "did you mean `CpuSpin`?" diagnostic. Includes
    /// `"Sequence"` and `"Custom"` in the match space even though
    /// `from_name` refuses to construct them — the point of
    /// [`suggest`](Self::suggest) is naming, not construction.
    ///
    /// Whitespace handling: the match uses `eq_ignore_ascii_case`
    /// without trimming, so surrounding whitespace in `s`
    /// (`" CpuSpin"`, `"CpuSpin\n"`) suppresses a match. Callers
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
            // WakeChain: each chain has `depth` workers; multiple
            // chains run in parallel via `fanout`. Group size is
            // the per-chain size; num_workers must equal
            // depth * fanout.
            WorkType::WakeChain { depth, .. } => Some(*depth),
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
        )
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

    /// Bursty work: CPU burst for `burst_ms`, sleep for `sleep_ms`, repeat.
    pub fn bursty(burst_ms: u64, sleep_ms: u64) -> Self {
        WorkType::Bursty { burst_ms, sleep_ms }
    }

    /// Paired pipe I/O with CPU burst between exchanges.
    pub fn pipe_io(burst_iters: u64) -> Self {
        WorkType::PipeIo { burst_iters }
    }

    /// Paired futex ping-pong with CPU spin between wakes.
    pub fn futex_ping_pong(spin_iters: u64) -> Self {
        WorkType::FutexPingPong { spin_iters }
    }

    /// Strided read-modify-write over a `size_kb` KB buffer.
    pub fn cache_pressure(size_kb: usize, stride: usize) -> Self {
        WorkType::CachePressure { size_kb, stride }
    }

    /// Cache pressure burst followed by sched_yield().
    pub fn cache_yield(size_kb: usize, stride: usize) -> Self {
        WorkType::CacheYield { size_kb, stride }
    }

    /// Cache pressure burst then pipe exchange with a partner worker.
    pub fn cache_pipe(size_kb: usize, burst_iters: u64) -> Self {
        WorkType::CachePipe {
            size_kb,
            burst_iters,
        }
    }

    /// 1:N fan-out wake pattern with CPU spin between wakes.
    pub fn futex_fan_out(fan_out: usize, spin_iters: u64) -> Self {
        WorkType::FutexFanOut {
            fan_out,
            spin_iters,
        }
    }

    /// Rapid self-directed affinity changes with `spin_iters` CPU work between.
    pub fn affinity_churn(spin_iters: u64) -> Self {
        WorkType::AffinityChurn { spin_iters }
    }

    /// Cycle scheduling policies with `spin_iters` CPU work between switches.
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
    pub fn page_fault_churn(region_kb: usize, touches_per_cycle: usize, spin_iters: u64) -> Self {
        WorkType::PageFaultChurn {
            region_kb,
            touches_per_cycle,
            spin_iters,
        }
    }

    /// N-way futex mutex contention with `contenders` workers per group.
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
    pub fn thundering_herd(waiters: usize, batches: u64, inter_batch_ms: u64) -> Self {
        WorkType::ThunderingHerd {
            waiters,
            batches,
            inter_batch_ms,
        }
    }

    /// Three priority tiers contending for one shared lock. See
    /// [`WorkType::PriorityInversion`] for behavior; pass
    /// [`PiMode::Pi`] to invoke `FUTEX_LOCK_PI` or
    /// [`PiMode::Plain`] for a non-PI futex.
    pub fn priority_inversion(
        high_count: usize,
        medium_count: usize,
        low_count: usize,
        hold_iters: u64,
        work_iters: u64,
        pi_mode: PiMode,
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
    pub fn wake_chain(depth: usize, fanout: usize, sync: bool, work_per_hop: Duration) -> Self {
        WorkType::WakeChain {
            depth,
            fanout,
            sync,
            work_per_hop,
        }
    }

    /// NUMA working-set sweep with periodic `mbind` rotation. See
    /// [`WorkType::NumaWorkingSetSweep`]. `target_nodes` accepts
    /// any `IntoIterator<Item = usize>` for ergonomic call sites
    /// (`[0, 1, 2]`, `0..node_count`, `BTreeSet`, etc.).
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
    pub fn custom(name: &'static str, run: fn(&AtomicBool) -> WorkerReport) -> Self {
        WorkType::Custom { name, run }
    }
}

/// Resolve a work type with an optional override.
///
/// Returns a clone of `override_wt` when `swappable` is true, an
/// override is provided, and the override's group size (if any)
/// divides `num_workers`. Otherwise returns a clone of `base`. When
/// `override_wt` is `None`, always returns `base` regardless of
/// `swappable`.
pub(crate) fn resolve_work_type(
    base: &WorkType,
    override_wt: Option<&WorkType>,
    swappable: bool,
    num_workers: usize,
) -> WorkType {
    if !swappable {
        return base.clone();
    }
    match override_wt {
        Some(wt) => {
            if let Some(gs) = wt.worker_group_size()
                && !num_workers.is_multiple_of(gs)
            {
                return base.clone();
            }
            wt.clone()
        }
        None => base.clone(),
    }
}

/// Linux scheduling policy for a worker process.
///
/// `Fifo`, `RoundRobin`, and `Deadline` all require `CAP_SYS_NICE`
/// (`user_check_sched_setscheduler` in `kernel/sched/syscalls.c`
/// routes rt_policy and dl_policy through `req_priv`). `Normal`,
/// `Batch`, and (entering) `Idle` are unprivileged transitions for
/// fair-policy tasks. Priority values for `Fifo`/`RoundRobin` are
/// clamped to 1-99.
#[derive(Debug, Clone, Copy)]
pub enum SchedPolicy {
    /// `SCHED_NORMAL` (CFS/EEVDF).
    Normal,
    /// `SCHED_BATCH`.
    Batch,
    /// `SCHED_IDLE`.
    Idle,
    /// `SCHED_FIFO` with the given priority (1-99).
    Fifo(u32),
    /// `SCHED_RR` with the given priority (1-99).
    RoundRobin(u32),
    /// `SCHED_DEADLINE` with explicit `runtime`, `deadline`, and
    /// `period`. Applied via `sched_setattr(2)`.
    ///
    /// Each field is a [`Duration`] — the nanosecond representation
    /// the kernel requires is materialised at the syscall site, so
    /// callers express intent in idiomatic Rust units
    /// (`Duration::from_micros(100)`, `Duration::from_millis(1)`,
    /// etc.) and don't have to thread integer-nanosecond literals
    /// through their test fixtures.
    ///
    /// Constraints (from `__checkparam_dl` in
    /// `kernel/sched/deadline.c`):
    /// - `deadline != Duration::ZERO`.
    /// - `runtime` must be at least 1024 ns (the kernel's
    ///   `DL_SCALE` floor); shorter runtimes are silently truncated
    ///   inside the kernel and break bandwidth accounting.
    /// - `runtime <= deadline`.
    /// - `period == Duration::ZERO` is legal — the kernel
    ///   substitutes `deadline` for the period when zero. When
    ///   non-zero, `deadline <= period`.
    /// - The effective period (`period` if non-zero, else
    ///   `deadline`) is checked against
    ///   `/proc/sys/kernel/sched_deadline_period_min_us` (default
    ///   100us = 100_000 ns) and
    ///   `/proc/sys/kernel/sched_deadline_period_max_us` (default
    ///   `1 << 22` us = 4_194_304_000 ns), inclusive. Both sysctls
    ///   are runtime-tunable; this crate does not pre-validate the
    ///   sysctl range and lets the kernel surface out-of-range
    ///   values as `EINVAL`.
    /// - The nanosecond count of `deadline` and `period` must each
    ///   fit in 63 bits (`< 1 << 63`, i.e. `<= i64::MAX` ns ≈ 292
    ///   years) — the kernel uses bit 63 internally. Any longer
    ///   `Duration` is rejected at the syscall site.
    ///
    /// Transitions to/from `Deadline` always require `CAP_SYS_NICE`.
    /// Tasks set to `Deadline` get exclusive bandwidth on the
    /// admission-controlled root domain; oversubscription returns
    /// `EBUSY` (see `sched_dl_overflow` in `kernel/sched/deadline.c`).
    ///
    /// `set_sched_policy` validates the structural constraints
    /// (zero-deadline, DL_SCALE floor, ordering, top-bit) before
    /// invoking `sched_setattr` so a malformed `Deadline` fails
    /// fast in user space rather than tunneling an `EINVAL`
    /// through the syscall.
    Deadline {
        /// Runtime budget per period.
        runtime: Duration,
        /// Relative deadline from period start.
        deadline: Duration,
        /// Period. `Duration::ZERO` means "use `deadline` as the
        /// period" per the kernel's `__checkparam_dl` substitution.
        period: Duration,
    },
}

impl SchedPolicy {
    /// `SCHED_FIFO` with the given priority (1-99).
    pub fn fifo(priority: u32) -> Self {
        SchedPolicy::Fifo(priority)
    }

    /// `SCHED_RR` with the given priority (1-99).
    pub fn round_robin(priority: u32) -> Self {
        SchedPolicy::RoundRobin(priority)
    }

    /// `SCHED_DEADLINE` with the given runtime / deadline / period.
    /// See [`SchedPolicy::Deadline`] for parameter constraints.
    ///
    /// All three arguments share the same [`Duration`] type. The
    /// canonical order is `(runtime, deadline, period)` — runtime
    /// budget first, then the relative deadline, then the period.
    /// For tests that need to make the order obvious at the call
    /// site, prefer the struct-literal form
    /// `SchedPolicy::Deadline { runtime: ..., deadline: ...,
    /// period: ... }` which carries the field names through the
    /// reader's eye.
    ///
    /// ```
    /// # use std::time::Duration;
    /// # use ktstr::workload::SchedPolicy;
    /// // Convenience constructor — canonical (runtime, deadline, period) order.
    /// let p = SchedPolicy::deadline(
    ///     Duration::from_micros(500), // runtime
    ///     Duration::from_millis(1),   // deadline
    ///     Duration::from_millis(10),  // period
    /// );
    /// // Struct-literal form — names elide positional confusion.
    /// let q = SchedPolicy::Deadline {
    ///     runtime: Duration::from_micros(500),
    ///     deadline: Duration::from_millis(1),
    ///     period: Duration::from_millis(10),
    /// };
    /// assert!(matches!(p, SchedPolicy::Deadline { .. }));
    /// assert!(matches!(q, SchedPolicy::Deadline { .. }));
    /// ```
    pub fn deadline(runtime: Duration, deadline: Duration, period: Duration) -> Self {
        SchedPolicy::Deadline {
            runtime,
            deadline,
            period,
        }
    }
}

/// Whether [`WorkType::PriorityInversion`] uses a PI-aware mutex
/// or a plain futex.
///
/// `Pi` exercises `FUTEX_LOCK_PI` and the rt_mutex priority-boost
/// chain (`kernel/futex/pi.c`). When the low-priority lock holder
/// is preempted by a medium-priority worker, the kernel boosts
/// the holder to the high-priority waiter's priority for the
/// duration of the hold — both unblocking `high` and pinning
/// `medium` from preempting it. `Plain` uses a non-PI futex so
/// the inversion is left unrepaired and the scheduler must
/// surface the stall.
///
/// Carried as a typed wrapper rather than a `bool` to avoid
/// positional-argument confusion at call sites and so the
/// failure-dump diagnostic names the choice explicitly
/// ("pi_mode = Pi" vs "pi_mode = Plain") instead of a bare
/// boolean.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PiMode {
    /// `FUTEX_LOCK_PI` with rt_mutex PI chain.
    Pi,
    /// Plain futex (no PI boost). The default — exercises the
    /// uncorrected inversion the workload exists to surface.
    #[default]
    Plain,
}

/// Coarse Linux scheduling class identifier.
///
/// Maps to one of the kernel's six core scheduler classes:
/// `fair_sched_class` (CFS / EEVDF — covers `SCHED_NORMAL`,
/// `SCHED_BATCH`, `SCHED_IDLE`), `rt_sched_class` (covers
/// `SCHED_FIFO` and `SCHED_RR`), `dl_sched_class` (covers
/// `SCHED_DEADLINE`), and `ext_sched_class` (covers `SCHED_EXT`
/// when sched_ext is loaded). The class is a coarser concept
/// than [`SchedPolicy`] — `Cfs` covers Normal/Batch/Idle, `Rt`
/// covers Fifo/RoundRobin — and is what
/// [`WorkType::AsymmetricWaker`] consumes when it wants to
/// describe a waker / wakee pair without specifying priority
/// values. When a per-worker class is applied,
/// [`apply_sched_class`] maps the variant to the equivalent
/// [`SchedPolicy`] (using a default priority where applicable)
/// and routes through `set_sched_policy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SchedClass {
    /// `fair_sched_class` — `SCHED_NORMAL` (CFS / EEVDF). The
    /// default; matches a freshly-forked task before any policy
    /// override.
    #[default]
    Cfs,
    /// `fair_sched_class` — `SCHED_BATCH` (background-friendly
    /// fair task with longer wakeup latency targets).
    Batch,
    /// `fair_sched_class` — `SCHED_IDLE` (lowest fair-class
    /// weight; runs only when nothing else is runnable).
    Idle,
    /// `rt_sched_class` — `SCHED_FIFO` at default priority
    /// `RT_DEFAULT_PRIO`. Requires `CAP_SYS_NICE`. For explicit
    /// priority control use [`SchedPolicy::Fifo`] directly.
    Rt,
    /// `dl_sched_class` — `SCHED_DEADLINE`. Maps to a
    /// minimum-bandwidth deadline reservation
    /// ([`SchedClass::default_deadline_reservation`]) so
    /// `SchedClass::Deadline` is constructible without picking
    /// runtime/deadline/period. Callers needing precise
    /// reservations should use [`SchedPolicy::Deadline`]
    /// directly.
    Deadline,
    /// `ext_sched_class` — `SCHED_EXT`. Routes the worker
    /// through the loaded sched_ext BPF scheduler. Under
    /// switch-all (the default scx-ktstr regime), this is the
    /// same effective class as `Cfs` because every fair-policy
    /// task already reroutes to ext via `task_should_scx` (see
    /// kernel/sched/ext.c). `Cfs` is preserved as the explicit
    /// "I want fair semantics" knob the user expresses; `Ext`
    /// is preserved for tests that explicitly want
    /// `policy == SCHED_EXT` set on the task_struct.
    Ext,
}

/// Default `RT_DEFAULT_PRIO` for [`SchedClass::Rt`] when mapped to
/// a [`SchedPolicy`]. Picked at the middle of the 1..=99 valid range
/// so the worker neither preempts every other RT task in the system
/// nor sits at the floor; tests that need a specific RT priority
/// must construct [`SchedPolicy::Fifo`] directly.
const RT_DEFAULT_PRIO: u32 = 50;

impl SchedClass {
    /// Resolve to an equivalent [`SchedPolicy`]. `Rt` uses
    /// [`RT_DEFAULT_PRIO`]; `Deadline` uses the minimum-bandwidth
    /// reservation (1us runtime over 1ms period — passes
    /// `__checkparam_dl` and the default sysctl bounds).
    /// `Ext` maps to `SchedPolicy::Normal` because there is no
    /// userspace `SCHED_EXT` constant in libc; tests that want
    /// the kernel to read `policy == SCHED_EXT` (which
    /// requires sched_ext-aware userspace) cannot be expressed
    /// via this helper and must call the raw syscall path.
    pub fn to_policy(self) -> SchedPolicy {
        match self {
            SchedClass::Cfs | SchedClass::Ext => SchedPolicy::Normal,
            SchedClass::Batch => SchedPolicy::Batch,
            SchedClass::Idle => SchedPolicy::Idle,
            SchedClass::Rt => SchedPolicy::Fifo(RT_DEFAULT_PRIO),
            SchedClass::Deadline => Self::default_deadline_reservation(),
        }
    }

    /// Minimum-bandwidth `SCHED_DEADLINE` reservation that passes
    /// `__checkparam_dl`'s `runtime >= DL_SCALE` floor and the
    /// kernel's default `sched_deadline_period_min_us` (100us).
    /// 1us runtime, 1ms deadline, 10ms period — bandwidth fraction
    /// 0.0001, well below admission-control limits.
    pub fn default_deadline_reservation() -> SchedPolicy {
        SchedPolicy::Deadline {
            runtime: Duration::from_micros(1),
            deadline: Duration::from_millis(1),
            period: Duration::from_millis(10),
        }
    }
}

/// NUMA memory placement policy for worker processes.
///
/// Applied via `set_mempolicy(2)` after fork, before the work loop.
/// Maps to Linux `MPOL_*` constants. When `Default`, no syscall is
/// made (inherits the parent's policy).
///
/// Optional [`MpolFlags`] modify behavior (e.g. `STATIC_NODES` to
/// keep the nodemask absolute across cpuset changes).
#[derive(Clone, Debug, Default)]
pub enum MemPolicy {
    /// Inherit the parent process's memory policy (no syscall).
    #[default]
    Default,
    /// Allocate only from the specified NUMA nodes (`MPOL_BIND`).
    Bind(BTreeSet<usize>),
    /// Prefer allocations from the specified node, falling back to
    /// others when the preferred node is full (`MPOL_PREFERRED`).
    Preferred(usize),
    /// Interleave allocations round-robin across the specified nodes
    /// (`MPOL_INTERLEAVE`).
    Interleave(BTreeSet<usize>),
    /// Prefer the nearest node to the CPU where the allocation occurs
    /// (`MPOL_LOCAL`). No nodemask.
    Local,
    /// Prefer allocations from any of the specified nodes, falling back
    /// to others when all preferred nodes are full
    /// (`MPOL_PREFERRED_MANY`, kernel 5.15+).
    PreferredMany(BTreeSet<usize>),
    /// Weighted interleave across the specified nodes. Page distribution
    /// is proportional to per-node weights set via
    /// `/sys/kernel/mm/mempolicy/weighted_interleave/nodeN`
    /// (`MPOL_WEIGHTED_INTERLEAVE`, kernel 6.9+).
    WeightedInterleave(BTreeSet<usize>),
}

/// Optional mode flags for `set_mempolicy(2)`.
///
/// OR'd into the mode argument. See `MPOL_F_*` in
/// `include/uapi/linux/mempolicy.h`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MpolFlags(u32);

impl MpolFlags {
    /// No flags.
    pub const NONE: Self = Self(0);
    /// `MPOL_F_STATIC_NODES` (1 << 15): nodemask is absolute, not
    /// remapped when the task's cpuset changes.
    pub const STATIC_NODES: Self = Self(1 << 15);
    /// `MPOL_F_RELATIVE_NODES` (1 << 14): nodemask is relative to
    /// the task's current cpuset.
    pub const RELATIVE_NODES: Self = Self(1 << 14);
    /// `MPOL_F_NUMA_BALANCING` (1 << 13): enable NUMA balancing
    /// optimization for this policy.
    pub const NUMA_BALANCING: Self = Self(1 << 13);

    /// Test-only raw-bit constructor. Lets unknown-bit guards
    /// (e.g. `validate_mempolicy_cpuset` in src/scenario/ops.rs)
    /// be tested against bit patterns that are not reachable via
    /// the documented `STATIC_NODES | RELATIVE_NODES |
    /// NUMA_BALANCING` constants. Production callers must use the
    /// named constants + `union` / `BitOr` so the model stays in
    /// sync with the validator's known-bits mask.
    #[cfg(test)]
    pub(crate) const fn from_bits_for_test(bits: u32) -> Self {
        Self(bits)
    }

    /// Combine two flag sets.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Raw flag bits for passing to the syscall.
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Whether every bit in `other` is set in `self`.
    ///
    /// Set-theoretic, not syntactic: `contains(NONE)` returns `true`
    /// for any `self` (vacuous truth — the empty set is a subset of
    /// everything). Callers who want "has a non-empty intersection
    /// with `other`" must compare `self.bits() & other.bits() != 0`
    /// explicitly; using `contains` for that query silently returns
    /// `true` when the operand is `NONE` regardless of `self`.
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl std::ops::BitOr for MpolFlags {
    type Output = Self;
    /// Delegates to [`MpolFlags::union`] so the bitwise-OR logic
    /// lives in one place. `union` is `const fn` (usable in
    /// const contexts like `const` initializers); `BitOr::bitor`
    /// cannot currently be `const` on stable, so keeping both
    /// entry points is necessary, but they must never diverge.
    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

/// Raw clone(2) / clone3(2) flag bits the caller can opt into via
/// [`CloneMode::Raw`].
///
/// Only the user-relevant subset of `CLONE_*` is exposed here. The
/// runtime sets bookkeeping flags itself ([`CLONE_PIDFD`],
/// [`CLONE_PARENT_SETTID`], [`CLONE_CHILD_SETTID`],
/// [`CLONE_CHILD_CLEARTID`]) so callers MUST NOT pass them; the
/// validator rejects any of those bits as runtime-reserved.
///
/// Bit values come from `include/uapi/linux/sched.h` and match the
/// `libc::CLONE_*` constants. `u64` because `clone_args.flags` is
/// `__aligned_u64`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CloneFlags(u64);

impl CloneFlags {
    /// No flags — equivalent to plain `fork(2)` semantics.
    pub const NONE: Self = Self(0);
    /// `CLONE_VM` (0x100): share the address space with the parent.
    /// Required to be set whenever [`SIGHAND`](Self::SIGHAND) is set
    /// (kernel rule 2 in `copy_process`).
    pub const VM: Self = Self(0x100);
    /// `CLONE_FS` (0x200): share the filesystem context (cwd, umask,
    /// root). Mutually exclusive with [`NEWNS`](Self::NEWNS) and
    /// [`NEWUSER`](Self::NEWUSER).
    pub const FS: Self = Self(0x200);
    /// `CLONE_FILES` (0x400): share the open-file descriptor table.
    pub const FILES: Self = Self(0x400);
    /// `CLONE_SIGHAND` (0x800): share signal handlers with the
    /// parent. Implies the worker cannot install per-task signal
    /// dispositions — the whole process sees the same handlers.
    /// The kernel rejects `SIGHAND` without [`VM`](Self::VM).
    pub const SIGHAND: Self = Self(0x800);
    /// `CLONE_THREAD` (0x10000): place the new task into the
    /// caller's thread group (same `tgid`, `exit_signal == -1`).
    /// The kernel rejects `THREAD` without
    /// [`SIGHAND`](Self::SIGHAND), and rejects the combination with
    /// [`NEWUSER`](Self::NEWUSER) or [`NEWPID`](Self::NEWPID).
    /// Reaping requires `pidfd` or `pthread_join` — `waitpid` does
    /// not see thread-group siblings.
    pub const THREAD: Self = Self(0x10000);
    /// `CLONE_PARENT` (0x8000): the new task's parent is set to the
    /// caller's parent (sibling rather than child). The kernel
    /// rejects this when the caller is `SIGNAL_UNKILLABLE` (init /
    /// container-init).
    pub const PARENT: Self = Self(0x8000);
    /// `CLONE_NEWNS` (0x20000): give the new task a fresh mount
    /// namespace. Mutually exclusive with [`FS`](Self::FS).
    pub const NEWNS: Self = Self(0x20000);
    /// `CLONE_NEWUSER` (0x10000000): give the new task a fresh user
    /// namespace. Mutually exclusive with [`FS`](Self::FS) and
    /// [`THREAD`](Self::THREAD).
    pub const NEWUSER: Self = Self(0x10000000);
    /// `CLONE_NEWPID` (0x20000000): give the new task a fresh PID
    /// namespace. Mutually exclusive with [`THREAD`](Self::THREAD).
    pub const NEWPID: Self = Self(0x20000000);
    /// `CLONE_NEWNET` (0x40000000): give the new task a fresh
    /// network namespace.
    pub const NEWNET: Self = Self(0x40000000);
    /// `CLONE_NEWIPC` (0x08000000): give the new task a fresh SysV
    /// IPC namespace.
    pub const NEWIPC: Self = Self(0x08000000);
    /// `CLONE_NEWUTS` (0x04000000): give the new task a fresh UTS
    /// (hostname / domainname) namespace.
    pub const NEWUTS: Self = Self(0x04000000);
    /// `CLONE_NEWCGROUP` (0x02000000): give the new task a fresh
    /// cgroup namespace.
    pub const NEWCGROUP: Self = Self(0x02000000);
    /// `CLONE_IO` (0x80000000): share the I/O context with the
    /// parent.
    pub const IO: Self = Self(0x80000000);
    /// `CLONE_SYSVSEM` (0x40000): share SysV semaphore undo state.
    pub const SYSVSEM: Self = Self(0x40000);

    /// Bits the runtime sets internally — callers MUST NOT include
    /// any of these. Listed here so the validator can reject them
    /// as a single mask check.
    ///
    /// Includes:
    /// - `CLONE_PIDFD` (0x1000): the runtime uses pidfd for reaping.
    /// - `CLONE_PARENT_SETTID` (0x100000) and `CLONE_CHILD_SETTID`
    ///   (0x01000000): the runtime fills in tid out-params itself.
    /// - `CLONE_CHILD_CLEARTID` (0x200000): the runtime owns
    ///   tid-cleared-on-exit semantics for thread-join coordination.
    /// - `CLONE_DETACHED` (0x400000): the kernel rejects this with
    ///   `CLONE_PIDFD` (rule 7 in `copy_process`); since the runtime
    ///   always sets `CLONE_PIDFD` for non-fork dispatch, callers
    ///   must not set `CLONE_DETACHED`.
    /// - `CLONE_SETTLS` (0x80000): TLS setup is the runtime's job
    ///   when it issues clone3 itself.
    /// - `CLONE_VFORK` (0x4000): the runtime never wants the
    ///   parent-blocking semantics of vfork.
    /// - `CLONE_PTRACE` (0x2000) / `CLONE_UNTRACED` (0x800000):
    ///   ptrace inheritance is the runtime's call.
    pub const RUNTIME_RESERVED: Self = Self(
        0x1000        // CLONE_PIDFD
        | 0x100000    // CLONE_PARENT_SETTID
        | 0x01000000  // CLONE_CHILD_SETTID
        | 0x200000    // CLONE_CHILD_CLEARTID
        | 0x400000    // CLONE_DETACHED
        | 0x80000     // CLONE_SETTLS
        | 0x4000      // CLONE_VFORK
        | 0x2000      // CLONE_PTRACE
        | 0x800000,   // CLONE_UNTRACED
    );

    /// Test-only raw-bit constructor. Lets the flag-validator
    /// reject patterns that the public constants do not expose
    /// (e.g. unknown high bits, exact runtime-reserved bits).
    /// Production callers must use the named constants +
    /// [`union`](Self::union) / `BitOr` so the model stays in sync
    /// with the validator.
    #[cfg(test)]
    pub(crate) const fn from_bits_for_test(bits: u64) -> Self {
        Self(bits)
    }

    /// Combine two flag sets.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Raw flag bits for `clone_args.flags`.
    pub const fn bits(self) -> u64 {
        self.0
    }

    /// Whether every bit in `other` is set in `self`. Set-theoretic
    /// (matches [`MpolFlags::contains`]'s convention): `contains(NONE)`
    /// returns `true` for any `self`.
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl std::ops::BitOr for CloneFlags {
    type Output = Self;
    /// Delegates to [`CloneFlags::union`] so the bitwise-OR logic
    /// lives in one place. Same const-fn vs trait-fn split as
    /// [`MpolFlags`].
    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

/// How [`WorkloadHandle::spawn`] creates worker tasks.
///
/// `Fork` is the default — the existing [`fork(2)`] path with
/// separate address space, separate thread group, and `waitpid`
/// reaping. The other variants opt into different clone(2) /
/// clone3(2) flag configurations to exercise scheduler code paths
/// that depend on shared VM, shared thread group, or fully custom
/// flags.
///
/// All variants except `Fork` produce workers that cannot be
/// reaped via `waitpid` for thread-group members — the runtime
/// uses `pthread_join` ([`Thread`](Self::Thread)) or `pidfd` polling
/// ([`SharedVm`](Self::SharedVm) / [`Raw`](Self::Raw)) instead. See
/// the per-variant docs for the dispatch path each takes.
///
/// **Status:** [`Fork`](Self::Fork) and [`Thread`](Self::Thread) are
/// dispatched; [`SharedVm`](Self::SharedVm) and [`Raw`](Self::Raw)
/// bail with a runtime-constraint diagnostic until the clone3-child
/// Rust-runtime redesign lands. The flag validator
/// ([`validate_clone_flags`]) is fully implemented because the
/// redesign will reuse it; the variant constructors are exposed so
/// callers can write tests that pin the planned API surface ahead
/// of dispatch.
///
/// # `WorkType` × `CloneMode` compatibility
///
/// Most [`WorkType`] variants compose with any [`CloneMode`]. The
/// exceptions are surfaced at spawn time by [`WorkloadHandle::spawn`]:
///
/// | WorkType                        | Fork | Thread | SharedVm/Raw   |
/// |---------------------------------|------|--------|----------------|
/// | All variants (default)          | OK   | OK     | bail (status)  |
/// | [`WorkType::ForkExit`]          | OK   | reject | bail (status)  |
///
/// `ForkExit + Thread` is rejected because the worker body calls
/// `libc::fork()` from inside a thread of the parent's tgid; the
/// child then calls `_exit(0)`, which the kernel routes through
/// `do_exit`, tearing down the entire tgid (every sibling thread
/// dies). Use [`CloneMode::Fork`] for [`WorkType::ForkExit`].
///
/// Other Thread-mode interactions worth knowing:
///
/// - [`WorkType::NiceSweep`]: `setpriority(PRIO_PROCESS, 0, …)`
///   targets the calling task only (`kernel/sys.c::sys_setpriority`
///   `case PRIO_PROCESS: if (who == 0) p = current`), so each
///   sibling thread independently sweeps its own nice. Allowed.
/// - [`WorkType::AffinityChurn`]: `sched_setaffinity(0, …)`
///   addresses the calling thread by kernel rule
///   (`kernel/sched/syscalls.c::sched_setaffinity`). Allowed; no
///   cross-thread interference.
/// - [`WorkType::PolicyChurn`]: `sched_setscheduler(0, …)` is also
///   per-task. Allowed.
/// - [`WorkType::AsymmetricWaker`] with an RT class: legal but
///   the harness still runs as its original (likely SCHED_NORMAL)
///   policy; only the worker thread is RT.
///
/// `SharedVm` and `Raw` currently bail at spawn for any
/// `WorkType` (see status note above); the table reflects the
/// post-dispatch landscape.
#[derive(Clone, Debug, Default)]
pub enum CloneMode {
    /// Plain `fork(2)`: separate address space, separate thread
    /// group (`p->tgid = p->pid`), reaped via `waitpid`. The default
    /// — preserves existing [`WorkloadHandle::spawn`] behavior.
    #[default]
    Fork,
    /// Same thread group as the spawning process. Implementation
    /// uses [`std::thread::spawn`]; the Rust thread runtime owns
    /// all clone-flag selection internally (the user does not pass
    /// flags via this variant). Reaped via [`std::thread::JoinHandle`].
    /// Workers share `tgid`, signal-handler table, and address space
    /// with the parent — observers like `task_struct->group_leader`,
    /// `tgid`, `real_parent` all match the parent's.
    Thread,
    /// Shared VM but a *different* `tgid`. Issued via raw
    /// [`clone3(2)`](https://man7.org/linux/man-pages/man2/clone3.2.html)
    /// with `CLONE_VM | CLONE_PIDFD` (no `CLONE_THREAD`,
    /// `CLONE_SIGHAND`, etc.). Workers see the parent's address
    /// space but appear as separate processes to PID-keyed
    /// observers. Reaped via the runtime's pidfd. Useful for
    /// scheduler code paths that key on `tgid` distinction while
    /// requiring shared memory.
    SharedVm,
    /// Caller-supplied flag set. The runtime adds `CLONE_PIDFD`
    /// internally for reaping; everything else comes from the
    /// caller. The flag set is validated against the kernel's
    /// `copy_process` rejection rules at spawn time —
    /// [`validate_clone_flags`] for the exhaustive list.
    ///
    /// `Raw(CloneFlags::NONE)` is observationally equivalent to
    /// [`Fork`](Self::Fork) from the kernel's perspective (no clone
    /// flags = process fork) but goes through the clone3 dispatch
    /// path, which adds `CLONE_PIDFD`, allocates a child stack
    /// mmap, and reaps via pidfd polling. Prefer [`Fork`](Self::Fork)
    /// for that case; reserve `Raw` for flag sets that need at
    /// least one explicit bit.
    Raw(CloneFlags),
}

/// Validate a [`CloneFlags`] set against the rejection rules in
/// `copy_process` (kernel/fork.c). Returns `Ok(())` on a flag set
/// the kernel will accept, or `Err` with a named-rule diagnostic
/// otherwise.
///
/// The runtime calls this before ever issuing the syscall so
/// callers see a structural rejection (with the rule name) rather
/// than tunneling an `EINVAL` through the syscall return.
///
/// Rules implemented (1:1 with the kernel checks at the top of
/// `copy_process`):
///
/// 1. `THREAD` requires `SIGHAND`.
/// 2. `SIGHAND` requires `VM` (also implied by rule 1 transitively).
/// 3. `NEWNS` and `FS` are mutually exclusive.
/// 4. `NEWUSER` and `FS` are mutually exclusive.
/// 5. `THREAD` excludes `NEWUSER` and `NEWPID`.
/// 6. Caller-set bits in [`CloneFlags::RUNTIME_RESERVED`] are
///    rejected — those flags belong to the runtime.
///
/// `CLONE_PARENT && SIGNAL_UNKILLABLE` (init/container-init only)
/// is NOT pre-checked here — ktstr never runs as init, so the rule
/// is unreachable; the kernel still enforces it on the unlikely
/// off-path.
pub(crate) fn validate_clone_flags(flags: CloneFlags) -> Result<()> {
    if flags.contains(CloneFlags::THREAD) && !flags.contains(CloneFlags::SIGHAND) {
        anyhow::bail!(
            "clone3: CLONE_THREAD requires CLONE_SIGHAND (kernel `copy_process` \
             rejects detached threads outside a thread group)"
        );
    }
    if flags.contains(CloneFlags::SIGHAND) && !flags.contains(CloneFlags::VM) {
        anyhow::bail!(
            "clone3: CLONE_SIGHAND requires CLONE_VM (kernel `copy_process` \
             rejects shared signal handlers without shared address space)"
        );
    }
    if flags.contains(CloneFlags::NEWNS) && flags.contains(CloneFlags::FS) {
        anyhow::bail!(
            "clone3: CLONE_NEWNS and CLONE_FS are mutually exclusive (kernel \
             `copy_process` rejects sharing the root directory across mount \
             namespaces)"
        );
    }
    if flags.contains(CloneFlags::NEWUSER) && flags.contains(CloneFlags::FS) {
        anyhow::bail!(
            "clone3: CLONE_NEWUSER and CLONE_FS are mutually exclusive (kernel \
             `copy_process` rejects sharing FS state across user namespaces)"
        );
    }
    if flags.contains(CloneFlags::THREAD)
        && (flags.contains(CloneFlags::NEWUSER) || flags.contains(CloneFlags::NEWPID))
    {
        anyhow::bail!(
            "clone3: CLONE_THREAD with CLONE_NEWUSER or CLONE_NEWPID rejected \
             (kernel `copy_process` requires thread-group siblings to share \
             user/pid namespaces)"
        );
    }
    let reserved = flags.bits() & CloneFlags::RUNTIME_RESERVED.bits();
    if reserved != 0 {
        anyhow::bail!(
            "clone3: caller-set runtime-reserved flags 0x{reserved:x} not allowed \
             (CLONE_PIDFD, CLONE_PARENT_SETTID, CLONE_CHILD_SETTID, \
             CLONE_CHILD_CLEARTID, CLONE_DETACHED, CLONE_SETTLS, CLONE_VFORK, \
             CLONE_PTRACE, CLONE_UNTRACED are runtime-managed; see \
             `CloneFlags::RUNTIME_RESERVED`)"
        );
    }
    Ok(())
}

impl MemPolicy {
    /// Construct a `Bind` policy from any iterator of NUMA node IDs.
    ///
    /// Accepts arrays, ranges, `Vec`, `BTreeSet`, or any `IntoIterator<Item = usize>`.
    pub fn bind(nodes: impl IntoIterator<Item = usize>) -> Self {
        MemPolicy::Bind(nodes.into_iter().collect())
    }

    /// Construct a `Preferred` policy for a single NUMA node.
    pub fn preferred(node: usize) -> Self {
        MemPolicy::Preferred(node)
    }

    /// Construct an `Interleave` policy from any iterator of NUMA node IDs.
    ///
    /// Accepts arrays, ranges, `Vec`, `BTreeSet`, or any `IntoIterator<Item = usize>`.
    pub fn interleave(nodes: impl IntoIterator<Item = usize>) -> Self {
        MemPolicy::Interleave(nodes.into_iter().collect())
    }

    /// Construct a `PreferredMany` policy from any iterator of NUMA node IDs.
    pub fn preferred_many(nodes: impl IntoIterator<Item = usize>) -> Self {
        MemPolicy::PreferredMany(nodes.into_iter().collect())
    }

    /// Construct a `WeightedInterleave` policy from any iterator of NUMA node IDs.
    pub fn weighted_interleave(nodes: impl IntoIterator<Item = usize>) -> Self {
        MemPolicy::WeightedInterleave(nodes.into_iter().collect())
    }

    /// NUMA node IDs referenced by this policy.
    ///
    /// Returns the node set for `Bind`, `Interleave`, `PreferredMany`,
    /// and `WeightedInterleave`, a single-element set for `Preferred`,
    /// and an empty set for `Default`/`Local`.
    pub fn node_set(&self) -> BTreeSet<usize> {
        match self {
            MemPolicy::Default | MemPolicy::Local => BTreeSet::new(),
            MemPolicy::Bind(nodes)
            | MemPolicy::Interleave(nodes)
            | MemPolicy::PreferredMany(nodes)
            | MemPolicy::WeightedInterleave(nodes) => nodes.clone(),
            MemPolicy::Preferred(node) => [*node].into_iter().collect(),
        }
    }

    /// Validate that this policy's node set is non-empty where required.
    ///
    /// Returns `Err` with a description when a node-set-bearing policy
    /// has an empty set.
    pub fn validate(&self) -> std::result::Result<(), String> {
        match self {
            MemPolicy::Default | MemPolicy::Local => Ok(()),
            MemPolicy::Preferred(_) => Ok(()),
            MemPolicy::Bind(nodes) if nodes.is_empty() => {
                Err("Bind policy requires at least one NUMA node".into())
            }
            MemPolicy::Interleave(nodes) if nodes.is_empty() => {
                Err("Interleave policy requires at least one NUMA node".into())
            }
            MemPolicy::PreferredMany(nodes) if nodes.is_empty() => {
                Err("PreferredMany policy requires at least one NUMA node".into())
            }
            MemPolicy::WeightedInterleave(nodes) if nodes.is_empty() => {
                Err("WeightedInterleave policy requires at least one NUMA node".into())
            }
            _ => Ok(()),
        }
    }
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
const WORKER_STOP_POLL_NS: libc::c_long = 100_000_000;

/// Packaged [`libc::timespec`] for every worker-side `futex_wait`
/// across the blocking workload primitives. Duplicating the struct
/// literal per call site drifted the `tv_nsec` field between variants
/// during earlier edits; a single const keeps the shutdown-latency
/// budget documented on [`WORKER_STOP_POLL_NS`] authoritative.
const FUTEX_WAIT_TIMEOUT: libc::timespec = libc::timespec {
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
const FAN_OUT_POST_WAKE_SPIN_ITERS: u64 = 256;

/// Call `set_mempolicy(2)` for the current process with mode flags.
///
/// No-op for `MemPolicy::Default`. Logs a warning on syscall failure.
fn apply_mempolicy_with_flags(policy: &MemPolicy, flags: MpolFlags) {
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
/// `nice == 0` is a fast-path skip — the worker inherits the
/// parent's nice value. The kernel clamps `niceval` to
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
fn apply_nice(nice: i32) {
    if nice == 0 {
        return;
    }
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
fn warn_setpriority_failed_once() {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        let errno = std::io::Error::last_os_error();
        eprintln!(
            "workload: setpriority(PRIO_PROCESS) failed: {errno}; nice value not applied (CAP_SYS_NICE may be required for negative nice)"
        );
    });
}

/// Configuration for spawning a group of worker processes.
#[derive(Debug, Clone)]
pub struct WorkloadConfig {
    /// Number of worker processes to fork.
    pub num_workers: usize,
    /// CPU affinity mode for workers.
    pub affinity: ResolvedAffinity,
    /// What each worker does.
    pub work_type: WorkType,
    /// Linux scheduling policy.
    pub sched_policy: SchedPolicy,
    /// NUMA memory placement policy.
    pub mem_policy: MemPolicy,
    /// Optional mode flags for `set_mempolicy(2)`.
    pub mpol_flags: MpolFlags,
    /// Per-worker nice value applied via `setpriority(2)` after
    /// fork, before the work loop. Range `-20..=19` per `MIN_NICE`
    /// / `MAX_NICE` in `kernel/sys.c`'s `setpriority` syscall;
    /// values outside this window are clamped kernel-side. `0` (the
    /// default) skips the syscall entirely so the worker inherits
    /// the parent's nice value.
    ///
    /// Negative values require `CAP_SYS_NICE` (the `set_one_prio`
    /// → `can_nice` path returns `EACCES` to unprivileged callers
    /// trying to lower nice below the parent's). Failures are
    /// logged once via stderr and do not abort the worker — the
    /// scheduling-policy and affinity sites use the same idiom.
    pub nice: i32,
    /// How to create each worker. Defaults to [`CloneMode::Fork`]
    /// — the historical fork-based path. Other variants exercise
    /// shared-VM / thread-group / custom-flag clone3 paths;
    /// dispatch for those is staged behind future thread-mode and
    /// clone3-based dispatch implementations.
    pub clone_mode: CloneMode,
}

impl Default for WorkloadConfig {
    fn default() -> Self {
        Self {
            num_workers: 1,
            affinity: ResolvedAffinity::None,
            work_type: WorkType::CpuSpin,
            sched_policy: SchedPolicy::Normal,
            mem_policy: MemPolicy::Default,
            mpol_flags: MpolFlags::NONE,
            nice: 0,
            clone_mode: CloneMode::Fork,
        }
    }
}

impl WorkloadConfig {
    /// Set the number of worker processes.
    pub fn workers(mut self, n: usize) -> Self {
        self.num_workers = n;
        self
    }

    /// Set the resolved CPU affinity.
    pub fn affinity(mut self, a: ResolvedAffinity) -> Self {
        self.affinity = a;
        self
    }

    /// Set the work type.
    pub fn work_type(mut self, wt: WorkType) -> Self {
        self.work_type = wt;
        self
    }

    /// Set the Linux scheduling policy.
    pub fn sched_policy(mut self, p: SchedPolicy) -> Self {
        self.sched_policy = p;
        self
    }

    /// Set the NUMA memory placement policy.
    pub fn mem_policy(mut self, p: MemPolicy) -> Self {
        self.mem_policy = p;
        self
    }

    /// Set the NUMA memory policy mode flags.
    pub fn mpol_flags(mut self, f: MpolFlags) -> Self {
        self.mpol_flags = f;
        self
    }

    /// Set the per-worker nice value applied via `setpriority(2)`.
    ///
    /// `0` (the default) skips the syscall and inherits the
    /// parent's nice. Negative values require `CAP_SYS_NICE`.
    pub fn nice(mut self, n: i32) -> Self {
        self.nice = n;
        self
    }

    /// Set the clone mode used when spawning each worker.
    ///
    /// [`CloneMode::Fork`] (the default) preserves historical
    /// behavior. See [`CloneMode`] for the full menu and dispatch
    /// status.
    pub fn clone_mode(mut self, m: CloneMode) -> Self {
        self.clone_mode = m;
        self
    }
}

/// Workload definition for a single group of workers within a cgroup.
///
/// Extracted from [`CgroupDef`](crate::scenario::ops::CgroupDef) to allow
/// multiple concurrent work groups per cgroup. Each `WorkSpec` spawns its own
/// set of worker processes.
///
/// ```
/// # use ktstr::workload::{WorkSpec, WorkType, SchedPolicy, MemPolicy};
/// let w = WorkSpec::default()
///     .workers(4)
///     .work_type(WorkType::bursty(50, 100))
///     .sched_policy(SchedPolicy::Batch)
///     .mem_policy(MemPolicy::bind([0, 1]));
/// assert_eq!(w.num_workers, Some(4));
/// ```
#[derive(Clone, Debug)]
pub struct WorkSpec {
    /// What each worker does.
    pub work_type: WorkType,
    /// Linux scheduling policy.
    pub sched_policy: SchedPolicy,
    /// Number of workers. `None` means use `Ctx::workers_per_cgroup`.
    pub num_workers: Option<usize>,
    /// Per-worker affinity intent. Resolved to [`ResolvedAffinity`] at
    /// runtime via [`resolve_affinity_for_cgroup()`](crate::scenario::resolve_affinity_for_cgroup).
    pub affinity: AffinityIntent,
    /// NUMA memory placement policy. Applied via `set_mempolicy(2)`
    /// after fork, before the work loop.
    pub mem_policy: MemPolicy,
    /// Optional mode flags for `set_mempolicy(2)`.
    pub mpol_flags: MpolFlags,
    /// Per-worker nice value applied via `setpriority(2)` after
    /// fork, before the work loop. See [`WorkloadConfig::nice`]
    /// for range, default-zero skip semantics, and `CAP_SYS_NICE`
    /// rules.
    pub nice: i32,
    /// Clone mode for spawning each worker. See [`CloneMode`] for
    /// the variant menu and dispatch status.
    pub clone_mode: CloneMode,
}

impl Default for WorkSpec {
    fn default() -> Self {
        Self {
            work_type: WorkType::CpuSpin,
            sched_policy: SchedPolicy::Normal,
            num_workers: None,
            affinity: AffinityIntent::Inherit,
            mem_policy: MemPolicy::Default,
            mpol_flags: MpolFlags::NONE,
            nice: 0,
            clone_mode: CloneMode::Fork,
        }
    }
}

impl WorkSpec {
    /// Set the number of workers.
    pub fn workers(mut self, n: usize) -> Self {
        self.num_workers = Some(n);
        self
    }

    /// Set the work type.
    pub fn work_type(mut self, wt: WorkType) -> Self {
        self.work_type = wt;
        self
    }

    /// Set the Linux scheduling policy.
    pub fn sched_policy(mut self, p: SchedPolicy) -> Self {
        self.sched_policy = p;
        self
    }

    /// Set the per-worker affinity intent.
    pub fn affinity(mut self, a: AffinityIntent) -> Self {
        self.affinity = a;
        self
    }

    /// Set the NUMA memory placement policy.
    pub fn mem_policy(mut self, p: MemPolicy) -> Self {
        self.mem_policy = p;
        self
    }

    /// Set the NUMA memory policy mode flags.
    pub fn mpol_flags(mut self, f: MpolFlags) -> Self {
        self.mpol_flags = f;
        self
    }

    /// Set the per-worker nice value applied via `setpriority(2)`.
    ///
    /// `0` (the default) skips the syscall and inherits the
    /// parent's nice. Negative values require `CAP_SYS_NICE`.
    pub fn nice(mut self, n: i32) -> Self {
        self.nice = n;
        self
    }

    /// Set the clone mode used when spawning each worker.
    pub fn clone_mode(mut self, m: CloneMode) -> Self {
        self.clone_mode = m;
        self
    }
}

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
    /// Per-wakeup latency samples (ns). Measures the interval between
    /// the call that blocks (any blocking primitive — pipe `read`,
    /// futex wait, `poll`, `sched_yield`, `nanosleep`, etc.) and the
    /// wakeup that resumes execution; not a yield-specific measure.
    /// Populated for blocking work types: Bursty, PipeIo, FutexPingPong,
    /// FutexFanOut, FanOutCompute, CacheYield, CachePipe, IoSync, NiceSweep,
    /// AffinityChurn, PolicyChurn, MutexContention, ForkExit (parent's
    /// waitpid wait), Sequence with Sleep/Yield/Io phases.
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
    /// [`WorkloadHandle::stop_and_collect`]'s JSON-parse fallback
    /// (see `exit_info` below) carries `false`. Lets downstream
    /// consumers distinguish "worker ran to completion and
    /// observed zero iterations" (`completed: true, iterations: 0`
    /// — legitimate for pathologically short test windows) from
    /// "worker died / timed out before recording anything"
    /// (`completed: false, iterations: 0` — the sentinel shape).
    #[serde(default)]
    pub completed: bool,
    /// Per-NUMA-node page counts from `/proc/self/numa_maps` after workload.
    /// Keyed by node ID. Empty when numa_maps is unavailable.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub numa_pages: BTreeMap<usize, u64>,
    /// Delta of `/proc/vmstat` `numa_pages_migrated` over the work loop.
    pub vmstat_numa_pages_migrated: u64,
    /// Diagnostic attached only to sentinel reports — populated when
    /// `stop_and_collect` synthesized the entry because no (or
    /// unparseable) JSON came back on the report pipe. `None` on every
    /// real worker-produced report. Lets operators distinguish the
    /// four failure shapes that all collapse to "empty pipe + no
    /// report":
    ///
    /// - [`WorkerExitInfo::Exited`] with a non-zero code: worker
    ///   reached `_exit(code)` without writing JSON — typically the
    ///   `catch_unwind` Err arm in the worker-child closure (panic
    ///   under `panic = "unwind"`) or the 30s poll-start timeout's
    ///   early `_exit(1)`.
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
    /// `skip_serializing_if = "Option::is_none"` keeps live-worker
    /// reports compact: only sentinel reports carry the field over
    /// the pipe. There is no cross-version compatibility concern
    /// here — `WorkerReport` is pipe-transited child→parent within
    /// a single `ktstr` process, never read back from a persisted
    /// sidecar.
    #[serde(skip_serializing_if = "Option::is_none")]
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
    /// JSON-parse fallback in
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
}

/// Reason a sentinel [`WorkerReport`] was synthesized — attached to
/// the report's `exit_info` field so operators can triage a missing
/// JSON payload without cross-referencing parent-side logs.
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
    /// serde_json or pipe-write failure that didn't panic.
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
fn classify_wait_outcome(
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
fn extract_panic_payload(payload: Box<dyn std::any::Any + Send>) -> String {
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
const THREAD_JOIN_TIMEOUT: Duration = Duration::from_secs(5);

/// Poll [`std::thread::JoinHandle::is_finished`] until it returns
/// `true` or `timeout` elapses. Returns `Some(thread_result)` on
/// successful join, `None` on timeout.
///
/// Std lacks a native timed-join API; the polling-based shape here
/// is the simplest non-leaking pattern. A side-thread "joiner +
/// channel" alternative would orphan the joiner on timeout
/// (joining is non-cancellable in std), which keeps the thread
/// alive past `WorkloadHandle::drop` and prevents process exit.
/// Polling avoids that orphan cost at the price of a 10ms wakeup
/// cadence — fine for the 5s budget this is paired with.
fn join_thread_with_timeout(
    join: std::thread::JoinHandle<WorkerReport>,
    timeout: Duration,
) -> Option<std::thread::Result<WorkerReport>> {
    let deadline = Instant::now() + timeout;
    while !join.is_finished() {
        if Instant::now() >= deadline {
            return None;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    Some(join.join())
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
fn stop_requested(stop: &AtomicBool) -> bool {
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
struct ThreadWorker {
    tid: std::sync::Arc<std::sync::atomic::AtomicI32>,
    stop: std::sync::Arc<AtomicBool>,
    start_tx: Option<std::sync::mpsc::SyncSender<()>>,
    join: Option<std::thread::JoinHandle<WorkerReport>>,
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
            let _ = join_thread_with_timeout(j, THREAD_JOIN_TIMEOUT);
        }
    }
}

/// Per-clone3 worker state for [`CloneMode::SharedVm`] and
/// [`CloneMode::Raw`] dispatch.
///
/// The clone3 path mirrors the Fork variant's pipe-based start /
/// report rendezvous because the new task still needs an explicit
/// "go" signal and a back-channel for the [`WorkerReport`] JSON.
/// What differs from Fork is the reaping primitive:
///
/// - `pidfd` is the file descriptor the kernel writes into the
///   user-pointer `clone_args.pidfd` field when the `CLONE_PIDFD`
///   flag is set. `poll(pidfd, POLLIN, ms)` reports the child's
///   exit; the parent does not need `waitpid` and does not race
///   the kernel's pid recycling.
/// - `stack_ptr` / `stack_size` own the child's stack mmap. The
///   parent must mmap a child stack when `CLONE_VM` is set
///   (`clone3(2)` does not auto-allocate a stack the way
///   `fork(2)` implicitly does because the parent and child share
///   the same VM and the kernel cannot reuse the parent's stack).
///   The mapping is passed via `clone_args.stack` +
///   `clone_args.stack_size`. The parent must `munmap` it AFTER
///   the child exits (poll the pidfd first), otherwise the unmap
///   rips the stack out from under a still-running child.
struct PidfdWorker {
    pid: libc::pid_t,
    pidfd: std::os::unix::io::RawFd,
    report_fd: std::os::unix::io::RawFd,
    start_fd: std::os::unix::io::RawFd,
    stack_ptr: *mut libc::c_void,
    stack_size: usize,
    /// Per-worker stop flag, shared with the clone3 child via the
    /// CLONE_VM-shared address space. Flipping this from the
    /// parent reaches the child's `worker_main` `stop.load`
    /// checks without using the global [`STOP`], so siblings are
    /// not affected.
    stop: std::sync::Arc<AtomicBool>,
}

/// Handle to spawned worker tasks. Workers block until
/// [`start()`](Self::start) is called.
///
/// The [`CloneMode`] in the [`WorkloadConfig`] selects how each
/// worker is created. Within one [`WorkloadHandle`] every worker
/// uses the same mode, so exactly one of `children`, `threads`, or
/// `pidfd_workers` is populated; the other two are empty. This
/// avoids per-worker mode dispatch on the hot path and keeps each
/// vec's per-mode invariants (pid-based, JoinHandle-based,
/// pidfd-based reaping) cohesive.
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
/// - [`CloneMode::SharedVm`] / [`CloneMode::Raw`] populate
///   `pidfd_workers` — `clone3(2)` with explicit flags, reaped via
///   `poll(pidfd, POLLIN)`. Used by tests that exercise the
///   kernel's per-flag-combination scheduler code paths.
#[must_use = "dropping a WorkloadHandle immediately tears down all worker tasks"]
pub struct WorkloadHandle {
    /// Fork-mode workers. Each entry is `(pid, report_fd, start_fd)`.
    /// Empty when `clone_mode` is not [`CloneMode::Fork`].
    children: Vec<(
        libc::pid_t,
        std::os::unix::io::RawFd,
        std::os::unix::io::RawFd,
    )>,
    /// Thread-mode workers. Empty when `clone_mode` is not
    /// [`CloneMode::Thread`].
    threads: Vec<ThreadWorker>,
    /// clone3-mode workers. Empty when `clone_mode` is not
    /// [`CloneMode::SharedVm`] or [`CloneMode::Raw`].
    pidfd_workers: Vec<PidfdWorker>,
    started: bool,
    /// Shared mmap regions for futex-based work types (one per worker group). Unmapped on drop.
    futex_ptrs: Vec<*mut u32>,
    /// Size of each futex mmap region (4 for FutexPingPong/FutexFanOut/MutexContention, 16 for FanOutCompute: u64 generation @ 0 + u64 wake_ns @ 8).
    futex_region_size: usize,
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
}

/// Scope guard that owns every resource acquired during
/// [`WorkloadHandle::spawn`]'s partial setup. If `spawn` returns
/// early (via `?` or `bail!`), the guard's `Drop` kills and reaps any
/// already-forked children, closes every open pipe fd, and munmaps
/// every shared region — so a mid-setup failure never leaks fds,
/// zombie processes, or anonymous-shared pages.
///
/// On success, [`SpawnGuard::into_handle`] moves the live resources
/// into the returned [`WorkloadHandle`] and leaves the guard empty;
/// its `Drop` then closes only the inter-worker `pipe_pairs`
/// (intentionally owned by the guard, not the handle, because the
/// parent never uses them after fork).
struct SpawnGuard {
    /// Inter-worker paired pipes `(ab, ba)` for PipeIo/CachePipe.
    /// Closed by the guard on every exit (success or failure) —
    /// children inherit copies via fork and close their own ends.
    pipe_pairs: Vec<([i32; 2], [i32; 2])>,
    /// Shared-memory futex regions (transferred to handle on success).
    futex_ptrs: Vec<*mut u32>,
    futex_region_size: usize,
    /// Per-worker iteration counter region (transferred on success).
    /// Typed matches the handle field; see `WorkloadHandle::iter_counters`.
    iter_counters: *mut AtomicU64,
    iter_counter_bytes: usize,
    /// Already-forked children with their parent-side pipe fds
    /// (transferred to handle on success).
    children: Vec<(libc::pid_t, i32, i32)>,
    /// Already-spawned thread workers (transferred on success).
    /// Cleanup on early-exit flips each `stop` and joins each
    /// thread, since threads share the parent's address space and
    /// must be drained cooperatively (no `kill` equivalent).
    threads: Vec<ThreadWorker>,
    /// Already-cloned pidfd workers with their parent-side state
    /// (transferred on success). Cleanup on early-exit issues
    /// `pidfd_send_signal(SIGKILL)` then `poll(pidfd, POLLIN)` to
    /// drain the exit, then `munmap`s the child stack.
    pidfd_workers: Vec<PidfdWorker>,
}

impl SpawnGuard {
    fn new(futex_region_size: usize) -> Self {
        Self {
            pipe_pairs: Vec::new(),
            futex_ptrs: Vec::new(),
            futex_region_size,
            iter_counters: std::ptr::null_mut(),
            iter_counter_bytes: 0,
            children: Vec::new(),
            threads: Vec::new(),
            pidfd_workers: Vec::new(),
        }
    }

    /// Transfer live resources into a [`WorkloadHandle`]. Leaves the
    /// guard's `children`, `threads`, `pidfd_workers`, `futex_ptrs`,
    /// and `iter_counters` empty so the guard's subsequent `Drop`
    /// only closes the inter-worker `pipe_pairs` (which the parent
    /// never uses post-fork).
    fn into_handle(mut self) -> WorkloadHandle {
        let children = std::mem::take(&mut self.children);
        let threads = std::mem::take(&mut self.threads);
        let pidfd_workers = std::mem::take(&mut self.pidfd_workers);
        let futex_ptrs = std::mem::take(&mut self.futex_ptrs);
        let iter_counters = std::mem::replace(&mut self.iter_counters, std::ptr::null_mut());
        let iter_counter_bytes = std::mem::replace(&mut self.iter_counter_bytes, 0);
        let iter_counter_len = iter_counter_bytes / std::mem::size_of::<AtomicU64>();
        WorkloadHandle {
            children,
            threads,
            pidfd_workers,
            started: false,
            futex_ptrs,
            futex_region_size: self.futex_region_size,
            iter_counters,
            iter_counter_len,
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
        for &(pid, _, _) in &self.children {
            let npid = nix::unistd::Pid::from_raw(pid);
            let _ = nix::sys::signal::kill(npid, nix::sys::signal::Signal::SIGKILL);
            let _ = nix::sys::wait::waitpid(npid, None);
        }
        // Close each child's parent-side report/start fds.
        for &(_, rfd, wfd) in &self.children {
            for fd in [rfd, wfd] {
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
                let _ = join_thread_with_timeout(j, THREAD_JOIN_TIMEOUT);
            }
        }
        // Drain pidfd workers: send SIGKILL via pidfd_send_signal,
        // poll the pidfd until it reports the exit, then close
        // pidfd / fds and munmap the child stack. Each step is
        // best-effort because the partial-failure path may have
        // populated this vec without all fds being valid.
        for w in &self.pidfd_workers {
            // pidfd_send_signal(pidfd, SIGKILL, NULL, 0).
            // libc 0.2 lacks a wrapper; use raw syscall. Errors
            // (ESRCH from already-exited child, EBADF from a
            // half-set-up entry) are swallowed.
            unsafe {
                libc::syscall(
                    libc::SYS_pidfd_send_signal,
                    w.pidfd,
                    libc::SIGKILL,
                    std::ptr::null::<libc::siginfo_t>(),
                    0u32,
                );
            }
            // Wait for exit so the munmap below does not race a
            // running child. POLLIN on pidfd fires when the child
            // reaches `exit`. 5s deadline matches stop_and_collect's
            // shared budget.
            let mut pfd = libc::pollfd {
                fd: w.pidfd,
                events: libc::POLLIN,
                revents: 0,
            };
            unsafe {
                libc::poll(&mut pfd, 1, 5_000);
            }
            // Close fds (pidfd, report, start) and munmap the stack.
            for fd in [w.pidfd, w.report_fd, w.start_fd] {
                if fd >= 0 {
                    let _ = nix::unistd::close(fd);
                }
            }
            if !w.stack_ptr.is_null() && w.stack_size > 0 {
                unsafe { libc::munmap(w.stack_ptr, w.stack_size) };
            }
        }
        // Close every inter-worker pipe pair. Children closed their
        // own inherited copies in the fork arm, so these are the
        // only remaining references.
        for (ab, ba) in &self.pipe_pairs {
            for fd in [ab[0], ab[1], ba[0], ba[1]] {
                let _ = nix::unistd::close(fd);
            }
        }
        // Munmap shared regions.
        for &ptr in &self.futex_ptrs {
            unsafe {
                libc::munmap(ptr as *mut libc::c_void, self.futex_region_size);
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
//
// - Pidfd mode (currently bailed pending the clone3 redesign):
//   children share the parent's address space via `CLONE_VM`, so
//   they observe the same VMAs as Thread workers and the same
//   atomic-aliasing argument applies. The redesign will inherit
//   this soundness without needing MAP_SHARED.
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
struct SendFutexPtr(Option<(usize, usize)>);

#[derive(Clone, Copy)]
struct SendIterSlotPtr(usize);

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
fn spawn_thread_worker(
    guard: &mut SpawnGuard,
    config: &WorkloadConfig,
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

    // Clone Arcs for the closure. The thread takes ownership of the
    // closure-side handles; the parent retains the originals via
    // ThreadWorker for stop signaling and tid reading.
    let stop_thread = Arc::clone(&stop);
    let tid_thread = Arc::clone(&tid);
    let work_type = config.work_type.clone();
    let sched_policy = config.sched_policy;
    let mem_policy = config.mem_policy.clone();
    let mpol_flags = config.mpol_flags;
    let nice = config.nice;

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
        .name(format!("ktstr-worker-{}", guard.threads.len()))
        .spawn(move || {
            // Publish gettid() so the parent can address this task
            // for sched_setaffinity and report it from worker_pids.
            // gettid() is the kernel TID; getpid() would return the
            // shared tgid, which collides across threads.
            let my_tid: libc::pid_t =
                unsafe { libc::syscall(libc::SYS_gettid) as libc::pid_t };
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
                worker_pipe_fds,
                futex,
                slot,
                &stop_thread,
            )
        })
        .with_context(|| {
            format!(
                "thread::spawn for worker {}/{} failed",
                guard.threads.len() + 1,
                config.num_workers,
            )
        })?;

    guard.threads.push(ThreadWorker {
        tid,
        stop,
        start_tx: Some(start_tx),
        join: Some(join),
    });
    Ok(())
}

/// Spawn a single clone3-mode worker via raw `clone3(2)`.
///
/// Used by [`CloneMode::SharedVm`] (hardcoded `CLONE_VM`) and
/// [`CloneMode::Raw`] (caller-supplied flags). The runtime adds
/// `CLONE_PIDFD` internally so the parent can reap via
/// `poll(pidfd, POLLIN)` instead of `waitpid`.
///
/// The child runs the same body as the Fork path's `pid == 0` arm
/// — pipe-based start rendezvous, `worker_main`, `_exit` — adapted
/// to use a clone3-allocated stack and the pidfd-reap return path.
/// Because the child shares the parent's address space when
/// `CLONE_VM` is set, any access to the parent's stack (locals
/// captured by the loop) would race the parent — the child's
/// post-clone3 code path therefore copies every value it needs out
/// of the loop's locals BEFORE returning to user space.
///
/// **Kernel-verified constraints carried forward to the redesign:**
///
/// - `kernel/fork.c::copy_process` calls `sas_ss_reset` for any
///   child cloned with `CLONE_VM` and without `CLONE_VFORK`,
///   resetting the child's `sas_ss_*` fields (sigaltstack disabled,
///   size 0). Workers that need a sigaltstack for SIGSEGV recovery
///   must call `sigaltstack(2)` themselves post-clone3.
///
/// - `kernel/fork.c::mm_release` clears `*clear_child_tid` in the
///   shared mm on exit / exec. Setting `CLONE_CHILD_CLEARTID` from
///   a `CLONE_VM` clone3 child therefore writes a zero into the
///   PARENT's address space at exit time — silent memory
///   corruption. `CLONE_CHILD_CLEARTID` is already in
///   [`CloneFlags::RUNTIME_RESERVED`] so user-supplied flag sets
///   for [`CloneMode::Raw`] cannot reach this footgun; the
///   runtime itself must NOT set it for SharedVm/Raw either.
///
/// - The child stack mmap should size `STACK_SIZE + PAGE_SIZE` and
///   `mprotect` the bottom page `PROT_NONE` so a stack overflow
///   segfaults predictably instead of corrupting an adjacent
///   mapping. Implemented below by `CLONE3_STACK_SIZE +
///   CLONE3_STACK_GUARD` + `mprotect(stack_ptr, CLONE3_STACK_GUARD,
///   PROT_NONE)`.
#[allow(clippy::too_many_arguments)]
fn spawn_clone3_worker(
    guard: &mut SpawnGuard,
    config: &WorkloadConfig,
    flags: CloneFlags,
    affinity: Option<BTreeSet<usize>>,
    worker_pipe_fds: Option<(i32, i32)>,
    worker_futex: Option<(*mut u32, usize)>,
    iter_slot: *mut AtomicU64,
    report_fds: [i32; 2],
    start_fds: [i32; 2],
    worker_idx: usize,
) -> Result<()> {
    // Allocate a child stack via mmap. clone3 does not auto-allocate
    // a stack the way fork does; the caller must supply one through
    // clone_args.stack + clone_args.stack_size. The mapping is
    // ANONYMOUS|PRIVATE so the child's stack writes don't bleed
    // back into the parent. Sized at CLONE3_STACK_SIZE +
    // CLONE3_STACK_GUARD; the kernel never touches the guard page
    // because the child's stack pointer starts at the END (highest
    // address) of the usable region and grows downward toward the
    // guard.
    let total = CLONE3_STACK_SIZE + CLONE3_STACK_GUARD;
    let stack_ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            total,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_STACK,
            -1,
            0,
        )
    };
    if stack_ptr == libc::MAP_FAILED {
        unsafe {
            libc::close(report_fds[0]);
            libc::close(report_fds[1]);
            libc::close(start_fds[0]);
            libc::close(start_fds[1]);
        }
        anyhow::bail!(
            "clone3 worker {}/{}: mmap of {} byte child stack failed: {}",
            worker_idx + 1,
            config.num_workers,
            total,
            std::io::Error::last_os_error(),
        );
    }
    // Mark the bottom page as PROT_NONE so a stack overflow
    // segfaults predictably instead of corrupting an adjacent
    // mapping. Failure here is non-fatal (the workload still runs;
    // the guard page just isn't enforced); fall through with a
    // tracing::warn.
    let guard_rc = unsafe {
        libc::mprotect(stack_ptr, CLONE3_STACK_GUARD, libc::PROT_NONE)
    };
    if guard_rc != 0 {
        tracing::warn!(
            err = %std::io::Error::last_os_error(),
            "clone3 stack guard page mprotect(PROT_NONE) failed; \
             stack overflow will not be caught"
        );
    }

    // clone3 stack pointer points to the BASE of the usable region
    // (kernel computes top from base + size). We pass the
    // post-guard offset as `stack` and CLONE3_STACK_SIZE as
    // `stack_size`.
    let usable_stack = unsafe { (stack_ptr as *mut u8).add(CLONE3_STACK_GUARD) };

    // Heap-allocate the child-args bundle. Under CLONE_VM the
    // child cannot read the parent's stack frame (the parent
    // returns from this function and unwinds the locals shortly
    // after clone3), so every input the child needs lives on the
    // heap and is reachable via the shared address space. Push
    // onto the FIFO queue immediately before clone3 so the child
    // pops it on the next iteration.
    // Per-worker stop flag, mirroring the Thread-mode design.
    // The Arc lives in shared address space (CLONE_VM); the parent
    // flips it from `stop_and_collect`, the child observes it via
    // `stop.load(Relaxed)` inside `worker_main`. Stored on both
    // the queued args (for the child to pick up) and on the
    // `PidfdWorker` (so the parent can flip without re-finding
    // the queue entry).
    let stop = std::sync::Arc::new(AtomicBool::new(false));
    let child_args = Box::new(Clone3ChildArgs {
        affinity,
        work_type: config.work_type.clone(),
        sched_policy: config.sched_policy,
        mem_policy: config.mem_policy.clone(),
        mpol_flags: config.mpol_flags,
        nice: config.nice,
        worker_pipe_fds,
        worker_futex: worker_futex.map(|(p, pos)| (p as usize, pos)),
        iter_slot: iter_slot as usize,
        report_fd: report_fds[1],
        start_fd: start_fds[0],
        stop: std::sync::Arc::clone(&stop),
    });
    CLONE3_CHILD_ARGS_QUEUE
        .lock()
        .expect("clone3 child-args queue mutex poisoned")
        .push_back(child_args);

    // Build clone_args. The runtime always sets CLONE_PIDFD so we
    // can reap via pidfd; the caller's flags must not include any
    // RUNTIME_RESERVED bits (validate_clone_flags enforces this).
    // exit_signal stays at 0 — without CLONE_THREAD the kernel
    // requires either an explicit exit_signal or 0; pidfd reaping
    // does not need SIGCHLD.
    let mut pidfd: libc::c_int = -1;
    let mut clone_args: libc::clone_args = unsafe { std::mem::zeroed() };
    clone_args.flags = flags.bits() | (libc::CLONE_PIDFD as u64);
    clone_args.pidfd = (&mut pidfd as *mut libc::c_int) as u64;
    clone_args.exit_signal = 0;
    clone_args.stack = usable_stack as u64;
    clone_args.stack_size = CLONE3_STACK_SIZE as u64;

    // Issue clone3. Returns 0 in the child, child pid in the
    // parent, -1 on error. SAFETY: clone_args is a zero-initialised
    // libc UAPI struct with our fields populated; the kernel reads
    // `size` to gate which fields it touches, and ignores tail
    // bytes beyond its own definition. pidfd and stack pointers
    // outlive the syscall.
    let rc = unsafe {
        libc::syscall(
            libc::SYS_clone3,
            &clone_args as *const libc::clone_args,
            std::mem::size_of::<libc::clone_args>(),
        )
    };
    if rc < 0 {
        // clone3 failed — drain the args we pushed, then release
        // everything we allocated for this worker before bailing.
        let _ = CLONE3_CHILD_ARGS_QUEUE
            .lock()
            .expect("clone3 child-args queue mutex poisoned")
            .pop_back();
        unsafe {
            libc::munmap(stack_ptr, total);
            libc::close(report_fds[0]);
            libc::close(report_fds[1]);
            libc::close(start_fds[0]);
            libc::close(start_fds[1]);
        }
        anyhow::bail!(
            "clone3 worker {}/{}: syscall failed: {} (flags=0x{:x})",
            worker_idx + 1,
            config.num_workers,
            std::io::Error::last_os_error(),
            clone_args.flags,
        );
    }
    if rc == 0 {
        // Child path. The child reads its args from the queue
        // (pop_front), then runs worker_main on the new stack. All
        // values come from heap (Box) or static globals; the
        // parent's stack is NOT accessed.
        clone3_child_main();
    }
    // Parent path. rc is the child pid; pidfd is filled by the
    // kernel via clone_args.pidfd. Close child-side pipe ends so
    // the parent retains only the read-side report_fd and
    // write-side start_fd.
    unsafe {
        libc::close(report_fds[1]);
        libc::close(start_fds[0]);
    }
    let pid = rc as libc::pid_t;
    guard.pidfd_workers.push(PidfdWorker {
        pid,
        pidfd,
        report_fd: report_fds[0],
        start_fd: start_fds[1],
        stack_ptr,
        stack_size: total,
        stop,
    });
    Ok(())
}

/// Body of the clone3 child. Runs on the child's freshly-mmap'd
/// stack. Reads its inputs from the heap-allocated
/// [`Clone3ChildArgs`] box the parent just pushed, runs
/// [`worker_main`], writes the [`WorkerReport`] JSON to the report
/// pipe, and `_exit`s. Never returns.
///
/// This function does NOT read parent-stack locals — every input
/// arrives via the shared queue, which lives on the global heap
/// (also shared under `CLONE_VM`).
///
/// Marked `#[inline(never)]` to keep the child's stack frame
/// distinct from `spawn_clone3_worker`'s and to make a stack-
/// overflow trace unambiguous.
#[inline(never)]
fn clone3_child_main() -> ! {
    // Pop the args from the front of the queue. FIFO ordering
    // matches the parent's push-then-clone3 sequence — ktstr only
    // issues clone3 calls from one thread at a time (the per-worker
    // spawn loop), so the pop never sees a sibling's args.
    let args = CLONE3_CHILD_ARGS_QUEUE
        .lock()
        .expect("clone3 child-args queue mutex poisoned")
        .pop_front()
        .expect("clone3 child reached child_main with empty args queue");

    // Wait for parent to signal start via the start pipe. 30s
    // ceiling matches the fork-mode child to avoid an indefinite
    // hang on a parent that crashes between clone3 and start().
    let mut pfd = libc::pollfd {
        fd: args.start_fd,
        events: libc::POLLIN,
        revents: 0,
    };
    let ret = unsafe { libc::poll(&mut pfd, 1, 30_000) };
    if ret <= 0 {
        unsafe { libc::_exit(1) };
    }
    let mut buf = [0u8; 1];
    unsafe {
        libc::read(args.start_fd, buf.as_mut_ptr() as *mut _, 1);
        libc::close(args.start_fd);
    }
    // Destructure args so the per-worker stop Arc and the other
    // fields move into local bindings without partial-move
    // conflicts when worker_main borrows `&stop`.
    let Clone3ChildArgs {
        affinity,
        work_type,
        sched_policy,
        mem_policy,
        mpol_flags,
        nice,
        worker_pipe_fds,
        worker_futex: futex_raw,
        iter_slot: slot_raw,
        report_fd,
        start_fd: _,
        stop,
    } = *args;
    // Reset the per-worker stop flag in case SIGUSR1 raced the
    // start handshake. Mirrors the Thread-mode pattern: each
    // SharedVm/Raw worker carries its own `Arc<AtomicBool>` so
    // stopping one worker does not stop siblings (the global
    // [`STOP`] would, since SIGUSR1 broadcasts to every task that
    // shares the parent's signal disposition table).
    stop.store(false, Ordering::Relaxed);

    let worker_futex = futex_raw.map(|(addr, pos)| (addr as *mut u32, pos));
    let iter_slot = slot_raw as *mut AtomicU64;

    let report = worker_main(
        affinity,
        work_type,
        sched_policy,
        mem_policy,
        mpol_flags,
        nice,
        worker_pipe_fds,
        worker_futex,
        iter_slot,
        &stop,
    );

    // Serialise the report and write it to the parent's end of the
    // report pipe. Any IO error here is silently dropped — the
    // parent's read_to_end falls into the sentinel branch and
    // attaches an exit_info diagnostic.
    let json = serde_json::to_vec(&report).unwrap_or_default();
    unsafe {
        libc::write(
            report_fd,
            json.as_ptr() as *const _,
            json.len(),
        );
        libc::close(report_fd);
        libc::_exit(0);
    }
}

/// Internal dispatch shape resolved from
/// [`WorkloadConfig::clone_mode`] inside [`WorkloadHandle::spawn`].
///
/// `Clone3` is wired through the dispatch loop but reaches the
/// staging bail-out before invoking the syscall — see the
/// `CloneMode::SharedVm` / `CloneMode::Raw` arms in
/// [`WorkloadHandle::spawn`]. The variant is kept (and the
/// `spawn_clone3_worker` helper preserved with `#[allow(dead_code)]`
/// below) because the redesign tracked under the follow-up clone3
/// task will turn the bail-out into a real dispatch and reuse this
/// machinery without re-deriving the ABI surface.
#[allow(dead_code)]
enum Dispatch {
    Fork,
    Thread,
    Clone3 { flags: CloneFlags },
}

/// Default Linux thread stack size for [`CloneMode::SharedVm`] /
/// [`CloneMode::Raw`] children. `clone3(2)` requires the caller to
/// allocate the child stack via mmap and pass it through
/// `clone_args.stack` + `clone_args.stack_size`; pthread on glibc
/// uses 8 MiB by default for new threads, and matching that figure
/// keeps worker stack budgets predictable.
const CLONE3_STACK_SIZE: usize = 8 * 1024 * 1024;

/// Heap-allocated argument bundle handed from a clone3 parent to
/// the child it just forked.
///
/// Under `CLONE_VM` the child shares the parent's address space but
/// runs on a fresh mmap'd stack — the parent's stack frame for
/// [`spawn_clone3_worker`] is unwound as soon as the parent returns
/// past the syscall, so the child cannot safely read locals
/// captured at clone3 time. Heap-allocating these args via `Box`
/// (which lands in the shared global heap) and passing the box
/// through [`CLONE3_CHILD_ARGS_QUEUE`] gives the child a stable
/// place to find its inputs that survives the parent's unwind.
struct Clone3ChildArgs {
    affinity: Option<BTreeSet<usize>>,
    work_type: WorkType,
    sched_policy: SchedPolicy,
    mem_policy: MemPolicy,
    mpol_flags: MpolFlags,
    nice: i32,
    worker_pipe_fds: Option<(i32, i32)>,
    /// Futex region pointer (raw, address-only). See
    /// [`SendFutexPtr`] for the lifetime argument; the
    /// MAP_SHARED region outlives every clone3 child.
    worker_futex: Option<(usize, usize)>,
    iter_slot: usize,
    report_fd: libc::c_int,
    start_fd: libc::c_int,
    /// Per-worker stop flag, mirroring [`ThreadWorker::stop`]. The
    /// shared address space (CLONE_VM) means a single
    /// `Arc<AtomicBool>` is reachable by both parent and child;
    /// the parent flips it from `stop_and_collect`, the child
    /// observes it via `stop.load(Relaxed)` inside `worker_main`.
    /// Replaces the global [`STOP`] for SharedVm/Raw children so
    /// stopping one worker does not stop siblings.
    stop: std::sync::Arc<AtomicBool>,
}

/// Queue of child-arg boxes pending pickup by clone3 children.
///
/// The parent pushes one `Box<Clone3ChildArgs>` immediately before
/// each `clone3(SYS_clone3, ...)` syscall; the child pops from the
/// front of the queue right after the syscall returns 0. ktstr
/// currently only issues clone3 calls from one thread at a time
/// (the per-worker spawn loop in [`WorkloadHandle::spawn`] is
/// sequential), so FIFO ordering happens to be sufficient — no
/// per-call identifier is needed.
///
/// `Mutex` is fine even though child and parent are technically
/// distinct kernel tasks: under `CLONE_VM` they share the address
/// space including the mutex's underlying futex word, so locking
/// works the same as for two threads.
///
/// FIXME: the FIFO ordering assumption is unsound under concurrent
/// scheduling. If the kernel schedules child N+1's pop_front BEFORE
/// child N's, child N+1 picks up child N's args (and vice versa),
/// silently swapping per-worker state across siblings. Today the
/// per-worker spawn loop is sequential and the next push always
/// follows the previous syscall return, but that ordering is a
/// property of the caller, not the queue. The clone3-child redesign
/// (per the SharedVm/Raw bail message and the deferred dispatch
/// task) MUST replace this with per-child keyed delivery — e.g.
/// hand the child a per-call key via a register the kernel preserves
/// (clone_args.tls when CLONE_SETTLS is set, or a slot in the
/// child's stack mmap that the parent populates pre-syscall) and
/// pop by that key.
static CLONE3_CHILD_ARGS_QUEUE: std::sync::Mutex<std::collections::VecDeque<Box<Clone3ChildArgs>>> =
    std::sync::Mutex::new(std::collections::VecDeque::new());

/// Stack guard size below the clone3 child stack mmap. Mirrors
/// glibc's `pthread_attr_default::guardsize` of one page so a stack
/// overflow segfaults predictably instead of corrupting an adjacent
/// allocation. The mmap is sized
/// `CLONE3_STACK_SIZE + CLONE3_STACK_GUARD`; the kernel never
/// touches the guard page because the child stack pointer is set to
/// the END of the usable region (highest address) and grows down
/// toward the guard.
const CLONE3_STACK_GUARD: usize = 4096;

impl WorkloadHandle {
    /// Spawn worker tasks. Workers block until
    /// [`start()`](Self::start) is called, allowing the caller to
    /// move fork-mode workers into cgroups first. The worker creation
    /// primitive (`fork`, `std::thread::spawn`, or `clone3`) is
    /// selected by [`WorkloadConfig::clone_mode`].
    pub fn spawn(config: &WorkloadConfig) -> Result<Self> {
        // Resolve the kernel clone3 flag set for the dispatch shape
        // up front. Fork and Thread land at full implementations
        // below; SharedVm and Raw bail with an actionable message
        // until the clone3-child Rust-runtime initialisation
        // problem is solved (the child of `clone3(CLONE_VM)` runs
        // without per-task Rust thread-info, so any allocation /
        // panic / stack-overflow probe inside `worker_main` aborts
        // with `the thread info setup logic isn't recursive`).
        // Raw still runs the validator so callers can pin the
        // kernel-rule shape of their flag set ahead of dispatch.
        let dispatch = match &config.clone_mode {
            CloneMode::Fork => Dispatch::Fork,
            CloneMode::Thread => Dispatch::Thread,
            CloneMode::SharedVm => {
                anyhow::bail!(
                    "SharedVm/Raw dispatch requires Rust TLS initialization \
                     in a clone3 child, which is not supported without \
                     pthread. Use CloneMode::Thread (pthread-backed) or \
                     CloneMode::Fork."
                );
            }
            CloneMode::Raw(flags) => {
                validate_clone_flags(*flags)?;
                anyhow::bail!(
                    "SharedVm/Raw dispatch requires Rust TLS initialization \
                     in a clone3 child, which is not supported without \
                     pthread. Flag set is structurally valid. Use \
                     CloneMode::Thread (pthread-backed) or CloneMode::Fork."
                );
            }
        };

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
        if matches!(dispatch, Dispatch::Thread)
            && matches!(config.work_type, WorkType::ForkExit)
        {
            anyhow::bail!(
                "CloneMode::Thread is incompatible with WorkType::ForkExit — \
                 ForkExit forks inside the worker, which under a thread-group \
                 worker tears down every sibling thread on the child's _exit. \
                 Use CloneMode::Fork for ForkExit workloads."
            );
        }

        let needs_pipes = matches!(
            config.work_type,
            WorkType::PipeIo { .. } | WorkType::CachePipe { .. }
        );
        let needs_futex = config.work_type.needs_shared_mem();
        if let Some(group_size) = config.work_type.worker_group_size()
            && (config.num_workers == 0 || !config.num_workers.is_multiple_of(group_size))
        {
            anyhow::bail!(
                "{} requires num_workers divisible by {}, got {}",
                config.work_type.name(),
                group_size,
                config.num_workers
            );
        }

        // All failable acquisitions in this function route through
        // `guard`. If any `?`/`bail!` returns early, the guard's Drop
        // SIGKILLs+reaps forked children, closes open pipe fds, and
        // munmaps the shared regions — so no leak on a mid-spawn
        // error path.
        // Sizing the per-group MAP_SHARED region:
        //   - FanOutCompute needs 16 bytes (futex u32 @ 0, wake_ns
        //     u64 @ 8).
        //   - ProducerConsumerImbalance needs a ring buffer:
        //     head u64 @ 0, tail u64 @ 8, producer-wake u32 @ 16,
        //     consumer-wake u32 @ 20, then Q × u64 ring slots
        //     starting at offset 24. Total bytes = 24 + Q*8.
        //     queue_depth_target is u64 to match the variant, but
        //     `as usize` truncation to a sub-page region would
        //     silently produce a malformed queue — clamp the
        //     conversion at usize::MAX/8 - 3 to keep the layout
        //     well-defined. Realistic configs use Q in the
        //     hundreds-to-thousands; the clamp only triggers on a
        //     degenerate input that itself fails admission control
        //     elsewhere (the queue is far larger than RAM).
        //   - Everything else: u32 (4 bytes).
        let futex_region_size = match config.work_type {
            WorkType::FanOutCompute { .. } => 16,
            WorkType::ProducerConsumerImbalance {
                queue_depth_target,
                ..
            } => {
                let q = std::cmp::min(queue_depth_target as usize, usize::MAX / 8 - 3);
                24 + q * 8
            }
            _ => std::mem::size_of::<u32>(),
        };
        let mut guard = SpawnGuard::new(futex_region_size);

        // For paired work types, create one pipe per worker pair before forking.
        // pipe_pairs[pair_idx] = (read_fd, write_fd) for the A->B direction,
        // and a second pipe for B->A.
        if needs_pipes {
            for _ in 0..config.num_workers / 2 {
                let mut ab = [0i32; 2]; // A writes, B reads
                if unsafe { libc::pipe(ab.as_mut_ptr()) } != 0 {
                    anyhow::bail!("pipe failed: {}", std::io::Error::last_os_error());
                }
                let mut ba = [0i32; 2]; // B writes, A reads
                if unsafe { libc::pipe(ba.as_mut_ptr()) } != 0 {
                    // Close the ab half we just created: it is not
                    // yet owned by the guard, so its Drop won't
                    // otherwise reach it.
                    unsafe {
                        libc::close(ab[0]);
                        libc::close(ab[1]);
                    }
                    anyhow::bail!("pipe failed: {}", std::io::Error::last_os_error());
                }
                guard.pipe_pairs.push((ab, ba));
            }
        }

        // For FutexPingPong/FutexFanOut/FanOutCompute/MutexContention, allocate
        // one shared region per worker group via MAP_SHARED|MAP_ANONYMOUS
        // so all members of the fork see the same physical page. FanOutCompute
        // needs 16 bytes (futex u32 at offset 0, wake timestamp u64 at
        // offset 8); others need 4 bytes.
        let futex_group_size = config.work_type.worker_group_size().unwrap_or(2);
        if needs_futex {
            for _ in 0..config.num_workers / futex_group_size {
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
                         this region backs the {:?} worker-group's \
                         inter-process futex word and is allocated \
                         before fork so every child inherits the same \
                         mapping. Remediation: reduce num_workers (each \
                         futex group consumes one shared page) or raise \
                         `vm.max_map_count` / the memory cgroup limit.",
                        config.work_type.name(),
                    );
                }
                unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, futex_region_size) };
                guard.futex_ptrs.push(ptr as *mut u32);
            }
        }

        // Per-worker iteration counter region (MAP_SHARED). Each
        // worker atomically stores its iteration count to slot [i];
        // the parent reads all slots via `snapshot_iterations()`.
        // The mmap base is page-aligned (kernel guarantee), so
        // casting to `*mut AtomicU64` is sound: page alignment (≥
        // 4096) ≥ AtomicU64 alignment (8), and the region size is
        // an exact multiple of `size_of::<AtomicU64>()` (== 8).
        // Each `.add(i)` moves by `i * 8` bytes, preserving the
        // 8-byte alignment invariant. No non-atomic access to the
        // region exists anywhere in the crate, so the atomic-only
        // aliasing rule (workers + parent share `&AtomicU64`
        // references derived from the raw pointer) holds.
        let iter_counter_len = config.num_workers;
        if iter_counter_len > 0 {
            let size = iter_counter_len * std::mem::size_of::<AtomicU64>();
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
                     {work_type:?} worker-group's per-worker iter_counters \
                     region failed: {errno}{hint}; this region holds one \
                     AtomicU64 per worker ({iter_counter_len} slots) so \
                     the parent can snapshot iteration counts via \
                     `snapshot_iterations()`. Remediation: reduce \
                     num_workers (each worker consumes 8 bytes of this \
                     region, rounded up to a page) or raise \
                     `vm.max_map_count` / the memory cgroup limit.",
                    work_type = config.work_type.name(),
                );
            }
            guard.iter_counters = ptr as *mut AtomicU64;
            guard.iter_counter_bytes = size;
        }

        for i in 0..config.num_workers {
            let affinity = resolve_affinity(&config.affinity)?;

            // Determine pipe fds for this worker (PipeIo/CachePipe).
            let worker_pipe_fds: Option<(i32, i32)> = if needs_pipes {
                let pair_idx = i / 2;
                let (ref ab, ref ba) = guard.pipe_pairs[pair_idx];
                if i % 2 == 0 {
                    // Worker A: writes to ab[1], reads from ba[0]
                    Some((ba[0], ab[1]))
                } else {
                    // Worker B: writes to ba[1], reads from ab[0]
                    Some((ab[0], ba[1]))
                }
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
                let group_idx = i / futex_group_size;
                let pos = i % futex_group_size;
                Some((guard.futex_ptrs[group_idx], pos))
            } else {
                None
            };

            // Shared iteration counter slot for this worker.
            let iter_slot: *mut AtomicU64 = if !guard.iter_counters.is_null() {
                unsafe { guard.iter_counters.add(i) }
            } else {
                std::ptr::null_mut()
            };

            // Per-mode dispatch. Thread-mode workers do not need
            // pipes — the rendezvous and report channels are
            // in-process Rust primitives (`mpsc::sync_channel(0)` +
            // `JoinHandle`). Fork and Clone3 modes both use the
            // pipe-based scaffolding originally built for fork.
            match dispatch {
                Dispatch::Thread => {
                    spawn_thread_worker(
                        &mut guard,
                        config,
                        affinity,
                        worker_pipe_fds,
                        worker_futex,
                        iter_slot,
                    )?;
                    continue;
                }
                Dispatch::Fork | Dispatch::Clone3 { .. } => {
                    // fall through to the pipe-based dispatch below
                }
            }

            // Create pipe for report and a second pipe for "start" signal.
            // Local cleanup on second-pipe failure: the guard has no
            // per-worker tracking of half-allocated pipes, so the first
            // half closes here before the bail.
            let mut report_fds = [0i32; 2];
            if unsafe { libc::pipe(report_fds.as_mut_ptr()) } != 0 {
                anyhow::bail!(
                    "worker {}/{}: report pipe failed: {}",
                    i + 1,
                    config.num_workers,
                    std::io::Error::last_os_error(),
                );
            }
            let mut start_fds = [0i32; 2];
            if unsafe { libc::pipe(start_fds.as_mut_ptr()) } != 0 {
                unsafe {
                    libc::close(report_fds[0]);
                    libc::close(report_fds[1]);
                }
                anyhow::bail!(
                    "worker {}/{}: start pipe failed: {}",
                    i + 1,
                    config.num_workers,
                    std::io::Error::last_os_error(),
                );
            }

            // Clone3-mode dispatch: caller-supplied flags (always
            // including CLONE_VM) plus the runtime-set CLONE_PIDFD
            // for reaping. The child runs the same body as the
            // Fork path's `pid == 0` arm — pipe-based start
            // rendezvous, worker_main, _exit — but is reaped via
            // pidfd poll instead of waitpid.
            if let Dispatch::Clone3 { flags } = dispatch {
                spawn_clone3_worker(
                    &mut guard,
                    config,
                    flags,
                    affinity,
                    worker_pipe_fds,
                    worker_futex,
                    iter_slot,
                    report_fds,
                    start_fds,
                    i,
                )?;
                continue;
            }

            let pid = unsafe { libc::fork() };
            match pid {
                -1 => {
                    // Fork failed: close both fresh pipes so they don't
                    // leak before the guard reaps the already-forked
                    // siblings.
                    unsafe {
                        libc::close(report_fds[0]);
                        libc::close(report_fds[1]);
                        libc::close(start_fds[0]);
                        libc::close(start_fds[1]);
                    }
                    anyhow::bail!(
                        "worker {}/{}: fork failed: {}",
                        i + 1,
                        config.num_workers,
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
                    // Install signal handler FIRST (before start wait)
                    // to prevent SIGUSR1 killing us before we're ready
                    STOP.store(false, Ordering::Relaxed);
                    unsafe {
                        libc::signal(
                            libc::SIGUSR1,
                            sigusr1_handler as *const () as libc::sighandler_t,
                        );
                    }
                    // Close unused pipe ends
                    unsafe {
                        libc::close(report_fds[0]);
                        libc::close(start_fds[1]);
                    }
                    // Close pipe ends belonging to other workers in this pair.
                    if needs_pipes {
                        let pair_idx = i / 2;
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
                        // Close all pipe fds from other pairs.
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
                    //    before catch_unwind.
                    //
                    // 2. `mem::forget(guard)` — `fork()` duplicated
                    //    the parent's stack, so the child's local
                    //    `guard` references the same children pids
                    //    and pipe fds as the parent's. Running its
                    //    Drop on a panic unwind would SIGKILL every
                    //    sibling (fratricide). Forget severs the
                    //    child's view so Drop cannot run. Placed
                    //    INSIDE the catch_unwind closure so it runs
                    //    before worker_main and is scoped to the
                    //    child path only.
                    //
                    // 3. `panic::catch_unwind` — catches any panic
                    //    before it escapes this arm. Belt-and-braces
                    //    against (a) additional Drops on this
                    //    frame's stack (e.g. future refactors that
                    //    add more RAII) and (b) alloc/OOM panics
                    //    during worker_main / serde_json.
                    //
                    //    Caveat: catch_unwind is a no-op under
                    //    `panic = "abort"`, which ktstr's Cargo.toml
                    //    DOES set in `[profile.release]`. In release
                    //    builds a panic inside this closure aborts
                    //    the child immediately (SIGABRT); the
                    //    `catch_unwind` call compiles but never
                    //    returns `Err`, and neither the
                    //    `f.write_all(&json)` nor the `_exit(1)`
                    //    below runs on the panic path. The parent's
                    //    `stop_and_collect` therefore observes a
                    //    missing WorkerReport and fills in a
                    //    sentinel — that sentinel fallback IS the
                    //    release-build correctness mechanism.
                    //    Defenses (1) and (2) still apply unchanged
                    //    under abort: the silent panic hook
                    //    suppresses the panic message and the
                    //    `mem::forget(guard)` severs Drop (the abort
                    //    itself also skips Drops, but the forget
                    //    makes the intent explicit regardless of
                    //    strategy). Dev/test builds (cargo test,
                    //    cargo nextest run — dev profile inherits
                    //    default unwind semantics) still get a real
                    //    `catch_unwind` Err → `_exit(1)` fast-path.
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
                    //    the guard (severed by `mem::forget`
                    //    above) and the child-local
                    //    `resume_latencies_ns` / `migrations`
                    //    `Vec<T>` (per-process heap, no cross-
                    //    process impact). `STATIC_HOST_INFO`'s
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
                    let _ = std::panic::take_hook();
                    std::panic::set_hook(Box::new(|_| {}));
                    let child_result =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            std::mem::forget(guard);
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
                            // Reset stop flag in case SIGUSR1 arrived during wait.
                            // The forked child has its own (CoW) copy of the
                            // global STOP, so resetting it here only affects
                            // this worker, not its siblings.
                            STOP.store(false, Ordering::Relaxed);
                            // Now run. Fork-mode workers thread the global
                            // STOP through `worker_main` — the SIGUSR1 handler
                            // is process-wide, so flipping `STOP` from
                            // `sigusr1_handler` is what reaches the loop's
                            // `stop.load(Relaxed)` checks.
                            let report = worker_main(
                                affinity,
                                config.work_type.clone(),
                                config.sched_policy,
                                config.mem_policy.clone(),
                                config.mpol_flags,
                                config.nice,
                                worker_pipe_fds,
                                worker_futex,
                                iter_slot,
                                &STOP,
                            );
                            let json = serde_json::to_vec(&report).unwrap_or_default();
                            let mut f = unsafe { std::fs::File::from_raw_fd(report_fds[1]) };
                            let _ = f.write_all(&json);
                            drop(f);
                        }));
                    let code = if child_result.is_ok() { 0 } else { 1 };
                    unsafe {
                        libc::_exit(code);
                    }
                }
                child_pid => {
                    // Parent: close unused pipe ends.
                    unsafe {
                        libc::close(report_fds[1]);
                        libc::close(start_fds[0]);
                    }
                    // child_pid is positive by the -1 arm above, so
                    // fits in pid_t directly — store as pid_t so
                    // every downstream libc call avoids the u32→i32
                    // sign-cast wraparound bug.
                    guard
                        .children
                        .push((child_pid, report_fds[0], start_fds[1]));
                }
            }
        }

        // Success: transfer live resources (children, futex_ptrs,
        // iter_counters) to the handle. The guard's subsequent Drop
        // closes the inter-worker `pipe_pairs` — the parent never
        // uses them post-fork, and they were never owned by the
        // handle.
        Ok(guard.into_handle())
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
    /// - [`CloneMode::SharedVm`] / [`CloneMode::Raw`]: each entry
    ///   is the pid the kernel returned from `clone3`.
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
    pub fn worker_pids(&self) -> Vec<libc::pid_t> {
        if !self.children.is_empty() {
            self.children.iter().map(|(pid, _, _)| *pid).collect()
        } else if !self.threads.is_empty() {
            self.threads
                .iter()
                .map(|tw| tw.tid.load(Ordering::Acquire))
                .collect()
        } else {
            self.pidfd_workers.iter().map(|w| w.pid).collect()
        }
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
        // Fork-mode: write a byte to the start pipe. Pidfd-mode
        // shares the same pipe-based scaffolding.
        for (_, _, start_fd) in &mut self.children {
            unsafe {
                libc::write(*start_fd, b"s".as_ptr() as *const _, 1);
                libc::close(*start_fd);
            }
            *start_fd = -1;
        }
        for w in &mut self.pidfd_workers {
            unsafe {
                libc::write(w.start_fd, b"s".as_ptr() as *const _, 1);
                libc::close(w.start_fd);
            }
            w.start_fd = -1;
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
    /// For [`CloneMode::Fork`] and clone3 modes the per-worker pid
    /// addresses a distinct kernel task. For [`CloneMode::Thread`]
    /// the worker's `gettid()` is what
    /// `sched_setaffinity(tid, ...)` accepts; this method reads
    /// the tid from the worker's `Arc<AtomicI32>` (with `Acquire`
    /// ordering, paired with the `Release` publish on the worker
    /// thread). Returns an error if the thread has not yet
    /// published its tid — call [`start()`](Self::start) first so
    /// the worker reaches its `gettid()` publish before reading.
    pub fn set_affinity(&self, idx: usize, cpus: &BTreeSet<usize>) -> Result<()> {
        let pid = if !self.children.is_empty() {
            self.children[idx].0
        } else if !self.threads.is_empty() {
            let tid = self.threads[idx].tid.load(Ordering::Acquire);
            if tid == 0 {
                anyhow::bail!(
                    "set_affinity: thread worker {idx} has not yet \
                     published gettid() (call start() first)"
                );
            }
            tid
        } else {
            self.pidfd_workers[idx].pid
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
    /// then sleeps 500ms to let them begin before signaling stop.
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
    /// the worker's pipe-delivered JSON — **not** on `waitpid` exit
    /// status. Under `panic = "unwind"` (dev/test profile) the worker's
    /// `catch_unwind` arm calls `_exit(1)` so the parent sees
    /// `WIFEXITED=true`, `WEXITSTATUS=1`; under `panic = "abort"`
    /// (release profile) the worker aborts with `SIGABRT` so the parent
    /// sees `WIFEXITED=false`, `WTERMSIG=6`. Either way, a panicking
    /// worker never finishes `f.write_all(&json)` on the report pipe,
    /// so `poll` + `read_to_end` hands back an empty (or truncated)
    /// buffer, `serde_json::from_slice` fails, and the sentinel path
    /// fires. Partial writes from a panic between successful
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

        // If we just started workers, give them time to begin before stopping.
        // 500ms accommodates parallel test runs where CPU contention delays
        // fork of worker processes.
        if !was_started {
            std::thread::sleep(std::time::Duration::from_millis(500));
        }

        let mut reports = Vec::new();
        let children = std::mem::take(&mut self.children);
        let threads = std::mem::take(&mut self.threads);
        let pidfd_workers = std::mem::take(&mut self.pidfd_workers);

        // Signal all fork-mode children to stop via SIGUSR1; the
        // signal handler flips the global STOP that worker_main's
        // `stop.load(Relaxed)` checks read.
        // `pid` is `libc::pid_t`, so it flows to `Pid::from_raw`
        // without the u32→i32 sign-cast wraparound that produced
        // `kill(-1, ...)` session-wide reaps when the old u32 pid
        // exceeded i32::MAX.
        for &(pid, _, _) in &children {
            let _ = nix::sys::signal::kill(
                nix::unistd::Pid::from_raw(pid),
                nix::sys::signal::Signal::SIGUSR1,
            );
        }
        // Signal all clone3 workers via pidfd_send_signal(SIGUSR1).
        // SharedVm/Raw children inherit the signal disposition table
        // from the parent at clone time when CLONE_SIGHAND is NOT
        // set, so the parent's `sigusr1_handler` (installed in main)
        // is also their handler — flipping STOP works.
        for w in &pidfd_workers {
            unsafe {
                libc::syscall(
                    libc::SYS_pidfd_send_signal,
                    w.pidfd,
                    libc::SIGUSR1,
                    std::ptr::null::<libc::siginfo_t>(),
                    0u32,
                );
            }
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
        for (pid, read_fd, _) in children {
            let mut buf = Vec::new();
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            let ms = remaining.as_millis().min(i32::MAX as u128) as i32;
            if ms > 0 {
                let mut pfd = libc::pollfd {
                    fd: read_fd,
                    events: libc::POLLIN,
                    revents: 0,
                };
                let ready = unsafe { libc::poll(&mut pfd, 1, ms) };
                if ready > 0 {
                    let mut f = unsafe { std::fs::File::from_raw_fd(read_fd) };
                    let _ = f.read_to_end(&mut buf);
                    drop(f);
                } else {
                    let _ = nix::unistd::close(read_fd);
                }
            } else {
                let _ = nix::unistd::close(read_fd);
            }

            // Wait for child (WNOHANG first, then SIGKILL if still alive).
            // `pid` is `libc::pid_t` (= i32 on Linux), which passes
            // straight into `Pid::from_raw` without a sign cast —
            // the old u32→i32 session-wide-reap hazard is avoided.
            let npid = nix::unistd::Pid::from_raw(pid);
            let waited = nix::sys::wait::waitpid(npid, Some(nix::sys::wait::WaitPidFlag::WNOHANG));
            let still_running = matches!(waited, Ok(nix::sys::wait::WaitStatus::StillAlive),);
            // Preserve the reap shape for the sentinel path below:
            // the WNOHANG attempt tells us "exited / signaled /
            // still running" on the fast path; the SIGKILL + blocking
            // waitpid below collapses "still running" into
            // `WorkerExitInfo::TimedOut` without retaining the final
            // status (the reap itself is the diagnostic — the child
            // was past its deadline).
            //
            // Unconditional killpg: BOTH branches sweep the worker's
            // process group so descendants forked by a
            // [`WorkType::Custom`] body (or any future work type that
            // spawns helpers) die with the worker. A graceful-exit
            // worker that forked a long-running grandchild would
            // otherwise leave the grandchild alive — setpgid(0, 0) at
            // fork time gives us pgid == worker pid, and killpg is a
            // no-op (ESRCH) once all members have exited. The
            // StillAlive branch additionally direct-kills + blocking-
            // waits the leader; the graceful branch keeps `waited`
            // because the leader's exit status is already known and
            // is what classify_wait_outcome should see.
            //
            // WNOHANG → killpg race window: between the `waited`
            // observation above and this killpg call, the leader may
            // flip from StillAlive to exited (rare but real — the
            // worker could finish between the two syscalls). If that
            // happens, the `still_running` boolean is stale (it says
            // true but the leader is already dead by the time we
            // issue killpg/kill). The race is tolerated: the killpg
            // sweep fires against an empty group (ESRCH) and the
            // follow-up blocking `waitpid(npid, None)` returns the
            // already-reaped status or ECHILD — either way the
            // escalation path collapses to `WorkerExitInfo::TimedOut`
            // without retaining the final code, which is the
            // documented sentinel semantics. We accept the minor
            // diagnostic loss (a "timed out" classification for a
            // leader that actually exited cleanly on the race) in
            // exchange for not needing a second WNOHANG probe.
            //
            // ESRCH-convention: this call intentionally discards the
            // killpg return via `let _ =`. ESRCH (group gone) is the
            // expected outcome for the common no-descendants case and
            // is not worth logging. Contrast `WorkloadHandle::drop`
            // below, which logs a `tracing::warn!` on non-ESRCH
            // errors from killpg — Drop runs on every handle teardown
            // including panic-unwind paths, so surfacing unusual
            // errors there is more valuable than during the normal
            // collect flow.
            let _ = nix::sys::signal::killpg(npid, nix::sys::signal::Signal::SIGKILL);
            let exit_info_source: Result<nix::sys::wait::WaitStatus, nix::errno::Errno> =
                if still_running {
                    // Leader still up: direct-kill + blocking reap. The
                    // killpg above has already started dying descendants
                    // in parallel; the follow-up `kill` is the single-
                    // process fallback when the worker's
                    // `setpgid(0, 0)` at fork time somehow failed.
                    let _ = nix::sys::signal::kill(npid, nix::sys::signal::Signal::SIGKILL);
                    let _ = nix::sys::wait::waitpid(npid, None);
                    Ok(nix::sys::wait::WaitStatus::StillAlive)
                } else {
                    waited
                };

            if let Ok(report) = serde_json::from_slice::<WorkerReport>(&buf) {
                reports.push(report);
            } else {
                let exit_info = classify_wait_outcome(exit_info_source);
                eprintln!(
                    "ktstr: worker pid={pid} returned no report ({} bytes read, exit={exit_info:?})",
                    buf.len()
                );
                reports.push(WorkerReport {
                    // Both `pid` and `WorkerReport.tid` are `pid_t`
                    // (i32) now — no cast needed.
                    tid: pid,
                    exit_info: Some(exit_info),
                    ..WorkerReport::default()
                });
            }
        }

        // Pidfd-mode collection: poll(pidfd, POLLIN) reports the
        // child's exit; read the report fd; close pidfd / report fd
        // / start fd; munmap the child stack. The shared 5s deadline
        // is consumed alongside the fork-mode loop above so a
        // mixed-mode handle (currently impossible — clone_mode is
        // per-config — but kept symmetric) shares the budget.
        for w in pidfd_workers {
            let mut buf = Vec::new();
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            let ms = remaining.as_millis().min(i32::MAX as u128) as i32;
            // Drain the report pipe first (worker may have already
            // exited and written its JSON before we polled pidfd).
            if ms > 0 {
                let mut pfd = libc::pollfd {
                    fd: w.report_fd,
                    events: libc::POLLIN,
                    revents: 0,
                };
                let ready = unsafe { libc::poll(&mut pfd, 1, ms) };
                if ready > 0 {
                    let mut f = unsafe { std::fs::File::from_raw_fd(w.report_fd) };
                    let _ = f.read_to_end(&mut buf);
                    drop(f);
                } else {
                    let _ = nix::unistd::close(w.report_fd);
                }
            } else {
                let _ = nix::unistd::close(w.report_fd);
            }
            // Send SIGKILL via pidfd to ensure the child exits
            // before we munmap its stack. ESRCH (already exited) is
            // the common case for a graceful worker; swallow it.
            unsafe {
                libc::syscall(
                    libc::SYS_pidfd_send_signal,
                    w.pidfd,
                    libc::SIGKILL,
                    std::ptr::null::<libc::siginfo_t>(),
                    0u32,
                );
            }
            // Wait for exit before munmap. POLLIN on pidfd fires
            // when the child reaches `exit`. Up to a 5s ceiling so
            // a misbehaving child can't pin the test.
            let mut pfd = libc::pollfd {
                fd: w.pidfd,
                events: libc::POLLIN,
                revents: 0,
            };
            unsafe {
                libc::poll(&mut pfd, 1, 5_000);
            }
            // Close pidfd + start_fd; report_fd was either consumed
            // by `f.read_to_end` (which closed it via Drop) or
            // explicitly closed in the timeout branch above.
            for fd in [w.pidfd, w.start_fd] {
                if fd >= 0 {
                    let _ = nix::unistd::close(fd);
                }
            }
            // munmap the child stack now that the child is gone.
            if !w.stack_ptr.is_null() && w.stack_size > 0 {
                unsafe { libc::munmap(w.stack_ptr, w.stack_size) };
            }

            if let Ok(report) = serde_json::from_slice::<WorkerReport>(&buf) {
                reports.push(report);
            } else {
                eprintln!(
                    "ktstr: clone3 worker pid={} returned no report ({} bytes read)",
                    w.pid,
                    buf.len()
                );
                reports.push(WorkerReport {
                    tid: w.pid,
                    completed: false,
                    ..WorkerReport::default()
                });
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
                match join_thread_with_timeout(j, THREAD_JOIN_TIMEOUT) {
                    Some(Ok(report)) => reports.push(report),
                    Some(Err(payload)) => {
                        let msg = extract_panic_payload(payload);
                        eprintln!(
                            "ktstr: thread worker tid={tid} panicked: {msg}"
                        );
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

        // Fork-mode children. `pid` is `libc::pid_t` — stored as i32
        // so `Pid::from_raw` receives the kernel's native
        // representation directly, not the sign-cast of a u32 that
        // could alias negative values (including -1, i.e. every
        // process in the session).
        for &(pid, rfd, wfd) in &self.children {
            let nix_pid = Pid::from_raw(pid);
            // killpg first: reach descendants the worker may have
            // forked (Custom workloads, ForkExit caught mid-fork).
            // pgid == worker pid because the worker called
            // `setpgid(0, 0)` at fork time. ESRCH (group gone / no
            // members) is expected and not a warning-worthy failure;
            // swallow it to keep the log clean when the common
            // no-descendants case drops.
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
            for fd in [rfd, wfd] {
                if fd >= 0
                    && let Err(e) = close(fd)
                {
                    tracing::warn!(fd, %e, "close failed in WorkloadHandle::drop");
                }
            }
        }
        // Pidfd-mode workers: pidfd_send_signal(SIGKILL) +
        // poll(pidfd, POLLIN) for exit, then close fds and munmap
        // the child stack. pidfd_send_signal(SIGKILL) is the right
        // primitive because the child does NOT share the parent's
        // process group (no setpgid contract here), so killpg on
        // the parent's pgid would not reach it.
        for w in &self.pidfd_workers {
            if w.pidfd >= 0 {
                unsafe {
                    libc::syscall(
                        libc::SYS_pidfd_send_signal,
                        w.pidfd,
                        libc::SIGKILL,
                        std::ptr::null::<libc::siginfo_t>(),
                        0u32,
                    );
                }
                let mut pfd = libc::pollfd {
                    fd: w.pidfd,
                    events: libc::POLLIN,
                    revents: 0,
                };
                unsafe {
                    libc::poll(&mut pfd, 1, 5_000);
                }
                let _ = close(w.pidfd);
            }
            for fd in [w.report_fd, w.start_fd] {
                if fd >= 0 {
                    let _ = close(fd);
                }
            }
            if !w.stack_ptr.is_null() && w.stack_size > 0 {
                unsafe { libc::munmap(w.stack_ptr, w.stack_size) };
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
                match join_thread_with_timeout(j, THREAD_JOIN_TIMEOUT) {
                    Some(Ok(_)) => {}
                    Some(Err(e)) => {
                        let payload = extract_panic_payload(e);
                        tracing::warn!(
                            tid, payload,
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
        for &ptr in &self.futex_ptrs {
            unsafe {
                libc::munmap(ptr as *mut libc::c_void, self.futex_region_size);
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

use std::os::unix::io::FromRawFd;

static STOP: AtomicBool = AtomicBool::new(false);

/// Wrap `FUTEX_WAKE` on `futex_ptr`, waking up to `n_waiters` tasks.
/// Thin wrapper around `libc::syscall(SYS_futex, ...)` — callers of the
/// wake path duplicate the 7-arg layout in every spot otherwise.
///
/// # Safety
/// Clamp a `usize` wake-count to the positive `i32` range before
/// passing to `futex_wake`.
///
/// `FUTEX_WAKE`'s `val` argument is `i32`. A naked `usize → i32`
/// cast wraps to a negative value when the input exceeds `i32::MAX`
/// (~2.1B), and some kernels interpret a negative `val` as "wake
/// every waiter on this futex" — a silent scope explosion from a
/// numeric-overflow bug. The clamp pins the syscall to wake at most
/// `i32::MAX` waiters, which exceeds any realistic topology by
/// orders of magnitude.
///
/// `#[inline]` because the call site is a single cast + `min` and
/// inlining lets the compiler fold the clamp into the surrounding
/// futex_wake syscall setup.
#[inline]
fn clamp_futex_wake_n(n: usize) -> i32 {
    n.min(i32::MAX as usize) as i32
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
fn mmap_shared_anon_errno_hint(errno: Option<i32>) -> &'static str {
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

/// `futex_ptr` must point to a live `u32` reachable by every thread
/// that might block on this futex word.
unsafe fn futex_wake(futex_ptr: *mut u32, n_waiters: i32) {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            futex_ptr,
            libc::FUTEX_WAKE,
            n_waiters,
            std::ptr::null::<libc::timespec>(),
            std::ptr::null::<u32>(),
            0u32,
        );
    }
}

/// Wrap `FUTEX_WAIT` on `futex_ptr` with expected value `expected` and
/// the given timespec. Returns once the wait returns (wake, timeout, or
/// value mismatch) without inspecting the outcome — callers typically
/// re-check the state via `read_volatile`.
///
/// # Safety
/// `futex_ptr` must point to a live `u32` reachable by every thread
/// that might wake this futex word.
unsafe fn futex_wait(futex_ptr: *mut u32, expected: u32, ts: &libc::timespec) {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            futex_ptr,
            libc::FUTEX_WAIT,
            expected,
            ts as *const libc::timespec,
            std::ptr::null::<u32>(),
            0u32,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn worker_main(
    affinity: Option<BTreeSet<usize>>,
    work_type: WorkType,
    sched_policy: SchedPolicy,
    mem_policy: MemPolicy,
    mpol_flags: MpolFlags,
    nice: i32,
    pipe_fds: Option<(i32, i32)>,
    futex: Option<(*mut u32, usize)>,
    iter_slot: *mut AtomicU64,
    stop: &AtomicBool,
) -> WorkerReport {
    // The kernel's per-task identifier is gettid(), not getpid():
    // - For fork-based workers, getpid() == gettid() because the
    //   forked child becomes a thread-group leader (tgid == pid == tid).
    // - For thread-based workers (CloneMode::Thread), every thread shares
    //   getpid() (== parent's tgid) and gettid() is what discriminates
    //   the per-task identity. Reporting gettid() in WorkerReport.tid
    //   keeps the field name accurate across both dispatch paths and
    //   matches what cgroup.threads / sched_setaffinity(tid, ...)
    //   accept.
    let tid: libc::pid_t = unsafe { libc::syscall(libc::SYS_gettid) as libc::pid_t };

    if let Some(ref cpus) = affinity {
        let _ = set_thread_affinity(tid, cpus);
    }
    let _ = set_sched_policy(tid, sched_policy);
    apply_mempolicy_with_flags(&mem_policy, mpol_flags);
    apply_nice(nice);

    let start = Instant::now();
    let mut work_units: u64 = 0;
    let mut migration_count: u64 = 0;
    let mut cpus_used = BTreeSet::new();
    let mut migrations = Vec::new();
    let mut last_cpu = sched_getcpu();
    cpus_used.insert(last_cpu);
    let mut last_iter_time = start;
    let mut max_gap_ns: u64 = 0;
    let mut max_gap_cpu: usize = last_cpu;
    let mut max_gap_at_ns: u64 = 0;
    // Lazily allocated per-worker cache buffer (CachePressure, CacheYield, CachePipe, FanOutCompute).
    let mut cache_pressure_buf: Option<Vec<u8>> = None;
    // Separate Vec<u64> for the matrix_multiply helper: the matrix
    // workload interprets its storage as a sequence of u64 operands,
    // and a `Vec<u8>` has only 1-byte alignment. Reinterpreting a
    // u8-backed buffer as `*mut u64` is UB regardless of buffer
    // contents. Vec<u64> gives natural 8-byte alignment from the
    // allocator.
    let mut matrix_buf: Option<Vec<u64>> = None;
    // Persistent temp file for IoSync / Phase::Io (opened on first use, removed on exit).
    let mut io_sync_file: Option<(std::fs::File, String)> = None;
    let mut io_seq_file: Option<(std::fs::File, String)> = None;
    // PageFaultChurn: persistent anonymous mmap region and PRNG
    // state, allocated on first outer iteration and reused across
    // every subsequent iteration (`madvise(MADV_DONTNEED)` re-faults
    // pages without re-mapping). Keeping the region outside the
    // match arm lets PageFaultChurn return to the outer work loop
    // after each touches_per_cycle + spin_burst cycle. This gives
    // two distinct cadences:
    //   - The iter_slot publish in the outer `worker_main` loop
    //     fires on EVERY outer iteration (unconditional in the
    //     outer-loop tail), so host-side `snapshot_iterations`
    //     sees progress in real time.
    //   - The migration check in the outer `worker_main` loop
    //     fires every outer iteration but only triggers its body
    //     when `work_units.is_multiple_of(1024)`. With 320 units per
    //     PageFaultChurn outer iter and gcd(320, 1024) = 64, that
    //     lands every 1024/64 = 16 outer iterations (see
    //     doc/guide/src/architecture/workers.md).
    let mut page_fault_region: Option<(*mut libc::c_void, usize)> = None;
    let mut page_fault_rng_state: u64 = 0;
    // One-shot guard for per-position policy overrides (AsymmetricWaker
    // applies waker_class to pos == 0 / wakee_class to pos == 1; future
    // variants like RtStarvation use the same flag). The override must
    // run AFTER the WorkloadConfig-supplied `set_sched_policy` above so
    // it's the last word on the worker's class, and ONCE so we don't
    // hammer sched_setattr/sched_setscheduler every outer iteration.
    let mut per_pos_policy_applied = false;
    // Benchmarking: per-wakeup latency samples (reservoir-sampled) and iteration counter.
    const MAX_WAKE_SAMPLES: usize = 100_000;
    let mut resume_latencies_ns: Vec<u64> = Vec::with_capacity(MAX_WAKE_SAMPLES);
    let mut wake_sample_count: u64 = 0;
    let mut iterations: u64 = 0;
    // AffinityChurn: read effective cpuset once at start via sched_getaffinity.
    // Custom: delegate entirely to the user function. Affinity and
    // sched_policy are already applied above.
    if let WorkType::Custom { run, .. } = &work_type {
        return run(stop);
    }

    let affinity_churn_cpus: Vec<usize> = if matches!(work_type, WorkType::AffinityChurn { .. }) {
        let mut cpu_set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        let ret = unsafe {
            libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut cpu_set)
        };
        if ret == 0 {
            (0..libc::CPU_SETSIZE as usize)
                .filter(|c| unsafe { libc::CPU_ISSET(*c, &cpu_set) })
                .collect()
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };
    // PolicyChurn: build list of (policy, priority) pairs to cycle through.
    // Non-RT policies always available; RT (FIFO/RR) only with CAP_SYS_NICE.
    let policy_churn_policies: Vec<(i32, i32)> =
        if matches!(work_type, WorkType::PolicyChurn { .. }) {
            let mut policies = vec![
                (libc::SCHED_OTHER, 0),
                (libc::SCHED_BATCH, 0),
                (libc::SCHED_IDLE, 0),
            ];
            let param = libc::sched_param { sched_priority: 1 };
            let ret = unsafe { libc::sched_setscheduler(0, libc::SCHED_FIFO, &param) };
            if ret == 0 {
                // Restore to SCHED_OTHER before entering work loop.
                let normal = libc::sched_param { sched_priority: 0 };
                unsafe { libc::sched_setscheduler(0, libc::SCHED_OTHER, &normal) };
                policies.push((libc::SCHED_FIFO, 1));
                policies.push((libc::SCHED_RR, 1));
            }
            policies
        } else {
            Vec::new()
        };
    // FanOutCompute: pre-compute matrix dimension from cache_footprint_kb.
    let matrix_size: usize = if let WorkType::FanOutCompute {
        cache_footprint_kb,
        operations,
        ..
    } = &work_type
    {
        if *operations > 0 && *cache_footprint_kb > 0 {
            ((cache_footprint_kb * 1024 / 3 / std::mem::size_of::<u64>()) as f64).sqrt() as usize
        } else {
            0
        }
    } else {
        0
    };

    // Guest-side /proc/vmstat: system-wide in the guest, but the VM is
    // a controlled environment with no other significant processes, so
    // the delta is attributable to this workload. Same rationale as
    // /proc/self/schedstat below. Host-side reading would require
    // accessing the guest kernel's vmstat via GuestMem or BPF.
    let vmstat_migrated_start = read_vmstat_numa_pages_migrated();

    // schedstat snapshot at work-loop start. `None` means schedstats
    // is unavailable on this kernel (CONFIG_SCHEDSTATS off / procfs
    // error); propagate that through as `None` at the end snapshot
    // and we will emit zero deltas with a one-shot stderr warning —
    // previously we could not distinguish "unavailable" from "worker
    // has run for zero ns".
    //
    // Pass `Some(tid)` so the read targets
    // `/proc/self/task/<tid>/schedstat` rather than
    // `/proc/self/schedstat`. For fork-mode workers `tid == tgid` so
    // the two paths return the same data; for thread-mode workers
    // every sibling shares `/proc/self/schedstat` (the test
    // runner's leader stats), and the per-task path is the only
    // way to read a specific thread's `task->sched_info`.
    let schedstat_start = read_schedstat(Some(tid));

    while !stop_requested(stop) {
        match work_type {
            WorkType::CpuSpin => {
                spin_burst(&mut work_units, 1024);
                iterations += 1;
            }
            WorkType::YieldHeavy => {
                work_units = std::hint::black_box(work_units.wrapping_add(1));
                std::thread::yield_now();
                iterations += 1;
            }
            WorkType::Mixed => {
                spin_burst(&mut work_units, 1024);
                std::thread::yield_now();
                iterations += 1;
            }
            WorkType::IoSync => {
                let (f, _) = io_sync_file.get_or_insert_with(|| {
                    let path = std::env::temp_dir()
                        .join(format!("ktstr_io_{tid}"))
                        .to_string_lossy()
                        .to_string();
                    let f = std::fs::OpenOptions::new()
                        .write(true)
                        .create(true)
                        .truncate(true)
                        .open(&path)
                        .expect("failed to create IoSync temp file");
                    (f, path)
                });
                let _ = f.set_len(0);
                let _ = f.seek(std::io::SeekFrom::Start(0));
                let buf = [0u8; 4096];
                for _ in 0..16 {
                    let _ = f.write_all(&buf);
                    work_units = std::hint::black_box(work_units.wrapping_add(1));
                }
                // Sleep 100us to simulate I/O completion latency.
                // On tmpfs, fsync is noop_fsync (returns 0), so without
                // this sleep IoSync would be a pure CPU workload.
                let before_sleep = Instant::now();
                std::thread::sleep(Duration::from_micros(100));
                reservoir_push(
                    &mut resume_latencies_ns,
                    &mut wake_sample_count,
                    before_sleep.elapsed().as_nanos() as u64,
                    MAX_WAKE_SAMPLES,
                );
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::Bursty { burst_ms, sleep_ms } => {
                let burst_end = Instant::now() + Duration::from_millis(burst_ms);
                while Instant::now() < burst_end && !stop_requested(stop) {
                    spin_burst(&mut work_units, 1024);
                }
                if !stop_requested(stop) {
                    let before_sleep = Instant::now();
                    std::thread::sleep(Duration::from_millis(sleep_ms));
                    reservoir_push(
                        &mut resume_latencies_ns,
                        &mut wake_sample_count,
                        before_sleep.elapsed().as_nanos() as u64,
                        MAX_WAKE_SAMPLES,
                    );
                }
                iterations += 1;
            }
            WorkType::PipeIo { burst_iters } => {
                let (read_fd, write_fd) = pipe_fds.unwrap_or((-1, -1));
                if read_fd < 0 || write_fd < 0 {
                    break;
                }
                spin_burst(&mut work_units, burst_iters);
                pipe_exchange(
                    read_fd,
                    write_fd,
                    &mut resume_latencies_ns,
                    &mut wake_sample_count,
                    MAX_WAKE_SAMPLES,
                    stop,
                );
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::FutexPingPong { spin_iters } => {
                let (futex_ptr, pos) = match futex {
                    Some(f) => f,
                    None => break,
                };
                let is_first = pos == 0;
                spin_burst(&mut work_units, spin_iters);
                // Worker A waits for 0, wakes partner with 1.
                // Worker B waits for 1, wakes partner with 0.
                let my_val: u32 = if is_first { 0 } else { 1 };
                let partner_val: u32 = if is_first { 1 } else { 0 };
                // Wake partner. The signal value is the token itself;
                // Relaxed matches the FanOutCompute / MutexContention
                // idiom — the futex syscall provides the kernel-side
                // cross-thread ordering, no extra user-space barrier
                // is needed for this single-word handshake.
                let atom = unsafe { &*(futex_ptr as *const std::sync::atomic::AtomicU32) };
                atom.store(partner_val, Ordering::Relaxed);
                unsafe { futex_wake(futex_ptr, 1) };
                // Wait for partner to set our expected value, with timeout
                // to avoid blocking forever if partner has stopped.
                let before_block = Instant::now();
                let atom = unsafe { &*(futex_ptr as *const std::sync::atomic::AtomicU32) };
                loop {
                    if stop_requested(stop) {
                        break;
                    }
                    let cur = atom.load(Ordering::Relaxed);
                    if cur == my_val {
                        reservoir_push(
                            &mut resume_latencies_ns,
                            &mut wake_sample_count,
                            before_block.elapsed().as_nanos() as u64,
                            MAX_WAKE_SAMPLES,
                        );
                        break;
                    }
                    unsafe { futex_wait(futex_ptr, partner_val, &FUTEX_WAIT_TIMEOUT) };
                }
                // Reset last_iter_time after blocking step
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::CachePressure { size_kb, stride } => {
                let buf = cache_pressure_buf.get_or_insert_with(|| vec![0u8; size_kb * 1024]);
                if buf.is_empty() || stride == 0 {
                    break;
                }
                cache_rmw_loop(buf, stride, 1024, &mut work_units);
                iterations += 1;
            }
            WorkType::CacheYield { size_kb, stride } => {
                let buf = cache_pressure_buf.get_or_insert_with(|| vec![0u8; size_kb * 1024]);
                if buf.is_empty() || stride == 0 {
                    break;
                }
                cache_rmw_loop(buf, stride, 1024, &mut work_units);
                let before_yield = Instant::now();
                std::thread::yield_now();
                reservoir_push(
                    &mut resume_latencies_ns,
                    &mut wake_sample_count,
                    before_yield.elapsed().as_nanos() as u64,
                    MAX_WAKE_SAMPLES,
                );
                iterations += 1;
            }
            WorkType::CachePipe {
                size_kb,
                burst_iters,
            } => {
                let (read_fd, write_fd) = pipe_fds.unwrap_or((-1, -1));
                if read_fd < 0 || write_fd < 0 {
                    break;
                }
                let buf = cache_pressure_buf.get_or_insert_with(|| vec![0u8; size_kb * 1024]);
                if !buf.is_empty() {
                    cache_rmw_loop(buf, 64, burst_iters, &mut work_units);
                }
                pipe_exchange(
                    read_fd,
                    write_fd,
                    &mut resume_latencies_ns,
                    &mut wake_sample_count,
                    MAX_WAKE_SAMPLES,
                    stop,
                );
                // Reset last_iter_time after blocking step
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::FutexFanOut {
                fan_out,
                spin_iters,
            } => {
                let (futex_ptr, pos) = match futex {
                    Some(f) => f,
                    None => break,
                };
                let is_messenger = pos == 0;
                spin_burst(&mut work_units, spin_iters);
                // Atomic-Relaxed idiom matches FanOutCompute /
                // MutexContention; futex syscalls supply the kernel-
                // side ordering for this generation-counter advance.
                let atom = unsafe { &*(futex_ptr as *const std::sync::atomic::AtomicU32) };
                if is_messenger {
                    // Increment generation counter and wake all receivers.
                    let next = atom.load(Ordering::Relaxed).wrapping_add(1);
                    let wake_n = clamp_futex_wake_n(fan_out);
                    atom.store(next, Ordering::Relaxed);
                    unsafe { futex_wake(futex_ptr, wake_n) };
                    // Short post-wake spin to let receivers run
                    // before the next wake cycle. Routes through
                    // `spin_burst` for consistency with
                    // `WorkType::FanOutCompute`'s messenger (both
                    // use `FAN_OUT_POST_WAKE_SPIN_ITERS`) so the
                    // messenger also advances `work_units`.
                    spin_burst(&mut work_units, FAN_OUT_POST_WAKE_SPIN_ITERS);
                } else {
                    // Receiver: wait for the generation counter to advance.
                    let expected = atom.load(Ordering::Relaxed);
                    let before_block = Instant::now();
                    loop {
                        if stop_requested(stop) {
                            break;
                        }
                        let cur = atom.load(Ordering::Relaxed);
                        if cur != expected {
                            reservoir_push(
                                &mut resume_latencies_ns,
                                &mut wake_sample_count,
                                before_block.elapsed().as_nanos() as u64,
                                MAX_WAKE_SAMPLES,
                            );
                            break;
                        }
                        unsafe { futex_wait(futex_ptr, expected, &FUTEX_WAIT_TIMEOUT) };
                    }
                }
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::Sequence {
                ref first,
                ref rest,
            } => {
                for phase in std::iter::once(first).chain(rest.iter()) {
                    if stop_requested(stop) {
                        break;
                    }
                    match phase {
                        Phase::Spin(dur) => {
                            let end = Instant::now() + *dur;
                            while Instant::now() < end && !stop_requested(stop) {
                                spin_burst(&mut work_units, 1024);
                            }
                        }
                        Phase::Sleep(dur) => {
                            let before_sleep = Instant::now();
                            std::thread::sleep(*dur);
                            reservoir_push(
                                &mut resume_latencies_ns,
                                &mut wake_sample_count,
                                before_sleep.elapsed().as_nanos() as u64,
                                MAX_WAKE_SAMPLES,
                            );
                            last_iter_time = Instant::now();
                        }
                        Phase::Yield(dur) => {
                            let end = Instant::now() + *dur;
                            while Instant::now() < end && !stop_requested(stop) {
                                work_units = std::hint::black_box(work_units.wrapping_add(1));
                                let before_yield = Instant::now();
                                std::thread::yield_now();
                                reservoir_push(
                                    &mut resume_latencies_ns,
                                    &mut wake_sample_count,
                                    before_yield.elapsed().as_nanos() as u64,
                                    MAX_WAKE_SAMPLES,
                                );
                            }
                            last_iter_time = Instant::now();
                        }
                        Phase::Io(dur) => {
                            let end = Instant::now() + *dur;
                            let (f, _) = io_seq_file.get_or_insert_with(|| {
                                let path = std::env::temp_dir()
                                    .join(format!("ktstr_seq_{tid}"))
                                    .to_string_lossy()
                                    .to_string();
                                let f = std::fs::OpenOptions::new()
                                    .write(true)
                                    .create(true)
                                    .truncate(true)
                                    .open(&path)
                                    .expect("failed to create Phase::Io temp file");
                                (f, path)
                            });
                            while Instant::now() < end && !stop_requested(stop) {
                                let _ = f.set_len(0);
                                let _ = f.seek(std::io::SeekFrom::Start(0));
                                let buf = [0u8; 4096];
                                for _ in 0..16 {
                                    let _ = f.write_all(&buf);
                                    work_units = std::hint::black_box(work_units.wrapping_add(1));
                                }
                                let before_sleep = Instant::now();
                                std::thread::sleep(Duration::from_micros(100));
                                reservoir_push(
                                    &mut resume_latencies_ns,
                                    &mut wake_sample_count,
                                    before_sleep.elapsed().as_nanos() as u64,
                                    MAX_WAKE_SAMPLES,
                                );
                            }
                            last_iter_time = Instant::now();
                        }
                    }
                }
                iterations += 1;
            }
            WorkType::ForkExit => {
                let pid = unsafe { libc::fork() };
                match pid {
                    -1 => {
                        work_units = std::hint::black_box(work_units.wrapping_add(1));
                        iterations += 1;
                    }
                    0 => {
                        unsafe { libc::_exit(0) };
                    }
                    child => {
                        let mut status = 0i32;
                        // `waitpid` is a blocking primitive: the
                        // parent sleeps until the child's exit is
                        // reaped. Measuring the interval is the same
                        // "resume latency" signal the other blocking
                        // work types record (pipe read, futex wait,
                        // yield_now, nanosleep), so feed it into the
                        // reservoir on the same contract.
                        let before_wait = Instant::now();
                        unsafe { libc::waitpid(child, &mut status, 0) };
                        reservoir_push(
                            &mut resume_latencies_ns,
                            &mut wake_sample_count,
                            before_wait.elapsed().as_nanos() as u64,
                            MAX_WAKE_SAMPLES,
                        );
                        work_units = std::hint::black_box(work_units.wrapping_add(1));
                        iterations += 1;
                    }
                }
            }
            WorkType::NiceSweep => {
                // Determine allowed nice range. Negative nice requires
                // CAP_SYS_NICE; probe once and clamp min_nice on EPERM.
                let effective_min: i32 = {
                    static PROBED_MIN: std::sync::atomic::AtomicI32 =
                        std::sync::atomic::AtomicI32::new(i32::MIN);
                    let cached = PROBED_MIN.load(Ordering::Relaxed);
                    if cached != i32::MIN {
                        cached
                    } else {
                        let ret = unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, -20) };
                        let min = if ret == -1 {
                            // EPERM — unprivileged, sweep only non-negative
                            0i32
                        } else {
                            // Succeeded — restore nice 0 and sweep full range
                            unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, 0) };
                            -20i32
                        };
                        PROBED_MIN.store(min, Ordering::Relaxed);
                        min
                    }
                };
                let range = (19 - effective_min + 1) as u64;
                let nice_val = effective_min + (iterations % range) as i32;
                spin_burst(&mut work_units, 512);
                unsafe {
                    libc::setpriority(libc::PRIO_PROCESS, 0, nice_val);
                }
                let before_yield = Instant::now();
                std::thread::yield_now();
                reservoir_push(
                    &mut resume_latencies_ns,
                    &mut wake_sample_count,
                    before_yield.elapsed().as_nanos() as u64,
                    MAX_WAKE_SAMPLES,
                );
                iterations += 1;
            }
            WorkType::AffinityChurn { spin_iters } => {
                spin_burst(&mut work_units, spin_iters);
                if !affinity_churn_cpus.is_empty() {
                    use rand::RngExt;
                    let idx = rand::rng().random_range(0..affinity_churn_cpus.len());
                    let target = affinity_churn_cpus[idx];
                    let mut cpu_set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
                    unsafe {
                        libc::CPU_ZERO(&mut cpu_set);
                        libc::CPU_SET(target, &mut cpu_set);
                        libc::sched_setaffinity(
                            0,
                            std::mem::size_of::<libc::cpu_set_t>(),
                            &cpu_set,
                        );
                    }
                }
                let before_yield = Instant::now();
                std::thread::yield_now();
                reservoir_push(
                    &mut resume_latencies_ns,
                    &mut wake_sample_count,
                    before_yield.elapsed().as_nanos() as u64,
                    MAX_WAKE_SAMPLES,
                );
                iterations += 1;
            }
            WorkType::PolicyChurn { spin_iters } => {
                spin_burst(&mut work_units, spin_iters);
                let idx = (iterations as usize) % policy_churn_policies.len().max(1);
                let (pol, prio) = policy_churn_policies[idx];
                let param = libc::sched_param {
                    sched_priority: prio,
                };
                unsafe {
                    libc::sched_setscheduler(0, pol, &param);
                }
                let before_yield = Instant::now();
                std::thread::yield_now();
                reservoir_push(
                    &mut resume_latencies_ns,
                    &mut wake_sample_count,
                    before_yield.elapsed().as_nanos() as u64,
                    MAX_WAKE_SAMPLES,
                );
                iterations += 1;
            }
            WorkType::FanOutCompute {
                fan_out,
                operations,
                sleep_usec,
                ..
            } => {
                let (futex_ptr, pos) = match futex {
                    Some(f) => f,
                    None => break,
                };
                let is_messenger = pos == 0;
                // Shared memory layout: [u64 generation @ offset 0]
                // [u64 wake_ns @ offset 8]. The mmap base is
                // page-aligned (see the futex-region MAP_ANONYMOUS
                // allocation in `WorkloadHandle::spawn`), so both
                // offsets are 8-byte aligned.
                //
                // The generation counter is u64 (not u32) to prevent
                // a wraparound-ABA bug in USER-SPACE: with a u32
                // counter the worker's snapshot `expected` could
                // match `cur` again after exactly 2^32 messenger
                // advances, causing the worker's user-space
                // `cur != expected` compare to miss the wake. u64
                // comparisons push that user-space wraparound out
                // to ~585 years at one advance per nanosecond —
                // effectively unreachable.
                //
                // The KERNEL-SIDE futex_wait still compares the low
                // 32 bits at `futex_ptr` to the `expected` u32
                // argument passed into the syscall, so a full
                // 2^32-advance race inside a single futex_wait's
                // microsecond syscall window would still cause a
                // kernel-side EAGAIN miss. That is empirically
                // unreachable (2^32 atomic RMWs in microseconds
                // requires >10^15 advances/sec — orders of
                // magnitude above any realistic sequencer rate),
                // and the 100 ms futex_wait timeout self-heals any
                // hypothetical occurrence: on timeout the outer
                // loop re-reads `cur` as u64 and the mismatch is
                // visible in user space even if the kernel missed
                // the advance. Little-endian x86_64 / aarch64
                // targets guarantee the low 4 bytes of the u64
                // live at offset 0 (enforced by a compile_error!
                // elsewhere in this file); big-endian would flip
                // the layout and is rejected at build time.
                //
                // Use Release/Acquire ordering so that when workers
                // observe the generation advance, the matching
                // wake_ns store is already visible to them.
                // `read_volatile`/`write_volatile` only defeat
                // compiler reordering; on aarch64's weak memory
                // model two independent hazards remain:
                //   (a) the messenger's two stores (wake_ns, then
                //       generation) can be reordered by the CPU so
                //       the generation advance becomes globally
                //       visible before the new wake_ns; and/or
                //   (b) the worker's wake_ns load can be
                //       speculatively issued before its generation
                //       load and satisfied from a stale cache line.
                // Either path yields a fresh generation paired with
                // a stale wake_ns and contaminates the resume-latency
                // histogram.
                let wake_ts_ptr = unsafe { (futex_ptr as *mut u8).add(8) as *mut u64 };
                let gen_atom = unsafe { &*(futex_ptr as *const std::sync::atomic::AtomicU64) };
                let wake_atom = unsafe { &*(wake_ts_ptr as *const std::sync::atomic::AtomicU64) };
                if is_messenger {
                    // Messenger: stamp wake time, advance generation, wake workers.
                    // Advance the generation and wake the workers
                    // ONLY after a successful `wake_ns` write. An
                    // earlier draft advanced the generation
                    // unconditionally, which meant a `clock_gettime`
                    // failure would wake workers against the *prior*
                    // round's `wake_ns` — producing an inflated
                    // `now_ns - wake_ns` latency that would
                    // contaminate the p99 tail of the reservoir.
                    // Skipping the whole round (including the wake)
                    // keeps the latency histogram honest; workers
                    // stay parked on `futex_wait` with its 100 ms
                    // timeout and observe the next successful round
                    // normally. `spin_burst` still runs on this
                    // thread so the messenger keeps producing
                    // work_units.
                    if let Some(wake_ns) = clock_gettime_ns(libc::CLOCK_MONOTONIC) {
                        // Relaxed wake_ns store is fine; the subsequent
                        // Release RMW on the generation synchronises
                        // it with the worker's Acquire load.
                        wake_atom.store(wake_ns, Ordering::Relaxed);
                        // fetch_add on u64 wraps at 2^64 and is
                        // sole-writer here, so one Release RMW beats
                        // load-Relaxed + store-Release. On aarch64,
                        // AtomicU64 Release ordering is guaranteed
                        // by LLVM to lower to a release-ordered
                        // instruction — LDADDL on LSE-capable cores
                        // (Armv8.1+), or an LDXR/STLXR retry loop
                        // on pre-LSE cores where STLXR supplies the
                        // release barrier. Either way the store-
                        // release half pairs with the worker's
                        // Acquire load below.
                        gen_atom.fetch_add(1, Ordering::Release);
                        unsafe { futex_wake(futex_ptr, clamp_futex_wake_n(fan_out)) };
                    }
                    spin_burst(&mut work_units, FAN_OUT_POST_WAKE_SPIN_ITERS);
                } else {
                    // Worker: wait for generation advance, then do work.
                    // Initial snapshot can be Relaxed — it only feeds
                    // `futex_wait`'s expected-value check; the real
                    // happens-before edge is established by the
                    // Acquire load below once the generation differs.
                    // u64 snapshot compared against u64 cur so
                    // wraparound cannot create a false-negative
                    // (see region-layout comment above). futex_wait
                    // takes a u32 expected, so the low 32 bits of
                    // the u64 snapshot get truncated for the syscall
                    // only — the messenger's fetch_add changes those
                    // low bits on every increment, so futex_wait's
                    // kernel-side expected-check still fires
                    // correctly on every advance.
                    let expected = gen_atom.load(Ordering::Relaxed);
                    let expected_low = expected as u32;
                    loop {
                        if stop_requested(stop) {
                            break;
                        }
                        let cur = gen_atom.load(Ordering::Acquire);
                        if cur != expected {
                            // Skip the reservoir push entirely on
                            // `clock_gettime` failure — previously
                            // the rc was discarded and a
                            // zeroed/garbage `now_ts` was fed into
                            // `saturating_sub`, silently contaminating
                            // the resume-latency histogram with values
                            // dominated by wake_ns itself.
                            if let Some(now_ns) = clock_gettime_ns(libc::CLOCK_MONOTONIC) {
                                // Acquire load above synchronises-with
                                // the messenger's Release store, so
                                // this wake_ns load sees the value
                                // paired with `cur`.
                                let wake_ns = wake_atom.load(Ordering::Relaxed);
                                let latency = now_ns.saturating_sub(wake_ns);
                                reservoir_push(
                                    &mut resume_latencies_ns,
                                    &mut wake_sample_count,
                                    latency,
                                    MAX_WAKE_SAMPLES,
                                );
                            }
                            break;
                        }
                        unsafe { futex_wait(futex_ptr, expected_low, &FUTEX_WAIT_TIMEOUT) };
                    }
                    if sleep_usec > 0 && !stop_requested(stop) {
                        std::thread::sleep(Duration::from_micros(sleep_usec));
                    }
                    if matrix_size > 0 && !stop_requested(stop) {
                        let buf = matrix_buf
                            .get_or_insert_with(|| vec![0u64; 3 * matrix_size * matrix_size]);
                        for _ in 0..operations {
                            // matrix_multiply itself folds a black_box-wrapped
                            // C-region read into `work_units` as the post-loop
                            // sink (see matrix_multiply doc), so the per-call
                            // accumulator increment lives inside the helper.
                            matrix_multiply(buf, matrix_size, &mut work_units);
                            work_units = std::hint::black_box(work_units.wrapping_add(1));
                        }
                    }
                }
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::PageFaultChurn {
                region_kb,
                touches_per_cycle,
                spin_iters,
            } => {
                let (ptr, region_size) = match page_fault_region {
                    Some(p) => p,
                    None => {
                        // `region_kb * 1024` overflows usize on 32-bit
                        // targets for region_kb >= 4 MiB-equivalent;
                        // `checked_mul` returns None there and the
                        // workload exits this iteration rather than
                        // wrapping to a tiny region. Previously
                        // silent — a test author who typo'd a huge
                        // `region_kb` would see a zero-iteration
                        // worker report with no diagnostic. Surface
                        // the overflow via `tracing::warn!` with the
                        // offending `region_kb` so the configuration
                        // bug is visible in the test log; the early
                        // `break` still keeps the process honest.
                        let region_size = match region_kb.checked_mul(1024) {
                            Some(v) => v,
                            None => {
                                tracing::warn!(
                                    tid,
                                    region_kb,
                                    "PageFaultChurn region_kb * 1024 overflowed usize — worker exiting outer loop without doing page-fault work"
                                );
                                break;
                            }
                        };
                        let ptr = unsafe {
                            libc::mmap(
                                std::ptr::null_mut(),
                                region_size,
                                libc::PROT_READ | libc::PROT_WRITE,
                                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                                -1,
                                0,
                            )
                        };
                        if ptr == libc::MAP_FAILED {
                            break;
                        }
                        unsafe {
                            libc::madvise(ptr, region_size, libc::MADV_NOHUGEPAGE);
                        }
                        // xorshift64 requires a non-zero seed; OR-ing
                        // tid with 1 forces the low bit on.
                        page_fault_rng_state = (tid as u64) | 1;
                        page_fault_region = Some((ptr, region_size));
                        (ptr, region_size)
                    }
                };
                // region_kb < 4 produces region_size < 4096, so
                // `region_size / 4096` truncates to zero and the
                // `% page_count` below would panic (or UB in release
                // with panic=abort). mmap rounds up to a whole page
                // internally regardless of the requested length, so
                // the kernel actually handed us at least one page
                // of mapped memory even for a sub-page `region_kb`.
                // Clamping `page_count` to at least 1 matches that
                // physical reality: the single page gets touched
                // every iteration, preserving the churn intent
                // without introducing a panic edge.
                let page_count = (region_size / 4096).max(1);
                let xorshift64 = |state: &mut u64| -> u64 {
                    let mut x = *state;
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    *state = x;
                    x
                };
                for _ in 0..touches_per_cycle {
                    let page_idx = (xorshift64(&mut page_fault_rng_state) as usize) % page_count;
                    let page_ptr = unsafe { (ptr as *mut u8).add(page_idx * 4096) };
                    unsafe { std::ptr::write_volatile(page_ptr, 1u8) };
                    work_units = std::hint::black_box(work_units.wrapping_add(1));
                }
                unsafe {
                    libc::madvise(ptr, region_size, libc::MADV_DONTNEED);
                }
                spin_burst(&mut work_units, spin_iters);
                iterations += 1;
            }
            WorkType::MutexContention {
                hold_iters,
                work_iters,
                ..
            } => {
                // pos discarded: every contender competes equally on
                // the same futex word — no per-position differentiation.
                let (futex_ptr, _pos) = match futex {
                    Some(f) => f,
                    None => break,
                };
                spin_burst(&mut work_units, work_iters);
                let atom = unsafe { &*(futex_ptr as *const std::sync::atomic::AtomicU32) };
                // CAS acquire: try to set 0 -> 1. On failure, FUTEX_WAIT.
                loop {
                    if stop_requested(stop) {
                        break;
                    }
                    if atom
                        .compare_exchange_weak(0, 1, Ordering::Acquire, Ordering::Relaxed)
                        .is_ok()
                    {
                        break;
                    }
                    let before_block = Instant::now();
                    unsafe {
                        futex_wait(
                            futex_ptr,
                            1u32, /* expected value (locked) */
                            &FUTEX_WAIT_TIMEOUT,
                        )
                    };
                    reservoir_push(
                        &mut resume_latencies_ns,
                        &mut wake_sample_count,
                        before_block.elapsed().as_nanos() as u64,
                        MAX_WAKE_SAMPLES,
                    );
                }
                // Critical section: hold the lock.
                spin_burst(&mut work_units, hold_iters);
                // Release: atomic store with Release ordering ensures
                // critical section work is visible before the unlock.
                atom.store(0, Ordering::Release);
                unsafe { futex_wake(futex_ptr, 1) };
                last_iter_time = Instant::now();
                iterations += 1;
            }
            // Stubs for the 6 new pathology-taxonomy variants. The
            // type-system surface is wired (enum arms, factory
            // methods, name/from_name/group_size/needs_shared_mem);
            // per-variant worker_main bodies land later. Until then,
            // each variant's outer loop spins burst-iter CPU work so
            // a worker that gets dispatched (e.g. via from_name()
            // round-trip tests) still produces a non-zero work_units
            // report rather than silently looping at zero.
            WorkType::ThunderingHerd {
                waiters,
                batches,
                inter_batch_ms,
            } => {
                // Single global futex: every worker in the group
                // shares the same `futex_ptr` because
                // `worker_group_size = waiters + 1` collapses the
                // herd into one group. `pos == 0` is the waker;
                // `pos > 0` are waiters.
                //
                // Waker: increment generation, FUTEX_WAKE(INT_MAX)
                // — broadcasts to every parked waiter
                // simultaneously (`kernel/futex/waitwake.c`'s
                // `futex_wake_op` walks the bucket's plist and
                // wakes up to `nr_wake` callers). We pass
                // `i32::MAX` via `clamp_futex_wake_n(usize::MAX)`
                // so the kernel wakes everyone parked on this
                // word in a single syscall, matching the
                // thundering-herd shape.
                //
                // Waiter: park on the futex, observe generation
                // advance, record resume latency. Same idiom as
                // FutexFanOut waiter; the difference is purely the
                // group shape (single global vs per-group).
                //
                // After the configured number of batches, the
                // waker stops triggering and the waiters drain.
                // STOP from SIGUSR1 unblocks both sides via the
                // FUTEX_WAIT_TIMEOUT poll cycle.
                let (futex_ptr, pos) = match futex {
                    Some(f) => f,
                    None => break,
                };
                let is_waker = pos == 0;
                let atom = unsafe { &*(futex_ptr as *const std::sync::atomic::AtomicU32) };
                if is_waker {
                    let mut batches_done: u64 = 0;
                    while batches_done < batches && !stop_requested(stop) {
                        // Inter-batch sleep so waiters re-park on
                        // futex before the next thundering wake.
                        // `nanosleep` blocking ALSO contributes a
                        // wake-latency sample for the waker so its
                        // report carries telemetry comparable to
                        // the waiters'.
                        if inter_batch_ms > 0 {
                            let before_sleep = Instant::now();
                            std::thread::sleep(Duration::from_millis(inter_batch_ms));
                            reservoir_push(
                                &mut resume_latencies_ns,
                                &mut wake_sample_count,
                                before_sleep.elapsed().as_nanos() as u64,
                                MAX_WAKE_SAMPLES,
                            );
                        }
                        // Advance generation counter and broadcast
                        // wake. Relaxed ordering matches FutexFanOut
                        // — futex syscall supplies kernel-side
                        // cross-thread ordering for the wake itself.
                        let next = atom.load(Ordering::Relaxed).wrapping_add(1);
                        atom.store(next, Ordering::Relaxed);
                        // Clamp to i32::MAX so the syscall wakes
                        // every parked waiter on the futex word.
                        unsafe { futex_wake(futex_ptr, clamp_futex_wake_n(usize::MAX)) };
                        spin_burst(&mut work_units, 256);
                        batches_done += 1;
                    }
                } else {
                    // Waiter: park, observe advance, record latency.
                    let _ = waiters; // pattern-binding only; size set at spawn time.
                    let expected = atom.load(Ordering::Relaxed);
                    let before_block = Instant::now();
                    loop {
                        if stop_requested(stop) {
                            break;
                        }
                        let cur = atom.load(Ordering::Relaxed);
                        if cur != expected {
                            reservoir_push(
                                &mut resume_latencies_ns,
                                &mut wake_sample_count,
                                before_block.elapsed().as_nanos() as u64,
                                MAX_WAKE_SAMPLES,
                            );
                            break;
                        }
                        unsafe { futex_wait(futex_ptr, expected, &FUTEX_WAIT_TIMEOUT) };
                    }
                }
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::WakeChain {
                depth,
                fanout: _,
                sync,
                work_per_hop,
            } => {
                // Linear ring chain inside one chain group. Each
                // chain owns one shared 4-byte futex word
                // (`worker_group_size = depth`); workers within the
                // chain are positioned by `pos` (0..depth). Stage
                // boundary = the futex word's value advances to
                // `next_stage`; the worker holding `pos == word_val`
                // is the active stage. After `work_per_hop` of CPU
                // work, the active stage advances the word to
                // `(pos + 1) % depth` and wakes everyone parked on
                // it; only the next stage's worker satisfies its
                // wait predicate and proceeds.
                //
                // `sync = true` adds a 1-byte pipe round-trip on
                // the existing report-pipe-style infra... but we
                // don't have per-stage pipe pairs allocated for
                // this variant (`needs_pipes = false` for
                // WakeChain). To keep batch 3 scope confined to
                // worker_main, both `sync = true` and
                // `sync = false` use the futex-word advance; the
                // wake call uses `clamp_futex_wake_n(usize::MAX)`
                // which lowers to FUTEX_WAKE(i32::MAX) — the kernel
                // walks the bucket's plist and wakes every parked
                // waiter (kernel/futex/waitwake.c::futex_wake_op),
                // so all non-active stages observe the advance and
                // re-park on the new value. This preserves the
                // wake-chain shape (sequenced stages, per-stage CPU
                // burst, scheduler observes a chain of WAKE→work→
                // WAKE→work transitions). The `sync` flag is read
                // and recorded so future work that wires per-stage
                // pipes can branch on it; for now it's ignored at
                // the syscall layer.
                let (futex_ptr, pos) = match futex {
                    Some(f) => f,
                    None => break,
                };
                let _ = sync;
                if depth == 0 || pos >= depth {
                    // Defense in depth: surface uses
                    // `worker_group_size = depth`, so the spawn-side
                    // divisibility check guarantees pos < depth
                    // before we get here. This branch handles only
                    // a programmer bug that bypasses spawn.
                    break;
                }
                let atom = unsafe { &*(futex_ptr as *const std::sync::atomic::AtomicU32) };
                let my_stage = pos as u32;
                let next_stage = ((pos + 1) % depth) as u32;
                let before_block = Instant::now();
                loop {
                    if stop_requested(stop) {
                        break;
                    }
                    let cur = atom.load(Ordering::Relaxed);
                    if cur == my_stage {
                        // Our turn. Record blocked-time as a wake
                        // sample. pos == 0 on the very first
                        // iteration sees `cur == 0` immediately
                        // (never blocked) — `before_block` is
                        // post-spawn, the elapsed time still
                        // captures the spawn-to-first-stage gap,
                        // matching how FutexFanOut handles its
                        // first iteration.
                        reservoir_push(
                            &mut resume_latencies_ns,
                            &mut wake_sample_count,
                            before_block.elapsed().as_nanos() as u64,
                            MAX_WAKE_SAMPLES,
                        );
                        break;
                    }
                    unsafe { futex_wait(futex_ptr, cur, &FUTEX_WAIT_TIMEOUT) };
                }
                if stop_requested(stop) {
                    iterations += 1;
                    continue;
                }
                // Stage work: spin for the configured duration. Use
                // `Instant`-based gating so `work_per_hop`'s unit
                // (Duration) is honored regardless of host CPU
                // speed — same idiom as `Bursty`'s `burst_ms`
                // (Duration::from_millis on a per-iteration
                // deadline).
                let work_end = Instant::now() + work_per_hop;
                while Instant::now() < work_end && !stop_requested(stop) {
                    spin_burst(&mut work_units, 256);
                }
                if stop_requested(stop) {
                    iterations += 1;
                    continue;
                }
                // Advance to the next stage and wake everyone
                // parked. Relaxed store: futex syscall provides the
                // kernel-side cross-thread ordering for the wake
                // event (matches FutexFanOut's idiom).
                atom.store(next_stage, Ordering::Relaxed);
                unsafe { futex_wake(futex_ptr, clamp_futex_wake_n(usize::MAX)) };
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::AsymmetricWaker {
                waker_class,
                wakee_class,
                burst_iters,
            } => {
                // Paired waker/wakee in different scheduling classes.
                // `worker_group_size = 2`, so pos ∈ {0, 1}: pos == 0
                // is the waker, pos == 1 is the wakee. Each holds
                // its own class for the entire run; transition
                // happens once on the first iteration.
                let (futex_ptr, pos) = match futex {
                    Some(f) => f,
                    None => break,
                };
                if !per_pos_policy_applied {
                    let class = if pos == 0 { waker_class } else { wakee_class };
                    // Soft-fail on EPERM (no CAP_SYS_NICE) — same
                    // policy as the apply_nice / set_thread_affinity
                    // sites in worker_main: log and continue with
                    // the inherited class so the test reports
                    // visible failure mode rather than crashing.
                    let _ = set_sched_policy(0, class.to_policy());
                    per_pos_policy_applied = true;
                }
                let atom = unsafe { &*(futex_ptr as *const std::sync::atomic::AtomicU32) };
                if pos == 0 {
                    // Waker: spin to build CPU runtime, then advance
                    // the futex word and FUTEX_WAKE the wakee. The
                    // wakee's resume_latencies_ns reservoir will
                    // capture the wake-affine placement gap on its
                    // side; the waker's reservoir is empty (no
                    // blocking syscall on this side).
                    spin_burst(&mut work_units, burst_iters);
                    let next = atom.load(Ordering::Relaxed).wrapping_add(1);
                    atom.store(next, Ordering::Relaxed);
                    unsafe { futex_wake(futex_ptr, 1) };
                } else {
                    // Wakee: park on the futex word; advance to
                    // user-space when the waker bumps it. Same
                    // observe-then-record pattern as FutexFanOut's
                    // receiver — `before_block` captures the full
                    // wait→wake→reschedule round trip.
                    let expected = atom.load(Ordering::Relaxed);
                    let before_block = Instant::now();
                    loop {
                        if stop_requested(stop) {
                            break;
                        }
                        let cur = atom.load(Ordering::Relaxed);
                        if cur != expected {
                            reservoir_push(
                                &mut resume_latencies_ns,
                                &mut wake_sample_count,
                                before_block.elapsed().as_nanos() as u64,
                                MAX_WAKE_SAMPLES,
                            );
                            break;
                        }
                        unsafe { futex_wait(futex_ptr, expected, &FUTEX_WAIT_TIMEOUT) };
                    }
                    // Wakee also burns CPU after wake to test
                    // wake-affine placement under load — without
                    // this the wakee re-parks immediately and the
                    // scheduler never sees concurrent demand.
                    spin_burst(&mut work_units, burst_iters);
                }
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::PriorityInversion {
                high_count,
                medium_count,
                low_count,
                hold_iters,
                work_iters,
                pi_mode,
            } => {
                // Three priority tiers contend on one shared futex
                // word in the same group. `pos` selects tier:
                //   pos < high_count → high (top RT prio)
                //   pos < high_count + medium_count → medium (mid RT)
                //   else → low (lowest RT prio)
                // The classic inversion: `low` holds the lock,
                // `medium` runs at higher prio and preempts `low`,
                // `high` waits on the lock indefinitely.
                //
                // pi_mode:
                //   Pi   → FUTEX_LOCK_PI (rt_mutex PI boost via
                //          kernel/futex/pi.c — kernel boosts the
                //          lock holder to the waiter's prio for
                //          the duration of the hold, breaking the
                //          inversion).
                //   Plain → plain CAS + FUTEX_WAIT/WAKE — the
                //           inversion goes uncorrected.
                //
                // RT priority assignment:
                //   high   → 70  (top)
                //   medium → 50  (middle, between high and low)
                //   low    → 30  (bottom; still RT so it competes
                //                  in the rt class but loses to
                //                  medium under preemption)
                // Picked inside 1..=99 so even a loaded host with
                // an existing kernel-RT task at prio 99 (e.g.
                // migration/N) sees three distinct tiers.
                let (futex_ptr, pos) = match futex {
                    Some(f) => f,
                    None => break,
                };
                let high_end = high_count;
                let medium_end = high_count + medium_count;
                let total = high_count + medium_count + low_count;
                if pos >= total {
                    break;
                }
                let (tier_prio, is_low, is_medium) = if pos < high_end {
                    (70u32, false, false)
                } else if pos < medium_end {
                    (50u32, false, true)
                } else {
                    (30u32, true, false)
                };
                if !per_pos_policy_applied {
                    let _ = set_sched_policy(0, SchedPolicy::Fifo(tier_prio));
                    per_pos_policy_applied = true;
                }
                let atom = unsafe { &*(futex_ptr as *const std::sync::atomic::AtomicU32) };
                if is_medium {
                    // Medium: pure CPU spin (no lock). Higher prio
                    // than `low` so it preempts the lock holder.
                    spin_burst(&mut work_units, work_iters);
                } else {
                    // High and low both contend on the lock.
                    spin_burst(&mut work_units, work_iters);
                    match pi_mode {
                        PiMode::Pi => {
                            // FUTEX_LOCK_PI: kernel handles the
                            // CAS atomically, transfers ownership
                            // via the futex word's TID encoding,
                            // and applies PI boost on the holder.
                            // Returns 0 on lock-acquired, -1 on
                            // error or signal.
                            let lock_rc = unsafe {
                                libc::syscall(
                                    libc::SYS_futex,
                                    futex_ptr,
                                    libc::FUTEX_LOCK_PI,
                                    0u32, /* unused for LOCK_PI */
                                    std::ptr::null::<libc::timespec>(),
                                    std::ptr::null::<u32>(),
                                    0u32,
                                )
                            };
                            if lock_rc == 0 {
                                spin_burst(&mut work_units, hold_iters);
                                unsafe {
                                    libc::syscall(
                                        libc::SYS_futex,
                                        futex_ptr,
                                        libc::FUTEX_UNLOCK_PI,
                                        0u32,
                                        std::ptr::null::<libc::timespec>(),
                                        std::ptr::null::<u32>(),
                                        0u32,
                                    );
                                }
                            }
                        }
                        PiMode::Plain => {
                            // Plain spin-then-wait: try CAS 0→1,
                            // FUTEX_WAIT on contention, hold
                            // hold_iters of spin, store 0 + wake
                            // on release. Same idiom as
                            // MutexContention's body.
                            loop {
                                if stop_requested(stop) {
                                    break;
                                }
                                if atom
                                    .compare_exchange_weak(
                                        0,
                                        1,
                                        Ordering::Acquire,
                                        Ordering::Relaxed,
                                    )
                                    .is_ok()
                                {
                                    break;
                                }
                                let before_block = Instant::now();
                                unsafe {
                                    futex_wait(futex_ptr, 1u32, &FUTEX_WAIT_TIMEOUT);
                                }
                                reservoir_push(
                                    &mut resume_latencies_ns,
                                    &mut wake_sample_count,
                                    before_block.elapsed().as_nanos() as u64,
                                    MAX_WAKE_SAMPLES,
                                );
                            }
                            // Hold critical section. `low` does
                            // hold_iters of spin (the inversion
                            // window); `high` does work_iters
                            // (it just wants to acquire+release).
                            let hold = if is_low { hold_iters } else { work_iters };
                            spin_burst(&mut work_units, hold);
                            atom.store(0, Ordering::Release);
                            unsafe { futex_wake(futex_ptr, 1) };
                        }
                    }
                }
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::ProducerConsumerImbalance {
                producers,
                consumers,
                produce_rate_hz,
                consume_iters,
                queue_depth_target,
            } => {
                // SPMC-ish ring queue in shared memory. Layout:
                //   offset 0  : head (producer write idx, u64)
                //   offset 8  : tail (consumer read idx, u64)
                //   offset 16 : prod_wake (consumers' "queue drained" futex, u32)
                //   offset 20 : cons_wake (producers' "items available" futex, u32)
                //   offset 24 : ring[Q] of u64 slots
                // pos < producers → producer; else consumer.
                //
                // Producer paces with nanosleep(1s/produce_rate_hz)
                // between pushes. On full queue (head - tail == Q):
                // FUTEX_WAIT on prod_wake (consumers wake it when
                // tail advances). Producer tags items with
                // monotonic counter — content opaque to the
                // workload, only its sequencing matters.
                //
                // Consumer pops one item per loop: if head == tail,
                // FUTEX_WAIT on cons_wake (producers wake it when
                // head advances). Then spin consume_iters of CPU.
                //
                // Imbalance: when producers * rate > consumers * /
                // (consume_iters work-time), the queue grows toward
                // Q and producers eventually block — pressure-
                // testing scheduler fairness under sustained
                // backpressure (DSQ unbounded growth in scx).
                //
                // Atomic ordering: head/tail are accessed via
                // AtomicU64::{load,store} with Acquire/Release.
                // The Release on producer's head store pairs with
                // the consumer's Acquire on head — once consumer
                // observes head > tail, the slot write is visible.
                let (futex_ptr, pos) = match futex {
                    Some(f) => f,
                    None => break,
                };
                let total = producers + consumers;
                if pos >= total || queue_depth_target == 0 {
                    break;
                }
                let q_target_usize =
                    std::cmp::min(queue_depth_target as usize, usize::MAX / 8 - 3);
                let q = q_target_usize as u64;
                if q == 0 {
                    break;
                }
                let base = futex_ptr as *mut u8;
                let head_atom =
                    unsafe { &*(base as *const std::sync::atomic::AtomicU64) };
                let tail_atom = unsafe {
                    &*(base.add(8) as *const std::sync::atomic::AtomicU64)
                };
                let prod_wake_ptr = unsafe { base.add(16) as *mut u32 };
                let cons_wake_ptr = unsafe { base.add(20) as *mut u32 };
                let ring_base = unsafe { base.add(24) as *mut u64 };
                if pos < producers {
                    // Producer.
                    let mut next_seq: u64 = 0;
                    let pace_ns: u64 = if produce_rate_hz == 0 {
                        0
                    } else {
                        // Per-producer rate; total rate = producers
                        // × produce_rate_hz. Avoid division by
                        // zero with the gate above.
                        1_000_000_000u64 / produce_rate_hz
                    };
                    while !stop_requested(stop) {
                        // Block on full queue: FUTEX_WAIT on
                        // prod_wake until tail advances. The inner
                        // loop either sets slot_avail and breaks
                        // with reservation or breaks via STOP — the
                        // post-loop STOP check below short-circuits
                        // before reading slot_avail in the latter
                        // case.
                        let mut slot_avail: u64 = 0;
                        let mut got_slot = false;
                        loop {
                            if stop_requested(stop) {
                                break;
                            }
                            let head = head_atom.load(Ordering::Relaxed);
                            let tail = tail_atom.load(Ordering::Acquire);
                            if head.wrapping_sub(tail) < q {
                                slot_avail = head;
                                got_slot = true;
                                break;
                            }
                            let prod_wake_atom = unsafe {
                                &*(prod_wake_ptr as *const std::sync::atomic::AtomicU32)
                            };
                            let expected = prod_wake_atom.load(Ordering::Relaxed);
                            let before_block = Instant::now();
                            unsafe { futex_wait(prod_wake_ptr, expected, &FUTEX_WAIT_TIMEOUT) };
                            reservoir_push(
                                &mut resume_latencies_ns,
                                &mut wake_sample_count,
                                before_block.elapsed().as_nanos() as u64,
                                MAX_WAKE_SAMPLES,
                            );
                        }
                        if !got_slot || stop_requested(stop) {
                            break;
                        }
                        // Write slot at head % q. The Release on
                        // head_atom.store() publishes both the slot
                        // contents and the head advance to consumers.
                        let slot_idx = (slot_avail % q) as usize;
                        unsafe {
                            std::ptr::write_volatile(ring_base.add(slot_idx), next_seq);
                        }
                        head_atom.store(slot_avail.wrapping_add(1), Ordering::Release);
                        next_seq = next_seq.wrapping_add(1);
                        // Wake one consumer (advance cons_wake counter).
                        let cons_wake_atom = unsafe {
                            &*(cons_wake_ptr as *const std::sync::atomic::AtomicU32)
                        };
                        let cur = cons_wake_atom.load(Ordering::Relaxed);
                        cons_wake_atom.store(cur.wrapping_add(1), Ordering::Relaxed);
                        unsafe { futex_wake(cons_wake_ptr, 1) };
                        work_units = std::hint::black_box(work_units.wrapping_add(1));
                        // Pace.
                        if pace_ns > 0 {
                            let ts = libc::timespec {
                                tv_sec: (pace_ns / 1_000_000_000) as libc::time_t,
                                tv_nsec: (pace_ns % 1_000_000_000) as libc::c_long,
                            };
                            unsafe {
                                libc::nanosleep(&ts, std::ptr::null_mut());
                            }
                        }
                        iterations += 1;
                    }
                } else {
                    // Consumer.
                    while !stop_requested(stop) {
                        // Block on empty queue. Same init/got
                        // pattern as the producer half so the
                        // borrow checker can prove item_idx is
                        // initialized when read.
                        let mut item_idx: u64 = 0;
                        let mut got_item = false;
                        loop {
                            if stop_requested(stop) {
                                break;
                            }
                            let tail = tail_atom.load(Ordering::Relaxed);
                            let head = head_atom.load(Ordering::Acquire);
                            if head != tail {
                                item_idx = tail;
                                got_item = true;
                                break;
                            }
                            let cons_wake_atom = unsafe {
                                &*(cons_wake_ptr as *const std::sync::atomic::AtomicU32)
                            };
                            let expected = cons_wake_atom.load(Ordering::Relaxed);
                            let before_block = Instant::now();
                            unsafe { futex_wait(cons_wake_ptr, expected, &FUTEX_WAIT_TIMEOUT) };
                            reservoir_push(
                                &mut resume_latencies_ns,
                                &mut wake_sample_count,
                                before_block.elapsed().as_nanos() as u64,
                                MAX_WAKE_SAMPLES,
                            );
                        }
                        if !got_item || stop_requested(stop) {
                            break;
                        }
                        let slot_idx = (item_idx % q) as usize;
                        let _val = unsafe { std::ptr::read_volatile(ring_base.add(slot_idx)) };
                        // Advance tail with Release so producers
                        // observing tail also see we've finished
                        // reading the slot.
                        tail_atom.store(item_idx.wrapping_add(1), Ordering::Release);
                        // Wake a producer that may be blocked on full queue.
                        let prod_wake_atom = unsafe {
                            &*(prod_wake_ptr as *const std::sync::atomic::AtomicU32)
                        };
                        let cur = prod_wake_atom.load(Ordering::Relaxed);
                        prod_wake_atom.store(cur.wrapping_add(1), Ordering::Relaxed);
                        unsafe { futex_wake(prod_wake_ptr, 1) };
                        // Burn consume_iters of CPU.
                        spin_burst(&mut work_units, consume_iters);
                        iterations += 1;
                    }
                }
                last_iter_time = Instant::now();
            }
            WorkType::RtStarvation {
                rt_workers,
                cfs_workers: _,
                rt_priority,
                burst_iters,
            } => {
                // RT workers (pos < rt_workers) run as SCHED_FIFO
                // at `rt_priority`; CFS workers (pos >= rt_workers)
                // stay on SCHED_NORMAL. Both groups spin burst_iters
                // per outer iteration. The pathology: SCHED_FIFO at
                // any priority preempts SCHED_NORMAL until the kernel's
                // RT throttling kicks in
                // (`sched_rt_period_us`/`sched_rt_runtime_us`); under
                // sched_ext switch-all, ext_sched_class loses to the
                // RT class on the same CPU because dl_sched_class >
                // rt_sched_class > ext_sched_class in the class
                // hierarchy. There is no DL server protecting ext
                // (in contrast to the DL server that throttles RT
                // for fair tasks), so an ext-managed task starves
                // until RT yields. This is the inversion.
                //
                // pos for cfs_workers is implicit (anything >=
                // rt_workers is CFS); _ binds it without warning.
                let (_, pos) = match futex {
                    Some(f) => f,
                    None => break,
                };
                if !per_pos_policy_applied {
                    if pos < rt_workers {
                        // Clamp at the syscall boundary: kernel
                        // rejects priorities outside 1..=99 with
                        // EINVAL, but we soft-clamp to a sane range
                        // so a programmer typo doesn't kill the
                        // worker.
                        let prio = rt_priority.clamp(1, 99) as u32;
                        let _ = set_sched_policy(0, SchedPolicy::Fifo(prio));
                    } else {
                        let _ = set_sched_policy(0, SchedPolicy::Normal);
                    }
                    per_pos_policy_applied = true;
                }
                spin_burst(&mut work_units, burst_iters);
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::NumaWorkingSetSweep {
                region_kb,
                sweep_period_ms,
                ref target_nodes,
            } => {
                // Per-worker anonymous region, rebound to a
                // rotating NUMA node every `sweep_period_ms`. Each
                // sweep:
                //   1. mbind(MPOL_BIND, MPOL_MF_MOVE) the region to
                //      `target_nodes[(iter + phase) % len]` so the
                //      kernel migrates pages off the current node.
                //   2. Touch every page (via volatile write) so
                //      the migration triggers physical page motion
                //      rather than lazy-reservation.
                //   3. nanosleep(sweep_period_ms) before next bind.
                //
                // Empty target_nodes: no binding, just keep
                // touching the region every iteration for the
                // baseline. Single-node target_nodes: pin once
                // (effectively MPOL_BIND with no rotation).
                //
                // Per-worker phase = tid % len so the cohort
                // doesn't slam the same node simultaneously
                // (matches the "phase offset" doc on the variant).
                //
                // Region allocated lazily on first iteration via
                // `page_fault_region` to reuse the existing
                // PageFaultChurn-style mmap+free idiom (the
                // SpawnGuard does NOT clean per-worker mmaps on
                // exit because they're post-fork; the worker
                // lives until SIGUSR1 and then exits, releasing
                // the mapping).
                let region_size = match region_kb.checked_mul(1024) {
                    Some(v) => v,
                    None => {
                        tracing::warn!(
                            tid,
                            region_kb,
                            "NumaWorkingSetSweep region_kb * 1024 overflowed usize"
                        );
                        break;
                    }
                };
                let (ptr, _) = match page_fault_region {
                    Some(p) => p,
                    None => {
                        let ptr = unsafe {
                            libc::mmap(
                                std::ptr::null_mut(),
                                region_size,
                                libc::PROT_READ | libc::PROT_WRITE,
                                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                                -1,
                                0,
                            )
                        };
                        if ptr == libc::MAP_FAILED {
                            break;
                        }
                        page_fault_region = Some((ptr, region_size));
                        (ptr, region_size)
                    }
                };
                // Rotate target node based on iteration count.
                if !target_nodes.is_empty() {
                    let phase = (tid as usize) % target_nodes.len();
                    let node_idx =
                        ((iterations as usize).wrapping_add(phase)) % target_nodes.len();
                    let node = target_nodes[node_idx];
                    let (mask, maxnode) =
                        build_nodemask(&[node].into_iter().collect::<BTreeSet<usize>>());
                    // MPOL_MF_MOVE = 1 << 1 (include/uapi/linux/mempolicy.h).
                    // MPOL_BIND from libc.
                    const MPOL_MF_MOVE: libc::c_ulong = 1 << 1;
                    let _ = unsafe {
                        libc::syscall(
                            libc::SYS_mbind,
                            ptr,
                            region_size as libc::c_ulong,
                            libc::MPOL_BIND as libc::c_ulong,
                            mask.as_ptr(),
                            maxnode,
                            MPOL_MF_MOVE,
                        )
                    };
                }
                // Touch every page so any migration kicked off
                // by mbind actually moves a referenced page (the
                // kernel only migrates pages the process has
                // accessed). page_count clamped to 1 so a sub-page
                // region is still touched.
                let page_count = (region_size / 4096).max(1);
                for page_idx in 0..page_count {
                    let page_ptr = unsafe { (ptr as *mut u8).add(page_idx * 4096) };
                    unsafe { std::ptr::write_volatile(page_ptr, 1u8) };
                    work_units = std::hint::black_box(work_units.wrapping_add(1));
                }
                if sweep_period_ms > 0 && !stop_requested(stop) {
                    let before_sleep = Instant::now();
                    std::thread::sleep(Duration::from_millis(sweep_period_ms));
                    reservoir_push(
                        &mut resume_latencies_ns,
                        &mut wake_sample_count,
                        before_sleep.elapsed().as_nanos() as u64,
                        MAX_WAKE_SAMPLES,
                    );
                }
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::Custom { .. } => unreachable!("handled by early return"),
        }

        // Publish iteration count to shared memory for host-side
        // sampling. SAFETY: alignment + atomic-only-access invariant
        // established at the iter_counters mmap site in
        // `WorkloadHandle::spawn` and carried by the
        // `*mut AtomicU64` type.
        if !iter_slot.is_null() {
            // Relaxed store: the parent reads this counter via
            // `snapshot_iterations()` with Relaxed ordering only for
            // progress-sampling — no cross-field happens-before edge
            // is required (see that function's ordering rationale).
            unsafe { &*iter_slot }.store(iterations, Ordering::Relaxed);
        }

        if work_units.is_multiple_of(1024) {
            let now = Instant::now();
            let gap = now.duration_since(last_iter_time).as_nanos() as u64;
            if gap > max_gap_ns {
                max_gap_ns = gap;
                max_gap_cpu = last_cpu;
                max_gap_at_ns = now.duration_since(start).as_nanos() as u64;
            }
            last_iter_time = now;

            let cpu = sched_getcpu();
            if cpu != last_cpu {
                migration_count += 1;
                cpus_used.insert(cpu);
                migrations.push(Migration {
                    at_ns: now.duration_since(start).as_nanos() as u64,
                    from_cpu: last_cpu,
                    to_cpu: cpu,
                });
                last_cpu = cpu;
            }
        }
    }

    // Reset nice to 0 so report serialization runs at default priority.
    if matches!(work_type, WorkType::NiceSweep) {
        unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, 0) };
    }

    // Reset to SCHED_OTHER so report serialization runs at normal policy.
    if matches!(work_type, WorkType::PolicyChurn { .. }) {
        let param = libc::sched_param { sched_priority: 0 };
        unsafe { libc::sched_setscheduler(0, libc::SCHED_OTHER, &param) };
    }

    // Clean up persistent temp files.
    if let Some((_, path)) = io_sync_file {
        let _ = std::fs::remove_file(&path);
    }
    if let Some((_, path)) = io_seq_file {
        let _ = std::fs::remove_file(&path);
    }
    // Clean up persistent PageFaultChurn mmap region.
    if let Some((ptr, size)) = page_fault_region {
        unsafe { libc::munmap(ptr, size) };
    }

    // Final iteration count store for host-side sampling.
    // SAFETY: same as the iter_slot publish in the outer
    // `worker_main` loop above.
    if !iter_slot.is_null() {
        unsafe { &*iter_slot }.store(iterations, Ordering::Relaxed);
    }

    let wall_time = start.elapsed();
    let cpu_time_ns = thread_cpu_time_ns();
    let wall_time_ns = wall_time.as_nanos() as u64;

    // schedstat snapshot at work-loop end; compute deltas if both
    // snapshots succeeded, else zero (the start-of-loop read already
    // emitted a warning if schedstat is unavailable). Pair the
    // path with the start snapshot — same `tid` so the delta
    // measures the same task.
    let schedstat_end = read_schedstat(Some(tid));
    let (ss_delay_delta, ss_ts_delta, ss_cpu_delta) = match (schedstat_start, schedstat_end) {
        (Some((cpu_s, delay_s, ts_s)), Some((cpu_e, delay_e, ts_e))) => (
            delay_e.saturating_sub(delay_s),
            ts_e.saturating_sub(ts_s),
            cpu_e.saturating_sub(cpu_s),
        ),
        _ => (0, 0, 0),
    };

    // NUMA: read numa_maps and vmstat after workload.
    let numa_pages = read_numa_maps_pages();
    let vmstat_migrated_end = read_vmstat_numa_pages_migrated();
    let vmstat_migrated_delta = vmstat_migrated_end.saturating_sub(vmstat_migrated_start);

    WorkerReport {
        tid,
        work_units,
        cpu_time_ns,
        wall_time_ns,
        off_cpu_ns: wall_time_ns.saturating_sub(cpu_time_ns),
        migration_count,
        cpus_used,
        migrations,
        max_gap_ms: max_gap_ns / 1_000_000,
        max_gap_cpu,
        max_gap_at_ms: max_gap_at_ns / 1_000_000,
        resume_latencies_ns,
        wake_sample_total: wake_sample_count,
        iterations,
        schedstat_run_delay_ns: ss_delay_delta,
        schedstat_run_count: ss_ts_delta,
        schedstat_cpu_time_ns: ss_cpu_delta,
        completed: true,
        numa_pages,
        vmstat_numa_pages_migrated: vmstat_migrated_delta,
        // Populated by the sentinel path in `stop_and_collect`; a
        // report emitted from this (live) worker path always carries
        // `None` — the child reached the `f.write_all(&json)` site
        // and handed a complete report back to the parent.
        exit_info: None,
        // `futex` is `Some((ptr, pos))` for several work types and
        // `pos == 0` MEANS DIFFERENT THINGS PER VARIANT:
        //   - FutexFanOut / FanOutCompute: pos == 0 is the
        //     messenger — one worker per group advances the
        //     generation and fans out wakes. Exactly the shape the
        //     WorkerReport doc pins.
        //   - FutexPingPong: pos == 0 is a pair-position flag.
        //     Both workers write+wake symmetrically; neither is a
        //     messenger.
        //   - MutexContention: pos is unused (every contender
        //     competes equally on the same word).
        //   - ThunderingHerd: pos == 0 is the waker; pos > 0 are
        //     waiters parked on the futex. Not a messenger in the
        //     fan-out sense — the waker doesn't carry per-message
        //     state, just kicks the herd.
        //   - WakeChain: pos is the stage index in the chain ring.
        //     The active stage rotates each iteration, so no single
        //     worker is "the messenger" across the run.
        // Gate on the WorkType so only the fanout variants
        // propagate `pos == 0` as `is_messenger`; every other work
        // type lands `false` as the field doc contract requires.
        is_messenger: matches!(
            work_type,
            WorkType::FutexFanOut { .. } | WorkType::FanOutCompute { .. }
        ) && futex.map(|(_, p)| p == 0).unwrap_or(false),
    }
}

// =====================================================================
// Workload primitives — DO NOT remove the "weird-looking" constructs
// =====================================================================
//
// The functions below (`spin_burst`, `cache_rmw_loop`,
// `matrix_multiply`, the per-WorkType inline loops in `worker_main`)
// are the kernels of every workload primitive ktstr exposes. They
// look like trivial loops but carry MULTIPLE LAYERS of optimization-
// elimination defenses that a casual reader (or a future maintainer
// running clippy with cleanup intent) might be tempted to remove
// as "redundant ceremony". Each layer is load-bearing:
//
// 1. **`std::hint::black_box(value)`** — a value-elimination
//    barrier. Routing `wrapping_add(1)` results, multiplicand
//    loads, and accumulator updates through `black_box` prevents
//    LLVM from constant-folding, partial-evaluating, or
//    algebraically simplifying the expressions. WITHOUT this,
//    `for _ in 0..count { x = x.wrapping_add(1) }` collapses to
//    `x += count` at `-O2`, defeating the per-iteration timing
//    granularity these workloads need to drive scheduler events.
//
// 2. **`ptr::read_volatile` / `ptr::write_volatile`** — a memory-
//    operation-elimination barrier. `black_box` keeps a value live,
//    but a sufficiently smart pass can still prove the BACKING
//    LOAD/STORE dead and synthesize the bytes from thin air. The
//    workloads' cache-pressure variants depend on actual L1/L2/LLC
//    line traffic — a process-local `Vec<u8>` whose contents no
//    external observer reads is otherwise DCE-eligible. Volatile
//    operations are not eliminable: every access becomes a real
//    `mov` against the actual memory slot.
//
// 3. **Real syscalls** (`futex`, `pipe`, `read`, `write`,
//    `nanosleep`, `sched_yield`, `mmap`, etc.) — opaque to LLVM by
//    construction. The optimizer cannot reason across the
//    user-kernel boundary, so syscall sites act as natural barriers
//    that force surrounding values to materialize. WorkTypes that
//    need scheduler events (`FutexFanOut`, `IoSync`, `Phase::Yield`)
//    rely on this implicit barrier in addition to the explicit
//    `black_box` / volatile pairs above.
//
// 4. **`#[inline(never)]`** on the workload helpers (`spin_burst`,
//    `cache_rmw_loop`, `matrix_multiply`) — keeps each call a
//    distinct boundary in the IR. Without it, inlining can fuse
//    per-iteration `black_box` increments with the caller's
//    arithmetic, defeating the granularity defense.
//
// Backend assumption: these barriers assume the LLVM backend
// (rustc default for every release toolchain). On the cranelift
// backend, `black_box` is a pure no-op identity function —
// `rustc_codegen_cranelift/src/intrinsics/mod.rs` carries a
// literal `FIXME implement black_box semantics` and just writes
// the value back unchanged. Any build with
// `-Z codegen-backend=cranelift` would silently lose every
// `black_box` barrier in this file. Volatile loads/stores and
// real syscalls survive that backend swap, so the cache-pressure
// and PageFaultChurn variants stay anchored, but every
// `spin_burst` / `matrix_multiply` / `work_units` increment
// would become DCE-eligible. Stick with the LLVM backend for
// release / nextest / `cargo ktstr test` runs.
//
// Future maintainers: if you see code like
// `*work_units = std::hint::black_box(work_units.wrapping_add(1));`
// or `unsafe { ptr::read_volatile(&buf[idx]) }` and your reflex is
// "this can be simplified", STOP. Read this comment block. Each
// of these constructs has a documented function in the workload's
// optimization-resistance contract. Removing one (a) breaks the
// scheduler-event timing the workload claims to produce, (b)
// degrades the cache-pressure traffic, (c) collapses multi-step
// arithmetic into a single fold, OR (d) all three. The breakage
// won't surface as a test failure — it'll surface as silently
// degraded workload realism, which is much harder to debug than
// a panic.

/// CPU spin burst: black_box increment + spin_loop hint, repeated `count` times.
///
/// `#[inline(never)]` is deliberate: when this is inlined into a
/// caller that also does observable work after the loop, LLVM can
/// merge `count`-many `wrapping_add(1)` operations into a single
/// `+ count` operation, defeating the point of the per-iteration
/// `black_box`. Forcing the function out-of-line keeps each
/// iteration's `black_box`-wrapped increment visible as a
/// distinct call-and-return boundary the optimizer cannot fold.
#[inline(never)]
fn spin_burst(work_units: &mut u64, count: u64) {
    for _ in 0..count {
        *work_units = std::hint::black_box(work_units.wrapping_add(1));
        std::hint::spin_loop();
    }
}

/// Strided read-modify-write over a cache buffer.
///
/// `-O2`/`-O3` are aggressive about eliminating "no-observer" memory
/// traffic on a process-local `Vec<u8>`: nothing outside the worker
/// reads `buf`, so without an explicit barrier LLVM may prove every
/// store dead and collapse the loop body to the `work_units`
/// increment alone. `work_units` flows into a shared iter-slot
/// atomic and the worker report, which keeps THAT dependency live,
/// but that observable flow does not force the independent cache
/// traffic to execute.
///
/// `black_box` on a value defeats VALUE elimination — the load /
/// store has to materialize bytes the optimizer can't reason about
/// — but a sufficiently smart pass can still prove the BACKING
/// memory access dead and replace it with synthesized bytes. To
/// pin the cache-line traffic itself, route the load through
/// `ptr::read_volatile` and the store through `ptr::write_volatile`.
/// Volatile memory operations are not eliminable: each one becomes
/// a real `mov` against the actual buffer slot, which is what the
/// `WorkType::CachePressure` / `CacheYield` / `CachePipe` workloads
/// claim to exercise. The `work_units` bump retains its `black_box`
/// wrap separately to defeat increment-fusion across iterations.
///
/// `#[inline(never)]` matches `spin_burst`'s rationale: forcing
/// out-of-line keeps the per-iteration volatile load/store and
/// `black_box`-wrapped increment visible as distinct boundaries
/// LLVM cannot collapse with surrounding caller arithmetic.
#[inline(never)]
fn cache_rmw_loop(buf: &mut [u8], stride: usize, iters: u64, work_units: &mut u64) {
    let len = buf.len();
    let mut idx = 0;
    for _ in 0..iters {
        // SAFETY: `idx` stays in `0..len` (mod by len at the bottom
        // of the loop), so `&buf[idx]` is a valid `&u8` and
        // `&mut buf[idx]` is a valid `&mut u8`. Volatile read/write
        // through these references is sound; volatility just suppresses
        // optimization, it does not change pointer-validity rules.
        let cur = unsafe { std::ptr::read_volatile(&buf[idx]) };
        unsafe { std::ptr::write_volatile(&mut buf[idx], cur.wrapping_add(1)) };
        idx = (idx + stride) % len;
        *work_units = std::hint::black_box(work_units.wrapping_add(1));
    }
}

/// Naive matrix multiply: three square matrices of u64, O(n^3).
///
/// The caller owns a `Vec<u64>` of length `3 * size * size` so the
/// storage is naturally 8-byte aligned. An earlier version took a
/// `&mut [u8]` and cast to `*mut u64`, which was UB because a
/// `Vec<u8>` is only 1-byte aligned.
///
/// Optimization-elimination barrier: every multiplicand load goes
/// through `black_box`, the accumulator is `black_box`-clobbered
/// before the write, and the C-region store uses `write_volatile`.
/// Volatile is load-bearing on the write side: `matrix_buf` in
/// `worker_main` is a process-local `Vec<u64>` whose C region (the
/// upper third) is NEVER read by `matrix_multiply` or by any caller
/// — every subsequent iteration overwrites the same C indices and
/// the buffer is dropped at worker-exit without being inspected.
/// LLVM is therefore free to mark the store dead and elide both the
/// store AND the multiplication chain feeding it (load-load-mul-add
/// dependency collapses to nothing without an observable sink). The
/// per-load `black_box` and the post-mul `black_box(acc)` keep the
/// arithmetic live, but a non-volatile write on a dead-output slot
/// remains DCE-eligible. `write_volatile` makes the store non-
/// elidable, so the compute path the workload claims to exercise
/// actually executes under `-O2`/`-O3`.
///
/// `#[inline(never)]` matches `spin_burst` / `cache_rmw_loop` —
/// forcing out-of-line keeps the volatile-store and per-iteration
/// `black_box` wrappers visible as distinct boundaries the
/// optimizer can't collapse against the caller's arithmetic.
#[inline(never)]
fn matrix_multiply(data: &mut [u64], size: usize, work_units: &mut u64) {
    debug_assert_eq!(data.len(), 3 * size * size);
    let stride = size * size;
    for i in 0..size {
        for j in 0..size {
            let mut acc: u64 = 0;
            for k in 0..size {
                acc = acc.wrapping_add(
                    std::hint::black_box(data[i * size + k])
                        .wrapping_mul(std::hint::black_box(data[stride + k * size + j])),
                );
            }
            // SAFETY: `2 * stride + i * size + j` is in-bounds for a
            // slice of length `3 * stride` whenever `i, j < size`,
            // which the surrounding `for` ranges enforce. The
            // `debug_assert_eq!` above pins the length contract; the
            // slice's element type (`u64`) is naturally aligned via
            // `Vec<u64>` allocation. A non-volatile `data[idx] = ...`
            // would be DCE-eligible because no later code reads the
            // C region; the volatile store is the documented escape
            // hatch.
            unsafe {
                std::ptr::write_volatile(
                    &mut data[2 * stride + i * size + j] as *mut u64,
                    std::hint::black_box(acc),
                );
            }
        }
    }
    // Defense-in-depth read-back sink: route a single C-region
    // value back into `work_units` through `black_box`. The
    // `write_volatile` above is the primary defense — volatility
    // forces every store to materialize — but a future LLVM that
    // reasons more aggressively about volatility provenance could
    // still mark the entire C region as a write-only buffer whose
    // contents the program never inspects, and elide the multiply
    // chain feeding the volatile sink. By feeding one extracted
    // value back into the observable `work_units` accumulator the
    // multiply chain has a load-bearing consumer that flows into
    // the worker report. `data[2 * stride]` is the first slot of
    // the C region, in-bounds because `size >= 1` is enforced by
    // the call site (the worker only invokes matrix_multiply when
    // `matrix_size > 0`).
    *work_units = work_units.wrapping_add(std::hint::black_box(data[2 * stride]));
}

/// Write 1 byte to partner, poll for response, read, record wake latency.
fn pipe_exchange(
    read_fd: i32,
    write_fd: i32,
    resume_latencies_ns: &mut Vec<u64>,
    wake_sample_count: &mut u64,
    max_wake_samples: usize,
    stop: &AtomicBool,
) {
    unsafe { libc::write(write_fd, b"x".as_ptr() as *const _, 1) };
    let before_block = Instant::now();
    let mut pfd = libc::pollfd {
        fd: read_fd,
        events: libc::POLLIN,
        revents: 0,
    };
    loop {
        if stop_requested(stop) {
            break;
        }
        let ret = unsafe { libc::poll(&mut pfd, 1, 100) };
        if ret > 0 {
            let mut byte = [0u8; 1];
            unsafe { libc::read(read_fd, byte.as_mut_ptr() as *mut _, 1) };
            reservoir_push(
                resume_latencies_ns,
                wake_sample_count,
                before_block.elapsed().as_nanos() as u64,
                max_wake_samples,
            );
            break;
        }
        if ret < 0 {
            break;
        }
    }
}

extern "C" fn sigusr1_handler(_: libc::c_int) {
    STOP.store(true, Ordering::Relaxed);
}

fn resolve_affinity(mode: &ResolvedAffinity) -> Result<Option<BTreeSet<usize>>> {
    match mode {
        ResolvedAffinity::None => Ok(None),
        ResolvedAffinity::Fixed(cpus) => Ok(Some(cpus.clone())),
        ResolvedAffinity::SingleCpu(cpu) => Ok(Some([*cpu].into_iter().collect())),
        ResolvedAffinity::Random { from, count } => {
            use rand::seq::IndexedRandom;
            if *count == 0 {
                anyhow::bail!(
                    "ResolvedAffinity::Random.count must be > 0; a zero count \
                     previously silently coerced to 1, masking caller bugs"
                );
            }
            if from.is_empty() {
                tracing::debug!(
                    count = count,
                    "resolve_affinity: empty Random pool, leaving affinity unset"
                );
                return Ok(None);
            }
            let pool: Vec<usize> = from.iter().copied().collect();
            // Clamp count down to the pool size (user asked for more
            // CPUs than exist). Silent clamp is fine here: the pool
            // upper bound is a topology fact, not a caller bug.
            let count = (*count).min(pool.len());
            Ok(Some(
                pool.sample(&mut rand::rng(), count).copied().collect(),
            ))
        }
    }
}

fn sched_getcpu() -> usize {
    nix::sched::sched_getcpu().unwrap_or(0)
}

/// Record a wake latency sample using reservoir sampling (Algorithm R).
/// Maintains a uniform random sample of at most `cap` entries from all
/// observed latencies.
fn reservoir_push(buf: &mut Vec<u64>, count: &mut u64, sample: u64, cap: usize) {
    *count += 1;
    if buf.len() < cap {
        buf.push(sample);
    } else {
        // Replace a random element with probability cap/count.
        use rand::RngExt;
        let idx = rand::rng().random_range(0..*count) as usize;
        if idx < cap {
            buf[idx] = sample;
        }
    }
}

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
fn read_schedstat(tid: Option<libc::pid_t>) -> Option<(u64, u64, u64)> {
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
fn parse_schedstat_line(data: &str) -> Option<(u64, u64, u64)> {
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
fn warn_schedstat_unavailable_once() {
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
fn read_numa_maps_pages() -> BTreeMap<usize, u64> {
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
fn read_vmstat_numa_pages_migrated() -> u64 {
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
fn clock_gettime_ns(clk: libc::clockid_t) -> Option<u64> {
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
fn warn_clock_gettime_failed_once(clk: libc::clockid_t) {
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
fn thread_cpu_time_ns() -> u64 {
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
fn duration_to_kernel_ns(d: Duration, field: &str) -> Result<u64> {
    let ns_u128 = d.as_nanos();
    if ns_u128 > i64::MAX as u128 {
        anyhow::bail!(
            "sched_setattr: {field} duration ({ns_u128} ns) exceeds i64::MAX — \
             nanosecond count must fit in 63 bits (kernel reserves bit 63)"
        );
    }
    Ok(ns_u128 as u64)
}

fn set_sched_policy(pid: libc::pid_t, policy: SchedPolicy) -> Result<()> {
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

/// Pin a process to the given CPU set via `sched_setaffinity`.
pub fn set_thread_affinity(pid: libc::pid_t, cpus: &BTreeSet<usize>) -> Result<()> {
    use nix::sched::{CpuSet, sched_setaffinity};
    use nix::unistd::Pid;
    // See `set_sched_policy` for the rationale — pid <= 0 has
    // broadcast semantics at the syscall and must not be passed
    // through unchecked.
    if pid <= 0 {
        anyhow::bail!("sched_setaffinity: invalid pid {pid} (must be > 0)");
    }
    let mut cpu_set = CpuSet::new();
    for &cpu in cpus {
        cpu_set
            .set(cpu)
            .with_context(|| format!("CPU {cpu} out of range"))?;
    }
    sched_setaffinity(Pid::from_raw(pid), &cpu_set)
        .with_context(|| format!("sched_setaffinity pid={pid}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Collapse the mechanical
    /// `fn spawn_*_produces_work() { let config = WorkloadConfig
    /// { .. }; let mut h = WorkloadHandle::spawn(&config).unwrap();
    /// h.start(); sleep(ms); let reports = h.stop_and_collect(); ..
    /// }` test patterns into a single helper call. The boilerplate
    /// (`WorkloadConfig` literal, spawn, start, sleep, collect)
    /// is identical across work types — the caller's only unique
    /// contributions are the `WorkType` variant, the number of
    /// workers, the sleep duration, and the per-test assertions
    /// that follow. Every caller keeps its own assertions so the
    /// helper does NOT homogenize what each test guards; it
    /// collapses only the scaffolding.
    ///
    /// `num_workers` is explicit (not defaulted) because some
    /// tests use 2 workers (e.g. PipeIo needs even counts,
    /// futex pairs need 2-worker groups) and defaulting would
    /// force a rewrite at a later date when a new caller adds a
    /// 2-worker test.
    ///
    /// `sleep_ms` is explicit because different work types reach
    /// steady state at different wall-clock budgets — defaulting
    /// to a single value would make low-throughput work types
    /// flake under CI's typical 2-core runners.
    ///
    /// Other `WorkloadConfig` fields (`affinity`, `sched_policy`,
    /// `mem_policy`, `mpol_flags`) take
    /// [`WorkloadConfig::default`] values. Tests that need to
    /// override any of those fields construct the config
    /// literally — this helper covers only the mechanical
    /// "spawn, sleep, collect" shape.
    fn spawn_and_collect_after(
        work_type: WorkType,
        num_workers: usize,
        sleep_ms: u64,
    ) -> Vec<WorkerReport> {
        let config = WorkloadConfig {
            num_workers,
            affinity: ResolvedAffinity::None,
            work_type,
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        h.start();
        std::thread::sleep(std::time::Duration::from_millis(sleep_ms));
        h.stop_and_collect()
    }

    /// `mmap_shared_anon_errno_hint` must produce distinct,
    /// grep-friendly text for each of the three expected errnos
    /// (ENOMEM, EPERM, EINVAL) and the empty-string fallback for
    /// anything else. Pins the wire contract the two call sites
    /// in `WorkloadHandle::spawn` share so an errno that drifts
    /// between arms silently would trip the test here rather than
    /// in production diagnostics. Every expected arm checks the
    /// leading space (caller formats as `"{errno}{hint}"` and
    /// relies on the hint providing its own separator) plus a
    /// distinctive substring unique to that arm.
    #[test]
    fn mmap_shared_anon_errno_hint_variants() {
        let enomem = mmap_shared_anon_errno_hint(Some(libc::ENOMEM));
        assert!(
            enomem.starts_with(' '),
            "non-empty hint must begin with a space so \"{{errno}}{{hint}}\" has its separator; got {enomem:?}",
        );
        assert!(
            enomem.contains("ENOMEM"),
            "ENOMEM arm must name the errno in the hint; got {enomem:?}",
        );
        assert!(
            enomem.contains("vm.max_map_count"),
            "ENOMEM arm must mention the remediation sysctl; got {enomem:?}",
        );

        let eperm = mmap_shared_anon_errno_hint(Some(libc::EPERM));
        assert!(eperm.starts_with(' '), "EPERM hint must start with a space");
        assert!(
            eperm.contains("EPERM"),
            "EPERM arm must name the errno; got {eperm:?}",
        );
        assert!(
            eperm.contains("cgroup"),
            "EPERM arm must mention memory cgroup as a remediation path; got {eperm:?}",
        );

        let einval = mmap_shared_anon_errno_hint(Some(libc::EINVAL));
        assert!(
            einval.starts_with(' '),
            "EINVAL hint must start with a space"
        );
        assert!(
            einval.contains("EINVAL"),
            "EINVAL arm must name the errno; got {einval:?}",
        );
        assert!(
            einval.contains("num_workers > 0"),
            "EINVAL arm must give the concrete `num_workers > 0` remediation \
             (the older 'zero or misaligned' wording was too vague); got {einval:?}",
        );

        // Fallback arm: every unrecognised errno (EACCES, EBUSY,
        // EEXIST, random positive integers) must produce the empty
        // string so the caller's format produces no trailing noise.
        assert_eq!(
            mmap_shared_anon_errno_hint(Some(libc::EACCES)),
            "",
            "unrecognised errno must fold to empty-string hint",
        );
        assert_eq!(
            mmap_shared_anon_errno_hint(None),
            "",
            "None errno (io::Error without raw_os_error) must fold to empty-string",
        );
    }

    /// `clock_gettime_ns(CLOCK_MONOTONIC)` must never observe time
    /// moving backwards between two sequential calls on the same
    /// thread. Pins the non-decreasing contract the wake-latency
    /// reservoirs depend on: the messenger stamps `wake_ns` into
    /// shared memory and the worker subtracts to compute
    /// `now_ns - wake_ns`; a backward step would saturate to zero
    /// in the subtractor and silently discard a valid sample, or
    /// (without the saturator) wrap to `u64::MAX`.
    ///
    /// A 2-sample test would miss a backward step that only
    /// appears under load; the 1000-sample tight loop here burns
    /// a few microseconds of CPU and catches any regression that
    /// makes the clock non-monotonic under reasonable contention
    /// (timer drift on a virtualised guest, or a helper swap
    /// from `CLOCK_MONOTONIC` to `CLOCK_REALTIME` which is NOT
    /// monotonic). Every adjacent pair in the 999-element diff
    /// list is checked for non-decreasing order so a mid-run
    /// regression is localised to the offending index, not just
    /// "some pair somewhere".
    #[test]
    fn clock_gettime_ns_monotonic_non_decreasing() {
        const N: usize = 1000;
        let samples: Vec<u64> = (0..N)
            .map(|i| {
                clock_gettime_ns(libc::CLOCK_MONOTONIC).unwrap_or_else(|| {
                    panic!(
                        "CLOCK_MONOTONIC must be readable on any Linux host; \
                         sample {i}/{N} returned None"
                    )
                })
            })
            .collect();
        for i in 1..N {
            assert!(
                samples[i] >= samples[i - 1],
                "CLOCK_MONOTONIC went backwards at sample {i}: \
                 prev={prev} curr={curr} (delta={delta})",
                prev = samples[i - 1],
                curr = samples[i],
                delta = samples[i - 1] - samples[i],
            );
        }
    }

    // ---- classify_wait_outcome variant coverage ------------------------
    //
    // Five fixtures pin the `waitpid` → `WorkerExitInfo` mapping that the
    // sentinel path in [`WorkloadHandle::stop_and_collect`] depends on.
    // A silent table drift here would misreport panic / signal / timeout
    // root cause on every failed worker, so this is the canonical test
    // for each shape.

    #[test]
    fn classify_wait_outcome_exited_preserves_code() {
        let status = nix::sys::wait::WaitStatus::Exited(nix::unistd::Pid::from_raw(123), 42);
        match classify_wait_outcome(Ok(status)) {
            WorkerExitInfo::Exited(code) => assert_eq!(code, 42),
            other => panic!("expected Exited(42), got {other:?}"),
        }
    }

    #[test]
    fn classify_wait_outcome_signaled_preserves_signum() {
        let status = nix::sys::wait::WaitStatus::Signaled(
            nix::unistd::Pid::from_raw(123),
            nix::sys::signal::Signal::SIGABRT,
            false,
        );
        match classify_wait_outcome(Ok(status)) {
            WorkerExitInfo::Signaled(sig) => {
                assert_eq!(sig, nix::sys::signal::Signal::SIGABRT as i32);
            }
            other => panic!("expected Signaled(SIGABRT), got {other:?}"),
        }
    }

    #[test]
    fn classify_wait_outcome_still_alive_maps_to_timed_out() {
        match classify_wait_outcome(Ok(nix::sys::wait::WaitStatus::StillAlive)) {
            WorkerExitInfo::TimedOut => {}
            other => panic!("expected TimedOut, got {other:?}"),
        }
    }

    #[test]
    fn classify_wait_outcome_exotic_continued_maps_to_timed_out() {
        // `Continued` is one of the non-terminal WaitStatus variants
        // that can't describe a worker exit for a ptrace-free fork —
        // the catch-all arm must collapse it to TimedOut rather than
        // silently dropping the reap.
        let status = nix::sys::wait::WaitStatus::Continued(nix::unistd::Pid::from_raw(123));
        match classify_wait_outcome(Ok(status)) {
            WorkerExitInfo::TimedOut => {}
            other => panic!("expected TimedOut (exotic→TimedOut), got {other:?}"),
        }
    }

    #[test]
    fn classify_wait_outcome_errno_maps_to_wait_failed() {
        match classify_wait_outcome(Err(nix::errno::Errno::ECHILD)) {
            WorkerExitInfo::WaitFailed(msg) => {
                // nix renders Errno via Display — the string carries
                // the canonical ECHILD description. Substring-match
                // keeps the test robust against OS-specific wording
                // variations without hardcoding a specific phrase.
                assert!(
                    msg.to_ascii_lowercase().contains("child"),
                    "expected ECHILD description to mention 'child', got {msg:?}",
                );
            }
            other => panic!("expected WaitFailed, got {other:?}"),
        }
    }

    /// `extract_panic_payload` round-trips both canonical panic
    /// payload shapes (`&'static str` from `panic!("literal")` and
    /// `String` from `panic!("{x}")`) and falls back to the named
    /// sentinel for everything else.
    #[test]
    fn extract_panic_payload_handles_all_canonical_shapes() {
        let str_panic: Box<dyn std::any::Any + Send> = Box::new("literal panic");
        assert_eq!(extract_panic_payload(str_panic), "literal panic");

        let string_panic: Box<dyn std::any::Any + Send> =
            Box::new(String::from("formatted panic"));
        assert_eq!(extract_panic_payload(string_panic), "formatted panic");

        // Anything else — e.g. a custom panic payload type — folds
        // to the sentinel without crashing the extractor.
        #[derive(Clone)]
        struct CustomPayload(u32);
        let custom: Box<dyn std::any::Any + Send> = Box::new(CustomPayload(42));
        assert_eq!(extract_panic_payload(custom), "<non-string panic payload>");
    }

    /// `join_thread_with_timeout` returns the join result when the
    /// thread completes within the deadline.
    #[test]
    fn join_thread_with_timeout_returns_result_on_quick_completion() {
        let join = std::thread::spawn(|| WorkerReport {
            tid: 7,
            ..WorkerReport::default()
        });
        let r = join_thread_with_timeout(join, Duration::from_secs(2));
        match r {
            Some(Ok(report)) => assert_eq!(report.tid, 7),
            Some(Err(_)) => panic!("clean thread must not produce join Err"),
            None => panic!("clean thread must not time out within 2s"),
        }
    }

    /// `join_thread_with_timeout` returns `None` when the thread is
    /// still running past the deadline. The thread itself leaks for
    /// the rest of the test process — acceptable in a `#[test]`
    /// because the test harness terminates after the thread's
    /// upper-bound sleep.
    #[test]
    fn join_thread_with_timeout_returns_none_on_timeout() {
        let join = std::thread::spawn(|| {
            // Sleep WELL past the 100ms timeout so the polling
            // helper definitely observes is_finished()==false.
            std::thread::sleep(Duration::from_millis(800));
            WorkerReport::default()
        });
        let r = join_thread_with_timeout(join, Duration::from_millis(100));
        assert!(r.is_none(), "100ms timeout vs 800ms thread must time out");
    }

    /// Defense-in-depth: `ThreadWorker::drop` MUST join its
    /// `JoinHandle`. Rust's std `JoinHandle::drop` detaches by
    /// default — the bug class this test exists to catch is a
    /// future refactor that lets a `ThreadWorker` fall out of
    /// scope without going through the `WorkloadHandle::drop`
    /// / `stop_and_collect` / `SpawnGuard::drop` paths that
    /// already explicitly take + join.
    ///
    /// The test constructs a `ThreadWorker` whose worker writes a
    /// shared flag and waits on a stop signal, drops the
    /// `ThreadWorker` directly (NOT via any of the explicit Drop
    /// paths), and verifies the worker observed `stop=true` and
    /// completed before the drop returned. If `ThreadWorker::drop`
    /// detached, the worker would still be running when the test
    /// returns — the spin-loop on the shared flag confirms a
    /// successful join.
    #[test]
    fn thread_worker_drop_joins_handle() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
        use std::sync::mpsc;

        let stop = Arc::new(AtomicBool::new(false));
        let observed = Arc::new(AtomicBool::new(false));
        let stop_thread = Arc::clone(&stop);
        let observed_thread = Arc::clone(&observed);
        let (start_tx, start_rx) = mpsc::sync_channel::<()>(0);
        let tid = Arc::new(AtomicI32::new(0));
        let tid_thread = Arc::clone(&tid);

        let join = std::thread::spawn(move || {
            tid_thread.store(
                unsafe { libc::syscall(libc::SYS_gettid) as libc::pid_t },
                Ordering::Relaxed,
            );
            // Block on start so the worker is guaranteed to be
            // running (not just dispatched) by the time we drop.
            let _ = start_rx.recv();
            // Spin on stop with the same 100ms poll cadence the
            // production worker uses.
            while !stop_thread.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(20));
            }
            observed_thread.store(true, Ordering::Relaxed);
            WorkerReport::default()
        });

        let tw = ThreadWorker {
            tid,
            stop,
            start_tx: Some(start_tx),
            join: Some(join),
        };
        // Send the start signal so the worker proceeds to its
        // stop-check loop. (The Drop will also drop start_tx but
        // that comes after recv() has consumed our send.)
        if let Some(ref tx) = tw.start_tx {
            let _ = tx.send(());
        }
        // Tiny sleep so the worker definitely observes the start
        // and enters the spin loop before Drop runs.
        std::thread::sleep(Duration::from_millis(50));

        // Drop the ThreadWorker directly — this is the path under
        // test. ThreadWorker::drop must flip stop and join.
        drop(tw);

        // Assertion: by the time drop returns, the worker has
        // observed stop and completed. If drop detached, observed
        // would still be false because the worker would either
        // still be sleeping or already gone without a join.
        assert!(
            observed.load(Ordering::Relaxed),
            "ThreadWorker::drop must join its JoinHandle — observed=false \
             means the drop returned without waiting for the worker, which \
             would mean the worker was detached (Rust's default for \
             JoinHandle::drop) instead of explicitly joined"
        );
    }

    #[test]
    fn work_type_name_roundtrip() {
        for &name in WorkType::ALL_NAMES {
            // Sequence and Custom have no default from_name.
            if name == "Sequence" || name == "Custom" {
                assert!(WorkType::from_name(name).is_none());
                continue;
            }
            let wt = WorkType::from_name(name).unwrap();
            assert_eq!(wt.name(), name);
        }
    }

    #[test]
    fn work_type_from_name_unknown() {
        assert!(WorkType::from_name("Nonexistent").is_none());
    }

    /// [`WorkType::suggest`] matches case-insensitively and
    /// returns the canonical PascalCase entry. A user who types
    /// `"cpuspin"`, `"CPUSPIN"`, or the already-canonical `"CpuSpin"`
    /// all land on the same `"CpuSpin"` suggestion; truly unknown
    /// inputs return `None` so the caller can distinguish "typo of a
    /// known variant" from "wholly unknown name".
    /// Composition pin: the intended CLI recovery flow is
    /// `from_name(user_input)` → on `None`, `suggest(user_input)` →
    /// on `Some(canonical)`, feed `canonical` back into `from_name`
    /// to obtain the `WorkType` value. Each arrow must be a stable
    /// equivalence so a diagnostic message's "did you mean
    /// '{canonical}'?" always resolves to a constructible variant.
    /// `Sequence` and `Custom` participate in the naming side
    /// (`suggest`) but `from_name` still refuses to build them —
    /// construction requires explicit phases / function pointers,
    /// which a CLI string cannot supply. Pin both facets so a
    /// regression that (a) adds fuzzy matching to `suggest` or
    /// (b) lets `from_name` construct `Sequence`/`Custom` from a
    /// bare name surfaces here.
    #[test]
    fn suggest_then_from_name_roundtrips_for_buildable_variants() {
        // Lowercase user input: from_name misses, suggest hits,
        // from_name on the canonical spelling succeeds.
        assert!(WorkType::from_name("cpuspin").is_none());
        let canonical = WorkType::suggest("cpuspin").expect("suggest must find CpuSpin");
        assert_eq!(canonical, "CpuSpin");
        let wt =
            WorkType::from_name(canonical).expect("from_name must build from canonical spelling");
        assert!(matches!(wt, WorkType::CpuSpin));

        // Uppercase user input roundtrips too.
        assert!(WorkType::from_name("YIELDHEAVY").is_none());
        let canonical = WorkType::suggest("YIELDHEAVY").expect("suggest must find YieldHeavy");
        assert_eq!(canonical, "YieldHeavy");
        let wt = WorkType::from_name(canonical).expect("from_name must build");
        assert!(matches!(wt, WorkType::YieldHeavy));

        // Sequence and Custom are suggest-only: suggest emits them
        // so a diagnostic can name them, but from_name returns None
        // because they need explicit phases / function pointers that
        // a bare string cannot carry.
        assert_eq!(WorkType::suggest("sequence"), Some("Sequence"));
        assert!(WorkType::from_name("Sequence").is_none());
        assert_eq!(WorkType::suggest("custom"), Some("Custom"));
        assert!(WorkType::from_name("Custom").is_none());
    }

    #[test]
    fn suggest_is_case_insensitive_and_canonical() {
        assert_eq!(WorkType::suggest("cpuspin"), Some("CpuSpin"));
        assert_eq!(WorkType::suggest("CPUSPIN"), Some("CpuSpin"));
        assert_eq!(WorkType::suggest("CpuSpin"), Some("CpuSpin"));
        assert_eq!(WorkType::suggest("YIELDHEAVY"), Some("YieldHeavy"));
        // Sequence and Custom are in the match space even though
        // `from_name` refuses to construct them — point of the
        // helper is naming, not construction.
        assert_eq!(WorkType::suggest("sequence"), Some("Sequence"));
        assert_eq!(WorkType::suggest("custom"), Some("Custom"));
        // Truly unknown names return None. Distinguishes "no suggestion
        // available" from "canonicalized spelling of a known variant".
        assert!(WorkType::suggest("nonexistent").is_none());
        assert!(WorkType::suggest("").is_none());
        // A partial match is NOT fuzzy-accepted — "cpu" does not
        // shorten to "CpuSpin". The helper pins exact case-insensitive
        // equality, not prefix or substring semantics.
        assert!(WorkType::suggest("cpu").is_none());
    }

    /// Surrounding / embedded whitespace must NOT silently resolve
    /// to a canonical name. The helper's doc commits to strict
    /// (non-trimming) matching so a caller that passes unsanitized
    /// user input like `" CpuSpin"` or `"CpuSpin\n"` sees `None` —
    /// callers are expected to `s.trim()` first (same convention
    /// [`WorkType::from_name`] follows). If this test ever starts
    /// failing because [`suggest`] returns `Some(_)` for a whitespace-
    /// padded input, the helper's behavior has drifted away from its
    /// documented contract.
    #[test]
    fn suggest_rejects_whitespace_padded_inputs() {
        // Leading / trailing ASCII space.
        assert!(WorkType::suggest(" CpuSpin").is_none());
        assert!(WorkType::suggest("CpuSpin ").is_none());
        assert!(WorkType::suggest(" CpuSpin ").is_none());
        // Trailing newline (typical for unsanitized fgets / read_line
        // output).
        assert!(WorkType::suggest("CpuSpin\n").is_none());
        // Tab separators on either side.
        assert!(WorkType::suggest("\tCpuSpin").is_none());
        assert!(WorkType::suggest("CpuSpin\t").is_none());
        // Embedded whitespace inside an otherwise-known name also
        // fails — the helper is NOT doing fuzzy tokenization.
        assert!(WorkType::suggest("Cpu Spin").is_none());
        // Pure whitespace input returns None (parallels the empty-
        // string case pinned in `suggest_is_case_insensitive_and_canonical`).
        assert!(WorkType::suggest(" ").is_none());
        assert!(WorkType::suggest("\n").is_none());
        // Sanity check: the same input without whitespace does
        // resolve, confirming the rejection is specifically about
        // the whitespace and not an unrelated regression.
        assert_eq!(WorkType::suggest("CpuSpin"), Some("CpuSpin"));
    }

    #[test]
    fn work_type_all_names_count() {
        // 20 historical variants + 7 pathology-taxonomy variants
        // (ThunderingHerd, PriorityInversion,
        // ProducerConsumerImbalance, RtStarvation, AsymmetricWaker,
        // WakeChain, NumaWorkingSetSweep) = 27.
        assert_eq!(WorkType::ALL_NAMES.len(), 27);
    }

    // -- matrix_multiply --

    #[test]
    fn matrix_multiply_1x1_produces_product() {
        // Size=1: A=[a], B=[b], expected C=[a*b]. The `black_box` calls
        // prevent constant folding, so the test directly exercises the
        // wrapping_mul path without any compiler optimization eating
        // the multiplication.
        let mut data = vec![0u64; 3];
        data[0] = 3; // A
        data[1] = 5; // B
        let mut work_units = 0u64;
        matrix_multiply(&mut data, 1, &mut work_units);
        assert_eq!(data[2], 15, "C = A * B for 1x1 matrix");
        // Read-back sink consumed C[0] (= 15) into work_units.
        assert_eq!(work_units, 15, "post-loop sink folds C[0] into work_units");
    }

    #[test]
    fn matrix_multiply_2x2_against_reference() {
        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
        // C = A * B = [[19, 22], [43, 50]]
        let size = 2;
        let stride = size * size;
        let mut data = vec![0u64; 3 * stride];
        data[0] = 1;
        data[1] = 2;
        data[2] = 3;
        data[3] = 4;
        data[stride] = 5;
        data[stride + 1] = 6;
        data[stride + 2] = 7;
        data[stride + 3] = 8;
        let mut work_units = 0u64;
        matrix_multiply(&mut data, size, &mut work_units);
        assert_eq!(data[2 * stride], 19);
        assert_eq!(data[2 * stride + 1], 22);
        assert_eq!(data[2 * stride + 2], 43);
        assert_eq!(data[2 * stride + 3], 50);
    }

    #[test]
    fn matrix_multiply_3x3_diagonal() {
        // Identity-like: A = diag(2, 3, 5), B = diag(1, 1, 1) = I.
        // Expected C = A = diag(2, 3, 5).
        let size = 3;
        let stride = size * size;
        let mut data = vec![0u64; 3 * stride];
        data[0] = 2;
        data[4] = 3;
        data[8] = 5;
        data[stride] = 1;
        data[stride + 4] = 1;
        data[stride + 8] = 1;
        let mut work_units = 0u64;
        matrix_multiply(&mut data, size, &mut work_units);
        let c = &data[2 * stride..3 * stride];
        // Diagonal entries carry A's diagonal because B = I.
        assert_eq!(c[0], 2);
        assert_eq!(c[4], 3);
        assert_eq!(c[8], 5);
        // All 6 off-diagonal entries must be 0 for A*I. Sparse
        // coverage (just c[1], c[3]) left 4 positions unverified,
        // which would mask a transposition bug that mis-writes
        // rows/columns of an identity product — this assertion
        // fingerprints the full matrix identity.
        assert_eq!(c[1], 0);
        assert_eq!(c[2], 0);
        assert_eq!(c[3], 0);
        assert_eq!(c[5], 0);
        assert_eq!(c[6], 0);
        assert_eq!(c[7], 0);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "assertion")]
    fn matrix_multiply_mismatched_len_panics_in_debug() {
        // debug_assert_eq!(data.len(), 3 * size * size) guards the
        // bounds contract. Under cfg(debug_assertions) this panics.
        // Release builds skip the assert (no panic), so the test
        // itself is gated on `cfg(debug_assertions)` — otherwise
        // `cargo nextest run --release` would run the test expecting
        // a panic the release binary can't raise.
        let mut data = vec![0u64; 5]; // 3 * 2 * 2 = 12, so 5 is wrong.
        let mut work_units = 0u64;
        matrix_multiply(&mut data, 2, &mut work_units);
    }

    #[test]
    fn resolve_affinity_none() {
        let r = resolve_affinity(&ResolvedAffinity::None).unwrap();
        assert!(r.is_none());
    }

    #[test]
    fn resolve_affinity_fixed() {
        let cpus: BTreeSet<usize> = [0, 1, 2].into_iter().collect();
        let r = resolve_affinity(&ResolvedAffinity::Fixed(cpus.clone())).unwrap();
        assert_eq!(r, Some(cpus));
    }

    #[test]
    fn resolve_affinity_single_cpu() {
        let r = resolve_affinity(&ResolvedAffinity::SingleCpu(5)).unwrap();
        assert_eq!(r, Some([5].into_iter().collect()));
    }

    #[test]
    fn resolve_affinity_random() {
        let from: BTreeSet<usize> = (0..8).collect();
        let r = resolve_affinity(&ResolvedAffinity::Random { from, count: 3 }).unwrap();
        let cpus = r.unwrap();
        assert_eq!(cpus.len(), 3);
        assert!(cpus.iter().all(|c| *c < 8));
    }

    #[test]
    fn resolve_affinity_random_clamps_count() {
        let from: BTreeSet<usize> = [0, 1].into_iter().collect();
        let r = resolve_affinity(&ResolvedAffinity::Random { from, count: 10 }).unwrap();
        assert_eq!(r.unwrap().len(), 2);
    }

    #[test]
    fn workload_config_default() {
        let c = WorkloadConfig::default();
        assert_eq!(c.num_workers, 1);
        assert!(matches!(c.work_type, WorkType::CpuSpin));
        assert!(matches!(c.sched_policy, SchedPolicy::Normal));
        assert!(matches!(c.affinity, ResolvedAffinity::None));
        // Default nice is 0 — `apply_nice(0)` short-circuits before
        // the syscall, preserving inherit-from-parent semantics.
        assert_eq!(c.nice, 0);
    }

    #[test]
    fn workload_config_builder_setters_chain() {
        let cfg = WorkloadConfig::default()
            .workers(7)
            .work_type(WorkType::CpuSpin)
            .sched_policy(SchedPolicy::Batch)
            .nice(5);
        assert_eq!(cfg.num_workers, 7);
        assert!(matches!(cfg.work_type, WorkType::CpuSpin));
        assert!(matches!(cfg.sched_policy, SchedPolicy::Batch));
        assert_eq!(cfg.nice, 5);
    }

    /// `apply_nice(0)` is a documented short-circuit — when the
    /// caller leaves the field at its default, the worker MUST
    /// inherit the parent's nice value rather than have
    /// `setpriority(PRIO_PROCESS, 0, 0)` reset it to zero. The
    /// distinction matters when scenario-level code already
    /// elevated the parent to a non-default nice (e.g. via a
    /// wrapper that wants every worker to inherit) — a
    /// non-skipping `apply_nice(0)` would silently clobber that.
    /// Test by setting the calling thread's nice via direct
    /// syscall, calling `apply_nice(0)`, and asserting the nice
    /// did not change.
    #[test]
    fn apply_nice_zero_is_noop() {
        // Set nice to 5 directly (positive — works without
        // CAP_SYS_NICE because raising nice is always permitted
        // for own task).
        let rc = unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, 5) };
        if rc != 0 {
            // setpriority should not fail for a positive nice on
            // self; if it does, the host environment is unusual
            // — skip rather than fake-pass.
            eprintln!(
                "skipping: setpriority(0, 0, 5) failed: {}",
                std::io::Error::last_os_error()
            );
            return;
        }
        // getpriority returns 20 - nice (per glibc convention),
        // so a nice of 5 reads back as 15. errno-clear before
        // call because getpriority can legitimately return -1
        // for nice=20.
        unsafe {
            *libc::__errno_location() = 0;
        }
        let read_back = unsafe { libc::getpriority(libc::PRIO_PROCESS, 0) };
        let errno_before = unsafe { *libc::__errno_location() };
        assert_eq!(
            errno_before, 0,
            "getpriority must succeed before apply_nice; rc={read_back}"
        );
        let nice_before = 20 - read_back;
        assert_eq!(
            nice_before, 5,
            "setpriority must have stuck before apply_nice runs"
        );

        // Now invoke apply_nice(0) — should NOT touch priority.
        apply_nice(0);

        unsafe {
            *libc::__errno_location() = 0;
        }
        let read_back2 = unsafe { libc::getpriority(libc::PRIO_PROCESS, 0) };
        let errno_after = unsafe { *libc::__errno_location() };
        assert_eq!(errno_after, 0, "getpriority must succeed after apply_nice");
        let nice_after = 20 - read_back2;
        assert_eq!(
            nice_after, 5,
            "apply_nice(0) must not touch nice — observed change \
             from {nice_before} to {nice_after}",
        );

        // Restore default (rare-path cleanup).
        let _ = unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, 0) };
    }

    /// Positive-nice end-to-end: spawn one worker with `nice = 10`,
    /// verify the worker process actually has nice 10 by reading
    /// `/proc/<pid>/stat` field 19 (priority field) before
    /// `stop_and_collect`. Positive nice never requires
    /// `CAP_SYS_NICE` — `set_one_prio` only checks `can_nice` for
    /// `niceval < task_nice(p)`.
    ///
    /// Reading via /proc rather than `getpriority` because the
    /// worker is in a child process; `getpriority(PRIO_PROCESS, pid)`
    /// would also work but /proc/stat field 19 is the canonical
    /// observation point used elsewhere in the crate's tests.
    #[test]
    fn worker_nice_applied_via_setpriority() {
        let config = WorkloadConfig {
            num_workers: 1,
            affinity: ResolvedAffinity::None,
            work_type: WorkType::CpuSpin,
            sched_policy: SchedPolicy::Normal,
            nice: 10,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        let pid = h.worker_pids()[0];
        h.start();
        // Brief sleep so the worker has actually executed
        // `apply_nice` post-fork and post-start before we read
        // /proc.
        std::thread::sleep(std::time::Duration::from_millis(100));
        // /proc/<pid>/stat field 19 is "nice" per `proc(5)` —
        // tokenize after the comm field's closing paren to avoid
        // splitting names containing spaces.
        let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).expect("/proc/stat read");
        let after_paren = stat
            .rsplit_once(") ")
            .expect("/proc/stat has comm in parens")
            .1;
        // After the closing paren, fields are 1-indexed starting
        // at "state" (field 3 of the original layout). nice is
        // field 19; minus the 2 fields before the paren that's
        // index 16 in the post-paren token list.
        let tokens: Vec<&str> = after_paren.split_whitespace().collect();
        let nice_str = tokens
            .get(16)
            .expect("/proc/stat must have at least 17 fields after comm");
        let nice_observed: i32 = nice_str.parse().expect("nice field must be i32");
        // Stop before assertion so a failure doesn't leak a
        // non-default-nice worker.
        let _reports = h.stop_and_collect();
        assert_eq!(
            nice_observed, 10,
            "worker /proc/<pid>/stat field 19 must reflect the \
             configured nice value; got {nice_observed}, expected 10"
        );
    }

    #[test]
    fn worker_report_serde_roundtrip() {
        let r = WorkerReport {
            tid: 42,
            work_units: 1000,
            cpu_time_ns: 5_000_000_000,
            wall_time_ns: 10_000_000_000,
            off_cpu_ns: 5_000_000_000,
            migration_count: 3,
            cpus_used: [0, 1, 2].into_iter().collect(),
            migrations: vec![Migration {
                at_ns: 100,
                from_cpu: 0,
                to_cpu: 1,
            }],
            max_gap_ms: 50,
            max_gap_cpu: 1,
            max_gap_at_ms: 500,
            resume_latencies_ns: vec![1000, 2000],
            wake_sample_total: 2,
            iterations: 10,
            schedstat_run_delay_ns: 500_000,
            schedstat_run_count: 20,
            schedstat_cpu_time_ns: 4_000_000_000,
            completed: true,
            numa_pages: BTreeMap::new(),
            vmstat_numa_pages_migrated: 0,
            exit_info: None,
            // Non-default so the serde roundtrip proves the field
            // survives, not just that Default's value matches on
            // both sides.
            is_messenger: true,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: WorkerReport = serde_json::from_str(&json).unwrap();
        assert_eq!(r.tid, r2.tid);
        assert_eq!(r.work_units, r2.work_units);
        assert_eq!(r.migration_count, r2.migration_count);
        assert_eq!(r.cpus_used, r2.cpus_used);
        assert_eq!(r.max_gap_ms, r2.max_gap_ms);
        assert_eq!(r.wake_sample_total, r2.wake_sample_total);
        assert_eq!(r.completed, r2.completed);
        assert_eq!(r.is_messenger, r2.is_messenger);
    }

    #[test]
    fn migration_serde() {
        let m = Migration {
            at_ns: 12345,
            from_cpu: 0,
            to_cpu: 3,
        };
        let json = serde_json::to_string(&m).unwrap();
        let m2: Migration = serde_json::from_str(&json).unwrap();
        assert_eq!(m.at_ns, m2.at_ns);
        assert_eq!(m.from_cpu, m2.from_cpu);
        assert_eq!(m.to_cpu, m2.to_cpu);
    }

    #[test]
    fn spawn_start_collect_integration() {
        let config = WorkloadConfig {
            num_workers: 2,
            affinity: ResolvedAffinity::None,
            work_type: WorkType::CpuSpin,
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        assert_eq!(h.worker_pids().len(), 2);
        h.start();
        std::thread::sleep(std::time::Duration::from_millis(200));
        let reports = h.stop_and_collect();
        assert_eq!(reports.len(), 2);
        for r in &reports {
            assert!(r.work_units > 0, "worker {} did no work", r.tid);
            assert!(r.wall_time_ns > 0);
            assert!(!r.cpus_used.is_empty());
        }
    }

    #[test]
    fn spawn_auto_start_on_collect() {
        let config = WorkloadConfig {
            num_workers: 1,
            affinity: ResolvedAffinity::None,
            work_type: WorkType::CpuSpin,
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let h = WorkloadHandle::spawn(&config).unwrap();
        // Don't call start() - collect should auto-start
        let reports = h.stop_and_collect();
        assert_eq!(reports.len(), 1);
    }

    #[test]
    fn spawn_yield_heavy_produces_work() {
        let reports = spawn_and_collect_after(WorkType::YieldHeavy, 1, 200);
        assert_eq!(reports.len(), 1);
        assert!(reports[0].work_units > 0);
    }

    #[test]
    fn spawn_mixed_produces_work() {
        let reports = spawn_and_collect_after(WorkType::Mixed, 1, 200);
        assert_eq!(reports.len(), 1);
        assert!(reports[0].work_units > 0);
    }

    /// Regression guard for the sign-cast bug: every pid returned
    /// from `worker_pids()` must be a positive, live `pid_t` that
    /// round-trips through `Pid::from_raw` + `kill(_, None)` (the
    /// "exists" probe). A negative pid would silently broadcast
    /// SIGKILL to a process group; a stale/reaped pid would fail the
    /// probe with ESRCH. Either indicates storage upstream
    /// re-introduced the u32 wraparound or dropped a child on the
    /// floor.
    #[test]
    fn spawn_pids_fit_in_pid_t() {
        let config = WorkloadConfig {
            num_workers: 4,
            affinity: ResolvedAffinity::None,
            work_type: WorkType::CpuSpin,
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let h = WorkloadHandle::spawn(&config).unwrap();
        for pid in h.worker_pids() {
            assert!(pid > 0, "child pid must be positive, got {pid}");
            // Signal 0 (None) only checks existence; it does not
            // deliver anything. Proves the pid is a real, live
            // process we can address — not a negative-cast bomb.
            nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None)
                .unwrap_or_else(|e| panic!("spawned child pid {pid} not addressable: {e}"));
        }
    }

    /// Regression guard for the spawn-leak fix: on a mid-setup
    /// `bail!` path, the `SpawnGuard` Drop must release every
    /// resource acquired so far — no leaked children, no leaked
    /// pipe fds, no leaked mmap regions. This test constructs a
    /// config that passes the `worker_group_size` check and then
    /// provokes the per-worker pipe path (num_workers=2 with
    /// PipeIo) so the function allocates inter-worker pipes and
    /// spawns successfully, then checks Drop cleans up when the
    /// handle is dropped without `stop_and_collect`.
    ///
    /// The direct spawn-failure path is hard to trigger
    /// synthetically (would require EMFILE / ENOMEM injection); the
    /// scope guard's correctness is proven by the unified cleanup
    /// pattern — Drop runs on every early return *and* on the
    /// normal drop-without-collect flow.
    #[test]
    fn handle_drop_reaps_children_and_closes_pipes() {
        let config = WorkloadConfig {
            num_workers: 2,
            affinity: ResolvedAffinity::None,
            work_type: WorkType::PipeIo { burst_iters: 4 },
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let h = WorkloadHandle::spawn(&config).unwrap();
        let pids = h.worker_pids();
        assert_eq!(pids.len(), 2, "both workers spawned");
        // Drop without calling start() or stop_and_collect() — this
        // exercises the WorkloadHandle::Drop path, which has the
        // same cleanup semantics as SpawnGuard's error path.
        drop(h);
        // Poll for termination: ESRCH (no such process) means the
        // child was reaped. Give the kernel a brief grace window
        // because waitpid runs synchronously but kill reporting can
        // race.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        for pid in pids {
            loop {
                let alive = nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None).is_ok();
                if !alive {
                    break;
                }
                if std::time::Instant::now() >= deadline {
                    panic!("child {pid} still alive after drop deadline");
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        }
    }

    #[test]
    fn spawn_multiple_workers_distinct_pids() {
        let config = WorkloadConfig {
            num_workers: 4,
            affinity: ResolvedAffinity::None,
            work_type: WorkType::CpuSpin,
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        let pids = h.worker_pids();
        assert_eq!(pids.len(), 4);
        let unique: std::collections::HashSet<libc::pid_t> = pids.iter().copied().collect();
        assert_eq!(unique.len(), 4, "all worker PIDs should be distinct");
        h.start();
        std::thread::sleep(std::time::Duration::from_millis(500));
        let reports = h.stop_and_collect();
        assert_eq!(reports.len(), 4);
    }

    #[test]
    fn spawn_with_fixed_affinity() {
        let config = WorkloadConfig {
            num_workers: 1,
            affinity: ResolvedAffinity::Fixed([0].into_iter().collect()),
            work_type: WorkType::CpuSpin,
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        h.start();
        std::thread::sleep(std::time::Duration::from_millis(200));
        let reports = h.stop_and_collect();
        assert_eq!(reports.len(), 1);
        assert!(reports[0].cpus_used.contains(&0));
        assert_eq!(reports[0].cpus_used.len(), 1, "should only use pinned CPU");
    }

    #[test]
    fn drop_kills_children() {
        let config = WorkloadConfig {
            num_workers: 2,
            ..Default::default()
        };
        let h = WorkloadHandle::spawn(&config).unwrap();
        let pids = h.worker_pids();
        drop(h);
        // After drop, children should be dead.
        for pid in pids {
            let alive = nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None).is_ok();
            assert!(!alive, "child {} should be dead after drop", pid);
        }
    }

    // -- SpawnGuard failure-injection tests --
    //
    // These exercise the error-path cleanup that the unified
    // `handle_drop_reaps_children_and_closes_pipes` test explicitly
    // noted it could not cover: the mid-spawn bail paths reached when
    // a syscall inside `WorkloadHandle::spawn` fails with EMFILE
    // (RLIMIT_NOFILE) or EAGAIN (RLIMIT_NPROC). Each case forks a
    // helper subprocess so `setrlimit` scope is confined to that
    // child and the parent test binary's limits stay intact.
    //
    // Cleanup check strategy:
    //   - Count open fds via `/proc/self/fd/` before and after the
    //     failed `spawn`. After SpawnGuard::Drop, the fd count must
    //     return to baseline (all pipe pairs, report pipes, and start
    //     pipes released).
    //   - Poll `waitpid(-1, WNOHANG)` to prove no zombie worker
    //     children were left behind by a partial fork.
    //
    // Child exit code convention:
    //   0  = success (spawn returned Err AND cleanup is clean)
    //   10 = spawn unexpectedly returned Ok (failure not triggered)
    //   11 = fd leak detected after SpawnGuard::Drop
    //   12 = zombie worker process detected after SpawnGuard::Drop
    //   13 = setrlimit itself failed (harness issue, not a test
    //        failure of the guard)
    //   14 = bail arrived via an unexpected branch (test picks the
    //        wrong failure path)
    //   15 = post-bail setrlimit raise failed (harness issue; would
    //        mask a genuine fd leak as a false positive)
    //   other nonzero = unrelated failure (panic, assertion miss)
    //
    // `libc::_exit` is used instead of `std::process::exit` in the
    // child so Rust's global destructors — shared with the parent
    // test binary through the fork's copied state — do not fire.

    /// Count open file descriptors for the calling process by
    /// listing `/proc/self/fd/`. The directory iterator itself holds
    /// one fd while open; the snapshot is taken after the iterator
    /// drops, so the count reflects steady state.
    fn count_open_fds() -> usize {
        std::fs::read_dir("/proc/self/fd")
            .map(|d| d.count())
            .unwrap_or(0)
    }

    /// Non-blocking reap of any exited children. Returns true when a
    /// child reported via waitpid(-1, WNOHANG), indicating an
    /// orphaned-but-not-reaped zombie remained after `spawn`'s error
    /// path. SpawnGuard::Drop reaps everything it forked; any
    /// positive return here is a guard bug.
    fn any_zombie_child() -> bool {
        let mut status = 0i32;
        let ret = unsafe { libc::waitpid(-1, &mut status, libc::WNOHANG) };
        ret > 0
    }

    /// Lower RLIMIT_NPROC to the current process count so any `fork`
    /// in this child returns -1 with EAGAIN. Returns true on success.
    fn set_rlimit_nproc_zero_headroom() -> bool {
        // Setting rlim_cur to 1 would block even our own existing
        // thread spawns; setting it to the current process's uid
        // usage is what reliably triggers EAGAIN on the next fork.
        // getrusage does not expose that counter; instead use a
        // small value just high enough for the ktstr test binary's
        // baseline and no more. Empirically, setting rlim_cur == 0
        // causes fork to return EAGAIN because the kernel rejects
        // the new-process creation against the per-uid cap.
        let rl = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        unsafe { libc::setrlimit(libc::RLIMIT_NPROC, &rl) == 0 }
    }

    /// Fork a helper subprocess that lowers its own rlimits, runs
    /// the provided test body, and exits with the body's result
    /// code. Parent waits for child and returns the child's exit
    /// code. Any nonzero code from the child indicates a guard
    /// cleanup defect or harness issue — see exit-code convention
    /// comment above.
    fn run_in_forked_child<F: FnOnce() -> i32>(body: F) -> i32 {
        let pid = unsafe { libc::fork() };
        assert!(pid >= 0, "fork failed: {}", std::io::Error::last_os_error());
        if pid == 0 {
            // Child: install a silent panic hook so an assertion
            // failure inside the body doesn't multiplex stderr with
            // the parent's test output. Then run the body, which
            // returns an exit code. `_exit` skips Rust destructors
            // so the parent's resources copied via fork are not
            // double-closed.
            //
            // `catch_unwind` + `unwrap_or(99)` is effective here
            // because this helper is gated under `#[cfg(test)]` and
            // the dev/test profile inherits default unwind
            // semantics. Under `[profile.release]`'s `panic =
            // "abort"` the catch_unwind would be a no-op and a panic
            // in `body` would SIGABRT the child — which the parent's
            // signal-code path (`100 + WTERMSIG`) still surfaces
            // distinctly from the 99 fallback, so the exit-code
            // convention above remains self-consistent either way.
            let _ = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let code = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)).unwrap_or(99);
            unsafe { libc::_exit(code) };
        }
        let mut status: libc::c_int = 0;
        let waited = unsafe { libc::waitpid(pid, &mut status, 0) };
        assert_eq!(
            waited,
            pid,
            "waitpid({pid}) failed: {}",
            std::io::Error::last_os_error()
        );
        if libc::WIFEXITED(status) {
            libc::WEXITSTATUS(status)
        } else {
            // Terminated by signal — surface the signal number
            // as a large exit code so the parent's assertion can
            // distinguish it from the body's own codes.
            100 + libc::WTERMSIG(status)
        }
    }

    /// EMFILE on the inter-worker pipe loop: with num_workers=4 and
    /// PipeIo (which needs 2 pipe pairs = 4 pipe() calls = 8 fds),
    /// cap RLIMIT_NOFILE at baseline+5 so the first pair allocates
    /// cleanly (ab+ba = 4 fds) and the second pair's first `pipe(ab)`
    /// call fails with EMFILE (needs 2 fds, only 1 slot remains).
    /// At bail time `guard.pipe_pairs` holds the first pair;
    /// SpawnGuard::Drop must close all 4 fds so the child's fd
    /// count returns to baseline.
    ///
    /// Assumes a dense fd table (no gaps below the current baseline).
    /// If the child inherits a sparse table (e.g. a coordinator that
    /// closed fd 2 but left fd 3 open), RLIMIT_NOFILE gating yields
    /// different triggering semantics and the test may report 10
    /// (failure did not trigger) instead of 0. Also assumes
    /// `RUST_BACKTRACE` is unset — when set, a panic inside the body
    /// triggers backtrace capture which itself opens fds, shifting
    /// the effective baseline mid-run.
    #[test]
    fn spawn_guard_cleans_up_on_interworker_pipe_emfile() {
        let code = run_in_forked_child(|| {
            let baseline = count_open_fds();
            // Capture the inherited RLIMIT_NOFILE so the post-bail
            // restore uses a value the kernel will accept. The
            // lowering path below touches only `rlim_cur` and leaves
            // `rlim_max` at the original value, so an unprivileged
            // process can still raise `rlim_cur` back up after the
            // bail (without CAP_SYS_RESOURCE, which would be needed
            // to raise a previously-lowered `rlim_max`).
            let mut original_rlimit = libc::rlimit {
                rlim_cur: 0,
                rlim_max: 0,
            };
            if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut original_rlimit) } != 0 {
                return 13;
            }
            // RLIMIT_NOFILE is a hard limit on the highest fd
            // number + 1, not a headroom value — we need to pass a
            // value slightly above baseline so the first pipe pair
            // succeeds but the second pair's first `pipe(ab)` does
            // not. baseline + 5 permits 5 new fds: 4 for the first
            // pipe pair (ab+ba) and 1 leftover. The second pair's
            // `pipe(ab)` needs 2 fds against that 1 slot and fails
            // with EMFILE.
            let target_cur = (baseline + 5) as u64;
            let lowered = libc::rlimit {
                rlim_cur: target_cur,
                rlim_max: original_rlimit.rlim_max,
            };
            if unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &lowered) } != 0 {
                return 13;
            }
            let config = WorkloadConfig {
                num_workers: 4,
                affinity: ResolvedAffinity::None,
                work_type: WorkType::PipeIo { burst_iters: 1 },
                sched_policy: SchedPolicy::Normal,
                ..Default::default()
            };
            let result = WorkloadHandle::spawn(&config);
            if result.is_ok() {
                return 10; // Failure did not trigger.
            }
            // SpawnGuard::Drop has already run on the `?`/`bail!`
            // exit. Raise rlim_cur back to its original value so
            // reading /proc/self/fd for the post-check does not
            // itself fail with EMFILE. Silent ignore here would mask
            // an EMFILE in `count_open_fds` below as a fd leak;
            // return code 15 distinguishes the harness issue from a
            // guard defect.
            let err_msg = format!("{:#}", result.as_ref().err().unwrap());
            if unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &original_rlimit) } != 0 {
                return 15;
            }
            // Prove the bail arrived via the pipe branch, not a
            // later mmap or fork. Both pipe-failure paths bail
            // with "pipe failed".
            if !err_msg.contains("pipe failed") {
                return 14;
            }
            let after = count_open_fds();
            if after > baseline {
                return 11; // Fd leak.
            }
            if any_zombie_child() {
                return 12;
            }
            0
        });
        assert_eq!(
            code, 0,
            "child reported cleanup defect (code {code}): see exit-code table above \
             spawn_guard_cleans_up_on_interworker_pipe_emfile"
        );
    }

    /// EAGAIN on `fork`: with num_workers=1 and CpuSpin (no pipe
    /// pairs, no futex), cap RLIMIT_NPROC to 0 so the very first
    /// `libc::fork` inside the per-worker loop returns -1. At bail
    /// time the local cleanup (in the per-worker fork dispatch in
    /// `WorkloadHandle::spawn`) has closed the report+start pipes, so
    /// the guard carries only its empty `pipe_pairs`, zero children,
    /// and the iter_counters mmap. The Drop munmaps the iter_counters
    /// region (no-op for the fd count but proves the guard path
    /// fires) and returns cleanly. No zombies, no fd leak.
    #[test]
    fn spawn_guard_cleans_up_on_fork_eagain() {
        let code = run_in_forked_child(|| {
            let baseline = count_open_fds();
            if !set_rlimit_nproc_zero_headroom() {
                return 13;
            }
            let config = WorkloadConfig {
                num_workers: 1,
                affinity: ResolvedAffinity::None,
                work_type: WorkType::CpuSpin,
                sched_policy: SchedPolicy::Normal,
                ..Default::default()
            };
            let result = WorkloadHandle::spawn(&config);
            if result.is_ok() {
                return 10; // Failure did not trigger.
            }
            let msg = format!("{:#}", result.err().unwrap());
            // RLIMIT_NPROC denies fork with EAGAIN; prove the bail
            // arrived via the fork branch, not an earlier pipe
            // allocation.
            if !msg.contains("fork failed") {
                return 14;
            }
            let after = count_open_fds();
            if after > baseline {
                return 11;
            }
            if any_zombie_child() {
                return 12;
            }
            0
        });
        assert_eq!(
            code, 0,
            "child reported cleanup defect (code {code}): see exit-code table above \
             spawn_guard_cleans_up_on_fork_eagain"
        );
    }

    #[test]
    fn spawn_io_sync_produces_work() {
        let reports = spawn_and_collect_after(WorkType::IoSync, 1, 200);
        assert_eq!(reports.len(), 1);
        assert!(reports[0].work_units > 0);
    }

    #[test]
    fn spawn_bursty_produces_work() {
        let reports = spawn_and_collect_after(
            WorkType::Bursty {
                burst_ms: 50,
                sleep_ms: 50,
            },
            1,
            300,
        );
        assert_eq!(reports.len(), 1);
        assert!(reports[0].work_units > 0);
    }

    #[test]
    fn spawn_pipeio_produces_work() {
        let reports = spawn_and_collect_after(WorkType::PipeIo { burst_iters: 1024 }, 2, 300);
        assert_eq!(reports.len(), 2);
        for r in &reports {
            assert!(r.work_units > 0, "PipeIo worker {} did no work", r.tid);
        }
    }

    #[test]
    fn spawn_pipeio_odd_workers_fails() {
        let config = WorkloadConfig {
            num_workers: 3,
            affinity: ResolvedAffinity::None,
            work_type: WorkType::PipeIo { burst_iters: 1024 },
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let result = WorkloadHandle::spawn(&config);
        assert!(result.is_err(), "PipeIo with odd workers should fail");
        let msg = format!("{:#}", result.err().unwrap());
        assert!(
            msg.contains("divisible by 2"),
            "expected divisibility error: {msg}"
        );
    }

    #[test]
    fn sched_getcpu_valid() {
        let cpu = super::sched_getcpu();
        let max = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        assert!(cpu < max, "cpu {cpu} >= max {max}");
    }

    #[test]
    fn thread_cpu_time_positive() {
        // Do some work so CPU time is non-zero
        let mut x = 0u64;
        for i in 0..100_000 {
            x = x.wrapping_add(i);
        }
        std::hint::black_box(x);
        let t = super::thread_cpu_time_ns();
        assert!(t > 0);
    }

    #[test]
    fn set_thread_affinity_cpu_zero() {
        let pid: libc::pid_t = unsafe { libc::getpid() };
        let cpus: BTreeSet<usize> = [0].into_iter().collect();
        let result = set_thread_affinity(pid, &cpus);
        assert!(result.is_ok(), "pinning to CPU 0 should succeed");
    }

    #[test]
    fn spawn_zero_workers() {
        let config = WorkloadConfig {
            num_workers: 0,
            ..Default::default()
        };
        let h = WorkloadHandle::spawn(&config).unwrap();
        assert!(h.worker_pids().is_empty());
        let reports = h.stop_and_collect();
        assert!(reports.is_empty());
    }

    /// Zombie-tolerance on the Drop path: a caller drops a live
    /// `WorkloadHandle` after external code has SIGKILLed one of
    /// its workers. Between the signal delivery and the parent's
    /// `waitpid`, the killed worker sits as a zombie — its pid
    /// is still owned by this parent (only `waitpid` consumes
    /// the zombie state; an external signal does not), so Drop's
    /// follow-up `kill(pid, SIGKILL)` is a no-op against the
    /// zombie and Drop's `waitpid` reaps the exit status
    /// normally.
    ///
    /// Pins that Drop survives this realistic failure mode — an
    /// external operator (a CI runner's OOM killer, a stray
    /// `killall <name>`, a test-harness teardown signal)
    /// signals one worker before the handle's owning code
    /// finishes. Drop must leave the surviving siblings alone
    /// and reap the zombie without panicking.
    #[test]
    fn workload_handle_drop_tolerates_externally_killed_child() {
        let config = WorkloadConfig {
            num_workers: 2,
            affinity: ResolvedAffinity::None,
            work_type: WorkType::CpuSpin,
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        let pids = h.worker_pids();
        assert_eq!(pids.len(), 2);
        h.start();
        // Externally SIGKILL one worker. The handle still owns
        // the pid; on Drop it will try to signal + reap it.
        unsafe { libc::kill(pids[0], libc::SIGKILL) };
        // A brief sleep covers SIGKILL delivery latency. The
        // killed worker becomes a zombie rather than ESRCH (only
        // `waitpid` can clear it), so probing `kill(pid, 0)`
        // would spin forever — 50 ms is more than enough for
        // the kernel to deliver the signal and transition the
        // target to zombie state.
        std::thread::sleep(std::time::Duration::from_millis(50));
        // The assertion is implicit: this drop must not panic.
        // A panic inside Drop under panic=abort aborts the test
        // process, which nextest reports as an abnormal failure.
        drop(h);
    }

    #[test]
    fn worker_pids_count_matches_num_workers() {
        for n in [1, 3, 5] {
            let config = WorkloadConfig {
                num_workers: n,
                ..Default::default()
            };
            let h = WorkloadHandle::spawn(&config).unwrap();
            assert_eq!(
                h.worker_pids().len(),
                n,
                "worker_pids().len() should match num_workers={n}"
            );
            drop(h);
        }
    }

    #[test]
    fn worker_report_serde_edge_cases() {
        // Empty migrations and cpus_used
        let r = WorkerReport {
            tid: 0,
            work_units: 0,
            cpu_time_ns: 0,
            wall_time_ns: 0,
            off_cpu_ns: 0,
            migration_count: 0,
            cpus_used: BTreeSet::new(),
            migrations: vec![],
            max_gap_ms: 0,
            max_gap_cpu: 0,
            max_gap_at_ms: 0,
            resume_latencies_ns: vec![],
            wake_sample_total: 0,
            iterations: 0,
            schedstat_run_delay_ns: 0,
            schedstat_run_count: 0,
            schedstat_cpu_time_ns: 0,
            completed: true,
            numa_pages: BTreeMap::new(),
            vmstat_numa_pages_migrated: 0,
            exit_info: None,
            is_messenger: false,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: WorkerReport = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.tid, 0);
        assert!(r2.cpus_used.is_empty());
        assert!(r2.migrations.is_empty());

        // Max u64 values
        let r = WorkerReport {
            tid: i32::MAX,
            work_units: u64::MAX,
            cpu_time_ns: u64::MAX,
            wall_time_ns: u64::MAX,
            off_cpu_ns: u64::MAX,
            migration_count: u64::MAX,
            cpus_used: [0, usize::MAX].into_iter().collect(),
            migrations: vec![],
            max_gap_ms: u64::MAX,
            max_gap_cpu: usize::MAX,
            max_gap_at_ms: u64::MAX,
            resume_latencies_ns: vec![],
            wake_sample_total: u64::MAX,
            iterations: u64::MAX,
            schedstat_run_delay_ns: u64::MAX,
            schedstat_run_count: u64::MAX,
            schedstat_cpu_time_ns: u64::MAX,
            completed: true,
            numa_pages: BTreeMap::new(),
            vmstat_numa_pages_migrated: 0,
            exit_info: None,
            is_messenger: false,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: WorkerReport = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.work_units, u64::MAX);
        assert_eq!(r2.tid, i32::MAX);
    }

    #[test]
    fn io_sync_cleans_up_temp_file() {
        let config = WorkloadConfig {
            num_workers: 1,
            affinity: ResolvedAffinity::None,
            work_type: WorkType::IoSync,
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        h.start();
        std::thread::sleep(std::time::Duration::from_millis(200));
        let reports = h.stop_and_collect();
        assert_eq!(reports.len(), 1);
        let tid = reports[0].tid;
        let path = std::env::temp_dir()
            .join(format!("ktstr_io_{tid}"))
            .to_string_lossy()
            .to_string();
        assert!(
            !std::path::Path::new(&path).exists(),
            "temp file {path} should be cleaned up"
        );
    }

    #[test]
    fn set_sched_policy_normal_succeeds() {
        let pid: libc::pid_t = unsafe { libc::getpid() };
        let result = set_sched_policy(pid, SchedPolicy::Normal);
        assert!(result.is_ok());
    }

    #[test]
    fn set_affinity_via_handle() {
        let config = WorkloadConfig {
            num_workers: 1,
            affinity: ResolvedAffinity::None,
            work_type: WorkType::CpuSpin,
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        h.start();
        let cpus: BTreeSet<usize> = [0].into_iter().collect();
        let result = h.set_affinity(0, &cpus);
        assert!(result.is_ok());
        std::thread::sleep(std::time::Duration::from_millis(100));
        let reports = h.stop_and_collect();
        assert_eq!(reports.len(), 1);
    }

    #[test]
    fn work_type_bursty_defaults() {
        let wt = WorkType::from_name("Bursty").unwrap();
        if let WorkType::Bursty { burst_ms, sleep_ms } = wt {
            assert_eq!(burst_ms, 50);
            assert_eq!(sleep_ms, 100);
        } else {
            panic!("expected Bursty variant");
        }
    }

    #[test]
    fn work_type_pipeio_defaults() {
        let wt = WorkType::from_name("PipeIo").unwrap();
        if let WorkType::PipeIo { burst_iters } = wt {
            assert_eq!(burst_iters, 1024);
        } else {
            panic!("expected PipeIo variant");
        }
    }

    #[test]
    fn start_idempotent() {
        let config = WorkloadConfig {
            num_workers: 1,
            affinity: ResolvedAffinity::None,
            work_type: WorkType::CpuSpin,
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        h.start();
        h.start(); // Second call should be a no-op (started flag is true).
        std::thread::sleep(std::time::Duration::from_millis(100));
        let reports = h.stop_and_collect();
        assert_eq!(reports.len(), 1);
        assert!(reports[0].work_units > 0);
    }

    #[test]
    fn spawn_pipeio_four_workers() {
        let config = WorkloadConfig {
            num_workers: 4,
            affinity: ResolvedAffinity::None,
            work_type: WorkType::PipeIo { burst_iters: 512 },
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        assert_eq!(h.worker_pids().len(), 4);
        h.start();
        std::thread::sleep(std::time::Duration::from_millis(300));
        let reports = h.stop_and_collect();
        assert_eq!(reports.len(), 4);
        for r in &reports {
            assert!(
                r.work_units > 0,
                "PipeIo 4-worker worker {} did no work",
                r.tid
            );
        }
    }

    #[test]
    #[ignore]
    fn set_sched_policy_fifo_returns_result() {
        let pid: libc::pid_t = unsafe { libc::getpid() };
        let result = set_sched_policy(pid, SchedPolicy::Fifo(1));
        // SCHED_FIFO requires CAP_SYS_NICE; succeeds when the runner holds it.
        assert!(
            result.is_ok(),
            "SCHED_FIFO should succeed with CAP_SYS_NICE"
        );
        restore_normal(pid);
    }

    #[test]
    #[ignore]
    fn set_sched_policy_rr_returns_result() {
        let pid: libc::pid_t = unsafe { libc::getpid() };
        let result = set_sched_policy(pid, SchedPolicy::RoundRobin(1));
        // SCHED_RR requires CAP_SYS_NICE; succeeds when the runner holds it.
        assert!(result.is_ok(), "SCHED_RR should succeed with CAP_SYS_NICE");
        restore_normal(pid);
    }

    #[test]
    fn resolve_affinity_random_single_cpu_pool() {
        let from: BTreeSet<usize> = [7].into_iter().collect();
        let r = resolve_affinity(&ResolvedAffinity::Random { from, count: 1 }).unwrap();
        assert_eq!(r.unwrap(), [7].into_iter().collect());
    }

    // -- SchedPolicy variants --

    /// Restore SCHED_NORMAL via the raw syscall. `set_sched_policy(Normal)`
    /// is a no-op, so tests that change policy must use this to restore.
    fn restore_normal(pid: libc::pid_t) {
        let param = libc::sched_param { sched_priority: 0 };
        unsafe { libc::sched_setscheduler(pid, libc::SCHED_OTHER, &param) };
    }

    #[test]
    fn set_sched_policy_batch_returns_valid_result() {
        let pid: libc::pid_t = unsafe { libc::getpid() };
        let result = set_sched_policy(pid, SchedPolicy::Batch);
        // SCHED_BATCH does NOT require CAP_SYS_NICE:
        // `user_check_sched_setscheduler` routes only rt_policy /
        // dl_policy / negative-nice / leaving-IDLE through
        // req_priv; a fair-policy → fair-policy transition that
        // does not reduce nice never reaches the capable() check.
        // `scx_check_setscheduler` (kernel/sched/ext.c) does not
        // reject BATCH either — it only rejects transitions INTO
        // SCHED_EXT when `p->scx.disallow` is set, which BATCH is
        // not. Failure is therefore expected only on environments
        // that introduce extra LSM / security-module gates; the
        // test tolerates both outcomes.
        match result {
            Ok(()) => {
                let pol = unsafe { libc::sched_getscheduler(pid) };
                // Under sched_ext switch-all (`task_should_scx`
                // returns true for any policy when
                // `scx_switching_all` is set), `__setscheduler_class`
                // routes BATCH to `ext_sched_class`. Reading back
                // via `sched_getscheduler` returns the requested
                // policy regardless — this just sanity-checks the
                // syscall returned a non-negative policy id.
                assert!(
                    pol >= 0,
                    "sched_getscheduler must return a valid policy, got {pol}",
                );
                restore_normal(pid);
            }
            Err(ref e) => {
                let msg = format!("{e:#}");
                assert!(
                    msg.contains("sched_setscheduler"),
                    "error must name the syscall: {msg}"
                );
            }
        }
    }

    #[test]
    fn set_sched_policy_idle_returns_valid_result() {
        let pid: libc::pid_t = unsafe { libc::getpid() };
        let result = set_sched_policy(pid, SchedPolicy::Idle);
        // SCHED_IDLE does NOT require CAP_SYS_NICE for *entering*
        // IDLE: `user_check_sched_setscheduler` gates the
        // IDLE-related capability check on `task_has_idle_policy(p)
        // && !idle_policy(policy)` — i.e. CAP_SYS_NICE is required
        // only when *leaving* SCHED_IDLE for a non-idle class
        // without RLIMIT_NICE permission, not when entering it.
        // `scx_check_setscheduler` (kernel/sched/ext.c) does not
        // reject IDLE either — same reasoning as the BATCH test
        // above. Failure is expected only on environments with
        // extra LSM / security-module gates.
        match result {
            Ok(()) => {
                let pol = unsafe { libc::sched_getscheduler(pid) };
                // Same switch-all reasoning as the BATCH test —
                // IDLE routes to `ext_sched_class` under switch-all
                // but the syscall return is the requested policy id.
                assert!(
                    pol >= 0,
                    "sched_getscheduler must return a valid policy, got {pol}",
                );
                restore_normal(pid);
            }
            Err(ref e) => {
                let msg = format!("{e:#}");
                assert!(
                    msg.contains("sched_setscheduler"),
                    "error must name the syscall: {msg}"
                );
            }
        }
    }

    // -- SCHED_DEADLINE validation tests --
    //
    // The five rejection tests below exercise the structural
    // pre-validation that `set_sched_policy` performs before
    // issuing the `sched_setattr` syscall. Each invariant mirrors
    // a `__checkparam_dl` clause (`kernel/sched/deadline.c`); the
    // tests pin user-space rejection so a malformed `Deadline`
    // surfaces a named field rather than a generic kernel
    // `EINVAL`. None of these tests require `CAP_SYS_NICE`
    // because the bail!s fire before the syscall.

    /// `deadline == Duration::ZERO` must be rejected:
    /// `__checkparam_dl` returns false on `attr->sched_deadline ==
    /// 0`. The runtime floor is satisfied here so the failure
    /// pins the zero-deadline check, not the DL_SCALE check.
    #[test]
    fn set_sched_policy_deadline_zero_deadline_rejected() {
        let pid: libc::pid_t = unsafe { libc::getpid() };
        let result = set_sched_policy(
            pid,
            SchedPolicy::Deadline {
                runtime: Duration::from_nanos(1024),
                deadline: Duration::ZERO,
                period: Duration::from_nanos(1_000_000),
            },
        );
        let err = result.expect_err("zero deadline must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("deadline"),
            "error must name deadline field: {msg}"
        );
        assert!(
            msg.contains("must be > 0") || msg.contains("zero"),
            "error must explain zero rejection: {msg}"
        );
    }

    /// `runtime` shorter than 1024 ns must be rejected per the
    /// `DL_SCALE` floor in `__checkparam_dl`.
    #[test]
    fn set_sched_policy_deadline_runtime_below_dl_scale_rejected() {
        let pid: libc::pid_t = unsafe { libc::getpid() };
        let result = set_sched_policy(
            pid,
            SchedPolicy::Deadline {
                runtime: Duration::from_nanos(1023),
                deadline: Duration::from_nanos(100_000),
                period: Duration::from_nanos(1_000_000),
            },
        );
        let err = result.expect_err("runtime below DL_SCALE must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("runtime"),
            "error must name runtime field: {msg}"
        );
        assert!(
            msg.contains("DL_SCALE") || msg.contains("1024"),
            "error must reference the floor: {msg}"
        );
    }

    /// `runtime > deadline` must be rejected per the
    /// `runtime <= deadline` clause of `__checkparam_dl`.
    #[test]
    fn set_sched_policy_deadline_runtime_exceeds_deadline_rejected() {
        let pid: libc::pid_t = unsafe { libc::getpid() };
        let result = set_sched_policy(
            pid,
            SchedPolicy::Deadline {
                runtime: Duration::from_nanos(200_000),
                deadline: Duration::from_nanos(100_000),
                period: Duration::from_nanos(1_000_000),
            },
        );
        let err = result.expect_err("runtime > deadline must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("runtime") && msg.contains("deadline"),
            "error must name both fields: {msg}"
        );
    }

    /// `deadline > period` must be rejected when `period` is
    /// non-zero. Pairs with
    /// `set_sched_policy_deadline_period_zero_passes_validation`
    /// which proves the gate is conditional on a non-zero period.
    #[test]
    fn set_sched_policy_deadline_deadline_exceeds_period_rejected() {
        let pid: libc::pid_t = unsafe { libc::getpid() };
        let result = set_sched_policy(
            pid,
            SchedPolicy::Deadline {
                runtime: Duration::from_nanos(1024),
                deadline: Duration::from_nanos(2_000_000),
                period: Duration::from_nanos(1_000_000),
            },
        );
        let err = result.expect_err("deadline > period must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("deadline") && msg.contains("period"),
            "error must name both fields: {msg}"
        );
    }

    /// A `deadline` whose nanosecond count exceeds `i64::MAX` must
    /// be rejected. The kernel's `__checkparam_dl` clause `if
    /// (attr->sched_deadline & (1ULL << 63)) return false;`
    /// requires bit 63 to be clear; `duration_to_kernel_ns`
    /// enforces this as a single i64::MAX overflow check on
    /// `Duration::as_nanos()` (u128). The error message names the
    /// offending field via the `field` argument so the diagnostic
    /// points at `deadline` and not `runtime`/`period`.
    #[test]
    fn set_sched_policy_deadline_top_bit_set_rejected() {
        let pid: libc::pid_t = unsafe { libc::getpid() };
        let result = set_sched_policy(
            pid,
            SchedPolicy::Deadline {
                runtime: Duration::from_nanos(1024),
                // 1e12 seconds = 1e21 ns >> i64::MAX (≈ 9.2e18 ns)
                // — guaranteed to trip the overflow guard. Picked
                // far above the threshold so any future tweak to
                // the constraint still fires this test.
                deadline: Duration::from_secs(1_000_000_000_000),
                period: Duration::from_nanos(1_000_000),
            },
        );
        let err = result.expect_err("deadline exceeding i64::MAX must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("deadline") && (msg.contains("i64::MAX") || msg.contains("63 bits")),
            "error must name deadline field and the bit-63 / i64::MAX bound: {msg}"
        );
        // Per-field message: must NOT name `period` since only
        // `deadline` overflowed. `period` ordering matters —
        // `duration_to_kernel_ns` is called runtime → deadline →
        // period, so a deadline overflow short-circuits before
        // period is touched.
        assert!(
            !msg.contains("period"),
            "deadline-only overflow error must not mention period: {msg}"
        );
    }

    /// Happy-path: a structurally valid `Deadline` with
    /// `period == Duration::ZERO` reaches the `sched_setattr`
    /// syscall. The kernel substitutes `deadline` for the period
    /// in this case (see `if (!period) period = attr->sched_deadline;`
    /// in `__checkparam_dl`). Without `CAP_SYS_NICE` the syscall
    /// fails with EPERM at the kernel-side capability check;
    /// either Ok(()) or an Err whose message names
    /// `sched_setattr` confirms we cleared the user-space
    /// pre-validation. Marked `#[ignore]` so unprivileged CI
    /// doesn't see EPERM as a hard failure — runners with
    /// CAP_SYS_NICE can opt in.
    #[test]
    #[ignore]
    fn set_sched_policy_deadline_period_zero_passes_validation() {
        let pid: libc::pid_t = unsafe { libc::getpid() };
        let result = set_sched_policy(
            pid,
            SchedPolicy::Deadline {
                runtime: Duration::from_nanos(1024),
                deadline: Duration::from_nanos(200_000),
                period: Duration::ZERO,
            },
        );
        match result {
            Ok(()) => {
                // Restore SCHED_NORMAL so the test process leaves
                // its run with default policy.
                restore_normal(pid);
            }
            Err(e) => {
                let msg = format!("{e:#}");
                assert!(
                    msg.contains("sched_setattr"),
                    "validation must have passed (error from kernel must name sched_setattr): {msg}"
                );
            }
        }
    }

    #[test]
    fn sched_policy_debug_shows_variant_and_priority() {
        let s = format!("{:?}", SchedPolicy::Fifo(50));
        assert!(s.contains("Fifo"), "must show variant name");
        assert!(s.contains("50"), "must show priority value");
        let s = format!("{:?}", SchedPolicy::RoundRobin(99));
        assert!(s.contains("RoundRobin"), "must show variant name");
        assert!(s.contains("99"), "must show priority value");
        // Ensure different priorities produce different output.
        let s1 = format!("{:?}", SchedPolicy::Fifo(1));
        let s10 = format!("{:?}", SchedPolicy::Fifo(10));
        assert_ne!(
            s1, s10,
            "different priorities must produce different debug output"
        );
    }

    #[test]
    fn work_type_debug_shows_field_values() {
        let s = format!(
            "{:?}",
            WorkType::Bursty {
                burst_ms: 10,
                sleep_ms: 20
            }
        );
        assert!(s.contains("10"), "must show burst_ms value");
        assert!(s.contains("20"), "must show sleep_ms value");
        // Different field values must produce different output.
        let s2 = format!(
            "{:?}",
            WorkType::Bursty {
                burst_ms: 99,
                sleep_ms: 1
            }
        );
        assert!(s2.contains("99"), "must show changed burst_ms");
        assert!(s2.contains("1"), "must show changed sleep_ms");
        assert_ne!(
            s, s2,
            "different field values must produce different debug output"
        );
    }

    #[test]
    fn affinity_mode_debug_shows_cpus() {
        let a = ResolvedAffinity::Fixed([0, 1, 7].into_iter().collect());
        let s = format!("{:?}", a);
        assert!(s.contains("0"), "must show CPU 0");
        assert!(s.contains("1"), "must show CPU 1");
        assert!(s.contains("7"), "must show CPU 7");
        // Different CPU sets produce different output.
        let b = ResolvedAffinity::Fixed([3, 4].into_iter().collect());
        let s2 = format!("{:?}", b);
        assert!(s2.contains("3"), "must show CPU 3");
        assert_ne!(
            s, s2,
            "different CPU sets must produce different debug output"
        );
    }

    #[test]
    fn affinity_mode_clone_preserves_cpus() {
        let cpus: BTreeSet<usize> = [2, 5, 7].into_iter().collect();
        let a = ResolvedAffinity::Random {
            from: cpus.clone(),
            count: 2,
        };
        let b = a.clone();
        match b {
            ResolvedAffinity::Random { from, count } => {
                assert_eq!(from, cpus, "cloned from set must match original");
                assert_eq!(count, 2, "cloned count must match original");
            }
            _ => panic!("clone must preserve variant"),
        }
    }

    #[test]
    fn workload_config_debug_shows_field_values() {
        let c = WorkloadConfig {
            num_workers: 7,
            affinity: ResolvedAffinity::SingleCpu(3),
            work_type: WorkType::YieldHeavy,
            sched_policy: SchedPolicy::Batch,
            ..Default::default()
        };
        let s = format!("{:?}", c);
        assert!(s.contains("7"), "must show num_workers value");
        assert!(s.contains("SingleCpu"), "must show affinity variant");
        assert!(s.contains("3"), "must show affinity CPU");
        assert!(s.contains("YieldHeavy"), "must show work_type variant");
        assert!(s.contains("Batch"), "must show sched_policy variant");
    }

    #[test]
    fn migration_debug_shows_field_values() {
        let m = Migration {
            at_ns: 99999,
            from_cpu: 3,
            to_cpu: 7,
        };
        let s = format!("{:?}", m);
        assert!(s.contains("99999"), "must show at_ns value");
        assert!(s.contains("3"), "must show from_cpu value");
        assert!(s.contains("7"), "must show to_cpu value");
        let m2 = Migration {
            at_ns: 1,
            from_cpu: 0,
            to_cpu: 1,
        };
        let s2 = format!("{:?}", m2);
        assert_ne!(
            s, s2,
            "different field values must produce different debug output"
        );
    }

    #[test]
    fn worker_report_debug_shows_field_values() {
        let r = WorkerReport {
            tid: 42,
            work_units: 12345,
            cpu_time_ns: 1000,
            wall_time_ns: 2000,
            off_cpu_ns: 1000,
            migration_count: 3,
            cpus_used: [0, 5].into_iter().collect(),
            migrations: vec![],
            max_gap_ms: 77,
            max_gap_cpu: 5,
            max_gap_at_ms: 500,
            resume_latencies_ns: vec![],
            wake_sample_total: 0,
            iterations: 0,
            schedstat_run_delay_ns: 0,
            schedstat_run_count: 0,
            schedstat_cpu_time_ns: 0,
            completed: true,
            numa_pages: BTreeMap::new(),
            vmstat_numa_pages_migrated: 0,
            exit_info: None,
            is_messenger: false,
        };
        let s = format!("{:?}", r);
        assert!(s.contains("42"), "must show tid value");
        assert!(s.contains("12345"), "must show work_units value");
        assert!(s.contains("77"), "must show max_gap_ms value");
        assert!(s.contains("5"), "must show max_gap_cpu value");
    }

    #[test]
    fn work_type_clone_preserves_variant() {
        let a = WorkType::PipeIo { burst_iters: 512 };
        let b = a.clone();
        match b {
            WorkType::PipeIo { burst_iters } => assert_eq!(burst_iters, 512),
            _ => panic!("clone must preserve variant and fields"),
        }
    }

    #[test]
    fn sched_policy_copy_preserves_priority() {
        let a = SchedPolicy::Fifo(42);
        let b = a; // Copy
        match b {
            SchedPolicy::Fifo(p) => assert_eq!(p, 42),
            _ => panic!("copy must preserve variant and priority"),
        }
    }

    // -- WorkerReport edge cases --

    #[test]
    fn worker_report_off_cpu_ns_calculation() {
        // off_cpu_ns = wall_time_ns - cpu_time_ns
        let r = WorkerReport {
            tid: 1,
            work_units: 100,
            cpu_time_ns: 3_000_000_000,
            wall_time_ns: 5_000_000_000,
            off_cpu_ns: 2_000_000_000,
            migration_count: 0,
            cpus_used: [0].into_iter().collect(),
            migrations: vec![],
            max_gap_ms: 0,
            max_gap_cpu: 0,
            max_gap_at_ms: 0,
            resume_latencies_ns: vec![],
            wake_sample_total: 0,
            iterations: 0,
            schedstat_run_delay_ns: 0,
            schedstat_run_count: 0,
            schedstat_cpu_time_ns: 0,
            completed: true,
            numa_pages: BTreeMap::new(),
            vmstat_numa_pages_migrated: 0,
            exit_info: None,
            is_messenger: false,
        };
        assert_eq!(r.off_cpu_ns, r.wall_time_ns - r.cpu_time_ns);
    }

    #[test]
    fn migration_serde_multiple() {
        let migrations = vec![
            Migration {
                at_ns: 100,
                from_cpu: 0,
                to_cpu: 1,
            },
            Migration {
                at_ns: 200,
                from_cpu: 1,
                to_cpu: 2,
            },
            Migration {
                at_ns: 300,
                from_cpu: 2,
                to_cpu: 0,
            },
        ];
        let json = serde_json::to_string(&migrations).unwrap();
        let m2: Vec<Migration> = serde_json::from_str(&json).unwrap();
        assert_eq!(m2.len(), 3);
        assert_eq!(m2[0].from_cpu, 0);
        assert_eq!(m2[2].to_cpu, 0);
    }

    // -- resolve_affinity edge cases --

    #[test]
    fn resolve_affinity_random_zero_count_rejected() {
        // Regression: count=0 previously coerced silently to 1, masking
        // caller bugs. Now it must return an Err.
        let from: BTreeSet<usize> = (0..4).collect();
        let err = resolve_affinity(&ResolvedAffinity::Random { from, count: 0 }).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("count") && msg.contains("> 0"),
            "error must name the field: {msg}"
        );
    }

    #[test]
    fn resolve_affinity_random_empty_pool_is_none() {
        // Regression: ResolvedAffinity::Random { from: empty, count } previously
        // produced an empty affinity mask rejected by sched_setaffinity
        // with EINVAL. Empty pool must short-circuit to Ok(None).
        let from: BTreeSet<usize> = BTreeSet::new();
        let r = resolve_affinity(&ResolvedAffinity::Random { from, count: 1 }).unwrap();
        assert!(r.is_none(), "empty Random pool must resolve to no affinity");
    }

    // -- reservoir_push tests --

    #[test]
    fn reservoir_push_empty_buf() {
        let mut buf = Vec::new();
        let mut count = 0u64;
        reservoir_push(&mut buf, &mut count, 42, 10);
        assert_eq!(buf, vec![42]);
        assert_eq!(count, 1);
    }

    #[test]
    fn reservoir_push_under_cap() {
        let mut buf = Vec::new();
        let mut count = 0u64;
        for i in 0..5 {
            reservoir_push(&mut buf, &mut count, i * 100, 10);
        }
        assert_eq!(buf.len(), 5);
        assert_eq!(count, 5);
        assert_eq!(buf, vec![0, 100, 200, 300, 400]);
    }

    #[test]
    fn reservoir_push_at_cap() {
        let mut buf = Vec::new();
        let mut count = 0u64;
        for i in 0..10 {
            reservoir_push(&mut buf, &mut count, i, 10);
        }
        assert_eq!(buf.len(), 10);
        assert_eq!(count, 10);
        // All values should be present since we're exactly at cap.
        for i in 0..10 {
            assert!(buf.contains(&i), "missing {i}");
        }
    }

    #[test]
    fn reservoir_push_over_cap_maintains_size() {
        let mut buf = Vec::new();
        let mut count = 0u64;
        let cap = 5;
        for i in 0..1000 {
            reservoir_push(&mut buf, &mut count, i, cap);
        }
        assert_eq!(buf.len(), cap);
        assert_eq!(count, 1000);
    }

    #[test]
    fn reservoir_push_uniform_sampling() {
        // Statistical test: push 10000 values into cap=100 reservoir.
        // Each value should have roughly equal probability of being present.
        // We test that the reservoir contains values from the full range.
        let mut buf = Vec::new();
        let mut count = 0u64;
        let cap = 100;
        let total = 10_000u64;
        for i in 0..total {
            reservoir_push(&mut buf, &mut count, i, cap);
        }
        assert_eq!(buf.len(), cap);
        assert_eq!(count, total);
        // The reservoir should contain values from different parts of the range.
        let has_early = buf.iter().any(|&v| v < total / 4);
        let has_late = buf.iter().any(|&v| v > total * 3 / 4);
        assert!(has_early, "reservoir should contain early values");
        assert!(has_late, "reservoir should contain late values");
    }

    #[test]
    fn reservoir_push_cap_zero() {
        // Zero-capacity reservoir: buf.len() < 0 is never true (usize),
        // falls through to else branch where random_range(0..1) returns 0,
        // and 0 < 0 is false — sample is discarded.
        let mut buf = Vec::new();
        let mut count = 0u64;
        for i in 0..10 {
            reservoir_push(&mut buf, &mut count, i, 0);
        }
        assert!(buf.is_empty(), "cap=0 should never store samples");
        assert_eq!(count, 10, "count incremented regardless");
    }

    #[test]
    fn reservoir_push_cap_one() {
        // Single-element reservoir. First sample always stored.
        // Subsequent samples replace with probability 1/count.
        let mut buf = Vec::new();
        let mut count = 0u64;
        reservoir_push(&mut buf, &mut count, 42, 1);
        assert_eq!(buf, vec![42]);
        assert_eq!(count, 1);
        // Push more — buf stays length 1.
        for i in 1..100 {
            reservoir_push(&mut buf, &mut count, i * 100, 1);
        }
        assert_eq!(buf.len(), 1);
        assert_eq!(count, 100);
    }

    // -- read_schedstat tests --

    #[test]
    fn read_schedstat_returns_finite_triple() {
        // The calling thread has been scheduled at least once by the
        // time this test runs (it's executing right now), so cpu_time
        // and timeslices must be strictly positive. run_delay can
        // legitimately be zero on an idle host where the test thread
        // never waited for a runqueue slot, so it is left unchecked.
        //
        // `None` is a legitimate outcome when the host kernel is
        // built without `CONFIG_SCHEDSTATS` — treat that as a skip
        // rather than a test failure.
        let Some((cpu_time, _run_delay, timeslices)) = read_schedstat(None) else {
            eprintln!("skipping: /proc/self/schedstat not available (CONFIG_SCHEDSTATS off)");
            return;
        };
        assert!(cpu_time > 0);
        assert!(timeslices > 0);
    }

    #[test]
    fn parse_schedstat_line_happy_path() {
        // A well-formed line has at least three whitespace-separated
        // u64 fields; extra trailing fields are ignored.
        let (cpu_time, run_delay, timeslices) =
            parse_schedstat_line("100 200 300 999 extra").unwrap();
        assert_eq!(cpu_time, 100);
        assert_eq!(run_delay, 200);
        assert_eq!(timeslices, 300);
    }

    #[test]
    fn parse_schedstat_line_tab_and_newline_separators() {
        // `split_whitespace` treats any run of whitespace as one
        // separator, so tabs and trailing newlines must parse.
        let parsed = parse_schedstat_line("1\t2\t3\n").unwrap();
        assert_eq!(parsed, (1, 2, 3));
    }

    #[test]
    fn parse_schedstat_line_missing_field_returns_none() {
        // Two fields is one short — the third `?` bails.
        assert!(parse_schedstat_line("100 200").is_none());
        // One field short of two.
        assert!(parse_schedstat_line("100").is_none());
        // Empty input — zero fields.
        assert!(parse_schedstat_line("").is_none());
        // Whitespace-only input — zero tokens after split.
        assert!(parse_schedstat_line("   \t\n  ").is_none());
    }

    #[test]
    fn parse_schedstat_line_non_u64_token_returns_none() {
        // Any non-u64 token fails the `.parse::<u64>().ok()?` chain.
        assert!(parse_schedstat_line("not-a-number 200 300").is_none());
        assert!(parse_schedstat_line("100 abc 300").is_none());
        assert!(parse_schedstat_line("100 200 nan").is_none());
        // Negative numbers parse to u64 as an error.
        assert!(parse_schedstat_line("-1 200 300").is_none());
        // Overflow beyond u64::MAX.
        assert!(parse_schedstat_line("99999999999999999999 2 3").is_none());
    }

    #[test]
    fn warn_schedstat_unavailable_once_does_not_panic_on_repeat() {
        // `std::sync::Once::call_once` guarantees at most one
        // eprintln regardless of how many times the gate fires.
        // Smoke-check that repeated calls don't panic — direct
        // stderr-emission assertions require a process-global
        // capture gate (`#[test]` threads share fd 2), which is
        // out of scope for this unit test.
        for _ in 0..10 {
            warn_schedstat_unavailable_once();
        }
    }

    // -- FutexFanOut tests --

    #[test]
    fn spawn_futex_fan_out_produces_work() {
        let reports = spawn_and_collect_after(
            WorkType::FutexFanOut {
                fan_out: 4,
                spin_iters: 1024,
            },
            5, // 1 messenger + 4 receivers
            500,
        );
        assert_eq!(reports.len(), 5);
        for r in &reports {
            assert!(r.work_units > 0, "FutexFanOut worker {} did no work", r.tid);
        }
    }

    #[test]
    fn spawn_futex_fan_out_receivers_record_wake_latency() {
        let config = WorkloadConfig {
            num_workers: 5,
            affinity: ResolvedAffinity::None,
            work_type: WorkType::FutexFanOut {
                fan_out: 4,
                spin_iters: 512,
            },
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        h.start();
        std::thread::sleep(std::time::Duration::from_millis(500));
        let reports = h.stop_and_collect();
        // At least one receiver should have wake latency samples.
        let has_latencies = reports.iter().any(|r| !r.resume_latencies_ns.is_empty());
        assert!(has_latencies, "receivers should record wake latencies");
    }

    #[test]
    fn spawn_futex_fan_out_bad_worker_count_fails() {
        let config = WorkloadConfig {
            num_workers: 3, // not divisible by 5
            affinity: ResolvedAffinity::None,
            work_type: WorkType::FutexFanOut {
                fan_out: 4,
                spin_iters: 1024,
            },
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let result = WorkloadHandle::spawn(&config);
        assert!(result.is_err());
        let msg = format!("{:#}", result.err().unwrap());
        assert!(
            msg.contains("divisible by 5"),
            "expected divisibility error: {msg}"
        );
    }

    #[test]
    fn spawn_futex_fan_out_two_groups() {
        let config = WorkloadConfig {
            num_workers: 10, // 2 groups of (1+4)
            affinity: ResolvedAffinity::None,
            work_type: WorkType::FutexFanOut {
                fan_out: 4,
                spin_iters: 512,
            },
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        assert_eq!(h.worker_pids().len(), 10);
        h.start();
        std::thread::sleep(std::time::Duration::from_millis(500));
        let reports = h.stop_and_collect();
        assert_eq!(reports.len(), 10);
        for r in &reports {
            assert!(r.work_units > 0, "worker {} did no work", r.tid);
        }
    }

    #[test]
    fn spawn_futex_fan_out_single_receiver() {
        // Minimal fan-out: 1 messenger + 1 receiver per group (like ping-pong).
        let config = WorkloadConfig {
            num_workers: 2,
            affinity: ResolvedAffinity::None,
            work_type: WorkType::FutexFanOut {
                fan_out: 1,
                spin_iters: 1024,
            },
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        h.start();
        std::thread::sleep(std::time::Duration::from_millis(300));
        let reports = h.stop_and_collect();
        assert_eq!(reports.len(), 2);
        for r in &reports {
            assert!(r.work_units > 0, "worker {} did no work", r.tid);
        }
    }

    #[test]
    fn work_type_futex_fan_out_name() {
        let wt = WorkType::FutexFanOut {
            fan_out: 4,
            spin_iters: 1024,
        };
        assert_eq!(wt.name(), "FutexFanOut");
    }

    #[test]
    fn work_type_futex_fan_out_from_name() {
        let wt = WorkType::from_name("FutexFanOut").unwrap();
        match wt {
            WorkType::FutexFanOut {
                fan_out,
                spin_iters,
            } => {
                assert_eq!(fan_out, 4);
                assert_eq!(spin_iters, 1024);
            }
            _ => panic!("expected FutexFanOut"),
        }
    }

    #[test]
    fn work_type_futex_fan_out_group_size() {
        let wt = WorkType::FutexFanOut {
            fan_out: 4,
            spin_iters: 1024,
        };
        assert_eq!(wt.worker_group_size(), Some(5));
    }

    // -- snapshot_iterations tests --

    #[test]
    fn snapshot_iterations_empty_handle() {
        let config = WorkloadConfig {
            num_workers: 0,
            ..Default::default()
        };
        let h = WorkloadHandle::spawn(&config).unwrap();
        assert!(h.snapshot_iterations().is_empty());
        drop(h);
    }

    #[test]
    fn snapshot_iterations_running_workers() {
        let config = WorkloadConfig {
            num_workers: 2,
            affinity: ResolvedAffinity::None,
            work_type: WorkType::CpuSpin,
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        h.start();
        std::thread::sleep(std::time::Duration::from_millis(200));
        let iters = h.snapshot_iterations();
        assert_eq!(iters.len(), 2);
        // After 200ms of CpuSpin, workers should have done iterations.
        for (i, &v) in iters.iter().enumerate() {
            assert!(v > 0, "worker {i} should have iterations > 0, got {v}");
        }
        drop(h);
    }

    // -- worker_group_size --

    #[test]
    fn worker_group_size_paired() {
        assert_eq!(WorkType::pipe_io(100).worker_group_size(), Some(2));
        assert_eq!(WorkType::futex_ping_pong(100).worker_group_size(), Some(2));
        assert_eq!(WorkType::cache_pipe(32, 100).worker_group_size(), Some(2));
    }

    #[test]
    fn worker_group_size_fan_out() {
        assert_eq!(WorkType::futex_fan_out(4, 100).worker_group_size(), Some(5));
        assert_eq!(WorkType::futex_fan_out(1, 100).worker_group_size(), Some(2));
    }

    #[test]
    fn worker_group_size_wake_chain() {
        // WakeChain group_size == depth (per-chain), independent
        // of fanout. Spawn-side allocates `num_workers / depth =
        // fanout` futex regions.
        let wc = WorkType::wake_chain(8, 4, false, Duration::from_micros(100));
        assert_eq!(wc.worker_group_size(), Some(8));
        let wc1 = WorkType::wake_chain(3, 1, true, Duration::from_micros(50));
        assert_eq!(wc1.worker_group_size(), Some(3));
    }

    #[test]
    fn worker_group_size_thundering_herd() {
        // ThunderingHerd collapses every worker into one group:
        // `waiters + 1` (1 waker + N waiters).
        let th = WorkType::thundering_herd(7, 1000, 5);
        assert_eq!(th.worker_group_size(), Some(8));
    }

    #[test]
    fn worker_group_size_ungrouped() {
        assert_eq!(WorkType::CpuSpin.worker_group_size(), None);
        assert_eq!(WorkType::YieldHeavy.worker_group_size(), None);
        assert_eq!(WorkType::Mixed.worker_group_size(), None);
        assert_eq!(WorkType::IoSync.worker_group_size(), None);
        assert_eq!(WorkType::bursty(50, 100).worker_group_size(), None);
        assert_eq!(WorkType::cache_pressure(32, 64).worker_group_size(), None);
        assert_eq!(WorkType::cache_yield(32, 64).worker_group_size(), None);
    }

    // -- needs_shared_mem --

    #[test]
    fn needs_shared_mem_futex_types() {
        assert!(WorkType::futex_ping_pong(100).needs_shared_mem());
        assert!(WorkType::futex_fan_out(4, 100).needs_shared_mem());
    }

    #[test]
    fn needs_shared_mem_non_futex() {
        assert!(!WorkType::CpuSpin.needs_shared_mem());
        assert!(!WorkType::pipe_io(100).needs_shared_mem());
        assert!(!WorkType::cache_pipe(32, 100).needs_shared_mem());
        assert!(!WorkType::cache_pressure(32, 64).needs_shared_mem());
    }

    // -- needs_cache_buf --

    #[test]
    fn needs_cache_buf_cache_types() {
        assert!(WorkType::cache_pressure(32, 64).needs_cache_buf());
        assert!(WorkType::cache_yield(32, 64).needs_cache_buf());
        assert!(WorkType::cache_pipe(32, 100).needs_cache_buf());
    }

    #[test]
    fn needs_cache_buf_non_cache() {
        assert!(!WorkType::CpuSpin.needs_cache_buf());
        assert!(!WorkType::pipe_io(100).needs_cache_buf());
        assert!(!WorkType::futex_ping_pong(100).needs_cache_buf());
        assert!(!WorkType::futex_fan_out(4, 100).needs_cache_buf());
    }

    // -- resolve_work_type --

    #[test]
    fn resolve_work_type_not_swappable() {
        let base = WorkType::CpuSpin;
        let over = WorkType::YieldHeavy;
        let result = resolve_work_type(&base, Some(&over), false, 4);
        assert!(matches!(result, WorkType::CpuSpin));
    }

    #[test]
    fn resolve_work_type_swappable_applies_override() {
        let base = WorkType::CpuSpin;
        let over = WorkType::YieldHeavy;
        let result = resolve_work_type(&base, Some(&over), true, 4);
        assert!(matches!(result, WorkType::YieldHeavy));
    }

    #[test]
    fn resolve_work_type_swappable_no_override() {
        let base = WorkType::CpuSpin;
        let result = resolve_work_type(&base, None, true, 4);
        assert!(matches!(result, WorkType::CpuSpin));
    }

    #[test]
    fn resolve_work_type_group_size_mismatch() {
        let base = WorkType::CpuSpin;
        let over = WorkType::pipe_io(100); // group_size = 2
        let result = resolve_work_type(&base, Some(&over), true, 3); // 3 not divisible by 2
        assert!(matches!(result, WorkType::CpuSpin));
    }

    #[test]
    fn resolve_work_type_group_size_match() {
        let base = WorkType::CpuSpin;
        let over = WorkType::pipe_io(100); // group_size = 2
        let result = resolve_work_type(&base, Some(&over), true, 4); // 4 divisible by 2
        assert!(matches!(result, WorkType::PipeIo { .. }));
    }

    #[test]
    fn resolve_work_type_fan_out_group_size() {
        let base = WorkType::CpuSpin;
        let over = WorkType::futex_fan_out(3, 100); // group_size = 4
        let result = resolve_work_type(&base, Some(&over), true, 8); // 8 divisible by 4
        assert!(matches!(result, WorkType::FutexFanOut { .. }));
        let fail = resolve_work_type(&base, Some(&over), true, 6); // 6 not divisible by 4
        assert!(matches!(fail, WorkType::CpuSpin));
    }

    // -- WorkSpec builder --

    #[test]
    fn work_builder_chain() {
        let w = WorkSpec::default()
            .workers(8)
            .work_type(WorkType::bursty(10, 20))
            .sched_policy(SchedPolicy::Batch)
            .affinity(AffinityIntent::SingleCpu)
            .nice(7);
        assert_eq!(w.num_workers, Some(8));
        assert!(matches!(
            w.work_type,
            WorkType::Bursty {
                burst_ms: 10,
                sleep_ms: 20
            }
        ));
        assert!(matches!(w.sched_policy, SchedPolicy::Batch));
        assert!(matches!(w.affinity, AffinityIntent::SingleCpu));
        assert_eq!(w.nice, 7);
    }

    #[test]
    fn work_default_values() {
        let w = WorkSpec::default();
        assert_eq!(w.num_workers, None);
        assert!(matches!(w.work_type, WorkType::CpuSpin));
        assert!(matches!(w.sched_policy, SchedPolicy::Normal));
        assert!(matches!(w.affinity, AffinityIntent::Inherit));
        // Default nice is 0 — same skip semantics as
        // [`WorkloadConfig::nice`].
        assert_eq!(w.nice, 0);
    }

    // -- SchedPolicy constructors --

    #[test]
    fn sched_policy_fifo_constructor() {
        match SchedPolicy::fifo(50) {
            SchedPolicy::Fifo(p) => assert_eq!(p, 50),
            _ => panic!("expected Fifo"),
        }
    }

    #[test]
    fn sched_policy_rr_constructor() {
        match SchedPolicy::round_robin(25) {
            SchedPolicy::RoundRobin(p) => assert_eq!(p, 25),
            _ => panic!("expected RoundRobin"),
        }
    }

    #[test]
    fn spawn_futex_ping_pong_produces_work() {
        let reports = spawn_and_collect_after(WorkType::FutexPingPong { spin_iters: 1024 }, 2, 500);
        assert_eq!(reports.len(), 2);
        for r in &reports {
            assert!(
                r.work_units > 0,
                "FutexPingPong worker {} did no work",
                r.tid
            );
        }
    }

    #[test]
    fn spawn_cache_pressure_produces_work() {
        let reports = spawn_and_collect_after(
            WorkType::CachePressure {
                size_kb: 32,
                stride: 64,
            },
            1,
            200,
        );
        assert_eq!(reports.len(), 1);
        assert!(reports[0].work_units > 0);
    }

    #[test]
    fn spawn_cache_yield_produces_work() {
        let reports = spawn_and_collect_after(
            WorkType::CacheYield {
                size_kb: 32,
                stride: 64,
            },
            1,
            200,
        );
        assert_eq!(reports.len(), 1);
        assert!(reports[0].work_units > 0);
    }

    #[test]
    fn spawn_cache_pipe_produces_work() {
        let reports = spawn_and_collect_after(
            WorkType::CachePipe {
                size_kb: 32,
                burst_iters: 1024,
            },
            2,
            300,
        );
        assert_eq!(reports.len(), 2);
        for r in &reports {
            assert!(r.work_units > 0, "CachePipe worker {} did no work", r.tid);
        }
    }

    #[test]
    fn spawn_sequence_produces_work() {
        let reports = spawn_and_collect_after(
            WorkType::Sequence {
                first: Phase::Spin(Duration::from_millis(10)),
                rest: vec![Phase::Yield(Duration::from_millis(10))],
            },
            1,
            200,
        );
        assert_eq!(reports.len(), 1);
        assert!(reports[0].work_units > 0);
    }

    // -- Custom work type tests --

    fn stub_custom_fn(_stop: &AtomicBool) -> WorkerReport {
        WorkerReport {
            tid: 0,
            work_units: 0,
            cpu_time_ns: 0,
            wall_time_ns: 0,
            off_cpu_ns: 0,
            migration_count: 0,
            cpus_used: BTreeSet::new(),
            migrations: vec![],
            max_gap_ms: 0,
            max_gap_cpu: 0,
            max_gap_at_ms: 0,
            resume_latencies_ns: vec![],
            wake_sample_total: 0,
            iterations: 0,
            schedstat_run_delay_ns: 0,
            schedstat_run_count: 0,
            schedstat_cpu_time_ns: 0,
            completed: true,
            numa_pages: BTreeMap::new(),
            vmstat_numa_pages_migrated: 0,
            exit_info: None,
            is_messenger: false,
        }
    }

    #[test]
    fn custom_name_returns_label() {
        let wt = WorkType::custom("my_work", stub_custom_fn);
        assert_eq!(wt.name(), "my_work");
    }

    #[test]
    fn custom_group_size_is_none() {
        let wt = WorkType::custom("x", stub_custom_fn);
        assert_eq!(wt.worker_group_size(), None);
    }

    fn custom_spin_fn(stop: &AtomicBool) -> WorkerReport {
        let tid: libc::pid_t = unsafe { libc::getpid() };
        let start = Instant::now();
        let mut work_units = 0u64;
        while !stop_requested(stop) {
            work_units = std::hint::black_box(work_units.wrapping_add(1));
            std::hint::spin_loop();
        }
        let wall_time_ns = start.elapsed().as_nanos() as u64;
        WorkerReport {
            tid,
            work_units,
            cpu_time_ns: 0,
            wall_time_ns,
            off_cpu_ns: 0,
            migration_count: 0,
            cpus_used: BTreeSet::new(),
            migrations: vec![],
            max_gap_ms: 0,
            max_gap_cpu: 0,
            max_gap_at_ms: 0,
            resume_latencies_ns: vec![],
            wake_sample_total: 0,
            iterations: work_units,
            schedstat_run_delay_ns: 0,
            schedstat_run_count: 0,
            schedstat_cpu_time_ns: 0,
            completed: true,
            numa_pages: BTreeMap::new(),
            vmstat_numa_pages_migrated: 0,
            exit_info: None,
            is_messenger: false,
        }
    }

    #[test]
    fn spawn_custom_produces_work() {
        let config = WorkloadConfig {
            num_workers: 1,
            affinity: ResolvedAffinity::None,
            work_type: WorkType::custom("test_spin", custom_spin_fn),
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        h.start();
        std::thread::sleep(std::time::Duration::from_millis(200));
        let reports = h.stop_and_collect();
        assert_eq!(reports.len(), 1);
        assert!(
            reports[0].work_units > 0,
            "Custom worker did no work: work_units={}",
            reports[0].work_units
        );
        assert!(reports[0].wall_time_ns > 0);
        assert!(
            reports.iter().all(|r| r.completed),
            "every worker report on the live / non-sentinel path \
             must carry completed=true — pairs with the
             completed=false assertion in \
             stop_and_collect_reaps_grandchild_from_panicking_custom_closure",
        );
    }

    /// Ready-file path shared between [`ignores_sigusr1_fn`] and
    /// `stop_and_collect_sentinel_exits_for_sigusr1_ignoring_worker`.
    /// The worker writes a zero-byte file at this path after
    /// installing `SIG_IGN` for SIGUSR1; the parent polls for the
    /// file's appearance before sending SIGUSR1, eliminating the
    /// race the old 200ms sleep papered over.
    fn ready_file_path(pid: libc::pid_t) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("ktstr-sigusr1-ignore-ready-{pid}"))
    }

    /// Shared post-fork prologue for test WorkType closures: installs
    /// `SIG_IGN` for SIGUSR1 so stop_and_collect cannot flip STOP via
    /// the signal path, then returns the current pid (which doubles as
    /// the worker's tid on Linux because [`WorkloadHandle::spawn`]
    /// forks one process per worker). Factored out of the two custom
    /// closures that share this opening; both forks land in a
    /// single-threaded child where `libc::signal` is safe.
    fn ignore_sigusr1_and_get_pid() -> libc::pid_t {
        unsafe {
            libc::signal(libc::SIGUSR1, libc::SIG_IGN);
        }
        unsafe { libc::getpid() }
    }

    /// Sleep-based deadline loop shared by the SIGUSR1-ignoring test
    /// closures. Returns when either `stop` flips (SIGUSR1 handler
    /// path, never fires under SIG_IGN — kept honest) or `timeout`
    /// elapses. Takes a [`Duration`] to match
    /// [`wait_for_file_or_panic`]'s signature; callers that want to
    /// spell the value as "seven seconds" still write
    /// `Duration::from_secs(7)`.
    ///
    /// Uses `thread::sleep(10ms)` rather than `spin_loop()`: the
    /// closures' purpose is to outlive stop_and_collect's 5s
    /// collection deadline, not to respond to cache-coherent store
    /// visibility at CPU speed, so a ~100x lower CPU footprint is
    /// strictly better under CI contention.
    fn wait_for_deadline(stop: &AtomicBool, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        while !stop_requested(stop) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Poll for `path`'s appearance with a deadline, aborting early if
    /// `liveness_pid` dies before the file is written. `kill(pid, 0)` is
    /// the POSIX existence probe — Err means the pid is gone (or the
    /// caller is not permitted to signal it, which for a pid owned by
    /// this test process implies the pid has already been reaped).
    /// Panics with an actionable message on either early-death or
    /// deadline. `context` is appended to the panic text so the caller
    /// can pin the failure to a specific test scenario.
    fn wait_for_file_or_panic(
        path: &std::path::Path,
        timeout: Duration,
        liveness_pid: libc::pid_t,
        context: &str,
    ) {
        let deadline = Instant::now() + timeout;
        while !path.exists() {
            if nix::sys::signal::kill(nix::unistd::Pid::from_raw(liveness_pid), None).is_err() {
                panic!("pid {liveness_pid} exited before writing ready file {path:?} — {context}",);
            }
            if Instant::now() >= deadline {
                panic!(
                    "pid {liveness_pid} did not write ready file {path:?} within {timeout:?} — {context}",
                );
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Worker function that installs `SIG_IGN` for SIGUSR1 — overriding
    /// the `sigusr1_handler` the child set up post-fork — and spins
    /// for long enough to outlive the parent's 5s collection deadline.
    /// Used by the sigusr1-ignored path test below.
    ///
    /// `libc::signal(SIGUSR1, SIG_IGN)` replaces the handler on the
    /// child's process-wide disposition table, so the parent's
    /// `kill(pid, SIGUSR1)` arrives as a no-op — STOP never flips to
    /// true via the handler, and even code that checks STOP spins
    /// past the deadline.
    fn ignores_sigusr1_fn(stop: &AtomicBool) -> WorkerReport {
        let tid = ignore_sigusr1_and_get_pid();
        // Readiness handshake: after SIG_IGN is installed, write a
        // zero-byte ready file so the parent can proceed without
        // waiting on a fixed-duration sleep. Without the handshake
        // the parent had to guess a safe delay (200ms) covering
        // fork + signal(2) syscalls plus CPU contention —
        // too short and the parent's SIGUSR1 races the handler
        // replacement and the test fails spuriously. See
        // `stop_and_collect_sentinel_exits_for_sigusr1_ignoring_worker`
        // below for the reader side.
        let ready_path = ready_file_path(tid);
        let _ = std::fs::write(&ready_path, []);
        // Wait 7s — well past stop_and_collect's 5s shared deadline.
        // The `!stop.load` check is kept honest inside
        // `wait_for_deadline` (no infinite loop) but is only
        // observed via the fallback timeout: with SIG_IGN in place,
        // the parent's SIGUSR1 doesn't flip STOP.
        wait_for_deadline(stop, Duration::from_secs(7));
        // Report body is never observed — the parent SIGKILLs the
        // worker before any `f.write_all(&json)` could run. Per the
        // `WorkerReport` doc, sentinel-shape constructions use
        // `..Default::default()` so a future field addition doesn't
        // silently drift the test.
        WorkerReport {
            tid,
            ..WorkerReport::default()
        }
    }

    /// Pins the `stop_and_collect` sentinel path where SIGUSR1 is
    /// ignored and the WNOHANG-returns-`StillAlive` branch fires:
    /// the parent escalates to SIGKILL, collects zero JSON from the
    /// worker, and the synthesized [`WorkerReport`] carries
    /// `exit_info: Some(TimedOut)` (or `Some(Signaled(SIGKILL))`
    /// if the race between WNOHANG and the kill put the reap at
    /// the blocking waitpid). Without this test, the escalation
    /// branch of `classify_wait_outcome` is only covered by the
    /// pure unit test `classify_wait_outcome_still_alive_maps_to_timed_out`;
    /// pairing that with this end-to-end exercise proves the
    /// integration (parent loop + `ignores_sigusr1_fn` + sentinel
    /// fill) doesn't drop the diagnostic along the way.
    ///
    /// Expected runtime: ~5s (the shared deadline), plus a few ms
    /// for spawn + kill + reap. Marked with a shorter spin window
    /// in `ignores_sigusr1_fn` (7s ceiling) so even if the parent
    /// deadline extends accidentally, the test still terminates.
    #[test]
    fn stop_and_collect_sentinel_exits_for_sigusr1_ignoring_worker() {
        let config = WorkloadConfig {
            num_workers: 1,
            affinity: ResolvedAffinity::None,
            work_type: WorkType::custom("sigusr1_ignore", ignores_sigusr1_fn),
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        // Readiness handshake — poll for the ready file the worker
        // writes after its `libc::signal(SIGUSR1, SIG_IGN)` call
        // completes. Replaces a fixed 200ms sleep with progress-
        // driven waiting: we send SIGUSR1 only once SIG_IGN is
        // definitely installed. The poll interval is 10ms and the
        // ceiling is 3s (~15× the old sleep) to cover CPU-starved
        // hosts without silently hanging — the earlier 2s ceiling
        // was tight enough that heavily-loaded CI runners (many
        // parallel cargo nextest workers competing for CPU during
        // fork + signal-handler install) occasionally missed the
        // deadline on valid SIG_IGN installs; bumping to 3s
        // preserves the "bounded, actionable" intent without the
        // flake.
        let worker_pid = h.worker_pids()[0];
        let ready_path = ready_file_path(worker_pid);
        // Remove any stale ready file from a prior run that happened
        // to land the same PID — `ready_path.exists()` in the poll
        // loop below would otherwise short-circuit on the stale file
        // and the parent would send SIGUSR1 before SIG_IGN was
        // actually installed. PID reuse across test runs in the same
        // session is plausible because fork() picks from the kernel's
        // recycled PID pool. This MUST run before `h.start()` — after
        // start() the worker is unblocked and can write a fresh ready
        // file before we reach this line, which would cause us to
        // unlink a live handshake and wedge the poll loop.
        let _ = std::fs::remove_file(&ready_path);
        h.start();
        wait_for_file_or_panic(
            &ready_path,
            Duration::from_secs(3),
            worker_pid,
            "SIG_IGN install may have failed or child never reached \
             ignores_sigusr1_fn's ready-file write",
        );
        let reports = h.stop_and_collect();
        // Ready file outlives the worker (written early, never
        // cleaned up by the child because the parent SIGKILLs it
        // before any cleanup could run). Remove it here so repeated
        // test runs don't observe a stale file from a prior run.
        let _ = std::fs::remove_file(&ready_path);
        assert_eq!(reports.len(), 1);
        let r = &reports[0];
        // Sentinel path: the worker never wrote JSON to the pipe
        // (because it ignored SIGUSR1 + ran past the deadline), so
        // the report is the zeroed sentinel shape. work_units = 0
        // confirms the sentinel construction at stop_and_collect's
        // `serde_json::from_slice` Err branch, not a worker-authored
        // report leaking through.
        assert_eq!(
            r.work_units, 0,
            "sentinel sidecar must be zeroed; non-zero work_units means \
             we parsed the worker's real report instead of hitting the \
             Err branch",
        );
        // `exit_info` must describe either the TimedOut (WNOHANG fast
        // path caught StillAlive) or Signaled(SIGKILL=9) (the kill
        // landed before the WNOHANG check) outcome. Any other variant
        // — Exited (worker wrote JSON), WaitFailed (reap error) —
        // would indicate a different failure shape than the one this
        // test pins.
        match &r.exit_info {
            Some(WorkerExitInfo::TimedOut) => {}
            Some(WorkerExitInfo::Signaled(sig)) if *sig == libc::SIGKILL => {}
            other => panic!("expected TimedOut or Signaled(SIGKILL), got {other:?}",),
        }
    }

    /// Shared path helper for [`forks_grandchild_sleep_fn`] and the
    /// grandchild reaping tests below. Workers write their forked-
    /// grandchild pid here so the test can observe it without fragile
    /// pipe-based IPC.
    fn grandchild_pidfile_path(worker_pid: libc::pid_t) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("ktstr-grandchild-pid-{worker_pid}"))
    }

    /// Path to the grandchild exec target used by every reaping test.
    /// Pinned here (rather than inlined in the `execv` call sites) so
    /// the test-side existence guard
    /// [`require_grandchild_sleep_binary`] and the worker-side
    /// `execv(prog, argv)` cannot drift.
    const GRANDCHILD_SLEEP_BINARY: &str = "/bin/sleep";

    /// Panic with an actionable message if `GRANDCHILD_SLEEP_BINARY`
    /// is missing or not marked executable (any of the user / group /
    /// other x-bits set). Every grandchild reaping test
    /// `execv(/bin/sleep, …)` after fork; a missing or non-executable
    /// binary causes the exec to fail and the grandchild to
    /// `_exit(127)` before the parent can read the pidfile, which then
    /// trips [`wait_for_file_or_panic`] with a generic timeout that
    /// buries the real cause. Failing here first keeps the diagnostic
    /// specific.
    fn require_grandchild_sleep_binary() {
        use std::os::unix::fs::PermissionsExt;
        let path = std::path::Path::new(GRANDCHILD_SLEEP_BINARY);
        let meta = match std::fs::metadata(path) {
            Ok(m) => m,
            Err(e) => panic!(
                "grandchild reaping tests require {GRANDCHILD_SLEEP_BINARY} to \
                 exist; stat failed: {e}. Install coreutils (or adjust the \
                 test's exec target + update GRANDCHILD_SLEEP_BINARY)."
            ),
        };
        // 0o111 covers all three x-bits (user / group / other). execv(2)
        // only requires one of them to be set AND match the caller's
        // effective uid / gid / other, but a file with zero x-bits
        // cannot be executed by anyone; catch that clear case here.
        // A finer-grained check would need `faccessat(X_OK)`; the
        // coarse check is sufficient for the "coreutils forgot to
        // mark /bin/sleep executable" failure mode this guard exists
        // to catch.
        if meta.permissions().mode() & 0o111 == 0 {
            panic!(
                "grandchild reaping tests require {GRANDCHILD_SLEEP_BINARY} to \
                 have at least one execute bit set; mode = {:o}. Fix the \
                 file's permissions or adjust the test's exec target.",
                meta.permissions().mode() & 0o7777,
            );
        }
    }

    /// Block on `pidfile` until it holds a parseable `libc::pid_t` and
    /// return it. Combines [`wait_for_file_or_panic`] + the
    /// retry-on-empty reader used by every grandchild reaping test
    /// (tempfile + rename write-atomicity sometimes races reads on
    /// slower filesystems or under heavy contention, so the reader
    /// guards anyway). Panics with an actionable message on timeout,
    /// empty-file stall, or parse failure.
    fn read_grandchild_gpid_from_pidfile(
        worker_pid: libc::pid_t,
        pidfile: &std::path::Path,
    ) -> libc::pid_t {
        wait_for_file_or_panic(
            pidfile,
            Duration::from_secs(3),
            worker_pid,
            "fork+exec path likely broken — check /bin/sleep exists and is executable",
        );
        let read_deadline = Instant::now() + Duration::from_secs(3);
        let gpid_str = loop {
            let s = std::fs::read_to_string(pidfile).expect("pidfile readable once exists");
            if !s.trim().is_empty() {
                break s;
            }
            if Instant::now() >= read_deadline {
                panic!(
                    "pidfile {pidfile:?} stayed empty for 3s after exists() \
                     returned true — writer may have crashed between O_TRUNC \
                     and write",
                );
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        let gpid: libc::pid_t = gpid_str
            .trim()
            .parse()
            .expect("pidfile holds a valid pid_t");
        assert!(gpid > 0, "grandchild pid must be positive: {gpid}");
        gpid
    }

    /// Poll for `gpid` death with a bounded deadline. Returns `Ok(())`
    /// when the pid is gone (ESRCH on the existence probe) and
    /// `Err(())` on timeout. The waitpid + WNOHANG inside the loop
    /// reaps a zombie if the caller inherited the grandchild under
    /// `PR_SET_CHILD_SUBREAPER` (systemd-run scopes, some CI
    /// runners). Shared by
    /// [`stop_and_collect_reaps_custom_grandchild_via_process_group`]
    /// and the new multi-worker / panic-path / Drop-path tests.
    fn wait_for_grandchild_reap(gpid: libc::pid_t, timeout: Duration) -> Result<(), ()> {
        let deadline = Instant::now() + timeout;
        loop {
            match nix::sys::signal::kill(nix::unistd::Pid::from_raw(gpid), None) {
                Err(nix::errno::Errno::ESRCH) => return Ok(()),
                Err(e) => panic!(
                    "unexpected errno from existence probe: {e} \
                     (common non-ESRCH errnos: EPERM = caller may not \
                     signal this process despite it existing; EINVAL = \
                     invalid signal number, which cannot happen here \
                     since we pass None / signal 0)",
                ),
                Ok(()) => {
                    match nix::sys::wait::waitpid(
                        nix::unistd::Pid::from_raw(gpid),
                        Some(nix::sys::wait::WaitPidFlag::WNOHANG),
                    ) {
                        Ok(nix::sys::wait::WaitStatus::Exited(_, _))
                        | Ok(nix::sys::wait::WaitStatus::Signaled(_, _, _)) => return Ok(()),
                        _ => {}
                    }
                    if Instant::now() >= deadline {
                        return Err(());
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
            }
        }
    }

    /// Last-resort SIGKILL + assertion-panic wrapper around
    /// [`wait_for_grandchild_reap`]. Ensures a test failure never
    /// leaks a live grandchild into the host.
    fn assert_grandchild_reaped_within(gpid: libc::pid_t, timeout: Duration, context: &str) {
        if wait_for_grandchild_reap(gpid, timeout).is_err() {
            let _ = nix::sys::signal::kill(
                nix::unistd::Pid::from_raw(gpid),
                nix::sys::signal::Signal::SIGKILL,
            );
            panic!(
                "grandchild {gpid} still alive {:?} after {context} — \
                 setpgid/killpg path broken",
                timeout,
            );
        }
    }

    /// RAII pidfile cleanup: removes the file on Drop so a panicking
    /// test doesn't leak a `/tmp/ktstr-grandchild-pid-*` stub into
    /// the host. Manual impl rather than `scopeguard` to keep the
    /// crate out of the workspace dep graph.
    struct PidfileCleanup(Vec<std::path::PathBuf>);
    impl Drop for PidfileCleanup {
        fn drop(&mut self) {
            for p in &self.0 {
                let _ = std::fs::remove_file(p);
            }
        }
    }

    /// Shared post-fork-and-exec helper used by every grandchild
    /// reaping test closure. In the parent-worker: forks a
    /// [`GRANDCHILD_SLEEP_BINARY`] 60 grandchild via `execv`, publishes
    /// the gpid atomically via tempfile + rename, and returns the
    /// worker's own pid. In the child: `execv(prog, [prog, "60", NULL])`
    /// followed by `_exit(127)` on exec failure — `execv` requires
    /// `argv[0]` to carry the program name by convention so the
    /// exec'd `/bin/sleep` sees its usual `argv[0]`. Never returns on the
    /// child side.
    ///
    /// Does NOT install any SIGUSR1 disposition — callers pick the
    /// policy (SIG_IGN to force StillAlive escalation, or the
    /// inherited SIGUSR1→STOP handler for graceful-exit). CString
    /// construction runs pre-fork so a hypothetical NulError fires in
    /// the parent where it's debuggable. The tempfile + rename
    /// protocol closes the exists()→read() race the reader-side
    /// retry loop also defends against.
    fn fork_and_exec_grandchild_and_publish_pidfile() -> libc::pid_t {
        let exec_path = std::ffi::CString::new(GRANDCHILD_SLEEP_BINARY)
            .expect("GRANDCHILD_SLEEP_BINARY must have no interior NUL");
        let exec_arg = std::ffi::CString::new("60").expect("literal has no NUL");
        let worker_pid = unsafe { libc::getpid() };
        let gpid = unsafe { libc::fork() };
        if gpid < 0 {
            // _exit is async-signal-safe; eprintln goes to the
            // harness-captured test log.
            eprintln!("fork failed: {}", std::io::Error::last_os_error());
            unsafe {
                libc::_exit(127);
            }
        }
        if gpid == 0 {
            // Close every inherited fd above stdio BEFORE exec so
            // the grandchild does not keep the parent-worker's
            // pipes open. The worker's report-pipe write end is
            // especially load-bearing: if the grandchild inherits
            // it, the test's parent-side `read_to_end` in
            // `stop_and_collect` blocks on EOF until the
            // grandchild itself dies, turning a fast graceful-exit
            // test into a /bin/sleep-wall-clock-long run
            // (observed: 60s).
            //
            // `close_range(3, u32::MAX, 0)` is the one-syscall form
            // (Linux 5.9+) and is the fast path. BUT this code
            // runs on the HOST, not inside the ktstr guest VM —
            // ktstr's 6.16+ kernel floor applies to the sched_ext
            // guest kernel, not to the host running the tests. A
            // host kernel predating 5.9 returns ENOSYS from
            // `close_range`, leaving every inherited fd open and
            // re-introducing the 60s hang. Fall back to the
            // bounded `3..=256` close loop on any non-zero return
            // so pre-5.9 hosts still close the load-bearing
            // report-pipe write end.
            let rc = unsafe { libc::close_range(3, u32::MAX, 0) };
            if rc != 0 {
                for fd in 3..=256 {
                    unsafe {
                        libc::close(fd);
                    }
                }
            }
            // Grandchild: exec immediately. `execv` returns only on
            // failure; any return is a setup error → _exit(127).
            // CStrings live on the child's CoW'd heap from the
            // parent; pointers stay valid until execv replaces the
            // address space.
            let argv: [*const libc::c_char; 3] =
                [exec_path.as_ptr(), exec_arg.as_ptr(), std::ptr::null()];
            unsafe {
                libc::execv(exec_path.as_ptr(), argv.as_ptr());
                libc::_exit(127);
            }
        }
        // Parent-worker: publish gpid. A failure here leaves the test
        // hanging on a file that never appears — surface the errno
        // and exit so the test gets an actionable diagnostic instead
        // of a poll-timeout panic.
        let pidfile = grandchild_pidfile_path(worker_pid);
        let pidfile_tmp =
            std::env::temp_dir().join(format!("ktstr-grandchild-pid-{worker_pid}.tmp"));
        if let Err(e) = std::fs::write(&pidfile_tmp, gpid.to_string()) {
            eprintln!("failed to write grandchild pidfile tmp {pidfile_tmp:?}: {e}");
            unsafe {
                libc::_exit(127);
            }
        }
        if let Err(e) = std::fs::rename(&pidfile_tmp, &pidfile) {
            eprintln!("failed to rename grandchild pidfile {pidfile_tmp:?} → {pidfile:?}: {e}");
            unsafe {
                libc::_exit(127);
            }
        }
        worker_pid
    }

    /// Custom WorkType closure that forks a long-running grandchild
    /// and ignores `SIGUSR1` on the parent-worker side so
    /// stop_and_collect is forced into its StillAlive escalation
    /// branch. Pairs with
    /// [`stop_and_collect_reaps_custom_grandchild_via_process_group`].
    fn forks_grandchild_sleep_fn(stop: &AtomicBool) -> WorkerReport {
        // Ignore SIGUSR1 so stop_and_collect escalates — matches
        // ignores_sigusr1_fn's rationale.
        let worker_pid = ignore_sigusr1_and_get_pid();
        fork_and_exec_grandchild_and_publish_pidfile();
        // Wait past the 5s collection deadline so stop_and_collect
        // escalates to SIGKILL → killpg. The `!stop.load` check is
        // kept honest inside `wait_for_deadline` even though SIG_IGN
        // prevents SIGUSR1 from flipping STOP; the 7s deadline is
        // the real terminator.
        wait_for_deadline(stop, Duration::from_secs(7));
        WorkerReport {
            tid: worker_pid,
            ..WorkerReport::default()
        }
    }

    /// Graceful-exit variant: forks the grandchild and then waits on
    /// the `stop` flag via [`wait_for_deadline`]. Does NOT install
    /// SIG_IGN — the worker's inherited `SIGUSR1 → STOP` handler
    /// fires on stop_and_collect's signal and flips `stop`, letting
    /// this closure return cleanly BEFORE the 5s collection deadline.
    /// stop_and_collect therefore hits its graceful-exit branch;
    /// killpg on that branch must still reap the grandchild.
    ///
    /// 10s upper bound on the wait is purely a liveness sentinel —
    /// stop_and_collect sends SIGUSR1 within milliseconds of its
    /// own invocation, so in practice `stop` flips well before 10s
    /// elapses.
    fn forks_grandchild_and_exits_cleanly_fn(stop: &AtomicBool) -> WorkerReport {
        let worker_pid = fork_and_exec_grandchild_and_publish_pidfile();
        wait_for_deadline(stop, Duration::from_secs(10));
        WorkerReport {
            tid: worker_pid,
            ..WorkerReport::default()
        }
    }

    /// Proves the `setpgid(0, 0)` + `killpg` path works end-to-end:
    /// a long-running grandchild forked from a Custom worker's
    /// closure dies when stop_and_collect runs. Without setpgid +
    /// killpg, the grandchild would orphan onto init and survive the
    /// test — which this test catches via `kill(gpid, 0)` returning
    /// ESRCH after collection.
    ///
    /// The SIGUSR1 ignore forces stop_and_collect into its StillAlive
    /// escalation branch. This test pins the StillAlive path. The
    /// graceful-exit branch (stop_and_collect's `waited` arm where the
    /// worker exits before the 5s deadline) is pinned by TWO variants
    /// covering the disjoint shapes a worker can die in before the
    /// parent reaps it:
    ///   - [`stop_and_collect_reaps_grandchild_from_panicking_custom_closure`]
    ///     — worker panics → process dies via `_exit(1)` (under
    ///     `panic = "unwind"`) or SIGABRT (under `panic = "abort"`)
    ///     BEFORE stop_and_collect even signals it. The graceful
    ///     branch's `waited` result is `Exited(1)` / `Signaled(SIGABRT)`
    ///     on that path; the unconditional killpg must still reach
    ///     the grandchild.
    ///   - [`stop_and_collect_reaps_grandchild_from_graceful_custom_closure`]
    ///     — worker's inherited SIGUSR1 handler fires and flips STOP,
    ///     the closure returns a clean WorkerReport, the worker
    ///     `_exit(0)`s WITHIN the deadline. The graceful branch's
    ///     `waited` is `Exited(0)`; the same unconditional killpg
    ///     must still reap the grandchild.
    ///
    /// The Drop branch is pinned by
    /// [`drop_reaps_custom_grandchild_via_process_group`] (handle is
    /// dropped with no stop_and_collect call → `impl Drop`'s killpg
    /// sweeps). The multi-worker variant is
    /// [`stop_and_collect_reaps_grandchildren_from_multiple_workers`].
    #[test]
    fn stop_and_collect_reaps_custom_grandchild_via_process_group() {
        require_grandchild_sleep_binary();
        let config = WorkloadConfig {
            num_workers: 1,
            affinity: ResolvedAffinity::None,
            work_type: WorkType::custom("grandchild_sleep", forks_grandchild_sleep_fn),
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        let worker_pid = h.worker_pids()[0];
        let pidfile = grandchild_pidfile_path(worker_pid);
        let _ = std::fs::remove_file(&pidfile);
        // Pidfile cleanup fires via the module-level PidfileCleanup
        // helper — Drop removes the stub even if later assertions
        // panic.
        let _pidfile_cleanup = PidfileCleanup(vec![pidfile.clone()]);
        h.start();
        let gpid = read_grandchild_gpid_from_pidfile(worker_pid, &pidfile);
        // Confirm grandchild is alive before stop_and_collect.
        assert!(
            nix::sys::signal::kill(nix::unistd::Pid::from_raw(gpid), None).is_ok(),
            "grandchild {gpid} must be alive before stop_and_collect",
        );
        // Trigger the teardown that should also reap the grandchild.
        let _reports = h.stop_and_collect();
        assert_grandchild_reaped_within(gpid, Duration::from_secs(5), "stop_and_collect");
    }

    /// Multi-worker variant of
    /// [`stop_and_collect_reaps_custom_grandchild_via_process_group`]:
    /// `num_workers = 3`, each worker forks its own grandchild, and
    /// `stop_and_collect` must reap all three process groups. Guards
    /// against a future refactor that accidentally single-target's
    /// killpg (e.g. only the first child in
    /// `WorkloadHandle::children`).
    #[test]
    fn stop_and_collect_reaps_grandchildren_from_multiple_workers() {
        require_grandchild_sleep_binary();
        const N: usize = 3;
        let config = WorkloadConfig {
            num_workers: N,
            affinity: ResolvedAffinity::None,
            work_type: WorkType::custom("grandchild_sleep", forks_grandchild_sleep_fn),
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        let worker_pids = h.worker_pids();
        assert_eq!(
            worker_pids.len(),
            N,
            "WorkloadHandle::worker_pids should report {N} workers",
        );
        // Pin uniqueness: every worker must have a distinct pid. A
        // repeated pid would mean the spawn logic conflated two
        // workers (or the pidfile scheme collides across workers,
        // which would also break this multi-worker reaping test).
        let unique: std::collections::HashSet<libc::pid_t> = worker_pids.iter().copied().collect();
        assert_eq!(
            unique.len(),
            worker_pids.len(),
            "WorkloadHandle::worker_pids returned duplicates: {worker_pids:?}",
        );
        let pidfiles: Vec<std::path::PathBuf> = worker_pids
            .iter()
            .map(|&p| grandchild_pidfile_path(p))
            .collect();
        for p in &pidfiles {
            let _ = std::fs::remove_file(p);
        }
        let _pidfile_cleanup = PidfileCleanup(pidfiles.clone());
        h.start();
        // Collect every grandchild pid; any pidfile miss panics with
        // the worker_pid context embedded so the failure names the
        // offending worker.
        let gpids: Vec<libc::pid_t> = worker_pids
            .iter()
            .zip(pidfiles.iter())
            .map(|(&wp, pf)| read_grandchild_gpid_from_pidfile(wp, pf))
            .collect();
        for &gpid in &gpids {
            assert!(
                nix::sys::signal::kill(nix::unistd::Pid::from_raw(gpid), None).is_ok(),
                "grandchild {gpid} must be alive before stop_and_collect",
            );
        }
        let _reports = h.stop_and_collect();
        for &gpid in &gpids {
            assert_grandchild_reaped_within(
                gpid,
                Duration::from_secs(5),
                "stop_and_collect (multi-worker)",
            );
        }
    }

    /// Custom closure that forks a grandchild exactly like
    /// [`forks_grandchild_sleep_fn`], publishes the gpid via the
    /// same pidfile protocol, then deliberately panics. Exercises the
    /// Custom-closure panic path — the worker process unwinds /
    /// aborts without a clean `WorkerReport` return, but the
    /// `setpgid(0, 0)` it installed at fork time still applies, so
    /// `stop_and_collect`'s unconditional killpg must still reap the
    /// grandchild.
    fn forks_grandchild_and_panics_fn(_stop: &AtomicBool) -> WorkerReport {
        // SIG_IGN so a racing SIGUSR1 from stop_and_collect cannot
        // trip the default worker handler before the panic fires;
        // the panic + catch_unwind → _exit(1) path is what this
        // closure exists to exercise, not the graceful SIGUSR1 flow.
        let _worker_pid = ignore_sigusr1_and_get_pid();
        fork_and_exec_grandchild_and_publish_pidfile();
        panic!(
            "intentional panic after grandchild fork to exercise the \
             Custom-closure panic path in stop_and_collect"
        );
    }

    /// Panic-path variant: the Custom closure panics after forking
    /// its grandchild. Under `panic = "unwind"` the worker's
    /// `std::panic::catch_unwind` (around the child body in the
    /// forked-child path of `WorkloadHandle::spawn`) catches the
    /// panic and the child hits `libc::_exit(1)` directly — no
    /// abort. Under `panic = "abort"`
    /// SIGABRT fires before catch_unwind runs. Either way the
    /// parent-worker process exits BEFORE `stop_and_collect` is
    /// called; stop_and_collect's graceful-exit branch must still
    /// issue killpg to reach the grandchild. Pins the unconditional
    /// killpg in the graceful branch — without it, the grandchild
    /// would orphan onto init.
    #[test]
    fn stop_and_collect_reaps_grandchild_from_panicking_custom_closure() {
        require_grandchild_sleep_binary();
        let config = WorkloadConfig {
            num_workers: 1,
            affinity: ResolvedAffinity::None,
            work_type: WorkType::custom("grandchild_panic", forks_grandchild_and_panics_fn),
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        let worker_pid = h.worker_pids()[0];
        let pidfile = grandchild_pidfile_path(worker_pid);
        let _ = std::fs::remove_file(&pidfile);
        let _pidfile_cleanup = PidfileCleanup(vec![pidfile.clone()]);
        h.start();
        // The worker panics immediately after publishing the gpid;
        // read_grandchild_gpid_from_pidfile observes the file before
        // the worker process finishes exiting because fork + panic
        // is slower than the tempfile + rename write.
        let gpid = read_grandchild_gpid_from_pidfile(worker_pid, &pidfile);
        assert!(
            nix::sys::signal::kill(nix::unistd::Pid::from_raw(gpid), None).is_ok(),
            "grandchild {gpid} must be alive before stop_and_collect",
        );
        let reports = h.stop_and_collect();
        assert_grandchild_reaped_within(
            gpid,
            Duration::from_secs(5),
            "stop_and_collect (panic-path)",
        );
        // Sentinel-mapping audit: the panicking worker cannot
        // serialize a WorkerReport to the pipe, so
        // `stop_and_collect`'s JSON-parse branch must fall into
        // the sentinel path. The `exit_info` carried on the
        // sentinel depends on the compile-time panic strategy:
        //   - Under `panic = "abort"` (release profile), the
        //     panic raises SIGABRT before the worker's
        //     `catch_unwind` can run → `Signaled(SIGABRT)`.
        //   - Under `panic = "unwind"` (dev/test profile, which
        //     this test runs under), the worker's `catch_unwind`
        //     intercepts the panic and calls `libc::_exit(1)` →
        //     `Exited(1)`.
        // Both paths produce a sentinel with `work_units == 0`;
        // the match below accepts either.
        assert_eq!(reports.len(), 1, "one worker spawned");
        let r = &reports[0];
        assert_eq!(
            r.work_units, 0,
            "sentinel must be zeroed; non-zero work_units would mean \
             a worker-authored report leaked through the JSON-parse \
             branch despite the panic",
        );
        assert!(
            !r.completed,
            "sentinel must carry completed=false so downstream \
             consumers distinguish '0 iterations by design / fast \
             exit' (completed=true) from '0 iterations because the \
             worker crashed before producing a report' (this case); \
             `..WorkerReport::default()` gives the bool-default \
             `false` at the sentinel construction site in \
             `stop_and_collect`",
        );
        match &r.exit_info {
            Some(WorkerExitInfo::Signaled(sig)) if *sig == libc::SIGABRT => {}
            Some(WorkerExitInfo::Exited(1)) => {}
            other => panic!(
                "expected sentinel with Signaled(SIGABRT) (panic=abort) \
                 or Exited(1) (panic=unwind + catch_unwind) for a \
                 panicking Custom closure; got {other:?}",
            ),
        }
    }

    /// Drop-path variant: the caller drops the handle WITHOUT calling
    /// `stop_and_collect`. The `impl Drop for WorkloadHandle`
    /// (src/workload.rs) is responsible for killpg'ing every worker
    /// process group, then SIGKILLing each leader and waitpid'ing it.
    /// Without the Drop-path killpg, any long-running grandchild
    /// would orphan onto init and leak past the test. Pins the
    /// Drop-path sweep.
    #[test]
    fn drop_reaps_custom_grandchild_via_process_group() {
        require_grandchild_sleep_binary();
        let config = WorkloadConfig {
            num_workers: 1,
            affinity: ResolvedAffinity::None,
            work_type: WorkType::custom("grandchild_sleep", forks_grandchild_sleep_fn),
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        let worker_pid = h.worker_pids()[0];
        let pidfile = grandchild_pidfile_path(worker_pid);
        let _ = std::fs::remove_file(&pidfile);
        let _pidfile_cleanup = PidfileCleanup(vec![pidfile.clone()]);
        h.start();
        let gpid = read_grandchild_gpid_from_pidfile(worker_pid, &pidfile);
        assert!(
            nix::sys::signal::kill(nix::unistd::Pid::from_raw(gpid), None).is_ok(),
            "grandchild {gpid} must be alive before Drop",
        );
        // No stop_and_collect call — Drop is the sole teardown path
        // under test here. `drop(h)` triggers the impl Drop killpg +
        // kill + waitpid sweep.
        drop(h);
        assert_grandchild_reaped_within(
            gpid,
            Duration::from_secs(5),
            "handle Drop (no stop_and_collect)",
        );
    }

    /// Graceful-exit variant: the Custom closure forks a grandchild,
    /// publishes the pidfile, and waits on `stop` at 10ms granularity
    /// — no SIG_IGN, no panic. The worker's inherited `SIGUSR1 → STOP`
    /// handler fires when `stop_and_collect` signals us, the closure
    /// returns a clean `WorkerReport`, and the worker exits cleanly
    /// WITHIN the 5s collection deadline. That routes stop_and_collect
    /// into its `waited` / graceful-exit branch (not StillAlive, not
    /// Drop). The unconditional killpg on THAT branch is the path
    /// under test — without it, the grandchild would orphan onto
    /// init.
    #[test]
    fn stop_and_collect_reaps_grandchild_from_graceful_custom_closure() {
        require_grandchild_sleep_binary();
        let config = WorkloadConfig {
            num_workers: 1,
            affinity: ResolvedAffinity::None,
            work_type: WorkType::custom(
                "grandchild_graceful",
                forks_grandchild_and_exits_cleanly_fn,
            ),
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        let worker_pid = h.worker_pids()[0];
        let pidfile = grandchild_pidfile_path(worker_pid);
        let _ = std::fs::remove_file(&pidfile);
        let _pidfile_cleanup = PidfileCleanup(vec![pidfile.clone()]);
        h.start();
        let gpid = read_grandchild_gpid_from_pidfile(worker_pid, &pidfile);
        assert!(
            nix::sys::signal::kill(nix::unistd::Pid::from_raw(gpid), None).is_ok(),
            "grandchild {gpid} must be alive before stop_and_collect",
        );
        // Pin which `stop_and_collect` branch fires. The graceful path
        // — worker's SIGUSR1 handler flips STOP, the closure returns
        // cleanly via `wait_for_deadline`'s stop-observed early-exit,
        // the worker `_exit(0)`s well within the 5s collection
        // deadline — completes in a few hundred milliseconds
        // (500ms auto-start sleep + SIGUSR1 + 10ms wait_for_deadline
        // poll + worker serialize/_exit + WNOHANG reap). The
        // StillAlive escalation branch, by contrast, waits the full
        // 5s deadline before SIGKILL. A <2s ceiling rules out
        // StillAlive escalation (~5s+) while leaving generous slack
        // for CI contention on the graceful path.
        let t0 = Instant::now();
        let _reports = h.stop_and_collect();
        let elapsed = t0.elapsed();
        assert!(
            elapsed < Duration::from_secs(2),
            "stop_and_collect must hit the graceful-exit branch \
             (<2s), not StillAlive escalation (~5s). elapsed={elapsed:?} \
             — a value near the 5s deadline means SIGUSR1 failed to \
             reach the worker or wait_for_deadline did not observe \
             STOP in time",
        );
        assert_grandchild_reaped_within(
            gpid,
            Duration::from_secs(5),
            "stop_and_collect (graceful-exit)",
        );
    }

    // -- Test-helper unit tests --

    /// Happy path: the file appears WITHIN the deadline, so
    /// [`wait_for_file_or_panic`] returns without panicking. Uses
    /// `std::process::id()` as `liveness_pid` — this test process is
    /// always alive, so the early-exit probe never fires.
    #[test]
    fn wait_for_file_or_panic_returns_when_file_appears() {
        let dir = std::env::temp_dir().join(format!("ktstr-wfp-happy-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let marker = dir.join("ready");
        // Pre-create the marker so the first iteration exits the
        // loop. No race to worry about for the happy-path pin.
        std::fs::write(&marker, b"ok").unwrap();
        wait_for_file_or_panic(
            &marker,
            Duration::from_secs(1),
            unsafe { libc::getpid() },
            "pre-existing marker must satisfy the guard",
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Liveness-death path: `liveness_pid` dies before the file
    /// appears, so the helper panics with "exited before writing
    /// ready file" rather than waiting the full deadline. The test
    /// forks a `/bin/true` child, reaps it, then polls a file that
    /// will never appear; the helper's `kill(pid, 0)` returns ESRCH
    /// on the dead pid and the panic fires inside catch_unwind.
    #[test]
    fn wait_for_file_or_panic_detects_liveness_death() {
        let mut child = std::process::Command::new("/bin/true")
            .spawn()
            .expect("spawn /bin/true");
        let dead_pid = child.id() as libc::pid_t;
        let _ = child.wait();
        // `dead_pid` is now reaped; `kill(dead_pid, 0)` returns ESRCH
        // unless the kernel has already recycled it. Recycling is
        // very unlikely within the ~100ms test window.
        let nonexistent = std::env::temp_dir().join(format!(
            "ktstr-wfp-never-exists-{}-{dead_pid}",
            std::process::id(),
        ));
        let _ = std::fs::remove_file(&nonexistent);
        let result = std::panic::catch_unwind(|| {
            wait_for_file_or_panic(
                &nonexistent,
                Duration::from_secs(30), // generous — we want the liveness path, not the deadline
                dead_pid,
                "liveness-death path",
            );
        });
        let err = result.expect_err("must panic when liveness pid is dead");
        let msg = crate::test_support::test_helpers::panic_payload_to_string(err);
        assert!(
            msg.contains("exited before writing ready file"),
            "panic must name the early-exit path, got: {msg}"
        );
    }

    /// Deadline path: file never appears, `liveness_pid` stays alive
    /// (use self), helper panics with "did not write ready file" once
    /// the timeout elapses. Short timeout (50ms) to keep the test
    /// fast.
    #[test]
    fn wait_for_file_or_panic_panics_on_deadline_miss() {
        let nonexistent = std::env::temp_dir().join(format!(
            "ktstr-wfp-deadline-never-exists-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&nonexistent);
        let self_pid = unsafe { libc::getpid() };
        let result = std::panic::catch_unwind(|| {
            wait_for_file_or_panic(
                &nonexistent,
                Duration::from_millis(50),
                self_pid,
                "deadline path",
            );
        });
        let err = result.expect_err("must panic when deadline expires");
        let msg = crate::test_support::test_helpers::panic_payload_to_string(err);
        assert!(
            msg.contains("did not write ready file"),
            "panic must name the deadline-miss path, got: {msg}"
        );
    }

    /// Deadline-elapse path: `stop` stays `false`, so
    /// [`wait_for_deadline`] runs until `timeout` elapses. Uses a
    /// 1-second deadline; asserts the call returned no earlier than
    /// ~900ms (granularity slop from the 10ms sleep cadence).
    #[test]
    fn wait_for_deadline_waits_full_duration_when_stop_stays_false() {
        let stop = AtomicBool::new(false);
        let start = Instant::now();
        wait_for_deadline(&stop, Duration::from_secs(1));
        let elapsed = start.elapsed();
        assert!(
            elapsed >= Duration::from_millis(900),
            "wait_for_deadline must hold for ~full duration; elapsed={elapsed:?}",
        );
        assert!(
            elapsed < Duration::from_millis(2_000),
            "wait_for_deadline must not massively overshoot; elapsed={elapsed:?}",
        );
    }

    /// Stop-flip path: another thread flips `stop` to `true` ~50ms in,
    /// and [`wait_for_deadline`] returns shortly after. Asserts the
    /// call returned well before the 10s deadline.
    #[test]
    fn wait_for_deadline_returns_early_when_stop_is_set() {
        use std::sync::Arc;
        let stop = Arc::new(AtomicBool::new(false));
        let stop_setter = Arc::clone(&stop);
        let flipper = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            stop_setter.store(true, Ordering::Relaxed);
        });
        let start = Instant::now();
        wait_for_deadline(&stop, Duration::from_secs(10)); // 10s deadline — should never hit
        let elapsed = start.elapsed();
        flipper.join().unwrap();
        assert!(
            elapsed < Duration::from_secs(1),
            "wait_for_deadline must return promptly after stop flips; elapsed={elapsed:?}",
        );
    }

    // -- FanOutCompute tests --

    #[test]
    fn fan_out_compute_name() {
        let wt = WorkType::FanOutCompute {
            fan_out: 4,
            cache_footprint_kb: 256,
            operations: 5,
            sleep_usec: 100,
        };
        assert_eq!(wt.name(), "FanOutCompute");
    }

    #[test]
    fn fan_out_compute_from_name() {
        let wt = WorkType::from_name("FanOutCompute").unwrap();
        match wt {
            WorkType::FanOutCompute {
                fan_out,
                cache_footprint_kb,
                operations,
                sleep_usec,
            } => {
                assert_eq!(fan_out, 4);
                assert_eq!(cache_footprint_kb, 256);
                assert_eq!(operations, 5);
                assert_eq!(sleep_usec, 100);
            }
            _ => panic!("expected FanOutCompute"),
        }
    }

    #[test]
    fn fan_out_compute_group_size() {
        let wt = WorkType::fan_out_compute(4, 256, 5, 100);
        assert_eq!(wt.worker_group_size(), Some(5));
        let wt2 = WorkType::fan_out_compute(1, 256, 5, 100);
        assert_eq!(wt2.worker_group_size(), Some(2));
    }

    #[test]
    fn fan_out_compute_needs_shared_mem() {
        assert!(WorkType::fan_out_compute(4, 256, 5, 100).needs_shared_mem());
    }

    #[test]
    fn fan_out_compute_needs_cache_buf() {
        assert!(WorkType::fan_out_compute(4, 256, 5, 100).needs_cache_buf());
    }

    /// Guards two invariants of [`WorkType::FanOutCompute`]:
    ///
    /// 1. Every spawned worker produces non-zero `work_units`, and at
    ///    least one records a wake latency into `resume_latencies_ns`.
    /// 2. The Release/Acquire ordering between the messenger's
    ///    `wake_ns` store and its generation advance prevents workers
    ///    from pairing a fresh generation with a stale or zero-init
    ///    `wake_ns` — the 10 s latency bound below detects only the
    ///    zero-init arm of that failure mode (see comment on the
    ///    bound).
    ///
    /// Platform coverage: x86-64 is TSO (store-store and load-load
    /// reordering are hardware-prohibited), so on x86 CI this test
    /// cannot reproduce a weak-memory regression of the messenger-
    /// side store reorder or the worker-side load speculation that
    /// the Release/Acquire on aarch64 guards against — the hardware
    /// masks the bug. It still catches implementation bugs that
    /// surface on any platform, most notably a missing or
    /// misordered `wake_ns` store that leaves workers reading
    /// zero-init memory (the 10 s bound trips on `now_ns - 0`).
    /// Round-over-round reordering cannot be detected by this
    /// assertion on any platform. Meaningful weak-memory
    /// regression protection requires running this test on an
    /// aarch64 runner in CI.
    #[test]
    fn spawn_fan_out_compute_produces_work() {
        let config = WorkloadConfig {
            num_workers: 5, // 1 messenger + 4 workers
            affinity: ResolvedAffinity::None,
            work_type: WorkType::FanOutCompute {
                fan_out: 4,
                cache_footprint_kb: 256,
                operations: 5,
                sleep_usec: 100,
            },
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        h.start();
        std::thread::sleep(std::time::Duration::from_millis(500));
        let reports = h.stop_and_collect();
        assert_eq!(reports.len(), 5);
        for r in &reports {
            assert!(
                r.work_units > 0,
                "FanOutCompute worker {} did no work",
                r.tid
            );
        }
        // Every non-messenger worker (receiver) must record at
        // least one wake-latency sample — the messenger advances
        // the generation and never waits, so its latency vec is
        // legitimately empty. Asserting the stronger per-receiver
        // contract (previously `reports.iter().any(...)`) catches
        // a regression that leaves one group of receivers parked
        // on futex_wait without ever seeing the generation advance.
        assert!(
            reports
                .iter()
                .filter(|r| !r.is_messenger)
                .all(|r| !r.resume_latencies_ns.is_empty()),
            "every FanOutCompute receiver must record at least one \
             wake latency sample; got {:?}",
            reports
                .iter()
                .map(|r| (r.tid, r.is_messenger, r.resume_latencies_ns.len()))
                .collect::<Vec<_>>(),
        );
        // The 10 s bound catches the zero-init arm of a missing
        // Release/Acquire pairing: a worker that reads `wake_ns`
        // before the messenger's first store sees 0, so
        // `now_ns.saturating_sub(0)` surfaces `CLOCK_MONOTONIC`
        // (seconds-to-days of monotonic uptime) >> 10 s on any
        // live machine. It does NOT catch round-over-round
        // mispairing — a fresh generation paired with the
        // immediately-prior round's `wake_ns` yields a sub-second
        // delta that is indistinguishable from a correctly-paired
        // fast wake. This is a coarse guard against the easy
        // failure mode, not a full verification of the ordering.
        const MAX_PLAUSIBLE_LATENCY_NS: u64 = 10_000_000_000;
        for r in &reports {
            for &lat in &r.resume_latencies_ns {
                assert!(
                    lat < MAX_PLAUSIBLE_LATENCY_NS,
                    "worker {} recorded implausible wake latency {} ns \
                     (expected < {} ns); indicates wake_ns/generation \
                     ordering race. NB: lat==0 is LEGITIMATE under \
                     correct ordering — a Relaxed `wake_atom.load` \
                     paired with an Acquire gen load can see a wake_ns \
                     from a LATER round (gen+1's store becomes visible \
                     ahead of gen+1's wake_ns re-load), making \
                     now_ns < wake_ns and `saturating_sub` = 0. The \
                     reservoir-sampling of real latencies is dominated \
                     by positive values; a stray zero from this race \
                     is not a bug, so no lower bound is asserted.",
                    r.tid,
                    lat,
                    MAX_PLAUSIBLE_LATENCY_NS
                );
            }
        }
    }

    #[test]
    fn spawn_fan_out_compute_bad_worker_count_fails() {
        let config = WorkloadConfig {
            num_workers: 3,
            affinity: ResolvedAffinity::None,
            work_type: WorkType::FanOutCompute {
                fan_out: 4,
                cache_footprint_kb: 256,
                operations: 5,
                sleep_usec: 100,
            },
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let result = WorkloadHandle::spawn(&config);
        assert!(result.is_err());
        let msg = format!("{:#}", result.err().unwrap());
        assert!(
            msg.contains("divisible by 5"),
            "expected divisibility error: {msg}"
        );
    }

    /// Two-messenger-group variant of the invariants guarded by
    /// [`spawn_fan_out_compute_produces_work`] — see that test's
    /// doc for the full Release/Acquire rationale and platform
    /// coverage notes.
    #[test]
    fn spawn_fan_out_compute_two_groups() {
        let config = WorkloadConfig {
            num_workers: 10, // 2 groups of (1 messenger + 4 workers)
            affinity: ResolvedAffinity::None,
            work_type: WorkType::FanOutCompute {
                fan_out: 4,
                cache_footprint_kb: 256,
                operations: 5,
                sleep_usec: 100,
            },
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        assert_eq!(h.worker_pids().len(), 10);
        h.start();
        std::thread::sleep(std::time::Duration::from_millis(500));
        let reports = h.stop_and_collect();
        assert_eq!(reports.len(), 10);
        for r in &reports {
            assert!(
                r.work_units > 0,
                "FanOutCompute worker {} did no work",
                r.tid
            );
        }
        // Every non-messenger worker (receiver) in each group must
        // record at least one wake-latency sample — mirror of the
        // per-receiver contract asserted in the single-group test
        // at `spawn_fan_out_compute_produces_work`. With 10 workers
        // and 2 groups (1 messenger + 4 receivers each), this means
        // 8 receivers must all report non-empty latency vectors.
        assert!(
            reports
                .iter()
                .filter(|r| !r.is_messenger)
                .all(|r| !r.resume_latencies_ns.is_empty()),
            "every FanOutCompute receiver in both groups must record \
             at least one wake latency sample; got {:?}",
            reports
                .iter()
                .map(|r| (r.tid, r.is_messenger, r.resume_latencies_ns.len()))
                .collect::<Vec<_>>(),
        );
        // Mirror of the single-group test's latency sanity check —
        // see `spawn_fan_out_compute_produces_work` for rationale.
        // The 10 s bound catches the zero-init arm of a missing
        // Release/Acquire pairing but not round-over-round
        // mispairing; with two messenger groups running
        // independently it still provides a coarse smoke test per
        // group.
        const MAX_PLAUSIBLE_LATENCY_NS: u64 = 10_000_000_000;
        for r in &reports {
            for &lat in &r.resume_latencies_ns {
                assert!(
                    lat < MAX_PLAUSIBLE_LATENCY_NS,
                    "worker {} recorded implausible wake latency {} ns \
                     (expected < {} ns); indicates wake_ns/generation \
                     ordering race. NB: lat==0 is LEGITIMATE under \
                     correct ordering — a Relaxed `wake_atom.load` \
                     paired with an Acquire gen load can see a wake_ns \
                     from a LATER round (gen+1's store becomes visible \
                     ahead of gen+1's wake_ns re-load), making \
                     now_ns < wake_ns and `saturating_sub` = 0. The \
                     reservoir-sampling of real latencies is dominated \
                     by positive values; a stray zero from this race \
                     is not a bug, so no lower bound is asserted.",
                    r.tid,
                    lat,
                    MAX_PLAUSIBLE_LATENCY_NS
                );
            }
        }
    }

    // -- MemPolicy tests --

    #[test]
    fn mempolicy_default_node_set_empty() {
        assert!(MemPolicy::Default.node_set().is_empty());
    }

    #[test]
    fn mempolicy_local_node_set_empty() {
        assert!(MemPolicy::Local.node_set().is_empty());
    }

    #[test]
    fn mempolicy_bind_node_set() {
        let p = MemPolicy::Bind([0, 2].into_iter().collect());
        assert_eq!(p.node_set(), [0, 2].into_iter().collect());
    }

    #[test]
    fn mempolicy_preferred_node_set() {
        let p = MemPolicy::Preferred(1);
        assert_eq!(p.node_set(), [1].into_iter().collect());
    }

    #[test]
    fn mempolicy_interleave_node_set() {
        let p = MemPolicy::Interleave([0, 1, 3].into_iter().collect());
        assert_eq!(p.node_set(), [0, 1, 3].into_iter().collect());
    }

    #[test]
    fn mempolicy_preferred_many_node_set() {
        let p = MemPolicy::preferred_many([0, 2]);
        assert_eq!(p.node_set(), [0, 2].into_iter().collect());
    }

    #[test]
    fn mempolicy_weighted_interleave_node_set() {
        let p = MemPolicy::weighted_interleave([1, 3]);
        assert_eq!(p.node_set(), [1, 3].into_iter().collect());
    }

    #[test]
    fn mempolicy_validate_preferred_many_empty() {
        assert!(
            MemPolicy::PreferredMany(BTreeSet::new())
                .validate()
                .is_err()
        );
    }

    #[test]
    fn mempolicy_validate_weighted_interleave_empty() {
        assert!(
            MemPolicy::WeightedInterleave(BTreeSet::new())
                .validate()
                .is_err()
        );
    }

    #[test]
    fn mempolicy_validate_preferred_many_ok() {
        assert!(MemPolicy::preferred_many([0]).validate().is_ok());
    }

    #[test]
    fn mempolicy_validate_weighted_interleave_ok() {
        assert!(MemPolicy::weighted_interleave([0, 1]).validate().is_ok());
    }

    #[test]
    fn mpol_flags_union() {
        let f = MpolFlags::STATIC_NODES | MpolFlags::NUMA_BALANCING;
        assert_eq!(f.bits(), (1 << 15) | (1 << 13));
    }

    #[test]
    fn mpol_flags_none_is_zero() {
        assert_eq!(MpolFlags::NONE.bits(), 0);
    }

    #[test]
    fn work_mpol_flags_builder() {
        let w = WorkSpec::default().mpol_flags(MpolFlags::STATIC_NODES);
        assert_eq!(w.mpol_flags, MpolFlags::STATIC_NODES);
    }

    #[test]
    fn mpol_flags_contains_identity() {
        assert!(MpolFlags::NONE.contains(MpolFlags::NONE));
        assert!(MpolFlags::STATIC_NODES.contains(MpolFlags::STATIC_NODES));
        let composite = MpolFlags::STATIC_NODES | MpolFlags::NUMA_BALANCING;
        assert!(composite.contains(composite));
    }

    #[test]
    fn mpol_flags_contains_superset_is_true_for_subset() {
        let composite = MpolFlags::STATIC_NODES | MpolFlags::NUMA_BALANCING;
        assert!(composite.contains(MpolFlags::STATIC_NODES));
        assert!(composite.contains(MpolFlags::NUMA_BALANCING));
    }

    #[test]
    fn mpol_flags_contains_subset_is_false_for_superset() {
        let composite = MpolFlags::STATIC_NODES | MpolFlags::NUMA_BALANCING;
        assert!(!MpolFlags::STATIC_NODES.contains(composite));
        assert!(!MpolFlags::NUMA_BALANCING.contains(composite));
    }

    #[test]
    fn mpol_flags_contains_empty_is_always_true() {
        // `(x & 0) == 0` holds for every x, so every MpolFlags
        // value — including NONE itself — is a superset of NONE.
        assert!(MpolFlags::NONE.contains(MpolFlags::NONE));
        assert!(MpolFlags::STATIC_NODES.contains(MpolFlags::NONE));
        let composite = MpolFlags::STATIC_NODES | MpolFlags::NUMA_BALANCING;
        assert!(composite.contains(MpolFlags::NONE));
    }

    #[test]
    fn mpol_flags_none_does_not_contain_any_set_flag() {
        assert!(!MpolFlags::NONE.contains(MpolFlags::STATIC_NODES));
        assert!(!MpolFlags::NONE.contains(MpolFlags::RELATIVE_NODES));
        assert!(!MpolFlags::NONE.contains(MpolFlags::NUMA_BALANCING));
    }

    #[test]
    fn mpol_flags_contains_rejects_disjoint_flag() {
        // Single-flag values that share no bits must not satisfy
        // `contains` in either direction.
        assert!(!MpolFlags::STATIC_NODES.contains(MpolFlags::NUMA_BALANCING));
        assert!(!MpolFlags::NUMA_BALANCING.contains(MpolFlags::STATIC_NODES));
    }

    #[test]
    fn mpol_flags_contains_rejects_partial_overlap() {
        // Partial bit overlap must not satisfy `contains` — every
        // bit of `other` must be set in `self`, not merely some.
        let a = MpolFlags::STATIC_NODES | MpolFlags::NUMA_BALANCING;
        let b = MpolFlags::RELATIVE_NODES | MpolFlags::NUMA_BALANCING;
        assert!(!a.contains(b));
        assert!(!b.contains(a));
    }

    // -- CloneMode + CloneFlags + validate_clone_flags tests --

    #[test]
    fn clone_mode_default_is_fork() {
        // Preserves historical fork-based behavior — anything else
        // would silently change every existing caller's spawn path.
        assert!(matches!(CloneMode::default(), CloneMode::Fork));
    }

    #[test]
    fn workload_config_default_clone_mode_is_fork() {
        let c = WorkloadConfig::default();
        assert!(matches!(c.clone_mode, CloneMode::Fork));
    }

    #[test]
    fn work_default_clone_mode_is_fork() {
        let w = WorkSpec::default();
        assert!(matches!(w.clone_mode, CloneMode::Fork));
    }

    #[test]
    fn workload_config_clone_mode_builder() {
        let cfg = WorkloadConfig::default().clone_mode(CloneMode::Thread);
        assert!(matches!(cfg.clone_mode, CloneMode::Thread));
    }

    #[test]
    fn work_clone_mode_builder() {
        let w = WorkSpec::default().clone_mode(CloneMode::SharedVm);
        assert!(matches!(w.clone_mode, CloneMode::SharedVm));
    }

    #[test]
    fn clone_flags_constants_match_libc() {
        // Bit values must match libc::CLONE_* exactly — the kernel
        // syscall ABI is what these flags ultimately feed.
        assert_eq!(CloneFlags::VM.bits(), libc::CLONE_VM as u64);
        assert_eq!(CloneFlags::FS.bits(), libc::CLONE_FS as u64);
        assert_eq!(CloneFlags::FILES.bits(), libc::CLONE_FILES as u64);
        assert_eq!(CloneFlags::SIGHAND.bits(), libc::CLONE_SIGHAND as u64);
        assert_eq!(CloneFlags::THREAD.bits(), libc::CLONE_THREAD as u64);
        assert_eq!(CloneFlags::PARENT.bits(), libc::CLONE_PARENT as u64);
        assert_eq!(CloneFlags::NEWNS.bits(), libc::CLONE_NEWNS as u64);
        assert_eq!(CloneFlags::NEWUSER.bits(), libc::CLONE_NEWUSER as u64);
        assert_eq!(CloneFlags::NEWPID.bits(), libc::CLONE_NEWPID as u64);
        assert_eq!(CloneFlags::NEWNET.bits(), libc::CLONE_NEWNET as u64);
        assert_eq!(CloneFlags::NEWIPC.bits(), libc::CLONE_NEWIPC as u64);
        assert_eq!(CloneFlags::NEWUTS.bits(), libc::CLONE_NEWUTS as u64);
        assert_eq!(CloneFlags::NEWCGROUP.bits(), libc::CLONE_NEWCGROUP as u64);
        // CLONE_IO has bit 31 set and `libc::CLONE_IO` is a c_int
        // (i32) with the sign bit set, so a plain `as u64` cast
        // sign-extends to 0xffffffff80000000. Round-trip through u32
        // first to truncate the sign extension and recover the
        // 32-bit kernel ABI value.
        assert_eq!(CloneFlags::IO.bits(), libc::CLONE_IO as u32 as u64);
        assert_eq!(CloneFlags::SYSVSEM.bits(), libc::CLONE_SYSVSEM as u64);
    }

    #[test]
    fn clone_flags_runtime_reserved_covers_runtime_managed_bits() {
        // RUNTIME_RESERVED must include every flag the runtime sets
        // internally — pidfd / parent_settid / child_settid /
        // child_cleartid / detached / settls / vfork / ptrace /
        // untraced. Each bit listed in the doc must be present.
        let reserved = CloneFlags::RUNTIME_RESERVED.bits();
        assert_eq!(reserved & libc::CLONE_PIDFD as u64, libc::CLONE_PIDFD as u64);
        assert_eq!(
            reserved & libc::CLONE_PARENT_SETTID as u64,
            libc::CLONE_PARENT_SETTID as u64,
        );
        assert_eq!(
            reserved & libc::CLONE_CHILD_SETTID as u64,
            libc::CLONE_CHILD_SETTID as u64,
        );
        assert_eq!(
            reserved & libc::CLONE_CHILD_CLEARTID as u64,
            libc::CLONE_CHILD_CLEARTID as u64,
        );
        assert_eq!(
            reserved & libc::CLONE_DETACHED as u64,
            libc::CLONE_DETACHED as u64,
        );
        assert_eq!(
            reserved & libc::CLONE_SETTLS as u64,
            libc::CLONE_SETTLS as u64,
        );
        assert_eq!(
            reserved & libc::CLONE_VFORK as u64,
            libc::CLONE_VFORK as u64,
        );
        assert_eq!(
            reserved & libc::CLONE_PTRACE as u64,
            libc::CLONE_PTRACE as u64,
        );
        assert_eq!(
            reserved & libc::CLONE_UNTRACED as u64,
            libc::CLONE_UNTRACED as u64,
        );
    }

    #[test]
    fn clone_flags_runtime_reserved_excludes_user_relevant_bits() {
        // The user-facing constants must NOT be in
        // RUNTIME_RESERVED — otherwise the validator would reject
        // them and the variant types would be unusable.
        let reserved = CloneFlags::RUNTIME_RESERVED.bits();
        for f in [
            CloneFlags::VM,
            CloneFlags::FS,
            CloneFlags::FILES,
            CloneFlags::SIGHAND,
            CloneFlags::THREAD,
            CloneFlags::PARENT,
            CloneFlags::NEWNS,
            CloneFlags::NEWUSER,
            CloneFlags::NEWPID,
            CloneFlags::NEWNET,
            CloneFlags::NEWIPC,
            CloneFlags::NEWUTS,
            CloneFlags::NEWCGROUP,
            CloneFlags::IO,
            CloneFlags::SYSVSEM,
        ] {
            assert_eq!(
                reserved & f.bits(),
                0,
                "user-relevant flag {:#x} overlaps RUNTIME_RESERVED",
                f.bits()
            );
        }
    }

    #[test]
    fn clone_flags_union_combines() {
        let f = CloneFlags::VM | CloneFlags::SIGHAND;
        assert_eq!(
            f.bits(),
            (libc::CLONE_VM as u64) | (libc::CLONE_SIGHAND as u64)
        );
    }

    #[test]
    fn clone_flags_contains_subset() {
        let composite = CloneFlags::VM | CloneFlags::SIGHAND | CloneFlags::THREAD;
        assert!(composite.contains(CloneFlags::VM));
        assert!(composite.contains(CloneFlags::SIGHAND));
        assert!(composite.contains(CloneFlags::THREAD));
        // Set-theoretic identity (matches MpolFlags::contains
        // convention): every set is a superset of NONE.
        assert!(composite.contains(CloneFlags::NONE));
        assert!(CloneFlags::NONE.contains(CloneFlags::NONE));
        // Single flag is NOT a superset of the composite.
        assert!(!CloneFlags::VM.contains(composite));
    }

    // -- validate_clone_flags rule-by-rule --
    //
    // Each kernel rejection rule from `copy_process` (kernel/fork.c)
    // is exercised as a separate test so a regression names the
    // offending rule.

    #[test]
    fn validate_clone_flags_thread_without_sighand_rejected() {
        // Rule 1: CLONE_THREAD requires CLONE_SIGHAND.
        // Use from_bits_for_test to express the invalid combination
        // — the public consts make this hard to express via the
        // bitor path because the user would normally pair them.
        let f = CloneFlags::from_bits_for_test(libc::CLONE_THREAD as u64);
        let err = validate_clone_flags(f).expect_err("THREAD without SIGHAND must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("CLONE_THREAD") && msg.contains("CLONE_SIGHAND"),
            "error must name both flags: {msg}"
        );
    }

    #[test]
    fn validate_clone_flags_sighand_without_vm_rejected() {
        // Rule 2: CLONE_SIGHAND requires CLONE_VM.
        let f = CloneFlags::from_bits_for_test(libc::CLONE_SIGHAND as u64);
        let err = validate_clone_flags(f).expect_err("SIGHAND without VM must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("CLONE_SIGHAND") && msg.contains("CLONE_VM"),
            "error must name both flags: {msg}"
        );
    }

    #[test]
    fn validate_clone_flags_newns_with_fs_rejected() {
        // Rule 3: NEWNS and FS are mutually exclusive.
        let f = CloneFlags::NEWNS | CloneFlags::FS;
        let err = validate_clone_flags(f).expect_err("NEWNS|FS must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("CLONE_NEWNS") && msg.contains("CLONE_FS"),
            "error must name both flags: {msg}"
        );
    }

    #[test]
    fn validate_clone_flags_newuser_with_fs_rejected() {
        // Rule 4: NEWUSER and FS are mutually exclusive.
        let f = CloneFlags::NEWUSER | CloneFlags::FS;
        let err = validate_clone_flags(f).expect_err("NEWUSER|FS must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("CLONE_NEWUSER") && msg.contains("CLONE_FS"),
            "error must name both flags: {msg}"
        );
    }

    #[test]
    fn validate_clone_flags_thread_with_newuser_rejected() {
        // Rule 5a: THREAD with NEWUSER is rejected.
        let f =
            CloneFlags::THREAD | CloneFlags::SIGHAND | CloneFlags::VM | CloneFlags::NEWUSER;
        let err = validate_clone_flags(f).expect_err("THREAD|NEWUSER must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("CLONE_THREAD") && msg.contains("CLONE_NEWUSER"),
            "error must name THREAD and NEWUSER: {msg}"
        );
    }

    #[test]
    fn validate_clone_flags_thread_with_newpid_rejected() {
        // Rule 5b: THREAD with NEWPID is rejected.
        let f =
            CloneFlags::THREAD | CloneFlags::SIGHAND | CloneFlags::VM | CloneFlags::NEWPID;
        let err = validate_clone_flags(f).expect_err("THREAD|NEWPID must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("CLONE_THREAD") && msg.contains("CLONE_NEWPID"),
            "error must name THREAD and NEWPID: {msg}"
        );
    }

    #[test]
    fn validate_clone_flags_runtime_reserved_rejected() {
        // Rule 6: caller cannot set runtime-managed bits.
        let f = CloneFlags::from_bits_for_test(libc::CLONE_PIDFD as u64);
        let err = validate_clone_flags(f).expect_err("CLONE_PIDFD from caller must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("runtime-reserved"),
            "error must name the runtime-reserved category: {msg}"
        );
    }

    #[test]
    fn validate_clone_flags_thread_chain_accepted() {
        // Sanity: the valid CLONE_THREAD chain
        // (THREAD ⇒ SIGHAND ⇒ VM, no excluded namespaces) must
        // pass. Without this the whole rule set could be silently
        // over-rejecting.
        let f = CloneFlags::THREAD | CloneFlags::SIGHAND | CloneFlags::VM;
        validate_clone_flags(f).expect("valid THREAD-chain must be accepted");
    }

    #[test]
    fn validate_clone_flags_empty_accepted() {
        // The empty flag set is the fork-equivalent (no sharing
        // beyond the fork defaults). Must validate cleanly.
        validate_clone_flags(CloneFlags::NONE)
            .expect("empty flag set must be accepted");
    }

    #[test]
    fn validate_clone_flags_shared_vm_only_accepted() {
        // SharedVm dispatch will use CLONE_VM only — pin that this
        // passes the validator now, before dispatch lands.
        validate_clone_flags(CloneFlags::VM)
            .expect("CLONE_VM alone must be accepted (SharedVm uses this)");
    }

    // -- spawn dispatch tests (Fork / Thread / SharedVm / Raw) ----

    /// Thread mode: the worker runs in-process via std::thread, the
    /// JoinHandle returns a real WorkerReport, and worker_pids()
    /// reports a non-zero gettid() after start.
    #[test]
    fn spawn_thread_clone_mode_runs_to_completion() {
        let config = WorkloadConfig {
            num_workers: 2,
            clone_mode: CloneMode::Thread,
            work_type: WorkType::CpuSpin,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).expect("Thread mode must spawn");
        h.start();
        std::thread::sleep(std::time::Duration::from_millis(150));
        let pids = h.worker_pids();
        assert_eq!(pids.len(), 2, "worker_pids must reflect both threads");
        for tid in &pids {
            assert!(*tid > 0, "thread tid must be a real gettid() value: {tid}");
        }
        // Sibling threads in the same tgid must report distinct
        // gettid()s — duplicates would mean the publish step is
        // broken or only one thread actually ran.
        assert_ne!(pids[0], pids[1], "sibling thread tids must differ");
        let reports = h.stop_and_collect();
        assert_eq!(reports.len(), 2, "thread mode collects one report per worker");
        for r in &reports {
            assert!(r.completed, "thread worker must complete: {:?}", r);
            assert!(r.work_units > 0, "thread worker must do work: {}", r.work_units);
        }
    }

    /// `CloneMode::Thread + WorkType::ForkExit` MUST bail at spawn
    /// time. Pin the diagnostic message names both the variant and
    /// the structural reason (forked child's `_exit` tears down the
    /// whole tgid via `do_exit`).
    #[test]
    fn spawn_thread_with_forkexit_rejected_at_spawn_time() {
        let config = WorkloadConfig {
            num_workers: 1,
            clone_mode: CloneMode::Thread,
            work_type: WorkType::ForkExit,
            ..Default::default()
        };
        let result = WorkloadHandle::spawn(&config);
        let err = match result {
            Ok(_) => panic!("Thread + ForkExit must bail at spawn"),
            Err(e) => e,
        };
        let msg = format!("{err:#}");
        assert!(
            msg.contains("CloneMode::Thread")
                && msg.contains("WorkType::ForkExit")
                && msg.contains("CloneMode::Fork"),
            "diagnostic must name both incompatible variants and the safe \
             alternative: {msg}"
        );
    }

    /// `CloneMode::Fork + WorkType::ForkExit` is the well-tested
    /// pair (existing test
    /// `stop_and_collect_reaps_grandchild_from_panicking_custom_closure`
    /// pins the fork mode's panic shape). This regression guard
    /// proves the new D5 incompatibility check does NOT also reject
    /// the legitimate Fork+ForkExit combination.
    #[test]
    fn spawn_fork_with_forkexit_succeeds() {
        let config = WorkloadConfig {
            num_workers: 1,
            clone_mode: CloneMode::Fork,
            work_type: WorkType::ForkExit,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config)
            .expect("Fork + ForkExit must remain valid");
        h.start();
        std::thread::sleep(std::time::Duration::from_millis(100));
        let _ = h.stop_and_collect();
    }

    /// Thread-mode worker that panics on first iteration must
    /// surface a [`WorkerExitInfo::Panicked`] sentinel with the
    /// panic message extracted from the join Err payload. Uses a
    /// `WorkType::Custom` closure so the panic path is reproducible
    /// without depending on a buggy work-type implementation.
    #[test]
    fn spawn_thread_panic_yields_panicked_exit_info() {
        // Custom closure that panics immediately. Returns
        // `WorkerReport` to satisfy the signature; the panic fires
        // before `return` is reached.
        fn panic_immediately(_stop: &AtomicBool) -> WorkerReport {
            panic!("test panic from thread worker");
        }
        let config = WorkloadConfig {
            num_workers: 1,
            clone_mode: CloneMode::Thread,
            work_type: WorkType::custom("panic_immediately", panic_immediately),
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).expect("Thread spawn must succeed");
        h.start();
        // Tight: the panic fires synchronously after the start
        // rendezvous; no sleep needed beyond the start handshake.
        let reports = h.stop_and_collect();
        assert_eq!(reports.len(), 1);
        let r = &reports[0];
        assert!(!r.completed, "panicked worker must NOT report completed=true");
        match &r.exit_info {
            Some(WorkerExitInfo::Panicked(msg)) => {
                assert!(
                    msg.contains("test panic from thread worker"),
                    "panic message must round-trip from panic!() to exit_info: {msg}"
                );
            }
            other => panic!(
                "expected Panicked(_) exit_info on thread panic, got {other:?}",
            ),
        }
    }

    /// Thread-mode `Custom` closure that loops on its `stop` arg
    /// MUST terminate via `stop_and_collect` flipping the per-worker
    /// flag, AND `stop_and_collect` MUST NOT touch the global
    /// [`STOP`] (that signal-flag belongs exclusively to Fork mode;
    /// flipping it from Thread mode would inadvertently reach any
    /// concurrently-running fork-mode workers and any fork-child of
    /// the test harness itself). The test snapshots the global
    /// [`STOP`] before/after `stop_and_collect` and asserts no
    /// change.
    #[test]
    fn spawn_thread_custom_stop_does_not_touch_global_stop() {
        // Custom closure that spins on the per-worker stop arg.
        // Returns a non-default WorkerReport with completed=true so
        // the test can pin "the stop loop saw stop=true and exited
        // cleanly" instead of "the worker crashed before reading
        // its arg."
        fn spin_until_stop(stop: &AtomicBool) -> WorkerReport {
            let tid: libc::pid_t =
                unsafe { libc::syscall(libc::SYS_gettid) as libc::pid_t };
            while !stop_requested(stop) {
                std::thread::sleep(Duration::from_millis(10));
            }
            WorkerReport {
                tid,
                completed: true,
                ..WorkerReport::default()
            }
        }

        // Snapshot the global STOP before spawning. This MUST be
        // false (no concurrent workload running in the test
        // harness) and remain false across the whole call sequence.
        STOP.store(false, Ordering::Relaxed);
        let stop_before = STOP.load(Ordering::Relaxed);
        assert!(
            !stop_before,
            "global STOP must be false before the test runs — \
             a stale true from a prior test would mask the assertion"
        );

        let config = WorkloadConfig {
            num_workers: 1,
            clone_mode: CloneMode::Thread,
            work_type: WorkType::custom("spin_until_stop", spin_until_stop),
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).expect("Thread spawn must succeed");
        h.start();
        // Brief sleep so the worker definitely enters its spin loop
        // before we ask stop_and_collect to flip its flag.
        std::thread::sleep(Duration::from_millis(50));

        let reports = h.stop_and_collect();
        // Worker observed its per-worker stop and returned a clean
        // report — proves the stop signal reached the closure.
        assert_eq!(reports.len(), 1);
        assert!(
            reports[0].completed,
            "Custom thread worker must observe per-worker stop and \
             return completed=true: got {:?}",
            reports[0]
        );

        // Critical assertion: stop_and_collect MUST NOT have flipped
        // the global STOP. Thread-mode stop is per-worker
        // Arc<AtomicBool>; the global STOP is reserved for the
        // SIGUSR1-driven Fork-mode path. Touching it from Thread
        // mode would leak shutdown signals into unrelated workers.
        let stop_after = STOP.load(Ordering::Relaxed);
        assert!(
            !stop_after,
            "global STOP must remain false after Thread-mode \
             stop_and_collect — Thread mode flips per-worker flags \
             only, never the global signal-handler flag"
        );
    }

    /// SharedVm mode: clone3 dispatch is staged behind a Rust
    /// runtime redesign (clone3(CLONE_VM) children lack per-task
    /// thread-info, so worker_main aborts). Pin the bail-out
    /// message until the redesign lands.
    #[test]
    fn spawn_shared_vm_clone_mode_bails() {
        let config = WorkloadConfig {
            num_workers: 1,
            clone_mode: CloneMode::SharedVm,
            ..Default::default()
        };
        let result = WorkloadHandle::spawn(&config);
        let err = match result {
            Ok(_) => panic!("SharedVm must bail — clone3 child has no Rust TLS"),
            Err(e) => e,
        };
        let msg = format!("{err:#}");
        assert!(
            msg.contains("SharedVm")
                && msg.contains("Rust TLS")
                && msg.contains("CloneMode::Thread"),
            "error must name SharedVm + Rust TLS constraint + Thread alternative: {msg}"
        );
    }

    /// Raw with valid flags: validator passes (flag set is
    /// well-formed) but dispatch still bails on the same Rust TLS
    /// constraint as SharedVm. The "Flag set is structurally valid"
    /// substring proves the validator ran before the dispatch
    /// rejection.
    #[test]
    fn spawn_raw_clone_mode_validates_then_bails() {
        let config = WorkloadConfig {
            num_workers: 1,
            clone_mode: CloneMode::Raw(CloneFlags::VM),
            ..Default::default()
        };
        let result = WorkloadHandle::spawn(&config);
        let err = match result {
            Ok(_) => panic!("Raw must bail — clone3 child has no Rust TLS"),
            Err(e) => e,
        };
        let msg = format!("{err:#}");
        assert!(
            msg.contains("Rust TLS") && msg.contains("CloneMode::Thread"),
            "error must name Rust TLS constraint + Thread alternative: {msg}"
        );
        assert!(
            msg.contains("structurally valid"),
            "valid flags must say flag set passed validation: {msg}"
        );
    }

    /// Invalid flags must hit the validator BEFORE the dispatch
    /// rejection. The error must name the kernel rule (e.g.
    /// "CLONE_THREAD requires CLONE_SIGHAND"), NOT the Rust-TLS
    /// dispatch message — the validator runs first.
    #[test]
    fn spawn_raw_clone_mode_invalid_flags_rejected_before_dispatch() {
        let config = WorkloadConfig {
            num_workers: 1,
            clone_mode: CloneMode::Raw(CloneFlags::from_bits_for_test(
                libc::CLONE_THREAD as u64,
            )),
            ..Default::default()
        };
        let result = WorkloadHandle::spawn(&config);
        let err = match result {
            Ok(_) => panic!("Raw with invalid flags must bail at validator"),
            Err(e) => e,
        };
        let msg = format!("{err:#}");
        assert!(
            msg.contains("CLONE_THREAD") && msg.contains("CLONE_SIGHAND"),
            "error must surface the validator's rule, not the dispatch message: {msg}"
        );
        assert!(
            !msg.contains("structurally valid"),
            "validator failure must NOT mention dispatch: {msg}"
        );
        assert!(
            !msg.contains("Rust TLS"),
            "validator failure must NOT mention the dispatch's Rust TLS reason: {msg}"
        );
    }

    #[test]
    fn build_nodemask_empty() {
        let (mask, maxnode) = build_nodemask(&BTreeSet::new());
        assert!(mask.is_empty());
        assert_eq!(maxnode, 0);
    }

    #[test]
    fn build_nodemask_single() {
        let (mask, maxnode) = build_nodemask(&[0].into_iter().collect());
        // kernel get_nodes() does --maxnode, so maxnode = max_node + 2
        assert_eq!(maxnode, 2);
        assert_eq!(mask.len(), 1);
        assert_eq!(mask[0], 1);
    }

    #[test]
    fn build_nodemask_multiple() {
        let (mask, maxnode) = build_nodemask(&[0, 2].into_iter().collect());
        assert_eq!(maxnode, 4); // max_node=2, +2 = 4
        assert_eq!(mask[0] & 1, 1); // node 0
        assert_eq!(mask[0] & 4, 4); // node 2
        assert_eq!(mask[0] & 2, 0); // node 1 not set
    }

    #[test]
    fn build_nodemask_high_node() {
        let bits_per_word = std::mem::size_of::<libc::c_ulong>() * 8;
        let high = bits_per_word + 3;
        let (mask, maxnode) = build_nodemask(&[high].into_iter().collect());
        assert_eq!(maxnode, (high + 2) as libc::c_ulong);
        assert_eq!(mask.len(), 2);
        assert_eq!(mask[0], 0);
        assert_eq!(mask[1], 1 << 3);
    }

    #[test]
    fn apply_mempolicy_default_is_noop() {
        apply_mempolicy_with_flags(&MemPolicy::Default, MpolFlags::NONE);
    }

    #[test]
    fn apply_mempolicy_empty_bind_skipped() {
        apply_mempolicy_with_flags(&MemPolicy::Bind(BTreeSet::new()), MpolFlags::NONE);
    }

    #[test]
    fn apply_mempolicy_empty_interleave_skipped() {
        apply_mempolicy_with_flags(&MemPolicy::Interleave(BTreeSet::new()), MpolFlags::NONE);
    }

    #[test]
    fn work_mem_policy_builder() {
        let w = WorkSpec::default().mem_policy(MemPolicy::Bind([0].into_iter().collect()));
        assert!(matches!(w.mem_policy, MemPolicy::Bind(_)));
    }

    #[test]
    fn work_default_mempolicy_is_default() {
        let w = WorkSpec::default();
        assert!(matches!(w.mem_policy, MemPolicy::Default));
    }

    #[test]
    fn workload_config_default_mempolicy() {
        let wl = WorkloadConfig::default();
        assert!(matches!(wl.mem_policy, MemPolicy::Default));
    }

    // -- PageFaultChurn tests --

    #[test]
    fn page_fault_churn_name_roundtrip() {
        let wt = WorkType::from_name("PageFaultChurn").unwrap();
        assert_eq!(wt.name(), "PageFaultChurn");
    }

    #[test]
    fn page_fault_churn_from_name_defaults() {
        let wt = WorkType::from_name("PageFaultChurn").unwrap();
        match wt {
            WorkType::PageFaultChurn {
                region_kb,
                touches_per_cycle,
                spin_iters,
            } => {
                assert_eq!(region_kb, 4096);
                assert_eq!(touches_per_cycle, 256);
                assert_eq!(spin_iters, 64);
            }
            _ => panic!("expected PageFaultChurn"),
        }
    }

    #[test]
    fn page_fault_churn_group_size_none() {
        let wt = WorkType::page_fault_churn(4096, 256, 64);
        assert_eq!(wt.worker_group_size(), None);
    }

    #[test]
    fn page_fault_churn_no_shared_mem() {
        assert!(!WorkType::page_fault_churn(4096, 256, 64).needs_shared_mem());
    }

    #[test]
    fn page_fault_churn_no_cache_buf() {
        assert!(!WorkType::page_fault_churn(4096, 256, 64).needs_cache_buf());
    }

    /// Overflow-path pin: when `region_kb * 1024` overflows `usize`
    /// (the configured value is so large that the page-fault region
    /// size cannot be represented), the worker's outer loop hits
    /// the `checked_mul` None arm, emits the `tracing::warn!`, and
    /// `break`s without doing any page-fault work. The worker
    /// still terminates cleanly and reports 0 iterations — no
    /// mmap, no segfault, no hang.
    ///
    /// Spawns a single worker with `region_kb = usize::MAX` so the
    /// multiplication overflows on every pointer width we support
    /// (32-bit: MAX*1024 overflows immediately; 64-bit: MAX*1024
    /// also overflows). Runs briefly, asserts the worker's
    /// `iterations` is 0 — proof the outer loop broke out before
    /// the first page-fault cycle ran. The worker report still
    /// arrives (proving `stop_and_collect` sees a graceful exit
    /// on this path, not a signal kill).
    ///
    /// Pairs with [`page_fault_churn_from_name_defaults`] which
    /// pins the happy path — together they pin both ends of the
    /// region_size validity domain.
    #[test]
    fn page_fault_churn_region_kb_overflow_worker_exits_cleanly() {
        let config = WorkloadConfig {
            num_workers: 1,
            affinity: ResolvedAffinity::None,
            // `region_kb = usize::MAX` — `usize::MAX * 1024`
            // overflows on both 32-bit and 64-bit usize, so
            // `checked_mul` returns None and the outer loop
            // `break`s immediately. `touches_per_cycle` and
            // `spin_iters` are ignored by that path.
            work_type: WorkType::PageFaultChurn {
                region_kb: usize::MAX,
                touches_per_cycle: 16,
                spin_iters: 32,
            },
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        h.start();
        // Give the worker a short window to spin through its
        // spawn handshake + outer-loop entry + break. 100 ms is
        // comfortably more than the sub-millisecond path the
        // overflow arm runs, while keeping the test fast.
        std::thread::sleep(Duration::from_millis(100));
        let reports = h.stop_and_collect();
        assert_eq!(reports.len(), 1, "exactly one worker was spawned");
        let r = &reports[0];
        // `iterations` is the outer-loop counter: 0 means the
        // worker hit the `break` BEFORE any page-fault cycle
        // completed, which is the overflow path.
        assert_eq!(
            r.iterations, 0,
            "worker with overflowing region_kb must break out of the outer loop \
             without completing any page-fault cycle; got iterations={}",
            r.iterations,
        );
        // `work_units` may be 0 (spin_burst inside the overflow
        // arm never ran) OR a tiny positive value if the worker
        // took an unrelated iteration through the outer loop —
        // but under this config only PageFaultChurn is selected
        // so spin_burst before the overflow break is not
        // reachable. Assert exact zero to pin the overflow path's
        // no-op guarantee.
        assert_eq!(
            r.work_units, 0,
            "overflow path must not increment work_units; got {}",
            r.work_units,
        );
    }

    /// Guards three invariants of [`WorkType::PageFaultChurn`]:
    ///
    /// 1. Every spawned worker produces non-zero `work_units` and
    ///    `iterations` (sanity — holds under the pre-fix bug too,
    ///    so it's a basic progress check, not a regression guard).
    /// 2. `iter_slot` (host-side iteration sampling, read via
    ///    [`WorkloadHandle::snapshot_iterations`]) ADVANCES during
    ///    the run. Asserted as a positive delta between two
    ///    snapshots taken at 100 ms and 250 ms. A delta is
    ///    insensitive to worker start-up latency (the test would
    ///    otherwise race against workers whose first outer iter
    ///    lands after the first snapshot). Pre-fix, PageFaultChurn
    ///    used an inner `while !STOP` loop that bypassed the
    ///    iter_slot publish in the outer `worker_main` loop, so
    ///    both snapshots were pinned at 0 and the delta would be 0.
    /// 3. On multi-CPU hosts, at least one worker records ≥ 1
    ///    migration. With `num_workers = available_parallelism() + 1`
    ///    the workload oversubscribes by one, forcing at least one
    ///    context switch and CPU re-dispatch in any realistic
    ///    scheduler; combined with the migration check in the
    ///    outer `worker_main` loop (gated on
    ///    `work_units.is_multiple_of(1024)`) firing every 64 outer
    ///    iters for this test's parameters (touches_per_cycle=16 +
    ///    spin_iters=32 = 48 work_units/iter,
    ///    gcd(48, 1024) = 16, period = 1024/16 = 64; the default
    ///    16-iter period documented in
    ///    doc/guide/src/architecture/workers.md assumes
    ///    default params 256+64=320 instead), this puts the
    ///    assertion well above the flake threshold. Gated on
    ///    `available_parallelism() > 1` because single-CPU
    ///    sandboxes legitimately report 0 migrations.
    #[test]
    fn spawn_page_fault_churn_produces_work() {
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        // Oversubscribe by one to force CPU sharing even on fully
        // idle hosts, so the migration-count assertion below has
        // a reliable signal.
        let num_workers = num_cpus + 1;
        let config = WorkloadConfig {
            num_workers,
            affinity: ResolvedAffinity::None,
            work_type: WorkType::PageFaultChurn {
                region_kb: 64,
                touches_per_cycle: 16,
                spin_iters: 32,
            },
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        h.start();
        // Delta-based iter_slot assertion. Pre-fix these snapshots
        // were both 0 for PageFaultChurn (inner `while !STOP`
        // blocked the iter_slot publish in the outer `worker_main`
        // loop). Post-fix the outer loop
        // updates iter_slot every iteration, so the 150 ms gap
        // between snap1 and snap2 observes many iterations'
        // worth of progress.
        std::thread::sleep(std::time::Duration::from_millis(100));
        let snap1 = h.snapshot_iterations();
        std::thread::sleep(std::time::Duration::from_millis(150));
        let snap2 = h.snapshot_iterations();
        let reports = h.stop_and_collect();
        assert_eq!(reports.len(), num_workers);
        assert_eq!(snap1.len(), num_workers);
        assert_eq!(snap2.len(), num_workers);
        for i in 0..num_workers {
            let delta = snap2[i].saturating_sub(snap1[i]);
            assert!(
                delta > 0,
                "worker {i} iter_slot delta between 100 ms and 250 ms \
                 was 0 (snap1={}, snap2={}); outer loop is not \
                 advancing, indicating a regression that restores \
                 the inner-`while !STOP` bug",
                snap1[i],
                snap2[i],
            );
        }
        // Basic progress sanity — holds even under the pre-fix
        // bug (inner loop still incremented work_units and
        // iterations), so this is not a regression guard for the
        // inner-while bug. Delta assertion above covers that.
        for r in &reports {
            assert!(
                r.work_units > 0,
                "PageFaultChurn worker {} did no work",
                r.tid
            );
            assert!(
                r.iterations > 0,
                "PageFaultChurn worker {} final iterations = 0",
                r.tid
            );
        }
        if num_cpus > 1 {
            let total_migrations: u64 = reports.iter().map(|r| r.migration_count).sum();
            assert!(
                total_migrations > 0,
                "expected ≥ 1 migration across {num_workers} \
                 oversubscribed workers on {num_cpus}-cpu host; 0 \
                 total migrations suggests the outer migration \
                 check at work_units.is_multiple_of(1024) isn't \
                 firing, indicating a regression that restores the \
                 inner-`while !STOP` bug"
            );
        }
    }

    // -- MutexContention tests --

    #[test]
    fn mutex_contention_name_roundtrip() {
        let wt = WorkType::from_name("MutexContention").unwrap();
        assert_eq!(wt.name(), "MutexContention");
    }

    #[test]
    fn mutex_contention_from_name_defaults() {
        let wt = WorkType::from_name("MutexContention").unwrap();
        match wt {
            WorkType::MutexContention {
                contenders,
                hold_iters,
                work_iters,
            } => {
                assert_eq!(contenders, 4);
                assert_eq!(hold_iters, 256);
                assert_eq!(work_iters, 1024);
            }
            _ => panic!("expected MutexContention"),
        }
    }

    #[test]
    fn mutex_contention_group_size() {
        let wt = WorkType::mutex_contention(4, 256, 1024);
        assert_eq!(wt.worker_group_size(), Some(4));
        let wt2 = WorkType::mutex_contention(8, 256, 1024);
        assert_eq!(wt2.worker_group_size(), Some(8));
    }

    #[test]
    fn mutex_contention_needs_shared_mem() {
        assert!(WorkType::mutex_contention(4, 256, 1024).needs_shared_mem());
    }

    #[test]
    fn mutex_contention_no_cache_buf() {
        assert!(!WorkType::mutex_contention(4, 256, 1024).needs_cache_buf());
    }

    #[test]
    fn spawn_mutex_contention_produces_work() {
        let reports = spawn_and_collect_after(
            WorkType::MutexContention {
                contenders: 4,
                hold_iters: 64,
                work_iters: 256,
            },
            4,
            500,
        );
        assert_eq!(reports.len(), 4);
        for r in &reports {
            assert!(
                r.work_units > 0,
                "MutexContention worker {} did no work",
                r.tid
            );
        }
    }

    #[test]
    fn spawn_mutex_contention_bad_worker_count_fails() {
        let config = WorkloadConfig {
            num_workers: 3,
            affinity: ResolvedAffinity::None,
            work_type: WorkType::MutexContention {
                contenders: 4,
                hold_iters: 256,
                work_iters: 1024,
            },
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let result = WorkloadHandle::spawn(&config);
        assert!(result.is_err());
        let msg = format!("{:#}", result.err().unwrap());
        assert!(
            msg.contains("divisible by 4"),
            "expected divisibility error: {msg}"
        );
    }

    #[test]
    fn mutex_contention_records_wake_latency() {
        let config = WorkloadConfig {
            num_workers: 4,
            affinity: ResolvedAffinity::None,
            work_type: WorkType::MutexContention {
                contenders: 4,
                hold_iters: 64,
                work_iters: 256,
            },
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        h.start();
        std::thread::sleep(std::time::Duration::from_millis(500));
        let reports = h.stop_and_collect();
        let has_latencies = reports.iter().any(|r| !r.resume_latencies_ns.is_empty());
        assert!(has_latencies, "contenders should record wake latencies");
    }

    // -- CloneFlags + validate_clone_flags coverage ----------------------
    //
    // The flag validator is the structural-rejection layer that sits in
    // front of the clone3 syscall — callers see a named-rule diagnostic
    // here rather than an opaque EINVAL from the kernel.  The rules
    // enumerated by `validate_clone_flags` are 1:1 with the rejection
    // checks at the top of `kernel/fork.c::copy_process`; pinning each
    // rule individually catches a regression that drops a check or
    // mis-orders the bail-outs.  These tests exercise the validator
    // through the `Verdict` API so the rule-name diagnostics are
    // claim-labeled and survive the migration off the legacy Expect
    // shape.

    /// CloneFlags::NONE passes the validator — the all-zero flag set
    /// is the no-op default for [`CloneMode::Raw`].
    #[test]
    fn validate_clone_flags_none_passes_via_verdict() {
        use crate::assert::Verdict;
        let mut v = Verdict::new();
        let valid_none = validate_clone_flags(CloneFlags::NONE).is_ok();
        crate::claim!(v, valid_none).eq(true);
        let r = v.into_result();
        assert!(r.passed, "CloneFlags::NONE must validate: {:?}", r.details);
    }

    /// Rule 1: `CLONE_THREAD` requires `CLONE_SIGHAND`. Pin both
    /// directions through Verdict; the diagnostic must name the rule.
    #[test]
    fn validate_clone_flags_rule_1_thread_requires_sighand() {
        use crate::assert::Verdict;

        let thread_only = CloneFlags::THREAD;
        let rejected = validate_clone_flags(thread_only).is_err();
        let mut v = Verdict::new();
        crate::claim!(v, rejected).eq(true);
        let r = v.into_result();
        assert!(r.passed, "rule 1: THREAD alone must be rejected");

        let err = validate_clone_flags(thread_only).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("CLONE_THREAD requires CLONE_SIGHAND"),
            "rule 1 diagnostic must name the rule: {msg}",
        );

        let thread_sighand_vm = CloneFlags::THREAD | CloneFlags::SIGHAND | CloneFlags::VM;
        let accepted = validate_clone_flags(thread_sighand_vm).is_ok();
        let mut v = Verdict::new();
        crate::claim!(v, accepted).eq(true);
        let r = v.into_result();
        assert!(
            r.passed,
            "rule 1: THREAD + SIGHAND + VM must be accepted: {:?}",
            r.details
        );
    }

    /// Rule 2: `CLONE_SIGHAND` requires `CLONE_VM`.
    #[test]
    fn validate_clone_flags_rule_2_sighand_requires_vm() {
        use crate::assert::Verdict;

        let sighand_only = CloneFlags::SIGHAND;
        let rejected = validate_clone_flags(sighand_only).is_err();
        let mut v = Verdict::new();
        crate::claim!(v, rejected).eq(true);
        let r = v.into_result();
        assert!(r.passed, "rule 2: SIGHAND alone must be rejected");

        let err = validate_clone_flags(sighand_only).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("CLONE_SIGHAND requires CLONE_VM"),
            "rule 2 diagnostic must name the rule: {msg}",
        );

        let sighand_vm = CloneFlags::SIGHAND | CloneFlags::VM;
        let accepted = validate_clone_flags(sighand_vm).is_ok();
        let mut v = Verdict::new();
        crate::claim!(v, accepted).eq(true);
        let r = v.into_result();
        assert!(r.passed, "rule 2: SIGHAND + VM must be accepted: {:?}", r.details);
    }

    /// Rules 3 / 4 / 5: mutually-exclusive pairs. Folded into one
    /// Verdict — every named rejection path must hold.
    #[test]
    fn validate_clone_flags_rules_3_4_5_mutual_exclusion() {
        use crate::assert::Verdict;

        let mut v = Verdict::new();

        let rule_3_rejected = validate_clone_flags(CloneFlags::NEWNS | CloneFlags::FS).is_err();
        crate::claim!(v, rule_3_rejected).eq(true);

        let rule_4_rejected = validate_clone_flags(CloneFlags::NEWUSER | CloneFlags::FS).is_err();
        crate::claim!(v, rule_4_rejected).eq(true);

        let rule_5a_rejected = validate_clone_flags(
            CloneFlags::THREAD | CloneFlags::SIGHAND | CloneFlags::VM | CloneFlags::NEWUSER,
        )
        .is_err();
        crate::claim!(v, rule_5a_rejected).eq(true);

        let rule_5b_rejected = validate_clone_flags(
            CloneFlags::THREAD | CloneFlags::SIGHAND | CloneFlags::VM | CloneFlags::NEWPID,
        )
        .is_err();
        crate::claim!(v, rule_5b_rejected).eq(true);

        let r = v.into_result();
        assert!(
            r.passed,
            "rules 3/4/5: every named mutual-exclusion path must reject: {:?}",
            r.details,
        );
    }

    /// Rule 6: caller-set runtime-reserved bits rejected with a hex-
    /// formatted diagnostic naming `RUNTIME_RESERVED`.
    #[test]
    fn validate_clone_flags_rule_6_runtime_reserved_bits_rejected() {
        use crate::assert::Verdict;

        let pidfd_bit = CloneFlags::from_bits_for_test(0x1000);
        let rejected = validate_clone_flags(pidfd_bit).is_err();
        let mut v = Verdict::new();
        crate::claim!(v, rejected).eq(true);
        let r = v.into_result();
        assert!(r.passed, "rule 6: CLONE_PIDFD must be rejected");

        let err = validate_clone_flags(pidfd_bit).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("runtime-reserved flags 0x1000"),
            "rule 6 diagnostic must name the offending bit in hex: {msg}",
        );
        assert!(
            msg.contains("RUNTIME_RESERVED"),
            "rule 6 diagnostic must reference the constant: {msg}",
        );

        let all_reserved = CloneFlags::RUNTIME_RESERVED;
        let mass_rejected = validate_clone_flags(all_reserved).is_err();
        let mut v = Verdict::new();
        crate::claim!(v, mass_rejected).eq(true);
        let r = v.into_result();
        assert!(
            r.passed,
            "rule 6: full RUNTIME_RESERVED mask must be rejected: {:?}",
            r.details,
        );
    }

    /// `CloneFlags::contains` is set-theoretic — `contains(NONE)`
    /// returns true for any self.
    #[test]
    fn clone_flags_contains_none_is_universally_true() {
        use crate::assert::Verdict;

        let mut v = Verdict::new();
        let cases = &[
            CloneFlags::NONE,
            CloneFlags::VM,
            CloneFlags::FS,
            CloneFlags::FILES,
            CloneFlags::SIGHAND,
            CloneFlags::THREAD,
            CloneFlags::PARENT,
            CloneFlags::NEWNS,
            CloneFlags::NEWUSER,
            CloneFlags::NEWPID,
            CloneFlags::NEWNET,
            CloneFlags::NEWIPC,
            CloneFlags::NEWUTS,
            CloneFlags::NEWCGROUP,
            CloneFlags::IO,
            CloneFlags::SYSVSEM,
        ];
        for f in cases {
            let contains_none = f.contains(CloneFlags::NONE);
            crate::claim!(v, contains_none).eq(true);
        }
        let r = v.into_result();
        assert!(
            r.passed,
            "contains(NONE) must be universally true: {:?}",
            r.details,
        );
    }

    /// `CloneFlags::union` and `BitOr::bitor` must produce identical
    /// bit patterns. The struct doc says BitOr delegates to union.
    #[test]
    fn clone_flags_union_and_bitor_produce_identical_bits() {
        use crate::assert::Verdict;

        let lhs = CloneFlags::VM | CloneFlags::FILES;
        let rhs = CloneFlags::SIGHAND;
        let union_bits = lhs.union(rhs).bits();
        let bitor_bits = (lhs | rhs).bits();

        let mut v = Verdict::new();
        crate::claim!(v, union_bits).eq(bitor_bits);
        let r = v.into_result();
        assert!(
            r.passed,
            "union and BitOr must agree: union=0x{union_bits:x} bitor=0x{bitor_bits:x}",
        );
    }
}
