//! Worker process management and telemetry.
//!
//! Workers are `fork()`ed processes (not threads) so each can be placed
//! in its own cgroup. Key types:
//! - [`WorkType`] -- what each worker does
//! - [`WorkloadConfig`] -- spawn configuration (count, affinity, work type, policy)
//! - [`WorkloadHandle`] -- RAII handle to spawned workers
//! - [`WorkerReport`] -- per-worker telemetry collected after stop
//! - [`AffinityKind`] -- per-worker affinity intent (Inherit, LlcAligned, Exact, etc.)
//! - [`AffinityMode`] -- resolved CPU affinity for workers
//! - [`Work`] -- workload definition for a single group of workers within a cgroup
//! - [`Phase`] -- a single phase in a [`WorkType::Sequence`] compound work pattern
//! - [`SchedPolicy`] -- Linux scheduling policy for a worker process
//! - [`MemPolicy`] -- NUMA memory placement policy for worker processes
//!
//! See the [Work Types](https://likewhatevs.github.io/ktstr/guide/concepts/work-types.html)
//! and [Worker Processes](https://likewhatevs.github.io/ktstr/guide/architecture/workers.html)
//! chapters of the guide.

use anyhow::{Context, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Seek, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Scenario-level affinity intent for a group of workers.
///
/// Resolved to a concrete [`AffinityMode`] at runtime based on the
/// cgroup's effective cpuset and the VM's topology. When attached to
/// a [`Work`], determines per-worker `sched_setaffinity` masks.
///
/// Resolution uses [`resolve_affinity_for_cgroup()`](crate::scenario::resolve_affinity_for_cgroup).
#[derive(Clone, Debug, Default)]
pub enum AffinityKind {
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

impl AffinityKind {
    /// Construct an `Exact` affinity from any iterator of CPU indices.
    ///
    /// Accepts arrays, ranges, `Vec`, `BTreeSet`, or any `IntoIterator<Item = usize>`.
    pub fn exact(cpus: impl IntoIterator<Item = usize>) -> Self {
        AffinityKind::Exact(cpus.into_iter().collect())
    }
}

/// Resolved CPU affinity for a worker process.
///
/// Created from [`AffinityKind`] at runtime based on topology and
/// cpuset assignments.
#[derive(Debug, Clone)]
pub enum AffinityMode {
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
    /// Receivers measure wake-to-run latency. For cache-aware fan-out with
    /// matrix multiply work, see [`SchBench`](Self::SchBench). Requires
    /// num_workers divisible by (fan_out + 1).
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
    /// Schbench-style messenger/worker workload. One messenger per group
    /// wakes `fan_out` workers via shared futex. Each worker does
    /// `operations` matrix multiplications over a `cache_footprint_kb`-sized
    /// working set, preceded by a `sleep_usec` microsecond sleep simulating
    /// think time. Workers measure wake-to-run latency (time from messenger's
    /// timestamp to worker getting the CPU). Requires num_workers divisible
    /// by (fan_out + 1).
    SchBench {
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
    Custom {
        name: &'static str,
        run: fn(&AtomicBool) -> WorkerReport,
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
            WorkType::SchBench { .. } => "SchBench",
            WorkType::PageFaultChurn { .. } => "PageFaultChurn",
            WorkType::MutexContention { .. } => "MutexContention",
            WorkType::Custom { name, .. } => name,
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
                burst_ms: 50,
                sleep_ms: 100,
            }),
            "PipeIo" => Some(WorkType::PipeIo { burst_iters: 1024 }),
            "FutexPingPong" => Some(WorkType::FutexPingPong { spin_iters: 1024 }),
            "CachePressure" => Some(WorkType::CachePressure {
                size_kb: 32,
                stride: 64,
            }),
            "CacheYield" => Some(WorkType::CacheYield {
                size_kb: 32,
                stride: 64,
            }),
            "CachePipe" => Some(WorkType::CachePipe {
                size_kb: 32,
                burst_iters: 1024,
            }),
            "FutexFanOut" => Some(WorkType::FutexFanOut {
                fan_out: 4,
                spin_iters: 1024,
            }),
            "ForkExit" => Some(WorkType::ForkExit),
            "NiceSweep" => Some(WorkType::NiceSweep),
            "AffinityChurn" => Some(WorkType::AffinityChurn { spin_iters: 1024 }),
            "PolicyChurn" => Some(WorkType::PolicyChurn { spin_iters: 1024 }),
            "SchBench" => Some(WorkType::SchBench {
                fan_out: 4,
                cache_footprint_kb: 256,
                operations: 5,
                sleep_usec: 100,
            }),
            "PageFaultChurn" => Some(WorkType::PageFaultChurn {
                region_kb: 4096,
                touches_per_cycle: 256,
                spin_iters: 64,
            }),
            "MutexContention" => Some(WorkType::MutexContention {
                contenders: 4,
                hold_iters: 256,
                work_iters: 1024,
            }),
            // Sequence requires explicit phases; no default from_name.
            _ => None,
        }
    }

    /// Worker group size for this work type, or None if ungrouped.
    ///
    /// `num_workers` must be divisible by this value. Paired types return 2,
    /// fan-out returns fan_out + 1 (1 messenger + N receivers).
    pub fn worker_group_size(&self) -> Option<usize> {
        match self {
            WorkType::PipeIo { .. }
            | WorkType::FutexPingPong { .. }
            | WorkType::CachePipe { .. } => Some(2),
            WorkType::FutexFanOut { fan_out, .. } => Some(fan_out + 1),
            WorkType::SchBench { fan_out, .. } => Some(fan_out + 1),
            WorkType::MutexContention { contenders, .. } => Some(*contenders),
            _ => None,
        }
    }

    /// Whether this work type needs a pre-fork shared memory region (MAP_SHARED mmap).
    pub fn needs_shared_mem(&self) -> bool {
        matches!(
            self,
            WorkType::FutexPingPong { .. }
                | WorkType::FutexFanOut { .. }
                | WorkType::SchBench { .. }
                | WorkType::MutexContention { .. }
        )
    }

    /// Whether this work type allocates a per-worker cache buffer post-fork.
    pub fn needs_cache_buf(&self) -> bool {
        matches!(
            self,
            WorkType::CachePressure { .. }
                | WorkType::CacheYield { .. }
                | WorkType::CachePipe { .. }
                | WorkType::SchBench { .. }
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

    /// Schbench-style messenger/worker workload with the given parameters.
    pub fn schbench(
        fan_out: usize,
        cache_footprint_kb: usize,
        operations: usize,
        sleep_usec: u64,
    ) -> Self {
        WorkType::SchBench {
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

    /// User-supplied work function with a display name.
    ///
    /// `run` receives a reference to the stop flag (set by SIGUSR1) and
    /// must return a [`WorkerReport`] when the flag becomes `true`. The
    /// framework handles fork, cgroup placement, affinity, scheduling
    /// policy, and signal setup; `run` owns only the work loop.
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
/// Returns a clone of `override_wt` when `swappable` is true and the
/// override's group size (if any) divides `num_workers`. Otherwise
/// returns a clone of `base`.
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
/// `Fifo` and `RoundRobin` require `CAP_SYS_NICE`. Priority values
/// are clamped to 1-99.
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

    /// Combine two flag sets.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Raw flag bits for passing to the syscall.
    pub const fn bits(self) -> u32 {
        self.0
    }
}

impl std::ops::BitOr for MpolFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
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

/// Configuration for spawning a group of worker processes.
#[derive(Debug, Clone)]
pub struct WorkloadConfig {
    /// Number of worker processes to fork.
    pub num_workers: usize,
    /// CPU affinity mode for workers.
    pub affinity: AffinityMode,
    /// What each worker does.
    pub work_type: WorkType,
    /// Linux scheduling policy.
    pub sched_policy: SchedPolicy,
    /// NUMA memory placement policy.
    pub mem_policy: MemPolicy,
    /// Optional mode flags for `set_mempolicy(2)`.
    pub mpol_flags: MpolFlags,
}

impl Default for WorkloadConfig {
    fn default() -> Self {
        Self {
            num_workers: 1,
            affinity: AffinityMode::None,
            work_type: WorkType::CpuSpin,
            sched_policy: SchedPolicy::Normal,
            mem_policy: MemPolicy::Default,
            mpol_flags: MpolFlags::NONE,
        }
    }
}

/// Workload definition for a single group of workers within a cgroup.
///
/// Extracted from [`CgroupDef`](crate::scenario::ops::CgroupDef) to allow
/// multiple concurrent work groups per cgroup. Each `Work` spawns its own
/// set of worker processes.
///
/// ```
/// # use ktstr::workload::{Work, WorkType, SchedPolicy, MemPolicy};
/// let w = Work::default()
///     .workers(4)
///     .work_type(WorkType::bursty(50, 100))
///     .sched_policy(SchedPolicy::Batch)
///     .mem_policy(MemPolicy::bind([0, 1]));
/// assert_eq!(w.num_workers, Some(4));
/// ```
#[derive(Clone, Debug)]
pub struct Work {
    /// What each worker does.
    pub work_type: WorkType,
    /// Linux scheduling policy.
    pub sched_policy: SchedPolicy,
    /// Number of workers. `None` means use `Ctx::workers_per_cgroup`.
    pub num_workers: Option<usize>,
    /// Per-worker affinity intent. Resolved to [`AffinityMode`] at
    /// runtime via [`resolve_affinity_for_cgroup()`](crate::scenario::resolve_affinity_for_cgroup).
    pub affinity: AffinityKind,
    /// NUMA memory placement policy. Applied via `set_mempolicy(2)`
    /// after fork, before the work loop.
    pub mem_policy: MemPolicy,
    /// Optional mode flags for `set_mempolicy(2)`.
    pub mpol_flags: MpolFlags,
}

impl Default for Work {
    fn default() -> Self {
        Self {
            work_type: WorkType::CpuSpin,
            sched_policy: SchedPolicy::Normal,
            num_workers: None,
            affinity: AffinityKind::Inherit,
            mem_policy: MemPolicy::Default,
            mpol_flags: MpolFlags::NONE,
        }
    }
}

impl Work {
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
    pub fn affinity(mut self, a: AffinityKind) -> Self {
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
/// Each field is populated by the worker itself (inside the VM) and
/// serialized via a pipe to the parent process.
///
/// [`Default`] produces a zero/empty report. Callers building a sentinel
/// report should spread `..WorkerReport::default()` rather than listing
/// every field by hand -- the sentinel drifts silently when a field
/// is added.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct WorkerReport {
    /// Worker process ID (from `getpid()` in the forked child).
    /// Stored as `pid_t` (i32) to match the kernel's native type and
    /// avoid the silent u32→i32 sign-cast wraparound at libc
    /// boundaries (kill/waitpid/Pid::from_raw).
    pub tid: i32,
    /// Cumulative work iterations (incremented by `spin_burst` or I/O loops).
    pub work_units: u64,
    /// Thread CPU time from `CLOCK_THREAD_CPUTIME_ID` (ns).
    pub cpu_time_ns: u64,
    /// Wall-clock time from fork-start to stop signal (ns).
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
    /// Longest gap between work iterations (ms). High = task was stuck waiting for CPU.
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
    /// FutexFanOut, SchBench, CacheYield, CachePipe, IoSync, NiceSweep,
    /// AffinityChurn, MutexContention, Sequence with Sleep/Yield/Io
    /// phases.
    pub resume_latencies_ns: Vec<u64>,
    /// Outer-loop iteration count.
    pub iterations: u64,
    /// Delta of /proc/self/schedstat field 2 (run_delay) over the work loop.
    pub schedstat_run_delay_ns: u64,
    /// Delta of /proc/self/schedstat field 3 (timeslices/context switches).
    pub schedstat_ctx_switches: u64,
    /// Delta of /proc/self/schedstat field 1 (cpu_time) over the work loop.
    pub schedstat_cpu_time_ns: u64,
    /// Per-NUMA-node page counts from `/proc/self/numa_maps` after workload.
    /// Keyed by node ID. Empty when numa_maps is unavailable.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub numa_pages: BTreeMap<usize, u64>,
    /// Delta of `/proc/vmstat` `numa_pages_migrated` over the work loop.
    pub vmstat_numa_pages_migrated: u64,
}

/// PID of the scheduler process. Workers kill it on stall to trigger dump.
static SCHED_PID: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

/// In repro mode, don't kill the scheduler on stall — keep it alive for assertions.
static REPRO_MODE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Set the scheduler PID for the work-conservation watchdog.
///
/// Workers send SIGUSR2 to this PID when stuck > 2 seconds,
/// unless repro mode is active (see [`set_repro_mode`]).
#[doc(hidden)]
pub(crate) fn set_sched_pid(pid: i32) {
    SCHED_PID.store(pid, std::sync::atomic::Ordering::Relaxed);
}

/// Enable/disable repro mode. When true, the watchdog is suppressed
/// so the scheduler stays alive for BPF kprobe assertions.
#[doc(hidden)]
pub(crate) fn set_repro_mode(v: bool) {
    REPRO_MODE.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// Handle to spawned worker processes (forked, not threads).
/// Workers block until [`start()`](Self::start) is called.
/// Each worker is a separate process so it can be in its own cgroup.
#[must_use = "dropping a WorkloadHandle immediately kills all worker processes"]
pub struct WorkloadHandle {
    /// Per-worker (pid, report_fd, start_fd). Pid is stored as the
    /// kernel's `pid_t` (i32) rather than u32: Linux `pid_max ≤ 2^22`
    /// always fits in the positive i32 range, so widening to u32 only
    /// risks the classic "value > i32::MAX silently becomes a negative
    /// pid_t, and kill(-1, ...) reaps the whole session" sign-cast bug
    /// at every boundary that feeds a libc function.
    children: Vec<(
        libc::pid_t,
        std::os::unix::io::RawFd,
        std::os::unix::io::RawFd,
    )>,
    started: bool,
    /// Shared mmap regions for futex-based work types (one per worker group). Unmapped on drop.
    futex_ptrs: Vec<*mut u32>,
    /// Size of each futex mmap region (4 for FutexPingPong/FutexFanOut, 16 for SchBench).
    futex_region_size: usize,
    /// MAP_SHARED region of per-worker u64 iteration counters. Workers
    /// atomically store their iteration count; parent reads via
    /// `snapshot_iterations()`. Pointer to the first element; length
    /// is `children.len()`.
    iter_counters: *mut u64,
    /// Number of u64 slots in iter_counters (== num_workers at spawn time).
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
    iter_counters: *mut u64,
    iter_counter_bytes: usize,
    /// Already-forked children with their parent-side pipe fds
    /// (transferred to handle on success).
    children: Vec<(libc::pid_t, i32, i32)>,
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
        }
    }

    /// Transfer live resources into a [`WorkloadHandle`]. Leaves the
    /// guard's `children`, `futex_ptrs`, and `iter_counters` empty so
    /// the guard's subsequent `Drop` only closes the inter-worker
    /// `pipe_pairs` (which the parent never uses post-fork).
    fn into_handle(mut self) -> WorkloadHandle {
        let children = std::mem::take(&mut self.children);
        let futex_ptrs = std::mem::take(&mut self.futex_ptrs);
        let iter_counters = std::mem::replace(&mut self.iter_counters, std::ptr::null_mut());
        let iter_counter_bytes = std::mem::replace(&mut self.iter_counter_bytes, 0);
        let iter_counter_len = iter_counter_bytes / std::mem::size_of::<u64>();
        WorkloadHandle {
            children,
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
        // side.
        for &(pid, _, _) in &self.children {
            unsafe {
                libc::kill(pid, libc::SIGKILL);
                let mut status = 0i32;
                libc::waitpid(pid, &mut status, 0);
            }
        }
        // Close each child's parent-side report/start fds.
        for &(_, rfd, wfd) in &self.children {
            for fd in [rfd, wfd] {
                if fd >= 0 {
                    unsafe {
                        libc::close(fd);
                    }
                }
            }
        }
        // Close every inter-worker pipe pair. Children closed their
        // own inherited copies in the fork arm, so these are the
        // only remaining references.
        for (ab, ba) in &self.pipe_pairs {
            for fd in [ab[0], ab[1], ba[0], ba[1]] {
                unsafe {
                    libc::close(fd);
                }
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
// futex word (via `std::ptr::read_volatile`/`write_volatile`) and
// atomically store into their dedicated iter_counters slot (via
// `AtomicU64::from_ptr`); the parent reads all slots via
// `snapshot_iterations` and is the sole process that munmaps the
// region, on WorkloadHandle::drop after every child has been reaped.
// No Rust reference is ever shared across processes — only raw
// pointers into a MAP_SHARED region — so the inherited alias is not
// an aliasing-rule violation.
unsafe impl Send for WorkloadHandle {}
unsafe impl Sync for WorkloadHandle {}

impl WorkloadHandle {
    /// Fork worker processes. Workers block on a pipe until [`start()`](Self::start)
    /// is called, allowing the caller to move them into cgroups first.
    pub fn spawn(config: &WorkloadConfig) -> Result<Self> {
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
        let futex_region_size = if matches!(config.work_type, WorkType::SchBench { .. }) {
            16
        } else {
            std::mem::size_of::<u32>()
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

        // For FutexPingPong/FutexFanOut/SchBench, allocate one shared region
        // per worker group via MAP_SHARED|MAP_ANONYMOUS so all members of the
        // fork see the same physical page. SchBench needs 16 bytes (futex u32
        // at offset 0, wake timestamp u64 at offset 8); others need 4 bytes.
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
                    anyhow::bail!("mmap failed: {}", std::io::Error::last_os_error());
                }
                unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, futex_region_size) };
                guard.futex_ptrs.push(ptr as *mut u32);
            }
        }

        // Per-worker iteration counter region (MAP_SHARED). Each worker
        // atomically stores its iteration count to slot [i]. The parent
        // reads all slots via snapshot_iterations().
        let iter_counter_len = config.num_workers;
        if iter_counter_len > 0 {
            let size = iter_counter_len * std::mem::size_of::<u64>();
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
                anyhow::bail!(
                    "mmap iter_counters failed: {}",
                    std::io::Error::last_os_error()
                );
            }
            guard.iter_counters = ptr as *mut u64;
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

            // Futex pointer for this worker (FutexPingPong/FutexFanOut).
            let worker_futex: Option<(*mut u32, bool)> = if needs_futex {
                let group_idx = i / futex_group_size;
                let is_first = i % futex_group_size == 0;
                Some((guard.futex_ptrs[group_idx], is_first))
            } else {
                None
            };

            // Shared iteration counter slot for this worker.
            let iter_slot: *mut u64 = if !guard.iter_counters.is_null() {
                unsafe { guard.iter_counters.add(i) }
            } else {
                std::ptr::null_mut()
            };

            // Create pipe for report and a second pipe for "start" signal.
            // Local cleanup on second-pipe failure: the guard has no
            // per-worker tracking of half-allocated pipes, so the first
            // half closes here before the bail.
            let mut report_fds = [0i32; 2];
            if unsafe { libc::pipe(report_fds.as_mut_ptr()) } != 0 {
                anyhow::bail!("pipe failed: {}", std::io::Error::last_os_error());
            }
            let mut start_fds = [0i32; 2];
            if unsafe { libc::pipe(start_fds.as_mut_ptr()) } != 0 {
                unsafe {
                    libc::close(report_fds[0]);
                    libc::close(report_fds[1]);
                }
                anyhow::bail!("pipe failed: {}", std::io::Error::last_os_error());
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
                    anyhow::bail!("fork failed: {}", std::io::Error::last_os_error());
                }
                0 => {
                    // Child: install signal handler FIRST (before start wait)
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
                    //    `panic = "abort"`. ktstr's Cargo.toml does
                    //    not set abort in any profile, so default
                    //    unwind semantics apply. If a downstream
                    //    consumer switches the panic strategy,
                    //    defense (2) still severs guard Drop via
                    //    mem::forget but the abort would also skip
                    //    the `f.write_all(&json)` call, so the
                    //    parent's `stop_and_collect` would see a
                    //    missing WorkerReport — acceptable, the
                    //    parent's sentinel fallback covers that
                    //    case.
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
                            // Reset stop flag in case SIGUSR1 arrived during wait
                            STOP.store(false, Ordering::Relaxed);
                            // Now run
                            let report = worker_main(
                                affinity,
                                config.work_type.clone(),
                                config.sched_policy,
                                config.mem_policy.clone(),
                                config.mpol_flags,
                                worker_pipe_fds,
                                worker_futex,
                                iter_slot,
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

    /// PIDs of all worker processes, in spawn order.
    ///
    /// Returned as `libc::pid_t` — the kernel's native type — so
    /// callers feed them directly into `kill`, `waitpid`,
    /// `Pid::from_raw`, and `cgroup.procs` writes without any
    /// sign-cast at the libc boundary.
    pub fn tids(&self) -> Vec<libc::pid_t> {
        self.children.iter().map(|(pid, _, _)| *pid).collect()
    }

    /// Signal all children to start working (after they've been moved to cgroups).
    pub fn start(&mut self) {
        if self.started {
            return;
        }
        self.started = true;
        for (_, _, start_fd) in &mut self.children {
            unsafe {
                libc::write(*start_fd, b"s".as_ptr() as *const _, 1);
                libc::close(*start_fd);
            }
            // Mark closed so Drop doesn't double-close.
            *start_fd = -1;
        }
    }

    /// Set CPU affinity for worker at `idx`.
    pub fn set_affinity(&self, idx: usize, cpus: &BTreeSet<usize>) -> Result<()> {
        let (pid, _, _) = self.children[idx];
        set_thread_affinity(pid, cpus)
    }

    /// Read all workers' current iteration counts from shared memory.
    ///
    /// Each element is the monotonically increasing iteration count for
    /// that worker, read with Relaxed ordering. Returns an empty vec
    /// if no workers were spawned.
    pub fn snapshot_iterations(&self) -> Vec<u64> {
        if self.iter_counters.is_null() || self.iter_counter_len == 0 {
            return Vec::new();
        }
        (0..self.iter_counter_len)
            .map(|i| {
                let ptr = unsafe { self.iter_counters.add(i) };
                unsafe { std::sync::atomic::AtomicU64::from_ptr(ptr).load(Ordering::Relaxed) }
            })
            .collect()
    }

    /// Send SIGUSR1 to all workers, collect their reports, and wait for exit.
    ///
    /// Auto-starts workers if [`start()`](Self::start) was not called,
    /// then sleeps 500ms to let them begin before signaling stop.
    /// Consumes `self` -- workers cannot be restarted.
    ///
    /// Workers that fail to produce a report (died, timed out, or wrote
    /// corrupt data) get a zeroed-out sentinel report with `work_units: 0`.
    /// This ensures `assert_not_starved` catches dead workers as starvation
    /// failures.
    pub fn stop_and_collect(mut self) -> Vec<WorkerReport> {
        // Auto-start if not explicitly started (workers in parent cgroup)
        let was_started = self.started;
        self.start();

        // If we just started workers, give them time to begin before stopping.
        // 500ms accommodates parallel test runs where CPU contention delays
        // fork/exec of worker processes.
        if !was_started {
            std::thread::sleep(std::time::Duration::from_millis(500));
        }

        let mut reports = Vec::new();
        let children = std::mem::take(&mut self.children);

        // Signal all children to stop.
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
                    unsafe {
                        libc::close(read_fd);
                    }
                }
            } else {
                unsafe {
                    libc::close(read_fd);
                }
            }

            // Wait for child (WNOHANG first, then SIGKILL if still alive).
            // `pid` is already `libc::pid_t`, so no sign cast is
            // needed at the libc boundary.
            let mut status = 0i32;
            let ret = unsafe { libc::waitpid(pid, &mut status, libc::WNOHANG) };
            if ret == 0 {
                unsafe {
                    libc::kill(pid, libc::SIGKILL);
                    libc::waitpid(pid, &mut status, 0);
                }
            }

            if let Ok(report) = serde_json::from_slice::<WorkerReport>(&buf) {
                reports.push(report);
            } else {
                eprintln!(
                    "ktstr: worker pid={pid} returned no report ({} bytes read)",
                    buf.len()
                );
                reports.push(WorkerReport {
                    // Both `pid` and `WorkerReport.tid` are `pid_t`
                    // (i32) now — no cast needed.
                    tid: pid,
                    work_units: 0,
                    cpu_time_ns: 0,
                    wall_time_ns: 0,
                    off_cpu_ns: 0,
                    migration_count: 0,
                    cpus_used: BTreeSet::new(),
                    migrations: Vec::new(),
                    max_gap_ms: 0,
                    max_gap_cpu: 0,
                    max_gap_at_ms: 0,
                    resume_latencies_ns: Vec::new(),
                    iterations: 0,
                    schedstat_run_delay_ns: 0,
                    schedstat_ctx_switches: 0,
                    schedstat_cpu_time_ns: 0,
                    numa_pages: BTreeMap::new(),
                    vmstat_numa_pages_migrated: 0,
                });
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

        // `pid` is `libc::pid_t` — stored as i32 so `Pid::from_raw`
        // receives the kernel's native representation directly, not
        // the sign-cast of a u32 that could alias negative values
        // (including -1, i.e. every process in the session).
        for &(pid, rfd, wfd) in &self.children {
            let nix_pid = Pid::from_raw(pid);
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
    pipe_fds: Option<(i32, i32)>,
    futex: Option<(*mut u32, bool)>,
    iter_slot: *mut u64,
) -> WorkerReport {
    // `getpid()` returns `pid_t` — keep the native type all the way
    // through to `WorkerReport.tid`.
    let tid: libc::pid_t = unsafe { libc::getpid() };

    if let Some(ref cpus) = affinity {
        let _ = set_thread_affinity(tid, cpus);
    }
    let _ = set_sched_policy(tid, sched_policy);
    apply_mempolicy_with_flags(&mem_policy, mpol_flags);

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
    // Lazily allocated per-worker cache buffer (CachePressure, CacheYield, CachePipe, SchBench).
    let mut cache_pressure_buf: Option<Vec<u8>> = None;
    // Separate Vec<u64> for schbench_matrix_multiply: the matrix
    // workload interprets its storage as a sequence of u64 operands,
    // and a `Vec<u8>` has only 1-byte alignment. Reinterpreting a
    // u8-backed buffer as `*mut u64` is UB regardless of buffer
    // contents. Vec<u64> gives natural 8-byte alignment from the
    // allocator.
    let mut schbench_matrix_buf: Option<Vec<u64>> = None;
    // Persistent temp file for IoSync / Phase::Io (opened on first use, removed on exit).
    let mut io_sync_file: Option<(std::fs::File, String)> = None;
    let mut io_seq_file: Option<(std::fs::File, String)> = None;
    // Benchmarking: per-wakeup latency samples (reservoir-sampled) and iteration counter.
    const MAX_WAKE_SAMPLES: usize = 100_000;
    let mut resume_latencies_ns: Vec<u64> = Vec::with_capacity(MAX_WAKE_SAMPLES);
    let mut wake_sample_count: u64 = 0;
    let mut iterations: u64 = 0;
    // AffinityChurn: read effective cpuset once at start via sched_getaffinity.
    // Custom: delegate entirely to the user function. Affinity and
    // sched_policy are already applied above.
    if let WorkType::Custom { run, .. } = &work_type {
        return run(&STOP);
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
    // SchBench: pre-compute matrix dimension from cache_footprint_kb.
    let schbench_matrix_size: usize = if let WorkType::SchBench {
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

    // schedstat snapshot at work-loop start.
    let (ss_cpu_start, ss_delay_start, ss_ts_start) = read_schedstat();

    while !STOP.load(Ordering::Relaxed) {
        match work_type {
            WorkType::CpuSpin => {
                spin_burst(&mut work_units, 1024);
                iterations += 1;
            }
            WorkType::YieldHeavy => {
                work_units = work_units.wrapping_add(1);
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
                    work_units = work_units.wrapping_add(1);
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
                while Instant::now() < burst_end && !STOP.load(Ordering::Relaxed) {
                    spin_burst(&mut work_units, 1024);
                }
                if !STOP.load(Ordering::Relaxed) {
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
                );
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::FutexPingPong { spin_iters } => {
                let (futex_ptr, is_first) = match futex {
                    Some(f) => f,
                    None => break,
                };
                spin_burst(&mut work_units, spin_iters);
                // Worker A waits for 0, wakes partner with 1.
                // Worker B waits for 1, wakes partner with 0.
                let my_val: u32 = if is_first { 0 } else { 1 };
                let partner_val: u32 = if is_first { 1 } else { 0 };
                // Wake partner
                unsafe {
                    std::ptr::write_volatile(futex_ptr, partner_val);
                    futex_wake(futex_ptr, 1);
                }
                // Wait for partner to set our expected value, with timeout
                // to avoid blocking forever if partner has stopped.
                let before_block = Instant::now();
                let ts = libc::timespec {
                    tv_sec: 0,
                    tv_nsec: 100_000_000, // 100ms
                };
                loop {
                    if STOP.load(Ordering::Relaxed) {
                        break;
                    }
                    let cur = unsafe { std::ptr::read_volatile(futex_ptr) };
                    if cur == my_val {
                        reservoir_push(
                            &mut resume_latencies_ns,
                            &mut wake_sample_count,
                            before_block.elapsed().as_nanos() as u64,
                            MAX_WAKE_SAMPLES,
                        );
                        break;
                    }
                    unsafe { futex_wait(futex_ptr, partner_val, &ts) };
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
                );
                // Reset last_iter_time after blocking step
                last_iter_time = Instant::now();
                iterations += 1;
            }
            WorkType::FutexFanOut {
                fan_out,
                spin_iters,
            } => {
                let (futex_ptr, is_messenger) = match futex {
                    Some(f) => f,
                    None => break,
                };
                spin_burst(&mut work_units, spin_iters);
                if is_messenger {
                    // Increment generation counter and wake all receivers.
                    let next = unsafe { std::ptr::read_volatile(futex_ptr) }.wrapping_add(1);
                    unsafe {
                        std::ptr::write_volatile(futex_ptr, next);
                        futex_wake(futex_ptr, fan_out as i32);
                    }
                    // Short spin to let receivers run before next wake cycle.
                    for _ in 0..256 {
                        std::hint::spin_loop();
                    }
                } else {
                    // Receiver: wait for the generation counter to advance.
                    let expected = unsafe { std::ptr::read_volatile(futex_ptr) };
                    let before_block = Instant::now();
                    let ts = libc::timespec {
                        tv_sec: 0,
                        tv_nsec: 100_000_000, // 100ms
                    };
                    loop {
                        if STOP.load(Ordering::Relaxed) {
                            break;
                        }
                        let cur = unsafe { std::ptr::read_volatile(futex_ptr) };
                        if cur != expected {
                            reservoir_push(
                                &mut resume_latencies_ns,
                                &mut wake_sample_count,
                                before_block.elapsed().as_nanos() as u64,
                                MAX_WAKE_SAMPLES,
                            );
                            break;
                        }
                        unsafe { futex_wait(futex_ptr, expected, &ts) };
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
                    if STOP.load(Ordering::Relaxed) {
                        break;
                    }
                    match phase {
                        Phase::Spin(dur) => {
                            let end = Instant::now() + *dur;
                            while Instant::now() < end && !STOP.load(Ordering::Relaxed) {
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
                            while Instant::now() < end && !STOP.load(Ordering::Relaxed) {
                                work_units = work_units.wrapping_add(1);
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
                            while Instant::now() < end && !STOP.load(Ordering::Relaxed) {
                                let _ = f.set_len(0);
                                let _ = f.seek(std::io::SeekFrom::Start(0));
                                let buf = [0u8; 4096];
                                for _ in 0..16 {
                                    let _ = f.write_all(&buf);
                                    work_units = work_units.wrapping_add(1);
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
                        work_units = work_units.wrapping_add(1);
                        iterations += 1;
                    }
                    0 => {
                        unsafe { libc::_exit(0) };
                    }
                    child => {
                        let mut status = 0i32;
                        unsafe { libc::waitpid(child, &mut status, 0) };
                        work_units = work_units.wrapping_add(1);
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
            WorkType::SchBench {
                fan_out,
                operations,
                sleep_usec,
                ..
            } => {
                let (futex_ptr, is_messenger) = match futex {
                    Some(f) => f,
                    None => break,
                };
                // Shared memory layout: [u32 generation @ offset 0] [u64 wake_ns @ offset 8]
                let wake_ts_ptr = unsafe { (futex_ptr as *mut u8).add(8) as *mut u64 };
                if is_messenger {
                    // Messenger: stamp wake time, advance generation, wake workers.
                    let mut ts = libc::timespec {
                        tv_sec: 0,
                        tv_nsec: 0,
                    };
                    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
                    let wake_ns = (ts.tv_sec as u64) * 1_000_000_000 + (ts.tv_nsec as u64);
                    unsafe { std::ptr::write_volatile(wake_ts_ptr, wake_ns) };
                    let next = unsafe { std::ptr::read_volatile(futex_ptr) }.wrapping_add(1);
                    unsafe {
                        std::ptr::write_volatile(futex_ptr, next);
                        futex_wake(futex_ptr, fan_out as i32);
                    }
                    spin_burst(&mut work_units, 256);
                } else {
                    // Worker: wait for generation advance, then do work.
                    let expected = unsafe { std::ptr::read_volatile(futex_ptr) };
                    let ts = libc::timespec {
                        tv_sec: 0,
                        tv_nsec: 100_000_000, // 100ms timeout
                    };
                    loop {
                        if STOP.load(Ordering::Relaxed) {
                            break;
                        }
                        let cur = unsafe { std::ptr::read_volatile(futex_ptr) };
                        if cur != expected {
                            let mut now_ts = libc::timespec {
                                tv_sec: 0,
                                tv_nsec: 0,
                            };
                            unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut now_ts) };
                            let now_ns =
                                (now_ts.tv_sec as u64) * 1_000_000_000 + (now_ts.tv_nsec as u64);
                            let wake_ns = unsafe { std::ptr::read_volatile(wake_ts_ptr) };
                            let latency = now_ns.saturating_sub(wake_ns);
                            reservoir_push(
                                &mut resume_latencies_ns,
                                &mut wake_sample_count,
                                latency,
                                MAX_WAKE_SAMPLES,
                            );
                            break;
                        }
                        unsafe { futex_wait(futex_ptr, expected, &ts) };
                    }
                    if sleep_usec > 0 && !STOP.load(Ordering::Relaxed) {
                        std::thread::sleep(Duration::from_micros(sleep_usec));
                    }
                    if schbench_matrix_size > 0 && !STOP.load(Ordering::Relaxed) {
                        let buf = schbench_matrix_buf.get_or_insert_with(|| {
                            vec![0u64; 3 * schbench_matrix_size * schbench_matrix_size]
                        });
                        for _ in 0..operations {
                            schbench_matrix_multiply(buf, schbench_matrix_size);
                            work_units = work_units.wrapping_add(1);
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
                let region_size = region_kb * 1024;
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
                let page_count = region_size / 4096;
                // xorshift64 PRNG seeded from process ID.
                let mut rng_state = (tid as u64) | 1;
                let xorshift64 = |state: &mut u64| -> u64 {
                    let mut x = *state;
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    *state = x;
                    x
                };
                while !STOP.load(Ordering::Relaxed) {
                    for _ in 0..touches_per_cycle {
                        let page_idx = (xorshift64(&mut rng_state) as usize) % page_count;
                        let page_ptr = unsafe { (ptr as *mut u8).add(page_idx * 4096) };
                        unsafe { std::ptr::write_volatile(page_ptr, 1u8) };
                        work_units = work_units.wrapping_add(1);
                    }
                    unsafe {
                        libc::madvise(ptr, region_size, libc::MADV_DONTNEED);
                    }
                    spin_burst(&mut work_units, spin_iters);
                    iterations += 1;
                }
                unsafe {
                    libc::munmap(ptr, region_size);
                }
            }
            WorkType::MutexContention {
                hold_iters,
                work_iters,
                ..
            } => {
                let (futex_ptr, _) = match futex {
                    Some(f) => f,
                    None => break,
                };
                spin_burst(&mut work_units, work_iters);
                let atom = unsafe { &*(futex_ptr as *const std::sync::atomic::AtomicU32) };
                // CAS acquire: try to set 0 -> 1. On failure, FUTEX_WAIT.
                loop {
                    if STOP.load(Ordering::Relaxed) {
                        break;
                    }
                    if atom
                        .compare_exchange_weak(0, 1, Ordering::Acquire, Ordering::Relaxed)
                        .is_ok()
                    {
                        break;
                    }
                    let before_block = Instant::now();
                    let ts = libc::timespec {
                        tv_sec: 0,
                        tv_nsec: 100_000_000, // 100ms
                    };
                    unsafe {
                        futex_wait(futex_ptr, 1u32 /* expected value (locked) */, &ts)
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
            WorkType::Custom { .. } => unreachable!("handled by early return"),
        }

        // Publish iteration count to shared memory for host-side sampling.
        if !iter_slot.is_null() {
            unsafe {
                std::sync::atomic::AtomicU64::from_ptr(iter_slot)
                    .store(iterations, Ordering::Relaxed);
            }
        }

        if work_units.is_multiple_of(1024) {
            let now = Instant::now();
            let gap = now.duration_since(last_iter_time).as_nanos() as u64;
            if gap > max_gap_ns {
                max_gap_ns = gap;
                max_gap_cpu = last_cpu;
                max_gap_at_ns = now.duration_since(start).as_nanos() as u64;
            }
            // If stuck >2s and not in repro mode, send SIGUSR2 to the
            // scheduler. Default POSIX disposition terminates it, which
            // ktstr detects as a scheduler death. In repro mode, keep it
            // alive for BPF probes.
            if gap > 2_000_000_000 && !REPRO_MODE.load(std::sync::atomic::Ordering::Relaxed) {
                let pid = SCHED_PID.load(std::sync::atomic::Ordering::Relaxed);
                if pid > 0 {
                    let _ = nix::sys::signal::kill(
                        nix::unistd::Pid::from_raw(pid),
                        nix::sys::signal::Signal::SIGUSR2,
                    );
                }
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

    // Final iteration count store for host-side sampling.
    if !iter_slot.is_null() {
        unsafe {
            std::sync::atomic::AtomicU64::from_ptr(iter_slot).store(iterations, Ordering::Relaxed);
        }
    }

    let wall_time = start.elapsed();
    let cpu_time_ns = thread_cpu_time_ns();
    let wall_time_ns = wall_time.as_nanos() as u64;

    // schedstat snapshot at work-loop end; compute deltas.
    let (ss_cpu_end, ss_delay_end, ss_ts_end) = read_schedstat();

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
        iterations,
        schedstat_run_delay_ns: ss_delay_end.saturating_sub(ss_delay_start),
        schedstat_ctx_switches: ss_ts_end.saturating_sub(ss_ts_start),
        schedstat_cpu_time_ns: ss_cpu_end.saturating_sub(ss_cpu_start),
        numa_pages,
        vmstat_numa_pages_migrated: vmstat_migrated_delta,
    }
}

/// CPU spin burst: black_box increment + spin_loop hint, repeated `count` times.
#[inline(always)]
fn spin_burst(work_units: &mut u64, count: u64) {
    for _ in 0..count {
        *work_units = std::hint::black_box(work_units.wrapping_add(1));
        std::hint::spin_loop();
    }
}

/// Strided read-modify-write over a cache buffer.
fn cache_rmw_loop(buf: &mut [u8], stride: usize, iters: u64, work_units: &mut u64) {
    let len = buf.len();
    let mut idx = 0;
    for _ in 0..iters {
        buf[idx] = buf[idx].wrapping_add(1);
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
fn schbench_matrix_multiply(data: &mut [u64], size: usize) {
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
            data[2 * stride + i * size + j] = std::hint::black_box(acc);
        }
    }
}

/// Write 1 byte to partner, poll for response, read, record wake latency.
fn pipe_exchange(
    read_fd: i32,
    write_fd: i32,
    resume_latencies_ns: &mut Vec<u64>,
    wake_sample_count: &mut u64,
    max_wake_samples: usize,
) {
    unsafe { libc::write(write_fd, b"x".as_ptr() as *const _, 1) };
    let before_block = Instant::now();
    let mut pfd = libc::pollfd {
        fd: read_fd,
        events: libc::POLLIN,
        revents: 0,
    };
    loop {
        if STOP.load(Ordering::Relaxed) {
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

fn resolve_affinity(mode: &AffinityMode) -> Result<Option<BTreeSet<usize>>> {
    match mode {
        AffinityMode::None => Ok(None),
        AffinityMode::Fixed(cpus) => Ok(Some(cpus.clone())),
        AffinityMode::SingleCpu(cpu) => Ok(Some([*cpu].into_iter().collect())),
        AffinityMode::Random { from, count } => {
            use rand::seq::IndexedRandom;
            if *count == 0 {
                anyhow::bail!(
                    "AffinityMode::Random.count must be > 0; a zero count \
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

/// Read /proc/self/schedstat and return (cpu_time_ns, run_delay_ns, timeslices).
/// Returns (0, 0, 0) on failure (e.g. schedstats not enabled).
fn read_schedstat() -> (u64, u64, u64) {
    let data = match std::fs::read_to_string("/proc/self/schedstat") {
        Ok(d) => d,
        Err(_) => return (0, 0, 0),
    };
    let mut parts = data.split_whitespace();
    let cpu_time = parts
        .next()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);
    let run_delay = parts
        .next()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);
    let timeslices = parts
        .next()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);
    (cpu_time, run_delay, timeslices)
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

fn thread_cpu_time_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    let ret = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    if ret != 0 {
        return 0;
    }
    (ts.tv_sec as u64) * 1_000_000_000 + (ts.tv_nsec as u64)
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

    #[test]
    fn work_type_all_names_count() {
        assert_eq!(WorkType::ALL_NAMES.len(), 20);
    }

    #[test]
    fn resolve_affinity_none() {
        let r = resolve_affinity(&AffinityMode::None).unwrap();
        assert!(r.is_none());
    }

    #[test]
    fn resolve_affinity_fixed() {
        let cpus: BTreeSet<usize> = [0, 1, 2].into_iter().collect();
        let r = resolve_affinity(&AffinityMode::Fixed(cpus.clone())).unwrap();
        assert_eq!(r, Some(cpus));
    }

    #[test]
    fn resolve_affinity_single_cpu() {
        let r = resolve_affinity(&AffinityMode::SingleCpu(5)).unwrap();
        assert_eq!(r, Some([5].into_iter().collect()));
    }

    #[test]
    fn resolve_affinity_random() {
        let from: BTreeSet<usize> = (0..8).collect();
        let r = resolve_affinity(&AffinityMode::Random { from, count: 3 }).unwrap();
        let cpus = r.unwrap();
        assert_eq!(cpus.len(), 3);
        assert!(cpus.iter().all(|c| *c < 8));
    }

    #[test]
    fn resolve_affinity_random_clamps_count() {
        let from: BTreeSet<usize> = [0, 1].into_iter().collect();
        let r = resolve_affinity(&AffinityMode::Random { from, count: 10 }).unwrap();
        assert_eq!(r.unwrap().len(), 2);
    }

    #[test]
    fn workload_config_default() {
        let c = WorkloadConfig::default();
        assert_eq!(c.num_workers, 1);
        assert!(matches!(c.work_type, WorkType::CpuSpin));
        assert!(matches!(c.sched_policy, SchedPolicy::Normal));
        assert!(matches!(c.affinity, AffinityMode::None));
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
            iterations: 10,
            schedstat_run_delay_ns: 500_000,
            schedstat_ctx_switches: 20,
            schedstat_cpu_time_ns: 4_000_000_000,
            numa_pages: BTreeMap::new(),
            vmstat_numa_pages_migrated: 0,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: WorkerReport = serde_json::from_str(&json).unwrap();
        assert_eq!(r.tid, r2.tid);
        assert_eq!(r.work_units, r2.work_units);
        assert_eq!(r.migration_count, r2.migration_count);
        assert_eq!(r.cpus_used, r2.cpus_used);
        assert_eq!(r.max_gap_ms, r2.max_gap_ms);
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
            affinity: AffinityMode::None,
            work_type: WorkType::CpuSpin,
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        assert_eq!(h.tids().len(), 2);
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
            affinity: AffinityMode::None,
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
        let config = WorkloadConfig {
            num_workers: 1,
            affinity: AffinityMode::None,
            work_type: WorkType::YieldHeavy,
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        h.start();
        std::thread::sleep(std::time::Duration::from_millis(200));
        let reports = h.stop_and_collect();
        assert_eq!(reports.len(), 1);
        assert!(reports[0].work_units > 0);
    }

    #[test]
    fn spawn_mixed_produces_work() {
        let config = WorkloadConfig {
            num_workers: 1,
            affinity: AffinityMode::None,
            work_type: WorkType::Mixed,
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        h.start();
        std::thread::sleep(std::time::Duration::from_millis(200));
        let reports = h.stop_and_collect();
        assert_eq!(reports.len(), 1);
        assert!(reports[0].work_units > 0);
    }

    /// Regression guard for the sign-cast bug: every pid returned
    /// from `tids()` must be a positive, live `pid_t` that round-trips
    /// through `Pid::from_raw` + `kill(_, None)` (the "exists" probe).
    /// A negative pid would silently broadcast SIGKILL to a process
    /// group; a stale/reaped pid would fail the probe with ESRCH.
    /// Either indicates storage upstream re-introduced the u32
    /// wraparound or dropped a child on the floor.
    #[test]
    fn spawn_pids_fit_in_pid_t() {
        let config = WorkloadConfig {
            num_workers: 4,
            affinity: AffinityMode::None,
            work_type: WorkType::CpuSpin,
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let h = WorkloadHandle::spawn(&config).unwrap();
        for pid in h.tids() {
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
    /// spawns successfully, then verifies Drop cleans up when the
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
            affinity: AffinityMode::None,
            work_type: WorkType::PipeIo { burst_iters: 4 },
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let h = WorkloadHandle::spawn(&config).unwrap();
        let pids = h.tids();
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
            affinity: AffinityMode::None,
            work_type: WorkType::CpuSpin,
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        let tids = h.tids();
        assert_eq!(tids.len(), 4);
        let unique: std::collections::HashSet<libc::pid_t> = tids.iter().copied().collect();
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
            affinity: AffinityMode::Fixed([0].into_iter().collect()),
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
        let pids = h.tids();
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
    // Cleanup verification strategy:
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

    /// Lower RLIMIT_NOFILE to `limit`. Returns true on success.
    /// Safe to call in the post-fork single-threaded child; not safe
    /// to call in the parent test (would affect sibling tests).
    fn set_rlimit_nofile(limit: u64) -> bool {
        let rl = libc::rlimit {
            rlim_cur: limit,
            rlim_max: limit,
        };
        unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &rl) == 0 }
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
            // RLIMIT_NOFILE is a hard limit on the highest fd
            // number + 1, not a headroom value — we need to pass a
            // value slightly above baseline so the first pipe pair
            // succeeds but the second pair's first `pipe(ab)` does
            // not. baseline + 5 permits 5 new fds: 4 for the first
            // pipe pair (ab+ba) and 1 leftover. The second pair's
            // `pipe(ab)` needs 2 fds against that 1 slot and fails
            // with EMFILE.
            let target = (baseline + 5) as u64;
            if !set_rlimit_nofile(target) {
                return 13;
            }
            let config = WorkloadConfig {
                num_workers: 4,
                affinity: AffinityMode::None,
                work_type: WorkType::PipeIo { burst_iters: 1 },
                sched_policy: SchedPolicy::Normal,
                ..Default::default()
            };
            let result = WorkloadHandle::spawn(&config);
            if result.is_ok() {
                return 10; // Failure did not trigger.
            }
            // SpawnGuard::Drop has already run on the `?`/`bail!`
            // exit. Raise the limit back up so reading
            // /proc/self/fd for the post-check does not itself
            // fail with EMFILE.
            let err_msg = format!("{:#}", result.as_ref().err().unwrap());
            let _ = set_rlimit_nofile(u64::MAX);
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
    /// time the local cleanup has closed the report+start pipes
    /// (lines 1278-1283), so the guard carries only its empty
    /// `pipe_pairs`, zero children, and the iter_counters mmap. The
    /// Drop munmaps the iter_counters region (no-op for the fd
    /// count but proves the guard path fires) and returns cleanly.
    /// No zombies, no fd leak.
    #[test]
    fn spawn_guard_cleans_up_on_fork_eagain() {
        let code = run_in_forked_child(|| {
            let baseline = count_open_fds();
            if !set_rlimit_nproc_zero_headroom() {
                return 13;
            }
            let config = WorkloadConfig {
                num_workers: 1,
                affinity: AffinityMode::None,
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
        let config = WorkloadConfig {
            num_workers: 1,
            affinity: AffinityMode::None,
            work_type: WorkType::IoSync,
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        h.start();
        std::thread::sleep(std::time::Duration::from_millis(200));
        let reports = h.stop_and_collect();
        assert_eq!(reports.len(), 1);
        assert!(reports[0].work_units > 0);
    }

    #[test]
    fn spawn_bursty_produces_work() {
        let config = WorkloadConfig {
            num_workers: 1,
            affinity: AffinityMode::None,
            work_type: WorkType::Bursty {
                burst_ms: 50,
                sleep_ms: 50,
            },
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        h.start();
        std::thread::sleep(std::time::Duration::from_millis(300));
        let reports = h.stop_and_collect();
        assert_eq!(reports.len(), 1);
        assert!(reports[0].work_units > 0);
    }

    #[test]
    fn spawn_pipeio_produces_work() {
        let config = WorkloadConfig {
            num_workers: 2,
            affinity: AffinityMode::None,
            work_type: WorkType::PipeIo { burst_iters: 1024 },
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        h.start();
        std::thread::sleep(std::time::Duration::from_millis(300));
        let reports = h.stop_and_collect();
        assert_eq!(reports.len(), 2);
        for r in &reports {
            assert!(r.work_units > 0, "PipeIo worker {} did no work", r.tid);
        }
    }

    #[test]
    fn spawn_pipeio_odd_workers_fails() {
        let config = WorkloadConfig {
            num_workers: 3,
            affinity: AffinityMode::None,
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
        assert!(h.tids().is_empty());
        let reports = h.stop_and_collect();
        assert!(reports.is_empty());
    }

    #[test]
    fn tids_count_matches_num_workers() {
        for n in [1, 3, 5] {
            let config = WorkloadConfig {
                num_workers: n,
                ..Default::default()
            };
            let h = WorkloadHandle::spawn(&config).unwrap();
            assert_eq!(
                h.tids().len(),
                n,
                "tids().len() should match num_workers={n}"
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
            iterations: 0,
            schedstat_run_delay_ns: 0,
            schedstat_ctx_switches: 0,
            schedstat_cpu_time_ns: 0,
            numa_pages: BTreeMap::new(),
            vmstat_numa_pages_migrated: 0,
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
            iterations: u64::MAX,
            schedstat_run_delay_ns: u64::MAX,
            schedstat_ctx_switches: u64::MAX,
            schedstat_cpu_time_ns: u64::MAX,
            numa_pages: BTreeMap::new(),
            vmstat_numa_pages_migrated: 0,
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
            affinity: AffinityMode::None,
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
    fn set_sched_pid_stores_value() {
        set_sched_pid(12345);
        let v = SCHED_PID.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(v, 12345);
        // Reset
        set_sched_pid(0);
    }

    #[test]
    fn set_repro_mode_stores_value() {
        set_repro_mode(true);
        assert!(REPRO_MODE.load(std::sync::atomic::Ordering::Relaxed));
        set_repro_mode(false);
        assert!(!REPRO_MODE.load(std::sync::atomic::Ordering::Relaxed));
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
            affinity: AffinityMode::None,
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
            affinity: AffinityMode::None,
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
            affinity: AffinityMode::None,
            work_type: WorkType::PipeIo { burst_iters: 512 },
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        assert_eq!(h.tids().len(), 4);
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
    fn set_sched_policy_fifo_returns_result() {
        let pid: libc::pid_t = unsafe { libc::getpid() };
        let result = set_sched_policy(pid, SchedPolicy::Fifo(1));
        // SCHED_FIFO requires CAP_SYS_NICE — fails without privileges.
        assert!(
            result.is_err(),
            "SCHED_FIFO should fail without CAP_SYS_NICE"
        );
    }

    #[test]
    fn set_sched_policy_rr_returns_result() {
        let pid: libc::pid_t = unsafe { libc::getpid() };
        let result = set_sched_policy(pid, SchedPolicy::RoundRobin(1));
        // SCHED_RR requires CAP_SYS_NICE — fails without privileges.
        assert!(result.is_err(), "SCHED_RR should fail without CAP_SYS_NICE");
    }

    #[test]
    fn resolve_affinity_random_single_cpu_pool() {
        let from: BTreeSet<usize> = [7].into_iter().collect();
        let r = resolve_affinity(&AffinityMode::Random { from, count: 1 }).unwrap();
        assert_eq!(r.unwrap(), [7].into_iter().collect());
    }

    // -- WorkType::name edge cases --

    #[test]
    fn work_type_name_io_sync() {
        assert_eq!(WorkType::IoSync.name(), "IoSync");
    }

    #[test]
    fn work_type_name_mixed() {
        assert_eq!(WorkType::Mixed.name(), "Mixed");
    }

    #[test]
    fn work_type_name_yield_heavy() {
        assert_eq!(WorkType::YieldHeavy.name(), "YieldHeavy");
    }

    // -- WorkType::from_name edge cases --

    #[test]
    fn work_type_from_name_case_sensitive() {
        assert!(WorkType::from_name("cpuspin").is_none());
        assert!(WorkType::from_name("CPUSPIN").is_none());
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
        // SCHED_BATCH may fail under sched_ext or without CAP_SYS_NICE.
        match result {
            Ok(()) => {
                let pol = unsafe { libc::sched_getscheduler(pid) };
                // sched_ext may override the effective policy, so the
                // kernel can report a different value than SCHED_BATCH
                // even after a successful sched_setscheduler.
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
        // SCHED_IDLE may fail under sched_ext or without CAP_SYS_NICE.
        match result {
            Ok(()) => {
                let pol = unsafe { libc::sched_getscheduler(pid) };
                // sched_ext may override the effective policy, so the
                // kernel can report a different value than SCHED_IDLE
                // even after a successful sched_setscheduler.
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
        let a = AffinityMode::Fixed([0, 1, 7].into_iter().collect());
        let s = format!("{:?}", a);
        assert!(s.contains("0"), "must show CPU 0");
        assert!(s.contains("1"), "must show CPU 1");
        assert!(s.contains("7"), "must show CPU 7");
        // Different CPU sets produce different output.
        let b = AffinityMode::Fixed([3, 4].into_iter().collect());
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
        let a = AffinityMode::Random {
            from: cpus.clone(),
            count: 2,
        };
        let b = a.clone();
        match b {
            AffinityMode::Random { from, count } => {
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
            affinity: AffinityMode::SingleCpu(3),
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
            iterations: 0,
            schedstat_run_delay_ns: 0,
            schedstat_ctx_switches: 0,
            schedstat_cpu_time_ns: 0,
            numa_pages: BTreeMap::new(),
            vmstat_numa_pages_migrated: 0,
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
            iterations: 0,
            schedstat_run_delay_ns: 0,
            schedstat_ctx_switches: 0,
            schedstat_cpu_time_ns: 0,
            numa_pages: BTreeMap::new(),
            vmstat_numa_pages_migrated: 0,
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
        let err = resolve_affinity(&AffinityMode::Random { from, count: 0 }).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("count") && msg.contains("> 0"),
            "error must name the field: {msg}"
        );
    }

    #[test]
    fn resolve_affinity_random_empty_pool_is_none() {
        // Regression: AffinityMode::Random { from: empty, count } previously
        // produced an empty affinity mask rejected by sched_setaffinity
        // with EINVAL. Empty pool must short-circuit to Ok(None).
        let from: BTreeSet<usize> = BTreeSet::new();
        let r = resolve_affinity(&AffinityMode::Random { from, count: 1 }).unwrap();
        assert!(r.is_none(), "empty Random pool must resolve to no affinity");
    }

    // -- spawn and collect edge cases --

    #[test]
    fn spawn_single_worker_reports_cpus() {
        let config = WorkloadConfig {
            num_workers: 1,
            affinity: AffinityMode::None,
            work_type: WorkType::CpuSpin,
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        h.start();
        std::thread::sleep(std::time::Duration::from_millis(100));
        let reports = h.stop_and_collect();
        assert_eq!(reports.len(), 1);
        assert!(
            !reports[0].cpus_used.is_empty(),
            "should report at least one CPU"
        );
    }

    #[test]
    fn workload_handle_tids_ordered() {
        let config = WorkloadConfig {
            num_workers: 3,
            ..Default::default()
        };
        let h = WorkloadHandle::spawn(&config).unwrap();
        let tids = h.tids();
        assert_eq!(tids.len(), 3);
        // PIDs should all be positive
        for tid in &tids {
            assert!(*tid > 0);
        }
        drop(h);
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
    fn read_schedstat_returns_triple() {
        // Verifies the function parses without panicking.
        let (cpu_time, run_delay, timeslices) = read_schedstat();
        let _ = (cpu_time, run_delay, timeslices);
    }

    // -- FutexFanOut tests --

    #[test]
    fn spawn_futex_fanout_produces_work() {
        let config = WorkloadConfig {
            num_workers: 5, // 1 messenger + 4 receivers
            affinity: AffinityMode::None,
            work_type: WorkType::FutexFanOut {
                fan_out: 4,
                spin_iters: 1024,
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
            assert!(r.work_units > 0, "FutexFanOut worker {} did no work", r.tid);
        }
    }

    #[test]
    fn spawn_futex_fanout_receivers_record_wake_latency() {
        let config = WorkloadConfig {
            num_workers: 5,
            affinity: AffinityMode::None,
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
    fn spawn_futex_fanout_bad_worker_count_fails() {
        let config = WorkloadConfig {
            num_workers: 3, // not divisible by 5
            affinity: AffinityMode::None,
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
    fn spawn_futex_fanout_two_groups() {
        let config = WorkloadConfig {
            num_workers: 10, // 2 groups of (1+4)
            affinity: AffinityMode::None,
            work_type: WorkType::FutexFanOut {
                fan_out: 4,
                spin_iters: 512,
            },
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        assert_eq!(h.tids().len(), 10);
        h.start();
        std::thread::sleep(std::time::Duration::from_millis(500));
        let reports = h.stop_and_collect();
        assert_eq!(reports.len(), 10);
        for r in &reports {
            assert!(r.work_units > 0, "worker {} did no work", r.tid);
        }
    }

    #[test]
    fn spawn_futex_fanout_fan_out_one() {
        // Minimal fan-out: 1 messenger + 1 receiver per group (like ping-pong).
        let config = WorkloadConfig {
            num_workers: 2,
            affinity: AffinityMode::None,
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
    fn work_type_futex_fanout_name() {
        let wt = WorkType::FutexFanOut {
            fan_out: 4,
            spin_iters: 1024,
        };
        assert_eq!(wt.name(), "FutexFanOut");
    }

    #[test]
    fn work_type_futex_fanout_from_name() {
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
    fn work_type_futex_fanout_group_size() {
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
            affinity: AffinityMode::None,
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

    // -- Work builder --

    #[test]
    fn work_builder_chain() {
        let w = Work::default()
            .workers(8)
            .work_type(WorkType::bursty(10, 20))
            .sched_policy(SchedPolicy::Batch)
            .affinity(AffinityKind::SingleCpu);
        assert_eq!(w.num_workers, Some(8));
        assert!(matches!(
            w.work_type,
            WorkType::Bursty {
                burst_ms: 10,
                sleep_ms: 20
            }
        ));
        assert!(matches!(w.sched_policy, SchedPolicy::Batch));
        assert!(matches!(w.affinity, AffinityKind::SingleCpu));
    }

    #[test]
    fn work_default_values() {
        let w = Work::default();
        assert_eq!(w.num_workers, None);
        assert!(matches!(w.work_type, WorkType::CpuSpin));
        assert!(matches!(w.sched_policy, SchedPolicy::Normal));
        assert!(matches!(w.affinity, AffinityKind::Inherit));
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
    fn sched_policy_fifo_default_priority() {
        match SchedPolicy::fifo(1) {
            SchedPolicy::Fifo(p) => assert_eq!(p, 1),
            _ => panic!("expected Fifo(1)"),
        }
    }

    #[test]
    fn sched_policy_rr_default_priority() {
        match SchedPolicy::round_robin(1) {
            SchedPolicy::RoundRobin(p) => assert_eq!(p, 1),
            _ => panic!("expected RoundRobin(1)"),
        }
    }

    #[test]
    fn spawn_futex_ping_pong_produces_work() {
        let config = WorkloadConfig {
            num_workers: 2,
            affinity: AffinityMode::None,
            work_type: WorkType::FutexPingPong { spin_iters: 1024 },
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        h.start();
        std::thread::sleep(std::time::Duration::from_millis(500));
        let reports = h.stop_and_collect();
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
        let config = WorkloadConfig {
            num_workers: 1,
            affinity: AffinityMode::None,
            work_type: WorkType::CachePressure {
                size_kb: 32,
                stride: 64,
            },
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        h.start();
        std::thread::sleep(std::time::Duration::from_millis(200));
        let reports = h.stop_and_collect();
        assert_eq!(reports.len(), 1);
        assert!(reports[0].work_units > 0);
    }

    #[test]
    fn spawn_cache_yield_produces_work() {
        let config = WorkloadConfig {
            num_workers: 1,
            affinity: AffinityMode::None,
            work_type: WorkType::CacheYield {
                size_kb: 32,
                stride: 64,
            },
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        h.start();
        std::thread::sleep(std::time::Duration::from_millis(200));
        let reports = h.stop_and_collect();
        assert_eq!(reports.len(), 1);
        assert!(reports[0].work_units > 0);
    }

    #[test]
    fn spawn_cache_pipe_produces_work() {
        let config = WorkloadConfig {
            num_workers: 2,
            affinity: AffinityMode::None,
            work_type: WorkType::CachePipe {
                size_kb: 32,
                burst_iters: 1024,
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
            assert!(r.work_units > 0, "CachePipe worker {} did no work", r.tid);
        }
    }

    #[test]
    fn spawn_sequence_produces_work() {
        let config = WorkloadConfig {
            num_workers: 1,
            affinity: AffinityMode::None,
            work_type: WorkType::Sequence {
                first: Phase::Spin(Duration::from_millis(10)),
                rest: vec![Phase::Yield(Duration::from_millis(10))],
            },
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        h.start();
        std::thread::sleep(std::time::Duration::from_millis(200));
        let reports = h.stop_and_collect();
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
            iterations: 0,
            schedstat_run_delay_ns: 0,
            schedstat_ctx_switches: 0,
            schedstat_cpu_time_ns: 0,
            numa_pages: BTreeMap::new(),
            vmstat_numa_pages_migrated: 0,
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
        while !stop.load(Ordering::Relaxed) {
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
            iterations: work_units,
            schedstat_run_delay_ns: 0,
            schedstat_ctx_switches: 0,
            schedstat_cpu_time_ns: 0,
            numa_pages: BTreeMap::new(),
            vmstat_numa_pages_migrated: 0,
        }
    }

    #[test]
    fn spawn_custom_produces_work() {
        let config = WorkloadConfig {
            num_workers: 1,
            affinity: AffinityMode::None,
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
    }

    // -- SchBench tests --

    #[test]
    fn schbench_name() {
        let wt = WorkType::SchBench {
            fan_out: 4,
            cache_footprint_kb: 256,
            operations: 5,
            sleep_usec: 100,
        };
        assert_eq!(wt.name(), "SchBench");
    }

    #[test]
    fn schbench_from_name() {
        let wt = WorkType::from_name("SchBench").unwrap();
        match wt {
            WorkType::SchBench {
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
            _ => panic!("expected SchBench"),
        }
    }

    #[test]
    fn schbench_group_size() {
        let wt = WorkType::schbench(4, 256, 5, 100);
        assert_eq!(wt.worker_group_size(), Some(5));
        let wt2 = WorkType::schbench(1, 256, 5, 100);
        assert_eq!(wt2.worker_group_size(), Some(2));
    }

    #[test]
    fn schbench_needs_shared_mem() {
        assert!(WorkType::schbench(4, 256, 5, 100).needs_shared_mem());
    }

    #[test]
    fn schbench_needs_cache_buf() {
        assert!(WorkType::schbench(4, 256, 5, 100).needs_cache_buf());
    }

    #[test]
    fn spawn_schbench_produces_work() {
        let config = WorkloadConfig {
            num_workers: 5, // 1 messenger + 4 workers
            affinity: AffinityMode::None,
            work_type: WorkType::SchBench {
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
            assert!(r.work_units > 0, "SchBench worker {} did no work", r.tid);
        }
        let has_latencies = reports.iter().any(|r| !r.resume_latencies_ns.is_empty());
        assert!(has_latencies, "workers should record wake latencies");
    }

    #[test]
    fn spawn_schbench_bad_worker_count_fails() {
        let config = WorkloadConfig {
            num_workers: 3,
            affinity: AffinityMode::None,
            work_type: WorkType::SchBench {
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

    #[test]
    fn spawn_schbench_two_groups() {
        let config = WorkloadConfig {
            num_workers: 10, // 2 groups of (1 messenger + 4 workers)
            affinity: AffinityMode::None,
            work_type: WorkType::SchBench {
                fan_out: 4,
                cache_footprint_kb: 256,
                operations: 5,
                sleep_usec: 100,
            },
            sched_policy: SchedPolicy::Normal,
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        assert_eq!(h.tids().len(), 10);
        h.start();
        std::thread::sleep(std::time::Duration::from_millis(500));
        let reports = h.stop_and_collect();
        assert_eq!(reports.len(), 10);
        for r in &reports {
            assert!(r.work_units > 0, "SchBench worker {} did no work", r.tid);
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
        let w = Work::default().mpol_flags(MpolFlags::STATIC_NODES);
        assert_eq!(w.mpol_flags, MpolFlags::STATIC_NODES);
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
        let w = Work::default().mem_policy(MemPolicy::Bind([0].into_iter().collect()));
        assert!(matches!(w.mem_policy, MemPolicy::Bind(_)));
    }

    #[test]
    fn work_default_mempolicy_is_default() {
        let w = Work::default();
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

    #[test]
    fn spawn_page_fault_churn_produces_work() {
        let config = WorkloadConfig {
            num_workers: 2,
            affinity: AffinityMode::None,
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
        std::thread::sleep(std::time::Duration::from_millis(300));
        let reports = h.stop_and_collect();
        assert_eq!(reports.len(), 2);
        for r in &reports {
            assert!(
                r.work_units > 0,
                "PageFaultChurn worker {} did no work",
                r.tid
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
        let config = WorkloadConfig {
            num_workers: 4,
            affinity: AffinityMode::None,
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
            affinity: AffinityMode::None,
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
            affinity: AffinityMode::None,
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
}
