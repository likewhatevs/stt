//! Worker process management and telemetry.
//!
//! Workers are `fork()`ed processes (not threads) so each can be placed
//! in its own cgroup. Key types:
//! - [`WorkType`] -- what each worker does (CPU spin, yield, I/O, bursty, pipe)
//! - [`WorkloadConfig`] -- spawn configuration (count, affinity, work type, policy)
//! - [`WorkloadHandle`] -- RAII handle to spawned workers
//! - [`WorkerReport`] -- per-worker telemetry collected after stop
//! - [`AffinityMode`] -- resolved CPU affinity for workers
//! - [`Work`] -- workload definition for a single group of workers within a cgroup
//! - [`Phase`] -- a single phase in a [`WorkType::Sequence`] compound work pattern
//! - [`SchedPolicy`] -- Linux scheduling policy for a worker process
//!
//! See the [Work Types](https://likewhatevs.github.io/stt/guide/concepts/work-types.html)
//! and [Worker Processes](https://likewhatevs.github.io/stt/guide/architecture/workers.html)
//! chapters of the guide.

use anyhow::{Context, Result};
use std::collections::BTreeSet;
use std::io::{Read, Seek, Write};
use std::time::{Duration, Instant};

/// Resolved CPU affinity for a worker process.
///
/// Created from [`AffinityKind`](crate::scenario::AffinityKind) at
/// runtime based on topology and cpuset assignments.
#[derive(Debug, Clone)]
pub enum AffinityMode {
    /// No affinity constraint.
    None,
    /// Pin to a specific set of CPUs.
    Fixed(BTreeSet<usize>),
    /// Pin to `count` randomly-chosen CPUs from `from`.
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
    /// Synchronous I/O (write + fsync) for the given duration.
    Io(Duration),
}

/// What each worker process does during a scenario.
///
/// Different work types exercise different scheduler code paths:
/// CPU-bound, yield-heavy, I/O, bursty, or inter-process communication.
///
/// ```
/// # use stt::workload::WorkType;
/// let wt = WorkType::from_name("CpuSpin").unwrap();
/// assert!(matches!(wt, WorkType::CpuSpin));
///
/// let bursty = WorkType::Bursty { burst_ms: 10, sleep_ms: 5 };
/// assert!(matches!(bursty, WorkType::Bursty { .. }));
///
/// assert!(WorkType::from_name("nonexistent").is_none());
/// ```
#[derive(Debug, Clone)]
pub enum WorkType {
    /// Tight CPU spin loop (1024 iterations per cycle).
    CpuSpin,
    /// Repeated sched_yield with minimal CPU work.
    YieldHeavy,
    /// CPU spin burst followed by sched_yield.
    Mixed,
    /// Synchronous file I/O: write 64 KB + fsync per cycle.
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
    /// Cache pressure followed by sched_yield(). Tests wake_affine
    /// placement after voluntary preemption.
    CacheYield { size_kb: usize, stride: usize },
    /// Cache pressure burst then 1-byte pipe exchange with a partner
    /// worker. Combines cache-hot working set with cross-CPU wake
    /// placement. Requires even num_workers.
    CachePipe { size_kb: usize, burst_iters: u64 },
    /// 1:N fan-out wake pattern (schbench-style). One messenger per group
    /// does CPU work then wakes N receivers via FUTEX_WAKE. Receivers
    /// measure wake-to-run latency. Requires num_workers divisible by
    /// (fan_out + 1).
    FutexFanOut { fan_out: usize, spin_iters: u64 },
    /// Compound work pattern: loop through phases in order, repeat.
    /// Each phase runs for its duration before the next starts.
    Sequence { first: Phase, rest: Vec<Phase> },
}

impl WorkType {
    /// PascalCase names for all variants, matching the enum arm names.
    ///
    /// Includes `"Sequence"` even though [`from_name`](Self::from_name)
    /// cannot construct it (sequences require explicit phases).
    pub const ALL_NAMES: &[&'static str] = &[
        "CpuSpin",
        "YieldHeavy",
        "Mixed",
        "IoSync",
        "Bursty",
        "PipeIo",
        "FutexPingPong",
        "CachePressure",
        "CacheYield",
        "CachePipe",
        "FutexFanOut",
        "Sequence",
    ];

    /// PascalCase name of this variant, matching [`ALL_NAMES`](Self::ALL_NAMES).
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
        }
    }

    /// Look up a variant by PascalCase name and return it with default
    /// parameters. Returns `None` for unknown names and for `"Sequence"`
    /// (which requires explicit phases).
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
            _ => None,
        }
    }

    /// Whether this work type needs a pre-fork shared memory region (MAP_SHARED mmap).
    pub fn needs_shared_mem(&self) -> bool {
        matches!(
            self,
            WorkType::FutexPingPong { .. } | WorkType::FutexFanOut { .. }
        )
    }

    /// Whether this work type allocates a per-worker cache buffer post-fork.
    pub fn needs_cache_buf(&self) -> bool {
        matches!(
            self,
            WorkType::CachePressure { .. }
                | WorkType::CacheYield { .. }
                | WorkType::CachePipe { .. }
        )
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
}

impl Default for WorkloadConfig {
    fn default() -> Self {
        Self {
            num_workers: 1,
            affinity: AffinityMode::None,
            work_type: WorkType::CpuSpin,
            sched_policy: SchedPolicy::Normal,
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
/// # use stt::workload::{Work, WorkType, SchedPolicy};
/// let w = Work::default()
///     .workers(4)
///     .work_type(WorkType::Bursty { burst_ms: 50, sleep_ms: 100 })
///     .sched_policy(SchedPolicy::Batch);
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
}

impl Default for Work {
    fn default() -> Self {
        Self {
            work_type: WorkType::CpuSpin,
            sched_policy: SchedPolicy::Normal,
            num_workers: None,
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorkerReport {
    /// Worker process ID (from `getpid()` in the forked child).
    pub tid: u32,
    /// Cumulative work iterations (incremented by `spin_burst` or I/O loops).
    pub work_units: u64,
    /// Thread CPU time from `CLOCK_PROCESS_CPUTIME_ID` (ns).
    pub cpu_time_ns: u64,
    /// Wall-clock time from fork-start to stop signal (ns).
    pub wall_time_ns: u64,
    /// `wall_time_ns - cpu_time_ns`: time spent runnable but not on-CPU (ns).
    pub runnable_ns: u64,
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
    /// Per-wakeup latency samples (ns). Populated for blocking work types
    /// (FutexPingPong, FutexFanOut, CachePipe, Bursty, CacheYield, PipeIo,
    /// Sequence with Sleep or Yield phases).
    #[serde(default)]
    pub wake_latencies_ns: Vec<u64>,
    /// Outer-loop iteration count.
    #[serde(default)]
    pub iterations: u64,
    /// Delta of /proc/self/schedstat field 2 (run_delay) over the work loop.
    #[serde(default)]
    pub schedstat_run_delay_ns: u64,
    /// Delta of /proc/self/schedstat field 3 (timeslices/context switches).
    #[serde(default)]
    pub schedstat_ctx_switches: u64,
    /// Delta of /proc/self/schedstat field 1 (cpu_time) over the work loop.
    #[serde(default)]
    pub schedstat_cpu_time_ns: u64,
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
    children: Vec<(u32, std::os::unix::io::RawFd, std::os::unix::io::RawFd)>,
    started: bool,
    /// Shared mmap regions for futex-based work types (one per worker group). Unmapped on drop.
    futex_ptrs: Vec<*mut u32>,
    /// MAP_SHARED region of per-worker u64 iteration counters. Workers
    /// atomically store their iteration count; parent reads via
    /// `snapshot_iterations()`. Pointer to the first element; length
    /// is `children.len()`.
    iter_counters: *mut u64,
    /// Number of u64 slots in iter_counters (== num_workers at spawn time).
    iter_counter_len: usize,
}

// SAFETY: futex_ptrs and iter_counters are MAP_SHARED anonymous pages owned
// exclusively by this handle. Only the parent process accesses them for
// munmap on drop.
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

        // For paired work types, create one pipe per worker pair before forking.
        // pipe_pairs[pair_idx] = (read_fd, write_fd) for the A->B direction,
        // and a second pipe for B->A.
        let mut pipe_pairs: Vec<([i32; 2], [i32; 2])> = Vec::new();
        if needs_pipes {
            for _ in 0..config.num_workers / 2 {
                let mut ab = [0i32; 2]; // A writes, B reads
                let mut ba = [0i32; 2]; // B writes, A reads
                if unsafe { libc::pipe(ab.as_mut_ptr()) } != 0
                    || unsafe { libc::pipe(ba.as_mut_ptr()) } != 0
                {
                    anyhow::bail!("pipe failed: {}", std::io::Error::last_os_error());
                }
                pipe_pairs.push((ab, ba));
            }
        }

        // For FutexPingPong/FutexFanOut, allocate one shared futex word per
        // worker group via MAP_SHARED|MAP_ANONYMOUS so all members of the
        // fork see the same physical page.
        let mut futex_ptrs: Vec<*mut u32> = Vec::new();
        let futex_group_size = config.work_type.worker_group_size().unwrap_or(2);
        if needs_futex {
            for _ in 0..config.num_workers / futex_group_size {
                let ptr = unsafe {
                    libc::mmap(
                        std::ptr::null_mut(),
                        std::mem::size_of::<u32>(),
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_SHARED | libc::MAP_ANONYMOUS,
                        -1,
                        0,
                    )
                };
                if ptr == libc::MAP_FAILED {
                    anyhow::bail!("mmap failed: {}", std::io::Error::last_os_error());
                }
                unsafe { *(ptr as *mut u32) = 0 };
                futex_ptrs.push(ptr as *mut u32);
            }
        }

        // Per-worker iteration counter region (MAP_SHARED). Each worker
        // atomically stores its iteration count to slot [i]. The parent
        // reads all slots via snapshot_iterations().
        let iter_counter_len = config.num_workers;
        let iter_counters = if iter_counter_len > 0 {
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
            ptr as *mut u64
        } else {
            std::ptr::null_mut()
        };

        let mut children = Vec::with_capacity(config.num_workers);

        for i in 0..config.num_workers {
            let affinity = resolve_affinity(&config.affinity, i)?;

            // Determine pipe fds for this worker (PipeIo/CachePipe).
            let worker_pipe_fds: Option<(i32, i32)> = if needs_pipes {
                let pair_idx = i / 2;
                let (ref ab, ref ba) = pipe_pairs[pair_idx];
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
                Some((futex_ptrs[group_idx], is_first))
            } else {
                None
            };

            // Shared iteration counter slot for this worker.
            let iter_slot: *mut u64 = if !iter_counters.is_null() {
                unsafe { iter_counters.add(i) }
            } else {
                std::ptr::null_mut()
            };

            // Create pipe for report and a second pipe for "start" signal
            let mut report_fds = [0i32; 2];
            let mut start_fds = [0i32; 2];
            if unsafe { libc::pipe(report_fds.as_mut_ptr()) } != 0
                || unsafe { libc::pipe(start_fds.as_mut_ptr()) } != 0
            {
                anyhow::bail!("pipe failed: {}", std::io::Error::last_os_error());
            }

            let pid = unsafe { libc::fork() };
            match pid {
                -1 => anyhow::bail!("fork failed: {}", std::io::Error::last_os_error()),
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
                        let (ref ab, ref ba) = pipe_pairs[pair_idx];
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
                        for (j, (ab2, ba2)) in pipe_pairs.iter().enumerate() {
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
                        worker_pipe_fds,
                        worker_futex,
                        iter_slot,
                    );
                    let json = serde_json::to_vec(&report).unwrap_or_default();
                    let mut f = unsafe { std::fs::File::from_raw_fd(report_fds[1]) };
                    let _ = f.write_all(&json);
                    drop(f);
                    unsafe {
                        libc::_exit(0);
                    }
                }
                child_pid => {
                    // Parent: close unused pipe ends
                    unsafe {
                        libc::close(report_fds[1]);
                        libc::close(start_fds[0]);
                    }
                    children.push((child_pid as u32, report_fds[0], start_fds[1]));
                }
            }
        }

        // Parent: close all inter-worker pipe fds (children inherited them).
        for (ab, ba) in &pipe_pairs {
            unsafe {
                libc::close(ab[0]);
                libc::close(ab[1]);
                libc::close(ba[0]);
                libc::close(ba[1]);
            }
        }

        Ok(Self {
            children,
            started: false,
            futex_ptrs,
            iter_counters,
            iter_counter_len,
        })
    }

    /// PIDs of all worker processes.
    pub fn tids(&self) -> Vec<u32> {
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

        // Signal all children to stop
        for &(pid, _, _) in &children {
            let _ = nix::sys::signal::kill(
                nix::unistd::Pid::from_raw(pid as i32),
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

            // Wait for child (WNOHANG first, then SIGKILL if still alive)
            let mut status = 0i32;
            let ret = unsafe { libc::waitpid(pid as i32, &mut status, libc::WNOHANG) };
            if ret == 0 {
                unsafe {
                    libc::kill(pid as i32, libc::SIGKILL);
                    libc::waitpid(pid as i32, &mut status, 0);
                }
            }

            if let Ok(report) = serde_json::from_slice::<WorkerReport>(&buf) {
                reports.push(report);
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

        for &(pid, rfd, wfd) in &self.children {
            let nix_pid = Pid::from_raw(pid as i32);
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
                libc::munmap(ptr as *mut libc::c_void, std::mem::size_of::<u32>());
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
use std::sync::atomic::{AtomicBool, Ordering};

static STOP: AtomicBool = AtomicBool::new(false);

fn worker_main(
    affinity: Option<BTreeSet<usize>>,
    work_type: WorkType,
    sched_policy: SchedPolicy,
    pipe_fds: Option<(i32, i32)>,
    futex: Option<(*mut u32, bool)>,
    iter_slot: *mut u64,
) -> WorkerReport {
    let tid = unsafe { libc::getpid() } as u32;

    if let Some(ref cpus) = affinity {
        let _ = set_thread_affinity(tid, cpus);
    }
    let _ = set_sched_policy(tid, sched_policy);

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
    // Lazily allocated per-worker cache buffer (CachePressure, CacheYield, CachePipe).
    let mut cache_pressure_buf: Option<Vec<u8>> = None;
    // Persistent temp file for IoSync / Phase::Io (opened on first use, removed on exit).
    let mut io_sync_file: Option<(std::fs::File, String)> = None;
    let mut io_seq_file: Option<(std::fs::File, String)> = None;
    // Benchmarking: per-wakeup latency samples (reservoir-sampled) and iteration counter.
    const MAX_WAKE_SAMPLES: usize = 100_000;
    let mut wake_latencies_ns: Vec<u64> = Vec::with_capacity(MAX_WAKE_SAMPLES);
    let mut wake_sample_count: u64 = 0;
    let mut iterations: u64 = 0;
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
                        .join(format!("stt_io_{tid}"))
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
                let _ = f.sync_all();
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
                        &mut wake_latencies_ns,
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
                    &mut wake_latencies_ns,
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
                    libc::syscall(
                        libc::SYS_futex,
                        futex_ptr,
                        libc::FUTEX_WAKE,
                        1, // wake one waiter
                        std::ptr::null::<libc::timespec>(),
                        std::ptr::null::<u32>(),
                        0u32,
                    );
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
                            &mut wake_latencies_ns,
                            &mut wake_sample_count,
                            before_block.elapsed().as_nanos() as u64,
                            MAX_WAKE_SAMPLES,
                        );
                        break;
                    }
                    unsafe {
                        libc::syscall(
                            libc::SYS_futex,
                            futex_ptr,
                            libc::FUTEX_WAIT,
                            partner_val, // expected value
                            &ts as *const libc::timespec,
                            std::ptr::null::<u32>(),
                            0u32,
                        );
                    }
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
                    &mut wake_latencies_ns,
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
                    &mut wake_latencies_ns,
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
                        libc::syscall(
                            libc::SYS_futex,
                            futex_ptr,
                            libc::FUTEX_WAKE,
                            fan_out as i32,
                            std::ptr::null::<libc::timespec>(),
                            std::ptr::null::<u32>(),
                            0u32,
                        );
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
                                &mut wake_latencies_ns,
                                &mut wake_sample_count,
                                before_block.elapsed().as_nanos() as u64,
                                MAX_WAKE_SAMPLES,
                            );
                            break;
                        }
                        unsafe {
                            libc::syscall(
                                libc::SYS_futex,
                                futex_ptr,
                                libc::FUTEX_WAIT,
                                expected,
                                &ts as *const libc::timespec,
                                std::ptr::null::<u32>(),
                                0u32,
                            );
                        }
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
                                &mut wake_latencies_ns,
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
                                    &mut wake_latencies_ns,
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
                                    .join(format!("stt_seq_{tid}"))
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
                                let _ = f.sync_all();
                            }
                            last_iter_time = Instant::now();
                        }
                    }
                }
                iterations += 1;
            }
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
            // stt detects as a scheduler death. In repro mode, keep it
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

    WorkerReport {
        tid,
        work_units,
        cpu_time_ns,
        wall_time_ns,
        runnable_ns: wall_time_ns.saturating_sub(cpu_time_ns),
        migration_count,
        cpus_used,
        migrations,
        max_gap_ms: max_gap_ns / 1_000_000,
        max_gap_cpu,
        max_gap_at_ms: max_gap_at_ns / 1_000_000,
        wake_latencies_ns,
        iterations,
        schedstat_run_delay_ns: ss_delay_end.saturating_sub(ss_delay_start),
        schedstat_ctx_switches: ss_ts_end.saturating_sub(ss_ts_start),
        schedstat_cpu_time_ns: ss_cpu_end.saturating_sub(ss_cpu_start),
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

/// Write 1 byte to partner, poll for response, read, record wake latency.
fn pipe_exchange(
    read_fd: i32,
    write_fd: i32,
    wake_latencies_ns: &mut Vec<u64>,
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
                wake_latencies_ns,
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

fn resolve_affinity(mode: &AffinityMode, _idx: usize) -> Result<Option<BTreeSet<usize>>> {
    match mode {
        AffinityMode::None => Ok(None),
        AffinityMode::Fixed(cpus) => Ok(Some(cpus.clone())),
        AffinityMode::SingleCpu(cpu) => Ok(Some([*cpu].into_iter().collect())),
        AffinityMode::Random { from, count } => {
            use rand::seq::IndexedRandom;
            let pool: Vec<usize> = from.iter().copied().collect();
            let count = (*count).min(pool.len()).max(1);
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

fn thread_cpu_time_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    let ret = unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut ts) };
    if ret != 0 {
        return 0;
    }
    (ts.tv_sec as u64) * 1_000_000_000 + (ts.tv_nsec as u64)
}

fn set_sched_policy(pid: u32, policy: SchedPolicy) -> Result<()> {
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
    if unsafe { libc::sched_setscheduler(pid as i32, pol, &param) } != 0 {
        anyhow::bail!("sched_setscheduler: {}", std::io::Error::last_os_error());
    }
    Ok(())
}

/// Pin a process to the given CPU set via `sched_setaffinity`.
pub fn set_thread_affinity(pid: u32, cpus: &BTreeSet<usize>) -> Result<()> {
    use nix::sched::{CpuSet, sched_setaffinity};
    use nix::unistd::Pid;
    let mut cpu_set = CpuSet::new();
    for &cpu in cpus {
        cpu_set
            .set(cpu)
            .with_context(|| format!("CPU {cpu} out of range"))?;
    }
    sched_setaffinity(Pid::from_raw(pid as i32), &cpu_set)
        .with_context(|| format!("sched_setaffinity pid={pid}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn work_type_name_roundtrip() {
        for &name in WorkType::ALL_NAMES {
            // Sequence has no default from_name; tested separately.
            if name == "Sequence" {
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
        assert_eq!(WorkType::ALL_NAMES.len(), 12);
    }

    #[test]
    fn resolve_affinity_none() {
        let r = resolve_affinity(&AffinityMode::None, 0).unwrap();
        assert!(r.is_none());
    }

    #[test]
    fn resolve_affinity_fixed() {
        let cpus: BTreeSet<usize> = [0, 1, 2].into_iter().collect();
        let r = resolve_affinity(&AffinityMode::Fixed(cpus.clone()), 0).unwrap();
        assert_eq!(r, Some(cpus));
    }

    #[test]
    fn resolve_affinity_single_cpu() {
        let r = resolve_affinity(&AffinityMode::SingleCpu(5), 0).unwrap();
        assert_eq!(r, Some([5].into_iter().collect()));
    }

    #[test]
    fn resolve_affinity_random() {
        let from: BTreeSet<usize> = (0..8).collect();
        let r = resolve_affinity(&AffinityMode::Random { from, count: 3 }, 0).unwrap();
        let cpus = r.unwrap();
        assert_eq!(cpus.len(), 3);
        assert!(cpus.iter().all(|c| *c < 8));
    }

    #[test]
    fn resolve_affinity_random_clamps_count() {
        let from: BTreeSet<usize> = [0, 1].into_iter().collect();
        let r = resolve_affinity(&AffinityMode::Random { from, count: 10 }, 0).unwrap();
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
            runnable_ns: 5_000_000_000,
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
            wake_latencies_ns: vec![1000, 2000],
            iterations: 10,
            schedstat_run_delay_ns: 500_000,
            schedstat_ctx_switches: 20,
            schedstat_cpu_time_ns: 4_000_000_000,
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
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        h.start();
        std::thread::sleep(std::time::Duration::from_millis(200));
        let reports = h.stop_and_collect();
        assert_eq!(reports.len(), 1);
        assert!(reports[0].work_units > 0);
    }

    #[test]
    fn spawn_multiple_workers_distinct_pids() {
        let config = WorkloadConfig {
            num_workers: 4,
            affinity: AffinityMode::None,
            work_type: WorkType::CpuSpin,
            sched_policy: SchedPolicy::Normal,
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        let tids = h.tids();
        assert_eq!(tids.len(), 4);
        let unique: std::collections::HashSet<u32> = tids.iter().copied().collect();
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
        // After drop, children should be dead
        for pid in pids {
            let alive =
                nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid as i32), None).is_ok();
            assert!(!alive, "child {} should be dead after drop", pid);
        }
    }

    #[test]
    fn spawn_io_sync_produces_work() {
        let config = WorkloadConfig {
            num_workers: 1,
            affinity: AffinityMode::None,
            work_type: WorkType::IoSync,
            sched_policy: SchedPolicy::Normal,
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
        let pid = std::process::id();
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
            runnable_ns: 0,
            migration_count: 0,
            cpus_used: BTreeSet::new(),
            migrations: vec![],
            max_gap_ms: 0,
            max_gap_cpu: 0,
            max_gap_at_ms: 0,
            wake_latencies_ns: vec![],
            iterations: 0,
            schedstat_run_delay_ns: 0,
            schedstat_ctx_switches: 0,
            schedstat_cpu_time_ns: 0,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: WorkerReport = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.tid, 0);
        assert!(r2.cpus_used.is_empty());
        assert!(r2.migrations.is_empty());

        // Max u64 values
        let r = WorkerReport {
            tid: u32::MAX,
            work_units: u64::MAX,
            cpu_time_ns: u64::MAX,
            wall_time_ns: u64::MAX,
            runnable_ns: u64::MAX,
            migration_count: u64::MAX,
            cpus_used: [0, usize::MAX].into_iter().collect(),
            migrations: vec![],
            max_gap_ms: u64::MAX,
            max_gap_cpu: usize::MAX,
            max_gap_at_ms: u64::MAX,
            wake_latencies_ns: vec![],
            iterations: u64::MAX,
            schedstat_run_delay_ns: u64::MAX,
            schedstat_ctx_switches: u64::MAX,
            schedstat_cpu_time_ns: u64::MAX,
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: WorkerReport = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.work_units, u64::MAX);
        assert_eq!(r2.tid, u32::MAX);
    }

    #[test]
    fn io_sync_cleans_up_temp_file() {
        let config = WorkloadConfig {
            num_workers: 1,
            affinity: AffinityMode::None,
            work_type: WorkType::IoSync,
            sched_policy: SchedPolicy::Normal,
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        h.start();
        std::thread::sleep(std::time::Duration::from_millis(200));
        let reports = h.stop_and_collect();
        assert_eq!(reports.len(), 1);
        let tid = reports[0].tid;
        let path = std::env::temp_dir()
            .join(format!("stt_io_{tid}"))
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
        let pid = std::process::id();
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
        let pid = std::process::id();
        let result = set_sched_policy(pid, SchedPolicy::Fifo(1));
        // SCHED_FIFO requires CAP_SYS_NICE — fails without privileges.
        assert!(
            result.is_err(),
            "SCHED_FIFO should fail without CAP_SYS_NICE"
        );
    }

    #[test]
    fn set_sched_policy_rr_returns_result() {
        let pid = std::process::id();
        let result = set_sched_policy(pid, SchedPolicy::RoundRobin(1));
        // SCHED_RR requires CAP_SYS_NICE — fails without privileges.
        assert!(result.is_err(), "SCHED_RR should fail without CAP_SYS_NICE");
    }

    #[test]
    fn resolve_affinity_random_single_cpu_pool() {
        let from: BTreeSet<usize> = [7].into_iter().collect();
        let r = resolve_affinity(&AffinityMode::Random { from, count: 1 }, 0).unwrap();
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
    fn restore_normal(pid: u32) {
        let param = libc::sched_param { sched_priority: 0 };
        unsafe { libc::sched_setscheduler(pid as i32, libc::SCHED_OTHER, &param) };
    }

    #[test]
    fn set_sched_policy_batch_returns_valid_result() {
        let pid = std::process::id();
        let result = set_sched_policy(pid, SchedPolicy::Batch);
        // SCHED_BATCH may fail under sched_ext or without CAP_SYS_NICE.
        match result {
            Ok(()) => {
                let pol = unsafe { libc::sched_getscheduler(pid as i32) };
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
        let pid = std::process::id();
        let result = set_sched_policy(pid, SchedPolicy::Idle);
        // SCHED_IDLE may fail under sched_ext or without CAP_SYS_NICE.
        match result {
            Ok(()) => {
                let pol = unsafe { libc::sched_getscheduler(pid as i32) };
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
            runnable_ns: 1000,
            migration_count: 3,
            cpus_used: [0, 5].into_iter().collect(),
            migrations: vec![],
            max_gap_ms: 77,
            max_gap_cpu: 5,
            max_gap_at_ms: 500,
            wake_latencies_ns: vec![],
            iterations: 0,
            schedstat_run_delay_ns: 0,
            schedstat_ctx_switches: 0,
            schedstat_cpu_time_ns: 0,
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
    fn worker_report_runnable_ns_calculation() {
        // runnable_ns = wall_time_ns - cpu_time_ns
        let r = WorkerReport {
            tid: 1,
            work_units: 100,
            cpu_time_ns: 3_000_000_000,
            wall_time_ns: 5_000_000_000,
            runnable_ns: 2_000_000_000,
            migration_count: 0,
            cpus_used: [0].into_iter().collect(),
            migrations: vec![],
            max_gap_ms: 0,
            max_gap_cpu: 0,
            max_gap_at_ms: 0,
            wake_latencies_ns: vec![],
            iterations: 0,
            schedstat_run_delay_ns: 0,
            schedstat_ctx_switches: 0,
            schedstat_cpu_time_ns: 0,
        };
        assert_eq!(r.runnable_ns, r.wall_time_ns - r.cpu_time_ns);
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
    fn resolve_affinity_random_zero_count() {
        let from: BTreeSet<usize> = (0..4).collect();
        let r = resolve_affinity(&AffinityMode::Random { from, count: 0 }, 0).unwrap();
        // count is clamped to max(1), so should get 1 CPU
        assert_eq!(r.unwrap().len(), 1);
    }

    // -- spawn and collect edge cases --

    #[test]
    fn spawn_single_worker_reports_cpus() {
        let config = WorkloadConfig {
            num_workers: 1,
            affinity: AffinityMode::None,
            work_type: WorkType::CpuSpin,
            sched_policy: SchedPolicy::Normal,
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
        };
        let mut h = WorkloadHandle::spawn(&config).unwrap();
        h.start();
        std::thread::sleep(std::time::Duration::from_millis(500));
        let reports = h.stop_and_collect();
        // At least one receiver should have wake latency samples.
        let has_latencies = reports.iter().any(|r| !r.wake_latencies_ns.is_empty());
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

    #[test]
    fn work_type_futex_fanout_needs_shared_mem() {
        let wt = WorkType::FutexFanOut {
            fan_out: 4,
            spin_iters: 1024,
        };
        assert!(wt.needs_shared_mem());
    }

    #[test]
    fn worker_group_size_paired_types() {
        assert_eq!(
            WorkType::PipeIo { burst_iters: 1024 }.worker_group_size(),
            Some(2)
        );
        assert_eq!(
            WorkType::FutexPingPong { spin_iters: 1024 }.worker_group_size(),
            Some(2)
        );
        assert_eq!(
            WorkType::CachePipe {
                size_kb: 32,
                burst_iters: 1024
            }
            .worker_group_size(),
            Some(2)
        );
    }

    #[test]
    fn worker_group_size_ungrouped_types() {
        assert_eq!(WorkType::CpuSpin.worker_group_size(), None);
        assert_eq!(WorkType::YieldHeavy.worker_group_size(), None);
        assert_eq!(WorkType::Mixed.worker_group_size(), None);
        assert_eq!(WorkType::IoSync.worker_group_size(), None);
        assert_eq!(
            WorkType::Bursty {
                burst_ms: 50,
                sleep_ms: 100
            }
            .worker_group_size(),
            None
        );
        assert_eq!(
            WorkType::CachePressure {
                size_kb: 32,
                stride: 64
            }
            .worker_group_size(),
            None
        );
        assert_eq!(
            WorkType::CacheYield {
                size_kb: 32,
                stride: 64
            }
            .worker_group_size(),
            None
        );
    }

    // -- WorkType::needs_cache_buf tests --

    #[test]
    fn work_type_needs_cache_buf_cache_types() {
        assert!(
            WorkType::CachePressure {
                size_kb: 32,
                stride: 64
            }
            .needs_cache_buf()
        );
        assert!(
            WorkType::CacheYield {
                size_kb: 32,
                stride: 64
            }
            .needs_cache_buf()
        );
        assert!(
            WorkType::CachePipe {
                size_kb: 32,
                burst_iters: 1024
            }
            .needs_cache_buf()
        );
    }

    #[test]
    fn work_type_needs_cache_buf_non_cache_types() {
        assert!(!WorkType::CpuSpin.needs_cache_buf());
        assert!(!WorkType::YieldHeavy.needs_cache_buf());
        assert!(!WorkType::Mixed.needs_cache_buf());
        assert!(!WorkType::IoSync.needs_cache_buf());
        assert!(
            !WorkType::Bursty {
                burst_ms: 50,
                sleep_ms: 100
            }
            .needs_cache_buf()
        );
        assert!(!WorkType::PipeIo { burst_iters: 1024 }.needs_cache_buf());
        assert!(!WorkType::FutexPingPong { spin_iters: 1024 }.needs_cache_buf());
        assert!(
            !WorkType::FutexFanOut {
                fan_out: 4,
                spin_iters: 1024
            }
            .needs_cache_buf()
        );
    }

    // -- resolve_work_type tests --

    #[test]
    fn resolve_work_type_not_swappable_returns_base() {
        let base = WorkType::CpuSpin;
        let override_wt = WorkType::YieldHeavy;
        let result = resolve_work_type(&base, Some(&override_wt), false, 4);
        assert!(matches!(result, WorkType::CpuSpin));
    }

    #[test]
    fn resolve_work_type_swappable_no_override_returns_base() {
        let base = WorkType::CpuSpin;
        let result = resolve_work_type(&base, None, true, 4);
        assert!(matches!(result, WorkType::CpuSpin));
    }

    #[test]
    fn resolve_work_type_swappable_ungrouped_override() {
        let base = WorkType::CpuSpin;
        let override_wt = WorkType::YieldHeavy;
        let result = resolve_work_type(&base, Some(&override_wt), true, 4);
        assert!(matches!(result, WorkType::YieldHeavy));
    }

    #[test]
    fn resolve_work_type_swappable_grouped_override_compatible() {
        let base = WorkType::CpuSpin;
        let override_wt = WorkType::PipeIo { burst_iters: 1024 };
        // num_workers=4 is divisible by group_size=2
        let result = resolve_work_type(&base, Some(&override_wt), true, 4);
        assert!(matches!(result, WorkType::PipeIo { .. }));
    }

    #[test]
    fn resolve_work_type_swappable_grouped_override_incompatible() {
        let base = WorkType::CpuSpin;
        let override_wt = WorkType::PipeIo { burst_iters: 1024 };
        // num_workers=3 is not divisible by group_size=2, falls back to base
        let result = resolve_work_type(&base, Some(&override_wt), true, 3);
        assert!(matches!(result, WorkType::CpuSpin));
    }

    #[test]
    fn resolve_work_type_swappable_fanout_compatible() {
        let base = WorkType::CpuSpin;
        let override_wt = WorkType::FutexFanOut {
            fan_out: 4,
            spin_iters: 1024,
        };
        // num_workers=10 is divisible by group_size=5
        let result = resolve_work_type(&base, Some(&override_wt), true, 10);
        assert!(matches!(result, WorkType::FutexFanOut { .. }));
    }

    #[test]
    fn resolve_work_type_swappable_fanout_incompatible() {
        let base = WorkType::CpuSpin;
        let override_wt = WorkType::FutexFanOut {
            fan_out: 4,
            spin_iters: 1024,
        };
        // num_workers=7 is not divisible by group_size=5, falls back to base
        let result = resolve_work_type(&base, Some(&override_wt), true, 7);
        assert!(matches!(result, WorkType::CpuSpin));
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
}
