# Work Types

`WorkType` controls what each worker process does during a scenario.

```rust,ignore
pub enum WorkType {
    CpuSpin,
    YieldHeavy,
    Mixed,
    IoSync,
    Bursty { burst_ms: u64, sleep_ms: u64 },
    PipeIo { burst_iters: u64 },
    FutexPingPong { spin_iters: u64 },
    CachePressure { size_kb: usize, stride: usize },
    CacheYield { size_kb: usize, stride: usize },
    CachePipe { size_kb: usize, burst_iters: u64 },
    FutexFanOut { fan_out: usize, spin_iters: u64 },
    FanOutCompute { fan_out: usize, cache_footprint_kb: usize, operations: usize, sleep_usec: u64 },
    Sequence { first: Phase, rest: Vec<Phase> },
    ForkExit,
    NiceSweep,
    AffinityChurn { spin_iters: u64 },
    PolicyChurn { spin_iters: u64 },
    PageFaultChurn { region_kb: usize, touches_per_cycle: usize, spin_iters: u64 },
    MutexContention { contenders: usize, hold_iters: u64, work_iters: u64 },
    Custom { name: &'static str, run: fn(&AtomicBool) -> WorkerReport },
}
```

Parameterized variants have convenience constructors:
`WorkType::bursty(50, 100)`, `WorkType::pipe_io(1024)`,
`WorkType::futex_ping_pong(1024)`, `WorkType::cache_pressure(32, 64)`,
`WorkType::cache_yield(32, 64)`, `WorkType::cache_pipe(32, 1024)`,
`WorkType::futex_fan_out(4, 1024)`,
`WorkType::fan_out_compute(4, 256, 5, 100)`,
`WorkType::affinity_churn(1024)`, `WorkType::policy_churn(1024)`,
`WorkType::page_fault_churn(4096, 256, 64)`,
`WorkType::mutex_contention(4, 256, 1024)`,
`WorkType::custom("my_work", my_fn)`.

## Choosing a work type

| Scheduler behavior to test | Recommended work type |
|---|---|
| Basic load balancing / fairness | `CpuSpin` (default) |
| Wake placement / sleep-wake cycles | `YieldHeavy`, `FutexPingPong` |
| CPU borrowing / idle balance | `Bursty` |
| Cross-CPU wake latency | `PipeIo`, `CachePipe` |
| Cache-aware scheduling | `CachePressure`, `CacheYield` |
| Cache-aware fan-out wake latency | `FanOutCompute` |
| Fan-out wake storms | `FutexFanOut` |
| Mixed real-world patterns | `Sequence` |
| Task creation/destruction pressure | `ForkExit` |
| Priority reweighting / nice dynamics | `NiceSweep` |
| Rapid CPU migration / affinity churn | `AffinityChurn` |
| Scheduling class transitions | `PolicyChurn` |
| Page fault / TLB pressure | `PageFaultChurn` |
| Lock contention / convoy effect | `MutexContention` |
| Arbitrary user-defined workload | `Custom` |

## Variants

**`CpuSpin`** -- tight spin loop with `spin_loop()` hints. 1024
iterations per check. Pure CPU-bound workload.

**`YieldHeavy`** -- `thread::yield_now()` on every iteration. Exercises
scheduler wake/sleep paths.

**`Mixed`** -- 1024 spin iterations then yield. Combines CPU and
voluntary preemption.

**`IoSync`** -- writes 64 KB to a temp file then sleeps 100 us to
simulate I/O completion latency. On tmpfs (which ktstr VMs use), fsync
is a kernel no-op and writes go to page cache, so the sleep provides
the blocking that real disk I/O would cause. Exercises scheduler
dequeue/requeue paths and page allocator pressure.

**`Bursty`** -- CPU burst for `burst_ms`, then sleep for `sleep_ms`.
Frees CPUs during sleep, exercising CPU borrowing.

**`PipeIo`** -- CPU burst then 1-byte pipe exchange with a partner
worker. Workers are paired: (0,1), (2,3), etc. Sleep duration depends
on partner scheduling, exercising cross-CPU wake placement. Requires
even `num_workers`.

**`FutexPingPong`** -- paired futex wait/wake between partner workers.
Each iteration does `spin_iters` of CPU work then wakes the partner
and waits on a shared futex word. Exercises the non-WF_SYNC wake path.
Requires even `num_workers`.

**`CachePressure`** -- strided read-modify-write over a buffer sized
to pressure the L1 cache. Each worker allocates its own buffer
post-fork. `size_kb` controls buffer size, `stride` controls the byte
step between accesses.

**`CacheYield`** -- cache pressure followed by `sched_yield()`. Tests
scheduler re-placement after voluntary yield with a cache-hot working set.

**`CachePipe`** -- cache pressure burst then 1-byte pipe exchange with
a partner worker. Combines cache-hot working set with cross-CPU wake
placement. Requires even `num_workers`.

**`FutexFanOut`** -- 1:N fan-out wake pattern without cache pressure.
One messenger per group does `spin_iters` of CPU spin work then wakes
`fan_out` receivers via `FUTEX_WAKE`. Receivers measure wake-to-run
latency. For cache-aware fan-out with matrix multiply work, see
`FanOutCompute`. Requires `num_workers` divisible by `fan_out + 1`.

**`FanOutCompute`** -- messenger/worker fan-out with compute work. One
messenger per group stamps a `CLOCK_MONOTONIC` timestamp then wakes
`fan_out` workers via `FUTEX_WAKE`. Workers measure wake-to-run latency
(time from messenger's timestamp to worker getting the CPU), sleep for
`sleep_usec` microseconds (simulating think time), then do `operations`
iterations of naive matrix multiply over a `cache_footprint_kb`-sized
working set (three square matrices of u64, O(n^3)). Requires
`num_workers` divisible by `fan_out + 1`.

**`Sequence`** -- compound work pattern: loop through phases in order,
repeat. Each phase runs for its specified duration before the next
starts. Phases are defined via the `Phase` enum:

- `Phase::Spin(Duration)` -- CPU spin for the given duration.
- `Phase::Sleep(Duration)` -- `thread::sleep` for the given duration.
- `Phase::Yield(Duration)` -- repeated `sched_yield` for the given duration.
- `Phase::Io(Duration)` -- simulated I/O (write 64 KB + 100 us sleep) for the given duration.

`Sequence` cannot be constructed via `WorkType::from_name()` because
it requires explicit phase definitions. Build it directly:

```rust,ignore
WorkType::Sequence {
    first: Phase::Spin(Duration::from_millis(100)),
    rest: vec![
        Phase::Sleep(Duration::from_millis(50)),
        Phase::Yield(Duration::from_millis(20)),
    ],
}
```

**`ForkExit`** -- rapid fork+`_exit` cycling. Each iteration forks a
child that immediately calls `_exit(0)`. The parent `waitpid`s then
repeats. Exercises `wake_up_new_task`, `do_exit`, and
`wait_task_zombie`.

**`NiceSweep`** -- cycles the worker's nice level from -20 to 19
across iterations. Each iteration: 512-iteration spin burst,
`setpriority(PRIO_PROCESS, 0, nice_val)`, then `sched_yield`. Exercises
`reweight_task` and dynamic priority reweighting. Skips negative nice values
when `CAP_SYS_NICE` is absent. Resets nice to 0 before exit. Records
per-yield wake latency.

**`AffinityChurn`** -- rapid self-directed CPU affinity changes. Each
iteration: `spin_iters` spin burst, `sched_setaffinity` to a random CPU
from the effective cpuset, then `sched_yield`. Exercises
`affine_move_task` and `migration_cpu_stop`. Records per-yield wake
latency.

**`PolicyChurn`** -- cycles through scheduling policies each iteration.
Each iteration: `spin_iters` spin burst, `sched_setscheduler` to the
next policy in the sequence, then `sched_yield`. Cycles through
`SCHED_OTHER`, `SCHED_BATCH`, `SCHED_IDLE` (and `SCHED_FIFO`/`SCHED_RR`
with priority 1 when `CAP_SYS_NICE` is available). Exercises
`__sched_setscheduler` and scheduling class transitions. Resets to
`SCHED_OTHER` before exit. Records per-yield wake latency.

**`PageFaultChurn`** -- rapid page fault cycling. Workers mmap a
`region_kb` KB region with `MADV_NOHUGEPAGE` (forcing 4 KB pages),
touch `touches_per_cycle` random pages via write faults through
`do_anonymous_page`, then `MADV_DONTNEED` to zap PTEs and repeat.
`spin_iters` iterations of CPU work separate cycles. Exercises
the page allocator, TLB pressure on migration, and rapid user/kernel
transitions. Uses xorshift64 PRNG for random page selection (seeded
from the process ID).

**`MutexContention`** -- N-way futex mutex contention. `contenders`
workers per group contend on a shared `AtomicU32` via CAS acquire
(`FUTEX_WAIT` on failure). Loop: `spin_burst(work_iters)` then CAS
acquire, `spin_burst(hold_iters)` in the critical section, then
store 0 + `FUTEX_WAKE(1)` to release. Exercises convoy effect,
lock-holder preemption cascading stalls, and futex wait/wake
contention paths. Requires `num_workers` divisible by `contenders`.

**`Custom`** -- user-supplied work function. The `run` function pointer
receives a reference to the stop flag (`&AtomicBool`, set by SIGUSR1)
and returns a `WorkerReport` when the flag becomes `true`. The
framework handles fork, cgroup placement, affinity, scheduling policy,
and signal setup; the user function owns the work loop and all
`WorkerReport` field population. Framework telemetry (migration
tracking, gap detection, schedstat deltas, iteration counter updates)
is not provided -- the user function is responsible for any telemetry
it needs.

Function pointers (`fn(&AtomicBool) -> WorkerReport`) are fork-safe
because they carry no captured state across the fork boundary. Closures
are not supported. Cannot be constructed via `WorkType::from_name()`.

```rust,ignore
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use ktstr::workload::{WorkType, WorkerReport};

fn my_workload(stop: &AtomicBool) -> WorkerReport {
    let tid: libc::pid_t = unsafe { libc::getpid() };
    let start = std::time::Instant::now();
    let mut work_units = 0u64;
    while !stop.load(Ordering::Relaxed) {
        // ... custom work ...
        work_units += 1;
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
        schedstat_run_count: 0,
        schedstat_cpu_time_ns: 0,
        numa_pages: BTreeMap::new(),
        vmstat_numa_pages_migrated: 0,
    }
}

let wt = WorkType::custom("my_workload", my_workload);
```

## Grouped work types

`PipeIo`, `FutexPingPong`, and `CachePipe` require `num_workers`
divisible by 2 (paired). `FutexFanOut` and `FanOutCompute` require
`num_workers` divisible by `fan_out + 1` (1 messenger + N receivers per
group). `MutexContention` requires `num_workers` divisible by
`contenders`. `WorkType::worker_group_size()` returns the group size
for these variants, or `None` for ungrouped types. `PipeIo` and
`CachePipe` use pipes; `FutexPingPong`, `FutexFanOut`, `FanOutCompute`,
and `MutexContention` use shared mmap pages with futex wait/wake.

## Default values

`WorkType::from_name()` uses these defaults:
- `Bursty`: `burst_ms=50`, `sleep_ms=100`
- `PipeIo`: `burst_iters=1024`
- `FutexPingPong`: `spin_iters=1024`
- `CachePressure`: `size_kb=32`, `stride=64`
- `CacheYield`: `size_kb=32`, `stride=64`
- `CachePipe`: `size_kb=32`, `burst_iters=1024`
- `FutexFanOut`: `fan_out=4`, `spin_iters=1024`
- `FanOutCompute`: `fan_out=4`, `cache_footprint_kb=256`, `operations=5`, `sleep_usec=100`
- `AffinityChurn`: `spin_iters=1024`
- `PolicyChurn`: `spin_iters=1024`
- `PageFaultChurn`: `region_kb=4096`, `touches_per_cycle=256`, `spin_iters=64`
- `MutexContention`: `contenders=4`, `hold_iters=256`, `work_iters=1024`

## String lookup

`WorkType::from_name()` accepts PascalCase names matching the enum
variants (e.g. `"CpuSpin"`, `"FutexPingPong"`). `Sequence` and `Custom`
return `None` because they require explicit construction parameters.
`WorkType::ALL_NAMES` lists every variant name. `WorkType::name()`
returns the PascalCase name for a given value; for `Custom`, it returns
the user-provided `name` field.

## WorkloadConfig

`WorkloadConfig` is the low-level struct passed to
`WorkloadHandle::spawn()`. `CgroupDef` builds one internally; use
`WorkloadConfig` directly when calling `setup_cgroups()` or
`WorkloadHandle::spawn()` in custom scenarios.

```rust,ignore
pub struct WorkloadConfig {
    pub num_workers: usize,       // Number of worker processes to fork
    pub affinity: AffinityMode,   // CPU affinity mode (None, Fixed, Random, SingleCpu)
    pub work_type: WorkType,      // What each worker does
    pub sched_policy: SchedPolicy, // Linux scheduling policy
    pub mem_policy: MemPolicy,    // NUMA memory placement policy
    pub mpol_flags: MpolFlags,    // Optional mode flags for set_mempolicy(2)
}
```

`Default`: 1 worker, no affinity, CpuSpin, Normal policy, Default
mem_policy, no mpol_flags.

See [MemPolicy](mem-policy.md) for the NUMA memory placement API.

## Scheduling policies

Workers can run under different Linux scheduling policies:

```rust,ignore
pub enum SchedPolicy {
    Normal,
    Batch,
    Idle,
    Fifo(u32),      // priority 1-99
    RoundRobin(u32), // priority 1-99
}
```

`Fifo` and `RoundRobin` require `CAP_SYS_NICE`.

## Overriding work types

The work type override (configured via gauntlet or
`Ctx.work_type_override`) replaces the default `CpuSpin` work type
for all scenarios that use it. Scenarios with non-`CpuSpin` work types
are not overridden.

Overrides to grouped work types (`PipeIo`, `FutexPingPong`,
`CachePipe`, `FutexFanOut`, `FanOutCompute`, `MutexContention`) are skipped
when `num_workers` is not divisible by the work type's group size.

Ops-based scenarios have a separate override mechanism via
`CgroupDef.swappable`. See [Ops and Steps](ops.md#work-type-overrides-and-swappable).
