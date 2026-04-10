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
    Sequence { first: Phase, rest: Vec<Phase> },
}
```

## Variants

**`CpuSpin`** -- tight spin loop with `spin_loop()` hints. 1024
iterations per check. Pure CPU-bound workload.

**`YieldHeavy`** -- `thread::yield_now()` on every iteration. Exercises
scheduler wake/sleep paths.

**`Mixed`** -- 1024 spin iterations then yield. Combines CPU and
voluntary preemption.

**`IoSync`** -- writes 64 KB to a temp file then sleeps 100 us to
simulate I/O completion latency. On tmpfs (which stt VMs use), fsync
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
wake_affine placement after voluntary preemption.

**`CachePipe`** -- cache pressure burst then 1-byte pipe exchange with
a partner worker. Combines cache-hot working set with cross-CPU wake
placement. Requires even `num_workers`.

**`FutexFanOut`** -- 1:N fan-out wake pattern (schbench-style). One
messenger per group does `spin_iters` of CPU work then wakes `fan_out`
receivers via `FUTEX_WAKE`. Receivers measure wake-to-run latency.
Requires `num_workers` divisible by `fan_out + 1`.

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

## Grouped work types

`PipeIo`, `FutexPingPong`, and `CachePipe` require `num_workers`
divisible by 2 (paired). `FutexFanOut` requires `num_workers` divisible
by `fan_out + 1` (1 messenger + N receivers per group).
`WorkType::worker_group_size()` returns the group size for these
variants, or `None` for ungrouped types. `PipeIo` and `CachePipe` use
pipes; `FutexPingPong` and `FutexFanOut` use shared mmap pages with
futex wait/wake.

## Default values

`WorkType::from_name()` uses these defaults:
- `Bursty`: `burst_ms=50`, `sleep_ms=100`
- `PipeIo`: `burst_iters=1024`
- `FutexPingPong`: `spin_iters=1024`
- `CachePressure`: `size_kb=32`, `stride=64`
- `CacheYield`: `size_kb=32`, `stride=64`
- `CachePipe`: `size_kb=32`, `burst_iters=1024`
- `FutexFanOut`: `fan_out=4`, `spin_iters=1024`

## String lookup

`WorkType::from_name()` accepts PascalCase names matching the enum
variants (e.g. `"CpuSpin"`, `"FutexPingPong"`).
`WorkType::ALL_NAMES` lists every variant name. `WorkType::name()`
returns the PascalCase name for a given value.

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
}
```

`Default`: 1 worker, no affinity, CpuSpin, Normal policy.

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
`CachePipe`, `FutexFanOut`) are skipped when `num_workers` is not
divisible by the work type's group size.

Ops-based scenarios have a separate override mechanism via
`CgroupDef.swappable`. See [Ops and Steps](ops.md#work-type-overrides-and-swappable).
