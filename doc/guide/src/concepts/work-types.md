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
}
```

## Variants

**`CpuSpin`** -- tight spin loop with `spin_loop()` hints. 1024
iterations per check. Pure CPU-bound workload.

**`YieldHeavy`** -- `thread::yield_now()` on every iteration. Exercises
scheduler wake/sleep paths.

**`Mixed`** -- 1024 spin iterations then yield. Combines CPU and
voluntary preemption.

**`IoSync`** -- writes 64 KB to a temp file and calls `fsync()`.
Exercises I/O scheduling.

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

## Preset names

`WorkType::from_preset()` resolves snake_case preset names to
`WorkType` values with default parameters:

| Preset | Resolves to |
|---|---|
| `cpu_spin` | `CpuSpin` |
| `mixed` | `Mixed` |
| `bursty` | `Bursty { burst_ms: 50, sleep_ms: 100 }` |
| `yield` | `YieldHeavy` |
| `io` | `IoSync` |
| `pipe` | `PipeIo { burst_iters: 1024 }` |
| `cache_l1` | `CachePressure { size_kb: 32, stride: 64 }` |
| `cache_yield` | `CacheYield { size_kb: 32, stride: 64 }` |
| `cache_pipe` | `CachePipe { size_kb: 32, burst_iters: 1024 }` |
| `futex` | `FutexPingPong { spin_iters: 1024 }` |
| `fanout` | `FutexFanOut { fan_out: 4, spin_iters: 1024 }` |

`WorkType::PRESET_NAMES` lists all available preset names.
`WorkType::from_name()` uses PascalCase names matching the enum
variants; `from_preset()` uses the snake_case aliases above.

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

Overrides to paired work types (`PipeIo`, `FutexPingPong`,
`CachePipe`) are skipped when a cgroup has an odd number of workers.

Ops-based scenarios have a separate override mechanism via
`CgroupDef.swappable`. See [Ops and Steps](ops.md#work-type-overrides-and-swappable).
