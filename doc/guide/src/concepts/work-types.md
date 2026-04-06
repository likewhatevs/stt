# Work Types

`WorkType` controls what each worker process does during a scenario.

```rust
pub enum WorkType {
    CpuSpin,
    YieldHeavy,
    Mixed,
    IoSync,
    Bursty { burst_ms: u64, sleep_ms: u64 },
    PipeIo { burst_iters: u64 },
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

## Default values

`WorkType::from_name()` uses these defaults:
- `Bursty`: `burst_ms=50`, `sleep_ms=100`
- `PipeIo`: `burst_iters=1024`

## Scheduling policies

Workers can run under different Linux scheduling policies:

```rust
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

The `--work-type` CLI flag overrides the default `CpuSpin` work type
for all scenarios that use it. Scenarios with non-`CpuSpin` work types
are not overridden.

`PipeIo` overrides are skipped when a cgroup has an odd number of
workers (PipeIo requires pairs).

Ops-based scenarios have a separate override mechanism via
`CgroupDef.swappable`. See [Ops and Steps](ops.md#work-type-overrides-and-swappable).
