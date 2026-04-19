# Worker Processes

Workers are the processes that generate load for scenarios. They run
inside the VM, each in its own cgroup.

## Fork, not threads

Workers are `fork()`ed processes. Cgroups operate on PIDs, so each
worker must be a separate process to be independently placed in a
cgroup.

## Two-phase start

Workers wait on a pipe for a "start" signal after fork:

1. Parent forks the worker.
2. Worker installs SIGUSR1 handler, then blocks on pipe read.
3. Parent moves the worker to its target cgroup.
4. Parent writes to the pipe, signaling the worker to start.

This ensures workers run inside their target cgroup from the first
instruction of their workload.

### Custom work types

`WorkType::Custom` workers follow the same two-phase start (fork,
cgroup placement, start signal), and the framework applies affinity
and scheduling policy before handing control to the user function.
After setup, the `run` function pointer takes over entirely --
the framework work loop is bypassed.

## Stop protocol

Workers install a SIGUSR1 handler that sets an atomic `STOP` flag. The
main work loop checks this flag each iteration. On stop:

1. Parent sends SIGUSR1 to all workers.
2. Workers exit their work loop.
3. Workers serialize their `WorkerReport` to a pipe.
4. Parent reads reports and waits for child exit.

## Telemetry

Each worker produces a `WorkerReport`:

```rust,ignore
pub struct WorkerReport {
    pub tid: i32,
    pub work_units: u64,
    pub cpu_time_ns: u64,
    pub wall_time_ns: u64,
    pub off_cpu_ns: u64,
    pub migration_count: u64,
    pub cpus_used: BTreeSet<usize>,
    pub migrations: Vec<Migration>,
    pub max_gap_ms: u64,
    pub max_gap_cpu: usize,
    pub max_gap_at_ms: u64,
    pub wake_latencies_ns: Vec<u64>,
    pub iterations: u64,
    pub schedstat_run_delay_ns: u64,
    pub schedstat_ctx_switches: u64,
    pub schedstat_cpu_time_ns: u64,
    pub numa_pages: BTreeMap<usize, u64>,
    pub vmstat_numa_pages_migrated: u64,
}
```

- `off_cpu_ns = wall_time_ns - cpu_time_ns`
- Migrations are tracked every 1024 work units
  (`work_units.is_multiple_of(1024)`). How often this fires depends
  on the work type: every outer iteration for CpuSpin/Mixed (1024
  units each), every 1024th yield for YieldHeavy (1 unit each),
  every 64th write-then-sleep cycle for IoSync (16 units each), after each
  inner spin batch for Bursty (1024 units per batch), and after each
  burst for PipeIo (`burst_iters` units per batch, 1024 by default)
- Scheduling gaps are the longest intervals between iterations

### Benchmarking fields

Workers collect two categories of timing data:

**Per-wakeup latency** (`wake_latencies_ns`): timestamp-based samples
recorded around blocking operations. Populated for work types with a
blocking step: Bursty (sleep), PipeIo (pipe read), FutexPingPong
(futex wait), FutexFanOut (futex wait, receivers only), SchBench
(futex wait, workers only — measured as `CLOCK_MONOTONIC` delta from
messenger's shared timestamp), CacheYield (yield), CachePipe (pipe
read), IoSync (sleep), NiceSweep (yield), AffinityChurn (yield), and
Sequence when its phases include Sleep, Yield, or Io. Each sample is
in nanoseconds; most work types use `Instant::elapsed()` across the
blocking call, while SchBench uses `clock_gettime(CLOCK_MONOTONIC)`
to measure against the messenger's pre-wake timestamp.

**schedstat deltas**: read from `/proc/self/schedstat` at work-loop
start and end. Three fields:
- `schedstat_cpu_time_ns` -- delta of field 1 (on-CPU time)
- `schedstat_run_delay_ns` -- delta of field 2 (time spent waiting
  for a CPU)
- `schedstat_ctx_switches` -- delta of field 3 (timeslice count)

`iterations` counts outer-loop iterations.

### NUMA fields

**`numa_pages`**: per-NUMA-node page counts parsed from
`/proc/self/numa_maps` after the workload completes. Keyed by node ID.
Empty when numa_maps is unavailable.

**`vmstat_numa_pages_migrated`**: delta of the `numa_pages_migrated`
counter from `/proc/vmstat` between pre- and post-workload snapshots.
Measures cross-node page migrations during the test.

These fields feed the NUMA [verification
checks](../concepts/verification.md#numa-checks).

Custom workers produce their own `WorkerReport`. The framework does
not populate any telemetry fields for Custom -- migration tracking,
gap detection, schedstat deltas, NUMA page counts, and iteration
counters are only present if the user's `run` function fills them.

## Work-conservation watchdog

Workers send SIGUSR2 to the scheduler when stuck > 2 seconds. The
default POSIX disposition terminates the scheduler process, which ktstr
detects as a scheduler death and captures the sched_ext dump from
dmesg.

In repro mode, the watchdog is disabled to keep the scheduler alive
for BPF probe assertions. The watchdog does not fire for Custom
workers because they bypass the framework work loop.

## RAII cleanup

`WorkloadHandle` implements `Drop`: it sends SIGKILL to all child
processes and waits for them. This prevents orphaned worker processes
on error paths.
