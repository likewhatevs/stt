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
    pub tid: u32,
    pub work_units: u64,
    pub cpu_time_ns: u64,
    pub wall_time_ns: u64,
    pub runnable_ns: u64,
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
}
```

- `runnable_ns = wall_time_ns - cpu_time_ns`
- Migrations are tracked every 1024 work units
  (`work_units.is_multiple_of(1024)`). How often this fires depends
  on the work type: every outer iteration for CpuSpin/Mixed (1024
  units each), every 1024th yield for YieldHeavy (1 unit each),
  every 64th fsync cycle for IoSync (16 units each), after each
  inner spin batch for Bursty (1024 units per batch), and after each
  burst for PipeIo (`burst_iters` units per batch, 1024 by default)
- Scheduling gaps are the longest intervals between iterations

### Benchmarking fields

Workers collect two categories of timing data:

**Per-wakeup latency** (`wake_latencies_ns`): timestamp-based samples
recorded around blocking operations. Populated for work types with a
blocking step: Bursty (sleep), PipeIo (pipe read), FutexPingPong
(futex wait), FutexFanOut (futex wait, receivers only), CacheYield
(yield), CachePipe (pipe read). Each sample
is `Instant::elapsed()` across the blocking call, in nanoseconds.

**schedstat deltas**: read from `/proc/self/schedstat` at work-loop
start and end. Three fields:
- `schedstat_cpu_time_ns` -- delta of field 1 (on-CPU time)
- `schedstat_run_delay_ns` -- delta of field 2 (time spent waiting
  for a CPU)
- `schedstat_ctx_switches` -- delta of field 3 (timeslice count)

`iterations` counts outer-loop iterations.

## Work-conservation watchdog

Workers send SIGUSR2 to the scheduler when stuck > 2 seconds. The
default POSIX disposition terminates the scheduler process, which stt
detects as a scheduler death and captures the sched_ext dump from
dmesg.

In repro mode, the watchdog is disabled to keep the scheduler alive
for BPF kprobe assertions.

## RAII cleanup

`WorkloadHandle` implements `Drop`: it sends SIGKILL to all child
processes and waits for them. This prevents orphaned worker processes
on error paths.
