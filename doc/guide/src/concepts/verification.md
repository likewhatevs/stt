# Verification

ktstr verifies scheduler behavior through two channels: worker-side
telemetry and host-side monitoring.

## Worker checks

After each scenario, ktstr collects
[`WorkerReport`](../architecture/workers.md#telemetry) from every worker
process. Several checks run against these reports:

**Starvation** -- any worker with `work_units == 0` fails the test.

**Fairness** -- workers in the same cgroup should get similar CPU time.
The "spread" (max off-CPU% - min off-CPU%) must be below a threshold
(15% in release builds, 35% in debug). Violations report the spread
and per-cgroup statistics.

**Scheduling gaps** -- the longest time between work iterations. Gaps
above a threshold (2000ms release, 3000ms debug) indicate the scheduler
dropped a task. Reports include the gap duration, CPU, and timing.

**Cpuset isolation** -- workers must only run on CPUs in their assigned
cpuset. Any execution on an unexpected CPU fails the test.

**Throughput parity** -- `assert_throughput_parity()` checks that
workers produce similar throughput (work_units per CPU-second). Two
thresholds:
- `max_throughput_cv`: coefficient of variation across workers. High
  CV means the scheduler gives some workers disproportionately less
  effective CPU. Requires at least 2 workers with nonzero CPU time.
- `min_work_rate`: minimum work_units per CPU-second per worker.
  Catches cases where all workers are equally slow (CV passes but
  absolute throughput is too low).

Neither threshold is set by default; enable via `Assert` setters or
`#[ktstr_test]` attributes.

**Benchmarking** -- `assert_benchmarks()` checks per-wakeup latency
and iteration throughput. Three thresholds:
- `max_p99_wake_latency_ns`: p99 of all `wake_latencies_ns` samples
  across workers in a cgroup. Populated only for blocking work types
  (FutexPingPong, FutexFanOut, CachePipe, Bursty, CacheYield, PipeIo,
  IoSync, Sequence with Sleep/Yield/Io phases).
- `max_wake_latency_cv`: coefficient of variation of wake latency
  samples. High CV means inconsistent scheduling latency.
- `min_iteration_rate`: minimum outer-loop iterations per wall-clock
  second per worker.

None are set by default. Set via `Assert` setters or `#[ktstr_test]`
attributes.

## Monitor checks

The [host-side monitor](../architecture/monitor.md) reads guest VM
memory (per-CPU runqueue structs via BTF offsets) and evaluates:

- **Imbalance ratio**: `max(nr_running) / max(1, min(nr_running))`
  across CPUs. The denominator is clamped to 1 so an all-idle sample
  does not divide by zero.
- **Local DSQ depth**: per-CPU dispatch queue depth.
- **Stall detection**: `rq_clock` not advancing on a CPU with
  runnable tasks. Idle CPUs and preempted vCPUs are exempt. See
  [Monitor: Stall detection](../architecture/monitor.md#stall-detection)
  for exemption details.
- **Event rates**: scx fallback and keep-last event counters.

Monitor thresholds use a sustained sample window (default: 5 samples).
A violation must persist for N consecutive samples before failing.

## Assert struct

`Assert` is a composable configuration that carries both worker checks
and monitor thresholds:

```rust,ignore
pub struct Assert {
    // Worker checks
    pub not_starved: Option<bool>,
    pub isolation: Option<bool>,
    pub max_gap_ms: Option<u64>,
    pub max_spread_pct: Option<f64>,

    // Throughput checks
    pub max_throughput_cv: Option<f64>,
    pub min_work_rate: Option<f64>,

    // Benchmarking checks
    pub max_p99_wake_latency_ns: Option<u64>,
    pub max_wake_latency_cv: Option<f64>,
    pub min_iteration_rate: Option<f64>,
    pub max_migration_ratio: Option<f64>,

    // Monitor checks
    pub max_imbalance_ratio: Option<f64>,
    pub max_local_dsq_depth: Option<u32>,
    pub fail_on_stall: Option<bool>,
    pub sustained_samples: Option<usize>,
    pub max_fallback_rate: Option<f64>,
    pub max_keep_last_rate: Option<f64>,
}
```

Every field is `Option`. `None` means "inherit from parent layer."

## Merge layers

Verification uses a three-layer merge:

1. `Assert::default_checks()` -- baseline: `not_starved` enabled,
   monitor thresholds from `MonitorThresholds::DEFAULT`.
2. `Scheduler.assert` -- scheduler-level overrides.
3. Per-test `assert` -- test-specific overrides via `#[ktstr_test]`
   attributes.

All fields use last-`Some`-wins semantics. A `Some(false)` in a
higher layer can disable a check that a lower layer enabled.

```rust,ignore
let final_assert = Assert::default_checks()
    .merge(&scheduler.assert)
    .merge(&test_assert);
```

## Default thresholds

### Worker checks

| Check | Default (release) | Default (debug) |
|---|---|---|
| Scheduling gap | 2000 ms | 3000 ms |
| Fairness spread | 15% | 35% |

Debug builds run in small VMs with higher scheduling overhead, so
thresholds are relaxed. Coverage-instrumented builds collect profraw
data for code coverage analysis; all assertion and monitor threshold
checks run normally.

### Monitor checks

| Threshold | Default | Rationale |
|---|---|---|
| `max_imbalance_ratio` | 4.0 | `max(nr_running) / max(1, min(nr_running))` across CPUs (denominator clamped to 1 so an all-idle sample does not divide by zero). Lower values (2-3) false-positive during cpuset transitions. |
| `max_local_dsq_depth` | 50 | Per-CPU dispatch queue overflow. Sustained depth above this means the scheduler is not consuming dispatched tasks. |
| `fail_on_stall` | true | Fail when `rq_clock` does not advance on a CPU with runnable tasks. Idle CPUs (NOHZ) and preempted vCPUs are exempt. |
| `sustained_samples` | 5 | At ~100ms sample interval, requires ~500ms of sustained violation. Filters transient spikes from cpuset reconfiguration. |
| `max_fallback_rate` | 200.0/s | `select_cpu_fallback` events per second across all CPUs. Sustained rate indicates systematic `select_cpu` failure. |
| `max_keep_last_rate` | 100.0/s | `dispatch_keep_last` events per second across all CPUs. Sustained rate indicates dispatch starvation. |

All monitor thresholds use the `sustained_samples` window -- a
violation must persist for N consecutive samples before failing.

## Worker checks via Assert

`Assert` provides `assert_cgroup()` for running worker-side checks
directly against collected reports:

```rust,ignore
let a = Assert::default_checks().max_gap_ms(5000);
let result = a.assert_cgroup(&reports, Some(&cpuset));
```

Use `Assert` for both the merge chain (`#[ktstr_test]` attributes,
`Scheduler.assert`, `execute_steps_with`) and direct report checking.

## Constants

- `Assert::NONE` -- all checks disabled, all overrides `None`.
- `Assert::default_checks()` -- `not_starved` enabled, monitor
  thresholds populated from `MonitorThresholds::DEFAULT`.

## AssertResult

`AssertResult` carries pass/fail status, diagnostic messages, and
aggregated statistics from a scenario run.

### Construction

- `AssertResult::pass()` -- creates a passing result with empty
  details and default stats.
- `AssertResult::skip(reason)` -- creates a passing result with a
  skip reason in `details`. Used when a scenario cannot run under the
  current topology or flag combination but is not a failure.

### Fields

- `passed: bool` -- whether all checks passed.
- `details: Vec<String>` -- human-readable diagnostic messages
  (failure reasons, warnings, skip reasons).
- `stats: ScenarioStats` -- aggregated worker telemetry across all
  cgroups (spread, gaps, migrations, wake latency, iterations).

### Merging

`result.merge(other)` combines two results. If `other.passed` is
false, the merged result is also false. Details and stats are
accumulated:

```rust,ignore
let mut combined = AssertResult::pass();
combined.merge(cgroup_0_result);
combined.merge(cgroup_1_result);
// combined.passed is false if either cgroup failed
// combined.details contains messages from both
```

Stats merging takes worst values across cgroups for spread, gap, wake
latency, and migration ratio. Counters (workers, cpus, migrations,
iterations) are summed.

For examples of overriding thresholds at the scheduler and per-test
level, see [Customize Verification](../recipes/custom-verification.md).
