# Verification

stt verifies scheduler behavior through two channels: worker-side
telemetry and host-side monitoring.

## Worker checks

After each scenario, stt collects
[`WorkerReport`](../architecture/workers.md#telemetry) from every worker
process. Three checks run against these reports:

**Starvation** -- any worker with `work_units == 0` fails the test.

**Fairness** -- workers in the same cgroup should get similar CPU time.
The "spread" (max runnable% - min runnable%) must be below a threshold
(15% in release builds, 35% in debug). Violations report the spread
and per-cgroup statistics.

**Scheduling gaps** -- the longest time between work iterations. Gaps
above a threshold (2000ms release, 3000ms debug) indicate the scheduler
dropped a task. Reports include the gap duration, CPU, and timing.

**Cpuset isolation** -- workers must only run on CPUs in their assigned
cpuset. Any execution on an unexpected CPU fails the test.

## Monitor checks

The [host-side monitor](../architecture/monitor.md) reads guest VM
memory (per-CPU runqueue structs via BTF offsets) and evaluates:

- **Imbalance ratio**: max/min `nr_running` across CPUs.
- **Local DSQ depth**: per-CPU dispatch queue depth.
- **Stall detection**: any CPU's `rq_clock` does not advance between
  consecutive samples. CPUs with `rq_clock == 0` are excluded (not
  yet initialized). Idle CPUs (`nr_running == 0` in both samples) are
  also excluded -- the kernel stops the tick (NOHZ) on idle CPUs, so
  `rq_clock` legitimately does not advance. Stalls use the same
  `sustained_samples` window as other monitor checks.
- **Event rates**: scx fallback and keep-last event counters.

Monitor thresholds use a sustained sample window (default: 5 samples).
A violation must persist for N consecutive samples before failing.

## Verify struct

`Verify` is a composable configuration that carries both worker checks
and monitor thresholds:

```rust
pub struct Verify {
    // Worker checks
    pub not_starved: Option<bool>,
    pub isolation: Option<bool>,
    pub max_gap_ms: Option<u64>,
    pub max_spread_pct: Option<f64>,

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

1. `Verify::default_checks()` -- baseline: `not_starved` enabled,
   monitor thresholds from `MonitorThresholds::DEFAULT`.
2. `Scheduler.verify` -- scheduler-level overrides.
3. Per-test `verify` -- test-specific overrides via `#[stt_test]`
   attributes.

All fields use last-`Some`-wins semantics. A `Some(false)` in a
higher layer can disable a check that a lower layer enabled.

```rust
let final_verify = Verify::default_checks()
    .merge(&scheduler.verify)
    .merge(&test_verify);
```

## Default thresholds

### Worker checks

| Check | Default (release) | Default (debug) |
|---|---|---|
| Scheduling gap | 2000 ms | 3000 ms |
| Fairness spread | 15% | 35% |

Debug builds run in small VMs with higher scheduling overhead, so
thresholds are relaxed. The coverage gap override
(`set_coverage_gap_ms`) further raises the gap threshold for
instrumented builds.

### Monitor checks

| Threshold | Default | Rationale |
|---|---|---|
| `max_imbalance_ratio` | 4.0 | Max/min `nr_running` across CPUs. Lower values (2-3) false-positive during cpuset transitions. |
| `max_local_dsq_depth` | 50 | Per-CPU dispatch queue overflow. Sustained depth above this means the scheduler is not consuming dispatched tasks. |
| `fail_on_stall` | true | Fail when `rq_clock` does not advance on a CPU with runnable tasks. Idle CPUs (NOHZ) are exempt. |
| `sustained_samples` | 5 | At ~100ms sample interval, requires ~500ms of sustained violation. Filters transient spikes from cpuset reconfiguration. |
| `max_fallback_rate` | 200.0/s | `select_cpu_fallback` events per second across all CPUs. Sustained rate indicates systematic `select_cpu` failure. |
| `max_keep_last_rate` | 100.0/s | `dispatch_keep_last` events per second across all CPUs. Sustained rate indicates dispatch starvation. |

All monitor thresholds use the `sustained_samples` window -- a
violation must persist for N consecutive samples before failing.

## Constants

- `Verify::NONE` -- all checks disabled, all overrides `None`.
- `Verify::default_checks()` -- `not_starved` enabled, monitor
  thresholds populated from `MonitorThresholds::DEFAULT`.

For examples of overriding thresholds at the scheduler and per-test
level, see [Customize Verification](../recipes/custom-verification.md).
