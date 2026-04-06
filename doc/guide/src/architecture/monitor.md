# Monitor

The monitor observes scheduler state from the host side by reading
guest VM memory directly. It does not instrument the guest kernel or
the scheduler under test.

## What it reads

The monitor resolves kernel structure offsets via BTF (BPF Type Format)
from the guest kernel. It reads per-CPU runqueue structures to extract:

- `nr_running` -- number of runnable tasks on each CPU
- `scx_nr_running` -- tasks managed by the sched_ext scheduler
- `rq_clock` -- runqueue clock value
- `local_dsq_depth` -- scx local dispatch queue depth
- `scx_flags` -- sched_ext flags for each CPU
- scx event counters (fallback, keep-last, offline dispatch,
  skip-exiting, skip-migration-disabled)

## Sampling

The monitor takes periodic snapshots (`MonitorSample`) of all per-CPU
state. Each sample captures a point-in-time view of every CPU.

`MonitorSummary` aggregates samples into event deltas and overall
statistics.

## Threshold evaluation

`MonitorThresholds` defines pass/fail conditions:

```rust
pub struct MonitorThresholds {
    pub max_imbalance_ratio: f64,
    pub max_local_dsq_depth: u32,
    pub fail_on_stall: bool,
    pub sustained_samples: usize,
    pub max_fallback_rate: f64,
    pub max_keep_last_rate: f64,
}
```

A violation must persist for `sustained_samples` consecutive samples
before triggering a failure. This filters transient spikes from cpuset
transitions and cgroup creation/destruction.

## Uninitialized memory detection

Before the guest kernel initializes per-CPU structures, monitor reads
return uninitialized data. The monitor detects this via
`sample_looks_valid()`: any `local_dsq_depth` exceeding
`DSQ_PLAUSIBILITY_CEILING` (10,000) marks the sample as invalid.

Invalid samples are discarded from summary computation and threshold
evaluation.
