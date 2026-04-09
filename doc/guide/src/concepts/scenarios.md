# Scenarios

A `Scenario` is a data-driven test case. It declares the test topology
(cgroups, CPU partitioning, workloads) as data, and stt interprets it.

```rust,ignore
pub struct Scenario {
    pub name: &'static str,
    pub category: &'static str,
    pub description: &'static str,
    pub required_flags: &'static [&'static flags::FlagDecl],
    pub excluded_flags: &'static [&'static flags::FlagDecl],
    pub num_cgroups: usize,
    pub cpuset_mode: CpusetMode,
    pub cgroup_works: Vec<CgroupWork>,
    pub action: Action,
}
```

## Fields

**`name`** -- unique identifier (e.g. `"cgroup_steady"`).

**`category`** -- one of: `basic`, `cpuset`, `affinity`, `sched_class`,
`dynamic`, `stress`, `stall`, `advanced`, `nested`, `interaction`,
`performance`.

**`required_flags` / `excluded_flags`** -- typed `&[&flags::FlagDecl]`
references that constrain which flag profiles are valid. `FlagDecl` is
in the [prelude](../writing-tests/scheduler-definitions.md#defining-flags).
Example: `required_flags: &[&MY_LLC_DECL]`.

**`num_cgroups`** -- number of cgroups to create.

**`cpuset_mode`** -- how to partition CPUs across cgroups:

| Variant | Behavior |
|---|---|
| `None` | No cpuset constraints |
| `LlcAligned` | One cgroup per LLC |
| `SplitHalf` | Split usable CPUs in half (see note below) |
| `SplitMisaligned` | Split at midpoint of LLC 0's CPUs (not at LLC boundary) |
| `Overlap(f64)` | Overlapping cpusets with specified fraction |
| `Uneven(f64)` | Asymmetric split (fraction for cgroup 0) |
| `Holdback(f64)` | Reserve a fraction of CPUs, split the rest |

**CPU pools**: `SplitHalf`, `Uneven`, and `SplitMisaligned` partition
`usable_cpus()`, which reserves the last CPU for the root cgroup when
the topology has more than 2 CPUs (on 8 CPUs: usable = 0-6, CPU 7
reserved). `Holdback` operates on `all_cpus()` (no reservation) --
it applies its own holdback fraction to the full CPU set.

**`cgroup_works`** -- per-cgroup workload definition:

```rust,ignore
pub struct CgroupWork {
    pub num_workers: Option<usize>, // None = use ctx.workers_per_cgroup
    pub work_type: WorkType,
    pub policy: SchedPolicy,
    pub affinity: AffinityKind,
}
```

`AffinityKind` controls per-worker CPU affinity:

| Variant | Behavior |
|---|---|
| `Inherit` | No constraint (inherit from cgroup) |
| `RandomSubset` | Random subset of cgroup's cpuset |
| `LlcAligned` | CPUs in the worker's LLC |
| `CrossCgroup` | All CPUs (crosses cgroup boundaries) |
| `SingleCpu` | Pin to one CPU |

**`action`** -- `Steady` (run workers for the duration) or
`Custom(fn(&Ctx) -> Result<AssertResult>)` for scenarios with custom
logic (dynamic cgroup operations, topology changes, etc.).

## How scenarios run

For `Steady` scenarios, `run_scenario()`:

1. Resolves cpusets from `cpuset_mode` and the VM's topology.
2. Creates cgroups via `CgroupManager`.
3. Forks worker processes via `WorkloadHandle::spawn()`.
4. Moves workers into their target cgroups.
5. Signals workers to start (two-phase start protocol).
6. Polls scheduler liveness during the workload phase.
7. Stops workers, collects `WorkerReport` telemetry.
8. Runs starvation, fairness, gap, and cpuset isolation checks (see
   [Worker checks](verification.md#worker-checks)).

`Custom` scenarios get a `Ctx` reference and implement their own logic
using the same building blocks. See
[Custom Scenarios](../writing-tests/custom-scenarios.md) for the `Ctx`
struct and helper functions.

## Scenario catalog

All scenarios are registered in `all_scenarios()`. The catalog has
scenarios across 11 categories.

### Canned scenarios (`scenarios::*`)

These thin wrappers in `stt::scenario::scenarios` call `execute_defs`
or delegate to a `custom_*` function:

| Function | Description |
|---|---|
| `steady` | 2 cgroups, no cpusets, equal CPU-spin load |
| `steady_llc` | 2 cgroups with LLC-aligned cpusets |
| `oversubscribed` | 2 cgroups, 32 mixed workers each |
| `cpuset_apply` | Disjoint cpusets applied mid-run |
| `cpuset_clear` | Cpusets cleared mid-run |
| `cpuset_resize` | Cpusets shrink then grow |
| `cgroup_add` | Cgroups added mid-run |
| `cgroup_remove` | Cgroups removed mid-run |
| `affinity_change` | Worker affinities randomized mid-run |
| `affinity_pinned` | Workers pinned to 2-CPU subset |
| `host_contention` | Host workers vs cgroup workers |
| `mixed_workloads` | Heavy + bursty + IO cgroups |
| `nested_steady` | Workers in nested sub-cgroups |
| `nested_task_move` | Tasks moved between nested cgroups |

### Custom scenario functions

These are in `stt::scenario::{module}` and can be used directly in
`Action::Custom(...)` or called from custom test functions.

**affinity**: `custom_cgroup_affinity_change`,
`custom_cgroup_multicpu_pin`, `custom_cgroup_cpuset_multicpu_pin`

**basic**: `custom_host_cgroup_contention`, `custom_sched_mixed`,
`custom_cgroup_pipe_io`

**cpuset**: `custom_cgroup_cpuset_apply_midrun`,
`custom_cgroup_cpuset_clear_midrun`, `custom_cgroup_cpuset_resize`,
`custom_cgroup_cpuset_swap_disjoint`,
`custom_cgroup_cpuset_workload_imbalance`,
`custom_cgroup_cpuset_change_imbalance`,
`custom_cgroup_cpuset_load_shift`

**dynamic**: `custom_cgroup_add_midrun`,
`custom_cgroup_remove_midrun`, `custom_cgroup_rapid_churn`,
`custom_cgroup_cpuset_add_remove`,
`custom_cgroup_add_during_imbalance`

**interaction**: `custom_cgroup_add_load_imbalance`,
`custom_cgroup_imbalance_mixed_workload`,
`custom_cgroup_load_oscillation`,
`custom_cgroup_4way_load_imbalance`,
`custom_cgroup_cpuset_imbalance_combined`,
`custom_cgroup_cpuset_overlap_imbalance_combined`,
`custom_cgroup_noctrl_task_migration`,
`custom_cgroup_noctrl_imbalance`,
`custom_cgroup_noctrl_cpuset_change`,
`custom_cgroup_noctrl_load_imbalance`,
`custom_cgroup_io_compute_imbalance`

**nested**: `custom_nested_cgroup_steady`,
`custom_nested_cgroup_task_move`,
`custom_nested_cgroup_rapid_churn`,
`custom_nested_cgroup_cpuset`,
`custom_nested_cgroup_imbalance`,
`custom_nested_cgroup_noctrl`

**performance**: `custom_cache_pressure_imbalance`,
`custom_cache_yield_wake_affine`,
`custom_cache_pipe_io_compute_imbalance`,
`custom_fanout_wake`

**stress**: `custom_cgroup_per_cpu`,
`custom_cgroup_exhaust_reuse`,
`custom_cgroup_dsq_contention`,
`custom_cgroup_workload_variety`,
`custom_cgroup_cpuset_workload_variety`,
`custom_cgroup_dynamic_workload_variety`,
`custom_cgroup_cpuset_crossllc_race`

## Flag profiles

Each scenario generates valid flag combinations from its
`required_flags` and `excluded_flags`. See [Flags](flags.md) for
details on how profiles are generated.
