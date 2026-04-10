# Scenarios

Most tests define cgroups with `CgroupDef` and run them via
`execute_defs` or `execute_steps` (see [Ops and Steps](ops.md)).
The `Scenario` struct described below is stt's internal catalog
format -- external test suites do not need it.

## Canned scenarios (`scenarios::*`)

`stt::scenario::scenarios` provides curated scenario functions that
can be called directly from `#[stt_test]`:

```rust,ignore
use stt::prelude::*;

#[stt_test(sockets = 1, cores = 2, threads = 1)]
fn my_test(ctx: &Ctx) -> Result<AssertResult> {
    scenarios::steady(ctx)
}
```

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

Additional `custom_*` functions are available in
`stt::scenario::{affinity, basic, cpuset, dynamic, interaction,
nested, performance, stress}`. See the
[API docs](https://likewhatevs.github.io/stt/api/stt/scenario/index.html)
for the full list.

## The Scenario struct (internal catalog)

`Scenario` is stt's internal catalog format. All catalog entries are
registered in `all_scenarios()` across 11 categories.

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

**`cpuset_mode`** -- how to partition CPUs across cgroups. This is an
internal type; external tests use
[`CpusetSpec`](ops.md#cpusetspec) instead.

| Variant | Behavior |
|---|---|
| `None` | No cpuset constraints |
| `LlcAligned` | One cgroup per LLC |
| `SplitHalf` | Split usable CPUs in half |
| `SplitMisaligned` | Split at midpoint of LLC 0's CPUs (not at LLC boundary) |
| `Overlap(f64)` | Overlapping cpusets with specified fraction |
| `Uneven(f64)` | Asymmetric split (fraction for cgroup 0) |
| `Holdback(f64)` | Reserve a fraction of CPUs, split the rest |

**CPU pools**: `SplitHalf`, `Uneven`, and `SplitMisaligned` partition
[`usable_cpus()`](topology.md#topology-queries), which reserves the
last CPU for the root cgroup. `Holdback` operates on `all_cpus()`
(no reservation).

**`cgroup_works`** -- per-cgroup workload definition:

```rust,ignore
pub struct CgroupWork {
    pub num_workers: Option<usize>, // None = use ctx.workers_per_cgroup
    pub work_type: WorkType,
    pub policy: SchedPolicy,
    pub affinity: AffinityKind,
}
```

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

## Flag profiles

Each scenario generates valid flag combinations from its
`required_flags` and `excluded_flags`. See [Flags](flags.md) for
details on how profiles are generated.
