# Scenarios

Scenarios define the scheduling conditions a test creates. Each
scenario sets up cgroups, workers, and cpusets to produce a specific
condition, then verifies the scheduler handles it correctly.

## Canned scenarios (`scenarios::*`)

`ktstr::scenario::scenarios` provides curated scenario functions that
can be called directly from `#[ktstr_test]`:

```rust,ignore
use ktstr::prelude::*;

#[ktstr_test(llcs = 1, cores = 2, threads = 1)]
fn my_test(ctx: &Ctx) -> Result<AssertResult> {
    scenarios::steady(ctx)
}
```

| Function | Condition tested | Setup |
|---|---|---|
| `steady` | Baseline fairness | 2 cgroups, no cpusets, equal CPU-spin load |
| `steady_llc` | LLC-boundary scheduling | 2 cgroups with LLC-aligned cpusets |
| `oversubscribed` | Dispatch under oversubscription | 2 cgroups, 32 mixed workers each |
| `cpuset_apply` | Cpuset assignment on running tasks | Disjoint cpusets applied mid-run |
| `cpuset_clear` | Cpuset removal on confined tasks | Cpusets cleared mid-run |
| `cpuset_resize` | Cpuset resizing adaptation | Cpusets shrink then grow |
| `cgroup_add` | New cgroup appearance | Cgroups added mid-run |
| `cgroup_remove` | Cgroup removal while others run | Cgroups removed mid-run |
| `affinity_change` | Affinity mask changes | Worker affinities randomized mid-run |
| `affinity_pinned` | Narrow-affinity contention | Workers pinned to 2-CPU subset |
| `host_contention` | Fairness between cgroup and host tasks | Host workers vs cgroup workers |
| `mixed_workloads` | Mixed workload fairness | Heavy + bursty + IO cgroups |
| `nested_steady` | Nested cgroup hierarchy | Workers in nested sub-cgroups |
| `nested_task_move` | Cross-level task migration | Tasks moved between nested cgroups |

Additional `custom_*` functions are available in
`ktstr::scenario::{affinity, basic, cpuset, dynamic, interaction,
nested, performance, stress}`. See the
[API docs](https://likewhatevs.github.io/ktstr/api/ktstr/scenario/index.html)
for the full list.

Most tests use these canned functions or build custom scenarios with
`CgroupDef` and `execute_defs` / `execute_steps` (see
[Ops and Steps](ops.md)). The `Scenario` struct below is ktstr's
internal catalog format used by the `ktstr run` CLI -- external test
suites do not need it.

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

## The Scenario struct (internal catalog)

`Scenario` is ktstr's internal catalog format. All catalog entries are
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
[`usable_cpus()`](topology.md#topology-queries).
`Holdback` operates on `all_cpus()` (no reservation).

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
