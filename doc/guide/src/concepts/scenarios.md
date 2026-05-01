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
[Ops and Steps](ops.md)). Custom scenarios receive a `Ctx` reference
and use the same building blocks; see
[Custom Scenarios](../writing-tests/custom-scenarios.md) for the
`Ctx` struct and helper functions.

## Flag profiles

`#[derive(Scheduler)]` declares the flag set for a scheduler. Each
test then uses `required_flags` and `excluded_flags` on
`#[ktstr_test]` to constrain which combinations the test runs under.
See [Flags](flags.md) for profile generation.
