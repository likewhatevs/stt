# Custom Scenarios

For dynamic scenarios (cgroup creation/removal, cpuset changes), prefer
the [ops/steps system](../concepts/ops.md) over raw `Action::Custom`.
See [Write a Dynamic Scenario](../recipes/dynamic-scenario.md) for
ops-based examples.

Use `Action::Custom` only when you need logic that the ops system
cannot express.

## Writing a custom scenario

```rust,ignore
use ktstr::prelude::*;
use ktstr::scenario::*;

fn my_custom_scenario(ctx: &Ctx) -> Result<AssertResult> {
    let wl = dfl_wl(ctx);
    let (handles, _guard) = setup_cgroups(ctx, 2, &wl)?;

    // Custom logic: resize cpusets, move workers, etc.
    std::thread::sleep(ctx.duration);

    Ok(collect_all(handles, &ctx.assert))
}
```

## Helper functions

**`setup_cgroups(ctx, n, wl)`** -- creates N cgroups, spawns workers,
returns `Result<(Vec<`[`WorkloadHandle`](../architecture/workload-handle.md)`>, `[`CgroupGroup`](../architecture/cgroup-group.md)`)>`.
Bind the `CgroupGroup` to a named variable (e.g. `_guard`) so it
lives until end of scope.
See [CgroupGroup](../architecture/cgroup-group.md) for drop semantics.

**`collect_all(handles, checks)`** -- stops all workers, collects reports,
runs worker-level checks when configured, otherwise falls back to
`assert_not_starved()`. Merges results: if any worker group fails, the
overall result fails.

**`dfl_wl(ctx)`** -- creates a `WorkloadConfig` with
`ctx.workers_per_cgroup` workers and default settings.

**`spawn_diverse(ctx, cgroup_names)`** -- spawns different
[work types](../concepts/work-types.md) across cgroups, rotating
through (CpuSpin, Bursty{50ms burst / 100ms sleep}, IoSync, Mixed,
YieldHeavy). Each cgroup uses `ctx.workers_per_cgroup` workers except
IoSync cgroups, which always use 2 workers so blocking IO does not
drown the scenario.

## The Ctx struct

Custom scenarios receive a `Ctx` reference:

```rust,ignore
pub struct Ctx<'a> {
    pub cgroups: &'a CgroupManager,
    pub topo: &'a TestTopology,
    pub duration: Duration,
    pub workers_per_cgroup: usize,
    pub sched_pid: u32,
    pub settle: Duration,
    pub work_type_override: Option<WorkType>,
    pub assert: Assert,
    pub wait_for_map_write: bool,
}
```

**`cgroups`** -- create/remove cgroups, set cpusets, move tasks. See
[CgroupManager](../architecture/cgroup-manager.md) for method details.

**`topo`** -- query CPU topology (LLCs, NUMA nodes, memory info,
distances). Provides CPU enumeration, LLC/NUMA partitioning, cpuset
generation, and inter-node distance queries. See
[TestTopology](../concepts/topology.md) for the full API reference.

**`sched_pid`** -- scheduler process ID for liveness checks.

**`settle`** -- time to wait after cgroup creation for the scheduler
to stabilize.

## Verification in custom scenarios

Use `Assert` for both direct report checking and ops-based scenarios.
Call `assert.assert_cgroup(reports, cpuset)` for manual report
collection, or use `execute_steps_with()` for ops-based scenarios. See
[Verification](../concepts/verification.md#worker-checks-via-assert).

## Registering a custom scenario (ktstr contributors only)

See [Write a Dynamic Scenario: Registering](../recipes/dynamic-scenario.md#registering-ktstr-contributors-only).
