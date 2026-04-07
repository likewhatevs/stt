# Custom Scenarios

For dynamic scenarios (cgroup creation/removal, cpuset changes), prefer
the [ops/steps system](../concepts/ops.md) over raw `Action::Custom`.
See [Write a Dynamic Scenario](../recipes/dynamic-scenario.md) for
ops-based examples.

Use `Action::Custom` only when you need logic that the ops system
cannot express.

## Writing a custom scenario

```rust,ignore
use stt::prelude::*;
use stt::scenario::*;

fn my_custom_scenario(ctx: &Ctx) -> Result<VerifyResult> {
    let wl = dfl_wl(ctx);
    let (handles, _guard) = setup_cells(ctx, 2, &wl)?;

    // Custom logic: resize cpusets, move workers, etc.
    std::thread::sleep(ctx.duration);

    Ok(collect_all(handles))
}
```

## Helper functions

**`setup_cells(ctx, n, wl)`** -- creates N cgroups, spawns workers in
each, starts them. Returns `(Vec<WorkloadHandle>, CgroupGroup)`. The
`CgroupGroup` is an RAII guard that removes cgroups on drop.

**`collect_all(handles)`** -- stops all workers, collects reports, runs
`verify_not_starved()` on each. Merges results: if any worker group
fails, the overall result fails. Details from all groups are combined.

**`dfl_wl(ctx)`** -- creates a `WorkloadConfig` with
`ctx.workers_per_cell` workers and default settings.

**`spawn_diverse(ctx, cell_names)`** -- spawns different work types
(CpuSpin, Bursty, IoSync, Mixed, YieldHeavy) across cells.

## The Ctx struct

Custom scenarios receive a `Ctx` reference:

```rust,ignore
pub struct Ctx<'a> {
    pub cgroups: &'a CgroupManager,
    pub topo: &'a TestTopology,
    pub duration: Duration,
    pub workers_per_cell: usize,
    pub sched_pid: u32,
    pub settle_ms: u64,
    pub work_type_override: Option<WorkType>,
}
```

**`cgroups`** -- create/remove cgroups, set cpusets, move tasks.

**`topo`** -- query CPU topology (LLCs, NUMA nodes, total CPUs).

**`sched_pid`** -- scheduler process ID for liveness checks.

**`settle_ms`** -- time to wait after cgroup creation for the scheduler
to stabilize.

## Registering a custom scenario

Add it to `all_scenarios()` in `src/scenario/catalog.rs`:

```rust,ignore
Scenario {
    name: "my_scenario",
    category: "dynamic",
    description: "Test dynamic cgroup resizing",
    required_flags: &[],
    excluded_flags: &[],
    num_cells: 0,
    cpuset_mode: CpusetMode::None,
    cell_works: vec![],
    action: Action::Custom(my_custom_scenario),
}
```
