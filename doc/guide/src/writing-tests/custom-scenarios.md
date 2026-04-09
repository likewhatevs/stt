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

fn my_custom_scenario(ctx: &Ctx) -> Result<AssertResult> {
    let wl = dfl_wl(ctx);
    let (handles, _guard) = setup_cgroups(ctx, 2, &wl)?;

    // Custom logic: resize cpusets, move workers, etc.
    std::thread::sleep(ctx.duration);

    Ok(collect_all(handles, &ctx.assert))
}
```

## Helper functions

**`setup_cgroups(ctx, n, wl)`** -- creates N cgroups, spawns workers in
each, starts them. Returns `Result<(Vec<WorkloadHandle>, CgroupGroup)>`. The
`CgroupGroup` is an RAII guard that removes cgroups on drop.

> **Warning:** `let _ = CgroupGroup::new(...)` drops immediately -- the
> guard is destroyed at the end of the statement, not the end of the
> scope. Always bind to a named variable (`let _guard = ...`) to keep
> cgroups alive for the duration of the test.

**`collect_all(handles, checks)`** -- stops all workers, collects reports,
runs `checks.assert_cgroup()` when worker-level checks are configured,
otherwise falls back to `assert_not_starved()`. Merges results: if any
worker group fails, the overall result fails.

**`dfl_wl(ctx)`** -- creates a `WorkloadConfig` with
`ctx.workers_per_cgroup` workers and default settings.

**`spawn_diverse(ctx, cgroup_names)`** -- spawns different work types
(CpuSpin, Bursty, IoSync, Mixed, YieldHeavy) across cgroups.

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

**`cgroups`** -- create/remove cgroups, set cpusets, move tasks.
`move_task(name, tid)` moves a single task; `move_tasks(name, &tids)`
moves all tasks in a slice (calls `move_task` per TID).

**`topo`** -- query CPU topology (LLCs, NUMA nodes, total CPUs).
Key methods:

- `all_cpus() -> &[usize]` -- all CPU IDs, sorted.
- `all_cpuset() -> BTreeSet<usize>` -- all CPU IDs as a set.
- `usable_cpus() -> &[usize]` -- all CPUs except the last (reserved
  for root cgroup) when topology has >2 CPUs.
- `usable_cpuset() -> BTreeSet<usize>` -- usable CPUs as a set.
- `split_by_llc() -> Vec<BTreeSet<usize>>` -- one BTreeSet per LLC.
- `num_llcs()`, `total_cpus()`, `num_numa_nodes()` -- counts.
- `cpus_in_llc(idx) -> &[usize]` -- CPUs in LLC at index.
- `llc_aligned_cpuset(idx) -> BTreeSet<usize>` -- same as
  `cpus_in_llc` but returns a set.

**`sched_pid`** -- scheduler process ID for liveness checks.

**`settle`** -- time to wait after cgroup creation for the scheduler
to stabilize.

## Verification in custom scenarios

Use `Assert` for both direct report checking and ops-based scenarios.
Call `assert.assert_cgroup(reports, cpuset)` for manual report
collection, or use `execute_steps_with()` for ops-based scenarios. See
[Verification](../concepts/verification.md#worker-checks-via-assert).

## Registering a custom scenario

Add it to `all_scenarios()` in `src/scenario/catalog.rs`:

```rust,ignore
Scenario {
    name: "my_scenario",
    category: "dynamic",
    description: "Test dynamic cgroup resizing",
    required_flags: &[],
    excluded_flags: &[],
    num_cgroups: 0,
    cpuset_mode: CpusetMode::None,
    cgroup_works: vec![],
    action: Action::Custom(my_custom_scenario),
}
```
