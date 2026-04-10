# CgroupGroup

`CgroupGroup` is an RAII guard that removes cgroups on drop. It
prevents cgroup leaks when workload spawning or other operations fail
between cgroup creation and cleanup.

```rust,ignore
use ktstr::prelude::*;

#[must_use = "dropping a CgroupGroup immediately destroys the cgroups it manages"]
pub struct CgroupGroup<'a> {
    cgroups: &'a CgroupManager,
    names: Vec<String>,
}
```

## Methods

**`new(cgroups: &CgroupManager) -> Self`** -- creates an empty group
bound to a `CgroupManager`.

**`add_cgroup(name, cpuset) -> Result<()>`** -- creates a cgroup and
sets its cpuset. The cgroup is tracked for removal on drop.

**`add_cgroup_no_cpuset(name) -> Result<()>`** -- creates a cgroup
without setting a cpuset. The cgroup is tracked for removal on drop.

**`names() -> &[String]`** -- returns the names of all tracked cgroups.

## Drop behavior

When the `CgroupGroup` is dropped, it calls
`CgroupManager::remove_cgroup()` on each tracked cgroup. Removal
errors are silently ignored (best-effort cleanup).

## Usage

`CgroupGroup` is the standard pattern for cgroup lifecycle management
in custom scenarios and in `run_scenario()` for data-driven scenarios.

```rust,ignore
fn custom_scenario(ctx: &Ctx) -> Result<AssertResult> {
    let mut guard = CgroupGroup::new(ctx.cgroups);
    guard.add_cgroup("cg_0", &cpuset_a)?;
    guard.add_cgroup("cg_1", &cpuset_b)?;

    // If WorkloadHandle::spawn() fails here, guard drops
    // and both cgroups are removed automatically.
    let mut h = WorkloadHandle::spawn(&config)?;
    ctx.cgroups.move_tasks("cg_0", &h.tids())?;
    h.start(); // workers block until start() is called

    // ... run workload ...

    // guard drops at end of scope, removing cg_0 and cg_1.
    Ok(result)
}
```

The helper function `setup_cgroups()` returns a `CgroupGroup` alongside
the worker handles:

```rust,ignore
let (handles, _guard) = setup_cgroups(ctx, 2, &wl)?;
// _guard lives until end of scope; cgroups are cleaned up on drop.
```

See also: [CgroupManager](cgroup-manager.md) for filesystem operations,
[WorkloadHandle](workload-handle.md) for worker lifecycle,
[TestTopology](../concepts/topology.md) for cpuset generation.
