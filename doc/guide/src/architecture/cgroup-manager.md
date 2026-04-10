# CgroupManager

`CgroupManager` manages cgroup v2 filesystem operations. It creates,
configures, and removes cgroups under a parent directory.

```rust,ignore
use stt::prelude::*;

pub struct CgroupManager {
    parent: PathBuf,
}
```

## Construction

```rust,ignore
let cgroups = CgroupManager::new("/sys/fs/cgroup/stt");
cgroups.setup(true)?; // create parent dir, enable cpuset + cpu controllers
```

`new()` sets the parent path. `setup()` creates the parent directory
if it does not exist and enables cgroup controllers (`+cpuset`, and
optionally `+cpu`) on every ancestor from `/sys/fs/cgroup` down to the
parent by writing to each level's `cgroup.subtree_control`.

## Methods

**`parent_path() -> &Path`** -- returns the parent cgroup directory path.

**`create_cgroup(name)`** -- creates a child cgroup directory. Idempotent:
no error if the directory already exists. Supports nested paths
(e.g. `"nested/deep"`). For nested paths, enables `+cpuset` on
intermediate cgroups' `subtree_control`.

**`remove_cgroup(name)`** -- drains tasks from the child cgroup back to
the parent, then removes the directory. No error if the cgroup does
not exist.

**`set_cpuset(name, cpus)`** -- writes `cpuset.cpus` for a child cgroup.
The `BTreeSet<usize>` is formatted as a compact range string via
`TestTopology::cpuset_string()` (e.g. `"0-3,5,7-9"`).

**`clear_cpuset(name)`** -- writes an empty string to `cpuset.cpus`,
which inherits the parent's cpuset.

**`move_task(name, tid)`** -- writes a single PID to the child cgroup's
`cgroup.procs`.

**`move_tasks(name, tids)`** -- moves all PIDs from a slice into the
child cgroup. Tolerates ESRCH (task exited between listing and
migration) with a warning. Propagates all other errors immediately.

**`drain_tasks(name)`** -- moves all tasks from a child cgroup back to
the parent cgroup by reading `cgroup.procs` and writing each PID to
the parent's `cgroup.procs`.

**`cleanup_all()`** -- recursively removes all child cgroups under the
parent (depth-first), draining tasks at each level. Keeps the parent
directory itself.

## Timeout protection

All cgroup filesystem writes use a 2-second timeout via
`write_with_timeout()`. The write runs in a spawned thread; if it does
not complete within the timeout, the caller gets an error. This prevents
test hangs when cgroup operations block in the kernel (e.g. during
scheduler reconfigurations).

## Usage in scenarios

Scenarios access `CgroupManager` through `Ctx.cgroups`. The typical
pattern is:

```rust,ignore
fn custom_scenario(ctx: &Ctx) -> Result<AssertResult> {
    ctx.cgroups.create_cgroup("cg_0")?;
    ctx.cgroups.set_cpuset("cg_0", &cpuset)?;

    let mut h = WorkloadHandle::spawn(&config)?;
    ctx.cgroups.move_tasks("cg_0", &h.tids())?;
    h.start(); // workers block until start() is called

    // ... run workload ...

    // CgroupGroup handles cleanup on drop (see [CgroupGroup](cgroup-group.md))
    Ok(result)
}
```

See also: [CgroupGroup](cgroup-group.md) for RAII cleanup,
[WorkloadHandle](workload-handle.md) for worker lifecycle,
[TestTopology](../concepts/topology.md) for cpuset generation.
