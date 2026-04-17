# MemPolicy

`MemPolicy` controls NUMA memory placement for worker processes. It
wraps `set_mempolicy(2)` and is applied after fork, before the work
loop starts.

```rust,ignore
pub enum MemPolicy {
    Default,
    Bind(BTreeSet<usize>),
    Preferred(usize),
    Interleave(BTreeSet<usize>),
    Local,
    PreferredMany(BTreeSet<usize>),
    WeightedInterleave(BTreeSet<usize>),
}
```

## Variants

**`Default`** -- inherit the parent process's memory policy. No
`set_mempolicy` syscall is made.

**`Bind(nodes)`** -- allocate only from the specified NUMA nodes
(`MPOL_BIND`). Allocation fails with `ENOMEM` if all specified nodes
are exhausted.

**`Preferred(node)`** -- prefer allocations from the specified node,
falling back to others when the preferred node is full
(`MPOL_PREFERRED`).

**`Interleave(nodes)`** -- interleave allocations round-robin across
the specified nodes (`MPOL_INTERLEAVE`).

**`Local`** -- prefer the nearest node to the CPU where the allocation
occurs (`MPOL_LOCAL`). No nodemask.

**`PreferredMany(nodes)`** -- prefer allocations from any of the
specified nodes, falling back to others when all preferred nodes are
full (`MPOL_PREFERRED_MANY`, kernel 5.15+).

**`WeightedInterleave(nodes)`** -- weighted interleave across the
specified nodes. Page distribution is proportional to per-node weights
set via `/sys/kernel/mm/mempolicy/weighted_interleave/nodeN`
(`MPOL_WEIGHTED_INTERLEAVE`, kernel 6.9+).

## Convenience constructors

```rust,ignore
MemPolicy::bind([0, 1])
MemPolicy::preferred(0)
MemPolicy::interleave([0, 1])
MemPolicy::preferred_many([0, 1])
MemPolicy::weighted_interleave([0, 1])
```

Node-set constructors (`bind`, `interleave`, `preferred_many`,
`weighted_interleave`) accept any `IntoIterator<Item = usize>` --
arrays, ranges, `Vec`, `BTreeSet`. `preferred` takes a single
`usize` node ID.

## MpolFlags

`MpolFlags` provides optional mode flags OR'd into the
`set_mempolicy(2)` mode argument:

| Flag | Value | Description |
|---|---|---|
| `NONE` | 0 | No flags |
| `STATIC_NODES` | `1 << 15` | Nodemask is absolute, not remapped when the task's cpuset changes |
| `RELATIVE_NODES` | `1 << 14` | Nodemask is relative to the task's current cpuset |
| `NUMA_BALANCING` | `1 << 13` | Enable NUMA balancing optimization for this policy |

Flags combine with `|` or `MpolFlags::union()`:

```rust,ignore
let flags = MpolFlags::STATIC_NODES | MpolFlags::NUMA_BALANCING;
```

## Usage in Work and CgroupDef

`Work` and `CgroupDef` both expose `.mem_policy()` and
`.mpol_flags()` builder methods:

```rust,ignore
use ktstr::prelude::*;

let w = Work::default()
    .workers(4)
    .mem_policy(MemPolicy::bind([0]))
    .mpol_flags(MpolFlags::STATIC_NODES);

let def = CgroupDef::named("cg_0")
    .with_cpuset(CpusetSpec::numa(0))
    .workers(4)
    .mem_policy(MemPolicy::bind([0]));
```

## Cpuset validation

When a cgroup has a cpuset, ktstr validates that the `MemPolicy`'s
node set is covered by the NUMA nodes reachable from that cpuset.
A `MemPolicy::Bind([1])` on a cgroup whose cpuset covers only NUMA
node 0 will fail with an error at setup time.

Policies without a node set (`Default`, `Local`) skip validation.

## node_set()

`MemPolicy::node_set()` returns the NUMA node IDs referenced by the
policy. Returns the node set for `Bind`, `Interleave`,
`PreferredMany`, and `WeightedInterleave`; a single-element set for
`Preferred`; and an empty set for `Default`/`Local`.

## NUMA verification

Page locality and migration results from workers using `MemPolicy` are
checked by the [NUMA verification
assertions](verification.md#numa-checks). The expected node set for
locality checks is derived from the worker's `MemPolicy` at evaluation
time.

## Example: NUMA-aware test

A complete test that verifies page locality across two NUMA nodes:

```rust,ignore
use ktstr::prelude::*;

#[ktstr_test(
    numa_nodes = 2, llcs = 4, cores = 4, threads = 1,
    min_numa_nodes = 2,
    min_page_locality = 0.8,
)]
fn numa_locality(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        CgroupDef::named("node0")
            .with_cpuset(CpusetSpec::numa(0))
            .workers(4)
            .mem_policy(MemPolicy::bind([0])),
        CgroupDef::named("node1")
            .with_cpuset(CpusetSpec::numa(1))
            .workers(4)
            .mem_policy(MemPolicy::bind([1])),
    ])
}
```

Each cgroup's workers are pinned to a single NUMA node's CPUs via
`CpusetSpec::numa()` and their memory allocations are bound to the
same node via `MemPolicy::bind()`. The `min_page_locality` threshold
fails the test if less than 80% of pages land on the expected node.
