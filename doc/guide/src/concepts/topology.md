# TestTopology

`TestTopology` provides CPU topology information for test
configuration. It discovers CPUs, last-level caches (LLCs), and NUMA
nodes, and generates cpuset partitions for scenarios.

## CPU topology hierarchy

ktstr models four levels of CPU topology, from largest to smallest:

- **NUMA node** -- a memory-proximity domain. Each node is a group of
  CPUs with fast access to a local memory bank. Cross-node memory
  access is slower.
- **LLC** (last-level cache) -- the largest cache shared by a group of
  cores. LLCs are the key scheduling boundary: tasks sharing an LLC
  benefit from shared cache lines.
- **Core** -- a physical execution unit with its own pipeline and L1/L2
  caches.
- **Thread** -- an SMT (simultaneous multithreading) sibling. Multiple
  threads share a single core's execution resources.

Containment: threads belong to a core, cores belong to an LLC, LLCs
belong to a NUMA node. For example, `2n4l4c2t` describes 2 NUMA
nodes, each with 2 LLCs, each LLC with 4 cores, each core with 2
threads = 32 CPUs total.

Most tests use a single NUMA node (the default). NUMA matters when a
scheduler makes placement decisions based on memory locality.
Single-NUMA topologies (`numa_nodes = 1`) test scheduling without
memory-locality effects. Multi-NUMA topologies test whether a
scheduler keeps tasks close to their memory. See the
[gauntlet NUMA presets](../running-tests/gauntlet.md#topology-presets)
for multi-NUMA configurations.

```rust,ignore
use ktstr::prelude::*;

pub struct TestTopology {
    // private fields — use the accessors below
}
```

## Construction

**`from_system() -> Result<Self>`** -- reads sysfs
(`/sys/devices/system/cpu/`) to discover the live topology. Reads LLC
IDs, NUMA node IDs, core IDs, and cache sizes for each online CPU.
Also scans `/sys/devices/system/node/` to discover memory-only nodes
(CXL), reads per-node meminfo and inter-node distances.

**`from_spec(numa_nodes, llcs, cores, threads) -> Self`** -- builds a
topology from a VM spec. Parameters are big-to-little: NUMA nodes,
last-level caches, cores per LLC, threads per core. Multiple LLCs
can share a NUMA node when `numa_nodes < llcs`; `llcs` must be an
exact multiple of `numa_nodes` so LLCs partition evenly across nodes
(the `#[derive(Scheduler)]` macro rejects violations at compile time
and runtime `from_spec` callers inside ktstr hold the same invariant).
CPUs numbered sequentially. Used as a fallback when sysfs is
incomplete inside a guest VM.

**`synthetic(num_cpus, num_llcs) -> Self`** (test-only) -- creates a
topology with evenly distributed CPUs across LLCs. Used in unit tests.

## Topology queries

**`total_cpus()`** -- total number of CPUs.

**`num_llcs()`** -- number of last-level caches.

**`num_numa_nodes()`** -- number of NUMA nodes.

**`all_cpus() -> &[usize]`** -- all CPU IDs, sorted.

**`all_cpuset() -> BTreeSet<usize>`** -- all CPU IDs as a set.

**`usable_cpus() -> &[usize]`** -- CPUs available for workload
placement. Reserves the last CPU for the root cgroup (cgroup 0) when
the topology has more than 2 CPUs. On 8 CPUs: usable = 0-6, CPU 7
reserved. Most built-in scenarios and `CgroupDef` cpuset specs
operate on `usable_cpus()` automatically; test authors rarely need
to query it directly.

**`usable_cpuset() -> BTreeSet<usize>`** -- usable CPUs as a set.

**`llcs() -> &[LlcInfo]`** -- all LLC domains with their CPUs, NUMA
node, cache size, and core map.

**`cpus_in_llc(idx) -> &[usize]`** -- CPUs belonging to LLC at index.

**`llc_aligned_cpuset(idx) -> BTreeSet<usize>`** -- CPUs in LLC as a set.

**`numa_aligned_cpuset(node) -> BTreeSet<usize>`** -- CPUs in all LLCs
belonging to NUMA node `node`. Filters LLCs by `numa_node() == node`
and collects their CPUs.

**`numa_node_ids() -> &BTreeSet<usize>`** -- NUMA node IDs as a
`BTreeSet`.

**`numa_nodes_for_cpuset(cpus) -> BTreeSet<usize>`** -- NUMA nodes
covered by the given CPU set. Returns the set of NUMA nodes that
contain at least one LLC with a CPU in the given set.

**`node_meminfo(node_id) -> Option<&NodeMemInfo>`** -- per-node
memory info (total and free KiB). Returns `None` when the node ID
is not present or meminfo is unavailable. `NodeMemInfo` has
`total_kb`, `free_kb`, and `used_kb()` (saturating subtraction).

**`numa_distance(from, to) -> u8`** -- inter-node NUMA distance.
Returns 255 when either node ID is not present (matches the kernel's
unreachable distance). For `from_spec()` topologies without explicit
distances, returns 10 for local and 20 for remote.

**`is_memory_only(node_id) -> bool`** -- whether the node is
memory-only (has RAM but no CPUs). Typical for CXL-attached memory
tiers.

## Construction from VM topology

**`from_vm_topology(topo) -> Self`** -- build a `TestTopology` from a
`Topology` (the VMM's topology spec). Populates LLCs, NUMA nodes,
distances, per-node memory info, and memory-only node flags.

**`from_vm_topology_with_memory(topo, total_memory_mb) -> Self`** --
same as `from_vm_topology` but accepts an optional total memory size
for uniform topologies. When `Some`, divides memory evenly across
nodes to populate `NodeMemInfo`. When `None`, memory info is omitted.

## Cpuset generation

**`split_by_llc() -> Vec<BTreeSet<usize>>`** -- one set of CPUs per LLC.

**`overlapping_cpusets(n, overlap_frac) -> Vec<BTreeSet<usize>>`** --
generates `n` cpusets with `overlap_frac` overlap between adjacent
sets. Used by `CpusetMode::Overlap`.

**`cpuset_string(cpus) -> String`** -- formats a CPU set as a compact
range string (e.g. `"0-3,5,7-9"`). Used when writing `cpuset.cpus`.

## LlcInfo

Each LLC domain is represented by an `LlcInfo`:

```rust,ignore
pub struct LlcInfo {
    cpus: Vec<usize>,
    numa_node: usize,
    cache_size_kb: Option<u64>,
    cores: BTreeMap<usize, Vec<usize>>, // core_id -> SMT siblings
}
```

Accessors: `cpus()`, `numa_node()`, `cache_size_kb()`, `cores()`,
`num_cores()`.

`num_cores()` returns the number of physical cores (from the core map),
or falls back to `cpus.len()` if no core map is populated (synthetic
topologies).

## How scenarios use topology

`TestTopology` is available to scenarios via `Ctx.topo`. The
`CpusetMode` variants use topology methods to partition CPUs:

| CpusetMode | Topology method |
|---|---|
| `LlcAligned` | `split_by_llc()` |
| `SplitHalf` | `usable_cpus()` split at midpoint |
| `SplitMisaligned` | `cpus_in_llc(0)` split at midpoint |
| `Overlap(frac)` | `overlapping_cpusets(n, frac)` |
| `Uneven(frac)` | `usable_cpus()` with asymmetric split |
| `Holdback(frac)` | `all_cpus()` with fraction held back |

The ops system's `CpusetSpec` also resolves against topology:

| CpusetSpec | Topology method |
|---|---|
| `Llc(idx)` | `llc_aligned_cpuset(idx)` |
| `Numa(node)` | `numa_aligned_cpuset(node)` |

`Llc` confines a cgroup to a single LLC's CPUs; `Numa` spans all
LLCs in a NUMA node.

See [Ops and Steps](ops.md#cpusetspec) for the full `CpusetSpec` enum.

## CPU list parsing

Two standalone functions parse CPU list strings:

**`parse_cpu_list(s) -> Result<Vec<usize>>`** -- strict parsing of
`"0-3,5,7-9"` format. Returns an error on invalid entries.

**`parse_cpu_list_lenient(s) -> Vec<usize>`** -- lenient parsing that
silently skips invalid entries.

See also: [CgroupManager](../architecture/cgroup-manager.md) for
`set_cpuset()` which consumes cpuset strings,
[CgroupGroup](../architecture/cgroup-group.md) for RAII cgroup
management, [WorkloadHandle](../architecture/workload-handle.md) for
worker lifecycle, [Scenarios](scenarios.md) for how `CpusetMode`
drives cpuset partitioning.
