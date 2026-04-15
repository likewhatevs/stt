# TestTopology

`TestTopology` provides CPU topology information for test
configuration. It discovers CPUs, last-level caches (LLCs), and NUMA
nodes, and generates cpuset partitions for scenarios.

```rust,ignore
use ktstr::prelude::*;

pub struct TestTopology {
    cpus: Vec<usize>,
    llcs: Vec<LlcInfo>,
    numa_nodes: BTreeSet<usize>,
}
```

## Construction

**`from_system() -> Result<Self>`** -- reads sysfs
(`/sys/devices/system/cpu/`) to discover the live topology. Reads LLC
IDs, NUMA node IDs, core IDs, and cache sizes for each online CPU.

**`from_spec(llcs, cores, threads) -> Self`** -- builds a topology
from a VM spec. One NUMA node, CPUs numbered sequentially. Used as a
fallback when sysfs is incomplete inside a guest VM.

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
reserved.

**`usable_cpuset() -> BTreeSet<usize>`** -- usable CPUs as a set.

**`llcs() -> &[LlcInfo]`** -- all LLC domains with their CPUs, NUMA
node, cache size, and core map.

**`cpus_in_llc(idx) -> &[usize]`** -- CPUs belonging to LLC at index.

**`llc_aligned_cpuset(idx) -> BTreeSet<usize>`** -- CPUs in LLC as a set.

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
