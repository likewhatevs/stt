# Scenarios

A `Scenario` is a data-driven test case. It declares the test topology
(cgroups, CPU partitioning, workloads) as data, and stt interprets it.

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

## Fields

**`name`** -- unique identifier (e.g. `"cgroup_steady"`).

**`category`** -- one of: `basic`, `cpuset`, `affinity`, `sched_class`,
`dynamic`, `stress`, `stall`, `advanced`, `nested`, `interaction`,
`performance`.

**`required_flags` / `excluded_flags`** -- typed `&[&flags::FlagDecl]`
references that constrain which flag profiles are valid. Import path:
`stt::scenario::flags::FlagDecl`. Example with constraints:
`required_flags: &[&flags::LLC_DECL]`.

**`num_cgroups`** -- number of cgroups to create.

**`cpuset_mode`** -- how to partition CPUs across cgroups:

| Variant | Behavior |
|---|---|
| `None` | No cpuset constraints |
| `LlcAligned` | One cgroup per LLC |
| `SplitHalf` | Split usable CPUs in half (see note below) |
| `SplitMisaligned` | Split at midpoint of LLC 0's CPUs (not at LLC boundary) |
| `Overlap(f64)` | Overlapping cpusets with specified fraction |
| `Uneven(f64)` | Asymmetric split (fraction for cgroup 0) |
| `Holdback(f64)` | Reserve a fraction of CPUs, split the rest |

**CPU pools**: `SplitHalf`, `Uneven`, and `SplitMisaligned` partition
`usable_cpus()`, which reserves the last CPU for the root cgroup when
the topology has more than 2 CPUs (on 8 CPUs: usable = 0-6, CPU 7
reserved). `Holdback` operates on `all_cpus()` (no reservation) --
it applies its own holdback fraction to the full CPU set.

**`cgroup_works`** -- per-cgroup workload definition:

```rust,ignore
pub struct CgroupWork {
    pub num_workers: usize,      // 0 = use ctx.workers_per_cgroup
    pub work_type: WorkType,
    pub policy: SchedPolicy,
    pub affinity: AffinityKind,
}
```

`AffinityKind` controls per-worker CPU affinity:

| Variant | Behavior |
|---|---|
| `Inherit` | No constraint (inherit from cgroup) |
| `RandomSubset` | Random subset of cgroup's cpuset |
| `LlcAligned` | CPUs in the worker's LLC |
| `CrossCgroup` | All CPUs (crosses cgroup boundaries) |
| `SingleCpu` | Pin to one CPU |

**`action`** -- `Steady` (run workers for the duration) or
`Custom(fn(&Ctx) -> Result<AssertResult>)` for scenarios with custom
logic (dynamic cgroup operations, topology changes, etc.).

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

## Scenario catalog

All scenarios are registered in `all_scenarios()`. The catalog has
scenarios across 11 categories.

## Flag profiles

Each scenario generates valid flag combinations from its
`required_flags` and `excluded_flags`. See [Flags](flags.md) for
details on how profiles are generated.
