# The #[ktstr_test] Macro

`#[ktstr_test]` registers a function as an integration test that runs
inside a VM.

## Basic usage

```rust,ignore
use ktstr::prelude::*;

#[ktstr_test(llcs = 2, cores = 4, threads = 2)]
fn my_test(ctx: &Ctx) -> Result<AssertResult> {
    // ctx provides cgroup manager, topology, duration, etc.
    Ok(AssertResult::pass())
}
```

When a scheduler with a default topology is specified, the topology
can be omitted:

```rust,ignore
const MY_SCHED: Scheduler = Scheduler::new("my_sched")
    .binary(SchedulerSpec::Name("scx_my_sched"))
    //            numa, llcs, cores/llc, threads/core
    .topology(1, 2, 4, 1);

#[ktstr_test(scheduler = MY_SCHED)]
fn inherited_topo(ctx: &Ctx) -> Result<AssertResult> {
    // Inherits 1n2l4c1t from MY_SCHED
    Ok(AssertResult::pass())
}
```

The function must have signature
`fn(&ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult>`.

## What the macro generates

1. Renames the function to `__ktstr_inner_{name}`.
2. Registers it in the `KTSTR_TESTS` distributed slice via linkme.
3. Emits a `#[test]` wrapper that calls `run_ktstr_test()`.

The `#[test]` wrapper boots a VM with the specified topology and runs
the function inside it.

## Attributes

All attributes are optional with defaults.

### Topology

| Attribute | Default | Description |
|---|---|---|
| `llcs` | inherited | Number of LLCs (`sockets` is accepted as an alias) |
| `numa_nodes` | inherited | Number of NUMA nodes |
| `cores` | inherited | Cores per LLC |
| `threads` | inherited | Threads per core |
| `memory_mb` | 2048 | VM memory in MB |

Each dimension independently inherits from `Scheduler.topology` when
a `scheduler` is specified and that dimension is not explicitly set.
Without a scheduler, unset dimensions use macro defaults (numa_nodes=1,
llcs=1, cores=2, threads=1). The default is a single-NUMA topology,
so most tests do not need to set `numa_nodes`. See
[Default topology](scheduler-definitions.md#default-topology).

### Scheduler

| Attribute | Default | Description |
|---|---|---|
| `scheduler = CONST` | `Scheduler::EEVDF` | Rust const path to a `Scheduler` definition |
| `extra_sched_args = [...]` | `[]` | Extra CLI args for the scheduler, appended after `Scheduler::sched_args`. |
| `watchdog_timeout_s` | 4 | scx watchdog override (seconds). Applied via `scx_sched.watchdog_timeout` on 7.1+ kernels (BTF-detected) and via the static `scx_watchdog_timeout` symbol on 6.16-7.0 kernels. When neither path is available the override silently no-ops. |

### Verification

| Attribute | Default | Description |
|---|---|---|
| `not_starved` | inherited | Enable starvation (zero work units), fairness spread, and scheduling gap checks |
| `isolation` | inherited | Enable cpuset isolation check (workers must stay on assigned CPUs) |
| `max_gap_ms` | inherited | Max scheduling gap threshold |
| `max_spread_pct` | inherited | Max fairness spread threshold |
| `max_throughput_cv` | inherited | Max coefficient of variation for worker throughput |
| `min_work_rate` | inherited | Minimum work_units per CPU-second per worker |
| `max_imbalance_ratio` | inherited | Monitor imbalance ratio |
| `max_local_dsq_depth` | inherited | Monitor DSQ depth |
| `fail_on_stall` | inherited | Fail on stall detection |
| `sustained_samples` | inherited | Sample window for sustained violations |
| `max_fallback_rate` | inherited | Max fallback event rate |
| `max_keep_last_rate` | inherited | Max keep-last event rate |
| `max_p99_wake_latency_ns` | inherited | Max p99 wake latency in nanoseconds |
| `max_wake_latency_cv` | inherited | Max wake latency coefficient of variation |
| `min_iteration_rate` | inherited | Minimum iterations per wall-clock second per worker |
| `max_migration_ratio` | inherited | Max migration ratio (migrations/iterations) per cgroup |

`not_starved = true` enables three distinct checks: starvation (any
worker with zero work units), fairness spread (max-min off-CPU% below
`max_spread_pct`), and scheduling gaps (longest gap below `max_gap_ms`).
Each threshold can be overridden independently. See
[Customize Verification](../recipes/custom-verification.md) for
override examples and [Verification](../concepts/verification.md) for
the merge chain.

### Flag constraints

| Attribute | Default | Description |
|---|---|---|
| `required_flags = [...]` | `[]` | Flags that must be present in every flag profile |
| `excluded_flags = [...]` | `[]` | Flags that must not be present in any flag profile |

Values are arrays of string literals or path expressions from
`#[derive(Scheduler)]`:

```rust,ignore
// String literals
#[ktstr_test(
    required_flags = ["llc", "borrow"],
    excluded_flags = ["no-ctrl"],
)]

// Path expressions (typed constants from derive)
#[ktstr_test(
    required_flags = [MySchedFlag::LLC, MySchedFlag::BORROW],
    excluded_flags = [MySchedFlag::NO_CTRL],
)]
```

These constrain `generate_profiles()`: `required_flags` are always
included, `excluded_flags` are never included. Invalid combinations
(e.g. `steal` without `llc`) are still rejected by dependency checks.
See [Gauntlet Tests](gauntlet-tests.md#controlling-flag-coverage)
for profile generation examples.

### Topology constraints

| Attribute | Default | Description |
|---|---|---|
| `min_llcs` | 1 | Minimum LLCs for gauntlet topology filtering |
| `max_llcs` | 12 | Maximum LLCs for gauntlet topology filtering |
| `min_cpus` | 1 | Minimum total CPU count for gauntlet topology filtering |
| `max_cpus` | 192 | Maximum total CPU count for gauntlet topology filtering |
| `min_numa_nodes` | 1 | Minimum NUMA nodes for gauntlet topology filtering (`min_sockets` accepted as alias) |
| `max_numa_nodes` | 1 | Maximum NUMA nodes for gauntlet topology filtering |
| `requires_smt` | `false` | Require SMT (threads > 1) topologies. On aarch64 the gauntlet ships only non-SMT presets, so any test with `requires_smt = true` is skipped entirely on that arch. |

The gauntlet skips presets that do not satisfy these constraints.
Multi-NUMA presets are excluded by default (`max_numa_nodes = 1`).
See [Gauntlet](../running-tests/gauntlet.md#constraint-filtering)
for filtering rules and
[Gauntlet Tests](gauntlet-tests.md#worked-example) for a worked
example.

### Execution

| Attribute | Default | Description |
|---|---|---|
| `auto_repro` | `true` | On scheduler crash, boot a second VM with probes attached. Set to `false` for fast iteration. |
| `replicas` | 1 | Number of times to run |
| `performance_mode` | `false` | Pin vCPUs to host cores, hugepages, NUMA mbind, RT scheduling, LLC exclusivity validation |
| `duration_s` | 2 | Per-scenario duration in seconds |
| `workers_per_cgroup` | 2 | Workers per cgroup |
| `expect_err` | `false` | Test expects `run_ktstr_test` to return `Err`; disables auto-repro |
| `bpf_map_write = CONST` | `None` | Rust const path to a `BpfMapWrite`; host writes this value to a BPF map after the scheduler loads |
| `host_only` | `false` | Run the test function directly on the host instead of inside a VM. Use for tests that need host tools (e.g. cargo, nested VMs) unavailable in the guest initramfs. |

See [Performance Mode](../concepts/performance-mode.md) for details on
what `performance_mode` enables, prerequisites, and validation behavior.

## Example with custom scheduler

Define the scheduler with `#[derive(Scheduler)]` (see
[Scheduler Definitions](scheduler-definitions.md)), then reference it
in `#[ktstr_test]`:

```rust,ignore
use ktstr::prelude::*;

#[derive(Scheduler)]
#[scheduler(
    name = "my_sched",
    binary = "scx_my_sched",
    topology(1, 2, 4, 1),
)]
#[allow(dead_code)]
enum MySchedFlag {
    #[flag(args = ["--enable-llc"])]
    Llc,
    #[flag(args = ["--enable-stealing"], requires = [Llc])]
    Steal,
}

#[ktstr_test(
    scheduler = MY_SCHED,
    not_starved = true,
    max_gap_ms = 5000,
    required_flags = [MySchedFlag::LLC],
)]
fn my_sched_basic(ctx: &Ctx) -> Result<AssertResult> {
    // Inherits 1n2l4c1t from MY_SCHED
    Ok(AssertResult::pass())
}
```

For the manual `FlagDecl` + builder pattern, see
[Scheduler Definitions: Manual definition](scheduler-definitions.md#manual-definition).
