# The #[stt_test] Macro

`#[stt_test]` registers a function as an integration test that runs
inside a VM.

## Basic usage

```rust,ignore
use stt::prelude::*;

#[stt_test(sockets = 2, cores = 4, threads = 2)]
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
    .topology(2, 4, 1);

#[stt_test(scheduler = MY_SCHED)]
fn inherited_topo(ctx: &Ctx) -> Result<AssertResult> {
    // VM boots with 2 sockets, 4 cores, 1 thread (from MY_SCHED)
    Ok(AssertResult::pass())
}
```

The function must have signature
`fn(&stt::scenario::Ctx) -> anyhow::Result<stt::assert::AssertResult>`.

## What the macro generates

1. Renames the function to `__stt_inner_{name}`.
2. Registers it in the `STT_TESTS` distributed slice via linkme.
3. Emits a `#[test]` wrapper that calls `run_stt_test()`.

The `#[test]` wrapper boots a VM with the specified topology and runs
the function inside it.

## Attributes

All attributes are optional with defaults.

### Topology

| Attribute | Default | Description |
|---|---|---|
| `sockets` | inherited | Number of CPU sockets |
| `cores` | inherited | Cores per socket |
| `threads` | inherited | Threads per core |
| `memory_mb` | 2048 | VM memory in MB |

Each dimension independently inherits from `Scheduler.topology` when
a `scheduler` is specified and that dimension is not explicitly set.
Without a scheduler, unset dimensions use macro defaults (sockets=1,
cores=2, threads=1). See
[Default topology](scheduler-definitions.md#default-topology).

### Scheduler

| Attribute | Default | Description |
|---|---|---|
| `scheduler = CONST` | `Scheduler::EEVDF` | Rust const path to a `Scheduler` definition |
| `extra_sched_args = [...]` | `[]` | Extra CLI args for the scheduler |
| `watchdog_timeout_s` | 4 | scx watchdog override (seconds) |

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
#[stt_test(
    required_flags = ["llc", "borrow"],
    excluded_flags = ["no-ctrl"],
)]

// Path expressions (typed constants from derive)
#[stt_test(
    required_flags = [MySchedFlag::LLC, MySchedFlag::BORROW],
    excluded_flags = [MySchedFlag::NO_CTRL],
)]
```

These constrain `generate_profiles()`: `required_flags` are always
included, `excluded_flags` are never included. Invalid combinations
(e.g. `steal` without `llc`) are still rejected by dependency checks.
The gauntlet uses these attributes to determine which flag
profiles each test runs with. See
[Gauntlet](../running-tests/gauntlet.md#flag-profiles).

### Topology constraints

| Attribute | Default | Description |
|---|---|---|
| `min_sockets` | 1 | Minimum sockets for gauntlet topology filtering |
| `min_llcs` | 1 | Minimum LLCs for gauntlet topology filtering |
| `requires_smt` | `false` | Require SMT (threads > 1) topologies |
| `min_cpus` | 1 | Minimum total CPU count for gauntlet topology filtering |

The gauntlet skips topology presets that do not satisfy these
constraints. A test with `min_llcs = 3` will not run on `tiny-1llc`
(1 LLC) or `tiny-2llc` (2 LLCs). The gauntlet uses these
attributes to filter which presets each test runs on. See
[Gauntlet](../running-tests/gauntlet.md#topology-presets).

### Execution

| Attribute | Default | Description |
|---|---|---|
| `auto_repro` | `true` | Auto-repro on crash |
| `replicas` | 1 | Number of times to run |
| `performance_mode` | `false` | Pin vCPUs to host cores, hugepages, NUMA mbind, RT scheduling, LLC exclusivity validation |
| `duration_s` | 2 | Per-scenario duration in seconds |
| `workers_per_cgroup` | 2 | Workers per cgroup |
| `expect_err` | `false` | Test expects `run_stt_test` to return `Err`; disables auto-repro |
| `bpf_map_write = CONST` | `None` | Rust const path to a `BpfMapWrite`; host writes this value to a BPF map after the scheduler loads |

See [Performance Mode](../concepts/performance-mode.md) for details on
what `performance_mode` enables, prerequisites, and validation behavior.

## Example with custom scheduler

Define the scheduler with `#[derive(Scheduler)]` (see
[Scheduler Definitions](scheduler-definitions.md)), then reference it
in `#[stt_test]`:

```rust,ignore
use stt::prelude::*;

#[derive(Scheduler)]
#[scheduler(
    name = "my_sched",
    binary = "scx_my_sched",
    topology(2, 4, 1),
)]
#[allow(dead_code)]
enum MySchedFlag {
    #[flag(args = ["--enable-llc"])]
    Llc,
    #[flag(args = ["--enable-stealing"], requires = [Llc])]
    Steal,
}

#[stt_test(
    scheduler = MY_SCHED,
    not_starved = true,
    max_gap_ms = 5000,
    required_flags = [MySchedFlag::LLC],
)]
fn my_sched_basic(ctx: &Ctx) -> Result<AssertResult> {
    // Inherits 2s4c1t from MY_SCHED
    Ok(AssertResult::pass())
}
```

For the manual `FlagDecl` + builder pattern, see
[Scheduler Definitions: Manual definition](scheduler-definitions.md#manual-definition).
