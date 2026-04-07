# The #[stt_test] Macro

`#[stt_test]` registers a function as an integration test that runs
inside a VM.

## Basic usage

```rust,ignore
use stt::prelude::*;

#[stt_test(sockets = 2, cores = 4, threads = 2)]
fn my_test(ctx: &Ctx) -> Result<VerifyResult> {
    // ctx provides cgroup manager, topology, duration, etc.
    Ok(VerifyResult::pass())
}
```

The function must have signature
`fn(&stt::scenario::Ctx) -> anyhow::Result<stt::verify::VerifyResult>`.

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
| `sockets` | 1 | Number of CPU sockets |
| `cores` | 2 | Cores per socket |
| `threads` | 1 | Threads per core |
| `memory_mb` | 2048 | VM memory in MB |

### Scheduler

| Attribute | Default | Description |
|---|---|---|
| `scheduler = CONST` | `Scheduler::EEVDF` | Rust const path to a `Scheduler` definition |
| `extra_sched_args = [...]` | `[]` | Extra CLI args for the scheduler |
| `watchdog_timeout_jiffies` | 0 | scx watchdog override |

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

`not_starved = true` enables three distinct checks: starvation (any
worker with zero work units), fairness spread (max-min runnable% below
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

Values are string arrays of flag names:

```rust,ignore
#[stt_test(
    required_flags = ["llc", "borrow"],
    excluded_flags = ["no-ctrl"],
)]
```

These constrain `generate_profiles()`: `required_flags` are always
included, `excluded_flags` are never included. Invalid combinations
(e.g. `steal` without `llc`) are still rejected by dependency checks.
`cargo stt gauntlet` reads these attributes to determine which flag
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
(1 LLC) or `tiny-2llc` (2 LLCs). `cargo stt gauntlet` reads these
attributes to filter which presets each test runs on. See
[Gauntlet](../running-tests/gauntlet.md#topology-presets).

### Execution

| Attribute | Default | Description |
|---|---|---|
| `auto_repro` | `true` | Auto-repro on crash |
| `replicas` | 1 | Number of times to run |
| `performance_mode` | `false` | Pin vCPUs to host cores, hugepages |
| `duration_s` | 0 | Per-scenario duration override (0 = use default 2s) |
| `workers_per_cell` | 0 | Workers per cgroup override (0 = use default 2) |

See [Performance Mode](../concepts/performance-mode.md) for details on
what `performance_mode` enables, prerequisites, and validation behavior.

## Example with custom scheduler

```rust,ignore
use stt::prelude::*;

const MITOSIS: Scheduler = Scheduler::new("mitosis")
    .binary(SchedulerSpec::Name("scx_mitosis"))
    .flags(&[
        &stt::scenario::flags::LLC_DECL,
        &stt::scenario::flags::BORROW_DECL,
        &stt::scenario::flags::STEAL_DECL,
        &stt::scenario::flags::REBAL_DECL,
        &stt::scenario::flags::REJECT_PIN_DECL,
        &stt::scenario::flags::NO_CTRL_DECL,
    ]);

#[stt_test(
    sockets = 2,
    cores = 4,
    threads = 2,
    scheduler = MITOSIS,
    not_starved = true,
    max_gap_ms = 5000,
)]
fn mitosis_basic(ctx: &Ctx) -> Result<VerifyResult> {
    // Test logic here
    Ok(VerifyResult::pass())
}
```
