# The #[stt_test] Macro

`#[stt_test]` registers a function as an integration test that runs
inside a VM.

## Basic usage

```rust
use stt::stt_test;
use stt::scenario::Ctx;
use stt::verify::VerifyResult;

#[stt_test(sockets = 2, cores = 4, threads = 2)]
fn my_test(ctx: &Ctx) -> anyhow::Result<VerifyResult> {
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
| `check_not_starved` | `false` | Enable starvation (zero work units), fairness spread, and scheduling gap checks |
| `check_isolation` | `false` | Enable cpuset isolation check (workers must stay on assigned CPUs) |
| `max_gap_ms` | inherited | Max scheduling gap threshold |
| `max_spread_pct` | inherited | Max fairness spread threshold |
| `max_imbalance` | inherited | Monitor imbalance ratio |
| `max_dsq_depth` | inherited | Monitor DSQ depth |
| `fail_on_stall` | inherited | Fail on stall detection |
| `sustained_samples` | inherited | Sample window for sustained violations |
| `max_fallback_rate` | inherited | Max fallback event rate |
| `max_keep_last_rate` | inherited | Max keep-last event rate |

`check_not_starved` enables three distinct checks: starvation (any
worker with zero work units), fairness spread (max-min runnable% below
`max_spread_pct`), and scheduling gaps (longest gap below `max_gap_ms`).
Each threshold can be overridden independently. See
[Customize Verification](../recipes/custom-verification.md) for
override examples and [Verification](../concepts/verification.md) for
the merge chain.

### Execution

| Attribute | Default | Description |
|---|---|---|
| `auto_repro` | `true` | Auto-repro on crash |
| `replicas` | 1 | Number of times to run |
| `performance_mode` | `false` | Pin vCPUs to host cores, hugepages |

See [Performance Mode](../concepts/performance-mode.md) for details on
what `performance_mode` enables, prerequisites, and validation behavior.

## Example with custom scheduler

```rust
use stt::stt_test;
use stt::scenario::Ctx;
use stt::test_support::Scheduler;
use stt::verify::VerifyResult;

const MITOSIS: Scheduler = Scheduler::new("mitosis")
    .binary(stt::test_support::SchedulerSpec::Name("scx_mitosis"))
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
    check_not_starved = true,
    max_gap_ms = 5000,
)]
fn mitosis_basic(ctx: &Ctx) -> anyhow::Result<VerifyResult> {
    // Test logic here
    Ok(VerifyResult::pass())
}
```
