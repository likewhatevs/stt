# Customize Verification

Override default [verification](../concepts/verification.md) thresholds
for schedulers that tolerate higher imbalance, different gap thresholds,
or relaxed event rates.

## Scheduler-level overrides

Define a `Scheduler` with custom verification:

```rust
const RELAXED: Scheduler = Scheduler::new("relaxed_sched")
    .binary(SchedulerSpec::Name("scx_relaxed"))
    .flags(&[&LLC_DECL, &BORROW_DECL])
    .verify(
        Verify::NONE
            .max_imbalance_ratio(5.0)    // tolerate 5:1 imbalance
            .max_fallback_rate(500.0)     // higher fallback rate ok
            .fail_on_stall(false)         // don't fail on stall
    );
```

These overrides sit between `Verify::default_checks()` and per-test
overrides in the merge chain.

## Per-test overrides via #[stt_test]

```rust
#[stt_test(
    sockets = 2,
    cores = 4,
    threads = 2,
    scheduler = RELAXED,
    not_starved = true,
    max_gap_ms = 5000,
    max_imbalance_ratio = 10.0,
    sustained_samples = 10,
)]
fn high_imbalance_test(ctx: &Ctx) -> Result<VerifyResult> {
    // ...
    Ok(VerifyResult::pass())
}
```

## Understanding not_starved

`not_starved = true` enables three distinct checks:

1. **Starvation**: any worker with zero work units fails.
2. **Fairness spread**: max - min runnable% across workers in a cgroup
   must be below `max_spread_pct` (default: 15% release, 35% debug).
3. **Scheduling gaps**: longest gap between work iterations must be
   below `max_gap_ms` (default: 2000ms release, 3000ms debug).

Each threshold can be overridden independently in `#[stt_test]`
attributes or in `Scheduler.verify`.

## Merge order

```text
Verify::default_checks()     <- baseline (not_starved, monitor defaults)
    .merge(&scheduler.verify) <- scheduler overrides
    .merge(&test.verify)      <- per-test attribute overrides
```

All fields use last-`Some`-wins -- a higher layer's `Some` replaces
the lower. A scheduler or test can disable a check by setting
`Some(false)` even if a lower layer enabled it.

## Using Verify directly in ops scenarios

```rust
fn my_scenario(ctx: &Ctx) -> Result<VerifyResult> {
    let verify = Verify::NONE
        .check_not_starved()
        .max_gap_ms(3000);

    let steps = vec![/* ... */];
    execute_steps_with(ctx, steps, Some(&verify))
}
```

`execute_steps_with` applies the given `Verify` for worker checks.
`execute_steps` (without `_with`) passes `None`, using the default
gap and spread thresholds from `verify_not_starved()` instead of
per-invocation `Verify` overrides.

See [Ops and Steps](../concepts/ops.md) for the full step execution
model.
