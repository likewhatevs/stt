# Customize Verification

Override default [verification](../concepts/verification.md) thresholds
for schedulers that tolerate higher imbalance, different gap thresholds,
or relaxed event rates.

## Scheduler-level overrides

Define a `Scheduler` with custom verification:

```rust,ignore
use ktstr::prelude::*;

static MY_LLC: FlagDecl = FlagDecl {
    name: "llc",
    args: &["--enable-llc"],
    requires: &[],
};

static MY_BORROW: FlagDecl = FlagDecl {
    name: "borrow",
    args: &["--enable-borrowing"],
    requires: &[],
};

const RELAXED: Scheduler = Scheduler::new("relaxed")
    .binary(SchedulerSpec::Name("scx_relaxed"))
    .flags(&[&MY_LLC, &MY_BORROW])
    .assert(
        Assert::NONE
            .max_imbalance_ratio(5.0)    // tolerate 5:1 imbalance
            .max_fallback_rate(500.0)     // higher fallback rate ok
            .fail_on_stall(false)         // don't fail on stall
    );
```

These overrides sit between `Assert::default_checks()` and per-test
overrides in the merge chain.

## Per-test overrides via #[ktstr_test]

```rust,ignore
#[ktstr_test(
    scheduler = RELAXED,
    not_starved = true,
    max_gap_ms = 5000,
    max_imbalance_ratio = 10.0,
    sustained_samples = 10,
)]
fn high_imbalance_test(ctx: &Ctx) -> Result<AssertResult> {
    // Inherits topology from RELAXED
    Ok(AssertResult::pass())
}
```

## Understanding not_starved

`not_starved = true` enables starvation, fairness spread, and
scheduling gap checks. Each threshold can be overridden independently.
See [Verification: Worker checks](../concepts/verification.md#worker-checks)
for details and default thresholds.

## Merge order

Three-layer merge with last-`Some`-wins semantics. See
[Verification: Merge layers](../concepts/verification.md#merge-layers).

## Using Assert directly in ops scenarios

```rust,ignore
fn my_scenario(ctx: &Ctx) -> Result<AssertResult> {
    let assertions = Assert::NONE
        .check_not_starved()
        .max_gap_ms(3000);

    let steps = vec![/* ... */];
    execute_steps_with(ctx, steps, Some(&assertions))
}
```

`execute_steps_with` applies the given `Assert` for worker checks.
`execute_steps` (without `_with`) passes `None`, falling back to
`ctx.assert` (the merged three-layer config: `default_checks` ->
scheduler -> per-test).

See [Ops and Steps](../concepts/ops.md) for the full step execution
model.
