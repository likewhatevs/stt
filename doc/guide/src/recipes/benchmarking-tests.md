# Benchmarking and Negative Tests

Recipes for writing tests that verify scheduler performance gates
(positive tests) and confirm that degraded schedulers fail those gates
(negative tests).

## Using Assert for verification

`Assert` carries all verification thresholds. Every field is `Option`;
`None` means "inherit from parent layer."

- In the merge chain: `Assert::default_checks()` -> `Scheduler.assert`
  -> per-test `#[stt_test]` attributes. Use with `execute_steps_with()`
  for ops-based scenarios. See
  [Verification](../concepts/verification.md#merge-layers).

- For direct report checking: call `Assert::assert_cgroup(reports, cpuset)`.

```rust,ignore
let a = Assert::default_checks().max_gap_ms(500);
let result = a.assert_cgroup(&reports, None);
```

## Positive benchmarking test

Verify that a scheduler passes performance gates under
`performance_mode`. Use `#[stt_test]` with Assert thresholds:

```rust,ignore
use stt::prelude::*;

const MY_SCHED: Scheduler = Scheduler::new("my_sched")
    .binary(SchedulerSpec::Name("scx_my_sched"));

#[stt_test(
    scheduler = MY_SCHED,
    sockets = 1,
    cores = 2,
    threads = 1,
    performance_mode = true,
    duration_s = 3,
    workers_per_cgroup = 2,
    sustained_samples = 15,
)]
fn perf_positive(ctx: &Ctx) -> Result<AssertResult> {
    use stt::scenario::ops::execute_steps_with;
    let checks = Assert::default_checks()
        .min_iteration_rate(5000.0)
        .max_gap_ms(500);
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup),
        ].into(),
        ops: vec![],
        hold: HoldSpec::Frac(1.0),
    }];
    execute_steps_with(ctx, steps, Some(&checks))
}
```

Key points:
- `performance_mode = true` pins vCPUs and uses hugepages for
  deterministic measurements.
- `Assert::default_checks()` starts from the standard baseline.
- Chain `.min_iteration_rate()`, `.max_gap_ms()`, or
  `.max_p99_wake_latency_ns()` to set gates.
- `execute_steps_with()` applies the `Assert` during worker checks.

## Negative test pattern

Verify that intentionally degraded scheduling fails the same gates.
This confirms that the gates actually catch regressions rather than
passing vacuously.

Use `expect_err = true` on `#[stt_test]` to assert that the test
fails. The macro wraps the test with `assert!(result.is_err())` and
disables auto-repro automatically.

```rust,ignore
use stt::prelude::*;

const MY_SCHED: Scheduler = Scheduler::new("my_sched")
    .binary(SchedulerSpec::Name("scx_my_sched"));

#[stt_test(
    scheduler = MY_SCHED,
    sockets = 1,
    cores = 2,
    threads = 1,
    performance_mode = true,
    duration_s = 5,
    workers_per_cgroup = 4,
    extra_sched_args = ["--degrade"],
    expect_err = true,
)]
fn perf_negative(ctx: &Ctx) -> Result<AssertResult> {
    let checks = Assert::default_checks().max_gap_ms(50);
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup),
        ].into(),
        ops: vec![],
        hold: HoldSpec::Frac(1.0),
    }];
    execute_steps_with(ctx, steps, Some(&checks))
}
```

Key points:
- `expect_err = true` tells the harness to assert failure and disable
  auto-repro.
- `extra_sched_args = ["--degrade"]` passes a flag that makes the
  scheduler intentionally slow (the scheduler must support this flag).
- The test function returns the scenario result normally; the harness
  checks that it produces an error.
