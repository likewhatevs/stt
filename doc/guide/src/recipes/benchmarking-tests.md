# Benchmarking and Negative Tests

Recipes for writing tests that verify scheduler performance gates
(positive tests) and confirm that degraded schedulers fail those gates
(negative tests).

## Assert vs AssertPlan

`Assert` and `AssertPlan` both carry verification thresholds but serve
different roles:

- **`Assert`** -- `Option`-based fields. Used in the three-layer merge
  chain: `Assert::default_checks()` -> `Scheduler.assert` ->
  per-test `#[stt_test]` attributes. Use with `execute_steps_with()`
  for ops-based scenarios. See
  [Verification](../concepts/verification.md#merge-layers).

- **`AssertPlan`** -- concrete fields (not `Option`). Used in custom
  scenarios that call `plan.assert_cgroup(reports, cpuset)` directly.
  `Assert::worker_plan()` extracts an `AssertPlan` from an `Assert`.

When writing ops-based tests, prefer `Assert` with
`execute_steps_with()`. When writing raw custom scenarios with manual
report collection, use `AssertPlan`.

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

Negative tests cannot use `#[stt_test]` directly because they need to
assert that the test *fails*. Instead, construct an `SttTestEntry`
manually and call `run_stt_test()`:

```rust,ignore
use stt::prelude::*;
use stt::test_support::{SttTestEntry, run_stt_test};

const MY_SCHED: Scheduler = Scheduler::new("my_sched")
    .binary(SchedulerSpec::Name("scx_my_sched"));

#[test]
fn perf_negative() {
    fn scenario(ctx: &stt::scenario::Ctx) -> Result<AssertResult> {
        use stt::scenario::ops::execute_steps_with;
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

    #[linkme::distributed_slice(stt::test_support::STT_TESTS)]
    #[linkme(crate = linkme)]
    static __STT_ENTRY: SttTestEntry = SttTestEntry {
        name: "perf_negative",
        func: scenario,
        sockets: 1,
        cores: 2,
        threads: 1,
        memory_mb: 2048,
        scheduler: &MY_SCHED,
        auto_repro: false,
        replicas: 1,
        assert: stt::assert::Assert::NONE,
        extra_sched_args: &["--degrade"],
        required_flags: &[],
        excluded_flags: &[],
        min_sockets: 1,
        min_llcs: 1,
        requires_smt: false,
        min_cpus: 1,
        watchdog_timeout_jiffies: 0,
        bpf_map_write: None,
        performance_mode: true,
        super_perf_mode: false,
        duration_s: 5,
        workers_per_cgroup: 4,
    };

    let result = run_stt_test(&__STT_ENTRY);
    assert!(
        result.is_err(),
        "degraded scheduler should fail gates, but test passed"
    );
    let err_msg = format!("{:#}", result.unwrap_err());
    assert!(
        err_msg.contains("stuck"),
        "error should mention scheduling gap: {err_msg}"
    );
}
```

Key points:
- `extra_sched_args: &["--degrade"]` passes a flag that makes the
  scheduler intentionally slow (the scheduler must support this flag).
- `assert!(result.is_err())` confirms the test fails as expected.
- Check error message content to verify the *right* gate tripped.
- `auto_repro: false` -- no point auto-reproing an intentional failure.
- Manual `SttTestEntry` is needed because `#[stt_test]` expects the
  scenario function to return success.
