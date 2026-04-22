# Benchmarking and Negative Tests

Recipes for writing tests that check scheduler performance gates
(positive tests) and confirm that degraded schedulers fail those gates
(negative tests).

## Using Assert for checking

`Assert` carries all checking thresholds. Every field is `Option`;
`None` means "inherit from parent layer."

- In the merge chain: `Assert::default_checks()` -> `Scheduler.assert`
  -> per-test `#[ktstr_test]` attributes. Use with `execute_steps_with()`
  for ops-based scenarios. See
  [Checking](../concepts/checking.md#merge-layers).

- For direct report checking: call `Assert::assert_cgroup(reports, cpuset)`.

```rust,ignore
let a = Assert::default_checks().max_gap_ms(500);
let result = a.assert_cgroup(&reports, None);
```

## Positive benchmarking test

Check that a scheduler passes performance gates under
`performance_mode`. Use `#[ktstr_test]` with Assert thresholds:

```rust,ignore
use ktstr::prelude::*;

const MY_SCHED: Scheduler = Scheduler::new("my_sched")
    .binary(SchedulerSpec::Discover("scx_my_sched"))
    .topology(1, 1, 2, 1);

#[ktstr_test(
    scheduler = MY_SCHED,
    performance_mode = true,
    duration_s = 3,
    workers_per_cgroup = 2,
    sustained_samples = 15,
)]
fn perf_positive(ctx: &Ctx) -> Result<AssertResult> {
    let checks = Assert::default_checks()
        .min_iteration_rate(5000.0)
        .max_gap_ms(500);
    let steps = vec![Step::with_defs(
        vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)],
        HoldSpec::FULL,
    )];
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

Check that intentionally degraded scheduling fails the same gates.
This confirms that the gates actually catch regressions rather than
passing vacuously.

Use `expect_err = true` on `#[ktstr_test]` to assert that the test
fails. The macro wraps the test with `assert!(result.is_err())` and
disables auto-repro automatically.

```rust,ignore
use ktstr::prelude::*;

const MY_SCHED: Scheduler = Scheduler::new("my_sched")
    .binary(SchedulerSpec::Discover("scx_my_sched"))
    .topology(1, 1, 2, 1);

#[ktstr_test(
    scheduler = MY_SCHED,
    performance_mode = true,
    duration_s = 5,
    workers_per_cgroup = 4,
    extra_sched_args = ["--fail-verify"],
    expect_err = true,
)]
fn perf_negative(ctx: &Ctx) -> Result<AssertResult> {
    let checks = Assert::default_checks().max_gap_ms(50);
    let steps = vec![Step::with_defs(
        vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)],
        HoldSpec::FULL,
    )];
    execute_steps_with(ctx, steps, Some(&checks))
}
```

Key points:
- `expect_err = true` tells the harness to assert failure and disable
  auto-repro.
- `extra_sched_args = [...]` passes CLI args to the scheduler
  binary. `"--fail-verify"` is a real knob that the test fixture
  scheduler `scx-ktstr` exposes to force a verifier failure (see
  `scx-ktstr/src/main.rs` and `scx-ktstr/src/bpf/main.bpf.c`);
  substitute your own scheduler's equivalent of the behaviour you
  want to exercise in a negative test.
- The test function returns the scenario result normally; the harness
  checks that it produces an error.

## Metric extraction from stderr

`OutputFormat::Json` and `OutputFormat::LlmExtract` read the
payload's STDOUT as the primary stream, then fall back to STDERR if
stdout is empty or yields no metrics. Some benchmarks emit their
numbers only to stderr — `schbench`, for example, writes its
`Wakeup Latencies percentiles` / `Request Latencies percentiles`
blocks via `fprintf(stderr, ...)` and leaves stdout blank. The
fallback keeps those benchmarks usable without a redirect.

Consequence: a payload that writes mixed output to both streams
will have metrics extracted from stdout **only**, because the
fallback fires solely when the primary stream is empty or yields
nothing parseable. If you care about stderr-side numbers for a
stdout-emitting binary, redirect stderr into stdout at the payload
layer (`extra_args = ["-c", "cmd 2>&1"]` for shell-wrapped
invocations, or whatever equivalent the binary supports).

`stress-ng` is the mirror trap: progress / per-stressor summaries
go to stderr and stdout is blank, so the fallback sees stress-ng's
prose. `OutputFormat::Json` returns zero metrics (stderr is prose,
not JSON); `OutputFormat::LlmExtract` may extract numbers from the
fallback but results depend on the local model's tolerance for
that prose format. Keep `OutputFormat::ExitCode` for stress-ng
unless you are prepared for that tradeoff.

## Declarative include_files on Payload

Payloads that need host binaries or fixtures in the guest initramfs
can declare them on the `Payload` itself instead of relying on the
CLI `-i` / `--include-files` flag at every invocation. The specs
are resolved at `run_ktstr_test` time through the same
`resolve_include_files` pipeline the CLI uses — bare names are
searched in the host's `PATH`, explicit paths must exist, and
directories are walked recursively.

Declare per-Payload via the `#[include_files(...)]` attribute on
`#[derive(Payload)]`:

```rust,ignore
use ktstr::prelude::*;

#[derive(Payload)]
#[payload(binary = "fio")]
#[include_files("fio", "bench-helper")]
#[metric(name = "iops", polarity = "higher_is_better", unit = "ops/s")]
struct FioPayload;
```

The generated `FIO` const carries `include_files: &["fio",
"bench-helper"]`. When any `#[ktstr_test]` uses `FIO` as a payload
or workload, those files get resolved and packed into the initramfs
automatically — no `-i` flag needed on the CLI.

Test-level extras that don't belong on any specific payload go on
the `#[ktstr_test]` attribute directly:

```rust,ignore
#[ktstr_test(
    scheduler = MY_SCHED,
    payload = FIO,
    extra_include_files = ["test-fixtures/workload.json"],
)]
fn fio_with_fixture(ctx: &Ctx) -> Result<AssertResult> {
    // test body
    # Ok(AssertResult::pass())
}
```

The declarative set (scheduler's `include_files` + payload's +
workloads' + `extra_include_files`) is additive with the CLI
`-i` flag — it does NOT replace CLI input. Running
`ktstr exec --test fio_with_fixture -i extra-binary` packs
`extra-binary` alongside the declaratively-declared files; the
union is deduped on identical `(archive_path, host_path)` pairs.
Two declarations that resolve to the same archive slot with
different host paths surface as a hard error with both host paths
in the diagnostic, rather than one silently overwriting the other.
