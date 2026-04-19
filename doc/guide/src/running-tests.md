# Running Tests

Tests run via `cargo nextest run`, which boots KVM virtual machines
for each `#[ktstr_test]` entry.

## Quick reference

```sh
# Run all tests
cargo nextest run --workspace

# Run a specific test
cargo nextest run -E 'test(sched_basic_proportional)'

# Run ignored gauntlet tests
cargo nextest run --run-ignored ignored-only -E 'test(gauntlet/)'
```

## Flags

Define flags via `#[derive(Scheduler)]` with `#[flag(...)]` attributes.
Use `required_flags` and `excluded_flags` in `#[ktstr_test]` to constrain
which flag profiles a test runs under.

ktstr includes built-in flags (`llc`, `borrow`, `steal`, `rebal`,
`reject-pin`, `no-ctrl`) for the internal catalog. See
[Flags](concepts/flags.md) for details.

## Budget-based test selection

Set `KTSTR_BUDGET_SECS` to select the subset of tests that maximizes
feature coverage within a time budget. Useful for CI pipelines or
quick smoke tests.

```sh
# Run the best 5 minutes of tests
KTSTR_BUDGET_SECS=300 cargo nextest run --workspace

# Budget applies to gauntlet variants too
KTSTR_BUDGET_SECS=600 cargo nextest run --run-ignored all
```

The selector encodes each test as a bitset of properties (scheduler,
flags, topology class, SMT, workload characteristics) and greedily
picks tests with the highest marginal coverage per estimated second.
Duration estimates account for VM boot overhead based on vCPU count.

A summary is printed to stderr during `--list`:

```
ktstr budget: 42/1200 tests, 295/300s used, 38/38 configurations covered
```

When `KTSTR_BUDGET_SECS` is not set, all tests are listed as usual.

## Custom scheduler

Define a `Scheduler` with `SchedulerSpec::Discover` or
`SchedulerSpec::Path` to test a pre-built scheduler binary:

```rust,ignore
const MY_SCHED: Scheduler = Scheduler::new("my_sched")
    .binary(SchedulerSpec::Discover("scx_my_sched"));
```

The binary is injected into the VM's initramfs and started before
scenarios run. See [Test a New Scheduler](recipes/test-new-scheduler.md)
for the full end-to-end workflow.
