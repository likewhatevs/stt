# Running Tests

Tests run via `cargo nextest run`, which boots KVM virtual machines
for each `#[stt_test]` entry.

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

Flags enable scheduler features. Declare them in the `Scheduler`
definition via `FlagDecl` structs. Use `required_flags` and
`excluded_flags` in `#[stt_test]` to constrain which flag
profiles a test runs under.

Available flags: `llc`, `borrow`, `steal`, `rebal`, `reject-pin`,
`no-ctrl`. `steal` requires `llc` -- this is enforced automatically.

See [Flags](concepts/flags.md) for details on flag declarations
and profile generation.

## Custom scheduler

Define a `Scheduler` with `SchedulerSpec::Name` or
`SchedulerSpec::Path` to test a pre-built scheduler binary:

```rust,ignore
const MY_SCHED: Scheduler = Scheduler::new("scx_my_scheduler")
    .binary(SchedulerSpec::Name("scx_my_scheduler"));
```

The binary is injected into the VM's initramfs and started before
scenarios run. See [Test a New Scheduler](recipes/test-new-scheduler.md)
for the full end-to-end workflow.
