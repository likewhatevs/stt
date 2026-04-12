# Test a New Scheduler

End-to-end workflow: define a scheduler, write tests, run them.

## 1. Define the scheduler

Use `#[derive(Scheduler)]` on an enum of flags:

```rust,ignore
use ktstr::prelude::*;

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
```

This generates `const MY_SCHED: Scheduler` and typed flag
constants (`MySchedFlag::LLC`, `MySchedFlag::STEAL`).

## 2. Write integration tests

Tests inherit the scheduler's topology. Override with explicit
`sockets`, `cores`, or `threads` when needed.

```rust,ignore
use ktstr::prelude::*;

#[ktstr_test(scheduler = MY_SCHED)]
fn basic_steady(ctx: &Ctx) -> Result<AssertResult> {
    // Inherits 2s4c1t from MY_SCHED
    scenarios::steady(ctx)
}

#[ktstr_test(
    scheduler = MY_SCHED,
    required_flags = [MySchedFlag::LLC],
)]
fn llc_aware_test(ctx: &Ctx) -> Result<AssertResult> {
    scenarios::steady_llc(ctx)
}
```

## 3. Build a kernel

Build a kernel with sched_ext support:

```sh
cargo ktstr kernel build
```

See [Getting Started: Build a kernel](../getting-started.md#build-a-kernel)
for version selection and local source builds.

## 4. Run

```sh
cargo nextest run
```

## 5. Check BPF complexity (optional)

Collect per-program verifier statistics:

```sh
cargo ktstr verifier --scheduler scx_my_sched
```

See [BPF Verifier](../running-tests/verifier.md) for output format and
cycle collapse.

## 6. Debug failures

Boot an interactive shell with the scheduler binary:

```sh
cargo ktstr shell -i ./target/debug/scx_my_sched
```

Inside the guest, run `/include-files/scx_my_sched` manually to
inspect behavior. See
[cargo-ktstr shell](../running-tests/cargo-ktstr.md#shell) for all
flags.

See [The #\[ktstr_test\] Macro](../writing-tests/ktstr-test-macro.md) for
all available attributes and
[Scheduler Definitions](../writing-tests/scheduler-definitions.md) for
the full `Scheduler` type and derive macro.
