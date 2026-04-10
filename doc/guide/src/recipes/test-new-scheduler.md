# Test a New Scheduler

End-to-end workflow: define a scheduler, write tests, run them.

## 1. Define the scheduler

Use `#[derive(stt::Scheduler)]` on an enum of flags:

```rust,ignore
use stt::prelude::*;

#[derive(stt::Scheduler)]
#[scheduler(
    name = "my_scheduler",
    binary = "scx_my_scheduler",
    topology(2, 4, 2),
)]
#[allow(dead_code)]
enum MySchedulerFlag {
    #[flag(args = ["--enable-llc"])]
    Llc,
    #[flag(args = ["--enable-stealing"], requires = [Llc])]
    Steal,
}
```

This generates `const MY_SCHEDULER: Scheduler` and typed flag
constants (`MySchedulerFlag::LLC`, `MySchedulerFlag::STEAL`).

## 2. Write integration tests

Tests inherit the scheduler's topology. Override with explicit
`sockets`, `cores`, or `threads` when needed.

```rust,ignore
use stt::prelude::*;

#[stt_test(scheduler = MY_SCHEDULER)]
fn basic_steady(ctx: &Ctx) -> Result<AssertResult> {
    // Inherits 2s4c2t from MY_SCHEDULER
    scenarios::steady(ctx)
}

#[stt_test(
    scheduler = MY_SCHEDULER,
    required_flags = [MySchedulerFlag::LLC],
)]
fn llc_aware_test(ctx: &Ctx) -> Result<AssertResult> {
    scenarios::steady_llc(ctx)
}
```

## 3. Run

```sh
cargo nextest run
```

See [The #\[stt_test\] Macro](../writing-tests/stt-test-macro.md) for
all available attributes and
[Scheduler Definitions](../writing-tests/scheduler-definitions.md) for
the full `Scheduler` type and derive macro.
