# Test a New Scheduler

End-to-end workflow: define a scheduler, write tests, run them.

## 1. Define the scheduler

```rust,ignore
use stt::prelude::*;

static MY_LLC: FlagDecl = FlagDecl {
    name: "llc",
    args: &["--enable-llc"],
    requires: &[],
};

static MY_STEAL: FlagDecl = FlagDecl {
    name: "steal",
    args: &["--enable-stealing"],
    requires: &[&MY_LLC],
};

const MY_SCHED: Scheduler = Scheduler::new("my_scheduler")
    .binary(SchedulerSpec::Name("scx_my_scheduler"))
    .flags(&[&MY_LLC, &MY_STEAL])
    .topology(2, 4, 2);
```

## 2. Write integration tests

Tests inherit the scheduler's topology. Override with explicit
`sockets`, `cores`, or `threads` when needed.

```rust,ignore
use stt::prelude::*;
use stt::scenario::*;

#[stt_test(scheduler = MY_SCHED)]
fn basic_proportional(ctx: &Ctx) -> Result<AssertResult> {
    // Inherits 2s4c2t from MY_SCHED
    let wl = dfl_wl(ctx);
    let (handles, _guard) = setup_cgroups(ctx, 2, &wl)?;
    std::thread::sleep(ctx.duration);
    Ok(collect_all(handles, &ctx.assert))
}
```

## 3. Run

```sh
cargo nextest run
```

See [The #\[stt_test\] Macro](../writing-tests/stt-test-macro.md) for
all available attributes and
[Scheduler Definitions](../writing-tests/scheduler-definitions.md) for
the full `Scheduler` builder API.
