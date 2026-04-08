# Test a New Scheduler

End-to-end workflow: define a scheduler, write tests, run them.

## 1. Define the scheduler

```rust,ignore
use stt::prelude::*;
use stt::scenario::flags::*;

const MY_SCHED: Scheduler = Scheduler::new("my_scheduler")
    .binary(SchedulerSpec::Name("scx_my_scheduler"))
    .flags(&[&LLC_DECL, &BORROW_DECL, &STEAL_DECL, &REBAL_DECL]);
```

## 2. Write integration tests

```rust,ignore
use stt::prelude::*;
use stt::scenario::*;

#[stt_test(sockets = 2, cores = 4, threads = 2, scheduler = MY_SCHED)]
fn basic_proportional(ctx: &Ctx) -> Result<AssertResult> {
    let wl = dfl_wl(ctx);
    let (handles, _guard) = setup_cgroups(ctx, 2, &wl)?;
    std::thread::sleep(ctx.duration);
    Ok(collect_all(handles))
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
