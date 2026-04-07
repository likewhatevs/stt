# Test a New Scheduler

End-to-end workflow: build a scheduler, run it through stt, analyze
results.

## 1. Run a single scenario

`-p` builds the scheduler and injects it into the VM:

```sh
cargo stt vm -p scx_my_scheduler \
  --sockets 2 --cores 4 --threads 2 \
  -- cgroup_steady --flags=llc,borrow --duration-s 30
```

Use `--scheduler-bin` instead of `-p` to skip the build and point
at a pre-built binary directly.

## 2. Run the gauntlet

```sh
cargo stt vm --gauntlet -p scx_my_scheduler --parallel 4
```

## 3. Save a baseline

```sh
cargo stt vm --gauntlet -p scx_my_scheduler \
  --save-baseline my-scheduler-v1.json
```

## 4. Compare against baseline after changes

```sh
cargo stt vm --gauntlet -p scx_my_scheduler \
  --compare my-scheduler-v1.json
```

See [Baselines](../running-tests/baselines.md) for details on save and
compare.

## 5. Run integration tests through the gauntlet

Once you have `#[stt_test]` tests, run them across topology presets:

```sh
cargo stt gauntlet --parallel 4
```

## 6. Write integration tests

Define the scheduler for `#[stt_test]`:

```rust,ignore
use stt::prelude::*;
use stt::scenario::flags::*;

const MY_SCHED: Scheduler = Scheduler::new("my_scheduler")
    .binary(SchedulerSpec::Name("scx_my_scheduler"))
    .flags(&[&LLC_DECL, &BORROW_DECL, &STEAL_DECL, &REBAL_DECL]);
```

Write a test:

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

Run with `cargo stt test`.

See [The #\[stt_test\] Macro](../writing-tests/stt-test-macro.md) for
all available attributes and
[Scheduler Definitions](../writing-tests/scheduler-definitions.md) for
the full `Scheduler` builder API.
