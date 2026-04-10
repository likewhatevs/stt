# Writing Tests

Tests are Rust functions annotated with `#[stt_test]`. Each test
boots a KVM VM, runs the scenario inside it, and evaluates results
on the host.

```rust,ignore
use stt::prelude::*;

#[stt_test(sockets = 1, cores = 2, threads = 1)]
fn my_test(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        CgroupDef::named("cg_0").workers(2),
        CgroupDef::named("cg_1").workers(2),
    ])
}
```

Run with `cargo nextest run`. See
[Getting Started](getting-started.md) for setup and
[The #\[stt_test\] Macro](writing-tests/stt-test-macro.md) for all
available attributes.
