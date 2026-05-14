# Test a New Scheduler

End-to-end workflow: define a scheduler, write tests, run them.

## 1. Define the scheduler

Use `declare_scheduler!` to register a scheduler in the
`KTSTR_SCHEDULERS` distributed slice. The verifier sweep picks
it up automatically.

```rust,ignore
use ktstr::declare_scheduler;
use ktstr::prelude::*;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    topology = (1, 2, 4, 1),
    kernels = ["6.14", "6.15..=7.0"],
    sched_args = ["--exit-dump-len", "1048576"],
});
```

The macro generates `pub static MY_SCHED: Scheduler` plus a
private `linkme` registration so `cargo ktstr verifier`
discovers the scheduler automatically. Tests reference the
bare `MY_SCHED` ident via
`#[ktstr_test(scheduler = MY_SCHED)]`.

See [Scheduler Definitions](../writing-tests/scheduler-definitions.md)
for every supported field.

## 2. Write integration tests

Tests inherit the scheduler's topology. Override with explicit
`llcs`, `cores`, or `threads` when needed.

```rust,ignore
use ktstr::prelude::*;

#[ktstr_test(scheduler = MY_SCHED)]
fn basic_steady(ctx: &Ctx) -> Result<AssertResult> {
    // Inherits 1n2l4c1t from MY_SCHED
    scenarios::steady(ctx)
}

#[ktstr_test(scheduler = MY_SCHED, threads = 2)]
fn smt_steady(ctx: &Ctx) -> Result<AssertResult> {
    // Inherits llcs=2, cores=4; overrides threads to exercise SMT
    scenarios::steady(ctx)
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
cargo ktstr test --kernel ../linux
```

## 5. Check BPF complexity (optional)

Collect per-program verifier statistics across the declared
kernels and accepted topology presets:

```sh
# Use the kernel auto-discovered via KTSTR_KERNEL / cache.
cargo ktstr verifier

# Pin to a specific kernel build.
cargo ktstr verifier --kernel ../linux

# Sweep across multiple kernels. Each scheduler's
# `kernels = [...]` declaration acts as a per-scheduler filter on
# the operator-supplied set; an empty (or omitted) `kernels` field
# means the scheduler runs against every kernel in the sweep.
cargo ktstr verifier --kernel 6.14 --kernel 7.0
```

See [BPF Verifier](../running-tests/verifier.md) for output
format, cycle collapse, and the cell-name → kernel matching
contract.

## 6. Manage the kernel cache

Cached kernel images accumulate under
`$XDG_CACHE_HOME/ktstr/kernels/`. Keep a handful of recent
builds and drop the rest when disk pressure grows:

```sh
cargo ktstr kernel list                # inspect cache contents
cargo ktstr kernel clean --keep 3      # keep the 3 most recent images
cargo ktstr kernel clean --force       # remove everything (non-interactive)
```

## 7. Debug failures

Boot an interactive shell with the scheduler binary:

```sh
cargo ktstr shell -i ./target/debug/scx_my_sched
```

Inside the guest, run `/include-files/scx_my_sched` manually to
inspect behavior. See
[cargo-ktstr shell](../running-tests/cargo-ktstr.md#shell) for
all flags.

See [The #\[ktstr_test\] Macro](../writing-tests/ktstr-test-macro.md)
for all available attributes and
[Scheduler Definitions](../writing-tests/scheduler-definitions.md)
for the full `Scheduler` type and the `declare_scheduler!` macro.
