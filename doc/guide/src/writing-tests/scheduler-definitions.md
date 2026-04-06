# Scheduler Definitions

A `Scheduler` tells the test framework how to launch and configure the
scheduler under test.

## The Scheduler type

```rust
pub struct Scheduler {
    pub name: &'static str,
    pub binary: SchedulerSpec,
    pub flags: &'static [&'static FlagDecl],
    pub sysctls: &'static [(&'static str, &'static str)],
    pub kargs: &'static [&'static str],
    pub verify: Verify,
}
```

## SchedulerSpec

How to find the scheduler binary:

```rust
pub enum SchedulerSpec {
    None,           // No binary -- use EEVDF (kernel default)
    Name(&'static str), // Auto-discover by name
    Path(&'static str), // Explicit path
}
```

## Built-in: EEVDF

`Scheduler::EEVDF` runs tests without a sched_ext scheduler, using the
kernel's default EEVDF scheduler.

## Defining a scheduler

Use the const builder pattern:

```rust
use stt::test_support::{Scheduler, SchedulerSpec};
use stt::scenario::flags::*;
use stt::verify::Verify;

const MY_SCHEDULER: Scheduler = Scheduler::new("my_sched")
    .binary(SchedulerSpec::Name("scx_my_sched"))
    .flags(&[&LLC_DECL, &BORROW_DECL, &STEAL_DECL, &REBAL_DECL])
    .verify(Verify::NONE.max_imbalance_ratio(2.0));
```

## Flag scoping

`Scheduler.flags` defines which flags the scheduler supports.
`generate_profiles()` on the scheduler only considers these flags,
not the global set. This prevents testing with flags the scheduler
doesn't implement.

## Verification overrides

`Scheduler.verify` provides scheduler-level verification defaults.
These sit between `Verify::default_checks()` and per-test overrides in
the merge chain.

A scheduler that tolerates higher imbalance:

```rust
const RELAXED: Scheduler = Scheduler::new("relaxed")
    .binary(SchedulerSpec::Name("scx_relaxed"))
    .verify(Verify::NONE.max_imbalance_ratio(5.0));
```

For an end-to-end workflow from building a scheduler to running the
gauntlet, see [Test a New Scheduler](../recipes/test-new-scheduler.md).
