# Scheduler Definitions

A `Scheduler` tells the test framework how to launch and configure the
scheduler under test.

## The Scheduler type

```rust,ignore
pub struct Scheduler {
    pub name: &'static str,
    pub binary: SchedulerSpec,
    pub flags: &'static [&'static FlagDecl],
    pub sysctls: &'static [(&'static str, &'static str)],
    pub kargs: &'static [&'static str],
    pub assert: Assert,
}
```

## SchedulerSpec

How to find the scheduler binary:

```rust,ignore
pub enum SchedulerSpec {
    None,                // No binary -- use EEVDF (kernel default)
    Name(&'static str),  // Auto-discover by name
    Path(&'static str),  // Explicit path
    KernelBuiltin {      // Kernel-built scheduler (no binary)
        enable: &'static [&'static str],
        disable: &'static [&'static str],
    },
}
```

`KernelBuiltin` is for schedulers compiled into the kernel (e.g.
BPF-less sched_ext or debugfs-tuned variants). The `enable` commands
run in the guest to activate the scheduler; `disable` commands run
to deactivate it. No binary is injected into the VM.

`SchedulerSpec::has_active_scheduling()` returns `true` for all
variants except `None`. When `true`, the framework runs monitor
threshold evaluation after the scenario and enables auto-repro on
crash.

## Built-in: EEVDF

`Scheduler::EEVDF` runs tests without a sched_ext scheduler, using the
kernel's default EEVDF scheduler. Its binary is `SchedulerSpec::None`.

## Defining a scheduler

Use the const builder pattern:

```rust,ignore
use stt::prelude::*;
use stt::scenario::flags::*;

const MY_SCHEDULER: Scheduler = Scheduler::new("my_sched")
    .binary(SchedulerSpec::Name("scx_my_sched"))
    .flags(&[&LLC_DECL, &BORROW_DECL, &STEAL_DECL, &REBAL_DECL])
    .assert(Assert::NONE.max_imbalance_ratio(2.0));
```

## Flag scoping

`Scheduler.flags` defines which flags the scheduler supports.
`generate_profiles()` on the scheduler only considers these flags,
not the global set. This prevents testing with flags the scheduler
doesn't implement.

## Verification overrides

`Scheduler.assert` provides scheduler-level verification defaults.
These sit between `Assert::default_checks()` and per-test overrides in
the merge chain.

A scheduler that tolerates higher imbalance:

```rust,ignore
const RELAXED: Scheduler = Scheduler::new("relaxed")
    .binary(SchedulerSpec::Name("scx_relaxed"))
    .assert(Assert::NONE.max_imbalance_ratio(5.0));
```

## Kernel-built scheduler example

For schedulers compiled into the kernel (no userspace binary),
use `SchedulerSpec::KernelBuiltin` with shell commands to
activate/deactivate the scheduler and `FlagDecl.shell_cmds` for
flag-specific tunables:

```rust,ignore
use stt::prelude::*;
use stt::scenario::flags::FlagDecl;

static MINLAT_LLC: FlagDecl = FlagDecl {
    name: "llc",
    args: &[],
    requires: &[],
    shell_cmds: &[
        "echo 1 > /sys/kernel/debug/sched/ext/minlat/llc_aware",
    ],
};

const MINLAT: Scheduler = Scheduler::new("minlat")
    .binary(SchedulerSpec::KernelBuiltin {
        enable: &["echo minlat > /sys/kernel/debug/sched/ext/root/ops"],
        disable: &["echo none > /sys/kernel/debug/sched/ext/root/ops"],
    })
    .flags(&[&MINLAT_LLC]);
```

The `enable` commands run in the guest before scenarios start.
The `disable` commands run after scenarios complete. Flag
`shell_cmds` run when the flag is active in the current profile.

For an end-to-end workflow from building a scheduler to running the
gauntlet, see [Test a New Scheduler](../recipes/test-new-scheduler.md).
