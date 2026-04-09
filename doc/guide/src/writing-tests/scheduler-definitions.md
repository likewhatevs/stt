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
    pub cgroup_parent: Option<&'static str>,
    pub sched_args: &'static [&'static str],
    pub topology: Topology,
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

const MY_SCHEDULER: Scheduler = Scheduler::new("my_sched")
    .binary(SchedulerSpec::Name("scx_my_sched"))
    .flags(&[&MY_LLC, &MY_STEAL])
    .topology(2, 4, 1)
    .assert(Assert::NONE.max_imbalance_ratio(2.0));
```

## Cgroup parent

`Scheduler.cgroup_parent` specifies a cgroup subtree under
`/sys/fs/cgroup` for the scheduler to manage. When set, the VM init
creates the directory before starting the scheduler, and
`--cell-parent-cgroup <path>` is injected into the scheduler args.

```rust,ignore
const MITOSIS: Scheduler = Scheduler::new("scx_mitosis")
    .binary(SchedulerSpec::Name("scx_mitosis"))
    .topology(2, 4, 1)
    .cgroup_parent("/stt");
```

This creates `/sys/fs/cgroup/stt` in the guest and passes
`--cell-parent-cgroup /stt` to the scheduler binary.

## Scheduler args

`Scheduler.sched_args` provides default CLI args that apply to every
test using this scheduler. They are prepended before per-test
`extra_sched_args` and flag-derived args.

```rust,ignore
const MITOSIS: Scheduler = Scheduler::new("scx_mitosis")
    .binary(SchedulerSpec::Name("scx_mitosis"))
    .topology(2, 4, 1)
    .cgroup_parent("/stt")
    .sched_args(&["--exit-dump-len", "1048576"]);
```

Merge order: `cgroup_parent` injection, then `sched_args`, then
per-test `extra_sched_args`, then flag-derived args.

## Default topology

`Scheduler.topology` sets the default VM topology for all tests using
this scheduler. When `#[stt_test]` omits `sockets`, `cores`, and
`threads`, the scheduler's topology is used. Explicit attributes on
`#[stt_test]` override the scheduler default.

```rust,ignore
const MITOSIS: Scheduler = Scheduler::new("scx_mitosis")
    .binary(SchedulerSpec::Name("scx_mitosis"))
    .cgroup_parent("/stt")
    .topology(2, 4, 1);
```

Arguments are `(sockets, cores_per_socket, threads_per_core)`.
`Scheduler::new()` defaults to `(1, 2, 1)`.

Tests that need a different topology (e.g. SMT) override individual
dimensions. Unset dimensions still inherit from the scheduler:

```rust,ignore
// Inherits sockets=2, cores=4 from MITOSIS; overrides threads to 2
#[stt_test(scheduler = MITOSIS, threads = 2)]
fn smt_test(ctx: &Ctx) -> Result<AssertResult> { /* ... */ }
```

## Defining flags

Each scheduler defines its own `FlagDecl` statics with the CLI args
that activate each feature. `FlagDecl` is re-exported from the
prelude.

```rust,ignore
use stt::prelude::*;

static MITOSIS_LLC: FlagDecl = FlagDecl {
    name: "llc",
    args: &["--enable-llc-awareness"],
    requires: &[],
};

static MITOSIS_BORROW: FlagDecl = FlagDecl {
    name: "borrow",
    args: &["--enable-borrowing"],
    requires: &[],
};

static MITOSIS_STEAL: FlagDecl = FlagDecl {
    name: "steal",
    args: &["--enable-work-stealing"],
    requires: &[&MITOSIS_LLC],
};

const MITOSIS: Scheduler = Scheduler::new("mitosis")
    .binary(SchedulerSpec::Name("scx_mitosis"))
    .flags(&[&MITOSIS_LLC, &MITOSIS_BORROW, &MITOSIS_STEAL])
    .topology(2, 4, 1);
```

The `args` field contains the scheduler CLI arguments passed when the
flag is active. The `requires` field expresses dependencies: `steal`
requires `llc`, so any profile containing `steal` automatically
includes `llc`. Invalid combinations are rejected by
`generate_profiles()`.

The built-in `*_DECL` constants in `stt::scenario::flags` (e.g.
`LLC_DECL`, `BORROW_DECL`) have empty `args` fields. They exist for
stt's internal scenario catalog. External consumers must define their
own `FlagDecl` statics with their scheduler's actual CLI arguments.

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
activate/deactivate the scheduler:

```rust,ignore
use stt::prelude::*;

static MINLAT_LLC: FlagDecl = FlagDecl {
    name: "llc",
    args: &[],
    requires: &[],
};

const MINLAT: Scheduler = Scheduler::new("minlat")
    .binary(SchedulerSpec::KernelBuiltin {
        enable: &["echo minlat > /sys/kernel/debug/sched/ext/root/ops"],
        disable: &["echo none > /sys/kernel/debug/sched/ext/root/ops"],
    })
    .flags(&[&MINLAT_LLC]);
```

The `enable` commands run in the guest before scenarios start.
The `disable` commands run after scenarios complete.

For an end-to-end workflow from building a scheduler to running the
gauntlet, see [Test a New Scheduler](../recipes/test-new-scheduler.md).
