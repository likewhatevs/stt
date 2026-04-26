# Scheduler Definitions

A `Scheduler` tells the test framework how to launch and configure the
scheduler under test.

## The Scheduler type

```rust,ignore
pub struct Scheduler {
    pub name: &'static str,
    pub binary: SchedulerSpec,
    pub flags: &'static [&'static FlagDecl],
    pub sysctls: &'static [Sysctl],
    pub kargs: &'static [&'static str],
    pub assert: Assert,
    pub cgroup_parent: Option<CgroupPath>,
    pub sched_args: &'static [&'static str],
    pub topology: Topology,
    pub constraints: TopologyConstraints,
    pub config_file: Option<&'static str>,
}
```

`sysctls` takes `Sysctl` values. Construct them with
`Sysctl::new("key", "value")` in `const` context. Use the
dot-separated form for the key (e.g. `"kernel.foo"`, not
`"kernel/foo"`); duplicate keys are applied in order and the last
write wins.

`kargs` is the extra GUEST KERNEL command-line (not the scheduler
binary's CLI — use `sched_args` for that). Do not override the kargs
ktstr injects itself (`nokaslr`, `console=`, `loglevel=`, `init=`):
those break guest-side init and leave the VM unable to run tests.

Both `sysctls` and `kargs` are accepted by `#[derive(Scheduler)]`:

```rust,ignore
#[derive(Scheduler)]
#[scheduler(
    name = "my_sched",
    binary = "scx_my_sched",
    sysctls = [Sysctl::new("kernel.sched_cfs_bandwidth_slice_us", "1000")],
    kargs = ["nosmt"],
)]
enum MySchedFlag { /* ... */ }
```

## SchedulerSpec

How to find the scheduler binary:

```rust,ignore
pub enum SchedulerSpec {
    Eevdf,                   // No sched_ext binary -- use kernel EEVDF
    Discover(&'static str),  // Auto-discover by name
    Path(&'static str),      // Explicit path
    KernelBuiltin {          // Kernel-built scheduler (no binary)
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
variants except `Eevdf`. When `true`, the framework runs monitor
threshold evaluation after the scenario and enables auto-repro on
crash.

## Built-in: EEVDF

`Scheduler::EEVDF` runs tests without a sched_ext scheduler, using the
kernel's default EEVDF scheduler. Its binary is `SchedulerSpec::Eevdf`.

## Defining a scheduler

Use `#[derive(Scheduler)]` on an enum whose variants are the
scheduler's flags:

```rust,ignore
use ktstr::prelude::*;

#[derive(Scheduler)]
#[scheduler(
    name = "my_sched",
    binary = "scx_my_sched",
    topology(1, 2, 4, 1),
    sched_args = ["--exit-dump-len", "1048576"]
)]
#[allow(dead_code)]
enum MySchedFlag {
    #[flag(args = ["--enable-llc"])]
    Llc,
    #[flag(args = ["--enable-stealing"], requires = [Llc])]
    Steal,
}
```

This generates:

- `static` `FlagDecl` entries for each variant
- A `&[&FlagDecl]` flags array
- `const MY_SCHED: Scheduler` with all builder methods applied —
  the bare `Scheduler` form, suitable for library code that
  composes `Scheduler` builders directly.
- `const MY_SCHED_PAYLOAD: Payload` — a `Payload` wrapper around
  `MY_SCHED` (kind: `PayloadKind::Scheduler(&MY_SCHED)`). This
  is the form that `#[ktstr_test(scheduler = ...)]` expects: the
  test entry's scheduler slot is `&'static Payload`, not
  `&'static Scheduler`. Pass `MY_SCHED_PAYLOAD` to that slot;
  the bare `MY_SCHED` form will not type-check.
- `impl MySchedFlag` with `&'static str` constants for each variant's
  kebab-case name (e.g. `MySchedFlag::LLC`, `MySchedFlag::STEAL`)

The const name stem is derived from the enum name by stripping a
trailing `Flag`/`Flags` suffix and converting to SCREAMING_SNAKE_CASE:
`MySchedFlag` -> `MY_SCHED`, `EevdfFlags` -> `EEVDF`. The
`Scheduler` const uses the stem verbatim; the `Payload` wrapper
appends `_PAYLOAD` (so `MY_SCHED` + `MY_SCHED_PAYLOAD`).

Variant names are converted to kebab-case for the flag name:
`RejectPin` -> `"reject-pin"`, `NoCtrl` -> `"no-ctrl"`.

### Manual definition

The const builder pattern still works for cases where the derive
doesn't fit:

```rust,ignore
use ktstr::prelude::*;

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

const MY_SCHED: Scheduler = Scheduler::new("my_sched")
    .binary(SchedulerSpec::Discover("scx_my_sched"))
    .flags(&[&MY_LLC, &MY_STEAL])
    .topology(1, 2, 4, 1)
    .assert(Assert::NO_OVERRIDES.max_imbalance_ratio(2.0));
```

## Cgroup parent

`Scheduler.cgroup_parent` specifies a cgroup subtree under
`/sys/fs/cgroup` for the scheduler to manage. When set, the VM init
creates the directory before starting the scheduler, and
`--cell-parent-cgroup <path>` is injected into the scheduler args.
The field is `Option<CgroupPath>`. `CgroupPath::new()` is a const
constructor that panics at compile time if the path does not begin
with `/` or is `"/"` alone. The `Scheduler::cgroup_parent()` builder
accepts `&'static str` and constructs a `CgroupPath` internally.

In the derive:

```rust,ignore
#[scheduler(cgroup_parent = "/ktstr")]
```

Or manually:

```rust,ignore
const MITOSIS: Scheduler = Scheduler::new("scx_mitosis")
    .binary(SchedulerSpec::Discover("scx_mitosis"))
    .topology(1, 2, 4, 1)
    .cgroup_parent("/ktstr");
```

This creates `/sys/fs/cgroup/ktstr` in the guest and passes
`--cell-parent-cgroup /ktstr` to the scheduler binary.

## Config file

`Scheduler.config_file` specifies a host-side path to an opaque config
file that the scheduler binary reads at startup. The framework packs
the file into the guest initramfs at `/include-files/{filename}` and
prepends `--config /include-files/{filename}` to the scheduler args.
ktstr does not parse or validate the file — it is passed through as-is.

The `--config` flag name is not configurable. Schedulers that use
`config_file` must accept `--config <path>`. For schedulers that use a
different flag, use `config_file` to place the file in the guest and
add the desired flag via `sched_args` — the scheduler will also
receive `--config` and must not reject unknown flags.

In the derive:

```rust,ignore
#[scheduler(config_file = "configs/my_sched.toml")]
```

Or manually:

```rust,ignore
const MY_SCHED: Scheduler = Scheduler::new("my_sched")
    .binary(SchedulerSpec::Discover("scx_my_sched"))
    .topology(1, 2, 4, 1)
    .config_file("configs/my_sched.toml");
```

This copies `configs/my_sched.toml` from the host into the guest at
`/include-files/my_sched.toml` and passes
`--config /include-files/my_sched.toml` to the scheduler binary.

## Scheduler args

`Scheduler.sched_args` provides default CLI args that apply to every
test using this scheduler. They are prepended before per-test
`extra_sched_args` and flag-derived args.

In the derive:

```rust,ignore
#[scheduler(sched_args = ["--exit-dump-len", "1048576"])]
```

Or manually:

```rust,ignore
const MITOSIS: Scheduler = Scheduler::new("scx_mitosis")
    .binary(SchedulerSpec::Discover("scx_mitosis"))
    .topology(1, 2, 4, 1)
    .cgroup_parent("/ktstr")
    .sched_args(&["--exit-dump-len", "1048576"]);
```

Merge order: `config_file` injection, then `cgroup_parent` injection,
then `sched_args`, then per-test `extra_sched_args`, then flag-derived
args.

## Default topology

`Scheduler.topology` sets the default VM topology for all tests using
this scheduler. When `#[ktstr_test]` omits `llcs`, `cores`, and
`threads`, the scheduler's topology is used. Explicit attributes on
`#[ktstr_test]` override the scheduler default.

In the derive:

```rust,ignore
//              numa_nodes, llcs, cores_per_llc, threads_per_core
#[scheduler(topology(1, 2, 4, 1))]
```

Arguments are `(numa_nodes, llcs, cores_per_llc, threads_per_core)`.
Most schedulers use `numa_nodes = 1` (single NUMA node).
`Scheduler::new()` defaults to `(1, 1, 2, 1)` — a minimal 2-CPU
single-NUMA VM, sufficient for tests that don't exercise
topology-dependent scheduling.

Tests that need a different topology (e.g. SMT) override individual
dimensions. Unset dimensions still inherit from the scheduler:

```rust,ignore
// Inherits llcs=2, cores=4 from MITOSIS; overrides threads to 2
#[ktstr_test(scheduler = MITOSIS, threads = 2)]
fn smt_test(ctx: &Ctx) -> Result<AssertResult> { /* ... */ }
```

## Defining flags

Each enum variant is a flag. `#[flag(args)]` lists CLI arguments
passed when the flag is active. `#[flag(requires)]` declares
dependencies. See [Flags](../concepts/flags.md) for dependency
semantics, profile generation, and using flags in
[`#[ktstr_test]`](ktstr-test-macro.md#flag-constraints).

## Flag scoping

`Scheduler.flags` defines which flags the scheduler supports.
`generate_profiles()` on the scheduler only considers these flags,
not the global set. This prevents testing with flags the scheduler
doesn't implement.

## Checking overrides

`Scheduler.assert` provides scheduler-level checking defaults.
These sit between `Assert::default_checks()` and per-test overrides in
the merge chain.

A scheduler that tolerates higher imbalance:

```rust,ignore
const RELAXED: Scheduler = Scheduler::new("relaxed")
    .binary(SchedulerSpec::Discover("scx_relaxed"))
    .assert(Assert::NO_OVERRIDES.max_imbalance_ratio(5.0));
```

## Payload Definitions {#derive-payload}

A `Payload` describes a binary workload that a test can run alongside
(or in place of) its cgroup workers. The same struct encodes both
`PayloadKind::Binary` (an external executable — `schbench`, `fio`,
`stress-ng`) and `PayloadKind::Scheduler` (the scheduler under test,
constructed via `Payload::from_scheduler`). Tests reference a
`Payload` via `#[ktstr_test(payload = FIXTURE)]` (primary slot) or
`#[ktstr_test(workloads = [FIXTURE_A, FIXTURE_B])]` (additional slots);
the test body then runs it via `ctx.payload(&FIXTURE)`.

### `#[non_exhaustive]` and construction rules

`Payload` is `#[non_exhaustive]` (see [`crate::non_exhaustive`](https://likewhatevs.github.io/ktstr/api/ktstr/non_exhaustive/index.html)).
Downstream crates **cannot** use struct-literal construction — a
future ktstr bump can add fields without breaking callers only if
everyone constructs through the provided associated functions:

- [`Payload::binary(name, binary)`](https://likewhatevs.github.io/ktstr/api/ktstr/test_support/struct.Payload.html#method.binary)
  — minimal binary-kind `Payload` with exit-code-only defaults (no
  declared args, checks, metrics, or include files). Fills `name`,
  sets `kind = PayloadKind::Binary(binary)`.
- [`Payload::from_scheduler(&'static Scheduler)`](https://likewhatevs.github.io/ktstr/api/ktstr/test_support/struct.Payload.html#method.from_scheduler)
  — wraps a `Scheduler` const into a `PayloadKind::Scheduler`
  payload. Used internally by the scheduler slot plumbing; test
  authors rarely call this directly.

For richer binary payloads (custom default args, declared `Check`s,
`MetricHint`s, `include_files`), use `#[derive(Payload)]` on a
marker struct — the derive generates the matching `const` via the
same non-exhaustive-preserving construction path. `tests/common/fixtures.rs`
has worked examples — `SCHBENCH`, `SCHBENCH_HINTED`, `SCHBENCH_JSON`
— suitable as reference shapes to copy.

### Quick reference: `Payload` fields

The fields are listed here for readers tracing the fixture files,
not as a license to hand-roll literals. Each is populated by
`Payload::binary` + the derive's builder methods:

- `name: &'static str` — display name that appears in sidecar JSON,
  stats tables, and test filtering. Distinct from the binary name
  (`kind`) so e.g. `SCHBENCH_HINTED` can run the same `schbench`
  binary with a different label.
- `kind: PayloadKind` — either `Binary(executable_name)` (for test
  payloads like `schbench`) or `Scheduler(&'static Scheduler)` (for
  the scheduler slot; constructed via `Payload::from_scheduler`).
- `output: OutputFormat` — how to interpret the payload's stdout/stderr.
  `ExitCode` (status code only), `Json` (parse numeric leaves), or
  `LlmExtract(Option<&'static str>)` (route through a local LLM with
  an optional hint — see [concepts/metrics](../concepts/work-types.md)).
- `default_args: &'static [&'static str]` — CLI args prepended to
  every invocation. Per-test `ctx.payload(...).args(...)` appends
  after these.
- `default_checks: &'static [Check]` — static assertions applied to
  the payload's output/exit. Merged with per-test `.checks(...)`.
- `metrics: &'static [MetricHint]` — declared metrics the payload
  emits (name, unit, polarity). Drives `list-metrics` and
  comparison thresholds.
- `include_files: &'static [&'static str]` — extra files packaged
  into the guest alongside the binary (config files, datasets).
- `uses_parent_pgrp: bool` — when true, the payload child inherits
  the test's process group so the teardown SIGKILL sweep reaches
  it. Most binaries leave this `false` and are reaped explicitly.
- `known_flags: Option<&'static [&'static str]>` — optional
  allow-list of CLI flags the payload accepts; used by the
  gauntlet-style flag expansion.

## Kernel-built scheduler example

For schedulers compiled into the kernel (no userspace binary),
use `SchedulerSpec::KernelBuiltin` with shell commands to
activate/deactivate the scheduler:

```rust,ignore
use ktstr::prelude::*;

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
