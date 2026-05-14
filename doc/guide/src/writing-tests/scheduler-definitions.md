# Scheduler Definitions

A `Scheduler` tells the test framework how to launch and configure
the scheduler under test. Tests reference one via
`#[ktstr_test(scheduler = MY_SCHED_PAYLOAD)]`; the verifier sweep
reads every declared scheduler from the `KTSTR_SCHEDULERS`
distributed slice automatically.

## The Scheduler type

```rust,ignore
pub struct Scheduler {
    pub name: &'static str,
    pub binary: SchedulerSpec,
    pub sysctls: &'static [Sysctl],
    pub kargs: &'static [&'static str],
    pub assert: Assert,
    pub cgroup_parent: Option<CgroupPath>,
    pub sched_args: &'static [&'static str],
    pub topology: Topology,
    pub constraints: TopologyConstraints,
    pub config_file: Option<&'static str>,
    pub config_file_def: Option<(&'static str, &'static str)>,
    pub kernels: &'static [&'static str],
}
```

`config_file` packs a host-side file into the initramfs at
`/include-files/{filename}` and prepends `--config /include-files/{filename}`
to scheduler args automatically.

`config_file_def` declares an arg-template + guest-path pair for
schedulers that take inline JSON content via the test attribute
`#[ktstr_test(config = …)]`: the framework writes the test's
`config_content` to the declared guest path and substitutes
`{file}` in the arg template before launching the scheduler. The
two fields are alternatives — `config_file` is the host-file path,
`config_file_def` is the inline-content path. See
[The #\[ktstr_test\] Macro](ktstr-test-macro.md#inline-scheduler-config)
for the inline pairing gate.

`sysctls` takes `Sysctl` values. Construct them with
`Sysctl::new("key", "value")` in `const` context. Use the
dot-separated form for the key (e.g. `"kernel.foo"`, not
`"kernel/foo"`); duplicate keys are applied in order and the
last write wins.

`kargs` is the extra GUEST KERNEL command-line (not the scheduler
binary's CLI — use `sched_args` for that). Do not override the
kargs ktstr injects itself (`console=`, `loglevel=`, `init=`):
those break guest-side init and leave the VM unable to run tests.

`kernels` drives the
[BPF Verifier sweep](../running-tests/verifier.md). Each entry is
a string consumed by `KernelId::parse` at verifier runtime — the
same parser as the `cargo ktstr verifier --kernel <SPEC>` CLI
flag. Accepts exact versions (`"6.14"`), closed ranges spelled
either `..` or `..=` (`"6.14..7.0"` or `"6.14..=7.0"` — both
inclusive on both endpoints), git refs (`"git+URL#REF"`), paths,
and cache keys. An empty `kernels` slice means no verifier cells
emit for this scheduler — `cargo ktstr verifier` silently skips
it.

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
BPF-less sched_ext or debugfs-tuned variants). The `enable`
commands run in the guest to activate the scheduler; `disable`
commands run to deactivate it. No binary is injected into the
VM.

`SchedulerSpec::has_active_scheduling()` returns `true` for all
variants except `Eevdf`. When `true`, the framework runs monitor
threshold evaluation after the scenario and enables auto-repro
on crash.

`Eevdf` and `KernelBuiltin` are excluded from the verifier sweep
at cell-emission time — neither has a userspace binary to load
BPF programs from, so the verifier has nothing to verify.

## Built-in: EEVDF

`Scheduler::EEVDF` runs tests without a sched_ext scheduler,
using the kernel's default EEVDF scheduler. Its binary is
`SchedulerSpec::Eevdf`. It is the default scheduler for
`#[ktstr_test]` entries that do not pass `scheduler = ...`.

## Defining a scheduler

`declare_scheduler!` is the preferred entry point: it constructs a
`pub static Scheduler` and registers it in the `KTSTR_SCHEDULERS`
distributed slice in one step, so the verifier sweep picks it up
automatically.

```rust,ignore
use ktstr::declare_scheduler;
use ktstr::prelude::*;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    sched_args = ["--exit-dump-len", "1048576"],
    topology = (1, 2, 4, 1),
    kernels = ["6.14", "6.15..=7.0"],
});

#[ktstr_test(scheduler = MY_SCHED_PAYLOAD)]
fn basic(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        CgroupDef::named("cg_0").workers(2),
        CgroupDef::named("cg_1").workers(2),
    ])
}
```

The macro emits:

- `pub static MY_SCHED: Scheduler` with the supplied fields.
- A `#[distributed_slice(KTSTR_SCHEDULERS)]` registration so
  `cargo ktstr verifier` discovers it.
- `pub const MY_SCHED_PAYLOAD: Payload` — a wrapper around
  `MY_SCHED` (kind `PayloadKind::Scheduler(&MY_SCHED)`).
  `#[ktstr_test(scheduler = ...)]` expects the `Payload` form;
  pass `MY_SCHED_PAYLOAD` to that slot.

### Accepted fields

Every key=value pair after `name` and `binary` is optional. The
key names match `Scheduler` struct fields:

- `name = "..."` — short human name (required).
- `binary = "scx_name"` — defaults to `SchedulerSpec::Discover(name)`.
  Accepts `SchedulerSpec::Path("/abs/path")`, `SchedulerSpec::Eevdf`,
  or `SchedulerSpec::KernelBuiltin { enable: &[...], disable: &[...] }`.
- `sched_args = ["--a", "--b"]` — CLI args prepended to every
  test that uses this scheduler.
- `kernels = ["6.14", "6.15..=7.0", "git+URL#REF", "/path", "cache-key"]`
  — verifier sweep set; see the field doc above.
- `cgroup_parent = "/path"` — must begin with `/`, must not be
  `"/"` alone.
- `kargs = ["nosmt"]` — guest-kernel cmdline additions.
- `sysctls = [Sysctl::new("kernel.foo", "1")]` — applied before
  the scheduler starts.
- `topology = (numa_nodes, llcs, cores, threads)` — default VM
  topology for `#[ktstr_test]` entries.
- `constraints = TopologyConstraints { ... }` — gauntlet
  topology constraints inherited by `#[ktstr_test]` entries.
- `config_file = "configs/my_sched.toml"` — opaque host-side
  config to pack into the guest initramfs.
- `config_file_def = ("--config={file}", "/include-files/my.json")`
  — alternative inline-config seam (see
  [The #\[ktstr_test\] Macro](ktstr-test-macro.md#inline-scheduler-config)).
- `assert = Assert::NO_OVERRIDES.method_chain()` — scheduler-level
  assertion overrides merged on top of `Assert::default_checks()`.

### Visibility

The identifier can be `pub` or `pub(crate)`:

```rust,ignore
declare_scheduler!(pub MY_SCHED, { name = "my_sched", binary = "scx_my_sched" });
declare_scheduler!(pub(crate) INTERNAL, { name = "internal", binary = "scx_internal" });
```

The macro emits `#[allow(missing_docs)]` on the generated static
so crates with `#![deny(missing_docs)]` compile cleanly.

### Manual definition

The const builder pattern still works when the macro doesn't
fit — e.g. when the scheduler is composed programmatically or
when test-only fixtures need to avoid the distributed-slice
registration:

```rust,ignore
use ktstr::prelude::*;

const MITOSIS: Scheduler = Scheduler::new("scx_mitosis")
    .binary(SchedulerSpec::Discover("scx_mitosis"))
    .topology(1, 2, 4, 1)
    .sched_args(&["--exit-dump-len", "1048576"])
    .cgroup_parent("/ktstr")
    .assert(Assert::NO_OVERRIDES.max_imbalance_ratio(2.0));
```

A manually-defined `Scheduler` is not registered in
`KTSTR_SCHEDULERS` automatically; the verifier sweep does not
see it. Use `declare_scheduler!` for any scheduler that should
participate in `cargo ktstr verifier`.

## Cgroup parent

`Scheduler.cgroup_parent` specifies a cgroup subtree under
`/sys/fs/cgroup` for the scheduler to manage. When set, the VM
init creates the directory before starting the scheduler, and
`--cell-parent-cgroup <path>` is injected into the scheduler
args. The field is `Option<CgroupPath>`. `CgroupPath::new()` is
a const constructor that panics at compile time if the path
does not begin with `/` or is `"/"` alone. The
`Scheduler::cgroup_parent()` builder and the
`declare_scheduler!` `cgroup_parent = "..."` field both accept
`&'static str` and construct a `CgroupPath` internally.

```rust,ignore
declare_scheduler!(MITOSIS, {
    name = "scx_mitosis",
    binary = "scx_mitosis",
    topology = (1, 2, 4, 1),
    cgroup_parent = "/ktstr",
});
```

This creates `/sys/fs/cgroup/ktstr` in the guest and passes
`--cell-parent-cgroup /ktstr` to the scheduler binary.

## Config file

`Scheduler.config_file` specifies a host-side path to an opaque
config file that the scheduler binary reads at startup. The
framework packs the file into the guest initramfs at
`/include-files/{filename}` and prepends `--config /include-files/{filename}`
to the scheduler args. ktstr does not parse or validate the
file — it is passed through as-is.

The `--config` flag name is not configurable. Schedulers that
use `config_file` must accept `--config <path>`. For schedulers
that use a different flag, use `config_file` to place the file
in the guest and add the desired flag via `sched_args` — the
scheduler will also receive `--config` and must not reject
unknown flags.

```rust,ignore
declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    topology = (1, 2, 4, 1),
    config_file = "configs/my_sched.toml",
});
```

This copies `configs/my_sched.toml` from the host into the
guest at `/include-files/my_sched.toml` and passes
`--config /include-files/my_sched.toml` to the scheduler binary.

## Scheduler args

`Scheduler.sched_args` provides default CLI args that apply to
every test using this scheduler. They are prepended before
per-test `extra_sched_args`.

```rust,ignore
declare_scheduler!(MITOSIS, {
    name = "scx_mitosis",
    binary = "scx_mitosis",
    topology = (1, 2, 4, 1),
    cgroup_parent = "/ktstr",
    sched_args = ["--exit-dump-len", "1048576"],
});
```

Merge order: `config_file` injection, then `cgroup_parent`
injection, then `sched_args`, then per-test `extra_sched_args`.

## Default topology

`Scheduler.topology` sets the default VM topology for all tests
using this scheduler. When `#[ktstr_test]` omits `llcs`,
`cores`, and `threads`, the scheduler's topology is used.
Explicit attributes on `#[ktstr_test]` override the scheduler
default.

```rust,ignore
// (numa_nodes, llcs, cores_per_llc, threads_per_core)
declare_scheduler!(MITOSIS, {
    name = "scx_mitosis",
    binary = "scx_mitosis",
    topology = (1, 2, 4, 1),
});
```

Arguments are `(numa_nodes, llcs, cores_per_llc, threads_per_core)`.
Most schedulers use `numa_nodes = 1` (single NUMA node).
`Scheduler::new()` defaults to `(1, 1, 2, 1)` — a minimal
2-CPU single-NUMA VM, sufficient for tests that don't exercise
topology-dependent scheduling.

Tests that need a different topology (e.g. SMT) override
individual dimensions. Unset dimensions still inherit from the
scheduler:

```rust,ignore
// Inherits llcs=2, cores=4 from MITOSIS_PAYLOAD; overrides threads to 2
#[ktstr_test(scheduler = MITOSIS_PAYLOAD, threads = 2)]
fn smt_test(ctx: &Ctx) -> Result<AssertResult> { /* ... */ }
```

## Checking overrides

`Scheduler.assert` provides scheduler-level checking defaults.
These sit between `Assert::default_checks()` and per-test
overrides in the merge chain.

A scheduler that tolerates higher imbalance:

```rust,ignore
declare_scheduler!(RELAXED, {
    name = "relaxed",
    binary = "scx_relaxed",
    assert = Assert::NO_OVERRIDES.max_imbalance_ratio(5.0),
});
```

## Kernel-built scheduler example

For schedulers compiled into the kernel (no userspace binary),
use `SchedulerSpec::KernelBuiltin` with shell commands to
activate/deactivate the scheduler:

```rust,ignore
use ktstr::declare_scheduler;
use ktstr::prelude::*;

declare_scheduler!(MINLAT, {
    name = "minlat",
    binary = SchedulerSpec::KernelBuiltin {
        enable: &["echo minlat > /sys/kernel/debug/sched/ext/root/ops"],
        disable: &["echo none > /sys/kernel/debug/sched/ext/root/ops"],
    },
});
```

The `enable` commands run in the guest before scenarios start.
The `disable` commands run after scenarios complete.

`KernelBuiltin` schedulers do not participate in the verifier
sweep (no userspace binary to load BPF programs from); the
declaration is still useful for `#[ktstr_test(scheduler = ...)]`
attribution and sidecar identification.

## Payload Definitions {#derive-payload}

A `Payload` describes a binary workload that a test can run
alongside (or in place of) its cgroup workers. The same struct
encodes both `PayloadKind::Binary` (an external executable —
`schbench`, `fio`, `stress-ng`) and `PayloadKind::Scheduler`
(the scheduler under test, constructed via
`Payload::from_scheduler` and re-emitted by `declare_scheduler!`
as `*_PAYLOAD`). Tests reference a `Payload` via
`#[ktstr_test(payload = FIXTURE)]` (primary slot) or
`#[ktstr_test(workloads = [FIXTURE_A, FIXTURE_B])]` (additional
slots); the test body then runs it via `ctx.payload(&FIXTURE)`.

### `#[non_exhaustive]` and construction rules

`Payload` is `#[non_exhaustive]` (see [`crate::non_exhaustive`](https://likewhatevs.github.io/ktstr/api/ktstr/non_exhaustive/index.html)).
Downstream crates **cannot** use struct-literal construction —
a future ktstr bump can add fields without breaking callers
only if everyone constructs through the provided associated
functions:

- [`Payload::binary(name, binary)`](https://likewhatevs.github.io/ktstr/api/ktstr/test_support/struct.Payload.html#method.binary)
  — minimal binary-kind `Payload` with exit-code-only defaults
  (no declared args, checks, metrics, or include files). Fills
  `name`, sets `kind = PayloadKind::Binary(binary)`.
- [`Payload::from_scheduler(&'static Scheduler)`](https://likewhatevs.github.io/ktstr/api/ktstr/test_support/struct.Payload.html#method.from_scheduler)
  — wraps a `Scheduler` const into a `PayloadKind::Scheduler`
  payload. `declare_scheduler!` uses this internally to emit
  the `*_PAYLOAD` const; test authors rarely call it directly.

For richer binary payloads (custom default args, declared
`MetricCheck`s, `MetricHint`s, `include_files`), use
`#[derive(Payload)]` on a marker struct — the derive generates
the matching `const` via the same non-exhaustive-preserving
construction path. `tests/common/fixtures.rs` has worked
examples — `SCHBENCH`, `SCHBENCH_HINTED`, `SCHBENCH_JSON` —
suitable as reference shapes to copy.

### Quick reference: `Payload` fields

The fields are listed here for readers tracing the fixture
files, not as a license to hand-roll literals. Each is
populated by `Payload::binary` + the derive's builder methods:

- `name: &'static str` — display name that appears in sidecar
  JSON, stats tables, and test filtering. Distinct from the
  binary name (`kind`) so e.g. `SCHBENCH_HINTED` can run the
  same `schbench` binary with a different label.
- `kind: PayloadKind` — either `Binary(executable_name)` (for
  test payloads like `schbench`) or `Scheduler(&'static Scheduler)`
  (for the scheduler slot; constructed via
  `Payload::from_scheduler`).
- `output: OutputFormat` — how to interpret the payload's
  stdout/stderr. `ExitCode` (status code only), `Json` (parse
  numeric leaves), or `LlmExtract(Option<&'static str>)` (route
  through a local LLM with an optional hint).
- `default_args: &'static [&'static str]` — CLI args prepended
  to every invocation. Per-test `ctx.payload(...).args(...)`
  appends after these.
- `default_checks: &'static [MetricCheck]` — static assertions
  applied to the payload's output/exit (`min` / `max` / `range`
  / `exists` / `exit_code_eq` constructors on `MetricCheck`).
  Merged with per-test `.checks(...)`.
- `metrics: &'static [MetricHint]` — declared metrics the
  payload emits (name, unit, polarity). Drives `list-metrics`
  and comparison thresholds.
- `metric_bounds: Option<&'static MetricBounds>` — optional
  per-metric host-side bounds applied AFTER the payload exits.
  Consumed by `LlmExtract` payloads (where extraction runs
  host-side post-VM-exit); `Json` and `ExitCode` payloads
  ignore this field and route assertions through
  `default_checks` instead.
- `include_files: &'static [&'static str]` — extra files
  packaged into the guest alongside the binary (config files,
  datasets).
- `uses_parent_pgrp: bool` — when true, the payload child
  inherits the test's process group so the teardown SIGKILL
  sweep reaches it. Most binaries leave this `false` and are
  reaped explicitly.
- `known_flags: Option<&'static [&'static str]>` — optional
  allow-list of CLI flags the payload accepts; used by the
  gauntlet-style flag expansion.

For an end-to-end workflow from building a scheduler to running
the gauntlet, see
[Test a New Scheduler](../recipes/test-new-scheduler.md).
