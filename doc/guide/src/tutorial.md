# Zero to ktstr

This tutorial walks through writing a complete `#[ktstr_test]` from
scratch. By the end you'll have a working scheduler test that runs
two cgroups with different lifecycle patterns across a multi-LLC
topology, tunes test duration and the watchdog, and asserts
fairness, throughput parity, and cpuset isolation.

## What you'll build

A test named `mixed_workloads` that:

- Runs **two cgroups** on **separate LLCs**:
  - `background_spinner` -- a persistent CPU-bound load that runs
    for the entire test duration.
  - `phased_worker` -- a worker that loops through explicit
    `Spin -> Yield -> Spin -> Yield ...` phases via
    `WorkType::Sequence`.
- Targets a **2-LLC, 4-core topology** so the scheduler has a real
  cache boundary to respect.
- Sets explicit **test duration** and **scx watchdog timeout** via
  `#[ktstr_test]` attributes.
- Asserts **fairness** (per-cgroup spread), **throughput parity**
  (CV across workers + minimum rate), and **cpuset isolation**
  (workers stay on their assigned CPUs). Scheduling gaps and
  host-side runqueue health are checked automatically.

The complete test is at the [end of this page](#the-complete-test).

## Prerequisites

Set up the host and a kernel before continuing:

- [Getting Started](getting-started.md) covers KVM access, the
  toolchain, and the dev-dependency.
- A bootable Linux kernel image is required. Build one with
  `cargo ktstr kernel build` or point at a source tree with
  `cargo ktstr test --kernel ../linux`. See
  [Getting Started: Build a kernel](getting-started.md#build-a-kernel)
  for the full kernel-management workflow.

Once the dependency is in place, create a file under your crate's
`tests/` directory (e.g. `tests/mixed_workloads.rs`) and follow along.

## Step 1: The skeleton

Every `#[ktstr_test]` is a Rust function that takes `&Ctx` and returns
`Result<AssertResult>`. Start with an empty body that passes
unconditionally:

```rust,ignore
use ktstr::prelude::*;

#[ktstr_test(llcs = 1, cores = 2, threads = 1)]
fn mixed_workloads(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    Ok(AssertResult::pass())
}
```

`use ktstr::prelude::*;` brings in every type the test body needs --
`Ctx`, `AssertResult`, `CgroupDef`, `WorkType`, `CpusetSpec`,
`execute_defs`, and the `Result` alias from `anyhow`. The
`#[ktstr_test]` attribute registers the function so `cargo ktstr test`
discovers it and boots a VM with the requested topology.

A test without a `scheduler = ...` attribute runs under the kernel's
default EEVDF scheduler — useful as a baseline. Step 2 swaps in a
sched_ext scheduler so the rest of the tutorial exercises that
scheduler instead.

For the full attribute reference, see
[The #\[ktstr_test\] Macro](writing-tests/ktstr-test-macro.md).

## Step 2: Define your scheduler

To target a sched_ext scheduler, declare it with
`declare_scheduler!` and reference the generated `_PAYLOAD` const
from `#[ktstr_test(scheduler = …)]`. The example uses `scx-ktstr`,
the test-fixture scheduler shipped in this workspace; substitute
your own binary name to target a different scheduler.

```rust,ignore
use ktstr::declare_scheduler;
use ktstr::prelude::*;

declare_scheduler!(KTSTR_SCHED, {
    name = "ktstr_sched",
    binary = "scx-ktstr",
});

#[ktstr_test(scheduler = KTSTR_SCHED_PAYLOAD, llcs = 1, cores = 2, threads = 1)]
fn mixed_workloads(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    Ok(AssertResult::pass())
}
```

`declare_scheduler!` emits two consts:

- `pub static KTSTR_SCHED: Scheduler` — the bare scheduler record,
  registered in the `KTSTR_SCHEDULERS` distributed slice so
  `cargo ktstr verifier` discovers it automatically.
- `pub const KTSTR_SCHED_PAYLOAD: Payload` — the `Payload`
  wrapper. The `scheduler =` slot on `#[ktstr_test]` expects a
  `Payload`, so pass `KTSTR_SCHED_PAYLOAD`.

The macro fields:

- `name` — scheduler name for display and sidecar keys.
- `binary` — binary name for auto-discovery in
  `target/{debug,release}/`, the directory containing the test
  binary, or a `KTSTR_SCHEDULER` override path. When the scheduler
  is a `[[bin]]` target in the same workspace, `cargo build`
  already places it where discovery looks. The resolved binary is
  packed into the VM's initramfs.
- `topology = (numa, llcs, cores, threads)` — optional default VM
  topology. Tests can override individual dimensions via
  `#[ktstr_test(llcs = ...)]`. Omitted here; the per-test
  attributes in Step 4 set every dimension explicitly.
- `sched_args = ["--flag", "--another"]` — optional CLI args
  prepended to every test that uses this scheduler. Useful when a
  scheduler needs the same `--enable-llc`-style switches in every
  run; for one-off variations, use `#[ktstr_test(extra_sched_args = [...])]`
  on the test instead.
- `kernels = ["6.14", "6.15..=7.0"]` — optional set of kernel
  specs the verifier sweep should exercise this scheduler against.
  See [BPF Verifier](running-tests/verifier.md) for the cell
  emission contract.

For the full attribute surface (`sysctls`, `kargs`, `config_file`,
gauntlet constraints, scheduler-level assertion overrides), see
[Scheduler Definitions](writing-tests/scheduler-definitions.md).

When the macro doesn't fit — the most common case being inline
JSON config supplied per-test or programmatic composition — define
the `Scheduler` const through the manual builder instead. Step 12
below walks through that path with `scx_layered`.

## Step 3: Add workloads

A `CgroupDef` declares a cgroup along with the workers that will run
inside it. The builder methods configure worker count, the work each
worker performs, scheduling policy, and cpuset assignment.

Add two cgroups -- both running tight CPU spinners for now. Step 5
will swap one of them for a phased workload:

```rust,ignore
use ktstr::prelude::*;

#[ktstr_test(scheduler = KTSTR_SCHED_PAYLOAD, llcs = 1, cores = 2, threads = 1)]
fn mixed_workloads(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        CgroupDef::named("background_spinner")
            .workers(2)
            .work_type(WorkType::SpinWait),
        CgroupDef::named("phased_worker")
            .workers(2)
            .work_type(WorkType::SpinWait),
    ])
}
```

Without `.with_cpuset(...)`, a cgroup's workers run on every CPU
in the test's topology — they share the VM's full CPU set
with all other cgroups. `.with_cpuset(CpusetSpec::Llc(idx))`
(introduced in Step 4) restricts a cgroup to one LLC's CPUs, and
the other `CpusetSpec` variants narrow further.

`WorkType::SpinWait` runs a tight CPU spin loop; it is one of many
work primitives -- see [WorkType](concepts/work-types.md) for the
full enum (`Bursty`, `FutexPingPong`, `CachePressure`,
`IoSyncWrite`, `PageFaultChurn`, `MutexContention`, `Sequence`, etc.)
and the work-type-to-scheduler-behavior mapping table.

`execute_defs` is a convenience wrapper that runs each cgroup
concurrently for the test's full duration. Both cgroups are
**persistent** -- they hold for the entire scenario. Use
`execute_steps` when you need to add cgroups mid-run or swap
cpusets between phases; see [Ops and Steps](concepts/ops.md) for
the multi-step API.

## Step 4: Set topology

The `#[ktstr_test]` attribute carries the VM's CPU topology.
Topology dimensions are big-to-little: `numa_nodes` (default 1),
`llcs` (total across all NUMA nodes), `cores` per LLC, and
`threads` per core. Total CPU count is `llcs * cores * threads`.

LLC count matters because the last-level cache is the primary
scheduling boundary -- tasks sharing an LLC benefit from shared
cache lines, while cross-LLC migration carries a cold-cache penalty.
A scheduler that ignores LLC topology will look fine on `llcs = 1`
and start failing as soon as there is a real cache boundary to
respect.

Bump the topology to two LLCs with two cores each (4 CPUs total) so
each cgroup can own its own LLC:

```rust,ignore
#[ktstr_test(scheduler = KTSTR_SCHED_PAYLOAD, llcs = 2, cores = 2, threads = 1)]
fn mixed_workloads(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        CgroupDef::named("background_spinner")
            .workers(2)
            .work_type(WorkType::SpinWait)
            .with_cpuset(CpusetSpec::Llc(0)),
        CgroupDef::named("phased_worker")
            .workers(2)
            .work_type(WorkType::SpinWait)
            .with_cpuset(CpusetSpec::Llc(1)),
    ])
}
```

`CpusetSpec::Llc(idx)` confines a cgroup to the CPUs that belong to
LLC `idx`. Other variants (`Numa`, `Range`, `Disjoint`, `Overlap`,
`Exact`) cover NUMA-node binding, fractional partitioning, and
hand-built CPU sets.

For the full topology surface (NUMA accessors, per-LLC info,
cpuset generation helpers), see [TestTopology](concepts/topology.md).

## Step 5: Compose phased work inside a cgroup

So far both cgroups run identical CPU spinners. The point of this
test is to exercise a scheduler against **different lifecycle
patterns** at once, so swap `phased_worker` for a worker that loops
through explicit phases.

`WorkType::Sequence { first: Phase, rest: Vec<Phase> }` runs each
phase for its specified duration and then advances to the next; when
the last phase ends the loop restarts from `first`. Phases:
`Phase::Spin(Duration)`, `Phase::Sleep(Duration)`,
`Phase::Yield(Duration)`, `Phase::Io(Duration)`. Use the
`WorkType::sequence(first, rest)` constructor.

`Phase`, `WorkType`, and `CpusetSpec` are all in
`ktstr::prelude::*`; only `std::time::Duration` needs an extra
`use` line — added on the first line of the example below:

```rust,ignore
use std::time::Duration;
use ktstr::prelude::*;

#[ktstr_test(scheduler = KTSTR_SCHED_PAYLOAD, llcs = 2, cores = 2, threads = 1)]
fn mixed_workloads(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        // Persistent CPU pressure on LLC 0 for the whole run.
        CgroupDef::named("background_spinner")
            .workers(2)
            .work_type(WorkType::SpinWait)
            .with_cpuset(CpusetSpec::Llc(0)),
        // Phased worker on LLC 1: spin 100 ms, yield for 20 ms,
        // then loop. Stresses the scheduler's wake-after-yield
        // placement repeatedly while the LLC-0 spinner keeps
        // host runqueue pressure constant.
        CgroupDef::named("phased_worker")
            .workers(2)
            .work_type(WorkType::sequence(
                Phase::Spin(Duration::from_millis(100)),
                [Phase::Yield(Duration::from_millis(20))],
            ))
            .with_cpuset(CpusetSpec::Llc(1)),
    ])
}
```

The two cgroups now exercise distinct paths concurrently:

- `background_spinner` keeps two CPUs continuously busy on LLC 0.
- `phased_worker` alternates between burning CPU and yielding on
  LLC 1, exercising the scheduler's voluntary-preemption + wakeup
  placement code paths.

Both cgroups still run for the **entire scenario duration**: the
phasing happens *within* each `phased_worker` worker's loop, while
`execute_defs` holds both cgroups across the whole run via
`HoldSpec::FULL`. To express phasing across cgroups (e.g. add
`phased_worker` only for the second half of the run), use
`execute_steps` with multiple `Step` entries -- see
[Ops and Steps](concepts/ops.md). Step 9 below adds an `Op::snapshot`
capture into a step's op list.

## Step 6: Tune execution

Several `#[ktstr_test]` attributes control how the VM runs the
scenario. The defaults are tuned for fast iteration; tune up for
longer / heavier runs:

| Attribute | Default | What it does |
|---|---|---|
| `duration_s` | `12` | Per-scenario wall-clock seconds. The framework keeps both cgroups running for `duration_s` seconds, then signals workers to stop and collects reports. |
| `watchdog_timeout_s` | `5` | sched_ext watchdog fire threshold. Applied via `scx_sched.watchdog_timeout` on 7.1+ kernels and the static `scx_watchdog_timeout` symbol on pre-7.1 kernels. When neither path is available the override silently no-ops. |
| `memory_mb` | `2048` | VM memory in MiB. |

`watchdog_timeout_s` is sched_ext's per-task stall threshold — if
a runnable task is not picked for `watchdog_timeout_s` seconds,
the scheduler exits with `SCX_EXIT_ERROR_STALL`. The scenario
duration and watchdog are independent; a 12 s scenario with a 5 s
watchdog is normal. Tune the watchdog only when the scheduler
under test is expected to legitimately leave a runnable task
parked longer than the default 5 s.

For the run we're building, set the duration to 20 s (so each
phase iteration repeats many times):

```rust,ignore
#[ktstr_test(
    scheduler = KTSTR_SCHED_PAYLOAD,
    llcs = 2,
    cores = 2,
    threads = 1,
    duration_s = 20,
)]
fn mixed_workloads(ctx: &Ctx) -> Result<AssertResult> {
    // body unchanged from Step 5 -- two cgroups via execute_defs
}
```

For the full attribute reference (auto-repro, performance mode,
topology constraints, etc.), see
[The #\[ktstr_test\] Macro](writing-tests/ktstr-test-macro.md).

## Step 7: Add assertions

Default checks already run with no configuration -- `not_starved` is
`Some(true)` in `Assert::default_checks()`, which enables:

- **Starvation** -- any worker with zero work units fails the test.
- **Fairness spread** -- per-cgroup `max(off-CPU%) - min(off-CPU%)`
  must stay under the spread threshold (release default 15%; debug
  default 35% — debug builds in small VMs show higher spread, so
  the threshold loosens automatically when `cfg!(debug_assertions)`
  is true).
- **Scheduling gaps** -- the longest wall-clock gap observed at
  work-unit checkpoints must stay under the gap threshold
  (release default 2000 ms; debug default 3000 ms — same
  `cfg!(debug_assertions)` gate as spread).

Host-side monitor checks (imbalance ratio, DSQ depth, stall
detection, fallback / keep-last event rates) are also enabled by
default with thresholds from `MonitorThresholds::DEFAULT`.

Cpuset isolation is **opt-in** -- enable it with `isolation = true`.
Override the spread threshold and add throughput-parity gates:

```rust,ignore
use std::time::Duration;
use ktstr::prelude::*;

#[ktstr_test(
    scheduler = KTSTR_SCHED_PAYLOAD,
    llcs = 2,
    cores = 2,
    threads = 1,
    duration_s = 20,
    isolation = true,
    max_spread_pct = 20.0,
    max_throughput_cv = 0.5,
    min_work_rate = 1.0,
)]
fn mixed_workloads(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        CgroupDef::named("background_spinner")
            .workers(2)
            .work_type(WorkType::SpinWait)
            .with_cpuset(CpusetSpec::Llc(0)),
        CgroupDef::named("phased_worker")
            .workers(2)
            .work_type(WorkType::sequence(
                Phase::Spin(Duration::from_millis(100)),
                [Phase::Yield(Duration::from_millis(20))],
            ))
            .with_cpuset(CpusetSpec::Llc(1)),
    ])
}
```

What each new attribute gates:

- `isolation = true` -- workers must only run on CPUs in their
  assigned cpuset; any execution on an unexpected CPU fails the test.
- `max_spread_pct = 20.0` -- per-cgroup fairness override (the
  release default is 15.0; this loosens it slightly to absorb noise
  from the phased worker's yield-driven re-placement). Bare
  `max_spread_pct = 15.0` would silently match the default and have
  no observable effect.
- `max_throughput_cv = 0.5` -- coefficient of variation of
  `work_units / cpu_time` across workers. Catches a scheduler that
  gives some workers disproportionately less effective CPU.
- `min_work_rate = 1.0` -- minimum work units per CPU-second per
  worker. Catches the case where every worker is equally slow
  (CV passes but absolute throughput is too low).

`#[ktstr_test]` exposes the full `Assert` surface (scheduling gaps,
monitor thresholds, NUMA locality, wake-latency benchmarks). See
[Checking](concepts/checking.md) for the merge chain
(`default_checks() -> Scheduler.assert -> per-test`) and the
complete threshold list.

## Step 8: Run it

Run the test with `cargo ktstr test`, scoped to this one test name:

```sh
cargo ktstr test --kernel ../linux -- -E 'test(mixed_workloads)'
```

If `cargo ktstr test` reports "kernel not found", the `--kernel` path
either points at a directory without a built vmlinux or at a kernel
the cache cannot locate. Run `cargo ktstr kernel build` to populate
the cache, or pass an explicit path to a built kernel source tree —
see [Getting Started: Build a kernel](getting-started.md#build-a-kernel)
for the resolution order.

If a probe-related error surfaces ("probe skeleton load failed",
"trigger attach failed"), re-run with `RUST_LOG=ktstr=debug` to
see the underlying libbpf reason. Common causes: missing tp_btf
target on older kernels (handled automatically by the two-phase
fallback), `CONFIG_DEBUG_INFO_BTF=n` in the guest kernel (rebuild
with BTF enabled), or a verifier rejection on a non-optional
program (the retry surfaces both the original and retry errors so
the verifier output is preserved).

`cargo ktstr test` resolves the kernel image, boots a VM with the
declared topology, runs the test as the guest's init, and reports
the result. A passing run looks like:

```text
    PASS [  11.34s] my_crate::mixed_workloads ktstr/mixed_workloads
```

A failure prints the violated threshold along with per-cgroup stats:

```text
    FAIL [  12.05s] my_crate::mixed_workloads ktstr/mixed_workloads

--- STDERR ---
ktstr_test 'mixed_workloads' [sched=scx-ktstr] [topo=1n2l2c1t] failed:
  unfair cgroup: spread=22% (10-32%) 2 workers on 2 cpus (threshold 20%)

--- stats ---
4 workers, 4 cpus, 12 migrations, worst_spread=22.4%, worst_gap=180ms
  cg0: workers=2 cpus=2 spread=22.4% gap=180ms migrations=8 iter=15230
  cg1: workers=2 cpus=2 spread=4.1% gap=120ms migrations=4 iter=14870
```

The detail line `unfair cgroup: spread=N% (min-max%) N workers on
N cpus (threshold N%)` is the exact format produced by
`assert::assert_not_starved`. Other detail-line shapes the
same producer emits:

- `tid {N} starved (0 work units)` — when a worker made no
  progress. Example:

  ```text
  ktstr_test 'mixed_workloads' [topo=1n2l2c1t] failed:
    tid 2 starved (0 work units)
  ```

- `tid {N} stuck {N}ms on cpu{N} at +{N}ms (threshold {N}ms)` —
  when a worker's longest off-CPU gap crossed
  `Assert::max_gap_ms`. Example:

  ```text
  ktstr_test 'mixed_workloads' [topo=1n2l2c1t] failed:
    tid 7 stuck 2500ms on cpu3 at +4200ms (threshold 2000ms)
  ```

- `unfair cgroup: spread={pct}% ({lo}-{hi}%)` — when per-cgroup
  fairness exceeded `max_spread_pct`. Example:

  ```text
  ktstr_test 'mixed_workloads' [topo=1n2l2c1t] failed:
    unfair cgroup: spread=22% (10-32%) 2 workers on 2 cpus (threshold 20%)
  ```

The reporting layer does NOT include the cgroup name — `cg{i}`
is the positional index in the stats roll-up (`cg0`, `cg1`, ...)
matching the `cg{i}: workers=... cpus=... spread=...` per-cgroup
stats line emitted by `test_support::eval::evaluate_vm_result`.

For the full run lifecycle, sidecar layout, and analysis workflow,
see [Running Tests](running-tests.md).

## Step 9: Capture a snapshot

Threshold-based assertions tell you something is off; snapshots tell
you *what* the scheduler's state actually was. `Op::snapshot(name)`
asks the host to freeze every vCPU long enough to read the BPF
(in-kernel program) map state, vCPU registers, and per-CPU counters
into a `FailureDumpReport` keyed by `name`, then resumes the guest.

`execute_defs` (used so far) takes a flat list of cgroups and runs
them concurrently. To inject a snapshot mid-run, switch to
`execute_steps`, which takes a list of `Step`s — each step has
`setup` cgroups, an `ops` list (where `Op::snapshot(...)` lives),
and a `hold` duration:

```rust,ignore
use ktstr::prelude::*;

#[ktstr_test(scheduler = KTSTR_SCHED_PAYLOAD, llcs = 2, cores = 2, threads = 1, duration_s = 20)]
fn mixed_workloads(ctx: &Ctx) -> Result<AssertResult> {
    execute_steps(ctx, vec![
        Step {
            setup: Setup::Defs(vec![
                CgroupDef::named("background_spinner")
                    .workers(2)
                    .work_type(WorkType::SpinWait)
                    .with_cpuset(CpusetSpec::Llc(0)),
                CgroupDef::named("phased_worker")
                    .workers(2)
                    .work_type(WorkType::SpinWait)
                    .with_cpuset(CpusetSpec::Llc(1)),
            ]),
            ops: vec![Op::snapshot("after_workload")],
            hold: HoldSpec::FULL,
        },
    ])
}
```

After the scenario completes, the captured report is keyed by name
on the active `SnapshotBridge` — the host-side store that owns the
captured `FailureDumpReport` map for the run. Downstream test code
drains it and walks scalar variables with the dotted-path accessor —
e.g. `snap.var("nr_cpus_onln").as_u64()?` reads a scheduler global
(any `.bss`/`.data`/`.rodata` symbol; `Snapshot::var` walks all
three) as a `u64`.

For the bridge wiring, the full traversal API
(`Snapshot::map`, `SnapshotEntry::get`, per-CPU narrowing,
error variants), and the symbol-driven
[`Op::watch_snapshot`](writing-tests/watch-snapshots.md) variant
that fires whenever the guest writes a kernel symbol, see
[Snapshots](writing-tests/snapshots.md).

## Step 10: Gauntlet expansion

The `#[ktstr_test]` macro doesn't just emit a single test -- it
also generates a **gauntlet** of variants that run the same body
across every accepted topology preset (single-LLC, multi-LLC,
multi-NUMA, with/without SMT).

Gauntlet variants are nextest-discovered and run with
`cargo ktstr test -- --run-ignored ignored-only -E 'test(gauntlet/)'`.
Constrain coverage with `min_llcs` / `max_llcs`, `min_cpus` /
`max_cpus`, and `requires_smt` on the attribute. See
[Gauntlet Tests](writing-tests/gauntlet-tests.md) for the full
filtering and worked examples.

## Step 11: Name and prioritize workers

Per-cgroup defaults travel through `CgroupDef`'s builder methods so
schedulers that key on `task->comm` or `task_struct->static_prio`
can be exercised with realistic, distinguishable workers:

```rust,ignore
CgroupDef::named("background_spinner")
    .workers(2)
    .comm("bg_spinner")           // prctl(PR_SET_NAME, "bg_spinner")
    .nice(10)                     // setpriority(PRIO_PROCESS, 0, 10)
    .work_type(WorkType::SpinWait)
```

- **`.comm("name")`** — every worker calls `prctl(PR_SET_NAME, name)`
  at startup. The kernel truncates `task->comm` to 15 bytes inside
  `__set_task_comm`. Distinguishes workers in `top` / `ps` output
  and in scheduler tracepoints.
- **`.nice(n)`** — every worker calls
  `setpriority(PRIO_PROCESS, 0, n)`. Values below the calling
  task's current nice require `CAP_SYS_NICE`; ktstr always runs as
  root in-VM so the full `-20..=19` range is available. Skips the
  syscall entirely when `.nice(...)` is not chained (workers
  inherit the parent's nice).
- **`.pcomm("name")`** — set the *thread-group leader*'s
  `task->comm`. Triggers ktstr's fork-then-thread spawn path:
  workers sharing a `pcomm` value coalesce into ONE forked leader
  process whose `task->group_leader->comm` is `name`, with worker
  threads inside it. Models real applications like `chrome` (pcomm)
  hosting `ThreadPoolForeg` (per-thread comm) and `java` (pcomm)
  hosting `GC Thread` / `C2 CompilerThre`.

`pcomm` is a `WorkSpec` field, NOT a `CloneMode` variant. The two
real `CloneMode` variants are `Fork` (default; each worker is its
own thread group) and `Thread` (workers share the harness's tgid as
`std::thread::spawn` threads). `pcomm` triggers an in-process
fork-then-thread shape that combines per-process leader visibility
schedulers expect with the in-process thread-spawn dispatch the
worker bodies use. `PipeIo` and `CachePipe` workers placed in a
`.pcomm(...)` cgroup run as threads inside the pcomm container;
their pipe-pair partner indices are computed within the
container's thread group, not across forked siblings.
`SignalStorm` uses `tkill` (per-task signal delivery,
`PIDTYPE_PID`) rather than `kill` (`PIDTYPE_TGID`), so the
partner-vs-self addressing is correct uniformly across `Fork` and
`Thread` modes — including inside pcomm-coalesced thread groups.

Per-`WorkSpec` overrides win over cgroup-level defaults — write
`.work(WorkSpec::default().nice(0).comm("hot_spinner"))` to opt a
specific worker out of the cgroup-level defaults.

## Step 12: Inline scheduler config

Schedulers like `scx_layered` and `scx_lavd` accept a JSON config via
`--config /path/to/file.json`. Declare the arg template + guest path
on a `Scheduler` const built via the manual builder, then supply the
inline content from the test attribute:

```rust,ignore
const LAYERED_SCHED: Scheduler = Scheduler::new("layered")
    .binary(SchedulerSpec::Discover("scx_layered"))
    .topology(1, 2, 4, 1)
    .config_file_def("--config {file}", "/include-files/layered.json");
const LAYERED_PAYLOAD: Payload = Payload::from_scheduler(&LAYERED_SCHED);

const LAYERED_CONFIG: &str = r#"{ "layers": [{ "name": "default" }] }"#;

#[ktstr_test(scheduler = LAYERED_PAYLOAD, config = LAYERED_CONFIG)]
fn layered_default(ctx: &Ctx) -> Result<AssertResult> {
    Ok(AssertResult::pass())
}
```

The framework writes `LAYERED_CONFIG` to the guest at the path
declared on the scheduler (`/include-files/layered.json`) and
substitutes `{file}` in the arg template with that path before
launching the scheduler binary. A scheduler that declares
`config_file_def` REQUIRES every test to supply `config = …`
(compile-time + runtime gate); a scheduler that doesn't declare it
REJECTS `config = …` (the content would be silently dropped). See
[The #\[ktstr_test\] Macro](writing-tests/ktstr-test-macro.md#inline-scheduler-config)
for the full pairing rules.

For schedulers whose config lives on disk on the host (no inline
content), use `Scheduler::config_file(host_path)` instead — the
host file is packed into the initramfs and `--config` is injected
into scheduler args automatically; no `config = …` on the test is
needed in that flavor.

## Step 13: Decouple virtual topology from host hardware

By default, ktstr pins vCPUs to host cores in a layout that mirrors
the declared virtual topology. A test declaring `numa_nodes = 2,
llcs = 8` cannot run on a 1-NUMA-node host — the gauntlet preset
filter rejects it. Set `no_perf_mode = true` to drop the host
mirroring and run the declared virtual topology unchanged:

```rust,ignore
#[ktstr_test(
    numa_nodes = 2,
    llcs = 8,             // 8 % 2 == 0; the macro requires divisibility
    cores = 4,
    no_perf_mode = true,  // VM built as declared, even on 1-NUMA hosts
)]
fn two_node_test(ctx: &Ctx) -> Result<AssertResult> { /* ... */ }
```

In `no_perf_mode`:
- The VM's virtual topology is built as declared via KVM vCPU
  layout, ACPI SRAT/SLIT (x86_64), or FDT cpu nodes (aarch64) —
  the guest sees the full requested NUMA / LLC structure.
- vCPU-to-host-core pinning, 2 MB hugepages, NUMA mbind, RT
  scheduling, and KVM exit suppression are skipped.
- Host topology constraints (`min_numa_nodes`, `min_llcs`,
  `requires_smt`, per-LLC CPU widths) are NOT compared against
  host hardware. The only host check that survives is "total host
  CPUs >= total vCPUs".

`no_perf_mode = true` is mutually exclusive with `performance_mode
= true` (`KtstrTestEntry::validate` rejects the combination at
runtime). Equivalent to setting `KTSTR_NO_PERF_MODE=1` per-test —
either source forces the no-perf path. See
[Performance Mode](concepts/performance-mode.md#tier-2-no-perf-mode-with-cpu-cap-reservation)
for the full lifecycle.

## Step 14: Periodic capture and temporal assertions

On-demand `Op::snapshot` (Step 9) captures the scheduler's BPF state
at a point you choose. **Periodic capture** fires automatically at
evenly-spaced points across the workload window, producing a
time-ordered `SampleSeries` (the host-side container of drained
snapshots, in capture order; `.periodic_only()` filters to
periodic-tagged samples) for temporal assertions. Periodic capture
is only useful when paired with a `post_vm` callback that drains
the bridge and asserts something about the sequence — the two
attributes belong together.

Enable periodic capture with `num_snapshots = N` and register the
host-side callback with `post_vm = function_name`. The callback
drains the bridge and runs assertions over the time-ordered series:

```rust,ignore
use ktstr::prelude::*;

fn check_dispatch_advances(result: &VmResult) -> Result<()> {
    let series = SampleSeries::from_drained(
        result.snapshot_bridge.drain_ordered_with_stats(),
    )
    .periodic_only();

    let mut v = Verdict::new();

    let nr_dispatched: SeriesField<u64> = series.bpf(
        "nr_dispatched",
        |snap| snap.var("nr_dispatched").as_u64(),
    );
    nr_dispatched.nondecreasing(&mut v);

    let r = v.into_result();
    anyhow::ensure!(r.passed, "temporal assertions failed: {:?}", r.details);
    Ok(())
}

#[ktstr_test(
    scheduler = KTSTR_SCHED_PAYLOAD,
    llcs = 2,
    cores = 2,
    threads = 1,
    duration_s = 20,
    num_snapshots = 5,
    post_vm = check_dispatch_advances,
)]
fn dispatch_advances(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        CgroupDef::named("workers").workers(2).work_type(WorkType::SpinWait),
    ])
}
```

`num_snapshots = 5` fires 5 freeze-and-capture boundaries inside the
10%-90% window of the 20 s workload — at roughly +5 s, +7 s, +10 s,
+13 s, +15 s. Each capture freezes every vCPU, reads BPF map state,
and resumes. The host watchdog deadline is extended by each freeze
duration so captures do not eat into the workload budget. The
captures are stored under `periodic_000`…`periodic_004` on the
`SnapshotBridge`.

`Verdict` is the assertion accumulator: every pattern call records
its outcome on the same `Verdict`, and `v.into_result()` consumes it
into a pass/fail `AssertResult`.

The seven temporal patterns on `SeriesField`:

| Pattern | Type | What it checks |
|---|---|---|
| `nondecreasing` | u64/f64 | Every consecutive pair: `v[i] <= v[i+1]` |
| `strictly_increasing` | u64/f64 | Every consecutive pair: `v[i] < v[i+1]` |
| `rate_within(lo, hi)` | f64 | Per-pair `delta_value / delta_ms` in `[lo, hi]` |
| `steady_within(warmup_ms, tol)` | f64 | Post-warmup values within `mean ± tol%` |
| `converges_to(target, tol, deadline_ms)` | f64 | 3 consecutive samples in `[target ± tol]` before deadline |
| `ratio_within(other, lo, hi)` | f64 | Per-sample `self / other` in `[lo, hi]` (cross-field) |
| `always_true` | bool | Every sample is `true` |

Every pattern method takes `&mut Verdict` as its first argument and
returns it, so calls chain into the same accumulator.

`SeriesField::each` provides per-sample scalar bounds:
`field.each(&mut v).at_least(1u64)`,
`field.each(&mut v).between(0.0, 100.0)`.

When a temporal pattern fails, the `AssertDetail` entries
identify the offending sample by tag and elapsed-ms timestamp.
Example for `nondecreasing` flagging a regression on
`nr_dispatched`:

```text
nr_dispatched (nondecreasing): regression at sample periodic_002 (+10000ms): \
value 41 after prior value 42 at sample periodic_001 (+7000ms)
```

The rate, steady, converges, ratio, and always-true variants emit
parallel shapes — every detail names the pattern, the specific
sample(s) involved, and the violating value, so a failing test
points at the data without re-running.

For boundary timing, spacing rules, and the bridge cap, see
[Periodic Capture](writing-tests/periodic-capture.md). For the full
projection API (`bpf`, `stats`, auto-projectors) and failure
rendering, see
[Temporal Assertions](writing-tests/temporal-assertions.md).

## Step 15: After the run — test statistics

`cargo ktstr stats` aggregates the sidecar JSON files that each test
variant writes — useful for tracking gauntlet coverage, BPF verifier
complexity, and scheduling behavior across commits. This is a
post-run CLI workflow, not part of the test definition:

```sh
cargo ktstr stats                                 # summary: gauntlet coverage, verifier, KVM stats
cargo ktstr stats list                            # list runs with date, test count, arch
cargo ktstr stats compare --a-kernel 6.14 \       # diff sidecar partitions defined by
    --b-kernel 6.15                               #   per-side --a-X / --b-X filter flags
```

Statistics are collected even on test failure (`if: !cancelled()` in
CI). For the full subcommand surface, see
[cargo-ktstr stats](running-tests/cargo-ktstr.md#stats).

## The complete test

The shape exercised by every step above, in one file. `Slow` and
`Scattershot` enable scx-ktstr's `--slow` / `--scattershot` modes via
the gauntlet (Step 10); `watchdog_timeout_s = 10` overrides the
sched_ext stall threshold (Step 6); `num_snapshots` + `post_vm`
enable periodic capture and a temporal assertion (Step 14):

```rust,ignore
use std::time::Duration;
use ktstr::prelude::*;

#[derive(Scheduler)]
#[scheduler(name = "ktstr_sched", binary = "scx-ktstr")]
enum KtstrSchedFlag {
    #[flag(args = ["--slow"])]
    Slow,
    #[flag(args = ["--scattershot"])]
    Scattershot,
}

fn check_dispatch_advances(result: &VmResult) -> Result<()> {
    let series = SampleSeries::from_drained(
        result.snapshot_bridge.drain_ordered_with_stats(),
    )
    .periodic_only();

    let mut v = Verdict::new();

    let nr_dispatched: SeriesField<u64> = series.bpf(
        "nr_dispatched",
        |snap| snap.var("nr_dispatched").as_u64(),
    );
    nr_dispatched.nondecreasing(&mut v);

    let r = v.into_result();
    anyhow::ensure!(r.passed, "temporal assertions failed: {:?}", r.details);
    Ok(())
}

#[ktstr_test(
    scheduler = KTSTR_SCHED_PAYLOAD,
    llcs = 2,
    cores = 2,
    threads = 1,
    duration_s = 20,
    watchdog_timeout_s = 10,
    isolation = true,
    max_spread_pct = 20.0,
    max_throughput_cv = 0.5,
    min_work_rate = 1.0,
    num_snapshots = 5,
    post_vm = check_dispatch_advances,
)]
fn mixed_workloads(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        CgroupDef::named("background_spinner")
            .workers(2)
            .work_type(WorkType::SpinWait)
            .with_cpuset(CpusetSpec::Llc(0)),
        CgroupDef::named("phased_worker")
            .workers(2)
            .work_type(WorkType::sequence(
                Phase::Spin(Duration::from_millis(100)),
                [Phase::Yield(Duration::from_millis(20))],
            ))
            .with_cpuset(CpusetSpec::Llc(1)),
    ])
}
```

Run it:

```sh
cargo ktstr test --kernel ../linux -- -E 'test(mixed_workloads)'
```

## What you'll see when things break

The output examples below are the shapes ktstr emits in real
runs. They're worth skimming before you ship a test so a future
failure is recognisable.

### Auto-repro probe chain

When the scheduler crashes, ktstr re-runs the scenario with BPF
probes attached and dumps the path leading to the exit. Decoded
struct fields appear inline, with `→` between fentry-captured
entry values and fexit-captured exit values:

```text
ktstr_test 'demo_host_crash_auto_repro' [sched=scx-ktstr] [topo=1n1l2c1t] failed:
  scheduler died

--- auto-repro ---
=== AUTO-PROBE: scx_exit fired ===

  ktstr_enqueue                                                   main.bpf.c:21
    task_struct *p
      pid         97
      cpus_ptr    0xf(0-3)
      dsq_id      SCX_DSQ_INVALID
      enq_flags   NONE
      slice       0
      vtime       0
      weight      100
      sticky_cpu  -1
      scx_flags   QUEUED|ENABLED
  do_enqueue_task                                               kernel/sched/ext.c:1344
    rq *rq
      cpu         1
    task_struct *p
      pid         97
      cpus_ptr    0xf(0-3)
      dsq_id      SCX_DSQ_INVALID          →  SCX_DSQ_LOCAL
      enq_flags   NONE
      slice       20000000
      vtime       0
      weight      100
      sticky_cpu  -1
      scx_flags   QUEUED|DEQD_FOR_SLEEP    →  QUEUED
```

For the probe pipeline architecture, the BTF resolution path,
event-stitching rules, and the `demo_host_crash_auto_repro`
fixture, see [Auto-Repro](running-tests/auto-repro.md).

### Failure dumps with cast-recovered pointers

The freeze coordinator builds a
[`FailureDumpReport`](architecture/monitor.md) on every snapshot,
periodic capture, and post-failure dump. Each captured map prints
as a `map <name> (type=..., value_size=..., max_entries=...)`
header followed by the rendered value (single-entry global
sections like `.bss`/`.data`) or `entry: key=...` blocks
(multi-entry maps). `u64` fields the
[cast analyzer](architecture/monitor.md#cast-analysis) flagged as
typed pointers chase to the recovered struct and print with a
`(cast→arena)` or `(cast→kernel)` annotation distinguishing them
from BTF-typed pointers; an `(sdt_alloc)` suffix is added when the
sdt_alloc bridge recovered the real payload struct from a
forward-declared pointee. A separate cross-BTF Fwd resolution
path also recovers a forward-declared pointee whose body lives
in a sibling embedded BPF object's BTF — that path adds no
annotation, the body is rendered transparently:

```text
map scx_lavd.bss (type=array, value_size=4096, max_entries=1)
.bss:
  nr_cpus_onln=4
  task_ctx_root 0xffff888103a01000 (cast→arena) → task_ctx{cpu_id=2, last_runtime_ns=12345678, nice=0}
  current_task 0xffff90124f80c000 (cast→kernel) → task_struct:
    pid=4321   weight=100
    cpus_ptr 0xffff888103b40000 → cpus={0-3}
  taskc_data 0x7f0000080000 (cast→arena (sdt_alloc)) → task_data{slice_ns=20000000, vtime=12345678}
```

A field that the analyzer cannot prove is a pointer falls back to
its raw `u64` shape, which is the prior behavior — no
test-author configuration is required either way.

### Verifier output

`cargo ktstr verifier` runs the BPF verifier against every
`declare_scheduler!`-registered scheduler's struct_ops programs
inside a real kernel and prints per-program verified-instruction
counts. The dispatcher hands off to
`cargo nextest run -E 'test(/^verifier/)'`; nextest fans out
across (scheduler × declared kernel × accepted topology preset)
cells, each cell booting its own VM. Per-cell output starts with
a banner identifying the axis values:

```text
=== ktstr_sched | kernel kernel_6_14_2 | topology tiny-1llc ===

verifier
  enqueue                                  verified_insns=42

verifier --- verifier stats ---
  processed=42  states=8/10

verifier --- scheduler log ---
func#0 @0
0: R1=ctx() R10=fp0
processed 42 insns (limit 1000000) max_states_per_insn 1 total_states 10 peak_states 8 mark_read 5
```

When the scheduler did not capture a log, the output is just the
per-program table:

```text
=== ktstr_sched | kernel kernel_6_14_2 | topology tiny-1llc ===

verifier
  enqueue                                  verified_insns=500
  dispatch                                 verified_insns=1200
  init                                     verified_insns=300
```

`--raw` disables cycle collapsing in the scheduler-log section.
`--kernel A --kernel B` runs the sweep against multiple kernels;
the cell handler walks `KTSTR_KERNEL_LIST` to match each cell's
sanitized kernel label against the resolved set. For the full
verifier-sweep model, cycle-collapse rules, and the
cell-name → kernel matching contract, see
[Verifier](running-tests/verifier.md).

## What's next

- [Custom Scenarios](writing-tests/custom-scenarios.md) -- when the
  declarative ops API is not enough and the scenario needs arbitrary
  Rust logic between phases.
- [Ops and Steps](concepts/ops.md) -- multi-phase scenarios:
  add/remove cgroups, swap cpusets, freeze, resume.
- [Watch Snapshots](writing-tests/watch-snapshots.md) --
  `Op::watch_snapshot("symbol")` registers a hardware data-write
  watchpoint (up to 3 per scenario; slot 0 is reserved for the
  error-class exit_kind trigger).
- [MemPolicy](concepts/mem-policy.md) -- NUMA-aware tests that bind
  memory allocations to specific nodes and check page locality.
- [Performance Mode](concepts/performance-mode.md) -- pinned vCPUs,
  hugepages, and LLC-exclusivity validation for benchmark-grade runs.
- [Auto-Repro](running-tests/auto-repro.md) -- on a scheduler crash,
  ktstr can boot a second VM with probes attached and dump the
  failing state automatically.
- [Recipes](recipes.md) -- task-specific guides
  (test a new scheduler, A/B compare branches, customize checking,
  benchmarking, host-state diff, ctprof).
