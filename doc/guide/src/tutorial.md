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

For the full attribute reference, see
[The #\[ktstr_test\] Macro](writing-tests/ktstr-test-macro.md).

## Step 2: Add workloads

A `CgroupDef` declares a cgroup along with the workers that will run
inside it. The builder methods configure worker count, the work each
worker performs, scheduling policy, and cpuset assignment.

Add two cgroups -- both running tight CPU spinners for now. Step 4
will swap one of them for a phased workload:

```rust,ignore
use ktstr::prelude::*;

#[ktstr_test(llcs = 1, cores = 2, threads = 1)]
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
(introduced in Step 3) restricts a cgroup to one LLC's CPUs, and
the other [`CpusetSpec`] variants narrow further.

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

## Step 3: Set topology

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
#[ktstr_test(llcs = 2, cores = 2, threads = 1)]
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

## Step 4: Compose phased work inside a cgroup

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

#[ktstr_test(llcs = 2, cores = 2, threads = 1)]
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
[Ops and Steps](concepts/ops.md). Step 8 below adds an `Op::snapshot`
capture into a step's op list.

## Step 5: Tune execution

Several `#[ktstr_test]` attributes control how the VM runs the
scenario. The defaults are tuned for fast iteration; tune up for
longer / heavier runs:

| Attribute | Default | What it does |
|---|---|---|
| `duration_s` | `12` | Per-scenario wall-clock seconds. The framework keeps both cgroups running for `duration_s` seconds, then signals workers to stop and collects reports. |
| `watchdog_timeout_s` | `5` | sched_ext watchdog fire threshold. Applied via `scx_sched.watchdog_timeout` on 7.1+ kernels and the static `scx_watchdog_timeout` symbol on pre-7.1 kernels. When neither path is available the override silently no-ops. |
| `workers_per_cgroup` | `2` | Default worker count when a `CgroupDef` does not call `.workers(n)`. Per-cgroup `.workers(n)` overrides this. |
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
    llcs = 2,
    cores = 2,
    threads = 1,
    duration_s = 20,
)]
fn mixed_workloads(ctx: &Ctx) -> Result<AssertResult> {
    // body unchanged from Step 4 -- two cgroups via execute_defs
}
```

For the full attribute reference (auto-repro, performance mode,
flag and topology constraints, etc.), see
[The #\[ktstr_test\] Macro](writing-tests/ktstr-test-macro.md).

## Step 6: Add assertions

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
    llcs = 2,
    cores = 2,
    threads = 1,
    duration_s = 20,
    isolation = true,
    max_spread_pct = 15.0,
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
- `max_spread_pct = 15.0` -- per-cgroup fairness, as above (this
  attribute also overrides the release default if you want a
  different threshold).
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

## Step 7: Run it

Run the test with `cargo ktstr test`, scoped to this one test name:

```sh
cargo ktstr test --kernel ../linux -- -E 'test(mixed_workloads)'
```

If `cargo ktstr test` reports "no kernel found", the `--kernel` path
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

```
    PASS [  11.34s] my_crate::mixed_workloads ktstr/mixed_workloads
```

A failure prints the violated threshold along with per-cgroup stats:

```
    FAIL [  12.05s] my_crate::mixed_workloads ktstr/mixed_workloads

--- STDERR ---
ktstr_test 'mixed_workloads' [topo=1n2l2c1t] failed:
  unfair cgroup: spread=22% (10-32%) 2 workers on 2 cpus (threshold 15%)

--- stats ---
4 workers, 4 cpus, 12 migrations, worst_spread=22.4%, worst_gap=180ms
  cg0: workers=2 cpus=2 spread=22.4% gap=180ms migrations=8 iter=15230
  cg1: workers=2 cpus=2 spread=4.1% gap=120ms migrations=4 iter=14870
```

The detail line `unfair cgroup: spread=N% (min-max%) N workers on
N cpus (threshold N%)` is the exact format produced by
[`assert::assert_not_starved`]. Other detail-line shapes the
same producer emits:

- `tid {N} starved (0 work units)` — when a worker made no
  progress.
- `tid {N} stuck {N}ms on cpu{N} at +{N}ms (threshold {N}ms)` —
  when a worker's longest off-CPU gap crossed
  [`Assert::max_gap_ms`]. Example:
  `tid 7 stuck 1500ms on cpu3 at +4200ms (threshold 2000ms)`.

The reporting layer does NOT include the cgroup name — `cg{i}`
is the positional index in the stats roll-up (`cg0`, `cg1`, ...)
matching the `cg{i}: workers=... cpus=... spread=...` per-cgroup
stats line emitted by [`test_support::eval::evaluate_vm_result`].

For the full run lifecycle, sidecar layout, and analysis workflow,
see [Running Tests](running-tests.md).

## Step 8: Capture a snapshot

Threshold-based assertions tell you something is off; snapshots tell
you *what* the scheduler's state actually was. `Op::snapshot(name)`
asks the host to freeze every vCPU long enough to read the BPF map
state, vCPU registers, and per-CPU counters into a
`FailureDumpReport` keyed by `name`, then resumes the guest.

Drop a snapshot into the step's `ops` list, then walk the captured
report by name with `Snapshot::var(...)`:

```rust,ignore
use ktstr::prelude::*;

// Inside a Step's ops:
ops: vec![Op::snapshot("after_workload")],
```

After the scenario completes, the captured report is keyed by name
on the active `SnapshotBridge`; downstream test code drains it and
walks scalar variables with the dotted-path accessor — e.g.
`snap.var("nr_cpus_onln").as_u64()?` reads a scheduler `.bss`
global as a `u64`.

For the bridge wiring, the full traversal API
(`Snapshot::map`, `SnapshotEntry::get`, per-CPU narrowing,
error variants), and the symbol-driven
[`Op::watch_snapshot`](writing-tests/watch-snapshots.md) variant
that fires whenever the guest writes a kernel symbol, see
[Snapshots](writing-tests/snapshots.md).

## Step 9: Gauntlet expansion

The `#[ktstr_test]` macro doesn't just emit a single test -- it
also generates a **gauntlet** of variants that run the same body
across the cartesian product of:

- Topology presets (single-LLC, multi-LLC, multi-NUMA, with/without
  SMT).
- Flag profiles (when the test references a
  `#[derive(Scheduler)]` enum).

Gauntlet variants are nextest-discovered and run with
`cargo ktstr test -- --run-ignored ignored-only -E 'test(gauntlet/)'`.
Constrain coverage with `min_llcs` / `max_llcs`, `min_cpus` /
`max_cpus`, `requires_smt`, and `required_flags` / `excluded_flags`
on the attribute. See
[Gauntlet Tests](writing-tests/gauntlet-tests.md) for the full
filtering and worked examples.

## Step 10: Name and prioritize workers

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

## Step 11: Inline scheduler config

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

## Step 12: Decouple virtual topology from host hardware

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
= true` (the macro rejects the combination). Equivalent to setting
`KTSTR_NO_PERF_MODE=1` per-test — either source forces the
no-perf path. See
[Performance Mode](concepts/performance-mode.md#tier-2-no-perf-mode-with-cpu-cap-reservation)
for the full lifecycle.

## The complete test

```rust,ignore
use std::time::Duration;
use ktstr::prelude::*;

#[ktstr_test(
    llcs = 2,
    cores = 2,
    threads = 1,
    duration_s = 20,
    isolation = true,
    max_spread_pct = 15.0,
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

Run it:

```sh
cargo ktstr test --kernel ../linux -- -E 'test(mixed_workloads)'
```

## What's next

- [Custom Scenarios](writing-tests/custom-scenarios.md) -- when the
  declarative ops API is not enough and the scenario needs arbitrary
  Rust logic between phases.
- [Ops and Steps](concepts/ops.md) -- multi-phase scenarios:
  add/remove cgroups, swap cpusets, freeze, resume.
- [Snapshots](writing-tests/snapshots.md) -- on-demand
  `Op::snapshot("name")` mid-scenario captures of guest BPF map
  state plus the typed `Snapshot` accessor for walking BTF-rendered
  values along dotted paths with structured per-field errors.
- [Watch Snapshots](writing-tests/watch-snapshots.md) --
  `Op::watch_snapshot("symbol")` registers a hardware data-write
  watchpoint (up to 3 per scenario; slot 0 is reserved for the
  error-class exit_kind trigger).
- [Periodic Capture](writing-tests/periodic-capture.md) -- cadenced
  `freeze_and_capture(false)` across the workload window (set
  `num_snapshots = N`); produces a time-ordered `SampleSeries`.
- [Temporal Assertions](writing-tests/temporal-assertions.md) --
  patterns over a `SampleSeries` (`nondecreasing`, `rate_within`,
  `steady_within`, `converges_to`, `always_true`, `ratio_within`).
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
