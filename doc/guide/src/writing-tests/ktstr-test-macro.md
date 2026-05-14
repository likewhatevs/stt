# The #\[ktstr_test\] Macro

`#[ktstr_test]` registers a function as an integration test that runs
inside a VM.

## Basic usage

```rust,ignore
use ktstr::prelude::*;

#[ktstr_test(llcs = 2, cores = 4, threads = 2)]
fn my_test(ctx: &Ctx) -> Result<AssertResult> {
    // ctx provides cgroup manager, topology, duration, etc.
    Ok(AssertResult::pass())
}
```

When a scheduler with a default topology is specified, the topology
can be omitted:

```rust,ignore
use ktstr::declare_scheduler;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    //          numa, llcs, cores/llc, threads/core
    topology = (1,    2,    4,         1),
});

#[ktstr_test(scheduler = MY_SCHED)]
fn inherited_topo(ctx: &Ctx) -> Result<AssertResult> {
    // Inherits 1n2l4c1t from MY_SCHED
    Ok(AssertResult::pass())
}
```

`declare_scheduler!` emits a `pub static MY_SCHED: Scheduler` and
registers a private linkme static in the `KTSTR_SCHEDULERS`
distributed slice. The `scheduler =` slot expects
`&'static Scheduler` — pass the bare `MY_SCHED` ident; the macro
takes a reference internally.

The function must have signature
`fn(&ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult>`.

## What the macro generates

1. Renames the function to `__ktstr_inner_{name}`.
2. Registers it in the `KTSTR_TESTS` distributed slice via linkme.
3. Emits a `#[test]` wrapper that calls `run_ktstr_test()`.

The `#[test]` wrapper boots a VM with the specified topology and runs
the function inside it.

## Attributes

All attributes are optional with defaults.

### Topology

| Attribute | Default | Description |
|---|---|---|
| `llcs` | inherited | Number of LLCs |
| `numa_nodes` | inherited | Number of NUMA nodes |
| `cores` | inherited | Cores per LLC |
| `threads` | inherited | Threads per core |
| `memory_mb` | 2048 | VM memory in MB |

Each dimension independently inherits from `Scheduler.topology` when
a `scheduler` is specified and that dimension is not explicitly set.
Without a scheduler, unset dimensions use macro defaults (numa_nodes=1,
llcs=1, cores=2, threads=1). The default is a single-NUMA topology,
so most tests do not need to set `numa_nodes`. See
[Default topology](scheduler-definitions.md#default-topology).

### Scheduler

| Attribute | Default | Description |
|---|---|---|
| `scheduler = CONST` | `&Scheduler::EEVDF` | Rust const path to a `&'static Scheduler`. The bare const emitted by `declare_scheduler!` (e.g. `MY_SCHED`) is the expected form. The default `Scheduler::EEVDF` runs tests under the kernel's default scheduler (EEVDF on Linux 6.6+) so tests without an explicit `scheduler =` run under the kernel default. |
| `extra_sched_args = [...]` | `[]` | Extra CLI args for the scheduler, appended after `Scheduler::sched_args`. |
| `watchdog_timeout_s` | 5 | scx watchdog override (seconds). Applied via `scx_sched.watchdog_timeout` on 7.1+ kernels (BTF-detected) and via the static `scx_watchdog_timeout` symbol on pre-7.1 kernels. When neither path is available the override silently no-ops. |

### Payloads

| Attribute | Default | Description |
|---|---|---|
| `payload = CONST` | `None` | Rust const path to a binary-kind `Payload` (`PayloadKind::Binary`). Populates `KtstrTestEntry::payload`; the test body can run it via `ctx.payload(&CONST)`. Scheduler-kind payloads are rejected at compile time — use the `scheduler = …` slot for those. |
| `workloads = [CONST, …]` | `[]` | Array of binary-kind `Payload` const paths composed alongside the primary `payload`. Each entry is runnable from the test body via `ctx.payload(&CONST)`; the include-file pipeline packages every referenced binary into the guest automatically. |
| `extra_include_files = ["path", …]` | `[]` | Array of string-literal paths to extra host-side files (datasets, fixture configs, helper scripts) that the framework packages into the guest initramfs alongside the binaries declared by `scheduler` / `payload` / `workloads`. Maps onto `KtstrTestEntry::extra_include_files` (`&'static [&'static str]`); union with per-payload `Payload::include_files` is computed at run time via `KtstrTestEntry::all_include_files`. Use this slot for test-level dependencies that don't belong on a specific `Payload`. |

See [Payload Definitions](scheduler-definitions.md#derive-payload) for
authoring new `Payload` fixtures; `tests/common/fixtures.rs` carries
reusable examples (`SCHBENCH`, `SCHBENCH_HINTED`, `SCHBENCH_JSON`).

### Checking

| Attribute | Default | Description |
|---|---|---|
| `not_starved` | inherited | Enable starvation (zero work units), fairness spread, and scheduling gap checks |
| `isolation` | inherited | Enable cpuset isolation check (workers must stay on assigned CPUs) |
| `max_gap_ms` | inherited | Max scheduling gap threshold |
| `max_spread_pct` | inherited | Max fairness spread threshold |
| `max_throughput_cv` | inherited | Max coefficient of variation for worker throughput |
| `min_work_rate` | inherited | Minimum work_units per CPU-second per worker |
| `max_imbalance_ratio` | inherited | Monitor imbalance ratio |
| `max_local_dsq_depth` | inherited | Monitor DSQ depth |
| `fail_on_stall` | inherited | Fail on stall detection |
| `sustained_samples` | inherited | Sample window for sustained violations |
| `max_fallback_rate` | inherited | Max fallback event rate |
| `max_keep_last_rate` | inherited | Max keep-last event rate |
| `max_p99_wake_latency_ns` | inherited | Max p99 wake latency in nanoseconds |
| `max_wake_latency_cv` | inherited | Max wake latency coefficient of variation |
| `min_iteration_rate` | inherited | Minimum iterations per wall-clock second per worker |
| `max_migration_ratio` | inherited | Max migration ratio (migrations/iterations) per cgroup |
| `min_page_locality` | inherited | Min fraction of pages on expected NUMA nodes (0.0-1.0) |
| `max_cross_node_migration_ratio` | inherited | Max ratio of NUMA-migrated pages to total pages (0.0-1.0) |
| `max_slow_tier_ratio` | inherited | Max fraction of pages on memory-only (CXL) nodes (0.0-1.0) |

`not_starved = true` enables three distinct checks: starvation (any
worker with zero work units), fairness spread (max-min off-CPU% below
`max_spread_pct`), and scheduling gaps (longest gap below `max_gap_ms`).
Each threshold can be overridden independently. See
[Customize Checking](../recipes/custom-checking.md) for
override examples and [Checking](../concepts/checking.md) for
the merge chain.

### Topology constraints

| Attribute | Default | Description |
|---|---|---|
| `min_llcs` | 1 | Minimum LLCs for gauntlet topology filtering |
| `max_llcs` | 12 | Maximum LLCs for gauntlet topology filtering |
| `min_cpus` | 1 | Minimum total CPU count for gauntlet topology filtering |
| `max_cpus` | 192 | Maximum total CPU count for gauntlet topology filtering |
| `min_numa_nodes` | 1 | Minimum NUMA nodes for gauntlet topology filtering |
| `max_numa_nodes` | 1 | Maximum NUMA nodes for gauntlet topology filtering |
| `requires_smt` | `false` | Require SMT (threads > 1) topologies. On aarch64 the gauntlet ships only non-SMT presets, so any test with `requires_smt = true` is skipped entirely on that arch. |

The gauntlet skips presets that do not satisfy these constraints.
Multi-NUMA presets are excluded by default (`max_numa_nodes = 1`).
See [Gauntlet](../running-tests/gauntlet.md#constraint-filtering)
for filtering rules and
[Gauntlet Tests](gauntlet-tests.md#worked-example) for a worked
example.

### Execution

| Attribute | Default | Description |
|---|---|---|
| `auto_repro` | `true` | On scheduler crash, boot a second VM with probes attached. Set to `false` for fast iteration. |
| `performance_mode` | `false` | Pin vCPUs to host cores, hugepages, NUMA mbind, RT scheduling, LLC exclusivity validation |
| `no_perf_mode` | `false` | Decouple the virtual topology from host hardware: build the VM with the declared `numa_nodes` / `llcs` / `cores` / `threads` even on smaller hosts; skip vCPU pinning, hugepages, NUMA mbind, RT scheduling, and KVM exit suppression; relax gauntlet preset filtering to the single "host has enough total CPUs" check. Mutually exclusive with `performance_mode = true` (validated at runtime by `KtstrTestEntry::validate`). Equivalent to setting `KTSTR_NO_PERF_MODE=1` per-test — either source forces the no-perf path. See [Performance Mode](../concepts/performance-mode.md#tier-2-no-perf-mode-with-cpu-cap-reservation). |
| `duration_s` | 12 | Per-scenario duration in seconds |
| `expect_err` | `false` | Test expects `run_ktstr_test` to return `Err`; disables auto-repro |
| `bpf_map_write = CONST` | empty | Rust const path to a `BpfMapWrite`; host writes this value to a BPF map after the scheduler loads. The entry field is a slice; the macro wraps the single path in a one-element slice. |
| `host_only` | `false` | Run the test function directly on the host instead of inside a VM. Use for tests that need host tools (e.g. cargo, nested VMs) unavailable in the guest initramfs. |
| `num_snapshots = N` | `0` | Fire `N` periodic `freeze_and_capture(false)` boundaries inside the workload's 10 %–90 % window; each capture is stored on the host `SnapshotBridge` under `periodic_NNN`. `0` disables periodic capture entirely. Validated against `MAX_STORED_SNAPSHOTS` (= 64), `host_only = true`, and a 100 ms minimum-spacing rule. See [Periodic Capture](periodic-capture.md) and [Temporal Assertions](temporal-assertions.md). |
| `cleanup_budget_ms = N` | `None` | Sub-watchdog cap on host-side VM teardown wall time. When the budget is exceeded the test's `AssertResult` is folded with a failing `AssertDetail`. `None` disables the check. |
| `post_vm = PATH` | `None` | Host-side callback invoked after `vm.run()` returns. Signature: `fn(&VmResult) -> anyhow::Result<()>`. Use for assertions that need host-side state — e.g. draining `VmResult.snapshot_bridge` for periodic-capture analysis (see [Periodic Capture](periodic-capture.md)). |
| `config = EXPR` | `None` | Inline scheduler config content (string literal or path to a `const &'static str`). Written to the guest path declared by the scheduler's `config_file_def`; the framework substitutes `{file}` in the scheduler's arg template with the guest path. Required when the scheduler declares `config_file_def`; rejected when it doesn't. The pairing is enforced at compile time via a `const` assertion against `Payload::config_file_def`, and again at runtime by `KtstrTestEntry::validate`. See [Inline scheduler config](#inline-scheduler-config). |

See [Performance Mode](../concepts/performance-mode.md) for details on
what `performance_mode` enables, prerequisites, and validation behavior.

## Inline scheduler config

Some schedulers (e.g. `scx_layered`, `scx_lavd`) accept a JSON config
file via a CLI argument like `--config /path/to/config.json`. Two
pieces wire this into a test:

1. **Scheduler declaration** — the `Scheduler` builder declares the
   arg template and the guest path via `.config_file_def`:

   ```rust,ignore
   const LAYERED_SCHED: Scheduler = Scheduler::new("layered")
       .binary(SchedulerSpec::Discover("scx_layered"))
       .config_file_def("--config {file}", "/include-files/layered.json");
   ```

   `{file}` in the arg template is replaced with the guest path. The
   framework `mkdir -p`s the parent and writes the config content to
   `/include-files/layered.json` inside the guest before the
   scheduler binary starts.

2. **Test attribute** — the test supplies the inline JSON via
   `config = …`:

   ```rust,ignore
   const LAYERED_CONFIG: &str = r#"{ "layers": [...] }"#;

   #[ktstr_test(scheduler = LAYERED_SCHED, config = LAYERED_CONFIG)]
   fn layered_test(ctx: &Ctx) -> Result<AssertResult> {
       Ok(AssertResult::pass())
   }
   ```

   `config = "..."` (string literal) and `config = SOME_CONST` (path
   to a `const &'static str`) are both accepted.

The pairing gate is bidirectional:
- A scheduler with `config_file_def` set **requires** `config = …`
  on every test (otherwise the scheduler binary would launch
  without `--config`).
- A scheduler without `config_file_def` **rejects** `config = …` on
  the test (the content would be silently dropped at dispatch).

Both halves are validated at compile time via a `const` assertion
emitted by the macro AND at runtime by `KtstrTestEntry::validate`,
so direct programmatic-entry construction sees the same gate.

For schedulers that take a config file from a host-side path
instead of inline content, use `Scheduler::config_file(host_path)`
instead of `config_file_def`. The framework packs the host file into
the initramfs at `/include-files/{filename}` and prepends `--config
/include-files/{filename}` to scheduler args; no `config = …` on
the test is needed in that flavor.

## Example with custom scheduler

Define the scheduler with `declare_scheduler!` (see
[Scheduler Definitions](scheduler-definitions.md)), then
reference it in `#[ktstr_test]`:

```rust,ignore
use ktstr::declare_scheduler;
use ktstr::prelude::*;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    topology = (1, 2, 4, 1),
    sched_args = ["--enable-llc", "--enable-stealing"],
});

#[ktstr_test(
    scheduler = MY_SCHED,
    not_starved = true,
    max_gap_ms = 5000,
)]
fn my_sched_basic(ctx: &Ctx) -> Result<AssertResult> {
    // Inherits 1n2l4c1t from MY_SCHED
    Ok(AssertResult::pass())
}
```

`declare_scheduler!` emits a `pub static MY_SCHED: Scheduler`
and registers it in the `KTSTR_SCHEDULERS` distributed slice via
a private linkme static so `cargo ktstr verifier` discovers it.
The bare `MY_SCHED` ident is what `#[ktstr_test(scheduler = ...)]`
expects. See
[Scheduler Definitions](scheduler-definitions.md#defining-a-scheduler)
for the full macro grammar.

For the manual builder pattern (no distributed-slice
registration), see
[Scheduler Definitions: Manual definition](scheduler-definitions.md#manual-definition).
