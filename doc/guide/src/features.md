<style>
:root { --content-max-width: 1000px; }
details { margin-bottom: 0.75em; }
</style>

# Features

ktstr is a test framework for Linux process schedulers.
See [Overview](overview.md) for a quick introduction with code
examples.

## Supported kernels

ktstr's runtime dispatches to per-kernel-version fallback paths for
the watchdog timeout and event counters. CI explicitly exercises
6.14 and 7.0 on both x86_64 and aarch64. On 7.1+ kernels the watchdog
override uses `scx_sched.watchdog_timeout` via BTF detection;
older kernels use the static `scx_watchdog_timeout` symbol.

Event counters follow a different layout split: 6.18+ kernels
(backported to 6.17.7+ stable) read via
`scx_sched.pcpu -> scx_sched_pcpu.event_stats`; 6.16-6.17 kernels
read `scx_sched.event_stats_cpu` directly. When neither path is
available, event-counter sampling is silently disabled.

## Testing

<details>
<summary><b>Real kernel, clean slate, x86/arm parity</b> — every VM test boots its own Linux kernel in KVM, fresh state each run</summary>

Tests boot a real Linux kernel in a KVM virtual machine with
configurable topology: NUMA nodes, LLCs, cores per LLC, threads per core.
Multi-NUMA topologies produce NUMA domains via ACPI SRAT/SLIT/HMAT
tables on x86_64. On aarch64, CPU topology is described via FDT cpu
nodes with MPIDR affinity. Both architectures are supported (24
topology presets on x86_64, 14 on aarch64).
See [Gauntlet](running-tests/gauntlet.md) for the full preset list.

</details>

<details>
<summary><b>Fast boot</b> — compressed initramfs SHM cache with COW overlay, VMs boot in ms, not s</summary>

The initramfs base (test binary + busybox + shared libraries) is
LZ4-compressed and cached in a shared memory segment. Concurrent VMs
COW-map the cached base into guest memory, avoiding per-VM
compression and copy. Per-test arguments are packed as a small
suffix appended to the cached base. Result: VM
boot time is dominated by kernel init, not initramfs preparation.

</details>

<details>
<summary><b>Automatic shared library resolution</b> — recursive DT_NEEDED discovery, no need to link statically</summary>

Shared library dependencies for the test binary and any injected
host files are resolved automatically by walking `DT_NEEDED` entries
in ELF headers. The framework builds a complete closure of transitive
dependencies — no manual `.so` lists or `LD_LIBRARY_PATH` hacks.

</details>

<details>
<summary><b>Bare-metal mode</b> — run the same scenarios on real hardware without VMs</summary>

`cargo ktstr export` packages a registered test as a self-extracting
`.run` script that reproduces the scenario on bare metal without
a VM. The runfile validates host topology and sched_ext support,
then dispatches the test directly under whatever scheduler is
already active. Used for testing under production schedulers and
real topology.

</details>

<details>
<summary><b>Declarative scheduler registration</b> — one macro declares the binary, default topology, kernels, and assertions</summary>

Tests load [sched_ext](https://github.com/sched-ext/scx) schedulers
via BPF struct_ops inside the VM. The
[`declare_scheduler!`](writing-tests/scheduler-definitions.md)
macro registers a scheduler in the `KTSTR_SCHEDULERS` distributed
slice — binary path, default topology, kernel filter for the
verifier sweep, assertion overrides, and always-on CLI args all
land in one declaration that tests reference via the bare const
ident the macro emits.

```rust,ignore
use ktstr::declare_scheduler;

declare_scheduler!(MITOSIS, {
    name = "mitosis",
    binary = "scx_mitosis",
    topology = (1, 2, 4, 1),
    sched_args = ["--exit-dump-len", "1048576"],
});
```

Without a `scheduler = …` attribute on `#[ktstr_test]`, tests run
under the kernel's default scheduler (EEVDF).

</details>

<details>
<summary><b>Data-driven scenarios</b> — declare what you want, the framework handles cgroups, cpusets, and workers</summary>

Scenarios are composable sequences of
[steps and ops](concepts/ops.md). You declare intent as data —
the framework creates cgroups, assigns cpusets, spawns workers,
sets scheduling policies, and manages affinity. 50+
[canned scenarios](concepts/scenarios.md) across 8 scenario
submodules cover basic, cpuset, dynamic, stress, interaction,
affinity, nested, and performance patterns. (The `ops` module
is the underlying DSL that every scenario is expressed in, not
a scenario category; the `scenarios` module is the top-level
catalog aggregator.)

**API types:**
- `CgroupDef` — declarative cgroup: name + cpuset + workload(s)
- `Step` — sequence of ops followed by a hold period
- `Op` — atomic operation (add/remove cgroup, set/swap/clear cpuset, spawn, stop, set affinity, move tasks)
- `CpusetSpec` — topology-relative cpuset (LLC-aligned, disjoint, overlapping, range, exact)
- `HoldSpec` — hold duration (fractional, fixed, or looped)
- `AffinityIntent` — per-worker affinity (inherit, random subset, LLC-aligned, cross-cgroup, single CPU, exact)
- `SchedPolicy` — Linux scheduling policy (Normal, Batch, Idle, FIFO, RoundRobin)
- `WorkSpec` — workload definition for a group of workers

</details>

<details>
<summary><b>Gauntlet</b> — one test declaration, dozens of topology variants with budget-aware CI selection</summary>

A single [`#[ktstr_test]`](writing-tests/ktstr-test-macro.md)
auto-expands across topology presets. Multi-kernel runs
(`cargo ktstr test --kernel A --kernel B`) add the kernel as
an additional dimension. Budget-based selection
(`KTSTR_BUDGET_SECS`) picks the subset that maximizes coverage
within a CI time limit.

**Constraint attributes:**
- `min_llcs`, `max_llcs`, `min_cpus`, `max_cpus`, `min_numa_nodes`, `max_numa_nodes`, `requires_smt` — topology gates
- `extra_sched_args` — per-test scheduler CLI arguments

</details>

<details>
<summary><b><code>#[ktstr_test]</code> proc macro</b> — zero-boilerplate test declaration with nextest integration</summary>

Declares tests with topology, scheduler, and constraint attributes.
Generates both nextest-compatible entries and standard `#[test]`
wrappers. No custom harness or `main()` needed.
See [The #\[ktstr_test\] Macro](writing-tests/ktstr-test-macro.md).

</details>

<details>
<summary><b>Library-first</b> — add as a dev-dependency, write tests in your own crate</summary>

Add `ktstr` as a dev-dependency. `ktstr::prelude` re-exports
all test-authoring types.
See [Getting Started](getting-started.md) for setup.

</details>

<details>
<summary><b>Automatic lifecycle</b> — boot, load scheduler, run scenario, collect results, shutdown — all handled</summary>

The framework manages the full VM lifecycle: boot, scheduler start,
scenario execution, result collection, and shutdown. Bidirectional
SHM signal slots coordinate graceful shutdown, BPF map writes, and
readiness gates between host and guest.

</details>

<details>
<summary><b>38 work types</b> — configurable workload profiles for different scheduling pressures</summary>

Workers are `fork()`ed processes placed in cgroups:

- `SpinWait` — tight CPU spin loop
- `YieldHeavy` — repeated sched_yield with minimal CPU work
- `Mixed` — CPU spin burst followed by sched_yield
- `AluHot` — dependent integer multiply chain at high IPC (≥ 2.0)
- `SmtSiblingSpin` — paired PAUSE-spin pinned across two SMT siblings
- `IpcVariance` — alternating high-IPC (multiplies) / low-IPC (cache touches) phases
- `IoSyncWrite` — 16 × 4 KB pwrites + fdatasync per iteration (O_SYNC)
- `IoRandRead` — 4 KB random pread (O_DIRECT)
- `IoConvoy` — interleaved sequential pwrite + random pread with periodic fdatasync (O_DIRECT)
- `Bursty` — CPU burst then sleep (parameterized via `Duration`)
- `IdleChurn` — CPU burst then `nanosleep` (hrtimer + idle-class path)
- `PipeIo` — CPU burst then pipe exchange (cross-CPU wake placement)
- `FutexPingPong` — paired futex wait/wake (non-WF_SYNC)
- `FutexFanOut` — 1:N fan-out wake
- `CachePressure` — strided RMW sized to pressure L1
- `CacheYield` — cache pressure + sched_yield
- `CachePipe` — cache pressure + pipe exchange
- `Sequence` — compound: loop through phases
- `ForkExit` — rapid fork+_exit cycling
- `NiceSweep` — cycle nice level from -20 to 19
- `AffinityChurn` — rapid self-directed sched_setaffinity
- `PolicyChurn` — cycle SCHED_OTHER → BATCH → IDLE (→ FIFO/RR with CAP_SYS_NICE)
- `NumaMigrationChurn` — rotate sched_setaffinity across NUMA nodes
- `CgroupChurn` — cycle cgroup membership between sibling cgroups
- `FanOutCompute` — messenger/worker fan-out with matrix-multiply compute
- `AsymmetricWaker` — paired workers in mismatched scheduling classes share one futex word
- `WakeChain` — ring of waker-wakee hops (Pipe with WF_SYNC, or Futex)
- `EpollStorm` — eventfd producers + epoll_wait consumers
- `ThunderingHerd` — N waiters on one global futex word; broadcast wake
- `PageFaultChurn` — rapid mmap/fault/MADV_DONTNEED cycling
- `NumaWorkingSetSweep` — rotate working-set memory across NUMA nodes via mbind
- `MutexContention` — N-way futex mutex contention
- `PriorityInversion` — three-tier lock contention (Pi or Plain futex)
- `ProducerConsumerImbalance` — unbalanced producer/consumer pipeline (queue grows)
- `SignalStorm` — paired workers fire tkill(partner, SIGUSR1) between CPU bursts
- `PreemptStorm` — one SCHED_FIFO worker preempts CFS spinners at ~kHz
- `RtStarvation` — SCHED_FIFO workers monopolise CPU; SCHED_NORMAL workers starve
- `Custom` — user-supplied work function

See [WorkType](concepts/work-types.md).

</details>

## Observability

<details>
<summary><b>Zero-overhead introspection</b> — read/write kernel state and BPF maps from the host w/o the guest knowing</summary>

All observability is built on direct read/write of guest physical
memory from the host via the KVM memory mapping, with page table
walks for dynamically allocated addresses. No guest-side
instrumentation, no BPF syscalls — the observer does not perturb the
scheduler under test.

**Kernel state**: per-CPU runqueues, sched_domain trees, schedstat
counters, and sched_ext event counters — read via BTF-resolved
struct offsets from vmlinux.
See [Monitor](architecture/monitor.md).

**BPF state**: maps discovered by walking kernel data structures
through page table translation. Array and percpu_array maps support
typed field access via BPF program BTF; hash maps return raw
key-value pairs. Read/write — write enables host-initiated crash
reproduction.

**Types:** `GuestMem`, `GuestKernel`, `BpfMapAccessor`

</details>

<details>
<summary><b>Cast analysis</b> — recover typed pointers from <code>u64</code> map fields automatically; no annotations required</summary>

Schedulers stash kernel kptrs (`task_struct *`, `cgroup *`, …) and
arena pointers in BPF map fields the BTF declares as `u64` because
BTF cannot express a pointer to a per-allocation type. The cast
analyzer walks the scheduler's `.bpf.o` instruction stream, tracks
register state across LDX / STX / stack-spill / kfunc-return, and
records every `(source_struct, field_offset) → target_struct`
mapping it can prove from the program's own access pattern.

The renderer feeds those mappings into `render_cast_pointer` so a
field that previously surfaced as a raw `0xffff…` integer now
chases through to the target struct's fields and prints with a
`(cast→arena)` or `(cast→kernel)` annotation distinguishing
cast-recovered pointers from BTF-typed ones. Failure dumps,
periodic captures, and on-demand snapshots all benefit
automatically.

A complementary `sdt_alloc` bridge recovers a chase target's real
struct id when the scheduler's program BTF declares the pointee as
a `BTF_KIND_FWD` forward declaration (the typical shape for
`struct sdt_data __arena *` fields whose body lives in a separate
library BTF). The freeze pre-pass populates a
`slot_start → ArenaSlotInfo` range index from each live
`sdt_alloc` allocator slot — one entry per slot, carrying
`elem_size`, `header_size`, and the resolved payload type id.
When a chase lands on a Fwd terminal, the renderer range-looks
up the slot the chased address falls in and renders the recovered
payload struct from the slot's payload start (skipping the
`union sdt_id` header when the chased pointer lands at slot-start
rather than payload-start). The result carries an `sdt_alloc`
annotation suffix: `(sdt_alloc)` for the BTF-typed `Type::Ptr`
arm, or `cast→arena (sdt_alloc)` / `cast→kernel (sdt_alloc)`
when the chase originated from a cast-analyzer hit.

A parallel cross-BTF Fwd resolution path covers a different
multi-BTF shape: a `BTF_KIND_FWD` whose body lives in a sibling
embedded BPF object's BTF rather than an `sdt_alloc` slot — the
typical scheduler shape where one `.bpf.c` declares
`struct cgx_target;` (forward) and another defines the body. The
cast-analysis pre-pass builds a name-keyed index over every
parsed embedded program BTF (one entry per complete `!is_fwd`
`Struct` / `Union`; first-write-wins on duplicate names; anonymous
types skipped). When a chase target survives the local same-BTF
Fwd resolve as a Fwd, the renderer consults the cross-BTF index
by name (matching aggregate kind — `struct` vs `union`); a hit
switches the recursion to the resolved sibling BTF and renders
the full body. No new annotation is introduced — the recovered
subtree carries whatever annotation it would have had if the
struct body lived in the entry BTF.

Runs unconditionally on every scheduler load; no test-author
configuration. False negatives (a missed cast — renderer falls
back to raw `u64`, the prior behavior) are acceptable; false
positives (a misidentified cast) are not, so the analyzer is
deliberately conservative on conflicting evidence and branch
joins. See [Monitor → Cast analysis](architecture/monitor.md#cast-analysis).

</details>

<details>
<summary><b>Unified timeline</b> — correlate scenario phases with scheduler telemetry</summary>

Stimulus events (cgroup ops, cpuset changes, step transitions)
correlated with monitor samples for per-phase scheduler behavior
analysis. Each event carries timestamps, operation details, and
cumulative worker iteration counts.

</details>

<details>
<summary><b>Periodic capture + temporal assertions</b> — cadenced sampling across the workload window with monotonicity, rate, steady-state, convergence, and ratio patterns</summary>

`#[ktstr_test(num_snapshots = N)]` fires `N` host-side
`freeze_and_capture` boundaries inside the workload's 10 %–90 %
window, anchored at the first `MSG_TYPE_SCENARIO_START`. Each
capture is stored on the `SnapshotBridge` under `periodic_NNN`
along with the parallel scx_stats JSON observed pre-freeze and a
pause-adjusted elapsed-ms timestamp.

A `post_vm` callback drains the bridge into a `SampleSeries` and
projects per-sample columns (`SeriesField<T>`) along the BPF or
stats axis. Seven temporal patterns evaluate the projections:

- `nondecreasing` — counter monotonicity (`v[i] <= v[i+1]`)
- `strictly_increasing` — strict counter monotonicity (`v[i] < v[i+1]`)
- `rate_within(lo, hi)` — bounded delta-per-millisecond
- `steady_within(warmup_ms, tolerance)` — post-warmup mean band
- `converges_to(target, tolerance, deadline_ms)` — three-consecutive-in-band witness before deadline
- `always_true` — boolean invariant at every sample
- `ratio_within(other, lo, hi)` — cross-field correlation between two series

Per-sample projection errors render with the underlying
`SnapshotError` variant (PlaceholderSample, MissingStats,
FieldNotFound, TypeMismatch, …) so coverage gaps surface with
their cause without re-running. See
[Periodic Capture](writing-tests/periodic-capture.md) and
[Temporal Assertions](writing-tests/temporal-assertions.md).

</details>

<details>
<summary><b>Worker comm / nice / pcomm</b> — set <code>task->comm</code>, nice level, and thread-group leader name on every worker</summary>

`CgroupDef::comm("name")` calls `prctl(PR_SET_NAME)` on every
worker. `CgroupDef::nice(n)` calls `setpriority(PRIO_PROCESS, 0, n)`
on every worker. `CgroupDef::pcomm("name")` triggers ktstr's
fork-then-thread spawn path: workers sharing a pcomm value coalesce
into ONE forked thread-group leader whose `task->group_leader->comm`
is the pcomm string, with worker threads inside it. Each worker
thread additionally sets its own `task->comm` via the per-WorkSpec
`.comm()`. Models real applications like `chrome` (pcomm) hosting
`ThreadPoolForeg` (per-thread comm). `PipeIo`/`CachePipe` and
`SignalStorm` work correctly under Fork and Thread clone modes,
including inside pcomm-coalesced thread groups. See [Tutorial: Step 11](tutorial.md#step-11-name-and-prioritize-workers).

</details>

<details>
<summary><b>Inline scheduler configs</b> — pass JSON config strings directly into the test, framework writes to guest path</summary>

Schedulers that take a `--config` JSON file (`scx_layered`,
`scx_lavd`, …) declare the arg template + guest path via
`Scheduler::config_file_def(arg_template, guest_path)`. Tests
supply the inline JSON via `#[ktstr_test(config = LAYERED_CONFIG)]`.
The framework writes the content to a temp file, packs it into the
initramfs at the guest path, and substitutes `{file}` in the arg
template before launching the scheduler. A bidirectional pairing
gate (compile time + runtime) catches mismatched declarations: a
scheduler with `config_file_def` REQUIRES `config = …` on every
test, and a scheduler without it REJECTS `config = …`. See
[Tutorial: Step 12](tutorial.md#step-12-inline-scheduler-config) and
[The #\[ktstr_test\] Macro](writing-tests/ktstr-test-macro.md#inline-scheduler-config).

</details>

<details>
<summary><b>no_perf_mode</b> — decouple virtual topology from host hardware for tests with NUMA / LLC counts the host can't satisfy</summary>

`#[ktstr_test(no_perf_mode = true)]` (or `KTSTR_NO_PERF_MODE=1`)
builds the VM with the declared `numa_nodes` / `llcs` / `cores` /
`threads` even on smaller hosts. vCPU pinning, hugepages, NUMA
mbind, RT scheduling, and KVM exit suppression are skipped, and
gauntlet preset filtering relaxes host-topology checks to the
single "host has enough total CPUs" inequality. Mutually exclusive
with `performance_mode = true` (validated at runtime by `KtstrTestEntry::validate`). See
[Tutorial: Step 13](tutorial.md#step-13-decouple-virtual-topology-from-host-hardware)
and [Performance Mode](concepts/performance-mode.md#tier-2-no-perf-mode-with-cpu-cap-reservation).

</details>

<details>
<summary><b>Statistical regression detection</b> — Polars-powered analysis across combinatoric test matrices</summary>

[Polars](https://pola.rs)-powered aggregation computes scheduling
metrics across runs. Run-to-run compare with dual-gate
significance thresholds (absolute and relative) catches regressions
that single-run assertions miss.

**Metrics:**
- `worst_spread` — CPU time fairness (0.0 = perfect)
- `worst_gap_ms` — longest scheduling gap
- `total_migrations` / `worst_migration_ratio` — cross-CPU migration volume
- `max_imbalance_ratio` — runqueue length imbalance
- `p99_wake_latency_us` — tail wake-to-run latency
- `mean_run_delay_us` — mean schedstat run delay
- `total_iterations` — throughput

</details>

## Debugging

<details>
<summary><b>Auto-repro</b> — automatically captures function arguments and struct state at crash-path call sites</summary>

On scheduler crash, extracts the crash stack and discovers
struct_ops callbacks. Attaches BPF kprobes and fentry/fexit probes,
triggers on `sched_ext_exit`, and reruns the scenario to capture
function arguments and struct field state at each crash-path call
site. See [Auto-Repro](running-tests/auto-repro.md).

</details>

<details>
<summary><b>Interactive shell</b> — busybox shell inside the VM with host file injection (debugging, not tests — too slow)</summary>

`ktstr shell` boots a VM with busybox and drops into an interactive
shell. `--include-files` injects host binaries and libraries with
automatic shared library resolution.

</details>

<details>
<summary><b>--exec mode</b> — run commands inside the VM non-interactively</summary>

`ktstr shell --exec "command"` runs a command inside the VM and
exits.

</details>

## Infrastructure

<details>
<summary><b>Kernel management</b> — build, cache, and auto-discover kernels from any source</summary>

`ktstr kernel build` builds and caches kernel images from version
numbers, local source paths, or git URLs. Automatic kernel discovery
resolves cached images, host kernels, and CI-provided paths without
manual configuration.

</details>

<details>
<summary><b>Performance mode</b> — host-side isolation-ish and topology mirroring for maybe workable results</summary>

vCPU pinning, 2MB hugepages with pre-fault allocation, NUMA mbind,
RT scheduling (SCHED_FIFO), KVM exit suppression (PAUSE + HLT), and
KVM_HINTS_REALTIME CPUID — isolates the VM from host noise for
reproducible measurements. Topology mirroring maps the VM's LLC
structure to match the host's physical layout, so cache-aware
scheduling decisions in the guest reflect real hardware behavior
rather than synthetic geometry. Kinda.
See [Performance Mode](concepts/performance-mode.md).

</details>

<details>
<summary><b>Resource-budget coordination</b> — <code>--cpu-cap N</code> bounds concurrent kernel builds and no-perf-mode VMs per host</summary>

`--cpu-cap N` (or `KTSTR_CPU_CAP=N`) constrains a no-perf-mode VM or
kernel build to exactly `N` host CPUs, selected by walking whole LLCs
in NUMA-aware, consolidation-aware order (filtered to the calling
process's sched_getaffinity cpuset), and partial-taking the last LLC
so `plan.cpus.len() == N`. The full LLC is still flocked for
per-LLC coordination with concurrent ktstr peers. When the flag is
absent, the planner defaults to 30% of the allowed-CPU set (minimum
1). The plan writes the reserved CPUs and NUMA nodes into a cgroup
v2 cpuset sandbox so `make -jN` gcc fan-out and vCPU soft-mask
affinity respect the budget. On `shell`, mutually exclusive with
`performance_mode=true` (clap parse rejection); library consumers
see the env var silently ignored under perf-mode. Mutually exclusive
with `KTSTR_BYPASS_LLC_LOCKS=1` at every entry point (contract vs.
bypass conflict rejected at CLI parse plus the library and
kernel-build-pipeline sites). `ktstr locks` / `ktstr locks --json`
enumerates every held LLC + per-CPU + cache-entry flock on the host
with holder PID + cmdline for contention diagnosis. See
[Resource Budget](concepts/resource-budget.md).

</details>

<details>
<summary><b>Guest coverage</b> — profraw collection from inside the VM, merged with host coverage</summary>

Guest-side profraw collection via shared memory. The host merges
guest and host coverage for unified `cargo llvm-cov` reports.

</details>

<details>
<summary><b>cargo-ktstr</b> — cargo subcommand for the full workflow, more introspection less shell scripts</summary>

Wraps `cargo nextest run` with automatic kernel resolution.
Subcommands (in `--help` order): `test`, `coverage`, `llvm-cov`,
`stats`, `kernel`, `model`, `verifier`, `funify`, `completions`,
`show-host`, `show-thresholds`, `export`, `locks`, `shell`.
See [`cargo-ktstr`](running-tests/cargo-ktstr.md).

</details>

<details>
<summary><b>Remote kernel cache</b> — GHA cache backend for cross-run kernel sharing</summary>

GHA cache backend for CI kernel sharing. When `KTSTR_GHA_CACHE=1`
and `ACTIONS_CACHE_URL` are set, all cache lookups check the remote
after a local miss before falling back to download (for versions) or
erroring (for cache keys). All successful builds are pushed to the
remote automatically. Non-fatal on failure; local cache is
authoritative.

</details>

<details>
<summary><b>Real-kernel verifier analysis</b> — boots the kernel, loads the scheduler, reads actual verified instruction counts</summary>

Runs the BPF verifier against every `declare_scheduler!`-registered
scheduler's struct_ops programs inside a real kernel. Reports
per-program verified instruction counts with cycle collapse —
deduplicating repeated verifier paths to show true unique cost.
The sweep emits one nextest cell per (declared scheduler ×
kernel-list entry × accepted topology preset) tuple, with
parallelism and retries handled natively by nextest.

Reads `bpf_prog_aux.verified_insns` from guest memory after loading
the scheduler via struct_ops — the same path production uses.
Captures real-world verification costs including map sharing, BTF
resolution, and program composition.
See [Verifier](running-tests/verifier.md).

</details>

<details>
<summary><b>Host-guest signaling</b> — bidirectional host/VM coordination built into the test library</summary>

SHM signal slots enable the host and guest to coordinate without
a network stack, serial protocol, or guest-side daemon. Graceful
shutdown, readiness gates, and BPF map write triggers flow through
shared memory mapped into both address spaces.

</details>

<details>
<summary><b>High test coverage</b> — broad self-coverage of the framework itself <a href="https://codecov.io/gh/likewhatevs/ktstr"><img src="https://codecov.io/gh/likewhatevs/ktstr/graph/badge.svg?token=E7GRAO2KZM" alt="codecov"></a></summary>

</details>

---

See [Getting Started](getting-started.md) to set up your first test,
or browse the [Recipes](recipes.md) for common workflows.
