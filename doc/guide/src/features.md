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
6.12 and 7.0 on both x86_64 and aarch64. On 7.1+ kernels the watchdog
override uses `scx_sched.watchdog_timeout` via BTF detection; on
6.16-7.0 kernels it uses the static `scx_watchdog_timeout` symbol;
older kernels silently no-op.

## Testing

<details>
<summary><b>Real kernel, clean slate, x86/arm parity</b> — every VM test boots its own Linux kernel in KVM, fresh state each run</summary>

Tests boot a real Linux kernel in a KVM virtual machine with
configurable topology: NUMA nodes, LLCs, cores per LLC, threads per core.
Multi-NUMA topologies produce NUMA domains via ACPI SRAT/SLIT
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

`ktstr run` executes scenarios directly on the host — no VM, real
hardware, real scheduler. Reads the host's CPU topology from sysfs,
creates cgroups under `/sys/fs/cgroup/ktstr-{pid}`, and runs under
whatever scheduler is already active. Used for testing with
production schedulers and real topology.

</details>

<details>
<summary><b>Combinatoric scheduler testing</b> — typed flag DSL generates every configuration automatically</summary>

Tests load [sched_ext](https://github.com/sched-ext/scx) schedulers
via BPF struct_ops inside the VM. The
[`#[derive(Scheduler)]`](writing-tests/scheduler-definitions.md)
macro declares scheduler binaries with typed flag profiles and
dependency constraints. Each flag combination becomes a test variant
— the framework generates the power set of valid configurations so
you test combinations you'd never write by hand.

```rust
#[derive(Scheduler)]
#[scheduler(name = "mitosis", binary = "scx_mitosis",
            topology(1, 2, 4, 1),
            sched_args = ["--exit-dump-len", "1048576"])]
enum MitosisFlag {
    #[flag(args = ["--enable-llc-awareness"])]
    Llc,
    #[flag(args = ["--enable-work-stealing"], requires = [Llc])]
    Steal,
}
```

Without a scheduler attribute, tests run under the kernel's default
scheduler (EEVDF).

</details>

<details>
<summary><b>Data-driven scenarios</b> — declare what you want, the framework handles cgroups, cpusets, and workers</summary>

Scenarios are composable sequences of
[steps and ops](concepts/ops.md). You declare intent as data —
the framework creates cgroups, assigns cpusets, spawns workers,
sets scheduling policies, and manages affinity. 14
[canned scenarios](concepts/scenarios.md) cover common patterns;
scenario submodules provide reusable building blocks for custom
tests.

**API types:**
- `CgroupDef` — declarative cgroup: name + cpuset + workload(s)
- `Step` — sequence of ops followed by a hold period
- `Op` — atomic operation (add/remove cgroup, set/swap/clear cpuset, spawn, stop, set affinity, move tasks)
- `CpusetSpec` — topology-relative cpuset (LLC-aligned, disjoint, overlapping, range, exact)
- `HoldSpec` — hold duration (fractional, fixed, or looped)
- `AffinityKind` — per-worker affinity (inherit, random subset, LLC-aligned, cross-cgroup, single CPU, exact)
- `SchedPolicy` — Linux scheduling policy (Normal, Batch, Idle, FIFO, RoundRobin)
- `Work` — workload definition for a group of workers

</details>

<details>
<summary><b>Gauntlet</b> — one test declaration, hundreds of topology × flag variants with budget-aware CI selection</summary>

A single [`#[ktstr_test]`](writing-tests/ktstr-test-macro.md)
auto-expands across topology presets and flag profiles.
Budget-based selection (`KTSTR_BUDGET_SECS`) picks the subset that
maximizes coverage within a CI time limit.

**Constraint attributes:**
- `min_llcs`, `max_llcs`, `min_cpus`, `max_cpus`, `min_numa_nodes`, `max_numa_nodes`, `requires_smt` — topology gates
- `required_flags`, `excluded_flags` — flag profile filters
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
<summary><b>12 work types</b> — configurable workload profiles for different scheduling pressures</summary>

Workers are `fork()`ed processes placed in cgroups:

- `CpuSpin` — tight CPU spin loop
- `YieldHeavy` — repeated sched_yield with minimal CPU work
- `Mixed` — CPU spin burst followed by sched_yield
- `IoSync` — 64 KB write + 100 us sleep (simulated blocking I/O)
- `Bursty` — CPU burst then sleep (parameterized)
- `PipeIo` — CPU burst then pipe exchange (cross-CPU wake placement)
- `FutexPingPong` — paired futex wait/wake
- `FutexFanOut` — 1:N fan-out wake (schbench-style)
- `CachePressure` — strided RMW sized to pressure L1
- `CacheYield` — cache pressure + sched_yield
- `CachePipe` — cache pressure + pipe exchange
- `Sequence` — compound: loop through phases

See [Work Types](concepts/work-types.md).

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
<summary><b>Unified timeline</b> — correlate scenario phases with scheduler telemetry</summary>

Stimulus events (cgroup ops, cpuset changes, step transitions)
correlated with monitor samples for per-phase scheduler behavior
analysis. Each event carries timestamps, operation details, and
cumulative worker iteration counts.

</details>

<details>
<summary><b>Statistical regression detection</b> — Polars-powered analysis across combinatoric test matrices</summary>

[Polars](https://pola.rs)-powered aggregation computes scheduling
metrics across runs. Baseline save/compare with dual-gate
significance thresholds (absolute and relative) catches regressions
that single-run assertions miss.

**Metrics:**
- `spread` — CPU time fairness (0.0 = perfect)
- `gap_ms` — longest scheduling gap
- `migrations` / `migration_ratio` — cross-CPU migration volume
- `imbalance_ratio` — runqueue length imbalance
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
<summary><b>Guest coverage</b> — profraw collection from inside the VM, merged with host coverage</summary>

Guest-side profraw collection via shared memory. The host merges
guest and host coverage for unified `cargo llvm-cov` reports.

</details>

<details>
<summary><b>cargo-ktstr</b> — cargo subcommand for the full workflow, more introspection less shell scripts</summary>

Wraps `cargo nextest run` with automatic kernel resolution.
Subcommands (in `--help` order): `test`, `coverage`, `test-stats`,
`kernel`, `verifier`, `completions`, `shell`.
See [`cargo-ktstr`](running-tests/cargo-ktstr.md).

</details>

<details>
<summary><b>Remote kernel cache</b> — GHA cache backend for cross-run kernel sharing</summary>

GHA cache backend for CI kernel sharing. When `KTSTR_GHA_CACHE=1`
and `ACTIONS_CACHE_URL` are set, version and cache-key resolution
checks the remote cache after a local-cache miss before falling back
to download (for versions) or erroring (for cache keys). Explicit
`cargo ktstr kernel build` invocations push successful builds to the
remote. Non-fatal on failure; local cache is authoritative.

</details>

<details>
<summary><b>Real-kernel verifier analysis</b> — boots the kernel, loads the scheduler, reads actual verified instruction counts</summary>

Runs the BPF verifier against a scheduler's struct_ops programs
inside a real kernel. Reports per-callback verified instruction
counts with cycle collapse — deduplicating repeated verifier paths
to show true unique cost. Per-flag-profile breakdown reveals which
flag combinations increase verification complexity.

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
