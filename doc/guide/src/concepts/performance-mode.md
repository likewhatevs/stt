# Performance Mode

Performance mode reduces noise during VM execution by applying
host-side isolation (vCPU pinning, hugepages, NUMA mbind, RT
scheduling). On x86_64, additionally: a guest-visible CPUID hint
(KVM_HINTS_REALTIME) and KVM exit suppression (PAUSE and HLT VM
exits disabled). On aarch64, the four host-side optimizations apply
(vCPU pinning, hugepages, NUMA mbind, RT scheduling); KVM exit
suppression and CPUID hints are not available.

## What it does

On x86_64, seven optimizations are applied when `performance_mode`
is enabled (six host-side, one guest-visible via CPUID). On aarch64,
four of these apply (vCPU pinning, hugepages, NUMA mbind, RT
scheduling); the x86-specific items (PAUSE/HLT exit disabling,
KVM_HINTS_REALTIME CPUID, halt poll) are not available.

Host-side `KVM_CAP_HALT_POLL` is explicitly skipped on x86_64 —
the guest haltpoll cpuidle driver disables it via
`MSR_KVM_POLL_CONTROL` (see below):

**vCPU pinning** -- each virtual LLC is mapped to a physical LLC
group on the host. vCPU threads are pinned to cores within their
assigned LLC via `sched_setaffinity`. This prevents the host scheduler
from migrating vCPU threads across LLCs, which would add cache
thrashing noise to measurements.

**Hugepages** -- guest memory is allocated with 2MB hugepages
(`MAP_HUGETLB`) when sufficient free hugepages exist. This eliminates
TLB pressure from host-side page walks during guest execution.

**NUMA mbind** -- guest memory is bound to the NUMA node(s) of the
pinned vCPUs via `mbind(MPOL_BIND)`. This ensures memory allocations
are local to the CPUs executing vCPU threads, avoiding cross-node
memory access latency.

**RT scheduling** -- vCPU threads are set to `SCHED_FIFO` priority 1.
The watchdog and monitor threads run at priority 2 on a dedicated
host CPU not assigned to any vCPU, so they can preempt for
timeout/sampling without competing for vCPU cores. The serial console
mutex uses `PTHREAD_PRIO_INHERIT` to avoid priority inversion between
RT vCPU threads and service threads.

**Disable PAUSE VM exits** (x86_64 only) -- `KVM_CAP_X86_DISABLE_EXITS` with
`KVM_X86_DISABLE_EXITS_PAUSE` suppresses VM exits on PAUSE
instructions. Guest spinlocks execute PAUSE in tight loops; each
PAUSE normally causes a vmexit so the hypervisor can schedule other
vCPUs. With dedicated cores (vCPU pinning), this reschedule is
unnecessary overhead. The capability is optional --
if unsupported, a warning is logged and the VM proceeds without it.

**Disable HLT VM exits** (x86_64 only) -- `KVM_X86_DISABLE_EXITS_HLT` suppresses
VM exits on HLT instructions, the most frequent exit type during
boot and idle. BSP shutdown detection uses I8042 reset (port 0x64,
value 0xFE via `reboot=k`) and `VcpuExit::Shutdown` instead of
`VcpuExit::Hlt`. KVM blocks HLT disable when `mitigate_smt_rsb` is
active (host has `X86_BUG_SMT_RSB` and `cpu_smt_possible()`); in that case,
only PAUSE exits are disabled.

**KVM_HINTS_REALTIME CPUID** (x86_64 only) -- sets bit 0 of CPUID leaf 0x40000001
EDX, telling the guest kernel that vCPUs are pinned to dedicated host
cores. The guest disables PV spinlocks, PV TLB flush, and PV
sched_yield (all add hypercall overhead unnecessary on dedicated
cores), and enables haltpoll cpuidle (polls briefly before halting,
reducing wakeup latency). PV spinlocks require
CONFIG_PARAVIRT_SPINLOCKS, which is not in ktstr.kconfig, so that
disable is a no-op for ktstr guests.

**Skip host-side halt poll** (x86_64 only) -- when a guest vCPU halts (executes
HLT with nothing to do), KVM can busy-wait briefly on the host
before putting the vCPU thread to sleep, reducing wakeup latency at
the cost of host CPU time. `KVM_CAP_HALT_POLL` controls this
per-VM ceiling. In performance mode it is not set because the guest
haltpoll cpuidle driver (enabled by KVM_HINTS_REALTIME above)
handles polling inside the guest and writes `MSR_KVM_POLL_CONTROL=0`
to disable host-side polling via `kvm_arch_no_poll()`.
Non-performance-mode VMs set `KVM_CAP_HALT_POLL` to 200µs (matching
the x86 kernel default), or 0 when vCPUs exceed host CPUs.

## Prerequisites

**Sufficient host CPUs** -- the host must have at least
`(llcs * cores_per_llc * threads_per_core) + 1` online CPUs. The extra
CPU is reserved for service threads (monitor, watchdog) so they do not
share a core with any RT vCPU. The host must also have at least as many
LLC groups as virtual LLCs.

**2MB hugepages** (optional) -- the host must have free 2MB hugepages
(check `/sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages`).
Without them, guest memory uses regular pages. A warning is printed.

**CAP_SYS_NICE or rtprio limit** (optional) -- `SCHED_FIFO` requires
either `CAP_SYS_NICE` (root) or an `RLIMIT_RTPRIO` >= the requested
priority. Set an rtprio limit for non-root use:

```
# /etc/security/limits.conf
username  -  rtprio  99
```

Log out and back in for the limit to take effect. Without either
capability, RT scheduling is skipped with a warning and vCPU threads
run at normal priority (results may be noisy).

## Validation

`validate_performance_mode()` runs during VM build and applies two
levels of checks:

**Errors (fatal):**
- Total vCPUs + 1 service CPU exceed available host CPUs.
- Virtual LLCs exceed available LLC groups.
- Pinning plan cannot be satisfied (an LLC group has fewer available
  CPUs than the virtual LLC requires).
- No free host CPU for service threads after vCPU assignment.

**Warnings (non-fatal):**
- Insufficient free hugepages -- regular page allocation is used.
- Host load is high -- `procs_running` from `/proc/stat` exceeds
  half the vCPU count, results may be noisy.
- TSC not stable (x86_64 only, checked at VM creation time) --
  `KVM_CLOCK_TSC_STABLE` not set after `KVM_GET_CLOCK`, kvmclock
  falls back to per-vCPU timekeeping. Timing measurements may have
  higher variance. Common in nested virtualization.

## Usage

In `#[ktstr_test]`:

```rust,ignore
#[ktstr_test(
    llcs = 2,
    cores = 4,
    threads = 2,
    performance_mode = true,
)]
fn my_perf_test(ctx: &Ctx) -> Result<AssertResult> {
    // vCPUs are pinned, hugepage-backed
    Ok(AssertResult::pass())
}
```

Via the builder API:

```rust,ignore
let vm = vmm::KtstrVm::builder()
    .kernel(&kernel_path)
    .init_binary(&ktstr_binary)
    .topology(1, 2, 4, 2)
    .memory_mb(4096)
    .performance_mode(true)
    .build()?
    .run()?;
```

## When to use

Performance mode is for tests where host-side scheduling noise affects
results -- fairness spread measurements, scheduling gap detection,
imbalance ratio checks. It is not needed for correctness tests (cpuset
isolation, starvation detection) where pass/fail is binary.

The gauntlet runs many VMs in parallel. Performance mode on
parallel VMs can oversubscribe the host if scheduled naively.
Avoid `performance_mode` unless the host has enough CPUs for the
topology matrix.

## Two dimensions

Performance mode serves two purposes:

**Noise reduction** -- pinning, hugepages, NUMA mbind, and RT
scheduling reduce measurement variance on both architectures. On
x86_64, PAUSE and HLT VM exit disabling, the KVM_HINTS_REALTIME
CPUID hint, and skipping host-side halt poll further reduce noise. Scheduling gaps, spread, and throughput checks
become meaningful because host jitter is controlled. Without
performance mode, a 50ms gap could be host noise; with it, the same
gap indicates a scheduler problem.

**Performance assertions** -- with stable measurements, tests can set
tight thresholds (`max_gap_ms`, `min_iteration_rate`,
`max_p99_wake_latency_ns`) to detect scheduling regressions. A test
using `execute_steps_with` can pass custom `Assert` checks that are
evaluated inside the guest against worker telemetry. These thresholds
are only meaningful under performance mode's controlled environment.

## Nextest parallelism

Performance-mode tests each consume one LLC group on the host.
The `vm-perf` test group in `.config/nextest.toml` sets a static
`max-threads` limit. The flock-based LLC slot reservation
(`acquire_resource_locks`) handles runtime contention: if all LLC
slots are busy, the test returns `ResourceContention`.

On contention, the test returns exit code 0 (skip) -- it never ran.
The `SKIP:` prefix in stderr distinguishes skips from real passes.

## LLC exclusivity validation

When `performance_mode` is enabled, the build step validates LLC
exclusivity: each virtual LLC must reserve the entire physical
LLC group it maps to. The validation sums the actual CPU count of
each LLC group and checks the total (plus service CPU) fits within
the host's online CPUs. If validation fails, the build returns an
error (tests skip with `ResourceContention`).

## Four-way mode tier

ktstr's host-side resource coordination has four effective tiers,
selected by the combination of `performance_mode`,
`--no-perf-mode`/`KTSTR_NO_PERF_MODE`, and `--llc-cap`/`KTSTR_LLC_CAP`:

### Tier 1: performance mode (full isolation)

Enabled when `performance_mode=true` is set on the VM builder (or
via `#[ktstr_test(performance_mode = true)]`). Acquires `LOCK_EX`
on each selected LLC's `/tmp/ktstr-llc-{N}.lock` — the LLC-level
exclusive lock already covers every CPU in the group, so per-CPU
`/tmp/ktstr-cpu-{C}.lock` files are NOT touched
(`try_acquire_all` in `vmm/host_topology.rs` short-circuits the
per-CPU loop when `LlcLockMode == Exclusive`). Applies every
isolation feature listed under "What it does": vCPU pinning via
`sched_setaffinity`, 2 MB hugepages, NUMA mbind, RT `SCHED_FIFO`
scheduling, and (x86_64) PAUSE/HLT exit suppression +
KVM_HINTS_REALTIME CPUID.

### Tier 2: no-perf-mode (unreserved)

Enabled by `--no-perf-mode` / `KTSTR_NO_PERF_MODE=1`. Acquires a
`LOCK_SH` reservation across every host LLC (the pre-flag default for
this mode) and does no isolation work. Multiple no-perf-mode VMs
coexist on the same LLCs because shared locks are reentrant, but a
concurrent perf-mode VM attempting `LOCK_EX` on any of those LLCs
blocks until every no-perf-mode peer has released. No pinning, no
hugepages, no mbind, no RT promotion, no KVM exit suppression.

### Tier 3: no-perf-mode with `--llc-cap N` (resource budget)

Enabled by passing `--llc-cap N` (or `KTSTR_LLC_CAP=N`) alongside
`--no-perf-mode`. The reservation is still `LOCK_SH`, but now
constrained to a NUMA-aware, consolidation-aware subset of `N` LLCs
(selected by [`acquire_llc_plan`]). Additional enforcement:

- **cgroup v2 cpuset sandbox** -- the reserved CPUs and derived NUMA
  nodes are written to a child cgroup's `cpuset.cpus` and
  `cpuset.mems`, and the build pid is migrated into that cgroup, so
  `make -jN` gcc children inherit the binding. Under `--llc-cap`,
  narrowing by a parent cgroup is a fatal error; without the flag the
  earlier tiers warn and proceed.
- **Soft-mask affinity** -- vCPU threads receive a
  `sched_setaffinity` mask covering only the reserved CPUs, so the
  guest's CPU placement respects the cap even though no pinning is
  applied.
- **No RT scheduling, no hugepages, no mbind, no KVM exit
  suppression** -- these remain off; `--llc-cap` is not a partial
  performance mode.
- **`make -jN` hint** -- kernel-build pipelines pass `plan.cpus.len()`
  to `make` so gcc's fan-out matches the reserved capacity rather
  than `nproc`.

This tier is mutually exclusive with `performance_mode=true` (on
the CLI, clap `requires = "no_perf_mode"` rejects `--llc-cap`
without `--no-perf-mode` at parse time) and with
`KTSTR_BYPASS_LLC_LOCKS=1` (rejected at every entry point because
the contract and the bypass escape hatch are contradictory).
Library consumers that set `performance_mode=true` on
`KtstrVmBuilder` directly bypass the CLI parse — `KTSTR_LLC_CAP`
is silently ignored in that path because the builder's perf-mode
branch never consults `LlcCap::resolve`.

See [Resource Budget](resource-budget.md) for the `LlcCap`,
`LlcPlan`, and `ktstr locks` surfaces in detail.

### Tier 4: default (per-CPU window + LLC LOCK_SH)

Selected when neither `performance_mode=true` nor
`--no-perf-mode`/`KTSTR_NO_PERF_MODE` is set — the default path
for `#[ktstr_test]` entries that don't declare `performance_mode`
(entry.rs `KtstrTestEntry::DEFAULT` sets `performance_mode:
false`). `acquire_cpu_locks` (in `vmm/host_topology.rs`) walks a
contiguous CPU window, takes `LOCK_EX` on each window CPU's
`/tmp/ktstr-cpu-{C}.lock`, then additionally takes `LOCK_SH` on
the LLC lockfiles covering those CPUs so a perf-mode (tier 1)
VM cannot grab `LOCK_EX` on an LLC that this path is using. No
pinning, no isolation, no cgroup sandbox — the per-CPU reservation
is purely for host-scheduling-noise avoidance between concurrent
VMs.

This is the ONLY tier that actually flocks per-CPU lockfiles.
Tier 1 skips them (LLC EX already covers all CPUs in the group),
tier 2 skips them (all-LLC SH does not need CPU-granularity
reservation), and tier 3 skips them (capped LLC SH is enforced
via the cgroup cpuset, not per-CPU flocks).

## Disabling performance mode

`--no-perf-mode` (or `KTSTR_NO_PERF_MODE=1`) forces
`performance_mode=false`. The result is **tier 2 or tier 3 above**,
depending on whether `--llc-cap` is also set. The feature differences
relative to tier 1 are:

- **LLC flock mode** -- tier 1 holds `LOCK_EX` on each reserved LLC;
  tiers 2 and 3 hold `LOCK_SH`. Multiple shared holders coexist; an
  exclusive holder blocks every shared acquirer and vice-versa.
- **Per-CPU flocks** -- tier 1 relies on LLC-level `LOCK_EX` for
  exclusivity; per-CPU `/tmp/ktstr-cpu-{C}.lock` files are skipped
  (`try_acquire_all` in `vmm/host_topology.rs` short-circuits the
  per-CPU loop when `LlcLockMode == Exclusive` because the LLC
  lock already covers every CPU in the group). Tiers 2 and 3 do
  not touch per-CPU locks either. The default-mode path (tier 4
  below) is the only one that flocks `/tmp/ktstr-cpu-{C}.lock`.
- **vCPU pinning** -- tier 1 pins via `sched_setaffinity` to the
  reserved LLC's CPUs. Tier 3 applies soft-mask affinity (cap-scoped
  but no 1:1 vCPU-to-CPU binding). Tier 2 applies no affinity.
- **RT scheduling** -- tier 1 only; tiers 2 and 3 run vCPU threads at
  normal priority.
- **Hugepages** -- tier 1 only; tiers 2 and 3 use regular pages.
- **NUMA mbind** -- tier 1 only; tier 3 instead writes `cpuset.mems`
  on its child cgroup to achieve NUMA locality at the cgroup layer.
- **KVM exit suppression** (x86_64) -- tier 1 only; tiers 2 and 3
  leave PAUSE and HLT exits enabled.
- **KVM_HINTS_REALTIME CPUID** (x86_64) -- tier 1 only; tiers 2 and
  3 leave the guest on PV spinlocks and standard cpuidle.

Use tier 2 on shared CI runners or unprivileged containers where any
LLC reservation is unacceptable. Use tier 3 on multi-tenant hosts
where you want bounded concurrency (at most `N` concurrent builds per
host) but cannot afford the full perf-mode contract. Use tier 1 for
regression measurement where host jitter must be controlled.

Available via:

- `ktstr run --no-perf-mode`
- `ktstr shell --no-perf-mode`
- `cargo ktstr test --no-perf-mode`
- `cargo ktstr coverage --no-perf-mode`
- `cargo ktstr llvm-cov --no-perf-mode`
- `cargo ktstr shell --no-perf-mode`
- `KTSTR_NO_PERF_MODE=1` (any value; presence is sufficient)

`--llc-cap N` layers on top of any of the above. The env var is read
by every VM builder call site (test harness, auto-repro, verifier,
shell). The CLI flags set the env var before test execution so
library consumers inherit it.
