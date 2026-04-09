# Performance Mode

Performance mode reduces noise during VM execution by applying
host-side isolation (vCPU pinning, hugepages, NUMA mbind, RT
scheduling), a guest-visible CPUID hint (KVM_HINTS_REALTIME), and
KVM exit suppression (PAUSE and HLT VM exits disabled).

## What it does

Seven optimizations are applied when `performance_mode` is enabled
(six host-side, one guest-visible via CPUID). Host-side
`KVM_CAP_HALT_POLL` is explicitly skipped — the guest haltpoll
cpuidle driver disables it via `MSR_KVM_POLL_CONTROL` (see below):

**vCPU pinning** -- each virtual socket is mapped to a physical LLC
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

**Disable PAUSE VM exits** -- `KVM_CAP_X86_DISABLE_EXITS` with
`KVM_X86_DISABLE_EXITS_PAUSE` suppresses VM exits on PAUSE
instructions. Guest spinlocks execute PAUSE in tight loops; each
PAUSE normally causes a vmexit so the hypervisor can schedule other
vCPUs. With dedicated cores (vCPU pinning), this reschedule is
unnecessary overhead. The capability is optional --
if unsupported, a warning is logged and the VM proceeds without it.

**Disable HLT VM exits** -- `KVM_X86_DISABLE_EXITS_HLT` suppresses
VM exits on HLT instructions, the most frequent exit type during
boot and idle. BSP shutdown detection uses I8042 reset (port 0x64,
value 0xFE via `reboot=k`) and `VcpuExit::Shutdown` instead of
`VcpuExit::Hlt`. KVM blocks HLT disable when `mitigate_smt_rsb` is
active (host has `X86_BUG_SMT_RSB` and `cpu_smt_possible()`); in that case,
only PAUSE exits are disabled.

**KVM_HINTS_REALTIME CPUID** -- sets bit 0 of CPUID leaf 0x40000001
EDX, telling the guest kernel that vCPUs are pinned to dedicated host
cores. The guest disables PV spinlocks, PV TLB flush, and PV
sched_yield (all add hypercall overhead unnecessary on dedicated
cores), and enables haltpoll cpuidle (polls briefly before halting,
reducing wakeup latency). PV spinlocks require
CONFIG_PARAVIRT_SPINLOCKS, which is not in stt.kconfig, so that
disable is a no-op for stt guests.

**Skip host-side halt poll** -- when a guest vCPU halts (executes
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
`(sockets * cores * threads) + 1` online CPUs. The extra CPU is
reserved for service threads (monitor, watchdog) so they do not share
a core with any RT vCPU. The host must also have at least as many LLC
groups as virtual sockets.

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
- Virtual sockets exceed available LLC groups.
- Pinning plan cannot be satisfied (an LLC group has fewer available
  CPUs than the virtual socket requires).
- No free host CPU for service threads after vCPU assignment.

**Warnings (non-fatal):**
- Insufficient free hugepages -- regular page allocation is used.
- Host load is high -- `procs_running` from `/proc/stat` exceeds
  half the vCPU count, results may be noisy.

## Usage

In `#[stt_test]`:

```rust,ignore
#[stt_test(
    sockets = 2,
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
let vm = vmm::SttVm::builder()
    .kernel(&kernel_path)
    .init_binary(&stt_binary)
    .topology(2, 4, 2)
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

**Noise reduction** -- pinning, hugepages, NUMA mbind, RT scheduling,
PAUSE and HLT VM exit disabling, the KVM_HINTS_REALTIME CPUID hint,
and skipping host-side halt poll (guest haltpoll disables it via
MSR_KVM_POLL_CONTROL) reduce measurement variance. Scheduling gaps, spread, and throughput checks
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
The `perf-vm` test group in `.config/nextest.toml` sets a static
`max-threads` limit. The flock-based LLC slot reservation
(`acquire_slot_with_locks`) handles runtime contention: if all LLC
slots are busy, the test is skipped with a `ResourceContention`
error rather than silently degrading to a shared LLC.

## super_perf_mode

`super_perf_mode` implies `performance_mode` and additionally
validates LLC exclusivity at build time. Each virtual socket must
reserve the entire physical LLC group it maps to. The validation
sums the actual CPU count of each LLC group and checks the total
(plus service CPU) fits within the host's online CPUs.

`super_perf_mode` validates at build time that sufficient LLC groups
exist for the test topology. Both modes apply the same runtime
optimizations (pinning, hugepages, NUMA mbind, RT scheduling,
PAUSE and HLT VM exit disabling, KVM_HINTS_REALTIME CPUID hint,
no host-side halt poll).

```rust,ignore
#[stt_test(
    sockets = 2,
    cores = 4,
    threads = 2,
    super_perf_mode = true,
)]
fn my_exclusive_perf_test(ctx: &Ctx) -> Result<AssertResult> {
    Ok(AssertResult::pass())
}
```
