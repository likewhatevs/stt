# Performance Mode

Performance mode reduces host-side noise during VM execution by pinning
vCPU threads to dedicated host cores and allocating guest memory with
hugepages.

## What it does

Two host-side optimizations are applied when `performance_mode` is
enabled:

**vCPU pinning** -- each virtual socket is mapped to a physical LLC
group on the host. vCPU threads are pinned to cores within their
assigned LLC via `sched_setaffinity`. This prevents the host scheduler
from migrating vCPU threads across LLCs, which would add cache
thrashing noise to measurements.

**Hugepages** -- guest memory is allocated with 2MB hugepages
(`MAP_HUGETLB`) when sufficient free hugepages exist. This eliminates
TLB pressure from host-side page walks during guest execution.

## Prerequisites

**Sufficient host CPUs** -- the host must have at least as many online
CPUs as the VM's total vCPU count (`sockets * cores * threads`). The
host must also have at least as many LLC groups as virtual sockets.
Oversubscription (more vCPUs than host CPUs) is a hard error.

**2MB hugepages** (optional) -- the host must have free 2MB hugepages
(check `/sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages`).
Without them, guest memory uses regular pages. A warning is printed.

## Validation

`validate_performance_mode()` runs during VM build and applies two
levels of checks:

**Errors (fatal):**
- Total vCPUs exceed available host CPUs.
- Virtual sockets exceed available LLC groups.
- Pinning plan cannot be satisfied (an LLC group has fewer available
  CPUs than the virtual socket requires).

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
fn my_perf_test(ctx: &Ctx) -> anyhow::Result<VerifyResult> {
    // vCPUs are pinned, hugepage-backed
    Ok(VerifyResult::pass())
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
parallel VMs can oversubscribe the host. Use it for targeted
single-scenario runs, not for gauntlet sweeps.
