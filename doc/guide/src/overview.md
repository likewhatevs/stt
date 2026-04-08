# stt

stt is a test harness for Linux process schedulers, with a focus on
sched_ext. It boots Linux kernels in KVM virtual machines with
controlled CPU topologies, runs workloads, and verifies scheduling
correctness. Also tests under the kernel's default EEVDF scheduler.

## Design

Two principles drive stt's architecture:

**Fidelity without overhead** -- every test boots a real Linux kernel
in a KVM VM with real cgroups and real BPF programs. No mocking, no
containers, no shared state. The VMM is minimal: no disks, no network
devices, no PCI. The device model is two serial ports (COM1 for kernel
console, COM2 for application I/O) and a shared-memory ring buffer.

**Direct access over tooling layers** -- the host-side monitor reads
guest memory directly via BTF-resolved struct offsets (`rq.nr_running`,
`rq.clock`) to observe scheduler state. No BPF instrumentation inside
the guest, no observer effects on scheduling decisions. Verifier
statistics come from the real kernel verifier inside the VM -- the
scheduler binary loads each program with `BPF_LOG_STATS` and emits
structured output; the host has no BPF dependency. Cycle collapse
reduces repetitive loop unrolling instead of truncating.

## Quick taste

```sh
cargo nextest run --workspace
```

## What it tests

- **Fair scheduling** -- workers get CPU time without starvation or
  excessive scheduling gaps.
- **Cpuset isolation** -- workers stay on assigned CPUs.
- **Dynamic operations** -- cgroups created, destroyed, and resized
  mid-run.
- **Affinity** -- scheduler respects thread affinity constraints.
- **Stress** -- many cgroups, many workers, rapid topology changes.
- **Stall detection** -- scheduler doesn't drop tasks.

## BPF verifier analysis

The `verifier_pipeline` test boots a scheduler in a VM and captures
per-program verifier output from the real kernel verifier. The
scheduler binary loads each BPF program inside the VM with
`BPF_LOG_STATS` enabled -- there is no host-side BPF loading.
Per-program statistics include instruction counts, states explored,
verification time, and stack depth. The default output applies
**cycle collapse** to reduce repetitive loop unrolling. See
[BPF Verifier](running-tests/verifier.md) for details.

## Auto-repro probe pipeline

When a scheduler crashes, stt can automatically rerun the failing
scenario with BPF probes attached to the crash-path functions. The
probe pipeline captures function arguments, BTF-resolved struct fields,
and source locations for both kernel functions (kprobe) and BPF
struct_ops callbacks (fentry). See [Auto-Repro](running-tests/auto-repro.md)
for details.

## Library API

The `stt::prelude` module re-exports the types needed for writing
tests. A minimal `#[stt_test]` function:

```rust,ignore
use stt::prelude::*;
use std::collections::BTreeSet;

#[stt_test(sockets = 1, cores = 2, threads = 1)]
fn my_test(ctx: &Ctx) -> Result<AssertResult> {
    let mut group = CgroupGroup::new(ctx.cgroups);
    group.add_cgroup_no_cpuset("workers")?;
    let cpus: BTreeSet<usize> = ctx.topo.all_cpus().iter().copied().collect();
    ctx.cgroups.set_cpuset("workers", &cpus)?;

    let cfg = WorkloadConfig {
        num_workers: 2,
        work_type: WorkType::CpuSpin,
        ..Default::default()
    };
    let mut handle = WorkloadHandle::spawn(&cfg)?;
    for tid in handle.tids() {
        ctx.cgroups.move_task("workers", tid)?;
    }
    handle.start();

    std::thread::sleep(ctx.duration);
    let reports = handle.stop_and_collect();

    let plan = AssertPlan::new().check_not_starved();
    Ok(plan.assert_cgroup(&reports, None))
}
```

The prelude also exports ops types (`CgroupDef`, `CpusetSpec`,
`HoldSpec`, `Step`, `execute_steps`), `Assert` for composable
assertions config, and `WorkerReport` for telemetry access.

## Crate structure

| Crate | Purpose |
|---|---|
| `stt` | Core library |
| `stt-macros` | `#[stt_test]` proc macro |
| `stt-sched` | Minimal BPF scheduler for testing |

## Kernel config

`stt.kconfig` in the repo root contains the kernel config fragment
needed for scheduler testing (sched_ext, BPF, kprobes, cgroups).
Copy it to your kernel source tree and run `make olddefconfig`.
