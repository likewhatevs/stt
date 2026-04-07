# stt

stt is a test harness for Linux process schedulers, with a focus on
sched_ext. It boots Linux kernels in KVM virtual machines with
controlled CPU topologies, runs workloads, and verifies scheduling
correctness. Also tests under the kernel's default EEVDF scheduler.

## Design

Testing schedulers requires three things that are hard to get on real
hardware: controlled CPU topologies, deterministic workloads, and
observation without interference.

stt addresses all three. It boots Linux in a KVM VM with synthetic
ACPI tables that present any topology (1 to 252 CPUs, arbitrary LLC
structure). It spawns `fork()`ed worker processes with controlled
workload patterns. It observes scheduler state by reading guest memory
directly from the host -- no BPF instrumentation, no observer effects
on scheduling decisions.

The same `stt` binary runs on both sides. On the host it manages the
VM, runs the monitor, and evaluates results. Inside the VM it creates
cgroups, forks workers, and executes test scenarios.

## Quick taste

```sh
cargo stt vm --sockets 2 --cores 4 --threads 2 \
    -- cgroup_steady --duration-s 10
[stt] booting VM: 2s4c2t (16 cpus), 4096 MB
[stt] running: cgroup_steady/default
[stt]   PASS  cgroup_steady/default (10.2s)
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
fn my_test(ctx: &Ctx) -> Result<VerifyResult> {
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

    let plan = VerificationPlan::new().check_not_starved();
    Ok(plan.verify_cgroup(&reports, None))
}
```

The prelude also exports ops types (`CgroupDef`, `CpusetSpec`,
`Step`, `execute_steps`), `Verify` for composable verification
config, and `WorkerReport` for telemetry access.

## Two binaries

stt ships two binaries:

- **`stt`** -- core binary. Hosts the VM manager, guest-side test
  runner, kernel build tools (`stt kernel build`), topology display
  (`stt topo`), and crash probe (`stt probe`).
- **`cargo-stt`** -- cargo plugin. Adds test discovery, scheduler
  builds, and gauntlet orchestration on top of `stt`. Provides
  `cargo stt vm`, `cargo stt test`, `cargo stt gauntlet`,
  `cargo stt list`, `cargo stt topo`, and `cargo stt probe`.

See [CLI Reference](cli-reference.md) for the complete subcommand
listing.

## Crate structure

| Crate | Purpose |
|---|---|
| `stt` | Core library and CLI binary |
| `stt-macros` | `#[stt_test]` proc macro |
| `stt-sched` | Minimal BPF scheduler for testing |
| `cargo-stt` | Cargo plugin (`cargo stt vm`, `cargo stt test`, `cargo stt gauntlet`, `cargo stt list`, `cargo stt topo`, `cargo stt probe`) |
