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
$ cargo stt vm --sockets 2 --cores 4 --threads 2 \
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

## Crate structure

| Crate | Purpose |
|---|---|
| `stt` | Core library and CLI binary |
| `stt-macros` | `#[stt_test]` proc macro |
| `stt-sched` | Minimal BPF scheduler for testing |
| `cargo-stt` | Cargo plugin (`cargo stt vm`, `cargo stt test`, `cargo stt gauntlet`) |
