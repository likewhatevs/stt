# ktstr

ktstr is a test harness for Linux process schedulers, with a focus on
sched_ext. It boots Linux kernels in KVM virtual machines with
controlled CPU topologies, runs workloads, and verifies scheduling
correctness. Also tests under the kernel's default EEVDF scheduler.

## Quick taste

The simplest test calls a canned scenario:

```rust,ignore
use ktstr::prelude::*;

#[ktstr_test(sockets = 1, cores = 2, threads = 1)]
fn my_test(ctx: &Ctx) -> Result<AssertResult> {
    scenarios::steady(ctx)
}
```

```sh
cargo nextest run --workspace
```

Without a `scheduler` attribute, tests run under EEVDF. See
[Getting Started](getting-started.md) for testing a sched_ext scheduler.

## Library API

The `ktstr::prelude` module re-exports the types needed for writing
tests. Declare cgroups and workloads as data with `CgroupDef`:

```rust,ignore
use ktstr::prelude::*;

#[ktstr_test(sockets = 1, cores = 2, threads = 1)]
fn my_test(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        CgroupDef::named("cg_0").workers(2),
        CgroupDef::named("cg_1").workers(2),
    ])
}
```

The prelude also exports low-level types (`CgroupGroup`,
`WorkloadConfig`, `WorkloadHandle`) for manual cgroup and worker
management, `Assert` for composable assertion config, and
`WorkerReport` for telemetry access.

## What it tests

- **Fair scheduling** -- workers get CPU time without starvation or
  excessive scheduling gaps.
- **Cpuset isolation** -- workers stay on assigned CPUs.
- **Dynamic operations** -- cgroups created, destroyed, and resized
  mid-run.
- **Affinity** -- scheduler respects thread affinity constraints.
- **Stress** -- many cgroups, many workers, rapid topology changes.
- **Stall detection** -- scheduler doesn't drop tasks.

## Design

Two principles drive ktstr's architecture:

**Fidelity without overhead** -- every test boots a real Linux kernel
in a KVM VM with real cgroups and real BPF programs. No mocking, no
containers, no shared state. The VMM is minimal: no disks, no network
devices, no PCI. The device model is two serial ports (COM1 for kernel
console, COM2 for application I/O) and a shared-memory ring buffer.

**Direct access over tooling layers** -- the host-side monitor reads
guest memory directly via BTF-resolved struct offsets to observe
scheduler state. No BPF instrumentation inside the guest, no observer
effects on scheduling decisions. See [Monitor](architecture/monitor.md)
for details on BTF resolution and guest memory introspection.

## BPF verifier analysis

The `verifier_pipeline` tests boot a scheduler in a VM and capture
per-program verifier output from the real kernel verifier. The
default output applies **cycle collapse** to reduce repetitive loop
unrolling. See [BPF Verifier](running-tests/verifier.md) for details.

## Auto-repro probe pipeline

When a scheduler crashes, ktstr can automatically rerun the failing
scenario with BPF probes attached to the crash-path functions. See
[Auto-Repro](running-tests/auto-repro.md) for details.

## Workspace structure

| Component | Purpose |
|---|---|
| `ktstr` (lib) | Core library |
| `ktstr-macros` | `#[ktstr_test]` and `#[derive(Scheduler)]` proc macros |
| `ktstr` (bin) | Host-side CLI (behind the `cli` feature) |
| `cargo-ktstr` | Dev workflow plugin: kernel build + nextest |
| `scx-ktstr` | Minimal BPF scheduler for testing |

## Kernel config

`ktstr.kconfig` in the repo root contains the kernel config fragment
needed for scheduler testing (sched_ext, BPF, kprobes, cgroups).
Copy it to your kernel source tree and run `make olddefconfig`.
