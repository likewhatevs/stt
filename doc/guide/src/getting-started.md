# Getting Started

## Prerequisites

- Linux host with KVM access (`/dev/kvm`)
- Rust toolchain (stable)

## Add the dependency

stt is not on crates.io. Add it as a git dependency:

```toml
[dev-dependencies]
stt = { git = "https://github.com/likewhatevs/stt" }
```

## Write a test

Write an `#[stt_test]` function. The
[`prelude`](https://likewhatevs.github.io/stt/api/stt/prelude/index.html)
module re-exports the types you need:

```rust
use stt::prelude::*;
use std::collections::BTreeSet;

#[stt_test(sockets = 1, cores = 2, threads = 1)]
fn my_scheduler_test(ctx: &Ctx) -> Result<VerifyResult> {
    // Create a cgroup and assign all CPUs.
    let mut group = CgroupGroup::new(ctx.cgroups);
    group.add_cgroup_no_cpuset("workers")?;
    let cpus: BTreeSet<usize> = ctx.topo.all_cpus().iter().copied().collect();
    ctx.cgroups.set_cpuset("workers", &cpus)?;

    // Spawn workers into the cgroup.
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

    // Let workers run, then collect results.
    std::thread::sleep(ctx.duration);
    let reports = handle.stop_and_collect();

    // Verify: no worker was starved.
    let plan = VerificationPlan::new().check_not_starved();
    Ok(plan.verify_cell(&reports, None))
}
```

Run with `cargo test` (requires `/dev/kvm`).

## Install the CLI

```sh
cargo install --path cargo-stt
```

## Run a scenario

```sh
cargo stt vm --sockets 2 --cores 4 --threads 2 \
  -- cgroup_steady --duration-s 30
```

`vm` boots a KVM virtual machine with the specified CPU topology.
Arguments after `--` are passed to `stt run` inside the VM.

To test with a scheduler, use `-p` to build and inject it:

```sh
cargo stt vm -p scx_mitosis --sockets 2 --cores 4 --threads 2 \
  -- cgroup_steady --duration-s 30
```

Expected output:

```text
[stt] booting VM: 2s4c2t (16 cpus), 4096 MB
[stt] running: cgroup_steady/default
[stt]   PASS  cgroup_steady/default (30.1s)
```

Omit the scenario name to run all scenarios:

```sh
cargo stt vm --sockets 2 --cores 4 --threads 2
```

## List scenarios

List catalog scenarios (data-driven):

```sh
stt list
```

List `#[stt_test]` integration tests:

```sh
cargo stt list
```

## View topology

```sh
cargo stt topo
```

Prints the host CPU topology (LLCs, NUMA nodes, CPU IDs).

## Next steps

To run existing tests with different flags, topologies, or schedulers:
[Running Tests](running-tests.md).

To understand scenarios, flags, and verification:
[Core Concepts](concepts.md).

To write new tests: [Writing Tests](writing-tests.md).
