# Getting Started

## Prerequisites

- Linux host with KVM access (`/dev/kvm`)
- Rust toolchain (stable, >= 1.88)
- clang and BPF toolchain (builds BPF skeletons via libbpf-cargo)
- libelf development headers
- pkg-config
- bpftool (generates vmlinux.h from the running kernel's BTF)

**Ubuntu/Debian:**

```sh
sudo apt install clang libelf-dev pkg-config bpftool
```

**Fedora:**

```sh
sudo dnf install clang elfutils-libelf-devel pkgconf bpftool
```

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

```rust,ignore
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

## CLI installation

The stt workspace has two binaries:

- **`stt`** -- the core binary (host-side VM management, guest-side
  test runner, kernel builds, topology display). Build it with:

  ```sh
  cargo build -p stt
  ```

  Or install it:

  ```sh
  cargo install --path .
  ```

- **`cargo-stt`** -- cargo plugin that wraps `stt` with test discovery,
  scheduler builds, and gauntlet orchestration. Install it with:

  ```sh
  cargo install --path cargo-stt
  ```

## Build a kernel

stt embeds a kernel config fragment tuned for scheduler testing
(sched_ext, BPF, kprobes, minimal boot). To build a kernel from a
Linux source tree:

```sh
stt kernel build ~/linux
```

This runs `make defconfig`, merges the stt config fragment, and
builds with `make -j$(nproc)`. To clean:

```sh
stt kernel clean ~/linux
```

Print the config fragment to stdout:

```sh
stt kernel kconfig
```

## Run a scenario

```sh
cargo stt vm --sockets 2 --cores 4 --threads 2 \
  -- cgroup_steady --duration-s 30
```

`vm` boots a KVM virtual machine with the specified CPU topology.
Arguments after `--` configure the test scenarios (names, flags,
duration).

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
