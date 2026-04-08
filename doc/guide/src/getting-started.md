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
fn my_scheduler_test(ctx: &Ctx) -> Result<AssertResult> {
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

    // Assert: no worker was starved.
    let plan = AssertPlan::new().check_not_starved();
    Ok(plan.assert_cgroup(&reports, None))
}
```

Run with `cargo nextest run` (requires `/dev/kvm`).

## Build a kernel

`stt.kconfig` in the repo root contains a kernel config fragment
tuned for scheduler testing (sched_ext, BPF, kprobes, minimal boot).
To build a kernel from a Linux source tree:

```sh
cd /path/to/linux
make defconfig
cat /path/to/stt/stt.kconfig >> .config
make olddefconfig
make -j$(nproc)
```

## Next steps

To understand scenarios, flags, and verification:
[Core Concepts](concepts.md).

To write new tests: [Writing Tests](writing-tests.md).
