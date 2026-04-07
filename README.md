# stt

[![CI](https://github.com/likewhatevs/stt/actions/workflows/ci.yml/badge.svg)](https://github.com/likewhatevs/stt/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/likewhatevs/stt/graph/badge.svg)](https://codecov.io/gh/likewhatevs/stt)
[![guide](https://img.shields.io/badge/docs-guide-blue)](https://likewhatevs.github.io/stt/guide/)
[![api](https://img.shields.io/badge/docs-api-blue)](https://likewhatevs.github.io/stt/api/stt/)

Test harness for Linux process schedulers, with a focus on
[sched_ext](https://github.com/sched-ext/scx). Boots kernels in KVM
VMs with synthetic CPU topologies, runs workloads, and verifies
scheduling correctness. Also tests under the kernel's default EEVDF
scheduler.

- **Real isolation** -- each test boots its own kernel. No host interference, no shared state.
- **Any topology** -- 1 to 252 CPUs with arbitrary LLC structure via synthetic ACPI tables.
- **Data-driven** -- scenarios declare cgroups, cpusets, workloads, and verification as data.
- **Gauntlet** -- all scenarios across 13 topology presets in parallel VMs. Baseline save/compare for A/B testing.
- **`#[stt_test]`** -- proc macro for integration tests that boot their own VMs.
- **Auto-repro** -- reruns failures with BPF kprobes on the crash call chain.

## Quick start

### As a library

Write a test that boots a VM, creates a cgroup, runs a workload, and
checks the result:

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

### From the CLI

```sh
cargo install --path cargo-stt

# single scenario
cargo stt vm --sockets 2 --cores 4 --threads 2 -- cgroup_steady

# with a scheduler
cargo stt vm -p scx_mitosis --sockets 2 --cores 4 --threads 2 -- cgroup_steady

# gauntlet (catalog scenarios)
cargo stt vm --gauntlet --parallel 4

# gauntlet (#[stt_test] integration tests)
cargo stt gauntlet --parallel 4
```

## Documentation

**[Guide](https://likewhatevs.github.io/stt/guide/)** -- getting started, concepts,
writing tests, recipes, architecture.

**[API docs](https://likewhatevs.github.io/stt/api/stt/)** -- rustdoc for all workspace crates.

## License

GPL-2.0-only
