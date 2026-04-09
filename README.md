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

Define cgroups declaratively and let the DSL handle lifecycle,
worker spawning, and assertion:

```rust
use stt::prelude::*;

#[stt_test(sockets = 1, cores = 2, threads = 1)]
fn my_scheduler_test(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step::with_defs(
        vec![
            CgroupDef::named("cg_0").workers(2),
            CgroupDef::named("cg_1").workers(2),
        ],
        HoldSpec::FULL,
    )];
    execute_steps(ctx, steps)
}
```

Run with `cargo nextest run` (requires `/dev/kvm`).

## Documentation

**[Guide](https://likewhatevs.github.io/stt/guide/)** -- getting started, concepts,
writing tests, recipes, architecture.

**[API docs](https://likewhatevs.github.io/stt/api/stt/)** -- rustdoc for all workspace crates.

## License

GPL-2.0-only
