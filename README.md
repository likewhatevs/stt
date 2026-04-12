# ktstr

[![CI](https://github.com/likewhatevs/ktstr/actions/workflows/ci.yml/badge.svg)](https://github.com/likewhatevs/ktstr/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/likewhatevs/ktstr/graph/badge.svg)](https://codecov.io/gh/likewhatevs/ktstr)
[![guide](https://img.shields.io/badge/docs-guide-blue)](https://likewhatevs.github.io/ktstr/guide/)
[![api](https://img.shields.io/badge/docs-api-blue)](https://likewhatevs.github.io/ktstr/api/ktstr/)

> **Early stage.** APIs, CLI, and internals are actively evolving.
> Expect breaking changes between releases.

Test harness for Linux process schedulers, with a focus on
[sched_ext](https://github.com/sched-ext/scx). Boots kernels in KVM
VMs with synthetic CPU topologies, runs workloads, and verifies
scheduling correctness. Also tests under the kernel's default EEVDF
scheduler.

- **Real isolation** -- each test boots its own kernel. No host interference, no shared state.
- **Any topology** -- 1 to 252 CPUs with arbitrary LLC structure via synthetic ACPI tables.
- **Data-driven** -- scenarios declare cgroups, cpusets, workloads, and verification as data.
- **Gauntlet** -- all scenarios across 13 topology presets in parallel VMs. Baseline save/compare for A/B testing.
- **`#[ktstr_test]`** -- proc macro for integration tests that boot their own VMs.
- **Auto-repro** -- reruns failures with BPF kprobes on the crash call chain.

## Installation

```sh
cargo install ktstr --features cli    # host-side test runner
cargo install cargo-ktstr             # dev workflow plugin (kernel build + nextest)
cargo install cargo-nextest           # required test runner
```

`ktstr` is the host-side CLI for running scenarios directly against a
scheduler (outside VMs). `cargo-ktstr` automates kernel configuration,
building, and test execution in one command. `scx-ktstr` (the test
fixture scheduler) is built automatically by the workspace and does
not need a separate install.

## Setup

**Prerequisites:** Linux with `/dev/kvm`, Rust >= 1.88, clang, pkg-config.

```sh
# Ubuntu/Debian
sudo apt install clang pkg-config
```

**Add to your crate**:

```toml
[dev-dependencies]
ktstr = "0.1"
```

**Test files** go in `tests/` as standard Rust integration tests. Use `#[ktstr_test]` from `ktstr::prelude::*`.

See the [getting started guide](https://likewhatevs.github.io/ktstr/guide/getting-started.html) for Fedora packages, kernel discovery, and building a test kernel.

## Quick start

### Define a scheduler

Use `#[derive(Scheduler)]` to declare the scheduler binary, default
topology, and feature flags:

```rust
use ktstr::prelude::*;

#[derive(Scheduler)]
#[scheduler(name = "my_sched", binary = "scx_my_sched", topology(2, 4, 1))]
#[allow(dead_code)]
enum MySchedFlag {
    #[flag(args = ["--enable-feature-a"])]
    FeatureA,
    #[flag(args = ["--enable-feature-b"], requires = [FeatureA])]
    FeatureB,
}
```

This generates a `const MY_SCHED: Scheduler` and per-variant flag
constants. Tests referencing `MY_SCHED` inherit its topology and
flags. Without a scheduler, tests run under EEVDF.

### Write a test

Declare cgroups and workloads as data with `CgroupDef` and
`execute_defs`:

```rust
use ktstr::prelude::*;

#[ktstr_test(scheduler = MY_SCHED)]
fn basic_proportional(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        CgroupDef::named("cg_0").workers(2),
        CgroupDef::named("cg_1").workers(2),
    ])
}
```

For multi-step scenarios with dynamic topology changes, use
`execute_steps` with `Step` and `HoldSpec`:

```rust
use ktstr::prelude::*;

#[ktstr_test(scheduler = MY_SCHED, sockets = 1, cores = 4, threads = 1)]
fn cpuset_split(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step::with_defs(
        vec![
            CgroupDef::named("cg_0").with_cpuset(CpusetSpec::Disjoint { index: 0, of: 2 }),
            CgroupDef::named("cg_1").with_cpuset(CpusetSpec::Disjoint { index: 1, of: 2 }),
        ],
        HoldSpec::FULL,
    )];
    execute_steps(ctx, steps)
}
```

### Run

```sh
cargo nextest run
```

Requires `/dev/kvm`.

### Dev workflow

`cargo ktstr` handles kernel config, build, and test execution:

```sh
cargo ktstr test --kernel ~/linux                          # build kernel + run all tests
cargo ktstr test --kernel ~/linux -- -E 'test(my_test)'    # pass nextest filter
```

### Host-side CLI

`ktstr` runs scenarios directly on the host (outside VMs) against
a scheduler:

```sh
cargo run --features cli --bin ktstr -- list
cargo run --features cli --bin ktstr -- run --scheduler scx_my_sched
cargo run --features cli --bin ktstr -- topo
cargo run --features cli --bin ktstr -- cleanup
```

## Documentation

**[Guide](https://likewhatevs.github.io/ktstr/guide/)** -- getting started, concepts,
writing tests, recipes, architecture.

**[API docs](https://likewhatevs.github.io/ktstr/api/ktstr/)** -- rustdoc for all workspace crates.

## License

GPL-2.0-only
