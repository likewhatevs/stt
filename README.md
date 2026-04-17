# ktstr

[![CI](https://github.com/likewhatevs/ktstr/actions/workflows/ci.yml/badge.svg)](https://github.com/likewhatevs/ktstr/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/likewhatevs/ktstr/graph/badge.svg?token=E7GRAO2KZM)](https://codecov.io/gh/likewhatevs/ktstr)
[![guide](https://img.shields.io/badge/docs-guide-blue)](https://likewhatevs.github.io/ktstr/guide/)
[![api](https://img.shields.io/badge/docs-api-blue)](https://likewhatevs.github.io/ktstr/api/ktstr/)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](https://github.com/likewhatevs/ktstr/issues)

> **Early stage.** APIs, CLI, and internals are actively evolving.
> Expect breaking changes between releases.

Test harness for Linux process schedulers, with a focus on
[sched_ext](https://github.com/sched-ext/scx). Boots kernels in KVM
VMs with synthetic CPU topologies, runs workloads, and verifies
scheduling correctness. Also tests under the kernel's default EEVDF
scheduler.

- **Clean slate** -- each test boots its own kernel in KVM. Fresh state each run.
- **Topology as code** -- `topology(1, 2, 4, 2)` gives you 1 NUMA node, 2 LLCs, 4 cores, 2 threads with real NUMA domains when the host hardware supports them. x86_64 and aarch64.
- **Data-driven** -- scenarios declare cgroups, cpusets, workloads, and verification as data.
- **Gauntlet** -- one `#[ktstr_test]`, hundreds of topology x flag variants. Budget-aware CI selection and baseline A/B comparison.
- **Host-side introspection** -- read/write kernel state and BPF maps from host memory. No guest instrumentation.
- **Auto-repro** -- reruns failures with BPF probes on the crash call chain. Captures arguments and struct state at each call site. In kernel and BPF progs.
- **`#[ktstr_test]`** -- proc macro for integration tests that boot their own VMs. No custom harness needed.
- **[And more](https://likewhatevs.github.io/ktstr/guide/features.html)** -- 25 features across testing, observability, debugging, and infrastructure.

## Installation

```sh
cargo install ktstr                   # both binaries (ktstr + cargo-ktstr)
cargo install cargo-nextest           # required test runner
```

`ktstr` is the host-side CLI for running scenarios and managing cached
kernel images (outside VMs). `cargo-ktstr` (included in the same crate)
automates kernel configuration, building, and test execution in one
command. `scx-ktstr` (the test fixture scheduler) is built automatically
by the workspace and does not need a separate install.

## Setup

**Prerequisites:** Linux with `/dev/kvm`, Rust >= 1.88, clang,
pkg-config, plus autotools and make for the vendored libbpf/libelf/zlib
builds pulled in via `libbpf-sys`'s `vendored` feature. Test kernel:
Linux 6.12+ with sched_ext; `cargo ktstr kernel build` fetches one if
your host (`uname -r`) is older. See
[Supported kernels](https://likewhatevs.github.io/ktstr/guide/features.html#supported-kernels)
for details.

```sh
# Ubuntu/Debian
sudo apt install clang pkg-config make autoconf autopoint flex bison gawk
```

**Add to your crate**:

```toml
[dev-dependencies]
ktstr = { version = "0.4" }
```

The `anyhow::Result` referenced in examples below is re-exported
through `ktstr::prelude`; consumers do not need a separate `anyhow`
dev-dependency for the shown code to compile.

**Test files** go in `tests/` as standard Rust integration tests. Use `#[ktstr_test]` from `ktstr::prelude::*`.

See the [getting started guide](https://likewhatevs.github.io/ktstr/guide/getting-started.html) for Fedora packages, kernel discovery, and building a test kernel.

## Quick start

### Define a scheduler

Use `#[derive(Scheduler)]` to declare the scheduler binary, default
topology, and feature flags:

```rust
use ktstr::prelude::*;

#[derive(Scheduler)]
#[scheduler(name = "my_sched", binary = "scx_my_sched", topology(1, 2, 4, 1))]
#[allow(dead_code)]
enum MySchedFlag {
    #[flag(args = ["--enable-llc"])]
    Llc,
    #[flag(args = ["--enable-stealing"], requires = [Llc])]
    Steal,
}
```

This generates a `const MY_SCHED: Scheduler` and per-variant flag
constants. Tests referencing `MY_SCHED` inherit its topology and
flags. Without a scheduler, tests run under EEVDF. The topology
macro argument requires `llcs` to be an exact multiple of
`numa_nodes`; `topology(1, 2, 4, 1)` (2 LLCs, 1 NUMA node) is fine,
`topology(2, 3, ...)` is rejected at compile time.

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

#[ktstr_test(scheduler = MY_SCHED, llcs = 1, cores = 4, threads = 1)]
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

The frictionless loop is to build a cached kernel once and then run
tests against the cache:

```sh
cargo ktstr kernel build                                   # latest stable into XDG cache
cargo nextest run                                          # tests find the cached kernel
```

`cargo ktstr` wraps the full workflow and has subcommands beyond
`test`:

```sh
cargo ktstr test                                           # build/resolve kernel + run tests
cargo ktstr test --kernel ~/linux -- -E 'test(my_test)'    # local source tree + nextest filter
cargo ktstr coverage                                       # tests under llvm-cov
cargo ktstr kernel build 6.14.2                            # cache a specific version
cargo ktstr kernel build --source ~/linux                  # build from local source tree
cargo ktstr kernel build --git URL --ref v6.14             # shallow-clone a git tree
cargo ktstr kernel list                                    # list cached kernels (shows (EOL) tags)
cargo ktstr kernel clean --keep 3                          # keep 3 most recent
cargo ktstr verifier --scheduler scx_my_sched              # BPF verifier stats
cargo ktstr test-stats                                     # aggregate gauntlet sidecars
cargo ktstr shell --kernel 6.14.2                          # interactive VM shell
cargo ktstr completions bash                               # shell completions
```

### Host-side CLI

`ktstr` runs scenarios on the host (outside VMs) under whatever
scheduler is already active, and manages cached kernel images. Every
`ktstr kernel ...` subcommand is identical to the corresponding
`cargo ktstr kernel ...`.

```sh
ktstr list                                                 # list available scenarios
ktstr run                                                  # run all scenarios on the host
ktstr topo                                                 # show host CPU topology
ktstr cleanup                                              # remove leftover cgroups
ktstr shell --kernel 6.14.2                                # interactive VM shell (kernel optional)
ktstr kernel list                                          # manage cached kernels
ktstr kernel build 6.14.2
ktstr kernel build --source ../linux
ktstr kernel build --git URL --ref v6.14
ktstr kernel clean --keep 3
ktstr completions bash
```

Or via `cargo run` from the workspace:

```sh
cargo run --bin ktstr -- list
cargo run --bin ktstr -- run
```

## Documentation

**[Guide](https://likewhatevs.github.io/ktstr/guide/)** -- getting started, concepts,
writing tests, recipes, architecture.

**[API docs](https://likewhatevs.github.io/ktstr/api/ktstr/)** -- rustdoc for all workspace crates.

## License

GPL-2.0-only
