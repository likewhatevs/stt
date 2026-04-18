# ktstr

[![CI](https://github.com/likewhatevs/ktstr/actions/workflows/ci.yml/badge.svg)](https://github.com/likewhatevs/ktstr/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/likewhatevs/ktstr/graph/badge.svg?token=E7GRAO2KZM)](https://codecov.io/gh/likewhatevs/ktstr)
[![guide](https://img.shields.io/badge/docs-guide-blue)](https://likewhatevs.github.io/ktstr/guide/)
[![api](https://img.shields.io/badge/docs-api-blue)](https://likewhatevs.github.io/ktstr/api/ktstr/)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](https://github.com/likewhatevs/ktstr/issues)

> **Early stage.** APIs, CLI, and internals are actively evolving.
> Expect breaking changes between releases.

Test harness for Linux process schedulers, with a focus on
[sched_ext](https://github.com/sched-ext/scx).

## Why ktstr?

sched_ext lets you write Linux process schedulers as BPF programs.
A scheduler runs on every CPU and
affects every process -- bugs cause system-wide stalls or crashes.
Scheduler behavior depends on CPU topology, cgroup hierarchy, workload
mix, and kernel version. You cannot test this with unit tests because
the relevant state only exists inside a running kernel. ktstr also
tests under EEVDF (the kernel's built-in scheduler) as a baseline.

Without ktstr, testing means manually booting a VM, setting up cgroups,
running workloads, and eyeballing whether things went wrong -- with no
reproducibility across machines because topology varies per host. ktstr
automates this:

- **Clean slate** -- each test boots its own kernel in a KVM VM. No
  shared state between tests.
- **Topology as code** -- `topology(1, 2, 4, 2)` gives you 1 NUMA
  node, 2 LLCs (last-level caches), 4 cores/LLC, 2 threads. x86_64
  and aarch64. The same
  test produces the same topology on any host.
- **Declarative scenarios** -- tests declare cgroups, cpusets, and
  workloads as data (`CgroupDef`, `Step`, `Op`). The framework
  handles the rest.
- **Automated assertions** -- checks for starvation, cgroup
  isolation violations, and CPU time fairness. No manual inspection.
- **[Gauntlet](https://likewhatevs.github.io/ktstr/guide/running-tests/gauntlet.html)** --
  one `#[ktstr_test]` expands across the cross-product of topology
  presets (4-252 vCPUs, 1-15 LLCs, optional SMT and multi-NUMA) and
  scheduler flag profiles, filtered by per-test constraints.
- **Host-side introspection** -- reads kernel state and BPF maps
  from guest memory without guest-side instrumentation.
- **Auto-repro** -- on failure, reruns the scenario with BPF probes
  on the crash call chain, capturing arguments and struct state at
  each call site.
- **[Features](https://likewhatevs.github.io/ktstr/guide/features.html)** --
  testing, observability, debugging, and infrastructure.

## Installation

```sh
cargo install ktstr                   # both binaries (ktstr + cargo-ktstr)
cargo install cargo-nextest           # recommended test runner (optional)
```

`ktstr` is the host-side CLI for running scenarios and managing cached
kernel images (outside VMs). `cargo-ktstr` (included in the same crate)
automates kernel configuration, building, and test execution in one
command. `scx-ktstr` (the test fixture scheduler) is built automatically
by the workspace and does not need a separate install.

## Setup

**Linux only (x86_64, aarch64).** ktstr boots KVM virtual machines;
it does not build or run on other platforms.

**Required:**

- Linux host with `/dev/kvm`
- Rust >= 1.88 (stable)
- clang (BPF skeleton compilation)
- pkg-config, make, gcc
- autotools (autoconf, autopoint, flex, bison, gawk) -- vendored
  libbpf/libelf/zlib build
- BTF (`/sys/kernel/btf/vmlinux` -- present by default on most
  distros; set `KTSTR_KERNEL` if missing)
- Internet access on first build (downloads busybox source)

**Optional:**

- [cargo-nextest](https://nexte.st/) -- enables gauntlet expansion;
  `cargo test` works without it for base topology.
- Test kernel: Linux 6.12+ with sched_ext for scheduler tests;
  `cargo ktstr kernel build` fetches and caches one. See
  [Supported kernels](https://likewhatevs.github.io/ktstr/guide/features.html#supported-kernels).

```sh
# Ubuntu/Debian
sudo apt install clang pkg-config make gcc autoconf autopoint flex bison gawk

# Fedora
sudo dnf install clang pkgconf make gcc autoconf gettext-devel flex bison gawk
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

See the [getting started guide](https://likewhatevs.github.io/ktstr/guide/getting-started.html) for kernel discovery and building a test kernel.

## Quick start

### Write a test

Declare cgroups and workers as data. No scheduler setup required:

```rust
use ktstr::prelude::*;

#[ktstr_test(llcs = 1, cores = 2, threads = 1)]
fn two_cgroups(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        CgroupDef::named("cg_0").workers(2),
        CgroupDef::named("cg_1").workers(2),
    ])
}
```

Each test boots a KVM VM, creates the declared cgroups and workers,
runs the workload, and checks for starvation and fairness. For
canned scenarios, see `scenarios::steady` in the
[getting started guide](https://likewhatevs.github.io/ktstr/guide/getting-started.html).

### Define a scheduler

To test a custom sched_ext scheduler, use `#[derive(Scheduler)]` to
declare the binary, default topology, and feature flags:

```rust
use ktstr::prelude::*;

#[derive(Scheduler)]
#[scheduler(name = "my_sched", binary = "scx_my_sched", topology(1, 2, 4, 1))]
enum MySchedFlag {
    #[flag(args = ["--enable-llc"])]
    Llc,
    #[flag(args = ["--enable-stealing"], requires = [Llc])]
    Steal,
}
```

`binary = "scx_my_sched"` tells ktstr to auto-discover the scheduler
binary in `target/{debug,release}/`, the directory containing the test
binary, or an explicit path via `KTSTR_SCHEDULER` env var. If the
scheduler is a `[[bin]]` target in the same workspace, `cargo build`
places it there and discovery is automatic. The resolved binary is
packed into the VM's initramfs. Tests without a `scheduler` attribute
run under EEVDF (the kernel's default scheduler).

`topology(numa_nodes, llcs, cores_per_llc, threads_per_core)` sets
the VM's CPU topology -- `topology(1, 2, 4, 1)` creates 1 NUMA node,
2 LLCs, 4 cores per LLC, 1 thread per core (8 vCPUs). Topologies
display as `NnNlNcNt` (e.g. `1n2l4c1t`). In `#[ktstr_test]`, use
named attributes instead: `llcs = 2, cores = 4, threads = 1,
numa_nodes = 1`. Unset dimensions inherit from the scheduler's
topology. For non-uniform NUMA, see `Topology::with_nodes()` in the
[topology guide](https://likewhatevs.github.io/ktstr/guide/concepts/topology.html).

This generates a `const MY_SCHED: Scheduler` and per-variant flag
constants. Tests referencing `MY_SCHED` inherit its topology and
flags. Add `scheduler = MY_SCHED` to `#[ktstr_test]` to use it:

```rust
#[ktstr_test(scheduler = MY_SCHED)]
fn sched_two_cgroups(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        CgroupDef::named("cg_0").workers(2),
        CgroupDef::named("cg_1").workers(2),
    ])
}
```

The topology macro argument requires `llcs` to be an exact multiple
of `numa_nodes`; `topology(1, 2, 4, 1)` (2 LLCs, 1 NUMA node) is
fine, `topology(2, 3, ...)` is rejected at compile time.

### Multi-step scenarios

For dynamic topology changes, use `execute_steps` with `Step` and
`HoldSpec`:

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
