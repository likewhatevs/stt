# Getting Started

## Prerequisites

**Linux only (x86_64, aarch64).** ktstr boots KVM virtual machines;
it does not build or run on other platforms.

- Linux host with KVM access (`/dev/kvm`)
- Rust toolchain (stable, >= 1.95; pinned via `rust-toolchain.toml`)
- clang (BPF skeleton compilation)
- pkg-config, make, gcc
- autotools (autoconf, autopoint, flex, bison, gawk) -- vendored
  libbpf/libelf/zlib build
- BTF (`/sys/kernel/btf/vmlinux` -- present by default on most
  distros; set `KTSTR_KERNEL` if missing)
- Internet access on first build (downloads busybox source)
- Linux kernel 6.12+ for sched_ext tests (check with `uname -r`).
  The host kernel has no version requirement beyond KVM; the test
  kernel is whichever you build or cache via `cargo ktstr kernel build`.
  See [Supported kernels](features.md#supported-kernels) for
  per-feature version boundaries.

**Ubuntu/Debian:**

```sh
sudo apt install clang pkg-config make gcc autoconf autopoint flex bison gawk
```

**Fedora:**

```sh
sudo dnf install clang pkgconf make gcc autoconf gettext-devel flex bison gawk
```

## Install tools

```sh
cargo install cargo-nextest           # recommended test runner (optional)
cargo install ktstr                   # both binaries: ktstr + cargo-ktstr (optional)
```

`cargo-nextest` enables gauntlet expansion; `cargo test` works
without it for base topology. `cargo install ktstr` installs
both the host-side CLI (`ktstr`) and the dev workflow plugin
(`cargo-ktstr`).

## Add the dependency

Add ktstr as a dev-dependency:

```toml
[dev-dependencies]
ktstr = { version = "0.4" }
```

## Kernel discovery

Tests require a bootable Linux kernel. The test harness checks these
locations in order:

1. `KTSTR_TEST_KERNEL` environment variable (direct image path).
2. `KTSTR_KERNEL` environment variable, parsed as one of three forms:
   - Path: search that directory for `arch/<arch>/boot/<image>`
   - Version (e.g. `6.14.2`): look up the version in XDG cache
   - Cache key (from `cargo ktstr kernel list`): exact cache lookup
3. XDG cache: most recent cached image (newest first); entries built
   with a different kconfig fragment are skipped. When `KTSTR_KERNEL`
   named an explicit version or cache key that was not present in the
   cache, the cache scan is skipped entirely -- discovery moves on to
   step 4 rather than substituting an unrelated cached kernel.
4. `./linux/arch/<arch>/boot/<image>` (workspace-local build tree)
5. `../linux/arch/<arch>/boot/<image>` (sibling directory)
6. `/lib/modules/$(uname -r)/build/arch/<arch>/boot/<image>` (installed kernel build tree)
7. `/lib/modules/$(uname -r)/vmlinuz` (installed kernel)
8. `/boot/vmlinuz-$(uname -r)`
9. `/boot/vmlinuz` (unversioned symlink)

On x86_64, the build-tree image is `arch/x86/boot/bzImage`; on
aarch64, `arch/arm64/boot/Image`.

The host's installed kernel works for basic testing. For sched_ext
tests, build a kernel with the ktstr config fragment (below). See
[Troubleshooting](troubleshooting.md#no-kernel-found) for details.

## Build a kernel

`cargo ktstr kernel build` downloads a kernel tarball from kernel.org,
configures it with the embedded `ktstr.kconfig` fragment, builds it,
and caches the result:

```sh
cargo ktstr kernel build               # latest stable series with >= 8 maintenance releases
cargo ktstr kernel build 6.14.2        # specific version
cargo ktstr kernel build 6.12          # highest 6.12.x patch release
cargo ktstr kernel build 6             # highest 6.x.y release
```

The bare `cargo ktstr kernel build` skips series that have fewer
than 8 maintenance releases to keep CI off brand-new majors whose
early point releases tend to hit build issues on older toolchains;
pass the specific version explicitly if you need a series that
hasn't reached `.8` yet.

Subsequent runs of `cargo ktstr test` or `cargo nextest run` will
find the cached kernel automatically (step 3 in the discovery chain
above).

To build from a local source tree:

```sh
cargo ktstr kernel build --source ../linux
```

To list and manage cached kernels:

```sh
cargo ktstr kernel list
cargo ktstr kernel clean --keep 3
```

See [cargo-ktstr](running-tests/cargo-ktstr.md#kernel) for all
options.

### Manual

```sh
cd /path/to/linux
make defconfig
cat /path/to/ktstr/ktstr.kconfig >> .config
make olddefconfig
make -j$(nproc)
```

`ktstr.kconfig` in the repo root contains a kernel config fragment
tuned for scheduler testing (sched_ext, BPF, kprobes, minimal boot).

## Write a test

Create a file in your crate's `tests/` directory (e.g.
`tests/sched_test.rs`) and write a `#[ktstr_test]` function. The
[`prelude`](https://likewhatevs.github.io/ktstr/api/ktstr/prelude/index.html)
module re-exports the types you need.

The simplest test uses a canned scenario. `AssertResult` carries the
pass/fail verdict, diagnostic messages, and per-cgroup statistics from
the run.

```rust,ignore
use ktstr::prelude::*;

#[ktstr_test(llcs = 1, cores = 2, threads = 1)]  // llcs = last-level caches
fn my_test(ctx: &Ctx) -> Result<AssertResult> {
    // `scenarios::steady` is a canned scenario: two cgroups of equal
    // CPU-spin workers, no cpuset restrictions, run for the default
    // duration.
    scenarios::steady(ctx)
}
```

For custom cgroup topology, declare cgroups with `CgroupDef` and run
them with `execute_defs`. A `CgroupDef` bundles the cgroup name,
optional cpuset, and workload specification into a single declaration.
This is the most common custom test pattern:

```rust,ignore
use ktstr::prelude::*;

#[ktstr_test(llcs = 1, cores = 2, threads = 1)]
fn my_test(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        CgroupDef::named("cg_0").workers(4),
        CgroupDef::named("cg_1")
            .workers(2)
            // bursty(50, 100): CPU burst for 50 ms, sleep for 100 ms, repeat.
            .work_type(WorkType::bursty(50, 100)),
    ])
}
```

`execute_defs` is a convenience wrapper that creates a single step
holding for the full duration -- use it when all cgroups run
concurrently for one phase. Use `execute_steps` when you need
multiple phases (e.g., adding cgroups mid-test or changing cpusets
between phases).

`Step::with_defs` pairs a list of `CgroupDef`s with a `HoldSpec` that
controls how long the step runs. This example starts two cgroups, then
adds a third mid-test:

```rust,ignore
use ktstr::prelude::*;

#[ktstr_test(llcs = 1, cores = 4, threads = 1)]
fn my_test(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![
        // Phase 1: two cgroups for the first half.
        Step::with_defs(
            vec![
                CgroupDef::named("cg_0").workers(2),
                CgroupDef::named("cg_1").workers(2),
            ],
            HoldSpec::Frac(0.5),
        ),
        // Phase 2: add a third cgroup for the remaining half.
        Step::with_defs(
            vec![CgroupDef::named("cg_2").workers(2)],
            HoldSpec::Frac(0.5),
        ),
    ];
    execute_steps(ctx, steps)
}
```

### How it runs

The framework boots a KVM VM with the requested topology and runs
your test binary as the guest's init process. Your test function
executes **inside the VM** -- `execute_defs` and `execute_steps`
immediately create cgroups, spawn workers, run the workload, and
return assertion results. `Ctx` provides the guest topology
(`ctx.topo`) and cgroup management (`ctx.cgroups`).

### What gets checked

Every test automatically checks for worker starvation, scheduling
fairness, scheduling gaps, and host-side runqueue health (including imbalance,
stalls, dispatch queue depth). These defaults come from
`Assert::default_checks()` and can be overridden per-scheduler or
per-test. See [Checking](concepts/checking.md) for the full
list of checks and thresholds.

## Run

No special setup is needed. `#[ktstr_test]` functions work with both
`cargo test` and `cargo nextest run` out of the box. The ktstr ctor
automatically intercepts nextest protocol args (`--list`, `--exact`)
for gauntlet expansion and budget-driven test selection.

- `cargo nextest run`: ctor intercepts, runs gauntlet-expanded tests.
- `cargo test`: standard harness runs the `#[test]` wrappers (base
  topology only, no gauntlet expansion).

```sh
cargo nextest run
```

Requires `/dev/kvm`. See
[Troubleshooting](troubleshooting.md#devkvm-not-accessible) if KVM
is unavailable.

Passing tests:

```
    PASS [  11.34s] my_crate::my_sched_tests ktstr/my_test
```

A failing test prints assertion details:

```
    FAIL [  12.05s] my_crate::my_sched_tests ktstr/my_test

--- STDERR ---
ktstr_test 'my_test' [topo=1n1l2c1t] failed:
  stuck 3500ms on cpu1 at +1200ms

--- stats ---
4 workers, 2 cpus, 8 migrations, worst_spread=12.3%, worst_gap=3500ms
  cg0: workers=2 cpus=2 spread=5.1% gap=3500ms migrations=4 iter=15230
  cg1: workers=2 cpus=2 spread=12.3% gap=890ms migrations=4 iter=14870
```

### Using cargo-ktstr

`cargo ktstr test` handles kernel resolution and test execution in
one command:

```sh
cargo ktstr test                                              # auto-discover kernel
cargo ktstr test --kernel ../linux                            # local source tree
cargo ktstr test --kernel 6.14.2                              # version (auto-downloads on miss)
cargo ktstr test -- -E 'test(my_test)'                        # pass nextest args
```

See [cargo-ktstr](running-tests/cargo-ktstr.md) for details.

### Interactive shell

`cargo ktstr shell` boots a VM with busybox for manual exploration:

```sh
cargo ktstr shell                              # default 1,1,1,1 topology
cargo ktstr shell --topology 1,2,4,1           # 1 NUMA node, 2 LLCs, 4 cores/LLC, 1 thread/core
cargo ktstr shell -i ./my-scheduler            # include a file in the guest
cargo ktstr shell -i ./test-data/              # include a directory recursively
```

Included ELF binaries get automatic shared library resolution.
Directories are walked recursively; their contents appear under
`/include-files/<dirname>/` preserving the original structure.
Individual files are available at `/include-files/<name>` inside the guest.
See [cargo-ktstr shell](running-tests/cargo-ktstr.md#shell) for
details.

## Next steps

To understand scenarios, flags, and checking:
[Core Concepts](concepts.md).

To write new tests: [Writing Tests](writing-tests.md).

To test your own scheduler:
[Test a New Scheduler](recipes/test-new-scheduler.md).
