# Getting Started

## Prerequisites

- Linux host with KVM access (`/dev/kvm`)
- Rust toolchain (stable, >= 1.88)
- clang and BPF toolchain (builds BPF skeletons via libbpf-cargo)
- pkg-config

**Ubuntu/Debian:**

```sh
sudo apt install clang pkg-config
```

**Fedora:**

```sh
sudo dnf install clang pkgconf
```

## Install tools

```sh
cargo install cargo-nextest           # required test runner
cargo install ktstr                   # both binaries: ktstr + cargo-ktstr (optional)
```

`cargo-nextest` is the test runner. `cargo install ktstr` installs
both the host-side CLI (`ktstr`) and the dev workflow plugin
(`cargo-ktstr`).

## Add the dependency

Add ktstr as a dependency:

```toml
[dev-dependencies]
ktstr = { version = "0.3" }
```

## Write a test

Write a `#[ktstr_test]` function. The
[`prelude`](https://likewhatevs.github.io/ktstr/api/ktstr/prelude/index.html)
module re-exports the types you need.

The simplest test uses a canned scenario:

```rust,ignore
use ktstr::prelude::*;

#[ktstr_test(llcs = 1, cores = 2, threads = 1)]  // llcs = last-level caches
fn my_test(ctx: &Ctx) -> Result<AssertResult> {
    scenarios::steady(ctx)
}
```

For custom cgroup topology, declare cgroups with `CgroupDef` and run
them with `execute_defs`. This is the most common custom test pattern:

```rust,ignore
use ktstr::prelude::*;

#[ktstr_test(llcs = 1, cores = 2, threads = 1)]
fn my_test(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        CgroupDef::named("cg_0").workers(4),
        CgroupDef::named("cg_1")
            .workers(2)
            .work_type(WorkType::bursty(50, 100)),
    ])
}
```

For multi-phase scenarios with dynamic topology changes, use
`Step::with_defs` and `execute_steps`:

```rust,ignore
use ktstr::prelude::*;

#[ktstr_test(llcs = 1, cores = 2, threads = 1)]
fn my_test(ctx: &Ctx) -> Result<AssertResult> {
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

## Test binary setup

No special setup is needed. `#[ktstr_test]` functions work with both
`cargo test` and `cargo nextest run` out of the box. The ktstr ctor
automatically intercepts nextest protocol args (`--list`, `--exact`)
for gauntlet expansion and budget-driven test selection.

- `cargo nextest run`: ctor intercepts, runs gauntlet-expanded tests.
- `cargo test`: standard harness runs the `#[test]` wrappers (base
  topology only, no gauntlet expansion).

## Kernel discovery

Tests require a bootable Linux kernel. ktstr searches (in order):

1. `KTSTR_TEST_KERNEL` environment variable (direct image path)
2. `KTSTR_KERNEL` environment variable, parsed as one of three forms:
   - Path: search that directory for `arch/<arch>/boot/<image>`
   - Version (e.g. `6.14.2`): look up the version in XDG cache
   - Cache key (from `cargo ktstr kernel list`): exact cache lookup
3. XDG cache: most recent cached image (newest first); entries built
   with a different kconfig fragment are skipped. Skipped entirely when
   `KTSTR_KERNEL` was an explicit version or cache key that missed.
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
cargo ktstr kernel build               # latest stable
cargo ktstr kernel build 6.14.2        # specific version
cargo ktstr kernel build 6.12          # latest 6.12.x patch release
```

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

## Run

```sh
cargo nextest run
```

Requires `/dev/kvm`. See
[Troubleshooting](troubleshooting.md#devkvm-not-accessible) if KVM
is unavailable.

### Using cargo-ktstr

`cargo ktstr test` handles kernel resolution and test execution in
one command:

```sh
cargo ktstr test                                              # auto-discover kernel
cargo ktstr test --kernel ../linux                            # local source tree
cargo ktstr test --kernel 6.14.2                              # cached version
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

To understand scenarios, flags, and verification:
[Core Concepts](concepts.md).

To write new tests: [Writing Tests](writing-tests.md).

To test your own scheduler:
[Test a New Scheduler](recipes/test-new-scheduler.md).
