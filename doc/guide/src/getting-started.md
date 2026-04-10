# Getting Started

## Prerequisites

- Linux host with KVM access (`/dev/kvm`)
- Rust toolchain (stable, >= 1.88)
- clang and BPF toolchain (builds BPF skeletons via libbpf-cargo)
- libelf development headers
- pkg-config
- bpftool (generates vmlinux.h from the running kernel's BTF)
- cargo-nextest (`cargo install cargo-nextest`)

**Ubuntu/Debian:**

```sh
sudo apt install clang libelf-dev pkg-config bpftool
cargo install cargo-nextest
```

**Fedora:**

```sh
sudo dnf install clang elfutils-libelf-devel pkgconf bpftool
cargo install cargo-nextest
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
module re-exports the types you need.

The simplest test uses a canned scenario:

```rust,ignore
use stt::prelude::*;

#[stt_test(sockets = 1, cores = 2, threads = 1)]
fn my_test(ctx: &Ctx) -> Result<AssertResult> {
    scenarios::steady(ctx)
}
```

For custom cgroup topology, declare cgroups with `CgroupDef` and run
them with `execute_defs`. This is the most common custom test pattern:

```rust,ignore
use stt::prelude::*;

#[stt_test(sockets = 1, cores = 2, threads = 1)]
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
use stt::prelude::*;

#[stt_test(sockets = 1, cores = 2, threads = 1)]
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

No special setup is needed. `#[stt_test]` functions work with both
`cargo test` and `cargo nextest run` out of the box. The stt ctor
automatically intercepts nextest protocol args (`--list`, `--exact`)
for gauntlet expansion and budget-driven test selection.

- `cargo nextest run`: ctor intercepts, runs gauntlet-expanded tests.
- `cargo test`: standard harness runs the `#[test]` wrappers (base
  topology only, no gauntlet expansion).

## Kernel discovery

Tests require a bootable Linux kernel. stt searches (in order):

1. `STT_TEST_KERNEL` environment variable
2. `./linux/arch/x86/boot/bzImage` (workspace-local build tree)
3. `../linux/arch/x86/boot/bzImage` (sibling directory)
4. `/lib/modules/$(uname -r)/vmlinuz` (installed kernel)
5. `/boot/vmlinuz-$(uname -r)`
6. `/boot/vmlinuz` (unversioned symlink)

The host's installed kernel works for basic testing. For sched_ext
tests, build a kernel with the stt config fragment (below). See
[Troubleshooting](troubleshooting.md#no-kernel-found) for details.

## Run

```sh
cargo nextest run
```

Requires `/dev/kvm`. See
[Troubleshooting](troubleshooting.md#devkvm-not-accessible) if KVM
is unavailable.

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

To test your own scheduler:
[Test a New Scheduler](recipes/test-new-scheduler.md).
