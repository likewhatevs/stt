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

#[stt_test(sockets = 1, cores = 2, threads = 1)]
fn my_scheduler_test(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step::with_defs(
        vec![
            CgroupDef::named("cg_0").workers(2),
            CgroupDef::named("cg_1").workers(2),
        ],
        HoldSpec::Frac(1.0),
    )];
    execute_steps(ctx, steps)
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
