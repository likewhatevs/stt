# cargo-ktstr

`cargo ktstr test` automates kernel configuration, building, and test
execution. It is a cargo plugin that wraps `cargo nextest run`.

## KTSTR_KERNEL

The `KTSTR_KERNEL` environment variable is **required**. Set it to
the path of a Linux kernel source tree:

```sh
export KTSTR_KERNEL=~/linux
```

## What it does

1. **Config check** -- reads `$KTSTR_KERNEL/.config` for
   `CONFIG_SCHED_CLASS_EXT=y`.
2. **Auto-configure** -- if the config sentinel is missing, runs
   `make defconfig` (when no `.config` exists), merges `ktstr.kconfig`
   via `scripts/kconfig/merge_config.sh`, then runs `make olddefconfig`.
3. **Kernel build** -- runs `make -j$(nproc) KCFLAGS=-Wno-error`.
   This always runs; `make` handles the no-op case when the kernel
   is already built.
4. **Test execution** -- execs `cargo nextest run` with `KTSTR_KERNEL`
   in the environment.

## Passing nextest arguments

Arguments after `test` are passed through to `cargo nextest run`:

```sh
cargo ktstr test                              # run all tests
cargo ktstr test -- -E 'test(my_test)'        # nextest filter
cargo ktstr test --workspace                  # all workspace tests
cargo ktstr test -- --retries 2               # nextest retries
```

## Install

```sh
cargo install cargo-ktstr
```

Or build from the workspace:

```sh
cargo build -p cargo-ktstr
```
