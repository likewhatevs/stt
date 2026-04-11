# cargo-ktstr

`cargo ktstr test` automates kernel configuration, building, and test
execution. It is a cargo plugin that wraps `cargo nextest run`.

## --kernel

The `--kernel` flag is **required**. Pass the path to a Linux kernel
source tree:

```sh
cargo ktstr test --kernel ~/linux
```

## What it does

1. **Config check** -- reads `<kernel>/.config` for
   `CONFIG_SCHED_CLASS_EXT=y`.
2. **Auto-configure** -- if the config sentinel is missing, runs
   `make defconfig` (when no `.config` exists), merges `ktstr.kconfig`
   via `scripts/kconfig/merge_config.sh`, then runs `make olddefconfig`.
3. **Kernel build** -- runs `make -j$(nproc) KCFLAGS=-Wno-error`.
   This always runs; `make` handles the no-op case when the kernel
   is already built.
4. **compile_commands.json** -- runs `make compile_commands.json` to
   generate the compilation database for clangd / LSP.
5. **Test execution** -- execs `cargo nextest run` with `KTSTR_KERNEL`
   set in the environment for test kernel discovery.

## Passing nextest arguments

Arguments after `test` are passed through to `cargo nextest run`:

```sh
cargo ktstr test --kernel ~/linux                              # run all tests
cargo ktstr test --kernel ~/linux -- -E 'test(my_test)'        # nextest filter
cargo ktstr test --kernel ~/linux --workspace                  # all workspace tests
cargo ktstr test --kernel ~/linux -- --retries 2               # nextest retries
```

## Install

```sh
cargo install cargo-ktstr
```

Or build from the workspace:

```sh
cargo build -p cargo-ktstr
```
