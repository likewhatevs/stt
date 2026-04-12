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
   `make defconfig` (when no `.config` exists), appends `ktstr.kconfig`
   to `.config`, then runs `make olddefconfig`.
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

## build-kernel

`cargo ktstr build-kernel` configures and builds the kernel without
running tests:

```sh
cargo ktstr build-kernel --kernel ~/linux
cargo ktstr build-kernel --kernel ~/linux --clean   # make mrproper first
```

Steps 1-4 from the `test` subcommand above apply. The `--clean` flag
runs `make mrproper` before configuring, forcing a full reconfigure
and rebuild. Both `build-kernel` and `test` generate
`compile_commands.json` after building.

## test-stats

`cargo ktstr test-stats` reads sidecar JSON files produced by ktstr
tests and prints the gauntlet analysis report.

```sh
cargo ktstr test-stats                     # default sidecar directory
cargo ktstr test-stats --dir path/to/dir   # explicit directory
```

The output includes:

- **Gauntlet analysis** -- outlier detection, per-scenario/flags/topology
  dimension summaries, stimulus cross-tab.
- **BPF verifier stats** -- per-program verified instruction counts,
  warnings for programs near the 1M complexity limit.
- **BPF callback profile** -- per-program invocation counts, total
  CPU time, and average nanoseconds per call.
- **KVM stats** -- cross-VM averages for exits, halt polling, host
  preemptions.

### Prerequisites

Run tests first to generate sidecar JSON files:

```sh
cargo nextest run --workspace        # generates target/ktstr/{branch}-{hash}/*.json
cargo ktstr test-stats               # reads them
```

Set `KTSTR_SIDECAR_DIR` to override the default sidecar directory.

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dir <PATH>` | -- | Path to the sidecar directory. Defaults to `KTSTR_SIDECAR_DIR` or `target/ktstr/{branch}-{hash}/`. |

## Install

```sh
cargo install cargo-ktstr
```

Or build from the workspace:

```sh
cargo build -p cargo-ktstr
```
