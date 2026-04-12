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

`cargo ktstr test-stats` prints aggregate statistics from the most
recent nextest run by parsing its JUnit XML output.

```sh
cargo ktstr test-stats                           # default profile
cargo ktstr test-stats --profile ci              # ci profile
cargo ktstr test-stats --junit path/to/junit.xml # explicit file
```

The output includes:

- **Summary line** -- total, passed, failed, flaky, skipped, retries,
  wall-clock time.
- **Per-suite table** -- pass/fail/flaky counts and cumulative test
  time per test binary.
- **Failed tests** -- names, suites, retry counts, and failure
  messages.
- **Flaky tests** -- tests that passed after one or more retries.
- **Slowest tests** -- top 10 by duration.

Output is colorized when writing to a terminal.
`CARGO_TERM_COLOR=always` forces color even without a terminal (useful
in CI); `CARGO_TERM_COLOR=never` disables it even in a terminal. Set
`NO_COLOR` to disable color regardless (see
[Environment Variables](../reference/environment-variables.md)).

### Prerequisites

The nextest profile must have JUnit output enabled. The default
profile is configured with `[profile.default.junit]` in
`.config/nextest.toml`. Run tests first to generate the XML:

```sh
cargo nextest run --workspace        # generates target/nextest/default/junit.xml
cargo ktstr test-stats               # reads it
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--junit <PATH>` | -- | Path to a JUnit XML file. Overrides `--profile`. |
| `--profile <NAME>` | `default` | Nextest profile whose `junit.xml` to read. |

## Install

```sh
cargo install cargo-ktstr
```

Or build from the workspace:

```sh
cargo build -p cargo-ktstr
```
