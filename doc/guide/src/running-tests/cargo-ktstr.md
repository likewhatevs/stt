# cargo-ktstr

`cargo ktstr` is a cargo plugin for kernel build, cache, and test
workflow. Subcommands: `test`, `kernel`, `verifier`, `shell`,
`completions`, `test-stats`.

## test

Build the kernel (if needed) and run tests via `cargo nextest run`.

```sh
cargo ktstr test                                               # auto-discover kernel
cargo ktstr test --kernel ../linux                             # local source tree
cargo ktstr test --kernel 6.14.2                               # cached version
cargo ktstr test --kernel 6.14.2-tarball-x86_64                # explicit cache key
```

`--kernel` accepts a path, version string, or cache key. When absent,
the test framework discovers a kernel via `find_kernel()` (cache then
filesystem). When `--kernel` is a path, cargo-ktstr configures and
builds the kernel before running tests. Version strings and cache
keys resolve from the cache only -- they error if not cached (run
`cargo ktstr kernel build VERSION` first).

### What it does (path mode only)

These steps run only when `--kernel` is a source directory path.
Version strings and cache keys skip to step 5.

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

### Passing nextest arguments

Arguments after `test` are passed through to `cargo nextest run`:

```sh
cargo ktstr test -- -E 'test(my_test)'        # nextest filter
cargo ktstr test -- --workspace               # all workspace tests
cargo ktstr test -- --retries 2               # nextest retries
```

## kernel

Manage cached kernel images. Three subcommands: `list`, `build`,
`clean`.

### kernel list

List cached kernel images, sorted newest first.

```sh
cargo ktstr kernel list
cargo ktstr kernel list --json       # JSON output for CI scripting
```

Human-readable output shows key, version, source type, arch, and
build timestamp. Entries built with a different `ktstr.kconfig` are
marked `(stale kconfig)`.

| Flag | Description |
|------|-------------|
| `--json` | Output in JSON format. Includes all metadata fields. |

### kernel build

Download, build, and cache a kernel image. Three source modes:
version (tarball download), `--source` (local tree), `--git` (clone).

```sh
cargo ktstr kernel build                               # latest stable from kernel.org
cargo ktstr kernel build 6.14.2                        # specific version
cargo ktstr kernel build 6.15-rc3                      # RC release
cargo ktstr kernel build --source ../linux             # local source tree
cargo ktstr kernel build --git URL --ref v6.14         # git clone (shallow, depth 1)
cargo ktstr kernel build --force 6.14.2                # rebuild even if cached
```

When no version or source is given, fetches the latest stable version
from kernel.org's `releases.json`. Skips building when a cached entry
already exists (use `--force` to override). Stale entries (built with
a different `ktstr.kconfig`) are rebuilt automatically. For
`--source`, generates `compile_commands.json` for LSP support. Dirty
local trees (uncommitted changes to tracked files) are built but not
cached.

| Flag | Description |
|------|-------------|
| `VERSION` | Kernel version to download (e.g. `6.14.2`, `6.15-rc3`). Conflicts with `--source` and `--git`. |
| `--source PATH` | Path to existing kernel source directory. Conflicts with `VERSION` and `--git`. |
| `--git URL` | Git URL to clone. Requires `--ref`. Conflicts with `VERSION` and `--source`. |
| `--ref REF` | Git ref to checkout (branch, tag, commit). Required with `--git`. |
| `--force` | Rebuild even if a cached image exists. |
| `--clean` | Run `make mrproper` before configuring. Only meaningful with `--source`. |

### kernel clean

Remove cached kernel images.

```sh
cargo ktstr kernel clean                 # remove all (with confirmation prompt)
cargo ktstr kernel clean --keep 3        # keep 3 most recent
cargo ktstr kernel clean --force         # skip confirmation prompt
```

| Flag | Description |
|------|-------------|
| `--keep N` | Keep the N most recent cached kernels. |
| `--force` | Skip confirmation prompt. Required in non-interactive contexts. |

## verifier

Collect BPF verifier statistics for a scheduler. Builds the
scheduler, boots a VM, loads the BPF programs, and reports per-program
verified instruction counts from host-side memory introspection.

```sh
cargo ktstr verifier --scheduler scx_rustland
cargo ktstr verifier --scheduler-bin ./target/release/scx_rustland
cargo ktstr verifier --scheduler scx_rustland --raw
cargo ktstr verifier --scheduler scx_rustland --all-profiles
cargo ktstr verifier --scheduler scx_rustland --profiles default,llc,llc+steal
```

Either `--scheduler` (package name, built via `cargo build`) or
`--scheduler-bin` (pre-built binary path) is required. They conflict
with each other.

`--all-profiles` discovers flags via `--ktstr-list-flags`, generates
the power set of valid flag combinations (respecting `requires`
constraints), and collects verifier stats for each profile. When more
than one profile is run, a summary table compares `verified_insns`
across profiles.

| Flag | Description |
|------|-------------|
| `--scheduler PKG` | Scheduler package name to build and analyze. |
| `--scheduler-bin PATH` | Path to pre-built scheduler binary. Conflicts with `--scheduler`. |
| `--kernel ID` | Kernel identifier (path, version, or cache key). Auto-resolves when absent. |
| `--raw` | Print raw verifier output without cycle collapse. |
| `--all-profiles` | Run verifier for all flag profiles (power set). |
| `--profiles LIST` | Run verifier for specific profiles only (comma-separated, e.g. `default,llc,llc+steal`). |

See [BPF Verifier](verifier.md) for the verifier pipeline design and
output format.

## shell

Boot an interactive shell in a KVM virtual machine. Launches a VM
with busybox and drops into a shell.

```sh
cargo ktstr shell
cargo ktstr shell --kernel 6.14.2
cargo ktstr shell --topology 2,4,1
cargo ktstr shell -i ./my-binary -i strace
```

Files passed via `-i` are available at `/include-files/<name>` inside
the guest. Bare names (without path separators) are resolved via
`PATH` lookup. Dynamically-linked ELF binaries get automatic shared
library resolution via `ldd`. Non-ELF files are copied as-is.
`ldd` executes the binary's ELF interpreter -- only include trusted
binaries.

Stdin is a terminal requirement. The host terminal enters raw mode
for bidirectional stdin/stdout forwarding. Terminal state is restored
on all exit paths.

| Flag | Default | Description |
|------|---------|-------------|
| `--kernel ID` | auto | Kernel identifier (path, version, or cache key). |
| `--topology S,C,T` | `1,1,1` | Virtual CPU topology as `sockets,cores,threads`. All values must be >= 1. |
| `-i, --include-files PATH` | -- | Files to include in the guest. Repeatable. Directories are not supported. |

## completions

Generate shell completions for cargo-ktstr.

```sh
cargo ktstr completions bash >> ~/.local/share/bash-completion/completions/cargo
cargo ktstr completions zsh > ~/.zfunc/_cargo-ktstr
cargo ktstr completions fish > ~/.config/fish/completions/cargo-ktstr.fish
```

| Arg | Description |
|------|-------------|
| `SHELL` | Shell to generate completions for (`bash`, `zsh`, `fish`, `elvish`, `powershell`). |
| `--binary NAME` | Binary name for completions. Default: `cargo`. |

## test-stats

Print gauntlet analysis from sidecar JSON files produced by ktstr
tests.

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

| Flag | Description |
|------|-------------|
| `--dir PATH` | Path to the sidecar directory. Defaults to `KTSTR_SIDECAR_DIR` or `target/ktstr/{branch}-{hash}/`. |

## Deprecated: build-kernel

`cargo ktstr build-kernel` is deprecated. Use `cargo ktstr kernel build --source` instead:

```sh
# Old (deprecated):
cargo ktstr build-kernel --kernel ../linux

# New:
cargo ktstr kernel build --source ../linux
```

## Install

```sh
cargo install cargo-ktstr
```

Or build from the workspace:

```sh
cargo build -p cargo-ktstr
```
