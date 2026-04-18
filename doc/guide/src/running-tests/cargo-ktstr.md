# cargo-ktstr

`cargo ktstr` is a cargo plugin for kernel build, cache, and test
workflow. Subcommands in `--help` order: `test`, `coverage`,
`stats`, `kernel`, `verifier`, `completions`, `shell`.

## test

Build the kernel (if needed) and run tests via `cargo nextest run`.

```sh
cargo ktstr test                                               # auto-discover kernel
cargo ktstr test --kernel ../linux                             # local source tree
cargo ktstr test --kernel 6.14.2                               # version (auto-downloads on miss)
cargo ktstr test --kernel 6.14.2-tarball-x86_64-kc...          # cache key (from kernel list)
```

`--kernel` accepts a path, version string, or cache key. When absent,
the test framework discovers a kernel from `KTSTR_TEST_KERNEL`, then
`KTSTR_KERNEL`, then falls back to cache and filesystem lookup.
When `--kernel` is a path,
cargo-ktstr configures and builds the kernel before running tests.
Version strings auto-download and build on cache miss (both explicit
patch versions like `6.14.2` and major.minor prefixes like `6.14`).
Cache keys resolve from the cache only — they error if not cached
(run `cargo ktstr kernel list` to see available keys).

| Flag | Default | Description |
|------|---------|-------------|
| `--kernel ID` | auto | Kernel identifier: path, version, or cache key. |
| `--no-perf-mode` | off | Disable all performance mode features (flock, pinning, RT scheduling, hugepages, NUMA mbind, KVM exit suppression). Also settable via `KTSTR_NO_PERF_MODE` env var. |

### What it does (path mode only)

These steps run only when `--kernel` is a source directory path.
Cached version and cache-key identifiers skip to step 5; uncached
version identifiers run through download + configure + build first.

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

## coverage

Build the kernel (if needed) and run tests with coverage via
`cargo llvm-cov nextest`. Same kernel resolution as `test`.

```sh
cargo ktstr coverage                                               # auto-discover kernel
cargo ktstr coverage --kernel ../linux                             # local source tree
cargo ktstr coverage --kernel 6.14.2                               # version (auto-downloads on miss)
cargo ktstr coverage -- --workspace --lcov --output-path lcov.info # lcov output
```

| Flag | Default | Description |
|------|---------|-------------|
| `--kernel ID` | auto | Kernel identifier: path, version, or cache key. |
| `--no-perf-mode` | off | Disable all performance mode features (flock, pinning, RT scheduling, hugepages, NUMA mbind, KVM exit suppression). Also settable via `KTSTR_NO_PERF_MODE` env var. |

Requires `cargo-llvm-cov` and the `llvm-tools-preview` rustup
component:

```sh
cargo install cargo-llvm-cov
rustup component add llvm-tools-preview
```

### Passing arguments

Arguments after `coverage` are passed through to
`cargo llvm-cov nextest`:

```sh
cargo ktstr coverage -- --workspace --profile ci --lcov --output-path lcov.info
cargo ktstr coverage -- --features integration
```

## kernel

Manage cached kernel images. Three subcommands: `list`, `build`,
`clean`. The standalone `ktstr kernel` subcommands are identical.

### kernel list

List cached kernel images, sorted newest first.

```sh
cargo ktstr kernel list
cargo ktstr kernel list --json       # JSON output for CI scripting
```

Human-readable output shows key, version, source type, arch, and
build timestamp. Entries built with a different `ktstr.kconfig` are
marked `(stale kconfig)`. Entries whose major.minor version is no
longer in kernel.org's active releases list are marked `(EOL)`;
prefix lookups for EOL series fall back to probing cdn.kernel.org
for the latest patch release.

| Flag | Description |
|------|-------------|
| `--json` | Output in JSON format. Each entry includes a boolean `eol` field (computed at list time by fetching kernel.org's `releases.json`) alongside the cached metadata. |

### kernel build

Download, build, and cache a kernel image. Three source modes:
version (tarball download), `--source` (local tree), `--git` (clone).

```sh
cargo ktstr kernel build                               # latest stable from kernel.org
cargo ktstr kernel build 6.14.2                        # specific version
cargo ktstr kernel build 6.15-rc3                      # RC release
cargo ktstr kernel build 6.12                          # latest 6.12.x patch release
cargo ktstr kernel build --source ../linux             # local source tree
cargo ktstr kernel build --git URL --ref v6.14         # git clone (shallow, depth 1)
cargo ktstr kernel build --force 6.14.2                # rebuild even if cached
```

When no version or source is given, fetches the latest stable
series that has had at least 8 maintenance releases — keeping CI
off brand-new majors whose early builds are more likely to break —
from kernel.org's `releases.json`. A major.minor prefix (e.g.
`6.12`) resolves to the highest patch release in that series. For
EOL series no longer in `releases.json`, probes cdn.kernel.org to
find the latest available tarball. Skips building when a cached entry already exists
(use `--force` to override). Stale entries (built with a different
`ktstr.kconfig`) are rebuilt automatically. For `--source`, generates
`compile_commands.json` for LSP support. Dirty local trees
(uncommitted changes to tracked files) are built but not cached.

| Flag | Description |
|------|-------------|
| `VERSION` | Kernel version or prefix to download (e.g. `6.14.2`, `6.12`, `6.15-rc3`). A major.minor prefix resolves to the highest patch release, probing cdn.kernel.org for EOL series. Conflicts with `--source` and `--git`. |
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
| `--kernel ID` | Kernel identifier: path, raw image file (`bzImage`/`Image`), version, or cache key. Source directories auto-build; version strings auto-download on cache miss. When absent, resolves via cache then filesystem, falling back to auto-download. |
| `--raw` | Print raw verifier output without cycle collapse. |
| `--all-profiles` | Run verifier for all flag profiles (power set). |
| `--profiles LIST` | Run verifier for specific profiles only (comma-separated, e.g. `default,llc,llc+steal`). |

See [BPF Verifier](verifier.md) for the verifier pipeline design and
output format.

## shell

Shares the VM boot flow with `ktstr shell` and accepts the same
flags. See [ktstr shell](ktstr.md#shell) for the full flag
reference. The one behavior difference from `ktstr shell` is that
`cargo ktstr shell` accepts raw image file paths for `--kernel`.

```sh
cargo ktstr shell
cargo ktstr shell --kernel 6.14.2
cargo ktstr shell --topology 1,2,4,1
cargo ktstr shell -i ./my-binary -i strace
```

## completions

Generate shell completions for cargo-ktstr. See
[ktstr completions](ktstr.md#completions) for the base subcommand.

```sh
cargo ktstr completions bash >> ~/.local/share/bash-completion/completions/cargo
cargo ktstr completions zsh > ~/.zfunc/_cargo-ktstr
cargo ktstr completions fish > ~/.config/fish/completions/cargo-ktstr.fish
```

| Arg | Description |
|------|-------------|
| `SHELL` | Shell to generate completions for (`bash`, `zsh`, `fish`, `elvish`, `powershell`). |
| `--binary NAME` | Binary name for completions. Default: `cargo`. |

## stats

Sidecar analysis and run-to-run comparison. See
[Runs](runs.md) for the directory layout.

```sh
cargo ktstr stats                          # print analysis of newest run
cargo ktstr stats list                     # list runs
cargo ktstr stats compare <a> <b>          # compare two runs
cargo ktstr stats compare <a> <b> -E filt  # compare with filter
```

When invoked without a subcommand, prints gauntlet analysis from the
most recent run directory under `{CARGO_TARGET_DIR or "target"}/ktstr/`
(or `KTSTR_SIDECAR_DIR` when set):

- **Gauntlet analysis** -- outlier detection, per-scenario/flags/topology
  dimension summaries, stimulus cross-tab.
- **BPF verifier stats** -- per-program verified instruction counts,
  warnings for programs near the 1M complexity limit.
- **BPF callback profile** -- per-program invocation counts, total
  CPU time, and average nanoseconds per call.
- **KVM stats** -- cross-VM averages for exits, halt polling, host
  preemptions.

### list

Print a table of run directories under
`{CARGO_TARGET_DIR or "target"}/ktstr/` with run key, test count,
and date. `KTSTR_SIDECAR_DIR` is **not** consulted -- runs written
to a custom path via that variable are invisible here.

### compare

Load two run directories from
`{CARGO_TARGET_DIR or "target"}/ktstr/`, join on
`(scenario, topology, work_type)`, compute per-metric deltas using
`MetricDef` polarity and thresholds from the unified metric
registry, and print colored output (red = regression, green =
improvement). Exits non-zero on regression. Like `list`, this
subcommand ignores `KTSTR_SIDECAR_DIR`.

| Flag | Default | Description |
|------|---------|-------------|
| `-E FILTER` | -- | Substring filter applied to `scenario topology work_type`. |
| `--threshold PCT` | per-metric `default_rel` | Relative significance threshold as a percentage. Overrides each metric's `default_rel` from the `MetricDef` registry. The absolute gate (`default_abs`) is independent and is **not** affected by this flag; deltas must clear both gates to count as significant. Omit to use each metric's built-in defaults. |

### Prerequisites

Run tests first to generate sidecar JSON files:

```sh
cargo nextest run --workspace        # generates target/ktstr/{kernel}-{git_short}/*.json
cargo ktstr stats                    # reads the newest run
```

Sidecars default to
`{CARGO_TARGET_DIR or "target"}/ktstr/{kernel}-{git_short}/`. Set
`KTSTR_SIDECAR_DIR` only when you need bare `cargo ktstr stats` to
read from a custom flat directory; the `list` and `compare`
subcommands always enumerate `target/ktstr/` and do not see runs
written to a custom path.

## Install

```sh
cargo install ktstr                   # installs both ktstr and cargo-ktstr
```

Or build from the workspace:

```sh
cargo build --bin cargo-ktstr
```
