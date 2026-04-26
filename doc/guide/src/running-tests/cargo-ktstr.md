# cargo-ktstr

`cargo ktstr` is a cargo plugin for kernel build, cache, and test
workflow. Subcommands in `--help` order: `test` (alias: `nextest`),
`coverage`, `llvm-cov`, `stats`, `kernel`, `model`, `verifier`,
`completions`, `show-host`, `show-thresholds`, `cleanup`, `shell`.

## test

Build the kernel (if needed) and run tests via `cargo nextest run`.
Also available as `cargo ktstr nextest` — a visible clap alias that
expands to the same subcommand, so the two forms are interchangeable.

```sh
cargo ktstr test                                               # auto-discover kernel
cargo ktstr test --kernel ../linux                             # local source tree
cargo ktstr test --kernel 6.14.2                               # version (auto-downloads on miss)
cargo ktstr test --kernel 6.14.2-tarball-x86_64-kc...          # cache key (from kernel list)
cargo ktstr test --kernel 6.12..6.14                           # range: every stable+longterm release in [6.12, 6.14]
cargo ktstr test --kernel git+https://example.com/r.git#v6.14  # git URL + ref (tag/branch)
cargo ktstr test --kernel git+https://example.com/r.git#deadbeef1234  # specific commit
cargo ktstr test --kernel 6.14.2 --kernel 6.15.0               # multi-kernel: repeatable
cargo ktstr test --release                                     # release profile (stricter assertions)
```

`--kernel` is **repeatable** and accepts a path, version string,
cache key, version range (`START..END`), or git source
(`git+URL#REF`). When absent, the test framework discovers a kernel
from `KTSTR_TEST_KERNEL`, then `KTSTR_KERNEL`, then falls back to
cache and filesystem lookup. When `--kernel` is a path,
cargo-ktstr configures and builds the kernel before running tests.
Version strings auto-download and build on cache miss (both
explicit patch versions like `6.14.2` and major.minor prefixes like
`6.14`). Cache keys resolve from the cache only — they error if not
cached (run `cargo ktstr kernel list` to see available keys).

Ranges (`START..END`) expand against kernel.org's `releases.json`
to every `stable` and `longterm` release whose version sits inside
`[START, END]` inclusive (mainline / linux-next rows are dropped).
The endpoints themselves do NOT need to appear in `releases.json` —
`6.10..6.16` brackets the surviving releases even if `6.10` and
`6.16` have aged out.

Git sources (`git+URL#REF`) clone the repo shallow at the given
ref, build, and cache the result. A repeat invocation against an
unchanged branch tip lands a cache hit; a moved tip rebuilds.

### Multi-kernel: kernel as a gauntlet dimension

When `--kernel` resolves to **two or more kernels** (multiple
`--kernel` flags, or a single `--kernel START..END` range that
expands to several releases), cargo-ktstr resolves all kernels
upfront and exports the resolved set to `cargo nextest` via the
`KTSTR_KERNEL_LIST` env var. The test binary's gauntlet expansion
adds the kernel as an additional dimension to the gauntlet
cross-product, so each `(test × scenario × topology × flags × kernel)`
tuple becomes a distinct nextest test case. Two name shapes carry
the kernel suffix:

- **Base tests**: `ktstr/{name}/{kernel_label}` — one variant per
  registered `#[ktstr_test]` per kernel.
- **Gauntlet variants**: `gauntlet/{name}/{preset}/{profile}/{kernel_label}` —
  one variant per (test × topology preset × flag profile × kernel).

Single-kernel runs (zero or one resolved kernel) keep the
historical name shapes `ktstr/{name}` and
`gauntlet/{name}/{preset}/{profile}` with no kernel suffix, so
existing CI baselines and per-test config overrides keep matching.

Kernel labels are semantic, operator-readable identifiers
sanitized to `kernel_[a-z0-9_]+`:

- Version / range expansion → `kernel_6_14_2`, `kernel_6_15_rc3`
- Cache key → version prefix only (`kernel_6_14_2` from
  `6.14.2-tarball-x86_64-kc<hash>`)
- Git source → `kernel_git_{owner}_{repo}_{ref}` (e.g.
  `kernel_git_tj_sched_ext_for_next` from
  `git+https://github.com/tj/sched_ext#for-next`)
- Path → `kernel_path_{basename}_{hash6}` (e.g.
  `kernel_path_linux_a3f2b1`); the 6-char crc32 of the canonical
  path disambiguates two `linux` directories under different
  parents.

Filter with nextest's `-E 'test(kernel_6_14)'` to pick a single
kernel from a multi-kernel matrix; nextest's parallelism, retries,
and `--ignored` flag all apply natively. Sidecars partition per
kernel: each kernel runs in its own
`target/ktstr/{kernel}-{timestamp}/` directory keyed on the
resolved kernel's identity. Coverage profraw does NOT partition
per kernel — `__llvm_profile_write_buffer` writes flat into
`target/llvm-cov-target/` with PID-keyed filenames
(`ktstr-test-{pid}-{counter}.profraw`), and cargo-llvm-cov merges
every variant's profraw automatically into the single output
report.

Build / download / clone failures abort BEFORE any test runs — a
missing kernel can't be tested, and continuing would mask which
kernel was requested-but-unavailable in the operator-visible error
stream. Test failures within a kernel are nextest-handled
normally.

**`host_only` tests under multi-kernel**: tests marked
`host_only` (those that run on the host without booting a VM)
currently still expand across the kernel dimension and run N
times for N kernels — N copies of identical work, since the test
never observes the kernel directory. A future change will skip
the kernel suffix for `host_only` entries so they list and run
once regardless of `KTSTR_KERNEL_LIST` cardinality.

| Flag | Default | Description |
|------|---------|-------------|
| `--kernel ID` (repeatable) | auto | Kernel identifier: path, version, cache key, range (`START..END`), or git source (`git+URL#REF`). Repeatable; a multi-kernel set fans the gauntlet across kernels. |
| `--no-perf-mode` | off | Disable all performance mode features (flock, pinning, RT scheduling, hugepages, NUMA mbind, KVM exit suppression). Also settable via `KTSTR_NO_PERF_MODE` env var. |
| `--release` | off | Build and run tests with the release profile (`--cargo-profile release` to nextest). Release mode applies **stricter assertion thresholds** (`gap_threshold_ms` 2000 vs debug's 3000, `spread_threshold_pct` 15% vs debug's 35%) — tests that barely pass in debug may fail under `--release`. `catch_unwind`-based tests and tests gated on `#[cfg(debug_assertions)]` are skipped. |

### What it does (path mode only)

These steps run only when `--kernel` is a source directory path.
Cached version and cache-key identifiers skip to step 5; uncached
version identifiers run through download + configure + build first.
Ranges fan out to per-version resolution (every release downloads
+ builds + caches independently if not already present) before
reaching step 5; git sources clone shallow at the ref, build, and
cache the result. Multi-kernel resolution finishes for every
requested kernel BEFORE step 5 — the cargo-nextest invocation in
step 5 sees the complete kernel set as a single `KTSTR_KERNEL_LIST`
export, so nextest fans the gauntlet across kernels in a single
run.

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
5. **Test execution** -- execs `cargo nextest run` once with
   `KTSTR_KERNEL` set in the environment (single-kernel) or with
   both `KTSTR_KERNEL` and `KTSTR_KERNEL_LIST` (multi-kernel; the
   latter encodes the resolved kernel set as
   `label1=path1;label2=path2;…`). The test binary's gauntlet
   expansion adds the kernel as a fifth dimension when the list
   carries 2+ entries; nextest's parallelism, retries, and `-E`
   filtering apply natively to every (test × kernel) variant.

### Passing nextest arguments

Arguments after `test` are passed through to `cargo nextest run`:

```sh
cargo ktstr test -- -E 'test(my_test)'        # nextest filter
cargo ktstr test -- --workspace               # all workspace tests
cargo ktstr test -- --retries 2               # nextest retries
```

## coverage

Build the kernel (if needed) and run tests with coverage via
`cargo llvm-cov nextest`. Same kernel resolution and multi-kernel
semantics as `test`: `--kernel` is repeatable; multi-kernel runs
add the kernel suffix to every test name and partition the
sidecar tree per kernel via
`target/ktstr/{kernel}-{timestamp}/`. Coverage profraw lands flat
in `target/llvm-cov-target/` with PID-keyed filenames — it does
NOT partition per kernel — and cargo-llvm-cov merges every
variant's profraw automatically into the single output report.

```sh
cargo ktstr coverage                                               # auto-discover kernel
cargo ktstr coverage --kernel ../linux                             # local source tree
cargo ktstr coverage --kernel 6.14.2                               # version (auto-downloads on miss)
cargo ktstr coverage --kernel 6.14.2 --kernel 6.15.0               # multi-kernel coverage matrix
cargo ktstr coverage --release                                     # release profile (stricter assertions)
cargo ktstr coverage -- --workspace --lcov --output-path lcov.info # lcov output
```

| Flag | Default | Description |
|------|---------|-------------|
| `--kernel ID` (repeatable) | auto | Same shapes and multi-kernel semantics as `cargo ktstr test --kernel`: each (test × kernel) variant runs as its own nextest subprocess so cargo-llvm-cov merges every variant's profraw automatically. |
| `--no-perf-mode` | off | Disable all performance mode features (flock, pinning, RT scheduling, hugepages, NUMA mbind, KVM exit suppression). Also settable via `KTSTR_NO_PERF_MODE` env var. |
| `--release` | off | Collect coverage with the release profile (`--cargo-profile release` to llvm-cov nextest). Same stricter-threshold caveats as `test --release` — release mode applies `gap_threshold_ms=2000` / `spread_threshold_pct=15%`, and skips `catch_unwind`-based tests along with `#[cfg(debug_assertions)]`-gated tests. |

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

## llvm-cov

Raw passthrough to `cargo llvm-cov` with arbitrary arguments. Use
this for `llvm-cov` subcommands that don't fit the `coverage`
flow — `report`, `clean`, `show-env`, etc. When you want
`cargo llvm-cov nextest`, prefer [`cargo ktstr coverage`](#coverage);
this subcommand carries the same kernel-resolution and
`--no-perf-mode` plumbing but hands every remaining argument to
`cargo llvm-cov` unchanged.

```sh
cargo ktstr llvm-cov report --lcov --output-path lcov.info    # generate report from prior run
cargo ktstr llvm-cov clean --workspace                         # wipe accumulated coverage data
cargo ktstr llvm-cov show-env                                  # print env cargo-llvm-cov would set
cargo ktstr llvm-cov --kernel ../linux report                  # pin kernel + passthrough
```

| Flag | Default | Description |
|------|---------|-------------|
| `--kernel ID` (repeatable) | auto | Kernel identifier: path, version, cache key, range (`START..END`), or git source (`git+URL#REF`). Same multi-kernel semantics as `cargo ktstr test --kernel`. |
| `--no-perf-mode` | off | Disable all performance mode features (flock, pinning, RT scheduling, hugepages, NUMA mbind, KVM exit suppression). Also settable via `KTSTR_NO_PERF_MODE` env var. |

Note: a bare `cargo ktstr llvm-cov` (no trailing subcommand)
dispatches to `cargo llvm-cov`, which runs `cargo test` — ktstr
tests rely on the nextest harness for gauntlet expansion
(flag-profile × topology-preset variants) and VM dispatch. Under
bare `cargo test`, only the `#[test]` stubs run and gauntlet
variants are silently skipped. Always pass a subcommand after
`llvm-cov` (most often `nextest`, for which `cargo ktstr coverage`
is the shorter route).

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
cargo ktstr stats                                             # print analysis of newest run
cargo ktstr stats list                                        # list runs
cargo ktstr stats list-metrics                                # list registered regression metrics
cargo ktstr stats compare --a-kernel 6.14 --b-kernel 6.15     # slice on kernel
cargo ktstr stats compare --a-scheduler scx_rusty --b-scheduler scx_alpha  # slice on scheduler
cargo ktstr stats compare --a-kernel 6.14 --b-kernel 6.15 --scheduler scx_rusty  # slice on kernel, pin scheduler on both sides
cargo ktstr stats compare --a-kernel 6.14 --b-kernel 6.15 -E cgroup_steady       # add substring filter
cargo ktstr stats compare --a-commit abcdef1 --b-commit fedcba2 --no-average     # opt out of trial averaging
```

When invoked without a subcommand, prints gauntlet analysis from
either the most recent run directory under
`{CARGO_TARGET_DIR or "target"}/ktstr/` (newest by mtime) or the
explicit directory in `KTSTR_SIDECAR_DIR` when that variable is
set. With `KTSTR_SIDECAR_DIR` set, that directory is the sidecar
source directly -- there is no newest-subdirectory walk under it:

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
and date.

### list-metrics

List the registered regression metrics and their default
thresholds. Enumerates the `ktstr::stats::METRICS` registry: metric
name, polarity (higher/lower better), default absolute-delta gate,
default relative-delta gate, display unit, and a one-line
description. Use this to see which metric names
`ComparisonPolicy.per_metric_percent` keys can reference, and what
each default absolute and relative gate starts at before an
override. Default output is a human-readable table; `--json` emits
a JSON array with the same fields.

```sh
cargo ktstr stats list-metrics              # table
cargo ktstr stats list-metrics --json       # JSON array
```

| Flag | Default | Description |
|------|---------|-------------|
| `--json` | off | Emit JSON instead of a table. |

### show-host

Print the archived `HostContext` for a specific run: CPU identity,
memory/hugepage config, transparent-hugepage policy, NUMA node
count, kernel uname triple, kernel cmdline, and every
`/proc/sys/kernel/sched_*` tunable captured at archive time. Useful
for inspecting the same fingerprint `compare`'s host-delta section
uses, available on a single run.

The command scans sidecars in the run directory in iteration order
and prints the FIRST sidecar that carries a populated host field —
older pre-enrichment sidecars may have `host: None`, and the
forward scan tolerates those. If no sidecar has a populated host
field the command fails with an actionable error rather than
returning empty output.

| Flag | Default | Description |
|------|---------|-------------|
| `--run ID` | required | Run key (from `cargo ktstr stats list`). |
| `--dir DIR` | `target/ktstr/` | Alternate run root. Same semantics as `compare --dir`: useful for archived sidecar trees copied off a CI host. |

### compare

Pool every sidecar under `target/ktstr/` (or `--dir`), partition
the rows into A and B sides via per-side filter flags, average
each side's matching sidecars per pairing key (or pass through
distinct sidecars under `--no-average`), and report regressions
on the A→B delta. Exits non-zero on regression.

The dimensions on which the A and B filters DIFFER are the
SLICING dimensions — the axes of the A/B contrast. Every other
dimension is part of the dynamic PAIRING key the comparison
joins on. Slicing dims are derived automatically from the
filters:

```sh
# Slice on kernel: A is 6.14, B is 6.15. Pair on every other dim.
cargo ktstr stats compare --a-kernel 6.14 --b-kernel 6.15

# Slice on kernel AND scheduler simultaneously.
cargo ktstr stats compare \
    --a-kernel 6.14 --a-scheduler scx_rusty \
    --b-kernel 6.15 --b-scheduler scx_alpha

# Slice on commit, narrow both sides to one scheduler+kernel.
cargo ktstr stats compare \
    --a-commit abcdef1 --b-commit fedcba2 \
    --kernel 6.14 --scheduler scx_rusty
```

**Symmetric sugar.** Shared `--X` flags (`--kernel`, `--scheduler`,
`--topology`, `--work-type`, `--commit`, `--flag`) pin BOTH
sides to the same value(s). Per-side `--a-X` / `--b-X` flags
REPLACE the corresponding shared `--X` value for that side
only — "more-specific replaces" semantics. So
`--kernel 6.14 --a-kernel 6.13` puts A on 6.13 and B on 6.14.

**Validation.** The dispatch site rejects two cases up front:
- **Empty slicing**: no `--a-X` / `--b-X` at all, OR the per-side
  flags resolve to identical effective filters. Bails with
  "specify at least one per-side filter (e.g. `--a-kernel 6.14
  --b-kernel 6.15`) to define what dimension separates the two
  sides."
- **Multi-dim slicing**: slicing on more than one dimension
  prints a warning to stderr ("warning: slicing on N
  dimensions; results compress multiple axes into a single A/B
  contrast") but continues — multi-dim contrasts are a
  deliberate feature for cohort sweeps.

**Averaging.** By default the comparison aggregates every
matching sidecar within each side into a single arithmetic-mean
row per pairing key, smoothing run-to-run jitter. Failing /
skipped contributors are excluded from the metric mean; the
aggregated row's `passed` is the AND across every contributor.
A header line above the comparison table reads `averaged across
N runs (A) and M runs (B)` and a per-group
`passes_observed/total_observed` block prints below the summary.

`--no-average` keeps each sidecar distinct. If multiple sidecars
on the same side share the same pairing key under `--no-average`,
the comparison bails with "duplicate pairing keys" — pairing
across A/B sides is ambiguous when one A-row could match many
B-rows. Either drop `--no-average` to average them, or add
another per-side filter to disambiguate.

**Kernel match shape.** A `--kernel 6.12` filter (two-segment
major.minor) PREFIX-matches every patch release in that series:
`6.12`, `6.12.0`, `6.12.5` all match. A three-or-more-segment
filter (`--kernel 6.14.2`, `--kernel 6.15-rc3`) is strict
equality — `6.14.2` does NOT match `6.14.20`. The same shape
applies to `--a-kernel` / `--b-kernel`.

| Flag | Default | Description |
|------|---------|-------------|
| `-E FILTER` | -- | Substring filter applied to the joined `scenario topology scheduler work_type flags` string. Composes with the typed dimension filters: typed narrows happen first, substring runs over the surviving set. |
| `--kernel VER` (repeatable) | -- | Pin BOTH sides to the listed kernel version(s). Sugar for `--a-kernel V1 --a-kernel V2 --b-kernel V1 --b-kernel V2`. Per-side `--a-kernel` / `--b-kernel` REPLACES this shared value for that side only. Major.minor (`6.12`) prefix-matches; three-segment (`6.14.2`) is strict. |
| `--scheduler NAME` | -- | Pin BOTH sides to one scheduler. Sugar for `--a-scheduler N --b-scheduler N`. Strict equality. |
| `--topology LABEL` | -- | Pin BOTH sides to one rendered topology label (e.g. `1n2l4c2t`). Strict equality. |
| `--work-type TYPE` | -- | Pin BOTH sides to one work_type (PascalCase variant of `WorkType`, e.g. `CpuSpin`). See [Work types](../concepts/work-types.md). |
| `--commit HASH` (repeatable) | -- | Pin BOTH sides to listed `project_commit` value(s) (7-char hex, optional `-dirty` suffix). Filters the ktstr framework commit; the scheduler binary's commit (`SidecarResult::scheduler_commit`) is not currently exposed as a filter. |
| `--flag NAME` | -- | Repeatable AND-combined flag filter pinning BOTH sides. Every listed flag must appear in `active_flags`; rows may carry additional flags. |
| `--a-kernel VER` (repeatable) | -- | A-side kernel filter. Replaces the shared `--kernel` for the A side only. |
| `--a-scheduler NAME` | -- | A-side scheduler filter. Replaces the shared `--scheduler` for the A side only. |
| `--a-topology LABEL` | -- | A-side topology filter. Replaces the shared `--topology` for the A side only. |
| `--a-work-type TYPE` | -- | A-side work_type filter. Replaces the shared `--work-type` for the A side only. |
| `--a-commit HASH` (repeatable) | -- | A-side commit filter. Replaces the shared `--commit` for the A side only. |
| `--a-flag NAME` (repeatable) | -- | A-side flag filter. Replaces the shared `--flag` for the A side only. |
| `--b-kernel VER` (repeatable) | -- | B-side kernel filter. Replaces the shared `--kernel` for the B side only. |
| `--b-scheduler NAME` | -- | B-side scheduler filter. Replaces the shared `--scheduler` for the B side only. |
| `--b-topology LABEL` | -- | B-side topology filter. Replaces the shared `--topology` for the B side only. |
| `--b-work-type TYPE` | -- | B-side work_type filter. Replaces the shared `--work-type` for the B side only. |
| `--b-commit HASH` (repeatable) | -- | B-side commit filter. Replaces the shared `--commit` for the B side only. |
| `--b-flag NAME` (repeatable) | -- | B-side flag filter. Replaces the shared `--flag` for the B side only. |
| `--no-average` | off | Disable averaging. Each sidecar stays distinct; bails with an actionable error when multiple sidecars on the same side share the same pairing key (since pairing across sides becomes ambiguous). |
| `--threshold PCT` | per-metric `default_rel` | Uniform relative significance threshold in percent. Overrides the per-metric `default_rel` for every metric; the absolute gate is always per-metric and cannot be tuned from the CLI. Mutually exclusive with `--policy`. |
| `--policy FILE` | -- | Path to a JSON `ComparisonPolicy` file with per-metric thresholds. Schema: `{ "default_percent": N, "per_metric_percent": { "worst_spread": 5.0, ... } }`. Priority is per-metric override → `default_percent` → each metric's registry `default_rel`. Per-metric keys are rejected at load time if they do not match a metric in the `METRICS` registry. Mutually exclusive with `--threshold`. |
| `--dir DIR` | `target/ktstr/` | Alternate runs root for pool collection. Defaults to `test_support::runs_root()` (typically `target/ktstr/`). Useful when comparing archived sidecar trees copied off a CI host. |

### Prerequisites

Run tests first to generate sidecar JSON files:

```sh
cargo nextest run --workspace        # generates target/ktstr/{kernel}-{timestamp}/*.json
cargo ktstr stats                    # reads the newest run
```

Set `KTSTR_SIDECAR_DIR` to override the sidecar directory; otherwise
the default is `{CARGO_TARGET_DIR or "target"}/ktstr/{kernel}-{timestamp}/`.

## Install

```sh
cargo install --locked ktstr          # installs both ktstr and cargo-ktstr
```

Or build from the workspace:

```sh
cargo build --bin cargo-ktstr
```
