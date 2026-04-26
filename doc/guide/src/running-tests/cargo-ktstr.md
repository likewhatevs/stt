# cargo-ktstr

`cargo ktstr` is a cargo plugin for kernel build, cache, and test
workflow. Subcommands in `--help` order: `test` (alias: `nextest`),
`coverage`, `llvm-cov`, `stats`, `kernel`, `model`, `verifier`,
`completions`, `show-host`, `show-thresholds`, `cleanup`, `locks`,
`shell`.

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
- Local cache entry → `kernel_local_{hash6}` (first 6 chars of
  the source tree's git short_hash, captured at cache-store
  time) or `kernel_local_unknown` for non-git trees. The
  hash6 keeps two distinct local trees from collapsing to the
  same label; the `unknown` literal is the shared bucket for
  every non-git tree (no discriminator exists at the cache
  layer to spread them apart).

Filter with nextest's `-E 'test(kernel_6_14)'` to pick a single
kernel from a multi-kernel matrix; nextest's parallelism, retries,
and `--ignored` flag all apply natively. Sidecars partition per
kernel: each kernel runs in its own
`target/ktstr/{kernel}-{project_commit}/` directory keyed on the
resolved kernel's identity and the project tree's HEAD short hex
(with `-dirty` suffix when the worktree differs). Coverage profraw does NOT partition
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
skip the kernel suffix and list / run **once** regardless of
`KTSTR_KERNEL_LIST` cardinality. The dispatch sites
(`list_tests`, `list_tests_budget`, and `--exact`'s
`run_host_only_test` in `src/test_support/dispatch.rs`) all gate
on `entry.host_only` before consulting the resolved kernel set,
so a host-side test never observes the kernel directory and
multiplying it across kernels would just run N copies of
identical work for no signal.

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
`target/ktstr/{kernel}-{project_commit}/`, where `{project_commit}`
is the project HEAD short hex (with `-dirty` when the worktree
differs). Coverage profraw lands flat in
`target/llvm-cov-target/` with PID-keyed filenames — it does
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

List cached kernel images, sorted newest first. With `--range`,
switches to PREVIEW MODE: prints the versions a `START..END` range
expands to without performing any download or build.

```sh
cargo ktstr kernel list
cargo ktstr kernel list --json                    # JSON output for CI scripting
cargo ktstr kernel list --range 6.12..6.14        # preview range expansion
cargo ktstr kernel list --range 6.12..6.14 --json # preview as JSON
```

Default mode walks the local cache. Human-readable output shows
key, version, source type, arch, and build timestamp. Entries built
with a different `ktstr.kconfig` are marked `(stale kconfig)`.
Entries whose major.minor version is no longer in kernel.org's
active releases list are marked `(EOL)`; prefix lookups for EOL
series fall back to probing cdn.kernel.org for the latest patch
release.

`--range` mode performs no cache reads: it fetches kernel.org's
`releases.json` once, expands the inclusive range against the
`stable` and `longterm` releases (mainline / linux-next dropped),
and prints one version per line on stdout. Use this to answer
"what does `--kernel 6.12..6.16` actually cover?" before paying
the build cost — no kernel is downloaded or compiled. With
`--json`, emits a JSON object carrying the literal range, the
parsed start / end, and the expanded `versions` array.

| Flag | Description |
|------|-------------|
| `--json` | Output in JSON format. Each entry includes a boolean `eol` field (computed at list time by fetching kernel.org's `releases.json`) alongside the cached metadata. With `--range`, emits a single object `{range, start, end, versions}` instead. |
| `--range START..END` | Switch to range-preview mode. Format: `MAJOR.MINOR[.PATCH][-rcN]..MAJOR.MINOR[.PATCH][-rcN]`. Performs the single `releases.json` fetch a real range resolve does, expands inclusively, and prints the version list — no downloads, no builds, no cache lookups. |

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
| `--cpu-cap N` | Reserve exactly N host CPUs for the build (integer ≥ 1; must be ≤ the calling process's `sched_getaffinity` cpuset size). When absent, 30% of the allowed CPUs are reserved (minimum 1). The planner walks whole LLCs in consolidation- and NUMA-aware order, partial-taking the last LLC so `plan.cpus.len() == N` exactly. Under `--cpu-cap`, `make -jN` parallelism matches the reserved CPU count and the build runs inside a cgroup v2 sandbox that pins gcc/ld to the reserved CPUs + NUMA nodes. Mutually exclusive with `KTSTR_BYPASS_LLC_LOCKS=1`. Also settable via `KTSTR_CPU_CAP` env var (CLI flag wins when both are present). |

### kernel clean

Remove cached kernel images.

```sh
cargo ktstr kernel clean                          # remove all (with confirmation prompt)
cargo ktstr kernel clean --keep 3                 # keep 3 most recent
cargo ktstr kernel clean --force                  # skip confirmation prompt
cargo ktstr kernel clean --corrupt-only --force   # remove only corrupt entries
```

| Flag | Description |
|------|-------------|
| `--keep N` | Keep the N most recent VALID cached kernels. Corrupt entries (metadata missing or unparseable, image file absent) are always candidates for removal regardless of this value — a corrupt entry never consumes a keep slot. Mutually exclusive with `--corrupt-only`. |
| `--force` | Skip confirmation prompt. Required in non-interactive contexts. |
| `--corrupt-only` | Remove only corrupt cache entries (metadata missing or unparseable, image file absent). Valid entries are left untouched regardless of `--force`. Useful for clearing broken entries after an interrupted build without risking the curated set of good kernels. Mutually exclusive with `--keep`. |

## model

Manage the LLM model cache used by `OutputFormat::LlmExtract`
payloads. `fetch` downloads the default pinned model into the
ktstr model cache; `status` reports whether a SHA-checked copy
is already cached; `clean` deletes the cached artifact plus
its warm-cache sidecar.

```sh
cargo ktstr model fetch                          # download + SHA-check (no-op if cached)
cargo ktstr model status                         # report cache path + verdict
cargo ktstr model clean                          # delete cached artifact + sidecar
```

`fetch` is a no-op when the cache already holds a SHA-checked
copy. Respects `KTSTR_MODEL_OFFLINE=1` — set to refuse network
fetches. Cache root resolution: `KTSTR_CACHE_DIR` (if set),
then `$XDG_CACHE_HOME/ktstr/models/`, then
`$HOME/.cache/ktstr/models/`.

`status` prints four fields and adds a one-line annotation
when the verdict is anything other than `Matches` (a clean
hit gets no annotation):

| Field | Description |
|---|---|
| `model:` | Model file name (the pinned default; e.g. `Qwen3-4B-Q4_K_M.gguf`). |
| `path:` | Absolute cache path (`{cache_root}/models/{file}`) the producer reads at LlmExtract time. |
| `cached:` | `true` if an entry exists at `path:`, `false` otherwise. |
| `checked:` | `true` if the cached entry's SHA-256 matches the pinned digest. |

The annotation distinguishes four verdicts: `NotCached` (no
entry — emit a `cargo ktstr model fetch` hint plus the
expected download size), `CheckFailed` (cached entry could
not be SHA-checked due to an I/O error — re-fetch),
`Mismatches` (cached entry hash does not match the pinned
digest — re-fetch), `Matches` (silent — the all-clear path).
Re-fetch is the shared remediation tail for every cached-but-
not-Matches branch.

`clean` removes both the GGUF artifact at
`{cache_root}/models/{file_name}` and its `.mtime-size`
warm-cache sidecar (a small companion file the SHA fast-path
uses to skip re-hashing on subsequent `status` calls). Per-
file output names what was deleted with an IEC-prefixed size
in parentheses (`removed /path/to/Qwen3-4B-Q4_K_M.gguf (2.34
GiB)`); a final `freed N total` line sums the artifact and
sidecar bytes. A no-op clean (nothing cached) prints a single
`no cached model found at {path}` line so an idempotent re-run
produces a clear "nothing to do" outcome instead of two
"(absent)" lines. Subsequent `cargo ktstr model fetch`
re-downloads the pin from scratch.

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
| `--kernel ID` | Kernel identifier: path, version, cache key, range (`START..END`), or git source (`git+URL#REF`). Raw image files (`bzImage`/`Image`) are NOT accepted by `verifier` — the verifier needs the cached `vmlinux` and kconfig fragment alongside the image. Source directories auto-build; version strings auto-download on cache miss. When absent, resolves via cache then filesystem, falling back to auto-download. Raw images are accepted only on `cargo ktstr shell`. |
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

Sidecar analysis, per-record diagnostics, and run-to-run comparison.
See [Runs](runs.md) for the directory layout.

```sh
cargo ktstr stats                                             # print analysis of newest run
cargo ktstr stats list                                        # list runs
cargo ktstr stats list-metrics                                # list registered regression metrics
cargo ktstr stats compare --a-kernel 6.14 --b-kernel 6.15     # slice on kernel
cargo ktstr stats compare --a-scheduler scx_rusty --b-scheduler scx_alpha  # slice on scheduler
cargo ktstr stats compare --a-kernel 6.14 --b-kernel 6.15 --scheduler scx_rusty  # slice on kernel, pin scheduler on both sides
cargo ktstr stats compare --a-kernel 6.14 --b-kernel 6.15 -E cgroup_steady       # add substring filter
cargo ktstr stats compare --a-project-commit abcdef1 --b-project-commit fedcba2 --no-average  # opt out of trial averaging
cargo ktstr stats compare --a-kernel-commit abcdef1 --b-kernel-commit fedcba2    # slice on kernel source HEAD
cargo ktstr stats compare --a-run-source ci --b-run-source local                 # slice on run environment
cargo ktstr stats explain-sidecar --run RUN_ID                                   # diagnose Option-field absences
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
`{CARGO_TARGET_DIR or "target"}/ktstr/` with four columns:

- `RUN`: the run-directory leaf name, formatted as
  `{kernel}-{project_commit}` per [Runs](runs.md). `list` does NOT
  consult `KTSTR_SIDECAR_DIR` — that override only affects where
  the test harness writes sidecars; `list` always enumerates the
  default runs-root.
- `TESTS`: number of sidecars in the directory (and one level of
  subdirectories — `collect_sidecars` walks per-job gauntlet
  layouts).
- `DATE`: the earliest sidecar timestamp present in the directory
  — under last-writer-wins this equals the most recent run's
  first sidecar timestamp (the prior run's sidecars were
  pre-cleared at the new run's first write, so only the new
  run's timestamps remain). See [Runs](runs.md) for the full
  semantics.
- `ARCH`: the `host.arch` value from the run's first sidecar
  (e.g. `x86_64`, `aarch64`). Renders as `-` when no sidecar in
  the directory carries a populated host context — pre-host-
  context archives and host-only test stubs that never populate
  the field land in this bucket.

Rows are sorted by directory mtime, **most recent first**, so the
latest run lands at the top — the operator's usual interest.
Entries whose mtime cannot be read fall back to filename order as
a deterministic tiebreaker and sort to the end of the listing.

### list-metrics

List the registered regression metrics and their default
thresholds. Enumerates the `ktstr::stats::METRICS` registry: metric
name, polarity (higher/lower better), default absolute-delta gate,
default relative-delta gate, and display unit. Use this to see which metric names
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

### list-values

List the distinct values present per filterable dimension in the
sidecar pool. Walks every run directory under `target/ktstr/`
(or `--dir`), pools the sidecars, and reports per-dimension sets
for all eight dimensions: `kernel`, `commit`, `kernel_commit`,
`source`, `scheduler`, `topology`, `work_type`, and `flags`
(individual flag names, exploded from each row's
`active_flags`). The `commit` and `source` keys map to the
internal `SidecarResult::project_commit` / `run_source` fields;
the JSON wire keys keep the shorter spellings.

Use this before crafting a `cargo ktstr stats compare`
invocation to discover what `--a-X` / `--b-X` values the pool
actually carries: `--a-kernel 6.20` against an empty pool fails
downstream with "no rows match filter A", and `list-values` is
the upstream answer to "what kernels do I have?".

```sh
cargo ktstr stats list-values                       # text per-dim blocks
cargo ktstr stats list-values --json                # JSON object
cargo ktstr stats list-values --dir /tmp/archived   # archived sidecar tree
```

The text shape renders one block per dimension with values one
per line. The JSON shape emits a single object keyed by
dimension name with arrays of values:

```json
{
  "kernel": [null, "6.14.2", "6.15.0"],
  "commit": [null, "abcdef1", "abcdef1-dirty"],
  "kernel_commit": [null, "kabcde7", "kabcde7-dirty"],
  "source": [null, "ci", "local"],
  "scheduler": ["eevdf", "scx_rusty"],
  "topology": ["1n2l4c1t", "1n4l2c1t"],
  "work_type": ["CpuSpin", "PageFaultChurn"],
  "flags": ["llc", "rusty_balance"]
}
```

The JSON keys `commit` and `source` are the wire contract;
internally the corresponding fields are
`SidecarResult::project_commit` and `SidecarResult::run_source`,
and the per-side filter flags spell as `--project-commit` /
`--run-source` (see [`compare`](#compare)).

`kernel`, `commit`, `kernel_commit`, and `source` are optional
on the source sidecar (`SidecarResult::kernel_version` /
`project_commit` / `kernel_commit` / `run_source` are
`Option<String>`); the textual sentinel `unknown` and JSON
`null` both denote a sidecar that did not record a value for
that dimension.

| Flag | Default | Description |
|------|---------|-------------|
| `--json` | off | Emit JSON instead of per-dimension text blocks. |
| `--dir DIR` | `target/ktstr/` | Alternate run root. Same semantics as `compare --dir`. |

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
| `--run ID` | required | Run key (e.g. `6.14-abc1234` or `6.14-abc1234-dirty`; from `cargo ktstr stats list`). |
| `--dir DIR` | `target/ktstr/` | Alternate run root. Same semantics as `compare --dir`: useful for archived sidecar trees copied off a CI host. |

### explain-sidecar

Diagnose `Option`-field absences across a run's sidecars. Loads
every `*.ktstr.json` under `--run ID` (or its subdirectories one
level deep, mirroring `compare`'s gauntlet-job layout) and reports,
per sidecar, which `Option<T>` fields landed as `None` plus the
documented causes for each absence and a classification:

- `expected` — `None` is the steady-state shape; no operator
  action recovers it (e.g. `payload` for a scheduler-only test,
  `scheduler_commit` which no `SchedulerSpec` variant exposes
  today).
- `actionable` — `None` indicates a recoverable gap; re-running
  in a different environment (in-repo cwd, non-tarball kernel,
  non-host-only test) would populate the field.

Different gauntlet variants on the same run legitimately differ
on which fields populate (host-only vs VM-backed,
scheduler-only vs payload-bearing), so the report is per-sidecar
rather than aggregate.

Sidecars are loaded verbatim — this command does NOT rewrite
`run_source` to `"archive"` even when `--dir` is set. Diverges
intentionally from `compare` / `list-values`; matches `show-host`.
The override would erase the only signal that surfaces the
pre-rename `source`-key drop case.

The output header reports `walked N sidecar file(s), parsed M valid`: `N` counts every
`.ktstr.json` file the walker visited, `M` counts how many
parsed against the current schema. `walked > parsed` signals a
corrupt or pre-1.0-schema sidecar — re-run the test to
regenerate under the current schema.

Per-`None` blocks in the text output also include a `fix:`
line for fields whose `None` is recoverable by an operator
action (e.g. `kernel_commit` recovers when `KTSTR_KERNEL`
points at a local kernel git tree). Fields whose `None` is
the steady-state shape (or a multi-cause set with no single
remediation) emit no `fix:` line.

When the walk encounters parse failures, the text output
appends a trailing `corrupt sidecars (N):` block listing
each corrupt path on its own line followed by the serde
error message indented as `error: ...`, optionally
followed by an `enriched: ...` line with operator-facing
remediation prose when the parse failure matches a known
schema-drift case (currently the `host` missing-field
case). When the walk encounters IO failures (file matched
the predicate but `read_to_string` failed before parsing
could begin — permission denied, mid-rotate truncation,
broken symlink, EISDIR), the text output appends a parallel
`io errors (N):` block, structured the same way (path on
its own line, `error: ...` line below) but carrying
`std::io::Error::Display` rather than serde-error text. IO
errors do NOT carry `enriched:` lines — there is no
schema-drift catalog for filesystem incidents; the raw
`std::io::Error` Display is the remediation surface.
Each block is suppressed independently when its source
vec is empty.

All-corrupt and all-IO-failure runs (every predicate-
matching file failed to parse, or every one failed to
read) are NOT a hard error — text output renders the
header (`walked N sidecar file(s), parsed 0 valid`)
followed directly by the `corrupt sidecars (N):` and/or
`io errors (N):` block(s), skipping the per-sidecar
breakdown that has nothing to render. JSON output mirrors
this with `valid: 0`, `_walk.errors` and/or
`_walk.io_errors` populated, and per-field counts at zero.
This preserves structured per-file visibility for
dashboard consumers facing total-failure runs of either
class.

All-corrupt and all-IO-failure runs exit 0 (not a hard
error); CI scripts must inspect the JSON channel for
failure detection rather than relying on exit code. Two
common gating policies, each appropriate for different
operational stances:

- **Lenient** (treat partial failures as warnings):
  `_walk.valid > 0`. Accepts any run with at least one
  successfully-parsed sidecar; per-file parse or IO
  failures surface in the JSON arrays for triage but do
  not fail the gate.
- **Strict** (fail on any sidecar failure):
  `_walk.errors.len() == 0 && _walk.io_errors.len() == 0`.
  Requires every predicate-matching file to parse cleanly.
  Both checks are required because the two arrays cover
  disjoint failure classes (parse vs read) — a run with
  zero parse errors but one IO error still has a missing
  sidecar.

The two policies are NOT equivalent: a run with one valid
and one corrupt sidecar passes lenient (`valid == 1 > 0`)
but fails strict (`errors.len() == 1 > 0`). Pick the
policy that matches the operational tolerance for partial
data.

`--json` emits a single object with three top-level keys:
`_schema_version` (a string version stamp — currently
`"1"` — that consumers can gate on for incompatible shape
changes), `_walk` (an envelope carrying `walked` / `valid`
counts — same numbers the text header reports under "walked
N sidecar file(s), parsed M valid" — plus an `errors` array
of `{path, error, enriched_message}` entries covering every
parse failure (`enriched_message` is a human-facing
remediation string when a known schema-drift case matches,
JSON null otherwise) AND an `io_errors` array of
`{path, error}` entries covering every IO failure (file
matched the predicate but `read_to_string` failed; `error`
carries the raw `std::io::Error` Display). Both arrays
emit on every render — empty array when no failures of
that class occurred — so dashboard consumers see a uniform
shape without `contains_key` branching. With both arrays,
`walked == valid + errors.len() + io_errors.len()` by
construction in the steady state — every predicate-matching
file lands in exactly one bucket. (Filesystem races between
the count and load passes can perturb this; see the rustdoc
on `WalkStats` for the full caveat.) Then `fields`. Each
entry under `fields` carries `none_count` and `some_count`
(counts across all valid sidecars in the run, summing to
`_walk.valid`), `classification`, `causes`, and `fix`
(string when a remediation applies, JSON null otherwise).

Output produced before the schema-version stamp landed has
no `_schema_version` key; consumers should treat the key's
absence as pre-stamp output (compatible with shape `"1"` in
practice but unstamped).

The version bumps on incompatible shape changes (key
rename, key removal, semantic shift in an existing key) but
NOT on additive changes (new optional top-level keys, new
entries in `fields`, new optional sub-keys under existing
entries). The stamp is emitted as a JSON string (e.g. `"1"`,
`"2"`); parse it by stripping the quotes and converting the
inner digits to an integer, then gate on `parsed >= 1`
(integer comparison) — never use raw string comparison, since
lexicographic order would put `"10"` ahead of `"2"`. Pin
loosely (e.g. accept any version `>= 1`) so dashboard code
keeps working when the catalog grows; tighten only on the
specific bumps a consumer cannot tolerate.

```sh
cargo ktstr stats explain-sidecar --run RUN_ID                       # text per-sidecar diagnostic
cargo ktstr stats explain-sidecar --run RUN_ID --json                 # aggregate JSON for dashboards
cargo ktstr stats explain-sidecar --run RUN_ID --dir /path/archive    # diagnose archived sidecars
```

| Flag | Default | Description |
|------|---------|-------------|
| `--run ID` | required | Run key (e.g. `6.14-abc1234` or `6.14-abc1234-dirty`; from `cargo ktstr stats list`). |
| `--dir DIR` | `target/ktstr/` | Alternate run root. Same semantics as `compare --dir`. |
| `--json` | off | Emit aggregate JSON instead of per-sidecar text. |

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

# Slice on project commit, narrow both sides to one scheduler+kernel.
cargo ktstr stats compare \
    --a-project-commit abcdef1 --b-project-commit fedcba2 \
    --kernel 6.14 --scheduler scx_rusty

# Slice on run environment: CI runs vs local developer runs.
cargo ktstr stats compare \
    --a-run-source ci --b-run-source local
```

**Symmetric sugar.** Shared `--X` flags (`--kernel`, `--scheduler`,
`--topology`, `--work-type`, `--project-commit`, `--kernel-commit`,
`--run-source`, `--flag`) pin BOTH sides to the same value(s).
Per-side `--a-X` / `--b-X` flags REPLACE the corresponding
shared `--X` value for that side only — "more-specific replaces"
semantics. So `--kernel 6.14 --a-kernel 6.13` puts A on 6.13
and B on 6.14. Together the eight slicing dimensions
(`kernel`, `scheduler`, `topology`, `work-type`,
`project-commit`, `kernel-commit`, `run-source`, `flags`) cover
every typed axis the comparison can contrast on.

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

**`+mixed` commit marker.** When contributors to an averaged
group disagree on the `-dirty` suffix for the same canonical
hex (some clean, some `-dirty`), the rendered `commit` and
`kernel_commit` columns show `{hex}+mixed` for that group.
`+mixed` is a COHORT-level marker (distinct from `-dirty`,
which is a per-record property of one sidecar): it indicates
mixed working-tree state across the group's contributors.
Mixed-dirty tracking spans EVERY contributor (passing,
failing, skipped) so WIP-vs-committed disagreement surfaces
in the averaged row even when one of the two states only
appears on a failing run. The marker is rendered against the
canonical un-suffixed hex, so a `abc1234` clean entry plus an
`abc1234-dirty` entry render as `abc1234+mixed` regardless of
which contributor was scanned first. Homogeneous cohorts
(every contributor clean, every contributor dirty, or every
contributor `None`) preserve the first-seen value verbatim
and never get the `+mixed` marker.

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

**Discovering filter values.** Run
[`cargo ktstr stats list-values`](#list-values) before
crafting a `compare` invocation to see what `kernel`, `commit`,
`kernel_commit`, `source`, `scheduler`, `topology`, `work_type`,
and `flags` values the sidecar pool actually carries; passing a
`--a-kernel 6.20` against an empty pool fails downstream with
"no rows match filter A" and `list-values` is the upstream
answer to "what have I got?". `list-values` reports all eight
filterable dimensions; the JSON keys `commit` and `source` map
to the per-side filter flags `--project-commit` and
`--run-source`.

When a side comes back as `unknown` for one of the optional
dimensions (`kernel`, `commit`, `kernel_commit`, `source`),
[`cargo ktstr stats explain-sidecar`](#explain-sidecar) on the
underlying run reports per-sidecar which optional fields are
missing and what each absence means.

| Flag | Default | Description |
|------|---------|-------------|
| `-E FILTER` | -- | Substring filter applied to the joined `scenario topology scheduler work_type flags` string. **Scope is limited**: `-E` does NOT match against `kernel`, `project_commit`, `kernel_commit`, or `run_source` — those are typed dimensions reachable only via the dedicated `--kernel` / `--project-commit` / `--kernel-commit` / `--run-source` flags. To narrow on those, use the typed flags. Composes with the typed dimension filters: typed narrows happen first, substring runs over the surviving set. |
| `--kernel VER` (repeatable) | -- | Pin BOTH sides to the listed kernel version(s). Sugar for `--a-kernel V1 --a-kernel V2 --b-kernel V1 --b-kernel V2`. Per-side `--a-kernel` / `--b-kernel` REPLACES this shared value for that side only. Major.minor (`6.12`) prefix-matches; three-segment (`6.14.2`) is strict. |
| `--scheduler NAME` (repeatable) | -- | Pin BOTH sides to the listed scheduler(s). Sugar for `--a-scheduler N1 --a-scheduler N2 --b-scheduler N1 --b-scheduler N2`. Per-side `--a-scheduler` / `--b-scheduler` REPLACES this shared value for that side only. OR-combined: a row matches iff its `scheduler` field equals ANY listed entry. Strict equality per entry. |
| `--topology LABEL` (repeatable) | -- | Pin BOTH sides to the listed rendered topology label(s) (e.g. `1n2l4c2t`). Sugar for `--a-topology L1 --a-topology L2 --b-topology L1 --b-topology L2`. Per-side `--a-topology` / `--b-topology` REPLACES this shared value for that side only. OR-combined: a row matches iff its rendered topology label equals ANY listed entry. Strict equality per entry. |
| `--work-type TYPE` (repeatable) | -- | Pin BOTH sides to the listed work_type(s) (PascalCase variants of `WorkType`, e.g. `CpuSpin`). Sugar for `--a-work-type T1 --a-work-type T2 --b-work-type T1 --b-work-type T2`. Per-side `--a-work-type` / `--b-work-type` REPLACES this shared value for that side only. OR-combined: a row matches iff its `work_type` field equals ANY listed entry. Strict equality per entry. See [Work types](../concepts/work-types.md). |
| `--project-commit HASH` (repeatable) | -- | Pin BOTH sides to listed `project_commit` value(s) (7-char hex, optional `-dirty` suffix). Also accepts git revspecs (`HEAD`, `HEAD~N`, tags, branches, `A..B` ranges) resolved against the project repo into the same 7-char short hashes; see `--help` for details. Filters the ktstr framework commit; the scheduler binary's commit (`SidecarResult::scheduler_commit`) is not currently exposed as a filter. Renamed from `--commit` for naming symmetry with `--kernel-commit`. |
| `--kernel-commit HASH` (repeatable) | -- | Pin BOTH sides to listed `kernel_commit` value(s) (7-char hex, optional `-dirty` suffix). Also accepts git revspecs (`HEAD`, `HEAD~N`, tags, branches, `A..B` ranges) resolved against the kernel repo (`gix::open` against `KTSTR_KERNEL`'s path); see `--help` for details. Filters the kernel SOURCE TREE commit (`SidecarResult::kernel_commit`), distinct from the kernel release version (`--kernel`): two runs of the same `kernel_version` with different `kernel_commit` values represent the same release rebuilt from different trees. Rows whose `kernel_commit` is `None` (KTSTR_KERNEL pointed at a non-git path, the underlying source was Tarball / Git rather than a `Local` tree, or the gix probe failed) NEVER match a populated filter. |
| `--run-source NAME` (repeatable) | -- | Pin BOTH sides to listed run-environment source(s). Filters `SidecarResult::run_source` set by `detect_run_source` at sidecar-write time: `"local"` for developer runs, `"ci"` when `KTSTR_CI` was set, or rewritten to `"archive"` at load time when `--dir` points at a non-default pool root. Rows whose `run_source` is `None` (sidecar pre-dates the field) NEVER match a populated filter — same opt-in policy as `--kernel` / `--project-commit` / `--kernel-commit`. Combine per-side `--a-run-source ci --b-run-source local` to contrast CI runs against developer runs of the same scenarios. |
| `--flag NAME` | -- | Repeatable AND-combined flag filter pinning BOTH sides. Every listed flag must appear in `active_flags`; rows may carry additional flags. |
| `--a-kernel VER` (repeatable) | -- | A-side kernel filter. Replaces the shared `--kernel` for the A side only. |
| `--a-scheduler NAME` (repeatable) | -- | A-side scheduler filter, OR-combined. Replaces the shared `--scheduler` value for the A side only. |
| `--a-topology LABEL` (repeatable) | -- | A-side topology filter, OR-combined. Replaces the shared `--topology` value for the A side only. |
| `--a-work-type TYPE` (repeatable) | -- | A-side work_type filter, OR-combined. Replaces the shared `--work-type` value for the A side only. |
| `--a-project-commit HASH` (repeatable) | -- | A-side project-commit filter. Replaces the shared `--project-commit` for the A side only. |
| `--a-kernel-commit HASH` (repeatable) | -- | A-side kernel-commit filter. Replaces the shared `--kernel-commit` for the A side only. |
| `--a-run-source NAME` (repeatable) | -- | A-side run-source filter. Replaces the shared `--run-source` for the A side only. |
| `--a-flag NAME` (repeatable) | -- | A-side flag filter. Replaces the shared `--flag` for the A side only. |
| `--b-kernel VER` (repeatable) | -- | B-side kernel filter. Replaces the shared `--kernel` for the B side only. |
| `--b-scheduler NAME` (repeatable) | -- | B-side scheduler filter, OR-combined. Replaces the shared `--scheduler` value for the B side only. |
| `--b-topology LABEL` (repeatable) | -- | B-side topology filter, OR-combined. Replaces the shared `--topology` value for the B side only. |
| `--b-work-type TYPE` (repeatable) | -- | B-side work_type filter, OR-combined. Replaces the shared `--work-type` value for the B side only. |
| `--b-project-commit HASH` (repeatable) | -- | B-side project-commit filter. Replaces the shared `--project-commit` for the B side only. |
| `--b-kernel-commit HASH` (repeatable) | -- | B-side kernel-commit filter. Replaces the shared `--kernel-commit` for the B side only. |
| `--b-run-source NAME` (repeatable) | -- | B-side run-source filter. Replaces the shared `--run-source` for the B side only. |
| `--b-flag NAME` (repeatable) | -- | B-side flag filter. Replaces the shared `--flag` for the B side only. |
| `--no-average` | off | Disable averaging. Each sidecar stays distinct; bails with an actionable error when multiple sidecars on the same side share the same pairing key (since pairing across sides becomes ambiguous). |
| `--threshold PCT` | per-metric `default_rel` | Uniform relative significance threshold in percent. Overrides the per-metric `default_rel` for every metric; the absolute gate is always per-metric and cannot be tuned from the CLI. Mutually exclusive with `--policy`. |
| `--policy FILE` | -- | Path to a JSON `ComparisonPolicy` file with per-metric thresholds. Schema: `{ "default_percent": N, "per_metric_percent": { "worst_spread": 5.0, ... } }`. Priority is per-metric override → `default_percent` → each metric's registry `default_rel`. Per-metric keys are rejected at load time if they do not match a metric in the `METRICS` registry. Mutually exclusive with `--threshold`. |
| `--dir DIR` | `target/ktstr/` | Alternate runs root for pool collection. Defaults to `test_support::runs_root()` (typically `target/ktstr/`). Useful when comparing archived sidecar trees copied off a CI host. |

### Prerequisites

Run tests first to generate sidecar JSON files:

```sh
cargo nextest run --workspace        # generates target/ktstr/{kernel}-{project_commit}/*.json
cargo ktstr stats                    # reads the newest run
```

Set `KTSTR_SIDECAR_DIR` to override the sidecar directory; otherwise
the default is `{CARGO_TARGET_DIR or "target"}/ktstr/{kernel}-{project_commit}/`,
where `{project_commit}` is the project HEAD short hex (with `-dirty`
when the worktree differs).

## show-host

Print the **live** host context used by the sidecar collector:
CPU identity, memory/hugepage config, transparent-hugepage
policy, NUMA node count, kernel uname triple
(sysname / release / machine), kernel cmdline, and every
`/proc/sys/kernel/sched_*` tunable. Useful for diagnosing
cross-run regressions that trace back to host-context drift
(sysctl change, THP policy flip, hugepage reservation) or for
confirming what `cargo ktstr stats compare` would record on
the next run produced here.

```sh
cargo ktstr show-host
```

This is a **live** snapshot (reads `/proc`, `/sys`, and
`uname()` at invocation time). For the **archived** host
context captured at sidecar-write time for a past run, use
[`cargo ktstr stats show-host --run RUN_ID`](#show-host)
instead — same `HostContext::format_human` formatter so the
two outputs are byte-for-byte comparable when the host is
unchanged.

For historical drift between archived runs (host-side diff
across two run partitions), use
[`cargo ktstr stats compare`](#compare) — its host-delta
section reports which host-context fields changed between
side A and side B using the same `HostContext::diff` logic.

## show-thresholds

Print the resolved assertion thresholds for the named test —
the same merged `Assert` value `run_ktstr_test_inner` evaluates
against worker reports, produced by the runtime merge chain
`Assert::default_checks().merge(entry.scheduler.assert()).merge(&entry.assert)`.
Surfaces every threshold field (or `none` when inherited or
unset) so an operator can see what the test will actually
check against without reading source or guessing which layer
contributed each bound.

```sh
cargo ktstr show-thresholds preempt_regression_fault_under_load
```

| Arg | Description |
|------|-------------|
| `TEST` | Function-name-only test identifier as registered in `#[ktstr_test]` (e.g. `preempt_regression_fault_under_load`). Use `cargo nextest list` to enumerate test names — then strip the `<binary>::` prefix that nextest prepends to each line before passing the name here. The `#[ktstr_test]` registry keys on the bare function name, so a name like `ktstr::my_test` (as printed by nextest) must be trimmed to `my_test` before it resolves. |

Fails with an actionable message when no registered test
matches the given name; the diagnostic includes a `Did you
mean ...?` Levenshtein suggestion when a near match exists.

## cleanup

Clean up leftover ktstr cgroups. With no flags, scans
`/sys/fs/cgroup` for the default ktstr parents — `ktstr/`
(used by the in-process test harness) and every
`ktstr-<pid>/` left behind by `ktstr run` instances — and
rmdirs each. `ktstr-<pid>` directories whose `<pid>` still
owns a live `ktstr` or `cargo-ktstr` process are skipped, so
a concurrent cleanup run does not yank an active run's
cgroup; each skip emits a `ktstr: skipping <path> (live
process)` line on stderr.

```sh
cargo ktstr cleanup                               # scan defaults
cargo ktstr cleanup --parent-cgroup /sys/fs/cgroup/ktstr-12345  # explicit path
```

| Flag | Description |
|------|-------------|
| `--parent-cgroup PATH` | Clean only this explicit path and leave the parent directory in place. No live-process check is performed. When omitted, the default scan path runs. |

## locks

Enumerate every ktstr flock held on this host — read-only,
does NOT attempt any flock acquire. Troubleshooting companion
for `--cpu-cap` contention: when a build or test is stalled
behind a peer's reservation, `cargo ktstr locks` names the
peer (PID + cmdline) without disturbing any of its flocks.

Scans four lock-file roots:

- `/tmp/ktstr-llc-*.lock` — per-LLC reservations held by
  perf-mode test runs and `--cpu-cap`-bounded builds.
- `/tmp/ktstr-cpu-*.lock` — per-CPU reservations from the
  same flow.
- `{cache_root}/.locks/*.lock` — cache-entry locks held
  during `kernel build` writes.
- `{runs_root}/.locks/{kernel}-{project_commit}.lock` —
  per-run-key sidecar-write locks held for the duration of
  the (pre-clear + write) cycle to serialize concurrent
  ktstr processes targeting the same run directory.

Each lock is cross-referenced against `/proc/locks` to name
the holder PID and cmdline.

```sh
cargo ktstr locks                       # one-shot snapshot
cargo ktstr locks --json                # JSON snapshot
cargo ktstr locks --watch 1s            # redraw every second until SIGINT
cargo ktstr locks --watch 1s --json     # ndjson stream, one object per interval
```

| Flag | Default | Description |
|------|---------|-------------|
| `--json` | off | Emit the snapshot as JSON. Pretty-printed in one-shot mode; compact (one object per line, ndjson-style) under `--watch`. Stable field names — schema documented on `ktstr::cli::list_locks`. |
| `--watch DURATION` | unset | Redraw the snapshot at the given interval until SIGINT. Value is parsed by `humantime`: `100ms`, `1s`, `5m`, `1h`. Human output clears and redraws in place; `--json` emits one line-terminated object per interval. |

The same subcommand is available as
[`ktstr locks`](ktstr.md#locks) with identical flag
semantics.

## Install

```sh
cargo install --locked ktstr          # installs both ktstr and cargo-ktstr
```

Or build from the workspace:

```sh
cargo build --bin cargo-ktstr
```
