# CI

Recipes for running ktstr tests in continuous integration.

## Runner requirements

ktstr boots KVM virtual machines. CI runners must provide:

- `/dev/kvm` access (hardware virtualization enabled)
- Self-hosted runners or a provider that exposes KVM to the guest

GitHub-hosted `ubuntu-latest` runners do **not** expose `/dev/kvm`.
Use self-hosted runners with KVM labels:

```yaml
runs-on: [self-hosted, X64]                              # x86_64 (minimum labels)
runs-on: [self-hosted, Linux, kvm, kernel-build, ARM64]  # aarch64 (adjust labels to your runner pool)
```

See [Troubleshooting: /dev/kvm not accessible](troubleshooting.md#devkvm-not-accessible)
for diagnosing KVM issues on runners.

Runners also need the build dependencies listed in
[Getting Started: Prerequisites](getting-started.md#prerequisites)
(clang, pkg-config, make, gcc, autotools).

## Workflow setup

A minimal workflow that builds a kernel, caches it, and runs tests:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: [self-hosted, X64]
    env:
      KTSTR_GHA_CACHE: "1"
    steps:
      - uses: actions/checkout@v5
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt
      - uses: taiki-e/install-action@v2
        with:
          tool: cargo-nextest
      - name: Cache kernel images
        uses: actions/cache@v5
        with:
          path: ~/.cache/ktstr/kernels
          key: ktstr-kernels-x64-${{ hashFiles('ktstr.kconfig') }}
          restore-keys: ktstr-kernels-x64-
      - name: Build test kernel
        run: cargo run --bin cargo-ktstr -- ktstr kernel build
      - run: cargo nextest run --workspace --profile ci
```

The kernel built by `cargo ktstr kernel build` is cached and
auto-discovered by the test harness. To pin a specific version, see
[Kernel pinning](#kernel-pinning) below.

If the workspace includes a scheduler binary target, build it before
running tests (e.g. `cargo build -p scx_my_sched`) so that
`SchedulerSpec::Name` discovery finds the binary.

## Kernel pinning

Pin a specific kernel version via the matrix strategy:

```yaml
strategy:
  fail-fast: false
  matrix:
    kernel-version: ['6.12', '7.1']
steps:
  # ...
  - name: Build test kernel
    run: cargo run --bin cargo-ktstr -- ktstr kernel build ${{ matrix.kernel-version }}
  - run: cargo nextest run --workspace --profile ci
    env:
      KTSTR_KERNEL: ${{ matrix.kernel-version }}
```

`KTSTR_KERNEL` tells the test harness which cached kernel to use at
runtime. A major.minor prefix (e.g. `6.12`) resolves to the highest
patch release in that series. See
[Kernel discovery](getting-started.md#kernel-discovery) for the
full resolution chain.

Cache keys should include the kernel version to avoid eviction when
testing multiple versions:

```yaml
key: ktstr-kernels-x64-${{ matrix.kernel-version }}-${{ hashFiles('ktstr.kconfig') }}
restore-keys: ktstr-kernels-x64-${{ matrix.kernel-version }}-
```

## Caching

### Local cache (actions/cache)

`actions/cache` persists `~/.cache/ktstr/kernels` across runs. The
cache key includes `hashFiles('ktstr.kconfig')` so that kconfig
changes trigger a rebuild.

### Remote cache (GHA cache service)

Set `KTSTR_GHA_CACHE=1` to enable the remote kernel cache via the
GHA cache service. This uses `ACTIONS_CACHE_URL` and
`ACTIONS_RUNTIME_TOKEN` (set automatically by the GHA runner).

```yaml
env:
  KTSTR_GHA_CACHE: "1"
```

With remote cache enabled:

- Cache lookups check local first, then remote on miss, before
  falling back to download.
- Successful builds are pushed to the remote automatically.
- Remote failures are non-fatal; local cache is authoritative.

The remote cache stores kernel images as zstd-compressed tar
archives containing the kernel image, vmlinux, .config, and
metadata. Local environment paths are stripped before upload.

Both caching layers can be used together. `actions/cache` provides
fast local restore; the remote cache shares kernels across jobs
and workflow runs that `actions/cache` keys would miss.

## Budget-based test selection

Set `KTSTR_BUDGET_SECS` to limit test runtime:

```yaml
- run: cargo nextest run --workspace --profile ci
  env:
    KTSTR_BUDGET_SECS: "300"
```

The selector greedily picks tests that maximize feature coverage
within the time budget. Useful for smoke-test jobs or constrained
runners. See [Running Tests: Budget-based test selection](running-tests.md#budget-based-test-selection).

## Coverage

Run tests under `cargo llvm-cov` for coverage reports:

```yaml
coverage:
  runs-on: [self-hosted, X64]
  env:
    RUSTC_WRAPPER: ""
    KTSTR_GHA_CACHE: "1"
  steps:
    - uses: actions/checkout@v5
    - uses: dtolnay/rust-toolchain@stable
      with:
        components: rustfmt,llvm-tools-preview
    - uses: taiki-e/install-action@v2
      with:
        tool: cargo-llvm-cov,cargo-nextest
    - name: Cache kernel images
      uses: actions/cache@v5
      with:
        path: ~/.cache/ktstr/kernels
        key: ktstr-kernels-x64-${{ hashFiles('ktstr.kconfig') }}
        restore-keys: ktstr-kernels-x64-
    - name: Build test kernel
      run: cargo run --bin cargo-ktstr -- ktstr kernel build
    - run: cargo llvm-cov nextest --workspace --profile ci --lcov --output-path lcov.info
```

`RUSTC_WRAPPER` must be empty (or unset) for `cargo llvm-cov` --
`sccache` is incompatible with coverage instrumentation.

Requires `llvm-tools-preview` rustup component and `cargo-llvm-cov`.
If the workspace includes a scheduler crate, pass
`--exclude-from-report <crate>` to exclude it from coverage reports.

## Test statistics

Collect test statistics after the test run:

```yaml
- name: Test statistics
  if: ${{ !cancelled() }}
  run: cargo run --bin cargo-ktstr -- ktstr test-stats
  env:
    KTSTR_KERNEL: ${{ matrix.kernel-version }}
```

`test-stats` reads sidecar JSON files from `target/ktstr/{branch}-{hash}/`
and prints gauntlet analysis, BPF verifier stats, callback profiles,
and KVM stats. The `if: !cancelled()` condition ensures stats are
collected even on test failure.

## aarch64

aarch64 runners use the same workflow structure. Differences:

- Runner labels: `[self-hosted, Linux, kvm, kernel-build, ARM64]`
  (adjust labels to match your runner pool).
- Cache key prefix: use `arm64` instead of `x64` to separate
  arch-specific kernel images.
- `RUSTC_WRAPPER`: set to empty string if the runner lacks sccache.

```yaml
test-arm64:
  runs-on: [self-hosted, Linux, kvm, kernel-build, ARM64]
  env:
    RUSTC_WRAPPER: ""
    KTSTR_GHA_CACHE: "1"
  steps:
    - uses: actions/checkout@v5
    - uses: dtolnay/rust-toolchain@stable
      with:
        components: rustfmt
    - uses: taiki-e/install-action@v2
      with:
        tool: cargo-nextest
    - name: Cache kernel images
      uses: actions/cache@v5
      with:
        path: ~/.cache/ktstr/kernels
        key: ktstr-kernels-arm64-${{ hashFiles('ktstr.kconfig') }}
        restore-keys: ktstr-kernels-arm64-
    - name: Build test kernel
      run: cargo run --bin cargo-ktstr -- ktstr kernel build
    - run: cargo nextest run --workspace --profile ci
```

## Performance mode

CI runners may lack `CAP_SYS_NICE`, rtprio limits, or enough host
CPUs for exclusive LLC reservation. Disable performance mode to skip
these features:

```yaml
- run: cargo nextest run --workspace --profile ci
  env:
    KTSTR_NO_PERF_MODE: "1"
```

Tests with `performance_mode=true` are skipped entirely under
`--no-perf-mode`. See
[Performance Mode: Disabling](concepts/performance-mode.md#disabling-performance-mode).

## Environment variables

CI-relevant variables from the
[full reference](reference/environment-variables.md):

| Variable | Purpose |
|---|---|
| `KTSTR_TEST_KERNEL` | Direct path to a bootable kernel image. |
| `KTSTR_KERNEL` | Kernel version or path for test discovery. |
| `KTSTR_GHA_CACHE` | Set to `"1"` to enable remote GHA cache. |
| `KTSTR_BUDGET_SECS` | Time budget for greedy test selection. |
| `KTSTR_NO_PERF_MODE` | Disable performance mode features. |
| `KTSTR_CACHE_DIR` | Override the kernel cache directory. |
| `KTSTR_SIDECAR_DIR` | Override the sidecar output directory. |
| `KTSTR_VERBOSE` | Set to `"1"` for verbose VM console output. |

## Nextest CI profile

The workspace ships a `ci` nextest profile in `.config/nextest.toml`:

```toml
[profile.ci]
slow-timeout = { period = "60s", terminate-after = 3 }
retries = { backoff = "exponential", count = 3, delay = "10s", jitter = true, max-delay = "60s" }
failure-output = "final"
fail-fast = false

[profile.ci.junit]
path = "junit.xml"
```

Use it with `--profile ci`. It allows one extra slow-timeout cycle
before termination, suppresses per-test output until the run
completes, and continues past failures to collect full results.
