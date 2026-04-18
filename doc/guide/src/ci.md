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
for diagnosing KVM issues on runners, including cloud VM nested
virtualization setup (GCP, AWS, Azure).

Runners also need the build dependencies listed in
[Getting Started: Prerequisites](getting-started.md#prerequisites)
(clang, pkg-config, make, gcc, autotools) and at least 5 GB of free
disk for kernel source extraction, build artifacts, and cached images.

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
      - name: Install ktstr
        run: cargo install --path .
      - name: Cache kernel images
        uses: actions/cache@v5
        with:
          path: ~/.cache/ktstr/kernels
          key: ktstr-kernels-x64-${{ hashFiles('ktstr.kconfig') }}
          restore-keys: ktstr-kernels-x64-
      - name: Build test kernel
        run: cargo ktstr kernel build
      - run: cargo ktstr test -- --profile ci --features integration
```

The test harness auto-discovers the built kernel. `--profile ci`
configures nextest timeouts and retry behavior; see
[Nextest CI profile](#nextest-ci-profile). `KTSTR_GHA_CACHE` enables
a remote kernel cache; see [Caching](#caching). To pin a specific
kernel version, see [Kernel pinning](#kernel-pinning) below.

## Kernel pinning

Pin a specific kernel version via the matrix strategy:

```yaml
strategy:
  fail-fast: false
  matrix:
    kernel-version: ['6.14', '7.0']
steps:
  # ...
  - name: Install ktstr
    run: cargo install --path .
  - name: Build test kernel
    run: cargo ktstr kernel build ${{ matrix.kernel-version }}
  - run: cargo ktstr test --kernel ${{ matrix.kernel-version }} -- --profile ci --features integration
```

`--kernel` tells `cargo ktstr test` which cached kernel to use at
runtime. A major.minor prefix (e.g. `6.14`) resolves to the highest
patch release in that series. See
[Kernel discovery](getting-started.md#kernel-discovery) for the
full resolution chain.

When testing multiple kernel versions, add the version to the cache
key (unlike the minimal workflow above, which omits it because it
builds a single kernel):

```yaml
key: ktstr-kernels-x64-${{ matrix.kernel-version }}-${{ hashFiles('ktstr.kconfig') }}
restore-keys: ktstr-kernels-x64-${{ matrix.kernel-version }}-
```

## Caching

`actions/cache` persists `~/.cache/ktstr/kernels` across runs, keyed
on `hashFiles('ktstr.kconfig')` so kconfig changes trigger a rebuild.

Set `KTSTR_GHA_CACHE=1` to enable a remote cache layer that shares
kernels across jobs and workflow runs. Remote failures are non-fatal;
local cache is authoritative.

## Budget-based test selection

Set `KTSTR_BUDGET_SECS` to limit test runtime:

```yaml
- run: cargo ktstr test -- --profile ci --features integration
  env:
    KTSTR_BUDGET_SECS: "300"
```

The selector greedily picks tests that maximize feature coverage
within the time budget. Useful for smoke-test jobs or constrained
runners. See [Running Tests: Budget-based test selection](running-tests.md#budget-based-test-selection).

## Coverage

Run tests under `cargo ktstr coverage` for coverage reports:

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
    - name: Install ktstr
      run: cargo install --path .
    - name: Cache kernel images
      uses: actions/cache@v5
      with:
        path: ~/.cache/ktstr/kernels
        key: ktstr-kernels-x64-${{ hashFiles('ktstr.kconfig') }}
        restore-keys: ktstr-kernels-x64-
    - name: Build test kernel
      run: cargo ktstr kernel build
    - run: cargo ktstr coverage -- --profile ci --lcov --output-path lcov.info --features integration --exclude-from-report scx-ktstr
```

`RUSTC_WRAPPER` must be empty (or unset) -- `sccache` is
incompatible with coverage instrumentation. Requires
`llvm-tools-preview` rustup component and `cargo-llvm-cov`. Pass
`--exclude-from-report <crate>` to exclude scheduler crates from
coverage reports (the example excludes `scx-ktstr`, the project's
own test fixture scheduler).

## Test statistics

Collect test statistics after the test run:

```yaml
- name: Test statistics
  if: ${{ !cancelled() }}
  run: cargo ktstr stats
```

`stats` reads sidecar JSON files from `target/ktstr/`
and prints gauntlet analysis, BPF verifier stats, callback profiles,
and KVM stats. The `if: !cancelled()` condition ensures stats are
collected even on test failure. See
[cargo-ktstr stats](running-tests/cargo-ktstr.md#stats) for
subcommands and options.

## aarch64

aarch64 runners use the same workflow as x64. Copy the x64 workflow
above and apply these three differences:

- Runner labels: `[self-hosted, Linux, kvm, kernel-build, ARM64]`
  (adjust to match your runner pool).
- Cache key prefix: `arm64` instead of `x64`.
- `RUSTC_WRAPPER`: set to empty string unless sccache is installed
  on the arm64 runner — the workflow's global `RUSTC_WRAPPER=sccache`
  breaks builds when sccache is absent.

## Performance mode

CI runners may lack `CAP_SYS_NICE`, rtprio limits, or enough host
CPUs for exclusive LLC reservation. Disable performance mode to skip
these features:

```yaml
- run: cargo ktstr test -- --profile ci --features integration
  env:
    KTSTR_NO_PERF_MODE: "1"
```

Tests with `performance_mode=true` are skipped entirely under
`--no-perf-mode`. See
[Performance Mode: Disabling](concepts/performance-mode.md#disabling-performance-mode).

## Environment variables

See the [full reference](reference/environment-variables.md) for all
environment variables. The CI-relevant ones are `KTSTR_GHA_CACHE`,
`KTSTR_BUDGET_SECS`, `KTSTR_NO_PERF_MODE`, `KTSTR_KERNEL`, and
`KTSTR_CACHE_DIR`.

## Nextest CI profile

The workspace ships a `ci` nextest profile in `.config/nextest.toml`.
Compared to the default profile, it raises the slow-timeout
termination threshold from 2 to 3 cycles (`terminate-after = 3`),
defers per-test output until the run completes
(`failure-output = "final"`), and continues past failures
(`fail-fast = false`). Use it with `--profile ci`.

See [Tests pass locally but fail in CI](troubleshooting.md#tests-pass-locally-but-fail-in-ci)
for common CI failure causes.
