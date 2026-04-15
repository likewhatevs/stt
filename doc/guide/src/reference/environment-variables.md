# Environment Variables

Environment variables that control ktstr behavior.

## User-facing

| Variable | Description | Default |
|---|---|---|
| `KTSTR_KERNEL` | Kernel identifier for build-time BTF resolution and runtime image discovery. Accepts a path (`../linux`), version string (`6.14.2`), or cache key (`6.14.2-tarball-x86_64`). At build time, only paths are used (BTF from `vmlinux`). At runtime, version strings and cache keys resolve via the XDG cache; paths search only the specified directory (error if no image found). Set automatically by `cargo ktstr test --kernel`. | Auto-discovered |
| `KTSTR_TEST_KERNEL` | Path to a bootable kernel image (bzImage). See [Getting Started](../getting-started.md#build-a-kernel) and [Troubleshooting](../troubleshooting.md#no-kernel-found) for search order. | Auto-discovered |
| `KTSTR_CACHE_DIR` | Override the kernel image cache directory. When set, all cache operations use this path instead of the XDG default. | `$XDG_CACHE_HOME/ktstr/kernels/` or `$HOME/.cache/ktstr/kernels/` |
| `KTSTR_GHA_CACHE` | Set to `"1"` to enable remote kernel cache via GitHub Actions cache service. Requires `ACTIONS_CACHE_URL` (set by the GHA runner). Local cache is always authoritative; remote failures are non-fatal. | None (disabled) |
| `KTSTR_SCHEDULER` | Path to a scheduler binary for `SchedulerSpec::Name`. See [Troubleshooting](../troubleshooting.md#scheduler-not-found) for search order. | Auto-discovered |
| `KTSTR_BUDGET_SECS` | Time budget in seconds for greedy test selection during `--list`. Must be positive. See [Running Tests](../running-tests.md). | None (all tests listed) |
| `KTSTR_SIDECAR_DIR` | Directory for per-test result sidecar JSON files. See [Baselines](../running-tests/baselines.md). | `target/ktstr/{branch}-{hash}/` |
| `KTSTR_VERBOSE` | Set to `"1"` for verbose VM console output (`earlyprintk`, `loglevel=7`). | None |
| `RUST_BACKTRACE` | Gates verbose diagnostic output on failure. Also enables verbose VM console output (same as `KTSTR_VERBOSE=1`) when set to `"1"` or `"full"`. Propagated to the guest. | None |
| `RUST_LOG` | Controls tracing filter for guest-side logging. Propagated to the VM kernel command line and parsed by the guest tracing subscriber. | None |
| `LINUX_ROOT` | Path to a Linux source tree containing `vmlinux` for BTF monitor tests. | None |

## LLVM coverage

| Variable | Description | Default |
|---|---|---|
| `LLVM_COV_TARGET_DIR` | Directory for extracted profraw files. | Parent of `LLVM_PROFILE_FILE`, or `<exe-dir>/llvm-cov-target/` |
| `LLVM_PROFILE_FILE` | Standard LLVM profiling output path. ktstr reads its parent as a fallback profraw directory. | None |

## Nextest protocol

| Variable | Description | Default |
|---|---|---|
| `NEXTEST` | Set by nextest. ktstr intercepts `--list` and `--exact` args when present. | None |

## VM-internal

These are set inside the VM by the guest init and are not intended for user configuration.

| Variable | Description | Default |
|---|---|---|
| `SCHED_PID` | PID of the scheduler process inside the guest. | Set after scheduler spawn |
