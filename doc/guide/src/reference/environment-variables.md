# Environment Variables

Environment variables that control ktstr behavior.

## User-facing

| Variable | Description | Default |
|---|---|---|
| `KTSTR_KERNEL` | Kernel identifier for cargo-build-time BTF resolution (read by `build.rs`) and runtime image discovery. Accepts a path (`../linux`), version string (`6.14.2`), or cache key (use `cargo ktstr kernel list` for actual keys). During `cargo build`, only paths are used (`build.rs` extracts BTF from `vmlinux`). At runtime, version strings and cache keys resolve via the XDG cache; paths search only the specified directory (error if no image found). Set automatically by `cargo ktstr test --kernel`. | Auto-discovered |
| `KTSTR_TEST_KERNEL` | Path to a bootable kernel image (`bzImage` on x86_64, `Image` on aarch64). See [Getting Started](../getting-started.md#kernel-discovery) and [Troubleshooting](../troubleshooting.md#no-kernel-found) for search order. | Auto-discovered |
| `KTSTR_CACHE_DIR` | Override the kernel image cache directory. When set, all cache operations use this path instead of the XDG default. | `$XDG_CACHE_HOME/ktstr/kernels/` or `$HOME/.cache/ktstr/kernels/` |
| `KTSTR_GHA_CACHE` | Set to `"1"` to enable remote kernel cache via GitHub Actions cache service. Requires `ACTIONS_CACHE_URL` (set by the GHA runner). Local cache is always authoritative; remote failures are non-fatal. | None (disabled) |
| `KTSTR_SCHEDULER` | Path to a scheduler binary for `SchedulerSpec::Name`. See [Troubleshooting](../troubleshooting.md#scheduler-not-found) for search order. | Auto-discovered |
| `KTSTR_BUDGET_SECS` | Time budget in seconds for greedy test selection during `--list`. Must be positive. See [Running Tests](../running-tests.md). | None (all tests listed) |
| `KTSTR_SIDECAR_DIR` | Directory for per-test result sidecar JSON files. See [Baselines](../running-tests/baselines.md). | `target/ktstr/{branch}-{hash}/` |
| `KTSTR_NO_PERF_MODE` | Force `performance_mode=false` and skip flock topology reservation. Disables all performance mode features (pinning, RT scheduling, hugepages, NUMA mbind, KVM exit suppression). Presence is sufficient (any value). See [Performance Mode](../concepts/performance-mode.md#disabling-performance-mode). Also settable via `--no-perf-mode` CLI flag. | None (disabled) |
| `KTSTR_VERBOSE` | Set to `"1"` for verbose VM console output (`earlyprintk`, `loglevel=7`). | None |
| `RUST_BACKTRACE` | Gates verbose diagnostic output on failure. Also enables verbose VM console output (same as `KTSTR_VERBOSE=1`) when set to `"1"` or `"full"`. Propagated to the guest. | None |
| `RUST_LOG` | Controls tracing filter for guest-side logging. Propagated to the VM kernel command line and parsed by the guest tracing subscriber. | None |

## LLVM coverage

| Variable | Description | Default |
|---|---|---|
| `LLVM_COV_TARGET_DIR` | Directory for extracted profraw files. | Parent of `LLVM_PROFILE_FILE`, or `<exe-dir>/llvm-cov-target/` |
| `LLVM_PROFILE_FILE` | Standard LLVM profiling output path. ktstr reads its parent as a fallback profraw directory. | None |

## Nextest protocol

| Variable | Description | Default |
|---|---|---|
| `NEXTEST` | Set by nextest when it invokes the test binary. ktstr's `#[ctor]` dispatch inspects this to decide whether to intercept the nextest protocol args (`--list`, `--exact`) for gauntlet expansion and budget-based selection before `main()` runs. Under plain `cargo test`, this is unset and the standard harness runs the `#[test]` wrappers directly. | None |

## VM-internal

Set by the host on the guest kernel command line and read by the
guest init (via `/proc/cmdline`). Not intended for user
configuration; listed here for debugging.

| Variable | Description |
|---|---|
| `SCHED_PID` | PID of the scheduler process inside the guest, published after scheduler spawn. |
| `KTSTR_MODE` | Guest execution mode (`run` for test dispatch, `shell` for interactive shell). |
| `KTSTR_TOPO` | Topology string (`numa_nodes,llcs,cores,threads`) for guest-side scenario resolution. |
| `KTSTR_SHM_BASE` | Host-physical base address of the SHM ring region (hex). |
| `KTSTR_SHM_SIZE` | Size in bytes of the SHM ring region (hex). |
| `KTSTR_TERM` | Terminal type forwarded from the host (sets guest `TERM`). |
| `KTSTR_COLORTERM` | Color capability forwarded from the host (sets guest `COLORTERM`). |
| `KTSTR_COLS` | Host terminal column count, used to size the guest pty when available. |
| `KTSTR_ROWS` | Host terminal row count, used to size the guest pty when available. |

Sentinel tokens (`===KTSTR_TEST_RESULT_START===`,
`===KTSTR_TEST_RESULT_END===`, `KTSTR_EXIT=N`,
`KTSTR_INIT_STARTED`, `KTSTR_PAYLOAD_STARTING`, `KTSTR_EXEC_EXIT`)
are protocol markers written to COM2; they are not environment
variables.
