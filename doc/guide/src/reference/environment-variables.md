# Environment Variables

Environment variables that control ktstr behavior.

## User-facing

| Variable | Description | Default |
|---|---|---|
| `KTSTR_KERNEL` | Kernel identifier for cargo-build-time BTF resolution (read by `build.rs`) and runtime image discovery. Accepts a path (`../linux`), version string (`6.14.2`), or cache key (use `cargo ktstr kernel list` for actual keys). During `cargo build`, only paths are used (`build.rs` extracts BTF from `vmlinux`). At runtime, version strings and cache keys resolve via the XDG cache; paths search only the specified directory (error if no image found). Set automatically by `cargo ktstr test --kernel`. **Overridden by `KTSTR_KERNEL_LIST` when present:** under multi-kernel runs the test binary's `--list` / `--exact` handlers consult `KTSTR_KERNEL_LIST` first and only fall back to `KTSTR_KERNEL` when the list env is unset; the producer-side `cargo ktstr` always sets `KTSTR_KERNEL` to the FIRST resolved entry alongside the full `KTSTR_KERNEL_LIST` so downstream code that inspects `KTSTR_KERNEL` directly still sees a valid path. | Auto-discovered |
| `KTSTR_KERNEL_LIST` | Multi-kernel wire format `label1=path1;label2=path2;…` consumed by the test binary's gauntlet expansion. Set by `cargo ktstr test` / `coverage` / `llvm-cov` when the resolved kernel set has 2 or more entries; the test binary's `--list` handler emits one variant per kernel (suffix `gauntlet/{name}/{preset}/{profile}/{kernel_label}` or `ktstr/{name}/{kernel_label}`) and the `--exact` handler strips the suffix and re-exports `KTSTR_KERNEL` to the matching directory before booting the VM. Semicolon is the entry separator (paths can carry `:` on POSIX); `=` separates label from path. Empty value or unset means "single-kernel mode" — the test binary falls back to `KTSTR_KERNEL`. | None (single-kernel) |
| `KTSTR_CI` | Set to any non-empty value to flip every sidecar's `run_source` field from `"local"` (developer-machine default) to `"ci"`. Read at sidecar-write time by `detect_run_source`; surfaces through `cargo ktstr stats compare --run-source ci` so CI-produced runs can be partitioned from developer runs without per-run directory bookkeeping. Empty string counts as unset. The third value `"archive"` is applied at LOAD time (not write time) when `cargo ktstr stats compare --dir` / `list-values --dir` pulls sidecars from a non-default pool root — `KTSTR_CI` does not control that. | None (`run_source = "local"`) |
| `KTSTR_TEST_KERNEL` | Path to a bootable kernel image (`bzImage` on x86_64, `Image` on aarch64). See [Getting Started](../getting-started.md#kernel-discovery) and [Troubleshooting](../troubleshooting.md#no-kernel-found) for search order. | Auto-discovered |
| `KTSTR_CACHE_DIR` | Override the kernel image cache directory. When set, all cache operations use this path instead of the XDG default. | `$XDG_CACHE_HOME/ktstr/kernels/` or `$HOME/.cache/ktstr/kernels/` |
| `KTSTR_GHA_CACHE` | Set to `"1"` to enable remote kernel cache via GitHub Actions cache service. Requires `ACTIONS_CACHE_URL` (set by the GHA runner). Local cache is always authoritative; remote failures are non-fatal. | None (disabled) |
| `KTSTR_SCHEDULER` | Path to a scheduler binary for `SchedulerSpec::Discover`. See [Troubleshooting](../troubleshooting.md#scheduler-not-found) for search order. | Auto-discovered |
| `KTSTR_BUDGET_SECS` | Time budget in seconds for greedy test selection during `--list`. Must be positive. See [Running Tests](../running-tests.md). | None (all tests listed) |
| `KTSTR_SIDECAR_DIR` | Directory for per-test result sidecar JSON files. Used as-is when set, no key suffix. Consumed by the test harness (sidecar write path) and by bare `cargo ktstr stats` (sidecar read path). When this override is set, **pre-clear is skipped** — the operator chose the directory and owns its contents, so any pre-existing sidecars there are preserved. `cargo ktstr stats list`, `cargo ktstr stats compare`, `cargo ktstr stats list-values`, and `cargo ktstr stats show-host` walk `{CARGO_TARGET_DIR or "target"}/ktstr/` by default and ignore `KTSTR_SIDECAR_DIR` — pass `--dir DIR` on `compare` / `list-values` / `show-host` to point them at an alternate run root. See [Runs](../running-tests/runs.md). | `{CARGO_TARGET_DIR or "target"}/ktstr/{kernel}-{project_commit}/` (where `{project_commit}` is the project HEAD short hex, suffixed `-dirty` when the worktree differs, or the literal `unknown` when not in a git repo — see [Runs](../running-tests/runs.md) for the unknown-commit collision semantics) |
| `KTSTR_NO_PERF_MODE` | Force `performance_mode=false` and skip flock topology reservation. Disables all performance mode features (pinning, RT scheduling, hugepages, NUMA mbind, KVM exit suppression). Presence is sufficient (any value). See [Performance Mode](../concepts/performance-mode.md#disabling-performance-mode). Also settable via `--no-perf-mode` CLI flag. | None (disabled) |
| `KTSTR_CPU_CAP` | Cap the number of host CPUs reserved by a no-perf-mode VM or kernel build to `N` (integer ≥ 1, a CPU count). The planner walks whole LLCs in consolidation- / NUMA-aware order, filtered to the calling process's sched_getaffinity cpuset, partial-taking the last LLC so `plan.cpus.len()` is EXACTLY `N`. CLI flag `--cpu-cap N` takes precedence; empty string is treated as unset; `0` or non-numeric values are rejected with a parse error. On `shell`, `--cpu-cap` is rejected at clap parse time unless `--no-perf-mode` is also passed (`requires = "no_perf_mode"`); on `kernel build`, no perf-mode concept applies. Library consumers that set `performance_mode=true` on `KtstrVmBuilder` directly see the env var silently ignored — the builder's perf-mode branch never consults `CpuCap::resolve`. Mutually exclusive with `KTSTR_BYPASS_LLC_LOCKS=1` at every entry point (rejection wording contains "resource contract"). See [Resource Budget](../concepts/resource-budget.md). | None (30% of allowed CPUs, minimum 1) |
| `KTSTR_BYPASS_LLC_LOCKS` | Skip host-side LLC flock acquisition entirely. No coordination against concurrent perf-mode runs. Presence is sufficient (any non-empty value). Mutually exclusive with `KTSTR_CPU_CAP` / `--cpu-cap` — the conflict is rejected at every entry point with an error containing "resource contract". See [Resource Budget](../concepts/resource-budget.md#ktstr_bypass_llc_locks--escape-hatch). | None (coordinate) |
| `KTSTR_KERNEL_PARALLELISM` | Override the rayon pool width `cargo ktstr` uses for `--kernel` per-spec fan-out in `resolve_kernel_set`. Parsed as `usize` after `.trim()`; whitespace around the value is tolerated. Values that fail to parse, are negative, or are `0` silently fall through to the default — a typoed export (`=abc`, `=0`) does NOT disable parallelism, it degrades to the host-CPU default. Useful when the default is wrong for the host: a fast NIC + slow CPU benefits from a higher value (more concurrent downloads); a contended CI runner benefits from a lower cap to leave bandwidth and CPU for sibling jobs. **Scope is narrow**: only the bounded `ThreadPool` `resolve_kernel_set` builds via `ThreadPoolBuilder::install` is affected — the global rayon pool that other code paths (nextest harness, polars groupby, etc.) consume is untouched. The build phase inside each per-spec resolve is already serialized at the LLC-flock layer, so raising this knob accelerates download fan-out only, not build time. | `std::thread::available_parallelism()` (host logical CPU count, falling back to `1` on a sandboxed host where `available_parallelism` errors) |
| `KTSTR_VERBOSE` | Set to `"1"` for verbose VM console output (`earlyprintk`, `loglevel=7`). | None |
| `RUST_BACKTRACE` | Gates verbose diagnostic output on failure. Also enables verbose VM console output (same as `KTSTR_VERBOSE=1`) when set to `"1"` or `"full"`. Propagated to the guest. | None |
| `RUST_LOG` | Controls tracing filter for guest-side logging. Propagated to the VM kernel command line and parsed by the guest tracing subscriber. | None |

## jemalloc probe wiring

These variables are only consulted by integration tests that boot a
jemalloc-linked allocator worker inside the VM and attach the
`ktstr-jemalloc-probe` to it (see `tests/jemalloc_probe_tests.rs`).
Both are set from a `#[ctor]` in the test binary so they land before
the test harness dispatches.

### What `#[ctor]` is and why these variables need it

`#[ctor]` is a Rust attribute (provided by the
[`ctor` crate](https://crates.io/crates/ctor)) that marks a
function to run automatically at binary initialization — after the
dynamic linker sets up the process but before `main()` is called.
Linux implements this via the `.init_array` ELF section; the
attribute's generated code registers the function there. A function
under `#[ctor]` therefore runs exactly once per process, on the
main thread, before any code inside `main()` executes.

The two environment variables above are consulted by ktstr's
nextest pre-dispatch path (`ktstr_test_early_dispatch`), which
itself runs under a ktstr-owned `#[ctor]` that intercepts the
nextest protocol args (`--list`, `--exact`) before the standard
Rust test harness sees them. The probe-wiring variables must
already be populated when that early dispatch fires, so setting
them from plain test-body code is too late — the sidecar
enumeration and initramfs packing decisions have already run.
Tests needing probe integration install their own `#[ctor]` that
writes the two variables via `std::env::set_var`, ensuring both
ktstr's early dispatch and the VM launch path downstream see the
populated values.

The ctor hook runs under the `ctor` crate re-exported at
`ktstr::__private::ctor`, so a new test crate does not need to
add `ctor` to its own dependencies — it can use the re-export
via `ktstr::__private::ctor::ctor` and stay in sync with the
version ktstr itself depends on, avoiding the "two ctor
crates, two `.init_array` entries, ordering undefined" pitfall.

Leaving either variable unset is the normal case — the VM
launcher skips probe wiring entirely, and no initramfs entry is
added.

| Variable | Description | Default |
|---|---|---|
| `KTSTR_JEMALLOC_PROBE_BINARY` | Absolute host path to the `ktstr-jemalloc-probe` binary. When set, the probe is packed into every VM's base initramfs at `/bin/ktstr-jemalloc-probe`. Typically set by a `#[ctor]` in the integration test crate to `env!("CARGO_BIN_EXE_ktstr-jemalloc-probe")`. Empty string is treated the same as unset. | None (no probe packed) |
| `KTSTR_JEMALLOC_ALLOC_WORKER_BINARY` | Absolute host path to the paired `ktstr-jemalloc-alloc-worker` binary. Packed alongside the probe for the closed-loop tests that run the probe against a live allocator target. Same `#[ctor]` shape as above using `env!("CARGO_BIN_EXE_ktstr-jemalloc-alloc-worker")`. Empty string is treated the same as unset. | None (no worker packed) |

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
