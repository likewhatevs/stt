# ktstr

`ktstr` is the standalone debugging companion to the
[`#[ktstr_test]`](../writing-tests/ktstr-test-macro.md) test harness.
It owns kernel cache management, interactive VM shells, host-wide
per-thread profiling, and lock introspection — the operations a
scheduler author reaches for when investigating a test failure.

To reproduce a test scenario as a self-contained shell script
without a VM, use [`cargo ktstr export`](cargo-ktstr.md#export).
To run the test suite, use
[`cargo ktstr test`](cargo-ktstr.md#test).

See also [`cargo ktstr`](cargo-ktstr.md) for the cargo-integrated
companion that also covers test execution, coverage, BPF verifier
stats, and gauntlet statistics.

Build from the workspace:

```sh
cargo build --bin ktstr
```

## Subcommands

### topo

Show the host CPU topology (CPUs, LLCs, NUMA nodes):

```sh
ktstr topo
```

### kernel

The `kernel` subcommand manages cached kernel images. Subcommands:
`list`, `build`, `clean`. See
[cargo-ktstr kernel](cargo-ktstr.md#kernel) for full documentation
-- the kernel subcommands are identical in both binaries.

### shell

Boot an interactive shell in a KVM virtual machine. Launches a VM
with busybox and drops into a shell.

```sh
ktstr shell
ktstr shell --kernel ../linux
ktstr shell --kernel 6.14.2
ktstr shell --topology 1,2,4,1
ktstr shell -i /path/to/binary
ktstr shell -i my_tool -i another_tool
```

Files and directories passed via `-i` are available at
`/include-files/<name>` inside the guest. Directories are walked
recursively, preserving structure (e.g. `-i ./release` includes all
files under `release/` at `/include-files/release/...`). Bare names
(without path separators) are resolved via `PATH` lookup.
Dynamically-linked ELF binaries get automatic shared library
resolution via ELF DT_NEEDED parsing. Non-ELF files are copied as-is.

Stdin is a terminal requirement. The host terminal enters raw mode
for bidirectional stdin/stdout forwarding. Terminal state is restored
on all exit paths.

| Flag | Default | Description |
|------|---------|-------------|
| `--kernel ID` | auto | Kernel identifier: a source directory path (e.g. `../linux`), a version (`6.14.2` or major.minor prefix `6.14`), or a cache key (see `ktstr kernel list`). Raw image files are rejected. Source directories auto-build; versions auto-download from kernel.org on cache miss. When absent, resolves via cache then filesystem and falls back to downloading the latest stable kernel. |
| `--topology N,L,C,T` | `1,1,1,1` | Virtual CPU topology as `numa_nodes,llcs,cores,threads`. All values must be >= 1. |
| `-i, --include-files PATH` | -- | Files or directories to include in the guest. Repeatable. Directories are walked recursively. |
| `--memory-mb MB` | auto | Guest memory in MB (minimum 128). When absent, estimated from payload and include file sizes. |
| `--dmesg` | off | Forward kernel console (COM1/dmesg) to stderr in real-time. Sets loglevel=7 for verbose kernel output. |
| `--exec CMD` | -- | Run a command in the VM instead of an interactive shell. The VM exits after the command completes. |
| `--no-perf-mode` | off | Disable all performance mode features (flock, pinning, RT scheduling, hugepages, NUMA mbind, KVM exit suppression). Also settable via `KTSTR_NO_PERF_MODE` env var. |
| `--cpu-cap N` | unset | Reserve only N host CPUs for the shell VM (integer ≥ 1). **Requires `--no-perf-mode`** — perf-mode already holds every LLC exclusively, so capping under perf-mode would double-reserve. The planner walks whole LLCs in consolidation- and NUMA-aware order, partial-taking the last LLC so `plan.cpus.len() == N` exactly. Mutually exclusive with `KTSTR_BYPASS_LLC_LOCKS=1`. Also settable via `KTSTR_CPU_CAP` env var (CLI flag wins when both are present). |

`cargo ktstr shell` runs the same VM boot flow and differs in one
respect: it accepts raw image file paths for `--kernel` (e.g.
`bzImage`, `Image`). Source-tree directories auto-build and no-kernel
invocations auto-download — same as `ktstr shell`.

### ctprof

Capture or compare a host-wide per-thread state snapshot. Useful
for diagnosing "the scheduler looks fine but something on the
host is still behaving oddly" by producing a baseline/candidate
diff of every live thread's scheduling, memory, and I/O
counters — a superset of what any single test's sidecar
captures.

```sh
ktstr ctprof capture --output baseline.ctprof.zst
# ... run workload of interest ...
ktstr ctprof capture --output candidate.ctprof.zst
ktstr ctprof compare baseline.ctprof.zst candidate.ctprof.zst
```

**`capture`** walks `/proc` at capture time and writes every
visible thread's metric values (cumulative counters from
schedstat / sched / status CSW / page faults / I/O bytes /
taskstats; lifetime peaks from schedstat `*_max` and
`hiwater_*`; instantaneous gauges sampled at capture time
including `nr_threads`, `fair_slice_ns`, `state`; categorical /
ordinal scalars including `policy`, `nice`, `cpu_affinity`,
identity strings) as zstd-compressed JSON (conventional
extension `.ctprof.zst`). Cumulative counters and lifetime
peaks are probe-timing-invariant — sampled twice, the value
either monotonically increased or stayed at its high-water mark
— so a diff between two snapshots measures exactly the
activity over the window. Instantaneous gauges and categorical
scalars are point-in-time readings that can legitimately differ
between two probes of the same thread. Per-cgroup aggregates
(`cpu.stat`, `memory.current`) are captured once per distinct
path. Capture is read-only; nothing is attached, no kprobes, no
tracing.

| Flag | Default | Description |
|------|---------|-------------|
| `-o`, `--output PATH` | required | Destination path (convention: `.ctprof.zst`). Existing files are overwritten. |

**`compare`** joins two snapshots on the selected grouping
axis (`pcomm` by default) and renders a per-metric
baseline/candidate/delta table. The join key survives across
captures taken on different hosts or after process restarts, so
deltas reflect the behavior of the named workload rather than a
specific pid. Metrics with cumulative semantics (CPU time, page
faults, wait time) show the candidate-minus-baseline delta;
instantaneous metrics (affinity, cgroup path) show the value at
candidate capture time. See the
[ctprof reference](../reference/ctprof.md) for the full metric
registry, aggregation rules, derived-metric formulas, and
taskstats kconfig gating.

| Arg / Flag | Default | Description |
|------|---------|-------------|
| `BASELINE` | required | Path to the baseline `.ctprof.zst` snapshot. |
| `CANDIDATE` | required | Path to the candidate `.ctprof.zst` snapshot. |
| `--group-by AXIS` | `pcomm` | Grouping axis: `pcomm` (process name), `cgroup` (cgroup v2 path), `comm` (thread-name pattern, token-normalized), or `comm-exact` (synonym for `comm --no-thread-normalize`). |
| `--cgroup-flatten GLOB` | -- | Glob pattern that collapses dynamic cgroup path segments before grouping (e.g. `'/kubepods/*/workload'`). Repeatable; explicit globs apply before auto-normalize. |
| `--no-thread-normalize` | off | Disable token-based pattern normalization for `--group-by comm`. Threads group by literal `comm`. |
| `--no-cg-normalize` | off | Disable token-based normalization for `--group-by cgroup`. Cgroup paths group by literal post-flatten path. |
| `--sort-by SPEC` | by largest `\|delta_pct\|` | Multi-key sort spec: `metric1[:dir1],metric2[:dir2],...`. Each `metric` is a name from `ctprof metric-list`; each `dir` is `asc` or `desc` (default `desc`). |
| `--display-format FORMAT` | `full` | Per-row column layout. `full` (default 7 columns), `delta-only` (drop baseline+candidate), `no-pct`, `arrow` (collapse baseline/candidate/delta into one cell), or `pct-only`. |
| `--columns NAMES` | -- | Comma-separated column list overriding `--display-format`. Valid names: `group`, `threads`, `metric`, `baseline`, `candidate`, `delta`, `%`, `arrow`. Order is the rendered order. |
| `--sections NAMES` | every | Comma-separated sub-table list. Valid names: `primary`, `taskstats-delay`, `derived`, `cgroup-stats`, `cgroup-limits`, `memory-stat`, `memory-events`, `pressure`, `host-pressure`, `smaps-rollup`, `sched-ext`. Empty renders every section that has data. |
| `--metrics NAMES` | every | Comma-separated metric-name allowlist for primary + derived rows. Names must come from `ctprof metric-list`. Composes multiplicatively with `--sections`. |
| `--wrap` | off | Wrap table cells to fit terminal width. Only fires when stdout is a TTY; piped output stays unwrapped so awk/grep pipelines see the same byte sequence. |

**`show`** renders a single snapshot's per-(group, metric)
values without diff math. Same flag vocabulary as `compare`,
minus the baseline/candidate/delta/pct columns:

```sh
ktstr ctprof show snapshot.ctprof.zst --group-by cgroup
ktstr ctprof show snapshot.ctprof.zst --sections taskstats-delay
```

| Arg / Flag | Default | Description |
|------|---------|-------------|
| `SNAPSHOT` | required | Path to the `.ctprof.zst` snapshot. |
| `--group-by AXIS` | `pcomm` | Same as `compare`. |
| `--cgroup-flatten GLOB` | -- | Same as `compare`. Repeatable. |
| `--no-thread-normalize` | off | Same as `compare`. |
| `--no-cg-normalize` | off | Same as `compare`. |
| `--sort-by SPEC` | alphabetical | Sort spec; ranks groups by absolute aggregated value (no delta — single snapshot). |
| `--columns NAMES` | -- | Comma-separated column list. Show-only valid names: `group`, `threads`, `metric`, `value`. The compare-only column names are rejected at parse time. |
| `--sections NAMES` | every | Same as `compare`. |
| `--metrics NAMES` | every | Same as `compare`. |
| `--wrap` | off | Same as `compare`. |

**`metric-list`** prints every registered metric (primary +
derived) with its description, unit, kconfig gate, and
sched_class scope. Use this to discover the vocabulary
`--sort-by` and `--metrics` accept.

```sh
ktstr ctprof metric-list
```

### locks

Enumerate every ktstr flock held on this host. Read-only —
does NOT attempt any flock acquire. Useful as a troubleshooting
companion for `--cpu-cap` contention: when a build or test is
stalled behind a peer's reservation, `ktstr locks` names the
peer (PID + cmdline) without disturbing any of its flocks.

Scans four lock-file roots:

- `/tmp/ktstr-llc-*.lock` — per-LLC reservations held by
  perf-mode test runs and `--cpu-cap`-bounded builds.
- `/tmp/ktstr-cpu-*.lock` — per-CPU reservations from the
  same flow.
- `{cache_root}/.locks/*.lock` — cache-entry locks held
  during `kernel build` writes, and `source-{path_hash}.lock`
  files held for the duration of `kernel build --source` and
  `cargo ktstr test --kernel <path>` against the same source tree.
- `{runs_root}/.locks/{kernel}-{project_commit}.lock` —
  per-run-key sidecar-write locks held for the duration of
  the (pre-clear + write) cycle to serialize concurrent
  ktstr processes targeting the same run directory.

Each lock is cross-referenced against `/proc/locks` to name the
holder PID and cmdline.

```sh
ktstr locks                       # one-shot snapshot
ktstr locks --json                # JSON snapshot
ktstr locks --watch 1s            # redraw every second until SIGINT
ktstr locks --watch 1s --json     # ndjson stream, one object per interval
```

| Flag | Default | Description |
|------|---------|-------------|
| `--json` | off | Emit the snapshot as JSON. Pretty-printed in one-shot mode; compact (one object per line, ndjson-style) under `--watch`. Stable field names — schema documented on `ktstr::cli::list_locks`. |
| `--watch DURATION` | unset | Redraw the snapshot at the given interval until SIGINT. Value is parsed by `humantime`: `100ms`, `1s`, `5m`, `1h`. Human output clears and redraws in place; `--json` emits one line-terminated object per interval. |

The same subcommand is available as `cargo ktstr locks` with
identical flag semantics.

### completions

Generate shell completions for ktstr.

```sh
ktstr completions bash
ktstr completions zsh
ktstr completions fish
```

| Arg / Flag | Default | Description |
|------|---------|-------------|
| `SHELL` | required | Shell to generate completions for (`bash`, `zsh`, `fish`, `elvish`, `powershell`). |
| `--binary NAME` | `ktstr` | Binary name to register the completion under. Override when invoking ktstr through a symlink with a different name (the shell looks up completions by argv\[0]). |

The same subcommand is available as `cargo ktstr completions`
with identical flag semantics (`--binary` accepted on both;
defaults to the respective binary name).
