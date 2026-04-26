# ktstr

`ktstr` runs ktstr scenarios directly on the host under whatever
scheduler is already active. Unlike `#[ktstr_test]` (which boots
KVM VMs), `ktstr` operates on the host's real topology and cgroups.
It does not manage scheduler lifecycle -- start your scheduler
externally before running.

See also [`cargo ktstr`](cargo-ktstr.md) for cargo-integrated
workflows including test execution, coverage, BPF verifier stats,
and gauntlet statistics.

Build from the workspace:

```sh
cargo build --bin ktstr
```

## Subcommands

### run

Run scenarios on the host:

```sh
ktstr run
ktstr run --flags llc,borrow --duration 30
ktstr run --filter cpuset --json
ktstr run --work-type YieldHeavy
ktstr run --repro --kernel-dir ../linux
ktstr run --auto-repro --probe-stack crash_stack.txt
```

Scenarios run under whatever scheduler is currently active on the
host. Start your scheduler before invoking `ktstr run`.

Without `--flags`, all valid flag profiles are generated for each
scenario. With `--flags`, only the specified profile is run. Flags
select which test profiles to run -- they do not configure the
scheduler. Start the scheduler with the desired features before
running ktstr.

`--filter` selects scenarios whose name contains the given substring.

| Flag | Default | Description |
|------|---------|-------------|
| `--duration SECS` | `20` | Scenario duration in seconds. |
| `--workers N` | `4` | Workers per cgroup. |
| `--flags LIST` | all profiles | Active flags (comma-separated). Omit for all valid profiles. Built-in catalog flags: `llc, borrow, steal, rebal, reject-pin, no-ctrl`. |
| `--filter SUB` | -- | Run only scenarios whose name contains the substring. |
| `--json` | off | Output results as JSON. |
| `--repro` | off | Attach BPF probes for crash capture while running. |
| `--probe-stack` | -- | Crash stack for auto-probe: a file path or comma-separated function names. |
| `--auto-repro` | off | Rerun a crashing scenario with probes attached. |
| `--kernel-dir PATH` | -- | Kernel build directory; used for DWARF source location lookup in probe output (requires `--repro` or `--auto-repro`). |
| `--work-type NAME` | per-scenario | Override the work type for all cgroups. Case-sensitive; see list below. |
| `--no-perf-mode` | off | Disable all performance mode features (flock, pinning, RT scheduling, hugepages, NUMA mbind, KVM exit suppression). Also settable via `KTSTR_NO_PERF_MODE` env var. |

**Work types (for `--work-type`):** `CpuSpin`, `YieldHeavy`, `Mixed`,
`IoSync`, `Bursty`, `PipeIo`, `FutexPingPong`, `CachePressure`,
`CacheYield`, `CachePipe`, `FutexFanOut`, `ForkExit`, `NiceSweep`,
`AffinityChurn`, `PolicyChurn`, `FanOutCompute`, `PageFaultChurn`,
`MutexContention`. Names are matched case-sensitively. `Sequence` and
`Custom` exist in the library but are not constructible from
`--work-type` (`Sequence` requires explicit phases; `Custom` requires
a function pointer).

### list

List available scenarios:

```sh
ktstr list
ktstr list --filter dynamic
ktstr list --json
```

### topo

Show the host CPU topology (CPUs, LLCs, NUMA nodes):

```sh
ktstr topo
```

### cleanup

Remove leftover cgroups from a previous run. With no arguments, cleans
up `/sys/fs/cgroup/ktstr` (used by the test harness) and every
`/sys/fs/cgroup/ktstr-<pid>` directory left behind by `ktstr run`
instances (normal-exit runs remove their own via an RAII guard, so
leftovers are from runs that crashed or were SIGKILLed), removing
the directories themselves. Directories whose `<pid>` still owns a
live `ktstr` or `cargo-ktstr` process are skipped so a concurrent
cleanup never yanks an active run's cgroup; each skip emits a
`ktstr: skipping <path> (live process)` line on stderr. Pass
`--parent-cgroup PATH` to clean a single explicit path (no
live-process check).

```sh
ktstr cleanup
ktstr cleanup --parent-cgroup /sys/fs/cgroup/ktstr-12345
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

### host-state

Capture or compare a host-wide per-thread state snapshot. Useful
for diagnosing "the scheduler looks fine but something on the
host is still behaving oddly" by producing a baseline/candidate
diff of every live thread's scheduling, memory, and I/O
counters — a superset of what any single test's sidecar
captures.

```sh
ktstr host-state capture --output baseline.hst.zst
# ... run workload of interest ...
ktstr host-state capture --output candidate.hst.zst
ktstr host-state compare baseline.hst.zst candidate.hst.zst
```

**`capture`** walks `/proc` at capture time and writes every
visible thread's cumulative counters (schedstat, sched,
status CSW, page faults, I/O bytes, CPU affinity, cgroup,
identity) as zstd-compressed JSON (conventional extension
`.hst.zst`). Every recorded field is cumulative-from-birth so
probe attachment time does not bias the reading — a diff
between two snapshots measures exactly the activity over the
window. Per-cgroup aggregates (`cpu.stat`, `memory.current`)
are captured once per distinct path. Capture is read-only;
nothing is attached, no kprobes, no tracing.

| Flag | Default | Description |
|------|---------|-------------|
| `-o`, `--output PATH` | required | Destination path (convention: `.hst.zst`). Existing files are overwritten. |

**`compare`** joins two snapshots on `(pcomm, comm)` and
renders a per-metric baseline/candidate/delta table. The join
key survives across captures taken on different hosts or
after process restarts, so deltas reflect the behavior of the
named workload rather than a specific pid. Metrics with
cumulative semantics (CPU time, page faults, wait time) show
the candidate-minus-baseline delta; instantaneous metrics
(affinity, cgroup path) show the value at candidate capture
time.

| Arg / Flag | Description |
|------|-------------|
| `BASELINE` | Path to the baseline `.hst.zst` snapshot. |
| `CANDIDATE` | Path to the candidate `.hst.zst` snapshot. |

### locks

Enumerate every ktstr flock held on this host. Read-only —
does NOT attempt any flock acquire. Useful as a troubleshooting
companion for `--cpu-cap` contention: when a build or test is
stalled behind a peer's reservation, `ktstr locks` names the
peer (PID + cmdline) without disturbing any of its flocks.

Scans three lock-file roots:

- `/tmp/ktstr-llc-*.lock` — per-LLC reservations held by
  perf-mode test runs and `--cpu-cap`-bounded builds.
- `/tmp/ktstr-cpu-*.lock` — per-CPU reservations from the
  same flow.
- `{cache_root}/.locks/*.lock` — cache-entry locks held
  during `kernel build` writes.

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

| Arg | Description |
|------|-------------|
| `SHELL` | Shell to generate completions for (`bash`, `zsh`, `fish`, `elvish`, `powershell`). |

The same subcommand is available as `cargo ktstr completions` (which
also accepts `--binary` to set the binary name for completions).
