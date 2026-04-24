# Host-State Profiler

The host-state profiler captures a host-wide per-thread snapshot
of scheduling counters, memory / I/O accounting, CPU affinity,
cgroup state, and thread identity, then compares two snapshots to
surface what changed. It is a manually-invoked CLI companion to
the automated scheduler tests — useful when a run passes on one
machine and fails on another, or for A/B comparing host behaviour
across kernel / sysctl / workload changes.

This is a **different tool** from `ktstr show-host` /
`cargo ktstr show-host`, which captures the host *context*
(kernel, CPU model, sched_\* tunables, NUMA layout, kernel
cmdline) — aggregate state that does not change between
scenarios. The profiler captures *per-thread* cumulative counters
that do change, and its comparison surface is designed for the
thread-level diff.

## When to use it

- **Workload investigation** — you observe a regression and want
  to know which process / thread pool moved in run time,
  context-switch rate, or migration count.
- **Kernel / sysctl A/B** — capture before and after flipping a
  sched_\* tunable on an otherwise-identical workload; the
  compare output surfaces every counter that responded.
- **Host baselining** — capture on a known-good host, capture on
  a failing host, compare to isolate what differs at the
  thread-behaviour level.

The profiler is **not** invoked automatically by scenarios or the
gauntlet. It is opt-in and operator-driven via the
`ktstr host-state` subcommand.

## Capture

```sh
ktstr host-state capture --output baseline.hst.zst
# ... run workload, change a tunable, reboot a kernel, etc. ...
ktstr host-state capture --output after.hst.zst
```

`capture` walks `/proc` for every live thread group, enumerates
each thread, and reads a handful of procfs sources for each one.
The output is a zstd-compressed JSON snapshot (conventional
extension: `.hst.zst`).

### What is captured per thread

- **Identity** — tid, tgid, `pcomm` (process name from
  `/proc/<tgid>/comm`), `comm` (thread name from
  `/proc/<tid>/comm`), cgroup v2 path, `start_time` in USER_HZ
  clock ticks, scheduling policy name, nice, CPU affinity mask.
- **Scheduling counters** (cumulative, from `/proc/<tid>/sched`;
  requires `CONFIG_SCHED_DEBUG`) — `run_time_ns`, `wait_time_ns`,
  `timeslices`, `voluntary_csw`, `nonvoluntary_csw`, `nr_wakeups`
  (plus `_local` / `_remote` / `_sync` / `_migrate` / `_idle`
  splits), `nr_migrations`, `wait_sum` / `wait_count`,
  `sleep_sum`, `block_sum` / `block_count`, `iowait_sum` /
  `iowait_count`.
- **Memory** — `minflt` / `majflt` from `/proc/<tid>/stat`.
  `allocated_bytes` / `deallocated_bytes` from the jemalloc
  per-thread-destructor TSD cache — populated only for processes
  linked against jemalloc; glibc arena counters are opaque and
  read as zero rather than failing capture.
- **I/O** — `rchar`, `wchar`, `syscr`, `syscw`, `read_bytes`,
  `write_bytes` from `/proc/<tid>/io` (requires
  `CONFIG_TASK_IO_ACCOUNTING`).

Every field is **cumulative-from-birth**, so the probe timing
does not alter the output: two snapshots of the same thread at
different wall-clock instants produce the same numbers as long
as their cumulative counters have not rolled over. Metrics that
reset on attachment (perf_event_open counters, BPF tracing
samples, etc.) are intentionally absent — they require long-lived
instrumentation the capture layer cannot install without
disturbing the system it is measuring.

### Capture is best-effort

Each internal reader returns `Option`; a kernel without
`CONFIG_SCHED_DEBUG` yields `None` from the schedstat reader
without failing the rest of the thread. Counters collapse to `0`,
identity strings collapse to empty, affinity collapses to an
empty vec. **A missing reading is indistinguishable from a
genuine zero in the output** — the contract is "never fail the
snapshot." Tests that need stronger guarantees inspect the
underlying readers directly (they remain `Option`-shaped and are
unit-tested in the module).

### Per-cgroup enrichment

Every cgroup at least one sampled thread resides in gets a
`CgroupStats` entry: `cpu_usage_usec`, `nr_throttled`,
`throttled_usec`, `memory_current` — read directly from
cgroup v2 files (`cpu.stat`, `memory.current`), NOT derived from
per-thread data, because those are aggregate-over-the-cgroup
values.

### Snapshot identity

The top-level `HostStateSnapshot` also embeds a `HostContext`
(the same structure `show-host` prints — kernel, CPU, memory,
sched_\* tunables, cmdline). Older tools or synthetic fixtures
that omit the context render `(host context unavailable)` rather
than failing the compare.

### Cgroup namespace caveat

The per-thread `cgroup` path is read verbatim from
`/proc/<tid>/cgroup` — it is therefore relative to the **cgroup
namespace root the capturing process sees**, NOT the
system-global v2 mount root. A process inside a nested cgroup
namespace sees a truncated path; a process outside sees a longer
one. Cross-namespace comparison requires external
canonicalization (the capture layer deliberately does not attempt
it because the right resolution depends on capture-site privilege
and namespace visibility).

## Compare

```sh
ktstr host-state compare before.hst.zst after.hst.zst
```

`compare` joins the two snapshots on `(pcomm, comm)` by default
and emits one row per `(group, metric)` pair. Groups present on
only one side surface as **unmatched** — a row is missing
because the process did not exist, not because it did zero work.

### Grouping

- `--group-by pcomm` (default) — aggregate every thread of the
  same process together.
- `--group-by cgroup` — aggregate by cgroup path. Useful for
  container-per-workload deployments where the process name is
  ambiguous across cgroups.
- `--group-by comm` — aggregate by thread name across every
  process. Useful when a thread-pool name like `tokio-worker`
  spans many binaries and you want one row per pool, not per
  binary.

### Cgroup-path flattening

```sh
ktstr host-state compare before.hst.zst after.hst.zst \
    --group-by cgroup \
    --cgroup-flatten '/kubepods/*/pod-*/container' \
    --cgroup-flatten '/system.slice/*.scope'
```

`--cgroup-flatten` accepts glob patterns that collapse dynamic
segments (pod UUIDs, session scopes, transient unit IDs) to a
canonical form before grouping, so the same logical workload
across two runs lands on the same row even if the kernel
assigned different UUIDs.

### Aggregation rules

Each metric declares its own aggregation rule
([`HOST_STATE_METRICS`] in `src/host_state_compare.rs`):

- **Sum** (most counters) — cumulative values add. Delta is the
  signed difference; percent delta is relative to the before-
  side.
- **OrdinalRange** (`nice`) — min / max across the group; the
  renderer shows `[min, max]` and the delta uses the midpoint
  so a shift on either end is visible.
- **Mode** (`policy`) — the most-common policy name and its
  share of the group. No scalar, so rows sort to the bottom of
  the default sort.
- **Affinity** (`cpu_affinity`) — aggregates into an
  `AffinitySummary` with `min_cpus` / `max_cpus` and a `uniform`
  flag. Heterogeneous groups render as `"N-M cpus (mixed)"`.

### Output and interpretation

The comparison prints **raw numbers and percent delta**. There
are no judgment labels (regression vs. improvement) — the
meaning of "run_time went up 15%" depends on whether you were
measuring a CPU-bound workload (more work done) or a spin-wait
pathology (more time wasted). The interpretation is scheduler-
specific and left to the operator.

Sort order: by default, rows are sorted by absolute delta
(largest movers first) so the most-changed metrics surface at
the top. Rows with no numeric scalar (`policy`, heterogeneous
affinity) fall to the bottom.

## File format

`.hst.zst` is zstd-compressed JSON of `HostStateSnapshot`. The
schema is `#[non_exhaustive]` so field additions do not break
existing snapshots:

```
HostStateSnapshot
├── captured_at_unix_ns: u64
├── host: Option<HostContext>
├── threads: Vec<ThreadState>
└── cgroup_stats: BTreeMap<String, CgroupStats>
```

`ThreadState::start_time_clock_ticks` is in USER_HZ (100 on
x86_64 and aarch64), NOT the kernel-internal CONFIG_HZ — so
cross-host comparison between differently-configured kernels on
those architectures is meaningful. Other in-tree architectures
(alpha, for instance, with USER_HZ=1024) would require normalization
at capture time; the capture layer currently targets x86_64 and
aarch64 only.

Compression level `3` (matching the ktstr remote-cache
convention): adequate ratio at fast speed, and host-state
captures are small enough that further compression produces
diminishing returns on I/O.

## Related

- [`cargo ktstr show-host`](../running-tests/cargo-ktstr.md) —
  captures the host *context* (kernel, CPU, tunables) that the
  profiler embeds as the `host` field. Use `show-host` when you
  want to inspect host configuration only, without the per-
  thread walk.
- [Capture and Compare Host State](../recipes/host-state.md) —
  recipe covering the `show-host` / `stats compare` flow for
  comparing host *context* across sidecars (not the per-thread
  profiler).
- [Environment Variables](environment-variables.md) — every
  ktstr-controlled env var.
