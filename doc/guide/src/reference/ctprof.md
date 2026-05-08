# ctprof

The ctprof profiler captures a host-wide per-thread snapshot
of scheduling counters, memory / I/O accounting, CPU affinity,
cgroup state, and thread identity, then compares two snapshots to
surface what changed. It is a manually-invoked CLI companion to
the automated scheduler tests — useful when a run passes on one
machine and fails on another, or for A/B comparing host behaviour
across kernel / sysctl / workload changes.

This is a **different tool** from `cargo ktstr show-host`,
which captures the host *context* (kernel, CPU model, sched_\*
tunables, NUMA layout, kernel cmdline) — aggregate state that
does not change between scenarios. The profiler captures
*per-thread* cumulative counters that do change, and its
comparison surface is designed for the thread-level diff.

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
`ktstr ctprof` subcommand.

## Capture

```sh
ktstr ctprof capture --output baseline.ctprof.zst
# ... run workload, change a tunable, reboot a kernel, etc. ...
ktstr ctprof capture --output after.ctprof.zst
```

`capture` walks `/proc` for every live thread group, enumerates
each thread, and reads a handful of procfs sources for each one.
The output is a zstd-compressed JSON snapshot (conventional
extension: `.ctprof.zst`).

### What is captured per thread

- **Identity** — tid, tgid, `pcomm` (process name from
  `/proc/<tgid>/comm`), `comm` (thread name from
  `/proc/<tid>/comm`), cgroup v2 path,
  `start_time_clock_ticks` (from `/proc/<tid>/stat` field 22,
  in USER_HZ clock ticks), scheduling policy name, nice, CPU
  affinity mask.
- **Scheduling counters** (cumulative, from `/proc/<tid>/sched`;
  schedstat fields gated by `CONFIG_SCHEDSTATS`,
  `run_time_ns`/`wait_time_ns`/`timeslices` gated by
  `CONFIG_SCHED_INFO`) — `run_time_ns`, `wait_time_ns`,
  `timeslices`, `voluntary_csw`, `nonvoluntary_csw`, `nr_wakeups`
  (plus `_local` / `_remote` / `_sync` / `_migrate` splits),
  `nr_migrations`, `wait_sum` / `wait_count`, `voluntary_sleep_ns`
  (capture-side normalized as `sum_sleep_runtime -
  sum_block_runtime` so the kernel's sleep/block double-count
  is stripped before the value reaches the snapshot),
  `block_sum`, `iowait_sum` / `iowait_count`,
  `core_forceidle_sum`, `wait_max` / `sleep_max` / `block_max` /
  `exec_max` / `slice_max` (lifetime peaks).
- **Memory** — `minflt` / `majflt` from `/proc/<tid>/stat`.
  `allocated_bytes` / `deallocated_bytes` from the jemalloc
  per-thread TSD counters (`tsd_s.thread_allocated` /
  `thread_deallocated`) read via ptrace + `process_vm_readv` —
  populated only for processes linked against jemalloc; glibc
  arena counters are opaque and read as zero rather than failing
  capture. `smaps_rollup_kb` (per-process map of the kernel's
  `/proc/<tid>/smaps_rollup` keys, populated leader-only).
- **I/O** — `rchar`, `wchar`, `syscr`, `syscw`, `read_bytes`,
  `write_bytes`, `cancelled_write_bytes` from `/proc/<tid>/io`
  (requires `CONFIG_TASK_IO_ACCOUNTING`). Note that
  `cancelled_write_bytes` records on the truncating task — not
  the original writer — so it pairs with `write_bytes` as a
  group-level signal but per-thread arithmetic between the two
  is not meaningful.
- **Taskstats delay accounting + watermarks** — eight delay
  categories × four fields each (count, total_ns, max_ns,
  min_ns) plus `hiwater_rss_bytes` and `hiwater_vm_bytes` peaks,
  pulled via the kernel's TASKSTATS genetlink family. Requires
  `CAP_NET_ADMIN` on the capturing process; delay-family fields
  additionally require `CONFIG_TASK_DELAY_ACCT` and the runtime
  `delayacct=on` toggle, watermark fields require
  `CONFIG_TASK_XACCT`. See the
  [Taskstats delay accounting](#taskstats-delay-accounting)
  section below for the full field list, gating, and per-bucket
  semantic caveats.
- **PSI host-level** — `cpu.stat` / `memory.current` aggregates
  per cgroup (see [Per-cgroup enrichment](#per-cgroup-enrichment))
  plus `psi` (Pressure Stall Information) under each cgroup and
  at the host level. Requires `CONFIG_PSI`.
- **sched_ext sysfs** — `state`, `switch_all`, `nr_rejected`,
  `hotplug_seq`, `enable_seq` from
  `/sys/kernel/sched_ext/`. Present only when
  `CONFIG_SCHED_CLASS_EXT` is built.

Field families and probe-timing invariance:

- **Cumulative counters and totals** (the majority): wakeups,
  migrations, csw, run/wait/sleep/block/iowait time, schedstat
  counts, page-fault counters, syscall counters, byte counters,
  the taskstats per-bucket `*_count` and `*_delay_total_ns`,
  the jemalloc per-thread TSD counters. Sampled twice at
  different instants the value increases monotonically; probe
  attachment time does not alter the reading.
- **Lifetime extrema**: schedstat `*_max` family
  (`wait_max`, `sleep_max`, `block_max`, `exec_max`,
  `slice_max`), every taskstats `*_delay_max_ns` /
  `*_delay_min_ns`, and the memory watermarks
  (`hiwater_rss_bytes`, `hiwater_vm_bytes`). Per-event
  extrema rather than sums. The `*_max` and `hiwater_*`
  fields are non-DECREASING over time (kernel keeps the
  largest); the `*_delay_min_ns` fields are non-INCREASING
  (kernel keeps the smallest non-zero observation, so
  sentinel 0 means "no events observed" — compare against
  the matching `*_count`).
- **Instantaneous gauges** (sensitive to probe timing):
  `nr_threads` (signal_struct->nr_threads snapshot),
  `fair_slice_ns` (current `p->se.slice`), and `state`
  (task_state_array letter). Sampled at capture time and can
  legitimately differ between two probes of the same thread.
- **Categorical / ordinal scalars**: `policy`, `nice`,
  `priority`, `processor`, `rt_priority`, plus identity strings
  (`pcomm`, `comm`, `cgroup`) and the `cpu_affinity` cpuset.
  Sampled at capture time and can change at runtime
  (e.g. `sched_setaffinity` mid-run flips `processor` and
  `cpu_affinity`), so they share the gauge family's
  probe-timing sensitivity.

Metrics that reset on attachment (perf_event_open counters, BPF
tracing samples, etc.) are intentionally absent — they require
long-lived instrumentation the capture layer cannot install
without disturbing the system it is measuring.

### Capture is best-effort

Each internal reader returns `Option`; a kernel without
`CONFIG_SCHED_DEBUG` yields `None` from the `/proc/<tid>/sched`
reader (and a kernel without `CONFIG_SCHEDSTATS` yields `None`
from `/proc/<tid>/schedstat` and the schedstat-gated
`/proc/<tid>/sched` keys) without failing the rest of the
thread. Counters collapse to `0`, identity strings collapse to
empty, affinity collapses to an empty vec. **A missing reading
is indistinguishable from a genuine zero in the output** — the
contract is "never fail the snapshot." Tests that need stronger
guarantees inspect the underlying readers directly (they remain
`Option`-shaped and are unit-tested in the module).

### Per-cgroup enrichment

Every cgroup at least one sampled thread resides in gets a
`CgroupStats` entry. Fields nest under per-controller
sub-structs:

- `cpu: CgroupCpuStats` — `usage_usec`, `nr_throttled`,
  `throttled_usec` (from `cpu.stat`); `max_quota_us`,
  `max_period_us` (from `cpu.max`); `weight`, `weight_nice`
  (from `cpu.weight` / `cpu.weight.nice`).
- `memory: CgroupMemoryStats` — `current` (from
  `memory.current`); `max`, `high`, `low`, `min` (from the
  matching `memory.*` files; `low` and `min` are protection
  floors, `max` and `high` are limits); `stat` and `events` as
  flat key-value maps mirroring `memory.stat` and
  `memory.events`.
- `pids: CgroupPidsStats` — `current` and `max` from the
  optional `pids` controller.
- `psi: Psi` — per-cgroup Pressure Stall Information from
  `<cgroup>/cpu.pressure` / `memory.pressure` / `io.pressure`
  / `irq.pressure` (gated on `CONFIG_PSI`).

All fields are read directly from cgroup v2 files, NOT derived
from per-thread data, because those are aggregate-over-the-cgroup
values.

### Snapshot identity

The top-level `CtprofSnapshot` also embeds a `HostContext`
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

### Taskstats delay accounting

The kernel's TASKSTATS genetlink family delivers per-task
delay-accounting and memory-watermark fields that are NOT
exposed via `/proc/<tid>/sched` or `/proc/<tid>/stat`. ctprof
captures them through `crate::taskstats` — a netlink socket
opens, the family-id resolves via `CTRL_CMD_GETFAMILY`, and one
`TASKSTATS_CMD_GET` query per tid is issued. The 34 captured
fields (8 delay categories × 4 bucket fields + 2 watermarks) all
tag `Section::TaskstatsDelay` so they can be filtered as a
unit.

#### Capability and kconfig gating

Calling the netlink family requires **`CAP_NET_ADMIN`** on the
capturing process (`kernel/taskstats.c::taskstats_ops` registers
`TASKSTATS_CMD_GET` with `GENL_ADMIN_PERM`). ktstr always runs
as root in production so the cap is implicit, but a non-root
operator running `ktstr ctprof capture` will hit `EPERM` on the
first `query_tid` call and every taskstats field will collapse
to zero per the best-effort capture contract.

Per-family kconfig gates and runtime toggles:

- **Delay-accounting fields** (`*_delay_count`, `*_delay_total_ns`,
  `*_delay_max_ns`, `*_delay_min_ns` across the eight
  categories): require `CONFIG_TASKSTATS=y` AND
  `CONFIG_TASK_DELAY_ACCT=y` AND the runtime `delayacct=on`
  toggle (sysctl `kernel.task_delayacct=1` or boot param
  `delayacct`). The runtime toggle is a separate condition
  beyond the build-time gates — a kernel built with both
  CONFIGs but launched without `delayacct=on` produces
  all-zero delay readings. ktstr's standard kernel build
  includes both kconfigs; the test harness adds `delayacct` to
  the guest cmdline.
- **Memory-watermark fields** (`hiwater_rss_bytes`,
  `hiwater_vm_bytes`): require `CONFIG_TASKSTATS=y` AND
  `CONFIG_TASK_XACCT=y`. They do NOT respond to the
  `delayacct=on` runtime toggle — `xacct_add_tsk`
  (`kernel/tsacct.c`) is unconditional once `CONFIG_TASK_XACCT`
  is built. `xacct_add_tsk` reads watermarks from the SHARED
  `mm_struct`, so sibling threads of the same tgid all report
  identical values; kernel threads (`mm == NULL`) read zero by
  design.

Any failed gate or missing cap collapses the affected fields
to zero. ktstr's capture pipeline emits an info-level tracing
line per snapshot summarizing taskstats outcomes AND attaches
the structured tally to `CtprofSnapshot::taskstats_summary`
(`ok_count` / `eperm_count` / `esrch_count` /
`other_err_count`), so an operator can distinguish "kernel
doesn't expose this" from "every tid raced exit" from
"`CAP_NET_ADMIN` missing" without scraping log lines.

#### Eight delay categories

| Category | Source | Notes |
|---|---|---|
| `cpu_delay_*` | `tsk->sched_info.{pcount,run_delay}` via `delayacct_add_tsk` (`kernel/delayacct.c`) | Time waiting on the runqueue. **RACY**: count + total are not updated atomically (lockless `sched_info` path); a concurrent reader may observe one ahead of the other. Captures the same wait-for-CPU bucket as schedstat `wait_*` via a different code path. |
| `blkio_delay_*` | `delayacct_blkio_start` / `_end` (`kernel/delayacct.c`) | Synchronous block I/O wait. Updates serialize through `task->delays->lock` so count + total are atomic (unlike `cpu_*`). The canonical delay-accounting block-I/O reading; distinct from schedstat `iowait_sum`. |
| `swapin_delay_*` | `delayacct_swapin_start` / `_end` (`include/linux/delayacct.h`) | Swap-in wait. **OVERLAPS** with `thrashing_*` — every thrashing event is also a swapin event from the syscall layer; do not sum the two. |
| `freepages_delay_*` | `delayacct_freepages_start` / `_end` (`mm/page_alloc.c`) | Direct memory reclaim wait. |
| `thrashing_delay_*` | `delayacct_thrashing_start` / `_end` (`mm/workingset.c`) | Thrashing wait. Refines swapin tracking — see `swapin_*`. |
| `compact_delay_*` | `delayacct_compact_start` / `_end` (`mm/compaction.c`) | Memory-compaction wait. |
| `wpcopy_delay_*` | `delayacct_wpcopy_start` / `_end` (`mm/memory.c`) | Write-protect-copy (CoW) fault wait. Introduced in taskstats v13. |
| `irq_delay_*` | `delayacct_irq` (`kernel/delayacct.c`) | IRQ-handler windows charged to the task by IRQ accounting. Introduced in taskstats v14. |

Each category has four fields:

- `*_count` — number of windows observed (`MonotonicCount`,
  `SumCount`).
- `*_delay_total_ns` — cumulative ns of delay (`MonotonicNs`,
  `SumNs`).
- `*_delay_max_ns` — longest single window observed
  (`PeakNs`, `MaxPeak`).
- `*_delay_min_ns` — shortest non-zero window observed
  (`PeakNs`, `MaxPeak`). **Sentinel 0 means "no events
  observed"**, NOT "saw a zero-ns event"; compare against the
  matching `*_count` to disambiguate.

The two memory watermarks (`hiwater_rss_bytes`,
`hiwater_vm_bytes`) are `PeakBytes` / `MaxPeakBytes` — see the
`MaxPeakBytes` row in the
[Aggregation rules](#aggregation-rules) section below for the
shared-mm semantics.

## Compare

```sh
ktstr ctprof compare before.ctprof.zst after.ctprof.zst
```

`compare` joins the two snapshots on `pcomm` (process name) by
default — see [Grouping](#grouping) for the other axes —
and emits one row per `(group, metric)` pair. Groups present
on only one side surface as **unmatched** — a row is missing
because the process did not exist, not because it did zero work.

### Grouping

- `--group-by pcomm` (default) — aggregate every thread of the
  same process together.
- `--group-by cgroup` — aggregate by cgroup path. Useful for
  container-per-workload deployments where the process name is
  ambiguous across cgroups.
- `--group-by comm` — aggregate by thread name across every
  process under token-based pattern normalization
  (`tokio-worker-{0..N}` → one bucket;
  `kworker/0:1H-events_highpri`,
  `kworker/1:0H-events_highpri`, … → one bucket). Useful when a
  thread-pool name spans many binaries and you want one row per
  pool, not per binary. Disable normalization with
  `--no-thread-normalize`.
- `--group-by comm-exact` — synonym for
  `--group-by comm --no-thread-normalize`. Aggregate by literal
  thread name, no pattern collapse. Use when distinct token
  values carry meaning (e.g. tracking each `kworker/u8:N`
  independently).

### Cgroup-path flattening

```sh
ktstr ctprof compare before.ctprof.zst after.ctprof.zst \
    --group-by cgroup \
    --cgroup-flatten '/kubepods/*/pod-*/container' \
    --cgroup-flatten '/system.slice/*.scope'
```

`--cgroup-flatten` accepts glob patterns that collapse dynamic
segments (pod UUIDs, session scopes, transient unit IDs) to a
canonical form before grouping, so the same logical workload
across two runs lands on the same row even if the kernel
assigned different UUIDs.

### Filtering output: `--sections` vs `--metrics`

Two complementary filters narrow the rendered output:

- **`--sections`** picks which **sub-tables** render. The
  default-empty value renders every section that has data;
  passing a comma-separated list restricts output to the named
  sub-tables — every section not listed is suppressed before
  its data-availability gate runs. Valid section names:
  `primary`, `taskstats-delay`, `derived`, `cgroup-stats`,
  `cgroup-limits`, `memory-stat`, `memory-events`, `pressure`,
  `host-pressure`, `smaps-rollup`, `sched-ext`. Five
  (`cgroup-stats`, `cgroup-limits`, `memory-stat`,
  `memory-events`, `pressure`) require `--group-by cgroup`;
  naming any of them under a non-cgroup grouping emits a
  stderr warning and renders zero rows.
- **`--metrics`** picks which **rows** render inside the
  primary and derived sub-tables. The default-empty value
  renders every metric; passing a comma-separated list
  restricts the rendered rows to the named metrics. Names must
  come from the `ctprof metric-list` vocabulary
  (`CTPROF_METRICS` ∪ `CTPROF_DERIVED_METRICS`). Has no effect
  on the secondary sub-tables (cgroup-stats, smaps-rollup,
  etc.) — those have fixed column shapes and ignore the row
  filter.

The two compose multiplicatively: `--sections primary
--metrics run_time_ns` shows a single row in the primary
sub-table and nothing else. `--sections primary` alone keeps
every primary row; `--metrics run_time_ns` alone keeps the
single row across every section that displays it.

Each metric carries exactly one `Section` tag in its
registry entry — the 34 taskstats-sourced primary rows and
the 9 taskstats-derived rows tag `Section::TaskstatsDelay`
rather than `Section::Primary` / `Section::Derived`. They
render inside the same primary / derived outer tables but
match a distinct section name, so `--sections taskstats-delay`
selects exactly the 34 + 9 taskstats rows alone, while
`--sections primary` excludes them and `--sections derived`
excludes the 9 taskstats derivations. The three-way split
lets an operator scope to non-taskstats only, taskstats
only, or any combination, without losing the visual grouping
under the same outer headers.

### Aggregation rules

Each metric declares its own aggregation rule
(`CTPROF_METRICS` in `src/ctprof_compare.rs`). The
`AggRule` enum is **typed**: each variant binds an accessor of a
specific `metric_types` newtype (`MonotonicCount`,
`MonotonicNs`, `PeakNs`, `Bytes`, etc.) so a registry entry that
pairs a peak field with a sum reduction (e.g. `t.wait_max`
(`PeakNs`) bound to a `Sum*` rule) fails to compile rather
than producing a meaningless `1×1s ⊕ 1000×1ms` aggregate. The
14 variants split into five families: Sum reductions, Max
reductions, Range reductions, Mode reductions, and the
Affinity reduction.

#### Sum reductions (cumulative counters)

| Variant | Newtype | Output unit | Examples |
|---|---|---|---|
| `SumCount` | `MonotonicCount` | unitless | `nr_wakeups` (+ `_local` / `_remote` / `_sync` / `_migrate` / `_affine` / `_affine_attempts`), `nr_migrations`, `nr_forced_migrations`, `nr_failed_migrations_*`, `voluntary_csw`, `nonvoluntary_csw`, `minflt`, `majflt`, `wait_count`, `iowait_count`, `timeslices`, `syscr`, `syscw`, every taskstats `*_delay_count` (8 entries) |
| `SumNs` | `MonotonicNs` | ns | `run_time_ns`, `wait_time_ns`, `wait_sum`, `voluntary_sleep_ns`, `block_sum`, `iowait_sum`, `core_forceidle_sum`, every taskstats `*_delay_total_ns` (8 entries) |
| `SumTicks` | `ClockTicks` | USER\_HZ ticks | `utime_clock_ticks`, `stime_clock_ticks` |
| `SumBytes` | `Bytes` | bytes (IEC) | `allocated_bytes`, `deallocated_bytes`, `rchar`, `wchar`, `read_bytes`, `write_bytes`, `cancelled_write_bytes` |

Group reduction: `saturating_add` per the no-wraparound contract.
Delta is the signed difference; percent delta is relative to the
before-side. Auto-scale ladder is decimal SI for ns / count,
USER\_HZ for ticks, IEC binary for bytes.

#### Max reductions (peaks and gauges)

| Variant | Newtype | Output unit | Examples |
|---|---|---|---|
| `MaxPeak` | `PeakNs` | ns | `wait_max`, `sleep_max`, `block_max`, `exec_max`, `slice_max`, every taskstats `*_delay_max_ns` (8 entries), every taskstats `*_delay_min_ns` (8 entries) |
| `MaxPeakBytes` | `PeakBytes` | bytes (IEC) | `hiwater_rss_bytes`, `hiwater_vm_bytes` (taskstats lifetime memory watermarks) |
| `MaxGaugeNs` | `GaugeNs` | ns | `fair_slice_ns` (current scheduler slice) |
| `MaxGaugeCount` | `GaugeCount` | unitless | `nr_threads` (process-wide thread count) |

`MaxPeak` / `MaxPeakBytes` rows surface the worst single window
or largest watermark any thread in the group has ever observed
— summing per-thread maxes would conflate "one thread with a 1s
spike" with "1000 threads with 1ms spikes each".
`MaxPeakBytes` is the byte-typed twin of `MaxPeak` and routes
through the IEC binary auto-scale ladder so a 7.5 GiB watermark
renders as `7.500GiB` rather than dominating the table with raw
byte counts. `xacct_add_tsk` (`kernel/tsacct.c`) reads the
watermarks from the SHARED `mm_struct`, so sibling threads of
the same tgid all report the same value; cross-thread Max
within a single process is a no-op, while cross-process Max
under a multi-tgid bucket picks the largest watermark any tgid
in the bucket reported.

`MaxGaugeNs` / `MaxGaugeCount` apply to instantaneous gauges
(read at capture time) where summing has no physical meaning.
`nr_threads` specifically is leader-only (populated on
`tid == tgid`, zero elsewhere); `Max` reads through the leader
so a comm-bucketed group still surfaces the largest process
represented in the bucket. The taskstats `*_delay_min_ns` rows
also use `MaxPeak`: `min` here is the kernel's per-task lifetime
shortest non-zero observation, so cross-thread Max picks "the
largest minimum any contributor reported"; sentinel 0 means
"no events observed" — compare against the matching count.

#### Range reductions (bounded ordinals)

| Variant | Newtype | Output | Examples |
|---|---|---|---|
| `RangeI32` | `OrdinalI32` | `[min, max]` (i64-widened) | `nice`, `priority`, `processor` |
| `RangeU32` | `OrdinalU32` | `[min, max]` (i64-widened) | `rt_priority` |

The renderer shows `[min, max]` and the delta uses the midpoint
so a shift on either end is visible.

#### Mode reductions (categorical)

| Variant | Newtype | Output | Examples |
|---|---|---|---|
| `Mode` | `CategoricalString` | most-frequent value + count/total | `policy` |
| `ModeChar` | `char` (coerced) | most-frequent char + count/total | `state` |
| `ModeBool` | `bool` (coerced) | most-frequent bool + count/total | `ext_enabled` |

Mode is textual: delta is `"same"` if both modes agree,
`"differs"` otherwise — there is no arithmetic on a categorical
value. `ModeChar` and `ModeBool` coerce to `String` via
`to_string()` before reducing because the underlying types are
not themselves `Modeable`. A 50/50 bool tie resolves
lex-smallest-wins (so `"false"` wins over `"true"`); operators
reading a `false` mode in a heterogeneous bucket should check
the `count/total` fraction.

#### Affinity reduction (CPU sets)

| Variant | Newtype | Output | Example |
|---|---|---|---|
| `Affinity` | `CpuSet` | `AffinitySummary { min_cpus, max_cpus, uniform }` | `cpu_affinity` |

Heterogeneous groups render as `"N-M cpus (mixed)"`. Unlike the
other rules, `Affinity` does not route through a
`metric_types` trait — its reduction produces a structured
summary, not a homogeneous newtype.

[`metric_types`]: https://likewhatevs.github.io/ktstr/api/ktstr/metric_types/index.html
[`PeakNs`]: https://likewhatevs.github.io/ktstr/api/ktstr/metric_types/struct.PeakNs.html

### Derived metrics

Derived metrics consume one or more already-aggregated input
metrics from `CTPROF_METRICS` and produce a single scalar
with its own auto-scale ladder. They render in a separate
`## Derived metrics` table below the per-thread table on both
`compare` and `show`, with rows colored blue to distinguish
them from the primary table on TTY stdout. Registered in
`CTPROF_DERIVED_METRICS` in `src/ctprof_compare.rs`.

The full registry is 17 entries: 8 schedstat / I/O / heap
derivations plus 9 taskstats-derived (the 8 per-bucket
`avg_*_delay_ns` averages plus the `total_offcpu_delay_ns`
rollup). Every formula is implemented as a closure over the
group's metrics map (`BTreeMap<String, Aggregated>`); a missing
input or a zero denominator yields `None`, which the renderer
surfaces as `-` so the operator can distinguish "not
computable" from "computed as zero".

| Metric | Formula | Inputs | Unit | Notes |
|---|---|---|---|---|
| `affine_success_ratio` | `nr_wakeups_affine / nr_wakeups_affine_attempts` | `nr_wakeups_affine`, `nr_wakeups_affine_attempts` | ratio (0..1) | `wake_affine()` success ratio. CFS-only signal — sched_ext does not increment the wakeup counters. Bare three-decimal scalar; the renderer suppresses the `%` column for ratio rows because absolute delta on a `[0, 1]` ratio is already in percentage points. |
| `avg_wait_ns` | `wait_sum / wait_count` | `wait_sum`, `wait_count` | ns | Average runqueue-wait duration per scheduling event. Rendered with the ns auto-scale ladder (ns → µs → ms → s). Schedstat-gated (see `wait_sum` and `wait_count`); zero across sched\_ext threads. |
| `cpu_efficiency` | `run_time_ns / (run_time_ns + wait_time_ns)` | `run_time_ns`, `wait_time_ns` | ratio (0..1) | Fraction of total scheduler-tracked time spent on-CPU. Higher = less time stuck on the runqueue. Both inputs gated by `CONFIG_SCHED_INFO`. |
| `avg_slice_ns` | `run_time_ns / timeslices` | `run_time_ns`, `timeslices` | ns | Average on-CPU slice length. Useful for spotting timeslice-tuning regressions (e.g. an `sched_min_granularity_ns` change that shrinks slices). Both inputs gated by `CONFIG_SCHED_INFO`. |
| `involuntary_csw_ratio` | `nonvoluntary_csw / (voluntary_csw + nonvoluntary_csw)` | `nonvoluntary_csw`, `voluntary_csw` | ratio (0..1) | Fraction of context switches that were preemptions (kernel pulled the task off-CPU) vs. voluntary blocks. High values indicate preemption pressure; low values indicate cooperative blocking. |
| `disk_io_fraction` | `read_bytes / rchar` | `read_bytes`, `rchar` | ratio (≥ 0) | Fraction of read syscall bytes that traveled past the pagecache layer (cache miss rate; covers local block devices and network filesystems alike). Typically ≤ 1.0, but **can exceed 1** when readahead pulls more bytes past the pagecache layer than the syscall requested. Both inputs gated by `CONFIG_TASK_IO_ACCOUNTING`. |
| `live_heap_estimate` | `allocated_bytes - deallocated_bytes` (signed) | `allocated_bytes`, `deallocated_bytes` | bytes (IEC, signed) | jemalloc-only live-heap estimate. Glibc and other allocators feed both inputs zero so the derived metric reads zero too — `-` would imply non-computable but here zero is the genuine reading. Renders on the IEC binary ladder (B → KiB → MiB → GiB → TiB). Per-thread reading carries cross-thread noise: a thread that purely frees objects allocated by other threads reads large negative values; group-level Sum across all threads of the process eliminates the asymmetry. |
| `avg_iowait_ns` | `iowait_sum / iowait_count` | `iowait_sum`, `iowait_count` | ns | Average iowait interval per blocking event. Schedstat-gated; zero across sched\_ext threads. |
| `avg_cpu_delay_ns` | `cpu_delay_total_ns / cpu_delay_count` | `cpu_delay_total_ns`, `cpu_delay_count` | ns | Average runqueue-wait per scheduling event from the taskstats delayacct path. **RACY**: the kernel updates count + total via the lockless `sched_info` path, so a concurrent reader may observe one ahead of the other; the quotient is approximate at the sub-event scale and stable at the integrated scale. Distinct from `avg_wait_ns` (schedstat) which captures the same wait-for-CPU bucket via a different code path. |
| `avg_blkio_delay_ns` | `blkio_delay_total_ns / blkio_delay_count` | `blkio_delay_total_ns`, `blkio_delay_count` | ns | Average synchronous block-I/O wait per event from the taskstats delayacct path. Distinct from `avg_iowait_ns` (schedstat) — this is the canonical delay-accounting block-I/O reading. |
| `avg_swapin_delay_ns` | `swapin_delay_total_ns / swapin_delay_count` | `swapin_delay_total_ns`, `swapin_delay_count` | ns | Average swap-in wait per event. **OVERLAPS with thrashing** — every thrashing event is also a swapin event from the syscall layer; do not sum the two averages or the underlying totals directly. |
| `avg_freepages_delay_ns` | `freepages_delay_total_ns / freepages_delay_count` | `freepages_delay_total_ns`, `freepages_delay_count` | ns | Average direct-reclaim wait per event. |
| `avg_thrashing_delay_ns` | `thrashing_delay_total_ns / thrashing_delay_count` | `thrashing_delay_total_ns`, `thrashing_delay_count` | ns | Average thrashing wait per event. **OVERLAPS with swapin** (see `avg_swapin_delay_ns`). |
| `avg_compact_delay_ns` | `compact_delay_total_ns / compact_delay_count` | `compact_delay_total_ns`, `compact_delay_count` | ns | Average memory-compaction wait per event. |
| `avg_wpcopy_delay_ns` | `wpcopy_delay_total_ns / wpcopy_delay_count` | `wpcopy_delay_total_ns`, `wpcopy_delay_count` | ns | Average write-protect-copy (CoW) fault wait per event. |
| `avg_irq_delay_ns` | `irq_delay_total_ns / irq_delay_count` | `irq_delay_total_ns`, `irq_delay_count` | ns | Average IRQ-handler window per event. |
| `total_offcpu_delay_ns` | `cpu + blkio + freepages + compact + wpcopy + irq + max(swapin, thrashing)` | every `*_delay_total_ns` | ns | Sum of every meaningful off-CPU delay-accounting bucket. The swapin/thrashing pair is OR'd with `.max()` rather than summed because the two share syscall-layer events (every thrashing event is also a swapin from the syscall perspective); summing both would double-count thrashing-induced swapins. When `CONFIG_TASK_DELAY_ACCT` is off, the runtime toggle is off, or the kernel predates a bucket's introduction (e.g. `wpcopy_*` lands in v13, `irq_*` in v14), the missing buckets read zero from the truncated taskstats payload — the rollup degrades to the sum of the populated buckets rather than returning `-`. The structured taskstats outcome lives on `CtprofSnapshot::taskstats_summary` for the operator to disambiguate "no data" from "zero data." |

The `is_ratio` column on the registry is load-bearing for the
renderer: ratio rows skip the `%` column entirely (the absolute
delta already carries percentage-point semantics for a `[0, 1]`
quantity), and the auto-scale ladder is `None` (bare three-
decimal scalar). Non-ratio derived metrics reuse the same
ladders as their unit family — `Ns` for nanosecond derivations,
`Bytes` for byte derivations.

The 9 taskstats-derived entries (the 8 `avg_*_delay_ns`
averages plus `total_offcpu_delay_ns`) tag
`Section::TaskstatsDelay` rather than `Section::Derived` so
`--sections taskstats-delay` renders the full taskstats view —
the 34 raw rows AND the 9 derivations that depend on them —
without dragging in unrelated derivations.

Derived metrics are surfaced by `ctprof metric-list`
alongside the primary registry, and are valid `--sort-by` keys
on both `compare` and `show`.

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

`.ctprof.zst` is zstd-compressed JSON of `CtprofSnapshot`. The
schema is `#[non_exhaustive]` so field additions do not break
existing snapshots:

```text
CtprofSnapshot
├── captured_at_unix_ns: u64
├── host: Option<HostContext>
├── threads: Vec<ThreadState>
├── cgroup_stats: BTreeMap<String, CgroupStats>
├── probe_summary: Option<CtprofProbeSummary>
├── parse_summary: Option<CtprofParseSummary>
├── taskstats_summary: Option<TaskstatsSummary>
├── psi: Psi
└── sched_ext: Option<SchedExtSysfs>
```

`TaskstatsSummary` carries per-snapshot taskstats genetlink
query outcomes — `ok_count`, `eperm_count`, `esrch_count`,
`other_err_count` — so an operator can distinguish "no
taskstats data because every tid raced exit" (high
`esrch_count`) from "no taskstats data because the kernel was
built without `CONFIG_TASKSTATS`" (the netlink open failed
up-front, every counter zero) from "no taskstats data because
`CAP_NET_ADMIN` is missing" (high `eperm_count`).

`ThreadState::start_time_clock_ticks` is in USER_HZ (100 on
x86_64 and aarch64), NOT the kernel-internal CONFIG_HZ — so
cross-host comparison between differently-configured kernels on
those architectures is meaningful. Other in-tree architectures
(alpha, for instance, with USER_HZ=1024) would require normalization
at capture time; the capture layer currently targets x86_64 and
aarch64 only.

Compression level `3` (matching the ktstr remote-cache
convention): adequate ratio at fast speed, and ctprof
captures are small enough that further compression produces
diminishing returns on I/O.

## Adding a metric

Adding a per-thread metric to the registry is a three-step
mechanical process. The type system enforces the wiring so a
mismatch between the kernel-source semantic and the aggregation
rule fails to compile rather than producing a silently-wrong
group reduction.

### 1. Add a `ThreadState` field with the right newtype

Pick the `metric_types` newtype that matches the kernel-source
semantic of the field — the per-newtype docs name the kernel
call sites that update each category. The shape determines what
aggregation rules are legal in step 3:

| Newtype | When to use |
|---|---|
| `MonotonicCount` | Pure counter — only goes up across the thread's lifetime. Examples: `nr_wakeups`, syscall counts, every taskstats `*_delay_count`. |
| `DeadCounter` | Same shape as `MonotonicCount` but tagged for kernel counters with no live writer (always reads zero). Captured for parser parity but does NOT implement any reduction trait — register with `is_dead: true` and the renderer flags it `[dead]`. |
| `MonotonicNs` | Cumulative-time counter in ns. Examples: `run_time_ns`, `wait_sum`, every taskstats `*_delay_total_ns`. |
| `PeakNs` | Lifetime high-water mark in ns. Kernel updates via `if (delta > stat->max) stat->max = delta`. Summing peaks is a category error. Examples: `wait_max`, `slice_max`, every taskstats `*_delay_max_ns` and `*_delay_min_ns`. |
| `PeakBytes` | Byte-typed twin of `PeakNs` — lifetime high-water mark in bytes. Routes through the IEC binary auto-scale ladder. Used for taskstats memory watermarks (`hiwater_rss_bytes`, `hiwater_vm_bytes`) read from the shared `mm_struct`. Pairs with `AggRule::MaxPeakBytes`. |
| `GaugeNs` | Instantaneous gauge sampled at capture time (ns). Cannot sum — N near-identical samples collapse to N×gauge with no meaning. Example: `fair_slice_ns`. |
| `GaugeCount` | Instantaneous unitless count that goes up AND down. Example: `nr_threads`. |
| `ClockTicks` | USER_HZ-scaled time. Examples: `utime_clock_ticks`, `stime_clock_ticks`. |
| `Bytes` | Byte counts. IEC binary auto-scale ladder. Examples: `read_bytes`, `wchar`. |
| `OrdinalI32` / `OrdinalU32` / `OrdinalU64` | Bounded scalar — range-aggregated, not summable. Examples: `nice` (i32), `rt_priority` (u32). The `Rangeable::range_across` reduction returns `Option<Range<Self>>` — see `Range<T>` below. `OrdinalU64` implements `Rangeable` but is currently unused in the registry; a metric that picks `OrdinalU64` requires adding `AggRule::RangeU64` alongside the existing `RangeI32` and `RangeU32` variants. |
| `CategoricalString` | Categorical value — mode-aggregated. Examples: `policy`. |
| `CpuSet` | CPU affinity mask — affinity-aggregated. Example: `cpu_affinity`. |
| `Range<T>` | Output type of the `Rangeable::range_across` reduction. Carries `min` and `max` of the same `T` with the `min <= max` invariant enforced at construction (`debug_assert!` in `Range::new`). Not stored on `ThreadState` — the `Aggregated::OrdinalRange` boundary unwraps it via `into_tuple()` to a `(i64, i64)` pair widened from the underlying `OrdinalI32` / `OrdinalU32` / `OrdinalU64`. |

Add the field to `ThreadState` in `src/ctprof.rs`:

```rust,ignore
// In ThreadState struct definition.
/// Description: what the field counts, what kernel call site
/// writes it, and what scheduler classes increment it. Cite
/// `kernel/sched/...` line numbers for the writer.
pub my_new_metric: crate::metric_types::MonotonicCount,
```

### 2. Wire the capture path

`capture_thread_at_with_tally` in `src/ctprof.rs` is the
single per-thread procfs walk. Add the per-source reader (or
extend an existing one) and stamp the field in the
`ThreadState { ... }` construction:

```rust,ignore
// Inside capture_thread_at_with_tally, after the existing
// per-source reads. Wrap in the newtype constructor; never use
// `.into()` (the typed-newtype style is explicit).
my_new_metric: MonotonicCount(sched.my_new_metric.unwrap_or(0)),
```

The `Option::unwrap_or(0)` collapse is load-bearing: the
profiler's contract is "never fail the snapshot," so a missing
reading lands at the newtype's `Default::default()` (zero). The
absent reading is indistinguishable from a genuine zero in the
output — see the *Capture is best-effort* section.

### 3. Register the metric

Append a `CtprofMetricDef` entry to `CTPROF_METRICS` in
`src/ctprof_compare.rs`. The `AggRule` variant must match the
newtype chosen in step 1 — the type system enforces this.

```rust,ignore
CtprofMetricDef {
    name: "my_new_metric",
    rule: AggRule::SumCount(|t| t.my_new_metric),
    sched_class: None, // or Some("cfs-only") / Some("non-ext") / Some("fair-policy")
    config_gates: &[], // or &["CONFIG_SCHEDSTATS"], etc.
    is_dead: false,    // true for kernel-side dead pointers
    description: "One-line operator-facing description; surfaces in `ctprof metric-list`.",
    section: Section::Primary, // or Section::TaskstatsDelay for taskstats-sourced rows
},
```

The `name` field is the canonical metric identifier — used by
`--sort-by`, `--metrics`, and the `metric-list` output. (The
`--columns` flag accepts layout names — `group`, `threads`,
`metric`, `baseline`, `candidate`, `delta`, `%`, `arrow`,
`value` — not metric names.) Names are ASCII short-form
(matching the capture-side field name where possible).
`sched_class` and `config_gates` render as bracketed suffixes
in `metric-list` output (`[cfs-only]`, `[SCHEDSTATS]`) so
operators reading a row know which kernels populate the
counter. The `section` tag drives the `--sections` per-row
filter — most rows take `Section::Primary`; taskstats-sourced
rows take `Section::TaskstatsDelay`.

### Compile-time guards

The type system catches the four most common mistakes:

- **Wrong reduction family**: pairing a `PeakNs` accessor with
  `AggRule::SumNs` fails with a type error — `PeakNs` does not
  implement `Summable` (only `Maxable`), and the closure's
  return type does not match the variant's expected newtype.
- **Wrong unit family**: pairing a `Bytes` accessor with
  `AggRule::SumNs` fails the same way.
- **Dead counter with live reduction**: `DeadCounter` does not
  implement `Summable` / `Maxable` / `Rangeable` / `Modeable`,
  so any `AggRule::Sum*` / `Max*` / `Range*` / `Mode*` variant
  bound to a dead-counter accessor fails to compile. Register
  the metric only via the `is_dead: true` flag with whichever
  variant matches its shape — the rendering layer surfaces it
  as `[dead]` and skips numeric reduction.
- **Categorical with numeric reduction**: pairing a
  `CategoricalString` accessor with `AggRule::SumCount` fails
  because `CategoricalString` does not implement `Summable`.

The closure body cannot be type-checked beyond the variant
boundary, so a body that actively miswraps a field — e.g.
`SumNs(|t| MonotonicNs(t.wait_max.0))` laundering a peak through
the sum wrapper — type-checks. Don't do that. The wrapper
category is load-bearing; the type system catches the variant
mismatch but not the lying inside.

### Optional: derived metric

If the new metric has a useful ratio or sum-of-ratios pairing
with existing inputs, register a `DerivedMetricDef` in
`CTPROF_DERIVED_METRICS` (same file). The `compute` closure
reads inputs via `input_scalar(metrics, name)?` and returns
`Option<DerivedValue>`; the `ratio_compute` and
`ratio_of_sum_compute` helpers cover the two most common
shapes. Set `is_ratio: true` when the output is in `[0, 1]` so
the renderer suppresses the `%` column. Set `section` to
`Section::Derived` for general derivations or
`Section::TaskstatsDelay` if every input is a taskstats field
(so `--sections taskstats-delay` keeps the derivation alongside
its raw inputs).

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
