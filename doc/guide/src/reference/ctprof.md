# ctprof

The ctprof profiler captures a host-wide per-thread snapshot
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
  `/proc/<tid>/comm`), cgroup v2 path, `start_time` in USER_HZ
  clock ticks, scheduling policy name, nice, CPU affinity mask.
- **Scheduling counters** (cumulative, from `/proc/<tid>/sched`;
  requires `CONFIG_SCHED_DEBUG`) — `run_time_ns`, `wait_time_ns`,
  `timeslices`, `voluntary_csw`, `nonvoluntary_csw`, `nr_wakeups`
  (plus `_local` / `_remote` / `_sync` / `_migrate` splits),
  `nr_migrations`, `wait_sum` / `wait_count`, `voluntary_sleep_ns`
  (capture-side normalized as `sum_sleep_runtime -
  sum_block_runtime` so the kernel's sleep/block double-count
  is stripped before the value reaches the snapshot),
  `block_sum`, `iowait_sum` / `iowait_count`.
- **Memory** — `minflt` / `majflt` from `/proc/<tid>/stat`.
  `allocated_bytes` / `deallocated_bytes` from the jemalloc
  per-thread-destructor TSD cache — populated only for processes
  linked against jemalloc; glibc arena counters are opaque and
  read as zero rather than failing capture.
- **I/O** — `rchar`, `wchar`, `syscr`, `syscw`, `read_bytes`,
  `write_bytes`, `cancelled_write_bytes` from `/proc/<tid>/io`
  (requires `CONFIG_TASK_IO_ACCOUNTING`). Note that
  `cancelled_write_bytes` records on the truncating task — not
  the original writer — so it pairs with `write_bytes` as a
  group-level signal but per-thread arithmetic between the two
  is not meaningful.

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

## Compare

```sh
ktstr ctprof compare before.ctprof.zst after.ctprof.zst
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

### Aggregation rules

Each metric declares its own aggregation rule
([`CTPROF_METRICS`] in `src/ctprof_compare.rs`). The
`AggRule` enum is **typed**: each variant binds an accessor of a
specific [`metric_types`] newtype (`MonotonicCount`,
`MonotonicNs`, `PeakNs`, `Bytes`, etc.) so a registry entry that
pairs a peak field with a sum reduction (e.g. `t.wait_max`
([`PeakNs`]) bound to a `Sum*` rule) fails to compile rather
than producing a meaningless `1×1s ⊕ 1000×1ms` aggregate. The
13 variants split into four families:

#### Sum reductions (cumulative counters)

| Variant | Newtype | Output unit | Examples |
|---|---|---|---|
| `SumCount` | `MonotonicCount` | unitless | `nr_wakeups`, `voluntary_csw`, `minflt`, `wait_count`, `iowait_count`, `timeslices` |
| `SumNs` | `MonotonicNs` | ns | `run_time_ns`, `wait_time_ns`, `wait_sum`, `voluntary_sleep_ns`, `block_sum`, `iowait_sum`, `core_forceidle_sum` |
| `SumTicks` | `ClockTicks` | USER\_HZ ticks | `utime_clock_ticks`, `stime_clock_ticks` |
| `SumBytes` | `Bytes` | bytes (IEC) | `allocated_bytes`, `deallocated_bytes`, `rchar`, `wchar`, `read_bytes`, `write_bytes`, `cancelled_write_bytes` |

Group reduction: `saturating_add` per the no-wraparound contract.
Delta is the signed difference; percent delta is relative to the
before-side. Auto-scale ladder is decimal SI for ns / count,
USER\_HZ for ticks, IEC binary for bytes.

#### Max reductions (peaks and gauges)

| Variant | Newtype | Output unit | Examples |
|---|---|---|---|
| `MaxPeak` | `PeakNs` | ns | `wait_max`, `sleep_max`, `block_max`, `exec_max`, `slice_max` |
| `MaxGaugeNs` | `GaugeNs` | ns | `fair_slice_ns` (current scheduler slice) |
| `MaxGaugeCount` | `GaugeCount` | unitless | `nr_threads` (process-wide thread count) |

`MaxPeak` rows surface the worst single window any thread in the
group has ever observed — summing per-thread maxes would
conflate "one thread with a 1s spike" with "1000 threads with
1ms spikes each". `MaxGaugeNs` / `MaxGaugeCount` apply to
instantaneous gauges (read at capture time) where summing has no
physical meaning. `nr_threads` specifically is leader-only
(populated on `tid == tgid`, zero elsewhere); `Max` reads through
the leader so a comm-bucketed group still surfaces the largest
process represented in the bucket.

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
[`metric_types`] trait — its reduction produces a structured
summary, not a homogeneous newtype.

[`metric_types`]: https://likewhatevs.github.io/ktstr/api/ktstr/metric_types/index.html
[`PeakNs`]: https://likewhatevs.github.io/ktstr/api/ktstr/metric_types/struct.PeakNs.html

### Derived metrics

Derived metrics consume one or more already-aggregated input
metrics from `CTPROF_METRICS` and produce a single scalar
with its own auto-scale ladder. They render in a separate
`## Derived metrics` table below the per-thread table on both
`compare` and `show`. Registered in `CTPROF_DERIVED_METRICS`
in `src/ctprof_compare.rs`.

The full registry as of this writing is eight entries. Every
formula is implemented as a closure over the group's metrics
map (`BTreeMap<String, Aggregated>`); a missing input or a
zero denominator yields `None`, which the renderer surfaces as
`-` so the operator can distinguish "not computable" from
"computed as zero".

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

The `is_ratio` column on the registry is load-bearing for the
renderer: ratio rows skip the `%` column entirely (the absolute
delta already carries percentage-point semantics for a `[0, 1]`
quantity), and the auto-scale ladder is `None` (bare three-
decimal scalar). Non-ratio derived metrics reuse the same
ladders as their unit family — `Ns` for nanosecond derivations,
`Bytes` for byte derivations.

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

```
CtprofSnapshot
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

Pick the [`metric_types`] newtype that matches the kernel-source
semantic of the field — the per-newtype docs name the kernel
call sites that update each category. The shape determines what
aggregation rules are legal in step 3:

| Newtype | When to use |
|---|---|
| `MonotonicCount` | Pure counter — only goes up across the thread's lifetime. Examples: `nr_wakeups`, syscall counts. |
| `DeadCounter` | Same shape as `MonotonicCount` but tagged for kernel counters with no live writer (always reads zero). Captured for parser parity but does NOT implement any reduction trait — register with `is_dead: true` and the renderer flags it `[dead]`. |
| `MonotonicNs` | Cumulative-time counter in ns. Examples: `run_time_ns`, `wait_sum`. |
| `PeakNs` | Lifetime high-water mark in ns. Kernel updates via `if (delta > stat->max) stat->max = delta`. Summing peaks is a category error. Examples: `wait_max`, `slice_max`. |
| `GaugeNs` | Instantaneous gauge sampled at capture time (ns). Cannot sum — N near-identical samples collapse to N×gauge with no meaning. Example: `fair_slice_ns`. |
| `GaugeCount` | Instantaneous unitless count that goes up AND down. Example: `nr_threads`. |
| `ClockTicks` | USER_HZ-scaled time. Examples: `utime_clock_ticks`, `stime_clock_ticks`. |
| `Bytes` | Byte counts. IEC binary auto-scale ladder. Examples: `read_bytes`, `wchar`. |
| `OrdinalI32` / `OrdinalU32` / `OrdinalU64` | Bounded scalar — range-aggregated, not summable. Examples: `nice` (i32), `rt_priority` (u32). |
| `CategoricalString` | Categorical value — mode-aggregated. Examples: `policy`. |
| `CpuSet` | CPU affinity mask — affinity-aggregated. Example: `cpu_affinity`. |

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
},
```

The `name` field is the canonical metric identifier — used by
`--sort-by`, `--columns`, and the `metric-list` output. Names
are ASCII short-form (matching the capture-side field name where
possible). `sched_class` and `config_gates` render as bracketed
suffixes in `metric-list` output (`[cfs-only]`, `[SCHEDSTATS]`)
so operators reading a row know which kernels populate the
counter.

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
the renderer suppresses the `%` column.

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
