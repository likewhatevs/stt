# Diagnose a Slow Scheduler with ctprof

When a scheduler change makes the workload slower but the test
suite still passes, the regression is usually buried in
per-thread off-CPU time. `ktstr ctprof capture` snapshots every
live thread's scheduling, memory, I/O, and taskstats delay
counters; `ktstr ctprof compare` diffs two snapshots and surfaces
the buckets where time went. This recipe walks through a typical
A/B comparison.

See the [ctprof reference](../reference/ctprof.md) for the full
metric registry, aggregation rules, derived-metric formulas, and
taskstats kconfig gating.

## Capture before and after

```sh
# Baseline: scheduler A loaded, workload running.
ktstr ctprof capture --output baseline.ctprof.zst

# Switch schedulers, restart workload, wait for steady state.
# ...

# Candidate: scheduler B, same workload.
ktstr ctprof capture --output candidate.ctprof.zst
```

`capture` walks `/proc` once and writes the snapshot. It is
read-only — no kprobes, no tracing — so the act of capturing
does not perturb the measurement. The default capture covers
every live tgid; on a busy host this is hundreds of threads.
The snapshot is zstd-compressed JSON, typically a few MB.

## Compare with the taskstats lens

The `taskstats-delay` section bundles the eight kernel
delay-accounting buckets (CPU, blkio, swapin, freepages,
thrashing, compact, wpcopy, irq) plus their nine derived
metrics (`avg_*_delay_ns` per bucket, `total_offcpu_delay_ns`
rollup). Running with `--sections taskstats-delay` filters the
output down to just the off-CPU view:

```sh
ktstr ctprof compare baseline.ctprof.zst candidate.ctprof.zst \
    --sections taskstats-delay \
    --sort-by total_offcpu_delay_ns:desc
```

The `--sort-by total_offcpu_delay_ns:desc` puts the processes
with the largest absolute off-CPU growth at the top. Each row
gives `baseline | candidate | delta | %`; large positive deltas
on a process that should not have moved are the suspects.

The `total_offcpu_delay_ns` derivation is:

```text
cpu + blkio + freepages + compact + wpcopy + irq + max(swapin, thrashing)
```

`max(swapin, thrashing)` rather than `swapin + thrashing`
because every thrashing event is also a swapin event from the
syscall perspective; summing both would double-count.

## Drill into the per-bucket averages

If `total_offcpu_delay_ns` jumped on a process, the per-bucket
`avg_*_delay_ns` derivations identify *which* off-CPU phase
grew. In the same compare output (the `--sections taskstats-delay`
filter keeps both the raw counters AND the 9 derivations
together), look at the suspect process's row in:

| Bucket | Average derivation | Meaning |
|---|---|---|
| CPU runqueue wait | `avg_cpu_delay_ns` | Time waiting for the scheduler to pick the task. RACY (count + total update lockless). |
| Block I/O wait | `avg_blkio_delay_ns` | Synchronous block-device wait. Distinct from schedstat `iowait_sum`; the canonical delay-accounting reading. |
| Swap-in / Thrashing | `avg_swapin_delay_ns` / `avg_thrashing_delay_ns` | Memory pressure. The two overlap — a thrashing event is also a swapin event. |
| Direct memory reclaim | `avg_freepages_delay_ns` | Allocator hit `__alloc_pages` slowpath. |
| Memory compaction | `avg_compact_delay_ns` | Allocator demanded a high-order page; compaction stalled. |
| CoW page-fault | `avg_wpcopy_delay_ns` | Write-protect-copy fault, e.g. fork-then-write. |
| IRQ handling | `avg_irq_delay_ns` | Time charged to the task by the IRQ accounting subsystem. |

A growing `avg_cpu_delay_ns` with flat blkio/swap/freepages
suggests the new scheduler is making poor placement choices —
the task is queueing more often or for longer, but no other
subsystem is to blame. A growing `avg_blkio_delay_ns` with flat
`avg_cpu_delay_ns` points away from the scheduler entirely
(disk, network filesystem, or a userspace lock pattern).

## Cross-reference the primary table

Once a bucket is identified, look at the underlying counters
without the section filter:

```sh
ktstr ctprof compare baseline.ctprof.zst candidate.ctprof.zst \
    --metrics nr_wakeups,nr_migrations,wait_sum,wait_count,run_time_ns,timeslices
```

`--metrics` restricts the rendered rows to the named primary
metrics. Useful pairings when the suspect bucket is CPU runqueue
wait:

- `wait_sum / wait_count` — schedstat's average wait per
  scheduling event (the `avg_wait_ns` derivation, exposed
  outside the `taskstats-delay` section). If this confirms
  `avg_cpu_delay_ns`, both delay-accounting paths agree.
- `nr_migrations` — the new scheduler may be moving the task
  more aggressively. Cross-CPU migrations cost wall-clock time
  even when `run_time_ns` is identical.
- `nr_wakeups_affine / nr_wakeups_affine_attempts` — the
  `affine_success_ratio` derivation; CFS-only signal that
  reflects how often `wake_affine()` succeeded. A large drop
  with growing `avg_cpu_delay_ns` is a strong signal for cache-
  unfriendly placement.

## Confirm taskstats data is actually populated

If every taskstats column reads zero, the snapshot likely hit a
gating problem rather than a real "no delay" reading. Inspect
[`CtprofSnapshot::taskstats_summary`] (the structured
per-snapshot tally written into the snapshot itself):

- `eperm_count > 0` — the capturing process lacked
  `CAP_NET_ADMIN`. Re-run as root, or grant
  `cap_net_admin+eip` via `setcap`.
- `esrch_count` near `tids_walked` — every tid raced exit
  before the per-tid query landed. Lengthen the workload's
  steady-state window and re-capture.
- `ok_count == 0` AND `eperm_count == 0` — the netlink open
  failed, almost always meaning the kernel was built without
  `CONFIG_TASKSTATS`. Rebuild with the kconfig.
- `ok_count > 0` but every delay column reads zero — kernel
  built with `CONFIG_TASKSTATS` and `CONFIG_TASK_DELAY_ACCT`
  but launched without the runtime `delayacct=on` toggle. Add
  `delayacct` to the kernel cmdline, or set
  `sysctl kernel.task_delayacct=1` and re-capture.

The structured fields above let an operator distinguish each
case without scraping the capture-pipeline tracing log.

## Related

- [ctprof reference](../reference/ctprof.md) — the full metric
  registry and gating documentation.
- [Capture and Compare Host State](host-state.md) — the
  `cargo ktstr show-host` recipe for *host-context* diffs
  (kernel, sched_\* tunables, NUMA layout); use that when the
  hypothesis is "the host config moved" rather than "a
  workload's per-thread behaviour moved."
- [A/B Compare Branches](ab-compare.md) — recipe for diffing
  scheduler-source-tree changes via ktstr's gauntlet runs;
  ctprof complements that by surfacing per-thread-level effects
  the scenario assertions miss.
