# Periodic Capture

`Op::snapshot` is **on-demand** — the test author picks the moment of
capture. **Periodic capture** is the cadenced complement: the freeze
coordinator fires `freeze_and_capture(false)` at evenly-spaced points
across the workload window without the scenario body asking. The
result is a time-ordered series of `(report, stats, elapsed_ms)`
samples that flows naturally into the
[temporal-assertion](temporal-assertions.md) patterns.

## Enabling periodic capture

Set `num_snapshots = N` on the `#[ktstr_test]` attribute. `N` is the
number of interior boundaries to fire; `0` (the default) disables
periodic capture entirely.

```rust,ignore
use ktstr::prelude::*;

#[ktstr_test(num_snapshots = 3, duration_s = 10)]
fn paced_capture(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        CgroupDef::named("workers").workers(2).work_type(WorkType::SpinWait),
    ])
}
```

## When boundaries fire

The window is the **10 %–90 % slice** of the workload duration,
anchored at the **first `MSG_TYPE_SCENARIO_START`** the freeze
coordinator observes. A 10 % pre-buffer at the start (workload
ramp-up) and a 10 % post-buffer at the end (ramp-down) keep periodic
samples off transient state.

The remaining 80 % is divided into `N + 1` equal intervals, yielding
`N` interior boundary points:

| `num_snapshots = N` | Boundary timestamps (relative to scenario start) |
|---|---|
| `1` | `0.5·d` (midpoint) |
| `3` | `0.3·d`, `0.5·d`, `0.7·d` |
| `N ≥ 2` | `0.1·d + (i+1)·0.8·d / (N+1)` for `i ∈ 0..N` |

For a 10 s workload, `N = 3` produces captures at scenario_start +
{3 s, 5 s, 7 s}.

Anchoring at `MSG_TYPE_SCENARIO_START` means VM boot, BPF verifier
time, and any other pre-scenario work do NOT eat the budget — every
boundary lands inside the workload's actual run window.

`MSG_TYPE_SCENARIO_PAUSE` / `MSG_TYPE_SCENARIO_RESUME` from the guest
shift every un-fired boundary by the cumulative pause duration. The
boundary clock is **workload time**, not wall-clock: a guest that
pauses for `P` ns delays each remaining boundary by `P` ns.

## Tag namespace

Each periodic capture is stored on the host's `SnapshotBridge` under
`"periodic_NNN"` — zero-padded 3-digit ordinal index, e.g.
`periodic_000`, `periodic_001`, `periodic_002`. The width is fixed at
3 digits because the bridge cap (see below) maxes out at
`MAX_STORED_SNAPSHOTS` (= 64 today), so 3 digits always suffices.

Periodic tags coexist with on-demand `Op::snapshot` tags and
watchpoint-fire tags on the same bridge. Use
`SampleSeries::periodic_only`(temporal-assertions.md#sampleseries) (or
`periodic_ref()` for the borrowed equivalent) to filter to the
periodic timeline before assertions.

## Capture cost

Each periodic boundary fires the same `freeze_and_capture(false)`
path that `Op::Snapshot` dispatches:

1. Every vCPU is parked under `FREEZE_RENDEZVOUS_TIMEOUT` (30 s
   hard ceiling).
2. BPF maps are walked.
3. The dump is serialised to JSON.
4. The report is stored on the bridge.

On a healthy guest with a typical scheduler-state map size, the
freeze is tens of milliseconds (10–100 ms steady state; cold-cache
or large guest-memory walks can push higher). The host-side
watchdog deadline is **extended by the freeze duration after each
fire**, so periodic captures do not eat into the workload's
wall-clock budget.

### Minimum spacing

`KtstrTestEntry::validate` rejects entries where the per-boundary
interval is below 100 ms — boundaries scheduled closer than that
would fire back-to-back without any workload progress in between.
The exact rule: `0.8 · duration / (N + 1) >= 100 ms`. Either
reduce `num_snapshots` or extend `duration_s` if validation refuses
the configuration.

### Bridge cap

`num_snapshots` cannot exceed `MAX_STORED_SNAPSHOTS` (= 64).
Validation rejects higher values rather than silently FIFO-evicting
the earliest periodic samples. Split into multiple test entries if
a longer timeline is needed.

## Best-effort delivery

Up to `N` captures fire, but the run-loop stops servicing periodic
boundaries the moment the kill flag fires. An early VM exit, BSP
done, rendezvous timeout, or watchdog deadline can cut the periodic
sequence short. Tests should assert
`result.periodic_fired >= some_lower_bound` rather than equality:

```rust,ignore
fn check_coverage(result: &VmResult) -> Result<()> {
    anyhow::ensure!(
        result.periodic_target == 3,
        "expected num_snapshots = 3, got {}",
        result.periodic_target,
    );
    anyhow::ensure!(
        result.periodic_fired >= 2,
        "too few periodic samples ({}/{})",
        result.periodic_fired,
        result.periodic_target,
    );
    Ok(())
}
```

`result.periodic_target` mirrors the configured `num_snapshots`;
`result.periodic_fired` is the count actually serviced (including
rendezvous-timeout placeholders). The pair lets a test compute
coverage without re-reading the entry table.

The run-loop additionally **abandons the remaining sequence after 2
consecutive rendezvous timeouts** and emits a `tracing::warn` naming
the consecutive-timeout count, so a sustained host overload does
not pile up dozens of placeholder samples.

`Op::snapshot` captures composed by the test author land on the
same bridge alongside the `periodic_NNN` tags; total bridge
occupancy is `num_snapshots + user_captures` and the bridge
FIFO-evicts past `MAX_STORED_SNAPSHOTS`.

## Draining the bridge

The temporal-assertion pipeline runs on the **host**, so the drain
happens after `vm.run()` returns — typically inside a `post_vm`
callback. Use
`SnapshotBridge::drain_ordered_with_stats`(snapshots.md) to take
ownership of the captured `(tag, report, stats, elapsed_ms)` tuples
in insertion order:

```rust,ignore
use ktstr::prelude::*;

fn post_vm(result: &VmResult) -> Result<()> {
    let series = SampleSeries::from_drained(
        result.snapshot_bridge.drain_ordered_with_stats(),
    )
    .periodic_only();

    anyhow::ensure!(
        !series.is_empty(),
        "no periodic samples — coordinator never fired",
    );

    // ... walk samples or feed into temporal patterns ...
    Ok(())
}

#[ktstr_test(num_snapshots = 3, duration_s = 10, post_vm = post_vm)]
fn my_test(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        CgroupDef::named("workers").workers(2).work_type(WorkType::SpinWait),
    ])
}
```

`drain_ordered_with_stats` returns a
`Vec<(String, FailureDumpReport, Option<serde_json::Value>, Option<u64>)>`
in the order `store()` saw inserts. Periodic boundaries land
`periodic_000` first, `periodic_NNN` last. The FIFO eviction at
`MAX_STORED_SNAPSHOTS` drops the oldest tags from `order` and
`reports` together, so a hot run that overflowed the cap returns
the most recent `MAX_STORED_SNAPSHOTS` captures in insertion order.

`drain_ordered` (without `_with_stats`) drops the parallel stats /
elapsed metadata; use it only when the test does not need either.
`drain` (no ordering, no stats) returns a `HashMap` and loses the
periodic timeline ordering — avoid for periodic data.

## Sample anatomy

Each drained tuple unpacks into a `Sample<'_>` view (via
`SampleSeries::iter_samples`):

```rust,ignore
for sample in series.iter_samples() {
    let tag: &str          = sample.tag;          // e.g. "periodic_001"
    let elapsed_ms: u64    = sample.elapsed_ms;   // ms since run_start
    let snap: Snapshot<'_> = sample.snapshot;     // BPF state view
    let stats: Option<&serde_json::Value> = sample.stats; // scx_stats JSON
    // ...
}
```

`elapsed_ms` is **pause-adjusted**: the coordinator subtracts
cumulative `MSG_TYPE_SCENARIO_PAUSE`/`RESUME` time (and any in-flight
pause window) before stamping the value. The timestamp is captured
AFTER the scx_stats request returns (or fails) and BEFORE entering
the freeze rendezvous, so `elapsed_ms` reflects when the running
scheduler's stats were observed; BPF state is observed up to
`FREEZE_RENDEZVOUS_TIMEOUT` later than that anchor.

`stats` is `None` when the stats client was not wired
(`scheduler_binary` is absent), or the per-sample stats request
failed (relay rejected, non-zero envelope errno, scheduler not yet
listening). A `None` slot surfaces through
`SampleSeries::stats`(temporal-assertions.md#projecting-from-scx_stats-json) as a
`SnapshotError::MissingStats { tag }` per-sample error — distinct
from in-JSON path misses so the assertion site can branch on the
cause.

A sample whose underlying `FailureDumpReport` is a placeholder
(rendezvous timeout fallback) surfaces through
`SampleSeries::bpf`(temporal-assertions.md#projecting-from-bpf-state) as a
`SnapshotError::PlaceholderSample { tag, reason }` per-sample error
rather than passing a hollow `Snapshot` to the projection closure.

## What to assert

The standard shape is two-stage:

1. **Compose the series** — drain, filter to periodic.
2. **Project + assert** — pick a column, choose a temporal pattern.

For monotonic counters (BPF `.bss` advancement, scx_stats counter
fields), `nondecreasing`(temporal-assertions.md#nondecreasing--strictly_increasing)
is the canonical choice. For utilisation-style metrics that should
hold steady once warmup ends,
`steady_within`(temporal-assertions.md#steady_withinwarmup_ms-tolerance-f64-only)
captures the invariant. For "system stabilizes near `target`",
`converges_to`(temporal-assertions.md#converges_totarget-tolerance-deadline_ms-f64-only)
witnesses the convergence.

For the full pattern surface, projection helpers, and failure
rendering, see [Temporal Assertions](temporal-assertions.md).
