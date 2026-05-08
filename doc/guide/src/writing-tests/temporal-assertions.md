# Temporal Assertions

Periodic snapshots produce a series of samples over time. **Temporal
assertions** answer questions about the *trajectory* — does a counter
only ever advance? Does a utilization metric stay near its mean once
warmup ends? Does a load average converge before a deadline?

The shape is two-stage:

1. Build a `SampleSeries`(#sampleseries) from the bridge's drained
   periodic captures.
2. **Project** a `SeriesField<T>`(#seriesfield) — one column of
   `T`-typed values across every sample — and feed it through a
   temporal pattern (`nondecreasing`, `rate_within`, `steady_within`,
   `converges_to`, `always_true`, `ratio_within`).

Each pattern records `DetailKind::Temporal`(#failure-rendering)
details on the `Verdict` when a sample violates the invariant, and
records `Note`s when projection errors leave a coverage gap.

For how to enable periodic capture and drain the bridge, see
[Periodic Capture](periodic-capture.md). This page covers the
projection + assertion surface only.

## SampleSeries

`SampleSeries` is the ordered sequence of `(tag, report, stats,
elapsed_ms)` tuples drained from the bridge after the VM exits. Build
it from
`SnapshotBridge::drain_ordered_with_stats`(snapshots.md#wiring-the-bridge):

```rust,ignore
use ktstr::prelude::*;

let drained = vm_result.snapshot_bridge.drain_ordered_with_stats();
let series = SampleSeries::from_drained(drained).periodic_only();
```

`periodic_only()` filters to entries whose tag begins with
`"periodic_"` — it strips on-demand `Op::snapshot` captures and
watchpoint fires that share the bridge's tag namespace. Use
`periodic_ref()` for the borrowed-iterator equivalent when one test
needs both views from the same series.

`SampleSeries` exposes:

- `len()`, `is_empty()` — sample count.
- `iter_samples()` — borrowed `Sample<'_>` views (each carrying
  `tag`, `elapsed_ms`, `Snapshot<'_>`, `Option<&Value>` stats).
- `bpf(label, |snap| …)` / `stats(label, |sv| …)` — manual closure
  projection along the BPF or stats axis.
- `bpf_map(map_name)` / `stats_path(path)` — typed auto-projection
  helpers (see [Auto-projection](#auto-projection)).

## SeriesField

A `SeriesField<T>` is one per-sample column extracted from a
`SampleSeries`. Each slot is a `SnapshotResult<T>` so a missing
field, type mismatch, or placeholder report on any individual sample
does NOT abort the whole projection — it surfaces at the temporal-
assertion site as a per-sample error the pattern decides how to
handle.

The field carries the per-sample tags and elapsed-ms timestamps
alongside the values, so failure messages name the offending sample
without the caller re-threading the source series.

### Projecting from BPF state

The `SampleSeries::bpf` closure receives each sample's
`Snapshot<'_>`:

```rust,ignore
let nr_dispatched: SeriesField<u64> = series.bpf(
    "nr_dispatched",
    |snap| snap.var("nr_dispatched").as_u64(),
);
```

The closure body is a normal [Snapshot accessor expression](snapshots.md);
its `SnapshotResult<T>` return value lands directly in the field.

### Projecting from scx_stats JSON

The `SampleSeries::stats` closure receives each sample's
`StatsValue<'_>` — a thin wrapper around the per-sample stats JSON
exposing `path("…").as_u64()` / `as_f64()` etc.:

```rust,ignore
let busy: SeriesField<f64> = series.stats(
    "busy",
    |sv| sv.path("busy").as_f64(),
);
```

A sample whose stats slot is `None` (the stats request failed, the
relay rejected, or the scheduler binary isn't wired) yields a
`SnapshotError::MissingStats { tag }` slot — distinct from an
in-JSON path miss (`FieldNotFound` / `TypeMismatch`) so the
assertion site can tell coverage gaps from data errors apart.

### Auto-projection

The typed auto-projectors discover available field names and emit
ready-to-feed `SeriesField`s without an explicit closure:

```rust,ignore
// Top-level scalar member of a BPF map's first entry.
let dispatched = series
    .bpf_map("scx_obj.bss")
    .at(0)
    .field_u64("nr_dispatched");

// Stats path drilling into nested layer/cgroup keys.
let layer_util = series
    .stats_path("layers")
    .key("batch")
    .field_f64("util");
```

Bulk discovery is also available — `member_names()` /
`u64_fields()` / `f64_fields()` on the BPF projector,
`key_names()` / `u64_fields()` / `f64_fields()` on the stats
projector. The `*_fields()` helpers project every member that
yields at least one `Ok` value across the series, dropping
non-numeric / type-mismatched fields silently. Useful for blanket
"every counter must be nondecreasing" sweeps.

**Top-level scalar fields only** for the typed `field_*` helpers.
Nested struct members (e.g. `"ctx.weight"`) and per-CPU maps need
the manual closure path through `SampleSeries::bpf`.

## The six temporal patterns

Every pattern takes `&mut Verdict` and returns the same `&mut
Verdict` so chains of assertions stack onto one accumulator. Each
pattern is a method on `SeriesField`:

### `nondecreasing` / `strictly_increasing`

Pass when every consecutive pair satisfies `values[i] <=
values[i+1]` (or `<`, for the strict variant). The common shape for
kernel counters whose only legal direction is up.

```rust,ignore
let mut v = Verdict::new();
nr_dispatched.nondecreasing(&mut v);
nr_dispatched.strictly_increasing(&mut v); // require advance every period
```

Per-sample projection errors are SKIPPED — the affected pair is
dropped, the skip count is logged as a verdict `Note`, and the
verdict is NOT flipped on missing-data conditions. Adjacent samples
on either side of a gap are still checked. A series with fewer than
2 samples records a `Note` ("vacuously holds") and passes.

### `rate_within(lo, hi)` (f64 only)

Pass when every consecutive `(delta_value / delta_ms)` lies in
`[lo, hi]`. Rate is computed from per-sample elapsed-ms timestamps,
so a counter that should advance at ~1 unit/ms reads as
`rate_within(0.5, 2.0)`.

```rust,ignore
let ticks: SeriesField<f64> = series.bpf("ticks",
    |snap| snap.var("ticks").as_f64());
ticks.rate_within(&mut v, 0.5, 2.0);
```

Failure modes:
- A zero-time delta between adjacent samples records a structured
  detail naming the offending pair.
- A non-finite rate (NaN / Inf endpoints, or a finite difference
  that overflows f64) records a `non-finite rate` detail rather
  than silently slipping past the band check.
- Caller error (`lo > hi`) lands as a single detail.

Per-sample projection errors are GAPS — no rate is computed across
the gap, the skip count is logged as a `Note` with the underlying
error variant.

### `steady_within(warmup_ms, tolerance)` (f64 only)

Pass when every post-warmup sample (`elapsed_ms >= warmup_ms`)
lies inside `[mean·(1-tolerance), mean·(1+tolerance)]`. The mean is
computed over the post-warmup samples only — the warmup region is
excluded so ramp-up does not bias the steady-state baseline.
`tolerance` is a fraction (`0.10` = ±10%).

```rust,ignore
let util: SeriesField<f64> = series.stats("busy",
    |sv| sv.path("busy").as_f64());
util.steady_within(&mut v, /*warmup_ms=*/ 1000, /*tolerance=*/ 0.10);
```

Per-sample projection errors are SKIPPED with a `Note`. When the
warmup window absorbs every sample, the pattern emits a "no
samples beyond warmup" `Note` and passes vacuously.

### `converges_to(target, tolerance, deadline_ms)` (f64 only)

Pass when **three consecutive samples** land inside `[target -
tolerance, target + tolerance]` AT OR BEFORE `deadline_ms`. The
intent is "the system stabilizes near `target` by the deadline" —
three consecutive in-band samples are the convergence-witness shape.

```rust,ignore
load.converges_to(&mut v, /*target=*/ 1.0, /*tol=*/ 0.5, /*deadline_ms=*/ 5_000);
```

Distinct outcomes:
- **Witness found** — pass.
- **No witness before deadline** — `DetailKind::Temporal` failure
  naming the sample count evaluated. If errored samples interrupted
  in-progress runs, the failure message lists them.
- **Insufficient samples** — fewer than 3 successfully-projected
  samples in the deadline window. Records a `Note` (NOT a verdict
  failure); absence of data is a coverage gap, not a negative
  finding. The note distinguishes "did not collect enough samples"
  from "collected enough samples but never converged".

### `always_true` (bool only)

Pass when every sample's value is `true`. Per-sample projection
errors FAIL the assertion (this is a strict pattern — a missing
boolean is a coverage gap that must surface).

```rust,ignore
let alive: SeriesField<bool> = series.bpf("scheduler_alive",
    |snap| snap.var("scheduler_alive").as_bool());
alive.always_true(&mut v);
```

### `ratio_within(other, lo, hi)` (f64 only)

Pass when every per-index `(self_value / other_value)` lies in
`[lo, hi]` — the two series are walked in lock-step at indices
`0..N`, comparing `self[i] / other[i]`. Cross-field correlation
across two same-length series.

```rust,ignore
util.ratio_within(&mut v, &runtime, 0.4, 0.6);
```

A length mismatch fires a single caller-error detail and aborts
the comparison. A sample where `rhs == 0` records a "cannot
compute ratio" detail naming the sample; out-of-band ratios
record a structured detail with the lhs/rhs values. Per-sample
projection errors on either side are SKIPPED with a `Note`
listing each gap and which side errored.

## Per-sample scalar checks: `each`

The temporal patterns are aggregate. For per-sample scalar bounds
(`>=`, `<=`, `lo..=hi`) bypass the patterns via `SeriesField::each`:

```rust,ignore
nr_dispatched.each(&mut v).at_least(1u64);
util.each(&mut v).between(0.0_f64, 100.0_f64);
ticks.each(&mut v).at_most(10_000.0_f64);
```

`each` runs the comparator on every successfully-projected sample
independently. The first failure records a detail; subsequent
failures pile on so the timeline shows every offending sample, not
just the first.

Per-sample projection errors record a detail and flip the verdict
— `each` is strict (matches `always_true`'s policy). NaN samples
report an `incomparable` failure naming the sample distinctly:
without this branch, IEEE-754 `<` against NaN is always false, so
a NaN sample would silently pass `value < floor` / `value > ceiling`
checks.

## Failure rendering

Every temporal failure carries the field's `label`, the pattern
name, and the offending sample's `tag` + `elapsed_ms`. A
nondecreasing regression at sample `periodic_004` (+850 ms) reads:

```text
nr_dispatched (nondecreasing): regression at sample periodic_004 (+850ms): \
    value 100 after prior value 200 at sample periodic_003 (+700ms)
```

Coverage `Note`s render WITH the per-sample error variant so the
operator can tell `PlaceholderSample` (rendezvous timeout),
`MissingStats` (stats request failed), `FieldNotFound` (typo /
wrong map), and `TypeMismatch` apart without re-running under a
debugger:

```text
nr_dispatched (nondecreasing): skipped 1 sample(s) with projection errors: \
    periodic_002(+500ms): snapshot has no global variable 'nrdispatch' \
    in any *.bss/*.data/*.rodata map (available globals: ["nr_dispatched", \
    "stall"])
```

## Worked example

The temporal-assertion pipeline draining the bridge runs on the
**host**, not inside the guest. `#[ktstr_test(post_vm = …)]` registers
a host-side callback that receives the `VmResult` after `vm.run()`
returns; the callback drains the bridge and walks the resulting
series:

```rust,ignore
use ktstr::prelude::*;

fn assert_temporal_patterns(result: &VmResult) -> Result<()> {
    let series = SampleSeries::from_drained(
        result.snapshot_bridge.drain_ordered_with_stats(),
    )
    .periodic_only();

    let mut v = Verdict::new();

    // BPF axis: counter must advance every periodic boundary.
    let nr_dispatched: SeriesField<u64> = series.bpf(
        "nr_dispatched",
        |snap| snap.var("nr_dispatched").as_u64(),
    );
    nr_dispatched.nondecreasing(&mut v);

    // Stats axis: stay under a generous ceiling.
    let stats_dispatched: SeriesField<u64> = series.stats(
        "nr_dispatched",
        |sv| sv.path("nr_dispatched").as_u64(),
    );
    stats_dispatched.each(&mut v).at_most(1_000_000_000u64);

    let r = v.into_result();
    anyhow::ensure!(r.passed, "temporal assertions failed: {:?}", r.details);
    Ok(())
}

#[ktstr_test(num_snapshots = 3, duration_s = 10, post_vm = assert_temporal_patterns)]
fn dispatch_counter_advances(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(ctx, vec![
        CgroupDef::named("workers").workers(2).work_type(WorkType::SpinWait),
    ])
}
```

For the periodic-capture wiring, `num_snapshots` semantics, and the
bridge-drain contract, see [Periodic Capture](periodic-capture.md).
For the underlying `Snapshot` / `SnapshotMap` / `SnapshotEntry`
accessors the projection closures call into, see
[Snapshots](snapshots.md).
