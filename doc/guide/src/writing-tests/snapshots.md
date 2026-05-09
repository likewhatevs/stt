# Snapshots

A **snapshot** is a frozen record of guest BPF map state and scheduler
globals captured at a specific point in a scenario. The freeze
coordinator pauses every vCPU long enough to walk the kernel's BPF
maps, BTF-render every captured value, and bundle the result into a
`FailureDumpReport` keyed by a name you choose. Test code then reads
it back via the [`Snapshot`](#reading-the-captured-report) accessor for typed traversal.

`Op::snapshot("name")` is the **on-demand** capture trigger. Use it to
ask "what does the scheduler look like *right now*?" at a precise
point in the scenario. For automatic capture on a kernel write to a
specific symbol, see [Watch Snapshots](watch-snapshots.md). For
**cadenced** capture across the workload window without invoking
`Op::snapshot` from the scenario body, see
[Periodic Capture](periodic-capture.md) — it produces a time-ordered
[`SampleSeries`](temporal-assertions.md#sampleseries) that flows into
the [temporal-assertion](temporal-assertions.md) patterns
(`nondecreasing`, `rate_within`, `steady_within`, `converges_to`,
`always_true`, `ratio_within`).

## Issuing a snapshot

`Op::snapshot(name)` is a single op in a [`Step`](../concepts/ops.md)'s op list. The
executor invokes the active [`SnapshotBridge`](#wiring-the-bridge)'s capture callback,
which performs the freeze rendezvous and returns the report; the
bridge stores the report under `name`.

```rust,ignore
use ktstr::prelude::*;

let steps = vec![Step {
    setup: vec![CgroupDef::named("workers").workers(2)].into(),
    ops: vec![
        Op::snapshot("after_spawn"),
        // ... other ops ...
        Op::snapshot("after_workload"),
    ],
    hold: HoldSpec::FULL,
}];
execute_steps(ctx, steps)?;
```

A scenario may issue any number of `Op::snapshot` ops with distinct
names. Reusing a name overwrites the prior capture (and emits a
`tracing::warn!`).

## Wiring the bridge

The bridge is what turns an `Op::snapshot` into stored data. The host
typically wires it before `execute_steps` runs, but a scenario can
install one inline:

```rust,ignore
use ktstr::prelude::*;

let cb: CaptureCallback = std::sync::Arc::new(|_name: &str| {
    // Production: freeze the VM and build a real FailureDumpReport.
    // Tests: return a hand-crafted report so the executor + bridge
    // pipeline runs without booting a guest.
    Some(FailureDumpReport::default())
});
let bridge = SnapshotBridge::new(cb);
let bridge_handle = bridge.clone();
let _guard = bridge.set_thread_local();

execute_steps(ctx, steps)?;

let captured = bridge_handle.drain();
let report = captured.get("after_spawn").expect("snapshot recorded");
```

`set_thread_local` returns a [`BridgeGuard`](#wiring-the-bridge) that restores the prior
bridge on drop, so a nested scenario inside an outer one cannot leak
its bridge into the outer scope. Bind the guard to an
underscore-prefixed identifier such as `_guard` so the binding lives
for the scope of the scenario — a bare `let _ = bridge.set_thread_local()`
drops the guard immediately and clears the bridge before any op runs.
`must_use` will warn if the return value is discarded entirely.

If no bridge is installed, `Op::snapshot` is a no-op with a
`tracing::warn!` and the scenario continues. If the capture callback
returns `None` (capture pipeline unavailable), the bridge stays empty
and the scenario continues. Existing scenarios that never declare
snapshot ops keep working unchanged.

## Reading the captured report

[`Snapshot::new(report)`](#reading-the-captured-report) builds a borrowed view over a
`FailureDumpReport`. The view does not copy the report; accessor
methods walk the report in place and return further borrowed views.

### Map-name lookup

```rust,ignore
let snap = Snapshot::new(report);
let map = snap.map("scx_per_task")?;        // SnapshotMap
```

`Snapshot::map(name)` returns `Result<SnapshotMap, SnapshotError>`. A
miss yields `SnapshotError::MapNotFound { requested, available }` —
the `available` list enumerates every captured map name so a typo
surfaces in test output.

### Top-level globals (.bss / .data / .rodata)

```rust,ignore
let nr_cpus = snap.var("nr_cpus_onln").as_u64()?;
```

`Snapshot::var(name)` walks every `*.bss`, `*.data`, and `*.rodata`
global-section map for a top-level member named `name` and returns
the unique match as a [`SnapshotField`](#terminal-accessors).
Multiple matches yield
`SnapshotError::AmbiguousVar { requested, found_in }` —
disambiguate via `Snapshot::map(name)`. A miss yields
`SnapshotError::VarNotFound { requested, available }` with the
union of every section's top-level member names.

### Entries inside a map

```rust,ignore
let map = snap.map("scx_per_task")?;
let first = map.at(0);                          // by ordinal index
let busy = map.find(|e| e.get("tid").as_i64().unwrap_or(-1) == 1234);
let busiest = map.max_by(|e| e.get("runtime_ns").as_u64().unwrap_or(0));
let all_active = map.filter(|e| e.get("runtime_ns").as_u64().unwrap_or(0) > 0);
```

`SnapshotMap` exposes:

- `at(n)` — entry at ordinal index `n`. Out of range returns
  `SnapshotEntry::Missing(SnapshotError::IndexOutOfRange)`.
- `find(predicate)` — first matching entry. No match returns
  `SnapshotEntry::Missing(SnapshotError::NoMatch { op: "find", ... })`.
- `filter(predicate)` — every matching entry collected into a `Vec`.
- `max_by(key_fn)` — entry whose `key_fn` produces the maximum `u64`.
  Empty map returns `Missing` with `op: "max_by"`.

### Per-CPU maps

`BPF_MAP_TYPE_PERCPU_ARRAY` / `_PERCPU_HASH` / `_LRU_PERCPU_HASH` maps
require narrowing to a CPU before reading individual values:

```rust,ignore
let map = snap.map("scx_pcpu")?;
let entry = map.cpu(1).at(0);                    // CPU 1's slot
let value = entry.get("").as_u64()?;             // empty path = root
```

`SnapshotMap::cpu(n)` narrows subsequent `at` / `find` calls to a
specific CPU's slot. An out-of-range CPU returns `Missing` with
`SnapshotError::PerCpuSlot { unmapped: false, len, ... }`; an
unmapped slot (`None` in the per-CPU vec) returns the same error
variant with `unmapped: true`.

Calling `entry.get(path)` on a per-CPU entry **without** narrowing
first surfaces `SnapshotError::PerCpuNotNarrowed { map }` — call
`.cpu(N)` first.

## Field accessors and dotted paths

`SnapshotEntry::get(path)` and `SnapshotField::get(path)` walk the
entry's value side along a dotted path. Each component matches a
struct member; pointer dereferences are followed transparently.

```rust,ignore
let weight = entry.get("ctx.weight").as_u64()?;
let policy = entry.get("ctx.policy").as_str()?;     // enum variant name
let pid    = entry.get("leader.pid").as_i64()?;     // pointer chase
```

The dotted-path walker:

1. **Pointer chase.** When a path step lands on
   `RenderedValue::Ptr { deref: Some(...) }`, the walker
   transparently follows the dereference (up to 16 hops) before
   matching the next component. The test author writes the path the
   BTF would suggest; pointer indirection is invisible.

2. **Empty path.** `get("")` returns the current value as a
   `SnapshotField::Value` — useful for terminal accessors on per-CPU
   slots that hold a scalar directly.

3. **Composability.** Two-segment paths are equivalent to chained
   `get` calls: `entry.get("ctx.weight")` ≡
   `entry.get("ctx").get("weight")`.

   Note that [`Snapshot::var`](#top-level-globals-bss--data--rodata) does **not** split — it treats the full
   string as one global name. To walk into a struct, use
   `snap.var("ctx").get("weight")`.

### Terminal accessors

`SnapshotField` exposes typed terminal reads, all returning
`Result<T, SnapshotError>`:

| Method | Returns | Accepts |
|---|---|---|
| `as_u64()` | `u64` | `Uint`, non-negative `Int`/`Enum`, `Bool` (0/1), `Char` (raw byte), `Ptr` (pointer value, including cast-recovered pointers — see [Cast-recovered pointers](#cast-recovered-pointers)), per-CPU array key |
| `as_i64()` | `i64` | `Int`, `Uint` ≤ i64::MAX, `Bool`, `Char`, `Enum`, per-CPU array key |
| `as_bool()` | `bool` | `Bool` direct; `Int`/`Uint`/`Char`/`Enum`/`Ptr` non-zero is true; per-CPU array key |
| `as_f64()` | `f64` | `Float`, `Int`, `Uint`, `Enum`, per-CPU array key |
| `as_str()` | `&str` | `Enum` with a resolved variant name |
| `rendered()` | `Option<&RenderedValue>` | the underlying value when present |

Type mismatches surface as `SnapshotError::TypeMismatch { expected,
actual, requested }` — for example, `as_str()` on a `Uint` reports
`expected: "Enum"`, `actual: "Uint"`.

### Cast-recovered pointers

Schedulers stash kernel pointers (`task_struct *`, `cgroup *`, …)
and arena pointers in BPF map fields whose BTF declares them as
`u64` because BTF cannot express a pointer to a per-allocation
type. The host-side
[cast analyzer](../architecture/monitor.md#cast-analysis) walks the
scheduler's `.bpf.o` instruction stream during load, recovers the
target struct for each provable `(source_struct, field_offset) →
target_struct` mapping, and feeds the result into the renderer.

When the renderer encounters a `u64` slot the analyzer flagged, it
emits a [`RenderedValue::Ptr`](#field-accessors-and-dotted-paths)
with `cast_annotation` set and chases the dereference through the
address-space-appropriate reader. The full set of
`cast_annotation` values:

| Annotation | Meaning |
|---|---|
| `"cast→arena"` | Cast analyzer flagged a `u64` field; chase resolved to an arena allocation via the BTF-typed pointee. |
| `"cast→kernel"` | Cast analyzer flagged a `u64` field; chase resolved to a kernel slab / vmalloc / per-cpu allocation. |
| `"sdt_alloc"` | BTF-typed `Type::Ptr` whose pointee was a `BTF_KIND_FWD`; the renderer recovered the real payload struct id via the `sdt_alloc` bridge. No cast-analyzer hit was involved. |
| `"cast→arena (sdt_alloc)"` | Cast analyzer flagged a `u64` field AND the chase target peeled to a Fwd; the bridge recovered the real arena payload struct id. |
| `"cast→kernel (sdt_alloc)"` | Cast analyzer flagged a `u64` field AND the chase target peeled to a Fwd; the bridge recovered the real kernel-side struct id. |

A parallel cross-BTF Fwd resolution path is consulted whenever a
chase target survives the local same-BTF Fwd resolve as a
`BTF_KIND_FWD`: when the body lives in a sibling embedded BPF
object's BTF (the multi-`.bpf.objs` shape), the renderer switches
the recursion to that sibling BTF and renders the full body.
Cross-BTF resolution does NOT add a new annotation — the body is
recovered transparently and the rendered subtree carries whichever
annotation (`"cast→arena"`, `"cast→kernel"`, or `None` for a
BTF-typed `Type::Ptr`) it would have had if the same struct lived
in the entry BTF.

From the test author's perspective:

- `as_u64()` returns the raw pointer value (matching pre-analysis
  behavior, so existing tests do not need updating).
- `entry.get("ctx.task")` and similar dotted-path walks transparently
  follow the cast-recovered chase; nested struct fields appear under
  the same path the BTF would suggest for a natively-typed pointer.
- The `cast_annotation` is visible in failure-dump rendering and
  diagnostic output so an operator can distinguish cast-recovered
  pointers from BTF-typed ones; the test API does not require any
  extra calls to consume them.

## Error handling

[`SnapshotError`](#error-handling) is the unified error type for every fallible
accessor. Each variant carries the path or available alternatives
needed to fix the call site without re-running the test:

- `MapNotFound { requested, available }` — `Snapshot::map(name)` miss.
- `VarNotFound { requested, available }` — `Snapshot::var(name)` miss.
- `AmbiguousVar { requested, found_in }` — more than one
  `*.bss`/`*.data`/`*.rodata` map exposes a top-level member with the
  requested name. `found_in` lists every map (in capture order)
  where the name was seen; disambiguate via `Snapshot::map(name)` +
  `.at(0).get(...)` against a specific map.
- `FieldNotFound { requested, walked, component, available }` — a
  path component did not match any struct member at that depth.
  `walked` is the prefix that resolved successfully; `component` is
  the failing segment; `requested` is the original user-supplied
  path.
- `NotAStruct { requested, walked, component, kind }` — a path
  component reached a non-struct value where a struct was expected
  (e.g. descending into a `Uint` leaf). `kind` names the actual
  variant.
- `TypeMismatch { expected, actual, requested }` — terminal
  accessor called on a rendered shape it cannot decode. `expected`
  names the scalar type the accessor requires; `actual` names the
  rendered variant; `requested` is the user-supplied lookup string
  (empty when the accessor was invoked on a leaf without a path
  walk).
- `IndexOutOfRange { map, index, len }` — `SnapshotMap::at(n)` past
  the entry list end.
- `PerCpuSlot { map, cpu, len, unmapped }` — out-of-range or unmapped
  per-CPU slot; `unmapped: true` distinguishes a `None` slot from an
  out-of-range CPU.
- `NoMatch { map, op }` — predicate-based lookup (`find`, `max_by`)
  found no match. `op` names the operation.
- `EmptyPathComponent { requested }` — a path string contained an
  empty component (e.g. `"a..b"`).
- `PerCpuNotNarrowed { map }` — `entry.get` called on a per-CPU entry
  without `cpu(N)` first.
- `NoRendered { map, side }` — entry has no rendered key/value side
  (BTF type id missing at capture time, leaving hex bytes only).
- `PlaceholderSample { tag, reason }` — a periodic-capture sample's
  underlying `FailureDumpReport` is a placeholder produced by the
  freeze-rendezvous timeout fallback. Surfaces when projecting via
  [`SampleSeries::bpf`](temporal-assertions.md#projecting-from-bpf-state);
  temporal patterns route the variant through their skip path so a
  placeholder never falsely registers as zero progress against a
  monotonicity / rate / steady / ratio band. `reason` carries the
  rendezvous-timeout cause text.
- `MissingStats { tag }` — a [`SampleSeries::stats`](temporal-assertions.md#projecting-from-scx_stats-json)
  projection ran on a sample whose `stats` slot is `None` (stats
  client not wired or per-sample stats request failed). Distinct
  from in-JSON path misses (`FieldNotFound` / `TypeMismatch`) so the
  assertion site can branch on the cause without re-walking the
  source.

`SnapshotError` implements `std::error::Error` and `Display`, so it
composes with `?` and `anyhow`. The `Display` impl includes the path
and any available alternatives so a failure message points the test
author at the fix.

## Worked example

Capture a snapshot, look up a map, walk into its first entry, and
read a nested field:

```rust,ignore
use ktstr::prelude::*;

fn snapshot_then_inspect(ctx: &Ctx) -> Result<AssertResult> {
    // Wire a bridge for the duration of the scenario.
    let cb: CaptureCallback = std::sync::Arc::new(|_name| {
        // Production: freeze + build a real FailureDumpReport. The
        // host installs this callback in real runs.
        Some(FailureDumpReport::default())
    });
    let bridge = SnapshotBridge::new(cb);
    let handle = bridge.clone();
    let _guard = bridge.set_thread_local();

    // Run the scenario, capturing once after spawn.
    let steps = vec![Step {
        setup: vec![CgroupDef::named("workers").workers(2)].into(),
        ops: vec![Op::snapshot("after_spawn")],
        hold: HoldSpec::FULL,
    }];
    let mut result = execute_steps(ctx, steps)?;

    // Drain the bridge and inspect the captured report.
    let captured = handle.drain();
    let report = captured
        .get("after_spawn")
        .ok_or_else(|| anyhow::anyhow!("snapshot 'after_spawn' missing"))?;
    let snap = Snapshot::new(report);

    // Top-level scalar.
    if let Ok(nr_cpus) = snap.var("nr_cpus_onln").as_u64() {
        result.details.push(AssertDetail::new(
            DetailKind::Other,
            format!("captured nr_cpus_onln = {nr_cpus}"),
        ));
    }

    Ok(result)
}
```

For the executor + bridge wiring outside a VM, see the host-side
smoke tests in `tests/snapshot_e2e.rs` — they exercise the same
pipeline against a hand-crafted `FailureDumpReport` so the assertion
shape is covered without booting a guest.

## Composing reads with writes

Snapshots are the **read** half of the host↔guest interaction. The
**write** half — pre-seeding a BPF map value before the scenario
starts — is the `#[ktstr_test]` attribute `bpf_map_write = CONST`,
which targets a `BpfMapWrite` constant:

```rust,ignore
use ktstr::prelude::*;

const TRIGGER_FAULT: BpfMapWrite = BpfMapWrite {
    map_name_suffix: ".bss",   // matched against discovered maps
    offset: 42,                // byte offset within the map's value
    value: 1,                  // u32 written by the host
};

#[ktstr_test(bpf_map_write = TRIGGER_FAULT, expect_err = true)]
fn fault_then_inspect(ctx: &Ctx) -> Result<AssertResult> {
    // The host has already written `1` at `.bss + 42` before
    // the scenario started. Capture and inspect the resulting
    // scheduler state mid-run.
    /* bridge wiring + Op::snapshot + Snapshot::new as above */
    Ok(AssertResult::pass())
}
```

The write is event-driven: the host polls for BPF map
discoverability (scheduler loaded), polls the SHM ring for
scenario start, then writes the configured u32 at the configured
offset. Only `BPF_MAP_TYPE_ARRAY` maps are supported; the framework
finds the map by `map_name_suffix` (e.g. `".bss"`) via
`BpfMapAccessor::find_map`. See [Monitor → BPF map writes](../architecture/monitor.md)
for the prerequisites (vmlinux) and the full host-side
contract.

Read+write workflows then compose naturally: the test pre-seeds
guest state with `bpf_map_write`, lets the scheduler run, and
asserts on the resulting state with `Op::snapshot` + the
[`Snapshot`](#reading-the-captured-report) accessor:

1. **Write (pre-scenario)** — `bpf_map_write` flips a `.bss` flag
   the scheduler reads.
2. **Run** — the scenario's ops drive workload behavior; the
   scheduler reacts to the flag.
3. **Read (mid-scenario)** — `Op::snapshot("after")` captures the
   scheduler state at the chosen point.
4. **Assert** — `Snapshot::var(...).as_u64()` /
   `Snapshot::map(...).find(...).get(...).as_*()` verifies the
   reaction. Errors carry the available alternatives so a typo or
   stale field name surfaces before the test author hand-edits the
   case.

The write side is a single one-shot poke at scheduler-load time;
there is no `Op` variant for runtime writes. Ergonomic mid-scenario
state mutation is reserved for cases where the scheduler itself
exports a writable interface (sysfs, debugfs, BPF map command
interface) and the test invokes that interface from a workload
process.
