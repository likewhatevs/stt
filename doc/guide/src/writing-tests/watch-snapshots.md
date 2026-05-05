# Watch Snapshots

A **watch snapshot** registers a hardware data-write watchpoint on a
named kernel symbol. The host arms a CPU debug register; the produced
captures share the [`Snapshot`] accessor surface documented in
[Snapshots](snapshots.md).

`Op::watch_snapshot("symbol")` is the **write-driven** capture
trigger. Use it when the question is "what does the scheduler look
like *whenever the kernel touches X*?" rather than "what does it
look like at this point in my scenario?". For time-driven capture,
use [`Op::snapshot`](snapshots.md) instead.

## How it works

The full pipeline is implemented and tested end-to-end:

1. `Op::watch_snapshot(symbol)` registers the symbol via SHM.
2. The freeze coordinator resolves the KVA from the vmlinux ELF,
   validates 4-byte alignment, and arms a DR1/DR2/DR3 slot via
   `KVM_SET_GUEST_DEBUG`.
3. When the guest writes to the watched address, `KVM_EXIT_DEBUG`
   fires with the corresponding `dr6` bit set.
4. The coordinator captures via `freeze_and_capture` and stores
   the report in the `SnapshotBridge` under the symbol tag.
5. The report is also mirrored to a sidecar JSON file for
   post-hoc inspection.

The per-scenario cap of [`MAX_WATCH_SNAPSHOTS`] (= 3) is enforced
(DR0 is reserved for the error-class exit_kind trigger; DR1-3 are
available for user watches). A 4th `Op::watch_snapshot` fails the
step with a "cap exceeded" message. Symbol-resolution failures
bail immediately so a typo surfaces visibly.

`Op::watch_snapshot` covers the full pipeline: registration,
cap enforcement, symbol resolution, hardware arming, and
automatic capture on write.

## Issuing a watch

```rust,ignore
use ktstr::prelude::*;

let steps = vec![Step {
    setup: vec![CgroupDef::named("workers").workers(2)].into(),
    ops: vec![
        Op::watch_snapshot("bss.scx_ktstr.alloc_count"),
        Op::watch_snapshot("kernel.jiffies"),
    ],
    hold: HoldSpec::FULL,
}];
execute_steps(ctx, steps)?;
```

Each `Op::watch_snapshot` invokes the active [`SnapshotBridge`]'s
`register_watch` callback with the symbol string. On success, the
callback is responsible for arming a hardware watchpoint that will
fire whenever the guest writes to the symbol's address. Each fire
produces one capture, tagged with the symbol path itself.

## Wiring the bridge

A watch-capable bridge needs both a capture callback and a
`register_watch` callback:

```rust,ignore
use ktstr::prelude::*;

let cb: CaptureCallback = std::sync::Arc::new(|_name| {
    Some(FailureDumpReport::default())
});
let reg: WatchRegisterCallback = std::sync::Arc::new(|symbol: &str| {
    // Production: resolve the symbol via BTF + kallsyms, allocate a
    // free DR register, arm via KVM_SET_GUEST_DEBUG. Tests: record
    // the symbol and return Ok.
    println!("would arm watchpoint on {symbol}");
    Ok(())
});

let bridge = SnapshotBridge::new(cb).with_watch_register(reg);
let _guard = bridge.set_thread_local();
```

A bridge built only with `SnapshotBridge::new(cb)` (no
`with_watch_register`) rejects every `Op::watch_snapshot` with an
error pointing the operator at the missing wiring.

## Symbol path conventions

The bridge's `register_watch` callback owns symbol resolution. Two
naming conventions are documented on [`Op::WatchSnapshot`]:

- `"bss.<obj>.<field>"` — scheduler program BTF Datasec walk +
  per-section offset → guest KVA. Use for fields the scheduler
  declares in its own `.bss` / `.data` / `.rodata`.
- `"kernel.<symbol>[.<field>...]"` — vmlinux BTF + kallsyms (with
  per-CPU offset for per-CPU symbols) → KVA. Use for kernel-side
  symbols.

The `register_watch` callback decides what shapes it accepts; the
strings above are the conventions production wiring follows. A
test-side callback can accept anything it wants — the e2e tests in
`tests/snapshot_e2e.rs` use `"kernel.a"` / `"kernel.b"` / etc. for
the cap-enforcement test and `"exit_kind"` for the in-VM test.

## Maximum of 3 watches per scenario

```rust,ignore
pub const MAX_WATCH_SNAPSHOTS: usize = 3;
```

The bridge enforces a per-scenario cap of 3 successfully-registered
watches. The number is tied to x86_64 CPU debug registers: DR0 is
reserved for the existing `*scx_root->exit_kind` watchpoint that
drives the error-class freeze trigger; DR1, DR2, and DR3 are the
three slots available for on-demand watches.

A 4th `Op::watch_snapshot` in the same scenario fails the step with
"cap exceeded" when the cap is exceeded:

```rust,ignore
let steps = vec![Step {
    setup: vec![CgroupDef::named("cg").workers(2)].into(),
    ops: vec![
        Op::watch_snapshot("kernel.a"),
        Op::watch_snapshot("kernel.b"),
        Op::watch_snapshot("kernel.c"),
        Op::watch_snapshot("kernel.d"),  // <-- cap exceeded
    ],
    hold: HoldSpec::FULL,
}];
let result = execute_steps(ctx, steps)?;
assert!(!result.passed);
// One AssertDetail carries the cap-exceeded message:
//   "Op::WatchSnapshot cap exceeded: scenario already registered 3
//    watchpoints (DR1-3 occupied; DR0 reserved for the error-class
//    exit_kind trigger)..."
```

A failed register (cap exceeded, callback error, missing
`register_watch`) does **not** consume a slot. The bridge rolls the
count back so the scenario can keep trying with different symbols up
to the cap.

## Failure modes

The register callback is the single integration point where
production resolution can fail. The reasons documented on
[`WatchRegisterCallback`]:

- The symbol path does not resolve (BTF lookup miss, kallsyms miss,
  per-CPU offset unavailable).
- The resolved KVA is not 4-byte aligned (DR_LEN_4 requires
  `addr & 0x3 == 0` per Intel SDM Vol. 3B Chapter 17).
- All three available DR registers (DR1-3) are already allocated
  inside the host's KVM plumbing.
- `KVM_SET_GUEST_DEBUG` rejected the arm.

When the callback returns `Err(reason)`, the executor bails the step
immediately with a message containing the symbol and the failure
reason. Silent degradation is deliberately avoided — a watch that
never fires would look identical to a healthy passing run, and the
test author would never notice the captures were missing.

## DR0 (exit_kind) is separate

The existing error-class freeze trigger watches
`*scx_root->exit_kind` on DR0 and is **not** an `Op::watch_snapshot`
slot. It is wired by the freeze coordinator independently to detect
`SCX_EXIT_ERROR` writes and drive the failure-dump pipeline. That
trigger is unrelated to the on-demand watch surface — it always
runs, regardless of whether a scenario declares any
`Op::watch_snapshot` ops. The cap of 3 reflects the three
remaining DR slots after DR0 is held back.

For tests that want the failure dump produced by `SCX_EXIT_ERROR`,
nothing needs to be declared; the trigger fires automatically and
the dump is written to `{sidecar_dir()}/{test_name}.failure-dump.json`.
The watch-snapshot in-VM test in `tests/snapshot_e2e.rs` reads that
file back and feeds it through the [`Snapshot`] accessor as a way
to demonstrate the full read path.

## Reading captures

Once a watchpoint fires, the resulting report is stored on the bridge
under the tag and read back exactly as `Op::snapshot` captures are.
Every accessor — `Snapshot::map`, `Snapshot::var`,
`SnapshotMap::at` / `find` / `filter` / `max_by`, dotted-path walks,
typed terminal reads — is shared. See [Snapshots](snapshots.md) for
the full surface.

[`MAX_WATCH_SNAPSHOTS`]: #maximum-of-3-watches-per-scenario
[`Op::WatchSnapshot`]: #symbol-path-conventions
[`Snapshot`]: snapshots.md#reading-the-captured-report
[`SnapshotBridge`]: snapshots.md#wiring-the-bridge
[`WatchRegisterCallback`]: #failure-modes
