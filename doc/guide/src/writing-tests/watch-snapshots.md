# Watch Snapshots

A **watch snapshot** registers a hardware data-write watchpoint on a
named kernel symbol. The host arms a watchpoint slot via the guest's
hardware debug facilities; the produced captures share the
[`Snapshot`] accessor surface documented in [Snapshots](snapshots.md).

Watch snapshots are supported on x86_64 and aarch64 KVM hosts. The
slot terminology below is arch-neutral — each architecture's KVM
plumbing maps the slots onto its native hardware-watchpoint
facility (debug registers on x86_64, hardware watchpoints on
aarch64).

`Op::watch_snapshot("symbol")` is the **write-driven** capture
trigger. Use it when the question is "what does the scheduler look
like *whenever the kernel touches X*?" rather than "what does it
look like at this point in my scenario?". For time-driven capture,
use [`Op::snapshot`](snapshots.md) instead.

## How it works

The full pipeline is implemented and tested end-to-end:

1. `Op::watch_snapshot(symbol)` registers the symbol via the
   virtio-console port 1 `MSG_TYPE_SNAPSHOT_REQUEST` TLV frame.
2. The freeze coordinator resolves the KVA from the vmlinux ELF,
   validates 4-byte alignment, and arms a free user watchpoint slot
   via `KVM_SET_GUEST_DEBUG`.
3. When the guest writes to the watched address, the corresponding
   debug exit fires and the host identifies which slot tripped.
4. The coordinator captures via `freeze_and_capture` and stores
   the report in the `SnapshotBridge` under the symbol tag.
5. The report is also mirrored to a sidecar JSON file for
   post-hoc inspection.

The per-scenario cap of [`MAX_WATCH_SNAPSHOTS`] (= 3) is enforced
(slot 0 is reserved for the error-class exit_kind trigger; the
remaining 3 slots are available for user watches). A 4th
`Op::watch_snapshot` fails the step with a "cap exceeded" message.
Symbol-resolution failures bail immediately so a typo surfaces
visibly.

`Op::watch_snapshot` covers the full pipeline: registration,
cap enforcement, symbol resolution, hardware arming, and
automatic capture on write.

## Issuing a watch

```rust,ignore
use ktstr::prelude::*;

let steps = vec![Step {
    setup: vec![CgroupDef::named("workers").workers(2)].into(),
    ops: vec![
        Op::watch_snapshot("jiffies_64"),
        Op::watch_snapshot("scx_watchdog_timestamp"),
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

> **In `#[ktstr_test]` scenarios that boot a VM, the bridge is wired
> automatically.** Use `post_vm` to read captures from
> `VmResult::snapshot_bridge`. Do **not** install a thread-local
> bridge inside the scenario function — the in-VM
> `Op::watch_snapshot` registers via the virtio-console port 1
> `MSG_TYPE_SNAPSHOT_REQUEST` TLV frame, the host coordinator
> arms the watchpoint and stores captures on the bridge it owns,
> and the test reads them after the VM exits.
>
> The `set_thread_local` pattern below is for **host-side unit
> tests** that exercise the executor in process without booting a
> guest.

A watch-capable bridge for host-side unit tests needs both a capture
callback and a `register_watch` callback:

```rust,ignore
use ktstr::prelude::*;

let cb: CaptureCallback = std::sync::Arc::new(|_name| {
    Some(FailureDumpReport::default())
});
let reg: WatchRegisterCallback = std::sync::Arc::new(|symbol: &str| {
    // Host-side unit tests: record the symbol and return Ok. In a
    // booted VM, the host coordinator's pipeline runs instead —
    // see arm_user_watchpoint in src/vmm/freeze_coord.rs.
    println!("would arm watchpoint on {symbol}");
    Ok(())
});

let bridge = SnapshotBridge::new(cb).with_watch_register(reg);
let _guard = bridge.set_thread_local();
```

A bridge built only with `SnapshotBridge::new(cb)` (no
`with_watch_register`) rejects every `Op::watch_snapshot` with an
error pointing the operator at the missing wiring.

## Symbol resolution

Production resolution is a verbatim match against the vmlinux ELF
symbol table. The freeze coordinator walks `Elf::syms` and accepts
the symbol whose strtab entry equals the requested string byte-for-byte
— there is no prefix stripping, BTF lookup, kallsyms walk, or per-CPU
offset arithmetic. Use the exact name `nm vmlinux` would print:

- `"jiffies_64"` — the kernel's monotonic tick counter.
- `"scx_watchdog_timestamp"` — sched_ext's watchdog timestamp.

> **Warning: high-frequency symbols soft-lock the guest.** Watching
> a symbol that the kernel writes every jiffie (e.g. `jiffies_64` at
> `HZ=1000`) fires 1000+ captures per second. Each capture freezes
> all vCPUs for the full dump pipeline. The guest spends almost all
> of its wall time paused, which is indistinguishable from a soft
> lock-up — schedulers stall, watchdogs fire, and the test wedges
> before any meaningful work runs. Pick symbols the kernel writes at
> scenario-relevant cadence (a state field, a per-event counter),
> not on every tick.

The string passed to `Op::watch_snapshot` must match a vmlinux ELF
symtab entry exactly; otherwise the step fails with
`symbol '...' not found in vmlinux symtab`. The
`register_watch` callback on a host-side test bridge can accept any
shape it wants — the e2e tests in `tests/snapshot_e2e.rs` use
`"kernel.a"` / `"kernel.b"` / etc. for the cap-enforcement test and
`"exit_kind"` for the in-VM test — but the `Op::watch_snapshot` ops
that flow through the production pipeline (in-VM scenarios with no
host-side bridge override) must use a verbatim ELF symbol.

## Maximum of 3 watches per scenario

```rust,ignore
pub const MAX_WATCH_SNAPSHOTS: usize = 3;
```

The bridge enforces a per-scenario cap of 3 successfully-registered
watches. The number is tied to the per-vCPU hardware-watchpoint
slots KVM exposes via `KVM_SET_GUEST_DEBUG`: slot 0 is reserved for
the existing `*scx_root->exit_kind` watchpoint that drives the
error-class freeze trigger; the remaining three user watchpoint
slots are available for on-demand watches.

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
//    watchpoints (3 user watchpoint slots occupied; slot 0 reserved for the
//    error-class exit_kind trigger)..."
```

A failed register (cap exceeded, callback error, missing
`register_watch`) does **not** consume a slot. The bridge rolls the
count back so the scenario can keep trying with different symbols up
to the cap.

## Failure modes

The register callback is the single integration point where
production resolution can fail. The reasons documented on
[`WatchRegisterCallback`]:

- The symbol does not match any vmlinux ELF symtab entry (typo,
  symbol stripped from the build, or a non-ELF kernel image).
- The resolved KVA is not 4-byte aligned (the 4-byte watch length
  the framework arms requires `addr & 0x3 == 0` on every supported
  architecture).
- All three available user watchpoint slots are already allocated
  inside the host's KVM plumbing.
- `KVM_SET_GUEST_DEBUG` rejected the arm.

When the callback returns `Err(reason)`, the executor bails the step
immediately with a message containing the symbol and the failure
reason. Silent degradation is deliberately avoided — a watch that
never fires would look identical to a healthy passing run, and the
test author would never notice the captures were missing.

## Slot 0 (exit_kind) is separate

The existing error-class freeze trigger watches
`*scx_root->exit_kind` on slot 0 and is **not** an
`Op::watch_snapshot` slot. It is wired by the freeze coordinator
independently to detect `SCX_EXIT_ERROR` writes and drive the
failure-dump pipeline. That trigger is unrelated to the on-demand
watch surface — it always runs, regardless of whether a scenario
declares any `Op::watch_snapshot` ops. The cap of 3 reflects the
three remaining user slots after slot 0 is held back.

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
[`Op::WatchSnapshot`]: #symbol-resolution
[`Snapshot`]: snapshots.md#reading-the-captured-report
[`SnapshotBridge`]: snapshots.md#wiring-the-bridge
[`WatchRegisterCallback`]: #failure-modes
