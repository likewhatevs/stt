//! End-to-end test for the on-demand diagnostic snapshot pipeline.
//!
//! Covers the wiring between [`Op::Snapshot`] / [`Op::WatchSnapshot`]
//! and the host-side [`SnapshotBridge`]:
//!   1. Install a [`SnapshotBridge`] in the executor's thread.
//!   2. Run a step sequence whose ops include `Op::snapshot(name)`
//!      and `Op::watch_snapshot(symbol)` — the executor invokes
//!      the bridge's capture / register callbacks.
//!   3. After the scenario finishes, drain the bridge and verify
//!      the report was stored under the requested name.
//!   4. Use the [`Snapshot`] accessor to query a [`FailureDumpReport`]
//!      shape via the public surface.
//!
//! Two layers of coverage:
//!
//! - **VM-free smoke pass** (the `#[test]` functions below): the
//!   capture callback returns a default or hand-crafted
//!   [`FailureDumpReport`] so the test exercises the executor +
//!   bridge + Op-variant pipeline without booting a guest. The
//!   kernel-grounded accessor traversal is covered by the in-crate
//!   unit tests in `src/scenario/snapshot.rs` which can build
//!   synthetic `RenderedValue::Struct` trees against the
//!   `#[non_exhaustive]` types directly.
//!
//! - **In-VM integration** (the `#[ktstr_test]`-registered scenarios
//!   at the bottom of this file): boot scx-ktstr, install a
//!   [`SnapshotBridge`] inside the scenario function, run
//!   `Op::snapshot` / `Op::watch_snapshot`, and verify the bridge's
//!   captured / registered state. The watch-fire test deserializes
//!   the host-written failure dump JSON (sidecar dir) into a real
//!   [`FailureDumpReport`] so the [`Snapshot`] accessor walks
//!   live BTF-rendered scheduler `.bss` fields.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use ktstr::cgroup::CgroupManager;
use ktstr::prelude::{
    CaptureCallback, FailureDumpReport, MAX_WATCH_SNAPSHOTS, Snapshot, SnapshotBridge,
    WatchRegisterCallback,
};
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Op, Step, execute_steps};
use ktstr::test_support::{Scheduler, SchedulerSpec, sidecar_dir};
use ktstr::topology::TestTopology;

#[test]
fn snapshot_op_drives_bridge_and_stores_report_under_name() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_clone = Arc::clone(&calls);

    let cb: CaptureCallback = Arc::new(move |_name: &str| {
        calls_clone.fetch_add(1, Ordering::Relaxed);
        Some(FailureDumpReport::default())
    });
    let bridge = SnapshotBridge::new(cb);
    let bridge_handle = bridge.clone();

    let _guard = bridge.set_thread_local();

    let cgroups = CgroupManager::new("/nonexistent/snapshot_e2e");
    let vm_topo = ktstr::prelude::Topology::new(1, 1, 2, 1);
    let topo = TestTopology::from_vm_topology(&vm_topo);
    let ctx = Ctx::builder(&cgroups, &topo).build();

    let steps = vec![Step {
        setup: Vec::<ktstr::scenario::ops::CgroupDef>::new().into(),
        ops: vec![Op::snapshot("after_setup"), Op::snapshot("after_workload")],
        hold: HoldSpec::Fixed(std::time::Duration::from_millis(1)),
    }];
    let result = execute_steps(&ctx, steps).expect("execute_steps must succeed");
    assert!(result.passed, "scenario must pass: {:?}", result.details);

    assert_eq!(
        calls.load(Ordering::Relaxed),
        2,
        "bridge.capture must have fired exactly once per Op::Snapshot"
    );

    let captured = bridge_handle.drain();
    assert_eq!(captured.len(), 2, "two captures expected");
    assert!(captured.contains_key("after_setup"));
    assert!(captured.contains_key("after_workload"));

    // Build a Snapshot view over the default report. The default
    // report has no maps, so the accessor must surface
    // MapNotFound; this exercises the public Snapshot constructor
    // and error-path roundtrip.
    let report = captured.get("after_setup").unwrap();
    let snap = Snapshot::new(report);
    assert_eq!(snap.map_count(), 0, "default report has no maps");
    let err = snap
        .map("nonexistent")
        .expect_err("must fail on default report");
    assert!(err.to_string().contains("nonexistent"));
}

#[test]
fn snapshot_op_with_no_bridge_is_a_no_op() {
    let cgroups = CgroupManager::new("/nonexistent/snapshot_e2e_no_bridge");
    let vm_topo = ktstr::prelude::Topology::new(1, 1, 2, 1);
    let topo = TestTopology::from_vm_topology(&vm_topo);
    let ctx = Ctx::builder(&cgroups, &topo).build();

    let steps = vec![Step {
        setup: Vec::<ktstr::scenario::ops::CgroupDef>::new().into(),
        ops: vec![Op::snapshot("orphan")],
        hold: HoldSpec::Fixed(std::time::Duration::from_millis(1)),
    }];
    let result = execute_steps(&ctx, steps).expect("execute_steps with no bridge must succeed");
    assert!(
        result.passed,
        "scenario without a bridge must still pass: {:?}",
        result.details
    );
}

#[test]
fn snapshot_op_with_failing_capture_does_not_abort_scenario() {
    let cb: CaptureCallback = Arc::new(|_| None);
    let bridge = SnapshotBridge::new(cb);
    let bridge_handle = bridge.clone();
    let _guard = bridge.set_thread_local();

    let cgroups = CgroupManager::new("/nonexistent/snapshot_e2e_failing");
    let vm_topo = ktstr::prelude::Topology::new(1, 1, 2, 1);
    let topo = TestTopology::from_vm_topology(&vm_topo);
    let ctx = Ctx::builder(&cgroups, &topo).build();

    let steps = vec![Step {
        setup: Vec::<ktstr::scenario::ops::CgroupDef>::new().into(),
        ops: vec![Op::snapshot("doomed")],
        hold: HoldSpec::Fixed(std::time::Duration::from_millis(1)),
    }];
    let result = execute_steps(&ctx, steps).expect("execute_steps must succeed");
    assert!(result.passed);
    assert!(bridge_handle.is_empty());
}

#[test]
fn watch_snapshot_op_drives_register_callback() {
    let attempts = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let attempts_clone = Arc::clone(&attempts);
    let cb: CaptureCallback = Arc::new(|_| Some(FailureDumpReport::default()));
    let reg: WatchRegisterCallback = Arc::new(move |symbol: &str| {
        attempts_clone.lock().unwrap().push(symbol.to_string());
        Ok(())
    });
    let bridge = SnapshotBridge::new(cb).with_watch_register(reg);
    let bridge_handle = bridge.clone();
    let _g = bridge.set_thread_local();

    let cgroups = CgroupManager::new("/nonexistent/snapshot_watch_e2e");
    let vm_topo = ktstr::prelude::Topology::new(1, 1, 2, 1);
    let topo = TestTopology::from_vm_topology(&vm_topo);
    let ctx = Ctx::builder(&cgroups, &topo).build();

    let steps = vec![Step {
        setup: Vec::<ktstr::scenario::ops::CgroupDef>::new().into(),
        ops: vec![
            Op::watch_snapshot("bss.scx_ktstr.alloc_count"),
            Op::watch_snapshot("kernel.jiffies"),
        ],
        hold: HoldSpec::Fixed(std::time::Duration::from_millis(1)),
    }];
    let result = execute_steps(&ctx, steps).expect("execute_steps must succeed");
    assert!(result.passed, "scenario must pass: {:?}", result.details);
    let recorded = attempts.lock().unwrap().clone();
    assert_eq!(recorded.len(), 2);
    assert_eq!(recorded[0], "bss.scx_ktstr.alloc_count");
    assert_eq!(recorded[1], "kernel.jiffies");
    assert_eq!(bridge_handle.watch_count(), 2);
}

#[test]
fn watch_snapshot_op_max_3_per_scenario_errors_fourth() {
    let cb: CaptureCallback = Arc::new(|_| Some(FailureDumpReport::default()));
    let reg: WatchRegisterCallback = Arc::new(|_symbol: &str| Ok(()));
    let bridge = SnapshotBridge::new(cb).with_watch_register(reg);
    let _g = bridge.set_thread_local();

    let cgroups = CgroupManager::new("/nonexistent/snapshot_watch_e2e_max");
    let vm_topo = ktstr::prelude::Topology::new(1, 1, 2, 1);
    let topo = TestTopology::from_vm_topology(&vm_topo);
    let ctx = Ctx::builder(&cgroups, &topo).build();

    let steps = vec![Step {
        setup: Vec::<ktstr::scenario::ops::CgroupDef>::new().into(),
        ops: vec![
            Op::watch_snapshot("kernel.a"),
            Op::watch_snapshot("kernel.b"),
            Op::watch_snapshot("kernel.c"),
            Op::watch_snapshot("kernel.d"),
        ],
        hold: HoldSpec::Fixed(std::time::Duration::from_millis(1)),
    }];
    let result = execute_steps(&ctx, steps).expect("execute_steps returns Ok with stamped error");
    assert!(
        !result.passed,
        "scenario must fail when 4th watchpoint is registered (cap exceeded)"
    );
    let detail = result
        .details
        .iter()
        .find(|d| d.message.contains("cap exceeded"))
        .expect("AssertResult must carry the cap-exceeded message");
    assert!(detail.message.contains("WatchSnapshot"));
    assert_eq!(MAX_WATCH_SNAPSHOTS, 3);
}

#[test]
fn watch_snapshot_op_unresolvable_symbol_bails_immediately() {
    let cb: CaptureCallback = Arc::new(|_| Some(FailureDumpReport::default()));
    let reg: WatchRegisterCallback = Arc::new(|symbol: &str| {
        Err(format!(
            "symbol '{symbol}' did not resolve via BTF + kallsyms"
        ))
    });
    let bridge = SnapshotBridge::new(cb).with_watch_register(reg);
    let _g = bridge.set_thread_local();

    let cgroups = CgroupManager::new("/nonexistent/snapshot_watch_e2e_resolve");
    let vm_topo = ktstr::prelude::Topology::new(1, 1, 2, 1);
    let topo = TestTopology::from_vm_topology(&vm_topo);
    let ctx = Ctx::builder(&cgroups, &topo).build();

    let steps = vec![Step {
        setup: Vec::<ktstr::scenario::ops::CgroupDef>::new().into(),
        ops: vec![Op::watch_snapshot("kernel.absent_symbol")],
        hold: HoldSpec::Fixed(std::time::Duration::from_millis(1)),
    }];
    let result = execute_steps(&ctx, steps).expect("execute_steps returns Ok with stamped error");
    assert!(!result.passed, "unresolvable symbol must fail the step");
    let detail = result
        .details
        .iter()
        .find(|d| d.message.contains("did not resolve"))
        .expect("AssertResult must surface the resolution error");
    assert!(detail.message.contains("absent_symbol"));
}

// ---------------------------------------------------------------------------
// In-VM integration: scx-ktstr boots, scenario function installs a
// SnapshotBridge inside the guest, exercises Op::snapshot /
// Op::watch_snapshot end-to-end. The host-side automatic bridge wiring
// is a follow-up; these tests install the bridge in the scenario thread
// before `execute_steps`, the same shape the executor's `with_active_bridge`
// path already drives in production code.
// ---------------------------------------------------------------------------

const KTSTR_SCHED: Scheduler =
    Scheduler::new("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Synthetic single-`.bss`-map `FailureDumpReport` used by the
/// no-stall snapshot scenario. Hand-built JSON — out-of-crate code
/// cannot directly construct `FailureDumpMap` (`#[non_exhaustive]`),
/// so the canonical workaround is `serde_json::from_str`. Mirrors
/// the libbpf `<obj>.<section>` naming convention so
/// `Snapshot::var(...)` discovers the global-section map and walks
/// its rendered Struct.
///
/// Pinned shape:
///   - one map named `bpf.bss` with `map_type = 2` (BPF_MAP_TYPE_ARRAY)
///   - rendered Struct value with one `Uint` member named `nr_cpus_onln`
///   - schema = SCHEMA_SINGLE
const SYNTHETIC_BSS_REPORT_JSON: &str = r#"{
    "schema": "single",
    "maps": [
        {
            "name": "bpf.bss",
            "map_type": 2,
            "value_size": 8,
            "max_entries": 1,
            "value": {
                "kind": "struct",
                "type_name": ".bss",
                "members": [
                    {
                        "name": "nr_cpus_onln",
                        "value": { "kind": "uint", "bits": 32, "value": 4 }
                    }
                ]
            }
        }
    ]
}"#;

/// Test 1: `Op::snapshot` runs inside scx-ktstr's guest VM and drives
/// the host-installed `SnapshotBridge`'s capture callback. The
/// callback returns a hand-crafted `FailureDumpReport`; the scenario
/// drains the bridge after `execute_steps` and walks the rendered
/// `.bss` shape via the public `Snapshot` accessor.
fn scenario_snapshot_op_captures_in_vm(
    ctx: &ktstr::scenario::Ctx,
) -> anyhow::Result<ktstr::assert::AssertResult> {
    // Capture callback: parses the synthetic JSON each call so the
    // closure stays `Send + Sync` without sharing a `FailureDumpReport`
    // (the type is `Clone` but parking it in an `Arc` would still work;
    // re-parsing keeps the call shape obvious).
    let cb: ktstr::prelude::CaptureCallback = std::sync::Arc::new(|_name: &str| {
        Some(
            serde_json::from_str::<FailureDumpReport>(SYNTHETIC_BSS_REPORT_JSON)
                .expect("synthetic JSON parses into FailureDumpReport"),
        )
    });
    let bridge = SnapshotBridge::new(cb);
    let bridge_handle = bridge.clone();
    let _bridge_guard = bridge.set_thread_local();

    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![Op::snapshot("test_snap")],
        hold: HoldSpec::FULL,
    }];
    let mut result = execute_steps(ctx, steps)?;

    // Drain the bridge — `Op::snapshot` must have driven `capture`,
    // and the synthetic report must be stored under the op's `name`.
    let captured = bridge_handle.drain();
    if !captured.contains_key("test_snap") {
        anyhow::bail!(
            "Op::snapshot('test_snap') did not store a report on the bridge; \
             captured keys: {:?}",
            captured.keys().collect::<Vec<_>>()
        );
    }
    let report = captured.get("test_snap").expect("contains_key checked");

    // 1. Snapshot has at least one map.
    let snap = Snapshot::new(report);
    if snap.map_count() == 0 {
        anyhow::bail!(
            "Op::snapshot captured a report but the report has no maps — \
             SnapshotBridge::capture lost the synthetic FailureDumpReport's \
             `maps` vec on store"
        );
    }

    // 2. Known map (`bpf.bss`) is present and its rendered value is
    //    a Struct with non-empty members (libbpf composes
    //    `<obj>.bss` for global-section maps; the synthetic JSON
    //    pins this naming).
    let bss = snap
        .map("bpf.bss")
        .map_err(|e| anyhow::anyhow!("Snapshot::map(\"bpf.bss\") failed: {e}"))?;
    let entry = bss.at(0);
    if !entry.is_present() {
        anyhow::bail!(
            "bpf.bss has no entry-at-0 — single-Value ARRAY rendering \
             expected (the synthetic report sets `value` on the map)"
        );
    }

    // 3. Field accessible via the dotted-path API. The synthetic
    //    JSON declares `nr_cpus_onln = 4` — exercise both the
    //    top-level `var()` shortcut (walks every global-section map)
    //    and the explicit map+entry+field walk.
    let via_var = snap
        .var("nr_cpus_onln")
        .as_u64()
        .map_err(|e| anyhow::anyhow!("Snapshot::var(\"nr_cpus_onln\").as_u64() failed: {e}"))?;
    if via_var != 4 {
        anyhow::bail!("Snapshot::var(\"nr_cpus_onln\") returned {via_var}, expected 4");
    }
    let via_path = entry
        .get("nr_cpus_onln")
        .as_u64()
        .map_err(|e| anyhow::anyhow!("entry.get(\"nr_cpus_onln\").as_u64() failed: {e}"))?;
    if via_path != 4 {
        anyhow::bail!("entry.get(\"nr_cpus_onln\") returned {via_path}, expected 4");
    }

    result.details.push(ktstr::assert::AssertDetail::new(
        ktstr::assert::DetailKind::Other,
        format!(
            "Op::snapshot('test_snap') captured {} map(s); bpf.bss has \
             nr_cpus_onln={via_var} via var() and {via_path} via dotted-path",
            snap.map_count()
        ),
    ));
    Ok(result)
}

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_SNAPSHOT_OP_IN_VM: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "snapshot_op_captures_in_vm",
        func: scenario_snapshot_op_captures_in_vm,
        scheduler: &KTSTR_SCHED,
        // No stall — exercise the no-error scenario path. duration
        // matches the scx-ktstr default-load tests in
        // `tests/ktstr_sched_tests.rs`.
        duration: std::time::Duration::from_secs(3),
        watchdog_timeout: std::time::Duration::from_secs(15),
        workers_per_cgroup: 2,
        // Skip auto-repro: this test drives bridge-side behaviour,
        // not scheduler correctness — a probe-attached repro VM
        // adds runtime without value when the scenario passes.
        auto_repro: false,
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

/// Compute the per-test failure-dump path. Mirrors
/// `tests/failure_dump_e2e.rs::failure_dump_path` (the freeze
/// coordinator writes `{sidecar_dir()}/{test_name}.failure-dump.json`
/// when an SCX_EXIT_ERROR_* trigger fires).
fn watch_snapshot_dump_path(test_name: &str) -> std::path::PathBuf {
    sidecar_dir().join(format!("{test_name}.failure-dump.json"))
}

/// Test 2: `Op::watch_snapshot` runs inside scx-ktstr's guest VM and
/// drives the host-installed `SnapshotBridge`'s `register_watch`
/// callback. Test invokes scx-ktstr with `--stall-after=1` so the
/// scheduler's `tp_btf/sched_ext_exit` handler fires
/// `SCX_EXIT_ERROR_STALL`, the freeze coordinator dumps the live
/// scheduler `.bss` to `{sidecar_dir()}/{test_name}.failure-dump.json`,
/// and the test reads that file back, deserializes into a
/// `FailureDumpReport`, manually drives `bridge.capture("exit_kind")`
/// to simulate the watchpoint fire, and walks the captured snapshot's
/// `stall` field to confirm the dump captured live exit state.
fn scenario_watch_snapshot_op_captures_exit_state(
    ctx: &ktstr::scenario::Ctx,
) -> anyhow::Result<ktstr::assert::AssertResult> {
    let dump_path = watch_snapshot_dump_path("watch_snapshot_op_captures_exit_state");

    // Track every symbol the executor passed through `register_watch`
    // — proves `Op::watch_snapshot` actually drove the bridge instead
    // of falling through the no-bridge no-op arm.
    let watch_attempts = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let watch_attempts_for_reg = std::sync::Arc::clone(&watch_attempts);
    let reg: WatchRegisterCallback = std::sync::Arc::new(move |symbol: &str| {
        watch_attempts_for_reg
            .lock()
            .expect("watch_attempts mutex is uncontended in the scenario thread")
            .push(symbol.to_string());
        Ok(())
    });

    // Capture callback reads the host-written dump JSON each fire.
    // Returning `None` is the "dump not yet written" signal — the
    // bridge logs a `tracing::warn!` and `capture()` returns `false`
    // without storing anything, mirroring the host's no-data
    // semantics.
    let dump_path_for_cb = dump_path.clone();
    let cb: CaptureCallback = std::sync::Arc::new(move |_name: &str| {
        let json = std::fs::read_to_string(&dump_path_for_cb).ok()?;
        serde_json::from_str::<FailureDumpReport>(&json).ok()
    });

    let bridge = SnapshotBridge::new(cb).with_watch_register(reg);
    let bridge_handle = bridge.clone();
    let _bridge_guard = bridge.set_thread_local();

    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![Op::watch_snapshot("exit_kind")],
        hold: HoldSpec::FULL,
    }];
    // `--stall-after=1` will stamp the scenario's AssertResult as
    // failed (scheduler-died); we verify our snapshot/watch
    // assertions on the host-side return value separately.
    // `expect_err: true` flips the failed AssertResult to PASS.
    let mut result = execute_steps(ctx, steps)?;

    // 1. The watch register callback was invoked with the requested
    //    symbol. Op::watch_snapshot('exit_kind') must have driven
    //    the bridge's register_watch — anything else means the
    //    op fell through to the no-bridge warn branch.
    let attempts = watch_attempts
        .lock()
        .expect("watch_attempts mutex is uncontended after scenario thread")
        .clone();
    if !attempts.iter().any(|s| s == "exit_kind") {
        anyhow::bail!(
            "Op::watch_snapshot('exit_kind') did not drive the register_watch \
             callback — recorded attempts: {attempts:?}"
        );
    }
    if bridge_handle.watch_count() != 1 {
        anyhow::bail!(
            "bridge watch_count={} — expected exactly one registered \
             watchpoint after Op::watch_snapshot('exit_kind')",
            bridge_handle.watch_count()
        );
    }

    // 2. Manually drive `bridge.capture("exit_kind")` to simulate the
    //    watchpoint firing on the kernel exit-state write. The
    //    capture callback reads the host-written failure-dump JSON
    //    and returns the live FailureDumpReport. Hardware-watchpoint
    //    plumbing (KVM_SET_GUEST_DEBUG / KVM_EXIT_DEBUG dispatch) is
    //    a follow-up; this manual fire lets the test verify the
    //    storage shape and accessor walk against real captured data.
    let captured_now = bridge_handle.capture("exit_kind");
    if !captured_now {
        anyhow::bail!(
            "bridge.capture('exit_kind') returned false — the failure-dump \
             JSON at {} was missing or unparseable, meaning the freeze \
             coordinator never wrote the dump (did SCX_EXIT_ERROR_STALL \
             fire? Watchdog default differs from --stall-after?)",
            dump_path.display()
        );
    }
    let captured = bridge_handle.drain();
    let report = captured.get("exit_kind").ok_or_else(|| {
        anyhow::anyhow!(
            "bridge drained but no snapshot under 'exit_kind' — \
             SnapshotBridge::capture stored under a different key"
        )
    })?;

    // 3. The captured snapshot contains the expected exit state.
    //    `stall` is set by scx-ktstr's `--stall-after=N` watchdog
    //    mechanism (main.bpf.c writes `stall = 1` before returning
    //    early from dispatch); the freeze coordinator captures the
    //    live `.bss` after the kernel sched_ext path emits
    //    SCX_EXIT_ERROR_STALL. Walking `Snapshot::var("stall")`
    //    proves we captured live error-state, not pre-init zeros.
    let snap = Snapshot::new(report);
    if snap.map_count() == 0 {
        anyhow::bail!(
            "captured FailureDumpReport has no maps — freeze coordinator \
             produced an empty dump"
        );
    }
    let stall = snap.var("stall").as_u64().map_err(|e| {
        anyhow::anyhow!(
            "Snapshot::var(\"stall\").as_u64() failed: {e} — the captured \
             dump's BTF-rendered .bss does not surface a `stall` field, or \
             scx-ktstr renamed it"
        )
    })?;
    if stall == 0 {
        anyhow::bail!(
            "captured snapshot's `stall` is 0 — the dump captured pre-stall \
             state (or scx-ktstr's --stall-after path never set the flag)"
        );
    }

    result.details.push(ktstr::assert::AssertDetail::new(
        ktstr::assert::DetailKind::Other,
        format!(
            "Op::watch_snapshot('exit_kind') drove register_watch \
             ({attempts:?}); manually-fired capture loaded \
             {} map(s) from {} with stall={stall}",
            snap.map_count(),
            dump_path.display(),
        ),
    ));
    Ok(result)
}

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_WATCH_SNAPSHOT_EXIT: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "watch_snapshot_op_captures_exit_state",
        func: scenario_watch_snapshot_op_captures_exit_state,
        scheduler: &KTSTR_SCHED,
        // --stall-after=1 forces SCX_EXIT_ERROR_STALL after 1s of
        // dispatching, mirroring `tests/failure_dump_e2e.rs`'s
        // mechanism for triggering a host-side dump.
        extra_sched_args: &["--stall-after=1"],
        // Watchdog timeout snug to the stall budget so the run
        // teardown stays inside the test duration.
        watchdog_timeout: std::time::Duration::from_secs(3),
        duration: std::time::Duration::from_secs(10),
        workers_per_cgroup: 2,
        // The scenario surfaces a failed AssertResult because the
        // scheduler intentionally dies; `expect_err: true` inverts
        // that to PASS. Snapshot/watch assertion failures bubble
        // up via `anyhow::bail!` (Err propagates as a runner-level
        // failure that `expect_err` cannot mask).
        expect_err: true,
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };
