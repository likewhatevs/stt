//! End-to-end regression tests for the failure-dump pipeline. The
//! first three scenarios pin the always-on framing (Captured-baseline
//! + clean-exit-gate-suppression) within which the silent-drop fixes
//! operate. The last two use the freeze coordinator's test-only seams
//! (`FREEZE_COORD_TEST_FORCE_TRANSLATE_NONE`,
//! `FREEZE_COORD_TEST_FORCE_BSS_TRIGGERED`) to force the
//! KVA-translate-failure path on demand — exercising the silent-drop
//! fix branches that real workloads only hit during timing-sensitive
//! teardown races.
//!
//! Five scenarios:
//!
//! 1. `scenario_watchdog_stall_captured_emit_schema` — a normal
//!    `--stall-after=1` watchdog stall reaches the late-trigger
//!    Captured dispatch and must produce `schema = SCHEMA_SINGLE` on
//!    disk. Pins the happy Captured-emit dispatch; a future
//!    regression that mis-dispatches to Degraded/Dual/Suppressed on
//!    the normal stall would surface here.
//!
//! 2. `scenario_clean_exit_gate_suppresses_dump` — a clean exit (no
//!    `--stall-after`, scheduler ends via `Drop`) drives the
//!    exit_kind gate's `kind < SCX_EXIT_ERROR` branch to suppress
//!    dump emit. Asserts no primary dump file AND no snapshot-tagged
//!    sibling files exist. Pins the gate's designed clean-exit
//!    suppression semantic; a regression that over-fires the gate
//!    (emitting dumps on clean exits) would surface here.
//!
//! 3. `scenario_watchdog_stall_dump_populates_vcpu_regs_and_maps`
//!    — like scenario 1 but asserts the Captured dump's `vcpu_regs`
//!    has at least one entry with a non-zero `instruction_pointer`,
//!    AND `maps` is non-empty. Pins the content-population invariant
//!    against a regression where the dump file lands with valid JSON
//!    but the BPF map enumeration or vCPU register capture silently
//!    dropped (a failure mode that would have an empty shell pass
//!    scenario 1's schema check).
//!
//! 4. `scenario_translate_none_with_latch_idle_suppresses_dump` —
//!    forces `FREEZE_COORD_TEST_FORCE_TRANSLATE_NONE = true` with
//!    the BPF latch left idle. Gate enters the translate-fail branch
//!    and suppresses; cross-reference's `bss_read_state` returns
//!    NotResolved/NotTriggered so no rescue fires. Asserts no dump.
//!    Pins the "translate-fail + idle latch → correctly suppressed"
//!    path — when both the gate AND the historical latch indicate
//!    no error, suppression is the right call.
//!
//! 5. `scenario_translate_none_with_latch_triggered_emits_dump` —
//!    forces BOTH `FREEZE_COORD_TEST_FORCE_TRANSLATE_NONE = true`
//!    AND `FREEZE_COORD_TEST_FORCE_BSS_TRIGGERED = true`. Gate
//!    decides to suppress via the translate-fail branch, but the
//!    BPF-latch rescue overrides and emits a Captured dump
//!    (`schema = SCHEMA_SINGLE`) instead. Pins the rescue branch —
//!    when the gate would suppress but the historical BPF latch
//!    reports an error-class exit, the dump must emit rather than
//!    silently drop.
//!
//! Silent-drop fix branches NOT exercised by these scenarios (each
//! deferred because the trigger is unavailable in the always-on
//! toolset):
//!
//! - The `sched_exit_final_pass` guard fires only on the SCHED_EXIT
//!   pidfd POLLIN race with the BPF watchpoint latch; no `scx-ktstr`
//!   flag drives it.
//! - The rendezvous-timeout Degraded emit fires only when the vCPU
//!   rendezvous misses its `FREEZE_RENDEZVOUS_TIMEOUT`. The
//!   `KtstrVmBuilder::rendezvous_timeout` setter unlocks this; the
//!   consumer scenario is tracked separately.

mod common;

use anyhow::Result;
use common::dump_paths::failure_dump_path;
use ktstr::assert::AssertResult;
use ktstr::prelude::SCHEMA_SINGLE;
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps};
use ktstr::test_support::{Scheduler, SchedulerSpec, sidecar_dir};

const KTSTR_SCHED: Scheduler =
    Scheduler::new("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Enumerate every snapshot-tagged sibling file the freeze coordinator
/// could have written for `test_name`. The tag glob is open-ended
/// (`{test_name}.snapshot.*.json`) so a future `SNAPSHOT_TAG_*`
/// addition inflates the scan without requiring a test-side update.
/// File-name pattern matches `src/vmm/freeze_coord/snapshot.rs`'s
/// `snapshot_tagged_path` (pub(super), so reproduced here). The
/// production helper strips `.failure-dump.json` from the stem before
/// appending `.snapshot.{safe_tag}.json` (snapshot.rs:385,398-399),
/// so the prefix-and-suffix match here aligns.
#[track_caller]
fn snapshot_sibling_files(test_name: &str) -> Vec<std::path::PathBuf> {
    let prefix = format!("{test_name}.snapshot.");
    let dir = sidecar_dir();
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    entries
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_str()
                .map(|n| n.starts_with(&prefix) && n.ends_with(".json"))
                .unwrap_or(false)
        })
        .map(|e| e.path())
        .collect()
}

fn scenario_watchdog_stall_captured_emit_schema(
    ctx: &ktstr::scenario::Ctx,
) -> Result<AssertResult> {
    let test_name = "silent_drop_watchdog_stall_captured_schema";
    let dump_path = failure_dump_path(test_name);

    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    let mut result = execute_steps(ctx, steps)?;

    let json = std::fs::read_to_string(&dump_path).map_err(|e| {
        anyhow::anyhow!(
            "Captured emit silently dropped the dump despite the framework-attached \
             sink at {} ({e})",
            dump_path.display()
        )
    })?;

    let value: serde_json::Value = serde_json::from_str(&json)
        .map_err(|e| anyhow::anyhow!("dump file is not valid JSON: {e}; payload: {json}"))?;

    let schema = value
        .get("schema")
        .and_then(|s| s.as_str())
        .ok_or_else(|| {
            anyhow::anyhow!("dump JSON missing top-level `schema` field; payload: {value}")
        })?;
    anyhow::ensure!(
        schema == SCHEMA_SINGLE,
        "Captured emit must produce SCHEMA_SINGLE ({SCHEMA_SINGLE:?}); got \
         schema={schema:?} (a `degraded` or `dual` schema here indicates the \
         freeze coordinator took a different dispatch arm than the happy \
         Captured path)"
    );

    let siblings = snapshot_sibling_files(test_name);
    anyhow::ensure!(
        siblings.is_empty(),
        "Captured emit must not leave snapshot-tagged sibling files when \
         dual_snapshot is off; found {}: {:?}",
        siblings.len(),
        siblings,
    );

    result.details.push(ktstr::assert::AssertDetail::new(
        ktstr::assert::DetailKind::Other,
        format!(
            "Captured emit produced schema={SCHEMA_SINGLE} dump at {}",
            dump_path.display()
        ),
    ));
    Ok(result)
}

fn scenario_clean_exit_gate_suppresses_dump(ctx: &ktstr::scenario::Ctx) -> Result<AssertResult> {
    let test_name = "silent_drop_clean_exit_gate_suppression";
    let dump_path = failure_dump_path(test_name);

    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::Fixed(std::time::Duration::from_secs(2)),
    }];
    let mut result = execute_steps(ctx, steps)?;

    anyhow::ensure!(
        !dump_path.exists(),
        "exit_kind gate must suppress dump emit on clean (non-error-class) exit \
         — found unexpected primary dump file at {}",
        dump_path.display(),
    );

    let siblings = snapshot_sibling_files(test_name);
    anyhow::ensure!(
        siblings.is_empty(),
        "exit_kind gate suppression must not leave snapshot-tagged sibling files; \
         found {}: {:?}",
        siblings.len(),
        siblings,
    );

    result.details.push(ktstr::assert::AssertDetail::new(
        ktstr::assert::DetailKind::Other,
        "clean exit produced no dump artifacts (primary path absent, no tagged siblings)",
    ));
    Ok(result)
}

fn scenario_watchdog_stall_dump_populates_vcpu_regs_and_maps(
    ctx: &ktstr::scenario::Ctx,
) -> Result<AssertResult> {
    let test_name = "silent_drop_watchdog_stall_captured_content";
    let dump_path = failure_dump_path(test_name);

    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    let mut result = execute_steps(ctx, steps)?;

    let json = std::fs::read_to_string(&dump_path).map_err(|e| {
        anyhow::anyhow!(
            "Captured invariant test: dump file missing at {} ({e})",
            dump_path.display()
        )
    })?;
    let value: serde_json::Value = serde_json::from_str(&json)
        .map_err(|e| anyhow::anyhow!("dump file is not valid JSON: {e}"))?;

    let vcpu_regs = value
        .get("vcpu_regs")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("Captured dump missing top-level `vcpu_regs` array"))?;
    let populated = vcpu_regs.iter().any(|s| {
        s.is_object()
            && s.get("instruction_pointer")
                .and_then(|ip| ip.as_u64())
                .is_some_and(|ip| ip != 0)
    });
    anyhow::ensure!(
        populated,
        "Captured dump's `vcpu_regs` has no entry with a non-zero instruction \
         pointer — vCPU register capture silently dropped before the dump landed"
    );

    let maps = value
        .get("maps")
        .and_then(|m| m.as_array())
        .ok_or_else(|| anyhow::anyhow!("Captured dump missing top-level `maps` array"))?;
    anyhow::ensure!(
        !maps.is_empty(),
        "Captured dump's `maps` array is empty — BPF map enumeration silently \
         dropped every entry, leaving the dump shell with no content"
    );

    let siblings = snapshot_sibling_files(test_name);
    anyhow::ensure!(
        siblings.is_empty(),
        "Captured emit must not leave snapshot-tagged sibling files when \
         dual_snapshot is off; found {}: {:?}",
        siblings.len(),
        siblings,
    );

    result.details.push(ktstr::assert::AssertDetail::new(
        ktstr::assert::DetailKind::Other,
        format!(
            "Captured dump populated vcpu_regs ({} entries) and maps ({} entries) at {}",
            vcpu_regs.len(),
            maps.len(),
            dump_path.display()
        ),
    ));
    Ok(result)
}

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_SILENT_DROP_WATCHDOG_STALL_CAPTURED_SCHEMA:
    ktstr::test_support::KtstrTestEntry = ktstr::test_support::KtstrTestEntry {
    name: "silent_drop_watchdog_stall_captured_schema",
    func: scenario_watchdog_stall_captured_emit_schema,
    scheduler: &KTSTR_SCHED,
    extra_sched_args: &["--stall-after=1"],
    watchdog_timeout: std::time::Duration::from_secs(3),
    duration: std::time::Duration::from_secs(10),
    expect_err: true,
    ..ktstr::test_support::KtstrTestEntry::DEFAULT
};

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_SILENT_DROP_CLEAN_EXIT_GATE_SUPPRESSION: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "silent_drop_clean_exit_gate_suppression",
        func: scenario_clean_exit_gate_suppresses_dump,
        scheduler: &KTSTR_SCHED,
        extra_sched_args: &[],
        watchdog_timeout: std::time::Duration::from_secs(10),
        duration: std::time::Duration::from_secs(3),
        expect_err: false,
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_SILENT_DROP_WATCHDOG_STALL_CAPTURED_CONTENT:
    ktstr::test_support::KtstrTestEntry = ktstr::test_support::KtstrTestEntry {
    name: "silent_drop_watchdog_stall_captured_content",
    func: scenario_watchdog_stall_dump_populates_vcpu_regs_and_maps,
    scheduler: &KTSTR_SCHED,
    extra_sched_args: &["--stall-after=1"],
    watchdog_timeout: std::time::Duration::from_secs(3),
    duration: std::time::Duration::from_secs(10),
    expect_err: true,
    ..ktstr::test_support::KtstrTestEntry::DEFAULT
};

/// RAII guard for the freeze coordinator's exit_kind gate test seams.
/// Sets `FREEZE_COORD_TEST_FORCE_TRANSLATE_NONE` and
/// `FREEZE_COORD_TEST_FORCE_BSS_TRIGGERED` on construction, restores
/// the prior values on Drop. Defends against state leakage between
/// sibling tests in the same process if the test panics between set
/// and assert — nextest's process-per-test isolation is the production
/// safety net; this guard is defense in depth and makes the
/// set/reset pairing explicit at the call site.
struct FreezeCoordTestSeamGuard {
    translate_none_was: bool,
    bss_triggered_was: bool,
}

impl FreezeCoordTestSeamGuard {
    fn set(translate_none: bool, bss_triggered: bool) -> Self {
        use std::sync::atomic::Ordering;
        let translate_none_was = ktstr::FREEZE_COORD_TEST_FORCE_TRANSLATE_NONE
            .swap(translate_none, Ordering::Relaxed);
        let bss_triggered_was =
            ktstr::FREEZE_COORD_TEST_FORCE_BSS_TRIGGERED.swap(bss_triggered, Ordering::Relaxed);
        Self {
            translate_none_was,
            bss_triggered_was,
        }
    }
}

impl Drop for FreezeCoordTestSeamGuard {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering;
        ktstr::FREEZE_COORD_TEST_FORCE_TRANSLATE_NONE
            .store(self.translate_none_was, Ordering::Relaxed);
        ktstr::FREEZE_COORD_TEST_FORCE_BSS_TRIGGERED
            .store(self.bss_triggered_was, Ordering::Relaxed);
    }
}

/// Forces the gate's `translate_any_kva` to return None
/// (FREEZE_COORD_TEST_FORCE_TRANSLATE_NONE = true) while leaving the
/// BPF `.bss` latch unset. Gate decides to suppress (translate-fail
/// branch), rescue's `bss_read_state` returns NotResolved/NotTriggered
/// (production default), no rescue fires, dump is suppressed. Pins
/// the "translate-fail + idle latch → correctly suppressed" path that
/// the silent-drop fix preserves: when both the gate AND the
/// historical latch indicate no error, suppression is the right call.
fn scenario_translate_none_with_latch_idle_suppresses_dump(
    ctx: &ktstr::scenario::Ctx,
) -> Result<AssertResult> {
    let _guard = FreezeCoordTestSeamGuard::set(true, false);
    let test_name = "silent_drop_translate_none_with_latch_idle";
    let dump_path = failure_dump_path(test_name);

    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::Fixed(std::time::Duration::from_secs(2)),
    }];
    let mut result = execute_steps(ctx, steps)?;

    anyhow::ensure!(
        !dump_path.exists(),
        "With FREEZE_COORD_TEST_FORCE_TRANSLATE_NONE set + BPF latch idle, the \
         gate must suppress dump emit (translate-fail branch suppresses; latch \
         confirms no historical error). Unexpected primary dump file at {}",
        dump_path.display(),
    );

    let siblings = snapshot_sibling_files(test_name);
    anyhow::ensure!(
        siblings.is_empty(),
        "Translate-fail-with-idle-latch suppression must not leave \
         snapshot-tagged sibling files; found {}: {:?}",
        siblings.len(),
        siblings,
    );

    result.details.push(ktstr::assert::AssertDetail::new(
        ktstr::assert::DetailKind::Other,
        "FORCE_TRANSLATE_NONE + idle BPF latch correctly suppressed dump \
         (translate-fail branch entered, no rescue fired)",
    ));
    Ok(result)
}

/// Forces both `translate_any_kva` to None and `bss_read_state` to
/// `Triggered`. Gate decides to suppress via the translate-fail
/// branch, but the BPF-latch rescue overrides and fires the dump
/// emit path anyway. Pins the silent-drop fix's rescue branch — when
/// the gate would suppress but the historical BPF latch reports an
/// error-class exit, the dump must emit (as a Captured SCHEMA_SINGLE
/// report) rather than silently drop. Regression here would mean the
/// rescue path stopped firing, re-introducing the silent-drop bug
/// for the KVA-translate-failure case.
fn scenario_translate_none_with_latch_triggered_emits_dump(
    ctx: &ktstr::scenario::Ctx,
) -> Result<AssertResult> {
    let _guard = FreezeCoordTestSeamGuard::set(true, true);
    let test_name = "silent_drop_translate_none_with_latch_triggered";
    let dump_path = failure_dump_path(test_name);

    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::Fixed(std::time::Duration::from_secs(2)),
    }];
    let mut result = execute_steps(ctx, steps)?;

    let json = std::fs::read_to_string(&dump_path).map_err(|e| {
        anyhow::anyhow!(
            "BSS-latch rescue must emit dump when gate suppresses (FORCE_TRANSLATE_NONE) \
             but latch reports Triggered (FORCE_BSS_TRIGGERED). Expected dump at {} \
             not found ({e})",
            dump_path.display()
        )
    })?;
    let value: serde_json::Value = serde_json::from_str(&json)
        .map_err(|e| anyhow::anyhow!("rescue-emitted dump is not valid JSON: {e}"))?;

    let schema = value
        .get("schema")
        .and_then(|s| s.as_str())
        .ok_or_else(|| {
            anyhow::anyhow!("rescue dump missing top-level `schema` field; payload: {value}")
        })?;
    anyhow::ensure!(
        schema == SCHEMA_SINGLE,
        "rescue-path dump must emit schema={SCHEMA_SINGLE:?} (Captured single-snapshot); \
         got schema={schema:?} (Degraded/Dual here would indicate the rescue arm \
         dispatched to a different code path than the regular Captured emit)"
    );

    let siblings = snapshot_sibling_files(test_name);
    anyhow::ensure!(
        siblings.is_empty(),
        "BSS-latch rescue must not leave snapshot-tagged sibling files when \
         dual_snapshot is off; found {}: {:?}. Regression here would mean the \
         rescue arm dispatched to a Dual-snapshot emit when it should be Single.",
        siblings.len(),
        siblings,
    );

    result.details.push(ktstr::assert::AssertDetail::new(
        ktstr::assert::DetailKind::Other,
        format!(
            "FORCE_TRANSLATE_NONE + FORCE_BSS_TRIGGERED drove BSS-latch rescue → \
             dump emitted at {}",
            dump_path.display()
        ),
    ));
    Ok(result)
}

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_SILENT_DROP_TRANSLATE_NONE_LATCH_IDLE:
    ktstr::test_support::KtstrTestEntry = ktstr::test_support::KtstrTestEntry {
    name: "silent_drop_translate_none_with_latch_idle",
    func: scenario_translate_none_with_latch_idle_suppresses_dump,
    scheduler: &KTSTR_SCHED,
    extra_sched_args: &[],
    watchdog_timeout: std::time::Duration::from_secs(10),
    duration: std::time::Duration::from_secs(3),
    expect_err: false,
    ..ktstr::test_support::KtstrTestEntry::DEFAULT
};

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_SILENT_DROP_TRANSLATE_NONE_LATCH_TRIGGERED:
    ktstr::test_support::KtstrTestEntry = ktstr::test_support::KtstrTestEntry {
    name: "silent_drop_translate_none_with_latch_triggered",
    func: scenario_translate_none_with_latch_triggered_emits_dump,
    scheduler: &KTSTR_SCHED,
    extra_sched_args: &[],
    watchdog_timeout: std::time::Duration::from_secs(10),
    duration: std::time::Duration::from_secs(3),
    expect_err: false,
    ..ktstr::test_support::KtstrTestEntry::DEFAULT
};
