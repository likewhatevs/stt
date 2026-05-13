//! End-to-end regression tests for the failure-dump pipeline that
//! land on the always-on (no-new-infra) code paths exercised by
//! [`scx-ktstr`]'s existing CLI flags. The silent-drop fix branches
//! themselves (commit `42389221` "preserve failure-dump.json across
//! all err-exit paths") need targeted triggers that don't exist
//! today; the three scenarios below pin Captured-baseline +
//! clean-exit-gate-suppression invariants, which together regress
//! the FRAMING within which the silent-drop fixes operate.
//!
//! Three scenarios:
//!
//! 1. `scenario_watchdog_stall_captured_emit_schema` — a normal
//!    `--stall-after=1` watchdog stall reaches the late-trigger
//!    Captured dispatch (LateCaptureOutcome::Captured at
//!    src/vmm/freeze_coord/mod.rs:5244+/5287+ construction sites) and
//!    must produce `schema = SCHEMA_SINGLE` on disk. Pins the happy
//!    Captured-emit dispatch; a future regression that mis-dispatches
//!    to Degraded/Dual/Suppressed on the normal stall would surface
//!    here.
//!
//! 2. `scenario_clean_exit_gate_suppresses_dump` — a clean exit
//!    (no `--stall-after`, scheduler ends via `Drop`) drives the
//!    exit_kind gate's `kind < SCX_EXIT_ERROR` branch at
//!    src/vmm/freeze_coord/mod.rs:4842-4854 to suppress dump emit.
//!    Asserts no primary dump file AND no snapshot-tagged sibling
//!    files exist. Pins the gate's designed clean-exit suppression
//!    semantic; a regression that over-fires the gate (emitting
//!    dumps on clean exits) would surface here.
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
//! Silent-drop fix branches NOT exercised by these scenarios (each
//! deferred because the trigger is unavailable in the always-on
//! toolset):
//!
//! - #50(a) `sched_exit_final_pass` guard at
//!   src/vmm/freeze_coord/mod.rs:2675-2701 — fires only on the
//!   SCHED_EXIT pidfd POLLIN race with the BPF watchpoint latch;
//!   no `scx-ktstr` flag drives it.
//! - #50(b)/#52 rendezvous-timeout Degraded emit at
//!   src/vmm/freeze_coord/mod.rs:4710-4758 — fires only when the
//!   vCPU rendezvous misses its `FREEZE_RENDEZVOUS_TIMEOUT`
//!   (currently a 30s compile-time const at
//!   src/vmm/freeze_coord/state.rs:34). A short-rendezvous-timeout
//!   builder setter would unlock this, tracked separately.
//! - #51 GAP B KVA-translate failure at
//!   src/vmm/freeze_coord/mod.rs:4865-4884 — fires only when
//!   `translate_any_kva` returns `None` (slab page freed mid-
//!   rendezvous); needs synthetic injection.
//! - #51 GAP C BSS-Triggered override at
//!   src/vmm/freeze_coord/mod.rs:4887-4972 — fires only when the
//!   gate decides `false` but the BPF probe latch had already
//!   observed `Triggered`; needs a benign watchpoint flap that
//!   scx-ktstr has no CLI surface to drive (scx-ktstr has no
//!   clap-based CLI today; only `--stall-after`/`--degrade-after`/
//!   etc. drive bss-data flips).

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

fn scenario_clean_exit_gate_suppresses_dump(
    ctx: &ktstr::scenario::Ctx,
) -> Result<AssertResult> {
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
    ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
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
static __KTSTR_ENTRY_SILENT_DROP_CLEAN_EXIT_GATE_SUPPRESSION:
    ktstr::test_support::KtstrTestEntry =
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
    ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "silent_drop_watchdog_stall_captured_content",
        func: scenario_watchdog_stall_dump_populates_vcpu_regs_and_maps,
        scheduler: &KTSTR_SCHED,
        extra_sched_args: &["--stall-after=1"],
        watchdog_timeout: std::time::Duration::from_secs(3),
        duration: std::time::Duration::from_secs(10),
        expect_err: true,
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };
