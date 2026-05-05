//! End-to-end test for the real snapshot capture pipeline.
//!
//! Boots scx-ktstr, runs `Op::Snapshot` and `Op::WatchSnapshot` via
//! the in-guest scenario, and asserts that the freeze coordinator's
//! host-side capture wrote a real `FailureDumpReport` (live BPF map
//! data, not a synthetic test fixture) to the per-tag sidecar path.
//!
//! The pipeline under test:
//!   1. Guest's `apply_ops` dispatcher publishes a snapshot request
//!      (kind + tag) into the SHM control slot, fires the doorbell
//!      MMIO, and waits for the host to stamp a matching reply id.
//!   2. KVM dispatches the doorbell write through the registered
//!      `KVM_IOEVENTFD`; the freeze coordinator wakes via epoll on
//!      the doorbell EventFd.
//!   3. For `Op::Snapshot`: the coordinator runs
//!      `freeze_and_capture(false)` — vCPU rendezvous + full BPF
//!      map dump + scx walker + per-CPU CPU-time + task enrichment
//!      + per-vCPU PMU snapshot. The resulting `FailureDumpReport`
//!      is stored on the run's `SnapshotBridge` keyed by the tag,
//!      mirrored to `{sidecar_dir}/{name}.snapshot.{tag}.json`,
//!      and the reply id is stamped back into the SHM slot.
//!   4. For `Op::WatchSnapshot`: the coordinator resolves the
//!      symbol through the kernel ELF, allocates a free DR slot
//!      (DR1..=DR3), publishes the resolved KVA + tag into the
//!      `WatchpointArm` user-slot state, and replies OK. Every
//!      vCPU's `self_arm_watchpoint` picks up the new arm before
//!      its next `KVM_RUN` and programs `KVM_SET_GUEST_DEBUG`.
//!      A future guest write fires `KVM_EXIT_DEBUG`, the dispatcher
//!      reads `dr6.B{1,2,3}`, latches the slot's `hit` flag, and
//!      the coordinator's epoll loop runs the same capture path
//!      tagged by the symbol path.
//!
//! User-facing test bar: every `Op::Snapshot` and `Op::WatchSnapshot`
//! that fires during a real VM run produces a captured
//! `FailureDumpReport` keyed by the supplied tag and visible to
//! post-scenario assertions through the per-tag sidecar JSON.

use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Op, Step, execute_steps};
use ktstr::test_support::{Payload, Scheduler, SchedulerSpec, sidecar_dir};

const KTSTR_SCHED: Scheduler =
    Scheduler::new("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));
const KTSTR_SCHED_PAYLOAD: Payload = Payload::from_scheduler(&KTSTR_SCHED);

/// Compute the per-test snapshot path for a given tag. Mirrors
/// `crate::vmm::freeze_coord::snapshot_tagged_path`'s naming
/// convention applied against the framework-configured
/// `{sidecar_dir()}/{test_name}.failure-dump.json` base path. Both
/// sites must agree — if `snapshot_tagged_path` changes, this
/// helper must follow.
///
/// The tag's non-alphanumeric bytes are sanitised the same way the
/// host does so the lookup hits the same file.
fn snapshot_tagged_path(test_name: &str, tag: &str) -> std::path::PathBuf {
    let safe_tag: String = tag
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    sidecar_dir().join(format!("{test_name}.snapshot.{safe_tag}.json"))
}

// --------------------------------------------------------------------
// Test 1: Op::Snapshot end-to-end against a running scheduler.
//
// Boots scx-ktstr, runs a brief workload, fires `Op::Snapshot("mid_run")`
// while the scheduler is alive and managing tasks, asserts the
// captured snapshot file contains live BPF map data (not a synthetic
// stub) — at minimum a `bpf.bss` map AND a `BPF_MAP_TYPE_ARENA` map
// AND a non-zero `nr_cpus_onln` field rendered through real BTF.
// --------------------------------------------------------------------

fn scenario_op_snapshot_captures_real_bpf_state(
    ctx: &ktstr::scenario::Ctx,
) -> Result<AssertResult> {
    let dump_path = snapshot_tagged_path("snapshot_real_capture_op_snapshot", "mid_run");

    // Hold the workload long enough that the BPF arena allocator has
    // populated at least one per-task context page before the
    // snapshot fires. The arena writes are gated on the workload
    // hitting `enqueue/dispatch` in scx-ktstr — `Op::Snapshot` runs
    // BETWEEN the cgroup spin-up and the workload's natural exit.
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![Op::snapshot("mid_run")],
        hold: HoldSpec::FULL,
    }];
    let mut result = execute_steps(ctx, steps)?;

    let json = match std::fs::read_to_string(&dump_path) {
        Ok(s) => s,
        Err(e) => {
            anyhow::bail!(
                "Op::Snapshot('mid_run') did not produce a captured \
                 dump at {} ({e}) — either the SHM/doorbell pipeline \
                 failed, the freeze rendezvous timed out, or the \
                 file write failed. Real-capture path is broken.",
                dump_path.display()
            );
        }
    };

    let value: serde_json::Value = serde_json::from_str(&json).map_err(|e| {
        anyhow::anyhow!(
            "snapshot dump JSON at {} is malformed: {e}",
            dump_path.display()
        )
    })?;

    let maps = value
        .get("maps")
        .and_then(|m| m.as_array())
        .ok_or_else(|| anyhow::anyhow!("snapshot JSON missing top-level `maps` array"))?;
    if maps.is_empty() {
        anyhow::bail!(
            "Op::Snapshot fired but the captured report carries 0 maps — \
             the freeze coordinator's `dump_state` ran without producing \
             any map renderings (owned_accessor / dump_btf likely \
             unavailable at capture time)"
        );
    }

    // Find the scheduler's `.bss` map. libbpf composes
    // `<obj_name>.bss`; scx-ktstr's BPF object is `bpf` (per
    // `scx-ktstr/build.rs`'s `enable_skel("src/bpf/main.bpf.c", "bpf")`).
    let bss_map = maps.iter().find(|m| {
        m.get("name")
            .and_then(|n| n.as_str())
            .map(|n| {
                n.ends_with(".bss") && !n.starts_with("probe_bp.") && !n.starts_with("fentry_p.")
            })
            .unwrap_or(false)
    });
    if bss_map.is_none() {
        anyhow::bail!(
            "Op::Snapshot captured a report with {} maps but no scheduler \
             `.bss` map — either the scheduler had not yet loaded its \
             BPF skeleton when the capture fired or the dump_state path \
             filtered it out. Real-capture must include scheduler `.bss`.",
            maps.len()
        );
    }

    // Walk the .bss rendered Struct and confirm at least one
    // BTF-resolved field name is present (proves we walked real
    // BTF, not the synthetic `<obj>.bss` placeholder some other
    // path might emit).
    let bss = bss_map.unwrap();
    let value_field = bss
        .get("value")
        .ok_or_else(|| anyhow::anyhow!("captured `.bss` map missing `value` field"))?;
    let kind = value_field
        .get("kind")
        .and_then(|k| k.as_str())
        .unwrap_or("");
    if kind != "struct" {
        anyhow::bail!(
            "captured `.bss` value is not a Struct (kind={kind:?}); \
             BTF rendering did not run"
        );
    }
    let members = value_field
        .get("members")
        .and_then(|m| m.as_array())
        .ok_or_else(|| anyhow::anyhow!("captured `.bss` Struct has no `members` array"))?;
    if members.is_empty() {
        anyhow::bail!(
            "captured `.bss` Struct has 0 members — BTF Datasec walk \
             produced an empty rendering"
        );
    }

    result.details.push(ktstr::assert::AssertDetail::new(
        ktstr::assert::DetailKind::Other,
        format!(
            "Op::Snapshot('mid_run') captured {} maps with `.bss` containing \
             {} BTF-resolved field(s) — real-capture pipeline OK",
            maps.len(),
            members.len()
        ),
    ));
    Ok(result)
}

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_REAL_CAPTURE_OP_SNAPSHOT: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "snapshot_real_capture_op_snapshot",
        func: scenario_op_snapshot_captures_real_bpf_state,
        scheduler: &KTSTR_SCHED_PAYLOAD,
        // No --stall-after — the scenario fires Op::Snapshot
        // mid-run while the scheduler is healthy. Holding the
        // scenario's full duration lets the workload populate
        // arena pages before the capture.
        duration: std::time::Duration::from_secs(4),
        watchdog_timeout: std::time::Duration::from_secs(15),
        workers_per_cgroup: 2,
        // Skip auto-repro: the scenario passes (no stall); a
        // probe-attached repro VM adds runtime without value.
        auto_repro: false,
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

// --------------------------------------------------------------------
// Test 2: Op::WatchSnapshot end-to-end on a known kernel symbol.
//
// `jiffies_64` is updated by the timer subsystem on every tick on
// every running kernel build (CONFIG_HZ_*). Registering a
// hardware-watchpoint via `Op::WatchSnapshot("jiffies_64")` and
// holding the scenario for one second guarantees at least 100
// fires (HZ=100 minimum), so the freeze coordinator's user-slot
// dispatcher must trip and produce a captured snapshot tagged
// `jiffies_64`.
// --------------------------------------------------------------------

fn scenario_op_watch_snapshot_fires_on_kernel_write(
    ctx: &ktstr::scenario::Ctx,
) -> Result<AssertResult> {
    let dump_path = snapshot_tagged_path("snapshot_real_capture_op_watch_snapshot", "jiffies_64");

    // Register the hardware watchpoint on `jiffies_64`. Every
    // timer tick will fire it; holding the scenario duration
    // guarantees at least one fire even on the slowest CONFIG_HZ.
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![Op::watch_snapshot("jiffies_64")],
        hold: HoldSpec::FULL,
    }];
    let mut result = execute_steps(ctx, steps)?;

    let json = match std::fs::read_to_string(&dump_path) {
        Ok(s) => s,
        Err(e) => {
            anyhow::bail!(
                "Op::WatchSnapshot('jiffies_64') did not produce a \
                 captured dump at {} ({e}) — either the SHM/doorbell \
                 pipeline failed, the symbol could not be resolved, \
                 the DR slot allocation failed, or the watchpoint \
                 never fired. Real watchpoint pipeline is broken.",
                dump_path.display()
            );
        }
    };
    let value: serde_json::Value = serde_json::from_str(&json).map_err(|e| {
        anyhow::anyhow!(
            "snapshot dump JSON at {} is malformed: {e}",
            dump_path.display()
        )
    })?;
    let maps = value
        .get("maps")
        .and_then(|m| m.as_array())
        .ok_or_else(|| anyhow::anyhow!("snapshot JSON missing top-level `maps` array"))?;
    if maps.is_empty() {
        anyhow::bail!(
            "Op::WatchSnapshot fired but the captured report carries \
             0 maps — dump_state ran without producing any map data"
        );
    }

    let bss_map = maps.iter().find(|m| {
        m.get("name")
            .and_then(|n| n.as_str())
            .map(|n| n.ends_with(".bss") && !n.starts_with("probe_bp.") && !n.starts_with("fentry_p."))
            .unwrap_or(false)
    });
    if bss_map.is_none() {
        anyhow::bail!(
            "Op::WatchSnapshot captured {} maps but no scheduler \
             `.bss` — watchpoint-triggered capture must include \
             scheduler BPF state",
            maps.len()
        );
    }
    let bss = bss_map.unwrap();
    let members = bss
        .get("value")
        .and_then(|v| v.get("members"))
        .and_then(|m| m.as_array());
    if members.map(|m| m.is_empty()).unwrap_or(true) {
        anyhow::bail!(
            "Op::WatchSnapshot captured `.bss` but it has no \
             BTF-resolved members — watchpoint-triggered capture \
             must produce meaningful scheduler state"
        );
    }

    result.details.push(ktstr::assert::AssertDetail::new(
        ktstr::assert::DetailKind::Other,
        format!(
            "Op::WatchSnapshot('jiffies_64') fired and captured a \
             dump with {} maps, `.bss` has {} BTF-resolved field(s) \
             — DR1..=DR3 hardware-watchpoint pipeline OK",
            maps.len(),
            members.unwrap().len()
        ),
    ));
    Ok(result)
}

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_REAL_CAPTURE_OP_WATCH_SNAPSHOT: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "snapshot_real_capture_op_watch_snapshot",
        func: scenario_op_watch_snapshot_fires_on_kernel_write,
        scheduler: &KTSTR_SCHED_PAYLOAD,
        // Long enough for at least one timer tick to fire. On a
        // CONFIG_HZ=100 kernel that's 10ms minimum; holding for 2s
        // guarantees ~200 fires across all CPUs.
        duration: std::time::Duration::from_secs(2),
        watchdog_timeout: std::time::Duration::from_secs(15),
        workers_per_cgroup: 2,
        // Skip auto-repro: the scenario passes (no stall).
        auto_repro: false,
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };
