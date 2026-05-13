//! End-to-end test for `KtstrTestEntry::num_snapshots` periodic
//! capture.
//!
//! Boots a real guest VM with `num_snapshots = 3` and a 10 s
//! workload duration (interior boundaries land at scenario_start
//! plus {3 s, 5 s, 7 s}). The guest just holds the cgroup for the
//! full duration; the freeze coordinator's periodic-capture loop
//! fires the captures from the host side and stores reports on
//! the host-side `SnapshotBridge`.
//!
//! The `post_vm` callback runs on the HOST after `vm.run()`
//! returns and asserts:
//!   * `result.periodic_target == 3` (the configured count)
//!   * `result.periodic_fired >= 1` (best-effort — CI cold-cache
//!     latency or kill-flag races may cut the sequence short, but
//!     a healthy guest should fire at least the first boundary)
//!   * `drain_ordered()` returns reports tagged `periodic_NNN` in
//!     ascending NNN order, with no `periodic_` tags missing
//!     between two stored entries.
//!   * Each successful (non-placeholder) report has at least one
//!     `.maps` entry — the freeze-and-capture path actually walked
//!     scheduler-state BPF maps, not just stored a placeholder.

use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::VmResult;
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps};
use ktstr::test_support::{Scheduler, SchedulerSpec};

const KTSTR_SCHED: Scheduler =
    Scheduler::new("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Host-side check: every periodic capture stored on the bridge
/// has the expected tag shape, ordering, and non-empty content
/// (when not a placeholder).
fn assert_periodic_captures(result: &VmResult) -> Result<()> {
    anyhow::ensure!(
        result.periodic_target == 3,
        "periodic_target must mirror the configured num_snapshots = 3, got {}",
        result.periodic_target,
    );
    anyhow::ensure!(
        result.periodic_fired >= 1,
        "periodic_fired must be at least 1 — a healthy guest should \
         cross the first boundary at scenario_start + 3 s during \
         the 10 s workload window. Got {} of {}.",
        result.periodic_fired,
        result.periodic_target,
    );
    anyhow::ensure!(
        result.periodic_fired <= result.periodic_target,
        "periodic_fired ({}) must not exceed periodic_target ({})",
        result.periodic_fired,
        result.periodic_target,
    );

    // Drain in insertion order so we can assert the tag sequence
    // is contiguous from periodic_000.
    let captured = result.snapshot_bridge.drain_ordered();
    let periodic_entries: Vec<_> = captured
        .iter()
        .filter(|(tag, _)| tag.starts_with("periodic_"))
        .collect();
    anyhow::ensure!(
        !periodic_entries.is_empty(),
        "bridge has no `periodic_*` entries despite periodic_fired = {}",
        result.periodic_fired,
    );
    anyhow::ensure!(
        periodic_entries.len() == result.periodic_fired as usize,
        "bridge has {} periodic_* entries but periodic_fired = {} — \
         counts must match (each fire stores exactly once)",
        periodic_entries.len(),
        result.periodic_fired,
    );

    // Tags must be `periodic_000`, `periodic_001`, ... contiguous
    // from index 0. Any gap indicates a fire path that advanced
    // `next_periodic_idx` without storing onto the bridge — a bug.
    for (i, (tag, _)) in periodic_entries.iter().enumerate() {
        let expected = format!("periodic_{:03}", i);
        anyhow::ensure!(
            tag.as_str() == expected.as_str(),
            "periodic entry at position {i} has tag {tag:?}; expected \
             {expected:?} (zero-based, contiguous, :03 padded)"
        );
    }

    // At least one entry must be a real capture (non-placeholder).
    // Placeholders set every `*_unavailable` field; real captures
    // populate the maps Vec. A run where every boundary timed out
    // should be flagged so an operator notices the rendezvous
    // problem instead of treating the all-placeholder bridge as a
    // pass.
    let real_captures = periodic_entries
        .iter()
        .filter(|(_, report)| !report.maps.is_empty())
        .count();
    anyhow::ensure!(
        real_captures >= 1,
        "every periodic entry on the bridge is a placeholder \
         (empty .maps) — the freeze coordinator never produced a \
         real capture. Most commonly a parked-vCPU rendezvous \
         timeout repeated past the 2-consecutive abandon \
         threshold; check the trace for \
         'periodic capture abandoned'."
    );

    Ok(())
}

/// 10 s workload with periodic captures at scenario_start + 3 s,
/// 5 s, 7 s. The cgroup holds for the full duration so the
/// workload has live tasks across every boundary; without that
/// the per-CPU runnable_at scanner would log "no aged tasks" and
/// the captures would still happen (periodic capture is not gated
/// on the scanner) but the reports would be sparse.
#[ktstr_test(
    scheduler = KTSTR_SCHED,
    duration_s = 10,
    watchdog_timeout_s = 15,
    num_snapshots = 3,
    auto_repro = false,
    post_vm = assert_periodic_captures,
)]
fn periodic_capture_three_boundaries(ctx: &ktstr::scenario::Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    let mut result = execute_steps(ctx, steps)?;
    result.details.push(ktstr::assert::AssertDetail::new(
        ktstr::assert::DetailKind::Other,
        "10s workload with num_snapshots=3 finished".to_string(),
    ));
    Ok(result)
}
