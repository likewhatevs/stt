//! End-to-end coverage of the stats-bridge round-trip path.
//!
//! Boots a real guest VM under `scx-ktstr` for 5 s with a single
//! periodic capture so the freeze coordinator's periodic-capture
//! loop fires exactly one boundary. The cgroup holds workers across
//! the entire duration so scx-ktstr's enqueue/dispatch callbacks
//! advance `nr_dispatched` (.bss + scx_stats `KtstrStats` envelope)
//! before the boundary fires.
//!
//! The `post_vm` callback runs on the host after `vm.run()` returns
//! and exercises the full stats-axis path:
//!
//! 1. Periodic boundary fires → freeze coordinator issues a
//!    scx_stats request over the port-2 dedicated channel.
//! 2. scx-ktstr's `Stats` derive answers with a `KtstrStats` JSON
//!    envelope carrying the BSS counter `nr_dispatched`.
//! 3. The relay routes the response back to the host bridge,
//!    coupled with the BPF capture into a single periodic sample.
//! 4. `SampleSeries::stats(...)` projects the JSON axis and the
//!    test asserts `nr_dispatched > 0` at the lone boundary.
//!
//! A non-zero observation proves every leg of the pipeline ran: the
//! relay landed a real envelope on the bridge, the JSON parsed into
//! `serde_json::Value`, the path projection resolved the field, and
//! the scheduler's dispatch path actually advanced.

use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::{SampleSeries, VmResult};
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps};
use ktstr::test_support::{Scheduler, SchedulerSpec};

const KTSTR_SCHED: Scheduler =
    Scheduler::new("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Drain the bridge's periodic captures and assert the
/// scheduler-stats path delivered a non-zero `nr_dispatched` count
/// at the lone interior boundary (num_snapshots = 1 → midpoint).
/// Proves the port-2 stats relay landed a real `scx_stats` JSON
/// envelope on the host bridge — the response carries the BSS
/// counter that the scheduler advertises via its `Stats` derive.
fn assert_stats_round_trip(result: &VmResult) -> Result<()> {
    anyhow::ensure!(
        result.periodic_target == 1,
        "periodic_target must mirror num_snapshots = 1, got {}",
        result.periodic_target,
    );
    anyhow::ensure!(
        result.periodic_fired >= 1,
        "the lone midpoint capture must have fired at least once \
         under a 5 s workload — periodic_fired = {} of {}",
        result.periodic_fired,
        result.periodic_target,
    );

    let drained = result.snapshot_bridge.drain_ordered_with_stats();
    anyhow::ensure!(
        !drained.is_empty(),
        "drain_ordered_with_stats returned an empty bundle despite \
         periodic_fired = {}",
        result.periodic_fired,
    );
    let series = SampleSeries::from_drained(drained).periodic_only();
    anyhow::ensure!(
        !series.is_empty(),
        "no periodic-tagged entries on the bridge after the run"
    );

    // .stats() projects the scheduler-stats JSON axis: every sample's
    // stats slot must be present (None would surface as
    // SnapshotError::MissingStats; absence here means the port-2
    // relay never delivered an envelope). The `series.is_empty()`
    // guard above ensures iter_full() yields at least one entry —
    // any Err slot bails immediately, so reaching the post-loop
    // assertion proves every slot was Ok and at least one sample
    // existed.
    let nr_dispatched = series.stats("nr_dispatched", |sv| sv.path("nr_dispatched").as_u64());
    let mut any_progress = false;
    for (tag, _elapsed_ms, slot) in nr_dispatched.iter_full() {
        match slot {
            Ok(v) => {
                if *v > 0 {
                    any_progress = true;
                }
            }
            Err(e) => anyhow::bail!(
                "stats projection for `nr_dispatched` failed at \
                 sample {tag}: {e}"
            ),
        }
    }
    anyhow::ensure!(
        any_progress,
        "scheduler reported nr_dispatched = 0 across every periodic \
         sample — the dispatch path never advanced under the 5 s \
         workload (was scx-ktstr loaded?)"
    );
    Ok(())
}

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    num_snapshots = 1,
    duration_s = 5,
    watchdog_timeout_s = 15,
    auto_repro = false,
    post_vm = assert_stats_round_trip,
)]
fn stats_bridge_round_trip(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}
