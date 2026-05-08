//! End-to-end exercise of the [`SampleSeries`] temporal assertion
//! patterns against scx-ktstr's BPF .bss counters and scx_stats
//! envelope.
//!
//! Boots a real guest VM with `num_snapshots = 3` and a 10 s
//! workload duration so the freeze coordinator's periodic-capture
//! loop fires three samples (interior boundaries at scenario_start
//! plus {3 s, 5 s, 7 s}). The cgroup holds workers across the full
//! window so scx-ktstr's `nr_dispatched` (.bss) and the parallel
//! `nr_dispatched` field on the scx_stats `KtstrStats` envelope
//! both advance through every boundary.
//!
//! The `post_vm` callback runs on the host after `vm.run()` returns
//! and exercises two temporal patterns over the resulting
//! [`SampleSeries`]:
//!
//! * `series.bpf("nr_dispatched", ...).nondecreasing(&mut verdict)` —
//!   pins that the cumulative dispatch counter on the BPF axis only
//!   ever advances. A regression at any sample fires a
//!   [`DetailKind::Temporal`] detail naming the offending pair.
//! * `series.stats("nr_dispatched", ...).each(&mut verdict).at_most(N)` —
//!   pins that the stats-axis dispatch counter stays under a generous
//!   ceiling. The ceiling is far above what a 10 s ktstr-fixture run
//!   can plausibly accumulate, so the bound is satisfied; the patternly
//!   exercised is the per-sample comparator path.
//!
//! Together they cover both projection axes (`bpf` and `stats`),
//! both temporal shapes (cross-sample monotonicity and per-sample
//! scalar bound), and the conversion of a verdict failure into an
//! `anyhow::Error` so the host-side callback fails the test on
//! either kind of regression.

use anyhow::Result;
use ktstr::assert::{AssertResult, Verdict};
use ktstr::ktstr_test;
use ktstr::prelude::{SampleSeries, VmResult};
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps};
use ktstr::test_support::{Payload, Scheduler, SchedulerSpec};

const KTSTR_SCHED: Scheduler =
    Scheduler::new("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));
const KTSTR_SCHED_PAYLOAD: Payload = Payload::from_scheduler(&KTSTR_SCHED);

/// Generous per-sample ceiling for `nr_dispatched`. A 10 s
/// scx-ktstr run on a small guest tops out far below this — the
/// number is chosen to exercise `each().at_most()` without
/// flapping on host-load variation. If a real run ever climbs
/// past 10^12 dispatches the bound itself would need rethinking,
/// not the test.
const DISPATCHED_CEILING: u64 = 1_000_000_000_000;

/// Host-side temporal-assertion checks over the periodic samples
/// stored on the bridge.
fn assert_temporal_patterns(result: &VmResult) -> Result<()> {
    // Drain in insertion order with the parallel scx_stats / elapsed
    // metadata so the resulting series carries both projection axes.
    // `periodic_only` strips any non-periodic capture entries the
    // bridge happened to also store under the same drain
    // (e.g. an Op::Snapshot fire from inside the scenario body) so
    // the temporal patterns walk a clean, contiguous timeline.
    let series =
        SampleSeries::from_drained(result.snapshot_bridge.drain_ordered_with_stats())
            .periodic_only();
    anyhow::ensure!(
        !series.is_empty(),
        "post_vm: no periodic samples on the bridge — the freeze \
         coordinator never fired (periodic_target={}, \
         periodic_fired={})",
        result.periodic_target,
        result.periodic_fired,
    );

    let mut verdict = Verdict::new();

    // BPF axis: cumulative dispatch counter must only advance. The
    // `__sync_fetch_and_add` increment in `ktstr_dispatch` (scx-ktstr
    // main.bpf.c) means the host-side .bss read at every freeze
    // boundary observes a value at or above the prior sample's
    // value. A regression here would indicate either a counter
    // wrap, a dropped capture re-using an older value, or a
    // monotonicity bug in the snapshot pipeline.
    series
        .bpf("nr_dispatched", |snap| {
            snap.var("nr_dispatched").as_u64()
        })
        .nondecreasing(&mut verdict);

    // Stats axis: per-sample ceiling on the same counter exposed
    // through the scx_stats `KtstrStats` envelope. `.each()` opens
    // a per-sample comparator chain; `.at_most(...)` records a
    // failure for any sample whose value exceeds the ceiling. A
    // generous ceiling keeps the assertion stable across host-load
    // variation while still exercising the comparator path on every
    // periodic boundary.
    series
        .stats("nr_dispatched", |s| s.path("nr_dispatched").as_u64())
        .each(&mut verdict)
        .at_most(DISPATCHED_CEILING);

    let r = verdict.into_result();
    if !r.passed {
        let detail_lines: Vec<String> = r
            .details
            .iter()
            .map(|d| format!("  [{:?}] {}", d.kind, d.message))
            .collect();
        anyhow::bail!(
            "temporal assertions failed across {} sample(s):\n{}",
            series.len(),
            detail_lines.join("\n"),
        );
    }
    Ok(())
}

/// 10 s workload with periodic captures at scenario_start + 3 s,
/// 5 s, 7 s. The cgroup holds workers across the entire duration
/// so scx-ktstr's enqueue/dispatch callbacks fire continuously and
/// both the BPF .bss `nr_dispatched` field and the scx_stats
/// `nr_dispatched` field advance at every boundary.
#[ktstr_test(
    scheduler = KTSTR_SCHED_PAYLOAD,
    duration_s = 10,
    watchdog_timeout_s = 15,
    workers_per_cgroup = 2,
    num_snapshots = 3,
    auto_repro = false,
    post_vm = assert_temporal_patterns,
)]
fn temporal_assertions_over_periodic_samples(
    ctx: &ktstr::scenario::Ctx,
) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}
