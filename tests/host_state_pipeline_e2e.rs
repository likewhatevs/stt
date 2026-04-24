//! End-to-end integration test for the host-state capture ↔
//! compare pipeline.
//!
//! Fills in the missing half of the coverage noted on
//! `tests/host_state_compare.rs` (docstring lines 10-17): the
//! compare-side file exercises `compare` and `write_diff` against
//! SYNTHETIC snapshots (no VM, no real procfs); the capture-side
//! file (`tests/host_state_capture.rs`) exercises ONE real
//! capture inside a guest. This file stitches the two halves
//! together: inside a VM-booted guest, drive a CPU-spinning
//! workload twice with a capture + disk-write between and after
//! each round, then load both snapshots back from disk and run
//! [`ktstr::host_state_compare::compare`] on the pair.
//!
//! The assertion verifies that real schedstat + page-fault deltas
//! survive the full pipeline — capture → zstd-serialize → write →
//! read → deserialize → compare — rather than collapsing to zero
//! at any stage. A regression anywhere on that path (capture layer
//! parse regression, zstd framing corruption, serde schema drift,
//! compare-side group-match miss) surfaces as either an empty
//! diff or a delta of zero where real activity produced real
//! counter growth.
//!
//! Why VM-backed, not host-side: the same capture / load / compare
//! code runs host-side, but a host-side test can race against
//! concurrent workloads on the CI worker and produce flakes.
//! Inside the guest the workload is the only process generating
//! meaningful activity, so the "baseline vs candidate ought to
//! differ" assertion is deterministic.

use anyhow::Result;
use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
use ktstr::host_state::{self, HostStateSnapshot};
use ktstr::host_state_compare::{CompareOptions, compare};
use ktstr::ktstr_test;
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps};

/// Drive the pipeline inside the guest:
///
/// 1. Capture `baseline` after a short workload.
/// 2. Write `baseline` to `/tmp/baseline.hst.zst`.
/// 3. Run a second workload window so counters advance.
/// 4. Capture `candidate`; write to `/tmp/candidate.hst.zst`.
/// 5. Load both snapshots back from disk via
///    [`HostStateSnapshot::load`].
/// 6. Run [`compare`] against the loaded pair with default
///    options (GroupBy::Pcomm, no flatten).
/// 7. Verify at least one metric row carries a non-zero delta.
///
/// A non-zero delta proves every stage of the pipeline wired
/// through correctly. An empty diff or an all-zero delta set
/// means either the capture layer produced identical snapshots
/// across the two windows (capture-side regression) or the
/// compare-side group match failed to join them (compare-side
/// regression).
///
/// Topology: 1 LLC / 2 cores / 1 thread mirrors the sibling
/// `host_state_capture` test — minimal guest, focused on the
/// pipeline surface, not on scheduler behavior. Two cores matter
/// because the workers need to genuinely move through the
/// scheduler rather than all pile onto a single runqueue that
/// might produce suspiciously uniform counter growth.
///
/// Duration: 6 s to give each of the TWO workload windows ~3 s
/// — the sibling capture test uses 3 s for a single window, so
/// doubling keeps the per-window activity at the same floor
/// rather than halving it.
#[ktstr_test(llcs = 1, cores = 2, threads = 1, duration_s = 6)]
fn host_state_pipeline_e2e_capture_write_load_compare(ctx: &Ctx) -> Result<AssertResult> {
    let baseline_path = std::path::PathBuf::from("/tmp/baseline.hst.zst");
    let candidate_path = std::path::PathBuf::from("/tmp/candidate.hst.zst");

    // First workload window: default HoldSpec::FULL occupies the
    // full step duration. `execute_steps` with one Step carrying
    // `HoldSpec::Frac(0.5)` uses half of `ctx.duration` so the
    // remaining half is available for the second window below.
    let baseline_steps = vec![Step {
        setup: vec![CgroupDef::named("cg_baseline").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::Frac(0.5),
    }];
    let baseline_workload_result = execute_steps(ctx, baseline_steps)?;

    // Capture snapshot #1 — activity accumulated during the
    // first window. `write` already threads full context chains
    // via `.with_context(|| format!("write host-state snapshot
    // to {}", path.display()))` so the bare `?` surfaces an
    // actionable error without re-wrapping.
    let baseline_snap = host_state::capture();
    baseline_snap.write(&baseline_path)?;

    // Second workload window under its own cgroup so the
    // compare-side group-match sees orthogonal pcomm / comm
    // populations — the baseline had `cg_baseline` workers, the
    // candidate has `cg_candidate` workers, and both share the
    // same parent `pcomm` (test-binary name) so the default
    // GroupBy::Pcomm collapses them together and the deltas
    // compute on merged totals.
    let candidate_steps = vec![Step {
        setup: vec![CgroupDef::named("cg_candidate").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::Frac(0.5),
    }];
    let candidate_workload_result = execute_steps(ctx, candidate_steps)?;

    // Capture snapshot #2. Counters for the threads visible at
    // capture time have now accumulated activity from BOTH
    // workload windows; the candidate totals should therefore
    // exceed the baseline totals for the pcomm group.
    let candidate_snap = host_state::capture();
    candidate_snap.write(&candidate_path)?;

    // Load both snapshots back — exercises the full
    // disk-read + zstd-decode + serde-deserialize path that the
    // host-side `ktstr host-state compare` CLI invocation would
    // run. A regression in the serialize-deserialize round trip
    // surfaces here as a deserialization error, not as a
    // silent field collapse. `load` threads its own anyhow
    // context chain so `?` surfaces an actionable error.
    let loaded_baseline = HostStateSnapshot::load(&baseline_path)?;
    let loaded_candidate = HostStateSnapshot::load(&candidate_path)?;

    // First-level sanity: the round-trip preserved thread counts
    // on both sides. An empty snapshot after load would mean the
    // zstd or serde layer silently produced `threads: vec![]` —
    // hiding any real deltas the compare would have surfaced.
    if loaded_baseline.threads.is_empty() {
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            "loaded baseline snapshot has zero threads — \
             disk round-trip (write + zstd-decode + \
             deserialize) produced an empty snapshot",
        )));
    }
    if loaded_candidate.threads.is_empty() {
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            "loaded candidate snapshot has zero threads — \
             disk round-trip (write + zstd-decode + \
             deserialize) produced an empty snapshot",
        )));
    }

    // Run compare with default options (GroupBy::Pcomm, no
    // flatten). Real captures inside a guest share the parent
    // `pcomm` across the workload's threads (they all fork from
    // the test binary), so the default grouping joins baseline
    // and candidate under one key.
    let diff = compare(
        &loaded_baseline,
        &loaded_candidate,
        &CompareOptions::default(),
    );

    if diff.rows.is_empty() {
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "compare produced zero rows for {} baseline × {} candidate threads — \
                 the pcomm-grouped join did not match any pair, suggesting the \
                 GroupBy::Pcomm key diverged between captures (workload processes \
                 carry different pcomm values) or the join logic regressed",
                loaded_baseline.threads.len(),
                loaded_candidate.threads.len(),
            ),
        )));
    }

    // A non-zero delta on AT LEAST ONE metric proves real
    // counter growth survived the full pipeline. The OR across
    // scheduling-activity fields follows the same robustness
    // pattern as the sibling `host_state_capture_returns_threads_with_nonzero_counters`
    // test: `CONFIG_SCHEDSTATS` may be runtime-off (every
    // schedstat row is zero), so page-fault counters are the
    // fallback signal that does not depend on sysctl state.
    let tracked_metrics = [
        "run_time_ns",
        "voluntary_csw",
        "nonvoluntary_csw",
        "nr_wakeups",
        "minflt",
    ];
    let has_nonzero_delta = diff
        .rows
        .iter()
        .any(|r| tracked_metrics.contains(&r.metric_name) && r.delta.is_some_and(|d| d != 0.0));

    if !has_nonzero_delta {
        // Dump the observed deltas so a reviewer chasing a
        // regression can tell "compare ran but all metrics were
        // zero" from "compare produced no rows at all" (the
        // preceding `rows.is_empty()` gate) without re-running.
        let delta_summary: String = diff
            .rows
            .iter()
            .filter(|r| tracked_metrics.contains(&r.metric_name))
            .map(|r| format!("{}={:?}", r.metric_name, r.delta))
            .collect::<Vec<_>>()
            .join(", ");
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "no tracked metric produced a non-zero delta across the \
                 capture → write → load → compare pipeline; observed: \
                 [{delta_summary}]. Either both capture rounds collapsed \
                 to identical counter snapshots (capture-layer regression) \
                 or the delta computation silently zeroed them."
            ),
        )));
    }

    // Pipeline-level checks all passed. Return the scheduling
    // scenario's own verdict so any scheduler-side failure
    // surfaces alongside — the two signals are orthogonal.
    // Merge the two workload verdicts: fail if either failed.
    if !baseline_workload_result.passed {
        return Ok(baseline_workload_result);
    }
    Ok(candidate_workload_result)
}
