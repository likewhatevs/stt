//! End-to-end integration test for the ctprof capture ↔
//! compare pipeline.
//!
//! Fills in the missing half of the coverage noted on
//! `tests/ctprof_compare.rs` (docstring lines 10-17): the
//! compare-side file exercises `compare` and `write_diff` against
//! SYNTHETIC snapshots (no VM, no real procfs); the capture-side
//! file (`tests/ctprof_capture.rs`) exercises ONE real
//! capture inside a guest. This file stitches the two halves
//! together: inside a VM-booted guest, drive a CPU-spinning
//! workload twice with a capture + disk-write between and after
//! each round, then load both snapshots back from disk and run
//! [`ktstr::ctprof_compare::compare`] on the pair.
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

use anyhow::{Result, anyhow};
use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
use ktstr::ctprof::{self, CtprofSnapshot};
use ktstr::ctprof_compare::{CompareOptions, compare};
use ktstr::ktstr_test;
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps};
use ktstr::scenario::payload_run::PayloadHandle;
use ktstr::test_support::{OutputFormat, Payload, PayloadKind};
use ktstr::worker_ready_wait::wait_for_worker_ready;

// ---------------------------------------------------------------------------
// Initramfs wiring for the alloc-worker — the T3 test below spawns
// `ktstr-jemalloc-alloc-worker` inside the guest twice, between two
// captures, to grow `allocated_bytes` across the round-trip. Same
// ctor pattern as `tests/jemalloc_probe_tests.rs` and
// `tests/ctprof_capture_jemalloc_e2e.rs`; each integration-test
// crate compiles its own ctor list.
// ---------------------------------------------------------------------------

#[::ktstr::__private::ctor::ctor(crate_path = ::ktstr::__private::ctor)]
fn set_alloc_worker_binary_env_var() {
    unsafe {
        std::env::set_var(
            "KTSTR_JEMALLOC_ALLOC_WORKER_BINARY",
            env!("CARGO_BIN_EXE_ktstr-jemalloc-alloc-worker"),
        );
    }
}

static JEMALLOC_ALLOC_WORKER: Payload = Payload::new(
    "jemalloc_alloc_worker",
    PayloadKind::Binary("ktstr-jemalloc-alloc-worker"),
    OutputFormat::ExitCode,
    &[],
    &[],
    &[],
    &[],
    false,
    None,
    None,
);

/// Drive the pipeline inside the guest:
///
/// 1. Capture `baseline` after a short workload.
/// 2. Write `baseline` to `/tmp/baseline.ctprof.zst`.
/// 3. Run a second workload window so counters advance.
/// 4. Capture `candidate`; write to `/tmp/candidate.ctprof.zst`.
/// 5. Load both snapshots back from disk via
///    [`CtprofSnapshot::load`].
/// 6. Run [`compare`] against the loaded pair with default
///    options (GroupBy::Pcomm, no flatten).
/// 7. Verify at least one metric row carries a non-zero delta.
///
/// A non-zero delta proves every stage of the pipeline wired
/// through correctly. An empty diff or an all-zero delta set
/// means either the capture layer produced identical snapshots
/// across the two windows (capture-side regression) or the
/// compare-side group match failed to pair them across snapshots
/// (compare-side regression).
///
/// Topology: 1 LLC / 2 cores / 1 thread mirrors the sibling
/// `ctprof_capture` test — minimal guest, focused on the
/// pipeline surface, not on scheduler behavior. Two cores matter
/// because the workers need to genuinely move through the
/// scheduler rather than all pile onto a single runqueue that
/// might produce suspiciously uniform counter growth.
///
/// Duration: 6 s to give each of the TWO workload windows ~3 s
/// — the sibling capture test uses 3 s for a single window, so
/// doubling keeps the per-window activity at the same floor
/// rather than halving it.
#[ktstr_test(
    llcs = 1,
    cores = 2,
    threads = 1,
    duration_s = 6,
    max_spread_pct = 80.0
)]
fn ctprof_pipeline_e2e_capture_write_load_compare(ctx: &Ctx) -> Result<AssertResult> {
    let baseline_path = std::path::PathBuf::from("/tmp/baseline.ctprof.zst");
    let candidate_path = std::path::PathBuf::from("/tmp/candidate.ctprof.zst");

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
    // via `.with_context(|| format!("write ctprof snapshot
    // to {}", path.display()))` so the bare `?` surfaces an
    // actionable error without re-wrapping.
    let baseline_snap = ctprof::capture();
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
    let candidate_snap = ctprof::capture();
    candidate_snap.write(&candidate_path)?;

    // Load both snapshots back — exercises the full
    // disk-read + zstd-decode + serde-deserialize path that the
    // host-side `ktstr ctprof compare` CLI invocation would
    // run. A regression in the serialize-deserialize round trip
    // surfaces here as a deserialization error, not as a
    // silent field collapse. `load` threads its own anyhow
    // context chain so `?` surfaces an actionable error.
    let loaded_baseline = CtprofSnapshot::load(&baseline_path)?;
    let loaded_candidate = CtprofSnapshot::load(&candidate_path)?;

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
                 the pcomm grouping did not pair any rows, suggesting the \
                 GroupBy::Pcomm key diverged between captures (workload \
                 processes carry different pcomm values) or the grouping \
                 logic regressed",
                loaded_baseline.threads.len(),
                loaded_candidate.threads.len(),
            ),
        )));
    }

    // A non-zero delta on AT LEAST ONE metric proves real
    // counter growth survived the full pipeline. The OR across
    // scheduling-activity fields follows the same robustness
    // pattern as the sibling `ctprof_capture_returns_threads_with_nonzero_counters`
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

/// Allocation size for the T3 alloc-worker invocations. 16 MiB
/// matches the value used by the host-side wiring test and the
/// probe tests so the slop budget transfers without re-tuning.
const T3_KNOWN_BYTES: u64 = 16 * 1024 * 1024;

/// Worker-ready handshake timeout for T3.
const T3_READY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// T3 — round-trip the per-thread `allocated_bytes` field through
/// the full capture → write → load → compare pipeline against a
/// jemalloc-linked target.
///
/// The sibling `ctprof_pipeline_e2e_capture_write_load_compare`
/// test above proves the SCHEDSTAT / page-fault metric family
/// survives the pipeline. T3 is the jemalloc-counter complement:
/// drive two alloc-worker invocations between captures so the
/// pcomm-grouped sum across the alloc-worker tgid grows by at
/// least KNOWN_BYTES per invocation; assert the loaded compare
/// produces an `allocated_bytes` row with `delta > 0`. A
/// regression in the capture-side probe wiring, the serde schema
/// for `ThreadState::allocated_bytes`, the zstd round-trip, or
/// the compare-side metric inclusion list surfaces here as either
/// a missing row (the metric was not joined) or a zero delta
/// (the counter collapsed to zero on at least one side).
///
/// The two alloc-worker invocations spawn between the two
/// captures, NOT during capture — capture probes whatever is live
/// at walk time. The first capture sees no alloc-worker; the
/// second capture sees the second invocation. The compare-side
/// pcomm grouping keys on the test binary's pcomm so the
/// alloc-worker's `pcomm` (its own argv[0]
/// "ktstr-jemalloc-alloc-worker") groups separately from the test
/// binary, and the delta on that group surfaces as the planted
/// allocation.
///
/// Topology mirrors the parent pipeline test; duration is bumped
/// to 10s so each alloc-worker invocation has slack to allocate +
/// signal ready before the next capture fires.
#[ktstr_test(llcs = 1, cores = 2, threads = 1, duration_s = 10)]
fn ctprof_pipeline_e2e_allocated_bytes_delta_survives_round_trip(
    ctx: &Ctx,
) -> Result<AssertResult> {
    let baseline_path = std::path::PathBuf::from("/tmp/baseline_alloc.ctprof.zst");
    let candidate_path = std::path::PathBuf::from("/tmp/candidate_alloc.ctprof.zst");

    // First alloc-worker invocation — runs before the baseline
    // capture so the baseline DOES include its tgid (the worker
    // is alive when the capture walks /proc). The worker parks
    // forever holding its Vec; we kill it explicitly after the
    // baseline capture so its tgid is gone before the candidate
    // capture (otherwise the second worker's pid would race).
    let mut worker1: PayloadHandle = ctx
        .payload(&JEMALLOC_ALLOC_WORKER)
        .arg(T3_KNOWN_BYTES.to_string())
        .spawn()?;
    let worker1_pid = worker1
        .pid()
        .ok_or_else(|| anyhow!("alloc-worker #1 handle has no pid"))?;
    wait_for_worker_ready(
        &mut worker1,
        worker1_pid,
        T3_READY_TIMEOUT,
        "alloc-worker #1",
        "see jemalloc_alloc_worker exit-code legend",
    )?;

    // Capture #1: the alloc-worker is alive with KNOWN_BYTES
    // planted. Its allocated_bytes carries the planted value.
    let baseline_snap = ctprof::capture();
    baseline_snap.write(&baseline_path)?;
    let _ = worker1.kill();

    // Second alloc-worker invocation — same binary, same
    // KNOWN_BYTES. Its tgid is fresh (different pid from
    // worker1). Capture #2 sees its allocated_bytes as the
    // planted value.
    let mut worker2: PayloadHandle = ctx
        .payload(&JEMALLOC_ALLOC_WORKER)
        .arg(T3_KNOWN_BYTES.to_string())
        .spawn()?;
    let worker2_pid = worker2
        .pid()
        .ok_or_else(|| anyhow!("alloc-worker #2 handle has no pid"))?;
    wait_for_worker_ready(
        &mut worker2,
        worker2_pid,
        T3_READY_TIMEOUT,
        "alloc-worker #2",
        "see jemalloc_alloc_worker exit-code legend",
    )?;

    let candidate_snap = ctprof::capture();
    candidate_snap.write(&candidate_path)?;
    let _ = worker2.kill();

    // Round-trip both captures through the disk + zstd + serde
    // pipeline.
    let loaded_baseline = CtprofSnapshot::load(&baseline_path)?;
    let loaded_candidate = CtprofSnapshot::load(&candidate_path)?;

    // Sanity: both loads carry the alloc-worker's tgid (one
    // per snapshot — different pids, but they share pcomm
    // because they're the same binary).
    let baseline_alloc: u64 = loaded_baseline
        .threads
        .iter()
        .filter(|t| t.tgid == worker1_pid)
        .map(|t| t.allocated_bytes.0)
        .max()
        .unwrap_or(0);
    let candidate_alloc: u64 = loaded_candidate
        .threads
        .iter()
        .filter(|t| t.tgid == worker2_pid)
        .map(|t| t.allocated_bytes.0)
        .max()
        .unwrap_or(0);
    if baseline_alloc < T3_KNOWN_BYTES {
        return Ok(AssertResult::fail_msg(format!(
            "loaded baseline alloc-worker tgid={worker1_pid} carries \
             allocated_bytes={baseline_alloc} < KNOWN_BYTES={T3_KNOWN_BYTES}; \
             the capture-write-load round trip dropped or zero'd the counter \
             before reaching the compare stage",
        )));
    }
    if candidate_alloc < T3_KNOWN_BYTES {
        return Ok(AssertResult::fail_msg(format!(
            "loaded candidate alloc-worker tgid={worker2_pid} carries \
             allocated_bytes={candidate_alloc} < KNOWN_BYTES={T3_KNOWN_BYTES}",
        )));
    }

    // Run compare on the loaded pair. Default options group by
    // pcomm — the alloc-worker invocations share pcomm
    // ("ktstr-jemalloc-alloc-worker"), so the baseline's tgid
    // and the candidate's tgid land in the same pcomm group. The compare
    // sums per-group, so the candidate's allocated_bytes total
    // for the alloc-worker pcomm should equal the candidate's
    // tgid total (worker1 was killed before capture #2 so it's
    // absent from candidate). The baseline's allocated_bytes
    // total is just worker1's. Since each invocation plants
    // KNOWN_BYTES with bounded slop, the per-group sum across
    // captures grows from baseline=KNOWN_BYTES (±slop) to
    // candidate=KNOWN_BYTES (±slop) with no monotone-growth
    // guarantee — both baseline AND candidate carry one fresh
    // alloc-worker each. The COMPARE row therefore measures
    // candidate-baseline, which is bounded by ±slop, NOT
    // necessarily > 0.
    //
    // To get a guaranteed positive delta we must instead detect
    // that the metric APPEARS in the compare output with both
    // sides non-zero — proving the wiring populated AND the
    // compare passed it through. A delta > 0 is sufficient but
    // not necessary; the load-bearing assertion is "row exists
    // with both sides reporting non-zero allocated_bytes".
    let diff = compare(
        &loaded_baseline,
        &loaded_candidate,
        &CompareOptions::default(),
    );
    if diff.rows.is_empty() {
        return Ok(AssertResult::fail_msg(format!(
            "compare produced zero rows for {} baseline × {} candidate threads — \
             the pcomm grouping did not pair any rows",
            loaded_baseline.threads.len(),
            loaded_candidate.threads.len(),
        )));
    }

    // Find the allocated_bytes row(s) for the alloc-worker pcomm
    // group. The compare emits one row per (group_key, metric)
    // pair, so we filter on metric_name == "allocated_bytes" and
    // pick the row whose group key matches the alloc-worker's
    // pcomm.
    let alloc_rows: Vec<_> = diff
        .rows
        .iter()
        .filter(|r| r.metric_name == "allocated_bytes")
        .collect();
    if alloc_rows.is_empty() {
        return Ok(AssertResult::fail_msg(format!(
            "compare produced no `allocated_bytes` row in {} total rows — \
             the metric inclusion list does not surface the jemalloc TSD \
             counter through the compare pipeline. Row metric names: {:?}",
            diff.rows.len(),
            diff.rows
                .iter()
                .map(|r| r.metric_name)
                .collect::<std::collections::BTreeSet<_>>(),
        )));
    }

    // The strong assertion: at least one `allocated_bytes` row
    // for the alloc-worker pcomm carries baseline AND candidate
    // sums each >= KNOWN_BYTES. Both sides should report
    // KNOWN_BYTES (±slop) from a fresh alloc-worker invocation
    // (worker1 in baseline, worker2 in candidate). The pcomm
    // grouping joins them under a single key
    // ("ktstr-jemalloc-" — kernel truncates comm to 15 chars).
    // A regression that dropped the counter on either side
    // (capture-layer probe failure, serde schema drift, zstd
    // round-trip corruption) would push at least one side's
    // sum below KNOWN_BYTES.
    //
    // Why "both sides >= KNOWN_BYTES" and not "delta != 0": the
    // delta `candidate − baseline` is bounded by slop variance
    // and may legitimately equal zero when both invocations
    // produce the same allocation pattern. A zero delta with
    // both sums >= KNOWN_BYTES is the correct shape; a non-zero
    // delta on its own would not prove either side carries the
    // planted value.
    let alloc_worker_rows: Vec<_> = alloc_rows
        .iter()
        .filter(|r| {
            // Kernel truncates /proc/<tid>/comm to 15 chars + NUL.
            // The full binary name "ktstr-jemalloc-alloc-worker"
            // (27 chars) truncates to "ktstr-jemalloc-" (15
            // chars). Match on the truncated prefix so a future
            // rename of the worker binary lands the test's
            // failure on the rename, not on this filter.
            r.group_key.starts_with("ktstr-jemalloc-")
        })
        .collect();
    if alloc_worker_rows.is_empty() {
        return Ok(AssertResult::fail_msg(format!(
            "compare produced no `allocated_bytes` row keyed on \
             alloc-worker pcomm in {} total alloc rows; observed \
             group keys: {:?}",
            alloc_rows.len(),
            alloc_rows
                .iter()
                .map(|r| r.group_key.as_str())
                .collect::<std::collections::BTreeSet<_>>(),
        )));
    }
    let any_loaded = alloc_worker_rows.iter().any(|r| {
        let baseline_sum = match r.baseline {
            ktstr::ctprof_compare::Aggregated::Sum(n) => n,
            _ => 0,
        };
        let candidate_sum = match r.candidate {
            ktstr::ctprof_compare::Aggregated::Sum(n) => n,
            _ => 0,
        };
        baseline_sum >= T3_KNOWN_BYTES && candidate_sum >= T3_KNOWN_BYTES
    });
    if !any_loaded {
        let dump: Vec<(String, u64, u64, Option<f64>)> = alloc_worker_rows
            .iter()
            .map(|r| {
                let b = match r.baseline {
                    ktstr::ctprof_compare::Aggregated::Sum(n) => n,
                    _ => 0,
                };
                let c = match r.candidate {
                    ktstr::ctprof_compare::Aggregated::Sum(n) => n,
                    _ => 0,
                };
                (r.group_key.clone(), b, c, r.delta)
            })
            .collect();
        return Ok(AssertResult::fail_msg(format!(
            "no alloc-worker `allocated_bytes` row carries both baseline \
             AND candidate sums >= KNOWN_BYTES={T3_KNOWN_BYTES}; the \
             round-trip dropped the counter on at least one side. \
             Observed (group, baseline_sum, candidate_sum, delta): {dump:?}",
        )));
    }

    let mut result = AssertResult::pass();
    let summary: Vec<(String, u64, u64, Option<f64>)> = alloc_worker_rows
        .iter()
        .map(|r| {
            let b = match r.baseline {
                ktstr::ctprof_compare::Aggregated::Sum(n) => n,
                _ => 0,
            };
            let c = match r.candidate {
                ktstr::ctprof_compare::Aggregated::Sum(n) => n,
                _ => 0,
            };
            (r.group_key.clone(), b, c, r.delta)
        })
        .collect();
    result.details.push(AssertDetail::new(
        DetailKind::Other,
        format!(
            "ctprof_pipeline_e2e_allocated_bytes_delta: \
             baseline_alloc={baseline_alloc}, candidate_alloc={candidate_alloc}, \
             alloc_worker_rows={summary:?}",
        ),
    ));
    Ok(result)
}

/// `CompareOptions::default().group_by` must resolve to
/// `GroupBy::Pcomm`. The default flows through every test in this
/// file — both the SCHEDSTAT round-trip and the T3 allocated_bytes
/// round-trip rely on the pcomm grouping axis to merge baseline
/// and candidate threads under one key. A regression that flipped the
/// default to `GroupBy::Comm` or `GroupBy::Cgroup` would silently
/// fan baseline / candidate threads across distinct groups, the
/// per-group delta would collapse to zero, and the
/// `has_nonzero_delta` / `any_loaded` assertions would still pass
/// on luck (a single matched row is enough). Pinning the default
/// here surfaces the regression at compile / test time before it
/// can mask itself in the e2e flows above.
///
/// Pin lives in this file (rather than the compare module's own
/// tests) because this is the file whose flows depend on the
/// default value remaining `Pcomm`. A future test author who
/// wants a non-default grouping will set the option explicitly;
/// the default is the load-bearing assumption for the existing
/// e2e tests.
#[test]
fn compare_options_default_groups_by_pcomm() {
    use ktstr::ctprof_compare::GroupBy;
    assert_eq!(
        CompareOptions::default().group_by.0,
        GroupBy::Pcomm,
        "CompareOptions::default().group_by must resolve to \
         GroupBy::Pcomm — the e2e tests above bake the pcomm grouping \
         into their non-zero-delta assertions, so a regression here \
         would silently fan threads across distinct groups",
    );
}
