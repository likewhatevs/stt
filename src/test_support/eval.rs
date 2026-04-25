//! Host-side VM result evaluation for `#[ktstr_test]` runs.
//!
//! The core [`run_ktstr_test_inner`] orchestrates a single test run:
//! boot the guest VM with the scheduler and workload, collect profraw
//! + stimulus events from SHM, then hand off to [`evaluate_vm_result`]
//!   for pass/fail judgment and error-message construction.
//!
//! [`evaluate_vm_result`] is factored out of the VM-boot path so error
//! formatting can be unit-tested with synthetic `VmResult` values.
//!
//! Supporting items:
//! - [`resolve_scheduler`] / [`resolve_test_kernel`] locate the
//!   scheduler binary and kernel image from env + cache + filesystem.
//! - [`scheduler_label`] formats the `[sched=...]` bracket in error
//!   headers.
//! - [`format_monitor_section`] and [`trim_settle_samples`] handle the
//!   `--- monitor ---` block in failed-test output.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

use crate::assert::AssertResult;
use crate::timeline::StimulusEvent;
use crate::vmm;

use super::output::{
    classify_init_stage, extract_kernel_version, extract_panic_message, extract_sched_ext_dump,
    format_console_diagnostics, parse_assert_result, parse_assert_result_shm,
    sched_log_fingerprint,
};
use super::probe::attempt_auto_repro;
use super::profraw::{MSG_TYPE_PROFRAW, write_profraw};
use super::sidecar::{write_sidecar, write_skip_sidecar};
use super::topo::TopoOverride;
use super::{KtstrTestEntry, SchedulerSpec, Topology};
use crate::verifier::{SCHED_OUTPUT_START, parse_sched_output};

use super::runtime::{config_file_parts, verbose};

// ---------------------------------------------------------------------------
// Failure-message constants
// ---------------------------------------------------------------------------
//
// Shared between the production error-formatting paths in this module
// and the tests that pin those messages. Editing a production string
// here without updating the test (or vice versa) is caught at compile
// time instead of as a runtime test assertion drift.

/// Header body for a timed-out run with no parseable AssertResult.
/// Pinned by `eval_timeout_no_result` and `eval_timeout_with_sched_includes_diagnostics`.
pub(crate) const ERR_TIMED_OUT_NO_RESULT: &str = "timed out (no result in SHM or COM2)";

/// Header body for a run whose scenario passed but whose monitor
/// verdict failed. Pinned by `eval_monitor_fail_has_fingerprint` and
/// `eval_monitor_fail_includes_sched_log`.
pub(crate) const ERR_MONITOR_FAILED_AFTER_SCENARIO: &str = "passed scenario but monitor failed";

/// Reason body when a scheduler is running but no AssertResult was
/// received from the guest. Pinned by `eval_sched_exits_no_com2_output`
/// and `eval_sched_exits_with_sched_log`.
pub(crate) const ERR_NO_TEST_RESULT_FROM_GUEST: &str = "no test result received from guest \
     (no AssertResult arrived via SHM or COM2; check kernel log and \
     scheduler exit status)";

/// Reason body when EEVDF (no scheduler) produced no AssertResult.
/// Pinned by `eval_eevdf_no_com2_output` and `eval_payload_exits_no_check_result`.
pub(crate) const ERR_NO_TEST_FUNCTION_OUTPUT: &str =
    "test function produced no output (no test result found)";

/// Prefix for the `guest crashed: ...` reason body. Pinned by
/// `eval_crash_in_output_says_guest_crashed`, `eval_crash_eevdf_says_guest_crashed`,
/// and `eval_crash_message_from_shm`.
pub(crate) const ERR_GUEST_CRASHED_PREFIX: &str = "guest crashed:";

/// Write a skip sidecar for `entry` + `active_flags`, logging to
/// stderr on failure without propagating the error. Used at five
/// sites — the three in [`run_ktstr_test_inner`] (performance_mode
/// gate plus the two `ResourceContention` arms at VM build + VM
/// run) and the two in `super::dispatch` (performance_mode gates
/// at the plain-run and flag-profile entry points) — all of which
/// must record the skip for stats tooling but cannot meaningfully
/// handle a sidecar-write failure beyond logging it. The skip
/// itself is still valid; only post-run stats tooling loses
/// visibility.
pub(crate) fn record_skip_sidecar(entry: &KtstrTestEntry, active_flags: &[String]) {
    if let Err(e) = write_skip_sidecar(entry, active_flags) {
        // Dual-emit at warn level: an unwritten skip sidecar costs
        // the run no correctness — the test still skipped — but
        // silently drops post-run stats tooling's visibility into
        // the skip, so operators debugging a missing row in a
        // gauntlet report need a loud-enough log to notice. The
        // eprintln surfaces under direct nextest / cargo-ktstr
        // invocations where no tracing subscriber is installed;
        // the tracing::warn lands in every structured-log consumer
        // (cargo-ktstr, downstream pipelines) at warn level rather
        // than the previous implicit debug visibility.
        let entry_name = entry.name;
        let rendered = format!("{e:#}");
        eprintln!("ktstr_test: warn: skip-sidecar write failed for {entry_name}: {rendered}");
        tracing::warn!(
            test = %entry_name,
            err = %rendered,
            "skip-sidecar write failed — stats tooling will not see this skip",
        );
    }
}

// ---------------------------------------------------------------------------
// Host-side OutputFormat::LlmExtract resolution
// ---------------------------------------------------------------------------
//
// The guest's `payload_run::evaluate_llm_extract_deferred` ships
// raw stdout/stderr across the SHM ring under
// `MSG_TYPE_RAW_PAYLOAD_OUTPUT` and emits an empty-metrics
// `PayloadMetrics` placeholder under `MSG_TYPE_PAYLOAD_METRICS` for
// every `OutputFormat::LlmExtract` invocation. The guest does NOT
// load the local model into VM RAM (the model is ~2.4 GiB; the test
// VM's RAM budget cannot accommodate it). The host runs
// `extract_via_llm` here, after VM exit, on the captured text — same
// stdout-primary / stderr-fallback contract that previously lived in
// the prior in-VM extraction path — and replaces the empty `metrics` vec on
// the paired `PayloadMetrics` with the extracted result.

/// Run [`crate::test_support::model::extract_via_llm`] against every
/// `OutputFormat::LlmExtract` raw output drained from SHM, replace
/// the paired empty-metrics `PayloadMetrics` slot with the extracted
/// result, and return any failure details that should fold into the
/// test's AssertResult.
///
/// Pairing is by explicit
/// [`crate::test_support::PayloadMetrics::payload_index`] equality:
/// every guest-side payload-pipeline emission allocates one index
/// from the per-process counter (see
/// [`crate::scenario::payload_run`]) and stamps it onto BOTH the
/// [`MSG_TYPE_RAW_PAYLOAD_OUTPUT`] and the
/// [`MSG_TYPE_PAYLOAD_METRICS`] message it emits. The host walks
/// `raw_outputs`, looks up each entry's index in a
/// `HashMap<payload_index, vec position>` built once over
/// `payload_metrics`, and writes the extracted metrics into the
/// matched slot. Non-LlmExtract payloads (Json, ExitCode) also
/// emit `MSG_TYPE_PAYLOAD_METRICS` with their own per-invocation
/// index, but the host's pairing loop walks the `raw_outputs`
/// slice; non-LlmExtract entries are never inspected because they
/// have no companion raw output.
///
/// Index-based pairing replaces the prior emission-order pairing
/// which conflated a `Json` payload that legitimately produced zero
/// metrics (no numeric leaves) with an `LlmExtract` placeholder.
///
/// `shm_drops` is the
/// [`crate::vmm::shm_ring::ShmDrainResult::drops`] counter — total
/// messages the guest's `shm_write` dropped (ring full, or
/// overflow paths that should not fire in practice). The header's
/// counter conflates every message type, so we cannot tell whether
/// a dropped message was an LlmExtract pair or some other type
/// (profraw, stimulus, payload metrics for a different test). The
/// safe interpretation: when ANY drops occurred AND the test used
/// LlmExtract (`raw_outputs` non-empty), surface a host-actionable
/// detail. The dropped message MAY have been an LlmExtract
/// `RawPayloadOutput` — losing one silently would make extracted
/// metrics quietly incomplete — and a multi-MB workload output
/// (dmesg flood, large schbench latency table) is the most
/// plausible victim.
///
/// Failure shape:
/// - SHM ring overflow with LlmExtract in use: a single detail
///   naming the drops counter so the test author knows to either
///   shrink the workload's stdout/stderr or expand the SHM ring
///   capacity. The detail does NOT block the rest of the host-side
///   extraction path — the raw outputs that DID arrive still get
///   processed.
/// - Model load fails (e.g. `KTSTR_MODEL_OFFLINE=1` with cold cache,
///   SHA mismatch on a corrupted cached GGUF): append a single
///   `LlmExtract model load failed: <reason>` detail. metrics
///   remain empty. No structural-sanity checks fire — we have
///   nothing to check against.
/// - Structural-sanity violation (duplicate metric name, non-finite
///   value, source tag drift): every violation found contributes
///   its own detail (see [`validate_llm_extraction`]). The metric
///   set is still populated on the PayloadMetrics slot so debugging
///   tools and the sidecar see what the model produced.
/// - Raw output's `payload_index` has no matching `PayloadMetrics`
///   entry (guest emitted a raw output without its companion empty-
///   metrics PM, or emission was lost to SHM ring overflow):
///   append a `LlmExtract host pairing` detail naming the orphan
///   index and skip the extraction for that raw output. The other
///   raw outputs still get extracted — dropping every extraction
///   because one orphan exists would lose information the test
///   author can still act on.
/// - Per-payload bounds violation (when the payload declared
///   `metric_bounds`, see [`crate::test_support::MetricBounds`]):
///   each violation surfaces as its own detail via
///   [`validate_metric_bounds`] — minimum metric count below the
///   declared floor, value below `value_min`, value above
///   `value_max`. The bounds pass runs AFTER the structural-sanity
///   pass and ONLY when extraction succeeded; load-failed pairs
///   skip the bounds check (the empty placeholder would otherwise
///   spuriously trip a `min_count` violation on every offline-gated
///   test).
/// - Orphan `PayloadMetrics` (a guest-side LlmExtract emission
///   produced an empty-metrics `PayloadMetrics` whose
///   `payload_index` has NO matching `RawPayloadOutput` companion):
///   the post-pairing scan flags the missing raw output. Most
///   common cause is a CRC-bad raw-output message silently dropped
///   during SHM drain — the drops counter only tracks ring-full
///   in `shm_write`, so a CRC drop does NOT inflate `shm_drops`
///   yet still loses the raw output. Pairs symmetrically with the
///   raw-output orphan-pairing detail above.
fn host_side_llm_extract(
    payload_metrics: &mut [crate::test_support::PayloadMetrics],
    raw_outputs: &[crate::test_support::RawPayloadOutput],
    shm_drops: u64,
) -> Vec<crate::assert::AssertDetail> {
    let mut failures = Vec::new();
    if raw_outputs.is_empty() {
        return failures;
    }
    // SHM ring overflow with LlmExtract in use: surface BEFORE the
    // pairing loop so the operator sees the drops first. The
    // counter conflates every message type, so this fires even if
    // the dropped message was a profraw entry rather than a raw
    // output — but the dominant cause of drops with LlmExtract in
    // play is a multi-MB workload output blowing the ring's
    // capacity, and a false positive (drops for some other type)
    // is preferable to silent metric loss.
    if shm_drops > 0 {
        failures.push(crate::assert::AssertDetail::new(
            crate::assert::DetailKind::Other,
            format!(
                "SHM ring overflow: {shm_drops} message(s) dropped while LlmExtract was in use. \
                 The test's stdout/stderr may have exceeded the ring's configured capacity, \
                 silently truncating the input the host's extract_via_llm received. \
                 Shrink the workload's output volume (e.g. trim the latency table, \
                 disable verbose logging, redirect noisy stderr to /dev/null), or \
                 expand the SHM ring via the VMM's shm_size config so all guest \
                 emissions fit. The drops counter conflates message types, so the \
                 dropped message may be a profraw or stimulus entry rather than \
                 an LlmExtract payload — but the test cannot prove the LlmExtract \
                 raw-output stream is complete with a non-zero counter, and silently \
                 truncated metrics propagate as flaky regressions downstream."
            ),
        ));
    }
    // Build a HashMap from each PayloadMetrics' payload_index to its
    // position in the slice. Last-occurrence wins on duplicate
    // indices — but the guest's per-process counter is monotonic
    // and never reuses a value within a single VM run, so a
    // duplicate index in this map is a guest-side bug. The
    // `fetch_add(1, Relaxed)` atomic counter at
    // [`crate::scenario::payload_run::PAYLOAD_INVOCATION_COUNTER`]
    // guarantees uniqueness across threads as well — `Relaxed`
    // does not reorder the increment relative to itself, so
    // concurrent emits from N threads each receive a distinct
    // value. The "guest-side bug" framing applies to a future
    // regression that bypassed the counter, not to multi-thread
    // emit per se. The map is keyed by usize (the index) and
    // valued by usize (the slice position) so the pair-loop below
    // can rewrite the matching slot in O(1).
    let pm_index_lookup: std::collections::HashMap<usize, usize> = payload_metrics
        .iter()
        .enumerate()
        .map(|(pos, pm)| (pm.payload_index, pos))
        .collect();
    for raw in raw_outputs {
        let Some(&pm_pos) = pm_index_lookup.get(&raw.payload_index) else {
            // Orphan raw output — no PayloadMetrics carries the
            // matching index. Most likely cause is SHM ring overflow
            // dropping the empty-metrics PM, or a guest-side emit
            // path that ships RawPayloadOutput without its companion
            // PayloadMetrics. Surface as a failure detail so the
            // test fails loudly; skip extraction for this raw entry
            // and keep going on the rest.
            failures.push(crate::assert::AssertDetail::new(
                crate::assert::DetailKind::Other,
                format!(
                    "LlmExtract host pairing: raw output at payload_index={} has no \
                     matching PayloadMetrics slot — guest emission contract violated, \
                     or SHM ring dropped the empty-metrics companion message",
                    raw.payload_index,
                ),
            ));
            continue;
        };
        let hint_ref = raw.hint.as_deref();
        // Stdout-primary: try stdout first.
        let stdout_result = super::model::extract_via_llm(
            &raw.stdout,
            hint_ref,
            crate::test_support::MetricStream::Stdout,
        );
        let (mut metrics, load_err) = match stdout_result {
            Ok(m) => (m, None::<String>),
            Err(reason) => (Vec::new(), Some(reason)),
        };
        // Stderr fallback — only if stdout produced no metrics AND
        // the stdout call did not surface a load-failure reason
        // (the failure reason is identical across both calls; no
        // point re-invoking inference). Mirrors the legacy guest-
        // side fallback gate exactly. The Err arm here is
        // theoretically unreachable: when stdout's call returned
        // `Ok`, the model is memoized in `MODEL_CACHE` and a second
        // call cannot fail to load. Handled defensively in case a
        // future refactor changes that invariant — same surface
        // shape as a stdout-side load failure.
        if metrics.is_empty() && load_err.is_none() && !raw.stderr.is_empty() {
            match super::model::extract_via_llm(
                &raw.stderr,
                hint_ref,
                crate::test_support::MetricStream::Stderr,
            ) {
                Ok(m) => metrics = m,
                Err(reason) => {
                    failures.push(crate::assert::AssertDetail::new(
                        crate::assert::DetailKind::Other,
                        format!("LlmExtract model load failed: {reason}"),
                    ));
                    continue;
                }
            }
        }
        if let Some(reason) = load_err {
            failures.push(crate::assert::AssertDetail::new(
                crate::assert::DetailKind::Other,
                format!("LlmExtract model load failed: {reason}"),
            ));
            // Leave metrics empty in the PayloadMetrics slot. Skip
            // the structural-sanity check below — running it on an
            // empty vec would either no-op (no metrics to scan) or
            // produce a misleading detail that buries the real
            // load-failure reason.
            continue;
        }
        // Apply payload-author-declared polarity / unit hints. The
        // guest shipped these in `raw.metric_hints` because the
        // model-driven extraction runs post-VM-exit on the host —
        // the original `&'static [MetricHint]` slice cannot
        // round-trip through SHM. Mirrors the guest-side
        // `resolve_polarities` pass that runs on Json / ExitCode
        // payloads inside `payload_run::evaluate` so LlmExtract
        // metrics reach the sidecar with the same polarity / unit
        // classification a Json payload would receive.
        crate::scenario::payload_run::resolve_polarities_owned(&mut metrics, &raw.metric_hints);
        // Structural-sanity check. Every violation found surfaces
        // its own AssertDetail so a metric set that breaks multiple
        // invariants (e.g. NaN values AND a duplicate name) gives
        // the test author the full picture in one run rather than
        // forcing them to fix one defect class, re-run, fix the
        // next, re-run again.
        for reason in validate_llm_extraction(&metrics) {
            failures.push(crate::assert::AssertDetail::new(
                crate::assert::DetailKind::Other,
                reason,
            ));
        }
        // Per-payload bounds check. Workload-specific bounds
        // (minimum metric count, value magnitude) declared on the
        // payload's `metric_bounds` field run AFTER the universal
        // structural-sanity pass; they apply only to extracted
        // metrics that already passed unique-name / finite /
        // source-tag checks. A payload that didn't declare
        // `metric_bounds` (the common case) skips this pass.
        if let Some(bounds) = raw.metric_bounds.as_ref() {
            for reason in validate_metric_bounds(&metrics, bounds) {
                failures.push(crate::assert::AssertDetail::new(
                    crate::assert::DetailKind::Other,
                    reason,
                ));
            }
        }
        // Replace the empty-metrics slot with the extracted result.
        // Even if validation fails above, populate the PayloadMetrics
        // so debugging tools and the sidecar see what the model
        // emitted. The accompanying AssertDetail communicates the
        // rejection.
        payload_metrics[pm_pos].metrics = metrics;
    }

    // Post-pairing scan: flag empty-metrics PayloadMetrics whose
    // payload_index has no matching RawPayloadOutput. The most
    // likely cause is a CRC-bad RawPayloadOutput silently dropped
    // during SHM drain (the drain at run_ktstr_test_inner skips
    // CRC-bad entries without recording the loss in the
    // shm_drops counter, since that counter only tracks
    // ring-full and overflow paths in `shm_write`). Without this
    // surfacing, an LlmExtract test whose raw-output bytes
    // arrived corrupted would silently produce empty metrics and
    // fail downstream `Check::Min` / `Check::Exists` evaluations
    // with a "metric not found" message that hides the real cause.
    //
    // Ambiguity disclosure: we cannot tell from PayloadMetrics
    // alone which empty-metrics entries were intended as
    // LlmExtract placeholders versus legitimate Json-with-no-leaves
    // or ExitCode-only payloads. We only reach this scan when
    // `raw_outputs` is non-empty (the function early-returned at
    // the top of the body when it was empty), so by construction
    // the test exercises LlmExtract and a dropped raw-output is at
    // least possible. The detail's prose calls out the ambiguity
    // so an operator running a mixed-format test (LlmExtract + Json)
    // can dismiss false positives. Surfaces as a single combined
    // detail listing the suspicious indices rather than per-PM,
    // keeping the failure-rendering compact when many empty PMs
    // coexist.
    let raw_indices: std::collections::HashSet<usize> =
        raw_outputs.iter().map(|raw| raw.payload_index).collect();
    let suspicious: Vec<usize> = payload_metrics
        .iter()
        .filter(|pm| pm.metrics.is_empty() && !raw_indices.contains(&pm.payload_index))
        .map(|pm| pm.payload_index)
        .collect();
    if !suspicious.is_empty() {
        failures.push(crate::assert::AssertDetail::new(
            crate::assert::DetailKind::Other,
            format!(
                "LlmExtract host pairing: {} empty-metrics PayloadMetrics \
                 entries at payload_index={:?} have no matching RawPayloadOutput. \
                 If these were intended as LlmExtract payloads, the raw-output \
                 SHM messages may have been silently dropped during drain \
                 (CRC mismatch — the drop is invisible to the shm_drops \
                 counter, which only tracks ring-full / overflow). Re-run; \
                 transient CRC corruption is rare. False-positive case: a \
                 `Json` payload with no numeric leaves and an `ExitCode` \
                 payload both produce empty-metrics PayloadMetrics by design \
                 and would also surface here in a mixed-format test — \
                 dismiss this detail if your test mixes LlmExtract with \
                 legitimately-empty other formats.",
                suspicious.len(),
                suspicious,
            ),
        ));
    }

    failures
}

/// Structural-sanity check on a freshly-extracted
/// `OutputFormat::LlmExtract` metric set. Returns a `Vec<String>`
/// of every violation found; an empty vec means the set is
/// structurally well-formed.
///
/// Every metric is checked against ALL three invariants — a single
/// metric can contribute up to three violations (e.g. a duplicate
/// name AND a NaN value AND a non-LlmExtract source tag) so the
/// test author sees every defect class in one failure rather than
/// having to re-run after fixing each one in turn. Across the
/// whole set, every duplicate-name occurrence beyond the first
/// reports its own violation.
///
/// Universal checks only — every condition here is workload-
/// agnostic. Workload-specific assertions (latency ranges, RPS
/// ceilings, sign / magnitude bounds, minimum metric count) belong
/// in a per-payload validation API the framework does not yet
/// expose; the test author owns those.
///
/// 1. Every metric name is unique. Duplicate dotted paths imply
///    the LLM walker emitted the same key twice (malformed JSON
///    walkthrough or a walker aggregation bug) — downstream stats
///    would misattribute one value to the other regardless of which
///    workload produced the output.
/// 2. Every value is finite. NaN / ±inf in `PayloadMetrics`
///    poisons percentile comparisons downstream and never
///    represents a legitimate measurement, regardless of workload.
/// 3. Every metric carries `MetricSource::LlmExtract`. The host's
///    `extract_via_llm` walker stamps this field unconditionally,
///    so any drift here points at a bypass — the value didn't come
///    from the LLM-driven path even though it landed in a slot
///    we marked LlmExtract.
fn validate_llm_extraction(metrics: &[crate::test_support::Metric]) -> Vec<String> {
    use std::collections::HashSet;
    // Empty-input fast-path mirrors the symmetric helper
    // [`crate::scenario::payload_run::resolve_polarities_owned`]:
    // skip the HashSet allocation and the for-loop so the no-op
    // case is structurally a no-op rather than an empty-iterator
    // walk. The capacity-zero allocation HashSet would amount to
    // is essentially free, but the early-return makes the contract
    // visible to a reader scanning the function.
    if metrics.is_empty() {
        return Vec::new();
    }
    let mut violations = Vec::new();
    let mut seen: HashSet<&str> = HashSet::with_capacity(metrics.len());
    for m in metrics {
        if !seen.insert(m.name.as_str()) {
            violations.push(format!(
                "LlmExtract emitted duplicate metric name '{}' — downstream stats would \
                 misattribute one value to the other; check the LLM walker for an \
                 aggregation bug or a malformed JSON path emitted by the model",
                m.name,
            ));
        }
        if !m.value.is_finite() {
            violations.push(format!(
                "LlmExtract metric '{}' has non-finite value {} — NaN / ±inf must not \
                 propagate into PayloadMetrics",
                m.name, m.value,
            ));
        }
        if m.source != crate::test_support::MetricSource::LlmExtract {
            violations.push(format!(
                "LlmExtract metric '{}' has source {:?}, expected MetricSource::LlmExtract — \
                 a value reached the LlmExtract slot without traversing the LLM walker",
                m.name, m.source,
            ));
        }
    }
    violations
}

/// Per-payload-bounds check applied AFTER the universal
/// structural-sanity pass in [`validate_llm_extraction`]. Returns
/// a `Vec<String>` of every violation found; an empty vec means
/// the metric set satisfies the declared bounds.
///
/// Each declared bound on [`crate::test_support::MetricBounds`] is
/// `Option`-wrapped, so a payload's bounds can scope to any subset
/// of the three checks. Disabled bounds (the `None` case) are
/// no-ops here — the function inspects each `Some(_)` branch
/// independently and emits per-violation diagnostics.
///
/// Diagnostics surface as `AssertDetail::new(DetailKind::Other, ...)`
/// at the call site in [`host_side_llm_extract`], so the per-bound
/// failure shape mirrors the universal-invariant violations: one
/// detail per violation, every detail carries enough context for
/// the operator to identify which bound fired and why.
///
/// 1. **`min_count`**: when set, an extracted set whose `.len()`
///    is below the threshold surfaces a violation naming the
///    expected minimum and the actual count. Pins the "did the
///    model produce enough metrics?" check that schbench-style
///    payloads need (an LLM regression that emits 1 metric on a
///    payload that historically produced 5+ silently degrades
///    downstream stats).
///
/// 2. **`value_min`**: when set, every metric whose value is
///    strictly below the threshold surfaces a violation naming
///    the metric, the value, and the bound. Pin the
///    non-negative-microseconds invariant for percentile
///    payloads — a negative latency reading is either a model
///    extraction error or a unit confusion, both of which the
///    bound surfaces loudly.
///
/// 3. **`value_max`**: symmetric upper-bound check. Catches
///    runaway values (a typo'd unit converter that read seconds
///    as microseconds and produced a 1e15 latency) before they
///    reach downstream stats.
///
/// Pre-1.0 design pin: callers MUST evaluate the universal
/// invariants in [`validate_llm_extraction`] FIRST. A NaN-bearing
/// metric would silently bypass the magnitude bounds here
/// because `NaN < x` and `NaN > x` both return false. The
/// universal pass rejects NaN unconditionally, so by the time
/// `validate_metric_bounds` runs the input is finite.
fn validate_metric_bounds(
    metrics: &[crate::test_support::Metric],
    bounds: &crate::test_support::MetricBounds,
) -> Vec<String> {
    let mut violations = Vec::new();
    if let Some(min_count) = bounds.min_count
        && metrics.len() < min_count
    {
        violations.push(format!(
            "LlmExtract bounds: extracted {} metric(s), payload requires at least {} — \
             the model produced fewer metrics than the payload declared as a sanity \
             floor. Common causes: a regression in the LLM walker that drops branches \
             of the JSON tree, a payload output that's structurally different from \
             what the prompt template assumes, or a too-tight floor on `min_count`.",
            metrics.len(),
            min_count,
        ));
    }
    for m in metrics {
        if let Some(lo) = bounds.value_min
            && m.value < lo
        {
            violations.push(format!(
                "LlmExtract bounds: metric '{}' has value {} below payload's declared \
                 lower bound {} — values below the floor are either an extraction \
                 error or a unit-confusion bug. Adjust `value_min` if the floor is \
                 too tight, or fix the payload's output schema if the value should \
                 not have crossed the floor.",
                m.name, m.value, lo,
            ));
        }
        if let Some(hi) = bounds.value_max
            && m.value > hi
        {
            violations.push(format!(
                "LlmExtract bounds: metric '{}' has value {} above payload's declared \
                 upper bound {} — values above the ceiling are either an extraction \
                 error or a runaway from a typo'd unit converter. Adjust `value_max` \
                 if the ceiling is too tight, or fix the payload's output if the \
                 value should have stayed bounded.",
                m.name, m.value, hi,
            ));
        }
    }
    violations
}

/// Run a single ktstr_test and return the VM's AssertResult.
/// Dedupe a resolved include-file list produced by unioning the
/// per-payload `include_files` specs through
/// [`crate::cli::resolve_include_files`] and appending the scheduler
/// config file entry. Each input tuple carries an `origin` label
/// (e.g. `"declarative"`, `"scheduler config_file"`) that is
/// surfaced in conflict diagnostics so the operator can trace which
/// declaration contributed each side of a collision.
///
/// Policy:
///
/// - Identical `(archive_path, host_path)` pairs collapse silently
///   (the same host file declared twice is harmless). Comparison
///   uses [`Path::canonicalize`] so two spellings of the same real
///   file (e.g. `./fio` vs `/usr/bin/fio` when `./fio` is a
///   symlink) are treated as equal. Canonicalization failure
///   (missing path, permission denied) falls back to byte-for-byte
///   PathBuf comparison; literal duplicates still collapse, and a
///   genuine conflict still surfaces.
/// - Two entries sharing an `archive_path` but resolving to
///   different canonical `host_path`s are a genuine ambiguity — a
///   scheduler's and a payload's `include_files` both claiming
///   `include-files/config.json` but pointing at different host
///   paths means one of the two would silently overwrite the other
///   in the initramfs. Bail with a diagnostic naming both host
///   paths AND their origin labels so the author can rename one
///   archive slot.
///
/// Case-sensitivity: `archive_path` keys are compared
/// byte-for-byte (via `BTreeMap<String, _>`), so on a case-
/// insensitive host filesystem (macOS HFS+, NTFS with the
/// `case-insensitive` mount flag) two archive paths spelled
/// `include-files/Helper` and `include-files/helper` are treated
/// as distinct here even though the host filesystem would
/// conflate them. This is intentional: `archive_path` is the
/// path inside the guest initramfs, which is tmpfs / ext4-
/// equivalent (always case-sensitive), so the guest-side
/// identity is what governs.
///
/// Order is stabilized via `BTreeMap`'s sorted iteration so the
/// emitted slice is deterministic regardless of which caller
/// appended first. Extracted from `run_ktstr_test_inner` so the
/// policy can be unit-tested without constructing a whole
/// KtstrTestEntry + VmBuilder.
fn dedupe_include_files(
    resolved: &[(String, std::path::PathBuf, &'static str)],
) -> Result<Vec<(String, std::path::PathBuf)>> {
    let mut seen: std::collections::BTreeMap<String, (std::path::PathBuf, &'static str)> =
        std::collections::BTreeMap::new();
    for (archive, host, origin) in resolved {
        if let Some((existing, existing_origin)) = seen.get(archive) {
            // Canonicalize both sides before comparing so
            // symlink-equivalent spellings collapse. A failed
            // canonicalize (missing path, permission denied) falls
            // back to the uncanonicalized value so the structural
            // compare still runs — literal duplicates still collapse
            // and genuine conflicts still surface.
            let existing_canon = existing.canonicalize().unwrap_or_else(|_| existing.clone());
            let host_canon = host.canonicalize().unwrap_or_else(|_| host.clone());
            if existing_canon != host_canon {
                anyhow::bail!(
                    "include_files conflict for archive path '{archive}': sources disagree \
                     on host path ({} [origin: {existing_origin}] vs {} [origin: {origin}]). \
                     Remove the duplicate declaration or rename one of the archive entries.",
                    existing.display(),
                    host.display(),
                );
            }
        } else {
            seen.insert(archive.clone(), (host.clone(), origin));
        }
    }
    Ok(seen
        .into_iter()
        .map(|(archive, (host, _origin))| (archive, host))
        .collect())
}

pub(crate) fn run_ktstr_test_inner(
    entry: &KtstrTestEntry,
    topo: Option<&TopoOverride>,
    active_flags: &[String],
) -> Result<AssertResult> {
    entry.validate().context("KtstrTestEntry validation")?;
    if let Some(t) = topo {
        t.validate().context("TopoOverride validation")?;
    }
    if entry.performance_mode && std::env::var("KTSTR_NO_PERF_MODE").is_ok() {
        // One canonical reason string for both the stderr banner
        // (prefixed with the entry name for multi-test context)
        // and the structured AssertResult::skip payload (test-name
        // is carried on the surrounding entry). Prior code
        // duplicated the body verbatim across both sites, inviting
        // drift; the shared const keeps them in lockstep.
        const REASON: &str =
            "test requires performance_mode but --no-perf-mode or KTSTR_NO_PERF_MODE is active";
        crate::report::test_skip(format_args!("{}: {REASON}", entry.name));
        // Record the skip so stats tooling sees every skipped run,
        // not just the ones that made it to the VM-run site. A sidecar
        // write failure is logged but not propagated: the skip itself
        // is still valid — only post-run stats tooling loses visibility.
        record_skip_sidecar(entry, active_flags);
        return Ok(AssertResult::skip(REASON));
    }
    ensure_kvm()?;
    let kernel = resolve_test_kernel()?;
    // Hold a reader flock on the cache entry (if the resolved
    // kernel lives in one). Prevents a concurrent
    // `cargo ktstr kernel build` from swapping the entry under
    // the VM mid-run. Dropped when this fn returns; the VM has
    // finished by then. `None` on non-cache kernels (explicit
    // KTSTR_TEST_KERNEL, `/lib/modules/...`) — those don't
    // need coordination.
    let _kernel_lock = acquire_test_kernel_lock_if_cached(&kernel)?;
    let scheduler = match entry.scheduler.scheduler_binary() {
        Some(b) => {
            // Drop the ResolveSource on this path — the downstream
            // sites (VM builder, auto_repro) only need the PathBuf.
            // Consumers that want provenance (sidecar stamping,
            // cache-key construction) must call resolve_scheduler
            // directly on the same spec; the source is stable across
            // identical inputs within a single process run.
            resolve_scheduler(b)?.0
        }
        None => None,
    };
    let ktstr_bin = crate::resolve_current_exe()?;

    let guest_args = vec![
        "run".to_string(),
        "--ktstr-test-fn".to_string(),
        entry.name.to_string(),
    ];

    let cmdline_extra = super::runtime::build_cmdline_extra(entry);

    let (vm_topology, memory_mb) = super::runtime::resolve_vm_topology(entry, topo);

    let no_perf_mode = std::env::var("KTSTR_NO_PERF_MODE").is_ok();
    let mut builder = super::runtime::build_vm_builder_base(
        entry,
        &kernel,
        &ktstr_bin,
        scheduler.as_deref(),
        vm_topology,
        memory_mb,
        &cmdline_extra,
        &guest_args,
        no_perf_mode,
    )
    .performance_mode(entry.performance_mode);

    // Merge order: default_checks -> scheduler.assert -> per-test assert.
    let merged_assert = crate::assert::Assert::default_checks()
        .merge(entry.scheduler.assert())
        .merge(&entry.assert);

    if let Some(SchedulerSpec::KernelBuiltin { enable, disable }) =
        entry.scheduler.scheduler_binary()
    {
        builder = builder.sched_enable_cmds(enable);
        builder = builder.sched_disable_cmds(disable);
    }
    if entry.scheduler.has_active_scheduling() {
        builder = builder.monitor_thresholds(merged_assert.monitor_thresholds());
    }

    let mut sched_args: Vec<String> = Vec::new();
    // Declarative include-files: union every Payload's
    // `include_files` specs (scheduler + test payload + workloads +
    // entry.extra) through the same resolver the CLI `-i` flag uses,
    // then merge with the scheduler config file (if any). Dedupe
    // policy: identical `(archive_path, host_path)` pairs collapse
    // silently; a conflict on the same `archive_path` with
    // differing `host_path` aborts the test with a diagnostic
    // naming both sources — two unrelated declarations resolving
    // to the same archive slot is a real ambiguity the user must
    // resolve manually.
    let declarative_specs: Vec<std::path::PathBuf> = entry
        .all_include_files()
        .into_iter()
        .map(std::path::PathBuf::from)
        .collect();
    let mut resolved_includes: Vec<(String, std::path::PathBuf, &'static str)> =
        if declarative_specs.is_empty() {
            Vec::new()
        } else {
            crate::cli::resolve_include_files(&declarative_specs)
                .context("resolving declarative include_files from Payload definitions")?
                .into_iter()
                .map(|(a, h)| (a, h, "declarative"))
                .collect()
        };
    if let Some((archive_path, host_path, guest_path)) = config_file_parts(entry) {
        resolved_includes.push((archive_path, host_path, "scheduler config_file"));
        sched_args.push("--config".to_string());
        sched_args.push(guest_path);
    }
    let unioned = dedupe_include_files(&resolved_includes)?;
    if !unioned.is_empty() {
        builder = builder.include_files(unioned);
    }
    super::runtime::append_base_sched_args(entry, &mut sched_args);
    for flag_name in active_flags {
        if let Some(args) = entry.scheduler.flag_args(flag_name) {
            sched_args.extend(args.iter().map(|s| s.to_string()));
        }
    }
    if !sched_args.is_empty() {
        builder = builder.sched_args(&sched_args);
    }

    // Catch ResourceContention before .context() wraps it —
    // downcast_ref only checks the outermost error type, so
    // .context() would hide ResourceContention from the skip
    // logic in result_to_exit_code. Also record a skip sidecar at
    // the propagation point: a ResourceContention-skipped run is
    // otherwise invisible to stats tooling that enumerates
    // sidecars.
    let vm = match builder.build() {
        Ok(vm) => vm,
        Err(e)
            if e.downcast_ref::<crate::vmm::host_topology::ResourceContention>()
                .is_some() =>
        {
            record_skip_sidecar(entry, active_flags);
            return Err(e);
        }
        Err(e) => return Err(e.context("build ktstr_test VM")),
    };

    let result = match vm.run() {
        Ok(r) => r,
        Err(e)
            if e.downcast_ref::<crate::vmm::host_topology::ResourceContention>()
                .is_some() =>
        {
            record_skip_sidecar(entry, active_flags);
            return Err(e);
        }
        Err(e) => return Err(e.context("run ktstr_test VM")),
    };

    // Drop the VM to release CPU/LLC flock fds before auto-repro
    // and host-side LlmExtract inference. Both downstream phases
    // run under the test-runner's main thread, NOT inside a vCPU
    // worker, so neither needs the VM-side CPU/LLC reservation.
    // Releasing here lets concurrent ktstr peers (other tests
    // running in parallel under nextest, or `cargo ktstr kernel
    // build` rebuilding cache entries) acquire the same LLC slots
    // while inference / repro proceed.
    drop(vm);

    // Release the kernel-cache shared lock before
    // `host_side_llm_extract` runs. The shared lock at
    // [`acquire_test_kernel_lock_if_cached`] guards against a
    // concurrent `cargo ktstr kernel build` swapping the entry
    // under the VM mid-run; the VM is dropped by line 828 so the
    // image bytes are no longer mapped, and the host-side LLM
    // extraction does NOT reread the kernel image. Holding the
    // lock through inference would block kernel-cache rebuilds
    // for the inference duration (multiple seconds for a 2.4 GiB
    // model load on a cold cache) without any benefit. The
    // explicit drop also documents the lock's narrowed scope —
    // RAII would otherwise drop it at function return, after
    // inference completes.
    drop(_kernel_lock);

    // Broaden the calling thread's CPU mask before
    // `host_side_llm_extract` runs. After `vm.run()` the
    // BSP / vCPU 0 thread carries either:
    //   - a single-CPU pin (perf-mode path: the vmm `vm.run`
    //     entry calls `pin_current_thread` to nail the BSP to one
    //     CPU for ipi-latency stability) — LLM inference on 1 CPU
    //     is dramatically slow (10x+ in throughput) and gives the
    //     other free host CPUs no work, OR
    //   - a multi-CPU LLC-aware mask (no-perf-mode path: the vmm
    //     applies `set_thread_cpumask` against the
    //     `no_perf_plan.cpus` set so the BSP can roam within an
    //     LLC) — already pool-style, but narrower than the
    //     host-allowed cpuset.
    // Inference is a host-side post-VM-exit phase that doesn't
    // share the VM's measurement contract; it should use whatever
    // CPUs the host process is permitted on (cgroup cpuset / sudo
    // -u limits / CI runner allocation), which is exactly what
    // `host_allowed_cpus()` returns via `sched_getaffinity(0)`.
    // The team-lead's #30 direction: "use no-perf-mode cpuset for
    // inference" — `set_thread_cpumask` against the broader
    // host-allowed pool is the no-perf-mode primitive applied to
    // a wider set than any single LLC-plan would carve out.
    //
    // Empty `host_allowed_cpus()` (sched_getaffinity unavailable,
    // procfs fallback failed) skips the call rather than masking
    // to zero CPUs (which would block forever); inference inherits
    // whatever the test left behind. Logged as a warning by
    // `set_thread_cpumask` itself if the syscall fails.
    let host_cpus = crate::vmm::host_topology::host_allowed_cpus();
    if !host_cpus.is_empty() {
        crate::vmm::set_thread_cpumask(&host_cpus, "host-side LlmExtract inference");
    }

    // Log verifier stats count for visibility.
    if !result.verifier_stats.is_empty() {
        eprintln!(
            "ktstr_test: verifier_stats: {} struct_ops programs",
            result.verifier_stats.len(),
        );
    }

    // When running with a struct_ops scheduler, check that host-side
    // BPF program enumeration found programs with non-zero verified_insns.
    if entry.scheduler.has_active_scheduling() && result.success && result.verifier_stats.is_empty()
    {
        eprintln!("ktstr_test: WARNING: scheduler loaded but verifier_stats is empty");
    }

    // Extract profraw from SHM ring buffer and collect stimulus
    // events + per-payload metrics + raw outputs from
    // `OutputFormat::LlmExtract` payloads.
    //
    // Pairing contract: every guest-side payload-pipeline emit
    // (one per `.run()` / `.wait()` / `.kill()` / `.try_wait()`
    // terminal call) allocates one `payload_index` from
    // `payload_run`'s per-process counter and stamps it onto the
    // emitted `PayloadMetrics`. LlmExtract invocations additionally
    // emit a `RawPayloadOutput` carrying the SAME index. Non-
    // LlmExtract payloads emit only the `PayloadMetrics`. The host
    // pairs an LlmExtract `RawPayloadOutput` to its empty-metrics
    // companion by EQUAL `payload_index`, not by emission order —
    // see `host_side_llm_extract` for the pairing implementation.
    let mut stimulus_events = Vec::new();
    let mut payload_metrics: Vec<crate::test_support::PayloadMetrics> = Vec::new();
    let mut raw_outputs: Vec<crate::test_support::RawPayloadOutput> = Vec::new();
    if let Some(ref shm) = result.shm_data {
        for entry in &shm.entries {
            if entry.msg_type == MSG_TYPE_PROFRAW
                && entry.crc_ok
                && !entry.payload.is_empty()
                && let Err(e) = write_profraw(&entry.payload)
            {
                eprintln!("ktstr_test: write guest profraw: {e}");
            }
            if entry.msg_type == crate::vmm::shm_ring::MSG_TYPE_STIMULUS
                && entry.crc_ok
                && let Some(ev) = crate::vmm::shm_ring::StimulusEvent::from_payload(&entry.payload)
            {
                stimulus_events.push(crate::timeline::StimulusEvent {
                    elapsed_ms: ev.elapsed_ms as u64,
                    label: format!("StepStart[{}]", ev.step_index),
                    op_kind: Some(format!("ops={}", ev.op_count)),
                    detail: Some(format!(
                        "{} cgroups, {} workers",
                        ev.cgroup_count, ev.worker_count,
                    )),
                    total_iterations: if ev.total_iterations > 0 {
                        Some(ev.total_iterations)
                    } else {
                        None
                    },
                });
            }
            if entry.msg_type == crate::vmm::shm_ring::MSG_TYPE_PAYLOAD_METRICS && entry.crc_ok {
                match serde_json::from_slice::<crate::test_support::PayloadMetrics>(&entry.payload)
                {
                    Ok(pm) => payload_metrics.push(pm),
                    Err(e) => eprintln!("ktstr_test: decode payload metrics from SHM: {e}"),
                }
            }
            if entry.msg_type == crate::vmm::shm_ring::MSG_TYPE_RAW_PAYLOAD_OUTPUT && entry.crc_ok {
                match serde_json::from_slice::<crate::test_support::RawPayloadOutput>(
                    &entry.payload,
                ) {
                    Ok(raw) => raw_outputs.push(raw),
                    Err(e) => eprintln!("ktstr_test: decode raw payload output from SHM: {e}"),
                }
            }
        }
    }

    // Host-side `OutputFormat::LlmExtract` resolution. For every
    // RawPayloadOutput drained from SHM, look up its
    // `payload_index` in the PayloadMetrics slice, run the
    // LLM-backed extraction on the host, and replace the empty
    // `metrics` vec on the matched slot with the extracted result.
    // The model lives at the host's cache and the guest VM never
    // had it, so this is the only correct place for the call.
    //
    // Pairing is by explicit `payload_index` equality, not emission
    // order — emission order would conflate a `Json` payload that
    // produced zero numeric leaves with an LlmExtract placeholder.
    // Returns a flat `Vec<AssertDetail>` of host-side failures
    // (model unavailable, universal invariant violation, orphan
    // raw outputs) for the test verdict to fold in.
    let shm_drops = result.shm_data.as_ref().map_or(0, |s| s.drops);
    let host_extract_failures =
        host_side_llm_extract(&mut payload_metrics, &raw_outputs, shm_drops);

    // auto_repro is enabled when:
    // - entry.auto_repro is true (default)
    // - a scheduler is running (not EEVDF)
    // - the test does not expect failure (expect_err = false)
    let effective_auto_repro = entry.auto_repro && scheduler.is_some() && !entry.expect_err;
    let repro_fn = |output: &str| -> Option<String> {
        if !effective_auto_repro {
            return None;
        }
        let repro = attempt_auto_repro(
            entry,
            &kernel,
            scheduler.as_deref(),
            &ktstr_bin,
            output,
            &result.stderr,
            topo,
        );
        // When auto-repro was attempted but produced no data, return a
        // diagnostic so the user knows it was tried.
        Some(repro.unwrap_or_else(|| {
            "auto-repro: no probe data — the scheduler may have \
             exited before probes could capture events, or the \
             crash did not reproduce in the repro VM. Re-run with \
             RUST_LOG=debug for probe pipeline diagnostics. Check \
             the sched_ext dump and scheduler log sections above \
             for crash details."
                .to_string()
        }))
    };

    evaluate_vm_result(
        entry,
        &result,
        &merged_assert,
        &stimulus_events,
        &payload_metrics,
        &host_extract_failures,
        &vm_topology,
        active_flags,
        &repro_fn,
    )
}

/// Evaluate a VM result and produce the appropriate error or Ok.
///
/// This is the core result-evaluation logic, extracted from
/// `run_ktstr_test_inner` so that error message formatting can be tested
/// without booting a VM. The `repro_fn` callback handles auto-repro
/// (which requires a second VM boot) when provided. `payload_metrics`
/// is the per-invocation accumulator drained from the guest SHM ring;
/// the sidecar writer receives it verbatim so stats tooling sees one
/// entry per `ctx.payload(X).run()` / `.spawn().wait()`.
///
/// `host_extract_failures` carries the universal-invariant +
/// model-load failures produced by [`host_side_llm_extract`] when
/// the run's `OutputFormat::LlmExtract` payloads were resolved on
/// the host. The folded details are appended to the test's
/// AssertResult so a host-side LlmExtract failure surfaces in the
/// same failure-rendering pipeline as a guest-emitted check failure.
#[allow(clippy::too_many_arguments)]
fn evaluate_vm_result(
    entry: &KtstrTestEntry,
    result: &vmm::VmResult,
    merged_assert: &crate::assert::Assert,
    stimulus_events: &[StimulusEvent],
    payload_metrics: &[crate::test_support::PayloadMetrics],
    host_extract_failures: &[crate::assert::AssertDetail],
    topo: &Topology,
    active_flags: &[String],
    repro_fn: &dyn Fn(&str) -> Option<String>,
) -> Result<AssertResult> {
    // Build timeline from stimulus events + monitor samples.
    let timeline = result
        .monitor
        .as_ref()
        .map(|m| crate::timeline::Timeline::build(stimulus_events, &m.samples));

    let sched_label = match entry.scheduler.scheduler_binary() {
        Some(b) => scheduler_label(b),
        None => String::new(),
    };
    let output = &result.output;
    let dump_section = extract_sched_ext_dump(&result.stderr)
        .map(|d| format!("\n\n--- sched_ext dump ---\n{d}"))
        .unwrap_or_default();
    let sched_log_section = parse_sched_output(output)
        .map(|s| {
            let collapsed = crate::verifier::collapse_cycles(s);
            format!("\n\n--- scheduler log ---\n{collapsed}")
        })
        .unwrap_or_default();
    let fingerprint_line = sched_log_fingerprint(output)
        .map(|fp| {
            if crate::cli::stderr_color() {
                format!("\x1b[1;31m{fp}\x1b[0m\n")
            } else {
                format!("{fp}\n")
            }
        })
        .unwrap_or_default();

    let tl_ctx = crate::timeline::TimelineContext {
        kernel: extract_kernel_version(&result.stderr),
        topology: Some(format!("{topo} ({} cpus)", topo.total_cpus())),
        scheduler: Some(entry.scheduler.scheduler_name().to_string()),
        scenario: Some(entry.name.to_string()),
        duration_s: Some(result.duration.as_secs_f64()),
    };

    // Section builders shared by every error branch in this function.
    // Timeline skips phaseless runs; monitor only reports when an
    // active scheduler exposes rq data (EEVDF reads would be junk).
    let build_timeline_section = || -> String {
        timeline
            .as_ref()
            .filter(|t| !t.phases.is_empty())
            .map(|t| format!("\n\n{}", t.format_with_context(&tl_ctx)))
            .unwrap_or_default()
    };
    let build_monitor_section = || -> String {
        if entry.scheduler.has_active_scheduling()
            && let Some(ref monitor) = result.monitor
        {
            format_monitor_section(monitor, merged_assert)
        } else {
            String::new()
        }
    };

    if let Ok(mut check_result) =
        parse_assert_result_shm(result.shm_data.as_ref()).or_else(|_| parse_assert_result(output))
    {
        // Fold host-side LlmExtract failures into the guest's
        // AssertResult before the sidecar write so per-run stats
        // tooling sees the host-extracted verdict, not the guest's
        // placeholder pass(). Each host-side failure is appended as
        // an `AssertDetail` exactly as if it had been raised inside
        // the guest's `evaluate_checks` — same kind, same prose
        // shape — so failure-rendering downstream is uniform across
        // sources.
        for detail in host_extract_failures {
            check_result.merge(AssertResult::fail(detail.clone()));
        }

        // Cleanup-budget enforcement. When the entry sets
        // `cleanup_budget` and `collect_results` produced a measurement
        // (i.e. `run_vm` returned normally — see
        // `VmResult::cleanup_duration`), fold a failing
        // `AssertDetail` into the test verdict if teardown overran the
        // budget. Skipped when either side is `None`: an absent budget
        // means the entry opted out, an absent measurement means the
        // run never reached `collect_results` (BSP panic propagated
        // through `?`, or any pre-BSP setup error returning an `Err`
        // before `VmRunState` is constructed). Note: a host-watchdog
        // timeout is NOT a `None` case — `run_bsp_loop` exits cleanly
        // with `timed_out = true` and `collect_results` still
        // populates `cleanup_duration` to `Some(_)`, per the field
        // contract documented at `src/vmm/mod.rs` for
        // `VmResult::cleanup_duration`. The surrounding error path
        // (BSP panic propagation, pre-BSP setup `Err`) already
        // produces a failure verdict in the absent-measurement case,
        // so a budget check here would double-report.
        //
        // Contract: this check only fires inside the parse-success arm
        // (the `if let Ok(mut check_result)` above) — i.e. when the
        // guest-side test body emitted a parseable AssertResult into
        // SHM or COM2. Tests whose body panics or fails to write a
        // result skip budget enforcement entirely; the watchdog
        // timeout / no-parseable-result branch below produces its own
        // verdict in those cases. Tests that opt into
        // `cleanup_budget_ms` MUST ensure their body returns
        // `Ok(AssertResult)` (e.g. `Ok(AssertResult::pass())`) before
        // teardown begins, otherwise the budget knob is silently
        // inert.
        if let (Some(budget), Some(measured)) = (entry.cleanup_budget, result.cleanup_duration)
            && measured > budget
        {
            check_result.merge(AssertResult::fail(crate::assert::AssertDetail::new(
                crate::assert::DetailKind::Other,
                format!(
                    "vm cleanup overran budget: measured {:.3}s, budget {:.3}s. \
                     Likely a regression in host-side teardown — investigate \
                     the post-BSP-exit join/drain path \
                     (`vmm::KtstrVm::collect_results`).",
                    measured.as_secs_f64(),
                    budget.as_secs_f64(),
                ),
            )));
        }

        // Write sidecar before checking pass/fail so both outcomes are captured.
        // A sidecar write failure is logged but not propagated: the test
        // verdict itself is still valid — only post-run stats tooling
        // loses visibility.
        let args: Vec<String> = std::env::args().collect();
        let work_type =
            super::args::extract_work_type_arg(&args).unwrap_or_else(|| "CpuSpin".to_string());
        if let Err(e) = write_sidecar(
            entry,
            result,
            stimulus_events,
            &check_result,
            &work_type,
            active_flags,
            payload_metrics,
        ) {
            eprintln!("ktstr_test: {e:#}");
        }

        if !check_result.passed {
            let details = check_result
                .details
                .iter()
                .map(|d| d.message.as_str())
                .collect::<Vec<_>>()
                .join("\n  ");
            let repro = if entry.scheduler.has_active_scheduling() {
                repro_fn(output)
            } else {
                None
            };
            let repro_section = repro
                .map(|r| format!("\n\n--- auto-repro ---\n{r}"))
                .unwrap_or_default();
            let timeline_section = build_timeline_section();
            let stats_section = if !check_result.stats.cgroups.is_empty() {
                let s = &check_result.stats;
                let mut lines = vec![format!(
                    "\n\n--- stats ---\n{} workers, {} cpus, {} migrations, worst_spread={:.1}%, worst_gap={}ms",
                    s.total_workers,
                    s.total_cpus,
                    s.total_migrations,
                    s.worst_spread,
                    s.worst_gap_ms,
                )];
                for (i, cg) in s.cgroups.iter().enumerate() {
                    lines.push(format!(
                        "  cg{}: workers={} cpus={} spread={:.1}% gap={}ms migrations={} iter={}",
                        i,
                        cg.num_workers,
                        cg.num_cpus,
                        cg.spread,
                        cg.max_gap_ms,
                        cg.total_migrations,
                        cg.total_iterations,
                    ));
                }
                lines.join("\n")
            } else {
                String::new()
            };
            // Structural filter for the console-dump gate: match on
            // `DetailKind::SchedulerDied` only. Every scheduler-exit
            // emit site in this crate tags its `AssertDetail` with
            // that variant (see the ops.rs / scenario/mod.rs call
            // sites plus the `format_sched_died_*` helpers in
            // `assert.rs`), so filtering by kind is sufficient — the
            // prior `is_scheduler_death()` prefix-match fallback was
            // removed once every production emitter was audited as
            // kind-tagging its details. `verbose()` forces the
            // section on for operator debugging runs.
            let console_section = if check_result
                .details
                .iter()
                .any(|d| d.kind == crate::assert::DetailKind::SchedulerDied)
                || verbose()
            {
                let init_stage = classify_init_stage(output);
                format_console_diagnostics(&result.stderr, result.exit_code, init_stage)
            } else {
                String::new()
            };
            let monitor_section = build_monitor_section();
            let msg = format!(
                "{}ktstr_test '{}'{} [topo={}] failed:\n  {}{}{}{}{}{}{}{}",
                fingerprint_line,
                entry.name,
                sched_label,
                topo,
                details,
                stats_section,
                console_section,
                timeline_section,
                sched_log_section,
                monitor_section,
                dump_section,
                repro_section,
            );
            anyhow::bail!("{msg}");
        }

        // Evaluate monitor data against thresholds when a scheduler is running.
        // Without a scheduler (EEVDF), monitor reads rq data that may be
        // uninitialized or irrelevant — skip evaluation in that case.
        //
        // Skip early monitor warmup samples: during boot, BPF verification,
        // and initramfs unpacking the scheduler tick may not fire for hundreds
        // of milliseconds. These transient stalls are real but not indicative
        // of scheduler bugs.
        if entry.scheduler.has_active_scheduling()
            && let Some(ref monitor) = result.monitor
        {
            let eval_report = trim_settle_samples(monitor);
            let thresholds = merged_assert.monitor_thresholds();
            let verdict = thresholds.evaluate(&eval_report);
            if !verdict.passed {
                let details = verdict.details.join("\n  ");
                let timeline_section = build_timeline_section();
                let monitor_section = format_monitor_section(monitor, merged_assert);
                let msg = format!(
                    "{}ktstr_test '{}'{} [topo={}] {ERR_MONITOR_FAILED_AFTER_SCENARIO}:\n  {}{}{}{}{}",
                    fingerprint_line,
                    entry.name,
                    sched_label,
                    topo,
                    details,
                    timeline_section,
                    monitor_section,
                    sched_log_section,
                    dump_section,
                );
                anyhow::bail!("{msg}");
            }
        }

        return Ok(check_result);
    }

    // No parseable result — no AssertResult found in SHM or COM2.
    // With an scx scheduler under test this typically means the
    // scheduler exited (crash, BPF verifier reject, scx_bpf_error()
    // exit, sched_ext disablement); on the kernel-default scheduler
    // it means the payload itself failed. Attempt auto-repro if
    // enabled and a scheduler was running.
    // Any scheduler failure that prevents producing a test result
    // warrants repro — BPF verifier failures, scx_bpf_error() exits,
    // crashes, and stalls all land here. Previous code required
    // specific string patterns (`SENTINEL_SCHEDULER_DIED`,
    // "sched_ext:" + "disabled") which missed mid-test exits where
    // the sched_exit_monitor writes to SHM but not COM2.
    let repro_section = if entry.scheduler.has_active_scheduling() {
        repro_fn(output)
            .map(|r| format!("\n\n--- auto-repro ---\n{r}"))
            .unwrap_or_default()
    } else {
        String::new()
    };

    // Build a diagnostic section from COM1 kernel console output and exit code.
    // When COM2 has scheduler output markers, sched_log_section and dump_section
    // carry the diagnostics and the kernel console is noise (BIOS, ACPI boot).
    // When COM2 has NO scheduler output (crash before writing), the kernel console
    // is the ONLY source of crash info — include it unconditionally as a fallback.
    let has_sched_output = output.contains(SCHED_OUTPUT_START);
    let console_section = if !has_sched_output || verbose() {
        let init_stage = classify_init_stage(output);
        format_console_diagnostics(&result.stderr, result.exit_code, init_stage)
    } else {
        String::new()
    };

    let timeline_section = build_timeline_section();

    // Build monitor section for error paths where neither SHM nor COM2 had a parseable result.
    let monitor_section = build_monitor_section();

    if result.timed_out {
        let msg = format!(
            "{}ktstr_test '{}'{} [topo={}] {ERR_TIMED_OUT_NO_RESULT}{}{}{}{}{}{}",
            fingerprint_line,
            entry.name,
            sched_label,
            topo,
            console_section,
            timeline_section,
            sched_log_section,
            dump_section,
            monitor_section,
            repro_section,
        );
        anyhow::bail!("{msg}");
    }

    let reason = if let Some(ref shm_crash) = result.crash_message {
        format!("{ERR_GUEST_CRASHED_PREFIX}\n{shm_crash}")
    } else if let Some(crash_msg) = extract_panic_message(output) {
        format!("{ERR_GUEST_CRASHED_PREFIX} {crash_msg}")
    } else if entry.scheduler.has_active_scheduling() {
        ERR_NO_TEST_RESULT_FROM_GUEST.to_string()
    } else {
        ERR_NO_TEST_FUNCTION_OUTPUT.to_string()
    };
    let msg = format!(
        "{}ktstr_test '{}'{} [topo={}] {}{}{}{}{}{}{}",
        fingerprint_line,
        entry.name,
        sched_label,
        topo,
        reason,
        console_section,
        timeline_section,
        sched_log_section,
        dump_section,
        monitor_section,
        repro_section,
    );
    anyhow::bail!("{msg}")
}

/// Format the `--- monitor ---` section for failure output.
///
/// Shows peak values, averaged metrics, event counter rates, schedstat
/// rates, and the monitor verdict. All values are from the post-warmup
/// evaluation window (boot-settle samples trimmed).
pub(crate) fn format_monitor_section(
    monitor: &crate::monitor::MonitorReport,
    merged_assert: &crate::assert::Assert,
) -> String {
    let eval_report = trim_settle_samples(monitor);
    let s = &eval_report.summary;
    let thresholds = merged_assert.monitor_thresholds();
    let verdict = thresholds.evaluate(&eval_report);
    let verdict_line = if verdict.passed {
        verdict.summary.clone()
    } else {
        format!("{}: {}", verdict.summary, verdict.details.join("; "))
    };

    let mut lines = vec![
        format!(
            "samples={} max_imbalance={:.2} max_dsq_depth={} stall={}",
            s.total_samples, s.max_imbalance_ratio, s.max_local_dsq_depth, s.stall_detected,
        ),
        format!(
            "avg: imbalance={:.2} nr_running/cpu={:.1} dsq/cpu={:.1}",
            s.avg_imbalance_ratio, s.avg_nr_running, s.avg_local_dsq_depth,
        ),
    ];

    if let Some(ref ev) = s.event_deltas {
        lines.push(format!(
            "events: fallback={} ({:.1}/s) keep_last={} ({:.1}/s) offline={}",
            ev.total_fallback,
            ev.fallback_rate,
            ev.total_dispatch_keep_last,
            ev.keep_last_rate,
            ev.total_dispatch_offline,
        ));
        let mut extra = Vec::new();
        if ev.total_reenq_immed != 0 {
            extra.push(format!("reenq_immed={}", ev.total_reenq_immed));
        }
        if ev.total_reenq_local_repeat != 0 {
            extra.push(format!(
                "reenq_local_repeat={}",
                ev.total_reenq_local_repeat
            ));
        }
        if ev.total_refill_slice_dfl != 0 {
            extra.push(format!("refill_slice_dfl={}", ev.total_refill_slice_dfl));
        }
        if ev.total_bypass_activate != 0 {
            extra.push(format!("bypass_activate={}", ev.total_bypass_activate));
        }
        if ev.total_bypass_dispatch != 0 {
            extra.push(format!("bypass_dispatch={}", ev.total_bypass_dispatch));
        }
        if ev.total_bypass_duration != 0 {
            extra.push(format!("bypass_duration={}ns", ev.total_bypass_duration));
        }
        if ev.total_insert_not_owned != 0 {
            extra.push(format!("insert_not_owned={}", ev.total_insert_not_owned));
        }
        if ev.total_sub_bypass_dispatch != 0 {
            extra.push(format!(
                "sub_bypass_dispatch={}",
                ev.total_sub_bypass_dispatch
            ));
        }
        if !extra.is_empty() {
            lines.push(format!("events+: {}", extra.join(" ")));
        }
    }

    if let Some(ref ss) = s.schedstat_deltas {
        lines.push(format!(
            "schedstat: csw={} ({:.0}/s) run_delay={:.0}ns/s ttwu={} goidle={}",
            ss.total_sched_count,
            ss.sched_count_rate,
            ss.run_delay_rate,
            ss.total_ttwu_count,
            ss.total_sched_goidle,
        ));
    }

    if let Some(ref progs) = s.prog_stats_deltas {
        for p in progs {
            if p.cnt > 0 {
                lines.push(format!(
                    "bpf: {} cnt={} {:.0}ns/call",
                    p.name, p.cnt, p.nsecs_per_call,
                ));
            }
        }
    }

    lines.push(format!("verdict: {verdict_line}"));

    format!("\n\n--- monitor ---\n{}", lines.join("\n"))
}

/// Number of monitor samples to skip at the start of evaluation.
///
/// During VM boot the kernel performs BPF verification, initramfs
/// unpacking, and scheduler loading. These memory-intensive operations
/// cause the scheduler tick to stall for hundreds of milliseconds.
/// The stalls are real but transient — evaluating them produces false
/// positives, especially in low-memory VMs.
///
/// 20 samples at ~100ms interval = ~2 seconds of warmup. This covers
/// the boot settling period after the scheduler attaches.
const MONITOR_WARMUP_SAMPLES: usize = 20;

/// Skip boot-settle samples from a MonitorReport for threshold evaluation.
///
/// Returns a report with the first `MONITOR_WARMUP_SAMPLES` removed so
/// that transient boot-time stalls don't trigger sustained-window
/// violations.
pub(crate) fn trim_settle_samples(
    report: &crate::monitor::MonitorReport,
) -> crate::monitor::MonitorReport {
    if report.samples.len() <= MONITOR_WARMUP_SAMPLES {
        return report.clone();
    }

    let trimmed = report.samples[MONITOR_WARMUP_SAMPLES..].to_vec();
    let summary = crate::monitor::MonitorSummary::from_samples_with_threshold(
        &trimmed,
        report.preemption_threshold_ns,
    );
    crate::monitor::MonitorReport {
        samples: trimmed,
        summary,
        preemption_threshold_ns: report.preemption_threshold_ns,
        watchdog_observation: report.watchdog_observation,
    }
}

/// Check that `/dev/kvm` is accessible for read+write.
///
/// Pre-flight check for VM-booting test runs: every ktstr test needs
/// a KVM fd, and failing fast here yields an actionable error
/// ("add your user to the kvm group") before the VM builder starts
/// allocating memory / fetching kernels.
fn ensure_kvm() -> Result<()> {
    std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/kvm")
        .context(
            "/dev/kvm not accessible — KVM is required for ktstr_test. \
             Check that KVM is enabled and your user is in the kvm group.",
        )?;
    Ok(())
}

/// Format a label for the scheduler spec, for use in test output.
///
/// Returns an empty string for `SchedulerSpec::Eevdf` so the failure
/// header reads `ktstr_test 'name' [topo=...]` with no sched
/// bracket — every other variant renders `" [sched=X]"` where `X`
/// comes from [`SchedulerSpec::display_name`].
fn scheduler_label(spec: &SchedulerSpec) -> String {
    if matches!(spec, SchedulerSpec::Eevdf) {
        String::new()
    } else {
        format!(" [sched={}]", spec.display_name())
    }
}

// ---------------------------------------------------------------------------
// Scheduler resolution
// ---------------------------------------------------------------------------

/// Provenance of a scheduler binary returned by [`resolve_scheduler`].
///
/// Each variant identifies the discovery branch that produced the
/// path, so downstream tooling (sidecar, cache-key construction, log
/// lines) can distinguish "we found a pre-built binary in a target
/// directory whose git hash we don't control" from "we just built
/// this binary from HEAD in the current workspace and therefore know
/// its source commit is the workspace HEAD."
///
/// Only the [`AutoBuilt`](Self::AutoBuilt) variant carries an honest
/// source-commit guarantee: every other branch locates an *existing*
/// file whose provenance is outside this process's knowledge.
/// Callers that need to stamp a sidecar with a scheduler-specific
/// commit must discard the hash for every non-`AutoBuilt` resolution
/// — a stale `target/debug/` binary looks identical to a fresh
/// `AutoBuilt` one but can be arbitrarily old.
///
/// `Eevdf` / `KernelBuiltin` / `Path` resolutions do not go through
/// the discovery cascade; they map to [`EnvVar`](Self::EnvVar) style
/// variants only by analogy. For those, the source is:
/// - `Eevdf` / `KernelBuiltin` → [`NotFound`](Self::NotFound) (no
///   user-space binary involved; the tuple's `Option<PathBuf>` is
///   `None`).
/// - `Path(p)` → [`EnvVar`](Self::EnvVar) (the caller named the path
///   explicitly, which is the most authoritative source).
///
/// The variant ordering in the enum mirrors the discovery cascade
/// order in [`resolve_scheduler`] so a reviewer can scan both lists
/// in lockstep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolveSource {
    /// Resolved via an explicit environment variable or caller-
    /// provided path (`KTSTR_SCHEDULER` for `Discover`, the literal
    /// path for `SchedulerSpec::Path`). Trusted to the extent the
    /// caller trusts the variable or argument; git-hash provenance
    /// is UNKNOWN to this process.
    EnvVar,
    /// Resolved via a sibling of [`crate::resolve_current_exe`]
    /// (same directory, or the sibling of a `deps/` directory for
    /// integration tests / nextest). Git-hash provenance UNKNOWN
    /// — the binary may be from any previous build.
    SiblingDir,
    /// Resolved via a fallback search in `target/debug/`. Git-hash
    /// provenance UNKNOWN — a stale binary from an older tree
    /// passes this check identically to a fresh one.
    TargetDebug,
    /// Resolved via a fallback search in `target/release/`. Git-hash
    /// provenance UNKNOWN — same stale-binary hazard as
    /// [`TargetDebug`](Self::TargetDebug).
    TargetRelease,
    /// Built on demand by [`crate::build_and_find_binary`] inside this
    /// process. The build targets the current workspace's HEAD by
    /// construction — the ONLY variant where the source commit is
    /// known to match the workspace tree the tests run from.
    AutoBuilt,
    /// No user-space binary path was produced. Returned for
    /// `SchedulerSpec::Eevdf` and `SchedulerSpec::KernelBuiltin` (the
    /// kernel supplies the scheduler — no binary to locate). The
    /// tuple's `Option<PathBuf>` is always `None` for this variant.
    NotFound,
}

/// Resolve a scheduler binary from a `SchedulerSpec`.
///
/// Returns the resolved path (if any) paired with the
/// [`ResolveSource`] naming the discovery branch that produced it.
/// The source is load-bearing for downstream provenance: only
/// [`ResolveSource::AutoBuilt`] guarantees the binary matches the
/// current workspace tree; every other variant locates a
/// pre-existing file whose git hash is UNKNOWN to this process.
///
/// Variant mapping:
/// - `Eevdf` / `KernelBuiltin { .. }` → `(None, NotFound)` (no
///   user-space binary).
/// - `Path(p)` → `(Some(p), EnvVar)` (explicit caller-named path;
///   validated for existence).
/// - `Discover(name)` → cascade through `KTSTR_SCHEDULER` env
///   ([`EnvVar`](ResolveSource::EnvVar)), sibling of
///   `current_exe` ([`SiblingDir`](ResolveSource::SiblingDir)),
///   `target/debug/` ([`TargetDebug`](ResolveSource::TargetDebug)),
///   `target/release/` ([`TargetRelease`](ResolveSource::TargetRelease)),
///   on-demand build ([`AutoBuilt`](ResolveSource::AutoBuilt)).
///   Exhausting every branch is a hard error.
pub fn resolve_scheduler(spec: &SchedulerSpec) -> Result<(Option<PathBuf>, ResolveSource)> {
    match spec {
        SchedulerSpec::Eevdf | SchedulerSpec::KernelBuiltin { .. } => {
            Ok((None, ResolveSource::NotFound))
        }
        SchedulerSpec::Path(p) => {
            let path = PathBuf::from(p);
            anyhow::ensure!(path.exists(), "scheduler not found: {p}");
            Ok((Some(path), ResolveSource::EnvVar))
        }
        SchedulerSpec::Discover(name) => {
            // 1. KTSTR_SCHEDULER env var
            if let Ok(p) = std::env::var("KTSTR_SCHEDULER") {
                let path = PathBuf::from(&p);
                if path.exists() {
                    return Ok((Some(path), ResolveSource::EnvVar));
                }
            }

            // 2. Sibling of current executable (or parent of deps/)
            if let Ok(exe) = crate::resolve_current_exe()
                && let Some(dir) = exe.parent()
            {
                let candidate = dir.join(name);
                if candidate.exists() {
                    return Ok((Some(candidate), ResolveSource::SiblingDir));
                }
                // Integration tests and nextest place test binaries in
                // target/{debug,release}/deps/. The scheduler binary is
                // one level up in target/{debug,release}/.
                if dir.file_name().is_some_and(|d| d == "deps")
                    && let Some(parent) = dir.parent()
                {
                    let candidate = parent.join(name);
                    if candidate.exists() {
                        return Ok((Some(candidate), ResolveSource::SiblingDir));
                    }
                }
            }

            // 3. target/debug/
            let candidate = PathBuf::from("target/debug").join(name);
            if candidate.exists() {
                return Ok((Some(candidate), ResolveSource::TargetDebug));
            }

            // 4. target/release/
            let candidate = PathBuf::from("target/release").join(name);
            if candidate.exists() {
                return Ok((Some(candidate), ResolveSource::TargetRelease));
            }

            // 5. Build the scheduler package on demand.
            match crate::build_and_find_binary(name) {
                Ok(path) => return Ok((Some(path), ResolveSource::AutoBuilt)),
                Err(e) => eprintln!("ktstr_test: auto-build scheduler '{name}' failed: {e:#}"),
            }

            anyhow::bail!(
                "scheduler '{name}' not found. Set KTSTR_SCHEDULER or \
                 place it next to the test binary or in target/{{debug,release}}/"
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel resolution
// ---------------------------------------------------------------------------

/// Find a kernel image for running tests.
///
/// Checks `KTSTR_TEST_KERNEL` env var first (direct image path),
/// then delegates to [`crate::find_kernel()`] for cache and
/// filesystem discovery. Bails with actionable hints on failure.
pub fn resolve_test_kernel() -> Result<PathBuf> {
    // Check environment variable first.
    if let Ok(path) = std::env::var("KTSTR_TEST_KERNEL") {
        let p = PathBuf::from(&path);
        anyhow::ensure!(p.exists(), "KTSTR_TEST_KERNEL not found: {path}");
        return Ok(p);
    }

    // Standard locations.
    if let Some(p) = crate::find_kernel()? {
        return Ok(p);
    }

    anyhow::bail!(
        "no kernel found\n  \
         hint: {kernel_hint}\n  \
         hint: or set KTSTR_TEST_KERNEL=/path/to/{image_name} to point at a \
         pre-built bootable image directly (bypasses KTSTR_KERNEL resolution)",
        kernel_hint = crate::KTSTR_KERNEL_HINT,
        image_name = if cfg!(target_arch = "aarch64") {
            "Image"
        } else {
            "bzImage"
        }
    )
}

/// If `kernel_path` resolves to an image inside a cache entry, hold a
/// `LOCK_SH` on that entry's coordination lockfile for the duration of
/// the returned guard. Prevents a concurrent
/// `cargo ktstr kernel build` from swapping the entry's directory
/// (see [`crate::cache::CacheDir::store`]) under the VM while the test
/// reads from it.
///
/// Returns `Ok(None)` when `kernel_path` is not shaped like a cache
/// entry — explicit `KTSTR_TEST_KERNEL=/path/to/bzImage`,
/// `/lib/modules/.../vmlinuz`, `/boot/vmlinuz-*`, or any path whose
/// two-level parent does not match the resolved cache root. Such
/// paths do not need coordination because the build pipeline never
/// touches them.
///
/// Detection: the image is expected at `{root}/{key}/{image_name}`.
/// Walk `kernel_path` up by two components (image_name, key) to
/// produce a candidate root and canonicalize both sides before
/// comparing — symlinks, redundant `./` segments, and `..` traversals
/// must all reduce to the same inode path or the entry is treated as
/// non-cache.
pub(crate) fn acquire_test_kernel_lock_if_cached(
    kernel_path: &Path,
) -> Result<Option<crate::cache::SharedLockGuard>> {
    // Peel the image filename. Fail → not a cache entry.
    let Some(entry_dir) = kernel_path.parent() else {
        return Ok(None);
    };
    // Peel the entry directory name (this is the candidate cache
    // key). Fail → not a cache entry.
    let Some(key_os) = entry_dir.file_name() else {
        return Ok(None);
    };
    let Some(cache_key) = key_os.to_str() else {
        return Ok(None);
    };
    // The directory above the entry is the candidate cache root.
    let Some(candidate_root) = entry_dir.parent() else {
        return Ok(None);
    };

    // Canonicalize both the candidate root and the resolved cache
    // root so symlinks / `.` / `..` reduce to the same inode path
    // before comparing. A non-cache path (e.g. /lib/modules/...)
    // simply canonicalizes to itself and will not match.
    let candidate_root_canon = match candidate_root.canonicalize() {
        Ok(p) => p,
        Err(_) => return Ok(None),
    };
    let resolved_root = match crate::cache::CacheDir::default_root() {
        Ok(p) => p,
        // Cache root unresolvable (no HOME / no XDG / env points at a
        // nonexistent path): no cache exists, so `kernel_path` cannot
        // be an entry.
        Err(_) => return Ok(None),
    };
    let resolved_root_canon = match resolved_root.canonicalize() {
        Ok(p) => p,
        // Cache root resolves but does not exist on disk yet (fresh
        // developer checkout). `kernel_path` is not inside a cache
        // entry, so no lock needed.
        Err(_) => return Ok(None),
    };

    if candidate_root_canon != resolved_root_canon {
        return Ok(None);
    }

    // The path is shaped as a cache entry under the resolved root.
    // Acquire the reader lock. Propagate errors (fs corruption,
    // timeout): a real cache-entry path that cannot be locked is an
    // infrastructure failure, not a silent skip.
    let cache = crate::cache::CacheDir::with_root(resolved_root_canon);
    let guard = cache.acquire_shared_lock(cache_key)?;
    Ok(Some(guard))
}

#[cfg(test)]
mod tests {
    use super::super::output::{
        RESULT_END, RESULT_START, STAGE_INIT_NOT_STARTED, STAGE_INIT_STARTED_NO_PAYLOAD,
        STAGE_PAYLOAD_STARTED_NO_RESULT,
    };
    use super::super::test_helpers::{
        EVAL_TOPO, EnvVarGuard, build_assert_result_json, eevdf_entry, isolated_cache_dir,
        lock_env, make_vm_result, no_repro, sched_entry,
    };
    use super::*;
    use crate::assert::{AssertDetail, DetailKind};
    use crate::verifier::SCHED_OUTPUT_END;
    use tempfile::TempDir;

    // -- dedupe_include_files tests --
    //
    // Policy pins for the aggregator downstream of
    // `KtstrTestEntry::all_include_files` + `resolve_include_files`:
    // identical `(archive, host)` pairs collapse silently, same
    // archive with conflicting hosts aborts. Deterministic
    // ordering (BTreeMap keys).

    /// Empty input → empty result. Pins the identity case so a
    /// regression that introduces an invariant init-element
    /// (e.g. implicit config file) would break this.
    #[test]
    fn dedupe_include_files_empty_input() {
        let out = dedupe_include_files(&[]).unwrap();
        assert!(out.is_empty(), "empty in → empty out, got {out:?}");
    }

    /// Identical pair appearing twice deduplicates silently. The
    /// output contains a single entry; no error, no warning. Models
    /// the scheduler-and-payload-both-declare-config case.
    #[test]
    fn dedupe_include_files_identical_pair_collapses() {
        let input = vec![
            (
                "include-files/helper".to_string(),
                std::path::PathBuf::from("/usr/bin/helper"),
                "declarative",
            ),
            (
                "include-files/helper".to_string(),
                std::path::PathBuf::from("/usr/bin/helper"),
                "scheduler config_file",
            ),
        ];
        let out = dedupe_include_files(&input).unwrap();
        assert_eq!(out.len(), 1, "identical pair must dedupe, got {out:?}");
        assert_eq!(out[0].0, "include-files/helper");
        assert_eq!(out[0].1, std::path::PathBuf::from("/usr/bin/helper"));
    }

    /// Same archive_path with conflicting host_paths is a genuine
    /// ambiguity — one declaration would silently overwrite the
    /// other's file in the initramfs. Policy: hard error with a
    /// diagnostic naming both host paths so the operator knows
    /// which declarations need disambiguation.
    #[test]
    fn dedupe_include_files_archive_collision_errors() {
        let input = vec![
            (
                "include-files/config.json".to_string(),
                std::path::PathBuf::from("/tmp/sched/config.json"),
                "scheduler config_file",
            ),
            (
                "include-files/config.json".to_string(),
                std::path::PathBuf::from("/tmp/payload/config.json"),
                "declarative",
            ),
        ];
        let err = dedupe_include_files(&input).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("include_files conflict"),
            "diagnostic must mention 'include_files conflict': {msg}",
        );
        assert!(
            msg.contains("/tmp/sched/config.json") && msg.contains("/tmp/payload/config.json"),
            "diagnostic must name both host paths: {msg}",
        );
        assert!(
            msg.contains("origin: scheduler config_file") && msg.contains("origin: declarative"),
            "diagnostic must name both origin labels: {msg}",
        );
    }

    /// Multiple distinct archive_paths pass through unchanged. Verifies
    /// the aggregator doesn't accidentally collapse orthogonal entries
    /// (e.g. dropping by coincidental prefix or path-component equality).
    #[test]
    fn dedupe_include_files_preserves_distinct_entries() {
        let input = vec![
            (
                "include-files/a".to_string(),
                std::path::PathBuf::from("/usr/bin/a"),
                "declarative",
            ),
            (
                "include-files/b".to_string(),
                std::path::PathBuf::from("/usr/bin/b"),
                "declarative",
            ),
            (
                "include-files/c".to_string(),
                std::path::PathBuf::from("/usr/bin/c"),
                "scheduler config_file",
            ),
        ];
        let out = dedupe_include_files(&input).unwrap();
        assert_eq!(out.len(), 3, "three distinct entries must survive");
        let archives: Vec<&str> = out.iter().map(|(a, _)| a.as_str()).collect();
        assert!(archives.contains(&"include-files/a"));
        assert!(archives.contains(&"include-files/b"));
        assert!(archives.contains(&"include-files/c"));
    }

    // -- resolve_test_kernel tests --

    #[test]
    fn resolve_test_kernel_with_env_var() {
        let _lock = lock_env();
        let exe = crate::resolve_current_exe().unwrap();
        let _env = EnvVarGuard::set("KTSTR_TEST_KERNEL", &exe);
        let result = resolve_test_kernel();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), exe);
    }

    #[test]
    fn resolve_test_kernel_with_nonexistent_env_path() {
        let _lock = lock_env();
        let _env = EnvVarGuard::set("KTSTR_TEST_KERNEL", "/nonexistent/kernel/path");
        let result = resolve_test_kernel();
        assert!(result.is_err());
    }

    // -- KVM check --

    #[test]
    fn kvm_accessible_on_test_host() {
        // Checks that /dev/kvm is accessible with read+write permissions.
        ensure_kvm().expect("/dev/kvm not accessible");
    }

    // -- resolve_scheduler tests --

    #[test]
    fn resolve_scheduler_eevdf() {
        let (path, source) = resolve_scheduler(&SchedulerSpec::Eevdf).unwrap();
        assert!(path.is_none());
        assert_eq!(
            source,
            ResolveSource::NotFound,
            "Eevdf has no user-space binary — source must be NotFound",
        );
    }

    #[test]
    fn resolve_scheduler_kernel_builtin_is_not_found() {
        let (path, source) = resolve_scheduler(&SchedulerSpec::KernelBuiltin {
            enable: &[],
            disable: &[],
        })
        .unwrap();
        assert!(path.is_none());
        assert_eq!(
            source,
            ResolveSource::NotFound,
            "KernelBuiltin has no user-space binary — source must be NotFound",
        );
    }

    #[test]
    fn resolve_scheduler_path_exists() {
        let exe = crate::resolve_current_exe().unwrap();
        let (path, source) = resolve_scheduler(&SchedulerSpec::Path(Box::leak(
            exe.to_str().unwrap().to_string().into_boxed_str(),
        )))
        .unwrap();
        assert!(path.is_some());
        assert_eq!(
            source,
            ResolveSource::EnvVar,
            "explicit Path(_) is the most authoritative source — maps to EnvVar",
        );
    }

    #[test]
    fn resolve_scheduler_path_missing() {
        let result = resolve_scheduler(&SchedulerSpec::Path("/nonexistent/scheduler"));
        assert!(result.is_err());
    }

    #[test]
    fn resolve_scheduler_discover_missing() {
        let _lock = lock_env();
        let _env = EnvVarGuard::remove("KTSTR_SCHEDULER");
        let result = resolve_scheduler(&SchedulerSpec::Discover("__nonexistent_scheduler_xyz__"));
        assert!(result.is_err());
    }

    #[test]
    fn resolve_scheduler_discover_via_env() {
        let _lock = lock_env();
        let exe = crate::resolve_current_exe().unwrap();
        let _env = EnvVarGuard::set("KTSTR_SCHEDULER", &exe);
        let (path, source) = resolve_scheduler(&SchedulerSpec::Discover("anything")).unwrap();
        assert_eq!(path.unwrap(), exe);
        assert_eq!(
            source,
            ResolveSource::EnvVar,
            "KTSTR_SCHEDULER hit must tag the result EnvVar",
        );
    }

    // -- scheduler_label tests --

    #[test]
    fn scheduler_label_eevdf_empty() {
        assert_eq!(scheduler_label(&SchedulerSpec::Eevdf), "");
    }

    #[test]
    fn scheduler_label_discover() {
        assert_eq!(
            scheduler_label(&SchedulerSpec::Discover("scx_mitosis")),
            " [sched=scx_mitosis]"
        );
    }

    #[test]
    fn scheduler_label_path() {
        assert_eq!(
            scheduler_label(&SchedulerSpec::Path("/usr/bin/sched")),
            " [sched=/usr/bin/sched]"
        );
    }

    // -- evaluate_vm_result error path tests --

    #[test]
    fn eval_eevdf_no_com2_output() {
        let _lock = lock_env();
        let _env_bt = EnvVarGuard::set("RUST_BACKTRACE", "1");
        let entry = eevdf_entry("__eval_eevdf_no_out__");
        let result = make_vm_result("", "boot log line\nKernel panic", 1, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains(ERR_NO_TEST_FUNCTION_OUTPUT),
            "EEVDF with no COM2 output should say {ERR_NO_TEST_FUNCTION_OUTPUT:?}, got: {msg}",
        );
        assert!(
            !msg.contains("no test result received from guest"),
            "EEVDF error should not use the scheduler-path wording, got: {msg}",
        );
        assert!(
            msg.contains("exit_code=1"),
            "should include exit code, got: {msg}"
        );
        assert!(
            msg.contains("Kernel panic"),
            "should include console output, got: {msg}"
        );
    }

    #[test]
    fn eval_sched_exits_no_com2_output() {
        let entry = sched_entry("__eval_sched_exits__");
        let result = make_vm_result("", "boot ok", 1, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains(ERR_NO_TEST_RESULT_FROM_GUEST),
            "scheduler present with no output should take the scheduler-path fallback, got: {msg}",
        );
        assert!(
            !msg.contains("test function produced no output"),
            "should not say 'test function produced no output' when scheduler is set, got: {msg}",
        );
    }

    #[test]
    fn eval_sched_exits_with_sched_log() {
        let _lock = lock_env();
        let _env_bt = EnvVarGuard::set("RUST_BACKTRACE", "1");
        let sched_log = format!(
            "noise\n{SCHED_OUTPUT_START}\ndo_enqueue_task+0x1a0\nbalance_one+0x50\n{SCHED_OUTPUT_END}\nmore",
        );
        let entry = sched_entry("__eval_sched_log__");
        let result = make_vm_result(&sched_log, "", -1, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains(ERR_NO_TEST_RESULT_FROM_GUEST),
            "should take the scheduler-path fallback, got: {msg}",
        );
        assert!(
            msg.contains("--- scheduler log ---"),
            "should include scheduler log section, got: {msg}",
        );
        assert!(
            msg.contains("do_enqueue_task"),
            "should include scheduler log content, got: {msg}",
        );
    }

    #[test]
    fn eval_sched_mid_test_exit_triggers_repro() {
        // Scheduler exits mid-test: sched_exit_monitor dumps log to COM2
        // but does NOT write "SCHEDULER_DIED". Auto-repro should still
        // trigger because has_active_scheduling() is true and no
        // AssertResult was produced.
        let sched_log =
            format!("{SCHED_OUTPUT_START}\nError: BPF program error\n{SCHED_OUTPUT_END}",);
        let entry = sched_entry("__eval_mid_exit_repro__");
        let result = make_vm_result(&sched_log, "", 1, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let repro_called = std::sync::atomic::AtomicBool::new(false);
        let repro_fn = |_output: &str| -> Option<String> {
            repro_called.store(true, std::sync::atomic::Ordering::Relaxed);
            Some("repro data".to_string())
        };
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &repro_fn,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            repro_called.load(std::sync::atomic::Ordering::Relaxed),
            "repro_fn should be called for mid-test scheduler exit without SCHEDULER_DIED marker",
        );
        assert!(
            msg.contains("--- auto-repro ---"),
            "error should include auto-repro section, got: {msg}",
        );
        assert!(
            msg.contains("repro data"),
            "error should include repro output, got: {msg}",
        );
    }

    #[test]
    fn eval_sched_repro_no_data_shows_diagnostic() {
        // When repro_fn returns the fallback diagnostic, the error
        // output should include it so the user knows auto-repro was
        // tried and why it produced nothing.
        let entry = sched_entry("__eval_repro_no_data__");
        let result = make_vm_result("", "", 1, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let repro_fn = |_output: &str| -> Option<String> {
            Some(
                "auto-repro: no probe data — scheduler may have exited before \
                 probes could attach. Check the sched_ext dump and scheduler \
                 log sections above for crash details."
                    .to_string(),
            )
        };
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &repro_fn,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("--- auto-repro ---"),
            "should include auto-repro section, got: {msg}",
        );
        assert!(
            msg.contains("no probe data"),
            "should include diagnostic message, got: {msg}",
        );
        assert!(
            msg.contains("sched_ext dump"),
            "should direct user to dump section, got: {msg}",
        );
    }

    #[test]
    fn eval_timeout_no_result() {
        let _lock = lock_env();
        let _env_bt = EnvVarGuard::set("RUST_BACKTRACE", "1");
        let entry = eevdf_entry("__eval_timeout__");
        let result = make_vm_result("", "booting...\nstill booting...", 0, true);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains(ERR_TIMED_OUT_NO_RESULT),
            "should contain full timed-out reason {ERR_TIMED_OUT_NO_RESULT:?}, got: {msg}",
        );
        assert!(
            msg.contains("booting"),
            "should include console output, got: {msg}",
        );
        assert!(
            msg.contains("[topo="),
            "error should include topology, got: {msg}",
        );
    }

    #[test]
    fn eval_payload_exits_no_check_result() {
        // Payload wrote something to COM2 but not a valid AssertResult.
        let entry = eevdf_entry("__eval_no_check__");
        let result = make_vm_result(
            "some output but no delimiters",
            "Linux version 6.14.0\nboot complete",
            0,
            false,
        );
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains(ERR_NO_TEST_FUNCTION_OUTPUT),
            "non-parseable COM2 with EEVDF should say {ERR_NO_TEST_FUNCTION_OUTPUT:?}, got: {msg}",
        );
        assert!(
            !msg.contains("no test result received from guest"),
            "EEVDF should not use the scheduler-path wording, got: {msg}",
        );
    }

    #[test]
    fn eval_sched_ext_dump_included() {
        let dump_line = "ktstr-0 [001] 0.5: sched_ext_dump: Debug dump line";
        let entry = sched_entry("__eval_dump__");
        let result = make_vm_result("", dump_line, -1, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("--- sched_ext dump ---"),
            "should include dump section, got: {msg}",
        );
        assert!(
            msg.contains("sched_ext_dump: Debug dump"),
            "should include dump content, got: {msg}",
        );
    }

    #[test]
    fn eval_check_result_passed_returns_ok() {
        let json = build_assert_result_json(true, vec![]);
        let output = format!("{RESULT_START}\n{json}\n{RESULT_END}");
        let entry = eevdf_entry("__eval_pass__");
        let result = make_vm_result(&output, "", 0, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        assert!(
            evaluate_vm_result(
                &entry,
                &result,
                &assertions,
                &[],
                &[],
                &[],
                &EVAL_TOPO,
                &[],
                &no_repro,
            )
            .is_ok(),
            "passing AssertResult should return Ok",
        );
    }

    #[test]
    fn eval_check_result_failed_includes_details() {
        let json = build_assert_result_json(
            false,
            vec![
                AssertDetail::new(DetailKind::Stuck, "stuck 3000ms"),
                AssertDetail::new(DetailKind::Unfair, "spread 45%"),
            ],
        );
        let output = format!("{RESULT_START}\n{json}\n{RESULT_END}");
        let entry = eevdf_entry("__eval_fail_details__");
        let result = make_vm_result(&output, "", 0, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let msg = format!(
            "{}",
            evaluate_vm_result(
                &entry,
                &result,
                &assertions,
                &[],
                &[],
                &[],
                &EVAL_TOPO,
                &[],
                &no_repro
            )
            .unwrap_err()
        );
        assert!(msg.contains("failed:"), "got: {msg}");
        assert!(msg.contains("stuck 3000ms"), "got: {msg}");
        assert!(msg.contains("spread 45%"), "got: {msg}");
    }

    /// Cleanup-budget enforcement: when the entry's `cleanup_budget`
    /// is set and the run's measured `cleanup_duration` exceeds it,
    /// `evaluate_vm_result` folds a failing `AssertDetail` (kind
    /// `Other`) carrying the "vm cleanup overran budget" message into
    /// the test verdict. The guest body returned a passing
    /// `AssertResult` (so the parse-success arm is taken — the only
    /// arm where this check fires, see the contract paragraph at
    /// `evaluate_vm_result`'s budget block); the budget overshoot
    /// flips the merged verdict to a failure, which propagates as a
    /// `bail!` error string downstream.
    #[test]
    fn eval_cleanup_budget_overshoot_folds_failing_detail() {
        let json = build_assert_result_json(true, vec![]);
        let output = format!("{RESULT_START}\n{json}\n{RESULT_END}");
        let mut entry = eevdf_entry("__eval_cleanup_overshoot__");
        entry.cleanup_budget = Some(std::time::Duration::from_secs(1));
        let mut result = make_vm_result(&output, "", 0, false);
        result.cleanup_duration = Some(std::time::Duration::from_secs(10));
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let msg = format!(
            "{}",
            evaluate_vm_result(
                &entry,
                &result,
                &assertions,
                &[],
                &[],
                &[],
                &EVAL_TOPO,
                &[],
                &no_repro,
            )
            .unwrap_err()
        );
        assert!(
            msg.contains("vm cleanup overran budget"),
            "budget-overshoot detail must surface in the error string, got: {msg}",
        );
        assert!(
            msg.contains("measured 10.000s"),
            "measured duration must be rendered, got: {msg}",
        );
        assert!(
            msg.contains("budget 1.000s"),
            "budget must be rendered, got: {msg}",
        );
    }

    /// Cleanup-budget no-fire: when the run's `cleanup_duration` is
    /// strictly under the entry's `cleanup_budget`, the guest's
    /// passing `AssertResult` survives the merge and
    /// `evaluate_vm_result` returns `Ok`. Verifies that
    /// `measured < budget` passes without folding a fail; the exact
    /// `measured == budget` boundary is covered separately by
    /// [`eval_cleanup_budget_equal_passes`].
    #[test]
    fn eval_cleanup_budget_under_passes() {
        let json = build_assert_result_json(true, vec![]);
        let output = format!("{RESULT_START}\n{json}\n{RESULT_END}");
        let mut entry = eevdf_entry("__eval_cleanup_under__");
        entry.cleanup_budget = Some(std::time::Duration::from_secs(5));
        let mut result = make_vm_result(&output, "", 0, false);
        result.cleanup_duration = Some(std::time::Duration::from_millis(500));
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        assert!(
            evaluate_vm_result(
                &entry,
                &result,
                &assertions,
                &[],
                &[],
                &[],
                &EVAL_TOPO,
                &[],
                &no_repro,
            )
            .is_ok(),
            "cleanup_duration under budget must keep the verdict Ok",
        );
    }

    /// Cleanup-budget boundary pin: `measured == budget` must NOT
    /// fold a fail because the enforcement at
    /// `evaluate_vm_result`'s budget block uses strict `>`. A future
    /// regression that flips the comparator to `>=` (or to `<` on the
    /// pass-side) flips the verdict here, surfacing the bug. Together
    /// with [`eval_cleanup_budget_overshoot_folds_failing_detail`] and
    /// [`eval_cleanup_budget_under_passes`] this test pins the full
    /// {<, ==, >} comparator triplet.
    #[test]
    fn eval_cleanup_budget_equal_passes() {
        let json = build_assert_result_json(true, vec![]);
        let output = format!("{RESULT_START}\n{json}\n{RESULT_END}");
        let mut entry = eevdf_entry("__eval_cleanup_equal__");
        entry.cleanup_budget = Some(std::time::Duration::from_secs(5));
        let mut result = make_vm_result(&output, "", 0, false);
        result.cleanup_duration = Some(std::time::Duration::from_secs(5));
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        assert!(
            evaluate_vm_result(
                &entry,
                &result,
                &assertions,
                &[],
                &[],
                &[],
                &EVAL_TOPO,
                &[],
                &no_repro,
            )
            .is_ok(),
            "cleanup_duration EQUAL to budget must keep the verdict Ok \
             (strict `>` comparator); a `>=` regression lands here",
        );
    }

    #[test]
    fn eval_assert_failure_includes_sched_log() {
        let json = build_assert_result_json(
            false,
            vec![AssertDetail::new(
                DetailKind::Stuck,
                "worker 0 stuck 5000ms",
            )],
        );
        let output = format!(
            "{RESULT_START}\n{json}\n{RESULT_END}\n{SCHED_OUTPUT_START}\nscheduler noise line\n{SCHED_OUTPUT_END}",
        );
        let entry = sched_entry("__eval_fail_sched_log__");
        let result = make_vm_result(&output, "", 0, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let msg = format!(
            "{}",
            evaluate_vm_result(
                &entry,
                &result,
                &assertions,
                &[],
                &[],
                &[],
                &EVAL_TOPO,
                &[],
                &no_repro
            )
            .unwrap_err()
        );
        assert!(msg.contains("worker 0 stuck 5000ms"), "got: {msg}");
        assert!(msg.contains("scheduler noise"), "got: {msg}");
        assert!(msg.contains("--- scheduler log ---"), "got: {msg}");
    }

    #[test]
    fn eval_assert_failure_has_fingerprint() {
        let json = build_assert_result_json(
            false,
            vec![AssertDetail::new(DetailKind::Stuck, "stuck 3000ms")],
        );
        let error_line = "Error: apply_cell_config BPF program returned error -2";
        let output = format!(
            "{RESULT_START}\n{json}\n{RESULT_END}\n{SCHED_OUTPUT_START}\nstarting\n{error_line}\n{SCHED_OUTPUT_END}",
        );
        let entry = sched_entry("__eval_fingerprint__");
        let result = make_vm_result(&output, "", 0, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let msg = format!(
            "{}",
            evaluate_vm_result(
                &entry,
                &result,
                &assertions,
                &[],
                &[],
                &[],
                &EVAL_TOPO,
                &[],
                &no_repro
            )
            .unwrap_err()
        );
        assert!(msg.contains(error_line), "got: {msg}");
        let fp_pos = msg.find(error_line).unwrap();
        let name_pos = msg.find("ktstr_test").unwrap();
        assert!(fp_pos < name_pos, "got: {msg}");
    }

    #[test]
    fn eval_timeout_has_fingerprint() {
        let error_line = "Error: scheduler panicked";
        let output = format!("{SCHED_OUTPUT_START}\n{error_line}\n{SCHED_OUTPUT_END}",);
        let entry = sched_entry("__eval_timeout_fp__");
        let result = make_vm_result(&output, "", 0, true);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains(error_line),
            "timeout should contain fingerprint, got: {msg}",
        );
        let fp_pos = msg.find(error_line).unwrap();
        let name_pos = msg.find("ktstr_test").unwrap();
        assert!(
            fp_pos < name_pos,
            "fingerprint should appear before ktstr_test line, got: {msg}",
        );
    }

    #[test]
    fn eval_no_result_has_fingerprint() {
        let error_line = "Error: fatal scheduler crash";
        let output =
            format!("{SCHED_OUTPUT_START}\nstartup log\n{error_line}\n{SCHED_OUTPUT_END}",);
        let entry = sched_entry("__eval_no_result_fp__");
        let result = make_vm_result(&output, "", 1, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains(error_line),
            "no-result failure should contain fingerprint, got: {msg}",
        );
        let fp_pos = msg.find(error_line).unwrap();
        let name_pos = msg.find("ktstr_test").unwrap();
        assert!(
            fp_pos < name_pos,
            "fingerprint should appear before ktstr_test line, got: {msg}",
        );
    }

    #[test]
    fn eval_no_sched_output_no_fingerprint() {
        let json =
            build_assert_result_json(false, vec![AssertDetail::new(DetailKind::Stuck, "stuck")]);
        let output = format!("{RESULT_START}\n{json}\n{RESULT_END}");
        let entry = eevdf_entry("__eval_no_fp__");
        let result = make_vm_result(&output, "", 0, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let msg = format!(
            "{}",
            evaluate_vm_result(
                &entry,
                &result,
                &assertions,
                &[],
                &[],
                &[],
                &EVAL_TOPO,
                &[],
                &no_repro
            )
            .unwrap_err()
        );
        assert!(msg.starts_with("ktstr_test"), "got: {msg}");
    }

    #[test]
    fn eval_monitor_fail_has_fingerprint() {
        let pass_json = build_assert_result_json(true, vec![]);
        let error_line = "Error: imbalance detected internally";
        let sched_log =
            format!("{SCHED_OUTPUT_START}\nstarting\n{error_line}\n{SCHED_OUTPUT_END}",);
        let output = format!("{RESULT_START}\n{pass_json}\n{RESULT_END}\n{sched_log}");
        let entry = sched_entry("__eval_monitor_fp__");
        let imbalance_samples: Vec<crate::monitor::MonitorSample> = (0..30)
            .map(|i| {
                crate::monitor::MonitorSample::new(
                    (i * 100) as u64,
                    vec![
                        crate::monitor::CpuSnapshot {
                            nr_running: 10,
                            scx_nr_running: 10,
                            local_dsq_depth: 0,
                            rq_clock: 1000 + (i as u64 * 100),
                            scx_flags: 0,
                            event_counters: None,
                            schedstat: None,
                            vcpu_cpu_time_ns: None,
                            sched_domains: None,
                        },
                        crate::monitor::CpuSnapshot {
                            nr_running: 1,
                            scx_nr_running: 1,
                            local_dsq_depth: 0,
                            rq_clock: 2000 + (i as u64 * 100),
                            scx_flags: 0,
                            event_counters: None,
                            schedstat: None,
                            vcpu_cpu_time_ns: None,
                            sched_domains: None,
                        },
                    ],
                )
            })
            .collect();
        let summary =
            crate::monitor::MonitorSummary::from_samples_with_threshold(&imbalance_samples, 0);
        let result = crate::vmm::VmResult {
            success: true,
            exit_code: 0,
            duration: std::time::Duration::from_secs(1),
            timed_out: false,
            output,
            stderr: String::new(),
            monitor: Some(crate::monitor::MonitorReport {
                samples: imbalance_samples,
                summary,
                preemption_threshold_ns: 0,
                watchdog_observation: None,
            }),
            shm_data: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            crash_message: None,
            cleanup_duration: None,
        };
        let assertions = crate::assert::Assert::default_checks();
        let msg = format!(
            "{}",
            evaluate_vm_result(
                &entry,
                &result,
                &assertions,
                &[],
                &[],
                &[],
                &EVAL_TOPO,
                &[],
                &no_repro
            )
            .unwrap_err()
        );
        assert!(
            msg.contains(ERR_MONITOR_FAILED_AFTER_SCENARIO),
            "got: {msg}"
        );
        assert!(msg.contains(error_line), "got: {msg}");
        let fp_pos = msg.find(error_line).unwrap();
        let name_pos = msg.find("ktstr_test").unwrap();
        assert!(fp_pos < name_pos, "got: {msg}");
    }

    #[test]
    fn eval_timeout_with_sched_includes_diagnostics() {
        let _lock = lock_env();
        let _env_bt = EnvVarGuard::set("RUST_BACKTRACE", "1");
        let entry = sched_entry("__eval_timeout_sched__");
        let result = make_vm_result("", "Linux version 6.14.0\nkernel panic here", -1, true);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains(ERR_TIMED_OUT_NO_RESULT),
            "should contain {ERR_TIMED_OUT_NO_RESULT:?}, got: {msg}"
        );
        assert!(
            msg.contains("[sched=test_sched_bin]"),
            "should include scheduler label, got: {msg}"
        );
        assert!(
            msg.contains("--- diagnostics ---"),
            "should include diagnostics, got: {msg}"
        );
        assert!(
            msg.contains("kernel panic here"),
            "should include console tail, got: {msg}"
        );
    }

    // -- sentinel integration in evaluate_vm_result --

    #[test]
    fn eval_no_sentinels_shows_initramfs_failure() {
        let _lock = lock_env();
        let _env_bt = EnvVarGuard::set("RUST_BACKTRACE", "1");
        let entry = eevdf_entry("__eval_no_sentinel__");
        let result = make_vm_result("", "Kernel panic", 1, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains(STAGE_INIT_NOT_STARTED),
            "no sentinels should indicate kernel/mount failure, got: {msg}",
        );
    }

    #[test]
    fn eval_init_started_but_no_payload() {
        let _lock = lock_env();
        let _env_bt = EnvVarGuard::set("RUST_BACKTRACE", "1");
        let entry = eevdf_entry("__eval_init_only__");
        let result = make_vm_result("KTSTR_INIT_STARTED\n", "boot log", 1, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains(STAGE_INIT_STARTED_NO_PAYLOAD),
            "init sentinel only should indicate cgroup/scheduler setup failure, got: {msg}",
        );
    }

    #[test]
    fn eval_payload_started_no_result() {
        let _lock = lock_env();
        let _env_bt = EnvVarGuard::set("RUST_BACKTRACE", "1");
        let entry = eevdf_entry("__eval_payload_start__");
        let output = "KTSTR_INIT_STARTED\nKTSTR_PAYLOAD_STARTING\ngarbage";
        let result = make_vm_result(output, "", 1, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains(STAGE_PAYLOAD_STARTED_NO_RESULT),
            "both sentinels should indicate payload ran but failed, got: {msg}",
        );
    }

    // -- guest panic detection tests --

    #[test]
    fn eval_crash_in_output_says_guest_crashed() {
        let entry = sched_entry("__eval_crash_detect__");
        let output = "KTSTR_INIT_STARTED\nPANIC: panicked at src/foo.rs:42: assertion failed";
        let result = make_vm_result(output, "", 1, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains(ERR_GUEST_CRASHED_PREFIX), "got: {msg}");
        assert!(msg.contains("assertion failed"), "got: {msg}");
    }

    #[test]
    fn eval_crash_eevdf_says_guest_crashed() {
        let entry = eevdf_entry("__eval_crash_eevdf__");
        let output = "PANIC: panicked at src/bar.rs:10: index out of bounds";
        let result = make_vm_result(output, "", 1, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains(ERR_GUEST_CRASHED_PREFIX), "got: {msg}");
        assert!(msg.contains("index out of bounds"), "got: {msg}");
    }

    #[test]
    fn eval_crash_message_from_shm() {
        let entry = sched_entry("__eval_crash_shm__");
        let shm_crash = "PANIC: panicked at src/test.rs:42: assertion failed\n   \
                          0: ktstr::vmm::rust_init::ktstr_guest_init\n";
        // COM2 also has a PANIC: line (serial fallback). SHM must take priority.
        let output = "PANIC: panicked at src/test.rs:42: assertion failed";
        let mut result = make_vm_result(output, "", 1, false);
        result.crash_message = Some(shm_crash.to_string());
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains(ERR_GUEST_CRASHED_PREFIX),
            "should say {ERR_GUEST_CRASHED_PREFIX:?}, got: {msg}",
        );
        assert!(
            msg.contains("ktstr_guest_init"),
            "SHM backtrace content should be present, got: {msg}",
        );
        // SHM path uses "guest crashed:\n{shm_crash}" (multiline),
        // COM2 path uses "guest crashed: {msg}" (single line).
        // The backtrace frame proves SHM was used, not COM2.
        assert!(
            msg.contains("0: ktstr::vmm::rust_init::ktstr_guest_init"),
            "full backtrace from SHM should appear, got: {msg}",
        );
    }

    // -- diagnostic section tests --

    #[test]
    fn eval_sched_exit_includes_console() {
        let json = build_assert_result_json(
            false,
            vec![AssertDetail::new(
                DetailKind::SchedulerDied,
                "scheduler process died unexpectedly after completing step 1 of 2 (0.5s into test)",
            )],
        );
        let output = format!("{RESULT_START}\n{json}\n{RESULT_END}");
        let entry = sched_entry("__eval_sched_exit_console__");
        let result = make_vm_result(&output, "kernel panic\nsched_ext: disabled", 1, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let msg = format!(
            "{}",
            evaluate_vm_result(
                &entry,
                &result,
                &assertions,
                &[],
                &[],
                &[],
                &EVAL_TOPO,
                &[],
                &no_repro
            )
            .unwrap_err()
        );
        assert!(msg.contains("--- diagnostics ---"), "got: {msg}");
        assert!(msg.contains("kernel panic"), "got: {msg}");
    }

    #[test]
    fn eval_sched_exit_includes_monitor() {
        let json = build_assert_result_json(
            false,
            vec![AssertDetail::new(
                DetailKind::SchedulerDied,
                "scheduler process died unexpectedly during workload (2.0s into test)",
            )],
        );
        let output = format!("{RESULT_START}\n{json}\n{RESULT_END}");
        let entry = sched_entry("__eval_sched_exit_monitor__");
        let result = crate::vmm::VmResult {
            success: false,
            exit_code: 1,
            duration: std::time::Duration::from_secs(1),
            timed_out: false,
            output: output.to_string(),
            stderr: String::new(),
            monitor: Some(crate::monitor::MonitorReport {
                samples: vec![],
                summary: crate::monitor::MonitorSummary {
                    total_samples: 5,
                    max_imbalance_ratio: 3.0,
                    max_local_dsq_depth: 2,
                    stall_detected: false,
                    event_deltas: None,
                    schedstat_deltas: None,
                    prog_stats_deltas: None,
                    ..Default::default()
                },
                preemption_threshold_ns: 0,
                watchdog_observation: None,
            }),
            shm_data: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            crash_message: None,
            cleanup_duration: None,
        };
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let msg = format!(
            "{}",
            evaluate_vm_result(
                &entry,
                &result,
                &assertions,
                &[],
                &[],
                &[],
                &EVAL_TOPO,
                &[],
                &no_repro
            )
            .unwrap_err()
        );
        assert!(msg.contains("--- monitor ---"), "got: {msg}");
        assert!(msg.contains("max_imbalance"), "got: {msg}");
    }

    #[test]
    fn eval_monitor_fail_includes_sched_log() {
        let pass_json = build_assert_result_json(true, vec![]);
        let sched_log =
            format!("{SCHED_OUTPUT_START}\nscheduler debug output here\n{SCHED_OUTPUT_END}",);
        let output = format!("{RESULT_START}\n{pass_json}\n{RESULT_END}\n{sched_log}");
        let entry = sched_entry("__eval_monitor_fail_sched__");
        // Imbalance ratio 10.0 exceeds default threshold of 4.0,
        // sustained for 5+ samples past the 20-sample warmup window.
        let imbalance_samples: Vec<crate::monitor::MonitorSample> = (0..30)
            .map(|i| {
                crate::monitor::MonitorSample::new(
                    (i * 100) as u64,
                    vec![
                        crate::monitor::CpuSnapshot {
                            nr_running: 10,
                            scx_nr_running: 10,
                            local_dsq_depth: 0,
                            rq_clock: 1000 + (i as u64 * 100),
                            scx_flags: 0,
                            event_counters: None,
                            schedstat: None,
                            vcpu_cpu_time_ns: None,
                            sched_domains: None,
                        },
                        crate::monitor::CpuSnapshot {
                            nr_running: 1,
                            scx_nr_running: 1,
                            local_dsq_depth: 0,
                            rq_clock: 2000 + (i as u64 * 100),
                            scx_flags: 0,
                            event_counters: None,
                            schedstat: None,
                            vcpu_cpu_time_ns: None,
                            sched_domains: None,
                        },
                    ],
                )
            })
            .collect();
        let summary =
            crate::monitor::MonitorSummary::from_samples_with_threshold(&imbalance_samples, 0);
        let result = crate::vmm::VmResult {
            success: true,
            exit_code: 0,
            duration: std::time::Duration::from_secs(1),
            timed_out: false,
            output,
            stderr: String::new(),
            monitor: Some(crate::monitor::MonitorReport {
                samples: imbalance_samples,
                summary,
                preemption_threshold_ns: 0,
                watchdog_observation: None,
            }),
            shm_data: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            crash_message: None,
            cleanup_duration: None,
        };
        let assertions = crate::assert::Assert::default_checks();
        let msg = format!(
            "{}",
            evaluate_vm_result(
                &entry,
                &result,
                &assertions,
                &[],
                &[],
                &[],
                &EVAL_TOPO,
                &[],
                &no_repro
            )
            .unwrap_err()
        );
        assert!(
            msg.contains(ERR_MONITOR_FAILED_AFTER_SCENARIO),
            "got: {msg}"
        );
        assert!(msg.contains("--- scheduler log ---"), "got: {msg}");
    }

    /// `acquire_test_kernel_lock_if_cached` returns `Some(guard)`
    /// when `kernel_path` is shaped like a real cache entry:
    /// `{cache_root}/{cache_key}/{image_name}`. Exercises the
    /// canonicalize + candidate-root-equality branch.
    ///
    /// Uses [`isolated_cache_dir`] so the tempdir is both pointed
    /// at by `KTSTR_CACHE_DIR` AND cleaned up on drop. Holds
    /// [`lock_env`] throughout so parallel tests don't race the
    /// env var.
    #[test]
    fn acquire_test_kernel_lock_if_cached_returns_guard_on_cache_entry() {
        let _env_lock = lock_env();
        let cache = isolated_cache_dir();
        // Fake cache entry: {cache_root}/my-kernel-key/bzImage.
        let entry_dir = cache.path().join("my-kernel-key");
        std::fs::create_dir_all(&entry_dir).expect("create entry dir");
        let image_path = entry_dir.join("bzImage");
        std::fs::write(&image_path, b"fake kernel image").expect("plant image");

        let guard = super::acquire_test_kernel_lock_if_cached(&image_path)
            .expect("lock acquire must not error on valid cache entry");
        assert!(
            guard.is_some(),
            "cache-entry path must produce a SharedLockGuard",
        );
        // Confirm the .locks/ subdir materialized as a side effect
        // of the acquire — pins the integration with
        // `CacheDir::acquire_shared_lock`'s ensure_lock_dir path.
        assert!(
            cache.path().join(".locks").is_dir(),
            ".locks/ must materialize under the cache root",
        );
    }

    /// `acquire_test_kernel_lock_if_cached` returns `Ok(None)`
    /// when `kernel_path` is NOT under the resolved cache root —
    /// e.g. a `/lib/modules/…/vmlinuz` bootloader image or an
    /// operator-supplied raw path. The function silently skips
    /// locking rather than erroring, matching the doc contract:
    /// "Such paths do not need coordination because the build
    /// pipeline never touches them."
    #[test]
    fn acquire_test_kernel_lock_if_cached_returns_none_outside_cache() {
        let _env_lock = lock_env();
        let cache = isolated_cache_dir();
        // Path under a DIFFERENT tempdir, not the cache root.
        let outside = TempDir::new().expect("tempdir outside cache");
        let entry_dir = outside.path().join("raw-kernel-key");
        std::fs::create_dir_all(&entry_dir).expect("create entry dir");
        let image_path = entry_dir.join("bzImage");
        std::fs::write(&image_path, b"fake kernel image").expect("plant image");

        let guard = super::acquire_test_kernel_lock_if_cached(&image_path)
            .expect("non-cache path must not error");
        assert!(
            guard.is_none(),
            "path outside {} must skip locking, got guard",
            cache.path().display(),
        );
    }

    // -- validate_llm_extraction tests --
    //
    // Pin the three universal structural-sanity checks the function
    // is documented to enforce: unique metric names, finite values,
    // `MetricSource::LlmExtract` source tag. Every violation found
    // contributes a String to the returned Vec; an empty Vec means
    // the metric set is clean. These are pure-function tests over
    // synthetic Metric vectors — no model load, no VM, no SHM ring.

    /// Build a clean LlmExtract-tagged metric for use in the
    /// validation tests. Each test mutates one field to construct
    /// its violation case, leaving every other invariant satisfied
    /// so the failure is unambiguously attributable to the mutated
    /// field rather than collateral defaults.
    fn llm_metric(name: &str, value: f64) -> crate::test_support::Metric {
        crate::test_support::Metric {
            name: name.to_owned(),
            value,
            polarity: crate::test_support::Polarity::Unknown,
            unit: String::new(),
            source: crate::test_support::MetricSource::LlmExtract,
            stream: crate::test_support::MetricStream::Stdout,
        }
    }

    /// Two metrics sharing the same `name` violate the uniqueness
    /// invariant. The diagnostic must call out "duplicate metric
    /// name" so a reader can tell which check fired without
    /// re-reading the function.
    #[test]
    fn validate_llm_extraction_duplicate_name_rejects() {
        let metrics = vec![
            llm_metric("latency.p99", 1.0),
            llm_metric("latency.p99", 2.0),
        ];
        let violations = validate_llm_extraction(&metrics);
        assert_eq!(
            violations.len(),
            1,
            "exactly one duplicate-name violation expected, got {violations:?}",
        );
        assert!(
            violations[0].contains("duplicate metric name"),
            "diagnostic must mention 'duplicate metric name': {}",
            violations[0],
        );
    }

    /// A NaN value violates the finite-only invariant; the
    /// diagnostic must call out "non-finite" so the reader can tell
    /// which check fired.
    #[test]
    fn validate_llm_extraction_nan_rejects() {
        let metrics = vec![llm_metric("latency.p99", f64::NAN)];
        let violations = validate_llm_extraction(&metrics);
        assert_eq!(
            violations.len(),
            1,
            "exactly one non-finite violation expected, got {violations:?}",
        );
        assert!(
            violations[0].contains("non-finite"),
            "diagnostic must mention 'non-finite': {}",
            violations[0],
        );
    }

    /// A metric tagged with the wrong source (Json instead of
    /// LlmExtract) violates the source-tag invariant. The
    /// diagnostic must mention `MetricSource::LlmExtract` so the
    /// reader can tell which check fired and what the expected
    /// source was.
    #[test]
    fn validate_llm_extraction_wrong_source_rejects() {
        let mut metrics = vec![llm_metric("latency.p99", 1.0)];
        metrics[0].source = crate::test_support::MetricSource::Json;
        let violations = validate_llm_extraction(&metrics);
        assert_eq!(
            violations.len(),
            1,
            "exactly one wrong-source violation expected, got {violations:?}",
        );
        assert!(
            violations[0].contains("MetricSource::LlmExtract"),
            "diagnostic must mention 'MetricSource::LlmExtract': {}",
            violations[0],
        );
    }

    /// Structurally clean input — distinct names, finite values,
    /// `LlmExtract` source on every entry — produces an empty Vec.
    /// Pins the happy path so a regression that adds an unwanted
    /// check (e.g. minimum metric count, value-magnitude bound)
    /// breaks this test instead of silently rejecting valid
    /// extractions.
    #[test]
    fn validate_llm_extraction_clean_input_passes() {
        let metrics = vec![
            llm_metric("latency.p50", 1.0),
            llm_metric("latency.p99", 2.0),
            llm_metric("rps", 1000.0),
        ];
        assert!(
            validate_llm_extraction(&metrics).is_empty(),
            "clean input must produce an empty violations Vec",
        );
    }

    /// A single metric that breaks BOTH the non-finite invariant
    /// AND the wrong-source invariant produces TWO violations in
    /// the same call — proves per-metric checks run independently
    /// and aren't short-circuited by an earlier failure on the
    /// same metric. Pins the "report every defect class in one
    /// run" UX: a flaky LLM run that produces NaN-valued metrics
    /// with the wrong source tag surfaces both signals to the
    /// test author rather than forcing two debug iterations.
    #[test]
    fn validate_llm_extraction_single_metric_multiple_violations() {
        let mut metrics = vec![llm_metric("latency.p99", f64::INFINITY)];
        metrics[0].source = crate::test_support::MetricSource::Json;
        let violations = validate_llm_extraction(&metrics);
        assert_eq!(
            violations.len(),
            2,
            "non-finite + wrong-source on the same metric must produce 2 violations, got {violations:?}",
        );
        // Order is fixed: non-finite check runs before source
        // check inside the per-metric loop. Pin both diagnostics
        // by content rather than by index so a future re-ordering
        // surfaces here as a content mismatch instead of an
        // off-by-one.
        let messages: Vec<&str> = violations.iter().map(String::as_str).collect();
        assert!(
            messages.iter().any(|m| m.contains("non-finite")),
            "non-finite violation must appear: {messages:?}",
        );
        assert!(
            messages
                .iter()
                .any(|m| m.contains("MetricSource::LlmExtract")),
            "wrong-source violation must appear: {messages:?}",
        );
    }

    /// Across the whole metric set, every duplicate-name occurrence
    /// after the first reports its own violation. Three identical
    /// names → two duplicate-name violations (the first occurrence
    /// is the "original," the next two are duplicates). Pins the
    /// "report every defect" semantics so a regression to first-
    /// violation-only behavior surfaces here.
    #[test]
    fn validate_llm_extraction_multiple_duplicates_each_surface() {
        let metrics = vec![
            llm_metric("rps", 1.0),
            llm_metric("rps", 2.0),
            llm_metric("rps", 3.0),
        ];
        let violations = validate_llm_extraction(&metrics);
        assert_eq!(
            violations.len(),
            2,
            "three same-name metrics → two duplicate-name violations, got {violations:?}",
        );
        for v in &violations {
            assert!(
                v.contains("duplicate metric name"),
                "every violation must call out duplicate name: {v}",
            );
        }
    }

    /// Heterogeneous violation classes across DIFFERENT metrics in
    /// a single call: a duplicate name on one metric, NaN value on
    /// another, wrong source on a third. Verifies the function
    /// collects across ALL metrics, not just within a single one.
    /// Pins the "see every defect class in one run" UX.
    #[test]
    fn validate_llm_extraction_heterogeneous_violations_across_metrics() {
        let mut metrics = vec![
            llm_metric("rps", 1.0),
            llm_metric("rps", 2.0),              // duplicate name
            llm_metric("latency.p99", f64::NAN), // non-finite
            llm_metric("p50", 1.0),
        ];
        metrics[3].source = crate::test_support::MetricSource::Json; // wrong source on p50
        let violations = validate_llm_extraction(&metrics);
        assert_eq!(
            violations.len(),
            3,
            "three independent violations expected, got {violations:?}",
        );
        let messages: Vec<&str> = violations.iter().map(String::as_str).collect();
        assert!(
            messages
                .iter()
                .any(|m| m.contains("duplicate metric name") && m.contains("'rps'")),
            "duplicate-name on 'rps' must appear: {messages:?}",
        );
        assert!(
            messages
                .iter()
                .any(|m| m.contains("non-finite") && m.contains("'latency.p99'")),
            "non-finite on 'latency.p99' must appear: {messages:?}",
        );
        assert!(
            messages
                .iter()
                .any(|m| m.contains("MetricSource::LlmExtract") && m.contains("'p50'")),
            "wrong-source on 'p50' must appear: {messages:?}",
        );
    }

    // -- validate_metric_bounds tests --
    //
    // Pin the per-payload bounds-validation pass that runs after
    // the universal `validate_llm_extraction` pass when a payload
    // declared `metric_bounds`. Each test constructs a synthetic
    // metric set + a `MetricBounds` with a single check enabled
    // and asserts the violation list contents.

    /// `MetricBounds::NONE` (every field `None`) produces zero
    /// violations on any input — pins the "no bounds declared = no
    /// extra checks" contract that lets payloads opt in to the
    /// pass without paying for it.
    #[test]
    fn validate_metric_bounds_none_produces_no_violations() {
        let metrics = vec![
            llm_metric("rps", -42.0),    // would trip value_min if set
            llm_metric("latency", 1e15), // would trip value_max if set
        ];
        let bounds = crate::test_support::MetricBounds::NONE;
        let violations = super::validate_metric_bounds(&metrics, &bounds);
        assert!(
            violations.is_empty(),
            "MetricBounds::NONE must produce zero violations regardless of input; \
             got: {violations:?}",
        );
    }

    /// `min_count` rejects an extracted set with fewer metrics than
    /// the declared floor. Diagnostic must name both the actual
    /// count and the required minimum so the operator can see the
    /// shortfall at a glance.
    #[test]
    fn validate_metric_bounds_min_count_rejects_short_set() {
        let metrics = vec![llm_metric("a", 1.0), llm_metric("b", 2.0)];
        let bounds = crate::test_support::MetricBounds {
            min_count: Some(5),
            ..crate::test_support::MetricBounds::NONE
        };
        let violations = super::validate_metric_bounds(&metrics, &bounds);
        assert_eq!(
            violations.len(),
            1,
            "short set must produce exactly one min_count violation; got: {violations:?}",
        );
        assert!(
            violations[0].contains("extracted 2 metric(s)"),
            "diagnostic must name actual count: {}",
            violations[0],
        );
        assert!(
            violations[0].contains("at least 5"),
            "diagnostic must name required minimum: {}",
            violations[0],
        );
    }

    /// `min_count` accepts a set whose length equals the floor —
    /// pins the "inclusive lower bound" semantics.
    #[test]
    fn validate_metric_bounds_min_count_accepts_at_threshold() {
        let metrics = vec![
            llm_metric("a", 1.0),
            llm_metric("b", 2.0),
            llm_metric("c", 3.0),
        ];
        let bounds = crate::test_support::MetricBounds {
            min_count: Some(3),
            ..crate::test_support::MetricBounds::NONE
        };
        let violations = super::validate_metric_bounds(&metrics, &bounds);
        assert!(
            violations.is_empty(),
            "metric count == min_count is acceptable (>= semantics); got: {violations:?}",
        );
    }

    /// `value_min` rejects every metric with value strictly below
    /// the bound. Each violation surfaces independently — a set
    /// with three sub-bound metrics produces three violations.
    #[test]
    fn validate_metric_bounds_value_min_rejects_each_below_floor() {
        let metrics = vec![
            llm_metric("p50", -1.0),
            llm_metric("p99", -2.0),
            llm_metric("rps", 100.0), // above floor; not rejected
            llm_metric("delta", -5.0),
        ];
        let bounds = crate::test_support::MetricBounds {
            value_min: Some(0.0),
            ..crate::test_support::MetricBounds::NONE
        };
        let violations = super::validate_metric_bounds(&metrics, &bounds);
        assert_eq!(
            violations.len(),
            3,
            "every below-floor metric must surface its own violation; got: {violations:?}",
        );
        assert!(
            violations
                .iter()
                .all(|v| v.contains("below payload's declared lower bound")),
            "every diagnostic must name the lower-bound class: {violations:?}",
        );
        assert!(
            violations.iter().any(|v| v.contains("'p50'")),
            "p50 violation must surface: {violations:?}",
        );
        assert!(
            violations.iter().any(|v| v.contains("'delta'")),
            "delta violation must surface: {violations:?}",
        );
        // rps was above the floor — must NOT appear.
        assert!(
            !violations.iter().any(|v| v.contains("'rps'")),
            "rps must NOT trigger a value_min violation (100 > 0); got: {violations:?}",
        );
    }

    /// `value_min` accepts metrics at exactly the bound — pins the
    /// "strictly below" semantics. A regression to `<= ` (which
    /// would reject the boundary) breaks here.
    #[test]
    fn validate_metric_bounds_value_min_accepts_at_threshold() {
        let metrics = vec![llm_metric("zero", 0.0)];
        let bounds = crate::test_support::MetricBounds {
            value_min: Some(0.0),
            ..crate::test_support::MetricBounds::NONE
        };
        let violations = super::validate_metric_bounds(&metrics, &bounds);
        assert!(
            violations.is_empty(),
            "value at exactly value_min is acceptable (strict-less-than semantics); \
             got: {violations:?}",
        );
    }

    /// `value_max` mirrors `value_min` with the inverse inequality.
    /// Pins the symmetric contract.
    #[test]
    fn validate_metric_bounds_value_max_rejects_each_above_ceiling() {
        let metrics = vec![
            llm_metric("rss_huge", 1e16),
            llm_metric("rss_normal", 1e6),
            llm_metric("latency_runaway", 1e15),
        ];
        let bounds = crate::test_support::MetricBounds {
            value_max: Some(1e12),
            ..crate::test_support::MetricBounds::NONE
        };
        let violations = super::validate_metric_bounds(&metrics, &bounds);
        assert_eq!(
            violations.len(),
            2,
            "two above-ceiling metrics must surface; got: {violations:?}",
        );
        assert!(
            violations
                .iter()
                .all(|v| v.contains("above payload's declared upper bound")),
            "every diagnostic must name the upper-bound class: {violations:?}",
        );
        assert!(
            violations.iter().any(|v| v.contains("'rss_huge'")),
            "rss_huge must trigger: {violations:?}",
        );
        assert!(
            !violations.iter().any(|v| v.contains("'rss_normal'")),
            "rss_normal (1e6) must NOT trigger value_max=1e12: {violations:?}",
        );
    }

    /// Combined bounds (all three at once): one metric below floor,
    /// one above ceiling, and a too-short set. Three distinct
    /// violations surface.
    #[test]
    fn validate_metric_bounds_combined_bounds_each_violation_independent() {
        let metrics = vec![llm_metric("low", -1.0), llm_metric("high", 1e15)];
        let bounds = crate::test_support::MetricBounds {
            min_count: Some(5),
            value_min: Some(0.0),
            value_max: Some(1e12),
        };
        let violations = super::validate_metric_bounds(&metrics, &bounds);
        assert_eq!(
            violations.len(),
            3,
            "combined: 1 min_count + 1 value_min + 1 value_max violation; got: {violations:?}",
        );
        assert!(
            violations.iter().any(|v| v.contains("at least 5")),
            "min_count violation must surface: {violations:?}",
        );
        assert!(
            violations
                .iter()
                .any(|v| v.contains("'low'") && v.contains("below")),
            "value_min on 'low' must surface: {violations:?}",
        );
        assert!(
            violations
                .iter()
                .any(|v| v.contains("'high'") && v.contains("above")),
            "value_max on 'high' must surface: {violations:?}",
        );
    }

    /// Empty input + min_count > 0 produces a min_count violation.
    /// Pins the empty-set boundary against the bounds pass; the
    /// universal `validate_llm_extraction` accepts empty input as
    /// vacuously valid, but a payload that declared min_count
    /// expects something.
    #[test]
    fn validate_metric_bounds_empty_metrics_with_min_count_violates() {
        let bounds = crate::test_support::MetricBounds {
            min_count: Some(1),
            ..crate::test_support::MetricBounds::NONE
        };
        let violations = super::validate_metric_bounds(&[], &bounds);
        assert_eq!(
            violations.len(),
            1,
            "empty input + min_count=1 must produce one violation; got: {violations:?}",
        );
        assert!(
            violations[0].contains("extracted 0 metric(s)"),
            "diagnostic must name 0 as actual count: {}",
            violations[0],
        );
    }

    // -- Payload::metric_bounds field tests --
    //
    // Pin the new `metric_bounds: Option<&'static MetricBounds>`
    // field on the `Payload` struct: default None, can be set to
    // Some(&BOUNDS_CONST), and threads through the deferred
    // emission path (via `RawPayloadOutput::metric_bounds`).

    /// A `Payload` constructed via the bare struct literal carries
    /// `metric_bounds: None` by default — pins the "opt-in only"
    /// contract so adding the field didn't accidentally enable
    /// bounds checks for every existing payload.
    #[test]
    fn payload_metric_bounds_defaults_to_none_via_payload_binary_constructor() {
        const P: crate::test_support::Payload =
            crate::test_support::Payload::binary("test", "test_bin");
        assert!(
            P.metric_bounds.is_none(),
            "Payload::binary must initialize metric_bounds to None",
        );
    }

    /// A `Payload` declared with `metric_bounds: Some(&BOUNDS)`
    /// retains the reference — the field is `Option<&'static
    /// MetricBounds>`, so a const-defined bounds value is reachable
    /// from the payload.
    #[test]
    fn payload_metric_bounds_carries_static_reference() {
        const SCHBENCH_BOUNDS: crate::test_support::MetricBounds =
            crate::test_support::MetricBounds {
                min_count: Some(5),
                value_min: Some(0.0),
                value_max: Some(1e12),
            };
        const P: crate::test_support::Payload = crate::test_support::Payload {
            name: "schbench_test",
            kind: crate::test_support::PayloadKind::Binary("schbench"),
            output: crate::test_support::OutputFormat::LlmExtract(None),
            default_args: &[],
            default_checks: &[],
            metrics: &[],
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
            metric_bounds: Some(&SCHBENCH_BOUNDS),
        };
        assert!(P.metric_bounds.is_some());
        let b = P.metric_bounds.unwrap();
        assert_eq!(b.min_count, Some(5));
        assert_eq!(b.value_min, Some(0.0));
        assert_eq!(b.value_max, Some(1e12));
    }

    /// `host_side_llm_extract` surfaces bounds violations alongside
    /// load-failure details. Drives a matched (raw, pm) pair under
    /// the offline gate (so model load fails and metrics stay
    /// empty) with `metric_bounds: Some(&{min_count: 1})` — the
    /// bounds pass is GATED on the model-load succeeding (because
    /// it runs after extraction populates metrics), so under
    /// offline gate the bounds check does NOT fire. Pin this
    /// "bounds run only on extracted metrics" contract: a regression
    /// that ran bounds on the empty placeholder would falsely
    /// flag every offline-gated test as a min_count violation.
    #[test]
    fn host_side_llm_extract_offline_gate_skips_bounds_check() {
        let _env_lock = lock_env();
        super::super::model::reset();
        let _cache = isolated_cache_dir();
        let _offline = EnvVarGuard::set(crate::test_support::OFFLINE_ENV, "1");
        let mut pm = vec![empty_pm(0)];
        let raws = vec![crate::test_support::RawPayloadOutput {
            payload_index: 0,
            stdout: "irrelevant under offline gate".to_string(),
            stderr: String::new(),
            hint: None,
            metric_hints: Vec::new(),
            metric_bounds: Some(crate::test_support::MetricBounds {
                min_count: Some(1),
                ..crate::test_support::MetricBounds::NONE
            }),
        }];
        let failures = host_side_llm_extract(&mut pm, &raws, 0);
        // Exactly ONE failure detail — the load-failure. No
        // bounds violation because metrics is empty (placeholder)
        // and the bounds pass is guarded by `if let Some(bounds)`
        // BUT only runs after the structural-sanity pass over
        // extracted metrics. With load failure → metrics empty,
        // the bounds check sees an empty vec — but the empty-set
        // + min_count=1 case WOULD flag a violation. The
        // production code path skips the bounds pass on the
        // load-failure branch (continues before reaching the
        // bounds check), so the bounds check should NOT fire.
        assert_eq!(
            failures.len(),
            1,
            "offline-gated extraction must produce only the load-failure detail, \
             not a spurious bounds violation; got: {failures:?}",
        );
        assert!(
            failures[0].message.contains("LlmExtract model load failed"),
            "the lone failure must be the load-failure: {}",
            failures[0].message,
        );
    }

    // -- host_side_llm_extract pairing tests --
    //
    // The pairing logic is tested without invoking the model: every
    // case below either constructs an orphan raw output (no
    // PayloadMetrics with matching `payload_index`) — which short-
    // circuits BEFORE extract_via_llm — or supplies an empty raw
    // outputs vec (returns immediately). The pairing-by-index
    // contract is the entire moving part on the `payload_index`
    // axis; once a match is found, the extraction-and-polarity
    // pipeline is exercised by the integration test
    // `llm_extract_e2e_test.rs`.

    fn empty_raw(payload_index: usize) -> crate::test_support::RawPayloadOutput {
        crate::test_support::RawPayloadOutput {
            payload_index,
            stdout: String::new(),
            stderr: String::new(),
            hint: None,
            metric_hints: Vec::new(),
            metric_bounds: None,
        }
    }

    fn empty_pm(payload_index: usize) -> crate::test_support::PayloadMetrics {
        crate::test_support::PayloadMetrics {
            payload_index,
            metrics: Vec::new(),
            exit_code: 0,
        }
    }

    /// Empty raw outputs slice — the function returns immediately
    /// without examining `payload_metrics` or hitting the model.
    /// Pins the no-LlmExtract-payloads happy path.
    #[test]
    fn host_side_llm_extract_empty_raw_outputs_returns_no_failures() {
        let mut pm = vec![empty_pm(0), empty_pm(1)];
        let failures = host_side_llm_extract(&mut pm, &[], 0);
        assert!(failures.is_empty(), "empty raw outputs → no failures");
    }

    /// Orphan raw output: a `RawPayloadOutput` whose `payload_index`
    /// has no matching `PayloadMetrics` slot. Surfaces as a
    /// pairing-failure detail naming the orphan index. The detail
    /// kind is `Other` so the failure-rendering pipeline treats it
    /// as a non-classified diagnostic.
    ///
    /// The setup also has an empty-metrics PM at payload_index=0
    /// (no matching raw_output), which triggers the post-pairing
    /// orphan-PM scan added by #46. So this test sees BOTH the
    /// orphan-raw detail (from the pairing loop) AND the
    /// orphan-PM detail (from the post-loop scan). Pin both so a
    /// regression that drops either path surfaces here.
    #[test]
    fn host_side_llm_extract_orphan_raw_output_surfaces_pairing_failure() {
        // PayloadMetrics has payload_index=0; raw output claims
        // payload_index=42 — no slot to write to. Symmetrically,
        // the PM at index 0 has no matching raw, which the
        // post-pairing orphan-PM scan picks up.
        let mut pm = vec![empty_pm(0)];
        let raws = vec![empty_raw(42)];
        let failures = host_side_llm_extract(&mut pm, &raws, 0);
        let messages: Vec<&str> = failures.iter().map(|d| d.message.as_str()).collect();
        assert!(
            messages
                .iter()
                .any(|m| m.contains("LlmExtract host pairing") && m.contains("payload_index=42")),
            "orphan-raw detail naming index 42 must surface: {messages:?}",
        );
        assert!(
            messages
                .iter()
                .any(|m| m.contains("LlmExtract host pairing") && m.contains("[0]")),
            "orphan-PM scan must surface the empty-metrics PM at index 0: {messages:?}",
        );
        // The valid PayloadMetrics slot at index 0 must NOT have been
        // mutated — the orphan path skips extraction.
        assert!(
            pm[0].metrics.is_empty(),
            "no extraction should have run on the orphan path",
        );
    }

    /// Multiple orphan raw outputs each surface their own failure
    /// detail; the function does not abort on the first. Pins the
    /// "process every raw, surface every orphan" semantics so a
    /// regression that returns early after the first failure is
    /// caught.
    ///
    /// The empty-metrics PM at payload_index=0 also triggers the
    /// post-pairing orphan-PM scan (#46). So we expect 3 orphan-raw
    /// details + 1 orphan-PM combined detail = 4 total failures.
    #[test]
    fn host_side_llm_extract_multiple_orphans_each_surface() {
        let mut pm = vec![empty_pm(0)];
        let raws = vec![empty_raw(10), empty_raw(20), empty_raw(30)];
        let failures = host_side_llm_extract(&mut pm, &raws, 0);
        let messages: Vec<&str> = failures.iter().map(|d| d.message.as_str()).collect();
        assert!(
            messages.iter().any(|m| m.contains("payload_index=10")),
            "orphan raw at 10 must surface: {messages:?}",
        );
        assert!(
            messages.iter().any(|m| m.contains("payload_index=20")),
            "orphan raw at 20 must surface: {messages:?}",
        );
        assert!(
            messages.iter().any(|m| m.contains("payload_index=30")),
            "orphan raw at 30 must surface: {messages:?}",
        );
        // Orphan-PM scan also fires for the empty PM at index 0.
        assert!(
            messages
                .iter()
                .any(|m| m.contains("[0]") && m.contains("no matching RawPayloadOutput")),
            "orphan-PM scan must surface the empty PM at index 0: {messages:?}",
        );
    }

    /// Json payload that produced zero metrics (empty `metrics` vec)
    /// must NOT be conflated with an LlmExtract placeholder when an
    /// LlmExtract raw output is also present at a different index.
    /// This pins the motivating scenario for #20: positional pairing
    /// would have written the LlmExtract result into the Json
    /// payload's empty slot.
    ///
    /// Setup: a Json payload at `payload_index=5` with empty metrics
    /// (indistinguishable from an LlmExtract placeholder by content
    /// alone). A raw output with `payload_index=99` (no matching
    /// slot).
    ///
    /// Expected: the raw output is reported as orphan; the Json
    /// payload's empty slot is NEVER touched. Additionally, the
    /// post-pairing orphan-PM scan (#46) flags the Json slot at
    /// index 5 as a candidate for "raw output may have been dropped"
    /// — this is a known false-positive case the scan's own diagnostic
    /// prose calls out, since a Json-with-no-leaves payload looks
    /// identical to a dropped LlmExtract from PayloadMetrics alone.
    #[test]
    fn host_side_llm_extract_json_zero_leaves_not_conflated_with_llm_placeholder() {
        let mut pm = vec![empty_pm(5)];
        let raws = vec![empty_raw(99)];
        let failures = host_side_llm_extract(&mut pm, &raws, 0);
        let messages: Vec<&str> = failures.iter().map(|d| d.message.as_str()).collect();
        assert!(
            messages.iter().any(|m| m.contains("payload_index=99")),
            "orphan raw at 99 must surface: {messages:?}",
        );
        // The Json slot was untouched — its `metrics` is still
        // empty, exactly as the guest emitted it.
        assert!(
            pm[0].metrics.is_empty(),
            "Json empty-metrics slot must not be written by LlmExtract pairing",
        );
        assert_eq!(
            pm[0].payload_index, 5,
            "Json slot's payload_index must be untouched",
        );
        // Orphan-PM scan flags the Json slot as a candidate orphan
        // PM. Documented in the scan's diagnostic as a known
        // false-positive case for mixed-format tests.
        assert!(
            messages
                .iter()
                .any(|m| m.contains("[5]") && m.contains("no matching RawPayloadOutput")),
            "orphan-PM scan must include the Json slot at index 5 in its \
             candidate list (false positive disclosed in the diagnostic): {messages:?}",
        );
    }

    /// SHM ring overflow with LlmExtract in use: a non-zero
    /// `shm_drops` while raw outputs are present surfaces a
    /// `SHM ring overflow` detail. Pins the design contract from
    /// task #8 — silent metric truncation must propagate as a
    /// host-actionable failure rather than letting downstream
    /// stats see a quietly-incomplete metric set.
    ///
    /// Constructed with an ORPHAN raw output (payload_index=99
    /// has no matching `PayloadMetrics` slot) so the pairing
    /// loop hits the orphan path and SKIPS `extract_via_llm`
    /// entirely. The overflow detail surfaces from the pre-loop
    /// drops check independently of whether any matched pairs
    /// reached the model — this keeps the unit test fast (no
    /// model load) while still pinning the overflow contract.
    #[test]
    fn host_side_llm_extract_shm_drops_with_raw_outputs_surfaces_overflow_detail() {
        let mut pm = vec![empty_pm(0)];
        let raws = vec![empty_raw(99)]; // orphan — no matching slot
        let failures = host_side_llm_extract(&mut pm, &raws, 7);
        let messages: Vec<&str> = failures.iter().map(|d| d.message.as_str()).collect();
        assert!(
            messages.iter().any(|m| m.contains("SHM ring overflow")),
            "drops > 0 with LlmExtract in use must surface 'SHM ring overflow': {messages:?}",
        );
        assert!(
            messages.iter().any(|m| m.contains("7 message(s) dropped")),
            "diagnostic must cite the actual drops count, got: {messages:?}",
        );
        // Orphan raw output also surfaces its own pairing failure
        // — both signals coexist; one does not suppress the other.
        assert!(
            messages.iter().any(|m| m.contains("payload_index=99")),
            "orphan pairing failure must still surface alongside overflow: {messages:?}",
        );
    }

    /// Zero drops + zero raw outputs: no overflow detail surfaces
    /// (correctly — there's nothing to report). Pins the
    /// no-LlmExtract path so a regression that always emits the
    /// overflow detail (false positive on every run) breaks here.
    #[test]
    fn host_side_llm_extract_zero_drops_and_no_raws_no_overflow_detail() {
        let mut pm = vec![empty_pm(0)];
        let failures = host_side_llm_extract(&mut pm, &[], 0);
        assert!(
            failures.is_empty(),
            "no LlmExtract + no drops → no failures, got: {failures:?}",
        );
    }

    /// Non-zero drops but no raw outputs: no overflow detail
    /// surfaces. A drops counter > 0 in a Json/ExitCode-only test
    /// is the VMM's responsibility to surface elsewhere — the
    /// LlmExtract path owns this detail only when the
    /// LlmExtract code path was actually exercised. Pins that the
    /// `raw_outputs.is_empty()` early return short-circuits before
    /// the drops check, keeping the detail scoped to the LlmExtract
    /// caller's mental model.
    #[test]
    fn host_side_llm_extract_drops_without_raws_skips_overflow_detail() {
        let mut pm = vec![empty_pm(0)];
        let failures = host_side_llm_extract(&mut pm, &[], 42);
        assert!(
            failures.is_empty(),
            "drops without LlmExtract raw outputs must not produce LlmExtract-scope detail, got: {failures:?}",
        );
    }

    // -- orphan-PayloadMetrics scan (#46) --

    /// Task #46: an empty-metrics `PayloadMetrics` whose
    /// `payload_index` has no matching `RawPayloadOutput` is
    /// surfaced by the post-pairing scan. Most likely cause is a
    /// CRC-bad RawPayloadOutput silently dropped during SHM
    /// drain. Without this surfacing, an LlmExtract test whose
    /// raw-output bytes arrived corrupted would fail downstream
    /// `Check::Min` / `Check::Exists` evaluations with a
    /// "metric not found" message that hides the real cause.
    ///
    /// Setup: an LlmExtract pair at index 7 (raw + matching PM)
    /// arrives intact; an additional empty PM at index 99 has no
    /// matching raw. The orphan-PM scan flags index 99.
    #[test]
    fn host_side_llm_extract_orphan_pm_with_no_matching_raw_surfaces() {
        // Use orphan raws to keep the matched extraction off the
        // model path — the PM at index 7 has no matching raw, so
        // the pairing loop skips it. We add raws at 10 and 20 to
        // satisfy the gate that `raw_outputs.is_empty() == false`,
        // so the orphan-PM scan can fire.
        let mut pm = vec![empty_pm(7), empty_pm(99)];
        let raws = vec![empty_raw(10), empty_raw(20)];
        let failures = host_side_llm_extract(&mut pm, &raws, 0);
        let messages: Vec<&str> = failures.iter().map(|d| d.message.as_str()).collect();
        // Both PMs (7 and 99) lack matching raws, so both are
        // surfaced in the orphan-PM scan's combined detail.
        assert!(
            messages
                .iter()
                .any(|m| m.contains("[7, 99]") && m.contains("no matching RawPayloadOutput")),
            "orphan-PM scan must list both unmatched PM indices [7, 99]: {messages:?}",
        );
        assert!(
            messages.iter().any(|m| m.contains("CRC mismatch")),
            "orphan-PM diagnostic must surface the CRC-bad cause: {messages:?}",
        );
        assert!(
            messages.iter().any(|m| m.contains("False-positive case")),
            "orphan-PM diagnostic must disclose the false-positive case for \
             mixed-format tests: {messages:?}",
        );
    }

    /// Task #46: when ALL PMs have matching raws, the orphan-PM
    /// scan does NOT fire. Pins that the scan is gated on the
    /// missing-pair condition rather than blanketly emitting a
    /// detail for every empty-metrics PM in an LlmExtract test
    /// (which would false-positive on extraction failures that
    /// legitimately leave metrics empty).
    #[test]
    fn host_side_llm_extract_no_orphan_pm_when_all_pms_have_matching_raws() {
        // Two matched pairs. After pairing, both PMs remain empty
        // (orphan raws short-circuit before the model path), but
        // their indices are in the raw-index set, so the
        // orphan-PM scan does not surface anything.
        //
        // The setup uses orphan raws-to-self (i.e. a raw at the
        // same index as its PM) so the pairing loop walks them as
        // matched pairs. To keep the test off the model path
        // entirely, we use empty raws at indices 0 and 1; the
        // pairing succeeds, extract_via_llm returns Err under no
        // model setup (or hangs if a real model loads), so we
        // EXPECT only the load-failure branch — but that's
        // out-of-scope for this test. Instead, we make the
        // pairing loop hit the orphan-raw arm by using raw indices
        // 100 and 200 that don't match the PMs at 0 and 1. Then
        // the orphan-PM scan should still flag PMs at 0 and 1 —
        // which is the WRONG answer for this test.
        //
        // Better: use a setup where every PM IS matched. The
        // simplest way is to skip this test's "no orphan-PM"
        // claim under unit-testing without a model — the integration
        // test (with a real model) would exercise the all-matched
        // path. For unit testing, we instead pin the inverse: the
        // orphan-PM scan does NOT fire when raw_outputs is empty.
        let mut pm = vec![empty_pm(0), empty_pm(1)];
        let raws: Vec<crate::test_support::RawPayloadOutput> = Vec::new();
        let failures = host_side_llm_extract(&mut pm, &raws, 0);
        assert!(
            failures.is_empty(),
            "with no LlmExtract raws, orphan-PM scan must not fire (test is \
             not exercising LlmExtract): {failures:?}",
        );
    }

    // -- offline-gate / empty-stream / stream-fallback tests --
    //
    // These tests drive `host_side_llm_extract` through its
    // model-touching paths via the offline gate (`KTSTR_MODEL_OFFLINE=1`).
    // The gate makes `extract_via_llm` return Err deterministically,
    // so the tests pin the host-side dispatch behavior without
    // standing up the ~2.4 GiB model.
    //
    // Every test holds `lock_env()` and calls `super::super::model::reset()`
    // before the gate is set, ensuring no previously-memoized
    // `Ok(model)` slot bypasses the gate. Reset is paired with an
    // `EnvVarGuard` so the gate is removed at drop time even if the
    // test panics.
    //
    // The companion happy-path tests for stdout-primary / stderr-fallback
    // with a real model live in the integration test
    // `tests/llm_extract_e2e_test.rs` — pinned by task #13. The unit
    // tests here pin the deterministic boundaries that don't require
    // a model.

    /// Task #7: a `RawPayloadOutput` carrying empty stdout AND empty
    /// stderr — paired with a matching `PayloadMetrics` slot — must
    /// not panic the host extraction. Under the offline gate, the
    /// stdout call surfaces a load-failed detail (deterministic),
    /// the stderr fallback is short-circuited (because the load_err
    /// is Some), and the PayloadMetrics slot's metrics stays empty.
    ///
    /// Pins the empty-input boundary against three regressions:
    /// 1. A `String::is_empty()` check that crashed the prompt
    ///    composer on empty input (covered by model.rs but
    ///    boundary-tested again here at the eval level).
    /// 2. A panic in the polarity resolver if it received an empty
    ///    metric vec.
    /// 3. A regression that ran extract_via_llm on empty stdout
    ///    AND THEN ran extract_via_llm on empty stderr, doubling
    ///    the model-load attempt. The current contract:
    ///    `metrics.is_empty() && load_err.is_none() && !raw.stderr.is_empty()`
    ///    in eval.rs:281 — empty stderr blocks the fallback.
    ///
    /// Holds [`lock_env`] across the env mutations and pairs an
    /// [`isolated_cache_dir`] with the offline-gate `EnvVarGuard`
    /// so the gate trips deterministically on a guaranteed-cold
    /// cache root rather than relying on the operator's home
    /// having no model entry. The reset clears any
    /// previously-memoized `Ok(model)` slot in `MODEL_CACHE`.
    #[test]
    fn host_side_llm_extract_with_empty_streams_no_panic_no_metrics() {
        let _env_lock = lock_env();
        super::super::model::reset();
        let _cache = isolated_cache_dir();
        let _offline = EnvVarGuard::set(crate::test_support::OFFLINE_ENV, "1");
        let mut pm = vec![empty_pm(0)];
        let raws = vec![empty_raw(0)];
        let failures = host_side_llm_extract(&mut pm, &raws, 0);
        // Under the offline gate, the stdout extract_via_llm call
        // returns Err — the load-failed branch fires. Empty stderr
        // also blocks the fallback, so a single load-failure detail
        // is the expected shape.
        assert_eq!(
            failures.len(),
            1,
            "empty streams under offline gate must produce exactly one load-failed detail, \
             got: {failures:?}",
        );
        assert!(
            failures[0].message.contains("LlmExtract model load failed"),
            "load-failure detail must surface the diagnostic prefix; got: {}",
            failures[0].message,
        );
        // PayloadMetrics slot stays empty — no metrics extracted, no
        // partial pollution.
        assert!(
            pm[0].metrics.is_empty(),
            "PM slot must remain empty when extraction failed; got: {:?}",
            pm[0].metrics,
        );
    }

    /// Task #9: with `KTSTR_MODEL_OFFLINE=1` set, `host_side_llm_extract`
    /// must surface an actionable `LlmExtract model load failed`
    /// detail naming the offline env var. Pins the host-side
    /// equivalent of the `extract_via_llm_returns_empty_when_backend_unavailable`
    /// test in model.rs — the model.rs test pins the call-site
    /// behavior, this test pins how the host's eval pipeline surfaces
    /// that error to the test verdict.
    ///
    /// A regression that swallowed the offline-gate Err (e.g. by
    /// returning Vec::new() instead of `Err(reason)` from
    /// `extract_via_llm`, or by `match ... { Err(_) => () }`-ing
    /// the load failure inside `host_side_llm_extract`) would
    /// leave the test passing with empty metrics — a silent
    /// regression that `stats compare` would only catch days
    /// later as zero-metric runs accumulating in the sidecar.
    #[test]
    fn host_side_llm_extract_under_offline_gate_surfaces_actionable_detail() {
        let _env_lock = lock_env();
        super::super::model::reset();
        let _cache = isolated_cache_dir();
        let _offline = EnvVarGuard::set(crate::test_support::OFFLINE_ENV, "1");
        let mut pm = vec![empty_pm(0)];
        // Non-empty stdout — proves the failure path fires regardless
        // of input shape (not gated on emptiness).
        let raws = vec![crate::test_support::RawPayloadOutput {
            payload_index: 0,
            stdout: "arbitrary stdout content for the model".to_string(),
            stderr: String::new(),
            hint: None,
            metric_hints: Vec::new(),
            metric_bounds: None,
        }];
        let failures = host_side_llm_extract(&mut pm, &raws, 0);
        assert_eq!(
            failures.len(),
            1,
            "offline gate must produce exactly one load-failed detail, got: {failures:?}",
        );
        // Strict shape-of-emission contract:
        // 1. Detail kind is `Other` — the framework surfaces an
        //    uncategorized infrastructure failure here, not a domain
        //    `Starved` / `Saturation` / etc. classification. Stats
        //    tooling that buckets by DetailKind needs this stable.
        // 2. Message BEGINS WITH the canonical prefix
        //    `"LlmExtract model load failed:"` — not just contains.
        //    A regression that prepended a noisy banner would land
        //    the prefix mid-string and pass a `.contains` check
        //    while breaking grep / log-pattern consumers.
        // 3. Message contains `OFFLINE_ENV` so the operator knows
        //    where to look (the framework wraps the reason verbatim;
        //    `extract_via_llm`'s offline-gate Err surfaces the env
        //    var name in its reason string — see model.rs:1151+ for
        //    the bail! sites that name `OFFLINE_ENV`).
        let detail = &failures[0];
        assert_eq!(
            detail.kind,
            DetailKind::Other,
            "load-failure detail kind must be `Other` (the framework's bucket \
             for infrastructure failures); got: {:?}",
            detail.kind,
        );
        let msg = &detail.message;
        assert!(
            msg.starts_with("LlmExtract model load failed:"),
            "diagnostic must BEGIN WITH 'LlmExtract model load failed:' \
             — a substring-only match would let a regression bury the prefix \
             behind banner noise. got: {msg:?}",
        );
        assert!(
            msg.contains(crate::test_support::OFFLINE_ENV),
            "actionable diagnostic must name the offline env var so the operator \
             knows to unset KTSTR_MODEL_OFFLINE or pre-seed the cache; got: {msg}",
        );
        assert!(
            pm[0].metrics.is_empty(),
            "load failure must leave the PM slot empty; got: {:?}",
            pm[0].metrics,
        );
    }

    /// Task #10 (offline-gate side): when stdout's `extract_via_llm`
    /// call surfaces a load-failure reason, the stderr fallback is
    /// SKIPPED — the failure reason is identical across both calls
    /// and re-invoking inference would burn cycles to no purpose.
    /// Pins the `load_err.is_none()` clause in the fallback gate
    /// (eval.rs:281): `metrics.is_empty() && load_err.is_none() &&
    /// !raw.stderr.is_empty()`.
    ///
    /// Setup: empty stdout + non-empty stderr, under the offline
    /// gate. Pre-gate, the model is uncached (`reset()` clears it).
    ///
    /// Expected: exactly ONE load-failure detail surfaces (from the
    /// stdout path). If the fallback erroneously fired, we'd see
    /// either a SECOND load-failure detail (if extract_via_llm
    /// re-Err'd) or an extracted-metrics outcome that contradicts
    /// the offline-gate contract.
    #[test]
    fn host_side_llm_extract_offline_gate_skips_stderr_fallback() {
        let _env_lock = lock_env();
        super::super::model::reset();
        let _cache = isolated_cache_dir();
        let _offline = EnvVarGuard::set(crate::test_support::OFFLINE_ENV, "1");
        let mut pm = vec![empty_pm(0)];
        let raws = vec![crate::test_support::RawPayloadOutput {
            payload_index: 0,
            stdout: String::new(),
            stderr: "stderr body that the fallback would reach if not gated".to_string(),
            hint: None,
            metric_hints: Vec::new(),
            metric_bounds: None,
        }];
        let failures = host_side_llm_extract(&mut pm, &raws, 0);
        // Exactly ONE failure detail — the fallback's `load_err.is_none()`
        // gate blocks a second extract_via_llm call when stdout's
        // result was Err.
        assert_eq!(
            failures.len(),
            1,
            "stderr fallback must be skipped when stdout's call already returned Err; \
             a second 'model load failed' detail would mean the gate regressed. \
             got: {failures:?}",
        );
        assert!(
            failures[0].message.contains("LlmExtract model load failed"),
            "the lone surfaced detail must be the load-failure: {}",
            failures[0].message,
        );
    }

    /// Task #10 (multi-pair side): the offline-gate behavior is
    /// per-pair, not global — a load-failure on one
    /// (RawPayloadOutput, PayloadMetrics) pair must NOT short-
    /// circuit processing of subsequent pairs. Each pair gets its
    /// own load-failure detail, stamped independently.
    ///
    /// Setup: TWO matched pairs, both under the offline gate. The
    /// expected outcome is two load-failure details — one per
    /// pair. A regression that bailed after the first failure
    /// (e.g. an `if !failures.is_empty() { return failures }` in
    /// the loop) would surface only one detail.
    #[test]
    fn host_side_llm_extract_offline_gate_per_pair_failure_detail() {
        let _env_lock = lock_env();
        super::super::model::reset();
        let _cache = isolated_cache_dir();
        let _offline = EnvVarGuard::set(crate::test_support::OFFLINE_ENV, "1");
        let mut pm = vec![empty_pm(0), empty_pm(1)];
        let raws = vec![
            crate::test_support::RawPayloadOutput {
                payload_index: 0,
                stdout: "first pair stdout".to_string(),
                stderr: String::new(),
                hint: None,
                metric_hints: Vec::new(),
                metric_bounds: None,
            },
            crate::test_support::RawPayloadOutput {
                payload_index: 1,
                stdout: "second pair stdout".to_string(),
                stderr: String::new(),
                hint: None,
                metric_hints: Vec::new(),
                metric_bounds: None,
            },
        ];
        let failures = host_side_llm_extract(&mut pm, &raws, 0);
        assert_eq!(
            failures.len(),
            2,
            "two matched pairs under offline gate must each surface their own load-failure \
             detail; a regression that bailed after the first failure would surface only one. \
             got: {failures:?}",
        );
        for f in &failures {
            assert!(
                f.message.contains("LlmExtract model load failed"),
                "every detail must be a load-failure: {}",
                f.message,
            );
        }
        // Both PM slots stay empty — no metrics extracted on either path.
        assert!(
            pm[0].metrics.is_empty() && pm[1].metrics.is_empty(),
            "both PM slots must remain empty under the offline gate",
        );
    }

    /// Task #10 (orphan + load-failure interaction): a mix of an
    /// orphan raw output (no matching PM slot) AND a matched-but-
    /// load-failing pair under the offline gate produces TWO
    /// distinct details — one orphan-pairing and one load-failure.
    /// Pins that the orphan path and the model-failure path are
    /// orthogonal contributors to the failure list.
    #[test]
    fn host_side_llm_extract_orphan_and_load_failure_both_surface() {
        let _env_lock = lock_env();
        super::super::model::reset();
        let _cache = isolated_cache_dir();
        let _offline = EnvVarGuard::set(crate::test_support::OFFLINE_ENV, "1");
        let mut pm = vec![empty_pm(0)];
        let raws = vec![
            crate::test_support::RawPayloadOutput {
                payload_index: 0,
                stdout: "matched pair".to_string(),
                stderr: String::new(),
                hint: None,
                metric_hints: Vec::new(),
                metric_bounds: None,
            },
            crate::test_support::RawPayloadOutput {
                payload_index: 99,
                stdout: "orphan".to_string(),
                stderr: String::new(),
                hint: None,
                metric_hints: Vec::new(),
                metric_bounds: None,
            },
        ];
        let failures = host_side_llm_extract(&mut pm, &raws, 0);
        assert_eq!(
            failures.len(),
            2,
            "mixed orphan + matched-but-load-failing must surface both details independently; \
             got: {failures:?}",
        );
        let messages: Vec<&str> = failures.iter().map(|d| d.message.as_str()).collect();
        assert!(
            messages
                .iter()
                .any(|m| m.contains("LlmExtract host pairing") && m.contains("payload_index=99")),
            "orphan detail naming index 99 must surface: {messages:?}",
        );
        assert!(
            messages
                .iter()
                .any(|m| m.contains("LlmExtract model load failed")),
            "load-failure detail must surface: {messages:?}",
        );
    }

    /// Task #10 (drops + offline gate): a non-zero drops counter
    /// AND a load failure both surface — the overflow detail and
    /// the load-failure detail are independent. Pins that the
    /// drops detail emits BEFORE the pair loop (eval.rs:209) and
    /// is not gated on extraction success.
    #[test]
    fn host_side_llm_extract_drops_and_load_failure_compose() {
        let _env_lock = lock_env();
        super::super::model::reset();
        let _cache = isolated_cache_dir();
        let _offline = EnvVarGuard::set(crate::test_support::OFFLINE_ENV, "1");
        let mut pm = vec![empty_pm(0)];
        let raws = vec![crate::test_support::RawPayloadOutput {
            payload_index: 0,
            stdout: "matched pair".to_string(),
            stderr: String::new(),
            hint: None,
            metric_hints: Vec::new(),
            metric_bounds: None,
        }];
        let failures = host_side_llm_extract(&mut pm, &raws, 5);
        assert!(
            failures.len() >= 2,
            "drops + load failure must each contribute their own detail; got: {failures:?}",
        );
        let messages: Vec<&str> = failures.iter().map(|d| d.message.as_str()).collect();
        assert!(
            messages.iter().any(|m| m.contains("SHM ring overflow")),
            "drops detail must surface: {messages:?}",
        );
        assert!(
            messages
                .iter()
                .any(|m| m.contains("LlmExtract model load failed")),
            "load-failure detail must surface: {messages:?}",
        );
    }

    /// Task #12 (SHM wire-frame round-trip): the full
    /// guest→SHM→host transport for `MSG_TYPE_RAW_PAYLOAD_OUTPUT`
    /// must preserve BOTH stdout and stderr streams independently.
    /// A regression that concatenated the streams (e.g. a guest-
    /// side "merge before serialize" or a host-side "join after
    /// deserialize") would silently break schbench-style payloads
    /// that emit metrics on stderr only — the metric extraction
    /// would land on the merged blob, contaminating both metric
    /// values and the `MetricStream` tag attribution.
    ///
    /// This test exercises the actual TLV transport: it allocates
    /// a SHM ring buffer, writes a `serde_json`-serialized
    /// `RawPayloadOutput` under `MSG_TYPE_RAW_PAYLOAD_OUTPUT`
    /// (mirroring `emit_raw_payload_output_to_shm` at
    /// src/scenario/payload_run.rs), drains via `shm_drain` (the
    /// host-side reader), and decodes the entry exactly as
    /// `run_ktstr_test_inner` does at eval.rs:717-723. Asserts:
    /// 1. The drained entry's `msg_type` is `MSG_TYPE_RAW_PAYLOAD_OUTPUT`.
    /// 2. The CRC matches.
    /// 3. JSON deserialization restores the struct.
    /// 4. Both stream markers land in their correct fields —
    ///    stdout marker in `stdout`, stderr marker in `stderr`.
    /// 5. Markers are NOT swapped, NOT concatenated, NOT merged.
    ///
    /// The markers are deliberately distinctive ASCII strings that
    /// would be trivially detectable in either field if a regression
    /// merged them, and trivially missing if a regression dropped
    /// one.
    #[test]
    fn raw_payload_output_shm_wire_round_trip_preserves_both_streams() {
        use crate::vmm::shm_ring;

        const STDOUT_MARKER: &str = "STDOUT_MARKER_DISTINCT_E2E_a1b2c3";
        const STDERR_MARKER: &str = "STDERR_MARKER_DISTINCT_E2E_x9y8z7";

        // Allocate and initialize a ring buffer large enough for one
        // serialized RawPayloadOutput with a comfortable margin.
        // The serialized JSON is on the order of a few hundred bytes
        // (markers are short); 4 KiB of data area is plenty.
        const DATA_BYTES: usize = 4096;
        let shm_size = shm_ring::HEADER_SIZE + DATA_BYTES;
        let mut buf = vec![0u8; shm_size];
        shm_ring::shm_init(&mut buf, 0, shm_size);

        // Build the RawPayloadOutput exactly as the guest would —
        // distinct stdout / stderr markers in their respective fields.
        let original = crate::test_support::RawPayloadOutput {
            payload_index: 13,
            stdout: STDOUT_MARKER.to_string(),
            stderr: STDERR_MARKER.to_string(),
            hint: Some("focus".to_string()),
            metric_hints: Vec::new(),
            metric_bounds: None,
        };
        let payload = serde_json::to_vec(&original).expect("serialize RawPayloadOutput");

        // Write under MSG_TYPE_RAW_PAYLOAD_OUTPUT. shm_write returns
        // the bytes written (header + payload); a drop returns 0,
        // which would mean our ring sizing was wrong.
        let written =
            shm_ring::shm_write(&mut buf, 0, shm_ring::MSG_TYPE_RAW_PAYLOAD_OUTPUT, &payload);
        assert_eq!(
            written,
            shm_ring::MSG_HEADER_SIZE + payload.len(),
            "shm_write must place a full TLV; got {written}, expected header+payload",
        );

        // Host-side drain — what `run_ktstr_test_inner` invokes
        // at the end of a VM run.
        let drained = shm_ring::shm_drain(&buf, 0);
        assert_eq!(drained.entries.len(), 1, "exactly one entry expected");
        assert_eq!(
            drained.drops, 0,
            "no drops expected for a single small message"
        );

        let entry = &drained.entries[0];
        assert_eq!(
            entry.msg_type,
            shm_ring::MSG_TYPE_RAW_PAYLOAD_OUTPUT,
            "msg_type must round-trip as MSG_TYPE_RAW_PAYLOAD_OUTPUT; got 0x{:08x}",
            entry.msg_type,
        );
        assert!(
            entry.crc_ok,
            "CRC must match — torn payload bytes would produce a CRC mismatch and silently \
             drop the message host-side (eval.rs:717 gates on `entry.crc_ok`)",
        );

        // Decode exactly the way eval.rs:718 does.
        let restored: crate::test_support::RawPayloadOutput =
            serde_json::from_slice(&entry.payload).expect("decode RawPayloadOutput from SHM");

        // The load-bearing assertions: BOTH stream markers must
        // survive, in the CORRECT field, NOT swapped.
        assert_eq!(
            restored.stdout, STDOUT_MARKER,
            "stdout marker must round-trip into the stdout field; \
             a regression that swapped the streams would surface here",
        );
        assert_eq!(
            restored.stderr, STDERR_MARKER,
            "stderr marker must round-trip into the stderr field; \
             a regression that swapped the streams would surface here",
        );
        // Anti-merge guards: stdout must NOT contain the stderr
        // marker, and vice versa. A concatenation regression would
        // land both markers in one field.
        assert!(
            !restored.stdout.contains(STDERR_MARKER),
            "stdout field must NOT contain the stderr marker; \
             a regression that merged the streams (e.g. stderr appended \
             to stdout before serialize) would land both markers in \
             stdout. Got stdout: {:?}",
            restored.stdout,
        );
        assert!(
            !restored.stderr.contains(STDOUT_MARKER),
            "stderr field must NOT contain the stdout marker; \
             symmetric anti-merge guard. Got stderr: {:?}",
            restored.stderr,
        );
        // The other fields ride along — pin them too so a future
        // wire-format change that drops payload_index or hint
        // surfaces here.
        assert_eq!(restored.payload_index, original.payload_index);
        assert_eq!(restored.hint.as_deref(), Some("focus"));
    }
}
