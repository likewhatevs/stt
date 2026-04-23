//! End-to-end validation of the `OutputFormat::LlmExtract` pipeline.
//!
//! Every test in this file boots a ktstr VM, runs `schbench` inside
//! with the stock [`SCHBENCH`] fixture (which emits its latency
//! tables and summary lines to stderr by default — see
//! `tests/common/fixtures.rs` for the `output = LlmExtract` contract),
//! captures the stderr text on the host, and routes it through the
//! local-model extraction pipeline (`extract_via_llm`). The test
//! body then asserts the model returned a sane metric set against
//! the captured output.
//!
//! Lives in its own integration-test binary (not `ktstr_test_macro.rs`)
//! because exercising the LLM backend pulls in the full model cache
//! — running the ~2.44 GiB `DEFAULT_MODEL` load and a multi-second
//! inference call — and isolating it keeps the cheap scheduler
//! tests free of that cost when filtering via nextest.
//!
//! **Non-determinism policy**: LLM-extracted metric NAMES are model-
//! dependent, prompt-dependent, and weight-dependent; they can drift
//! between `Qwen3-4B Q4_K_M` and any successor model, and between
//! patch revisions of the same model. The assertions here therefore
//! pin only model-invariant shape properties:
//! 1. At least 5 metrics surface (schbench's latency tables + summary
//!    always emit dozens; a short extraction indicates a truncated
//!    model response).
//! 2. Every metric name is unique (duplicate dotted paths imply
//!    malformed JSON or a walker aggregation bug).
//! 3. Every metric carries `MetricSource::LlmExtract`.
//! 4. Every value is finite (no NaN / ±inf).
//! 5. Every value is non-negative (schbench emits latencies,
//!    percentiles, and RPS — all non-negative quantities).
//! 6. Every value is below 1e12 (a loose ceiling that catches
//!    hallucinated magnitudes without false-failing realistic
//!    schbench output).
//!
//! **Stability disclaimer**: passing this test does NOT mean
//! `LlmExtract` output is run-to-run stable for regression
//! comparisons. The assertions above pin only structural sanity,
//! not the extracted values or names. For stable schemas suitable
//! for run-to-run comparison and regression classification, use
//! [`SCHBENCH_JSON`](common::fixtures::SCHBENCH_JSON) with
//! `OutputFormat::Json` — the dotted-path schema lives in
//! schbench's `write_json_stats` and is fixed by the schbench
//! source, independent of the model.
//!
//! **Model availability**: tests here depend on `KTSTR_TESTS`'s
//! `any_test_requires_model() == true`, which triggers
//! `prefetch_if_required()` at nextest setup to populate the model
//! cache. With `KTSTR_MODEL_OFFLINE=1` and a cold cache, the model
//! load fails and the pipeline emits `LlmExtract model load failed:
//! <reason>` as a failed `AssertResult` — the test body forwards
//! that verdict via `if !assert_result.passed { return Ok(...) }`
//! rather than panicking.

mod common;

use anyhow::Result;
use common::fixtures::SCHBENCH;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::scenario::Ctx;
use ktstr::test_support::{LLM_DEBUG_RESPONSES_ENV, MetricSource};

/// Run schbench under the [`SCHBENCH`] fixture and verify the
/// [`OutputFormat::LlmExtract`](ktstr::test_support::OutputFormat::LlmExtract)
/// pipeline surfaces a sane metric set.
///
/// `llcs = 1, cores = 2, threads = 1, memory_mb = 2048`: schbench
/// wants at least one messenger + one worker thread, so two logical
/// CPUs is the minimum topology that gives it room to measure wake
/// latency. 2048 MiB memory_mb matches the other in-VM benchmark
/// tests and leaves headroom for schbench's 2 MiB per-thread shm
/// allocations plus kernel overhead.
///
/// Assertions pin MODEL-INVARIANT properties only (see module doc
/// for why):
/// 1. >= 5 metrics in the returned set.
/// 2. Every metric name is unique (no duplicate dotted paths).
/// 3. `source == MetricSource::LlmExtract` on every metric.
/// 4. `value.is_finite()` on every metric.
/// 5. `value >= 0.0` on every metric.
/// 6. `value < 1e12` on every metric.
#[ktstr_test(llcs = 1, cores = 2, threads = 1, memory_mb = 2048)]
fn llm_extract_schbench_surfaces_sane_metrics(ctx: &Ctx) -> Result<AssertResult> {
    let (assert_result, metrics) = ctx.payload(&SCHBENCH).run()?;
    // Forward the gate's verdict on any upstream failure:
    // - schbench exited non-zero (Check::ExitCodeEq(0) on SCHBENCH
    //   gates this);
    // - or the LlmExtract backend could not load the model
    //   (e.g. KTSTR_MODEL_OFFLINE=1 with a cold cache, or a
    //   corrupt cached GGUF that SHA mismatches).
    // Both surface as `assert_result.passed == false` with an
    // actionable reason; returning the AssertResult preserves the
    // upstream diagnostic instead of reducing it to a pass/fail bit.
    if !assert_result.passed {
        return Ok(assert_result);
    }

    // A successful schbench run + LLM inference produces dozens of
    // numeric leaves (wakeup/request latency percentiles at p50/p90/
    // p95/p99/p99.9, avg_rps, sched_delay_mean, etc. — schbench's
    // `show_latencies` table alone is typically 7+ metrics per
    // section across 2+ sections). A floor of 5 catches a model
    // that degraded to emitting one or two fields of a partial
    // response while staying well below the actual minimum output.
    // `> 0` would pass a broken extraction that only surfaces a
    // single hallucinated scalar.
    const MIN_METRICS: usize = 5;
    if metrics.metrics.len() < MIN_METRICS {
        return Ok(AssertResult::fail_other(format!(
            "LlmExtract produced {} metric(s), expected >= {MIN_METRICS}. \
             Schbench's latency tables + summary emit dozens of numeric \
             leaves; a short extraction indicates the model truncated \
             its response or returned a sparse JSON. Check tracing::warn \
             from extract_via_llm for details; set {LLM_DEBUG_RESPONSES_ENV}=1 \
             to dump the raw model output.",
            metrics.metrics.len(),
        )));
    }

    // Unique metric names: `walk_json_leaves` tags each leaf with
    // its dotted JSON path, so two metrics sharing a name means the
    // model emitted the same dotted path twice in its JSON
    // response — indicating either a malformed / duplicate-key JSON
    // that should have been sanitized, or (more likely) a path
    // collision from a post-walker aggregation bug. Either way the
    // downstream stats pipeline will misattribute one value to the
    // other. Pinning uniqueness catches the class before it hits
    // comparison.
    {
        use std::collections::HashSet;
        let names: HashSet<&str> = metrics.metrics.iter().map(|m| m.name.as_str()).collect();
        if names.len() != metrics.metrics.len() {
            return Ok(AssertResult::fail_other(format!(
                "LlmExtract produced {} metric(s) but only {} unique name(s); \
                 duplicate metric names indicate the model emitted the same \
                 dotted path twice or a walker aggregation bug",
                metrics.metrics.len(),
                names.len(),
            )));
        }
    }

    // Loose absolute-value ceiling: schbench latencies are
    // microseconds (u64 fits comfortably in hundreds of millions on
    // a broken system), RPS rarely exceeds a few hundred thousand,
    // and sched_delay is microseconds. Any numeric leaf above
    // 1e12 (one trillion) is an absurd hallucination — the model
    // inventing an absurd exponent or sign-extending a parsed
    // literal. 1e12 is loose enough that no realistic schbench
    // output will false-fail, tight enough to catch invented
    // magnitudes that would poison downstream regression
    // comparisons.
    const MAX_VALUE: f64 = 1e12;

    for m in &metrics.metrics {
        // Source attribution: every metric must come from the LLM
        // path, not the Json or LLM-unrelated fallback paths. A
        // regression that mixed sources (e.g. a future walker
        // tag-propagation bug) would surface here before it
        // poisoned downstream stats aggregation.
        if m.source != MetricSource::LlmExtract {
            return Ok(AssertResult::fail_other(format!(
                "metric '{}' has source {:?}, expected MetricSource::LlmExtract",
                m.name, m.source,
            )));
        }
        // `walk_json_leaves` already filters non-finite numbers
        // (see its `f.is_finite()` gate) before emitting a
        // `Metric`. Pinning the invariant at the consumer side
        // guards against a future extraction path that bypasses
        // the walker — NaN / ±inf propagating into PayloadMetrics
        // poisons percentile comparisons downstream.
        if !m.value.is_finite() {
            return Ok(AssertResult::fail_other(format!(
                "metric '{}' has non-finite value {} — LlmExtract must \
                 not propagate NaN / ±inf",
                m.name, m.value,
            )));
        }
        // Schbench emits latencies (microseconds), percentiles,
        // and RPS — all non-negative. A negative value here
        // indicates an LLM hallucination (the model invented a
        // negative-signed number) or a walker sign-extension bug.
        // Either way, non-negative is a hard schbench-semantic
        // invariant that the test should pin.
        if m.value < 0.0 {
            return Ok(AssertResult::fail_other(format!(
                "metric '{}' has negative value {} — schbench emits \
                 latencies / percentiles / RPS, all non-negative",
                m.name, m.value,
            )));
        }
        // Loose absolute-value ceiling (see MAX_VALUE const above).
        // Catches hallucinated magnitudes without pinning specific
        // names — LLM extractions that invent a 1e15 latency or a
        // 1e20 throughput fail here before they reach stats
        // comparison.
        if m.value > MAX_VALUE {
            return Ok(AssertResult::fail_other(format!(
                "metric '{}' has value {} > {MAX_VALUE:e} — exceeds the \
                 loose ceiling for realistic schbench output; likely an \
                 LLM hallucination",
                m.name, m.value,
            )));
        }
    }

    Ok(AssertResult::pass())
}
