//! End-to-end validation of the `OutputFormat::LlmExtract` pipeline.
//!
//! Every test in this file boots a ktstr VM, runs `schbench` inside
//! with the stock [`SCHBENCH`] fixture (which emits its latency
//! tables and summary lines to stderr by default — see
//! `tests/common/fixtures.rs` for the `output = LlmExtract` contract),
//! ships the captured stderr text from the guest to the host across
//! the SHM ring as a `RawPayloadOutput`, and routes it through the
//! local-model extraction pipeline (`extract_via_llm`) on the HOST
//! after VM exit. The host then applies a fixed set of universal
//! invariants against the extracted metrics; any violation folds
//! into the test's `AssertResult` as an `AssertDetail`.
//!
//! Lives in its own integration-test binary (not `ktstr_test_macro.rs`)
//! because exercising the LLM backend pulls in the full model cache
//! — running the ~2.44 GiB `DEFAULT_MODEL` load and a multi-second
//! inference call — and isolating it keeps the cheap scheduler
//! tests free of that cost when filtering via nextest.
//!
//! **Host-only LLM extraction.** The model (~2.4 GiB GGUF) does
//! NOT load inside the guest VM: the test VM's RAM
//! budget cannot fit it, and the cache lives on the host. The
//! guest-side `evaluate()` skips every model code path for
//! `OutputFormat::LlmExtract` payloads, ships the raw
//! stdout/stderr across the SHM ring, and the host's
//! `eval.rs::host_side_llm_extract` runs `extract_via_llm`
//! post-VM-exit. As a consequence, `ctx.payload(&SCHBENCH).run()`
//! returns a `PayloadMetrics` with `metrics: vec![]` inside the
//! guest test body — extraction is deferred. The body therefore
//! cannot inspect individual metrics here. The framework owns the
//! sanity checks below.
//!
//! **Universal structural-sanity checks enforced host-side**:
//! 1. Every metric name is unique (duplicate dotted paths imply
//!    the LLM walker emitted the same key twice — a walker
//!    aggregation bug or malformed JSON path that would
//!    misattribute downstream stats).
//! 2. Every value is finite (no NaN / ±inf leaking into
//!    PayloadMetrics).
//! 3. Every metric carries `MetricSource::LlmExtract` (drift here
//!    points at a bypass: the value reached the LlmExtract slot
//!    without traversing the LLM walker).
//!
//! Workload-specific assertions (minimum metric count, sign,
//! magnitude bounds, semantic ranges) are intentionally NOT
//! enforced at the framework level — those vary per payload
//! (schbench's > 5 latency rows vs a hypothetical single-throughput
//! benchmark, or schbench's non-negative microseconds vs a
//! delta-emitting payload that legitimately reports negative deltas)
//! and require a per-payload validation API that ktstr does not yet
//! expose. See `eval.rs::validate_llm_extraction` for the host-side
//! enforcement.
//!
//! **Stability disclaimer**: passing this test does NOT mean
//! `LlmExtract` output is run-to-run stable for regression
//! comparisons. The invariants above pin only structural sanity,
//! not the extracted values or names. For stable schemas suitable
//! for run-to-run comparison and regression classification, use
//! [`SCHBENCH_JSON`](common::fixtures::SCHBENCH_JSON) with
//! `OutputFormat::Json` — the dotted-path schema lives in
//! schbench's `write_json_stats` and is fixed by the schbench
//! source, independent of the model.
//!
//! Model availability: tests here lazy-load the LLM model on the
//! first `extract_via_llm` invocation (see `load_inference` in
//! `src/test_support/model.rs`). With `KTSTR_MODEL_OFFLINE=1` and a
//! cold cache, the load fails and `host_side_llm_extract` appends an
//! `LlmExtract model load failed` detail. Inference itself is
//! multi-minute on host CPU regardless of cache state, so the
//! `test(model_loaded_)` nextest override at `.config/nextest.toml`
//! extends the slow-timeout to cover EVERY run; a cold cache
//! additionally pays the GGUF download cost on top of that.

mod common;

use anyhow::Result;
use common::fixtures::SCHBENCH;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::scenario::Ctx;

/// Run schbench under the [`SCHBENCH`] fixture and validate the
/// [`OutputFormat::LlmExtract`](ktstr::test_support::OutputFormat::LlmExtract)
/// pipeline.
///
/// `llcs = 1, cores = 2, threads = 1, memory_mb = 2048`: schbench
/// wants at least one messenger + one worker thread, so two logical
/// CPUs is the minimum topology that gives it room to measure wake
/// latency. 2048 MiB memory_mb matches the other in-VM benchmark
/// tests and leaves headroom for schbench's 2 MiB per-thread shm
/// allocations plus kernel overhead.
///
/// Test body shape: returns the `AssertResult` from
/// `ctx.payload(&SCHBENCH).run()` directly. The metric-set sanity
/// checks live host-side in `eval.rs::host_side_llm_extract` —
/// extraction is deferred until after VM exit because the model
/// does not fit in guest RAM. See the module doc for the universal
/// invariants the host applies.
#[ktstr_test(llcs = 1, cores = 2, threads = 1, memory_mb = 2048, payload = SCHBENCH)]
fn model_loaded_llm_extract_schbench(ctx: &Ctx) -> Result<AssertResult> {
    let (assert_result, _metrics) = ctx.payload(&SCHBENCH).run()?;
    // For `OutputFormat::LlmExtract` payloads, `metrics.metrics` is
    // intentionally empty inside the guest body — extraction is
    // deferred host-side. Forwarding `assert_result` lets the host's
    // `host_side_llm_extract` populate the metric set, apply the
    // universal invariants, and fold any failure details into the
    // final test verdict. A non-passing `assert_result` here means
    // the guest's `Check::ExitCodeEq(0)` pre-pass on SCHBENCH already
    // detected a non-zero exit; that detail surfaces unchanged.
    Ok(assert_result)
}
