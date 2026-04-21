//! Ready-made [`Payload`](ktstr::Payload) fixtures for the
//! benchmark binaries that dominate scheduler-regression testing:
//! `fio` (disk IO throughput, emits JSON), `stress-ng`
//! (synthetic CPU/memory stressors, exit-code only), and
//! `schbench` (latency percentiles, routed through LlmExtract).
//!
//! Each fixture is declared via
//! [`#[derive(Payload)]`](ktstr::Payload), the same path downstream
//! test authors use — so this module doubles as an end-to-end
//! exercise of the derive macro. The emitted `const` follows the
//! derive's naming convention: `struct FooPayload` produces
//! `const FOO: Payload`.
//!
//! These fixtures live under `tests/common/` rather than inside the
//! library's `src/` tree because they are TEST SCAFFOLDING, not
//! shipped API. A downstream scheduler author who wants the same
//! `fio` / `stress-ng` / `schbench` shapes should either copy the
//! declarations below into their own crate or write their own via
//! `#[derive(Payload)]`. The library does not ship fio, stress-ng,
//! or schbench binaries — the `kind = PayloadKind::Binary(name)`
//! just declares the name; host-side include-files resolution picks
//! the path up at test time.
//!
//! The fixtures cover all three
//! [`OutputFormat`](ktstr::test_support::OutputFormat) variants
//! — plus the hinted subvariant of `LlmExtract`:
//!
//! - [`FIO`] and [`FIO_JSON`] declare `OutputFormat::Json` with a
//!   set of [`MetricHint`](ktstr::test_support::MetricHint)s
//!   describing the canonical read/write throughput + latency paths.
//!   Extracted metrics land with correct polarity/unit automatically.
//! - [`STRESS_NG`] uses `OutputFormat::ExitCode` with a single
//!   `exit_code_eq(0)` default — stress-ng reports via exit code
//!   (bogo_ops land in stderr and are not machine-extractable
//!   without `--metrics-brief --yaml`).
//! - [`SCHBENCH`] uses `OutputFormat::LlmExtract(None)` — schbench
//!   emits human-readable percentile tables, so extraction is
//!   routed through the local LLM pipeline rather than the JSON
//!   walker.
//! - [`SCHBENCH_HINTED`] declares
//!   `OutputFormat::LlmExtract(Some("wakeup latency percentiles"))`
//!   — identical to [`SCHBENCH`] in every other field, exercising
//!   the derive's `LlmExtract("hint")` call form and the
//!   hint-threading path through
//!   [`extract_via_llm`](ktstr::test_support::model::extract_via_llm).
//!
//! All fixtures use short, stable `name` fields matching their
//! binary names — except FIO_JSON (`"fio_json"`) and
//! SCHBENCH_HINTED (`"schbench_hinted"`), which use distinct
//! names so they can coexist with FIO and SCHBENCH respectively
//! under the pairwise-dedup rule on `#[ktstr_test(workloads =
//! [...])]`. The binary names themselves (`"fio"`, `"stress-ng"`,
//! `"schbench"`) are what ktstr's include-files infrastructure
//! resolves inside the guest.
//!
//! # Polarity::Unknown downstream
//!
//! Metrics extracted from a hinted payload are matched against the
//! payload's `metrics` table by name in
//! [`PayloadRun`](ktstr::scenario::payload_run::PayloadRun)'s post-exit pipeline;
//! names with no matching hint land with
//! [`Polarity::Unknown`](ktstr::test_support::Polarity::Unknown) and
//! an empty unit. Unknown propagates as follows:
//!
//! - **`Check` assertion pass** — [`Check`](ktstr::test_support::Check)
//!   variants (`Min`, `Max`, `Range`, `Exists`, `ExitCodeEq`) compare
//!   values to thresholds without consulting polarity. An Unknown
//!   metric fails checks the same way a hinted metric does; polarity
//!   plays no role at assert time.
//! - **`AssertResult::merge` per-key worst-case** — when multiple
//!   cgroups contribute the same ext_metric, the merge consults the
//!   crate-internal `MetricDef` from the `METRICS` registry. Names
//!   absent from the registry (the case for any Unknown metric not
//!   also registered at crate scope) default to `higher_is_worse=true`
//!   and merge by taking the max — conservative for regressions, but
//!   NOT a declared polarity for the metric.
//! - **`cargo ktstr test-stats` cross-run comparison** — the
//!   crate-internal `compare_runs` iterates the `METRICS` registry
//!   only, so Unknown metrics extracted purely via `MetricHint`
//!   absence are NOT classified as regression or improvement. They
//!   are recorded to the sidecar for later manual inspection; to
//!   surface them in a comparison verdict, register a `MetricDef` in
//!   `src/stats.rs` or add a `MetricHint` on the payload with an
//!   explicit polarity.

use ktstr::Payload;

/// `fio` — flexible IO tester. Canonical workload for disk/IO
/// scheduler regressions.
///
/// Output format: JSON. Supply `--output-format=json` at the call
/// site (via `.arg(...)` on the
/// [`PayloadRun`](ktstr::scenario::payload_run::PayloadRun) builder returned by
/// `ctx.payload(&FIO)`, or via a scheduler default_args entry) or
/// use [`FIO_JSON`] which bakes it into `default_args` for the
/// common "just give me metrics" path.
///
/// **Caveat:** `FIO` leaves `default_args` empty, so invoking it
/// without `--output-format=json` causes `fio` to emit its
/// human-readable output, `extract_metrics` finds no JSON region,
/// and the check pass records every referenced metric as missing
/// without otherwise failing. Prefer [`FIO_JSON`] unless the test
/// author intentionally overrides the output mode.
///
/// Metric hints cover the first-job read/write leaf names. Fio's
/// JSON output is deeply nested (`jobs[N].read.iops`,
/// `.write.iops`, `.read.lat_ns.mean`, etc.); the hints pin the
/// four most-commonly-asserted paths. Unhinted paths land as
/// [`Polarity::Unknown`](ktstr::test_support::Polarity::Unknown)
/// and are still extracted for sidecar regression tracking.
#[derive(Payload)]
#[payload(binary = "fio", output = Json)]
#[default_check(exit_code_eq(0))]
#[metric(name = "jobs.0.read.iops", polarity = HigherBetter, unit = "iops")]
#[metric(name = "jobs.0.write.iops", polarity = HigherBetter, unit = "iops")]
#[metric(name = "jobs.0.read.lat_ns.mean", polarity = LowerBetter, unit = "ns")]
#[metric(name = "jobs.0.write.lat_ns.mean", polarity = LowerBetter, unit = "ns")]
#[allow(dead_code)]
pub struct FioPayload;

/// `fio` with `--output-format=json` pre-baked into `default_args`.
///
/// Compared to [`FIO`], this fixture differs in exactly two
/// fields:
///
/// 1. **`name`** — `"fio_json"` instead of `"fio"`. Uses a
///    distinct name so sidecar files and log output can
///    disambiguate the two fixtures. The `binary` field (the name
///    resolved by the include-files infrastructure) is still
///    `"fio"` in both.
/// 2. **`default_args`** — `&["--output-format=json"]` instead of
///    `&[]`. Everything else — `kind`, `output`, `default_checks`,
///    `metrics` — is character-for-character identical to [`FIO`].
///
/// **Caveat: simultaneous FIO + FIO_JSON.** Both fixtures have
/// `kind = PayloadKind::Binary("fio")`, so a scenario that lists
/// `#[ktstr_test(workloads = [FIO, FIO_JSON])]` spawns the `fio`
/// binary TWICE — each with its own argv set, inside whatever
/// cgroup the framework places each fixture in. The pairwise-dedup
/// on the `workloads` attribute only rejects identical Payload
/// paths; two distinct Payload constants that happen to share a
/// binary are NOT deduped. Test authors who want the same fio
/// binary once should pick ONE of the two fixtures, and extend it
/// via `ctx.payload(&FIO).arg("--output-format=json")` if the
/// `FIO_JSON` preset's args don't match their scenario.
#[derive(Payload)]
#[payload(binary = "fio", name = "fio_json", output = Json)]
#[default_args("--output-format=json")]
#[default_check(exit_code_eq(0))]
#[metric(name = "jobs.0.read.iops", polarity = HigherBetter, unit = "iops")]
#[metric(name = "jobs.0.write.iops", polarity = HigherBetter, unit = "iops")]
#[metric(name = "jobs.0.read.lat_ns.mean", polarity = LowerBetter, unit = "ns")]
#[metric(name = "jobs.0.write.lat_ns.mean", polarity = LowerBetter, unit = "ns")]
#[allow(dead_code)]
pub struct FioJsonPayload;

/// `stress-ng` — synthetic load generator (CPU, memory, IO, VM,
/// etc.). Canonical workload for exercising scheduler decisions
/// under configurable contention.
///
/// Output format: `ExitCode`. stress-ng emits human-readable
/// progress lines to stderr; metrics require `--metrics-brief
/// --yaml`, which produces YAML on stderr (not stdout). Since the
/// extraction pipeline only consumes stdout, even `--yaml` does
/// not currently feed `extract_metrics`; the fixture stays in
/// exit-code mode and the happy path is a zero exit.
///
/// **Caveat:** `default_args` is empty, so invoking `STRESS_NG`
/// without at least one stressor flag (e.g. `--cpu 1`, `--vm 1`)
/// causes stress-ng to print usage and exit nonzero on some
/// versions. Always append a stressor via `.arg(...)` on the
/// [`PayloadRun`](ktstr::scenario::payload_run::PayloadRun) builder returned
/// by `ctx.payload(&STRESS_NG)`.
///
/// Tests that want bogo_ops/sec metrics should declare their own
/// custom `Payload` via [`#[derive(Payload)]`](ktstr::Payload) and
/// pair it with a post-hoc stderr-to-stdout bridge, or wait for
/// the LlmExtract backend to land and declare
/// `output = LlmExtract("bogo ops")`.
#[derive(Payload)]
#[payload(binary = "stress-ng")]
#[default_check(exit_code_eq(0))]
#[allow(dead_code)]
pub struct StressNgPayload;

/// Latency-focused scheduler benchmark. Uses `LlmExtract` to
/// exercise the LLM extraction pipeline (schbench supports `--json`
/// but this fixture intentionally uses the third acquisition path).
#[derive(Payload)]
#[payload(binary = "schbench", output = LlmExtract)]
#[default_args("--runtime", "30", "--message-threads", "2")]
#[default_check(exit_code_eq(0))]
#[allow(dead_code)]
pub struct SchbenchPayload;

/// Hint-carrying sibling of [`SCHBENCH`] — identical in every
/// field except `name` (uses a distinct name so sidecar files and
/// log output can disambiguate the two fixtures) and `output`.
///
/// Declares `output = LlmExtract("wakeup latency percentiles")`.
/// The derive macro translates the call form into
/// [`OutputFormat::LlmExtract(Some(...))`](ktstr::test_support::OutputFormat::LlmExtract),
/// and the stored `&'static str` is inserted between the template
/// and the stdout block as a `Focus:` directive by
/// [`extract_via_llm`](ktstr::test_support::model::extract_via_llm)
/// when the fixture runs — steering the model toward the stat the
/// scheduler regression cares about instead of whatever numeric
/// leaf the model picks first.
///
/// Exists as a fixture (rather than only as an ad-hoc
/// `#[derive(Payload)]` inside the test file) so downstream
/// scheduler-author crates have a copy-ready template for the
/// hint-carrying shape — the bare [`SCHBENCH`] covers the
/// no-hint form, this fixture covers the with-hint form.
#[derive(Payload)]
#[payload(
    binary = "schbench",
    name = "schbench_hinted",
    output = LlmExtract("wakeup latency percentiles"),
)]
#[default_args("--runtime", "30", "--message-threads", "2")]
#[default_check(exit_code_eq(0))]
#[allow(dead_code)]
pub struct SchbenchHintedPayload;
