//! Ready-made [`Payload`](super::Payload) fixtures for the two
//! benchmark binaries that dominate scheduler-regression testing:
//! `fio` (disk IO throughput, emits JSON) and `stress-ng`
//! (synthetic CPU/memory stressors, exit-code only).
//!
//! Each fixture is declared via [`#[derive(Payload)]`](crate::Payload),
//! the same path downstream test authors use — so this module
//! doubles as an end-to-end exercise of the derive macro. The
//! emitted `const` follows the derive's naming convention:
//! `struct FooPayload` produces `const FOO: Payload`.
//!
//! Test authors import these and either pass them directly as
//! `#[ktstr_test(payload = FIO, ...)]` or run them through the
//! [`PayloadRun`](super::super::scenario::payload_run::PayloadRun)
//! builder for custom arg/check composition. The fixtures
//! illustrate the two ends of the [`OutputFormat`](super::OutputFormat)
//! spectrum:
//!
//! - [`FIO`] and [`FIO_JSON`] declare `OutputFormat::Json` with a
//!   set of [`MetricHint`](super::MetricHint)s describing the
//!   canonical read/write throughput + latency paths. Extracted
//!   metrics land with correct polarity/unit automatically.
//! - [`STRESS_NG`] uses `OutputFormat::ExitCode` with a single
//!   `exit_code_eq(0)` default — stress-ng reports via exit code
//!   (bogo_ops land in stderr and are not machine-extractable
//!   without `--metrics-brief --yaml`).
//!
//! Both fixtures use short, stable `name` fields (`"fio"`,
//! `"stress-ng"`) matching the binary names that ktstr's
//! include-files infrastructure resolves inside the guest.
//!
//! # Extending
//!
//! These fixtures are intentionally minimal. Test authors who want
//! a tuned fio job can do:
//!
//! ```rust,no_run
//! use ktstr::prelude::*;
//! use ktstr::test_support::fixtures::FIO_JSON;
//!
//! # fn stub(ctx: &Ctx) -> Result<AssertResult> {
//! ctx.payload(&FIO_JSON)
//!     .arg("--rw=randread")
//!     .arg("--bs=4k")
//!     .arg("--runtime=30")
//!     .check(Check::min("jobs.0.read.iops", 1000.0))
//!     .run()?;
//! # Ok(AssertResult::pass())
//! # }
//! ```
//!
//! The example uses [`FIO_JSON`] rather than [`FIO`] because fio
//! does not emit JSON on stdout without `--output-format=json`.
//! `FIO` leaves `default_args` empty so callers who want a
//! different fio output mode (terse, normal, ...) can plug one in
//! without fighting a pre-baked flag; `FIO_JSON` is the opinionated
//! "just give me metrics" convenience.
//!
//! Per-test `.arg` / `.check` appends on top of the fixture's
//! defaults; use `.clear_args()` / `.clear_checks()` to start from
//! a blank slate.

use crate::Payload;

/// `fio` — flexible IO tester. Canonical workload for disk/IO
/// scheduler regressions.
///
/// Output format: JSON. Supply `--output-format=json` at the call
/// site (via `.arg(...)` on the runtime builder, or via a scheduler
/// default_args entry) or use [`FIO_JSON`] which bakes it into
/// `default_args` for the common "just give me metrics" path.
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
/// [`Polarity::Unknown`](super::Polarity::Unknown) and are still
/// extracted for sidecar regression tracking.
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
/// 1. **`name`** — `"fio_json"` instead of `"fio"`. The distinct
///    name lets both fixtures register as workloads in the same
///    `#[ktstr_test(workloads = [FIO, FIO_JSON])]` attribute
///    without hitting the pairwise-dedup rejection. The `binary`
///    field (the name resolved by the include-files
///    infrastructure) is still `"fio"` in both.
/// 2. **`default_args`** — `&["--output-format=json"]` instead of
///    `&[]`. Everything else — `kind`, `output`, `default_checks`,
///    `metrics` — is character-for-character identical to
///    [`FIO`]. See the unit tests for the pinned invariants.
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
/// runtime builder.
///
/// Tests that want bogo_ops/sec metrics should declare their own
/// custom `Payload` via [`#[derive(Payload)]`](crate::Payload) and
/// pair it with a post-hoc stderr-to-stdout bridge, or wait for
/// the LlmExtract backend to land and declare
/// `output = LlmExtract("bogo ops")`.
#[derive(Payload)]
#[payload(binary = "stress-ng")]
#[default_check(exit_code_eq(0))]
#[allow(dead_code)]
pub struct StressNgPayload;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{
        Check, MetricHint, MetricSource, OutputFormat, PayloadKind, Polarity, extract_metrics,
    };

    /// FIO is a binary-kind payload named "fio" with JSON output.
    /// Pins the identity fields so a test author can rely on them.
    #[test]
    fn fio_identity_fields_are_stable() {
        assert_eq!(FIO.name, "fio");
        assert!(matches!(FIO.kind, PayloadKind::Binary("fio")));
        assert!(matches!(FIO.output, OutputFormat::Json));
    }

    /// FIO's default_checks include the exit-code gate so a
    /// misconfigured invocation fails loudly instead of silently
    /// landing zero metrics.
    #[test]
    fn fio_default_checks_include_exit_code_zero() {
        assert_eq!(FIO.default_checks.len(), 1);
        assert!(matches!(FIO.default_checks[0], Check::ExitCodeEq(0)));
    }

    /// FIO's metric hints cover iops (higher-better) and lat_ns
    /// (lower-better) for both read and write sides of job 0. The
    /// test asserts every hint's polarity + unit so a silent drift
    /// in the MetricHint shape surfaces here, not in downstream
    /// regression reports.
    #[test]
    fn fio_metric_hints_cover_canonical_paths() {
        let by_name: std::collections::BTreeMap<&str, &MetricHint> =
            FIO.metrics.iter().map(|m| (m.name, m)).collect();

        let rops = by_name.get("jobs.0.read.iops").expect("read iops hint");
        assert_eq!(rops.polarity, Polarity::HigherBetter);
        assert_eq!(rops.unit, "iops");

        let wops = by_name.get("jobs.0.write.iops").expect("write iops hint");
        assert_eq!(wops.polarity, Polarity::HigherBetter);
        assert_eq!(wops.unit, "iops");

        let rlat = by_name
            .get("jobs.0.read.lat_ns.mean")
            .expect("read lat hint");
        assert_eq!(rlat.polarity, Polarity::LowerBetter);
        assert_eq!(rlat.unit, "ns");

        let wlat = by_name
            .get("jobs.0.write.lat_ns.mean")
            .expect("write lat hint");
        assert_eq!(wlat.polarity, Polarity::LowerBetter);
        assert_eq!(wlat.unit, "ns");
    }

    /// FIO_JSON bakes `--output-format=json` into default_args.
    /// Distinct name + binary (same binary, different name for
    /// the fixture) so both fio fixtures can coexist in a
    /// `#[ktstr_test(workloads = [...])]` attribute without the
    /// pairwise dedup rejecting them.
    #[test]
    fn fio_json_identity_and_default_args() {
        assert_eq!(FIO_JSON.name, "fio_json");
        assert!(matches!(FIO_JSON.kind, PayloadKind::Binary("fio")));
        assert_eq!(FIO_JSON.default_args, &["--output-format=json"]);
        assert_eq!(FIO_JSON.metrics.len(), FIO.metrics.len());
    }

    /// STRESS_NG is an exit-code-only binary-kind payload with the
    /// exit-code gate as its sole default check.
    #[test]
    fn stress_ng_identity_fields_are_stable() {
        assert_eq!(STRESS_NG.name, "stress-ng");
        assert!(matches!(STRESS_NG.kind, PayloadKind::Binary("stress-ng")));
        assert!(matches!(STRESS_NG.output, OutputFormat::ExitCode));
        assert!(STRESS_NG.metrics.is_empty());
        assert_eq!(STRESS_NG.default_checks.len(), 1);
        assert!(matches!(STRESS_NG.default_checks[0], Check::ExitCodeEq(0),));
    }

    /// Smoke test: FIO's extraction pipeline produces Json-sourced
    /// metrics from a realistic fio JSON payload. Exercises the
    /// OutputFormat::Json branch of extract_metrics end-to-end
    /// against the fixture's declared output format.
    #[test]
    fn fio_extract_metrics_smoke_from_realistic_json() {
        let stdout = r#"{
          "jobs": [{
            "jobname": "example",
            "read":  {"iops": 12345.6, "lat_ns": {"mean": 500.0}},
            "write": {"iops": 78.9,    "lat_ns": {"mean": 2500.0}}
          }]
        }"#;
        let metrics = extract_metrics(stdout, &FIO.output);
        let by_name: std::collections::BTreeMap<&str, f64> =
            metrics.iter().map(|m| (m.name.as_str(), m.value)).collect();

        assert_eq!(by_name.get("jobs.0.read.iops"), Some(&12345.6));
        assert_eq!(by_name.get("jobs.0.write.iops"), Some(&78.9));
        assert_eq!(by_name.get("jobs.0.read.lat_ns.mean"), Some(&500.0));
        assert_eq!(by_name.get("jobs.0.write.lat_ns.mean"), Some(&2500.0));
        for m in &metrics {
            assert_eq!(
                m.source,
                MetricSource::Json,
                "fixture declares Json output; every metric must land with Json source tag"
            );
        }
    }

    /// Smoke test: STRESS_NG's exit-code format produces an empty
    /// metric set (check pass/fail is handled by the ExitCodeEq
    /// pre-pass, not by metric values).
    #[test]
    fn stress_ng_extract_metrics_smoke_returns_empty() {
        let metrics = extract_metrics("irrelevant stdout", &STRESS_NG.output);
        assert!(
            metrics.is_empty(),
            "ExitCode output emits no metrics; got {metrics:?}"
        );
    }

    /// Neither fixture is a scheduler-kind payload — they must not
    /// be accepted by CgroupDef::workload's scheduler-kind rejection
    /// gate (which panics at builder time).
    #[test]
    fn fixtures_are_not_scheduler_kind() {
        assert!(!FIO.is_scheduler());
        assert!(!FIO_JSON.is_scheduler());
        assert!(!STRESS_NG.is_scheduler());
    }
}
