//! Integration-level smoke tests for the public [`fixtures`] module.
//!
//! These tests exercise the fixtures exactly as a downstream crate
//! would — via the public re-export path
//! `ktstr::test_support::fixtures::{FIO, FIO_JSON, STRESS_NG}` — so a
//! silent drop-in-visibility change (e.g. accidentally making
//! `fixtures` private in `test_support/mod.rs`) surfaces here
//! instead of manifesting as a compile error in user code.
//!
//! Lives in its own test crate rather than `ktstr_test_macro.rs`
//! because that file's `#[ktstr_test]` entries drive nextest's
//! `--list` through `ktstr_main`, hiding plain `#[test]` functions.
//! Isolating the fixture smoke tests here keeps them visible to the
//! standard Rust test harness.
//!
//! [`fixtures`]: ktstr::test_support::fixtures

use ktstr::test_support::fixtures::{FIO, FIO_JSON, STRESS_NG};
use ktstr::test_support::{Check, OutputFormat, PayloadKind, Polarity, extract_metrics};

/// The public fixtures are reachable from a downstream crate via
/// the documented path. This test would fail at the `use` line
/// above if `fixtures` were accidentally demoted to `pub(crate)`.
#[test]
fn fixtures_are_publicly_reachable() {
    assert_eq!(FIO.name, "fio");
    assert_eq!(FIO_JSON.name, "fio_json");
    assert_eq!(STRESS_NG.name, "stress-ng");
}

/// End-to-end: feed realistic fio JSON stdout to `extract_metrics`
/// using `FIO.output`, verify the canonical metric paths land with
/// the expected values. Mirrors the in-module smoke test but goes
/// through the public API surface.
#[test]
fn fio_fixture_extract_metrics_from_public_api() {
    let stdout = r#"{"jobs": [{
        "jobname": "pub_api",
        "read":  {"iops": 50000.0, "lat_ns": {"mean": 200.0}},
        "write": {"iops":    10.0, "lat_ns": {"mean": 1500.0}}
    }]}"#;
    let metrics = extract_metrics(stdout, &FIO.output);

    let by_name: std::collections::BTreeMap<&str, f64> =
        metrics.iter().map(|m| (m.name.as_str(), m.value)).collect();

    assert_eq!(by_name.get("jobs.0.read.iops"), Some(&50000.0));
    assert_eq!(by_name.get("jobs.0.write.iops"), Some(&10.0));
    assert_eq!(by_name.get("jobs.0.read.lat_ns.mean"), Some(&200.0));
    assert_eq!(by_name.get("jobs.0.write.lat_ns.mean"), Some(&1500.0));
}

/// FIO_JSON bakes `--output-format=json` — a downstream test
/// relying on that for the common "just emit JSON" path should
/// see it.
#[test]
fn fio_json_default_args_include_output_format() {
    assert!(
        FIO_JSON.default_args.contains(&"--output-format=json"),
        "FIO_JSON must bake --output-format=json; got: {:?}",
        FIO_JSON.default_args,
    );
}

/// STRESS_NG uses ExitCode output — emissions carry zero metrics.
#[test]
fn stress_ng_fixture_outputs_nothing_from_stdout() {
    let metrics = extract_metrics("anything at all", &STRESS_NG.output);
    assert!(metrics.is_empty());
}

/// Both fixtures are binary-kind (`PayloadKind::Binary`), not
/// scheduler-kind. The runtime rejects scheduler-kind Payloads from
/// `CgroupDef::workload` at declaration time; this test pins the
/// `is_scheduler()` predicate so a future Payload-field
/// rearrangement that accidentally swaps the kind stays visible
/// from the consumer perspective.
#[test]
fn fixtures_are_binary_kind_not_scheduler_kind() {
    assert!(matches!(FIO.kind, PayloadKind::Binary(_)));
    assert!(matches!(FIO_JSON.kind, PayloadKind::Binary(_)));
    assert!(matches!(STRESS_NG.kind, PayloadKind::Binary(_)));
    assert!(!FIO.is_scheduler());
    assert!(!FIO_JSON.is_scheduler());
    assert!(!STRESS_NG.is_scheduler());
}

/// Polarity hints flow through `extract_metrics` via the
/// `resolve_polarities` pass only when the payload is run through
/// `PayloadRun::run` — `extract_metrics` itself emits unhinted
/// metrics. Pin that invariant from the consumer-visible side so
/// a silent change to `extract_metrics` shape is caught here.
#[test]
fn extract_metrics_does_not_apply_polarity_hints() {
    let stdout = r#"{"jobs":[{"read":{"iops": 1.0}}]}"#;
    let metrics = extract_metrics(stdout, &FIO.output);
    assert_eq!(metrics.len(), 1);
    assert_eq!(metrics[0].name, "jobs.0.read.iops");
    // The hint says HigherBetter + "iops"; extract_metrics leaves
    // it at Unknown because polarity resolution is a PayloadRun-
    // side pass, not an extract_metrics-side one.
    assert_eq!(metrics[0].polarity, Polarity::Unknown);
    assert_eq!(metrics[0].unit, "");
}

/// Compile-time invariant: both fixtures carry the exit-code gate
/// as their first default check. Declared as a `const` block so
/// the assertion runs at compile time — a silent drift of
/// `default_checks[0]` breaks the build, not a test.
#[test]
fn fixtures_default_checks_pin_exit_code_gate() {
    const _: () = {
        assert!(matches!(FIO.default_checks[0], Check::ExitCodeEq(0)));
        assert!(matches!(FIO_JSON.default_checks[0], Check::ExitCodeEq(0)));
        assert!(matches!(STRESS_NG.default_checks[0], Check::ExitCodeEq(0)));
    };
    // Also assert at runtime so the test name shows as pass rather
    // than no-op in nextest output.
    assert!(!FIO.default_checks.is_empty());
    assert!(!FIO_JSON.default_checks.is_empty());
    assert!(!STRESS_NG.default_checks.is_empty());
}

/// Identity-tag both fixtures' output formats so a consumer reading
/// this file sees the two cases side-by-side — Json vs ExitCode —
/// the canonical two-ends-of-the-spectrum demonstration #7 ships
/// with.
#[test]
fn fixture_output_formats_span_json_and_exit_code() {
    assert!(matches!(FIO.output, OutputFormat::Json));
    assert!(matches!(FIO_JSON.output, OutputFormat::Json));
    assert!(matches!(STRESS_NG.output, OutputFormat::ExitCode));
}
