//! Integration smoke tests for the FIO / FIO_JSON / STRESS_NG /
//! SCHBENCH / SCHBENCH_HINTED payload fixtures.
//!
//! The fixtures themselves live under `tests/common/fixtures.rs`
//! because they are test scaffolding, not shipped API. This test
//! file exercises the declarations the same way a downstream
//! scheduler-author test crate would — via plain `#[derive(Payload)]`
//! on local consts.
//!
//! Lives in its own test crate rather than `ktstr_test_macro.rs`
//! because that file's `#[ktstr_test]` entries drive nextest's
//! `--list` through `ktstr_main`, hiding plain `#[test]` functions.
//! Isolating the fixture smoke tests here keeps them visible to the
//! standard Rust test harness.

mod common;

use common::fixtures::{FIO, FIO_JSON, SCHBENCH, SCHBENCH_HINTED, STRESS_NG};
use ktstr::test_support::{
    Check, MetricHint, MetricSource, MetricStream, OutputFormat, PayloadKind, Polarity,
    extract_metrics,
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
    let metrics = extract_metrics(stdout, &FIO.output, MetricStream::Stdout).unwrap();
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
    let metrics =
        extract_metrics("irrelevant stdout", &STRESS_NG.output, MetricStream::Stdout).unwrap();
    assert!(
        metrics.is_empty(),
        "ExitCode output emits no metrics; got {metrics:?}"
    );
}

/// SCHBENCH identity fields are pinned so tests relying on the
/// fixture can detect silent drift.
#[test]
fn schbench_identity_fields_are_stable() {
    assert_eq!(SCHBENCH.name, "schbench");
    assert!(matches!(SCHBENCH.kind, PayloadKind::Binary("schbench")));
    assert!(matches!(SCHBENCH.output, OutputFormat::LlmExtract(None)));
    assert_eq!(
        SCHBENCH.default_args,
        &["--runtime", "5", "--message-threads", "2"]
    );
    assert!(SCHBENCH.metrics.is_empty());
    assert_eq!(SCHBENCH.default_checks.len(), 1);
    assert!(matches!(SCHBENCH.default_checks[0], Check::ExitCodeEq(0)));
}

/// SCHBENCH_HINTED is the hint-carrying sibling of [`SCHBENCH`].
/// Its `output` must decode to `LlmExtract(Some(...))` with the
/// exact hint string baked into the fixture — the round-trip from
/// derive-macro call form (`LlmExtract("hint")`) through the
/// emitted `const` and into [`OutputFormat`] is the path this test
/// pins. A silent drop of the hint at the macro layer, a
/// re-interpretation at the emit layer, or a structural change
/// to [`OutputFormat`] that loses the `Option<&'static str>`
/// payload surfaces here, not at runtime inside extract_via_llm.
///
/// The hint string itself ("wakeup latency percentiles") is the
/// fixture's invariant: it is deliberately asserted by
/// value — not by `matches!(.., Some(_))` — so a future refactor
/// that accidentally substitutes, truncates, or duplicates the
/// hint breaks this assertion rather than landing a quietly
/// wrong prompt. Every other field (`kind`, `default_args`,
/// `default_checks`, `metrics`) must match SCHBENCH exactly,
/// since the two fixtures are defined to differ in only `name`
/// and `output`.
#[test]
fn schbench_hinted_output_carries_hint_through_derive() {
    assert_eq!(SCHBENCH_HINTED.name, "schbench_hinted");
    assert!(matches!(
        SCHBENCH_HINTED.kind,
        PayloadKind::Binary("schbench"),
    ));
    match SCHBENCH_HINTED.output {
        OutputFormat::LlmExtract(Some(hint)) => {
            assert_eq!(hint, "wakeup latency percentiles");
        }
        other => panic!("expected OutputFormat::LlmExtract(Some(hint)), got {other:?}",),
    }
    assert_eq!(
        SCHBENCH_HINTED.default_args, SCHBENCH.default_args,
        "hinted fixture must differ from SCHBENCH only in name and output",
    );
    assert_eq!(
        SCHBENCH_HINTED.metrics.len(),
        SCHBENCH.metrics.len(),
        "hinted fixture must differ from SCHBENCH only in name and output",
    );
    assert_eq!(
        SCHBENCH_HINTED.default_checks.len(),
        SCHBENCH.default_checks.len(),
        "hinted fixture must differ from SCHBENCH only in name and output",
    );
    assert!(SCHBENCH_HINTED.metrics.is_empty());
    assert_eq!(SCHBENCH_HINTED.default_checks.len(), 1);
    assert!(matches!(
        SCHBENCH_HINTED.default_checks[0],
        Check::ExitCodeEq(0),
    ));
}

/// SCHBENCH and SCHBENCH_HINTED share the same binary ("schbench")
/// but use distinct `name` fields. The `name` is threaded through
/// every log + error context in [`PayloadRun`](ktstr::scenario::payload_run::PayloadRun)
/// (e.g. `spawn payload '{name}'`, `reap payload '{name}'`,
/// `with_context(|| format!("… payload '{name}'"))` sites) and into
/// the `Debug` impl on `Payload`, so if both fixtures shared a name
/// a run that used both would emit log lines that could not be
/// attributed to the right fixture after the fact. Asserting the
/// distinction here pins the log-attribution contract so a rename
/// that collapsed the two fixtures into a single `name` breaks
/// this test rather than surfacing as ambiguous log output.
#[test]
fn schbench_and_schbench_hinted_have_distinct_names() {
    assert_ne!(SCHBENCH.name, SCHBENCH_HINTED.name);
    let PayloadKind::Binary(b1) = SCHBENCH.kind else {
        panic!("SCHBENCH must be a Binary-kind payload, not Scheduler");
    };
    let PayloadKind::Binary(b2) = SCHBENCH_HINTED.kind else {
        panic!("SCHBENCH_HINTED must be a Binary-kind payload, not Scheduler");
    };
    assert_eq!(b1, b2, "hinted fixture must point at same binary");
}

/// No fixture is a scheduler-kind payload — they must not
/// be accepted by CgroupDef::workload's scheduler-kind rejection
/// gate (which panics at builder time).
#[test]
fn fixtures_are_not_scheduler_kind() {
    assert!(!FIO.is_scheduler());
    assert!(!FIO_JSON.is_scheduler());
    assert!(!STRESS_NG.is_scheduler());
    assert!(!SCHBENCH.is_scheduler());
    assert!(!SCHBENCH_HINTED.is_scheduler());
}

/// Polarity hints flow through `extract_metrics` via the
/// `resolve_polarities` pass only when the payload is run through
/// `PayloadRun::run` — `extract_metrics` itself emits unhinted
/// metrics. Pin that invariant from the consumer-visible side so
/// a silent change to `extract_metrics` shape is caught here.
#[test]
fn extract_metrics_does_not_apply_polarity_hints() {
    let stdout = r#"{"jobs":[{"read":{"iops": 1.0}}]}"#;
    let metrics = extract_metrics(stdout, &FIO.output, MetricStream::Stdout).unwrap();
    assert_eq!(metrics.len(), 1);
    assert_eq!(metrics[0].name, "jobs.0.read.iops");
    // The hint says HigherBetter + "iops"; extract_metrics leaves
    // it at Unknown because polarity resolution is a PayloadRun-
    // side pass, not an extract_metrics-side one.
    assert_eq!(metrics[0].polarity, Polarity::Unknown);
    assert_eq!(metrics[0].unit, "");
}

/// Compile-time invariant: every fixture carries the exit-code gate
/// as its first default check. Declared as a `const` block so
/// the assertion runs at compile time — a silent drift of
/// `default_checks[0]` breaks the build, not a test.
#[test]
fn fixtures_default_checks_pin_exit_code_gate() {
    const _: () = {
        assert!(matches!(FIO.default_checks[0], Check::ExitCodeEq(0)));
        assert!(matches!(FIO_JSON.default_checks[0], Check::ExitCodeEq(0)));
        assert!(matches!(STRESS_NG.default_checks[0], Check::ExitCodeEq(0)));
        assert!(matches!(SCHBENCH.default_checks[0], Check::ExitCodeEq(0)));
        assert!(matches!(
            SCHBENCH_HINTED.default_checks[0],
            Check::ExitCodeEq(0),
        ));
    };
    assert!(!FIO.default_checks.is_empty());
    assert!(!FIO_JSON.default_checks.is_empty());
    assert!(!STRESS_NG.default_checks.is_empty());
    assert!(!SCHBENCH.default_checks.is_empty());
    assert!(!SCHBENCH_HINTED.default_checks.is_empty());
}

/// Identity-tag every fixture's output format so a consumer
/// reading this file sees the cases side-by-side — Json,
/// ExitCode, LlmExtract(None), and LlmExtract(Some(_)) — the
/// three canonical acquisition paths plus the hint-carrying
/// subvariant of the LLM path.
#[test]
fn fixture_output_formats_span_json_exit_code_and_llm_extract() {
    assert!(matches!(FIO.output, OutputFormat::Json));
    assert!(matches!(FIO_JSON.output, OutputFormat::Json));
    assert!(matches!(STRESS_NG.output, OutputFormat::ExitCode));
    assert!(matches!(SCHBENCH.output, OutputFormat::LlmExtract(None)));
    assert!(matches!(
        SCHBENCH_HINTED.output,
        OutputFormat::LlmExtract(Some(_)),
    ));
}
