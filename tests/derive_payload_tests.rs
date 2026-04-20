//! Structural tests for `#[derive(Payload)]`.
//!
//! These live in their own test crate rather than
//! `ktstr_test_macro.rs` because that file carries `#[ktstr_test]`
//! entries whose `#[ctor]` discovery routes nextest's `--list`
//! through `ktstr_main`, hiding plain `#[test]` functions.
//! Isolating the struct-only tests here keeps them visible to the
//! standard Rust test harness.

use ktstr::test_support::{Check, OutputFormat, PayloadKind, Polarity};

/// Minimal derive: only `binary` is set, everything else defaults.
/// Verifies the const-name strip + uppercase conversion,
/// name-falls-back-to-binary, and the ExitCode output default.
#[derive(ktstr::Payload)]
#[payload(binary = "fio")]
#[allow(dead_code)]
struct FioMinimalPayload;

#[test]
fn derive_payload_minimal_const_name_and_defaults() {
    assert_eq!(FIO_MINIMAL.name, "fio");
    assert!(matches!(FIO_MINIMAL.kind, PayloadKind::Binary("fio")));
    assert!(matches!(FIO_MINIMAL.output, OutputFormat::ExitCode));
    assert!(FIO_MINIMAL.default_args.is_empty());
    assert!(FIO_MINIMAL.default_checks.is_empty());
    assert!(FIO_MINIMAL.metrics.is_empty());
}

/// Full grammar: every optional attribute at once. Verifies
/// accumulation of `default_args` across multiple attrs, Check
/// expressions, MetricHint with every polarity variant surface.
#[derive(ktstr::Payload)]
#[payload(
    binary = "fio",
    name = "fio_custom",
    output = Json,
)]
#[default_args("--output-format=json", "--minimal")]
#[default_args("--runtime=30")]
#[default_check(exit_code_eq(0))]
#[default_check(min("jobs.0.read.iops", 1000.0))]
#[metric(name = "jobs.0.read.iops", polarity = HigherBetter, unit = "iops")]
#[metric(name = "lat_ns", polarity = LowerBetter, unit = "ns")]
#[metric(name = "target_cpu", polarity = TargetValue(50.0), unit = "%")]
#[metric(name = "unlabeled")]
#[allow(dead_code)]
struct FioFullPayload;

#[test]
fn derive_payload_full_grammar() {
    assert_eq!(FIO_FULL.name, "fio_custom");
    assert!(matches!(FIO_FULL.kind, PayloadKind::Binary("fio")));
    assert!(matches!(FIO_FULL.output, OutputFormat::Json));

    assert_eq!(
        FIO_FULL.default_args,
        &["--output-format=json", "--minimal", "--runtime=30"],
    );

    assert_eq!(FIO_FULL.default_checks.len(), 2);
    assert!(matches!(FIO_FULL.default_checks[0], Check::ExitCodeEq(0)));
    assert!(matches!(
        FIO_FULL.default_checks[1],
        Check::Min { metric, value } if metric == "jobs.0.read.iops" && value == 1000.0,
    ));

    assert_eq!(FIO_FULL.metrics.len(), 4);
    assert_eq!(FIO_FULL.metrics[0].name, "jobs.0.read.iops");
    assert_eq!(FIO_FULL.metrics[0].polarity, Polarity::HigherBetter);
    assert_eq!(FIO_FULL.metrics[0].unit, "iops");
    assert_eq!(FIO_FULL.metrics[1].name, "lat_ns");
    assert_eq!(FIO_FULL.metrics[1].polarity, Polarity::LowerBetter);
    assert_eq!(FIO_FULL.metrics[1].unit, "ns");
    assert_eq!(FIO_FULL.metrics[2].name, "target_cpu");
    assert_eq!(FIO_FULL.metrics[2].polarity, Polarity::TargetValue(50.0));
    assert_eq!(FIO_FULL.metrics[2].unit, "%");
    assert_eq!(FIO_FULL.metrics[3].name, "unlabeled");
    assert_eq!(FIO_FULL.metrics[3].polarity, Polarity::Unknown);
    assert_eq!(FIO_FULL.metrics[3].unit, "");
}

/// `output = LlmExtract` (bare-ident shorthand) resolves to
/// `LlmExtract(None)`.
#[derive(ktstr::Payload)]
#[payload(binary = "spec_cpu", output = LlmExtract)]
#[allow(dead_code)]
struct SpecCpuPayload;

#[test]
fn derive_payload_llm_extract_bare_is_no_hint() {
    assert!(matches!(SPEC_CPU.output, OutputFormat::LlmExtract(None)));
}

/// `output = LlmExtract("hint")` emits `LlmExtract(Some("hint"))`
/// so the value carries through to the runtime prompt.
#[derive(ktstr::Payload)]
#[payload(binary = "bench_with_hint", output = LlmExtract("focus on throughput"))]
#[allow(dead_code)]
struct BenchHintedPayload;

#[test]
fn derive_payload_llm_extract_call_carries_hint() {
    match BENCH_HINTED.output {
        OutputFormat::LlmExtract(Some(hint)) => {
            assert_eq!(hint, "focus on throughput");
        }
        other => panic!("expected LlmExtract(Some(..)), got {other:?}"),
    }
}

/// Empty `LlmExtract()` call = no hint.
#[derive(ktstr::Payload)]
#[payload(binary = "bench_empty_call", output = LlmExtract())]
#[allow(dead_code)]
struct BenchEmptyCallPayload;

#[test]
fn derive_payload_llm_extract_empty_call_has_no_hint() {
    assert!(matches!(
        BENCH_EMPTY_CALL.output,
        OutputFormat::LlmExtract(None),
    ));
}

/// Struct name with NO `Payload` suffix: the derive converts the
/// full CamelCase ident to SCREAMING_SNAKE and uses that as the
/// const name. `StressNg` → `STRESS_NG`.
///
/// This test pins ONLY the emitted `const` identifier path. It
/// does NOT pin `Payload.name` — that field comes from the
/// `#[payload(...)]` attribute: here neither `name = "..."` nor a
/// short alias is supplied, so the `.name` field falls back to
/// the `binary` attribute (`"stress-ng"`), which happens to be
/// identical to the lowercased const name. That coincidence is
/// NOT the invariant under test; the const-identifier derivation
/// is. If a future const-name rule changed (e.g. to keep CamelCase
/// untouched), only line 1's assertion would break — line 2's
/// `Binary("stress-ng")` would still hold because it comes from
/// the attribute, not the ident. The two assertions exercise
/// different code paths that happen to produce matching strings
/// here; do not collapse them.
#[derive(ktstr::Payload)]
#[payload(binary = "stress-ng")]
#[allow(dead_code)]
struct StressNg;

#[test]
fn derive_payload_no_suffix_keeps_full_name() {
    assert_eq!(STRESS_NG.name, "stress-ng");
    assert!(matches!(STRESS_NG.kind, PayloadKind::Binary("stress-ng")));
}

/// Multi-word struct named with a `Payload` suffix. The suffix
/// strip happens BEFORE the SCREAMING_SNAKE conversion so the
/// generated const name is `MEMCHECK`, not `MEMCHECK_PAYLOAD`.
#[derive(ktstr::Payload)]
#[payload(binary = "memcheck-bin", name = "memcheck")]
#[allow(dead_code)]
struct MemcheckPayload;

#[test]
fn derive_payload_suffix_strip_happens_before_uppercase() {
    assert_eq!(MEMCHECK.name, "memcheck");
    assert!(matches!(MEMCHECK.kind, PayloadKind::Binary("memcheck-bin"),));
}

/// Metric with only `name` set (no polarity, no unit) defaults to
/// `Polarity::Unknown` and empty unit — matches the frozen design's
/// "unhinted metrics" contract.
#[derive(ktstr::Payload)]
#[payload(binary = "bare_metric")]
#[metric(name = "throughput")]
#[allow(dead_code)]
struct BareMetricPayload;

#[test]
fn derive_payload_metric_minimal_defaults_unknown_polarity() {
    assert_eq!(BARE_METRIC.metrics.len(), 1);
    assert_eq!(BARE_METRIC.metrics[0].name, "throughput");
    assert_eq!(BARE_METRIC.metrics[0].polarity, Polarity::Unknown);
    assert_eq!(BARE_METRIC.metrics[0].unit, "");
}

/// Default check uses `range(...)` — verifies the macro's bare
/// Check-constructor resolution against every const-fn constructor,
/// not just the ones used above.
#[derive(ktstr::Payload)]
#[payload(binary = "range_check_bin")]
#[default_check(range("cpu_pct", 10.0, 90.0))]
#[default_check(max("latency_us", 500.0))]
#[default_check(exists("sampling_key"))]
#[allow(dead_code)]
struct RangeCheckPayload;

#[test]
fn derive_payload_default_checks_resolve_all_constructors() {
    assert_eq!(RANGE_CHECK.default_checks.len(), 3);
    assert!(matches!(
        RANGE_CHECK.default_checks[0],
        Check::Range { metric, lo, hi } if metric == "cpu_pct" && lo == 10.0 && hi == 90.0,
    ));
    assert!(matches!(
        RANGE_CHECK.default_checks[1],
        Check::Max { metric, value } if metric == "latency_us" && value == 500.0,
    ));
    assert!(matches!(
        RANGE_CHECK.default_checks[2],
        Check::Exists("sampling_key"),
    ));
}

/// #62: both bare (`min(...)`) and qualified (`Check::min(...)`)
/// constructor forms must resolve to the same generated `Check::...`
/// variant. The macro detects the explicit `Check` segment on the
/// callee path and skips its implicit `::ktstr::test_support::Check::`
/// prepend so a user who imports `Check` and writes the prefix
/// themselves gets `Check::min(...)`, not `Check::Check::min(...)`.
#[derive(ktstr::Payload)]
#[payload(binary = "qualified_check_bin")]
#[default_check(Check::min("iops", 1000.0))]
#[default_check(Check::max("latency_us", 500.0))]
#[default_check(exists("sampling_key"))]
#[allow(dead_code)]
struct QualifiedCheckPayload;

#[test]
fn derive_payload_accepts_qualified_check_prefix() {
    assert_eq!(QUALIFIED_CHECK.default_checks.len(), 3);
    assert!(matches!(
        QUALIFIED_CHECK.default_checks[0],
        Check::Min { metric, value } if metric == "iops" && value == 1000.0,
    ));
    assert!(matches!(
        QUALIFIED_CHECK.default_checks[1],
        Check::Max { metric, value } if metric == "latency_us" && value == 500.0,
    ));
    assert!(matches!(
        QUALIFIED_CHECK.default_checks[2],
        Check::Exists("sampling_key"),
    ));
}

/// #60: Explicit `output = ExitCode` must parse through the same
/// PascalCase output grammar as `Json` / `LlmExtract` and emit a
/// Payload whose `.output == OutputFormat::ExitCode`. The default
/// (no `output =` kwarg) also lands at `ExitCode`, but a future
/// change that silently promoted an absent `output` to
/// `OutputFormat::Json` would go undetected without this test —
/// explicit + default both must resolve to `ExitCode`.
#[derive(ktstr::Payload)]
#[payload(binary = "exit_code_bin", output = ExitCode)]
#[allow(dead_code)]
struct ExplicitExitCodePayload;

#[test]
fn derive_payload_explicit_exit_code_output() {
    assert!(matches!(EXPLICIT_EXIT_CODE.output, OutputFormat::ExitCode));
    assert_eq!(EXPLICIT_EXIT_CODE.name, "exit_code_bin");
    assert!(matches!(
        EXPLICIT_EXIT_CODE.kind,
        PayloadKind::Binary("exit_code_bin"),
    ));
}

/// #62: the fully-qualified crate path
/// `::ktstr::test_support::Check::min(...)` must also resolve
/// through the same prefix-detection branch as the shorter
/// `Check::min(...)` form. `expr_has_check_prefix` scans every
/// segment for an ident named `Check`, so any absolute path that
/// carries the type still lands on the user-written callee
/// without the macro double-prepending its implicit
/// `::ktstr::test_support::Check::` segment.
#[derive(ktstr::Payload)]
#[payload(binary = "fully_qualified_check_bin")]
#[default_check(::ktstr::test_support::Check::min("iops", 500.0))]
#[default_check(::ktstr::test_support::Check::exit_code_eq(0))]
#[allow(dead_code)]
struct FullyQualifiedCheckPayload;

#[test]
fn derive_payload_accepts_fully_qualified_check_path() {
    assert_eq!(FULLY_QUALIFIED_CHECK.default_checks.len(), 2);
    assert!(matches!(
        FULLY_QUALIFIED_CHECK.default_checks[0],
        Check::Min { metric, value } if metric == "iops" && value == 500.0,
    ));
    assert!(matches!(
        FULLY_QUALIFIED_CHECK.default_checks[1],
        Check::ExitCodeEq(0),
    ));
}
