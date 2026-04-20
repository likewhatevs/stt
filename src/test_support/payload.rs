//! Generalized test payload — scheduler or binary workload.
//!
//! [`Payload`] is the primitive that `#[ktstr_test]` consumes for both
//! the scheduler slot and the optional binary/workload slots. A
//! payload's [`PayloadKind`] determines how it's launched: a
//! [`Scheduler`](crate::test_support::Scheduler) reference invokes the
//! existing scheduler-spawn path; a bare binary name spawns the binary
//! via the runtime [`PayloadRun`](crate::scenario::payload_run::PayloadRun)
//! builder.
//!
//! The constants this module exposes — particularly [`Payload::EEVDF`] —
//! are used as the default scheduler slot when no `scheduler = ...`
//! attribute is supplied on a `#[ktstr_test]`.
//!
//! [`KtstrTestEntry`](crate::test_support::KtstrTestEntry) carries
//! `payload` and `workloads` fields populated by the `#[ktstr_test]`
//! macro's `payload = ...` and `workloads = [...]` attributes.

use crate::test_support::Scheduler;

// ---------------------------------------------------------------------------
// Payload + PayloadKind
// ---------------------------------------------------------------------------

/// A test payload — either a scheduler or a userspace binary to run
/// inside the guest VM.
///
/// `Payload` unifies the two launch modes under one `#[ktstr_test]`
/// attribute surface: tests declare `scheduler = SOME_SCHED` for
/// scheduler-centric runs, `payload = SOME_BIN` for binary runs, or
/// both with `workloads = [...]` to compose binaries under a
/// scheduler. See [`PayloadKind`] for the two variants.
///
/// Use [`Payload::EEVDF`] as the default scheduler placeholder (no scx
/// scheduler, kernel default).
///
/// `Payload` intentionally does NOT implement [`serde::Serialize`] /
/// [`serde::Deserialize`]. It is a compile-time-static definition that
/// references `&'static Scheduler` and `&'static [&'static str]`
/// slices — lifetimes that serialization cannot round-trip. Runtime
/// telemetry (per-payload metrics, exit codes, names) is serialized
/// via [`PayloadMetrics`] and [`Metric`] instead; those own their
/// data.
#[derive(Clone, Copy)]
pub struct Payload {
    /// Short, stable name used in logs and sidecar records.
    pub name: &'static str,
    /// Launch kind — scheduler reference or binary name.
    pub kind: PayloadKind,
    /// How the framework extracts metrics from the payload's stdout.
    pub output: OutputFormat,
    /// Default CLI args appended when this payload runs. Test bodies
    /// can extend via `.arg(...)` or replace via `.clear_args()` +
    /// `.arg(...)` on the runtime builder.
    pub default_args: &'static [&'static str],
    /// Author-declared default checks evaluated against extracted
    /// [`PayloadMetrics`]. Payloads that need exit-code gating
    /// should include [`Check::ExitCodeEq(0)`](Check::ExitCodeEq)
    /// here; the runtime evaluates `ExitCodeEq` as a pre-pass
    /// before metric checks.
    pub default_checks: &'static [Check],
    /// Declared metric hints — polarity, unit. Unhinted metrics
    /// extracted from output land as [`Polarity::Unknown`].
    pub metrics: &'static [MetricHint],
}

impl std::fmt::Debug for Payload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The inner `Scheduler` does not implement `Debug`; render
        // the payload via its public identity fields instead so
        // downstream Debug-requiring contexts (test panics, trace
        // logs) can stamp a payload without a full struct dump.
        f.debug_struct("Payload")
            .field("name", &self.name)
            .field("kind", &self.kind)
            .field("output", &self.output)
            .field("default_args_len", &self.default_args.len())
            .field("default_checks_len", &self.default_checks.len())
            .field("metrics_len", &self.metrics.len())
            .finish()
    }
}

/// How a payload is launched inside the guest.
///
/// Two variants — scheduler and binary — map to the two launch paths
/// in the runtime. "Kernel default" (EEVDF) is represented as
/// `Scheduler(&Scheduler::EEVDF)` rather than a dedicated variant
/// because [`Scheduler`] already carries the no-userspace-binary
/// taxonomy via its own `binary: SchedulerSpec` field.
#[derive(Clone, Copy)]
pub enum PayloadKind {
    /// Wraps an existing [`Scheduler`] definition. The scheduler's
    /// own `binary: SchedulerSpec` carries the Eevdf/Discover/Path/
    /// KernelBuiltin taxonomy — no duplication at the Payload level.
    Scheduler(&'static Scheduler),
    /// Bare userspace binary looked up by name in the guest (via the
    /// include-files infrastructure). Not a scheduler — runs as a
    /// workload under whatever scheduler the test declares.
    Binary(&'static str),
}

impl std::fmt::Debug for PayloadKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Scheduler does not implement Debug; render variant +
        // identity summary.
        match self {
            PayloadKind::Scheduler(s) => f.debug_tuple("Scheduler").field(&s.name).finish(),
            PayloadKind::Binary(name) => f.debug_tuple("Binary").field(name).finish(),
        }
    }
}

impl Payload {
    /// Placeholder payload that wraps [`Scheduler::EEVDF`], the "no
    /// scx scheduler" const. Used as the default value of the
    /// `scheduler` slot on [`KtstrTestEntry`](crate::test_support::KtstrTestEntry)
    /// so tests without an explicit `scheduler = ...` attribute still
    /// get a valid, non-optional reference.
    pub const EEVDF: Payload = Payload {
        name: "eevdf",
        kind: PayloadKind::Scheduler(&Scheduler::EEVDF),
        output: OutputFormat::ExitCode,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
    };

    /// Short, human-readable name for logging and sidecar output.
    pub const fn display_name(&self) -> &'static str {
        self.name
    }

    /// Return the inner [`Scheduler`] reference when this payload
    /// wraps one. Returns `None` for [`PayloadKind::Binary`].
    pub const fn as_scheduler(&self) -> Option<&'static Scheduler> {
        match self.kind {
            PayloadKind::Scheduler(s) => Some(s),
            PayloadKind::Binary(_) => None,
        }
    }

    /// True when this payload wraps a [`Scheduler`] (scheduler
    /// slot). False for binary payloads.
    pub const fn is_scheduler(&self) -> bool {
        matches!(self.kind, PayloadKind::Scheduler(_))
    }
}

// ---------------------------------------------------------------------------
// OutputFormat
// ---------------------------------------------------------------------------

/// How the framework extracts metrics from a payload's stdout.
///
/// `ExitCode` records only the exit code; no stdout parsing. `Json`
/// finds a JSON document region within stdout and walks numeric
/// leaves into [`Metric`] values. `LlmExtract` routes stdout through a
/// local small-model prompt that produces JSON, then runs the same
/// JSON walker — one extraction pipeline, two acquisition paths.
#[derive(Debug, Clone, Copy)]
pub enum OutputFormat {
    /// Pass/fail from exit code alone. Stdout is archived for
    /// debugging but not parsed.
    ExitCode,
    /// Parse stdout as JSON (finding the JSON region within mixed
    /// output), extract numeric leaves as metrics keyed by dotted
    /// path (e.g. `jobs.0.read.iops`).
    Json,
    /// Feed stdout to a local small model; model emits JSON; walk
    /// that JSON as in [`OutputFormat::Json`] but tag each metric with
    /// [`MetricSource::LlmExtract`]. The optional `&'static str` is a
    /// user-provided focus hint appended to the default prompt.
    LlmExtract(Option<&'static str>),
}

// ---------------------------------------------------------------------------
// Polarity, Check, Metric, MetricSource
// ---------------------------------------------------------------------------

/// Regression direction for a metric.
///
/// Used by `cargo ktstr test-stats` to classify deltas between runs.
/// Declared explicitly on [`MetricHint`]; unhinted metrics default to
/// [`Polarity::Unknown`] and are recorded without regression
/// classification.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Polarity {
    /// Bigger is better (throughput, IOPS, bogo_ops/sec). Regression
    /// = decrease from baseline.
    HigherBetter,
    /// Smaller is better (latency percentiles, error rates).
    /// Regression = increase from baseline.
    LowerBetter,
    /// A target value that the metric should hover near. Regression
    /// = absolute distance exceeds a threshold, symmetric in either
    /// direction. The inner `f64` MUST be finite (not NaN/inf);
    /// construct via [`Polarity::target`], which enforces this at
    /// runtime in both debug and release.
    TargetValue(f64),
    /// Direction not declared; the metric is recorded but not
    /// classified as regression-relevant.
    Unknown,
}

impl Polarity {
    /// Map the legacy `higher_is_worse: bool` used by
    /// [`MetricDef`](crate::stats::MetricDef) to a `Polarity`.
    ///
    /// The sense is INVERSE: `true` (bigger values are regressions)
    /// maps to [`Polarity::LowerBetter`] (we want the metric to go
    /// down); `false` maps to [`Polarity::HigherBetter`].
    pub const fn from_higher_is_worse(higher_is_worse: bool) -> Polarity {
        if higher_is_worse {
            Polarity::LowerBetter
        } else {
            Polarity::HigherBetter
        }
    }

    /// Construct a [`Polarity::TargetValue`] after asserting that
    /// `target` is finite. Non-finite `target` (`NaN`, `±inf`)
    /// produces incorrect regression verdicts in the comparison
    /// pipeline, so the check runs in release builds too.
    pub fn target(target: f64) -> Polarity {
        assert!(
            target.is_finite(),
            "Polarity::TargetValue target must be finite, got {target}"
        );
        Polarity::TargetValue(target)
    }
}

/// Payload-author metric declaration: polarity + display unit.
///
/// Attached to a [`Payload`] via the `metrics` field. Metrics
/// extracted from output are looked up against this table by name to
/// set their [`Polarity`] and [`Metric::unit`]. Unmatched metrics
/// land with `Polarity::Unknown` and an empty unit string.
#[derive(Debug, Clone, Copy)]
pub struct MetricHint {
    /// Dotted-path metric name (e.g. `jobs.0.read.iops`).
    pub name: &'static str,
    /// Regression direction for this metric.
    pub polarity: Polarity,
    /// Human-readable unit for display (e.g. `iops`, `ns`). Empty
    /// string means "no unit"; matches the sentinel used by
    /// [`MetricDef`](crate::stats::MetricDef).
    pub unit: &'static str,
}

/// Assertion check evaluated against an extracted
/// [`PayloadMetrics`] (or the exit code for
/// [`Check::ExitCodeEq`](Check::ExitCodeEq)).
#[derive(Debug, Clone, Copy)]
pub enum Check {
    /// Fail when the named metric is below `value`.
    Min { metric: &'static str, value: f64 },
    /// Fail when the named metric exceeds `value`.
    Max { metric: &'static str, value: f64 },
    /// Fail when the named metric is outside `[lo, hi]`.
    Range {
        metric: &'static str,
        lo: f64,
        hi: f64,
    },
    /// Fail when the named metric is missing from the extracted set.
    Exists(&'static str),
    /// Fail when the payload's exit code is not equal to `expected`.
    ExitCodeEq(i32),
}

impl Check {
    pub const fn min(metric: &'static str, value: f64) -> Check {
        Check::Min { metric, value }
    }

    pub const fn max(metric: &'static str, value: f64) -> Check {
        Check::Max { metric, value }
    }

    pub const fn range(metric: &'static str, lo: f64, hi: f64) -> Check {
        Check::Range { metric, lo, hi }
    }

    pub const fn exists(metric: &'static str) -> Check {
        Check::Exists(metric)
    }

    pub const fn exit_code_eq(expected: i32) -> Check {
        Check::ExitCodeEq(expected)
    }
}

/// Provenance of a [`Metric`] — tells downstream tooling whether the
/// value came from a structured-output parse or from LLM-derived
/// extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MetricSource {
    /// Extracted directly from JSON output via
    /// [`OutputFormat::Json`].
    Json,
    /// Extracted by feeding stdout through the local model (LlmExtract
    /// path). Treat with somewhat lower confidence than `Json`.
    LlmExtract,
}

/// A single extracted metric from a payload's output.
///
/// Populated by the extraction pipeline after the payload exits.
/// Sidecar serialization carries these alongside the pass/fail
/// verdict so test-stats can classify regressions across runs.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Metric {
    /// Dotted-path name matching the JSON leaf or the LLM-emitted key.
    pub name: String,
    /// Numeric value.
    pub value: f64,
    /// Regression direction, copied from the matching
    /// [`MetricHint`] or left as [`Polarity::Unknown`] when no hint
    /// matches.
    pub polarity: Polarity,
    /// Display unit string; empty when no unit was declared.
    pub unit: String,
    /// Where this metric came from — JSON parse or LLM extraction.
    pub source: MetricSource,
}

/// All metrics extracted from a single payload run plus the process
/// exit code.
///
/// Each concurrent payload (primary or workload, foreground or
/// background) produces one `PayloadMetrics` value. Sidecar stores
/// these as a `Vec<PayloadMetrics>` keyed by payload name so
/// per-payload provenance is preserved across composed tests.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PayloadMetrics {
    /// Extracted metrics. Empty when [`OutputFormat::ExitCode`] is
    /// used or when JSON parsing found no numeric leaves.
    pub metrics: Vec<Metric>,
    /// Process exit code (0 = success). Used by
    /// [`Check::ExitCodeEq`](Check::ExitCodeEq) in the check
    /// evaluation pre-pass.
    pub exit_code: i32,
}

impl PayloadMetrics {
    /// Look up a metric by exact name. Returns `None` when the
    /// metric is not in the set.
    pub fn get(&self, name: &str) -> Option<f64> {
        self.metrics
            .iter()
            .find(|m| m.name == name)
            .map(|m| m.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_eevdf_const_is_scheduler_kind() {
        assert!(matches!(Payload::EEVDF.kind, PayloadKind::Scheduler(_)));
        assert_eq!(Payload::EEVDF.display_name(), "eevdf");
        assert!(matches!(Payload::EEVDF.output, OutputFormat::ExitCode));
        assert!(Payload::EEVDF.default_args.is_empty());
        assert!(Payload::EEVDF.default_checks.is_empty());
        assert!(Payload::EEVDF.metrics.is_empty());
    }

    #[test]
    fn payload_eevdf_wraps_scheduler_eevdf() {
        match Payload::EEVDF.kind {
            PayloadKind::Scheduler(s) => {
                assert_eq!(s.name, Scheduler::EEVDF.name);
            }
            PayloadKind::Binary(_) => panic!("EEVDF should be Scheduler-kind, got Binary"),
        }
    }

    #[test]
    fn check_constructors() {
        assert!(matches!(Check::min("x", 1.0), Check::Min { .. }));
        assert!(matches!(Check::max("x", 1.0), Check::Max { .. }));
        assert!(matches!(Check::range("x", 1.0, 2.0), Check::Range { .. }));
        assert!(matches!(Check::exists("x"), Check::Exists("x")));
        assert!(matches!(Check::exit_code_eq(0), Check::ExitCodeEq(0)));
    }

    #[test]
    fn metric_set_get_returns_value() {
        let pm = PayloadMetrics {
            metrics: vec![Metric {
                name: "iops".to_string(),
                value: 1000.0,
                polarity: Polarity::HigherBetter,
                unit: "iops".to_string(),
                source: MetricSource::Json,
            }],
            exit_code: 0,
        };
        assert_eq!(pm.get("iops"), Some(1000.0));
        assert_eq!(pm.get("missing"), None);
    }

    #[test]
    fn polarity_target_value_carries_data() {
        let p = Polarity::TargetValue(42.0);
        match p {
            Polarity::TargetValue(v) => assert_eq!(v, 42.0),
            _ => panic!("expected TargetValue variant"),
        }
    }

    #[test]
    fn output_format_variants() {
        let _: OutputFormat = OutputFormat::ExitCode;
        let _: OutputFormat = OutputFormat::Json;
        let _: OutputFormat = OutputFormat::LlmExtract(None);
        let _: OutputFormat = OutputFormat::LlmExtract(Some("focus on iops"));
    }

    #[test]
    fn metric_source_serde_round_trip() {
        let js = serde_json::to_string(&MetricSource::Json).unwrap();
        let de: MetricSource = serde_json::from_str(&js).unwrap();
        assert_eq!(de, MetricSource::Json);
        let js = serde_json::to_string(&MetricSource::LlmExtract).unwrap();
        let de: MetricSource = serde_json::from_str(&js).unwrap();
        assert_eq!(de, MetricSource::LlmExtract);
    }

    #[test]
    fn polarity_serde_round_trip() {
        for p in [
            Polarity::HigherBetter,
            Polarity::LowerBetter,
            Polarity::TargetValue(2.78),
            Polarity::Unknown,
        ] {
            let js = serde_json::to_string(&p).unwrap();
            let de: Polarity = serde_json::from_str(&js).unwrap();
            assert_eq!(de, p);
        }
    }

    // Item 3: PayloadKind::Binary construction + pattern match.
    #[test]
    fn payload_kind_binary_construction_and_match() {
        const FIO: Payload = Payload {
            name: "fio",
            kind: PayloadKind::Binary("fio"),
            output: OutputFormat::Json,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
        };
        match FIO.kind {
            PayloadKind::Binary(name) => assert_eq!(name, "fio"),
            PayloadKind::Scheduler(_) => panic!("expected Binary, got Scheduler"),
        }
        assert!(!FIO.is_scheduler());
        assert!(FIO.as_scheduler().is_none());
    }

    // Item 4: const bindings verify const-fn actually works in const context.
    const _MIN: Check = Check::min("x", 1.0);
    const _MAX: Check = Check::max("x", 2.0);
    const _RANGE: Check = Check::range("x", 1.0, 2.0);
    const _EXISTS: Check = Check::exists("x");
    const _EXIT: Check = Check::exit_code_eq(0);
    const _EEVDF_REF: &Payload = &Payload::EEVDF;
    const _EEVDF_IS_SCHED: bool = Payload::EEVDF.is_scheduler();
    const _EEVDF_DISPLAY: &str = Payload::EEVDF.display_name();

    // Proves an arbitrary `Payload` (not just `Payload::EEVDF`) is
    // const-constructible via struct literal — the #[derive(Payload)]
    // proc-macro (WO-162-J) will emit exactly this shape.
    const _PAYLOAD_CONST_BUILD: Payload = Payload {
        name: "fio",
        kind: PayloadKind::Binary("fio"),
        output: OutputFormat::Json,
        default_args: &["--output-format=json"],
        default_checks: &[Check::exit_code_eq(0)],
        metrics: &[MetricHint {
            name: "jobs.0.read.iops",
            polarity: Polarity::HigherBetter,
            unit: "iops",
        }],
    };

    #[test]
    fn const_bindings_are_usable() {
        assert!(matches!(_MIN, Check::Min { .. }));
        assert!(matches!(_MAX, Check::Max { .. }));
        assert!(matches!(_RANGE, Check::Range { .. }));
        assert!(matches!(_EXISTS, Check::Exists("x")));
        assert!(matches!(_EXIT, Check::ExitCodeEq(0)));
        assert_eq!(_EEVDF_REF.name, "eevdf");
        const { assert!(_EEVDF_IS_SCHED) };
        assert_eq!(_EEVDF_DISPLAY, "eevdf");
    }

    // Item 5: from_higher_is_worse helper.
    #[test]
    fn polarity_from_higher_is_worse_flips_sense() {
        assert_eq!(Polarity::from_higher_is_worse(true), Polarity::LowerBetter);
        assert_eq!(
            Polarity::from_higher_is_worse(false),
            Polarity::HigherBetter
        );
    }

    #[test]
    fn polarity_target_accepts_finite() {
        let p = Polarity::target(0.5);
        assert_eq!(p, Polarity::TargetValue(0.5));
    }

    // Item 1 + 2: Debug + helper method surface.
    #[test]
    fn payload_debug_renders_identity_fields() {
        let s = format!("{:?}", Payload::EEVDF);
        assert!(s.contains("Payload"), "debug output: {s}");
        assert!(s.contains("eevdf"), "debug output: {s}");
        assert!(
            s.contains("kind: Scheduler(\"eevdf\")"),
            "debug output: {s}"
        );
    }

    #[test]
    fn payload_kind_debug_renders_variant_and_identity() {
        let binary = PayloadKind::Binary("fio");
        let s = format!("{binary:?}");
        assert!(s.contains("Binary"), "debug output: {s}");
        assert!(s.contains("fio"), "debug output: {s}");

        let sched = Payload::EEVDF.kind;
        let s = format!("{sched:?}");
        assert!(s.contains("Scheduler"), "debug output: {s}");
        assert!(s.contains("eevdf"), "debug output: {s}");
    }

    #[test]
    fn output_format_derive_debug_clone_copy() {
        let a = OutputFormat::Json;
        let b = a; // Copy
        let _ = format!("{a:?} {b:?}"); // Debug
    }

    #[test]
    fn as_scheduler_extracts_ref_for_scheduler_kind() {
        let s = Payload::EEVDF.as_scheduler().expect("Scheduler kind");
        assert_eq!(s.name, "eevdf");
    }

    #[test]
    fn payload_clone_preserves_identity() {
        let a = Payload::EEVDF;
        assert_eq!(a.name, Payload::EEVDF.name);
        assert_eq!(a.is_scheduler(), Payload::EEVDF.is_scheduler());
        assert_eq!(a.as_scheduler().map(|s| s.name), Some("eevdf"));
    }
}
