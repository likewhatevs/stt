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

    /// Minimal const wrapper: build a `Payload` that references an
    /// existing `&'static Scheduler`. Used by unit tests and by the
    /// `#[derive(Scheduler)]` wrapper emission to produce the
    /// `{CONST}_PAYLOAD` const alongside the Scheduler const. Copies
    /// the scheduler's `name` into the payload's `name` so the two
    /// surfaces render with matching identity.
    pub const fn from_scheduler(sched: &'static Scheduler) -> Payload {
        Payload {
            name: sched.name,
            kind: PayloadKind::Scheduler(sched),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
        }
    }

    // -----------------------------------------------------------------
    // Scheduler-slot forwarding accessors
    //
    // These methods let every site that consumed `entry.scheduler:
    // &Scheduler` read the equivalent field off `entry.scheduler:
    // &Payload` without the caller having to unwrap
    // `as_scheduler()`. For a scheduler-kind payload the accessor
    // forwards to the inner `Scheduler`. For a binary-kind payload
    // the accessor returns a sensible default — usually the empty
    // slice or the no-op value — matching the semantics a binary
    // payload in the scheduler slot should carry (no sysctls, no
    // kargs, no scheduler-specific CLI flags).
    //
    // The binary-kind branch is not "best effort": a binary payload
    // in the scheduler slot is a valid configuration (pure userspace
    // test under the kernel default scheduler), and every accessor
    // below returns exactly what that scenario should see.
    // -----------------------------------------------------------------

    /// The scheduler's display name. For scheduler-kind payloads this
    /// is the inner `Scheduler::name`; for binary-kind it is
    /// `"eevdf"` (a binary payload runs under the kernel default).
    pub const fn scheduler_name(&self) -> &'static str {
        match self.kind {
            PayloadKind::Scheduler(s) => s.name,
            PayloadKind::Binary(_) => "eevdf",
        }
    }

    /// The scheduler's binary spec when scheduler-kind; `None` for
    /// binary-kind payloads. Consumers that dispatch on the
    /// `SchedulerSpec` variant (e.g. `KernelBuiltin { enable, disable }`
    /// hook invocation) use this rather than the `scheduler_name`
    /// shortcut.
    pub const fn scheduler_binary(&self) -> Option<&'static crate::test_support::SchedulerSpec> {
        match self.kind {
            PayloadKind::Scheduler(s) => Some(&s.binary),
            PayloadKind::Binary(_) => None,
        }
    }

    /// True when this payload drives an active scheduling policy
    /// (anything other than the kernel default EEVDF). Forwards to
    /// `SchedulerSpec::has_active_scheduling` for scheduler-kind
    /// payloads; binary-kind payloads always return `false` — a
    /// binary runs under whatever scheduler the test declares, and
    /// does not itself impose one.
    pub const fn has_active_scheduling(&self) -> bool {
        match self.kind {
            PayloadKind::Scheduler(s) => s.binary.has_active_scheduling(),
            PayloadKind::Binary(_) => false,
        }
    }

    /// Scheduler flag declarations. Empty slice for binary-kind
    /// payloads (binaries have no scheduler flags).
    pub const fn flags(&self) -> &'static [&'static crate::scenario::flags::FlagDecl] {
        match self.kind {
            PayloadKind::Scheduler(s) => s.flags,
            PayloadKind::Binary(_) => &[],
        }
    }

    /// Guest sysctls applied before the scheduler starts. Empty slice
    /// for binary-kind payloads.
    pub const fn sysctls(&self) -> &'static [crate::test_support::Sysctl] {
        match self.kind {
            PayloadKind::Scheduler(s) => s.sysctls,
            PayloadKind::Binary(_) => &[],
        }
    }

    /// Extra guest kernel command-line arguments appended when
    /// booting the VM. Empty slice for binary-kind payloads.
    pub const fn kargs(&self) -> &'static [&'static str] {
        match self.kind {
            PayloadKind::Scheduler(s) => s.kargs,
            PayloadKind::Binary(_) => &[],
        }
    }

    /// Scheduler CLI args prepended before per-test `extra_sched_args`.
    /// Empty slice for binary-kind payloads.
    pub const fn sched_args(&self) -> &'static [&'static str] {
        match self.kind {
            PayloadKind::Scheduler(s) => s.sched_args,
            PayloadKind::Binary(_) => &[],
        }
    }

    /// Cgroup parent path. `None` for binary-kind payloads and for
    /// scheduler-kind payloads that did not set one.
    pub const fn cgroup_parent(&self) -> Option<crate::test_support::CgroupPath> {
        match self.kind {
            PayloadKind::Scheduler(s) => s.cgroup_parent,
            PayloadKind::Binary(_) => None,
        }
    }

    /// Host-side path to the scheduler config file. `None` for
    /// binary-kind payloads and for scheduler-kind payloads that
    /// did not set one.
    pub const fn config_file(&self) -> Option<&'static str> {
        match self.kind {
            PayloadKind::Scheduler(s) => s.config_file,
            PayloadKind::Binary(_) => None,
        }
    }

    /// Scheduler-wide assertion overrides. For binary-kind payloads
    /// returns `Assert::NO_OVERRIDES` — the default identity value
    /// merge that leaves per-entry assertions untouched.
    pub const fn assert(&self) -> &'static crate::assert::Assert {
        match self.kind {
            PayloadKind::Scheduler(s) => &s.assert,
            PayloadKind::Binary(_) => &crate::assert::Assert::NO_OVERRIDES,
        }
    }

    /// Names of all scheduler flags the scheduler-kind payload
    /// supports. Empty for binary-kind.
    pub fn supported_flag_names(&self) -> Vec<&'static str> {
        match self.kind {
            PayloadKind::Scheduler(s) => s.supported_flag_names(),
            PayloadKind::Binary(_) => Vec::new(),
        }
    }

    /// Extra CLI args associated with a scheduler flag. Always
    /// `None` for binary-kind.
    pub fn flag_args(&self, name: &str) -> Option<&'static [&'static str]> {
        match self.kind {
            PayloadKind::Scheduler(s) => s.flag_args(name),
            PayloadKind::Binary(_) => None,
        }
    }

    /// Default VM topology for this payload. Scheduler-kind payloads
    /// expose the topology declared on the inner `Scheduler` so tests
    /// that inherit from the scheduler slot stay consistent with the
    /// rest of the scheduler's test surface; binary-kind payloads
    /// return a minimal placeholder
    /// ([`Topology::DEFAULT_FOR_PAYLOAD`](crate::test_support::Topology::DEFAULT_FOR_PAYLOAD))
    /// — a pure binary workload has no scheduler-level topology
    /// opinion, so per-entry `#[ktstr_test(...)]` overrides are what
    /// actually drive the VM shape.
    pub const fn topology(&self) -> crate::test_support::Topology {
        match self.kind {
            PayloadKind::Scheduler(s) => s.topology,
            PayloadKind::Binary(_) => crate::test_support::Topology::DEFAULT_FOR_PAYLOAD,
        }
    }

    /// Gauntlet topology constraints. Scheduler-kind payloads forward
    /// to the inner `Scheduler::constraints`; binary-kind payloads
    /// return [`TopologyConstraints::DEFAULT`].
    pub const fn constraints(&self) -> crate::test_support::TopologyConstraints {
        match self.kind {
            PayloadKind::Scheduler(s) => s.constraints,
            PayloadKind::Binary(_) => crate::test_support::TopologyConstraints::DEFAULT,
        }
    }

    /// Generate scheduler-flag profiles for gauntlet expansion.
    /// Forwards to [`Scheduler::generate_profiles`] for scheduler-kind
    /// payloads; returns a single empty profile for binary-kind (a
    /// binary has no scheduler flags, and the gauntlet expander still
    /// wants one profile to run the test under).
    pub fn generate_profiles(
        &self,
        required: &[&'static str],
        excluded: &[&'static str],
    ) -> Vec<crate::scenario::FlagProfile> {
        match self.kind {
            PayloadKind::Scheduler(s) => s.generate_profiles(required, excluded),
            PayloadKind::Binary(_) => vec![crate::scenario::FlagProfile { flags: Vec::new() }],
        }
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

    /// #33: Round-trip bool → Polarity → bool for HigherBetter /
    /// LowerBetter yields the identity. Pins the "inverse sense"
    /// contract documented on `MetricDef::higher_is_worse` and
    /// `Polarity::from_higher_is_worse` so a future polarity
    /// refactor can't accidentally flip one direction without the
    /// other and silently break delta-classification downstream.
    ///
    /// The test synthesizes a throw-away `MetricDef` for each
    /// polarity because the production `METRICS` table's entries
    /// live in `stats.rs` and are test-only not importable from
    /// here — constructing the struct literal directly keeps the
    /// round-trip self-contained.
    #[test]
    fn higher_is_worse_polarity_round_trip() {
        use crate::stats::{Aggregator, MetricDef};

        // true (higher-is-worse) → LowerBetter → true.
        let m = MetricDef {
            name: "t",
            polarity: Polarity::from_higher_is_worse(true),
            default_abs: 0.0,
            default_rel: 0.0,
            display_unit: "",
            aggregate: Aggregator::Max,
            accessor: |_| None,
        };
        assert_eq!(m.polarity, Polarity::LowerBetter);
        assert!(m.higher_is_worse(), "LowerBetter → higher_is_worse = true");

        // false (higher-is-better) → HigherBetter → false.
        let m = MetricDef {
            name: "f",
            polarity: Polarity::from_higher_is_worse(false),
            default_abs: 0.0,
            default_rel: 0.0,
            display_unit: "",
            aggregate: Aggregator::Max,
            accessor: |_| None,
        };
        assert_eq!(m.polarity, Polarity::HigherBetter);
        assert!(
            !m.higher_is_worse(),
            "HigherBetter → higher_is_worse = false"
        );
    }

    /// #33: `MetricDef::higher_is_worse` is total over every
    /// `Polarity` variant — the current implementation lumps
    /// `LowerBetter`, `TargetValue`, and `Unknown` all into
    /// `true`. Pin that so a subtle change (e.g. TargetValue → its
    /// own category) doesn't silently flip regression direction
    /// for every test using target metrics.
    #[test]
    fn higher_is_worse_covers_all_polarity_variants() {
        use crate::stats::{Aggregator, MetricDef};
        fn make(p: Polarity) -> MetricDef {
            MetricDef {
                name: "x",
                polarity: p,
                default_abs: 0.0,
                default_rel: 0.0,
                display_unit: "",
                aggregate: Aggregator::Max,
                accessor: |_| None,
            }
        }
        assert!(!make(Polarity::HigherBetter).higher_is_worse());
        assert!(make(Polarity::LowerBetter).higher_is_worse());
        assert!(make(Polarity::TargetValue(42.0)).higher_is_worse());
        assert!(make(Polarity::Unknown).higher_is_worse());
    }

    #[test]
    fn polarity_target_accepts_finite() {
        let p = Polarity::target(0.5);
        assert_eq!(p, Polarity::TargetValue(0.5));
    }

    /// #43: `Polarity::target(NaN)` must panic in release
    /// too — non-finite target values produce silent incorrect
    /// regression verdicts in `compare_rows`, so the gate is a
    /// runtime `assert!` (not `debug_assert!`). Pins that a
    /// release build won't silently let NaN slip through.
    #[test]
    #[should_panic(expected = "Polarity::TargetValue target must be finite")]
    fn polarity_target_rejects_nan_panics() {
        let _ = Polarity::target(f64::NAN);
    }

    /// #43: `Polarity::target(+inf)` panics symmetrically with
    /// NaN. `compare_rows` would produce inf-vs-finite verdicts
    /// that depend on IEEE-754 infinity arithmetic rather than
    /// meaningful regression direction.
    #[test]
    #[should_panic(expected = "Polarity::TargetValue target must be finite")]
    fn polarity_target_rejects_positive_infinity_panics() {
        let _ = Polarity::target(f64::INFINITY);
    }

    /// #43: `Polarity::target(-inf)` ditto.
    #[test]
    #[should_panic(expected = "Polarity::TargetValue target must be finite")]
    fn polarity_target_rejects_negative_infinity_panics() {
        let _ = Polarity::target(f64::NEG_INFINITY);
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
